"""Two-stage partial fine-tune MobileNetV3-Large (ImageNet) with advanced improvements:
1. Loss function upgrade: Class-Balanced Focal Loss / Focal Loss / CrossEntropy
2. Smoothed class-balanced resampling: 1/(N^pow), default pow=0.5 (1/sqrt(N))
3. Knowledge & Feature Distillation: Distill DINOv2 teacher representations and probe logits
4. Enhanced data augmentation against source/resolution shortcuts

Usage: python3 src/finetune.py [BUDGET_SECONDS]
Env (optional):
  FT_WORKERS     DataLoader workers (default: min(8, cpu_count); 0 = main process)
  FT_BS          batch size (default 32)
  FT_SAMPLES     samples per epoch (default 1000)
  FT_EP2         stage-2 epochs for cosine schedule (default 12)
  FT_AMP         1/0 mixed precision on CUDA (default 1)
  FT_LOSS        loss type: cb_focal (default), focal, ce
  FT_SAMPLE_POW  sampling power: 0.5 (1/sqrt(N), default), 1.0 (strict 1/N), 0.0 (uniform)
  FT_GAMMA       focal loss gamma (default 1.5)
  FT_BETA        class-balanced effective number beta (default 0.999)
  FT_KD_ALPHA    logits KD weight (default 0.3 if teacher available, else 0.0)
  FT_FEAT_ALPHA  feature alignment KD weight (default 0.2 if teacher available, else 0.0)
  FT_KD_TEMP     KD temperature (default 2.0)
  FT_UNFREEZE    stage-2 unfreeze starting block (default 14)
"""
import os, sys, time, json, warnings
import numpy as np, torch, torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms as T
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
AUG = T.Compose([
    T.RandomResizedCrop(224, scale=(0.5, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(0.2),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    T.RandomApply([T.GaussianBlur(3)], 0.2),
    T.ToTensor(),
    NORM,
    T.RandomErasing(p=0.2, scale=(0.02, 0.2))
])
EV = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), NORM])


def load(p):  # pre-resize shorter side to 256 once (decode cost dominates otherwise)
    im = Image.open(p).convert("RGB"); w, h = im.size; s = 256 / min(w, h)
    return im.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BILINEAR)


class ImgDS(Dataset):
    """Images live in RAM (list of PIL); returns (image, label, original_manifest_index)."""
    def __init__(self, imgs, y, idx, tf):
        self.imgs, self.y, self.idx, self.tf = imgs, y, idx, tf
    def __len__(self): return len(self.idx)
    def __getitem__(self, k):
        j = self.idx[k]
        return self.tf(self.imgs[j]), int(self.y[j]), int(j)


def seed_worker(_):
    s = torch.initial_seed() % 2**32; np.random.seed(s); __import__("random").seed(s)


def mf1(yt, yp): return f1_score(yt, yp, labels=sorted(set(yt)), average="macro", zero_division=0)


class CBFocalLoss(nn.Module):
    """Class-Balanced Focal Loss / Standard Focal Loss / CrossEntropy with Label Smoothing."""
    def __init__(self, class_counts=None, beta=0.999, gamma=1.5, label_smoothing=0.05, loss_type="cb_focal"):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type
        if loss_type == "cb_focal" and beta > 0 and class_counts is not None:
            eff = 1.0 - np.power(beta, np.maximum(class_counts, 1))
            eff = np.where(eff == 0, 1e-6, eff)
            weights = (1.0 - beta) / eff
            weights = weights / np.sum(weights) * len(class_counts)
            self.alpha = torch.tensor(weights, dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction='none', label_smoothing=self.label_smoothing)
        if self.loss_type == "ce":
            return ce.mean()
        p = torch.softmax(logits, dim=-1)
        pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - pt).clamp(min=0.0) ** self.gamma
        loss = focal_weight * ce
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = alpha_t * loss
        return loss.mean()


def load_teacher(df, cls, ci):
    """Attempt to load pre-extracted DINOv2 features and trained linear probe for distillation."""
    feats_path = f"{MODELS}/feats_plain.npy"
    probe_path = f"{MODELS}/probe_best.joblib"
    if not (os.path.exists(feats_path) and os.path.exists(probe_path)):
        return None, None
    try:
        import joblib
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probe_dict = joblib.load(probe_path)
        feats = np.load(feats_path)
        if feats.shape[0] != len(df):
            print(f"Teacher feats shape {feats.shape[0]} != manifest {len(df)}, skipping distillation.", flush=True)
            return None, None
        sc = probe_dict.get("scaler")
        clf = probe_dict.get("clf")
        teacher_classes = [int(c) for c in probe_dict.get("classes", [])]
        # Align teacher class logits to student's class indices
        raw_logits = clf.decision_function(sc.transform(feats))
        if teacher_classes == cls:
            aligned_logits = raw_logits
        else:
            # Map columns
            aligned_logits = np.zeros((len(df), len(cls)), dtype=np.float32)
            for t_idx, c_id in enumerate(teacher_classes):
                if c_id in ci:
                    aligned_logits[:, ci[c_id]] = raw_logits[:, t_idx]
        print(f"Teacher distillation active: DINOv2 ({feats.shape}) + Probe ({len(teacher_classes)} classes)", flush=True)
        return torch.tensor(feats, dtype=torch.float32), torch.tensor(aligned_logits, dtype=torch.float32)
    except Exception as e:
        print(f"Teacher loading skipped: {e}", flush=True)
        return None, None


def main():
    torch.set_num_threads(min(16, os.cpu_count() or 4)); torch.manual_seed(0); np.random.seed(0)
    ensure_dirs(); check_free_space(MODELS); dev = device()
    BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 22 * 60
    default_workers = 0 if sys.platform == "win32" else min(4, os.cpu_count() or 1)
    WORKERS = int(os.environ.get("FT_WORKERS", default_workers))
    BS = int(os.environ.get("FT_BS", 32)); NSAMP = int(os.environ.get("FT_SAMPLES", 1000)); EP2 = int(os.environ.get("FT_EP2", 12))
    
    # Hyperparameters
    LOSS_TYPE = os.environ.get("FT_LOSS", "cb_focal").lower()  # cb_focal, focal, ce
    SAMPLE_POW = float(os.environ.get("FT_SAMPLE_POW", "0.5")) # 0.5 = 1/sqrt(N)
    GAMMA = float(os.environ.get("FT_GAMMA", "1.5"))
    BETA = float(os.environ.get("FT_BETA", "0.999"))
    KD_ALPHA = float(os.environ.get("FT_KD_ALPHA", "0.3"))
    FEAT_ALPHA = float(os.environ.get("FT_FEAT_ALPHA", "0.2"))
    KD_TEMP = float(os.environ.get("FT_KD_TEMP", "2.0"))
    UNFREEZE_FROM = int(os.environ.get("FT_UNFREEZE", "14"))

    cuda = dev.type == "cuda"; AMP = cuda and os.environ.get("FT_AMP", "1") == "1"
    if cuda: torch.backends.cudnn.benchmark = True
    print(f"device={dev} ({torch.cuda.get_device_name(0) if cuda else 'cpu'}) amp={AMP} workers={WORKERS} bs={BS} "
          f"samples/ep={NSAMP} ep2={EP2} loss={LOSS_TYPE} pow={SAMPLE_POW} unfreeze={UNFREEZE_FROM} budget={BUDGET:.0f}s", flush=True)
    t0 = time.time()

    df = load_manifest(); cls = sorted(df.class_id.unique()); ci = {c: i for i, c in enumerate(cls)}
    y = df.class_id.map(ci).values
    with ThreadPoolExecutor(max(1, min(16, os.cpu_count() or 4))) as ex:  # PIL decode releases the GIL
        imgs = list(ex.map(load, df.fpath))
    print("cached imgs", round(time.time() - t0, 1), flush=True)

    tr = np.where(((df.split == "train") & (df.label_conflict == 0)).values)[0]
    va = np.where((df.split == "val").values)[0]; te = np.where((df.split == "test").values)[0]

    # Smoothed Class-Balanced Sampler (1 / N^pow)
    cnt = np.bincount(y[tr], minlength=len(cls))
    w = 1.0 / np.power(cnt[y[tr]], SAMPLE_POW)

    dl_kw = dict(num_workers=WORKERS, pin_memory=cuda, persistent_workers=WORKERS > 0, worker_init_fn=seed_worker)
    g = torch.Generator(); g.manual_seed(0)
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), NSAMP, replacement=True, generator=g)
    train_dl = DataLoader(ImgDS(imgs, y, tr, AUG), batch_size=BS, sampler=sampler, drop_last=False, generator=g, **dl_kw)
    ev_dl = {k: DataLoader(ImgDS(imgs, y, v, EV), batch_size=128, shuffle=False, **dl_kw) for k, v in (("val", va), ("test", te))}

    # Teacher setup
    teacher_feats, teacher_logits = load_teacher(df, cls, ci)
    if teacher_logits is None:
        KD_ALPHA = 0.0
    if teacher_feats is None:
        FEAT_ALPHA = 0.0

    m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(cls))
    m = m.to(dev, memory_format=torch.channels_last if cuda else torch.contiguous_format)

    # Feature Distillation Projection Layer (MobileNet 960d -> DINOv2 768d)
    proj = nn.Sequential(nn.Linear(960, 768), nn.LayerNorm(768)).to(dev)

    crit = CBFocalLoss(class_counts=cnt, beta=BETA, gamma=GAMMA, label_smoothing=0.05, loss_type=LOSS_TYPE)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)

    def to_dev(x): return x.to(dev, non_blocking=True, memory_format=torch.channels_last if cuda else torch.contiguous_format)

    def predict(split, flip=False):
        m.eval(); out = []
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=AMP):
            for x, _, _ in ev_dl[split]:
                x = to_dev(x); x = x.flip(-1) if flip else x; out.append(m(x).float().cpu())
        return torch.cat(out).numpy()

    def run_epoch(opt, freeze_until=17):
        m.train()
        for i in range(freeze_until):
            m.features[i].eval()
        tl = 0; n = 0
        for x, yy, jj in train_dl:
            x, yy = to_dev(x), yy.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=AMP):
                # Forward features and logits
                f = m.features(x)
                f = m.avgpool(f)
                feat = torch.flatten(f, 1)  # [B, 960]
                logits = m.classifier(feat) # [B, num_classes]

                loss_task = crit(logits, yy)
                loss = loss_task

                # Knowledge distillation (logits)
                if teacher_logits is not None and KD_ALPHA > 0:
                    t_logits = teacher_logits[jj].to(dev)
                    t_soft = torch.softmax(t_logits / KD_TEMP, dim=-1)
                    s_log_soft = torch.log_softmax(logits / KD_TEMP, dim=-1)
                    loss_kd = nn.functional.kl_div(s_log_soft, t_soft, reduction="batchmean") * (KD_TEMP ** 2)
                    loss = (1.0 - KD_ALPHA) * loss + KD_ALPHA * loss_kd

                # Feature distillation (DINOv2 representation alignment)
                if teacher_feats is not None and FEAT_ALPHA > 0:
                    t_feat = teacher_feats[jj].to(dev).float()
                    s_proj = proj(feat)
                    loss_feat = (1.0 - nn.functional.cosine_similarity(s_proj, t_feat, dim=-1)).mean()
                    loss = (1.0 - FEAT_ALPHA) * loss + FEAT_ALPHA * loss_feat

            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += loss.item() * len(yy); n += len(yy)
        return tl / n

    hist = []; global_best = [-1]
    def log(stage, ep, tl):
        lv = predict("val"); s = mf1(y[va], lv.argmax(1))
        hist.append(dict(stage=stage, ep=ep, loss=tl, val_macroF1=s, val_acc=accuracy_score(y[va], lv.argmax(1)), t=time.time() - t0))
        print(hist[-1], flush=True); return s
    def keep(s):
        if s > global_best[0]: global_best[0] = s; torch.save(m.state_dict(), f"{MODELS}/mnv3_best.pt")

    # Stage 1: head only (freeze all feature blocks)
    for p in m.features.parameters(): p.requires_grad = False
    stage1_params = list(m.classifier.parameters())
    if FEAT_ALPHA > 0 and teacher_feats is not None:
        stage1_params += list(proj.parameters())
    opt = torch.optim.AdamW(stage1_params, lr=1e-3, weight_decay=1e-2)
    for ep in range(2):
        tl = run_epoch(opt, freeze_until=17); keep(log(1, ep, tl))

    # Stage 2: partial fine-tune with differential LR
    for p in m.features[UNFREEZE_FROM:].parameters(): p.requires_grad = True
    stage2_params = [
        {"params": m.features[UNFREEZE_FROM:].parameters(), "lr": 2.5e-5},
        {"params": m.classifier.parameters(), "lr": 5e-4}
    ]
    if FEAT_ALPHA > 0 and teacher_feats is not None:
        stage2_params.append({"params": proj.parameters(), "lr": 2.5e-4})
    opt = torch.optim.AdamW(stage2_params, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP2); ep2_done = 0
    for ep in range(EP2):
        if time.time() - t0 > BUDGET: print("time budget hit, stopping early", flush=True); break
        tl = run_epoch(opt, freeze_until=UNFREEZE_FROM); sched.step(); keep(log(2, ep, tl)); ep2_done += 1

    m.load_state_dict(torch.load(f"{MODELS}/mnv3_best.pt", map_location=dev))
    from scipy.optimize import minimize_scalar
    from scipy.special import softmax, log_softmax
    lv = predict("val"); iv = y[va]
    T_ = minimize_scalar(lambda t: -log_softmax(lv / t, 1)[np.arange(len(iv)), iv].mean(), bounds=(.05, 20), method="bounded").x
    lt = predict("test"); ltf = predict("test", True); yt = y[te]
    P = softmax(lt / T_, 1); Pt = (P + softmax(ltf / T_, 1)) / 2
    np.save(f"{MODELS}/mnv3_val_logits.npy", lv); np.save(f"{MODELS}/mnv3_test_logits.npy", lt)
    res = dict(device=str(dev), amp=AMP, workers=WORKERS, batch_size=BS, samples_per_epoch=NSAMP,
               ep2_planned=EP2, ep2_done=ep2_done, loss_type=LOSS_TYPE, sample_pow=SAMPLE_POW,
               kd_alpha=KD_ALPHA, feat_alpha=FEAT_ALPHA, unfreeze_from=UNFREEZE_FROM,
               T=float(T_), test_acc=accuracy_score(yt, P.argmax(1)), test_macroF1=mf1(yt, P.argmax(1)), tta_macroF1=mf1(yt, Pt.argmax(1)),
               val_macroF1_best=global_best[0], hist=hist, runtime_s=time.time() - t0)
    # bootstrap CI (stratified by class)
    rng = np.random.RandomState(0); by = [np.where(yt == c)[0] for c in np.unique(yt)]; pr = P.argmax(1); bs = []
    for _ in range(2000):
        ii = np.concatenate([rng.choice(v, len(v)) for v in by]); bs.append(mf1(yt[ii], pr[ii]))
    res["macroF1_CI"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    p, r, f, s = precision_recall_fscore_support(yt, pr, labels=range(len(cls)), zero_division=0)
    res["per_class"] = {int(cls[i]): dict(precision=float(p[i]), recall=float(r[i]), support=int(s[i])) for i in range(len(cls))}
    json.dump(res, open(f"{REPORTS}/finetune_results.json", "w"), indent=1, default=float)
    print("DONE", {k: v for k, v in res.items() if k not in ("hist", "per_class")})


if __name__ == "__main__":
    main()
