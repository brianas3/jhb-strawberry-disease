"""Two-stage fine-tune MobileNetV3-Large (ImageNet). GPU-ready (CUDA + AMP + DataLoader workers), falls back to CPU.
Train split minus label_conflict rows; select epoch on val; test once.

Usage: python3 src/finetune.py [BUDGET_SECONDS]
Env (optional):
  FT_WORKERS  DataLoader workers (default: min(8, cpu_count); 0 = main process)
  FT_BS       batch size (default 32, same as CPU baseline)
  FT_SAMPLES  class-balanced samples per epoch (default 1000)
  FT_EP2      stage-2 epochs for the cosine schedule (default 12)
  FT_AMP      1/0 mixed precision on CUDA (default 1)
"""
import os, sys, time, json
import numpy as np, torch, torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms as T
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
AUG = T.Compose([T.RandomResizedCrop(224, scale=(0.4, 1)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(0.2), T.RandomRotation(20),
                 T.ColorJitter(0.3, 0.3, 0.2, 0.03), T.RandomApply([T.GaussianBlur(3)], 0.2), T.ToTensor(), NORM, T.RandomErasing(0.25)])
EV = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), NORM])


def load(p):  # pre-resize shorter side to 256 once (decode cost dominates otherwise)
    im = Image.open(p).convert("RGB"); w, h = im.size; s = 256 / min(w, h)
    return im.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BILINEAR)


class ImgDS(Dataset):
    """Images live in RAM (list of PIL); nothing written to disk."""
    def __init__(self, imgs, y, idx, tf):
        self.imgs, self.y, self.idx, self.tf = imgs, y, idx, tf
    def __len__(self): return len(self.idx)
    def __getitem__(self, k):
        j = self.idx[k]; return self.tf(self.imgs[j]), int(self.y[j])


def seed_worker(_):
    s = torch.initial_seed() % 2**32; np.random.seed(s); __import__("random").seed(s)


def mf1(yt, yp): return f1_score(yt, yp, labels=sorted(set(yt)), average="macro", zero_division=0)


def main():
    torch.set_num_threads(min(16, os.cpu_count() or 4)); torch.manual_seed(0); np.random.seed(0)
    ensure_dirs(); check_free_space(MODELS); dev = device()
    BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 22 * 60
    WORKERS = int(os.environ.get("FT_WORKERS", min(8, os.cpu_count() or 1)))
    BS = int(os.environ.get("FT_BS", 32)); NSAMP = int(os.environ.get("FT_SAMPLES", 1000)); EP2 = int(os.environ.get("FT_EP2", 12))
    cuda = dev.type == "cuda"; AMP = cuda and os.environ.get("FT_AMP", "1") == "1"
    if cuda: torch.backends.cudnn.benchmark = True
    print(f"device={dev} ({torch.cuda.get_device_name(0) if cuda else 'cpu'}) amp={AMP} workers={WORKERS} bs={BS} "
          f"samples/ep={NSAMP} ep2={EP2} budget={BUDGET:.0f}s", flush=True)
    t0 = time.time()

    df = load_manifest(); cls = sorted(df.class_id.unique()); ci = {c: i for i, c in enumerate(cls)}
    y = df.class_id.map(ci).values
    with ThreadPoolExecutor(max(1, min(16, os.cpu_count() or 4))) as ex:  # PIL decode releases the GIL
        imgs = list(ex.map(load, df.fpath))
    print("cached imgs", round(time.time() - t0, 1), flush=True)

    tr = np.where(((df.split == "train") & (df.label_conflict == 0)).values)[0]
    va = np.where((df.split == "val").values)[0]; te = np.where((df.split == "test").values)[0]
    cnt = np.bincount(y[tr], minlength=len(cls)); w = 1.0 / cnt[y[tr]]

    dl_kw = dict(num_workers=WORKERS, pin_memory=cuda, persistent_workers=WORKERS > 0, worker_init_fn=seed_worker)
    g = torch.Generator(); g.manual_seed(0)
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), NSAMP, replacement=True, generator=g)
    train_dl = DataLoader(ImgDS(imgs, y, tr, AUG), batch_size=BS, sampler=sampler, drop_last=False, generator=g, **dl_kw)
    ev_dl = {k: DataLoader(ImgDS(imgs, y, v, EV), batch_size=128, shuffle=False, **dl_kw) for k, v in (("val", va), ("test", te))}

    m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(cls))
    m = m.to(dev, memory_format=torch.channels_last if cuda else torch.contiguous_format)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)

    def to_dev(x): return x.to(dev, non_blocking=True, memory_format=torch.channels_last if cuda else torch.contiguous_format)

    def predict(split, flip=False):
        m.eval(); out = []
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=AMP):
            for x, _ in ev_dl[split]:
                x = to_dev(x); x = x.flip(-1) if flip else x; out.append(m(x).float().cpu())
        return torch.cat(out).numpy()

    def run_epoch(opt):
        m.train(); tl = 0; n = 0
        for x, yy in train_dl:
            x, yy = to_dev(x), yy.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=AMP):
                l = crit(m(x), yy)
            scaler.scale(l).backward(); scaler.step(opt); scaler.update()
            tl += l.item() * len(yy); n += len(yy)
        return tl / n

    hist = []; global_best = [-1]
    def log(stage, ep, tl):
        lv = predict("val"); s = mf1(y[va], lv.argmax(1))
        hist.append(dict(stage=stage, ep=ep, loss=tl, val_macroF1=s, val_acc=accuracy_score(y[va], lv.argmax(1)), t=time.time() - t0))
        print(hist[-1], flush=True); return s
    def keep(s):
        if s > global_best[0]: global_best[0] = s; torch.save(m.state_dict(), f"{MODELS}/mnv3_best.pt")

    # stage 1: head only
    for p in m.features.parameters(): p.requires_grad = False
    opt = torch.optim.AdamW(m.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(2):
        tl = run_epoch(opt); keep(log(1, ep, tl))
    # stage 2: unfreeze features[9:] at lower LR, cosine over EP2 epochs
    for p in m.features[9:].parameters(): p.requires_grad = True
    opt = torch.optim.AdamW([{"params": m.features[9:].parameters(), "lr": 1e-4}, {"params": m.classifier.parameters(), "lr": 3e-4}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP2); ep2_done = 0
    for ep in range(EP2):
        if time.time() - t0 > BUDGET: print("time budget hit, stopping early", flush=True); break
        tl = run_epoch(opt); sched.step(); keep(log(2, ep, tl)); ep2_done += 1

    m.load_state_dict(torch.load(f"{MODELS}/mnv3_best.pt", map_location=dev))
    from scipy.optimize import minimize_scalar
    from scipy.special import softmax, log_softmax
    lv = predict("val"); iv = y[va]
    T_ = minimize_scalar(lambda t: -log_softmax(lv / t, 1)[np.arange(len(iv)), iv].mean(), bounds=(.05, 20), method="bounded").x
    lt = predict("test"); ltf = predict("test", True); yt = y[te]
    P = softmax(lt / T_, 1); Pt = (P + softmax(ltf / T_, 1)) / 2
    np.save(f"{MODELS}/mnv3_val_logits.npy", lv); np.save(f"{MODELS}/mnv3_test_logits.npy", lt)
    res = dict(device=str(dev), amp=AMP, workers=WORKERS, batch_size=BS, samples_per_epoch=NSAMP, ep2_planned=EP2, ep2_done=ep2_done,
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


if __name__ == "__main__":  # required: DataLoader workers re-import this module under spawn (macOS/Windows)
    main()
