"""把一張圖變成很多張：為每張圖產生 K 個「增強版本」，再抽 DINOv2 特徵（放在 src/ 底下，和 extract_features.py 並排）。

輸出（列順序和 manifest 無關，用 idx 對回去）：
  models/feats_aug.npy      float16, (總 view 數, 768)，特徵格式和 feats_plain.npy 相同（CLS + mean patch）
  models/feats_aug_idx.npy  int32,  (總 view 數,)       每個 view 來自 manifest 的第幾列

為什麼「全部」圖片都做增強，而不是只做 train？
  因為要用 5 次重新切分來比較（eval_aug_resplit.py），每次切分 train 都不一樣。
  所以先把每張圖的增強特徵都準備好，訓練時「只拿當次 train 的增強特徵」，
  val/test 一律只用原圖特徵（feats_plain.npy）。增強版的 val/test 圖永遠不會被拿來評估。

每類做幾個版本：K = clamp(ceil(AUG_TARGET / 該類張數), AUG_KMIN, AUG_KMAX)
  預設 AUG_TARGET=600, AUG_KMIN=3, AUG_KMAX=20
  例：Macrophomina 15 張 → 20 個版本；Leaf Scorch 1480 張 → 3 個版本。張數少的病做比較多種。

用法：
  python3 src/extract_features_aug.py                 # 抽增強特徵
  python3 src/extract_features_aug.py --preview 6     # 只輸出 6 張圖的增強預覽到 reports/aug_preview/，不抽特徵
  （預覽圖含資料集影像，不要 commit 到 repo）
Env：AUG_TARGET, AUG_KMIN, AUG_KMAX, AUG_SEED（預設 0），以及 common.py 的 DATA_DIR / MANIFEST / OUT_DIR
"""
import io, os, sys, math, random
import numpy as np, torch
from PIL import Image
from torchvision import transforms as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

TARGET = int(os.environ.get("AUG_TARGET", 600))
KMIN = int(os.environ.get("AUG_KMIN", 3))
KMAX = int(os.environ.get("AUG_KMAX", 20))
SEED = int(os.environ.get("AUG_SEED", 0))


# ---------- 自訂的兩種「打破捷徑」增強 ----------
class RandomDownUp:
    """隨機縮小再放大回原尺寸：模擬低解析度照片，讓模型不能靠「圖片清不清楚」猜類別。"""
    def __init__(self, min_side=(64, 200), p=0.5):
        self.min_side, self.p = min_side, p
    def __call__(self, im):
        if random.random() > self.p: return im
        w, h = im.size; s = random.randint(*self.min_side) / min(w, h)
        if s >= 1: return im
        small = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


class RandomJPEG:
    """隨機用較低品質重新存成 JPEG：模擬手機壓縮、網路轉存，讓模型不能靠「壓縮痕跡」猜類別。"""
    def __init__(self, quality=(30, 95), p=0.5):
        self.quality, self.p = quality, p
    def __call__(self, im):
        if random.random() > self.p: return im
        buf = io.BytesIO(); im.save(buf, format="JPEG", quality=random.randint(*self.quality))
        buf.seek(0); return Image.open(buf).convert("RGB")


NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# 先把所有圖統一縮到短邊 256：原本 419x419、256x256、4000 多 px 的圖都變成差不多大小，減少「尺寸」這條捷徑
PRE = T.Resize(256, interpolation=T.InterpolationMode.BICUBIC)
AUG = T.Compose([
    PRE,
    RandomDownUp(p=0.5),                                     # 畫質：縮小再放大
    RandomJPEG(p=0.5),                                       # 畫質：變糊的 JPEG
    T.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.8, 1.25),
                        interpolation=T.InterpolationMode.BICUBIC),  # 幾何：隨機裁切（scale 不低於 0.5，避免裁掉病斑）
    T.RandomHorizontalFlip(),                                # 幾何：左右翻
    T.RandomVerticalFlip(0.2),                               # 幾何：上下翻（葉片照沒有固定方向）
    T.RandomRotation(20, interpolation=T.InterpolationMode.BILINEAR),  # 幾何：轉一點
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),  # 顏色：只做輕微調整，病斑顏色是重要訊號
    T.RandomGrayscale(p=0.1),                                # 顏色：偶爾轉黑白
])
TO_TENSOR = T.Compose([T.ToTensor(), NORM])


def k_per_class(df):
    n = df.class_id.value_counts()
    return {c: int(min(KMAX, max(KMIN, math.ceil(TARGET / v)))) for c, v in n.items()}


def preview(df, n_img):
    out = os.path.join(REPORTS, "aug_preview"); os.makedirs(out, exist_ok=True)
    rows = df.groupby("class_id").head(1).head(n_img)
    for _, r in rows.iterrows():
        im = Image.open(r.fpath).convert("RGB")
        views = [T.CenterCrop(224)(PRE(im))] + [AUG(im) for _ in range(7)]
        grid = Image.new("RGB", (224 * 4, 224 * 2))
        for i, v in enumerate(views): grid.paste(v, ((i % 4) * 224, (i // 4) * 224))
        grid.save(os.path.join(out, f"class_{r.class_id:02d}.jpg"), quality=90)
    print("preview written to", out, "(左上是原圖，其餘 7 張是增強版本)")


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(min(16, os.cpu_count() or 4))
    ensure_dirs(); df = load_manifest()
    if "--preview" in sys.argv:
        preview(df, int(sys.argv[sys.argv.index("--preview") + 1])); return
    check_free_space(MODELS)

    K = k_per_class(df)
    total = sum(K[c] for c in df.class_id)
    print("views per image by class:", dict(sorted(K.items())), "| total views:", total, flush=True)

    from transformers import AutoModel
    dev = device()
    m = AutoModel.from_pretrained(os.environ.get("BACKBONE", "facebook/dinov2-small")).eval().to(dev)

    feats, idx, batch, bidx = [], [], [], []
    def flush():
        x = torch.stack(batch).to(dev)
        with torch.no_grad():
            h = m(pixel_values=x).last_hidden_state.float().cpu()
        feats.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).numpy().astype(np.float16))
        idx.extend(bidx); batch.clear(); bidx.clear()

    for i, (p, c) in enumerate(zip(df.fpath, df.class_id)):
        im = Image.open(p).convert("RGB")
        for _ in range(K[c]):
            batch.append(TO_TENSOR(AUG(im))); bidx.append(i)
            if len(batch) == 32: flush()
        if i % 200 == 0: print(f"{i}/{len(df)} images", flush=True)
    if batch: flush()

    F = np.concatenate(feats); I = np.asarray(idx, np.int32)
    np.save(f"{MODELS}/feats_aug.npy", F); np.save(f"{MODELS}/feats_aug_idx.npy", I)
    print("saved", F.shape, "->", f"{MODELS}/feats_aug.npy")


if __name__ == "__main__":
    main()
