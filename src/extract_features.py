"""Cache frozen DINOv2 ViT embeddings for all manifest images.
Supports dinov2-small (768-d) and dinov2-base (1536-d) via BACKBONE env var.
Supports TTA variants: plain, --flip (horizontal), --vflip (vertical), --rot90 (rotate 90)."""
import os, sys, numpy as np, pandas as pd, torch
from PIL import Image
from torchvision import transforms as T
from transformers import AutoModel

torch.set_num_threads(min(16, os.cpu_count() or 4))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

ensure_dirs(); check_free_space(MODELS)
dev = device(); df = load_manifest()

BACKBONE = os.environ.get("BACKBONE", "facebook/dinov2-small")
m = AutoModel.from_pretrained(BACKBONE).eval().to(dev)
feat_dim = m.config.hidden_size * 2

tf = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

mode = "plain"
if "--flip" in sys.argv:
    mode = "flip"
elif "--vflip" in sys.argv:
    mode = "vflip"
elif "--rot90" in sys.argv:
    mode = "rot90"

out = np.zeros((len(df), feat_dim), np.float32)
B = 32
print(f"Extracting DINOv2 ({BACKBONE}, {feat_dim}-d) | mode={mode} | total images={len(df)}", flush=True)

for i in range(0, len(df), B):
    x = torch.stack([tf(Image.open(p).convert("RGB")) for p in df.fpath[i:i+B]])
    if mode == "flip":
        x = x.flip(-1)
    elif mode == "vflip":
        x = x.flip(-2)
    elif mode == "rot90":
        x = torch.rot90(x, 1, [-2, -1])
    
    with torch.no_grad():
        h = m(pixel_values=x.to(dev)).last_hidden_state.float().cpu()
    out[i:i+B] = torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).numpy()
    if (i // B) % 10 == 0:
        print(f"{i}/{len(df)}", flush=True)

save_path = f"{MODELS}/feats_{mode}.npy"
np.save(save_path, out.astype(np.float16))
print(f"Saved: {save_path} shape={out.shape}")
