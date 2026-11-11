"""Cache frozen DINOv2 ViT-S/14 (HF facebook/dinov2-small) embeddings for all manifest images.
Feature = concat(CLS, mean patch tokens) -> 768-d. Outputs models/feats_{plain,flip}.npy (row order = manifest)."""
import sys, numpy as np, pandas as pd, torch
from PIL import Image
from torchvision import transforms as T
from transformers import AutoModel
torch.set_num_threads(min(16, __import__("os").cpu_count() or 4))
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from common import *
ensure_dirs(); check_free_space(MODELS)
dev = device(); df = load_manifest()
m = AutoModel.from_pretrained("facebook/dinov2-small").eval().to(dev)
tf = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
flip = "--flip" in sys.argv
out = np.zeros((len(df), 768), np.float32)
B = 32
for i in range(0, len(df), B):
    x = torch.stack([tf(Image.open(p).convert("RGB")) for p in df.fpath[i:i+B]])
    if flip: x = x.flip(-1)
    with torch.no_grad():
        h = m(pixel_values=x.to(dev)).last_hidden_state.float().cpu()
    out[i:i+B] = torch.cat([h[:,0], h[:,1:].mean(1)], 1).numpy()
    if (i//B) % 10 == 0: print(i, flush=True)
np.save(f"{MODELS}/feats_{'flip' if flip else 'plain'}.npy", out.astype(np.float16))  # float16 cache
