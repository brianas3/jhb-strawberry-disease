"""Shared portable config. Env vars: DATA_DIR (image root), MANIFEST (csv), OUT_DIR (models/ + reports/ written here)."""
import os, shutil, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE)
DATA_DIR = os.path.abspath(os.environ.get("DATA_DIR", os.path.join(PROJ, "..", "JHB training dataset")))
MANIFEST = os.path.abspath(os.environ.get("MANIFEST", os.path.join(PROJ, "splits", "manifest.csv")))
OUT_DIR = os.path.abspath(os.environ.get("OUT_DIR", PROJ))
MODELS = os.path.join(OUT_DIR, "models"); REPORTS = os.path.join(OUT_DIR, "reports")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
MARKER = "JHB training dataset"

def ensure_dirs():
    for d in (MODELS, REPORTS): os.makedirs(d, exist_ok=True)

def check_free_space(path, need_mb=200):
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(path)) if not os.path.isdir(path) else path).free / 2**20
    if free < need_mb: raise SystemExit(f"ABORT: only {free:.0f} MB free near {path}; need >= {need_mb} MB. Free disk or set OUT_DIR.")

def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_manifest():
    """Returns df with an extra column 'fpath' resolved relative to DATA_DIR (manifest paths may be absolute from another machine)."""
    df = pd.read_csv(MANIFEST)
    def res(p):
        rel = p.split(MARKER + "/", 1)[1] if MARKER + "/" in p else p
        return os.path.join(DATA_DIR, rel)
    df["fpath"] = df.path.map(res)
    miss = (~df.fpath.map(os.path.exists)).sum()
    if miss: raise SystemExit(f"ABORT: {miss} images not found under DATA_DIR={DATA_DIR}")
    return df
