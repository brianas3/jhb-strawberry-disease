"""Data audit: counts, formats, corrupt files, sizes, modes, exact + near duplicates, sample grids.
Read-only on dataset. Outputs: reports/image_table.csv, reports/dup_groups.csv, reports/audit_summary.json, reports/samples/*.png
"""
import os, re, json, hashlib, random
import numpy as np, pandas as pd
from PIL import Image, ImageFile
import imagehash

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.abspath(os.environ.get("DATA_DIR", os.path.join(_PROJ, "..", "JHB training dataset")))
OUT = os.path.join(os.path.abspath(os.environ.get("OUT_DIR", _PROJ)), "reports")
# paths are written to CSV/JSON as "JHB training dataset/<class>/<file>" (relative, portable)
_REL = lambda p: "JHB training dataset/" + os.path.relpath(p, ROOT)
HAM = 6
os.makedirs(OUT + "/samples", exist_ok=True)

rows, skipped = [], []
for d in sorted(os.listdir(ROOT)):
    p = os.path.join(ROOT, d)
    m = re.match(r"(\d+)\.\s*(.*?)\s*$", d)
    if not os.path.isdir(p) or not m: continue
    cid, cname = int(m.group(1)), m.group(2)
    for f in sorted(os.listdir(p)):
        fp = os.path.join(p, f)
        if f.startswith(".") or not os.path.isfile(fp):
            skipped.append(fp); continue
        r = dict(path=fp, class_id=cid, class_name=cname, ext=os.path.splitext(f)[1].lower(),
                 bytes=os.path.getsize(fp), md5=hashlib.md5(open(fp, "rb").read()).hexdigest(),
                 ok=True, w=None, h=None, mode=None, fmt=None, phash=None, dhash=None, anim=False)
        try:
            with Image.open(fp) as im:
                r["fmt"], r["mode"] = im.format, im.mode
                r["w"], r["h"] = im.size
                r["anim"] = getattr(im, "n_frames", 1) > 1
                im.load()
                rgb = im.convert("RGB")
                r["phash"] = str(imagehash.phash(rgb)); r["dhash"] = str(imagehash.dhash(rgb))
        except Exception as e:
            r["ok"] = False; r["err"] = repr(e)[:100]
        r["ext_mismatch"] = bool(r["fmt"]) and (
            {"JPEG": ".jpg", "PNG": ".png", "GIF": ".gif"}.get(r["fmt"], r["ext"]) not in (r["ext"], ".jpeg" if r["ext"] == ".jpeg" else r["ext"]))
        rows.append(r)
df = pd.DataFrame(rows)
df["min_side"] = df[["w", "h"]].min(axis=1)
df["tiny"] = df["min_side"] < 128

# ---- duplicates ----
parent = list(range(len(df)))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(a, b): parent[find(a)] = find(b)
ok_idx = df.index[df.ok].tolist()
# exact
for _, g in df[df.ok].groupby("md5"):
    ix = g.index.tolist()
    for i in ix[1:]: union(ix[0], i)
exact_root = {i: find(i) for i in range(len(df))}
df["exact_group"] = [exact_root[i] if df.md5.duplicated(keep=False)[i] else -1 for i in range(len(df))]
# near: pHash AND dHash both <= HAM
def tobits(s): return np.unpackbits(np.frombuffer(bytes.fromhex(s), dtype=np.uint8))
P = np.array([tobits(df.phash[i]) for i in ok_idx], dtype=np.int16)
D = np.array([tobits(df.dhash[i]) for i in ok_idx], dtype=np.int16)
n_near_pairs = 0
for a in range(len(ok_idx)):
    dp = (P[a+1:] != P[a]).sum(1); dd = (D[a+1:] != D[a]).sum(1)
    for b in np.where((dp <= HAM) & (dd <= HAM))[0]:
        union(ok_idx[a], ok_idx[a+1+b]); n_near_pairs += 1
df["group_root"] = [find(i) for i in range(len(df))]
sizes = df.groupby("group_root").size()
df["group_size"] = df.group_root.map(sizes)
gid = {r: k for k, r in enumerate(sorted(df.group_root.unique()))}
df["group_id"] = df.group_root.map(gid)
ncls = df.groupby("group_id").class_id.nunique()
df["group_n_classes"] = df.group_id.map(ncls)
df.drop(columns=["group_root"]).assign(path=lambda x: x.path.map(_REL)).to_csv(OUT + "/image_table.csv", index=False)

dg = df[df.group_size > 1].copy()
dg["exact"] = dg.exact_group >= 0
dg.sort_values(["group_id", "class_id", "path"])[["group_id", "group_size", "group_n_classes", "class_id", "class_name", "exact", "path"]].assign(path=lambda x: x.path.map(_REL)).to_csv(OUT + "/dup_groups.csv", index=False)

# ---- summary ----
S = {}
S["skipped_non_image"] = [os.path.relpath(s, ROOT) for s in skipped]
S["total_images"] = len(df); S["corrupt"] = df[~df.ok].path.map(_REL).tolist()
S["ext_counts"] = df.ext.value_counts().to_dict(); S["mode_counts"] = df["mode"].value_counts().to_dict()
S["animated"] = df[df.anim].path.map(_REL).tolist(); S["ext_mismatch"] = df[df.ext_mismatch].path.map(_REL).tolist()
S["exact_dup_groups"] = int(df[df.exact_group >= 0].exact_group.nunique())
S["exact_dup_extra_files"] = int((df.exact_group >= 0).sum() - S["exact_dup_groups"])
S["near_groups"] = int((sizes > 1).sum()); S["images_in_dup_groups"] = int((df.group_size > 1).sum())
S["unique_groups"] = int(df.group_id.nunique())
S["cross_class_groups"] = int((ncls > 1).sum()); S["images_in_cross_class_groups"] = int((df.group_n_classes > 1).sum())
S["near_pairs"] = n_near_pairs
cc = dg[dg.group_n_classes > 1].groupby("group_id").class_id.apply(lambda s: tuple(sorted(set(s))))
S["cross_class_pairs"] = {str(k): int(v) for k, v in cc.value_counts().items()}
json.dump(S, open(OUT + "/audit_summary.json", "w"), indent=1)

pc = df.groupby(["class_id", "class_name"]).agg(
    n=("path", "size"), corrupt=("ok", lambda s: int((~s).sum())),
    min_w=("w", "min"), med_w=("w", "median"), max_w=("w", "max"),
    min_h=("h", "min"), med_h=("h", "median"), max_h=("h", "max"),
    tiny=("tiny", "sum"), non_rgb=("mode", lambda s: int((s != "RGB").sum())),
    in_dup_grp=("group_size", lambda s: int((s > 1).sum())),
    in_cross=("group_n_classes", lambda s: int((s > 1).sum())), n_groups=("group_id", "nunique")).reset_index()
pc.to_csv(OUT + "/per_class.csv", index=False)
print(pc.to_string()); print(json.dumps(S, indent=1)[:3000])

# ---- sample grids ----
rng = random.Random(42)
for (cid, cname), g in df[df.ok].groupby(["class_id", "class_name"]):
    paths = g.path.tolist(); pick = rng.sample(paths, min(8, len(paths)))
    T = 128; canvas = Image.new("RGB", (4 * T, 2 * T), "white")
    for k, pth in enumerate(pick):
        try:
            im = Image.open(pth).convert("RGB"); im.thumbnail((T, T))
            canvas.paste(im, ((k % 4) * T, (k // 4) * T))
        except Exception: pass
    canvas.save(f"{OUT}/samples/class_{cid:02d}.png")
