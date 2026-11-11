"""Group-level stratified 70/15/15 split (seed 42). Reads reports/image_table.csv from audit.py. Manifest only."""
import random, pandas as pd
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(BASE + "/reports/image_table.csv")
df = df[df.ok].copy()
rng = random.Random(42)
RATIO = {"train": .70, "val": .15, "test": .15}
assign = {}  # (group_id) -> split ; groups are global components

# Cross-class groups (label conflicts): all copies must share one split -> force train, flagged.
for g in df[df.group_n_classes > 1].group_id.unique(): assign[g] = "train"

for cid, cdf in sorted(df.groupby("class_id")):
    gs = cdf[~cdf.group_id.isin(assign)].groupby("group_id").size().to_dict()
    # classes sharing conflict groups: count their images already fixed in train
    fixed = int(cdf.group_id.isin([g for g, s in assign.items()]).sum())
    n = len(cdf); tgt = {k: v * n for k, v in RATIO.items()}
    cur = {"train": fixed, "val": 0, "test": 0}
    ids = sorted(gs); rng.shuffle(ids)
    # guarantee: test (if >=2 groups) then val (if >=3 groups) get one group, preferring the smallest
    by_small = sorted(ids, key=lambda g: (gs[g], ids.index(g)))
    for sp, need in (("test", 2), ("val", 3)):
        if len(ids) >= need:
            g = by_small.pop(0); ids.remove(g); assign[g] = sp; cur[sp] += gs[g]
    for g in ids:  # largest-deficit-first (groups in random order)
        sp = max(cur, key=lambda k: tgt[k] - cur[k]); assign[g] = sp; cur[sp] += gs[g]

df["split"] = df.group_id.map(assign)
df["label_conflict"] = (df.group_n_classes > 1).astype(int)
out = df[["path", "class_id", "class_name", "group_id", "split", "label_conflict"]].sort_values(["class_id", "path"])
out.to_csv(BASE + "/splits/manifest.csv", index=False)
t = pd.crosstab(out.class_id, out.split).reindex(columns=["train", "val", "test"], fill_value=0)
t["total"] = t.sum(axis=1); print(t.to_string()); print(t.sum())
t.to_csv(BASE + "/reports/split_counts.csv")
# leakage check
assert out.groupby("group_id").split.nunique().max() == 1
