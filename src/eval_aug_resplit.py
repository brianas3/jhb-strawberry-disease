"""比較「沒有增強」和「有增強」：用 5 次重新切分（和 redteam_core.py 相同的切法）報 macro-F1 平均 ± 標準差。
放在 src/ 底下。需要先跑過 extract_features.py（feats_plain.npy）和 extract_features_aug.py（feats_aug.npy）。

重點：增強特徵「只」加進當次的 train；選 C 用 val 原圖，最後分數用 test 原圖。
用法：python3 src/eval_aug_resplit.py
"""
import os, sys, json, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

df = load_manifest(); y = df.class_id.values; grp = df.group_id.values; conf = (df.label_conflict == 1).values
X = np.load(f"{MODELS}/feats_plain.npy").astype(np.float32)
XA = np.load(f"{MODELS}/feats_aug.npy").astype(np.float32); IA = np.load(f"{MODELS}/feats_aug_idx.npy")
assert IA.max() < len(df), "feats_aug_idx 和 manifest 對不起來，請重抽"
CS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, .1, 1)

def mf1(a, b): return f1_score(a, b, labels=sorted(set(a)), average="macro", zero_division=0)

def train_set(mtr, use_aug):
    """原圖特徵 + （可選）同一批 train 圖的增強特徵。標籤沿用原圖。"""
    Xt, yt = X[mtr], y[mtr]
    if use_aug:
        keep = mtr[IA]                       # 只留「原圖在這次 train 裡」的增強 view
        Xt = np.vstack([Xt, XA[keep]]); yt = np.concatenate([yt, y[IA[keep]]])
    return Xt, yt

CW_ENV = os.environ.get("CLASS_WEIGHT", "balanced")
CW = None if str(CW_ENV).lower() in ("none", "false", "0") else "balanced"
print(f"Using class_weight={CW} (set CLASS_WEIGHT=none to toggle)", flush=True)

def fitsel(mtr, mva, use_aug):
    Xt, yt = train_set(mtr, use_aug); best = None
    for C in CS:
        sc = StandardScaler().fit(Xt)
        clf = LogisticRegression(C=C, class_weight=CW, max_iter=3000).fit(sc.transform(Xt), yt)
        s = mf1(y[mva], clf.predict(sc.transform(X[mva])))   # val 只用原圖
        if best is None or s > best[0]: best = (s, C, sc, clf)
    return best

res = {"baseline": [], "aug": []}
idxs = np.where(~conf)[0]
for seed in range(5):
    folds = [f for _, f in StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed).split(idxs, y[idxs], grp[idxs])]
    mte = np.zeros(len(df), bool); mva = mte.copy(); mte[idxs[folds[0]]] = True; mva[idxs[folds[1]]] = True
    mtr = ~(mte | mva); mtr &= ~np.isin(grp, list(set(grp[mte | mva]))); mtr &= ~conf   # 同 redteam_core 的 "drop"
    for name, ua in (("baseline", False), ("aug", True)):
        s, C, sc, clf = fitsel(mtr, mva, ua); p = clf.predict(sc.transform(X[mte]))   # test 只用原圖
        res[name].append(dict(seed=seed, C=C, val_mf1=s, acc=accuracy_score(y[mte], p), mf1=mf1(y[mte], p)))
        print(seed, name, {k: round(v, 4) if isinstance(v, float) else v for k, v in res[name][-1].items()}, flush=True)

summary = {}
for name, r in res.items():
    a = np.array([x["mf1"] for x in r]); b = np.array([x["acc"] for x in r])
    summary[name] = dict(mf1_mean=a.mean(), mf1_sd=a.std(), acc_mean=b.mean(), per_seed_mf1=a.round(3).tolist())
d = np.array([x["mf1"] for x in res["aug"]]) - np.array([x["mf1"] for x in res["baseline"]])
summary["aug_minus_baseline_per_seed"] = d.round(3).tolist()
print(json.dumps(summary, indent=1))
print(f"\n增強 - 無增強：平均 {d.mean():+.3f}，5 次中有 {(d > 0).sum()} 次變好")
json.dump(dict(summary=summary, runs=res), open(f"{REPORTS}/aug_resplit.json", "w"), indent=1, default=float)
