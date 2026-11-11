"""Logistic-regression probe on cached DINOv2 features. Train split only; tune C on val; temperature on val; test report."""
import json, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import minimize_scalar
from scipy.special import softmax, log_softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
ensure_dirs(); check_free_space(MODELS); df = load_manifest()
X = np.load(f"{MODELS}/feats_plain.npy").astype(np.float32); Xf = np.load(f"{MODELS}/feats_flip.npy").astype(np.float32)
y = df.class_id.values; cls = sorted(df.class_id.unique())
names = df.drop_duplicates("class_id").set_index("class_id").class_name.to_dict()
tr, va, te = [(df.split == s).values for s in ("train", "val", "test")]
conf = (df.label_conflict == 1).values
assert not conf[va].any() and not conf[te].any()
rng = np.random.RandomState(0)

def mf1(yt, yp): return f1_score(yt, yp, labels=sorted(set(yt)), average="macro", zero_division=0)
def fit(mask, C):
    sc = StandardScaler().fit(X[mask])
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(sc.transform(X[mask]), y[mask])
    return sc, clf
def logits(sc, clf, A): return clf.decision_function(sc.transform(A))
res = {}; models = {}
for variant, mask in [("with_conflicts", tr), ("drop_conflicts", tr & ~conf)]:
    best = None; grid = {}
    for C in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 1]:
        sc, clf = fit(mask, C); pv = clf.classes_[logits(sc, clf, X[va]).argmax(1)]
        s = mf1(y[va], pv); grid[C] = s
        if best is None or s > best[0]: best = (s, C, sc, clf)
    s, C, sc, clf = best; models[variant] = (sc, clf, C)
    res[variant] = dict(C=C, val_macroF1=s, grid=grid)
    print(variant, "best C", C, "val mF1 %.4f" % s, grid)

# ---- choose variant for main reporting: pre-declared = drop_conflicts? choose by val, report both
main = max(res, key=lambda k: res[k]["val_macroF1"]); print("main variant (by val):", main)
out = {}
def temp_fit(L, yy):
    idx = np.searchsorted(cls, yy)
    f = lambda t: -log_softmax(L / t, 1)[np.arange(len(yy)), idx].mean()
    return minimize_scalar(f, bounds=(0.05, 20), method="bounded").x
def ece(P, yy, nb=10):
    conf_, pred = P.max(1), np.array(cls)[P.argmax(1)]; acc = (pred == yy); e = 0
    for lo in np.linspace(0, 1, nb + 1)[:-1]:
        m = (conf_ > lo) & (conf_ <= lo + 1 / nb)
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf_[m].mean())
    return e

def evaluate(variant):
    sc, clf, C = models[variant]
    assert list(clf.classes_) == cls
    Lv = logits(sc, clf, X[va]); T = temp_fit(Lv, y[va])
    Lt = logits(sc, clf, X[te]); Ltf = logits(sc, clf, Xf[te])
    Pv = softmax(Lv / T, 1); Pt = softmax(Lt / T, 1); Ptt = (softmax(Lt / T, 1) + softmax(Ltf / T, 1)) / 2
    pred = np.array(cls)[Pt.argmax(1)]; yt = y[te]
    r = dict(variant=variant, C=C, T=T, acc=accuracy_score(yt, pred), macroF1=mf1(yt, pred),
             val_acc=accuracy_score(y[va], np.array(cls)[Pv.argmax(1)]), val_macroF1=mf1(y[va], np.array(cls)[Pv.argmax(1)]),
             ece_test_before=ece(softmax(Lt, 1), yt), ece_test_after=ece(Pt, yt),
             tta_macroF1=mf1(yt, np.array(cls)[Ptt.argmax(1)]), tta_acc=accuracy_score(yt, np.array(cls)[Ptt.argmax(1)]))
    # bootstrap stratified by class
    idx_by = {c: np.where(yt == c)[0] for c in cls}; bs = []
    for _ in range(2000):
        ii = np.concatenate([rng.choice(v, len(v)) for v in idx_by.values()]); bs.append(mf1(yt[ii], pred[ii]))
    r["macroF1_CI"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    p, rc, f, s = precision_recall_fscore_support(yt, pred, labels=cls, zero_division=0)
    r["per_class"] = {int(c): dict(name=names[c], precision=float(p[i]), recall=float(rc[i]), f1=float(f[i]), support=int(s[i])) for i, c in enumerate(cls)}
    r["cm"] = confusion_matrix(yt, pred, labels=cls).tolist()
    return r, Pv, Pt, T

def abstain(Pv, Pt, yv, yt, target=0.95):
    def sc_(P): s = np.sort(P, 1); return s[:, -1], s[:, -1] - s[:, -2]
    mv, dv = sc_(Pv); mt, dt = sc_(Pt); pv = np.array(cls)[Pv.argmax(1)]; pt = np.array(cls)[Pt.argmax(1)]
    best = None
    for tau in np.linspace(0.3, 0.99, 70):
        for mg in [0, .05, .1, .2, .3, .4, .5, .6]:
            k = (mv >= tau) & (dv >= mg)
            if k.sum() < 0.3 * len(yv): continue
            a = (pv[k] == yv[k]).mean()
            if a >= target and (best is None or k.mean() > best[0]): best = (k.mean(), tau, mg, a)
    if best is None: return None
    _, tau, mg, va_acc = best
    k = (mt >= tau) & (dt >= mg)
    d = dict(tau=float(tau), margin=float(mg), val_coverage=float(best[0]), val_sel_acc=float(va_acc), target=target,
             test_coverage=float(k.mean()), test_sel_acc=float((pt[k] == yt[k]).mean()) if k.any() else None,
             test_n_answered=int(k.sum()),
             per_class_abstain={int(c): dict(name=names[c], n=int((yt == c).sum()), abstained=int((~k & (yt == c)).sum())) for c in cls})
    root = [1, 2, 5, 6, 14, 15]  # Fusarium, Macrophomina, Phytophthora, Verticillium, 2 nematodes
    rm = np.isin(yt, root); d["root_crown_abstain_frac"] = float((~k & rm).sum() / rm.sum()); d["root_crown_n"] = int(rm.sum())
    d["root_crown_sel_acc"] = float((pt[k & rm] == yt[k & rm]).mean()) if (k & rm).any() else None
    d["nonroot_abstain_frac"] = float((~k & ~rm).sum() / (~rm).sum())
    # risk-coverage curve on test (by max-prob), for plot
    o = np.argsort(-mt); cor = (pt[o] == yt[o]).cumsum() / np.arange(1, len(o) + 1)
    d["_rc"] = (np.arange(1, len(o) + 1) / len(o), cor)
    return d

allr = {}
for v in models:
    r, Pv, Pt, T = evaluate(v)
    for tgt in (0.95, 0.90):
        a = abstain(Pv, Pt, y[va], y[te], tgt)
        if a: rc = a.pop("_rc")
        r[f"abstain_{int(tgt*100)}"] = a
        if tgt == 0.95: r["_rc"] = rc if a else None
    allr[v] = r
    print(v, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items() if k in ("acc", "macroF1", "macroF1_CI", "T", "tta_macroF1", "ece_test_before", "ece_test_after")})

# source-shortcut control: metadata only (width, height, aspect, filesize, ext)
meta = []
for p in df.fpath:
    w, h = Image.open(p).size; meta.append([w, h, w / h, np.log(os.path.getsize(p)), p.lower().endswith(".png")])
M = np.array(meta, float); M[:, 0:2] = np.log(M[:, 0:2])
scm = StandardScaler().fit(M[tr]); cm = LogisticRegression(C=1, class_weight="balanced", max_iter=3000).fit(scm.transform(M[tr]), y[tr])
pm = cm.predict(scm.transform(M[te]))
ctrl = dict(acc=accuracy_score(y[te], pm), macroF1=mf1(y[te], pm)); print("metadata-only control", ctrl)
# also a tiny-corner (border) pixel probe: mean colour of 8px border strips + 16x16 downsample of outer ring
def border(p):
    a = np.asarray(Image.open(p).convert("RGB").resize((32, 32))).astype(float) / 255
    ring = np.concatenate([a[:3].reshape(-1, 3), a[-3:].reshape(-1, 3), a[:, :3].reshape(-1, 3), a[:, -3:].reshape(-1, 3)])
    return ring.reshape(-1)
Bm = np.array([border(p) for p in df.fpath]); scb = StandardScaler().fit(Bm[tr])
cb = LogisticRegression(C=0.01, class_weight="balanced", max_iter=2000).fit(scb.transform(Bm[tr]), y[tr]); pb = cb.predict(scb.transform(Bm[te]))
ctrl2 = dict(acc=accuracy_score(y[te], pb), macroF1=mf1(y[te], pb)); print("border-pixel control", ctrl2)

# plots + artifacts for main variant
r = allr[main]; cmx = np.array(r["cm"]); lab = [f"{c} {names[c][:14]}" for c in cls]
fig, ax = plt.subplots(1, 2, figsize=(16, 7))
for a, M_, t, fmt in [(ax[0], cmx, "counts", "d"), (ax[1], cmx / cmx.sum(1, keepdims=True), "row-normalised", ".2f")]:
    a.imshow(M_, cmap="Blues"); a.set_xticks(range(len(cls))); a.set_yticks(range(len(cls))); a.set_xticklabels(lab, rotation=75, ha="right", fontsize=7); a.set_yticklabels(lab, fontsize=7)
    for i in range(len(cls)):
        for j in range(len(cls)): a.text(j, i, format(M_[i, j], fmt), ha="center", va="center", fontsize=6, color="red" if M_[i,j]>M_.max()*.6 else "black")
    a.set_title(f"Test confusion ({t}) - {main}"); a.set_xlabel("pred"); a.set_ylabel("true")
plt.tight_layout(); plt.savefig(f"{REPORTS}/confusion_probe.png", dpi=130); plt.close()
if r["_rc"] is not None:
    plt.figure(figsize=(5, 4)); plt.plot(*r["_rc"]); plt.xlabel("coverage"); plt.ylabel("accuracy on answered"); plt.title("Test risk-coverage (max-softmax)"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{REPORTS}/risk_coverage.png", dpi=130); plt.close()
for v in allr: allr[v].pop("_rc", None)
sc, clf, C = models[main]
joblib.dump(dict(scaler=sc, clf=clf, T=allr[main]["T"], classes=cls, variant=main), f"{MODELS}/probe_best.joblib")
json.dump(dict(results=allr, grids={k: res[k] for k in res}, main=main, ctrl_metadata=ctrl, ctrl_border=ctrl2), open(f"{REPORTS}/probe_results.json", "w"), indent=1, default=float)
