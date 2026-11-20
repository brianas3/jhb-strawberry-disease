"""找出「可能標錯」的圖片，做成名單和縮圖表，交給客戶或專家確認（放在 src/ 底下）。

不需要懂草莓病害：讓模型先挑出可疑的圖，最後由人判斷。
  1. 用已經抽好的 DINOv2 特徵（models/feats_plain.npy），對 train + val 做分組 5-fold 交叉驗證，
     每張圖都由「沒看過它」的模型來預測（out-of-fold 機率）。
  2. cleanlab 的 find_label_issues：模型很有把握猜成別的類別的圖，就標成可疑。
  3. 補充兩個訊號：DINOv2 特徵空間裡最像的 10 張圖有幾成標籤不一樣（knn_disagree），
     以及 audit 已經找到的跨類別重複圖（cross_class_dup，同一張圖放在兩個病害資料夾）。

重要規則：
  - 這支程式「不會刪任何圖」，只輸出名單。刪不刪、改不改由專家決定。
  - test 不參與，也不出現在名單裡：不能根據模型的意見去改 test，否則分數會被灌水。
  - 同一個近似重複群組（group_id）的圖一定放在同一個 fold，避免模型「看過幾乎一樣的圖」。

輸出：
  reports/label_issues.csv          全部 train+val 圖片，依可疑程度排序；最後一欄 expert_label 留給專家填
  reports/label_issues_sheet.jpg    前 TOP_N 張可疑圖的縮圖表（含資料集影像，不要 commit 到 repo）
用法：
  pip install cleanlab              # 沒裝的話會退回只用「給定標籤的機率」排序
  python3 src/find_label_issues.py
Env：TOP_N（預設 40）、PROBE_C（LR 的 C，預設 0.01）、KNN（預設 10）
"""
import os, sys, warnings
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import StratifiedGroupKFold
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

TOP_N = int(os.environ.get("TOP_N", 40))
C = float(os.environ.get("PROBE_C", 0.01))
KNN = int(os.environ.get("KNN", 10))

ensure_dirs(); df = load_manifest()
X_all = np.load(f"{MODELS}/feats_plain.npy").astype(np.float32)
assert len(X_all) == len(df), "feats_plain.npy 和 manifest 列數不同，請先重跑 extract_features.py"

# ---- 1. 只用 train + val，test 不碰 ----
keep = (df.split != "test").values
d = df[keep].reset_index(drop=True); X = X_all[keep]
cls = sorted(d.class_id.unique()); ci = {c: i for i, c in enumerate(cls)}
labels = d.class_id.map(ci).values                     # cleanlab 需要 0..K-1 的標籤
names = d.drop_duplicates("class_id").set_index("class_id").class_name.to_dict()

# ---- 2. 分組 5-fold，取得每張圖的 out-of-fold 機率 ----
probs = np.zeros((len(d), len(cls)))
for k, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, labels, d.group_id)):
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(sc.transform(X[tr]), labels[tr])
    p = np.zeros((len(te), len(cls))); p[:, clf.classes_] = clf.predict_proba(sc.transform(X[te]))  # 某類在 fold 缺席時補 0
    probs[te] = p
    print(f"fold {k}: {len(te)} 張", flush=True)

# ---- 3. cleanlab：找出可能標錯的圖 ----
try:
    from cleanlab.filter import find_label_issues
    ranked = find_label_issues(labels, probs, return_indices_ranked_by="self_confidence")
    flag = np.zeros(len(d), bool); flag[ranked] = True
    print(f"cleanlab 標出 {flag.sum()} 張可疑圖")
except ImportError:
    print("沒有安裝 cleanlab，改用簡單規則：模型猜別的類別，而且給定標籤的機率 < 0.2")
    flag = (probs.argmax(1) != labels) & (probs[np.arange(len(d)), labels] < 0.2)

# ---- 4. 補充訊號：最像的 K 張圖裡，有幾成標籤不一樣 ----
Z = normalize(X); sim = Z @ Z.T
same_group = d.group_id.values[:, None] == d.group_id.values[None, :]
sim[same_group] = -np.inf                              # 不把自己和自己的重複圖算進鄰居
nn = np.argsort(-sim, 1)[:, :KNN]
knn_disagree = (labels[nn] != labels[:, None]).mean(1)

# ---- 5. 整理名單 ----
p_given = probs[np.arange(len(d)), labels]
pred = np.array(cls)[probs.argmax(1)]
out = pd.DataFrame(dict(
    path=d.path, split=d.split, class_id=d.class_id, class_name=d.class_name,
    pred_class=pred, pred_name=[names[c] for c in pred], pred_prob=probs.max(1).round(3),
    given_label_prob=p_given.round(3), knn_disagree=knn_disagree.round(2),
    cleanlab_flag=flag.astype(int), cross_class_dup=d.label_conflict.astype(int), group_id=d.group_id))
out["suspect"] = ((out.cleanlab_flag == 1) | (out.cross_class_dup == 1)).astype(int)
# 排序：可疑的排前面，再依「給定標籤的機率」由低到高
out = out.sort_values(["suspect", "given_label_prob"], ascending=[False, True]).reset_index(drop=True)
out["expert_label"] = ""                                # 留給專家填：正確類別 / 非草莓 / 無法判斷
out.to_csv(f"{REPORTS}/label_issues.csv", index=False)
print(f"可疑圖共 {out.suspect.sum()} 張（cleanlab {out.cleanlab_flag.sum()}、跨類別重複 {out.cross_class_dup.sum()}）")
print(out[out.suspect == 1].groupby("class_name").size().sort_values(ascending=False).to_string())

# ---- 6. 縮圖表：前 TOP_N 張可疑圖，標示原本標籤和模型猜的標籤 ----
top = out[out.suspect == 1].head(TOP_N)
fpath = dict(zip(df.path, df.fpath))
W, H, COLS = 220, 260, 5
rows = max(1, -(-len(top) // COLS))
sheet = Image.new("RGB", (W * COLS, H * rows), "white"); draw = ImageDraw.Draw(sheet)
try: font = ImageFont.load_default(size=12)
except TypeError: font = ImageFont.load_default()
for i, r in enumerate(top.itertuples()):
    x, y0 = (i % COLS) * W, (i // COLS) * H
    im = Image.open(fpath[r.path]).convert("RGB"); im.thumbnail((W - 10, H - 60))
    sheet.paste(im, (x + 5, y0 + 5))
    tag = " [dup]" if r.cross_class_dup else ""
    draw.text((x + 5, H - 52 + y0), f"#{i+1} label: {r.class_id} {r.class_name[:22]}", fill="black", font=font)
    draw.text((x + 5, H - 36 + y0), f"pred: {r.pred_class} {r.pred_name[:20]} ({r.pred_prob:.2f})", fill="red", font=font)
    draw.text((x + 5, H - 20 + y0), f"p(label)={r.given_label_prob:.2f} knn={r.knn_disagree:.1f}{tag}", fill="gray", font=font)
sheet.save(f"{REPORTS}/label_issues_sheet.jpg", quality=90)
print("已輸出 reports/label_issues.csv 和 reports/label_issues_sheet.jpg（#編號對應 csv 的可疑排序）")
