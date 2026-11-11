# JHB 草莓病害影像分類：目前進度

JH Biotech（JHB）草莓病害影像辨識專案（GenAI 課程，2027 Fall）。
本 repo 只放程式碼與分析報告，**不含影像資料集**（資料為網路蒐集，另有授權疑慮；放在本機 `JHB training dataset/`）。

> 更新日期：2026-11-11。以下數字皆為單一切分、單一 seed，請當作初步結果。

## 1. 資料集概況

- 15 個類別，**只有 12 類有影像**，共 2,989 張。第 8（Angular Leaf Spot）、9（Common Leaf Spot）、10（Gnomonia leaf blotch）是空的，目前無法訓練與評估。
- **嚴重不平衡**：Leaf Scorch（11）佔 49.5%，最小的 Macrophomina（2）只有 15 張，最大與最小約 99:1。
- 0 張損毀。有 87 組完全相同的重複圖，加上近似重複後，230 張（7.7%）落在重複群組。
- **跨類別重複 36 組、79 張**：同一張圖被放進兩個以上不同疾病資料夾（標籤雜訊，最多的是 14 與 15、2 與 5）。
- **來源捷徑風險**：第 3、4 類全是 419×419，第 11 類 75% 是 256×256（Kaggle、iNaturalist）。模型可能學到「來源與解析度」而不是病徵。

詳見 [`reports/data_audit.md`](reports/data_audit.md)。

## 2. 資料切分

- 70/15/15，每類分層，並以「近似重複群組」為單位切分，確保 train/val/test 之間沒有群組重疊。
- Train 2,091 / Val 451 / Test 447。
- 79 張標籤衝突圖只放 train，並標記 `label_conflict=1`。
- 切分檔：[`splits/manifest.csv`](splits/manifest.csv)

## 3. 模型與結果（Test，12 類，n=447）

| 模型 | Accuracy | Macro-F1 |
|---|---|---|
| **DINOv2-S/14 凍結特徵 + Logistic Regression**（主模型，丟棄衝突圖） | 0.879 | **0.677** |
| 同上 + flip-TTA | 0.886 | 0.680 |
| 同上，保留衝突圖 | 0.839 | 0.603 |
| MobileNetV3-L 兩階段微調（CPU，只訓練 8 個 epoch） | 0.779 | 0.514 |
| 對照：只用影像 metadata（寬、高、檔案大小、格式） | 0.613 | 0.270 |
| 對照：只用影像邊框像素 | 0.532 | 0.214 |
| 多數類別基準（全猜 Leaf Scorch） | 0.497 | n/a |

- 主模型：先用 DINOv2 抽取並快取特徵，再訓練線性分類器。CPU 上幾分鐘就能跑完。
- 微調版本比凍結特徵差，但只跑了縮短的單次實驗，不能下「微調較差」的一般性結論。
- **只用 metadata 就有 61% 準確率**，代表資料中的來源捷徑是真實存在的。

### 各類別表現（主模型）

強的類別是大型、單一來源的類別：Leaf Scorch、Gray Mold、Powdery mildew 的 F1 都很高。
弱的類別是小類別：

- Verticillium Wilt：recall 只有 **0.09**。
- Phytophthora、Foliar nematode、Fusarium：precision 與 recall 約 0.4~0.6。
- 小類別測試集只有 2~20 張（Macrophomina 只有 2 張），per-class 數字波動很大。

混淆矩陣：[`reports/confusion_probe.png`](reports/confusion_probe.png)

### 校準與「不確定就拒答」

溫度縮放後，用 max-softmax 加 top-2 margin 設定拒答門檻（在 val 上選，test 只用一次）：

- 目標 95%：覆蓋率 74.5%，已回答的準確率 95.2%。
- 但這個 95% 主要由 Leaf Scorch、Gray Mold、Powdery mildew 撐起來（佔已回答的 89.6%）。小類別（根部、冠部病害）有 62.7% 被拒答，已回答的準確率只有 77%。
- 風險覆蓋曲線：[`reports/risk_coverage.png`](reports/risk_coverage.png)

## 4. Red-team 檢查結果（數字可能被高估）

獨立重跑後：

| 結論 | 狀態 |
|---|---|
| Acc 0.879 / macro-F1 0.677 可重現 | 成立 |
| 沒有資料洩漏 | 部分削弱：測試集有 34 張與 train 很相似（pHash ≤ 10），移除後 macro-F1 掉約 0.02 |
| 0.677 具代表性 | **削弱**：重新切分 5 次平均約 **0.59 ± 0.04**，group-CV 約 0.61 |
| 丟棄衝突圖會變好 | **推翻**：多次重切後差異可忽略，方向不一致 |
| 模型對來源不敏感 | 削弱：降解析度、JPEG、灰階後 macro-F1 最多掉 0.075 |
| 第 11 類的泛化 | 用非 256 解析度的第 11 類圖片測試，recall 只有 0.02 |

目視抽查也發現小類別的問題：浮水印、文字說明、非草莓植物（番茄、萵苣、玉簪等）、疑似 AI 生成圖，以及 Macrophomina 的說明文字其實寫 Colletotrichum。

**目前比較誠實的 macro-F1 是約 0.6，不是 0.677。**

詳見 [`reports/baseline_results.md`](reports/baseline_results.md) 與 [`reports/redteam_report.md`](reports/redteam_report.md)。

## 5. 待辦與需要討論

1. 第 8、9、10 類沒有資料。向客戶要至少 100 張以上、有專家確認的田間照片，或先交付 12 類模型加「未知」拒絕機制。
2. 清理小類別：移除非草莓、浮水印、混合疾病的圖片，並請有領域知識的人判定 36 組跨類別重複。
3. 補強去重（pHash ≤ 10 加 embedding 相似度），然後重新切分。
4. 報告改用多次 group-aware 切分的 mean ± sd，不要只看單一 split。
5. 第 3、4、11 類需要多來源影像，並加入縮放、JPEG、灰階等增強；最好用客戶實拍照片當外部測試集。
6. 確認應用情境（手機田間照片，還是網路品質影像）與成功指標（macro-F1，或特定疾病的 recall）。
7. 網路爬取影像的授權（Plantix、iNaturalist、期刊圖片）要確認後才能用在交付產品。

## 6. 專案結構

```
src/
  audit.py              資料稽核（損毀、重複、跨類重複）
  make_splits.py        群組感知的分層切分
  common.py             共用函式
  extract_features.py   DINOv2 特徵抽取（--flip 為 TTA）
  train_probe.py        線性探針、溫度縮放、拒答門檻
  finetune.py           MobileNetV3 兩階段微調
  redteam_core.py       洩漏與重現檢查
  redteam_stress.py     來源壓力測試
splits/manifest.csv     train/val/test 切分
reports/                稽核、基準、red-team 與研究摘要
models/probe_best.joblib  主模型（標準化器、LR、溫度、類別）
run.sh                  一鍵跑特徵抽取與探針
```

## 7. 重現方式

資料集不在 repo 中，先自行放置影像，再設定環境變數：

```bash
pip install -r requirements.txt

export DATA_DIR="/path/to/JHB training dataset"   # 影像資料夾
export MANIFEST="splits/manifest.csv"
export OUT_DIR="."                                  # 輸出 models/ 與 reports/

bash run.sh                      # 特徵抽取 + 探針
RUN_FINETUNE=1 bash run.sh       # 另外跑微調（CPU 約 20+ 分鐘）
```

- `manifest.csv` 內的路徑是相對路徑（`JHB training dataset/<類別>/<檔名>`），程式會把 `JHB training dataset/` 之後的部分對應到 `DATA_DIR` 下。`audit.py` 也可用 `DATA_DIR`、`OUT_DIR` 環境變數設定。
- 裝置自動選擇 cuda 或 cpu。CUDA 路徑尚未測試。
- 為了縮小 repo，未上傳 `models/feats_*.npy`（可由 `run.sh` 重新產生）與 `models/mnv3_best.pt`（非建議模型），也未上傳 `reports/samples/`、`reports/redteam_grids/`（內含資料集影像縮圖）。
