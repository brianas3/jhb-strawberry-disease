# JHB 草莓病害影像分類：目前進度

JH Biotech（JHB）草莓病害影像辨識專案（GenAI 課程，2027 Fall）。
本 repo 只放程式碼與分析報告，**不含影像資料集**（資料為網路蒐集，另有授權疑慮；放在本機 `JHB training dataset/`）。

> 更新日期：2026-11-20。主要探針結果來自 15 類、476 張 test 的固定切分；多次重切分結果另行標示。資料均為網路蒐集，尚無獨立田間外部測試集，以下結果請視為研究原型的初步評估。

## 1. 資料集概況

- 15 個類別已補齊影像（包含先前缺少的第 8、9、10 類），現共 3,198 張。
- **嚴重不平衡**：Leaf Scorch（11）佔 46.3%，最小的 Macrophomina（2）只有 14 張，最大與最小約 105:1。
- 0 張損毀。資料稽核列出 160 組完全重複與 179 組近似重複群組；去重統計詳見 [`reports/data_audit.md`](reports/data_audit.md)。
- 現行 manifest 將 **52 個跨類別衝突群組、共 136 張圖片**標記為 `label_conflict=1`，且都在 train split；同一圖片出現在多個類別的標籤可能有誤。
- **來源捷徑風險**：第 3、4 類全是 419×419，第 11 類 75% 是 256×256（Kaggle、iNaturalist）。模型可能學到「來源與解析度」而不是病徵。

詳見 [`reports/data_audit.md`](reports/data_audit.md)。

## 2. 資料切分

- 以「近似重複群組」為單位切分，確保 train/val/test 之間沒有相同 group_id。現行 manifest：train 2,239、val 483、test 476。
- 衝突圖（跨類別重複）標記 `label_conflict=1`。
- 切分檔：[`splits/manifest.csv`](splits/manifest.csv)

## 3. 模型與結果

| 模型 | Test Accuracy | 單視角 Macro-F1 | 四視角 TTA Accuracy | 四視角 TTA Macro-F1 | 備註 |
|---|---:|---:|---:|---:|---|
| **DINOv2-S/14 凍結特徵 + Linear Probe（保留衝突圖）** | 0.817 | 0.576 | 0.807 | **0.588** | 依 validation Macro-F1 選為主模型；TTA 對 Macro-F1 小幅提升、Accuracy 略降 |
| 同上，丟棄衝突圖 | **0.826** | 0.568 | 0.830 | 0.582 | 不同指標的排序不完全相同 |
| **MobileNetV3-L 兩階段微調（既有結果，2026-11-13）** | 0.758 | 0.496 | - | 0.545 | CB-Focal Loss、$1/\sqrt{N}$ 平滑採樣、DINOv2 雙重蒸餾；微調的 TTA 為水平翻轉 |
| MobileNetV3-L 舊版微調基準 | 0.727 | 0.478 | - | 0.461 | CE Loss + 1/N 採樣 |
| 對照：只用影像 metadata（寬、高、大小、格式） | 0.586 | 0.212 | - | - | 顯示資料中來源捷徑依然存在 |
| 對照：只用影像邊框像素 | 0.494 | 0.182 | - | - | 邊框偽影基準 |

- DINOv2 探針使用原圖、水平翻轉、垂直翻轉、旋轉 90 度四種特徵做 TTA 機率平均。主模型 Macro-F1 由 0.576 升至 0.588（+0.012），但 Accuracy 由 0.817 降至 0.807；改善幅度小，且不是所有指標都提升。
- 主模型依 validation Macro-F1 選擇保留衝突圖版本（val Macro-F1 0.616）；丟棄衝突圖版本的 test Accuracy 較高。請勿把「主模型」解讀為所有指標最佳。
- MobileNetV3-L 是 11 月 13 日保存的既有微調結果，非本次重新訓練。CPU 微調實際保存的執行時間約 434 秒；不同硬體和設定會有差異。
- **長尾增強五次重切分比較未見改善**：baseline Macro-F1 為 0.547 ± 0.041，增強版為 0.521 ± 0.066；五次中增強版只在一次勝出。
- **只用 metadata 就有近 59% 準確率**，代表資料中的來源捷徑是真實存在的。

### 各類別表現（主模型）

單視角結果中，較高的類別包括 Powdery mildew（類別 4，F1 0.949）、Leaf Scorch（類別 11，F1 0.946）、Gray Mold（類別 3，F1 0.929）。較弱的類別包括：

- Common Leaf Spot（類別 9，F1 0.214）、Macrophomina Crown Rot（類別 2，F1 0.250）、Rhizopus Fruit Rot（類別 13，F1 0.286）、Phytophthora Crown rot（類別 5，F1 0.316）。
- 小類別 test support 只有 2~19 張；上述 per-class 分數波動很大，Macrophomina 僅 2 張 test 圖片。

以上為主模型單視角數字；混淆矩陣：[`reports/confusion_probe.png`](reports/confusion_probe.png)

### 校準與「不確定就拒答」

溫度縮放後，用 max-softmax 加 top-2 margin 設定拒答門檻（在 val 上選，test 僅用於報告）：

- validation 目標 95%：選得 tau=0.66、margin=0.40；test 覆蓋率 77.3%，已回答樣本準確率 93.2，未達 95%。
- validation 目標 90%：選得 tau=0.49、margin=0.20；test 覆蓋率 85.1%，已回答樣本準確率 88.1，未達 90%。
- 以類別 1、2、5、6、14、15 作為根／冠部相關組別時，目標 95% 門檻在 test 拒答 49.1%；已回答樣本準確率 75.9%。少數類別的拒答與準確率仍不穩定。
- 風險覆蓋曲線：[`reports/risk_coverage.png`](reports/risk_coverage.png)

## 4. Red-team 檢查結果（數字可能被高估）

以目前保存的 15 類評估檔為準（詳見 [`reports/redteam_core.json`](reports/redteam_core.json)）：

| 結論 | 狀態 |
|---|---|
| train/val/test group_id 重疊 | 0；但 pHash ≤ 10 的 train/test 近似圖片有 19 張，仍有殘餘近重複風險 |
| 固定切分的主模型 test Macro-F1 | 單視角 0.576；四視角 TTA 0.588 |
| 五次重切分、丟棄衝突圖 | Macro-F1 0.546 ± 0.034 |
| 五次重切分、保留衝突圖 | Macro-F1 0.555 ± 0.062；丟棄衝突圖沒有穩定優勢 |
| 一般 group-CV | Macro-F1 0.578 |
| 依類別／尺寸群組切分的 cluster-CV | Macro-F1 0.205；此為嚴格分布外測試，部分類別可能因整個群組被留出而難以學習 |
| 第 11 類跨解析度泛化 | 非 256 寬度圖片 recall 0.019（n=371）；256 寬度 test recall 1.000，顯示強烈來源／尺寸風險 |

另有一項已完成的長尾增強比較：五次重切分平均 Macro-F1 由 baseline 0.547 降至 0.521，尚未證明增強有幫助。目視抽查也發現小類別的問題：浮水印、文字說明、非草莓植物、疑似 AI 生成圖，以及 Macrophomina 說明文字與標籤不符等。

**目前應同時報告固定切分 TTA Macro-F1 0.588 與重切分約 0.55；前者不是外部測試或田間泛化保證。** 最近一次 red-team 重跑只留下部分 pHash 輸出，未更新最終 JSON；上表引用的是目前保存的完整結果，不是該次未完成重跑。

完整成果：[`reports/model_results_2026-11-20.md`](reports/model_results_2026-11-20.md)。舊版 12 類結果仍保留於 [`reports/baseline_results.md`](reports/baseline_results.md) 和 [`reports/redteam_report.md`](reports/redteam_report.md)，閱讀時請留意其資料版本。

## 5. 待辦與需要討論

1. 第 8、9、10 類影像已補充加入訓練（角斑病 92 張、普通葉斑病 58 張、葉枯病 76 張），達成 15 類全覆蓋；但這三類均屬葉斑病徵，與第 7、11、12 類極易混淆，且部分帶有跨類別衝突標籤，後續仍需請領域專家協助審定田間真圖。
2. 清理小類別：移除非草莓、浮水印、混合疾病的圖片，並請領域專家判定 manifest 標記的跨類別衝突群組。
3. 補強去重（pHash ≤ 10 加 embedding 相似度），然後重新切分。
4. 已加入五次 group-aware 重切分結果；後續仍需在更多 seed 與外部資料上確認穩定性。
5. 已實作縮放、JPEG、灰階等長尾增強，但目前五次重切分沒有提升；第 3、4、11 類仍需多來源影像，並以客戶實拍照片作外部測試集。
6. 確認應用情境（手機田間照片，還是網路品質影像）與成功指標（macro-F1，或特定疾病的 recall）。
7. 網路爬取影像的授權（Plantix、iNaturalist、期刊圖片）要確認後才能用在交付產品。

## 6. 專案結構

```
src/
  audit.py              資料稽核（損毀、重複、跨類重複）
  make_splits.py        群組感知的分層切分
  common.py             共用函式
  extract_features.py   DINOv2 特徵抽取（--flip/--vflip/--rot90 可供 TTA）
  extract_features_aug.py 長尾資料增強特徵抽取
  eval_aug_resplit.py   五次重切分比較增強與 baseline
  train_probe.py        線性探針、多視角 TTA、溫度縮放、拒答門檻
  find_label_issues.py  標籤問題候選與抽查表
  finetune.py           MobileNetV3 兩階段微調
  redteam_core.py       洩漏與重現檢查
  redteam_stress.py     來源壓力測試
splits/manifest.csv     train/val/test 切分
reports/                稽核、基準、red-team 與研究摘要
models/probe_best.joblib  主模型（標準化器、LR、溫度、類別）
reports/model_results_2026-11-20.md 本次成果報告
run.sh                  一鍵跑特徵抽取與探針
```

## 7. 重現方式

資料集不在 repo 中，先自行放置影像。`run.sh` 是 Bash 腳本，且預設只抽原圖與水平翻轉特徵；Windows PowerShell 可直接呼叫虛擬環境 Python。若使用專案預設的 `JHB training dataset/` 路徑，通常不必另外設定環境變數。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 若資料集不在預設位置，設定實際資料集路徑
$env:DATA_DIR = "C:\path\to\JHB training dataset"
$env:MANIFEST = "splits\manifest.csv"
$env:OUT_DIR = "."

# 四視角 TTA：特徵抽取後重新訓練／評估探針
.\.venv\Scripts\python.exe src\extract_features.py
.\.venv\Scripts\python.exe src\extract_features.py --flip
.\.venv\Scripts\python.exe src\extract_features.py --vflip
.\.venv\Scripts\python.exe src\extract_features.py --rot90
.\.venv\Scripts\python.exe src\train_probe.py

# 長尾增強比較（另一項實驗；需要先有 plain 特徵）
.\.venv\Scripts\python.exe src\extract_features_aug.py
.\.venv\Scripts\python.exe src\eval_aug_resplit.py

# 搜尋標籤問題候選
.\.venv\Scripts\python.exe src\find_label_issues.py
```

- `manifest.csv` 內的路徑是相對路徑（`JHB training dataset/<類別>/<檔名>`），程式會把 `JHB training dataset/` 之後的部分對應到 `DATA_DIR` 下。`audit.py` 也可用 `DATA_DIR`、`OUT_DIR` 環境變數設定。
- 已有 `feats_plain.npy`、`feats_flip.npy`、`feats_vflip.npy`、`feats_rot90.npy` 時，可跳過相應特徵抽取，直接執行 `train_probe.py`；它會自動載入存在的 TTA 特徵。
- DINOv2 權重第一次使用時可能需要網路下載。裝置自動選擇 cuda 或 cpu；CUDA 路徑尚未完整驗證。
- 為了縮小 repo，未上傳 `models/feats_*.npy`（可由 `run.sh` 重新產生）與 `models/mnv3_best.pt`（非建議模型），也未上傳 `reports/samples/`、`reports/redteam_grids/`（內含資料集影像縮圖）。
