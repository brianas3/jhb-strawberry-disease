# 草莓病害分類模型成果報告

日期：2026-11-20

## 摘要

目前資料集有 3,198 張圖片、15 個類別；本次固定切分為 train 2,239 張、validation 483 張、test 476 張。最佳驗證集版本是保留訓練集標籤衝突圖片的 DINOv2-S/14 凍結特徵線性探針。加入原圖、水平翻轉、垂直翻轉與旋轉 90 度的 TTA 投票後，測試 Macro-F1 從 0.576 升至 0.588；Accuracy 則從 0.817 降至 0.807。

TTA 帶來小幅 Macro-F1 改善，但單一測試切分、少數類測試圖片很少，且不同種子重切分的結果波動明顯。因此目前結果適合做原型與研究比較，不足以證明模型可直接用於田間診斷。

## 1. 本次評估設定

- 特徵：DINOv2-S/14 凍結特徵，768 維。
- 分類器：標準化後的 class-balanced Logistic Regression；依 validation Macro-F1 選擇 C。
- 主模型：保留 train split 中標記為 `label_conflict=1` 的圖片；validation/test 不含衝突圖片。
- 主模型 C：0.03；validation Macro-F1：0.616。
- 溫度縮放參數：T = 1.057，在 validation 上估計。
- TTA：平均原圖、水平翻轉、垂直翻轉、旋轉 90 度四種視角的預測機率。
- 這次 TTA 評估使用同一份 476 張 test split；沒有另行訓練新的 DINOv2 backbone。

## 2. 固定 test split 結果

| 訓練版本 | Test Accuracy | 單視角 Macro-F1 | 四視角 TTA Accuracy | 四視角 TTA Macro-F1 |
|---|---:|---:|---:|---:|
| 保留衝突圖片（validation 選出的主模型） | 0.817 | 0.576 | 0.807 | **0.588** |
| 丟棄衝突圖片 | **0.826** | 0.568 | 0.830 | 0.582 |

在主模型上，TTA 的 Macro-F1 增加約 0.012，但 Accuracy 減少約 0.011。丟棄衝突圖片的版本則在 TTA 後 Macro-F1 增加約 0.014，Accuracy 增加約 0.004。這表示 TTA 對 Macro-F1 有幫助，但不代表每一項指標都同步改善。

主模型單視角 Macro-F1 的分層 bootstrap 95% 區間為 0.512–0.641；此區間不是 TTA 的信賴區間，且未納入不同資料切分造成的變異。少數類 test support 很小，因此單次結果不宜過度解讀。

### 類別表現觀察（主模型、單視角）

- 表現較高：Powdery mildew（類別 4，F1 0.949，n=60）、Leaf Scorch（類別 11，F1 0.946，n=222）、Gray Mold（類別 3，F1 0.929，n=71）。
- 表現較弱：Common Leaf Spot（類別 9，F1 0.214，n=9）、Macrophomina Crown Rot（類別 2，F1 0.250，n=2）、Rhizopus Fruit Rot（類別 13，F1 0.286，n=8）、Phytophthora Crown rot（類別 5，F1 0.316，n=10）。
- 以上 per-class 數字是單視角結果；目前報表沒有提供各類別的 TTA 指標。

## 3. 長尾增強的五次重切分比較

增強特徵只用在各次切分的訓練資料，validation/test 仍使用原圖特徵。

| 方法 | 五次平均 Macro-F1 | 標準差 | 平均 Accuracy |
|---|---:|---:|---:|
| 不使用長尾增強 | 0.547 | 0.041 | 0.858 |
| 使用長尾增強 | 0.521 | 0.066 | 0.840 |

增強版 Macro-F1 平均低約 0.026；五次中只有一次高於 baseline。以目前實驗而言，長尾增強沒有帶來穩定改善。

## 4. 其他已保存結果

- MobileNetV3-L 兩階段微調的已保存結果（2026-11-13，CPU、完成 12 個 stage-2 epochs）：Accuracy 0.758、單視角 Macro-F1 0.496、Flip-TTA Macro-F1 0.545、最佳 validation Macro-F1 0.526。這是既有微調結果，並非本次重新訓練。
- 可用的 red-team 結果檔記錄：五次重切分 Macro-F1 為 drop-conflicts 0.546 ± 0.034、keep-conflicts 0.555 ± 0.062；group-CV Macro-F1 0.578。依類別／尺寸群組切分的 cluster-CV Macro-F1 為 0.205；類別 11 在非 256 寬度圖片上的 recall 為 0.019（n=371），而 256 寬度 test 圖片上的 recall 為 1.000。這些結果提示來源／解析度捷徑風險，但 cluster-CV 是刻意嚴格的分布外測試，不等同一般隨機切分成績。
- Red-team 腳本最近一次執行曾產生部分 pHash 檔案，但沒有更新最終 JSON；上面的 red-team 數字是目前保存的完整結果，不代表那次未完成執行的結果。

## 5. 結論與限制

1. 目前最佳單次 test Macro-F1 是四視角 TTA 的 0.588；相較主模型單視角 0.576 有小幅提升，但 TTA Accuracy 反而降低。
2. 五次重切分的長尾增強結果未勝過 baseline；不應宣稱這項增強已提升泛化能力。
3. 大類別表現較好，小類別測試樣本少且結果不穩；部分類別只有 2–10 張 test 圖片。
4. 解析度／來源分組測試顯示明顯泛化落差；目前資料以網路蒐集圖片為主，尚無獨立的田間或客戶外部測試集。
5. 因此模型目前應視為研究原型，不應把這些指標解讀為實際田間診斷準確率。

## 6. 主要結果檔案

- [探針與 TTA 指標](./probe_results.json)
- [五次重切分增強比較](./aug_resplit.json)
- [已保存的 MobileNet 微調指標](./finetune_results.json)
- [已保存的 red-team 指標](./redteam_core.json)
- [探針混淆矩陣](./confusion_probe.png)
- [模型與評估程式](../src/train_probe.py)、[增強重切分程式](../src/eval_aug_resplit.py)
