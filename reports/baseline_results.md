# Baseline results: strawberry disease classifier (12 trainable classes)

Code: `src/extract_features.py`, `src/train_probe.py`, `src/finetune.py` (+ `src/common.py`, `run.sh`, `requirements.txt`). Raw numbers: `reports/probe_results.json`, `reports/finetune_results.json`. Splits are the unmodified `splits/manifest.csv` (train 2091 / val 451 / test 447). Classes 8, 9, 10 have no images: they cannot be predicted or evaluated; all numbers below are 12-class, closed-set.

## 1. Headline (test set, n=447, single split, single seed)
| Model | Accuracy | Macro-F1 (12 cls) | Macro-F1 95% CI* |
|---|---|---|---|
| **DINOv2-S/14 frozen + LogReg (conflict rows dropped from train), T-scaled** | 0.879 | **0.677** | 0.578 - 0.748 |
| same, + flip-TTA | 0.886 | 0.680 | n/a |
| same, conflict rows kept in train | 0.839 | 0.603 | 0.507 - 0.675 |
| MobileNetV3-L two-stage fine-tune (conflict rows dropped) | 0.779 | 0.514 | 0.452 - 0.573 |
| Control: image metadata only (w, h, aspect, file size, is-png) + LogReg | 0.613 | 0.270 | - |
| Control: border-pixel ring only + LogReg | 0.532 | 0.214 | - |
*Percentile bootstrap, 2000 resamples, stratified by class (resamples images within class). It captures image sampling noise only, not split/seed variance, and not group correlation.
Majority-class (Leaf Scorch) baseline accuracy is 0.497. Macro-F1 averaged over 12 classes that have test support.

Protocol: train on train split only; C chosen on val macro-F1 (grid 1e-4..1, standardised features, class_weight=balanced); temperature fitted on val logits (T=0.94 for the main model); test touched once per variant. Which variant is "main" was picked by val macro-F1 (drop_conflicts 0.666 vs keep 0.647). An MLP head was not run (sklearn MLP has no class weights; LR was judged sufficient).

## 2. Label-conflict handling
79 train-only images have `label_conflict=1` (same picture under 2+ classes). Dropping them from train: test macro-F1 0.603 -> 0.677 (+0.074), accuracy 0.839 -> 0.879; val macro-F1 only 0.647 -> 0.666. The test difference is larger than the val difference and is mostly in small classes (e.g. Anthracnose recall 0.36 -> 0.73, precision 0.44 -> 0.80; I did not analyse which classes the conflict rows came from per class beyond the audit). With 2-20 test images per class, this gap is within the bootstrap CIs (they overlap) and one split; treat as "dropping is not worse, probably better". Conflict rows are never in val/test, so evaluation is not contaminated. Cost: fewer training images for already tiny classes 2, 5, 14, 15.

## 3. Per-class (main probe, test)
| ID | Class | Precision | Recall | Test n |
|---|---|---|---|---|
| 1 | Fusarium Wilt | 0.43 | 0.75 | 4 |
| 2 | Macrophomina Crown Rot | 1.00 | 0.50 | 2 |
| 3 | Gray Mold | 0.93 | 0.94 | 71 |
| 4 | Powdery mildew | 0.97 | 0.93 | 60 |
| 5 | Phytophthora Crown rot | 0.40 | 0.60 | 10 |
| 6 | Verticillium Wilt | 0.33 | **0.09** | 11 |
| 7 | Anthracnose Leaf Spot | 0.80 | 0.73 | 11 |
| 11 | Leaf Scorch | 0.98 | 0.96 | 222 |
| 12 | Neopestalotiopsis | 0.53 | 0.60 | 15 |
| 13 | Rhizopus Fruit Rot | 0.54 | 0.78 | 9 |
| 14 | Foliar nematode | 0.48 | 0.55 | 20 |
| 15 | N. Root-Knot Nematode | 1.00 | 0.92 | 12 |
Confusion matrix: `reports/confusion_probe.png`. Classes with test n <= 11 (1, 2, 5, 6, 7, 13) have recall resolution of 9-50 points per image; class 2 has 2 test images, so its recall is 0, 0.5 or 1. The strong classes are exactly the large, single-source ones (3, 4, 11, and 15).

## 4. Calibration and abstention (main probe)
Temperature scaling reduced test ECE only 0.058 -> 0.042 (for the keep-conflicts variant 0.089 -> 0.015). Abstain if max-softmax < tau OR top-2 margin < m; (tau, m) picked on val to maximise coverage subject to val selective accuracy >= target (min 30% coverage). Risk-coverage plot: `reports/risk_coverage.png`.

| Target | tau / margin | Val cov / sel-acc | **Test coverage** | **Test acc on answered** | Root/crown* abstained | Root/crown sel-acc | Other classes abstained |
|---|---|---|---|---|---|---|---|
| 0.95 | 0.75 / 0.60 | 0.76 / 0.950 | 0.745 | 0.952 | **62.7%** (37/59) | 0.77 | 19.8% |
| 0.90 | 0.46 / 0.20 | 0.92 / 0.902 | 0.899 | 0.920 | 32.2% | 0.68 | 6.7% |
*Root/crown set used = classes 1, 2, 5, 6, 14, 15 (Fusarium, Macrophomina, Phytophthora, Verticillium, foliar nematode, root-knot nematode); n=59 test images. Foliar nematode is a leaf/foliage disease; including it is the brief's wording, not my botanical judgement. Per-class abstain counts are in `probe_results.json`. At target 0.95 the model abstains on 8/10 Phytophthora, 9/11 Verticillium and 15/20 foliar nematode, which is the desired "ask for a better photo" behaviour, but answered root/crown cases are still only 77% correct (target not met on that subgroup). Thresholds were tuned on a 451-image val set and transferred to test with little loss, but the stricter 0.95 target is a noisy estimate (~0.95 +/- 0.02).

## 5. Fine-tune (ran)
MobileNetV3-Large (ImageNet V2 weights), train minus conflict rows, class-balanced sampler (1000 samples/epoch), label smoothing 0.1, RandomResizedCrop/flip/rotation/colour jitter/blur/erasing. Stage 1: head only, 2 epochs, lr 1e-3. Stage 2: unfreeze features[9:], lr 1e-4 (head 3e-4), cosine over 12 planned epochs. CPU (10 cores), ~150 s/epoch in stage 2; I set a 22-min budget, so training **stopped after 8 stage-2 epochs of the 12 planned (cosine schedule not completed)**; total runtime 25 min incl. image caching and evaluation. Best epoch picked by val macro-F1 (0.592, epoch 5); val macro-F1 then declined (0.51 at the last epoch), suggesting noisy val for small classes and/or overfitting. Test: acc 0.779, macro-F1 0.514 (CI 0.45-0.57), TTA 0.529. It is clearly worse than the frozen probe here; it is a single, shortened run, not tuned, so this does not prove fine-tuning is inferior in general. Weights: `models/mnv3_best.pt` (not the recommended model). Fine-tune per-class: Macrophomina recall 0/2, Verticillium 0.09.

## 6. Artifacts
`models/feats_plain.npy`, `feats_flip.npy` (float16, 768-d = CLS + mean patch tokens), `models/probe_best.joblib` (scaler, LR, T, classes), `models/mnv3_best.pt`. Portability: paths via `DATA_DIR`, `MANIFEST`, `OUT_DIR` env vars (manifest paths are remapped under DATA_DIR after the `JHB training dataset/` segment); device auto (cuda/cpu); disk-space check before writing; no resized images written to disk. The portable probe was re-run end-to-end locally after the fp16 change and reproduced the numbers above (test macro-F1 0.6767). The fine-tune numbers come from the pre-refactor script (only path/device changes since; not re-run on this machine; the CUDA path has not been tested at all).

## 7. Caveats
- Test set is tiny for 8 of 12 classes (n <= 20); six classes have n <= 11. Macro-F1 is dominated by their noise. CIs are wide (about +/- 0.08).
- **Source shortcut (auditor's flag) is real in these data**: metadata alone (size, file size, format) gets 61% accuracy / 0.27 macro-F1, and a border-pixel-only probe 53% / 0.21 - far above chance (1/12 = 0.08 for macro-F1-ish) and above the 0.50 majority baseline for metadata. Classes 3, 4, 11 are single-source with fixed resolution. The headline numbers therefore measure "recognise this web/Kaggle source" at least in part, and will not transfer to client field photos.
- Evaluation covers only 12 of 15 classes; the model cannot output 8, 9, 10 and will confidently misassign such images (no OOD detection tested).
- Near-duplicate grouping used Hamming <= 6 on pHash and dHash; re-crops may still leak across splits (audit's own caveat), inflating test.
- Label noise unadjudicated; test labels themselves may be wrong.

## 8. Self-critique: five ways these numbers may be inflated
1. **Source/style shortcut**: the same website/dataset appears in train and test for classes 3, 4, 11, 15; the metadata control shows 61% accuracy without looking at the lesion at all.
2. **Residual near-duplicate leakage**: group dedup misses re-crops/screenshots with padding; DINOv2 features are very good at matching them, so test items may have train twins.
3. **Dominant classes inflate accuracy**: Leaf Scorch + Gray Mold + Powdery mildew are 66% of the test set (accuracy 0.88 is mostly those); even macro-F1 gives 3 of 12 near-perfect classes.
4. **Selection on a small val + single split**: the variant (drop conflicts), C, and abstention thresholds were chosen using val (451 images, 2-12 per small class); the test-vs-val gap for the conflict comparison suggests luck is possible. No repeated group k-fold or multiple seeds were run.
5. **Abstention numbers are conditional on answered cases**: 95% selective accuracy is obtained by discarding 25% of the hardest test items, concentrated in the small, clinically important root/crown classes; high accuracy on the answered set says nothing about those classes. Also, test precision/recall for classes with n=2-4 are essentially anecdotes, and the bootstrap CI ignores split-level variance, so it is too narrow.
