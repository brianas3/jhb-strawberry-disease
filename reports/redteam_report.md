# Red-team report (code: src/redteam_core.py, src/redteam_stress.py; raw: reports/redteam_*.json, redteam_neardup.csv, redteam_grids/)

## Checks done / skipped
Done: 1,2,3,4,5,6. Skipped: independent re-run of the metadata-only/border-only controls (not reproduced; taken from engineer), MobileNet claim, flip-TTA.

## 1. Reproduction and leakage
- Own code from cached feats: C=0.01 chosen on val, test acc 0.8792, macro-F1 0.6767 (matches). Keep-conflicts: 0.8389 / 0.6032 (matches). No test use in C/T/threshold selection (read train_probe.py; test only enters evaluate()).
- group_id overlap train/val/test: 0/0/0.
- Fresh pHash (8x8, own code) test-vs-train: 0 exact, 3 at <=6, 34 at <=10 (7.6% of test). 33/34 share the label, all 34 predicted correctly. Excluding them: acc 0.869, macro-F1 0.656 (-0.02). Val-vs-train <=10: 17. Mild leakage, not fatal, but the auditor's <=6 threshold missed 31 of these.
## 2. Source-stress (test n=447, probe frozen, features re-extracted)
| Variant | acc | mF1 | rec cls3 | rec cls4 | rec cls11 |
|---|---|---|---|---|---|
| identity (re-extracted, 100% agree with cache) | .879 | .677 | .944 | .933 | .959 |
| fixed 256x256 squash | .859 | .627 | .944 | .933 | .955 |
| low-res (64px then upsample) | .834 | .602 | .930 | .850 | .950 |
| JPEG q30 | .861 | .611 | .915 | .900 | .950 |
| grayscale | .848 | .623 | .859 | .883 | .946 |
| crop outer 15% | .866 | .675 | .887 | .950 | .937 |
Single-source classes degrade modestly (worst: cls4 -8 pts lowres, cls3 -8 pts grayscale). Macro-F1 loses up to 0.075 (mostly small, noisy classes). Not a pure shortcut model, but not invariant either. Caveat: classes 3/4 are all 419x419 so resize-to-fixed barely changes them; real cross-source robustness is untested.
## 3. Leave-source-out proxies
- Class 11 has 4 resolutions (256: 1109, 768x1024: 172, 1024x768: 65, 128: 43). Training without all non-256 class-11 images and testing on them: recall of class 11 = 0.02 (n=371). Within-distribution 256-wide test recall = 1.00. Strong evidence the class-11 notion is tied to the source/resolution cluster, though this is an extreme split (those images may be truly different content).
- GroupKFold on (class, width-bucket) clusters: acc 0.20, mF1 0.25 (classes 3, 4 are single-cluster so recall 0 by construction; uninformative for them). Plain 5-fold group CV: acc 0.862, mF1 0.606 (also lower than 0.677 headline).
## 4. Label conflicts
Conflict rows per class (train): 14:22, 5:11, 15:11, 2:8, 6:8, 12:7, 1:4, 3:3, 13:3, 7:2. So 79 rows fall on the small classes (2 loses 8/19 of its images, 5 loses 11 etc.).
5 random group-aware stratified resplits (eval excludes conflicts; train groups overlapping eval removed): macro-F1 drop 0.592+-0.036 vs keep 0.597+-0.022; per-seed diff (drop-keep) = +.007,-.037,-.012,+.040,-.024; accuracy .875 vs .867. **"Dropping conflicts is better" is REFUTED**: it is a wash. Also the resplit mean macro-F1 (~0.59) is ~0.085 below the headline 0.677, so the official split is on the favourable end (seed sd 0.036).
## 5. Visual QA (limited, non-expert; contact sheets in reports/redteam_grids/)
- Verticillium (10 viewed): 2 alamy watermarks, 1 text overlay "Verticillium Wilt", 1 gray-mold fruit photo, 1 apparent AI-style generated image, 1 leaf that may not be strawberry; symptoms often generic leaf scorch/browning (and Leaf Scorch is a competing class).
- Phytophthora (10): 1 alamy watermark; several are fruit leather-rot images that look like ordinary fruit; 2 near-identical images in different groups (likely dup missed by group_id).
- Macrophomina (7 of 15): captions explicitly say Colletotrichum/Anthracnose crown rot and "Macrophomina in chickpea" -> label noise and source-caption contamination.
- Foliar nematode (10): hosta, a willow-like leaf, Heuchera ground cover, plain strawberry fruit close-up, hand holding gray-mold berries: roughly half are non-diagnostic or non-strawberry.
- Root-knot (10): tomato, lettuce, beet/root crops, field row photo: mostly not strawberry. The 0.92 recall for class 15 probably reflects "bare root" appearance, not strawberry-specific diagnosis.
Not expert diagnosis; images downloaded from web pages with unreliable labels.
## 6. Abstention
Val-chosen tau=0.74, margin=0.6 (joint grid search on val, 560 combos, then tested once). Val: coverage 0.758, sel-acc 0.953. Test: coverage 0.749, sel-acc 0.952 (bootstrap 95% CI 0.928-0.974; coverage CI 0.709-0.790). Transfers well. But: 89.6% of answered images are classes 3/4/11; coverage on the other 9 classes is 0.37 with selective acc 0.71 (n=~100 answered... small). The headline 95% is carried by the three source-clustered classes. Engineer's quoted 74.5% vs my 74.9% (335/447) is a rounding/ typo-level difference.
## Verdict
| Claim | Status |
|---|---|
| acc 0.879 / macro-F1 0.677 reproducible | HOLDS (exact reproduction) |
| No leakage | WEAKENED: group clean; 34 test imgs within pHash<=10 of train (-0.02 mF1 if removed); 4 conflict-free but suspicious near dups in class 5 |
| 0.677 is a representative estimate | WEAKENED: resplit mean 0.59+-0.04, group-CV 0.606; test per-class n=2-20 |
| Dropping conflicts improves macro-F1 | REFUTED (resplits: -0.005 on average, sign flips across seeds) |
| Abstention 95% at ~74.5% coverage | HOLDS on this split (cov 0.749, acc 0.952, CI .93-.97) but WEAKENED as a product claim: rare classes only 37% coverage |
| Source shortcut controls (61% / 53%) | NOT RE-RUN; consistent with my stress/LOSO results that source matters (cls11 off-256 recall 0.02) |
| Classifier is robust to source cues | WEAKENED: up to -8 pts per-class recall, -0.075 mF1 under benign corruption |
## Top 5 fixes
1. Relabel/clean the rare classes (Macrophomina, Verticillium, both nematodes): remove non-strawberry, watermarked, mixed-disease images; get expert-verified field photos from JH Biotech.
2. Report CV over many group-aware resplits (mean+-sd), not a single split; use macro-F1 ~0.6 as the honest number.
3. Tighten dedup to pHash<=10 plus embedding-cosine near-dup grouping, then re-split.
4. Collect multi-source images for classes 3/4/11 (and hold out by source) before claiming deployability; add watermark/border cropping + resolution/JPEG/grayscale augmentation.
5. Drop the "dropping conflicts helps" claim; resolve conflicts by relabeling (multi-label for 14/15, 2/5) instead of discarding, and report abstention coverage per class group.
