# JH Biotech Strawberry Disease Dataset - Data Audit

Dataset (read-only): `JHB training dataset`. Scripts: `src/audit.py`, `src/make_splits.py`. Seed 42. Machine-readable outputs: `reports/image_table.csv`, `per_class.csv`, `dup_groups.csv`, `audit_summary.json`, `splits/manifest.csv`.

## 1. Headline
- 2,989 images in 12 non-empty classes. Classes 8, 9, 10 are empty (0 images).
- 0 corrupt/unreadable files. Only non-image file is `.DS_Store` (ignored). Formats: 2,951 jpg, 37 png, 1 gif (static). Modes: 2,962 RGB, 20 RGBA, 7 P (all converted to RGB for hashing; the training loader must do the same).
- 0 images below 128 px on the shorter side (min is 128 px, class 11), but several are small (<224 px: classes 5, 6, 12, 13, 14, 15).
- 9 files have an extension that does not match the real format (e.g. `.png` actually JPEG); harmless with PIL, listed in `audit_summary.json`.
- Duplicates: 87 exact (md5) groups = 89 redundant files. Including perceptual near-duplicates (pHash AND dHash Hamming <= 6), 230 images (7.7%) sit in 104 multi-image groups; 2,863 independent groups remain.
- **Cross-class duplicates (label noise): 36 groups / 79 images** carry two or more different class labels.

## 2. Per-class table
| ID | Class | N | Groups | Img in dup grp | Img in cross-class grp | Size (w) min/med/max | <128px |
|---|---|---|---|---|---|---|---|
| 1 | Fusarium Wilt | 30 | 30 | 4 | 4 | 400/1271/4279 | 0 |
| 2 | Macrophomina Crown Rot | 15 | 15 | 8 | 8 | 372/607/3713 | 0 |
| 3 | Gray Mold | 477 | 470 | 17 | 3 | 419/419/419 | 0 |
| 4 | Powdery mildew | 399 | 398 | 2 | 0 | 419/419/419 | 0 |
| 5 | Phytophthora Crown rot | 67 | 65 | 15 | 11 | 177/640/2500 | 0 |
| 6 | Verticillium Wilt | 76 | 37 | 67 | 8 | 220/800/2560 | 0 |
| 7 | Anthracnose Leaf Spot | 75 | 40 | 66 | 2 | 482/1000/2252 | 0 |
| 8 | Angular Leaf Spot | 0 | - | - | - | - | - |
| 9 | Common Leaf Spot | 0 | - | - | - | - | - |
| 10 | Gnomonia leaf blotch | 0 | - | - | - | - | - |
| 11 | Leaf Scorch | 1480 | 1478 | 4 | 0 | 128/256/1024 | 0 |
| 12 | Neopestalotiopsis Leaf Spot | 98 | 96 | 11 | 7 | 199/746/4164 | 0 |
| 13 | Rhizopus Fruit Rot | 58 | 58 | 3 | 3 | 226/783/2756 | 0 |
| 14 | Foliar nematode | 134 | 134 | 22 | 22 | 220/766/3264 | 0 |
| 15 | Northern Root-Knot Nematode | 80 | 80 | 11 | 11 | 250/763/3264 | 0 |

Imbalance: class 11 is 49.5% of all images; largest:smallest (non-empty) = 99:1.

## 3. Cross-class duplicates (top label-noise pairs)
Classes (14,15): 10 groups; (2,5): 7; (12,14): 5; (13,14): 3; (1,6): 3; (3,14): 2; others singletons. Foliar nematode (14) is involved in 22/134 images. These are the same picture filed under two diseases. At least some are genuine ambiguity (nematode vs. crown-rot symptoms; web pages showing several diseases), but a label cannot be right in both. Full list: `reports/dup_groups.csv` (filter `group_n_classes > 1`). **I have not visually adjudicated which label is correct.**

## 4. Key risks
1. **Source/shortcut bias (highest).** Classes 3 and 4 are 100% 419x419 images with uniform filenames (`gray*`, `powdery*`); class 11 is 75% 256x256, from `kgl*` (1,152, apparently Kaggle) and `inat*` (328, iNaturalist). Other classes are mixed-size web scrapes (often article screenshots with captions). A model can learn "image size/source style = class" instead of disease. Expect inflated val/test accuracy that will not transfer to client field photos. (Source inference from filenames/sizes; not verified.)
2. **Tiny effective sample size.** Class 2 has 15 images (8 in dup/cross groups); class 1 has 30. Classes 6 and 7 look like 76/75 images but contain only 37/40 independent groups (heavy near-duplication, filenames `image*`), so the real information is about half.
3. **Test set noise.** Val/test for small classes are 2-5 images; accuracy per class is statistically meaningless (class 2: 2 test images).
4. **Label noise** from 36 cross-class groups (above), plus unchecked mislabels, watermarks, diagrams and non-strawberry content.
5. **Three of 15 classes have no data.**
6. Many web-captured images contain captions/UI text (filenames like "... _ Plantix.png", "Download Scientific Diagram"); models may read text/layout.

## 5. Splits (`splits/manifest.csv`)
Columns: path, class_id, class_name, group_id, split, label_conflict. 70/15/15 stratified per class, done at near-duplicate-group level (asserted: no group spans splits). Result: train 2,091 / val 451 / test 447.

- Small classes: for every class with >= 2 groups one group is forced into test; with >= 3 groups one is also forced into val (smallest groups chosen, so class 2 gets 2 test + 2 val, class 1 gets 4 test). All 12 classes have >= 1 test image.
- Cross-class duplicate groups (79 images) are placed in **train** and flagged `label_conflict=1`, because one copy cannot be in two classes' test sets. Recommend resolving or dropping them before training; they are excluded from val/test, which keeps evaluation clean.
- Splits are by group, not by exact class ratio, so actual proportions deviate slightly (see `reports/split_counts.csv`).
- Classes 8, 9, 10 have no rows.
- Limitation: grouping only catches visually near-identical images (Hamming <= 6 on both hashes). Re-crops, rescales with big padding, or screenshots of the same photo with different framing may escape and still leak. Not quantified.

## 6. Recommended handling
- **Empty classes 8/9/10:** cannot be trained or evaluated. Needs from client: >= 100 labelled field photos each (aim 200+ from multiple farms/cultivars/lighting; leaf close-ups and whole-plant), with expert confirmation of diagnosis. Alternative: ship a 12-class model plus "unknown/other" rejection, and state 3 classes are unsupported.
- **Class 2 (15 imgs) and class 1 (30):** do not promise per-class accuracy. Either collect more data or merge into a broader "Crown/wilt" group (2+5, and 1+6 are the pairs the data itself confuses), or keep with heavy augmentation and clearly flag as low-confidence.
- **Imbalance:** use class-weighted loss (inverse sqrt frequency) and/or capped sampling for class 11 (e.g. <= 400 per epoch); report macro-F1 and per-class recall, not overall accuracy.
- **Shortcut mitigation:** resize/center-crop all images to one resolution, random-resized-crop plus colour jitter and JPEG-compression augmentation; strip or crop caption areas; evaluate on an external, client-collected hold-out set.
- **Cleaning pass before training:** manual review of the 36 cross-class groups, the sample grids in `reports/samples/`, and a quick non-strawberry/diagram screen (not done here).
- Keep 1 image per exact-duplicate group only if training turns out to overweight them; splits already avoid leakage.

## 7. Open questions for the department head
1. Can the client supply real images for classes 8, 9, 10 and by when, or should the deliverable be a 12-class model with explicit scope limits?
2. Who is the domain expert able to adjudicate the 36 cross-class duplicate groups (especially nematode vs. crown rot/Fusarium)?
3. What will the model be used on (phone photos in the field vs. lab/web-quality images)? This decides whether the 419x419 and Kaggle-style classes are acceptable training sources.
4. Is it acceptable to merge or drop classes 1 and 2, or must all 15 remain as outputs even with unreliable accuracy?
5. Which metric defines success for the client (macro-F1, per-disease recall for the costly ones), and may we hold out a client-collected test set instead of web-scraped test images?
6. (Extra) Are there licence/copyright constraints on web-scraped images (scientific articles, Plantix, iNaturalist) for a delivered product?

## 8. Not verified
- No visual review of images beyond generating the sample grids (`reports/samples/class_XX.png`, 8 random 128px thumbnails per class); I did not inspect them for watermarks, mislabels or non-strawberry content. Corruption check only means PIL could decode the file.
- Source attribution (Kaggle/iNaturalist) is inferred from filename prefixes.
- The Hamming threshold of 6 on both hashes is the requested setting, not tuned; false-positive grouping of genuinely distinct images is possible, and misses are possible too.
