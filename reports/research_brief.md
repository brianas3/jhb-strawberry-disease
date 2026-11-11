# Research Brief: Strawberry Disease Classifier (15 classes, CPU-only, imbalanced)

Date: 2026-11-09. Legend: [V] = fetched/seen this session; [M] = from memory, URL not fetched; [U] = could not verify.

## 1. Fine-tuning recipe (small, imbalanced, CPU)

**Recommended default (CPU-realistic): frozen-feature baseline first, then light fine-tune.**
1. Extract embeddings once (no gradients) from a strong pretrained backbone (DINOv2 ViT-S/14 or B/14, or CLIP ViT-B/16), cache them, train logistic regression / small MLP on top. Minutes on CPU, multiple seeds feasible. DINOv2: https://arxiv.org/abs/2304.07193 [M]. CLIP: https://arxiv.org/abs/2103.00020 [M].
2. Then fine-tune a small CNN on CPU: MobileNetV3 / EfficientNet-B0 at 224px. ConvNeXt-Tiny (28M params) and ViT-B fine-tuning is feasible only for a few epochs; treat as optional. Backbone papers [M]: MobileNetV3 https://arxiv.org/abs/1905.02244 ; EfficientNet https://arxiv.org/abs/1905.11946 ; ConvNeXt https://arxiv.org/abs/2201.03545.
3. Evidence on frozen DINOv2 vs fine-tuning for plant disease specifically: I found NO direct benchmark [U]. Related: plant-specific pretraining helps few-shot plant disease https://arxiv.org/pdf/2604.27538 [V, snippet only]; DINOv2 frozen-vs-finetune tutorial on a cotton-disease set (results not seen) https://debuggercafe.com/dinov2-for-image-classification-fine-tuning-vs-transfer-learning/ [V, snippet only]. So treat "linear probe is competitive" as a hypothesis to test, not a fact.

**Two-stage fine-tuning:** train the new head with backbone frozen, then unfreeze top blocks with a ~10x lower LR. Standard practice; no single authoritative source fetched [U]. Related long-tail result: decoupling representation learning from classifier re-balancing (cRT / tau-norm) works well, Kang et al. https://arxiv.org/abs/1910.09217 [M, not fetched this session].

**Imbalance handling (one class ~1480, others 15-480):**
- Class-balanced loss using effective number of samples, E_n = (1-b^n)/(1-b): Cui et al. https://arxiv.org/pdf/1901.05555 [V]. Simple alternative: sqrt-inverse-frequency class-weighted CE or class-balanced sampler. Do not use both full reweighting and full oversampling together (over-corrects) [opinion].
- Oversampling tiny classes with only flips duplicates ~15 images; use strong augmentation for them and report that the metric on them is high-variance.
- Label smoothing 0.1: [M] https://arxiv.org/abs/1906.02629 (helps calibration slightly; interacts with temperature scaling, fit T after training).

**Augmentation:** RandomResizedCrop (scale 0.5-1), flips, rotation, mild ColorJitter, optionally RandAugment/TrivialAugment. Avoid heavy color jitter (disease color is the signal). Mixup/CutMix optional; they can erase small lesions. TTA: average softmax over original + h-flip + a few crops (cheap on CPU; gains usually small) [opinion/M].

**CPU feasibility estimate [opinion, not benchmarked]:** ~3000 images, 224px. Embedding extraction with ViT-S/B: minutes to ~30 min. MobileNetV3 / EfficientNet-B0 fine-tune: roughly tens of minutes per few epochs on a laptop CPU. ConvNeXt-T/ViT-B full fine-tune: hours; avoid.

## 2. Calibration + abstention ("not sure -> ask for root/crown photos")

- Temperature scaling (single scalar T fit on held-out val by NLL): Guo et al. 2017 https://arxiv.org/abs/1706.04599 [M]. Used as a pre-step in the RAPS paper https://arxiv.org/pdf/2009.14193 [V].
- Max-softmax threshold (Hendrycks & Gimpel baseline) https://arxiv.org/abs/1610.02136 [M]; top-2 margin and entropy are alternative scores; all are monotone-ish in confidence and rarely differ much in practice [opinion].
- Conformal prediction gives a coverage guarantee: RAPS/APS sets, Angelopoulos et al. https://arxiv.org/pdf/2009.14193 [V]; code https://github.com/aangelopoulos/conformal_classification/ [V]. Caveat: needs a held-out calibration set with exchangeability; with 15-image classes the per-class guarantee is weak and web-image duplicates break exchangeability.

**Recommendation:** temperature scaling + max-softmax threshold, with abstain if p_max < tau OR top-2 margin small. Choose tau on the validation set from a risk-coverage curve (e.g. accuracy on answered cases >= target, report coverage). Report ECE and the risk-coverage curve. If time allows, add APS conformal sets as a "prediction set size > 1 => ask for photo" rule and show it as a secondary comparison. Also: add an explicit "uncertain / other" route for the 3 classes with zero training images; a closed-set softmax can never output them (see section 3).

## 3. Public strawberry disease data and licenses

Identify only; verify each license on the page before use; respect site terms/robots; do not bulk-download.

| Source | What it has | License (as seen) | URL |
|---|---|---|---|
| Mendeley "Strawberry Disease Detection Dataset with Hybrid Data Augmentation" | 8 classes incl. Angular Leafspot, Leaf Spot, Powdery Mildew (leaf/fruit), Gray Mold, Anthracnose, Blossom Blight; 5,610 imgs but only 2,459 are base images; 1,018 are DALL-E 3 synthetic (train only) | CC BY-NC 4.0 [V] (non-commercial; client JH Biotech is commercial, so ask the instructor/client before use) | https://data.mendeley.com/datasets/k2ptxmfjhj |
| Roboflow Universe "strawberry_disease" (Hunan Univ.) | Angular Leafspot, Leaf Spot, powdery mildew classes | CC BY 4.0 per search snippet [U: not opened] | https://universe.roboflow.com/hunan-university-ezvjj/strawberry_disease-9uxup |
| Roboflow Universe "Strawberry" (for-japs) | 7 classes incl. angular leafspot, leaf spot | "Public Domain" per snippet [U: not opened] | https://universe.roboflow.com/strawberry-for-japs/strawberry-qnt3p |
| PlantDoc | includes "Strawberry leaf" (healthy) class, in-the-wild web images | CC BY 4.0 [V, snippet] | https://github.com/pratikkayal/PlantDoc-Dataset |
| PlantVillage | strawberry healthy + leaf scorch only; lab background | One source states CC BY-SA 3.0; a CC0 claim is NOT supported [V/U]; check original | https://arxiv.org/abs/1511.08060 (paper) |
| UC IPM photos | angular leaf spot, common leaf spot, leaf blotch pages | Photos credited (e.g. Jack Kelly Clark); some individual images CC0; reuse terms differ per image [V, partial]. Treat as "ask permission" | https://ipm.ucanr.edu/PMG/GARDEN/FRUIT/DISEASE/strawberryanugularleafspot.html ; https://ipm.ucanr.edu/PMG/GARDEN/FRUIT/DISEASE/strleafspot.html ; https://ipm.ucanr.edu/PMG/GARDEN/FRUIT/DISEASE/ID/idleafspot.html |
| NC State Extension | angular leafspot, Gnomonia comari leaf blotch pages with photos | License not checked [U] | https://content.ces.ncsu.edu/angular-leafspot-of-strawberry ; https://content.ces.ncsu.edu/gnomonia-comari-leaf-blotch-of-strawberry |
| UF/IFAS EDIS PP359 | leaf blotch / Pestalotia / leaf spot guide | License not checked [U] | https://edis.ifas.ufl.edu/publication/PP359 |
| Penn State Extension | strawberry disease ID | License not checked [U] | https://extension.psu.edu/strawberry-disease-identification |

**Gaps:**
- Angular leaf spot (Xanthomonas fragariae): the Mendeley and Roboflow sets above appear to have it (labels unverified; Roboflow user labels are often noisy).
- Common leaf spot (Mycosphaerella fragariae): the "Leaf Spot" class in those sets is likely this, but the label is not stated to be Mycosphaerella [U].
- Gnomonia leaf blotch: I found NO public image dataset [U]; only extension-page photos. Naming is also inconsistent (Gnomonia comari = Zythia fragariae; WA DPIRD uses Gnomoniopsis, a different, more serious fruit-rot disease): https://www.agric.wa.gov.au/strawberries/gnomoniopsis-fruit-rot-and-leaf-blotch-strawberries
- Practical options: (a) request permission from extension authors for a handful of reference photos; (b) treat the 3 classes as zero-shot only (LLaVA/OpenAI) and state clearly the CNN cannot predict them; (c) route to "uncertain" for ask-for-photo. Do not report macro-F1 over 15 classes as if all were learned.
- Differential diagnosis (useful for LLM prompts): early Gnomonia lesions resemble common leaf spot and Pestalotia (UF/IFAS PP359, NC State page above); angular leaf spot is often confused with common leaf spot and leaf scorch (UC IPM).

## 4. Evaluation protocol

- Primary: macro-F1 over classes that have test support; also per-class precision/recall/support table and a confusion matrix (row-normalized + raw counts). Report micro/accuracy only as secondary since one class is ~1480 imgs. (sklearn metrics; standard practice.)
- Classes with <~10 test images: report counts, not percentages alone. Use stratified, group-aware splits (see section 5), and repeated stratified group k-fold (e.g. 5 folds x 3 seeds) instead of one split.
- Bootstrap CIs: resample test images (stratified by class) 1000-2000 times; report percentile 95% CI for macro-F1 and per-class recall. For tiny classes, use Wilson/Clopper-Pearson intervals; expect very wide CIs and say so. [standard methodology; no URL fetched]
- Abstention metrics: coverage, selective accuracy, risk-coverage curve, ECE.
- Fair zero-shot comparison: identical test images, identical label list (same names, plus an "uncertain/other" option for all systems), identical abstention/mapping rule, no few-shot examples in prompts, temperature 0, fixed prompt, report parse-failure rate. Report both on the 12 trained classes and on all 15. Include the 3 zero-image classes only for zero-shot models and mark the CNN as N/A there. Run LLM models on the test split only (cost/CPU time), with CI on the same bootstrap resamples (paired bootstrap for differences).
- LLaVA: llava-hf/llava-1.5-7b-hf (7.06B params, BF16; prompt format "USER: <image>\n... ASSISTANT:") https://huggingface.co/llava-hf/llava-1.5-7b-hf [V]. Third-party estimate ~15 GB VRAM FP16, ~3.8 GB at INT4 [V, third-party]; CPU inference is possible but slow (inference, not benchmarked [U]); official snippets use GPU. Consider a 4-bit GGUF build via Ollama (`ollama` llava tag) for CPU [U: not verified]. LLaVA-NeXT (llava-hf/llava-v1.6-mistral-7b-hf) not verified this session [U].
- OpenAI vision + JSON schema: Responses API, `text.format = {type:"json_schema", name, schema, strict:true}`; strict mode needs `additionalProperties:false` and all properties in `required`; check for a `refusal` item and `status:"incomplete"` before parsing. Source: https://developers.openai.com/api/docs/guides/structured-outputs [V, first 100k chars only]. Chat Completions equivalent: `response_format`. Use an enum field for the label so output is constrained to the 15 names + "uncertain".
- OpenAI model ids: the models page I fetched (https://developers.openai.com/api/docs/models) listed `gpt-6-astra`, `gpt-6.1-sol`, `gpt-6.1-terra`, `gpt-6-luna` as flagship, all accepting image input, and the structured-outputs page recommends `gpt-6-astra`. I cannot cross-check these names against my own knowledge (my knowledge ends June 2026) and the fetch went through a summarizer, so [U]: before coding, open the page yourself or call `client.models.list()` and pin the exact id plus snapshot/date in the report. A community report of empty output with gpt-5-mini on some images + schema output: https://community.openai.com/t/gpt-5-mini-returns-empty-output-text-on-certain-images/1373907 [V, snippet].
- Record model id, date, prompt, schema, seed/params for every zero-shot run (models change).

## 5. Pitfalls

- Background/capture shortcuts: a model trained on only 8 background pixels of PlantVillage images reached 49.0% test accuracy vs 2.6% chance: Noyan, https://arxiv.org/pdf/2206.04374 [V]. Web-collected images are worse: each class likely comes from different sites/searches, so watermark, resolution, JPEG quality, background and style can become the label. Mitigations: random crops, leaf-focused cropping/segmentation, grayscale/color-stat sanity checks, "background-only" or "border-only" probe as a control (if it scores well, you have a shortcut), Grad-CAM spot checks, and leave-source-out validation (group by source domain/URL).
- Duplicates/leakage: web scrapes contain exact and near-duplicates (resizes, crops, re-posts) across classes and splits. Dedupe with perceptual hash (pHash/dHash) plus embedding cosine similarity before splitting; split by duplicate-cluster/source group, never by random image. Do this before any augmentation or oversampling (oversample only the train fold). Public sets above also overlap each other (the Mendeley set extends earlier public sets; its synthetic DALL-E images are train-only, so exclude synthetic images from evaluation).
- Field vs lab: PlantVillage-style lab images (uniform background) transfer poorly to field photos; PlantDoc exists specifically because of this gap (https://github.com/pratikkayal/PlantDoc-Dataset [V]). Keep a field-image test set and report per-domain results; do not tune thresholds on the final test set.
- Label noise: web and Roboflow user labels are often wrong; manually audit tiny classes and high-confidence errors.
- Tiny classes: 15-image classes mean a single image changes recall by ~7 points; do not over-interpret.
- Licensing: CC BY-NC data in a client (commercial) deliverable is a risk; keep a data-provenance table.

## Not verified / limitations
- No direct DINOv2-vs-fine-tune benchmark on plant disease found.
- Roboflow license claims come from search snippets only.
- OpenAI model ids, LLaVA-NeXT id, Ollama CPU speed: unverified.
- Several arXiv URLs marked [M] were not fetched this session.
