---
node_id: paper:sincan2025_unbiased_eval
title: "Gloss-Free Sign Language Translation: An Unbiased Evaluation of Progress in the Field"
authors: ["Ozge Mercanoglu Sincan", "Jian He", "Low Sobhan Asasi", "Richard Bowden"]
year: 2025
venue: "International Journal of Computer Vision"
arxiv: "2603.13240"
tags: ["sign-language-translation", "evaluation", "benchmark", "calibration", "critical"]
---

## One-line thesis

Under controlled, fair re-implementation with identical preprocessing/encoders, published contrastive SLT gains shrink to +0.4–1.0 B@4, and the field's apparent progress is largely due to evaluation inconsistencies.

## Problem / Gap

Different papers use different preprocessing, video encoders, and data splits, making BLEU comparisons unreliable. A model that "achieves 16 B@4" on CSL-Daily in one paper may use a different evaluation setup than one "achieving 11 B@4" in another.

## Method

Re-implements GFSLT-VLP, SignCL, Sign2GPT, FLa-LLM, and C2RL under a single controlled evaluation pipeline:
- Identical visual encoder (S3D pretrained on Kinetics)
- Identical tokenizer and BPE vocabulary
- Identical data split and preprocessing

## Key Results (reproduced, controlled conditions)

**PHOENIX14T:**

| Method | B@4 ± std |
|---|---|
| GFSLT-VLP | 21.97 ± 0.27 |
| + SignCL | 22.35 ± 0.28 |
| + CiCO | 22.95 ± 0.63 |

**CSL-Daily:**

| Method | B@4 ± std |
|---|---|
| GFSLT-VLP | 10.41 ± 0.17 |
| + SignCL | 11.22 ± 0.18 |
| + CiCO | 12.03 ± 0.06 |

Contrastive learning contributes **+0.4–0.8 B@4** on PHOENIX, **+0.8–1.6 B@4** on CSL-Daily — not the +5 B@4 often reported.

## Limitations

- Only covers methods up to early 2025. Newer pseudo-gloss methods (Guo 2025) not included.
- All methods use the same S3D encoder — encoder-specific effects not studied.

## Reusable Ingredients

- **Calibration for our M2 result**: Our best B@4=1.78 (contrastive) vs. 1.74 (baseline) is consistent with the field's real contrastive gains of +0.4–0.8 B@4 under controlled conditions. Our result is NOT unusual — it's the honest baseline.
- **For reporting**: We should report results with multiple seeds and confidence intervals. Single-run BLEU differences of <1 point are within noise.
- **CSL-Daily ceiling**: under this evaluation, best CSL-Daily B@4 is ~12 with CiCO. Our B@4~1.7 means we are far from even this baseline — suggesting a more fundamental problem (not contrastive pretraining).

## Open Questions

- What is the CSL-Daily ceiling for VAE-embedding-based (non-video) methods?
- Is the gap between our ~1.7 B@4 and the field's ~12 B@4 due to: (a) video encoder vs. VAE embeddings, (b) data preprocessing, (c) model architecture, or (d) LoRA rank/training recipe?
