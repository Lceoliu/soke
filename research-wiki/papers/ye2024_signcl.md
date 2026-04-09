---
node_id: paper:ye2024_signcl
title: "Improving Gloss-free Sign Language Translation by Reducing Representation Density"
authors: ["Jinhui Ye", "Xing Wang", "Wenxiang Jiao", "Junwei Liang", "Hui Xiong"]
year: 2024
venue: NeurIPS 2024
arxiv: "2405.14312"
tags: ["sign-language-translation", "contrastive-learning", "representation-density", "gloss-free", "critical"]
---

## One-line thesis

Gloss-free sign language translation fails because contrastive learning without gloss supervision compresses semantically distinct gestures into an overly dense embedding space; margin-based frame-level contrastive loss (SignCL) mitigates this by explicitly separating adjacent vs. distant frames.

## Problem / Gap

Without glosses, sign video encoders produce representations where 92.59% of features are semantically similar (vs. 66.23% for gloss-based methods). This **representation density** makes it nearly impossible for downstream decoders to distinguish signs, capping BLEU regardless of LM quality.

## Method

- Treats temporally adjacent frames as positive pairs and distant frames as hard negatives in a margin-based contrastive loss.
- Added to existing backbone (GFSLT-VLP) as auxiliary pre-training objective.
- No glosses required. Loss applied directly to video encoder outputs.

## Key Results

| Dataset | Baseline (GFSLT-VLP) | +SignCL | Δ |
|---|---|---|---|
| CSL-Daily | 11.00 B@4 | **16.16 B@4** | +47% |
| PHOENIX-2014T | 10.73 B@4 | **13.51 B@4** | +26% |

## Limitations

- SignCL only reduces density; does not solve the fundamental grounding problem (absolute scores still low vs. gloss-based: 27+ B@4).
- Applied to video frames — not directly applicable to our VAE embedding space (pre-computed, not video).
- Requires video encoder; our pipeline skips video and uses pre-computed VAE embeddings.

## Reusable Ingredients

- **Representation density as diagnostic**: compute the fraction of near-identical embeddings in our sign VAE output before feeding to the LM — high density = failure mode.
- **Margin-based contrastive on embeddings**: can be applied to our 1536-dim sign embeddings across time steps within a sequence, not just frame pairs from video.
- **Key insight for our failure (M2)**: Our contrastive pre-train aligned *sequence-level* pooled embeddings to text — it did not address *within-sequence* density. The LM still sees a dense trajectory of near-identical tokens → can't decode.

## Open Questions

- Does representation density in the VAE embedding space correlate with LM loss plateau?
- Can we measure density before vs. after contrastive pre-training in our setup?
