---
node_id: paper:zhou2023_gfslt_vlp
title: "Gloss-free Sign Language Translation: Improving from Visual-Language Pretraining"
authors: ["Benjia Zhou", "Zhigang Chen", "Albert Clapés", "Jun Wan", "Yanyan Liang", "Sergio Escalera", "Zhen Lei", "Du Zhang"]
year: 2023
venue: ICCV 2023
arxiv: "2307.14768"
tags: ["sign-language-translation", "gloss-free", "visual-language-pretraining", "contrastive", "foundational"]
---

## One-line thesis

The dominant gloss-free SLT paradigm: first align visual and text modalities via CLIP-style masked self-supervised pretraining, then fine-tune an encoder-decoder (mBERT/mT5) for translation.

## Problem / Gap

Gloss-free SLT was far behind gloss-based methods. The gap was attributed to lack of intermediate semantic anchors (glosses). This work proposes using visual-language pretraining as a substitute.

## Method

Two-stage:
1. **Stage 1**: CLIP-style pretraining aligns sign video embeddings with text embeddings via InfoNCE on video-sentence pairs. Masked self-supervised learning on sign frames.
2. **Stage 2**: Encoder-decoder fine-tuning for SLT. Initialized from Stage 1 pretrained weights.

Architecture: S3D visual encoder → projection → mBERT decoder.

## Key Results

- PHOENIX14T: best gloss-free at the time (~21 B@4 by later unbiased re-evaluation)
- CSL-Daily: ~10-11 B@4 (under controlled evaluation per Sincan et al. 2025)

Note: Original reported numbers inflated by preprocessing/tuning differences; unbiased re-eval (arXiv:2603.13240) shows 21.97 ± 0.27 on PHOENIX14T, 10.41 ± 0.17 on CSL-Daily.

## Limitations

- Requires video frames; pretraining stage is expensive.
- Absolute scores still far below gloss-based methods (~27+ B@4 with glosses).
- Stage 1 → Stage 2 gap: pretraining alignment does NOT guarantee downstream translation quality.

## Reusable Ingredients

- GFSLT-VLP is the **standard baseline** that all later work (SignCL, hierarchical) builds on.
- The two-stage paradigm (align → translate) is the dominant recipe — our M2 experiment followed this exactly.
- Key lesson: **even the best two-stage approach plateaus ~21-22 B@4 on PHOENIX**; our runs at ~1 B@4 suggest a more fundamental problem (data scale, embedding quality, or loss).

## Open Questions

- What exactly is being aligned in Stage 1? Retrieval accuracy ≠ translation quality.
- Is mBERT or mT5 the bottleneck in Stage 2?
