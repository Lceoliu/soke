---
type: paper
node_id: paper:li2025_unisign
title: "Uni-Sign: Toward Unified Sign Language Understanding at Scale"
authors: ["Zecheng Li", "et al."]
year: 2025
venue: ICLR 2025
external_ids:
  arxiv: "2501.15187"
  doi: null
  s2: null
tags: [sign-language-understanding, sign-language-translation, pretraining, seq2seq, pose-rgb-fusion, csl, unified]
relevance: core
origin_skill: research-wiki
created_at: 2026-04-08T00:00:00Z
updated_at: 2026-04-08T00:00:00Z
---

# One-line thesis

Uni-Sign eliminates the pre-training / fine-tuning gap for sign language understanding by adopting a large-scale generative pre-training strategy that unifies all downstream SLU tasks as sign language translation.

## Problem / Gap

Existing sign language pre-training methods suffer from a gap between pre-training objectives and fine-tuning tasks, leading to suboptimal transfer. Large-scale CSL data was unavailable.

## Method

- **CSL-News dataset**: 1,985 hours of Chinese Sign Language video with text annotations (enables large-scale pre-training)
- **Prior-Guided Fusion (PGF)** module: fuses pose keypoints and RGB features, using pose as prior to guide attention on RGB frames; handles keypoint inaccuracy
- **Score-aware sampling**: efficient sampling strategy to reduce redundancy
- Fine-tuning: all SLU tasks treated as SLT (sign language translation) — a seq2seq formulation
- Uses a transformer encoder-decoder (not decoder-only)

## Key Results

State-of-the-art on CSL-Daily, PHOENIX-2014T, and How2Sign at ICLR 2025. Demonstrates that unifying pre-training and fine-tuning objectives is key to strong SLU performance.

## Assumptions

Treating all SLU tasks as translation (generative, seq2seq) is a sufficient unification. Pose + RGB fusion is preferable to pose-only or RGB-only.

## Limitations / Failure Modes

- Focused on understanding (SLT, SLR), not generation (t2m / SLG)
- Requires large-scale labeled data (CSL-News) for pre-training — may not generalize without it
- Encoder-decoder is not a decoder-only AR model

## Reusable Ingredients

- CSL-News dataset as a source of large-scale pre-training data
- PGF module for pose-guided RGB feature extraction
- Seq2seq unification pattern: treating diverse tasks as SLT during fine-tuning
- Insight: pre-training and fine-tuning objective alignment is critical for generalization

## Open Questions

- Does the seq2seq generative pre-training insight transfer to decoder-only models?
- Would CSL-News pre-training help with the current full-train generalization failures?

## Claims

claim:C2

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project

Uni-Sign's insight — that **pre-training / fine-tuning objective alignment** is critical, and that seq2seq (encoder-decoder) is more stable than decoder-only for sign language understanding at current data scales — directly motivated the pivot from Qwen (decoder-only) to mT5 (seq2seq). Its CSL-Daily results are a key benchmark target.
