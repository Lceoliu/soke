---
node_id: paper:motionmillion2024
title: "MotionMillion: A Large-Scale Motion Dataset with 2D-LFQ Motion Tokenizer"
authors: ["Anonymous (under review)"]
year: 2024
venue: "arXiv preprint"
arxiv: "2410.03311"
tags: ["dataset", "motion-tokenizer", "LFQ", "2D-LFQ", "autoregressive"]
---

## One-line thesis

Introduces 2D-LFQ (Lookup-Free Quantization treating motion as a 2D image T×D) as a tokenizer for SMPL-D135 body motion, achieving 16K codebook capacity vs. 2K-4K for standard VQ.

## Motion representation

**SMPL-D135**: 135-dim structured feature per frame:
- Root: 9D (translation + velocity)
- Body: 6D rotation per joint (converted from axis-angle)
- Hands: 6D rotation per hand joint

**Key**: Uses **6D rotation representation** (not raw axis-angle) to avoid periodicity. This is standard practice.

## 2D-LFQ tokenizer

Treats motion sequence as 2D image M ∈ ℝ^(T × 135), applies 2D convolution (not 1D) + LFQ quantization:
- Temporal downsampling factor α=4
- Codebook: 2^16 = 16,384 tokens (vs. 512-4K for standard VQ)
- Binary sign-based thresholding

## Relevance to our project

1. **6D rotation conversion is confirmed standard**: before any neural processing of SMPL-X axis-angle, convert to 6D rotation. This applies to our VAE encoder redesign.
2. **2D-LFQ idea**: our current 1D-LFQ could be extended to 2D by treating [T, joint_dim] as a 2D image — analogous to our flat 133-dim input seen as T×133 2D structure.
3. **Codebook size**: 16K tokens is significantly larger than our current setup (512 per VAE). If we want richer discrete tokens, 2D-LFQ is worth exploring.
