---
node_id: paper:ye2024_dartcontrol
title: "DartControl: A Diffusion-based Autoregressive Motion Primitive Model for Real-time Text-driven Motion Control"
authors: ["Haonan Ye", "Jiawei Shao", "Zhibo Wang"]
year: 2024
venue: ICLR 2025
arxiv: "2410.05260"
tags: ["autoregressive", "motion-generation", "smplx", "whole-body", "diffusion", "real-time"]
---

## One-line thesis

Autoregressive motion generation with SMPL-X whole-body by chunking into overlapping H+F primitives; each primitive compressed by a transformer VAE, then generated via latent diffusion conditioned on text and motion history.

## SMPL-X encoding approach

**Input**: 276-dim per frame (overparameterized: body root + local joint rotations + joint positions + temporal diffs).

**Encoder**: Two-stage VAE:
1. Motion Primitive VAE with **transformer encoder-decoder** (not GCN) operating on H=2 history + F=8 future frames
2. Latent diffusion on compressed primitives

**No graph network** — uses standard Transformer on flattened SMPL-X vectors.

## Autoregressive mechanism

Generate motion in overlapping chunks of F frames, conditioned on H history frames. Sequential generation → >300 fps on RTX 4090.

## Key insight for our project

- **DartControl pattern is relevant for sign generation (t2m)**: chunk into sign primitives, generate F frames autoregressively conditioned on text token + pose history. Natural for sign language where signs have discrete temporal boundaries.
- **Encoder is Transformer, not GCN** — confirms the field hasn't converged on GCN for SMPL-X axis-angle; transformer on flat vector is the default.
- **Overparameterization (276D)**: includes positions + rotations + diffs simultaneously, giving the model redundant but complementary cues.
