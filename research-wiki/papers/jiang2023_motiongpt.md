---
type: paper
node_id: paper:jiang2023_motiongpt
title: "MotionGPT: Human Motion as a Foreign Language"
authors: ["Biao Jiang", "Xin Chen", "et al."]
year: 2023
venue: NeurIPS 2023
external_ids:
  arxiv: "2306.14795"
  doi: null
  s2: null
tags: [motion-generation, vq-vae, llm, unified-modeling, t2m, m2t, motion-prediction]
relevance: core
origin_skill: research-wiki
created_at: 2026-04-08T00:00:00Z
updated_at: 2026-04-08T00:00:00Z
---

# One-line thesis

MotionGPT treats human motion as a foreign language, using a VQ-VAE motion vocabulary and instruction-tuned LLM to handle multiple motion-language tasks in a single unified model.

## Problem / Gap

Prior motion generation models are task-specific; no unified framework existed for text-to-motion, motion captioning, motion prediction, and motion in-between with a single model.

## Method

- VQ-VAE encodes raw motion into discrete motion tokens ("motion vocabulary")
- Motion tokens added to LLM vocabulary; LLM fine-tuned with instruction-following prompts
- Tasks handled: t2m, m2t (motion captioning), motion prediction, motion in-between
- Instruction tuning protocol enables flexible task switching

## Key Results

State-of-the-art on multiple motion tasks at NeurIPS 2023. Demonstrates that a single LLM can handle diverse motion-language tasks simultaneously.

## Assumptions

Discrete VQ tokenization is sufficient for motion semantics. Human body motion (HumanML3D / KIT) is representative enough for general motion understanding.

## Limitations / Failure Modes

- Designed for general body motion (HumanML3D), not sign language — no hand articulation, no per-part codebooks
- VQ tokenization quality bounds downstream LM performance
- Single codebook; does not model body-part dependencies explicitly

## Reusable Ingredients

- Unified instruction-tuning framework for multiple motion tasks (t2m / m2t / mc / prediction)
- Motion token vocabulary extension pattern (add tokens to existing LLM vocab)
- Training / evaluation protocols for motion-language unified models

## Open Questions

- Does the unified LLM benefit from each task, or do tasks interfere?
- How well does the approach scale to more complex body models (hands, face)?

## Claims

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project

The **training framework skeleton** (Lightning trainer, data module structure, t2m/m2t/mc task split) was inherited from MotionGPT. The core idea of treating sign tokens as a "new language" added to an LLM vocabulary directly follows MotionGPT's paradigm. Current work extends this to sign language with per-body-part LFQ tokenization.
