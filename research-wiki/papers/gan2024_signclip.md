---
type: paper
node_id: paper:gan2024_signclip
title: "SignCLIP: Connecting Text and Sign Language by Contrastive Learning"
authors: ["Roshan Sharma", "et al."]
year: 2024
venue: arXiv
external_ids:
  arxiv: "2407.01264"
  doi: null
  s2: null
tags: [contrastive-learning, sign-language, text-alignment, clip, embedding-alignment]
relevance: related
origin_skill: research-wiki
created_at: 2026-04-08T00:00:00Z
updated_at: 2026-04-08T00:00:00Z
---

# One-line thesis

SignCLIP applies CLIP-style contrastive pre-training to align sign language video embeddings with text sentence embeddings in a shared latent space.

## Problem / Gap

Sign language embeddings and text representations live in separate spaces with no principled alignment; retrieval and translation require this alignment.

## Method

- Contrastive pre-training: sign video encoder + text encoder → shared embedding space
- Loss: InfoNCE between sign and text embeddings
- Large-scale multilingual training

## Key Results

Strong sign-text retrieval and improved downstream translation performance. Demonstrates that contrastive alignment significantly helps sign-text grounding.

## Limitations / Failure Modes

- Works at the sentence level; doesn't address token-level or frame-level alignment
- Video-based, not pose-based — not directly applicable to our 3D pose VAE pipeline

## Reusable Ingredients

- Contrastive sign-text pre-training as a pre-training stage before LM fine-tuning
- The idea of aligning sign embeddings to text space before using them in a language model

## Open Questions

- Could a contrastive pre-training stage on VAE embeddings (not video) help align our continuous embeddings to text space before mT5/Qwen fine-tuning?

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project

**Directly addresses gap:G4.** If our frozen VAE embeddings were contrastively aligned with text before being fed to mT5/Qwen, the downstream LM might find it easier to bind sign semantics to text output. This is a concrete hypothesis to test next.
