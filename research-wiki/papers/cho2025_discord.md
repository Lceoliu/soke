---
type: paper
node_id: paper:cho2025_discord
title: "DisCoRD: Discrete Tokens to Continuous Motion via Rectified Flow Decoding"
authors: ["et al."]
year: 2025
venue: ICCV 2025
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: [discrete-tokens, continuous-motion, rectified-flow, decoding, motion-generation]
relevance: related
origin_skill: research-wiki
created_at: 2026-04-08T00:00:00Z
updated_at: 2026-04-08T00:00:00Z
---

# One-line thesis

DisCoRD bridges the gap between discrete LM-generated motion tokens and continuous motion by using a rectified flow decoder, recovering fine-grained details lost in discretization.

## Problem / Gap

Discrete tokenization loses fine-grained motion detail; naive detokenization (e.g., nearest-neighbor lookup) produces jerky or unnatural motion.

## Method

- LM generates discrete motion tokens autoregressively
- Rectified flow model decodes discrete tokens back to continuous motion, guided by the token sequence
- Recovers spatial detail and smoothness lost in VQ discretization

## Key Results

Significantly improved motion quality vs. hard VQ decode at ICCV 2025.

## Reusable Ingredients

- Two-stage pipeline: AR LM on discrete tokens → flow-based continuous decoder
- Addresses the "quantization bottleneck" from a different angle than continuous embedding approaches

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project

Relevant to gap:G1 — offers a different perspective: instead of making discrete tokens better, use a better decoder after LM generation. If we keep discrete tokens for LM but use a flow decoder instead of hard codebook lookup for t2m output, we might get better generation quality without changing the LM training.
