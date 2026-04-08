---
type: paper
node_id: paper:zuo2024_soke
title: "Signs as Tokens: A Retrieval-Enhanced Multilingual Sign Language Generator"
authors: ["Zuo et al."]
year: 2024
venue: ICCV 2025
external_ids:
  arxiv: "2411.17799"
  doi: null
  s2: null
tags: [sign-language-generation, vq-tokenizer, autoregressive, multilingual, retrieval, body-part-tokenization]
relevance: core
origin_skill: research-wiki
created_at: 2026-04-08T00:00:00Z
updated_at: 2026-04-08T00:00:00Z
---

# One-line thesis

SOKE discretizes multilingual sign language into per-body-part VQ tokens and generates them autoregressively via a pretrained multilingual LM augmented with sign dictionary retrieval.

## Problem / Gap

Existing sign language generation approaches lack a unified multilingual framework; prior VQ-based tokenizers flatten all body-part tokens into a single sequence, causing inefficiency and poor cross-part fusion.

## Method

- **Decoupled tokenizer**: separate VQ codebooks per body part (body, left hand, right hand)
- **Multi-head decoding**: predicts multiple body-part tokens simultaneously rather than one at a time — improves inference speed and cross-part fusion
- **Retrieval-Enhanced Generation**: conditions on external sign dictionaries to inject accurate word-level signs as auxiliary context
- Backbone: pretrained multilingual LM (mBART-style) with extended sign token vocabulary
- Supports: American SL, Chinese SL (CSL-Daily), German SL (PHOENIX)

## Key Results

State-of-the-art on multilingual SLG benchmarks (ICCV 2025). Multi-head decoding improves both quality and inference efficiency over flattened-sequence baselines.

## Assumptions

Retrieval-enhanced generation assumes access to a sign dictionary at inference time. Quality depends on dictionary coverage.

## Limitations / Failure Modes

- mBART + multi-head lm head is not a unified decoder-only framework — separate heads per body part
- Retrieval quality is a bottleneck when dictionary coverage is low or the gloss alignment is poor
- Not a true AR continuation model — multi-head decoding breaks the strict token-by-token causal structure

## Reusable Ingredients

- Per-body-part decoupled VQ tokenizer design pattern (body / lhand / rhand)
- Multi-head decoding idea for simultaneous body-part token prediction
- CSL-Daily and PHOENIX as multilingual benchmarks

## Open Questions

- How does multi-head decoding compare to interleaved token ordering?
- Can retrieval augmentation be replaced by stronger in-context learning?

## Claims

claim:C1

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project

This is the **direct codebase ancestor**. The current repo (`Qwen` branch) was forked from SOKE and retains its data modules, training framework (Lightning), and VQ tokenizer architecture. Key divergences: (1) current work replaces mBART+multi-head with Qwen/mT5, (2) replaces VQ with LFQ, (3) adds per-quantizer-layer interleaved token ordering rather than multi-head.
