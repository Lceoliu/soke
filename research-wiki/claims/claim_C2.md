---
type: claim
node_id: claim:C2
title: "Better tokenizer reconstruction does not automatically improve downstream LM generalization"
status: supported
created_at: 2026-04-08T00:00:00Z
updated_at: 2026-04-08T00:00:00Z
tags: [tokenizer, lm, generalization, reconstruction]
---

# Claim

Improving VAE reconstruction quality (as measured by human-recognizability of reconstructed signs) does not reliably transfer to improved sign-language-to-text or text-to-sign LM performance at full-train scale.

## Evidence

- exp:001 (reconstruction improved), exp:003 (Qwen full train failed), exp:004 (mT5 motion shuffle, model does use embeddings), exp:005 (embeddings are discriminative)
- Consistent with paper:li2025_unisign insight: pre-training / fine-tuning objective alignment matters more than raw embedding quality

## Implication

Optimizing tokenizer reconstruction is necessary but not sufficient. The downstream training objective, data volume, and model inductive bias are likely equally or more important.
