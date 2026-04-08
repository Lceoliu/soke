---
type: idea
node_id: idea:002
title: "Qwen decoder-only unified sign language modeling (t2m + m2t + mc)"
stage: tested
outcome: negative
created_at: 2026-04-08T00:00:00Z
updated_at: 2026-04-08T00:00:00Z
tags: [decoder-only, qwen, unified-modeling, t2m, m2t, sign-language, lora]
---

# Summary

Treat sign tokens as a new language, add them to Qwen2.5's vocabulary, and train a unified decoder-only continuation model covering t2m, m2t, and motion continuation (mc).

## Hypothesis

A sufficiently capable decoder-only LLM (Qwen2.5) with sign tokens added to its vocabulary can learn unified sign language generation and translation from a single AR loss.

## Based On

paper:jiang2023_motiongpt (paradigm), paper:zuo2024_soke (tokenizer)

## Target Gaps

gap:G2, gap:G3

## Implementation

- `mGPT/archs/mgpt_qwen.py`
- `mGPT/models/mgpt.py`
- `mGPT/archs/task_formatting.py`
- Token ordering: time-chunk first, then quantizer-layer, then body part (interleaved)
- LoRA: rank=64, alpha=128, dropout=0.05; embed_tokens + lm_head trainable
- Configs: `configs/lm/qwen2_5_0_5b.yaml`, `configs/lm/qwen2_5_3b.yaml`, `configs/lm/qwen2_5_8b.yaml`

## Outcome

**Negative.** Overfit experiments (1/4/12 samples) succeeded — confirming pipeline correctness. Full train failed: train loss decreases normally, val ppl increases, test m2t output is semantically unrelated to GT. Model retains text language capability but fails to bind sign tokens to semantics at scale.

## Failure Notes

- Qwen2.5-0.5B and 3B both fail full train similarly
- `m2t_prefix_loss_weight=0.5` helps val ppl vs 0.0 but does not fix generalization
- Failure pattern: model learns text priors, not sign-to-text binding
- Possible causes: (1) insufficient data volume for emergent generalization in decoder-only, (2) sign tokens lack linguistic inductive bias, (3) task interference in multi-task AR training, (4) discrete sign token space too high-dimensional and non-smooth for LM
