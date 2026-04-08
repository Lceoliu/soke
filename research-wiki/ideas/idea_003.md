---
type: idea
node_id: idea:003
title: "mT5 seq2seq with frozen VAE continuous embeddings for m2t"
stage: tested
outcome: negative
created_at: 2026-04-08T00:00:00Z
updated_at: 2026-04-08T00:00:00Z
tags: [mt5, seq2seq, continuous-embeddings, frozen-vae, m2t, sign-language-translation]
---

# Summary

Instead of discrete sign tokens, feed frozen VAE continuous embeddings directly into mT5's encoder via a linear projection, and fine-tune only LoRA weights + projection for m2t translation.

## Hypothesis

Continuous embeddings carry more information than discrete tokens and may give seq2seq mT5 a better signal for sign-to-text translation than the Qwen decoder-only approach.

## Based On

paper:li2025_unisign (seq2seq motivation, objective alignment), paper:zuo2024_soke (VAE architecture)

## Target Gaps

gap:G2, gap:G3

## Implementation

- `mGPT/archs/mgpt_mt5.py`
- Pipeline: raw 133-dim motion → frozen body/lhand/rhand VAE `encode_continuous()` → concat (1536-dim) → MLP projection → mT5 encoder `inputs_embeds`
- Prompt: `把下面这句{lang}手语翻译为{lang}文本:`
- LoRA: rank=64, alpha=128; only m2t task implemented (`generate_conditional()` only)
- Configs: `configs/lm/mt5_base.yaml`, `configs/soke_mt5_m2t.yaml`, `configs/soke_mt5_csl_m2t.yaml`

## Outcome

**Negative.** Overfit succeeds. Full train produces grammatically fluent sentences but semantically unrelated to GT — model collapses to high-frequency training set templates (e.g., introduces itself, describes common actions). Failure mode is similar to Qwen despite different architecture and input format.

## Failure Notes

- BLEU4 ~2 on test; shuffle experiment confirms model is NOT ignoring motion (BLEU4 drops to ~0.2 when motion is randomized)
- Model reads motion but cannot form stable sign-to-text bindings at scale
- Typical failure: predicts plausible English, wrong semantics (see `experiments/mgpt/SOKE_MT5_M2T_FULL/auto_reports/downstream/m2t_examples.jsonl`)
- mT5 only supports m2t — no t2m or mc — making it a single-task fallback, not the unified model goal
