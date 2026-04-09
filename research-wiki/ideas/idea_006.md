---
node_id: idea:006
title: "VoxtLM-Style Unified Sign AR: Semantic Sign Tokens + Decoder-Only LM"
status: active
confidence: high
origin: speech_ar_analogy
linked_gaps: [G1, G2, G3, G4]
linked_papers: [paper:hassid2024_voxtlm, paper:zhang2023_speechgpt, paper:defossez2024_moshi, paper:jiang2023_motiongpt]
date: 2026-04-09
---

## Core Insight: We Have the Wrong Tokenizer Class

The unified speech AR field (SpeechGPT, VoxtLM, Moshi, Spirit-LM) has converged on a single design principle:

> **Semantic tokens (HuBERT) enable unified understanding+generation. Acoustic tokens (EnCodec) enable only generation.**

**HuBERT tokens**: trained with self-supervised masked prediction → encode *what* phoneme/word → direction-agnostic → same tokens work as input (ASR) and output (TTS).

**EnCodec tokens**: trained for reconstruction fidelity → encode *how* the audio sounds → good for generation quality, poor for understanding.

**Our LFQ tokens**: trained for reconstruction (VAE loss, SMPL-X pose fidelity) → **they are the sign language equivalent of EnCodec, not HuBERT.** This is why:
- idea:001 (LFQ, better reconstruction) → C1 confirmed, but C2 failed: better acoustic tokens ≠ better LM generalization
- idea:002 (Qwen unified AR) → failed: LLM can't build language grounding on acoustic-class tokens
- idea:003 (mT5 seq2seq) → M2 failed: same problem at the token level

**The fix is not the LM architecture. The fix is the token semantics.**

## The Sign Language HuBERT Problem

Speech has HuBERT: self-supervised transformer trained on 960h+ LibriSpeech, masked prediction of frame clusters → semantically rich discrete units. Sign language has NO equivalent:
- No large-scale unlabeled sign pose dataset for pretraining (or insufficient scale)
- No standard self-supervised sign tokenizer

The closest available proxy: **better encoder (ST-GCN, idea:005) → LFQ codes that capture more semantic structure even under reconstruction training**, because the encoder will be sensitive to kinematic structure (joint angles, inter-joint relationships) rather than raw flat vectors.

This is why ST-GCN is the right first step: it makes the reconstruction-trained tokens *more semantic* without requiring a separate self-supervised pretraining stage.

## The Full Architecture (VoxtLM-Style for Sign)

```
Input pose → [ST-GCN encoder] → 6D rotation features per joint
           → [LFQ quantizer]   → discrete sign tokens {s_1, ..., s_T}
           + text tokenizer    → text tokens {w_1, ..., w_N}

Unified vocabulary: [text_vocab | sign_tokens_body | sign_tokens_lhand | sign_tokens_rhand]

Decoder-only LM (Qwen-0.5B) with 4 task tokens:
  m2t: <start-sign> s_1...s_T <generate-text>  → w_1...w_N
  t2m: <start-text> w_1...w_N <generate-sign>  → s_1...s_T
  mc:  <start-sign> s_1...s_K <generate-sign>  → s_K+1...s_T
```

This is **exactly idea:002** but with:
1. ST-GCN encoder → semantically richer sign tokens
2. Per-body-part tokens interleaved (body, lhand, rhand per frame step)
3. Task prefix tokens instead of task-weight multipliers

## Why This Now Makes Sense

| Failure | Root cause | Fixed by |
|---|---|---|
| idea:002 Qwen unified AR fails | LFQ tokens are acoustic, not semantic | ST-GCN encoder → better tokens |
| idea:003 mT5 seq2seq fails (M2) | Same token class problem + seq2seq alignment | Better tokens + unified AR formulation |
| M1 contrastive helps retrieval (acc=27.7%) but not translation | Global alignment ≠ token-level decodability | Per-token semantic quality fix |

## Experiment Sequence

**Phase 1** (prerequisite): ST-GCN VAE encoder redesign (idea:005)
- Implement ST-GCN on 6D rotation features for body/lhand/rhand
- Retrain 3 VAEs with new encoder, same LFQ quantizer
- Validate: classifier probe accuracy should exceed current 81.8% (claim:C5)

**Phase 2**: Unified AR with new tokens
- Expand Qwen vocabulary: text vocab + body_tokens + lhand_tokens + rhand_tokens
- Add 4 task control tokens
- Train jointly on m2t + t2m + mc (CSL-Daily)
- This is idea:002 retry with correct tokenizer

**Phase 3**: Semantic token quality analysis
- Measure semantic token quality: cluster purity (do same-sign clips map to same token clusters?)
- Compare pre-ST-GCN vs. post-ST-GCN token distribution
- Analogy: compare our tokens to HuBERT phone-purity benchmark in speech

## Open Questions

1. Is ST-GCN encoder alone sufficient to push tokens from "acoustic" to "semantic" class?
   - Alternative if not: add a masked sign prediction auxiliary loss during VAE training (sign-language HuBERT style)
2. Should body/lhand/rhand tokens be interleaved per frame, or sequential (all body first, then hands)?
   - VoxtLM uses single stream; Moshi uses multi-stream. For sign, per-frame interleaving matches the synchronous nature of signing.
3. What vocabulary size per body part? Current LFQ codebooks are small (~512 per VAE). May need larger codebook for semantic richness.
