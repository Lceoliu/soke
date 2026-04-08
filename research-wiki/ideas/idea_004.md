---
type: idea
node_id: idea:004
title: "CSL-Daily cross-signer classifier as VAE embedding discriminability probe"
stage: tested
outcome: positive
created_at: 2026-04-08T00:00:00Z
updated_at: 2026-04-08T00:00:00Z
tags: [vae, embedding-quality, classifier-probe, csl-daily, cross-signer, discriminability]
---

# Summary

Use CSL-Daily's natural multi-signer structure (same sentence class, multiple signers) to probe whether frozen VAE continuous embeddings carry cross-signer discriminative information, via a supervised temporal-conv classifier.

## Hypothesis

If VAE embeddings are informationally rich, a classifier trained on some signers should generalize to unseen signers for the same sign class (cross-signer transfer).

## Based On

paper:zuo2024_soke (CSL-Daily dataset), idea:001 (LFQ VAE)

## Target Gaps

gap:G1

## Implementation

- `scripts/analysis/analyze_csl_vae_encoder_similarity.py`
- `scripts/analysis/train_csl_vae_action_classifier.py`
- Split: classes with ≥3 signers; 2 signers → train, 1 signer → test
- Classifier: temporal conv (not mean pooling)
- `max_classes = 1024`

## Outcome

**Positive.** Strong cross-signer discriminability confirmed:
- `num_classes = 1024`, `train_samples = 2048`, `test_samples = 1024`
- `test_acc = 0.6221`, `test_top5_acc = 0.8174`
- Overfit: `train_acc = 1.0`

## Interpretation

VAE embeddings are NOT the bottleneck. The failure of Qwen/mT5 full train is not because the embeddings lack information — they clearly carry strong class-discriminative signal across signers. The problem lies in the downstream training setup, supervision design, or model inductive bias.
