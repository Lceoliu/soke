---
type: idea
node_id: idea:001
title: "LFQ tokenizer + enhanced reconstruction losses for sign language"
stage: tested
outcome: partial
created_at: 2026-04-08T00:00:00Z
updated_at: 2026-04-08T00:00:00Z
tags: [lfq, tokenizer, vae, reconstruction, sign-language]
---

# Summary

Replace the original SOKE VQ codebook with multi-layer LFQ (Lookup-Free Quantization) and add richer reconstruction losses targeting hand kinematics (velocity, FK, acceleration, contact).

## Hypothesis

Better tokenizer reconstruction quality → better downstream LM performance.

## Based On

paper:jiang2023_motiongpt (VQ-VAE pattern), paper:zuo2024_soke (base tokenizer architecture)

## Target Gaps

gap:G1

## Implementation

- `mGPT/archs/mgpt_vq.py`
- `mGPT/archs/tools/quantize_lfq.py`
- `mGPT/losses/mgpt.py`
- Losses: `recons_feature`, `recons_velocity`, `recons_fk_hand`, `recons_accel_hand`, `recons_accel_wrist_rel`, `recons_contact`, `vq_commit`
- Configs: `configs/vq/re128_lfq4.yaml`, `configs/vq/hand256_lfq4.yaml`, `configs/vq/hand256_lfq5.yaml`

## Outcome

**Partial success.** LFQ + richer losses does improve reconstruction quality — generated signs are more humanly recognizable. However, the hypothesis that better reconstruction → better downstream LM was **falsified**: full-train generalization for both Qwen and mT5 remained poor despite better tokenizer.

## Failure Notes

The core assumption that reconstruction quality and LM-compatibility are aligned was wrong. A tokenizer can reconstruct well while producing a latent space that is hard for an LM to model (e.g., non-smooth, high-entropy, no linguistic structure in the discrete token distribution).
