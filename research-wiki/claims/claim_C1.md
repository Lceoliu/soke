---
type: claim
node_id: claim:C1
title: "LFQ + enhanced reconstruction losses improve sign language VAE reconstruction quality"
status: supported
created_at: 2026-04-08T00:00:00Z
updated_at: 2026-04-08T00:00:00Z
tags: [lfq, vae, reconstruction]
---

# Claim

Multi-layer LFQ quantization combined with richer reconstruction losses (velocity, FK hands, acceleration, contact) produces more humanly-recognizable sign language reconstructions than original SOKE VQ.

## Evidence

- exp:001 — qualitative and quantitative reconstruction improvement confirmed
- Supported by: paper:zuo2024_soke (baseline), MAGVIT-v2 LFQ theory

## Caveat

Better reconstruction quality does NOT automatically imply better downstream LM performance (see claim:C2).
