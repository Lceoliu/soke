---
type: claim
node_id: claim:C5
title: "Frozen VAE continuous embeddings carry strong cross-signer discriminative information"
status: supported
created_at: 2026-04-08T00:00:00Z
updated_at: 2026-04-08T00:00:00Z
tags: [vae, embeddings, discriminability, cross-signer, csl-daily]
---

# Claim

A temporal-conv classifier trained on 2 signers per class and tested on a held-out 3rd signer achieves 62.2% top-1 and 81.7% top-5 accuracy in a 1024-way classification task, using frozen VAE continuous embeddings as input.

## Evidence

- exp:005 — CSL-Daily 1024-way cross-signer classifier results

## Implication

The frozen VAE embeddings are NOT the bottleneck for downstream task failure. They encode sufficient discriminative information across signer identities. The failure of Qwen/mT5 must originate in the downstream training setup, data volume, or model architecture, not in embedding quality.
