---
type: claim
node_id: claim:C4
title: "mT5 actively uses motion embeddings — it is not ignoring the sign modality"
status: supported
created_at: 2026-04-08T00:00:00Z
updated_at: 2026-04-08T00:00:00Z
tags: [mt5, motion-embeddings, modality-collapse]
---

# Claim

The mT5 model does not silently collapse to language-only generation — it processes and is influenced by the frozen VAE motion embeddings fed to its encoder.

## Evidence

- exp:004 — motion shuffle drops BLEU4 from ~2 to ~0.2 (10× reduction), confirming motion is causally relevant to output

## Implication

The full-train generalization failure is NOT modality collapse. The model reads sign embeddings but cannot form stable, generalizable sign-to-text mappings at current data scale. The problem is in the binding / alignment, not in signal presence.
