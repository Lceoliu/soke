---
type: claim
node_id: claim:C3
title: "Qwen sign language pipeline is functionally correct (overfit succeeds)"
status: supported
created_at: 2026-04-08T00:00:00Z
updated_at: 2026-04-08T00:00:00Z
tags: [qwen, pipeline, overfit, sanity]
---

# Claim

The Qwen-based sign language AR pipeline (token formatting, model forward pass, loss computation, generation) is internally consistent: given sufficient capacity and few training samples, the model can perfectly fit all three tasks (t2m, m2t, mc).

## Evidence

- exp:002 — overfit1/4/12 all succeed for all three tasks after pipeline bug fixes

## Implication

Full-train failure (exp:003) is NOT caused by implementation bugs. It is a genuine generalization problem, not a pipeline error.
