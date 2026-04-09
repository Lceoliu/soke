---
node_id: paper:defossez2024_moshi
title: "Moshi: a speech-text foundation model for real-time dialogue"
authors: ["Alexandre Défossez", "Laurent Mazaré", "Manu Orsini", "et al. (Kyutai)"]
year: 2024
venue: "arXiv preprint"
arxiv: "2410.00037"
tags: ["unified-ar", "speech", "decoder-only", "semantic-acoustic-split", "rq-transformer", "real-time"]
---

## One-line thesis

Full-duplex speech LLM using a split RVQ codec (level 1 = WavLM-distilled semantic tokens, levels 2-8 = acoustic reconstruction) with a multi-stream RQ-Transformer, enabling simultaneous listening and speaking in real time.

## The semantic/acoustic split — key insight

**Level 1 (Mimi RVQ)**: Distilled from WavLM self-supervised model → **semantic tokens** (encode *what* is said)
**Levels 2-8**: Standard acoustic quantizers → **acoustic tokens** (encode *how* it sounds)

The LM operates primarily on level-1 semantic tokens for language understanding; acoustic tokens provide reconstruction quality for speech output.

## Inner monologue

Text tokens are predicted *before* each audio frame: the model generates aligned text as an internal representation, then generates the audio frame conditioned on it. This creates an implicit ASR "thought" before speaking — bridging understanding and generation in the same forward pass.

## Relevance for sign language

**Direct analogy to our 3-VAE design**:
- Level 1 (semantic): should capture *what sign* is made — the linguistic content (handshape + location + movement)
- Levels 2-N (acoustic/kinematic): capture fine-grained joint trajectories, reconstruction quality

Our current LFQ is all "acoustic" — there is no level-1 semantic layer. **Adding a semantic training objective (self-supervised masked sign prediction) to the first LFQ codebook** would mirror Moshi's split and enable the unified AR.

Inner monologue analogy: gloss tokens (if available) or implicit sign-type tokens predicted before text output → could help m2t.
