---
node_id: paper:guo2025_pseudogloss
title: "Bridging Sign and Spoken Languages: Pseudo Gloss Generation for Sign Language Translation"
authors: ["Jianyuan Guo", "Peike Li", "Trevor Cohn"]
year: 2025
venue: "arXiv preprint"
arxiv: "2505.15438"
tags: ["sign-language-translation", "pseudo-gloss", "LLM", "intermediate-representation", "promising"]
---

## One-line thesis

Use an LLM (Gemma2) with in-context learning to generate pseudo-glosses from spoken text, then align them to sign video via weakly-supervised matching — achieving 27+ B@4 on PHOENIX without gloss annotations.

## Problem / Gap

Gloss-free methods plateau at ~22 B@4 on PHOENIX. The gap to gloss-based methods (~30+ B@4) remains. This paper asks: can LLMs generate the intermediate gloss-like structure that is otherwise missing?

## Method

1. **Pseudo-gloss generation**: Prompt Gemma2 with few-shot examples to generate gloss-order text from spoken-language sentences (reordering and lexical simplification to match sign order).
2. **Weak alignment**: Match generated pseudo-glosses to sign video segments via cosine similarity.
3. **Two-stage training**: Pretrain with pseudo-gloss alignment, then fine-tune for SLT.

No manual gloss annotation required at inference time.

## Key Results

| Dataset | Best gloss-free (prior) | PGG-SLT (Gemma2) | Δ |
|---|---|---|---|
| PHOENIX-2014T dev | ~22 B@4 | **27.53 B@4** | +5.5 |
| PHOENIX-2014T test | ~22 B@4 | **27.32 B@4** | +5.3 |
| How2Sign | baseline | +6.1 B@4 | +6.1 |

## Limitations

- Depends on LLM quality; Gemma2 is large and the gloss generation quality degrades for non-English languages.
- CSL-Daily not evaluated (Chinese sign language, Chinese text — LLM gloss generation harder).
- Still below the best gloss-based results (~30+ B@4 with gold glosses).

## Reusable Ingredients

- **Critical insight for our direction**: Jumping from 22 → 27 B@4 by adding *intermediate semantic signals* (pseudo-glosses) strongly suggests the bottleneck is the lack of compositional structure, not the LM or the pretraining objective.
- For CSL-Daily: could use a Chinese LLM (Qwen/ChatGLM) to generate pseudo-glosses from Chinese text annotations.
- **New hypothesis for our M2 failure**: Our contrastive pre-training aligned *sequence-level* sign embeddings to *sentence-level* text — but no *intermediate-level* (gloss-like) signal was provided. The LM has no compositional structure to ground on.

## Open Questions

- Does pseudo-gloss generation work for CSL-Daily (Chinese)?
- Can we generate pseudo-glosses without LLM by using sign annotation structure from the dataset?
- What if we use VAE codebook assignments as "pseudo-glosses" for intermediate alignment?
