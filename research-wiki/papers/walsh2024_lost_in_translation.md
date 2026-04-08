---
type: paper
node_id: paper:walsh2024_lost_in_translation
title: "Lost in Translation, Found in Embeddings: Sign Language Translation and Alignment"
authors: ["et al."]
year: 2024
venue: arXiv
external_ids:
  arxiv: "2512.08040"
  doi: null
  s2: null
tags: [sign-language-translation, gloss-free, embedding-alignment, multilingual, generalization]
relevance: related
origin_skill: research-wiki
created_at: 2026-04-08T00:00:00Z
updated_at: 2026-04-08T00:00:00Z
---

# One-line thesis

Gloss-free sign language translation via large-scale video-text embedding alignment achieves strong open-vocabulary generalization across signers and languages.

## Problem / Gap

Gloss-annotated SLT datasets are expensive; gloss-free approaches struggle to generalize across diverse signers and sign languages.

## Method

- Gloss-free: uses video-text pairs without gloss annotations
- Embedding alignment between sign video and text at sentence level
- Large-scale training enables cross-signer, cross-language generalization

## Key Results

Strong generalization across signers and languages without gloss supervision.

## Reusable Ingredients

- Gloss-free training reduces annotation dependency
- Cross-signer/cross-language generalization as a key evaluation criterion (similar to our CSL cross-signer classifier setup)

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project

Validates that gloss-free approaches with good embedding alignment can generalize. Complements gap:G2 (why does our approach fail?) — their success with alignment-first training suggests our pipeline is missing an explicit alignment stage between sign embeddings and text.
