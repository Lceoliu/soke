# Research Wiki Index

> Auto-generated categorical index.

Last updated: 2026-04-08T00:00:00Z

## Papers

| Node ID | Title | Year | Venue | Relevance |
|---------|-------|------|-------|-----------|
| paper:zuo2024_soke | Signs as Tokens: A Retrieval-Enhanced Multilingual Sign Language Generator | 2024 | ICCV 2025 | core |
| paper:jiang2023_motiongpt | MotionGPT: Human Motion as a Foreign Language | 2023 | NeurIPS 2023 | core |
| paper:li2025_unisign | Uni-Sign: Toward Unified Sign Language Understanding at Scale | 2025 | ICLR 2025 | core |
| paper:gan2024_signclip | SignCLIP: Connecting Text and Sign Language by Contrastive Learning | 2024 | arXiv | related |
| paper:walsh2024_lost_in_translation | Lost in Translation, Found in Embeddings: SLT and Alignment | 2024 | arXiv | related |
| paper:cho2025_discord | DisCoRD: Discrete Tokens to Continuous Motion via Rectified Flow Decoding | 2025 | ICCV 2025 | related |

## Ideas

| Node ID | Title | Stage | Outcome |
|---------|-------|-------|---------|
| idea:001 | LFQ tokenizer + enhanced reconstruction losses | tested | partial |
| idea:002 | Qwen decoder-only unified modeling (t2m+m2t+mc) | tested | negative |
| idea:003 | mT5 seq2seq with frozen VAE continuous embeddings | tested | negative |
| idea:004 | CSL-Daily cross-signer classifier as embedding probe | tested | positive |

## Experiments

| Node ID | Title | Status |
|---------|-------|--------|
| exp:001 | LFQ VAE reconstruction quality evaluation | completed |
| exp:002 | Qwen overfit experiments (1/4/12 samples) | completed |
| exp:003 | Qwen full train (0.5B and 3B) | completed |
| exp:004 | mT5 motion shuffle / replacement test | completed |
| exp:005 | CSL-Daily cross-signer action classifier (1024-way) | completed |

## Claims

| Node ID | Title | Status |
|---------|-------|--------|
| claim:C1 | LFQ + enhanced losses improve reconstruction quality | supported |
| claim:C2 | Better reconstruction does not automatically improve LM generalization | supported |
| claim:C3 | Qwen pipeline is functionally correct (overfit succeeds) | supported |
| claim:C4 | mT5 actively uses motion embeddings (not ignoring them) | supported |
| claim:C5 | Frozen VAE embeddings carry strong cross-signer discriminability | supported |

## Gaps

| ID | Gap | Status |
|----|-----|--------|
| G1 | What makes a tokenizer LM-compatible (not just good at reconstruction)? | unresolved |
| G2 | Why does full-train generalization fail for both Qwen and mT5? | unresolved |
| G3 | How to make unified multi-task sign AR training generalize at current data scale? | unresolved |
| G4 | How to align sign embedding space with text LM space beyond next-token prediction? | unresolved |
