---
node_id: paper:zhang2023_speechgpt
title: "SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities"
authors: ["Dong Zhang", "Shimin Li", "Xin Zhang", "Jun Zhan", "Pengyu Wang", "Yaqian Zhou", "Xipeng Qiu"]
year: 2023
venue: "EMNLP 2023 Findings"
arxiv: "2305.11000"
tags: ["unified-ar", "speech", "asr", "tts", "decoder-only", "hubert", "semantic-tokens"]
---

## One-line thesis

A single decoder-only LLM (LLaMA) handles both ASR and TTS by expanding the text vocabulary with 256 HuBERT discrete speech tokens — proving that recognition and generation are the same next-token prediction problem.

## Key design choices

- **Tokenizer**: HuBERT → k-means (256 clusters). Semantic, not acoustic.
- **Architecture**: Decoder-only (LLaMA), expanded embedding matrix. NO separate speech encoder.
- **Task unification**: Implicit — both ASR (`speech_tokens → text`) and TTS (`text → speech_tokens`) are learned as next-token prediction from data. No architectural asymmetry.
- **Training**: 3 stages: (1) modality-adaptation pretrain on speech-only, (2) cross-modal instruction fine-tune, (3) chain-of-modality fine-tune.

## Why HuBERT (semantic) works for both directions

HuBERT tokens capture *what* was said (phonetic/linguistic content), independent of *how* it sounded (timbre, pitch). This direction-agnosticism is the key: the same token sequence can be predicted (TTS) or predicted from (ASR) by the same model.

**Contrast with acoustic codecs (EnCodec)**: acoustic tokens encode reconstruction quality, NOT linguistic content. A model trained with acoustic tokens can generate speech but struggles to "understand" content bidirectionally.

## Sign language analogy

Our LFQ tokens are trained for **reconstruction** (VAE loss) → closer to EnCodec (acoustic) than HuBERT (semantic). The sign language equivalent of HuBERT would be a tokenizer trained with self-supervised masked prediction on pose sequences — encoding *what sign* is being made, not reconstruction quality. This is why our unified Qwen AR failed: wrong tokenizer class.
