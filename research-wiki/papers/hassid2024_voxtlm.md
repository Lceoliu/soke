---
node_id: paper:hassid2024_voxtlm
title: "VoxtLM: Unified Decoder-Only Models for Consolidating Speech Recognition/Synthesis and Speech/Text Language Tasks"
authors: ["Michael Hassid", "Tal Remez", "Tu Anh Nguyen", "Itai Gat", "Alexis Conneau", "Felix Kreuk", "Jade Copet", "Alexandre Défossez", "Gabriel Synnaeve", "Emmanuel Dupoux", "Roy Schwartz", "Yossi Adi"]
year: 2024
venue: "ICASSP 2024"
arxiv: "2309.07937"
tags: ["unified-ar", "speech", "decoder-only", "task-tokens", "hubert", "asr", "tts", "minimal-overhead"]
---

## One-line thesis

Four special tokens (`<start-text>`, `<start-speech>`, `<generate-text>`, `<generate-speech>`) are all that's needed to turn a decoder-only text LM into a unified ASR+TTS+LM model — proving the tasks are architecturally identical.

## Task unification mechanism

```
ASR:  <start-speech> [speech_tokens] <generate-text>  [text_tokens]
TTS:  <start-text>   [text_tokens]   <generate-speech> [speech_tokens]
LM:   <start-text>   [text_tokens]   <generate-text>   [text_tokens]
```

Same decoder, same weights. Direction is controlled by conditioning prefix only.

## Key design choices

- **Tokenizer**: HuBERT discrete units, k ∈ {50, 200, 1000}
- **Architecture**: OPT decoder-only, single unified vocabulary
- **No separate encoder** for any modality — all tokens are treated as integers in the same sequence

## Most relevant paper for our project

VoxtLM is the **direct template** for sign language AR:
- Replace HuBERT speech tokens → LFQ sign tokens (once semantic quality improves)
- Replace text tokens → mT5/Qwen text tokens
- Use same 4-token control scheme: `<start-sign>`, `<generate-text>` for m2t; `<start-text>`, `<generate-sign>` for t2m
- Add `<generate-sign>` continuation for mc (motion continuation)
- Same decoder-only backbone (Qwen-0.5B in our case)

**This is exactly idea:002 (Qwen unified AR) but with the right tokenizer type (semantic, not reconstruction-optimized).**
