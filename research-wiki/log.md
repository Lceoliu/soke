# Research Wiki Log

> Append-only timeline of all wiki mutations.

---

- **2026-04-08T00:00:00Z** — Wiki initialized
- **2026-04-08T00:01:00Z** — Ingested research history from `docs/research_history_zh.md`
  - Added 3 core papers: paper:zuo2024_soke, paper:jiang2023_motiongpt, paper:li2025_unisign
  - Added 4 ideas: idea:001 (LFQ VAE, partial), idea:002 (Qwen, negative), idea:003 (mT5, negative), idea:004 (CSL classifier, positive)
  - Added 5 experiments: exp:001–exp:005
  - Added 5 claims: claim:C1–claim:C5
  - Added 4 gaps: G1–G4
  - Added 28 edges to graph/edges.jsonl
- **2026-04-08T00:02:00Z** — Preliminary literature research sweep
  - Added 3 related papers from 2024–2025:
    - paper:gan2024_signclip (arxiv:2407.01264) — contrastive sign-text alignment → addresses G4
    - paper:walsh2024_lost_in_translation (arxiv:2512.08040) — gloss-free SLT with embedding alignment → addresses G2/G4
    - paper:cho2025_discord (ICCV 2025) — flow-based discrete→continuous motion decoding → addresses G1
  - Key finding: alignment-first methods (SignCLIP, Lost in Translation) suggest missing alignment stage in current pipeline may explain G2
  - Key finding: DisCoRD flow decoder separates LM training from decoding quality — relevant to G1
  - Rebuilt index.md, query_pack.md
  - Re-ideation trigger: 3+ new papers ingested, 2 new failed ideas → consider running /idea-creator
