# Query Pack

> Compressed summary for /idea-creator. Auto-generated. Max 8000 chars.

Last generated: 2026-04-08T00:00:00Z

---

## Project Direction [~280 chars]

Build a **Sign Language Autoregressive Model**: a unified model for sign generation (t2m), translation (m2t), and continuation (mc). Target: CSL-Daily, PHOENIX, How2Sign. Core challenge: making discrete or continuous sign representations work with LLMs at current data scale.

---

## Top Gaps [~1200 chars]

**G1 — Tokenizer LM-Compatibility** ★★★ [unresolved, 2 linked ideas, 1 failed exp]
What makes a sign tokenizer/embedding suitable for downstream LM — not just good reconstruction? Better reconstruction (claim:C1) did NOT improve LM generalization (claim:C2). Criteria for LM-friendly tokenization are unknown. No existing SL paper addresses this directly.

**G2 — Full-Train Generalization Failure** ★★★ [unresolved, 2 linked ideas, 2 failed exps]
Both Qwen (decoder-only) and mT5 (seq2seq) fail full-train generalization: train loss ↓, val ppl ↑, test output semantically unrelated to GT. Pipeline is correct (exp:002). Motion IS used (claim:C4). VAE embeddings ARE discriminative (claim:C5). Root cause unknown. Candidate causes: data volume, training objective misalignment, task interference, token distribution structure.

**G3 — Unified Multi-Task AR at Current Data Scale** ★★ [unresolved, 1 linked idea]
MotionGPT works for body motion (HumanML3D ~15k). Sign language datasets are smaller + harder (hands + semantics). Right recipe for unified t2m+m2t+mc sign AR training at current scale unknown.

**G4 — Sign-Text Space Alignment** ★★ [unresolved, no linked ideas yet]
No explicit alignment between sign embedding space and text LM space. Contrastive pre-training (SignCLIP) works for video; unclear if it would help for pose-based VAE embeddings. Uni-Sign shows objective alignment is critical — but only for SLT, not generation.

---

## Paper Clusters [~1600 chars]

**Cluster A: Base codebase + paradigm (core)**
- paper:zuo2024_soke (ICCV 2025, arxiv:2411.17799): current code ancestor; VQ + mBART + multi-head decoding + retrieval; benchmark: CSL/PHOENIX/ASL
- paper:jiang2023_motiongpt (NeurIPS 2023, arxiv:2306.14795): VQ-VAE + instruction-tuned LLM for unified motion tasks; the paradigm we follow
- These two define the current approach. SOKE is the fork point; MotionGPT is the conceptual framework.

**Cluster B: Generalization-focused SLU (motivation for mT5 pivot)**
- paper:li2025_unisign (ICLR 2025, arxiv:2501.15187): Uni-Sign; seq2seq + large-scale CSL-News pre-training; objective alignment key insight; achieves SOTA on CSL-Daily
- paper:walsh2024_lost_in_translation (arxiv:2512.08040): gloss-free SLT via embedding alignment; strong cross-signer generalization without gloss annotations
- Both motivate: alignment-first training > pure next-token prediction for SLT at current scale.

**Cluster C: Sign-text alignment via contrastive learning (addresses G4)**
- paper:gan2024_signclip (arxiv:2407.01264): CLIP-style contrastive pre-training for sign-text; shared embedding space
- Gap: none of these are pose/VAE-based — all video-based. Open question: does contrastive alignment carry over to 3D VAE embeddings?

**Cluster D: Motion tokenizer quality + decoding (addresses G1)**
- paper:cho2025_discord (ICCV 2025): rectified flow decoder for discrete tokens → continuous motion; improves quality vs. hard codebook lookup
- LFQ literature (MAGVIT-v2): LFQ enables larger codebooks with entropy regularization
- Key gap: for sign language specifically, no paper studies what makes a tokenizer LM-compatible.

---

## Failed Ideas [~1400 chars] ← NEVER PRUNE

**idea:001 — LFQ tokenizer + enhanced losses** [outcome: partial]
Hypothesis: better reconstruction → better downstream LM. FALSIFIED. LFQ does improve reconstruction (claim:C1, exp:001) but full-train LM generalization still fails (exp:003). Lesson: reconstruction quality and LM-compatibility are different axes. Don't optimize tokenizer by reconstruction alone.

**idea:002 — Qwen decoder-only unified modeling (t2m+m2t+mc)** [outcome: negative]
Hypothesis: add sign tokens to Qwen vocab, AR continuation over all tasks. FAILED at full scale. Overfit works (exp:002), full train doesn't (exp:003). m2t_prefix_loss_weight=0.5 helps val ppl but doesn't fix generalization. Failure pattern: model learns text priors, fails to bind sign tokens to semantics. Do not retry this approach without a fundamentally different training objective or massive data scaling.

**idea:003 — mT5 seq2seq + frozen VAE continuous embeddings** [outcome: negative]
Hypothesis: continuous embeddings + seq2seq is more stable than discrete + decoder-only. FAILED similarly. Overfit works. Full train generates fluent but semantically wrong text (templates from training data). mT5 DOES use motion (exp:004, 10× BLEU4 drop when shuffled). Failure mode: motion is read but not semantically bound to output text at scale. Do not retry without an alignment stage.

---

## Top Papers (ranked by gap coverage + centrality) [~900 chars]

1. paper:li2025_unisign — G2, G4 | key insight: objective alignment; CSL-Daily SOTA
2. paper:jiang2023_motiongpt — G3 | unified motion AR paradigm; code ancestor
3. paper:zuo2024_soke — G1, G3 | direct code base; multi-head decoding
4. paper:gan2024_signclip — G4 | contrastive sign-text alignment
5. paper:cho2025_discord — G1 | flow-based decoding from discrete tokens
6. paper:walsh2024_lost_in_translation — G2, G4 | gloss-free alignment + generalization

---

## Active Chains [~500 chars]

Chain 1 (most critical): **G2 root cause → fix**
exp:005 (embeddings ok) → claim:C5 → G2 still open → candidate: add contrastive alignment stage (G4) → paper:gan2024_signclip as template

Chain 2: **G1 tokenizer** → idea:001 failed → alternative: flow decoder (paper:cho2025_discord) → test discrete LM + flow decode for t2m

Chain 3: **G3 data scale** → MotionGPT succeeded at 15k; current SL datasets smaller → explore data augmentation via SLP (Using SLP as Data Augmentation paper, 2506.09643)

---

## Open Unknowns [~400 chars]

1. Would a contrastive pre-training stage (sign VAE embedding ↔ text sentence) before LM fine-tuning fix or improve generalization? (G2, G4)
2. Is full-train failure due to data volume, or fundamentally the training objective? Can we test with a 10× larger dataset?
3. Does task interference (t2m+m2t+mc jointly) hurt individual task learning, vs. single-task mT5?
4. Can DisCoRD-style flow decoding improve t2m output quality independent of LM training?
5. What is the minimum data scale for emergent sign-text binding in decoder-only vs. seq2seq models?
