# Experiment Plan

**Problem**: Sign language LM fine-tuning fails to generalize at full-train scale: both decoder-only (Qwen) and seq2seq (mT5) produce semantically unrelated output despite a correct pipeline (exp:002), discriminative VAE embeddings (claim:C5), and confirmed use of motion signal (claim:C4).

**Method Thesis**: Explicit contrastive pre-training of a sign projection module — aligning frozen VAE embeddings with text sentence embeddings before LM fine-tuning — enables sign-to-text translation generalization that pure cross-entropy LM fine-tuning cannot achieve.

**Date**: 2026-04-08

---

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|------------------------------|---------------|
| C-A (primary): Contrastive sign-text pre-training enables m2t generalization | Directly addresses the core failure mode; distinguishes this paper from failed baselines | ≥2× BLEU4 over direct mT5 fine-tune on CSL-Daily test; holds on PHOENIX (cross-dataset) | B1, B2 |
| C-B (supporting): Failure is not embedding quality but objective misalignment | Rules out the "bad tokenizer" explanation; motivates the alignment-first approach as a general principle | Cross-signer classifier (exp:005, already done) + ablation showing same embeddings succeed with contrastive pre-train | B2, B5 |
| Anti-claim to rule out: Gain is from more parameters or compute, not alignment objective | Without this, reviewers will argue "just use a bigger model" | Same-compute ablation: longer mT5 direct fine-tune vs. contrastive pre-train + mT5 fine-tune | B2 |

---

## Paper Storyline

**Main paper must prove:**
1. Direct LM fine-tuning baseline fails (quantitative — already have this from exp:003/004)
2. Contrastive pre-training + mT5 fine-tuning succeeds on CSL-Daily m2t (B1)
3. The contrastive objective — not compute or model size — is responsible (B2 ablation)
4. Results generalize across datasets (PHOENIX in B1)

**Appendix can support:**
- Quantitative analysis of sign-text alignment geometry before/after contrastive pre-training (t-SNE, cosine similarity distribution) — B5
- Cross-signer generalization analysis — B5
- Does contrastive pre-train also fix Qwen? — B4
- DisCoRD-style flow decoder for t2m — B3 variant

**Experiments intentionally cut:**
- Full unified t2m+m2t+mc with contrastive pre-train (defer to next paper after m2t is proven)
- Ablations over VAE architecture variants
- Very large model scaling (Qwen-8B full fine-tune) — budget prohibitive and not the main point

---

## Experiment Blocks

### Block B1: Main Anchor — Contrastive Pre-train + mT5 vs. Direct Fine-tune

- **Claim tested**: C-A
- **Why this block exists**: This is the paper's core result — without it, there is no contribution.
- **Dataset / split / task**:
  - Primary: CSL-Daily m2t (train/val/test standard split)
  - Transfer: PHOENIX-2014T m2t (same model, no CSL-Daily-specific tuning — test generality)
- **Compared systems**:
  - **Ours**: contrastive pre-train (sign VAE embedding ↔ text sentence, InfoNCE) → mT5 LoRA fine-tune
  - **Baseline A**: mT5 direct LoRA fine-tune (existing exp:003 result — already failed, use as-is)
  - **Baseline B**: Uni-Sign (ICLR 2025) — the strongest existing seq2seq SLT baseline; reproduce or take from paper
- **Metrics**: BLEU4 (primary), BLEU1/2/3, ROUGE-L, WER (lower is better)
- **Setup details**:
  - Sign encoder: frozen VAE (body+lhand+rhand, 1536-dim continuous latent) — same as idea:003
  - Contrastive pre-train: train projection MLP (sign → 512-dim) and text projection (mT5 encoder sentence pooling → 512-dim) with InfoNCE; temperature τ=0.07; batch=256; 50–100 epochs on CSL-Daily train set
  - mT5 fine-tune: same LoRA config as before (rank=64, alpha=128); projection is trainable; VAE frozen
  - 3 seeds per system
- **Success criterion**: BLEU4(ours) ≥ 2× BLEU4(baseline A) on CSL-Daily test; PHOENIX test also improves over baseline A.
- **Failure interpretation**: If contrastive pre-training fails to improve BLEU4, the hypothesis that alignment is the missing ingredient is wrong. Fall back to investigating: (1) data volume (G3), (2) task design (single vs. multi-task), (3) whether end-to-end training is needed.
- **Table / figure target**: Main Table 1 (CSL-Daily), Table 2 (PHOENIX)
- **Priority**: MUST-RUN

---

### Block B2: Novelty Isolation — Ablating the Contrastive Objective

- **Claim tested**: C-A (mechanism), Anti-claim (rules out trivial explanation)
- **Why this block exists**: Reviewers will ask "what exactly does the contrastive pre-training add?" and "is this just more training?"
- **Dataset / split / task**: CSL-Daily m2t, same split as B1
- **Compared systems**:
  - **Ours (full)**: contrastive pre-train → mT5 fine-tune (from B1)
  - **Ablation A — matched compute**: direct mT5 fine-tune for (contrastive_steps + finetune_steps) total steps, no contrastive objective; tests if it is "just more training"
  - **Ablation B — frozen projection**: contrastive pre-train, then freeze projection, fine-tune only LoRA; tests whether projection adaptation during LM fine-tune matters
  - **Ablation C — MSE instead of InfoNCE**: replace contrastive loss with mean-squared embedding matching; tests if InfoNCE specifically is needed or just any alignment signal
  - **Ablation D — text encoder only (no sign pre-train)**: randomly initialized projection, train directly on m2t; baseline to confirm pre-training matters
- **Metrics**: BLEU4, ROUGE-L
- **Setup details**: Same backbone as B1; 3 seeds
- **Success criterion**: Full > Ablation A (rules out compute explanation), Full > Ablation D (confirms pre-training helps), Full ≥ Ablation B (shows projection adaptation matters at least somewhat)
- **Failure interpretation**: If Full ≈ Ablation A, the gain is from compute, not alignment → rethink. If Full ≈ Ablation D, pre-training is irrelevant → alignment hypothesis wrong.
- **Table / figure target**: Table 3 (ablation), or Appendix if space-constrained
- **Priority**: MUST-RUN

---

### Block B3: Simplicity Check — vs. Stronger / Bigger Baselines

- **Claim tested**: C-A (robustness), simplicity defense
- **Why this block exists**: Reviewers will ask "why not just use a bigger mT5?" or "why not Uni-Sign fine-tuned on your data?"
- **Dataset / split / task**: CSL-Daily m2t
- **Compared systems**:
  - **mT5-XL direct fine-tune** (without contrastive pre-train): tests if model scale alone fixes the problem; if mT5-XL direct ≈ ours, the paper's contribution is weakened
  - **Uni-Sign fine-tuned on CSL-Daily** (fair comparison with their framework): the strongest known baseline
  - **Ours (mT5-base + contrastive pre-train)**: main method; should match or exceed mT5-XL direct
- **Metrics**: BLEU4, ROUGE-L, WER; training cost in GPU-hours
- **Setup details**: mT5-XL requires ~4× more memory; estimate ~8 A100-hours for fine-tune; if budget prohibitive, use mT5-large
- **Success criterion**: mT5-XL direct < ours (contrastive pre-train matters more than scale); our method competitive with Uni-Sign fine-tuned under same data conditions
- **Failure interpretation**: If mT5-XL direct ≥ ours, scale is the real variable and alignment is secondary → conclusion becomes "scale + alignment" rather than "alignment alone".
- **Table / figure target**: Main Table 1 (include all baselines in one table)
- **Priority**: MUST-RUN (mT5-XL comparison), NICE-TO-HAVE (Uni-Sign reproduction if compute allows)

---

### Block B4: Frontier Necessity — Does Alignment Also Fix Qwen?

- **Claim tested**: C-A generality; whether the fix is architecture-specific to seq2seq
- **Why this block exists**: If contrastive pre-train only fixes mT5 (seq2seq) but not Qwen (decoder-only), the fix is architecture-specific and the paper's scope must reflect that.
- **Dataset / split / task**: CSL-Daily m2t only (no t2m for this block)
- **Compared systems**:
  - **Qwen-0.5B + contrastive pre-train → LoRA fine-tune (m2t only)**: single-task, no t2m/mc interference
  - **Qwen-0.5B direct fine-tune (m2t only)**: existing exp:003 partial result; re-run for exact comparability
  - **mT5-base + contrastive pre-train (from B1)**: for direct arch comparison
- **Metrics**: BLEU4, ROUGE-L
- **Setup details**: Contrastive pre-train same as B1; Qwen LoRA fine-tune m2t-only with pw=0.5
- **Success criterion**: If Qwen + contrastive pre-train also improves, the finding generalizes to decoder-only → stronger claim. If not, the paper focuses on seq2seq.
- **Failure interpretation**: Qwen architecture is less compatible with this alignment approach → seq2seq is the right tool here (supports claim:C-B indirectly).
- **Table / figure target**: Table 3 or Appendix (architecture comparison)
- **Priority**: NICE-TO-HAVE (run if B1 succeeds with budget to spare)

---

### Block B5: Failure Analysis and Alignment Geometry

- **Claim tested**: C-B (mechanistic support for why alignment helps)
- **Why this block exists**: The paper needs to explain *why* contrastive pre-training works, not just *that* it works; also shows what is still unsolved.
- **Dataset / split / task**: CSL-Daily test set
- **Analysis components**:
  1. **Embedding geometry**: cosine similarity (sign, text) before vs. after contrastive pre-training (distribution plot); t-SNE of aligned embeddings colored by class
  2. **Cross-signer analysis**: does the full pipeline (contrastive + mT5) generalize better across signers than baseline? Measure BLEU4 per-signer held-out test
  3. **Qualitative examples**: 10 examples side-by-side (baseline mT5, ours, GT) — include failure cases where ours still fails
  4. **Error categorization**: semantic error (wrong topic), fluency error, proper noun error
- **Metrics**: Cosine similarity distribution; BLEU4 per signer; qualitative table
- **Setup details**: Use frozen representations before and after pre-training; no extra runs needed
- **Success criterion**: Post-contrastive embeddings show tighter sign-text cluster alignment than pre-contrastive; cross-signer BLEU4 improves
- **Failure interpretation**: If alignment geometry looks similar before/after, contrastive loss did not actually change the embedding structure → might be optimizer effect, not alignment
- **Table / figure target**: Figure 1 (embedding geometry), Figure 2 (qualitative examples), Appendix (cross-signer table)
- **Priority**: MUST-RUN (geometry plot), NICE-TO-HAVE (cross-signer BLEU4)

---

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0 — Sanity | Verify contrastive pre-training pipeline: sign embedding → projection → InfoNCE loss decreases; overfit 100 pairs | Contrastive pre-train loop on 100 CSL-Daily train samples; verify cosine sim increases over 5 epochs | Cosine sim must improve vs. random init; InfoNCE loss must decrease | <1 GPU-hour | High: projection initialization, numerical stability |
| M1 — Contrastive Pre-train | Full contrastive pre-training on CSL-Daily train set | 1 run, mT5-base projection; 50–100 epochs | InfoNCE loss converges; cosine(sign,text) distribution shifts visibly right | ~4–8 GPU-hours | Medium: batch size vs. GPU memory; need sufficient negatives per batch |
| M2 — Fine-tune Comparison | B1 main result: contrastive pre-train + mT5 vs. direct mT5 (baseline) | 2 systems × 3 seeds = 6 fine-tune runs | **GO** if BLEU4(ours) > BLEU4(baseline) at ≥1 seed. **STOP** if no improvement across all seeds → rethink hypothesis | ~12 GPU-hours | HIGH: this is the core bet; if it fails, replan |
| M3 — Ablations | B2 novelty isolation ablations | 4 ablation variants × 1 seed first, then 3 seeds for kept variants | Must show matched-compute ablation < full method | ~16 GPU-hours | Medium |
| M4 — Scale + Generalization | B3: mT5-XL baseline; PHOENIX transfer test | 2 runs + PHOENIX eval on M2 checkpoint | mT5-XL direct < ours on CSL; ours improves on PHOENIX | ~8 GPU-hours | Medium: mT5-XL memory; PHOENIX may be too small to show clear signal |
| M5 — Analysis | B5 geometry plots + qualitative examples | Post-hoc analysis on M2 checkpoint | Always go | ~2 GPU-hours | Low |
| M6 — Optional: Qwen | B4: contrastive + Qwen fine-tune | 2 runs × 1 seed | Run only if M2 succeeds and compute allows | ~6 GPU-hours | Medium |

**Total must-run estimate**: ~42–50 GPU-hours (A100 equivalent)
**Total with nice-to-haves**: ~56–64 GPU-hours

---

## Compute and Data Budget

- **Total estimated GPU-hours**: 42–50 (must-run), 56–64 (full)
- **Data preparation needs**: CSL-Daily train/val/test already loaded; no new annotation required; may need sentence encoder pre-computation (mT5 encoder forward pass, cached)
- **Human evaluation needs**: None planned (metrics are automatic)
- **Biggest bottleneck**: M2 decision gate — if contrastive pre-train + mT5 fine-tune does not beat baseline, the entire hypothesis needs revision. Plan 2 weeks for M0–M2 before committing to M3–M6.

---

## Risks and Mitigations

- **Risk: Contrastive pre-training with small batch produces poor negatives, alignment doesn't work**
  - Mitigation: Use in-batch negatives (256 pairs/batch minimum); cache all sign embeddings upfront to enable large effective batch; try hard negative mining from same sentence class (different signer)

- **Risk: Contrastive pre-training takes too long or overfits on CSL-Daily**
  - Mitigation: Stop at best val InfoNCE; use weight decay + dropout on projection MLP; early stopping on downstream BLEU4 proxy (evaluate m2t every N contrastive epochs)

- **Risk: mT5 fine-tune after contrastive pre-train still fails (M2 stop signal)**
  - Mitigation: If hypothesis fails, pivot immediately to investigating data scale (G3) — design a data augmentation experiment using SLP output as pseudo-labels, or investigate whether freezing vs. un-freezing the projection during LM fine-tune matters

- **Risk: PHOENIX transfer doesn't work (too few samples, different sign language)**
  - Mitigation: Report PHOENIX as secondary; focus main claim on CSL-Daily; if PHOENIX works, that's a bonus

- **Risk: Uni-Sign comparison is unfair (they use RGB + larger pre-training data)**
  - Mitigation: Clearly state comparison is under same data conditions (CSL-Daily only, no external pre-training data for ours); report this as a "data-efficient" angle

---

## Final Checklist

- [ ] Main paper tables are covered (CSL-Daily m2t — B1 provides Table 1)
- [ ] Novelty is isolated (B2 matched-compute ablation)
- [ ] Simplicity is defended (B3: ours with mT5-base beats mT5-XL direct)
- [ ] Frontier contribution is justified: contrastive pre-training is not novel in general, but applying it to 3D pose VAE embeddings for sign language is; the paper's novelty is the insight + demonstration for this domain
- [ ] Nice-to-have runs are separated from must-run runs (B4 Qwen, PHOENIX marked NICE-TO-HAVE)

---

## REPLAN: B5 Failure Analysis — Activated 2026-04-09

> M2 gate FAILED. Best csl_BLEU4: contrastive=1.778 vs. baseline=1.742 (+0.04, not 2×).
> B5 failure analysis is now the primary path. Blocks B2-B4 are paused.

### Failure Hypotheses (ranked by confidence)

| Priority | Hypothesis | Evidence | Diagnostic Experiment |
|---|---|---|---|
| H1 (HIGH) | Sequence-level InfoNCE alignment ≠ token-level decodability | Mean-pool contrastive gives global retrieval but not per-frame discrimination; consistent with SignCL (Ye 2024 NeurIPS) | R015: temporal density audit |
| H2 (HIGH) | VAE temporal density too high for LM decoding | Adjacent sign frames near-identical → attention degeneracy | R015: pairwise cosine_sim within-sample |
| H3 (MEDIUM) | val_m2t_loss / PPL metric bug | PPL ~1e8 inconsistent with BLEU1=16; teacher-forcing label shift? | Code inspection + manual forward pass check |
| H4 (MEDIUM) | Data scale insufficient (18k CSL-Daily vs. 100k+ in GFSLT-VLP pretraining) | Field's best gloss-free needs large-scale pretraining corpus | Compare with GFSLT-VLP (S3D on Kinetics) |
| H5 (LOW) | LoRA capacity insufficient for projection adaptation | rank=64 should be sufficient for 768-dim | Ablation: increase rank or unfreeze more params |

### New Experiment Sequence

**Phase F1: Diagnosis (no GPU needed / <1h)**
- [ ] **F1-A**: Inspect val_m2t_loss computation in mgpt_mt5.py — check label shift, padding, reduction
- [ ] **F1-B**: Run R015 (density audit): compute within-sample temporal cosine_sim of VAE embeddings for 50 CSL-Daily samples; report mean/std/percentile distribution

**Phase F2: Targeted Fixes (based on F1 diagnosis)**
- **If H2 confirmed (high density)**: 
  - [ ] **F2-A**: Apply SignCL-style temporal contrastive within VAE embedding sequences (adjacent=positive, distant=negative) before LM fine-tune
  - This is a 1-GPU, ~4h run; add as R019 in tracker
- **If H3 confirmed (metric bug)**:
  - [ ] **F2-B**: Fix metric computation; re-evaluate R003/R006 checkpoints without retraining
- **If H1 confirmed (global vs. local)**:
  - [ ] **F2-C**: Frame-level contrastive: align each sign frame embedding to its gloss-label text (requires CSL-Daily gloss annotations — check if available)

**Phase F3: New Direction — ST-GCN VAE Encoder (PRIMARY)**
> Pseudo-gloss direction is abandoned. Primary architectural hypothesis: Conv1d encoder is too naive for SMPL-X pose.

Data format note: our 133-dim input is SMPL-X axis-angle rotations (≈41 joints × 3). Reshape to [N_joints, 3] per frame → apply graph convolution using SMPL-X kinematic tree as adjacency.

- [ ] **F3-A**: Implement ST-GCN encoder for VAE
  - Shared variant: single ST-GCN on all joints with SMPL-X tree adjacency
  - Or Uni-Sign-style: separate sub-graph GCN per body / lhand / rhand (matches existing 3-VAE split)
  - Output: [T', code_dim=512] → same interface as Conv1d encoder; LFQ quantizer unchanged
- [ ] **F3-B**: Re-train body/hand VAEs with ST-GCN encoder (same reconstruction + LFQ losses)
- [ ] **F3-C**: Re-run M2 (contrastive+mT5 vs. baseline mT5) on new embeddings; check if csl_BLEU4 breaks through ~5

### New Success Criterion

- **Phase F1 success**: Confirm primary failure mode (temporal density via R015, metric bug via code inspection)
- **Phase F2 success**: Temporal contrastive fix or metric fix → at least one run at csl_BLEU4 ≥ 3.0
- **Phase F3 success**: ST-GCN encoder → csl_BLEU4 ≥ 5.0 (approaching unbiased GFSLT-VLP CSL-Daily baseline ~10-12)
- **Paper-ready**: csl_BLEU4 ≥ 10 with ablation confirming encoder architecture is the critical variable

### Updated Claim Under Investigation

**Old**: "Contrastive pre-training of sign projection (sequence-level) enables m2t generalization."
→ **FALSIFIED** by M2.

**New (post-replan)**: "The Conv1d VAE encoder is architecturally insufficient for sign pose — it destroys the skeletal graph structure inherent to SMPL-X data. A graph-structured encoder (ST-GCN on the SMPL-X kinematic tree) produces richer, more discriminative embeddings that enable downstream SLT."

**New working hypothesis**: "Sign-to-text translation requires *temporal* rather than *global* sign-text alignment. Sequence-level contrastive pre-training is insufficient because: (1) it aligns global summaries, not per-frame decodable representations; (2) VAE embeddings have high temporal density that makes individual frame discrimination degenerate without explicit temporal contrastive regularization."

This hypothesis leads to a cleaner contribution if confirmed: **temporal contrastive alignment (frame-level, intra-sequence) is the missing ingredient**, not global sentence-level alignment.

---

## Phase G: Semantic VAE Encoder Fine-tuning — Activated 2026-04-12

> **Root cause (R025)**: mT5 oracle BLEU4=1.890 ≈ mT5 baseline=1.742. Mean NN cosine=0.87 across different sign sentences.
> mT5 is already near its theoretical ceiling. The bottleneck is the VAE encoder, not the LM.
> The VAE was trained purely for reconstruction → "acoustic class" embeddings (sign-language equivalent of EnCodec).
> Fix: inject semantic training signal into the VAE encoder to reduce inter-sentence cosine similarity.

### Diagnosis Recap

| Evidence | Finding |
|---|---|
| R025: oracle BLEU4=1.890, mT5=1.742 | mT5 already near theoretical ceiling |
| R025: mean NN cosine=0.87 | Different sign sentences are nearly indistinguishable in embedding space |
| R026: 10k→18k ×2.3 jump | More data has marginal benefit, already near oracle |
| G8 confirmed | ST-GCN encoder alone (without semantic objective) did not help (R022 BLEU4=0.993) |

### Plan A: Contrastive VAE Encoder Fine-tuning (PRIMARY — R027)

**Hypothesis**: Injecting InfoNCE sign↔text loss into VAE encoder training pushes the encoder to produce embeddings where different sign sentences are more separable in cosine space. mT5 downstream performance is constrained by oracle ceiling; raising the ceiling (lower mean NN cosine) should raise mT5 BLEU4.

**Architecture**:
```
Frozen: VAE decoder, LFQ quantizer, quantize_out (all three VAEs)
Trainable: VAE encoder, quantize_in (all three VAEs)

Loss = λ_recon × SmoothL1(reconstruct(pose), pose)
      + λ_contra × InfoNCE(mean_pool(encode_continuous(pose)), mT5_encode(text))

λ_recon=1.0, λ_contra=0.5 (default; ablate 0.1, 1.0)
```

**Why freeze decoder/quantizer**: Reconstruction quality (claim:C1) is already good. We only need to improve the encoder's representation without breaking the VAE. The quantizer is downstream of the encoder — by keeping it frozen and only training the encoder, we ensure the codebook usage patterns remain valid.

**Validation metric (R025-style re-run after fine-tuning)**:
- Run oracle BLEU4 analysis with fine-tuned encoder
- Target: oracle BLEU4 > 5.0 (vs. current 1.890)
- Target: mean NN cosine < 0.70 (vs. current 0.87)
- Then re-run mT5 fine-tune (R028) on improved embeddings

**Script**: `scripts/train_semantic_vae_finetune.py`
**Config**: `configs/vae/semantic_vae_finetune_contra.yaml`

**Success criterion**: After R027 encoder fine-tuning, oracle BLEU4 > 3.0 AND mean NN cosine < 0.78.
**Failure criterion**: oracle BLEU4 < 2.0 or training loss diverges → Plan B.

---

### Plan B: Masked Sign Prediction (SECONDARY — R029, if Plan A fails)

**Hypothesis**: Inspired by HuBERT — self-supervised masked prediction forces the encoder to produce representations that distinguish individual sign tokens, without requiring text supervision. This is architecture-level semantic pretraining.

**Method**:
- Mask ~15% of input frames with learnable mask token before VAE encoder
- Add a lightweight prediction head: predict the original masked frame's LFQ code
- Cross-entropy loss on masked positions (sign-HuBERT style)
- Does not require parallel text annotations → can use unlabeled sign pose data

**Limitation**: Requires either (a) a large unlabeled pose corpus, or (b) CSL-Daily train set as pseudo-unlabeled data. With only 18k samples, may overfit. Treat as backup if Plan A fails.

---

### Plan C: Semantic/Acoustic Split — Moshi-Style (EXPLORATORY — R030)

**Hypothesis**: Instead of a single quantizer, use a 2-level design:
- Level 1 (L1): semantic quantizer trained with contrastive loss → captures *what sign* is made
- Level 2 (L2): reconstruction quantizer → captures fine-grained kinematics

The mT5 downstream model uses only L1 tokens/embeddings. L2 is used only for motion generation.

**This is a full redesign of the tokenizer** — implement only if Plans A and B both fail and a new architecture is justified.

---

### Phase G Run Order

| Run | Plan | Purpose | Input | Key Metric | Status |
|---|---|---|---|---|---|
| R027 | A | Fine-tune VAE encoder with contrastive loss | Pre-trained Conv1d VAE (R006 config) | oracle BLEU4 after fine-tune | TODO |
| R027-oracle | A | R025-style oracle re-run on R027 encoder | R027 checkpoint | oracle BLEU4, mean NN cosine | TODO |
| R028 | A | mT5 fine-tune on R027 embeddings | R027 checkpoint | BLEU4 (CSL-Daily test) | TODO |
| R029 | B | Masked sign prediction VAE | CSL-Daily train | oracle BLEU4 | TODO (if R027 fails) |
| R030 | C | Semantic/acoustic split VAE | Full redesign | oracle BLEU4 | TODO (if R027+R029 fail) |

### Phase G Success Criterion

- **R027 success gate**: oracle BLEU4 > 3.0 AND mean NN cosine < 0.78 → proceed to R028
- **R028 success gate**: mT5 BLEU4 > 3.0 on CSL-Daily test → meaningful improvement; target ≥ 5.0 for paper-worthy result
- **Paper ready**: BLEU4 ≥ 5.0 with ablation confirming semantic VAE encoder training is the critical variable
