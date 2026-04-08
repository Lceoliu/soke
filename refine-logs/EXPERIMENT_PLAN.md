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
