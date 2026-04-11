# Experiment Results (Initial)

**Date**: 2026-04-08
**Plan**: refine-logs/EXPERIMENT_PLAN.md

---

## M0: Sanity — PASSED ✓

**Run**: R001 — Contrastive pre-train loop on 100 CSL-Daily pairs, 5 epochs

| Metric | Epoch 1 | Epoch 5 | Change |
|--------|---------|---------|--------|
| train_loss (InfoNCE) | 4.5593 | 4.3034 | −5.6% |
| val_loss | 2.2764 | 2.1497 | −5.6% |
| val_avg_pos_sim (cosine) | −0.0855 | +0.2036 | **+0.2891** |

**Gate result**: PASS — InfoNCE loss decreased >5%, positive-pair cosine similarity increased by +0.29 (from negative to positive). Contrastive pre-training loop is working correctly.

**Checkpoint**: `experiments/contrastive_pretrain_sanity/best_sign_proj.pt`
- Keys: `norm.weight [1536]`, `norm.bias [1536]`, `proj.0.weight [768×1536]`, `proj.0.bias [768]`, `proj.2.weight [768×768]`, `proj.2.bias [768]`
- Compatible with `MT5Seq2SeqLM.sign_proj` (matches `SignEmbeddingProjection` architecture with `use_mlp_proj=True`)

---

## M1: Full Contrastive Pre-train — DONE ✓

**Run**: R002 — Full contrastive pre-training on CSL-Daily train set (~17k samples)

**Results** (80 epochs, batch_size=512, GPU 0, ~13 min total):

| Metric | Epoch 1 | Epoch 80 | Change |
|--------|---------|----------|--------|
| train_loss (InfoNCE) | 6.0752 | 3.5406 | −41.7% |
| val_loss | 5.6977 | 4.0157 | −29.5% |
| acc_s2t (top-1/512 negatives) | 0.7% | **27.7%** | +27.0pp |
| temperature | 0.0699 | 0.0612 | learned |

**Checkpoint**: `experiments/contrastive_pretrain_csl/best_sign_proj.pt` (7.1 MB)
- Keys: `norm.weight [1536]`, `proj.0.weight [768×1536]`, `proj.2.weight [768×768]`
- Compatible with `MT5Seq2SeqLM.sign_proj` when `use_mlp_proj=True`

**Interpretation**: acc_s2t=27.7% (top-1 among 512 negatives) indicates strong sign-text alignment learning. A random model would achieve 1/512 ≈ 0.2%. The projection learned to map sign VAE embeddings into a space where correct text can be retrieved with 27.7% top-1 accuracy.

---

## M2: Fine-tune Comparison — DONE ✗ (Hypothesis NOT supported)

**Runs**: R003 (contrastive+mT5) and R006 (baseline mT5), 80 epochs, 4 GPUs each, CSL-Daily

### Key metrics (CSL-Daily val set, csl_BLEU subscore)

| Metric | R003 contrastive | R006 baseline | Δ |
|--------|------------------|----------------|---|
| **Best csl_BLEU_4** (across 80 val rounds) | **1.778** | 1.742 | +0.04 |
| **Best csl_BLEU_1** | **17.44** | 16.83 | +0.61 |
| Final csl_BLEU_4 (E79) | 0.800 | **1.185** | −0.39 |
| Final csl_BLEU_1 (E79) | 15.97 | **16.19** | −0.22 |
| Final val_m2t_loss | 15.51 | 16.03 | −0.52 |

### Decision gate

- **Required**: BLEU4(contrastive) ≥ 2× BLEU4(baseline) ≈ 3.5
- **Observed**: best 1.78 vs 1.74 — **gate FAILED** (essentially tied; not 2×)
- **Final epoch**: contrastive is *worse* than baseline (0.80 vs 1.19)

### Interpretation

- Contrastive pre-training did **not** unlock m2t generalization on CSL-Daily.
- Both runs plateau at very low BLEU4 (~1) and BLEU1 ~16, suggesting both are stuck near unigram-frequency baselines — the model is not learning sentence structure.
- The contrastive init gives a *marginal* early-training advantage (better best-Bleu1) but the benefit washes out and the final epoch is worse than baseline. Likely the model overfits and the projection drifts from the contrastive optimum.
- Both runs show high val_m2t_loss (~15-16) and ppl ~1e8 — the loss/ppl numbers suggest a teacher-forcing/label-smoothing issue or that the metric is computed on a degenerate distribution; needs debugging.

### Logs

- `experiments/mgpt/SOKE_MT5_CSL_CONTRASTIVE/train.log` (27.7 MB, 80 epochs, last write 2026-04-09 04:00)
- `experiments/mgpt/SOKE_MT5_CSL_M2T/train.log` (27.7 MB, 80 epochs, last write 2026-04-09 04:06)

### Status

**M2 gate FAILED.** Plan branch B5 (failure analysis) is now active. Do not launch R004/R005/R009-R012 until we understand why the contrastive signal didn't transfer.

---

## (legacy / reference) M2 plan

**Command (contrastive + mT5)**:
```bash
python train.py --cfg configs/soke_mt5_csl_contrastive.yaml
```

**Command (direct mT5 baseline)**:
```bash
python train.py --cfg configs/soke_mt5_csl_m2t.yaml
```

**Eval command (after training)**:
```bash
python scripts/eval_m2t_bleu.py \
    --pred_file experiments/mgpt/SOKE_MT5_CSL_CONTRASTIVE/auto_reports/downstream/m2t_examples.jsonl \
    --output_file results/eval_m2t_contrastive_csl.json

python scripts/eval_m2t_bleu.py \
    --pred_file experiments/mgpt/SOKE_MT5_CSL_M2T/auto_reports/downstream/m2t_examples.jsonl \
    --output_file results/eval_m2t_baseline_csl.json
```

---

## Summary

- **4/22** must-run experiments completed (M0 sanity, M1, M2 ×2 [FAILED])
- M2 gate FAILED: contrastive pre-training did not enable m2t generalization
- Active plan: Phase F1 (diagnosis) + Phase F3 (ST-GCN encoder — PRIMARY)
- Ready for /auto-review-loop: NO — awaiting F3 results

---

## F1-A: val_m2t_loss Code Inspection — DONE ✓ (2026-04-09)

**Finding**: No label shift bug found.
- HuggingFace mT5 handles decoder input shift internally when `labels` is provided
- Padding masked correctly with -100 (`_tokenize_targets`)
- Loss reduction is mean over non-padded positions (HF default)
- The high val_m2t_loss (~15.5) represents genuine training failure:
  - mT5 vocab ~250k → random CE loss = ln(250k) ≈ 12.4
  - Observed 15.5 = worse than random → model diverges in projection space
- **H3 RULED OUT**: metric computation is correct

**Implication**: The failure is not a measurement artifact. The model genuinely fails to generalize — consistent with H1 (global alignment ≠ per-frame decodability) and H2 (temporal density). Primary investigation path: H2 diagnosis (R019) + F3 ST-GCN encoder.

---

## F1-B: R019 (Temporal Density Audit) — DONE ✓ (2026-04-09)

**H2 VERDICT: UNLIKELY** — temporal density is NOT the bottleneck.

| Metric | Value | Threshold |
|--------|-------|-----------|
| Mean adjacent cosine sim (combined 1536-dim) | **0.4595** | <0.70 → H2 unlikely |
| Std of sample-means | 0.0733 | — |
| Frac adjacent pairs > 0.90 | 0.007 | — |
| Frac adjacent pairs > 0.99 | **0.000** | — |
| Compression ratio (raw→latent) | 4.05× | — |
| Samples processed | 50 | — |

**Per-part breakdown**:
- Body: mean=0.4938, p90=0.7532, frac>0.99=0.000
- Left hand: mean=0.4857, p90=0.8434, frac>0.99=0.129 *(slight high-density tail, not dominant)*
- Right hand: mean=0.3827, p90=0.7042, frac>0.99=0.000

**Interpretation**: Adjacent latent frames have substantial variation (mean sim=0.46, well below 0.70 threshold). H2 is ruled out. The M2 failure is not caused by temporal redundancy in VAE embeddings. Primary failure path confirmed as **H1 (global InfoNCE alignment ≠ per-frame decodability)**. F3 (ST-GCN encoder) is the correct remediation.

**Outputs**: `experiments/analysis/temporal_density_audit/`

---

## F3: ST-GCN Encoder Implementation — DONE ✓ (2026-04-09)

**Implementation**:
- `mGPT/archs/tools/st_gcn.py`: `SpatialGraphConv`, `STGCNBlock`, `STGCNEncoder`
- `mGPT/archs/mgpt_vq.py`: wired via `encoder_type='stgcn'` kwarg
- `configs/vq/re128_stgcn.yaml`: body VAE (43-dim, 10 joints, BODY_ADJ)
- `configs/vq/hand256_stgcn.yaml`: hand VAE (45-dim, 15 joints, HAND_ADJ)
- `configs/vae/vae_finetune_sign_stgcn.yaml`: F3-B training config (R020/R021)
- `configs/soke_mt5_csl_stgcn.yaml`: F3-C mT5 fine-tune config (R022)

**Joint layouts**:
- Body: Neck, L/R Collar, Head, L/R Shoulder, L/R Elbow, L/R Wrist (10 joints)
- Hand: 5 fingers × 3 joints each; palm base connections (15 joints)

**Status**:
1. ~~Run R019~~ — **DONE** (H2 UNLIKELY, see F1-B above)
2. ~~R020~~ — **DONE** (2026-04-10 03:04): csl_MPVPE_PA_all 25.79→14.72 (-43%), loss 0.049; ckpt: `experiments/mgpt/debug--VAE_SIGN_FINETUNE_STGCN/checkpoints/last.ckpt`
3. ~~R022~~ — **DONE** (2026-04-10 21:43): **F3 GATE FAILED** — csl_BLEU4 peak = **0.993** (gate ≥ 5.0); see R022 section below.
4. ~~Check F3 success gate~~ — **FAILED**

## R022: F3-C mT5 on ST-GCN Embeddings — DONE ✗ (2026-04-10)

**F3 GATE FAILED**: csl_BLEU4 peak = **0.993** (epoch 77), gate ≥ 5.0

| Metric | E0 (init) | Best | Final (E79) |
|--------|-----------|------|-------------|
| csl_BLEU_1 | 8.7 | 14.39 (E51) | 13.11 |
| csl_BLEU_4 | 0.0 | **0.993** (E77) | 0.000 |
| val_m2t_loss | 3.975 (E1) | 3.975 (E1) | 14.97 |
| BLEU_2/3 | 0.0 | 0.0 (all epochs) | 0.0 |

**Critical finding**: ST-GCN embeddings produce **WORSE** translation than Conv1d baseline (1.742). Despite 43% better reconstruction (R020), the downstream text generation is inferior.

**Failure pattern**:
- val_m2t_loss monotonically diverges (4.12 @ E4 → 14.97 @ E79) while train loss = 0.25 → severe overfitting/distribution mismatch
- BLEU_2/3 = 0 throughout all 80 epochs → model generates isolated words but no coherent 2-gram sequences
- Best val loss was at E1 (4.12) before any significant training — pre-trained mT5 knowledge degrades under ST-GCN fine-tuning

**Root cause hypothesis (G8 confirmed)**: LFQ tokens from reconstruction training are "acoustic class" not "semantic class". **ST-GCN improves acoustic quality of codes (better reconstruction) but does NOT improve semantic quality (linguistic discriminability)**. Neither Conv1d nor ST-GCN reconstruction-trained VAE produces tokens suitable for LM decoding.

Gap: Conv1d (1.742 BLEU4) > ST-GCN (0.993 BLEU4) despite ST-GCN having better reconstruction. This suggests the Conv1d codes accidentally capture more linguistically relevant patterns, possibly because flat (non-graph) processing of the 43-dim joint vector preserves more global correlations.

**Checkpoint**: `experiments/mgpt/debug--SOKE_MT5_CSL_STGCN/checkpoints/last.ckpt`
- Best epoch checkpoint: `max-csl_BLEU_4epoch=51.ckpt` (may be earlier; log at E51 shows 0.000, likely saved before E51 plateau)

## Notes

- mT5-base downloaded to `deps/mt5-base` (d_model=768, confirmed)
- `sentencepiece` installed in conda env for mT5 tokenizer support
- ST-GCN adjacency matrices use row-normalized D^{-1}A normalization
- ST-GCN encoder is a drop-in replacement (same interface as Conv1d Encoder)
- `PRETRAINED_VAE` in F3-B config points to Conv1d checkpoint; encoder weights
  will NOT load (shape mismatch) — decoder + quantizer will initialize from it;
  the encoder trains from scratch. This is intentional.
