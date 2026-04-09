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

- **1/18** must-run experiments completed (M0 sanity)
- Main result: pending (M1 → M2 gate)
- Ready for /auto-review-loop: NO — awaiting M2 results

## Next Steps (in order)

1. **Launch R002** (M1): full contrastive pre-training — ~4–8 GPU-hours
2. **Launch R003** (M2): contrastive + mT5 fine-tune, seed 1 — ~4 GPU-hours
3. **Check M2 gate**: does BLEU4(R003) > BLEU4(R006=~2.0)?
4. If gate passes: launch R004, R005 (seeds 2–3) + ablations R009–R012
5. If gate fails: diagnose — try (a) longer contrastive pre-train, (b) end-to-end contrastive regularizer

## Notes

- mT5-base downloaded to `deps/mt5-base` (d_model=768, confirmed)
- `sentencepiece` installed in conda env for mT5 tokenizer support
- Checkpoint key structure confirmed compatible: `norm.*`, `proj.0.*`, `proj.2.*`
