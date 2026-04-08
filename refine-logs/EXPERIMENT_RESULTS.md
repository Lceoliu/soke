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

## M2: Fine-tune Comparison — PENDING

**Runs**: R003–R007 — Contrastive pre-train + mT5 vs. direct mT5 fine-tune

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
