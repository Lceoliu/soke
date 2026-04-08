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

## M1: Full Contrastive Pre-train — PENDING

**Run**: R002 — Full contrastive pre-training on CSL-Daily train set (~17k samples)

**Command**:
```bash
python scripts/train_contrastive_pretrain.py \
    --cfg configs/soke_mt5_csl_m2t.yaml \
    --output_dir experiments/contrastive_pretrain_csl \
    --epochs 80 \
    --batch_size 256 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --proj_dim 768 \
    --temperature 0.07 \
    --seed 42 \
    --gpu 0 \
    --num_workers 8 \
    --cache_dir experiments/contrastive_pretrain_csl/cache \
    2>&1 | tee experiments/contrastive_pretrain_csl/train.log
```

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
