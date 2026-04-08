# Experiment Tracker

| Run ID | Milestone | Block | Purpose | System / Variant | Dataset | Split | Metrics | Priority | Status | Notes |
|--------|-----------|-------|---------|------------------|---------|-------|---------|----------|--------|-------|
| R001 | M0 | — | Sanity: contrastive pre-train loop correctness | Contrastive pre-train (100 CSL pairs) | CSL-Daily | train (100 samples) | InfoNCE↓, cosine_sim↑ | MUST | DONE | PASS ✓ train_loss 4.56→4.30 (-5.6%), val_pos_sim -0.085→+0.204 (+0.289). Ckpt: experiments/contrastive_pretrain_sanity/best_sign_proj.pt |
| R002 | M1 | — | Full contrastive pre-training | Contrastive pre-train (sign projection + text projection, InfoNCE) | CSL-Daily | train | InfoNCE↓, cosine_sim val distribution | MUST | TODO | Batch≥256; cache sign embeddings; 50–100 epochs; save best checkpoint |
| R003 | M2 | B1 | Main result seed 1 | Ours: contrastive pre-train → mT5-base LoRA fine-tune | CSL-Daily | test | BLEU4, ROUGE-L, WER | MUST | TODO | Use R002 checkpoint; LoRA rank=64, alpha=128, pw=0.5 |
| R004 | M2 | B1 | Main result seed 2 | Ours: contrastive pre-train → mT5-base LoRA fine-tune | CSL-Daily | test | BLEU4, ROUGE-L, WER | MUST | TODO | Different random seed |
| R005 | M2 | B1 | Main result seed 3 | Ours: contrastive pre-train → mT5-base LoRA fine-tune | CSL-Daily | test | BLEU4, ROUGE-L, WER | MUST | TODO | Different random seed |
| R006 | M2 | B1 | Direct fine-tune baseline seed 1 | Baseline: mT5-base LoRA direct fine-tune (existing exp:003 re-run) | CSL-Daily | test | BLEU4, ROUGE-L, WER | MUST | DONE | exp:003 result; re-run for matched hyperparams if needed |
| R007 | M2 | B1 | Direct fine-tune baseline seed 2 | Baseline: mT5-base LoRA direct fine-tune | CSL-Daily | test | BLEU4, ROUGE-L, WER | MUST | TODO | |
| R008 | M2 | B1 | PHOENIX transfer | Ours (R003 checkpoint) evaluated on PHOENIX | PHOENIX-2014T | test | BLEU4, ROUGE-L, WER | MUST | TODO | No PHOENIX fine-tuning — zero-shot transfer from CSL checkpoint |
| R009 | M3 | B2 | Ablation: matched compute | mT5-base LoRA, same total steps as R003 (no contrastive pre-train) | CSL-Daily | test | BLEU4, ROUGE-L | MUST | TODO | Steps = contrastive_steps + finetune_steps |
| R010 | M3 | B2 | Ablation: frozen projection | Contrastive pre-train → freeze projection → LoRA fine-tune only | CSL-Daily | test | BLEU4, ROUGE-L | MUST | TODO | Tests whether projection adaptation during LM fine-tune matters |
| R011 | M3 | B2 | Ablation: MSE alignment instead of InfoNCE | Replace InfoNCE with MSE(sign_proj, text_proj); then mT5 fine-tune | CSL-Daily | test | BLEU4, ROUGE-L | MUST | TODO | Tests whether contrastive structure is needed or just any alignment |
| R012 | M3 | B2 | Ablation: no pre-training (random projection) | Random projection init → mT5 fine-tune | CSL-Daily | test | BLEU4, ROUGE-L | MUST | TODO | Equivalent to idea:003 (mT5 direct) but with same projection arch |
| R013 | M4 | B3 | Scale check | mT5-large LoRA direct fine-tune (no contrastive pre-train) | CSL-Daily | test | BLEU4, ROUGE-L, WER | MUST | TODO | Tests if model scale alone fixes the problem |
| R014 | M4 | B3 | Scale check (XL, if budget allows) | mT5-XL LoRA direct fine-tune | CSL-Daily | test | BLEU4, ROUGE-L, WER | NICE | TODO | ~4× memory vs. base; run only if M2 successful and compute available |
| R015 | M5 | B5 | Alignment geometry | Cosine sim distribution: sign vs. text embedding pre/post contrastive | CSL-Daily | train+val | cosine_sim distribution, t-SNE | MUST | TODO | Post-hoc on frozen R002 vs. random init projections |
| R016 | M5 | B5 | Qualitative examples | Sample 10 test examples, compare baseline vs. ours vs. GT | CSL-Daily | test | qualitative | MUST | TODO | Include failure cases; analyze error types |
| R017 | M5 | B5 | Cross-signer BLEU4 | Evaluate R003 on per-signer held-out subset | CSL-Daily | test (cross-signer) | BLEU4 per signer | NICE | TODO | Use same signer split from exp:005 |
| R018 | M6 | B4 | Architecture: contrastive + Qwen m2t | Contrastive pre-train → Qwen-0.5B LoRA fine-tune, m2t only | CSL-Daily | test | BLEU4, ROUGE-L | NICE | TODO | Tests if fix is seq2seq-specific or general; run after B1 success |

---

## Decision Gates

| Gate | After Run(s) | Condition | Action if Pass | Action if Fail |
|------|-------------|-----------|----------------|----------------|
| G-M2 | R003–R007 | BLEU4(R003-5) > BLEU4(R006) at ≥2/3 seeds | Proceed to M3–M5 | Stop; diagnose: try (a) longer contrastive pre-train, (b) end-to-end fine-tune with contrastive regularizer, (c) data volume hypothesis |
| G-M3 | R009–R012 | R009 (matched compute) < R003 (full method) | Novelty isolated; proceed to M4 | Reframe: compute is the variable; design new ablation |
| G-M4 | R013 | R013 (mT5-large direct) < R003 (ours, mT5-base) | Simplicity defended | Add scale + contrastive variant to understand interaction |

---

## Current Status Summary

**Completed (from prior experiments):**
- R006 (baseline mT5 direct fine-tune) ≈ DONE (exp:003 result — BLEU4 ~2)

**Next to run:**
1. R001 — M0 sanity (< 1 GPU-hour)
2. R002 — M1 full contrastive pre-train
3. R003 — first main result seed
