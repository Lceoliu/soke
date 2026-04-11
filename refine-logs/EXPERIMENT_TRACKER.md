# Experiment Tracker

| Run ID | Milestone | Block | Purpose | System / Variant | Dataset | Split | Metrics | Priority | Status | Notes |
|--------|-----------|-------|---------|------------------|---------|-------|---------|----------|--------|-------|
| R001 | M0 | — | Sanity: contrastive pre-train loop correctness | Contrastive pre-train (100 CSL pairs) | CSL-Daily | train (100 samples) | InfoNCE↓, cosine_sim↑ | MUST | DONE | PASS ✓ train_loss 4.56→4.30 (-5.6%), val_pos_sim -0.085→+0.204 (+0.289). Ckpt: experiments/contrastive_pretrain_sanity/best_sign_proj.pt |
| R002 | M1 | — | Full contrastive pre-training | Contrastive pre-train (sign projection + text projection, InfoNCE) | CSL-Daily | train | InfoNCE↓, cosine_sim val distribution | MUST | DONE | PASS ✓ train_loss 6.08→3.54 (−41.7%), val_loss 5.70→4.02 (−29.5%), acc_s2t 0.7%→27.7% (top-1/512). T=0.07→0.061. Ckpt: experiments/contrastive_pretrain_csl/best_sign_proj.pt (7.1MB) |
| R003 | M2 | B1 | Main result seed 1 | Ours: contrastive pre-train → mT5-base LoRA fine-tune | CSL-Daily | val | BLEU4, ROUGE-L, WER | MUST | DONE | 80ep, 4 GPUs. Best csl_BLEU4=1.778 (val#68), best csl_BLEU1=17.44; final E79: csl_BLEU4=0.800, csl_BLEU1=15.97. Gate G-M2 FAILED. |
| R004 | M2 | B1 | Main result seed 2 | Ours: contrastive pre-train → mT5-base LoRA fine-tune | CSL-Daily | test | BLEU4, ROUGE-L, WER | MUST | TODO | Different random seed |
| R005 | M2 | B1 | Main result seed 3 | Ours: contrastive pre-train → mT5-base LoRA fine-tune | CSL-Daily | test | BLEU4, ROUGE-L, WER | MUST | TODO | Different random seed |
| R006 | M2 | B1 | Direct fine-tune baseline seed 1 | Baseline: mT5-base LoRA direct fine-tune | CSL-Daily | val | BLEU4, ROUGE-L, WER | MUST | DONE | 80ep, 4 GPUs (re-run matched hyperparams). Best csl_BLEU4=1.742 (val#261), best csl_BLEU1=16.83; final E79: csl_BLEU4=1.185, csl_BLEU1=16.19. |
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

| R019 | F1-B | — | Temporal density audit | Within-sample adjacent cosine_sim of VAE embeddings (50 CSL train samples) | CSL-Daily | train (50 samples) | cosine_sim distribution | MUST | DONE | H2 UNLIKELY ✓: mean adj sim=0.4595 (< 0.70 threshold), frac>0.99=0.000; lhand p90=0.84 but overall fine. Compression=4.05×. Output: experiments/analysis/temporal_density_audit/ |
| R020 | F3-B | F3 | Re-train body VAE with ST-GCN encoder | ST-GCN body VAE (43-dim, 10 joints, body adj) | how2sign_csl_phoenix | train/val | MRMetrics (recon) | MUST | DONE | Finished 2026-04-10 03:04. Final loss=0.049 (E119). csl_MPVPE_PA_all: 25.79→14.72 (-43%), csl_MPJPE_hand: 42.39→24.32 (-43%). Checkpoint: experiments/mgpt/debug--VAE_SIGN_FINETUNE_STGCN/checkpoints/last.ckpt |
| R021 | F3-B | F3 | Re-train hand VAEs with ST-GCN encoder | ST-GCN hand VAE (45-dim, 15 joints, hand adj) — body+hands co-trained in R020 | how2sign_csl_phoenix | train/val | MRMetrics (recon) | MUST | DONE | Co-trained with R020 (same run). Best hand ckpt: min-csl_MPJPE_PA_handepoch=104.ckpt |
| R022 | F3-C | F3 | M2 re-run: mT5 fine-tune on ST-GCN embeddings | mT5-base LoRA on ST-GCN VAE embeddings (R020/R021 checkpoint) | CSL-Daily | val/test | BLEU4, ROUGE-L, WER | MUST | DONE ✗ | F3 GATE FAILED: peak csl_BLEU4=0.993 (gate≥5.0). WORSE than Conv1d baseline 1.742. val_m2t_loss diverges 4.12→14.97. G8 confirmed: reconstruction tokens ≠ semantic tokens. |

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
