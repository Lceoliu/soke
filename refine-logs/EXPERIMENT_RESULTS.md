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

## R025: G11 连续嵌入 Oracle — DONE ✓ (2026-04-11)

**实验**：用 `encode_continuous()` 做 NN 检索（余弦相似度），测量 mT5 真实输入空间的理论上界。

| VAE | Oracle BLEU4 | Oracle BLEU1 | Mean NN Cosine | P90 NN Cosine |
|-----|-------------|-------------|---------------|--------------|
| Conv1d continuous | **1.890** | 15.588 | 0.8693 | 0.9169 |
| ST-GCN continuous | **1.652** | 15.133 | 0.8815 | 0.9288 |
| **mT5 baseline (R006)** | *1.742* | *16.83* | — | — |

**关键发现**：
1. **连续嵌入 oracle ≈ mT5 实际训练结果**（±0.15 BLEU4）→ mT5 已接近其输入嵌入所允许的理论上界
2. **平均 NN 余弦 = 0.87（很高！）**：从 2000 个训练样本中找到的"最近邻"余弦相似度高达 0.87，但翻译仍然错误（pred ≠ ref） → 不同手语句子的连续嵌入在余弦空间中高度密集，无法通过 NN 检索区分
3. **与 62% 分类器的矛盾**（重要分析）：cross-signer 分类器能以 62% 准确率区分 1024-way，但 oracle NN 仅给出 1.9 BLEU4。矛盾的解释：分类器是**经过训练**的决策边界，能从噪声中提取细粒度判别信息；而 oracle 用的是**原始余弦距离**，在高密度空间中信噪比太低。换言之：判别信息**存在**但**很难通过简单线性变换提取**。

**新诊断**（G11 更新）：
- ~~H1（全局对齐 ≠ 局部可解码性）~~ → 已更精确：原始嵌入空间句间密度太高，mT5 LoRA 微调已几乎达到上界
- ~~H4（数据量不足）~~ → 有待验证（R026），但即使增加数据，如果嵌入本身不可区分，增加训练样本帮助有限
- **新 H5（嵌入空间语义密度）**：连续嵌入的句间语义密度是核心瓶颈；需要让相同语义的手语在嵌入空间更近、不同语义的更远 → 需要序列级判别性预训练（如 InfoNCE 在序列级而非批次级）

**输出**：`experiments/analysis/g2_oracle_continuous/`

---

## R026: 数据量学习曲线 — DONE ✓ (2026-04-11)

**H4 验证**：mT5 在不同训练样本量下的 BLEU4 学习曲线。

| 数据量 | Best csl_BLEU4 | Best csl_BLEU1 | Final BLEU4 | BLEU_2/3 |
|-------|--------------|--------------|-------------|---------|
| 1k (1000) | 0.213 | 12.550 | 0.000 | 全 0 |
| 5k (5000) | 0.747 | 14.310 | 0.000 | 全 0 |
| 10k (10000) | 0.761 | 15.610 | 0.384 | 全 0 |
| 18k (R006 全量) | **1.742** | **16.830** | 1.185 | 有值 |

**关键发现**：
1. **5k→10k 近乎平台**（+0.014，几乎无增益），**10k→18k 出现跳跃**（0.761→1.742，×2.3）
2. **只有 18k 能生成连贯 2-gram 以上序列**（BLEU_2/3/4 全为 0 直到 18k） → 有效翻译存在数据量阈值（约 15k+ 样本）
3. **H4 部分成立**：数据量对 mT5 翻译有边际改善，但曲线非单调、非线性，且 18k 时 mT5(1.742) 已近 oracle 上界(1.890)

**R025 + R026 联合结论**：
- **嵌入层和数据量双重制约**：oracle(1.890) ≈ mT5(1.742) 说明 mT5 已近理论上界，增加数据只能缓慢提升上界
- **阈值效应**：~15k 样本以下，模型无法学到足够的 sign-text 统计规律来生成 2-gram 序列；阈值以上才有有效学习
- **根本解法方向**：提高连续 VAE 嵌入的句间语义区分能力（降低 mean NN cosine 从 0.87 到 <0.7），而非单纯增加数据量

**Log**：`experiments/r026_scale_run.log`

---

## Summary

- **7/22+ 实验完成**（R001/R002 M0/M1, R003/R006 M2, R019, R020/R021, R022 ✗, R023/R024 [前提有误], R025）
- M2 gate FAILED；F3 gate FAILED；G1 诊断前提有误
- **R025 关键结论**：连续嵌入 oracle ≈ mT5 实际结果（1.9 ≈ 1.7）；嵌入空间句间余弦密度 = 0.87，是真正瓶颈
- **R025 关键结论**：连续嵌入 oracle ≈ mT5（1.89 ≈ 1.74）；句间 NN cosine=0.87，嵌入空间高度密集
- **R026 关键结论**：数据量有边际帮助，但 mT5 已近 oracle 上界；有效翻译存在 ~15k 阈值效应
- **诊断完成**：问题根源 = 连续 VAE 嵌入句间语义密度过高（H5）+ 数据量不足（H4）双重制约
- **下一步方向**：序列级判别性预训练（降低 mean NN cosine 从 0.87 到 <0.7），同时考虑外部数据增强
- Ready for /auto-review-loop: YES（已有足够诊断证据）

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

---

## G1 Diagnostics — DONE ✓ (2026-04-11) ⚠️ 分析前提有误，见下方更正

### R023: G1-a Codebook Utilization

| VAE | Part | Utilization | Entropy (bits) | Entropy% | Dead Codes | Top-10 Conc. |
|-----|------|------------|---------------|---------|-----------|-------------|
| Conv1d | body | **100%** | 6.97 / 7.00 | **99.6%** | 0 | 11.5% |
| Conv1d | lhand | **100%** | 7.41 / 8.00 | **92.6%** | 0 | 24.9% |
| Conv1d | rhand | **100%** | 7.98 / 8.00 | **99.8%** | 0 | 5.4% |
| ST-GCN | body | **100%** | 6.98 / 7.00 | **99.8%** | 0 | 10.1% |
| ST-GCN | lhand | **100%** | 7.27 / 8.00 | **90.8%** | 0 | 27.9% |
| ST-GCN | rhand | **100%** | 7.97 / 8.00 | **99.7%** | 0 | 5.7% |

**数据有效**，但与 mT5 无关——见下方更正。

### R024: G1-c Oracle NN Decoder

| VAE | Oracle BLEU4 | Oracle BLEU1 | Exact Retrieval Rate |
|-----|-------------|-------------|---------------------|
| Conv1d | **0.195** | 10.21 | 0.0% |
| ST-GCN | **0.121** | 10.83 | 0.0% |

**数据有效**，但测量的是离散 LFQ token 空间的上界——见下方更正。

---

### ⚠️ G1 更正 (2026-04-11)

**前提错误：R023/R024 测量的是 LFQ 离散 token 空间，但 mT5 实际使用的是连续嵌入。**

mT5 的输入管线（见 `docs/research_history_zh.md` 第 6 章阶段三）：
```
pose (133-dim) → VAE.encode_continuous() → 连续隐变量 (1536-dim) → MLP projection → mT5 inputs_embeds
```
`encode_continuous()` 返回的是**量化前**的连续隐变量，**不经过 LFQ codebook**。R023/R024 调用的是 `vae.encode()`（离散 token），这对 mT5 的分析完全无关。

**R023 正确结论**（有限）：LFQ codebook 没有坍缩（对 Qwen 路径仍有参考价值）。  
**R024 正确结论**（有限）：离散 LFQ token 的检索上界几乎为零，但这不是 mT5 失败的原因——mT5 根本不看这些 token。

**原 "G1 Verdict" 中的结论已撤回**：
- ~~"Root cause confirmed: LFQ tokenizer is the bottleneck"~~ → 不适用于 mT5
- ~~"Decision → G2-a (continuous embeddings)"~~ → mT5 已经使用连续嵌入，这个"解法"已是现状

**已有证据表明连续 VAE 嵌入是有区分能力的**（见 `docs/research_history_zh.md` 第 8.5 节）：跨签名者 VAE 分类器在 1024-way 任务上测试准确率 62%，证明连续嵌入已编码身份/语义信息。

**真正的开放问题**：为什么具有区分能力的连续 VAE 嵌入，经过 MLP projection + mT5 微调后仍无法实现有效翻译？  
→ 候选假说：H1（全局 InfoNCE 对齐 ≠ 每帧可解码性）、H4（数据量不足，18k vs 100k+）  
→ 下一步见更新后的实验方案。
