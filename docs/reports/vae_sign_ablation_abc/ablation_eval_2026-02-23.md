# VAE Sign Ablation A/B/C 评估报告（2026-02-23）

## 1. 报告产物位置

- A: `experiments/mgpt/VAE_SIGN_ABLATION_A_BASELINE_PRETRAINED_ALL/auto_reports/rvq_stage1/rvq_stage1_report.md`
- B: `experiments/mgpt/VAE_SIGN_ABLATION_B_BODY_PRETRAINED_HAND_SCRATCH/auto_reports/rvq_stage1/rvq_stage1_report.md`
- C: `experiments/mgpt/VAE_SIGN_ABLATION_C_SCRATCH_ALL/auto_reports/rvq_stage1/rvq_stage1_report.md`

每组报告内都包含：
- `figures/loss_curve.png`
- `figures/codebook_utilization_curve.png`
- `rvq_stage1_report.md`

## 2. 评估设置

- checkpoint 选择：各实验目录 `checkpoints/min-val_loss-epoch=*.ckpt`
- codebook 统计样本：2000
- 统计来源：训练日志中的 epoch 指标 + checkpoint 的 codebook 使用统计

## 3. 关键结果对比

### 3.1 Min-val checkpoint 对比（越低越好）

| Exp | ckpt epoch | loss_total | h2s MPVPE_PA_all | h2s MPJPE_PA_hand | csl MPVPE_PA_all | csl MPJPE_PA_hand | pho MPVPE_PA_all | pho MPJPE_PA_hand |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 109 | 0.2011 | 12.59 | 4.150 | 13.40 | 3.273 | 15.43 | 4.364 |
| B | 89  | 0.2375 | 12.48 | 3.857 | 13.93 | 3.101 | 15.95 | 4.157 |
| C | 89  | 0.1476 | 12.20 | 3.921 | 13.70 | 3.142 | 15.47 | 4.187 |

### 3.2 Final epoch 对比（epoch 119）

| Exp | loss_total | h2s MPVPE_PA_all | h2s MPJPE_PA_hand | csl MPVPE_PA_all | csl MPJPE_PA_hand | pho MPVPE_PA_all | pho MPJPE_PA_hand |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0.2007 | 12.56 | 4.249 | 13.44 | 3.310 | 15.38 | 4.465 |
| B | 0.2337 | 12.36 | 3.830 | 13.76 | 3.085 | 15.75 | 4.137 |
| C | 0.1447 | 12.21 | 3.897 | 13.66 | 3.124 | 15.52 | 4.180 |

### 3.3 RVQ Codebook 利用率（摘要）

- A（Pretrained All）：
  - 手部 L2-L4 的 effective utilization 约 `0.51~0.55`，明显偏低。
- B（Body Pretrained + Hand Scratch）：
  - 手部 L2-L4 effective utilization 约 `0.94~0.98`，最充分。
- C（Scratch All）：
  - 手部 L2-L4 effective utilization 约 `0.91~0.97`，也较高。

## 4. 结论

- 若目标优先是手部细节/语义：B 最优（3 个数据集的 hand 指标均最低）。
- 若目标优先是整体稳定与全身误差：C 更平衡，A 在 phoenix 全身指标上略好。
- A 的手部 codebook 有明显“可用但不充分激活”迹象，和你之前观察到的“手语语义丢失/平均化”是吻合的。

## 5. 建议

- 下一轮主线建议以 B 为基础继续（保留 hand scratch 路线），并在此基础上做手语语义强化（如 FK/关键手指约束/词级片段重加权）。
- 训练产物部署时，优先使用 `min-val_loss-epoch=*.ckpt`，而不是 `last.ckpt`。
