# 2026-03-02 Loss 更新说明（LFQ + FK Hand + Acceleration）

本文记录在 `LFQ` 分支上新增的两项加速度损失，实现目标是提升手语中快速动作（尤其手指爆发段与双手配合段）的还原能力。

参考文档风格：`docs/20260227_model_construction.md`。

## 1. 更新目标

在已启用 `FK hand loss` 的基础上，增加时间二阶差分监督：

1. 手部关节加速度损失（每只手，wrist-relative）
2. 双手相对向量加速度损失（global wrists relation）

核心思路：速度约束一阶动态，加速度约束二阶动态，用于强化快速变化段的重建。

## 2. 本次改动文件

- `mGPT/models/mgpt.py`
- `mGPT/losses/mgpt.py`
- `configs/vae/vae_finetune_sign_lfq4_fkhand_body_pretrained.yaml`

## 3. 数据流与实现细节

### 3.1 `mGPT/models/mgpt.py`

#### 3.1.1 新增开关参数

读取配置：

- `LOSS.LAMBDA_ACCEL_HAND`
- `LOSS.LAMBDA_ACCEL_WRIST_REL`

并新增聚合开关：

- `_use_hand_fk_supervision = (LAMBDA_FK_HAND > 0) or (LAMBDA_ACCEL_HAND > 0) or (LAMBDA_ACCEL_WRIST_REL > 0)`

含义：只要任何 hand/FK/accel 监督开启，就走 FK 前向分支。

#### 3.1.2 扩展 FK 输出

函数 `_compute_hand_fk_joints(...)` 由原来的 2 个返回值扩展为 4 个：

1. `joints_lhand`（左手 wrist-relative）
2. `joints_rhand`（右手 wrist-relative）
3. `l_wrist`（左手腕 global）
4. `r_wrist`（右手腕 global）

说明：

- `joints_lhand/joints_rhand` 继续用于 `recons_fk_hand` 和新加的 `recons_accel_hand`。
- `l_wrist/r_wrist` 用于新加的 `recons_accel_wrist_rel`。

#### 3.1.3 `rs_set` 新增字段

在 `train_vae_forward` 输出中新增：

- `wrist_l_rst`, `wrist_r_rst`
- `wrist_l_ref`, `wrist_r_ref`

用于 loss 模块计算双手相对向量加速度。

### 3.2 `mGPT/losses/mgpt.py`

#### 3.2.1 新增 loss 项

在 `stage == "vae"` 下新增：

- `recons_accel_hand`，权重 `LAMBDA_ACCEL_HAND`
- `recons_accel_wrist_rel`，权重 `LAMBDA_ACCEL_WRIST_REL`

两项都复用当前 reconstruction loss 类型（`l1`/`l2`/`l1_smooth`）与长度 mask 机制。

#### 3.2.2 手部关节加速度损失（wrist-relative）

输入：

- `fk_lhand_rst/ref`: `[B, T, J, 3]`
- `fk_rhand_rst/ref`: `[B, T, J, 3]`

二阶差分：

`a_t = x_t - 2*x_{t-1} + x_{t-2}`

分别计算左右手后拼接为 `[B, T-2, 2J, 3]`，再做重建损失。

有效长度：

- `acc_lengths = length - 2`
- 当 `max(acc_lengths) > 0` 时计算。

#### 3.2.3 双手相对向量加速度损失

输入：

- `wrist_l_rst/ref`, `wrist_r_rst/ref`: `[B, T, 1, 3]`

先构造双手相对向量（右减左）：

`v_t = wrist_r_t - wrist_l_t`

再做二阶差分：

`a_t = v_t - 2*v_{t-1} + v_{t-2}`

对 `a_t` 做重建损失，长度同样使用 `length - 2`。

## 4. 配置项更新

在 `configs/vae/vae_finetune_sign_lfq4_fkhand_body_pretrained.yaml` 增加：

- `LOSS.LAMBDA_ACCEL_HAND: 0.1`
- `LOSS.LAMBDA_ACCEL_WRIST_REL: 0.05`

当前默认是温和权重，建议先观察：

- `recons/accel_hand/train|val`
- `recons/accel_wrist_rel/train|val`
- `recons/fk_hand/train|val`

再做逐步调参。

## 5. 日志与命名

新增日志键：

- `recons/accel_hand/{split}`
- `recons/accel_wrist_rel/{split}`

命名符合现有 `loss2logname` 规则，不影响已有日志系统。

## 6. 工程稳定性说明

- 兼容当前 LFQ 训练流程，不改动 VAE 主干结构。
- 新增 loss 仅在对应 lambda 非零时生效。
- 当序列过短（`T < 3`）时，二阶差分 loss 自动跳过（由 `length-2` 判断）。

## 7. 训练使用建议（本阶段）

建议起步权重：

1. `LAMBDA_FK_HAND = 0.5`
2. `LAMBDA_ACCEL_HAND = 0.1`
3. `LAMBDA_ACCEL_WRIST_REL = 0.05`

若观察到动作抖动增大：

1. 先下调 `LAMBDA_ACCEL_HAND`（例如 `0.1 -> 0.05`）
2. 保持 `LAMBDA_ACCEL_WRIST_REL`，优先稳定双手关系
3. 再根据可视化结果微调 `LAMBDA_FK_HAND`

