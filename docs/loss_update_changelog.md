# Loss Update Changelog

本文档记录 LFQ 分支 VAE 训练损失的增量变化，便于协作者追踪 loss 设计演进。

---

## 2026-03-09

### Changed
- 将文档从日期命名迁移为语义命名：`loss_update_changelog.md`。
- 统一整理当前有效 loss 组合与配置字段。

### Current VAE Loss Set (`mGPT/losses/mgpt.py`)
- `recons_feature`
- `recons_velocity`
- `recons_fk_hand`
- `recons_accel_hand`
- `recons_accel_wrist_rel`
- `recons_contact`
- `vq_commit`

---

## 2026-03-02

### Added
- `recons_accel_hand`：手部 wrist-relative 关节二阶差分损失。
- `recons_accel_wrist_rel`：左右手腕相对向量二阶差分损失。

### Implementation Notes
- 二阶差分统一形式：`x_t - 2x_{t-1} + x_{t-2}`。
- 有效长度为 `length-2`，短序列自动跳过。

---

## 2026-03-01

### Added
- `recons_contact`：接触分类 BCE 监督（3 通道）。
- 数据侧接入 `gt_contact_labels [B,T,3]`。

### Channel Order
- `lhand_rhand`
- `lhand_face`
- `rhand_face`

### Dependency
- 需先运行接触标签预计算脚本：
  - `scripts/pipeline/precompute_contact_labels.py`
  - `scripts/pipeline/precompute_contact_labels_all.sh`

---

## 2026-02-28

### Added
- `recons_fk_hand`：基于 SMPL-X FK 的手部监督。

### FK Coordinate Convention
- 手部损失使用 wrist-relative 坐标，不使用绝对世界坐标。
- 目的：降低全局漂移影响，聚焦手部 articulation。

---

## 2026-02-27

### Added
- `VELOCITY_PART_WEIGHTS`：分部位速度权重（UPPER/HAND/FACE）。
- `PART_WEIGHTS`：分部位重建权重（UPPER/HAND/FACE）。

---

## 常用配置参考

示例配置：`configs/vae/vae_finetune_sign_lfq4_fkhand_body_pretrained.yaml`

- `LAMBDA_FEATURE=1.0`
- `LAMBDA_VELOCITY=0.3`
- `LAMBDA_FK_HAND=0.5`
- `LAMBDA_ACCEL_HAND=0.1`
- `LAMBDA_ACCEL_WRIST_REL=0.1`
- `LAMBDA_CONTACT=0.05`

