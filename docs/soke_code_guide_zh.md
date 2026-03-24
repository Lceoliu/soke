# SOKE 代码导读与训练评估手册（LFQ 最新版）

更新日期：2026-03-09  
适用范围：当前 `LFQ` 分支（不再维护旧 VQ/RVQ 兼容训练路径）

---

## 1. 项目目标与当前主流程

SOKE 当前可分为两个阶段：

1. `STAGE=vae`：训练动作 tokenizer（LFQ VAE，分 body/lhand/rhand）
2. `STAGE=lm_pretrain` / `lm_instruct`：训练下游 mBART 生成器（text -> sign token）

核心入口：
- 训练：`train.py`
- 测试：`test.py`
- 配置解析：`mGPT/config.py`
- 主模型：`mGPT/models/mgpt.py`

---

## 2. 当前代码结构（按模块）

### 2.1 Tokenizer 主干（LFQ）

- 文件：`mGPT/archs/mgpt_vq.py`
- 量化器：`mGPT/archs/tools/quantize_lfq.py`
- 当前分支强制 `quantizer='lfq'`

编码解码结构：
1. `Encoder`（1D Conv + Resnet1D + 下采样）
2. `ResidualLFQ`（多层残差二值量化）
3. `Decoder`（Resnet1D + 上采样 + Conv）

关键实现更新：
- Decoder 上采样已改为 `mode='linear', align_corners=False`
- Decoder 支持 `return_hidden=True`，用于接触分类旁支

### 2.2 VAE 训练监督

- 文件：`mGPT/losses/mgpt.py`
- 当前支持（可配开关）：
  - `recons_feature`
  - `recons_velocity`（支持 `VELOCITY_PART_WEIGHTS`）
  - `recons_fk_hand`（wrist-relative）
  - `recons_accel_hand`
  - `recons_accel_wrist_rel`
  - `recons_contact`（BCE with logits）
  - `vq_commit`

### 2.3 数据模块

- 常规手语训练：`mGPT/data/H2S.py`（How2Sign/CSL/Phoenix）
- 大规模预训练：`mGPT/data/LargeMotion.py`

H2S 现已支持可选接触标签读取：
- `DATASET.H2S.USE_CONTACT_LABELS`
- `DATASET.H2S.CONTACT_DIR_NAME`

### 2.4 LM 与 token 适配

- 文件：`mGPT/models/mgpt.py`
- 对多层量化 token 使用共享层数：
  - `shared_Q = min(body_Q, lhand_Q, rhand_Q)`
- LM 侧输入采用展平方案：
  - `[B, T, Q] -> [B, T*Q]`

### 2.5 Checkpoint 兼容加载

- 文件：`mGPT/utils/load_checkpoint.py`
- 已加入旧 decoder key 的自动 remap（`decoder.model.* -> 新结构`）
- 可继续加载历史 VAE ckpt（包含老命名）

---

## 3. 数据格式与目录约定

### 3.1 动作特征

当前训练统一使用 133 维（SOKE 布局）：
- `0:30` upper body
- `30:75` left hand
- `75:120` right hand
- `120:133` face/jaw/expr

### 3.2 token 文件

`scripts/get_motion_code.py` 产物默认写到：
- `{DATA_ROOT}/{CODE_PATH}/{src}/{name}.npy`

LM 读取时支持多层 token（展平前可为 `[T, Q, P]` 或等价结构）。

### 3.3 接触标签文件

由 `scripts/pipeline/precompute_contact_labels.py` 预计算，单样本文件内容：
- `labels: [T, 3]`（`uint8`）
- 三个通道顺序：`lhand_rhand`, `lhand_face`, `rhand_face`

---

## 4. 最新改动摘要（协作者需知）

1. VAE 量化已切换到 LFQ；旧 VQ/RVQ 不再作为主路径维护。
2. Decoder 上采样从 nearest 改为 linear。
3. 新增 Contact Head（解码器隐层旁支）+ `LAMBDA_CONTACT`。
4. 新增 FK/加速度损失链：
   - `LAMBDA_FK_HAND`
   - `LAMBDA_ACCEL_HAND`
   - `LAMBDA_ACCEL_WRIST_REL`
5. 支持 body-only 预训练加载（hand/rhand 随机初始化）用于消融。
6. LM 下游新增一键脚本，自动训练 + BLEU eval + t2m 可视化。
7. 修复 `m2t` 测试阶段返回兼容问题，避免自动评估崩溃。
8. `scripts/get_motion_code.py` 已清理调试 shape 输出并修复模块导入路径。
9. **Resume 机制重构**（`mGPT/config.py` + `mGPT/utils/logger.py` + `train.py`）：
   - `TRAIN.RESUME` 和 `TRAIN.PRETRAINED` 语义明确化，两者互斥。
   - Resume 时自动从实验目录加载原始训练 config，仅允许白名单 key 覆盖。
   - Resume 时 LR scheduler `T_max` 自动与 `END_EPOCH` 同步。
   - Resume 时保留当前 `DATASET` 块，支持跨机器/挂载点恢复训练。
   - 新实验不再误用旧 config 中遗留的 `FOLDER_EXP`。
   - Resume 写入的 config 快照使用 `resumed_*` 前缀，不干扰后续 resume 的 config 查找。
   - `resume_config()` 仅在 train 阶段触发，test/eval 不受影响。
   - PyTorch 2.6+ `weights_only` 兼容：注册 OmegaConf 类型为安全全局类。
10. **Eval-only 自动 config 发现**（`scripts/pipeline/train_qwen_downstream_auto.sh`）：
    - `TRAIN_LM=0` + `EVAL_CKPT` 时自动从实验目录发现训练 config，无需手动指定。

---

## 5. 训练流程（推荐）

### 5.1 阶段一：MotionX 上 LFQ 预训练

```bash
conda activate soke
GPU_IDS=0,1,2,3,4,5,6,7 \
CFG=configs/vae/motionx_vae_pretrain_lfq4.yaml \
bash scripts/pipeline/train_vae_pretrain_lfq4_ddp.sh
```

默认脚本包含：
- DDP 启动
- 训练结束后自动 report + 重建可视化（可通过 `AUTO_POST=0` 关闭）

### 5.2 （可选）预计算接触标签

```bash
conda activate soke
DATASET=all SPLITS=train,val,test \
HOW2SIGN_ROOT=data/How2Sign \
CSL_ROOT=data/CSL-Daily \
PHOENIX_ROOT=data/Phoenix_2014T \
OUTPUT_DIR_NAME=contact_labels \
THRESHOLD=0.02 \
bash scripts/pipeline/precompute_contact_labels_all.sh
```

### 5.3 阶段一.5：手语微调（LFQ + FK + ACC + Contact）

```bash
conda activate soke
PRETRAINED_BODY_CKPT=/home/SOKE/experiments/mgpt/VAE_MOTIONX_PRETRAIN_LFQ4_C128H256/checkpoints/last.ckpt \
GPU_IDS=0,1,2,3,4,5,6,7 \
CFG=configs/vae/vae_finetune_sign_lfq4_fkhand_body_pretrained.yaml \
bash scripts/pipeline/train_vae_finetune_sign_lfq4_fkhand_body_pretrained_ddp.sh
```

---

## 6. 断点续训（Resume）

### 6.1 概念

训练入口 `train.py` 支持两种 checkpoint 加载方式（互斥）：

| 配置项 | 作用 | 加载内容 |
|--------|------|----------|
| `TRAIN.RESUME` | 断点续训 | 权重 + 优化器 + LR scheduler + epoch 等全部状态 |
| `TRAIN.PRETRAINED` | 权重初始化 | 仅加载模型权重，训练从 epoch 0 开始 |

两者不可同时设置，否则会报 `ValueError`。

### 6.2 Resume 机制详解（`mGPT/config.py: resume_config`）

当 `TRAIN.RESUME` 非空时，`resume_config()` 会：

1. **从 checkpoint 路径推断实验目录**：`<exp_dir>/checkpoints/last.ckpt` → `<exp_dir>/`
2. **加载原始训练 config**：读取 `<exp_dir>/config_*_train.yaml`（取最新的一份）
3. **应用安全覆盖**：只有白名单中的 key 允许从当前 CLI config 覆盖到原始 config：
   - 硬件相关：`USE_GPUS`, `DEVICE`, `NUM_NODES`, `PRECISION`, `ACCELERATOR`
   - 训练调度：`TRAIN.END_EPOCH`, `TRAIN.BATCH_SIZE`, `TRAIN.NUM_WORKERS`, `TRAIN.ACCUMULATE_GRAD_BATCHES`
   - 评估相关：`EVAL.BATCH_SIZE`, `EVAL.DISABLE_VAL`, `LOGGER.VAL_EVERY_STEPS` 等
4. **同步 LR scheduler**：如果 `TRAIN.END_EPOCH` 被覆盖，自动同步 `TRAIN.LR_SCHEDULER.params.T_max`
5. **保留当前 DATASET 块**：整个 `DATASET` 配置取自当前运行的 config，支持跨机器/挂载点恢复
6. **恢复 wandb run ID**：从 `<exp_dir>/wandb/latest-run/` 自动提取，实现日志无缝续接
7. **设置 FOLDER_EXP**：指向原实验目录，确保 checkpoint 和日志写回同一位置

### 6.3 通过脚本 Resume

**Qwen 下游脚本**（`scripts/pipeline/train_qwen_downstream_auto.sh`）：

```bash
RESUME_CKPT=experiments/mgpt/SOKE_QWEN_LM/checkpoints/last.ckpt \
END_EPOCH=200 \
GPU_IDS=0,1,2,3 \
bash scripts/pipeline/train_qwen_downstream_auto.sh
```

**mBART 下游脚本**（`scripts/pipeline/train_lm_downstream_auto.sh`）：

```bash
RESUME_CKPT=experiments/mgpt/SOKE_LFQ4_ACC_LM/checkpoints/last.ckpt \
END_EPOCH=200 \
bash scripts/pipeline/train_lm_downstream_auto.sh
```

脚本内部会将 `RESUME_CKPT` 设为 `cfg.TRAIN.RESUME`，并清空 `cfg.TRAIN.PRETRAINED`。

### 6.4 Config 快照命名

实验目录中保存的 config yaml 使用不同的命名前缀区分来源：

| 文件名模式 | 来源 |
|------------|------|
| `config_<timestamp>_train.yaml` | 首次训练时写入的原始 config |
| `resumed_<timestamp>_train.yaml` | 每次 resume 时写入的运行快照 |

`resume_config()` 只匹配 `config_*_train.yaml`，因此无论 resume 多少次，都始终基于原始训练 config 恢复，不会被中间 resume 快照污染。`resumed_*` 文件仅用于回溯每次 resume 的实际运行参数。

### 6.5 注意事项

- `TRAIN.RESUME` 必须指向 **checkpoint 文件**（如 `last.ckpt`），不是实验目录。
- Resume 时不需要手动指定 `EXP_NAME`——实验目录从 checkpoint 路径自动推断。
- 如果需要修改 `END_EPOCH`，LR scheduler 的 `T_max` 会自动同步，无需手动设置。
- 如果需要跨机器 resume，只需确保当前 config 中的数据集路径（`DATASET.H2S.ROOT` 等）正确即可。
- 新实验（非 resume）即使使用了包含 `FOLDER_EXP` 的旧 config yaml 作为基础，也会正确生成新的实验目录。
- `resume_config()` 仅在 `phase="train"` 时触发，test/demo/render 阶段不会被影响。

---

## 7. 下游 LM 训练与自动评估

### 7.1 一键脚本（推荐）

脚本：`scripts/pipeline/train_lm_downstream_auto.sh`

功能：
1. 可选 token 预生成（`PREPARE_TOKENS=1`）
2. LM 训练
3. 训练后自动跑 `m2t`（BLEU/ROUGE）
4. 自动跑 `t2m` 预测并抽样 mesh 可视化

示例：

```bash
conda activate soke
GPU_IDS=0,1,2,3,4,5,6,7 \
PRETRAINED_VAE=/home/SOKE/experiments/mgpt/VAE_SIGN_FINETUNE_LFQ4_ACC/checkpoints/last.ckpt \
EXP_NAME=SOKE_LFQ4_ACC_LM \
PREPARE_TOKENS=1 \
AUTO_EVAL_BLEU=1 \
AUTO_VIS=1 \
VIS_NUM_SAMPLES=12 \
VIS_CAM_Y=-0.5 \
bash scripts/pipeline/train_lm_downstream_auto.sh
```

仅做后处理（不训练）：

```bash
TRAIN_LM=0 \
EVAL_CKPT=experiments/mgpt/SOKE_LFQ4_ACC_LM/checkpoints/last.ckpt \
AUTO_EVAL_BLEU=1 AUTO_VIS=1 \
bash scripts/pipeline/train_lm_downstream_auto.sh
```

Qwen 下游脚本同理（只需指定 checkpoint，训练 config 自动从实验目录发现）：

```bash
TRAIN_LM=0 \
EVAL_CKPT=experiments/mgpt/SOKE_QWEN_FULL_RETRAIN_0323/checkpoints/last.ckpt \
AUTO_EVAL_BLEU=1 AUTO_SHOW_M2T=1 AUTO_VIS=1 \
bash scripts/pipeline/train_qwen_downstream_auto.sh
```

> **自动 config 发现**：当 `TRAIN_LM=0` 且 `EVAL_CKPT` 非空时，Qwen 脚本会自动从 checkpoint 所在实验目录查找 `config_*_train.yaml` 作为运行 config，无需手动指定 `CFG`。如果用户显式设置了 `CFG=xxx.yaml`，则以用户指定的为准。

### 7.2 关键输出目录

- 训练实验：`experiments/mgpt/<EXP_NAME>`
- 测试预测：`results/mgpt/<EXP_NAME>/<split>_rank_*`
- 下游自动报告：
  - `experiments/mgpt/<EXP_NAME>/auto_reports/downstream/bleu_eval_summary.txt`
  - `experiments/mgpt/<EXP_NAME>/auto_reports/downstream/visualization_summary.txt`
- 自动可视化：`experiments/mgpt/<EXP_NAME>/auto_vis/<timestamp>/videos`

---

## 8. 手动 eval（按任务）

### 8.1 m2t（BLEU/ROUGE）

```bash
python test.py --cfg <eval_cfg.yaml> --nodebug --task m2t --use_gpus 0 --device 0
```

要求：
- `model.params.task = m2t`
- `METRIC.TYPE = [M2TMetrics]`

### 8.2 t2m（DTW/动作指标）

```bash
python test.py --cfg <eval_cfg.yaml> --nodebug --task t2m --use_gpus 0 --device 0
```

要求：
- `model.params.task = t2m`
- `METRIC.TYPE = [TM2TMetrics]`

---

## 9. 协作改动建议（避免踩坑）

1. 改 `num_quantizers` 后，必须确认 LM token flatten/unflatten 路径一致。  
2. 分部位量化层数不一致时，默认使用 `shared_Q=min(...)`；不要直接假设三部分层数相同。  
3. 载入旧 ckpt 时若 key 不匹配，优先检查 `load_checkpoint.py` remap 是否生效。  
4. 新增 loss 名称建议保持 `prefix_name`（例如 `recons_xxx`），避免日志拆分逻辑报错。  
5. 大规模训练前先跑小样本 smoke（含 tokenize -> train -> test 一整条链路）。
6. Resume 时不要同时设置 `TRAIN.RESUME` 和 `TRAIN.PRETRAINED`，两者互斥。
7. 如需修改 resume 时允许覆盖的配置项，编辑 `mGPT/config.py` 中的 `_RESUME_SAFE_OVERRIDES`。

---

## 10. 相关文档

- 架构细节：`docs/model_construction_changelog.md`
- Loss 更新：`docs/loss_update_changelog.md`
- 大规模数据 pipeline：`docs/vae_scaling_pipeline_zh.md`
- 自定义数据接入：`docs/data_onboarding_guide_zh.md`
