# SOKE 仓库代码导读与训练评估手册

本文档基于当前仓库代码整理，目标是让你可以按仓库真实实现完成：
1. 数据准备与格式对齐
2. 两阶段训练（Tokenizer + 生成器）
3. 推理、可视化与评估

> 如果你需要在 **MOTION-X 等超大规模数据上先预训练 VAE，再在手语数据上微调**，请配合阅读：`docs/vae_scaling_pipeline_zh.md`

---

## 1. 项目整体架构

SOKE 是一个典型的两阶段流程：

1. 阶段一（`STAGE=vae`）：训练离散化动作 Tokenizer（VQ-VAE）
2. 阶段二（`STAGE=lm_pretrain/lm_instruct`）：训练文本到手语 token 的多头自回归生成器

核心入口：
- 训练入口：`train.py`
- 测试入口：`test.py`
- 配置解析与实例化：`mGPT/config.py`
- 数据模块构建：`mGPT/data/build_data.py`
- 模型构建：`mGPT/models/build_model.py`
- 统一模型壳：`mGPT/models/mgpt.py`（`MotionGPT`）

配置驱动方式：
- 所有模块通过 YAML 中 `target + params` 反射实例化
- 通过 `OmegaConf` 合并 `configs/default.yaml` + 实验配置（如 `configs/deto.yaml` / `configs/soke.yaml`）

---

## 2. 从原始数据到训练输入：处理流程与目录规范

> 重点：本仓库训练直接使用“每帧 SMPL-X 参数 pkl”，不是直接从 RGB 视频端到端训练。  
> 如果你只有原始视频，需要先在仓库外完成人体/手部拟合，得到每帧 3D 参数文件。

### 2.1 原始单帧数据格式（每个 pkl）

代码在 `mGPT/data/humanml/load_data.py` 中固定读取以下 key：

- `smplx_root_pose` (3)
- `smplx_body_pose` (63)
- `smplx_lhand_pose` (45)
- `smplx_rhand_pose` (45)
- `smplx_jaw_pose` (3)
- `smplx_shape` (10)
- `smplx_expr` (10)

拼接后是 179 维。

### 2.2 训练前特征裁剪规则（179 -> 133）

仓库统一做了两次裁剪：

1. 去下半身：`clip_poses = clip_poses[:, (3 + 3*11):]`  
2. 去 shape 参数：`concat([:-20], [-10:])`

最终得到 **133 维**特征。

在模型里，133 维通常被视为：
- `0:30`：上肢主体相关
- `30:75`：左手（45）
- `75:120`：右手（45）
- `120:133`：其余（13，通常含 jaw/expr）

### 2.3 多数据源目录与注释文件

#### How2Sign
- 文本标注：`data/How2Sign/{split}/re_aligned/how2sign_realigned_{split}_preprocessed_fps.csv`
- 姿态目录：`data/How2Sign/{split}/poses/{SENTENCE_NAME}/..._3D.pkl`

#### CSL-Daily
- 标注：`data/CSL-Daily/csl_clean.{train|val|test}`（gzip pickle）
- 姿态目录：`data/CSL-Daily/poses/{name}/*.pkl`

#### Phoenix-2014T
- 标注：`data/Phoenix_2014T/phoenix14t.{train|dev|test}`（代码中 `val` 对应 `dev`）
- 姿态目录：`data/Phoenix_2014T/{name}/*.pkl`

### 2.4 采样与长度处理

- How2Sign 如果 fps > 24，会做均匀下采样到 24fps
- 长度 < 最小阈值：均匀插值拉长到最小长度
- 长度 > 最大阈值：均匀采样到最大长度
- 其余：按 `UNIT_LEN` 对齐并居中裁剪

主要参数在 `configs/soke.yaml` / `configs/deto.yaml`：
- `DATASET.H2S.MAX_MOTION_LEN`
- `DATASET.H2S.MIN_MOTION_LEN`
- `DATASET.H2S.UNIT_LEN`

### 2.5 归一化统计量（mean/std）

在 `H2SDataModule` 中从配置读取：
- `DATASET.H2S.MEAN_PATH`
- `DATASET.H2S.STD_PATH`

并应用与特征相同的裁剪规则后用于归一化。

---

## 3. 数据格式说明（训练、token、预测结果）

### 3.1 DataLoader 批字段（核心）

`humanml3d_collate` 统一产出：
- `motion`: `B x T x C`
- `length`: 每条序列长度列表
- `text`: 文本列表（文本任务）
- `tasks`: 指令模板（LM 阶段）
- `src`: 数据源（`how2sign/csl/phoenix`）
- `name`: 样本 id

### 3.2 token 文件格式（`scripts/get_motion_code.py` 产物）

输出到：`{data_root}/{CODE_PATH}/{src}/{name}.npy`

常见形状：
- body-only: `[1, T_code]`
- body+hand: `[1, T_code, 2]`
- body+lhand+rhand: `[1, T_code, 3]`

训练 LM 时读取 `np.load(...)[0]`，即 `T_code x heads`。

### 3.3 测试预测文件格式

若 `TEST.SAVE_PREDICTIONS=True`，测试会保存：
- `results/mgpt/{EXP}/{split}_rank_{r}/{name}.pkl`
- 字段：`feats_rst`, `feats_ref`, `text`

同时保存每样本指标：
- `results/mgpt/{EXP}/{split}_rank_{r}/test_scores.json`

---

## 4. 训练前准备（环境与依赖）

### 4.1 Python 环境

```bash
conda create -n soke python=3.10 -y
conda activate soke
pip install -r requirements.txt
```

说明：
- 优先尝试终端中是否存在 `soke` 环境，若不存在，再考虑创建。
- 当前终端中 `python` 可能未建立别名，建议优先用 `python3`。

### 4.2 模型与外部依赖

1. SMPL/SMPL-X 人体模型放到：`deps/smpl_models`
2. mBART 模型放到：`deps/mbart-h2s-csl-phoenix`
3. 下载 evaluator：`prepare/download_t2m_evalutors.sh`
4. 若用 T5 分支，下载 T5：`prepare/prepare_t5.sh`

---

## 5. 开始训练：完整操作步骤

## 5.1 阶段一：训练 Tokenizer（DETO）

配置：`configs/deto.yaml`（`TRAIN.STAGE: vae`）

```bash
python3 train.py --cfg configs/deto.yaml --nodebug
```

或（README 方式）

```bash
python3 -m train --cfg configs/deto.yaml --nodebug
```

阶段一测试：

```bash
python3 test.py --cfg configs/deto.yaml --nodebug
```

产物：
- 实验目录：`experiments/mgpt/DETO`（按 `NAME` 和 logger 规则）
- checkpoint：`experiments/mgpt/DETO/checkpoints/last.ckpt`

## 5.2 中间步骤：离线提取 motion token

README 写法是 `python -m get_motion_code`，但仓库脚本实际在 `scripts/get_motion_code.py`。

推荐命令：

```bash
python3 scripts/get_motion_code.py --cfg configs/soke.yaml --nodebug
```

该脚本会：
1. 强制 `cfg.TRAIN.STAGE = "token"`
2. 加载 `TRAIN.PRETRAINED_VAE`
3. 把每个样本编码为 token 并保存到 `DATASET.CODE_PATH`

## 5.3 阶段二：训练生成器（SOKE）

配置：`configs/soke.yaml`（`TRAIN.STAGE: lm_pretrain`）

先确认：
- `TRAIN.PRETRAINED_VAE` 指向阶段一 tokenizer ckpt
- `DATASET.CODE_PATH` 指向上一步 token 输出目录
- `DATASET.H2S.ROOT/CSL_ROOT/PHOENIX_ROOT` 路径正确

启动训练：

```bash
python3 train.py --cfg configs/soke.yaml --nodebug
```

也可用带 NCCL 诊断的脚本：

```bash
bash start_train.sh configs/soke.yaml
```

## 5.4 训练命令常用参数

`mGPT/config.py` 支持关键 CLI：
- `--cfg`: 主配置文件
- `--cfg_assets`: 资产路径配置（默认 `configs/assets.yaml`）
- `--use_gpus`: 设置 `CUDA_VISIBLE_DEVICES`
- `--batch_size`: 覆盖 `TRAIN.BATCH_SIZE`
- `--device`: 覆盖 `DEVICE` 列表
- `--num_nodes`: 多机节点数
- `--task`: 测试任务（如 `t2m/m2t`）
- `--nodebug`: 关闭 debug

注意：
- 训练时未传 `--nodebug` 可能进入 debug 逻辑（会改名字、降低验证间隔、WandB 离线）
- 测试阶段代码会强制 `DEBUG=False`

---

## 6. 关键参数作用（按配置文件）

### 6.1 `TRAIN` 段
- `STAGE`: 决定走 VAE 还是 LM 训练分支
- `PRETRAINED_VAE`: LM 阶段加载 tokenizer 权重
- `PRETRAINED` / `RESUME`: 恢复训练/测试 checkpoint
- `BATCH_SIZE`, `NUM_WORKERS`, `END_EPOCH`
- `OPTIM`, `LR_SCHEDULER`

### 6.2 `DATASET` 段
- `target`: 数据模块类型（SOKE 用 `mGPT.data.H2S.H2SDataModule`）
- `H2S.DATASET_NAME`: 控制混合哪些数据源（如 `how2sign_csl_phoenix`）
- `CODE_PATH`: 离线 token 根目录名
- `TASK_PATH`: 指令模板路径（可覆盖默认模板）
- `MEAN_PATH/STD_PATH`: 归一化统计量

### 6.3 `model.params` 段
- `motion_vae`: body tokenizer 配置（如 `vq.re96`）
- `hand_vae_cfg` / `rhand_vae_cfg`: 手部 tokenizer
- `lm`: 语言模型配置（SOKE 用 mBART multi-head）
- `task`: 当前任务（常见 `t2m`）

### 6.4 `METRIC` 与 `TEST`
- `METRIC.TYPE`: 启用哪类指标（`MRMetrics`, `TM2TMetrics`, `M2TMetrics`）
- `TEST.REPLICATION_TIMES`: 重复测试次数
- `TEST.SAVE_PREDICTIONS`: 是否落盘预测结果

---

## 7. 推理与评估

## 7.1 文本到手语（t2m）推理

```bash
python3 test.py --cfg configs/soke.yaml --task t2m
```

默认行为：
- 若 `TEST.CHECKPOINTS` 为空，自动取 `experiments/.../checkpoints/last.ckpt`
- 按 `TEST.REPLICATION_TIMES` 重复评估并输出均值

## 7.2 评估指标说明

### VAE 阶段（`MRMetrics`）
- 关注重建质量
- 主要为 MPJPE/MPVPE（含 PA 与非 PA），按 `how2sign/csl/phoenix` 分别统计

### LM 阶段 t2m（`TM2TMetrics`）
- 主要为 DTW 对齐后的关节误差
- 分 `body/lhand/rhand` 与不同数据源统计

### m2t（`M2TMetrics`）
- 文本生成指标：BLEU-1..4 与 ROUGE-L

## 7.3 结果输出位置

- 训练日志与 ckpt：`experiments/mgpt/{NAME}`
- 测试结果与样本预测：`results/mgpt/{NAME}`

---

## 8. 可视化方法

### 8.1 快速 mesh 可视化

```bash
python3 vis_mesh.py --cfg configs/soke.yaml --demo_dataset csl
```

用途：
- 读取测试结果对比 baseline / ours
- 输出视频对比与可选 mesh 文件

### 8.2 并行可视化

```bash
python3 vis_mesh_parallel.py --cfg configs/soke.yaml --demo_dataset csl
```

用途：
- 多进程并行处理样本
- 适合批量渲染

### 8.3 单样本 tokenizer 重建检查

```bash
python3 scripts/tokenize_reconstruct_one.py \
  --cfg configs/soke.yaml \
  --pose_dir <你的单样本帧目录> \
  --tokenizer_ckpt experiments/mgpt/vae/checkpoints/tokenizer.ckpt
```

Mesh 视频版：

```bash
python3 scripts/tokenize_reconstruct_mesh_one.py \
  --cfg configs/soke.yaml \
  --pose_dir <你的单样本帧目录> \
  --tokenizer_ckpt experiments/mgpt/vae/checkpoints/tokenizer.ckpt
```

### 8.4 Blender 高质量渲染

```bash
python3 vis_blender.py
```

需先配置 `BlenderToolBox/` 和 blender 相关环境。

---

## 9. 训练与调试建议

1. 先跑通 `deto.yaml` 的小批量训练，确认 VAE 可收敛
2. 再跑 `scripts/get_motion_code.py`，抽查 `CODE_PATH` 下 token 是否完整
3. 最后启动 `soke.yaml` 训练
4. 多卡训练优先用 `start_train.sh`，便于定位 NCCL 问题
5. 若结果异常，优先检查：
- `PRETRAINED_VAE` 是否正确
- `CODE_PATH` 是否与 tokenizer 版本匹配
- mean/std 与数据域是否一致
- `DATASET_NAME` 与根目录路径是否对应

---

## 10. 与 README 的一个实践差异

README 给出：

```bash
python -m get_motion_code --cfg configs/soke.yaml --nodebug
```

但当前仓库脚本在 `scripts/get_motion_code.py`，推荐直接：

```bash
python3 scripts/get_motion_code.py --cfg configs/soke.yaml --nodebug
```

---

## 11. Scaling 入口（新增）

针对超大规模数据（MOTION-X 等）与“VAE 预训练 -> 手语微调”需求，仓库已新增完整 pipeline，详见：

- `docs/vae_scaling_pipeline_zh.md`

对应核心文件：

- 数据模块：`mGPT/data/LargeMotion.py`
- 预处理脚本：`scripts/pipeline/preprocess_vae_corpus.py`
- 流式统计：`scripts/pipeline/compute_mean_std_stream.py`
- 训练脚本：`scripts/pipeline/train_vae_pretrain_ddp.sh`、`scripts/pipeline/train_vae_finetune_sign_ddp.sh`
- 配置：`configs/vae/large_vae_pretrain.yaml`、`configs/vae/vae_finetune_sign.yaml`
