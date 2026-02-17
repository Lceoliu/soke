# 大规模 VAE 预训练 + 手语微调 Pipeline（MOTION-X 等）

本文档对应以下新增文件：
- 数据模块：`mGPT/data/LargeMotion.py`
- 预处理脚本：
  - `scripts/pipeline/build_raw_manifest_from_scan.py`
  - `scripts/pipeline/preprocess_vae_corpus.py`
  - `scripts/pipeline/compute_mean_std_stream.py`
- Bash 脚本：
  - `scripts/pipeline/build_raw_manifest.sh`
  - `scripts/pipeline/prepare_large_vae_data.sh`
  - `scripts/pipeline/train_vae_pretrain_ddp.sh`
  - `scripts/pipeline/train_vae_pretrain_rvq4_ddp.sh`
  - `scripts/pipeline/train_vae_finetune_sign_ddp.sh`
- 配置：
  - `configs/vae/large_vae_pretrain.yaml`
  - `configs/vae/motionx_vae_pretrain_rvq4.yaml`
  - `configs/vae/vae_finetune_sign.yaml`
  - `configs/vq/re96_rvq4.yaml`
  - `configs/vq/hand192_rvq4.yaml`

---

## 1. 设计目标

这条 pipeline 解决四件事：
1. 把多来源动作数据统一成仓库可训练格式（133 维，兼容 SOKE 的 decoupled VAE）
2. 支持超大规模数据处理（流式、并行、manifest 驱动）
3. 支持多卡 DDP 训练（沿用现有 Lightning + `--device`）
4. 先做大规模 VAE 预训练，再迁移到手语数据集微调

---

## 2. 数据输入规范（raw-manifest）

预处理脚本接收 `jsonl`，每行一个样本，示例：

```json
{"path": "/data/MOTION-X/train/xxx.npy", "source": "motionx", "split": "train", "format": "npy", "layout": "smplx179", "name": "xxx"}
{"path": "/data/SomeSet/clip_001", "source": "some", "split": "train", "format": "smplx_pkl_dir", "name": "clip_001"}
```

字段说明：
- `path`: 样本路径（npy 文件 或 一段序列目录）
- `source`: 数据源名（用于后续追踪）
- `split`: `train/val/test`
- `format`:
  - `npy`: 直接读 npy
  - `smplx_pkl_dir`: 目录内逐帧 `.pkl/.pt`（含 `smplx_*` key）
- `layout`（仅 `npy`）:
  - `smplx179`: 预处理时转换到 133
  - `soke133`: 已是 133
- `name`: 可选，默认根据路径推断并加 hash

---

## 3. Step-by-step 使用

## 3.1 构建 raw-manifest（扫描目录）

```bash
bash scripts/pipeline/build_raw_manifest.sh \
  --scan "source=motionx,split=train,format=npy,path=/data/MOTION-X/train,glob=**/*.npy,layout=smplx179" \
  --scan "source=motionx,split=val,format=npy,path=/data/MOTION-X/val,glob=**/*.npy,layout=smplx179" \
  --scan "source=amass,split=train,format=npy,path=/data/AMASS,glob=**/*.npy,layout=soke133" \
  --output data/large_vae/raw_manifest.jsonl \
  --sort
```

## 3.2 统一预处理 + 流式统计 mean/std

```bash
RAW_MANIFEST=data/large_vae/raw_manifest.jsonl \
OUTPUT_ROOT=data/large_vae/processed \
PROCESSED_MANIFEST=data/large_vae/processed_manifest.jsonl \
MEAN_PATH=data/large_vae/mean.pt \
STD_PATH=data/large_vae/std.pt \
NUM_WORKERS=16 \
TARGET_FPS=24 \
MIN_FRAMES=40 \
MAX_FRAMES=0 \
bash scripts/pipeline/prepare_large_vae_data.sh
```

输出：
- 统一样本：`data/large_vae/processed/{source}/{split}/*.npy`
- 训练清单：`data/large_vae/processed_manifest.jsonl`
- 统计量：`data/large_vae/mean.pt`, `data/large_vae/std.pt`

---

## 4. 大规模 VAE 预训练（多卡）

默认配置：`configs/vae/large_vae_pretrain.yaml`

```bash
GPU_IDS=0,1,2,3,4,5,6,7 \
NUM_NODES=1 \
CFG=configs/vae/large_vae_pretrain.yaml \
bash scripts/pipeline/train_vae_pretrain_ddp.sh
```

说明：
- 脚本会自动把 `GPU_IDS` 转为 `--device 0 1 ...`
- 配置中使用 `mGPT.data.LargeMotion.LargeMotionDataModule`
- 模型是 **纯 VAE stage**（`TRAIN.STAGE=vae`）

### 4.1 RVQ 版本（细节增强）

新增 RVQ 配置：
- `configs/vae/motionx_vae_pretrain_rvq4.yaml`
- body 使用 `vq.re96_rvq4`
- hand/rhand 使用 `vq.hand192_rvq4`

```bash
GPU_IDS=0,1,2,3,4,5,6,7 \
NUM_NODES=1 \
CFG=configs/vae/motionx_vae_pretrain_rvq4.yaml \
bash scripts/pipeline/train_vae_pretrain_rvq4_ddp.sh
```

说明：
- 量化器为 `quantizer='rvq_ema_reset'`
- 每个分支 `num_quantizers=4`，每层都保留 EMA+reset（含 dead code reset）
- 预设 `EVAL/TEST.BATCH_SIZE=1` 且 `VAL_EVERY_STEPS` 很大，避免大规模预训练时评估阶段 OOM
- 当前 RVQ 已完整支持 VAE 训练/重建评估；LM token 协议暂未扩展到 RVQ 多级 token

---

## 5. 手语数据集微调（多卡）

默认配置：`configs/vae/vae_finetune_sign.yaml`

```bash
GPU_IDS=0,1,2,3 \
PRETRAINED_CKPT=experiments/mgpt/VAE_LARGE_PRETRAIN/checkpoints/last.ckpt \
CFG=configs/vae/vae_finetune_sign.yaml \
bash scripts/pipeline/train_vae_finetune_sign_ddp.sh
```

如果你只想联合 `How2Sign + CSL-Daily`（不含 Phoenix），可直接使用：

```bash
GPU_IDS=0,1,2,3 \
PRETRAINED_CKPT=experiments/mgpt/VAE_MOTIONX_PRETRAIN/checkpoints/min-how2sign_MPJPE_PA_handepoch=259.ckpt \
CFG=configs/vae/vae_finetune_h2s_csl.yaml \
bash scripts/pipeline/train_vae_finetune_h2s_csl_ddp.sh
```

如果你发现重建中“手指细节好，但手臂幅度/手与身体相对位置不稳”，可尝试 v2 配置：
- `configs/vae/vae_finetune_h2s_csl_v2.yaml`
- 变化：启用 `LAMBDA_VELOCITY=0.3`，并对 133 维特征做分段加权（`UPPER/HAND/FACE`）

```bash
GPU_IDS=0,1,2,3 \
PRETRAINED_CKPT=experiments/mgpt/VAE_MOTIONX_PRETRAIN/checkpoints/min-how2sign_MPJPE_PA_handepoch=259.ckpt \
CFG=configs/vae/vae_finetune_h2s_csl_v2.yaml \
bash scripts/pipeline/train_vae_finetune_h2s_csl_ddp.sh
```

可选路径覆盖（不改 yaml 直接传环境变量）：
- `DATASET_NAME`（例如：`how2sign_csl`、`how2sign_csl_phoenix`）
- `H2S_ROOT`
- `CSL_ROOT`
- `PHOENIX_ROOT`
- `MEAN_PATH`
- `STD_PATH`

脚本会在启动前做快速数据检查（CSV/GZIP 标注、mean/std 文件）。

---

## 6. 关键实现点（适合超大数据）

1. Manifest 驱动：避免把数据组织逻辑写死在代码中
2. 样本级惰性加载：训练时每次只读当前 batch 的 npy
3. 预处理并行：`ProcessPoolExecutor` 多进程转码
4. 统计流式：`compute_mean_std_stream.py` 不一次性加载全量数据
5. 训练多卡：通过 `--use_gpus + --device` 触发 Lightning DDP

---

## 7. 注意事项

1. 该 pipeline 预设目标特征是 133 维（SOKE 手语特征布局）
2. 若你的源数据不是 SMPL-X 179 或 SOKE 133，需要先做额外转换
3. `configs/vae/large_vae_pretrain.yaml` 中 `VAL_EVERY_STEPS` 设置很大，默认近似关闭验证
4. 若你需要在大语料上开启 MR 指标验证，请保证数据与 `feats2joints` 假设一致（SMPL-X 133）
