# 自定义数据接入配置指南（视频 + SMPL-X）

本指南面向协作者：你已经有自己的视频，以及从视频提取出的 SMPL-X 结果，希望快速接入本仓库训练（尤其 VAE 预训练/微调）。

目标是两件事：

1. 不改或少改代码，把新数据稳定接入。
2. 明确哪些文件必须长什么样，避免训练时才报错。

---

## 1. 先理解两条接入路径

本仓库当前有两条推荐路径：

1. 路径 A（推荐，零改代码）：`LargeMotionDataModule`  
适合大规模预训练，直接吃 `manifest + npy(133)`。

2. 路径 B（用于手语微调）：`H2SDataModule`  
适合和 How2Sign/CSL/Phoenix 一起训练，推荐把你的数据整理成 **CSL 风格**（`poses + csl_clean.{train|val|test}`）。

建议：先走路径 A 验证数据质量和 VAE 可训练，再走路径 B 做手语微调。

---

## 2. 输入数据硬性格式要求

### 2.1 支持的原始输入格式（预处理脚本）

脚本 `scripts/pipeline/preprocess_vae_corpus.py` 支持：

1. `format=smplx_pkl_dir`  
一个样本是一个目录，目录内是逐帧 `.pkl/.pt`。  
每帧必须有以下 key（见 `mGPT/data/humanml/load_data.py`）：
- `smplx_root_pose` (3)
- `smplx_body_pose` (63)
- `smplx_lhand_pose` (45)
- `smplx_rhand_pose` (45)
- `smplx_jaw_pose` (3)
- `smplx_shape` (10)
- `smplx_expr` (10)

拼接后是 179 维。

2. `format=npy`  
每个样本一个 `npy`，形状 `[T, C]`，支持：
- `layout=smplx179`
- `layout=motionx322`
- `layout=soke133`

### 2.2 训练使用的标准特征维度

VAE 训练主流程默认使用 **133 维**（上肢+双手+jaw/expr）：
- `0:30` upper body
- `30:75` left hand
- `75:120` right hand
- `120:133` jaw/expr

`preprocess_vae_corpus.py` 会把 `smplx179` / `motionx322` 自动转成 `soke133`。

---

## 3. 路径 A：零改代码接入（推荐）

这条路径最稳，特别适合“自定义视频+SMPL-X”先做 VAE 预训练。

### 3.1 准备环境

```bash
conda activate soke
```

### 3.2 构建 raw manifest

用扫描脚本把样本写成 `jsonl`（每行一个样本）：

```bash
bash scripts/pipeline/build_raw_manifest.sh \
  --scan "source=myset,split=train,format=smplx_pkl_dir,path=/path/to/myset/train,glob=*" \
  --scan "source=myset,split=val,format=smplx_pkl_dir,path=/path/to/myset/val,glob=*" \
  --scan "source=myset,split=test,format=smplx_pkl_dir,path=/path/to/myset/test,glob=*" \
  --output data/myset/raw_manifest.jsonl \
  --sort
```

如果你已经是 `npy`：

```bash
bash scripts/pipeline/build_raw_manifest.sh \
  --scan "source=myset,split=train,format=npy,path=/path/to/myset_npy/train,glob=**/*.npy,layout=smplx179" \
  --scan "source=myset,split=val,format=npy,path=/path/to/myset_npy/val,glob=**/*.npy,layout=smplx179" \
  --output data/myset/raw_manifest.jsonl \
  --sort
```

### 3.3 统一预处理并计算 mean/std

```bash
RAW_MANIFEST=data/myset/raw_manifest.jsonl \
OUTPUT_ROOT=data/myset/processed \
PROCESSED_MANIFEST=data/myset/processed_manifest.jsonl \
MEAN_PATH=data/myset/mean.pt \
STD_PATH=data/myset/std.pt \
NUM_WORKERS=16 \
TARGET_FPS=24 \
MIN_FRAMES=40 \
MAX_FRAMES=0 \
bash scripts/pipeline/prepare_large_vae_data.sh
```

输出：
- `data/myset/processed/.../*.npy`（统一 133 维）
- `data/myset/processed_manifest.jsonl`
- `data/myset/mean.pt`, `data/myset/std.pt`（133 维）

### 3.4 配置训练文件

复制 `configs/vae/motionx_vae_pretrain_rvq4.yaml` 为你的配置，例如：

```bash
cp configs/vae/motionx_vae_pretrain_rvq4.yaml configs/vae/myset_vae_pretrain_rvq4.yaml
```

至少改这几项：

1. `NAME`
2. `DATASET.target: mGPT.data.LargeMotion.LargeMotionDataModule`
3. `DATASET.LARGE.MANIFEST: data/myset/processed_manifest.jsonl`
4. `DATASET.LARGE.MEAN_PATH: data/myset/mean.pt`
5. `DATASET.LARGE.STD_PATH: data/myset/std.pt`
6. `DATASET.NFEATS: 133`

说明：`LargeMotion` 的 `mean/std` 必须与训练特征同维（133 维）。

### 3.5 启动预训练

```bash
GPU_IDS=0,1,2,3,4,5,6,7 \
CFG=configs/vae/myset_vae_pretrain_rvq4.yaml \
bash scripts/pipeline/train_vae_pretrain_rvq4_ddp.sh
```

---

## 4. 路径 B：接入手语微调（CSL 风格，最小改代码）

如果你希望把自定义数据和 How2Sign/CSL/Phoenix 一起用于 `H2SDataModule` 微调，建议把你的数据组织成 **CSL 风格**。

### 4.1 目录结构

```text
data/MySign/
  poses/
    sample_0001/
      000000.pkl
      000001.pkl
      ...
    sample_0002/
      ...
  csl_clean.train
  csl_clean.val
  csl_clean.test
  mean.pt
  std.pt
```

其中 `csl_clean.*` 是 `gzip pickle`，内容是 `list[dict]`，每条至少包含：
- `name`: 对应 `poses/<name>/`
- `text`: 文本（VAE 阶段可先给占位文本）

### 4.2 配置接入

以 `configs/vae/vae_finetune_sign_rvq4.yaml` 为例，修改：

1. `DATASET.H2S.DATASET_NAME: csl`（只用你的数据）
2. `DATASET.H2S.CSL_ROOT: data/MySign`
3. `DATASET.H2S.MEAN_PATH: data/MySign/mean.pt`
4. `DATASET.H2S.STD_PATH: data/MySign/std.pt`

如果要和现有多数据源混训，可保留 `how2sign_csl_phoenix`，再把你的数据并到 CSL 集合里。

注意：`H2SDataModule` 读取的 `mean/std` 应是 **179 维统计量**（内部会裁成 133）。

### 4.3 启动微调

```bash
PRETRAINED_CKPT=/home/SOKE/experiments/mgpt/VAE_MOTIONX_PRETRAIN_RVQ4_C128H256/checkpoints/last.ckpt \
GPU_IDS=0,1,2,3,4,5,6,7 \
CFG=configs/vae/vae_finetune_sign_rvq4.yaml \
bash scripts/pipeline/train_vae_finetune_sign_rvq4_ddp.sh
```

---

## 5. 数据自检清单（强烈建议）

训练前至少检查以下项目：

1. 帧数  
- 每个样本帧数建议 >= `MIN_FRAMES`（默认 40）。

2. 关键字段齐全  
- 每帧 `pkl/pt` 是否含 7 个 `smplx_*` key。

3. 维度一致  
- 输入是 179/322/133 三类之一；预处理后应为 133。

4. split 有效  
- `train/val/test` 是否都有样本；否则 eval 阶段会 fallback 或失真。

5. mean/std 维度对应  
- `LargeMotion` 用 133 维 mean/std。  
- `H2SDataModule` 配置文件中的 mean/std 按当前实现应使用 179 维统计量。

---

## 6. 常见问题与定位

1. `Feature dim mismatch`  
- 先看 manifest 里对应 `path` 的 `npy` 形状是否是 `[T,133]`。

2. `missing key smplx_*`  
- 说明 SMPL-X 导出字段不完整，需在提取脚本补齐 7 个 key。

3. DDP sanity-check 阶段指标报 `*_count` 不存在  
- 自定义 `source` 不是 `how2sign/csl/phoenix` 时，`LargeMotion` 会映射到 `SOURCE_DEFAULT`。  
- 建议 `SOURCE_DEFAULT: how2sign`（配置里已有）。

4. 训练能跑但重建异常  
- 用 `scripts/visualize_smplx_raw_mesh.py` 先看原始 SMPL-X 质量，再看 tokenizer 重建，分离“数据问题”与“模型问题”。

---

## 7. 最小可复现模板（给协作者）

协作者只需替换三类路径即可完整跑通：

1. 原始数据路径：`/path/to/myset/{train,val,test}`
2. 输出路径：`data/myset/*`
3. 配置路径：`configs/vae/myset_vae_pretrain_rvq4.yaml`

推荐先跑一轮 smoke（少量样本）：

1. `raw_manifest` 只放 100~500 个样本。
2. `END_EPOCH` 设 5~10。
3. `GPU_IDS` 先用 1~2 卡确认无误，再扩到 8 卡。

---

## 8. 相关代码位置（便于二次开发）

1. 原始数据读取与 179->133 裁剪：`mGPT/data/humanml/load_data.py`
2. 手语 VAE 数据集拼装：`mGPT/data/humanml/dataset_m_vq_sign.py`
3. 大规模数据 DataModule：`mGPT/data/LargeMotion.py`
4. 预处理入口：`scripts/pipeline/preprocess_vae_corpus.py`
5. 扫描建 manifest：`scripts/pipeline/build_raw_manifest_from_scan.py`
6. 流式统计 mean/std：`scripts/pipeline/compute_mean_std_stream.py`

