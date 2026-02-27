# 2026-02-27 模型结构说明（LFQ 分支）

本文对应当前 `LFQ` 分支，目标是说明：
- 卷积如何实现
- 关键参数与总体架构
- Encoder 每一层输入输出 shape（含公式与示例）

## 1. 总体结构（VQVae in LFQ branch）

当前 tokenizer 主体在 `mGPT/archs/mgpt_vq.py`，结构为：

1. `preprocess`: `[B, T, F] -> [B, F, T]`
2. `Encoder`（1D Conv + ResNet1D + Downsample）
3. `quantize_in`（`1x1 Conv` 或 `Identity`）
4. `ResidualLFQ` 量化（多层）
5. `quantize_out`（`1x1 Conv` 或 `Identity`）
6. `Decoder`（ResNet1D + Nearest 上采样 + 1D Conv）
7. `postprocess`: `[B, F, T] -> [B, T, F]`

对应代码：
- `mGPT/archs/mgpt_vq.py:95`
- `mGPT/archs/mgpt_vq.py:156`
- `mGPT/archs/mgpt_vq.py:193`

## 2. 卷积实现细节

### 2.1 Encoder 主干卷积

在 `mGPT/archs/mgpt_vq.py:170` 开始：

- 首层：
  - `Conv1d(input_emb_width -> width, kernel=3, stride=1, padding=1)`
  - 时间长度不变
- 下采样层（重复 `down_t` 次）：
  - `Conv1d(width -> width, kernel=2*stride_t, stride=stride_t, padding=stride_t//2)`
  - 默认 `stride_t=2` 时为 `kernel=4, stride=2, padding=1`
  - 时间长度约减半
  - 后接 `Resnet1D(width, depth, dilation_growth_rate)`
- 末层：
  - `Conv1d(width -> output_emb_width, kernel=3, stride=1, padding=1)`
  - 时间长度不变

### 2.2 Resnet1D 内部卷积

`Resnet1D` 在 `mGPT/archs/tools/resnet.py`：

- 一个 `Resnet1D` 由 `n_depth` 个 `ResConv1DBlock` 组成（默认 `depth=3`）
- 每个 block 包含：
  - `Conv1d(n_in -> n_state, kernel=3, stride=1, padding=dilation, dilation=dilation)`
  - `Conv1d(n_state -> n_in, kernel=1, stride=1, padding=0)`
  - 残差连接 `x + block(x)`
- 默认 `dilation_growth_rate=3`，深度 3 时 dilation 为：
  - `1, 3, 9`

对应代码：
- `mGPT/archs/tools/resnet.py:45`
- `mGPT/archs/tools/resnet.py:46`
- `mGPT/archs/tools/resnet.py:75`

### 2.3 Decoder 卷积与上采样

在 `mGPT/archs/mgpt_vq.py:208` 开始：

- 首层：
  - `Conv1d(output_emb_width -> width, kernel=3, stride=1, padding=1)`
- 上采样块（重复 `down_t` 次）：
  - `Resnet1D(...)`
  - `Upsample(scale_factor=2, mode='nearest')`
  - `Conv1d(width -> width, kernel=3, stride=1, padding=1)`
- 尾部：
  - `Conv1d(width -> width, kernel=3, stride=1, padding=1)`
  - `ReLU`
  - `Conv1d(width -> input_emb_width, kernel=3, stride=1, padding=1)`

## 3. 当前常用 LFQ 配置参数

当前你新增并使用的 LFQ4 配置：

- Body tokenizer: `configs/vq/re128_lfq4.yaml`
  - `num_quantizers=4`
  - `code_num=128`
  - `code_dim=512`
  - `output_emb_width=512`
  - `down_t=2, stride_t=2, width=512, depth=3, dilation_growth_rate=3`
- Hand tokenizer: `configs/vq/hand256_lfq4.yaml`
  - `num_quantizers=4`
  - `code_num=256`
  - 其余主干参数同上

在微调配置中挂载方式：
- `configs/vae/vae_finetune_sign_lfq4.yaml:85`
- `configs/vae/vae_finetune_sign_lfq4.yaml:86`
- `configs/vae/vae_finetune_sign_lfq4.yaml:87`

## 4. Encoder 每层 shape 推导

下面以输入 `x: [B, T, F]` 说明。进入 Encoder 前先 `preprocess` 到 `[B, F, T]`。

记：
- `F = nfeats`（body=43, hand=45）
- `W = width`（默认 512）
- `E = output_emb_width`（默认 512）
- `D = down_t`（默认 2）
- `S = stride_t`（默认 2）

### 4.1 通用长度公式

`Conv1d` 时间长度公式：

`L_out = floor((L_in + 2p - d*(k-1) - 1)/s + 1)`

对于默认下采样卷积 `k=4, s=2, p=1, d=1`：

`L_out = floor(L_in / 2)`

重复 `D=2` 次后：

`T_enc = floor(floor(T/2)/2) = floor(T/4)`

### 4.2 分层表（默认参数）

1. 输入到 Encoder  
`[B, F, T]`

2. `Conv1d(F -> W, k3 s1 p1)`  
`[B, 512, T]`

3. `ReLU`  
`[B, 512, T]`

4. Down Block 1  
- `Conv1d(512 -> 512, k4 s2 p1)` -> `[B, 512, floor(T/2)]`
- `Resnet1D(depth=3, dilation=1/3/9)` -> `[B, 512, floor(T/2)]`

5. Down Block 2  
- `Conv1d(512 -> 512, k4 s2 p1)` -> `[B, 512, floor(T/4)]`
- `Resnet1D(depth=3, dilation=1/3/9)` -> `[B, 512, floor(T/4)]`

6. `Conv1d(512 -> E, k3 s1 p1)`  
`[B, 512, floor(T/4)]`

7. `quantize_in`  
- 若 `E == code_dim`：`Identity`
- 否则 `Conv1d(E -> code_dim, k1)`  
当前配置里 `E=512, code_dim=512`，所以 shape 不变：
`[B, 512, floor(T/4)]`

### 4.3 具体例子（T=64）

- Body 输入：`[B, 64, 43]`
  - preprocess 后：`[B, 43, 64]`
  - Encoder 输出：`[B, 512, 16]`
  - `encode` 输出 token：`[B, 16, 4]`（4 层 LFQ）
- Hand 输入：`[B, 64, 45]`
  - Encoder 输出同样：`[B, 512, 16]`
  - token：`[B, 16, 4]`

## 5. 关于时间长度对齐

Decoder 每个 upsample 块都会 `x2`，两次后回到 `4 * T_enc`。  
若输入 `T` 不是 4 的倍数，理论上会有边界差异。当前数据流程通常用 `UNIT_LEN=4`，因此长度对齐通常是稳定的。

## 6. 小结

- 当前卷积主干是典型的时序 1D Conv 编解码器：`Conv + Downsample + ResBlocks` 对应 `ResBlocks + Upsample + Conv`。
- Encoder 的空间（通道）固定到 `512`，时间维默认压缩到 `T/4`。
- LFQ 在该 latent 序列上做 4 层离散化，最终 token 形状为 `[B, T/4, 4]`（默认配置）。
