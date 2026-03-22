# Qwen Overfit 诊断方案与 TODO

本文档用于定位当前 Qwen 下游训练中 `t2m / m2t / mc` 的真实问题来源。目标不是继续猜测，而是把现象、证据、实验和判定标准串成一条可执行路径。

## 1. 当前事实

### 1.1 当前框架
- 上游是 LFQ-VAE，将动作离散为：
  - body: `<motion_id_*>`
  - lhand: `<hand_id_*>`
  - rhand: `<rhand_id_*>`
- 下游是 `Qwen2.5-3B`，decoder-only，统一词表，单一 `lm_head`。
- 输入格式采用 interleaved sign token：
  - `<sign> B1 L1 R1 B2 L2 R2 ... </sign>`
- 当前已经启用：
  - LoRA
  - `embed_tokens` 解冻
  - `lm_head` 解冻
  - differential learning rate

### 1.2 已确认无误的部分
- `embed_tokens` 与 `lm_head` 均处于可训练状态。
- optimizer 已对参数分组：
  - backbone = `1x base lr`
  - `embed_tokens` = `10x base lr`
  - `lm_head` = `10x base lr`
- `m2t` 的 label mask 正确：
  - 只监督 `<text> ... </text>` 内部的 text token
  - `sign` 区域和 task prefix 不参与 loss
- 新增 token 的输入 embedding 与 `lm_head` 输出行都已验证会收到梯度并发生更新。

### 1.3 最终结论

经过 `overfit1 / 4 / 12`、`synthetic probe`、`rhand-only`、`noaug`、`teacher-forced`、`first-token prob` 等实验后，当前结论已经收敛：

1. **Qwen m2t 主链本身是能学会的。**
   - `synthetic probe` 可以稳定 `4/4`；
   - 修复后真实 `m2t overfit4` 已可成功；
   - `m2t overfit12` 也已成功。

2. **之前的 overfit 失败，主因不是“Qwen 不会学”，而是工程链路中的多处不一致/污染。**

3. **真正起决定作用的根本原因有三类：**
   - 训练/评估使用了不同 token 来源；
   - 评估/诊断路径没有正确处理 `lengths`，把 padding 一起送进编码/生成；
   - 所谓 overfit 训练实际上还带着随机 token-drop 增强。

4. **“长 new token 序列 + 短 text supervision 天然学不会”这个解释不成立。**
   - 因为在去掉链路错误后，真实长 sign token 序列也可以 overfit；
   - synthetic 长前缀 probe 也能 overfit。

### 1.4 根本原因

#### 根因 1：训练/评估 token 来源不一致

训练阶段 `lm_pretrain/lm_instruct` 使用的是 **预计算 `.npy` token**：
- `mGPT/data/humanml/dataset_t2m_cb.py`
- `mGPT/models/mgpt.py:train_lm_forward`

而旧的 `val_m2t_forward()` / `inspect_m2t_predictions.py` 走的是 **原始 `feat133` -> VAE 重新编码** 路径：
- `mGPT/data/humanml/dataset_t2m_eval.py`
- `mGPT/models/mgpt.py:val_m2t_forward`

这会导致：
- 模型训练时看到的是预计算 token；
- 验证/导出时看到的是重新编码 token；
- 两者不完全一致时，生成结果会被错误归因为“模型没学会”。

#### 根因 2：长度处理错误，把 padding 当真实 token/pose 用了

这个问题有两层：

1. **旧的 `val_m2t_forward()` 在重新编码前没有按 `lengths[i]` 截断 raw motion**
   - padded frame 一起进入 VAE；
   - 编出来的 sign token 被污染。

2. **旧的 `diagnose_qwen_overfit.py` free-run `m2t` 没传 `lengths`**
   - `[T_max, 3]` 的 padded token 直接整段参与 prompt 构造；
   - 末尾的 padding token 被当成真实 sign token。

这类 bug 会直接把本该成功的 overfit 结果拖成失败。

#### 根因 3：overfit 实验默认带着随机 token-drop augmentation

真实训练集 `Text2MotionDatasetCB` 里原本有：
- 以一定概率从 token 序列头/尾删掉一整组 token

这意味着：
- 训练看到的输入并不是固定 4 条样本；
- 所谓 overfit 其实是在带随机扰动的数据族上训练；
- 这会显著削弱最小过拟合实验的诊断意义。

关闭增强后的实验结果已经证明，这一步会明显改善坍缩。

### 1.5 修复经验

本轮修复可以总结成下面几条工程经验：

1. **overfit 诊断必须保证训练路径和评估路径吃的是同一种 token。**
   - 训练用预计算 token，评估/导出也必须优先用预计算 token；
   - 不能一边训 cache token，一边测 VAE 在线重编码 token。

2. **所有 `m2t` 生成路径都必须显式传入 `lengths`。**
   - 不能依赖 padded batch 的形状；
   - 不能让 `[T_max]` 或 `[T_max,3]` 直接整段参与 prompt 构造。

3. **不能用 dtype 区分“raw feature” 和 “precomputed token”。**
   - 因为 collate 可能把 token 也转成 `float`；
   - 正确做法是按 shape/最后一维特征数区分。

4. **overfit 配置必须关闭随机增强。**
   - 否则实验在逻辑上就不是“固定样本记忆测试”。

5. **自动导出脚本必须和训练 split 对齐。**
   - `split=train` 就应该走 `train_dataloader()`；
   - 不能默认绕回 `test_dataloader()` 或 raw-feature eval path。

### 1.6 历史诊断记录

下面的内容保留为诊断过程中的历史记录，用于追溯问题是如何一步步收敛出来的。

### 1.6.1 历史 overfit 结果

#### `m2t-only overfit12`
- 训练 loss 已非常低，约在 `5e-5 ~ 1e-4`
- 但生成结果仍然明显错误，表现为：
  - 输出塌缩到少数训练句子
  - 样本间串扰
  - 之前还出现过 sign token 泄漏到文本输出中

典型现象：
- `你们好！ -> 对不起！`
- `对不起！ -> 我`
- `你叫什么名字？ -> 你是老师吗？`
- `他是谁？ -> <rhand_id_160> <rhand_id_96>你们好！`

#### `t2m-only overfit12`
- 训练 loss 也很低，约在 `2e-3 ~ 1e-2`
- 但生成动作与 GT 仍然明显对不上
- 说明：
  - 训练目标本身在优化
  - 但自由生成或后续解析没有和训练结果一致地工作

#### `m2t-only overfit1`
- 在修复 generation mask 之后，`overfit1` 已经可以正确生成目标文本。
- 典型样例：

```json
{"name": "S000000_P0000_T00", "src": "csl", "gt": "你们好！", "pred": "你们好！", "length": 52}
```

- 这说明：
  - 当前 `m2t` 训练目标并非完全失效；
  - 单样本条件下，Qwen + unified vocab + current formatter 是可以工作的；
  - 之后的问题不再是“1 条样本都学不会”，而是“从 1 条扩到 4 条时发生模式坍缩”。

#### `m2t-only overfit4`
- 在修复 generation mask 之后，`overfit4` 的自由生成结果如下：

```json
{"name": "S000000_P0000_T00", "src": "csl", "gt": "你们好！", "pred": "你们好！", "length": 52}
{"name": "S000001_P0000_T00", "src": "csl", "gt": "对不起！", "pred": "你们好！", "length": 40}
{"name": "S000002_P0000_T00", "src": "csl", "gt": "没关系！", "pred": "你们好！", "length": 40}
{"name": "S000003_P0000_T00", "src": "csl", "gt": "谢谢！", "pred": "你们好！", "length": 40}
```

- 这说明：
  - `overfit4` 不是随机失败，而是明显塌缩到单一高频句子 `你们好！`；
  - 失败模式是“首 token 判别即塌缩”，不是长序列后半段才漂移。

#### `m2t-only overfit4`（重跑后，`experiments/overfit_RE`）
- 在修复 `220` token 污染、重跑 `overfit4 m2t` 后，自由生成结果为：

```json
{"name":"S000000_P0000_T00","gt":"你们好！","pred":"你们好！"}
{"name":"S000001_P0000_T00","gt":"对不起！","pred":"没关系"}
{"name":"S000002_P0000_T00","gt":"没关系！","pred":"没关系！"}
{"name":"S000003_P0000_T00","gt":"谢谢！","pred":"你们好！"}
```

- 这说明：
  - 当前 `m2t overfit4` 已不是“完全失败”；
  - 但仍然只有 `2/4` 命中，存在稳定的类别塌缩。
- 配套的 teacher-forced 分析结果：
  - `mean_teacher_forced_acc = 0.8333`
  - 两条失败样本在 teacher forcing 下也已经从 **第 0 个文本 token** 开始出错。
- 结论：
  - 这不是单纯的 free-run 自回归漂移；
  - 当前失败点是 **首个语义 token 的条件判别**。

#### `m2t-only overfit4`（1000 epoch，`experiments/overfit_RE/...RE1k`）
- 将同一实验拉长到 `1000 epoch` 后，自由生成结果变为：

```json
{"name":"S000000_P0000_T00","gt":"你们好！","pred":"你们好！"}
{"name":"S000001_P0000_T00","gt":"对不起！","pred":"对不起！"}
{"name":"S000002_P0000_T00","gt":"没关系！","pred":"你们好！"}
{"name":"S000003_P0000_T00","gt":"谢谢！","pred":"你们好！"}
```

- 依然只有 `2/4` 正确。
- 首 token 概率分析显示：
  - `对不起！`：`P(对不起)=0.7253`，已学会；
  - `没关系！`：`P(没关系)=1.46e-6`，而 `P(你们)=0.9999`；
  - `谢谢！`：`P(谢谢)=4.56e-4`，而 `P(你们)=0.9954`。
- 这说明：
  - 继续堆 epoch 不会自然修复问题；
  - 错误样本已经被模型**高置信度**吸到 `你们` 这个错误吸引子上；
  - 当前问题不是“训练不够久”，而是输入判别边界没有被正确拉开。

#### `t2m-only overfit1/4`
- `t2m` 当前最关键的问题已经定位为 **训练 prompt 与推理 prompt 不一致**。
- 训练时 `t2m` 的 formatter 直接按 token id 手工拼接：
  - `<t2m> <text> TEXT </text> <sign>`
- 推理时 `_make_t2m_prompt()` 之前却走了字符串再 tokenizer 的路径：
  - `"<t2m> <text> {text} </text> <sign>"`
- 这会引入额外空格 token。
- `prompt parity` 已抓到这个问题：
  - `t2m prefix_match = 0`
  - `m2t prefix_match = 1`
- 典型 mismatch：
  - train ids 开头：`[151665, 151670, ...]`
  - infer ids 开头：`[151665, 220, 151670, 220, ...]`
- 直接后果：
  - `t2m` 生成 token 长度会异常短；
  - 后续 `_decode_generated_motion_parts()` 把 `len(body_tokens) <= 1` 的结果回填成全零动作；
  - 自动可视化中的 `*_pred.npy` 出现 `(1,133)` 全零张量。

### 1.6.2 历史阶段性判断
仅根据 loss 和生成结果，最合理的判断不是“模型完全没学会”，而是：

1. `m2t overfit1` 已经证明 teacher forcing 与自由生成在单样本上可以对齐；
2. `m2t overfit4` 表明从 1 条扩到 4 条后，模型在首 token 上即发生模式坍缩；
3. `t2m` 当前存在明确实现 bug，即训练-推理 prompt mismatch；
4. 问题更像是“训练-推理不一致 / 生成约束实现错误 / 少样本下输入可分性不足”，而不是简单的“轮数不够”。

### 1.6.3 sign token 相似度分析（4 条样本）
- 分析目录：
  - `experiments/overfit_RE/SOKE_QWEN_CSL_OVERFIT4_M2T_RE/sign_token_similarity/summary.md`
- 样本：
  - `你们好！`
  - `对不起！`
  - `没关系！`
  - `谢谢！`

关键观察：

1. 在严格时间对齐层面，这 4 条样本并不相同：
  - 所有 pair 的 `aligned_group = 0.0`
  - 说明不存在“整个 sign token 序列完全一样”的低级错误。

2. 但在 namespaced bag-of-tokens 层面，出现了非常强的近邻团簇：
  - `你们好！` vs `没关系！`: cosine `0.8490`
  - `你们好！` vs `谢谢！`: cosine `0.8481`
  - `没关系！` vs `谢谢！`: cosine `0.8089`
  - 而 `对不起！` 与其他三条的 cosine 只有 `0.08 ~ 0.12`

3. 左手 token 流出现了极端重合：
  - `你们好！ / 没关系！ / 谢谢！` 三条样本的 `lhand` 序列完全一致：
    - `aligned_match_lhand = 1.0000`
    - `prefix8 lhand = 1.0000`
  - 其左手 token 只在 4 个 id 上循环：
    - `[195, 38, 145, 238]`

4. 这与 `m2t overfit4 / overfit4_RE1k` 的错误模式高度一致：
  - `谢谢！ -> 你们好！`
  - `没关系！ -> 你们好！`（1k 版）
  - 即：模型会把 token 空间中高度相近的样本吸到同一个文本模式。

阶段性判断：
- 当前 `m2t overfit4` 的主问题已经不是 prompt/mask 实现 bug；
- 更像是：
  - `lhand` 流对这几条短句几乎不提供区分信息；
  - `body/rhand` 提供的信息不足以把 `你们好！ / 没关系！ / 谢谢！` 完全拉开；
  - 因此首个文本 token 的条件判别发生塌缩。

## 2. 诊断原则

后续所有实验都遵循以下原则：

1. 先验证“训练是否学会”，再验证“自由生成是否学会”。
2. 先从 `1` 条样本开始做最小 overfit，再扩到 `4` 条、`12` 条。
3. 先排实现错误，再考虑模型容量、tokenizer 上限和任务难度。
4. 每个实验都必须给出：
  - 输入数据
  - 观测指标
  - 通过标准
  - 失败后的下一步动作

## 3. 当前最优先的怀疑点

按优先级排序如下：

1. `t2m` 的训练 prompt 与推理 prompt 并不一致。
2. `m2t` 曾存在 generation mask 实现错误，已修复，但仍需继续验证修复后的自由生成行为。
3. `m2t` 在 `overfit4` 上的 teacher-forced token accuracy 本身也未满分，说明不是纯推理 bug。
4. `</sign>` / `</text>` 的停止规则、`max_new_tokens`、decode/parse 逻辑仍需继续做系统性检查。
5. sign token 对短句语义的可分性不足，但这属于第二层问题，不应先于训练-推理一致性排查。

## 3.1 已定位并修复的实现问题

### A. `m2t` generation mask bug
- 之前的 `AllowedTokensLogitsProcessor` 逻辑错误地把“允许 token 的 logits 全部重置为 0”，而不是保留它们的原始 logits。
- 旧逻辑等价于：

```python
masked_scores = full(-inf)
masked_scores[allowed] = 0
```

- 这会导致：
  - 所有允许 token 变成完全等价；
  - greedy decode 容易反复选择某个固定 token；
  - 在中文 `m2t` 中，实际观察到输出塌成 `！！！！！！！！...`。
- 修复后：
  - 只把不允许 token 设为 `-inf`
  - 对允许 token 保留原始 logits
- 修复后的直接结果：
  - `overfit1 m2t` 可以正常生成 `你们好！`
  - `overfit4 m2t` 不再输出纯标点，而是塌缩到真实中文句子 `你们好！`

### B. `t2m` prompt parity bug
- 该问题尚未修复完成，但已经有充分证据证明是当前 `t2m` 失败的主要来源。
- 证据：
  - `prompt_parity` 中 `t2m prefix_match = 0`
  - `m2t prefix_match = 1`
- 说明：
  - `m2t` 不能直接拿 `t2m` 的问题来解释；
  - `t2m` 当前需要优先修 prompt builder，而不是继续盲目调参。

## 4. 参考的官方经验

这里仅记录与当前问题直接相关的官方经验，避免把调参经验当作证据：

1. Qwen 官方 SFT 文档强调：
  - 数据模板必须稳定一致
  - `model_max_length` / `cutoff_len` 需要严格受控
  - LoRA、warmup、mixed precision 只是训练稳定性手段，不能替代模板正确性  
  参考：
  - https://qwen.readthedocs.io/en/v1.5/training/SFT/example.html
  - https://qwen.readthedocs.io/en/v1.5/training/SFT/

2. Hugging Face PEFT 文档强调：
  - 新 token 必须先 `add_tokens()`
  - 再 `resize_token_embeddings()`
  - 如果需要，必须显式让 embedding / `lm_head` 可训练，或通过 `modules_to_save` 保持它们可更新  
  参考：
  - https://huggingface.co/docs/peft/package_reference/trainable_tokens
  - https://huggingface.co/docs/peft/developer_guides/lora

这些经验支持当前路线合理，但不能单独证明实现已正确。

## 5. TODO 总表

下面的 TODO 不是泛泛建议，而是必须按顺序执行的定位实验。

### 5.0 执行约定

以下命令默认在项目根目录执行，并默认使用 `soke` 环境：

```bash
cd /home/SOKE
source /opt/conda/etc/profile.d/conda.sh
conda activate soke
```

为了避免 overfit 诊断被自动后处理干扰，下列训练命令默认都关闭：
- `AUTO_SHOW_M2T=0`
- `AUTO_VIS=0`
- `AUTO_VIS_MC=0`

迁移到 H20 后，优先保持：
- `BATCH_SIZE=8`
- `NUM_WORKERS=2`

如果只是 smoke，可先把：
- `END_EPOCH=50`
确认链路通了再升到 `200`。

### TODO 1：做 `overfit1` 最小验证
- 目标：
  - 确认 `t2m-only` 和 `m2t-only` 是否能记住单条样本。
- 数据：
  - 从 `CSL-Daily-overfit12` 中只保留 1 条样本。
- 实验内容：
  - 跑 `t2m-only overfit1`
  - 跑 `m2t-only overfit1`
- 观测：
  - train loss 曲线
  - 单条样本的自由生成结果
- 判定标准：
  - 如果 1 条样本都不能记住，优先判定为实现问题。
  - 如果 1 条能记住但 12 条不能，再考虑 token 可分性和任务难度。
- 输出：
  - `experiments/mgpt_overfit/<EXP_NAME>/`
  - 单条样本的生成结果对比

推荐命令：

`t2m overfit1`
```bash
cd /home/SOKE && source /opt/conda/etc/profile.d/conda.sh && conda activate soke && GPU_IDS=0 NUM_SAMPLES=1 SIGNER=P0000 EXP_NAME=SOKE_QWEN_CSL_OVERFIT1_T2M END_EPOCH=200 BATCH_SIZE=8 NUM_WORKERS=2 PRETRAINED_VAE=experiments/mgpt/VAE_SIGN_FINETUNE_LFQ4_ACC/checkpoints/last.ckpt AUTO_SHOW_M2T=0 AUTO_VIS=0 AUTO_VIS_MC=0 bash scripts/pipeline/train_qwen_csl_overfit_t2m.sh
```

`m2t overfit1`
```bash
cd /home/SOKE && source /opt/conda/etc/profile.d/conda.sh && conda activate soke && GPU_IDS=0 NUM_SAMPLES=1 SIGNER=P0000 EXP_NAME=SOKE_QWEN_CSL_OVERFIT1_M2T END_EPOCH=200 BATCH_SIZE=8 NUM_WORKERS=2 PRETRAINED_VAE=experiments/mgpt/VAE_SIGN_FINETUNE_LFQ4_ACC/checkpoints/last.ckpt AUTO_SHOW_M2T=0 AUTO_VIS=0 AUTO_VIS_MC=0 bash scripts/pipeline/train_qwen_csl_overfit_m2t.sh
```

扩展命令：

`t2m overfit4`
```bash
cd /home/SOKE && source /opt/conda/etc/profile.d/conda.sh && conda activate soke && GPU_IDS=0 NUM_SAMPLES=4 SIGNER=P0000 EXP_NAME=SOKE_QWEN_CSL_OVERFIT4_T2M END_EPOCH=200 BATCH_SIZE=8 NUM_WORKERS=2 PRETRAINED_VAE=experiments/mgpt/VAE_SIGN_FINETUNE_LFQ4_ACC/checkpoints/last.ckpt AUTO_SHOW_M2T=0 AUTO_VIS=0 AUTO_VIS_MC=0 bash scripts/pipeline/train_qwen_csl_overfit_t2m.sh
```

`m2t overfit4`
```bash
cd /home/SOKE && source /opt/conda/etc/profile.d/conda.sh && conda activate soke && GPU_IDS=0 NUM_SAMPLES=4 SIGNER=P0000 EXP_NAME=SOKE_QWEN_CSL_OVERFIT4_M2T END_EPOCH=200 BATCH_SIZE=8 NUM_WORKERS=2 PRETRAINED_VAE=experiments/mgpt/VAE_SIGN_FINETUNE_LFQ4_ACC/checkpoints/last.ckpt AUTO_SHOW_M2T=0 AUTO_VIS=0 AUTO_VIS_MC=0 bash scripts/pipeline/train_qwen_csl_overfit_m2t.sh
```

### TODO 2：补 teacher-forced token accuracy 检查
- 目标：
  - 把“训练目标是否学会”与“自由生成是否学会”分开。
- 数据：
  - 优先使用 `overfit1` 和 `overfit4`
- 实验内容：
  - 在训练样本上直接 forward
  - 仅统计 target 区域 token 的逐位准确率
  - 分任务记录：
    - `t2m` sign token accuracy
    - `m2t` text token accuracy
- 观测：
  - accuracy 是否接近 `100%`
- 判定标准：
  - 若 accuracy 很高但 free-run 错，问题在推理路径。
  - 若 accuracy 本身不高，问题在训练目标或数据表示。
- 输出：
  - 每个样本的 token accuracy
  - 按任务汇总的平均 accuracy

当前已经完成的 `overfit4 m2t` 结果：

```json
{"name": "S000000_P0000_T00", "gt": "你们好！", "pred": "你们好！", "teacher_forced_token_acc": 1.0, "teacher_forced_valid_tokens": 4, "first_divergence_index": -1, "free_run_exact_match": true}
{"name": "S000001_P0000_T00", "gt": "对不起！", "pred": "你们好！", "teacher_forced_token_acc": 0.6667, "teacher_forced_valid_tokens": 3, "first_divergence_index": 0, "free_run_exact_match": false}
{"name": "S000002_P0000_T00", "gt": "没关系！", "pred": "你们好！", "teacher_forced_token_acc": 0.6667, "teacher_forced_valid_tokens": 3, "first_divergence_index": 0, "free_run_exact_match": false}
{"name": "S000003_P0000_T00", "gt": "谢谢！", "pred": "你们好！", "teacher_forced_token_acc": 0.6667, "teacher_forced_valid_tokens": 3, "first_divergence_index": 0, "free_run_exact_match": false}
```

这组结果的意义：
- `teacher_forced_token_acc` 平均只有 `0.75`；
- 说明 `overfit4 m2t` 当前并非“训练目标完全学会，只是推理有 bug”；
- 至少后三条样本在 teacher forcing 条件下，也已经在首 token 上发生混淆。

推荐命令：

`t2m teacher-forced / free-run 诊断`
```bash
cd /home/SOKE && source /opt/conda/etc/profile.d/conda.sh && conda activate soke && python scripts/analysis/diagnose_qwen_overfit.py --cfg configs/soke_csl_overfit_t2m.yaml --ckpt experiments/mgpt/SOKE_QWEN_CSL_OVERFIT1_T2M/checkpoints/last.ckpt --split train --task t2m --num_examples 1 --batch_size 1 --use_gpus 0 --device 0
```

`m2t teacher-forced / free-run 诊断`
```bash
cd /home/SOKE && source /opt/conda/etc/profile.d/conda.sh && conda activate soke && python scripts/analysis/diagnose_qwen_overfit.py --cfg configs/soke_csl_overfit_m2t.yaml --ckpt experiments/mgpt/SOKE_QWEN_CSL_OVERFIT1_M2T/checkpoints/last.ckpt --split train --task m2t --num_examples 1 --batch_size 1 --use_gpus 0 --device 0
```

### TODO 3：补 free-run exact match 与 divergence point 检查
- 目标：
  - 找出生成序列从哪个 token 开始偏离 GT。
- 数据：
  - 优先使用 `overfit1`
- 实验内容：
  - 用 greedy generation 生成完整序列
  - 与 GT token 序列逐位比较
  - 记录：
    - exact match rate
    - first divergence index
- 观测：
  - 偏差是否一开始就出现
  - 还是前缀较长后才漂移
- 判定标准：
  - 一开始就偏离：优先怀疑 prompt / special token / start token
  - 前缀正确后漂移：优先怀疑 stop rule / output space / exposure bias
- 输出：
  - 每个样本的 divergence report

当前已经完成的 `overfit4 m2t` divergence 结果：
- `divergences = [-1, 0, 0, 0]`

解释：
- 第一条样本完全对齐；
- 后三条样本都在 **第 0 个 token** 就偏离；
- 因而当前失败不是“后面滚歪了”，而是 **首 token 级别的模式坍缩**。

说明：
- 当前由 `scripts/analysis/diagnose_qwen_overfit.py` 一并导出：
  - `teacher_forced_token_acc`
  - `free_run_exact_match`
  - `first_divergence_index`
- 推荐先对 `overfit1` 跑，再对 `overfit4` 跑。

### TODO 4：严格核对训练 prompt 与推理 prompt
- 目标：
  - 排除训练-推理模板不一致。
- 数据：
  - `overfit1`
- 实验内容：
  - 导出训练时的完整：
    - `input_ids`
    - `labels`
  - 导出推理时的完整：
    - prompt `input_ids`
  - 对照检查：
    - `<t2m> / <m2t>`
    - `<text> / </text>`
    - `<sign> / </sign>`
    - BOS/EOS
    - tokenizer 是否应用了额外模板
- 观测：
  - 训练 prompt 和推理 prompt 是否严格一致
- 判定标准：
  - 一旦出现任何一个 special token 不一致，就必须先修正再做后续实验。
- 输出：
  - `prompt_parity_report.md`

推荐命令：

```bash
cd /home/SOKE && source /opt/conda/etc/profile.d/conda.sh && conda activate soke && python scripts/analysis/export_prompt_parity_report.py --cfg configs/soke_csl_overfit_t2m.yaml --split train --tasks t2m,m2t --num_examples 20 --output_dir /tmp/qwen_prompt_parity
```

正式训练配置也可以直接检查：

```bash
cd /home/SOKE && source /opt/conda/etc/profile.d/conda.sh && conda activate soke && python scripts/analysis/export_prompt_parity_report.py --cfg configs/soke.yaml --split train --tasks t2m,m2t --num_examples 20 --output_dir /tmp/qwen_prompt_parity_full
```

### TODO 5：为 `t2m / m2t / mc` 添加 generation-time vocab mask
- 目标：
  - 避免 unified vocab 下的错误子空间污染。
- 数据：
  - `overfit1`，随后扩到 `overfit4`
- 实验内容：
  - `t2m` / `mc` 生成时，仅允许：
    - `<motion_id_*>`
    - `<hand_id_*>`
    - `<rhand_id_*>`
    - `</sign>`
  - `m2t` 生成时，仅允许：
    - 普通文本 token
    - `</text>`
  - 禁止 sign/text 交叉输出
- 观测：
  - `m2t` 是否不再泄漏 sign token
  - overfit 生成是否显著改善
- 判定标准：
  - 如果加 mask 后 `m2t` 立即改善，说明主要问题在生成约束而非训练目标。
- 输出：
  - mask 前后对比结果

当前状态：
- 已实现到 `mGPT/archs/mgpt_qwen.py`
- 可通过诊断脚本比较 masked vs unmasked

推荐命令：

`t2m mask 诊断`
```bash
cd /home/SOKE && source /opt/conda/etc/profile.d/conda.sh && conda activate soke && python scripts/analysis/inspect_qwen_generation_diagnostics.py --cfg experiments/mgpt/SOKE_QWEN_CSL_OVERFIT1_T2M/config_*.yaml --ckpt experiments/mgpt/SOKE_QWEN_CSL_OVERFIT1_T2M/checkpoints/last.ckpt --task t2m --split test --num_examples 1 --compare_unmasked --output_jsonl /tmp/qwen_t2m_diag.jsonl
```

`m2t mask 诊断`
```bash
cd /home/SOKE && source /opt/conda/etc/profile.d/conda.sh && conda activate soke && python scripts/analysis/inspect_qwen_generation_diagnostics.py --cfg experiments/mgpt/SOKE_QWEN_CSL_OVERFIT1_M2T/config_*.yaml --ckpt experiments/mgpt/SOKE_QWEN_CSL_OVERFIT1_M2T/checkpoints/last.ckpt --task m2t --split test --num_examples 1 --compare_unmasked --output_jsonl /tmp/qwen_m2t_diag.jsonl
```

### TODO 6：核对 stop token、`max_new_tokens` 与 parse 逻辑
- 目标：
  - 排除“模型其实生成对了，但被截断/裁坏”的情况。
- 数据：
  - `overfit1`
- 实验内容：
  - 检查：
    - `t2m` 是否以 `</sign>` 停止
    - `m2t` 是否以 `</text>` 停止
    - 若没遇到 stop token，是否被 `max_new_tokens` 截断
    - decode 后 parse 是否错误剔除了合法 token
- 观测：
  - 生成序列原始文本
  - parse 后序列
- 判定标准：
  - 若原始输出正确、parse 后错误，则问题完全在后处理。
- 输出：
  - raw generation vs parsed generation 对照

说明：
- 当前由 `scripts/analysis/inspect_qwen_generation_diagnostics.py` 一并完成。
- 重点看输出字段：
  - `raw_tail_text`
  - `parsed_output`
  - `stop_token_pos`
  - `disallowed_tail_ids`

### TODO 7：做 sign token 可分性分析的强化版
- 目标：
  - 判断 sign token 本身是否足以支撑 `m2t` 反推文本。
- 数据：
  - `overfit12`
- 实验内容：
  - 已有：
    - token-hist cosine
    - aligned exact match
  - 追加：
    - 1-NN retrieval
    - token edit distance
    - 简单 DTW on token ids
    - 错误样本的最近邻文本分析
- 观测：
  - 不同文本之间是否有稳定的 token 分离
- 判定标准：
  - 若语义不同样本长期互为最近邻，说明 tokenizer 对短句区分不足。
- 输出：
  - 热力图
  - retrieval summary

当前已有基础结果位置：
- `experiments/mgpt/SOKE_QWEN_CSL_OVERFIT12/analysis_sign_tokens/`

建议执行顺序：
1. 先完成 TODO 1-6
2. 再回到 `overfit12` 做 token 可分性强化分析

备注：
- 当前这一项还没有专门的一键脚本，需要在现有基础统计上继续扩展 `1-NN retrieval / token edit distance / DTW`。

### TODO 8：测 VAE ceiling，分离“LM 问题”与“tokenizer 上限”
- 目标：
  - 为 `t2m` 建立理论上限。
- 数据：
  - `overfit12`
- 实验内容：
  - 对 GT pose 做：
    - `pose -> tokenize -> reconstruct`
  - 记录纯 VAE 重建误差
  - 再与 `t2m` overfit 误差比较
- 观测：
  - `t2m` 误差距离 VAE floor 还有多远
- 判定标准：
  - 如果 `t2m` 已接近 VAE floor，说明 LM 已基本学会，剩余误差来自 tokenizer。
  - 如果距离很大，说明 LM 还没学到。
- 输出：
  - `vae_ceiling_vs_t2m.md`

建议实验内容：
1. 对 `overfit12` 的 GT pose 运行：
   - `pose -> tokenize -> reconstruct`
2. 记录对应的重建误差和可视化
3. 再与 `t2m-only overfit12` 的结果对比

备注：
- 当前没有单独的一键命令，建议在 TODO 1-6 完成后实现专门脚本，避免现在同时改动太多变量。

### TODO 9：检查新增 token 的梯度与更新幅度
- 目标：
  - 排除“训练开了，但关键 token 实际没更新”的情况。
- 数据：
  - `overfit1`
- 实验内容：
  - 在若干 step 上记录：
    - sign token embedding 的梯度范数
    - text special token embedding 的梯度范数
    - `lm_head` 对应行的梯度范数
  - 对比：
    - 高频训练 token
    - 很少出现 token
- 观测：
  - 梯度是否为 0 或极小
- 判定标准：
  - 若关键 token 行几乎无梯度，说明当前训练信号并未有效传到这些 token。
- 输出：
  - 梯度统计表

备注：
- 这是第二阶段定位项。
- 只有在 TODO 1-6 完成后仍然异常时，才值得继续做。
- 当前尚未实现独立脚本。

### TODO 10：从 `overfit1 -> overfit4 -> overfit12` 做阶梯实验
- 目标：
  - 判断失败是“完全不会”还是“规模一上来就退化”。
- 数据：
  - 1 条、4 条、12 条
- 实验内容：
  - 固定任务，逐步扩样本数
  - 对每一级都记录：
    - train loss
    - teacher-forced accuracy
    - free-run exact match
- 观测：
  - 性能在哪个规模开始崩塌
- 判定标准：
  - `1` 条成功、`4` 条失败：优先怀疑输入可分性或生成约束
  - `1` 条就失败：优先怀疑实现错误
- 输出：
  - `overfit_scaling_report.md`

推荐执行顺序：

1. `t2m overfit1`
2. `m2t overfit1`
3. `t2m overfit4`
4. `m2t overfit4`
5. 如前四步通过，再回到 `overfit12`

## 6. 推荐执行顺序

为了避免继续在错误层面上浪费时间，执行顺序必须固定：

1. `TODO 1`：先做 `overfit1`
2. `TODO 2`：teacher-forced accuracy
3. `TODO 3`：free-run exact match
4. `TODO 4`：训练/推理 prompt parity
5. `TODO 5`：generation-time vocab mask
6. `TODO 6`：stop / parse 规则核对
7. `TODO 10`：`1 -> 4 -> 12` 规模扩展
8. `TODO 7`：token 可分性分析
9. `TODO 8`：VAE ceiling 对比
10. `TODO 9`：梯度检查

## 7. 暂不建议做的事

在上述定位完成前，不建议先做以下动作：

- 继续增加 epoch
- 反复调整 task ratio
- 盲目提高 LoRA rank
- 在 `overfit12` 上继续直接看可视化猜问题
- 先把问题归咎于 tokenizer 或数据集

这些都可能掩盖实现层面的错误。

## 8. 预期产出

完成上述 TODO 后，应该能得到非常清楚的结论：

1. 是训练目标没学会，还是自由生成没对齐；
2. 是 prompt/stop/mask 的实现问题，还是 unified vocab 的任务约束问题；
3. 是 sign token 可分性不足，还是 Qwen 对这套 token 语言建模不充分；
4. `t2m` 距离 VAE ceiling 还有多远，`m2t` 是否需要更强的输入建模或独立架构。

在此之前，不应继续把问题简化为“再训久一点试试”。
