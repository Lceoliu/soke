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

### 1.3 当前 overfit 结果

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

### 1.4 当前最重要的判断
仅根据 loss 和生成结果，最合理的判断不是“模型完全没学会”，而是：

1. `m2t overfit1` 已经证明 teacher forcing 与自由生成在单样本上可以对齐；
2. `m2t overfit4` 表明从 1 条扩到 4 条后，模型在首 token 上即发生模式坍缩；
3. `t2m` 当前存在明确实现 bug，即训练-推理 prompt mismatch；
4. 问题更像是“训练-推理不一致 / 生成约束实现错误 / 少样本下输入可分性不足”，而不是简单的“轮数不够”。

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
