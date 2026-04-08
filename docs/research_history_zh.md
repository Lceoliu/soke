# 研究路径与实验历史（Qwen 分支）

更新日期：2026-04-08  
适用范围：当前 `Qwen` 分支代码与实验结论  
目的：把本仓库从原始 SOKE 基底演化到当前 `LFQ tokenizer + Qwen + mT5 + CSL classifier` 的研究路径、关键实现和阶段性结论统一记录下来，作为后续清理旧文档与继续实验的主参考。

---

## 1. 总体科研目标

当前的核心科研目标是：

> 构建一个 **Sign Language Auto-regressive Model**，即一个统一的、基于自回归思想的手语生成 / 翻译 / 理解模型与训练方法。

这里的“统一”主要指三层含义：

1. **统一表示**  
   把连续手语 pose 序列转换为可重建、可建模的离散 token 或连续 latent embedding。

2. **统一建模**  
   尽量用一套大语言模型式的 token continuation / seq2seq 建模框架，同时覆盖：
   - `t2m`：text -> motion / sign
   - `m2t`：motion / sign -> text
   - `mc`：motion continuation

3. **统一训练动机**  
   希望通过语言模型 loss 把 sign token / sign embedding 逐步拉开、分散，并和文本语言空间形成对齐。

---

## 2. 主要 Related Works 与仓库基底

当前代码仓库的研究脉络主要基于以下三条工作：

1. **Signs as Tokens: A Retrieval-Enhanced Multilingual Sign Language Generator (SOKE)**  
   这是当前仓库的直接基底，也是最初的代码起点。

2. **MotionGPT**  
   是 SOKE 的进一步代码基底。当前仓库中的很多训练框架、数据模块、Lightning 结构都继承自 MotionGPT 风格。

3. **Uni-Sign: Toward Unified Sign Language Understanding at Scale**  
   是当前阶段的重要参考对象，尤其影响了后续从 decoder-only 路线转向 `mT5` 路线的判断。

代码上可以看到：

- 主训练 / 测试入口仍是 MotionGPT 风格：
  - `train.py`
  - `test.py`
- 主模型统一挂在：
  - `mGPT/models/mgpt.py`
- 数据模块统一挂在：
  - `mGPT/data/H2S.py`

---

## 3. 阶段一：改造 SOKE 的 tokenizer / VAE（LFQ + 更细的重建损失）

### 3.1 动机

最初并不满足于原始 SOKE 中 tokenizer 的重建效果，因此把 **reconstruction 质量** 作为 tokenizer 好坏的首要评价标准。

这里的判断标准不是“离散 token 是否更适合下游 LM”，而是更直接的：

> 经过 VAE / tokenizer 编码再解码后，重建出来的手语动作是否仍然是 **人类可辨识** 的。

### 3.2 主要改动

相对原始 SOKE / 旧 VQ 路线，做了两类核心改动：

1. **量化器切到多层 LFQ**
   - 主实现：
     - `mGPT/archs/mgpt_vq.py`
     - `mGPT/archs/tools/quantize_lfq.py`
   - 当前主路径不再维护旧 VQ / RVQ 兼容训练语义，仓库主配置默认走 `quantizer='lfq'`。

2. **增强 VAE 重建损失**
   - loss 主实现：
     - `mGPT/losses/mgpt.py`
   - 当前实际在用的重建项包括：
     - `recons_feature`
     - `recons_velocity`
     - `recons_fk_hand`
     - `recons_accel_hand`
     - `recons_accel_wrist_rel`
     - `recons_contact`
     - `vq_commit`

### 3.3 关键结构证据

- `VQVae.encode_continuous()` 已支持直接输出 quantization 之前的连续 latent：
  - `mGPT/archs/mgpt_vq.py`
- Decoder 已改成 `linear` 上采样，并支持 `return_hidden=True` 给 contact head：
  - `mGPT/archs/mgpt_vq.py`
- 当前主 VAE 配置常见为：
  - `configs/vq/re128_lfq4.yaml`
  - `configs/vq/hand256_lfq4.yaml`
  - `configs/vq/hand256_lfq5.yaml`

### 3.4 阶段性结论

实验上，LFQ 与更细的重建 loss **确实提高了 reconstruction 质量**，生成的手语重建更自然、更接近人类可辨识动作。

但这里也得到一个重要的负结论：

> **重建更好，不代表更适合后续 LLM。**

也就是说，tokenizer 在 reconstruction 指标上的收益，并没有自动转化为下游统一建模的收益。

---

## 4. 阶段二：从 SOKE/mBART 转向统一的 Qwen decoder-only 路线

### 4.1 动机

原始 SOKE 中的 `mBART + 多头 lm head` 方案，并不符合这里对“大一统翻译 / 生成”的目标设定。因此新开了当前 `Qwen` 分支，核心思路是：

> 把 sign token 视作一门新语言，把新的 sign token 直接加到 Qwen 词表中，训练统一的 decoder-only continuation 能力。

### 4.2 当前 Qwen 路线的实现

主实现：

- Qwen 模型适配器：
  - `mGPT/archs/mgpt_qwen.py`
- 主模型调度：
  - `mGPT/models/mgpt.py`
- task formatting：
  - `mGPT/archs/task_formatting.py`

常用配置：

- `configs/lm/qwen2_5_0_5b.yaml`
- `configs/lm/qwen2_5_3b.yaml`
- `configs/lm/qwen2_5_8b.yaml`
- `configs/soke.yaml`
- `configs/soke_full_m2t_0_5b_pw00.yaml`
- `configs/soke_full_m2t_0_5b_pw05.yaml`
- `configs/soke_full_qwen2_5_8b_pw05.yaml`

### 4.3 Qwen 任务定义

当前 Qwen 路线主要统一了三类任务：

1. `t2m`
2. `m2t`
3. `mc`

对应代码入口在：

- `mGPT/models/mgpt.py`
  - `train_lm_forward()`
  - `val_t2m_forward()`
  - `val_m2t_forward()`
  - `val_mc_forward()`

### 4.4 Sign token 组织方式

当前 sign token 默认来自单独训练好的 frozen tokenizer / VAE，然后直接缓存到 `.npy` 文件中供 LM 使用。

代码证据：

- token cache 数据集：
  - `mGPT/data/humanml/dataset_t2m_cb.py`
- token 预生成：
  - `scripts/get_motion_code.py`

对于多部位 + 多层 quantizer，当前实际协议是：

- 3 个部位：
  - `body`
  - `lhand`
  - `rhand`
- 多层 quantizer：
  - 例如 body `Q=4`
- LM 看到的是 interleaved / flattened 后的 sign token 序列

例如：

- `S000000_P0000_T00` 的缓存 token 形状可能是 `(1, 52, 3)`
- 若 `num_quantizers = 4`
- 原视频每 `4` 帧一个 chunk，则 `52 / 4 = 13` 个 chunk
- 每层 3 个 token，共 `13 * 4 * 3 = 156` 个 sign token

更具体地说，缓存 token 在 LM 进入格式化之前，逻辑上可理解为：

```text
[
  [107, 195,  62],
  [177, 294, 295],
  [380, 657, 529],
  [485,1006, 873],
  [ 26, 195, 134],
  [185, 294, 503],
  [348, 657, 630],
  [509,1006, 951],
  ...
]
```

这里每一行表示“同一 quantizer 层上的 3 个部位 token”：

- 第 1 列：`body`
- 第 2 列：`lhand`
- 第 3 列：`rhand`

以 `S000000_P0000_T00` 为例：

- `t1` 时刻（第一个 chunk）的 4 层 token 依次为：

```text
<motion_id_107> <hand_id_195>  <rhand_id_62>
<motion_id_177> <hand_id_294>  <rhand_id_295>
<motion_id_380> <hand_id_657>  <rhand_id_529>
<motion_id_485> <hand_id_1006> <rhand_id_873>
```

- `t2` 时刻（第二个 chunk）的 4 层 token 依次为：

```text
<motion_id_26>  <hand_id_195>  <rhand_id_134>
<motion_id_185> <hand_id_294>  <rhand_id_503>
<motion_id_348> <hand_id_657>  <rhand_id_630>
<motion_id_509> <hand_id_1006> <rhand_id_951>
```

也就是说，LM 最终看到的顺序不是“先 body 全部，再 hand 全部”，而是：

> 先按时间 chunk 展开，再在同一个 chunk 内优先把 `q1 -> q4` 的三个部位 token 放在邻近位置。

当前实现里，这种“时间优先 + quantizer 层展开 + 部位并排”的协议，是通过：

- `mGPT/models/mgpt.py`
  - `_flatten_batch_tokens_for_lm()`
  - `_flatten_single_tokens_for_lm()`
- `mGPT/archs/task_formatting.py`
  - `serialize_sign_token_strings()`
  - `build_batch()`

共同完成的。

代码里，flatten / unflatten 与 per-q offset 的处理集中在：

- `mGPT/models/mgpt.py`
  - `_flatten_batch_tokens_for_lm()`
  - `_flatten_single_tokens_for_lm()`
  - `_unflatten_single_tokens_from_lm()`

### 4.5 Qwen 中 sign token 的接入方式

Qwen 路线并不是把 motion 输入当作现有词表里已有语言，而是：

1. 根据 codebook size 动态构造：
   - `<motion_id_*>`
   - `<hand_id_*>`
   - `<rhand_id_*>`
2. 调用 tokenizer 的 `add_tokens`
3. 扩容 embedding / lm_head

直接代码证据：

- `mGPT/archs/mgpt_qwen.py`
  - `self.tokenizer.add_tokens(...)`
  - `self.language_model.resize_token_embeddings(...)`

### 4.6 Qwen 训练参数与可训练部分

当前默认 LoRA 配置：

- `use_lora: true`
- `lora_rank: 64`
- `lora_alpha: 128`
- `lora_dropout: 0.05`

对应配置：

- `configs/lm/qwen2_5_0_5b.yaml`
- `configs/lm/qwen2_5_3b.yaml`
- `configs/lm/qwen2_5_8b.yaml`

同时，代码会额外确保以下部分可训练：

- `embed_tokens`
- `lm_head`

相关证据：

- `mGPT/archs/mgpt_qwen.py`
  - `_enable_embedding_and_lm_head_training()`
- `mGPT/models/base.py`
  - differential LR param groups
  - `embed_tokens` / `lm_head` 的 LR multiplier

---

## 5. Qwen 路线的 overfit 修复与当前结论

### 5.1 先前问题

Qwen 路线早期曾出现 “少量数据都 overfit 不起来” 的问题。后续诊断表明，这并不是模型本身不会学，而主要是工程链路错误：

1. 训练和验证使用了不同 token 来源
2. pad / `lengths` 处理错误
3. `t2m` 训练 prompt 与推理 prompt 不一致
4. overfit 实验默认还带随机 token-drop 增强

### 5.2 修复后的结论

修复上述问题后，当前已经确认：

- `overfit1 / overfit4 / overfit12` 可以成功
- 三个任务 `m2t / t2m / mc` 在少量数据上都可以完全拟合

因此，**Qwen pipeline 的训练 / 推理一致性问题本身已经基本打通**。

### 5.3 但 full train 依然失败

尽管 overfit 成功，full train 依然非常失败：

- 无论是 `Qwen2.5-0.5B` 还是 `Qwen2.5-3B`
- 在 full train 上都无法真正学会 `m2t / t2m / mc`
- 表现为：
  - train loss 正常下降
  - val loss / ppl 上升
  - test 上生成文本与 GT 基本无关
  - 但又没有灾难性遗忘，仍保留文本 continuation 能力

这说明：

> 当前 decoder-only 路线在现有数据规模与 tokenizer 条件下，虽然可以 overfit，但没有建立出可泛化的统一建模能力。

### 5.4 prefix-weight loss 的尝试

为缓解 `m2t` 中“只有 text token 计算 loss，其余 prefix 被 mask”的监督不足问题，代码中加入了：

- `m2t_prefix_loss_weight`

代码证据：

- `mGPT/archs/mgpt_qwen.py`
- `mGPT/archs/task_formatting.py`

对应配置：

- `configs/soke_full_m2t_0_5b_pw00.yaml`
- `configs/soke_full_m2t_0_5b_pw05.yaml`
- `configs/soke_full_qwen2_5_8b_pw05.yaml`

阶段性实验结论：

- 在 `Qwen2.5-0.5B`
- `m2t-only`
- `44 epoch`

的对比里，`pw=0.5` 的 val ppl 显著优于 `pw=0.0`，同时 train loss 明显更高。当前判断是：

> 这个改动是必要的，它提高了训练目标对 prefix 部分的约束强度。

但它**没有根本解决** full train 泛化失败的问题。

### 5.5 当前对 Qwen full train 的阶段性判断

截至目前，Qwen 路线的总体判断可以概括为：

1. **训练 pipeline 本身是通的。**
   - 少量数据 overfit 已经证明这一点。

2. **模型对 sign token 并非完全无感。**
   - 新增 sign token 的 embedding 与 `lm_head` 输出行会收到梯度并发生更新。

3. **full train 失败不是单一 bug，而更像是系统性泛化失败。**
   - `train loss` 正常下降
   - `val loss / ppl` 持续变差
   - `test m2t` 保留文本语言 continuation 能力，但生成内容与 GT 基本无关

4. **`m2t_prefix_loss_weight` 是必要但不充分的修补。**
   - 它缓解了 prefix 监督不足问题，但并未从根本上解决 decoder-only 统一建模失败。

---

## 6. 阶段三：转向 mT5，直接读取 frozen VAE embeddings

### 6.1 动机

在 Qwen 路线长期无法突破后，参考 Uni-Sign 的思路，开始怀疑：

> 在当前数据规模和 tokenizer 质量下，decoder-only 模型很难获得足够强的“涌现式统一建模能力”。

于是转向 `mT5`，尝试先做更稳的 `m2t` 路线。

### 6.2 当前 mT5 路线的实现

主实现：

- `mGPT/archs/mgpt_mt5.py`

主配置：

- `configs/lm/mt5_base.yaml`
- `configs/soke_mt5_m2t.yaml`
- `configs/soke_mt5_csl_m2t.yaml`

### 6.3 输入形式

mT5 当前不是读取 sign token，而是直接读取 **frozen VAE continuous embeddings**。

代码证据：

- `mGPT/archs/mgpt_mt5.py`
  - `needs_raw_features = True`
  - `_encode_with_vaes()`
  - `_encode_and_project()`

当前具体做法是：

1. 输入原始 motion feature（133 维）
2. 用 frozen body / hand / rhand VAE 分别做 `encode_continuous()`
3. 拼接成连续 latent
4. 再经过一个 MLP / Linear projection 投到 mT5 的 `d_model`
5. 通过 `inputs_embeds` 喂给 mT5 encoder

其中 latent 维度约为：

- `body 512 + lhand 512 + rhand 512 = 1536`

### 6.4 当前 prompt 形式

当前代码里的 prompt 并不是自由文本模板，而是固定中文 prompt：

- `把下面这句{lang}手语翻译为{lang}文本:`

语言映射写死在：

- `mGPT/archs/mgpt_mt5.py`
  - `csl -> 中文`
  - `phoenix -> 德语`
  - `how2sign / h2s -> 英语`

### 6.5 LoRA 设定

当前 mT5 默认也使用和 Qwen 同等级别的 LoRA：

- `lora_rank: 64`
- `lora_alpha: 128`
- `lora_dropout: 0.05`

代码 / 配置证据：

- `mGPT/archs/mgpt_mt5.py`
- `configs/lm/mt5_base.yaml`

### 6.6 当前 mT5 路线的边界

需要特别注意：

> 当前 mT5 代码只实现了 `m2t`，并没有实现统一的 `t2m / mc`。

直接证据：

- `mGPT/archs/mgpt_mt5.py`
  - `generate_conditional()` 里只支持 `task="m2t"`
  - `generate_direct()` 明确 `NotImplemented`

因此，mT5 路线当前更像是：

> 用 frozen VAE embeddings 做手语到文本翻译的单任务替代路线

而不是最终统一模型的完整实现。

### 6.7 mT5 的实验结论

实验上：

- overfit 仍然成功
- full train 与 Qwen 类似，test / val 上无法真正学会

所以阶段性结论是：

> 从 Qwen 切到 mT5，并没有自动解决泛化问题。

### 6.8 mT5 full train 的实际生成问题

当前 `mT5` 的典型失败模式不是输出乱码，而是：

> 生成语法基本通顺、看起来像正常英文句子，但与 GT 语义明显无关，并且容易塌缩到模板化叙述。

直接样例可见：

- 文件：
  - `experiments/mgpt/SOKE_MT5_M2T_FULL/auto_reports/downstream/m2t_examples.jsonl`

样例 1：

```text
name = -fZc293MpJk_2-1-rgb_front
ref  = The aileron is the control surface in the wing that is controlled by lateral movement right and left of the stick.
pred = So, when you're going to throw a blanket, you're going to want to throw a blanket on the floor, and you're going to want to throw it in the corner.
```

这个样例说明：

- 生成句子形式完全像自然语言
- 但内容与航空控制面的 GT 描述毫无关系
- 模型更像是落入某种高频模板化叙述，而不是从 sign embedding 中恢复语义

样例 2：

```text
name = -g0iPSnQt6w_0-1-rgb_front
ref  = Buenos Dias, I'm Bobby Larew, you didn't know I spoke Spanish, did you?
pred = Hi, my name is Jennifer Bailey, and I'm going to show you how to make a beard.
```

这个样例说明：

- 模型并没有完全失去英文生成能力
- 但生成的人名、主题和叙述内容都偏向训练集中其他高频模板
- 这和 Qwen 路线中“保持文本语言能力，但与真实 sign 语义脱钩”的现象是相似的

---

## 7. mT5 是否参考 motion：已做过的反证

为了排除“mT5 根本没看 motion embeddings”的可能，做过 motion shuffle / 替换实验。

结论是：

- 当把 motion 部分随机替换或打乱时
- `BLEU4` 会从约 `2` 掉到约 `0.2`

因此可以确认：

> mT5 并不是完全忽略 motion embedding。

也就是说，问题不是“motion branch 完全没被使用”，而是“虽然参考了 motion，但还不足以形成稳定泛化”。

---

## 8. 阶段四：利用 CSL-Daily 的多 signer 特性验证 VAE embeddings

### 8.1 动机

在 Qwen 与 mT5 full train 都难以突破后，开始转而验证更底层的问题：

> frozen VAE 生成出来的 embedding，是否本身就带有足够的动作判别信息？

CSL-Daily 很适合做这个验证，因为：

- 样本名形如 `Sxxxxxx_Pxxxx_Txx`
- 同一个 `Sxxxxxx` 表示同一句意 / 同一个动作类别
- `Pxxxx` 表示不同 signer

这就天然支持：

> 同一句意、不同 signer 的动作，是否能在 VAE embedding 空间里被判成同一类？

### 8.2 当前验证脚本

相关实现：

- `scripts/analysis/analyze_csl_vae_encoder_similarity.py`
- `scripts/analysis/train_csl_vae_action_classifier.py`

相关配置：

- `configs/soke_mt5_csl_m2t.yaml`

### 8.3 第一版验证：mean pooling + linear / MLP

最初使用了：

- frozen VAE continuous sequence embeddings
- 对时序做 `mean pooling`
- 接一个简单 classifier

初始全量 `5697+` 类实验失败，但后续分析表明主要问题是：

- 类别数过大
- 每类训练样本几乎只有 1 个
- 任务本质上退化成极端 one-shot 分类

这个结果不能证明 embedding 无效，只能说明设定太苛刻。

### 8.4 第二版验证：严格 signer 切分 + temporal conv classifier

后续脚本已升级为：

1. 只保留满足 `>= 3 signer` 的动作类
2. 保证每类：
   - 至少 `2 signer` 进 train
   - `1 signer` 进 test
3. 对 sequence embedding 使用 `temporal conv classifier`

代码证据：

- `scripts/analysis/train_csl_vae_action_classifier.py`

### 8.5 当前最重要的结果

在 `max_classes = 1024` 的实验里，得到：

- `num_classes = 1024`
- `train_samples = 2048`
- `val_samples = 209`
- `test_samples = 1024`
- `train_acc = 1.0000`
- `val_acc = 0.5789`
- `test_acc = 0.6221`
- `test_top5_acc = 0.8174`

这个结果的意义非常大：

1. **VAE embedding 不是坏的，也不是塌的。**
2. **在 cross-signer 条件下，同一句意的动作在 embedding 空间中有很强的可分性。**
3. **之前 Qwen / mT5 full train 的失败，不能简单归因于“VAE embedding 完全无效”。**

更具体地说：

> 至少在 `1024-way`、`2 signer train + 1 signer test`、`temporal conv` 设定下，VAE embedding 对动作类别有显著且强的跨 signer 判别能力。

### 8.6 但这不代表问题已经解决

这个 classifier 实验只能说明：

- embedding **有信息**
- embedding **能跨 signer 区分类别**

但它不能直接推出：

- mT5 一定能学好
- Qwen 一定能学好
- 统一 AR 模型的训练目标已经正确

当前更合理的解释是：

> VAE embedding 本身有足够信号，但 Qwen / mT5 的下游训练方式、任务设计、监督分配或模型归纳偏置，仍然不足以把这种信号转化为可泛化的生成 / 翻译能力。

---

## 9. 到目前为止的总体判断

### 9.1 已经基本确认的事情

1. **LFQ + 更细的 VAE loss 确实提升了重建质量。**
2. **Qwen pipeline 的训练 / 推理一致性问题已经修好。**
3. **Qwen 三任务在少量数据上都能 overfit。**
4. **mT5 不是完全不看 motion embeddings。**
5. **VAE continuous embeddings 在 CSL 上具有显著的 cross-signer 可分性。**

### 9.2 仍未解决的核心问题

1. **为什么 Qwen full train 无法泛化？**
2. **为什么 mT5 full train 也无法泛化？**
3. **token / embedding 的判别信息，为什么没有顺利转化为 LM 的稳定生成 / 翻译能力？**

### 9.3 当前更可信的研究判断

当前最可信的中间判断是：

> 问题不在于 “VAE tokenizer / embeddings 完全无效”，而更可能在于：

- 下游 LM 的训练目标设计
- decoder-only / seq2seq 模型对当前输入形式的归纳偏置
- 数据规模与 supervision 信号强度不足
- 统一多任务训练本身的困难

---

## 10. 当前仓库最值得优先阅读的文件

如果要从代码侧理解当前研究现状，建议优先看：

1. `mGPT/archs/mgpt_vq.py`
2. `mGPT/archs/mgpt_qwen.py`
3. `mGPT/archs/mgpt_mt5.py`
4. `mGPT/models/mgpt.py`
5. `mGPT/data/H2S.py`
6. `mGPT/data/humanml/dataset_t2m_cb.py`
7. `scripts/analysis/train_csl_vae_action_classifier.py`
8. `configs/soke.yaml`
9. `configs/soke_mt5_m2t.yaml`
10. `configs/soke_mt5_csl_m2t.yaml`

---

## 11. 后续清理文档时的原则

后续清理 `docs/` 时，建议统一以本文为准：

1. 删除仍把当前主线写成 `LFQ 分支 + mBART` 的描述
2. 删除已经不再推进的 quick ablation / TODO 草稿
3. 保留真正仍然有效的：
   - VAE 结构文档
   - loss 文档
   - 数据接入文档
   - classifier 验证结论
4. 所有“当前状态”类文档，优先反映：
   - `Qwen` 主分支
   - `Qwen` 三任务统一建模尝试
   - `mT5` 作为 `m2t` 替代路线
   - `CSL classifier` 对 embedding 有效性的支撑证据
