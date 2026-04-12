# Gap Map

> Known field gaps with stable IDs. Updated as papers are ingested.

| ID | Gap | Status | Linked Papers | Linked Ideas |
|----|-----|--------|---------------|--------------|
| G1 | What makes a tokenizer/embedding suitable for downstream LM (not just reconstruction)? | unresolved | paper:zuo2024_soke, paper:jiang2023_motiongpt | idea:001, idea:004 |
| G2 | Why does full-train generalization fail for both decoder-only and seq2seq sign-LM despite functional pipeline and discriminative embeddings? | unresolved | paper:li2025_unisign | idea:002, idea:003 |
| G3 | How can unified multi-task sign language AR training (t2m + m2t + mc) be made to generalize at current data scales? | unresolved | paper:jiang2023_motiongpt, paper:li2025_unisign | idea:002 |
| G4 | Is there a principled way to align sign language embedding space with text LM space beyond next-token prediction? | partially_addressed | paper:li2025_unisign, paper:guo2025_pseudogloss | — |
| G5 | Why is VAE-embedding-based SLT at ~1-2 B@4 when video-encoder-based methods reach ~12 B@4 on CSL-Daily? | unresolved | paper:sincan2025_unbiased_eval | — |
| G6 | Does representation density in pre-computed VAE embeddings (not video) cause the same downstream translation failure as in video encoders? | resolved_no | paper:ye2024_signcl | — |
| G7 | Is Conv1d on flattened pose the architectural bottleneck? Would a graph-structured encoder (ST-GCN on 6D rotation) produce semantically richer VAE embeddings that enable unified AR? | resolved_no | paper:li2025_unisign | idea:005 |
| G8 | Our LFQ tokens are "acoustic class" (reconstruction-trained) not "semantic class" (self-supervised, like HuBERT). Can ST-GCN encoder alone push them into the semantic class needed for unified AR? | confirmed_negative | paper:zhang2023_speechgpt, paper:hassid2024_voxtlm | idea:005, idea:006 |
| G9 | Does the LFQ codebook discriminate between different sign sentences (oracle BLEU on discrete token space)? | resolved_negative_qwen_only | — | — |
| G10 | ~~Would continuous VAE embeddings enable translation that discrete LFQ tokens cannot?~~ **INVALID**: mT5 already uses encode_continuous() continuous embeddings. Real question: Why can't mT5 convert discriminative continuous embeddings into translation? | reframed→G11 | — | — |
| G11 | Why does mT5 fail to translate using discriminative continuous VAE embeddings (proven 62% 1024-way classification accuracy)? Is it H1 (local decodability), H4 (data scale 18k), or projection misalignment? | open | — | — |

## Gap Details

### G1 — Tokenizer LM-Compatibility

**Description:** Reconstruction quality and LM-compatibility are not the same objective. The field lacks clear criteria for what makes a sign tokenizer "LM-friendly": is it codebook utilization, smoothness of the discrete trajectory, entropy of the token distribution, or something else?

**Why it matters:** Current work shows that improving reconstruction (claim:C1) did not improve LM generalization (claim:C2). Understanding what the LM actually needs from the tokenizer is prerequisite to fixing this.

**Unresolved because:** No existing paper directly studies this for sign language. Some work exists for image/video tokenizers (MAGVIT-v2, BSQ) but the sign language case has additional structure (temporal, multi-body-part, linguistic).

---

### G2 — Full-Train Generalization Failure Mechanism

**Description:** Both Qwen (decoder-only) and mT5 (seq2seq) fail to generalize on full-dataset training despite (a) correct pipeline (exp:002), (b) active use of motion embeddings (claim:C4), and (c) discriminative embeddings (claim:C5). The failure mechanism is unknown.

**Why it matters:** Without understanding *why* generalization fails, it is impossible to systematically fix it. Current experiments rule out several hypotheses but the root cause remains open.

**Candidate hypotheses:**
1. Data volume is simply insufficient for emergent sign-text binding in current model sizes
2. The training objective (pure next-token or seq2seq CE) is misaligned with sign-text grounding
3. Task interference in multi-task (t2m+m2t+mc) setup hurts individual task performance
4. Sign token distribution has structural properties that break LM assumptions (non-Markovian, high entropy)

---

### G3 — Unified Sign Language AR Training at Current Data Scale

**Description:** MotionGPT succeeded for general body motion with HumanML3D (~15k samples), but sign language datasets are smaller and the task is harder (multilingual, fine-grained hand articulation, semantic binding). The right training recipe for unified sign language AR models at current scale is unknown.

---

### G4 — Sign-Text Space Alignment Beyond Next-Token Prediction

**Description:** Uni-Sign achieves good SLT by aligning pre-training and fine-tuning objectives. But it doesn't generate motion. A model that must both generate signs and translate them needs a deeper form of cross-modal alignment that next-token prediction alone may not provide.

---

### G5 — VAE-Embedding vs. Video-Encoder SLT Gap

**Description:** The best gloss-free SLT methods using video encoders (S3D + contrastive) reach ~12 B@4 on CSL-Daily. Our pipeline using pre-computed VAE embeddings (skipping video) achieves ~1.7 B@4. The gap (~10 B@4) is unexplained.

**Why it matters:** If the gap is due to information loss from VAE quantization/compression, then no amount of LM improvement will fix it. The tokenizer is the bottleneck, not the LM.

**Candidate causes:**
1. VAE embeddings have lower temporal resolution (downsampled) vs. S3D frame-level features
2. VAE codebook collapse or high representation density → LM can't distinguish signs
3. CSL-Daily training split used differently (video encoder methods vs. our pose-based pipeline)
4. Missing spatial/visual features: VAE operates on pose, losing appearance info that helps translation

**Needed experiment:** Run GFSLT-VLP-style evaluation on CSL-Daily with identical preprocessing to get a calibrated baseline; compare to our VAE pipeline.

---

### G6 — Representation Density in VAE Embedding Space

**Description:** SignCL (Ye et al. 2024) shows that gloss-free video encoders produce embeddings where 92.59% of features are semantically similar, preventing downstream translation. The analogous metric for our VAE embedding space is unknown.

**Why it matters:** If our 1536-dim VAE embeddings are even denser than video encoders, contrastive pre-training at the sequence level (as in M2) cannot help because the LM still sees an undifferentiated stream.

**Needed experiment (R015):** Compute pairwise cosine similarity distribution of our VAE embeddings per sample (temporal density); compare before/after contrastive pre-training; compare to sign VAE embeddings from SOKE original.

---

### G9 — LFQ Token Semantic Discriminability (Qwen 路径专用)

**Description:** R024 oracle decoder 测量 LFQ 离散 token 空间的检索上界：Conv1d oracle BLEU4=0.195，ST-GCN=0.121，均接近零。高熵（均匀分布）+ 低检索 BLEU4 = token 在运动学轴方向均匀分布，但在语义轴方向无区分性。

**Scope 限制（重要）：** G9 仅适用于使用离散 LFQ token 的路径（即 Qwen）。**mT5 使用 `encode_continuous()` 连续嵌入，不经过 LFQ codebook，G9 对 mT5 失败分析无关。**

**Status:** resolved_negative_qwen_only — 对 Qwen 路径，LFQ token 不具备语义区分能力。

---

### G11 — 为什么 mT5 无法将有区分能力的连续 VAE 嵌入转化为翻译能力？

**Description:** mT5 的输入管线使用 `encode_continuous()`（连续隐变量，非 LFQ token）。已有证据：跨签名者分类器在 1024-way 任务上测试准确率 62%（`docs/research_history_zh.md` 第 8.5 节），shuffle 测试确认 mT5 确实引用了运动信息（打乱后 BLEU4 从 ~2 降至 ~0.2）。然而 mT5 微调后翻译性能仍只有 ~1.7 BLEU4，远低于视频编码器方法的 12 BLEU4。

**Why it matters:** 如果嵌入质量已足够（可区分、被模型引用），那么失败原因必然在训练过程本身：数据量（18k vs 100k+）、MLP projection 对齐质量、或序列级信息整合方式。理解此点是继续改进的前提。

**候选假说（基于 R025 更新）：**
1. **H5（核心瓶颈）** — 连续 VAE 嵌入的**句间语义密度**过高：不同手语句子在余弦空间 mean NN cosine=0.87，近乎无法区分（R025 证实）。mT5 已接近理论上界，增加训练量或改进 LM 帮助有限。
2. **H4** — 数据量是否有边际帮助：即使嵌入密集，更多样本可能让 mT5 学到更细粒度统计规律（R026 验证中）
3. ~~H1（全局 InfoNCE ≠ 局部可解码性）~~ → 已被 R025 精确化：问题在嵌入层，不在 LM 层

**R025 已完成（2026-04-11）**：
- Conv1d oracle BLEU4=1.890，mT5 baseline=1.742 → oracle 仅领先 0.15，mT5 已近理论上界
- Mean NN cosine=0.87：不同句子嵌入在余弦空间高度密集
- **核心诊断**：连续 VAE 嵌入在句间维度不具备语义区分能力；需要判别性序列级预训练

**R026 运行中**：1k/5k/10k/18k 数据量学习曲线

**下一步（基于 R025 结论）**：
- 需要让相同语义手语嵌入更近、不同语义更远 → 序列级对比学习（而非批次级 InfoNCE）
- 或利用现有 mT5 微调的分类梯度反向优化 projection 层
