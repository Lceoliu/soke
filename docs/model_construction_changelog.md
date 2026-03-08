# Model Construction Changelog

本文档记录 LFQ 分支下模型结构的增量变化，供协作者快速定位“为什么架构与旧版本不同”。

---

## 2026-03-09

### Changed
- 将文档从日期命名迁移为语义命名：`model_construction_changelog.md`。
- 明确当前 tokenizer 主路径为 LFQ（`mGPT/archs/mgpt_vq.py` + `mGPT/archs/tools/quantize_lfq.py`）。

### Architecture Snapshot
- 总体前向：`preprocess -> Encoder -> quantize_in -> ResidualLFQ -> quantize_out -> Decoder -> postprocess`。
- `Encoder`：1D Conv + `down_t` 次下采样块 + ResNet1D。
- `Decoder`：ResNet1D + 上采样块 + 输出卷积。

### Key Structural Behavior
- Decoder 上采样模式为 `linear`，且 `align_corners=False`。
- Decoder 支持 `return_hidden=True`，用于接触预测旁支。

---

## 2026-03-02

### Added
- 在 VAE 解码器末端隐层引入可选 `contact_head`：
  - 结构：`Conv1d(width,128,3,pad=1) -> ReLU -> Conv1d(128,3,1)`
  - 输出：`contact_logits [B,3,T]`

### Notes
- `contact_head` 仅在 `use_contact=True` 时启用。

---

## 2026-02-27

### Added
- 将量化器实现切换到 `ResidualLFQ`（多层残差 lookup-free quantization）。
- 支持 `num_quantizers` 多层离散化，token 形状从 `[B,T]` 扩展到 `[B,T,Q]`。

### Encoder/Decoder Shape Rule
- 默认 `down_t=2, stride_t=2` 时，编码时间步约为 `floor(T/4)`。
- 对应多层 token 常见形状：`[B, floor(T/4), Q]`。

---

## 兼容性说明

### 旧 checkpoint decoder key 兼容
- 旧命名：`decoder.model.*`
- 新命名：`decoder.input_proj / decoder.upsample_blocks / decoder.pre_out / decoder.out_proj`
- 兼容逻辑：`mGPT/utils/load_checkpoint.py::_remap_legacy_decoder_keys`

### 协作建议
- 修改 `down_t/stride_t` 后，务必同步检查 token 时间长度与 LM 侧处理。
- 修改 `num_quantizers` 后，务必验证 token flatten/unflatten 逻辑（LM 侧 shared Q）。

