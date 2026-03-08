# 文档总览

当前仓库文档（按推荐阅读顺序）：

1. `docs/soke_code_guide_zh.md`  
   主手册：最新 LFQ 架构、关键改动、训练与评估全流程（含下游 LM 自动脚本）。

2. `docs/model_construction_changelog.md`  
   模型结构细节：Encoder/Decoder/LFQ 量化器、shape 推导、checkpoint 兼容说明。

3. `docs/loss_update_changelog.md`  
   Loss 细节：FK hand、加速度、contact 监督与配置项说明。

4. `docs/vae_scaling_pipeline_zh.md`  
   大规模数据 pipeline：manifest 预处理、多卡预训练、手语微调，以及 LFQ 下游衔接。

5. `docs/data_onboarding_guide_zh.md`  
   自定义数据接入指南：从视频+SMPL-X 到可训练数据目录。
