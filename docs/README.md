# 文档总览

当前仓库文档（按推荐阅读顺序）：

1. `docs/research_history_zh.md`  
   研究主线总览：从原始 SOKE 到当前 `Qwen / mT5 / CSL classifier` 的研究路径、实验结论和代码落点。

2. `docs/soke_code_guide_zh.md`  
   代码主手册：当前仓库主模块、训练入口、数据流与配置关系。

3. `docs/model_construction_changelog.md`  
   模型结构细节：Encoder/Decoder/LFQ 量化器、shape 推导、checkpoint 兼容说明。

4. `docs/loss_update_changelog.md`  
   Loss 细节：FK hand、加速度、contact 监督与配置项说明。

5. `docs/vae_scaling_pipeline_zh.md`  
   大规模数据 pipeline：manifest 预处理、多卡预训练、手语微调，以及 LFQ 下游衔接。

6. `docs/data_onboarding_guide_zh.md`  
   自定义数据接入指南：从视频+SMPL-X 到可训练数据目录。

7. `docs/qwen_overfit_diagnosis_plan.md`  
   Qwen 下游专项诊断：记录 overfit 修复路径与关键工程结论。偏历史归档，不作为当前总览文档。

8. `docs/mt5_m2t_quick_ablation_plan.md`  
   mT5 早期 quick ablation 草案。偏历史归档，需与 `research_history_zh.md` 一起阅读。
