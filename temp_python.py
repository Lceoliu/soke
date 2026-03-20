from omegaconf import OmegaConf

cfg = OmegaConf.load(
    '/home/SOKE/experiments/overfit_RE/SOKE_QWEN_CSL_OVERFIT4_M2T_RE/config_2026-03-20-15-31-56_train.yaml'
)
cfg.TRAIN.NUM_WORKERS = 0
cfg.TEST.BATCH_SIZE = 1
cfg.EVAL.BATCH_SIZE = 1
cfg.TEST.CHECKPOINTS = '/home/SOKE/experiments/overfit_RE/SOKE_QWEN_CSL_OVERFIT4_M2T_RE/checkpoints/last.ckpt'
cfg.TEST.SPLIT = 'train'
cfg.model.params.task = 'm2t'
cfg.METRIC.TYPE = []
OmegaConf.save(cfg, '/tmp/overfit_re_m2t_eval.yaml')
print('saved /tmp/overfit_re_m2t_eval.yaml')
