from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/vae/motionx_vae_pretrain.yaml")
cfg.TEST.CHECKPOINTS = "experiments/mgpt/VAE_MOTIONX_PRETRAIN/checkpoints/min-how2sign_MPJPE_PA_handepoch=259.ckpt"
cfg.TEST.SPLIT = "test"
cfg.TEST.SAVE_PREDICTIONS = True
cfg.TEST.REPLICATION_TIMES = 1
OmegaConf.save(cfg, "/tmp/motionx_vae_eval_test.yaml")
