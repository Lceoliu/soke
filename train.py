import os
import glob
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from omegaconf import OmegaConf
from mGPT.callback import build_callbacks
from mGPT.config import parse_args, instantiate_from_config
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.utils.logger import create_logger
from mGPT.utils.load_checkpoint import load_pretrained, load_pretrained_vae

def main():
    # Configs
    cfg = parse_args(phase="train")  # parse config file
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.USE_GPUS
    # print(cfg)

    # Logger
    logger = create_logger(cfg, phase="train")  # create logger
    logger.info(
        "Run config: "
        f"NAME={cfg.NAME} "
        f"STAGE={cfg.TRAIN.STAGE} "
        f"VAL_EVERY_EPOCHS={cfg.LOGGER.VAL_EVERY_STEPS} "
        f"EXP_DIR={cfg.FOLDER_EXP}"
    )

    # Seed
    pl.seed_everything(cfg.SEED_VALUE)

    # Environment Variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Metric Logger
    pl_loggers = []
    for loggerName in cfg.LOGGER.TYPE:
        if loggerName == 'tenosrboard' or cfg.LOGGER.WANDB.params.project:
            pl_logger = instantiate_from_config(
                eval(f'cfg.LOGGER.{loggerName.upper()}'))
            pl_loggers.append(pl_logger)

    # Callbacks
    callbacks = build_callbacks(cfg, logger=logger, phase='train')
    logger.info("Callbacks initialized")

    # Dataset
    datamodule = build_data(cfg)
    logger.info("datasets module {} initialized".format("".join(
        cfg.DATASET.target.split('.')[-2])))

    # Model
    model = build_model(cfg, datamodule)
    logger.info("model {} loaded".format(cfg.model.target))

    # Lightning Trainer
    if len(cfg.DEVICE) > 1:
        if "DDP_FIND_UNUSED_PARAMETERS" in cfg.TRAIN:
            find_unused = bool(cfg.TRAIN.DDP_FIND_UNUSED_PARAMETERS)
        else:
            # VAE stage usually uses all params and can save memory with find_unused_parameters=False.
            find_unused = False if str(cfg.TRAIN.STAGE) == "vae" else True
        strategy = f"ddp_find_unused_parameters_{str(find_unused).lower()}"
    else:
        strategy = "auto"

    trainer_kwargs = dict(
        default_root_dir=cfg.FOLDER_EXP,
        max_epochs=cfg.TRAIN.END_EPOCH,
        logger=pl_loggers,
        callbacks=callbacks,
        check_val_every_n_epoch=cfg.LOGGER.VAL_EVERY_STEPS,
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        num_nodes=cfg.NUM_NODES,
        strategy=strategy,
        benchmark=False,
        deterministic=False,
        num_sanity_val_steps=int(cfg.TRAIN.get("NUM_SANITY_VAL_STEPS", 0)),
        accumulate_grad_batches=int(cfg.TRAIN.get("ACCUMULATE_GRAD_BATCHES", 1)),
    )
    if cfg.PRECISION is not None:
        trainer_kwargs["precision"] = cfg.PRECISION

    trainer = pl.Trainer(**trainer_kwargs)
    logger.info("Trainer initialized")

    # Strict load pretrianed model
    # 只在非RESUME模式下加载，RESUME时由trainer.fit自动加载
    if cfg.TRAIN.PRETRAINED and not cfg.TRAIN.RESUME:
        load_pretrained(cfg, model, logger)

    # Strict load vae model (supports per-module checkpoint control)
    if (
        cfg.TRAIN.PRETRAINED_VAE
        or cfg.TRAIN.get("PRETRAINED_VAE_BODY", "")
        or cfg.TRAIN.get("PRETRAINED_VAE_HAND", "")
        or cfg.TRAIN.get("PRETRAINED_VAE_RHAND", "")
    ):
        load_pretrained_vae(cfg, model, logger)

    # Pytorch 2.0 Compile
    # if torch.__version__ >= "2.0.0":
    #     model = torch.compile(model, mode="reduce-overhead")
    # model = torch.compile(model)

    print('tmax: ', cfg.TRAIN.LR_SCHEDULER.params.T_max)
    # Lightning Fitting
    if cfg.TRAIN.RESUME:
        trainer.fit(model,
                    datamodule=datamodule,
                    ckpt_path=cfg.TRAIN.PRETRAINED,
                    weights_only=False)
    else:
        trainer.fit(model, datamodule=datamodule)

    # Training ends
    logger.info(
        f"The outputs of this experiment are stored in {cfg.FOLDER_EXP}")
    logger.info("Training ends!")


if __name__ == "__main__":
    main()
