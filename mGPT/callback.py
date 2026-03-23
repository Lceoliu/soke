import os
import time
from pathlib import Path
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, TQDMProgressBar, ModelCheckpoint


def build_callbacks(cfg, logger=None, phase='test', **kwargs):
    callbacks = []
    logger = logger

    # Rich Progress Bar
    callbacks.append(progressBar())

    # Checkpoint Callback
    if phase == 'train':
        ckpt_cfg = cfg.TRAIN.get("CHECKPOINT", {})
        if bool(ckpt_cfg.get("SAVE_ON_TRAIN_END", True)):
            callbacks.append(FinalCheckpointCallback())
        if bool(ckpt_cfg.get("SAVE_BEFORE_VAL", True)):
            callbacks.append(PreValidationCheckpointCallback())
        callbacks.extend(getCheckpointCallback(cfg, logger=logger, **kwargs))
        
    return callbacks

def getCheckpointCallback(cfg, logger=None, **kwargs):
    callbacks = []
    ckpt_cfg = cfg.TRAIN.get("CHECKPOINT", {})
    periodic_every_n_epochs = int(ckpt_cfg.get("PERIODIC_EVERY_N_EPOCHS", 5) or 0)
    enable_rolling_last = bool(ckpt_cfg.get("ENABLE_ROLLING_LAST", True))
    # Logging
    metric_monitor = {
        "loss_total": "total/train",
        "Train_jf": "recons/text2jfeats/train",
        "Val_jf": "recons/text2jfeats/val",
        "Train_rf": "recons/text2rfeats/train",
        "Val_rf": "recons/text2rfeats/val",
        "APE root": "Metrics/APE_root",
        "APE mean pose": "Metrics/APE_mean_pose",
        "AVE root": "Metrics/AVE_root",
        "AVE mean pose": "Metrics/AVE_mean_pose",
        "R_TOP_1": "Metrics/R_precision_top_1",
        "R_TOP_2": "Metrics/R_precision_top_2",
        "R_TOP_3": "Metrics/R_precision_top_3",
        "gt_R_TOP_3": "Metrics/gt_R_precision_top_3",
        "FID": "Metrics/FID",
        "gt_FID": "Metrics/gt_FID",
        "Diversity": "Metrics/Diversity",
        "MM dist": "Metrics/Matching_score",
        "Accuracy": "Metrics/accuracy",
        "how2sign_DTW_MPJPE_PA_lhand": "Metrics/how2sign_DTW_MPJPE_PA_lhand",
        "how2sign_DTW_MPJPE_PA_rhand": "Metrics/how2sign_DTW_MPJPE_PA_rhand",
        "how2sign_DTW_MPJPE_PA_body": "Metrics/how2sign_DTW_MPJPE_PA_body",
        "csl_DTW_MPJPE_PA_lhand": "Metrics/csl_DTW_MPJPE_PA_lhand",
        "csl_DTW_MPJPE_PA_rhand": "Metrics/csl_DTW_MPJPE_PA_rhand",
        "csl_DTW_MPJPE_PA_body": "Metrics/csl_DTW_MPJPE_PA_body",
        "phoenix_DTW_MPJPE_PA_lhand": "Metrics/phoenix_DTW_MPJPE_PA_lhand",
        "phoenix_DTW_MPJPE_PA_rhand": "Metrics/phoenix_DTW_MPJPE_PA_rhand",
        "phoenix_DTW_MPJPE_PA_body": "Metrics/phoenix_DTW_MPJPE_PA_body",
        "how2sign_MPVPE_PA_all": "Metrics/how2sign_MPVPE_PA_all",
        "how2sign_MPJPE_PA_hand": "Metrics/how2sign_MPJPE_PA_hand",
        "csl_MPVPE_PA_all": "Metrics/csl_MPVPE_PA_all",
        "csl_MPJPE_PA_hand": "Metrics/csl_MPJPE_PA_hand",
        "phoenix_MPVPE_PA_all": "Metrics/phoenix_MPVPE_PA_all",
        "phoenix_MPJPE_PA_hand": "Metrics/phoenix_MPJPE_PA_hand",
        "BLEU_1": "Metrics/Bleu_1",
        "BLEU_2": "Metrics/Bleu_2",
        "BLEU_3": "Metrics/Bleu_3",
        "BLEU_4": "Metrics/Bleu_4",
        "val_t2m_loss": "val/t2m_loss",
        "val_m2t_loss": "val/m2t_loss",
        "val_mc_loss": "val/mc_loss",
        "val_t2m_ppl": "val/t2m_ppl",
        "val_m2t_ppl": "val/m2t_ppl",
        "val_mc_ppl": "val/mc_ppl",
        "how2sign_BLEU_1": "Metrics/how2sign_Bleu_1",
        "how2sign_BLEU_4": "Metrics/how2sign_Bleu_4",
        "csl_BLEU_1": "Metrics/csl_Bleu_1",
        "csl_BLEU_4": "Metrics/csl_Bleu_4",
        "phoenix_BLEU_1": "Metrics/phoenix_Bleu_1",
        "phoenix_BLEU_4": "Metrics/phoenix_Bleu_4",
        "ROUGE_L": "Metrics/ROUGE_L",
        "mc_how2sign_DTW_MPJPE_PA_lhand": "Metrics/mc_how2sign_DTW_MPJPE_PA_lhand",
        "mc_csl_DTW_MPJPE_PA_lhand": "Metrics/mc_csl_DTW_MPJPE_PA_lhand",
        "mc_phoenix_DTW_MPJPE_PA_lhand": "Metrics/mc_phoenix_DTW_MPJPE_PA_lhand",
    }
    callbacks.append(
        progressLogger(logger,metric_monitor=metric_monitor,log_every_n_steps=1))

    # # Save latest checkpoints
    # checkpointParams = {
    #     'dirpath': os.path.join(cfg.FOLDER_EXP, "checkpoints"),
    #     'filename': "{epoch}",
    #     'monitor': "step",
    #     'mode': "max",
    #     'every_n_epochs': cfg.LOGGER.VAL_EVERY_STEPS,
    #     'save_top_k': 8,
    #     'save_last': True,
    #     'save_on_train_epoch_end': True
    # }
    # callbacks.append(ModelCheckpoint(**checkpointParams))

    # # Save checkpoint every n*10 epochs
    # checkpointParams.update({
    #     'every_n_epochs': cfg.LOGGER.VAL_EVERY_STEPS * 10,
    #     'save_top_k': -1,
    #     'save_last': False
    # })
    # callbacks.append(ModelCheckpoint(**checkpointParams))

    checkpoint_dir = os.path.join(cfg.FOLDER_EXP, "checkpoints")

    if enable_rolling_last:
        last_checkpoint_params = {
            'dirpath': checkpoint_dir,
            'filename': "{epoch}",
            'monitor': "step",
            'mode': "max",
            'every_n_epochs': None,
            'save_top_k': 0,
            'save_last': True,
            'save_on_train_epoch_end': False
        }
        callbacks.append(ModelCheckpoint(**last_checkpoint_params))

    if periodic_every_n_epochs > 0:
        periodic_checkpoint_params = {
            'dirpath': checkpoint_dir,
            'filename': "epoch{epoch:04d}",
            'monitor': "step",
            'mode': "max",
            'every_n_epochs': periodic_every_n_epochs,
            'save_top_k': -1,
            'save_last': False,
            'save_on_train_epoch_end': True,
        }
        callbacks.append(ModelCheckpoint(**periodic_checkpoint_params))

    checkpointParams = {
        'dirpath': checkpoint_dir,
        'filename': "{epoch}",
        'monitor': "step",
        'mode': "max",
        'every_n_epochs': None,  #cfg.LOGGER.VAL_EVERY_STEPS,
        'save_top_k': 1,
        'save_last': False,
        'save_on_train_epoch_end': False
    }

    # Only VAE stage logs `total/val`. LM validation is metric-only and does not
    # produce a validation loss scalar, so monitoring `total/val` there would crash.
    if cfg.TRAIN.STAGE == 'vae':
        val_loss_ckpt_params = {
            'dirpath': os.path.join(cfg.FOLDER_EXP, "checkpoints"),
            'filename': "min-val_loss-{epoch}",
            'monitor': "total/val",
            'mode': "min",
            'save_top_k': 1,
            'save_last': False,
            'save_on_train_epoch_end': False,
            'every_n_epochs': None,
        }
        callbacks.append(ModelCheckpoint(**val_loss_ckpt_params))

    metrics = cfg.METRIC.TYPE
    metric_monitor_map = {
        'TemosMetric': {
            'Metrics/APE_root': {
                'abbr': 'APEroot',
                'mode': 'min'
            },
        },
        'TM2TMetrics': {
            'Metrics/how2sign_DTW_MPJPE_PA_lhand': {
                'abbr': 'how2sign_DTW_MPJPE_PA_lhand',
                'mode': 'min'
            },
            # 'Metrics/how2sign_DTW_MPJPE_PA_body': {
            #     'abbr': 'how2sign_DTW_MPJPE_PA_body',
            #     'mode': 'min'
            # },
            'Metrics/csl_DTW_MPJPE_PA_lhand': {
                'abbr': 'csl_DTW_MPJPE_PA_lhand',
                'mode': 'min'
            },
            # 'Metrics/csl_DTW_MPJPE_PA_body': {
            #     'abbr': 'csl_DTW_MPJPE_PA_body',
            #     'mode': 'min'
            # }
            'Metrics/phoenix_DTW_MPJPE_PA_lhand': {
                'abbr': 'phoenix_DTW_MPJPE_PA_lhand',
                'mode': 'min'
            },
            # 'Metrics/phoenix_DTW_MPJPE_PA_body': {
            #     'abbr': 'phoenix_DTW_MPJPE_PA_body',
            #     'mode': 'min'
            # }
        },
        'M2TMetrics': {
            'Metrics/Bleu_4': {
                'abbr': 'BLEU_4',
                'mode': 'max'
            },
            'Metrics/how2sign_Bleu_4': {
                'abbr': 'how2sign_BLEU_4',
                'mode': 'max'
            },
            'Metrics/csl_Bleu_4': {
                'abbr': 'csl_BLEU_4',
                'mode': 'max'
            },
            'Metrics/phoenix_Bleu_4': {
                'abbr': 'phoenix_BLEU_4',
                'mode': 'max'
            },
            'Metrics/ROUGE_L': {
                'abbr': 'ROUGE_L',
                'mode': 'max'
            },
        },
        'MCMetrics': {
            'Metrics/mc_how2sign_DTW_MPJPE_PA_lhand': {
                'abbr': 'mc_how2sign_DTW_MPJPE_PA_lhand',
                'mode': 'min'
            },
            'Metrics/mc_csl_DTW_MPJPE_PA_lhand': {
                'abbr': 'mc_csl_DTW_MPJPE_PA_lhand',
                'mode': 'min'
            },
            'Metrics/mc_phoenix_DTW_MPJPE_PA_lhand': {
                'abbr': 'mc_phoenix_DTW_MPJPE_PA_lhand',
                'mode': 'min'
            },
        },
        'MRMetrics': {
            'Metrics/how2sign_MPJPE_PA_hand': {
                'abbr': 'how2sign_MPJPE_PA_hand',
                'mode': 'min'
            },
            # 'Metrics/how2sign_MPVPE_PA_all': {
            #     'abbr': 'how2sign_MPVPE_PA_all',
            #     'mode': 'min'
            # },
            'Metrics/csl_MPJPE_PA_hand': {
                'abbr': 'csl_MPJPE_PA_hand',
                'mode': 'min'
            },
            # 'Metrics/csl_MPVPE_PA_all': {
            #     'abbr': 'csl_MPVPE_PA_all',
            #     'mode': 'min'
            # },
            'Metrics/phoenix_MPJPE_PA_hand': {
                'abbr': 'phoenix_MPJPE_PA_hand',
                'mode': 'min'
            },
            # 'Metrics/phoenix_MPVPE_PA_all': {
            #     'abbr': 'phoenix_MPVPE_PA_all',
            #     'mode': 'min'
            # },
        },
        'HUMANACTMetrics': {
            'Metrics/Accuracy': {
                'abbr': 'Accuracy',
                'mode': 'max'
            }
        },
        'UESTCMetrics': {
            'Metrics/Accuracy': {
                'abbr': 'Accuracy',
                'mode': 'max'
            }
        },
        'UncondMetrics': {
            'Metrics/FID': {
                'abbr': 'FID',
                'mode': 'min'
            }
        }
    }

    # checkpointParams.update({
    #     'every_n_epochs': None,  #cfg.LOGGER.VAL_EVERY_STEPS,
    #     'save_top_k': 1,
    # })

    for metric in metrics:
        if metric in metric_monitor_map.keys():
            metric_monitors = metric_monitor_map[metric]

            # Delete R3 if training VAE
            if cfg.TRAIN.STAGE == 'vae' and metric == 'TM2TMetrics':
                del metric_monitors['Metrics/R_precision_top_3']

            for metric_monitor in metric_monitors:
                checkpointParams.update({
                    'filename':
                    metric_monitor_map[metric][metric_monitor]['mode']
                    + "-" +
                    metric_monitor_map[metric][metric_monitor]['abbr']
                    + "{epoch}",
                    'monitor':
                    metric_monitor,
                    'mode':
                    metric_monitor_map[metric][metric_monitor]['mode'],
                })
                callbacks.append(
                    ModelCheckpoint(**checkpointParams))
    return callbacks

class progressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__(refresh_rate=1, process_position=0, leave=True)
        self._train_epoch_start_time = None
        self._val_epoch_start_time = None

    @staticmethod
    def _format_metric(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            value = value.detach().float().item()
        try:
            return f"{float(value):.3e}"
        except Exception:
            return None

    def _maybe_disable_non_global_zero(self, trainer: Trainer) -> None:
        if not getattr(trainer, "is_global_zero", True):
            self.disable()

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._maybe_disable_non_global_zero(trainer)
        super().on_train_start(trainer, pl_module)

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._maybe_disable_non_global_zero(trainer)
        super().on_validation_start(trainer, pl_module)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._maybe_disable_non_global_zero(trainer)
        super().on_test_start(trainer, pl_module)

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._maybe_disable_non_global_zero(trainer)
        super().on_predict_start(trainer, pl_module)

    def get_metrics(self, trainer, model):
        # Don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items

    def _build_train_description(self, trainer: Trainer) -> str:
        epoch = int(trainer.current_epoch)
        max_epochs = int(trainer.max_epochs) if trainer.max_epochs is not None else None
        desc = f"Train E{epoch}"
        if max_epochs is not None and max_epochs > 0:
            desc += f"/{max_epochs - 1}"
        if self._train_epoch_start_time is not None:
            elapsed_min = (time.perf_counter() - self._train_epoch_start_time) / 60.0
            desc += f" [{elapsed_min:.1f}m]"
        train_loss = self._format_metric(trainer.callback_metrics.get("total/train", None))
        if train_loss is not None:
            desc += f" loss {train_loss}"
        return desc

    def _build_val_description(self, trainer: Trainer) -> str:
        epoch = int(trainer.current_epoch)
        max_epochs = int(trainer.max_epochs) if trainer.max_epochs is not None else None
        desc = f"Val E{epoch}"
        if max_epochs is not None and max_epochs > 0:
            desc += f"/{max_epochs - 1}"
        if self._val_epoch_start_time is not None:
            elapsed_min = (time.perf_counter() - self._val_epoch_start_time) / 60.0
            desc += f" [{elapsed_min:.1f}m]"
        return desc

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._train_epoch_start_time = time.perf_counter()
        super().on_train_epoch_start(trainer, pl_module)
        if getattr(self, "train_progress_bar", None) is not None:
            self.train_progress_bar.set_description(self._build_train_description(trainer))

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if getattr(self, "train_progress_bar", None) is not None:
            self.train_progress_bar.set_description(self._build_train_description(trainer))

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._val_epoch_start_time = time.perf_counter()
        super().on_validation_epoch_start(trainer, pl_module)
        if getattr(self, "val_progress_bar", None) is not None:
            self.val_progress_bar.set_description(self._build_val_description(trainer))

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if getattr(self, "val_progress_bar", None) is not None:
            self.val_progress_bar.set_description(self._build_val_description(trainer))

class progressLogger(Callback):
    def __init__(self,
                 logger,
                 metric_monitor: dict,
                 precision: int = 3,
                 log_every_n_steps: int = 1):
        # Metric to monitor
        self.logger = logger
        self.metric_monitor = metric_monitor
        self.precision = precision
        self.log_every_n_steps = log_every_n_steps
        self._last_val_epoch = None
        self._train_epoch_start_time = None
        self._val_epoch_start_time = None

    @staticmethod
    def _should_log(trainer: Trainer) -> bool:
        return bool(getattr(trainer, "is_global_zero", True))

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule,
                       **kwargs) -> None:
        if not self._should_log(trainer):
            return
        self.logger.info("Training started")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule,
                     **kwargs) -> None:
        if not self._should_log(trainer):
            return
        self.logger.info("Training done")

    def on_validation_epoch_end(self, trainer: Trainer,
                                pl_module: LightningModule, **kwargs) -> None:
        if not self._should_log(trainer):
            return
        if trainer.sanity_checking:
            self.logger.info("Sanity checking ok.")
            return
        self._last_val_epoch = int(trainer.current_epoch)
        metric_format = f"{{:.{self.precision}e}}"
        val_duration = None
        if self._val_epoch_start_time is not None:
            val_duration = time.perf_counter() - self._val_epoch_start_time

        metrics_str = []
        losses_dict = trainer.callback_metrics
        for metric_name, dico_name in self.metric_monitor.items():
            if not self._is_validation_metric(dico_name):
                continue
            if dico_name not in losses_dict:
                continue
            metric = losses_dict[dico_name].item()
            metric = metric_format.format(metric)
            metrics_str.append(f"{metric_name} {metric}")

        line = f"Val Epoch {trainer.current_epoch}"
        if val_duration is not None:
            line += f" [{val_duration / 60.0:.1f} min]"
        if metrics_str:
            line += ": " + "   ".join(metrics_str)
        self.logger.info(line)
        self._val_epoch_start_time = None

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule, **kwargs) -> None:
        self._train_epoch_start_time = time.perf_counter()

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule, **kwargs) -> None:
        if trainer.sanity_checking:
            return
        self._val_epoch_start_time = time.perf_counter()

    @staticmethod
    def _is_validation_metric(metric_key: str) -> bool:
        metric_key = str(metric_key)
        return (
            metric_key.startswith("Metrics/")
            or metric_key.startswith("val/")
            or metric_key.endswith("/val")
            or "/val/" in metric_key
        )

    def on_train_epoch_end(self,
                           trainer: Trainer,
                           pl_module: LightningModule,
                           padding=False,
                           **kwargs) -> None:
        self._train_epoch_start_time = None


class FinalCheckpointCallback(Callback):
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule, **kwargs) -> None:
        ckpt_dir = Path(trainer.default_root_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(ckpt_dir / "last.ckpt"), weights_only=False)


class PreValidationCheckpointCallback(Callback):
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule, **kwargs) -> None:
        if trainer.sanity_checking:
            return
        ckpt_dir = Path(trainer.default_root_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(ckpt_dir / "last.ckpt"), weights_only=False)
