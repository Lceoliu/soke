import os
import math
import numpy as np
import torch
import logging
from pathlib import Path
from pytorch_lightning import LightningModule
from os.path import join as pjoin
from collections import OrderedDict
from mGPT.metrics import BaseMetrics
from mGPT.config import get_obj_from_str
import json, pickle
from copy import deepcopy
import torch.distributed as dist


class BaseModel(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.configure_metrics()

        # Ablation
        self.test_step_outputs = []
        self.all_name2scores = {}
        self.times = []
        self.rep_i = 0
        cfg = self.hparams.cfg
        self.output_dir = Path(
            os.path.join(
                cfg.FOLDER,
                str(cfg.model.target.split('.')[-2].lower()),
                str(cfg.NAME)
            ))
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _safe_rank() -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    def training_step(self, batch, batch_idx):
        return self.allsplit_step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.allsplit_step("val", batch, batch_idx, dataloader_idx=dataloader_idx)

    def test_step(self, batch, batch_idx):
        outputs = self.allsplit_step("test", batch, batch_idx)

        name = outputs['name']
        feats_rst = outputs['feats_rst'].detach().cpu().numpy()
        feats_ref = outputs['feats_ref'].detach().cpu().numpy()
        text = outputs['text']
        ref_len = outputs['lengths']
        rst_len = outputs['lengths_rst']
        rank = self._safe_rank()
        save_dir = os.path.join(self.output_dir, f'{self.hparams.cfg.TEST.SPLIT}_rank_{rank}')
        os.makedirs(save_dir, exist_ok=True)
        if self.hparams.cfg.TEST.SAVE_PREDICTIONS:
            for i in range(len(name)):
                cur_n = name[i].split('/')[-1]
                f_rst = feats_rst[i, :rst_len[i]]
                f_ref = feats_ref[i, :ref_len[i]]
                t = text[i]
                pkl_dict = {'feats_rst': f_rst, 'feats_ref': f_ref, 'text': t}
                with open(os.path.join(save_dir, f'{cur_n}.pkl'), 'wb') as f:
                    pickle.dump(pkl_dict, f)
        # self.test_step_outputs.append(outputs)
        return outputs

    def predict_step(self, batch, batch_idx):
        return self.forward(batch)

    def on_train_epoch_end(self):
        # Log steps and losses
        dico = self.step_log_dict()
        # Log losses
        dico.update(self.loss_log_dict('train'))
        # Write to log only if not sanity check
        if not self.trainer.sanity_checking:
            self.log_dict(dico, sync_dist=True, rank_zero_only=True)
        # dist.barrier()

    def on_validation_epoch_end(self):
        # Log steps and losses
        dico = self.step_log_dict()
        # Log losses
        dico.update(self.loss_log_dict('train'))
        if 'lm' not in str(self.hparams.stage):
            dico.update(self.loss_log_dict('val'))
        # Log metrics
        dico.update(self.metrics_log_dict())
        # print('dico', dico)
        # Write to log only if not sanity check
        if not self.trainer.sanity_checking:
            self.log_dict(dico, sync_dist=True, rank_zero_only=True)
            # print('dico', dico)
        # dist.barrier()
        # print(f'{dist.get_rank()}: val epoch end')

    def on_test_epoch_end(self):
        #print before sync
        if 'lm' in self.hparams.stage and hasattr(self.metrics, "TM2TMetrics"):
            name2scores = getattr(self.metrics.TM2TMetrics, 'name2scores')
            metrics = ["how2sign_DTW_MPJPE_PA_lhand", "how2sign_DTW_MPJPE_PA_rhand", "how2sign_DTW_MPJPE_PA_body", 
                           "csl_DTW_MPJPE_PA_lhand", "csl_DTW_MPJPE_PA_rhand", "csl_DTW_MPJPE_PA_body",
                           "phoenix_DTW_MPJPE_PA_lhand", "phoenix_DTW_MPJPE_PA_rhand", "phoenix_DTW_MPJPE_PA_body"]
            scores, count = {}, {}
            for m in metrics:
                scores[m] = count[m] = 0
            for name, value_dict in name2scores.items():
                for n, val in value_dict.items():
                    scores[n] = scores[n] + val
                    count[n] = count[n] + 1
            for k in scores.keys():
                scores[k] = scores[k] / max(count[k], 1)
            print('rank: ', self._safe_rank(), scores)

        # Log metrics
        dico = self.metrics_log_dict()
        # Write to log only if not sanity check
        if not self.trainer.sanity_checking:
            self.log_dict(dico, sync_dist=True, rank_zero_only=True)
        # self.save_npy(self.test_step_outputs)

        # save prediction
        rank = self._safe_rank()
        save_dir = os.path.join(self.output_dir, f'{self.hparams.cfg.TEST.SPLIT}_rank_{rank}')
        os.makedirs(save_dir, exist_ok=True)
        if 'lm' in self.hparams.stage and hasattr(self.metrics, "TM2TMetrics"):
            with open(os.path.join(save_dir, 'test_scores.json'), 'w') as f:
                json.dump(getattr(self.metrics.TM2TMetrics, 'name2scores'), f)
        elif 'vae' in self.hparams.stage:
            with open(os.path.join(save_dir, 'test_scores.json'), 'w') as f:
                json.dump(getattr(self.metrics.MRMetrics, 'name2scores'), f)
        
        self.rep_i = self.rep_i + 1
        # Free up the memory
        self.test_step_outputs.clear()
        self.all_name2scores = {}

    def preprocess_state_dict(self, state_dict):
        new_state_dict = OrderedDict()
        
        metric_state_dict = self.metrics.state_dict()
        loss_state_dict = self._losses.state_dict()

        for k, v in metric_state_dict.items():
            new_state_dict['metrics.' + k] = v

        for k, v in loss_state_dict.items():
            new_state_dict['_losses.' + k] = v

        for k, v in state_dict.items():
            if '_losses' not in k and 'Metrics' not in k:
                new_state_dict[k] = v

        return new_state_dict

    def load_state_dict(self, state_dict, strict=True):
        new_state_dict = self.preprocess_state_dict(state_dict)
        super().load_state_dict(new_state_dict, strict)

    def step_log_dict(self):
        return {
            "epoch": float(self.trainer.current_epoch),
            "step": float(self.trainer.current_epoch)
        }

    def loss_log_dict(self, split: str):
        losses = self._losses['losses_' + split]
        loss_dict = losses.compute(split)
        return loss_dict

    def metrics_log_dict(self):

        # For TM2TMetrics MM
        if self.trainer.datamodule.is_mm and "TM2TMetrics" in self.hparams.metrics_dict:
            metrics_dicts = ['MMMetrics']
        else:
            metrics_dicts = self.hparams.metrics_dict

        # Compute all metrics
        metrics_log_dict = {}
        for metric in metrics_dicts:
            metrics_dict = getattr(
                self.metrics,
                metric).compute(sanity_flag=self.trainer.sanity_checking)
            metrics_log_dict.update({
                f"Metrics/{metric}": value.item()
                for metric, value in metrics_dict.items()
            })

        return metrics_log_dict
    
    def configure_optimizers(self):
        # Optimizer
        optim_target = self.hparams.cfg.TRAIN.OPTIM.target
        if len(optim_target.split('.')) == 1:
            optim_target = 'torch.optim.' + optim_target
        optim_params = dict(self.hparams.cfg.TRAIN.OPTIM.params)
        param_groups = self._build_optimizer_param_groups()
        optimizer = get_obj_from_str(optim_target)(
            params=param_groups if param_groups is not None else self.parameters(),
            **optim_params,
        )

        # Scheduler
        lr_scheduler = self._build_lr_scheduler(optimizer)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def _build_optimizer_param_groups(self):
        cfg = self.hparams.cfg
        stage = str(getattr(self.hparams, "stage", cfg.TRAIN.STAGE))
        pg_cfg = cfg.TRAIN.OPTIM.get("PARAM_GROUPS", {})
        if stage not in ["lm_instruct", "lm_pretrain", "lm_rl"]:
            return None
        if not bool(pg_cfg.get("ENABLE_DIFF_LR", False)):
            return None

        base_lr = float(cfg.TRAIN.OPTIM.params.lr)
        embed_mult = float(pg_cfg.get("EMBED_LR_MULT", 1.0))
        lm_head_mult = float(pg_cfg.get("LM_HEAD_LR_MULT", 1.0))
        backbone_mult = float(pg_cfg.get("BACKBONE_LR_MULT", 1.0))

        groups = {
            "backbone": [],
            "embed_tokens": [],
            "lm_head": [],
        }
        seen = set()
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            pid = id(param)
            if pid in seen:
                continue
            seen.add(pid)

            if "embed_tokens" in name:
                groups["embed_tokens"].append(param)
            elif "lm_head" in name:
                groups["lm_head"].append(param)
            else:
                groups["backbone"].append(param)

        param_groups = []
        if groups["backbone"]:
            param_groups.append({
                "params": groups["backbone"],
                "lr": base_lr * backbone_mult,
                "group_name": "backbone",
                "lr_mult": backbone_mult,
            })
        if groups["embed_tokens"]:
            param_groups.append({
                "params": groups["embed_tokens"],
                "lr": base_lr * embed_mult,
                "group_name": "embed_tokens",
                "lr_mult": embed_mult,
            })
        if groups["lm_head"]:
            param_groups.append({
                "params": groups["lm_head"],
                "lr": base_lr * lm_head_mult,
                "group_name": "lm_head",
                "lr_mult": lm_head_mult,
            })

        return param_groups if len(param_groups) > 0 else None

    def _build_lr_scheduler(self, optimizer):
        cfg = self.hparams.cfg
        scheduler_target = cfg.TRAIN.LR_SCHEDULER.target
        if len(scheduler_target.split('.')) == 1:
            scheduler_target = 'torch.optim.lr_scheduler.' + scheduler_target

        pg_cfg = cfg.TRAIN.OPTIM.get("PARAM_GROUPS", {})
        warm_cfg = pg_cfg.get("FAST_WARMUP", {})
        enable_diff_lr = bool(pg_cfg.get("ENABLE_DIFF_LR", False))
        enable_fast_warmup = bool(warm_cfg.get("ENABLED", False))
        is_cosine = scheduler_target.endswith("CosineAnnealingLR")

        if not (enable_diff_lr and enable_fast_warmup and is_cosine):
            return get_obj_from_str(scheduler_target)(
                optimizer=optimizer, **cfg.TRAIN.LR_SCHEDULER.params)

        total_epochs = int(cfg.TRAIN.LR_SCHEDULER.params.T_max)
        eta_min = float(cfg.TRAIN.LR_SCHEDULER.params.get("eta_min", 0.0))
        start_factor = float(warm_cfg.get("START_FACTOR", 0.2))
        base_warmup_epochs = max(int(warm_cfg.get("BASE_WARMUP_EPOCHS", 0)), 0)
        head_warmup_epochs = max(int(warm_cfg.get("EMBED_LM_HEAD_WARMUP_EPOCHS", 0)), 0)

        def build_group_lambda(initial_lr, min_lr, warmup_epochs):
            if initial_lr <= 0:
                return lambda epoch: 1.0

            def lr_lambda(epoch):
                epoch = int(epoch)
                if warmup_epochs > 0 and epoch < warmup_epochs:
                    progress = float(epoch + 1) / float(max(warmup_epochs, 1))
                    warm_lr = initial_lr * (start_factor + (1.0 - start_factor) * progress)
                    return warm_lr / initial_lr

                cosine_total = max(total_epochs - warmup_epochs, 1)
                progress = min(max(epoch - warmup_epochs, 0), cosine_total)
                cosine = min_lr + 0.5 * (initial_lr - min_lr) * (
                    1.0 + math.cos(math.pi * float(progress) / float(cosine_total))
                )
                return cosine / initial_lr

            return lr_lambda

        lambdas = []
        for group in optimizer.param_groups:
            initial_lr = float(group["lr"])
            group_mult = float(group.get("lr_mult", 1.0))
            group_name = str(group.get("group_name", "backbone"))
            warmup_epochs = head_warmup_epochs if group_name in ["embed_tokens", "lm_head"] else base_warmup_epochs
            min_lr = eta_min * group_mult
            lambdas.append(build_group_lambda(initial_lr, min_lr, warmup_epochs))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)

    def configure_metrics(self):
        self.metrics = BaseMetrics(datamodule=self.datamodule, **self.hparams)

    def save_npy(self, outputs):
        cfg = self.hparams.cfg
        output_dir = Path(
            os.path.join(
                cfg.FOLDER,
                str(cfg.model.target.split('.')[-2].lower()),
                str(cfg.NAME),
                "samples_" + cfg.TIME,
            ))
        if cfg.TEST.SAVE_PREDICTIONS:
            lengths = [i[1] for i in outputs]
            outputs = [i[0] for i in outputs]

            if cfg.TEST.DATASETS[0].lower() in ["humanml3d", "kit"]:
                keyids = self.trainer.datamodule.test_dataset.name_list
                for i in range(len(outputs)):
                    for bid in range(
                            min(cfg.TEST.BATCH_SIZE, outputs[i].shape[0])):
                        keyid = keyids[i * cfg.TEST.BATCH_SIZE + bid]
                        data = self.trainer.datamodule.test_dataset.data_dict[
                            keyid]

                        motion = torch.tensor(data['motion'],
                                              device=outputs[i].device)
                        motion = self.datamodule.normalize(motion)
                        length = data['length']
                        text_list = data['text']
                        gen_joints = outputs[i][bid][:lengths[i][bid]].cpu(
                        ).numpy()
                        if cfg.TEST.REPLICATION_TIMES > 1:
                            name = f"{keyid}.npy"
                        else:
                            name = f"{keyid}.npy"
                        # save predictions results
                        npypath = output_dir / name
                        np.save(npypath, gen_joints)
                        npypath = output_dir / f"{keyid}_gt.npy"
                        joints = self.feats2joints(motion).cpu().numpy()
                        np.save(npypath, joints)

                        with open(output_dir / f"{keyid}.txt", "a") as f:
                            for text in text_list:
                                f.write(f"{text['caption']}\n")

            elif cfg.TEST.DATASETS[0].lower() in ["humanact12", "uestc"]:
                keyids = range(len(self.trainer.datamodule.test_dataset))
                for i in range(len(outputs)):
                    for bid in range(
                            min(cfg.TEST.BATCH_SIZE, outputs[i].shape[0])):
                        keyid = keyids[i * cfg.TEST.BATCH_SIZE + bid]
                        gen_joints = outputs[i][bid].cpu()
                        gen_joints = gen_joints.permute(2, 0,
                                                        1)[:lengths[i][bid],
                                                           ...].numpy()
                        if cfg.TEST.REPLICATION_TIMES > 1:
                            name = f"{keyid}_{self.rep_i}"
                        else:
                            name = f"{keyid}.npy"
                        # save predictions results
                        npypath = output_dir / name
                        np.save(npypath, gen_joints)
