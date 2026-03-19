import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import time
from mGPT.config import instantiate_from_config
from os.path import join as pjoin
from mGPT.losses.mgpt import GPTLosses
from mGPT.models.base import BaseModel
from .base import BaseModel
import json
import mGPT.render.matplot.plot_3d_global as plot_3d
from mGPT.utils.human_models import get_coord, smpl_x


class MotionGPT(BaseModel):
    """
    Stage 1 Motion Tokenizer
    Stage 2 Motion-language pretrian
    Stage 3 Motion-language instruction tuning
    """

    def __init__(self,
                 cfg,
                 datamodule,
                 lm,
                 motion_vae,
                 codebook_size=512,
                 stage='vae',
                 debug=True,
                 condition='text',
                 task='t2m',
                 metrics_dict=['TM2TMetrics'],
                 **kwargs):

        self.save_hyperparameters(ignore='datamodule', logger=False)
        self.datamodule = datamodule
        super().__init__()

        # Instantiate motion tokenizer
        if motion_vae != None:
            self.vae = instantiate_from_config(motion_vae)
            lm['params']['motion_codebook_size'] = self.vae.code_num
        
        # additional hand vae
        self.hand_vae_cfg = kwargs.get('hand_vae_cfg', None)
        if self.hand_vae_cfg is not None:
            self.hand_vae = instantiate_from_config(self.hand_vae_cfg)
            lm['params']['hand_codebook_size'] = self.hand_vae.code_num
        
        self.rhand_vae_cfg = kwargs.get('rhand_vae_cfg', None)
        if self.rhand_vae_cfg is not None:
            self.rhand_vae = instantiate_from_config(self.rhand_vae_cfg)
            lm['params']['rhand_codebook_size'] = self.rhand_vae.code_num
        
        self.face_vae_cfg = kwargs.get('face_vae_cfg', None)
        if self.face_vae_cfg is not None:
            self.face_vae = instantiate_from_config(self.face_vae_cfg)

        self.lm_body_num_quantizers = int(getattr(self.vae, "num_quantizers", 1))
        self.lm_hand_num_quantizers = int(
            getattr(getattr(self, "hand_vae", None), "num_quantizers", self.lm_body_num_quantizers)
        )
        self.lm_rhand_num_quantizers = int(
            getattr(getattr(self, "rhand_vae", None), "num_quantizers", self.lm_body_num_quantizers)
        )
        q_candidates = [self.lm_body_num_quantizers]
        self.lm_num_token_parts = 1
        if self.hand_vae_cfg is not None:
            q_candidates.append(self.lm_hand_num_quantizers)
            self.lm_num_token_parts += 1
        if self.rhand_vae_cfg is not None:
            q_candidates.append(self.lm_rhand_num_quantizers)
            self.lm_num_token_parts += 1
        # For multi-head LM text formatting, all parts must share one temporal token length.
        # We use the minimum quantizer level count to avoid invalid shape coupling when parts differ.
        self.lm_shared_num_quantizers = int(min(q_candidates))

        # Freeze the motion tokenizer for lm training
        if 'lm' in self.hparams.stage:
            self.vae.training = False
            for p in self.vae.parameters():
                p.requires_grad = False
            if self.hand_vae_cfg is not None:
                self.hand_vae.training = False
                for p in self.hand_vae.parameters():
                    p.requires_grad = False
            if self.rhand_vae_cfg is not None:
                self.rhand_vae.training = False
                for p in self.rhand_vae.parameters():
                    p.requires_grad = False
            # Instantiate motion-language model
            self.lm = instantiate_from_config(lm)

        # Instantiate the losses
        self._losses = torch.nn.ModuleDict({
            split: GPTLosses(cfg, self.hparams.stage, self.datamodule.njoints)
            for split in ["losses_train", "losses_test", "losses_val"]
        })

        # Data transform
        self.feats2joints = datamodule.feats2joints
        self.lambda_fk_hand = float(cfg.LOSS.get("LAMBDA_FK_HAND", 0.0))
        self.lambda_accel_hand = float(cfg.LOSS.get("LAMBDA_ACCEL_HAND", 0.0))
        self.lambda_accel_wrist_rel = float(cfg.LOSS.get("LAMBDA_ACCEL_WRIST_REL", 0.0))
        self.lambda_contact = float(cfg.LOSS.get("LAMBDA_CONTACT", 0.0))
        self._use_hand_fk_supervision = (
            self.lambda_fk_hand > 0.0
            or self.lambda_accel_hand > 0.0
            or self.lambda_accel_wrist_rel > 0.0
        )
        self._use_contact_supervision = self.lambda_contact > 0.0
        self.register_buffer(
            "_smplx_shape_template",
            torch.tensor(
                [[[-0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172,
                   0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842]]],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_smplx_lhand_idx",
            torch.tensor(list(smpl_x.joint_part2idx["lhand"]), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_smplx_rhand_idx",
            torch.tensor(list(smpl_x.joint_part2idx["rhand"]), dtype=torch.long),
            persistent=False,
        )

        # Count codebook frequency
        self.codePred = []
        self.codeFrequency = torch.zeros((self.hparams.codebook_size, ))

    def _denormalize_motion(self, features: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.datamodule.hparams.mean, device=features.device, dtype=features.dtype)
        std = torch.as_tensor(self.datamodule.hparams.std, device=features.device, dtype=features.dtype)
        return features * std + mean

    def _compute_hand_fk_joints(self, features_norm: torch.Tensor):
        if features_norm.shape[-1] != 133:
            return None, None, None, None

        # FK path used by hand loss:
        # 1) Denormalize 133-dim training features back to SMPL-X pose space.
        # 2) Rebuild a full 169-dim SMPL-X parameter vector by prepending 36 zeros
        #    (keeps the same convention as datamodule.feats2joints).
        # 3) Run SMPL-X forward (get_coord) to obtain joints in camera-centered
        #    global coordinates.
        #
        # Origin / frame:
        # - Use wrist-relative hand coordinates for loss supervision:
        #   left_hand_rel = left_hand_joints - left_wrist,
        #   right_hand_rel = right_hand_joints - right_wrist.
        # - This removes global translation drift and focuses loss on hand articulation.
        #
        # Hand joint selection:
        # - Use smpl_x.joint_part2idx["lhand"/"rhand"] indices to avoid hard-coded ids.
        features = self._denormalize_motion(features_norm)
        bsz, tlen = features.shape[:2]
        zero_pose = torch.zeros((bsz, tlen, 36), device=features.device, dtype=features.dtype)
        shape_param = self._smplx_shape_template.to(features).repeat(bsz, tlen, 1).view(bsz * tlen, -1)
        features = torch.cat([zero_pose, features], dim=-1).view(bsz * tlen, -1)

        _, joints = get_coord(
            root_pose=features[..., 0:3],
            body_pose=features[..., 3:66],
            lhand_pose=features[..., 66:111],
            rhand_pose=features[..., 111:156],
            jaw_pose=features[..., 156:159],
            shape=shape_param,
            expr=features[..., 159:169],
            return_verts=False,
        )
        joints = joints.view(bsz, tlen, joints.shape[1], 3)
        joints_lhand = joints.index_select(2, self._smplx_lhand_idx.to(joints.device))
        joints_rhand = joints.index_select(2, self._smplx_rhand_idx.to(joints.device))
        l_wrist = joints[:, :, smpl_x.J_regressor_idx["lwrist"]:smpl_x.J_regressor_idx["lwrist"] + 1, :]
        r_wrist = joints[:, :, smpl_x.J_regressor_idx["rwrist"]:smpl_x.J_regressor_idx["rwrist"] + 1, :]
        joints_lhand = joints_lhand - l_wrist
        joints_rhand = joints_rhand - r_wrist
        return joints_lhand, joints_rhand, l_wrist, r_wrist

    def forward(self, batch, task="t2m"):
        texts = batch["text"]
        lengths_ref = batch["length"]

        # Forward
        # texts = ['Generate motion: ' + text for text in texts]
        outputs, output_texts = self.lm.generate_direct(texts, do_sample=True)

        # Motion Decode
        feats_rst_lst = []
        lengths = []
        max_len = 0

        for i in range(len(texts)):
            if task == "pred":
                motion = self.vae.decode(
                    torch.cat((batch["motion"][i], outputs[i])))
            elif task in ["t2m", "m2t", "inbetween"]:
                motion = self.vae.decode(outputs[i])
                # motion = self.datamodule.denormalize(motion)
                lengths.append(motion.shape[1])
            else:
                raise NotImplementedError

            if motion.shape[1] > max_len:
                max_len = motion.shape[1]

            if task in ["t2m", "m2t", "pred"]:
                feats_rst_lst.append(motion)

            elif task == "inbetween":
                motion = torch.cat(
                    (batch["motion_heading"][i][None],
                     motion[:, lengths_ref[i] // 4:lengths_ref[i] // 4 * 3,
                            ...], batch["motion_tailing"][i][None]),
                    dim=1)
                feats_rst_lst.append(motion)

        feats_rst = torch.zeros(
            (len(feats_rst_lst), max_len, motion.shape[-1])).to(self.device)

        # padding and concat
        for i in range(len(feats_rst_lst)):
            feats_rst[i, :feats_rst_lst[i].shape[1], ...] = feats_rst_lst[i]

        # Recover joints for evaluation
        joints_rst = self.feats2joints(feats_rst)

        # return set
        outputs = {
            "texts": output_texts,
            "feats": feats_rst,
            "joints": joints_rst,
            "length": lengths
        }

        return outputs

    def train_lm_forward(self, batch, forced_task=None):
        tokens_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        tasks = batch["tasks"]
        all_captions = batch['all_captions']
        tokens_ref, lengths = self._flatten_batch_tokens_for_lm(tokens_ref, lengths)
        if self.hparams.condition == 'caption':
            texts = [random.choice(all_captions[i]) for i in range(len(texts))]
        if forced_task is not None:
            tasks = [{"class": str(forced_task).lower()} for _ in range(len(texts))]

        # LLM Forward
        outputs = self.lm(texts, tokens_ref, lengths, tasks, src=batch['src'], name=batch['name'])
        # outputs = self.t2m_gpt.generate(texts)
        return {'outputs': outputs}

    def _set_lfq_temperature_progress(self, progress: float):
        for name in ["vae", "hand_vae", "rhand_vae", "face_vae"]:
            module = getattr(self, name, None)
            if module is None:
                continue
            quantizer = getattr(module, "quantizer", None)
            if quantizer is None:
                continue
            if hasattr(quantizer, "set_anneal_progress"):
                quantizer.set_anneal_progress(progress)

    @staticmethod
    def _resize_contact_logits(logits: torch.Tensor, target_t: int) -> torch.Tensor:
        if logits is None or logits.shape[-1] == target_t:
            return logits
        return F.interpolate(logits, size=target_t, mode='linear', align_corners=False)

    def _merge_contact_logits(self, aux_list, target_t: int):
        logits_list = []
        for aux in aux_list:
            if not isinstance(aux, dict):
                continue
            logits = aux.get("contact_logits", None)
            if logits is None:
                continue
            logits = self._resize_contact_logits(logits, target_t)
            logits_list.append(logits)
        if len(logits_list) == 0:
            return None
        if len(logits_list) == 1:
            merged = logits_list[0]
        else:
            merged = torch.stack(logits_list, dim=0).mean(dim=0)
        # [B, C=3, T] -> [B, T, C]
        return merged.permute(0, 2, 1).contiguous()

    def _flatten_batch_tokens_for_lm(self, tokens: torch.Tensor, lengths):
        if tokens is None:
            return tokens, lengths
        if not torch.is_tensor(tokens):
            return tokens, lengths

        # [B, T, Q, 3] -> [B, T*Q, 3]
        if tokens.dim() == 4:
            q_use = min(int(tokens.shape[2]), int(self.lm_shared_num_quantizers))
            if q_use > 0 and int(tokens.shape[2]) != q_use:
                tokens = tokens[:, :, :q_use, :]
            bsz, tlen, _, pnum = tokens.shape
            tokens = tokens.reshape(bsz, tlen * q_use, pnum)
            lengths = [int(l) * q_use for l in lengths]
            return tokens, lengths

        # [B, T, Q] (single stream multi-level) -> [B, T*Q]
        if tokens.dim() == 3:
            is_multihead_flat = (
                int(self.lm_num_token_parts) > 1
                and int(tokens.shape[-1]) == int(self.lm_num_token_parts)
            )
            if is_multihead_flat:
                return tokens, lengths
            q_use = min(int(tokens.shape[-1]), int(self.lm_body_num_quantizers))
            if q_use > 0 and int(tokens.shape[-1]) != q_use:
                tokens = tokens[:, :, :q_use]
            bsz, tlen, _ = tokens.shape
            tokens = tokens.reshape(bsz, tlen * q_use)
            lengths = [int(l) * q_use for l in lengths]
            return tokens, lengths

        return tokens, lengths

    @staticmethod
    def _flatten_single_tokens_for_lm(tokens: torch.Tensor, q_use: int):
        if tokens.dim() == 1:
            return tokens, int(tokens.shape[0])
        if tokens.dim() == 2:
            q_keep = min(int(tokens.shape[-1]), int(max(q_use, 1)))
            tokens = tokens[:, :q_keep].reshape(-1)
            return tokens, int(tokens.shape[0])
        return tokens.reshape(-1), int(tokens.numel())

    @staticmethod
    def _unflatten_single_tokens_from_lm(tokens: torch.Tensor, q_use: int):
        if tokens.dim() != 1:
            tokens = tokens.reshape(-1)
        q_use = int(max(q_use, 1))
        if q_use == 1:
            return tokens
        valid = (int(tokens.shape[0]) // q_use) * q_use
        if valid <= 0:
            return tokens[:1]
        return tokens[:valid].view(-1, q_use)

    def _encode_sign_tokens_from_motion(self, feats_ref: torch.Tensor):
        if self.hand_vae_cfg is None and self.rhand_vae_cfg is None:
            motion_token, _ = self.vae.encode(feats_ref)
            flat_motion, _ = self._flatten_single_tokens_for_lm(
                motion_token[0], self.lm_body_num_quantizers
            )
            return flat_motion

        if self.hand_vae_cfg is not None and self.rhand_vae_cfg is not None:
            feats_ref_lhand = feats_ref[..., 30:75]
            feats_ref_rhand = feats_ref[..., 75:120]
            feats_ref_re = torch.cat([feats_ref[..., :30], feats_ref[..., 120:]], dim=-1)
            token_body, _ = self.vae.encode(feats_ref_re)
            token_lhand, _ = self.hand_vae.encode(feats_ref_lhand)
            token_rhand, _ = self.rhand_vae.encode(feats_ref_rhand)
            q_use = int(self.lm_shared_num_quantizers)
            flat_body, _ = self._flatten_single_tokens_for_lm(token_body[0], q_use)
            flat_lhand, _ = self._flatten_single_tokens_for_lm(token_lhand[0], q_use)
            flat_rhand, _ = self._flatten_single_tokens_for_lm(token_rhand[0], q_use)
            min_len = min(flat_body.shape[0], flat_lhand.shape[0], flat_rhand.shape[0])
            return torch.stack(
                [
                    flat_body[:min_len],
                    flat_lhand[:min_len],
                    flat_rhand[:min_len],
                ],
                dim=-1,
            )

        feats_ref_hand = feats_ref[..., 30:120]
        feats_ref_re = torch.cat([feats_ref[..., :30], feats_ref[..., 120:]], dim=-1)
        token_body, _ = self.vae.encode(feats_ref_re)
        token_hand, _ = self.hand_vae.encode(feats_ref_hand)
        q_use = int(self.lm_shared_num_quantizers)
        flat_body, _ = self._flatten_single_tokens_for_lm(token_body[0], q_use)
        flat_hand, _ = self._flatten_single_tokens_for_lm(token_hand[0], q_use)
        min_len = min(flat_body.shape[0], flat_hand.shape[0])
        return torch.stack([flat_body[:min_len], flat_hand[:min_len]], dim=-1)

    def _resolve_eval_task_name(self, split: str, dataloader_idx: int = 0):
        if split != "val":
            return str(self.hparams.task).lower()
        if hasattr(self.datamodule, "get_val_task_name"):
            return str(self.datamodule.get_val_task_name(int(dataloader_idx))).lower()
        return str(self.hparams.task).lower()

    def _decode_generated_motion_parts(
        self,
        feats_ref: torch.Tensor,
        outputs_tokens,
        outputs_tokens_hand=None,
        outputs_tokens_rhand=None,
        base_lengths=None,
    ):
        B, _, C = feats_ref.shape
        if base_lengths is None:
            base_lengths = [1] * B
        rst_len = list(base_lengths)

        q_body_use = self.lm_body_num_quantizers
        q_hand_use = self.lm_hand_num_quantizers
        q_rhand_use = self.lm_rhand_num_quantizers
        if outputs_tokens_hand is not None or outputs_tokens_rhand is not None:
            q_body_use = q_hand_use = q_rhand_use = self.lm_shared_num_quantizers

        outputs_tokens = [self._unflatten_single_tokens_from_lm(tok, q_body_use) for tok in outputs_tokens]
        if outputs_tokens_hand is not None:
            outputs_tokens_hand = [
                self._unflatten_single_tokens_from_lm(tok, q_hand_use) for tok in outputs_tokens_hand
            ]
        if outputs_tokens_rhand is not None:
            outputs_tokens_rhand = [
                self._unflatten_single_tokens_from_lm(tok, q_rhand_use) for tok in outputs_tokens_rhand
            ]

        max_len = max(map(len, outputs_tokens)) if len(outputs_tokens) > 0 else 1
        if outputs_tokens_hand is not None and len(outputs_tokens_hand) > 0:
            max_len = max(max_len, max(map(len, outputs_tokens_hand)))
        if outputs_tokens_rhand is not None and len(outputs_tokens_rhand) > 0:
            max_len = max(max_len, max(map(len, outputs_tokens_rhand)))
        max_len = max(int(max_len) * 4, 1)

        feats_rst = torch.zeros(B, max_len, C).to(feats_ref)
        for i in range(B):
            body_tokens = torch.clamp(outputs_tokens[i], 0, self.vae.code_num - 1)
            if len(body_tokens) > 1:
                motion = self.vae.decode(body_tokens)
                rst_len[i] = motion.shape[1]
                motion = F.pad(motion, (0, 0, 0, max_len - motion.shape[1]), mode='replicate')
            else:
                if outputs_tokens_hand is None:
                    motion = torch.zeros((1, max_len, C), device=feats_ref.device, dtype=feats_ref.dtype)
                else:
                    motion = torch.zeros((1, max_len, self.vae.nfeats), device=feats_ref.device, dtype=feats_ref.dtype)
                rst_len[i] = 1
            feats_rst[i:i + 1, :, :30] = motion[..., :30]
            feats_rst[i:i + 1, :, -13:] = motion[..., 30:43]

            if outputs_tokens_hand is not None:
                hand_tokens = torch.clamp(outputs_tokens_hand[i], 0, self.hand_vae.code_num - 1)
                if len(hand_tokens) > 1:
                    motion_hand = self.hand_vae.decode(hand_tokens)
                    rst_len[i] = max(rst_len[i], motion_hand.shape[1])
                    motion_hand = F.pad(motion_hand, (0, 0, 0, max_len - motion_hand.shape[1]), mode='replicate')
                else:
                    motion_hand = torch.zeros((1, max_len, self.hand_vae.nfeats), device=feats_ref.device, dtype=feats_ref.dtype)
                feats_rst[i:i + 1, :, 30:30 + self.hand_vae.nfeats] = motion_hand

            if outputs_tokens_rhand is not None:
                rhand_tokens = torch.clamp(outputs_tokens_rhand[i], 0, self.rhand_vae.code_num - 1)
                if len(rhand_tokens) > 1:
                    motion_rhand = self.rhand_vae.decode(rhand_tokens)
                    rst_len[i] = max(rst_len[i], motion_rhand.shape[1])
                    motion_rhand = F.pad(motion_rhand, (0, 0, 0, max_len - motion_rhand.shape[1]), mode='replicate')
                else:
                    motion_rhand = torch.zeros((1, max_len, self.rhand_vae.nfeats), device=feats_ref.device, dtype=feats_ref.dtype)
                feats_rst[i:i + 1, :, 75:75 + self.rhand_vae.nfeats] = motion_rhand

        return feats_rst, rst_len

    @staticmethod
    def _build_suffix_reference_batch(feats_ref: torch.Tensor, lengths, ratio: float):
        B, _, C = feats_ref.shape
        ratio = float(ratio)
        suffix_list = []
        suffix_lengths = []
        for i in range(B):
            cur_len = int(lengths[i])
            if cur_len <= 1:
                split_idx = 0
            else:
                split_idx = int(round(cur_len * ratio))
                split_idx = min(max(split_idx, 1), cur_len - 1)
            cur_suffix = feats_ref[i, split_idx:cur_len]
            if cur_suffix.shape[0] <= 0:
                cur_suffix = feats_ref[i, cur_len - 1:cur_len]
            suffix_list.append(cur_suffix)
            suffix_lengths.append(int(cur_suffix.shape[0]))
        max_len = max(suffix_lengths) if len(suffix_lengths) > 0 else 1
        suffix_batch = feats_ref.new_zeros((B, max_len, C))
        for i, cur_suffix in enumerate(suffix_list):
            suffix_batch[i, :cur_suffix.shape[0]] = cur_suffix
        return suffix_batch, suffix_lengths

    def on_train_epoch_start(self):
        if str(self.hparams.stage) != "vae":
            return
        total_epochs = max(int(self.hparams.cfg.TRAIN.END_EPOCH), 1)
        progress = float(self.current_epoch) / float(max(total_epochs - 1, 1))
        self._set_lfq_temperature_progress(progress)

    @torch.no_grad()
    def val_t2m_forward(self, batch, vis=False):
        feats_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        tasks = None
        # if self.trainer.datamodule.is_mm:
        #     texts = texts * self.hparams.cfg.METRIC.MM_NUM_REPEATS
        #     feats_ref = feats_ref.repeat_interleave(
        #         self.hparams.cfg.METRIC.MM_NUM_REPEATS, dim=0)
        #     lengths = lengths * self.hparams.cfg.METRIC.MM_NUM_REPEATS
        #     instructions = pjoin(self.datamodule.hparams.data_root,
        #                          'template_instructions.json')
        #     instructions = json.load(open(instructions, 'r'))
        #     tasks = [instructions["Text-to-Motion"]["caption"]] * len(texts)

        if vis:
            func = max
        else:
            func = max

        if self.hparams.condition == 'caption':
            tasks = [{
                'input': ['<Caption_Placeholder>'],
                'output': ['']
            }] * len(texts)

        if self.hparams.cfg.DATASET.TASK_PATH:
            instructions = pjoin(self.hparams.cfg.DATASET.TASK_PATH)
            instructions = json.load(open(instructions, 'r'))
            tasks = [instructions["Text-to-Motion"]["t2m"]] * len(texts)

        rst_len = lengths.copy()
        # Forward
        gen_results = self.lm.generate_conditional(texts,
                                               lengths=lengths,
                                               stage='test',
                                               tasks=tasks,
                                               src=batch['src'],
                                               name=batch['name'])
        outputs_tokens = gen_results['outputs_tokens']
        outputs_tokens_hand = gen_results['outputs_tokens_hand']
        outputs_tokens_rhand = gen_results['outputs_tokens_rhand']
        feats_rst, rst_len = self._decode_generated_motion_parts(
            feats_ref=feats_ref,
            outputs_tokens=outputs_tokens,
            outputs_tokens_hand=outputs_tokens_hand,
            outputs_tokens_rhand=outputs_tokens_rhand,
            base_lengths=rst_len,
        )

        # Recover joints for evaluation
        vertices_ref, joints_ref = self.feats2joints(feats_ref)
        vertices_rst, joints_rst = self.feats2joints(feats_rst)

        # Renorm for evaluation
        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        # return set
        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "vertices_ref": vertices_ref,
            "vertices_rst": vertices_rst,
            "length": lengths,
            "lengths_rst": rst_len
            # "length": lengths
        }

        return rs_set

    @torch.no_grad()
    def val_m2t_forward(self, batch):
        feats_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        motion_tokens = []
        for i in range(len(feats_ref)):
            motion_tokens.append(self._encode_sign_tokens_from_motion(feats_ref[i:i + 1]))

        # Forward
        outputs = self.lm.generate_conditional(motion_tokens=motion_tokens,
                                               task="m2t",
                                               stage='test',
                                               src=batch['src'],
                                               name=batch['name'])
        # print(outputs, texts)
        # return set
        rs_set = {
            "m_ref": feats_ref,
            # "t_ref": all_captions,
            "t_ref": texts,
            "t_pred": outputs,
            "length": lengths
        }

        return rs_set

    @torch.no_grad()
    def val_mc_forward(self, batch):
        feats_ref_full = batch["motion"]
        lengths_full = batch["length"]
        ratio = float(getattr(self.lm, "mc_prefix_ratio", 0.5))
        motion_tokens = []
        for i in range(len(feats_ref_full)):
            motion_tokens.append(self._encode_sign_tokens_from_motion(feats_ref_full[i:i + 1]))

        gen_results = self.lm.generate_conditional(
            motion_tokens=motion_tokens,
            task="mc",
            stage='test',
            src=batch['src'],
            name=batch['name'],
        )
        feats_ref, lengths = self._build_suffix_reference_batch(feats_ref_full, lengths_full, ratio)
        feats_rst, rst_len = self._decode_generated_motion_parts(
            feats_ref=feats_ref,
            outputs_tokens=gen_results['outputs_tokens'],
            outputs_tokens_hand=gen_results.get('outputs_tokens_hand', None),
            outputs_tokens_rhand=gen_results.get('outputs_tokens_rhand', None),
            base_lengths=lengths,
        )

        vertices_ref, joints_ref = self.feats2joints(feats_ref)
        vertices_rst, joints_rst = self.feats2joints(feats_rst)

        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "vertices_ref": vertices_ref,
            "vertices_rst": vertices_rst,
            "length": lengths,
            "lengths_rst": rst_len,
        }
        return rs_set

    @torch.no_grad()
    def val_m2m_forward(self, batch, task="pred"):
        feats_ref = batch["motion"]
        lengths = batch["length"]

        # Motion Encode
        motion_tokens = []
        lengths_tokens = []
        for i in range(len(feats_ref)):
            motion_token, _ = self.vae.encode(feats_ref[i:i + 1])
            flat_motion, len_motion = self._flatten_single_tokens_for_lm(
                motion_token[0], self.lm_body_num_quantizers
            )
            motion_tokens.append(flat_motion)
            lengths_tokens.append(len_motion)

        # Forward
        outputs = self.lm.generate_conditional(motion_tokens=motion_tokens,
                                               lengths=lengths_tokens,
                                               task=task,
                                               stage='test')

        # Motion Decode
        feats_rst = torch.zeros_like(feats_ref)
        min_len = lengths.copy()

        for i in range(len(lengths)):
            outputs[i] = torch.clamp(outputs[i],
                                     0,
                                     self.hparams.codebook_size - 1,
                                     out=None)
            outputs[i] = self._unflatten_single_tokens_from_lm(
                outputs[i], self.lm_body_num_quantizers
            )

            if len(outputs[i]) > 1:
                motion = self.vae.decode(outputs[i])
            else:
                motion = torch.zeros_like(feats_ref[i:i + 1, ...])

            min_len[i] = min(motion.shape[1], lengths[i])

            # Cut Motion
            feats_rst[i:i + 1, :min_len[i], ...] = motion[:, :lengths[i]]

        # Recover joints for evaluation
        joints_ref = self.feats2joints(feats_ref)
        joints_rst = self.feats2joints(feats_rst)

        # Renorm for evaluation
        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        # return set
        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "length": min_len
            # "length": lengths
        }

        return rs_set

    def train_vae_forward(self, batch):
        # batch detach
        feats_ref = batch["motion"]
        joints_ref = None #self.feats2joints(feats_ref)
        feats_rst_hand = feats_rst_re = loss_commit_hand = loss_commit_re = perplexity_re = perplexity_hand = None
        fk_lhand_rst = fk_rhand_rst = fk_lhand_ref = fk_rhand_ref = None
        wrist_l_rst = wrist_r_rst = wrist_l_ref = wrist_r_ref = None
        gt_contact_labels = batch.get("gt_contact_labels", None)
        gt_contact_has_label = batch.get("gt_contact_has_label", None)
        need_contact_aux = self._use_contact_supervision and (gt_contact_labels is not None)
        contact_aux_list = []
        contact_logits = None

        def _vae_forward(module, x):
            if need_contact_aux:
                x_rst, x_commit, x_perplex, x_aux = module(x, return_aux=True)
                return x_rst, x_commit, x_perplex, x_aux
            x_rst, x_commit, x_perplex = module(x)
            return x_rst, x_commit, x_perplex, None

        # motion encode & decode
        if self.hand_vae_cfg is None:
            feats_rst, loss_commit, perplexity, aux = _vae_forward(self.vae, feats_ref)
            contact_aux_list.append(aux)
        elif self.rhand_vae_cfg is None:
            feats_ref_hand = feats_ref[..., 30:120]
            feats_ref_re = torch.cat([feats_ref[..., :30], feats_ref[..., 120:]], dim=-1)
            feats_rst_hand, loss_commit_hand, perplexity_hand, aux_hand = _vae_forward(self.hand_vae, feats_ref_hand)
            feats_rst_re, loss_commit_re, perplexity_re, aux_re = _vae_forward(self.vae, feats_ref_re)
            feats_rst = torch.cat([feats_rst_re[..., :30], feats_rst_hand, feats_rst_re[..., 30:]], dim=-1)
            loss_commit = loss_commit_hand + loss_commit_re
            perplexity = perplexity_hand + perplexity_re
            contact_aux_list.extend([aux_hand, aux_re])
        elif self.face_vae_cfg is None:
            feats_ref_lhand = feats_ref[..., 30:75]
            feats_ref_rhand = feats_ref[..., 75:120]
            feats_ref_re = torch.cat([feats_ref[..., :30], feats_ref[..., 120:]], dim=-1)
            feats_rst_lhand, loss_commit_lhand, perplexity_lhand, aux_lhand = _vae_forward(self.hand_vae, feats_ref_lhand)
            feats_rst_rhand, loss_commit_rhand, perplexity_rhand, aux_rhand = _vae_forward(self.rhand_vae, feats_ref_rhand)
            feats_rst_re, loss_commit_re, perplexity_re, aux_re = _vae_forward(self.vae, feats_ref_re)
            feats_rst = torch.cat([feats_rst_re[..., :30], feats_rst_lhand, feats_rst_rhand, feats_rst_re[..., 30:]], dim=-1)
            loss_commit = loss_commit_lhand + loss_commit_rhand + loss_commit_re
            perplexity = perplexity_lhand + perplexity_rhand + perplexity_re
            contact_aux_list.extend([aux_lhand, aux_rhand, aux_re])
        else:
            feats_ref_lhand = feats_ref[..., 30:75]
            feats_ref_rhand = feats_ref[..., 75:120]
            feats_ref_face = feats_ref[..., 123:]
            feats_ref_re = torch.cat([feats_ref[..., :30], feats_ref[..., 120:123]], dim=-1)
            feats_rst_lhand, loss_commit_lhand, perplexity_lhand, aux_lhand = _vae_forward(self.hand_vae, feats_ref_lhand)
            feats_rst_rhand, loss_commit_rhand, perplexity_rhand, aux_rhand = _vae_forward(self.rhand_vae, feats_ref_rhand)
            feats_rst_face, loss_commit_face, perplexity_face, aux_face = _vae_forward(self.face_vae, feats_ref_face)
            feats_rst_re, loss_commit_re, perplexity_re, aux_re = _vae_forward(self.vae, feats_ref_re)
            feats_rst = torch.cat([feats_rst_re[..., :30], feats_rst_lhand, feats_rst_rhand, feats_rst_re[..., 30:], feats_rst_face], dim=-1)
            loss_commit = loss_commit_lhand + loss_commit_rhand + loss_commit_re + loss_commit_face
            perplexity = perplexity_lhand + perplexity_rhand + perplexity_re + perplexity_face
            contact_aux_list.extend([aux_lhand, aux_rhand, aux_face, aux_re])

        if need_contact_aux:
            contact_logits = self._merge_contact_logits(contact_aux_list, target_t=feats_ref.shape[1])

        if self._use_hand_fk_supervision and feats_ref.shape[-1] == 133:
            # Compute FK joints in one batched SMPL-X pass for both prediction/reference
            # to reduce overhead and guarantee identical FK pipeline.
            feats_both = torch.cat([feats_rst, feats_ref.detach()], dim=0)
            joints_lhand_both, joints_rhand_both, joints_lwrist_both, joints_rwrist_both = self._compute_hand_fk_joints(feats_both)
            if (
                joints_lhand_both is not None and joints_rhand_both is not None
                and joints_lwrist_both is not None and joints_rwrist_both is not None
            ):
                bsz = feats_ref.shape[0]
                fk_lhand_rst = joints_lhand_both[:bsz]
                fk_rhand_rst = joints_rhand_both[:bsz]
                fk_lhand_ref = joints_lhand_both[bsz:].detach()
                fk_rhand_ref = joints_rhand_both[bsz:].detach()
                wrist_l_rst = joints_lwrist_both[:bsz]
                wrist_r_rst = joints_rwrist_both[:bsz]
                wrist_l_ref = joints_lwrist_both[bsz:].detach()
                wrist_r_ref = joints_rwrist_both[bsz:].detach()

        joints_rst = None #self.feats2joints(feats_rst)
        # return set
        rs_set = {
            "m_ref": feats_ref,
            "joints_ref": joints_ref,
            "m_rst": feats_rst,
            "joints_rst": joints_rst,
            "loss_commit": loss_commit,
            "perplexity": perplexity,
            "length": batch['length'],
            "fk_lhand_rst": fk_lhand_rst,
            "fk_rhand_rst": fk_rhand_rst,
            "fk_lhand_ref": fk_lhand_ref,
            "fk_rhand_ref": fk_rhand_ref,
            "wrist_l_rst": wrist_l_rst,
            "wrist_r_rst": wrist_r_rst,
            "wrist_l_ref": wrist_l_ref,
            "wrist_r_ref": wrist_r_ref,
            "contact_logits": contact_logits,
            "gt_contact_labels": gt_contact_labels,
            "gt_contact_has_label": gt_contact_has_label,
        }
        return rs_set


    @torch.no_grad()
    def val_vae_forward(self, batch, split="train", stage=None):
        # Detach batch
        feats_ref = batch["motion"]
        lengths = batch["length"]

        # Repeat for multimodal evaluation
        if stage!='demo' and self.trainer.datamodule.is_mm:
            feats_ref = feats_ref.repeat_interleave(
                self.hparams.cfg.METRIC.MM_NUM_REPEATS, dim=0)
            lengths = lengths * self.hparams.cfg.METRIC.MM_NUM_REPEATS

        # Motion encode & decode
        feats_rst = torch.zeros_like(feats_ref)
        code_pred_all = []

        for i in range(len(feats_ref)):
            if lengths[i] == 0:
                continue
            if self.hand_vae_cfg is None:
                feats_pred, _, _ = self.vae(feats_ref[i:i + 1, :lengths[i]])
                code_pred, _ = self.vae.encode(feats_ref[i:i + 1, :lengths[i]])
                code_pred_all.append(code_pred[0].tolist())
            elif self.rhand_vae_cfg is None:
                feats_ref_hand = feats_ref[i:i + 1, :lengths[i], 30:120]
                feats_ref_re = torch.cat([feats_ref[i:i + 1, :lengths[i], :30], feats_ref[i:i + 1, :lengths[i], 120:]], dim=-1)

                feats_pred_hand, _, _ = self.hand_vae(feats_ref_hand)
                feats_pred, _, _ = self.vae(feats_ref_re)
                feats_pred = torch.cat([feats_pred[..., :30], feats_pred_hand, feats_pred[..., 30:]], dim=-1)

                code_pred_hand, _ = self.hand_vae.encode(feats_ref_hand)
                code_pred_re, _ = self.vae.encode(feats_ref_re)
                code_pred_all.append([code_pred_hand[0].tolist(), code_pred_re[0].tolist()])
            elif self.face_vae_cfg is None:
                feats_ref_lhand = feats_ref[i:i + 1, :lengths[i], 30:75]
                feats_ref_rhand = feats_ref[i:i + 1, :lengths[i], 75:120]
                feats_ref_re = torch.cat([feats_ref[i:i + 1, :lengths[i], :30], feats_ref[i:i + 1, :lengths[i], 120:]], dim=-1)

                feats_pred_lhand, _, _ = self.hand_vae(feats_ref_lhand)
                feats_pred_rhand, _, _ = self.rhand_vae(feats_ref_rhand)
                feats_pred, _, _ = self.vae(feats_ref_re)
                feats_pred = torch.cat([feats_pred[..., :30], feats_pred_lhand, feats_pred_rhand, feats_pred[..., 30:]], dim=-1)
                # print(feats_pred.shape)

                code_pred_hand, _ = self.hand_vae.encode(feats_ref_lhand)
                code_pred_re, _ = self.vae.encode(feats_ref_re)
                code_pred_all.append([code_pred_hand[0].tolist(), code_pred_re[0].tolist()])
            else:
                feats_ref_lhand = feats_ref[i:i + 1, :lengths[i], 30:75]
                feats_ref_rhand = feats_ref[i:i + 1, :lengths[i], 75:120]
                feats_ref_re = torch.cat([feats_ref[i:i + 1, :lengths[i], :30], feats_ref[i:i + 1, :lengths[i], 120:123]], dim=-1)
                feats_ref_face = feats_ref[i:i + 1, :lengths[i], 123:]

                feats_pred_lhand, _, _ = self.hand_vae(feats_ref_lhand)
                feats_pred_rhand, _, _ = self.rhand_vae(feats_ref_rhand)
                feats_pred_face, _, _ = self.face_vae(feats_ref_face)
                feats_pred, _, _ = self.vae(feats_ref_re)
                feats_pred = torch.cat([feats_pred[..., :30], feats_pred_lhand, feats_pred_rhand, feats_pred[..., 30:], feats_pred_face], dim=-1)
                # print(feats_pred.shape)

                code_pred_hand, _ = self.hand_vae.encode(feats_ref_lhand)
                code_pred_re, _ = self.vae.encode(feats_ref_re)
                code_pred_all.append([code_pred_hand[0].tolist(), code_pred_re[0].tolist()])

            feats_rst[i:i + 1, :feats_pred.shape[1], :] = feats_pred

            # codeFre_pred = torch.bincount(code_pred[0],
            #                               minlength=self.hparams.codebook_size).to(
            #                                   self.codeFrequency.device)
            # self.codePred.append(code_pred[0])
            # self.codeFrequency += codeFre_pred

        # np.save('../memData/results/codeFrequency.npy',
        #         self.codeFrequency.cpu().numpy())

        # Recover joints for evaluation
        vertices_ref, joints_ref = self.feats2joints(feats_ref)
        vertices_rst, joints_rst = self.feats2joints(feats_rst)
        # print(vertices_ref.shape, vertices_rst.shape)
        # print(joints_ref.shape, joints_rst.shape)

        # Renorm for evaluation
        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        # Return set
        rs_set = {
            "m_ref": feats_ref,
            "joints_ref": joints_ref,
            "vertices_ref": vertices_ref,
            "m_rst": feats_rst,
            "joints_rst": joints_rst,
            "vertices_rst": vertices_rst,
            "length": lengths,
            "code_pred": code_pred_all
        }

        return rs_set
    

    def allsplit_step(self, split: str, batch, batch_idx, dataloader_idx: int = 0):
        # Compute the losses
        loss = None
        lengths = batch['length']
        src = batch['src']
        name = batch['name']
        # print('task: ', self.hparams.task)
        if self.hparams.stage == "vae" and split in ["train", "val"]:
            rs_set = self.train_vae_forward(batch)
            loss = self._losses['losses_' + split].update(rs_set)
        elif self.hparams.stage in ["lm_instruct", "lm_pretrain"] and split in ["train"]:
            rs_set = self.train_lm_forward(batch)
            loss = self._losses['losses_' + split].update(rs_set)

        # Compute the metrics
        if split in ["val", "test"]:
            if self.hparams.stage == "vae":
                rs_set = self.val_vae_forward(batch, split)
                getattr(self.metrics,'MRMetrics').update(
                            feats_rst=rs_set["m_rst"],
                            feats_ref=rs_set["m_ref"],
                            joints_rst=rs_set["joints_rst"],
                            joints_ref=rs_set["joints_ref"],
                            vertices_rst=rs_set["vertices_rst"],
                            vertices_ref=rs_set["vertices_ref"], 
                            lengths=lengths,
                            src=src,
                            name=name
                        )
            elif self.hparams.stage in ["lm_instruct", "lm_pretrain", "lm_rl"]:
                eval_task = self._resolve_eval_task_name(split, dataloader_idx=dataloader_idx)
                if split == "val":
                    rs_set_loss = self.train_lm_forward(batch, forced_task=eval_task)
                    cur_val_loss = rs_set_loss['outputs'].loss if hasattr(rs_set_loss['outputs'], "loss") else rs_set_loss['outputs']['loss']
                    cur_val_loss = cur_val_loss.detach().float()
                    cur_val_ppl = torch.exp(torch.clamp(cur_val_loss, max=20.0))
                    batch_size = int(lengths.shape[0]) if hasattr(lengths, "shape") else int(len(lengths))
                    self.log(
                        f"val/{eval_task}_loss",
                        cur_val_loss,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                        batch_size=batch_size,
                    )
                    self.log(
                        f"val/{eval_task}_ppl",
                        cur_val_ppl,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                        batch_size=batch_size,
                    )

                if eval_task == "t2m":
                    rs_set = self.val_t2m_forward(batch)
                    if hasattr(self.metrics, 'TM2TMetrics'):
                        getattr(self.metrics, 'TM2TMetrics').update(
                            feats_rst=rs_set["m_rst"],
                            feats_ref=rs_set["m_ref"],
                            joints_rst=rs_set["joints_rst"],
                            joints_ref=rs_set["joints_ref"],
                            vertices_rst=rs_set["vertices_rst"],
                            vertices_ref=rs_set["vertices_ref"],
                            lengths=lengths,
                            lengths_rst=rs_set['lengths_rst'],
                            split=split,
                            src=src,
                            name=name
                        )
                elif eval_task == "m2t":
                    rs_set_m2t = self.val_m2t_forward(batch)
                    if hasattr(self.metrics, 'M2TMetrics'):
                        getattr(self.metrics, 'M2TMetrics').update(
                            pred_texts=rs_set_m2t["t_pred"],
                            gt_texts=rs_set_m2t["t_ref"],
                            lengths=rs_set_m2t['length'],
                            src=src,
                        )
                elif eval_task == "mc":
                    rs_set_mc = self.val_mc_forward(batch)
                    if hasattr(self.metrics, 'MCMetrics'):
                        getattr(self.metrics, 'MCMetrics').update(
                            feats_rst=rs_set_mc["m_rst"],
                            feats_ref=rs_set_mc["m_ref"],
                            joints_rst=rs_set_mc["joints_rst"],
                            joints_ref=rs_set_mc["joints_ref"],
                            vertices_rst=rs_set_mc["vertices_rst"],
                            vertices_ref=rs_set_mc["vertices_ref"],
                            lengths=rs_set_mc["length"],
                            lengths_rst=rs_set_mc['lengths_rst'],
                            split=split,
                            src=src,
                            name=name
                        )
                # elif self.hparams.task in ["m2m", "pred", "inbetween"]:
                #     rs_set = self.val_m2m_forward(batch, self.hparams.task)

            # if self.hparams.task not in ["m2t"]:
            #     # MultiModality evaluation sperately
            #     if self.trainer.datamodule.is_mm:
            #         metrics_dicts = ['MMMetrics']
            #     else:
            #         metrics_dicts = self.hparams.metrics_dict
                    
            #     if self.hparams.task not in ['pred', 'inbetween'] and 'PredMetrics' in metrics_dicts:
            #         metrics_dicts.remove('PredMetrics')

            #     for metric in metrics_dicts:
            #         lengths = batch['length']
            #         if metric == "TemosMetric":
            #             getattr(self.metrics,
            #                     metric).update(rs_set["joints_rst"],
            #                                    rs_set["joints_ref"], lengths)
            #         elif metric == "TM2TMetrics":
            #             if self.hparams.stage in [
            #                     "lm_instruct", "lm_pretrain", "lm_rl"
            #             ]:
            #                 getattr(self.metrics, metric).update(
            #                     joints_rst=rs_set["joints_rst"], joints_ref=rs_set["joints_ref"],
            #                     vertices_rst=rs_set["vertices_rst"], vertices_ref=rs_set["vertices_ref"],
            #                     lengths=lengths,
            #                 )
            #         elif metric == "UncondMetrics":
            #             getattr(self.metrics, metric).update(
            #                 recmotion_embeddings=rs_set["lat_rm"],
            #                 gtmotion_embeddings=rs_set["lat_m"],
            #                 lengths=lengths,
            #             )
            #         elif metric == "MRMetrics":
            #             getattr(self.metrics,
            #                     metric).update(rs_set["joints_rst"],
            #                                    rs_set["joints_ref"], lengths)
            #         elif metric == "PredMetrics":
            #             getattr(self.metrics,
            #                     metric).update(rs_set["joints_rst"],
            #                                    rs_set["joints_ref"], lengths)
            #         elif metric == "MMMetrics":
            #             # pass
            #             getattr(self.metrics,
            #                     metric).update(rs_set["m_rst"],
            #                                    rs_set['length'])
            #         else:
            #             raise TypeError(f"Not support this metric {metric}")

            # elif self.hparams.task == "m2t" and self.hparams.stage in [
            #         "lm_instruct", "lm_pretrain", "lm_rl"
            # ]:
            #     self.hparams.metrics_dict = metrics_dicts = ['M2TMetrics']
            #     for metric in metrics_dicts:
            #         if metric == "M2TMetrics":
                        # print(rs_set["t_pred"], batch["all_captions"])

        # return forward output rather than loss during test
        if split in ["test"]:
            batch_text = batch["text"] if "text" in batch else [""] * len(name)
            if self.hparams.stage == "vae":
                # return rs_set["joints_rst"], rs_set["joints_ref"], rs_set["vertices_rst"], rs_set["vertices_ref"], rs_set["m_ref"], rs_set["m_rst"], batch["length"]
                return {'name': name, 'feats_ref': rs_set["m_ref"], 'feats_rst': rs_set['m_rst'], 'lengths': batch['length'], 'lengths_rst': batch['length'], 'text': batch_text}
            elif "lm" in self.hparams.stage:
                task_name = str(self.hparams.task).lower()
                if task_name == "m2t":
                    # `test_step` expects motion-like tensors for optional dumping.
                    # For m2t evaluation, generated text is already consumed by metrics,
                    # so we return references as placeholders to keep the interface stable.
                    return {
                        'name': name,
                        'feats_ref': rs_set_m2t["m_ref"],
                        'feats_rst': rs_set_m2t["m_ref"],
                        'lengths': batch['length'],
                        'lengths_rst': batch['length'],
                        'text': batch_text,
                    }
                if task_name == "mc":
                    return {
                        'name': name,
                        'feats_ref': rs_set_mc["m_ref"],
                        'feats_rst': rs_set_mc['m_rst'],
                        'lengths': rs_set_mc['length'],
                        'lengths_rst': rs_set_mc['lengths_rst'],
                        'text': batch_text,
                    }
                return {
                    'name': name,
                    'feats_ref': rs_set["m_ref"],
                    'feats_rst': rs_set['m_rst'],
                    'lengths': batch['length'],
                    'lengths_rst': rs_set['lengths_rst'],
                    'text': batch_text,
                }
               
        return loss

    # def on_validation_epoch_end(self):
    #     # Log steps and losses
    #     dico = self.step_log_dict()
    #     # Log losses
    #     dico.update(self.loss_log_dict('train'))
    #     dico.update(self.loss_log_dict('val'))
    #     # Log metrics
    #     dico.update(self.metrics_log_dict())
    #     # print('dico', dico)
    #     # Write to log only if not sanity check
    #     if not self.trainer.sanity_checking:
    #         self.log_dict(dico, sync_dist=True, rank_zero_only=True)
    #         # print('dico', dico)
    #     # dist.barrier()
        
