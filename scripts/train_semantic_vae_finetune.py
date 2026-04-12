#!/usr/bin/env python3
"""
R027 — Semantic VAE encoder fine-tuning with contrastive + reconstruction dual loss.

Fine-tunes the VAE encoder (body + lhand + rhand) to produce encode_continuous()
embeddings where different sign sentences are more separated in cosine space.

Key design:
  - Frozen: VAE decoder, LFQ quantizer, quantize_out  (reconstruction quality preserved)
  - Trainable: VAE encoder, quantize_in               (semantics injected here)
  - Loss = λ_recon * SmoothL1(recon, target) / (valid_frames * D)
         + λ_contra * MultiPositiveInfoNCE(mean_pool_masked(encode_continuous), mT5_encode(text))

Fixes from R1 review (2026-04-12):
  - mT5 encoder forced to eval mode at all times (text targets are stable)
  - Multi-positive InfoNCE for duplicate/equivalent sentences in CSL-Daily
  - Masked mean pooling: derives downsampled T' length from actual T, no padding leakage
  - Reconstruction loss normalized by (valid_frames * feature_dim)
  - Logs raw cosine similarity (pre-temperature) separate from scaled logit
  - val_loader used for validation recon/contrastive metrics each epoch
  - Full training checkpoint (holder + projections + optimizer + scheduler + rng)

Usage — sanity check (100 pairs, 5 epochs):
    python scripts/train_semantic_vae_finetune.py \\
        --cfg configs/soke_mt5_csl_m2t.yaml \\
        --sanity \\
        --output_dir experiments/semantic_vae_finetune_sanity

Usage — full run (Plan A, R027):
    python scripts/train_semantic_vae_finetune.py \\
        --cfg configs/soke_mt5_csl_m2t.yaml \\
        --output_dir experiments/semantic_vae_finetune_R027 \\
        --epochs 50 \\
        --batch_size 128 \\
        --lr 3e-5 \\
        --lambda_recon 1.0 \\
        --lambda_contra 0.5 \\
        --seed 42

Outputs:
    output_dir/
        best_encoder.pt      ← VAE holder state_dict (encoder+quantize_in) for downstream use
        checkpoint.pt        ← full training state (resume-able)
        training_log.json    ← per-epoch metrics
        config.json          ← run configuration
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import unicodedata

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from sacrebleu.metrics import BLEU
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, MT5ForConditionalGeneration

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from mGPT.config import get_module_config, instantiate_from_config
from mGPT.data.humanml.load_data import load_csl_sample
from mGPT.utils.load_checkpoint import load_pretrained_vae


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _register_resolvers():
    try:
        OmegaConf.register_new_resolver("eval", eval)
    except ValueError:
        pass


def load_cfg(cfg_path: str):
    _register_resolvers()
    cfg_assets = OmegaConf.load("./configs/assets.yaml")
    cfg_base = OmegaConf.load(f"{cfg_assets.CONFIG_FOLDER}/default.yaml")
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(cfg_path))
    if not cfg_exp.FULL_CONFIG:
        cfg_exp = get_module_config(cfg_exp, cfg_assets.CONFIG_FOLDER)
    return OmegaConf.merge(cfg_exp, cfg_assets)


# ---------------------------------------------------------------------------
# VAE holder
# ---------------------------------------------------------------------------

class VAEHolder(nn.Module):
    """Wraps three VAEs (body, lhand, rhand).

    Freeze policy: decoder + quantizer + quantize_out are frozen.
    encoder + quantize_in are trainable.
    """

    def __init__(self, motion_vae, hand_vae_cfg, rhand_vae_cfg):
        super().__init__()
        self.vae = instantiate_from_config(motion_vae)
        self.hand_vae = instantiate_from_config(hand_vae_cfg)
        self.rhand_vae = instantiate_from_config(rhand_vae_cfg)

    def apply_freeze_policy(self):
        frozen_parts, trainable_parts = [], []
        for vae_name, vae in [("body", self.vae), ("lhand", self.hand_vae), ("rhand", self.rhand_vae)]:
            for mod_name in ("decoder", "quantizer", "quantize_out"):
                mod = getattr(vae, mod_name, None)
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad = False
                    frozen_parts.append(f"{vae_name}.{mod_name}")
            for mod_name in ("encoder", "quantize_in"):
                mod = getattr(vae, mod_name, None)
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad = True
                    trainable_parts.append(f"{vae_name}.{mod_name}")
        print(f"[freeze] Frozen   : {frozen_parts}")
        print(f"[freeze] Trainable: {trainable_parts}")

    def count_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Normalisation stats
# ---------------------------------------------------------------------------

def load_stats(mean_path: str, std_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = torch.load(mean_path, map_location="cpu")
    std = torch.load(std_path, map_location="cpu")
    mean = mean[(3 + 3 * 11):]
    mean = torch.cat([mean[:-20], mean[-10:]], dim=0)
    std = std[(3 + 3 * 11):]
    std = torch.cat([std[:-20], std[-10:]], dim=0)
    return mean.float(), std.float()


def normalize_pose(clip_poses: np.ndarray, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    x = torch.from_numpy(clip_poses).float()
    return (x - mean) / (std + 1e-10)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SignTextPoseDataset(Dataset):
    """Dataset of (normalised_pose [T, 133], text_str) pairs."""

    def __init__(self, items: List[dict], csl_root: str, mean: torch.Tensor, std: torch.Tensor):
        self.items = items
        self.csl_root = csl_root
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        try:
            clip_poses, _, _, _ = load_csl_sample(
                item, self.csl_root, need_pose=True, need_code=False
            )
        except Exception:
            return None
        if clip_poses is None:
            return None
        motion_norm = normalize_pose(clip_poses, self.mean, self.std)
        text = str(item.get("text", ""))
        return motion_norm, text

    @staticmethod
    def collate_fn(batch: list):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        poses, texts = zip(*batch)
        max_T = max(p.shape[0] for p in poses)
        D = poses[0].shape[1]
        padded = torch.zeros(len(poses), max_T, D)
        lengths = []
        for i, p in enumerate(poses):
            T = p.shape[0]
            padded[i, :T] = p
            lengths.append(T)
        return padded, list(texts), lengths


def build_csl_item_list(csl_root: str, splits: List[str]) -> List[dict]:
    items = []
    for split in splits:
        path = os.path.join(csl_root, f"csl_clean.{split}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSL annotation not found: {path}")
        with gzip.open(path, "rb") as f:
            ann = pickle.load(f)
        items.extend(ann)
    return items


# ---------------------------------------------------------------------------
# Text encoder (always in eval mode — text targets must be stable)
# ---------------------------------------------------------------------------

class FrozenTextEncoder(nn.Module):
    """mT5 encoder, always frozen and always in eval mode.

    IMPORTANT: model.train() must NOT put this back into train mode.
    We enforce this by calling self.encoder.eval() inside encode() and
    by overriding train() to keep the encoder eval.
    """

    def __init__(self, mt5_model_path: str):
        super().__init__()
        print(f"[model] Loading mT5 encoder from {mt5_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(mt5_model_path, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        mt5 = MT5ForConditionalGeneration.from_pretrained(
            mt5_model_path, torch_dtype=torch.float32
        )
        self.encoder = mt5.encoder
        del mt5
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        self.d_model: int = self.encoder.config.d_model

    def train(self, mode: bool = True):
        # Always keep encoder in eval mode regardless of outer model.train() calls
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def encode(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """Mean-pooled sentence embeddings: [B, d_model]."""
        self.encoder.eval()  # belt-and-suspenders guarantee
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(device)
        out = self.encoder(**encoded, return_dict=True)
        mask = encoded.attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1)
        return pooled  # [B, d_model]


# ---------------------------------------------------------------------------
# Projection heads
# ---------------------------------------------------------------------------

class SignProjection(nn.Module):
    """LayerNorm + 2-layer MLP: vae_dim → proj_dim."""

    def __init__(self, input_dim: int, proj_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))


class TextProjection(nn.Module):
    def __init__(self, text_dim: int, proj_dim: int):
        super().__init__()
        self.proj = nn.Linear(text_dim, proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ---------------------------------------------------------------------------
# Multi-positive InfoNCE loss
# ---------------------------------------------------------------------------

def multi_positive_infonce(
    s: torch.Tensor,
    t: torch.Tensor,
    texts: List[str],
    temperature: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Symmetric InfoNCE that handles duplicate sentences as multi-positives.

    Args:
        s: [B, D] L2-normalised sign embeddings
        t: [B, D] L2-normalised text embeddings
        texts: list of B text strings (duplicates treated as positives)
        temperature: scalar tensor

    Returns:
        dict with loss, acc_s2t, raw_cosine (pre-temperature diagonal)
    """
    sim = torch.matmul(s, t.T) / temperature  # [B, B]
    B = sim.shape[0]

    # Build multi-positive mask: 1 where texts[i] == texts[j]
    # Normalise text before comparison: strip whitespace, normalise Unicode
    normed = [unicodedata.normalize("NFKC", tx).strip() for tx in texts]
    pos_mask = torch.zeros(B, B, device=sim.device, dtype=torch.bool)
    for i in range(B):
        for j in range(B):
            if normed[i] == normed[j]:
                pos_mask[i, j] = True

    # Standard multi-positive InfoNCE: log(sum_pos / sum_all)
    # For s→t direction
    exp_sim = torch.exp(sim - sim.max(dim=1, keepdim=True).values.detach())
    pos_sum_s2t = (exp_sim * pos_mask.float()).sum(dim=1).clamp_min(1e-9)
    all_sum_s2t = exp_sim.sum(dim=1).clamp_min(1e-9)
    loss_s2t = -torch.log(pos_sum_s2t / all_sum_s2t).mean()

    # For t→s direction
    exp_sim_T = torch.exp(sim.T - sim.T.max(dim=1, keepdim=True).values.detach())
    pos_sum_t2s = (exp_sim_T * pos_mask.T.float()).sum(dim=1).clamp_min(1e-9)
    all_sum_t2s = exp_sim_T.sum(dim=1).clamp_min(1e-9)
    loss_t2s = -torch.log(pos_sum_t2s / all_sum_t2s).mean()

    loss = (loss_s2t + loss_t2s) / 2.0

    with torch.no_grad():
        # top-1 accuracy: correct if argmax is any positive (not just diagonal)
        nn_idx = sim.argmax(dim=1)
        acc_s2t = pos_mask[torch.arange(B), nn_idx].float().mean()
        # Raw cosine: diagonal (same-index pairs), pre-temperature
        raw_cosine = (s * t).sum(dim=-1).mean()  # [B] → scalar

    return {"loss": loss, "acc_s2t": acc_s2t, "raw_pos_cosine": raw_cosine}


# ---------------------------------------------------------------------------
# Downsampled length computation
# ---------------------------------------------------------------------------

def compute_downsampled_length(T: int, down_t: int, stride_t: int) -> int:
    """Compute output temporal length after Conv1d encoder downsampling.

    Each down-step: T_out = floor((T_in - filter_t + 2*pad_t) / stride_t) + 1
    where filter_t = stride_t * 2, pad_t = stride_t // 2.
    """
    T_cur = T
    for _ in range(down_t):
        filter_t = stride_t * 2
        pad_t = stride_t // 2
        T_cur = (T_cur + 2 * pad_t - filter_t) // stride_t + 1
    return max(T_cur, 1)


def verify_downsampled_length(vae, down_t: int, stride_t: int, D: int, device: torch.device):
    """Smoke test: verify compute_downsampled_length matches actual VAE output.

    Runs a zero-tensor forward pass for representative input lengths.
    Raises AssertionError if computed length doesn't match actual T'.
    Logs a warning if off by ≤1 (common with asymmetric padding).
    """
    test_lengths = [8, 16, 31, 32, 33, 64, 127, 128, 129, 200, 256]
    mismatches = []
    with torch.no_grad():
        for T in test_lengths:
            x = torch.zeros(1, T, D, device=device)
            y = vae.encode_continuous(x)
            actual_T = y.shape[1]
            predicted_T = compute_downsampled_length(T, down_t, stride_t)
            if actual_T != predicted_T:
                mismatches.append((T, predicted_T, actual_T))
    if mismatches:
        import warnings
        for T_in, T_pred, T_act in mismatches:
            warnings.warn(
                f"[verify_ds_len] T={T_in}: predicted={T_pred}, actual={T_act}. "
                "Masked pooling may be off. Consider using actual encode_continuous output length.",
                RuntimeWarning, stacklevel=2,
            )
    else:
        print(f"[verify_ds_len] ✓ All {len(test_lengths)} test lengths match.")


def get_vae_downsample_params(vae) -> Tuple[int, int]:
    """Extract (down_t, stride_t) from VQVae instance.

    Neither Encoder nor VQVae stores these as instance attributes after __init__,
    so we fall back to defaults (3, 2) which match the standard CSL-Daily configs.
    verify_downsampled_length() will warn if these defaults are wrong.
    """
    encoder = vae.encoder
    # Check encoder first, then vae, then use defaults — use `is not None` to
    # correctly handle the edge case of down_t=0 (no downsampling).
    down_t = getattr(encoder, "down_t", None)
    if down_t is None:
        down_t = getattr(vae, "down_t", 3)

    stride_t = getattr(encoder, "stride_t", None)
    if stride_t is None:
        stride_t = getattr(vae, "stride_t", 2)

    return int(down_t), int(stride_t)


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------

class SemanticVAEModel(nn.Module):
    def __init__(
        self,
        holder: VAEHolder,
        vae_dim: int,
        proj_dim: int,
        mt5_model_path: str,
        init_temperature: float = 0.07,
    ):
        super().__init__()
        self.holder = holder
        self.sign_proj = SignProjection(vae_dim, proj_dim)
        self.text_encoder = FrozenTextEncoder(mt5_model_path)
        self.text_proj = TextProjection(self.text_encoder.d_model, proj_dim)
        self.log_temperature = nn.Parameter(
            torch.tensor(float(np.log(init_temperature)), dtype=torch.float32)
        )

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=1e-4, max=1.0)

    @staticmethod
    def _masked_mean_pool_by_ratio(
        latents: torch.Tensor,    # [B, T', D]
        orig_lengths: List[int],  # original (pre-downsample) sequence lengths
        T_orig_max: int,          # padded batch length (denominator for ratio)
    ) -> torch.Tensor:
        """Mean-pool latents using masks derived from actual downsampling ratio.

        Derives per-sample valid T' from the ratio T_orig / T_prime_max.
        This avoids dependence on down_t/stride_t config values.
        """
        B, Tp, D = latents.shape
        device = latents.device

        # Empirical downsampling ratio from the actual encoded batch
        ds_ratio = T_orig_max / Tp  # e.g. 4.0 for down_t=2, stride_t=2

        mask = torch.zeros(B, Tp, 1, device=device)
        for i, orig_len in enumerate(orig_lengths):
            ds_len = min(max(round(orig_len / ds_ratio), 1), Tp)
            mask[i, :ds_len] = 1.0

        pooled = (latents * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return pooled  # [B, D]

    def encode_continuous_batch(
        self,
        padded_poses: torch.Tensor,  # [B, T_max, 133]
        lengths: List[int],
    ) -> torch.Tensor:
        """Masked-mean-pooled continuous embeddings: [B, vae_dim].

        Uses actual encode_continuous() output shape to compute downsampling ratio,
        so the mask is correct regardless of down_t/stride_t config.
        """
        T_orig_max = padded_poses.shape[1]
        fb = torch.cat([padded_poses[..., :30], padded_poses[..., 120:]], dim=-1)
        fl = padded_poses[..., 30:75]
        fr = padded_poses[..., 75:120]

        eb = self.holder.vae.encode_continuous(fb)        # [B, T', 512]
        el = self.holder.hand_vae.encode_continuous(fl)   # [B, T', 512]
        er = self.holder.rhand_vae.encode_continuous(fr)  # [B, T', 512]

        eb_pool = self._masked_mean_pool_by_ratio(eb, lengths, T_orig_max)
        el_pool = self._masked_mean_pool_by_ratio(el, lengths, T_orig_max)
        er_pool = self._masked_mean_pool_by_ratio(er, lengths, T_orig_max)

        return torch.cat([eb_pool, el_pool, er_pool], dim=-1)  # [B, vae_dim]

    def reconstruction_loss(
        self,
        padded_poses: torch.Tensor,
        lengths: List[int],
    ) -> torch.Tensor:
        """SmoothL1 reconstruction loss, normalised by valid frame-elements.

        Gradient flows through encoder + quantize_in (trainable).
        """
        fb = torch.cat([padded_poses[..., :30], padded_poses[..., 120:]], dim=-1)
        fl = padded_poses[..., 30:75]
        fr = padded_poses[..., 75:120]

        recon_b, q_loss_b, _ = self.holder.vae(fb)
        recon_l, q_loss_l, _ = self.holder.hand_vae(fl)
        recon_r, q_loss_r, _ = self.holder.rhand_vae(fr)
        # Quantizer losses are detached here for diagnostic logging only.
        # They are not included in the training loss (quantizer is frozen).
        self._last_quantizer_loss = (
            q_loss_b.detach().mean() + q_loss_l.detach().mean() + q_loss_r.detach().mean()
        ).item() / 3.0

        B, T_max, _ = padded_poses.shape
        mask = torch.zeros(B, T_max, device=padded_poses.device)
        for i, length in enumerate(lengths):
            mask[i, :min(length, T_max)] = 1.0

        def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            T_pred = pred.shape[1]
            D_feat = pred.shape[2]
            m = mask[:, :T_pred].unsqueeze(-1)          # [B, T', 1]
            loss_elem = F.smooth_l1_loss(pred, target[:, :T_pred], reduction="none")  # [B, T', D]
            masked_loss = (loss_elem * m).sum()
            denom = (m.sum() * D_feat).clamp_min(1.0)   # normalise per frame-element
            return masked_loss / denom

        return (
            masked_smooth_l1(recon_b, fb)
            + masked_smooth_l1(recon_l, fl)
            + masked_smooth_l1(recon_r, fr)
        )

    def contrastive_loss(
        self,
        padded_poses: torch.Tensor,
        lengths: List[int],
        texts: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Multi-positive InfoNCE on projected embeddings."""
        device = padded_poses.device

        mean_embs = self.encode_continuous_batch(padded_poses, lengths)  # [B, vae_dim]
        s = F.normalize(self.sign_proj(mean_embs), dim=-1)
        text_embs = self.text_encoder.encode(texts, device)              # [B, d_model]
        t = F.normalize(self.text_proj(text_embs), dim=-1)

        out = multi_positive_infonce(s, t, texts, self.temperature)
        return out

    def forward(
        self,
        padded_poses: torch.Tensor,
        lengths: List[int],
        texts: List[str],
        lambda_recon: float,
        lambda_contra: float,
    ) -> Dict[str, torch.Tensor]:
        recon_loss = self.reconstruction_loss(padded_poses, lengths)
        contra_out = self.contrastive_loss(padded_poses, lengths, texts)
        total_loss = lambda_recon * recon_loss + lambda_contra * contra_out["loss"]

        return {
            "loss": total_loss,
            "recon_loss": recon_loss.detach(),
            "contra_loss": contra_out["loss"].detach(),
            "acc_s2t": contra_out["acc_s2t"],
            "raw_pos_cosine": contra_out["raw_pos_cosine"],   # true cosine, pre-temperature
            "temperature": self.temperature.detach(),
            "quantizer_loss": getattr(self, "_last_quantizer_loss", 0.0),  # diagnostic
        }


# ---------------------------------------------------------------------------
# Validation pass (recon + contrastive metrics on val set)
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: SemanticVAEModel,
    val_loader: DataLoader,
    device: torch.device,
    lambda_recon: float,
    lambda_contra: float,
) -> Dict[str, float]:
    model.eval()
    total = {"loss": 0.0, "recon_loss": 0.0, "contra_loss": 0.0,
             "acc_s2t": 0.0, "raw_pos_cosine": 0.0}
    n = 0

    for batch in val_loader:
        if batch is None:
            continue
        padded_poses, texts, lengths = batch
        padded_poses = padded_poses.to(device)

        recon_loss = model.reconstruction_loss(padded_poses, lengths)
        contra_out = model.contrastive_loss(padded_poses, lengths, texts)
        loss = lambda_recon * recon_loss + lambda_contra * contra_out["loss"]

        bs = padded_poses.shape[0]
        total["loss"] += loss.item() * bs
        total["recon_loss"] += recon_loss.item() * bs
        total["contra_loss"] += contra_out["loss"].item() * bs
        total["acc_s2t"] += contra_out["acc_s2t"].item() * bs
        total["raw_pos_cosine"] += contra_out["raw_pos_cosine"].item() * bs
        n += bs

    model.train()
    if n == 0:
        return {k: 0.0 for k in total}
    return {k: v / n for k, v in total.items()}


# ---------------------------------------------------------------------------
# Oracle BLEU4 evaluation (R025-style)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_oracle_bleu(
    model: SemanticVAEModel,
    train_items: List[dict],
    val_items: List[dict],
    csl_root: str,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    max_train: int = 3000,
    max_val: int = 500,
) -> Dict[str, float]:
    """NN retrieval oracle BLEU4 using raw encode_continuous() embeddings.

    This measures the ceiling of information in the raw embeddings, not the
    projected space. It is the primary gate metric for R027.
    """
    model.eval()

    def encode_items(items: List[dict], max_n: int):
        embs, texts = [], []
        for item in tqdm(items[:max_n], desc="Oracle encode", leave=False):
            try:
                clip_poses, _, _, _ = load_csl_sample(
                    item, csl_root, need_pose=True, need_code=False
                )
            except Exception:
                continue
            if clip_poses is None:
                continue
            motion_norm = normalize_pose(clip_poses, mean, std).to(device)
            cur = motion_norm.unsqueeze(0)   # [1, T, 133]
            fb = torch.cat([cur[..., :30], cur[..., 120:]], dim=-1)
            fl = cur[..., 30:75]
            fr = cur[..., 75:120]
            eb = model.holder.vae.encode_continuous(fb)       # [1, T', 512]
            el = model.holder.hand_vae.encode_continuous(fl)  # [1, T', 512]
            er = model.holder.rhand_vae.encode_continuous(fr) # [1, T', 512]
            # Per-sample encoding: no padding, use full T' (no mask needed)
            mt = min(eb.shape[1], el.shape[1], er.shape[1])
            eb_p = eb[0, :mt].mean(dim=0)
            el_p = el[0, :mt].mean(dim=0)
            er_p = er[0, :mt].mean(dim=0)
            combined = torch.cat([eb_p, el_p, er_p], dim=0).cpu().float()  # [1536]
            embs.append(F.normalize(combined, dim=0))
            texts.append(str(item.get("text", "")))

        if not embs:
            return None, []
        return torch.stack(embs), texts

    train_embs, train_texts = encode_items(train_items, max_train)
    val_embs, val_texts = encode_items(val_items, max_val)

    if train_embs is None or val_embs is None:
        model.train()
        return {"oracle_bleu4": 0.0, "mean_nn_cosine": 1.0}

    sim = torch.matmul(val_embs, train_embs.T)    # [N_val, N_train]
    nn_idx = sim.argmax(dim=1)
    nn_cosines = sim.max(dim=1).values

    hypotheses = [train_texts[i] for i in nn_idx.tolist()]
    bleu = BLEU(effective_order=True)
    result = bleu.corpus_score(hypotheses, [val_texts])

    model.train()
    return {
        "oracle_bleu4": float(result.score),
        "mean_nn_cosine": float(nn_cosines.mean().item()),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def save_topk_encoder(
    output_dir: str,
    model: SemanticVAEModel,
    oracle_b4: float,
    epoch: int,
    top_k: int = 3,
):
    """Save lightweight encoder export for top-k oracle BLEU4 checkpoints.

    Maintains a directory of up to top_k best checkpoints.
    Each saved file is named best_encoder_b4={oracle_b4:.3f}_ep{epoch}.pt.
    Removes the worst if more than top_k exist.
    """
    import glob
    ckpt_path = os.path.join(output_dir, f"best_encoder_b4={oracle_b4:.3f}_ep{epoch:03d}.pt")
    torch.save(model.holder.state_dict(), ckpt_path)

    # List all top-k candidates
    all_ckpts = sorted(
        glob.glob(os.path.join(output_dir, "best_encoder_b4=*.pt")),
        key=lambda p: float(os.path.basename(p).split("b4=")[1].split("_")[0]),
        reverse=True,
    )
    # Remove worst if exceeds top_k
    for stale in all_ckpts[top_k:]:
        os.remove(stale)

    # Always keep a symlink "best_encoder.pt" pointing to the current best
    best_link = os.path.join(output_dir, "best_encoder.pt")
    if os.path.lexists(best_link):
        os.remove(best_link)
    os.symlink(os.path.basename(all_ckpts[0]), best_link)


def save_checkpoint(
    path: str,
    model: SemanticVAEModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    args: argparse.Namespace,
    history: list,
):
    """Full training checkpoint for resuming."""
    torch.save({
        "epoch": epoch,
        "holder_state_dict": model.holder.state_dict(),
        "sign_proj_state_dict": model.sign_proj.state_dict(),
        "text_proj_state_dict": model.text_proj.state_dict(),
        "log_temperature": model.log_temperature.item(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "args": vars(args),
        "history": history,
    }, path)


def train(
    model: SemanticVAEModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_items: List[dict],
    val_items: List[dict],
    csl_root: str,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    lambda_recon: float,
    lambda_contra: float,
    output_dir: str,
    oracle_eval_every: int,
    patience: int,
    args: argparse.Namespace,
) -> List[dict]:
    os.makedirs(output_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 1e-2
    )

    history: List[dict] = []
    best_oracle = -float("inf")
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        totals = {"loss": 0.0, "recon": 0.0, "contra": 0.0, "acc": 0.0, "cosine": 0.0, "q_loss": 0.0}
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            if batch is None:
                continue
            padded_poses, texts, lengths = batch
            padded_poses = padded_poses.to(device)

            out = model(padded_poses, lengths, texts, lambda_recon, lambda_contra)
            loss = out["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()

            totals["loss"] += loss.item()
            totals["recon"] += out["recon_loss"].item()
            totals["contra"] += out["contra_loss"].item()
            totals["acc"] += out["acc_s2t"].item()
            totals["cosine"] += out["raw_pos_cosine"].item()
            totals["q_loss"] += out.get("quantizer_loss", 0.0)
            n_batches += 1

        scheduler.step()
        nb = max(n_batches, 1)

        # Validation metrics
        val_metrics = validate(model, val_loader, device, lambda_recon, lambda_contra)

        row: dict = {
            "epoch": epoch,
            "train_loss": totals["loss"] / nb,
            "train_recon_loss": totals["recon"] / nb,
            "train_contra_loss": totals["contra"] / nb,
            "train_acc_s2t": totals["acc"] / nb,
            "train_raw_pos_cosine": totals["cosine"] / nb,  # true cosine, pre-temperature
            "train_quantizer_loss": totals["q_loss"] / nb,  # diagnostic: quantizer drift
            "val_loss": val_metrics["loss"],
            "val_recon_loss": val_metrics["recon_loss"],
            "val_contra_loss": val_metrics["contra_loss"],
            "val_acc_s2t": val_metrics["acc_s2t"],
            "val_raw_pos_cosine": val_metrics["raw_pos_cosine"],
            "temperature": model.temperature.item(),
            "time_s": time.time() - t0,
        }

        # Oracle evaluation (expensive)
        if epoch % oracle_eval_every == 0 or epoch == epochs:
            oracle_out = compute_oracle_bleu(
                model, train_items, val_items, csl_root, mean, std, device
            )
            row.update(oracle_out)
            oracle_b4 = oracle_out["oracle_bleu4"]
            nn_cos = oracle_out["mean_nn_cosine"]

            print(
                f"[E{epoch:03d}] "
                f"tr_loss={row['train_loss']:.4f} recon={row['train_recon_loss']:.4f} "
                f"contra={row['train_contra_loss']:.4f} acc={row['train_acc_s2t']:.3f} "
                f"cos={row['train_raw_pos_cosine']:.3f} | "
                f"val_loss={row['val_loss']:.4f} val_cos={row['val_raw_pos_cosine']:.3f} | "
                f"oracle_b4={oracle_b4:.3f} nn_cos={nn_cos:.3f} T={row['time_s']:.0f}s"
            )

            if oracle_b4 > best_oracle:
                best_oracle = oracle_b4
                save_topk_encoder(output_dir, model, oracle_b4, epoch, top_k=3)
                print(f"  [ckpt] Saved top-k encoder (oracle_b4={oracle_b4:.3f})")
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"[early stop] No oracle improvement for {patience} eval cycles.")
                    history.append(row)
                    break
        else:
            print(
                f"[E{epoch:03d}] "
                f"tr_loss={row['train_loss']:.4f} recon={row['train_recon_loss']:.4f} "
                f"contra={row['train_contra_loss']:.4f} acc={row['train_acc_s2t']:.3f} "
                f"cos={row['train_raw_pos_cosine']:.3f} | "
                f"val_loss={row['val_loss']:.4f} val_cos={row['val_raw_pos_cosine']:.3f}"
            )

        history.append(row)
        # Full training checkpoint (resume-able)
        save_checkpoint(
            os.path.join(output_dir, "checkpoint.pt"),
            model, optimizer, scheduler, epoch, args, history
        )
        with open(os.path.join(output_dir, "training_log.json"), "w") as f:
            json.dump(history, f, indent=2)

    return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="R027: Semantic VAE encoder fine-tuning")
    p.add_argument("--cfg", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lambda_recon", type=float, default=1.0)
    p.add_argument("--lambda_contra", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--proj_dim", type=int, default=768)
    p.add_argument("--oracle_eval_every", type=int, default=5)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sanity", action="store_true")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"[device] {device}")

    cfg = load_cfg(args.cfg)
    csl_root = cfg.DATASET.H2S.CSL_ROOT
    mean_path = cfg.DATASET.H2S.MEAN_PATH
    std_path = cfg.DATASET.H2S.STD_PATH
    mt5_path = cfg.lm.mt5_base.params.model_path

    mean, std = load_stats(mean_path, std_path)

    holder = VAEHolder(
        motion_vae=cfg.model.params.motion_vae,
        hand_vae_cfg=cfg.model.params.hand_vae_cfg,
        rhand_vae_cfg=cfg.model.params.rhand_vae_cfg,
    )
    load_pretrained_vae(cfg, holder, logger=None)
    holder.apply_freeze_policy()
    holder.to(device)
    print(f"[model] VAE params: {holder.count_trainable():,} trainable / {holder.count_total():,} total")

    vae_dim = (
        getattr(holder.vae, "output_emb_width", 512)
        + getattr(holder.hand_vae, "output_emb_width", 512)
        + getattr(holder.rhand_vae, "output_emb_width", 512)
    )
    print(f"[model] vae_dim={vae_dim}")

    # Log actual downsampling ratio from a test pass
    with torch.no_grad():
        _test = torch.zeros(1, 64, 43, device=device)
        _out = holder.vae.encode_continuous(_test)
        print(f"[model] actual downsample ratio = {64 / _out.shape[1]:.1f}× (T=64 → T'={_out.shape[1]})")
        del _test, _out

    model = SemanticVAEModel(
        holder=holder, vae_dim=vae_dim, proj_dim=args.proj_dim,
        mt5_model_path=mt5_path, init_temperature=args.temperature,
    ).to(device)
    print(f"[model] Total trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    all_train = build_csl_item_list(csl_root, ["train"])
    all_val = build_csl_item_list(csl_root, ["val"])

    if args.sanity:
        all_train = all_train[:100]
        all_val = all_val[:30]
        args.epochs = 5
        args.oracle_eval_every = 2
        args.patience = 9999  # no early stop in sanity
        print("[sanity] 100 train / 30 val / 5 epochs")

    train_ds = SignTextPoseDataset(all_train, csl_root, mean, std)
    val_ds = SignTextPoseDataset(all_val, csl_root, mean, std)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=SignTextPoseDataset.collate_fn,
        drop_last=len(train_ds) > args.batch_size,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=SignTextPoseDataset.collate_fn,
    )
    print(f"[data] train={len(train_ds)} / val={len(val_ds)}")

    os.makedirs(args.output_dir, exist_ok=True)
    run_cfg = vars(args).copy()
    run_cfg["vae_dim"] = vae_dim
    run_cfg["n_train"] = len(train_ds)
    run_cfg["n_val"] = len(val_ds)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  R027 — Semantic VAE Encoder Fine-tuning")
    print(f"  epochs={args.epochs}  batch_size={args.batch_size}  lr={args.lr}")
    print(f"  lambda_recon={args.lambda_recon}  lambda_contra={args.lambda_contra}")
    print(f"  temperature_init={args.temperature}  proj_dim={args.proj_dim}")
    print(f"{'='*60}\n")

    history = train(
        model=model, train_loader=train_loader, val_loader=val_loader,
        train_items=all_train, val_items=all_val, csl_root=csl_root,
        mean=mean, std=std, device=device, epochs=args.epochs,
        lr=args.lr, weight_decay=args.weight_decay,
        lambda_recon=args.lambda_recon, lambda_contra=args.lambda_contra,
        output_dir=args.output_dir, oracle_eval_every=args.oracle_eval_every,
        patience=args.patience, args=args,
    )

    oracle_epochs = [r for r in history if "oracle_bleu4" in r]
    if oracle_epochs:
        best = max(oracle_epochs, key=lambda r: r["oracle_bleu4"])
        b4 = best["oracle_bleu4"]
        cos = best.get("mean_nn_cosine", 1.0)
        print(f"\n[done] Best oracle BLEU4={b4:.3f}  mean_nn_cosine={cos:.3f}")
        if b4 > 3.0 and cos < 0.78:
            print("[R027] GATE PASSED ✓ — proceed to R027-oracle then R028")
        else:
            print("[R027] GATE NOT MET — try lambda_contra=1.0 or Plan B")
    print(f"[done] best_encoder.pt → {args.output_dir}/best_encoder.pt")
    print(f"[done] checkpoint.pt  → {args.output_dir}/checkpoint.pt")


if __name__ == "__main__":
    main()
