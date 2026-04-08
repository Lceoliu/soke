#!/usr/bin/env python3
"""
Contrastive pre-training of the sign projection module.

Trains a sign embedding projection (VAE continuous embeddings → d_model-dim)
to be aligned with frozen mT5 text encoder representations, using InfoNCE loss.
After training, the projection checkpoint can be loaded into MT5Seq2SeqLM to
initialise sign_proj before LM fine-tuning (see mgpt_mt5.py --contrastive_ckpt).

Usage — sanity check (100 pairs, 5 epochs, CPU/single GPU):
    python scripts/train_contrastive_pretrain.py \
        --cfg configs/soke_mt5_csl_m2t.yaml \
        --sanity \
        --output_dir experiments/contrastive_pretrain_sanity

Usage — full run:
    python scripts/train_contrastive_pretrain.py \
        --cfg configs/soke_mt5_csl_m2t.yaml \
        --output_dir experiments/contrastive_pretrain_csl \
        --epochs 80 \
        --batch_size 256 \
        --lr 1e-4 \
        --temperature 0.07 \
        --proj_dim 768 \
        --seed 42

Results saved to:
    output_dir/
        best_sign_proj.pt          ← sign projection weights (loads into MT5 sign_proj)
        last_sign_proj.pt          ← last-epoch checkpoint
        training_log.json          ← per-epoch metrics
        config.json                ← run configuration
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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
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
# Config / resolver setup
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
# VAE holder (re-uses pattern from train_csl_vae_action_classifier.py)
# ---------------------------------------------------------------------------

class VAEHolder(nn.Module):
    def __init__(self, motion_vae, hand_vae_cfg, rhand_vae_cfg):
        super().__init__()
        self.vae = instantiate_from_config(motion_vae)
        self.hand_vae = instantiate_from_config(hand_vae_cfg)
        self.rhand_vae = instantiate_from_config(rhand_vae_cfg)


@torch.no_grad()
def encode_sample(holder: VAEHolder, motion_norm: torch.Tensor) -> torch.Tensor:
    """Encode a single normalised motion sample through all three frozen VAEs.

    Returns combined continuous embedding: [T', body_dim + lhand_dim + rhand_dim]
    """
    cur = motion_norm.unsqueeze(0)  # [1, T, 133]
    fb = torch.cat([cur[..., :30], cur[..., 120:]], dim=-1)
    fl = cur[..., 30:75]
    fr = cur[..., 75:120]
    eb = holder.vae.encode_continuous(fb)       # [1, T', 512]
    el = holder.hand_vae.encode_continuous(fl)  # [1, T', 512]
    er = holder.rhand_vae.encode_continuous(fr) # [1, T', 512]
    mt = min(eb.shape[1], el.shape[1], er.shape[1])
    combined = torch.cat([eb[:, :mt], el[:, :mt], er[:, :mt]], dim=-1)  # [1, T', 1536]
    return combined[0].cpu().float()  # [T', 1536]


# ---------------------------------------------------------------------------
# Normalisation stats
# ---------------------------------------------------------------------------

def load_stats(mean_path: str, std_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = torch.load(mean_path, map_location="cpu")
    std = torch.load(std_path, map_location="cpu")
    # Apply same trimming as train_csl_vae_action_classifier.py
    mean = mean[(3 + 3 * 11):]
    mean = torch.cat([mean[:-20], mean[-10:]], dim=0)
    std = std[(3 + 3 * 11):]
    std = torch.cat([std[:-20], std[-10:]], dim=0)
    return mean.float(), std.float()


def normalize_pose(clip_poses: np.ndarray, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    x = torch.from_numpy(clip_poses).float()
    return (x - mean) / (std + 1e-10)


# ---------------------------------------------------------------------------
# Data loading (CSL-Daily annotation + pose files)
# ---------------------------------------------------------------------------

def load_csl_annotations(csl_root: str) -> Dict[str, dict]:
    ann_by_name = {}
    for split in ["train", "val", "test"]:
        path = os.path.join(csl_root, f"csl_clean.{split}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSL annotation not found: {path}")
        with gzip.open(path, "rb") as f:
            ann = pickle.load(f)
        for item in ann:
            ann_by_name[str(item["name"])] = item
    return ann_by_name


def build_sign_text_pairs(
    ann_by_name: Dict[str, dict],
    csl_root: str,
    holder: VAEHolder,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    splits: List[str],
    max_samples: Optional[int] = None,
    cache_path: Optional[str] = None,
) -> List[Tuple[torch.Tensor, str]]:
    """Build list of (mean_pooled_vae_embedding [1536], text) pairs.

    Loads or caches pre-computed VAE embeddings to avoid repeated encoding.
    """
    if cache_path is not None and os.path.exists(cache_path):
        print(f"[cache] Loading pre-computed pairs from {cache_path}")
        data = torch.load(cache_path, map_location="cpu")
        return [(emb, txt) for emb, txt in zip(data["embeddings"], data["texts"])]

    # Collect all names for specified splits
    all_items = []
    for split in splits:
        path = os.path.join(csl_root, f"csl_clean.{split}")
        if not os.path.exists(path):
            continue
        with gzip.open(path, "rb") as f:
            ann = pickle.load(f)
        for item in ann:
            all_items.append(item)

    if max_samples is not None:
        all_items = all_items[:max_samples]

    pairs = []
    skipped = 0
    holder.eval()

    for item in tqdm(all_items, desc="Encoding VAE embeddings"):
        try:
            clip_poses, _, name, _ = load_csl_sample(
                item, csl_root, need_pose=True, need_code=False
            )
        except Exception as exc:
            skipped += 1
            continue
        if clip_poses is None:
            skipped += 1
            continue

        motion_norm = normalize_pose(clip_poses, mean, std).to(device)
        with torch.no_grad():
            emb = encode_sample(holder, motion_norm)  # [T', 1536]
        # Mean-pool over temporal dimension → sentence-level sign embedding
        mean_emb = emb.mean(dim=0)  # [1536]
        text = str(item.get("text", ""))
        pairs.append((mean_emb, text))

    print(f"[data] Built {len(pairs)} pairs ({skipped} skipped)")

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        embeddings = torch.stack([p[0] for p in pairs])
        texts = [p[1] for p in pairs]
        torch.save({"embeddings": embeddings, "texts": texts}, cache_path)
        print(f"[cache] Saved to {cache_path}")

    return pairs


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SignTextPairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[torch.Tensor, str]]):
        self.embeddings = [p[0] for p in pairs]  # list of [1536] tensors
        self.texts = [p[1] for p in pairs]

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.texts[idx]

    @staticmethod
    def collate_fn(batch):
        embs = torch.stack([b[0] for b in batch])  # [B, 1536]
        texts = [b[1] for b in batch]
        return embs, texts


# ---------------------------------------------------------------------------
# Model: Sign projection + Text projection
# ---------------------------------------------------------------------------

class SignProjection(nn.Module):
    """LayerNorm + 2-layer MLP: vae_dim → proj_dim.

    Designed to match SignEmbeddingProjection in mgpt_mt5.py so weights can
    be transferred. When proj_dim == mt5_d_model, loaded directly into sign_proj.
    """
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
    """Project mT5 sentence embedding → proj_dim (for InfoNCE alignment)."""
    def __init__(self, text_dim: int, proj_dim: int):
        super().__init__()
        self.proj = nn.Linear(text_dim, proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class SignTextContrastiveModel(nn.Module):
    def __init__(
        self,
        vae_dim: int,
        proj_dim: int,
        mt5_model_path: str,
        init_temperature: float = 0.07,
    ):
        super().__init__()
        # Sign encoder side
        self.sign_proj = SignProjection(vae_dim, proj_dim)

        # Text encoder side (frozen mT5 encoder)
        print(f"[model] Loading mT5 encoder from {mt5_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(mt5_model_path, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        mt5 = MT5ForConditionalGeneration.from_pretrained(
            mt5_model_path, torch_dtype=torch.float32
        )
        self.text_encoder = mt5.encoder
        del mt5
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        self.text_proj = TextProjection(self.text_encoder.config.d_model, proj_dim)

        # Learnable log-temperature for InfoNCE
        self.log_temperature = nn.Parameter(
            torch.tensor(float(np.log(init_temperature)), dtype=torch.float32)
        )

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=1e-4, max=1.0)

    def encode_sign(self, sign_embs: torch.Tensor) -> torch.Tensor:
        """sign_embs: [B, vae_dim] → [B, proj_dim], L2-normalised"""
        z = self.sign_proj(sign_embs)
        return F.normalize(z, dim=-1)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """texts: List[str] → [B, proj_dim], L2-normalised"""
        device = next(self.text_encoder.parameters()).device
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = self.text_encoder(**encoded, return_dict=True)
        # Mean-pool over non-padding tokens
        mask = encoded.attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1)
        z = self.text_proj(pooled)
        return F.normalize(z, dim=-1)

    def forward(
        self,
        sign_embs: torch.Tensor,
        texts: List[str],
    ) -> Dict[str, torch.Tensor]:
        s = self.encode_sign(sign_embs)  # [B, D]
        t = self.encode_text(texts)      # [B, D]

        # InfoNCE: symmetric cross-entropy on [B×B] similarity matrix
        sim = torch.matmul(s, t.T) / self.temperature  # [B, B]
        labels = torch.arange(sim.shape[0], device=sim.device)
        loss_s2t = F.cross_entropy(sim, labels)
        loss_t2s = F.cross_entropy(sim.T, labels)
        loss = (loss_s2t + loss_t2s) / 2.0

        with torch.no_grad():
            diag_sim = sim.diagonal()
            avg_pos_sim = diag_sim.mean()
            # top-1 accuracy (sign→text)
            acc_s2t = (sim.argmax(dim=1) == labels).float().mean()

        return {
            "loss": loss,
            "loss_s2t": loss_s2t.detach(),
            "loss_t2s": loss_t2s.detach(),
            "avg_pos_sim": avg_pos_sim,
            "acc_s2t": acc_s2t,
            "temperature": self.temperature.detach(),
        }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def evaluate(
    model: SignTextContrastiveModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_pos_sim = 0.0
    n = 0
    with torch.no_grad():
        for sign_embs, texts in loader:
            sign_embs = sign_embs.to(device)
            out = model(sign_embs, texts)
            bs = sign_embs.shape[0]
            total_loss += out["loss"].item() * bs
            total_acc += out["acc_s2t"].item() * bs
            total_pos_sim += out["avg_pos_sim"].item() * bs
            n += bs
    model.train()
    if n == 0:
        return {"loss": float("inf"), "acc_s2t": 0.0, "avg_pos_sim": 0.0}
    return {
        "loss": total_loss / n,
        "acc_s2t": total_acc / n,
        "avg_pos_sim": total_pos_sim / n,
    }


def train(
    model: SignTextContrastiveModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    output_dir: str,
    patience: int = 10,
) -> List[Dict]:
    os.makedirs(output_dir, exist_ok=True)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.99),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 1e-2
    )

    history = []
    best_val_loss = float("inf")
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0
        t0 = time.time()

        for sign_embs, texts in train_loader:
            sign_embs = sign_embs.to(device)
            out = model(sign_embs, texts)
            loss = out["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += out["acc_s2t"].item()
            n_batches += 1

        scheduler.step()

        train_metrics = {
            "train_loss": epoch_loss / max(n_batches, 1),
            "train_acc_s2t": epoch_acc / max(n_batches, 1),
        }
        val_metrics = {}
        if val_loader is not None:
            val_metrics = {f"val_{k}": v for k, v in evaluate(model, val_loader, device).items()}

        row = {
            "epoch": epoch,
            "elapsed": time.time() - t0,
            "lr": scheduler.get_last_lr()[0],
            "temperature": model.temperature.item(),
            **train_metrics,
            **val_metrics,
        }
        history.append(row)

        val_loss = val_metrics.get("val_loss", train_metrics["train_loss"])
        print(
            f"[epoch {epoch:03d}/{epochs}] "
            f"loss={row.get('train_loss', -1):.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"acc_s2t={row.get('train_acc_s2t', -1):.3f}  "
            f"T={row['temperature']:.4f}  "
            f"({row['elapsed']:.1f}s)"
        )

        # Save last
        torch.save(model.sign_proj.state_dict(), os.path.join(output_dir, "last_sign_proj.pt"))

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            torch.save(model.sign_proj.state_dict(), os.path.join(output_dir, "best_sign_proj.pt"))
            print(f"  → New best val_loss={best_val_loss:.4f} — saved best_sign_proj.pt")
        else:
            wait += 1
            if wait >= patience and epochs > 20:
                print(f"  → Early stopping (patience={patience})")
                break

    # Save full log
    with open(os.path.join(output_dir, "training_log.json"), "w") as f:
        json.dump(history, f, indent=2)

    return history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Contrastive sign-text pre-training")
    p.add_argument("--cfg", type=str, default="configs/soke_mt5_csl_m2t.yaml",
                   help="Config file (same as mT5 training)")
    p.add_argument("--output_dir", type=str, default="experiments/contrastive_pretrain_csl",
                   help="Directory to save checkpoints and logs")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.07,
                   help="Initial InfoNCE temperature (learnable)")
    p.add_argument("--proj_dim", type=int, default=768,
                   help="Projection dimension — set to mT5 d_model (768 for mt5-base)")
    p.add_argument("--patience", type=int, default=10,
                   help="Early stopping patience (epochs without val_loss improvement)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0, help="GPU index; -1 for CPU")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cache_dir", type=str, default=None,
                   help="Directory to cache pre-computed VAE embeddings (recommended)")
    p.add_argument("--sanity", action="store_true",
                   help="Sanity mode: 100 samples, 5 epochs, validate alignment improves")
    p.add_argument("--val_split_ratio", type=float, default=0.1,
                   help="Fraction of training pairs to hold out as validation")
    return p.parse_args()


def main():
    args = parse_args()

    # --- Seed ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- Device ---
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"[device] Using {device}")

    # --- Config ---
    cfg = load_cfg(args.cfg)
    csl_root = cfg.DATASET.H2S.CSL_ROOT
    mean_path = cfg.DATASET.H2S.MEAN_PATH
    std_path = cfg.DATASET.H2S.STD_PATH
    mt5_path = cfg.lm.mt5_base.params.model_path

    print(f"[config] CSL root: {csl_root}")
    print(f"[config] mT5 path: {mt5_path}")

    # --- Load stats ---
    mean, std = load_stats(mean_path, std_path)

    # --- Load VAEs ---
    print("[model] Loading frozen VAEs...")
    holder = VAEHolder(
        motion_vae=cfg.model.params.motion_vae,
        hand_vae_cfg=cfg.model.params.hand_vae_cfg,
        rhand_vae_cfg=cfg.model.params.rhand_vae_cfg,
    )
    load_pretrained_vae(cfg, holder, logger=None)
    holder.eval().to(device)
    for p in holder.parameters():
        p.requires_grad = False

    # Infer vae_dim from the actual VAE
    vae_dim = (
        getattr(holder.vae, "output_emb_width", 512)
        + getattr(holder.hand_vae, "output_emb_width", 512)
        + getattr(holder.rhand_vae, "output_emb_width", 512)
    )
    print(f"[model] VAE embedding dim: {vae_dim}")

    # --- Build sign-text pairs ---
    sanity_max = 100 if args.sanity else None
    cache_path = None
    if args.cache_dir is not None:
        tag = "sanity" if args.sanity else "full"
        cache_path = os.path.join(args.cache_dir, f"csl_sign_text_pairs_{tag}.pt")

    pairs = build_sign_text_pairs(
        ann_by_name={},  # unused (annotations loaded inside)
        csl_root=csl_root,
        holder=holder,
        mean=mean,
        std=std,
        device=device,
        splits=["train"],
        max_samples=sanity_max,
        cache_path=cache_path,
    )

    if len(pairs) == 0:
        raise RuntimeError("No valid sign-text pairs found. Check CSL-Daily path.")

    # --- Train/val split ---
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_split_ratio))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    train_ds = SignTextPairDataset(train_pairs)
    val_ds = SignTextPairDataset(val_pairs)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=SignTextPairDataset.collate_fn,
        drop_last=len(train_ds) > args.batch_size,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=SignTextPairDataset.collate_fn,
    )

    print(f"[data] train: {len(train_ds)} / val: {len(val_ds)}")

    # --- Build contrastive model ---
    model = SignTextContrastiveModel(
        vae_dim=vae_dim,
        proj_dim=args.proj_dim,
        mt5_model_path=mt5_path,
        init_temperature=args.temperature,
    ).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Trainable params: {n_trainable:,} "
          f"(sign_proj + text_proj + log_temperature)")

    # --- Save run config ---
    os.makedirs(args.output_dir, exist_ok=True)
    run_cfg = vars(args)
    run_cfg["vae_dim"] = vae_dim
    run_cfg["n_train"] = len(train_ds)
    run_cfg["n_val"] = len(val_ds)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2)

    # --- Sanity mode: 5 epochs ---
    epochs = 5 if args.sanity else args.epochs
    patience = epochs + 1 if args.sanity else args.patience  # no early stopping in sanity

    print(f"\n{'='*60}")
    print(f"  Contrastive pre-training — {'SANITY MODE' if args.sanity else 'FULL RUN'}")
    print(f"  epochs={epochs}  batch_size={args.batch_size}  lr={args.lr}")
    print(f"  proj_dim={args.proj_dim}  temperature_init={args.temperature}")
    print(f"  output: {args.output_dir}")
    print(f"{'='*60}\n")

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        output_dir=args.output_dir,
        patience=patience,
    )

    # --- Sanity verification ---
    if args.sanity:
        first_loss = history[0]["train_loss"]
        last_loss = history[-1]["train_loss"]
        first_sim = history[0].get("train_acc_s2t", 0)
        last_sim = history[-1].get("train_acc_s2t", 0)
        print(f"\n[sanity] first_loss={first_loss:.4f}  last_loss={last_loss:.4f}")
        print(f"[sanity] first_acc={first_sim:.3f}  last_acc={last_sim:.3f}")
        if last_loss < first_loss * 0.9:
            print("[sanity] PASS ✓ — InfoNCE loss decreased by >10%")
        else:
            print("[sanity] WARN — loss did not decrease significantly; check config/data")

    print(f"\n[done] Sign projection saved to {args.output_dir}/best_sign_proj.pt")
    print("[done] Load into mT5 with: --contrastive_pretrain_ckpt <path>/best_sign_proj.pt")


if __name__ == "__main__":
    main()
