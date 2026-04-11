#!/usr/bin/env python3
"""R015: Temporal density audit for VAE embeddings.

Computes within-sample adjacent-frame cosine similarity of the 3-VAE combined
embeddings for a sample of CSL-Daily sequences. High similarity between adjacent
frames → high temporal density → attention degeneracy risk in downstream LM.

Usage (from repo root):
    python scripts/analysis/temporal_density_audit.py \
        --cfg configs/soke_mt5_csl_m2t.yaml \
        --n_samples 50 \
        --output_dir experiments/analysis/temporal_density_audit

Results saved to:
    experiments/analysis/temporal_density_audit/summary.json
    experiments/analysis/temporal_density_audit/report.txt
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from mGPT.config import get_module_config, instantiate_from_config
from mGPT.data.humanml.load_data import load_csl_sample
from mGPT.utils.load_checkpoint import load_pretrained_vae


# ---------------------------------------------------------------------------
# VAE wrapper (mirrors analyze_csl_vae_encoder_similarity.py)
# ---------------------------------------------------------------------------

class VAEHolder(nn.Module):
    def __init__(self, motion_vae, hand_vae_cfg, rhand_vae_cfg):
        super().__init__()
        self.vae = instantiate_from_config(motion_vae)
        self.hand_vae = instantiate_from_config(hand_vae_cfg)
        self.rhand_vae = instantiate_from_config(rhand_vae_cfg)


def register_resolver():
    try:
        OmegaConf.register_new_resolver("eval", eval)
    except ValueError:
        pass


def load_cfg(cfg_path: str):
    register_resolver()
    cfg_assets = OmegaConf.load("./configs/assets.yaml")
    cfg_base = OmegaConf.load(f"{cfg_assets.CONFIG_FOLDER}/default.yaml")
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(cfg_path))
    if not cfg_exp.FULL_CONFIG:
        cfg_exp = get_module_config(cfg_exp, cfg_assets.CONFIG_FOLDER)
    return OmegaConf.merge(cfg_exp, cfg_assets)


def force_hand_lfq4(cfg):
    cfg.model.params.hand_vae_cfg = OmegaConf.create(
        OmegaConf.to_container(cfg.vq.hand256_lfq4, resolve=True)
    )
    cfg.model.params.rhand_vae_cfg = OmegaConf.create(
        OmegaConf.to_container(cfg.vq.hand256_lfq4, resolve=True)
    )
    return cfg


def load_stats(mean_path: str, std_path: str):
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


def build_model(cfg, device):
    holder = VAEHolder(
        motion_vae=cfg.model.params.motion_vae,
        hand_vae_cfg=cfg.model.params.hand_vae_cfg,
        rhand_vae_cfg=cfg.model.params.rhand_vae_cfg,
    )
    load_pretrained_vae(cfg, holder, logger=None)
    holder.eval().to(device)
    for p in holder.parameters():
        p.requires_grad = False
    return holder


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_sample(holder: VAEHolder, motion_norm: torch.Tensor, device):
    """Returns combined VAE embedding of shape [T', 1536]."""
    with torch.no_grad():
        cur = motion_norm.unsqueeze(0).to(device)  # [1, T, 133]
        fb = torch.cat([cur[..., :30], cur[..., 120:]], dim=-1)
        fl = cur[..., 30:75]
        fr = cur[..., 75:120]
        eb = holder.vae.encode_continuous(fb)       # [1, T', 512]
        el = holder.hand_vae.encode_continuous(fl)  # [1, T', 512]
        er = holder.rhand_vae.encode_continuous(fr) # [1, T', 512]
        mt = min(eb.shape[1], el.shape[1], er.shape[1])
        combined = torch.cat([eb[:, :mt], el[:, :mt], er[:, :mt]], dim=-1)  # [1, T', 1536]
        return combined[0].cpu()  # [T', 1536]


# ---------------------------------------------------------------------------
# Temporal density metrics
# ---------------------------------------------------------------------------

def within_sample_adjacent_cosine(seq: torch.Tensor) -> dict:
    """Compute pairwise cosine similarity between adjacent frames.

    seq: [T', D]
    Returns dict with mean/std/percentiles of adjacent cosine similarities.
    If T' < 2, returns None.
    """
    T = seq.shape[0]
    if T < 2:
        return None
    # L2-normalize each frame
    norms = seq.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    seq_n = seq / norms  # [T', D]
    # Dot product of consecutive frames
    sims = (seq_n[:-1] * seq_n[1:]).sum(dim=-1).numpy()  # [T'-1]
    return {
        "n_pairs": int(len(sims)),
        "mean": float(sims.mean()),
        "std": float(sims.std()),
        "min": float(sims.min()),
        "max": float(sims.max()),
        "p25": float(np.percentile(sims, 25)),
        "p50": float(np.percentile(sims, 50)),
        "p75": float(np.percentile(sims, 75)),
        "p90": float(np.percentile(sims, 90)),
        "p95": float(np.percentile(sims, 95)),
        "frac_gt_0.9": float((sims > 0.9).mean()),
        "frac_gt_0.95": float((sims > 0.95).mean()),
        "frac_gt_0.99": float((sims > 0.99).mean()),
    }


def within_sample_all_pairs_cosine(seq: torch.Tensor, max_pairs: int = 1000) -> dict:
    """Sample all-pairs cosine similarity to check if non-adjacent frames are also similar.

    seq: [T', D]
    """
    T = seq.shape[0]
    if T < 2:
        return None
    norms = seq.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    seq_n = seq / norms
    # All pairs
    sims_mat = (seq_n @ seq_n.T).numpy()  # [T', T']
    # Take upper triangle (excluding diagonal)
    idx = np.triu_indices(T, k=1)
    sims = sims_mat[idx]
    if len(sims) > max_pairs:
        rng = np.random.RandomState(42)
        sims = rng.choice(sims, size=max_pairs, replace=False)
    return {
        "n_pairs": int(len(sims)),
        "mean": float(sims.mean()),
        "std": float(sims.std()),
        "p50": float(np.percentile(sims, 50)),
        "p90": float(np.percentile(sims, 90)),
    }


def per_part_adjacent_cosine(holder: VAEHolder, motion_norm: torch.Tensor, device) -> dict:
    """Return per-VAE-part adjacent cosine similarity."""
    with torch.no_grad():
        cur = motion_norm.unsqueeze(0).to(device)
        fb = torch.cat([cur[..., :30], cur[..., 120:]], dim=-1)
        fl = cur[..., 30:75]
        fr = cur[..., 75:120]
        eb = holder.vae.encode_continuous(fb)[0].cpu()
        el = holder.hand_vae.encode_continuous(fl)[0].cpu()
        er = holder.rhand_vae.encode_continuous(fr)[0].cpu()
    return {
        "body": within_sample_adjacent_cosine(eb),
        "lhand": within_sample_adjacent_cosine(el),
        "rhand": within_sample_adjacent_cosine(er),
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_stats(per_sample_stats: list) -> dict:
    """Aggregate per-sample stat dicts into a single summary."""
    all_means = [s["mean"] for s in per_sample_stats if s is not None]
    all_p50 = [s["p50"] for s in per_sample_stats if s is not None]
    all_p90 = [s["p90"] for s in per_sample_stats if s is not None]
    all_frac99 = [s.get("frac_gt_0.99", float("nan")) for s in per_sample_stats if s is not None]
    all_frac95 = [s.get("frac_gt_0.95", float("nan")) for s in per_sample_stats if s is not None]
    all_frac90 = [s.get("frac_gt_0.9", float("nan")) for s in per_sample_stats if s is not None]
    arr_means = np.array(all_means)
    arr_p50 = np.array(all_p50)
    arr_p90 = np.array(all_p90)
    arr_frac99 = np.array([x for x in all_frac99 if not np.isnan(x)])
    arr_frac95 = np.array([x for x in all_frac95 if not np.isnan(x)])
    arr_frac90 = np.array([x for x in all_frac90 if not np.isnan(x)])
    return {
        "n_samples": len(all_means),
        "mean_of_mean_adjacent_sim": float(arr_means.mean()) if len(arr_means) else None,
        "std_of_mean_adjacent_sim": float(arr_means.std()) if len(arr_means) else None,
        "mean_of_p50_adjacent_sim": float(arr_p50.mean()) if len(arr_p50) else None,
        "mean_of_p90_adjacent_sim": float(arr_p90.mean()) if len(arr_p90) else None,
        "mean_frac_gt_0.99": float(arr_frac99.mean()) if len(arr_frac99) else None,
        "mean_frac_gt_0.95": float(arr_frac95.mean()) if len(arr_frac95) else None,
        "mean_frac_gt_0.9": float(arr_frac90.mean()) if len(arr_frac90) else None,
    }


# ---------------------------------------------------------------------------
# Interpretation helper
# ---------------------------------------------------------------------------

def interpret(agg: dict) -> str:
    mean_adj = agg.get("mean_of_mean_adjacent_sim")
    frac99 = agg.get("mean_frac_gt_0.99")
    frac95 = agg.get("mean_frac_gt_0.95")
    if mean_adj is None:
        return "Insufficient data."
    lines = []
    lines.append(f"Mean adjacent-frame cosine similarity: {mean_adj:.4f}")
    if frac95 is not None:
        lines.append(f"  Fraction of adjacent pairs with sim > 0.95: {frac95:.3f}")
    if frac99 is not None:
        lines.append(f"  Fraction of adjacent pairs with sim > 0.99: {frac99:.3f}")

    if mean_adj > 0.97:
        verdict = (
            "H2 CONFIRMED (HIGH CONFIDENCE): Temporal density is extremely high. "
            "Adjacent frames are near-identical (mean sim > 0.97). "
            "Attention over this sequence is likely degenerate — "
            "the LM cannot distinguish individual sign frames. "
            "→ Apply temporal contrastive (SignCL-style F2-A) or temporal subsampling."
        )
    elif mean_adj > 0.90:
        verdict = (
            "H2 LIKELY: Temporal density is high (mean sim > 0.90). "
            "Adjacent frames carry very similar information. "
            "Some temporal redundancy present — temporal contrastive may help (F2-A). "
            "Also consider investigating global vs. local alignment (H1)."
        )
    elif mean_adj > 0.70:
        verdict = (
            "H2 PARTIAL: Moderate temporal density. "
            "Adjacent frames overlap significantly but some variation present. "
            "Likely not the primary failure mode — investigate H1 (global vs. local alignment) first."
        )
    else:
        verdict = (
            "H2 UNLIKELY: Temporal density is low (mean sim < 0.70). "
            "Adjacent frames have enough variation — temporal density is not the bottleneck. "
            "Primary failure mode is likely H1 (sequence-level InfoNCE ≠ per-frame decodability) "
            "or H4 (data scale). Proceed with F3 (ST-GCN encoder)."
        )
    lines.append("")
    lines.append("VERDICT:")
    lines.append(verdict)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="R015: Temporal density audit of CSL-Daily VAE embeddings.")
    ap.add_argument("--cfg", default="configs/soke_mt5_csl_m2t.yaml")
    ap.add_argument("--n_samples", type=int, default=50, help="Number of CSL-Daily train samples to audit")
    ap.add_argument("--output_dir", default="experiments/analysis/temporal_density_audit")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force_hand_lfq4", action="store_true", default=True)
    ap.add_argument("--no_force_hand_lfq4", dest="force_hand_lfq4", action="store_false")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_cfg(args.cfg)
    if args.force_hand_lfq4:
        cfg = force_hand_lfq4(cfg)

    csl_root = str(cfg.DATASET.H2S.CSL_ROOT)
    mean, std = load_stats(
        str(cfg.DATASET.H2S.MEAN_PATH),
        str(cfg.DATASET.H2S.STD_PATH),
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    holder = build_model(cfg, device)

    # Load CSL-Daily train split
    ann_path = os.path.join(csl_root, "csl_clean.train")
    with gzip.open(ann_path, "rb") as f:
        ann = pickle.load(f)

    # Subsample
    rng = np.random.RandomState(args.seed)
    indices = rng.choice(len(ann), size=min(args.n_samples, len(ann)), replace=False)
    samples = [ann[i] for i in indices]

    combined_adj_stats = []
    per_part_adj_stats = {"body": [], "lhand": [], "rhand": []}
    combined_all_pairs_stats = []
    latent_lengths = []
    raw_frame_lengths = []
    compression_ratios = []

    for item in tqdm(samples, desc="Auditing temporal density"):
        clip_poses, clip_text, name, _ = load_csl_sample(
            item, csl_root, need_pose=True, need_code=False
        )
        if clip_poses is None:
            continue
        motion_norm = normalize_pose(clip_poses, mean, std)
        T_raw = clip_poses.shape[0]

        # Combined embedding
        emb = encode_sample(holder, motion_norm, device)  # [T', 1536]
        T_lat = emb.shape[0]

        raw_frame_lengths.append(T_raw)
        latent_lengths.append(T_lat)
        if T_raw > 0:
            compression_ratios.append(T_raw / max(T_lat, 1))

        # Adjacent cosine similarity on combined embedding
        adj_stats = within_sample_adjacent_cosine(emb)
        combined_adj_stats.append(adj_stats)

        # All-pairs (sample up to 1000 pairs per video)
        all_pairs = within_sample_all_pairs_cosine(emb, max_pairs=1000)
        combined_all_pairs_stats.append(all_pairs)

        # Per-part
        pp = per_part_adjacent_cosine(holder, motion_norm, device)
        for part in ["body", "lhand", "rhand"]:
            per_part_adj_stats[part].append(pp[part])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate
    combined_agg = aggregate_stats(combined_adj_stats)
    part_agg = {k: aggregate_stats(v) for k, v in per_part_adj_stats.items()}
    all_pairs_agg = aggregate_stats(combined_all_pairs_stats)

    results = {
        "config": {
            "cfg": args.cfg,
            "n_samples_requested": args.n_samples,
            "n_samples_processed": int(combined_agg["n_samples"]),
            "device": str(device),
            "seed": int(args.seed),
        },
        "raw_frames": {
            "mean": float(np.mean(raw_frame_lengths)) if raw_frame_lengths else None,
            "std": float(np.std(raw_frame_lengths)) if raw_frame_lengths else None,
            "min": int(np.min(raw_frame_lengths)) if raw_frame_lengths else None,
            "max": int(np.max(raw_frame_lengths)) if raw_frame_lengths else None,
        },
        "latent_frames": {
            "mean": float(np.mean(latent_lengths)) if latent_lengths else None,
            "std": float(np.std(latent_lengths)) if latent_lengths else None,
            "min": int(np.min(latent_lengths)) if latent_lengths else None,
            "max": int(np.max(latent_lengths)) if latent_lengths else None,
        },
        "compression_ratio_raw_to_latent": {
            "mean": float(np.mean(compression_ratios)) if compression_ratios else None,
        },
        "adjacent_cosine_similarity_combined": combined_agg,
        "all_pairs_cosine_similarity_combined": all_pairs_agg,
        "adjacent_cosine_similarity_per_part": part_agg,
        "per_sample_adjacent_stats": combined_adj_stats,
    }

    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # Human-readable report
    lines = []
    lines.append("=" * 70)
    lines.append("R015: Temporal Density Audit — CSL-Daily VAE Embeddings")
    lines.append("=" * 70)
    lines.append(f"Samples processed: {combined_agg['n_samples']}")
    raw = results["raw_frames"]
    lat = results["latent_frames"]
    cr = results["compression_ratio_raw_to_latent"]
    if raw["mean"] is not None:
        lines.append(f"Raw frames: mean={raw['mean']:.1f} std={raw['std']:.1f} "
                     f"[{raw['min']}, {raw['max']}]")
    if lat["mean"] is not None:
        lines.append(f"Latent frames: mean={lat['mean']:.1f} std={lat['std']:.1f} "
                     f"[{lat['min']}, {lat['max']}]")
    if cr["mean"] is not None:
        lines.append(f"Compression ratio (raw/latent): {cr['mean']:.2f}×")
    lines.append("")
    lines.append("--- Adjacent-Frame Cosine Similarity (Combined 1536-dim) ---")
    agg = combined_agg
    lines.append(f"  Mean of sample-means:  {agg['mean_of_mean_adjacent_sim']:.4f}")
    lines.append(f"  Std of sample-means:   {agg['std_of_mean_adjacent_sim']:.4f}")
    lines.append(f"  Mean of sample-p50:    {agg['mean_of_p50_adjacent_sim']:.4f}")
    lines.append(f"  Mean of sample-p90:    {agg['mean_of_p90_adjacent_sim']:.4f}")
    lines.append(f"  Mean frac > 0.90:      {agg['mean_frac_gt_0.9']:.3f}")
    lines.append(f"  Mean frac > 0.95:      {agg['mean_frac_gt_0.95']:.3f}")
    lines.append(f"  Mean frac > 0.99:      {agg['mean_frac_gt_0.99']:.3f}")
    lines.append("")
    lines.append("--- All-Pairs Cosine Similarity (Combined, sampled) ---")
    apa = all_pairs_agg
    if apa["mean_of_mean_adjacent_sim"] is not None:
        lines.append(f"  Mean of sample-means:  {apa['mean_of_mean_adjacent_sim']:.4f}")
        lines.append(f"  Mean of sample-p50:    {apa['mean_of_p50_adjacent_sim']:.4f}")
        lines.append(f"  Mean of sample-p90:    {apa['mean_of_p90_adjacent_sim']:.4f}")
    lines.append("")
    lines.append("--- Per-Part Adjacent Cosine Similarity ---")
    for part in ["body", "lhand", "rhand"]:
        pa = part_agg[part]
        if pa["mean_of_mean_adjacent_sim"] is not None:
            lines.append(f"  [{part:5s}] mean={pa['mean_of_mean_adjacent_sim']:.4f}  "
                         f"p50={pa['mean_of_p50_adjacent_sim']:.4f}  "
                         f"p90={pa['mean_of_p90_adjacent_sim']:.4f}  "
                         f"frac>0.99={pa['mean_frac_gt_0.99']:.3f}")
    lines.append("")
    lines.append("--- Interpretation ---")
    lines.append(interpret(combined_agg))
    lines.append("")
    lines.append(f"Saved: {json_path}")

    report = "\n".join(lines)
    report_path = out_dir / "report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
