#!/usr/bin/env python3
import argparse
import gzip
import json
import os
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    cfg.model.params.hand_vae_cfg = OmegaConf.create(OmegaConf.to_container(cfg.vq.hand256_lfq4, resolve=True))
    cfg.model.params.rhand_vae_cfg = OmegaConf.create(OmegaConf.to_container(cfg.vq.hand256_lfq4, resolve=True))
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


def encode_sample(holder: VAEHolder, motion_norm: torch.Tensor):
    with torch.no_grad():
        cur = motion_norm.unsqueeze(0)  # [1, T, 133]
        fb = torch.cat([cur[..., :30], cur[..., 120:]], dim=-1)
        fl = cur[..., 30:75]
        fr = cur[..., 75:120]
        eb = holder.vae.encode_continuous(fb)
        el = holder.hand_vae.encode_continuous(fl)
        er = holder.rhand_vae.encode_continuous(fr)
        mt = min(eb.shape[1], el.shape[1], er.shape[1])
        eb = eb[:, :mt]
        el = el[:, :mt]
        er = er[:, :mt]
        combined = torch.cat([eb, el, er], dim=-1)
        return {
            "body_seq": eb[0].cpu(),
            "lhand_seq": el[0].cpu(),
            "rhand_seq": er[0].cpu(),
            "combined_seq": combined[0].cpu(),
            "latent_len": int(mt),
        }


def mean_pool(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=0)


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def sample_pairs(groups, rng, target_count):
    keys = [k for k, v in groups.items() if len(v) >= 2]
    pairs = []
    if not keys:
        return pairs
    while len(pairs) < target_count:
        k = rng.choice(keys)
        items = groups[k]
        i, j = rng.sample(range(len(items)), 2)
        pairs.append((items[i], items[j]))
    return pairs


def pairwise_stats(reps, pairs):
    if not pairs:
        return {"count": 0, "mean": None, "std": None}
    sims = []
    for i, j in pairs:
        sims.append(float(torch.dot(reps[i], reps[j]).item()))
    arr = np.asarray(sims, dtype=np.float64)
    return {
        "count": int(arr.shape[0]),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
    }


def matched_win_rate(reps, pos_pairs, neg_pairs):
    n = min(len(pos_pairs), len(neg_pairs))
    if n == 0:
        return None
    wins = 0
    for idx in range(n):
        i, j = pos_pairs[idx]
        a, b = neg_pairs[idx]
        pos = float(torch.dot(reps[i], reps[j]).item())
        neg = float(torch.dot(reps[a], reps[b]).item())
        wins += float(pos > neg)
    return float(wins / n)


def top1_retrieval_accuracy(reps, sentence_ids, signer_ids, eligible_indices, device, batch_size=512):
    if len(eligible_indices) == 0:
        return None
    x = reps.to(device)
    x = l2_normalize(x)
    sentence_ids = np.asarray(sentence_ids)
    signer_ids = np.asarray(signer_ids)
    correct = 0
    total = 0
    all_idx = torch.arange(x.shape[0], device=device)
    for start in range(0, len(eligible_indices), batch_size):
        batch_ids = eligible_indices[start:start + batch_size]
        batch = x[batch_ids]
        sims = batch @ x.T  # [B, N]
        self_mask = all_idx.unsqueeze(0) == torch.as_tensor(batch_ids, device=device).unsqueeze(1)
        signer_mask = torch.as_tensor(signer_ids[batch_ids], device=device).unsqueeze(1) == torch.as_tensor(signer_ids, device=device).unsqueeze(0)
        sims = sims.masked_fill(self_mask | signer_mask, -1e9)
        nn_idx = sims.argmax(dim=1).cpu().numpy()
        for anchor, nn in zip(batch_ids, nn_idx):
            total += 1
            if sentence_ids[anchor] == sentence_ids[nn]:
                correct += 1
    return float(correct / max(total, 1))


def sample_plot_indices(meta, per_sentence_limit, max_sentences, seed):
    rng = random.Random(seed)
    by_sentence = defaultdict(list)
    for idx, item in enumerate(meta):
        by_sentence[item["sentence_id"]].append(idx)
    eligible = [sid for sid, idxs in by_sentence.items() if len({meta[i]["signer_id"] for i in idxs}) >= 2]
    eligible.sort(key=lambda sid: (-len(by_sentence[sid]), sid))
    chosen_sentences = eligible[:max_sentences]
    if len(chosen_sentences) < max_sentences:
        extra = [sid for sid in by_sentence.keys() if sid not in chosen_sentences]
        rng.shuffle(extra)
        chosen_sentences.extend(extra[: max_sentences - len(chosen_sentences)])
    chosen = []
    for sid in chosen_sentences:
        idxs = by_sentence[sid][:]
        rng.shuffle(idxs)
        chosen.extend(idxs[:per_sentence_limit])
    return chosen


def pca_project(x: np.ndarray, n_components: int = 2) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    u, s, vh = np.linalg.svd(x, full_matrices=False)
    return x @ vh[:n_components].T


def tsne_project(x: np.ndarray, seed: int):
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        return None
    perplexity = min(30, max(5, (x.shape[0] - 1) // 3))
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(x)


def plot_similarity_histograms(results, output_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    colors = {
        "same_sentence_diff_signer": "#1b9e77",
        "diff_sentence_same_signer": "#d95f02",
        "diff_sentence_diff_signer": "#7570b3",
    }
    titles = {
        "same_sentence_diff_signer": "Same Sentence, Different Signer",
        "diff_sentence_same_signer": "Different Sentence, Same Signer",
        "diff_sentence_diff_signer": "Different Sentence, Different Signer",
    }
    for ax, rep_name in zip(axes, ["body", "lhand", "rhand", "combined"]):
        rep = results["representations"][rep_name]
        for key in ["same_sentence_diff_signer", "diff_sentence_same_signer", "diff_sentence_diff_signer"]:
            stats = rep[key]
            mu = stats["mean"]
            sigma = max(stats["std"], 1e-6)
            xs = np.linspace(max(-1.0, mu - 4 * sigma), min(1.0, mu + 4 * sigma), 300)
            ys = np.exp(-0.5 * ((xs - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
            ax.plot(xs, ys, label=titles[key], color=colors[key], linewidth=2)
            ax.axvline(mu, color=colors[key], linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title(rep_name)
        ax.set_xlim(-0.2, 1.02)
        ax.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / "similarity_density.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def scatter_embeddings(coords: np.ndarray, meta_subset, output_path: Path, title: str):
    sentence_ids = [m["sentence_id"] for m in meta_subset]
    signer_ids = [m["signer_id"] for m in meta_subset]
    unique_sentences = sorted(set(sentence_ids))
    unique_signers = sorted(set(signer_ids))
    cmap = plt.get_cmap("tab20", max(len(unique_sentences), 1))
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8"]
    sentence_to_color = {sid: cmap(i % cmap.N) for i, sid in enumerate(unique_sentences)}
    signer_to_marker = {sid: markers[i % len(markers)] for i, sid in enumerate(unique_signers)}

    fig, ax = plt.subplots(figsize=(10, 8))
    for sid in unique_sentences:
        idxs = [i for i, x in enumerate(sentence_ids) if x == sid]
        for signer in sorted(set(signer_ids[i] for i in idxs)):
            sub = [i for i in idxs if signer_ids[i] == signer]
            ax.scatter(
                coords[sub, 0],
                coords[sub, 1],
                c=[sentence_to_color[sid]],
                marker=signer_to_marker[signer],
                s=55,
                alpha=0.85,
                edgecolors="black",
                linewidths=0.3,
                label=f"{sid} / signer {signer}",
            )
    ax.set_title(title)
    ax.grid(alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    dedup = {}
    for handle, label in zip(handles, labels):
        dedup.setdefault(label, handle)
    ax.legend(
        dedup.values(),
        dedup.keys(),
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_visualizations(results, pooled, meta, output_dir: Path, seed: int, viz_sentences: int, viz_per_sentence: int):
    plot_similarity_histograms(results, output_dir)

    selected = sample_plot_indices(meta, per_sentence_limit=viz_per_sentence, max_sentences=viz_sentences, seed=seed)
    if not selected:
        return
    rep = pooled["combined"][selected].cpu().numpy()
    meta_subset = [meta[i] for i in selected]
    pca_coords = pca_project(rep, n_components=2)
    scatter_embeddings(pca_coords, meta_subset, output_dir / "combined_pca.png", "Combined Embeddings PCA")

    tsne_coords = tsne_project(rep, seed=seed)
    if tsne_coords is not None:
        scatter_embeddings(tsne_coords, meta_subset, output_dir / "combined_tsne.png", "Combined Embeddings t-SNE")


def main():
    ap = argparse.ArgumentParser(description="Analyze CSL-Daily VAE encoder embedding discrimination.")
    ap.add_argument("--cfg", default="configs/soke_mt5_m2t.yaml")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--output_dir", default="experiments/analysis/csl_vae_encoder_similarity")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_samples", type=int, default=0, help="0 means use all samples")
    ap.add_argument("--neg_multiplier", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--retrieval_anchors", type=int, default=2048)
    ap.add_argument("--force_hand_lfq4", action="store_true", default=True)
    ap.add_argument("--no_force_hand_lfq4", dest="force_hand_lfq4", action="store_false")
    ap.add_argument("--viz_sentences", type=int, default=20)
    ap.add_argument("--viz_per_sentence", type=int, default=4)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_cfg(args.cfg)
    if args.force_hand_lfq4:
        cfg = force_hand_lfq4(cfg)
    csl_root = str(cfg.DATASET.H2S.CSL_ROOT)
    mean, std = load_stats(str(cfg.DATASET.H2S.MEAN_PATH), str(cfg.DATASET.H2S.STD_PATH))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    holder = build_model(cfg, device)

    samples = []
    for split in args.splits:
        with gzip.open(os.path.join(csl_root, f"csl_clean.{split}"), "rb") as f:
            ann = pickle.load(f)
        for item in ann:
            item = dict(item)
            item["split"] = split
            samples.append(item)
    if args.max_samples > 0:
        samples = samples[: int(args.max_samples)]

    pooled = {"body": [], "lhand": [], "rhand": [], "combined": []}
    meta = []

    for item in tqdm(samples, desc="Encoding CSL samples"):
        clip_poses, clip_text, name, _ = load_csl_sample(item, csl_root, need_pose=True, need_code=False)
        if clip_poses is None:
            continue
        motion_norm = normalize_pose(clip_poses, mean, std).to(device)
        emb = encode_sample(holder, motion_norm)
        pooled["body"].append(mean_pool(emb["body_seq"]))
        pooled["lhand"].append(mean_pool(emb["lhand_seq"]))
        pooled["rhand"].append(mean_pool(emb["rhand_seq"]))
        pooled["combined"].append(mean_pool(emb["combined_seq"]))
        meta.append({
            "name": name,
            "text": clip_text,
            "sentence_id": name.split("_")[0],
            "signer_id": int(item["signer"]),
            "split": item["split"],
            "num_frames": int(clip_poses.shape[0]),
            "latent_len": int(emb["latent_len"]),
        })

    for k in pooled:
        pooled[k] = l2_normalize(torch.stack(pooled[k], dim=0))

    by_sentence = defaultdict(list)
    by_signer = defaultdict(list)
    for idx, info in enumerate(meta):
        by_sentence[info["sentence_id"]].append(idx)
        by_signer[info["signer_id"]].append(idx)

    positive_pairs = []
    for _, indices in by_sentence.items():
        if len(indices) < 2:
            continue
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a, b = indices[i], indices[j]
                if meta[a]["signer_id"] != meta[b]["signer_id"]:
                    positive_pairs.append((a, b))

    neg_same_signer_groups = defaultdict(list)
    neg_diff_signer_groups = defaultdict(list)
    all_indices = list(range(len(meta)))
    same_signer_neg = []
    diff_signer_neg = []
    target_negs = max(len(positive_pairs) * max(args.neg_multiplier, 1), len(positive_pairs))
    while len(same_signer_neg) < target_negs or len(diff_signer_neg) < target_negs:
        a, b = rng.sample(all_indices, 2)
        if meta[a]["sentence_id"] == meta[b]["sentence_id"]:
            continue
        if meta[a]["signer_id"] == meta[b]["signer_id"]:
            if len(same_signer_neg) < target_negs:
                same_signer_neg.append((a, b))
        else:
            if len(diff_signer_neg) < target_negs:
                diff_signer_neg.append((a, b))

    multi_signer_indices = [i for i, m in enumerate(meta) if len(by_sentence[m["sentence_id"]]) >= 2]
    rng.shuffle(multi_signer_indices)
    retrieval_anchors = multi_signer_indices[: min(args.retrieval_anchors, len(multi_signer_indices))]

    sentence_ids = [m["sentence_id"] for m in meta]
    signer_ids = [m["signer_id"] for m in meta]

    results = {
        "config": {
            "cfg": args.cfg,
            "splits": args.splits,
            "device": str(device),
            "seed": int(args.seed),
            "force_hand_lfq4": bool(args.force_hand_lfq4),
            "num_samples": int(len(meta)),
            "num_sentence_ids": int(len(by_sentence)),
            "num_positive_pairs": int(len(positive_pairs)),
            "num_same_signer_neg_pairs": int(len(same_signer_neg)),
            "num_diff_signer_neg_pairs": int(len(diff_signer_neg)),
        },
        "representations": {},
    }

    for rep_name, reps in pooled.items():
        pos_stats = pairwise_stats(reps, positive_pairs)
        same_signer_stats = pairwise_stats(reps, same_signer_neg)
        diff_signer_stats = pairwise_stats(reps, diff_signer_neg)
        results["representations"][rep_name] = {
            "same_sentence_diff_signer": pos_stats,
            "diff_sentence_same_signer": same_signer_stats,
            "diff_sentence_diff_signer": diff_signer_stats,
            "win_rate_pos_gt_same_signer_neg": matched_win_rate(reps, positive_pairs, same_signer_neg),
            "win_rate_pos_gt_diff_signer_neg": matched_win_rate(reps, positive_pairs, diff_signer_neg),
        }
        if rep_name == "combined":
            results["representations"][rep_name]["top1_retrieval_acc_diff_signer_only"] = top1_retrieval_accuracy(
                reps, sentence_ids, signer_ids, retrieval_anchors, device=device
            )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    save_visualizations(
        results,
        pooled,
        meta,
        out_dir,
        seed=args.seed,
        viz_sentences=args.viz_sentences,
        viz_per_sentence=args.viz_per_sentence,
    )

    # Save a compact human-readable report
    lines = []
    cfg_info = results["config"]
    lines.append("CSL VAE Encoder Similarity Analysis")
    lines.append(f"num_samples: {cfg_info['num_samples']}")
    lines.append(f"num_sentence_ids: {cfg_info['num_sentence_ids']}")
    lines.append(f"num_positive_pairs: {cfg_info['num_positive_pairs']}")
    lines.append(f"force_hand_lfq4: {cfg_info['force_hand_lfq4']}")
    lines.append("")
    for rep_name, rep in results["representations"].items():
        lines.append(f"[{rep_name}]")
        for tag in ["same_sentence_diff_signer", "diff_sentence_same_signer", "diff_sentence_diff_signer"]:
            stats = rep[tag]
            lines.append(
                f"  {tag}: mean={stats['mean']:.4f} std={stats['std']:.4f} "
                f"p25={stats['p25']:.4f} p50={stats['p50']:.4f} p75={stats['p75']:.4f}"
            )
        lines.append(f"  win_rate_pos_gt_same_signer_neg: {rep['win_rate_pos_gt_same_signer_neg']:.4f}")
        lines.append(f"  win_rate_pos_gt_diff_signer_neg: {rep['win_rate_pos_gt_diff_signer_neg']:.4f}")
        if "top1_retrieval_acc_diff_signer_only" in rep:
            lines.append(f"  top1_retrieval_acc_diff_signer_only: {rep['top1_retrieval_acc_diff_signer_only']:.4f}")
        lines.append("")
    report_path = out_dir / "report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved: {json_path}")
    print(f"saved: {report_path}")
    print(f"saved: {out_dir / 'similarity_density.png'}")
    print(f"saved: {out_dir / 'combined_pca.png'}")
    if (out_dir / "combined_tsne.png").exists():
        print(f"saved: {out_dir / 'combined_tsne.png'}")


if __name__ == "__main__":
    main()
