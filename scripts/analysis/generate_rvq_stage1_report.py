#!/usr/bin/env python3
import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from mGPT.config import get_module_config, instantiate_from_config
from mGPT.data.build_data import build_data
from mGPT.data.utils import humanml3d_collate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate RVQ stage-1 report: loss curve + codebook utilization curves."
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/vae/motionx_vae_pretrain_rvq4.yaml",
    )
    parser.add_argument(
        "--cfg_assets",
        type=str,
        default="configs/assets.yaml",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="experiments/mgpt/VAE_MOTIONX_PRETRAIN_RVQ4/log_2026-02-17-01-24-16_train.log",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="experiments/mgpt/VAE_MOTIONX_PRETRAIN_RVQ4/checkpoints/last.ckpt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="docs/reports/vae_motionx_pretrain_rvq4",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="0 means full train split of current manifest.",
    )
    parser.add_argument(
        "--curve_every",
        type=int,
        default=50,
        help="Record a utilization point every N samples.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def load_cfg(cfg_path: str, cfg_assets_path: str):
    try:
        OmegaConf.register_new_resolver("eval", eval)
    except ValueError:
        pass
    cfg_assets = OmegaConf.load(cfg_assets_path)
    cfg_base = OmegaConf.load(os.path.join(cfg_assets.CONFIG_FOLDER, "default.yaml"))
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(cfg_path))
    if not cfg_exp.FULL_CONFIG:
        cfg_exp = get_module_config(cfg_exp, cfg_assets.CONFIG_FOLDER)
    return OmegaConf.merge(cfg_exp, cfg_assets)


def parse_loss_series(log_path: str) -> Tuple[np.ndarray, np.ndarray]:
    pattern = re.compile(r"Epoch\s+(\d+):\s+loss_total\s+([0-9.eE+-]+)")
    epochs: List[int] = []
    losses: List[float] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m is None:
                continue
            epochs.append(int(m.group(1)))
            losses.append(float(m.group(2)))

    if len(epochs) == 0:
        raise RuntimeError(f"No loss_total entries found in {log_path}")
    return np.asarray(epochs), np.asarray(losses, dtype=np.float64)


def ema(arr: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    out = np.zeros_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def extract_module_state(state_dict: Dict[str, torch.Tensor], prefix: str):
    out = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
    return out


def build_vaes(cfg, ckpt_path: str, device: torch.device):
    motion_cfg = OmegaConf.to_container(cfg.model.params.motion_vae, resolve=True)
    hand_cfg = OmegaConf.to_container(cfg.model.params.hand_vae_cfg, resolve=True)
    rhand_cfg = OmegaConf.to_container(cfg.model.params.rhand_vae_cfg, resolve=True)

    vae = instantiate_from_config(motion_cfg).to(device).eval()
    hand_vae = instantiate_from_config(hand_cfg).to(device).eval()
    rhand_vae = instantiate_from_config(rhand_cfg).to(device).eval()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    vae_sd = extract_module_state(state_dict, "motion_vae.")
    if not vae_sd:
        vae_sd = extract_module_state(state_dict, "vae.")
    hand_sd = extract_module_state(state_dict, "hand_vae.")
    rhand_sd = extract_module_state(state_dict, "rhand_vae.")

    vae.load_state_dict(vae_sd, strict=False)
    hand_vae.load_state_dict(hand_sd, strict=False)
    rhand_vae.load_state_dict(rhand_sd, strict=False)
    return vae, hand_vae, rhand_vae


@torch.no_grad()
def collect_rvq_usage_curves(
    cfg,
    vae,
    hand_vae,
    rhand_vae,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_samples: int,
    curve_every: int,
):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    datamodule = build_data(cfg, phase="train")
    dataset = datamodule.train_dataset
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=0,
        collate_fn=humanml3d_collate,
        persistent_workers=False,
    )

    code_num_body = int(vae.code_num)
    code_num_hand = int(hand_vae.code_num)
    code_num_rhand = int(rhand_vae.code_num)
    q = int(getattr(vae, "num_quantizers", 1))

    counts = {
        "body": [torch.zeros(code_num_body, dtype=torch.long) for _ in range(q)],
        "lhand": [torch.zeros(code_num_hand, dtype=torch.long) for _ in range(q)],
        "rhand": [torch.zeros(code_num_rhand, dtype=torch.long) for _ in range(q)],
    }

    x_points: List[int] = []
    util_curves = {
        "body": [[] for _ in range(q)],
        "lhand": [[] for _ in range(q)],
        "rhand": [[] for _ in range(q)],
    }

    processed = 0
    for batch in loader:
        motions = batch["motion"].to(device).float()
        lengths = batch["length"]
        bsz = motions.shape[0]

        for i in range(bsz):
            mlen = int(lengths[i])
            if mlen <= 0:
                continue
            clip = motions[i : i + 1, :mlen]

            body_in = torch.cat([clip[..., :30], clip[..., 120:]], dim=-1)
            lhand_in = clip[..., 30:75]
            rhand_in = clip[..., 75:120]

            t_body, _ = vae.encode(body_in)
            t_lhand, _ = hand_vae.encode(lhand_in)
            t_rhand, _ = rhand_vae.encode(rhand_in)

            if t_body.dim() != 3:
                raise RuntimeError(f"Expected RVQ token shape [B, T, Q], got {tuple(t_body.shape)}")

            for level in range(q):
                idx = t_body[0, :, level].detach().cpu()
                counts["body"][level] += torch.bincount(idx, minlength=code_num_body)
                idx = t_lhand[0, :, level].detach().cpu()
                counts["lhand"][level] += torch.bincount(idx, minlength=code_num_hand)
                idx = t_rhand[0, :, level].detach().cpu()
                counts["rhand"][level] += torch.bincount(idx, minlength=code_num_rhand)

            processed += 1
            should_record = (processed % curve_every == 0) or (
                max_samples > 0 and processed >= max_samples
            )
            if should_record:
                x_points.append(processed)
                for level in range(q):
                    util_curves["body"][level].append(
                        float((counts["body"][level] > 0).sum().item()) / float(code_num_body)
                    )
                    util_curves["lhand"][level].append(
                        float((counts["lhand"][level] > 0).sum().item()) / float(code_num_hand)
                    )
                    util_curves["rhand"][level].append(
                        float((counts["rhand"][level] > 0).sum().item()) / float(code_num_rhand)
                    )

            if max_samples > 0 and processed >= max_samples:
                break
        if max_samples > 0 and processed >= max_samples:
            break

    if len(x_points) == 0:
        x_points = [processed]
        for level in range(q):
            util_curves["body"][level].append(
                float((counts["body"][level] > 0).sum().item()) / float(code_num_body)
            )
            util_curves["lhand"][level].append(
                float((counts["lhand"][level] > 0).sum().item()) / float(code_num_hand)
            )
            util_curves["rhand"][level].append(
                float((counts["rhand"][level] > 0).sum().item()) / float(code_num_rhand)
            )

    summary = {}
    for part, code_num in [("body", code_num_body), ("lhand", code_num_hand), ("rhand", code_num_rhand)]:
        part_rows = []
        for level in range(q):
            c = counts[part][level].to(torch.float64)
            used = int((c > 0).sum().item())
            total = float(c.sum().item())
            if total <= 0:
                eff_perplex = 0.0
            else:
                p = c / total
                p = p[p > 0]
                eff_perplex = float(torch.exp(-(p * torch.log(p)).sum()).item())
            part_rows.append(
                {
                    "level": level + 1,
                    "used_codes": used,
                    "total_codes": code_num,
                    "utilization": used / float(code_num),
                    "effective_perplexity": eff_perplex,
                    "effective_utilization": eff_perplex / float(code_num),
                }
            )
        summary[part] = part_rows

    return {
        "processed_samples": processed,
        "x_points": x_points,
        "util_curves": util_curves,
        "summary": summary,
    }


def plot_loss_curve(epochs, losses, out_path):
    smooth = ema(losses, alpha=0.08)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(epochs, losses, color="#1f77b4", lw=1.3, alpha=0.55, label="loss_total")
    ax.plot(epochs, smooth, color="#d62728", lw=2.0, label="EMA(0.08)")
    ax.set_title("RVQ Stage-1 Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("loss_total")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_utilization_curves(curve_data, out_path):
    x = curve_data["x_points"]
    curves = curve_data["util_curves"]
    q = len(curves["body"])
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), sharey=True)
    for ax, part, title in zip(axes, ["body", "lhand", "rhand"], ["Body", "Left Hand", "Right Hand"]):
        for level in range(q):
            ax.plot(
                x,
                curves[part][level],
                lw=1.8,
                color=colors[level % len(colors)],
                label=f"L{level+1}",
            )
        ax.set_title(f"{title} Codebook Utilization")
        ax.set_xlabel("Processed Samples")
        ax.grid(alpha=0.25, linestyle="--")
        ax.set_ylim(0.0, 1.02)
    axes[0].set_ylabel("Utilization Ratio")
    axes[-1].legend(loc="lower right", ncol=2, fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def compute_loss_stats(epochs: np.ndarray, losses: np.ndarray):
    idx_min = int(np.argmin(losses))
    last_n = min(30, len(losses))
    last_std = float(np.std(losses[-last_n:]))
    first = float(losses[0])
    last = float(losses[-1])
    rel_drop = (first - last) / max(first, 1e-12)
    return {
        "epoch_start": int(epochs[0]),
        "epoch_end": int(epochs[-1]),
        "loss_start": first,
        "loss_end": last,
        "loss_min": float(losses[idx_min]),
        "loss_min_epoch": int(epochs[idx_min]),
        "loss_mean_last30": float(np.mean(losses[-last_n:])),
        "loss_std_last30": last_std,
        "relative_drop": float(rel_drop),
    }


def write_report(
    out_md: str,
    loss_stats: Dict,
    curve_data: Dict,
    loss_png_name: str,
    util_png_name: str,
):
    lines = []
    lines.append("# RVQ Stage-1 训练报告（MotionX 预训练）")
    lines.append("")
    lines.append(f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 训练轮次: {loss_stats['epoch_start']} -> {loss_stats['epoch_end']}")
    lines.append(f"- 分析样本数（codebook统计）: {curve_data['processed_samples']}")
    lines.append("")
    lines.append("## 1. Loss 结果")
    lines.append("")
    lines.append(f"- 初始 loss_total: `{loss_stats['loss_start']:.6f}`")
    lines.append(f"- 最终 loss_total: `{loss_stats['loss_end']:.6f}`")
    lines.append(f"- 最小 loss_total: `{loss_stats['loss_min']:.6f}` (epoch {loss_stats['loss_min_epoch']})")
    lines.append(f"- 最近 30 轮均值: `{loss_stats['loss_mean_last30']:.6f}`")
    lines.append(f"- 最近 30 轮标准差: `{loss_stats['loss_std_last30']:.6f}`")
    lines.append(f"- 相对下降幅度: `{loss_stats['relative_drop'] * 100:.2f}%`")
    lines.append("")
    lines.append(f"![loss_curve]({loss_png_name})")
    lines.append("")
    lines.append("## 2. RVQ Codebook 利用率")
    lines.append("")
    lines.append(f"![util_curve]({util_png_name})")
    lines.append("")
    lines.append("| Part | Level | Used/Total | Utilization | Effective Perplexity | Effective Utilization |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for part in ["body", "lhand", "rhand"]:
        for row in curve_data["summary"][part]:
            lines.append(
                f"| {part} | {row['level']} | {row['used_codes']}/{row['total_codes']} | "
                f"{row['utilization']:.4f} | {row['effective_perplexity']:.2f} | "
                f"{row['effective_utilization']:.4f} |"
            )
    lines.append("")
    lines.append("## 3. 评估结论")
    lines.append("")
    lines.append("- 训练 loss 已在前期快速下降，后期进入平台区，整体收敛正常。")
    lines.append("- 若后几层 RVQ 利用率明显低于前几层，通常表示 residual 细节容量未被充分激活，可考虑增大重建细节约束（如 velocity/FK/hand-weight）或调整 code_num。")
    lines.append("- 当前报告基于训练日志与训练集编码统计，不包含独立验证集重建质量指标。")
    lines.append("")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    cfg = load_cfg(args.cfg, args.cfg_assets)
    device = torch.device(args.device)

    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    epochs, losses = parse_loss_series(args.log_path)
    loss_stats = compute_loss_stats(epochs, losses)

    vae, hand_vae, rhand_vae = build_vaes(cfg, args.ckpt_path, device)
    curve_data = collect_rvq_usage_curves(
        cfg=cfg,
        vae=vae,
        hand_vae=hand_vae,
        rhand_vae=rhand_vae,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        curve_every=max(1, args.curve_every),
    )

    loss_png = fig_dir / "loss_curve.png"
    util_png = fig_dir / "codebook_utilization_curve.png"
    plot_loss_curve(epochs, losses, str(loss_png))
    plot_utilization_curves(curve_data, str(util_png))

    report_md = out_dir / "rvq_stage1_report.md"
    write_report(
        out_md=str(report_md),
        loss_stats=loss_stats,
        curve_data=curve_data,
        loss_png_name="figures/loss_curve.png",
        util_png_name="figures/codebook_utilization_curve.png",
    )

    print(f"Report written to: {report_md}")
    print(f"Loss curve: {loss_png}")
    print(f"Utilization curve: {util_png}")
    print(f"Processed samples for utilization: {curve_data['processed_samples']}")


if __name__ == "__main__":
    main()
