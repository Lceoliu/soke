#!/usr/bin/env python3
import argparse
import gzip
import json
import os
import pickle
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from mGPT.config import get_module_config, instantiate_from_config
from mGPT.data.humanml.load_data import load_csl_sample
from mGPT.utils.load_checkpoint import load_pretrained_vae


@dataclass
class SampleRecord:
    name: str
    signer_id: int
    sentence_id: str
    split: str
    text: str


class VAEHolder(nn.Module):
    def __init__(self, motion_vae, hand_vae_cfg, rhand_vae_cfg):
        super().__init__()
        self.vae = instantiate_from_config(motion_vae)
        self.hand_vae = instantiate_from_config(hand_vae_cfg)
        self.rhand_vae = instantiate_from_config(rhand_vae_cfg)


class TemporalConvClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so temporal padding stays symmetric.")
        padding = kernel_size // 2
        self.in_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        x = F.normalize(x, dim=-1)
        x = x.transpose(1, 2).contiguous()
        x = self.in_proj(x)
        x = self.temporal(x)
        x = x.transpose(1, 2).contiguous()
        max_len = x.shape[1]
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        x = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return self.head(x)


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


def encode_sample(holder: VAEHolder, motion_norm: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        cur = motion_norm.unsqueeze(0)
        fb = torch.cat([cur[..., :30], cur[..., 120:]], dim=-1)
        fl = cur[..., 30:75]
        fr = cur[..., 75:120]
        eb = holder.vae.encode_continuous(fb)
        el = holder.hand_vae.encode_continuous(fl)
        er = holder.rhand_vae.encode_continuous(fr)
        mt = min(eb.shape[1], el.shape[1], er.shape[1])
        combined = torch.cat([eb[:, :mt], el[:, :mt], er[:, :mt]], dim=-1)
        return combined[0].cpu()


def load_csl_records(csl_root: str) -> List[SampleRecord]:
    records: List[SampleRecord] = []
    for split in ["train", "val", "test"]:
        with gzip.open(os.path.join(csl_root, f"csl_clean.{split}"), "rb") as f:
            ann = pickle.load(f)
        for item in ann:
            name = str(item["name"])
            records.append(
                SampleRecord(
                    name=name,
                    signer_id=int(item["signer"]),
                    sentence_id=name.split("_")[0],
                    split=split,
                    text=str(item["text"]),
                )
            )
    return records


def build_cross_signer_split(
    records: Sequence[SampleRecord],
    seed: int,
    min_train_signers: int,
    max_classes: int,
):
    rng = random.Random(seed)
    by_sentence: Dict[str, List[SampleRecord]] = defaultdict(list)
    for rec in records:
        by_sentence[rec.sentence_id].append(rec)

    eligible = []
    for sentence_id, items in by_sentence.items():
        signer_map: Dict[int, List[SampleRecord]] = defaultdict(list)
        for item in items:
            signer_map[item.signer_id].append(item)
        if len(signer_map) >= int(min_train_signers) + 1:
            eligible.append((sentence_id, signer_map))

    eligible.sort(key=lambda x: (len(x[1]), x[0]), reverse=True)
    if max_classes > 0:
        eligible = eligible[: int(max_classes)]

    train_records = []
    val_records = []
    test_records = []

    for sentence_id, signer_map in eligible:
        signer_ids = sorted(signer_map.keys())
        rng.shuffle(signer_ids)
        test_signer = signer_ids[-1]
        remaining_signers = signer_ids[:-1]
        if len(remaining_signers) < int(min_train_signers):
            continue

        val_signers = []
        if len(remaining_signers) >= int(min_train_signers) + 1:
            val_signers = [remaining_signers[-1]]
            train_signers = remaining_signers[:-1]
        else:
            train_signers = remaining_signers

        if len(train_signers) < int(min_train_signers):
            continue

        train_pool = []
        for signer_id in train_signers:
            items = signer_map[signer_id][:]
            rng.shuffle(items)
            train_pool.extend(items)
        rng.shuffle(train_pool)

        if len(train_pool) < int(min_train_signers):
            continue
        train_records.extend(train_pool)

        for signer_id in val_signers:
            val_records.extend(signer_map[signer_id])

        test_records.extend(signer_map[test_signer])

    return train_records, val_records, test_records


def extract_features(
    holder: VAEHolder,
    csl_root: str,
    records: Sequence[SampleRecord],
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
):
    features = []
    meta = []
    skipped = []
    ann_by_name = {}
    for split in ["train", "val", "test"]:
        with gzip.open(os.path.join(csl_root, f"csl_clean.{split}"), "rb") as f:
            ann = pickle.load(f)
        ann_by_name.update({str(item["name"]): item for item in ann})

    for rec in tqdm(records, desc="Extracting VAE features"):
        ann = ann_by_name[rec.name]
        try:
            clip_poses, _, name, _ = load_csl_sample(ann, csl_root, need_pose=True, need_code=False)
        except Exception as exc:
            skipped.append(
                {
                    "name": rec.name,
                    "sentence_id": rec.sentence_id,
                    "signer_id": rec.signer_id,
                    "source_split": rec.split,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if len(skipped) <= 10:
                tqdm.write(
                    f"[skip] {rec.name} ({rec.split}) failed during pose load: "
                    f"{type(exc).__name__}: {exc}"
                )
            continue
        if clip_poses is None:
            skipped.append(
                {
                    "name": rec.name,
                    "sentence_id": rec.sentence_id,
                    "signer_id": rec.signer_id,
                    "source_split": rec.split,
                    "error_type": "EmptySample",
                    "error": "load_csl_sample returned None",
                }
            )
            continue
        try:
            motion_norm = normalize_pose(clip_poses, mean, std).to(device)
            seq = encode_sample(holder, motion_norm)
        except Exception as exc:
            skipped.append(
                {
                    "name": rec.name,
                    "sentence_id": rec.sentence_id,
                    "signer_id": rec.signer_id,
                    "source_split": rec.split,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if len(skipped) <= 10:
                tqdm.write(
                    f"[skip] {rec.name} ({rec.split}) failed during encoding: "
                    f"{type(exc).__name__}: {exc}"
            )
            continue
        features.append(seq.float())
        meta.append(
            {
                "name": name,
                "sentence_id": rec.sentence_id,
                "signer_id": rec.signer_id,
                "source_split": rec.split,
                "num_frames": int(clip_poses.shape[0]),
                "latent_len": int(seq.shape[0]),
                "text": rec.text,
            }
        )

    if len(features) == 0:
        raise RuntimeError("No valid CSL samples were encoded.")
    return features, meta, skipped


def build_sequence_data(features: Sequence[torch.Tensor], meta: Sequence[dict], sentence_to_label: Dict[str, int]):
    keep_indices = [idx for idx, item in enumerate(meta) if item["sentence_id"] in sentence_to_label]
    if len(keep_indices) != len(meta):
        features = [features[idx] for idx in keep_indices]
        meta = [meta[idx] for idx in keep_indices]
    labels = torch.tensor([sentence_to_label[item["sentence_id"]] for item in meta], dtype=torch.long)
    signer_ids = torch.tensor([int(item["signer_id"]) for item in meta], dtype=torch.long)
    return list(features), labels, signer_ids, meta


def collate_sequences(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([int(seq.shape[0]) for seq in sequences], dtype=torch.long)
    feat_dim = int(sequences[0].shape[1])
    max_len = int(lengths.max().item())
    padded = torch.zeros((len(sequences), max_len, feat_dim), dtype=sequences[0].dtype)
    for idx, seq in enumerate(sequences):
        padded[idx, : seq.shape[0]] = seq
    return padded, lengths, torch.tensor(labels, dtype=torch.long)


def make_loader(features: Sequence[torch.Tensor], labels: torch.Tensor, batch_size: int, shuffle: bool):
    dataset = [(features[idx], int(labels[idx].item())) for idx in range(len(features))]
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=collate_sequences,
    )


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor):
    loss = F.cross_entropy(logits, labels)
    pred = logits.argmax(dim=1)
    acc = (pred == labels).float().mean()
    top5 = logits.topk(k=min(5, logits.shape[1]), dim=1).indices
    top5_acc = (top5 == labels.unsqueeze(1)).any(dim=1).float().mean()
    return {
        "loss": float(loss.item()),
        "acc": float(acc.item()),
        "top5_acc": float(top5_acc.item()),
    }


def evaluate(model: nn.Module, features: Sequence[torch.Tensor], labels: torch.Tensor, batch_size: int, device: torch.device):
    model.eval()
    all_logits = []
    with torch.no_grad():
        for xb, lengths, yb in make_loader(features, labels, batch_size=batch_size, shuffle=False):
            logits = model(xb.to(device), lengths.to(device))
            all_logits.append(logits.cpu())
    logits = torch.cat(all_logits, dim=0)
    return compute_metrics(logits, labels)


def train_classifier(
    model: nn.Module,
    train_features: Sequence[torch.Tensor],
    train_labels: torch.Tensor,
    val_features: Sequence[torch.Tensor],
    val_labels: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = -1.0
    best_state = None
    best_epoch = -1
    wait = 0
    history = []

    train_loader = make_loader(train_features, train_labels, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        for xb, lengths, yb in train_loader:
            xb = xb.to(device)
            lengths = lengths.to(device)
            yb = yb.to(device)
            logits = model(xb, lengths)
            loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * int(yb.shape[0])
            total_correct += int((logits.argmax(dim=1) == yb).sum().item())
            total_count += int(yb.shape[0])

        train_metrics = {
            "loss": total_loss / max(total_count, 1),
            "acc": total_correct / max(total_count, 1),
        }
        val_metrics = evaluate(model, val_features, val_labels, batch_size=batch_size, device=device)
        row = {
            "epoch": epoch,
            "train_loss": float(train_metrics["loss"]),
            "train_acc": float(train_metrics["acc"]),
            "val_loss": float(val_metrics["loss"]),
            "val_acc": float(val_metrics["acc"]),
            "val_top5_acc": float(val_metrics["top5_acc"]),
        }
        history.append(row)
        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_acc']:.4f} val_top5={row['val_top5_acc']:.4f}"
        )

        if row["val_acc"] > best_val_acc:
            best_val_acc = row["val_acc"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}, best_val_acc={best_val_acc:.4f} @ epoch {best_epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    return history, best_epoch


def save_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Train a simple action classifier on frozen CSL VAE embeddings.")
    ap.add_argument("--cfg", default="configs/soke_mt5_csl_m2t.yaml")
    ap.add_argument("--output_dir", default="experiments/analysis/csl_vae_action_classifier")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--classifier", choices=["temporal_conv"], default="temporal_conv")
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--kernel_size", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--min_train_signers", type=int, default=2)
    ap.add_argument("--max_classes", type=int, default=0, help="0 means use all eligible classes.")
    ap.add_argument("--feature_cache", default="")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(args.cfg)
    csl_root = str(cfg.DATASET.H2S.CSL_ROOT)
    mean, std = load_stats(str(cfg.DATASET.H2S.MEAN_PATH), str(cfg.DATASET.H2S.STD_PATH))

    all_records = load_csl_records(csl_root)
    train_records, val_records, test_records = build_cross_signer_split(
        all_records,
        seed=args.seed,
        min_train_signers=args.min_train_signers,
        max_classes=args.max_classes,
    )

    cache_path = Path(args.feature_cache) if args.feature_cache else output_dir / "features.pt"
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("cache_format") == "sequence_v2":
            train_features, train_meta = payload["train_features"], payload["train_meta"]
            val_features, val_meta = payload["val_features"], payload["val_meta"]
            test_features, test_meta = payload["test_features"], payload["test_meta"]
            train_skipped = payload.get("train_skipped", [])
            val_skipped = payload.get("val_skipped", [])
            test_skipped = payload.get("test_skipped", [])
        else:
            payload = None
    else:
        payload = None

    if payload is None:
        holder = build_model(cfg, device)
        train_features, train_meta, train_skipped = extract_features(
            holder, csl_root, train_records, mean, std, device
        )
        val_features, val_meta, val_skipped = extract_features(
            holder, csl_root, val_records, mean, std, device
        )
        test_features, test_meta, test_skipped = extract_features(
            holder, csl_root, test_records, mean, std, device
        )
        torch.save(
            {
                "cache_format": "sequence_v2",
                "train_features": train_features,
                "train_meta": train_meta,
                "val_features": val_features,
                "val_meta": val_meta,
                "test_features": test_features,
                "test_meta": test_meta,
                "train_skipped": train_skipped,
                "val_skipped": val_skipped,
                "test_skipped": test_skipped,
            },
            cache_path,
        )

    class_ids = sorted({item["sentence_id"] for item in train_meta})
    sentence_to_label = {sid: idx for idx, sid in enumerate(class_ids)}

    train_x, train_y, _, train_meta = build_sequence_data(train_features, train_meta, sentence_to_label)
    val_x, val_y, _, val_meta = build_sequence_data(val_features, val_meta, sentence_to_label)
    test_x, test_y, _, test_meta = build_sequence_data(test_features, test_meta, sentence_to_label)
    if len(train_x) == 0 or len(val_x) == 0 or len(test_x) == 0:
        raise RuntimeError(
            f"Unexpected empty split after feature extraction/filtering: "
            f"train={len(train_x)} val={len(val_x)} test={len(test_x)}"
        )

    input_dim = int(train_x[0].shape[1])
    num_classes = int(len(sentence_to_label))
    model = TemporalConvClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=int(args.hidden_dim),
        kernel_size=int(args.kernel_size),
        dropout=float(args.dropout),
    )
    model.to(device)

    history, best_epoch = train_classifier(
        model=model,
        train_features=train_x,
        train_labels=train_y,
        val_features=val_x,
        val_labels=val_y,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )

    train_metrics = evaluate(model, train_x, train_y, batch_size=args.batch_size, device=device)
    val_metrics = evaluate(model, val_x, val_y, batch_size=args.batch_size, device=device)
    test_metrics = evaluate(model, test_x, test_y, batch_size=args.batch_size, device=device)

    summary = {
        "config": {
            "cfg": args.cfg,
            "device": str(device),
            "seed": int(args.seed),
            "classifier": args.classifier,
            "input_dim": input_dim,
            "num_classes": num_classes,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "patience": int(args.patience),
            "min_train_signers": int(args.min_train_signers),
            "kernel_size": int(args.kernel_size),
            "max_classes": int(args.max_classes),
            "feature_cache": str(cache_path),
        },
        "split_stats": {
            "train_samples": int(len(train_x)),
            "val_samples": int(len(val_x)),
            "test_samples": int(len(test_x)),
            "train_classes": int(len({item['sentence_id'] for item in train_meta})),
            "val_classes": int(len({item['sentence_id'] for item in val_meta})),
            "test_classes": int(len({item['sentence_id'] for item in test_meta})),
            "train_skipped": int(len(train_skipped)),
            "val_skipped": int(len(val_skipped)),
            "test_skipped": int(len(test_skipped)),
            "train_latent_mean_len": float(np.mean([item["latent_len"] for item in train_meta])),
            "val_latent_mean_len": float(np.mean([item["latent_len"] for item in val_meta])),
            "test_latent_mean_len": float(np.mean([item["latent_len"] for item in test_meta])),
        },
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
            "best_epoch": int(best_epoch),
        },
        "history": history,
    }

    save_json(output_dir / "summary.json", summary)
    save_json(output_dir / "train_meta.json", train_meta)
    save_json(output_dir / "val_meta.json", val_meta)
    save_json(output_dir / "test_meta.json", test_meta)
    save_json(output_dir / "train_skipped.json", train_skipped)
    save_json(output_dir / "val_skipped.json", val_skipped)
    save_json(output_dir / "test_skipped.json", test_skipped)
    torch.save({"state_dict": model.state_dict(), "sentence_to_label": sentence_to_label}, output_dir / "best_classifier.pt")

    report_lines = [
        "CSL VAE Action Classifier",
        f"cfg: {args.cfg}",
        f"device: {device}",
        f"classifier: {args.classifier}",
        f"num_classes: {num_classes}",
        f"train_samples: {len(train_x)}",
        f"val_samples: {len(val_x)}",
        f"test_samples: {len(test_x)}",
        f"train_skipped: {len(train_skipped)}",
        f"val_skipped: {len(val_skipped)}",
        f"test_skipped: {len(test_skipped)}",
        f"best_epoch: {best_epoch}",
        "",
        f"train_acc: {train_metrics['acc']:.4f}",
        f"val_acc: {val_metrics['acc']:.4f}",
        f"test_acc: {test_metrics['acc']:.4f}",
        f"test_top5_acc: {test_metrics['top5_acc']:.4f}",
        "",
        "Note: each class keeps at least 2 signers in train and 1 signer in test; val is sampled from train-signers only.",
    ]
    (output_dir / "report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
