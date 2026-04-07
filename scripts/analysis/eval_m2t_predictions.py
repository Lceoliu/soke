#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from mGPT.config import get_module_config
from mGPT.data.build_data import build_data
from mGPT.metrics.utils import bleu, rouge
from mGPT.models.build_model import build_model
from mGPT.utils.load_checkpoint import load_pretrained, load_pretrained_vae


class PermutedMotionDataset(Dataset):
    def __init__(self, base_dataset, index_map):
        self.base_dataset = base_dataset
        self.index_map = index_map

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = list(self.base_dataset[idx])
        perm_item = self.base_dataset[self.index_map[idx]]
        item[1] = perm_item[1]
        item[2] = perm_item[2]
        if len(item) >= 12:
            item[10] = perm_item[10]
            item[11] = perm_item[11]
        return tuple(item)


def load_cfg(cfg_path: str):
    try:
        OmegaConf.register_new_resolver("eval", eval)
    except ValueError:
        pass
    cfg_assets = OmegaConf.load("./configs/assets.yaml")
    cfg_base = OmegaConf.load(f"{cfg_assets.CONFIG_FOLDER}/default.yaml")
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(cfg_path))
    if not cfg_exp.FULL_CONFIG:
        cfg_exp = get_module_config(cfg_exp, cfg_assets.CONFIG_FOLDER)
    return OmegaConf.merge(cfg_exp, cfg_assets)


def build_index_permutation(dataset, seed: int):
    by_src = defaultdict(list)
    for idx in range(len(dataset)):
        src = str(dataset[idx][9])
        by_src[src].append(idx)

    mapping = {}
    rng = random.Random(seed)
    for _, indices in by_src.items():
        if len(indices) <= 1:
            mapping[indices[0]] = indices[0]
            continue
        shuffled = list(indices)
        rng.shuffle(shuffled)
        for i, src_idx in enumerate(shuffled):
            mapping[src_idx] = shuffled[(i + 1) % len(shuffled)]
    return mapping


def make_loader(datamodule, split: str, motion_mode: str, batch_size: int, num_workers: int, shuffle_seed: int):
    dataloader_options = datamodule.dataloader_options.copy()
    dataloader_options["batch_size"] = int(batch_size)
    dataloader_options["num_workers"] = int(num_workers)
    persistent_workers = int(num_workers) > 0

    if split == "train":
        datamodule.setup("fit")
        dataset = datamodule.train_dataset
    elif split == "val":
        datamodule.setup("fit")
        dataset = datamodule.val_dataset
    else:
        datamodule.setup("test")
        dataset = datamodule.test_dataset

    if motion_mode == "shuffle_by_src":
        dataset = PermutedMotionDataset(dataset, build_index_permutation(dataset, shuffle_seed))

    return DataLoader(dataset, shuffle=False, persistent_workers=persistent_workers, **dataloader_options)


def compute_metrics(rows):
    refs = [row["ref_text"] for row in rows]
    preds = [row["pred_text"] for row in rows]
    metrics = {}
    if refs:
        global_bleu = bleu(refs, preds, level="word")
        metrics["bleu_1"] = float(global_bleu["bleu1"])
        metrics["bleu_2"] = float(global_bleu["bleu2"])
        metrics["bleu_3"] = float(global_bleu["bleu3"])
        metrics["bleu_4"] = float(global_bleu["bleu4"])
        metrics["rouge_l"] = float(rouge(refs, preds, level="word"))
    else:
        metrics["bleu_1"] = metrics["bleu_2"] = metrics["bleu_3"] = metrics["bleu_4"] = metrics["rouge_l"] = 0.0

    src_groups = defaultdict(list)
    for row in rows:
        src_groups[row["src"]].append(row)

    src_bleu4_vals = []
    for src_name in ["how2sign", "csl", "phoenix"]:
        group = src_groups.get(src_name, [])
        if not group:
            metrics[f"{src_name}_bleu_4"] = 0.0
            metrics[f"{src_name}_rouge_l"] = 0.0
            continue
        level = "char" if src_name == "csl" else "word"
        src_refs = [row["ref_text"] for row in group]
        src_preds = [row["pred_text"] for row in group]
        src_bleu = bleu(src_refs, src_preds, level=level)
        src_rouge = rouge(src_refs, src_preds, level=level)
        metrics[f"{src_name}_bleu_4"] = float(src_bleu["bleu4"])
        metrics[f"{src_name}_rouge_l"] = float(src_rouge)
        src_bleu4_vals.append(float(src_bleu["bleu4"]))
    metrics["mean_src_bleu_4"] = float(sum(src_bleu4_vals) / len(src_bleu4_vals)) if src_bleu4_vals else 0.0
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate m2t predictions with optional motion ablations.")
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_gpus", type=str, default="0")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--motion_mode", type=str, default="normal", choices=["normal", "shuffle_by_src"])
    parser.add_argument("--shuffle_seed", type=int, default=1234)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, default="")
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    cfg.DEBUG = False
    cfg.model.params.task = "m2t"
    cfg.TRAIN.RESUME = ""
    cfg.TRAIN.PRETRAINED = ""
    cfg.TEST.CHECKPOINTS = args.checkpoint
    cfg.TEST.SPLIT = args.split
    cfg.TEST.BATCH_SIZE = int(args.batch_size)
    cfg.TEST.NUM_WORKERS = int(args.num_workers)
    cfg.EVAL.BATCH_SIZE = int(args.batch_size)
    cfg.EVAL.NUM_WORKERS = int(args.num_workers)
    cfg.EVAL.VAL_SUBSET_RATIO = 1.0
    cfg.EVAL.VAL_SUBSET_MAX_SAMPLES = 0
    cfg.TEST.SAVE_PREDICTIONS = False
    cfg.METRIC.TYPE = []

    os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpus
    pl.seed_everything(cfg.SEED_VALUE)

    datamodule = build_data(cfg, phase="test")
    model = build_model(cfg, datamodule)
    if cfg.TRAIN.PRETRAINED_VAE:
        load_pretrained_vae(cfg, model, logger=None)
    load_pretrained(cfg, model, logger=None, phase="test")
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda(args.device)

    loader = make_loader(
        datamodule=datamodule,
        split=args.split,
        motion_mode=args.motion_mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle_seed=args.shuffle_seed,
    )

    rows = []
    loss_sum = 0.0
    sample_count = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Evaluating[{args.motion_mode}]"):
            if torch.cuda.is_available():
                batch["motion"] = batch["motion"].cuda(args.device)
            out = model.train_lm_forward(batch, forced_task="m2t")["outputs"]
            cur_bs = len(batch["text"])
            loss_sum += float(out.loss.detach().cpu()) * cur_bs
            sample_count += cur_bs

            rs = model.val_m2t_forward(batch)
            for i in range(cur_bs):
                rows.append({
                    "name": batch["name"][i],
                    "src": batch["src"][i],
                    "length": int(rs["length"][i]),
                    "ref_text": rs["t_ref"][i],
                    "pred_text": rs["t_pred"][i],
                })

    mean_loss = loss_sum / max(sample_count, 1)
    metrics = compute_metrics(rows)
    metrics.update({
        "split": args.split,
        "motion_mode": args.motion_mode,
        "num_samples": int(sample_count),
        "m2t_loss": float(mean_loss),
        "m2t_ppl": float(torch.exp(torch.clamp(torch.tensor(mean_loss), max=20.0)).item()),
    })

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.output_jsonl:
        output_jsonl = Path(args.output_jsonl)
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with output_jsonl.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
