#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mGPT.config import get_module_config
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.utils.load_checkpoint import load_pretrained, load_pretrained_vae


def _load_cfg(cfg_path: str):
    cfg_assets = OmegaConf.load("./configs/assets.yaml")
    cfg_base = OmegaConf.load(f"{cfg_assets.CONFIG_FOLDER}/default.yaml")
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(cfg_path))
    if not cfg_exp.FULL_CONFIG:
        cfg_exp = get_module_config(cfg_exp, cfg_assets.CONFIG_FOLDER)
    return OmegaConf.merge(cfg_exp, cfg_assets)


def _resolve_outputs(gen_outputs):
    if isinstance(gen_outputs, dict):
        if "outputs" in gen_outputs:
            return list(gen_outputs["outputs"])
        if "output_texts" in gen_outputs:
            return list(gen_outputs["output_texts"])
    if isinstance(gen_outputs, (list, tuple)):
        return list(gen_outputs)
    raise TypeError(f"Unsupported generation output type: {type(gen_outputs)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--exp_name", default="")
    ap.add_argument("--pretrained_vae", default="")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = _load_cfg(args.cfg)
    cfg.model.params.task = "m2t"
    cfg.TEST.SPLIT = str(args.split)
    cfg.TEST.CHECKPOINTS = str(args.checkpoint)
    cfg.TEST.SAVE_PREDICTIONS = False
    cfg.TEST.REPLICATION_TIMES = 1
    cfg.EVAL.DISABLE_VAL = True
    cfg.TRAIN.RESUME = ""
    cfg.TRAIN.PRETRAINED = ""
    if args.batch_size is not None:
        cfg.TEST.BATCH_SIZE = int(args.batch_size)
        cfg.EVAL.BATCH_SIZE = int(args.batch_size)
    if args.num_workers is not None:
        cfg.TEST.NUM_WORKERS = int(args.num_workers)
        cfg.EVAL.NUM_WORKERS = int(args.num_workers)
        cfg.TRAIN.NUM_WORKERS = int(args.num_workers)
    if args.exp_name:
        cfg.NAME = str(args.exp_name)
    if args.pretrained_vae:
        cfg.TRAIN.PRETRAINED_VAE = str(args.pretrained_vae)

    datamodule = build_data(cfg, phase="test")
    model = build_model(cfg, datamodule)
    if cfg.TRAIN.PRETRAINED_VAE:
        load_pretrained_vae(cfg, model, logger=None)
    load_pretrained(cfg, model, logger=None, phase="test")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    datamodule.setup("test")
    dataloader = datamodule.test_dataloader()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with torch.inference_mode():
        for batch in dataloader:
            motion = batch["motion"].to(device)
            lengths = [int(x) for x in batch["length"]]
            names = list(batch["name"])
            refs = list(batch["text"])
            src = list(batch.get("src", ["csl"] * len(names)))
            outputs = model.lm.generate_conditional(
                motion_features=motion,
                lengths=lengths,
                task="m2t",
                stage="test",
                src=src,
                name=names,
            )
            preds = _resolve_outputs(outputs)
            for name, src_name, length, ref_text, pred_text in zip(names, src, lengths, refs, preds):
                rows.append({
                    "name": str(name),
                    "src": str(src_name),
                    "length": int(length),
                    "ref_text": str(ref_text),
                    "pred_text": str(pred_text),
                })

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"saved {len(rows)} generations to {out_path}")


if __name__ == "__main__":
    main()
