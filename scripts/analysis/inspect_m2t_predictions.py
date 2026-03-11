import argparse
import json
import os
import sys

import pytorch_lightning as pl
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from mGPT.config import parse_args
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.utils.load_checkpoint import load_pretrained, load_pretrained_vae


def main():
    parser = argparse.ArgumentParser(description="Inspect a few m2t predictions without running full test metrics.")
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_examples", type=int, default=12)
    parser.add_argument("--output_jsonl", type=str, default="")
    parser.add_argument("--use_gpus", type=str, default="0")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    sys.argv = [
        "inspect_m2t_predictions.py",
        "--cfg",
        args.cfg,
        "--use_gpus",
        args.use_gpus,
        "--device",
        str(args.device),
        "--batch_size",
        str(args.batch_size),
        "--task",
        "m2t",
        "--nodebug",
    ]
    cfg = parse_args(phase="test")
    cfg.model.params.task = "m2t"
    cfg.TEST.CHECKPOINTS = args.ckpt
    cfg.TEST.SPLIT = args.split
    cfg.TEST.BATCH_SIZE = args.batch_size
    cfg.EVAL.BATCH_SIZE = args.batch_size
    cfg.TEST.SAVE_PREDICTIONS = False
    cfg.METRIC.TYPE = []

    os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpus
    pl.seed_everything(cfg.SEED_VALUE)

    datamodule = build_data(cfg)
    datamodule.setup("test")
    model = build_model(cfg, datamodule)
    if cfg.TRAIN.PRETRAINED_VAE:
        load_pretrained_vae(cfg, model, logger=None)
    load_pretrained(cfg, model, logger=None, phase="test")
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda(args.device)

    rows = []
    seen = 0
    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            if torch.cuda.is_available():
                batch["motion"] = batch["motion"].cuda(args.device)
            rs = model.val_m2t_forward(batch)
            cur_bs = len(rs["t_pred"])
            for i in range(cur_bs):
                row = {
                    "name": batch["name"][i],
                    "src": batch["src"][i],
                    "gt": rs["t_ref"][i],
                    "pred": rs["t_pred"][i],
                    "length": int(rs["length"][i]),
                }
                rows.append(row)
                seen += 1
                print("=" * 80)
                print(f"[{seen}] src={row['src']} name={row['name']} len={row['length']}")
                print(f"GT  : {row['gt']}")
                print(f"PRED: {row['pred']}")
                if seen >= args.num_examples:
                    break
            if seen >= args.num_examples:
                break

    if args.output_jsonl:
        os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
        with open(args.output_jsonl, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"saved {args.output_jsonl}")


if __name__ == "__main__":
    main()
