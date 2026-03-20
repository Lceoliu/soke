#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from mGPT.config import parse_args
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.utils.load_checkpoint import load_pretrained, load_pretrained_vae


def _set_sys_argv(cfg_path: str, use_gpus: str, device: int, batch_size: int):
    sys.argv = [
        "analyze_first_token_probs.py",
        "--cfg",
        cfg_path,
        "--use_gpus",
        use_gpus,
        "--device",
        str(device),
        "--batch_size",
        str(batch_size),
        "--task",
        "m2t",
        "--nodebug",
    ]


def main():
    parser = argparse.ArgumentParser(description="Inspect first-token probabilities for m2t.")
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--num_examples", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_gpus", default="0")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output_jsonl", default="")
    args = parser.parse_args()

    _set_sys_argv(args.cfg, args.use_gpus, args.device, args.batch_size)
    cfg = parse_args(phase="test")
    cfg.model.params.task = "m2t"
    cfg.TEST.CHECKPOINTS = args.ckpt
    cfg.TEST.SPLIT = args.split
    cfg.TEST.BATCH_SIZE = args.batch_size
    cfg.EVAL.BATCH_SIZE = args.batch_size
    cfg.TEST.SAVE_PREDICTIONS = False
    cfg.METRIC.TYPE = []
    cfg.EVAL.DISABLE_VAL = True

    os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpus
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    pl.seed_everything(cfg.SEED_VALUE)

    datamodule = build_data(cfg, phase=args.split)
    if args.split == "train":
        datamodule.setup("fit")
        loader = datamodule.train_dataloader()
    elif args.split == "val":
        datamodule.setup("fit")
        loader = datamodule.val_dataloader()
    else:
        datamodule.setup("test")
        loader = datamodule.test_dataloader()

    model = build_model(cfg, datamodule)
    if cfg.TRAIN.PRETRAINED_VAE:
        load_pretrained_vae(cfg, model, logger=None)
    load_pretrained(cfg, model, logger=None, phase="test")
    model.eval()
    model.to(torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"))

    rows = []
    with torch.no_grad():
        for batch in loader:
            texts = list(batch["text"])
            names = list(batch["name"])
            srcs = list(batch["src"])
            lengths = [int(x) for x in batch["length"]]
            motion = batch["motion"].long().to(model.device)

            sign_token_ids = []
            for i in range(len(texts)):
                body, lhand, rhand = model.lm._split_motion_sample(motion[i], lengths[i])
                sign_token_ids.append(model.lm._build_sign_token_ids(body, lhand, rhand))

            packed = model.lm.task_formatter.build_batch(
                task_names=["m2t"] * len(texts),
                texts=texts,
                sign_token_ids=sign_token_ids,
                mc_prefix_ratio=getattr(model, "mc_prefix_ratio", 0.5),
                mc_group_size=getattr(model.lm, "num_token_parts", 1),
            )

            input_ids = packed.input_ids.to(model.device)
            attention_mask = packed.attention_mask.to(model.device)
            labels = packed.labels.to(model.device)
            outputs = model.lm.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            logits = outputs.logits.float()

            for i in range(input_ids.shape[0]):
                lab = labels[i]
                valid_pos = (lab != -100).nonzero(as_tuple=False).flatten()
                if len(valid_pos) == 0:
                    continue
                first_pos = int(valid_pos[0].item())
                dist = F.softmax(logits[i, first_pos - 1], dim=-1)
                gt_id = int(lab[first_pos].item())
                gt_prob = float(dist[gt_id].item())
                gt_rank = int((dist > dist[gt_id]).sum().item()) + 1

                topk_prob, topk_id = torch.topk(dist, k=5)
                topk_id = topk_id.tolist()
                topk_prob = topk_prob.tolist()
                pred_id = int(topk_id[0])
                row = {
                    "name": names[i],
                    "src": srcs[i],
                    "gt_text": texts[i],
                    "first_token_id_gt": gt_id,
                    "first_token_text_gt": model.lm.tokenizer.decode([gt_id]),
                    "first_token_prob_gt": gt_prob,
                    "first_token_rank_gt": gt_rank,
                    "first_token_id_pred": pred_id,
                    "first_token_text_pred": model.lm.tokenizer.decode([pred_id]),
                    "first_token_prob_pred": float(topk_prob[0]),
                    "top5": [
                        {
                            "id": int(tok_id),
                            "text": model.lm.tokenizer.decode([int(tok_id)]),
                            "prob": float(prob),
                        }
                        for tok_id, prob in zip(topk_id, topk_prob)
                    ],
                }
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False))
                if len(rows) >= args.num_examples:
                    break
            if len(rows) >= args.num_examples:
                break

    if not rows:
        raise RuntimeError("No rows produced.")

    output_jsonl = args.output_jsonl
    if not output_jsonl:
        output_jsonl = str(Path(args.ckpt).resolve().parent.parent / "first_token_probs" / f"first_token_probs_{args.split}.jsonl")
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
