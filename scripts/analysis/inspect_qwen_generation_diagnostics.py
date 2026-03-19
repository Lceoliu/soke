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
from mGPT.archs.task_formatting import serialize_sign_tokens


def _build_cli_cfg(args):
    sys.argv = [
        "inspect_qwen_generation_diagnostics.py",
        "--cfg",
        args.cfg,
        "--use_gpus",
        args.use_gpus,
        "--device",
        str(args.device),
        "--batch_size",
        str(args.batch_size),
        "--task",
        args.task,
        "--nodebug",
    ]
    cfg = parse_args(phase="test")
    cfg.model.params.task = args.task
    cfg.TEST.CHECKPOINTS = args.ckpt
    cfg.TEST.SPLIT = args.split
    cfg.TEST.BATCH_SIZE = args.batch_size
    cfg.EVAL.BATCH_SIZE = args.batch_size
    cfg.TEST.SAVE_PREDICTIONS = False
    cfg.METRIC.TYPE = []
    return cfg


def _motion_tokens_from_batch(model, batch, idx):
    motion = batch["motion"][idx:idx + 1]
    return model._encode_sign_tokens_from_motion(motion)


def _serialize_parsed_sign(parsed_output):
    body = parsed_output["body"]
    lhand = parsed_output.get("lhand", None)
    rhand = parsed_output.get("rhand", None)
    return serialize_sign_tokens(body, lhand, rhand)


def main():
    parser = argparse.ArgumentParser(description="Inspect Qwen generation masks and parse/stop behavior.")
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--task", type=str, default="t2m", choices=["t2m", "m2t", "mc"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_examples", type=int, default=12)
    parser.add_argument("--output_jsonl", type=str, default="")
    parser.add_argument("--compare_unmasked", action="store_true")
    parser.add_argument("--use_gpus", type=str, default="0")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    cfg = _build_cli_cfg(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpus
    pl.seed_everything(cfg.SEED_VALUE)

    datamodule = build_data(cfg)
    datamodule.setup(args.split)
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
            if args.task == "t2m":
                task_texts = batch["text"]
                dbg_masked = model.lm.generate_conditional(
                    texts=task_texts,
                    task="t2m",
                    do_sample=False,
                    apply_generation_mask=True,
                    return_debug=True,
                )
                dbg_unmasked = None
                if args.compare_unmasked:
                    dbg_unmasked = model.lm.generate_conditional(
                        texts=task_texts,
                        task="t2m",
                        do_sample=False,
                        apply_generation_mask=False,
                        return_debug=True,
                    )
                for i in range(len(task_texts)):
                    row = {
                        "name": batch["name"][i],
                        "src": batch["src"][i],
                        "task": "t2m",
                        "prompt": task_texts[i],
                        "masked": {
                            "parsed": _serialize_parsed_sign(dbg_masked["debug_rows"][i]["parsed_output"]),
                            "raw_tail_text": dbg_masked["debug_rows"][i]["raw_tail_text"],
                            "stop_token_pos": dbg_masked["debug_rows"][i]["stop_token_pos"],
                            "disallowed_tail_ids": dbg_masked["debug_rows"][i]["disallowed_tail_ids"],
                            "tail_ids": dbg_masked["debug_rows"][i]["tail_ids"],
                        },
                    }
                    if args.compare_unmasked and dbg_unmasked is not None:
                        row["unmasked"] = {
                            "parsed": _serialize_parsed_sign(dbg_unmasked["debug_rows"][i]["parsed_output"]),
                            "raw_tail_text": dbg_unmasked["debug_rows"][i]["raw_tail_text"],
                            "stop_token_pos": dbg_unmasked["debug_rows"][i]["stop_token_pos"],
                            "disallowed_tail_ids": dbg_unmasked["debug_rows"][i]["disallowed_tail_ids"],
                            "tail_ids": dbg_unmasked["debug_rows"][i]["tail_ids"],
                        }
                    rows.append(row)
                    seen += 1
                    print("=" * 100)
                    print(f"[{seen}] task=t2m src={row['src']} name={row['name']}")
                    print(f"PROMPT : {row['prompt']}")
                    print(f"MASKED : {row['masked']['parsed']}")
                    print(f"RAW    : {row['masked']['raw_tail_text']}")
                    print(f"STOP   : {row['masked']['stop_token_pos']}")
                    print(f"BAD IDS: {len(row['masked']['disallowed_tail_ids'])}")
                    if args.compare_unmasked and "unmasked" in row:
                        print(f"UNMASK : {row['unmasked']['parsed']}")
                        print(f"UNRAW  : {row['unmasked']['raw_tail_text']}")
                    if seen >= args.num_examples:
                        break
            else:
                motion_tokens = [_motion_tokens_from_batch(model, batch, i) for i in range(len(batch["name"]))]
                dbg_masked = model.lm.generate_conditional(
                    motion_tokens=motion_tokens,
                    task=args.task,
                    do_sample=False,
                    apply_generation_mask=True,
                    return_debug=True,
                )
                dbg_unmasked = None
                if args.compare_unmasked:
                    dbg_unmasked = model.lm.generate_conditional(
                        motion_tokens=motion_tokens,
                        task=args.task,
                        do_sample=False,
                        apply_generation_mask=False,
                        return_debug=True,
                    )
                for i in range(len(batch["name"])):
                    row = {
                        "name": batch["name"][i],
                        "src": batch["src"][i],
                        "task": args.task,
                        "prompt_motion_len": int(len(motion_tokens[i])),
                        "gt": batch["text"][i] if "text" in batch else "",
                        "masked": dbg_masked["debug_rows"][i],
                    }
                    if args.compare_unmasked and dbg_unmasked is not None:
                        row["unmasked"] = dbg_unmasked["debug_rows"][i]
                    rows.append(row)
                    seen += 1
                    print("=" * 100)
                    print(f"[{seen}] task={args.task} src={row['src']} name={row['name']}")
                    print(f"GT     : {row['gt']}")
                    print(f"MASKED : {row['masked']['parsed_output']}")
                    print(f"RAW    : {row['masked']['raw_tail_text']}")
                    print(f"STOP   : {row['masked']['stop_token_pos']}")
                    print(f"BAD IDS: {len(row['masked']['disallowed_tail_ids'])}")
                    if args.compare_unmasked and "unmasked" in row:
                        print(f"UNMASK : {row['unmasked']['parsed_output']}")
                        print(f"UNRAW  : {row['unmasked']['raw_tail_text']}")
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
