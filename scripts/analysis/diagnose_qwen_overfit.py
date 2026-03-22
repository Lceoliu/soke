#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pytorch_lightning as pl
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from mGPT.config import parse_args
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.utils.load_checkpoint import load_pretrained, load_pretrained_vae


def _set_sys_argv(cfg_path: str, use_gpus: str, device: int, batch_size: int, task: str):
    argv = [
        "diagnose_qwen_overfit.py",
        "--cfg",
        cfg_path,
        "--use_gpus",
        use_gpus,
        "--device",
        str(device),
        "--batch_size",
        str(batch_size),
        "--nodebug",
    ]
    if task != "auto":
        argv.extend(["--task", task])
    sys.argv = argv


def _strip_prompt_and_stop(
    output_ids: Sequence[int],
    prompt_len: int,
    stop_id: int,
) -> List[int]:
    tail = []
    for tok in list(output_ids)[prompt_len:]:
        if int(tok) == int(stop_id):
            break
        tail.append(int(tok))
    return tail


def _first_divergence(pred: Sequence[int], tgt: Sequence[int]) -> int:
    min_len = min(len(pred), len(tgt))
    for idx in range(min_len):
        if int(pred[idx]) != int(tgt[idx]):
            return idx
    if len(pred) == len(tgt):
        return -1
    return min_len


def _token_acc_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[float, int]:
    """
    CausalLM logits: [B, L, V]
    Labels are aligned to input positions; loss is computed on shifted positions.
    """
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError(f"Unexpected shapes: logits={tuple(logits.shape)} labels={tuple(labels.shape)}")
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    preds = shift_logits.argmax(dim=-1)
    mask = shift_labels != -100
    valid = int(mask.sum().item())
    if valid == 0:
        return 0.0, 0
    acc = (preds[mask] == shift_labels[mask]).float().mean().item()
    return float(acc), valid


def _format_sample_tokens(model, motion_sample: torch.Tensor, length: int) -> List[int]:
    if not hasattr(model, "lm"):
        raise AttributeError("Expected model.lm to exist for Qwen diagnostics.")
    body, lhand, rhand = model.lm._split_motion_sample(motion_sample, int(length))
    return model.lm._build_sign_token_ids(body, lhand, rhand)


def _build_task_batch(model, task: str, texts: List[str], motion_batch: torch.Tensor, lengths: Sequence[int]):
    task_names = [task] * len(texts)
    sign_token_ids = []
    for i in range(len(texts)):
        sign_token_ids.append(_format_sample_tokens(model, motion_batch[i], lengths[i]))
    batch = model.lm.task_formatter.build_batch(
        task_names=task_names,
        texts=texts,
        sign_token_ids=sign_token_ids,
        mc_prefix_ratio=getattr(model, "mc_prefix_ratio", 0.5),
        mc_group_size=getattr(model.lm, "num_token_parts", 1),
    )
    return batch, sign_token_ids


def _teacher_forced_metrics(model, task: str, texts: List[str], motion_batch: torch.Tensor, lengths: Sequence[int]):
    batch, sign_token_ids = _build_task_batch(model, task, texts, motion_batch, lengths)
    input_ids = batch.input_ids.to(model.device)
    attention_mask = batch.attention_mask.to(model.device)
    labels = batch.labels.to(model.device)
    if hasattr(model, "language_model"):
        outputs = model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
    else:
        outputs = model.lm.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
    logits = outputs.logits.detach().float().cpu()
    labels_cpu = labels.detach().cpu()
    acc, valid_tokens = _token_acc_from_logits(logits, labels_cpu)

    per_sample = []
    shift_logits = logits[:, :-1, :]
    shift_labels = labels_cpu[:, 1:]
    preds = shift_logits.argmax(dim=-1)
    for i in range(labels_cpu.shape[0]):
        mask = shift_labels[i] != -100
        sample_valid = int(mask.sum().item())
        sample_acc = 0.0
        if sample_valid > 0:
            sample_acc = (preds[i][mask] == shift_labels[i][mask]).float().mean().item()
        tgt_with_eos = [int(x) for x in labels_cpu[i].tolist() if int(x) != -100]
        tgt_no_eos = tgt_with_eos[:-1] if len(tgt_with_eos) > 0 else []
        per_sample.append(
            {
                "teacher_forced_token_acc": float(sample_acc),
                "teacher_forced_valid_tokens": sample_valid,
                "target_ids_with_eos": tgt_with_eos,
                "target_ids_no_eos": tgt_no_eos,
            }
        )

    return {
        "batch_acc": float(acc),
        "batch_valid_tokens": valid_tokens,
        "per_sample": per_sample,
        "sign_token_ids": sign_token_ids,
        "batch_labels": labels_cpu,
    }


def _free_run_metrics(model, task: str, texts: List[str], motion_batch: torch.Tensor, lengths: Sequence[int]):
    if task == "t2m":
        gen_outputs = model.lm.generate_direct(texts=texts, do_sample=False)
        _, pred_texts = gen_outputs
        pred_texts = list(pred_texts)
        pred_ids_list = [
            model.lm.tokenizer(t, add_special_tokens=False).input_ids for t in pred_texts
        ]
    elif task == "m2t":
        motion_tokens = [motion_batch[i] for i in range(len(texts))]
        pred_texts = model.lm.generate_conditional(
            motion_tokens=motion_tokens,
            lengths=lengths,
            task="m2t",
            stage="test",
            do_sample=False,
        )
        pred_ids_list = [
            model.lm.tokenizer(t, add_special_tokens=False).input_ids for t in pred_texts
        ]
    else:
        raise NotImplementedError(f"Unsupported task for free-run diagnostics: {task}")

    return pred_texts, pred_ids_list


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose Qwen overfit behavior with teacher-forced and free-run metrics."
    )
    parser.add_argument("--cfg", required=True, help="Config yaml")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Dataset split to inspect")
    parser.add_argument("--task", default="auto", choices=["auto", "t2m", "m2t"], help="Task to diagnose")
    parser.add_argument("--num_examples", type=int, default=12, help="Number of samples to inspect")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for diagnosis")
    parser.add_argument("--use_gpus", default="0", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--device", type=int, default=0, help="Local visible CUDA device index")
    parser.add_argument("--output_dir", default="", help="Optional output dir override")
    args = parser.parse_args()

    task = args.task
    _set_sys_argv(args.cfg, args.use_gpus, args.device, args.batch_size, task)
    cfg = parse_args(phase="test")
    if task != "auto":
        cfg.model.params.task = str(task).lower()
    else:
        cfg.model.params.task = str(cfg.model.params.task).lower()

    if "lm" in str(cfg.TRAIN.STAGE) and args.split != "train":
        raise ValueError(
            "This diagnostic script expects split=train for LM-stage overfit diagnosis, "
            "because the current LM train split is the one that carries code tokens."
        )
    cfg.TEST.CHECKPOINTS = args.ckpt
    cfg.TEST.SPLIT = args.split
    cfg.TEST.BATCH_SIZE = int(args.batch_size)
    cfg.TEST.SAVE_PREDICTIONS = False
    cfg.EVAL.BATCH_SIZE = int(args.batch_size)
    cfg.EVAL.DISABLE_VAL = True
    cfg.METRIC.TYPE = []

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

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.ckpt).resolve().parent.parent / "diagnostics" / f"{cfg.model.params.task}_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "diagnostics.jsonl"
    summary_path = out_dir / "summary.txt"
    summary_json_path = out_dir / "summary.json"

    rows: List[Dict] = []
    total_teacher_acc = []
    total_valid_tokens = 0
    exact_match_count = 0
    divergence_points = []
    num_seen = 0

    with torch.no_grad():
        for batch in loader:
            if num_seen >= args.num_examples:
                break
            texts = list(batch["text"])
            names = list(batch["name"])
            srcs = list(batch["src"])
            lengths = [int(x) for x in batch["length"]]
            motion_batch = batch["motion"].long()
            motion_batch = motion_batch.to(model.device)

            cur_task = str(cfg.model.params.task).lower()
            tf = _teacher_forced_metrics(model, cur_task, texts, motion_batch, lengths)
            pred_texts, pred_ids_list = _free_run_metrics(model, cur_task, texts, motion_batch, lengths)

            for i in range(len(texts)):
                if num_seen >= args.num_examples:
                    break
                per = tf["per_sample"][i]
                target_no_eos = per["target_ids_no_eos"]
                pred_ids = [int(x) for x in pred_ids_list[i]]
                exact_match = pred_ids == target_no_eos
                div_idx = _first_divergence(pred_ids, target_no_eos)
                if exact_match:
                    exact_match_count += 1
                else:
                    divergence_points.append(div_idx)
                total_teacher_acc.append(per["teacher_forced_token_acc"])
                total_valid_tokens += int(per["teacher_forced_valid_tokens"])
                row = {
                    "name": names[i],
                    "src": srcs[i],
                    "task": cur_task,
                    "text": texts[i],
                    "teacher_forced_token_acc": per["teacher_forced_token_acc"],
                    "teacher_forced_valid_tokens": per["teacher_forced_valid_tokens"],
                    "free_run_exact_match": exact_match,
                    "first_divergence_index": div_idx,
                    "target_ids_with_eos": per["target_ids_with_eos"],
                    "target_ids_no_eos": target_no_eos,
                    "pred_ids": pred_ids,
                    "target_tokens_no_eos": model.lm.tokenizer.convert_ids_to_tokens(target_no_eos),
                    "pred_tokens": model.lm.tokenizer.convert_ids_to_tokens(pred_ids),
                    "pred_text": pred_texts[i],
                    "target_text": texts[i],
                    "target_len_no_eos": len(target_no_eos),
                    "pred_len": len(pred_ids),
                }
                rows.append(row)
                num_seen += 1
                print("=" * 80)
                print(f"[{num_seen}] task={cur_task} src={srcs[i]} name={names[i]}")
                print(f"TF acc: {row['teacher_forced_token_acc']:.4f}  valid={row['teacher_forced_valid_tokens']}")
                print(f"GT ids : {row['target_ids_no_eos']}")
                print(f"PR ids : {row['pred_ids']}")
                print(f"GT txt : {row['target_text']}")
                print(f"PR txt : {row['pred_text']}")
                print(f"EM     : {row['free_run_exact_match']}  div={row['first_divergence_index']}")

    if not rows:
        raise RuntimeError("No samples were processed. Check split/dataset configuration.")

    tf_mean = float(np.mean(total_teacher_acc)) if total_teacher_acc else 0.0
    tf_median = float(np.median(total_teacher_acc)) if total_teacher_acc else 0.0
    em_rate = float(exact_match_count / len(rows))
    div_mean = float(np.mean(divergence_points)) if divergence_points else -1.0
    div_median = float(np.median(divergence_points)) if divergence_points else -1.0
    summary = {
        "task": cfg.model.params.task,
        "split": args.split,
        "num_examples": len(rows),
        "teacher_forced_token_acc_mean": tf_mean,
        "teacher_forced_token_acc_median": tf_median,
        "teacher_forced_valid_tokens_total": int(total_valid_tokens),
        "free_run_exact_match_rate": em_rate,
        "free_run_exact_match_count": int(exact_match_count),
        "free_run_first_divergence_mean": div_mean,
        "free_run_first_divergence_median": div_median,
        "output_jsonl": str(jsonl_path),
    }

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(summary_path, "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"saved {jsonl_path}")
    print(f"saved {summary_path}")


if __name__ == "__main__":
    main()
