#!/usr/bin/env python3
"""Evaluate per-quantizer-layer token perplexity.

Usage:
    python scripts/eval_per_q_perplexity.py \
        --cfg configs/soke.yaml \
        --checkpoint experiments/mgpt/.../last.ckpt \
        [--max_batches 50] [--task t2m]

If the shared-embedding hypothesis holds (all Q layers share the same
embedding rows), deeper quantizer layers (q2, q3, q4) should show
significantly higher perplexity than q1, because the model cannot
distinguish which layer a token belongs to except by positional context.

With per-q vocabulary offset the gap should shrink, proving the benefit.
"""
import argparse
import math
import os
import sys

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from collections import defaultdict
from rich.table import Table
from rich import get_console

# ── project imports ──────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mGPT.config import parse_args
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.utils.logger import create_logger
from mGPT.utils.load_checkpoint import load_pretrained, load_pretrained_vae


def identify_sign_token_sets(tokenizer):
    """Return sets of token IDs for each sign stream."""
    vocab = tokenizer.get_vocab()
    motion_ids, hand_ids, rhand_ids = set(), set(), set()
    for tok, tok_id in vocab.items():
        if tok.startswith("<motion_id_"):
            motion_ids.add(tok_id)
        elif tok.startswith("<hand_id_"):
            hand_ids.add(tok_id)
        elif tok.startswith("<rhand_id_"):
            rhand_ids.add(tok_id)
    return motion_ids, hand_ids, rhand_ids


def preprocess_batch_tokens(model, batch, forced_task="t2m"):
    """Replicate train_lm_forward's token preprocessing, returning
    (motion_tokens_or_list, texts, lengths, tasks) ready for lm.forward().
    """
    has_precomputed = "motion_tokens" in batch and batch["motion_tokens"] is not None
    tokens_ref = batch["motion_tokens"] if has_precomputed else batch["motion"]
    texts = list(batch["text"])
    lengths = batch["motion_token_length"] if has_precomputed else batch["length"]
    tasks = batch["tasks"]
    all_captions = batch["all_captions"]

    expected_nfeats = int(
        getattr(model.datamodule, "nfeats",
                getattr(model.hparams.cfg.DATASET, "NFEATS", 133))
    )
    is_raw = (
        not has_precomputed
        and torch.is_tensor(tokens_ref)
        and tokens_ref.dim() == 3
        and int(tokens_ref.shape[-1]) == expected_nfeats
    )
    if is_raw:
        tokens_ref = model._build_eval_motion_tokens(tokens_ref, lengths)
    else:
        tokens_ref, lengths = model._flatten_batch_tokens_for_lm(tokens_ref, lengths)

    if model.hparams.condition == "caption":
        # Keep eval deterministic. Training may sample a random caption variant,
        # but perplexity comparison should use a stable text input.
        texts = [all_captions[i][0] if len(all_captions[i]) > 0 else texts[i] for i in range(len(texts))]
    if forced_task:
        tasks = [{"class": str(forced_task).lower()} for _ in range(len(texts))]

    return tokens_ref, texts, lengths, tasks


def lm_forward_with_labels(lm, texts, motion_tokens, lengths, tasks, src=None, name=None):
    """Run LM forward and return (logits, labels, input_ids) — all aligned.

    This replicates QwenCausalLM.forward() but returns the intermediate
    labels tensor so we can compute per-token losses outside.
    """
    task_names = []
    sign_token_ids = []
    for idx in range(len(texts)):
        task_entry = tasks[idx] if tasks is not None else None
        task_names.append(lm._resolve_task_name(task_entry, default_task="t2m"))
        body, lhand, rhand = lm._split_motion_sample(motion_tokens[idx], lengths[idx])
        sign_token_ids.append(lm._build_sign_token_ids(body, lhand, rhand))

    batch = lm.task_formatter.build_batch(
        task_names=task_names,
        texts=texts,
        sign_token_ids=sign_token_ids,
        mc_prefix_ratio=lm.mc_prefix_ratio,
        mc_group_size=lm.num_token_parts,
    )

    input_ids = batch.input_ids.to(lm.device)
    attention_mask = batch.attention_mask.to(lm.device)
    labels = batch.labels.to(lm.device)

    outputs = lm.language_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        return_dict=True,
    )
    return outputs.logits, labels, input_ids


@torch.no_grad()
def compute_per_q_perplexity(
    model,
    datamodule,
    num_quantizers: int,
    num_parts: int,
    max_batches: int = 0,
    forced_task: str = "t2m",
    split: str = "train",
):
    model.eval()
    lm = model.lm
    tokenizer = lm.tokenizer
    motion_ids, hand_ids, rhand_ids = identify_sign_token_sets(tokenizer)
    all_sign_ids = motion_ids | hand_ids | rhand_ids

    q_losses = defaultdict(list)
    part_q_losses = defaultdict(lambda: defaultdict(list))
    total_batches = 0

    split = str(split).lower()
    if split == "train":
        dataloader = datamodule.train_dataloader()
    elif split == "val":
        dataloader = datamodule.val_dataloader()
        if isinstance(dataloader, list):
            loader_idx = 0
            if hasattr(datamodule, "get_val_task_name"):
                for i in range(len(dataloader)):
                    if str(datamodule.get_val_task_name(i)).lower() == str(forced_task).lower():
                        loader_idx = i
                        break
            dataloader = dataloader[loader_idx]
    elif split == "test":
        dataloader = datamodule.test_dataloader()
        if isinstance(dataloader, list):
            dataloader = dataloader[0]
    else:
        raise ValueError(f"Unsupported split: {split}")

    for batch_idx, batch in enumerate(dataloader):
        if 0 < max_batches <= batch_idx:
            break

        # Move tensors to device
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(model.device)

        try:
            tokens_ref, texts, lengths, tasks = preprocess_batch_tokens(
                model, batch, forced_task=forced_task
            )
            logits, labels, input_ids = lm_forward_with_labels(
                lm, texts, tokens_ref, lengths, tasks,
                src=batch.get("src"), name=batch.get("name"),
            )
        except Exception as e:
            print(f"  [skip batch {batch_idx}] {e}")
            continue

        # Causal LM shift: logits[t] predicts labels[t+1]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        B, T, V = shift_logits.shape
        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(B, T).float()

        # Classify each valid label position by q-layer
        for b in range(B):
            valid_mask = shift_labels[b] != -100
            if not valid_mask.any():
                continue

            valid_pos = valid_mask.nonzero(as_tuple=True)[0]
            valid_tids = shift_labels[b][valid_pos].tolist()
            valid_loss = per_token_loss[b][valid_pos].detach().cpu().float().numpy()

            # Filter sign tokens and tag part
            sign_indices = []
            parts = []
            for j, tid in enumerate(valid_tids):
                if tid in all_sign_ids:
                    sign_indices.append(j)
                    if tid in motion_ids:
                        parts.append("body")
                    elif tid in hand_ids:
                        parts.append("lhand")
                    else:
                        parts.append("rhand")

            if not sign_indices:
                continue

            sign_loss = valid_loss[sign_indices]

            # Q-layer assignment:
            # Sign tokens are interleaved [B L R] [B L R] ...
            # local_idx // num_parts = group_idx
            # group_idx % num_quantizers = q_layer
            for local_idx, (lv, part) in enumerate(zip(sign_loss, parts)):
                group_idx = local_idx // num_parts
                q_layer = group_idx % num_quantizers
                q_losses[q_layer].append(float(lv))
                part_q_losses[part][q_layer].append(float(lv))

        total_batches += 1
        if total_batches % 10 == 0:
            # Print running stats
            summary = ", ".join(
                f"q{q+1}={math.exp(min(np.mean(q_losses[q]), 100)):.1f}"
                for q in sorted(q_losses.keys())
            )
            print(f"  [{total_batches} batches] {summary}")

    results = {}
    for q in sorted(q_losses.keys()):
        avg = np.mean(q_losses[q])
        results[q] = (math.exp(min(avg, 100)), len(q_losses[q]))

    part_results = {}
    for part in sorted(part_q_losses.keys()):
        part_results[part] = {}
        for q in sorted(part_q_losses[part].keys()):
            avg = np.mean(part_q_losses[part][q])
            part_results[part][q] = (math.exp(min(avg, 100)), len(part_q_losses[part][q]))

    return results, part_results, total_batches


def print_results(results, part_results, num_quantizers, total_batches, split):
    console = get_console()

    table = Table(title=f"Per-Quantizer Perplexity [{split}] ({total_batches} batches)")
    table.add_column("Q Layer", style="cyan", no_wrap=True)
    table.add_column("Perplexity", style="magenta", justify="right")
    table.add_column("Avg NLL", style="yellow", justify="right")
    table.add_column("Num Tokens", style="green", justify="right")

    for q in range(num_quantizers):
        if q in results:
            ppl, n = results[q]
            nll = math.log(ppl) if ppl > 0 else float("nan")
            table.add_row(f"q{q+1}", f"{ppl:.2f}", f"{nll:.4f}", f"{n:,}")
        else:
            table.add_row(f"q{q+1}", "N/A", "N/A", "0")
    console.print(table, justify="center")

    if part_results:
        table2 = Table(title="Per-Part Per-Quantizer Perplexity")
        table2.add_column("Part", style="cyan")
        table2.add_column("Q Layer", style="cyan")
        table2.add_column("Perplexity", style="magenta", justify="right")
        table2.add_column("Num Tokens", style="green", justify="right")

        for part in ["body", "lhand", "rhand"]:
            if part not in part_results:
                continue
            for q in range(num_quantizers):
                if q in part_results[part]:
                    ppl, n = part_results[part][q]
                    table2.add_row(part, f"q{q+1}", f"{ppl:.2f}", f"{n:,}")
        console.print(table2, justify="center")

    if len(results) >= 2:
        q0_ppl = results.get(0, (0, 0))[0]
        q_last = max(results.keys())
        qlast_ppl = results.get(q_last, (0, 0))[0]
        if q0_ppl > 0:
            ratio = qlast_ppl / q0_ppl
            console.print(
                f"\n  q{q_last+1}/q1 perplexity ratio: [bold]{ratio:.2f}x[/bold]"
                f"  (ratio >> 1 => shared-embedding bottleneck)",
                justify="center",
            )


def main():
    parser = argparse.ArgumentParser(description="Per-quantizer perplexity evaluation")
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max_batches", type=int, default=50)
    parser.add_argument("--task", type=str, default="t2m")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--gpus", type=str, default="0")
    args = parser.parse_args()

    task = str(args.task).lower()
    if task == "m2t":
        raise ValueError(
            "scripts/eval_per_q_perplexity.py only supports sign-target tasks "
            "(for example t2m/mc). m2t predicts text tokens, so per-q sign perplexity "
            "is not defined for that task."
        )

    sys.argv = ["eval_per_q_perplexity", "--cfg", args.cfg]

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    cfg = parse_args(phase="test")
    cfg.TEST.CHECKPOINTS = args.checkpoint
    cfg.DEVICE = [0]
    if args.batch_size > 0:
        cfg.TRAIN.BATCH_SIZE = args.batch_size
        cfg.EVAL.BATCH_SIZE = args.batch_size
        cfg.TEST.BATCH_SIZE = args.batch_size

    logger = create_logger(cfg, phase="test")
    pl.seed_everything(cfg.SEED_VALUE)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    datamodule = build_data(cfg)
    model = build_model(cfg, datamodule)

    if cfg.TRAIN.PRETRAINED_VAE:
        load_pretrained_vae(cfg, model, logger)
    load_pretrained(cfg, model, logger, phase="test")

    model = model.cuda().eval()

    num_q = int(model.lm_shared_num_quantizers)
    num_parts = int(model.lm_num_token_parts)
    print(f"\nnum_quantizers={num_q}, num_parts={num_parts}")
    print(f"body_codebook={model.body_codebook_size}, "
          f"hand_codebook={model.hand_codebook_size}, "
          f"rhand_codebook={model.rhand_codebook_size}")
    print(f"task={task}, split={args.split}, max_batches={args.max_batches}\n")

    results, part_results, total = compute_per_q_perplexity(
        model, datamodule, num_q, num_parts,
        max_batches=args.max_batches, forced_task=task, split=args.split,
    )
    print_results(results, part_results, num_q, total, args.split)


if __name__ == "__main__":
    main()
