#!/usr/bin/env python3
"""Export prompt parity diagnostics for Qwen t2m / m2t.

This script compares:
1) the training sample sequence built from the raw sign/text example, and
2) the inference prompt sequence used for generation.

It writes:
  - JSONL with per-sample details
  - Markdown summary report
  - plain-text summary

The script is intentionally analysis-only and does not touch training code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from omegaconf import OmegaConf
from transformers import AutoTokenizer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from mGPT.archs.task_formatting import SignLanguageTaskFormatter, add_missing_special_tokens, serialize_sign_tokens
from mGPT.data.build_data import build_data
from mGPT.data.humanml.load_data import load_csl_sample, load_h2s_sample, load_phoenix_sample
from mGPT.config import get_module_config


def parse_args():
    parser = argparse.ArgumentParser(description="Export Qwen prompt parity diagnostics.")
    parser.add_argument("--cfg", required=True, help="Training config used to build datasets and tokenizer.")
    parser.add_argument("--cfg_assets", default="configs/assets.yaml")
    parser.add_argument("--split", default="train", choices=["train"], help="Only train split is supported.")
    parser.add_argument("--tasks", default="t2m,m2t", help="Comma-separated tasks to diagnose.")
    parser.add_argument("--num_examples", type=int, default=12)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--output_prefix", default="")
    parser.add_argument("--strict", action="store_true", help="Fail fast on malformed samples.")
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


def resolve_path(root: str, maybe_relative: str) -> str:
    p = Path(maybe_relative)
    if p.is_absolute():
        return str(p)
    return str(Path(root) / p)


def build_tokenizer(cfg):
    model_path = resolve_path(ROOT_DIR, str(cfg.model.params.lm.params.model_path))
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    add_missing_special_tokens(tokenizer)
    motion_codebook_size = int(cfg.model.params.motion_vae.params.code_num)
    hand_codebook_size = int(cfg.model.params.hand_vae_cfg.params.code_num)
    rhand_codebook_size = int(cfg.model.params.rhand_vae_cfg.params.code_num)

    motion_tokens = [f"<motion_id_{i}>" for i in range(motion_codebook_size + 3)]
    hand_tokens = [f"<hand_id_{i}>" for i in range(hand_codebook_size + 3)]
    rhand_tokens = [f"<rhand_id_{i}>" for i in range(rhand_codebook_size + 3)]
    tokenizer.add_tokens(motion_tokens + hand_tokens + rhand_tokens)
    return tokenizer


def load_training_samples(datamodule, limit: int):
    dataset = datamodule.train_dataset
    samples = []
    for ann in dataset.all_data:
        if len(samples) >= limit:
            break
        src = str(ann.get("src", "")).lower()
        if src == "how2sign":
            clip_poses, text, name, code = load_h2s_sample(
                ann,
                dataset.data_dir,
                need_pose=False,
                code_path=os.path.join(dataset.data_root, dataset.code_path),
                need_code=True,
            )
        elif src == "csl":
            clip_poses, text, name, code = load_csl_sample(
                ann,
                dataset.csl_root,
                need_pose=False,
                code_path=os.path.join(dataset.data_root, dataset.code_path),
                need_code=True,
            )
        elif src == "phoenix":
            clip_poses, text, name, code = load_phoenix_sample(
                ann,
                dataset.phoenix_root,
                need_pose=False,
                code_path=os.path.join(dataset.data_root, dataset.code_path),
                need_code=True,
            )
        else:
            continue

        if text is None or code is None:
            continue
        samples.append(
            {
                "name": name,
                "src": src,
                "text": text,
                "code": code,
                "ann": ann,
            }
        )
    return samples


def normalize_code_array(code):
    arr = np.asarray(code)
    if arr.ndim == 0:
        raise ValueError(f"Unsupported code shape: {arr.shape}")
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = arr.reshape(-1, 3)
    return arr


def split_code_to_parts(code) -> Tuple[List[int], Optional[List[int]], Optional[List[int]]]:
    arr = normalize_code_array(code)
    if arr.ndim == 1:
        return arr.astype(int).tolist(), None, None
    if arr.ndim != 2:
        raise ValueError(f"Unsupported motion token shape after normalization: {arr.shape}")
    if arr.shape[1] == 1:
        return arr[:, 0].astype(int).tolist(), None, None
    if arr.shape[1] == 2:
        return arr[:, 0].astype(int).tolist(), arr[:, 1].astype(int).tolist(), None
    if arr.shape[1] >= 3:
        return (
            arr[:, 0].astype(int).tolist(),
            arr[:, 1].astype(int).tolist(),
            arr[:, 2].astype(int).tolist(),
        )
    raise ValueError(f"Unsupported motion token shape after normalization: {arr.shape}")


def convert_ids_to_tokens(tokenizer, ids: Sequence[int]) -> List[str]:
    return [tokenizer.convert_ids_to_tokens(int(x)) for x in ids]


def decode_ids(tokenizer, ids: Sequence[int]) -> str:
    return tokenizer.decode(list(map(int, ids)), skip_special_tokens=False).strip()


def first_mismatch(a: Sequence[int], b: Sequence[int]) -> Optional[Dict[str, int]]:
    n = min(len(a), len(b))
    for idx in range(n):
        if int(a[idx]) != int(b[idx]):
            return {"index": idx, "left": int(a[idx]), "right": int(b[idx])}
    if len(a) != len(b):
        return {"index": n, "left": len(a), "right": len(b)}
    return None


def build_train_and_prompt(task: str, text: str, sign_ids: Sequence[int], tokenizer, formatter):
    task = str(task).lower()
    if task == "t2m":
        train = formatter.build_batch([task], [text], [list(sign_ids)])
        prompt = tokenizer(
            f"<t2m> <text> {text} </text> <sign>",
            add_special_tokens=False,
        ).input_ids
        return train, prompt

    if task == "m2t":
        train = formatter.build_batch([task], [text], [list(sign_ids)])
        prompt = [
            tokenizer.convert_tokens_to_ids("<m2t>"),
            tokenizer.convert_tokens_to_ids("<sign>"),
            *list(map(int, sign_ids)),
            tokenizer.convert_tokens_to_ids("</sign>"),
            tokenizer.convert_tokens_to_ids("<text>"),
        ]
        return train, prompt

    raise NotImplementedError(f"Unsupported task for parity export: {task}")


def get_expected_tokens(task: str):
    task = str(task).lower()
    if task == "t2m":
        return {
            "train_required": ["<t2m>", "<text>", "</text>", "<sign>", "</sign>"],
            "prompt_required": ["<t2m>", "<text>", "</text>", "<sign>"],
            "train_forbidden": ["<m2t>", "<mc>", "<cont>", "</cont>"],
            "prompt_forbidden": ["<m2t>", "<mc>", "<cont>", "</cont>", "</sign>", "</text>"],
        }
    if task == "m2t":
        return {
            "train_required": ["<m2t>", "<sign>", "</sign>", "<text>", "</text>"],
            "prompt_required": ["<m2t>", "<sign>", "</sign>", "<text>"],
            "train_forbidden": ["<t2m>", "<mc>", "<cont>", "</cont>"],
            "prompt_forbidden": ["<t2m>", "<mc>", "<cont>", "</cont>", "</text>"],
        }
    raise NotImplementedError(task)


def special_token_check(tokens: Sequence[str], required: Sequence[str], forbidden: Sequence[str]):
    present = set(tokens)
    missing = [tok for tok in required if tok not in present]
    unexpected = [tok for tok in forbidden if tok in present]
    return {
        "ok": len(missing) == 0 and len(unexpected) == 0,
        "missing": missing,
        "unexpected": unexpected,
    }


def make_output_dir(output_dir: str, output_prefix: str):
    if output_dir:
        base = Path(output_dir)
    else:
        base = Path("reports") / f"qwen_prompt_parity_{time.strftime('%Y%m%d_%H%M%S')}"
    base.mkdir(parents=True, exist_ok=True)
    prefix = output_prefix or "prompt_parity"
    return base, prefix


def main():
    args = parse_args()
    cfg = load_cfg(args.cfg, args.cfg_assets)

    tasks = [t.strip().lower() for t in args.tasks.split(",") if t.strip()]
    for task in tasks:
        if task not in {"t2m", "m2t"}:
            raise NotImplementedError(f"Only t2m and m2t are supported right now, got {task}")

    datamodule = build_data(cfg)
    tokenizer = build_tokenizer(cfg)
    formatter = SignLanguageTaskFormatter(tokenizer)

    samples = load_training_samples(datamodule, args.num_examples)
    if len(samples) == 0:
        raise RuntimeError("No training samples were loaded for prompt parity export.")

    out_dir, prefix = make_output_dir(args.output_dir, args.output_prefix)
    jsonl_path = out_dir / f"{prefix}.jsonl"
    md_path = out_dir / f"{prefix}.md"
    txt_path = out_dir / f"{prefix}.txt"

    rows = []
    summary = {}
    for task in tasks:
        summary[task] = {
            "count": 0,
            "prefix_match": 0,
            "train_special_ok": 0,
            "prompt_special_ok": 0,
            "label_mask_ok": 0,
        }

    for sample_idx, sample in enumerate(samples):
        text = sample["text"]
        code = sample["code"]
        body, lhand, rhand = split_code_to_parts(code)
        sign_str = serialize_sign_tokens(body, lhand, rhand)
        sign_ids = tokenizer(sign_str, add_special_tokens=False).input_ids

        for task in tasks:
            train_batch, prompt_ids = build_train_and_prompt(task, text, sign_ids, tokenizer, formatter)
            train_input_ids = train_batch.input_ids[0].tolist()
            train_labels = train_batch.labels[0].tolist()
            train_tokens = convert_ids_to_tokens(tokenizer, train_input_ids)
            prompt_tokens = convert_ids_to_tokens(tokenizer, prompt_ids)
            train_decode = decode_ids(tokenizer, train_input_ids)
            prompt_decode = decode_ids(tokenizer, prompt_ids)

            expected = get_expected_tokens(task)
            train_check = special_token_check(train_tokens, expected["train_required"], expected["train_forbidden"])
            prompt_check = special_token_check(prompt_tokens, expected["prompt_required"], expected["prompt_forbidden"])

            prompt_match = train_input_ids[: len(prompt_ids)] == list(prompt_ids)
            mismatch = None if prompt_match else first_mismatch(train_input_ids[: len(prompt_ids)], prompt_ids)

            num_supervised = int(sum(int(x != -100) for x in train_labels))
            if task == "t2m":
                expected_supervised = len(sign_ids) + 1
            elif task == "m2t":
                text_ids = tokenizer(text, add_special_tokens=False).input_ids
                expected_supervised = len(text_ids) + 1
            else:
                expected_supervised = num_supervised
            label_mask_ok = num_supervised == expected_supervised

            summary[task]["count"] += 1
            summary[task]["prefix_match"] += int(prompt_match)
            summary[task]["train_special_ok"] += int(train_check["ok"])
            summary[task]["prompt_special_ok"] += int(prompt_check["ok"])
            summary[task]["label_mask_ok"] += int(label_mask_ok)

            rows.append(
                {
                    "sample_index": sample_idx,
                    "task": task,
                    "name": sample["name"],
                    "src": sample["src"],
                    "text": text,
                    "sign_token_count": len(sign_ids),
                    "train_input_ids": train_input_ids,
                    "train_tokens": train_tokens,
                    "train_decode": train_decode,
                    "train_labels": train_labels,
                    "prompt_ids": list(prompt_ids),
                    "prompt_tokens": prompt_tokens,
                    "prompt_decode": prompt_decode,
                    "prompt_prefix_match": prompt_match,
                    "prompt_prefix_mismatch": mismatch,
                    "train_special_check": train_check,
                    "prompt_special_check": prompt_check,
                    "label_mask_ok": label_mask_ok,
                    "supervised_token_count": num_supervised,
                    "expected_supervised_token_count": expected_supervised,
                }
            )

            if args.strict and (not prompt_match or not train_check["ok"] or not prompt_check["ok"] or not label_mask_ok):
                raise RuntimeError(
                    f"Parity check failed for sample={sample['name']} task={task}: "
                    f"prompt_match={prompt_match}, train_special_ok={train_check['ok']}, "
                    f"prompt_special_ok={prompt_check['ok']}, label_mask_ok={label_mask_ok}"
                )

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    md_lines = []
    md_lines.append("# Qwen Prompt Parity Report")
    md_lines.append("")
    md_lines.append(f"- cfg: `{args.cfg}`")
    md_lines.append(f"- split: `{args.split}`")
    md_lines.append(f"- num_examples: `{args.num_examples}`")
    md_lines.append(f"- output_dir: `{out_dir}`")
    md_lines.append(f"- jsonl: `{jsonl_path}`")
    md_lines.append("")
    md_lines.append("## Summary")
    md_lines.append("")
    md_lines.append("| task | count | prefix_match | train_special_ok | prompt_special_ok | label_mask_ok |")
    md_lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for task in tasks:
        s = summary[task]
        md_lines.append(
            f"| {task} | {s['count']} | {s['prefix_match']} | {s['train_special_ok']} | {s['prompt_special_ok']} | {s['label_mask_ok']} |"
        )
    md_lines.append("")
    md_lines.append("## Per Sample")
    md_lines.append("")
    for row in rows:
        md_lines.append(f"### `{row['name']}` / `{row['task']}`")
        md_lines.append(f"- src: `{row['src']}`")
        md_lines.append(f"- prompt_prefix_match: `{row['prompt_prefix_match']}`")
        md_lines.append(f"- label_mask_ok: `{row['label_mask_ok']}`")
        md_lines.append(f"- train_special_ok: `{row['train_special_check']['ok']}`")
        md_lines.append(f"- prompt_special_ok: `{row['prompt_special_check']['ok']}`")
        if row["prompt_prefix_mismatch"] is not None:
            md_lines.append(f"- first_prefix_mismatch: `{row['prompt_prefix_mismatch']}`")
        md_lines.append("")
        md_lines.append("```text")
        md_lines.append(f"TRAIN IDS: {row['train_input_ids']}")
        md_lines.append(f"PROMPT IDS: {row['prompt_ids']}")
        md_lines.append(f"TRAIN TOKENS: {' '.join(row['train_tokens'])}")
        md_lines.append(f"PROMPT TOKENS: {' '.join(row['prompt_tokens'])}")
        md_lines.append(f"TRAIN DECODE: {row['train_decode']}")
        md_lines.append(f"PROMPT DECODE: {row['prompt_decode']}")
        md_lines.append("```")
        md_lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    txt_lines = []
    txt_lines.append("Qwen Prompt Parity Summary")
    txt_lines.append(f"cfg: {args.cfg}")
    txt_lines.append(f"jsonl: {jsonl_path}")
    for task in tasks:
        s = summary[task]
        total = max(s["count"], 1)
        txt_lines.append(
            f"{task}: prefix_match={s['prefix_match']}/{total}, "
            f"train_special_ok={s['train_special_ok']}/{total}, "
            f"prompt_special_ok={s['prompt_special_ok']}/{total}, "
            f"label_mask_ok={s['label_mask_ok']}/{total}"
        )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))

    print(f"saved {jsonl_path}")
    print(f"saved {md_path}")
    print(f"saved {txt_path}")


if __name__ == "__main__":
    main()
