#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mGPT.config import get_module_config  # noqa: E402
from mGPT.archs.mgpt_qwen import QwenCausalLM  # noqa: E402


PRESETS = {
    "single": [
        (["<debug_sign_A>"], "你们好！"),
        (["<debug_sign_B>"], "早上好！"),
        (["<debug_sign_C>"], "晚上好！"),
        (["<debug_sign_D>"], "再见！"),
    ],
    "compositional": [
        (
            [
                '<debug_sign_151666>',
                '<debug_sign_151668>',
                '<debug_sign_152126>',
                '<debug_sign_152103>',
                '<debug_sign_152081>',
                '<debug_sign_152169>',
                '<debug_sign_152198>',
                '<debug_sign_152311>',
                '<debug_sign_152182>',
                '<debug_sign_152247>',
                '<debug_sign_152066>',
                '<debug_sign_152289>',
                '<debug_sign_152093>',
                '<debug_sign_152295>',
                '<debug_sign_152224>',
                '<debug_sign_152229>',
                '<debug_sign_152157>',
                '<debug_sign_152104>',
                '<debug_sign_152164>',
                '<debug_sign_152168>',
                '<debug_sign_152145>',
                '<debug_sign_152312>',
                '<debug_sign_152173>',
                '<debug_sign_152185>',
                '<debug_sign_152243>',
                '<debug_sign_152213>',
                '<debug_sign_152134>',
                '<debug_sign_152126>',
                '<debug_sign_152105>',
                '<debug_sign_152083>',
                '<debug_sign_152174>',
                '<debug_sign_152120>',
                '<debug_sign_152305>',
                '<debug_sign_152099>',
                '<debug_sign_152190>',
                '<debug_sign_152203>',
                '<debug_sign_152142>',
                '<debug_sign_152145>',
                '<debug_sign_152215>',
                '<debug_sign_152202>',
                '<debug_sign_152121>',
                '<debug_sign_152187>',
                '<debug_sign_152255>',
                '<debug_sign_152203>',
                '<debug_sign_152264>',
                '<debug_sign_152188>',
                '<debug_sign_152223>',
                '<debug_sign_152224>',
                '<debug_sign_152234>',
                '<debug_sign_152175>',
                '<debug_sign_152125>',
                '<debug_sign_152274>',
                '<debug_sign_152145>',
                '<debug_sign_152195>',
                '<debug_sign_151669>',
                '<debug_sign_151670>',
            ],
            "你们好！",
        ),
        (
            [
                '<debug_sign_151666>',
                '<debug_sign_151668>',
                '<debug_sign_152310>',
                '<debug_sign_152233>',
                '<debug_sign_152109>',
                '<debug_sign_152304>',
                '<debug_sign_152184>',
                '<debug_sign_152296>',
                '<debug_sign_152270>',
                '<debug_sign_152301>',
                '<debug_sign_152152>',
                '<debug_sign_152176>',
                '<debug_sign_152236>',
                '<debug_sign_152290>',
                '<debug_sign_152128>',
                '<debug_sign_152091>',
                '<debug_sign_152152>',
                '<debug_sign_152067>',
                '<debug_sign_152128>',
                '<debug_sign_152240>',
                '<debug_sign_152125>',
                '<debug_sign_152137>',
                '<debug_sign_152224>',
                '<debug_sign_152307>',
                '<debug_sign_152155>',
                '<debug_sign_152316>',
                '<debug_sign_152128>',
                '<debug_sign_152242>',
                '<debug_sign_152183>',
                '<debug_sign_152073>',
                '<debug_sign_152089>',
                '<debug_sign_152149>',
                '<debug_sign_152151>',
                '<debug_sign_152101>',
                '<debug_sign_152160>',
                '<debug_sign_152252>',
                '<debug_sign_152203>',
                '<debug_sign_152138>',
                '<debug_sign_152186>',
                '<debug_sign_152281>',
                '<debug_sign_152086>',
                '<debug_sign_152262>',
                '<debug_sign_151669>',
                '<debug_sign_151670>',
            ],
            "早上好！",
        ),
        (
            [
                '<debug_sign_151666>',
                '<debug_sign_151668>',
                '<debug_sign_152314>',
                '<debug_sign_152232>',
                '<debug_sign_152302>',
                '<debug_sign_152319>',
                '<debug_sign_152152>',
                '<debug_sign_152245>',
                '<debug_sign_152159>',
                '<debug_sign_152067>',
                '<debug_sign_152288>',
                '<debug_sign_152250>',
                '<debug_sign_152303>',
                '<debug_sign_152114>',
                '<debug_sign_152168>',
                '<debug_sign_152121>',
                '<debug_sign_152302>',
                '<debug_sign_152226>',
                '<debug_sign_152160>',
                '<debug_sign_152127>',
                '<debug_sign_152159>',
                '<debug_sign_152066>',
                '<debug_sign_152160>',
                '<debug_sign_152114>',
                '<debug_sign_152190>',
                '<debug_sign_152106>',
                '<debug_sign_152160>',
                '<debug_sign_152123>',
                '<debug_sign_152287>',
                '<debug_sign_152163>',
                '<debug_sign_152166>',
                '<debug_sign_152119>',
                '<debug_sign_152110>',
                '<debug_sign_152109>',
                '<debug_sign_152168>',
                '<debug_sign_152312>',
                '<debug_sign_152122>',
                '<debug_sign_152247>',
                '<debug_sign_152138>',
                '<debug_sign_152237>',
                '<debug_sign_152085>',
                '<debug_sign_152167>',
                '<debug_sign_151669>',
                '<debug_sign_151670>',
            ],
            "晚上好！",
        ),
        (
            [
                '<debug_sign_151666>',
                '<debug_sign_151668>',
                '<debug_sign_152087>',
                '<debug_sign_152235>',
                '<debug_sign_152249>',
                '<debug_sign_152148>',
                '<debug_sign_152223>',
                '<debug_sign_152296>',
                '<debug_sign_152299>',
                '<debug_sign_152253>',
                '<debug_sign_152255>',
                '<debug_sign_152257>',
                '<debug_sign_152282>',
                '<debug_sign_152316>',
                '<debug_sign_152215>',
                '<debug_sign_152202>',
                '<debug_sign_152207>',
                '<debug_sign_152312>',
                '<debug_sign_152213>',
                '<debug_sign_152234>',
                '<debug_sign_152159>',
                '<debug_sign_152251>',
                '<debug_sign_152087>',
                '<debug_sign_152106>',
                '<debug_sign_152139>',
                '<debug_sign_152315>',
                '<debug_sign_152215>',
                '<debug_sign_152202>',
                '<debug_sign_152091>',
                '<debug_sign_152314>',
                '<debug_sign_152087>',
                '<debug_sign_152102>',
                '<debug_sign_152095>',
                '<debug_sign_152251>',
                '<debug_sign_152223>',
                '<debug_sign_152225>',
                '<debug_sign_152071>',
                '<debug_sign_152313>',
                '<debug_sign_152095>',
                '<debug_sign_152064>',
                '<debug_sign_152223>',
                '<debug_sign_152164>',
                '<debug_sign_151669>',
                '<debug_sign_151670>',
            ],
            "再见！",
        ),
    ],
}

CONTROL_DEBUG_TOKEN_SUFFIXES = {"151666", "151668", "151669", "151670"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Synthetic sign-prefix -> text probe for m2t pipeline."
    )
    parser.add_argument("--cfg", required=True, help="Base YAML config.")
    parser.add_argument(
        "--output_dir", required=True, help="Where to save logs and results."
    )
    parser.add_argument(
        "--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu."
    )
    parser.add_argument("--epochs", type=int, default=400, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Backbone LR.")
    parser.add_argument(
        "--embed_lr_mult", type=float, default=10.0, help="Embedding LR multiplier."
    )
    parser.add_argument(
        "--lm_head_lr_mult", type=float, default=10.0, help="LM head LR multiplier."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="AdamW weight decay."
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument(
        "--log_every", type=int, default=50, help="Epoch logging interval."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8,
        help="Greedy generation max new tokens.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="single",
        help="Synthetic prompt preset.",
    )
    parser.add_argument(
        "--strip_control_tokens",
        action="store_true",
        help="Strip debug tokens that mirror control ids 151666/151668/151669/151670.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cfg(cfg_path: str):
    cfg_assets = OmegaConf.load("./configs/assets.yaml")
    cfg_base = OmegaConf.load(f"{cfg_assets.CONFIG_FOLDER}/default.yaml")
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(cfg_path))
    if not cfg_exp.FULL_CONFIG:
        cfg_exp = get_module_config(cfg_exp, cfg_assets.CONFIG_FOLDER)
    cfg = OmegaConf.merge(cfg_exp, cfg_assets)
    return cfg


def build_model(cfg, device: torch.device, max_new_tokens: int):
    lm_cfg = OmegaConf.to_container(cfg.model.params.lm.params, resolve=True)
    lm_cfg["generation_max_new_tokens"] = int(max_new_tokens)
    model = QwenCausalLM(**lm_cfg)
    model.to(device)
    return model


def add_synthetic_tokens(
    model: QwenCausalLM, synthetic_tokens: Sequence[str]
) -> List[int]:
    tokenizer = model.tokenizer
    existing = set(tokenizer.get_vocab().keys())
    missing = [tok for tok in synthetic_tokens if tok not in existing]
    if missing:
        tokenizer.add_tokens(missing, special_tokens=True)
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.language_model.config.pad_token_id = tokenizer.pad_token_id
    token_ids = tokenizer.convert_tokens_to_ids(list(synthetic_tokens))
    if isinstance(token_ids, int):
        token_ids = [token_ids]
    unk_id = tokenizer.unk_token_id
    unresolved = [
        tok
        for tok, tok_id in zip(synthetic_tokens, token_ids)
        if tok_id is None or tok_id == unk_id
    ]
    if unresolved:
        raise ValueError(f"Failed to resolve synthetic tokens: {unresolved}")
    return [int(x) for x in token_ids]


def build_preset_examples(preset: str):
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}")
    examples = PRESETS[preset]
    token_strings = []
    for seq, _ in examples:
        token_strings.extend(seq)
    ordered_unique = list(dict.fromkeys(token_strings))
    return examples, ordered_unique


def _is_control_debug_token(token: str) -> bool:
    if not token.startswith("<debug_sign_") or not token.endswith(">"):
        return False
    suffix = token[len("<debug_sign_"):-1]
    return suffix in CONTROL_DEBUG_TOKEN_SUFFIXES


def strip_control_tokens_from_examples(
    examples: Sequence[Tuple[Sequence[str], str]]
) -> Tuple[List[Tuple[List[str], str]], List[str]]:
    cleaned_examples: List[Tuple[List[str], str]] = []
    token_strings: List[str] = []
    for seq, text in examples:
        filtered = [tok for tok in seq if not _is_control_debug_token(tok)]
        if not filtered:
            raise ValueError(f"Sequence became empty after stripping control tokens: {seq}")
        cleaned_examples.append((filtered, text))
        token_strings.extend(filtered)
    ordered_unique = list(dict.fromkeys(token_strings))
    return cleaned_examples, ordered_unique


def build_optimizer(
    model: QwenCausalLM,
    base_lr: float,
    embed_mult: float,
    head_mult: float,
    weight_decay: float,
):
    embed_params = {
        id(p)
        for p in model.language_model.get_input_embeddings().parameters()
        if p.requires_grad
    }
    lm_head_params = {
        id(p) for p in model.language_model.lm_head.parameters() if p.requires_grad
    }

    backbone: List[torch.nn.Parameter] = []
    embeds: List[torch.nn.Parameter] = []
    head: List[torch.nn.Parameter] = []

    for param in model.language_model.parameters():
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in embed_params:
            embeds.append(param)
        elif pid in lm_head_params:
            head.append(param)
        else:
            backbone.append(param)

    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": float(base_lr), "name": "backbone"})
    if embeds:
        groups.append(
            {
                "params": embeds,
                "lr": float(base_lr) * float(embed_mult),
                "name": "embed_tokens",
            }
        )
    if head:
        groups.append(
            {"params": head, "lr": float(base_lr) * float(head_mult), "name": "lm_head"}
        )
    return torch.optim.AdamW(
        groups, betas=(0.9, 0.99), weight_decay=float(weight_decay)
    )


def build_train_batch(
    model: QwenCausalLM,
    sign_ids: Sequence[Sequence[int]],
    texts: Sequence[str],
    device: torch.device,
):
    batch = model.task_formatter.build_batch(
        task_names=["m2t"] * len(texts),
        texts=list(texts),
        sign_token_ids=[list(map(int, seq)) for seq in sign_ids],
    )
    return (
        batch.input_ids.to(device),
        batch.attention_mask.to(device),
        batch.labels.to(device),
    )


def greedy_generate(model: QwenCausalLM, sign_token_ids: Sequence[int]) -> str:
    prompt_ids = model._make_m2t_prompt([int(tok_id) for tok_id in sign_token_ids]).to(
        model.device
    )
    output_ids = model._generate_ids(
        prompt_ids,
        stop_token_id=model.special_token_ids["</text>"],
        task="m2t",
        do_sample=False,
        apply_generation_mask=True,
    )
    tail_ids = model._extract_after_prompt(output_ids, prompt_ids.shape[1])
    return model._parse_generated_text(tail_ids, model.special_token_ids["</text>"])


@torch.no_grad()
def first_token_probs_for_sequence(
    model: QwenCausalLM, sign_token_ids: Sequence[int], text: str
) -> Dict[str, object]:
    prompt_ids = model._make_m2t_prompt([int(tok_id) for tok_id in sign_token_ids]).to(
        model.device
    )
    out = model.language_model(
        input_ids=prompt_ids,
        attention_mask=torch.ones_like(prompt_ids),
        return_dict=True,
    )
    probs = torch.softmax(out.logits[0, -1].float(), dim=-1)
    text_ids = model.task_formatter._tokenize_text(text)
    if not text_ids:
        raise ValueError(f"Text tokenized to empty ids: {text!r}")
    correct_id = int(text_ids[0])
    top_prob, top_id = torch.max(probs, dim=0)
    gt_prob = float(probs[correct_id].item())
    gt_rank = int((probs > probs[correct_id]).sum().item()) + 1
    top5_probs, top5_ids = torch.topk(probs, k=5)
    return {
        "correct_first_token_id": correct_id,
        "correct_first_token": model.tokenizer.decode(
            [correct_id], skip_special_tokens=False
        ),
        "correct_prob": gt_prob,
        "correct_rank": gt_rank,
        "pred_first_token_id": int(top_id.item()),
        "pred_first_token": model.tokenizer.decode(
            [int(top_id.item())], skip_special_tokens=False
        ),
        "pred_prob": float(top_prob.item()),
        "top5": [
            {
                "token_id": int(tok_id.item()),
                "token": model.tokenizer.decode(
                    [int(tok_id.item())], skip_special_tokens=False
                ),
                "prob": float(tok_prob.item()),
            }
            for tok_prob, tok_id in zip(top5_probs, top5_ids)
        ],
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(args.cfg)
    device = torch.device(args.device)
    model = build_model(cfg, device=device, max_new_tokens=args.max_new_tokens)
    model.train()

    preset_examples, synthetic_tokens = build_preset_examples(args.preset)
    if args.strip_control_tokens:
        preset_examples, synthetic_tokens = strip_control_tokens_from_examples(
            preset_examples
        )
    texts = [text for _, text in preset_examples]
    synthetic_token_ids = add_synthetic_tokens(model, synthetic_tokens)
    token_id_map = {
        tok: tok_id for tok, tok_id in zip(synthetic_tokens, synthetic_token_ids)
    }
    sign_id_sequences = [
        [token_id_map[tok] for tok in seq] for seq, _ in preset_examples
    ]

    optimizer = build_optimizer(
        model=model,
        base_lr=args.lr,
        embed_mult=args.embed_lr_mult,
        head_mult=args.lm_head_lr_mult,
        weight_decay=args.weight_decay,
    )

    input_ids, attention_mask, labels = build_train_batch(
        model, sign_id_sequences, texts, device
    )

    losses: List[Dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        out = model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        loss = out.loss
        loss.backward()
        optimizer.step()

        cur_loss = float(loss.detach().item())
        losses.append({"epoch": epoch, "loss": cur_loss})
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"epoch={epoch} loss={cur_loss:.6e}")

    model.eval()

    examples = []
    first_token_rows = []
    for (token_seq, text), sign_token_ids in zip(preset_examples, sign_id_sequences):
        pred = greedy_generate(model, sign_token_ids)
        examples.append(
            {
                "synthetic_sign_tokens": token_seq,
                "synthetic_sign_token_ids": [int(x) for x in sign_token_ids],
                "gt": text,
                "pred": pred,
            }
        )
        row = first_token_probs_for_sequence(model, sign_token_ids, text)
        row["synthetic_sign_tokens"] = token_seq
        row["synthetic_sign_token_ids"] = [int(x) for x in sign_token_ids]
        row["gt"] = text
        row["pred"] = pred
        first_token_rows.append(row)

    with open(output_dir / "loss_curve.json", "w", encoding="utf-8") as f:
        json.dump(losses, f, ensure_ascii=False, indent=2)
    with open(output_dir / "m2t_examples.jsonl", "w", encoding="utf-8") as f:
        for row in examples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(output_dir / "first_token_probs.jsonl", "w", encoding="utf-8") as f:
        for row in first_token_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "epochs": int(args.epochs),
        "device": str(device),
        "preset": args.preset,
        "strip_control_tokens": bool(args.strip_control_tokens),
        "final_loss": float(losses[-1]["loss"]) if losses else math.nan,
        "exact_match_count": int(sum(int(x["gt"] == x["pred"]) for x in examples)),
        "count": len(examples),
        "synthetic_token_ids": {
            tok: int(tok_id) for tok, tok_id in token_id_map.items()
        },
        "synthetic_pairs": preset_examples,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
