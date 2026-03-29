from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch


SPECIAL_TASK_TOKENS = [
    "<t2m>",
    "<m2t>",
    "<mc>",
    "<sign>",
    "</sign>",
    "<text>",
    "</text>",
    "<cont>",
    "</cont>",
]

VALID_SIGN_STREAMS = ("body", "lhand", "rhand")


@dataclass
class CausalTaskBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    raw_sequences: List[str]
    task_names: List[str]
    loss_weights: Optional[torch.Tensor] = None


def add_missing_special_tokens(tokenizer) -> List[str]:
    vocab = tokenizer.get_vocab()
    missing = [tok for tok in SPECIAL_TASK_TOKENS if tok not in vocab]
    if missing:
        tokenizer.add_tokens(missing, special_tokens=True)
    return missing


def normalize_sign_streams(streams: Optional[Sequence[str]] = None) -> List[str]:
    if streams is None:
        return list(VALID_SIGN_STREAMS)
    normalized: List[str] = []
    for stream in streams:
        key = str(stream).strip().lower()
        if key not in VALID_SIGN_STREAMS:
            raise ValueError(f"Unsupported sign stream: {stream}. Expected one of {VALID_SIGN_STREAMS}.")
        if key not in normalized:
            normalized.append(key)
    if not normalized:
        raise ValueError("sign_streams must contain at least one active stream.")
    return normalized


def serialize_sign_tokens(
    body_tokens: Optional[Sequence[int]] = None,
    lhand_tokens: Optional[Sequence[int]] = None,
    rhand_tokens: Optional[Sequence[int]] = None,
    sign_streams: Optional[Sequence[str]] = None,
) -> str:
    return " ".join(
        serialize_sign_token_strings(
            body_tokens=body_tokens,
            lhand_tokens=lhand_tokens,
            rhand_tokens=rhand_tokens,
            sign_streams=sign_streams,
        )
    )


def serialize_sign_token_strings(
    body_tokens: Optional[Sequence[int]] = None,
    lhand_tokens: Optional[Sequence[int]] = None,
    rhand_tokens: Optional[Sequence[int]] = None,
    sign_streams: Optional[Sequence[str]] = None,
) -> List[str]:
    active_streams = normalize_sign_streams(sign_streams)
    token_map = {
        "body": list(body_tokens) if body_tokens is not None else None,
        "lhand": list(lhand_tokens) if lhand_tokens is not None else None,
        "rhand": list(rhand_tokens) if rhand_tokens is not None else None,
    }
    missing = [stream for stream in active_streams if token_map[stream] is None]
    if missing:
        raise ValueError(f"Missing token sequences for active sign streams: {missing}")

    active_lengths = [len(token_map[stream]) for stream in active_streams]
    if len(set(active_lengths)) != 1:
        raise ValueError(f"Active sign stream lengths do not match: {dict((s, len(token_map[s])) for s in active_streams)}")
    seq_len = active_lengths[0]

    pieces: List[str] = []
    for idx in range(seq_len):
        for stream in active_streams:
            tok = int(token_map[stream][idx])
            if stream == "body":
                pieces.append(f"<motion_id_{tok}>")
            elif stream == "lhand":
                pieces.append(f"<hand_id_{tok}>")
            elif stream == "rhand":
                pieces.append(f"<rhand_id_{tok}>")
    return pieces


def sign_token_strings_to_ids(tokenizer, token_strings: Sequence[str]) -> List[int]:
    token_ids = tokenizer.convert_tokens_to_ids(list(token_strings))
    if isinstance(token_ids, int):
        token_ids = [token_ids]
    if len(token_ids) != len(token_strings):
        raise ValueError("Tokenizer returned unexpected number of token ids for sign token sequence.")

    unk_id = getattr(tokenizer, "unk_token_id", None)
    missing = [
        token for token, token_id in zip(token_strings, token_ids)
        if token_id is None or (unk_id is not None and int(token_id) == int(unk_id))
    ]
    if missing:
        raise ValueError(f"Tokenizer failed to resolve sign tokens: {missing[:8]}")
    return [int(token_id) for token_id in token_ids]


class SignLanguageTaskFormatter:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        add_missing_special_tokens(self.tokenizer)
        self.special_id_map: Dict[str, int] = {
            tok: self.tokenizer.convert_tokens_to_ids(tok) for tok in SPECIAL_TASK_TOKENS
        }

    def _tokenize_text(self, text: str) -> List[int]:
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def _token_id(self, token: str) -> int:
        return int(self.special_id_map[token])

    def _pad_and_stack(
        self,
        sequences: List[List[int]],
        labels: List[List[int]],
        raw_sequences: List[str],
        task_names: List[str],
        weights: Optional[List[Optional[List[float]]]] = None,
    ) -> CausalTaskBatch:
        if len(sequences) == 0:
            raise ValueError("Empty batch is not supported.")

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must define pad_token_id for causal task batching.")

        max_len = max(len(x) for x in sequences)
        batch = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
        lab = torch.full((len(sequences), max_len), -100, dtype=torch.long)
        attn = torch.zeros((len(sequences), max_len), dtype=torch.long)

        has_weights = weights is not None and any(w is not None for w in weights)
        wt = None
        if has_weights:
            wt = torch.ones((len(sequences), max_len), dtype=torch.float32)

        for i, (seq, cur_lab) in enumerate(zip(sequences, labels)):
            seq_len = len(seq)
            batch[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
            lab[i, :seq_len] = torch.tensor(cur_lab, dtype=torch.long)
            attn[i, :seq_len] = 1
            if has_weights and weights[i] is not None:
                wt[i, :seq_len] = torch.tensor(weights[i], dtype=torch.float32)

        return CausalTaskBatch(
            input_ids=batch,
            labels=lab,
            attention_mask=attn,
            raw_sequences=raw_sequences,
            task_names=task_names,
            loss_weights=wt,
        )

    def _build_t2m_sample(self, text: str, sign_token_ids: Sequence[int]):
        seq = [
            self._token_id("<t2m>"),
            self._token_id("<text>"),
            *self._tokenize_text(text),
            self._token_id("</text>"),
            self._token_id("<sign>"),
            *list(sign_token_ids),
            self._token_id("</sign>"),
        ]
        loss_start = len(seq) - len(sign_token_ids) - 1
        labels = [-100] * loss_start + list(sign_token_ids) + [self._token_id("</sign>")]
        raw = f"<t2m> <text> {text} </text> <sign> {' '.join(self.tokenizer.convert_ids_to_tokens(sign_token_ids))} </sign>"
        return seq, labels, None, raw

    def _build_m2t_sample(
        self, text: str, sign_token_ids: Sequence[int], prefix_loss_weight: float = 0.0,
    ):
        text_ids = self._tokenize_text(text)
        seq = [
            self._token_id("<m2t>"),
            self._token_id("<sign>"),
            *list(sign_token_ids),
            self._token_id("</sign>"),
            self._token_id("<text>"),
            *text_ids,
            self._token_id("</text>"),
        ]
        raw = f"<m2t> <sign> {' '.join(self.tokenizer.convert_ids_to_tokens(sign_token_ids))} </sign> <text> {text} </text>"

        prefix_len = len(seq) - len(text_ids) - 1
        if prefix_loss_weight > 0.0:
            # First token has no prior context to predict it → stays -100.
            # All remaining prefix tokens get real labels with lambda weight.
            labels = [-100] + list(seq[1:prefix_len]) + text_ids + [self._token_id("</text>")]
            weights = [0.0] + [prefix_loss_weight] * (prefix_len - 1) + [1.0] * (len(text_ids) + 1)
            return seq, labels, weights, raw
        else:
            labels = [-100] * prefix_len + text_ids + [self._token_id("</text>")]
            return seq, labels, None, raw

    def _build_mc_sample(
        self,
        sign_token_ids: Sequence[int],
        prefix_ratio: float = 0.5,
        min_prefix_tokens: int = 6,
        group_size: int = 1,
        random_prefix_ratio: bool = False,
        prefix_ratio_min: float = 0.3,
        prefix_ratio_max: float = 0.7,
    ):
        sign_token_ids = list(sign_token_ids)
        group_size = max(int(group_size), 1)
        if random_prefix_ratio:
            low = float(min(prefix_ratio_min, prefix_ratio_max))
            high = float(max(prefix_ratio_min, prefix_ratio_max))
            prefix_ratio = random.uniform(low, high)
        if len(sign_token_ids) < 2:
            prefix = sign_token_ids[:1]
            target = sign_token_ids[1:]
        else:
            split_idx = max(min_prefix_tokens, int(round(len(sign_token_ids) * prefix_ratio)))
            split_idx = min(max(split_idx, 1), len(sign_token_ids) - 1)
            if group_size > 1:
                split_idx = max(group_size, (split_idx // group_size) * group_size)
                if split_idx >= len(sign_token_ids):
                    split_idx = max(group_size, ((len(sign_token_ids) - 1) // group_size) * group_size)
                split_idx = min(max(split_idx, group_size), len(sign_token_ids) - 1)
            prefix = sign_token_ids[:split_idx]
            target = sign_token_ids[split_idx:]

        seq = [
            self._token_id("<mc>"),
            self._token_id("<sign>"),
            *prefix,
            self._token_id("</sign>"),
            self._token_id("<cont>"),
            *target,
            self._token_id("</cont>"),
        ]
        prefix_len = len(seq) - len(target) - 1
        labels = [-100] * prefix_len + target + [self._token_id("</cont>")]
        raw = (
            f"<mc> <sign> {' '.join(self.tokenizer.convert_ids_to_tokens(prefix))} </sign> "
            f"<cont> {' '.join(self.tokenizer.convert_ids_to_tokens(target))} </cont>"
        )
        return seq, labels, None, raw

    def build_batch(
        self,
        task_names: Sequence[str],
        texts: Sequence[str],
        sign_token_ids: Sequence[Sequence[int]],
        mc_prefix_ratio: float = 0.5,
        mc_group_size: int = 1,
        mc_random_prefix_ratio: bool = False,
        mc_prefix_ratio_min: float = 0.3,
        mc_prefix_ratio_max: float = 0.7,
        mc_min_prefix_tokens: int = 6,
        m2t_prefix_loss_weight: float = 0.0,
    ) -> CausalTaskBatch:
        if not (len(task_names) == len(texts) == len(sign_token_ids)):
            raise ValueError("task_names, texts, and sign_token_ids must have the same batch size.")

        sequences: List[List[int]] = []
        labels: List[List[int]] = []
        weights: List[Optional[List[float]]] = []
        raw_sequences: List[str] = []
        normalized_tasks: List[str] = []

        for task_name, text, cur_sign_ids in zip(task_names, texts, sign_token_ids):
            task_name = str(task_name).lower()
            if task_name == "t2m":
                seq, lab, wt, raw = self._build_t2m_sample(text=text, sign_token_ids=cur_sign_ids)
            elif task_name == "m2t":
                seq, lab, wt, raw = self._build_m2t_sample(
                    text=text, sign_token_ids=cur_sign_ids,
                    prefix_loss_weight=m2t_prefix_loss_weight,
                )
            elif task_name in ["mc", "pred", "continuation"]:
                seq, lab, wt, raw = self._build_mc_sample(
                    sign_token_ids=cur_sign_ids,
                    prefix_ratio=mc_prefix_ratio,
                    min_prefix_tokens=mc_min_prefix_tokens,
                    group_size=mc_group_size,
                    random_prefix_ratio=mc_random_prefix_ratio,
                    prefix_ratio_min=mc_prefix_ratio_min,
                    prefix_ratio_max=mc_prefix_ratio_max,
                )
                task_name = "mc"
            else:
                raise NotImplementedError(f"Unsupported causal task: {task_name}")

            sequences.append(seq)
            labels.append(lab)
            weights.append(wt)
            raw_sequences.append(raw)
            normalized_tasks.append(task_name)

        return self._pad_and_stack(sequences, labels, raw_sequences, normalized_tasks, weights)
