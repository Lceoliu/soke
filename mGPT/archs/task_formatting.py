from __future__ import annotations

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


@dataclass
class CausalTaskBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    raw_sequences: List[str]
    task_names: List[str]


def add_missing_special_tokens(tokenizer) -> List[str]:
    vocab = tokenizer.get_vocab()
    missing = [tok for tok in SPECIAL_TASK_TOKENS if tok not in vocab]
    if missing:
        tokenizer.add_tokens(missing, special_tokens=True)
    return missing


def serialize_sign_tokens(
    body_tokens: Sequence[int],
    lhand_tokens: Optional[Sequence[int]] = None,
    rhand_tokens: Optional[Sequence[int]] = None,
) -> str:
    body = list(body_tokens)
    lhand = list(lhand_tokens) if lhand_tokens is not None else None
    rhand = list(rhand_tokens) if rhand_tokens is not None else None

    if lhand is not None and len(lhand) != len(body):
        raise ValueError(f"Left-hand token length {len(lhand)} != body token length {len(body)}")
    if rhand is not None and len(rhand) != len(body):
        raise ValueError(f"Right-hand token length {len(rhand)} != body token length {len(body)}")

    pieces: List[str] = []
    for idx, b in enumerate(body):
        pieces.append(f"<motion_id_{int(b)}>")
        if lhand is not None:
            pieces.append(f"<hand_id_{int(lhand[idx])}>")
        if rhand is not None:
            pieces.append(f"<rhand_id_{int(rhand[idx])}>")
    return " ".join(pieces)


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

        for i, (seq, cur_lab) in enumerate(zip(sequences, labels)):
            seq_len = len(seq)
            batch[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
            lab[i, :seq_len] = torch.tensor(cur_lab, dtype=torch.long)
            attn[i, :seq_len] = 1

        return CausalTaskBatch(
            input_ids=batch,
            labels=lab,
            attention_mask=attn,
            raw_sequences=raw_sequences,
            task_names=task_names,
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
        return seq, labels, raw

    def _build_m2t_sample(self, text: str, sign_token_ids: Sequence[int]):
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
        prefix_len = len(seq) - len(text_ids) - 1
        labels = [-100] * prefix_len + text_ids + [self._token_id("</text>")]
        raw = f"<m2t> <sign> {' '.join(self.tokenizer.convert_ids_to_tokens(sign_token_ids))} </sign> <text> {text} </text>"
        return seq, labels, raw

    def _build_mc_sample(
        self,
        sign_token_ids: Sequence[int],
        prefix_ratio: float = 0.5,
        min_prefix_tokens: int = 6,
    ):
        sign_token_ids = list(sign_token_ids)
        if len(sign_token_ids) < 2:
            prefix = sign_token_ids[:1]
            target = sign_token_ids[1:]
        else:
            split_idx = max(min_prefix_tokens, int(round(len(sign_token_ids) * prefix_ratio)))
            split_idx = min(max(split_idx, 1), len(sign_token_ids) - 1)
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
        return seq, labels, raw

    def build_batch(
        self,
        task_names: Sequence[str],
        texts: Sequence[str],
        sign_token_ids: Sequence[Sequence[int]],
        mc_prefix_ratio: float = 0.5,
    ) -> CausalTaskBatch:
        if not (len(task_names) == len(texts) == len(sign_token_ids)):
            raise ValueError("task_names, texts, and sign_token_ids must have the same batch size.")

        sequences: List[List[int]] = []
        labels: List[List[int]] = []
        raw_sequences: List[str] = []
        normalized_tasks: List[str] = []

        for task_name, text, cur_sign_ids in zip(task_names, texts, sign_token_ids):
            task_name = str(task_name).lower()
            if task_name == "t2m":
                seq, lab, raw = self._build_t2m_sample(text=text, sign_token_ids=cur_sign_ids)
            elif task_name == "m2t":
                seq, lab, raw = self._build_m2t_sample(text=text, sign_token_ids=cur_sign_ids)
            elif task_name in ["mc", "pred", "continuation"]:
                seq, lab, raw = self._build_mc_sample(
                    sign_token_ids=cur_sign_ids,
                    prefix_ratio=mc_prefix_ratio,
                )
                task_name = "mc"
            else:
                raise NotImplementedError(f"Unsupported causal task: {task_name}")

            sequences.append(seq)
            labels.append(lab)
            raw_sequences.append(raw)
            normalized_tasks.append(task_name)

        return self._pad_and_stack(sequences, labels, raw_sequences, normalized_tasks)
