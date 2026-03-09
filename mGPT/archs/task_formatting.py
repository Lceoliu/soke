from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch


SPECIAL_TASK_TOKENS = ["<mc>", "<m2t>", "<sign>", "</sign>", "<text>", "</text>"]


@dataclass
class CausalTaskBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    spans: List[Dict[str, Tuple[int, int]]]
    raw_sequences: List[str]


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
    return "".join(pieces)


class SignLanguageTaskFormatter:
    """
    Build causal training sequences for future decoder-only sign-language tasks.

    This module is intentionally independent from the current lm training loop.
    It only serializes task strings and produces label masks that can be fed into
    an autoregressive model with teacher forcing.
    """

    def __init__(
        self,
        tokenizer,
        map_token_ids: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.map_token_ids = map_token_ids
        self.device = device
        add_missing_special_tokens(self.tokenizer)

    def _tokenize_piece(self, text: str) -> List[int]:
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def _stack_batch(self, sequences: List[List[int]], labels: List[List[int]], spans):
        if len(sequences) == 0:
            raise ValueError("Empty batch is not supported.")

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must define pad_token_id for causal batch building.")

        max_len = max(len(x) for x in sequences)
        batch = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
        lab = torch.full((len(sequences), max_len), -100, dtype=torch.long)
        attn = torch.zeros((len(sequences), max_len), dtype=torch.long)

        for i, (seq, cur_lab) in enumerate(zip(sequences, labels)):
            cur_len = len(seq)
            batch[i, :cur_len] = torch.tensor(seq, dtype=torch.long)
            lab[i, :cur_len] = torch.tensor(cur_lab, dtype=torch.long)
            attn[i, :cur_len] = 1

        if self.map_token_ids is not None:
            batch = self.map_token_ids(batch)
            valid = lab != -100
            if valid.any():
                lab_mapped = self.map_token_ids(lab.masked_fill(~valid, pad_id))
                lab = torch.where(valid, lab_mapped, torch.full_like(lab_mapped, -100))

        if self.device is not None:
            batch = batch.to(self.device)
            lab = lab.to(self.device)
            attn = attn.to(self.device)

        return CausalTaskBatch(
            input_ids=batch,
            labels=lab,
            attention_mask=attn,
            spans=spans,
            raw_sequences=[],
        )

    def build_motion_continuation(
        self,
        sign_strings: Sequence[str],
    ) -> CausalTaskBatch:
        sequences: List[List[int]] = []
        labels: List[List[int]] = []
        spans = []
        raw_sequences = []

        mc_ids = self._tokenize_piece("<mc>")
        sign_open_ids = self._tokenize_piece("<sign>")
        sign_close_ids = self._tokenize_piece("</sign>")

        for sign_str in sign_strings:
            sign_ids = self._tokenize_piece(sign_str)
            seq = mc_ids + sign_open_ids + sign_ids + sign_close_ids
            cur_labels = [-100] * (len(mc_ids) + len(sign_open_ids))
            cur_labels += list(sign_ids)
            cur_labels += [-100] * len(sign_close_ids)
            sequences.append(seq)
            labels.append(cur_labels)
            sign_start = len(mc_ids) + len(sign_open_ids)
            sign_end = sign_start + len(sign_ids)
            spans.append(
                {
                    "sign": (sign_start, sign_end),
                    "loss": (sign_start, sign_end),
                }
            )
            raw_sequences.append(f"<mc><sign>{sign_str}</sign>")

        batch = self._stack_batch(sequences, labels, spans)
        batch.raw_sequences = raw_sequences
        return batch

    def build_motion_to_text(
        self,
        sign_strings: Sequence[str],
        texts: Sequence[str],
    ) -> CausalTaskBatch:
        if len(sign_strings) != len(texts):
            raise ValueError("sign_strings and texts must have the same batch size.")

        sequences: List[List[int]] = []
        labels: List[List[int]] = []
        spans = []
        raw_sequences = []

        m2t_ids = self._tokenize_piece("<m2t>")
        sign_open_ids = self._tokenize_piece("<sign>")
        sign_close_ids = self._tokenize_piece("</sign>")
        text_open_ids = self._tokenize_piece("<text>")
        text_close_ids = self._tokenize_piece("</text>")

        for sign_str, text in zip(sign_strings, texts):
            sign_ids = self._tokenize_piece(sign_str)
            text_ids = self._tokenize_piece(text)
            seq = m2t_ids + sign_open_ids + sign_ids + sign_close_ids + text_open_ids + text_ids + text_close_ids
            prefix_len = len(m2t_ids) + len(sign_open_ids) + len(sign_ids) + len(sign_close_ids) + len(text_open_ids)
            cur_labels = [-100] * prefix_len + list(text_ids) + [-100] * len(text_close_ids)
            sequences.append(seq)
            labels.append(cur_labels)
            text_start = prefix_len
            text_end = text_start + len(text_ids)
            spans.append(
                {
                    "sign": (len(m2t_ids) + len(sign_open_ids), len(m2t_ids) + len(sign_open_ids) + len(sign_ids)),
                    "text": (text_start, text_end),
                    "loss": (text_start, text_end),
                }
            )
            raw_sequences.append(f"<m2t><sign>{sign_str}</sign><text>{text}</text>")

        batch = self._stack_batch(sequences, labels, spans)
        batch.raw_sequences = raw_sequences
        return batch
