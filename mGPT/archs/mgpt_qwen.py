import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import LoraConfig, get_peft_model

from mGPT.archs.task_formatting import (
    SignLanguageTaskFormatter,
    add_missing_special_tokens,
    serialize_sign_tokens,
)


class QwenCausalLM(nn.Module):
    def __init__(
        self,
        model_path: str,
        model_type: str = "qwen2_causal",
        stage: str = "lm_pretrain",
        motion_codebook_size: int = 512,
        hand_codebook_size: int = 0,
        rhand_codebook_size: int = 0,
        max_length: int = 2048,
        generation_max_new_tokens: int = 1024,
        trust_remote_code: bool = True,
        use_lora: bool = True,
        lora_rank: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        gradient_checkpointing: bool = True,
        mc_prefix_ratio: float = 0.5,
        torch_dtype: str = "bfloat16",
        **kwargs,
    ) -> None:
        super().__init__()

        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Qwen model path not found: {model_path}")

        self.model_type = model_type
        self.stage = stage
        self.max_length = int(max_length)
        self.generation_max_new_tokens = int(generation_max_new_tokens)
        self.m_codebook_size = int(motion_codebook_size)
        self.hand_codebook_size = int(hand_codebook_size)
        self.rhand_codebook_size = int(rhand_codebook_size)
        self.mc_prefix_ratio = float(mc_prefix_ratio)
        self.num_token_parts = 1 + int(self.hand_codebook_size > 0) + int(self.rhand_codebook_size > 0)
        self.model_dtype = self._resolve_torch_dtype(torch_dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        add_missing_special_tokens(self.tokenizer)
        self.motion_tokens = [f"<motion_id_{i}>" for i in range(self.m_codebook_size + 3)]
        self.hand_tokens = [f"<hand_id_{i}>" for i in range(self.hand_codebook_size + 3)]
        self.rhand_tokens = [f"<rhand_id_{i}>" for i in range(self.rhand_codebook_size + 3)]
        self.tokenizer.add_tokens(self.motion_tokens + self.hand_tokens + self.rhand_tokens)

        self.language_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            dtype=self.model_dtype,
        )
        self.language_model.resize_token_embeddings(len(self.tokenizer))
        self.language_model.config.pad_token_id = self.tokenizer.pad_token_id
        if gradient_checkpointing:
            self.language_model.gradient_checkpointing_enable()
        if hasattr(self.language_model.config, "use_cache"):
            self.language_model.config.use_cache = False if "lm" in str(stage) else True

        self._untie_lm_head_if_needed()

        self.task_formatter = SignLanguageTaskFormatter(self.tokenizer)
        self.special_token_ids: Dict[str, int] = {
            tok: self.tokenizer.convert_tokens_to_ids(tok)
            for tok in ["<t2m>", "<m2t>", "<mc>", "<sign>", "</sign>", "<text>", "</text>", "<cont>", "</cont>"]
        }

        if use_lora:
            self._apply_lora(
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
        else:
            self._freeze_backbone_only()

        self._enable_embedding_and_lm_head_training()

    @property
    def device(self):
        return next(self.language_model.parameters()).device

    def _freeze_backbone_only(self):
        for param in self.language_model.parameters():
            param.requires_grad = False

    @staticmethod
    def _resolve_torch_dtype(torch_dtype):
        if isinstance(torch_dtype, torch.dtype):
            return torch_dtype
        if torch_dtype is None:
            return torch.bfloat16
        key = str(torch_dtype).lower()
        mapping = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        if key not in mapping:
            raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")
        return mapping[key]

    def _untie_lm_head_if_needed(self):
        if not getattr(self.language_model.config, "tie_word_embeddings", False):
            return
        input_embeddings = self.language_model.get_input_embeddings()
        if input_embeddings is None:
            return
        old_lm_head = getattr(self.language_model, "lm_head", None)
        if old_lm_head is None:
            return

        new_lm_head = nn.Linear(
            input_embeddings.weight.shape[1],
            input_embeddings.weight.shape[0],
            bias=False,
            dtype=input_embeddings.weight.dtype,
            device=input_embeddings.weight.device,
        )
        with torch.no_grad():
            new_lm_head.weight.copy_(input_embeddings.weight.detach())
        self.language_model.lm_head = new_lm_head
        self.language_model.config.tie_word_embeddings = False

    def _apply_lora(self, rank: int, alpha: int, dropout: float):
        lora_cfg = LoraConfig(
            r=int(rank),
            lora_alpha=int(alpha),
            lora_dropout=float(dropout),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        self.language_model = get_peft_model(self.language_model, lora_cfg)

    def _enable_embedding_and_lm_head_training(self):
        for param in self.language_model.parameters():
            if not hasattr(param, "requires_grad"):
                continue

        embed_module = self.language_model.get_input_embeddings()
        if embed_module is not None:
            for param in embed_module.parameters():
                param.requires_grad = True

        if hasattr(self.language_model, "lm_head") and self.language_model.lm_head is not None:
            for param in self.language_model.lm_head.parameters():
                param.requires_grad = True

    def _resolve_task_name(self, task_entry, default_task: str = "t2m") -> str:
        if task_entry is None:
            return default_task
        if isinstance(task_entry, dict):
            return str(task_entry.get("class", default_task)).lower()
        return str(task_entry).lower()

    def _split_motion_sample(
        self,
        motion_tokens: Tensor,
        length: int,
    ) -> Tuple[List[int], Optional[List[int]], Optional[List[int]]]:
        cur = motion_tokens[: int(length)]
        if cur.dim() == 0:
            return [int(cur.item())], None, None
        if cur.dim() == 1:
            return cur.long().tolist(), None, None
        if cur.dim() != 2:
            raise ValueError(f"Unsupported motion token shape: {tuple(cur.shape)}")

        body = cur[:, 0].long().tolist()
        lhand = cur[:, 1].long().tolist() if cur.shape[1] > 1 else None
        rhand = cur[:, 2].long().tolist() if cur.shape[1] > 2 else None
        return body, lhand, rhand

    def _build_sign_token_ids(
        self,
        body_tokens: Sequence[int],
        lhand_tokens: Optional[Sequence[int]] = None,
        rhand_tokens: Optional[Sequence[int]] = None,
    ) -> List[int]:
        sign_str = serialize_sign_tokens(body_tokens, lhand_tokens, rhand_tokens)
        return self.tokenizer(sign_str, add_special_tokens=False).input_ids

    def _motion_tensor_to_sign_token_ids(self, motion_tokens: Tensor) -> List[int]:
        if motion_tokens.dim() == 1:
            return self._build_sign_token_ids(motion_tokens.long().tolist())
        if motion_tokens.dim() != 2:
            raise ValueError(f"Unsupported motion tensor shape for sign serialization: {tuple(motion_tokens.shape)}")
        body = motion_tokens[:, 0].long().tolist()
        lhand = motion_tokens[:, 1].long().tolist() if motion_tokens.shape[1] > 1 else None
        rhand = motion_tokens[:, 2].long().tolist() if motion_tokens.shape[1] > 2 else None
        return self._build_sign_token_ids(body, lhand, rhand)

    def forward(
        self,
        texts: List[str],
        motion_tokens: Tensor,
        lengths: List[int],
        tasks=None,
        src: Optional[List[str]] = None,
        name: Optional[List[str]] = None,
    ):
        task_names = []
        sign_token_ids = []
        for idx in range(len(texts)):
            task_entry = tasks[idx] if tasks is not None else None
            task_names.append(self._resolve_task_name(task_entry, default_task="t2m"))
            body, lhand, rhand = self._split_motion_sample(motion_tokens[idx], lengths[idx])
            sign_token_ids.append(self._build_sign_token_ids(body, lhand, rhand))

        batch = self.task_formatter.build_batch(
            task_names=task_names,
            texts=texts,
            sign_token_ids=sign_token_ids,
            mc_prefix_ratio=self.mc_prefix_ratio,
            mc_group_size=self.num_token_parts,
        )

        input_ids = batch.input_ids.to(self.device)
        attention_mask = batch.attention_mask.to(self.device)
        labels = batch.labels.to(self.device)
        return self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

    def _make_t2m_prompt(self, text: str) -> torch.Tensor:
        seq = (
            f"<t2m> <text> {text} </text> <sign>"
        )
        ids = self.tokenizer(seq, add_special_tokens=False, return_tensors="pt").input_ids
        return ids

    def _make_m2t_prompt(self, sign_token_ids: Sequence[int]) -> torch.Tensor:
        seq = [
            self.special_token_ids["<m2t>"],
            self.special_token_ids["<sign>"],
            *list(sign_token_ids),
            self.special_token_ids["</sign>"],
            self.special_token_ids["<text>"],
        ]
        return torch.tensor(seq, dtype=torch.long).unsqueeze(0)

    def _make_mc_prompt(self, sign_token_ids: Sequence[int]) -> torch.Tensor:
        sign_token_ids = list(sign_token_ids)
        split_idx = max(1, int(round(len(sign_token_ids) * self.mc_prefix_ratio)))
        split_idx = min(split_idx, max(len(sign_token_ids) - 1, 1))
        if self.num_token_parts > 1:
            split_idx = max(self.num_token_parts, (split_idx // self.num_token_parts) * self.num_token_parts)
            if split_idx >= len(sign_token_ids):
                split_idx = max(
                    self.num_token_parts,
                    ((len(sign_token_ids) - 1) // self.num_token_parts) * self.num_token_parts,
                )
        prefix = sign_token_ids[:split_idx]
        seq = [
            self.special_token_ids["<mc>"],
            self.special_token_ids["<sign>"],
            *prefix,
            self.special_token_ids["</sign>"],
            self.special_token_ids["<cont>"],
        ]
        return torch.tensor(seq, dtype=torch.long).unsqueeze(0)

    def _generate_ids(
        self,
        prompt_ids: torch.Tensor,
        stop_token_id: int,
        do_sample: bool = False,
    ) -> torch.Tensor:
        prompt_ids = prompt_ids.to(self.device)
        attention_mask = torch.ones_like(prompt_ids, device=self.device)
        generate_kwargs = dict(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.generation_max_new_tokens,
            eos_token_id=int(stop_token_id),
            pad_token_id=int(self.tokenizer.pad_token_id),
            do_sample=bool(do_sample),
            use_cache=True,
        )
        outputs = self.language_model.generate(**generate_kwargs)
        return outputs[0]

    @staticmethod
    def _extract_after_prompt(output_ids: torch.Tensor, prompt_len: int) -> List[int]:
        return output_ids[prompt_len:].tolist()

    def _parse_generated_sign_tokens(
        self,
        generated_ids: Sequence[int],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        body: List[int] = []
        lhand: List[int] = []
        rhand: List[int] = []

        for tok_id in generated_ids:
            if int(tok_id) == int(self.special_token_ids["</sign>"]):
                break
            token = self.tokenizer.convert_ids_to_tokens(int(tok_id))
            if token.startswith("<motion_id_"):
                body.append(int(token[len("<motion_id_"):-1]))
            elif token.startswith("<hand_id_"):
                lhand.append(int(token[len("<hand_id_"):-1]))
            elif token.startswith("<rhand_id_"):
                rhand.append(int(token[len("<rhand_id_"):-1]))

        min_len = len(body)
        if lhand:
            min_len = min(min_len, len(lhand))
        if rhand:
            min_len = min(min_len, len(rhand))
        body = body[:min_len]
        if lhand:
            lhand = lhand[:min_len]
        if rhand:
            rhand = rhand[:min_len]

        body_tensor = torch.tensor(body, dtype=torch.long, device=self.device)
        lhand_tensor = (
            torch.tensor(lhand, dtype=torch.long, device=self.device)
            if lhand else None
        )
        rhand_tensor = (
            torch.tensor(rhand, dtype=torch.long, device=self.device)
            if rhand else None
        )
        return body_tensor, lhand_tensor, rhand_tensor

    def _parse_generated_text(
        self,
        generated_ids: Sequence[int],
        stop_token_id: int,
    ) -> str:
        text_ids: List[int] = []
        for tok_id in generated_ids:
            if int(tok_id) == int(stop_token_id):
                break
            text_ids.append(int(tok_id))
        text = self.tokenizer.decode(text_ids, skip_special_tokens=True)
        return text.strip()

    @torch.no_grad()
    def generate_conditional(
        self,
        texts: Optional[List[str]] = None,
        motion_tokens: Optional[List[Tensor]] = None,
        lengths: Optional[List[int]] = None,
        task: str = "t2m",
        stage: str = "test",
        tasks=None,
        src: Optional[List[str]] = None,
        name: Optional[List[str]] = None,
        do_sample: bool = False,
    ):
        task = str(task).lower()
        if task == "t2m":
            if texts is None:
                raise ValueError("texts must be provided for t2m generation.")
            outputs_tokens = []
            outputs_tokens_hand = []
            outputs_tokens_rhand = []
            for text in texts:
                prompt_ids = self._make_t2m_prompt(text)
                output_ids = self._generate_ids(
                    prompt_ids,
                    stop_token_id=self.special_token_ids["</sign>"],
                    do_sample=do_sample,
                )
                tail_ids = self._extract_after_prompt(output_ids, prompt_ids.shape[1])
                body, lhand, rhand = self._parse_generated_sign_tokens(tail_ids)
                outputs_tokens.append(body)
                outputs_tokens_hand.append(
                    lhand if lhand is not None else torch.zeros(0, dtype=torch.long, device=self.device)
                )
                outputs_tokens_rhand.append(
                    rhand if rhand is not None else torch.zeros(0, dtype=torch.long, device=self.device)
                )
            has_lhand = any(x.numel() > 0 for x in outputs_tokens_hand)
            has_rhand = any(x.numel() > 0 for x in outputs_tokens_rhand)
            return {
                "outputs_tokens": outputs_tokens,
                "outputs_tokens_hand": outputs_tokens_hand if has_lhand else None,
                "outputs_tokens_rhand": outputs_tokens_rhand if has_rhand else None,
            }

        if task == "m2t":
            if motion_tokens is None:
                raise ValueError("motion_tokens must be provided for m2t generation.")
            outputs: List[str] = []
            for cur_tokens in motion_tokens:
                sign_token_ids = self._motion_tensor_to_sign_token_ids(cur_tokens)
                prompt_ids = self._make_m2t_prompt(sign_token_ids)
                output_ids = self._generate_ids(
                    prompt_ids,
                    stop_token_id=self.special_token_ids["</text>"],
                    do_sample=do_sample,
                )
                tail_ids = self._extract_after_prompt(output_ids, prompt_ids.shape[1])
                outputs.append(self._parse_generated_text(tail_ids, self.special_token_ids["</text>"]))
            return outputs

        if task in ["mc", "pred", "continuation"]:
            if motion_tokens is None:
                raise ValueError("motion_tokens must be provided for mc generation.")
            outputs_tokens = []
            outputs_tokens_hand = []
            outputs_tokens_rhand = []
            for cur_tokens in motion_tokens:
                sign_token_ids = self._motion_tensor_to_sign_token_ids(cur_tokens)
                prompt_ids = self._make_mc_prompt(sign_token_ids)
                output_ids = self._generate_ids(
                    prompt_ids,
                    stop_token_id=self.special_token_ids["</cont>"],
                    do_sample=do_sample,
                )
                tail_ids = self._extract_after_prompt(output_ids, prompt_ids.shape[1])
                body, lhand, rhand = self._parse_generated_sign_tokens(tail_ids)
                outputs_tokens.append(body)
                outputs_tokens_hand.append(
                    lhand if lhand is not None else torch.zeros(0, dtype=torch.long, device=self.device)
                )
                outputs_tokens_rhand.append(
                    rhand if rhand is not None else torch.zeros(0, dtype=torch.long, device=self.device)
                )
            has_lhand = any(x.numel() > 0 for x in outputs_tokens_hand)
            has_rhand = any(x.numel() > 0 for x in outputs_tokens_rhand)
            return {
                "outputs_tokens": outputs_tokens,
                "outputs_tokens_hand": outputs_tokens_hand if has_lhand else None,
                "outputs_tokens_rhand": outputs_tokens_rhand if has_rhand else None,
            }

        raise NotImplementedError(f"Unsupported generation task: {task}")

    @torch.no_grad()
    def generate_direct(self, texts: List[str], do_sample: bool = False):
        gen_results = self.generate_conditional(texts=texts, task="t2m", do_sample=do_sample)
        outputs_tokens = gen_results["outputs_tokens"]
        outputs_tokens_hand = gen_results.get("outputs_tokens_hand", None)
        outputs_tokens_rhand = gen_results.get("outputs_tokens_rhand", None)
        output_texts = []
        for idx, body in enumerate(outputs_tokens):
            lhand = None
            rhand = None
            if outputs_tokens_hand is not None:
                lhand = outputs_tokens_hand[idx].tolist()
            if outputs_tokens_rhand is not None:
                rhand = outputs_tokens_rhand[idx].tolist()
            output_texts.append(serialize_sign_tokens(body.tolist(), lhand, rhand))
        return outputs_tokens, output_texts
