"""mT5-based seq2seq model for sign-language motion-to-text translation."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor, nn
from transformers import AutoTokenizer, MT5ForConditionalGeneration

from peft import LoraConfig, TaskType, get_peft_model


# ---------------------------------------------------------------------------
# Projection: VAE / raw features  -->  mT5 hidden dimension
# ---------------------------------------------------------------------------

class SignEmbeddingProjection(nn.Module):
    """LayerNorm + Linear (or 2-layer MLP when input_dim >> output_dim)."""

    def __init__(self, input_dim: int, output_dim: int, use_mlp: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        if use_mlp or input_dim > output_dim * 2:
            self.proj = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.GELU(),
                nn.Linear(output_dim, output_dim),
            )
        else:
            self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.norm(x))


# ---------------------------------------------------------------------------
# Language mapping
# ---------------------------------------------------------------------------

_SRC_TO_LANG: Dict[str, str] = {
    "csl": "中文",
    "phoenix": "德语",
    "h2s": "英语",
    "how2sign": "英语",
}


# ---------------------------------------------------------------------------
# MT5Seq2SeqLM
# ---------------------------------------------------------------------------

class MT5Seq2SeqLM(nn.Module):
    """Encoder-decoder wrapper around mT5 for motion-to-text translation.

    Instead of discrete motion-token IDs, this model receives *continuous*
    embeddings (either raw motion features or VAE encoder outputs) and
    projects them into the mT5 encoder via ``inputs_embeds``.
    """

    needs_raw_features: bool = True  # flag read by MotionGPT

    def __init__(
        self,
        model_path: str,
        model_type: str = "mt5_seq2seq",
        stage: str = "lm_pretrain",
        vae_dim: Optional[int] = None,
        max_length: int = 512,
        generation_max_new_tokens: int = 256,
        use_lora: bool = True,
        lora_rank: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        gradient_checkpointing: bool = True,
        torch_dtype: str = "bfloat16",
        use_mlp_proj: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"mT5 model path not found: {model_path}")

        self.model_type = model_type
        self.stage = stage
        self.max_length = int(max_length)
        self.generation_max_new_tokens = int(generation_max_new_tokens)
        self.model_dtype = self._resolve_torch_dtype(torch_dtype)

        # --- VAE references (frozen, optional) ---
        # Set to None here; MotionGPT attaches the actual modules *after*
        # instantiation so they stay out of Lightning hparams.
        self._vae: Optional[nn.Module] = None
        self._hand_vae: Optional[nn.Module] = None
        self._rhand_vae: Optional[nn.Module] = None

        # --- Tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # --- Language model ---
        self.language_model = MT5ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=self.model_dtype,
        )
        if gradient_checkpointing:
            self.language_model.gradient_checkpointing_enable()
        if hasattr(self.language_model.config, "use_cache"):
            self.language_model.config.use_cache = False

        # --- Projection layer ---
        d_model = int(self.language_model.config.d_model)  # 768 for mt5-base
        input_dim = self._resolve_input_dim(vae_dim)
        self.sign_proj = SignEmbeddingProjection(input_dim, d_model, use_mlp=use_mlp_proj)

        # --- LoRA ---
        if use_lora:
            self._apply_lora(rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_input_dim(vae_dim: Optional[int]) -> int:
        """Return projection input dimension.

        ``vae_dim`` is computed by MotionGPT from the frozen VAEs and passed
        as a plain integer via the config so that no nn.Module leaks into
        Lightning hparams.  Falls back to raw 133-dim features.
        """
        if vae_dim is not None:
            return int(vae_dim)
        return 133

    @staticmethod
    def _resolve_torch_dtype(torch_dtype):
        if isinstance(torch_dtype, torch.dtype):
            return torch_dtype
        mapping = {
            "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
            "fp16": torch.float16, "float16": torch.float16,
            "fp32": torch.float32, "float32": torch.float32,
        }
        key = str(torch_dtype or "bfloat16").lower()
        if key not in mapping:
            raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")
        return mapping[key]

    def _apply_lora(self, rank: int, alpha: int, dropout: float):
        lora_cfg = LoraConfig(
            r=int(rank),
            lora_alpha=int(alpha),
            lora_dropout=float(dropout),
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
            target_modules=["q", "k", "v", "o"],
        )
        self.language_model = get_peft_model(self.language_model, lora_cfg)

    @property
    def device(self):
        return next(self.language_model.parameters()).device

    # ------------------------------------------------------------------
    # Encoding: motion features -> projected embeddings
    # ------------------------------------------------------------------

    def _encode_and_project(
        self,
        motion_features: Tensor,
        lengths: List[int],
    ) -> tuple[Tensor, List[int]]:
        """Encode motion features (optionally via VAE) and project to d_model.

        Returns:
            sign_embeds: ``[B, T', d_model]``
            sign_lengths: valid lengths after potential VAE downsampling
        """
        if self._vae is not None:
            sign_embeds, sign_lengths = self._encode_with_vaes(motion_features, lengths)
        else:
            sign_embeds = motion_features  # [B, T, 133]
            sign_lengths = [int(l) for l in lengths]

        sign_embeds = self.sign_proj(sign_embeds.to(dtype=self.model_dtype))
        return sign_embeds, sign_lengths

    def _encode_with_vaes(
        self,
        features: Tensor,
        lengths: List[int],
    ) -> tuple[Tensor, List[int]]:
        """Run frozen VAE encoders per-sample to avoid padding artifacts.

        Each sample is sliced to its real length before encoding so that
        (a) the convolutional encoder never sees padding values and
        (b) the output length is deterministic for a given input length.
        """
        B = features.shape[0]
        sample_embeds: List[Tensor] = []
        sign_lengths: List[int] = []
        has_hand = self._hand_vae is not None
        has_rhand = self._rhand_vae is not None

        with torch.no_grad():
            for i in range(B):
                cur_len = int(lengths[i])
                cur = features[i:i + 1, :cur_len]  # [1, L_i, D]

                if has_hand and has_rhand:
                    fb = torch.cat([cur[..., :30], cur[..., 120:]], dim=-1)
                    fl = cur[..., 30:75]
                    fr = cur[..., 75:120]
                    eb = self._vae.encode_continuous(fb)
                    el = self._hand_vae.encode_continuous(fl)
                    er = self._rhand_vae.encode_continuous(fr)
                    mt = min(eb.shape[1], el.shape[1], er.shape[1])
                    emb = torch.cat([eb[:, :mt], el[:, :mt], er[:, :mt]], dim=-1)
                elif has_hand:
                    fb = torch.cat([cur[..., :30], cur[..., 120:]], dim=-1)
                    fh = cur[..., 30:120]
                    eb = self._vae.encode_continuous(fb)
                    eh = self._hand_vae.encode_continuous(fh)
                    mt = min(eb.shape[1], eh.shape[1])
                    emb = torch.cat([eb[:, :mt], eh[:, :mt]], dim=-1)
                else:
                    emb = self._vae.encode_continuous(cur)

                sample_embeds.append(emb[0])  # [T'_i, D_vae]
                sign_lengths.append(int(emb.shape[1]))

        # Pad to batch max length
        d_vae = sample_embeds[0].shape[-1]
        max_t = max(sign_lengths)
        sign_embeds = features.new_zeros(B, max_t, d_vae)
        for i, emb in enumerate(sample_embeds):
            sign_embeds[i, : emb.shape[0]] = emb

        return sign_embeds, sign_lengths

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt_embeds(
        self,
        src_labels: List[str],
        batch_size: int,
    ) -> tuple[Tensor, Tensor]:
        """Build prompt token embeddings and attention mask."""
        prompt_texts = []
        for src in src_labels:
            lang = _SRC_TO_LANG.get(str(src).lower(), "手语")
            prompt_texts.append(f"把下面这句{lang}手语翻译为{lang}文本:")

        encoded = self.tokenizer(
            prompt_texts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encoded.input_ids.to(self.device)
        prompt_mask = encoded.attention_mask.to(self.device)

        # Look up embeddings from mT5 shared embedding table
        embed_layer = self._get_input_embeddings()
        prompt_embeds = embed_layer(input_ids).to(dtype=self.model_dtype)
        return prompt_embeds, prompt_mask

    def _get_input_embeddings(self):
        """Get the shared input embedding layer, handling PEFT wrapping."""
        model = self.language_model
        if hasattr(model, "get_input_embeddings"):
            return model.get_input_embeddings()
        if hasattr(model, "base_model"):
            return model.base_model.model.shared
        return model.shared

    # ------------------------------------------------------------------
    # Target tokenization
    # ------------------------------------------------------------------

    def _tokenize_targets(self, texts: List[str]) -> Tensor:
        """Tokenize target texts and build labels with -100 at pad positions."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )
        labels = encoded.input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return labels.to(self.device)

    # ------------------------------------------------------------------
    # Mask helpers
    # ------------------------------------------------------------------

    def _build_length_mask(self, max_len: int, lengths: List[int]) -> Tensor:
        mask = torch.zeros(len(lengths), max_len, dtype=torch.long, device=self.device)
        for i, l in enumerate(lengths):
            mask[i, : int(l)] = 1
        return mask

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        texts: List[str],
        motion_features: Tensor,
        lengths: List[int],
        tasks=None,
        src: Optional[List[str]] = None,
        name: Optional[List[str]] = None,
        **kwargs,
    ):
        B = len(texts)
        src = src or ["csl"] * B

        # 1. Encode & project sign embeddings
        sign_embeds, sign_lengths = self._encode_and_project(motion_features, lengths)

        # 2. Prompt embeddings
        prompt_embeds, prompt_mask = self._build_prompt_embeds(src, B)

        # 3. Concatenate encoder inputs
        inputs_embeds = torch.cat([prompt_embeds, sign_embeds], dim=1)
        sign_mask = self._build_length_mask(sign_embeds.shape[1], sign_lengths)
        attention_mask = torch.cat([prompt_mask, sign_mask], dim=1)

        # 4. Decoder labels
        labels = self._tokenize_targets(texts)

        # 5. Forward
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        return outputs

    # ------------------------------------------------------------------
    # Generation (inference)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_conditional(
        self,
        texts: Optional[List[str]] = None,
        motion_tokens: Optional[List[Tensor]] = None,
        motion_features: Optional[Tensor] = None,
        lengths: Optional[List[int]] = None,
        task: str = "m2t",
        stage: str = "test",
        src: Optional[List[str]] = None,
        name: Optional[List[str]] = None,
        **kwargs,
    ):
        if task != "m2t":
            raise NotImplementedError(
                f"MT5Seq2SeqLM currently only supports m2t generation, got: {task}"
            )
        if motion_features is None:
            raise ValueError("MT5Seq2SeqLM.generate_conditional requires motion_features")

        B = motion_features.shape[0]
        src = src or ["csl"] * B

        sign_embeds, sign_lengths = self._encode_and_project(motion_features, lengths)
        prompt_embeds, prompt_mask = self._build_prompt_embeds(src, B)

        inputs_embeds = torch.cat([prompt_embeds, sign_embeds], dim=1)
        sign_mask = self._build_length_mask(sign_embeds.shape[1], sign_lengths)
        attention_mask = torch.cat([prompt_mask, sign_mask], dim=1)

        # Encode first, then generate (robust with PEFT)
        encoder = self._get_encoder()
        encoder_outputs = encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )

        generated = self.language_model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            max_new_tokens=self.generation_max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        output_texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return {"outputs": output_texts}

    def _get_encoder(self):
        """Get the mT5 encoder (with LoRA adapters applied)."""
        model = self.language_model
        if hasattr(model, "base_model"):
            # PeftModel -> LoraModel -> MT5ForConditionalGeneration
            return model.base_model.model.encoder
        return model.encoder

    @torch.no_grad()
    def generate_direct(self, texts: List[str], **kwargs):
        raise NotImplementedError(
            "MT5Seq2SeqLM only supports m2t (motion-to-text). "
            "Use generate_conditional(motion_features=..., task='m2t') instead."
        )
