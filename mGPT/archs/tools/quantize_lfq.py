import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualLFQ(nn.Module):
    """
    Residual Lookup-Free Quantization.
    Each level predicts binary bits directly (no embedding lookup table), then
    projects bits back to latent space and quantizes the residual.
    """

    def __init__(
        self,
        nb_code: int,
        code_dim: int,
        num_quantizers: int = 1,
        aggregate: str = "mean",
        ste_temperature: float = 1.0,
        ste_temperature_end: float = 0.1,
        entropy_loss_weight: float = 0.1,
        entropy_global_weight: float = 1.0,
        entropy_local_weight: float = 1.0,
    ):
        super().__init__()
        if nb_code < 2:
            raise ValueError(f"nb_code must be >= 2, got {nb_code}")
        if code_dim < 1:
            raise ValueError(f"code_dim must be >= 1, got {code_dim}")
        if num_quantizers < 1:
            raise ValueError(f"num_quantizers must be >= 1, got {num_quantizers}")
        if aggregate not in ("mean", "sum"):
            raise ValueError(f"aggregate must be one of ['mean', 'sum'], got {aggregate}")

        self.nb_code = int(nb_code)
        self.code_dim = int(code_dim)
        self.num_quantizers = int(num_quantizers)
        self.aggregate = aggregate
        self.ste_temperature_start = float(ste_temperature)
        self.ste_temperature_end = float(ste_temperature_end)
        self.ste_temperature = float(ste_temperature)
        self.entropy_loss_weight = float(entropy_loss_weight)
        self.entropy_global_weight = float(entropy_global_weight)
        self.entropy_local_weight = float(entropy_local_weight)

        self.bits_per_code = int(math.ceil(math.log2(self.nb_code)))
        self.full_code_num = int(2 ** self.bits_per_code)

        self.to_bits = nn.ModuleList(
            [nn.Linear(self.code_dim, self.bits_per_code) for _ in range(self.num_quantizers)]
        )
        self.from_bits = nn.ModuleList(
            [nn.Linear(self.bits_per_code, self.code_dim) for _ in range(self.num_quantizers)]
        )
        self.register_buffer("_bit_shifts", torch.arange(self.bits_per_code, dtype=torch.long))

    def _reduce(self, values):
        stacked = torch.stack(values, dim=0)
        if self.aggregate == "sum":
            return stacked.sum(dim=0)
        return stacked.mean(dim=0)

    def _flatten_nct(self, x: torch.Tensor):
        if x.dim() != 3:
            raise ValueError(f"Expected [N, C, T], got {tuple(x.shape)}")
        n, c, t = x.shape
        if c != self.code_dim:
            raise ValueError(f"Expected channel dim={self.code_dim}, got {c}")
        tokens = x.permute(0, 2, 1).contiguous().view(-1, c)
        return tokens, n, t

    def _restore_nct(self, tokens: torch.Tensor, n: int, t: int):
        return tokens.view(n, t, self.code_dim).permute(0, 2, 1).contiguous()

    def _hard_bits(self, logits: torch.Tensor):
        return torch.where(logits >= 0, torch.ones_like(logits), -torch.ones_like(logits))

    def _bits_to_indices(self, bits_pm1: torch.Tensor):
        bits01 = (bits_pm1 > 0).long()
        idx = (bits01 * (2 ** self._bit_shifts)).sum(dim=-1)
        if self.full_code_num != self.nb_code:
            idx = idx.remainder(self.nb_code)
        return idx

    def _indices_to_bits(self, code_idx: torch.Tensor):
        idx = code_idx.long()
        if self.full_code_num != self.nb_code:
            idx = idx.remainder(self.nb_code)
        bits01 = ((idx.unsqueeze(-1) >> self._bit_shifts) & 1).float()
        return bits01.mul(2.0).sub(1.0)

    @torch.no_grad()
    def _compute_perplexity(self, code_idx: torch.Tensor):
        counts = torch.bincount(code_idx.view(-1), minlength=self.nb_code).float()
        probs = counts / counts.sum().clamp_min(1.0)
        entropy = -(probs * (probs + 1e-7).log()).sum()
        return torch.exp(entropy)

    def _global_bit_balance_loss(self, logits: torch.Tensor):
        # Maximize global bit entropy by driving per-bit Bernoulli p to 0.5.
        probs = torch.sigmoid(logits)
        p_mean = probs.mean(dim=0)
        return ((p_mean - 0.5) ** 2).mean()

    def _local_bit_confidence_loss(self, logits: torch.Tensor):
        # Minimize local bit entropy by making logits stay away from 0.
        return torch.exp(-torch.abs(logits)).mean()

    def set_anneal_progress(self, progress: float):
        progress = float(max(0.0, min(1.0, progress)))
        self.ste_temperature = (
            self.ste_temperature_start
            + (self.ste_temperature_end - self.ste_temperature_start) * progress
        )

    def quantize(self, x: torch.Tensor):
        """
        x:
        - [NT, C] => returns [NT] or [NT, Q]
        - [N, C, T] => returns [N, T] or [N, T, Q]
        """
        if x.dim() == 2:
            tokens = x
            n = t = None
        elif x.dim() == 3:
            tokens, n, t = self._flatten_nct(x)
        else:
            raise ValueError(f"Unsupported input dim for quantize: {x.dim()}")

        if tokens.shape[-1] != self.code_dim:
            raise ValueError(f"Expected last dim={self.code_dim}, got {tokens.shape[-1]}")

        residual = tokens
        all_idx = []
        for level in range(self.num_quantizers):
            logits = self.to_bits[level](residual)
            bits_hard = self._hard_bits(logits)
            idx = self._bits_to_indices(bits_hard)
            quantized = self.from_bits[level](bits_hard)
            residual = residual - quantized
            all_idx.append(idx)

        if self.num_quantizers == 1:
            idx = all_idx[0]
        else:
            idx = torch.stack(all_idx, dim=-1)

        if x.dim() == 3:
            if self.num_quantizers == 1:
                idx = idx.view(n, t)
            else:
                idx = idx.view(n, t, self.num_quantizers)
        return idx

    def dequantize(self, code_idx: torch.Tensor):
        """
        Accepts:
        - [T] (single-level)
        - [T, Q]
        - [B, T] (single-level)
        - [B, T, Q]
        Returns [..., C].
        """
        if code_idx.dim() == 1:
            flat_idx = code_idx.view(-1, 1)
            out_shape = (code_idx.shape[0],)
        elif code_idx.dim() == 2 and (
            self.num_quantizers == 1 or code_idx.shape[-1] != self.num_quantizers
        ):
            flat_idx = code_idx.reshape(-1, 1)
            out_shape = tuple(code_idx.shape)
        elif code_idx.dim() in (2, 3):
            flat_idx = code_idx.reshape(-1, code_idx.shape[-1])
            out_shape = tuple(code_idx.shape[:-1])
        else:
            raise ValueError(f"Unexpected code_idx shape: {tuple(code_idx.shape)}")

        used_levels = flat_idx.shape[-1]
        if used_levels > self.num_quantizers:
            raise ValueError(
                f"Input uses {used_levels} levels, but model has only {self.num_quantizers}"
            )

        quantized = None
        for level in range(used_levels):
            bits = self._indices_to_bits(flat_idx[:, level]).to(self.from_bits[level].weight.dtype)
            q = self.from_bits[level](bits)
            quantized = q if quantized is None else (quantized + q)

        return quantized.view(*out_shape, self.code_dim).contiguous()

    def forward(self, x: torch.Tensor):
        # x: [N, C, T]
        tokens, n, t = self._flatten_nct(x)
        residual = tokens
        quantized_total = torch.zeros_like(tokens)
        commit_losses = []
        perplexities = []
        entropy_losses = []

        temp = max(self.ste_temperature, 1e-4)
        for level in range(self.num_quantizers):
            logits = self.to_bits[level](residual)
            bits_soft = torch.tanh(logits / temp)
            bits_hard = self._hard_bits(logits)
            bits_st = bits_soft + (bits_hard - bits_soft).detach()

            code_idx = self._bits_to_indices(bits_hard)
            quantized = self.from_bits[level](bits_st)
            commit_losses.append(F.mse_loss(residual, quantized.detach()))
            perplexities.append(self._compute_perplexity(code_idx))

            if self.entropy_loss_weight > 0.0:
                global_loss = self._global_bit_balance_loss(logits)
                local_loss = self._local_bit_confidence_loss(logits)
                entropy_losses.append(
                    self.entropy_global_weight * global_loss
                    + self.entropy_local_weight * local_loss
                )

            quantized_total = quantized_total + quantized
            residual = residual - quantized.detach()

        commit = self._reduce(commit_losses)
        if entropy_losses:
            commit = commit + self.entropy_loss_weight * self._reduce(entropy_losses)
        perplexity = self._reduce(perplexities)

        quantized_nct = self._restore_nct(quantized_total, n, t)
        return quantized_nct, commit, perplexity
