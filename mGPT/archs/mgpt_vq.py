# Partially from https://github.com/Mael-zys/T2M-GPT

from typing import Union
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.distribution import Distribution
from .tools.resnet import Resnet1D
from .tools.quantize_lfq import ResidualLFQ


class VQVae(nn.Module):

    def __init__(self,
                 nfeats: int,
                 quantizer: str = "lfq",
                 code_num=512,
                 code_dim=512,
                 num_quantizers=1,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 norm=None,
                 activation: str = "relu",
                 **kwargs) -> None:

        super().__init__()
        self.code_num = code_num
        self.code_dim = code_dim
        self.nfeats = nfeats
        self.num_quantizers = int(num_quantizers)

        self.encoder = Encoder(nfeats,
                               output_emb_width,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)

        self.decoder = Decoder(nfeats,
                               output_emb_width,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)
        self.use_contact = bool(kwargs.get("use_contact", False))
        self.num_contact_classes = int(kwargs.get("num_contact_classes", 3))
        self.contact_head = None
        if self.use_contact:
            self.contact_head = nn.Sequential(
                nn.Conv1d(width, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(128, self.num_contact_classes, kernel_size=1),
            )

        if quantizer != "lfq":
            raise ValueError(
                f"This branch only supports LFQ quantizer, got: {quantizer}. "
                "Please set quantizer='lfq' in config."
            )

        self.quantize_in = (
            nn.Identity()
            if output_emb_width == code_dim
            else nn.Conv1d(output_emb_width, code_dim, kernel_size=1)
        )
        self.quantize_out = (
            nn.Identity()
            if output_emb_width == code_dim
            else nn.Conv1d(code_dim, output_emb_width, kernel_size=1)
        )

        self.quantizer = ResidualLFQ(
            nb_code=code_num,
            code_dim=code_dim,
            num_quantizers=self.num_quantizers,
            aggregate="mean",
            ste_temperature=float(kwargs.get("lfq_ste_temperature", 1.0)),
            ste_temperature_end=float(kwargs.get("lfq_ste_temperature_end", 0.1)),
            entropy_loss_weight=float(kwargs.get("lfq_entropy_loss_weight", 0.1)),
            entropy_global_weight=float(kwargs.get("lfq_entropy_global_weight", 1.0)),
            entropy_local_weight=float(kwargs.get("lfq_entropy_local_weight", 1.0)),
        )

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1)
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, features: Tensor, return_aux: bool = False):
        # Preprocess
        x_in = self.preprocess(features)

        # Encode
        x_encoder = self.encoder(x_in)
        x_encoder = self.quantize_in(x_encoder)
        # print('encoder: ', x_encoder.shape)

        # quantization
        x_quantized, loss, perplexity = self.quantizer(x_encoder)
        x_quantized = self.quantize_out(x_quantized)
        # print('quantized: ', x_quantized.shape)

        # decoder
        if return_aux:
            x_decoder, dec_hidden = self.decoder(x_quantized, return_hidden=True)
        else:
            x_decoder = self.decoder(x_quantized)
            dec_hidden = None
        x_out = self.postprocess(x_decoder)

        if not return_aux:
            return x_out, loss, perplexity

        aux = {"decoder_hidden": dec_hidden}
        if self.contact_head is not None and dec_hidden is not None:
            # [B, 3, T]
            aux["contact_logits"] = self.contact_head(dec_hidden)
        else:
            aux["contact_logits"] = None
        return x_out, loss, perplexity, aux

    def encode(
        self,
        features: Tensor,
    ) -> Union[Tensor, Distribution]:

        N, T, _ = features.shape
        x_in = self.preprocess(features)
        x_encoder = self.encoder(x_in)
        x_encoder = self.quantize_in(x_encoder)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1,
                                                x_encoder.shape[-1])  # (NT, C)
        # print('encoder: ', x_encoder.shape)
        code_idx = self.quantizer.quantize(x_encoder)
        # print('code_idx: ', code_idx.shape)
        if code_idx.dim() == 1:
            code_idx = code_idx.view(N, -1)
        else:
            code_idx = code_idx.view(N, -1, code_idx.shape[-1])

        # latent, dist
        return code_idx, None

    def decode(self, z: Tensor):

        x_d = self.quantizer.dequantize(z)
        if x_d.dim() == 2:
            x_d = x_d.unsqueeze(0)
        elif x_d.dim() == 3:
            pass
        else:
            raise ValueError(f"Unexpected dequantized tensor shape: {tuple(x_d.shape)}")
        x_d = x_d.permute(0, 2, 1).contiguous()
        x_d = self.quantize_out(x_d)

        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out


class Encoder(nn.Module):

    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         activation=activation,
                         norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):

    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        self.input_proj = nn.Conv1d(output_emb_width, width, 3, 1, 1)
        self.input_act = nn.ReLU()
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         reverse_dilation=True,
                         activation=activation,
                         norm=norm), nn.Upsample(scale_factor=2,
                                                 mode='linear',
                                                 align_corners=False),
                nn.Conv1d(width, out_dim, 3, 1, 1))
            blocks.append(block)
        self.upsample_blocks = nn.ModuleList(blocks)
        self.pre_out = nn.Conv1d(width, width, 3, 1, 1)
        self.pre_out_act = nn.ReLU()
        self.out_proj = nn.Conv1d(width, input_emb_width, 3, 1, 1)
        self.hidden_width = width

    def forward(self, x, return_hidden: bool = False):
        x = self.input_act(self.input_proj(x))
        for block in self.upsample_blocks:
            x = block(x)
        hidden = self.pre_out_act(self.pre_out(x))
        out = self.out_proj(hidden)
        if return_hidden:
            return out, hidden
        return out
