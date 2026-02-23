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
            entropy_loss_weight=float(kwargs.get("lfq_entropy_loss_weight", 0.0)),
        )

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1)
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, features: Tensor):
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
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)

        return x_out, loss, perplexity

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
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         reverse_dilation=True,
                         activation=activation,
                         norm=norm), nn.Upsample(scale_factor=2,
                                                 mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1))
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
