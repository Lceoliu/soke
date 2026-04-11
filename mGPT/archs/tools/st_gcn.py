"""Spatial-Temporal Graph Convolutional Network (ST-GCN) components for pose VAE.

Reference: Yan et al., "Spatial Temporal Graph Convolutional Networks for
Skeleton-Based Action Recognition", AAAI 2018.

Adapted for SMPL-X axis-angle pose encoding in the SOKE VAE.

Joint layouts assumed (axis-angle, 3 dims per joint):

  Body VAE input (43-dim = 30 body joints + 13 face):
    [0:30]  = 10 upper-body joints × 3:
              0-Neck, 1-L_Collar, 2-R_Collar, 3-Head,
              4-L_Shoulder, 5-R_Shoulder, 6-L_Elbow,
              7-R_Elbow, 8-L_Wrist, 9-R_Wrist
    [30:43] = jaw (3) + face expression (10) — processed separately

  Hand VAE input (45-dim = 15 hand joints × 3):
    0-2:   Index_1/2/3 (base to tip)
    3-5:   Middle_1/2/3
    6-8:   Pinky_1/2/3
    9-11:  Ring_1/2/3
    12-14: Thumb_1/2/3
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import Resnet1D


# ---------------------------------------------------------------------------
# Adjacency matrices (normalized)
# ---------------------------------------------------------------------------

def _normalize_adj(A: torch.Tensor) -> torch.Tensor:
    """Row-normalize adjacency matrix: D^{-1} A."""
    D = A.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return A / D


def _build_body_adj() -> torch.Tensor:
    """Bidirectional adjacency for 10 upper-body joints with self-connections.

    Kinematic tree (parent → child):
        Neck(0) → L_Collar(1), R_Collar(2), Head(3)
        L_Collar(1) → L_Shoulder(4)
        R_Collar(2) → R_Shoulder(5)
        L_Shoulder(4) → L_Elbow(6)
        R_Shoulder(5) → R_Elbow(7)
        L_Elbow(6) → L_Wrist(8)
        R_Elbow(7) → R_Wrist(9)
    """
    N = 10
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (2, 5),
        (4, 6), (5, 7),
        (6, 8), (7, 9),
    ]
    A = torch.zeros(N, N)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    A = A + torch.eye(N)  # self-connections
    return _normalize_adj(A)


def _build_hand_adj() -> torch.Tensor:
    """Bidirectional adjacency for 15 hand joints with self-connections.

    Five fingers × 3 joints each:
        Index:  0→1→2
        Middle: 3→4→5
        Pinky:  6→7→8
        Ring:   9→10→11
        Thumb:  12→13→14

    Palm connections: all finger bases (0,3,6,9,12) are mutually connected
    (they all connect through the palm/wrist which is not in the input).
    """
    N = 15
    finger_bases = [0, 3, 6, 9, 12]  # base joint of each finger
    edges = []
    # Finger chains
    for b in finger_bases:
        edges.append((b, b + 1))
        edges.append((b + 1, b + 2))
    # Palm connections (all bases to each other)
    for i in range(len(finger_bases)):
        for j in range(i + 1, len(finger_bases)):
            edges.append((finger_bases[i], finger_bases[j]))
    A = torch.zeros(N, N)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    A = A + torch.eye(N)  # self-connections
    return _normalize_adj(A)


# Pre-built normalized adjacency matrices
BODY_ADJ: torch.Tensor = _build_body_adj()   # [10, 10]
HAND_ADJ: torch.Tensor = _build_hand_adj()   # [15, 15]


# ---------------------------------------------------------------------------
# Spatial graph convolution
# ---------------------------------------------------------------------------

class SpatialGraphConv(nn.Module):
    """Graph convolutional layer operating on joint dimension.

    Input:  [B, N_joints, C_in, T]
    Output: [B, N_joints, C_out, T]

    The adjacency matrix A is normalized and registered as a buffer.
    Conv weights W act on C_in → C_out per joint (shared across joints).
    """

    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor):
        super().__init__()
        self.register_buffer("A", A)          # [N, N]
        self.W = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_uniform_(self.W.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C_in, T]
        B, N, C, T = x.shape
        # Apply linear weight across C dimension: [B, T, N, C_in] → [B, T, N, C_out]
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, T, N, C_in]
        x = self.W(x)                            # [B, T, N, C_out]
        # Graph aggregation: sum over neighbors via A [N, N]
        # [B, T, N, C_out] → aggregate over N dim with A
        x = torch.einsum("mn,btnc->btmc", self.A, x)  # [B, T, N, C_out]
        x = x + self.bias  # broadcast bias
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, N, C_out, T]
        return x


# ---------------------------------------------------------------------------
# ST-GCN block: spatial GCN + temporal conv with residual
# ---------------------------------------------------------------------------

class STGCNBlock(nn.Module):
    """One ST-GCN block: spatial GCN → BN → activation → temporal ResNet.

    Input:  [B, N_joints, C_in, T]
    Output: [B, N_joints, C_out, T]  (T unchanged — no temporal downsampling here)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        temporal_kernel_size: int = 9,
        activation: str = "relu",
    ):
        super().__init__()
        self.spatial_gcn = SpatialGraphConv(in_channels, out_channels, A)
        self.bn = nn.BatchNorm1d(out_channels)  # applied per-joint
        self.act = nn.ReLU() if activation == "relu" else nn.GELU()

        # Temporal conv per joint (shared weights, acts on T dimension)
        padding = (temporal_kernel_size - 1) // 2
        self.temporal_conv = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=temporal_kernel_size,
            padding=padding,
        )
        self.temporal_bn = nn.BatchNorm1d(out_channels)

        # Residual projection if channels change
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C_in, T]
        B, N, C, T = x.shape

        # Residual branch (per-joint: [B*N, C_in, T] → [B*N, C_out, T])
        x_flat = x.reshape(B * N, C, T)
        res = self.residual(x_flat)  # [B*N, C_out, T]

        # Spatial graph convolution
        out = self.spatial_gcn(x)  # [B, N, C_out, T]
        out_flat = out.reshape(B * N, -1, T)
        out_flat = self.bn(out_flat)
        out_flat = self.act(out_flat)

        # Temporal convolution (per joint, shared weights)
        out_flat = self.temporal_conv(out_flat)  # [B*N, C_out, T]
        out_flat = self.temporal_bn(out_flat)

        # Residual + activation
        out_flat = self.act(out_flat + res)
        return out_flat.reshape(B, N, -1, T)


# ---------------------------------------------------------------------------
# ST-GCN Encoder
# ---------------------------------------------------------------------------

class STGCNEncoder(nn.Module):
    """Drop-in replacement for the Conv1d ``Encoder`` in ``mgpt_vq.py``.

    Input:  [B, input_emb_width, T]  (temporal-first after VAE preprocess)
    Output: [B, output_emb_width, T']

    Architecture:
        1. Split joint dims from extra dims (jaw+expr for body VAE).
        2. Reshape joint dims → [B, N_joints, feat_per_joint, T].
        3. N_gcn_layers of STGCNBlock (spatial + temporal mixing).
        4. Mean-pool over joint dimension → [B, gcn_out, T].
        5. (Optional) concatenate with extra-dim Conv1d features.
        6. Temporal downsampling via strided Conv1d (same as original Encoder).
        7. Output projection Conv1d.
    """

    def __init__(
        self,
        input_emb_width: int,
        output_emb_width: int,
        down_t: int,
        stride_t: int,
        width: int,
        depth: int,
        dilation_growth_rate: int = 3,
        activation: str = "relu",
        norm: Optional[str] = None,
        # ST-GCN specific
        num_joints: int = 15,
        feat_per_joint: int = 3,
        adj: Optional[str] = None,          # "body" | "hand" | None (fully-connected)
        n_gcn_layers: int = 2,
        gcn_hidden: int = 64,
        gcn_out: Optional[int] = None,      # None → width // 2 if extra_dims else width
        temporal_kernel_size: int = 9,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.feat_per_joint = feat_per_joint
        joint_dims = num_joints * feat_per_joint
        self.extra_dims = input_emb_width - joint_dims
        assert self.extra_dims >= 0, (
            f"input_emb_width={input_emb_width} < num_joints*feat_per_joint="
            f"{joint_dims}. Check num_joints / feat_per_joint."
        )

        # Adjacency matrix
        if adj == "body":
            A = BODY_ADJ
        elif adj == "hand":
            A = HAND_ADJ
        else:
            # Fully connected (learnable-style: uniform weights)
            N = num_joints
            A = torch.ones(N, N) / N
        assert A.shape == (num_joints, num_joints), (
            f"Adjacency shape mismatch: expected ({num_joints},{num_joints}), got {tuple(A.shape)}"
        )

        # Compute gcn_out
        if gcn_out is None:
            gcn_out = width // 2 if self.extra_dims > 0 else width

        # ST-GCN layers
        gcn_layers = []
        in_ch = feat_per_joint
        for layer_idx in range(n_gcn_layers):
            out_ch = gcn_hidden if layer_idx < n_gcn_layers - 1 else gcn_out
            gcn_layers.append(STGCNBlock(in_ch, out_ch, A, temporal_kernel_size, activation))
            in_ch = out_ch
        self.gcn_layers = nn.ModuleList(gcn_layers)
        self.gcn_out = gcn_out

        # Extra dims processing (jaw + expression in body VAE)
        extra_out = width - gcn_out if self.extra_dims > 0 else 0
        if self.extra_dims > 0:
            self.extra_conv = nn.Sequential(
                nn.Conv1d(self.extra_dims, extra_out, kernel_size=3, padding=1),
                nn.BatchNorm1d(extra_out),
                nn.ReLU() if activation == "relu" else nn.GELU(),
            )
            combined_width = gcn_out + extra_out
        else:
            self.extra_conv = None
            combined_width = gcn_out

        # Projection to width if combined_width != width
        if combined_width != width:
            self.combine_proj = nn.Sequential(
                nn.Conv1d(combined_width, width, kernel_size=1),
                nn.ReLU() if activation == "relu" else nn.GELU(),
            )
        else:
            self.combine_proj = nn.Identity()

        # Temporal downsampling (same structure as original Encoder)
        temporal_blocks = []
        filter_t = stride_t * 2
        pad_t = stride_t // 2
        for _ in range(down_t):
            block = nn.Sequential(
                nn.Conv1d(width, width, filter_t, stride_t, pad_t),
                Resnet1D(
                    width,
                    depth,
                    dilation_growth_rate,
                    activation=activation,
                    norm=norm,
                ),
            )
            temporal_blocks.append(block)
        temporal_blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.temporal = nn.Sequential(*temporal_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, input_emb_width, T]"""
        B, D, T = x.shape

        # --- 1. Split joint vs. extra features ---
        joint_dims = self.num_joints * self.feat_per_joint
        x_joints = x[:, :joint_dims, :]  # [B, N*C, T]

        # Reshape to [B, N_joints, feat_per_joint, T]
        x_joints = x_joints.view(B, self.num_joints, self.feat_per_joint, T)

        # --- 2. ST-GCN layers ---
        for gcn_block in self.gcn_layers:
            x_joints = gcn_block(x_joints)  # [B, N, gcn_ch, T]

        # --- 3. Mean pool over joint dimension ---
        x_spatial = x_joints.mean(dim=1)  # [B, gcn_out, T]

        # --- 4. Extra dims ---
        if self.extra_dims > 0:
            x_extra = x[:, joint_dims:, :]  # [B, extra_dims, T]
            x_extra = self.extra_conv(x_extra)  # [B, extra_out, T]
            x_combined = torch.cat([x_spatial, x_extra], dim=1)  # [B, width, T]
        else:
            x_combined = x_spatial

        # --- 5. Projection to width ---
        x_combined = self.combine_proj(x_combined)  # [B, width, T]

        # --- 6. Temporal downsampling ---
        return self.temporal(x_combined)  # [B, output_emb_width, T']
