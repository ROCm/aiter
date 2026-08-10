# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""
chunk_delta_attn – Triton kernels for the chunk-based delta-attention
forward pass (prefill / training).

Public API
----------
chunk_delta_attn_fwd          Top-level forward: gate cumsum → intra → inter → output.
chunk_delta_attn_fwd_intra    Intra-chunk attention (Aqk/Akk + W/U recompute).
chunk_gla_fwd_o               GLA output kernel (q*exp2(g)*h + A*v).
recompute_w_u_fwd             W/U recompute from Akk inverse.
chunk_gate_cumsum             Fused gate (A_log/softplus/sigmoid) + chunk cumsum.
beta_sigmoid_fwd              Elementwise sigmoid for beta gate.
chunk_delta_attn_gate_fwd     Per-token gate without cumsum (forward only).
"""

from .chunk_fwd import chunk_delta_attn_fwd
from .gate import beta_sigmoid_fwd, chunk_delta_attn_gate_fwd
from .gla_output import chunk_gla_fwd_o
from .intra_attn import chunk_delta_attn_fwd_intra
from .utils.cumsum import chunk_gate_cumsum
from .wy_fast import recompute_w_u_fwd

__all__ = [
    "beta_sigmoid_fwd",
    "chunk_delta_attn_fwd",
    "chunk_delta_attn_fwd_intra",
    "chunk_delta_attn_gate_fwd",
    "chunk_gate_cumsum",
    "chunk_gla_fwd_o",
    "recompute_w_u_fwd",
]
