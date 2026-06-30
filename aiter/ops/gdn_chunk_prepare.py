# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
from ..jit.core import compile_ops


@compile_ops("module_gdn_chunk_prepare")
def gdn_chunk_prepare_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    w_bar: torch.Tensor,
    u_bar: torch.Tensor,
    g_cumsum: torch.Tensor,
) -> None: ...


def gdn_chunk_prepare(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    BT: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused intra-chunk GDN prefill preparation (forward only).

    Fuses the four FLA prefill kernels that run before the h-recurrence:
      chunk_local_cumsum + chunk_scaled_dot_kkt_fwd + solve_tril + recompute_w_u_fwd
    into a single token-parallel HIP kernel (BT=64, K=V=128, gfx942/gfx950).

    Args:
        k: [B, T, H, K] bf16
        v: [B, T, H, V] bf16
        g: [B, T, H] — log-gate (cast to fp32)
        beta: [B, T, H] — sigmoid gating (cast to fp32)
        BT: chunk size (64).

    Returns:
        (w_bar, u_bar, g_cumsum):
          w_bar    [B, T, H, K] bf16  — C @ (k * beta * exp(g_cumsum))
          u_bar    [B, T, H, V] bf16  — C @ (v * beta)
          g_cumsum [B, T, H]    fp32  — inclusive prefix sum of g within each chunk
    """
    B, T, H, K = k.shape
    V = v.shape[-1]

    k = k.contiguous().to(torch.bfloat16)
    v = v.contiguous().to(torch.bfloat16)
    g = g.contiguous().float()
    beta = beta.contiguous().float()

    pad_len = (BT - T % BT) % BT
    if pad_len > 0:
        k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, 0, 0, pad_len))
        g = F.pad(g, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, 0, 0, pad_len))

    Tp = k.shape[1]
    opts_bf16 = dict(dtype=torch.bfloat16, device=k.device)
    opts_fp32 = dict(dtype=torch.float32, device=k.device)
    w_bar = torch.empty(B, Tp, H, K, **opts_bf16)
    u_bar = torch.empty(B, Tp, H, V, **opts_bf16)
    g_cumsum = torch.empty(B, Tp, H, **opts_fp32)

    gdn_chunk_prepare_fwd(k, v, g, beta, w_bar, u_bar, g_cumsum)

    if pad_len > 0:
        w_bar = w_bar[:, :T]
        u_bar = u_bar[:, :T]
        g_cumsum = g_cumsum[:, :T]
    return w_bar, u_bar, g_cumsum
