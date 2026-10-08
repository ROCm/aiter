# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""fmha v3 forward (hd128, bf16, non-causal, no dropout) with RMSNorm + interleaved RoPE applied to Q on load.

``out, lse = fmha_v3_fwd_qnorm_rope(q, k, v, softmax_scale, wq_a, tab_a, wq_b, tab_b, ntile_a, eps, q_n, q_rstd)``

* ``q``: the RAW query as a bshd view [B, S, H, 128] (any strides, ``stride(-1) == 1``), e.g. the q columns of a
  fused QKV projection ``mixed_qkv.view(S, B, H, 3 * D)[..., :D].transpose(0, 1)``.
* ``k``, ``v``: bshd views [B, S, H, 128] as for the plain v3 forward (already final; no transform).
* Per Q row: ``rstd = rsqrt(mean(x^2) + eps)``, ``n = (x * rstd) * w``, then interleaved RoPE
  ``(n[2i], n[2i+1]) -> (n[2i] c_i - n[2i+1] s_i, n[2i+1] c_i + n[2i] s_i)`` in fp32 with one bf16 rounding.
* Two streams along the sequence: 256-row Q tiles ``[0, ntile_a)`` use ``(wq_a, tab_a)``, the rest ``(wq_b, tab_b)``;
  a single stream passes ``ntile_a = S // 256`` (``wq_b`` / ``tab_b`` may then repeat ``wq_a`` / ``tab_a``).
* ``wq_*``: bf16 [128]. ``tab_*``: bf16 [S_stream * B, 128], row ``s_local * B + b`` = ``[cos_i (64) | sin_i (64)]``,
  i.e. ``torch.cat([cos[:, 0::2], sin[:, 0::2]], -1)`` of a pair-repeated (interleaved) [S_stream * B, 128] table.
* ``q_rstd`` (written): fp32 [S * B * H], index ``(s * B + b) * H + h``.
* ``q_n`` (written, optional): the normalized + rotated q as a bshd view of an sbhd-contiguous [S, B, H, 128] bf16
  buffer (the q the v3 backward takes). ``None`` skips the write.
* Returns ``out`` (``torch.empty(S, B, H, 128).permute(1, 0, 2, 3)``) and ``lse`` (fp32 [B, H, S]).

``fmha_v3_fwd_qnorm_rope_ok(b, s, h, d, ntile_a)`` says whether a shape is supported (pure arithmetic).
"""

import torch
from torch import Tensor

from ..jit.core import compile_ops

__all__ = ["fmha_v3_fwd_qnorm_rope", "fmha_v3_fwd_qnorm_rope_ok"]

_TILE = 256


@compile_ops("module_fmha_v3_fwd_qnorm_rope_asm", fc_name="fmha_v3_fwd_qnorm_rope_asm", ffi_type="ctypes")
def _fmha_v3_fwd_qnorm_rope_asm(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    lse: Tensor,
    w_a: Tensor,
    tab_a: Tensor,
    w_b: Tensor,
    tab_b: Tensor,
    q_rstd: Tensor,
    q_n: Tensor | None,
    softmax_scale: float,
    eps: float,
    ntile_a: int,
) -> None: ...


def fmha_v3_fwd_qnorm_rope_ok(b: int, s: int, h: int, d: int, ntile_a: int) -> bool:
    """Whether the fused forward supports [b, s, h, d] with ``ntile_a`` leading 256-row tiles in stream a."""
    return d == 128 and b > 0 and h > 0 and s > 0 and s % _TILE == 0 and 0 <= ntile_a <= s // _TILE


def fmha_v3_fwd_qnorm_rope(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    softmax_scale: float,
    wq_a: Tensor,
    tab_a: Tensor,
    wq_b: Tensor,
    tab_b: Tensor,
    ntile_a: int,
    eps: float,
    q_n: Tensor | None,
    q_rstd: Tensor,
) -> tuple[Tensor, Tensor]:
    B, S, H, D = q.shape
    out = torch.empty((S, B, H, D), dtype=q.dtype, device=q.device).permute(1, 0, 2, 3)
    lse = torch.empty((B, H, S), dtype=torch.float32, device=q.device)
    _fmha_v3_fwd_qnorm_rope_asm(
        q, k, v, out, lse, wq_a, tab_a, wq_b, tab_b, q_rstd, q_n, float(softmax_scale), float(eps), int(ntile_a)
    )
    return out, lse
