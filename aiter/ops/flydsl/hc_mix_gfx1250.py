# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 FlyDSL hyper-connection mix (Qwen3.8-Flash-Next): torch-facing wrapper."""

from __future__ import annotations

import functools

import torch

# (max M, (tile_m, tile_n, tile_k, m_warp, n_warp)). Measured on MI455X for
# H=2560, HC=4, R=320 with the weights cold, as across layers in a forward:
# the fused kernel replaces scaled SiLU + up GEMM + gated mean up to M=48 and
# loses to them at M=64.
_CONFIGS = (
    (16, (16, 128, 64, 1, 4)),
    (48, (16, 64, 64, 1, 4)),
)
HC_SILU_UP_MIX_MAX_M = _CONFIGS[-1][0]


@functools.cache
def _launcher(H, HC, R, tile_m, tile_n, tile_k, m_warp, n_warp):
    from aiter.ops.flydsl.kernels.hc_mix_gfx1250 import compile_hc_up_mix

    return compile_hc_up_mix(
        H=H,
        HC=HC,
        R=R,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=m_warp,
        n_warp=n_warp,
        silu_scale=HC,
    )


def hc_interleave_up_weight(w_up: torch.Tensor, hc: int) -> torch.Tensor:
    """``[HC*H, R]`` up weight -> the stream-minor layout `hc_silu_up_mix` reads."""
    from aiter.ops.flydsl.kernels.hc_mix_gfx1250 import hc_interleave_up_weight

    return hc_interleave_up_weight(w_up, hc)


def hc_silu_up_mix(
    d: torch.Tensor,
    w_up_interleaved: torch.Tensor,
    normed: torch.Tensor,
    hc: int,
) -> torch.Tensor:
    """``mean_s(sigmoid(scaled_silu(d[:, :R], hc) @ w_up.T) * normed)`` in one launch.

    ``d`` is the ``[M, >= R]`` down-projection output (only its first ``R``
    columns are read), ``w_up_interleaved`` the ``[HC*H, R]`` up weight from
    :func:`hc_interleave_up_weight`, ``normed`` the ``[M, HC*H]`` normalized
    streams. Returns ``[M, H]``. ``M`` must be at most ``HC_SILU_UP_MIX_MAX_M``.
    """
    import flydsl.expr as fx

    from aiter.ops.flydsl.kernels.tensor_shim import (
        _run_compiled,
        current_stream_arg,
        ptr_arg,
    )

    m = d.shape[0]
    n, r = w_up_interleaved.shape
    hidden = n // hc
    if m > HC_SILU_UP_MIX_MAX_M:
        raise ValueError(
            f"hc_silu_up_mix supports M <= {HC_SILU_UP_MIX_MAX_M}, got {m}"
        )
    for name, t in (
        ("d", d),
        ("w_up_interleaved", w_up_interleaved),
        ("normed", normed),
    ):
        if t.dtype != torch.bfloat16 or t.stride(-1) != 1:
            raise ValueError(f"{name} must be BF16 with unit inner stride")
    if normed.shape != (m, n) or d.shape[1] < r:
        raise ValueError(
            f"shape mismatch: d {tuple(d.shape)}, w_up {tuple(w_up_interleaved.shape)}, "
            f"normed {tuple(normed.shape)}"
        )
    cfg = next(c for max_m, c in _CONFIGS if m <= max_m)
    mixed = torch.empty((m, hidden), dtype=d.dtype, device=d.device)
    if m == 0:
        return mixed

    def bp(t):
        return ptr_arg(t, fx.BFloat16)

    _run_compiled(
        _launcher(hidden, hc, r, *cfg),
        bp(mixed),
        bp(d),
        bp(w_up_interleaved),
        bp(normed),
        m,
        mixed.stride(0),
        d.stride(0),
        normed.stride(0),
        current_stream_arg(),
    )
    return mixed


__all__ = ["HC_SILU_UP_MIX_MAX_M", "hc_interleave_up_weight", "hc_silu_up_mix"]
