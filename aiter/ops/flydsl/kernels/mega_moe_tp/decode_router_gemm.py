# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Router GEMM front end for decode-shape (1-128 row) MoE: tier dispatch over the bf16x3 split-K router kernels
(``decode_router_gemm_direct`` / ``_lines`` / ``_nblocks``) -> FP32 partials [m, split_k, 128].

Ported from amd-kernels' ``moe/mxfp4_h6144_i768_e128_top4/router_gemm.py`` (``split_tier`` / ``nblocks_config`` /
``router_gemm_partials`` dispatch, row range restricted to 1-128) and ``layout.py`` (``router_line_layout``), Apache-2.0,
"Copyright contributors to the amd-kernels project". The split-K tier, the sum order and the per-row-count kernel
choice are the ones amd-kernels measured; the partials are bit-identical.
"""

import torch

from .decode_moe_cfg import RouterCfg, router_partials_default
from .decode_router_common import MAX_BUFFER_BYTES, on_supported_device, run_on_device

H, E = 6144, 128
MAX_ROWS = 128
# (inclusive row bound) -> (split_k, BLOCK_K): the partials' K split and sum order
SPLIT_TIERS = ((64, (48, 64)), (MAX_ROWS, (16, 128)))
# two waves up to 32 rows, 4 / 8 waves (<= 8 / > 8 16-row tiles) above
NBLOCKS_TWO_WAVES_MAX = 32
# router_gemm_direct (no LDS) on the line-layout gate copy: 1-16 rows
DIRECT_LINES_MAX = 16
# router_gemm_nblocks_lines: 17-32 rows
NBLOCKS_LINES_FROM, NBLOCKS_LINES_MAX = 17, 32
# nblocks ``x_direct`` (four waves): 33-57 rows
X_DIRECT_FROM, X_DIRECT_MAX = 33, 57
# split-48 partials are token-major [m, 48, 128] to here (the top-k reads constant split offsets)
TOKEN_MAJOR_MAX = 64


def router_line_layout(weight: torch.Tensor) -> torch.Tensor:
    """FP32 [128, 6144] -> [192, 8, 2, 4, 16, 4]: a wave's 64 lanes read eight complete 128-byte lines. A pure
    permutation, run once at weight loading."""
    assert weight.shape == (E, H) and weight.dtype == torch.float32
    return weight.reshape(8, 16, 192, 4, 2, 4).permute(2, 0, 4, 3, 1, 5).contiguous()


def split_config(num_rows: int):
    """(split_k, BLOCK_K) for ``num_rows`` (1-128), else None."""
    for bound, tier in SPLIT_TIERS:
        if 0 < num_rows <= bound:
            return tier
    return None


def nblocks_config(num_rows: int):
    """(split_k, BLOCK_K, waves, n-blocks per workgroup) of ``decode_router_gemm_nblocks`` for ``num_rows``."""
    tier = split_config(num_rows)
    if tier is None:
        return None
    split_k, block_k = tier
    if num_rows <= NBLOCKS_TWO_WAVES_MAX:
        waves = 2
    else:
        waves = 8 if (num_rows + 15) // 16 > 8 else 4
    return split_k, block_k, waves, 1


def router_gemm_partials(
    hidden_states, router_weight, router_weight_lines=None, cfg=None
):
    """FP32 partials [m, split_k, 128]: contiguous (token-major) to 64 rows, a permuted view of [split_k, m, 128]
    above. None outside the covered shapes (1-128 rows, bf16 [m, 6144] with non-overlapping rows, fp32 [128, 6144]
    gate, all on one gfx950 device), so the caller falls back.

    Args:
        router_weight_lines: ``router_line_layout(router_weight)``, read by the ``direct`` / ``lines`` kernels when
            given (the ``nblocks`` kernel reads ``router_weight``). The two must hold the same gate: the kernels
            cannot check that.
        cfg: a ``decode_moe_cfg.RouterCfg`` (quant == 0), default ``router_partials_default(m)``: the split-K,
            kernel and tile of the router launch. None is returned for a ``direct`` / ``lines`` cfg without
            ``router_weight_lines``.
    """
    m = hidden_states.shape[0]
    if (
        m == 0
        or m > MAX_ROWS
        or hidden_states.dim() != 2
        or hidden_states.shape[1] != H
        or hidden_states.dtype != torch.bfloat16
        or hidden_states.stride(1) != 1
        or hidden_states.stride(0) % 2
        # rows must not overlap (a broadcast row, stride 0, would get an empty buffer descriptor) and the span
        # (the kernels' descriptor spans m * stride * 2 bytes) must fit its 32-bit extent
        or hidden_states.stride(0) < H
        or m * hidden_states.stride(0) * 2 > MAX_BUFFER_BYTES
        or router_weight.shape != (E, H)
        or router_weight.dtype != torch.float32
        or not router_weight.is_contiguous()
        or not on_supported_device(hidden_states, router_weight)
    ):
        return None
    lines_ok = (
        router_weight_lines is not None
        and router_weight_lines.dtype == torch.float32
        and router_weight_lines.is_contiguous()
        and router_weight_lines.numel() == E * H
        and on_supported_device(hidden_states, router_weight_lines)
    )
    if cfg is None:
        cfg = router_partials_default(m)
        if (
            cfg.uses_lines and not lines_ok
        ):  # the default tier falls back to the generic tile
            split_k, block_k = split_config(m)
            cfg = RouterCfg(
                "nblocks",
                split_k,
                block_k,
                nblocks_config(m)[2],
                1,
                0,
                X_DIRECT_FROM <= m <= X_DIRECT_MAX,
            )
    if cfg.quant != 0 or not cfg.fits(m) or (cfg.uses_lines and not lines_ok):
        return None
    split_k, block_k = cfg.split_k, cfg.block_k
    token_major = cfg.token_major(m)
    partials = torch.empty(
        (split_k, m, E), dtype=torch.float32, device=hidden_states.device
    )
    result = partials.view(m, split_k, E) if token_major else partials.permute(1, 0, 2)
    x, p, sx = hidden_states.data_ptr(), partials.data_ptr(), hidden_states.stride(0)
    if cfg.kind == "direct":
        from .decode_router_gemm_direct import _get_kernel

        exe = _get_kernel(split_k, block_k, (m + 15) // 16, 2, True, token_major)
        args = (x, router_weight_lines.data_ptr(), p, m, sx)
    elif cfg.kind == "lines":
        from .decode_router_gemm_lines import _get_kernel

        exe = _get_kernel(token_major)
        args = (x, router_weight_lines.data_ptr(), p, m, sx)
    else:
        from .decode_router_gemm_nblocks import _get_kernel

        exe = _get_kernel(
            split_k, block_k, cfg.waves, cfg.nbw, 0, cfg.x_direct, token_major
        )
        args = (x, router_weight.data_ptr(), p, m, sx, 0, 0, 0, 0)  # no quant outputs
    run_on_device(exe, hidden_states.device, *args)
    return result
