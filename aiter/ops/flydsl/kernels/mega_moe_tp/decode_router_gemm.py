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

from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

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


def router_gemm_partials(hidden_states, router_weight, router_weight_lines=None):
    """FP32 partials [m, split_k, 128]: contiguous (token-major) to 64 rows, a permuted view of [split_k, m, 128]
    above. None outside the covered shapes (1-128 rows, bf16 [m, 6144], fp32 [128, 6144] gate).

    Args:
        router_weight_lines: the gate in ``router_line_layout``, used by the 1-32 row tiers when given.
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
        or router_weight.shape != (E, H)
        or router_weight.dtype != torch.float32
        or not router_weight.is_contiguous()
    ):
        return None
    split_k, _ = split_config(m)
    token_major = m <= TOKEN_MAJOR_MAX
    partials = torch.empty(
        (split_k, m, E), dtype=torch.float32, device=hidden_states.device
    )
    result = partials.view(m, split_k, E) if token_major else partials.permute(1, 0, 2)
    stream = torch.cuda.current_stream()
    lines_ok = (
        router_weight_lines is not None
        and router_weight_lines.dtype == torch.float32
        and router_weight_lines.is_contiguous()
        and router_weight_lines.numel() == E * H
    )
    if lines_ok and m <= DIRECT_LINES_MAX:
        from .decode_router_gemm_direct import _get_kernel

        _run_compiled(
            _get_kernel(48, 64, (m + 15) // 16, 2, True, token_major),
            hidden_states.data_ptr(),
            router_weight_lines.data_ptr(),
            partials.data_ptr(),
            m,
            hidden_states.stride(0),
            stream,
        )
    elif lines_ok and NBLOCKS_LINES_FROM <= m <= NBLOCKS_LINES_MAX:
        from .decode_router_gemm_lines import _get_kernel

        _run_compiled(
            _get_kernel(token_major),
            hidden_states.data_ptr(),
            router_weight_lines.data_ptr(),
            partials.data_ptr(),
            m,
            hidden_states.stride(0),
            stream,
        )
    else:
        from .decode_router_gemm_nblocks import _get_kernel

        x_direct = X_DIRECT_FROM <= m <= X_DIRECT_MAX
        _run_compiled(
            _get_kernel(*nblocks_config(m), 0, x_direct, token_major),
            hidden_states.data_ptr(),
            router_weight.data_ptr(),
            partials.data_ptr(),
            m,
            hidden_states.stride(0),
            0,
            0,
            0,
            0,
            stream,
        )
    return result
