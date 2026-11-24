# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Router GEMM front end for decode-shape (1-32 row) MoE: tier dispatch over the bf16x3 split-K router kernels
(``decode_router_gemm_direct`` / ``_lines``) -> FP32 partials [m, split_k, 128].

Ported from amd-kernels' ``moe/mxfp4_h6144_i768_e128_top4/router_gemm.py`` (``split_tier`` / ``nblocks_config`` /
``router_gemm_partials`` dispatch, row range restricted to 1-32) and ``layout.py`` (``router_line_layout``), Apache-2.0,
"Copyright contributors to the amd-kernels project". The split-K tier, the sum order and the per-row-count kernel
choice are the ones amd-kernels measured; the partials are bit-identical.
"""

import torch

from .decode_router_common import MAX_BUFFER_BYTES, on_supported_device, run_on_device

H, E = 6144, 128
MAX_ROWS = 32
# (inclusive row bound) -> (split_k, BLOCK_K): the partials' K split and sum order
SPLIT_TIERS = ((MAX_ROWS, (48, 64)),)
# router_gemm_direct (no LDS) on the line-layout gate copy: 1-16 rows
DIRECT_LINES_MAX = 16
# router_gemm_nblocks_lines: from 17 rows
NBLOCKS_LINES_FROM = 17
# split-48 partials are token-major [m, 48, 128] to here (the top-k reads constant split offsets)
TOKEN_MAJOR_MAX = 64


def router_line_layout(weight: torch.Tensor) -> torch.Tensor:
    """FP32 [128, 6144] -> [192, 8, 2, 4, 16, 4]: a wave's 64 lanes read eight complete 128-byte lines. A pure
    permutation, run once at weight loading."""
    assert weight.shape == (E, H) and weight.dtype == torch.float32
    return weight.reshape(8, 16, 192, 4, 2, 4).permute(2, 0, 4, 3, 1, 5).contiguous()


def split_config(num_rows: int):
    """(split_k, BLOCK_K) for ``num_rows`` (1-32), else None."""
    for bound, tier in SPLIT_TIERS:
        if 0 < num_rows <= bound:
            return tier
    return None


def router_gemm_partials(hidden_states, router_weight, router_weight_lines=None):
    """FP32 partials [m, split_k, 128]: contiguous (token-major) to 64 rows, a permuted view of [split_k, m, 128]
    above. None outside the covered shapes (1-32 rows, bf16 [m, 6144] with non-overlapping rows, fp32 [128, 6144]
    gate, all on one gfx950 device), so the caller falls back.

    Args:
        router_weight: the fp32 gate; only validated here, the 1-32 row kernels read ``router_weight_lines``.
        router_weight_lines: ``router_line_layout(router_weight)`` (required). The two must hold the same gate: the
            kernels cannot check that.
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
        or router_weight_lines is None
        or router_weight_lines.dtype != torch.float32
        or not router_weight_lines.is_contiguous()
        or router_weight_lines.numel() != E * H
        or not on_supported_device(hidden_states, router_weight, router_weight_lines)
    ):
        return None
    split_k, block_k = split_config(m)
    token_major = m <= TOKEN_MAJOR_MAX
    partials = torch.empty(
        (split_k, m, E), dtype=torch.float32, device=hidden_states.device
    )
    result = partials.view(m, split_k, E) if token_major else partials.permute(1, 0, 2)
    if m <= DIRECT_LINES_MAX:
        from .decode_router_gemm_direct import _get_kernel

        exe = _get_kernel(split_k, block_k, (m + 15) // 16, 2, True, token_major)
    else:
        from .decode_router_gemm_lines import _get_kernel

        exe = _get_kernel(token_major)
    run_on_device(
        exe,
        hidden_states.device,
        hidden_states.data_ptr(),
        router_weight_lines.data_ptr(),
        partials.data_ptr(),
        m,
        hidden_states.stride(0),
    )
    return result
