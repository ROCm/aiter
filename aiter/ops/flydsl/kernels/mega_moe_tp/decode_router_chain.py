# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Decode-shape (1-128 row) MoE routing front end for E128 top-4 at H6144: the split-K router GEMM, the biased
sigmoid top-4 and the MXFP4 + MXFP8 input quant in one or two launches.

Ported from amd-kernels' ``moe/mxfp4_h6144_i768_e128_top4/chain.py`` (``_route_quant_small``) and
``router_gemm_topk_mxfp4_mxfp8_quant.py``, Apache-2.0, "Copyright contributors to the amd-kernels project".
"""

import struct

import torch

from .decode_router_common import on_supported_device, run_on_device
from .decode_router_gemm import (
    MAX_ROWS,
    nblocks_config,
    router_gemm_partials,
)
from .decode_router_gemm_nblocks import _get_kernel as _get_router
from .decode_router_topk_quant import splitk_biased_topk_mxfp4_mxfp8_quant
from .decode_router_topk_sorted import _get_kernel as _get_topk

H, E, TOPK = 6144, 128, 4
# (first row, last row, quant placement: 1 appended / 2 prepended to the router launch)
QUANT_TIERS = ((58, 64, 1), (65, MAX_ROWS, 2))


def router_quant_placement(m: int) -> int:
    """0 (the quant stays in the top-k launch), 1 appended, 2 prepended."""
    for lo, hi, placement in QUANT_TIERS:
        if lo <= m <= hi:
            return placement
    return 0


def _router_gemm_topk_quant(x, router_weight, correction_bias, routed_scaling_factor):
    m = x.shape[0]
    config = nblocks_config(m)
    split = config[0]
    token_major = True  # split-48 (<= 64 rows) and the 65-128 row split-16 tier both write token-major
    dev = x.device
    partials = torch.empty((split, m, E), dtype=torch.float32, device=dev)
    ids = torch.empty((m, TOPK), dtype=torch.int32, device=dev)
    weights = torch.empty((m, TOPK), dtype=torch.float32, device=dev)
    q = torch.empty((m, H // 2), dtype=torch.uint8, device=dev)
    s = torch.empty((m, H // 32), dtype=torch.uint8, device=dev)
    q8 = torch.empty((m, H), dtype=torch.uint8, device=dev)
    s8 = torch.empty((m, H // 32), dtype=torch.uint8, device=dev)
    run_on_device(
        _get_router(*config, router_quant_placement(m), False, token_major),
        dev,
        x.data_ptr(),
        router_weight.data_ptr(),
        partials.data_ptr(),
        m,
        x.stride(0),
        q.data_ptr(),
        s.data_ptr(),
        q8.data_ptr(),
        s8.data_ptr(),
    )
    run_on_device(
        _get_topk(E, TOPK, split, True) if token_major else _get_topk(E, TOPK, split),
        dev,
        partials.data_ptr(),
        correction_bias.data_ptr(),
        weights.data_ptr(),
        ids.data_ptr(),
        partials.numel() * 4,
        m,
        split * E if token_major else E,
        E if token_major else m * E,
        TOPK,
        struct.unpack("<i", struct.pack("<f", routed_scaling_factor))[0],
    )
    return (q, s), (q8, s8), weights, ids


def route_quant_small(
    x, router_weight, router_weight_lines, correction_bias, routed_scaling_factor=2.0
):
    """1-128 rows: router GEMM -> top-4 + MXFP4 / MXFP8 input quant ->
    ((q4, s4), (q8, s8), topk weights, topk ids); None outside the covered inputs, so the caller falls back.

    Args:
        x: bf16 [m, 6144], contiguous, on a gfx950 device.
        router_weight: fp32 [128, 6144]; router_weight_lines: ``router_line_layout(router_weight)`` or None.
        correction_bias: fp32 [128]. All three on x's device.
    """
    if not (
        x.dim() == 2
        and 0 < x.shape[0] <= MAX_ROWS
        and x.shape[1] == H
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and router_weight.shape == (E, H)
        and router_weight.dtype == torch.float32
        and router_weight.is_contiguous()
        and correction_bias.shape == (E,)
        and correction_bias.dtype == torch.float32
        and correction_bias.is_contiguous()
        and on_supported_device(x, router_weight, correction_bias)
    ):
        return None
    n = x.shape[0]
    if router_quant_placement(n):
        return _router_gemm_topk_quant(
            x, router_weight, correction_bias, routed_scaling_factor
        )
    parts = router_gemm_partials(x, router_weight, router_weight_lines)
    if (
        parts is None
    ):  # e.g. a router_weight_lines that is not usable: not an error of the covered inputs
        return None
    ids = torch.empty((n, TOPK), dtype=torch.int32, device=x.device)
    wt = torch.empty((n, TOPK), dtype=torch.float32, device=x.device)
    q4, q8 = splitk_biased_topk_mxfp4_mxfp8_quant(
        parts, correction_bias, wt, ids, routed_scaling_factor, x
    )
    return q4, q8, wt, ids
