# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 decode MoE front: ``flydsl_moe_router_topk`` runs ``topk_gating(
tgemm.mm(x, w), bias, score_func="sigmoid")`` in one FlyDSL launch, where the
tuned table says it is faster (``router_topk_config``).
"""

from __future__ import annotations

import csv
import functools

import torch

from aiter.ops.flydsl.kernels.tensor_shim import (
    _run_compiled,
    current_stream_arg,
    ptr_arg,
)

ROUTER_EXPERTS = 128
_TILE = 16


@functools.lru_cache(maxsize=1)
def _tuned_rows() -> dict:
    from aiter.jit.core import AITER_CONFIGS

    with open(AITER_CONFIGS.AITER_CONFIG_MOE_ROUTER_TOPK_FILE, newline="") as f:
        rows = list(csv.DictReader(f))
    key = ("cu_num", "M", "N", "K", "topk")
    cfg = ("split_k", "prefetch", "bitexact")
    return {
        (r["gfx"], *(int(r[c]) for c in key)): (
            tuple(int(r[c]) for c in cfg) if r["fused"] == "1" else ()
        )
        for r in rows
    }


@functools.lru_cache(maxsize=4096)
def router_topk_config(
    m: int, n: int, k: int, topk: int, with_counts: bool = False
) -> dict | None:
    """``split_k`` / ``prefetch`` of ``flydsl_moe_router_topk`` for this call, or
    None to keep ``tgemm.mm`` + ``topk_gating``: the tuned table
    (``AITER_CONFIG_MOE_ROUTER_TOPK``, padded M) has no fused row, a bitexact
    row's split_k is no longer the router GEMM's, or ``with_counts`` (rows are
    tuned without ``group_counts``)."""
    if with_counts:
        return None
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
    from aiter.ops.gemm_op_common import find_padded_m_row

    gfx, cu_num = get_gfx_runtime(), get_cu_num()
    row, _ = find_padded_m_row(
        _tuned_rows(), lambda pm: (gfx, cu_num, pm, n, k, topk), m, n, k
    )
    if not row or (row[2] and row[0] != router_gemm_split_k(m, n, k)):
        return None
    return {"split_k": row[0], "prefetch": row[1]}


@functools.lru_cache(maxsize=4096)
def router_gemm_split_k(m: int, n: int, k: int) -> int | None:
    """split_k of the gfx1250 FlyDSL A16W16 GEMM ``tgemm.mm`` launches for this
    bf16 shape (the fused launch then gives its bytes), or None when the tuned
    config names anything else."""
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

    gfx = get_gfx_runtime()
    if gfx != "gfx1250" or n != ROUTER_EXPERTS:
        return None
    from aiter.tuned_gemm import get_GEMM_A16W16_config

    bf16 = str(torch.bfloat16)
    try:
        config = get_GEMM_A16W16_config(
            m, n, k, False, bf16, bf16, gfx=gfx, cu_num=get_cu_num()
        )
    except Exception:  # noqa: BLE001 - no config: not this kernel's GEMM
        return None
    if config is None or config.get("libtype") != "flydsl":
        return None
    from aiter.ops.flydsl.gemm_kernels import get_flydsl_hgemm_kernel_params

    p = get_flydsl_hgemm_kernel_params(config["kernelName"]) or {}
    if (
        p.get("policy") not in ("ft", "ht")
        or p.get("k_waves", 1) != 1
        or p.get("has_bias")
        or (p.get("dtype"), p.get("out_dtype")) != ("bf16", "bf16")
        or not 1 <= p["split_k"] <= 16
        or k % (p["split_k"] * 32)
    ):
        return None
    return p["split_k"]


@functools.lru_cache(maxsize=256)
def _router_launcher(*args):
    from aiter.ops.flydsl.kernels.moe_router_topk_gfx1250 import (
        build_moe_router_topk_module,
    )

    return build_moe_router_topk_module(*args)


def router_group_counts_shape(m: int) -> tuple[int, int]:
    """``group_counts`` of ``flydsl_moe_router_topk``: one int32 row of 128
    per 8 token rows, two per 16-row tile."""
    return (2 * ((m + _TILE - 1) // _TILE), ROUTER_EXPERTS)


def flydsl_moe_router_topk(
    x: torch.Tensor,
    w: torch.Tensor,
    correction_bias: torch.Tensor | None,
    topk: int,
    split_k: int,
    *,
    need_renorm: bool = True,
    routed_scaling_factor: float = 1.0,
    num_valid_rows: torch.Tensor | None = None,
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    group_counts: torch.Tensor | None = None,
    prefetch: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(topk_weights fp32 [M, topk], topk_ids int32 [M, topk])`` of
    ``topk_gating(tgemm.mm(x, w), correction_bias, score_func="sigmoid")``.
    ``group_counts`` (``router_group_counts_shape(M)`` int32) also receives
    each 8-row group's routes per expert."""
    m, k = x.shape
    assert (
        w.shape == (ROUTER_EXPERTS, k) and w.is_contiguous()
    ), f"router weight {tuple(w.shape)}"
    assert x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16
    assert x.stride(1) == 1 and x.stride(0) % 8 == 0
    dev = x.device
    if topk_weights is None:
        topk_weights = torch.empty((m, topk), dtype=torch.float32, device=dev)
    if topk_ids is None:
        topk_ids = torch.empty((m, topk), dtype=torch.int32, device=dev)
    assert topk_weights.is_contiguous() and topk_ids.is_contiguous()
    if m == 0:
        return topk_weights, topk_ids
    tiles = (m + _TILE - 1) // _TILE
    ws = torch.empty(
        tiles * split_k * _TILE * ROUTER_EXPERTS, dtype=torch.float32, device=dev
    )
    has_bias = correction_bias is not None
    if has_bias:
        assert (
            correction_bias.dtype == torch.float32
            and correction_bias.numel() == ROUTER_EXPERTS
        )
    with_counts = group_counts is not None
    if with_counts:
        assert (
            group_counts.dtype == torch.int32
            and group_counts.is_contiguous()
            and tuple(group_counts.shape) == router_group_counts_shape(m)
        ), "group_counts must be a contiguous int32 router_group_counts_shape(M)"
    launch = _router_launcher(
        k,
        split_k,
        topk,
        need_renorm,
        float(routed_scaling_factor),
        has_bias,
        prefetch,
        with_counts,
    )
    _run_compiled(
        launch,
        ptr_arg(x),
        ptr_arg(w),
        ptr_arg(correction_bias if has_bias else ws),
        ptr_arg(num_valid_rows if num_valid_rows is not None else ws),
        ptr_arg(ws),
        ptr_arg(topk_weights),
        ptr_arg(topk_ids),
        ptr_arg(group_counts if with_counts else ws),
        m,
        x.stride(0),
        int(num_valid_rows is not None),
        current_stream_arg(),
    )
    return topk_weights, topk_ids


__all__ = ["flydsl_moe_router_topk", "router_group_counts_shape", "router_topk_config"]
