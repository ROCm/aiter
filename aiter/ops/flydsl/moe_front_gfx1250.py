# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 decode MoE front: router GEMM + biased sigmoid top-k in one launch.

``flydsl_moe_router_topk`` returns what ``topk_gating(tgemm.mm(x, w), bias,
score_func="sigmoid")`` returns, byte for byte, from one FlyDSL launch
(``kernels/moe_router_topk_gfx1250.py``). It reproduces the router GEMM the
tuned config launches, so it only takes the shapes whose config names the
gfx1250 FlyDSL A16W16 kernel (``router_split_k`` says which).
"""

from __future__ import annotations

import functools
import os

import torch

from aiter.ops.flydsl.kernels.tensor_shim import (
    _run_compiled,
    current_stream_arg,
    ptr_arg,
)

ROUTER_EXPERTS = 128
_TILE = 16


@functools.lru_cache(maxsize=4096)
def router_split_k(m: int, n: int, k: int, dtype: str, otype: str) -> int | None:
    """split_k of the gfx1250 FlyDSL A16W16 GEMM ``tgemm.mm`` launches for this
    shape, or None when the tuned config names anything else (whose logits
    this kernel would not reproduce)."""
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx

    gfx = get_gfx()
    if not str(gfx).startswith("gfx1250") or n != ROUTER_EXPERTS:
        return None
    from aiter.tuned_gemm import get_GEMM_A16W16_config

    try:
        config = get_GEMM_A16W16_config(
            M=m,
            N=n,
            K=k,
            bias=False,
            dtype=dtype,
            otype=otype,
            scaleAB=False,
            bpreshuffle=False,
            gfx=gfx,
            cu_num=get_cu_num(),
        )
    except Exception:  # noqa: BLE001 - no config: not this kernel's GEMM
        return None
    if config is None or config.get("libtype") != "flydsl":
        return None
    from aiter.ops.flydsl.gemm_kernels import get_flydsl_hgemm_kernel_params

    params = get_flydsl_hgemm_kernel_params(config["kernelName"])
    if params is None or params.get("policy") not in ("ft", "ht"):
        return None
    if params.get("k_waves", 1) != 1 or params.get("has_bias"):
        return None
    if params.get("dtype") != "bf16" or params.get("out_dtype") != "bf16":
        return None
    split_k = int(params["split_k"])
    if split_k < 1 or split_k > 16 or k % (split_k * 32):
        return None
    from aiter.ops.flydsl.kernels.moe_router_topk_gfx1250 import router_topk_geometry

    try:
        router_topk_geometry(split_k)
    except ValueError:
        return None
    return split_k


@functools.cache
def _router_tuning() -> tuple[int, bool]:
    """(chains, count_atomics) defaults, read once: AITER_MOE_ROUTER_CHAINS
    (16-expert slices per wave, default 1) and AITER_MOE_ROUTER_COUNT_ATOMICS
    (group counts by device-scope adds instead of a second cluster barrier,
    default 0). Every setting gives the same bytes."""
    chains = max(1, int(os.environ.get("AITER_MOE_ROUTER_CHAINS", "1")))
    atomics = os.environ.get("AITER_MOE_ROUTER_COUNT_ATOMICS", "0") in (
        "1",
        "true",
        "True",
    )
    return chains, atomics


def _chains_for(split_k: int, chains: int) -> int:
    from aiter.ops.flydsl.kernels.moe_router_topk_gfx1250 import router_topk_geometry

    slices = router_topk_geometry(split_k)[2]
    while slices % chains:
        chains -= 1
    return chains


@functools.lru_cache(maxsize=256)
def _router_launcher(
    k,
    split_k,
    topk,
    need_renorm,
    scale,
    has_bias,
    prefetch,
    with_counts,
    chains,
    count_atomics,
):
    from aiter.ops.flydsl.kernels.moe_router_topk_gfx1250 import (
        build_moe_router_topk_module,
    )

    return build_moe_router_topk_module(
        K=k,
        split_k=split_k,
        topk=topk,
        need_renorm=need_renorm,
        routed_scaling_factor=scale,
        has_bias=has_bias,
        prefetch=prefetch,
        with_counts=with_counts,
        chains=chains,
        count_atomics=count_atomics,
    )


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
    chains: int | None = None,
    count_atomics: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(topk_weights fp32 [M, topk], topk_ids int32 [M, topk])`` of
    ``topk_gating(tgemm.mm(x, w), correction_bias, score_func="sigmoid")``;
    ``split_k`` must be ``router_split_k`` of the shape. ``group_counts``
    (``router_group_counts_shape(M)`` int32) also receives each 8-row
    group's routes per expert, which the grouped MoE's A1 route can take in
    place of re-counting every id (``moe_route_group_counts``)."""
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
    default_chains, default_atomics = _router_tuning()
    chains = _chains_for(split_k, default_chains if chains is None else chains)
    count_atomics = default_atomics if count_atomics is None else bool(count_atomics)
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
        chains,
        count_atomics,
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


__all__ = ["flydsl_moe_router_topk", "router_group_counts_shape", "router_split_k"]
