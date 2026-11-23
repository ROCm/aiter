# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused MoE router GEMM + biased sigmoid top-k gating (gfx1250).

One launch in place of the decode pair ``tgemm.mm(x, w)`` (FlyDSL gfx1250
``gemm_a16w16``, bf16 logits) + ``topk_gating(..., score_func="sigmoid")``.
Both are reproduced bit for bit: the kernel runs the FlyDSL GEMM's WMMA chain
with the same split-K cut (``gate_split_k`` reads it from the tuned config the
GEMM would launch) and the top-k is ``topk_gating_kernel_opt``'s own code, so
the selected experts (including ties) and their weights are those of the
unfused path. See ``csrc/kernels/moe_router_topk_fused.cu``.
"""

from __future__ import annotations

import functools

import torch

from ..jit.core import compile_ops
from ..utility.graph_alloc import capture_scoped_cache, persistent_alloc

ROUTER_EXPERTS = 128
_TILE = 16
# Counters per 16-row tile; 64k rows is far past any decode batch.
_MAX_M_TILES = 4096
_MAX_SPLIT_K = 8


@compile_ops(
    "module_moe_router_topk_fused", fc_name="moe_router_topk_fused", develop=True
)
def moe_router_topk_fused_fwd(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    correction_bias: torch.Tensor,
    workspace: torch.Tensor,
    semaphore: torch.Tensor,
    split_k: int,
    waves: int,
    chunk: int,
    need_renorm: bool,
    routed_scaling_factor: float,
    num_valid_rows: torch.Tensor | None = None,
    logits_out: torch.Tensor | None = None,
) -> None: ...


@capture_scoped_cache
def _router_semaphore(device: torch.device, stream: torch.cuda.Stream):
    # Self-cleaning: the reducing workgroup puts its tile's counter back to 0.
    with persistent_alloc(torch.device(device)):
        # [arrivals | releases] per 16-row tile
        return torch.zeros(2 * _MAX_M_TILES, dtype=torch.int32, device=device)


@functools.lru_cache(maxsize=4096)
def gate_split_k(m: int, n: int, k: int, dtype: str, otype: str) -> int | None:
    """split_k of the gfx1250 FlyDSL GEMM ``tgemm.mm`` launches for this shape,
    or None when the tuned config names any other kernel (then the fused router
    could not reproduce its logits)."""
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx
    from aiter.tuned_gemm import get_GEMM_A16W16_config

    gfx = get_gfx()
    if not str(gfx).startswith("gfx1250"):
        return None
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
    except Exception:  # noqa: BLE001 - no config means no fused router
        return None
    if config is None or config.get("libtype") != "flydsl":
        return None
    from aiter.ops.flydsl.gemm_kernels import get_flydsl_hgemm_kernel_params

    params = get_flydsl_hgemm_kernel_params(config["kernelName"])
    if params is None or params.get("policy") in ("sk", "bs"):
        return None
    if params.get("k_waves", 1) != 1 or params.get("has_bias"):
        return None
    if params.get("dtype") != "bf16" or params.get("out_dtype") != "bf16":
        return None
    split_k = int(params["split_k"])
    if split_k > _MAX_SPLIT_K or k % (split_k * 32):
        return None
    return split_k


def moe_router_topk_fused(
    x: torch.Tensor,
    w: torch.Tensor,
    correction_bias: torch.Tensor | None,
    topk: int,
    split_k: int,
    *,
    need_renorm: bool = True,
    routed_scaling_factor: float = 1.0,
    num_valid_rows: torch.Tensor | None = None,
    logits_out: torch.Tensor | None = None,
    waves: int = 1,
    chunk: int = 4,
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(topk_weights fp32 [M, topk], topk_ids int32 [M, topk])`` of
    ``topk_gating(tgemm.mm(x, w), correction_bias, score_func="sigmoid")``.

    ``split_k`` must be the router GEMM's (``gate_split_k``); ``logits_out``
    optionally receives the bf16 logits.
    """
    m, k = x.shape
    assert w.shape == (ROUTER_EXPERTS, k), f"router weight {tuple(w.shape)}"
    if topk_weights is None:
        topk_weights = torch.empty((m, topk), dtype=torch.float32, device=x.device)
    if topk_ids is None:
        topk_ids = torch.empty((m, topk), dtype=torch.int32, device=x.device)
    if correction_bias is None:
        correction_bias = torch.empty(0, dtype=torch.float32, device=x.device)
    m_tiles = (m + _TILE - 1) // _TILE
    assert m_tiles <= _MAX_M_TILES
    workspace = torch.empty(
        split_k * m_tiles * _TILE * ROUTER_EXPERTS, dtype=torch.float32, device=x.device
    )
    semaphore = _router_semaphore(x.device, torch.cuda.current_stream(x.device))
    moe_router_topk_fused_fwd(
        topk_weights,
        topk_ids,
        x,
        w,
        correction_bias,
        workspace,
        semaphore,
        split_k,
        waves,
        chunk,
        need_renorm,
        float(routed_scaling_factor),
        num_valid_rows,
        logits_out,
    )
    return topk_weights, topk_ids
