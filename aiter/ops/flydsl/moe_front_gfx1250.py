# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 decode MoE front: ``flydsl_moe_router_topk`` runs ``topk_gating(
tgemm.mm(x, w), ...)`` in one FlyDSL launch. Callers check
``flydsl_moe_router_topk_supported`` (same arguments) and otherwise keep
``tgemm.mm`` + ``topk_gating``.
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
def router_topk_config(m: int, n: int, k: int, topk: int) -> dict | None:
    """``split_k`` / ``prefetch`` of ``flydsl_moe_router_topk`` for this call, or
    None to keep ``tgemm.mm`` + ``topk_gating``: the tuned table
    (``AITER_CONFIG_MOE_ROUTER_TOPK``, padded M) has no fused row, or a bitexact
    row's split_k is no longer the router GEMM's."""
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
    if gfx != "gfx1250":
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


def _unsupported_reason(
    x: torch.Tensor,
    w: torch.Tensor,
    topk: int,
    correction_bias: torch.Tensor | None,
    score_func: str,
    num_valid_rows: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    topk_ids: torch.Tensor | None,
    tuned: bool,
) -> str | None:
    """Why ``flydsl_moe_router_topk`` cannot take this call, or None."""
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx1250":
        return "gfx1250 only"
    from aiter.ops.flydsl.kernels.moe_router_topk_gfx1250 import (
        EXPERT_COUNTS,
        MAX_TOPK,
        SCORE_FUNCS,
    )

    if score_func not in SCORE_FUNCS:
        return f"score_func {score_func!r} is not one of {SCORE_FUNCS}"
    if x.dim() != 2 or w.dim() != 2 or x.shape[1] != w.shape[1]:
        return f"x {tuple(x.shape)} and w {tuple(w.shape)} are not [M, K] and [E, K]"
    m, k = x.shape
    experts = w.shape[0]
    if experts not in EXPERT_COUNTS:
        return f"{experts} experts is not one of {EXPERT_COUNTS}"
    if not 1 <= topk <= MAX_TOPK:
        return f"topk {topk} is not in [1, {MAX_TOPK}]"
    if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        return "x and w must be bfloat16"
    if x.stride(1) != 1 or x.stride(0) % 8 or not w.is_contiguous():
        return "x needs unit column stride and an 8-element row stride; w contiguous"
    tensors = [x, w]
    if correction_bias is not None:
        if (
            correction_bias.dtype != torch.float32
            or correction_bias.shape != (experts,)
            or not correction_bias.is_contiguous()
        ):
            return f"correction_bias must be a contiguous float32 [{experts}]"
        tensors.append(correction_bias)
    if num_valid_rows is not None:
        if num_valid_rows.dtype != torch.int32 or num_valid_rows.numel() < 1:
            return "num_valid_rows must be an int32 tensor"
        tensors.append(num_valid_rows)
    for out, dtype in ((topk_weights, torch.float32), (topk_ids, torch.int32)):
        if out is not None:
            if out.shape != (m, topk) or out.dtype != dtype or not out.is_contiguous():
                return f"topk outputs must be contiguous [{m}, {topk}] float32 / int32"
            tensors.append(out)
    if not x.is_cuda or any(t.device != x.device for t in tensors):
        return "every tensor must be on the same GPU"
    if tuned and router_topk_config(m, experts, k, topk) is None:
        return (
            f"no fused row for M {m}, E {experts}, K {k}, top-{topk} in "
            "AITER_CONFIG_MOE_ROUTER_TOPK"
        )
    return None


def flydsl_moe_router_topk_supported(
    x: torch.Tensor,
    w: torch.Tensor,
    topk: int,
    correction_bias: torch.Tensor | None = None,
    need_renorm: bool = True,
    routed_scaling_factor: float = 1.0,
    score_func: str = "sigmoid",
    *,
    num_valid_rows: torch.Tensor | None = None,
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
) -> bool:
    """Whether ``flydsl_moe_router_topk`` takes these arguments (the op's own):
    implemented (score_func, expert count, topk, dtypes, layouts, gfx1250) and
    tuned faster than ``tgemm.mm`` + ``topk_gating`` for this shape on this
    GPU. False means keep the unfused pair."""
    return (
        _unsupported_reason(
            x,
            w,
            topk,
            correction_bias,
            score_func,
            num_valid_rows,
            topk_weights,
            topk_ids,
            tuned=True,
        )
        is None
    )


@functools.lru_cache(maxsize=256)
def _router_launcher(**kwargs):
    from aiter.ops.flydsl.kernels.moe_router_topk_gfx1250 import (
        build_moe_router_topk_module,
    )

    return build_moe_router_topk_module(**kwargs)


def flydsl_moe_router_topk(
    x: torch.Tensor,
    w: torch.Tensor,
    topk: int,
    correction_bias: torch.Tensor | None = None,
    need_renorm: bool = True,
    routed_scaling_factor: float = 1.0,
    score_func: str = "sigmoid",
    *,
    num_valid_rows: torch.Tensor | None = None,
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    split_k: int | None = None,
    prefetch: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``topk_gating(topk_weights, topk_ids, tgemm.mm(x, w), correction_bias,
    need_renorm, routed_scaling_factor, score_func,
    num_valid_rows=num_valid_rows)`` in one launch: returns ``(topk_weights
    fp32 [M, topk], topk_ids int32 [M, topk])``, allocated when not given.

    x is [M, K] and w [E, K], both bfloat16. Raises ValueError where
    ``flydsl_moe_router_topk_supported`` is False. ``split_k`` / ``prefetch``
    replace the tuned launch (the tuner); then no tuned row is needed."""
    reason = _unsupported_reason(
        x,
        w,
        topk,
        correction_bias,
        score_func,
        num_valid_rows,
        topk_weights,
        topk_ids,
        tuned=split_k is None,
    )
    if reason is not None:
        raise ValueError(f"flydsl_moe_router_topk: {reason}")
    m, k = x.shape
    experts = w.shape[0]
    if split_k is None:
        cfg = router_topk_config(m, experts, k, topk)
        split_k = cfg["split_k"]
        prefetch = cfg["prefetch"] if prefetch is None else prefetch
    dev = x.device
    if topk_weights is None:
        topk_weights = torch.empty((m, topk), dtype=torch.float32, device=dev)
    if topk_ids is None:
        topk_ids = torch.empty((m, topk), dtype=torch.int32, device=dev)
    if m == 0:
        return topk_weights, topk_ids
    tiles = (m + _TILE - 1) // _TILE
    ws = torch.empty(tiles * split_k * _TILE * experts, dtype=torch.float32, device=dev)
    has_bias = correction_bias is not None
    launch = _router_launcher(
        K=k,
        split_k=split_k,
        topk=topk,
        need_renorm=bool(need_renorm),
        routed_scaling_factor=float(routed_scaling_factor),
        has_bias=has_bias,
        prefetch=8 if prefetch is None else prefetch,
        score_func=score_func,
        experts=experts,
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
        m,
        x.stride(0),
        int(num_valid_rows is not None),
        current_stream_arg(),
    )
    return topk_weights, topk_ids


__all__ = ["flydsl_moe_router_topk", "flydsl_moe_router_topk_supported"]
