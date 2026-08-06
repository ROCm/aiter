# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Python entry points for the FlyDSL warp-decode MoE kernels (SILOTIGER-667).

Warp-decode MoE targets very small M (B = 1..4 decode tokens): one wave (64
lanes) cooperatively computes one output scalar with ``v_dot2_f32_bf16`` instead
of the matrix cores.  See ``aiter/ops/flydsl/kernels/warp_decode_moe.py`` for the
kernels and ``SILOTIGER-667-plan.md`` for the design.

Phase 2 exposes ``flydsl_warp_decode_gate_up`` (BF16 activation, FP8 e4m3
weights, PerTensor / PerToken weight scales).
"""

from __future__ import annotations

import functools

import flydsl.compiler as flyc
import torch

from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg
from aiter.ops.flydsl.kernels.warp_decode_moe import (
    build_down_reduce_fp8_module,
    build_gate_up_fp8_module,
    pick_kvector,
)


@functools.lru_cache(maxsize=64)
def _get_gate_up(
    hidden, inter, top_k, kvector, w_scale_mode, serialize_dot2, scale_bn, scale_bk
):
    return build_gate_up_fp8_module(
        hidden,
        inter,
        top_k,
        kvector=kvector,
        w_scale_mode=w_scale_mode,
        serialize_dot2=serialize_dot2,
        scale_bn=scale_bn,
        scale_bk=scale_bk,
    )


@functools.lru_cache(maxsize=64)
def _get_down_reduce(
    inter,
    hidden,
    top_k,
    kvector,
    w_scale_mode,
    serialize_dot2,
    scale_bn,
    scale_bk,
    kh_per_warp,
):
    return build_down_reduce_fp8_module(
        inter,
        hidden,
        top_k,
        kvector=kvector,
        w_scale_mode=w_scale_mode,
        serialize_dot2=serialize_dot2,
        scale_bn=scale_bn,
        scale_bk=scale_bk,
        kh_per_warp=kh_per_warp,
    )


def _run(launcher, args):
    """JIT-compile on first call, then dispatch via the cached CompiledFunction."""
    cf = getattr(launcher, "_cf", None)
    if cf is None:
        cf = flyc.compile(launcher, *args)
        launcher._cf = cf
    else:
        cf(*args)


def flydsl_warp_decode_gate_up(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    router_ids: torch.Tensor,
    w_gate_scale: torch.Tensor,
    w_up_scale: torch.Tensor,
    *,
    w_scale_mode: str = "pertensor",
    scale_block: tuple[int, int] | None = None,
    serialize_dot2: bool = True,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """gate_up stage of warp-decode MoE (BF16 activation, FP8 e4m3 weights).

    Computes, per token ``b`` and each of its TOPK experts ``k``
    (``e = router_ids[b, k]``)::

        gate_acc = (?_i x[b,i].w_gate[e,j,i]) . gs
        up_acc   = (?_i x[b,i].w_up  [e,j,i]) . us
        out[b,k,j] = silu(gate_acc) . up_acc            # silu(z)=z/(1+e^-z)

    Args:
        x:            [B, HIDDEN] bfloat16 (row-major, contiguous).
        w_gate/w_up:  [E, INTER, HIDDEN] float8_e4m3fn (row = e*INTER + j).
        router_ids:   [B, TOPK] int32.
        w_gate_scale/w_up_scale: float32 weight scales.  ``pertensor`` -> shape
            [1]; ``pertoken`` -> shape [E*INTER] (one per weight row);
            ``block2d`` -> shape [(E*INTER)//BN * HIDDEN//BK] row-major over
            (row-block, K-block) with (BN, BK) = ``scale_block``.
        w_scale_mode: "pertensor", "pertoken" or "block2d".
        scale_block:  (BN, BK) block dims, required when ``w_scale_mode='block2d'``.
        out:          optional [B, TOPK, INTER] bfloat16 output buffer.

    Returns:
        [B, TOPK, INTER] bfloat16 intermediate.
    """
    if w_scale_mode not in ("pertensor", "pertoken", "block2d"):
        raise ValueError(f"unsupported w_scale_mode: {w_scale_mode!r}")
    assert x.dtype == torch.bfloat16, "activation must be bfloat16 for this path"
    assert x.is_contiguous() and w_gate.is_contiguous() and w_up.is_contiguous()

    B, HIDDEN = x.shape
    E, INTER, Hk = w_gate.shape
    assert Hk == HIDDEN, f"w_gate HIDDEN {Hk} != x HIDDEN {HIDDEN}"
    assert w_up.shape == w_gate.shape, "w_gate and w_up must share shape"
    TOPK = router_ids.shape[1]

    scale_bn = scale_bk = None
    if w_scale_mode == "block2d":
        if scale_block is None:
            raise ValueError("block2d requires scale_block=(BN, BK)")
        scale_bn, scale_bk = int(scale_block[0]), int(scale_block[1])
        assert (E * INTER) % scale_bn == 0, "(E*INTER) must be divisible by BN"
        assert HIDDEN % scale_bk == 0, "HIDDEN must be divisible by BK"

    kvector = pick_kvector(HIDDEN)
    if out is None:
        out = torch.empty((B, TOPK, INTER), dtype=torch.bfloat16, device=x.device)

    launcher = _get_gate_up(
        HIDDEN, INTER, TOPK, kvector, w_scale_mode, serialize_dot2, scale_bn, scale_bk
    )
    grid_x = B * TOPK * INTER
    _run(
        launcher,
        (
            ptr_arg(x),
            ptr_arg(w_gate),
            ptr_arg(w_up),
            ptr_arg(w_gate_scale),
            ptr_arg(w_up_scale),
            ptr_arg(router_ids),
            ptr_arg(out),
            grid_x,
            torch.cuda.current_stream(),
        ),
    )
    return out


def flydsl_warp_decode_down_reduce(
    intermediate: torch.Tensor,
    w_down: torch.Tensor,
    router_ids: torch.Tensor,
    router_wts: torch.Tensor,
    w_down_scale: torch.Tensor,
    *,
    w_scale_mode: str = "pertensor",
    scale_block: tuple[int, int] | None = None,
    serialize_dot2: bool = True,
    kh_per_warp: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """down_reduce stage of warp-decode MoE (BF16 intermediate, FP8 e4m3 weights).

    Computes, per token ``b`` and output ``out_j``::

        y[b,out_j] = ?_k router_wts[b,k] . ds_k . (?_i inter[b,k,i].w_down[e,out_j,i])

    where ``e = router_ids[b,k]`` and ``ds_k`` is the weight scale for row
    ``e*HIDDEN + out_j`` (Block2D also varies with the K index ``i``).

    Args:
        intermediate: [B, TOPK, INTER] bfloat16 (row = b*TOPK + k, contiguous).
        w_down:       [E, HIDDEN, INTER] float8_e4m3fn (row = e*HIDDEN + out_j).
        router_ids:   [B, TOPK] int32.
        router_wts:   [B, TOPK] float32 (normalized to sum 1 per token).
        w_down_scale: float32 weight scales.  ``pertensor`` -> [1];
            ``pertoken`` -> [E*HIDDEN] (one per weight row); ``block2d`` ->
            [(E*HIDDEN)//BN * INTER//BK] row-major over (row-block, K-block).
        w_scale_mode: "pertensor", "pertoken" or "block2d".
        scale_block:  (BN, BK) block dims, required when ``w_scale_mode='block2d'``.
        kh_per_warp:  outputs per wave (memory-level parallelism / H-tiling). Defaults
            to 2 (H2, the FP8 best) when HIDDEN is even, else 1.
        out:          optional [B, HIDDEN] bfloat16 output buffer.

    Returns:
        [B, HIDDEN] bfloat16 output.
    """
    if w_scale_mode not in ("pertensor", "pertoken", "block2d"):
        raise ValueError(f"unsupported w_scale_mode: {w_scale_mode!r}")
    assert intermediate.dtype == torch.bfloat16, "intermediate must be bfloat16"
    assert intermediate.is_contiguous() and w_down.is_contiguous()
    assert router_wts.dtype == torch.float32, "router_wts must be float32"

    B, TOPK, INTER = intermediate.shape
    E, HIDDEN, Ik = w_down.shape
    assert Ik == INTER, f"w_down INTER {Ik} != intermediate INTER {INTER}"
    assert router_ids.shape == (B, TOPK), "router_ids must be [B, TOPK]"

    scale_bn = scale_bk = None
    if w_scale_mode == "block2d":
        if scale_block is None:
            raise ValueError("block2d requires scale_block=(BN, BK)")
        scale_bn, scale_bk = int(scale_block[0]), int(scale_block[1])
        assert (E * HIDDEN) % scale_bn == 0, "(E*HIDDEN) must be divisible by BN"
        assert INTER % scale_bk == 0, "INTER must be divisible by BK"

    if kh_per_warp is None:
        kh_per_warp = 2 if HIDDEN % 2 == 0 else 1
    assert HIDDEN % kh_per_warp == 0, "HIDDEN must be divisible by kh_per_warp"

    kvector = pick_kvector(INTER)
    if out is None:
        out = torch.empty((B, HIDDEN), dtype=torch.bfloat16, device=intermediate.device)

    launcher = _get_down_reduce(
        INTER,
        HIDDEN,
        TOPK,
        kvector,
        w_scale_mode,
        serialize_dot2,
        scale_bn,
        scale_bk,
        kh_per_warp,
    )
    grid_x = B * (HIDDEN // kh_per_warp)
    _run(
        launcher,
        (
            ptr_arg(intermediate),
            ptr_arg(w_down),
            ptr_arg(w_down_scale),
            ptr_arg(router_ids),
            ptr_arg(router_wts),
            ptr_arg(out),
            grid_x,
            torch.cuda.current_stream(),
        ),
    )
    return out
