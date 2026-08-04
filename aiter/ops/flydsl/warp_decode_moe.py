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
    build_gate_up_fp8_module,
    pick_kvector,
)


@functools.lru_cache(maxsize=64)
def _get_gate_up(hidden, inter, top_k, kvector, w_scale_mode, serialize_dot2):
    return build_gate_up_fp8_module(
        hidden,
        inter,
        top_k,
        kvector=kvector,
        w_scale_mode=w_scale_mode,
        serialize_dot2=serialize_dot2,
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
            [1]; ``pertoken`` -> shape [E*INTER] (one per weight row).
        w_scale_mode: "pertensor" or "pertoken".
        out:          optional [B, TOPK, INTER] bfloat16 output buffer.

    Returns:
        [B, TOPK, INTER] bfloat16 intermediate.
    """
    if w_scale_mode not in ("pertensor", "pertoken"):
        raise ValueError(f"unsupported w_scale_mode: {w_scale_mode!r}")
    assert x.dtype == torch.bfloat16, "activation must be bfloat16 for this path"
    assert x.is_contiguous() and w_gate.is_contiguous() and w_up.is_contiguous()

    B, HIDDEN = x.shape
    E, INTER, Hk = w_gate.shape
    assert Hk == HIDDEN, f"w_gate HIDDEN {Hk} != x HIDDEN {HIDDEN}"
    assert w_up.shape == w_gate.shape, "w_gate and w_up must share shape"
    TOPK = router_ids.shape[1]

    kvector = pick_kvector(HIDDEN)
    if out is None:
        out = torch.empty((B, TOPK, INTER), dtype=torch.bfloat16, device=x.device)

    launcher = _get_gate_up(HIDDEN, INTER, TOPK, kvector, w_scale_mode, serialize_dot2)
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
