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
    build_down_reduce_fp4_module,
    build_down_reduce_fp8_module,
    build_gate_up_fp4_module,
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


@functools.lru_cache(maxsize=64)
def _get_gate_up_fp4(
    hidden, inter, top_k, kvector, serialize_dot2, scale_bn, scale_bk, dot2_acc
):
    return build_gate_up_fp4_module(
        hidden,
        inter,
        top_k,
        kvector=kvector,
        serialize_dot2=serialize_dot2,
        scale_bn=scale_bn,
        scale_bk=scale_bk,
        dot2_acc=dot2_acc,
    )


@functools.lru_cache(maxsize=64)
def _get_down_reduce_fp4(
    inter,
    hidden,
    top_k,
    kvector,
    serialize_dot2,
    scale_bn,
    scale_bk,
    kh_per_warp,
    dot2_acc,
    prefetch,
):
    return build_down_reduce_fp4_module(
        inter,
        hidden,
        top_k,
        kvector=kvector,
        serialize_dot2=serialize_dot2,
        scale_bn=scale_bn,
        scale_bk=scale_bk,
        kh_per_warp=kh_per_warp,
        dot2_acc=dot2_acc,
        prefetch=prefetch,
    )


def pick_kvector_fp4(contraction: int) -> int:
    """Default FP4 kVector = **8** (one i32 = 8 FP4 = one weight dword/lane/iter).

    We deliberately do *not* auto-scale to 16/32 even when the contraction dim
    allows it: warp-decode runs one wave per output scalar, so a lane's VGPR
    footprint gates wave occupancy, and the larger kVector's extra live
    activation/weight dwords (and more pairs held through the G7 drain) cut
    occupancy more than the fewer iterations save.  A/B on gfx950 (DeepSeek
    I2048/H7168/E8) confirms kVector=8 is best or tied-best across B?{1,2,4,8}
    for both stages; kV16 only wins ~4% at ``down`` B=1 and loses at B>=2, kV32 is
    uniformly worse.  Callers may still override via ``kvector=`` for tuning; the
    builders accept 16/32 (needs ``contraction % 1024`` / ``% 2048``).

    Requires ``contraction % 512 == 0`` (64*8); all ticket shapes satisfy this.
    """
    if contraction % 512 == 0:
        return 8
    raise ValueError(
        f"contraction={contraction} not divisible by 64*8=512; unsupported for "
        "warp-decode MXFP4 (FP4 packs 8 nibbles/dword)"
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


def flydsl_warp_decode_gate_up_fp4(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    router_ids: torch.Tensor,
    w_gate_scale: torch.Tensor,
    w_up_scale: torch.Tensor,
    *,
    scale_block: tuple[int, int] = (1, 32),
    serialize_dot2: bool = True,
    dot2_acc: int = 1,
    kvector: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """gate_up stage with **MXFP4** weights (BF16 activation, FP4 e2m1 + E8M0).

    Same ``silu(gate)*up`` epilogue as :func:`flydsl_warp_decode_gate_up`, but
    ``w_gate`` / ``w_up`` are MXFP4 (8 FP4 nibbles / i32, half the FP8 weight
    bandwidth) with per-block E8M0 scales applied in the hardware scaled convert.

    Args:
        x:            [B, HIDDEN] bfloat16 (row-major, contiguous).
        w_gate/w_up:  MXFP4 weights for [E, INTER, HIDDEN], packed 2 FP4/byte:
            ``uint8`` [E, INTER, HIDDEN//2] (row = e*INTER + j). Logical HIDDEN
            is taken from ``x``.
        router_ids:   [B, TOPK] int32.
        w_gate_scale/w_up_scale: ``uint8`` E8M0 block scales, each
            [(E*INTER)//BN, HIDDEN//BK] row-major over (weight-row-block, K-block),
            (BN, BK) = ``scale_block``.
        scale_block:  (BN, BK); MXFP4 default (1, 32).
        dot2_acc:     G7 independent dot2 accumulators per stream. **Default 1
            (serialized)**: G7 measured ~4% slower for gate_up on gfx950 (the two
            gate/up streams already cover the hazard and the B=1 grid is
            occupancy-bound); ``>1`` enables the s_nop-free ILP path (see builder).
        kvector:      elements/lane/iter; ``None`` auto-picks the largest that tiles
            HIDDEN (32/16/8, see :func:`pick_kvector_fp4`). Override for A/B.
        out:          optional [B, TOPK, INTER] bfloat16 output buffer.

    Returns:
        [B, TOPK, INTER] bfloat16 intermediate.
    """
    assert x.dtype == torch.bfloat16, "activation must be bfloat16 for this path"
    assert x.is_contiguous() and w_gate.is_contiguous() and w_up.is_contiguous()
    assert w_gate.shape == w_up.shape, "w_gate and w_up must share shape"
    assert w_gate.dim() == 3, "w_gate must be [E, INTER, HIDDEN//2] (uint8)"

    B, HIDDEN = x.shape
    E, INTER, packed_h = w_gate.shape
    assert HIDDEN % 2 == 0, "HIDDEN must be even (2 FP4 per byte)"
    assert (
        packed_h == HIDDEN // 2
    ), f"w_gate last dim {packed_h} != HIDDEN//2 {HIDDEN // 2}"
    TOPK = router_ids.shape[1]

    scale_bn, scale_bk = int(scale_block[0]), int(scale_block[1])
    assert (E * INTER) % scale_bn == 0, "(E*INTER) must be divisible by BN"
    assert HIDDEN % scale_bk == 0, "HIDDEN must be divisible by BK"

    if kvector is None:
        kvector = pick_kvector_fp4(HIDDEN)
    if out is None:
        out = torch.empty((B, TOPK, INTER), dtype=torch.bfloat16, device=x.device)

    launcher = _get_gate_up_fp4(
        HIDDEN, INTER, TOPK, kvector, serialize_dot2, scale_bn, scale_bk, dot2_acc
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


def flydsl_warp_decode_down_reduce_fp4(
    intermediate: torch.Tensor,
    w_down: torch.Tensor,
    router_ids: torch.Tensor,
    router_wts: torch.Tensor,
    w_down_scale: torch.Tensor,
    *,
    scale_block: tuple[int, int] = (1, 32),
    serialize_dot2: bool = True,
    kh_per_warp: int | None = None,
    dot2_acc: int = 4,
    kvector: int | None = None,
    prefetch: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """down_reduce stage with **MXFP4** weights (BF16 intermediate, FP4 e2m1 + E8M0).

    Same reduction as :func:`flydsl_warp_decode_down_reduce`, but ``w_down`` is
    MXFP4 (8 FP4 nibbles / i32, half the FP8 weight bandwidth) with a Block2D
    E8M0 per-block scale applied in the hardware scaled convert.

    Args:
        intermediate: [B, TOPK, INTER] bfloat16 (row = b*TOPK + k, contiguous).
        w_down:       MXFP4 weights for [E, HIDDEN, INTER], packed 2 FP4/byte:
            a ``uint8`` tensor of [E, HIDDEN, INTER//2] (or any contiguous view
            with that many bytes; row = e*HIDDEN + out_j). Logical INTER is taken
            from ``intermediate``.
        router_ids:   [B, TOPK] int32.
        router_wts:   [B, TOPK] float32 (normalized to sum 1 per token).
        w_down_scale: ``uint8`` E8M0 block scales, [(E*HIDDEN)//BN, INTER//BK]
            row-major over (weight-row-block, K-block), (BN, BK) = ``scale_block``.
        scale_block:  (BN, BK); MXFP4 default (1, 32).
        kh_per_warp:  outputs per wave (defaults to 2 when HIDDEN is even, else 1).
        dot2_acc:     G7 independent dot2 accumulators (default 4; s_nop-free ILP).
            ``<=1`` falls back to the serialized ``s_nop`` chain (``serialize_dot2``).
        kvector:      elements/lane/iter; ``None`` auto-picks the largest that tiles
            INTER (32/16/8, see :func:`pick_kvector_fp4`). Override for A/B.
        prefetch:     G8 software prefetch -- hoist all weight/scale loads ahead of
            the converts (default off; A/B lever for B=1 cold-HBM).
        out:          optional [B, HIDDEN] bfloat16 output buffer.

    Returns:
        [B, HIDDEN] bfloat16 output.
    """
    assert intermediate.dtype == torch.bfloat16, "intermediate must be bfloat16"
    assert intermediate.is_contiguous() and w_down.is_contiguous()
    assert router_wts.dtype == torch.float32, "router_wts must be float32"
    assert router_ids.shape == router_wts.shape, "router_ids/router_wts shape mismatch"

    B, TOPK, INTER = intermediate.shape
    assert router_ids.shape == (B, TOPK), "router_ids must be [B, TOPK]"

    # w_down carries INTER/2 bytes per (e, out_j) row; derive HIDDEN from the
    # byte count so callers may pass either [E, HIDDEN, INTER//2] or a flat view.
    total_bytes = w_down.numel() * w_down.element_size()
    assert INTER % 2 == 0, "INTER must be even (2 FP4 per byte)"
    row_bytes = INTER // 2
    assert total_bytes % row_bytes == 0, "w_down byte count not a multiple of INTER/2"
    n_rows = total_bytes // row_bytes  # E * HIDDEN

    scale_bn, scale_bk = int(scale_block[0]), int(scale_block[1])
    assert INTER % scale_bk == 0, "INTER must be divisible by BK"
    assert n_rows % scale_bn == 0, "(E*HIDDEN) must be divisible by BN"

    # HIDDEN follows from the routed expert count; recover E from w_down shape
    # when it is 3-D, otherwise require the caller to make HIDDEN explicit via a
    # 3-D [E, HIDDEN, INTER//2] layout (the supported/tested form).
    assert w_down.dim() == 3, "w_down must be [E, HIDDEN, INTER//2] (uint8)"
    E, HIDDEN, packed_inter = w_down.shape
    assert (
        packed_inter == row_bytes
    ), f"w_down last dim {packed_inter} != INTER//2 {row_bytes}"
    assert E * HIDDEN == n_rows

    if kh_per_warp is None:
        kh_per_warp = 2 if HIDDEN % 2 == 0 else 1
    assert HIDDEN % kh_per_warp == 0, "HIDDEN must be divisible by kh_per_warp"

    if kvector is None:
        kvector = pick_kvector_fp4(INTER)
    if out is None:
        out = torch.empty((B, HIDDEN), dtype=torch.bfloat16, device=intermediate.device)

    launcher = _get_down_reduce_fp4(
        INTER,
        HIDDEN,
        TOPK,
        kvector,
        serialize_dot2,
        scale_bn,
        scale_bk,
        kh_per_warp,
        dot2_acc,
        prefetch,
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
