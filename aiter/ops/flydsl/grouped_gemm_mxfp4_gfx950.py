# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Ragged MXFP4 (a4w4) grouped GEMM launchers for gfx950.

Groups partition a token axis with **unequal, device-resident** sizes: each
block reads its own bounds from ``group_end_offsets`` and the host never syncs
to learn them. That is the difference from every other grouped MXFP4 path in
aiter -- ``flydsl_grouped_gemm_a8w4_masked`` is gfx1250 + contiguous-M, the MoE
``flydsl_mxfp4_gemm1/gemm2`` need ``moe_sorting`` output and fuse an activation,
and ``flydsl_batched_gemm_mxfp4`` is strided-batched with equal M per batch.

Both kernels take plain per-1x32 cast output: **no preshuffled weights and no
shuffled scales**. One e8m0 is broadcast to 4 bytes so the MFMA ``op_sel`` is a
don't-care, which is what removes the preshuffle requirement that the dense fp4
paths carry.

    flydsl_grouped_gemm_a4w4_ragged   out[group_g] = X[group_g] @ W[g]^T
    flydsl_grouped_wgrad_a4w4_ragged  grad_W[g]    = go[group_g]^T @ ia[group_g]

The wgrad is the transpose-contraction direction: there the groups partition the
CONTRACTION rather than the output rows.
"""

from __future__ import annotations

import os

import torch

from aiter.jit.utils.chip_info import get_gfx

from .kernels.mxfp4_grouped_gemm_ragged import (
    BLOCK_R,
    FP4_PER_BYTE,
    SCALE_BLOCK,
    cached_launch as _cached_launch_gemm,
    pick_block_c,
)
from .kernels.mxfp4_grouped_wgrad_ragged import (
    BLOCK_C,
    BLOCK_M,
    LPT_MAX_E,
    ORDER_MODES,
    cached_launch as _cached_launch_wgrad,
)
from .kernels.mxfp4_grouped_wgrad_ragged import BLOCK_R as WGRAD_BLOCK_R
from .kernels.tensor_shim import _run_compiled


def _require_gfx950(what: str) -> None:
    gfx = get_gfx()
    if gfx != "gfx950":
        raise NotImplementedError(
            f"[FlyDSL] {what} requires gfx950 (CDNA4 scaled MFMA), got {gfx}"
        )


def ceildiv(a: int, b: int) -> int:
    return -(-a // b)


def flydsl_grouped_gemm_a4w4_ragged(
    input_act,
    weight,
    input_act_scales,
    weight_scales,
    group_end_offsets,
    out_dtype=torch.bfloat16,
    block_c=None,
    out=None,
):
    """MXFP4 forward/dgrad grouped GEMM over RAGGED token groups.

        out[group_g] = input_act[group_g] @ weight[g]^T

    Groups partition the token (output row) dim with device-resident, unequal
    sizes. Because the ragged axis is NOT the contraction, the whole pipelined
    K-walk carries over from the even-groups body untouched -- only the
    block->(group, tile) mapping and the epilogue mask change.

    input_act         (M, K//2)      packed fp4 e2m1, 2 per byte, row-major
    weight            (E, N, K//2)   packed fp4 e2m1, row-major
    input_act_scales  (M, K//32)     e8m0 as uint8
    weight_scales     (E, N, K//32)
    group_end_offsets (E,) int32  cumulative group ends along M, ON DEVICE.
                                         Read by the block itself -- never synced here.
    block_c           column tile width; None picks it from N (see pick_block_c)
    out               optional (M, N) bf16 destination to write in place.

    K is passed as the LOGICAL element count, derived from the packed width, so
    every caller-visible shape is in elements and only the kernel's internals
    deal in bytes.

    Pass ``out`` when you already have the destination -- it skips both an
    allocation and the zero-fill. That matters: the zero-fill is a full memset of
    M*N, 805 MB at the e2e shape, measured at 0.117 ms against a 1.225 ms kernel,
    i.e. **9.5%** -- almost the entire cost of ragged support over the even body.

    When we allocate, we zero. Rows at and past ``group_end_offsets[-1]`` are
    written by no block (the dispatcher pads M past the last group), and handing
    back uninitialised memory there is the worse default. Callers matching ATen's
    grouped-mm contract can skip it: ATen's own reference leaves that tail
    untouched too, so it is not part of the result.
    """
    _require_gfx950("grouped MXFP4 GEMM")
    M, KP = input_act.shape
    E, N, KP2 = weight.shape
    assert KP == KP2, f"packed-K mismatch: A={KP}, B={KP2}"
    K = KP * FP4_PER_BYTE
    assert input_act_scales.shape == (
        M,
        K // SCALE_BLOCK,
    ), f"x scales {tuple(input_act_scales.shape)} != {(M, K // SCALE_BLOCK)}"
    assert weight_scales.shape == (
        E,
        N,
        K // SCALE_BLOCK,
    ), f"w scales {tuple(weight_scales.shape)} != {(E, N, K // SCALE_BLOCK)}"
    assert (
        group_end_offsets.numel() == E
    ), f"offsets {group_end_offsets.numel()} != E {E}"

    # Same int32 ceiling the wgrad carries, and for two reasons at once here.
    # FlyDSL's CABI packs a tensor's flattened element count as int32 and the
    # operands go over as 1-D views; separately, the epilogue store offset is
    # `r_ * out_n + col`, an i32 element index into `out`. An overflow there
    # does NOT fault: the bounded buffer descriptor clamps out-of-range, but a
    # wrapped index lands back INSIDE the allocation and silently corrupts a
    # row, which is the worse failure of the two.
    #
    # M*N is the binding term: at N=2048 it is reached at M = 1,048,576 tokens
    # on one rank -- unreachable under balanced routing, but real routing
    # concentrates enough tokens on a single rank to cross it.
    #
    # UNDER torch.compile this cannot be a Python branch: M is the routed token
    # count, an unbacked SymInt, so a data-dependent `>` guard kills the whole
    # compiled path. torch._check registers a deferred runtime assert (an
    # over-limit shape still fails loudly rather than wrapping) and teaches the
    # shape env the fact. Eager keeps the graceful fallback so a caller can drop
    # to another kernel instead of losing the run.
    INT32_MAX = 2**31 - 1
    biggest = max(M * N, M * KP, E * N * KP)
    if torch.compiler.is_compiling():
        torch._check(biggest <= INT32_MAX)
    elif biggest > INT32_MAX:
        raise NotImplementedError(
            f"largest operand has {biggest} elements, over the int32 limit "
            f"FlyDSL's argument packing and this kernel's epilogue offset "
            f"impose ({INT32_MAX}); M={M} N={N} K={K} E={E}"
        )

    BLOCK_C = pick_block_c(N) if block_c is None else int(block_c)
    n_c = ceildiv(N, BLOCK_C)
    # Worst-case row tiles: a full cover of M plus at most one partial tile per
    # group. Computed from M and E alone -- deriving it from the actual group
    # sizes would need them on the host, i.e. the sync this kernel avoids.
    # Slots past the real tile count predicate themselves off via `active`.
    n_slots = ceildiv(M, BLOCK_R) + E
    n_blocks = n_slots * n_c

    if out is None:
        # zeros, not empty: rows at and past the last group's end are written by
        # no block, so torch.empty would hand back uninitialised memory there.
        out = torch.zeros(M, N, dtype=torch.bfloat16, device=input_act.device)
    else:
        assert (
            out.shape == (M, N) and out.dtype == torch.bfloat16
        ), f"out {tuple(out.shape)}/{out.dtype} != {(M, N)}/torch.bfloat16"
        assert out.is_contiguous(), "out must be contiguous"
    launch = _cached_launch_gemm(K, N, E, BLOCK_C)
    _run_compiled(
        launch,
        input_act.contiguous().view(torch.int8).reshape(-1),
        weight.contiguous().view(torch.int8).reshape(-1),
        out.view(-1),
        input_act_scales.contiguous().view(torch.uint8).view(-1),
        weight_scales.contiguous().view(torch.uint8).view(-1),
        group_end_offsets.to(torch.int32).contiguous(),
        n_blocks,
        n_c,
        M,
        N,
        torch.cuda.current_stream(),
    )
    return out if out_dtype == torch.bfloat16 else out.to(out_dtype)


def flydsl_grouped_wgrad_a4w4_ragged(
    go_t,
    go_scale,
    ia_t,
    ia_scale,
    group_end_offsets,
    out_dtype=torch.bfloat16,
    m_total=None,
    _order=None,
):
    """Ragged MXFP4 wgrad, tuned body.

           grad_W[g] = go[group_g]^T @ ia[group_g]

       go_t   (R, M//2) packed fp4, dim1-quantized (tokens contiguous)
       ia_t   (C, M//2) packed fp4
       scales (R, M//32) / (C, M//32) e8m0 -- FULL-WIDTH planes at the operand's
              row stride, not pre-sliced to the routed token count.
    One block per (group, output tile); each block
       reads its own group bounds from ``group_end_offsets`` on device.

       ``m_total`` is unused by the body -- the grid is E * n_r * n_c regardless of
       the token count, and every boundary comes from the offsets. It is accepted
       for signature compatibility with the caller's integration.
    """
    _require_gfx950("grouped MXFP4 wgrad")
    del m_total

    R, M_PACKED = go_t.shape
    C, M2 = ia_t.shape
    assert M_PACKED == M2, f"token-dim mismatch: {M_PACKED} vs {M2}"
    # Callers see logical token counts; only the kernel body deals in bytes.
    M_ROW = M_PACKED * FP4_PER_BYTE
    E = group_end_offsets.shape[0]

    # The cooperative dwordx2 scale load needs an even element index; with a
    # 256-aligned window base that reduces to SCALE_I32_ROW = M_ROW/128 being
    # even. The MoE caller already row-pads its operands for the dim1 cast, so
    # this is a pad-multiple change (128 -> 256) rather than a copy it was not
    # already paying. NotImplementedError so the integration falls back instead
    # of crashing.
    # 128, not 256: the scale plane packs 4 e8m0 per i32, so a row must be a whole
    # number of i32 (M_ROW/32 divisible by 4). Whether the cooperative scale load
    # can use a dwordx2 or needs two dwords then depends on M_ROW % 256, which is
    # passed to the compiler as `sc_pair` rather than demanded of the caller --
    # see the SC_PAIR note in _compile. Requiring 256 here would decline 7 of 8
    # shapes the MoE dispatcher actually produces.
    if M_ROW % BLOCK_M != 0:
        raise NotImplementedError(
            f"row stride {M_ROW} must be a multiple of {BLOCK_M} so a scale-plane "
            f"row is a whole number of i32"
        )
    sc_pair = (M_ROW % (2 * BLOCK_M)) == 0

    # FlyDSL's CABI packs a tensor's flattened element count as int32, and the
    # operands go over as 1-D views (the kernel addresses them flat), so an
    # operand with >= 2**31 elements raises `struct.error` from inside the
    # dispatch -- which kills the whole training job rather than failing one op.
    # Raise NotImplementedError instead: the integration catches it and falls
    # back to the Triton wgrad, so the run survives at reduced speed.
    #
    # Measured threshold: R*M_ROW == 2**31 exactly. At R=2048 that is
    # M_ROW = 1,048,576 tokens on a single rank -- unreachable under balanced
    # routing, but real routing at bs8/seq8192 concentrates enough tokens on one
    # rank to cross it (it took down that sweep cell twice).
    #
    # Passing the operands 2-D instead does NOT fix it: tested, and the result
    # is garbage (SQNR -2.26 dB), because every offset in the body is a flat
    # element index. A real fix needs the addressing reworked to 2-D, or the
    # contraction split so no single launch sees 2**31 elements.
    #
    # UNDER torch.compile this cannot be a Python branch: M_ROW is the routed
    # token count, an unbacked SymInt, so `biggest > INT32_MAX` is a
    # data-dependent guard Dynamo cannot resolve and the whole compiled path
    # dies. Assert the precondition instead -- torch._check registers a deferred
    # runtime assert (an over-limit operand still fails loudly rather than
    # wrapping) and teaches the shape env the fact, so the comparison resolves
    # statically. Eager keeps the graceful fallback.
    INT32_MAX = 2**31 - 1
    # The operands go over as int8 (BYTE) views, so the packed width is what the
    # CABI actually counts -- fp4 buys back a factor of 2 of headroom here.
    biggest = max(R, C) * M_PACKED
    if torch.compiler.is_compiling():
        torch._check(biggest <= INT32_MAX)
    elif biggest > INT32_MAX:
        raise NotImplementedError(
            f"operand has {biggest} elements, over the int32 limit FlyDSL's "
            f"argument packing imposes ({INT32_MAX}); R={R} C={C} "
            f"M_ROW={M_ROW}"
        )

    # The scale rows are indexed at the *operand's* row stride, so they must be
    # the matching full-width e8m0 planes, not ones pre-sliced to m_total.
    assert go_scale.shape == (
        R,
        M_ROW // SCALE_BLOCK,
    ), f"go_scale {tuple(go_scale.shape)} != {(R, M_ROW // SCALE_BLOCK)}"
    assert ia_scale.shape == (
        C,
        M_ROW // SCALE_BLOCK,
    ), f"ia_scale {tuple(ia_scale.shape)} != {(C, M_ROW // SCALE_BLOCK)}"

    n_r = ceildiv(R, WGRAD_BLOCK_R)
    n_c = ceildiv(C, BLOCK_C)
    n_blocks = E * n_r * n_c

    # torch.empty, not zeros: one block owns each (g, tile) and stores every valid
    # (row, col) exactly once, including zeros for empty groups. A chunked
    # schedule needs zeros
    # because its atomic epilogue accumulates onto the buffer; this one does not.
    out = torch.empty(E, R, C, dtype=torch.float32, device=go_t.device)
    # Read per call, not at import: the bench flips it between interleaved arms
    # inside one process, and lru_cache keys on it so every variant coexists.
    order = (
        _order
        if _order is not None
        else (
            "lpt"
            if os.environ.get("AITER_FLYDSL_MXFP4_WGRAD_LPT", "1") != "0"
            else "id"
        )
    )
    # The remap ranks every group against every other -- E*E selects and E scalar
    # loads in every block's prologue. At E=8 that is tens of SALU ops
    # against a 100+ iteration K-walk and one 32-byte line of
    # offsets. At E in the hundreds it is a large unrolled prologue, a slow
    # compile, and E loads per block, for a benefit that shrinks anyway (more
    # groups per CU averages the imbalance out on its own).
    #
    # Falls back to `id`, not `idsel`: `idsel` is also O(E) loads, and its
    # measured gain is ~1%, so it is not worth 256 scalar loads per block. `id`
    # is the O(1) mapping and is what every number before 2026-07-31 was taken
    # on. All of this is unmeasured above E=8 -- the cutoff is a guard against
    # a shape nobody has run, not a tuned value.
    if order not in ORDER_MODES:
        raise ValueError(f"order must be one of {ORDER_MODES}, got {order!r}")
    if order == "lpt" and E > LPT_MAX_E:
        order = "id"
    launch = _cached_launch_wgrad(R, C, E, sc_pair, order)
    _run_compiled(
        launch,
        go_t.contiguous().view(torch.int8).reshape(-1),
        ia_t.contiguous().view(torch.int8).reshape(-1),
        out.view(-1),
        go_scale.contiguous().view(torch.uint8).view(-1),
        ia_scale.contiguous().view(torch.uint8).view(-1),
        group_end_offsets.to(torch.int32).contiguous(),
        n_blocks,
        n_r,
        n_c,
        R,
        C,
        M_ROW,
        torch.cuda.current_stream(),
    )
    return out.to(out_dtype)
