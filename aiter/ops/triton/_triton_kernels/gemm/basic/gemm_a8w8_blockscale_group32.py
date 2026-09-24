# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Native E4M3/E8M0 group32 GEMMs, including packed-K small-M reductions."""

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config

_gemm_group32_repr = make_kernel_repr(
    "_gemm_a8w8_blockscale_group32_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_N",
        "SPLITK_BLOCK_SIZE",
        "FUSED_SPLITS",
        "N_FIRST",
        "N",
        "K",
        "LAUNCH_OPTIONS",
    ],
)
_gemm_group32_packed_repr = make_kernel_repr(
    "_gemm_a8w8_blockscale_group32_packed_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "K_PACK",
        "SPLITK_BLOCK_SIZE",
        "FUSED_SPLITS",
        "N",
        "K",
        "LAUNCH_OPTIONS",
    ],
)


@triton.jit
def _tile_on_xcd(N: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, SPLITS: tl.constexpr):
    """(pid_m, pid_n, tile, split) of a 1D grid, all splits of a tile on one XCD.

    CTAs go to XCDs by pid % 8, a multiple of every gfx950 XCD count.
    """
    pid = tl.program_id(0)
    tile = (pid // 8 // SPLITS) * 8 + pid % 8
    grid_n: tl.constexpr = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    return tile // grid_n, tile % grid_n, tile, pid // 8 % SPLITS


@triton.jit
def _sum_splits_on_xcd(
    partial, c_ptrs, mask, slot, counter, split, SPLITS: tl.constexpr
):
    """The tile's last CTA to arrive sums its splits in split order.

    The splits share one XCD's L2, so a completed store is visible without an
    agent-scope release and its L2 writeback.
    """
    BM: tl.constexpr = partial.shape[0]
    BN: tl.constexpr = partial.shape[1]
    local = tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]
    tl.store(slot + split * BM * BN + local, partial)
    # debug_barrier is a bare s_barrier: drain this wave's stores first.
    tl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)", "=v,v", [split], dtype=tl.int32, is_pure=False, pack=1
    )
    tl.debug_barrier()
    if tl.atomic_add(counter, 1, sem="acq_rel", scope="cta") == SPLITS - 1:
        total = tl.zeros((BM, BN), tl.float32)
        for s in tl.static_range(SPLITS):
            # .cv skips this CU's L1, which never saw the other splits.
            total += tl.load(slot + s * BM * BN + local, cache_modifier=".cv")
        tl.store(c_ptrs, total.to(c_ptrs.dtype.element_ty), mask)
        tl.store(counter, 0)


@triton.jit(repr=_gemm_group32_repr, do_not_specialize=["M"])
def _gemm_a8w8_blockscale_group32_kernel(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    c_ptr,
    ws_ptr,
    cnt_ptr,
    # Runtime, not constexpr: M is the token count, and specializing on it
    # recompiles the kernel for every prefill chunk length.
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_N: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    LAUNCH_OPTIONS: tl.constexpr,
    N_FIRST: tl.constexpr = False,
    FUSED_SPLITS: tl.constexpr = 1,
):
    """E4M3 x E4M3 with E8M0 group scales, on the CDNA4 microscaling MFMA.

    A tile spans BLOCK_SIZE_K/32 scale groups. Below a K tile of 64,
    dot_scaled lowers to slower BF16 emulation instead of microscaling MFMA.
    FUSED_SPLITS > 1 sums split-K partials in this launch on a 1D grid;
    otherwise c_ptr takes one FP32 partial per program_id(2).
    """
    tl.static_assert(BLOCK_SIZE_K >= 64 and BLOCK_SIZE_K % 32 == 0)
    if FUSED_SPLITS > 1:
        pid_m, pid_n, tile, split = _tile_on_xcd(N, BLOCK_SIZE_N, FUSED_SPLITS)
        if pid_m * BLOCK_SIZE_M >= M:
            return
    else:
        # The first grid dimension advances fastest. N-first traversal reuses
        # A; M-first traversal reuses B. The wrapper swaps the launch dimensions.
        pid_m = tl.program_id(1 if N_FIRST else 0)
        pid_n = tl.program_id(0 if N_FIRST else 1)
        split = tl.program_id(2)
    row = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    groups: tl.constexpr = K // 32
    ks = tl.arange(0, BLOCK_SIZE_K)
    gs = tl.arange(0, BLOCK_SIZE_K // 32)
    rows = row[:, None] < M
    cols = col < N
    # One scale row per GROUP_N output columns, so the column index is
    # divided rather than the grid expanded.
    bs_row = (col[:, None] // GROUP_N) * groups

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32)
    start = split * SPLITK_BLOCK_SIZE
    for base in range(start, tl.minimum(start + SPLITK_BLOCK_SIZE, K), BLOCK_SIZE_K):
        offs = base + ks
        span = base // 32 + gs
        live = offs < K
        held = span < groups
        a = tl.load(
            a_ptr + row[:, None] * K + offs[None, :], rows & live[None, :], other=0.0
        )
        b = tl.load(
            b_ptr + col[:, None] * K + offs[None, :],
            cols[:, None] & live[None, :],
            other=0.0,
        )
        a_code = tl.load(
            a_scale_ptr + row[:, None] * groups + span[None, :],
            rows & held[None, :],
            other=127,
        )
        b_code = tl.load(
            b_scale_ptr + bs_row + span[None, :],
            cols[:, None] & held[None, :],
            other=127,
        )
        # acc= leaves the sum in the matrix core's registers: one rounding per
        # tile instead of two, and no separate vector add.
        accumulator = tl.dot_scaled(
            a, a_code, "e4m3", b.T, b_code, "e4m3", acc=accumulator
        )
    c_ptrs = c_ptr + row[:, None] * N + col[None, :]
    if FUSED_SPLITS > 1:
        _sum_splits_on_xcd(
            accumulator,
            c_ptrs,
            rows & cols[None, :],
            ws_ptr + tile * (FUSED_SPLITS * BLOCK_SIZE_M * BLOCK_SIZE_N),
            cnt_ptr + tile,
            split,
            FUSED_SPLITS,
        )
    else:
        tl.store(c_ptrs + split * M * N, accumulator, rows & cols[None, :])


@triton.jit(repr=_gemm_group32_packed_repr, do_not_specialize=["M"])
def _gemm_a8w8_blockscale_group32_packed_kernel(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    c_ptr,
    ws_ptr,
    cnt_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    K_PACK: tl.constexpr,
    LAUNCH_OPTIONS: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr = 0,
    FUSED_SPLITS: tl.constexpr = 1,
):
    """Small-M group32 GEMM with K panels packed into MFMA rows/columns.

    Each packed row/column retains its own E8M0 scales. Only matching panel
    pairs contribute to the output; cross-panel products are discarded. One
    CTA owns the K reduction, or with FUSED_SPLITS > 1 one SPLITK_BLOCK_SIZE
    of it, summed in this launch. BLOCK_SIZE_M is an unpacked token tile,
    independent of runtime M.
    """
    tl.static_assert(K_PACK == 1 or K_PACK == 2 or K_PACK == 4)
    tl.static_assert(BLOCK_SIZE_M * K_PACK >= 16)
    tl.static_assert(BLOCK_SIZE_K >= 128 and BLOCK_SIZE_K % 32 == 0)
    STEP: tl.constexpr = BLOCK_SIZE_K * K_PACK
    if FUSED_SPLITS > 1:
        tl.static_assert(SPLITK_BLOCK_SIZE % STEP == 0)
        pid_m, pid_n, tile, split = _tile_on_xcd(N, BLOCK_SIZE_N, FUSED_SPLITS)
        if pid_m * BLOCK_SIZE_M >= M:
            return
        start = split * SPLITK_BLOCK_SIZE
        stop = tl.minimum(start + SPLITK_BLOCK_SIZE, K)
    else:
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        start = 0
        stop = K
    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M * K_PACK) // K_PACK
    a_panel = tl.arange(0, BLOCK_SIZE_M * K_PACK) % K_PACK
    cols = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N * K_PACK) // K_PACK
    b_panel = tl.arange(0, BLOCK_SIZE_N * K_PACK) % K_PACK
    ks = tl.arange(0, BLOCK_SIZE_K)
    gs = tl.arange(0, BLOCK_SIZE_K // 32)
    groups: tl.constexpr = K // 32
    accumulator = tl.zeros((BLOCK_SIZE_M * K_PACK, BLOCK_SIZE_N * K_PACK), tl.float32)
    for base in range(start, stop, STEP):
        ak = base + a_panel[:, None] * BLOCK_SIZE_K + ks[None, :]
        bk = base + b_panel[:, None] * BLOCK_SIZE_K + ks[None, :]
        a = tl.load(
            a_ptr + rows[:, None] * K + ak,
            (rows[:, None] < M) & (ak < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + cols[:, None] * K + bk,
            (cols[:, None] < N) & (bk < K),
            other=0.0,
        )
        ag = base // 32 + a_panel[:, None] * (BLOCK_SIZE_K // 32) + gs[None, :]
        bg = base // 32 + b_panel[:, None] * (BLOCK_SIZE_K // 32) + gs[None, :]
        a_code = tl.load(
            a_scale_ptr + rows[:, None] * groups + ag,
            (rows[:, None] < M) & (ag < groups),
            other=127,
        )
        b_code = tl.load(
            b_scale_ptr + (cols[:, None] // 32) * groups + bg,
            (cols[:, None] < N) & (bg < groups),
            other=127,
        )
        accumulator = tl.dot_scaled(
            a, a_code, "e4m3", b.T, b_code, "e4m3", acc=accumulator
        )
    panels = accumulator.reshape(BLOCK_SIZE_M, K_PACK, BLOCK_SIZE_N, K_PACK).trans(
        0, 2, 1, 3
    )
    pair = tl.arange(0, K_PACK)
    diagonal = tl.where(
        pair[None, None, :, None] == pair[None, None, None, :], panels, 0.0
    )
    output = tl.sum(tl.sum(diagonal, 3), 2)
    row = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + row[:, None] * N + col[None, :]
    mask = (row[:, None] < M) & (col[None, :] < N)
    if FUSED_SPLITS > 1:
        _sum_splits_on_xcd(
            output,
            c_ptrs,
            mask,
            ws_ptr + tile * (FUSED_SPLITS * BLOCK_SIZE_M * BLOCK_SIZE_N),
            cnt_ptr + tile,
            split,
            FUSED_SPLITS,
        )
    else:
        tl.store(c_ptrs, output, mask)


def _get_config(M: int, N: int, K: int):
    return get_gemm_config("GEMM-A8W8_BLOCKSCALE_GROUP32", M, N, K)
