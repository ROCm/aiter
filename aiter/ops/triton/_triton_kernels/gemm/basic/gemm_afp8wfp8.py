# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config

_gemm_afp8wfp8_repr = make_kernel_repr(
    "_gemm_afp8wfp8_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "A_SCALE_K_GROUP",
        "B_SCALE_N_GROUP",
        "B_SCALE_K_GROUP",
        "FUSED_SPLITS",
        "N_FIRST",
        "num_warps",
        "num_stages",
        "waves_per_eu",
        "matrix_instr_nonkdim",
        "cache_modifier",
        "NUM_KSPLIT",
        "SPLITK_BLOCK_SIZE",
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: (args["K"] % args["BLOCK_SIZE_K"] == 0),
        "BROADCAST_A": lambda args: args["M"] == 1,
    }
)
@triton.jit(repr=_gemm_afp8wfp8_repr)
def _gemm_afp8wfp8_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scales_ptr,
    b_scales_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_ck: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_asm: tl.constexpr,
    stride_ask: tl.constexpr,
    stride_bsn: tl.constexpr,
    stride_bsk: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    A_SCALE_K_GROUP: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    BROADCAST_A: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    waves_per_eu: tl.constexpr,
    matrix_instr_nonkdim: tl.constexpr,
    cache_modifier: tl.constexpr,
    B_SCALE_N_GROUP: tl.constexpr = 128,
    B_SCALE_K_GROUP: tl.constexpr = 128,
    N_FIRST: tl.constexpr = False,
    FUSED_SPLITS: tl.constexpr = 1,
    ws_ptr=None,
    cnt_ptr=None,
):
    """
    Kernel for computing the matmul C = A x B.
    A and B inputs are FP8 e4m3 (1 byte per element).
    A_scales are e8m0 (uint8) with shape (M, K // A_SCALE_K_GROUP), where
    A_SCALE_K_GROUP is 32 for MX activations or 128 for blockscale activations;
    coarser-than-32 scales are broadcast to the 32-element groups tl.dot_scaled
    requires. The caller folds a transposed scale buffer into stride_asm /
    stride_ask, so both layouts are handled here identically.
    B_scales are stored compact e8m0 (uint8) with shape (ceil(N / B_SCALE_N_GROUP), K // B_SCALE_K_GROUP),
    representing configurable weight blocks. Broadcast inside kernel to (N, K // 32).
    A has shape (M, K), B has shape (K, N) and C has shape (M, N).
    Output dtype is determined by c_ptr (bf16, fp16, or fp32).
    When NUM_KSPLIT > 1, K is split into NUM_KSPLIT partitions of
    SPLITK_BLOCK_SIZE elements and the partial result for partition pid_k is
    written to c_ptr + pid_k * stride_ck; a downstream reduce kernel sums them.
    With FUSED_SPLITS > 1, the last CTA combines FP32 partials in this launch.
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_asm > 0)
    tl.assume(stride_ask > 0)
    tl.assume(stride_bsk > 0)
    tl.assume(stride_bsn > 0)

    GRID_MN = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)

    pid_unified = tl.program_id(axis=0)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if FUSED_SPLITS > 1:
        pid_m, pid_n, tile, pid_k = _tile_on_xcd(N, BLOCK_SIZE_N, FUSED_SPLITS)
        if pid_m * BLOCK_SIZE_M >= M:
            return
    elif N_FIRST:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    elif NUM_KSPLIT == 1:
        pid = remap_xcd(pid, GRID_MN, NUM_XCDS=8)
        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)

    # Scale group sizes
    SCALE_GROUP_SIZE: tl.constexpr = 32  # A: per 32 elements along K
    tl.static_assert(BLOCK_SIZE_K >= 64 and BLOCK_SIZE_K % 32 == 0)
    tl.static_assert(SPLITK_BLOCK_SIZE % BLOCK_SIZE_K == 0)

    if (pid_k * SPLITK_BLOCK_SIZE) < K:
        # K-block iteration range for this split (absolute block indices).
        num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE, BLOCK_SIZE_K)

        # Create pointers for first block of A and B input matrices. The K
        # offset is the absolute start of this split's K range.
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_k_split = pid_k * SPLITK_BLOCK_SIZE + offs_k
        # Preserve the original M=1 broadcast layout; other tiles use masked
        # rows rather than runtime modulo, avoiding repeated padded loads.
        if BROADCAST_A:
            offs_am = tl.full((BLOCK_SIZE_M,), 0, tl.int32)
        else:
            offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        a_ptrs = a_ptr + (
            offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak
        )
        b_ptrs = b_ptr + (
            offs_k_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        )

        # A-scale row offsets. The K index is computed per-iteration below from
        # absolute K, so a scale group coarser than 32 is simply read by several
        # of the 32-element groups (and split-K addresses the right group).
        offs_asm = offs_am * stride_asm

        # Compact weight scales are broadcast by their logical N/K groups.
        # K indices are computed per-iteration below
        # using absolute K (so split-K naturally addresses the right b-scale block).
        offs_bsn = offs_bn // B_SCALE_N_GROUP  # (BLOCK_SIZE_N,)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        offs_scale_k_a = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_SIZE)

        for k in range(
            pid_k * num_k_iter,
            tl.minimum((pid_k + 1) * num_k_iter, tl.cdiv(K, BLOCK_SIZE_K)),
        ):
            # K base for this iteration (in elements, absolute).
            k_base = k * BLOCK_SIZE_K

            # ---- Load A scales (BLOCK_SIZE_M, BLOCK_SIZE_K // 32) ----
            offs_ask = (k_base // SCALE_GROUP_SIZE + offs_scale_k_a) // (
                A_SCALE_K_GROUP // SCALE_GROUP_SIZE
            )  # (BLOCK_SIZE_K // 32,)
            a_scale_ptrs = (
                a_scales_ptr + offs_asm[:, None] + offs_ask[None, :] * stride_ask
            )
            if EVEN_K:
                a_scales = tl.load(a_scale_ptrs, mask=offs_am[:, None] < M, other=127)
            else:
                a_scale_mask = offs_scale_k_a[None, :] < (
                    K // SCALE_GROUP_SIZE - k * (BLOCK_SIZE_K // SCALE_GROUP_SIZE)
                )
                a_scales = tl.load(
                    a_scale_ptrs, mask=a_scale_mask & (offs_am[:, None] < M), other=127
                )

            # ---- Load and broadcast B scales (BLOCK_SIZE_N, BLOCK_SIZE_K // 32) ----
            offs_bsk = (k_base // SCALE_GROUP_SIZE + offs_scale_k_a) // (
                B_SCALE_K_GROUP // SCALE_GROUP_SIZE
            )  # (BLOCK_SIZE_K // 32,)
            b_scale_ptrs = (
                b_scales_ptr
                + offs_bsn[:, None] * stride_bsn
                + offs_bsk[None, :] * stride_bsk
            )
            if EVEN_K:
                b_scales = tl.load(
                    b_scale_ptrs,
                    mask=offs_bn[:, None] < N,
                    other=127,
                    cache_modifier=cache_modifier,
                )
            else:
                # OOB along K: load with the same mask as a-scales
                b_scale_mask = offs_scale_k_a[None, :] < (
                    K // SCALE_GROUP_SIZE - k * (BLOCK_SIZE_K // SCALE_GROUP_SIZE)
                )
                b_scales = tl.load(
                    b_scale_ptrs,
                    mask=b_scale_mask & (offs_bn[:, None] < N),
                    other=127,
                    cache_modifier=cache_modifier,
                )

            # ---- Load A, B data ----
            if EVEN_K:
                a = tl.load(a_ptrs, mask=offs_am[:, None] < M, other=0.0)
                b = tl.load(
                    b_ptrs,
                    mask=offs_bn[None, :] < N,
                    other=0.0,
                    cache_modifier=cache_modifier,
                )
            else:
                a = tl.load(
                    a_ptrs,
                    mask=(offs_k[None, :] < K - k * BLOCK_SIZE_K)
                    & (offs_am[:, None] < M),
                    other=0.0,
                )
                b = tl.load(
                    b_ptrs,
                    mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K)
                    & (offs_bn[None, :] < N),
                    other=0.0,
                    cache_modifier=cache_modifier,
                )

            accumulator = tl.dot_scaled(
                a, a_scales, "e4m3", b, b_scales, "e4m3", accumulator
            )

            # Advance the ptrs to the next K block (scale ptrs are rebuilt from
            # absolute K each iteration).
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        c = accumulator.to(c_ptr.type.element_ty)

        # Write back the block of the output matrix C with masks. For
        # NUM_KSPLIT > 1, each pid_k writes to a separate slab of c_ptr.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        c_ptrs = (
            c_ptr
            + stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
            + pid_k * stride_ck
        )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if FUSED_SPLITS > 1:
            _sum_splits_on_xcd(
                accumulator,
                c_ptrs,
                c_mask,
                ws_ptr + tile * (FUSED_SPLITS * BLOCK_SIZE_M * BLOCK_SIZE_N),
                cnt_ptr + tile,
                pid_k,
                FUSED_SPLITS,
            )
        else:
            tl.store(c_ptrs, c, mask=c_mask)


_gemm_afp8wfp8_packed_repr = make_kernel_repr(
    "_gemm_afp8wfp8_packed_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "K_PACK",
        "A_SCALE_K_GROUP",
        "B_SCALE_N_GROUP",
        "B_SCALE_K_GROUP",
        "SPLITK_BLOCK_SIZE",
        "FUSED_SPLITS",
        "B_CACHE_MODIFIER",
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


@triton.jit(repr=_gemm_afp8wfp8_packed_repr, do_not_specialize=["M"])
def _gemm_afp8wfp8_packed_kernel(
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
    B_CACHE_MODIFIER: tl.constexpr = None,
    A_SCALE_K_GROUP: tl.constexpr = 32,
    B_SCALE_N_GROUP: tl.constexpr = 32,
    B_SCALE_K_GROUP: tl.constexpr = 32,
    stride_am: tl.constexpr = 0,
    stride_ak: tl.constexpr = 0,
    stride_bn: tl.constexpr = 0,
    stride_bk: tl.constexpr = 0,
    stride_asm: tl.constexpr = 0,
    stride_ask: tl.constexpr = 0,
    stride_bsn: tl.constexpr = 0,
    stride_bsk: tl.constexpr = 0,
    stride_cm: tl.constexpr = 0,
    stride_cn: tl.constexpr = 0,
    stride_ck: tl.constexpr = 0,
    NUM_KSPLIT: tl.constexpr = 1,
):
    """Small-M E4M3 GEMM with K panels packed into MFMA rows/columns.

    Each packed row/column retains its own E8M0 scales. Only matching panel
    pairs contribute to the output; cross-panel products are discarded. One
    CTA owns the K reduction, or with FUSED_SPLITS > 1 one SPLITK_BLOCK_SIZE
    of it, summed in this launch. BLOCK_SIZE_M is an unpacked token tile,
    independent of runtime M. B_CACHE_MODIFIER applies to the weight and
    weight-scale loads.
    """
    tl.static_assert(K_PACK == 1 or K_PACK == 2 or K_PACK == 4)
    tl.static_assert(BLOCK_SIZE_M * K_PACK >= 16)
    tl.static_assert(BLOCK_SIZE_K >= 128 and BLOCK_SIZE_K % 32 == 0)
    STEP: tl.constexpr = BLOCK_SIZE_K * K_PACK
    tl.static_assert(SPLITK_BLOCK_SIZE % STEP == 0)
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
        if NUM_KSPLIT == 1:
            # Preserve a compile-time trip count for the common unsplit path.
            split = 0
            start = 0
            stop = K
        else:
            split = tl.program_id(2)
            start = split * SPLITK_BLOCK_SIZE
            stop = tl.minimum(start + SPLITK_BLOCK_SIZE, K)
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
            a_ptr + rows[:, None] * stride_am + ak * stride_ak,
            (rows[:, None] < M) & (ak < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + cols[:, None] * stride_bn + bk * stride_bk,
            (cols[:, None] < N) & (bk < K),
            other=0.0,
            cache_modifier=B_CACHE_MODIFIER,
        )
        ag = base // 32 + a_panel[:, None] * (BLOCK_SIZE_K // 32) + gs[None, :]
        bg = base // 32 + b_panel[:, None] * (BLOCK_SIZE_K // 32) + gs[None, :]
        a_code = tl.load(
            a_scale_ptr
            + rows[:, None] * stride_asm
            + (ag // (A_SCALE_K_GROUP // 32)) * stride_ask,
            (rows[:, None] < M) & (ag < groups),
            other=127,
        )
        b_code = tl.load(
            b_scale_ptr
            + (cols[:, None] // B_SCALE_N_GROUP) * stride_bsn
            + (bg // (B_SCALE_K_GROUP // 32)) * stride_bsk,
            (cols[:, None] < N) & (bg < groups),
            other=127,
            cache_modifier=B_CACHE_MODIFIER,
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
    c_ptrs = (
        c_ptr + row[:, None] * stride_cm + col[None, :] * stride_cn + split * stride_ck
    )
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


_gemm_afp8wfp8_preshuffle_repr = make_kernel_repr(
    "_gemm_afp8wfp8_preshuffle_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "A_SCALE_K_GROUP",
        "num_warps",
        "num_stages",
        "waves_per_eu",
        "matrix_instr_nonkdim",
        "cache_modifier",
        "NUM_KSPLIT",
        "SPLITK_BLOCK_SIZE",
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: (args["K"] % args["BLOCK_SIZE_K"] == 0),
    }
)
@triton.jit(repr=_gemm_afp8wfp8_preshuffle_repr)
def _gemm_afp8wfp8_preshuffle_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scales_ptr,
    b_scales_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_ck,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    A_SCALE_K_GROUP: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    waves_per_eu: tl.constexpr,
    matrix_instr_nonkdim: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """
    Preshuffle variant of _gemm_afp8wfp8_kernel. Weight tensor has been shuffled
    via aiter.ops.shuffle.shuffle_weight(layout=(16, 16)) so that 16-row N tiles
    are interleaved with their 32-col K chunks in storage. The kernel loads the
    shuffled tile in storage order (BLOCK_SIZE_N // 16, BLOCK_SIZE_K * 16) then
    reshape+permute+trans inside the kernel to restore logical (K, N) layout
    before tl.dot_scaled. Scales remain in the unshuffled compact 128x128 layout.
    When NUM_KSPLIT > 1, K is split into NUM_KSPLIT partitions of
    SPLITK_BLOCK_SIZE elements; each pid_k writes to c_ptr + pid_k * stride_ck.
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_asm > 0)
    tl.assume(stride_ask > 0)
    tl.assume(stride_bsk > 0)
    tl.assume(stride_bsn > 0)

    GRID_MN = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)

    pid_unified = tl.program_id(axis=0)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if NUM_KSPLIT == 1:
        pid = remap_xcd(pid, GRID_MN, NUM_XCDS=8)
        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)

    SCALE_GROUP_SIZE: tl.constexpr = 32  # A: per-32 along K
    B_SCALE_K_GROUP: tl.constexpr = 128  # B compact: per-128 along K
    B_SCALE_N_GROUP: tl.constexpr = 128  # B compact: per-128 along N

    if (pid_k * SPLITK_BLOCK_SIZE) < K:
        num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE, BLOCK_SIZE_K)

        # A pointers (offset by this split's K start).
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_k_split = pid_k * SPLITK_BLOCK_SIZE + offs_k
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        a_ptrs = a_ptr + (
            offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak
        )

        # B pointers for preshuffled layout. The shuffled storage is viewed as
        # (N // 16, K * 16) elements. pid_n indexes BLOCK_SIZE_N // 16 N-tiles per
        # step; the K dimension is expanded by 16x in byte addresses. The split
        # offsets the K-byte axis by pid_k * SPLITK_BLOCK_SIZE * 16.
        offs_bn_shuffle = pid_n * (BLOCK_SIZE_N // 16) + tl.arange(
            0, BLOCK_SIZE_N // 16
        )
        offs_k_shuffle_arr = tl.arange(0, BLOCK_SIZE_K * 16)
        offs_k_shuffle = pid_k * SPLITK_BLOCK_SIZE * 16 + offs_k_shuffle_arr
        b_ptrs = b_ptr + (
            offs_bn_shuffle[:, None] * stride_bn + offs_k_shuffle[None, :] * stride_bk
        )

        # A-scale row offsets. The K index is computed per-iteration below from
        # absolute K, so a scale group coarser than 32 is simply read by several
        # of the 32-element groups (and split-K addresses the right group).
        offs_asm = offs_am * stride_asm

        # B-scale pointers: compact (N // 128, K // 128). The N index needs the
        # ORIGINAL (logical) row, not the shuffled row index.
        offs_bn_logical = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_bsn = offs_bn_logical // B_SCALE_N_GROUP

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        offs_scale_k_a = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_SIZE)

        for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
            k_base = k * BLOCK_SIZE_K  # absolute K base

            # Load A scales (broadcast when A_SCALE_K_GROUP > 32).
            offs_ask = (k_base + offs_scale_k_a * SCALE_GROUP_SIZE) // A_SCALE_K_GROUP
            a_scale_ptrs = (
                a_scales_ptr + offs_asm[:, None] + offs_ask[None, :] * stride_ask
            )
            if EVEN_K:
                a_scales = tl.load(a_scale_ptrs)
            else:
                a_scale_mask = offs_scale_k_a[None, :] < (
                    K // SCALE_GROUP_SIZE - k * (BLOCK_SIZE_K // SCALE_GROUP_SIZE)
                )
                a_scales = tl.load(a_scale_ptrs, mask=a_scale_mask, other=127)

            # Load and broadcast B scales (computed from absolute K).
            offs_bsk = (k_base + offs_scale_k_a * SCALE_GROUP_SIZE) // B_SCALE_K_GROUP
            b_scale_ptrs = (
                b_scales_ptr
                + offs_bsn[:, None] * stride_bsn
                + offs_bsk[None, :] * stride_bsk
            )
            if EVEN_K:
                b_scales = tl.load(b_scale_ptrs, cache_modifier=cache_modifier)
            else:
                b_scale_mask = offs_scale_k_a[None, :] < (
                    K // SCALE_GROUP_SIZE - k * (BLOCK_SIZE_K // SCALE_GROUP_SIZE)
                )
                b_scales = tl.load(
                    b_scale_ptrs,
                    mask=b_scale_mask,
                    other=127,
                    cache_modifier=cache_modifier,
                )

            # Load A and B (preshuffled).
            if EVEN_K:
                a = tl.load(a_ptrs)
                b_shuf = tl.load(b_ptrs, cache_modifier=cache_modifier)
            else:
                a = tl.load(
                    a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0
                )
                b_shuf = tl.load(
                    b_ptrs,
                    mask=offs_k_shuffle_arr[None, :] < (K - k * BLOCK_SIZE_K) * 16,
                    other=0,
                    cache_modifier=cache_modifier,
                )

            # Unshuffle B in-kernel. Inverse of the shuffle_weight permute:
            # shuffle:   (N // 16, 16, K // 32, 2, 16) --[perm 0,1,3,4,2,5]-> (N // 16, K // 32, 2, 16, 16)
            # unshuffle: (N // 16, K // 32, 2, 16, 16) --[perm 0,1,4,2,3,5]-> (N // 16, 16, K // 32, 2, 16)
            # then flatten to (N, K) and trans to (K, N).
            b = (
                b_shuf.reshape(
                    1,
                    BLOCK_SIZE_N // 16,
                    BLOCK_SIZE_K // 32,
                    2,
                    16,
                    16,
                )
                .permute(0, 1, 4, 2, 3, 5)
                .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)
                .trans(1, 0)
            )

            accumulator = tl.dot_scaled(
                a, a_scales, "e4m3", b, b_scales, "e4m3", accumulator
            )

            # Advance pointers (scale ptrs are rebuilt from absolute K each iter).
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * 16 * stride_bk

        c = accumulator.to(c_ptr.type.element_ty)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        c_ptrs = (
            c_ptr
            + stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
            + pid_k * stride_ck
        )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


def _get_config(
    M: int,
    N: int,
    K: int,
    shuffle: bool = False,
    backend: str = "triton",
    x_scale_group_size: int = 128,
    w_scale_group_size: tuple[int, int] = (128, 128),
):
    # backend selects the per-backend config dir (<arch>/<backend>/gemm/), so the
    # triton and gluon kernels can carry different tuning keys for the same shape.
    if shuffle:
        return get_gemm_config("GEMM-AFP8WFP8_PRESHUFFLED", M, N, K, backend=backend)
    else:
        name = "GEMM-AFP8WFP8"
        if (x_scale_group_size, w_scale_group_size) != (128, (128, 128)):
            gn, gk = w_scale_group_size
            name += f"_A{x_scale_group_size}_W{gn}X{gk}"
        return get_gemm_config(name, M, N, K, backend=backend)
