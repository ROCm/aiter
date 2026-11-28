# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from AMD-Triton/gfx12-gluon-tutorials, gemm_4waves, 16f17ab11f4e.

import functools

from triton._C.libtriton.gluon_ir import make_cga_layout
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_multicast_repr = make_kernel_repr(
    "_batched_gemm_bf16_multicast_kernel",
    [
        "LOCAL_M",
        "LOCAL_N",
        "BLOCK_K",
        "CTA_M",
        "CTA_N",
        "EVEN_K",
        "TILES_ODD",
        "num_warps",
        "num_ctas",
    ],
)


@gluon.jit
def _halves(a_desc, b_desc, LOCAL_M: gl.constexpr, LOCAL_N: gl.constexpr):
    a_bot = gl.amd.gfx1250.tdm.update_tensor_descriptor(
        a_desc, add_offsets=[0, LOCAL_M // 2, 0], pred=True, clamp_bounds=True
    )
    b_right = gl.amd.gfx1250.tdm.update_tensor_descriptor(
        b_desc, add_offsets=[0, LOCAL_N // 2, 0], pred=True, clamp_bounds=True
    )
    return (a_bot, b_right)


@gluon.jit
def _issue(a_desc, a_bot, b_desc, b_right, smemA, smemB, SLOT: gl.constexpr):
    # Each wave issues one half-load. Zero CGA bases multicast across that axis.
    gl.amd.gfx1250.tdm.async_load_fused(
        [
            (a_desc, smemA.index(2 * SLOT), 1),
            (a_bot, smemA.index(2 * SLOT + 1), 2),
            (b_desc, smemB.index(2 * SLOT), 4),
            (b_right, smemB.index(2 * SLOT + 1), 8),
        ]
    )


@gluon.jit
def _advance(
    a_desc,
    b_desc,
    t,
    NUM_TILES,
    K,
    CTA_M: gl.constexpr,
    CTA_N: gl.constexpr,
    LOCAL_M: gl.constexpr,
    LOCAL_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    EVEN_K: gl.constexpr,
):
    step = (t + 1 <= NUM_TILES - 1).to(gl.int32)
    a_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
        a_desc, add_offsets=[0, 0, step * BLOCK_K]
    )
    b_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
        b_desc, add_offsets=[0, 0, step * BLOCK_K]
    )
    t = t + step
    if not EVEN_K:
        a_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            a_desc, set_bounds=[CTA_M, LOCAL_M, K - t * BLOCK_K]
        )
        b_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            b_desc, set_bounds=[CTA_N, LOCAL_N, K - t * BLOCK_K]
        )
    return (a_desc, b_desc, t)


@gluon.jit
def _load_next_tile(
    a_desc,
    b_desc,
    smemA,
    smemB,
    SLOT: gl.constexpr,
    t,
    NUM_TILES,
    K,
    CTA_M: gl.constexpr,
    CTA_N: gl.constexpr,
    LOCAL_M: gl.constexpr,
    LOCAL_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    EVEN_K: gl.constexpr,
):
    a_bot, b_right = _halves(a_desc, b_desc, LOCAL_M, LOCAL_N)
    _issue(a_desc, a_bot, b_desc, b_right, smemA, smemB, SLOT)
    return _advance(
        a_desc, b_desc, t, NUM_TILES, K, CTA_M, CTA_N, LOCAL_M, LOCAL_N, BLOCK_K, EVEN_K
    )


@gluon.jit
def _A(
    smemA,
    SLOT: gl.constexpr,
    ROW: gl.constexpr,
    KOFF: gl.constexpr,
    HALF_M: gl.constexpr,
    CHUNK: gl.constexpr,
    CTA_M: gl.constexpr,
    layout: gl.constexpr,
):
    half = smemA.index(2 * SLOT + ROW // HALF_M)
    return (
        half.reshape([CTA_M * HALF_M, 2 * CHUNK])
        .slice(KOFF, CHUNK, dim=1)
        .load(layout=layout)
    )


@gluon.jit
def _B(
    smemB,
    SLOT: gl.constexpr,
    KOFF: gl.constexpr,
    COL: gl.constexpr,
    HALF_N: gl.constexpr,
    CHUNK: gl.constexpr,
    CTA_N: gl.constexpr,
    layout: gl.constexpr,
):
    half = smemB.index(2 * SLOT + COL // HALF_N)
    return (
        half.reshape([CTA_N * HALF_N, 2 * CHUNK])
        .slice(KOFF, CHUNK, dim=1)
        .permute([1, 0])
        .load(layout=layout)
    )


@gluon.jit
def _stage(
    smemC,
    acc,
    HALF_M: gl.constexpr,
    HALF_N: gl.constexpr,
    CTA_M: gl.constexpr,
    CTA_N: gl.constexpr,
    dtype: gl.constexpr,
):
    smemC.permute([0, 2, 1, 3]).reshape([CTA_M * HALF_M, CTA_N * HALF_N]).store(
        acc.to(dtype)
    )


@gluon.jit(repr=_multicast_repr)
def _batched_gemm_bf16_multicast_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_ab: gl.constexpr,
    stride_bb: gl.constexpr,
    stride_cb: gl.constexpr,
    CTA_M: gl.constexpr,
    CTA_N: gl.constexpr,
    LOCAL_M: gl.constexpr,
    LOCAL_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    EVEN_K: gl.constexpr,
    TILES_ODD: gl.constexpr,
    SHARED_LAYOUT_A: gl.constexpr,
    SHARED_LAYOUT_B: gl.constexpr,
    SHARED_LAYOUT_C: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
):
    gl.static_assert(
        gl.num_ctas() == CTA_M * CTA_N, "num_ctas must match the cluster shape"
    )
    BLOCK_M: gl.constexpr = CTA_M * LOCAL_M
    BLOCK_N: gl.constexpr = CTA_N * LOCAL_N
    batch = gl.program_id(axis=1)
    a_ptr += batch * stride_ab
    b_ptr += batch * stride_bb
    c_ptr += batch * stride_cb
    pid = gl.program_id(axis=0)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_ptr + pid_m * BLOCK_M * stride_am,
        shape=(CTA_M, LOCAL_M, K),
        strides=(LOCAL_M * stride_am, stride_am, stride_ak),
        block_shape=(CTA_M, LOCAL_M // 2, BLOCK_K),
        layout=SHARED_LAYOUT_A,
    )
    b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * BLOCK_N * stride_bn,
        shape=(CTA_N, LOCAL_N, K),
        strides=(LOCAL_N * stride_bn, stride_bn, stride_bk),
        block_shape=(CTA_N, LOCAL_N // 2, BLOCK_K),
        layout=SHARED_LAYOUT_B,
    )
    c_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn,
        shape=(CTA_M, CTA_N, LOCAL_M, LOCAL_N),
        strides=(LOCAL_M * stride_cm, LOCAL_N * stride_cn, stride_cm, stride_cn),
        block_shape=(CTA_M, CTA_N, LOCAL_M // 2, LOCAL_N // 2),
        layout=SHARED_LAYOUT_C,
    )
    smemA = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, (4, CTA_M, LOCAL_M // 2, BLOCK_K), layout=a_desc.layout
    )
    smemB = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, (4, CTA_N, LOCAL_N // 2, BLOCK_K), layout=b_desc.layout
    )
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=WMMA_LAYOUT, k_width=8
    )
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=WMMA_LAYOUT, k_width=8
    )
    HALF_M: gl.constexpr = LOCAL_M // 2
    HALF_N: gl.constexpr = LOCAL_N // 2
    CHUNK: gl.constexpr = BLOCK_K // 2
    acc_tl = gl.zeros((CTA_M * HALF_M, CTA_N * HALF_N), gl.float32, WMMA_LAYOUT)
    acc_bl = gl.zeros((CTA_M * HALF_M, CTA_N * HALF_N), gl.float32, WMMA_LAYOUT)
    acc_tr = gl.zeros((CTA_M * HALF_M, CTA_N * HALF_N), gl.float32, WMMA_LAYOUT)
    acc_br = gl.zeros((CTA_M * HALF_M, CTA_N * HALF_N), gl.float32, WMMA_LAYOUT)
    TOP: gl.constexpr = 0
    BOT: gl.constexpr = LOCAL_M // 2
    KL: gl.constexpr = 0
    KR: gl.constexpr = BLOCK_K // 2
    LEFT: gl.constexpr = 0
    RIGHT: gl.constexpr = LOCAL_N // 2
    NUM_TILES = gl.cdiv(K, BLOCK_K)
    t = 0
    a_desc, b_desc, t = _load_next_tile(
        a_desc,
        b_desc,
        smemA,
        smemB,
        0,
        t,
        NUM_TILES,
        K,
        CTA_M,
        CTA_N,
        LOCAL_M,
        LOCAL_N,
        BLOCK_K,
        EVEN_K,
    )
    a_desc, b_desc, t = _load_next_tile(
        a_desc,
        b_desc,
        smemA,
        smemB,
        1,
        t,
        NUM_TILES,
        K,
        CTA_M,
        CTA_N,
        LOCAL_M,
        LOCAL_N,
        BLOCK_K,
        EVEN_K,
    )
    gl.amd.gfx1250.tdm.async_wait(1)
    btl = _B(smemB, 0, KL, LEFT, HALF_N, CHUNK, CTA_N, dotOpLayoutB)
    atl = _A(smemA, 0, TOP, KL, HALF_M, CHUNK, CTA_M, dotOpLayoutA)
    # Refill a slot after its last LDS read; wait before reading the other slot.
    for k in range(0, NUM_TILES - 2, 2):
        gl.amd.hint.sched_barrier()
        acc_tl = gl.amd.gfx1250.wmma(atl, btl, acc_tl)
        abl = _A(smemA, 0, BOT, KL, HALF_M, CHUNK, CTA_M, dotOpLayoutA)
        gl.amd.hint.sched_barrier()
        acc_bl = gl.amd.gfx1250.wmma(abl, btl, acc_bl)
        btr = _B(smemB, 0, KL, RIGHT, HALF_N, CHUNK, CTA_N, dotOpLayoutB)
        gl.amd.hint.sched_barrier()
        acc_tr = gl.amd.gfx1250.wmma(atl, btr, acc_tr)
        bbl = _B(smemB, 0, KR, LEFT, HALF_N, CHUNK, CTA_N, dotOpLayoutB)
        gl.amd.hint.sched_barrier()
        acc_br = gl.amd.gfx1250.wmma(abl, btr, acc_br)
        atr = _A(smemA, 0, TOP, KR, HALF_M, CHUNK, CTA_M, dotOpLayoutA)
        gl.amd.hint.sched_barrier()
        acc_tl = gl.amd.gfx1250.wmma(atr, bbl, acc_tl)
        abr = _A(smemA, 0, BOT, KR, HALF_M, CHUNK, CTA_M, dotOpLayoutA)
        gl.amd.hint.sched_barrier()
        acc_bl = gl.amd.gfx1250.wmma(abr, bbl, acc_bl)
        bbr = _B(smemB, 0, KR, RIGHT, HALF_N, CHUNK, CTA_N, dotOpLayoutB)
        a_bot, b_right = _halves(a_desc, b_desc, LOCAL_M, LOCAL_N)
        a_nxt, b_nxt, t_nxt = _advance(
            a_desc,
            b_desc,
            t,
            NUM_TILES,
            K,
            CTA_M,
            CTA_N,
            LOCAL_M,
            LOCAL_N,
            BLOCK_K,
            EVEN_K,
        )
        gl.amd.hint.sched_barrier()
        _issue(a_desc, a_bot, b_desc, b_right, smemA, smemB, 0)
        a_desc, b_desc, t = (a_nxt, b_nxt, t_nxt)
        gl.amd.hint.sched_barrier()
        acc_tr = gl.amd.gfx1250.wmma(atr, bbr, acc_tr)
        gl.amd.gfx1250.tdm.async_wait(1)
        btl = _B(smemB, 1, KL, LEFT, HALF_N, CHUNK, CTA_N, dotOpLayoutB)
        gl.amd.hint.sched_barrier()
        acc_br = gl.amd.gfx1250.wmma(abr, bbr, acc_br)
        atl = _A(smemA, 1, TOP, KL, HALF_M, CHUNK, CTA_M, dotOpLayoutA)
        gl.amd.hint.sched_barrier()
        acc_tl = gl.amd.gfx1250.wmma(atl, btl, acc_tl)
        abl = _A(smemA, 1, BOT, KL, HALF_M, CHUNK, CTA_M, dotOpLayoutA)
        gl.amd.hint.sched_barrier()
        acc_bl = gl.amd.gfx1250.wmma(abl, btl, acc_bl)
        btr = _B(smemB, 1, KL, RIGHT, HALF_N, CHUNK, CTA_N, dotOpLayoutB)
        gl.amd.hint.sched_barrier()
        acc_tr = gl.amd.gfx1250.wmma(atl, btr, acc_tr)
        bbl = _B(smemB, 1, KR, LEFT, HALF_N, CHUNK, CTA_N, dotOpLayoutB)
        gl.amd.hint.sched_barrier()
        acc_br = gl.amd.gfx1250.wmma(abl, btr, acc_br)
        atr = _A(smemA, 1, TOP, KR, HALF_M, CHUNK, CTA_M, dotOpLayoutA)
        gl.amd.hint.sched_barrier()
        acc_tl = gl.amd.gfx1250.wmma(atr, bbl, acc_tl)
        abr = _A(smemA, 1, BOT, KR, HALF_M, CHUNK, CTA_M, dotOpLayoutA)
        gl.amd.hint.sched_barrier()
        acc_bl = gl.amd.gfx1250.wmma(abr, bbl, acc_bl)
        bbr = _B(smemB, 1, KR, RIGHT, HALF_N, CHUNK, CTA_N, dotOpLayoutB)
        a_bot, b_right = _halves(a_desc, b_desc, LOCAL_M, LOCAL_N)
        a_nxt, b_nxt, t_nxt = _advance(
            a_desc,
            b_desc,
            t,
            NUM_TILES,
            K,
            CTA_M,
            CTA_N,
            LOCAL_M,
            LOCAL_N,
            BLOCK_K,
            EVEN_K,
        )
        gl.amd.hint.sched_barrier()
        _issue(a_desc, a_bot, b_desc, b_right, smemA, smemB, 1)
        a_desc, b_desc, t = (a_nxt, b_nxt, t_nxt)
        gl.amd.hint.sched_barrier()
        acc_tr = gl.amd.gfx1250.wmma(atr, bbr, acc_tr)
        gl.amd.gfx1250.tdm.async_wait(1)
        btl = _B(smemB, 0, KL, LEFT, HALF_N, CHUNK, CTA_N, dotOpLayoutB)
        gl.amd.hint.sched_barrier()
        acc_br = gl.amd.gfx1250.wmma(abr, bbr, acc_br)
        atl = _A(smemA, 0, TOP, KL, HALF_M, CHUNK, CTA_M, dotOpLayoutA)
        gl.amd.hint.sched_barrier()
    if not TILES_ODD:
        acc_tl = gl.amd.gfx1250.wmma(atl, btl, acc_tl)
        abl = _A(smemA, 0, BOT, KL, HALF_M, CHUNK, CTA_M, dotOpLayoutA)
        acc_bl = gl.amd.gfx1250.wmma(abl, btl, acc_bl)
        btr = _B(smemB, 0, KL, RIGHT, HALF_N, CHUNK, CTA_N, dotOpLayoutB)
        acc_tr = gl.amd.gfx1250.wmma(atl, btr, acc_tr)
        bbl = _B(smemB, 0, KR, LEFT, HALF_N, CHUNK, CTA_N, dotOpLayoutB)
        acc_br = gl.amd.gfx1250.wmma(abl, btr, acc_br)
        atr = _A(smemA, 0, TOP, KR, HALF_M, CHUNK, CTA_M, dotOpLayoutA)
        acc_tl = gl.amd.gfx1250.wmma(atr, bbl, acc_tl)
        abr = _A(smemA, 0, BOT, KR, HALF_M, CHUNK, CTA_M, dotOpLayoutA)
        acc_bl = gl.amd.gfx1250.wmma(abr, bbl, acc_bl)
        bbr = _B(smemB, 0, KR, RIGHT, HALF_N, CHUNK, CTA_N, dotOpLayoutB)
        acc_tr = gl.amd.gfx1250.wmma(atr, bbr, acc_tr)
        gl.amd.gfx1250.tdm.async_wait(0)
        btl = _B(smemB, 1, KL, LEFT, HALF_N, CHUNK, CTA_N, dotOpLayoutB)
        acc_br = gl.amd.gfx1250.wmma(abr, bbr, acc_br)
        atl = _A(smemA, 1, TOP, KL, HALF_M, CHUNK, CTA_M, dotOpLayoutA)
    LAST: gl.constexpr = 0 if TILES_ODD else 1
    dtype: gl.constexpr = c_ptr.dtype.element_ty
    smemC0 = gl.allocate_shared_memory(
        dtype, (CTA_M, CTA_N, HALF_M, HALF_N), layout=SHARED_LAYOUT_C
    )
    acc_tl = gl.amd.gfx1250.wmma(atl, btl, acc_tl)
    abl = _A(smemA, LAST, BOT, KL, HALF_M, CHUNK, CTA_M, dotOpLayoutA)
    acc_bl = gl.amd.gfx1250.wmma(abl, btl, acc_bl)
    btr = _B(smemB, LAST, KL, RIGHT, HALF_N, CHUNK, CTA_N, dotOpLayoutB)
    acc_tr = gl.amd.gfx1250.wmma(atl, btr, acc_tr)
    bbl = _B(smemB, LAST, KR, LEFT, HALF_N, CHUNK, CTA_N, dotOpLayoutB)
    acc_br = gl.amd.gfx1250.wmma(abl, btr, acc_br)
    atr = _A(smemA, LAST, TOP, KR, HALF_M, CHUNK, CTA_M, dotOpLayoutA)
    # Pipeline the final WMMA quadrants with conversion and TDM stores.
    with gl.amd.warp_pipeline_stage("r4"):
        acc_tl = gl.amd.gfx1250.wmma(atr, bbl, acc_tl)
        abr = _A(smemA, LAST, BOT, KR, HALF_M, CHUNK, CTA_M, dotOpLayoutA)
    with gl.amd.warp_pipeline_stage("r5"):
        acc_bl = gl.amd.gfx1250.wmma(abr, bbl, acc_bl)
        bbr = _B(smemB, LAST, KR, RIGHT, HALF_N, CHUNK, CTA_N, dotOpLayoutB)
        _stage(smemC0, acc_tl, HALF_M, HALF_N, CTA_M, CTA_N, dtype)
    gl.amd.gfx1250.tdm.async_wait(0)
    smemC1 = gl.allocate_shared_memory(
        dtype, (CTA_M, CTA_N, HALF_M, HALF_N), layout=SHARED_LAYOUT_C
    )
    smemC2 = gl.allocate_shared_memory(
        dtype, (CTA_M, CTA_N, HALF_M, HALF_N), layout=SHARED_LAYOUT_C
    )
    smemC3 = gl.allocate_shared_memory(
        dtype, (CTA_M, CTA_N, HALF_M, HALF_N), layout=SHARED_LAYOUT_C
    )
    with gl.amd.warp_pipeline_stage("r6"):
        acc_tr = gl.amd.gfx1250.wmma(atr, bbr, acc_tr)
        gl.amd.gfx1250.tdm.async_store(c_desc, offsets=[0, 0, TOP, LEFT], src=smemC0)
        _stage(smemC1, acc_bl, HALF_M, HALF_N, CTA_M, CTA_N, dtype)
    with gl.amd.warp_pipeline_stage("r7"):
        acc_br = gl.amd.gfx1250.wmma(abr, bbr, acc_br)
        gl.amd.gfx1250.tdm.async_store(c_desc, offsets=[0, 0, BOT, LEFT], src=smemC1)
        _stage(smemC2, acc_tr, HALF_M, HALF_N, CTA_M, CTA_N, dtype)
    with gl.amd.warp_pipeline_stage("stage_br"):
        gl.amd.gfx1250.tdm.async_store(c_desc, offsets=[0, 0, TOP, RIGHT], src=smemC2)
        _stage(smemC3, acc_br, HALF_M, HALF_N, CTA_M, CTA_N, dtype)
        gl.amd.gfx1250.tdm.async_store(c_desc, offsets=[0, 0, BOT, RIGHT], src=smemC3)
    gl.amd.gfx1250.tdm.async_wait(0)


@functools.lru_cache(maxsize=32)
def create_multicast_layouts(CTA_M, CTA_N, LOCAL_M, LOCAL_N, BLOCK_K, NUM_WARPS):
    local_a = gl.PaddedSharedLayout.with_identity_for(
        [[128, 8]], [LOCAL_M, BLOCK_K], [1, 0]
    )
    local_b = gl.PaddedSharedLayout.with_identity_for(
        [[128, 8]], [LOCAL_N, BLOCK_K], [1, 0]
    )
    _, _, local_wmma = gl.amd.cdna5.make_partitioned_dot_layouts(
        LOCAL_M,
        LOCAL_N,
        local_a,
        local_b,
        num_warps=NUM_WARPS,
        instr_shape=[16, 16, 32],
        a_transposed=False,
        b_transposed=True,
        transposed=True,
        slice_m=LOCAL_M // 2,
        slice_n=LOCAL_N // 2,
    )
    mb = [1 << i for i in range(CTA_M.bit_length() - 1)]
    nb = [1 << i for i in range(CTA_N.bit_length() - 1)]
    cga = make_cga_layout([CTA_M, CTA_N], [CTA_M, CTA_N], [0, 1])
    wmma = gl.amd.AMDWMMALayout(
        3,
        local_wmma.transposed,
        local_wmma.warp_bases,
        local_wmma.reg_bases,
        local_wmma.instr_shape,
        cga,
    )
    shared_a = gl.PaddedSharedLayout.with_identity_for(
        [[128, 8]],
        [CTA_M, LOCAL_M // 2, BLOCK_K],
        [2, 1, 0],
        [[i, 0, 0] for i in mb] + [[0, 0, 0] for i in nb],
    )
    shared_b = gl.PaddedSharedLayout.with_identity_for(
        [[128, 8]],
        [CTA_N, LOCAL_N // 2, BLOCK_K],
        [2, 1, 0],
        [[0, 0, 0] for i in mb] + [[i, 0, 0] for i in nb],
    )
    shared_c = gl.PaddedSharedLayout.with_identity_for(
        [[LOCAL_N // 2, 8]],
        [CTA_M, CTA_N, LOCAL_M // 2, LOCAL_N // 2],
        [3, 2, 1, 0],
        [[i, 0, 0, 0] for i in mb] + [[0, i, 0, 0] for i in nb],
    )
    return (shared_a, shared_b, shared_c, wmma)
