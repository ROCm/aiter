# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

exp2 = tl.math.exp2


@gluon.jit
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64_gluon(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: gl.constexpr,
    HV: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BT: gl.constexpr,
    BV: gl.constexpr,
    USE_G: gl.constexpr,
    USE_GK: gl.constexpr,
    USE_INITIAL_STATE: gl.constexpr,
    STORE_FINAL_STATE: gl.constexpr,
    SAVE_NEW_VALUE: gl.constexpr,
    USE_EXP2: gl.constexpr,
    TRANSPOSE_STATE: gl.constexpr,
    IS_VARLEN: gl.constexpr,
    LOAD_CACHE: gl.constexpr,
    STORE_CACHE: gl.constexpr,
):
    dtype = w.type.element_ty
    blocked_w: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((0, 1), (0, 2), (0, 4), (2, 0), (4, 0)),
        lane_bases=((0, 8), (0, 16), (0, 32), (8, 0), (16, 0), (32, 0)),
        warp_bases=((1, 0),),
        block_bases=[],
        shape=[64, 64],
    )
    shared_w: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs=[[512, 8]],
        offset_bases=[
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [8, 0],
            [16, 0],
            [32, 0],
            [1, 0],
            [2, 0],
            [4, 0],
        ],
        cga_layout=[],
        shape=[64, 64],
    )
    # double-buffered for 2-stage pipeline
    bufs_w1 = gl.allocate_shared_memory(dtype, shape=[2, BT, 64], layout=shared_w)
    bufs_w2 = gl.allocate_shared_memory(dtype, shape=[2, BT, 64], layout=shared_w)

    # v layout
    blocked_v: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((0, 1), (0, 2), (0, 4), (2, 0)),
        lane_bases=((0, 8), (0, 16), (4, 0), (8, 0), (16, 0), (32, 0)),
        warp_bases=((1, 0),),
        block_bases=[],
        shape=[64, 32],
    )
    shared_v: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs=[[512, 16]],
        offset_bases=[
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [4, 0],
            [8, 0],
            [16, 0],
            [32, 0],
            [1, 0],
            [2, 0],
        ],
        cga_layout=[],
        shape=[64, 32],
    )
    bufs_v = gl.allocate_shared_memory(dtype, shape=[2, BT, BV], layout=shared_v)

    # k layout
    blocked_k: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((1, 0), (2, 0), (4, 0), (0, 2), (0, 32)),
        lane_bases=((8, 0), (16, 0), (32, 0), (0, 8), (0, 4), (0, 16)),
        warp_bases=((0, 1),),
        block_bases=[],
        shape=[64, BT],
    )
    shared_k: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs=[[512, 16]],
        offset_bases=[
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [16, 0],
            [32, 0],
            [0, 8],
            [0, 4],
            [0, 16],
            [0, 1],
            [0, 2],
            [0, 32],
        ],
        cga_layout=[],
        shape=[64, BT],
    )
    bufs_k1 = gl.allocate_shared_memory(dtype, shape=[2, 64, BT], layout=shared_k)
    bufs_k2 = gl.allocate_shared_memory(dtype, shape=[2, 64, BT], layout=shared_k)

    linear_k: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((0, 1), (0, 2), (16, 0), (32, 0), (0, 16), (0, 32)),
        lane_bases=((1, 0), (2, 0), (4, 0), (8, 0), (0, 4), (0, 8)),
        warp_bases=((0, 0),),
        block_bases=[],
        shape=[64, BT],
    )

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 16],
        transposed=False,
        warps_per_cta=[1, 2],
    )
    mfma_layout_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=4
    )
    mfma_layout_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=4
    )

    # store layout
    blocked_st: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((0, 1), (0, 2), (0, 4), (32, 0)),
        lane_bases=((0, 8), (1, 0), (2, 0), (4, 0), (8, 0), (16, 0)),
        warp_bases=((0, 16),),
        block_bases=[],
        shape=[64, BV],
    )
    # gk layout
    blocked_gk: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[],
        lane_bases=[[1], [2], [4], [8], [16], [32]],
        warp_bases=[[0]],  # broadcast across 2 warps
        block_bases=[],
        shape=[64],
    )
    shared_gk: gl.constexpr = gl.SharedLinearLayout(
        offset_bases=[[1], [2], [4], [8], [16], [32]],
        block_bases=[],
        alignment=8,
    )
    bufs_gk1 = gl.allocate_shared_memory(gl.float32, shape=[2, 64], layout=shared_gk)
    bufs_gk2 = gl.allocate_shared_memory(gl.float32, shape=[2, 64], layout=shared_gk)

    i_v, i_nh = gl.program_id(1), gl.program_id(0)
    i_n, i_h = i_nh // HV, i_nh % HV
    if IS_VARLEN:
        bos, eos = (
            gl.load(cu_seqlens + i_n).to(gl.int32),
            gl.load(cu_seqlens + i_n + 1).to(gl.int32),
        )
        T = eos - bos
        NT = gl.cdiv(T, BT)
        boh = gl.load(chunk_offsets + i_n).to(gl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = gl.cdiv(T, BT)
        boh = i_n * NT

    if TRANSPOSE_STATE:
        b_h1 = gl.zeros([BV, 64], dtype=gl.float32)
        if K > 64:
            b_h2 = gl.zeros([BV, 64], dtype=gl.float32)
    else:
        b_h1 = gl.zeros([64, BV], dtype=gl.float32, layout=mfma_layout)
        if K > 64:
            b_h2 = gl.zeros([64, BV], dtype=gl.float32, layout=mfma_layout)

    base_h = (boh * HV + i_h) * K * V
    base_v = (bos * HV + i_h) * V
    base_k = (bos * H + i_h // (HV // H)) * K
    base_w = (bos * HV + i_h) * K
    if SAVE_NEW_VALUE:
        base_v_new = (bos * HV + i_h) * V

    # Active config only (chunk_kda fwd path):
    gl.static_assert(not TRANSPOSE_STATE, "pipelined path: TRANSPOSE_STATE=False only")
    gl.static_assert(USE_GK, "pipelined path: USE_GK=True only")
    gl.static_assert(not USE_G, "pipelined path: USE_G=False only")
    gl.static_assert(
        not USE_INITIAL_STATE, "pipelined path: USE_INITIAL_STATE=False only"
    )
    gl.static_assert(
        not STORE_FINAL_STATE, "pipelined path: STORE_FINAL_STATE=False only"
    )
    gl.static_assert(SAVE_NEW_VALUE, "pipelined path: SAVE_NEW_VALUE=True only")
    gl.static_assert(USE_EXP2, "pipelined path: USE_EXP2=True only")
    gl.static_assert(not IS_VARLEN, "pipelined path: IS_VARLEN=False only")
    gl.static_assert(K == 128, "pipelined path: K=128 only (b_h1, b_h2)")

    # offset helpers (constexpr-derivable layouts; arange recomputed per use as needed)
    range_w_bt = gl.arange(0, BT, layout=gl.SliceLayout(1, blocked_w))
    range_w_64 = gl.arange(0, 64, layout=gl.SliceLayout(0, blocked_w))
    base_offs_w = range_w_bt[:, None] * HV * K + range_w_64[None, :]

    range_v_bt = gl.arange(0, BT, layout=gl.SliceLayout(1, blocked_v))
    range_v_bv = gl.arange(0, BV, layout=gl.SliceLayout(0, blocked_v))
    base_offs_v = range_v_bt[:, None] * HV * V + i_v * BV + range_v_bv[None, :]

    range_k_64 = gl.arange(0, 64, layout=gl.SliceLayout(1, blocked_k))
    range_k_bt = gl.arange(0, BT, layout=gl.SliceLayout(0, blocked_k))
    # + i_t * BT * H*K
    base_offs_k = range_k_64[:, None] + range_k_bt[None, :] * H * K

    o_gk_async = gl.arange(0, 64, layout=blocked_gk)  # for async gk last_idx load → LDS

    # store-h offsets (mfma_layout — match b_h's natural accumulator layout for vectorized stores)
    range_h_64 = gl.arange(0, 64, layout=gl.SliceLayout(1, blocked_st))
    range_h_bv = gl.arange(0, BV, layout=gl.SliceLayout(0, blocked_st))
    base_offs_h1 = range_h_64[:, None] * V + i_v * BV + range_h_bv[None, :]
    base_offs_h2 = base_offs_h1 + 64 * V

    # store v_new offsets (mfma_layout — accumulator layout for vectorized stores; second LDS read in this layout)
    range_vst_bt = gl.arange(0, BT, layout=gl.SliceLayout(1, blocked_st))
    range_vst_bv = gl.arange(0, BV, layout=gl.SliceLayout(0, blocked_st))
    base_offs_v_st = range_vst_bt[:, None] * HV * V + i_v * BV + range_vst_bv[None, :]

    offs_w1_0 = base_offs_w  # i_t = 0
    offs_w2_0 = offs_w1_0 + 64
    offs_v_0 = base_offs_v
    offs_k1_0 = base_offs_k
    offs_k2_0 = offs_k1_0 + 64

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        bufs_w1.index(0), w, base_w + offs_w1_0, cache_modifier=LOAD_CACHE
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        bufs_w2.index(0), w, base_w + offs_w2_0, cache_modifier=LOAD_CACHE
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        bufs_v.index(0), v, base_v + offs_v_0, cache_modifier=LOAD_CACHE
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        bufs_k1.index(0), k, base_k + offs_k1_0, cache_modifier=LOAD_CACHE
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        bufs_k2.index(0), k, base_k + offs_k2_0, cache_modifier=LOAD_CACHE
    )
    gl.amd.cdna4.async_copy.commit_group()

    # gk[0] async → LDS (2 more commits → 7 outstanding after prologue)
    last_idx_0 = min(BT, T) - 1
    offs_gk1_0 = (bos + last_idx_0) * HV * K + i_h * K + o_gk_async
    offs_gk2_0 = offs_gk1_0 + 64
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        bufs_gk1.index(0), gk, offs_gk1_0, cache_modifier=LOAD_CACHE
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        bufs_gk2.index(0), gk, offs_gk2_0, cache_modifier=LOAD_CACHE
    )
    gl.amd.cdna4.async_copy.commit_group()

    gl.amd.cdna4.async_copy.wait_group(0)

    buf_idx = 0

    for i_t in range(NT - 1):
        async_idx = (buf_idx + 1) % 2
        i_t_next = i_t + 1

        # === Stage 1: prefetch iter i_t+1 into async_idx, store h[i_t], load gk[i_t+1] ===
        offs_w1_n = i_t_next * BT * HV * K + base_offs_w
        offs_w2_n = offs_w1_n + 64
        offs_v_n = i_t_next * BT * HV * V + base_offs_v
        offs_k1_n = i_t_next * BT * H * K + base_offs_k
        offs_k2_n = offs_k1_n + 64

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            bufs_w1.index(async_idx), w, base_w + offs_w1_n, cache_modifier=LOAD_CACHE
        )
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            bufs_w2.index(async_idx), w, base_w + offs_w2_n, cache_modifier=LOAD_CACHE
        )
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            bufs_k1.index(async_idx), k, base_k + offs_k1_n, cache_modifier=LOAD_CACHE
        )
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            bufs_k2.index(async_idx), k, base_k + offs_k2_n, cache_modifier=LOAD_CACHE
        )
        gl.amd.cdna4.async_copy.commit_group()

        gl.amd.cdna4.async_copy.wait_group(4)

        b_h1_bf16 = b_h1.to(dtype)
        b_h2_bf16 = b_h2.to(dtype)

        # no-op
        b_h1_cvt = gl.convert_layout(b_h1_bf16, mfma_layout_b)
        b_h2_cvt = gl.convert_layout(b_h2_bf16, mfma_layout_b)

        # move here to help llvm better schedule apart: ds_read, mma
        b_w1 = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bufs_w1.index(buf_idx), mfma_layout_a
        )
        b_w2 = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bufs_w2.index(buf_idx), mfma_layout_a
        )

        # move store here to separate from the above 9 buffer_load
        offs_h1_st = i_t * HV * K * V + base_offs_h1
        offs_h2_st = i_t * HV * K * V + base_offs_h2
        b_h1_st = gl.convert_layout(b_h1_bf16, blocked_st)
        b_h2_st = gl.convert_layout(b_h2_bf16, blocked_st)
        gl.amd.cdna4.buffer_store(
            b_h1_st, ptr=h, offsets=base_h + offs_h1_st, cache=STORE_CACHE
        )
        gl.amd.cdna4.buffer_store(
            b_h2_st, ptr=h, offsets=base_h + offs_h2_st, cache=STORE_CACHE
        )

        zeros = gl.zeros([BT, BV], dtype=gl.float32, layout=mfma_layout)
        b_v = gl.amd.cdna4.mfma(b_w1, b_h1_cvt, zeros)

        b_v = gl.amd.cdna4.mfma(b_w2, b_h2_cvt, b_v)

        v_load = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bufs_v.index(buf_idx), mfma_layout_b
        )
        b_v = gl.convert_layout(b_v, mfma_layout_b)  # no-op
        b_v = v_load.to(gl.float32) - b_v
        b_v = b_v.to(dtype)
        # b_v (in mfma_layout_b) feeds the k·b_v MFMA below

        # store v_new[i_t] via SECOND LDS read in mfma_layout — vectorized buffer_store
        b_v_st = gl.convert_layout(b_v, blocked_st)
        offs_v_st_i = i_t * BT * HV * V + base_offs_v_st
        gl.amd.cdna4.buffer_store(
            b_v_st, ptr=v_new, offsets=base_v_new + offs_v_st_i, cache=STORE_CACHE
        )

        # gk[i_t+1] async → LDS (2 more commits → 14 outstanding entering stage 2)
        last_idx_n = min((i_t_next + 1) * BT, T) - 1
        offs_gk1_n = (bos + last_idx_n) * HV * K + i_h * K + o_gk_async
        offs_gk2_n = offs_gk1_n + 64
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            bufs_gk1.index(async_idx), gk, offs_gk1_n, cache_modifier=LOAD_CACHE
        )
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            bufs_gk2.index(async_idx), gk, offs_gk2_n, cache_modifier=LOAD_CACHE
        )
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            bufs_v.index(async_idx), v, base_v + offs_v_n, cache_modifier=LOAD_CACHE
        )
        gl.amd.cdna4.async_copy.commit_group()
        # drain iter i_t's W/V/K/gk
        gl.amd.cdna4.async_copy.wait_group(3)

        # decay h with gk[i_t] — load directly in mfma slice layout (no convert)
        b_gk1 = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bufs_gk1.index(buf_idx), gl.SliceLayout(1, mfma_layout)
        )
        b_gk2 = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bufs_gk2.index(buf_idx), gl.SliceLayout(1, mfma_layout)
        )
        b_h1 *= exp2(b_gk1)[:, None]
        b_h2 *= exp2(b_gk2)[:, None]

        b_k1 = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bufs_k1.index(buf_idx), linear_k
        )
        b_k1 = gl.convert_layout(b_k1, mfma_layout_a)  # no-op
        b_h1 = gl.amd.cdna4.mfma(b_k1, b_v, b_h1)

        b_k2 = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bufs_k2.index(buf_idx), linear_k
        )
        b_k2 = gl.convert_layout(b_k2, mfma_layout_a)  # no-op
        b_h2 = gl.amd.cdna4.mfma(b_k2, b_v, b_h2)

        # roll
        buf_idx = async_idx

    i_t_last = NT - 1

    # store h[NT-1] (in mfma_layout — vectorized)
    b_h1_bf16 = b_h1.to(dtype)
    b_h2_bf16 = b_h2.to(dtype)
    offs_h1_st = i_t_last * HV * K * V + base_offs_h1
    offs_h2_st = i_t_last * HV * K * V + base_offs_h2
    b_h1_st = gl.convert_layout(b_h1_bf16, blocked_st)
    b_h2_st = gl.convert_layout(b_h2_bf16, blocked_st)
    gl.amd.cdna4.buffer_store(
        b_h1_st, ptr=h, offsets=base_h + offs_h1_st, cache=STORE_CACHE
    )
    gl.amd.cdna4.buffer_store(
        b_h2_st, ptr=h, offsets=base_h + offs_h2_st, cache=STORE_CACHE
    )
    b_h1_cvt = gl.convert_layout(b_h1_bf16, mfma_layout_b)
    b_h2_cvt = gl.convert_layout(b_h2_bf16, mfma_layout_b)

    # Stage 2 for iter NT-1 — outstanding=5
    gl.amd.cdna4.async_copy.wait_group(0)
    b_w1 = gl.amd.cdna4.async_copy.load_shared_relaxed(
        bufs_w1.index(buf_idx), mfma_layout_a
    )
    zeros = gl.zeros([BT, BV], dtype=gl.float32, layout=mfma_layout)
    b_v = gl.amd.cdna4.mfma(b_w1, b_h1_cvt, zeros)

    b_w2 = gl.amd.cdna4.async_copy.load_shared_relaxed(
        bufs_w2.index(buf_idx), mfma_layout_a
    )
    b_v = gl.amd.cdna4.mfma(b_w2, b_h2_cvt, b_v)

    v_load = gl.amd.cdna4.async_copy.load_shared_relaxed(
        bufs_v.index(buf_idx), mfma_layout_b
    )
    b_v = gl.convert_layout(b_v, mfma_layout_b)
    b_v = v_load.to(gl.float32) - b_v
    b_v = b_v.to(dtype)

    # vectorized v_new store via second LDS read in mfma_layout
    b_v_st = gl.convert_layout(b_v, blocked_st)
    offs_v_st_last = i_t_last * BT * HV * V + base_offs_v_st
    gl.amd.cdna4.buffer_store(
        b_v_st, ptr=v_new, offsets=base_v_new + offs_v_st_last, cache=STORE_CACHE
    )

    b_gk1 = gl.amd.cdna4.async_copy.load_shared_relaxed(
        bufs_gk1.index(buf_idx), gl.SliceLayout(1, mfma_layout)
    )
    b_gk2 = gl.amd.cdna4.async_copy.load_shared_relaxed(
        bufs_gk2.index(buf_idx), gl.SliceLayout(1, mfma_layout)
    )
    b_h1 *= exp2(b_gk1)[:, None]
    b_h2 *= exp2(b_gk2)[:, None]

    b_k1 = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_k1.index(buf_idx), linear_k)
    b_k1 = gl.convert_layout(b_k1, mfma_layout_a)
    b_h1 = gl.amd.cdna4.mfma(b_k1, b_v, b_h1)

    b_k2 = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_k2.index(buf_idx), linear_k)
    b_k2 = gl.convert_layout(b_k2, mfma_layout_a)
    b_h2 = gl.amd.cdna4.mfma(b_k2, b_v, b_h2)
