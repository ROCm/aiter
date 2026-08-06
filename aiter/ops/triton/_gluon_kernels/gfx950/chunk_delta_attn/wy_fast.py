# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

exp2 = tl.math.exp2


@gluon.jit
def recompute_w_u_fwd_kda_kernel_persistent_gluon(
    k,
    kg,
    v,
    beta,
    w,
    u,
    A,
    gk,
    T,
    H: gl.constexpr,
    HV: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BT: gl.constexpr,
    BLOCK_T: gl.constexpr,
    LOAD_CACHE: gl.constexpr,
    STORE_CACHE: gl.constexpr,
):
    # Specialized path: chunk_kda fwd at K=V=128, BT=64, STORE_QG=False, STORE_KG=True, IS_VARLEN=False.
    gl.static_assert(K == 128, "gluon path: K=128 only")
    gl.static_assert(V == 128, "gluon path: V=128 only")
    gl.static_assert(BT == 64, "gluon path: BT=64 only")

    NT_PER_BLOCK: gl.constexpr = BLOCK_T // BT

    ## layout begin
    dtype = k.type.element_ty
    blocked_b: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((4, 0), (8, 0), (16, 0)),  # (0, 0)
        lane_bases=((0, 1), (0, 2), (0, 4), (0, 8), (0, 16), (0, 32)),
        warp_bases=((1, 0), (2, 0)),
        block_bases=[],
        shape=[32, 64],
    )
    shared_b: gl.constexpr = gl.SharedLinearLayout(
        offset_bases=[
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [16, 0],
        ],
        block_bases=[],
        alignment=8,
    )
    shared_b_row: gl.constexpr = gl.SharedLinearLayout(
        offset_bases=[[1], [2], [4], [8], [16], [32]],
        block_bases=[],
        alignment=8,
    )
    blocked_gn: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((0, 1), (0, 2)),
        lane_bases=((0, 4), (0, 8), (0, 16), (0, 32), (0, 64), (1, 0)),
        warp_bases=((2, 0), (4, 0)),
        block_bases=[],
        shape=[8, 128],
    )
    shared_gn: gl.constexpr = gl.SharedLinearLayout(
        offset_bases=[
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
            [1, 0],
            [2, 0],
            [4, 0],
        ],
        block_bases=[],
        alignment=8,
    )
    shared_gn_row: gl.constexpr = gl.SharedLinearLayout(
        offset_bases=[[1], [2], [4], [8], [16], [32], [64]],
        block_bases=[],
        alignment=8,
    )
    buf_b = gl.allocate_shared_memory(
        gl.float32, shape=[NT_PER_BLOCK, BT], layout=shared_b
    )
    buf_gn = gl.allocate_shared_memory(gl.float32, shape=[8, K], layout=shared_gn)

    blocked_rhs: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((0, 1), (0, 2), (0, 4), (8, 0), (4, 0)),
        lane_bases=((0, 8), (0, 16), (0, 32), (0, 64), (16, 0), (32, 0)),
        warp_bases=((1, 0), (2, 0)),
        block_bases=[],
        shape=[64, 128],
    )
    shared_rhs: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs=[[512, 16]],
        offset_bases=[
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
            [16, 0],
            [32, 0],
            [1, 0],
            [2, 0],
            [8, 0],
            [4, 0],
        ],
        cga_layout=[],
        shape=[64, 128],
    )

    bufs_v = gl.allocate_shared_memory(dtype, shape=[2, BT, V], layout=shared_rhs)
    bufs_k = gl.allocate_shared_memory(dtype, shape=[2, BT, K], layout=shared_rhs)

    # gk layout (fp32)
    blocked_rhs_fp32: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((0, 1), (0, 2), (8, 0), (16, 0), (32, 0)),
        lane_bases=((0, 4), (0, 8), (0, 16), (0, 32), (0, 64), (1, 0)),
        warp_bases=((2, 0), (4, 0)),
        block_bases=[],
        shape=[64, 128],
    )
    shared_rhs_fp32: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs=[[1024, 16]],
        offset_bases=[
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [16, 0],
            [32, 0],
        ],
        cga_layout=[],
        shape=[64, 128],
    )
    bufs_gk = gl.allocate_shared_memory(
        gl.float32, shape=[2, BT, K], layout=shared_rhs_fp32
    )

    # mfma layout
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[2, 2],
    )
    mfma_layout_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=8
    )
    mfma_layout_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=8
    )

    # store layout of u, w, kg
    blocked_st: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((0, 1), (0, 2), (0, 4), (4, 0), (8, 0)),
        lane_bases=((0, 8), (0, 16), (0, 32), (0, 64), (1, 0), (2, 0)),
        warp_bases=((16, 0), (32, 0)),
        block_bases=[],
        shape=[64, 128],
    )

    i_t_outer = gl.program_id(0)
    i_bh = gl.program_id(1)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    base_k = (i_b * T * H + i_h) * K
    base_v = (i_b * T * HV + i_hv) * V
    base_u = (i_b * T * HV + i_hv) * V
    base_w = (i_b * T * HV + i_hv) * K
    base_gk = (i_b * T * HV + i_hv) * K
    base_beta = i_b * T * HV + i_hv
    base_A = (i_b * T * HV + i_hv) * BT
    base_kg = (i_b * T * HV + i_hv) * K

    # A [BT, BT] sync register load → mfma_layout_a (rolling _curr/_next), stride (HV*BT, 1)
    range_a0 = gl.arange(0, BT, layout=gl.SliceLayout(1, mfma_layout_a))
    range_a1 = gl.arange(0, BT, layout=gl.SliceLayout(0, mfma_layout_a))
    base_offs_a = range_a0[:, None] * (HV * BT) + range_a1[None, :]

    # v [BT, V] async load → bufs_v (shared_rhs); offsets in blocked_rhs, stride (HV*V, 1)
    range_v_bt = gl.arange(0, BT, layout=gl.SliceLayout(1, blocked_rhs))
    range_v_dim = gl.arange(0, V, layout=gl.SliceLayout(0, blocked_rhs))
    base_offs_v = range_v_bt[:, None] * (HV * V) + range_v_dim[None, :]

    # k [BT, K] async load → bufs_k (shared_rhs); stride (H*K, 1) — note H, not HV
    range_k_bt = gl.arange(0, BT, layout=gl.SliceLayout(1, blocked_rhs))
    range_k_dim = gl.arange(0, K, layout=gl.SliceLayout(0, blocked_rhs))
    base_offs_k = range_k_bt[:, None] * (H * K) + range_k_dim[None, :]

    # gk [BT, K] async load → bufs_gk (shared_rhs_fp32); stride (HV*K, 1)
    range_gk_bt = gl.arange(0, BT, layout=gl.SliceLayout(1, blocked_rhs_fp32))
    range_gk_dim = gl.arange(0, K, layout=gl.SliceLayout(0, blocked_rhs_fp32))
    base_offs_gk = range_gk_bt[:, None] * (HV * K) + range_gk_dim[None, :]

    # buf_b [NT_PER_BLOCK, BT] async load → buf_b (shared_b); offsets in blocked_b, stride (HV, 1)
    # row r holds beta for iter (i_t_outer*NT_PER_BLOCK + r); col c indexes the BT positions for that iter
    # → element offset in beta = ((i_t_outer*NT_PER_BLOCK + r) * BT + c) * HV  (HV is the per-T stride)
    # 2D offsets for buf_b bulk load — shape [NT_PER_BLOCK, BT] in blocked_b.
    range_buf_b_iter = gl.arange(0, NT_PER_BLOCK, layout=gl.SliceLayout(1, blocked_b))
    range_buf_b_bt = gl.arange(0, BT, layout=gl.SliceLayout(0, blocked_b))
    base_offs_buf_b = (
        range_buf_b_iter[:, None] * (BT * HV) + range_buf_b_bt[None, :] * HV
    )

    # buf_gn [8, K] async load → buf_gn (shared_gn); one batch = 8 iters' last-row gn.
    # row r holds gk[last_idx_of_iter, :] for iter (i_t_outer*NT_PER_BLOCK + batch*8 + r)
    # last_idx of iter j = j*BT + BT - 1; element offset in gk = last_idx * HV*K + col
    range_buf_gn_iter = gl.arange(0, 8, layout=gl.SliceLayout(1, blocked_gn))
    range_buf_gn_k = gl.arange(0, K, layout=gl.SliceLayout(0, blocked_gn))
    base_offs_buf_gn = (
        range_buf_gn_iter[:, None] * (BT * HV * K)
        + (BT - 1) * HV * K
        + range_buf_gn_k[None, :]
    )

    # Store offsets in blocked_st, stride (HV*V, 1) for u, (HV*K, 1) for w/kg
    range_st_bt = gl.arange(0, BT, layout=gl.SliceLayout(1, blocked_st))
    range_st_dim = gl.arange(0, V, layout=gl.SliceLayout(0, blocked_st))
    base_offs_st_v = range_st_bt[:, None] * (HV * V) + range_st_dim[None, :]
    base_offs_st_k = range_st_bt[:, None] * (HV * K) + range_st_dim[None, :]

    i_t0 = i_t_outer * NT_PER_BLOCK
    offs_v_0 = i_t0 * BT * HV * V + base_offs_v
    offs_k_0 = i_t0 * BT * H * K + base_offs_k
    offs_gk_0 = i_t0 * BT * HV * K + base_offs_gk
    offs_A_0 = i_t0 * BT * HV * BT + base_offs_a
    offs_buf_b = i_t0 * BT * HV + base_offs_buf_b
    offs_buf_gn = i_t0 * BT * HV * K + base_offs_buf_gn

    # Async LDS DB prefetch of v/k/gk for iter 0
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        bufs_v.index(0), v, base_v + offs_v_0, cache_modifier=LOAD_CACHE
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        bufs_k.index(0), k, base_k + offs_k_0, cache_modifier=LOAD_CACHE
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        bufs_gk.index(0), gk, base_gk + offs_gk_0, cache_modifier=LOAD_CACHE
    )
    gl.amd.cdna4.async_copy.commit_group()

    # Bulk async load of beta for entire program (one shot, 32 betas).
    # Bulk async load of gn for first batch only (iters 0..7); subsequent batches
    # are refilled in the main loop at iters 7, 15, 23.
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        buf_b, beta, base_beta + offs_buf_b, cache_modifier=LOAD_CACHE
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        buf_gn, gk, base_gk + offs_buf_gn, cache_modifier=LOAD_CACHE
    )
    gl.amd.cdna4.async_copy.commit_group()

    # Rank-1 view onto buf_b / buf_gn so .index(runtime i) returns a 1D row.
    # Same physical bytes as buf_b / buf_gn — pure type cast.
    bufs_b_view = buf_b._reinterpret(
        gl.float32, shape=[NT_PER_BLOCK, BT], layout=shared_b_row
    )
    bufs_gn_view = buf_gn._reinterpret(gl.float32, shape=[8, K], layout=shared_gn_row)

    # Sync register load of A for iter 0 (rolled across iters via a_curr ↔ a_next).
    a_curr = gl.amd.cdna4.buffer_load(A, base_A + offs_A_0, cache=LOAD_CACHE)

    buf_idx = 0

    for i_t_local in range(NT_PER_BLOCK - 1):
        gl.amd.cdna4.async_copy.wait_group(0)

        async_idx = (buf_idx + 1) % 2
        i_t = i_t_outer * NT_PER_BLOCK + i_t_local
        i_t_next = i_t + 1

        # === Stage 1: prefetch iter i_t_next into bufs[async_idx] (v, k, gk) ===
        offs_v_n = i_t_next * BT * HV * V + base_offs_v
        offs_k_n = i_t_next * BT * H * K + base_offs_k
        offs_gk_n = i_t_next * BT * HV * K + base_offs_gk

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            bufs_v.index(async_idx), v, base_v + offs_v_n, cache_modifier=LOAD_CACHE
        )
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            bufs_k.index(async_idx), k, base_k + offs_k_n, cache_modifier=LOAD_CACHE
        )
        gl.amd.cdna4.async_copy.commit_group()

        # === Stage 2: consume iter i_t from bufs[buf_idx]; beta/gn pulled from bulk LDS views ===
        b_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bufs_b_view.index(i_t_local), gl.SliceLayout(1, mfma_layout_b)
        )

        # u-leg: u = A @ (v * beta)
        b_v = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bufs_v.index(buf_idx), mfma_layout_b
        )
        b_vb = (b_v * b_b[:, None]).to(dtype)
        b_u = gl.zeros([BT, V], dtype=gl.float32, layout=mfma_layout)
        b_u = gl.amd.cdna4.mfma(a_curr, b_vb, b_u)
        b_u_st = gl.convert_layout(b_u.to(dtype), blocked_st)
        gl.amd.cdna4.buffer_store(
            b_u_st,
            ptr=u,
            offsets=base_u + i_t * BT * HV * V + base_offs_st_v,
            cache=STORE_CACHE,
        )

        # w-leg: w = A @ (k * beta * exp2(gk))
        b_k = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bufs_k.index(buf_idx), mfma_layout_b
        )
        b_gk = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bufs_gk.index(buf_idx), mfma_layout_b
        )
        b_kb = (b_k.to(gl.float32) * b_b[:, None] * exp2(b_gk)).to(dtype)
        b_w = gl.zeros([BT, K], dtype=gl.float32, layout=mfma_layout)
        b_w = gl.amd.cdna4.mfma(a_curr, b_kb, b_w)

        # gk async-copy positioned after w store (current location).
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            bufs_gk.index(async_idx), gk, base_gk + offs_gk_n, cache_modifier=LOAD_CACHE
        )
        gl.amd.cdna4.async_copy.commit_group()

        b_w_st = gl.convert_layout(b_w.to(dtype), blocked_st)

        # Sync register load for iter i_t_next's A (rolled). Issued late so VMEM latency
        # overlaps with kg compute + stores below.
        offs_A_n = i_t_next * BT * HV * BT + base_offs_a
        a_next = gl.amd.cdna4.buffer_load(A, base_A + offs_A_n, cache=LOAD_CACHE)

        # kg-leg: kg = k * exp2(gn - gk). bufs_gn_view holds 8 rows per batch;
        # within-batch row index = i_t_local % 8.
        b_gn = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bufs_gn_view.index(i_t_local % 8), gl.SliceLayout(0, mfma_layout_b)
        )
        b_kg = (b_k.to(gl.float32) * exp2(b_gn[None, :] - b_gk)).to(dtype)
        b_kg_st = gl.convert_layout(b_kg, blocked_st)
        gl.amd.cdna4.buffer_store(
            b_kg_st,
            ptr=kg,
            offsets=base_kg + i_t * BT * HV * K + base_offs_st_k,
            cache=STORE_CACHE,
        )
        gl.amd.cdna4.buffer_store(
            b_w_st,
            ptr=w,
            offsets=base_w + i_t * BT * HV * K + base_offs_st_k,
            cache=STORE_CACHE,
        )

        # buf_gn batch refill: at end of last iter of each batch (iter 7, 15, 23),
        # issue async load of next 8-row batch into the same buf_gn LDS region.
        # The next iter's wait_group(0) drains it before its kg-leg consumes.
        if (i_t_local + 1) % 8 == 0 and (i_t_local + 1) < NT_PER_BLOCK:
            offs_buf_gn_n = (i_t0 + (i_t_local + 1)) * BT * HV * K + base_offs_buf_gn
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                buf_gn, gk, base_gk + offs_buf_gn_n, cache_modifier=LOAD_CACHE
            )
            gl.amd.cdna4.async_copy.commit_group()

        # Roll A register + buffer index
        a_curr = a_next
        buf_idx = async_idx

    # a_curr was rolled to iter NT_PER_BLOCK-1 by the main loop.
    # Beta/gn for last iter are extracted from LDS bulk buffers via .index(constexpr).
    i_t_last_local: gl.constexpr = NT_PER_BLOCK - 1
    i_t_last = i_t_outer * NT_PER_BLOCK + i_t_last_local

    gl.amd.cdna4.async_copy.wait_group(0)

    b_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
        bufs_b_view.index(i_t_last_local), gl.SliceLayout(1, mfma_layout_b)
    )

    b_v = gl.amd.cdna4.async_copy.load_shared_relaxed(
        bufs_v.index(buf_idx), mfma_layout_b
    )
    b_vb = (b_v * b_b[:, None]).to(dtype)
    b_u = gl.zeros([BT, V], dtype=gl.float32, layout=mfma_layout)
    b_u = gl.amd.cdna4.mfma(a_curr, b_vb, b_u)
    b_u_st = gl.convert_layout(b_u.to(dtype), blocked_st)
    gl.amd.cdna4.buffer_store(
        b_u_st,
        ptr=u,
        offsets=base_u + i_t_last * BT * HV * V + base_offs_st_v,
        cache=STORE_CACHE,
    )

    b_k = gl.amd.cdna4.async_copy.load_shared_relaxed(
        bufs_k.index(buf_idx), mfma_layout_b
    )
    b_gk = gl.amd.cdna4.async_copy.load_shared_relaxed(
        bufs_gk.index(buf_idx), mfma_layout_b
    )
    b_kb = (b_k.to(gl.float32) * b_b[:, None] * exp2(b_gk)).to(dtype)
    b_w = gl.zeros([BT, K], dtype=gl.float32, layout=mfma_layout)
    b_w = gl.amd.cdna4.mfma(a_curr, b_kb, b_w)
    b_w_st = gl.convert_layout(b_w.to(dtype), blocked_st)
    gl.amd.cdna4.buffer_store(
        b_w_st,
        ptr=w,
        offsets=base_w + i_t_last * BT * HV * K + base_offs_st_k,
        cache=STORE_CACHE,
    )

    # Last iter is in the final 8-row batch; within-batch index = 7.
    i_gn_last_local: gl.constexpr = (NT_PER_BLOCK - 1) % 8
    b_gn = gl.amd.cdna4.async_copy.load_shared_relaxed(
        bufs_gn_view.index(i_gn_last_local), gl.SliceLayout(0, mfma_layout_b)
    )
    b_kg = (b_k.to(gl.float32) * exp2(b_gn[None, :] - b_gk)).to(dtype)
    b_kg_st = gl.convert_layout(b_kg, blocked_st)
    gl.amd.cdna4.buffer_store(
        b_kg_st,
        ptr=kg,
        offsets=base_kg + i_t_last * BT * HV * K + base_offs_st_k,
        cache=STORE_CACHE,
    )


@gluon.jit
def recompute_w_u_fwd_kda_kernel_gluon_small_h(
    k,
    kg,
    v,
    beta,
    w,
    u,
    A,
    gk,
    T,
    H: gl.constexpr,
    HV: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BT: gl.constexpr,
    LOAD_CACHE: gl.constexpr,
    STORE_CACHE: gl.constexpr,
):
    """Non-persistent Gluon kernel for small H (<32).

    Grid = (NT=T//BT, B*HV): one CTA per (chunk, head) pair.
    LDS ~65 KB (vs 159 KB for persistent) → potentially occupancy=2.
    No BLOCK_T persistence; same MFMA/layout constants as persistent kernel.
    """
    gl.static_assert(K == 128, "K=128 only")
    gl.static_assert(V == 128, "V=128 only")
    gl.static_assert(BT == 64, "BT=64 only")

    dtype = k.type.element_ty

    # Layouts (same as persistent kernel)
    blocked_rhs: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((0, 1), (0, 2), (0, 4), (8, 0), (4, 0)),
        lane_bases=((0, 8), (0, 16), (0, 32), (0, 64), (16, 0), (32, 0)),
        warp_bases=((1, 0), (2, 0)),
        block_bases=[],
        shape=[64, 128],
    )
    shared_rhs: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs=[[512, 16]],
        offset_bases=[
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
            [16, 0],
            [32, 0],
            [1, 0],
            [2, 0],
            [8, 0],
            [4, 0],
        ],
        cga_layout=[],
        shape=[64, 128],
    )
    blocked_rhs_fp32: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((0, 1), (0, 2), (8, 0), (16, 0), (32, 0)),
        lane_bases=((0, 4), (0, 8), (0, 16), (0, 32), (0, 64), (1, 0)),
        warp_bases=((2, 0), (4, 0)),
        block_bases=[],
        shape=[64, 128],
    )
    shared_rhs_fp32: gl.constexpr = gl.PaddedSharedLayout(
        interval_padding_pairs=[[1024, 16]],
        offset_bases=[
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [16, 0],
            [32, 0],
        ],
        cga_layout=[],
        shape=[64, 128],
    )
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[2, 2],
    )
    mfma_layout_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=8
    )
    mfma_layout_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=8
    )
    blocked_st: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=((0, 1), (0, 2), (0, 4), (4, 0), (8, 0)),
        lane_bases=((0, 8), (0, 16), (0, 32), (0, 64), (1, 0), (2, 0)),
        warp_bases=((16, 0), (32, 0)),
        block_bases=[],
        shape=[64, 128],
    )

    # Single (non-double-buffered) LDS buffers
    buf_v = gl.allocate_shared_memory(dtype, shape=[BT, V], layout=shared_rhs)
    buf_k = gl.allocate_shared_memory(dtype, shape=[BT, K], layout=shared_rhs)
    buf_gk = gl.allocate_shared_memory(
        gl.float32, shape=[BT, K], layout=shared_rhs_fp32
    )

    i_t = gl.program_id(0)  # chunk index (0..NT-1)
    i_bh = gl.program_id(1)
    i_b = i_bh // HV
    i_hv = i_bh % HV
    i_h = i_hv // (HV // H)

    base_k = (i_b * T * H + i_h) * K
    base_v = (i_b * T * HV + i_hv) * V
    base_u = (i_b * T * HV + i_hv) * V
    base_w = (i_b * T * HV + i_hv) * K
    base_gk = (i_b * T * HV + i_hv) * K
    base_beta = i_b * T * HV + i_hv
    base_A = (i_b * T * HV + i_hv) * BT
    base_kg = (i_b * T * HV + i_hv) * K

    # A [BT, BT] → mfma_layout_a registers
    range_a0 = gl.arange(0, BT, layout=gl.SliceLayout(1, mfma_layout_a))
    range_a1 = gl.arange(0, BT, layout=gl.SliceLayout(0, mfma_layout_a))
    base_offs_a = range_a0[:, None] * (HV * BT) + range_a1[None, :]

    # v [BT, V] / k [BT, K] → LDS via blocked_rhs
    range_v_bt = gl.arange(0, BT, layout=gl.SliceLayout(1, blocked_rhs))
    range_v_dim = gl.arange(0, V, layout=gl.SliceLayout(0, blocked_rhs))
    base_offs_v = range_v_bt[:, None] * (HV * V) + range_v_dim[None, :]

    range_k_bt = gl.arange(0, BT, layout=gl.SliceLayout(1, blocked_rhs))
    range_k_dim = gl.arange(0, K, layout=gl.SliceLayout(0, blocked_rhs))
    base_offs_k = range_k_bt[:, None] * (H * K) + range_k_dim[None, :]

    # gk [BT, K] → LDS via blocked_rhs_fp32
    range_gk_bt = gl.arange(0, BT, layout=gl.SliceLayout(1, blocked_rhs_fp32))
    range_gk_dim = gl.arange(0, K, layout=gl.SliceLayout(0, blocked_rhs_fp32))
    base_offs_gk = range_gk_bt[:, None] * (HV * K) + range_gk_dim[None, :]

    # beta [BT]: strided column load — element j at offset (i_t*BT+j)*HV in beta flat
    # SliceLayout(1, mfma_layout_b) gives M-dim (BT=64) distribution of B-operand.
    # Direct buffer_load into this layout avoids LDS roundtrip for the small beta vector.
    range_b = gl.arange(0, BT, layout=gl.SliceLayout(1, mfma_layout_b))
    offs_b = i_t * BT * HV + range_b * HV

    # gn [K]: last row of gk for this chunk — gk[(i_t+1)*BT-1, hv, :]
    # SliceLayout(0, mfma_layout_b) gives N-dim (K=128) distribution of B-operand.
    range_gn = gl.arange(0, K, layout=gl.SliceLayout(0, mfma_layout_b))
    offs_gn = i_t * BT * HV * K + (BT - 1) * HV * K + range_gn

    # store offsets for u/w/kg [BT, V/K]
    range_st_bt = gl.arange(0, BT, layout=gl.SliceLayout(1, blocked_st))
    range_st_dim = gl.arange(0, V, layout=gl.SliceLayout(0, blocked_st))
    base_offs_st_v = range_st_bt[:, None] * (HV * V) + range_st_dim[None, :]
    base_offs_st_k = range_st_bt[:, None] * (HV * K) + range_st_dim[None, :]

    # Issue async loads: v, k, gk → LDS
    offs_v = i_t * BT * HV * V + base_offs_v
    offs_k = i_t * BT * H * K + base_offs_k
    offs_gk = i_t * BT * HV * K + base_offs_gk

    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        buf_v, v, base_v + offs_v, cache_modifier=LOAD_CACHE
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        buf_k, k, base_k + offs_k, cache_modifier=LOAD_CACHE
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        buf_gk, gk, base_gk + offs_gk, cache_modifier=LOAD_CACHE
    )
    gl.amd.cdna4.async_copy.commit_group()

    # Sync register loads: A, beta, gn (overlap with async above)
    offs_A = i_t * BT * HV * BT + base_offs_a
    a = gl.amd.cdna4.buffer_load(A, base_A + offs_A, cache=LOAD_CACHE)
    b_b = gl.amd.cdna4.buffer_load(beta, base_beta + offs_b, cache=LOAD_CACHE)
    b_gn = gl.amd.cdna4.buffer_load(gk, base_gk + offs_gn, cache=LOAD_CACHE)

    # Wait for async copies
    gl.amd.cdna4.async_copy.wait_group(0)

    # u-leg: u = A @ (v * beta)
    b_v = gl.amd.cdna4.async_copy.load_shared_relaxed(buf_v, mfma_layout_b)
    b_vb = (b_v * b_b[:, None]).to(dtype)
    b_u = gl.zeros([BT, V], dtype=gl.float32, layout=mfma_layout)
    b_u = gl.amd.cdna4.mfma(a, b_vb, b_u)
    b_u_st = gl.convert_layout(b_u.to(dtype), blocked_st)
    gl.amd.cdna4.buffer_store(
        b_u_st,
        ptr=u,
        offsets=base_u + i_t * BT * HV * V + base_offs_st_v,
        cache=STORE_CACHE,
    )

    # w-leg: w = A @ (k * beta * exp2(gk))
    b_k = gl.amd.cdna4.async_copy.load_shared_relaxed(buf_k, mfma_layout_b)
    b_gk = gl.amd.cdna4.async_copy.load_shared_relaxed(buf_gk, mfma_layout_b)
    b_kb = (b_k.to(gl.float32) * b_b[:, None] * exp2(b_gk)).to(dtype)
    b_w = gl.zeros([BT, K], dtype=gl.float32, layout=mfma_layout)
    b_w = gl.amd.cdna4.mfma(a, b_kb, b_w)
    b_w_st = gl.convert_layout(b_w.to(dtype), blocked_st)
    gl.amd.cdna4.buffer_store(
        b_w_st,
        ptr=w,
        offsets=base_w + i_t * BT * HV * K + base_offs_st_k,
        cache=STORE_CACHE,
    )

    # kg-leg: kg = k * exp2(gn - gk)
    b_kg = (b_k.to(gl.float32) * exp2(b_gn[None, :] - b_gk)).to(dtype)
    b_kg_st = gl.convert_layout(b_kg, blocked_st)
    gl.amd.cdna4.buffer_store(
        b_kg_st,
        ptr=kg,
        offsets=base_kg + i_t * BT * HV * K + base_offs_st_k,
        cache=STORE_CACHE,
    )
