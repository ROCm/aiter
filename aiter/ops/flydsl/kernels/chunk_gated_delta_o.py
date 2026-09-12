# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""GDN K6 chunk output projection — standalone FlyDSL kernel.

The counterpart of the Triton ``chunk_fwd_kernel_o_opt_vk``: it consumes the
per-chunk ``h`` snapshot and the ungated ``v_new`` that a separate K5 drained to
HBM, and produces ``o``. One CTA owns one (sequence, head, chunk, V-tile):

  o_inter = q @ h^T                                (GEMM3, contraction over K)
  A       = q @ k^T                                (GEMM4a, contraction over K)
  A'[i,j] = (i >= j) ? A[i,j] * exp(g_i - g_j) : 0  (causal mask + pair gate)
  o_intra = A' @ v_new                             (GEMM4b, contraction over BT)
  o       = scale * (exp(g_i) * o_inter + o_intra)

Gating follows the Triton reference rather than the telescoping form the fused
K5+K6 kernel uses. The fused kernel can gate the pair term with the cheaper
``exp(g_i - g_last)`` because K5 hands it a *pre-gated* ``v_new`` for free; here
``v_new`` arrives ungated from HBM, so buying the same identity would cost a
column gate over v AND put ``exp(g_i - g_last) >= 1`` on the critical path,
which overflows f32 once a chunk's total decay exceeds ~88. The pair form is
evaluated only under the causal mask, where ``g_i <= g_j`` bounds it by 1.

Unlike the fused kernel, whose CTA walks the chunks serially because K5's state
recurrence forces it to, K6 is chunk-parallel: the chunk index is a grid axis,
which is what keeps the kernel occupancy-bound rather than latency-bound at low
batch*head.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import arith as _arith
from flydsl.expr import as_ir_value, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

_LOG2E = math.log2(math.e)


def _make_fast_exp(g_is_log2_scaled: bool):
    """Return the ``exp`` helper, pre-specialised on the gate's log2 scaling.

    When the gate is already log2-scaled the multiply folds away and this is a
    single ``v_exp_f32``. ``as_ir_value`` is required because ``rocdl.exp2``
    takes a raw ``ir.Value`` while the gate arrives as a FlyDSL wrapper.
    """
    if g_is_log2_scaled:

        def _fast_exp(x):
            return rocdl.exp2(T.f32, as_ir_value(x))

    else:

        def _fast_exp(x):
            return rocdl.exp2(T.f32, as_ir_value(x * _LOG2E))

    return _fast_exp


def _to_bf16(val, n=1):
    """f32 -> bf16 by IEEE round-to-nearest-even.

    ``n`` is the element count: 1 for a scalar ``Float32``, N for an f32xN
    ``Vector``. NaN and Inf are carried through by construction, which a
    bias-and-shift conversion would not do.
    """
    bf16_ty = T.vec(n, T.bf16) if n > 1 else T.bf16
    return _arith.truncf(bf16_ty, as_ir_value(val))


def compile_chunk_gated_delta_o(
    *,
    K: int,
    V: int,
    H: int,
    Hg: int,
    SCALE: float,
    BT: int = 64,
    BV: int = 64,
    USE_G: bool = True,
    IS_VARLEN: bool = True,
    G_IS_LOG2_SCALED: bool = False,
    INDEX_STRIDE: int = 1,
    NR_SPLIT: int = 1,
):
    """Build the standalone K6 launcher for one compile-time configuration.

    ``INDEX_STRIDE`` matches the Triton kernel: 1 for the prefill-metadata
    schedule (separate ``sequence_ids`` / ``chunk_ids`` arrays) and 2 for the
    legacy interleaved ``chunk_indices``.
    """
    assert K % 64 == 0 and K <= 256, f"K={K} must be a multiple of 64, <= 256"
    assert K & (K - 1) == 0, f"K={K} must be a power of two"
    assert BT == 64, f"BT={BT}: the GDN pipeline fixes the chunk size at 64"
    assert BV % 16 == 0, f"BV={BV} must be a multiple of 16"
    assert V % BV == 0, f"BV={BV} must divide V={V}"
    assert H % Hg == 0, f"Hg={Hg} must divide H={H}"

    MFMA_M = MFMA_N = MFMA_K = 16
    WARP_SIZE = 64

    M_WAVES = BT // MFMA_M
    # NR_SPLIT widens the CTA along N: the same LDS footprint gets twice the
    # resident waves, which is the only lever left once a tile is large enough
    # that LDS caps the CTAs per CU.
    NUM_WARPS = M_WAVES * NR_SPLIT
    BLOCK_THREADS = NUM_WARPS * WARP_SIZE

    N_REPEAT = BV // MFMA_N  # V-tiles, GEMM3 / GEMM4b
    BT_STEPS = BT // MFMA_K  # key-column tiles (GEMM4a) / contraction steps
    K_TILES = K // MFMA_K
    GRID_V = V // BV

    assert (
        N_REPEAT % NR_SPLIT == 0
    ), f"NR_SPLIT={NR_SPLIT} must divide the {N_REPEAT} V-tiles of BV={BV}"
    assert BT_STEPS % NR_SPLIT == 0, (
        f"NR_SPLIT={NR_SPLIT} must divide the {BT_STEPS} key-column tiles of "
        f"BT={BT}; GEMM4a splits them the same way"
    )
    assert NUM_WARPS <= 16, f"{NUM_WARPS} waves exceeds the 1024-thread block"
    # Per-wave tile counts under the split.
    N_REPEAT_LOCAL = N_REPEAT // NR_SPLIT
    BT_STEPS_LOCAL = BT_STEPS // NR_SPLIT

    _fast_exp = _make_fast_exp(G_IS_LOG2_SCALED)

    # -- LDS budget --
    # Row-major, no XOR swizzle: the bank-conflict work belongs to a later
    # tuning pass, and a plain layout keeps every stage below a straight
    # HBM-tile -> LDS-tile copy with no transpose.
    #
    # q is deliberately absent. It is the A operand of both GEMM3 and GEMM4a,
    # and the M=BT axis is already split across waves, so a wave only ever
    # touches its own 16 rows -- there is nothing to share through LDS. Loading
    # it straight into an A fragment saves BT*K bf16 (16 KiB at K=128), which is
    # what lets a second CTA fit per CU.
    LDS_K_ELEMS = BT * K
    LDS_H_ELEMS = BV * K
    LDS_V_ELEMS = BT * BV
    LDS_A_ELEMS = BT * BT
    _lds_kib = (LDS_K_ELEMS + LDS_H_ELEMS + LDS_V_ELEMS + LDS_A_ELEMS) * 2 / 1024

    @fx.struct
    class SharedStorage:
        lds_k: fx.Array[fx.BFloat16, LDS_K_ELEMS, 16]
        lds_h: fx.Array[fx.BFloat16, LDS_H_ELEMS, 16]
        lds_v: fx.Array[fx.BFloat16, LDS_V_ELEMS, 16]
        lds_A: fx.Array[fx.BFloat16, LDS_A_ELEMS, 16]

    # -- Cooperative tile staging --
    # Same decomposition as the fused kernel's w tile: a thread takes one
    # 8-element (16 B) run along the contiguous axis, THREADS_PER_ROW_64
    # consecutive threads cover a 64-wide row segment.
    LOAD_VEC = 8
    ROW_SEG = 64  # bytes-wide row segment one THREADS_PER_ROW group covers
    THREADS_PER_ROW = ROW_SEG // LOAD_VEC  # 8
    ROWS_PER_BATCH = BLOCK_THREADS // THREADS_PER_ROW  # 32 at 256 threads
    assert (
        ROWS_PER_BATCH <= BT and BT % ROWS_PER_BATCH == 0
    ), f"BT={BT} must tile {ROWS_PER_BATCH} rows per staging pass"
    # The h tile is BV rows tall and K wide, so it reuses the q/k decomposition
    # only while BV is at least one full pass.
    assert ROWS_PER_BATCH <= BV and BV % ROWS_PER_BATCH == 0, (
        f"BV={BV} must be a multiple of {ROWS_PER_BATCH} (the h tile is BV rows "
        f"tall); BV < {ROWS_PER_BATCH} needs its own h decomposition"
    )

    # The v tile is BT rows tall but only BV wide, and BV can be narrower than
    # ROW_SEG. Sizing its thread decomposition off BV instead of ROW_SEG is what
    # keeps the staging tile inside the tensor.
    V_COLS = min(BV, ROW_SEG)
    THREADS_PER_ROW_V = V_COLS // LOAD_VEC
    ROWS_PER_BATCH_V = BLOCK_THREADS // THREADS_PER_ROW_V
    assert ROWS_PER_BATCH_V <= BT and BT % ROWS_PER_BATCH_V == 0, (
        f"BV={BV} gives {ROWS_PER_BATCH_V} rows per v staging pass, which does "
        f"not tile BT={BT}"
    )

    # The XOR swizzle folds the row index into the 4-element group index, so an
    # MFMA fragment's 16 lanes spread over all 32 banks instead of piling onto
    # one. Without it the B-operand read of a [rows, 256B] tile puts every lane
    # on bank 0 -- a 16-way conflict on the hottest read in the kernel.
    # It splits a thread's 8-element run into two non-adjacent 4-element groups,
    # which is why staging writes go out as a lo/hi pair.
    LDS_HALF = LOAD_VEC // 2

    # Loads per thread for each tile, i.e. how many (row-batch, column-segment)
    # passes it takes to cover the tile.
    N_LOADS_K = (BT // ROWS_PER_BATCH) * (K // ROW_SEG)
    N_LOADS_H = (BV // ROWS_PER_BATCH) * (K // ROW_SEG)
    N_LOADS_V = (BT // ROWS_PER_BATCH_V) * (BV // V_COLS)

    STRIDE_QK_C = Hg * K  # q / k HBM row pitch (token-major, GQA-shared)

    _kernel_name = f"chunk_gdn_fwd_o_flydsl_vk_bv{BV}"
    _kernel_deco_kwargs = (
        {} if BLOCK_THREADS == 256 else {"known_block_size": [BLOCK_THREADS, 1, 1]}
    )

    @flyc.kernel(name=_kernel_name, **_kernel_deco_kwargs)
    def gdn_o_kernel(
        q_tensor: fx.Tensor,
        k_tensor: fx.Tensor,
        v_tensor: fx.Tensor,
        h_tensor: fx.Tensor,
        g_tensor: fx.Tensor,
        o_tensor: fx.Tensor,
        cu_seqlens_tensor: fx.Tensor,
        sequence_ids_tensor: fx.Tensor,
        chunk_ids_tensor: fx.Tensor,
        T_val: fx.Int32,
        T_flat: fx.Int32,
    ):
        i_v = fx.block_idx.x
        i_tg = fx.block_idx.y
        i_nh = fx.block_idx.z

        i_h = i_nh % fx.Int32(H)
        i_b = i_nh // fx.Int32(H)

        tid = fx.thread_idx.x
        wid = tid // fx.Int32(WARP_SIZE)
        lane = tid % fx.Int32(WARP_SIZE)
        lane_n = lane % fx.Int32(16)
        lane_m_base = lane // fx.Int32(16)

        # Wave grid: wid_m owns 16 query rows, wid_n owns a slice of the N axis.
        # At NR_SPLIT == 1 this collapses to wid_m == wid, wid_n == 0.
        if const_expr(NR_SPLIT == 1):
            wid_m = wid
        else:
            wid_m = wid % fx.Int32(M_WAVES)
        wid_n = wid // fx.Int32(M_WAVES)

        def _nr_n(nr_local):
            """Column offset (in elements) of this wave's local N tile."""
            if const_expr(NR_SPLIT == 1):
                return fx.Int32(nr_local * 16)
            return (fx.Int32(nr_local * NR_SPLIT) + wid_n) * fx.Int32(16)

        def _flat_buffer(tensor):
            """1-D bounds-checked view over ``tensor``'s whole footprint."""
            buf = fx.rocdl.make_buffer_tensor(tensor, max_size=False)
            n = fx.get_scalar(fx.cosize(fx.get_layout(buf)))
            return fx.Tensor(fx.make_view(fx.get_iter(buf), fx.make_layout((n,), (1,))))

        def _elems(tensor):
            return fx.get_scalar(fx.cosize(fx.get_layout(tensor)))

        def _seq_view(tensor, base_elems, rows, row_stride, shape, stride):
            """Buffer view rooted at ``base_elems``, bounded to ``rows`` rows.

            The offset is applied to the raw global pointer BEFORE the buffer
            resource is built, so the (sequence, head) base lands in the
            descriptor's base word and ``num_records`` is measured from it. A
            read past the sequence end therefore returns a hardware zero rather
            than the neighbouring sequence's live data, which is what lets the
            tail chunk drop its row guards.
            """
            elem_bytes = tensor.element_type.width // 8
            it = fx.add_offset(fx.get_iter(tensor), base_elems)
            view = fx.make_view(it, fx.make_layout(shape, stride))
            return fx.Tensor(
                fx.rocdl.make_buffer_tensor(
                    view,
                    num_records_bytes=fx.Int64(rows * row_stride)
                    * fx.Int64(elem_bytes),
                )
            )

        o_buf = _flat_buffer(o_tensor)

        # -- Resolve (sequence, chunk) and the sequence's token window --
        if const_expr(IS_VARLEN):
            cu_buf = fx.rocdl.make_buffer_tensor(cu_seqlens_tensor, max_size=False)
            sid_buf = fx.rocdl.make_buffer_tensor(sequence_ids_tensor, max_size=False)
            cid_buf = fx.rocdl.make_buffer_tensor(chunk_ids_tensor, max_size=False)
            # The schedule is already flat over (sequence, chunk), so the grid
            # axis IS the global chunk index that h is indexed by.
            idx = i_tg * fx.Int32(INDEX_STRIDE)
            i_n = sid_buf[(idx,)]
            i_t = cid_buf[(idx,)]
            bos = cu_buf[(i_n,)]
            eos = cu_buf[(i_n + fx.Int32(1),)]
            T_local = eos - bos
            chunk_flat = i_tg
        else:
            # Dense: the grid axis counts chunks WITHIN a sequence, so h's
            # global chunk index has to be rebuilt from the batch index.
            NT = (T_val + fx.Int32(BT - 1)) // fx.Int32(BT)
            i_t = i_tg
            bos = i_b * T_val
            T_local = T_val
            chunk_flat = i_b * NT + i_t

        # -- Base offsets (element counts) --
        gqa_ratio = H // Hg
        qk_base = (bos * fx.Int32(Hg) + i_h // fx.Int32(gqa_ratio)) * fx.Int32(K)

        if const_expr(IS_VARLEN):
            # v_new is head-major [H, T_flat, V]; o is token-major [T_flat, H, V].
            v_base = (i_h * T_flat + bos) * fx.Int32(V) + i_v * fx.Int32(BV)
            o_base = (bos * fx.Int32(H) + i_h) * fx.Int32(V)
            g_base = i_h * T_flat + bos
        else:
            v_base = ((i_b * fx.Int32(H) + i_h) * T_flat) * fx.Int32(
                V
            ) + i_v * fx.Int32(BV)
            o_base = (i_b * T_val * fx.Int32(H) + i_h) * fx.Int32(V)
            g_base = (i_b * fx.Int32(H) + i_h) * T_flat
        stride_o = fx.Int32(H * V)

        # h is [NT_flat, H, V, K] with K contiguous; this CTA reads the BV rows
        # of its V-tile.
        h_base = (chunk_flat * fx.Int32(H) + i_h) * fx.Int32(V * K) + i_v * fx.Int32(
            BV * K
        )

        # -- MMA atom, shared by all three GEMMs --
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(MFMA_M, MFMA_N, MFMA_K, fx.BFloat16))
        mma = fx.make_tiled_mma(
            mma_atom, fx.make_layout((M_WAVES, NR_SPLIT, 1), (1, M_WAVES, 0))
        )

        cp_lds_x4 = fx.make_copy_atom(fx.UniversalCopy64b(), fx.BFloat16)
        cp_lds_x1 = fx.make_copy_atom(fx.UniversalCopy16b(), fx.BFloat16)
        tc_c_x1 = fx.make_tiled_copy_C(cp_lds_x1, mma).get_slice(tid)

        # -- LDS views (group-major + XOR swizzle) --
        def _swz(cols):
            """Row-keyed XOR fold over the 4-element groups of a `cols`-wide row."""
            ng = cols // 4
            return fx.static(
                fx.SwizzleType.get(int(math.log2(ng)), 2, int(math.log2(cols)) - 2)
            )

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        swz_k = _swz(K)
        swz_v = _swz(BV)
        swz_A = _swz(BT)
        k_inner = fx.make_ordered_layout((BT, K), (1, 0))
        h_inner = fx.make_ordered_layout((BV, K), (1, 0))
        v_inner = fx.make_ordered_layout((BT, BV), (1, 0))

        sK = fx.make_view(lds.lds_k.ptr, fx.make_composed_layout(swz_k, k_inner))
        sH = fx.make_view(lds.lds_h.ptr, fx.make_composed_layout(swz_k, h_inner))
        sV_store = fx.make_view(lds.lds_v.ptr, fx.make_composed_layout(swz_v, v_inner))
        # lds_v keeps the HBM orientation [BT, BV] so staging stays a plain
        # copy. GEMM4b wants (n=V, contraction=BT), so its B operand reads the
        # same bytes through a transposed view, which strides the contraction by
        # BV -- four 16b reads per fragment instead of one 64b read. Both views
        # compose the SAME swizzle over layouts that agree on the physical
        # offset, so they address identical slots.
        sV_gemm = fx.make_view(
            lds.lds_v.ptr,
            fx.make_composed_layout(swz_v, fx.make_layout((BV, BT), (1, BV))),
        )
        sA = fx.make_view(
            lds.lds_A.ptr,
            fx.make_composed_layout(swz_A, fx.make_ordered_layout((BT, BT), (1, 0))),
        )

        # Staging destinations: the swizzle scatters a thread's 8-element run
        # into a lo and a hi 4-element group, addressed from the same tile
        # coordinate via the +LDS_HALF composed offset.
        sK_hi = fx.make_view(
            lds.lds_k.ptr, fx.make_composed_layout(swz_k, LDS_HALF, k_inner)
        )
        sH_hi = fx.make_view(
            lds.lds_h.ptr, fx.make_composed_layout(swz_k, LDS_HALF, h_inner)
        )
        sV_hi = fx.make_view(
            lds.lds_v.ptr, fx.make_composed_layout(swz_v, LDS_HALF, v_inner)
        )

        # -- Tiled copies for the HBM -> LDS staging --
        cp_g2r = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        cp_r2s = fx.make_copy_atom(fx.UniversalCopy64b(), fx.BFloat16)

        def _make_tiled_pair(threads_per_row, rows_per_batch):
            """(global->reg, reg->LDS) tiled copies for one tile geometry.

            The two differ in value width: the global read moves a full
            8-element run, the LDS write moves one swizzled 4-element group.
            """
            tile = fx.make_tile(rows_per_batch, LOAD_VEC * threads_per_row)
            thr = (threads_per_row, rows_per_batch)
            thr_stride = (rows_per_batch * LOAD_VEC, 1)
            tv_load = fx.make_layout(
                (thr, (1, LOAD_VEC)), (thr_stride, (1, rows_per_batch))
            )
            tv_store = fx.make_layout(
                (thr, (1, LDS_HALF)), (thr_stride, (1, rows_per_batch))
            )
            return (
                fx.make_tiled_copy(cp_g2r, tv_load, tile).get_slice(tid),
                fx.make_tiled_copy(cp_r2s, tv_store, tile).get_slice(tid),
            )

        tc_g2r, tc_r2s = _make_tiled_pair(THREADS_PER_ROW, ROWS_PER_BATCH)
        tc_g2r_v, tc_r2s_v = _make_tiled_pair(THREADS_PER_ROW_V, ROWS_PER_BATCH_V)

        # Staging is split into issue / commit so that ALL the global reads are
        # in flight before the first LDS write forces a wait on one of them.
        def _issue(gsrc, row_offset_elems, g2r=None):
            """Start this tile's global->register read."""
            pS = (g2r or tc_g2r).partition_S(gsrc)
            frag = fx.make_fragment_like(pS)
            fx.copy(cp_g2r, pS, frag, soffset=row_offset_elems)
            return frag

        def _commit(frag, dst_lo, dst_hi, n_loads, r2s=None):
            """Land an issued tile in LDS as its two swizzled halves."""
            del r2s  # destinations already carry the partitioning
            vec = fx.Vector(frag.load())
            for off, dst in ((0, dst_lo), (LDS_HALF, dst_hi)):
                half = fx.Vector.from_elements(
                    [
                        vec[j * LOAD_VEC + off + e]
                        for j in range_constexpr(n_loads)
                        for e in range_constexpr(LDS_HALF)
                    ],
                    dtype=fx.BFloat16,
                )
                f = fx.make_fragment_like(dst)
                f.store(half)
                fx.copy(cp_r2s, f, dst)

        # q / k: [BT, K] token-major, bounded to the sequence so tail rows read
        # a hardware zero (which makes A and o_inter exactly zero there).
        chunk_row0 = i_t * fx.Int32(BT)
        gQ = _seq_view(
            q_tensor, qk_base, T_local, fx.Int32(STRIDE_QK_C), (BT, K), (STRIDE_QK_C, 1)
        )
        gK = _seq_view(
            k_tensor, qk_base, T_local, fx.Int32(STRIDE_QK_C), (BT, K), (STRIDE_QK_C, 1)
        )
        # h: [BV, K], one chunk, no tail concern -- a chunk that exists has a
        # snapshot. The bound is the tile itself.
        gH = _seq_view(h_tensor, h_base, BV, fx.Int32(K), (BV, K), (K, 1))
        # v_new: [BT, BV] head-major, bounded to the sequence like q / k.
        gV = _seq_view(v_tensor, v_base, T_local, fx.Int32(V), (BT, BV), (V, 1))

        qk_soffset = chunk_row0 * fx.Int32(STRIDE_QK_C)

        # q goes straight to an A fragment. Its 4 elements per lane run along K,
        # which is contiguous in HBM, so this is one 64b buffer load per K-tile.
        # GEMM3 and GEMM4a share the fragment -- q is read from HBM once.
        cp_q_g2r = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)
        q_cp_a = fx.make_tiled_copy_A(cp_q_g2r, mma).get_slice(tid)
        q_pS = q_cp_a.partition_S(gQ)
        frag_q = mma.make_fragment_A(gQ)
        frag_q_rt = q_cp_a.retile(frag_q)

        f_k = _issue(gK, qk_soffset)
        f_h = _issue(gH, fx.Int32(0))
        f_v = _issue(gV, chunk_row0 * fx.Int32(V), tc_g2r_v)
        for kt in range_constexpr(K_TILES):
            fx.copy(
                cp_q_g2r,
                q_pS[None, None, kt],
                frag_q_rt[None, None, kt],
                soffset=qk_soffset,
            )
        _commit(f_k, tc_r2s.partition_D(sK), tc_r2s.partition_D(sK_hi), N_LOADS_K)
        _commit(f_h, tc_r2s.partition_D(sH), tc_r2s.partition_D(sH_hi), N_LOADS_H)
        _commit(
            f_v,
            tc_r2s_v.partition_D(sV_store),
            tc_r2s_v.partition_D(sV_hi),
            N_LOADS_V,
        )

        # -- Gates --
        # The C fragment gives each thread 4 query rows of the BT tile; GEMM4a's
        # accumulator additionally spans BT_STEPS key columns. So the pair gate
        # needs g at 4 row positions and BT_STEPS column positions.
        frag_row_local = [
            wid_m * fx.Int32(16) + lane_m_base * fx.Int32(4) + fx.Int32(e)
            for e in range_constexpr(4)
        ]
        frag_row = [chunk_row0 + r for r in frag_row_local]
        frag_row_ok = [r < T_local for r in frag_row]

        if const_expr(USE_G):
            g_buf = _seq_view(
                g_tensor, g_base, T_local, fx.Int32(1), (_elems(g_tensor),), (1,)
            )
            g_row = [g_buf[(r,)] for r in frag_row]
            g_col = [
                g_buf[(chunk_row0 + _nr_n(nr) + lane_n,)]
                for nr in range_constexpr(BT_STEPS_LOCAL)
            ]

        gpu.barrier()

        # -- GEMM3: o_inter = q @ h^T  (contraction over K) --
        g3_cp_b = fx.make_tiled_copy_B(cp_lds_x4, mma).get_slice(tid)
        g3_pS_h = g3_cp_b.partition_S(sH)
        g3_fh = mma.make_fragment_B(sH)
        g3_fh_rt = g3_cp_b.retile(g3_fh)
        frag_o = fx.make_rmem_tensor(
            fx.tiled_mma_partition_shape(fx.MmaOperand.C, mma, (BT, BV)), fx.Float32
        )
        frag_o.fill(0.0)
        for kt in range_constexpr(K_TILES):
            fx.copy(cp_lds_x4, g3_pS_h[None, None, kt], g3_fh_rt[None, None, kt])
            fx.gemm(mma, frag_o, frag_q[None, None, kt], g3_fh[None, None, kt], frag_o)

        # -- GEMM4a: A = q @ k^T  (contraction over K) --
        # M = query row, N = key row, so B wants k as (n=BT, contraction=K) --
        # exactly how lds_k is stored.
        g4a_cp_b = fx.make_tiled_copy_B(cp_lds_x4, mma).get_slice(tid)
        g4a_pS_k = g4a_cp_b.partition_S(sK)
        g4a_fk = mma.make_fragment_B(sK)
        g4a_fk_rt = g4a_cp_b.retile(g4a_fk)
        frag_a = fx.make_rmem_tensor(
            fx.tiled_mma_partition_shape(fx.MmaOperand.C, mma, (BT, BT)), fx.Float32
        )
        frag_a.fill(0.0)
        for kt in range_constexpr(K_TILES):
            fx.copy(cp_lds_x4, g4a_pS_k[None, None, kt], g4a_fk_rt[None, None, kt])
            fx.gemm(mma, frag_a, frag_q[None, None, kt], g4a_fk[None, None, kt], frag_a)

        # -- Causal mask + pair gate, then publish A' to LDS --
        # The gate multiply sits INSIDE the select, not after it: above the
        # diagonal exp(g_i - g_j) can overflow to +inf, and gating a masked-to-
        # zero accumulator would turn that into 0 * inf = NaN. Evaluating the
        # product in the discarded arm keeps the surviving arm finite.
        masked = []
        for nr in range_constexpr(BT_STEPS_LOCAL):
            bt_col = _nr_n(nr) + lane_n
            a_acc = fx.Vector(frag_a[None, None, nr].load())
            for e in range_constexpr(4):
                causal = frag_row_local[e] >= bt_col
                if const_expr(USE_G):
                    gated = a_acc[e] * fx.Float32(_fast_exp(g_row[e] - g_col[nr]))
                else:
                    gated = a_acc[e]
                masked.append(causal.select(gated, fx.Float32(0.0)))
        pD_A = tc_c_x1.partition_D(sA)
        frag_A_out = fx.make_fragment_like(pD_A)
        frag_A_out.store(
            _to_bf16(
                fx.Vector.from_elements(masked, dtype=fx.Float32), BT_STEPS_LOCAL * 4
            )
        )
        fx.copy(cp_lds_x1, frag_A_out, pD_A)

        gpu.barrier()

        # -- GEMM4b: o_intra = A' @ v_new  (contraction over BT) --
        g4b_cp_a = fx.make_tiled_copy_A(cp_lds_x4, mma).get_slice(tid)
        g4b_cp_b = fx.make_tiled_copy_B(cp_lds_x1, mma).get_slice(tid)
        g4b_pS_a = g4b_cp_a.partition_S(sA)
        g4b_pS_v = g4b_cp_b.partition_S(sV_gemm)
        g4b_fa = mma.make_fragment_A(sA)
        g4b_fv = mma.make_fragment_B(sV_gemm)
        g4b_fa_rt = g4b_cp_a.retile(g4b_fa)
        g4b_fv_rt = g4b_cp_b.retile(g4b_fv)
        frag_oi = fx.make_rmem_tensor(
            fx.tiled_mma_partition_shape(fx.MmaOperand.C, mma, (BT, BV)), fx.Float32
        )
        frag_oi.fill(0.0)
        for bt_s in range_constexpr(BT_STEPS):
            fx.copy(cp_lds_x4, g4b_pS_a[None, None, bt_s], g4b_fa_rt[None, None, bt_s])
            fx.copy(cp_lds_x1, g4b_pS_v[None, None, bt_s], g4b_fv_rt[None, None, bt_s])
            fx.gemm(
                mma,
                frag_oi,
                g4b_fa[None, None, bt_s],
                g4b_fv[None, None, bt_s],
                frag_oi,
            )

        # -- Combine and store --
        # A' already carries the pair gate, so only the inter term takes a row
        # gate here.
        def _emit_o_store(off, value):
            o_buf[(off,)] = value

        scale_vec = fx.Vector.filled(4, SCALE, fx.Float32)
        if const_expr(USE_G):
            exp_gi = fx.Vector.from_elements(
                [_fast_exp(g_row[e]) for e in range_constexpr(4)], dtype=fx.Float32
            )
        for nr in range_constexpr(N_REPEAT_LOCAL):
            inter = fx.Vector(frag_o[None, None, nr].load())
            intra = fx.Vector(frag_oi[None, None, nr].load())
            if const_expr(USE_G):
                o_val = (inter * exp_gi + intra) * scale_vec
            else:
                o_val = (inter + intra) * scale_vec
            o_col = i_v * fx.Int32(BV) + _nr_n(nr) + lane_n
            for e in range_constexpr(4):
                if frag_row_ok[e].ir_value():
                    o_off = o_base + frag_row[e] * stride_o + o_col
                    _emit_o_store(o_off, _to_bf16(o_val[e]))

    @flyc.jit
    def launch_gdn_o(
        q_tensor: fx.Tensor,
        k_tensor: fx.Tensor,
        v_tensor: fx.Tensor,
        h_tensor: fx.Tensor,
        g_tensor: fx.Tensor,
        o_tensor: fx.Tensor,
        cu_seqlens_tensor: fx.Tensor,
        sequence_ids_tensor: fx.Tensor,
        chunk_ids_tensor: fx.Tensor,
        T_val: fx.Int32,
        T_flat: fx.Int32,
        grid_nt: fx.Int32,
        grid_nh: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        gdn_o_kernel(
            q_tensor,
            k_tensor,
            v_tensor,
            h_tensor,
            g_tensor,
            o_tensor,
            cu_seqlens_tensor,
            sequence_ids_tensor,
            chunk_ids_tensor,
            T_val,
            T_flat,
        ).launch(
            grid=(GRID_V, grid_nt, grid_nh),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch_gdn_o._lds_kib = _lds_kib
    launch_gdn_o._block_threads = BLOCK_THREADS
    return launch_gdn_o
