# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL FMHA backward for MLA-style d_qk=192 / d_v=128, varlen THD, causal, bf16, gfx942.

The MFMA primitive used everywhere is ``v_mfma_f32_16x16x16_bf16_1k``:

    C[16m, 16n] += A[16m, 16k] . B[16n, 16k]^T

``k_bwd`` carries TWO JOB TYPES on one grid.  Each output element is written exactly once by
exactly one workgroup -- no atomics, fully deterministic -- at the cost of computing the score
matrix once per job type:

  dK/dV job  one workgroup per (key block of 128, sequence, head); streams query tiles of 32.
             S = Q.K^T and dP = dO.V^T contract over d, so those operands come straight out of
             memory; dV = P^T.dO and dK = dS^T.Q contract over the query index, so dO^T and Q^T
             are staged transposed in LDS while P^T / dS^T come free from the accumulators.

  dQ job     one workgroup per (query block of 128, sequence, head); streams key tiles of 32.
             Computes the scores TRANSPOSED (S^T = K.Q^T) so the dS^T fragment is, for free,
             the dS operand that dQ = dS.K needs; only K^T is staged in LDS.

  pre        ``k_delta``  D = rowsum(dO * O), fp32, laid out [H, T] like lse.

Bounds: every global tensor is read and written through a buffer tensor (``_buffer_view``)
whose descriptor carries an exact ``num_records``, so out-of-range loads return 0 and stores
steered past the end are dropped by the hardware.  The causal tails, the empty-workgroup case
and every epilogue mask rely on this instead of predication.
"""

import functools
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.primitive import const_expr

DQK = 192
DV = 128
LOG2E = 1.4426950408889634

ROWS_DELTA = 32  # rows reduced per delta workgroup (2 unrolled 16-row passes)
NUM_THREADS = 512

KEYS_PER_WG = 128  # dK/dV job: keys owned by a workgroup (16 per wave)
QUERY_TILE = 32  # dK/dV job: queries streamed per iteration
LD_TOK_DKDV = QUERY_TILE + 4  # padded leading dim of the transposed LDS tiles

QUERIES_PER_WG = 128  # dQ job: queries owned by a workgroup (16 per wave)
KEY_TILE = 32  # dQ job: keys streamed per iteration
LD_TOK_DQ = KEY_TILE + 4

# Row strides of the "natural" LDS tiles.  The +4 is load-bearing: an unpadded 192-element
# stride is 96 dwords = 3*32, which puts all 16 lanes of an MFMA A-operand read on one bank
# (16-way conflict, dominated the whole kernel).
LD_QK = DQK + 4
LD_V = DV + 4

N_TILES_QK = DQK // 16  # 12 MFMA tiles along d_qk
N_TILES_V = DV // 16  # 8 MFMA tiles along d_v

# Row strides in 4-element load tiles: one token's d_qk / d_v span as seen through the
# 4-wide buffer views the operands are read from.
FRAGS_PER_ROW_QK = DQK // 4
FRAGS_PER_ROW_V = DV // 4

# ---- shared bf16 LDS arena; the two job branches are mutually exclusive so they alias --------
OFF_Q_NAT = 0  # dK/dV: Q natural   [QUERY_TILE, LD_QK]
OFF_Q_T = OFF_Q_NAT + QUERY_TILE * LD_QK  # dK/dV: Q^T         [DQK, LD_TOK_DKDV]
OFF_DO_NAT = OFF_Q_T + DQK * LD_TOK_DKDV  # dK/dV: dO natural  [QUERY_TILE, LD_V]
OFF_DO_T = OFF_DO_NAT + QUERY_TILE * LD_V  # dK/dV: dO^T        [DV, LD_TOK_DKDV]
DKDV_ARENA_END = OFF_DO_T + DV * LD_TOK_DKDV

OFF_K_NAT = 0  # dQ: K natural   [KEY_TILE, LD_QK]
OFF_K_T = OFF_K_NAT + KEY_TILE * LD_QK  # dQ: K^T         [DQK, LD_TOK_DQ]
OFF_V_NAT = OFF_K_T + DQK * LD_TOK_DQ  # dQ: V natural   [KEY_TILE, LD_V]
DQ_ARENA_END = OFF_V_NAT + KEY_TILE * LD_V

ARENA = max(DKDV_ARENA_END, DQ_ARENA_END)

# ---- VALU-reduction ablation knobs -----------------------------------------------------------
# The default is the measured winner: neither `amask` nor `bpeel` wins alone, together they are
# -2.7 % on the main case.  `ascale` is a measured LOSS (+3.6 %) kept only so it stays
# re-checkable -- do not enable it.  Timing table in DESIGN.md §8.
_OPTS = {
    x
    for x in os.environ.get("AITER_FMHA_BWD_FLYDSL_OPT", "amask,bpeel").split(",")
    if x
}
OPT_AMASK = "amask" in _OPTS  # dK/dV: one unsigned compare instead of 2 signed + and
OPT_ASCALE = (
    "ascale" in _OPTS
)  # dK/dV: LOG2E onto the LDS lse, `scale` onto the dK epilogue
OPT_BPEEL = (
    "bpeel" in _OPTS
)  # dQ: peel the causal-diagonal trips out of the streaming loop


# --------------------------------------------------------------------------- small helpers
def _mfma(a, b, acc):
    """One ``v_mfma_f32_16x16x16_bf16_1k`` on bare v4 register values."""

    mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
    fa = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.BFloat16)
    fb = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.BFloat16)
    fc = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
    fx.memref_store_vec(a, fa)
    fx.memref_store_vec(b, fb)
    fx.memref_store_vec(acc, fc)
    fx.mma_atom_call(mma, fc, fa, fb, fc)
    return fx.memref_load_vec(fc)


def _exp2(x):
    return fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, x.ir_value()))


def _zero4():
    return fx.Vector.filled(4, 0.0, fx.Float32)


def _elem(acc4, i):
    """Element ``i`` of an fp32 MFMA accumulator / LDS fragment."""
    return fx.Vector(acc4)[i]


def _pack4_trunc(vals):
    """4 fp32 -> one v4bf16 MFMA operand fragment."""
    c_mask = fx.Int32(0xFFFF0000)

    def _pair(a, b):
        a_bits = a.bitcast(fx.Int32)
        b_bits = b.bitcast(fx.Int32)
        return (b_bits & c_mask) | fx.arith.shrui(a_bits, fx.Int32(16))

    packed = fx.Vector.from_elements(
        [_pair(vals[0], vals[1]), _pair(vals[2], vals[3])], dtype=fx.Int32
    )
    return packed.bitcast(fx.BFloat16)


def _to_bf16(v):
    """fp32 -> bf16 in TWO VALU ops (add + shift), for the epilogue's scalar stores."""
    bits = v.bitcast(fx.Uint32) + fx.Uint32(0x00008000)
    return (bits >> fx.Uint32(16)).to(fx.Uint16).bitcast(fx.BFloat16)


def _buffer_view(tensor, nbytes, fx_dt, vec=1):
    """Flat ``(ntile, vec)`` buffer-tensor view with an EXACT num_records.
    ``num_records`` is a 32-bit BYTE count and ``nbytes`` is computed in i32, so this addressing
    tops out at 2 GiB per tensor; the launcher asserts that up front rather than letting it wrap.
    """
    ntile = (1 << 31) // (vec * (fx_dt.width // 8))
    buf = fx.rocdl.make_buffer_tensor(tensor, num_records_bytes=fx.Int64(nbytes))
    return fx.Tensor(
        fx.make_view(fx.get_iter(buf), fx.make_layout((ntile, vec), (vec, 1)))
    )


def _atom(fx_dt, vec):
    """Widest single buffer copy atom covering a ``vec``-wide ``fx_dt`` fragment."""
    return fx.make_copy_atom(fx.rocdl.BufferCopy(vec * fx_dt.width), fx_dt)


def _ldv(buf, tile, fx_dt, vec):
    """``vec`` elements at element offset ``tile * vec``, as a raw vector value."""
    frag = fx.make_rmem_tensor(fx.make_layout(vec, 1), fx_dt)
    fx.copy_atom_call(_atom(fx_dt, vec), fx.slice(buf, (tile, None)), frag)
    return frag.load().ir_value()


def _ld1(buf, idx, fx_dt):
    """One ``fx_dt`` element at element index ``idx``, typed."""
    frag = fx.make_rmem_tensor(fx.make_layout(1, 1), fx_dt)
    fx.copy_atom_call(_atom(fx_dt, 1), fx.slice(buf, (idx, None)), frag)
    return frag.load()[0]


def _stv(val, buf, tile, fx_dt, vec):
    """Store the ``vec``-wide vector ``val`` at element offset ``tile * vec``."""
    frag = fx.make_rmem_tensor(fx.make_layout(vec, 1), fx_dt)
    fx.memref_store_vec(val, frag)
    fx.copy_atom_call(_atom(fx_dt, vec), frag, fx.slice(buf, (tile, None)))


def _st1(val, buf, idx, fx_dt):
    """Store one ``fx_dt`` value at element index ``idx``."""
    frag = fx.make_rmem_tensor(fx.make_layout(1, 1), fx_dt)
    fx.memref_store_vec(fx.Vector.from_elements([val], dtype=fx_dt), frag)
    fx.copy_atom_call(_atom(fx_dt, 1), frag, fx.slice(buf, (idx, None)))


def _bf16x4(ptr):
    return fx.ptr_load(ptr, result_type=fx.Vector.make_type(4, fx.BFloat16))


def _packn_rne(vals):
    """2N fp32 -> one v(2N)bf16, round-to-nearest (add 0x8000 before the truncating pack).

    Same 2-op-per-element trick as ``_to_bf16`` but producing a wide packed vector, for the split-K
    reduction kernel's vectorised bf16 stores.
    """
    c_mask = fx.Int32(0xFFFF0000)
    rnd = fx.Int32(0x00008000)

    def _pair(a, b):
        a_bits = a.bitcast(fx.Int32) + rnd
        b_bits = b.bitcast(fx.Int32) + rnd
        return (b_bits & c_mask) | fx.arith.shrui(a_bits, fx.Int32(16))

    npair = len(vals) // 2
    packed = fx.Vector.from_elements(
        [_pair(vals[2 * i], vals[2 * i + 1]) for i in range(npair)], dtype=fx.Int32
    )
    return packed.bitcast(fx.BFloat16)


RED_THREADS = 256  # threads per split-K reduction workgroup
RED_VEC = 8  # bf16 lanes per thread (one dwordx4 per partial slab)


@functools.lru_cache(maxsize=4)
def build(n_split=1):
    """Compile the kernel set.  ``n_split`` = split-K factor along the STREAMED index.

    ``n_split == 1`` is the two-kernel path (k_delta + k_bwd writing bf16 straight into dq/dk/dv)
    and every Python-level branch below is written so that this variant emits exactly the code
    it emitted before split-K existed.

    ``n_split > 1`` compiles a SECOND, independently cached variant used only for small workloads
    (see ``_split()`` in aiter/ops/flydsl/fmha_bwd_gfx942.py), where every workgroup is
    co-resident and the makespan is the LONGEST SINGLE WORKGROUP rather than the total work.
    """

    # LDS storage layouts
    @fx.struct
    class SmemBwd:
        arena: fx.Array[fx.BFloat16, ARENA, 16]
        lse_s: fx.Array[fx.Float32, QUERY_TILE, 16]
        del_s: fx.Array[fx.Float32, QUERY_TILE, 16]

    # delta = rowsum(dO*O)
    DELTA_THREADS = 256
    ROWS_PER_PASS = (
        DELTA_THREADS // 16
    )  # rows covered by one 256-thread pass (16 threads per 128-wide row)
    PASSES_PER_WG = (
        ROWS_DELTA // ROWS_PER_PASS
    )  # passes per workgroup -> ROWS_DELTA rows per workgroup

    @flyc.kernel(known_block_size=[DELTA_THREADS, 1, 1])
    def k_delta(
        DO: fx.Tensor, O: fx.Tensor, DEL: fx.Tensor, Tlen: fx.Int32, Hn: fx.Int32
    ):
        """D = rowsum(dO * O).

        The 16 partial sums of a row live in 16 CONSECUTIVE LANES of one wave, so the reduction
        is a pure cross-lane XOR butterfly (4 shuffles): no LDS array and no ``fx.barrier()``.
        The row groups are unrolled PASSES_PER_WG-deep with every global load issued up front,
        because this kernel is pure streaming bandwidth.  DESIGN.md §7
        """
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        n_rows = Tlen * Hn
        g_do = _buffer_view(DO, n_rows * (DV * 2), fx.BFloat16, 8)
        g_o = _buffer_view(O, n_rows * (DV * 2), fx.BFloat16, 8)
        g_delta = _buffer_view(DEL, n_rows * 4, fx.Float32)

        # One 8-wide tile per thread; the u-th pass steps a whole ROWS_PER_PASS x DV element
        # block, a compile-time delta that folds into the load's immediate offset field.
        tile = bid * (ROWS_DELTA * DV // 8) + tid
        do_vecs = [
            _ldv(g_do, tile + u * (ROWS_PER_PASS * DV // 8), fx.BFloat16, 8)
            for u in range_constexpr(PASSES_PER_WG)
        ]
        o_vecs = [
            _ldv(g_o, tile + u * (ROWS_PER_PASS * DV // 8), fx.BFloat16, 8)
            for u in range_constexpr(PASSES_PER_WG)
        ]

        lane_in_row = tid % 16
        row_in_group = tid // 16
        for u in range_constexpr(PASSES_PER_WG):
            # Even/odd lanes accumulate into two independent fp32 chains, so the 8-element dot
            # product is not one serial dependency line.
            do8 = fx.Vector(do_vecs[u])
            o8 = fx.Vector(o_vecs[u])
            e0 = fx.Float32(0.0)
            e1 = fx.Float32(0.0)
            for c in range_constexpr(4):
                e0 = e0 + fx.Float32(do8[2 * c]) * fx.Float32(o8[2 * c])
                e1 = e1 + fx.Float32(do8[2 * c + 1]) * fx.Float32(o8[2 * c + 1])
            acc = e0 + e1
            for s in range_constexpr(4):
                acc = acc + fx.gpu.shuffle_xor(acc, 1 << s, 64)
            idx = bid * ROWS_DELTA + u * ROWS_PER_PASS + row_in_group
            ok = (lane_in_row == 0) & (idx < n_rows)
            t = idx // Hn
            h = idx % Hn
            # Only lane 0 of each row group holds the reduced sum; the other 15 and any row
            # past the end are steered one element past DEL so the hardware drops the store.
            _st1(acc, g_delta, ok.select(h * Tlen + t, n_rows), fx.Float32)

    @flyc.jit
    def launch_delta(
        DO: fx.Tensor,
        O: fx.Tensor,
        DEL: fx.Tensor,
        Tlen: fx.Int32,
        Hn: fx.Int32,
        nblk: fx.Int32,
        stream: fx.Stream,
    ):
        k_delta(DO, O, DEL, Tlen, Hn).launch(
            grid=(nblk, 1, 1), block=(DELTA_THREADS, 1, 1), stream=stream
        )

    # the fused-schedule backward kernel
    @flyc.kernel(known_block_size=[NUM_THREADS, 1, 1])
    def k_bwd(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        DO: fx.Tensor,
        LSE: fx.Tensor,
        DEL: fx.Tensor,
        CU: fx.Tensor,
        DQO: fx.Tensor,
        DKO: fx.Tensor,
        DVO: fx.Tensor,
        Tlen: fx.Int32,
        Hn: fx.Int32,
        nseq: fx.Int32,
        n_dkdv_blocks: fx.Int32,
        n_blocks: fx.Int32,
        scale: fx.Float32,
        interleave: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        lane = tid % 64
        wave = tid // 64
        lane_row = lane % 16
        lane_k = (lane // 16) * 4
        lane_k_frag = lane // 16  # lane_k as a 4-element tile index

        blk_x = fx.Int32(fx.block_idx.x)
        seq_idx = blk_x % nseq
        head = blk_x // nseq
        blk_y = fx.Int32(fx.block_idx.y)

        n_rows = Tlen * Hn
        # Q / K / V / dO are only ever read as 4-element MFMA fragments, so their views are
        # tiled 4 wide and every offset into them below is a TILE index (element offset / 4).
        gQ = _buffer_view(Q, n_rows * (DQK * 2), fx.BFloat16, 4)
        gK = _buffer_view(K, n_rows * (DQK * 2), fx.BFloat16, 4)
        gV = _buffer_view(V, n_rows * (DV * 2), fx.BFloat16, 4)
        gDO = _buffer_view(DO, n_rows * (DV * 2), fx.BFloat16, 4)
        gLSE = _buffer_view(LSE, n_rows * 4, fx.Float32)
        gDEL = _buffer_view(DEL, n_rows * 4, fx.Float32)
        gCU = _buffer_view(CU, (nseq + 1) * 4, fx.Int32)
        gDQ = _buffer_view(DQO, n_rows * (DQK * 2 * n_split), fx.BFloat16)
        gDK = _buffer_view(DKO, n_rows * (DQK * 2 * n_split), fx.BFloat16)
        gDV = _buffer_view(DVO, n_rows * (DV * 2 * n_split), fx.BFloat16)
        if const_expr(n_split > 1):
            slab_qk = n_rows * DQK  # elements per partial slab of dq / dk
            slab_v = n_rows * DV  # elements per partial slab of dv
        # Total elements behind each output view; any index >= this is past num_records, so
        # the hardware drops the store.  These are the masked-store steer targets.
        oob_qk = n_rows * (DQK * n_split)  # dq / dk, all slabs
        oob_v = n_rows * (DV * n_split)  # dv, all slabs

        seq_lo = _ld1(gCU, seq_idx, fx.Int32)
        seq_hi = _ld1(gCU, seq_idx + 1, fx.Int32)
        seq_len = seq_hi - seq_lo

        scale_log2e = scale * fx.Float32(LOG2E)

        lds = fx.SharedAllocator().allocate(SmemBwd).peek()
        lds_arena = lds.arena.ptr
        lds_lse = lds.lse_s.ptr
        lds_delta = lds.del_s.ptr

        stage_row = tid // 16
        stage_col = (tid % 16) * 4
        stage_col_frag = tid % 16  # stage_col as a 4-element tile index
        stage_tok = tid % QUERY_TILE

        # ---- XOR swizzle of the TRANSPOSED LDS tiles (Q^T, dO^T, K^T) ---------------------
        # Those tiles are filled by scalar ds_write_b16 with adjacent lanes stepping the tile ROW
        # index d, which lands the 16 lanes of a row group on only FOUR banks.  Padding cannot fix
        # it (the dword stride is always even), so the TOKEN axis is permuted instead, at the
        # 4-element granularity of the ds_read_b64 that consumes the tile:
        #
        #     element(d, tok)  ->  d*LD_TOK_DKDV + ((SW(d) ^ (tok>>2)) << 2) + (tok & 3)
        #     SW(d) = (d >> 4) & 7
        #
        # SW keys on bit 4 and up of d ON PURPOSE: that makes it constant across the 16 lanes of a
        # read (so the reads cannot regress) while varying across the storing lanes (so the stores
        # spread over 16 banks).  Swizzling on (d>>2)&7 instead is the obvious alternative and was
        # measured 3% SLOWER -- do not re-try it.
        swz_lane = (tid % 16) >> 2
        swz_store_even = ((swz_lane ^ (stage_row >> 2)) << 2) + (stage_row & 3)
        swz_store_odd = swz_store_even ^ 16
        swz_lane_base = (lane >> 4) << 2
        swz_read = [swz_lane_base ^ (k << 2) for k in range_constexpr(4)]

        def _store_col(d_chunk):
            """Swizzled token-column of a staging store of 64-element chunk `d_chunk`."""
            return swz_store_odd if (d_chunk & 1) else swz_store_even

        def _read_col(d_tile, tok_tile):
            """Swizzled token-column of an MFMA read of row block `d_tile`, tokens `tok_tile`."""
            return swz_read[d_tile & 3] + 16 * (((d_tile >> 2) & 1) ^ tok_tile)

        # ---- GRID DECODE: MERGED-LPT interleave of the two job types -------------------------
        # Both job types' per-block work is an arithmetic sequence with the SAME slope of 4
        # iterations per index step, so the true longest-processing-time-first order over their
        # UNION is a 1:1 interleave of the two descending lists:
        #   blk_y even -> dK/dV block blk_y/2 ; blk_y odd -> dQ block (n_dq_blocks-1 - blk_y/2)
        # The machine then always holds a mix of both job types.  Simply CONCATENATING the two
        # lists instead was measured to cost the full SUM of the two job types' standalone times.
        #
        # The n_paired / tail_idx arithmetic covers n_dkdv_blocks != n_dq_blocks, unreachable
        # while max_seqlen_q == max_seqlen_k but the kernel must stay general.
        n_dq_blocks = n_blocks - n_dkdv_blocks
        n_paired = (n_dkdv_blocks < n_dq_blocks).select(n_dkdv_blocks, n_dq_blocks)
        pair_idx = blk_y // 2
        tail_idx = blk_y - n_paired * 2
        in_paired = blk_y < n_paired * 2
        use_interleave = interleave > 0
        is_dkdv = use_interleave.select(
            in_paired.select(
                ((blk_y % 2) == 0).select(fx.Int32(1), fx.Int32(0)),
                (n_dkdv_blocks > n_dq_blocks).select(fx.Int32(1), fx.Int32(0)),
            ),
            (blk_y < n_dkdv_blocks).select(fx.Int32(1), fx.Int32(0)),
        )
        dkdv_blk = use_interleave.select(
            in_paired.select(pair_idx, n_paired + tail_idx), blk_y
        )
        dq_blk = use_interleave.select(
            in_paired.select(
                n_dq_blocks - 1 - pair_idx, n_dq_blocks - 1 - n_paired - tail_idx
            ),
            (n_blocks - 1) - blk_y,
        )

        if const_expr(n_split > 1):
            # The host multiplied the block counts by n_split, so the decoded index carries the
            # split index in its low digit.  All splits of one block cost the same, so the
            # division keeps both job lists descending as the merged-LPT decode assumes.
            dkdv_split = dkdv_blk % n_split
            dkdv_blk = dkdv_blk // n_split
            dq_split = dq_blk % n_split
            dq_blk = dq_blk // n_split

        if is_dkdv > 0:
            # dK/dV JOB
            lds_q = lds_arena + OFF_Q_NAT
            lds_qt = lds_arena + OFF_Q_T
            lds_do = lds_arena + OFF_DO_NAT
            lds_dot = lds_arena + OFF_DO_T

            key_block_base = dkdv_blk * KEYS_PER_WG

            if key_block_base < seq_len:
                # --- K / V operand fragments for this wave's 16 keys, register-resident -------
                key_in_blk = wave * 16 + lane_row  # key row inside the block
                key_row = seq_lo + key_block_base + key_in_blk
                # The 12 K / 8 V loads differ only by a CONSTANT d-tile delta, so one vgpr address
                # is computed here and each load folds its delta into the immediate offset field.
                k_frag_base = (key_row * Hn + head) * FRAGS_PER_ROW_QK + lane_k_frag
                v_frag_base = (key_row * Hn + head) * FRAGS_PER_ROW_V + lane_k_frag
                k_frags = []
                for d_tile in range_constexpr(N_TILES_QK):
                    k_frags.append(_ldv(gK, k_frag_base + d_tile * 4, fx.BFloat16, 4))
                v_frags = []
                for d_tile in range_constexpr(N_TILES_V):
                    v_frags.append(_ldv(gV, v_frag_base + d_tile * 4, fx.BFloat16, 4))

                # software-pipelined staging
                def _load_query_tile(q0):
                    frags = []
                    row = (seq_lo + q0 + stage_row) * Hn + head
                    for d_chunk in range_constexpr(DQK // 64):
                        frags.append(
                            _ldv(
                                gQ,
                                row * FRAGS_PER_ROW_QK + stage_col_frag + d_chunk * 16,
                                fx.BFloat16,
                                4,
                            )
                        )
                    for d_chunk in range_constexpr(DV // 64):
                        frags.append(
                            _ldv(
                                gDO,
                                row * FRAGS_PER_ROW_V + stage_col_frag + d_chunk * 16,
                                fx.BFloat16,
                                4,
                            )
                        )
                    frags.append(
                        _ld1(gLSE, head * Tlen + seq_lo + q0 + stage_tok, fx.Float32)
                    )
                    frags.append(
                        _ld1(gDEL, head * Tlen + seq_lo + q0 + stage_tok, fx.Float32)
                    )
                    return frags

                N_Q_CHUNKS = DQK // 64
                N_DO_CHUNKS = DV // 64
                N_PREFETCH_DKDV = N_Q_CHUNKS + N_DO_CHUNKS + 2
                if const_expr(n_split == 1):
                    q_lo = key_block_base
                    q_hi = seq_len
                else:
                    # Take chunk `dkdv_split` of this block's full streamed range
                    # [key_block_base, seq_len), cut on QUERY_TILE boundaries so the causal mask
                    # logic is untouched.
                    q_ntiles = (
                        seq_len - key_block_base + (QUERY_TILE - 1)
                    ) // QUERY_TILE
                    q_lo = (
                        key_block_base
                        + ((q_ntiles * dkdv_split) // n_split) * QUERY_TILE
                    )
                    q_hi_split = (
                        key_block_base
                        + ((q_ntiles * (dkdv_split + 1)) // n_split) * QUERY_TILE
                    )
                    q_hi = (q_hi_split < seq_len).select(q_hi_split, seq_len)
                init_state = [
                    _zero4() for _ in range_constexpr(N_TILES_QK + N_TILES_V)
                ] + _load_query_tile(q_lo)
                out_state = init_state
                for q_base_raw, state in range(q_lo, q_hi, QUERY_TILE, init=init_state):
                    q_base = fx.Int32(q_base_raw)
                    dk_acc = [state[i] for i in range_constexpr(N_TILES_QK)]
                    dv_acc = [state[N_TILES_QK + i] for i in range_constexpr(N_TILES_V)]
                    staged = [
                        state[N_TILES_QK + N_TILES_V + i]
                        for i in range_constexpr(N_PREFETCH_DKDV)
                    ]

                    fx.barrier()
                    for d_chunk in range_constexpr(N_Q_CHUNKS):
                        d_col = stage_col + d_chunk * 64
                        q_vec = fx.Vector(staged[d_chunk])
                        fx.ptr_store(q_vec, lds_q + (stage_row * LD_QK + d_col))
                        for elem in range_constexpr(4):
                            fx.ptr_store(
                                q_vec[elem],
                                lds_qt
                                + ((d_col + elem) * LD_TOK_DKDV + _store_col(d_chunk)),
                            )
                    for d_chunk in range_constexpr(N_DO_CHUNKS):
                        d_col = stage_col + d_chunk * 64
                        do_vec = fx.Vector(staged[N_Q_CHUNKS + d_chunk])
                        fx.ptr_store(do_vec, lds_do + (stage_row * LD_V + d_col))
                        for elem in range_constexpr(4):
                            fx.ptr_store(
                                do_vec[elem],
                                lds_dot
                                + ((d_col + elem) * LD_TOK_DKDV + _store_col(d_chunk)),
                            )
                    if const_expr(OPT_ASCALE):
                        # LOG2E folded onto lse ONCE per staged row (1 VALU/thread/iteration)
                        # instead of once per score element (8/thread/iteration).
                        fx.ptr_store(
                            fx.Float32(staged[N_Q_CHUNKS + N_DO_CHUNKS])
                            * fx.Float32(LOG2E),
                            lds_lse + stage_tok,
                        )
                    else:
                        fx.ptr_store(
                            staged[N_Q_CHUNKS + N_DO_CHUNKS], lds_lse + stage_tok
                        )
                    fx.ptr_store(
                        staged[N_Q_CHUNKS + N_DO_CHUNKS + 1], lds_delta + stage_tok
                    )
                    fx.barrier()
                    next_tile = _load_query_tile(
                        q_base + QUERY_TILE
                    )  # prefetch for the next iteration

                    key_col = (
                        key_block_base + key_in_blk
                    )  # this lane's key column (C-frag n index)
                    # per-iteration base of the unsigned causal+tail test (see OPT_AMASK)
                    mask_base = (q_base + lane_k) - key_col
                    keys_to_end = seq_len - key_col  # loop-invariant; LICM hoists it
                    for q_sub in range_constexpr(QUERY_TILE // 16):
                        s_acc = _zero4()
                        for d_tile in range_constexpr(N_TILES_QK):
                            s_acc = _mfma(
                                _bf16x4(
                                    lds_q
                                    + (
                                        (q_sub * 16 + lane_row) * LD_QK
                                        + d_tile * 16
                                        + lane_k
                                    )
                                ),
                                k_frags[d_tile],
                                s_acc,
                            )
                        dp_acc = _zero4()
                        for d_tile in range_constexpr(N_TILES_V):
                            dp_acc = _mfma(
                                _bf16x4(
                                    lds_do
                                    + (
                                        (q_sub * 16 + lane_row) * LD_V
                                        + d_tile * 16
                                        + lane_k
                                    )
                                ),
                                v_frags[d_tile],
                                dp_acc,
                            )

                        lse4 = fx.ptr_load(
                            lds_lse + (q_sub * 16 + lane_k),
                            result_type=fx.Vector.make_type(4, fx.Float32),
                        )
                        delta4 = fx.ptr_load(
                            lds_delta + (q_sub * 16 + lane_k),
                            result_type=fx.Vector.make_type(4, fx.Float32),
                        )

                        p_vals = []
                        ds_vals = []
                        for i in range_constexpr(4):
                            if const_expr(OPT_AMASK):
                                in_causal = fx.Uint32(
                                    mask_base + (q_sub * 16 + i)
                                ) < fx.Uint32(keys_to_end)
                            else:
                                query_idx = q_base + q_sub * 16 + lane_k + i
                                in_causal = (query_idx < seq_len) & (
                                    key_col <= query_idx
                                )
                            if const_expr(OPT_ASCALE):
                                # lse already carries LOG2E (folded at the LDS store): one v_fma.
                                logit = _elem(s_acc, i) * scale_log2e - _elem(lse4, i)
                            else:
                                logit = (
                                    _elem(s_acc, i) * scale - _elem(lse4, i)
                                ) * fx.Float32(LOG2E)
                            p_raw = _exp2(logit)
                            # Both mask forms are fx predicates, so one select serves either.
                            p_val = in_causal.select(p_raw, fx.Float32(0.0))
                            if const_expr(OPT_ASCALE):
                                # `scale` rides on the dK epilogue instead (dK = dS^T.Q is
                                # linear in it; dV = P^T.dO must NOT be scaled).
                                ds_val = p_val * (_elem(dp_acc, i) - _elem(delta4, i))
                            else:
                                ds_val = (
                                    p_val
                                    * (_elem(dp_acc, i) - _elem(delta4, i))
                                    * scale
                                )
                            p_vals.append(p_val)
                            ds_vals.append(ds_val)
                        p_frag = _pack4_trunc(p_vals)  # == P^T operand (row = j, k = i)
                        ds_frag = _pack4_trunc(
                            ds_vals
                        )  # == dS^T operand (row = j, k = i)

                        for d_tile in range_constexpr(N_TILES_V):
                            do_t_frag = _bf16x4(
                                lds_dot
                                + (
                                    (d_tile * 16 + lane_row) * LD_TOK_DKDV
                                    + _read_col(d_tile, q_sub)
                                )
                            )
                            dv_acc[d_tile] = _mfma(p_frag, do_t_frag, dv_acc[d_tile])
                        for d_tile in range_constexpr(N_TILES_QK):
                            q_t_frag = _bf16x4(
                                lds_qt
                                + (
                                    (d_tile * 16 + lane_row) * LD_TOK_DKDV
                                    + _read_col(d_tile, q_sub)
                                )
                            )
                            dk_acc[d_tile] = _mfma(ds_frag, q_t_frag, dk_acc[d_tile])

                    out_state = yield dk_acc + dv_acc + next_tile

                # epilogue: C fragments are (m = key row, n = d)
                for i in range_constexpr(4):
                    key_idx = key_block_base + wave * 16 + lane_k + i
                    in_seq = key_idx < seq_len
                    out_row = seq_lo + key_idx
                    if const_expr(n_split == 1):
                        dk_base = in_seq.select(
                            (out_row * Hn + head) * DQK + lane_row, oob_qk
                        )
                        dv_base = in_seq.select(
                            (out_row * Hn + head) * DV + lane_row, oob_v
                        )
                    else:
                        # Partial into slab `dkdv_split`.  Every (token, d, split) slot is written
                        # by exactly one workgroup, so the workspace needs no zeroing.
                        dk_base = in_seq.select(
                            dkdv_split * slab_qk
                            + (out_row * Hn + head) * DQK
                            + lane_row,
                            oob_qk,
                        )
                        dv_base = in_seq.select(
                            dkdv_split * slab_v + (out_row * Hn + head) * DV + lane_row,
                            oob_v,
                        )
                    for d_tile in range_constexpr(N_TILES_QK):
                        _st1(
                            _to_bf16(_elem(out_state[d_tile], i)),
                            gDK,
                            dk_base + d_tile * 16,
                            fx.BFloat16,
                        )
                    for d_tile in range_constexpr(N_TILES_V):
                        _st1(
                            _to_bf16(_elem(out_state[N_TILES_QK + d_tile], i)),
                            gDV,
                            dv_base + d_tile * 16,
                            fx.BFloat16,
                        )
        else:
            # dQ JOB
            lds_k = lds_arena + OFF_K_NAT
            lds_kt = lds_arena + OFF_K_T
            lds_v = lds_arena + OFF_V_NAT

            # dq_blk is reversed by the grid decode: this job's cost grows with the query block
            # index, so reversing keeps its half of the list descending like the other one.
            query_block_base = dq_blk * QUERIES_PER_WG
            # EMPTY-WORKGROUP EARLY EXIT -- see the dK/dV job's note.
            if query_block_base < seq_len:

                # --- Q / dO operand fragments for this wave's 16 queries, register-resident ---
                query_in_blk = wave * 16 + lane_row
                query_idx = query_block_base + query_in_blk
                query_row = seq_lo + query_idx
                q_frag_base = (query_row * Hn + head) * FRAGS_PER_ROW_QK + lane_k_frag
                do_frag_base = (query_row * Hn + head) * FRAGS_PER_ROW_V + lane_k_frag
                q_frags = []
                for d_tile in range_constexpr(N_TILES_QK):
                    q_frags.append(_ldv(gQ, q_frag_base + d_tile * 4, fx.BFloat16, 4))
                do_frags = []
                for d_tile in range_constexpr(N_TILES_V):
                    do_frags.append(
                        _ldv(gDO, do_frag_base + d_tile * 4, fx.BFloat16, 4)
                    )
                # Pre-scaled once per workgroup so the inner loop costs one v_fma per score
                # element instead of mul + sub + mul.
                lse_scaled = _ld1(
                    gLSE, head * Tlen + query_row, fx.Float32
                ) * fx.Float32(LOG2E)
                delta_scaled = _ld1(gDEL, head * Tlen + query_row, fx.Float32) * scale
                query_in_seq = query_idx < seq_len

                # A key j is needed iff j <= i for some owned query i, so j < query_block_base +
                # QUERIES_PER_WG, and j < seq_len.
                query_end = query_block_base + QUERIES_PER_WG
                key_end_full = (query_end < seq_len).select(query_end, seq_len)
                key_end = (query_block_base < seq_len).select(key_end_full, fx.Int32(0))

                def _load_key_tile(k0):
                    frags = []
                    row = (seq_lo + k0 + stage_row) * Hn + head
                    for d_chunk in range_constexpr(DQK // 64):
                        frags.append(
                            _ldv(
                                gK,
                                row * FRAGS_PER_ROW_QK + stage_col_frag + d_chunk * 16,
                                fx.BFloat16,
                                4,
                            )
                        )
                    for d_chunk in range_constexpr(DV // 64):
                        frags.append(
                            _ldv(
                                gV,
                                row * FRAGS_PER_ROW_V + stage_col_frag + d_chunk * 16,
                                fx.BFloat16,
                                4,
                            )
                        )
                    return frags

                N_K_CHUNKS = DQK // 64
                N_V_CHUNKS = DV // 64
                N_PREFETCH_DQ = N_K_CHUNKS + N_V_CHUNKS
                if const_expr(n_split == 1):
                    k_lo = fx.Int32(0)
                    k_hi = key_end
                else:
                    k_ntiles = (key_end + (KEY_TILE - 1)) // KEY_TILE
                    k_lo = ((k_ntiles * dq_split) // n_split) * KEY_TILE
                    k_hi_split = ((k_ntiles * (dq_split + 1)) // n_split) * KEY_TILE
                    k_hi = (k_hi_split < key_end).select(k_hi_split, key_end)
                init_state = [
                    _zero4() for _ in range_constexpr(N_TILES_QK)
                ] + _load_key_tile(k_lo)

                def _stream_key_tile(state, key_base, masked):
                    """One streamed key tile.  `masked` selects the causal-diagonal variant.

                    With OPT_BPEEL the caller runs the mask-free variant over the prefix of key
                    tiles that lie wholly below the causal diagonal -- all but the last <=4 trips
                    -- saving 4 VALU per score element there.
                    """
                    dq_acc = [state[i] for i in range_constexpr(N_TILES_QK)]
                    staged = [
                        state[N_TILES_QK + i] for i in range_constexpr(N_PREFETCH_DQ)
                    ]

                    fx.barrier()
                    for d_chunk in range_constexpr(N_K_CHUNKS):
                        d_col = stage_col + d_chunk * 64
                        k_vec = fx.Vector(staged[d_chunk])
                        fx.ptr_store(k_vec, lds_k + (stage_row * LD_QK + d_col))
                        for elem in range_constexpr(4):
                            fx.ptr_store(
                                k_vec[elem],
                                lds_kt
                                + ((d_col + elem) * LD_TOK_DQ + _store_col(d_chunk)),
                            )
                    for d_chunk in range_constexpr(N_V_CHUNKS):
                        d_col = stage_col + d_chunk * 64
                        fx.ptr_store(
                            staged[N_K_CHUNKS + d_chunk],
                            lds_v + (stage_row * LD_V + d_col),
                        )
                    fx.barrier()
                    next_tile = _load_key_tile(
                        key_base + KEY_TILE
                    )  # prefetch for the next iteration

                    for key_sub in range_constexpr(KEY_TILE // 16):
                        s_t_acc = _zero4()
                        for d_tile in range_constexpr(N_TILES_QK):
                            s_t_acc = _mfma(
                                _bf16x4(
                                    lds_k
                                    + (
                                        (key_sub * 16 + lane_row) * LD_QK
                                        + d_tile * 16
                                        + lane_k
                                    )
                                ),
                                q_frags[d_tile],
                                s_t_acc,
                            )
                        dp_t_acc = _zero4()
                        for d_tile in range_constexpr(N_TILES_V):
                            dp_t_acc = _mfma(
                                _bf16x4(
                                    lds_v
                                    + (
                                        (key_sub * 16 + lane_row) * LD_V
                                        + d_tile * 16
                                        + lane_k
                                    )
                                ),
                                do_frags[d_tile],
                                dp_t_acc,
                            )

                        ds_vals = []
                        for i in range_constexpr(4):
                            logit = _elem(s_t_acc, i) * scale_log2e - lse_scaled
                            p_raw = _exp2(logit)
                            if const_expr(masked):
                                key_idx = key_base + key_sub * 16 + lane_k + i
                                if const_expr(OPT_BPEEL):
                                    in_causal = key_idx <= query_idx
                                else:
                                    in_causal = query_in_seq & (key_idx <= query_idx)
                                p_val = in_causal.select(p_raw, fx.Float32(0.0))
                            else:
                                p_val = p_raw
                            ds_vals.append(
                                p_val * (_elem(dp_t_acc, i) * scale - delta_scaled)
                            )
                        ds_frag = _pack4_trunc(
                            ds_vals
                        )  # == dS operand (row = i, k = j)

                        for d_tile in range_constexpr(N_TILES_QK):
                            k_t_frag = _bf16x4(
                                lds_kt
                                + (
                                    (d_tile * 16 + lane_row) * LD_TOK_DQ
                                    + _read_col(d_tile, key_sub)
                                )
                            )
                            dq_acc[d_tile] = _mfma(ds_frag, k_t_frag, dq_acc[d_tile])
                    return dq_acc + next_tile

                out_state = init_state
                if const_expr(OPT_BPEEL):
                    # peel point = query_block_base clamped into this split's [k_lo, k_hi)
                    peel_point = (query_block_base < k_hi).select(
                        query_block_base, k_hi
                    )
                    peel_point = (peel_point > k_lo).select(peel_point, k_lo)
                    # mask-free prefix: key tiles wholly below the diagonal
                    for key_base_raw, state in range(
                        k_lo, peel_point, KEY_TILE, init=init_state
                    ):
                        out_state = yield _stream_key_tile(
                            state, fx.Int32(key_base_raw), False
                        )
                    # <=4 diagonal-straddling trips keep the full predicate
                    mid_state = out_state
                    for key_base_raw, state in range(
                        peel_point, k_hi, KEY_TILE, init=mid_state
                    ):
                        out_state = yield _stream_key_tile(
                            state, fx.Int32(key_base_raw), True
                        )
                else:
                    for key_base_raw, state in range(
                        k_lo, k_hi, KEY_TILE, init=init_state
                    ):
                        out_state = yield _stream_key_tile(
                            state, fx.Int32(key_base_raw), True
                        )

                for i in range_constexpr(4):
                    query_idx_out = query_block_base + wave * 16 + lane_k + i
                    in_seq = query_idx_out < seq_len
                    out_row = seq_lo + query_idx_out
                    if const_expr(n_split == 1):
                        dq_base = in_seq.select(
                            (out_row * Hn + head) * DQK + lane_row, oob_qk
                        )
                    else:
                        dq_base = in_seq.select(
                            dq_split * slab_qk + (out_row * Hn + head) * DQK + lane_row,
                            oob_qk,
                        )
                    for d_tile in range_constexpr(N_TILES_QK):
                        _st1(
                            _to_bf16(_elem(out_state[d_tile], i)),
                            gDQ,
                            dq_base + d_tile * 16,
                            fx.BFloat16,
                        )

    @flyc.jit
    def launch_bwd(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        DO: fx.Tensor,
        LSE: fx.Tensor,
        DEL: fx.Tensor,
        CU: fx.Tensor,
        DQO: fx.Tensor,
        DKO: fx.Tensor,
        DVO: fx.Tensor,
        Tlen: fx.Int32,
        Hn: fx.Int32,
        scale: fx.Float32,
        n_dkdv_blocks: fx.Int32,
        n_blocks: fx.Int32,
        nseq: fx.Int32,
        interleave: fx.Int32,
        stream: fx.Stream,
    ):
        k_bwd(
            Q,
            K,
            V,
            DO,
            LSE,
            DEL,
            CU,
            DQO,
            DKO,
            DVO,
            Tlen,
            Hn,
            nseq,
            n_dkdv_blocks,
            n_blocks,
            scale,
            interleave,
        ).launch(
            grid=(nseq * Hn, n_blocks, 1), block=(NUM_THREADS, 1, 1), stream=stream
        )

    if const_expr(n_split == 1):
        return launch_delta, launch_bwd

    # ------------------------------------------ split-K reduction (n_split > 1 variant only)
    @flyc.kernel(known_block_size=[RED_THREADS, 1, 1])
    def k_red(WS: fx.Tensor, OUT: fx.Tensor, n: fx.Int32):
        """OUT[i] = bf16( sum_p WS[p*n + i] ), i over one whole gradient tensor.

        Perfectly coalesced dwordx4 streams; n is always a multiple of RED_VEC (it is
        T*H*192 or T*H*128), so a thread's 8-element group is either wholly inside
        num_records or wholly outside and the hardware bounds check makes the tail branchless.
        """
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        gWS = _buffer_view(WS, n * (2 * n_split), fx.BFloat16, RED_VEC)
        gOUT = _buffer_view(OUT, n * 2, fx.BFloat16, RED_VEC)
        tile = bid * RED_THREADS + tid
        tiles_per_slab = n // RED_VEC  # tile stride between partial slabs
        partials = []
        for slab in range_constexpr(n_split):
            partials.append(
                fx.Vector(_ldv(gWS, tile + tiles_per_slab * slab, fx.BFloat16, RED_VEC))
            )
        vals = []
        for e in range_constexpr(RED_VEC):
            acc = fx.Float32(partials[0][e])
            for slab in range_constexpr(1, n_split):
                acc = acc + fx.Float32(partials[slab][e])
            vals.append(acc)
        _stv(_packn_rne(vals), gOUT, tile, fx.BFloat16, RED_VEC)

    @flyc.jit
    def launch_red(
        WS: fx.Tensor,
        OUT: fx.Tensor,
        n: fx.Int32,
        nblk: fx.Int32,
        stream: fx.Stream,
    ):
        k_red(WS, OUT, n).launch(
            grid=(nblk, 1, 1), block=(RED_THREADS, 1, 1), stream=stream
        )

    return launch_delta, launch_bwd, launch_red
