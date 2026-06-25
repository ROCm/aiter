# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""FlyDSL **layout-API** port of aiter ``gemm1_a4w4`` (MXFP4 MoE up/gate-proj).

Variant: ``(BM=32, inline_quant=False)`` for both ``use_nt`` and both gate modes
(kSubBlocks=1, kMChunks=2, kAStages=3, kStages=2). ``use_nt`` is the B-load cache
policy (tuned config BM32_NT vs BM32_CACHED): True -> non-temporal, False -> cached.
``interleave`` picks the gate/up layout (True=interleaved, False=separated); it only
changes the B-load column, the B-scale n-pack base, and the mfma opsel (the epilogue
gate/up split is the same for both, since accm[J] holds the same logical (g/u, n0)).

Layout-API pieces (the B/B-scale views, copy atoms, and the scaled-MFMA atom set +
``gemm_mma`` are shared with gemm2 via ``mxfp4_moe_layout``):
  * B load -> ``ml.bq_view`` + ``fx.copy`` into register fragments; the nt/cached
    policy rides on the copy atom's ``cache_modifier``.
  * B-scale load -> ``ml.bscale_view`` + ``fx.copy`` (32b, cached) into per-stage i32
    fragments; the per-K-tile offset rides the voffset (no hi/lo split).
  * MMA -> one ``ml.gemm_mma`` (fx.gemm) per mfma over rank-1 register fragments
    (A/B/C), with per-K-block e8m0 scales via ``scale_a=/scale_b=`` and a pre-built
    opsel-specialized ``MFMA_Scale`` atom per (opselA,opselB). C accumulates in
    place (d == c). mem2reg folds the fragment plumbing -> ISA == the raw intrinsic.

Kept raw (self-contained helpers below): math/quant/pointer helpers, the A-side
LDS stage + ds-read addressing + A-scale machinery, and the epilogue math. (B and
B-scale now both ride the layout API; only the A path and epilogue stay raw.)
Acceptance: KIMI BM32 interleave numeric gate (mean_row_cos > 0.85).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import memref as memref_dialect
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from . import mxfp4_moe_layout as ml

# ---- constants (KIMI defaults; per-shape values come from compile args) ------
NE_DEFAULT, K_DEFAULT, INTER_DEFAULT, TOPK_DEFAULT = 385, 7168, 512, 9
BN = BK = 256
KH_TILE = BK // 2  # 128 packed bytes per K-tile
kStages = 2
kBS_stride_k0_dw = 64
LOG2E = 1.4426950408889634
_PTR3 = "!llvm.ptr<3>"

# BM32 path: fixed for the single supported variant.
BM = 32
kAStages = 3
kSubBlocks = 1
kMChunks = 2  # BM // 16
M_REPS = BM // 16
BN_INT = BN // 2  # 128


# ---- self-contained math / pointer / size helpers ---------------------------
def _raw(v):
    """Unwrap an fx value to a raw ir.Value."""
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def _lds_ptr3(base_i32, byte_off_i32):
    return llvm.inttoptr(ir.Type.parse(_PTR3), _raw(fx.Int64(base_i32 + byte_off_i32)))


def _lds_base_ptr3(lds_view):
    base_i32 = fx.Int32(memref_dialect.extract_aligned_pointer_as_index(lds_view))
    return llvm.inttoptr(ir.Type.parse(_PTR3), _raw(fx.Int64(base_i32)))


def _gep3(base_ptr, byte_off_i32):
    return buffer_ops.get_element_ptr(base_ptr, byte_offset=_raw(byte_off_i32), elem_type=T.i8)


def _global_base_ptr1(addr_i64):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), _raw(fx.Int64(addr_i64)))


def _gep1(base_ptr, byte_off_i32):
    return buffer_ops.get_element_ptr(base_ptr, byte_offset=_raw(byte_off_i32), elem_type=T.i8)


def _global_ptr1(arg, byte_off_i32):
    return _gep1(_global_base_ptr1(arg), byte_off_i32)


def _lds_swizzle_mask(row):
    """lds_swizzle_mask<ROW_BYTES=128>(row) = (row & 14) << 3."""
    return (row & fx.Int32(14)) << fx.Int32(3)


def _silu_mul_batch(gs, us):
    """silu(g)*u via exp2/rcp (matches HIP silu_mul_fast)."""
    e = [fx.Float32(rocdl.exp2(T.f32, _raw(g * fx.Float32(-LOG2E)))) for g in gs]
    sig = [fx.Float32(rocdl.rcp(T.f32, _raw(fx.Float32(1.0) + ei))) for ei in e]
    return [gs[i] * sig[i] * us[i] for i in range(len(gs))]


def _fabs_f32(x):
    """fabsf via sign-bit clear (FlyDSL has no arith.absf)."""
    abs_bits = _raw(x).bitcast(T.i32) & _raw(fx.Int32(0x7FFFFFFF))
    return fx.Float32(abs_bits.bitcast(T.f32))


def _e8m0_from_amax(amax_f32):
    """(e8m0_i32, quant_scale_f32) = ceil_pow2(amax/6) clamped to 254."""
    wi = fx.Int32(_raw(amax_f32 * fx.Float32(1.0 / 6.0)).bitcast(T.i32))
    bexp = (wi + fx.Int32(0x7FFFFF)).shrui(fx.Int32(23)) & fx.Int32(0xFF)
    lt = arith.cmpi(arith.CmpIPredicate.ult, _raw(bexp), _raw(fx.Int32(254)))
    e8m0 = fx.Int32(arith.select(lt, _raw(bexp), _raw(fx.Int32(254))))
    qscale = fx.Float32(_raw(e8m0 << fx.Int32(23)).bitcast(T.f32))
    return e8m0, qscale


def gemm1_grid(n_tokens, BM=32, NE=NE_DEFAULT, TOPK=TOPK_DEFAULT, INTER=INTER_DEFAULT):
    """Host-side grid size (BM=32 active-experts bound)."""
    active = min(n_tokens * TOPK, NE)
    max_m_blocks = (n_tokens * TOPK + active * (BM - 1) + BM - 1) // BM
    return max_m_blocks * ((2 * INTER) // 256)


@flyc.jit
def _gemm1_body_v2(
    allocator,
    lds_off,
    arg_aq,
    arg_ascale,
    arg_bq,
    arg_bscale,
    arg_eids,
    arg_mind,
    arg_aqout,
    arg_ascaleout,
    bx_i32,
    lane,
    wave,
    i32_ntok,
    i32_total_m_blocks,
    *,
    K,
    INTER,
    NE,
    interleave,
    b_nontemporal,
):
    # K-/INTER-derived sizes (compile-time Python ints; were the v1 *_for helpers).
    _kc = (K // 32) // 4 // 2
    K_HALF = K // 2
    K_TILES_TOTAL = K // BK
    kUnroll = K_TILES_TOTAL - kStages
    kAS_per_chunk_dw = _kc * 64
    kBS_stride_n0_dw = _kc * 64
    N_OUT = 2 * INTER
    kBS_per_expert_dw = (N_OUT // 16 // 2) * kBS_stride_n0_dw
    NUM_N_BLOCKS = N_OUT // 256
    OUT_AS_PER_CHUNK_DW = ((INTER // 32) // 4 // 2) * 64
    K_G2_HALF = INTER // 2

    # block -> (m_block_idx, n_block_idx) ; e = sorted_expert_ids[m_block_idx]
    n_block_idx = bx_i32 % fx.Int32(NUM_N_BLOCKS)
    m_block_idx = bx_i32 // fx.Int32(NUM_N_BLOCKS)
    e = rocdl.readfirstlane(
        T.i32, llvm.load(T.i32, _global_ptr1(arg_eids, m_block_idx * fx.Int32(4)))
    )
    m_row = m_block_idx * fx.Int32(BM)

    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lane_div_8 = lane // fx.Int32(8)
    lane_mod_8 = lane % fx.Int32(8)

    # buffer resources (A-gather + scales)
    aq_num_records = arith.index_cast(T.index, _raw(i32_ntok * fx.Int32(K_HALF)))
    aq_rsrc = buffer_ops.create_buffer_resource_from_addr(
        _raw(fx.Int64(arg_aq)), num_records_bytes=aq_num_records
    )
    _asc_per_mb = max(BM // 32, 1) * kAS_per_chunk_dw * 4
    ascale_num = arith.index_cast(T.index, _raw(i32_total_m_blocks)) * fx.Index(
        _asc_per_mb
    )
    ascale_rsrc = buffer_ops.create_buffer_resource_from_addr(
        _raw(fx.Int64(arg_ascale)), num_records_bytes=ascale_num
    )

    # LDS views (s_aq / s_asc, union-overlapping lds_acc)
    lds_base = allocator.get_base()
    s_aq = SmemPtr(lds_base, lds_off, T.i8, shape=(kAStages * BM * KH_TILE,))
    s_asc = SmemPtr(
        lds_base,
        lds_off + kAStages * BM * KH_TILE,
        T.i8,
        shape=(kSubBlocks * K_TILES_TOTAL * 256,),
    )
    lds_acc = SmemPtr(lds_base, lds_off, T.f32, shape=(BM * BN,))

    # cached A rows (A-gather base offset)
    cached_actual_row = []
    for sub in range_constexpr(kSubBlocks):
        idx = m_row + wave * fx.Int32(BM // 4) + fx.Int32(sub * 8) + lane_div_8
        cached_actual_row.append(
            llvm.load(T.i32, _global_ptr1(arg_mind, idx * fx.Int32(4)))
        )

    # B-scale n-pack words (gate/up split differs by mode); the per-(wave,mw) base
    # is uniform, the per-lane + per-K-tile parts become layout axes (see views below).
    if const_expr(interleave):
        mni_base = n_block_idx * fx.Int32(BN // 32) + wave * fx.Int32(BN // 128)
        np_list = [mni_base, mni_base + fx.Int32(1)]
    else:
        np_gate = n_block_idx * fx.Int32(BN // 64) + wave
        np_list = [np_gate, np_gate + fx.Int32(N_OUT // 64)]

    def issue_a_load_lds(slot, kt):
        for sub in range_constexpr(kSubBlocks):
            lds_row = wave * fx.Int32(BM // 4) + fx.Int32(sub * 8)
            mask = _lds_swizzle_mask(lds_row + lane_div_8)
            voffset = ((lane_mod_8 * fx.Int32(16)) ^ mask) + cached_actual_row[
                sub
            ] * fx.Int32(K_HALF)
            base_i32 = fx.Int32(
                memref_dialect.extract_aligned_pointer_as_index(s_aq.get())
            )
            off = fx.Int32(slot * (BM * KH_TILE)) + lds_row * fx.Int32(KH_TILE)
            rocdl.raw_ptr_buffer_load_lds(
                aq_rsrc,
                _lds_ptr3(base_i32, off),
                fx.Int32(16),
                voffset,
                fx.Int32(kt * KH_TILE),
                fx.Int32(0),
                fx.Int32(0),
            )

    def issue_a_ds_read(slot):
        # ds_read A from LDS (raw addressing) into the A register fragments for fx.gemm.
        mask = _lds_swizzle_mask(lane_mod_16)
        base_ptr = _lds_base_ptr3(s_aq.get())
        for k in range_constexpr(2):
            lds_col = (lane_div_16 * fx.Int32(16) + fx.Int32(k * 64)) ^ mask
            for i in range_constexpr(kMChunks):
                lds_row = lane_mod_16 + fx.Int32(i * 16)
                off = (
                    fx.Int32(slot * (BM * KH_TILE))
                    + lds_row * fx.Int32(KH_TILE)
                    + lds_col
                )
                vec = llvm.load(T.vec(4, T.i32), _gep3(base_ptr, off))
                _a_frags[i][k].store(Vec(vec))

    def issue_a_scale_load():
        chunk_base = m_row // fx.Int32(32)
        v16 = (wave * fx.Int32(64) + lane) * fx.Int32(16)
        v4 = (wave * fx.Int32(64) + lane) * fx.Int32(4)
        asc_base = fx.Int32(
            memref_dialect.extract_aligned_pointer_as_index(s_asc.get())
        )
        for sub in range_constexpr(kSubBlocks):
            s_chunk = rocdl.readfirstlane(
                T.i32, (chunk_base + fx.Int32(sub)) * fx.Int32(kAS_per_chunk_dw * 4)
            )
            lds_sub = fx.Int32(sub * kAS_per_chunk_dw * 4)
            rocdl.raw_ptr_buffer_load_lds(
                ascale_rsrc,
                _lds_ptr3(asc_base, lds_sub + wave * fx.Int32(1024)),
                fx.Int32(16),
                v16,
                s_chunk,
                fx.Int32(0),
                fx.Int32(0),
            )
            for d in range_constexpr(3):
                byte_off = 4096 + d * 1024
                s_off = rocdl.readfirstlane(T.i32, s_chunk + fx.Int32(byte_off))
                rocdl.raw_ptr_buffer_load_lds(
                    ascale_rsrc,
                    _lds_ptr3(
                        asc_base, lds_sub + fx.Int32(byte_off) + wave * fx.Int32(256)
                    ),
                    fx.Int32(4),
                    v4,
                    s_off,
                    fx.Int32(0),
                    fx.Int32(0),
                )

    def issue_a_scale_ds_read(kt):
        base_ptr = _lds_base_ptr3(s_asc.get())
        out = []
        for sub in range_constexpr(kSubBlocks):
            lds_dw = (
                fx.Int32(sub * kAS_per_chunk_dw)
                + fx.Int32(kt * 64)
                + lane_div_16 * fx.Int32(16)
                + lane_mod_16
            )
            out.append(llvm.load(T.i32, _gep3(base_ptr, lds_dw * fx.Int32(4))))
        return out

    # B load: CK preshuffle as an fx.make_layout view over bq. The descriptor base
    # MUST stay uniform per wave (folding the per-lane part in makes make_buffer_tensor
    # emit a per-lane WATERFALL, ~14x slower), so the base is the uniform col offset
    # and the per-lane (klane,nlane) are layout axes -> a VGPR voffset at copy time.
    # nt/cached rides on the copy atom's cache_modifier (2=nt/0=cached).
    KH4 = K_HALF // 4  # i32 stride for the col axis
    _b_copy_atom = ml.b_copy_atom(b_nontemporal)
    _bs_copy_atom = ml.bscale_copy_atom()

    N0_HALF = N_OUT // 32  # separate-mode gate/up column split

    # B-load view per j-tile (shared layout primitive). interleave / separated only
    # change which logical N-row `col` maps to; the view layout is identical.
    def _make_bq_view_for_jtile(j):
        if const_expr(interleave):
            col = n_block_idx * fx.Int32(BN) + wave * fx.Int32(BN // 4) + fx.Int32(j * 16)
        else:
            tile_il = n_block_idx * fx.Int32(16) + wave * fx.Int32(4) + fx.Int32(j)
            col = ((tile_il & fx.Int32(1)) * fx.Int32(N0_HALF) + (tile_il >> fx.Int32(1))) * fx.Int32(16)
        return ml.bq_view(arg_bq, e * fx.Int32(N_OUT) + col, KH4, K_TILES_TOTAL)

    _bq_views = [_make_bq_view_for_jtile(j) for j in range_constexpr(4)]

    # B-scale view per n-pack word (shared layout primitive).
    _bscale_views = [
        ml.bscale_view(
            arg_bscale,
            e * fx.Int32(kBS_per_expert_dw) + np_list[mw] * fx.Int32(kBS_stride_n0_dw),
            K_TILES_TOTAL,
            k0_stride_dw=kBS_stride_k0_dw,
        )
        for mw in range_constexpr(2)
    ]

    # All MMA operands are register fragments for fx.gemm (rank-1 -> one MmaAtomCall).
    # i32<4:1> (16B = 32 fp4) for A/B; f32<4:1> (vec4) for C. B is PER-STAGE (kStages)
    # to hold the prefetch pipeline; A is refilled per K iter; C accumulates in place.
    _frag_tmpl = ml.bq_frag_tmpl(_bq_views[0])  # i32<4:1>
    _bq_frags = [
        [
            [fx.make_fragment_like(_frag_tmpl) for _ in range_constexpr(2)]
            for _ in range_constexpr(4)
        ]
        for _ in range_constexpr(kStages)
    ]
    _a_frags = [
        [fx.make_fragment_like(_frag_tmpl) for _ in range_constexpr(2)]
        for _ in range_constexpr(kMChunks)
    ]
    _c_frags = [
        [
            fx.make_fragment_like(_frag_tmpl, dtype=fx.Float32.ir_type)
            for _ in range_constexpr(4)
        ]
        for _ in range_constexpr(kMChunks)
    ]
    # B-scale fragments: i32<1:1>, PER-STAGE (kStages) double-buffer like _bq_frags.
    _bs_frag_tmpl = ml.bscale_frag_tmpl(_bscale_views[0])  # i32<1:1>
    _bs_frags = [
        [fx.make_fragment_like(_bs_frag_tmpl) for _ in range_constexpr(2)]
        for _ in range_constexpr(kStages)
    ]

    def issue_b_load_j(stage, K_C, j):
        view = _bq_views[j]
        for half in range_constexpr(2):
            fx.copy(
                _b_copy_atom,
                view[lane_div_16, lane_mod_16, K_C, half, None],
                _bq_frags[stage][j][half],
            )

    def issue_b_scale_load(stage, K_C):
        for mw in range_constexpr(2):
            fx.copy(
                _bs_copy_atom,
                _bscale_views[mw][lane_div_16, lane_mod_16, K_C, None],
                _bs_frags[stage][mw],
            )

    # MMA -- one fx.gemm per mfma over rank-1 fragments (shared layout primitive).
    # Scales ride scale_a=/scale_b=; the (opselA,opselB) atom set is pre-built once.
    # C accumulates in place (d == c).
    zero4 = Vec.filled(4, 0.0, fx.Float32)
    _scale_mma_atoms = ml.scale_mma_atoms()

    def _gemm_mma(a_frag, b_frag, c_frag, opsel_a, opsel_b, sa, sb):
        ml.gemm_mma(_scale_mma_atoms, a_frag, b_frag, c_frag, opsel_a, opsel_b, sa, sb)

    def mfma_cluster(stage, a_scale, J):
        # interleave: mni=J//2 (n0), in_b=J%2 (gate/up); separate: swapped.
        if const_expr(interleave):
            mni, in_b = J // 2, J % 2
        else:
            mni, in_b = J % 2, J // 2
        sb = _raw(Vec(_bs_frags[stage][mni].load())[0])
        bJ0, bJ1 = _bq_frags[stage][J][0], _bq_frags[stage][J][1]
        for sub in range_constexpr(kSubBlocks):
            i0 = sub * 2 + 0
            i1 = sub * 2 + 1
            sa = a_scale[sub]
            _gemm_mma(_a_frags[i0][0], bJ0, _c_frags[i0][J], 0, 0 + in_b, sa, sb)
            _gemm_mma(_a_frags[i1][0], bJ0, _c_frags[i1][J], 1, 0 + in_b, sa, sb)
            _gemm_mma(_a_frags[i0][1], bJ1, _c_frags[i0][J], 2, 2 + in_b, sa, sb)
            _gemm_mma(_a_frags[i1][1], bJ1, _c_frags[i1][J], 3, 2 + in_b, sa, sb)

    # zero C (accumulate in place thereafter)
    for i in range_constexpr(kMChunks):
        for J in range_constexpr(4):
            _c_frags[i][J].store(zero4)

    # prologue: stages 0,1
    issue_a_scale_load()
    for K_C in range_constexpr(kStages):
        issue_a_load_lds(K_C, K_C)
        for j in range_constexpr(4):
            issue_b_load_j(K_C, K_C, j)
        issue_b_scale_load(K_C, K_C)

    # main loop. sched_barrier/s_setprio fence the mfma chain from the B loads (mirror
    # v1's BM!=128 hints) so it stays dense -- closes the small-M gap.
    for OFFSET in range_constexpr(kUnroll):
        K_C = kStages + OFFSET
        read_slot = OFFSET % kAStages
        write_slot = K_C % kAStages
        slot_b = OFFSET % kStages
        gpu.barrier()
        issue_a_ds_read(read_slot)
        asc_cur = issue_a_scale_ds_read(K_C - kStages)
        issue_a_load_lds(write_slot, K_C)
        for J in range_constexpr(4):
            rocdl.sched_barrier(0)
            rocdl.s_setprio(1)
            mfma_cluster(slot_b, asc_cur, J)
            rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
            issue_b_load_j(slot_b, K_C, J)
            rocdl.sched_barrier(0)
        issue_b_scale_load(slot_b, K_C)

    # drain: last kStages
    for S in range_constexpr(kStages):
        kt = K_TILES_TOTAL - kStages + S
        gpu.barrier()
        issue_a_ds_read(kt % kAStages)
        asc_cur = issue_a_scale_ds_read(kt)
        for J in range_constexpr(4):
            mfma_cluster(kt % kStages, asc_cur, J)

    gpu.barrier()
    s_aq._view_cache = None
    s_asc._view_cache = None
    lds_acc._view_cache = None

    # epilog: cshuffle -> SwiGLU -> fp4 + e8m0 requant (raw math)
    wave_n = wave
    lds_acc_base = _lds_base_ptr3(lds_acc.get())

    # Read accumulators back from the C register fragments (flat slot [i,J,v]).
    _acc_vecs = [[Vec(_c_frags[i][J].load()) for J in range(4)] for i in range(kMChunks)]

    def _acc(i, J, v):
        return _acc_vecs[i][J][v]

    for i in range_constexpr(kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for J in range_constexpr(4):
            is_up = (J % 2) == 1
            J_local = J // 2
            col_local = wave_n * fx.Int32(32) + fx.Int32(J_local * 16) + lane_mod_16
            lds_col = (fx.Int32(128) + col_local) if is_up else col_local
            for v in range_constexpr(4):
                idx = (row_base + fx.Int32(v)) * fx.Int32(BN) + lds_col
                llvm.StoreOp(
                    _raw(fx.Float32(_acc(i, J, v))),
                    _gep3(lds_acc_base, idx * fx.Int32(4)),
                )

    gpu.barrier()

    tx_i32 = arith.index_cast(T.i32, gpu.thread_id("x"))
    m_lane = tx_i32 // fx.Int32(16)
    n_lane = tx_i32 % fx.Int32(16)
    wave_grp = n_lane // fx.Int32(4)
    kk = n_lane % fx.Int32(4)

    aqout_base = _global_base_ptr1(arg_aqout)
    scales_per_mr = [None] * M_REPS

    for mr in range_constexpr(M_REPS):
        row_local = fx.Int32(mr * 16) + m_lane
        gate_vs = [None] * 8
        up_vs = [None] * 8
        for ee in range_constexpr(8):
            col_in_grp = fx.Int32(8) * kk + fx.Int32(ee)
            gate_col = wave_grp * fx.Int32(32) + col_in_grp
            up_col = fx.Int32(128) + gate_col
            gate_off = (row_local * fx.Int32(BN) + gate_col) * fx.Int32(4)
            up_off = (row_local * fx.Int32(BN) + up_col) * fx.Int32(4)
            gate_vs[ee] = fx.Float32(llvm.load(T.f32, _gep3(lds_acc_base, gate_off)))
            up_vs[ee] = fx.Float32(llvm.load(T.f32, _gep3(lds_acc_base, up_off)))
        result = _silu_mul_batch(gate_vs, up_vs)

        local_max = _fabs_f32(result[0])
        for ee in range_constexpr(1, 8):
            local_max = local_max.maximumf(_fabs_f32(result[ee]))
        local_max = local_max.maximumf(local_max.shuffle_xor(fx.Int32(1), fx.Int32(64)))
        local_max = local_max.maximumf(local_max.shuffle_xor(fx.Int32(2), fx.Int32(64)))

        e8m0, qscale = _e8m0_from_amax(local_max)
        scales_per_mr[mr] = e8m0

        packed_i32 = _raw(fx.Int32(0))
        qscale_raw = _raw(qscale)
        for w in range_constexpr(4):
            packed_i32 = rocdl.cvt_scalef32_pk_fp4_f32(
                T.i32,
                packed_i32,
                _raw(result[2 * w]),
                _raw(result[2 * w + 1]),
                qscale_raw,
                w,
            )
        packed = fx.Int32(packed_i32)

        byte_pos = (
            n_block_idx * fx.Int32(BN_INT // 2)
            + wave_grp * fx.Int32(16)
            + kk * fx.Int32(4)
        )
        out_row = m_row + row_local
        store_off = out_row * fx.Int32(K_G2_HALF) + byte_pos
        llvm.StoreOp(
            _raw(packed),
            _gep1(aqout_base, store_off),
            alignment=4,
            nontemporal=True,
        )

    ascaleout_base = _global_base_ptr1(arg_ascaleout)
    if kk == fx.Int32(0):
        ku = n_block_idx >> fx.Int32(1)
        ikxdl = n_block_idx & fx.Int32(1)
        for sub in range_constexpr(kSubBlocks):
            chunk = m_block_idx * fx.Int32(kSubBlocks) + fx.Int32(sub)
            dword_off = (
                chunk * fx.Int32(OUT_AS_PER_CHUNK_DW)
                + ku * fx.Int32(64)
                + wave_grp * fx.Int32(16)
                + m_lane
            )
            pair_i32 = scales_per_mr[sub * 2 + 0] | (
                scales_per_mr[sub * 2 + 1] << fx.Int32(8)
            )
            pair_i16 = arith.TruncIOp(T.i16, _raw(pair_i32)).result
            addr = dword_off * fx.Int32(4) + ikxdl * fx.Int32(2)
            llvm.StoreOp(
                pair_i16,
                _gep1(ascaleout_base, addr),
                alignment=2,
            )


def _lds_bytes_for(K_TILES_TOTAL):
    s_aq_bytes = kAStages * BM * KH_TILE
    s_asc_bytes = kSubBlocks * K_TILES_TOTAL * 256
    lds_acc_bytes = BM * BN * 4
    return max(s_aq_bytes + s_asc_bytes, lds_acc_bytes)


def compile_gemm1_a4w4_port(
    BM=32,
    use_nt=True,
    inline_quant=False,
    D_HIDDEN=K_DEFAULT,
    D_INTER=INTER_DEFAULT,
    NE=NE_DEFAULT,
    TOPK=TOPK_DEFAULT,
    interleave=True,
):
    # use_nt IS the B-load cache policy (v1's `b_aux = 2 if use_nt else 0`;
    # tuned config BM32_NT vs BM32_CACHED): True -> nt (decode), False -> cached.
    b_nontemporal = use_nt
    if (BM, inline_quant) != (32, False):
        raise AssertionError(
            f"mxfp4_gemm1_v2 supports only (BM=32, inline_quant=False); "
            f"got (BM={BM}, inline_quant={inline_quant})"
        )

    _K, _INTER, _NE = D_HIDDEN, D_INTER, NE
    assert _K % BK == 0, f"D_HIDDEN (K) must be a multiple of {BK}, got {_K}"
    _N_OUT = 2 * _INTER
    assert _N_OUT % BN == 0, f"2*D_INTER (N_OUT) must be a multiple of {BN}, got {_N_OUT}"

    lds_bytes = _lds_bytes_for(_K // BK)  # K_TILES_TOTAL

    gu_tag = "il" if interleave else "sep"
    bnt_tag = "nt" if b_nontemporal else "cached"
    name_suffix = f"h{_K}_i{_INTER}_ne{_NE}_bm{BM}_{bnt_tag}_{gu_tag}_v2"

    allocator = SmemAllocator(
        None, arch="gfx950", global_sym_name=f"gemm1port_v2_smem_{name_suffix}"
    )
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + lds_bytes

    @flyc.kernel(name=f"gemm1_a4w4_port_{name_suffix}", known_block_size=[256, 1, 1])
    def gemm1_kernel(
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_mind: fx.Int64,
        i32_ntok: fx.Int32,
        arg_aqout: fx.Int64,
        arg_ascaleout: fx.Int64,
        arg_hidden: fx.Int64,
    ):
        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        tx_i32 = arith.index_cast(T.i32, tx)
        bx_i32 = arith.index_cast(T.i32, bx)
        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))
        cumsum0 = llvm.load(T.i32, _global_ptr1(arg_cumsum, fx.Int32(0)))
        total_m_blocks = cumsum0 // fx.Int32(BM)
        bound = total_m_blocks * fx.Int32(_N_OUT // 256)  # * NUM_N_BLOCKS
        if fx.Int32(bx_i32) < bound:
            _gemm1_body_v2(
                allocator,
                lds_off,
                arg_aq,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_mind,
                arg_aqout,
                arg_ascaleout,
                bx_i32,
                lane,
                wave,
                i32_ntok,
                total_m_blocks,
                K=_K,
                INTER=_INTER,
                NE=_NE,
                interleave=interleave,
                b_nontemporal=b_nontemporal,
            )

    @flyc.jit
    def launch_gemm1(
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_mind: fx.Int64,
        i32_ntok: fx.Int32,
        i32_grid: fx.Int32,
        arg_aqout: fx.Int64,
        arg_ascaleout: fx.Int64,
        arg_hidden: fx.Int64,
        stream: fx.Stream,
    ):
        from flydsl.compiler.kernel_function import CompilationContext

        ctx = CompilationContext.get_current()
        allocator.finalized = False
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        grid_x = arith.index_cast(T.index, i32_grid)
        gemm1_kernel(
            arg_aq,
            arg_ascale,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_cumsum,
            arg_mind,
            i32_ntok,
            arg_aqout,
            arg_ascaleout,
            arg_hidden,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm1
