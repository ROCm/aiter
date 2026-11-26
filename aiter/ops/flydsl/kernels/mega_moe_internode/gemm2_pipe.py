# SPDX-License-Identifier: MIT
"""GEMM2 core with a compile-time unrolled K loop (fp8 or fp4 A, fp4 B)."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Float32, Int32, T
from flydsl.expr.typing import Vector as Vec

from ..mxfp4_gemm_common import (
    _buffer_rsrc,
    _lds_ptr3,
    _lds_swizzle_mask,
    global_typed_ptr,
    kBS_stride_k0_dw,
    lds_swizzle_mask_f8,
)
from ..tensor_shim import buf_copy_load, ptr_buf_tensor
from .gemm import mma_one_j, scale_mma_atoms, scale_view


@flyc.jit
def gemm2_compute_pipe(
    lds_base_i32,
    arg_ascale,
    arg_bq,
    arg_bscale,
    arg_eids,
    arg_aq,
    i32_max_m_blocks,
    bx_i32,
    lane,
    wave,
    i32_inter,
    i32_hidden,
    *,
    BM,
    BN=256,
    BK=256,
    INTER_MAX,
    aStages,
    explicit_a_row=None,
    eids_stride=1,
    NW=4,
    a_dtype="fp8",
    b_stages=3,
):
    """gemm2_compute_v2 with the K loop unrolled at compile time (K = INTER_MAX).

    B and its scales rotate through ``b_stages`` register stages without
    loop-carried copies, so a K step waits only for its own stage instead of
    vmcnt(0). A keeps an LDS DMA ring of aStages slots (aStages-1 steps ahead) but issues the
    DMA and the ds_reads with per-slot alias scopes and a bare s_barrier, so LLVM
    does not wait for every outstanding LDS DMA before reading one slot. The
    caller must not issue A loads itself.
    """
    assert a_dtype in ("fp8", "fp4") and BK == 256 and BM >= 32 and NW in (4, 8)
    assert aStages >= 2 and b_stages >= 2
    is_f8_a = a_dtype == "fp8"
    a_pack = 1 if is_f8_a else 2
    A_AHEAD = aStages - 1  # A DMA runs this many K steps ahead
    wave_c = wave % fx.Int32(4) if NW == 8 else wave
    KH_TILE_A = BK // a_pack  # A bytes per row and K tile: 256 (fp8) / 128 (fp4)
    lanes_per_row = KH_TILE_A // 16  # one LDS DMA moves 64 lanes x 16 B
    rows_per_call = 64 // lanes_per_row  # 4 (fp8) / 8 (fp4) rows per DMA
    # With more waves than row groups (fp4 BM32: 4 groups, 8 waves) wave w and
    # w + A_NW load the same rows into the same LDS bytes: the duplicate DMA carries
    # identical data, and every wave waits for its own DMA before the step barrier.
    # (Guarding the DMA with a runtime "if wave < A_NW" here gave wrong results.)
    A_NW = min(NW, BM // rows_per_call)
    wave_a = (
        wave % fx.Int32(A_NW) if A_NW < NW else wave
    )  # this wave's A row group owner
    rows_per_wave = BM // A_NW
    assert rows_per_wave % rows_per_call == 0
    n_row_groups = rows_per_wave // rows_per_call
    kMChunks = BM // 16
    kHalves = BK // 128
    numAccN = (BN // 4) // 16
    nPairs = max(1, numAccN // 2)
    kScaleSubBlocks = kMChunks // 2
    slot_bytes = BM * KH_TILE_A
    K_TILES = INTER_MAX // BK
    K_rt = fx.Int32(i32_inter)
    K_BYTES = K_rt // fx.Int32(a_pack)  # A row stride bytes
    kc_rt = K_rt // 256
    kAS_per_chunk_dw = kc_rt * 64
    kBS_stride_n0_dw = kc_rt * 64
    N_OUT_rt = fx.Int32(i32_hidden)
    kbs_per_expert_dw = (N_OUT_rt // 32) * kBS_stride_n0_dw
    num_n_blocks = N_OUT_rt // fx.Int32(BN)
    KH4 = K_rt // 8
    K_SCALE_CHUNKS_MAX = INTER_MAX // 256

    m_block_idx = bx_i32 // num_n_blocks
    n_block_idx = bx_i32 - m_block_idx * num_n_blocks
    eids_ptr = global_typed_ptr(arg_eids, T.i32)
    e_idx = m_block_idx if eids_stride == 1 else m_block_idx * fx.Int32(eids_stride)
    e = rocdl.readfirstlane(T.i32, eids_ptr[e_idx])
    m_row = m_block_idx * BM
    a_row = m_row if explicit_a_row is None else fx.Int32(explicit_a_row)
    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16
    mma_atoms = scale_mma_atoms(a_dtype)

    _dom = '#llvm.alias_scope_domain<id = distinct[0]<>, description = "gmm2_lds">'
    _sc = _ir.ArrayAttr(
        _ir.Attribute.parse(
            "["
            + ", ".join(
                f"#llvm.alias_scope<id = distinct[{i + 1}]<>, domain = {_dom}>"
                for i in range(aStages)
            )
            + "]"
        )
    )
    _scopes = [_sc[i] for i in range(aStages)]

    def _scope_attrs(idx):
        return {
            "alias_scopes": _ir.ArrayAttr.get([_scopes[idx]]),
            "noalias_scopes": _ir.ArrayAttr.get(
                [_scopes[j] for j in range(aStages) if j != idx]
            ),
        }

    def _swz(row):
        return (
            lds_swizzle_mask_f8(row, KH_TILE_A) if is_f8_a else _lds_swizzle_mask(row)
        )

    # A: one buffer descriptor per unit; one LDS DMA moves 64 lanes x 16 B.
    a_rsrc = _buffer_rsrc(
        fx.Int64(arg_aq) + fx.Int64(a_row) * fx.Int64(K_BYTES),
        fx.Int64(BM) * fx.Int64(K_BYTES),
    )
    a_lane_row = lane // fx.Int32(lanes_per_row)
    a_lane_col = (lane % fx.Int32(lanes_per_row)) * fx.Int32(16)
    a_voff = []
    for g in range_constexpr(n_row_groups):
        tile_row = (
            wave_a * fx.Int32(rows_per_wave) + fx.Int32(g * rows_per_call) + a_lane_row
        )
        a_voff.append((a_lane_col ^ _swz(tile_row)) + tile_row * K_BYTES)

    def issue_a(kt):
        slot = kt % aStages
        for g in range_constexpr(n_row_groups):
            lds_off = fx.Int32(slot * slot_bytes) + (
                wave_a * fx.Int32(rows_per_wave) + fx.Int32(g * rows_per_call)
            ) * fx.Int32(KH_TILE_A)
            rocdl.raw_ptr_buffer_load_lds(
                a_rsrc,
                _lds_ptr3(lds_base_i32, lds_off),
                fx.Int32(16),
                a_voff[g],
                fx.Int32(kt * KH_TILE_A),
                fx.Int32(0),
                fx.Int32(0),
                **_scope_attrs(slot),
            )

    a_mask = _swz(lane_mod_16)
    A_NDW = 8 if is_f8_a else 4  # dwords of one 16x128-K A fragment per lane
    a_frags = [
        [fx.make_rmem_tensor(A_NDW, Int32) for _ in range_constexpr(kHalves)]
        for _ in range_constexpr(kMChunks)
    ]

    def read_a(kt):
        slot = kt % aStages
        for k in range_constexpr(kHalves):
            col0 = lane_div_16 * fx.Int32(16) + fx.Int32(k * (128 if is_f8_a else 64))
            for i in range_constexpr(kMChunks):
                row_off = fx.Int32(slot * slot_bytes) + (
                    lane_mod_16 + fx.Int32(i * 16)
                ) * fx.Int32(KH_TILE_A)
                lo = Vec(
                    llvm.LoadOp(
                        Vec.make_type(4, fx.Int32),
                        _lds_ptr3(lds_base_i32, row_off + (col0 ^ a_mask)),
                        alignment=16,
                        **_scope_attrs(slot),
                    ).result
                )
                if const_expr(not is_f8_a):
                    a_frags[i][k].store(lo)
                else:
                    hi = Vec(
                        llvm.LoadOp(
                            Vec.make_type(4, fx.Int32),
                            _lds_ptr3(
                                lds_base_i32, row_off + ((col0 + fx.Int32(64)) ^ a_mask)
                            ),
                            alignment=16,
                            **_scope_attrs(slot),
                        ).result
                    )
                    a_frags[i][k].store(
                        Vec.from_elements(
                            [lo[j] for j in range_constexpr(4)]
                            + [hi[j] for j in range_constexpr(4)],
                            fx.Int32,
                        )
                    )

    sc_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)
    asc_per_mb = fx.Int32(kScaleSubBlocks) * kAS_per_chunk_dw * 4
    asc_num = fx.Int64(i32_max_m_blocks) * fx.Int64(asc_per_mb)
    scale_chunk0 = a_row // 32
    ascale_views = []
    for sub in range_constexpr(kScaleSubBlocks):
        base_dw = (scale_chunk0 + fx.Int32(sub)) * kAS_per_chunk_dw
        ascale_views.append(
            scale_view(
                arg_ascale,
                base_dw,
                K_SCALE_CHUNKS_MAX,
                k0_stride_dw=64,
                num_records_bytes=asc_num - fx.Int64(base_dw) * 4,
            )
        )
    sc_frag_tmpl = ascale_views[0][0, 0, 0, None]

    # B: the per-lane offset in 16-byte units is base + lane (lane_div_16*16 + lane_mod_16).
    bq_buffer = ptr_buf_tensor(arg_bq, fx.Int32, unit_elems=4)
    bq_base16 = [
        rocdl.readfirstlane(
            T.i32,
            ((e * N_OUT_rt + n_block_idx * BN + wave_c * (BN // 4) + j * 16) * KH4)
            // fx.Int32(4),
        )
        for j in range_constexpr(numAccN)
    ]
    mni_base = n_block_idx * (BN // 16 // 2) + wave_c * (BN // 64 // 2)
    bscale_views = [
        scale_view(
            arg_bscale,
            e * kbs_per_expert_dw + (mni_base + mw) * kBS_stride_n0_dw,
            K_SCALE_CHUNKS_MAX,
            k0_stride_dw=kBS_stride_k0_dw,
        )
        for mw in range_constexpr(nPairs)
    ]
    frag_tmpl = fx.make_rmem_tensor(4, Int32)
    st_bq = [
        [
            [fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(kHalves)]
            for _ in range_constexpr(numAccN)
        ]
        for _ in range_constexpr(b_stages)
    ]
    st_bs = [
        [fx.make_fragment_like(sc_frag_tmpl) for _ in range_constexpr(nPairs)]
        for _ in range_constexpr(b_stages)
    ]
    st_sa = [
        [fx.make_fragment_like(sc_frag_tmpl) for _ in range_constexpr(kScaleSubBlocks)]
        for _ in range_constexpr(b_stages)
    ]

    def issue_b(kt):
        s = kt % b_stages
        for j in range_constexpr(numAccN):
            for half in range_constexpr(kHalves):
                st_bq[s][j][half].store(
                    Vec(
                        buf_copy_load(
                            bq_buffer,
                            bq_base16[j]
                            + lane
                            + fx.Int32(kt * kHalves * 64 + half * 64),
                            fx.Int32,
                            unit_elems=4,
                            cache_modifier=0,
                        )
                    )
                )
        for mw in range_constexpr(nPairs):
            fx.copy(
                sc_copy_atom,
                bscale_views[mw][lane_div_16, lane_mod_16, kt, None],
                st_bs[s][mw],
            )
        for sub in range_constexpr(kScaleSubBlocks):
            fx.copy(
                sc_copy_atom,
                ascale_views[sub][lane_div_16, lane_mod_16, kt, None],
                st_sa[s][sub],
            )

    zero4 = fx.full(4, 0.0, Float32)
    c_frags = [
        [fx.make_rmem_tensor(4, Float32) for _ in range_constexpr(numAccN)]
        for _ in range_constexpr(kMChunks)
    ]
    for i in range_constexpr(kMChunks):
        for J in range_constexpr(numAccN):
            c_frags[i][J].store(zero4)

    def emit_a(kt):
        if const_expr(kt < K_TILES):
            issue_a(kt)

    def emit_b(kt):
        if const_expr(kt < K_TILES):
            issue_b(kt)

    for kt in range_constexpr(A_AHEAD):
        emit_a(kt)
    for kt in range_constexpr(b_stages - 1):
        emit_b(kt)
    for kt in range_constexpr(K_TILES):
        # A scoped volatile probe of slot kt makes LLVM wait exactly for this
        # wave's DMA into that slot (the compiler reorders the B loads, so a
        # hand-counted vmcnt is not reliable). lgkmcnt(0) also retires every
        # read of the slot refilled below; the barrier publishes A(kt).
        llvm.LoadOp(
            T.i32,
            _lds_ptr3(lds_base_i32, fx.Int32((kt % aStages) * slot_bytes)),
            alignment=4,
            volatile_=True,
            **_scope_attrs(kt % aStages),
        )
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)
        emit_a(kt + A_AHEAD)
        emit_b(kt + b_stages - 1)
        rocdl.sched_barrier(0)
        read_a(kt)
        s = kt % b_stages
        sa = [Vec(st_sa[s][sub].load())[0] for sub in range_constexpr(kScaleSubBlocks)]
        sbw = [Vec(st_bs[s][mni].load())[0] for mni in range_constexpr(nPairs)]
        rocdl.s_setprio(1)
        order = [(J, sub) for J in range(numAccN) for sub in range(kScaleSubBlocks)]
        for oi in range_constexpr(len(order)):
            J, sub = order[oi]
            mma_one_j(
                J,
                J % 2,
                sa[sub],
                sbw[J // 2],
                st_bq[s],
                a_frags,
                c_frags,
                mma_atoms,
                i0=2 * sub,
                k_halves=kHalves,
            )
        rocdl.s_setprio(0)
        rocdl.sched_barrier(0)
    # The epilogue reuses the A ring as its accumulator slab.
    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
    rocdl.s_barrier()
    accm_vecs = [
        [c_frags[i][J].load() for J in range(numAccN)] for i in range(kMChunks)
    ]
    return accm_vecs, m_row, n_block_idx, N_OUT_rt
