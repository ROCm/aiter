# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MegaMoEV2 GEMM2 on an 8-wave 256x256 A8W4 main loop, for the SDMA Stage2.

``out = act(A) @ w2[expert].T`` for the Stage1 output rows, weighted by the routing
weight and quantised to fp8 1x32, then written to the local send buffer ``S`` at
the plan kernel's ``out_row``; workgroup 0 ships finished chunks of ``S`` to the
peers with SDMA (see ``stage2_sdma``).

Inputs are MegaMoEV2's Stage2 inputs as they are:

* A    Stage1 output, fp8 ``[rows, K]`` in sorted-row (expert-major) order.
* A-s  Stage1 output scale ``_s1_osd``: ``shuffle_scale`` layout over rows padded to 256.
* B    w2, ``shuffle_weight_a16w4(w2, 16, False)``: per 16-row group and 128-K step a
       contiguous 1 KB block ``[k_lane 4][n 16][16 B]``, copied to LDS as is and read
       conflict-free -- lane (g, n) at ``g*256 + n*16``, 32 contiguous K of fp4.
* B-s  w2 scale, ``shuffle_scale_a16w4``: the same super-row layout as A-s.

Main loop: the 8-wave mxfp8 pipeline of ``mxfp8_gemm_8wave`` (2 x 4 waves, BK=128,
A and B through LDS with cur/next double buffers), grouped by expert and with an
fp4 B operand. A 256-row tile never straddles experts: each expert's SBM-padded
segment is split into 256-row tiles, found in-kernel from ``expert_tile_end``.

Epilogue: a wave's 32 columns of an N-half are exactly one fp8 1x32 group, so rows
are weighted and quantised in registers (group max over 16 lanes x 2 tiles) -- the
bytes the default Stage2 writes. fp8 rows and e8m0 go to a dedicated LDS stage,
then two rows per store (lanes 0-31 / 32-63) are written through (sc0|sc1) to
``S``; a row the plan marked ``OUT_ROW_P2P`` goes to the origin rank's
``comb_inp[slot]`` instead. Each finished work item counts once per sort tile
toward its chunk's completion counter.

Persistent: one workgroup per CU; workgroup 0 is the comm workgroup, the others
walk the (tile, N-block) work items.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Float4E2M1FN, T
from flydsl.expr.typing import Vector as Vec

from .. import communication_ops_utils as comm_ops
from ..buffer_ops import buffer_load, buffer_store, create_buffer_resource_from_addr
from ..gemm_a8w8_8wave import (
    G2SLoader,
    S2RLoader,
    compute_global_swizzle,
    make_fp8_buffer_tensor,
    wait_barrier,
)
from ..kernels_common import set_slp_vectorize_once_used
from ..mxfp4_gemm_common import _fabs_f32 as fabs_f32
from ..mxfp4_gemm_common import global_typed_ptr, lds_typed_ptr, lds_vec_load
from ..tensor_shim import ptr_buf_tensor
from .dispatch import _wave_inclusive_scan_i32, _wave_reduce_max_i32
from .stage2_sdma import (
    CDONE_STRIDE,
    INV_P2P,
    OUT_ROW_P2P,
    PUSH_U,
    publish_flag,
    sdma_put_run,
    sdma_signal_peer,
    sdma_wait_consumed,
)

__all__ = ["compile_gemm2_8wave_sdma"]

BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 128
LDS_LIMIT_BYTES = 160 * 1024
#: Send-buffer stores: sc0|sc1, write-through to memory, so retiring them is the
#: release the copy engine needs (a system-scope release per item would instead
#: write back the whole L2).
SEND_CM = 1 | 16


class ScalePreshuffledS2R:
    """Coalesced reader for ``shuffle_scale``-packed E8M0 -- no LDS staging."""

    def __init__(self, scale_arg, rows, K, n_tiles, lane):
        assert n_tiles % 2 == 0, "the shuffled scale layout pairs tiles two at a time"
        self.n_pairs = n_tiles // 2
        self.k1_stride = K // 256  # i32 groups of 64 per 32-row super-row
        self.lane = lane
        t_i8 = fx.rocdl.make_buffer_tensor(
            scale_arg,
            max_size=False,
            num_records_bytes=fx.Int64(rows) * fx.Int64(K // 32),
        )
        i32_ptr = fx.PointerType.get(
            elem_ty=fx.Int32.ir_type,
            address_space=rocdl.TargetAddressSpace.BufferDesc,
            alignment=4,
        )
        iter_i32 = fx.recast_iter(i32_ptr, fx.get_iter(t_i8))
        n_i32 = fx.Int32(rows) * fx.Int32(K // 128)
        self.g_div = fx.logical_divide(
            fx.Tensor(fx.make_view(iter_i32, fx.make_layout(n_i32, 1))),
            fx.make_layout(1, 1),
        )
        self.atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        # Two register sets: the next K-pair is prefetched while the current one
        # still feeds MFMAs.
        self.regs = [
            [
                fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
                for _ in range_constexpr(self.n_pairs)
            ]
            for _ in range_constexpr(2)
        ]

    def read(self, row_base16, k):
        """``n_tiles`` scale operands for K-step ``k``; tiles of a pair share one word."""
        k1 = k // 2  # compile-time; k % 2 is the k_pack the opsel encodes
        regs = self.regs[k1 % 2]
        words = []
        for p in range_constexpr(self.n_pairs):
            # row_base16 is even (wave offsets are multiples of 32 rows), so the
            # 32-row super-row is row_base16 // 2 + p and the tile parity picks the byte.
            n1 = row_base16 // 2 + p
            base = rocdl.readfirstlane(
                fx.Int32.ir_type, (n1 * self.k1_stride + k1) * 64
            )
            fx.copy(
                self.atom,
                fx.slice(self.g_div, (None, fx.Int32(base) + self.lane)),
                regs[p],
            )
            w = fx.Int32(regs[p].load()[0])
            words += [w, w]
        return words


class S2RF4G:
    """fp4 B operands from the ``shuffle_weight_a16w4`` layout staged 1:1 in LDS.

    LDS holds 1 KB per 16-row group, ``[k_lane 4][n 16][16 B]``; lane (g, n) of tile
    ``i`` reads its 16 bytes (32 contiguous K, the fp4 MFMA operand) conflict-free.
    """

    def __init__(self, wave_idx, n_tiles, lane):
        self.lane_id = lane
        self.wave_idx = wave_idx
        self.n_tiles = n_tiles

    def load(self, lds_src):
        frag = []
        for i in range_constexpr(self.n_tiles):
            group = self.wave_idx * self.n_tiles + i
            offset = (
                group * 1024 + (self.lane_id // 16) * 256 + (self.lane_id % 16) * 16
            )
            ptr_off = fx.add_offset(lds_src.ptr, fx.make_int_tuple(offset))
            view = fx.make_view(
                fx.recast_iter(fx.Uint8, ptr_off), fx.make_layout(16, 1)
            )
            frag.append(view.load().bitcast(fx.Int32))
        return frag


class MxMfma:
    """n_tiles_a x n_tiles_b scaled 16x16x128 fp8 x fp4 MFMAs; a scale word's byte is
    ``k_pack * 2 + tile parity``."""

    def __init__(self, n_tiles_a, n_tiles_b):
        self.atoms = {
            (osa, osb): fx.make_mma_atom(
                fx.rocdl.cdna4.MFMA_Scale(
                    16, 16, 128, fx.Float8E4M3FN, Float4E2M1FN, opsel_a=osa, opsel_b=osb
                )
            )
            for osa in range(4)
            for osb in range(4)
        }
        self.zero_value = fx.full(4, 0.0, fx.Float32)
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b

    def idx(self, i, j):
        return i * self.n_tiles_b + j

    def _frag(self, value, n, dtype):
        frag = fx.make_rmem_tensor(n, dtype)
        frag.store(Vec(value))
        return frag

    def call(self, a, b, c, sa, sb, *, k_pack, set_prio=True):
        a_frags = [self._frag(v, 8, fx.Int32) for v in a]
        b_frags = [self._frag(v, 4, fx.Int32) for v in b]
        c_frags = [self._frag(v, 4, fx.Float32) for v in c]
        if const_expr(set_prio):
            rocdl.s_setprio(1)
        for i in range_constexpr(self.n_tiles_a):
            for j in range_constexpr(self.n_tiles_b):
                cf = c_frags[self.idx(i, j)]
                fx.gemm(
                    self.atoms[(k_pack * 2 + i % 2, k_pack * 2 + j % 2)],
                    cf,
                    a_frags[i],
                    b_frags[j],
                    cf,
                    scale_a=sa[i],
                    scale_b=sb[j],
                )
        if const_expr(set_prio):
            rocdl.s_setprio(0)
            rocdl.s_barrier()
        return [cf.load().ir_value() for cf in c_frags]


# --- 256-row tiles of the expert segments --------------------------------------------


def tile_table(tile_end, lane_id, *, SBM, experts):
    """Per-lane 256-row tile counts of the local experts (lane e: expert e), from
    Prepare's cumulative SBM-tile ends. Returns the per-lane values and the tile total.
    """
    ends = global_typed_ptr(tile_end, T.i32)
    ok = lane_id < fx.Int32(experts)
    li = ok.select(lane_id, fx.Int32(0))
    end_l = ends[li]
    prev_l = (ok & (lane_id > fx.Int32(0))).select(
        ends[(li > fx.Int32(0)).select(li - fx.Int32(1), fx.Int32(0))], fx.Int32(0)
    )
    seg_l = ok.select((end_l - prev_l) * fx.Int32(SBM), fx.Int32(0))
    tiles_l = (seg_l + fx.Int32(255)) >> fx.Int32(8)
    cum_l = _wave_inclusive_scan_i32(tiles_l, lane_id)
    total = fx.Int32(
        rocdl.readfirstlane(
            T.i32, fx.Int32(rocdl.ds_bpermute(T.i32, fx.Int32(63 * 4), cum_l))
        )
    )
    return (prev_l, seg_l, tiles_l, cum_l), total


def tile_table_to_lds(values, tab_lds, lane_id):
    """Park the per-lane table in LDS (every wave writes the same values), so the
    persistent loop does not keep it in VGPRs."""
    p = lds_typed_ptr(tab_lds, T.i32)
    for i, v in enumerate(values):
        p[fx.Int32(i * 64) + lane_id] = v


def tile_find(tab_lds, j, lane_id, *, SBM, experts):
    """(row_base, expert, valid_rows) of 256-row tile ``j`` -- LDS and wave ops only."""
    p = lds_typed_ptr(tab_lds, T.i32)
    prev_l, seg_l, tiles_l, cum_l = [
        fx.Int32(p[fx.Int32(i * 64) + lane_id]) for i in range(4)
    ]
    ok = lane_id < fx.Int32(experts)
    before = _wave_reduce_max_i32(
        (ok & (cum_l <= j)).select(lane_id + fx.Int32(1), fx.Int32(0)), lane_id
    )
    e = fx.Int32(rocdl.readfirstlane(T.i32, before))
    e = (e < fx.Int32(experts)).select(e, fx.Int32(experts - 1))

    def from_lane(v):
        return fx.Int32(
            rocdl.readfirstlane(
                T.i32, fx.Int32(rocdl.ds_bpermute(T.i32, e * fx.Int32(4), v))
            )
        )

    t_in = j - from_lane(cum_l - tiles_l)
    row_base = from_lane(prev_l * fx.Int32(SBM)) + t_in * fx.Int32(256)
    left = from_lane(seg_l) - t_in * fx.Int32(256)
    valid = (left > fx.Int32(256)).select(
        fx.Int32(256), (left > fx.Int32(0)).select(left, fx.Int32(0))
    )
    return row_base, e, valid


# --- Small helpers -------------------------------------------------------------------


def opaque_i32(v):
    """``v`` through an identity the compiler cannot see through.

    In the persistent loop, lane-derived addresses would otherwise be hoisted out of
    it as loop invariants and spilled (the main loop leaves no VGPRs for them), and a
    scratch reload counts in vmcnt like the epilogue's stores.
    """
    return fx.Int32(
        _llvm.inline_asm(
            res=T.i32,
            operands_=[fx.Int32(v).ir_value()],
            asm_string="v_mov_b32 $0, $1",
            constraints="=v,v",
            has_side_effects=True,
        )
    )


def lds_barrier():
    """Workgroup barrier for LDS only: unlike ``fx.barrier`` it never waits on vmcnt."""
    _llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="s_waitcnt lgkmcnt(0)\ns_barrier",
        constraints="",
        has_side_effects=True,
    )


def e8m0_of(amax):
    """The fp8 1x32 block exponent of a group's absolute max (the default Stage2's rule)."""
    working_bits = (amax * fx.Int32(0x3B124925).bitcast(fx.Float32)).bitcast(fx.Int32)
    mantissa = working_bits & fx.Int32(0x7FFFFF)
    biased_exp = (working_bits >> fx.Int32(23)) & fx.Int32(0xFF)
    e8m0 = (mantissa != fx.Int32(0)).select(biased_exp + fx.Int32(1), biased_exp)
    e8m0 = (e8m0 > fx.Int32(0xFF)).select(fx.Int32(0xFF), e8m0)
    return (amax.bitcast(fx.Int32) == fx.Int32(0)).select(fx.Int32(0), e8m0)


def epi_quant_half(
    frags_n0,
    frags_n1,
    *,
    h_m,
    wave_m,
    wave_n,
    lane_id,
    mfma_idx,
    n_tiles_a,
    wts_lds,
    stage_lds,
    scale_lds,
):
    """Weight and fp8 1x32-quantise one 128-row half in registers; bytes and e8m0 to LDS.

    A wave's 32 columns of an N-half are exactly one quant group: per row the group
    max spans the 16 lanes of a lane-quarter and its two 16-column tiles.
    """
    wts_p = lds_typed_ptr(wts_lds, T.f32)
    st_p = lds_typed_ptr(stage_lds, T.i8, align=1)
    sc_p = lds_typed_ptr(scale_lds, T.i8, align=1)
    pk_ty = T.vec(2, T.i16)
    for ti in range_constexpr(n_tiles_a):
        for i in range_constexpr(4):
            r = (
                wave_m * (n_tiles_a * 16) + ti * 16 + (lane_id // 16) * 4 + i
            )  # row in the half
            w = fx.Float32(wts_p[fx.Int32(h_m * 128) + r])
            for h_n in range_constexpr(2):
                frags = frags_n1 if h_n else frags_n0
                v0 = Vec(frags[mfma_idx(ti, 0)])[i] * w
                v1 = Vec(frags[mfma_idx(ti, 1)])[i] * w
                amax = fabs_f32(v0).maximumf(fabs_f32(v1))
                for sh in (1, 2, 4, 8):
                    amax = amax.maximumf(amax.shuffle_xor(fx.Int32(sh), fx.Int32(64)))
                e8m0 = e8m0_of(amax)
                block_scale = (e8m0 << 23).bitcast(fx.Float32)
                pkw = rocdl.cvt_scalef32_pk_fp8_f32(
                    pk_ty,
                    fx.full(2, 0, fx.Int16).ir_value(),
                    v0.ir_value(),
                    v1.ir_value(),
                    block_scale.ir_value(),
                    0,
                )
                word = fx.Vector(pkw).bitcast(fx.Int32)[0]
                col = h_n * 128 + wave_n * 32 + lane_id % 16
                st_p[r * 256 + col] = (word & fx.Int32(0xFF)).to(fx.Int8)
                st_p[r * 256 + col + 16] = ((word >> fx.Int32(8)) & fx.Int32(0xFF)).to(
                    fx.Int8
                )
                sc_p[r * 8 + h_n * 4 + wave_n] = e8m0.to(fx.Int8)


# --- The kernel ------------------------------------------------------------------------


def compile_gemm2_8wave_sdma(
    *,
    K: int,
    N: int,
    experts: int,
    topk: int,
    npes: int,
    max_tok: int,
    recv_cap: int,
    SBM: int,
    rank: int,
    layout,
    chunk_tiles: int,
    grid_ctas: int,
    comb_bytes: int,
    waves_per_eu: int = 2,
):
    """Build the launcher of the 8-wave GEMM2 + SDMA comm kernel.

    ``layout``: the ``stage2_sdma.ArenaLayout`` of the cco window; ``chunk_tiles``: sort
    tiles per shipped chunk; ``grid_ctas``: one per CU, workgroup 0 being the comm
    workgroup; ``comb_bytes``: size of a rank's ``comb_inp`` (rows past the reservation).
    """
    assert K % 256 == 0 and N % BLOCK_N == 0, f"K={K} needs % 256, N={N} % {BLOCK_N}"
    assert 256 % SBM == 0, f"SBM={SBM} must divide 256"
    assert npes <= 8, "the comm workgroup has one wave per peer"
    assert layout.row_bytes == N + N // 32

    K_ITERS = K // BLOCK_K
    N_TILES_A = BLOCK_M // 64
    N_TILES_B = BLOCK_N // 128
    N_ACCUMS = N_TILES_A * N_TILES_B
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    N_LDS_STEPS_A = LDS_BLOCK_M // 64
    B_ROW_BYTES = BLOCK_K // 2  # fp4: one K step of a B row is 64 bytes
    N_LDS_STEPS_B = LDS_BLOCK_N * B_ROW_BYTES // (8 * 1024)  # 16 B per lane, 8 waves
    assert N_LDS_STEPS_B == 1
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)
    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * B_ROW_BYTES
    A_GRP_ROWS = N_TILES_A * 16  # rows one wave_m owns inside one LDS half
    B_GRP_ROWS = N_TILES_B * 16
    B_K_STEP = 1024  # one 16-row group x 128 K of the a16w4 layout
    send_nbytes = layout.n_rows * layout.row_bytes
    row_bytes = layout.row_bytes
    gemm_ctas = grid_ctas - 1
    log2_max_tok = max_tok.bit_length() - 1

    epi_bytes = 128 * 256 + 128 * 8 + 3 * 256 * 4 + 4 * 64 * 4
    assert 4 * (a_lds_size + b_lds_size) + epi_bytes <= LDS_LIMIT_BYTES

    kname = (
        f"mega_moe_gemm2_8wave_sdma_n{N}_k{K}_e{experts}_t{topk}_p{npes}_m{max_tok}_s{SBM}"
        f"_c{layout.cap}_ct{chunk_tiles}_g{grid_ctas}_r{rank}"
    )

    @fx.struct
    class SharedStorage:
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        # Epilogue, never aliased with the main loop's buffers.
        E_stage: fx.Array[
            fx.Float8E4M3FN, 128 * 256, 16
        ]  # fp8 rows of one 128-row half
        E_scale: fx.Array[fx.Float8E4M3FN, 128 * 8, 16]  # their e8m0, 8 per row
        E_orow: fx.Array[fx.Float8E4M3FN, 256 * 4, 16]  # out_row per tile row
        E_src: fx.Array[fx.Float8E4M3FN, 256 * 4, 16]  # packed srcmap word per tile row
        E_wts: fx.Array[fx.Float8E4M3FN, 256 * 4, 16]  # routing weight per tile row
        E_tab: fx.Array[fx.Float8E4M3FN, 4 * 64 * 4, 16]  # the tile table

    @flyc.kernel(name=kname, known_block_size=[512, 1, 1])
    def kernel(
        A: fx.Tensor,
        B_T: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        tile_end: fx.Int64,
        stids: fx.Int64,
        sweights: fx.Int64,
        trb: fx.Int64,
        p2p: fx.Int64,
        num_valid: fx.Int64,
        out_row: fx.Int64,
        send_base: fx.Int64,
        dev_comm: fx.Int64,
        win: fx.Int64,
        cdone: fx.Int64,
        tile_base: fx.Int64,
        cursor: fx.Int64,
        idx_send: fx.Int64,
    ):
        import mori.cco.device.flydsl as cco

        F8_IR_t = fx.Float8E4M3FN.ir_type
        n_blocks = fx.Int32(N // BLOCK_N)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        def lds_addr(arr):
            return fx.Int32(fx.ptrtoint(arr.ptr))

        def lane_ids():
            # Recomputed per work item and per epilogue (see opaque_i32).
            tid = opaque_i32(fx.thread_idx.x)
            wave_id = tid // fx.Int32(64)
            return (
                tid,
                tid % fx.Int32(64),
                wave_id,
                wave_id // fx.Int32(4),
                wave_id % fx.Int32(4),
            )

        def process_tile(block_m, block_n, tab_lds):
            _, lane_id, wave_id, wave_m, wave_n = lane_ids()
            a_cur0, a_cur1 = lds.A_lds_cur_0, lds.A_lds_cur_1
            a_next0, a_next1 = lds.A_lds_next_0, lds.A_lds_next_1
            b_cur0, b_cur1 = lds.B_lds_cur_0, lds.B_lds_cur_1
            b_next0, b_next1 = lds.B_lds_next_0, lds.B_lds_next_1
            row_base, expert, n_valid = tile_find(
                tab_lds, block_m, lane_id, SBM=SBM, experts=experts
            )
            b_row_block = expert * n_blocks + block_n

            A0_gl_offset = row_base * K
            A1_gl_offset = (row_base + LDS_BLOCK_M) * K
            B0_gl_offset = (b_row_block * BLOCK_N) * (K // 2)
            B1_gl_offset = (b_row_block * BLOCK_N + LDS_BLOCK_N) * (K // 2)
            a_div = fx.logical_divide(
                make_fp8_buffer_tensor(A, F8_IR_t), fx.make_layout(1, 1)
            )
            b_div = fx.logical_divide(
                make_fp8_buffer_tensor(B_T, F8_IR_t), fx.make_layout(1, 1)
            )
            gl_off_a = compute_global_swizzle(
                lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False
            )
            # a16w4 layout: wave w copies 16-row group w's contiguous 1 KB of a K step.
            gl_off_b = [wave_id * (16 * (K // 2)) + lane_id * 16]

            mfma = MxMfma(N_TILES_A, N_TILES_B)
            a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
            b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
            a_s2r = S2RLoader(wave_m, N_TILES_A)
            a_s2r.lane_id = lane_id
            b_s2r = S2RF4G(wave_n, N_TILES_B, lane_id)
            a_sc = ScalePreshuffledS2R(A_scale, c_m, K, N_TILES_A, lane_id)
            b_sc = ScalePreshuffledS2R(B_scale, experts * N, K, N_TILES_B, lane_id)
            # 16-row tile index of each wave's first row, per LDS half.
            a_base16 = [
                (row_base + h * LDS_BLOCK_M + wave_m * A_GRP_ROWS) // 16
                for h in range_constexpr(2)
            ]
            b_base16 = [
                (b_row_block * BLOCK_N + h * LDS_BLOCK_N + wave_n * B_GRP_ROWS) // 16
                for h in range_constexpr(2)
            ]
            # The wait_barrier counts below assume 6 scale loads per K-pair.
            assert 2 * (N_TILES_A // 2) + 2 * (N_TILES_B // 2) == 6

            def scale_prefetch(sc, k0):
                if const_expr(k0 >= K_ITERS or k0 // 2 in sc):
                    return sc
                words = {
                    w: (a_sc if w[0] == "a" else b_sc).read(
                        (a_base16 if w[0] == "a" else b_base16)[int(w[1])], k0
                    )
                    for w in ("a0", "a1", "b0", "b1")
                }
                return {**sc, k0 // 2: words}

            def scale_read(sc, k, which):
                return sc[k // 2][which]

            c00_frag = [mfma.zero_value] * N_ACCUMS
            c01_frag = [mfma.zero_value] * N_ACCUMS
            c10_frag = [mfma.zero_value] * N_ACCUMS
            c11_frag = [mfma.zero_value] * N_ACCUMS

            sc_pf = scale_prefetch({}, 0)
            b_g2s.load(b_cur0, B0_gl_offset)
            a_g2s.load(a_cur0, A0_gl_offset)
            b_g2s.load(b_cur1, B1_gl_offset)
            a_g2s.load(a_cur1, A1_gl_offset)
            # The wave_m == 1 group runs one barrier behind the other from here on.
            if wave_m == 1:
                rocdl.s_barrier()
            wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)
            b_g2s.load(b_next0, B0_gl_offset + B_K_STEP)
            a_g2s.load(a_next0, A0_gl_offset + BLOCK_K)
            b_g2s.load(b_next1, B1_gl_offset + B_K_STEP)
            wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

            for k in range_constexpr(K_ITERS - 2):
                if const_expr(k % 2 == 1):
                    sc_pf = scale_prefetch(sc_pf, k + 1)
                sa0 = scale_read(sc_pf, k, "a0")
                sb0 = scale_read(sc_pf, k, "b0")
                b0_frag = b_s2r.load(b_cur0)
                a0_frag = a_s2r.load(a_cur0)
                a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
                rocdl.s_barrier()

                c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, sa0, sb0, k_pack=k % 2)

                sb1 = scale_read(sc_pf, k, "b1")
                b1_frag = b_s2r.load(b_cur1)
                b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * B_K_STEP)
                rocdl.s_barrier()

                c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, sa0, sb1, k_pack=k % 2)

                sa1 = scale_read(sc_pf, k, "a1")
                a1_frag = a_s2r.load(a_cur1)
                a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
                # This barrier of the lagging group pairs with the other group's
                # end-of-step wait, after which that group reads step k + 1's
                # b_cur0 / a_cur0, loaded in step k - 1: drain everything but this
                # step's three loads and step k - 1's b_cur1.
                wait_barrier(2 * N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

                c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, sa1, sb0, k_pack=k % 2)

                b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * B_K_STEP)
                wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

                c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, sa1, sb1, k_pack=k % 2)

                a_cur0, a_next0 = a_next0, a_cur0
                a_cur1, a_next1 = a_next1, a_cur1
                b_cur0, b_next0 = b_next0, b_cur0
                b_cur1, b_next1 = b_next1, b_cur1

            # Step K_ITERS - 2: no loads left to overlap but the last A1 tile.
            k = K_ITERS - 2
            sa0 = scale_read(sc_pf, k, "a0")
            sb0 = scale_read(sc_pf, k, "b0")
            b0_frag = b_s2r.load(b_cur0)
            a0_frag = a_s2r.load(a_cur0)
            # The lagging group loaded step K_ITERS - 1's tiles in step K_ITERS - 3:
            # drain them here, the barrier the other group's next reads pair with.
            wait_barrier(0)
            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, sa0, sb0, k_pack=k % 2)
            sb1 = scale_read(sc_pf, k, "b1")
            b1_frag = b_s2r.load(b_cur1)
            rocdl.s_barrier()
            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, sa0, sb1, k_pack=k % 2)
            sa1 = scale_read(sc_pf, k, "a1")
            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_next1, A1_gl_offset + (K_ITERS - 1) * BLOCK_K)
            rocdl.s_barrier()
            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag, sa1, sb0, k_pack=k % 2)
            b0_frag = b_s2r.load(b_next0)
            rocdl.s_barrier()
            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag, sa1, sb1, k_pack=k % 2)
            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

            # Step K_ITERS - 1
            k = K_ITERS - 1
            sa0 = scale_read(sc_pf, k, "a0")
            sb0 = scale_read(sc_pf, k, "b0")
            a0_frag = a_s2r.load(a_cur0)
            wait_barrier(0)
            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag, sa0, sb0, k_pack=k % 2)
            sb1 = scale_read(sc_pf, k, "b1")
            b1_frag = b_s2r.load(b_cur1)
            rocdl.s_barrier()
            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag, sa0, sb1, k_pack=k % 2)
            sa1 = scale_read(sc_pf, k, "a1")
            a1_frag = a_s2r.load(a_cur1)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c10_frag = mfma.call(
                a1_frag, b0_frag, c10_frag, sa1, sb0, k_pack=k % 2, set_prio=False
            )
            c11_frag = mfma.call(
                a1_frag, b1_frag, c11_frag, sa1, sb1, k_pack=k % 2, set_prio=False
            )
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            # Even the two groups' barrier counts again before the epilogue's barriers.
            if wave_m == 0:
                rocdl.s_barrier()

            epilogue(
                (c00_frag, c01_frag),
                (c10_frag, c11_frag),
                row_base,
                n_valid,
                block_n,
                mfma.idx,
            )

        def epilogue(half0, half1, row_base, n_valid, block_n, mfma_idx):
            tid, lane_id, wave_id, wave_m, wave_n = lane_ids()
            orow_lds, src_lds, wts_lds = (
                lds_addr(lds.E_orow),
                lds_addr(lds.E_src),
                lds_addr(lds.E_wts),
            )
            stage_lds, scale_lds = lds_addr(lds.E_stage), lds_addr(lds.E_scale)

            # Row descriptors of the tile: out_row, srcmap word, weight.
            row = tid % fx.Int32(256)
            in_tile = (row < n_valid) & (tid < fx.Int32(256))
            grow = row_base + in_tile.select(row, fx.Int32(0))
            sm_idx = global_typed_ptr(trb, T.i32)[
                grow // fx.Int32(SBM)
            ] + grow % fx.Int32(SBM)
            orow = in_tile.select(
                fx.Int32(global_typed_ptr(out_row, T.i32)[grow]), fx.Int32(-1)
            )
            packed = fx.Int32(global_typed_ptr(stids, T.i32)[sm_idx])
            weight = fx.Float32(global_typed_ptr(sweights, T.f32)[sm_idx])
            if tid < fx.Int32(256):
                lds_typed_ptr(orow_lds, T.i32)[row] = orow
                lds_typed_ptr(src_lds, T.i32)[row] = packed
                lds_typed_ptr(wts_lds, T.f32)[row] = weight
            lds_barrier()

            send = create_buffer_resource_from_addr(
                fx.Int64(rocdl.readfirstlane(T.i64, fx.Int64(send_base).ir_value())),
                num_records_bytes=send_nbytes,
            )
            orow_p = lds_typed_ptr(orow_lds, T.i32)
            src_p = lds_typed_ptr(src_lds, T.i32)
            half = lane_id // fx.Int32(32)
            col = (lane_id % fx.Int32(32)) * fx.Int32(8)
            row_lead = (lane_id % fx.Int32(32)) == fx.Int32(0)
            for h_m in range_constexpr(2):
                if const_expr(h_m > 0):
                    lds_barrier()
                frags = half1 if h_m else half0
                epi_quant_half(
                    frags[0],
                    frags[1],
                    h_m=h_m,
                    wave_m=wave_m,
                    wave_n=wave_n,
                    lane_id=lane_id,
                    mfma_idx=mfma_idx,
                    n_tiles_a=N_TILES_A,
                    wts_lds=wts_lds,
                    stage_lds=stage_lds,
                    scale_lds=scale_lds,
                )
                lds_barrier()
                # Each wave copies 16 rows of the half, two per store (lanes 0-31 / 32-63).
                for it in range_constexpr(8):
                    r = wave_id * fx.Int32(16) + fx.Int32(2 * it) + half
                    o = fx.Int32(orow_p[fx.Int32(h_m * 128) + r])
                    payload = Vec(
                        lds_vec_load(
                            stage_lds,
                            r * 256 + col,
                            Vec.make_type(2, fx.Int32),
                            fx.Int32,
                            align=8,
                        )
                    )
                    scales = Vec(
                        lds_vec_load(
                            scale_lds,
                            r * 8,
                            Vec.make_type(2, fx.Int32),
                            fx.Int32,
                            align=8,
                        )
                    )
                    base = o * fx.Int32(row_bytes)
                    ok = o >= fx.Int32(0)
                    p_off = ok.select(
                        base + block_n * fx.Int32(256) + col, fx.Int32(send_nbytes)
                    )
                    buffer_store(
                        payload,
                        send,
                        p_off,
                        cache_modifier=SEND_CM,
                        offset_is_bytes=True,
                    )
                    s_off = (ok & row_lead).select(
                        base + fx.Int32(N) + block_n * fx.Int32(8),
                        fx.Int32(send_nbytes),
                    )
                    buffer_store(
                        scales,
                        send,
                        s_off,
                        cache_modifier=SEND_CM,
                        offset_is_bytes=True,
                    )
                    # Rows past the per-pair reservation go straight to the origin rank.
                    for h in range_constexpr(2):
                        rh = (
                            wave_id * fx.Int32(16)
                            + fx.Int32(2 * it + h)
                            + fx.Int32(h_m * 128)
                        )
                        if fx.Int32(rocdl.readfirstlane(T.i32, orow_p[rh])) == fx.Int32(
                            OUT_ROW_P2P
                        ):
                            direct_row(
                                fx.Int32(rocdl.readfirstlane(T.i32, src_p[rh])),
                                payload,
                                scales,
                                half == fx.Int32(h),
                                row_lead,
                                col,
                                block_n,
                            )
            publish(row_base, n_valid)

        def direct_row(p, payload, scales, mine, row_lead, col, block_n):
            """Store one row's 256 columns into ``comb_inp[slot]`` of its origin rank, as
            the default Stage2 does, and point that rank's ``inv[slot]`` at it."""
            t = p & fx.Int32(0x00FFFFFF)
            s = p.shrui(fx.Int32(24)) & fx.Int32(0xFF)
            dest_pe = t >> fx.Int32(log2_max_tok)
            slot = (t & fx.Int32(max_tok - 1)) * fx.Int32(topk) + s
            peer = fx.Int64(
                rocdl.readfirstlane(
                    T.i64, global_typed_ptr(p2p, T.i64, align=8)[dest_pe]
                )
            )
            w = cco.Window(win)
            # Like the comm waves: not before the origin rank has read the last call's
            # comb_inp and inv (its combine publishes done[] to this rank).
            if dest_pe != fx.Int32(rank):
                rsrc_counter = create_buffer_resource_from_addr(
                    fx.Int64(w.lsa_ptr(rank, layout.off_counter))
                )
                phase = fx.Int32(
                    buffer_load(rsrc_counter, 0, vec_width=1, dtype=T.i32)
                ) + fx.Int32(1)
                sdma_wait_consumed(w, dest_pe, phase, layout=layout, rank=rank)
            # Written through like S: retired stores are in the peer's memory before the
            # item publishes, with no system-scope release.
            buf = create_buffer_resource_from_addr(peer, num_records_bytes=comb_bytes)
            base = slot * fx.Int32(row_bytes)
            p_off = mine.select(
                base + block_n * fx.Int32(256) + col, fx.Int32(comb_bytes)
            )
            buffer_store(
                payload, buf, p_off, cache_modifier=SEND_CM, offset_is_bytes=True
            )
            s_off = (mine & row_lead).select(
                base + fx.Int32(N) + block_n * fx.Int32(8), fx.Int32(comb_bytes)
            )
            buffer_store(
                scales, buf, s_off, cache_modifier=SEND_CM, offset_is_bytes=True
            )
            if (block_n == fx.Int32(0)) & mine & row_lead:
                comm_ops.store_i32_global_system_monotonic(
                    fx.Int64(w.lsa_ptr(dest_pe, layout.off_inv))
                    + fx.Int64(slot) * fx.Int64(4),
                    fx.Int32(INV_P2P),
                )

        def publish(row_base, n_valid):
            """Every wave's rows have retired (written through); count each sort tile
            of the item once toward its chunk."""
            rocdl.s_waitcnt(vmcnt=0)
            fx.barrier()
            if fx.thread_idx.x == fx.Int32(0):
                st0 = row_base // fx.Int32(SBM)
                n_st = n_valid // fx.Int32(SBM)
                for i in range_constexpr(256 // SBM):
                    if fx.Int32(i) < n_st:
                        comm_ops.atomic_add_agent(
                            fx.Int64(cdone)
                            + fx.Int64((st0 + fx.Int32(i)) // fx.Int32(chunk_tiles))
                            * fx.Int64(CDONE_STRIDE * 4),
                            fx.Int32(1),
                        )

        def comm_role():
            """One wave per peer: fill the peer's ``inv`` for the rows I send it, put
            each finished chunk of my run for it, then signal it."""
            lane = fx.thread_idx.x % fx.Int32(64)
            peer = fx.Int32(rocdl.readfirstlane(T.i32, fx.thread_idx.x // fx.Int32(64)))
            is_lane0 = lane == fx.Int32(0)
            queue = fx.Int32(0)  # one SDMA channel per peer
            lay = layout
            if (peer < fx.Int32(npes)) & (peer != fx.Int32(rank)):
                w = cco.Window(win)
                sdma = cco.DevComm(dev_comm).sdma()
                rsrc_counter = create_buffer_resource_from_addr(
                    fx.Int64(w.lsa_ptr(rank, lay.off_counter))
                )
                # Read-only here: the combine advances it after the call.
                phase = fx.Int32(
                    buffer_load(rsrc_counter, 0, vec_width=1, dtype=T.i32)
                ) + fx.Int32(1)
                tile_base_buf = ptr_buf_tensor(tile_base, fx.Int32)
                idx_buf = ptr_buf_tensor(idx_send, fx.Int32)
                # Do not overwrite the peer's landing rows or inv before it has read the last call's.
                if is_lane0:
                    sdma_wait_consumed(w, peer, phase, layout=lay, rank=rank)
                # inv entries go out while the GEMM computes its first chunk; they must be
                # visible before any put or signal of mine, hence the vmcnt wait.
                cnt = ptr_buf_tensor(cursor, fx.Int32)[peer]
                inv_peer = fx.Int64(w.lsa_ptr(peer, lay.off_inv))
                land0 = fx.Int32(npes * lay.cap + rank * lay.cap)
                for base in range(fx.Int32(0), cnt, fx.Int32(64 * PUSH_U)):
                    ok, js, slots = [], [], []
                    for u in range_constexpr(PUSH_U):
                        j = base + fx.Int32(u * 64) + lane
                        in_run = j < cnt
                        ok.append(in_run)
                        js.append(j)
                        slots.append(
                            idx_buf[
                                peer * fx.Int32(lay.cap) + in_run.select(j, fx.Int32(0))
                            ]
                        )
                    for u in range_constexpr(PUSH_U):
                        if ok[u]:
                            comm_ops.store_i32_global_system_monotonic(
                                inv_peer + fx.Int64(slots[u]) * fx.Int64(4),
                                land0 + js[u],
                            )
                rocdl.s_waitcnt(vmcnt=0)

                total_tiles = (
                    global_typed_ptr(num_valid, T.i32)[0] + fx.Int32(SBM - 1)
                ) // fx.Int32(SBM)
                n_chunks = (total_tiles + fx.Int32(chunk_tiles - 1)) // fx.Int32(
                    chunk_tiles
                )
                last = n_chunks - fx.Int32(1)
                for c in range(fx.Int32(0), n_chunks, fx.Int32(1)):
                    ci = fx.Int32(c)
                    t0 = ci * fx.Int32(chunk_tiles)
                    t1 = t0 + fx.Int32(chunk_tiles)
                    t1 = (t1 < total_tiles).select(t1, total_tiles)
                    # Every (sort tile, N-block) of this chunk has stored.
                    if is_lane0:
                        comm_ops.spin_until_ge_i32_agent(
                            fx.Int64(cdone) + fx.Int64(ci) * fx.Int64(CDONE_STRIDE * 4),
                            (t1 - t0) * n_blocks,
                        )
                    comm_ops.fence_agent_acquire()
                    lo = tile_base_buf[t0 * fx.Int32(npes) + peer]
                    hi = tile_base_buf[t1 * fx.Int32(npes) + peer]
                    if (hi > lo) & is_lane0:
                        # The last chunk's put carries the completion signal.
                        if ci == last:
                            sdma_put_run(
                                sdma,
                                win,
                                peer,
                                queue,
                                layout=lay,
                                rank=rank,
                                lo=lo,
                                hi=hi,
                                cco=cco,
                                remote_signal=True,
                            )
                        else:
                            sdma_put_run(
                                sdma,
                                win,
                                peer,
                                queue,
                                layout=lay,
                                rank=rank,
                                lo=lo,
                                hi=hi,
                                cco=cco,
                            )
                if is_lane0:
                    t_last = last * fx.Int32(chunk_tiles)
                    t_last = (t_last > fx.Int32(0)).select(t_last, fx.Int32(0))
                    lo_l = tile_base_buf[t_last * fx.Int32(npes) + peer]
                    hi_l = tile_base_buf[total_tiles * fx.Int32(npes) + peer]
                    # A peer whose last chunk was empty (or with no chunks) still needs the signal.
                    if (n_chunks <= fx.Int32(0)) | (hi_l <= lo_l):
                        sdma_signal_peer(
                            sdma, win, peer, queue, layout=lay, rank=rank, cco=cco
                        )
                    publish_flag(w, peer, phase, layout=lay, rank=rank)

        def gemm_role():
            tab_lds = lds_addr(lds.E_tab)
            lane = fx.thread_idx.x % fx.Int32(64)
            values, n_tiles = tile_table(tile_end, lane, SBM=SBM, experts=experts)
            tile_table_to_lds(values, tab_lds, lane)
            lds_barrier()
            for w in range(
                fx.Int32(fx.block_idx.x) - fx.Int32(1),
                n_tiles * n_blocks,
                fx.Int32(gemm_ctas),
            ):
                fx.barrier()
                process_tile(w // n_blocks, w % n_blocks, tab_lds)

        if fx.Int32(fx.block_idx.x) == fx.Int32(0):
            comm_role()
        else:
            gemm_role()

    @flyc.jit
    def launch(
        A: fx.Tensor,
        B_T: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        tile_end: fx.Int64,
        stids: fx.Int64,
        sweights: fx.Int64,
        trb: fx.Int64,
        p2p: fx.Int64,
        num_valid: fx.Int64,
        out_row: fx.Int64,
        send_base: fx.Int64,
        dev_comm: fx.Int64,
        win: fx.Int64,
        cdone: fx.Int64,
        tile_base: fx.Int64,
        cursor: fx.Int64,
        idx_send: fx.Int64,
        stream: fx.Stream,
    ):
        kernel(
            A,
            B_T,
            A_scale,
            B_scale,
            c_m,
            tile_end,
            stids,
            sweights,
            trb,
            p2p,
            num_valid,
            out_row,
            send_base,
            dev_comm,
            win,
            cdone,
            tile_base,
            cursor,
            idx_send,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu,
                "rocdl.flat_work_group_size": "512,512",
            },
        ).launch(grid=(grid_ctas, 1, 1), block=(512, 1, 1), stream=stream)

    return set_slp_vectorize_once_used(launch, False)
