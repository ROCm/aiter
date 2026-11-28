# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Persistent 8-wave block-FP8 grouped GEMMs over AITER's ``moe_sorting`` tiles.

The K pipeline is ``mxfp8_gemm_8wave``'s: 256x256x128 tiles, 8 waves as 2 (M)
x 4 (N), the wave_m == 1 group one barrier behind, ``s_setprio`` around each
MFMA burst, A/B double-buffered in LDS. On top of it:

* Row tile ``m`` belongs to expert ``sorted_expert_ids[m]``, and each tile
  rebases the weight descriptor to that expert, so the 32-bit buffer offsets
  never span the whole ``[E, N, K]`` tensor (2.4 GB for 289 experts).
* Stage 1 (``gather=True``) reads A rows straight from the token-major
  activations, row ``r`` at ``token * K`` with the token from ``sorted_ids``
  (padding rows name token ``T``, past the descriptor, and load zeros). Its
  two B halves are gate rows ``[n * 128, +128)`` and up rows
  ``[I + n * 128, +128)``, so a wave holds matching gate and up columns.
  Stage 2 reads the sorted rows contiguously.
* A persistent grid (one workgroup per CU) walks the tile count read on
  device from ``num_valid_ids``. Workgroup ``w`` starts at
  ``(w % 8) * G / 8 + w / 8`` and strides by ``G``, so each XCD runs
  consecutive tiles: the N-blocks of a row tile share its A tile in L2.
* One K-step stream runs across a workgroup's tiles: the last two K-steps of
  a tile load the next tile's first two, and step K - 4 its scales, so the
  pipeline never drains. Stage 1 runs its SwiGLU epilogue at the tile's end
  while those loads land (``run_tile_stream_s1``); stage 2 stores a tile in
  the next tile's first K-step, a quadrant at a time (``run_tile_stream``).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.act import clamp_gate_up, sigmoid_f32
from aiter.ops.flydsl.kernels.gemm_a8w8_8wave import (
    G2SLoader,
    compute_global_swizzle,
    make_fp8_buffer_tensor,
    swizzle_128,
)
from aiter.ops.flydsl.kernels.kernels_common import LOG2E
from aiter.ops.flydsl.kernels.mxfp8_gemm_8wave import MxMfma
from aiter.ops.flydsl.kernels.quant_utils import emit_mx_e8m0_scale
from aiter.utility.mx_types import MX_DEFAULT_ROUND_MODE, MxDtypeInt

BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 128
N_WAVES = 8
NUM_XCDS = 8
LDS_BLOCK_M = BLOCK_M // 2
LDS_BLOCK_N = BLOCK_N // 2
N_TILES_A = BLOCK_M // 64  # 16-row MFMA tiles per wave per LDS half
N_TILES_B = BLOCK_N // 128
N_LDS_STEPS = LDS_BLOCK_M // 64  # 128-byte rows a wave copies per half, /64
A_GRP_ROWS = N_TILES_A * 16
B_GRP_ROWS = N_TILES_B * 16
B_K_STEP = 2 * 1024  # bytes per K-step inside a shuffle_weight (16, 16) 16-row strip


def _load_i32(div, idx):
    """One i32 from a buffer tensor, made wave-uniform (all lanes ask for the same one)."""
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
    reg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
    fx.copy(atom, fx.slice(div, (None, fx.Int32(idx))), reg)
    return _sgpr(reg.load()[0])


def _sgpr(v):
    """A wave-uniform value in an SGPR (readfirstlane).

    Left in VGPRs, wave-derived terms competed with the accumulators and
    spilled, and the reloads' vmcnt waits drained the LDS-DMA. Raw rocdl:
    FlyDSL 0.5.3 has no typed uniformity API.
    """
    return fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(v)))


def _pow2_div(x, d):
    """``x // d`` for ``x >= 0`` and a power-of-two ``d``: a shift (``//`` adds signed fixups)."""
    assert d & (d - 1) == 0
    return fx.Int32(x) >> (d.bit_length() - 1)


def _pow2_mod(x, d):
    """``x % d`` for a power-of-two ``d`` as a mask (floor-mod for negative ``x`` too)."""
    assert d & (d - 1) == 0
    return fx.Int32(x) & (d - 1)


def _load_i32_lane(div, idx):
    """One i32 per lane from a buffer tensor."""
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
    reg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
    fx.copy(atom, fx.slice(div, (None, fx.Int32(idx))), reg)
    return fx.Int32(reg.load()[0])


def _lane_col8(lane):
    """First of the 8 columns ``lane`` holds after the pair swap, within a 32-column pair."""
    q = lane // 16
    return (q % 2 == 1).select(16 + 4 * (q - 1), 4 * q)


def _permlane_swap(op, x, y):
    """``rocdl.permlane{16,32}_swap`` of two i32 words: both words of the swapped pair.

    Raw rocdl + llvm.extractvalue: FlyDSL 0.5.3 has no typed permlane swap,
    and ``fx.shuffle_xor`` adds a v_cmp + v_cndmask per use.
    """
    i32 = fx.Int32.ir_type
    pair = op(
        _llvm.StructType.get_literal([i32, i32]),
        fx.Int32(x).ir_value(),
        fx.Int32(y).ir_value(),
        False,
        False,
    )
    return [fx.Int32(_llvm.extractvalue(i32, pair, [h])) for h in (0, 1)]


def _swap_pair(x, y):
    """(lower-row word, upper-row word) in every lane: one v_permlane16_swap_b32."""
    return _permlane_swap(rocdl.permlane16_swap, x, y)


def _i32_view(t, n_bytes, n_words, align):
    """Buffer tensor over ``t`` (``n_bytes`` checked), indexed in i32 words."""
    g = fx.rocdl.make_buffer_tensor(t, max_size=False, num_records_bytes=n_bytes)
    ptr = fx.PointerType.get(
        elem_ty=fx.Int32.ir_type,
        address_space=rocdl.TargetAddressSpace.BufferDesc,
        alignment=align,
    )
    return fx.logical_divide(
        fx.Tensor(
            fx.make_view(
                fx.recast_iter(ptr, fx.get_iter(g)), fx.make_layout(n_words, 1)
            )
        ),
        fx.make_layout(1, 1),
    )


class StoreSlotBf16:
    """Stage-2 epilogue: unweighted bf16 rows scattered to ``[T * topk, D]``.

    Sorted row r goes to row ``token * topk + slot`` (both from
    ``sorted_ids[r]``); the top-k reduction applies the routing weights
    (HK's split: a weight multiply here is VALU issue in the store phases,
    ATT). Padding rows name token T, past the descriptor, so their stores
    are dropped without a branch.

    A tile's 256 routes are loaded in its first K-step (``fetch``), parked in
    ``route`` LDS once that step's waits have passed (``stash``) and read
    back when the next tile stores this one (``rows_half``); loaded in the
    epilogue, their latency sat behind a vmcnt(0) every tile (25% of the
    stage). ``route`` has one slot per tile parity.
    """

    def __init__(self, C, n_tokens, topk, D, ids_div, route, rt_zero):
        self.rt_zero = rt_zero
        self.D = D
        self.topk = topk
        self.ids_div = ids_div
        self.route = route
        self.wave_row = _sgpr(fx.thread_idx.x // 64) * 32 + fx.thread_idx.x % 32
        n_rows = n_tokens * topk
        self.c_div = _i32_view(C, n_rows * D * 2, n_rows * (D // 2), 16)
        # One past the last word: a dropped store. (2**30 words would wrap to
        # byte 0 in the 32-bit byte offset.)
        self.oob = n_rows * (D // 2)
        self.atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        self.reg = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
        # Quad order: lane 4a + b stores 8-column chunk b of row a, so every 4
        # lanes write 64 contiguous bytes. The TA splits a store by lane
        # quads; in the MFMA order (lane 16q + r holds row r) a quad touches 4
        # rows and moved ~17 B/clk a CU against ~47 in quad order
        # (microbenchmark). Lane 4a + b takes the words of lane 16q + a, q the
        # group holding chunk b (_lane_col8: q = 0, 2, 1, 3 for chunks 0..3),
        # and reads row a's route itself (``rows_half``).
        lane = fx.thread_idx.x % 64
        b = lane % 4
        cinv = (b == 1).select(2, (b == 2).select(1, b))
        self.perm = cinv * 16 + lane // 4
        self.quad_col = b * 8

    def _bperm(self, v):
        """``v`` of lane ``perm`` (a ds_bpermute)."""
        return fx.Int32(fx.shuffle_idx(fx.Int32(v), self.perm, 64))

    def _at(self, idx):
        return fx.make_view(
            fx.add_offset(self.route.ptr, fx.make_int_tuple(idx)), fx.make_layout(1, 1)
        )

    def fetch(self, row0):
        """Load this lane's route of the tile (row ``wave_row``)."""
        return _load_i32_lane(self.ids_div, row0 + self.wave_row)

    def stash(self, route, slot):
        # Lanes l and l + 32 hold the same row and store the same value.
        fx.memref_store_vec(
            fx.full(1, route, fx.Int32), self._at(slot * 512 + self.wave_row)
        )

    def rows_half(self, wave_m, slot, h):
        """Output row (quad order) of each row tile of row half ``h``.

        The reads are constant offsets from one opaque base: per-read
        addresses lost the field's alias scope on row half 1, whose first
        read then waited (vmcnt(0)) on the streamed LDS-DMA (ISA).
        """
        tile0 = slot * 512 + wave_m * A_GRP_ROWS
        base = _opaque(tile0 + fx.thread_idx.x % 64 // 4, self.rt_zero)
        out = []
        for i in range_constexpr(N_TILES_A):
            sid = Vec(self._at(base + h * LDS_BLOCK_M + i * 16).load())[0]
            out.append((sid & 0xFFFFFF) * self.topk + (sid >> 24))
        return out

    def quad_offsets(self, rows, valid):
        """Per row tile: this lane's i32-word offset at column 0 (``oob`` if not ``valid``)."""
        offs = []
        for i in range_constexpr(N_TILES_A):
            off = rows[i] * (self.D // 2) + self.quad_col // 2
            offs.append(valid.select(off, self.oob))
        return offs

    def pack(self, c_frag):
        """bf16 words of ``c_frag`` (``[j * N_TILES_A + i]``) in quad order: 4 per row tile."""
        words = []
        for i in range_constexpr(N_TILES_A):
            x, y = Vec(c_frag[i]), Vec(c_frag[N_TILES_A + i])
            x = x.to(fx.BFloat16).bitcast(fx.Int32)
            y = y.to(fx.BFloat16).bitcast(fx.Int32)
            # After the swap lane 16q + r holds 8 consecutive columns of row r.
            lo = _swap_pair(x[0], y[0])
            hi = _swap_pair(x[1], y[1])
            words += [self._bperm(v) for v in (lo[0], hi[0], lo[1], hi[1])]
        return words

    def store_dummy(self):
        """``N_TILES_A`` 4-byte stores past the descriptor: ``store``'s vmcnt count with ~no TA data."""
        atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        reg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
        fx.memref_store_vec(fx.full(1, fx.Int32(0), fx.Int32), reg)
        for _ in range_constexpr(N_TILES_A):
            fx.copy(atom, reg, fx.slice(self.c_div, (None, self.oob)))

    def store(self, words, offs, col):
        """Store ``pack``'s words at ``quad_offsets`` + column ``col`` (wave-uniform: the soffset)."""
        for i in range_constexpr(N_TILES_A):
            fx.memref_store_vec(
                Vec.from_elements(words[4 * i : 4 * i + 4], fx.Int32), self.reg
            )
            fx.copy(
                self.atom,
                self.reg,
                fx.slice(self.c_div, (None, offs[i])),
                # In elements of the view (i32 words), like the offset.
                soffset=_pow2_div(col, 2),
            )


class SwigluFp8Epilogue:
    """Stage-1 epilogue: SwiGLU, 1x128 power-of-two requant, fp8 + E8M0 stores.

    Tile (m, nb) holds gate (c?0) and up (c?1) of intermediate features
    ``nb * 128 + [0, 128)``, one 1x128 scale block of the stage-2 input. A lane
    holds features ``wave_n * 32 + j * 16 + 4q + [0, 4)`` of one row, the
    gate and up of each in the same register slot. Per row: the lane max, then
    the max over the four q lanes (two lane swaps), then over the group's four
    wave_n waves through LDS (``amax`` holds [wave_m][wave_n][128 rows]); the
    one barrier this takes pairs between the two groups like any other.
    ``clamp=True`` computes ``silu(min(g, L)) * clamp(u, -L, L)`` with
    ``L = limit``; ``clamp=False`` is the no-limit variant (three fewer VALU
    ops a value).
    """

    def __init__(
        self, C, C_scale, m_pad, inter, limit, amax_lds, wave_m, wave_n, clamp=True
    ):
        self.clamp = clamp
        self.inter = inter
        self.m_pad = m_pad
        self.neg_limit = -limit
        self.wave_m = wave_m
        self.wave_n = wave_n
        self.c_div = _i32_view(C, m_pad * inter, m_pad * (inter // 4), 8)
        self.atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Int32)
        self.reg = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
        n_sc = m_pad * (inter // 128)
        self.sc_div = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                C_scale, max_size=False, num_records_bytes=n_sc
            ),
            fx.make_layout(1, 1),
        )
        self.sc_atom = fx.make_copy_atom(fx.rocdl.BufferCopy8b(), fx.Int8)
        self.sc_reg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int8)
        # Past the descriptor (n_sc bytes): a dropped store. A constant, not
        # n_sc: a VGPR copy of n_sc was held across the tile loop and spilled.
        self.sc_oob = fx.Int32(1 << 30)
        self.amax = amax_lds

    def _amax_at(self, wn, r):
        return fx.make_view(
            fx.add_offset(
                self.amax.ptr, fx.make_int_tuple((self.wave_m * 4 + wn) * 128 + r)
            ),
            fx.make_layout(1, 1),
        )

    def _rmax(self, v):
        """Max over the four q lanes of a row (lanes l ^ 16, l ^ 32).

        The max of a swap's two outputs needs no lane select (``fx.shuffle_xor``:
        +50 VALU a wave a tile).
        """
        for op in (rocdl.permlane32_swap, rocdl.permlane16_swap):
            o = _permlane_swap(op, v.bitcast(fx.Int32), v.bitcast(fx.Int32))
            v = o[0].bitcast(fx.Float32).maximumf(o[1].bitcast(fx.Float32))
        return v

    def _act(self, g4, u4):
        """SwiGLU of one accumulator's four (gate, up) pairs."""
        if const_expr(self.clamp):
            out = []
            for v in range_constexpr(4):
                g, u = clamp_gate_up(g4[v], u4[v], self.neg_limit)
                out.append(g * sigmoid_f32(g) * u)
            return out
        # silu(g) * u = g u / (1 + 2^(-g log2 e)): products and sums as
        # two-wide packed ops, exponentials and reciprocals one at a time.
        e = g4 * fx.Float32(-LOG2E)
        e = Vec.from_elements(
            [fx.exp2(e[v], fastmath="afn") for v in range_constexpr(4)], fx.Float32
        )
        e = e + fx.Float32(1.0)
        # rocdl.rcp: v_rcp_f32; ``1 / e`` is the IEEE divide (four ops) and
        # takes no fast-math flag.
        r = Vec.from_elements(
            [
                fx.Float32(rocdl.rcp(fx.Float32.ir_type, fx.Float32(e[v])))
                for v in range_constexpr(4)
            ],
            fx.Float32,
        )
        a = g4 * u4 * r
        return [a[v] for v in range_constexpr(4)]

    def store(self, gate_up, row0, nb):
        """``gate_up[h] = (gate accs, up accs)`` of row half h; acc ``j * N_TILES_A + i``."""
        lane = fx.thread_idx.x % 64
        self.lane_row = lane % 16
        self.q = lane // 16
        self.lane_col = _lane_col8(lane)
        acts = {}
        for h in range_constexpr(2):
            g_frag, u_frag = gate_up[h]
            for i in range_constexpr(N_TILES_A):
                m = fx.Float32(0.0)
                for j in range_constexpr(2):
                    g4 = Vec(g_frag[j * N_TILES_A + i])
                    u4 = Vec(u_frag[j * N_TILES_A + i])
                    vals = self._act(g4, u4)
                    for v in range_constexpr(4):
                        m = m.maximumf(fx.math.absf(vals[v]))
                    acts[(h, i, j)] = vals
                m = self._rmax(m)
                # The four q lanes of a row hold the same maximum; all store it.
                r = h * 64 + i * 16 + self.lane_row
                fx.memref_store_vec(
                    fx.full(1, m, fx.Float32), self._amax_at(self.wave_n, r)
                )
        # Every wave of the group has written its maxima.
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        rocdl.s_barrier()
        for h in range_constexpr(2):
            for i in range_constexpr(N_TILES_A):
                r = h * 64 + i * 16 + self.lane_row
                m = Vec(self._amax_at(0, r).load())[0]
                for wn in range_constexpr(1, 4):
                    m = m.maximumf(Vec(self._amax_at(wn, r).load())[0])
                # ceil_pow2(max(amax, 1e-10) / 448), HK's and the reference's rule.
                e = fx.Int32(
                    emit_mx_e8m0_scale(
                        m.maximumf(fx.Float32(1e-10)).ir_value(),
                        mode=MX_DEFAULT_ROUND_MODE,
                        dtype=MxDtypeInt.FP8_E4M3,
                    )
                )
                # One v_cvt_scalef32_pk_fp8_f32 a value pair divides by 2^(e - 127)
                # and rounds anything below 464 to 448 (gfx950), covering the pow2
                # scale's 1-ulp overflow without a clamp. Raw rocdl: FlyDSL has no
                # scaled fp8 convert.
                scale = (e << 23).bitcast(fx.Float32)
                i16x2 = Vec.make_type(2, fx.Int16)
                words = []
                for j in range_constexpr(2):
                    vals = acts[(h, i, j)]
                    w = Vec.from_elements([0, 0], fx.Int16)
                    for p in range_constexpr(2):
                        w = Vec(
                            rocdl.cvt_scalef32_pk_fp8_f32(
                                i16x2,
                                w.ir_value(),
                                fx.Float32(vals[2 * p]).ir_value(),
                                fx.Float32(vals[2 * p + 1]).ir_value(),
                                scale.ir_value(),
                                p == 1,
                            )
                        )
                    words.append(w.bitcast(fx.Int32)[0])
                pair = _swap_pair(words[0], words[1])
                fx.memref_store_vec(Vec.from_elements(pair, fx.Int32), self.reg)
                row = row0 + h * LDS_BLOCK_M + self.wave_m * A_GRP_ROWS + i * 16
                row = row + self.lane_row
                col = nb * 128 + self.wave_n * B_GRP_ROWS + self.lane_col
                fx.copy(
                    self.atom,
                    self.reg,
                    fx.slice(self.c_div, (None, (row * self.inter + col) // 4)),
                )
                # One writer per row: q == 0 lanes of the wave_n == 0 wave.
                writer = (self.q == 0) & (self.wave_n == 0)
                sc_idx = writer.select(nb * self.m_pad + row, self.sc_oob)
                fx.memref_store_vec(fx.full(1, e.to(fx.Int8), fx.Int8), self.sc_reg)
                fx.copy(
                    self.sc_atom, self.sc_reg, fx.slice(self.sc_div, (None, sc_idx))
                )


def _interleave_store_burst(n_row_tiles):
    """Schedule an 8-MFMA burst that also packs and stores one quadrant.

    Per 16-row tile: MFMA, its 4 bf16 converts + 2 permlane swaps, MFMA, its
    4 ds_bpermutes, then the previous tile's store (one tile behind, so the
    store's lgkmcnt finds its bpermutes done). Raw
    ``rocdl.sched_group_barrier``: flydsl 0.5.3 has no ``fx`` scheduling API.
    """
    mfma, valu, ds, vmem_w = 8, 2, 128, 64
    for t in range_constexpr(n_row_tiles):
        rocdl.sched_group_barrier(mfma, 1, 0)
        rocdl.sched_group_barrier(valu, 6, 0)
        rocdl.sched_group_barrier(mfma, 1, 0)
        rocdl.sched_group_barrier(ds, 4, 0)
        if const_expr(t > 0):
            rocdl.sched_group_barrier(vmem_w, 1, 0)
    rocdl.sched_group_barrier(vmem_w, 1, 0)


def _wait_barrier_seen(count):
    """``s_waitcnt vmcnt(count)`` + ``s_barrier``.

    As intrinsics (not inline asm) the compiler's waitcnt pass sees the wait
    and does not add its own, tighter ones (a vmcnt(0) of ~550 cycles a
    stage-1 tile).
    """
    fx.rocdl.s_waitcnt(vmcnt=count)
    rocdl.s_barrier()


def _opaque(value, rt_zero):
    """``value`` OR-ed with a runtime zero (``m_pad >> 31``): a base the compiler cannot fold.

    LDS reads then add constant offsets to one base VGPR (see
    ``LdsFragReader`` and ``StoreSlotBf16.rows_half``).
    """
    return fx.Int32(value) | rt_zero


class LdsFragReader:
    """MFMA operand fragments from one LDS region, one address VGPR per K half.

    Tile ``i`` sits ``i * 2048`` bytes past tile 0 in both the A and the
    ``shuffle_weight`` B layouts, and a region's four buffers lie within 64 KB.
    The address is ``field + (opaque(lane + bias) - bias)``, ``bias`` the
    region's address: ``field - bias + tile`` folds into the ds_read
    immediate, and the pointer keeps the field's alias scope. With plain
    addresses the compiler held a VGPR per buffer, tile and K half and
    spilled them; with ``inttoptr`` every read waited for vmcnt(0).
    """

    TILE_BYTES = 16 * BLOCK_K

    def __init__(self, region, wave_idx, n_tiles, preshuffled, rt_zero):
        lane = fx.thread_idx.x % 64
        row = wave_idx * (n_tiles * 16) + lane % 16
        col = (lane // 16) * 16
        self.n_tiles = n_tiles
        bias = fx.Int32(fx.ptrtoint(region.ptr))
        if const_expr(preshuffled):
            off = (row // 8) * 1024 + (row % 8) * 16 + (col // 16) * 128
            rel = _opaque(off + bias, rt_zero) - bias
            # K half 1 is 64 columns = 4 x 16-byte groups further on: an
            # immediate, not a second live VGPR.
            self.rel = [rel, rel + 4 * 128]
        else:
            offs = []
            for step in range_constexpr(2):
                r, c = swizzle_128(row, col + step * 64)
                offs.append(r * 128 + c)
            self.rel = [_opaque(o + bias, rt_zero) - bias for o in offs]

    def load(self, lds_src):
        frag = []
        for i in range_constexpr(self.n_tiles):
            halves = []
            for step in range_constexpr(2):
                ptr = fx.add_offset(
                    lds_src.ptr,
                    fx.make_int_tuple(self.rel[step] + i * self.TILE_BYTES),
                )
                v = fx.make_view(fx.recast_iter(fx.Uint8, ptr), fx.make_layout(16, 1))
                halves.append(v.load().bitcast(fx.Int32))
            frag.append(halves[0].shuffle(halves[1], list(range(8))))
        return frag


def _lds_at(lds_ptr, off):
    """``lds_ptr`` + ``off`` elements: an LDS-DMA destination keeping the field's alias scope."""
    return fx.add_offset(lds_ptr, fx.make_int_tuple(off))


def _lds_dst(lds_ptr, off):
    return fx.make_view(_lds_at(lds_ptr, off), fx.make_layout(1, 1))


def _lds_at_opaque(lds_ptr, off):
    """An LDS-DMA destination as a bare address (``inttoptr``), without the field's alias scope.

    For fields read while a DMA into another part is in flight (the two
    tiles' scale slots, stage 2's A/B buffers): the waitcnt pass cannot tell
    the parts apart and drained the loads in flight before the next read.
    """
    ptr_t = fx.PointerType.get(fx.Float8E4M3FN.ir_type, 2, 512)
    base = fx.Int32(fx.ptrtoint(lds_ptr))
    return fx.inttoptr(ptr_t, base + fx.Int32(off))


def _lds_dst_opaque(lds_ptr, off):
    return fx.make_view(_lds_at_opaque(lds_ptr, off), fx.make_layout(1, 1))


class _FieldDst:
    """``G2SLoader`` with its per-wave LDS destinations taken off the field pointer."""

    def _lds_dst_at(self, lds_dst, step):
        return _lds_dst(lds_dst.ptr, self.wave_id * 1024 + step * (self.n_waves * 1024))


class G2SRounds(G2SLoader):
    """``G2SLoader`` whose rounds sit a constant ``round_bytes`` apart in the source.

    Round r of the 8-wave copy reads rows ``r * 64`` further on, so its offset
    is round 0's plus ``r * 64 * K``: that goes into the SGPR soffset, and the
    loader holds one lane offset instead of one per round.
    """

    def __init__(
        self, gl_src, base_offset, n_load_steps, lds_dtype, wave_id, round_bytes
    ):
        super().__init__(gl_src, [base_offset], n_load_steps, lds_dtype, wave_id)
        self.round_bytes = round_bytes

    def load(self, lds_dst, k_offset, skip=(None, None)):
        """``skip[r]`` != 0 sends round r out of range: no memory traffic, zeros in LDS, same vmcnt."""
        for step in range_constexpr(self.n_load_steps):
            src = fx.slice(self.gl_src, (None, fx.Int32(self.gl_offsets[0])))
            dst = self._lds_dst_at(lds_dst, step)
            soff = fx.Int32(k_offset) + step * self.round_bytes
            if const_expr(skip[step] is not None):
                soff = (skip[step] != 0).select(fx.Int32(1 << 30), soff)
            fx.copy(self.g2lds_atom, src, dst, soffset=soff)


class G2SRoundsNt(G2SRounds):
    """``G2SRounds`` with the nontemporal (nt) cache policy, for data read once.

    Raw intrinsic: the ``BufferCopyLDS`` atom has no cache-policy state.
    """

    NT = 2  # buffer instruction aux bit 1 (nt) on gfx94x / gfx950

    def __init__(
        self, rsrc, base_offset, n_load_steps, lds_dtype, wave_id, round_bytes
    ):
        super().__init__(
            None, base_offset, n_load_steps, lds_dtype, wave_id, round_bytes
        )
        self.rsrc = rsrc

    def load(self, lds_dst, k_offset):
        for step in range_constexpr(self.n_load_steps):
            step_off = self.wave_id * 1024 + step * (self.n_waves * 1024)
            lds_ptr = _lds_at_opaque(lds_dst.ptr, step_off)
            soff = fx.Int32(k_offset) + step * self.round_bytes
            rocdl.raw_ptr_buffer_load_lds(
                self.rsrc,
                fx.to_llvm_ptr(lds_ptr),
                16,
                fx.Int32(self.gl_offsets[0]).ir_value(),
                soff.ir_value(),
                0,
                self.NT,
            )


class G2SGather(_FieldDst, G2SLoader):
    """Gathered A rows: one lane offset per round, the K offset an SGPR set at the load."""

    def __init__(self, gl_src, wave_id):
        super().__init__(gl_src, [], N_LDS_STEPS, fx.Float8E4M3FN.ir_type, wave_id)

    def load(self, lds_dst, offsets, k_offset):
        soff = fx.Int32(k_offset)
        for step in range_constexpr(len(offsets)):
            src = fx.slice(self.gl_src, (None, fx.Int32(offsets[step])))
            fx.copy(self.g2lds_atom, src, self._lds_dst_at(lds_dst, step), soffset=soff)


class TileScaleLds:
    """Stage 2: a tile's E8M0 A / B scales for every K block, one LDS slot per tile parity.

    A (column-major ``[K // 128, M_pad]``): wave ``w`` loads the 256 bytes of K
    block ``(w + rot) % k128`` of the tile's rows, a dword per lane, into
    position ``w``, so step ``k`` of a K-rotated tile reads position ``k``. B:
    every wave loads both weight row blocks' ``k128`` bytes twice over into
    its own 256 bytes (lanes 8..63 repeat lanes 0..7), so the read at byte
    ``rot + k`` needs no wrap. Two DMA instructions a wave a tile: the TA
    limits this stage.
    """

    def __init__(self, a_div, b_div, k128, m_pad, lds, wave_id, a_row):
        assert k128 == N_WAVES, "one A scale block per wave"
        self.a_div = a_div
        self.b_div = b_div
        self.k128 = k128
        self.m_pad = m_pad
        self.lds = lds
        self.wave_id = wave_id
        self.a_area = k128 * 256
        self.slot_bytes = self.a_area + N_WAVES * 256
        self.lane = fx.thread_idx.x % 64
        self.a_lane = a_row + self.lane % 16
        self.atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS32b(), 32)

    BYTES = 2 * (8 * 256 + N_WAVES * 256)

    def dma(self, slot, row0, rot, b_rows):
        """Stage a tile's scales into ``slot``; ``b_rows`` = its two weight row blocks."""
        slot_off = slot * self.slot_bytes + self.wave_id * 256
        blk = _pow2_mod(self.wave_id + rot, self.k128)
        fx.copy(
            self.atom,
            fx.slice(self.a_div, (None, self.lane * 4)),
            _lds_dst_opaque(self.lds.ptr, slot_off),
            soffset=blk * self.m_pad + row0,
        )
        j = self.lane % 8
        off = (j < 4).select(b_rows[0], b_rows[1]) * self.k128 + j % 2 * 4
        fx.copy(
            self.atom,
            fx.slice(self.b_div, (None, off)),
            _lds_dst_opaque(self.lds.ptr, slot_off + self.a_area),
        )

    def bases(self, slot, rot):
        """This tile's read bases, as offsets off the field (keeping its alias scope).

        The wave-uniform parts go through readfirstlane: folded into the field
        address, the reads drew 14 vmcnt(0) drains.
        """
        bias = fx.Int32(fx.ptrtoint(self.lds.ptr))
        slot_off = slot * self.slot_bytes
        a = _sgpr(slot_off + bias) - bias + self.a_lane
        b_off = slot_off + self.a_area + self.wave_id * 256 + rot
        return a, _sgpr(b_off + bias) - bias

    def _byte(self, base, off):
        v = fx.make_view(
            fx.recast_iter(
                fx.Uint8, fx.add_offset(self.lds.ptr, fx.make_int_tuple(base + off))
            ),
            fx.make_layout(1, 1),
        )
        return fx.Int32(v.load()[0])

    def read_a(self, base, k, half_row, n_tiles):
        return [
            self._byte(base, k * 256 + half_row + t * 16)
            for t in range_constexpr(n_tiles)
        ]

    def read_b(self, base, k, h, n_tiles):
        return [self._byte(base, h * 16 + k)] * n_tiles


class GatherTileScaleLds:
    """Stage 1: a tile's E8M0 A / B scales for every K block, two DMA instructions a wave.

    A (gathered row-major ``[T, K // 128]``, 32 bytes a token at K = 4096):
    lane ``l`` of wave ``w`` copies the 16 bytes of K blocks ``16 (w // 4) +
    [0, 16)`` of tile row ``64 (w % 4) + l``, so the LDS image is ``[2][256
    rows][16]``: a read's 16 rows sit 16 bytes apart, on distinct banks (the
    ``[256, 32]`` order has two-way conflicts). B: every wave loads both
    weight row blocks' ``k128`` bytes (lanes 0..7 / 8..15, the rest repeat
    them) into its own 256 bytes.
    """

    def __init__(self, a_div, b_div, k128, lds, wave_id, a_row):
        assert (
            k128 * BLOCK_M == N_WAVES * 64 * 16
        ), "16 bytes a lane cover the tile's A scales"
        self.a_div = a_div
        self.b_div = b_div
        self.k128 = k128
        self.lds = lds
        self.wave_id = wave_id
        self.a_area = BLOCK_M * k128
        self.slot_bytes = self.BYTES_PER_SLOT
        lane = fx.thread_idx.x % 64
        self.dma_row = wave_id % 4 * 64 + lane
        self.dma_col = wave_id // 4 * 16
        self.a_row = a_row
        self.a_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        self.b_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS32b(), 32)

    BYTES_PER_SLOT = 8 * 1024 + N_WAVES * 256

    def dma(self, slot, tok, b_rows):
        """Stage a tile's scales into ``slot``; ``tok``: token of row ``dma_row``, ``b_rows``: weight row blocks."""
        slot_off = slot * self.slot_bytes
        fx.copy(
            self.a_atom,
            fx.slice(self.a_div, (None, tok * self.k128 + self.dma_col)),
            _lds_dst_opaque(self.lds.ptr, slot_off + self.wave_id * 1024),
        )
        j = fx.thread_idx.x % 16
        off = (j < 8).select(b_rows[0], b_rows[1]) * self.k128 + j % 8 * 4
        fx.copy(
            self.b_atom,
            fx.slice(self.b_div, (None, off)),
            _lds_dst_opaque(self.lds.ptr, slot_off + self.a_area + self.wave_id * 256),
        )

    def bases(self, slot):
        """Read bases (element offsets off the field)."""
        slot_off = slot * self.slot_bytes
        a = slot_off + (self.a_row + fx.thread_idx.x % 16) * 16
        return a, slot_off + self.a_area + self.wave_id * 256

    def _byte(self, base, off):
        v = fx.make_view(
            fx.recast_iter(
                fx.Uint8, fx.add_offset(self.lds.ptr, fx.make_int_tuple(base + off))
            ),
            fx.make_layout(1, 1),
        )
        return fx.Int32(v.load()[0])

    def read_a(self, base, k, half_row, n_tiles):
        return [
            self._byte(
                base, k // 16 * (BLOCK_M * 16) + (half_row + t * 16) * 16 + k % 16
            )
            for t in range_constexpr(n_tiles)
        ]

    def read_b(self, base, k, h, n_tiles):
        return [self._byte(base, h * 32 + k)] * n_tiles


def compile_moe_bs_gemm_8w(
    *,
    K: int,
    N: int,
    gather: bool,
    topk: int = 0,
    swiglu_clamp: bool = True,
    half_skip: bool = False,
):
    """Build the launcher of one stage; ``N`` is the weight rows per expert.

    * ``gather=True`` (stage 1, ``N`` = 2 * I): a tile pairs gate row block
      ``n`` with up row block ``I / 128 + n``. ``C`` fp8 ``[M_pad, I]`` is the
      requantized SwiGLU, ``C_scale`` its E8M0 1x128 scales column-major
      ``[I // 128, M_pad]`` (stage 2's A operand as is).
    * ``gather=False`` (stage 2, ``N`` = D): ``C`` bf16 ``[T * topk, D]``
      holds the unweighted expert output of route (token, slot) in row
      ``token * topk + slot``; the top-k reduction applies the weights.
    """
    assert gather or topk > 0
    K_ITERS = K // BLOCK_K
    assert K_ITERS % 2 == 0
    k128 = K // 128
    gate_up_inter = N // 2 if gather else 0
    if gather:
        assert gate_up_inter % 128 == 0
        NB = gate_up_inter // 128
    else:
        assert N % BLOCK_N == 0
        NB = N // BLOCK_N

    def half_row0(nb, h):
        """First weight row (within the expert) of LDS half ``h`` of N-block ``nb``."""
        if const_expr(gate_up_inter):
            return h * gate_up_inter + nb * LDS_BLOCK_N
        return nb * BLOCK_N + h * LDS_BLOCK_N

    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * BLOCK_K

    _kname = (
        f"flydsl_moe_bs_8w_gather_n{N}_k{K}_gu_swiglu_fp8"
        if gather
        else f"flydsl_moe_bs_8w_sorted_n{N}_k{K}_slot_bf16_t{topk}"
    )

    @fx.struct
    class SharedStorage:
        if gather:
            S_lds: fx.Array[fx.Float8E4M3FN, 2 * GatherTileScaleLds.BYTES_PER_SLOT, 16]
            R_lds: fx.Array[fx.Float32, 2 * 4 * 128, 16]
            # A tile's sorted_ids, one slot per tile parity.
            I_lds: fx.Array[fx.Int32, 2 * BLOCK_M, 16]
        else:
            S_lds: fx.Array[fx.Float8E4M3FN, TileScaleLds.BYTES, 16]
            T_lds: fx.Array[fx.Int32, 2 * 512, 16]
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

    @flyc.kernel(name=_kname, known_block_size=[512, 1, 1])
    def kernel_gemm(
        A: fx.Tensor,
        B: fx.Pointer,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        C_scale: fx.Tensor,
        m_pad: fx.Int32,
        limit: fx.Float32,
    ):
        F8_IR_t = fx.Float8E4M3FN.ir_type
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        lane_id = fx.thread_idx.x % 64
        wave_id = _sgpr(fx.thread_idx.x // 64)
        wave_m = wave_id // 4
        wave_n = wave_id % 4
        lagging = 1 - wave_m
        leading = wave_m

        def barrier_unless(flag):
            """``s_barrier`` in the waves whose wave-uniform ``flag`` is 0: opens / closes the lag."""
            if flag == 0:
                rocdl.s_barrier()

        # 0 at run time (m_pad >= 0), unknown to the compiler: ``_opaque``.
        rt_zero = _sgpr(m_pad >> 31)

        a_div = fx.logical_divide(
            make_fp8_buffer_tensor(A, F8_IR_t), fx.make_layout(1, 1)
        )
        a_scale_div = fx.logical_divide(
            make_fp8_buffer_tensor(A_scale, F8_IR_t), fx.make_layout(1, 1)
        )
        b_scale_div = fx.logical_divide(
            make_fp8_buffer_tensor(B_scale, F8_IR_t), fx.make_layout(1, 1)
        )
        ids_div = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(sorted_ids, max_size=False),
            fx.make_layout(1, 1),
        )
        eids_div = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(sorted_expert_ids, max_size=False),
            fx.make_layout(1, 1),
        )
        nv_div = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(num_valid_ids, max_size=False),
            fx.make_layout(1, 1),
        )
        # A raw pointer: [E, N, K] passes 2**31 bytes, past a tensor argument's i32 shape.
        b_iter = fx.recast_iter(fx.Float8E4M3FN, B)

        # Lane-only parts of the LDS-DMA addresses.
        gl_off_b = compute_global_swizzle(
            lane_id, wave_id, K, N_LDS_STEPS, preshuffled=True
        )
        if const_expr(not gather):
            gl_off_a = compute_global_swizzle(
                lane_id, wave_id, K, N_LDS_STEPS, preshuffled=False
            )

        # Operands swapped (weights as the MFMA's A): a lane then holds four
        # consecutive output columns of one row, which pack into wide stores.
        mfma = MxMfma(N_TILES_B, N_TILES_A, paired_tiles=False)
        if const_expr(gather):
            gsc = GatherTileScaleLds(
                a_scale_div,
                b_scale_div,
                k128,
                lds.S_lds,
                wave_id,
                wave_m * A_GRP_ROWS,
            )
        a_s2r = LdsFragReader(
            lds.A_lds_cur_0, wave_m, N_TILES_A, preshuffled=False, rt_zero=rt_zero
        )
        b_s2r = LdsFragReader(
            lds.B_lds_cur_0, wave_n, N_TILES_B, preshuffled=True, rt_zero=rt_zero
        )
        if const_expr(gather):
            store_c = SwigluFp8Epilogue(
                C,
                C_scale,
                m_pad,
                gate_up_inter,
                limit,
                lds.R_lds,
                wave_m,
                wave_n,
                clamp=swiglu_clamp,
            )
        else:
            n_tokens = _load_i32(nv_div, 1)
            store_c = StoreSlotBf16(C, n_tokens, topk, N, ids_div, lds.T_lds, rt_zero)

        QUAD_STORES = N_TILES_A

        def slot_cols(nb):
            return (
                half_row0(nb, 0) + wave_n * B_GRP_ROWS,
                half_row0(nb, 1) + wave_n * B_GRP_ROWS,
            )

        LDS_BUFS = (
            lds.A_lds_cur_0,
            lds.A_lds_cur_1,
            lds.A_lds_next_0,
            lds.A_lds_next_1,
            lds.B_lds_cur_0,
            lds.B_lds_cur_1,
            lds.B_lds_next_0,
            lds.B_lds_next_1,
        )

        if const_expr(not gather):
            tsc = TileScaleLds(
                a_scale_div,
                b_scale_div,
                k128,
                m_pad,
                lds.S_lds,
                wave_id,
                wave_m * A_GRP_ROWS,
            )

        def make_ctx(tile, expert, pmask):
            """Stage 2's loaders of one tile (contiguous A rows), from its index and expert."""
            m_blk = _pow2_div(tile, NB)
            nb = _pow2_mod(tile, NB)
            row0 = m_blk * BLOCK_M
            # K-order rotation: tiles sharing an A panel (an m-block) or a B
            # panel (an expert) start at different K-steps, so most find their
            # K slice already in L2 instead of all missing on the same one
            # (stage 2 -2 to -6%).
            rot = _pow2_mod(m_blk + nb, K_ITERS)
            a_g2s = G2SRounds(a_div, gl_off_a[0], N_LDS_STEPS, F8_IR_t, wave_id, 64 * K)
            a_base = [(row0 + h * LDS_BLOCK_M) * K for h in range_constexpr(2)]
            b_buf = fx.rocdl.make_buffer_tensor(
                fx.make_view(
                    b_iter + fx.Int64(expert) * (N * K), fx.make_layout(N * K, 1)
                ),
                max_size=False,
                num_records_bytes=N * K,
            )
            b_g2s = G2SRounds(
                fx.logical_divide(b_buf, fx.make_layout(1, 1)),
                gl_off_b[0],
                N_LDS_STEPS,
                F8_IR_t,
                wave_id,
                64 * K,
            )
            b_g2s_nt = G2SRoundsNt(
                fx.rocdl.get_buffer_rsrc(fx.get_iter(b_buf)),
                gl_off_b[0],
                N_LDS_STEPS,
                F8_IR_t,
                wave_id,
                64 * K,
            )
            b_off = [half_row0(nb, h) * K for h in range_constexpr(2)]
            b_rows = [
                expert * (N // 128) + _pow2_div(half_row0(nb, h), 128)
                for h in range_constexpr(2)
            ]

            # Rounds whose 64 rows are all padding (``pmask`` bit q: rows 64 q
            # + 64 on are padding) load nothing: at T = 4096 an expert fills
            # ~114 of its 256 rows.
            a_skip = [
                (None, pmask & 1),
                ((pmask >> 1) & 1, (pmask >> 2) & 1),
            ]

            def a_load(h, dst, k):
                a_g2s.load(
                    dst, a_base[h] + _pow2_mod(rot + k, K_ITERS) * BLOCK_K, a_skip[h]
                )

            def b_load(h, dst, k):
                off = b_off[h] + _pow2_mod(rot + k, K_ITERS) * B_K_STEP
                # A weight panel no other m-block of the expert reads goes
                # through L2 nontemporally, leaving L2 to A panels and output
                # lines (stage 2 -1.5 to -6%; nt on every panel is +10% at
                # T = 16384).
                either(
                    (pmask >> 3) & 1,
                    lambda: b_g2s_nt.load(dst, off),
                    lambda: b_g2s.load(dst, off),
                )

            def sc_dma(slot):
                tsc.dma(slot, row0, rot, b_rows)

            return {"a": a_load, "b": b_load, "sc": sc_dma, "row0": row0, "rot": rot}

        def pad_rows(tile):
            """Issue ``pad_mask``'s loads: rows 64 / 128 / 192 of ``tile``'s m-block, the neighbours' experts."""
            m_blk = _pow2_div(tile, NB)
            row0 = m_blk * BLOCK_M
            vals = [_load_i32_lane(ids_div, row0 + 64 * q) for q in range(1, 4)]
            prev_m = (m_blk > 0).select(m_blk - 1, m_blk)
            next_m = (m_blk + 1 < _pow2_div(n_tiles, NB)).select(m_blk + 1, m_blk)
            for m in (prev_m, m_blk, next_m):
                vals.append(_load_i32_lane(eids_div, m))
            return vals

        def pad_mask(sids):
            """Bit q - 1 (q = 1..3): rows 64 q .. 255 of the m-block are all padding.

            Padding rows name token T (moe_sorting) and sit at the end of an
            expert's block, so a padded row means every later one is too.
            """
            m = fx.Int32(0)
            for q in range_constexpr(3):
                pad = ((sids[q] & 0xFFFFFF) >= n_tokens).select(
                    fx.Int32(1), fx.Int32(0)
                )
                m = m | (pad << q)
            # Bit 3: the m-block before or after has the same expert, so its
            # weight panel is read again and should stay in L2.
            shared = (sids[3] == sids[4]) | (sids[5] == sids[4])
            m = m | shared.select(fx.Int32(8), fx.Int32(0))
            return _sgpr(m)

        def either(flag, if_zero, otherwise):
            """``if_zero()`` or ``otherwise()`` on a wave-uniform flag (each issues the same VMEM count)."""
            if flag == 0:
                if_zero()
            else:
                otherwise()

        def either_val(flag, if_zero, otherwise, init):
            """``either`` whose arms return a value: each arm is one straight-line block."""
            out = init
            if flag == 0:
                out = if_zero()
            else:
                out = otherwise()
            return out

        def mfma_skip(skip, burst, c):
            """``burst()`` unless ``skip``: a wave-uniform branch around an MFMA burst."""
            out = c
            if skip == 0:
                out = burst()
            return out

        def mfma_q(b, a, c, sb, sa, skip):
            """``mfma.call`` of a quadrant, dropped when this wave's rows of it are all padding."""
            rocdl.s_setprio(1)
            c = mfma_skip(
                skip,
                lambda: mfma.call(b, a, c, sb, sa, k_pack=0, set_prio=False),
                c,
            )
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            return c

        def run_tile_stream(
            tile, expert, pmask, next_tile, parity, prev_tile, prev_acc, prev_mask
        ):
            """Stage 2: K-steps 0 .. K - 1 of ``tile``; returns (accumulators, next expert, next mask).

            Step k loads A half 1 of step k + 1 and all of step k + 2 (for the
            last two steps, ``next_tile``'s steps 0 and 1); step K - 4 loads
            the next tile's scales. Step 0 also fetches this tile's routes and
            the next tile's expert, and stores the previous tile's
            accumulators ``prev_acc``, each quadrant before its registers
            accumulate again. c00 and c11 go in their own phase's VALU part;
            c01 and c10 ride in the c00 and c01 MFMA bursts, where the pack
            overlaps the other group's VALU part (stage 2 -1.5 to -2.3%).
            """
            valid = prev_tile >= 0
            pcols = slot_cols(_pow2_mod(prev_tile, NB))
            prev_slot = 1 - parity
            prev_offs = {}

            def store_prev(g):
                """Accumulator group g = (row half, column half) g // 2, g % 2."""
                if const_expr(g % 2 == 0):
                    rows = store_c.rows_half(wave_m, prev_slot, g // 2)
                    prev_offs[g // 2] = store_c.quad_offsets(rows, valid)
                offs = prev_offs[g // 2]
                if const_expr(g < 2):
                    store_c.store(store_c.pack(prev_acc[g]), offs, pcols[g % 2])
                else:
                    # This wave's row-half-1 rows of the previous tile are all
                    # padding: the same count of stores, without the 1 KB each.
                    either(
                        prev_skip_h1,
                        lambda: store_c.store(
                            store_c.pack(prev_acc[g]), offs, pcols[g % 2]
                        ),
                        store_c.store_dummy,
                    )

            a_cur0, a_cur1, a_next0, a_next1, b_cur0, b_cur1, b_next0, b_next1 = (
                LDS_BUFS
            )
            cur = make_ctx(tile, expert, pmask)
            # This wave's rows of row half h: wave_m * 64 + 128 h + [0, 64).
            skip_h1 = (pmask >> (wave_m + 1)) & 1
            prev_skip_h1 = (prev_mask >> (wave_m + 1)) & 1
            nxt = None
            nxt_expert = None
            nxt_mask = None
            c00 = [mfma.zero_value] * (N_TILES_A * N_TILES_B)
            c01 = list(c00)
            c10 = list(c00)
            c11 = list(c00)

            def src(kk):
                return (cur, kk) if kk < K_ITERS else (nxt, kk - K_ITERS)

            sc_a, sc_b = tsc.bases(parity, cur["rot"])

            def sc_read(k, which):
                op, h = int(which[0] == "b"), int(which[1])
                if const_expr(op == 1):
                    return tsc.read_b(sc_b, k, h, N_TILES_B)
                return tsc.read_a(sc_a, k, h * LDS_BLOCK_M, N_TILES_A)

            for k in range_constexpr(K_ITERS):
                if const_expr(k == K_ITERS - 4):
                    nxt = make_ctx(next_tile, nxt_expert, nxt_mask)
                sa0 = sc_read(k, "a0")
                sb0 = sc_read(k, "b0")
                b0_frag = b_s2r.load(b_cur0)
                a0_frag = a_s2r.load(a_cur0)
                if const_expr(k == 0):
                    # Older than this step's A1 load: its end wait drains them.
                    routes = store_c.fetch(cur["row0"])
                    nxt_eid = _load_i32_lane(eids_div, _pow2_div(next_tile, NB))
                    nxt_pad = pad_rows(next_tile)
                ctx, kk = src(k + 1)
                ctx["a"](1, a_next1, kk)
                if const_expr(k == 0):
                    store_prev(0)
                rocdl.s_barrier()
                # The next tile's scales go to the other parity's slot, which
                # the previous tile finished reading tiles ago.
                sc_dma = k == K_ITERS - 4
                if const_expr(sc_dma):
                    nxt["sc"](1 - parity)

                if const_expr(k == 0):
                    # The previous tile's c01 rides in this burst.
                    rocdl.s_setprio(1)
                    c00 = mfma.call(
                        b0_frag, a0_frag, c00, sb0, sa0, k_pack=0, set_prio=False
                    )
                    store_prev(1)
                    _interleave_store_burst(N_TILES_A)
                    rocdl.s_setprio(0)
                    rocdl.s_barrier()
                else:
                    c00 = mfma.call(b0_frag, a0_frag, c00, sb0, sa0, k_pack=0)

                sb1 = sc_read(k, "b1")
                b1_frag = b_s2r.load(b_cur1)
                ctx, kk = src(k + 2)
                ctx["b"](0, b_cur0, kk)
                if const_expr(k == 0):
                    rows1 = store_c.rows_half(wave_m, prev_slot, 1)
                    prev_offs[1] = store_c.quad_offsets(rows1, valid)
                rocdl.s_barrier()

                if const_expr(k == 0):
                    # The previous tile's c10 rides in this burst. Its
                    # real-or-dummy choice is taken around the whole burst, so
                    # each arm is one block the scheduler can interleave.
                    rocdl.s_setprio(1)
                    c01_ops = (b1_frag, a0_frag, c01, sb1, sa0)

                    def c01_with_c10(ops=c01_ops):
                        c = mfma.call(*ops, k_pack=0, set_prio=False)
                        store_c.store(store_c.pack(prev_acc[2]), prev_offs[1], pcols[0])
                        _interleave_store_burst(N_TILES_A)
                        return c

                    def c01_with_dummy(ops=c01_ops):
                        c = mfma.call(*ops, k_pack=0, set_prio=False)
                        store_c.store_dummy()
                        return c

                    c01 = either_val(prev_skip_h1, c01_with_c10, c01_with_dummy, c01)
                    rocdl.s_setprio(0)
                    rocdl.s_barrier()
                else:
                    c01 = mfma.call(b1_frag, a0_frag, c01, sb1, sa0, k_pack=0)

                sa1 = sc_read(k, "a1")
                a1_frag = a_s2r.load(a_cur1)
                ctx["a"](0, a_cur0, kk)
                n_sc = 2 if sc_dma else 0
                # vmcnt is in order: a store is drained by the first wait for a
                # load issued after it, so step 0's stores stay in flight
                # through its own waits. Step 1's middle wait drains c00, c01
                # and c10, all issued before A0 of step 2; only c11 may fly.
                _wait_barrier_seen(
                    4 * N_LDS_STEPS
                    + n_sc
                    # Step 0 also issued 8 loads: route, next expert, pad_rows' 6.
                    + (8 + 3 * QUAD_STORES if k == 0 else 0)
                    + (QUAD_STORES if k == 1 else 0)
                )

                c10 = mfma_q(b0_frag, a1_frag, c10, sb0, sa1, skip_h1)

                ctx["b"](1, b_cur1, kk)
                if const_expr(k == 0):
                    store_prev(3)
                _wait_barrier_seen(
                    3 * N_LDS_STEPS + n_sc + (4 * QUAD_STORES if k == 0 else 0)
                )
                if const_expr(k == 0):
                    store_c.stash(routes, parity)
                    nxt_expert = _sgpr(nxt_eid)
                    nxt_mask = pad_mask(nxt_pad)

                c11 = mfma_q(b1_frag, a1_frag, c11, sb1, sa1, skip_h1)

                a_cur0, a_next0 = a_next0, a_cur0
                a_cur1, a_next1 = a_next1, a_cur1
                b_cur0, b_next0 = b_next0, b_cur0
                b_cur1, b_next1 = b_next1, b_cur1

            return [c00, c01, c10, c11], nxt_expert, nxt_mask

        if const_expr(gather):
            n_tokens_s1 = _sgpr(_load_i32(nv_div, 1))
            ids_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS32b(), 32)
            a_gather = G2SGather(a_div, wave_id)

        def ids_dma(tile, slot):
            """``tile``'s 256 sorted_ids into ``I_lds`` slot ``slot``, a dword a lane (waves 4..7 repeat 0..3)."""
            row0 = _pow2_div(tile, NB) * BLOCK_M
            q = wave_id % 4
            fx.copy(
                ids_atom,
                fx.slice(ids_div, (None, row0 + q * 64 + lane_id)),
                _lds_dst(lds.I_lds.ptr, slot * BLOCK_M + q * 64),
            )

        def ids_at(idx):
            return Vec(
                fx.make_view(
                    fx.add_offset(lds.I_lds.ptr, fx.make_int_tuple(idx)),
                    fx.make_layout(1, 1),
                ).load()
            )[0]

        def make_ctx_s1(tile, expert, slot, lazy):
            """Stage 1's loaders of one tile, its ids in ``I_lds`` slot ``slot`` (also its scale slot).

            ``lazy``: each A load reads its two tokens from LDS when issued;
            else the tile's four A offsets are read once, here, and held.
            """
            m_blk = _pow2_div(tile, NB)
            nb = _pow2_mod(tile, NB)

            def a_offs(h):
                lane = fx.thread_idx.x % 64
                row = lane // 8 + wave_id * 8
                a_ids = slot * BLOCK_M + row
                _, col = swizzle_128(row, (lane % 8) * 16)
                # Row r * 64 further on: the same swizzled column.
                return [
                    (ids_at(a_ids + h * LDS_BLOCK_M + r * 64) & 0xFFFFFF) * K + col
                    for r in range_constexpr(N_LDS_STEPS)
                ]

            offs = None if lazy else [a_offs(h) for h in range_constexpr(2)]

            def a_load(h, dst, k):
                a_gather.load(dst, a_offs(h) if lazy else offs[h], k * BLOCK_K)

            b_buf = fx.rocdl.make_buffer_tensor(
                fx.make_view(
                    b_iter + fx.Int64(expert) * (N * K), fx.make_layout(N * K, 1)
                ),
                max_size=False,
                num_records_bytes=N * K,
            )
            b_g2s = G2SRounds(
                fx.logical_divide(b_buf, fx.make_layout(1, 1)),
                gl_off_b[0],
                N_LDS_STEPS,
                F8_IR_t,
                wave_id,
                64 * K,
            )
            b_off = [half_row0(nb, h) * K for h in range_constexpr(2)]
            b_rows = [
                expert * (N // 128) + _pow2_div(half_row0(nb, h), 128)
                for h in range_constexpr(2)
            ]

            def b_load(h, dst, k):
                b_g2s.load(dst, b_off[h] + k * B_K_STEP)

            def sc_dma():
                row = wave_id % 4 * 64 + lane_id
                tok = ids_at(slot * BLOCK_M + row) & 0xFFFFFF
                gsc.dma(slot, tok, b_rows)

            return {"a": a_load, "b": b_load, "sc": sc_dma, "m_blk": m_blk, "nb": nb}

        # Epilogue VMEM per wave: an 8-byte act store and a 1-byte scale store per row tile.
        EPI_STORES = 2 * 2 * N_TILES_A

        def s1_pad_mask(slot):
            """Bit q - 1 (q = 1..3): rows 64 q .. 255 of the tile are padding (``half_skip``).

            Reads ids other waves DMA'd into ``I_lds`` slot ``slot``: call it
            many barriers after that DMA, as ``s_barrier`` has no memory
            semantics and a load can move above it.
            """
            msk = fx.Int32(0)
            base = slot * BLOCK_M
            for q in range_constexpr(1, 4):
                pad = (ids_at(base + 64 * q) & 0xFFFFFF) >= n_tokens_s1
                msk = msk | pad.select(fx.Int32(1 << (q - 1)), fx.Int32(0))
            return _sgpr(msk)

        def run_tile_stream_s1(tile, expert, pmask, next_tile, parity):
            """Stage 1: K-steps 0 .. K - 1 of ``tile``, then its SwiGLU epilogue.

            As stage 2's stream, the last two steps load ``next_tile``'s steps
            0 and 1 and step K - 4 its scales; step 0 loads its expert id and
            DMAs its sorted_ids into the other parity's ``I_lds`` slot. The
            epilogue stays at the tile's end (SwiGLU needs every accumulator).
            ``pmask``: ``s1_pad_mask``'s bits. Returns the next tile's expert
            and mask.
            """
            a_cur0, a_cur1, a_next0, a_next1, b_cur0, b_cur1, b_next0, b_next1 = (
                LDS_BUFS
            )
            cur = make_ctx_s1(tile, expert, parity, False)
            # This wave's rows of row half 1: 128 + wave_m * 64 + [0, 64).
            skip_h1 = (pmask >> (wave_m + 1)) & 1 if half_skip else None
            nxt = None
            nxt_expert = None
            nxt_mask = pmask
            c00 = [mfma.zero_value] * (N_TILES_A * N_TILES_B)
            c01 = list(c00)
            c10 = list(c00)
            c11 = list(c00)

            def src(kk):
                return (cur, kk) if kk < K_ITERS else (nxt, kk - K_ITERS)

            sc_a, sc_b = gsc.bases(parity)

            def sc_read(k, which):
                op, h = int(which[0] == "b"), int(which[1])
                if const_expr(op == 1):
                    return gsc.read_b(sc_b, k, h, N_TILES_B)
                return gsc.read_a(sc_a, k, h * LDS_BLOCK_M, N_TILES_A)

            for k in range_constexpr(K_ITERS):
                if const_expr(k == K_ITERS - 4):
                    if const_expr(half_skip):
                        nxt_mask = s1_pad_mask(1 - parity)
                    nxt = make_ctx_s1(next_tile, nxt_expert, 1 - parity, True)
                sa0 = sc_read(k, "a0")
                sb0 = sc_read(k, "b0")
                b0_frag = b_s2r.load(b_cur0)
                a0_frag = a_s2r.load(a_cur0)
                if const_expr(k == 0):
                    # Older than this step's A1 load: its end wait drains them.
                    nxt_eid = _load_i32_lane(eids_div, _pow2_div(next_tile, NB))
                    ids_dma(next_tile, 1 - parity)
                ctx, kk = src(k + 1)
                ctx["a"](1, a_next1, kk)
                rocdl.s_barrier()
                # The next tile's scales go to the other parity's slot, which
                # the previous tile finished reading tiles ago.
                sc_dma = k == K_ITERS - 4
                if const_expr(sc_dma):
                    nxt["sc"]()

                c00 = mfma.call(b0_frag, a0_frag, c00, sb0, sa0, k_pack=0)

                sb1 = sc_read(k, "b1")
                b1_frag = b_s2r.load(b_cur1)
                ctx, kk = src(k + 2)
                ctx["b"](0, b_cur0, kk)
                rocdl.s_barrier()

                c01 = mfma.call(b1_frag, a0_frag, c01, sb1, sa0, k_pack=0)

                sa1 = sc_read(k, "a1")
                a1_frag = a_s2r.load(a_cur1)
                ctx["a"](0, a_cur0, kk)
                n_sc = 2 if sc_dma else 0
                # vmcnt is in order: step 0's middle wait must not drain the
                # previous tile's epilogue stores or this step's two fetches,
                # all issued after K-step 1's B1 load.
                _wait_barrier_seen(
                    4 * N_LDS_STEPS + n_sc + (EPI_STORES + 2 if k == 0 else 0)
                )

                if const_expr(half_skip):
                    # Its burst's closing barrier stays outside the branch.
                    c10 = mfma_q(b0_frag, a1_frag, c10, sb0, sa1, skip_h1)
                else:
                    c10 = mfma.call(b0_frag, a1_frag, c10, sb0, sa1, k_pack=0)

                ctx["b"](1, b_cur1, kk)
                _wait_barrier_seen(3 * N_LDS_STEPS + n_sc)
                if const_expr(k == 0):
                    nxt_expert = _sgpr(nxt_eid)

                if const_expr(half_skip):
                    c11 = mfma_q(b1_frag, a1_frag, c11, sb1, sa1, skip_h1)
                else:
                    c11 = mfma.call(b1_frag, a1_frag, c11, sb1, sa1, k_pack=0)

                a_cur0, a_next0 = a_next0, a_cur0
                a_cur1, a_next1 = a_next1, a_cur1
                b_cur0, b_next0 = b_next0, b_cur0
                b_cur1, b_next1 = b_next1, b_cur1

            store_c.store([(c00, c01), (c10, c11)], cur["m_blk"] * BLOCK_M, cur["nb"])
            return nxt_expert, nxt_mask

        grid = fx.Int32(fx.grid_dim.x)
        bid = fx.Int32(fx.block_idx.x)
        start = (bid % NUM_XCDS) * (grid // NUM_XCDS) + bid // NUM_XCDS
        n_tiles = _sgpr((_load_i32(nv_div, 0) // BLOCK_M) * NB)
        if const_expr(not gather):
            # Prologue: the first tile's scales and K-steps 0 and 1, as the
            # stream's steps -4, -2 and -1 would have loaded them. A workgroup
            # without a tile runs it on tile 0 of expert 0: sorted_expert_ids
            # past the last valid tile is unwritten, and a garbage expert would
            # rebase the weight descriptor anywhere.
            has_tile = start < n_tiles
            first_tile = has_tile.select(start, fx.Int32(0))
            first_expert = has_tile.select(
                _load_i32(eids_div, _pow2_div(first_tile, NB)), fx.Int32(0)
            )
            first_mask = pad_mask(pad_rows(first_tile))
            first = make_ctx(first_tile, first_expert, first_mask)
            first["sc"](0)
            a_cur0, a_cur1, a_next0, _, b_cur0, b_cur1, b_next0, b_next1 = LDS_BUFS
            first["b"](0, b_cur0, 0)
            first["a"](0, a_cur0, 0)
            first["b"](1, b_cur1, 0)
            first["a"](1, a_cur1, 0)
            # The wave_m == 1 group runs one barrier behind, for the whole kernel.
            barrier_unless(lagging)
            _wait_barrier_seen(2 * N_LDS_STEPS)
            first["b"](0, b_next0, 1)
            first["a"](0, a_next0, 1)
            first["b"](1, b_next1, 1)
            _wait_barrier_seen(3 * N_LDS_STEPS)

            n_acc = N_TILES_A * N_TILES_B
            init = [
                fx.Int32(0),
                fx.Int32(-1),
                first_expert,
                first_mask,
                fx.Int32(0),
            ] + [mfma.zero_value] * (4 * n_acc)
            for tile, state in range(start, n_tiles, grid, init=init):
                nt = fx.Int32(tile) + grid
                # The last tile streams itself again; nothing reads those loads.
                nt = (nt < n_tiles).select(nt, fx.Int32(tile))
                acc = state[5:]
                acc, nxt_expert, nxt_mask = run_tile_stream(
                    fx.Int32(tile),
                    state[2],
                    state[3],
                    nt,
                    state[0] & 1,
                    state[1],
                    [acc[g * n_acc : (g + 1) * n_acc] for g in range(4)],
                    state[4],
                )
                final = yield [
                    state[0] + 1,
                    fx.Int32(tile),
                    nxt_expert,
                    nxt_mask,
                    state[3],
                ] + [v for g in acc for v in g]
            # The last tile's stores (a workgroup without a tile sends them
            # out of bounds).
            last_slot = (final[0] + 1) & 1
            last_cols = slot_cols(_pow2_mod(final[1], NB))
            for g in range_constexpr(4):
                if const_expr(g % 2 == 0):
                    rows = store_c.rows_half(wave_m, last_slot, g // 2)
                    offs = store_c.quad_offsets(rows, final[1] >= 0)
                acc = final[5 + g * n_acc : 5 + (g + 1) * n_acc]
                store_c.store(store_c.pack(acc), offs, last_cols[g % 2])
            # No LDS-DMA may land after the workgroup exits; the leading
            # group's barrier pairs the lagging group's prologue one.
            fx.rocdl.s_waitcnt(vmcnt=0)
            barrier_unless(leading)
        else:
            # Prologue: the first tile's ids, scales and K-steps 0 and 1, as
            # the previous tile's stream would have loaded them (a workgroup
            # without a tile runs it on tile 0 of expert 0, as stage 2 does).
            has_tile = start < n_tiles
            first_tile = has_tile.select(start, fx.Int32(0))
            first_expert = has_tile.select(
                _load_i32(eids_div, _pow2_div(first_tile, NB)), fx.Int32(0)
            )
            ids_dma(first_tile, 0)
            # A barrier with memory semantics: the mask's ids reads must not
            # move above it (the other waves' DMA wrote them).
            fx.rocdl.s_waitcnt(vmcnt=0)
            fx.gpu.barrier()
            first_mask = s1_pad_mask(fx.Int32(0)) if half_skip else fx.Int32(0)
            first = make_ctx_s1(first_tile, first_expert, fx.Int32(0), False)
            first["sc"]()
            a_cur0, a_cur1, a_next0, _, b_cur0, b_cur1, b_next0, b_next1 = LDS_BUFS
            first["b"](0, b_cur0, 0)
            first["a"](0, a_cur0, 0)
            first["b"](1, b_cur1, 0)
            first["a"](1, a_cur1, 0)
            # The wave_m == 1 group runs one barrier behind, for the whole kernel.
            barrier_unless(lagging)
            _wait_barrier_seen(2 * N_LDS_STEPS)
            first["b"](0, b_next0, 1)
            first["a"](0, a_next0, 1)
            first["b"](1, b_next1, 1)
            # Drained, so step 0's waits (counted for the previous tile's
            # epilogue stores) also hold for the first tile.
            _wait_barrier_seen(0)
            # A while loop keeps the carried state and the bound i32: an
            # init= range loop is index (i64), its bound compare needs a VGPR
            # pair, which spilled, and the reload's vmcnt(0) at the tile's end
            # drained the epilogue stores and the streamed next-tile loads.
            tile = start
            expert = first_expert
            pmask = first_mask
            count = fx.Int32(0)
            while tile < n_tiles:
                nt = tile + grid
                # The last tile streams itself again; nothing reads those loads.
                nt = (nt < n_tiles).select(nt, tile)
                expert, pmask = run_tile_stream_s1(tile, expert, pmask, nt, count & 1)
                count = count + 1
                tile = tile + grid
            # No LDS-DMA may land after the workgroup exits; the leading
            # group's barrier pairs the lagging group's prologue one.
            fx.rocdl.s_waitcnt(vmcnt=0)
            barrier_unless(leading)

    @flyc.jit
    def launch(
        A: fx.Tensor,
        B: fx.Pointer,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        C_scale: fx.Tensor,
        m_pad: fx.Int32,
        limit: fx.Float32,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_gemm(
            A,
            B,
            C,
            A_scale,
            B_scale,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            C_scale,
            m_pad,
            limit,
            value_attrs={
                # 512 threads put two waves on every SIMD.
                "rocdl.waves_per_eu": 2,
                "rocdl.flat_work_group_size": "512,512",
            },
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    if const_expr(gather):
        # Stage 1 compiles without MachineLICM: hoisted out of the tile loop,
        # its lane terms and the A gather's constant K offsets were held
        # across the K loop and spilled, and the reloads' vmcnt(0) drained the
        # streamed loads. Stage 2 keeps the pass (1-2% slower without it).
        launch.compile_hints["llvm_options"] = {"disable-machine-licm": True}
    return launch
