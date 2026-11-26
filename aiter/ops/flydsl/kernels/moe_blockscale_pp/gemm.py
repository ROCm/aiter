# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Persistent 8-wave block-FP8 grouped GEMM over AITER's ``moe_sorting`` tiles.

The K pipeline is ``mxfp8_gemm_8wave``'s (256x256x128, 8 waves = 2 (M) x 4 (N),
the wave_m == 1 group one barrier behind, ``s_setprio`` around each MFMA burst,
128 KiB of A/B LDS double buffers, E8M0 scales staged through LDS four K-steps
at a time). What changes:

* Row tile ``m`` of the sorted list belongs to expert ``sorted_expert_ids[m]``;
  each tile rebases the weight descriptor to that expert, so the 32-bit buffer
  offsets never span the whole ``[E, N, K]`` tensor (2.4 GB for 289 experts).
* ``gather=True`` (stage 1) reads A rows straight from the token-major
  activations: each lane's row offset is ``token * K`` with the token taken from
  ``sorted_token_ids`` (padding rows name token ``T``, past the descriptor, so
  they load zeros), and the 1x128 A scales likewise from a row-major
  ``[T, K // 128]``. ``gather=False`` (stage 2) reads the sorted rows
  contiguously with a column-major ``[K // 128, M_pad]`` scale, as the dense kernel.
* The two 128-row B halves of a tile are independent weight row blocks: for
  stage 1 they are the gate rows ``[n * 128, +128)`` and the up rows
  ``[I + n * 128, +128)``, so every wave holds 32 gate and the matching 32 up
  columns; for stage 2 they are the two halves of a 256-row block.
* A persistent grid (one workgroup per CU) walks the tiles, the number of which
  it reads on device from ``num_valid_ids``. Workgroup ``w`` (on XCD ``w % 8``)
  starts at ``(w % 8) * G / 8 + w / 8`` and strides by ``G``, so each XCD runs
  32 consecutive tiles at a time: N-blocks of a row tile are adjacent, so the
  A tile and an expert's weight panel stay in that XCD's L2.
* Stage 2 (K = 1024, 8 K-steps a tile) runs one K-step stream across its
  tiles: the last two K-steps load the next tile's first two, so the
  pipeline never drains, and a tile's output is converted and stored in the
  next tile's first K-step, a quadrant at a time (``run_tile_stream``).
* Stage 1 (K = 4096) streams its loads the same way (``run_tile_stream_s1``):
  the next tile's sorted_ids are DMA'd into LDS in step 0, its scales in
  step K - 4 and its K-steps 0 and 1 in the last two steps, and its SwiGLU
  epilogue runs at the tile's end while those loads land.
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
    wait_barrier,
)
from aiter.ops.flydsl.kernels.mxfp8_gemm_8wave import (
    MxMfma,
    barrier_if_zero,
)
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
    return fx.Int32(
        rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(reg.load()[0]).ir_value())
    )


def _sgpr(v):
    """A wave-uniform value moved to an SGPR.

    Wave-derived terms (wave_m, wave_n, the tile count) left in VGPRs
    competed with the 128 accumulators and spilled; their reloads' vmcnt
    waits drained LDS-DMA and epilogue stores (ATT).
    """
    return fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(v).ir_value()))


def _pow2_div(x, d):
    """``x // d`` for ``x >= 0`` and a power-of-two ``d``: a shift (``//`` adds signed floor fixups, ISA)."""
    assert d & (d - 1) == 0
    return fx.Int32(x) >> (d.bit_length() - 1)


def _pow2_mod(x, d):
    """``x % d`` for a power-of-two ``d`` as a mask (floor-mod for negative ``x`` too)."""
    assert d & (d - 1) == 0
    return fx.Int32(x) & (d - 1)


def _load_i32_lane(div, idx):
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
    reg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
    fx.copy(atom, fx.slice(div, (None, fx.Int32(idx))), reg)
    return fx.Int32(reg.load()[0])


def _lane_col8(lane=None):
    """Column (in elements) of the 8 a lane holds after the pair swap, within a 32-column pair."""
    q = (fx.thread_idx.x % 64 if lane is None else lane) // 16
    return (q % 2 == 1).select(16 + 4 * (q - 1), 4 * q)


def _fresh_lane():
    """``thread_idx.x % 64`` recomputed where it is used.

    A side-effecting move, so LICM cannot hoist it (or anything derived from
    it) out of the tile loop: values used only in the stage-1 epilogue were
    otherwise held across the K loop, spilled, and their reloads' vmcnt(0)
    drained the next tile's streamed loads (ISA). Inline asm: no stable API
    pins a value in place.
    """
    tid = _llvm.inline_asm(
        res=fx.Int32.ir_type,
        operands_=[fx.Int32(fx.thread_idx.x).ir_value()],
        asm_string="v_mov_b32 $0, $1",
        constraints="=v,v",
        has_side_effects=True,
    )
    return fx.Int32(tid) % 64


def _swap_pair(x, y):
    """``permlane16_swap`` of two i32 words: (lower-row word, upper-row word) in every lane.

    No fx-level cross-lane swap exists; this is one v_permlane16_swap_b32.
    """
    i32 = fx.Int32.ir_type
    pair = rocdl.permlane16_swap(
        _llvm.StructType.get_literal([i32, i32]),
        fx.Int32(x).ir_value(),
        fx.Int32(y).ir_value(),
        False,
        False,
    )
    return [fx.Int32(_llvm.extractvalue(i32, pair, [h])) for h in (0, 1)]


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


class StoreBf16Rows:
    """bf16 stores of the swapped-operand accumulators, 16 bytes per lane (probe output).

    With the weights as the MFMA's A operand, lane ``l`` of a 16x16 tile holds
    output row ``l % 16`` at columns ``4 * (l // 16) + [0, 4)``. One
    ``permlane16_swap`` between the two N-tiles of a pair gives every lane
    eight consecutive columns: even 16-lane rows q get tile 0's columns
    ``4q .. 4q + 7``, odd ones tile 1's ``4(q - 1) .. 4(q - 1) + 7`` (HK's
    ``store_acc_rows``).
    """

    def __init__(self, C, c_rows, ldc):
        self.ldc = ldc
        self.lane_row = fx.thread_idx.x % 16
        self.lane_col = _lane_col8()
        self.c_div = _i32_view(C, c_rows * ldc * 2, c_rows * (ldc // 2), 16)
        self.atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        self.reg = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)

    def store(self, c_frag, base_row, base_col):
        """``c_frag[j * N_TILES_A + i]``: N-tile j (of 2), row tile i."""
        for i in range_constexpr(N_TILES_A):
            x = Vec(c_frag[i]).to(fx.BFloat16).bitcast(fx.Int32)
            y = Vec(c_frag[N_TILES_A + i]).to(fx.BFloat16).bitcast(fx.Int32)
            lo = _swap_pair(x[0], y[0])
            hi = _swap_pair(x[1], y[1])
            fx.memref_store_vec(
                Vec.from_elements([lo[0], hi[0], lo[1], hi[1]], fx.Int32), self.reg
            )
            row = base_row + i * 16 + self.lane_row
            idx = (row * self.ldc + base_col + self.lane_col) // 2  # in i32 words
            fx.copy(self.atom, self.reg, fx.slice(self.c_div, (None, idx)))


class StoreSlotBf16:
    """Stage-2 epilogue: bf16 rows scattered to ``[T * topk, D]``.

    Sorted row r goes to row ``token * topk + slot`` (both from
    ``sorted_ids[r]``). With ``w_div`` (``sorted_weights``) the row is first
    multiplied by its routing weight in fp32, so a plain top-k sum finishes
    the layer; without, the reduction applies the weights (HK's stage-2
    split), and the pack loses a v_pk_mul_f32 per two values: the store
    phases of a tile's first K-step are VALU-issue bound (ATT). Padding rows
    name token T, which lands past the descriptor: their stores are dropped
    without a branch.

    The tile's 256 routes are loaded in its first K-step (``fetch``, one or two
    loads a lane, 32 rows a wave), parked in ``route`` LDS once that step's
    waits have passed (``stash``) and read back when the next tile stores
    this one (``rows_half``). Loaded at the epilogue instead, their latency
    sat behind a vmcnt(0) on every tile: 25% of the kernel (stage 2 453 ->
    342 us at T=4096 with constant routes). ``route`` is double-buffered by
    tile parity: a tile's routes are read while the next tile's are parked.
    """

    def __init__(self, C, n_tokens, topk, D, ids_div, w_div, route):
        self.D = D
        self.topk = topk
        self.ids_div = ids_div
        self.w_div = w_div
        self.weighted = w_div is not None
        self.n_fetch = 2 if self.weighted else 1
        self.route = route
        self.wave_row = _sgpr(fx.thread_idx.x // 64) * 32 + fx.thread_idx.x % 32
        self.lane_row = fx.thread_idx.x % 16
        n_rows = n_tokens * topk
        self.c_div = _i32_view(C, n_rows * D * 2, n_rows * (D // 2), 16)
        # One past the last word: a dropped store's offset (2**30 words, as
        # before, wrapped to byte 0 in the 32-bit offset and wrote row 0).
        self.oob = n_rows * (D // 2)
        self.atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        self.reg = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
        # Quad order: lane 4a + b stores 8-column chunk b of row a, so every 4
        # consecutive lanes write one contiguous 64 bytes. The TA splits a
        # store by lane quads: in the MFMA order (lane 16q + r holds row r)
        # each quad touches 4 rows, and 16 rows x 64 B moved ~17 B/clk a CU
        # against ~47 B/clk in quad order (microbenchmark, random rows either
        # way). Lane 4a + b takes the words of lane 16q + a, q the lane group
        # holding chunk b (_lane_col8: q = 0, 2, 1, 3 for chunks 0..3); every
        # q of lane group a holds row a, so lane 4a + b reads row a's route
        # itself (``rows_half``) instead of permuting lane a's.
        lane = fx.thread_idx.x % 64
        b = lane % 4
        cinv = (b == 1).select(2, (b == 2).select(1, b))
        self.perm = (cinv * 16 + lane // 4) * 4
        self.quad_col = b * 8

    def _bperm(self, v):
        """``v`` of lane ``perm`` (no fx-level cross-lane gather: ds_bpermute)."""
        return fx.Int32(
            rocdl.ds_bpermute(
                fx.Int32.ir_type, fx.Int32(self.perm).ir_value(), fx.Int32(v).ir_value()
            )
        )

    def _at(self, idx):
        return fx.make_view(
            fx.add_offset(self.route.ptr, fx.make_int_tuple(idx)), fx.make_layout(1, 1)
        )

    def fetch(self, row0):
        """Issue this lane's share of the tile's route loads (row ``wave_row``)."""
        r = row0 + self.wave_row
        if const_expr(not self.weighted):
            return (_load_i32_lane(self.ids_div, r),)
        return _load_i32_lane(self.ids_div, r), _load_i32_lane(self.w_div, r)

    def stash(self, fetched, slot):
        # Lanes l and l + 32 hold the same row and store the same values.
        base = slot * 512 + self.wave_row
        fx.memref_store_vec(fx.full(1, fetched[0], fx.Int32), self._at(base))
        if const_expr(self.weighted):
            fx.memref_store_vec(
                fx.full(1, fetched[1], fx.Int32), self._at(base + 256)
            )

    def rows_half(self, wave_m, slot, h):
        """Per row tile of row half ``h``: (output row in quad order, routing weight in MFMA order).

        Every read is a constant offset from one opaque base. Formed as
        ``slot * 512 + h * 128 + ...`` per read, the row half 1 reads lost the
        route field's alias scope (the half 0 address plus a constant: a
        deeper GEP chain), so the first drew an s_waitcnt vmcnt(0) behind the
        streamed next-tile LDS-DMA (ISA).
        """
        tile0 = slot * 512 + wave_m * A_GRP_ROWS
        base = _opaque_vgpr(tile0 + fx.thread_idx.x % 64 // 4)
        if const_expr(self.weighted):
            w_base = _opaque_vgpr(tile0 + self.lane_row + 256)
        out = []
        for i in range_constexpr(N_TILES_A):
            off = h * LDS_BLOCK_M + i * 16
            sid = Vec(self._at(base + off).load())[0]
            w = None
            if const_expr(self.weighted):
                w = Vec(self._at(w_base + off).load())[0].bitcast(fx.Float32)
            out.append(((sid & 0xFFFFFF) * self.topk + (sid >> 24), w))
        return out

    def quad_offsets(self, rows, valid):
        """Per row tile: this lane's quad-order i32-word offset at column 0 (``oob`` if not ``valid``)."""
        offs = []
        for i in range_constexpr(N_TILES_A):
            off = rows[i][0] * (self.D // 2) + self.quad_col // 2
            offs.append(valid.select(off, self.oob))
        return offs

    def pack(self, c_frag, rows):
        """(Weighted) bf16 words of ``c_frag`` (``[j * N_TILES_A + i]``) in quad order: 4 per row tile."""
        words = []
        for i in range_constexpr(N_TILES_A):
            x, y = Vec(c_frag[i]), Vec(c_frag[N_TILES_A + i])
            if const_expr(self.weighted):
                x, y = x * rows[i][1], y * rows[i][1]
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
    ``limit`` > 0 clamps as ``silu(min(g, L)) * clamp(u, -L, L)``; the host
    passes float max for "no clamp".
    """

    def __init__(self, C, C_scale, m_pad, inter, limit, amax_lds, wave_m, wave_n):
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
        for half in (32, 16):
            if const_expr(half == 32):
                i32 = fx.Int32.ir_type
                pair = rocdl.permlane32_swap(
                    _llvm.StructType.get_literal([i32, i32]),
                    v.bitcast(fx.Int32).ir_value(),
                    v.bitcast(fx.Int32).ir_value(),
                    False,
                    False,
                )
                o = [fx.Int32(_llvm.extractvalue(i32, pair, [h])) for h in (0, 1)]
            else:
                o = _swap_pair(v.bitcast(fx.Int32), v.bitcast(fx.Int32))
            v = o[0].bitcast(fx.Float32).maximumf(o[1].bitcast(fx.Float32))
        return v

    def store(self, gate_up, row0, nb):
        """``gate_up[h] = (gate accs, up accs)`` of row half h; acc ``j * N_TILES_A + i``."""
        lane = _fresh_lane()
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
                    vals = []
                    for v in range_constexpr(4):
                        g, u = clamp_gate_up(g4[v], u4[v], self.neg_limit)
                        a = g * sigmoid_f32(g) * u
                        vals.append(a)
                        m = m.maximumf(fx.math.absf(a))
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
                inv = ((254 - e) << 23).bitcast(fx.Float32)
                words = []
                for j in range_constexpr(2):
                    vals = acts[(h, i, j)]
                    q8 = []
                    for p in range_constexpr(2):
                        # The scaled value can exceed 448 by one ulp of the division.
                        lo = (
                            (vals[2 * p] * inv)
                            .maximumf(fx.Float32(-448.0))
                            .minimumf(fx.Float32(448.0))
                        )
                        hi = (
                            (vals[2 * p + 1] * inv)
                            .maximumf(fx.Float32(-448.0))
                            .minimumf(fx.Float32(448.0))
                        )
                        q8.append((lo, hi))
                    w = fx.Int32(
                        rocdl.cvt_pk_fp8_f32(
                            fx.Int32.ir_type,
                            q8[0][0].ir_value(),
                            q8[0][1].ir_value(),
                            fx.Int32(0).ir_value(),
                            False,
                        )
                    )
                    w = fx.Int32(
                        rocdl.cvt_pk_fp8_f32(
                            fx.Int32.ir_type,
                            q8[1][0].ir_value(),
                            q8[1][1].ir_value(),
                            w.ir_value(),
                            True,
                        )
                    )
                    words.append(w)
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
    """``wait_barrier`` as intrinsics: the compiler's waitcnt pass sees the vmcnt.

    Behind inline asm the pass assumed the stage-1 sorted_ids LDS-DMA (issued
    at step 0) still in flight when step K - 4 read it back, and drained
    every streamed load with a vmcnt(0) there (~550 cycles a tile, ATT).
    """
    fx.rocdl.s_waitcnt(vmcnt=count)
    rocdl.s_barrier()


def _opaque_vgpr(value):
    """``value`` in a VGPR the compiler cannot fold constants through.

    Inline asm, as no stable API hides a value from constant folding. See
    ``LdsFragReader`` for why the operand reads need it.
    """
    return fx.Int32(
        _llvm.inline_asm(
            res=fx.Int32.ir_type,
            operands_=[fx.Int32(value).ir_value()],
            asm_string="v_mov_b32 $0, $1",
            constraints="=v,v",
            has_side_effects=False,
        )
    )


class LdsFragReader:
    """MFMA operand fragments from one LDS region, one address VGPR per K half.

    Tile ``i`` sits exactly ``i * 2048`` bytes past tile 0, in the plain (A)
    and the ``shuffle_weight`` (16, 16) (B) layouts alike (the swizzle only
    flips bits 4-6 from row bits 1-3), and a region's four 16 KB buffers lie
    within 64 KB of its first one. Left alone, the compiler folds each
    buffer's link-time address into its reads; past the 64 KB ds_read
    immediate it then kept one VGPR per buffer, tile and K half (16 for B,
    ISA) and spilled them, every reload draining the LDS-DMA prefetch with
    an s_waitcnt vmcnt. Here the address is ``field + (opaque(lane + bias) -
    bias)`` with ``bias`` the region's address: the compiler folds ``field -
    bias + tile`` (< 64 KB) into the immediate, and the pointer still derives
    from the struct field, whose alias scope keeps these reads from waiting
    on LDS-DMA writes to other buffers (an inttoptr address loses it, and
    every read then waited for vmcnt(0), ISA).
    """

    TILE_BYTES = 16 * BLOCK_K

    def __init__(self, region, wave_idx, n_tiles, preshuffled):
        lane = fx.thread_idx.x % 64
        row = wave_idx * (n_tiles * 16) + lane % 16
        col = (lane // 16) * 16
        self.n_tiles = n_tiles
        bias = fx.Int32(fx.ptrtoint(region.ptr))
        if const_expr(preshuffled):
            off = (row // 8) * 1024 + (row % 8) * 16 + (col // 16) * 128
            rel = _opaque_vgpr(off + bias) - bias
            # K half 1 is 64 columns = 4 x 16-byte groups further on: an
            # immediate, not a second live VGPR.
            self.rel = [rel, rel + 4 * 128]
        else:
            offs = []
            for step in range_constexpr(2):
                r, c = swizzle_128(row, col + step * 64)
                offs.append(r * 128 + c)
            self.rel = [_opaque_vgpr(o + bias) - bias for o in offs]

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

    def load(self, lds_dst, k_offset, skip=(None, None)):
        for step in range_constexpr(self.n_load_steps):
            step_off = self.wave_id * 1024 + step * (self.n_waves * 1024)
            base_i32 = fx.Int32(fx.ptrtoint(lds_dst.ptr))
            lds_ptr = fx.inttoptr(self.LdsPtr_t, base_i32 + fx.Int32(step_off))
            soff = fx.Int32(k_offset) + step * self.round_bytes
            if const_expr(skip[step] is not None):
                soff = (skip[step] != 0).select(fx.Int32(1 << 30), soff)
            rocdl.raw_ptr_buffer_load_lds(
                self.rsrc,
                fx.to_llvm_ptr(lds_ptr),
                16,
                fx.Int32(self.gl_offsets[0]).ir_value(),
                soff.ir_value(),
                0,
                self.NT,
            )


def _sgpr_here(c):
    """The constant ``c`` in an SGPR materialized where it is used (an ``s_mov`` per use).

    Side-effecting inline asm, so LICM cannot hoist it: as plain constants,
    the 31 K offsets of the stage-1 gather loads were all hoisted out of the
    tile loop into SGPRs and the SGPR file spilled (24 spills, ISA). No
    stable API pins a value in place.
    """
    return fx.Int32(
        _llvm.inline_asm(
            res=fx.Int32.ir_type,
            operands_=[],
            asm_string=f"s_mov_b32 $0, {int(c)}",
            constraints="=s",
            has_side_effects=True,
        )
    )


class G2SGather(G2SLoader):
    """Gathered A rows: one lane offset per round, the K offset an SGPR set at the load."""

    def __init__(self, gl_src, wave_id):
        super().__init__(gl_src, [], N_LDS_STEPS, fx.Float8E4M3FN.ir_type, wave_id)

    def load(self, lds_dst, offsets, k_offset):
        soff = _sgpr_here(k_offset) if k_offset else fx.Int32(0)
        for step in range_constexpr(len(offsets)):
            src = fx.slice(self.gl_src, (None, fx.Int32(offsets[step])))
            fx.copy(self.g2lds_atom, src, self._lds_dst_at(lds_dst, step), soffset=soff)


class MoeScaleLds:
    """E8M0 A / B block scales of one tile, staged through LDS ``CHUNK`` K-steps at a time.

    The LDS image is ``ScaleRawLds``'s 1x128 / 128x128 one, so the reads are
    unchanged: A slot ``c * 256 + row`` (a byte per dword) holds column ``c`` of
    the chunk for tile row ``row``; B slots 0 / 1 hold a dword of K // 128
    scales of the half-0 / half-1 weight row block. Every operand issues two
    copies per chunk, as there, so the pipeline's vmcnt counts hold.

    ``a_offs`` (per step, per lane) and ``a_k_stride`` locate the A bytes:
    gathered row-major ``[T, K // 128]`` (``tok * k128 + c``, stride 1) or
    column-major ``[K // 128, M_pad]`` (``c * M_pad + row``, stride M_pad).
    ``b_offs`` are the two B scale rows' byte offsets (``row * k128``).
    """

    CHUNK = 4
    BYTES = 2 * 256 * 16

    def __init__(
        self, a_rsrc, a_offs, a_k_stride, b_div, b_offs, k128, wave_id, a_row, lds
    ):
        self.a_rsrc = a_rsrc
        self.a_offs = a_offs
        self.a_k_stride = a_k_stride
        self.b_div = b_div
        self.b_offs = b_offs
        self.k128 = k128
        self.wave_id = wave_id
        self.lds = lds
        lane = fx.thread_idx.x % 64
        self.atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS32b(), 32)
        self.LdsPtr_t = fx.PointerType.get(fx.Float8E4M3FN.ir_type, 2, 512)
        self.bases = [
            fx.add_offset(lds.ptr, fx.make_int_tuple((a_row + lane % 16) * 4)),
            fx.add_offset(lds.ptr, fx.make_int_tuple(4096)),
        ]

    def dma(self, chunk):
        """Stage chunk ``chunk``; the last chunk may run past K (those bytes are never read)."""
        base_i32 = fx.Int32(fx.ptrtoint(self.lds.ptr))
        for step in range_constexpr(2):
            dst_off = chunk % 2 * self.BYTES + step * 2048 + self.wave_id * 256
            lds_dst = fx.inttoptr(self.LdsPtr_t, base_i32 + fx.Int32(dst_off))
            # A byte per lane into its own dword slot: neither the gathered
            # token rows nor the column offsets c * M_pad are dword aligned.
            rocdl.buffer_load_to_lds(
                self.a_rsrc,
                fx.to_llvm_ptr(lds_dst),
                fx.Int32(self.a_offs[step]).ir_value(),
                size_bytes=1,
                soffset=(fx.Int32(chunk * self.CHUNK) * self.a_k_stride).ir_value(),
            )
        for step in range_constexpr(2):
            dst_off = chunk % 2 * self.BYTES + 4096 + step * 2048 + self.wave_id * 256
            lds_dst = fx.inttoptr(self.LdsPtr_t, base_i32 + fx.Int32(dst_off))
            dst = fx.make_view(lds_dst, fx.make_layout(1, 1))
            # The dword never runs past the row's end (K // 128 >= 4).
            fx.copy(
                self.atom,
                fx.slice(self.b_div, (None, fx.Int32(self.b_offs))),
                dst,
                soffset=fx.Int32(min(chunk * self.CHUNK, self.k128 - 4)),
            )

    def _byte(self, op, off):
        v = fx.make_view(
            fx.recast_iter(
                fx.Uint8, fx.add_offset(self.bases[op], fx.make_int_tuple(off))
            ),
            fx.make_layout(1, 1),
        )
        return fx.Int32(v.load()[0])

    def read(self, op, half_row, n_tiles, k):
        """MFMA scale operands of ``n_tiles`` tiles, ``half_row`` rows past this wave's first row."""
        slot = k // self.CHUNK % 2 * self.BYTES
        if const_expr(op == 1):
            chunk = k // self.CHUNK
            start = min(chunk * self.CHUNK, self.k128 - 4)
            v = self._byte(op, slot + half_row // 128 * 4 + (k - start))
            return [v] * n_tiles
        return [
            self._byte(op, slot + ((k % self.CHUNK) * 256 + half_row + t * 16) * 4)
            for t in range_constexpr(n_tiles)
        ]


class TileScaleLds:
    """Stage 2: a tile's E8M0 A / B scales for every K block, one LDS slot per tile parity.

    A (column-major ``[K // 128, M_pad]``): wave ``w`` loads the 256 bytes of K
    block ``(w + rot) % k128`` of the tile's rows, a dword per lane, into
    position ``w``, so step ``k`` of a K-rotated tile reads position ``k``. B:
    every wave loads both weight row blocks' ``k128`` bytes twice over into
    its own 256 bytes (lanes 8..63 repeat lanes 0..7), so the read at byte
    ``rot + k`` needs no wrap. Two DMA instructions a wave a tile, where
    ``MoeScaleLds`` issues four a 4-step chunk (most of them redundant B
    copies); the TA was the stage's bottleneck (PMC: TA busy 75%).
    """

    def __init__(self, a_rsrc, b_div, k128, m_pad, lds, wave_id, a_row):
        assert k128 == N_WAVES, "one A scale block per wave"
        self.a_rsrc = a_rsrc
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
        self.LdsPtr_t = fx.PointerType.get(fx.Float8E4M3FN.ir_type, 2, 512)

    BYTES = 2 * (8 * 256 + N_WAVES * 256)

    def dma(self, slot, row0, rot, b_rows):
        """Stage a tile's scales into ``slot``; ``b_rows`` = its two weight row blocks."""
        base_i32 = fx.Int32(fx.ptrtoint(self.lds.ptr))
        slot_off = slot * self.slot_bytes + self.wave_id * 256
        dst = fx.inttoptr(self.LdsPtr_t, base_i32 + slot_off)
        blk = _pow2_mod(self.wave_id + rot, self.k128)
        rocdl.buffer_load_to_lds(
            self.a_rsrc,
            fx.to_llvm_ptr(dst),
            fx.Int32(self.lane * 4).ir_value(),
            size_bytes=4,
            soffset=(blk * self.m_pad + row0).ir_value(),
        )
        j = self.lane % 8
        off = (j < 4).select(b_rows[0], b_rows[1]) * self.k128 + j % 2 * 4
        bdst = fx.inttoptr(self.LdsPtr_t, base_i32 + slot_off + self.a_area)
        fx.copy(
            self.atom,
            fx.slice(self.b_div, (None, off)),
            fx.make_view(bdst, fx.make_layout(1, 1)),
        )

    def bases(self, slot, rot):
        """This tile's read bases: opaque offsets off the field, which keep the reads' alias scope."""
        bias = fx.Int32(fx.ptrtoint(self.lds.ptr))
        slot_off = slot * self.slot_bytes
        a = _opaque_vgpr(slot_off + self.a_lane + bias) - bias
        b_off = slot_off + self.a_area + self.wave_id * 256 + rot
        return a, _opaque_vgpr(b_off + bias) - bias

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
    rows][16]``. A read's 16 rows then sit 16 bytes apart, on distinct banks
    (in the tile's ``[256, 32]`` order they were 32 bytes apart: two-way bank
    conflicts, ~120 cycles a K-step a wave, ATT). B: every wave loads both
    weight row blocks' ``k128`` bytes (lanes 0..7 / 8..15, the rest repeat
    them) into its own 256 bytes. Chunked through ``MoeScaleLds`` instead,
    the A bytes were one 1-byte gather a lane, 32 such instructions a wave a
    tile, and every fourth K-step ran ~1.3k cycles slower (ATT, T = 16384).
    """

    def __init__(self, a_div, b_div, k128, lds, wave_id, a_row, slot_bytes):
        assert (
            k128 * BLOCK_M == N_WAVES * 64 * 16
        ), "16 bytes a lane cover the tile's A scales"
        self.a_div = a_div
        self.b_div = b_div
        self.k128 = k128
        self.lds = lds
        self.wave_id = wave_id
        self.a_area = BLOCK_M * k128
        self.slot_bytes = slot_bytes
        lane = fx.thread_idx.x % 64
        self.dma_row = wave_id % 4 * 64 + lane
        self.dma_col = wave_id // 4 * 16
        self.a_row = a_row
        self.a_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        self.b_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS32b(), 32)
        self.LdsPtr_t = fx.PointerType.get(fx.Float8E4M3FN.ir_type, 2, 512)

    BYTES_PER_SLOT = 8 * 1024 + N_WAVES * 256

    def dma(self, slot, tok, b_rows):
        """Stage a tile's scales into ``slot``: ``tok`` = token of tile row ``dma_row``, ``b_rows`` its two weight row blocks."""
        base_i32 = fx.Int32(fx.ptrtoint(self.lds.ptr))
        slot_off = slot * self.slot_bytes
        dst = fx.inttoptr(self.LdsPtr_t, base_i32 + slot_off + self.wave_id * 1024)
        fx.copy(
            self.a_atom,
            fx.slice(self.a_div, (None, tok * self.k128 + self.dma_col)),
            fx.make_view(dst, fx.make_layout(1, 1)),
        )
        j = _fresh_lane() % 16
        off = (j < 8).select(b_rows[0], b_rows[1]) * self.k128 + j % 8 * 4
        bdst = fx.inttoptr(
            self.LdsPtr_t, base_i32 + slot_off + self.a_area + self.wave_id * 256
        )
        fx.copy(
            self.b_atom,
            fx.slice(self.b_div, (None, off)),
            fx.make_view(bdst, fx.make_layout(1, 1)),
        )

    def bases(self, slot):
        """Read bases: opaque offsets off the field, which keep the reads' alias scope."""
        bias = fx.Int32(fx.ptrtoint(self.lds.ptr))
        slot_off = slot * self.slot_bytes
        a_lane = (self.a_row + _fresh_lane() % 16) * 16
        a = _opaque_vgpr(slot_off + a_lane + bias) - bias
        b_off = slot_off + self.a_area + self.wave_id * 256
        return a, _opaque_vgpr(b_off + bias) - bias

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
    gate_up_inter: int = 0,
    epilogue: str = "bf16",
    topk: int = 0,
    weighted: bool = True,
):
    """Build the persistent grouped GEMM launcher.

    ``N``: weight rows per expert (2 * I for stage 1, the model dim for stage 2).
    ``gate_up_inter`` = I > 0 pairs gate row block ``n`` with up row block
    ``I / 128 + n`` in one tile (stage 1); 0 takes 256 consecutive rows.
    ``epilogue``:

    * ``"bf16"`` (probe): the GEMM per sorted row, ``C`` bf16 ``[M_pad, N]``,
      columns in weight-row order.
    * ``"swiglu_fp8"`` (stage 1, needs ``gate_up_inter``): ``C`` fp8
      ``[M_pad, I]`` = requantized SwiGLU, ``C_scale`` its E8M0 1x128 scales
      column-major ``[I // 128, M_pad]`` (stage 2's A operand as is).
    * ``"slot_bf16"`` (stage 2): ``C`` bf16 ``[T * topk, N]``, the expert
      output of route (token, slot) in row ``token * topk + slot``, multiplied
      by its routing weight if ``weighted``.
    """
    assert epilogue in ("bf16", "swiglu_fp8", "slot_bf16"), epilogue
    assert epilogue != "swiglu_fp8" or gate_up_inter
    assert epilogue != "slot_bf16" or topk > 0
    assert K % 512 == 0, "a 128x128 B scale needs K a multiple of 512 here"
    K_ITERS = K // BLOCK_K
    assert K_ITERS % 2 == 0
    k128 = K // 128
    if gate_up_inter:
        assert N == 2 * gate_up_inter and gate_up_inter % 128 == 0
        NB = gate_up_inter // 128
    else:
        assert N % BLOCK_N == 0
        NB = N // BLOCK_N

    def half_row0(nb, h):
        """First weight row (within the expert) of LDS half ``h`` of N-block ``nb``."""
        if const_expr(gate_up_inter):
            return h * gate_up_inter + nb * LDS_BLOCK_N
        return nb * BLOCK_N + h * LDS_BLOCK_N

    # Stage 1 streams K-steps across its tiles (run_tile_stream_s1).
    stream_s1 = gather and epilogue == "swiglu_fp8"
    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * BLOCK_K
    n_chunks = (K_ITERS + MoeScaleLds.CHUNK - 1) // MoeScaleLds.CHUNK

    _kname = (
        f"flydsl_moe_bs_8w_{'gather' if gather else 'sorted'}_n{N}_k{K}"
        f"{'_gu' if gate_up_inter else ''}_{epilogue}{topk or ''}"
        f"{'_uw' if epilogue == 'slot_bf16' and not weighted else ''}"
    )

    @fx.struct
    class SharedStorage:
        if epilogue == "slot_bf16":
            S_lds: fx.Array[fx.Float8E4M3FN, TileScaleLds.BYTES, 16]
        elif gather:
            S_lds: fx.Array[fx.Float8E4M3FN, 2 * GatherTileScaleLds.BYTES_PER_SLOT, 16]
        else:
            S_lds: fx.Array[fx.Float8E4M3FN, 2 * MoeScaleLds.BYTES, 16]
        if epilogue == "swiglu_fp8":
            R_lds: fx.Array[fx.Float32, 2 * 4 * 128, 16]
            # A tile's sorted_ids, one slot per tile parity (the stage-1 stream).
            I_lds: fx.Array[fx.Int32, 2 * BLOCK_M, 16]
        if epilogue == "slot_bf16":
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
        sorted_weights: fx.Tensor,
        m_pad: fx.Int32,
        limit: fx.Float32,
    ):
        F8_IR_t = fx.Float8E4M3FN.ir_type
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        lane_id = fx.thread_idx.x % 64
        wave_id = _sgpr(fx.thread_idx.x // 64)
        wave_m = wave_id // 4
        wave_n = wave_id % 4
        lagging = (1 - wave_m).ir_value()
        leading = wave_m.ir_value()

        a_div = fx.logical_divide(
            make_fp8_buffer_tensor(A, F8_IR_t), fx.make_layout(1, 1)
        )
        a_scale_rsrc = fx.rocdl.get_buffer_rsrc(
            fx.get_iter(fx.rocdl.make_buffer_tensor(A_scale, max_size=False))
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
        if const_expr(gather):
            # Row (within a 128-row half) and swizzled column of each round.
            a_rows, a_cols = [], []
            for r in range_constexpr(N_LDS_STEPS):
                row = lane_id // 8 + wave_id * 8 + r * (N_WAVES * 8)
                _, col = swizzle_128(row, (lane_id % 8) * 16)
                a_rows.append(row)
                a_cols.append(col)
        else:
            gl_off_a = compute_global_swizzle(
                lane_id, wave_id, K, N_LDS_STEPS, preshuffled=False
            )
        # A scale slot w = step * 512 + wave * 64 + lane holds column w // 256
        # of tile row w % 256.
        sc_row = (wave_id % 4) * 64 + lane_id
        sc_col = [step * 2 + wave_id // 4 for step in range_constexpr(2)]

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
                GatherTileScaleLds.BYTES_PER_SLOT,
            )
        a_s2r = LdsFragReader(lds.A_lds_cur_0, wave_m, N_TILES_A, preshuffled=False)
        b_s2r = LdsFragReader(lds.B_lds_cur_0, wave_n, N_TILES_B, preshuffled=True)
        if const_expr(epilogue == "bf16"):
            store_c = StoreBf16Rows(C, m_pad, N)
        elif const_expr(epilogue == "swiglu_fp8"):
            store_c = SwigluFp8Epilogue(
                C, C_scale, m_pad, gate_up_inter, limit, lds.R_lds, wave_m, wave_n
            )
        else:
            w_div = None
            if const_expr(weighted):
                w_div = fx.logical_divide(
                    fx.rocdl.make_buffer_tensor(sorted_weights, max_size=False),
                    fx.make_layout(1, 1),
                )
            n_tokens = _load_i32(nv_div, 1)
            store_c = StoreSlotBf16(C, n_tokens, topk, N, ids_div, w_div, lds.T_lds)

        # Stage 2 runs one K-step stream across its tiles (run_tile_stream).
        # Each tile's output (16 stores, 128 KB a CU) is converted and stored
        # in the next tile's step 0, a quadrant at a time just before its
        # accumulators are reused. Issued in the epilogue, all 8 waves' stores
        # drained together while no MFMA ran (ATT, T=8192: last MFMA -> next
        # first MFMA was ~43% of a wave's time).
        deferred = epilogue == "slot_bf16"
        QUAD_STORES = N_TILES_A

        def slot_cols(nb):
            return (
                half_row0(nb, 0) + wave_n * B_GRP_ROWS,
                half_row0(nb, 1) + wave_n * B_GRP_ROWS,
            )

        def run_tile(tile):
            m_blk = tile // NB
            nb = tile % NB
            expert = _load_i32(eids_div, m_blk)
            row0 = m_blk * BLOCK_M

            a_cur0 = lds.A_lds_cur_0
            a_cur1 = lds.A_lds_cur_1
            a_next0 = lds.A_lds_next_0
            a_next1 = lds.A_lds_next_1
            b_cur0 = lds.B_lds_cur_0
            b_cur1 = lds.B_lds_cur_1
            b_next0 = lds.B_lds_next_0
            b_next1 = lds.B_lds_next_1

            if const_expr(gather):
                toks = [
                    [
                        _load_i32_lane(ids_div, row0 + h * LDS_BLOCK_M + a_rows[r])
                        & 0xFFFFFF
                        for r in range_constexpr(N_LDS_STEPS)
                    ]
                    for h in range_constexpr(2)
                ]
                a_g2s = [
                    G2SLoader(
                        a_div,
                        [
                            toks[h][r] * K + a_cols[r]
                            for r in range_constexpr(N_LDS_STEPS)
                        ],
                        N_LDS_STEPS,
                        F8_IR_t,
                        wave_id,
                    )
                    for h in range_constexpr(2)
                ]
                a_base = [0, 0]
                sc_tok = _load_i32_lane(ids_div, row0 + gsc.dma_row) & 0xFFFFFF
            else:
                g2s = G2SRounds(
                    a_div, gl_off_a[0], N_LDS_STEPS, F8_IR_t, wave_id, 64 * K
                )
                a_g2s = [g2s, g2s]
                a_base = [(row0 + h * LDS_BLOCK_M) * K for h in range_constexpr(2)]
                a_sc_offs = [
                    sc_col[s] * m_pad + row0 + sc_row for s in range_constexpr(2)
                ]
                a_k_stride = m_pad

            # This expert's [N, K] weights as their own descriptor.
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
            B0_off = half_row0(nb, 0) * K
            B1_off = half_row0(nb, 1) * K
            if const_expr(gather):
                b_rows = [
                    expert * (N // 128) + half_row0(nb, h) // 128
                    for h in range_constexpr(2)
                ]
            else:
                # Half h's B scale row; lane parity picks the half (slot w % 2).
                b_sc_row = expert * (N // 128) + (
                    (lane_id % 2 == 0).select(half_row0(nb, 0), half_row0(nb, 1)) // 128
                )
                raw_sc = MoeScaleLds(
                    a_scale_rsrc,
                    a_sc_offs,
                    a_k_stride,
                    b_scale_div,
                    b_sc_row * k128,
                    k128,
                    wave_id,
                    wave_m * A_GRP_ROWS,
                    lds.S_lds,
                )

            def a_load(h, dst, k):
                a_g2s[h].load(dst, a_base[h] + k * BLOCK_K)

            if const_expr(gather):
                sc_a, sc_b = gsc.bases(0)

            def scale_read(k, which):
                op, h = int(which[0] == "b"), int(which[1])
                if const_expr(gather):
                    if const_expr(op == 1):
                        return gsc.read_b(sc_b, k, h, N_TILES_B)
                    return gsc.read_a(sc_a, k, h * LDS_BLOCK_M, N_TILES_A)
                half_row = h * (LDS_BLOCK_N if op else LDS_BLOCK_M)
                return raw_sc.read(op, half_row, N_TILES_B if op else N_TILES_A, k)

            def scale_chunk_dma(k):
                # As the dense kernel: chunk k // CHUNK + 1 after step k's first barrier.
                chunk = k // MoeScaleLds.CHUNK + 1
                if const_expr(
                    not gather
                    and k % MoeScaleLds.CHUNK == 0
                    and k > 0
                    and chunk < n_chunks
                ):
                    raw_sc.dma(chunk)

            c00 = [mfma.zero_value] * (N_TILES_A * N_TILES_B)
            c01 = list(c00)
            c10 = list(c00)
            c11 = list(c00)

            if const_expr(gather):
                gsc.dma(0, sc_tok, b_rows)
            else:
                for chunk in range_constexpr(min(2, n_chunks)):
                    raw_sc.dma(chunk)

            b_g2s.load(b_cur0, B0_off)
            a_load(0, a_cur0, 0)
            b_g2s.load(b_cur1, B1_off)
            a_load(1, a_cur1, 0)

            # The wave_m == 1 group runs one barrier behind.
            barrier_if_zero(lagging)

            wait_barrier(2 * N_LDS_STEPS)

            b_g2s.load(b_next0, B0_off + B_K_STEP)
            a_load(0, a_next0, 1)
            b_g2s.load(b_next1, B1_off + B_K_STEP)

            wait_barrier(3 * N_LDS_STEPS)

            for k in range_constexpr(K_ITERS - 2):
                sa0 = scale_read(k, "a0")
                sb0 = scale_read(k, "b0")
                b0_frag = b_s2r.load(b_cur0)
                a0_frag = a_s2r.load(a_cur0)
                a_load(1, a_next1, k + 1)
                rocdl.s_barrier()
                scale_chunk_dma(k)

                c00 = mfma.call(b0_frag, a0_frag, c00, sb0, sa0, k_pack=0)

                sb1 = scale_read(k, "b1")
                b1_frag = b_s2r.load(b_cur1)
                b_g2s.load(b_cur0, B0_off + (k + 2) * B_K_STEP)
                rocdl.s_barrier()

                c01 = mfma.call(b1_frag, a0_frag, c01, sb1, sa0, k_pack=0)

                sa1 = scale_read(k, "a1")
                a1_frag = a_s2r.load(a_cur1)
                a_load(0, a_cur0, k + 2)
                wait_barrier(4 * N_LDS_STEPS)

                c10 = mfma.call(b0_frag, a1_frag, c10, sb0, sa1, k_pack=0)

                b_g2s.load(b_cur1, B1_off + (k + 2) * B_K_STEP)
                wait_barrier(3 * N_LDS_STEPS)

                c11 = mfma.call(b1_frag, a1_frag, c11, sb1, sa1, k_pack=0)

                a_cur0, a_next0 = a_next0, a_cur0
                a_cur1, a_next1 = a_next1, a_cur1
                b_cur0, b_next0 = b_next0, b_cur0
                b_cur1, b_next1 = b_next1, b_cur1

            k = K_ITERS - 2
            sa0 = scale_read(k, "a0")
            sb0 = scale_read(k, "b0")
            b0_frag = b_s2r.load(b_cur0)
            a0_frag = a_s2r.load(a_cur0)
            wait_barrier(0)

            c00 = mfma.call(b0_frag, a0_frag, c00, sb0, sa0, k_pack=0)

            sb1 = scale_read(k, "b1")
            b1_frag = b_s2r.load(b_cur1)
            rocdl.s_barrier()

            c01 = mfma.call(b1_frag, a0_frag, c01, sb1, sa0, k_pack=0)

            sa1 = scale_read(k, "a1")
            a1_frag = a_s2r.load(a_cur1)
            a_load(1, a_next1, K_ITERS - 1)
            rocdl.s_barrier()

            c10 = mfma.call(b0_frag, a1_frag, c10, sb0, sa1, k_pack=0)

            b0_frag = b_s2r.load(b_next0)
            rocdl.s_barrier()

            c11 = mfma.call(b1_frag, a1_frag, c11, sb1, sa1, k_pack=0)
            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

            k = K_ITERS - 1
            sa0 = scale_read(k, "a0")
            sb0 = scale_read(k, "b0")
            a0_frag = a_s2r.load(a_cur0)
            wait_barrier(0)

            c00 = mfma.call(b0_frag, a0_frag, c00, sb0, sa0, k_pack=0)

            sb1 = scale_read(k, "b1")
            b1_frag = b_s2r.load(b_cur1)
            rocdl.s_barrier()

            c01 = mfma.call(b1_frag, a0_frag, c01, sb1, sa0, k_pack=0)

            sa1 = scale_read(k, "a1")
            a1_frag = a_s2r.load(a_cur1)
            # The next tile's prologue overwrites every LDS buffer once the
            # leading group passes its closing barrier, which pairs with this
            # one: the lagging group's last LDS reads must have landed.
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c10 = mfma.call(b0_frag, a1_frag, c10, sb0, sa1, k_pack=0, set_prio=False)
            c11 = mfma.call(b1_frag, a1_frag, c11, sb1, sa1, k_pack=0, set_prio=False)
            rocdl.s_setprio(0)
            # Pairs the lagging group's last barrier above (see the dense kernel).
            barrier_if_zero(leading)

            base_row = row0 + wave_m * A_GRP_ROWS
            col0 = half_row0(nb, 0) + wave_n * B_GRP_ROWS
            col1 = half_row0(nb, 1) + wave_n * B_GRP_ROWS
            if const_expr(epilogue == "bf16"):
                store_c.store(c00, base_row, col0)
                store_c.store(c01, base_row, col1)
                store_c.store(c10, base_row + LDS_BLOCK_M, col0)
                store_c.store(c11, base_row + LDS_BLOCK_M, col1)
            else:
                store_c.store([(c00, c01), (c10, c11)], row0, nb)

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

        if const_expr(deferred):
            tsc = TileScaleLds(
                a_scale_rsrc,
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
            # K-order rotation: the 16 tiles of an m-block (one A panel, one
            # XCD round) and the m-blocks of an expert (one B panel) start at
            # different K steps, so most of them find a K slice already in L2
            # instead of all waiting on the same miss. Stage 2 alone, T = 4096
            # / 8192 / 16384: 327 / 383 / 662 -> 316 / 376 / 620 us with the
            # per-tile scales that make it possible (ABAB).
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
                # A weight panel no other m-block of the expert reads streams
                # through L2 nontemporally (leaving it to A panels and output
                # lines): stage 2 alone, T = 4096 / 8192 / 16384: 295 / 375 /
                # 609 -> 278 / 367 / 600 us; nt on every panel is +10% at
                # 16384, where experts span ~2 m-blocks.
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
            """Stage 2: K-steps 0 .. K - 1 of ``tile``.

            One K-step stream runs across tiles: step k loads A half 1 of step
            k + 1 and all of step k + 2, which for the last two steps are
            ``next_tile``'s steps 0 and 1, and scale chunk k / 4 + 1 at
            k % 4 == 0, which at k = K - 4 is the next tile's chunk 0. Step 0
            also fetches this tile's routes, loads the next tile's expert and
            converts and stores the previous tile's accumulators ``prev_acc``,
            each before its registers start accumulating again (HK's stage-2
            epilogue). c00 and c11 go in the V part of their own phase; c01
            and c10 are interleaved into the c00 and c01 bursts, one phase
            early. A step-0 V part is ~5x its burst, so the two groups' V parts
            otherwise ran back to back while the MFMA pipe idled (ATT), and in
            a burst the pack overlaps the other group's V part (stage 2 -1.5% /
            -2.3% at T = 4096 / 16384). Returns (accumulators, next expert,
            next skip).
            """
            valid = prev_tile >= 0
            pcols = slot_cols(_pow2_mod(prev_tile, NB))
            prev_slot = 1 - parity
            prev_rows = {}

            def store_prev(g):
                """Accumulator group g = (row half, column half) g // 2, g % 2."""
                if const_expr(g % 2 == 0):
                    rows = store_c.rows_half(wave_m, prev_slot, g // 2)
                    prev_rows[g // 2] = (rows, store_c.quad_offsets(rows, valid))
                rows, offs = prev_rows[g // 2]
                if const_expr(g < 2):
                    store_c.store(store_c.pack(prev_acc[g], rows), offs, pcols[g % 2])
                else:
                    # This wave's row-half-1 rows of the previous tile are all
                    # padding: the same count of stores, without the 1 KB each.
                    either(
                        prev_skip_h1,
                        lambda: store_c.store(
                            store_c.pack(prev_acc[g], rows), offs, pcols[g % 2]
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
                    prev_rows[1] = (rows1, store_c.quad_offsets(rows1, valid))
                rocdl.s_barrier()

                if const_expr(k == 0):
                    # The previous tile's c10 rides in this burst. Its
                    # real-or-dummy choice is taken around the whole burst, so
                    # each arm is one block the scheduler can interleave.
                    rocdl.s_setprio(1)
                    c01_in = c01

                    def c01_with_c10():
                        c = mfma.call(
                            b1_frag, a0_frag, c01_in, sb1, sa0, k_pack=0, set_prio=False
                        )
                        rows, offs = prev_rows[1]
                        store_c.store(store_c.pack(prev_acc[2], rows), offs, pcols[0])
                        _interleave_store_burst(N_TILES_A)
                        return c

                    def c01_with_dummy():
                        c = mfma.call(
                            b1_frag, a0_frag, c01_in, sb1, sa0, k_pack=0, set_prio=False
                        )
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
                wait_barrier(
                    4 * N_LDS_STEPS
                    + n_sc
                    + (7 + store_c.n_fetch + 3 * QUAD_STORES if k == 0 else 0)
                    + (QUAD_STORES if k == 1 else 0)
                )

                c10 = mfma_q(b0_frag, a1_frag, c10, sb0, sa1, skip_h1)

                ctx["b"](1, b_cur1, kk)
                if const_expr(k == 0):
                    store_prev(3)
                wait_barrier(
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

        if const_expr(stream_s1):
            ids8_div = fx.logical_divide(
                make_fp8_buffer_tensor(sorted_ids, F8_IR_t), fx.make_layout(1, 1)
            )
            ids_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS32b(), 32)
            a_gather = G2SGather(a_div, wave_id)
            LdsF8Ptr_t = fx.PointerType.get(F8_IR_t, 2, 512)

        def ids_dma(tile, slot):
            """``tile``'s 256 sorted_ids into ``I_lds`` slot ``slot``, a dword a lane (waves 4..7 repeat 0..3)."""
            row0 = _pow2_div(tile, NB) * BLOCK_M
            q = wave_id % 4
            base_i32 = fx.Int32(fx.ptrtoint(lds.I_lds.ptr))
            dst = fx.inttoptr(LdsF8Ptr_t, base_i32 + slot * (BLOCK_M * 4) + q * 256)
            fx.copy(
                ids_atom,
                fx.slice(ids8_div, (None, (row0 + q * 64 + _fresh_lane()) * 4)),
                fx.make_view(dst, fx.make_layout(1, 1)),
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

            # Lane terms from a fresh lane id at each use (_fresh_lane): held
            # across the K loop they spilled, and their reloads drained the
            # streamed loads with vmcnt(0).
            def a_offs(h):
                lane = _fresh_lane()
                row = lane // 8 + wave_id * 8
                # One opaque base, every read a constant past it.
                a_ids = _opaque_vgpr(slot * BLOCK_M + row)
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
                row = wave_id % 4 * 64 + _fresh_lane()
                tok = ids_at(_opaque_vgpr(slot * BLOCK_M + row)) & 0xFFFFFF
                gsc.dma(slot, tok, b_rows)

            return {"a": a_load, "b": b_load, "sc": sc_dma, "m_blk": m_blk, "nb": nb}

        # Epilogue VMEM per wave: an 8-byte act store and a 1-byte scale store per row tile.
        EPI_STORES = 2 * 2 * N_TILES_A

        def run_tile_stream_s1(tile, expert, next_tile, parity):
            """Stage 1: K-steps 0 .. K - 1 of ``tile``, then its SwiGLU epilogue.

            One K-step stream runs across tiles, as stage 2's: the last two
            steps load ``next_tile``'s steps 0 and 1, step K - 4 its scales
            (into the other parity's slot) and step 0 its expert id and
            sorted_ids (into the other parity's ``I_lds`` slot). Only the loads
            move: the accumulators are converted and stored at the tile's end
            (SwiGLU needs all of them). Returns the next tile's expert.
            """
            a_cur0, a_cur1, a_next0, a_next1, b_cur0, b_cur1, b_next0, b_next1 = (
                LDS_BUFS
            )
            cur = make_ctx_s1(tile, expert, parity, False)
            nxt = None
            nxt_expert = None
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

                c10 = mfma.call(b0_frag, a1_frag, c10, sb0, sa1, k_pack=0)

                ctx["b"](1, b_cur1, kk)
                _wait_barrier_seen(3 * N_LDS_STEPS + n_sc)
                if const_expr(k == 0):
                    nxt_expert = _sgpr(nxt_eid)

                c11 = mfma.call(b1_frag, a1_frag, c11, sb1, sa1, k_pack=0)

                a_cur0, a_next0 = a_next0, a_cur0
                a_cur1, a_next1 = a_next1, a_cur1
                b_cur0, b_next0 = b_next0, b_cur0
                b_cur1, b_next1 = b_next1, b_cur1

            store_c.store([(c00, c01), (c10, c11)], cur["m_blk"] * BLOCK_M, cur["nb"])
            return nxt_expert

        grid = fx.Int32(fx.grid_dim.x)
        bid = fx.Int32(fx.block_idx.x)
        start = (bid % NUM_XCDS) * (grid // NUM_XCDS) + bid // NUM_XCDS
        n_tiles = _sgpr((_load_i32(nv_div, 0) // BLOCK_M) * NB)
        if const_expr(gather and not stream_s1):
            # i32 bounds keep this an i32 loop: an init= loop is index (i64),
            # and with no scalar 64-bit less-than its bound sits in a VGPR
            # pair that spilled; the reload's vmcnt(0) at the tile's end then
            # also waited out the epilogue stores (stage 1 ~2% slower).
            for tile in range(start, n_tiles, grid):
                run_tile(fx.Int32(tile))
        elif const_expr(deferred):
            # Prologue: the first tile's scale chunk 0 and K-steps 0 and 1, as
            # the stream's steps -4, -2 and -1 would have loaded them. A
            # workgroup without a tile still runs it, on tile 0 of expert 0:
            # sorted_expert_ids past the last valid tile is unwritten, and a
            # garbage expert rebases the weight descriptor anywhere (with no
            # valid tile at all, a memory fault).
            has_tile = start < n_tiles
            first_tile = has_tile.select(start, fx.Int32(0))
            first_expert = has_tile.select(
                _load_i32(eids_div, _pow2_div(first_tile, NB)), fx.Int32(0)
            )
            first_mask = pad_mask(pad_rows(first_tile))
            first = make_ctx(first_tile, first_expert, first_mask)
            first["sc"](0)
            a_cur0, a_cur1, a_next0, a_next1, b_cur0, b_cur1, b_next0, b_next1 = (
                LDS_BUFS
            )
            first["b"](0, b_cur0, 0)
            first["a"](0, a_cur0, 0)
            first["b"](1, b_cur1, 0)
            first["a"](1, a_cur1, 0)
            # The wave_m == 1 group runs one barrier behind, for the whole kernel.
            barrier_if_zero(lagging)
            wait_barrier(2 * N_LDS_STEPS)
            first["b"](0, b_next0, 1)
            first["a"](0, a_next0, 1)
            first["b"](1, b_next1, 1)
            wait_barrier(3 * N_LDS_STEPS)

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
                store_c.store(store_c.pack(acc, rows), offs, last_cols[g % 2])
            # No LDS-DMA may land after the workgroup exits; the leading
            # group's barrier pairs the lagging group's prologue one.
            fx.rocdl.s_waitcnt(vmcnt=0)
            barrier_if_zero(leading)
        elif const_expr(stream_s1):
            # Prologue: the first tile's ids, scales and K-steps 0 and 1, as
            # the previous tile's stream would have loaded them (a workgroup
            # without a tile runs it on tile 0 of expert 0, as stage 2 does).
            has_tile = start < n_tiles
            first_tile = has_tile.select(start, fx.Int32(0))
            first_expert = has_tile.select(
                _load_i32(eids_div, _pow2_div(first_tile, NB)), fx.Int32(0)
            )
            ids_dma(first_tile, 0)
            _wait_barrier_seen(0)
            first = make_ctx_s1(first_tile, first_expert, fx.Int32(0), False)
            first["sc"]()
            a_cur0, a_cur1, a_next0, a_next1, b_cur0, b_cur1, b_next0, b_next1 = (
                LDS_BUFS
            )
            first["b"](0, b_cur0, 0)
            first["a"](0, a_cur0, 0)
            first["b"](1, b_cur1, 0)
            first["a"](1, a_cur1, 0)
            # The wave_m == 1 group runs one barrier behind, for the whole kernel.
            barrier_if_zero(lagging)
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
            count = fx.Int32(0)
            while tile < n_tiles:
                nt = tile + grid
                # The last tile streams itself again; nothing reads those loads.
                nt = (nt < n_tiles).select(nt, tile)
                expert = run_tile_stream_s1(tile, expert, nt, count & 1)
                count = count + 1
                tile = tile + grid
            # No LDS-DMA may land after the workgroup exits; the leading
            # group's barrier pairs the lagging group's prologue one.
            fx.rocdl.s_waitcnt(vmcnt=0)
            barrier_if_zero(leading)
        else:
            for tile in range(start, n_tiles, grid):
                run_tile(fx.Int32(tile))

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
        sorted_weights: fx.Tensor,
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
            sorted_weights,
            m_pad,
            limit,
            value_attrs={
                # 512 threads put two waves on every SIMD.
                "rocdl.waves_per_eu": 2,
                "rocdl.flat_work_group_size": "512,512",
            },
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch
