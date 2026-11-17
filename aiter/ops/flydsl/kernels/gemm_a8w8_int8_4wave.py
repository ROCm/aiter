# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""4-wave true-INT8 matmul with row-wise scaling for AMD CDNA4 (gfx950 only).

One wave per SIMD, a 2x2 wave grid over a 256x256 block and a 128x128 tile
per wave, accumulated in AGPRs with ``v_mfma_i32_16x16x64_i8``. The K loop
follows the hipBLASLt FP8 schedule: two LDS buffers, three barriers per K128
tile, and LDS reads / LDS-DMA loads issued between MFMAs. The loads are
``global_load_lds``, whose signed offset lets one M0 value cover a wave's
whole share of an operand tile.

LDS holds both operands in 16-row blocks per K128 tile. A is stored
[row][k16 chunk] with an XOR swizzle on the chunk, so its global loads cover
whole 128-byte rows. B keeps its ``shuffle_weight(layout=(16, 16))`` block
layout [k16 chunk][row][16 B], so its global loads are contiguous, with the
blocks padded to 2064 bytes. B tile ``j`` of a wave takes rows ``8 * k + j``,
so each lane ends up with eight adjacent output columns and the epilogue
writes whole 256-byte rows, as hipBLASLt does.

gfx950 serves a ds_read_b128 in four phases of 16 lanes: lane groups
``l // 4`` in {0, 3, 5, 6} and {1, 2, 4, 7} of each 32-lane half. Both
layouts give each phase 16 distinct 16-byte bank groups.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx

# The LLVM dialect is used only for inline assembly at instructions FlyDSL 0.4
# cannot represent through its stable surface: M0 ownership, signed-offset LDS
# DMA, the scheduled scale loads, K64 INT8 MFMA, and the accumulator hazard
# boundary.
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.rocdl import update_dpp
from flydsl.expr.typing import ReductionOp
from flydsl.expr.typing import Vector as Vec

from aiter.jit.utils.chip_info import get_cdna4_num_xcds
from aiter.ops.flydsl.kernels.communication_ops_utils import spin_until_ge_i32_agent
from aiter.ops.flydsl.kernels.gemm_a8w8_8wave import (
    S2RLoader,
    StoreC,
    _xcd_swizzle_any,
)
from aiter.ops.flydsl.kernels.gemm_hooks import GemmHooks
from aiter.ops.flydsl.kernels.kernels_common import (
    ceildiv,
    set_slp_vectorize_once_used,
)
from aiter.ops.flydsl.kernels.mfma_preshuffle_pipeline import split_row_major_2d
from aiter.ops.flydsl.kernels.tensor_shim import buf_copy_store, ptr_buf_tensor

BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 128
N_WAVES = 4
WAVE_ROWS = 128  # rows (and columns) of C per wave
TILE = 16  # MFMA output tile
MFMA_K = 64
WAVE_TILES = WAVE_ROWS // TILE
K_STEPS = BLOCK_K // MFMA_K
K_GROUPS = 4  # lane groups along K within one operand
ROW_BLOCKS = BLOCK_M // 16  # 16-row blocks per region
A_BLOCK_STRIDE = 16 * BLOCK_K
B_BLOCK_STRIDE = A_BLOCK_STRIDE + 16  # shifts block b by b bank groups
A_REGION_BYTES = ROW_BLOCKS * A_BLOCK_STRIDE
B_REGION_BYTES = ROW_BLOCKS * B_BLOCK_STRIDE
LOADS_PER_REGION = BLOCK_M * BLOCK_K // (N_WAVES * 1024)


# DPP moves within a 16-lane row and the lane bit that orders each pair: quad_perm
# [1,0,3,2] and [2,3,0,1] pair lanes i ^ 1 and i ^ 2, row_half_mirror i ^ 7 and
# row_ror:8 i ^ 8. The four masks span the row's lane bits, so a butterfly over them
# is a Hadamard transform, with rows in a different order than Sylvester's.
_ROW_BUTTERFLY = ((0xB1, 1), (0x4E, 2), (0x141, 4), (0x128, 8))
_ROW_ROR8 = 0x128  # lane i of a 16-lane row reads lane i ^ 8


@dataclass(frozen=True)
class PrepSpec:
    """The Q/K prep of ``compile_int8_gemm_4w``'s epilogue, fixed at compile time.

    ``norm``:
      - ``"rms_across_heads"`` (Wan): RMSNorm over every head of a row. The epilogue applies
        the norm weight and stores per-head sums of squares; the caller scales each row by
        ``rsqrt(sum / N + eps)``.
      - ``"rms_per_head"``: RMSNorm over each 128-wide head, applied in the epilogue.
      - ``"none"``: no norm and no norm weights.

    ``rope``: ``"interleaved"`` (Wan) rotates channel pairs (2i, 2i + 1) of each head,
    ``"rotate_half"`` pairs i and i + 64, ``"none"`` skips RoPE. The cos and sin tables hold
    one value per token and rotation pair, ``[M, 64]``, the same for every head.

    ``bias``: a per-column bias before the norm.

    ``hadamard``: the per-head Hadamard after RoPE.
    """

    norm: str = "rms_across_heads"
    rope: str = "interleaved"
    bias: bool = True
    hadamard: bool = True

    def __post_init__(self):
        if self.norm not in ("none", "rms_per_head", "rms_across_heads"):
            raise ValueError(f"unknown PrepSpec norm {self.norm!r}")
        if self.rope not in ("none", "interleaved", "rotate_half"):
            raise ValueError(f"unknown PrepSpec rope {self.rope!r}")

    @property
    def gammas(self):
        """Whether B_scale holds norm weights."""
        return self.norm != "none"

    @property
    def suffix(self):
        """Kernel-name suffix after ``_prep``. Wan's prep keeps the names it had."""
        suffix = "" if self.hadamard else "_nohadamard"
        if self.norm != "rms_across_heads":
            suffix += "_n" + ("head" if self.norm == "rms_per_head" else "none")
        if self.rope != "interleaved":
            suffix += "_r" + ("half" if self.rope == "rotate_half" else "none")
        return suffix


def _row_move(value, ctrl):
    # Every lane of the row has a source under these controls, so zero fill never
    # shows; it lets LLVM fold the move into the instruction that consumes it.
    return fx.Float32(
        update_dpp(
            fx.Float32.ir_type, value.ir_value(), value.ir_value(), ctrl, 0xF, 0xF, True
        )
    )


def _butterfly_signs(lane):
    """Per stage of _ROW_BUTTERFLY: +1 in the lane that keeps the sum, -1 in its partner."""
    return [
        ((lane & bit) == 0).select(fx.Float32(1.0), fx.Float32(-1.0))
        for _, bit in _ROW_BUTTERFLY
    ]


def _row_sum(value):
    """Sum over the 16 lanes of a DPP row, in every lane of it."""
    for ctrl, _ in _ROW_BUTTERFLY:
        value = value + _row_move(value, ctrl)
    return value


def _row_max(value):
    """Maximum over the 16 lanes of a DPP row, in every lane of it."""
    for ctrl, _ in _ROW_BUTTERFLY:
        value = value.maximumf(_row_move(value, ctrl))
    return value


def _max_abs(values):
    """Largest magnitude among the 8 values of a vector."""
    result = fx.Float32(0.0)
    for k in range(8):
        result = result.maximumf(values[k].maximumf(-values[k]))
    return result


def _global_ptr(address, dtype):
    return fx.inttoptr(
        fx.PointerType.get(dtype.ir_type, fx.AddressSpace.Global, 4), fx.Int64(address)
    )


def _load_f32x4(view, index, atom, reg):
    """Four FP32 values of ``view`` from element ``index``."""
    fx.copy(atom, fx.slice(view, (None, fx.Int32(index))), reg)
    return Vec(fx.memref_load_vec(reg))


def _hadamard128(values, signs):
    """Unnormalized Hadamard over one 128-wide head held as 8 values in each of 16 lanes.

    Q and K go through the same transform, so their attention scores are unchanged.
    ``signs`` comes from _butterfly_signs.
    """
    result = list(values)
    for stage in range(3):
        shift = 1 << stage
        result = [
            (
                result[i ^ shift] - result[i]
                if i & shift
                else result[i] + result[i ^ shift]
            )
            for i in range(8)
        ]
    for (ctrl, _), sign in zip(_ROW_BUTTERFLY, signs):
        result = [_row_move(value, ctrl) + sign * value for value in result]
    return result


def _a_chunk_swizzle(row):
    """XOR key for row ``row`` (0-15) of an A row block.

    Rows ``r`` and ``r ^ 1`` share 64 banks. A phase of the A operand read
    covers all 16 rows, with k16 chunk ``c ^ 1`` for rows 4-11 and ``c`` for
    the others, so keys 0-3 go to rows 4-11 and 4-7 to the rest, distinct
    within each row parity.
    """
    quad = row // 4
    return ((1 ^ (quad % 2) ^ (quad // 2)) << 2) | ((quad // 2) << 1) | ((row % 4) // 2)


def _a_lds_read_base(lane_id, wave_m, kk):
    """LDS byte offset for A tile zero; tile ``i`` adds ``i*A_BLOCK_STRIDE``."""
    row = lane_id % 16
    k_group = lane_id // 16
    return (
        wave_m * (WAVE_ROWS // 16) * A_BLOCK_STRIDE
        + row * BLOCK_K
        + ((kk * K_GROUPS + k_group) ^ _a_chunk_swizzle(row)) * 16
    )


def _b_lds_read_base(lane_id, wave_n, kk):
    """LDS byte offset for B tile zero; tile ``j`` adds ``j*16``."""
    row = lane_id % 16
    k_group = lane_id // 16
    return (
        (wave_n * (WAVE_ROWS // 16) + row // 2) * B_BLOCK_STRIDE
        + (kk * K_GROUPS + k_group) * 256
        + (row % 2) * 128
    )


def _lane_offset(lane_id, wave_id, K, preshuffled, block_row0, n_rows, step):
    """Per-lane global offset for LDS-DMA step ``step`` of one region.

    Step ``s`` of wave ``w`` fills half ``s % 2`` (1 KB) of row block
    ``4 * w + s // 2``: rows ``(s % 2) * 8`` onwards for A, k16 chunks
    ``(s % 2) * 4`` onwards for B. A stores row ``r`` as eight 16-byte chunks
    with chunk ``c`` at slot ``c ^ _a_chunk_swizzle(r)``, so every 8 lanes
    load one 128-byte row.

    The offset is from the operand base, for the block starting at row
    ``block_row0``. Nothing bounds a global load, so rows at or past
    ``n_rows`` (whole 16-row strips for B) read the last valid one instead;
    they only reach C rows and columns that are not stored.

    There is no stable layout object shared with the LDS read path because the
    LDS destination is implicit in M0 plus the instruction's signed immediate.
    The named strides and ``_a_chunk_swizzle`` are therefore the shared layout
    contract used by this address calculation and the read-base helpers above.
    """
    row_block = wave_id * (LOADS_PER_REGION // 2) + step // 2
    if const_expr(preshuffled):
        strip = block_row0 // 16 + row_block
        strip = (strip < n_rows // 16).select(strip, n_rows // 16 - 1)
        return strip * (16 * K) + (step % 2) * 1024 + lane_id * 16
    row = (step % 2) * 8 + lane_id // 8
    chunk = (lane_id % 8) ^ _a_chunk_swizzle(row)
    g_row = block_row0 + row_block * 16 + row
    g_row = (g_row < n_rows).select(g_row, n_rows - 1)
    return g_row * K + chunk * 16


class G2SGlobal:
    """``global_load_lds`` steps of one region, all on one M0 value.

    The instruction offset applies to both the global and the LDS address and
    is signed 13-bit, so with M0 4 KB into the wave's 8 KB of the region it
    reaches all eight steps; the buffer form's unsigned 12-bit offset needs
    four M0 changes per K tile, this one two.

    The loads are inline asm because LLVM asserts on the intrinsic's negative
    LDS memory-operand offset. LLVM does not count them in vmcnt, so the K
    loop waits for them explicitly, and the epilogue's scale loads, which
    persistent workgroups issue before the next block's, are asm as well.
    M0 is written by ``set_m0`` ops, placed an MFMA slot ahead of the loads
    so the write and its hazard ``s_nop`` stay off the loads' slots. LLVM
    does not see M0 in use, so nothing else in the kernel may use it.
    """

    CENTER = 4096

    def __init__(self, tensor, wave_id, block_stride, lds_pair):
        piece = [
            (s // 2) * block_stride + (s % 2) * 1024 for s in range(LOADS_PER_REGION)
        ]
        self.imm = [p - self.CENTER for p in piece]
        assert all(-4096 <= i <= self.CENTER for i in self.imm)
        # The VGPR offset is unsigned, so the SGPR base carries -CENTER.
        self.base = fx.Int64(fx.ptrtoint(fx.get_iter(tensor))) - self.CENTER
        wave_off = wave_id * (LOADS_PER_REGION // 2) * block_stride + self.CENTER
        # M0 is written from an SGPR, so the wave-uniform LDS address is read
        # from lane 0. FlyDSL 0.4 exports no stable readfirstlane.
        self.m0 = [
            fx.rocdl.readfirstlane(
                fx.Int32.ir_type,
                (fx.Int32(fx.ptrtoint(region.ptr)) + wave_off).ir_value(),
            )
            for region in lds_pair
        ]

    def voffset(self, step, lane_offset):
        """VGPR offset of ``step`` for a lane at ``lane_offset`` from the base."""
        return lane_offset - self.imm[step] + self.CENTER

    def set_m0(self, buf):
        """Point M0 at LDS buffer ``buf`` for the ``ops`` that follow.

        FlyDSL 0.4 has no stable M0 setter. Remove this boundary when a setter
        with an explicit post-write hazard is exported.
        """
        _llvm.inline_asm(
            res=None,
            operands_=[fx.Int32(self.m0[buf]).ir_value()],
            asm_string="s_mov_b32 m0, $0\ns_nop 0",
            constraints="s",
            has_side_effects=True,
        )

    def _load(self, voffset, k_offset, step):
        # FlyDSL 0.4 cannot encode this signed LDS memory-operand offset; its
        # intrinsic path reaches an LLVM assertion. Keep the raw instruction
        # until signed-offset global_load_lds is supported.
        _llvm.inline_asm(
            res=None,
            operands_=[
                fx.Int32(voffset).ir_value(),
                (self.base + fx.Int64(k_offset)).ir_value(),
            ],
            asm_string=f"global_load_lds_dwordx4 $0, $1 offset:{self.imm[step]}",
            constraints="v,s",
            has_side_effects=True,
        )

    def ops(self, voffsets, k_offset):
        """One op per step, each loading at ``voffsets[step]``."""
        return [
            lambda step=step: self._load(voffsets[step], k_offset, step)
            for step in range(LOADS_PER_REGION)
        ]


class StoreCInt8Rows(StoreC):
    """Scale and store the wave's INT32 accumulators as BF16 rows.

    Lane ``l`` of MFMA tile (i, j) holds row ``16 i + 4 (l // 16) + r`` in
    register ``r`` and column ``8 (l % 16) + j``, so register ``r`` of the
    eight j tiles is one 16-byte store and 16 lanes cover a 256-byte row.
    Preshuffled B has N % 16 == 0: a store is fully inside C or past column N.
    """

    SCALE_LOADS = WAVE_TILES + 2
    # B_scale holds the N column scales, then with a bias N FP32 biases, then with a
    # normalizing prep N FP32 RMSNorm weights.
    BIAS_LOADS = 2
    GAMMA_LOADS = 2

    def __init__(
        self,
        A_scale,
        B_scale,
        C,
        c_rows,
        c_cols,
        *args,
        has_bias=False,
        prep=None,
        prep_spec=None,
        eps=1e-6,
        amax_partials=None,
        planar_cols=0,
    ):
        super().__init__(A_scale, B_scale, C, c_rows, c_cols, *args)
        self.has_bias = has_bias
        self.planar_cols = planar_cols
        self.prep = prep is not None
        self.spec = prep_spec
        self.has_gamma = self.prep and prep_spec.gammas
        # Index of the first norm-weight load in scale_load_ops' result.
        self.gamma_index = WAVE_TILES + 2 + (self.BIAS_LOADS if has_bias else 0)
        self.eps = eps
        self.amax = self.prep or amax_partials is not None
        if self.amax and not self.prep:
            self._init_partials(c_rows, c_cols, amax_partials)
        if self.prep:
            cos, sin, partials = prep
            freq_bytes = fx.Int64(c_rows) * 64 * 4
            self.cos_div, self.sin_div = (
                fx.logical_divide(
                    fx.rocdl.make_buffer_tensor(
                        table, max_size=False, num_records_bytes=freq_bytes
                    ),
                    fx.make_layout(1, 1),
                )
                for table in (cos, sin)
            )
            self._init_partials(c_rows, c_cols, partials)
        self.out_atom_8 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        self.reg_bf16_8 = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
        sa_tensor = fx.rocdl.make_buffer_tensor(
            A_scale, num_records_bytes=fx.Int64(c_rows) * 4
        )
        sb_tensor = fx.rocdl.make_buffer_tensor(
            B_scale,
            num_records_bytes=fx.Int64(c_cols) * 4 * (1 + has_bias + self.has_gamma),
        )
        self.sa_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(sa_tensor))
        self.sb_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(sb_tensor))

    @staticmethod
    def _load_f32x4(rsrc, byte_offset):
        # A stable copy atom would enter LLVM's vmcnt accounting, but these
        # loads must be ordered beside opaque LDS-DMA loads that LLVM cannot
        # count. Retain the scheduled raw load until both operations share one
        # stable dependency model.
        # LLVM inserts no hazard wait states before inline asm. Under SGPR
        # pressure it restores the resource with v_readlane right before the
        # load, and a VALU SGPR write needs five wait states before a VMEM
        # instruction reads that SGPR.
        return _llvm.inline_asm(
            res=fx.Vector.make_type(4, fx.Float32),
            operands_=[fx.Int32(byte_offset).ir_value(), rsrc],
            asm_string="s_nop 4\nbuffer_load_dwordx4 $0, $1, $2, 0 offen",
            constraints="=v,v,s",
            has_side_effects=True,
        )

    def scale_load_ops(self, row_base, col_base):
        """Return scheduled scale-load operations for ``store_tiles``.

        The loads are asm, so LLVM leaves them out of its vmcnt waits: an LLVM
        wait for them would also wait for the LDS-DMA loads that follow them,
        which it cannot count. The caller waits before the stores.
        """

        def op(rsrc, element):
            return lambda: self._load_f32x4(rsrc, element() * 4)

        def row(i):
            return row_base + (self.lane_id // 16) * 4 + i * TILE

        def col(half):
            return col_base + (self.lane_id % 16) * 8 + half * 4

        ops = [op(self.sa_rsrc, lambda i=i: row(i)) for i in range(WAVE_TILES)] + [
            op(self.sb_rsrc, lambda h=h: col(h)) for h in (0, 1)
        ]
        if self.has_bias:
            ops += [op(self.sb_rsrc, lambda h=h: self.c_cols + col(h)) for h in (0, 1)]
        if self.has_gamma:
            gamma_cols = (1 + self.has_bias) * self.c_cols
            ops += [op(self.sb_rsrc, lambda h=h: gamma_cols + col(h)) for h in (0, 1)]
        return ops

    def _store_bf16x8(self, values, c_index):
        fx.memref_store_vec(values, self.reg_bf16_8)
        fx.copy(
            self.out_atom_8,
            self.reg_bf16_8,
            fx.slice(self.c_div, (None, fx.Int32(c_index))),
        )

    def _init_partials(self, c_rows, c_cols, partials):
        """[2, M, N // 128] FP32: per-head sums of squares, then per-head maxima."""
        self.heads = c_cols // 128
        self.maxima_base = fx.Int32(c_rows * self.heads)
        self.partials_oob = fx.Int32(c_rows * self.heads * 2)
        self.partials_div = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                partials,
                max_size=False,
                num_records_bytes=fx.Int64(c_rows) * self.heads * 8,
            ),
            fx.make_layout(1, 1),
        )

    def _store_row_max(self, values, row, col_base):
        """Store the largest magnitude of this row's head, as it will be stored in BF16."""
        stored = values.to(fx.BFloat16).to(fx.Float32)
        maximum = _row_max(_max_abs(stored))
        store = (row < self.c_rows) & ((self.lane_id % 16) == 0)
        index = self.maxima_base + row * self.heads + col_base // 128
        self._store_f32(maximum, store.select(index, self.partials_oob))

    def _load_freq4(self, div, index):
        fx.copy(
            self.scale_atom_4, fx.slice(div, (None, fx.Int32(index))), self.reg_f32_4
        )
        return Vec(fx.memref_load_vec(self.reg_f32_4))

    def _store_f32(self, value, index):
        fx.memref_store_vec(fx.full(1, value, fx.Float32), self.reg_f32_1)
        fx.copy(
            self.scale_atom_1,
            self.reg_f32_1,
            fx.slice(self.partials_div, (None, fx.Int32(index))),
        )

    def _load_freqs(self, row):
        """cos and sin of the rotation pairs this lane's 8 channels of ``row`` take part in.

        Interleaved: the 4 pairs the lane holds. Rotate-half: lanes l and l ^ 8 hold the two
        halves of 8 pairs, so both read the same 8 values.
        """
        row = (row < self.c_rows).select(row, fx.Int32(0))
        if const_expr(self.spec.rope == "rotate_half"):
            index = row * 64 + (self.lane_id % 8) * 8
            return tuple(
                [
                    v[k]
                    for v in (
                        self._load_freq4(div, index),
                        self._load_freq4(div, index + 4),
                    )
                    for k in range(4)
                ]
                for div in (self.cos_div, self.sin_div)
            )
        index = row * 64 + (self.lane_id % 16) * 4
        return self._load_freq4(self.cos_div, index), self._load_freq4(
            self.sin_div, index
        )

    def _prep_row(self, x, gammas, freqs, signs, row, col_base):
        """The spec's Q/K prep of the 8 channels this lane holds of one row.

        A norm across heads is left to the caller: its factor is one scalar per row, and
        RoPE and the Hadamard are linear, so the caller applies it later from the per-head
        sums of squares stored here.
        """
        spec = self.spec
        store = (row < self.c_rows) & ((self.lane_id % 16) == 0)
        if const_expr(spec.gammas):
            sq = _row_sum(fx.Float32((x * x).reduce(ReductionOp.ADD)))
        if const_expr(spec.norm == "rms_per_head"):
            x = x * Vec.filled(8, fmath.rsqrt(sq * (1.0 / 128) + self.eps), fx.Float32)
        y = x * gammas if const_expr(spec.gammas) else x
        if const_expr(spec.rope == "interleaved"):
            cos, sin = freqs
            rotated = []
            for k in range_constexpr(4):
                even, odd = y[2 * k], y[2 * k + 1]
                rotated += [even * cos[k] - odd * sin[k], odd * cos[k] + even * sin[k]]
        elif const_expr(spec.rope == "rotate_half"):
            cos, sin = freqs
            # Channel c < 64 takes -x[c + 64] sin, channel c >= 64 takes +x[c - 64] sin.
            low = (self.lane_id % 16) < 8
            rotated = []
            for k in range_constexpr(8):
                partner = _row_move(y[k], _ROW_ROR8)
                rotated.append(y[k] * cos[k] + low.select(-partner, partner) * sin[k])
        else:
            rotated = [y[k] for k in range(8)]
        if const_expr(spec.norm == "rms_across_heads"):
            p_index = row * self.heads + col_base // 128
            self._store_f32(sq, store.select(p_index, self.partials_oob))
        if const_expr(spec.hadamard):
            rotated = _hadamard128(rotated, signs)
        result = Vec.from_elements(rotated, fx.Float32)
        self._store_row_max(result, row, col_base)
        return result

    def store_tiles(self, accs, row_base, col_base, scales, prep=None, col_max=None):
        """``scales`` comes from ``scale_load_ops`` and must have landed.

        ``prep=False`` stores a prep-built instance's tile with maxima only (V).
        ``col_max`` (an address) also stores each column's largest magnitude over the
        wave's 128 rows, as stored in BF16: FP32 at [row // 128][column within its plane].
        """
        prep = self.prep if prep is None else prep
        if const_expr(self.has_gamma and not prep):
            # Norm-weight loads went out for this tile too; their registers stay claimed
            # until here, past the wait that lands them.
            _llvm.inline_asm(
                res=None,
                operands_=[Vec(s).ir_value() for s in scales[self.gamma_index :]],
                asm_string="",
                constraints="v,v",
                has_side_effects=True,
            )
        row0 = row_base + (self.lane_id // 16) * 4
        col = col_base + (self.lane_id % 16) * 8
        a_scales = [Vec(s) for s in scales[:WAVE_TILES]]
        b_lo, b_hi = (Vec(s) for s in scales[WAVE_TILES : WAVE_TILES + 2])
        b_scales = Vec.from_elements(
            [b_lo[j] for j in range(4)] + [b_hi[j] for j in range(4)], fx.Float32
        )
        if const_expr(self.has_bias):
            bias_lo, bias_hi = (Vec(s) for s in scales[WAVE_TILES + 2 : WAVE_TILES + 4])
            biases = Vec.from_elements(
                [bias_lo[j] for j in range(4)] + [bias_hi[j] for j in range(4)],
                fx.Float32,
            )
        gammas = None
        if const_expr(prep and self.has_gamma):
            gamma_lo, gamma_hi = (Vec(s) for s in scales[self.gamma_index :])
            gammas = Vec.from_elements(
                [gamma_lo[j] for j in range(4)] + [gamma_hi[j] for j in range(4)],
                fx.Float32,
            )
        oob = fx.Int32(self.c_rows * self.c_cols)
        col_valid = col < self.c_cols
        signs = None
        if const_expr(prep and self.spec.hadamard):
            signs = _butterfly_signs(self.lane_id)
        peaks = [fx.Float32(0.0) for _ in range(8)]
        rope = prep and self.spec.rope != "none"
        for i in range_constexpr(WAVE_TILES):
            tile_freqs = [None] * 4
            if const_expr(rope):
                # One wait for the tile's four rows rather than one per row.
                tile_freqs = [self._load_freqs(row0 + i * TILE + r) for r in range(4)]
            for r in range_constexpr(4):
                acc = Vec.from_elements(
                    [Vec(accs[i][j])[r] for j in range(WAVE_TILES)], fx.Int32
                )
                a_scale = Vec.filled(WAVE_TILES, a_scales[i][r], fx.Float32)
                scaled = acc.to(fx.Float32) * (a_scale * b_scales)
                if const_expr(self.has_bias):
                    scaled = scaled + biases
                row = row0 + i * TILE + r
                if const_expr(prep):
                    scaled = self._prep_row(
                        scaled, gammas, tile_freqs[r], signs, row, col_base
                    )
                elif const_expr(self.amax):
                    self._store_row_max(scaled, row, col_base)
                scaled = scaled.to(fx.BFloat16)
                if const_expr(col_max is not None):
                    stored = scaled.to(fx.Float32)
                    row_valid = row < self.c_rows
                    peaks = [
                        peaks[k].maximumf(
                            row_valid.select(
                                stored[k].maximumf(-stored[k]), fx.Float32(0.0)
                            )
                        )
                        for k in range(8)
                    ]
                if const_expr(self.planar_cols):
                    width = self.planar_cols
                    c_index = (
                        (col // width) * (self.c_rows * width)
                        + row * width
                        + col % width
                    )
                else:
                    c_index = row * self.c_cols + col
                self._store_bf16x8(scaled, col_valid.select(c_index, oob))
        if const_expr(col_max is not None):
            # Lanes l, l + 16, l + 32 and l + 48 hold the same columns of different rows.
            for shift in (16, 32):
                peaks = [peak.maximumf(peak.shuffle_xor(shift, 64)) for peak in peaks]
            width = self.planar_cols if self.planar_cols else self.c_cols
            entries = (self.c_rows + WAVE_ROWS - 1) // WAVE_ROWS * width
            table = ptr_buf_tensor(
                col_max,
                fx.Float32,
                unit_elems=4,
                unit_stride=1,
                num_records_bytes=fx.Int64(entries) * 4,
            )
            # One lane group stores; the others' stores fall past the table and are dropped.
            index = (row_base // WAVE_ROWS) * width + col % width
            index = ((self.lane_id < 16) & col_valid).select(index, entries)
            buf_copy_store(
                table, index, Vec.from_elements(peaks[:4], fx.Float32), fx.Float32, 4
            )
            buf_copy_store(
                table,
                index + 4,
                Vec.from_elements(peaks[4:], fx.Float32),
                fx.Float32,
                4,
            )


def _mfma_i8(a, b, c):
    """``c is None`` starts the accumulation from an inline zero.

    The accumulators are pinned to AGPRs like hipBLASLt's: with the LLVM
    intrinsic, the allocator keeps them in VGPRs and the operands in AGPRs,
    leaving too few VGPRs and adding accumulator copies inside the loop.
    FlyDSL 0.4 has no gfx950 K64 INT8 MMA atom. LLVM does not see this as an
    MFMA, so ``_mfma_drain`` must separate the last one from any read of its
    result outside another MFMA. Remove both boundaries together once a stable
    K64 atom is available.
    """
    res_t = fx.Vector.make_type(4, fx.Int32)
    operands = [Vec(a).ir_value(), Vec(b).ir_value()]
    if c is None:
        asm, constraints = "v_mfma_i32_16x16x64_i8 $0, $1, $2, 0", "=a,v,v"
    else:
        asm, constraints = "v_mfma_i32_16x16x64_i8 $0, $1, $2, $3", "=a,v,v,0"
        operands.append(Vec(c).ir_value())
    return _llvm.inline_asm(
        res=res_t,
        operands_=operands,
        asm_string=asm,
        constraints=constraints,
        has_side_effects=True,
    )


def _mfma_drain(accs):
    """Return ``accs`` readable after the XDL-write -> v_accvgpr_read wait states.

    LLVM copies an accumulator out of its AGPRs right at the defining asm, so
    each one passes through an empty asm placed after the nops. This is paired
    with the opaque K64 MFMA above and is removable only with that boundary.
    """
    _llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="s_nop 7\ns_nop 7\ns_nop 7",
        constraints="",
        has_side_effects=True,
    )
    return [
        [
            _llvm.inline_asm(
                res=Vec(acc).ir_value().type,
                operands_=[Vec(acc).ir_value()],
                asm_string="",
                constraints="=a,0",
                has_side_effects=True,
            )
            for acc in row
        ]
        for row in accs
    ]


def compile_int8_gemm_4w(
    *,
    K: int,
    xcd_swizzle: int = 8,
    num_cu: int = 0,
    num_xcds: int | None = None,
    bias: bool = False,
    prep: "bool | PrepSpec" = False,
    prep_hadamard: bool = True,
    amax: bool = False,
    roles: int = 1,
    role_cols: int = 0,
    consumers: int = 0,
    eps: float = 1e-6,
    hooks: GemmHooks | None = None,
    role_hook=None,
    hook_key: str = "",
    first_block_hook=None,
    tile_hook=None,
    col_amax_offset: int | None = None,
):
    """``num_cu > 0`` makes the kernel persistent over that many workgroups.

    ``bias`` adds a per-column FP32 bias in the epilogue. B_scale then holds the
    N column scales followed by the N biases.

    ``prep`` applies a Q/K prep in the epilogue: a ``PrepSpec``, whose ``bias`` replaces
    ``bias``, or True for Wan's (``PrepSpec(bias=bias, hadamard=prep_hadamard)``: RMSNorm
    across heads, interleaved RoPE and the per-head Hadamard, without the RMS factor). With
    a norm, B_scale also holds the N norm weights after the scales and biases, multiplied
    by 128 ** -0.5 when the Hadamard is applied here. The launcher takes FP32 ``[M, 64]``
    cos and sin tables (one value per rotation pair) and writes FP32 ``[2, M, N // 128]``:
    per-head sums of squares of the projection (for a norm across heads), then per-head
    maxima of the stored output. For a norm across heads the caller scales each row by
    ``rsqrt(sum / N + eps)``. All three are flat: FlyDSL indexes a 2-D tensor argument
    column-major.

    ``amax`` (without ``prep``) writes only the maxima half of that buffer, for a
    projection the consumer quantizes as it is, such as V.

    ``roles > 1`` (with ``prep`` and ``num_cu``) runs a fused weight of ``roles`` blocks of
    ``role_cols`` columns, Q/K/V for 3, and stores C planar, ``[roles, M, role_cols]``. The
    prep applies to every role but the last. ``aux`` starts with an Int32 tile counter per
    role (``gemm_hooks``). Without hooks, the last ``consumers`` workgroups write the RMS
    factors of the prep roles to ``aux + 64``, FP32 ``[roles - 1, M]``, for a norm across
    heads.

    ``hooks`` (with ``roles > 1``) are fused producers that run inside the GEMM, under the
    contract in ``gemm_hooks``; with hooks, ``consumers`` may be 0. ``role_hook``,
    ``hook_key``, ``first_block_hook`` and ``tile_hook`` are the same, one by one.
    ``col_amax_offset`` (with ``roles > 1``) makes the last role's epilogue also write FP32
    ``[M / 128, role_cols]`` per-column maxima of its stored output to ``aux`` plus that many
    bytes, one row per 128 rows of C; they land before the tile is counted.

    A persistent workgroup loads the first two K tiles of its next output
    block during the last two of the current one, so the next block's K loop
    starts while the epilogue runs. That needs an even number of K tiles, so
    the next block's first tile lands in LDS buffer 0; otherwise each
    workgroup takes one block.
    """
    if hooks is not None:
        if (
            role_hook is not None
            or first_block_hook is not None
            or tile_hook is not None
        ):
            raise ValueError(
                "pass hooks or role_hook, first_block_hook and tile_hook, not both"
            )
        role_hook, first_block_hook, tile_hook = (
            hooks.role,
            hooks.first_block,
            hooks.tile,
        )
        hook_key = hooks.key
    if prep is True:
        prep_spec = PrepSpec(bias=bias, hadamard=prep_hadamard)
    elif prep is False or prep is None:
        prep_spec = None
    elif isinstance(prep, PrepSpec):
        if bias and not prep.bias:
            raise ValueError("bias=True conflicts with a PrepSpec without bias")
        prep_spec = prep
    else:
        raise TypeError(f"prep must be a bool or a PrepSpec, got {prep!r}")
    prep = prep_spec is not None
    if prep:
        bias = prep_spec.bias
    assert K % BLOCK_K == 0 and K >= 2 * BLOCK_K
    if num_xcds is None:
        num_xcds = get_cdna4_num_xcds() if xcd_swizzle > 0 else 8
    num_xcds = int(num_xcds)
    if num_xcds <= 0 or num_xcds > 8:
        raise ValueError(f"num_xcds must be in [1, 8], got {num_xcds}")
    K_ITERS = K // BLOCK_K
    B_K_STEP = 16 * BLOCK_K  # bytes per K128 tile inside a preshuffled 16-row strip
    HALF = WAVE_TILES // 2
    if num_cu > 0 and K_ITERS % 2:
        raise ValueError(f"persistent mode requires an even K128 tile count, got K={K}")
    persistent = num_cu > 0
    assert not persistent or num_cu % num_xcds == 0
    producers = num_cu - consumers
    if roles > 1 and (
        not (prep and persistent)
        or role_cols % BLOCK_N
        or (consumers < 1 and role_hook is None)
        or producers <= 0
        or producers % num_xcds
        or xcd_swizzle <= 0
    ):
        raise ValueError(
            "roles needs prep, a persistent swizzled grid, role_cols % 256 == 0 "
            "and 0 < consumers < num_cu with producers a multiple of the XCDs"
        )

    # Quadrants of the wave's tile grid by (A half, B half).
    def quadrant(a_half, b_half):
        return [
            (a_half * HALF + ia, b_half * HALF + jb)
            for jb in range(HALF)
            for ia in range(HALF)
        ]

    LL, LH, HL, HH = quadrant(0, 0), quadrant(1, 0), quadrant(0, 1), quadrant(1, 1)
    K_FIRST, K_SECOND = range(K_STEPS // 2), range(K_STEPS // 2, K_STEPS)
    ALL_K = range(K_STEPS)

    def kk_major(pairs, kks):
        return [(i, j, kk) for kk in kks for (i, j) in pairs]

    # MFMA order per phase. A-lo/B-lo operands are read in the previous tile's
    # phase 4, A-hi in phase 1, B-hi in phase 2; all lo operands must be dead
    # before phase 4 overwrites them with the next tile.
    LH_ORDER = kk_major(LH, ALL_K)
    LH_SPLIT = 6 * len(LH_ORDER) // 16
    PHASE_MFMAS = (
        kk_major(LL, K_FIRST),
        kk_major(LL, K_SECOND) + LH_ORDER[:LH_SPLIT],
        LH_ORDER[LH_SPLIT:] + kk_major(HL, ALL_K),
        kk_major(HH, ALL_K),
    )
    assert sum(len(p) for p in PHASE_MFMAS) == WAVE_TILES * WAVE_TILES * K_STEPS

    # Tile t + 2's A loads go in phase 2 and its B loads in phase 4, so the
    # waves' loads do not crowd one phase. At phase 3's wait, the A loads are
    # the only ones issued after tile t + 1's.
    P3_VMCNT = LOADS_PER_REGION
    # In a persistent workgroup, the previous block's epilogue (four stores
    # per A tile) also comes between tile 1's loads and tile 0's phase 3 wait.
    P3_VMCNT_TILE0 = P3_VMCNT + 4 * WAVE_TILES if persistent else P3_VMCNT
    # The epilogue's scale loads go in phase 1 of tile K_ITERS - 2, ahead of
    # the next block's LDS-DMA loads, and add to that tile's phase 3 wait.
    # Tile K_ITERS - 1's phase 3 wait, or with no next block tile
    # K_ITERS - 2's wait for everything, then covers them.
    SCALE_LOADS = (
        StoreCInt8Rows.SCALE_LOADS
        + (StoreCInt8Rows.BIAS_LOADS if bias else 0)
        + (StoreCInt8Rows.GAMMA_LOADS if prep and prep_spec.gammas else 0)
    )
    # Only a norm across heads leaves a per-row factor to the consumers.
    ACROSS_HEADS = prep and prep_spec.norm == "rms_across_heads"
    FIRST_BLOCK = first_block_hook is not None
    assert P3_VMCNT_TILE0 + SCALE_LOADS <= 63
    # Phase 1's last A-hi reads and phase 2's last B-hi reads must land
    # before the phase's barrier. Phase 2's A loads do not count against that
    # wait and run to the end of the phase.
    P1_READS_END = len(PHASE_MFMAS[0]) - 6
    P2_READS_END = len(PHASE_MFMAS[1]) - 6

    _kname = (
        f"flydsl_bpreshuffle_4w_{BLOCK_M}x{BLOCK_N}x{BLOCK_K}_I8_I8_B16_"
        f"1x{xcd_swizzle}_k{K}"
        + (f"_p{num_cu}" if persistent else "")
        + (f"_xcd{num_xcds}" if num_xcds != 8 else "")
        + ("_bias" if bias else "")
        + ("_prep" + prep_spec.suffix if prep else "")
        + (f"_eps{eps:g}" if (prep or consumers) and eps != 1e-6 else "")
        + ("_amax" if amax and not prep else "")
        + (f"_roles{roles}x{role_cols}c{consumers}" if roles > 1 else "")
        + (f"_hook{hook_key}" if role_hook is not None else "")
        + ("_fb" if first_block_hook is not None else "")
        + ("_th" if tile_hook is not None else "")
        + (f"_cmax{col_amax_offset}" if col_amax_offset is not None else "")
    )
    if col_amax_offset is not None and roles < 2:
        raise ValueError("col_amax_offset needs roles > 1")

    @fx.struct
    class SharedStorage:
        A_lds_0: fx.Array[fx.Int8, A_REGION_BYTES, 16]
        B_lds_0: fx.Array[fx.Int8, B_REGION_BYTES, 16]
        A_lds_1: fx.Array[fx.Int8, A_REGION_BYTES, 16]
        B_lds_1: fx.Array[fx.Int8, B_REGION_BYTES, 16]

    @flyc.kernel(name=_kname, known_block_size=[N_WAVES * 64, 1, 1])
    def kernel_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        Cos: fx.Tensor,
        Sin: fx.Tensor,
        Partials: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        aux: fx.Int64,
        h0: fx.Int64,
        h1: fx.Int64,
        h2: fx.Int64,
        h3: fx.Int64,
        h4: fx.Int64,
        h5: fx.Int64,
    ):
        m_blocks = ceildiv(c_m, BLOCK_M)
        n_blocks = ceildiv(c_n, BLOCK_N)
        num_blocks = m_blocks * n_blocks

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_lds = (lds.A_lds_0, lds.A_lds_1)
        b_lds = (lds.B_lds_0, lds.B_lds_1)

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 2
        wave_n = wave_id % 2

        def block_coords(block, role=0):
            # A persistent grid is a multiple of num_xcds workgroups or
            # covers all blocks, so block b stays on its workgroup's XCD.
            if const_expr(roles > 1):
                # ``block`` counts within the role, so it keeps that property.
                m, n = _xcd_swizzle_any(
                    m_blocks, role_cols // BLOCK_N, xcd_swizzle, block, num_xcds
                )
                return m, n + role * (role_cols // BLOCK_N)
            if const_expr(xcd_swizzle > 0):
                return _xcd_swizzle_any(
                    m_blocks,
                    n_blocks,
                    xcd_swizzle,
                    block,
                    num_xcds,
                )
            return split_row_major_2d(block, n_blocks)

        a_g2s = G2SGlobal(A, wave_id, A_BLOCK_STRIDE, a_lds)
        b_g2s = G2SGlobal(B_T, wave_id, B_BLOCK_STRIDE, b_lds)
        s2r = S2RLoader(wave_m, WAVE_TILES)

        def voffset_ops():
            """Return scheduled operations producing A/B LDS-DMA offsets.

            ``apply_scheduled_op`` passes the current (block_m, block_n)
            explicitly when each operation runs.
            """

            def op(g2s, preshuffled, step):
                def emit(coords):
                    row0, n_rows = (
                        (coords[1] * BLOCK_N, c_n)
                        if preshuffled
                        else (coords[0] * BLOCK_M, c_m)
                    )
                    lane_offset = _lane_offset(
                        lane_id, wave_id, K, preshuffled, row0, n_rows, step
                    )
                    return g2s.voffset(step, lane_offset)

                return ("voffset", int(preshuffled), emit)

            return [op(a_g2s, False, s) for s in range(LOADS_PER_REGION)] + [
                op(b_g2s, True, s) for s in range(LOADS_PER_REGION)
            ]

        # Operand (tile, kk): lane l reads k16 chunk kk * K_GROUPS + l // 16 of
        # A row l % 16 of the tile, or of B row 8 * (l % 16) + tile of the wave.
        # It sits at base[kk] + tile * A_BLOCK_STRIDE (A) or + tile * 16 (B).
        a_lane_base = [
            _a_lds_read_base(lane_id, wave_m, kk) for kk in range_constexpr(K_STEPS)
        ]
        b_lane_base = [
            _b_lds_read_base(lane_id, wave_n, kk) for kk in range_constexpr(K_STEPS)
        ]

        # K tile t lives in LDS buffer t % 2; ``buf`` is that parity, static
        # even where t is the loop's runtime value. ``voffsets`` (A, B) picks
        # the output block.
        def tile_loads(voffsets, tile_k):
            """A and B loads of a K tile; M0 must point at its buffer."""
            return (
                a_g2s.ops(voffsets[0], tile_k * BLOCK_K),
                b_g2s.ops(voffsets[1], tile_k * B_K_STEP),
            )

        a_ops = [[None] * K_STEPS for _ in range(WAVE_TILES)]
        b_ops = [[None] * K_STEPS for _ in range(WAVE_TILES)]

        def a_read(buf, i, kk):
            def emit():
                return s2r.load_one(a_lds[buf], a_lane_base[kk] + i * A_BLOCK_STRIDE)

            return ("a", i, kk, emit)

        def b_read(buf, j, kk):
            def emit():
                return s2r.load_one(b_lds[buf], b_lane_base[kk] + j * 16)

            return ("b", j, kk, emit)

        accs = [[None] * WAVE_TILES for _ in range(WAVE_TILES)]

        def apply_read(a_values, b_values, read):
            operand, tile, kk, emit = read
            value = emit()
            if operand == "a":
                a_values[tile][kk] = value
            else:
                b_values[tile][kk] = value
            return a_values, b_values

        def apply_scheduled_op(
            a_values,
            b_values,
            scale_values,
            next_coord_values,
            next_offset_values,
            op,
        ):
            if not isinstance(op, tuple):
                op()
                return (
                    a_values,
                    b_values,
                    scale_values,
                    next_coord_values,
                    next_offset_values,
                )
            kind = op[0]
            if kind in ("a", "b"):
                a_values, b_values = apply_read(a_values, b_values, op)
            elif kind == "scale":
                scale_values.append(op[1]())
            elif kind == "coords":
                next_coord_values = op[1]()
            elif kind == "effect":
                op[1]()
            else:
                _, operand, emit = op
                next_offset_values[operand].append(emit(next_coord_values))
            return (
                a_values,
                b_values,
                scale_values,
                next_coord_values,
                next_offset_values,
            )

        def do_mfma(acc_values, a_values, b_values, i, j, kk):
            acc_values[i][j] = _mfma_i8(
                a_values[i][kk], b_values[j][kk], acc_values[i][j]
            )
            return acc_values

        def spread(ops, first, last):
            """Place ``ops`` evenly after MFMAs ``first`` .. ``last - 1``."""
            n = len(ops)
            return [(first + q * (last - first) // n, op) for q, op in enumerate(ops)]

        def run_phase(
            acc_values,
            a_values,
            b_values,
            scale_values,
            next_coord_values,
            next_offset_values,
            mfmas,
            placed,
            wait=None,
        ):
            """Issue MFMAs with each memory op after its assigned MFMA.

            The four waves reach each phase together, so back-to-back LDS-DMA
            loads from all of them overrun the load path and are spread out.
            ``wait`` (vmcnt, lgkmcnt) goes before the last MFMA so the MFMA
            pipe stays busy while the following barrier resolves.
            ``sched_barrier(0)`` keeps LLVM from moving any op off its slot;
            FlyDSL 0.4 does not export it as a stable API.
            """
            for idx, (i, j, kk) in enumerate(mfmas):
                if const_expr(wait is not None and idx == len(mfmas) - 1):
                    fx.rocdl.s_waitcnt(vmcnt=wait[0], lgkmcnt=wait[1])
                    fx.rocdl.sched_barrier(0)
                acc_values = do_mfma(acc_values, a_values, b_values, i, j, kk)
                fx.rocdl.sched_barrier(0)
                for slot, op in placed:
                    if const_expr(slot == idx):
                        (
                            a_values,
                            b_values,
                            scale_values,
                            next_coord_values,
                            next_offset_values,
                        ) = apply_scheduled_op(
                            a_values,
                            b_values,
                            scale_values,
                            next_coord_values,
                            next_offset_values,
                            op,
                        )
                        fx.rocdl.sched_barrier(0)
            return (
                acc_values,
                a_values,
                b_values,
                scale_values,
                next_coord_values,
                next_offset_values,
            )

        def interleave(loads, reads):
            ops, step = [], len(reads) // max(len(loads), 1)
            for q, load in enumerate(loads):
                ops += [load] + reads[q * step : (q + 1) * step]
            return ops + reads[len(loads) * step :]

        def lo_reads(buf):
            return (
                [a_read(buf, i, kk) for kk in K_FIRST for i in range(HALF)]
                + [b_read(buf, j, kk) for kk in K_FIRST for j in range(HALF)]
                + [a_read(buf, i, kk) for kk in K_SECOND for i in range(HALF)]
                + [b_read(buf, j, kk) for kk in K_SECOND for j in range(HALF)]
            )

        def k_tile(
            acc_values,
            a_values,
            b_values,
            scale_values,
            next_coord_values,
            next_offset_values,
            buf,
            refill,
            has_next,
            p3_vmcnt=P3_VMCNT,
            p1_ops=(),
            p3_ops=(),
        ):
            """One K128 tile.

            ``refill`` (voffsets, tile_k) loads the K tile two ahead into the
            freed buffer; ``has_next`` reads the next one's lo operands.
            ``p1_ops`` go after phase 1's reads, ``p3_ops`` over phase 3.
            """
            a_hi_reads = [
                a_read(buf, HALF + i, kk) for kk in ALL_K for i in range(HALF)
            ]
            b_hi_reads = [
                b_read(buf, HALF + j, kk) for kk in ALL_K for j in range(HALF)
            ]
            p2_loads, p4_loads = [], []
            p1_m0, p3_m0 = [], []
            if const_expr(refill is not None):
                p2_loads, p4_loads = tile_loads(*refill)
                p2_loads = [("effect", op) for op in p2_loads]
                p4_loads = [("effect", op) for op in p4_loads]
                p1_m0 = [
                    (
                        len(PHASE_MFMAS[0]) - 2,
                        ("effect", lambda: a_g2s.set_m0(buf)),
                    )
                ]
                p3_m0 = [
                    (
                        len(PHASE_MFMAS[2]) - 2,
                        ("effect", lambda: b_g2s.set_m0(buf)),
                    )
                ]

            p1_placed = spread(a_hi_reads, 0, P1_READS_END)
            p1_placed += spread(list(p1_ops), P1_READS_END, len(PHASE_MFMAS[0]) - 1)
            (
                acc_values,
                a_values,
                b_values,
                scale_values,
                next_coord_values,
                next_offset_values,
            ) = run_phase(
                acc_values,
                a_values,
                b_values,
                scale_values,
                next_coord_values,
                next_offset_values,
                PHASE_MFMAS[0],
                p1_placed + p1_m0,
                wait=(None, 0),
            )
            # The stable fx.barrier is gpu.barrier, which also makes prior
            # memory accesses visible and so would wait for the LDS-DMA loads
            # this schedule keeps in flight across barriers. The bare
            # s_barrier is not a stable FlyDSL 0.4 export.
            fx.rocdl.s_barrier()
            p2_placed = spread(b_hi_reads, 0, P2_READS_END)
            p2_placed += spread(p2_loads, 1, len(PHASE_MFMAS[1]) - 1)
            (
                acc_values,
                a_values,
                b_values,
                scale_values,
                next_coord_values,
                next_offset_values,
            ) = run_phase(
                acc_values,
                a_values,
                b_values,
                scale_values,
                next_coord_values,
                next_offset_values,
                PHASE_MFMAS[1],
                p2_placed,
                wait=(None, 0),
            )
            fx.rocdl.s_barrier()
            if const_expr(has_next):
                p3_placed = spread(list(p3_ops), 0, len(PHASE_MFMAS[2]) - 2)
                p3_wait = p3_vmcnt if refill is not None else 0
                (
                    acc_values,
                    a_values,
                    b_values,
                    scale_values,
                    next_coord_values,
                    next_offset_values,
                ) = run_phase(
                    acc_values,
                    a_values,
                    b_values,
                    scale_values,
                    next_coord_values,
                    next_offset_values,
                    PHASE_MFMAS[2],
                    p3_placed + p3_m0,
                    wait=(p3_wait, None),
                )
                fx.rocdl.s_barrier()
                p4_ops = interleave(p4_loads, lo_reads(1 - buf))
                (
                    acc_values,
                    a_values,
                    b_values,
                    scale_values,
                    next_coord_values,
                    next_offset_values,
                ) = run_phase(
                    acc_values,
                    a_values,
                    b_values,
                    scale_values,
                    next_coord_values,
                    next_offset_values,
                    PHASE_MFMAS[3],
                    spread(p4_ops, 0, len(PHASE_MFMAS[3]) - 2),
                )
            else:
                (
                    acc_values,
                    a_values,
                    b_values,
                    scale_values,
                    next_coord_values,
                    next_offset_values,
                ) = run_phase(
                    acc_values,
                    a_values,
                    b_values,
                    scale_values,
                    next_coord_values,
                    next_offset_values,
                    PHASE_MFMAS[2],
                    [],
                )
                (
                    acc_values,
                    a_values,
                    b_values,
                    scale_values,
                    next_coord_values,
                    next_offset_values,
                ) = run_phase(
                    acc_values,
                    a_values,
                    b_values,
                    scale_values,
                    next_coord_values,
                    next_offset_values,
                    PHASE_MFMAS[3],
                    [],
                )
            return (
                acc_values,
                a_values,
                b_values,
                scale_values,
                next_coord_values,
                next_offset_values,
            )

        def lo_state(a_values, b_values):
            return [a_values[i][kk] for i in range(HALF) for kk in ALL_K] + [
                b_values[j][kk] for j in range(HALF) for kk in ALL_K
            ]

        def unpack_lo_state(values, a_values, b_values):
            values = list(values)
            a_values = [list(row) for row in a_values]
            b_values = [list(row) for row in b_values]
            for i in range_constexpr(HALF):
                for kk in range_constexpr(K_STEPS):
                    a_values[i][kk] = values.pop(0)
            for j in range_constexpr(HALF):
                for kk in range_constexpr(K_STEPS):
                    b_values[j][kk] = values.pop(0)
            return a_values, b_values

        # Values live across a K tile boundary: the accumulators and the next
        # tile's lo operands.
        def live_state(acc_values, a_values, b_values):
            accs_flat = [
                acc_values[i][j] for i in range(WAVE_TILES) for j in range(WAVE_TILES)
            ]
            return accs_flat + lo_state(a_values, b_values)

        def unpack_state(values, a_values, b_values):
            values = list(values)
            acc_values = [[None] * WAVE_TILES for _ in range(WAVE_TILES)]
            for i in range_constexpr(WAVE_TILES):
                for j in range_constexpr(WAVE_TILES):
                    acc_values[i][j] = values.pop(0)
            a_values, b_values = unpack_lo_state(values, a_values, b_values)
            return acc_values, a_values, b_values

        store_c = StoreCInt8Rows(
            A_scale,
            B_scale,
            C,
            c_m,
            c_n,
            None,
            WAVE_TILES,
            WAVE_TILES,
            has_bias=bias,
            prep=(Cos, Sin, Partials) if prep else None,
            prep_spec=prep_spec,
            eps=eps,
            amax_partials=Partials if amax and not prep else None,
            planar_cols=role_cols if roles > 1 else 0,
        )

        for role in range_constexpr(roles):
            if const_expr(roles > 1):
                block_end = m_blocks * (role_cols // BLOCK_N)
            else:
                block_end = num_blocks
            # Prologue: the first block's K tiles 0 and 1 in flight, wait for
            # tile 0, read its lo operands. Tile 0's phase 3 wait in a persistent
            # workgroup counts on an epilogue before it, so the first block also
            # waits for tile 1 here.
            block0 = fx.Int32(fx.block_idx.x)
            if const_expr(roles > 1):
                # Consumer workgroups run no output blocks.
                block0 = (block0 < producers).select(block0, block_end)
            coords = list(block_coords(block0, role))
            voffsets = ([], [])
            for _, operand, emit in voffset_ops():
                voffsets[operand].append(emit(coords))
            for t in range_constexpr(2):
                a_loads, b_loads = tile_loads(voffsets, t)
                for g2s, loads in ((a_g2s, a_loads), (b_g2s, b_loads)):
                    g2s.set_m0(t)
                    for op in loads:
                        op()
            fx.rocdl.s_waitcnt(vmcnt=0 if persistent else 2 * LOADS_PER_REGION)
            fx.rocdl.s_barrier()
            for read in lo_reads(0):
                a_ops, b_ops = apply_read(a_ops, b_ops, read)

            # Values live across an output block boundary: its coordinates and
            # offsets, and the lo operands of its first K tile.
            def block_state(coord_values, offset_values, a_values, b_values):
                return (
                    coord_values
                    + offset_values[0]
                    + offset_values[1]
                    + lo_state(a_values, b_values)
                )

            def unpack_block_state(values, a_values, b_values):
                values = list(values)
                coord_values = [values.pop(0) for _ in range(2)]
                offset_values = (
                    [values.pop(0) for _ in range(LOADS_PER_REGION)],
                    [values.pop(0) for _ in range(LOADS_PER_REGION)],
                )
                a_values, b_values = unpack_lo_state(values, a_values, b_values)
                return coord_values, offset_values, a_values, b_values

            def static_tile(
                acc_values,
                a_values,
                b_values,
                scale_values,
                next_coord_values,
                next_offset_values,
                t,
                next_voffsets,
                next_ops,
                scale_ops,
            ):
                """K tile ``t`` of a block, outside the runtime K loop.

                The last two refill from the next block at ``next_voffsets``;
                tile 0 spreads ``next_ops`` over phase 3 and tile K_ITERS - 2
                issues ``scale_ops``.
                """
                refill = None
                if const_expr(t + 2 < K_ITERS):
                    # The current block's offsets, read when the tile is traced.
                    refill = (voffsets, t + 2)  # noqa: B023
                elif const_expr(persistent):
                    refill = (next_voffsets, t + 2 - K_ITERS)
                p3_vmcnt = P3_VMCNT_TILE0 if t == 0 else P3_VMCNT
                p1_ops = []
                if const_expr(t == K_ITERS - 2):
                    p1_ops = scale_ops
                    p3_vmcnt += SCALE_LOADS
                has_next = t + 1 < K_ITERS or persistent
                p3_ops = next_ops if t == 0 else ()
                return k_tile(
                    acc_values,
                    a_values,
                    b_values,
                    scale_values,
                    next_coord_values,
                    next_offset_values,
                    t % 2,
                    refill,
                    has_next,
                    p3_vmcnt,
                    p1_ops,
                    p3_ops,
                )

            n_pairs = max(0, (K_ITERS - 4) // 2)
            if const_expr(roles > 1):
                grid = fx.Int32(producers)
            else:
                grid = fx.Int32(fx.grid_dim.x)
            for block, bstate in range(
                block0,
                block_end,
                grid,
                init=block_state(coords, voffsets, a_ops, b_ops),
            ):
                coords, voffsets, a_ops, b_ops = unpack_block_state(
                    bstate, a_ops, b_ops
                )
                accs = [[None] * WAVE_TILES for _ in range(WAVE_TILES)]

                # The next block's coordinates and offsets, computed during tile
                # 0's phase 3 unless tile 0 already loads that block. The last
                # block reloads itself; nothing reads those loads.
                next_coords, next_voffsets, next_ops = [], ([], []), []
                scales = []
                if const_expr(persistent):
                    this_block = fx.Int32(block)

                    def next_coords_op(
                        this_block=this_block, grid=grid, block_end=block_end, role=role
                    ):
                        nxt = this_block + grid
                        nxt = (nxt < block_end).select(nxt, this_block)
                        return list(block_coords(nxt, role))

                    next_ops = [("coords", next_coords_op)] + voffset_ops()
                    if const_expr(K_ITERS == 2):
                        for op in next_ops:
                            (
                                a_ops,
                                b_ops,
                                scales,
                                next_coords,
                                next_voffsets,
                            ) = apply_scheduled_op(
                                a_ops,
                                b_ops,
                                scales,
                                next_coords,
                                next_voffsets,
                                op,
                            )
                        next_ops = []

                row_base = coords[0] * BLOCK_M + wave_m * WAVE_ROWS
                col_base = coords[1] * BLOCK_N + wave_n * WAVE_ROWS
                scale_ops = [
                    ("scale", op) for op in store_c.scale_load_ops(row_base, col_base)
                ]
                # Tiles 0 and 1 start the accumulators. A runtime loop then takes
                # the tiles that refill from this block, two per trip so the
                # buffer parity is static; a fully unrolled K loop overflows the
                # instruction cache.
                (
                    accs,
                    a_ops,
                    b_ops,
                    scales,
                    next_coords,
                    next_voffsets,
                ) = static_tile(
                    accs,
                    a_ops,
                    b_ops,
                    scales,
                    next_coords,
                    next_voffsets,
                    0,
                    next_voffsets,
                    next_ops,
                    scale_ops,
                )
                (
                    accs,
                    a_ops,
                    b_ops,
                    scales,
                    next_coords,
                    next_voffsets,
                ) = static_tile(
                    accs,
                    a_ops,
                    b_ops,
                    scales,
                    next_coords,
                    next_voffsets,
                    1,
                    next_voffsets,
                    next_ops,
                    scale_ops,
                )
                if const_expr(n_pairs > 0):
                    for pair, state in range(
                        0, n_pairs, 1, init=live_state(accs, a_ops, b_ops)
                    ):
                        accs, a_ops, b_ops = unpack_state(state, a_ops, b_ops)
                        t0 = fx.Int32(pair) * 2 + 2
                        (
                            accs,
                            a_ops,
                            b_ops,
                            scales,
                            next_coords,
                            next_voffsets,
                        ) = k_tile(
                            accs,
                            a_ops,
                            b_ops,
                            scales,
                            next_coords,
                            next_voffsets,
                            0,
                            (voffsets, t0 + 2),
                            True,
                        )
                        (
                            accs,
                            a_ops,
                            b_ops,
                            scales,
                            next_coords,
                            next_voffsets,
                        ) = k_tile(
                            accs,
                            a_ops,
                            b_ops,
                            scales,
                            next_coords,
                            next_voffsets,
                            1,
                            (voffsets, t0 + 3),
                            True,
                        )
                        results = yield live_state(accs, a_ops, b_ops)
                    accs, a_ops, b_ops = unpack_state(results, a_ops, b_ops)
                for t in range_constexpr(2 + 2 * n_pairs, K_ITERS):
                    (
                        accs,
                        a_ops,
                        b_ops,
                        scales,
                        next_coords,
                        next_voffsets,
                    ) = static_tile(
                        accs,
                        a_ops,
                        b_ops,
                        scales,
                        next_coords,
                        next_voffsets,
                        t,
                        next_voffsets,
                        next_ops,
                        scale_ops,
                    )

                if const_expr(roles > 1):
                    col_max = None
                    if const_expr(col_amax_offset is not None and role == roles - 1):
                        col_max = aux + col_amax_offset
                    store_c.store_tiles(
                        _mfma_drain(accs),
                        row_base,
                        col_base,
                        scales,
                        prep=role < roles - 1,
                        col_max=col_max,
                    )
                    # Every wave's stores of the tile land before the tile is counted.
                    fx.rocdl.s_waitcnt(vmcnt=0)
                    fx.barrier()
                    if const_expr(tile_hook is not None):
                        tile_hook(
                            role, aux, (h0, h1, h2, h3, h4, h5), c_m, row_base, col_base
                        )
                    if fx.thread_idx.x == 0:
                        fx.atomic_add(
                            _global_ptr(aux + role * 4, fx.Int32),
                            fx.Int32(1),
                            syncscope=fx.rocdl.SyncScope.Agent,
                            ordering=fx.AtomicOrdering.Release,
                        )
                    # A trace-time branch around a runtime one; they do not merge.
                    if const_expr(FIRST_BLOCK and role > 0):  # noqa: SIM102
                        if fx.Int32(block) == block0:
                            first_block_hook(role, aux, (h0, h1, h2, h3, h4, h5), c_m)
                else:
                    store_c.store_tiles(_mfma_drain(accs), row_base, col_base, scales)
                if const_expr(persistent):
                    coords = list(next_coords)
                    voffsets = (
                        list(next_voffsets[0]),
                        list(next_voffsets[1]),
                    )
                _ = yield block_state(coords, voffsets, a_ops, b_ops)

            if const_expr(persistent):
                # LDS-DMA must not land in LDS that a later workgroup owns.
                fx.rocdl.s_waitcnt(vmcnt=0)
            if const_expr(role_hook is not None):
                role_hook(role, aux, (h0, h1, h2, h3, h4, h5), c_m)

        if const_expr(roles > 1 and consumers > 0):
            partials_view = store_c.partials_div
            maxima_base = store_c.maxima_base
            load_atom = store_c.scale_atom_4
            load_reg = store_c.reg_f32_4
            if fx.Int32(fx.block_idx.x) >= producers:
                tid = fx.Int32(fx.thread_idx.x)
                worker = (fx.Int32(fx.block_idx.x) - producers) * (N_WAVES * 64) + tid
                heads = c_n // 128
                tiles_per_role = m_blocks * (role_cols // BLOCK_N)
                factors_out = ptr_buf_tensor(
                    aux + 64, fx.Float32, num_records_bytes=c_m * 4 * (roles - 1)
                )
                for role in range_constexpr(roles):
                    if tid == 0:
                        spin_until_ge_i32_agent(aux + role * 4, tiles_per_role)
                    fx.barrier()
                    fx.memory_fence(
                        ordering=fx.AtomicOrdering.Acquire,
                        syncscope=fx.rocdl.SyncScope.Agent,
                    )
                    for row, state in range(
                        worker,
                        c_m,
                        fx.Int32(consumers * N_WAVES * 64),
                        init=[fx.Float32(0.0)],
                    ):
                        row = fx.Int32(row)
                        current = state[0]
                        sq = fx.Float32(0.0)
                        peak = fx.Float32(0.0)
                        for quad in range_constexpr(role_cols // 512):
                            head = role * (role_cols // 128) + quad * 4
                            maxima = _load_f32x4(
                                partials_view,
                                maxima_base + row * heads + head,
                                load_atom,
                                load_reg,
                            )
                            for k in range_constexpr(4):
                                peak = peak.maximumf(maxima[k])
                            if const_expr(role < roles - 1 and ACROSS_HEADS):
                                sums = _load_f32x4(
                                    partials_view,
                                    row * heads + head,
                                    load_atom,
                                    load_reg,
                                )
                                for k in range_constexpr(4):
                                    sq = sq + sums[k]
                        if const_expr(role < roles - 1 and ACROSS_HEADS):
                            factor = fmath.rsqrt(sq * (1.0 / role_cols) + eps)
                            buf_copy_store(
                                factors_out, role * c_m + row, factor, fx.Float32
                            )
                            peak = peak * factor
                        elif const_expr(role < roles - 1):
                            # The epilogue already normalized (or there is no norm).
                            buf_copy_store(
                                factors_out,
                                role * c_m + row,
                                fx.Float32(1.0),
                                fx.Float32,
                            )
                        current = current.maximumf(peak)
                        result = yield [current]
                    local = fx.Float32(result)
                    for shift in (32, 16, 8, 4, 2, 1):
                        local = local.maximumf(local.shuffle_xor(shift, 64))
                    if tid % 64 == 0:
                        fx.atomic_fmax(
                            _global_ptr(aux + 16 + role * 4, fx.Float32),
                            local,
                            is_positive=True,
                            syncscope=fx.rocdl.SyncScope.Agent,
                        )

    value_attrs = {
        "rocdl.waves_per_eu": 1,
        "rocdl.flat_work_group_size": f"{N_WAVES * 64},{N_WAVES * 64}",
    }

    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        if const_expr(persistent):
            grid_x = (grid_x > num_cu).select(fx.Int32(num_cu), grid_x)
        # Without prep the kernel never reads its cos, sin and partials arguments.
        kernel_gemm(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            A_scale,
            A_scale,
            A_scale,
            c_m,
            c_n,
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            value_attrs=value_attrs,
        ).launch(grid=(grid_x, 1, 1), block=(N_WAVES * 64, 1, 1), stream=stream)

    @flyc.jit
    def launch_gemm_prep(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        Cos: fx.Tensor,
        Sin: fx.Tensor,
        Partials: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        if const_expr(persistent):
            grid_x = (grid_x > num_cu).select(fx.Int32(num_cu), grid_x)
        kernel_gemm(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            Cos,
            Sin,
            Partials,
            c_m,
            c_n,
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            value_attrs=value_attrs,
        ).launch(grid=(grid_x, 1, 1), block=(N_WAVES * 64, 1, 1), stream=stream)

    @flyc.jit
    def launch_gemm_amax(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        Partials: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        if const_expr(persistent):
            grid_x = (grid_x > num_cu).select(fx.Int32(num_cu), grid_x)
        # Without prep the kernel never reads its cos and sin arguments.
        kernel_gemm(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            A_scale,
            A_scale,
            Partials,
            c_m,
            c_n,
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            value_attrs=value_attrs,
        ).launch(grid=(grid_x, 1, 1), block=(N_WAVES * 64, 1, 1), stream=stream)

    @flyc.jit
    def launch_gemm_roles(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        Cos: fx.Tensor,
        Sin: fx.Tensor,
        Partials: fx.Tensor,
        aux: fx.Int64,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_gemm(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            Cos,
            Sin,
            Partials,
            c_m,
            c_n,
            aux,
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            value_attrs=value_attrs,
        ).launch(grid=(num_cu, 1, 1), block=(N_WAVES * 64, 1, 1), stream=stream)

    @flyc.jit
    def launch_gemm_roles_hook(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        Cos: fx.Tensor,
        Sin: fx.Tensor,
        Partials: fx.Tensor,
        aux: fx.Int64,
        h0: fx.Int64,
        h1: fx.Int64,
        h2: fx.Int64,
        h3: fx.Int64,
        h4: fx.Int64,
        h5: fx.Int64,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_gemm(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            Cos,
            Sin,
            Partials,
            c_m,
            c_n,
            aux,
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            value_attrs=value_attrs,
        ).launch(grid=(num_cu, 1, 1), block=(N_WAVES * 64, 1, 1), stream=stream)

    if roles > 1 and role_hook is not None:
        launcher = launch_gemm_roles_hook
    elif roles > 1:
        launcher = launch_gemm_roles
    elif prep:
        launcher = launch_gemm_prep
    elif amax:
        launcher = launch_gemm_amax
    else:
        launcher = launch_gemm
    return set_slp_vectorize_once_used(launcher, False)
