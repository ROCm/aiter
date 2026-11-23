# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""A GEMM epilogue with a compile-time Q/K prep, on the 16x16 MFMA accumulator layout.

``StoreCRows`` stores a wave's 128x128 tile of C as BF16 rows, from INT32 accumulators
with row and column scales or from FP32 accumulators that a block-scaled MFMA has already
scaled. It can add a bias, apply a Q/K prep (``PrepSpec``) and write per-head and
per-column maxima. ``mfma_drain`` makes inline-asm MFMA accumulators readable before the
epilogue reads them.
"""

from dataclasses import dataclass

import flydsl.expr as fx

# The LLVM dialect is used only for inline assembly: the scheduled scale loads and the
# accumulator hazard boundary.
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.rocdl import update_dpp
from flydsl.expr.typing import ReductionOp
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.gemm_a8w8_8wave import StoreC
from aiter.ops.flydsl.kernels.tensor_shim import buf_copy_store, ptr_buf_tensor

TILE = 16  # MFMA output tile
WAVE_ROWS = 128  # rows (and columns) of C per wave
WAVE_TILES = WAVE_ROWS // TILE


# DPP moves within a 16-lane row and the lane bit that orders each pair: quad_perm
# [1,0,3,2] and [2,3,0,1] pair lanes i ^ 1 and i ^ 2, row_half_mirror i ^ 7 and
# row_ror:8 i ^ 8. The four masks span the row's lane bits, so a butterfly over them
# is a Hadamard transform, with rows in a different order than Sylvester's.
_ROW_BUTTERFLY = ((0xB1, 1), (0x4E, 2), (0x141, 4), (0x128, 8))
_ROW_ROR8 = 0x128  # lane i of a 16-lane row reads lane i ^ 8


@dataclass(frozen=True)
class PrepSpec:
    """The Q/K prep of ``StoreCRows``, fixed at compile time.

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


class StoreCRows(StoreC):
    """Scale and store a wave's INT32 accumulators as BF16 rows.

    Lane ``l`` of MFMA tile (i, j) holds row ``16 i + 4 (l // 16) + r`` in
    register ``r`` and column ``8 (l % 16) + j``, so register ``r`` of the
    eight j tiles is one 16-byte store and 16 lanes cover a 256-byte row.
    Preshuffled B has N % 16 == 0: a store is fully inside C or past column N.

    ``fp32_acc`` takes FP32 accumulators that are already scaled, as a block-scaled
    MFMA leaves them, in the same layout: no row or column scales are loaded or applied.
    B_scale keeps its column scales' place, so the biases and norm weights stay at the
    same offsets.
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
        fp32_acc=False,
    ):
        super().__init__(A_scale, B_scale, C, c_rows, c_cols, *args)
        self.has_bias = has_bias
        self.planar_cols = planar_cols
        self.fp32_acc = fp32_acc
        self.prep = prep is not None
        self.spec = prep_spec
        self.has_gamma = self.prep and prep_spec.gammas
        # Indices of the first bias and norm-weight loads in scale_load_ops' result.
        self.bias_index = 0 if fp32_acc else WAVE_TILES + 2
        self.gamma_index = self.bias_index + (self.BIAS_LOADS if has_bias else 0)
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
        if fp32_acc:
            self.columns_div = fx.logical_divide(sb_tensor, fx.make_layout(1, 1))

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
        which it cannot count. The caller waits before the stores. With
        ``fp32_acc`` they are stable loads, issued after the host's last LDS-DMA
        load has landed.
        """

        def op(rsrc, element):
            if self.fp32_acc:
                # No LDS-DMA loads are in flight where an fp32_acc host issues these,
                # so they are stable loads: LLVM waits for them before any use or copy.
                return lambda: self._load_freq4(self.columns_div, element())
            return lambda: self._load_f32x4(rsrc, element() * 4)

        def row(i):
            return row_base + (self.lane_id // 16) * 4 + i * TILE

        def col(half):
            return col_base + (self.lane_id % 16) * 8 + half * 4

        ops = []
        if not self.fp32_acc:
            ops += [op(self.sa_rsrc, lambda i=i: row(i)) for i in range(WAVE_TILES)]
            ops += [op(self.sb_rsrc, lambda h=h: col(h)) for h in (0, 1)]
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
        if const_expr(not self.fp32_acc):
            a_scales = [Vec(s) for s in scales[:WAVE_TILES]]
            b_lo, b_hi = (Vec(s) for s in scales[WAVE_TILES : WAVE_TILES + 2])
            b_scales = Vec.from_elements(
                [b_lo[j] for j in range(4)] + [b_hi[j] for j in range(4)], fx.Float32
            )
        if const_expr(self.has_bias):
            bias_lo, bias_hi = (
                Vec(s) for s in scales[self.bias_index : self.bias_index + 2]
            )
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
                if const_expr(self.fp32_acc):
                    scaled = Vec.from_elements(
                        [Vec(accs[i][j])[r] for j in range(WAVE_TILES)], fx.Float32
                    )
                else:
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


def mfma_drain(accs):
    """Return ``accs`` readable after the XDL-write -> v_accvgpr_read wait states.

    LLVM copies an accumulator out of its AGPRs right at the defining asm, so
    each one passes through an empty asm placed after the nops. This is paired
    with the GEMMs' opaque inline-asm MFMAs and is removable only with them.
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
