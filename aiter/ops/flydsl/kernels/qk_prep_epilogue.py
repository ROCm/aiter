# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""A GEMM epilogue with a compile-time Q/K prep, on the 16x16 MFMA accumulator layout.

``StoreCRows`` stores a wave's 128x128 tile of C as BF16 rows, from INT32 accumulators
with row and column scales or from FP32 accumulators that a block-scaled MFMA has already
scaled. It can add a bias, apply a Q/K prep (``qk_prep.PrepSpec``) and write per-head and
per-column maxima. ``mfma_drain`` makes inline-asm MFMA accumulators readable before the
epilogue reads them.
"""

import flydsl.expr as fx

# The LLVM dialect is used only for inline assembly: the scheduled scale loads and the
# accumulator hazard boundary.
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.gemm_a8w8_8wave import StoreC
from aiter.ops.flydsl.kernels.qk_prep import HEAD_DIM, _max_abs, _row_max, _vector
from aiter.ops.flydsl.kernels.tensor_shim import buf_copy_store, ptr_buf_tensor

TILE = 16  # MFMA output tile
WAVE_ROWS = 128  # rows (and columns) of C per wave
WAVE_TILES = WAVE_ROWS // TILE


class _PrepContext:
    """What a ``PrepOp`` sees of the epilogue (``qk_prep``)."""

    def __init__(self, store, columns, col_base):
        self.lane = store.lane_id
        self.row = None
        self._store = store
        self._columns = columns
        self._col_base = col_base

    def column(self, name):
        return self._columns[name]

    def table(self, name, row, index, count):
        return self._store._load_table(name, row, index, count)

    def leave(self, value):
        self._store._leave(value, self.row, self._col_base)


class StoreCRows(StoreC):
    """Scale and store a wave's INT32 accumulators as BF16 rows.

    Lane ``l`` of MFMA tile (i, j) holds row ``16 i + 4 (l // 16) + r`` in
    register ``r`` and column ``8 (l % 16) + j``, so register ``r`` of the
    eight j tiles is one 16-byte store and 16 lanes cover a 256-byte row.
    Preshuffled B has N % 16 == 0: a store is fully inside C or past column N.

    ``fp32_acc`` takes FP32 accumulators that are already scaled, as a block-scaled
    MFMA leaves them, in the same layout: no row or column scales are loaded or applied.
    B_scale keeps its column scales' place, so the bias and the prep's column vectors
    stay at the same offsets.

    ``prep`` is ``(tables, partials)``: the spec's tables, FP32 ``[M, table_width]``, and
    the FP32 ``[2, M, N // 128]`` partials (``_init_partials``).
    """

    SCALE_LOADS = WAVE_TILES + 2
    # B_scale holds the N column scales, then with a bias N FP32 biases, then N FP32
    # values per column vector of the prep, in spec order. A lane loads its 8 columns of
    # each vector in two loads.
    VECTOR_LOADS = 2

    @classmethod
    def load_count(cls, bias, spec, fp32_acc=False):
        """How many loads ``scale_load_ops`` issues."""
        vectors = bias + (len(spec.columns) if spec is not None else 0)
        return (0 if fp32_acc else cls.SCALE_LOADS) + cls.VECTOR_LOADS * vectors

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
        self.columns = prep_spec.columns if self.prep else ()
        # Indices of the bias's and the first column vector's loads in scale_load_ops'
        # result.
        self.bias_index = 0 if fp32_acc else WAVE_TILES + 2
        self.columns_index = self.bias_index + (self.VECTOR_LOADS if has_bias else 0)
        self.amax = self.prep or amax_partials is not None
        if self.amax and not self.prep:
            self._init_partials(c_rows, c_cols, amax_partials)
        if self.prep:
            tables, partials = prep
            # One buffer resource per table, based at the table's first column, so the
            # loads of every table of a row share one offset register.
            self.tables_div = {}
            entries = fx.Int64(c_rows) * prep_spec.table_width
            for name, _ in prep_spec.tables:
                offset = prep_spec.table_offset(name)
                view = fx.make_view(fx.get_iter(tables) + offset, fx.get_layout(tables))
                self.tables_div[name] = fx.logical_divide(
                    fx.rocdl.make_buffer_tensor(
                        view, max_size=False, num_records_bytes=(entries - offset) * 4
                    ),
                    fx.make_layout(1, 1),
                )
            self._init_partials(c_rows, c_cols, partials)
        self.out_atom_8 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        self.reg_bf16_8 = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
        sa_tensor = fx.rocdl.make_buffer_tensor(
            A_scale, num_records_bytes=fx.Int64(c_rows) * 4
        )
        vectors = 1 + has_bias + len(self.columns)
        sb_tensor = fx.rocdl.make_buffer_tensor(
            B_scale, num_records_bytes=fx.Int64(c_cols) * 4 * vectors
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
                return lambda: self._load_vec4(self.columns_div, element())
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
        for k in range(len(self.columns)):
            base = (1 + self.has_bias + k) * self.c_cols
            ops += [op(self.sb_rsrc, lambda h=h, b=base: b + col(h)) for h in (0, 1)]
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
        self.heads = c_cols // HEAD_DIM
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
        index = self.maxima_base + row * self.heads + col_base // HEAD_DIM
        self._store_f32(maximum, store.select(index, self.partials_oob))

    def _load_vec4(self, div, index):
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

    def _load_table(self, name, row, index, count):
        """``count`` values (a multiple of 4) of table ``name`` from ``index`` in ``row``,
        as a list."""
        base = row * self.spec.table_width + index
        values = []
        for k in range(0, count, 4):
            chunk = self._load_vec4(self.tables_div[name], base + k if k else base)
            values += [chunk[j] for j in range(4)]
        return values

    def _load_row(self, ctx, row):
        """Each op's data for ``row``; rows past M read row 0."""
        row = (row < self.c_rows).select(row, fx.Int32(0))
        return [op.load(ctx, row) for op in self.spec.ops]

    def _leave(self, value, row, col_base):
        """Store the row-factor op's partial of this row's head."""
        store = (row < self.c_rows) & ((self.lane_id % 16) == 0)
        index = row * self.heads + col_base // HEAD_DIM
        self._store_f32(value, store.select(index, self.partials_oob))

    def _prep_row(self, x, ctx, states, data, row, col_base):
        """The spec's ops on the 8 channels this lane holds of one row, then the head's
        maximum."""
        ctx.row = row
        for op, state, values in zip(self.spec.ops, states, data):
            x = op.apply(x, ctx, state, values)
        result = _vector(x)
        self._store_row_max(result, row, col_base)
        return result

    def store_tiles(self, accs, row_base, col_base, scales, prep=None, col_max=None):
        """``scales`` comes from ``scale_load_ops`` and must have landed.

        ``prep=False`` stores a prep-built instance's tile with maxima only (V).
        ``col_max`` (an address) also stores each column's largest magnitude over the
        wave's 128 rows, as stored in BF16: FP32 at [row // 128][column within its plane].
        """
        prep = self.prep if prep is None else prep
        if const_expr(bool(self.columns) and not prep):
            # Column-vector loads went out for this tile too; their registers stay
            # claimed until here, past the wait that lands them.
            claimed = scales[self.columns_index :]
            _llvm.inline_asm(
                res=None,
                operands_=[Vec(s).ir_value() for s in claimed],
                asm_string="",
                constraints=",".join("v" for _ in claimed),
                has_side_effects=True,
            )
        row0 = row_base + (self.lane_id // 16) * 4
        col = col_base + (self.lane_id % 16) * 8

        def vector(index):
            """This lane's 8 columns of the vector whose two loads start at ``index``."""
            lo, hi = (Vec(s) for s in scales[index : index + 2])
            return Vec.from_elements(
                [lo[j] for j in range(4)] + [hi[j] for j in range(4)], fx.Float32
            )

        if const_expr(not self.fp32_acc):
            a_scales = [Vec(s) for s in scales[:WAVE_TILES]]
            b_scales = vector(WAVE_TILES)
        if const_expr(self.has_bias):
            biases = vector(self.bias_index)
        ops = self.spec.ops if prep else ()
        if const_expr(prep):
            columns = {
                name: vector(self.columns_index + self.VECTOR_LOADS * k)
                for k, name in enumerate(self.columns)
            }
            ctx = _PrepContext(self, columns, col_base)
        oob = fx.Int32(self.c_rows * self.c_cols)
        col_valid = col < self.c_cols
        states = [op.setup(ctx) for op in ops]
        peaks = [fx.Float32(0.0) for _ in range(8)]
        loads = any(op.tables for op in ops)
        for i in range_constexpr(WAVE_TILES):
            tile_data = [[None] * len(ops)] * 4
            if const_expr(loads):
                # One wait for the tile's four rows rather than one per row.
                tile_data = [self._load_row(ctx, row0 + i * TILE + r) for r in range(4)]
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
                        scaled, ctx, states, tile_data[r], row, col_base
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
