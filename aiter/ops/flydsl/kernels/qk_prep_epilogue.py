# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""A GEMM epilogue with a compile-time Q/K prep, on the 16x16 MFMA accumulator layout.

``StoreCRows`` stores a wave's 128x128 tile of C as BF16 rows. Its accumulator source
(``Int32Source``, ``Fp32Source``) says what the host's accumulators hold and which scales
they need. It can add a bias, apply a Q/K prep (``qk_prep.PrepSpec``) and write per-head
and per-column maxima. ``WaveTile`` is the accumulator layout: the elements of C that a lane holds.
``mfma_drain`` makes the accumulators of a host's inline-asm MFMAs readable.
"""

import flydsl.expr as fx

# The LLVM dialect is used only for inline assembly: the scheduled scale loads and the
# accumulator hazard boundary.
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.qk_prep import HEAD_DIM, _max_abs, _row_max, _vector
from aiter.ops.flydsl.kernels.tensor_shim import buf_copy_store, ptr_buf_tensor

TILE = 16  # MFMA output tile
WAVE_ROWS = 128  # rows (and columns) of C per wave
WAVE_TILES = WAVE_ROWS // TILE
VECTOR_LOADS = 2  # a lane loads its 8 values of a column vector in two 16-byte loads


class WaveTile:
    """The elements of C that a lane holds of the wave's 128x128 tile at (``row_base``,
    ``col_base``), in the 16x16 MFMA output layout.

    Lane ``l`` of MFMA tile (i, j) holds row ``16 i + 4 (l // 16) + r`` in register ``r``
    and column ``8 (l % 16) + j``. Register ``r`` of the eight j tiles is then the lane's 8
    adjacent columns of one row, one 16-byte store, and the 16 lanes of a lane group hold
    a 128-column row of the tile, one head of the prep. Preshuffled B has N % 16 == 0, so
    a lane's 8 columns are all inside C or all past it.

    Every index the epilogue forms into C, the column operand, the tables and the maxima
    starts from these. They are integer expressions rather than FlyDSL layouts because
    the scheduled loads take a byte offset (``_load_f32x4``) and the planar C and the
    per-head entries are not affine in the lane. They are built where they are used, so
    none is live across the host's K loop.
    """

    def __init__(self, lane, row_base, col_base, c_rows, c_cols):
        self.lane = lane
        self.row_base, self.col_base = row_base, col_base
        self._c_rows, self._c_cols = c_rows, c_cols

    def row(self, i, r=0):
        """The row of register ``r`` of MFMA tile row ``i``."""
        row = self.row_base + (self.lane // 16) * 4 + i * TILE
        return row + r if r else row

    def column(self, half=0):
        """The first of the lane's 8 columns, or with ``half`` its second 4."""
        column = self.col_base + (self.lane % 16) * 8
        return column + half * 4 if half else column

    @property
    def col_valid(self):
        return self.column() < self._c_cols

    @property
    def head(self):
        """The tile's head: its 128 columns are one."""
        return self.col_base // HEAD_DIM

    @property
    def head_lane(self):
        """The lane that stores its row's per-head values: the first of the lane group,
        in a tile inside C (with N % 256 == 128, a block's second tile is past N)."""
        return ((self.lane % 16) == 0) & (self.col_base < self._c_cols)


def _load_f32x4(rsrc, byte_offset):
    # A stable copy atom would enter LLVM's vmcnt accounting, but these loads must be
    # ordered beside opaque LDS-DMA loads that LLVM cannot count. Retain the scheduled
    # raw load until both operations share one stable dependency model.
    # LLVM inserts no hazard wait states before inline asm. Under SGPR pressure it
    # restores the resource with v_readlane right before the load, and a VALU SGPR write
    # needs five wait states before a VMEM instruction reads that SGPR: the s_nop 4.
    return _llvm.inline_asm(
        res=fx.Vector.make_type(4, fx.Float32),
        operands_=[fx.Int32(byte_offset).ir_value(), rsrc],
        asm_string="s_nop 4\nbuffer_load_dwordx4 $0, $1, $2, 0 offen",
        constraints="=v,v,s",
        has_side_effects=True,
    )


class Int32Source:
    """INT32 accumulators of an integer MFMA, scaled in the epilogue by M FP32 row scales
    (``A_scale``) and by the N column scales that start the column operand.

    The loads are scheduled asm (``_load_f32x4``): a persistent host issues them among
    the next block's LDS-DMA loads, which LLVM cannot count, so LLVM leaves them out of
    its vmcnt waits as well, and the host waits for them before the stores.
    """

    column_scales = True
    ROW_LOADS = WAVE_TILES

    def __init__(self, A_scale, c_rows):
        rows = fx.rocdl.make_buffer_tensor(
            A_scale, num_records_bytes=fx.Int64(c_rows) * 4
        )
        self.rows_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(rows))

    def row_loads(self, tile):
        return [
            lambda i=i: _load_f32x4(self.rows_rsrc, tile.row(i) * 4)
            for i in range(WAVE_TILES)
        ]

    def column_load(self, store, element):
        return lambda: _load_f32x4(store.columns_rsrc, element() * 4)

    def values(self, accs, i, r, row_scales, column_scales):
        """Register ``r`` of MFMA tile row ``i`` over the 8 j tiles, scaled, in FP32."""
        acc = Vec.from_elements(
            [Vec(accs[i][j])[r] for j in range(WAVE_TILES)], fx.Int32
        )
        row_scale = Vec.filled(WAVE_TILES, Vec(row_scales[i])[r], fx.Float32)
        return acc.to(fx.Float32) * (row_scale * column_scales)


class Fp32Source:
    """FP32 accumulators that a block-scaled MFMA has already scaled: the epilogue applies
    no scales, and the column operand starts with the bias, or the prep's vectors.

    The loads are stable copies: the host issues them after its last LDS-DMA load has
    landed, so LLVM's own waits cover them.
    """

    column_scales = False
    ROW_LOADS = 0

    def row_loads(self, tile):
        return []

    def column_load(self, store, element):
        return lambda: store._load_vec4(store.columns_div, element())

    def values(self, accs, i, r, row_scales, column_scales):
        """Register ``r`` of MFMA tile row ``i`` over the 8 j tiles."""
        return Vec.from_elements(
            [Vec(accs[i][j])[r] for j in range(WAVE_TILES)], fx.Float32
        )


class _PrepContext:
    """What a ``PrepOp`` sees of the epilogue (``qk_prep``)."""

    def __init__(self, store, tile, columns):
        self.lane = store.lane
        self.row = None
        self._store = store
        self._tile = tile
        self._columns = columns

    def column(self, name):
        return self._columns[name]

    def table(self, name, row, index, count):
        return self._store._load_table(name, row, index, count)

    def leave(self, value):
        self._store._leave(value, self._tile, self.row)


class StoreCRows:
    """Store a wave's 128x128 tile of C as BF16 rows, in the layout of ``WaveTile``.

    ``source`` is the accumulator source (``Int32Source``). ``columns`` is the FP32
    column operand, N values per vector: the source's column scales if it has them, then
    with ``bias`` the N biases, then with a prep the spec's column vectors in spec order
    (``PrepSpec.pack_columns``).

    ``prep`` is a ``PrepSpec``, with its FP32 ``tables`` (``[M, table_width]``,
    ``PrepSpec.pack_tables``). ``partials``, with a prep or ``amax``, is FP32
    ``[2, M, N // 128]``: the row-factor op's per-head partials, then per-head maxima of
    the stored output. A prep and ``amax`` need N % 128 == 0. ``planar_cols`` stores C
    planar, ``[N // planar_cols, M, planar_cols]``.
    """

    def __init__(
        self,
        source,
        columns,
        C,
        c_rows,
        c_cols,
        *,
        bias=False,
        prep=None,
        tables=None,
        partials=None,
        amax=False,
        planar_cols=0,
    ):
        self.source = source
        self.c_rows = c_rows
        self.c_cols = c_cols
        self.lane = fx.thread_idx.x % 64
        self.bias = bias
        self.spec = prep
        self.amax = prep is not None or amax
        self.planar_cols = planar_cols
        self.vectors = (
            (("scale",) if source.column_scales else ())
            + (("bias",) if bias else ())
            + (prep.columns if prep is not None else ())
        )
        c_tensor = fx.rocdl.make_buffer_tensor(
            C, max_size=False, num_records_bytes=c_rows * c_cols * 2
        )
        self.c_div = fx.logical_divide(c_tensor, fx.make_layout(1, 1))
        self.f32x4_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
        self.f32_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        self.reg_f32x4 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
        self.reg_f32 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
        if self.amax and prep is None:
            self._init_partials(c_rows, c_cols, partials)
        if prep is not None:
            # One buffer resource per table, based at the table's first column, so the
            # loads of every table of a row share one offset register.
            self.tables_div = {}
            entries = fx.Int64(c_rows) * prep.table_width
            for name, _ in prep.tables:
                offset = prep.table_offset(name)
                view = fx.make_view(fx.get_iter(tables) + offset, fx.get_layout(tables))
                self.tables_div[name] = fx.logical_divide(
                    fx.rocdl.make_buffer_tensor(
                        view, max_size=False, num_records_bytes=(entries - offset) * 4
                    ),
                    fx.make_layout(1, 1),
                )
            self._init_partials(c_rows, c_cols, partials)
        self.bf16x8_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        self.reg_bf16x8 = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
        columns_tensor = fx.rocdl.make_buffer_tensor(
            columns, num_records_bytes=fx.Int64(c_cols) * 4 * len(self.vectors)
        )
        self.columns_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(columns_tensor))
        self.columns_div = fx.logical_divide(columns_tensor, fx.make_layout(1, 1))

    @staticmethod
    def load_count(source, bias, prep):
        """The most loads ``load_ops`` issues for a tile: with the prep's vectors."""
        vectors = source.column_scales + bias
        vectors += len(prep.columns) if prep is not None else 0
        return source.ROW_LOADS + VECTOR_LOADS * vectors

    def tile(self, row_base, col_base):
        return WaveTile(self.lane, row_base, col_base, self.c_rows, self.c_cols)

    def _tile_vectors(self, apply_prep):
        if apply_prep:
            return self.vectors
        return self.vectors[
            : len(self.vectors) - len(self.spec.columns if self.spec else ())
        ]

    def load_ops(self, tile, apply_prep=None):
        """The tile's loads, as ops that issue them: the source's row scales, then two per
        column vector the tile uses, those of the prep only when it applies the prep.
        ``store_tiles`` takes their results, in order, as ``scales``."""
        apply_prep = self.spec is not None if apply_prep is None else apply_prep
        ops = self.source.row_loads(tile)
        for name in self._tile_vectors(apply_prep):
            k = self.vectors.index(name)
            for half in range(VECTOR_LOADS):

                def element(k=k, half=half):
                    column = tile.column(half)
                    if k == 0:
                        return column
                    return (self.c_cols if k == 1 else k * self.c_cols) + column

                ops.append(self.source.column_load(self, element))
        return ops

    def _store_bf16x8(self, values, c_index):
        fx.memref_store_vec(values, self.reg_bf16x8)
        fx.copy(
            self.bf16x8_atom,
            self.reg_bf16x8,
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

    def _store_row_max(self, values, tile, row):
        """Store the largest magnitude of this row's head, as it will be stored in BF16."""
        stored = values.to(fx.BFloat16).to(fx.Float32)
        maximum = _row_max(_max_abs(stored))
        store = (row < self.c_rows) & tile.head_lane
        index = self.maxima_base + row * self.heads + tile.head
        self._store_f32(maximum, store.select(index, self.partials_oob))

    def _load_vec4(self, div, index):
        fx.copy(self.f32x4_atom, fx.slice(div, (None, fx.Int32(index))), self.reg_f32x4)
        return Vec(fx.memref_load_vec(self.reg_f32x4))

    def _store_f32(self, value, index):
        fx.memref_store_vec(fx.full(1, value, fx.Float32), self.reg_f32)
        fx.copy(
            self.f32_atom,
            self.reg_f32,
            fx.slice(self.partials_div, (None, fx.Int32(index))),
        )

    def _load_table(self, name, row, index, count):
        """``count`` values (a multiple of 4) of table ``name`` from ``index`` in ``row``,
        as a list."""
        # Each table's view starts at its first column (__init__), so one element index
        # serves every table of the row and its loads share one offset register.
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

    def _leave(self, value, tile, row):
        """Store the row-factor op's partial of this row's head."""
        store = (row < self.c_rows) & tile.head_lane
        index = row * self.heads + tile.head
        self._store_f32(value, store.select(index, self.partials_oob))

    def _prep_row(self, x, ctx, states, data, tile, row):
        """The spec's ops on the 8 channels this lane holds of one row, then the head's
        maximum."""
        ctx.row = row
        for op, state, values in zip(self.spec.ops, states, data):
            x = op.apply(x, ctx, state, values)
        result = _vector(x)
        self._store_row_max(result, tile, row)
        return result

    def _c_index(self, tile, row, oob):
        """Element index in C of the lane's 8 columns of ``row``, or ``oob``, past C, where
        the store is dropped."""
        col = tile.column()
        if const_expr(self.planar_cols):
            width = self.planar_cols
            index = (col // width) * (self.c_rows * width) + row * width + col % width
            # A row past M would land in the next plane.
            return (tile.col_valid & (row < self.c_rows)).select(index, oob)
        # Dense, a row past M is already past C.
        return tile.col_valid.select(row * self.c_cols + col, oob)

    def store_tiles(self, accs, tile, scales, apply_prep=None, col_max=None):
        """``scales`` are the results of ``load_ops`` for the same ``apply_prep``, landed.

        ``apply_prep=False`` stores a prep-built instance's tile with maxima only (V).
        ``col_max`` (an address) also stores each column's largest magnitude over the
        wave's 128 rows, as stored in BF16: FP32 at [row // 128][column within its plane].
        """
        apply_prep = self.spec is not None if apply_prep is None else apply_prep
        vectors = self._tile_vectors(apply_prep)

        def vector(name):
            """This lane's 8 values of the column vector ``name``."""
            index = self.source.ROW_LOADS + VECTOR_LOADS * vectors.index(name)
            lo, hi = (Vec(s) for s in scales[index : index + 2])
            return Vec.from_elements(
                [lo[j] for j in range(4)] + [hi[j] for j in range(4)], fx.Float32
            )

        row_scales = scales[: self.source.ROW_LOADS]
        column_scales = vector("scale") if self.source.column_scales else None
        if const_expr(self.bias):
            biases = vector("bias")
        ops = self.spec.ops if apply_prep else ()
        ctx = None
        if const_expr(apply_prep):
            columns = {name: vector(name) for name in self.spec.columns}
            ctx = _PrepContext(self, tile, columns)
        oob = fx.Int32(self.c_rows * self.c_cols)
        states = [op.setup(ctx) for op in ops]
        peaks = [fx.Float32(0.0) for _ in range(8)]
        loads = any(op.tables for op in ops)
        for i in range_constexpr(WAVE_TILES):
            tile_data = [[None] * len(ops)] * 4
            if const_expr(loads):
                # One wait for the tile's four rows rather than one per row.
                tile_data = [self._load_row(ctx, tile.row(i, r)) for r in range(4)]
            for r in range_constexpr(4):
                scaled = self.source.values(accs, i, r, row_scales, column_scales)
                if const_expr(self.bias):
                    scaled = scaled + biases
                row = tile.row(i, r)
                if const_expr(apply_prep):
                    scaled = self._prep_row(
                        scaled, ctx, states, tile_data[r], tile, row
                    )
                elif const_expr(self.amax):
                    self._store_row_max(scaled, tile, row)
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
                self._store_bf16x8(scaled, self._c_index(tile, row, oob))
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
            index = (tile.row_base // WAVE_ROWS) * width + tile.column() % width
            index = ((self.lane < 16) & tile.col_valid).select(index, entries)
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

    For a host whose MFMAs are inline asm, which LLVM does not see as MFMAs: it would
    copy an accumulator out of its AGPRs right at the defining asm, so each passes
    through an empty asm placed after the nops. The three ``s_nop 7`` are sized for the
    two hosts whose accumulators the epilogue reads right after the last MFMA: the INT8
    host's ``v_mfma_i32_16x16x64_i8`` and the A6W6 host's
    ``v_mfma_scale_f32_16x16x128_f8f6f4``. A host with another MFMA, or other work between
    its last MFMA and the read, sizes its own, as the standalone A6W6 kernel does.
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
