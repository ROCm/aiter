# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.
#
# ruff: noqa: B023
# The scatter builds small closures over the tile loops and calls each one
# inside the same iteration, so the loop variable always holds the current
# value. Binding them as default arguments is not possible -- `bias_val` only
# exists on the has_bias path -- and per-line waivers do not survive
# `ruff format`, which moves the flagged column onto continuation lines.

"""What the conv3d kernels are assembled from, as ``gemm_a16w16_gfx950_utils.py``
is for the GEMM.

Four layers, in the order a kernel uses them:

1. What the machine fixes rather than what this operator chose -- the MFMA
   shape, the wave width, the vector widths a gfx950 load and an LDS write
   come in, and thin wrappers over the rocdl intrinsics that spell a barrier,
   a scalar broadcast or a buffer atomic.
2. ``ConvGeometry``: the implicit GEMM's shape, as the convolution's own
   implies it. Derived once and shared, because the gather, the grid and the
   epilogue must all be sized by the same one.
3. ``LaunchGrid`` / ``block_coords``: how the work is spread over blocks, and
   what each block decodes to find the tile it owns.
4. ``OutputScatterPlan`` / ``OutputScatter``: how C goes back to NCDHW.

The gather that makes A look like a matrix is the one piece that did not fit
here -- it is large enough to be its own module, ``conv3d_im2col.py``, which
builds on this one. The tile sizes, the barrier interval and the pipeline
depth are the algorithm's own and stay in ``conv3d_implicit_gfx950.py``.

Running a compiled launcher is a host concern and lives in
``../conv_kernels.py`` with the rest of the dispatch; only the stream coercion
both launchers take is here.
"""

from typing import NamedTuple

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

# One MFMA instruction's output tile, and how many accumulator values of it a
# lane holds. gfx950's bf16 MFMA is 16x16x16 with 4 values per lane.
MFMA_M = 16
MFMA_N = 16
MFMA_C_VALUES = 4

WARP_SIZE = 64

BF16_BYTES = 2

# Elements per gather/DMA vector: 8 bf16 is the 16 bytes a buffer_load_lds
# moves per lane, and the width ds_write_b128 wants on the far side.
LDG_VEC = 8

# An offset no buffer descriptor can hold a record for, so the access is
# dropped rather than performed: *2 = 0xFFFFFF00 bytes (~4.2950 GB), just under
# the 2^32 a voffset spans. The gather sends a padded tap here to read zero,
# and the epilogue a masked element to write nowhere -- in both cases turning
# a predicate into an address, which needs no branch.
OOB_SENTINEL_ELEM = 0x7FFFFF80
OOB_SENTINEL_BYTES = OOB_SENTINEL_ELEM * BF16_BYTES

# Compile hints applied to both conv3d kernels. Empty by default; a caller that
# needs to pass FlyDSL a hint sets it before the first compile.
CONV_COMPILE_HINTS = {}


def _as_stream(stream):
    return stream if hasattr(stream, "_is_stream_param") else fx.Stream(stream)


def buffer_atomic_add(vdata, rsrc, offset, soffset, aux):
    """Buffer-resource atomic fadd (AMD ``raw.ptr.buffer.atomic.fadd``).

    Upstream lives in flydsl's repo-level ``kernels/common/mem_ops.py``, which
    its wheel does not ship, so aiter keeps this one-line equivalent alongside
    the vendored ``buffer_ops`` / ``vector`` modules. Operates on a buffer
    resource plus byte offset, not an ``!llvm.ptr``.
    """
    return fx.rocdl.raw_ptr_buffer_atomic_fadd(vdata, rsrc, offset, soffset, aux)


def barrier(vmcnt=0, lgkmcnt=None):
    """Wait on the named counters, then barrier.

    Not gpu.barrier(): which counters this waits on is the whole point. The
    caller names only the ones it needs, so the DMAs prefetching the next K
    tiles stay in flight across the barrier instead of being drained by it.
    Naming a counter here is a scheduling decision.
    """
    fx.rocdl.s_waitcnt(vmcnt=vmcnt, lgkmcnt=lgkmcnt)
    fx.rocdl.s_barrier()


def sgpr(x):
    """Broadcast lane 0's value into a scalar register."""
    return fx.Int64(fx.rocdl.readfirstlane(T.i64, fx.Int64(x)))


def flat_buffer_view(ptr, elems, num_records_bytes):
    """A 1-D buffer view, on which ``slice(view, (None, off))`` is element ``off``.

    Both the gather and the epilogue address their buffer by a flat element
    index, so the view is one-dimensional over the buffer rather than the
    tensor's own n-D layout: dividing that by a 1-element tile makes a slice
    exactly one element, with no coordinate decomposition. ``elems`` only
    shapes the view -- the sentinel above deliberately points past it, and
    num_records, not the layout, is what turns that into a zero-fill.
    """
    buf = fx.rocdl.make_buffer_ptr(ptr, num_records_bytes=num_records_bytes)
    return fx.logical_divide(
        fx.make_view(buf, fx.make_layout(elems, 1)), fx.make_layout(1, 1)
    )


def in_range(v, hi):
    return (v >= 0) & (v < fx.Int64(hi))


def dil(tap, factor):
    """Scale a filter tap by its dilation, folding the factor away when it is 1."""
    return tap * factor if const_expr(factor != 1) else tap


def gather_valid(base, *masks):
    """AND the masks that apply; a None mask is one the caller does not need."""
    for m in masks:
        if const_expr(m is not None):
            base = base & m
    return base


# ---------------------------------------------------------------------------
# The implicit GEMM's shape
# ---------------------------------------------------------------------------


class ConvGeometry(NamedTuple):
    """The implicit GEMM's shape, as the convolution's own shape implies it.

    A row is one output element, so there are ``npq = N * Do * Ho * Wo`` of
    them, and the K axis is one group's filter footprint, ``crs``. That makes
    this both what the gather maps between and what the grid and the epilogue
    are sized by, which is why it is derived once and shared: two derivations
    that drifted apart would put the epilogue on a different grid than the
    gather, for no visible reason.
    """

    do: int
    ho: int
    wo: int
    dhw: int
    hw_o: int
    npq: int
    cgp: int
    crs: int


def make_conv_geometry(param):
    """The ConvGeometry of one ``Conv3dImplicitParam``.

    Dilation only stretches the filter's footprint, so it moves the output
    extents but leaves the K axis (CRS) alone.
    """
    cgp = param.c // param.groups
    do = (param.d + 2 * param.pt - (param.dt * (param.kt - 1) + 1)) // param.st + 1
    ho = (param.h + 2 * param.ph - (param.dh * (param.kh - 1) + 1)) // param.sh + 1
    wo = (param.w + 2 * param.pw - (param.dw * (param.kw - 1) + 1)) // param.sw + 1
    dhw = do * ho * wo
    return ConvGeometry(
        do=do,
        ho=ho,
        wo=wo,
        dhw=dhw,
        hw_o=ho * wo,
        npq=param.n * dhw,
        cgp=cgp,
        crs=cgp * param.kt * param.kh * param.kw,
    )


# ---------------------------------------------------------------------------
# How the work is spread over blocks, and what each block owns
#
# Four independent things are folded onto three grid axes:
#
# - M over ``grid.x``, spilling into ``grid.z`` as "M chunks" when the tile
#   count passes what one axis holds;
# - N over ``grid.y``, over-provisioned to ``groups * tiles_per_group`` so a
#   grouped conv's N tail is per group rather than global;
# - split-K over the rest of ``grid.z``;
# - and, when M fits one axis, a WGM swizzle over x/y that walks WGM rows of M
#   before moving on in N, so concurrent blocks share B tiles.
#
# The grid, its limits and the decode share one arithmetic and so live
# together: deriving the decode from anything but the grid it was launched
# with is how a block ends up owning a tile nobody sized for.
# ---------------------------------------------------------------------------


# A grid dimension is 32-bit in blocks on x and 16-bit on y/z; x is further
# capped so that block_id.x * block_threads stays inside 32 bits.
MAX_GRID_YZ = 65535


class LaunchGrid(NamedTuple):
    """The grid one compiled conv3d launches on, and what a block decodes with.

    A NamedTuple for the same reason the other plans are: only tuples and
    scalars reach FlyDSL's cache key.
    """

    # The launch itself.
    grid_x: int
    grid_y: int
    grid_z: int
    block_threads: int

    # M: how many tiles there are, and how they fold onto x and z.
    grid_m: int
    m_chunks: int
    tile_m: int
    row_chk: bool
    wgm: int

    # N: per-group tiling, and whether the last tile of a group is partial.
    tile_n: int
    tiles_per_group: int
    n_tail: bool
    groups: int
    kg: int
    cgp: int

    # K: the split, in whole tiles.
    tile_k: int
    tiles_per_split: int
    splitk: int
    use_splitk: bool


def make_launch_grid(param, geom, *, tile_m, tile_n, tile_k, block_threads):
    """The LaunchGrid for one problem and launch config, or an assertion."""
    k, groups = param.k, param.groups
    kg = k // groups
    npq = geom.npq

    tiles_per_group = (kg + tile_n - 1) // tile_n
    n_tail = kg % tile_n != 0
    grid_y = groups * tiles_per_group

    k_tiles = (geom.crs + tile_k - 1) // tile_k
    splitk = max(1, min(param.splitk, k_tiles))
    tiles_per_split = k_tiles // splitk

    grid_m = (npq + tile_m - 1) // tile_m
    max_grid_x = 0xFFFFFFFF // block_threads
    grid_x = min(grid_m, max_grid_x)
    m_chunks = (grid_m + grid_x - 1) // grid_x

    assert grid_y <= MAX_GRID_YZ, (
        f"grid.y = {grid_y} exceeds the {MAX_GRID_YZ}-block limit"
    )
    assert m_chunks * splitk <= MAX_GRID_YZ, (
        f"grid.z = {m_chunks} M-chunks x {splitk} splits exceeds the {MAX_GRID_YZ}-block limit"
    )

    return LaunchGrid(
        grid_x=grid_x,
        grid_y=grid_y,
        grid_z=m_chunks * splitk,
        block_threads=block_threads,
        grid_m=grid_m,
        m_chunks=m_chunks,
        tile_m=tile_m,
        # The last M tile is partial, or chunking over-provisioned the x axis:
        # either way some block owns rows past npq and must not write them.
        row_chk=(npq % tile_m != 0) or (grid_x * m_chunks > grid_m),
        # Chunked M already uses z, so a swizzle over x/y would reorder blocks
        # that are no longer adjacent in M. WGM only applies to the flat case.
        wgm=1 if m_chunks > 1 else max(1, int(param.wgm)),
        tile_n=tile_n,
        tiles_per_group=tiles_per_group,
        n_tail=n_tail,
        groups=groups,
        kg=kg,
        cgp=geom.cgp,
        tile_k=tile_k,
        tiles_per_split=tiles_per_split,
        splitk=splitk,
        use_splitk=splitk > 1,
    )


class BlockCoords(NamedTuple):
    """Where in the GEMM this block's tile sits. Device values, not constants."""

    m_offset: object
    n_offset: object
    n_local: object
    ch_base: object
    k_off: object


def block_coords(grid):
    """Decode this block's ids into the tile it owns.

    ``n_local`` is the column within the group and ``n_offset`` the global
    one; they differ only for a grouped conv, where the N grid is per group.
    ``ch_base`` is the group's first input channel, which only the gather
    needs, and is None when there is one group.
    """
    if const_expr(grid.m_chunks > 1):
        m_chunk = fx.Int64(gpu.block_id("z")) % fx.Int64(grid.m_chunks)
        m_offset = (
            fx.Int64(gpu.block_id("x")) + m_chunk * fx.Int64(grid.grid_x)
        ) * grid.tile_m
        n_tile = fx.Int32(gpu.block_id("y"))
    elif const_expr(grid.wgm > 1):
        pid = fx.Int64(gpu.block_id("x")) + fx.Int64(gpu.block_id("y")) * fx.Int64(
            grid.grid_m
        )
        blocks_per_swizzle = fx.Int64(grid.wgm * grid.grid_y)
        swizzle_id = pid // blocks_per_swizzle
        first_m = swizzle_id * fx.Int64(grid.wgm)
        # The last swizzle group is short when grid_m is not a multiple of WGM.
        swizzle_rows = fx.min(fx.Int64(grid.grid_m) - first_m, fx.Int64(grid.wgm))
        local = pid % blocks_per_swizzle
        m_offset = (first_m + (local % swizzle_rows)) * grid.tile_m
        n_tile = local // swizzle_rows
    else:
        m_offset = fx.Int32(gpu.block_id("x")) * grid.tile_m
        n_tile = fx.Int32(gpu.block_id("y"))

    if const_expr(grid.groups > 1):
        gi = n_tile // grid.tiles_per_group
        n_local = (n_tile % grid.tiles_per_group) * grid.tile_n
        n_offset = gi * grid.kg + n_local
        ch_base = gi * grid.cgp
    else:
        n_offset = n_tile * grid.tile_n
        n_local = n_offset
        ch_base = None

    if const_expr(grid.use_splitk):
        if const_expr(grid.m_chunks > 1):
            split_idx = fx.Int64(gpu.block_id("z")) // fx.Int64(grid.m_chunks)
        else:
            split_idx = fx.Int64(gpu.block_id("z"))
        k_off = split_idx * (grid.tiles_per_split * grid.tile_k)
    else:
        k_off = 0

    return BlockCoords(
        m_offset=m_offset,
        n_offset=n_offset,
        n_local=n_local,
        ch_base=ch_base,
        k_off=k_off,
    )


# ---------------------------------------------------------------------------
# The scatter: C back to a convolution's output
#
# The GEMM produces a (npq, K) tile; NCDHW wants it as (N, K, Do, Ho, Wo), so
# a row has to be decomposed back into its sample and spatial position and the
# store strides along K instead of along the row. That, plus what the tail of
# an over-provisioned grid must not write, plus split-K accumulating into fp32
# staging with atomics, is what ``store`` does.
#
# Only ``n == 1``, contiguous NCDHW, no split-K and a small enough output let
# the four accumulator values of an MFMA atom land contiguously; that case
# takes a single 64-bit store and every other one stores element by element.
# Which case applies is decided once, in the plan.
# ---------------------------------------------------------------------------


# Split-K accumulates through a buffer descriptor, whose num_records is a
# 32-bit byte count, so the fp32 staging buffer has to fit one.
SPLITK_MAX_STAGING_BYTES = 0xFFFFFFFF


class OutputScatterPlan(NamedTuple):
    """What the epilogue knows before the kernel runs.

    A NamedTuple for the same reason ``Im2colPlan`` is one: FlyDSL keys a
    compiled kernel on the scalar values its closure captures, and anything
    that is not a scalar or a tuple silently drops out of that key.
    """

    n: int
    k: int
    kg: int
    groups: int
    out_ndhwc: bool
    has_bias: bool

    # The grid C is produced on, shared with the gather and the launch config.
    geom: ConvGeometry

    # MFMA atoms per wave along M and N: the epilogue walks them.
    mi_m: int
    mi_n: int

    # How the store is done, decided once below.
    use_splitk: bool
    big_out: bool
    y_bytes: int
    row_chk: bool
    n_tail: bool
    need_chk: bool
    route_store: bool
    vec_store: bool


def make_output_scatter_plan(
    param, geom, *, kg, mi_m, mi_n, use_splitk, row_chk, n_tail
):
    """An OutputScatterPlan for one problem and launch config, or an assertion.

    ``row_chk`` and ``n_tail`` say whether the grid over-provisions M and N,
    which only the caller's grid arithmetic knows; everything else about how C
    is written follows from the problem and is derived here.
    """
    n, k, out_ndhwc = param.n, param.k, param.out_ndhwc
    npq, dhw = geom.npq, geom.dhw

    big_out = (n * k * geom.do * geom.ho * geom.wo * BF16_BYTES) > 0x7FFFFFFF
    y_bytes = npq * k * (4 if use_splitk else BF16_BYTES)

    assert not use_splitk or npq * k * 4 <= SPLITK_MAX_STAGING_BYTES, (
        f"split-K staging {npq * k * 4}B exceeds the {SPLITK_MAX_STAGING_BYTES}B buffer window"
    )

    need_chk = row_chk or n_tail
    return OutputScatterPlan(
        n=n,
        k=k,
        kg=kg,
        groups=param.groups,
        out_ndhwc=out_ndhwc,
        has_bias=param.has_bias,
        geom=geom,
        mi_m=mi_m,
        mi_n=mi_n,
        use_splitk=use_splitk,
        big_out=big_out,
        y_bytes=y_bytes,
        row_chk=row_chk,
        n_tail=n_tail,
        need_chk=need_chk,
        # Routing a masked element to the OOB sentinel drops it without a
        # branch, but only where the store goes through a buffer descriptor:
        # split-K atomics and the 64-bit BIG_OUT path have no such address.
        route_store=need_chk and not use_splitk and not big_out,
        # The four values of an MFMA atom are consecutive rows, so they are
        # only contiguous in memory where a row's neighbour is the next
        # spatial position: NCDHW, one sample, one store, no staging.
        vec_store=(
            (n == 1)
            and (not use_splitk)
            and (dhw % MFMA_C_VALUES == 0)
            and (not big_out)
            and (not out_ndhwc)
        ),
    )


class OutputScatter:
    """The epilogue of one kernel: build with the other descriptors, then store.

    Stateless per block, unlike the gather -- the accumulator is written once,
    so the block's coordinates are arguments of ``store`` rather than of a
    separate bind.
    """

    def __init__(self, plan, y, bias, elem_ty):
        self._plan = plan
        self._y = y
        self._elem_ty = elem_ty

        y_buf = fx.rocdl.make_buffer_tensor(y, num_records_bytes=plan.y_bytes)
        if const_expr(plan.use_splitk):
            # buffer_atomic_add needs the raw !llvm.ptr<8> descriptor, not a tensor.
            self._y_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(y_buf))
        else:
            self._y_div = fx.logical_divide(
                fx.Tensor(
                    fx.make_view(
                        fx.get_iter(y_buf),
                        fx.make_layout(plan.geom.npq * plan.k, 1),
                    )
                ),
                fx.make_layout(1, 1),
            )
            self._y_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), elem_ty)
            self._y_reg_1 = fx.make_rmem_tensor(1, elem_ty)
            if const_expr(plan.vec_store):
                self._y_atom_4 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), elem_ty)
                self._y_reg_4 = fx.make_rmem_tensor(MFMA_C_VALUES, elem_ty)
        if const_expr(plan.has_bias):
            self._bias_div = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(bias), fx.make_layout(1, 1)
            )
            self._bias_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
            self._bias_reg = fx.make_rmem_tensor(1, fx.Float32)

        # A type, not an operation: constructing it emits no IR.
        self._big_st_ptr_ty = fx.PointerType.get(
            elem_ty.ir_type, fx.AddressSpace.Global, BF16_BYTES
        )

    def store(self, acc, *, m_offset, n_offset, n_local, c_row, c_col):
        """Write this block's accumulator out.

        ``c_row`` / ``c_col`` are the tile-local coordinates of the
        accumulator's elements, taken from the same tiled_mma that owns
        ``acc`` so the epilogue cannot drift from the MMA's own partitioning.
        """
        plan = self._plan
        elem_ty = self._elem_ty
        if const_expr(plan.big_out):
            self._y_elem_base = fx.Int64(fx.ptrtoint(fx.get_iter(self._y)))

        if const_expr(plan.has_bias and not plan.use_splitk):
            bias_vals = self._load_bias(n_offset, n_local, c_col)

        for mi in range_constexpr(plan.mi_m):
            row_base = m_offset + fx.get_scalar(c_row[MFMA_C_VALUES * mi])
            for ni in range_constexpr(plan.mi_n):
                col, col_loc = self._cols(ni, n_offset, n_local, c_col)
                a = fx.Vector(acc[None, mi, ni].load())
                if const_expr(plan.has_bias and not plan.use_splitk):
                    bias_val = bias_vals[ni]

                if const_expr(plan.vec_store):
                    row0 = fx.Int64(row_base)
                    off_nk0 = col * plan.geom.dhw + row0

                    def _emit_vec():
                        vals = []
                        for i in range_constexpr(MFMA_C_VALUES):
                            cval = (
                                (a[i] + bias_val) if const_expr(plan.has_bias) else a[i]
                            )
                            vals.append(cval.to(elem_ty))
                        v4 = fx.Vector.from_elements(vals, dtype=elem_ty)
                        fx.memref_store_vec(v4, self._y_reg_4)
                        fx.copy(
                            self._y_atom_4,
                            self._y_reg_4,
                            fx.slice(
                                self._y_div,
                                (None, self._route(off_nk0, row0, col_loc)),
                            ),
                        )

                    if const_expr(plan.need_chk and not plan.route_store):
                        if self._valid(row0, col_loc):
                            _emit_vec()
                    else:
                        _emit_vec()
                    continue

                for i in range_constexpr(MFMA_C_VALUES):
                    row = fx.Int64(row_base + i)
                    off_sk = row * plan.k + col
                    off_nk = self._off_nk(row, col, off_sk)

                    def _emit():
                        if const_expr(plan.use_splitk):
                            off_b = fx.Int32(off_sk * 4)
                            z0 = fx.Int32(0)
                            buffer_atomic_add(a[i], self._y_rsrc, off_b, z0, z0)
                        else:
                            cval = (
                                (a[i] + bias_val).to(elem_ty)
                                if const_expr(plan.has_bias)
                                else a[i].to(elem_ty)
                            )
                            if const_expr(plan.big_out):
                                self._big_store(fx.Int64(off_nk), cval)
                            else:
                                fx.memref_store_vec(
                                    fx.Vector.filled(1, cval, elem_ty), self._y_reg_1
                                )
                                fx.copy(
                                    self._y_atom_1,
                                    self._y_reg_1,
                                    fx.slice(
                                        self._y_div,
                                        (None, self._route(off_nk, row, col_loc)),
                                    ),
                                )

                    if const_expr(plan.need_chk and not plan.route_store):
                        if self._valid(row, col_loc):
                            _emit()
                    else:
                        _emit()

    def _load_bias(self, n_offset, n_local, c_col):
        """One bias value per MFMA column block, indexed by global out-channel."""
        plan = self._plan
        bias_vals = []
        for ni in range_constexpr(plan.mi_n):
            col, col_loc = self._cols(ni, n_offset, n_local, c_col)
            col_i = fx.Int32(col)
            if const_expr(plan.n_tail):
                col_i = (col_loc < fx.Int64(plan.kg)).select(col_i, fx.Int32(0))
            fx.copy(
                self._bias_atom, fx.slice(self._bias_div, (None, col_i)), self._bias_reg
            )
            bias_vals.append(fx.Float32(fx.memref_load_vec(self._bias_reg)[0]))
        return bias_vals

    def _cols(self, ni, n_offset, n_local, c_col):
        """Global out-channel for MFMA column block ni, and its index within the group."""
        plan = self._plan
        col_off = fx.Int64(fx.get_scalar(c_col[MFMA_C_VALUES * plan.mi_m * ni]))
        col = n_offset + col_off
        return col, ((n_local + col_off) if const_expr(plan.groups > 1) else col)

    def _off_nk(self, row, col, off_sk):
        """The GEMM's (row, col) as an element offset into y."""
        plan = self._plan
        dhw = plan.geom.dhw
        # NDHWC is already (npq, k) row-major, so the scatter is off_sk.
        if const_expr(plan.out_ndhwc):
            return off_sk
        if const_expr(plan.n == 1):
            return col * dhw + row
        n_idx = row // dhw
        return n_idx * (plan.k * dhw) + col * dhw + (row % dhw)

    def _valid(self, row, col_loc):
        plan = self._plan
        if const_expr(plan.row_chk and plan.n_tail):
            return (row < fx.Int64(plan.geom.npq)) & (col_loc < fx.Int64(plan.kg))
        if const_expr(plan.row_chk):
            return row < fx.Int64(plan.geom.npq)
        return col_loc < fx.Int64(plan.kg)

    def _route(self, off, row, col_loc):
        if const_expr(not self._plan.route_store):
            return fx.Int32(off)
        return self._valid(row, col_loc).select(
            fx.Int32(off), fx.Int32(OOB_SENTINEL_ELEM)
        )

    def _big_store(self, off_nk_i64, value):
        # BIG_OUT means y is past what a buffer descriptor's 32-bit voffset
        # reaches, so there is no buffer-resource form to route this through
        # and the store is addressed by a flat 64-bit address instead. That
        # is also why this path gives up y_div and the store copy atoms.
        addr = self._y_elem_base + off_nk_i64 * fx.Int64(BF16_BYTES)
        fx.ptr_store(value, fx.inttoptr(self._big_st_ptr_ty, addr))
