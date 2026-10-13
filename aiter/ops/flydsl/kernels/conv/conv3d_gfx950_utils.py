# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.
#
# ruff: noqa: B023  # scatter closures over tile loops; per-line waivers break under format

"""Shared pieces of the conv3d kernels (geometry, grid, scatter). Gather is in conv3d_im2col."""

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

# Predicate-as-address: OOB load/store via a voffset no descriptor can hold.
OOB_SENTINEL_ELEM = 0x7FFFFF80
OOB_SENTINEL_BYTES = OOB_SENTINEL_ELEM * BF16_BYTES

# Compile hints applied to both conv3d kernels. Empty by default; a caller that
# needs to pass FlyDSL a hint sets it before the first compile.
CONV_COMPILE_HINTS = {}


def _as_stream(stream):
    return stream if hasattr(stream, "_is_stream_param") else fx.Stream(stream)


def buffer_atomic_add(vdata, rsrc, offset, soffset, aux):
    """Buffer-resource atomic fadd (``raw.ptr.buffer.atomic.fadd``)."""
    return fx.rocdl.raw_ptr_buffer_atomic_fadd(vdata, rsrc, offset, soffset, aux)


def barrier(vmcnt=0, lgkmcnt=None):
    """Wait on the named counters, then barrier. Not ``gpu.barrier()``."""
    fx.rocdl.s_waitcnt(vmcnt=vmcnt, lgkmcnt=lgkmcnt)
    fx.rocdl.s_barrier()


def sgpr(x):
    """Broadcast lane 0's value into a scalar register."""
    return fx.Int64(fx.rocdl.readfirstlane(T.i64, fx.Int64(x)))


def flat_buffer_view(ptr, elems, num_records_bytes):
    """A 1-D buffer view: ``slice(view, (None, off))`` is element ``off``."""
    buf = fx.rocdl.make_buffer_ptr(ptr, num_records_bytes=num_records_bytes)
    return fx.logical_divide(
        fx.make_view(buf, fx.make_layout(elems, 1)), fx.make_layout(1, 1)
    )


def in_range(v, hi):
    return (v >= 0) & (v < fx.Int64(hi))


# Variable resolution: runtime extents via magic-number division (Divisor).
# Booleans (big_in, vec_store, ...) stay in the cache key as a few variants.


def magic_u32(d: int):
    """``(m, s)`` such that ``v // d == (v * m) >> (32 + s)``.

    Holds for ``0 <= v < 2**31``, which is what the caller has to guarantee
    (see ``MAX_DYN_DIVIDEND``). Round-up form: ``m = ceil(2**(32+s) / d)``
    at the smallest ``s`` that keeps ``m`` inside 32 bits.
    """
    if d < 1:
        raise ValueError(f"magic_u32 needs a positive divisor, got {d}")
    for s in range(32):
        if (1 << (32 + s)) >= d * (1 << 31):
            break
    m = ((1 << (32 + s)) + d - 1) // d
    if m >= (1 << 32):
        raise ValueError(f"magic for d={d} does not fit 32 bits")
    return m, s


# The bound magic_u32's derivation assumes of a dividend. Everything a Divisor
# divides here is a GEMM row or a remainder of one, so all of them are under
# npq and this is the same statement as "fewer than 2**31 output elements".
MAX_DYN_DIVIDEND = 1 << 31


class Divisor:
    """Compile-time int or runtime magic reciprocal. Must not enter the cache key."""

    __slots__ = ("_magic", "_shift", "is_static", "value")

    def __init__(self, value, magic=None, shift=None):
        self.value = value
        self._magic = magic
        self._shift = shift
        self.is_static = magic is None

    def divmod(self, v):
        """``(v // self, v % self)``, both Int64."""
        vi = fx.Int64(v)
        if const_expr(self.is_static):
            return vi // fx.Int64(self.value), vi % fx.Int64(self.value)
        q = (vi * fx.Int64(self._magic)).shrui(fx.Int64(32) + fx.Int64(self._shift))
        return q, vi - q * fx.Int64(self.value)

    def div(self, v):
        return self.divmod(v)[0]

    def mod(self, v):
        return self.divmod(v)[1]


def static_divisor(value):
    """The folded form, for a kernel compiled against one resolution."""
    return Divisor(int(value))


# Pack (magic, shift) into one i64 kernarg.
RCP_SHIFT_BITS = 6
RCP_SHIFT_MASK = (1 << RCP_SHIFT_BITS) - 1


def pack_reciprocal(d: int) -> int:
    """``magic_u32(d)`` as the single scalar the kernel unpacks.

    A divisor of 1 has no 32-bit magic form -- it would need ``m == 2**32`` --
    and needs none: dividing by it is the identity. Those are reported through
    ``unit_divisors`` and keep the folded form, so the value here is never read.
    """
    if d == 1:
        return 0
    m, s = magic_u32(d)
    return (m << RCP_SHIFT_BITS) | s


def dyn_divisor(value, rcp):
    """The magic form, from the divisor and its packed reciprocal."""
    rcp_i = fx.Int64(rcp)
    return Divisor(
        value,
        magic=rcp_i.shrui(fx.Int64(RCP_SHIFT_BITS)),
        shift=rcp_i & fx.Int64(RCP_SHIFT_MASK),
    )


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
# The launch config: one (TILE_M, TILE_N, WAVE_M, WAVE_N) and what follows
# from it
# ---------------------------------------------------------------------------

TILE_K = 32

# K tiles between barriers. TILE_K=64 is slower (LDS bank conflicts).
TILES_PER_BARRIER = 2

DEFAULT_TILE = (128, 128, 2, 4)


def validate_launch_config(tile_m, tile_n, wave_m, wave_n):
    """Why this tile cannot compile, or None. Problem-shape asserts stay at use sites."""
    block_threads = wave_m * wave_n * WARP_SIZE
    if block_threads > 1024:
        return f"BLOCK_THREADS={block_threads} exceeds 1024"
    if tile_m % (wave_m * MFMA_M):
        return f"TILE_M={tile_m} not divisible by WAVE_M*{MFMA_M}"
    if tile_n % (wave_n * MFMA_N):
        return f"TILE_N={tile_n} not divisible by WAVE_N*{MFMA_N}"
    # LDG_{A,B}_COUNT >= 1 needs no check of its own: TILE_K is 32 and BLOCK_VECS
    # is 8*BLOCK_THREADS, so both divisibility tests already imply a count of at
    # least one for any positive tile.
    block_vecs = LDG_VEC * block_threads
    if (tile_m * TILE_K) % block_vecs:
        return f"A tile {tile_m}x{TILE_K} not a multiple of {block_vecs} vecs"
    if (tile_n * TILE_K) % block_vecs:
        return f"B tile {tile_n}x{TILE_K} not a multiple of {block_vecs} vecs"
    return None


class TileConfig(NamedTuple):
    """One launch config, with everything the kernel derives from it.

    Derived next to the validation of the same arithmetic so the two cannot
    disagree about what a tile implies -- which is the failure mode the
    docstring above describes, one level down.
    """

    tile_m: int
    tile_n: int
    tile_k: int
    wave_m: int
    wave_n: int
    block_threads: int

    # MFMA atoms per wave. tiled_mma replicates the atom over the
    # (WAVE_M, WAVE_N) wave grid and tiles THAT over (TILE_M, TILE_N), so a
    # wave's atoms are strided by the whole wave grid rather than contiguous.
    # The epilogue takes its row/col from partition_C rather than rederiving it.
    mi_m: int
    mi_n: int

    # Vectors one block loads per K tile, per operand.
    ldg_a_count: int
    ldg_b_count: int

    # LDS stages, and the elements each operand's staging buffer holds.
    pipe_stages: int
    lds_a_elems: int
    lds_b_elems: int


def make_tile_config(tile):
    """The TileConfig for one (TILE_M, TILE_N, WAVE_M, WAVE_N), or an assertion."""
    tile_m, tile_n, wave_m, wave_n = tile
    invalid = validate_launch_config(tile_m, tile_n, wave_m, wave_n)
    assert invalid is None, invalid

    block_threads = wave_m * wave_n * WARP_SIZE
    block_vecs = LDG_VEC * block_threads
    pipe_stages = 2 * TILES_PER_BARRIER
    return TileConfig(
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=TILE_K,
        wave_m=wave_m,
        wave_n=wave_n,
        block_threads=block_threads,
        mi_m=tile_m // wave_m // MFMA_M,
        mi_n=tile_n // wave_n // MFMA_N,
        ldg_a_count=tile_m * TILE_K // block_vecs,
        ldg_b_count=tile_n * TILE_K // block_vecs,
        pipe_stages=pipe_stages,
        lds_a_elems=pipe_stages * tile_m * TILE_K,
        lds_b_elems=pipe_stages * tile_n * TILE_K,
    )


# ---------------------------------------------------------------------------
# B: the weight, which is already the matrix the GEMM wants
# ---------------------------------------------------------------------------


def weight_bytes(param, geom):
    """Bytes of the packed (K, CRS) weight, checked against a descriptor's reach."""
    w_bytes = param.k * geom.crs * BF16_BYTES
    assert (
        w_bytes < OOB_SENTINEL_BYTES
    ), f"weight {w_bytes}B exceeds limit {OOB_SENTINEL_BYTES}B"
    return w_bytes


class WeightLoader:
    """Load packed (K, CRS) weight. Kernel descriptor, then per-block columns."""

    def __init__(self, cfg, grid, crs, weight, w_bytes):
        # ``crs`` rather than the whole geometry: it is the only field this
        # needs, and it follows from C/groups and the filter, so taking it
        # alone keeps the output extents out of the kernel's closure -- which
        # is what lets one artifact serve several resolutions.
        self._cfg, self._grid, self._crs = cfg, grid, crs
        self._src = flat_buffer_view(
            fx.get_iter(weight), w_bytes // BF16_BYTES, w_bytes
        )
        self._tid = self._n_offset = self._n_local = None

    def bind_block(self, tid, n_offset, n_local):
        self._tid, self._n_offset, self._n_local = tid, n_offset, n_local

    def taps(self, k_base):
        """Yield ``(i, src, voff)`` per B vector of the K tile at ``k_base``."""
        assert self._n_offset is not None, "bind_block() before taps()"
        cfg, grid = self._cfg, self._grid
        for i in range_constexpr(cfg.ldg_b_count):
            linear = (self._tid + i * cfg.block_threads) * LDG_VEC
            local_n = linear // cfg.tile_k
            local_k = linear % cfg.tile_k
            col = self._n_offset + fx.Int64(local_n)
            g_off = fx.Int32(col * self._crs + (fx.Int64(k_base) + fx.Int64(local_k)))
            if const_expr(grid.n_tail):
                # The tail is per group: the N grid is over-provisioned to
                # groups * tiles_per_group, so a block past this group's last
                # out-channel reads zero rather than the next group's weights.
                in_group = (self._n_local + fx.Int64(local_n)) < fx.Int64(grid.kg)
                g_off = in_group.select(g_off, fx.Int32(OOB_SENTINEL_ELEM))
            yield i, self._src, g_off


# ---------------------------------------------------------------------------
# LDS staging: the DMA that fills a stage, and the MMA that reads it back
# ---------------------------------------------------------------------------


def make_shared_storage(elem_ty, cfg):
    """The LDS struct one block allocates: ``pipe_stages`` tiles of A and of B."""

    @fx.struct
    class SharedStorage:
        a: fx.Array[elem_ty, cfg.lds_a_elems, 16]
        b: fx.Array[elem_ty, cfg.lds_b_elems, 16]

    return SharedStorage


class LdsStager:
    """Where a block's DMAs land, for both operands.

    One instance per kernel, shared by the gather and the weight loader: what
    differs between them is where the data comes from, not how a stage is
    addressed.
    """

    def __init__(self, cfg, elem_ty, tid):
        self._cfg = cfg
        self._tid = tid
        dma_bytes = LDG_VEC * BF16_BYTES  # 16
        self._lds_ptr_ty = fx.PointerType.get(
            elem_ty.ir_type, fx.AddressSpace.Shared, dma_bytes
        )
        self._atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), dma_bytes * 8)

    def dst(self, lds_array, stage_tile, i):
        """The LDS address vector ``i`` of this stage writes to."""
        # buffer_load_lds takes one wave-uniform LDS base and fans the wave's
        # lanes out from it, so the lane-0 address is the base the whole wave
        # writes from.
        off_elems = fx.Int64(stage_tile) + (
            fx.Int64(self._tid) + fx.Int64(i * self._cfg.block_threads)
        ) * fx.Int64(LDG_VEC)
        base_bytes = off_elems * fx.Int64(BF16_BYTES)
        addr = fx.Int64(fx.ptrtoint(lds_array.ptr)) + fx.Int64(base_bytes)
        return fx.make_view(
            fx.inttoptr(self._lds_ptr_ty, sgpr(addr)), fx.make_layout(1, 1)
        )

    def copy(self, src, dst, voff_elem):
        """Issue one async global-to-LDS DMA."""
        fx.copy(self._atom, fx.slice(src, (None, voff_elem)), dst)


class MmaTiling:
    """The MFMA assembly of one block: who computes what, and out of which LDS.

    Holds the tiled MMA and the two LDS-to-register copies derived from it,
    the accumulator, and the tile-local coordinates of the accumulator's
    elements -- all from one ``tiled_mma``, so the epilogue cannot drift from
    the MMA's own partitioning.
    """

    def __init__(self, cfg, elem_ty, tid, lds, scratch):
        self._cfg = cfg
        self._lds_copy = fx.make_copy_atom(fx.UniversalCopy128b(), elem_ty)
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(MFMA_M, MFMA_N, cfg.tile_k, elem_ty))
        self.tiled_mma = fx.make_tiled_mma(
            mma_atom,
            fx.make_layout((cfg.wave_m, cfg.wave_n, 1), (cfg.wave_n, 1, 0)),
        )
        self._thr_mma = self.tiled_mma.thr_slice(tid)
        self._thr_copy_a = fx.make_tiled_copy_A(
            self._lds_copy, self.tiled_mma
        ).get_slice(tid)
        self._thr_copy_b = fx.make_tiled_copy_B(
            self._lds_copy, self.tiled_mma
        ).get_slice(tid)

        self._a_lds, self._b_lds = lds.a, lds.b
        self._a_layout = fx.make_layout((cfg.tile_m, cfg.tile_k), (cfg.tile_k, 1))
        self._b_layout = fx.make_layout((cfg.tile_n, cfg.tile_k), (cfg.tile_k, 1))

        # `scratch` is a pointer to give the fragment a view to be shaped by;
        # make_fragment_C reads the layout, never the memory, so any live
        # buffer does -- the accumulator lives in registers.
        self.acc = self._thr_mma.make_fragment_C(
            fx.make_view(
                fx.get_iter(scratch),
                fx.make_layout((cfg.tile_m, cfg.tile_n), (cfg.tile_n, 1)),
            )
        )
        self.acc.fill(0.0)

        # partition_C returns coordinates; index acc flat (layout rank assert).
        self.c_row = self._thr_mma.partition_C(
            fx.make_view(0, fx.make_layout((cfg.tile_m, cfg.tile_n), (1, 0)))
        )
        self.c_col = self._thr_mma.partition_C(
            fx.make_view(0, fx.make_layout((cfg.tile_m, cfg.tile_n), (0, 1)))
        )

    # A and B are read into separate fragments and returned separately, never
    # as one tuple: combining them once took the compile wall from ~5s to ~2h.
    def read_a(self, stage):
        sA = fx.make_view(
            fx.add_offset(self._a_lds.ptr, stage * self._cfg.tile_m * self._cfg.tile_k),
            self._a_layout,
        )
        frag_a = self._thr_mma.make_fragment_A(sA)
        fx.copy(
            self._lds_copy,
            self._thr_copy_a.partition_S(sA),
            self._thr_copy_a.retile(frag_a),
        )
        fx.rocdl.sched_dsrd(self._cfg.mi_m)
        return frag_a

    def read_b(self, stage):
        sB = fx.make_view(
            fx.add_offset(self._b_lds.ptr, stage * self._cfg.tile_n * self._cfg.tile_k),
            self._b_layout,
        )
        frag_b = self._thr_mma.make_fragment_B(sB)
        fx.copy(
            self._lds_copy,
            self._thr_copy_b.partition_S(sB),
            self._thr_copy_b.retile(frag_b),
        )
        fx.rocdl.sched_dsrd(self._cfg.mi_n)
        return frag_b

    def compute(self, acc_values, a_frag_values, b_frag_values):
        """One K tile's MFMAs, at raised priority so they are not interleaved."""
        fx.rocdl.s_setprio(1)
        fx.gemm(
            self.tiled_mma,
            acc_values,
            a_frag_values,
            b_frag_values,
            acc_values,
        )
        fx.rocdl.sched_mfma(self._cfg.mi_m * self._cfg.mi_n)
        fx.rocdl.s_setprio(0)
        return acc_values


# ---------------------------------------------------------------------------
# The implicit GEMM's shape
# ---------------------------------------------------------------------------


class ConvGeometry(NamedTuple):
    """Implicit GEMM shape (npq, crs). Derived once for gather, grid, and epilogue."""

    do: int
    ho: int
    wo: int
    dhw: int
    hw_o: int
    npq: int
    cgp: int
    crs: int


def out_extent(size, pad, dil, kernel, stride):
    """One output axis, torch's rule. The single copy used by host, kernel, and tuner."""
    return (size + 2 * pad - (dil * (kernel - 1) + 1)) // stride + 1


def make_conv_geometry(param):
    """The ConvGeometry of one ``Conv3dImplicitParam``.

    Dilation only stretches the filter's footprint, so it moves the output
    extents but leaves the K axis (CRS) alone.
    """
    cgp = param.c // param.groups
    do = out_extent(param.d, param.pt, param.dt, param.kt, param.st)
    ho = out_extent(param.h, param.ph, param.dh, param.kh, param.sh)
    wo = out_extent(param.w, param.pw, param.dw, param.kw, param.sw)
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


class ConvExtents:
    """Extents the gather/epilogue address against (ints or runtime scalars)."""

    __slots__ = (
        "d",
        "dhw",
        "div_d",
        "div_dhw",
        "div_hw_o",
        "div_wo",
        "do",
        "h",
        "ho",
        "hw_o",
        "is_static",
        "npq",
        "w",
        "wo",
        "x_elems",
        "x_sample_elems",
    )

    def __init__(
        self,
        *,
        d,
        h,
        w,
        do,
        ho,
        wo,
        dhw,
        hw_o,
        npq,
        x_elems,
        x_sample_elems,
        div_d,
        div_dhw,
        div_hw_o,
        div_wo,
        is_static,
    ):
        self.d, self.h, self.w = d, h, w
        self.do, self.ho, self.wo = do, ho, wo
        self.dhw, self.hw_o, self.npq = dhw, hw_o, npq
        self.x_elems = x_elems
        self.x_sample_elems = x_sample_elems
        self.div_d = div_d
        self.div_dhw = div_dhw
        self.div_hw_o = div_hw_o
        self.div_wo = div_wo
        self.is_static = is_static


class StaticInputExtents(NamedTuple):
    """Input extents folded into a static kernel. Must stay a NamedTuple (cache key)."""

    d: int
    h: int
    w: int
    x_elems: int
    x_sample_elems: int


def static_input_extents(param):
    """The ``StaticInputExtents`` of one problem."""
    x_sample_elems = param.c * param.d * param.h * param.w
    return StaticInputExtents(
        d=param.d,
        h=param.h,
        w=param.w,
        x_elems=param.n * x_sample_elems,
        x_sample_elems=x_sample_elems,
    )


def static_extents(in_ext, geom):
    """The compile-time form: every extent and every divisor is a literal.

    Takes ``StaticInputExtents`` rather than the param it comes from, so that
    what the kernel closes over is a tuple FlyDSL keys on. See that class.
    """
    return ConvExtents(
        d=in_ext.d,
        h=in_ext.h,
        w=in_ext.w,
        do=geom.do,
        ho=geom.ho,
        wo=geom.wo,
        dhw=geom.dhw,
        hw_o=geom.hw_o,
        npq=geom.npq,
        x_elems=in_ext.x_elems,
        x_sample_elems=in_ext.x_sample_elems,
        # Only the temporal_only_fast path divides by d, but a folded divisor
        # costs nothing to build for the paths that do not.
        div_d=static_divisor(in_ext.d),
        div_dhw=static_divisor(geom.dhw),
        div_hw_o=static_divisor(geom.hw_o),
        div_wo=static_divisor(geom.wo),
        is_static=True,
    )


class DynShapeArgs:
    """The runtime scalars a variable-resolution kernel takes on top of its
    four tensors.

    Field order is the kernel's parameter order and also the order
    ``dyn_shape_values`` produces on the host; both read it off ``FIELDS`` so
    adding an extent cannot update one side only.
    """

    FIELDS = (
        "d",
        "h",
        "w",
        "wo",
        "hw_o",
        "dhw",
        "npq",
        "rcp_d",
        "rcp_wo",
        "rcp_hw_o",
        "rcp_dhw",
        "grid_m",
        "x_elems",
        "x_sample_elems",
    )

    __slots__ = FIELDS

    def __init__(self, *values):
        if len(values) != len(self.FIELDS):
            raise ValueError(
                f"DynShapeArgs takes {len(self.FIELDS)} values, got {len(values)}"
            )
        for name, v in zip(self.FIELDS, values):
            setattr(self, name, v)


def dyn_shape_values(param, geom, grid):
    """The values behind ``DynShapeArgs``, in ``FIELDS`` order.

    ``param`` has to be the one carrying the *real* d/h/w: what the kernel
    closure holds under variable resolution is the zeroed stand-in, which
    would derive the wrong geometry here.
    """
    if geom.npq >= MAX_DYN_DIVIDEND:
        raise ValueError(
            f"npq={geom.npq} reaches the {MAX_DYN_DIVIDEND} bound the magic-number "
            "division assumes; this shape has to stay on the static path"
        )
    x_sample_elems = param.c * param.d * param.h * param.w
    return (
        param.d,
        param.h,
        param.w,
        geom.wo,
        geom.hw_o,
        geom.dhw,
        geom.npq,
        pack_reciprocal(param.d),
        pack_reciprocal(geom.wo),
        pack_reciprocal(geom.hw_o),
        pack_reciprocal(geom.dhw),
        grid.grid_m,
        param.n * x_sample_elems,
        x_sample_elems,
    )


def unit_divisors(param, geom):
    """Which of (d, wo, hw_o, dhw) are 1, and so keep the folded divisor.

    These follow the resolution, not only the layer: one layer run at D=1 and
    at D>1 compiles to two artifacts, which is why they are part of
    ``_dyn_hw_closure_key``.
    """
    return (param.d == 1, geom.wo == 1, geom.hw_o == 1, geom.dhw == 1)


def dyn_extents(s, unit=(False, False, False, False)):
    """Runtime ConvExtents from DynShapeArgs + unit_divisors flags."""
    unit_d, unit_wo, unit_hw_o, unit_dhw = unit

    def _div(is_unit, value, rcp):
        return static_divisor(1) if is_unit else dyn_divisor(value, rcp)

    return ConvExtents(
        d=s.d,
        h=s.h,
        w=s.w,
        # do and ho only shape the grid, which the host already sized, so the
        # kernel never reads them back.
        do=None,
        ho=None,
        wo=s.wo,
        dhw=s.dhw,
        hw_o=s.hw_o,
        npq=s.npq,
        x_elems=s.x_elems,
        x_sample_elems=s.x_sample_elems,
        div_d=_div(unit_d, s.d, s.rcp_d),
        div_dhw=_div(unit_dhw, s.dhw, s.rcp_dhw),
        div_hw_o=_div(unit_hw_o, s.hw_o, s.rcp_hw_o),
        div_wo=_div(unit_wo, s.wo, s.rcp_wo),
        is_static=False,
    )


# Grid: M on x (chunk into z if needed), N on y (per group), split-K on z, WGM swizzle.


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


def make_launch_grid(param, geom, cfg):
    """The LaunchGrid for one problem and launch config, or an assertion."""
    tile_m, tile_n, tile_k = cfg.tile_m, cfg.tile_n, cfg.tile_k
    block_threads = cfg.block_threads
    k, groups = param.k, param.groups
    kg = k // groups
    npq = geom.npq

    tiles_per_group = (kg + tile_n - 1) // tile_n
    n_tail = kg % tile_n != 0
    grid_y = groups * tiles_per_group

    k_tiles = (geom.crs + tile_k - 1) // tile_k
    splitk = max(1, min(param.splitk, k_tiles))
    # splitK must divide k_tiles or the K tail is silently dropped.
    assert k_tiles % splitk == 0, (
        f"splitk={splitk} does not divide k_tiles={k_tiles}: splits would cover only "
        f"{splitk * (k_tiles // splitk)} of them and the rest of the K axis would be "
        f"dropped. Pick it through _resolve_splitk."
    )
    tiles_per_split = k_tiles // splitk

    grid_m = (npq + tile_m - 1) // tile_m
    max_grid_x = 0xFFFFFFFF // block_threads
    grid_x = min(grid_m, max_grid_x)
    m_chunks = (grid_m + grid_x - 1) // grid_x

    assert (
        grid_y <= MAX_GRID_YZ
    ), f"grid.y = {grid_y} exceeds the {MAX_GRID_YZ}-block limit"
    assert (
        m_chunks * splitk <= MAX_GRID_YZ
    ), f"grid.z = {m_chunks} M-chunks x {splitk} splits exceeds the {MAX_GRID_YZ}-block limit"

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


def block_coords(grid, grid_m=None):
    """Decode this block's tile. Pass runtime ``grid_m`` on the dyn_hw path."""
    if const_expr(grid_m is None):
        grid_m = grid.grid_m
    if const_expr(grid.m_chunks > 1):
        m_chunk = fx.Int64(gpu.block_id("z")) % fx.Int64(grid.m_chunks)
        m_offset = (
            fx.Int64(gpu.block_id("x")) + m_chunk * fx.Int64(grid.grid_x)
        ) * grid.tile_m
        n_tile = fx.Int32(gpu.block_id("y"))
    elif const_expr(grid.wgm > 1):
        pid = fx.Int64(gpu.block_id("x")) + fx.Int64(gpu.block_id("y")) * fx.Int64(
            grid_m
        )
        blocks_per_swizzle = fx.Int64(grid.wgm * grid.grid_y)
        swizzle_id = pid // blocks_per_swizzle
        first_m = swizzle_id * fx.Int64(grid.wgm)
        # The last swizzle group is short when grid_m is not a multiple of WGM.
        swizzle_rows = fx.min(fx.Int64(grid_m) - first_m, fx.Int64(grid.wgm))
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


# Scatter: (npq, K) tile back to NCDHW / NDHWC, plus split-K atomics.


# Split-K staging must fit a 32-bit buffer num_records (unsigned voffset).
SPLITK_MAX_STAGING_BYTES = 0xFFFFFFFF

# Heuristic cap: auto splitK stays under 2**31; explicit/CSV splitK may use the rest.
SPLITK_AUTO_MAX_STAGING_BYTES = 0x7FFFFFFF


class OutputScatterPlan(NamedTuple):
    """Epilogue compile constants. NamedTuple so FlyDSL keys on the fields."""

    # N is not a field: NCDHW always splits a row into (sample, position).
    k: int
    kg: int
    groups: int
    out_ndhwc: bool
    has_bias: bool

    # The output extents C is scattered against are absent for the same reason
    # they are absent from ``Im2colPlan``: they are what a variable-resolution
    # kernel reads at runtime, and they arrive as ``ConvExtents``.

    # MFMA atoms per wave along M and N: the epilogue walks them.
    mi_m: int
    mi_n: int

    # How the store is done, decided once below. Booleans, so they cost a
    # constant number of artifacts rather than one per resolution.
    use_splitk: bool
    big_out: bool
    row_chk: bool
    n_tail: bool
    need_chk: bool
    route_store: bool
    vec_store: bool


def make_output_scatter_plan(param, geom, cfg, grid):
    """An OutputScatterPlan for one problem and launch config, or an assertion.

    Takes the grid because whether M and N are over-provisioned -- and so what
    the tail must not write -- is the grid's arithmetic, not the problem's;
    everything else about how C is written follows from the problem.
    """
    kg, use_splitk = grid.kg, grid.use_splitk
    row_chk, n_tail = grid.row_chk, grid.n_tail
    mi_m, mi_n = cfg.mi_m, cfg.mi_n
    n, k, out_ndhwc = param.n, param.k, param.out_ndhwc
    npq, dhw = geom.npq, geom.dhw

    big_out = (n * k * geom.do * geom.ho * geom.wo * BF16_BYTES) > 0x7FFFFFFF

    assert (
        not use_splitk or npq * k * 4 <= SPLITK_MAX_STAGING_BYTES
    ), f"split-K staging {npq * k * 4}B exceeds the {SPLITK_MAX_STAGING_BYTES}B buffer window"

    need_chk = row_chk or n_tail
    # A tail is masked by routing the store to the OOB sentinel, which needs a
    # buffer descriptor to land in. Split-K accumulates through atomics and
    # BIG_OUT addresses y flat, so neither has one, and ``OutputScatter.store``
    # would have to branch on a runtime predicate it cannot evaluate at trace
    # time. ``_resolve_splitk`` never auto-splits a shape with a tail (it wants
    # npq % tile_m == 0 and kg % tile_n == 0 first), so this is only reachable
    # through an explicit ``splitk=`` or a hand-written tuned row -- named here
    # rather than left to fail inside the trace with no shape to point at.
    assert not need_chk or not (use_splitk or big_out), (
        f"{'split-K' if use_splitk else 'BIG_OUT'} cannot mask a tile tail, and this "
        f"launch has one (row_chk={row_chk}, n_tail={n_tail}): npq={npq} against "
        f"tile_m={cfg.tile_m}, kg={kg} against tile_n={cfg.tile_n}. Use a tile that "
        "divides the problem, or drop the split."
    )
    return OutputScatterPlan(
        k=k,
        kg=kg,
        groups=param.groups,
        out_ndhwc=out_ndhwc,
        has_bias=param.has_bias,
        mi_m=mi_m,
        mi_n=mi_n,
        use_splitk=use_splitk,
        big_out=big_out,
        row_chk=row_chk,
        n_tail=n_tail,
        need_chk=need_chk,
        # Routing a masked element to the OOB sentinel drops it without a
        # branch, but only where the store goes through a buffer descriptor:
        # split-K atomics and the 64-bit BIG_OUT path have no such address.
        route_store=need_chk and not use_splitk and not big_out,
        vec_store=(
            (not use_splitk)
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

    def __init__(self, plan, y, bias, elem_ty, ext):
        self._plan = plan
        self._ext = ext
        self._y = y
        self._elem_ty = elem_ty

        # fp32 while split-K stages through it, bf16 once it is the output.
        y_elems = ext.npq * plan.k
        y_bytes = y_elems * (4 if plan.use_splitk else BF16_BYTES)
        y_buf = fx.rocdl.make_buffer_tensor(y, num_records_bytes=y_bytes)
        if const_expr(plan.use_splitk):
            # buffer_atomic_add needs the raw !llvm.ptr<8> descriptor, not a tensor.
            self._y_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(y_buf))
        else:
            self._y_div = fx.logical_divide(
                fx.Tensor(
                    fx.make_view(
                        fx.get_iter(y_buf),
                        fx.make_layout(y_elems, 1),
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
                    off_nk0 = self._off_nk(row0, col, None)

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
        plan, ext = self._plan, self._ext
        # NDHWC is already (npq, k) row-major, so the scatter is off_sk.
        if const_expr(plan.out_ndhwc):
            return off_sk
        dhw = fx.Int64(ext.dhw)
        n_idx, row_in_sample = ext.div_dhw.divmod(row)
        return n_idx * (fx.Int64(plan.k) * dhw) + col * dhw + row_in_sample

    def _valid(self, row, col_loc):
        plan = self._plan
        if const_expr(plan.row_chk and plan.n_tail):
            return (row < fx.Int64(self._ext.npq)) & (col_loc < fx.Int64(plan.kg))
        if const_expr(plan.row_chk):
            return row < fx.Int64(self._ext.npq)
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
