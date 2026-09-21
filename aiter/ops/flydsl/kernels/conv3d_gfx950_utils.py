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


# ---------------------------------------------------------------------------
# Variable resolution: the extents that stop being compile-time constants
#
# Folding im2col's div/mod against the output extents is where much of this
# kernel's performance comes from, and folding needs a literal divisor -- so
# one artifact belongs to one D/H/W. Where the resolution is arbitrary that
# is a JIT per new size, which is the cost this exists to remove.
#
# Magic-number division is the way out: the divisor arrives as a runtime
# value carrying its own reciprocal, and the decomposition stays a multiply
# and a shift -- the two constants move from immediates into kernargs, which
# is the whole of the difference. ``Divisor`` presents the folded and the
# magic form behind one interface so the gather and the epilogue keep one
# body between them.
#
# Only the *integers* need this. ``big_in``, ``vec_store`` and the rest of
# the addressing decisions are booleans, and a boolean left in the cache key
# costs a constant number of variants rather than one per resolution.
# ---------------------------------------------------------------------------


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
    """A divisor that is either a compile-time constant or a runtime value.

    The static form holds a Python int and lets the compiler fold ``//`` and
    ``%`` exactly as it did before this layer existed. The dynamic form
    carries a magic reciprocal and a shift, which turns the same
    decomposition into a multiply and a shift. Both answer ``divmod``, so no
    call site has to branch on which one it got.

    Not a NamedTuple, and it must never reach a cache key: the dynamic form
    holds fx runtime values whose repr differs per trace, so a key containing
    one would never hit. What the compiler needs to know about the choice is
    carried by the ``dyn_hw`` boolean instead.
    """

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


# A divisor's reciprocal travels as one scalar rather than two: the shift is
# under 32 and the magic under 2**32, so ``magic << RCP_SHIFT_BITS | shift``
# fits an i64 and halves the number of kernargs the extents need.
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

# K tiles consumed between two barriers. Each one is MI_M * MI_N MFMAs, and that
# product is the only thing that hides global latency here -- the pipeline depth
# cannot. Costs no LDS (the tiles are stages that already exist) and no extra
# ds_read/DMA traffic; it just halves the number of barriers. Worth +8..16% on
# the 3x3 conv2d/conv3d shapes at 2. Reaching the same ratio through TILE_K = 64
# instead is a trap: it makes the LDS row stride 128B, exactly one bank rotation,
# and the resulting ds_read_b128 conflicts cost more than the batching wins
# (measured ~15% slower).
TILES_PER_BARRIER = 2

DEFAULT_TILE = (128, 128, 2, 4)


def validate_launch_config(tile_m, tile_n, wave_m, wave_n):
    """Why this (TILE_M, TILE_N, WAVE_M, WAVE_N) cannot compile, or None.

    The launch-config half of ``compile_conv3d_implicit``'s asserts, in a
    function that costs nothing to call, so a candidate sweep can filter on it
    instead of paying a compile per rejected config. ``conv3d_policy`` used to
    carry its own closed form of the same arithmetic -- the two agreed over all
    8281 combinations of its enumeration, but nothing made them, and a policy
    that drifts stricter prunes configs that would have compiled, which shows
    up as neither an error nor a wrong answer, only as a tuned pick that could
    have been faster.

    Only the tile-shape constraints live here. c/groups and the channel padding
    are properties of the problem, not of the launch config, so they stay as
    asserts at their point of use.
    """
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
    """B's counterpart to ``Im2colGather``, and much the smaller of the two.

    ``_prep_weight`` already packed the filter as a (K, CRS) row-major matrix,
    so a tap is one multiply-add rather than a coordinate decomposition -- the
    asymmetry between this and the gather is the whole difference between an
    implicit GEMM and a real one.

    Two stages for the same reason the gather has them: the descriptor is the
    kernel's, the columns are the block's.
    """

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

        # Each view's layout IS the coordinate, so partition_C hands back
        # coordinates rather than data.
        #
        # They have to be indexed flat: acc is ((MFMA_C_VALUES, 1), MI_M, MI_N),
        # and the hierarchical spellings trip a rank assertion in the layout
        # algebra. Flat index is v + MFMA_C_VALUES * (mi + MI_M * ni); a lane
        # holds one column and MFMA_C_VALUES consecutive rows per atom, so v = 0
        # of atom (mi, ni) is all the epilogue needs.
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


def out_extent(size, pad, dil, kernel, stride):
    """One output axis of a convolution, by torch's rule.

    The one copy of this. It decides ``npq``, so the host sizes the output
    tensor by it, the kernel sizes its grid by it, and the tuner sizes the GEMM
    it reports by it -- three places that must agree on what a stride or a
    dilation does to an extent, or the epilogue writes a shape the caller did
    not allocate.
    """
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
    """The extents the gather and the epilogue actually address against.

    What separates this from ``ConvGeometry`` is when the values are known:
    a ``ConvGeometry`` is compile-time and reaches the cache key, while this
    is what the kernel body reads and may hold either Python ints or fx
    runtime scalars. The field names and the division interface are the same
    either way, which is what keeps one body across both.

    ``static_extents`` copies a ``ConvGeometry`` straight across with folded
    divisors, so a kernel that does not opt into variable resolution compiles
    to what it did before this layer existed.

    Never reaches a cache key, for the reason ``Divisor`` gives.
    """

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


def static_extents(param, geom):
    """The compile-time form: every extent and every divisor is a literal."""
    return ConvExtents(
        d=param.d,
        h=param.h,
        w=param.w,
        do=geom.do,
        ho=geom.ho,
        wo=geom.wo,
        dhw=geom.dhw,
        hw_o=geom.hw_o,
        npq=geom.npq,
        x_elems=param.n * param.c * param.d * param.h * param.w,
        x_sample_elems=param.c * param.d * param.h * param.w,
        # Only the temporal_only_fast path divides by d, but a folded divisor
        # costs nothing to build for the paths that do not.
        div_d=static_divisor(param.d),
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
    """Which of the four dynamic divisors are 1, as a compile-time tuple.

    A unit divisor keeps the folded form (see ``pack_reciprocal``), and which
    ones are unit follows from the layer rather than the resolution -- a 2D
    conv has ``d == 1`` at every size -- so putting these four booleans in the
    cache key costs a constant number of artifacts.

    Order matches ``dyn_extents``: (d, wo, hw_o, dhw).
    """
    return (param.d == 1, geom.wo == 1, geom.hw_o == 1, geom.dhw == 1)


def dyn_extents(s, unit=(False, False, False, False)):
    """The runtime form, assembled from the scalars the kernel was handed.

    ``s`` is a ``DynShapeArgs``: the host already derived every extent and
    its magic reciprocal, so all this does is pair them into ``Divisor``s.
    ``unit`` is ``unit_divisors``' answer, which decides per divisor between
    the magic form and the identity the folded form collapses to.
    """
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
    # Every split has to take a whole number of K tiles. Where it does not,
    # splitk * tiles_per_split < k_tiles and the tail tiles belong to no block at
    # all: the K they carry is neither an error nor an out-of-bounds access, just
    # missing from the sum. ``_resolve_splitk`` walks sk down until it divides, but
    # the paths that bypass it -- AOT reads the CSV's splitK column straight -- have
    # no such step, which is what makes this the last line of defence.
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
    """Decode this block's ids into the tile it owns.

    ``n_local`` is the column within the group and ``n_offset`` the global
    one; they differ only for a grouped conv, where the N grid is per group.
    ``ch_base`` is the group's first input channel, which only the gather
    needs, and is None when there is one group.

    ``grid_m`` overrides the tile count along M, which the WGM swizzle needs
    as a number. It is the one part of the grid that follows from the
    resolution, so a variable-resolution kernel passes the runtime value and
    everything else keeps reading the compile-time grid.
    """
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
#
# The whole 2**32 is reachable, but only because every part of that address is
# unsigned: ``OutputScatter.store`` computes the element offset in i64 and hands
# the atomic an ``fx.Int32(off_sk * 4)``, which is a negative i32 past 2**31 --
# the access still lands on the right byte, since the hardware reads voffset and
# num_records as unsigned. This is the hard limit, and what
# ``make_output_scatter_plan`` asserts against.
SPLITK_MAX_STAGING_BYTES = 0xFFFFFFFF

# What ``_resolve_splitk`` will put a split on *by itself*. Half the window
# above, and deliberately so: everything past 2**31 depends on the unsigned
# reinterpretation described above, so a split nobody asked for does not go
# there. Only an explicit ``splitk=`` from the caller, or a tuned CSV row, can
# reach the rest of the window -- and either way it is a value someone measured.
#
# Written as its own constant rather than a bare literal next to the one above
# because the two are not a copy that drifted: they are the hardware's limit and
# the heuristic's, and a reader who assumes otherwise will "fix" one of them.
SPLITK_AUTO_MAX_STAGING_BYTES = 0x7FFFFFFF


class OutputScatterPlan(NamedTuple):
    """What the epilogue knows before the kernel runs.

    A NamedTuple for the same reason ``Im2colPlan`` is one: FlyDSL keys a
    compiled kernel on the scalar values its closure captures, and anything
    that is not a scalar or a tuple silently drops out of that key.
    """

    # No field here carries the batch size. A row is split back into
    # (sample, position) whenever the output is NCDHW, single sample or not:
    # the divmod costs nothing measurable (a Wan 480x832 encode moved 0.0%),
    # and making it unconditional is what keeps N out of the cache key
    # entirely rather than costing two variants.
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
        # The four values of an MFMA atom are consecutive rows, so they are
        # only contiguous in memory where a row's neighbour is the next
        # spatial position: NCDHW, one store, no staging.
        #
        # A batch is fine, which is worth saying because it looks like it
        # should not be. The four rows start at a multiple of 4 and dhw is
        # required to be one just below, so they cannot straddle a sample,
        # and their offsets stay contiguous whatever n is.
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
