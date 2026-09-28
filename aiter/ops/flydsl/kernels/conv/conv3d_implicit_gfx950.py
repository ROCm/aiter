# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Double-buffered implicit-GEMM conv3d (BF16).

Dispatch lives in ``conv_kernels.py``; pre-transpose in ``conv3d_transpose.py``.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr

from .conv3d_gfx950_utils import (
    CONV_COMPILE_HINTS,
    DEFAULT_TILE,
    LDG_VEC,
    TILES_PER_BARRIER,
    DynShapeArgs,
    LdsStager,
    MmaTiling,
    OutputScatter,
    WeightLoader,
    _as_stream,
    barrier,
    block_coords,
    dyn_extents,
    dyn_shape_values,
    make_conv_geometry,
    make_launch_grid,
    make_output_scatter_plan,
    make_shared_storage,
    make_tile_config,
    static_extents,
    static_input_extents,
    unit_divisors,
    weight_bytes,
)
from .conv3d_im2col import Im2colGather, make_im2col_plan


@fx.struct
class Conv3dImplicitParam:
    """Problem + launch config. Fields are compile-time; ``dyn_hw`` is the exception."""

    n: fx.Constexpr[int]
    c: fx.Constexpr[int]
    d: fx.Constexpr[int]
    h: fx.Constexpr[int]
    w: fx.Constexpr[int]
    k: fx.Constexpr[int]
    kt: fx.Constexpr[int]
    kh: fx.Constexpr[int]
    kw: fx.Constexpr[int]
    st: fx.Constexpr[int]
    sh: fx.Constexpr[int]
    sw: fx.Constexpr[int]
    pt: fx.Constexpr[int]
    ph: fx.Constexpr[int]
    pw: fx.Constexpr[int]
    dt: fx.Constexpr[int]
    dh: fx.Constexpr[int]
    dw: fx.Constexpr[int]
    pad_mode: fx.Constexpr[str]
    has_bias: fx.Constexpr[bool]
    splitk: fx.Constexpr[int]
    tile: fx.Constexpr[tuple]
    wgm: fx.Constexpr[int]
    groups: fx.Constexpr[int]
    out_ndhwc: fx.Constexpr[bool]
    # Runtime D/H/W; host still uses d/h/w to build geometry. Closure must not capture extents.
    dyn_hw: fx.Constexpr[bool]


def make_conv3d_implicit_param(
    n,
    c,
    d,
    h,
    w,
    k,
    kt,
    kh,
    kw,
    st,
    sh,
    sw,
    pt,
    ph,
    pw,
    dt=1,
    dh=1,
    dw=1,
    pad_mode="zeros",
    has_bias=False,
    splitk=1,
    tile=DEFAULT_TILE,
    wgm=1,
    groups=1,
    out_ndhwc=False,
    dyn_hw=False,
):
    """Defaults for ``Conv3dImplicitParam`` (fx.struct has none)."""
    return Conv3dImplicitParam(
        n=n,
        c=c,
        d=d,
        h=h,
        w=w,
        k=k,
        kt=kt,
        kh=kh,
        kw=kw,
        st=st,
        sh=sh,
        sw=sw,
        pt=pt,
        ph=ph,
        pw=pw,
        dt=dt,
        dh=dh,
        dw=dw,
        pad_mode=pad_mode,
        has_bias=has_bias,
        splitk=splitk,
        tile=tuple(tile),
        wgm=wgm,
        groups=groups,
        out_ndhwc=out_ndhwc,
        dyn_hw=dyn_hw,
    )


def _dyn_hw_closure_key(grid, im2col_plan, scatter_plan, unit):
    """The compile-time constants a dyn_hw kernel closes over, booleans included.

    ``grid`` as the kernel holds it (x/z/m blanked). Every boolean here, and
    every ``unit`` flag, may follow the resolution and selects a different
    artifact, so AOT dedupe must key on this, not ``_shape_agnostic_key``.
    """
    return (*grid, *im2col_plan, *scatter_plan, *unit)


def _shape_agnostic_key(grid, im2col_plan, scatter_plan, unit):
    """``_dyn_hw_closure_key`` with booleans blanked. Integers must not carry D/H/W."""

    def _blank(value):
        if isinstance(value, bool):
            return None
        if isinstance(value, tuple):  # nested NamedTuple (TileConfig)
            return tuple(_blank(v) for v in value)
        return value

    return tuple(
        _blank(v) for v in _dyn_hw_closure_key(grid, im2col_plan, scatter_plan, unit)
    )


def _assert_shape_agnostic(param, cfg, kernel_grid, im2col_plan, scatter_plan, unit):
    """Check dyn_hw plans match a probe one output step larger. Unexpressible probes skip."""
    st, sh, sw = param.st, param.sh, param.sw
    probe = make_conv3d_implicit_param(
        param.n,
        param.c,
        param.d + st,
        param.h + sh,
        param.w + sw,
        param.k,
        param.kt,
        param.kh,
        param.kw,
        st,
        sh,
        sw,
        param.pt,
        param.ph,
        param.pw,
        param.dt,
        param.dh,
        param.dw,
        param.pad_mode,
        param.has_bias,
        param.splitk,
        param.tile,
        param.wgm,
        param.groups,
        param.out_ndhwc,
        param.dyn_hw,
    )
    try:
        probe_geom = make_conv_geometry(probe)
        probe_grid = make_launch_grid(probe, probe_geom, cfg)
        if probe_grid.m_chunks != 1:
            # The probe needs M chunking, which this path rules out anyway.
            return
        probe_plans = (
            make_im2col_plan(probe, probe_geom, cfg),
            make_output_scatter_plan(probe, probe_geom, cfg, probe_grid),
            unit_divisors(probe, probe_geom),
        )
    except AssertionError:
        return

    # Against the grid as the kernel closes over it: x/z/m are blanked there
    # because the launch reads them from the runtime scalars instead, so they
    # are the one part of the grid that is allowed to follow the resolution.
    mine = _shape_agnostic_key(kernel_grid, im2col_plan, scatter_plan, unit)
    theirs = _shape_agnostic_key(
        probe_grid._replace(grid_x=0, grid_z=0, grid_m=0), *probe_plans
    )
    assert mine == theirs, (
        "variable resolution asked for, but a compile-time constant still "
        f"carries the input extents: {param.d}x{param.h}x{param.w} and "
        f"{probe.d}x{probe.h}x{probe.w} derive different kernel constants "
        f"({[(a, b) for a, b in zip(mine, theirs) if a != b]}). One artifact "
        "per layer is no longer what this compiles to."
    )


# Tuning walks ~100 configs per shape in one process; 256 would evict.
@functools.lru_cache(maxsize=1024)
def compile_conv3d_implicit(param: Conv3dImplicitParam):
    c, k, groups = param.c, param.k, param.groups

    cfg = make_tile_config(param.tile)
    BLOCK_THREADS = cfg.block_threads

    geom = make_conv_geometry(param)
    CGP = geom.cgp

    # The launch config was validated by make_tile_config; these are the
    # problem's own constraints, which no tile can satisfy on its behalf.
    assert c % groups == 0, f"c={c} not divisible by groups={groups}"
    assert k % groups == 0, f"k={k} not divisible by groups={groups}"
    assert (
        CGP % LDG_VEC == 0
    ), f"c/groups={CGP} must be a multiple of LDG_VEC={LDG_VEC}; use _conv3d_impl to pad"

    W_BYTES = weight_bytes(param, geom)

    # How A is read: everything about the gather, including whether the input
    # fits what a buffer descriptor reaches and how it is rebased if not.
    im2col_plan = make_im2col_plan(param, geom, cfg)

    # How the work is spread: the grid, its limits, and what a block decodes
    # to find its own (M, N, K) tile.
    grid = make_launch_grid(param, geom, cfg)

    # How C is written back: the 5D scatter, the tail masking and the split-K
    # staging, against the same grid the gather reads A on.
    scatter_plan = make_output_scatter_plan(param, geom, cfg, grid)

    # CRS is the K axis of the implicit GEMM: C/groups times the filter, so it
    # is fixed for a layer whatever its resolution. Taken out of the geometry
    # here because it is all the weight loader needs from it.
    CRS = geom.crs

    if const_expr(param.dyn_hw):
        assert grid.m_chunks == 1, (
            f"variable resolution needs a flat M grid, got {grid.m_chunks} chunks "
            f"for npq={geom.npq} at tile_m={cfg.tile_m}"
        )
        extra_args = dyn_shape_values(param, geom, grid)
        dyn_unit = unit_divisors(param, geom)
        kernel_grid = grid._replace(grid_x=0, grid_z=0, grid_m=0)
        _assert_shape_agnostic(
            param, cfg, kernel_grid, im2col_plan, scatter_plan, dyn_unit
        )
    else:
        extra_args = ()
        dyn_unit = None
        kernel_grid = grid
        # NamedTuple, not param.d/h/w: fx.struct is not in the cache key.
        static_in_ext = static_input_extents(param)

    elem_ty = fx.BFloat16
    SharedStorage = make_shared_storage(elem_ty, cfg)

    def _body(y, x, weight, bias, ext, grid_m):
        # A's whole convolution: which input element each A vector taps, and
        # through which descriptor. Everything downstream of the gather treats
        # A as an ordinary GEMM operand.
        im2col = Im2colGather(im2col_plan, x, ext)

        # B needs no gather at all: the weight is already a (K, CRS) matrix.
        weights = WeightLoader(cfg, kernel_grid, CRS, weight, W_BYTES)

        # And how C goes back: the epilogue's descriptors and copy atoms, built
        # here with the others; the store itself happens at the end.
        scatter = OutputScatter(scatter_plan, y, bias, elem_ty, ext)

        lds = fx.SharedAllocator(static=False).allocate(SharedStorage).peek()

        tid = fx.Int32(gpu.thread_id("x"))
        # Which (M, N, K) tile this block owns: WGM swizzle or M chunking,
        # then grouped N, then split-K.
        blk = block_coords(kernel_grid, grid_m)

        mma = MmaTiling(cfg, elem_ty, tid, lds, scratch=y)
        acc = mma.acc

        im2col.bind_block(tid, blk.m_offset, blk.ch_base)
        weights.bind_block(tid, blk.n_offset, blk.n_local)

        stager = LdsStager(cfg, elem_ty, tid)

        def async_load_a_to_lds(k_tile, stage):
            stage_tile = fx.Int64(stage) * cfg.tile_m * cfg.tile_k
            for i, src, voff in im2col.taps(blk.k_off + k_tile * cfg.tile_k):
                stager.copy(src, stager.dst(lds.a, stage_tile, i), voff)

        def async_load_b_to_lds(k_tile, stage):
            stage_tile = fx.Int64(stage) * cfg.tile_n * cfg.tile_k
            for i, src, voff in weights.taps(blk.k_off + k_tile * cfg.tile_k):
                stager.copy(src, stager.dst(lds.b, stage_tile, i), voff)

        # Double-buffer TILES_PER_BARRIER K-tiles: prefetch, then compute
        # while issuing the next DMA.
        PREFETCH = TILES_PER_BARRIER
        for s in range_constexpr(PREFETCH):
            if const_expr(s < kernel_grid.tiles_per_split):
                async_load_a_to_lds(s, s)
                async_load_b_to_lds(s, s)

        for kt_idx in range_constexpr(
            0, kernel_grid.tiles_per_split, TILES_PER_BARRIER
        ):
            batch = range_constexpr(
                kt_idx, min(kt_idx + TILES_PER_BARRIER, kernel_grid.tiles_per_split)
            )

            barrier(vmcnt=0, lgkmcnt=0)
            a_frags = [mma.read_a(k_tile % cfg.pipe_stages) for k_tile in batch]
            b_frags = [mma.read_b(k_tile % cfg.pipe_stages) for k_tile in batch]
            issued = 0
            for k_tile in batch:
                nxt = k_tile + PREFETCH
                if const_expr(nxt < kernel_grid.tiles_per_split):
                    async_load_a_to_lds(nxt, nxt % cfg.pipe_stages)
                    async_load_b_to_lds(nxt, nxt % cfg.pipe_stages)
                    issued += cfg.ldg_a_count + cfg.ldg_b_count
            if const_expr(issued):
                fx.rocdl.sched_vmem(issued)
            for j in range_constexpr(len(batch)):
                acc = mma.compute(acc, a_frags[j], b_frags[j])

        scatter.store(
            acc,
            m_offset=blk.m_offset,
            n_offset=blk.n_offset,
            n_local=blk.n_local,
            c_row=mma.c_row,
            c_col=mma.c_col,
        )

    if const_expr(param.dyn_hw):
        # The extents arrive as scalars. FlyDSL binds kernel parameters by
        # name, so the fourteen are spelled out rather than packed behind a
        # *args -- DynShapeArgs.FIELDS is the order, and reassembling one on
        # both sides is what keeps the two in step.
        @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
        def conv3d_implicit_kernel(
            y: fx.Tensor,
            x: fx.Tensor,
            weight: fx.Tensor,
            bias: fx.Tensor,
            d: fx.Int64,
            h: fx.Int64,
            w: fx.Int64,
            wo: fx.Int64,
            hw_o: fx.Int64,
            dhw: fx.Int64,
            npq: fx.Int64,
            rcp_d: fx.Int64,
            rcp_wo: fx.Int64,
            rcp_hw_o: fx.Int64,
            rcp_dhw: fx.Int64,
            grid_m: fx.Int64,
            x_elems: fx.Int64,
            x_sample_elems: fx.Int64,
        ):
            s = DynShapeArgs(
                d,
                h,
                w,
                wo,
                hw_o,
                dhw,
                npq,
                rcp_d,
                rcp_wo,
                rcp_hw_o,
                rcp_dhw,
                grid_m,
                x_elems,
                x_sample_elems,
            )
            _body(y, x, weight, bias, dyn_extents(s, dyn_unit), s.grid_m)

        @flyc.jit
        def launch(
            y: fx.Tensor,
            x: fx.Tensor,
            weight: fx.Tensor,
            bias: fx.Tensor,
            d: fx.Int64,
            h: fx.Int64,
            w: fx.Int64,
            wo: fx.Int64,
            hw_o: fx.Int64,
            dhw: fx.Int64,
            npq: fx.Int64,
            rcp_d: fx.Int64,
            rcp_wo: fx.Int64,
            rcp_hw_o: fx.Int64,
            rcp_dhw: fx.Int64,
            grid_m: fx.Int64,
            x_elems: fx.Int64,
            x_sample_elems: fx.Int64,
            stream: fx.Stream = fx.Stream(None),  # noqa: B008
        ):
            conv3d_implicit_kernel(
                y,
                x,
                weight,
                bias,
                d,
                h,
                w,
                wo,
                hw_o,
                dhw,
                npq,
                rcp_d,
                rcp_wo,
                rcp_hw_o,
                rcp_dhw,
                grid_m,
                x_elems,
                x_sample_elems,
            ).launch(
                # M is the only axis the resolution moves; chunking it is
                # ruled out on this path (see the assert in compile), so
                # grid.x is the tile count and z is the split alone.
                grid=(grid_m, kernel_grid.grid_y, kernel_grid.splitk),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

    else:

        @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
        def conv3d_implicit_kernel(
            y: fx.Tensor, x: fx.Tensor, weight: fx.Tensor, bias: fx.Tensor
        ):
            _body(y, x, weight, bias, static_extents(static_in_ext, geom), None)

        @flyc.jit
        def launch(
            y: fx.Tensor,
            x: fx.Tensor,
            weight: fx.Tensor,
            bias: fx.Tensor,
            stream: fx.Stream = fx.Stream(None),  # noqa: B008
        ):
            conv3d_implicit_kernel(y, x, weight, bias).launch(
                grid=(grid.grid_x, grid.grid_y, grid.grid_z),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

    def _launch(y, x, weight, bias, stream=None):
        with CompilationContext.compile_hints(CONV_COMPILE_HINTS):
            return launch(y, x, weight, bias, *extra_args, stream=_as_stream(stream))

    def _compile(y, x, weight, bias, stream=None):
        with CompilationContext.compile_hints(CONV_COMPILE_HINTS):
            return flyc.compile(
                launch, y, x, weight, bias, *extra_args, _as_stream(stream)
            )

    _launch.compile = _compile
    # The compiled function is called directly on the steady-state path (see
    # ``conv_kernels._dispatch``), which bypasses this closure, so the extents
    # have to be reachable from the object rather than captured here alone.
    _launch.extra_args = extra_args
    return _launch
