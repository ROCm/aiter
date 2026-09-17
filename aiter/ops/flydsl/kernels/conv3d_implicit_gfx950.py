# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Double-buffered implicit-GEMM conv3d (BF16), vendored into aiter.

Upstream is FlyDSL ``kernels/conv/conv3d_implicit.py`` and the public entry point
still matches its keyword surface, but the body has diverged. aiter-only here:
``buffer_atomic_add`` -- which upstream imports from ``kernels/common/``, a
directory flydsl's wheel does not ship, as with the vendored ``buffer_ops``
and ``vector`` modules. The tile heuristics and the offline tuned-config
lookup are aiter-only too but live in ``../conv_kernels.py``, mirroring how
``tuned_gemm.py`` sits outside ``kernels/gemm_a16w16_gfx950.py``. The NCDHW
pre-transpose is a second kernel with its own cache, so it lives in
``conv3d_transpose.py``; what the two share is in ``conv3d_gfx950_utils.py``.

Launching goes through ``conv_kernels._dispatch`` rather than aiter's
``tensor_shim._run_compiled``: keeping the launcher shape comparable to upstream
is what makes a re-sync a readable diff, and a conv is launched once per layer
rather than in a tight loop, so the per-call dispatch ``_run_compiled`` saves
does not pay for that divergence.

x: (N, C, D, H, W) bf16 NCDHW by default, weight: (K, C/groups, T, R, S) bf16 KCTRS.
Returns (N, K, Do, Ho, Wo) bf16 by default. ``input_layout`` / ``output_layout`` select
NCDHW or NDHWC independently; the GEMM itself is channels-last, so NDHWC input skips the
pre-transpose and NDHWC output is the raw row-major (npq, K) the epilogue produces.
Supports stride, padding (int, per-axis tuple, or torch's "same" / "valid"),
padding_mode, dilation, bias, groups, and split-K.
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
    LdsStager,
    MmaTiling,
    OutputScatter,
    WeightLoader,
    _as_stream,
    barrier,
    block_coords,
    make_conv_geometry,
    make_launch_grid,
    make_output_scatter_plan,
    make_shared_storage,
    make_tile_config,
    weight_bytes,
)
from .conv3d_im2col import Im2colGather, make_im2col_plan


@fx.struct
class Conv3dImplicitParam:
    """One compiled conv3d: the problem it solves and the config it runs.

    Every field is a compile-time constant -- the im2col div/mod folding
    against the filter extents and C/groups is where this kernel's performance
    comes from -- so one of these is one artifact, and it is the cache key
    ``compile_conv3d_implicit`` is memoised on. Build it through
    ``make_conv3d_implicit_param``, which supplies the defaults fx.struct
    cannot.
    """

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
):
    """Conv3dImplicitParam with the defaults filled in.

    fx.struct has no field defaults, so the ones a caller may leave out live
    here, as ``make_gemm_a16w16_gfx950_param`` does for the GEMM.
    """
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
    )


# One entry per (shape, launch config). A tuning sweep walks ~100 configs per
# shape and several shapes land in the same worker process, so the upstream 256
# would evict entries that the same process still needs.
@functools.lru_cache(maxsize=1024)
def compile_conv3d_implicit(param: Conv3dImplicitParam):
    # Only what sizes the GEMM and its grid. The filter extents, strides,
    # padding and dilation belong to the gather, and the output layout and
    # bias to the scatter; each reaches its own plan below from `param`
    # directly, which is why none of them is unpacked here.
    c, k, groups = param.c, param.k, param.groups

    cfg = make_tile_config(param.tile)
    BLOCK_THREADS = cfg.block_threads

    # The implicit GEMM this convolution is, derived once and shared with the
    # gather so the grid the epilogue writes cannot drift from the one A is
    # read against. `c` is the padded TOTAL channel count and stays the NDHWC
    # row stride, while CGP is the per-group channel count the GEMM K axis
    # decomposes against; the two coincide only when groups == 1.
    geom = make_conv_geometry(param)
    CGP = geom.cgp

    # The launch config was validated by make_tile_config; these are the
    # problem's own constraints, which no tile can satisfy on its behalf.
    assert c % groups == 0, f"c={c} not divisible by groups={groups}"
    assert k % groups == 0, f"k={k} not divisible by groups={groups}"
    assert CGP % LDG_VEC == 0, (
        f"c/groups={CGP} must be a multiple of LDG_VEC={LDG_VEC}; use _conv3d_impl to pad"
    )

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

    elem_ty = fx.BFloat16
    SharedStorage = make_shared_storage(elem_ty, cfg)

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def conv3d_implicit_kernel(
        y: fx.Tensor, x: fx.Tensor, weight: fx.Tensor, bias: fx.Tensor
    ):
        # A's whole convolution: which input element each A vector taps, and
        # through which descriptor. Everything downstream of the gather treats
        # A as an ordinary GEMM operand.
        im2col = Im2colGather(im2col_plan, x)

        # B needs no gather at all: the weight is already a (K, CRS) matrix.
        weights = WeightLoader(cfg, grid, geom, weight, W_BYTES)

        # And how C goes back: the epilogue's descriptors and copy atoms, built
        # here with the others; the store itself happens at the end.
        scatter = OutputScatter(scatter_plan, y, bias, elem_ty)

        lds = fx.SharedAllocator(static=False).allocate(SharedStorage).peek()

        tid = fx.Int32(gpu.thread_id("x"))
        # Which (M, N, K) tile this block owns: WGM swizzle or M chunking,
        # then grouped N, then split-K.
        blk = block_coords(grid)

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
            if const_expr(s < grid.tiles_per_split):
                async_load_a_to_lds(s, s)
                async_load_b_to_lds(s, s)

        for kt_idx in range_constexpr(0, grid.tiles_per_split, TILES_PER_BARRIER):
            batch = range_constexpr(
                kt_idx, min(kt_idx + TILES_PER_BARRIER, grid.tiles_per_split)
            )

            barrier(vmcnt=0, lgkmcnt=0)
            a_frags = [mma.read_a(k_tile % cfg.pipe_stages) for k_tile in batch]
            b_frags = [mma.read_b(k_tile % cfg.pipe_stages) for k_tile in batch]
            issued = 0
            for k_tile in batch:
                nxt = k_tile + PREFETCH
                if const_expr(nxt < grid.tiles_per_split):
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
            return launch(y, x, weight, bias, stream=_as_stream(stream))

    def _compile(y, x, weight, bias, stream=None):
        with CompilationContext.compile_hints(CONV_COMPILE_HINTS):
            return flyc.compile(launch, y, x, weight, bias, _as_stream(stream))

    _launch.compile = _compile
    return _launch
