# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.
#
# ruff: noqa: B023
# The epilogue builds small closures over the tile loops and calls each one
# inside the same iteration, so the loop variable always holds the current
# value. Binding them as default arguments is not possible -- `bias_val` only
# exists on the has_bias path -- and per-line waivers do not survive
# `ruff format`, which moves the flagged column onto continuation lines.

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
    BF16_BYTES,
    CONV_COMPILE_HINTS,
    LDG_VEC,
    MFMA_C_VALUES,
    MFMA_M,
    MFMA_N,
    WARP_SIZE,
    _as_stream,
    barrier,
    buffer_atomic_add,
    dil,
    gather_valid,
    in_range,
    sgpr,
)

TILE_K = 32

# K tiles consumed between two barriers. Each one is MI_M * MI_N MFMAs, and that product
# is the only thing that hides global latency here -- see the PIPE_STAGES comment for why
# the pipeline depth cannot. Costs no LDS (the tiles are stages that already exist) and no
# extra ds_read/DMA traffic; it just halves the number of barriers. Worth +8..16% on the
# 3x3 conv2d/conv3d shapes at 2. Reaching the same ratio through TILE_K = 64 instead is a
# trap: it makes the LDS row stride 128B, exactly one bank rotation, and the resulting
# ds_read_b128 conflicts cost more than the batching wins (measured ~15% slower).
TILES_PER_BARRIER = 2

DEFAULT_TILE = (128, 128, 2, 4)


PADDING_MODES = ("zeros", "reflect", "replicate", "circular")


def validate_launch_config(tile_m, tile_n, wave_m, wave_n):
    """Why this (TILE_M, TILE_N, WAVE_M, WAVE_N) cannot compile, or None.

    The launch-config half of compile_conv3d_implicit's asserts, in a function
    that costs nothing to call, so a candidate sweep can filter on it instead of
    paying a compile per rejected config. conv3d_policy used to carry its own
    closed form of the same arithmetic -- the two agreed over all 8281
    combinations of its enumeration, but nothing made them, and a policy that
    drifts stricter prunes configs that would have compiled, which shows up as
    neither an error nor a wrong answer, only as a tuned pick that could have
    been faster.

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
    n, c, d, h, w, k = param.n, param.c, param.d, param.h, param.w, param.k
    kt, kh, kw = param.kt, param.kh, param.kw
    st, sh, sw = param.st, param.sh, param.sw
    pt, ph, pw = param.pt, param.ph, param.pw
    dt, dh, dw = param.dt, param.dh, param.dw
    pad_mode, has_bias, splitk = param.pad_mode, param.has_bias, param.splitk
    tile, wgm, groups, out_ndhwc = param.tile, param.wgm, param.groups, param.out_ndhwc

    TILE_M, TILE_N, WAVE_M, WAVE_N = tile
    BLOCK_THREADS = WAVE_M * WAVE_N * WARP_SIZE
    # MFMA atoms per wave. tiled_mma replicates the atom over the (WAVE_M, WAVE_N) wave
    # grid and tiles THAT over (TILE_M, TILE_N), so a wave's atoms are strided by the
    # whole wave grid rather than contiguous. The epilogue takes its row/col from
    # partition_C rather than rederiving that.
    MI_M = TILE_M // WAVE_M // MFMA_M
    MI_N = TILE_N // WAVE_N // MFMA_N
    BLOCK_VECS = LDG_VEC * BLOCK_THREADS
    LDG_A_COUNT = TILE_M * TILE_K // BLOCK_VECS
    LDG_B_COUNT = TILE_N * TILE_K // BLOCK_VECS

    # `c` is the padded TOTAL channel count and stays the NDHWC row stride. CGP is the
    # per-group channel count and is what the GEMM K axis decomposes against; the two
    # coincide only when groups == 1.
    CGP = c // groups
    KG = k // groups

    assert TILE_K == 32
    _invalid = validate_launch_config(TILE_M, TILE_N, WAVE_M, WAVE_N)
    assert _invalid is None, _invalid
    assert c % groups == 0, f"c={c} not divisible by groups={groups}"
    assert k % groups == 0, f"k={k} not divisible by groups={groups}"
    assert (
        CGP % LDG_VEC == 0
    ), f"c/groups={CGP} must be a multiple of LDG_VEC={LDG_VEC}; use _conv3d_impl to pad"
    assert BLOCK_THREADS <= 1024, f"BLOCK_THREADS={BLOCK_THREADS} exceeds 1024"

    # Dilation only stretches the filter's footprint; the K axis (CRS) is unchanged.
    do = (d + 2 * pt - (dt * (kt - 1) + 1)) // st + 1
    ho = (h + 2 * ph - (dh * (kh - 1) + 1)) // sh + 1
    wo = (w + 2 * pw - (dw * (kw - 1) + 1)) // sw + 1
    dhw = do * ho * wo
    hw_o = ho * wo
    npq = n * dhw
    crs = CGP * kt * kh * kw
    k_tiles = (crs + TILE_K - 1) // TILE_K

    BIG_IN = (n * c * d * h * w) > 0x7FFFFFFF
    BIG_OUT = (n * k * do * ho * wo * BF16_BYTES) > 0x7FFFFFFF

    X_BYTES = n * c * d * h * w * BF16_BYTES
    W_BYTES = k * crs * BF16_BYTES
    OOB_SENTINEL_ELEM = (
        0x7FFFFF80  # *2 = 0xFFFFFF00 bytes (~4.2950 GB), just under 2^32
    )
    OOB_SENTINEL_BYTES = OOB_SENTINEL_ELEM * BF16_BYTES
    BIG_IN_NR = 0x80000000  # 2 GB num_records for the rebased BIG_IN resource
    assert (
        W_BYTES < OOB_SENTINEL_BYTES
    ), f"weight {W_BYTES}B exceeds limit {OOB_SENTINEL_BYTES}B"
    assert X_BYTES < OOB_SENTINEL_BYTES or BIG_IN, f"input {X_BYTES}B exceeds limit"
    BIG_IN_N1 = BIG_IN and n == 1
    BIG_IN_NM = BIG_IN and n > 1
    X_SAMPLE_ELEMS = c * d * h * w

    # n > 1 rebases the descriptor once per sample, so a tap can sit anywhere in
    # the sample and the whole sample has to fit the 2 GB num_records -- unlike
    # the per-tile rebasing below, whose reach is bounded by the tile. Without
    # this check, taps past 2 GB fall outside num_records and read as zero, which
    # is silently wrong rather than an error. Note how little room that leaves:
    # BIG_IN needs n * sample > 2 GiB of elements, so at n == 2 the only sample
    # size that both trips BIG_IN and fits is exactly 2 GB.
    assert not BIG_IN_NM or X_SAMPLE_ELEMS * BF16_BYTES <= BIG_IN_NR, (
        f"batched input sample too large for the 32-bit gather: one sample spans "
        f"{X_SAMPLE_ELEMS * BF16_BYTES / 2**30:.2f} GiB, past the "
        f"{BIG_IN_NR / 2**30:.0f} GiB the per-sample buffer descriptor addresses. "
        f"Loop over N instead of batching."
    )

    _t_aligned = BIG_IN_N1 and hw_o % TILE_M == 0
    if BIG_IN_N1:
        _ot_span = (TILE_M - 1) // hw_o + (1 if _t_aligned else 2)
        _t_span = min(d - 1, (_ot_span - 1) * st + dt * (kt - 1))
        _h_span = (
            min(h - 1, ((TILE_M - 1) // wo + 1) * sh + dh * (kh - 1))
            if _t_aligned
            else h - 1
        )
        _span = (((_t_span * h + _h_span) * w + (w - 1)) * c + c) * BF16_BYTES
        assert _span <= BIG_IN_NR, (
            f"input sample too large for the 32-bit gather: a {TILE_M}-row tile reaches "
            f"{_span / 2**30:.2f} GiB from its rebased origin, past the "
            f"{BIG_IN_NR / 2**30:.0f} GiB the buffer descriptor addresses. Split the batch "
            f"over N, or pass a narrower tile=(TILE_M, ...)."
        )

    assert (
        pad_mode in PADDING_MODES
    ), f"pad_mode must be one of {PADDING_MODES}, got {pad_mode!r}"
    assert (
        pad_mode == "zeros" or not BIG_IN
    ), "non-zero pad_mode requires the non-BIG_IN address path"

    tiles_per_group = (KG + TILE_N - 1) // TILE_N
    n_tail = KG % TILE_N != 0
    grid_n = groups * tiles_per_group

    splitk = max(1, min(splitk, k_tiles))
    tiles_per_split = k_tiles // splitk
    use_splitk = splitk > 1

    Y_BYTES = npq * k * (4 if use_splitk else BF16_BYTES)

    assert (
        not use_splitk or npq * k * 4 <= SPLITK_MAX_STAGING_BYTES
    ), f"split-K staging {npq * k * 4}B exceeds the {SPLITK_MAX_STAGING_BYTES}B buffer window"

    PIPE_STAGES = 2 * TILES_PER_BARRIER

    LDS_A_SIZE = PIPE_STAGES * TILE_M * TILE_K
    LDS_B_SIZE = PIPE_STAGES * TILE_N * TILE_K

    grid_m = (npq + TILE_M - 1) // TILE_M

    MAX_GRID_X = 0xFFFFFFFF // BLOCK_THREADS
    MAX_GRID_YZ = 65535
    grid_x = min(grid_m, MAX_GRID_X)
    m_chunks = (grid_m + grid_x - 1) // grid_x

    _row_chk = (npq % TILE_M != 0) or (grid_x * m_chunks > grid_m)
    _need_chk = _row_chk or n_tail
    _vec_store = (
        (n == 1)
        and (not use_splitk)
        and (dhw % MFMA_C_VALUES == 0)
        and (not BIG_OUT)
        and (not out_ndhwc)
    )

    assert (
        grid_n <= MAX_GRID_YZ
    ), f"grid.y = {grid_n} exceeds the {MAX_GRID_YZ}-block limit"
    assert (
        m_chunks * splitk <= MAX_GRID_YZ
    ), f"grid.z = {m_chunks} M-chunks x {splitk} splits exceeds the {MAX_GRID_YZ}-block limit"

    WGM = 1 if m_chunks > 1 else max(1, int(wgm))
    elem_ty = fx.BFloat16
    temporal_only_fast = (
        kh == 1
        and kw == 1
        and st == 1
        and sh == 1
        and sw == 1
        and ph == 0
        and pw == 0
        and do == d
        and ho == h
        and wo == w
    )

    @fx.struct
    class SharedStorage:
        a: fx.Array[elem_ty, LDS_A_SIZE, 16]
        b: fx.Array[elem_ty, LDS_B_SIZE, 16]

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def conv3d_implicit_kernel(
        y: fx.Tensor, x: fx.Tensor, weight: fx.Tensor, bias: fx.Tensor
    ):
        # The im2col gather addresses its source by a flat element index, so the DMA source
        # is a 1-D view over the buffer rather than the tensor's own n-D layout: dividing
        # that by a 1-element tile makes slice(src, (None, off)) exactly element `off`,
        # with no coordinate decomposition. `elems` only shapes the view -- the sentinel
        # offset that masks a tap deliberately points past it, and num_records (not the
        # layout) is what turns that into a zero-fill.
        def _dma_src(ptr, elems, num_records_bytes):
            buf = fx.rocdl.make_buffer_ptr(ptr, num_records_bytes=num_records_bytes)
            return fx.logical_divide(
                fx.make_view(buf, fx.make_layout(elems, 1)), fx.make_layout(1, 1)
            )

        def _x_rebased(off_elems):
            # BIG_IN moves the descriptor base to the block's own origin so a 32-bit
            # voffset still reaches the tile; num_records bounds it at 2 GB from there.
            ptr = fx.add_offset(fx.get_iter(x), fx.make_int_tuple(off_elems))
            return _dma_src(ptr, BIG_IN_NR // BF16_BYTES, BIG_IN_NR)

        w_src = _dma_src(fx.get_iter(weight), W_BYTES // BF16_BYTES, W_BYTES)
        if const_expr(not BIG_IN):
            x_src = _dma_src(fx.get_iter(x), X_BYTES // BF16_BYTES, X_BYTES)
        y_buf = fx.rocdl.make_buffer_tensor(y, num_records_bytes=Y_BYTES)
        if const_expr(use_splitk):
            # buffer_atomic_add needs the raw !llvm.ptr<8> descriptor, not a tensor.
            y_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(y_buf))
        else:
            y_div = fx.logical_divide(
                fx.Tensor(fx.make_view(fx.get_iter(y_buf), fx.make_layout(npq * k, 1))),
                fx.make_layout(1, 1),
            )
            y_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), elem_ty)
            y_reg_1 = fx.make_rmem_tensor(1, elem_ty)
            if const_expr(_vec_store):
                y_atom_4 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), elem_ty)
                y_reg_4 = fx.make_rmem_tensor(MFMA_C_VALUES, elem_ty)
        if const_expr(has_bias):
            bias_div = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(bias), fx.make_layout(1, 1)
            )
            bias_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
            bias_reg = fx.make_rmem_tensor(1, fx.Float32)

        lds = fx.SharedAllocator(static=False).allocate(SharedStorage).peek()
        a_lds = lds.a
        b_lds = lds.b

        tid = fx.Int32(gpu.thread_id("x"))
        if const_expr(m_chunks > 1):
            m_chunk = fx.Int64(gpu.block_id("z")) % fx.Int64(m_chunks)
            m_offset = (
                fx.Int64(gpu.block_id("x")) + m_chunk * fx.Int64(grid_x)
            ) * TILE_M
            n_tile = fx.Int32(gpu.block_id("y"))
        elif const_expr(WGM > 1):
            pid = fx.Int64(gpu.block_id("x")) + fx.Int64(gpu.block_id("y")) * fx.Int64(
                grid_m
            )
            blocks_per_swizzle = fx.Int64(WGM * grid_n)
            swizzle_id = pid // blocks_per_swizzle
            first_m = swizzle_id * fx.Int64(WGM)
            swizzle_rows = fx.min(fx.Int64(grid_m) - first_m, fx.Int64(WGM))
            local = pid % blocks_per_swizzle
            m_offset = (first_m + (local % swizzle_rows)) * TILE_M
            n_tile = local // swizzle_rows
        else:
            m_offset = fx.Int32(gpu.block_id("x")) * TILE_M
            n_tile = fx.Int32(gpu.block_id("y"))

        if const_expr(groups > 1):
            gi = n_tile // tiles_per_group
            n_local = (n_tile % tiles_per_group) * TILE_N
            n_offset = gi * KG + n_local
            ch_base = gi * CGP
        else:
            n_offset = n_tile * TILE_N
            n_local = n_offset
        if const_expr(use_splitk):
            if const_expr(m_chunks > 1):
                split_idx = fx.Int64(gpu.block_id("z")) // fx.Int64(m_chunks)
            else:
                split_idx = fx.Int64(gpu.block_id("z"))
            k_off = split_idx * (tiles_per_split * TILE_K)
        else:
            k_off = 0

        if const_expr(BIG_IN_N1):
            nbase = m_offset // dhw
            rem0 = m_offset % dhw
            ot_base0 = rem0 // hw_o

            base_t = fx.max(ot_base0 * fx.Int64(st) - fx.Int64(pt), fx.Int64(0))
            if const_expr(_t_aligned):
                oh_base0 = (rem0 % hw_o) // wo
                base_h = fx.max(oh_base0 * fx.Int64(sh) - fx.Int64(ph), fx.Int64(0))
            else:
                base_h = fx.Int64(0)
            x_base_elem = (
                ((nbase * fx.Int64(d) + base_t) * fx.Int64(h) + base_h)
                * fx.Int64(w)
                * fx.Int64(c)
            )
            x_src = _x_rebased(fx.Int64(x_base_elem))

        Vec = fx.Vector
        lds_copy = fx.make_copy_atom(fx.UniversalCopy128b(), elem_ty)
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(MFMA_M, MFMA_N, TILE_K, elem_ty))
        tiled_mma = fx.make_tiled_mma(
            mma_atom, fx.make_layout((WAVE_M, WAVE_N, 1), (WAVE_N, 1, 0))
        )
        thr_mma = tiled_mma.thr_slice(tid)
        thr_copy_A = fx.make_tiled_copy_A(lds_copy, tiled_mma).get_slice(tid)
        thr_copy_B = fx.make_tiled_copy_B(lds_copy, tiled_mma).get_slice(tid)
        a_lds_layout = fx.make_layout((TILE_M, TILE_K), (TILE_K, 1))
        b_lds_layout = fx.make_layout((TILE_N, TILE_K), (TILE_K, 1))

        def stage_a(stage):
            return fx.make_view(
                fx.add_offset(a_lds.ptr, stage * TILE_M * TILE_K),
                a_lds_layout,
            )

        def stage_b(stage):
            return fx.make_view(
                fx.add_offset(b_lds.ptr, stage * TILE_N * TILE_K),
                b_lds_layout,
            )

        acc = thr_mma.make_fragment_C(
            fx.make_view(fx.get_iter(y), fx.make_layout((TILE_M, TILE_N), (TILE_N, 1)))
        )
        acc.fill(0.0)

        # Tile-local (row, col) of each accumulator element, taken from the same
        # tiled_mma that owns acc so the epilogue cannot drift from the MMA's own
        # partitioning. Each view's layout IS the coordinate, so partition_C
        # hands back coordinates rather than data.
        #
        # They have to be indexed flat: acc is ((MFMA_C_VALUES, 1), MI_M, MI_N),
        # and the hierarchical spellings trip a rank assertion in the layout
        # algebra. Flat index is v + MFMA_C_VALUES * (mi + MI_M * ni); a lane
        # holds one column and MFMA_C_VALUES consecutive rows per atom, so v = 0
        # of atom (mi, ni) is all the epilogue needs.
        c_row = thr_mma.partition_C(
            fx.make_view(0, fx.make_layout((TILE_M, TILE_N), (1, 0)))
        )
        c_col = thr_mma.partition_C(
            fx.make_view(0, fx.make_layout((TILE_M, TILE_N), (0, 1)))
        )

        def pad_coord(v, ext, pad):
            """Tap coordinate -> in-bounds input coordinate; returns (coord, mask).

            "zeros" leaves the coordinate alone and returns a range mask, which the
            caller folds into the OOB-sentinel routing so the load reads as zero. Every
            other mode resolves the coordinate into [0, ext) instead and returns no mask
            """
            if const_expr(pad_mode == "zeros"):
                return v, in_range(v, ext)
            u = v + fx.Int64(pad)
            low = u < fx.Int64(pad)  # v < 0
            high = u >= fx.Int64(pad + ext)  # v >= ext
            mid = u - fx.Int64(pad)  # v, where in range
            if const_expr(pad_mode == "replicate"):
                r = high.select(fx.Int64(ext - 1), mid)
                r = low.select(fx.Int64(0), r)
            elif const_expr(pad_mode == "reflect"):
                # [a b c d e] pad 2 -> [c b a b c d e d c]: -v near, 2*(ext-1) - v far.
                r = high.select(fx.Int64(2 * (ext - 1) + pad) - u, mid)
                r = low.select(fx.Int64(pad) - u, r)
            else:  # circular: v + ext near, v - ext far
                r = high.select(u - fx.Int64(pad + ext), mid)
                r = low.select(u + fx.Int64(ext - pad), r)
            return r, None

        _row_dec = []
        for i in range_constexpr(LDG_A_COUNT):
            linear = (tid + i * BLOCK_THREADS) * LDG_VEC
            local_m = linear // TILE_K
            local_k = linear % TILE_K
            row = m_offset + local_m
            row_valid = row < fx.Int64(npq)
            if const_expr(temporal_only_fast):
                out_t = (row // hw_o) % d
                _row_dec.append((local_k, row, row_valid, out_t))
            else:
                n_idx = row // dhw
                rem = row % dhw
                ot = rem // hw_o
                rem2 = rem % hw_o
                oh = rem2 // wo
                ow = rem2 % wo
                in_t0 = ot * st - pt
                in_h0 = oh * sh - ph
                in_w0 = ow * sw - pw
                if const_expr(BIG_IN_N1):
                    di = n_idx - nbase
                    _row_dec.append((local_k, row_valid, di, in_t0, in_h0, in_w0))
                elif const_expr(BIG_IN_NM):
                    _row_dec.append((local_k, row_valid, n_idx, in_t0, in_h0, in_w0))
                else:
                    _row_dec.append((local_k, row_valid, n_idx, in_t0, in_h0, in_w0))

        SCALAR_K = CGP % TILE_K == 0

        # The K axis decomposes against CGP (per-group channels) while every g_off below
        # keeps `c` (padded total channels) as the NDHWC row stride. `cc` is the absolute
        # input channel: the group base plus the offset within the group.
        def _a_addr(i, kbase_i, cc_base, ckk_base):
            dec = _row_dec[i]
            local_k = dec[0]
            k_abs = kbase_i + fx.Int64(local_k)
            if const_expr(SCALAR_K):
                cc = cc_base + fx.Int64(local_k)  # cc_base already carries ch_base
            else:
                cc = k_abs % CGP
                if const_expr(groups > 1):
                    cc = ch_base + cc
            k_valid = k_abs < fx.Int64(crs)
            if const_expr(temporal_only_fast):
                _, row, row_valid, out_t = dec
                kt_i = ckk_base if const_expr(SCALAR_K) else k_abs // CGP
                temporal_delta = dil(kt_i, dt) - pt
                in_t, m_t = pad_coord(out_t + temporal_delta, d, pt)
                valid = gather_valid(row_valid & k_valid, m_t)

                delta = (
                    temporal_delta
                    if const_expr(pad_mode == "zeros")
                    else (in_t - out_t)
                )
                if const_expr(BIG_IN_N1):
                    g_off = (
                        (row + delta * hw_o) - (fx.Int64(nbase) * dhw + base_t * hw_o)
                    ) * c + cc
                else:
                    g_off = (row + delta * hw_o) * c + cc
            else:
                ckk = ckk_base if const_expr(SCALAR_K) else k_abs // CGP
                kw_i = ckk % kw
                ckk2 = ckk // kw
                kh_i = ckk2 % kh
                kt_i = ckk2 // kh
                if const_expr(BIG_IN_N1):
                    _, row_valid, di, in_t0, in_h0, in_w0 = dec
                    in_t, m_t = pad_coord(in_t0 + dil(kt_i, dt), d, pt)
                    in_h, m_h = pad_coord(in_h0 + dil(kh_i, dh), h, ph)
                    in_w, m_w = pad_coord(in_w0 + dil(kw_i, dw), w, pw)
                    valid = gather_valid(row_valid & k_valid, m_t, m_h, m_w)
                    g_off = (
                        ((di * d + (in_t - base_t)) * h + (in_h - base_h)) * w + in_w
                    ) * c + cc
                elif const_expr(BIG_IN_NM):
                    _, row_valid, n_idx, in_t0, in_h0, in_w0 = dec
                    in_t, m_t = pad_coord(in_t0 + dil(kt_i, dt), d, pt)
                    in_h, m_h = pad_coord(in_h0 + dil(kh_i, dh), h, ph)
                    in_w, m_w = pad_coord(in_w0 + dil(kw_i, dw), w, pw)
                    valid = gather_valid(row_valid & k_valid, m_t, m_h, m_w)
                    g_off = ((in_t * h + in_h) * w + in_w) * c + cc
                    return fx.Int32(g_off), valid, n_idx
                else:
                    _, row_valid, n_idx, in_t0, in_h0, in_w0 = dec
                    in_t, m_t = pad_coord(in_t0 + dil(kt_i, dt), d, pt)
                    in_h, m_h = pad_coord(in_h0 + dil(kh_i, dh), h, ph)
                    in_w, m_w = pad_coord(in_w0 + dil(kw_i, dw), w, pw)
                    valid = gather_valid(row_valid & k_valid, m_t, m_h, m_w)
                    g_off = (((n_idx * d + in_t) * h + in_h) * w + in_w) * c + cc
            return fx.Int32(g_off), valid

        def _b_addr(i, k_base):
            linear = (tid + i * BLOCK_THREADS) * LDG_VEC
            local_n = linear // TILE_K
            local_k = linear % TILE_K
            col = n_offset + fx.Int64(local_n)
            g_off = fx.Int32(col * crs + (fx.Int64(k_base) + fx.Int64(local_k)))
            # Tail is per group: the N grid is over-provisioned to groups*tiles_per_group.
            col_valid = (
                ((n_local + fx.Int64(local_n)) < fx.Int64(KG))
                if const_expr(n_tail)
                else None
            )
            return g_off, col_valid

        DMA_BYTES = LDG_VEC * BF16_BYTES  # 16
        OOB_ELEM = fx.Int32(OOB_SENTINEL_ELEM)

        _lds_dma_ptr_ty = fx.PointerType.get(
            elem_ty.ir_type, fx.AddressSpace.Shared, DMA_BYTES
        )

        _dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), DMA_BYTES * 8)

        def _lds_dma_ptr(lds_array, stage_tile, i):
            # buffer_load_lds takes one wave-uniform LDS base and fans the wave's lanes
            # out from it, so the lane-0 address is the base the whole wave writes from.
            off_elems = fx.Int64(stage_tile) + (
                fx.Int64(tid) + fx.Int64(i * BLOCK_THREADS)
            ) * fx.Int64(LDG_VEC)
            base_bytes = off_elems * fx.Int64(BF16_BYTES)
            addr = fx.Int64(fx.ptrtoint(lds_array.ptr)) + fx.Int64(base_bytes)
            return fx.make_view(
                fx.inttoptr(_lds_dma_ptr_ty, sgpr(addr)), fx.make_layout(1, 1)
            )

        def _dma_to_lds(src, dst, voff_elem):
            fx.copy(_dma_atom, fx.slice(src, (None, voff_elem)), dst)

        def _load_a(stage, k_base):
            kbase_i = fx.Int64(k_base)
            cc_base = ckk_base = None
            if const_expr(SCALAR_K):
                cc_base = kbase_i % CGP
                if const_expr(groups > 1):
                    cc_base = ch_base + cc_base
                ckk_base = kbase_i // CGP
            stage_tile = fx.Int64(stage) * TILE_M * TILE_K
            for i in range_constexpr(LDG_A_COUNT):
                if const_expr(BIG_IN_NM):
                    addr_ret = _a_addr(i, kbase_i, cc_base, ckk_base)
                    g_off_i, valid, n_idx_i = addr_ret
                    x_src_i = _x_rebased(fx.Int64(n_idx_i) * fx.Int64(X_SAMPLE_ELEMS))
                    voff = valid.select(g_off_i, OOB_ELEM)
                    _dma_to_lds(x_src_i, _lds_dma_ptr(a_lds, stage_tile, i), voff)
                else:
                    g_off_i, valid = _a_addr(i, kbase_i, cc_base, ckk_base)
                    voff = valid.select(g_off_i, OOB_ELEM)
                    _dma_to_lds(x_src, _lds_dma_ptr(a_lds, stage_tile, i), voff)

        def _load_b(stage, k_base):
            stage_tile = fx.Int64(stage) * TILE_N * TILE_K
            for i in range_constexpr(LDG_B_COUNT):
                g_off, col_valid = _b_addr(i, k_base)
                if const_expr(n_tail):
                    voff = col_valid.select(g_off, OOB_ELEM)
                else:
                    voff = g_off
                _dma_to_lds(w_src, _lds_dma_ptr(b_lds, stage_tile, i), voff)

        def read_a_frags(stage):
            sA = stage_a(stage)
            frag_A = thr_mma.make_fragment_A(sA)
            fx.copy(lds_copy, thr_copy_A.partition_S(sA), thr_copy_A.retile(frag_A))
            fx.rocdl.sched_dsrd(MI_M)
            return frag_A

        def read_b_frags(stage):
            sB = stage_b(stage)
            frag_B = thr_mma.make_fragment_B(sB)
            fx.copy(lds_copy, thr_copy_B.partition_S(sB), thr_copy_B.retile(frag_B))
            fx.rocdl.sched_dsrd(MI_N)
            return frag_B

        def do_compute(acc_values, a_frag_values, b_frag_values):
            fx.rocdl.s_setprio(1)
            fx.gemm(
                tiled_mma,
                acc_values,
                a_frag_values,
                b_frag_values,
                acc_values,
            )
            fx.rocdl.sched_mfma(MI_M * MI_N)
            fx.rocdl.s_setprio(0)
            return acc_values

        PREFETCH = TILES_PER_BARRIER
        for s in range_constexpr(PREFETCH):
            if const_expr(s < tiles_per_split):
                _load_a(s, k_off + s * TILE_K)
                _load_b(s, k_off + s * TILE_K)

        for kt_idx in range_constexpr(0, tiles_per_split, TILES_PER_BARRIER):
            batch = range_constexpr(
                kt_idx, min(kt_idx + TILES_PER_BARRIER, tiles_per_split)
            )

            barrier(vmcnt=0, lgkmcnt=0)
            a_frags = [read_a_frags(k_tile % PIPE_STAGES) for k_tile in batch]
            b_frags = [read_b_frags(k_tile % PIPE_STAGES) for k_tile in batch]
            issued = 0
            for k_tile in batch:
                nxt = k_tile + PREFETCH
                if const_expr(nxt < tiles_per_split):
                    _load_a(nxt % PIPE_STAGES, k_off + nxt * TILE_K)
                    _load_b(nxt % PIPE_STAGES, k_off + nxt * TILE_K)
                    issued += LDG_A_COUNT + LDG_B_COUNT
            if const_expr(issued):
                fx.rocdl.sched_vmem(issued)
            for j in range_constexpr(len(batch)):
                acc = do_compute(acc, a_frags[j], b_frags[j])

        if const_expr(BIG_OUT):
            y_elem_base = fx.Int64(fx.ptrtoint(fx.get_iter(y)))

        _big_st_ptr_ty = fx.PointerType.get(
            elem_ty.ir_type, fx.AddressSpace.Global, BF16_BYTES
        )

        def _big_store(off_nk_i64, value):
            # BIG_OUT means y is past what a buffer descriptor's 32-bit voffset
            # reaches, so there is no buffer-resource form to route this through
            # and the store is addressed by a flat 64-bit address instead. That
            # is also why this path gives up y_div and the store copy atoms.
            addr = y_elem_base + off_nk_i64 * fx.Int64(BF16_BYTES)
            fx.ptr_store(value, fx.inttoptr(_big_st_ptr_ty, addr))

        def _valid(row, col_loc):
            if const_expr(_row_chk and n_tail):
                return (row < fx.Int64(npq)) & (col_loc < fx.Int64(KG))
            if const_expr(_row_chk):
                return row < fx.Int64(npq)
            return col_loc < fx.Int64(KG)

        _route_store = _need_chk and not use_splitk and not BIG_OUT

        def _route(off, row, col_loc):
            if const_expr(not _route_store):
                return fx.Int32(off)
            return _valid(row, col_loc).select(fx.Int32(off), OOB_ELEM)

        def _cols(ni):
            """Global out-channel for MFMA column block ni, and its index within the group."""
            col_off = fx.Int64(fx.get_scalar(c_col[MFMA_C_VALUES * MI_M * ni]))
            col = n_offset + col_off
            return col, ((n_local + col_off) if const_expr(groups > 1) else col)

        def store_acc():
            if const_expr(has_bias and not use_splitk):
                bias_vals = []
                for ni in range_constexpr(MI_N):
                    col, col_loc = _cols(ni)
                    col_i = fx.Int32(col)  # bias is indexed by the global out-channel
                    if const_expr(n_tail):
                        col_i = (col_loc < fx.Int64(KG)).select(col_i, fx.Int32(0))
                    fx.copy(bias_atom, fx.slice(bias_div, (None, col_i)), bias_reg)
                    bias_vals.append(fx.Float32(fx.memref_load_vec(bias_reg)[0]))

            for mi in range_constexpr(MI_M):
                row_base = m_offset + fx.get_scalar(c_row[MFMA_C_VALUES * mi])
                for ni in range_constexpr(MI_N):
                    col, col_loc = _cols(ni)
                    a = Vec(acc[None, mi, ni].load())
                    if const_expr(has_bias and not use_splitk):
                        bias_val = bias_vals[ni]

                    if const_expr(_vec_store):
                        row0 = fx.Int64(row_base)
                        off_nk0 = col * dhw + row0

                        def _emit_vec():
                            vals = []
                            for i in range_constexpr(MFMA_C_VALUES):
                                cval = (
                                    (a[i] + bias_val) if const_expr(has_bias) else a[i]
                                )
                                vals.append(cval.to(elem_ty))
                            v4 = fx.Vector.from_elements(vals, dtype=elem_ty)
                            fx.memref_store_vec(v4, y_reg_4)
                            fx.copy(
                                y_atom_4,
                                y_reg_4,
                                fx.slice(y_div, (None, _route(off_nk0, row0, col_loc))),
                            )

                        if const_expr(_need_chk and not _route_store):
                            if _valid(row0, col_loc):
                                _emit_vec()
                        else:
                            _emit_vec()
                        continue

                    for i in range_constexpr(MFMA_C_VALUES):
                        row = fx.Int64(row_base + i)
                        off_sk = row * k + col

                        if const_expr(out_ndhwc):
                            # (n, do, ho, wo, k) is row-major over exactly the GEMM's own
                            # (npq, k) index space, so the scatter collapses to off_sk and
                            # the row decomposition disappears.
                            off_nk = off_sk
                        elif const_expr(n == 1):
                            off_nk = col * dhw + row
                        else:
                            n_idx = row // dhw
                            sp = row % dhw
                            off_nk = n_idx * (k * dhw) + col * dhw + sp

                        def _emit():
                            if const_expr(use_splitk):
                                off_b = fx.Int32(off_sk * 4)
                                z0 = fx.Int32(0)
                                buffer_atomic_add(a[i], y_rsrc, off_b, z0, z0)
                            else:
                                cval = (
                                    (a[i] + bias_val).to(elem_ty)
                                    if const_expr(has_bias)
                                    else a[i].to(elem_ty)
                                )
                                if const_expr(BIG_OUT):
                                    _big_store(fx.Int64(off_nk), cval)
                                else:
                                    fx.memref_store_vec(
                                        fx.Vector.filled(1, cval, elem_ty), y_reg_1
                                    )
                                    fx.copy(
                                        y_atom_1,
                                        y_reg_1,
                                        fx.slice(
                                            y_div, (None, _route(off_nk, row, col_loc))
                                        ),
                                    )

                        if const_expr(_need_chk and not _route_store):
                            if _valid(row, col_loc):
                                _emit()
                        else:
                            _emit()

        store_acc()

    @flyc.jit
    def launch(
        y: fx.Tensor,
        x: fx.Tensor,
        weight: fx.Tensor,
        bias: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        conv3d_implicit_kernel(y, x, weight, bias).launch(
            grid=(grid_x, grid_n, m_chunks * splitk),
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


SPLITK_MAX_STAGING_BYTES = 0xFFFFFFFF
