# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""gfx1250 double-buffered implicit-GEMM conv3d (BF16, TDM im2col + WMMA).

Host pre-pads so the kernel always sees zeros padding. Input is NDHWC with
C/groups aligned to 32. Dispatch lives in ``conv3d_gfx1250.py``.

The software pipeline matches ``compile_gemm_a16w16`` (prologue / scf.for main
loop / drain, bandwidth_bound or compute_bound fragment prefetch;
``b_prefetch`` reorders bandwidth_bound's fragment loads). Supports
BIG_OUT (64-bit stores), split-K (fp32 atomic staging), and dyn_hw (runtime
extents via magic reciprocal divisors).

With ``patch`` (stride-1 kH x kW > 1 convs, the gfx1250 counterpart of the
gfx950 kH x kW reuse) the im2col input is fetched once per (kt, cc) group as
an (R + kH - 1) x (C + kW - 1) TDM patch and all kH * kW taps read it at
constant LDS offsets; the weights keep the per-tap ring.
"""

import functools
from types import SimpleNamespace
from typing import NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.compiler.protocol import dsl_size_of
from flydsl.expr import const_expr, gpu, idx2crd, range_constexpr, rocdl, tdm_ops
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from aiter.jit.utils.chip_info import get_lds_capacity_bytes

from .conv3d_gfx1250_utils import (
    BF16_BYTES,
    CGP_ALIGN,
    CONV_COMPILE_HINTS,
    DEFAULT_BLOCK_ROWS,
    DEFAULT_TILE,
    LDS_K_CHUNK,
    OOB_SENTINEL_ELEM,
    TILE_K,
    WAVE_SIZE,
    WMMA_K,
    WMMA_M,
    WMMA_N,
    DynShapeArgs1250,
    _as_stream,
    buffer_atomic_add,
    decode_block_2d,
    dyn_shape_values_gfx1250,
    lane_bases_rowmajor,
    load_wmma_frag,
    make_conv_geometry_gfx1250,
    make_launch_grid_gfx1250,
    make_shared_storage_gfx1250,
    make_tile_config_gfx1250,
    patch_extent_gfx1250,
    patch_stage_elems_gfx1250,
    patch_unsupported_gfx1250,
    rmem_vec,
    unit_divisors_gfx1250,
)

_SCHED_ALLOW_SALU = 1 << 2
_TDMS_PER_TILE = 2  # A + B
# b_prefetch: WMMA groups between the last read of an A fragment register and
# the load that refills it with the next tile's fragment.
A_ROT_LAG = 2


class Conv3dImplicitParam1250(NamedTuple):
    n: int
    c: int  # padded C (groups * cgp)
    d: int
    h: int
    w: int
    k: int
    kt: int
    kh: int
    kw: int
    st: int
    sh: int
    sw: int
    dt: int
    dh: int
    dw: int
    has_bias: bool
    tile: tuple
    groups: int
    out_ndhwc: bool
    block_rows: int
    num_buffers: int
    splitk: int
    dyn_hw: bool
    variant: str
    main_loop_unroll: bool
    tile_k: int
    patch: bool
    b_prefetch: bool


def make_conv3d_implicit_param_gfx1250(
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
    dt=1,
    dh=1,
    dw=1,
    has_bias=False,
    tile=DEFAULT_TILE,
    groups=1,
    out_ndhwc=False,
    block_rows=DEFAULT_BLOCK_ROWS,
    num_buffers=2,
    splitk=1,
    dyn_hw=False,
    variant="bandwidth_bound",
    main_loop_unroll=False,
    tile_k=TILE_K,
    patch=False,
    b_prefetch=False,
    **_ignored,
):
    del _ignored  # e.g. legacy patch_reuse from older call sites
    if variant not in ("bandwidth_bound", "compute_bound"):
        raise ValueError(f"bad variant {variant!r}")
    return Conv3dImplicitParam1250(
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
        dt=dt,
        dh=dh,
        dw=dw,
        has_bias=has_bias,
        tile=tuple(tile),
        groups=groups,
        out_ndhwc=out_ndhwc,
        block_rows=block_rows,
        num_buffers=num_buffers,
        splitk=max(1, int(splitk)),
        dyn_hw=bool(dyn_hw),
        variant=variant,
        main_loop_unroll=bool(main_loop_unroll),
        tile_k=int(tile_k),
        patch=bool(patch),
        b_prefetch=bool(b_prefetch) and variant == "bandwidth_bound",
    )


def _dyn_hw_closure_key_gfx1250(grid, unit, out_ndhwc, has_bias, splitk):
    """Compile-time constants a dyn_hw kernel closes over (booleans included)."""
    return (
        grid.tile_m,
        grid.tile_n,
        grid.block_rows,
        grid.block_cols,
        grid.groups,
        grid.kg,
        grid.cgp,
        grid.tiles_per_group,
        grid.n_tail,
        grid.row_chk,
        grid.k_tiles,
        grid.splitk,
        grid.tiles_per_split,
        grid.use_splitk,
        grid.big_out,
        unit,
        out_ndhwc,
        has_bias,
        splitk,
    )


def _shape_agnostic_key_gfx1250(grid, unit, out_ndhwc, has_bias, splitk):
    """Closure key with booleans blanked — integers must not carry D/H/W."""

    def _blank(value):
        if isinstance(value, bool):
            return None
        if isinstance(value, tuple):
            return tuple(_blank(v) for v in value)
        return value

    return tuple(
        _blank(v)
        for v in _dyn_hw_closure_key_gfx1250(grid, unit, out_ndhwc, has_bias, splitk)
    )


def _assert_shape_agnostic_gfx1250(param, cfg, kernel_grid, unit):
    """Probe one stride larger; integer (non-bool) closure must not move."""
    st, sh, sw = param.st, param.sh, param.sw
    probe = make_conv3d_implicit_param_gfx1250(
        param.n,
        param.c,
        param.d + st,
        param.h + sh,
        param.w + sw,
        param.k,
        param.kt,
        param.kh,
        param.kw,
        param.st,
        param.sh,
        param.sw,
        param.dt,
        param.dh,
        param.dw,
        param.has_bias,
        param.tile,
        param.groups,
        param.out_ndhwc,
        param.block_rows,
        param.num_buffers,
        param.splitk,
        True,
        param.variant,
        param.main_loop_unroll,
        param.tile_k,
        param.patch,
        param.b_prefetch,
    )
    cgp = probe.c // probe.groups
    try:
        geom = make_conv_geometry_gfx1250(
            probe.n,
            cgp,
            probe.d,
            probe.h,
            probe.w,
            probe.k,
            probe.kt,
            probe.kh,
            probe.kw,
            probe.st,
            probe.sh,
            probe.sw,
            0,
            0,
            0,
            probe.dt,
            probe.dh,
            probe.dw,
        )
        grid = make_launch_grid_gfx1250(
            probe.n, geom, cfg, probe.groups, probe.k, splitk=probe.splitk
        )
    except AssertionError:
        return
    blank = grid._replace(grid_m=0, hb=0, wb=0, do=0, ho=0, wo=0, n=0)
    probe_unit = unit_divisors_gfx1250(grid, geom)
    mine = _shape_agnostic_key_gfx1250(
        kernel_grid, unit, param.out_ndhwc, param.has_bias, param.splitk
    )
    theirs = _shape_agnostic_key_gfx1250(
        blank,
        probe_unit,
        probe.out_ndhwc,
        probe.has_bias,
        probe.splitk,
    )
    assert mine == theirs, (
        "variable resolution asked for, but a compile-time constant still "
        f"carries the input extents: {param.d}x{param.h}x{param.w} vs "
        f"{probe.d}x{probe.h}x{probe.w} "
        f"({[(a, b) for a, b in zip(mine, theirs) if a != b]})"
    )


@functools.lru_cache(maxsize=512)
def compile_conv3d_implicit_gfx1250(param: Conv3dImplicitParam1250):
    gpu_arch = str(get_hip_arch())
    assert gpu_arch.startswith("gfx1250"), f"Expected gfx1250, got {gpu_arch}"

    cfg = make_tile_config_gfx1250(param.tile, param.block_rows, param.tile_k)
    cgp = param.c // param.groups
    assert cgp % CGP_ALIGN == 0, f"cgp={cgp} not multiple of {CGP_ALIGN}"
    assert param.k % param.groups == 0

    geom = make_conv_geometry_gfx1250(
        param.n,
        cgp,
        param.d,
        param.h,
        param.w,
        param.k,
        param.kt,
        param.kh,
        param.kw,
        param.st,
        param.sh,
        param.sw,
        0,
        0,
        0,
        param.dt,
        param.dh,
        param.dw,
    )
    grid = make_launch_grid_gfx1250(
        param.n, geom, cfg, param.groups, param.k, splitk=param.splitk
    )

    R, C = cfg.block_rows, cfg.block_cols
    num_buffers = param.num_buffers
    assert 2 <= num_buffers <= 8
    num_k_tiles = grid.tiles_per_split
    assert num_k_tiles >= num_buffers - 1, (
        f"num_k_tiles {num_k_tiles} < {num_buffers - 1} "
        f"(crs={geom.crs}, tile_k={cfg.tile_k}, splitk={param.splitk})"
    )

    use_patch = param.patch
    TAPS = param.kh * param.kw
    if use_patch:
        reason = patch_unsupported_gfx1250(
            param.kh, param.kw, param.sh, param.sw, grid.splitk, num_buffers
        )
        assert reason is None, reason
    patch_geom = (R, C, param.kh, param.kw, param.dh, param.dw, cfg.num_warps)
    PATCH_ROWS, PATCH_COLS = patch_extent_gfx1250(*patch_geom)
    PATCH_ELEMS = patch_stage_elems_gfx1250(*patch_geom, cfg.tile_k)
    # Patch-relative LDS offset of tap t = (kh, kw).
    TAP_OFFS = [
        ((t // param.kw) * param.dh * PATCH_COLS + (t % param.kw) * param.dw)
        * cfg.lds_k_stride
        for t in range(TAPS)
    ]

    SharedStorage = make_shared_storage_gfx1250(
        fx.BFloat16,
        cfg,
        num_buffers,
        patch_stage_elems=PATCH_ELEMS if use_patch else None,
    )

    smem_bytes = dsl_size_of(SharedStorage)
    lds_cap = get_lds_capacity_bytes(gpu_arch)
    assert (
        smem_bytes <= lds_cap
    ), f"SharedStorage {smem_bytes}B exceeds LDS capacity {lds_cap}B on {gpu_arch}"

    BLOCK_THREADS = cfg.block_threads
    num_warps = cfg.num_warps
    wmma_m_rep, wmma_n_rep = cfg.mi_m, cfg.mi_n
    TK = cfg.tile_k
    k_wmma_steps = TK // WMMA_K
    # A wider K tile is tile_k / TILE_K channel chunks, each padded in LDS. TDM
    # needs the global and LDS tiles to share their contiguous innermost run, so
    # the global channel axis is split into (KC, TILE_K) as well.
    KC = TK // TILE_K
    N_ACCS = wmma_m_rep * wmma_n_rep
    N_A_FRAGS = wmma_m_rep * k_wmma_steps
    N_B_FRAGS = wmma_n_rep * k_wmma_steps
    elem_ty = fx.BFloat16
    CRS = geom.crs
    Y_ELEMS = param.n * geom.do * geom.ho * geom.wo * param.k
    use_splitk = grid.use_splitk
    big_out = grid.big_out
    # Staging / output byte window for the buffer descriptor (split-K is fp32).
    # BIG_OUT drops the buffer path; non-big uses this size for the descriptor.
    y_bytes = geom.npq * param.k * 4 if use_splitk else Y_ELEMS * BF16_BYTES

    assert cgp % TK == 0, f"tile_k={TK} does not divide cgp={cgp}"
    assert TK == TILE_K or not use_splitk, "split-K runs tile_k=32 only"
    N_CC = cgp // TK
    assert N_CC * param.kt * param.kh * param.kw == grid.k_tiles
    N_GROUPS = param.kt * N_CC

    variant = param.variant
    main_loop_unroll = param.main_loop_unroll
    b_prefetch = param.b_prefetch
    # A lane's 8 accumulator columns are 8 consecutive NDHWC channels; with K and
    # the per-group K multiples of 8 they are 16-byte aligned and wholly in or
    # out of a channel tail, so they go out as one store (bias as two loads).
    VEC_EPILOGUE = (
        param.out_ndhwc
        and not use_splitk
        and not big_out
        and param.k % 8 == 0
        and grid.kg % 8 == 0
    )
    # The bias read only needs the 8 columns aligned and in or out together.
    VEC_BIAS = not use_splitk and grid.kg % 8 == 0

    if const_expr(param.dyn_hw):
        extra_args = dyn_shape_values_gfx1250(param, geom, grid)
        dyn_unit = unit_divisors_gfx1250(grid, geom)
        kernel_grid = grid._replace(grid_m=0, hb=0, wb=0, do=0, ho=0, wo=0, n=0)
        _assert_shape_agnostic_gfx1250(param, cfg, kernel_grid, dyn_unit)
    else:
        extra_args = ()
        dyn_unit = None
        kernel_grid = grid

    def _make_store_accs(
        y,
        bias,
        warp_m_base,
        warp_n_base,
        lane16,
        lane_kgrp,
        n_idx,
        ot,
        oh0,
        ow0,
        n_offset,
        n_local,
        do_v,
        ho_v,
        wo_v,
    ):
        """Build epilogue: bf16 buffer / BIG_OUT ptr_store / split-K atomic."""
        if const_expr(use_splitk):
            y_buf = fx.rocdl.make_buffer_tensor(
                y, max_size=False, num_records_bytes=y_bytes
            )
            y_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(y_buf))
            y_div = st1_atom = None
            big_st_ptr_ty = None
        elif const_expr(big_out):
            y_rsrc = y_div = st1_atom = None
            big_st_ptr_ty = fx.PointerType.get(
                elem_ty.ir_type, fx.AddressSpace.Global, BF16_BYTES
            )
            y_elem_base = fx.Int64(fx.ptrtoint(fx.get_iter(y)))
        else:
            y_buf = fx.rocdl.make_buffer_tensor(
                y, max_size=False, num_records_bytes=y_bytes
            )
            y_div = fx.logical_divide(
                fx.Tensor(fx.make_view(fx.get_iter(y_buf), fx.make_layout(Y_ELEMS, 1))),
                fx.make_layout(1, 1),
            )
            st1_atom = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), elem_ty)
            st8_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_ty)
            y_rsrc = big_st_ptr_ty = None

        apply_bias = const_expr(param.has_bias and not use_splitk)
        if apply_bias:
            bias_buf = fx.rocdl.make_buffer_tensor(bias)
            bias_div = fx.logical_divide(bias_buf, fx.make_layout(1, 1))
            bias_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
            bias_reg = fx.make_rmem_tensor(1, fx.Float32)
            if const_expr(VEC_BIAS):
                bias_quads = fx.logical_divide(bias_buf, fx.make_layout(4, 1))
                bias4_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)

        def bias_for(wn):
            """This lane's 8 bias values of WMMA column block wn (two dwordx4 loads)."""
            col_off0 = warp_n_base + wn * WMMA_N + lane_kgrp * 8
            col_base = fx.Int32(n_offset + col_off0)
            if const_expr(kernel_grid.n_tail):
                in_grp = fx.Int64(n_local) + col_off0 < fx.Int64(kernel_grid.kg)
                col_base = in_grp.select(col_base, fx.Int32(0))
            elems = []
            for half in range_constexpr(2):
                r = fx.make_rmem_tensor(4, fx.Float32)
                fx.copy(
                    bias4_atom,
                    fx.slice(bias_quads, (None, col_base // 4 + half)),
                    r,
                )
                v = fx.Vector(r.load())
                for i in range_constexpr(4):
                    elems.append(v[i])
            return fx.Vector.from_elements(elems, fx.Float32)

        def store_accs(final_accs):
            k_out = param.k
            if const_expr(apply_bias and VEC_BIAS):
                bias_vecs = [bias_for(wn) for wn in range_constexpr(wmma_n_rep)]
            for wm in range_constexpr(wmma_m_rep):
                local_m = warp_m_base + wm * WMMA_M + lane16
                oh = oh0 + local_m // C
                ow = ow0 + local_m % C
                row_ok = (oh < fx.Int64(ho_v)) & (ow < fx.Int64(wo_v))
                for wn in range_constexpr(wmma_n_rep):
                    col_off0 = warp_n_base + wn * WMMA_N + lane_kgrp * 8
                    col_base = n_offset + col_off0
                    acc = fx.Vector(final_accs[wm * wmma_n_rep + wn])
                    if const_expr(apply_bias and VEC_BIAS):
                        acc = acc + bias_vecs[wn]
                    elif apply_bias:
                        bias_elems = []
                        for e in range_constexpr(8):
                            col_i = fx.Int32(col_base + e)
                            if const_expr(kernel_grid.n_tail):
                                in_grp = fx.Int64(n_local) + col_off0 + e < fx.Int64(
                                    kernel_grid.kg
                                )
                                col_i = in_grp.select(col_i, fx.Int32(0))
                            fx.copy(
                                bias_atom,
                                fx.slice(bias_div, (None, col_i)),
                                bias_reg,
                            )
                            bias_elems.append(
                                fx.Float32(fx.memref_load_vec(bias_reg)[0])
                            )
                        acc = acc + fx.Vector.from_elements(bias_elems, fx.Float32)

                    if const_expr(use_splitk):
                        # Tails are compile-gated out with split-K.
                        for e in range_constexpr(8):
                            col = col_base + e
                            row = ((n_idx * do_v + ot) * ho_v + oh) * wo_v + ow
                            off_sk = row * k_out + col
                            buffer_atomic_add(
                                acc[e],
                                y_rsrc,
                                fx.Int32(off_sk * 4),
                                fx.Int32(0),
                                fx.Int32(0),
                            )
                        continue

                    h_vec = acc.to(elem_ty)
                    if const_expr(VEC_EPILOGUE):
                        # 8 consecutive NDHWC channels, 16-byte aligned: one store.
                        off = (
                            ((n_idx * do_v + ot) * ho_v + oh) * wo_v * k_out
                            + ow * k_out
                            + col_base
                        )
                        valid = row_ok
                        if const_expr(kernel_grid.n_tail):
                            valid = valid & (
                                fx.Int64(n_local) + col_off0 < fx.Int64(kernel_grid.kg)
                            )
                        routed = valid.select(
                            fx.Int32(off), fx.Int32(OOB_SENTINEL_ELEM)
                        )
                        fx.copy(
                            st8_atom,
                            rmem_vec(h_vec, 8, elem_ty),
                            fx.slice(y_div, (None, routed)),
                        )
                        continue
                    for e in range_constexpr(8):
                        col = col_base + e
                        if const_expr(kernel_grid.n_tail):
                            col_ok = fx.Int64(n_local) + col_off0 + e < fx.Int64(
                                kernel_grid.kg
                            )
                        else:
                            col_ok = True
                        if const_expr(param.out_ndhwc):
                            off = (
                                ((n_idx * do_v + ot) * ho_v + oh) * wo_v * k_out
                                + ow * k_out
                                + col
                            )
                        else:
                            dhw = do_v * ho_v * wo_v
                            off = (
                                n_idx * (k_out * dhw)
                                + col * dhw
                                + (ot * ho_v + oh) * wo_v
                                + ow
                            )
                        valid = (
                            row_ok
                            if const_expr(not kernel_grid.n_tail)
                            else (row_ok & col_ok)
                        )
                        if const_expr(big_out):
                            # Tails gated out with BIG_OUT; always in-bounds.
                            addr = y_elem_base + fx.Int64(off) * fx.Int64(BF16_BYTES)
                            fx.ptr_store(h_vec[e], fx.inttoptr(big_st_ptr_ty, addr))
                        else:
                            routed = valid.select(
                                fx.Int32(off), fx.Int32(OOB_SENTINEL_ELEM)
                            )
                            one = fx.Vector.from_elements([h_vec[e]], elem_ty)
                            fx.copy(
                                st1_atom,
                                rmem_vec(one, 1, elem_ty),
                                fx.slice(y_div, (None, routed)),
                            )

        return store_accs

    def _prelude(y, x, weight, bias, dyn):
        """Block coordinates, B TDM, LDS fragment loads, WMMA and the epilogue.

        Shared by the per-tap and the patch K loops; it emits no runtime control
        flow, so it traces inline into either ``@flyc.jit`` body.
        """
        tid = fx.Int32(gpu.thread_id("x"))
        layout_thr = fx.make_layout(
            (cfg.wave_m, cfg.wave_n, 2, 16),
            (cfg.wave_n * WAVE_SIZE, WAVE_SIZE, 16, 1),
        )
        thr = idx2crd(tid, layout_thr)
        wave_m_idx = fx.Uint64(fx.get(thr, 0))
        wave_n_idx = fx.Uint64(fx.get(thr, 1))
        lane_kgrp = fx.Uint64(fx.get(thr, 2))
        lane16 = fx.Uint64(fx.get(thr, 3))
        warp_m_base = wave_m_idx * cfg.warp_tile_m
        warp_n_base = wave_n_idx * cfg.warp_tile_n

        n_idx, ot, oh0, ow0, n_offset, n_local, gi, k_off = decode_block_2d(
            kernel_grid, dyn
        )

        if const_expr(dyn is not None):
            Dp, Hp, Wp = dyn.d, dyn.h, dyn.w
            do_v, ho_v, wo_v = dyn.do, dyn.ho, dyn.wo
        else:
            Dp, Hp, Wp = param.d, param.h, param.w
            do_v, ho_v, wo_v = geom.do, geom.ho, geom.wo
        Cp = param.c

        lds = fx.SharedAllocator(static=True).allocate(SharedStorage).peek()
        big_a_mem, big_b_mem = lds.a.ptr, lds.b.ptr

        lds_k = cfg.lds_k_stride

        a_base = (
            ((n_idx * Dp + ot * param.st) * Hp + oh0 * param.sh) * Wp + ow0 * param.sw
        ) * Cp + fx.Int64(gi) * cgp
        row_stride = param.sh * Wp * Cp
        col_stride = param.sw * Cp
        # The tap shift is a runtime byte offset inside the flat K loop, so the
        # input gather builds its tiled atom per issue instead of indexing a tile.
        ho_left = fx.Int32(fx.Int64(ho_v) - oh0)
        wo_left = fx.Int32(fx.Int64(wo_v) - ow0)

        # Weights are K-tile aligned: one descriptor covers the whole channel
        # range, the tail is clamped by the view extent, and each round indexes
        # the K tile it needs. The LDS tile stride carries pad_interval/pad_amount.
        n_left = fx.Int32(fx.Int64(kernel_grid.kg) - fx.Int64(n_local))
        if const_expr(KC == 1):
            gB_layout = fx.make_layout((n_left, CRS), (CRS, 1))
            b_tiler = (cfg.tile_n, TILE_K)
            b_smem = fx.make_layout((cfg.tile_n, TILE_K), (lds_k, 1))
            b_box = fx.make_layout(
                ((cfg.tile_n, TILE_K), 1), ((lds_k, 1), cfg.tile_n * lds_k)
            )
            b_rest = (0, None)
        else:
            gB_layout = fx.make_layout(
                (n_left, CRS // TILE_K, TILE_K), (CRS, TILE_K, 1)
            )
            b_tiler = (cfg.tile_n, KC, TILE_K)
            b_smem = fx.make_layout((cfg.tile_n, KC, TILE_K), (lds_k, LDS_K_CHUNK, 1))
            b_box = fx.make_layout(
                ((cfg.tile_n, KC, TILE_K), 1),
                ((lds_k, LDS_K_CHUNK, 1), cfg.tile_n * lds_k),
            )
            b_rest = (0, None, 0)
        gB = fx.Tensor(
            fx.make_view(
                fx.add_offset(fx.get_iter(weight), fx.Int64(n_offset) * CRS),
                gB_layout,
            )
        )
        atom_b, coord_b = fx.rocdl.cdna5.make_tiled_tdm_atom(
            fx.rocdl.TensorLoad(),
            gB,
            b_smem,
            b_tiler,
            num_warps=num_warps,
        )
        blk_b = fx.zipped_divide(coord_b, b_tiler)[None, b_rest]

        wave = fx.rocdl.wave_id()
        warp_layout = fx.make_layout(num_warps, 1)

        a_bases = lane_bases_rowmajor(warp_m_base, wmma_m_rep, lds_k, lane16, lane_kgrp)
        b_bases = lane_bases_rowmajor(warp_n_base, wmma_n_rep, lds_k, lane16, lane_kgrp)

        def slot_a(buf):
            return fx.Uint64(buf * cfg.lds_a_elems_per_buf)

        def slot_b(buf):
            return fx.Uint64(buf * cfg.lds_b_elems_per_buf)

        def _decode_k(k_idx):
            ki = fx.Int64(k_idx)
            n_cc = fx.Int64(N_CC)
            cc = ki % n_cc
            t = ki // n_cc
            kw_i = t % param.kw
            t2 = t // param.kw
            kh_i = t2 % param.kh
            kt_i = t2 // param.kh
            return kt_i, kh_i, kw_i, cc

        def issue_tdm(buf_idx, abs_k):
            kt_i, kh_i, kw_i, cc = _decode_k(abs_k)
            tap_elems = (
                (kt_i * param.dt * Hp + kh_i * param.dh) * Wp + kw_i * param.dw
            ) * Cp + fx.Int64(cc) * TK
            if const_expr(KC == 1):
                gA_layout = fx.make_layout(
                    (ho_left, wo_left, TILE_K), (row_stride, col_stride, 1)
                )
                a_tiler = (R, C, TILE_K)
                a_box = fx.make_layout(
                    ((R, C, TILE_K), 1),
                    ((C * lds_k, lds_k, 1), R * C * lds_k),
                )
                a_rest = (0, 0, None)
            else:
                gA_layout = fx.make_layout(
                    (ho_left, wo_left, KC, TILE_K),
                    (row_stride, col_stride, TILE_K, 1),
                )
                a_tiler = (R, C, KC, TILE_K)
                a_box = fx.make_layout(
                    ((R, C, KC, TILE_K), 1),
                    ((C * lds_k, lds_k, LDS_K_CHUNK, 1), R * C * lds_k),
                )
                a_rest = (0, 0, 0, None)
            gA = fx.Tensor(
                fx.make_view(
                    fx.add_offset(fx.get_iter(x), a_base + tap_elems), gA_layout
                )
            )
            atom_a, coord_a = fx.rocdl.cdna5.make_tiled_tdm_atom(
                fx.rocdl.TensorLoad(),
                gA,
                a_box,
                a_tiler,
                num_warps=num_warps,
            )
            blk_a = fx.zipped_divide(coord_a, a_tiler)[None, a_rest]
            lds_a = fx.Tensor(
                fx.make_view(fx.add_offset(big_a_mem, slot_a(buf_idx)), a_box)
            )
            ts_a, tg_a = fx.rocdl.cdna5.tdm_partition(
                atom_a, wave, warp_layout, lds_a, blk_a
            )
            fx.copy(atom_a, tg_a[None, 0], ts_a[None, 0])
            issue_b(buf_idx, abs_k)

        def issue_b(buf_idx, k_tile):
            # Partition mode 0 is the complete box; the singleton rest mode
            # lets the copy select this already-offset pipeline slot with 0.
            lds_b = fx.Tensor(
                fx.make_view(fx.add_offset(big_b_mem, slot_b(buf_idx)), b_box)
            )
            ts_b, tg_b = fx.rocdl.cdna5.tdm_partition(
                atom_b, wave, warp_layout, lds_b, blk_b
            )
            fx.copy(atom_b, tg_b[None, fx.Int32(k_tile)], ts_b[None, 0])

        def load_a_at(bases, off):
            return [
                load_wmma_frag(big_a_mem, None, off, ks, rep, bases, lds_k, elem_ty)
                for ks in range_constexpr(k_wmma_steps)
                for rep in range_constexpr(wmma_m_rep)
            ]

        def load_a_frags(buf_idx):
            return load_a_at(a_bases, slot_a(buf_idx))

        def load_b_frags(buf_idx):
            off = slot_b(buf_idx)
            return [
                load_wmma_frag(big_b_mem, None, off, ks, rep, b_bases, lds_k, elem_ty)
                for ks in range_constexpr(k_wmma_steps)
                for rep in range_constexpr(wmma_n_rep)
            ]

        wmma_atom = fx.make_mma_atom(
            fx.rocdl.WMMA(WMMA_M, WMMA_N, WMMA_K, elem_ty, fx.Float32)
        )

        def wmma_tile(accs_in, a_frags, b_frags, rotate=None):
            """WMMA one K tile; ``rotate=(a_bases, a_off, b_off)`` interleaves the
            next tile's LDS fragment loads (compute_bound)."""
            a_ts = [rmem_vec(f, 16, elem_ty) for f in a_frags]
            b_ts = [rmem_vec(f, 16, elem_ty) for f in b_frags]
            acc_ts = [rmem_vec(a, 8, fx.Float32) for a in accs_in]
            rotate_on = rotate is not None
            next_a = [None] * N_A_FRAGS if rotate_on else None
            next_b = [None] * N_B_FRAGS if rotate_on else None
            if const_expr(rotate_on):
                rot_a_bases, a_off, b_off = rotate
            for ks in range_constexpr(k_wmma_steps):
                for wm in range_constexpr(wmma_m_rep):
                    a_f = a_ts[ks * wmma_m_rep + wm]
                    for wn in range_constexpr(wmma_n_rep):
                        idx = wm * wmma_n_rep + wn
                        fx.gemm(
                            wmma_atom,
                            acc_ts[idx],
                            b_ts[ks * wmma_n_rep + wn],
                            a_f,
                            acc_ts[idx],
                        )
                    if const_expr(rotate_on):
                        rocdl.sched_barrier(_SCHED_ALLOW_SALU)
                        next_a[ks * wmma_m_rep + wm] = load_wmma_frag(
                            big_a_mem, None, a_off, ks, wm, rot_a_bases, lds_k, elem_ty
                        )
                if const_expr(rotate_on):
                    rocdl.sched_barrier(_SCHED_ALLOW_SALU)
                    for wn in range_constexpr(wmma_n_rep):
                        next_b[ks * wmma_n_rep + wn] = load_wmma_frag(
                            big_b_mem, None, b_off, ks, wn, b_bases, lds_k, elem_ty
                        )
            return [t.load() for t in acc_ts], next_a, next_b

        def wmma_tile_b_prefetch(accs_in, a_frags, b_frags, nxt_a_bases, a_off, b_off):
            """WMMA one K tile (b_prefetch); returns the next tile's fragments too.

            The next tile's B goes out before the first WMMA, into registers of
            its own. Its A is rotated in place, A_ROT_LAG WMMA groups behind the
            group that last read the register, so the load does not drain the
            WMMAs still reading it (s_wait_alu depctr_va_vdst(0)).
            """
            next_b = [
                load_wmma_frag(big_b_mem, None, b_off, ks, wn, b_bases, lds_k, elem_ty)
                for ks in range_constexpr(k_wmma_steps)
                for wn in range_constexpr(wmma_n_rep)
            ]
            rocdl.sched_barrier(0)
            a_ts = [rmem_vec(f, 16, elem_ty) for f in a_frags]
            b_ts = [rmem_vec(f, 16, elem_ty) for f in b_frags]
            acc_ts = [rmem_vec(a, 8, fx.Float32) for a in accs_in]
            next_a = [None] * N_A_FRAGS
            pending = []

            def load_a(ks, wm):
                next_a[ks * wmma_m_rep + wm] = load_wmma_frag(
                    big_a_mem, None, a_off, ks, wm, nxt_a_bases, lds_k, elem_ty
                )

            for ks in range_constexpr(k_wmma_steps):
                for wm in range_constexpr(wmma_m_rep):
                    a_f = a_ts[ks * wmma_m_rep + wm]
                    for wn in range_constexpr(wmma_n_rep):
                        idx = wm * wmma_n_rep + wn
                        fx.gemm(
                            wmma_atom,
                            acc_ts[idx],
                            b_ts[ks * wmma_n_rep + wn],
                            a_f,
                            acc_ts[idx],
                        )
                    pending.append((ks, wm))
                    if const_expr(len(pending) > A_ROT_LAG):
                        rocdl.sched_barrier(_SCHED_ALLOW_SALU)
                        load_a(*pending.pop(0))
                        rocdl.sched_barrier(_SCHED_ALLOW_SALU)
            rocdl.sched_barrier(_SCHED_ALLOW_SALU)
            for ks, wm in pending:
                load_a(ks, wm)
            return [t.load() for t in acc_ts], next_a, next_b

        store_accs = _make_store_accs(
            y,
            bias,
            warp_m_base,
            warp_n_base,
            lane16,
            lane_kgrp,
            n_idx,
            ot,
            oh0,
            ow0,
            n_offset,
            n_local,
            do_v,
            ho_v,
            wo_v,
        )

        rocdl.disable_xdl_arb_stall()
        return SimpleNamespace(
            a_base=a_base,
            a_bases=a_bases,
            big_a_mem=big_a_mem,
            col_stride=col_stride,
            dims=(Hp, Wp, Cp),
            issue_b=issue_b,
            issue_tdm=issue_tdm,
            k_off=k_off,
            lane16=lane16,
            lane_kgrp=lane_kgrp,
            load_a_at=load_a_at,
            load_a_frags=load_a_frags,
            load_b_frags=load_b_frags,
            origin=(oh0, ow0),
            row_stride=row_stride,
            slot_a=slot_a,
            slot_b=slot_b,
            store_accs=store_accs,
            warp_layout=warp_layout,
            warp_m_base=warp_m_base,
            wave=wave,
            wmma_tile=wmma_tile,
            wmma_tile_b_prefetch=wmma_tile_b_prefetch,
        )

    def _pack_state(accs_, a_, b_):
        return list(accs_) + list(a_) + list(b_)

    def _unpack_state(state):
        out, i = [], 0
        for n in (N_ACCS, N_A_FRAGS, N_B_FRAGS):
            out.append(list(state[i : i + n]))
            i += n
        return out

    # @flyc.jit so the AST rewriter lowers the yield-carrying main loop to scf.for;
    # a plain helper with `yield` is a Python generator and traces nothing.
    @flyc.jit
    def _body(y, x, weight, bias, dyn=None):
        p = _prelude(y, x, weight, bias, dyn)
        issue_tdm, wmma_tile = p.issue_tdm, p.wmma_tile
        load_a_frags, load_b_frags = p.load_a_frags, p.load_b_frags
        acc_zero = fx.Vector.filled(8, fx.Float32(0.0), fx.Float32)
        accs = [acc_zero] * N_ACCS
        k0 = p.k_off // TILE_K

        def _run_tile(accs_, ca, cb, cidx, lidx):
            # Absolute K-tile index = split-K base + local load index.
            issue_tdm(lidx % num_buffers, k0 + lidx)
            tdm_ops.tensor_wait(
                0 if num_buffers == 2 else (num_buffers - 2) * _TDMS_PER_TILE
            )
            # The next trip's TDM refills the slot this wave's last fragment
            # reads came from, and s_barrier does not wait for LDS reads on
            # gfx1250: land them first, or a fast wave's refill can race a slow
            # wave's reads (seen as sporadic mismatches with tile_k=96).
            rocdl.s_wait_dscnt(0)
            gpu.barrier()
            next_buf_i32 = (cidx + 1) % num_buffers
            if const_expr(variant == "compute_bound"):
                return wmma_tile(
                    accs_,
                    ca,
                    cb,
                    rotate=(p.a_bases, p.slot_a(next_buf_i32), p.slot_b(next_buf_i32)),
                )
            if const_expr(b_prefetch):
                return p.wmma_tile_b_prefetch(
                    accs_,
                    ca,
                    cb,
                    p.a_bases,
                    p.slot_a(next_buf_i32),
                    p.slot_b(next_buf_i32),
                )
            na = load_a_frags(next_buf_i32)
            nb = load_b_frags(next_buf_i32)
            accs_, _, _ = wmma_tile(accs_, ca, cb)
            return accs_, na, nb

        # Prologue: issue B-1 tiles, wait for the oldest, load frags for tile 0.
        for i in range_constexpr(num_buffers - 1):
            issue_tdm(i, k0 + i)
        tdm_ops.tensor_wait((num_buffers - 2) * _TDMS_PER_TILE)
        gpu.barrier()
        cur_a, cur_b = load_a_frags(0), load_b_frags(0)

        main_loop_iters = num_k_tiles - (num_buffers - 1)
        load_idx_init, compute_idx_init = fx.Uint32(num_buffers - 1), fx.Uint32(0)

        TILES_PER_TRIP = 2 if variant == "bandwidth_bound" and main_loop_unroll else 1
        num_trips = main_loop_iters // TILES_PER_TRIP
        load_idx_s, compute_idx_s = load_idx_init, compute_idx_init

        if const_expr(num_trips > 0):
            init_state = _pack_state(accs, cur_a, cur_b) + [
                load_idx_init,
                compute_idx_init,
            ]
            results = init_state
            for trip, state in range(0, num_trips, 1, init=init_state):
                cidx, lidx = fx.Uint32(state[-1]), fx.Uint32(state[-2])
                p_accs, ca, cb = _unpack_state(state[:-2])
                for _sub in range_constexpr(TILES_PER_TRIP):
                    p_accs, ca, cb = _run_tile(p_accs, ca, cb, cidx, lidx)
                    cidx, lidx = cidx + 1, lidx + 1
                results = yield _pack_state(p_accs, ca, cb) + [lidx, cidx]

            accs, cur_a, cur_b = _unpack_state(results[:-2])
            # loop results come back untyped; re-wrap so % stays unsigned (remui)
            load_idx_s = fx.Uint32(results[-2])
            compute_idx_s = fx.Uint32(results[-1])
        else:
            accs = list(accs)

        if const_expr(main_loop_iters % TILES_PER_TRIP == 1):
            accs, cur_a, cur_b = _run_tile(
                accs, cur_a, cur_b, compute_idx_s, load_idx_s
            )

        # Drain the remaining in-flight tiles.
        drain_count_d = (
            num_buffers - 1
            if main_loop_iters > 0
            else min(num_buffers - 1, num_k_tiles)
        )
        drain_base = max(0, main_loop_iters)

        accs = list(accs)
        drain_a, drain_b = cur_a, cur_b
        for j in range_constexpr(drain_count_d):
            tile_idx = drain_base + j
            if const_expr(j < drain_count_d - 1):
                next_tile = tile_idx + 1
                tdm_ops.tensor_wait(
                    max(0, num_k_tiles - 1 - next_tile) * _TDMS_PER_TILE
                )
                gpu.barrier()
                next_a = load_a_frags(next_tile % num_buffers)
                accs, _, _ = wmma_tile(accs, drain_a, drain_b)
                next_b = load_b_frags(next_tile % num_buffers)
                drain_a, drain_b = next_a, next_b
            else:
                accs, _, _ = wmma_tile(accs, drain_a, drain_b)

        if const_expr(num_buffers > 2):
            rocdl.sched_barrier(0)
        if const_expr(num_k_tiles > 0):
            p.store_accs(accs)

    @flyc.jit
    def _body_patch(y, x, weight, bias, dyn=None):
        """kH x kW input reuse: one TDM patch per (kt, cc) group feeds all taps.

        Step s = g * TAPS + t reads tap t of patch g (A stage g % 2) and weight
        K tile ``b_tile(g, t)`` (B stage s % num_buffers). The B ring is the
        GEMM pipeline at step granularity. Patch g + 1 is issued a whole group
        ahead, at step (g, 0) after that step's B: its stage held patch g - 1,
        whose last reads landed before the barrier of step (g - 1, TAPS - 1).
        With num_buffers - 1 < TAPS it sits in the TDM queue ahead of B of step
        (g + 1, 0), so in-order completion makes "B of step s landed" imply
        "patch of step s landed", and each wait counts the patch issued after B
        of the step it waits for. One scf.for trip is one group with its taps
        unrolled, which keeps every wait count a compile-time immediate.
        """
        p = _prelude(y, x, weight, bias, dyn)
        Hp, Wp, Cp = p.dims
        oh0, ow0 = p.origin
        wmma_tile = p.wmma_tile
        lds_k = cfg.lds_k_stride
        hp_left = fx.Int32(fx.Int64(Hp) - oh0)
        wp_left = fx.Int32(fx.Int64(Wp) - ow0)
        # Lane bases inside a patch: output pixel m sits at patch (m // C, m % C).
        # C % 16 == 0, so a WMMA atom's 16 rows share one patch row.
        pa_bases = []
        for rep in range_constexpr(wmma_m_rep):
            m = p.warp_m_base + rep * WMMA_M + p.lane16
            pa_bases.append(((m // C) * PATCH_COLS + m % C) * lds_k + p.lane_kgrp * 8)
        if const_expr(KC == 1):
            p_shape = (PATCH_ROWS, PATCH_COLS, TILE_K)
            p_box_stride = (PATCH_COLS * lds_k, lds_k, 1)
            p_rest = (0, 0, None)
        else:
            p_shape = (PATCH_ROWS, PATCH_COLS, KC, TILE_K)
            p_box_stride = (PATCH_COLS * lds_k, lds_k, LDS_K_CHUNK, 1)
            p_rest = (0, 0, 0, None)
        patch_box = fx.make_layout(
            (p_shape, 1), (p_box_stride, PATCH_ROWS * PATCH_COLS * lds_k)
        )

        def issue_patch(g):
            g64 = fx.Int64(g)
            kt_i = g64 // N_CC
            cc = g64 % N_CC
            grp_elems = kt_i * (param.dt * Hp * Wp * Cp) + cc * TK
            if const_expr(KC == 1):
                gP_layout = fx.make_layout(
                    (hp_left, wp_left, TILE_K), (p.row_stride, p.col_stride, 1)
                )
            else:
                gP_layout = fx.make_layout(
                    (hp_left, wp_left, KC, TILE_K),
                    (p.row_stride, p.col_stride, TILE_K, 1),
                )
            gP = fx.Tensor(
                fx.make_view(
                    fx.add_offset(fx.get_iter(x), p.a_base + grp_elems), gP_layout
                )
            )
            atom_p, coord_p = fx.rocdl.cdna5.make_tiled_tdm_atom(
                fx.rocdl.TensorLoad(),
                gP,
                patch_box,
                p_shape,
                num_warps=num_warps,
            )
            blk_p = fx.zipped_divide(coord_p, p_shape)[None, p_rest]
            lds_p = fx.Tensor(
                fx.make_view(
                    fx.add_offset(p.big_a_mem, fx.Uint64((g % 2) * PATCH_ELEMS)),
                    patch_box,
                )
            )
            ts_p, tg_p = fx.rocdl.cdna5.tdm_partition(
                atom_p, p.wave, p.warp_layout, lds_p, blk_p
            )
            fx.copy(atom_p, tg_p[None, 0], ts_p[None, 0])

        def b_tile(g, t):
            # Packed weights run (kt, kh, kw, c); tap t is (kh, kw) row-major.
            return ((g // N_CC) * TAPS + t) * N_CC + g % N_CC

        def a_off(g, t):
            return fx.Uint64((g % 2) * PATCH_ELEMS) + TAP_OFFS[t]

        def frags(g, t, buf):
            return p.load_a_at(pa_bases, a_off(g, t)), p.load_b_frags(buf)

        def issue_step(g, t):
            p.issue_b((g * TAPS + t) % num_buffers, fx.Int32(b_tile(g, t)))

        def run_step(accs_, ca, cb, patch_g, issue, wait, nxt):
            """``issue``/``nxt`` are (g, t) or None; nxt's fragments are loaded.

            ``patch_g`` (or None) is the group whose patch follows that B."""
            if const_expr(issue is not None):
                issue_step(*issue)
            if const_expr(patch_g is not None):
                issue_patch(patch_g)
            if const_expr(nxt is None):
                accs_, _, _ = wmma_tile(accs_, ca, cb)
                return accs_, None, None
            tdm_ops.tensor_wait(wait)
            # Same LDS-read vs. refill race as the per-tap loop.
            rocdl.s_wait_dscnt(0)
            gpu.barrier()
            ng, nt = nxt
            nbuf = (ng * TAPS + nt) % num_buffers
            if const_expr(variant == "compute_bound"):
                return wmma_tile(
                    accs_, ca, cb, rotate=(pa_bases, a_off(ng, nt), p.slot_b(nbuf))
                )
            if const_expr(b_prefetch):
                return p.wmma_tile_b_prefetch(
                    accs_, ca, cb, pa_bases, a_off(ng, nt), p.slot_b(nbuf)
                )
            na, nb = frags(ng, nt, nbuf)
            accs_, _, _ = wmma_tile(accs_, ca, cb)
            return accs_, na, nb

        def ahead(g, t):
            """The step num_buffers - 1 after (g, t); num_buffers - 1 < TAPS."""
            t2 = t + num_buffers - 1
            return (g, t2) if t2 < TAPS else (g + 1, t2 - TAPS)

        def in_flight(t):
            # TDM issues after B of step s + 1: B of s + 2 .. s + num_buffers - 1,
            # plus patch g + 1 while it still trails B of step s + 1 (it went
            # out at (g, 0), right after B of step g * TAPS + num_buffers - 1).
            return num_buffers - 2 + (1 if t <= num_buffers - 2 else 0)

        acc_zero = fx.Vector.filled(8, fx.Float32(0.0), fx.Float32)
        accs = [acc_zero] * N_ACCS

        # Prologue: patch 0 and steps 0 .. num_buffers - 2 (all in group 0).
        issue_patch(0)
        for i in range_constexpr(num_buffers - 1):
            issue_step(0, i)
        tdm_ops.tensor_wait(num_buffers - 2)
        gpu.barrier()
        cur_a, cur_b = frags(0, 0, 0)

        if const_expr(N_GROUPS > 1):
            init_state = _pack_state(accs, cur_a, cur_b) + [fx.Uint32(0)]
            results = init_state
            for _trip, state in range(0, N_GROUPS - 1, 1, init=init_state):
                g = fx.Uint32(state[-1])
                p_accs, ca, cb = _unpack_state(state[:-1])
                for t in range_constexpr(TAPS):
                    nxt = (g, t + 1) if t + 1 < TAPS else (g + 1, 0)
                    p_accs, ca, cb = run_step(
                        p_accs,
                        ca,
                        cb,
                        g + 1 if t == 0 else None,
                        ahead(g, t),
                        in_flight(t),
                        nxt,
                    )
                results = yield _pack_state(p_accs, ca, cb) + [g + 1]
            accs, cur_a, cur_b = _unpack_state(results[:-1])

        # Last group, unrolled: issue only steps that exist, then drain.
        g_last = N_GROUPS - 1
        for t in range_constexpr(TAPS):
            issue = (g_last, t + num_buffers - 1)
            if const_expr(t + num_buffers - 1 >= TAPS):
                issue = None
            nxt = (g_last, t + 1) if t + 1 < TAPS else None
            wait = max(0, min(t + num_buffers - 1, TAPS - 1) - (t + 2) + 1)
            accs, cur_a, cur_b = run_step(accs, cur_a, cur_b, None, issue, wait, nxt)

        if const_expr(num_buffers > 2):
            rocdl.sched_barrier(0)
        p.store_accs(accs)

    _k_body = _body_patch if use_patch else _body

    if const_expr(param.dyn_hw):

        @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
        def conv3d_implicit_kernel(
            y: fx.Tensor,
            x: fx.Tensor,
            weight: fx.Tensor,
            bias: fx.Tensor,
            d: fx.Int64,
            h: fx.Int64,
            w: fx.Int64,
            do: fx.Int64,
            ho: fx.Int64,
            wo: fx.Int64,
            hw_o: fx.Int64,
            dhw: fx.Int64,
            npq: fx.Int64,
            hb: fx.Int64,
            wb: fx.Int64,
            rcp_wb: fx.Int64,
            rcp_hb: fx.Int64,
            rcp_do: fx.Int64,
            grid_m: fx.Int64,
        ):
            dyn = DynShapeArgs1250(
                d,
                h,
                w,
                do,
                ho,
                wo,
                hw_o,
                dhw,
                npq,
                hb,
                wb,
                rcp_wb,
                rcp_hb,
                rcp_do,
                grid_m,
                unit=dyn_unit,
            )
            _k_body(y, x, weight, bias, dyn)

        @flyc.jit
        def launch(
            y: fx.Tensor,
            x: fx.Tensor,
            weight: fx.Tensor,
            bias: fx.Tensor,
            d: fx.Int64,
            h: fx.Int64,
            w: fx.Int64,
            do: fx.Int64,
            ho: fx.Int64,
            wo: fx.Int64,
            hw_o: fx.Int64,
            dhw: fx.Int64,
            npq: fx.Int64,
            hb: fx.Int64,
            wb: fx.Int64,
            rcp_wb: fx.Int64,
            rcp_hb: fx.Int64,
            rcp_do: fx.Int64,
            grid_m: fx.Int64,
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
                do,
                ho,
                wo,
                hw_o,
                dhw,
                npq,
                hb,
                wb,
                rcp_wb,
                rcp_hb,
                rcp_do,
                grid_m,
            ).launch(
                grid=(grid_m, kernel_grid.grid_y, kernel_grid.splitk),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

    else:

        @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
        def conv3d_implicit_kernel(
            y: fx.Tensor, x: fx.Tensor, weight: fx.Tensor, bias: fx.Tensor
        ):
            _k_body(y, x, weight, bias, None)

        @flyc.jit
        def launch(
            y: fx.Tensor,
            x: fx.Tensor,
            weight: fx.Tensor,
            bias: fx.Tensor,
            stream: fx.Stream = fx.Stream(None),  # noqa: B008
        ):
            conv3d_implicit_kernel(y, x, weight, bias).launch(
                grid=(grid.grid_m, grid.grid_y, grid.splitk),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

    launch.compile_hints["llvm_options"] = {
        "amdgpu-expert-scheduling-mode": True,
        "unroll-threshold": 0,
    }
    if use_patch and num_buffers == 2 and cfg.tile_k > TILE_K:
        # These configs spill VGPRs. AMDGPUMarkLastScratchLoad tags reloads
        # th:TH_LOAD_LU before the post-RA scheduler runs, and that scheduler
        # may move an LU reload above another reload of the same slot; the
        # later reload then reads stale scratch (bad bias offsets -> fault).
        launch.compile_hints["llvm_options"]["disable-post-ra"] = True

    def _launch(y, x, weight, bias, stream=None):
        with CompilationContext.compile_hints(CONV_COMPILE_HINTS):
            if const_expr(param.dyn_hw):
                return launch(
                    y, x, weight, bias, *extra_args, stream=_as_stream(stream)
                )
            return launch(y, x, weight, bias, stream=_as_stream(stream))

    def _compile(y, x, weight, bias, stream=None):
        with CompilationContext.compile_hints(CONV_COMPILE_HINTS):
            if const_expr(param.dyn_hw):
                return flyc.compile(
                    launch, y, x, weight, bias, *extra_args, _as_stream(stream)
                )
            return flyc.compile(launch, y, x, weight, bias, _as_stream(stream))

    _launch.compile = _compile
    _launch.extra_args = extra_args
    return _launch
