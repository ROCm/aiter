# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""gfx1250 double-buffered implicit-GEMM conv3d (BF16, TDM im2col + WMMA).

Host pre-pads so the kernel always sees zeros padding. Input is NDHWC with
C/groups aligned to 32. Dispatch lives in ``conv3d_gfx1250.py``.

The software pipeline matches ``compile_gemm_a16w16`` (prologue / scf.for main
loop / drain, bandwidth_bound or compute_bound fragment prefetch). Supports
BIG_OUT (64-bit stores), split-K (fp32 atomic staging), and dyn_hw (runtime
extents via magic reciprocal divisors).
"""

import functools
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
    LDS_PAD,
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
    rmem_vec,
    unit_divisors_gfx1250,
)

_SCHED_ALLOW_SALU = 1 << 2
_TDMS_PER_TILE = 2  # A + B


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

    cfg = make_tile_config_gfx1250(param.tile, param.block_rows)
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

    SharedStorage = make_shared_storage_gfx1250(fx.BFloat16, cfg, num_buffers)

    smem_bytes = dsl_size_of(SharedStorage)
    lds_cap = get_lds_capacity_bytes(gpu_arch)
    assert (
        smem_bytes <= lds_cap
    ), f"SharedStorage {smem_bytes}B exceeds LDS capacity {lds_cap}B on {gpu_arch}"

    BLOCK_THREADS = cfg.block_threads
    num_warps = cfg.num_warps
    wmma_m_rep, wmma_n_rep = cfg.mi_m, cfg.mi_n
    k_wmma_steps = 1
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

    N_CC = cgp // TILE_K
    assert N_CC * param.kt * param.kh * param.kw == grid.k_tiles

    variant = param.variant
    main_loop_unroll = param.main_loop_unroll

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
            y_rsrc = big_st_ptr_ty = None

        apply_bias = const_expr(param.has_bias and not use_splitk)
        if apply_bias:
            bias_div = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(bias), fx.make_layout(1, 1)
            )
            bias_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
            bias_reg = fx.make_rmem_tensor(1, fx.Float32)

        def store_accs(final_accs):
            k_out = param.k
            for wm in range_constexpr(wmma_m_rep):
                local_m = warp_m_base + wm * WMMA_M + lane16
                oh = oh0 + local_m // C
                ow = ow0 + local_m % C
                row_ok = (oh < fx.Int64(ho_v)) & (ow < fx.Int64(wo_v))
                for wn in range_constexpr(wmma_n_rep):
                    col_off0 = warp_n_base + wn * WMMA_N + lane_kgrp * 8
                    col_base = n_offset + col_off0
                    acc = fx.Vector(final_accs[wm * wmma_n_rep + wn])
                    if apply_bias:
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

    # @flyc.jit so the AST rewriter lowers the yield-carrying main loop to scf.for;
    # a plain helper with `yield` is a Python generator and traces nothing.
    @flyc.jit
    def _body(y, x, weight, bias, dyn=None):
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

        a_base = (
            ((n_idx * Dp + ot * param.st) * Hp + oh0 * param.sh) * Wp + ow0 * param.sw
        ) * Cp + fx.Int64(gi) * cgp
        row_stride = param.sh * Wp * Cp
        col_stride = param.sw * Cp
        gA = fx.Tensor(
            fx.make_view(
                fx.add_offset(fx.get_iter(x), a_base),
                fx.make_layout((R, C, TILE_K), (row_stride, col_stride, 1)),
            )
        )
        ho_left = fx.Int32(fx.Int64(ho_v) - oh0)
        wo_left = fx.Int32(fx.Int64(wo_v) - ow0)
        atom_a = fx.rocdl.make_tdm_atom(
            gA,
            [ho_left, wo_left, None],
            strides=[row_stride, col_stride, None],
            num_warps=num_warps,
            pad_interval=TILE_K,
            pad_amount=LDS_PAD,
            early_timeout=True,
        )

        b_base = fx.Int64(n_offset) * CRS
        gB = fx.Tensor(
            fx.make_view(
                fx.add_offset(fx.get_iter(weight), b_base),
                fx.make_layout((cfg.tile_n, TILE_K), (CRS, 1)),
            )
        )
        n_left = fx.Int32(fx.Int64(kernel_grid.kg) - fx.Int64(n_local))
        atom_b = fx.rocdl.make_tdm_atom(
            gB,
            [n_left, None],
            strides=[CRS, None],
            num_warps=num_warps,
            pad_interval=TILE_K,
            pad_amount=LDS_PAD,
            early_timeout=True,
        )

        lds_k = cfg.lds_k_stride
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

        def imm_a(kt_i, kh_i, kw_i, cc):
            elems = (
                (kt_i * param.dt * Hp + kh_i * param.dh) * Wp + kw_i * param.dw
            ) * Cp + cc * TILE_K
            return elems * BF16_BYTES

        def imm_b(kt_i, kh_i, kw_i, cc):
            k_base = ((kt_i * param.kh + kh_i) * param.kw + kw_i) * cgp + cc * TILE_K
            return k_base * BF16_BYTES

        def issue_tdm(buf_idx, abs_k):
            kt_i, kh_i, kw_i, cc = _decode_k(abs_k)
            lds_a = fx.Tensor(
                fx.make_view(
                    fx.add_offset(big_a_mem, slot_a(buf_idx)),
                    fx.make_layout((R, C, TILE_K), (C * lds_k, lds_k, 1)),
                )
            )
            lds_b = fx.Tensor(
                fx.make_view(
                    fx.add_offset(big_b_mem, slot_b(buf_idx)),
                    fx.make_layout((cfg.tile_n, TILE_K), (lds_k, 1)),
                )
            )
            fx.copy(atom_a, gA, lds_a, imm_offset=imm_a(kt_i, kh_i, kw_i, cc))
            fx.copy(atom_b, gB, lds_b, imm_offset=imm_b(kt_i, kh_i, kw_i, cc))

        def load_a_frags(buf_idx):
            off = slot_a(buf_idx)
            return [
                load_wmma_frag(big_a_mem, None, off, ks, rep, a_bases, lds_k, elem_ty)
                for ks in range_constexpr(k_wmma_steps)
                for rep in range_constexpr(wmma_m_rep)
            ]

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

        def wmma_tile(accs_in, a_frags, b_frags, rotate_buf=None):
            a_ts = [rmem_vec(f, 16, elem_ty) for f in a_frags]
            b_ts = [rmem_vec(f, 16, elem_ty) for f in b_frags]
            acc_ts = [rmem_vec(a, 8, fx.Float32) for a in accs_in]
            rotate = rotate_buf is not None
            next_a = [None] * N_A_FRAGS if rotate else None
            next_b = [None] * N_B_FRAGS if rotate else None
            if const_expr(rotate):
                a_off = slot_a(rotate_buf)
                b_off = slot_b(rotate_buf)
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
                    if const_expr(rotate):
                        rocdl.sched_barrier(_SCHED_ALLOW_SALU)
                        next_a[ks * wmma_m_rep + wm] = load_wmma_frag(
                            big_a_mem, None, a_off, ks, wm, a_bases, lds_k, elem_ty
                        )
                if const_expr(rotate):
                    rocdl.sched_barrier(_SCHED_ALLOW_SALU)
                    for wn in range_constexpr(wmma_n_rep):
                        next_b[ks * wmma_n_rep + wn] = load_wmma_frag(
                            big_b_mem, None, b_off, ks, wn, b_bases, lds_k, elem_ty
                        )
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
        acc_zero = fx.Vector.filled(8, fx.Float32(0.0), fx.Float32)
        accs = [acc_zero] * N_ACCS
        k0 = k_off // TILE_K

        def _pack_state(accs_, a_, b_):
            return list(accs_) + list(a_) + list(b_)

        def _unpack_state(state):
            out, i = [], 0
            for n in (N_ACCS, N_A_FRAGS, N_B_FRAGS):
                out.append(list(state[i : i + n]))
                i += n
            return out

        def _run_tile(accs_, ca, cb, cidx, lidx):
            # Absolute K-tile index = split-K base + local load index.
            issue_tdm(lidx % num_buffers, k0 + lidx)
            tdm_ops.tensor_wait(
                0 if num_buffers == 2 else (num_buffers - 2) * _TDMS_PER_TILE
            )
            gpu.barrier()
            next_buf_i32 = (cidx + 1) % num_buffers
            if const_expr(variant == "compute_bound"):
                return wmma_tile(accs_, ca, cb, rotate_buf=next_buf_i32)
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
            store_accs(accs)

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
            _body(y, x, weight, bias, dyn)

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
            _body(y, x, weight, bias, None)

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
