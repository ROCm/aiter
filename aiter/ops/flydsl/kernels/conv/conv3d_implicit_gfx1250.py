# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""gfx1250 double-buffered implicit-GEMM conv3d (BF16, TDM im2col + WMMA).

Host pre-pads so the kernel always sees zeros padding. Input is NDHWC with
C/groups aligned to 32. Dispatch lives in ``conv3d_gfx1250.py``.

Supports BIG_OUT (64-bit stores), split-K (fp32 atomic staging), and dyn_hw
(runtime extents via magic reciprocal divisors).
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
    patch_reuse: bool  # unit-stride kH×kW reuse when eligible
    splitk: int
    dyn_hw: bool


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
    patch_reuse=False,
    splitk=1,
    dyn_hw=False,
):
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
        patch_reuse=patch_reuse,
        splitk=max(1, int(splitk)),
        dyn_hw=bool(dyn_hw),
    )


def _patch_reuse_eligible(param):
    return (
        param.patch_reuse
        and param.st == param.sh == param.sw == 1
        and (param.kh > 1 or param.kw > 1)
        and param.groups == 1
        and param.splitk == 1
    )


def _dyn_hw_closure_key_gfx1250(grid, unit, use_patch, out_ndhwc, has_bias, splitk):
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
        use_patch,
        out_ndhwc,
        has_bias,
        splitk,
    )


def _shape_agnostic_key_gfx1250(grid, unit, use_patch, out_ndhwc, has_bias, splitk):
    """Closure key with booleans blanked — integers must not carry D/H/W."""

    def _blank(value):
        if isinstance(value, bool):
            return None
        if isinstance(value, tuple):
            return tuple(_blank(v) for v in value)
        return value

    return tuple(
        _blank(v)
        for v in _dyn_hw_closure_key_gfx1250(
            grid, unit, use_patch, out_ndhwc, has_bias, splitk
        )
    )


def _assert_shape_agnostic_gfx1250(param, cfg, kernel_grid, unit, use_patch):
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
        param.patch_reuse,
        param.splitk,
        True,
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
        kernel_grid, unit, use_patch, param.out_ndhwc, param.has_bias, param.splitk
    )
    theirs = _shape_agnostic_key_gfx1250(
        blank,
        probe_unit,
        use_patch,
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

    use_patch = _patch_reuse_eligible(param)
    R, C = cfg.block_rows, cfg.block_cols
    num_buffers = param.num_buffers
    assert 2 <= num_buffers <= 8

    if use_patch:
        patch_rows = R + (param.kh - 1) * param.dh
        patch_cols = C + (param.kw - 1) * param.dw
        patch_elems = patch_rows * patch_cols * cfg.lds_k_stride + LDS_PAD
        patch_b_tiles = 2 * param.kh * param.kw
        SharedStorage = make_shared_storage_gfx1250(
            fx.BFloat16,
            cfg,
            patch_b_tiles,
            patch_elems=patch_elems,
        )
    else:
        patch_rows = patch_cols = patch_elems = 0
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

    if use_patch:
        PATCH_GROUPS = [(kt_i, cc) for kt_i in range(param.kt) for cc in range(N_CC)]

    if const_expr(param.dyn_hw):
        extra_args = dyn_shape_values_gfx1250(param, geom, grid)
        dyn_unit = unit_divisors_gfx1250(grid, geom)
        kernel_grid = grid._replace(grid_m=0, hb=0, wb=0, do=0, ho=0, wo=0, n=0)
        _assert_shape_agnostic_gfx1250(param, cfg, kernel_grid, dyn_unit, use_patch)
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

    def _body_tap(y, x, weight, bias, dyn=None):
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

        def wmma_tile(accs_in, a_frags, b_frags):
            a_ts = [rmem_vec(f, 16, elem_ty) for f in a_frags]
            b_ts = [rmem_vec(f, 16, elem_ty) for f in b_frags]
            acc_ts = [rmem_vec(a, 8, fx.Float32) for a in accs_in]
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
            return [t.load() for t in acc_ts]

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
        num_k = kernel_grid.tiles_per_split
        k0 = k_off // TILE_K

        if const_expr(num_k > 0):
            issue_tdm(0, k0)
        for k_idx in range_constexpr(num_k):
            buf = k_idx % num_buffers
            if const_expr(k_idx + 1 < num_k):
                issue_tdm((k_idx + 1) % num_buffers, k0 + k_idx + 1)
            tdm_ops.tensor_wait(0)
            gpu.barrier()
            a_frags = load_a_frags(buf)
            b_frags = load_b_frags(buf)
            accs = wmma_tile(accs, a_frags, b_frags)

        if const_expr(num_k > 0):
            store_accs(accs)

    def _body_patch(y, x, weight, bias, dyn=None):
        """Unit-stride kH×kW reuse: one TDM patch per (kt, cc), tap frags from LDS."""
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

        n_idx, ot, oh0, ow0, n_offset, n_local, gi, _k_off = decode_block_2d(
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
        PR, PC = patch_rows, patch_cols

        a_base = (((n_idx * Dp + ot * param.st) * Hp + oh0) * Wp + ow0) * Cp + fx.Int64(
            gi
        ) * cgp
        gA = fx.Tensor(
            fx.make_view(
                fx.add_offset(fx.get_iter(x), a_base),
                fx.make_layout((PR, PC, TILE_K), (Wp * Cp, Cp, 1)),
            )
        )
        h_left = fx.Int32(fx.Int64(Hp) - oh0)
        w_left = fx.Int32(fx.Int64(Wp) - ow0)
        atom_a = fx.rocdl.make_tdm_atom(
            gA,
            [h_left, w_left, None],
            strides=[Wp * Cp, Cp, None],
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

        def a_lane_bases_at_tap(kh_i, kw_i):
            tap_r = kh_i * param.dh
            tap_c = kw_i * param.dw
            bases = []
            for rep in range_constexpr(wmma_m_rep):
                local_m = warp_m_base + rep * WMMA_M + lane16
                pr = local_m // C + tap_r
                pc = local_m % C + tap_c
                bases.append(pr * PC * lds_k + pc * lds_k + lane_kgrp * 8)
            return bases

        b_bases = lane_bases_rowmajor(warp_n_base, wmma_n_rep, lds_k, lane16, lane_kgrp)

        def slot_patch(gs):
            return fx.Uint64(gs * patch_elems)

        def slot_b(buf):
            return fx.Uint64(buf * cfg.lds_b_elems_per_buf)

        def load_patch(g, gs):
            kt_i, cc = PATCH_GROUPS[g]
            imm = fx.Int64((kt_i * param.dt * Hp * Wp * Cp + cc * TILE_K) * BF16_BYTES)
            dst = fx.Tensor(
                fx.make_view(
                    fx.add_offset(big_a_mem, slot_patch(gs)),
                    fx.make_layout((PR, PC, TILE_K), (PC * lds_k, lds_k, 1)),
                )
            )
            fx.copy(atom_a, gA, dst, imm_offset=imm)

        def read_a_tap(gs, kh_i, kw_i):
            bases = a_lane_bases_at_tap(kh_i, kw_i)
            off = slot_patch(gs)
            return [
                load_wmma_frag(big_a_mem, None, off, 0, rep, bases, lds_k, elem_ty)
                for rep in range_constexpr(wmma_m_rep)
            ]

        def read_b(buf):
            off = slot_b(buf)
            return [
                load_wmma_frag(big_b_mem, None, off, 0, rep, b_bases, lds_k, elem_ty)
                for rep in range_constexpr(wmma_n_rep)
            ]

        wmma_atom = fx.make_mma_atom(
            fx.rocdl.WMMA(WMMA_M, WMMA_N, WMMA_K, elem_ty, fx.Float32)
        )

        def wmma_tile(accs_in, a_frags, b_frags):
            a_ts = [rmem_vec(f, 16, elem_ty) for f in a_frags]
            b_ts = [rmem_vec(f, 16, elem_ty) for f in b_frags]
            acc_ts = [rmem_vec(a, 8, fx.Float32) for a in accs_in]
            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    idx = wm * wmma_n_rep + wn
                    fx.gemm(wmma_atom, acc_ts[idx], b_ts[wn], a_ts[wm], acc_ts[idx])
            return [t.load() for t in acc_ts]

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
        KH, KW = param.kh, param.kw

        def load_group(g, gs):
            load_patch(g, gs)
            kt_i, cc = PATCH_GROUPS[g]
            for kh_i in range_constexpr(KH):
                for kw_i in range_constexpr(KW):
                    s_local = kh_i * KW + kw_i
                    k_base = ((kt_i * KH + kh_i) * KW + kw_i) * cgp + cc * TILE_K
                    dst = fx.Tensor(
                        fx.make_view(
                            fx.add_offset(
                                big_b_mem,
                                slot_b(gs * KH * KW + s_local),
                            ),
                            fx.make_layout((cfg.tile_n, TILE_K), (lds_k, 1)),
                        )
                    )
                    fx.copy(
                        atom_b,
                        gB,
                        dst,
                        imm_offset=fx.Int64(k_base * BF16_BYTES),
                    )

        load_group(0, 0)
        for g in range_constexpr(len(PATCH_GROUPS)):
            gs = g % 2
            tdm_ops.tensor_wait(0)
            gpu.barrier()
            pending_a = []
            pending_b = []
            for kh_i in range_constexpr(KH):
                for kw_i in range_constexpr(KW):
                    pending_a.append(read_a_tap(gs, kh_i, kw_i))
                    pending_b.append(read_b(gs * KH * KW + kh_i * KW + kw_i))
            if const_expr(g + 1 < len(PATCH_GROUPS)):
                load_group(g + 1, (g + 1) % 2)
            for j in range_constexpr(KH * KW):
                accs = wmma_tile(accs, pending_a[j], pending_b[j])

        store_accs(accs)

    body = _body_patch if use_patch else _body_tap

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
            body(y, x, weight, bias, dyn)

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
            body(y, x, weight, bias, None)

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
