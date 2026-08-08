# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl.compiler.protocol import dsl_size_of
from flydsl.expr import const_expr, gpu, idx2crd, range_constexpr, rocdl, tdm_ops
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import check_smem_capacity

WMMA_M, WMMA_N, WMMA_K = 16, 16, 32
WAVE_SIZE, LDS_PAD_A, LDS_PAD_B = 32, 8, 8
_SCHED_ALLOW_SALU = 1 << 2
KERNARG_PRELOAD_COUNT = 8
SCOPE_DEV = 16


def _byte_off_i32(elem_off, elem_bytes):
    return (fx.Int32(elem_off) * elem_bytes).ir_value()


def _f32(x):
    return fx.Float32(fx.Float32(x).ir_value())


def apply_activation_scalar(val, activation: str):
    if activation == "relu":
        zero = _f32(0.0)
        return (val > zero).select(val, zero)
    one, half = _f32(1.0), _f32(0.5)
    if activation in ("silu", "silu_exp2"):
        return val / (one + fx.exp(_f32(0.0) - val))
    if activation == "gelu":
        return half * val * (one + fx.erf(val * (1.0 / math.sqrt(2.0))))
    if activation == "gelu_tanh":
        two = _f32(2.0)
        inner = math.sqrt(2.0 / math.pi) * (val + 0.044715 * val * val * val)
        tanh_val = one - two / (one + fx.exp(two * inner))
        return half * val * (one + tanh_val)
    return val


def compile_gemm_a16w16(
    *,
    M: int = 0,
    N: int = 0,
    K: int,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 32,
    m_warp: int = 2,
    n_warp: int = 4,
    in_dtype: str = "fp16",
    out_dtype: str | None = None,
    num_buffers: int = 2,
    waves_per_eu: int | None = None,
    activation: str | None = None,
    add_bias: bool = False,
    physical_mk: bool = True,  # True=M-major (row-major X), False=K-major (col-major X)
    physical_kn: bool = False,  # False=N-major (row-major W), True=K-major (W^T)
    kernarg_preload: bool = False,
    split_k: int = 1,
    sched_strategy: str | None = None,
    main_loop_unroll: bool = False,
    variant: str = "bandwidth_bound",
):
    def _req(cond, msg):
        if not cond:
            raise ValueError(msg)

    if out_dtype is None:
        out_dtype = "f16" if in_dtype == "fp16" else "bf16"
    is_f16 = in_dtype == "fp16"

    _req(2 <= num_buffers <= 8, f"num_buffers must be in [2, 8], got {num_buffers}")
    _req(variant in ("bandwidth_bound", "compute_bound"), f"bad variant {variant!r}")
    _req(in_dtype in ("fp16", "bf16"), f"bad in_dtype {in_dtype!r}")
    _req(out_dtype in ("f32", "f16", "bf16"), f"bad out_dtype {out_dtype!r}")
    _req(
        sched_strategy in (None, "max-ilp", "max-memory-clause"),
        f"bad sched_strategy {sched_strategy!r}",
    )
    _req(split_k >= 1, f"split_k must be >= 1, got {split_k}")

    elem_bytes = 2
    elem_bytes_d = 2 if out_dtype in ("f16", "bf16") else 4
    num_warps, block_threads = m_warp * n_warp, m_warp * n_warp * WAVE_SIZE
    warp_tile_m, warp_tile_n = tile_m // m_warp, tile_n // n_warp
    split_k_chunk = K // split_k
    num_k_tiles = split_k_chunk // tile_k

    for _nm, _v, _d in (
        ("K", K, tile_k),
        ("K", K, split_k),
        ("K/split_k", split_k_chunk, tile_k),
        ("tile_k", tile_k, WMMA_K),
        ("tile_m", tile_m, WMMA_M),
        ("tile_n", tile_n, WMMA_N),
        ("warp_tile_m", warp_tile_m, WMMA_M),
        ("warp_tile_n", warp_tile_n, WMMA_N),
    ):
        _req(_v % _d == 0, f"{_nm} ({_v}) must be a multiple of {_d}")

    _req(
        tile_k & (tile_k - 1) == 0, f"tile_k must be a power of 2 for TDM, got {tile_k}"
    )

    if physical_kn:
        _req(N > 0, "N must be > 0 at compile time when physical_kn=True")
        _req(tile_n & (tile_n - 1) == 0, f"tile_n must be a power of 2, got {tile_n}")
    if not physical_mk:
        _req(M > 0, "M must be > 0 at compile time when physical_mk=False")
        _req(tile_m & (tile_m - 1) == 0, f"tile_m must be a power of 2, got {tile_m}")
    _req(
        num_k_tiles >= num_buffers - 1,
        f"num_k_tiles {num_k_tiles} < {num_buffers - 1} (K={K}, tile_k={tile_k})",
    )

    gpu_arch = str(get_hip_arch())
    assert gpu_arch.startswith("gfx1250"), f"Expected gfx1250, got {gpu_arch}"

    k_wmma_steps = tile_k // WMMA_K
    _fx_elem = fx.Float16 if is_f16 else fx.BFloat16
    wmma_m_rep, wmma_n_rep = warp_tile_m // WMMA_M, warp_tile_n // WMMA_N
    N_ACCS = n_accs = wmma_m_rep * wmma_n_rep
    N_A_FRAGS, N_B_FRAGS = wmma_m_rep * k_wmma_steps, wmma_n_rep * k_wmma_steps

    if physical_mk:
        a_tile_shape, a_outer_stride, a_pad_interval = (tile_m, tile_k), K, tile_k
        a_imm_bytes, lds_a_stride = tile_k * elem_bytes, tile_k + LDS_PAD_A
        lds_a_elems = tile_m * lds_a_stride + LDS_PAD_A
    else:
        a_tile_shape, a_outer_stride, a_pad_interval = (tile_k, tile_m), M, tile_m
        a_imm_bytes, lds_a_stride = tile_k * M * elem_bytes, tile_m + LDS_PAD_A
        lds_a_elems = tile_k * lds_a_stride + LDS_PAD_A

    if physical_kn:
        b_tile_shape, b_outer_stride, b_pad_interval = (tile_k, tile_n), N, tile_n
        b_imm_bytes, lds_b_stride = tile_k * N * elem_bytes, tile_n + LDS_PAD_B
        lds_b_elems = tile_k * lds_b_stride + LDS_PAD_B
    else:
        b_tile_shape, b_outer_stride, b_pad_interval = (tile_n, tile_k), K, tile_k
        b_imm_bytes, lds_b_stride = tile_k * elem_bytes, tile_k + LDS_PAD_B
        lds_b_elems = tile_n * lds_b_stride + LDS_PAD_B

    _USE_XT = split_k > 1 and out_dtype == "f32"

    if _USE_XT:

        @fx.struct
        class SharedStorage:
            a: fx.Array[_fx_elem, num_buffers * lds_a_elems, 16]
            b: fx.Array[_fx_elem, num_buffers * lds_b_elems, 16]
            xt: fx.Array[fx.Float32, num_warps * WMMA_M * WMMA_N, 16]

    else:

        @fx.struct
        class SharedStorage:
            a: fx.Array[_fx_elem, num_buffers * lds_a_elems, 16]
            b: fx.Array[_fx_elem, num_buffers * lds_b_elems, 16]

    check_smem_capacity(dsl_size_of(SharedStorage), gpu_arch)

    _TDMS_PER_TILE = 2  # A + B

    @flyc.kernel
    def kernel_gemm_a16w16(
        arg_y: fx.Pointer,
        arg_x: fx.Pointer,
        arg_w: fx.Pointer,
        arg_bias: fx.Pointer,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
    ):
        rocdl.disable_xdl_arb_stall()
        elem_ty = _fx_elem.ir_type

        tx, bx, by = gpu.thread_id("x"), gpu.block_id("x"), gpu.block_id("y")
        blk_m, blk_n = fx.Uint64(bx) * tile_m, fx.Uint64(by) * tile_n
        if const_expr(split_k > 1):
            split_k_base = fx.Uint64(gpu.block_id("z")) * split_k_chunk
        else:
            split_k_base = fx.Uint64(0)

        layout_thr = fx.make_layout(
            (m_warp, n_warp, 2, 16), (n_warp * WAVE_SIZE, WAVE_SIZE, 16, 1)
        )
        thr_coord = idx2crd(tx, layout_thr)
        wave_m_idx, wave_n_idx, lane_kgrp, lane16 = (
            fx.Uint64(fx.get(thr_coord, 0)),
            fx.Uint64(fx.get(thr_coord, 1)),
            fx.Uint64(fx.get(thr_coord, 2)),
            fx.Uint64(fx.get(thr_coord, 3)),
        )

        warp_m_base, warp_n_base = wave_m_idx * warp_tile_m, wave_n_idx * warp_tile_n
        m_idx, n_stride = fx.Uint64(i32_m), fx.Uint64(i32_n)
        y_rsrc = fx.rocdl.get_buffer_rsrc(
            fx.rocdl.make_buffer_ptr(arg_y, m_idx * n_stride * elem_bytes_d)
        )

        lds = fx.SharedAllocator(static=True).allocate(SharedStorage).peek()
        big_a_mem, big_b_mem = lds.a.ptr, lds.b.ptr
        big_a_base_idx = fx.Uint64(fx.ptrtoint(big_a_mem))
        big_b_base_idx = fx.Uint64(fx.ptrtoint(big_b_mem))
        blk_m64, blk_n64 = fx.Int64(blk_m), fx.Int64(blk_n)
        split_k_base64 = fx.Int64(split_k_base)

        def _mk_side(
            ptr, base_off, shape, outer_stride, pad_interval, pad_amount, extents
        ):
            gt = fx.Tensor(
                fx.make_view(
                    fx.add_offset(ptr, base_off),
                    fx.make_layout(shape, (outer_stride, 1)),
                )
            )
            return gt, fx.rocdl.make_tdm_atom(
                gt,
                extents,
                strides=[outer_stride, None],
                num_warps=num_warps,
                pad_interval=pad_interval,
                pad_amount=pad_amount,
            )

        m_oob = fx.Int32(m_idx - blk_m)
        n_oob = fx.Int32(fx.Uint64(N) - blk_n) if const_expr(N > 0) else None
        a_extents = [m_oob, None] if const_expr(physical_mk) else [None, m_oob]
        b_extents = [None, n_oob] if const_expr(physical_kn) else [n_oob, None]

        if const_expr(physical_mk):
            a_base_off = blk_m64 * K + split_k_base64
        else:
            a_base_off = split_k_base64 * M + blk_m64
        gA, atom_a = _mk_side(
            arg_x,
            a_base_off,
            a_tile_shape,
            a_outer_stride,
            a_pad_interval,
            LDS_PAD_A,
            a_extents,
        )
        if const_expr(physical_kn):
            b_base_off = split_k_base64 * N + blk_n64
        else:
            b_base_off = blk_n64 * K + split_k_base64
        gB, atom_b = _mk_side(
            arg_w,
            b_base_off,
            b_tile_shape,
            b_outer_stride,
            b_pad_interval,
            LDS_PAD_B,
            b_extents,
        )

        def _imm64(k_tile, mul):
            if const_expr(isinstance(k_tile, int)):
                return fx.Int64(k_tile * mul)
            return fx.Int64(fx.Int32(k_tile) * mul)  # mul in i32, widen for TDM

        def _slot_off(buf_idx, slot_elems):
            if const_expr(isinstance(buf_idx, int)):
                return fx.Uint64(buf_idx * slot_elems)
            return fx.Uint64(fx.Int32(buf_idx) * slot_elems)  # mul in i32, widen

        def _lane_bases(warp_base, reps, lds_stride, transpose):
            if const_expr(not transpose):
                row_off = (warp_base + lane16) * lds_stride
                k_off = lane_kgrp * 8
                return [
                    row_off + rep * WMMA_M * lds_stride + k_off
                    for rep in range_constexpr(reps)
                ]
            k_off = (lane_kgrp * 8 + lane16 % 8) * lds_stride
            grp_off = (lane16 // 8) * 8
            return [
                k_off + warp_base + rep * WMMA_M + grp_off
                for rep in range_constexpr(reps)
            ]

        # One spec per operand side; A and B differ only in these fields.
        MEM, BASE, BASES, STRIDE, TR, ELEMS, REPS, ATOM, GT, SHAPE, IMM = range(11)
        A_SIDE = (
            big_a_mem,
            big_a_base_idx,
            _lane_bases(warp_m_base, wmma_m_rep, lds_a_stride, not physical_mk),
            lds_a_stride,
            not physical_mk,
            lds_a_elems,
            wmma_m_rep,
            atom_a,
            gA,
            a_tile_shape,
            a_imm_bytes,
        )
        B_SIDE = (
            big_b_mem,
            big_b_base_idx,
            _lane_bases(warp_n_base, wmma_n_rep, lds_b_stride, physical_kn),
            lds_b_stride,
            physical_kn,
            lds_b_elems,
            wmma_n_rep,
            atom_b,
            gB,
            b_tile_shape,
            b_imm_bytes,
        )

        def issue_tdm_loads(buf_idx, k_tile):
            for sd in (A_SIDE, B_SIDE):
                fx.copy(
                    sd[ATOM],
                    sd[GT],
                    fx.Tensor(
                        fx.make_view(
                            fx.add_offset(
                                sd[MEM],
                                _slot_off(buf_idx, sd[ELEMS]),
                            ),
                            fx.make_layout(sd[SHAPE], (sd[STRIDE], 1)),
                        )
                    ),
                    imm_offset=_imm64(k_tile, sd[IMM]),
                )

        def _frag(sd, slot_off, ks, rep):
            vec8_ty = T.vec(8, elem_ty)
            halves = []
            for k_half in range_constexpr(2):
                if const_expr(sd[TR]):
                    off = (
                        sd[BASES][rep]
                        + slot_off
                        + (ks * WMMA_K + k_half * 16) * sd[STRIDE]
                    )
                    addr = fx.Int32(sd[BASE] + off * elem_bytes).ir_value()
                    ptr = llvm_dialect.inttoptr(ir.Type.parse("!llvm.ptr<3>"), addr)
                    halves.append(fx.Vector(rocdl.ds_load_tr16_b128(vec8_ty, ptr)))
                else:
                    off = sd[BASES][rep] + slot_off + (ks * WMMA_K + k_half * 16)
                    halves.append(
                        fx.ptr_load(fx.add_offset(sd[MEM], off), result_type=vec8_ty)
                    )
            return halves[0].shuffle(halves[1], list(range(16)))

        def load_frags(sd, buf_idx):
            off = _slot_off(buf_idx, sd[ELEMS])
            return [
                _frag(sd, off, ks, rep)
                for ks in range_constexpr(k_wmma_steps)
                for rep in range_constexpr(sd[REPS])
            ]

        wmma_atom = fx.make_mma_atom(
            fx.rocdl.WMMA(WMMA_M, WMMA_N, WMMA_K, _fx_elem, fx.Float32)
        )

        def _rmem_vec(v, n, dtype):
            t = fx.make_rmem_tensor(n, dtype)
            t.store(v if isinstance(v, fx.Vector) else fx.Vector(v))
            return t

        def wmma_tile(accs_in, a_frags, b_frags, rotate_buf=None):
            a_ts = [_rmem_vec(f, 16, _fx_elem) for f in a_frags]
            b_ts = [_rmem_vec(f, 16, _fx_elem) for f in b_frags]
            acc_ts = [_rmem_vec(a, 8, fx.Float32) for a in accs_in]
            rotate = rotate_buf is not None
            next_a = [None] * N_A_FRAGS if rotate else None
            next_b = [None] * N_B_FRAGS if rotate else None
            if const_expr(rotate):
                a_off = _slot_off(rotate_buf, lds_a_elems)
                b_off = _slot_off(rotate_buf, lds_b_elems)
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
                        next_a[ks * wmma_m_rep + wm] = _frag(A_SIDE, a_off, ks, wm)
                if const_expr(rotate):
                    rocdl.sched_barrier(_SCHED_ALLOW_SALU)
                    for wn in range_constexpr(wmma_n_rep):
                        next_b[ks * wmma_n_rep + wn] = _frag(B_SIDE, b_off, ks, wn)
            return [t.load() for t in acc_ts], next_a, next_b

        _half_out = out_dtype in ("f16", "bf16")
        _out_num = (
            fx.Float16 if out_dtype == "f16" else fx.BFloat16 if _half_out else None
        )

        if const_expr(add_bias):
            bias_lay = fx.make_layout(4, 1)
            bias_tiles = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(
                    fx.make_view(arg_bias, fx.make_layout(N, 1)),
                    max_size=False,
                    num_records_bytes=N * elem_bytes,
                ),
                bias_lay,
            )
            bias_atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), _fx_elem)

            def load_bias4(tile_idx):
                r = fx.make_rmem_tensor(bias_lay, _fx_elem)
                fx.copy_atom_call(bias_atom, fx.slice(bias_tiles, (None, tile_idx)), r)
                return r.load()

        def epilogue_stores(final_accs):
            zero_off = fx.Int32(0).ir_value()
            if const_expr(split_k > 1):
                zero_i32 = zero_off
                dev_scope = fx.Int32(SCOPE_DEV).ir_value()
            if const_expr(_USE_XT):
                xt = fx.add_offset(
                    lds.xt.ptr,
                    (wave_m_idx * n_warp + wave_n_idx) * (WMMA_M * WMMA_N),
                )
            if const_expr(add_bias):
                bias_vecs = []
                for wn in range_constexpr(wmma_n_rep):
                    col_tile = (blk_n + warp_n_base + wn * WMMA_N + lane_kgrp * 8) // 4
                    elems = []
                    for half in range_constexpr(2):
                        bv = load_bias4(col_tile + half)
                        for i in range_constexpr(4):
                            elems.append(bv[i].to(fx.Float32))
                    bias_vecs.append(
                        fx.Vector.from_elements(elems, fx.Float32).ir_value()
                    )

            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    acc = final_accs[wm * wmma_n_rep + wn]
                    row = blk_m + warp_m_base + wm * WMMA_M + lane16
                    col_base = blk_n + warp_n_base + wn * WMMA_N + lane_kgrp * 8
                    if const_expr(add_bias):
                        acc = acc + bias_vecs[wn]
                    if const_expr(activation is not None):
                        acc = fx.Vector.from_elements(
                            [
                                apply_activation_scalar(
                                    fx.Vector(acc)[i].ir_value(), activation
                                )
                                for i in range_constexpr(8)
                            ],
                            fx.Float32,
                        ).ir_value()

                    if const_expr(_half_out):
                        h_vec = fx.Vector(acc).to(_out_num)
                        c_off_bytes = (row * n_stride + col_base) * elem_bytes_d
                        if const_expr(split_k > 1):
                            for pair in range_constexpr(4):
                                pair_vec = fx.Vector.from_elements(
                                    [
                                        h_vec[pair * 2].ir_value(),
                                        h_vec[pair * 2 + 1].ir_value(),
                                    ],
                                    _out_num,
                                ).ir_value()
                                rocdl.raw_ptr_buffer_atomic_fadd(
                                    pair_vec,
                                    y_rsrc,
                                    fx.Int32(c_off_bytes + pair * 4).ir_value(),
                                    zero_i32,
                                    dev_scope,
                                )
                        else:
                            rocdl.raw_ptr_buffer_store(
                                h_vec.bitcast(fx.Int32).ir_value(),
                                y_rsrc,
                                fx.Int32(c_off_bytes).ir_value(),
                                zero_off,
                                zero_off,
                            )
                    elif const_expr(split_k > 1):
                        for e in range_constexpr(8):
                            fx.ptr_store(
                                fx.Vector(acc)[e],
                                fx.add_offset(xt, lane16 * WMMA_N + lane_kgrp * 8 + e),
                            )
                        t_base = blk_m + warp_m_base + wm * WMMA_M
                        t_col = blk_n + warp_n_base + wn * WMMA_N + lane16
                        for e in range_constexpr(8):
                            if t_base + 2 * e < m_idx:
                                rocdl.raw_ptr_buffer_atomic_fadd(
                                    fx.ptr_load(
                                        fx.add_offset(
                                            xt, (lane_kgrp + 2 * e) * WMMA_N + lane16
                                        )
                                    ).ir_value(),
                                    y_rsrc,
                                    _byte_off_i32(
                                        (t_base + lane_kgrp + 2 * e) * n_stride + t_col,
                                        4,
                                    ),
                                    zero_i32,
                                    dev_scope,
                                )
                    else:
                        for half in range_constexpr(2):
                            vec4 = fx.Vector.from_elements(
                                [
                                    fx.Vector(acc)[half * 4 + vi].ir_value()
                                    for vi in range_constexpr(4)
                                ],
                                fx.Float32,
                            ).ir_value()
                            col = col_base + half * 4
                            rocdl.raw_ptr_buffer_store(
                                vec4,
                                y_rsrc,
                                _byte_off_i32(row * n_stride + col, 4),
                                zero_off,
                                zero_off,
                            )

        def _pack_state(accs_, a_, b_):
            return list(accs_) + list(a_) + list(b_)

        def _unpack_state(state):
            out, i = [], 0
            for n in (N_ACCS, N_A_FRAGS, N_B_FRAGS):
                out.append(list(state[i : i + n]))
                i += n
            return out

        def _run_tile(accs_, ca, cb, cidx, lidx):
            issue_tdm_loads(lidx % num_buffers, lidx)
            tdm_ops.tensor_wait(
                0 if num_buffers == 2 else (num_buffers - 2) * _TDMS_PER_TILE
            )
            gpu.barrier()
            next_buf_i32 = (cidx + 1) % num_buffers
            if const_expr(variant == "compute_bound"):
                return wmma_tile(accs_, ca, cb, rotate_buf=next_buf_i32)
            na = load_frags(A_SIDE, next_buf_i32)
            nb = load_frags(B_SIDE, next_buf_i32)
            accs_, _, _ = wmma_tile(accs_, ca, cb)
            return accs_, na, nb

        acc_zero = fx.Vector.filled(8, fx.Float32(0.0), fx.Float32)
        accs = [acc_zero] * n_accs

        # Prologue:
        for i in range_constexpr(num_buffers - 1):
            issue_tdm_loads(i, i)
        tdm_ops.tensor_wait((num_buffers - 2) * _TDMS_PER_TILE)
        gpu.barrier()
        cur_a, cur_b = load_frags(A_SIDE, 0), load_frags(B_SIDE, 0)

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

        # Drain
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
                next_a = load_frags(A_SIDE, next_tile % num_buffers)
                accs, _, _ = wmma_tile(accs, drain_a, drain_b)
                next_b = load_frags(B_SIDE, next_tile % num_buffers)
                drain_a, drain_b = next_a, next_b
            else:
                accs, _, _ = wmma_tile(accs, drain_a, drain_b)

        if const_expr(num_buffers > 2):
            rocdl.sched_barrier(0)
        epilogue_stores(accs)

    @flyc.jit
    def launch_gemm_a16w16(
        arg_y: fx.Pointer,
        arg_x: fx.Pointer,
        arg_w: fx.Pointer,
        arg_bias: fx.Pointer,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        stream: fx.Stream,
    ):
        gx = (fx.Uint64(i32_m) + (tile_m - 1)) // tile_m
        gy = (fx.Uint64(i32_n) + (tile_n - 1)) // tile_n

        wpe = int(waves_per_eu) if waves_per_eu is not None else 0
        launcher = kernel_gemm_a16w16(
            arg_y,
            arg_x,
            arg_w,
            arg_bias,
            i32_m,
            i32_n,
            value_attrs={
                "rocdl.flat_work_group_size": f"{block_threads},{block_threads}",
                "rocdl.waves_per_eu": wpe if wpe >= 1 else None,
            },
        )

        launcher.launch(
            grid=(gx, gy, split_k), block=(block_threads, 1, 1), stream=stream
        )

    _llvm_opts = {
        "amdgpu-expert-scheduling-mode": True,
        "unroll-threshold": 0,
        "amdgpu-kernarg-preload": kernarg_preload,
        "amdgpu-kernarg-preload-count": KERNARG_PRELOAD_COUNT,
    }
    if sched_strategy is not None:
        _llvm_opts["amdgpu-sched-strategy"] = sched_strategy
    launch_gemm_a16w16.compile_hints["llvm_options"] = _llvm_opts

    return launch_gemm_a16w16


__all__ = ["compile_gemm_a16w16"]
