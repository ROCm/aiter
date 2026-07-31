# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.compiler.protocol import dsl_size_of
from flydsl.expr import const_expr, gpu, idx2crd, range_constexpr, rocdl, tdm_ops
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import check_smem_capacity

WMMA_M, WMMA_N, WMMA_K = 16, 16, 128
WAVE_SIZE = 32
LDS_PAD_A_BYTES = 16
B_CYCLE_BYTES = 16 * 16
B_KSTEP_BYTES = WMMA_K * 16


def lds_load_b128(lds_i8_ptr, byte_off):
    return fx.ptr_load(fx.add_offset(lds_i8_ptr, byte_off), result_type=T.vec(4, T.i32))


def _vec(v):
    return v if isinstance(v, fx.Vector) else fx.Vector(v)


def _byte_off_i32(elem_off, elem_bytes):
    return (fx.Int32(elem_off) * elem_bytes).ir_value()


def buffer_load_f32(rsrc, elem_off):
    zero = fx.Int32(0).ir_value()
    return rocdl.raw_ptr_buffer_load(
        T.f32, rsrc, _byte_off_i32(elem_off, 4), zero, zero
    )


def store_acc_vec8_to_buffer(acc_vec8, c_rsrc, addr, out_dt=None, byte_addr=False):
    zero = fx.Int32(0).ir_value()
    acc = _vec(acc_vec8)

    def _store(data, a):
        off = fx.Int32(a).ir_value() if byte_addr else _byte_off_i32(a, 4)
        rocdl.raw_ptr_buffer_store(data, c_rsrc, off, zero, zero)

    if out_dt is not None:
        _store(acc.to(out_dt).bitcast(fx.Int32).ir_value(), addr)
    else:
        for half in range(2):
            vals = [acc[half * 4 + vi] for vi in range(4)]
            _store(fx.Vector.from_elements(vals, fx.Float32).ir_value(), addr[half])


def compile_gemm_a8w8_blockscale(
    *,
    K: int,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    m_warp: int = 2,
    n_warp: int = 4,
    scale_block_k: int = 128,
    scale_block_n: int = 128,
    num_buffers: int = 2,
    waves_per_eu: int | None = None,
    out_dtype: str = "bf16",
    variant: str = "compute_bound",
    kernarg_preload: bool = False,
    split_k: int = 1,
):
    num_warps, block_threads = m_warp * n_warp, m_warp * n_warp * WAVE_SIZE
    warp_tile_m, warp_tile_n = tile_m // m_warp, tile_n // n_warp

    def _req(cond, msg):
        if not cond:
            raise ValueError(msg)

    def _req_multiples(pairs):
        for name, val, div in pairs:
            _req(val % div == 0, f"{name} ({val}) must be a multiple of {div}")

    _req(variant in ("compute_bound", "memory_bound"), f"bad variant {variant!r}")
    _req(out_dtype in ("bf16", "fp16", "f32"), f"bad out_dtype {out_dtype!r}")
    _req(2 <= num_buffers <= 8, f"num_buffers must be in [2, 8], got {num_buffers}")
    _req(split_k >= 1, f"split_k must be >= 1, got {split_k}")
    _req(warp_tile_n <= scale_block_n, f"warp_tile_n {warp_tile_n} > {scale_block_n}")
    _req_multiples(
        (
            ("tile_m", tile_m, WMMA_M),
            ("tile_n", tile_n, WMMA_N),
            ("tile_k", tile_k, WMMA_K),
            ("warp_tile_m", warp_tile_m, WMMA_M),
            ("warp_tile_n", warp_tile_n, WMMA_N),
            ("scale_block_k", scale_block_k, WMMA_K),
            ("tile_k", tile_k, scale_block_k),
            ("K", K, tile_k),
            ("K", K, scale_block_k),
        )
    )
    if split_k > 1:
        _req(
            variant == "memory_bound", f"split_k>1 needs memory_bound, got {variant!r}"
        )
        _req(out_dtype == "f32", "split_k>1 requires out_dtype='f32' (f32 atomic-fadd)")
        _req(K % split_k == 0, f"K ({K}) must be divisible by split_k ({split_k})")
        _req_multiples(
            (
                ("K/split_k", K // split_k, tile_k),
                ("K/split_k", K // split_k, scale_block_k),
            )
        )

    wmma_m_rep, wmma_n_rep = warp_tile_m // WMMA_M, warp_tile_n // WMMA_N
    n_accs = wmma_m_rep * wmma_n_rep
    k_wmma_steps, scales_per_tile = tile_k // WMMA_K, tile_k // scale_block_k
    wmma_steps_per_scale = scale_block_k // WMMA_K
    USE_KSTEP = k_wmma_steps > 1
    acc_coords = [
        (wm, wn, wm * wmma_n_rep + wn)
        for wm in range(wmma_m_rep)
        for wn in range(wmma_n_rep)
    ]

    split_k_chunk = K // split_k
    num_k_tiles, scale_k = split_k_chunk // tile_k, K // scale_block_k
    scale_k_per_split = split_k_chunk // scale_block_k

    NUM_W_CHUNKS = (scale_k_per_split + 31) // 32
    USES_W_CHUNK_PREFETCH = NUM_W_CHUNKS > 1

    _req(
        num_k_tiles >= num_buffers - 1, f"num_k_tiles {num_k_tiles} < {num_buffers - 1}"
    )

    gpu_arch = str(get_hip_arch())
    assert gpu_arch.startswith("gfx1250"), f"Expected gfx1250, got {gpu_arch}"

    elem_bytes_d = 2 if out_dtype in ("bf16", "fp16") else 4

    lds_a_stride_bytes, lds_b_stride_bytes = tile_k + LDS_PAD_A_BYTES, tile_k * 16
    lds_a_data_bytes = tile_m * lds_a_stride_bytes
    lds_b_data_bytes = (tile_n // 16) * lds_b_stride_bytes

    xs_row_bytes = scales_per_tile * 4
    lds_x_scale_data_bytes = tile_m * xs_row_bytes

    @fx.struct
    class SharedStorage:
        a: fx.Array[fx.Uint8, num_buffers * lds_a_data_bytes, 16]
        b: fx.Array[fx.Uint8, num_buffers * lds_b_data_bytes, 16]
        xs: fx.Array[fx.Uint8, num_buffers * lds_x_scale_data_bytes, 16]

    check_smem_capacity(dsl_size_of(SharedStorage), gpu_arch)

    _TDMS_PER_TILE = 3  # X + W + X-scale
    PROLOGUE_FILL = num_buffers if variant == "compute_bound" else num_buffers - 1
    MAIN_LOOP_ITERS = num_k_tiles - PROLOGUE_FILL
    DRAIN_COUNT = (
        PROLOGUE_FILL if MAIN_LOOP_ITERS > 0 else min(PROLOGUE_FILL, num_k_tiles)
    )
    REG_PROLOGUE_WAIT = (PROLOGUE_FILL - 1) * _TDMS_PER_TILE
    MAIN_TDM_OUTSTANDING = _TDMS_PER_TILE * (
        PROLOGUE_FILL - 1 if variant == "compute_bound" else max(0, PROLOGUE_FILL - 2)
    )

    READ_CUR = PROLOGUE_FILL < 2
    FRAGS_IN_COMPUTE = USE_KSTEP or READ_CUR

    _w_window_span = max(3, PROLOGUE_FILL) * scales_per_tile
    _req(
        not USES_W_CHUNK_PREFETCH or _w_window_span - 1 <= 32,
        f"2-chunk W-scale window too narrow: max(3, PROLOGUE_FILL)*scales_per_tile="
        f"{_w_window_span} exceeds 33. Reduce tile_k, raise scale_block_k, or lower "
        f"num_buffers.",
    )

    @flyc.kernel
    def kernel_gemm_a8w8_blockscale(
        arg_y: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_x_scale: fx.Tensor,
        arg_w_scale: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
    ):
        tx, bx, by = gpu.thread_id("x"), gpu.block_id("x"), gpu.block_id("y")
        blk_m, blk_n = fx.Uint64(bx) * tile_m, fx.Uint64(by) * tile_n
        if const_expr(split_k > 1):
            bz = fx.Uint64(gpu.block_id("z"))
            split_k_base, split_kb_base = bz * split_k_chunk, bz * scale_k_per_split
        else:
            split_k_base, split_kb_base = fx.Uint64(0), fx.Uint64(0)

        layout_thr = fx.make_layout(
            (m_warp, n_warp, 2, 16), (n_warp * WAVE_SIZE, WAVE_SIZE, 16, 1)
        )
        thr_coord = idx2crd(tx, layout_thr)
        wave_m_idx = fx.Uint64(fx.get(thr_coord, 0))
        wave_n_idx = fx.Uint64(fx.get(thr_coord, 1))
        lane_kgrp, lane16 = (
            fx.Uint64(fx.get(thr_coord, 2)),
            fx.Uint64(fx.get(thr_coord, 3)),
        )
        warp_m_base, warp_n_base = wave_m_idx * warp_tile_m, wave_n_idx * warp_tile_n
        m_idx, n_idx = fx.Uint64(i32_m), fx.Uint64(i32_n)

        y_buf = fx.rocdl.get_buffer_rsrc(
            fx.rocdl.make_buffer_ptr(fx.get_iter(arg_y), m_idx * n_idx * elem_bytes_d)
        )
        n_scale_blocks = (n_idx + (scale_block_n - 1)) // scale_block_n
        w_scale_buf = fx.rocdl.get_buffer_rsrc(
            fx.rocdl.make_buffer_ptr(
                fx.get_iter(arg_w_scale), n_scale_blocks * scale_k * 4
            )
        )

        scale_zero = fx.Float32(0.0)

        lds = fx.SharedAllocator(static=True).allocate(SharedStorage).peek()

        def _slot_byte_off(buf_idx, slot_bytes):
            if const_expr(isinstance(buf_idx, int)):
                return buf_idx * slot_bytes
            return fx.Uint64(buf_idx) * slot_bytes

        def _imm64(k_tile, mul):
            if const_expr(isinstance(k_tile, int)):
                return fx.Int64(k_tile * mul)
            return fx.Int64(fx.Int32(k_tile) * mul)  # mul in i32, widen for TDM

        def _byte_view(tensor, byte_off, shape, stride):
            return fx.Tensor(
                fx.make_view(
                    fx.add_offset(
                        fx.recast_iter(fx.Int8, fx.get_iter(tensor)), byte_off
                    ),
                    fx.make_layout(shape, stride),
                )
            )

        def _issue_tile(buf_idx, k_tile):
            for atom, gt, base, shape, row in _TDM_SPECS:
                fx.copy(
                    atom,
                    gt,
                    fx.Tensor(
                        fx.make_view(
                            fx.add_offset(
                                base, _slot_byte_off(buf_idx, shape[0] * row)
                            ),
                            fx.make_layout(shape, (row, 1)),
                        )
                    ),
                    imm_offset=_imm64(k_tile, shape[1]),
                )

        def ds_read_x_scales(buf_idx):
            slot_elems = lds_x_scale_data_bytes // 4
            if const_expr(isinstance(buf_idx, int)):
                slot_off = fx.Uint64(buf_idx * slot_elems)
            else:
                slot_off = fx.Uint64(buf_idx) * slot_elems
            out = []
            for sc in range_constexpr(scales_per_tile):
                for wm in range_constexpr(wmma_m_rep):
                    row = warp_m_base + wm * WMMA_M + lane16
                    off = slot_off + row * scales_per_tile + sc
                    out.append(fx.ptr_load(fx.add_offset(big_x_scale_mem, off)))
            return out

        def _issue_w_chunk(chunk):
            if const_expr(isinstance(chunk, int)):
                offset = fx.Uint64(chunk * 32)
            else:
                hi = fx.Uint32(NUM_W_CHUNKS - 1)
                offset = fx.Uint64((chunk < hi).select(chunk, hi) * 32)
            idx = wave_n_block * scale_k + lane_id_full + offset
            if const_expr(split_k > 1):
                idx = idx + split_kb_base
            return buffer_load_f32(w_scale_buf, idx)

        def _w_readlane(kb):
            if const_expr(NUM_W_CHUNKS == 1):
                return rocdl.readlane(T.f32, bulk_w_cur, kb.ir_value())
            is_cur = (kb >> 5) == fx.Uint32(cur_chunk_idx)
            chosen = is_cur.select(fx.Float32(bulk_w_cur), fx.Float32(bulk_w_prefetch))
            return rocdl.readlane(T.f32, chosen.ir_value(), (kb & 31).ir_value())

        def _advance_w_chunks(next_compute_idx):
            next_chunk = (next_compute_idx * scales_per_tile) >> 5
            need_advance = next_chunk != fx.Uint32(cur_chunk_idx)
            new_cur = need_advance.select(
                fx.Float32(bulk_w_prefetch), fx.Float32(bulk_w_cur)
            )
            return new_cur, _issue_w_chunk(next_chunk + 1), next_chunk

        def _a_lane_base(wm):
            return (
                (warp_m_base + lane16) * lds_a_stride_bytes
                + wm * WMMA_M * lds_a_stride_bytes
                + k_half_byte_offset
            )

        def _b_lane_base(wn):
            stripe = (warp_n_base + wn * WMMA_N) // WMMA_N * lds_b_stride_bytes
            return stripe + b_k_half_byte_offset + lane16 * 16

        def _load_frag(
            lds_memref, lane_base, ks, cycle_stride_bytes=16, ks_stride_bytes=WMMA_K
        ):
            off = lane_base + ks * ks_stride_bytes
            v = [
                lds_load_b128(lds_memref, off + cycle_stride_bytes * j)
                for j in range_constexpr(4)
            ]
            v01 = v[0].shuffle(v[1], list(range(8)))
            v23 = v[2].shuffle(v[3], list(range(8)))
            return v01.shuffle(v23, list(range(16)))

        def _rmem_vec(v, n, dtype):
            t = fx.make_rmem_tensor(n, dtype)
            t.store(_vec(v))
            return t

        def _rmem_frag_lists(a_frags, b_frags):
            return (
                [_rmem_vec(f, 16, fx.Int32) for f in a_frags],
                [_rmem_vec(f, 16, fx.Int32) for f in b_frags],
            )

        def issue_wmma_step(sc_base, wm, wn, a_frags, b_frags):
            temp_t = _rmem_vec(acc_zero, 8, fx.Float32)
            for ks_inner in range_constexpr(wmma_steps_per_scale):
                ks = sc_base * wmma_steps_per_scale + ks_inner
                a_t = a_frags[ks * wmma_m_rep + wm]
                b_t = b_frags[ks * wmma_n_rep + wn]
                fx.gemm(wmma_atom, temp_t, b_t, a_t, temp_t)
            return temp_t.load()

        def apply_scale(temp, scale, acc):
            return fx.fma(
                temp, fx.Vector.filled(8, fx.Float32(scale), fx.Float32), _vec(acc)
            )

        def _pack_state(accs_, a_, b_, x_, w_, pw_, load_idx, compute_idx):
            state = list(accs_) + list(a_) + list(b_) + list(x_) + list(w_) + list(pw_)
            if const_expr(USES_W_CHUNK_PREFETCH):
                state = state + [bulk_w_cur, bulk_w_prefetch, cur_chunk_idx]
            return state + [load_idx, compute_idx]

        def _unpack_state(state):
            regs, i = [], 0
            for n in REG_STATE_SIZES:
                regs.append(list(state[i : i + n]))
                i += n
            return regs, list(state[i : i + N_W_TAIL]), state[-2], state[-1]

        def issue_w_raw_scales(k_base):
            kb_base = k_base // scale_block_k
            w_raw = []
            for sc in range_constexpr(scales_per_tile):
                w_val = _w_readlane(fx.Uint32(kb_base + sc))
                for wn in range_constexpr(wmma_n_rep):
                    w_raw.append(w_val)
            return w_raw

        def issue_w_raw_scales_masked(future_tile):
            valid = future_tile < num_k_tiles
            safe_tile = valid.select(future_tile, fx.Uint32(0))
            raw_w = issue_w_raw_scales(fx.Uint64(safe_tile) * tile_k)
            return [valid.select(fx.Float32(v), scale_zero) for v in raw_w]

        def _slot_off(buffer_idx, data_bytes):
            if const_expr(isinstance(buffer_idx, int)):
                return fx.Uint64(buffer_idx * data_bytes)
            return fx.Uint64(buffer_idx * data_bytes)  # mul in i32, widen

        def _a_step_frags(slot_off_a, ks):
            return [
                _load_frag(big_a_mem, _a_lane_base(wm) + slot_off_a, ks)
                for wm in range_constexpr(wmma_m_rep)
            ]

        def _b_step_frags(slot_off_b, ks):
            return [
                _load_frag(
                    big_b_mem,
                    _b_lane_base(wn) + slot_off_b,
                    ks,
                    cycle_stride_bytes=B_CYCLE_BYTES,
                    ks_stride_bytes=B_KSTEP_BYTES,
                )
                for wn in range_constexpr(wmma_n_rep)
            ]

        def load_a_frags(buffer_idx):
            slot_off_a = _slot_off(buffer_idx, lds_a_data_bytes)
            return [
                f
                for ks in range_constexpr(k_wmma_steps)
                for f in _a_step_frags(slot_off_a, ks)
            ]

        def load_b_frags(buffer_idx):
            slot_off_b = _slot_off(buffer_idx, lds_b_data_bytes)
            return [
                f
                for ks in range_constexpr(k_wmma_steps)
                for f in _b_step_frags(slot_off_b, ks)
            ]

        def precompute_scales(x_raw, w_raw):
            scales = []
            for sc in range_constexpr(scales_per_tile):
                row = [None] * n_accs
                for wm, wn, idx in acc_coords:
                    row[idx] = fx.Float32(x_raw[sc * wmma_m_rep + wm]) * fx.Float32(
                        w_raw[sc * wmma_n_rep + wn]
                    )
                scales.append(row)
            return scales

        def compute_wmma(
            global_accs, a_frags, b_frags, x_raw, w_raw, scales=None, buf=None
        ):
            if const_expr(buf is None):
                a_ts, b_ts = _rmem_frag_lists(a_frags, b_frags)
            else:
                slot_a = _slot_off(buf, lds_a_data_bytes)
                slot_b = _slot_off(buf, lds_b_data_bytes)
            for sc in range_constexpr(scales_per_tile):
                sc_base = sc
                if const_expr(buf is not None):
                    a_step, b_step = [], []
                    for ki in range_constexpr(wmma_steps_per_scale):
                        ks = sc * wmma_steps_per_scale + ki
                        b_step += _b_step_frags(slot_b, ks)
                        a_step += _a_step_frags(slot_a, ks)
                    a_ts, b_ts = _rmem_frag_lists(a_step, b_step)
                    sc_base = 0
                for wm, wn, idx in acc_coords:
                    temp = issue_wmma_step(sc_base, wm, wn, a_ts, b_ts)
                    if const_expr(scales is None):
                        scale = fx.Float32(x_raw[sc * wmma_m_rep + wm]) * fx.Float32(
                            w_raw[sc * wmma_n_rep + wn]
                        )
                    else:
                        scale = scales[sc][idx]
                    global_accs[idx] = apply_scale(temp, scale, global_accs[idx])
            return global_accs

        big_a_mem, big_b_mem = lds.a.ptr, lds.b.ptr
        blk_m64, blk_n64 = fx.Int64(blk_m), fx.Int64(blk_n)
        split_k_base64 = fx.Int64(split_k_base)
        x_base_bytes = blk_m64 * K + split_k_base64
        w_base_bytes = (blk_n64 // 16) * (K * 16) + split_k_base64 * 16

        gX = _byte_view(arg_x, x_base_bytes, (tile_m, tile_k), (tile_k, 1))
        gW = _byte_view(
            arg_w, w_base_bytes, (tile_n // 16, tile_k * 16), (tile_k * 16, 1)
        )
        atom_x = fx.rocdl.make_tdm_atom(
            gX,
            [None, None],
            strides=[K, None],
            num_warps=num_warps,
            pad_interval=tile_k,
            pad_amount=LDS_PAD_A_BYTES,
        )
        atom_w = fx.rocdl.make_tdm_atom(
            gW,
            [None, None],
            strides=[K * 16, None],
            num_warps=num_warps,
        )

        # X-scale TDM descriptor + LDS staging (hoisted)
        big_x_scale_mem = fx.recast_iter(fx.Float32, lds.xs.ptr)

        xs_base_bytes = blk_m64 * (scale_k * 4) + fx.Int64(split_kb_base) * 4
        gXS = _byte_view(
            arg_x_scale, xs_base_bytes, (tile_m, xs_row_bytes), (xs_row_bytes, 1)
        )
        atom_xs = fx.rocdl.make_tdm_atom(
            gXS,
            [None, None],
            strides=[scale_k * 4, None],
            num_warps=num_warps,
        )

        _TDM_SPECS = (
            (atom_x, gX, big_a_mem, (tile_m, tile_k), lds_a_stride_bytes),
            (atom_w, gW, big_b_mem, (tile_n // 16, tile_k * 16), lds_b_stride_bytes),
            (atom_xs, gXS, lds.xs.ptr, (tile_m, xs_row_bytes), xs_row_bytes),
        )

        wave_n_block = (blk_n + warp_n_base) // scale_block_n
        lane_id_full = lane_kgrp * 16 + lane16
        bulk_w_cur = bulk_w_prefetch = None
        cur_chunk_idx = fx.Uint32(0)
        k_half_byte_offset, b_k_half_byte_offset = lane_kgrp * 64, lane_kgrp * 1024
        acc_zero = fx.Vector.filled(8, fx.Float32(0.0), fx.Float32)

        wmma_atom = fx.make_mma_atom(
            fx.rocdl.WMMA(WMMA_M, WMMA_N, WMMA_K, fx.Float8E4M3FN, fx.Float32)
        )

        N_ACCS = n_accs
        N_A_FRAGS = 0 if FRAGS_IN_COMPUTE else wmma_m_rep * k_wmma_steps
        N_B_FRAGS = 0 if FRAGS_IN_COMPUTE else wmma_n_rep * k_wmma_steps
        N_CUR_X_RAW = 0 if READ_CUR else scales_per_tile * wmma_m_rep
        N_CUR_W_RAW = scales_per_tile * wmma_n_rep
        N_PREFETCH_W = N_CUR_W_RAW
        REG_STATE_SIZES = (
            N_ACCS,
            N_A_FRAGS,
            N_B_FRAGS,
            N_CUR_X_RAW,
            N_CUR_W_RAW,
            N_PREFETCH_W,
        )
        N_W_TAIL = 3 if USES_W_CHUNK_PREFETCH else 0
        zero_w_raw = [scale_zero] * N_CUR_W_RAW

        # PROLOGUE — pre-fill state for main-loop iter 0.
        for i in range_constexpr(PROLOGUE_FILL):
            _issue_tile(i, i)

        bulk_w_cur = _issue_w_chunk(0)
        if const_expr(USES_W_CHUNK_PREFETCH):
            bulk_w_prefetch = _issue_w_chunk(1)
        else:
            bulk_w_prefetch = bulk_w_cur

        tdm_ops.tensor_wait(REG_PROLOGUE_WAIT)
        gpu.barrier()

        cur_b = [] if FRAGS_IN_COMPUTE else load_b_frags(0)
        cur_x_raw = [] if READ_CUR else ds_read_x_scales(0)
        cur_a = [] if FRAGS_IN_COMPUTE else load_a_frags(0)

        cur_w_raw = issue_w_raw_scales(fx.Uint64(0))
        if const_expr(num_k_tiles > 1):
            prefetch_w_raw = issue_w_raw_scales(fx.Uint64(tile_k))
        else:
            prefetch_w_raw = zero_w_raw

        accs = [acc_zero] * n_accs
        load_idx_init, compute_idx_init = fx.Uint32(PROLOGUE_FILL), fx.Uint32(0)

        if const_expr(MAIN_LOOP_ITERS > 0):
            init_state = _pack_state(
                accs,
                cur_a,
                cur_b,
                cur_x_raw,
                cur_w_raw,
                prefetch_w_raw,
                load_idx_init,
                compute_idx_init,
            )

            for tile_step, state in range(0, MAIN_LOOP_ITERS, 1, init=init_state):
                (
                    (cur_accs, cur_a, cur_b, cur_x_raw, cur_w_raw, prefetch_w_raw),
                    _w_tail,
                    cur_load_idx,
                    cur_compute_idx,
                ) = _unpack_state(state)
                if const_expr(USES_W_CHUNK_PREFETCH):
                    bulk_w_cur, bulk_w_prefetch, cur_chunk_idx = _w_tail

                # SSA buf indices for this iteration.
                cur_load_idx = fx.Uint32(cur_load_idx)
                cur_compute_idx = fx.Uint32(cur_compute_idx)
                load_buf_i32 = cur_load_idx % num_buffers
                next_compute_idx = cur_compute_idx + 1
                next_buf_i32 = next_compute_idx % num_buffers
                cur_buf_i32 = (
                    cur_compute_idx % num_buffers if FRAGS_IN_COMPUTE else None
                )

                if const_expr(variant == "compute_bound"):
                    cur_accs = compute_wmma(
                        cur_accs, cur_a, cur_b, cur_x_raw, cur_w_raw, buf=cur_buf_i32
                    )
                    gpu.barrier()
                    _issue_tile(load_buf_i32, cur_load_idx)
                    tdm_ops.tensor_wait(MAIN_TDM_OUTSTANDING)
                    gpu.barrier()
                    next_b = [] if FRAGS_IN_COMPUTE else load_b_frags(next_buf_i32)
                    next_x_raw = ds_read_x_scales(next_buf_i32)
                    next_a = [] if FRAGS_IN_COMPUTE else load_a_frags(next_buf_i32)
                else:
                    tdm_ops.tensor_wait(MAIN_TDM_OUTSTANDING)
                    if const_expr(FRAGS_IN_COMPUTE):
                        gpu.barrier()
                    _issue_tile(load_buf_i32, cur_load_idx)
                    gpu.barrier()
                    if const_expr(READ_CUR):
                        cur_x_raw = ds_read_x_scales(cur_buf_i32)
                        next_x_raw = []
                    else:
                        next_x_raw = ds_read_x_scales(next_buf_i32)
                    scales = precompute_scales(cur_x_raw, cur_w_raw)
                    cur_accs = compute_wmma(
                        cur_accs, cur_a, cur_b, None, None, scales, buf=cur_buf_i32
                    )
                    next_b = [] if FRAGS_IN_COMPUTE else load_b_frags(next_buf_i32)
                    next_a = [] if FRAGS_IN_COMPUTE else load_a_frags(next_buf_i32)

                cur_a = next_a
                cur_b = next_b
                cur_x_raw = next_x_raw
                cur_w_raw = prefetch_w_raw
                prefetch_w_raw = issue_w_raw_scales_masked(cur_compute_idx + 2)

                if const_expr(USES_W_CHUNK_PREFETCH):
                    bulk_w_cur, bulk_w_prefetch, cur_chunk_idx = _advance_w_chunks(
                        next_compute_idx
                    )

                results = yield _pack_state(
                    cur_accs,
                    cur_a,
                    cur_b,
                    cur_x_raw,
                    cur_w_raw,
                    prefetch_w_raw,
                    cur_load_idx + 1,
                    next_compute_idx,
                )

            final_compute_idx = results[-1]
            if const_expr(USES_W_CHUNK_PREFETCH):
                bulk_w_cur, bulk_w_prefetch, cur_chunk_idx = results[-5:-2]
            accs = list(results[:N_ACCS])
        else:
            final_compute_idx = fx.Uint32(0)

        # EPILOGUE — drain DRAIN_COUNT tiles, no new TDMs. Carries only accs +
        # tile idx; frags are reloaded fresh so nothing else crosses the loop.
        drain_init = list(accs) + [final_compute_idx]
        for _drain_i, dstate in range(0, DRAIN_COUNT, 1, init=drain_init):
            accs_d = list(dstate[:N_ACCS])
            cur_dci_d = fx.Uint32(dstate[-1])
            drain_buf_i32 = cur_dci_d % num_buffers

            # cur tile == num_k_tiles-1-_k  ->  _k later tiles still in flight.
            for _k in range_constexpr(DRAIN_COUNT):
                if cur_dci_d == num_k_tiles - 1 - _k:
                    tdm_ops.tensor_wait(_k * _TDMS_PER_TILE)
            gpu.barrier()

            # Reload this tile's operands FRESH (not carried across the loop).
            cur_b_d = [] if FRAGS_IN_COMPUTE else load_b_frags(drain_buf_i32)
            cur_x_d = ds_read_x_scales(drain_buf_i32)
            cur_a_d = [] if FRAGS_IN_COMPUTE else load_a_frags(drain_buf_i32)
            cur_w_d = issue_w_raw_scales_masked(cur_dci_d)
            accs_d = compute_wmma(
                accs_d,
                cur_a_d,
                cur_b_d,
                cur_x_d,
                cur_w_d,
                buf=drain_buf_i32 if FRAGS_IN_COMPUTE else None,
            )
            dresults = yield list(accs_d) + [cur_dci_d + 1]
        accs = list(dresults[:N_ACCS])

        if const_expr(num_buffers > 2):
            rocdl.sched_barrier(0)

        out_dt = (
            fx.BFloat16
            if out_dtype == "bf16"
            else fx.Float16 if out_dtype == "fp16" else None
        )

        if const_expr(split_k > 1):
            zero_i32_s = fx.Int32(0).ir_value()
        for wm, wn, idx in acc_coords:
            row = blk_m + warp_m_base + wm * WMMA_M + lane16
            col_base = blk_n + warp_n_base + wn * WMMA_N + lane_kgrp * 8
            elem_off = row * n_idx + col_base

            if const_expr(split_k > 1):
                for e in range_constexpr(8):
                    rocdl.raw_ptr_buffer_atomic_fadd(
                        _vec(accs[idx])[e].ir_value(),
                        y_buf,
                        _byte_off_i32(elem_off + e, 4),
                        zero_i32_s,
                        zero_i32_s,
                    )
            elif const_expr(out_dt is not None):
                store_acc_vec8_to_buffer(
                    accs[idx],
                    y_buf,
                    elem_off * elem_bytes_d,
                    out_dt=out_dt,
                    byte_addr=True,
                )
            else:
                store_acc_vec8_to_buffer(
                    accs[idx], y_buf, [elem_off + half * 4 for half in range(2)]
                )

    @flyc.jit
    def launch_gemm_a8w8_blockscale(
        arg_y: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_x_scale: fx.Tensor,
        arg_w_scale: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        stream: fx.Stream,
    ):
        gx = (fx.Uint64(i32_m) + (tile_m - 1)) // tile_m
        gy = (fx.Uint64(i32_n) + (tile_n - 1)) // tile_n

        wpe = int(waves_per_eu) if waves_per_eu is not None else 0
        launcher = kernel_gemm_a8w8_blockscale(
            arg_y,
            arg_x,
            arg_w,
            arg_x_scale,
            arg_w_scale,
            i32_m,
            i32_n,
            value_attrs={
                "rocdl.flat_work_group_size": f"{block_threads},{block_threads}",
                "rocdl.waves_per_eu": wpe if wpe >= 1 else None,
            },
        )

        # Mark kernel args as inreg so AMDGPU preloads them into user SGPRs at dispatch.
        if kernarg_preload:
            ctx = CompilationContext.get_current()
            inreg_attr = ir.UnitAttr.get()
            for op in ctx.gpu_module_body.operations:
                if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                    num_args = len(op.regions[0].blocks[0].arguments)
                    per_arg = [
                        ir.DictAttr.get({"llvm.inreg": inreg_attr})
                        for _ in range(num_args)
                    ]
                    op.attributes["arg_attrs"] = ir.ArrayAttr.get(per_arg)

        launcher.launch(
            grid=(gx, gy, split_k), block=(block_threads, 1, 1), stream=stream
        )

    # Commented out until coexec branch is merged.
    launch_gemm_a8w8_blockscale.compile_hints["llvm_options"] = {
        "amdgpu-expert-scheduling-mode": True,
        # "amdgpu-anti-hints-for-va-vdst": True,
        # "amdgpu-enable-static-simulator": True,
        # "amdgpu-static-sim-inline": True,
        # "amdgpu-sched-strategy": "coexec",
        "unroll-threshold": 0,
        # "amdgpu-block-carried-latency": EnumOpt("all"),  # enable per the note above
    }

    return launch_gemm_a8w8_blockscale


__all__ = ["compile_gemm_a8w8_blockscale"]
