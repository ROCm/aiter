# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Preshuffle GEMM (layout API): f16/bf16/fp8/int8, ping-pong scf.for loop with scheduler hints."""

from typing import Optional

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, math, range_constexpr, rocdl, vector
from flydsl.expr.typing import BFloat16, Float8E4M3FN, Float8E4M3FNUZ, Float16, Float32, Int8, Int32, T
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch

# (dsrd_preload, dvmem_preload) per (tile_m, tile_n, tile_k); ported from v1.
_TILE_PRELOAD_TABLE = {
    # (tile_m, tile_n, tile_k): (dsrd_preload, dvmem_preload)
    # ── tile_m = 16 ──
    (16, 64, 256): (2, 2),
    (16, 64, 512): (8, 8),
    (16, 128, 256): (2, 2),
    (16, 128, 512): (2, 2),
    (16, 192, 256): (2, 2),
    (16, 256, 256): (2, 2),
    (16, 256, 512): (2, 2),
    (16, 512, 256): (2, 2),
    # ── tile_m = 32 ──
    (32, 64, 128): (6, 6),
    (32, 64, 256): (6, 6),
    (32, 64, 512): (2, 2),
    (32, 128, 128): (6, 6),
    (32, 128, 256): (6, 6),
    (32, 192, 128): (6, 6),
    (32, 192, 256): (6, 6),
    (32, 256, 128): (6, 6),
    (32, 256, 256): (6, 6),
    # ── tile_m = 48 ──
    (48, 64, 128): (8, 8),
    (48, 64, 256): (2, 2),
    (48, 128, 256): (6, 6),
    (48, 192, 256): (6, 6),
    (48, 256, 256): (6, 6),
    # ── tile_m = 64 ──
    (64, 64, 128): (4, 4),
    (64, 64, 256): (4, 4),
    (64, 128, 128): (8, 8),
    (64, 128, 256): (8, 8),
    (64, 192, 128): (8, 8),
    (64, 192, 256): (8, 8),
    (64, 256, 64): (8, 8),
    (64, 256, 128): (8, 8),
    (64, 256, 256): (8, 8),
    # ── tile_m = 80 ──
    (80, 64, 256): (4, 4),
    (80, 128, 256): (8, 8),
    (80, 192, 256): (8, 8),
    (80, 256, 256): (8, 8),
    # ── tile_m = 96 ──
    (96, 64, 128): (6, 6),
    (96, 64, 256): (6, 6),
    (96, 128, 128): (8, 8),
    (96, 128, 256): (6, 6),
    (96, 192, 128): (8, 8),
    (96, 192, 256): (8, 8),
    (96, 256, 128): (8, 8),
    (96, 256, 256): (8, 8),
    # ── tile_m = 112 ──
    (112, 64, 256): (8, 8),
    (112, 128, 256): (4, 4),
    (112, 192, 256): (8, 8),
    (112, 256, 256): (8, 8),
    # ── tile_m = 128 ──
    (128, 64, 128): (6, 6),
    (128, 64, 256): (8, 8),
    (128, 128, 64): (4, 4),
    (128, 128, 128): (8, 8),
    (128, 128, 256): (4, 4),
    (128, 192, 128): (8, 8),
    (128, 192, 256): (8, 8),
    (128, 256, 128): (6, 6),
    (128, 256, 256): (4, 4),
    # ── tile_m = 160 ──
    (160, 192, 128): (8, 8),
    # ── tile_m = 192 ──
    (192, 64, 128): (6, 6),
    (192, 128, 128): (6, 6),
    # ── tile_m = 224 ──
    (224, 64, 128): (4, 4),
    (224, 128, 128): (6, 6),
    (224, 192, 128): (6, 6),
    # ── tile_m = 256 ──
    (256, 64, 128): (4, 4),
    (256, 128, 128): (6, 6),
    (256, 192, 128): (6, 6),
    (256, 256, 128): (4, 4),
}

_TILE_PRELOAD_DEFAULT = (0, 0)


def _get_preload(tile_m, tile_n, tile_k):
    """Look up (dsrd_preload, dvmem_preload) from the tile table."""
    return _TILE_PRELOAD_TABLE.get((int(tile_m), int(tile_n), int(tile_k)), _TILE_PRELOAD_DEFAULT)


def compile_preshuffle_gemm(
    *,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    in_dtype: str = "fp8",
    out_dtype: str = "bf16",
    epilogue: str = "none",  # "none", "bias", "bias_relu", "bias_silu", "bias_gelu"
    waves_per_eu: Optional[int] = None,
    enable_scheduler: bool = True,
    use_async_copy: bool = False,
):
    """Compile preshuffle GEMM (layout API, fp8/int8/fp16/bf16).

    Signature: fn(C, A, B, scale_a, scale_b, bias, M, N, stream). bias is the fused
    epilogue bias (per-N, out_dtype); unused when epilogue == "none".
    """
    if in_dtype not in ("fp8", "int8", "fp16", "bf16"):
        raise ValueError(f"in_dtype must be fp8/int8/fp16/bf16, got {in_dtype!r}")
    if tile_k <= 0 or K % tile_k != 0:
        raise ValueError(f"tile_k must be a positive divisor of K; got tile_k={tile_k}, K={K}")
    if epilogue not in ("none", "bias", "bias_relu", "bias_silu", "bias_gelu"):
        raise ValueError(f"epilogue must be none/bias/bias_relu/bias_silu/bias_gelu, got {epilogue!r}")
    _has_epilogue = epilogue != "none"
    _has_bias = epilogue in ("bias", "bias_relu", "bias_silu", "bias_gelu")
    _has_relu = epilogue == "bias_relu"
    _has_silu = epilogue == "bias_silu"
    _has_gelu = epilogue == "bias_gelu"

    is_fp8 = in_dtype == "fp8"
    is_int8 = in_dtype == "int8"
    is_f16 = in_dtype == "fp16"
    is_bf16 = in_dtype == "bf16"
    is_f16_or_bf16 = is_f16 or is_bf16
    is_8bit = is_fp8 or is_int8
    elem_bytes = 1 if is_8bit else 2

    # The async gmem->LDS DMA (buffer_load_lds 128b) only lowers for 8-bit inputs.
    if use_async_copy and not is_8bit:
        raise ValueError("use_async_copy is only supported for 8-bit inputs (fp8/int8)")

    gpu_arch = get_rocm_arch()
    is_gfx942 = str(gpu_arch).startswith("gfx942")
    is_gfx950 = str(gpu_arch).startswith("gfx950")
    use_mfma_scale_128 = is_fp8 and is_gfx950 and (tile_k % 128 == 0)
    use_mfma_k32 = is_f16_or_bf16 and is_gfx950

    if is_f16_or_bf16:
        layout_elem = Float16 if is_f16 else BFloat16
    elif is_int8:
        layout_elem = Int8
    else:
        layout_elem = Float8E4M3FN if is_gfx950 else Float8E4M3FNUZ
    out_elem_cls = BFloat16 if out_dtype == "bf16" else Float16

    # Tile geometry (tile_K_perm = K-elements grouped per MMA k-step)
    tile_K_perm = 128 if use_mfma_scale_128 else (64 if is_8bit else 32)
    k_iters = tile_k // tile_K_perm
    num_tiles = K // tile_k
    m_repeat = tile_m // 16
    num_waves = 4
    n_per_wave = tile_n // num_waves
    num_acc_n = n_per_wave // 16
    acc_size = m_repeat * num_acc_n * 4

    total_threads = 256
    a_load_bytes = 16
    bytes_per_thread_a = (tile_m * tile_k * elem_bytes) // total_threads
    num_a_loads = bytes_per_thread_a // a_load_bytes
    num_b_loads = (tile_n * tile_k * elem_bytes) // total_threads // 16
    num_ds_load = (tile_m * tile_k * elem_bytes) // 64 // 16  # A LDS reads per wave
    num_gmem_loads = num_a_loads + num_b_loads
    if is_8bit and is_gfx950:
        dsrd_preload, dvmem_preload = _get_preload(tile_m, tile_n, tile_k)
    else:
        dsrd_preload, dvmem_preload = (0, 0)

    a_lds_elems = tile_m * tile_k

    @fx.struct
    class SharedStorage:
        a0: fx.Array[layout_elem, a_lds_elems, 16]
        a1: fx.Array[layout_elem, a_lds_elems, 16]

    # ── Kernel ────────────────────────────────────────────────────────
    @flyc.kernel
    def kernel_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        arg_bias: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        tiled_mma_arg: fx.TiledMma,
        tiled_copy_g2s: fx.TiledCopy,
    ):
        tid = fx.thread_idx.x
        bid_x, bid_y, _ = fx.block_idx

        if const_expr(use_mfma_scale_128):
            _scale_atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, layout_elem))
            tiled_mma = fx.make_tiled_mma(
                _scale_atom,
                fx.make_layout((1, 4, 1), (0, 1, 0)),
                fx.make_tile(None, None, fx.make_layout((32, 4), (1, 32))),
            )
        else:
            tiled_mma = tiled_mma_arg

        # Bound A (read) and C (store) to the actual M extent so blocks covering
        # rows past M (ragged M) drop their OOB loads/stores at the descriptor
        # instead of faulting / writing past the allocation. B and scales are
        # exact-multiple in N and stay max_size.
        gA = fx.rocdl.make_buffer_tensor(
            arg_a, max_size=False, num_records_bytes=fx.Int64(i32_m) * fx.Int64(K) * fx.Int64(elem_bytes)
        )
        gB = fx.rocdl.make_buffer_tensor(arg_b)
        gC = fx.rocdl.make_buffer_tensor(
            arg_c, max_size=False, num_records_bytes=fx.Int64(i32_m) * fx.Int64(N) * fx.Int64(2)
        )

        tA = fx.flat_divide(gA, fx.make_tile(tile_m, tile_k))[None, None, bid_x, None]
        tB = fx.flat_divide(gB, fx.make_tile(tile_n, tile_k))[None, None, bid_y, None]
        tC = fx.flat_divide(gC, fx.make_tile(tile_m, tile_n))[None, None, bid_x, bid_y]

        # 128b copy atoms (buffer_load_dwordx4 / ds_read_b128)
        buf_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), layout_elem)
        uni_copy = fx.make_copy_atom(fx.UniversalCopy128b(), layout_elem)

        # Per-thread slices
        thr_mma = tiled_mma.thr_slice(tid)
        thr_g2s = tiled_copy_g2s.get_slice(tid)
        thr_s2r = fx.make_tiled_copy_A(buf_copy, tiled_mma).get_slice(tid)
        thr_g2r_B = fx.make_tiled_copy_B(buf_copy, tiled_mma).get_slice(tid)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        swz = fx.SwizzleType.get(4, 4, 3) if const_expr(is_8bit) else fx.SwizzleType.get(3, 3, 3)

        def _make_sA(arr):
            return fx.make_view(
                arr.ptr,
                fx.make_composed_layout(
                    fx.static(swz),
                    fx.make_ordered_layout((tile_m, tile_k), (1, 0)),
                ),
            )

        sA_stages = [_make_sA(lds.a0), _make_sA(lds.a1)]

        # Partitions
        pA_g = thr_g2s.partition_S(tA)
        pA_s_stages = [thr_g2s.partition_D(s) for s in sA_stages]
        pA_s2r_stages = [thr_s2r.partition_S(s) for s in sA_stages]
        pB_g = thr_g2r_B.partition_S(tB)

        # Fragments — 2 separate B fragments (split double buffer for VGPR lifetime)
        frag_copy_A = fx.make_fragment_like(pA_s_stages[0][None, None, None])
        frag_A = thr_mma.make_fragment_A(sA_stages[0])
        frag_B_single_layout = thr_mma.partition_B(tB).layout(None, None, None, 0)
        frag_B_stages = [fx.make_fragment_like(frag_B_single_layout, layout_elem.ir_type) for _ in range(2)]
        frag_C = thr_mma.make_fragment_C(tC)
        frag_A_retile = thr_s2r.retile(frag_A)
        frag_B_retile_stages = [thr_g2r_B.retile(b) for b in frag_B_stages]
        buf_copy_out = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), out_elem_cls)
        thr_r2g_C = fx.make_tiled_copy_C(buf_copy_out, tiled_mma).get_slice(tid)
        pC_g = thr_r2g_C.partition_S(tC)
        frag_C_out = fx.make_fragment_like(frag_C, out_elem_cls.ir_type)
        frag_C_retile = thr_r2g_C.retile(frag_C_out)

        # ── Async gmem->LDS DMA (buffer_load_lds) for the A tile ──
        if const_expr(use_async_copy):
            dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
            # Bound to the real M extent (like the sync gA) so ragged-M blocks DMA-read
            # OOB rows as 0 instead of faulting past the allocation.
            gA_flat = fx.rocdl.make_buffer_tensor(
                fx.Tensor(fx.make_view(fx.get_iter(arg_a), fx.make_layout(65536 * K, 1))),
                max_size=False,
                num_records_bytes=fx.Int64(i32_m) * fx.Int64(K) * fx.Int64(elem_bytes),
            )
            gA_div = fx.logical_divide(gA_flat, fx.make_layout(1, 1))
            sA_i8_ptr = [fx.recast_iter(Int8, lds.a0.ptr), fx.recast_iter(Int8, lds.a1.ptr)]
            bx_m = bid_x * tile_m
            wave_id = tid // 64
            step_bytes = total_threads * a_load_bytes
            wave_stride_bytes = 64 * a_load_bytes
            k_blocks16_dma = (tile_k * elem_bytes) // 16
            elems_per_16b = 16 // elem_bytes

            def dma_a_to_lds(k_tile_val, stage):
                wave_off = rocdl.readfirstlane(fx.Int32.ir_type, wave_id * wave_stride_bytes)
                lds_ptr = fx.add_offset(sA_i8_ptr[stage], wave_off)
                base_k = k_tile_val * tile_k
                for i in range_constexpr(num_a_loads):
                    if const_expr(i > 0):
                        lds_ptr = fx.add_offset(lds_ptr, step_bytes)
                    pos_bytes = i * total_threads * a_load_bytes + tid * a_load_bytes
                    elem_idx = pos_bytes // elem_bytes
                    m = elem_idx // tile_k
                    k = elem_idx % tile_k
                    k_swz = k ^ ((m % k_blocks16_dma) * elems_per_16b)
                    gmem_byte = ((bx_m + m) * K + base_k + k_swz) * elem_bytes
                    dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
                    src = fx.slice(gA_div, (None, fx.Int32(gmem_byte)))
                    fx.copy(dma_atom, src, dst)

        # ── Scheduling hints (ported from old pipeline) ───────────
        def build_scheduler(numer: int, denom: int):
            if const_expr(denom <= 0):
                return []
            if const_expr(numer <= 0):
                return [0] * denom
            out = []
            prev = 0
            for i in range_constexpr(denom):
                cur = ((i + 1) * numer + (denom - 1)) // denom
                out.append(cur - prev)
                prev = cur
            return out

        def hot_loop_scheduler():
            mfma_group = num_acc_n

            if const_expr(is_gfx942):
                mfma_total = (k_iters * 2) * m_repeat * mfma_group
                mfma_per_iter = 2 * mfma_group
                sche_iters = 0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)

                rocdl.sched_dsrd(2)
                rocdl.sched_mfma(1)
                if const_expr(tile_m == 16):
                    rocdl.sched_vmem(1)
                rocdl.sched_mfma(1)
                if const_expr(tile_m == 16):
                    rocdl.sched_vmem(1)

                if const_expr(num_acc_n < 4):
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(1)
                    if const_expr(tile_m == 16):
                        rocdl.sched_vmem(1)
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(1)
                    if const_expr(tile_m == 16):
                        rocdl.sched_vmem(1)
                    rocdl.sched_mfma(1)

                dswr_tail = num_a_loads
                dstr_advance = 2
                if const_expr(dswr_tail > sche_iters):
                    dswr_tail = sche_iters
                dswr_start = max(sche_iters - dswr_tail - dstr_advance, 0)

                for sche_i in range_constexpr(sche_iters):
                    rocdl.sched_vmem(1)
                    rocdl.sched_mfma(mfma_group)
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(mfma_group)
                    if const_expr(sche_i >= dswr_start - 1):
                        rocdl.sched_dswr(1)
            else:
                if const_expr(use_mfma_scale_128):
                    element_k_per_mfma = 128
                else:
                    element_k_per_mfma = 32
                num_mfma_per_tile_k = tile_k // element_k_per_mfma
                mfma_total = num_mfma_per_tile_k * m_repeat * mfma_group
                dswr_tail = num_a_loads
                dstr_advance = 2
                if const_expr(dswr_tail > mfma_total):
                    dswr_tail = mfma_total
                dsrd_preload_eff = min(int(dsrd_preload), num_ds_load)
                dvmem_preload_eff = min(int(dvmem_preload), num_gmem_loads)
                vmem_remaining = num_gmem_loads - dvmem_preload_eff
                dsrd_remaining = num_ds_load - dsrd_preload_eff
                if const_expr(vmem_remaining > 0 and vmem_remaining < mfma_total):
                    vmem_schedule = build_scheduler(vmem_remaining, vmem_remaining) + [0] * (
                        mfma_total - vmem_remaining
                    )
                else:
                    vmem_schedule = build_scheduler(vmem_remaining, mfma_total)
                dsrd_schedule = build_scheduler(dsrd_remaining, mfma_total)
                dswr_start = max(mfma_total - dswr_tail - dstr_advance, 0)
                last_dsrd_mfma_idx = -1
                for sched_idx in range_constexpr(mfma_total):
                    if const_expr(dsrd_schedule[sched_idx]):
                        last_dsrd_mfma_idx = sched_idx
                dswr_start = max(dswr_start, last_dsrd_mfma_idx + 1)
                idx_ds_read = dsrd_preload_eff
                idx_gmem_load = dvmem_preload_eff
                idx_ds_write = 0
                if const_expr(dvmem_preload_eff):
                    rocdl.sched_vmem(dvmem_preload_eff)
                if const_expr(dsrd_preload_eff):
                    rocdl.sched_dsrd(dsrd_preload_eff)
                for mfma_idx in range_constexpr(mfma_total):
                    rocdl.sched_mfma(1)
                    n_dsrd = dsrd_schedule[mfma_idx]
                    if const_expr(n_dsrd and (idx_ds_read < num_ds_load)):
                        if const_expr(idx_ds_read + n_dsrd > num_ds_load):
                            n_dsrd = num_ds_load - idx_ds_read
                        if const_expr(n_dsrd):
                            rocdl.sched_dsrd(n_dsrd)
                            idx_ds_read += n_dsrd
                    n_vmem = vmem_schedule[mfma_idx]
                    if const_expr(n_vmem and (idx_gmem_load < num_gmem_loads)):
                        if const_expr(idx_gmem_load + n_vmem > num_gmem_loads):
                            n_vmem = num_gmem_loads - idx_gmem_load
                        if const_expr(n_vmem):
                            rocdl.sched_vmem(n_vmem)
                            idx_gmem_load += n_vmem
                    if const_expr((not use_async_copy) and (idx_ds_write < dswr_tail) and (mfma_idx >= dswr_start)):
                        rocdl.sched_dswr(1)
                        idx_ds_write += 1
                if const_expr((not use_async_copy) and idx_ds_write < num_a_loads):
                    rocdl.sched_dswr(num_a_loads - idx_ds_write)

            rocdl.sched_barrier(0)

        # ── Pipeline stage (double-buffered B via split fragments) ─
        def mma_kloop(read_stage, cur_frag_B):
            # s2r the A tile then issue the MMAs. K=128/K=32 (1 atom): flat k_iters → ki;
            # K=16 gfx942 (2 atoms): (None, ki).
            for ki in range_constexpr(k_iters):
                fx.copy(uni_copy, pA_s2r_stages[read_stage][None, None, ki], frag_A_retile[None, None, ki])
                k_coord = ki if (use_mfma_scale_128 or use_mfma_k32) else (None, ki)
                fx.gemm(tiled_mma, frag_C, frag_A[None, None, k_coord], cur_frag_B[None, None, k_coord], frag_C)

        def pipeline_stage(read_stage, next_k_val=None, read_next=True):
            write_stage = read_stage ^ 1
            cur_frag_B = frag_B_stages[read_stage]
            do_next = read_next and next_k_val is not None
            if const_expr(use_async_copy):
                if const_expr(do_next):
                    dma_a_to_lds(next_k_val, write_stage)
                    fx.copy(buf_copy, pB_g[None, None, None, next_k_val], frag_B_retile_stages[write_stage])
                mma_kloop(read_stage, cur_frag_B)
                if const_expr(enable_scheduler):
                    hot_loop_scheduler()
                if const_expr(do_next):
                    rocdl.s_waitcnt(num_b_loads)
                gpu.barrier()
                return
            if const_expr(do_next):
                fx.copy(buf_copy, pA_g[None, None, None, next_k_val], frag_copy_A)
                fx.copy(buf_copy, pB_g[None, None, None, next_k_val], frag_B_retile_stages[write_stage])
            mma_kloop(read_stage, cur_frag_B)
            fx.copy(uni_copy, frag_copy_A, pA_s_stages[write_stage][None, None, None])
            if const_expr(enable_scheduler):
                hot_loop_scheduler()
            gpu.barrier()

        # ── Prologue ──────────────────────────────────────────────
        acc_zero = Vec.filled(acc_size, 0, Int32) if const_expr(is_int8) else Vec.filled(acc_size, 0.0, Float32)
        if const_expr(use_async_copy):
            dma_a_to_lds(fx.Int32(0), 0)
            fx.copy(buf_copy, pB_g[None, None, None, 0], frag_B_retile_stages[0])
            frag_C.store(acc_zero)
            rocdl.s_waitcnt(num_b_loads)
            gpu.barrier()
        else:
            fx.copy(buf_copy, pA_g[None, None, None, 0], frag_copy_A)
            fx.copy(buf_copy, pB_g[None, None, None, 0], frag_B_retile_stages[0])
            frag_C.store(acc_zero)
            fx.copy(uni_copy, frag_copy_A, pA_s_stages[0][None, None, None])
            gpu.barrier()
        rocdl.sched_barrier(0)

        # ── Main tile loop (scf.for with ping-pong) ──────────────
        if const_expr(num_tiles == 1):
            pipeline_stage(read_stage=0, read_next=False)
        elif const_expr(num_tiles == 2):
            pipeline_stage(read_stage=0, next_k_val=fx.Int32(1))
            pipeline_stage(read_stage=1, read_next=False)
        else:
            # Ping-pong loop, 2 tiles/iter, acc-only carry; odd num_tiles → 1-stage tail
            is_odd_tiles = (num_tiles % 2) == 1
            loop_end = fx.Index((num_tiles - 1) // 2 if is_odd_tiles else (num_tiles - 2) // 2)
            for iv, state in range(fx.Index(0), loop_end, fx.Index(1), init=[frag_C.load()]):
                frag_C.store(state[0])
                k_base = fx.Int32(iv * 2)
                pipeline_stage(read_stage=0, next_k_val=k_base + fx.Int32(1))
                pipeline_stage(read_stage=1, next_k_val=k_base + fx.Int32(2))
                results = yield [frag_C.load()]
            frag_C.store(results)
            if const_expr(is_odd_tiles):
                pipeline_stage(read_stage=0, read_next=False)
            else:
                pipeline_stage(read_stage=0, next_k_val=fx.Int32(num_tiles - 1))
                pipeline_stage(read_stage=1, read_next=False)

        # ── Epilogue ─────────────────────────────────────────────
        # Fast path: f16/bf16 with no fused epilogue → vectorized truncate + copy.
        # Otherwise (8-bit scaling and/or fused bias+activation) build the output
        # element-wise so per-column bias and the activation can be applied.
        if const_expr(not is_8bit and not _has_epilogue):
            frag_C_out.store(Vec(frag_C.load()).to(out_elem_cls))
            fx.copy(buf_copy_out, frag_C_retile, pC_g)
        else:
            bx_m = gpu.block_id("x") * tile_m
            by_n = gpu.block_id("y") * tile_n
            wave_id = gpu.thread_id("x") // 64
            lane_id = gpu.thread_id("x") % 64
            lane_div_16 = lane_id // 16
            lane_mod_16 = lane_id % 16
            n_tile_base = wave_id * n_per_wave

            if const_expr(is_8bit):
                # FP8/INT8: per-row(scale_a) × per-col(scale_b) scaling applied inline.
                scale_a_buf = fx.rocdl.make_buffer_tensor(arg_scale_a, max_size=True)
                scale_b_buf = fx.rocdl.make_buffer_tensor(arg_scale_b, max_size=True)
                scale_copy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)
                scale_reg_lay = fx.make_layout(1, 1)
                scale_a_div = fx.logical_divide(scale_a_buf, fx.make_layout(1, 1))
                scale_b_div = fx.logical_divide(scale_b_buf, fx.make_layout(1, 1))

                def load_scale(div_tensor, index):
                    r = fx.make_rmem_tensor(scale_reg_lay, fx.Float32)
                    fx.copy_atom_call(scale_copy, fx.slice(div_tensor, (None, fx.Int32(index))), r)
                    return Vec(fx.memref_load_vec(r))[0]

                # Per-column scales (1/N-block) and per-row scales (1/row/thread).
                s_b_vals = [
                    load_scale(scale_b_div, by_n + n_tile_base + ni * 16 + lane_mod_16)
                    for ni in range_constexpr(num_acc_n)
                ]
                s_a_vals = [
                    [load_scale(scale_a_div, bx_m + mi * 16 + lane_div_16 * 4 + ii) for ii in range_constexpr(4)]
                    for mi in range_constexpr(m_repeat)
                ]

            # Per-column bias (out_dtype), one scalar per N-block, shared across rows.
            if const_expr(_has_bias):
                bias_buf = fx.rocdl.make_buffer_tensor(arg_bias, max_size=True)
                bias_copy = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), out_elem_cls)
                bias_div = fx.logical_divide(bias_buf, fx.make_layout(1, 1))

                def load_bias(index):
                    r = fx.make_rmem_tensor(fx.make_layout(1, 1), out_elem_cls)
                    fx.copy_atom_call(bias_copy, fx.slice(bias_div, (None, fx.Int32(index))), r)
                    return fx.Float32(Vec(fx.memref_load_vec(r))[0])

                bias_vals = [load_bias(by_n + n_tile_base + ni * 16 + lane_mod_16) for ni in range_constexpr(num_acc_n)]

            def apply_activation(val_s):
                # ReLU/SiLU/GeLU ported from v1: maximumf for relu; exp+rcp for silu;
                # tanh-approx gelu expanded through a non-positive exponent (no overflow).
                if const_expr(_has_relu):
                    return fx.Float32(val_s).maximumf(fx.Float32(0.0))
                if const_expr(_has_silu):
                    exp_neg = math.exp(val_s * fx.Float32(-1.0))
                    return val_s * (fx.Float32(1.0) / (fx.Float32(1.0) + exp_neg))
                if const_expr(_has_gelu):
                    half_f32 = fx.Float32(0.5)
                    one_f32 = fx.Float32(1.0)
                    zero_f32 = fx.Float32(0.0)
                    two_f32 = fx.Float32(2.0)
                    x3 = val_s * val_s * val_s
                    y = fx.Float32(0.7978845608) * (val_s + fx.Float32(0.044715) * x3)
                    abs_y = fx.Float32(y).maximumf(zero_f32 - y)
                    e_neg2abs = math.exp(fx.Float32(-2.0) * abs_y)
                    denom = one_f32 + e_neg2abs
                    numerator = (y > zero_f32).select(two_f32, two_f32 * e_neg2abs)
                    return half_f32 * val_s * (numerator * (one_f32 / denom))
                return val_s

            acc_vec = Vec(frag_C.load())
            out_elems = []
            for mi in range_constexpr(m_repeat):
                for ni in range_constexpr(num_acc_n):
                    for ii in range_constexpr(4):
                        idx = mi * num_acc_n * 4 + ni * 4 + ii
                        val = acc_vec[idx]
                        if const_expr(is_int8):
                            val = val.to(Float32)
                        if const_expr(is_8bit):
                            val_s = (val * s_a_vals[mi][ii]) * s_b_vals[ni]
                        else:
                            val_s = val
                        if const_expr(_has_bias):
                            val_s = val_s + bias_vals[ni]
                        val_s = apply_activation(val_s)
                        out_elems.append(val_s.to(out_elem_cls))

            out_vec = vector.from_elements(T.vec(acc_size, out_elem_cls.ir_type), out_elems)
            frag_C_out.store(out_vec)
            fx.copy(buf_copy_out, frag_C_retile, pC_g)

    # ── Host launcher ─────────────────────────────────────────────
    @flyc.jit
    def launch_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        arg_bias: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()

        # MMA atom — layout_elem carries the dtype (Float16/BFloat16/Float8E4M3FN/etc)
        if const_expr(use_mfma_k32):
            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, layout_elem))
            k_perm = fx.make_layout((8, 4), (1, 8))
        elif const_expr(is_f16_or_bf16):
            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, layout_elem))
            k_perm = fx.make_layout((4, 4, 2), (1, 8, 4))
        elif const_expr(is_int8):
            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, layout_elem, Int32))
            k_perm = fx.make_layout((8, 4, 2), (1, 16, 8))
        else:
            # fp8: narrow atom here; the scale (16x16x128) tiled_mma is rebuilt in-kernel
            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, layout_elem))
            k_perm = fx.make_layout((8, 4, 2), (1, 16, 8))

        tiled_mma = fx.make_tiled_mma(
            mma_atom,
            fx.make_layout((1, 4, 1), (0, 1, 0)),
            fx.make_tile(None, None, k_perm),
        )

        # G2S tiled copy
        val_per_thr = a_load_bytes // elem_bytes
        thrs_k = tile_k // val_per_thr
        thrs_m = total_threads // thrs_k
        tiled_copy_g2s = fx.make_tiled_copy(
            fx.make_copy_atom(fx.UniversalCopy128b(), layout_elem),
            fx.make_layout(
                ((thrs_k, thrs_m), (1, val_per_thr)),
                ((thrs_m * val_per_thr, 1), (1, thrs_m)),
            ),
            fx.make_tile(thrs_m, tile_k),
        )

        # Preshuffle B layout (2D hierarchical)
        kp_bytes = 16
        kp_elems = kp_bytes if elem_bytes == 1 else kp_bytes // elem_bytes
        k_bytes_b = K * elem_bytes
        n0 = N // 16
        k0 = k_bytes_b // 64
        s_nlane = kp_elems
        s_klane = 16 * s_nlane
        s_k0 = 4 * s_klane
        s_n0 = k0 * s_k0
        preshuffle_B = fx.Tensor(
            fx.make_view(
                fx.get_iter(arg_b),
                fx.make_layout(((16, n0), (kp_elems, 4, k0)), ((s_nlane, s_n0), (1, s_klane, s_k0))),
            )
        )

        # Reshape A and C to 2D
        M_max = 65536
        arg_a_2d = fx.Tensor(fx.make_view(fx.get_iter(arg_a), fx.make_layout((M_max, K), (K, 1))))
        arg_c_2d = fx.Tensor(fx.make_view(fx.get_iter(arg_c), fx.make_layout((M_max, N), (N, 1))))

        gx = (i32_m + (tile_m - 1)) // tile_m
        gy = i32_n // tile_n

        kernel_gemm(
            arg_c_2d,
            arg_a_2d,
            preshuffle_B,
            arg_scale_a,
            arg_scale_b,
            arg_bias,
            i32_m,
            i32_n,
            tiled_mma,
            tiled_copy_g2s,
            value_attrs={"rocdl.waves_per_eu": waves_per_eu},
        ).launch(
            grid=(gx, gy, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    if const_expr(is_f16_or_bf16 and num_acc_n <= 2):
        launch_gemm.compile_hints["llvm_options"] = {"enable-post-misched": False}

    return launch_gemm
