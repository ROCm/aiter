# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Masked Grouped FP8 GEMM kernel (M-grouped masked layout).

API matching DeepGEMM's m_grouped_fp8_gemm_nt_masked:
 - A: [G, expected_m, K] FP8 - padded activation tensor per group
 - scale_a: [G, scale_k, expected_m] FP32 - per-token, per-128K scales (transposed)
 - B: [G, N, K] FP8 - one weight matrix per group
 - scale_b: [G, scale_n, scale_k] FP32 - per-block scales
 - D: [G, expected_m, N] BF16 - padded output tensor per group
 - masked_m: [G] INT32 - tracks the actual number of valid tokens per group
 - expected_m: INT32 - the padded capacity (max_m) for the M dimension

Block scaling granularity (matching DeepGEMM's 1D2D configuration):
 - A: (1, 128) - per-token, per-128-K-elements
 - B: (128, 128) - per-128-N, per-128-K block

Optimizations applied:
 - LDS ping-pong double buffering for A tiles
 - XOR swizzle for LDS bank conflict avoidance
 - Preshuffle B layout with load_b_pack_k32
 - Dynamic block-level early exit using masked_m to skip computing padded garbage
"""

import functools
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, gpu, buffer_ops, vector, rocdl
from flydsl.expr import range_constexpr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl._mlir.dialects import math as math_dialect
from flydsl.expr.typing import T
from flydsl.expr.arith import ArithValue

from .mfma_preshuffle_pipeline import (
    crd2idx,
    lds_store_16b_xor16,
    load_b_pack_k32,
    make_preshuffle_b_layout,
    swizzle_xor16,
    tile_chunk_coord_i32,
)


@functools.lru_cache(maxsize=128)
def compile_masked_grouped_fp8_gemm(
    *,
    n: int,
    k: int,
    num_groups: int,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    scale_block_k: int = 128,
    scale_block_n: int = 128,
    out_dtype: str = "bf16",
):
    """Compile masked grouped FP8 GEMM kernel and return the JIT launcher.

    Args:
        n: N dimension (output columns per group)
        k: K dimension (reduction dimension)
        num_groups: Number of groups (experts)
        tile_m: M tile size (default 128)
        tile_n: N tile size (default 128)
        tile_k: K tile size (default 128)
        scale_block_k: K-dimension scale block size (default 128)
        scale_block_n: N-dimension scale block size (default 128)
        out_dtype: Output data type ("bf16" or "f16")

    Returns:
        JIT launcher function.
    """
    gpu_arch = get_hip_arch()
    _is_gfx950 = str(gpu_arch).startswith("gfx95")

    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem_masked_grouped_gemm")

    # Validate parameters
    if k % tile_k != 0:
        raise ValueError(f"k ({k}) must be divisible by tile_k ({tile_k})")
    if n % tile_n != 0:
        raise ValueError(f"n ({n}) must be divisible by tile_n ({tile_n})")
    if tile_k % scale_block_k != 0:
        raise ValueError(f"tile_k ({tile_k}) must be divisible by scale_block_k ({scale_block_k})")
    if tile_n % scale_block_n != 0:
        raise ValueError(f"tile_n ({tile_n}) must be divisible by scale_block_n ({scale_block_n})")

    # Output type
    if out_dtype not in ("bf16", "f16"):
        raise ValueError(f"out_dtype must be 'bf16' or 'f16', got {out_dtype!r}")
    out_mlir = lambda: T.bf16 if out_dtype == "bf16" else T.f16

    # Compile-time constants
    total_threads = 256
    elem_bytes = 1  # FP8
    num_k_tiles = k // tile_k
    scale_k = k // scale_block_k
    scale_n = n // scale_block_n
    sb_per_tile = tile_k // scale_block_k  # scale blocks per K-tile
    k_unroll = tile_k // 64  # K64-byte micro-steps (for K32 MFMA pairs)
    kpack_bytes = 16  # 16-byte packs for FP8

    # LDS allocation: 2x for ping-pong double buffer
    lds_a_bytes = tile_m * tile_k * elem_bytes
    lds_total_bytes = 2 * lds_a_bytes
    lds_alloc_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_alloc_offset + lds_total_bytes
    lds_tile_elems = tile_m * tile_k  # element offset between ping and pong

    # Module name for caching
    module_name = (
        f"masked_grouped_fp8_gemm_{out_dtype}"
        f"_n{n}_k{k}_g{num_groups}"
        f"_t{tile_m}x{tile_n}x{tile_k}"
        f"_pingpong"
    ).replace("-", "_")

    # Thread -> tile element mapping for A loads
    tile_k_bytes = tile_k * elem_bytes
    tile_k_dwords = tile_k_bytes // 4
    bytes_a_per_tile = tile_m * tile_k * elem_bytes
    bytes_per_thread_a = bytes_a_per_tile // total_threads
    a_load_bytes = 16  # 16-byte loads (dwordx4)
    chunk_i32_a = a_load_bytes // 4  # 4 dwords per load
    num_a_loads = bytes_per_thread_a // a_load_bytes

    @flyc.kernel(name=module_name)
    def masked_grouped_fp8_gemm_kernel(
        arg_d: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        arg_masked_m: fx.Tensor,
        i32_expected_m: fx.Int32,
        i32_n: fx.Int32,
        i32_k: fx.Int32,
        i32_num_groups: fx.Int32,
    ):
        # Convert runtime parameters to index type
        # In the masked kernel, expected_m acts as our padded max capacity per group.
        m_in = arith.index_cast(T.index, i32_expected_m)
        n_in = arith.index_cast(T.index, i32_n)
        k_in = arith.index_cast(T.index, i32_k)
        num_groups_in = arith.index_cast(T.index, i32_num_groups)

        # Thread and 3D block IDs
        tx = gpu.thread_id("x")
        by = gpu.block_id("x")  # N-block index
        bx = gpu.block_id("y")  # M-block index
        bz = gpu.block_id("z")  # Group ID index
        group_idx = bz

        # Block positions
        bx_m = bx * fx.Index(tile_m)
        by_n = by * fx.Index(tile_n)

        # Wave/lane decomposition (256 threads = 4 waves x 64 lanes)
        layout_wave_lane = fx.make_layout((4, 64), stride=(64, 1))
        coord_wave_lane = fx.idx2crd(tx, layout_wave_lane)
        wave_id = fx.get(coord_wave_lane, 0)
        lane_id = fx.get(coord_wave_lane, 1)

        # Lane decomposition for MFMA (lane_id -> lane_div_16, lane_mod_16)
        layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))
        coord_lane16 = fx.idx2crd(lane_id, layout_lane16)
        lane_div_16 = fx.get(coord_lane16, 0)
        lane_mod_16 = fx.get(coord_lane16, 1)

        # LDS setup: single memref for both ping-pong buffers
        base_ptr = allocator.get_base()
        lds_a = SmemPtr(base_ptr, lds_alloc_offset, T.f8, shape=(2 * tile_m * tile_k,)).get()
        lds_stride = tile_k
        layout_lds = fx.make_layout((tile_m, tile_k), stride=(lds_stride, 1))
        lds_base_pong = fx.Index(0)
        lds_base_ping = fx.Index(lds_tile_elems)

        # Buffer resources
        a_nbytes = num_groups_in * m_in * k_in
        a_rsrc = buffer_ops.create_buffer_resource(
            arg_a, max_size=False, num_records_bytes=arith.index_cast(T.i64, a_nbytes)
        )

        b_nbytes = num_groups_in * n_in * k_in
        b_rsrc = buffer_ops.create_buffer_resource(
            arg_b, max_size=False, num_records_bytes=arith.index_cast(T.i64, b_nbytes)
        )

        d_nbytes = num_groups_in * m_in * n_in * fx.Index(2)  # bf16/f16 = 2 bytes
        d_rsrc = buffer_ops.create_buffer_resource(
            arg_d, max_size=False, num_records_bytes=arith.index_cast(T.i64, d_nbytes)
        )

        # Scale buffers
        # scale_a: [G, scale_k, max_m]
        sa_nbytes = num_groups_in * fx.Index(scale_k) * m_in * fx.Index(4)
        sa_rsrc = buffer_ops.create_buffer_resource(
            arg_scale_a, max_size=False, num_records_bytes=arith.index_cast(T.i64, sa_nbytes)
        )

        # scale_b: [G, scale_n, scale_k]
        sb_nbytes = num_groups_in * fx.Index(scale_n * scale_k * 4)
        sb_rsrc = buffer_ops.create_buffer_resource(
            arg_scale_b, max_size=False, num_records_bytes=arith.index_cast(T.i64, sb_nbytes)
        )

        # masked_m: [G]
        mask_nbytes = num_groups_in * fx.Index(4)
        mask_rsrc = buffer_ops.create_buffer_resource(
            arg_masked_m, max_size=False, num_records_bytes=arith.index_cast(T.i64, mask_nbytes)
        )

        # Early exit for invalid blocks that fall entirely within the padded garbage
        bx_m_i32 = arith.index_cast(T.i32, bx_m)
        valid_m_i32 = buffer_ops.buffer_load(mask_rsrc, group_idx, vec_width=1, dtype=T.i32)
        is_valid = arith.cmpi(arith.CmpIPredicate.ult, bx_m_i32, valid_m_i32)

        _if_valid = scf.IfOp(is_valid)
        with ir.InsertionPoint(_if_valid.then_block):

            # MFMA tiling constants
            m_repeat = tile_m // 16  # 8 for tile_m=128
            num_waves = 4
            n_per_wave = tile_n // num_waves  # 32 for tile_n=128
            num_acc_n = n_per_wave // 16  # 2 for n_per_wave=32

            # Initialize accumulators (FP32)
            acc_init = arith.constant_vector(0.0, T.f32x4)
            num_accs = m_repeat * num_acc_n
            accs = [acc_init] * num_accs

            # Wave's N-tile base
            wave_mod_4 = wave_id % fx.Index(4)
            n_tile_base = wave_mod_4 * fx.Index(n_per_wave)

            # Precompute N-block indices for scale_b
            c_scale_block_n = fx.Index(scale_block_n)
            c_scale_k = fx.Index(scale_k)
            n_block_for_scale = []
            for ni in range_constexpr(num_acc_n):
                col_base = by_n + n_tile_base + arith.index(ni * 16)
                n_blk = col_base // c_scale_block_n
                n_block_for_scale.append(n_blk)

            # B preshuffle layout: total N = num_groups * N (all groups concatenated)
            c_n_total = num_groups_in * n_in
            b_layout = make_preshuffle_b_layout(
                arith, c_n=c_n_total, c_k=k_in,
                kpack_bytes=kpack_bytes, elem_bytes=elem_bytes,
            )
            layout_b = b_layout.layout_b

            # Decompose global N column into (n_blk, n_intra) for preshuffle layout
            c_n0 = c_n_total // fx.Index(16)
            c_n0_i32 = arith.index_cast(T.i32, c_n0)
            layout_n_blk_intra = fx.make_layout((c_n0_i32, 16), stride=(16, 1))
            n_blk_list = []
            n_intra_list = []
            group_n_off = group_idx * n_in  # N-offset for this group in concatenated B
            for ni in range_constexpr(num_acc_n):
                col_global = group_n_off + by_n + n_tile_base + arith.index(ni * 16) + lane_mod_16
                coord_ni = fx.idx2crd(col_global, layout_n_blk_intra)
                n_blk_list.append(fx.get(coord_ni, 0))
                n_intra_list.append(fx.get(coord_ni, 1))

            # A load mapping: thread -> (row, col_i32) in tile (dword-indexed K)
            layout_a_tile_div4 = fx.make_layout(
                (tile_m, tile_k_dwords), stride=(tile_k_dwords, 1)
            )
            c_chunk_a = fx.Index(chunk_i32_a)
            tx_i32_base = tx * c_chunk_a
            _k_div4_factor = k_in // fx.Index(4)
            group_a_off_div4 = group_idx * m_in * _k_div4_factor  # 3D A Offset
            k_blocks16 = arith.index(tile_k_bytes // 16)
            c4_bytes = fx.Index(4)

            # Precompute per-load tile coordinates (row_local, col_local_i32)
            a_row_local = []
            a_col_local_i32 = []
            for i in range_constexpr(num_a_loads):
                row_local, col_local_i32 = tile_chunk_coord_i32(
                    arith, tx_i32_base=tx_i32_base, i=i,
                    total_threads=total_threads,
                    layout_tile_div4=layout_a_tile_div4,
                    chunk_i32=chunk_i32_a,
                )
                a_row_local.append(row_local)
                a_col_local_i32.append(col_local_i32)

            # ── Helper: load A tile to LDS with XOR swizzle ─────────────
            def load_a_tile(k_tile_idx_py, lds_base):
                """Load A tile from global to LDS with XOR16 swizzle."""
                base_k_div4 = fx.Index(k_tile_idx_py * tile_k_dwords)
                for i in range_constexpr(num_a_loads):
                    row_global = bx_m + a_row_local[i]
                    idx_i32 = group_a_off_div4 + row_global * _k_div4_factor + base_k_div4 + a_col_local_i32[i]
                    a_vec = buffer_ops.buffer_load(a_rsrc, idx_i32, vec_width=4, dtype=T.i32)
                    lds_store_16b_xor16(
                        arith, vector,
                        lds_memref=lds_a, vec16_ty=T.f8x16,
                        layout_lds=layout_lds,
                        row_local=a_row_local[i],
                        col_local_i32=a_col_local_i32[i],
                        tx_c4=c4_bytes, k_blocks16=k_blocks16,
                        lds_base=lds_base,
                        vec_part_i32x4=a_vec, elem_bytes=elem_bytes,
                    )

            # ── Helper: load A K64 pack from LDS with XOR swizzle ────────
            def lds_load_packs_k64(curr_row_a_lds, col_base_bytes, lds_base):
                """Load 16B from LDS with XOR16 swizzle, return two i64 halves."""
                col_base_swz_bytes = swizzle_xor16(curr_row_a_lds, col_base_bytes, k_blocks16)
                idx_a16 = crd2idx((curr_row_a_lds, col_base_swz_bytes), layout_lds) + lds_base
                loaded_a16 = vector.load_op(T.vec(16, T.f8), lds_a, [idx_a16])
                a_i64x2 = vector.bitcast(T.i64x2, loaded_a16)
                a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
                return a0, a1

            # ── Helper: load one B pack (K32 micro-step) ────────────────
            def load_b_pack(base_k, ki_step, ni):
                return load_b_pack_k32(
                    buffer_ops, arith, vector,
                    arg_b=arg_b, b_rsrc=b_rsrc,
                    layout_b=layout_b,
                    base_k=base_k, ki_step=ki_step,
                    n_blk=n_blk_list[ni],
                    n_intra=n_intra_list[ni],
                    lane_div_16=lane_div_16,
                    elem_type=T.f8,
                    kpack_bytes=kpack_bytes,
                    elem_bytes=elem_bytes,
                )

            # ── Helper: prefetch entire B tile (gmem -> regs) ───────────
            def load_b_tile(base_k):
                """Load all B packs for one K-tile.

                Returns list of length k_unroll, each entry is
                (packs_half0[ni], packs_half1[ni]) for one K64 micro-step.
                """
                b_tile = []
                for ku in range_constexpr(k_unroll):
                    packs0 = []
                    packs1 = []
                    for ni in range_constexpr(num_acc_n):
                        ki0 = (ku * 2) + 0
                        ki1 = (ku * 2) + 1
                        b0 = load_b_pack(base_k, ki0, ni)
                        b1 = load_b_pack(base_k, ki1, ni)
                        packs0.append(b0)
                        packs1.append(b1)
                    b_tile.append((packs0, packs1))
                return b_tile

            # ── Helper: compute one K-tile from LDS + B tile ────────────
            c_scale_k = fx.Index(scale_k)
            sa_group_off = group_idx * c_scale_k * m_in  # 3D scale_a Offset

            def compute_tile(accs_in, k_tile_idx_py, lds_base, b_tile_in):
                """Compute MFMA tiles for one K-tile, return updated accumulators."""
                current_accs = list(accs_in)

                for sb in range_constexpr(sb_per_tile):
                    kb = fx.Index(k_tile_idx_py * sb_per_tile + sb)

                    # Load scale_a for this K-block (per-token scale)
                    # scale_a layout: [G, scale_k, expected_m] transposed
                    sa_base = sa_group_off + kb * m_in
                    s_a_vecs = []
                    row_off_base = lane_div_16 * fx.Index(4)
                    for mi in range_constexpr(m_repeat):
                        s_a_row = []
                        for ii in range_constexpr(4):
                            row_in_tile = arith.index(mi * 16) + row_off_base + fx.Index(ii)
                            row_global = bx_m + row_in_tile
                            sa_idx = sa_base + row_global
                            s_a_val = buffer_ops.buffer_load(sa_rsrc, sa_idx, vec_width=1, dtype=T.f32)
                            s_a_row.append(s_a_val)
                        s_a_vec4 = vector.from_elements(T.f32x4, s_a_row)
                        s_a_vecs.append(s_a_vec4)

                    # Load scale_b for this K-block
                    # scale_b layout: [G, scale_n, scale_k]
                    sb_group_offset = group_idx * fx.Index(scale_n * scale_k)
                    s_b_vals = []
                    for ni in range_constexpr(num_acc_n):
                        sb_idx = sb_group_offset + n_block_for_scale[ni] * c_scale_k + kb
                        s_b_val = buffer_ops.buffer_load(sb_rsrc, sb_idx, vec_width=1, dtype=T.f32)
                        s_b_vals.append(s_b_val)

                    # MFMA computation for this scale block
                    ku_per_sb = scale_block_k // 64

                    for ku_local in range_constexpr(ku_per_sb):
                        ku = sb * ku_per_sb + ku_local
                        k_offset_bytes = ku * 64
                        b_packs0, b_packs1 = b_tile_in[ku]

                        for mi in range_constexpr(m_repeat):
                            row_a_lds = lane_mod_16 + arith.index(mi * 16)
                            col_a_base_bytes = lane_div_16 * fx.Index(16) + fx.Index(k_offset_bytes)
                            a0, a1 = lds_load_packs_k64(row_a_lds, col_a_base_bytes, lds_base)

                            for ni in range_constexpr(num_acc_n):
                                acc_idx = mi * num_acc_n + ni

                                # Two K32 MFMAs using preshuffle B packs
                                mfma_fn = rocdl.mfma_f32_16x16x32_fp8_fp8
                                mfma_mid = mfma_fn(T.f32x4, [a0, b_packs0[ni], acc_init, 0, 0, 0])
                                mfma_result = mfma_fn(T.f32x4, [a1, b_packs1[ni], mfma_mid, 0, 0, 0])

                                # Apply scales: accum += mfma_result * scale_a * scale_b
                                s_a_v4 = s_a_vecs[mi]
                                s_b_bc = vector.broadcast(T.f32x4, s_b_vals[ni])
                                scaled = ArithValue(mfma_result) * ArithValue(s_a_v4)
                                current_accs[acc_idx] = math_dialect.fma(scaled, s_b_bc, current_accs[acc_idx])

                return current_accs

            # ===== Ping-pong K-loop =====
            # Prologue: load first A tile and B tile
            load_a_tile(0, lds_base_pong)
            b_tile_pong = load_b_tile(fx.Index(0))
            gpu.barrier()

            for k_pair in range_constexpr(0, num_k_tiles, 2):
                # Load next A+B into ping while computing current from pong
                if k_pair + 1 < num_k_tiles:
                    load_a_tile(k_pair + 1, lds_base_ping)
                    b_tile_ping = load_b_tile(fx.Index((k_pair + 1) * tile_k))
                accs = compute_tile(accs, k_pair, lds_base_pong, b_tile_pong)
                gpu.barrier()

                # Load next A+B into pong while computing current from ping
                if k_pair + 1 < num_k_tiles:
                    if k_pair + 2 < num_k_tiles:
                        load_a_tile(k_pair + 2, lds_base_pong)
                        b_tile_pong = load_b_tile(fx.Index((k_pair + 2) * tile_k))
                    accs = compute_tile(accs, k_pair + 1, lds_base_ping, b_tile_ping)
                    gpu.barrier()

            # ===== Epilogue: store results =====
            c_n = n_in
            lane_div_16_mul4 = lane_div_16 * fx.Index(4)
            d_group_off = group_idx * m_in * n_in  # 3D D Offset

            for mi in range_constexpr(m_repeat):
                for ii in range_constexpr(4):
                    row_off = lane_div_16_mul4 + fx.Index(ii)
                    row_in_tile = arith.index(mi * 16) + row_off
                    row_global = bx_m + row_in_tile

                    for ni in range_constexpr(num_acc_n):
                        acc_idx = mi * num_acc_n + ni
                        col_base = by_n + n_tile_base + arith.index(ni * 16) + lane_mod_16

                        # Extract scalar from accumulator
                        val_f32 = vector.extract(accs[acc_idx], static_position=[ii], dynamic_position=[])
                        val_out = arith.trunc_f(out_mlir(), val_f32)

                        # Store to D
                        d_idx = d_group_off + row_global * c_n + col_base
                        buffer_ops.buffer_store(val_out, d_rsrc, d_idx)

            scf.YieldOp([])

    # ===== JIT Launcher =====
    @flyc.jit
    def launch_masked_grouped_fp8_gemm(
        arg_d: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        arg_masked_m: fx.Tensor,
        i32_expected_m: fx.Int32,
        i32_n: fx.Int32,
        i32_k: fx.Int32,
        i32_num_groups: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        # Grid dimensions
        max_m_in = arith.index_cast(T.index, i32_expected_m)
        n_in = arith.index_cast(T.index, i32_n)
        num_groups_in = arith.index_cast(T.index, i32_num_groups)

        gx = n_in // fx.Index(tile_n)  # N-blocks
        gy = (max_m_in + fx.Index(tile_m - 1)) // fx.Index(tile_m)  # M-blocks (ceil)
        gz = num_groups_in

        launcher = masked_grouped_fp8_gemm_kernel(
            arg_d,
            arg_a,
            arg_b,
            arg_scale_a,
            arg_scale_b,
            arg_masked_m,
            i32_expected_m,
            i32_n,
            i32_k,
            i32_num_groups,
        )
        launcher.launch(grid=(gx, gy, gz), block=(total_threads, 1, 1), stream=stream)

    return launch_masked_grouped_fp8_gemm
