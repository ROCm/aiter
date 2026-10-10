# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx950 dense BF16 x MXFP4 GEMM kernel.

The activation is consumed as BF16 without quantization. Preshuffled packed
E2M1 weights and E8M0 block scales are loaded directly into registers, where
``v_cvt_scalef32_pk_bf16_fp4`` produces transient BF16 fragments for
``mfma_f32_16x16x32_bf16``. A dequantized weight matrix is never written to
global memory.

The K loop double-buffers A-LDS. Prefill (``k_wave=1``) still issues the next
A/B tile before MFMA. Decode (``k_wave>1``) issues after MFMA so only one B
tile is live in the cvt cluster and occupancy can rise above 1.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch

from .gemm_a16wfp4_helpers import (
    DenseGemmTraits,
    advance_k_tile,
    compute_dense_tile,
    load_a_frags,
    load_b_tile,
    make_dense_accumulators,
    make_dense_operand_loaders,
    make_dense_store,
    reduce_dense_k_wave,
    store_dense_bf16,
)

__all__ = ["compile_gemm_a16wfp4"]


@functools.cache
def compile_gemm_a16wfp4(
    *,
    N: int,
    K: int,
    BM: int = 16,
    TILE_N: int = 256,
    TILE_K: int = 256,
    b_cache_mod: int = 0,
    k_wave: int = 1,
    waves_per_eu: int | None = None,
):
    """Compile a cached gfx950 launcher; B loads default to cache reuse."""
    arch = str(get_rocm_arch()).split(":", 1)[0]
    if arch != "gfx950":
        raise ValueError(f"dense a16wfp4 FlyDSL kernel requires gfx950, got {arch!r}")
    if BM <= 0 or BM % 16:
        raise ValueError(f"BM must be a positive multiple of 16, got {BM}")
    if TILE_K <= 0 or TILE_K % 128 or K % TILE_K:
        raise ValueError(
            f"TILE_K must be a multiple of 128 dividing K; got TILE_K={TILE_K}, K={K}"
        )
    if K % 256:
        raise ValueError(f"K must be divisible by 256 for the scale layout, got {K}")
    if k_wave not in (1, 2, 4):
        raise ValueError(f"k_wave must be 1, 2, or 4, got {k_wave}")
    if K % (k_wave * TILE_K):
        raise ValueError(f"K={K} must be divisible by k_wave*TILE_K={k_wave * TILE_K}")
    # The N tile is split across 4//k_wave waves, each owning whole 16-wide MFMA
    # N blocks. Split-K decode therefore admits tiles below 64 to widen the grid.
    n_tile_align = 16 * (4 // k_wave)
    if TILE_N < n_tile_align or TILE_N % n_tile_align or N % TILE_N:
        raise ValueError(
            f"TILE_N must be a multiple of {n_tile_align} dividing N; "
            f"got TILE_N={TILE_N}, N={N}, k_wave={k_wave}"
        )

    traits = DenseGemmTraits(
        N,
        K,
        BM,
        TILE_N,
        TILE_K,
        b_cache_mod,
        k_wave=k_wave,
        waves_per_eu=waves_per_eu,
    )
    cache_tag = traits.cache_key
    block_m = traits.block_m
    n_blocks = traits.n_blocks
    a_lds_stages = traits.a_lds_stages

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, traits.lds_bytes, 16]

    @flyc.kernel(name=traits.kernel_name, known_block_size=[256, 1, 1])
    def kernel(
        arg_a: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_out: fx.Int64,
        i32_m: fx.Int32,
    ):
        const_expr(cache_tag)
        lds_raw_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        tx = fx.Int32(gpu.thread_id("x"))
        lane = tx % 64
        wave = rocdl.readfirstlane(T.i32, tx // 64)

        # X varies fastest, so consecutive workgroups reuse one cached B tile
        # across M blocks without runtime div/mod in the kernel prologue.
        m_block = fx.Int32(gpu.block_id("x"))
        n_block = fx.Int32(gpu.block_id("y"))
        m_row = m_block * traits.block_m
        n_row = n_block * traits.tile_n

        a_loader, b_loader, columns, wave_n_id, wave_k_id = make_dense_operand_loaders(
            traits,
            lds_raw_ptr,
            arg_a,
            arg_bq,
            arg_bscale,
            i32_m,
            m_row,
            n_row,
            lane,
            wave,
        )

        accumulators = make_dense_accumulators(traits)
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
        k_base = (
            wave_k_id * fx.Int32(traits.k_len) if traits.k_wave > 1 else fx.Int32(0)
        )

        a_loader.store_tile(k_base, slot=0)
        b_cur = load_b_tile(traits, b_loader, columns, k_base)
        for kt in range_constexpr(traits.k_tiles):
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()
            a_frags = load_a_frags(traits, a_loader, kt % a_lds_stages)
            if const_expr(traits.k_wave > 1):
                # Split-K keeps a single B tile live so occupancy can rise;
                # prefill issues the next tile first to keep the pipeline full.
                compute_dense_tile(
                    traits, mma_atom, accumulators, b_loader, b_cur, a_frags
                )
                if const_expr(kt + 1 < traits.k_tiles):
                    b_cur = advance_k_tile(
                        traits, a_loader, b_loader, columns, k_base, kt
                    )
            else:
                if const_expr(kt + 1 < traits.k_tiles):
                    b_nxt = advance_k_tile(
                        traits, a_loader, b_loader, columns, k_base, kt
                    )
                compute_dense_tile(
                    traits, mma_atom, accumulators, b_loader, b_cur, a_frags
                )
                if const_expr(kt + 1 < traits.k_tiles):
                    b_cur = b_nxt
            if const_expr(traits.k_tiles == 1):
                gpu.barrier()

        if const_expr(traits.k_wave > 1):
            reduce_dense_k_wave(
                traits, accumulators, lds_raw_ptr, wave, wave_n_id, lane
            )

        store = make_dense_store(arg_out, traits.n)
        is_store_wave = wave_k_id == fx.Int32(0) if traits.k_wave > 1 else True
        lane_div_16 = lane // 16
        lane_mod_16 = lane % 16
        for mi in range_constexpr(traits.m_repeats):
            row_base = m_row + mi * 16 + lane_div_16 * 4
            for ni in range_constexpr(traits.n_repeats):
                col = n_row + wave_n_id * traits.n_per_wave + ni * 16 + lane_mod_16
                values = Vec(accumulators[mi][ni].load())
                for vi in range_constexpr(4):
                    row = row_base + vi
                    if const_expr(traits.k_wave > 1):
                        store_ok = (row < i32_m) & is_store_wave
                    else:
                        store_ok = row < i32_m
                    if store_ok:
                        store_dense_bf16(store, values[vi], row * traits.n + col)

    @flyc.jit
    def launch(
        arg_a: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_out: fx.Int64,
        i32_m: fx.Int32,
        stream: fx.Stream,
    ):
        const_expr(cache_tag)
        grid_m = (i32_m + block_m - 1) // block_m
        kernel(
            arg_a,
            arg_bq,
            arg_bscale,
            arg_out,
            i32_m,
            value_attrs={"rocdl.waves_per_eu": waves_per_eu} if waves_per_eu else None,
        ).launch(
            grid=(fx.Int64(grid_m), n_blocks, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch
