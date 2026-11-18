# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# Do NOT add `from __future__ import annotations`: PEP 563 stringifies the
# annotations and defeats flydsl's runtime-arg detection in the JIT cache key.

"""K1 of the two-stage Gated-Residual kernel: combine +
grouped-RMSNorm + down GEMM, fused so ``xn`` never touches HBM.

:func:`flydsl_k1_combine_norm_down` is the entry point. A high-occupancy
combine+RMS prologue (:func:`_build_combine_rms`) emits the combined residual
``r2`` (bf16, a required output) plus the tiny per-stream ``rrms``; a down+inject
GEMM then re-forms

    xn[m, c] = r2[m, c] * rrms[m, stream(c)] * (1 + w[c])

*inside the A-load* (never materializing ``xn``) and contracts it against the
merged ``[n_pad, hidden]`` down+inject weight, emitting the packed
``SiLU bottleneck | raw injection`` output. The K reduction is dispatched by
token count: split-K partials at small/mid M
(:func:`_build_down_norm_partial_pipe`), a decoupled async-LDS pipeline at large
M (:func:`_build_down_norm_pipe`), and an MMA-free GEMV two-stage at decode M
(:func:`flydsl_k1k2_skinny_decode`).

Numerics: the reference normalizes in fp32, rounds ``xn`` to bf16, then matmuls;
the A-load reproduces that order (widen ``r2``, scale by ``rrms*(1+w)`` in fp32,
re-round to bf16). ``fold_w=True`` bakes ``(1+w)`` into the weight so the A-load
skips the affine. The split-K/decouple f32 reduction order differs from a single
workgroup, so the result is oracle-accurate rather than bit-exact.
"""

import math
import os
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from aiter.jit.utils.chip_info import get_lds_capacity_bytes
from aiter.ops.flydsl.kernels.act import sigmoid_f32
from aiter.ops.flydsl.kernels.hyper_connection_gated_residual.common import (
    ab_k_perm,
    arch_name,
    check_hidden,
    check_no_overlap,
    check_norm_weight,
    check_tensor,
    mfma_bf16,
    norm_weight_f32,
    require_supported_arch,
    stream_scope,
)
from aiter.ops.flydsl.kernels.hyper_connection_gated_residual.tuned import k1_plan
from aiter.ops.flydsl.kernels.tensor_shim import GTensor, _run_compiled

WAVE = 64
CHUNK = 8  # bf16 elements per 128-bit buffer access


@lru_cache(maxsize=32)
def _build_down_norm_pipe(
    hidden: int,
    n_pad: int,
    block_n: int,
    silu_cols: int,
    hc_count: int,
    w_len: int,
    block_m: int,
    block_k: int,
    m_waves: int,
    n_waves: int,
    stages: int,
    mma_m: int,
    mma_n: int,
    mma_k: int,
    fold_w: bool = False,
    w_lds: bool = True,
    _skip_norm: bool = False,
):
    """Async-LDS pipelined down+inject GEMM with the norm folded into the A-load.

    ``_skip_norm`` is diagnostic and produces invalid output by omitting the
    A-transform. ``w_lds=False`` reads a shared ``(1+w)`` from global memory
    instead of staging it, for widths whose weight does not fit beside the panels.

    A separate combine/RMS prologue emits ``r2`` and ``rrms``. This kernel forms
    ``xn = r2*rrms*(1+w)`` while loading A and reduces K through an async
    global-to-LDS pipeline, without materializing ``xn``.

    N-tiled: the ``[n_pad]`` output width is split into ``n_pad//block_n``
    column blocks over ``grid.y``. Smaller B tiles reduce LDS use and expose more
    workgroups.
    """
    assert (
        n_pad % block_n == 0
    ), f"n_pad={n_pad} must be a multiple of block_n={block_n}"
    n_cblocks = n_pad // block_n
    if mma_k == 16:  # gfx942 uses the architecture-specific GEMM helpers.
        from aiter.ops.flydsl.kernels.hyper_connection_gated_residual.gemm_a16w16_gfx942 import (
            GEMM_A16W16_DTYPE_BF16,
            AsyncLoadTile,
            async_load_to_lds,
            make_gemm_a16w16_gfx950_param,
            make_gemm_ab_lds_layouts,
            make_gemm_ab_load_context,
        )
    else:  # gfx950 uses the shared GEMM helpers.
        from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 import (
            GEMM_A16W16_DTYPE_BF16,
            AsyncLoadTile,
            async_load_to_lds,
            make_gemm_a16w16_gfx950_param,
            make_gemm_ab_lds_layouts,
            make_gemm_ab_load_context,
        )

    inv_hc = 1.0 / hc_count
    stream_dim = hidden // hc_count
    # all_silu is over the FULL output width (n_pad), not the N-tile: with N-tiling
    # block_n < silu_cols must still take the per-column split path.
    all_silu = silu_cols >= n_pad
    block_threads = m_waves * n_waves * 64
    assert block_m % (m_waves * mma_m) == 0
    assert block_n % (n_waves * mma_n) == 0
    assert block_k % mma_k == 0 and hidden % block_k == 0
    assert stream_dim % block_k == 0
    assert stages >= 2
    assert (
        block_m * block_k >= block_threads * 8
    ), "async-LDS pipe needs block_m*block_k >= block_threads*async_vec"
    k_tiles = hidden // block_k
    # The prologue preloads stages-1 tiles before the main loop.
    assert k_tiles >= stages - 1, f"k_tiles={k_tiles} too short for stages={stages}"
    # Stage (1+w) in LDS only for an unfolded shared [stream_dim] weight. A full
    # [hidden] weight does not fit alongside the A/B panels at large M. Folded
    # weights need no staging because (1+w) is already baked into w_dn.
    stage_w = w_lds and (not fold_w) and w_len != hidden
    assert not stage_w or w_len % block_threads == 0
    w_stage_iters = (w_len // block_threads) if stage_w else 0
    w1_len = w_len if stage_w else 1
    # Stage this M-tile's rrms ([block_m, hc]) in LDS once: the A-transform
    # otherwise re-loads rrms from global for every fragment element on every
    # k-tile. Only when the tile divides the workgroup evenly (it does for the
    # decouple's block_m=64, hc=4, 256 threads); else fall back to per-element.
    stage_rrms = (block_m * hc_count) % block_threads == 0
    rr_stage_iters = (block_m * hc_count) // block_threads if stage_rrms else 0

    param = make_gemm_a16w16_gfx950_param(
        in_dtype_id=GEMM_A16W16_DTYPE_BF16,
        out_dtype_id=GEMM_A16W16_DTYPE_BF16,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        stages=stages,
        split_k=1,
        m_waves=m_waves,
        n_waves=n_waves,
        k_waves=1,
        a_is_transposed=False,
        b_is_transposed=True,
        mma_m=mma_m,
        mma_n=mma_n,
        mma_k=mma_k,
    )
    ldg_a_iters = param.ldg_a_iters
    ldg_b_iters = param.ldg_b_iters

    @flyc.kernel(
        name=f"gr_downnormpipe_hc{hc_count}_h{hidden}_n{block_n}_s{silu_cols}"
        f"_bm{block_m}_bk{block_k}_nw{n_waves}_st{stages}{'' if w_lds else '_wg'}",
        known_block_size=[block_threads, 1, 1],
    )
    def kernel(
        r2: fx.Tensor,  # [M, hidden] bf16 (combined residual, un-normalized)
        rrms: fx.Tensor,  # [M, hc] f32
        w: fx.Tensor,  # [w_len] f32
        w_dn: fx.Tensor,  # [block_n, hidden] bf16
        out: fx.Tensor,  # [M, block_n] bf16
    ):
        tid = fx.thread_idx.x
        bid_m = fx.block_idx.x
        bid_n = fx.block_idx.y  # N-tile in [0, n_cblocks)
        m = fx.Int32(fx.get_scalar(r2.shape[0]))
        block_m_offset = bid_m * block_m
        n_offset = bid_n * block_n

        r2_buf = fx.rocdl.make_buffer_tensor(r2, max_size=True)
        wdn_buf = fx.rocdl.make_buffer_tensor(w_dn, max_size=True)
        out_buf = fx.rocdl.make_buffer_tensor(out, max_size=True)
        rrms_g = GTensor(rrms, T.f32, (1, hc_count))
        w_g = GTensor(w, T.f32, (1, w_len))

        @fx.struct
        class Smem:
            a: fx.Array[fx.BFloat16, stages * block_m * block_k, 16]
            b: fx.Array[fx.BFloat16, stages * block_n * block_k, 16]
            w1: fx.Array[fx.Float32, w1_len, 16]
            rr: fx.Array[fx.Float32, block_m * hc_count, 16]

        smem = fx.SharedAllocator().allocate(Smem)
        smem_a = smem.a.peek().ptr
        smem_b = smem.b.peek().ptr
        sW1 = fx.make_view(smem.w1.peek().ptr, fx.make_layout((w1_len,), (1,)))
        sRrms = fx.make_view(
            smem.rr.peek().ptr, fx.make_layout((block_m, hc_count), (hc_count, 1))
        )
        if const_expr(stage_w):
            for wi in range_constexpr(w_stage_iters):
                widx = wi * block_threads + tid
                sW1[widx] = fx.Float32(1.0) + fx.Float32(w_g.load(widx, vec_size=1))
        for ri in range_constexpr(rr_stage_iters):
            ridx = ri * block_threads + tid
            local_m = ridx // hc_count
            st = ridx % hc_count
            sRrms[local_m, st] = fx.Float32(
                rrms_g.load((block_m_offset + local_m) * hc_count + st, vec_size=1)
            )
        fx.gpu.barrier()

        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(mma_m, mma_n, mma_k, fx.BFloat16))
        tiled_mma = fx.make_tiled_mma(
            mma_atom,
            fx.make_layout((m_waves, n_waves, 1), (n_waves, 1, 0)),
            fx.make_tile(None, None, ab_k_perm(mma_k)),
        )
        thr_mma = tiled_mma.thr_slice(tid)
        ctx = make_gemm_ab_load_context(
            fx.BFloat16, load_tid=tid, ks_begin=fx.Int32(0), param=param
        )
        a_lds_layout, b_lds_layout = make_gemm_ab_lds_layouts(
            block_m, block_n, block_k, False, True
        )
        thr_copy_A = fx.make_tiled_copy_A(ctx.a_tiled_copy_atom, tiled_mma).get_slice(
            tid
        )
        thr_copy_B = fx.make_tiled_copy_B(ctx.b_tiled_copy_atom, tiled_mma).get_slice(
            tid
        )

        c_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        thr_copy_C = fx.make_tiled_copy_C(c_copy_atom, tiled_mma).get_slice(tid)
        gC = fx.flat_divide(out_buf, (block_m, block_n))[None, None, bid_m, bid_n]
        frag_C = thr_mma.make_fragment_C(gC)

        sA0 = fx.make_view(smem_a, a_lds_layout)
        sB0 = fx.make_view(smem_b, b_lds_layout)
        frag_A = thr_mma.make_fragment_A(sA0)
        frag_B = thr_mma.make_fragment_B(sB0)
        frag_A_ret = thr_copy_A.retile(frag_A)
        frag_B_ret = thr_copy_B.retile(frag_B)

        a_elems = fx.size(frag_A.shape).unpack()
        aRow = thr_mma.partition_A(
            fx.make_view(0, fx.make_layout((block_m, block_k), (1, 0)))
        )
        aK = thr_mma.partition_A(
            fx.make_view(0, fx.make_layout((block_m, block_k), (0, 1)))
        )

        def load_a(k_tile, stage):
            async_load_to_lds(
                tile=AsyncLoadTile(
                    lds_base=smem_a + stage * block_m * block_k,
                    src_base=fx.get_iter(r2_buf),
                    lds_layout=a_lds_layout,
                    outer_tile_size=block_m,
                    outer_bound=m,
                    global_outer_offset=block_m_offset,
                    leading_stride=fx.Int32(hidden),
                    k_tile=k_tile,
                ),
                context=ctx,
                load_iters=ldg_a_iters,
                is_k_major=False,
            )

        def load_b(k_tile, stage):
            async_load_to_lds(
                tile=AsyncLoadTile(
                    lds_base=smem_b + stage * block_n * block_k,
                    src_base=fx.get_iter(wdn_buf),
                    lds_layout=b_lds_layout,
                    outer_tile_size=block_n,
                    outer_bound=fx.Int32(n_pad),
                    global_outer_offset=n_offset,
                    leading_stride=fx.Int32(hidden),
                    k_tile=k_tile,
                ),
                context=ctx,
                load_iters=ldg_b_iters,
                is_k_major=False,
            )

        scale_frag = fx.make_fragment_like(frag_A, fx.Float32)

        def transform(kt):
            # Match the reference order: normalize r2 in f32, then round xn to
            # bf16. A scale fragment turns scalar normalization into one packed
            # fragment multiply.
            va = frag_A.load().to(fx.Float32)
            scales = []
            for i in range_constexpr(a_elems):
                m_i = fx.get_scalar(aRow[i])
                k_i = fx.get_scalar(aK[i])
                col = kt * block_k + k_i
                stream = col // stream_dim
                if const_expr(stage_rrms):
                    rr = sRrms[m_i, stream]
                else:
                    g_m = bid_m * block_m + m_i
                    rr = fx.Float32(rrms_g.load(g_m * hc_count + stream, vec_size=1))
                if const_expr(fold_w):
                    scales.append(rr)
                elif const_expr(stage_w):
                    scales.append(rr * sW1[col % w_len])
                else:  # full [hidden] weight, unfolded: load (1+w) from global
                    onepw = fx.Float32(1.0) + fx.Float32(
                        w_g.load(col % w_len, vec_size=1)
                    )
                    scales.append(rr * onepw)
            scale_frag.store(fx.Vector.from_elements(scales, dtype=fx.Float32))
            frag_A.store((va * scale_frag.load()).to(fx.BFloat16))

        def compute_stage(read_stage, kt):
            thr_sA = thr_copy_A.partition_S(
                fx.make_view(smem_a + read_stage * block_m * block_k, a_lds_layout)
            )
            thr_sB = thr_copy_B.partition_S(
                fx.make_view(smem_b + read_stage * block_n * block_k, b_lds_layout)
            )
            fx.copy(ctx.a_s2r_copy_atom, thr_sA, frag_A_ret)
            fx.copy(ctx.b_s2r_copy_atom, thr_sB, frag_B_ret)
            if const_expr(not _skip_norm):
                transform(kt)
            fx.gemm(
                tiled_mma,
                frag_C,
                frag_A,
                frag_B,
                frag_C,
                traversal_order=fx.GemmTraversalOrder.KNM,
            )

        frag_C.fill(0.0)
        for stage in range_constexpr(stages - 1):
            load_b(stage, stage)
            load_a(stage, stage)
            rocdl.asyncmark()

        main_loop_end = k_tiles - (stages - 1)
        for k_tile in range(main_loop_end):
            current_stage = k_tile % stages
            write_stage = (current_stage + stages - 1) % stages
            rocdl.wait_asyncmark(stages - 2)
            fx.gpu.barrier()
            load_b(k_tile + (stages - 1), write_stage)
            load_a(k_tile + (stages - 1), write_stage)
            rocdl.asyncmark()
            compute_stage(current_stage, k_tile)

        current_stage = main_loop_end % stages
        for s in range_constexpr(stages - 1):
            rocdl.wait_asyncmark(stages - 2 - s)
            fx.gpu.barrier()
            compute_stage(current_stage, main_loop_end + s)
            current_stage = (current_stage + 1) % stages

        n_elems = fx.size(frag_C.shape).unpack()
        cCol = thr_mma.partition_C(
            fx.make_view(0, fx.make_layout((block_m, block_n), (0, 1)))
        )
        vc = frag_C.load()
        out_vals = []
        for i in range_constexpr(n_elems):
            raw_v = vc[i]
            sv = raw_v * fx.Float32(inv_hc)
            silu_v = sv * sigmoid_f32(sv)
            if all_silu:
                out_vals.append(silu_v)
            else:
                # Global output column = n-tile offset + within-tile column
                # (runtime, since bid_n is a grid index): SiLU below silu_cols,
                # raw injection logits above.
                gcol = n_offset + fx.Int32(fx.get_scalar(cCol[i]))
                is_lora = gcol < fx.Int32(silu_cols)
                out_vals.append(is_lora.select(silu_v, raw_v))
        frag_out = fx.make_fragment_like(frag_C, fx.BFloat16)
        frag_out.store(
            fx.Vector.from_elements(out_vals, dtype=fx.Float32).to(fx.BFloat16)
        )
        fx.copy(c_copy_atom, thr_copy_C.retile(frag_out), thr_copy_C.partition_S(gC))

    @flyc.jit
    def launch(
        r2: fx.Tensor,
        rrms: fx.Tensor,
        w: fx.Tensor,
        w_dn: fx.Tensor,
        out: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        tokens = fx.Int32(fx.get_scalar(r2.shape[0]))
        grid_m = (tokens + block_m - 1) // block_m
        kernel(r2, rrms, w, w_dn, out).launch(
            grid=(fx.Int64(grid_m), n_cblocks, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch


@lru_cache(maxsize=32)
def _build_down_norm_partial_pipe(
    hidden: int,
    n_pad: int,
    block_n: int,
    hc_count: int,
    w_len: int,
    block_m: int,
    block_k: int,
    split_k: int,
    m_waves: int,
    n_waves: int,
    stages: int,
    mma_m: int,
    mma_n: int,
    mma_k: int,
    fold_w: bool = False,
    w_lds: bool = True,
):
    """Async-LDS pipelined split-K partial.

    Each ``grid.y`` workgroup reduces a disjoint K-slice and writes an f32
    partial. The reduction applies SiLU after combining the K-slices.
    ``w_lds`` is as in :func:`_build_down_norm_pipe`.
    """
    if mma_k == 16:  # gfx942 uses the architecture-specific GEMM helpers.
        from aiter.ops.flydsl.kernels.hyper_connection_gated_residual.gemm_a16w16_gfx942 import (
            GEMM_A16W16_DTYPE_BF16,
            AsyncLoadTile,
            async_load_to_lds,
            make_gemm_a16w16_gfx950_param,
            make_gemm_ab_lds_layouts,
            make_gemm_ab_load_context,
        )
    else:  # gfx950 uses the shared GEMM helpers.
        from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 import (
            GEMM_A16W16_DTYPE_BF16,
            AsyncLoadTile,
            async_load_to_lds,
            make_gemm_a16w16_gfx950_param,
            make_gemm_ab_lds_layouts,
            make_gemm_ab_load_context,
        )

    assert n_pad % block_n == 0
    n_cblocks = n_pad // block_n
    stream_dim = hidden // hc_count
    block_threads = m_waves * n_waves * 64
    assert block_m % (m_waves * mma_m) == 0
    assert block_n % (n_waves * mma_n) == 0
    assert block_k % mma_k == 0 and hidden % block_k == 0
    assert stream_dim % block_k == 0
    assert stages >= 2 and block_m * block_k >= block_threads * 8
    k_tiles = hidden // block_k
    assert k_tiles % split_k == 0
    k_tiles_local = k_tiles // split_k
    assert k_tiles_local >= stages - 1, "split-K slice too short for the pipeline"
    # Stage (1+w) in LDS only for an unfolded shared [stream_dim] weight. A full
    # [hidden] weight does not fit alongside the A/B panels at large M. Folded
    # weights need no staging because (1+w) is already baked into w_dn.
    stage_w = w_lds and (not fold_w) and w_len != hidden
    assert not stage_w or w_len % block_threads == 0
    w_stage_iters = (w_len // block_threads) if stage_w else 0
    w1_len = w_len if stage_w else 1
    stage_rrms = (block_m * hc_count) % block_threads == 0
    rr_stage_iters = (block_m * hc_count) // block_threads if stage_rrms else 0

    param = make_gemm_a16w16_gfx950_param(
        in_dtype_id=GEMM_A16W16_DTYPE_BF16,
        out_dtype_id=GEMM_A16W16_DTYPE_BF16,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        stages=stages,
        split_k=1,
        m_waves=m_waves,
        n_waves=n_waves,
        k_waves=1,
        a_is_transposed=False,
        b_is_transposed=True,
        mma_m=mma_m,
        mma_n=mma_n,
        mma_k=mma_k,
    )
    ldg_a_iters = param.ldg_a_iters
    ldg_b_iters = param.ldg_b_iters

    @flyc.kernel(
        name=f"gr_downnormpartpipe_hc{hc_count}_h{hidden}_np{n_pad}_n{block_n}"
        f"_bm{block_m}_bk{block_k}_sk{split_k}_st{stages}{'' if w_lds else '_wg'}",
        known_block_size=[block_threads, 1, 1],
    )
    def kernel(
        r2: fx.Tensor,  # [M, hidden] bf16
        rrms: fx.Tensor,  # [M, hc] f32
        w: fx.Tensor,  # [w_len] f32
        w_dn: fx.Tensor,  # [n_pad, hidden] bf16
        partial: fx.Tensor,  # [split_k*M, n_pad] f32
    ):
        tid = fx.thread_idx.x
        bid_m = fx.block_idx.x
        bid_k = fx.block_idx.y
        bid_n = fx.block_idx.z
        grid_m = fx.grid_dim.x
        m = fx.Int32(fx.get_scalar(r2.shape[0]))
        block_m_offset = bid_m * block_m
        n_offset = bid_n * block_n
        k_base = bid_k * k_tiles_local

        r2_buf = fx.rocdl.make_buffer_tensor(r2, max_size=True)
        wdn_buf = fx.rocdl.make_buffer_tensor(w_dn, max_size=True)
        part_buf = fx.rocdl.make_buffer_tensor(partial, max_size=True)
        rrms_g = GTensor(rrms, T.f32, (1, hc_count))
        w_g = GTensor(w, T.f32, (1, w_len))

        @fx.struct
        class Smem:
            a: fx.Array[fx.BFloat16, stages * block_m * block_k, 16]
            b: fx.Array[fx.BFloat16, stages * block_n * block_k, 16]
            w1: fx.Array[fx.Float32, w1_len, 16]
            rr: fx.Array[fx.Float32, block_m * hc_count, 16]

        smem = fx.SharedAllocator().allocate(Smem)
        smem_a = smem.a.peek().ptr
        smem_b = smem.b.peek().ptr
        sW1 = fx.make_view(smem.w1.peek().ptr, fx.make_layout((w1_len,), (1,)))
        sRrms = fx.make_view(
            smem.rr.peek().ptr, fx.make_layout((block_m, hc_count), (hc_count, 1))
        )
        if const_expr(stage_w):
            for wi in range_constexpr(w_stage_iters):
                widx = wi * block_threads + tid
                sW1[widx] = fx.Float32(1.0) + fx.Float32(w_g.load(widx, vec_size=1))
        for ri in range_constexpr(rr_stage_iters):
            ridx = ri * block_threads + tid
            local_m = ridx // hc_count
            st = ridx % hc_count
            sRrms[local_m, st] = fx.Float32(
                rrms_g.load((block_m_offset + local_m) * hc_count + st, vec_size=1)
            )
        fx.gpu.barrier()

        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(mma_m, mma_n, mma_k, fx.BFloat16))
        tiled_mma = fx.make_tiled_mma(
            mma_atom,
            fx.make_layout((m_waves, n_waves, 1), (n_waves, 1, 0)),
            fx.make_tile(None, None, ab_k_perm(mma_k)),
        )
        thr_mma = tiled_mma.thr_slice(tid)
        ctx = make_gemm_ab_load_context(
            fx.BFloat16, load_tid=tid, ks_begin=fx.Int32(0), param=param
        )
        a_lds_layout, b_lds_layout = make_gemm_ab_lds_layouts(
            block_m, block_n, block_k, False, True
        )
        thr_copy_A = fx.make_tiled_copy_A(ctx.a_tiled_copy_atom, tiled_mma).get_slice(
            tid
        )
        thr_copy_B = fx.make_tiled_copy_B(ctx.b_tiled_copy_atom, tiled_mma).get_slice(
            tid
        )

        c_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        thr_copy_C = fx.make_tiled_copy_C(c_copy_atom, tiled_mma).get_slice(tid)
        gC = fx.flat_divide(part_buf, (block_m, block_n))[
            None, None, bid_k * grid_m + bid_m, bid_n
        ]
        frag_C = thr_mma.make_fragment_C(gC)

        sA0 = fx.make_view(smem_a, a_lds_layout)
        sB0 = fx.make_view(smem_b, b_lds_layout)
        frag_A = thr_mma.make_fragment_A(sA0)
        frag_B = thr_mma.make_fragment_B(sB0)
        frag_A_ret = thr_copy_A.retile(frag_A)
        frag_B_ret = thr_copy_B.retile(frag_B)

        a_elems = fx.size(frag_A.shape).unpack()
        aRow = thr_mma.partition_A(
            fx.make_view(0, fx.make_layout((block_m, block_k), (1, 0)))
        )
        aK = thr_mma.partition_A(
            fx.make_view(0, fx.make_layout((block_m, block_k), (0, 1)))
        )

        def load_a(k_tile, stage):
            async_load_to_lds(
                tile=AsyncLoadTile(
                    lds_base=smem_a + stage * block_m * block_k,
                    src_base=fx.get_iter(r2_buf),
                    lds_layout=a_lds_layout,
                    outer_tile_size=block_m,
                    outer_bound=m,
                    global_outer_offset=block_m_offset,
                    leading_stride=fx.Int32(hidden),
                    k_tile=k_tile,
                ),
                context=ctx,
                load_iters=ldg_a_iters,
                is_k_major=False,
            )

        def load_b(k_tile, stage):
            async_load_to_lds(
                tile=AsyncLoadTile(
                    lds_base=smem_b + stage * block_n * block_k,
                    src_base=fx.get_iter(wdn_buf),
                    lds_layout=b_lds_layout,
                    outer_tile_size=block_n,
                    outer_bound=fx.Int32(n_pad),
                    global_outer_offset=n_offset,
                    leading_stride=fx.Int32(hidden),
                    k_tile=k_tile,
                ),
                context=ctx,
                load_iters=ldg_b_iters,
                is_k_major=False,
            )

        scale_frag = fx.make_fragment_like(frag_A, fx.Float32)

        def transform(gkt):
            # Vectorized normalize: fill a scale fragment, one packed
            # va*scale multiply instead of per-element scalar muls.
            va = frag_A.load().to(fx.Float32)
            scales = []
            for i in range_constexpr(a_elems):
                m_i = fx.get_scalar(aRow[i])
                k_i = fx.get_scalar(aK[i])
                col = gkt * block_k + k_i
                stream = col // stream_dim
                if const_expr(stage_rrms):
                    rr = sRrms[m_i, stream]
                else:
                    g_m = bid_m * block_m + m_i
                    rr = fx.Float32(rrms_g.load(g_m * hc_count + stream, vec_size=1))
                if const_expr(fold_w):
                    scales.append(rr)
                elif const_expr(stage_w):
                    scales.append(rr * sW1[col % w_len])
                else:  # full [hidden] weight, unfolded: load (1+w) from global
                    onepw = fx.Float32(1.0) + fx.Float32(
                        w_g.load(col % w_len, vec_size=1)
                    )
                    scales.append(rr * onepw)
            scale_frag.store(fx.Vector.from_elements(scales, dtype=fx.Float32))
            frag_A.store((va * scale_frag.load()).to(fx.BFloat16))

        def compute_stage(read_stage, gkt):
            thr_sA = thr_copy_A.partition_S(
                fx.make_view(smem_a + read_stage * block_m * block_k, a_lds_layout)
            )
            thr_sB = thr_copy_B.partition_S(
                fx.make_view(smem_b + read_stage * block_n * block_k, b_lds_layout)
            )
            fx.copy(ctx.a_s2r_copy_atom, thr_sA, frag_A_ret)
            fx.copy(ctx.b_s2r_copy_atom, thr_sB, frag_B_ret)
            transform(gkt)
            fx.gemm(
                tiled_mma,
                frag_C,
                frag_A,
                frag_B,
                frag_C,
                traversal_order=fx.GemmTraversalOrder.KNM,
            )

        frag_C.fill(0.0)
        for stage in range_constexpr(stages - 1):
            load_b(k_base + stage, stage)
            load_a(k_base + stage, stage)
            rocdl.asyncmark()

        main_loop_end = k_tiles_local - (stages - 1)
        for kt in range(main_loop_end):
            current_stage = kt % stages
            write_stage = (current_stage + stages - 1) % stages
            rocdl.wait_asyncmark(stages - 2)
            fx.gpu.barrier()
            load_b(k_base + kt + (stages - 1), write_stage)
            load_a(k_base + kt + (stages - 1), write_stage)
            rocdl.asyncmark()
            compute_stage(current_stage, k_base + kt)

        current_stage = main_loop_end % stages
        for s in range_constexpr(stages - 1):
            rocdl.wait_asyncmark(stages - 2 - s)
            fx.gpu.barrier()
            compute_stage(current_stage, k_base + main_loop_end + s)
            current_stage = (current_stage + 1) % stages

        # Raw f32 partial; SiLU / inject-split happen in the cross-K reduction.
        fx.copy(c_copy_atom, thr_copy_C.retile(frag_C), thr_copy_C.partition_S(gC))

    @flyc.jit
    def launch(
        r2: fx.Tensor,
        rrms: fx.Tensor,
        w: fx.Tensor,
        w_dn: fx.Tensor,
        partial: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        tokens = fx.Int32(fx.get_scalar(r2.shape[0]))
        grid_m = (tokens + block_m - 1) // block_m
        kernel(r2, rrms, w, w_dn, partial).launch(
            grid=(fx.Int64(grid_m), split_k, n_cblocks),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch


@lru_cache(maxsize=64)
def _build_down_gemv_partial(
    hidden: int,
    n_pad: int,
    hc_count: int,
    stream_dim: int,
    m_rows: int,
    split_k_per_stream: int,
    waves_per_block: int,
):
    """MMA-free skinny (GEMV) down+inject partial for the decode regime (small M).

    One wave owns one output column ``n``; its 64 lanes stride over a K-slice of a
    single residual stream and wave-reduce (``shuffle_xor``) to the column's dot
    product. The ``grid.z = hc_count`` axis pins the stream so ``rrms[m, stream]``
    is loaded once per block (no per-element ``k // stream_dim`` divide) and
    factors out of the K-sum (``down = rrms * sum_k r2*Wd'``). ``Wd`` must be the
    ``fold_w`` weight (``(1+w)`` baked in), so ``xn`` is re-formed as ``r2*rrms``
    with no ``(1+w)`` multiply and never materialized.

    Unlike the MMA partial this does **no 64-row tile padding**: it runs the true
    ``m_rows`` rows (M held in registers), so decode M=1 does 1 row of work, not 64.
    Writes a raw f32 partial ``[hc_count*split_k_per_stream * m_rows, n_pad]``; the
    shared :func:`common._build_reduce_silu` sums the ``hc_count*spk`` blocks and
    applies ``silu(down/nr)`` (inject columns kept raw).
    """
    assert (
        n_pad % waves_per_block == 0
    ), f"n_pad={n_pad} must be a multiple of waves_per_block={waves_per_block}"
    assert stream_dim % split_k_per_stream == 0
    kslice = stream_dim // split_k_per_stream
    assert kslice % WAVE == 0, f"kslice={kslice} must be a multiple of WAVE={WAVE}"
    iters = kslice // WAVE
    block_threads = waves_per_block * WAVE
    log2_wave = int(math.log2(WAVE))

    @flyc.kernel(
        name=f"gr_down_gemv_hc{hc_count}_h{hidden}_np{n_pad}_m{m_rows}"
        f"_spk{split_k_per_stream}_w{waves_per_block}",
        known_block_size=[block_threads, 1, 1],
    )
    def kernel(
        r2: fx.Tensor,  # [m_rows, hidden] bf16
        rrms: fx.Tensor,  # [m_rows, hc] f32
        w_dn: fx.Tensor,  # [n_pad, hidden] bf16 (fold_w: (1+w) baked in)
        partial: fx.Tensor,  # [hc*spk*m_rows, n_pad] f32
    ):
        tid = fx.thread_idx.x
        wid = tid // WAVE
        lane = tid % WAVE
        bcol = fx.block_idx.x
        sk = fx.block_idx.y
        stream = fx.block_idx.z
        n = bcol * waves_per_block + wid

        r2_g = GTensor(r2, T.bf16, (1, m_rows * hidden))
        rrms_g = GTensor(rrms, T.f32, (1, m_rows * hc_count))
        wdn_g = GTensor(w_dn, T.bf16, (1, n_pad * hidden))
        part_g = GTensor(
            partial, T.f32, (1, hc_count * split_k_per_stream * m_rows * n_pad)
        )

        rr = [
            rrms_g.load(m * hc_count + stream, vec_size=1)
            for m in range_constexpr(m_rows)
        ]
        acc = [fx.Float32(0.0) for _ in range_constexpr(m_rows)]
        k_base = stream * stream_dim + sk * kslice + lane
        # Decode is latency-bound, so lane-strided scalar loads avoid MMA and
        # tiled-copy setup while remaining coalesced across each wave.
        for i in range_constexpr(iters):
            kk = k_base + i * WAVE
            wd = fx.BFloat16(wdn_g.load(n * hidden + kk, vec_size=1)).to(fx.Float32)
            for m in range_constexpr(m_rows):
                rv = fx.BFloat16(r2_g.load(m * hidden + kk, vec_size=1)).to(fx.Float32)
                acc[m] = acc[m] + wd * rv
        for m in range_constexpr(m_rows):
            a = acc[m]
            for sh in range_constexpr(log2_wave):
                a = a + fx.gpu.shuffle_xor(a, WAVE // (2 << sh), WAVE)
            acc[m] = a * rr[m]
        b = stream * split_k_per_stream + sk
        for m in range_constexpr(m_rows):
            part_g.store((b * m_rows + m) * n_pad + n, acc[m], vec_size=1)

    @flyc.jit
    def launch(
        r2: fx.Tensor,
        rrms: fx.Tensor,
        w_dn: fx.Tensor,
        partial: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel(r2, rrms, w_dn, partial).launch(
            grid=(n_pad // waves_per_block, split_k_per_stream, hc_count),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch


@lru_cache(maxsize=64)
def _build_up_gate_mix_gemv(
    hidden: int,
    stream_dim: int,
    hc_count: int,
    lowrank: int,
    w_len: int,
    m_rows: int,
    waves_per_block: int,
    lora_stride: int = 0,
):
    """MMA-free skinny (GEMV) K2 (up-GEMM + gated mean) for the decode regime.

    One wave owns one output channel ``c in [0, stream_dim)``. For each of the
    ``hc_count`` streams it computes ``gate = lora @ Wu[s*stream_dim+c].T`` as a
    lane-strided dot over ``lowrank`` (wave-reduced), re-forms
    ``xn = bf16(r2*rrms[m,s]*(1+w))`` in registers, and accumulates
    ``sigmoid(gate)*xn``; the output is ``mean_s`` (``*1/hc_count``). No 64-row tile
    padding is used. ``lora_stride=0`` means contiguous rows; a larger stride
    reads the low-rank columns directly from a packed K1 output.
    """
    assert stream_dim % waves_per_block == 0
    assert lowrank % WAVE == 0, f"lowrank={lowrank} must be a multiple of WAVE={WAVE}"
    lstride = lora_stride or lowrank
    k_iters = lowrank // WAVE
    inv_hc = 1.0 / hc_count
    block_threads = waves_per_block * WAVE
    log2_wave = int(math.log2(WAVE))
    shared_w = w_len != hidden  # norm_weight is [stream_dim] (shared) vs [hidden]

    @flyc.kernel(
        name=f"gr_up_gemv_hc{hc_count}_hs{stream_dim}_r{lowrank}_m{m_rows}"
        f"_w{waves_per_block}_ls{lstride}",
        known_block_size=[block_threads, 1, 1],
    )
    def kernel(
        lora: fx.Tensor,  # [m_rows, lstride] bf16 (lora is cols [0, lowrank))
        r2: fx.Tensor,  # [m_rows, hidden] bf16
        rrms: fx.Tensor,  # [m_rows, hc] f32
        w: fx.Tensor,  # [w_len] f32
        w_up: fx.Tensor,  # [hidden, lowrank] bf16
        block_input: fx.Tensor,  # [m_rows, stream_dim] bf16
    ):
        tid = fx.thread_idx.x
        wid = tid // WAVE
        lane = tid % WAVE
        c = fx.block_idx.x * waves_per_block + wid

        lora_g = GTensor(lora, T.bf16, (1, m_rows * lstride))
        r2_g = GTensor(r2, T.bf16, (1, m_rows * hidden))
        rrms_g = GTensor(rrms, T.f32, (1, m_rows * hc_count))
        w_g = GTensor(w, T.f32, (1, w_len))
        wup_g = GTensor(w_up, T.bf16, (1, hidden * lowrank))
        out_g = GTensor(block_input, T.bf16, (1, m_rows * stream_dim))

        w_shared = w_g.load(c, vec_size=1) if shared_w else None
        xacc = [fx.Float32(0.0) for _ in range_constexpr(m_rows)]
        # Lane-strided scalar loads avoid MMA and tiled-copy setup in the
        # latency-bound decode path.
        # lora does not depend on the stream: load it once, not once per stream.
        lo = [
            [
                fx.BFloat16(lora_g.load(m * lstride + lane + i * WAVE, vec_size=1)).to(
                    fx.Float32
                )
                for i in range_constexpr(k_iters)
            ]
            for m in range_constexpr(m_rows)
        ]
        for s in range_constexpr(hc_count):
            n = s * stream_dim + c
            g = [fx.Float32(0.0) for _ in range_constexpr(m_rows)]
            for i in range_constexpr(k_iters):
                r = lane + i * WAVE
                wu = fx.BFloat16(wup_g.load(n * lowrank + r, vec_size=1)).to(fx.Float32)
                for m in range_constexpr(m_rows):
                    g[m] = g[m] + lo[m][i] * wu
            onepw = (
                (fx.Float32(1.0) + w_shared)
                if shared_w
                else (fx.Float32(1.0) + w_g.load(n, vec_size=1))
            )
            for m in range_constexpr(m_rows):
                a = g[m]
                for sh in range_constexpr(log2_wave):
                    a = a + fx.gpu.shuffle_xor(a, WAVE // (2 << sh), WAVE)
                rr = rrms_g.load(m * hc_count + s, vec_size=1)
                r2v = fx.BFloat16(r2_g.load(m * hidden + n, vec_size=1)).to(fx.Float32)
                xn = fx.BFloat16(r2v * rr * onepw).to(fx.Float32)
                xacc[m] = xacc[m] + sigmoid_f32(a) * xn
        for m in range_constexpr(m_rows):
            out_g.store(
                m * stream_dim + c,
                (xacc[m] * fx.Float32(inv_hc)).to(fx.BFloat16),
                vec_size=1,
            )

    @flyc.jit
    def launch(
        lora: fx.Tensor,
        r2: fx.Tensor,
        rrms: fx.Tensor,
        w: fx.Tensor,
        w_up: fx.Tensor,
        block_input: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel(lora, r2, rrms, w, w_up, block_input).launch(
            grid=(stream_dim // waves_per_block, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch


def _decode_reduce_params(total: int):
    """(block_threads, vec) with ``bt*vec | total`` for the tiny decode reduce.

    ``total = tokens * n_pad``. Any ``n_pad`` that is a multiple of 4 keeps
    ``vec=4`` available, which matters because the vectorized reduce kernel cannot
    emit ``vec=1`` (it can't wrap a scalar load in a Vector). Scanning all block
    sizes lets a caller pass any padding, e.g. a merged weight padded to 16 rows
    instead of 64. Odd and sub-wavefront blocks are fine here -- the decode reduce
    is a tiny one-shot elementwise pass.
    """
    for vec in (4, 2, 1):
        for bt in range(min(256, total // vec), 0, -1):
            if total % (bt * vec) == 0:
                return bt, vec
    return 1, 1


# Temporary rollback switch for diagnosing the two-kernel skinny implementation.
# It is not a supported tuning knob and may be removed once the path is proven.
_SKINNY_TWO_KERNEL = os.environ.get("AITER_GR_SKINNY_TWO_KERNEL", "1") == "1"
# Chunked partial stores keep the two-kernel path within its register budget
# through M=8. The unchunked fallback is limited to M=4.
DECODE_MAX_M = 8 if _SKINNY_TWO_KERNEL else 4


@lru_cache(maxsize=64)
def skinny_decode_supported(
    hidden: int,
    n_pad: int,
    hc_count: int,
    lowrank: int,
    split_k_per_stream: int = 2,
    waves_per_block: int = 4,
) -> bool:
    """Whether :func:`flydsl_k1k2_skinny_decode` can build kernels for this shape.

    The op checks this before dispatch and otherwise uses the tiled path.
    """
    if _SKINNY_TWO_KERNEL:
        from aiter.ops.flydsl.kernels.hyper_connection_gated_residual.skinny import (
            skinny_supported,
        )

        return skinny_supported(hidden, n_pad, hc_count, lowrank, waves_per_block)
    stream_dim = hidden // hc_count
    return (
        n_pad % waves_per_block == 0
        and stream_dim % split_k_per_stream == 0
        and (stream_dim // split_k_per_stream) % WAVE == 0
        and stream_dim % waves_per_block == 0
        and lowrank % WAVE == 0
    )


def flydsl_k1k2_skinny_decode(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection: torch.Tensor,
    norm_weight: torch.Tensor,
    w_up: torch.Tensor,
    w_down_merged: torch.Tensor,  # MUST be fold_w (1+w baked into each column)
    lowrank: int,
    hc_count: int,
    eps: float,
    need_inj: bool,
    # Split each stream's K-slice across waves to expose decode parallelism.
    split_k_per_stream: int = 2,
    waves_per_block: int = 4,
    stream: torch.cuda.Stream | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """MMA-free skinny GEMV two-stage for decode M (combine+RMS -> down GEMV ->
    reduce/SiLU -> up GEMV + gated mean). No 64-row tile padding; ``xn`` never
    materialized. Returns ``(r2, block_input, inj_next)``.

    ``w_down_merged`` must be the ``(1+w)``-folded merged weight (see
    :func:`op.fold_norm_weight`). The down GEMV bakes that in -- it forms
    ``xn = r2*rrms`` with **no** ``(1+w)`` multiply -- so passing an unfolded merged
    weight here produces silently wrong output (no shape/dtype tripwire catches it).
    The op only routes here when ``fold_w=True``; do not call it directly otherwise.

    Inputs must satisfy :func:`validate_k1_inputs`; the op checks them before
    dispatch. ``block_output=None`` normalizes ``residual`` without a combine.
    """
    require_supported_arch()
    ctx, stream = stream_scope(stream, residual.device)
    with ctx:
        return _skinny_decode(
            residual,
            block_output,
            injection,
            norm_weight,
            w_up,
            w_down_merged,
            lowrank,
            hc_count,
            eps,
            need_inj,
            split_k_per_stream,
            waves_per_block,
            stream,
        )


def _skinny_decode(
    residual,
    block_output,
    injection,
    norm_weight,
    w_up,
    w_down_merged,
    lowrank,
    hc_count,
    eps,
    need_inj,
    split_k_per_stream,
    waves_per_block,
    stream,
):
    from aiter.ops.flydsl.kernels.hyper_connection_gated_residual.common import (
        _build_reduce_silu,
    )

    tokens, hidden = residual.shape
    stream_dim = hidden // hc_count
    n_pad = w_down_merged.shape[0]
    combine = block_output is not None
    # The injection may be a column slice of the previous packed output.
    inj_stride = injection.stride(0) if combine and tokens > 1 else hc_count
    skinny_reads_bf16 = (
        _SKINNY_TWO_KERNEL
        and norm_weight.dtype == torch.bfloat16
        and norm_weight.is_contiguous()
    )
    w = norm_weight.reshape(-1) if skinny_reads_bf16 else norm_weight_f32(norm_weight)
    if _SKINNY_TWO_KERNEL:
        from aiter.ops.flydsl.kernels.hyper_connection_gated_residual.skinny import (
            skinny_chunked_two_kernel,
            skinny_two_kernel,
        )

        # Use token groups from M=3: two rows for even M and one for odd M.
        chunked = tokens >= 3
        skinny_impl = skinny_chunked_two_kernel if chunked else skinny_two_kernel
        skinny_kwargs = {"chunk_m": 2 if tokens % 2 == 0 else 1} if chunked else {}
        r2, x, packed = skinny_impl(
            residual,
            block_output,
            injection,
            inj_stride,
            w,
            w_up,
            w_down_merged,
            lowrank,
            hc_count,
            eps,
            stream,
            **skinny_kwargs,
        )
        inj_next = packed[:, lowrank : lowrank + hc_count] if need_inj else None
        return r2, x, inj_next
    dev = residual.device
    r2 = (
        torch.empty(tokens, hidden, dtype=torch.bfloat16, device=dev)
        if combine
        else residual
    )
    rrms = torch.empty(tokens, hc_count, dtype=torch.float32, device=dev)
    nb = hc_count * split_k_per_stream
    partial = torch.empty(nb * tokens, n_pad, dtype=torch.float32, device=dev)
    packed = torch.empty(tokens, n_pad, dtype=torch.bfloat16, device=dev)
    x = torch.empty(tokens, stream_dim, dtype=torch.bfloat16, device=dev)

    pro = _build_combine_rms(
        hc_count,
        stream_dim,
        float(eps),
        0 if inj_stride == hc_count else inj_stride,
        combine=combine,
        write_r2=combine,
    )
    if not combine:
        block_output = injection = residual
    gd = _build_down_gemv_partial(
        hidden, n_pad, hc_count, stream_dim, tokens, split_k_per_stream, waves_per_block
    )
    total = tokens * n_pad
    bt, vec = _decode_reduce_params(total)
    red = _build_reduce_silu(total, nb, lowrank, n_pad, hc_count, bt, vec)
    # Up GEMV reads lora straight from ``packed`` (cols [0, lowrank), row stride
    # n_pad) -- no ``.contiguous()`` copy of the lora slice.
    gu = _build_up_gate_mix_gemv(
        hidden,
        stream_dim,
        hc_count,
        lowrank,
        w.numel(),
        tokens,
        waves_per_block,
        lora_stride=n_pad,
    )
    fxs = fx.Stream(stream)
    _run_compiled(pro, residual, block_output, injection, r2, rrms, fxs)
    _run_compiled(gd, r2, rrms, w_down_merged, partial, fxs)
    _run_compiled(red, partial, packed, fxs)
    _run_compiled(gu, packed, r2, rrms, w, w_up, x, fxs)
    # A view, not a copy: the next layer's prologue reads it strided.
    inj_next = packed[:, lowrank : lowrank + hc_count] if need_inj else None
    return r2, x, inj_next


@lru_cache(maxsize=16)
def _build_combine_rms(
    hc_count: int,
    stream_dim: int,
    eps: float,
    inj_stride: int = 0,
    combine: bool = True,
    write_r2: bool = True,
    write_rrms: bool = True,
):
    """Prologue for the split-K K1 path: combine + per-stream RMS, emit r2 + rrms.

    Writes the combined residual ``r2`` and per-stream reciprocal RMS values.
    The following GEMM reconstructs ``xn`` while loading A. ``combine=False``
    normalizes ``residual`` as-is (``block_output``/``injection`` are unread);
    ``write_r2=False`` then skips the copy when the GEMM reads ``residual``.
    ``write_rrms=False`` is the write-only combine: no RMS is computed and
    ``rrms_out`` is unread.
    """
    assert combine or inj_stride == 0
    assert write_r2 or not combine
    assert write_rrms or combine
    hidden = hc_count * stream_dim
    assert stream_dim % (WAVE * CHUNK) == 0
    chunks = stream_dim // (WAVE * CHUNK)
    inv_hc = 1.0 / hc_count
    inv_hs = 1.0 / stream_dim
    log2_wave = int(math.log2(WAVE))
    # Injection row stride in elements; 0 -> hc_count (contiguous). Callers pass
    # the logits as a column slice of the previous down-GEMM output.
    istride = inj_stride or hc_count
    stride_tag = f"_is{istride}" if inj_stride else ""
    mode_tag = "" if combine else ("_rms_copy" if write_r2 else "_rms")
    if not write_rrms:
        mode_tag = "_norms"

    @flyc.kernel(
        name=f"gr_combine_rms_hc{hc_count}_hs{stream_dim}{stride_tag}{mode_tag}",
        known_block_size=[WAVE, 1, 1],
    )
    def kernel(
        residual: fx.Tensor,  # [M, hidden] bf16
        block_output: fx.Tensor,  # [M, stream_dim] bf16
        injection: fx.Tensor,  # [M, hc] bf16
        r2_out: fx.Tensor,  # [M, hidden] bf16
        rrms_out: fx.Tensor,  # [M, hc] f32
    ):
        stream = fx.block_idx.x
        tok = fx.block_idx.y
        tid = fx.thread_idx.x
        r = GTensor(residual, T.bf16, (1, hidden))
        y = GTensor(block_output, T.bf16, (1, stream_dim))
        inj = GTensor(injection, T.bf16, (1, hc_count))
        r2 = GTensor(r2_out, T.bf16, (1, hidden))
        if const_expr(write_rrms):
            rrms_g = GTensor(rrms_out, T.f32, (1, hc_count))

        if const_expr(combine):
            inj_val = fx.BFloat16(inj.load(tok * istride + stream, vec_size=1)).to(
                fx.Float32
            )
            gate = fx.Float32(2.0) * sigmoid_f32(inj_val * fx.Float32(inv_hc))
        row_r = tok * hidden + stream * stream_dim
        row_y = tok * stream_dim
        sq = fx.Float32(0.0)
        for c in range_constexpr(chunks):
            off = c * (WAVE * CHUNK) + tid * CHUNK
            r_raw = r.load(row_r + off, vec_size=CHUNK)
            if const_expr(combine):
                rv = fx.Vector(r_raw).to(fx.Float32)
                yv = fx.Vector(y.load(row_y + off, vec_size=CHUNK)).to(fx.Float32)
                comb = [rv[i] + yv[i] * gate for i in range_constexpr(CHUNK)]
                comb_bf16 = fx.Vector.from_elements(comb, dtype=fx.Float32).to(
                    fx.BFloat16
                )
            else:
                comb_bf16 = fx.Vector(r_raw)
            if const_expr(write_r2):
                r2.store(row_r + off, comb_bf16)
            if const_expr(write_rrms):
                comb_r = comb_bf16.to(fx.Float32)
                for i in range_constexpr(CHUNK):
                    sq = sq + comb_r[i] * comb_r[i]
        if const_expr(write_rrms):
            for sh in range_constexpr(log2_wave):
                sq = sq + fx.gpu.shuffle_xor(sq, WAVE // (2 << sh), WAVE)
            rstd = fmath.rsqrt(sq * fx.Float32(inv_hs) + fx.Float32(eps))
            # Every lane has the same rstd. Redundant identical stores avoid
            # divergent control flow.
            rrms_g.store(tok * hc_count + stream, rstd, vec_size=1)

    @flyc.jit
    def launch(
        residual: fx.Tensor,
        block_output: fx.Tensor,
        injection: fx.Tensor,
        r2_out: fx.Tensor,
        rrms_out: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        tokens = fx.Int32(fx.get_scalar(residual.shape[0]))
        kernel(residual, block_output, injection, r2_out, rrms_out).launch(
            grid=(hc_count, fx.Int64(tokens), 1),
            block=(WAVE, 1, 1),
            stream=stream,
        )

    return launch


@lru_cache(maxsize=32)
def _build_down_norm_partial(
    hidden: int,
    n_pad: int,
    block_n: int,
    hc_count: int,
    w_len: int,
    block_m: int,
    block_k: int,
    split_k: int,
    m_waves: int,
    n_waves: int,
    mma_m: int,
    mma_n: int,
    mma_k: int,
    fold_w: bool = False,
):
    """Split-K partial down+inject GEMM with the norm folded into the A-load.

    Each of ``split_k`` workgroups reduces a disjoint K-slice and writes an f32
    partial. The cross-K reduction combines the slices before applying SiLU.

    N-tiled + LDS-rrms: the ``[n_pad]`` width is split into
    ``n_pad//block_n`` column blocks over ``grid.z``, and this M-tile's ``rrms``
    is staged in LDS once instead of a per-element global load in ``norm_A``.
    """
    assert (
        n_pad % block_n == 0
    ), f"n_pad={n_pad} must be a multiple of block_n={block_n}"
    n_cblocks = n_pad // block_n
    stream_dim = hidden // hc_count
    block_threads = m_waves * n_waves * 64
    assert block_m % (m_waves * mma_m) == 0
    assert block_n % (n_waves * mma_n) == 0
    assert block_k % mma_k == 0 and hidden % block_k == 0
    assert stream_dim % block_k == 0
    k_tiles = hidden // block_k
    assert (
        k_tiles % split_k == 0
    ), f"k_tiles={k_tiles} must be a multiple of split_k={split_k}"
    k_tiles_local = k_tiles // split_k
    stage_rrms = (block_m * hc_count) % block_threads == 0
    rr_stage_iters = (block_m * hc_count) // block_threads if stage_rrms else 0

    @flyc.kernel(
        name=f"gr_down_norm_partial_hc{hc_count}_h{hidden}_np{n_pad}_n{block_n}"
        f"_bm{block_m}_bk{block_k}_sk{split_k}",
        known_block_size=[block_threads, 1, 1],
    )
    def kernel(
        r2: fx.Tensor,  # [M, hidden] bf16
        rrms: fx.Tensor,  # [M, hc] f32
        w: fx.Tensor,  # [w_len] f32
        w_dn: fx.Tensor,  # [n_pad, hidden] bf16
        partial: fx.Tensor,  # [split_k*M, n_pad] f32
    ):
        tid = fx.thread_idx.x
        bid_m = fx.block_idx.x
        bid_k = fx.block_idx.y
        bid_n = fx.block_idx.z
        grid_m = fx.grid_dim.x

        r2_buf = fx.rocdl.make_buffer_tensor(r2, max_size=True)
        wdn_buf = fx.rocdl.make_buffer_tensor(w_dn, max_size=True)
        part_buf = fx.rocdl.make_buffer_tensor(partial, max_size=True)
        rrms_g = GTensor(rrms, T.f32, (1, hc_count))
        w_g = GTensor(w, T.f32, (1, w_len))

        if const_expr(stage_rrms):

            @fx.struct
            class Smem:
                rr: fx.Array[fx.Float32, block_m * hc_count, 16]

            smem = fx.SharedAllocator().allocate(Smem)
            sRrms = fx.make_view(
                smem.rr.peek().ptr, fx.make_layout((block_m, hc_count), (hc_count, 1))
            )
            for ri in range_constexpr(rr_stage_iters):
                ridx = ri * block_threads + tid
                lm = ridx // hc_count
                st = ridx % hc_count
                sRrms[lm, st] = fx.Float32(
                    rrms_g.load((bid_m * block_m + lm) * hc_count + st, vec_size=1)
                )
            fx.gpu.barrier()

        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(mma_m, mma_n, mma_k, fx.BFloat16))
        tiled_mma = fx.make_tiled_mma(
            mma_atom,
            fx.make_layout((m_waves, n_waves, 1), (n_waves, 1, 0)),
            fx.make_tile(None, None, ab_k_perm(mma_k)),
        )
        thr_mma = tiled_mma.thr_slice(tid)

        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        c_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        thr_copy_A = fx.make_tiled_copy_A(copy_atom, tiled_mma).get_slice(tid)
        thr_copy_B = fx.make_tiled_copy_B(copy_atom, tiled_mma).get_slice(tid)
        thr_copy_C = fx.make_tiled_copy_C(c_copy_atom, tiled_mma).get_slice(tid)

        gA = fx.flat_divide(r2_buf, (block_m, block_k))[None, None, bid_m, None]
        gB = fx.flat_divide(wdn_buf, (block_n, block_k))[None, None, bid_n, None]
        gC = fx.flat_divide(part_buf, (block_m, block_n))[
            None, None, bid_k * grid_m + bid_m, bid_n
        ]
        k_base = bid_k * k_tiles_local

        frag_C = thr_mma.make_fragment_C(gC)
        frag_A = [thr_mma.make_fragment_A(gA[None, None, 0]) for _ in range(2)]
        frag_B = [thr_mma.make_fragment_B(gB[None, None, 0]) for _ in range(2)]
        frag_A_ret = [thr_copy_A.retile(f) for f in frag_A]
        frag_B_ret = [thr_copy_B.retile(f) for f in frag_B]

        a_elems = fx.size(frag_A[0].shape).unpack()
        aRow = thr_mma.partition_A(
            fx.make_view(0, fx.make_layout((block_m, block_k), (1, 0)))
        )
        aK = thr_mma.partition_A(
            fx.make_view(0, fx.make_layout((block_m, block_k), (0, 1)))
        )

        def norm_A(kt, stage):
            va = frag_A[stage].load().to(fx.Float32)
            xn_vals = []
            for i in range_constexpr(a_elems):
                m_i = fx.get_scalar(aRow[i])
                k_i = fx.get_scalar(aK[i])
                col = k_base * block_k + kt * block_k + k_i
                stream = col // stream_dim
                if const_expr(stage_rrms):
                    rr = sRrms[m_i, stream]
                else:
                    g_m = bid_m * block_m + m_i
                    rr = fx.Float32(rrms_g.load(g_m * hc_count + stream, vec_size=1))
                if const_expr(fold_w):
                    xn_vals.append(va[i] * rr)
                else:
                    wv = fx.Float32(w_g.load(col % w_len, vec_size=1))
                    xn_vals.append(va[i] * rr * (fx.Float32(1.0) + wv))
            frag_A[stage].store(
                fx.Vector.from_elements(xn_vals, dtype=fx.Float32).to(fx.BFloat16)
            )

        def load_k(kt, stage):
            fx.copy(
                copy_atom,
                thr_copy_A.partition_S(gA[None, None, k_base + kt]),
                frag_A_ret[stage],
            )
            fx.copy(
                copy_atom,
                thr_copy_B.partition_S(gB[None, None, k_base + kt]),
                frag_B_ret[stage],
            )
            norm_A(kt, stage)

        frag_C.fill(0.0)
        load_k(0, 0)
        for kt in range_constexpr(k_tiles_local):
            cur = kt % 2
            if kt + 1 < k_tiles_local:
                load_k(kt + 1, (kt + 1) % 2)
            fx.gemm(
                tiled_mma,
                frag_C,
                frag_A[cur],
                frag_B[cur],
                frag_C,
                traversal_order=fx.GemmTraversalOrder.KNM,
            )
        # Raw f32 partial; SiLU / inject-split happen in the cross-K reduction.
        fx.copy(c_copy_atom, thr_copy_C.retile(frag_C), thr_copy_C.partition_S(gC))

    @flyc.jit
    def launch(
        r2: fx.Tensor,
        rrms: fx.Tensor,
        w: fx.Tensor,
        w_dn: fx.Tensor,
        partial: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        tokens = fx.Int32(fx.get_scalar(r2.shape[0]))
        grid_m = (tokens + block_m - 1) // block_m
        kernel(r2, rrms, w, w_dn, partial).launch(
            grid=(fx.Int64(grid_m), split_k, n_cblocks),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch


def _k1_auto_split_k(tokens: int, block_m: int, k_tiles: int, stages: int = 2) -> int:
    """Choose a K divisor that fills the GPU without excess partial traffic.

    Small token grids target up to 256 workgroups and cap split-K at 16.
    Grids with at least 48 token blocks use the decoupled path. Each K-slice
    keeps at least ``stages - 1`` tiles for the pipeline prologue.
    """
    grid_m = (tokens + block_m - 1) // block_m
    if grid_m >= 48:
        return 1
    target = max(1, min(16, 256 // grid_m))
    best = 1
    for d in range(1, target + 1):
        if k_tiles % d == 0 and k_tiles // d >= stages - 1:
            best = d
    return best


def _decouple_block_n(n_pad: int, preferred: int = 128) -> int:
    """Choose a common MMA-aligned N tile that divides the packed width."""
    for block_n in (128, 112, 96, 64, 48, 32, 16):
        if block_n <= preferred and n_pad % block_n == 0:
            return block_n
    return n_pad


def _pipe_lds_bytes(stages, block_m, block_n, block_k, w1_len, hc_count):
    """LDS of the async pipelines' ``Smem`` struct (16-byte aligned arrays)."""

    def _aligned(nbytes):
        return ((nbytes + 15) // 16) * 16

    return (
        _aligned(stages * block_m * block_k * 2)
        + _aligned(stages * block_n * block_k * 2)
        + _aligned(w1_len * 4)
        + _aligned(block_m * hc_count * 4)
    )


def _fit_pipe_tile(
    arch,
    n_pad,
    block_n,
    block_m,
    block_k,
    stages,
    m_waves,
    n_waves,
    mma_n,
    w_len,
    hc_count,
    stage_w,
):
    """Largest ``(block_n, n_waves, w_lds)`` the async-LDS pipeline can run.

    ``block_n`` is the preferred N tile. Candidates are MMA-aligned divisors of
    ``n_pad`` no wider than it, whose A/B tiles are whole async loads and whose
    LDS (A/B panels, staged ``(1+w)``, rrms scratch) fits the device. A shared
    unfolded ``(1+w)`` (``stage_w``) is staged when some tile fits beside it;
    otherwise ``w_lds=False`` makes the builder read it from global memory.
    """
    cap = get_lds_capacity_bytes(arch)
    # gfx942 issues 32-bit async loads, gfx950 128-bit ones.
    async_vec = 2 if arch == "gfx942" else CHUNK
    start = min(block_n, n_pad) // mma_n * mma_n
    cands = [bn for bn in range(start, 0, -mma_n) if n_pad % bn == 0]
    for w_lds in (True, False) if stage_w else (True,):
        staged = w_lds and stage_w
        for bn in cands:
            nw = n_waves
            while nw > 1 and bn % (nw * mma_n):
                nw //= 2
            threads = m_waves * nw * WAVE
            if bn % (nw * mma_n) or (block_m * block_k) % (threads * async_vec):
                continue
            if (bn * block_k) % (threads * async_vec):
                continue
            if staged and w_len % threads:
                continue
            w1_len = w_len if staged else 1
            lds = _pipe_lds_bytes(stages, block_m, bn, block_k, w1_len, hc_count)
            if lds <= cap:
                return bn, nw, w_lds
    raise ValueError(
        f"no K1 pipeline N tile fits {cap} B LDS on {arch}: n_pad={n_pad}, "
        f"block_m={block_m}, block_k={block_k}, stages={stages}"
    )


def flydsl_k1_combine_norm_down(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection: torch.Tensor,
    norm_weight: torch.Tensor,
    w_dn_merged: torch.Tensor,
    lowrank: int,
    hc_count: int,
    eps: float,
    r2_out: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    rrms_out: torch.Tensor | None = None,
    block_m: int | None = None,
    block_k: int | None = None,
    m_waves: int = 1,
    n_waves: int = 4,
    stages: int | None = None,
    split_k: int | str = "auto",
    dn_block_n: int | None = None,
    dn_block_m: int | None = None,
    dn_m_waves: int | None = None,
    dn_n_waves: int | None = None,
    sk_block_m: int | None = None,
    fold_w: bool = False,
    gemm_pad: int | None = None,
    use_tuned: bool = True,
    stream: torch.cuda.Stream | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused K1: combine + grouped-RMSNorm + down.

    Combines ``block_output`` (gated by ``injection``) into ``residual``, forms
    the normalized ``xn`` in-flight (never materialized), and returns the packed
    down+inject output.

    ``split_k>1`` emits f32 K-slice partials and reduces them before SiLU.
    Its reduction order is accurate to the oracle but not bit-exact with a
    single-workgroup GEMM. ``split_k=1`` selects the decoupled async-LDS path;
    ``"auto"`` selects between them from the token grid.

    Returns ``(r2, out)``: ``r2`` is the combined residual ``[M, hidden]`` (must
    be stored per the contract); ``out`` is the packed ``[M, n_pad]`` with
    columns ``[0, lowrank)`` the SiLU'd bottleneck and ``[lowrank, lowrank+hc)``
    the raw next-injection logits (slice with
    :func:`~...common.split_down_inject`).

    Caller-provided ``r2_out`` (bf16 ``[P, hidden]``), ``out`` (bf16
    ``[P, n_pad]``) and ``rrms_out`` (f32 ``[P, hc_count]``) must be contiguous
    on the input device, where ``P`` is ``gemm_pad`` (default: ``M``), and
    must not overlap the inputs or each other. ``stages`` must be at least 2.
    """
    require_supported_arch()
    validate_k1_inputs(
        residual, block_output, injection, norm_weight, w_dn_merged, lowrank, hc_count
    )
    tokens, hidden = residual.shape
    if gemm_pad is not None and gemm_pad < tokens:
        raise ValueError(f"gemm_pad={gemm_pad} must be >= tokens={tokens}")
    if stages is not None and stages < 2:
        raise ValueError(f"stages={stages} must be >= 2 (async-LDS pipeline)")
    rows = tokens if gemm_pad is None else gemm_pad
    dev = residual.device
    for name, t, shape, dtype in (
        ("r2_out", r2_out, (rows, hidden), torch.bfloat16),
        ("out", out, (rows, w_dn_merged.shape[0]), torch.bfloat16),
        ("rrms_out", rrms_out, (rows, hc_count), torch.float32),
    ):
        if t is not None:
            check_tensor(name, t, shape, dev, dtype=dtype)
    check_no_overlap(
        {"r2_out": r2_out, "out": out, "rrms_out": rrms_out},
        {
            "residual": residual,
            "block_output": block_output,
            "injection": injection,
            "norm_weight": norm_weight,
            "w_dn_merged": w_dn_merged,
        },
    )
    ctx, stream = stream_scope(stream, residual.device)
    with ctx:
        return _k1_combine_norm_down(
            residual,
            block_output,
            injection,
            norm_weight,
            w_dn_merged,
            lowrank,
            hc_count,
            eps,
            stream,
            r2_out=r2_out,
            out=out,
            rrms_out=rrms_out,
            block_m=block_m,
            block_k=block_k,
            m_waves=m_waves,
            n_waves=n_waves,
            stages=stages,
            split_k=split_k,
            dn_block_n=dn_block_n,
            dn_block_m=dn_block_m,
            dn_m_waves=dn_m_waves,
            dn_n_waves=dn_n_waves,
            sk_block_m=sk_block_m,
            fold_w=fold_w,
            gemm_pad=gemm_pad,
            use_tuned=use_tuned,
        )


def validate_k1_inputs(
    residual: torch.Tensor,
    block_output: torch.Tensor | None,
    injection: torch.Tensor | None,
    norm_weight: torch.Tensor,
    w_dn_merged: torch.Tensor,
    lowrank: int,
    hc_count: int,
) -> None:
    """Shape/dtype/layout contract shared by the skinny and tiled K1 paths.

    ``block_output``/``injection`` are ``None`` for the no-combine (mix-only)
    variant. ``injection`` may be a column slice with unit inner stride.
    ``hidden / hc_count`` must be a multiple of
    :data:`~.common.STREAM_DIM_ALIGN`. The tiled path additionally needs a
    multiple of 16 ``w_dn_merged`` rows; the skinny path checks its own widths.
    """
    if residual.ndim != 2 or residual.dtype != torch.bfloat16:
        raise ValueError("residual must be a 2-D bf16 tensor")
    if not residual.is_contiguous():
        raise ValueError("residual must be row-major contiguous")
    tokens, hidden = residual.shape
    stream_dim = check_hidden(hidden, hc_count)
    dev = residual.device
    if (block_output is None) != (injection is None):
        raise ValueError("block_output and injection must both be given or omitted")
    if block_output is not None:
        check_tensor("block_output", block_output, (tokens, stream_dim), dev)
        check_tensor("injection", injection, (tokens, hc_count), dev, contiguous=False)
        if injection.stride(-1) != 1:
            raise ValueError("injection must have a contiguous inner dimension")
    check_norm_weight(norm_weight, stream_dim, hidden, dev)
    if w_dn_merged.ndim != 2:
        raise ValueError("w_dn_merged must be 2-D")
    n_pad = w_dn_merged.shape[0]
    check_tensor("w_dn_merged", w_dn_merged, (n_pad, hidden), dev)
    if n_pad < lowrank:
        raise ValueError(f"w_dn_merged has {n_pad} rows < lowrank={lowrank}")


@lru_cache(maxsize=256)
def _k1_tuned_plan(arch, gemm_tokens, n_pad, hidden, stream_dim):
    """Tuned K1 plan, or ``None`` when the entry cannot tile this shape.

    The table is keyed by token count and packed width only, so an entry
    measured for one hidden size may not divide another.
    """
    plan = k1_plan(arch, gemm_tokens, n_pad)
    if plan is None:
        return None
    block_k = int(plan.get("block_k", 64))
    if hidden % block_k or stream_dim % block_k:
        return None
    k_tiles = hidden // block_k
    stages = int(plan.get("stages", 2))
    split_k = 1 if plan.get("method") == "decouple" else int(plan.get("split_k", 1))
    if k_tiles % split_k or k_tiles // split_k < stages - 1:
        return None
    return plan


def _k1_combine_norm_down(
    residual: torch.Tensor,
    block_output: torch.Tensor | None,
    injection: torch.Tensor | None,
    norm_weight: torch.Tensor,
    w_dn_merged: torch.Tensor,
    lowrank: int,
    hc_count: int,
    eps: float,
    stream: int,
    r2_out: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    rrms_out: torch.Tensor | None = None,
    block_m: int | None = None,
    block_k: int | None = None,
    m_waves: int = 1,
    n_waves: int = 4,
    stages: int | None = None,
    split_k: int | str = "auto",
    dn_block_n: int | None = None,
    dn_block_m: int | None = None,
    dn_m_waves: int | None = None,
    dn_n_waves: int | None = None,
    sk_block_m: int | None = None,
    fold_w: bool = False,
    gemm_pad: int | None = None,
    use_tuned: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """K1 body for validated inputs; ``stream`` is the raw handle of the current stream.

    ``block_output=None`` skips the combine: ``residual`` is normalized as-is
    and returned as ``r2`` when no row padding is needed.
    """
    combine = block_output is not None
    tokens, hidden = residual.shape
    n_pad = w_dn_merged.shape[0]
    # GEMM stages use padded rows, but the combine prologue reads only real
    # tokens. Padded outputs are independent per row and are discarded.
    assert gemm_pad is None or gemm_pad >= tokens, "gemm_pad must be >= tokens"
    gemm_tokens = tokens if gemm_pad is None else gemm_pad
    if gemm_tokens == 0:
        return (
            residual.new_empty(0, hidden) if r2_out is None else r2_out,
            residual.new_empty(0, n_pad) if out is None else out,
        )
    if n_pad % 16:
        raise ValueError(
            f"w_dn_merged rows={n_pad} must be a multiple of 16 for the tiled K1 "
            "(one MFMA N tile)"
        )
    # Configuration precedence is explicit argument, tuned plan, then heuristic.
    # Use the padded token count because it determines the GEMM launch shape.
    arch = arch_name(residual.device)
    stream_dim = hidden // hc_count
    plan = (
        _k1_tuned_plan(arch, gemm_tokens, n_pad, hidden, stream_dim)
        if use_tuned
        else None
    )
    if stages is None:
        stages = int(plan.get("stages", 2)) if plan is not None else 2
    if stages < 2:
        raise ValueError(f"stages={stages} must be >= 2 (async-LDS pipeline)")
    # Async-LDS loads require at least 32 rows for whole-thread coverage.
    if block_m is None:
        block_m = 32
    # gfx942 uses 32-bit async loads, so split-K uses register prefetch to avoid
    # multiplying load instructions. Other architectures use the async pipeline.
    # GR_FORCE_PIPE is a temporary diagnostic rollback switch, not a supported
    # tuning knob.
    _gfx942_regp_partial = arch == "gfx942" and not os.environ.get("GR_FORCE_PIPE")
    _plan_sk_block_m = None
    if plan is not None:
        if split_k == "auto":
            split_k = (
                1 if plan.get("method") == "decouple" else int(plan.get("split_k", 1))
            )
        if block_k is None:
            block_k = plan.get("block_k")
        if dn_block_n is None:
            dn_block_n = plan.get("dn_block_n")
        if dn_block_m is None:
            dn_block_m = plan.get("dn_block_m")
        if dn_m_waves is None:
            dn_m_waves = plan.get("dn_m_waves")
        if dn_n_waves is None:
            dn_n_waves = plan.get("dn_n_waves")
        _plan_sk_block_m = plan.get("sk_block_m")
    if block_k is None:
        block_k = 64
    # Resolve split-K after block_k because valid factors must divide k_tiles.
    k_tiles = hidden // block_k
    # Large token grids use 64-row partials; small grids use 32 rows for more
    # parallelism. Fall back when neither height divides the padded token count.
    if sk_block_m is None:
        sk_block_m = _plan_sk_block_m
    if sk_block_m is None:
        if gemm_tokens % 64 == 0 and gemm_tokens >= 1024:
            sk_block_m = 64
        elif gemm_tokens % 32 == 0:
            sk_block_m = 32
        else:
            sk_block_m = block_m
    # Split-K stores are not row-masked. A nearest-neighbour tuned plan can
    # select a tile height that does not divide a direct caller's token count.
    if gemm_tokens % sk_block_m:
        sk_block_m = 64 if gemm_tokens % 64 == 0 else block_m
    if gemm_tokens % sk_block_m:
        raise ValueError(
            f"tokens={gemm_tokens} must be a multiple of the {sk_block_m}-row K1 "
            f"tile; pass gemm_pad rounded up to a multiple of {sk_block_m}"
        )
    if split_k == "auto":
        split_k = _k1_auto_split_k(gemm_tokens, sk_block_m, k_tiles, stages)
    use_splitk = isinstance(split_k, int) and split_k > 1
    # split_k=1 separates the memory-bound combine/RMS prologue from the
    # pipelined down GEMM while keeping xn implicit.
    use_decouple = isinstance(split_k, int) and split_k == 1
    if stages >= 2 and not use_splitk:
        assert block_m * block_k >= (m_waves * n_waves * 64) * 8, (
            f"async-LDS pipeline (stages={stages}) needs block_m*block_k >= "
            f"block_threads*async_vec; got block_m={block_m}, block_k={block_k}"
        )
    # The async pipelines preload stages-1 K tiles (per K-slice for split-K).
    k_tiles_slice = k_tiles // split_k if use_splitk else k_tiles
    if (use_splitk or use_decouple) and k_tiles_slice < stages - 1:
        raise ValueError(
            f"stages={stages} needs at least {stages - 1} K tiles per slice; "
            f"hidden={hidden}, block_k={block_k} and split_k={split_k} give "
            f"{k_tiles_slice}"
        )
    w_len = norm_weight.numel()
    # The pipelines stage an unfolded shared (1+w) in LDS when it fits.
    stage_w = (not fold_w) and w_len != hidden
    assert (
        gemm_tokens % block_m == 0
    ), f"gemm_tokens={gemm_tokens} must be a multiple of block_m={block_m}"
    # Padded rows remain uninitialized. Every operation is row-independent, and
    # callers discard these rows, so no memset is needed.
    # The folded down weight already contains (1 + norm_weight), so this tensor
    # is never read by the down GEMM. Avoid launching a bf16->f32 conversion
    # whose result would only be passed through as an unused kernel argument.
    w = norm_weight if fold_w else norm_weight_f32(norm_weight)
    # Without a combine, the GEMM can read the residual itself unless it needs
    # padded rows past the caller's allocation.
    reuse_residual = not combine and r2_out is None and gemm_tokens == tokens
    if reuse_residual:
        r2_out = residual
    elif r2_out is None:
        r2_out = torch.empty(
            gemm_tokens, hidden, dtype=residual.dtype, device=residual.device
        )
    if out is None:
        out = torch.empty(
            gemm_tokens, n_pad, dtype=torch.bfloat16, device=residual.device
        )
    inj_stride = injection.stride(0) if combine and tokens > 1 else hc_count
    prologue = _build_combine_rms(
        hc_count,
        stream_dim,
        float(eps),
        0 if inj_stride == hc_count else inj_stride,
        combine=combine,
        write_r2=not reuse_residual,
    )
    # The no-combine prologue never reads block_output/injection; any tensor
    # satisfies the launch signature.
    prologue_args = (
        residual,
        block_output if combine else residual,
        injection if combine else residual,
        r2_out,
    )

    # K2 needs per-stream reciprocal RMS values to reconstruct xn from r2.
    # Caller buffers are validated by the public entry point.
    def _rrms_buf():
        if rrms_out is not None:
            return rrms_out
        return torch.empty(
            gemm_tokens, hc_count, dtype=torch.float32, device=residual.device
        )

    mma = mfma_bf16(arch)

    if use_splitk:
        from aiter.ops.flydsl.kernels.hyper_connection_gated_residual.common import (
            _build_reduce_silu,
        )

        assert (
            k_tiles % split_k == 0
        ), f"k_tiles={k_tiles} must be a multiple of split_k={split_k}"
        rrms = _rrms_buf()
        partial = torch.empty(
            split_k * gemm_tokens, n_pad, dtype=torch.float32, device=residual.device
        )
        # N-tiling increases parallelism on large grids. A tuned width may not
        # divide the final mixer's narrower output, so require exact divisibility.
        if dn_block_n is not None and n_pad % dn_block_n == 0:
            sk_bn = dn_block_n
        elif gemm_tokens >= 1024 and n_pad % 128 == 0:
            sk_bn = 128
        else:
            sk_bn = n_pad
        _sk_mw = dn_m_waves if dn_m_waves is not None else 2
        _sk_nw = dn_n_waves if dn_n_waves is not None else 2
        # Reduce N waves until each wave owns whole MMA columns.
        while _sk_nw > 1 and sk_bn % (_sk_nw * mma.mma_n):
            _sk_nw //= 2
        if not _gfx942_regp_partial:
            sk_bn, _sk_nw, sk_w_lds = _fit_pipe_tile(
                arch,
                n_pad,
                sk_bn,
                sk_block_m,
                block_k,
                stages,
                _sk_mw,
                _sk_nw,
                mma.mma_n,
                w_len,
                hc_count,
                stage_w,
            )
        if _gfx942_regp_partial:
            part = _build_down_norm_partial(
                hidden,
                n_pad,
                sk_bn,
                hc_count,
                w_len,
                sk_block_m,
                block_k,
                split_k,
                _sk_mw,
                _sk_nw,
                mma.mma_m,
                mma.mma_n,
                mma.mma_k,
                fold_w=fold_w,
            )
        else:
            part = _build_down_norm_partial_pipe(
                hidden,
                n_pad,
                sk_bn,
                hc_count,
                w_len,
                sk_block_m,
                block_k,
                split_k,
                _sk_mw,
                _sk_nw,
                stages,
                mma.mma_m,
                mma.mma_n,
                mma.mma_k,
                fold_w=fold_w,
                w_lds=sk_w_lds,
            )
        total = gemm_tokens * n_pad
        # f32 buffer loads are at most 128 bits.
        vec = 4 if total % (256 * 4) == 0 else 2
        reduce = _build_reduce_silu(total, split_k, lowrank, n_pad, hc_count, 256, vec)
        fx_stream = fx.Stream(stream)
        _run_compiled(prologue, *prologue_args, rrms, fx_stream)
        _run_compiled(part, r2_out, rrms, w, w_dn_merged, partial, fx_stream)
        _run_compiled(reduce, partial, out, fx_stream)
        return r2_out, out

    if use_decouple:
        rrms = _rrms_buf()
        # N-tiling trades smaller LDS panels and more workgroups for repeated r2
        # reads. Explicit or tuned dimensions override these fallback choices.
        _dn_bn = dn_block_n
        _dn_bm = dn_block_m
        if _dn_bn is None:
            _dn_bn = _decouple_block_n(n_pad)
        elif n_pad % _dn_bn:
            # A tuned tile may not divide a differently padded merged weight.
            _dn_bn = _decouple_block_n(n_pad, _dn_bn)
        # Stores are not row-masked, so the tile height must divide gemm_tokens.
        if _dn_bm is None or gemm_tokens % _dn_bm:
            _dn_bm = 64 if gemm_tokens % 64 == 0 else block_m
        assert gemm_tokens % _dn_bm == 0
        _dn_mw = dn_m_waves if dn_m_waves is not None else 2
        _dn_nw = dn_n_waves if dn_n_waves is not None else 2
        # A/B panels share LDS with the staged (1+w) and rrms scratch.
        _dn_bn, _dn_nw, dn_w_lds = _fit_pipe_tile(
            arch,
            n_pad,
            _dn_bn,
            _dn_bm,
            block_k,
            stages,
            _dn_mw,
            _dn_nw,
            mma.mma_n,
            w_len,
            hc_count,
            stage_w,
        )
        downpipe = _build_down_norm_pipe(
            hidden,
            n_pad,
            _dn_bn,
            lowrank,
            hc_count,
            w_len,
            _dn_bm,
            block_k,
            _dn_mw,
            _dn_nw,
            stages,
            mma.mma_m,
            mma.mma_n,
            mma.mma_k,
            fold_w=fold_w,
            w_lds=dn_w_lds,
        )
        fx_stream = fx.Stream(stream)
        _run_compiled(prologue, *prologue_args, rrms, fx_stream)
        _run_compiled(downpipe, r2_out, rrms, w, w_dn_merged, out, fx_stream)
        return r2_out, out

    # The two-stage implementation supports only split-K and decoupled plans.
    raise AssertionError(
        f"split_k={split_k!r} resolved to neither split-K nor decouple; this "
        "two-stage package supports split_k='auto'/1/>1 only"
    )
