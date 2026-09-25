# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# Do NOT add `from __future__ import annotations`: PEP 563 stringifies the
# annotations and defeats flydsl's runtime-arg detection in the JIT cache key.

"""K1 of the two-stage Gated-Residual kernel (SILOTIGER-1042): combine +
grouped-RMSNorm + down GEMM, fused so ``xn`` never touches HBM.

:func:`flydsl_k1_combine_norm_down` is the entry point. A high-occupancy
combine+RMS prologue (:func:`_build_combine_rms`) emits the combined residual
``r2`` (bf16, a required output) plus the tiny per-stream ``rrms``; a down+inject
GEMM then re-forms

    xn[m, c] = r2[m, c] * rrms[m, stream(c)] * (1 + w[c])

*inside the A-load* (never materializing ``xn``) and contracts it against the
merged ``[n_pad, hidden]`` down+inject weight, emitting the packed
``silu'd bottleneck | raw injection`` output. The K reduction is dispatched by
token count (``§6.3``/``§6.4``): split-K partials at small/mid M
(:func:`_build_down_norm_partial_pipe`), a decoupled async-LDS pipeline at large
M (:func:`_build_down_norm_pipe`), and an MMA-free GEMV two-stage at decode M on
non-gfx950 (:func:`flydsl_k1k2_skinny_decode`).

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

from aiter.ops.flydsl.kernels.act import _sigmoid_f32
from aiter.ops.flydsl.kernels.tensor_shim import GTensor, _run_compiled
from aiter.ops.flydsl.kernels.hyper_connection_gated_residual.common import ab_k_perm, arch_name, mfma_bf16
from aiter.ops.flydsl.kernels.hyper_connection_gated_residual.tuned import k1_plan

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
    _skip_norm: bool = False,
):
    """Async-LDS pipelined down+inject GEMM with the norm folded into the A-load.

    ``_skip_norm`` is a measurement-only switch (produces wrong output): it drops
    the A-transform so the kernel is a plain GEMM+epilogue, isolating the
    normalization tax (§6.8) as the delta vs the normal path.

    The GEMM half of the *decoupled* large-M path (§6.4): a high-occupancy
    ``_build_combine_rms`` prologue emits ``r2`` + ``rrms`` separately, then this
    kernel forms ``xn = r2*rrms*(1+w)`` in the A-load and reduces the full K with
    the vendored ``gemm_a16w16_gfx950`` async global->LDS staged pipeline. It is
    ``_build_k1_pipe`` minus the fused combine/RMS prologue and the ``y`` panel
    (only ``r2`` and ``B`` are staged) -- so the memory-bound RMS reduction no
    longer runs pinned to this kernel's low (LDS-limited) occupancy, and ``xn`` is
    still never materialized.

    N-tiled (§6.7): the ``[n_pad]`` output width is split into ``n_pad//block_n``
    column blocks over ``grid.y``, mirroring ``down_inject_best``'s ``block_n=64``.
    Small B tiles shrink the LDS panel (higher occupancy) and give more workgroups
    -- the mid-M ~1.85x the full-width (block_n=n_pad) tile left on the table.
    """
    assert n_pad % block_n == 0, f"n_pad={n_pad} must be a multiple of block_n={block_n}"
    n_cblocks = n_pad // block_n
    if mma_k == 16:  # gfx942/CDNA3 -> HC-internal adapted GEMM (see package copy)
        from aiter.ops.flydsl.kernels.hyper_connection_gated_residual.gemm_a16w16_gfx942 import (
        GEMM_A16W16_DTYPE_BF16,
        AsyncLoadTile,
        async_load_to_lds,
        make_gemm_a16w16_gfx950_param,
        make_gemm_ab_load_context,
        make_gemm_ab_lds_layouts,
    )
    else:  # gfx950/CDNA4 -> vendored aiter GEMM (rides upstream)
        from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 import (
        GEMM_A16W16_DTYPE_BF16,
        AsyncLoadTile,
        async_load_to_lds,
        make_gemm_a16w16_gfx950_param,
        make_gemm_ab_load_context,
        make_gemm_ab_lds_layouts,
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
    assert block_m * block_k >= block_threads * 8, (
        "async-LDS pipe needs block_m*block_k >= block_threads*async_vec"
    )
    assert w_len % block_threads == 0
    k_tiles = hidden // block_k
    w_stage_iters = w_len // block_threads
    k_group = mma_k // 4
    # Stage this M-tile's rrms ([block_m, hc]) in LDS once (§6.7): the A-transform
    # otherwise re-loads rrms from global for every fragment element on every
    # k-tile. Only when the tile divides the workgroup evenly (it does for the
    # decouple's block_m=64, hc=4, 256 threads); else fall back to per-element.
    stage_rrms = (block_m * hc_count) % block_threads == 0
    rr_stage_iters = (block_m * hc_count) // block_threads if stage_rrms else 0

    param = make_gemm_a16w16_gfx950_param(
        in_dtype_id=GEMM_A16W16_DTYPE_BF16,
        out_dtype_id=GEMM_A16W16_DTYPE_BF16,
        block_m=block_m, block_n=block_n, block_k=block_k, stages=stages,
        split_k=1, m_waves=m_waves, n_waves=n_waves, k_waves=1,
        a_is_transposed=False, b_is_transposed=True,
        mma_m=mma_m, mma_n=mma_n, mma_k=mma_k,
    )
    ldg_a_iters = param.ldg_a_iters
    ldg_b_iters = param.ldg_b_iters

    @flyc.kernel(
        name=f"gr_downnormpipe_hc{hc_count}_h{hidden}_n{block_n}_s{silu_cols}"
        f"_bm{block_m}_bk{block_k}_nw{n_waves}_st{stages}",
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
        rrms_g = GTensor(rrms, fx.Float32, (1, hc_count))
        w_g = GTensor(w, fx.Float32, (1, w_len))

        @fx.struct
        class Smem:
            a: fx.Array[fx.BFloat16, stages * block_m * block_k, 16]
            b: fx.Array[fx.BFloat16, stages * block_n * block_k, 16]
            w1: fx.Array[fx.Float32, w_len, 16]
            rr: fx.Array[fx.Float32, block_m * hc_count, 16]

        smem = fx.SharedAllocator().allocate(Smem)
        smem_a = smem.a.peek().ptr
        smem_b = smem.b.peek().ptr
        sW1 = fx.make_view(smem.w1.peek().ptr, fx.make_layout((w_len,), (1,)))
        sRrms = fx.make_view(
            smem.rr.peek().ptr, fx.make_layout((block_m, hc_count), (hc_count, 1))
        )
        if const_expr(not fold_w):
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
        thr_copy_A = fx.make_tiled_copy_A(ctx.a_tiled_copy_atom, tiled_mma).get_slice(tid)
        thr_copy_B = fx.make_tiled_copy_B(ctx.b_tiled_copy_atom, tiled_mma).get_slice(tid)

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
                context=ctx, load_iters=ldg_a_iters, is_k_major=False,
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
                context=ctx, load_iters=ldg_b_iters, is_k_major=False,
            )

        scale_frag = fx.make_fragment_like(frag_A, fx.Float32)

        def transform(kt):
            # frag_A holds the r2 tile; normalize into xn: widen, * rrms * (1+w),
            # re-round to bf16 (matching _build_down_norm.norm_A). The per-element
            # rrms/w gathers fill a scale *fragment* (A's layout), so the normalize
            # is one packed vector multiply va*scale instead of a_elems scalar muls
            # -- the down GEMM is VALU-bound on this (§6.7e).
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
                scales.append(rr if const_expr(fold_w) else rr * sW1[col % w_len])
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
                tiled_mma, frag_C, frag_A, frag_B, frag_C,
                traversal_order=fx.GemmTraversalOrder.KNM,
            )

        frag_C.fill(0.0)
        for stage in range_constexpr(stages - 1):
            load_b(stage, stage)
            load_a(stage, stage)
            rocdl.asyncmark()
        # NOTE (api-stability): rocdl.sched_barrier / rocdl.s_barrier used in this
        # mainloop are UNSTABLE FlyDSL APIs (not in fx.rocdl.__all__). Intentional:
        # they order the async global->LDS DMAs against the MFMA in the software
        # pipeline and have no stable wrapper. Revisit if a stable primitive appears.
        rocdl.sched_barrier(0)

        main_loop_end = k_tiles - (stages - 1)
        for k_tile in range(0, main_loop_end):
            current_stage = k_tile % stages
            write_stage = (current_stage + stages - 1) % stages
            rocdl.wait_asyncmark(stages - 2)
            rocdl.s_barrier()
            load_b(k_tile + (stages - 1), write_stage)
            load_a(k_tile + (stages - 1), write_stage)
            rocdl.asyncmark()
            compute_stage(current_stage, k_tile)
            rocdl.sched_barrier(0)

        current_stage = main_loop_end % stages
        for s in range_constexpr(stages - 1):
            rocdl.wait_asyncmark(stages - 2 - s)
            rocdl.s_barrier()
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
            silu_v = sv * _sigmoid_f32(sv)
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
        frag_out.store(fx.Vector.from_elements(out_vals, dtype=fx.Float32).to(fx.BFloat16))
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
):
    """Async-LDS pipelined split-K partial (§6.7c): the decouple pipe body, but
    each ``grid.y`` workgroup reduces a disjoint K-slice and writes a raw f32
    partial ``[split_k*M, n_pad]`` (SiLU deferred to the reduce). Combines split-K
    parallelism (mid-M fill) with the async global->LDS pipeline + N-tile +
    LDS-rrms + fold_w -- replacing the register-prefetch ``_build_down_norm_partial``.
    """
    if mma_k == 16:  # gfx942/CDNA3 -> HC-internal adapted GEMM (see package copy)
        from aiter.ops.flydsl.kernels.hyper_connection_gated_residual.gemm_a16w16_gfx942 import (
        GEMM_A16W16_DTYPE_BF16,
        AsyncLoadTile,
        async_load_to_lds,
        make_gemm_a16w16_gfx950_param,
        make_gemm_ab_load_context,
        make_gemm_ab_lds_layouts,
    )
    else:  # gfx950/CDNA4 -> vendored aiter GEMM (rides upstream)
        from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 import (
        GEMM_A16W16_DTYPE_BF16,
        AsyncLoadTile,
        async_load_to_lds,
        make_gemm_a16w16_gfx950_param,
        make_gemm_ab_load_context,
        make_gemm_ab_lds_layouts,
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
    assert w_len % block_threads == 0
    k_tiles = hidden // block_k
    assert k_tiles % split_k == 0
    k_tiles_local = k_tiles // split_k
    assert k_tiles_local >= stages - 1, "split-K slice too short for the pipeline"
    w_stage_iters = w_len // block_threads
    k_group = mma_k // 4
    stage_rrms = (block_m * hc_count) % block_threads == 0
    rr_stage_iters = (block_m * hc_count) // block_threads if stage_rrms else 0

    param = make_gemm_a16w16_gfx950_param(
        in_dtype_id=GEMM_A16W16_DTYPE_BF16,
        out_dtype_id=GEMM_A16W16_DTYPE_BF16,
        block_m=block_m, block_n=block_n, block_k=block_k, stages=stages,
        split_k=1, m_waves=m_waves, n_waves=n_waves, k_waves=1,
        a_is_transposed=False, b_is_transposed=True,
        mma_m=mma_m, mma_n=mma_n, mma_k=mma_k,
    )
    ldg_a_iters = param.ldg_a_iters
    ldg_b_iters = param.ldg_b_iters

    @flyc.kernel(
        name=f"gr_downnormpartpipe_hc{hc_count}_h{hidden}_np{n_pad}_n{block_n}"
        f"_bm{block_m}_bk{block_k}_sk{split_k}_st{stages}",
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
        rrms_g = GTensor(rrms, fx.Float32, (1, hc_count))
        w_g = GTensor(w, fx.Float32, (1, w_len))

        @fx.struct
        class Smem:
            a: fx.Array[fx.BFloat16, stages * block_m * block_k, 16]
            b: fx.Array[fx.BFloat16, stages * block_n * block_k, 16]
            w1: fx.Array[fx.Float32, w_len, 16]
            rr: fx.Array[fx.Float32, block_m * hc_count, 16]

        smem = fx.SharedAllocator().allocate(Smem)
        smem_a = smem.a.peek().ptr
        smem_b = smem.b.peek().ptr
        sW1 = fx.make_view(smem.w1.peek().ptr, fx.make_layout((w_len,), (1,)))
        sRrms = fx.make_view(
            smem.rr.peek().ptr, fx.make_layout((block_m, hc_count), (hc_count, 1))
        )
        if const_expr(not fold_w):
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
        thr_copy_A = fx.make_tiled_copy_A(ctx.a_tiled_copy_atom, tiled_mma).get_slice(tid)
        thr_copy_B = fx.make_tiled_copy_B(ctx.b_tiled_copy_atom, tiled_mma).get_slice(tid)

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
                context=ctx, load_iters=ldg_a_iters, is_k_major=False,
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
                context=ctx, load_iters=ldg_b_iters, is_k_major=False,
            )

        scale_frag = fx.make_fragment_like(frag_A, fx.Float32)

        def transform(gkt):
            # Vectorized normalize (§6.7e): fill a scale fragment, one packed
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
                scales.append(rr if const_expr(fold_w) else rr * sW1[col % w_len])
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
                tiled_mma, frag_C, frag_A, frag_B, frag_C,
                traversal_order=fx.GemmTraversalOrder.KNM,
            )

        frag_C.fill(0.0)
        for stage in range_constexpr(stages - 1):
            load_b(k_base + stage, stage)
            load_a(k_base + stage, stage)
            rocdl.asyncmark()
        # NOTE (api-stability): rocdl.sched_barrier / rocdl.s_barrier used in this
        # mainloop are UNSTABLE FlyDSL APIs (not in fx.rocdl.__all__). Intentional:
        # they order the async global->LDS DMAs against the MFMA in the software
        # pipeline and have no stable wrapper. Revisit if a stable primitive appears.
        rocdl.sched_barrier(0)

        main_loop_end = k_tiles_local - (stages - 1)
        for kt in range(0, main_loop_end):
            current_stage = kt % stages
            write_stage = (current_stage + stages - 1) % stages
            rocdl.wait_asyncmark(stages - 2)
            rocdl.s_barrier()
            load_b(k_base + kt + (stages - 1), write_stage)
            load_a(k_base + kt + (stages - 1), write_stage)
            rocdl.asyncmark()
            compute_stage(current_stage, k_base + kt)
            rocdl.sched_barrier(0)

        current_stage = main_loop_end % stages
        for s in range_constexpr(stages - 1):
            rocdl.wait_asyncmark(stages - 2 - s)
            rocdl.s_barrier()
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
    shared :func:`down_silu._build_reduce_silu` sums the ``hc_count*spk`` blocks and
    applies ``silu(down/nr)`` (inject columns kept raw).
    """
    assert n_pad % waves_per_block == 0, (
        f"n_pad={n_pad} must be a multiple of waves_per_block={waves_per_block}"
    )
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

        r2_g = GTensor(r2, fx.BFloat16, (1, m_rows * hidden))
        rrms_g = GTensor(rrms, fx.Float32, (1, m_rows * hc_count))
        wdn_g = GTensor(w_dn, fx.BFloat16, (1, n_pad * hidden))
        part_g = GTensor(partial, fx.Float32, (1, hc_count * split_k_per_stream * m_rows * n_pad))

        rr = [rrms_g.load(m * hc_count + stream, vec_size=1) for m in range_constexpr(m_rows)]
        acc = [fx.Float32(0.0) for _ in range_constexpr(m_rows)]
        k_base = stream * stream_dim + sk * kslice + lane
        # MMA-free scalar GEMV: lanes stride the K-slice with 1-element bf16 loads
        # (coalesced across the wave). Intentional -- decode M<=DECODE_MAX_M is
        # launch/latency-bound, not bandwidth-bound, so we skip MMA/tiled-copy and
        # wide vector loads; revisit vec_size>1 only if decode turns bandwidth-bound.
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
):
    """MMA-free skinny (GEMV) K2 (up-GEMM + gated mean) for the decode regime.

    One wave owns one output channel ``c in [0, stream_dim)``. For each of the
    ``hc_count`` streams it computes ``gate = lora @ Wu[s*stream_dim+c].T`` as a
    lane-strided dot over ``lowrank`` (wave-reduced), re-forms
    ``xn = bf16(r2*rrms[m,s]*(1+w))`` in registers (never materialized, same order
    as :func:`fused._build_up_gate_mix_norm`), and accumulates
    ``sigmoid(gate)*xn``; the output is ``mean_s`` (``*1/hc_count``). No 64-row tile
    padding -- runs the true ``m_rows`` (M held in registers).
    """
    assert stream_dim % waves_per_block == 0
    assert lowrank % WAVE == 0, f"lowrank={lowrank} must be a multiple of WAVE={WAVE}"
    k_iters = lowrank // WAVE
    inv_hc = 1.0 / hc_count
    block_threads = waves_per_block * WAVE
    log2_wave = int(math.log2(WAVE))
    shared_w = w_len != hidden  # norm_weight is [stream_dim] (shared) vs [hidden]

    @flyc.kernel(
        name=f"gr_up_gemv_hc{hc_count}_hs{stream_dim}_r{lowrank}_m{m_rows}"
        f"_w{waves_per_block}",
        known_block_size=[block_threads, 1, 1],
    )
    def kernel(
        lora: fx.Tensor,  # [m_rows, lowrank] bf16
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

        lora_g = GTensor(lora, fx.BFloat16, (1, m_rows * lowrank))
        r2_g = GTensor(r2, fx.BFloat16, (1, m_rows * hidden))
        rrms_g = GTensor(rrms, fx.Float32, (1, m_rows * hc_count))
        w_g = GTensor(w, fx.Float32, (1, w_len))
        wup_g = GTensor(w_up, fx.BFloat16, (1, hidden * lowrank))
        out_g = GTensor(block_input, fx.BFloat16, (1, m_rows * stream_dim))

        w_shared = w_g.load(c, vec_size=1) if shared_w else None
        xacc = [fx.Float32(0.0) for _ in range_constexpr(m_rows)]
        # MMA-free scalar GEMV (see _build_down_gemv_partial): lane-strided 1-element
        # bf16 dot over lowrank, wave-reduced. Intentional for decode M<=DECODE_MAX_M
        # (launch-bound); no MMA/tiled-copy or wide vector loads here.
        for s in range_constexpr(hc_count):
            n = s * stream_dim + c
            g = [fx.Float32(0.0) for _ in range_constexpr(m_rows)]
            for i in range_constexpr(k_iters):
                r = lane + i * WAVE
                wu = fx.BFloat16(wup_g.load(n * lowrank + r, vec_size=1)).to(fx.Float32)
                for m in range_constexpr(m_rows):
                    lo = fx.BFloat16(lora_g.load(m * lowrank + r, vec_size=1)).to(fx.Float32)
                    g[m] = g[m] + lo * wu
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
                xacc[m] = xacc[m] + _sigmoid_f32(a) * xn
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

    ``total = tokens * n_pad`` and ``n_pad`` is 64-padded, so ``total`` is always a
    multiple of 64 -- the ``bt=16, vec=4`` (=64) candidate therefore always
    divides it, guaranteeing a vectorized path. This matters for the final mixer
    (``n_pad=lowrank=320=64*5``): with only ``bt>=64`` candidates the search would
    fall to ``vec=1``, which the vectorized reduce kernel cannot emit (it can't
    wrap a scalar load in a Vector). The sub-wavefront blocks are fine here -- the
    decode reduce is a tiny one-shot elementwise pass.
    """
    for vec in (4, 2, 1):
        for bt in (256, 192, 128, 64, 32, 16):
            if bt * vec <= total and total % (bt * vec) == 0:
                return bt, vec
    return 16, 1


# Decode skinny path holds m_rows accumulators per lane, so the GEMV bodies spill
# for m_rows >= 8 (rocprofv3: down_gemv M=8 ~134us, M=32 ~514us -- §13.13). It only
# wins at very small M (M<=4: ~17-26us, ~1.4-1.8x vs Triton); above that the padded
# split-K/pipe path is far better, so gate the skinny decode at 4.
DECODE_MAX_M = 4


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
    split_k_per_stream: int = 1,
    waves_per_block: int = 4,
    stream: torch.cuda.Stream | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """MMA-free skinny GEMV two-stage for decode M (combine+RMS -> down GEMV ->
    reduce/silu -> up GEMV + gated mean). No 64-row tile padding; ``xn`` never
    materialized. Returns the op contract ``(r2, block_input, inj_next)``.

    CONTRACT: ``w_down_merged`` MUST be the ``(1+w)``-folded merged weight (see
    :func:`op.fold_norm_weight`). The down GEMV bakes that in -- it forms
    ``xn = r2*rrms`` with **no** ``(1+w)`` multiply -- so passing an unfolded merged
    weight here produces silently wrong output (no shape/dtype tripwire catches it).
    The op only routes here when ``fold_w=True``; do not call it directly otherwise.
    """
    from aiter.ops.flydsl.kernels.hyper_connection_gated_residual.common import _build_reduce_silu

    tokens, hidden = residual.shape
    stream_dim = hidden // hc_count
    n_pad = w_down_merged.shape[0]
    # Can't verify "folded" numerically (no unfolded ref here), but pin the
    # checkable half of the contract so a wrong-tensor call fails loudly.
    assert w_down_merged.dtype == torch.bfloat16 and w_down_merged.is_contiguous(), (
        "skinny decode needs a bf16 contiguous merged down weight"
    )
    assert w_down_merged.shape == (n_pad, hidden), (
        f"w_down_merged {tuple(w_down_merged.shape)} must be (n_pad, {hidden})"
    )
    w = norm_weight.reshape(-1).float().contiguous()
    if stream is None:
        stream = torch.cuda.current_stream()
    dev = residual.device
    r2 = torch.empty(tokens, hidden, dtype=torch.bfloat16, device=dev)
    rrms = torch.empty(tokens, hc_count, dtype=torch.float32, device=dev)
    nb = hc_count * split_k_per_stream
    partial = torch.empty(nb * tokens, n_pad, dtype=torch.float32, device=dev)
    packed = torch.empty(tokens, n_pad, dtype=torch.bfloat16, device=dev)
    x = torch.empty(tokens, stream_dim, dtype=torch.bfloat16, device=dev)

    pro = _build_combine_rms(hc_count, stream_dim, float(eps))
    gd = _build_down_gemv_partial(
        hidden, n_pad, hc_count, stream_dim, tokens, split_k_per_stream, waves_per_block
    )
    total = tokens * n_pad
    bt, vec = _decode_reduce_params(total)
    red = _build_reduce_silu(total, nb, lowrank, n_pad, hc_count, bt, vec)
    gu = _build_up_gate_mix_gemv(
        hidden, stream_dim, hc_count, lowrank, w.numel(), tokens, waves_per_block
    )
    fxs = fx.Stream(stream)
    _run_compiled(pro, residual, block_output, injection, r2, rrms, fxs)
    _run_compiled(gd, r2, rrms, w_down_merged, partial, fxs)
    _run_compiled(red, partial, packed, fxs)
    lora = packed[:, :lowrank].contiguous()
    _run_compiled(gu, lora, r2, rrms, w, w_up, x, fxs)
    inj_next = packed[:, lowrank:lowrank + hc_count].contiguous() if need_inj else None
    return r2, x, inj_next


@lru_cache(maxsize=16)
def _build_combine_rms(hc_count: int, stream_dim: int, eps: float):
    """Prologue for the split-K K1 path: combine + per-stream RMS, emit r2 + rrms.

    A trimmed fork of :func:`combine_norm._build` that stops after the reduction:
    it writes the combined residual ``r2`` (bf16, required output) and the
    per-row-per-stream ``rrms`` (f32, tiny ``[M, hc]``) instead of the full
    ``[M, hidden]`` ``xn``. The split-K down-norm GEMM then re-forms ``xn`` in its
    A-load from ``r2`` + ``rrms``, so ``xn`` is still never materialized -- while
    letting the GEMM split the K reduction freely (rrms no longer lives inside it).
    """
    hidden = hc_count * stream_dim
    assert stream_dim % (WAVE * CHUNK) == 0
    chunks = stream_dim // (WAVE * CHUNK)
    inv_hc = 1.0 / hc_count
    inv_hs = 1.0 / stream_dim
    log2_wave = int(math.log2(WAVE))

    @flyc.kernel(
        name=f"gr_combine_rms_hc{hc_count}_hs{stream_dim}", known_block_size=[WAVE, 1, 1]
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
        r = GTensor(residual, fx.BFloat16, (1, hidden))
        y = GTensor(block_output, fx.BFloat16, (1, stream_dim))
        inj = GTensor(injection, fx.BFloat16, (1, hc_count))
        r2 = GTensor(r2_out, fx.BFloat16, (1, hidden))
        rrms_g = GTensor(rrms_out, fx.Float32, (1, hc_count))

        inj_val = fx.BFloat16(inj.load(tok * hc_count + stream, vec_size=1)).to(
            fx.Float32
        )
        gate = fx.Float32(2.0) * _sigmoid_f32(inj_val * fx.Float32(inv_hc))
        row_r = tok * hidden + stream * stream_dim
        row_y = tok * stream_dim
        sq = fx.Float32(0.0)
        for c in range_constexpr(chunks):
            off = c * (WAVE * CHUNK) + tid * CHUNK
            rv = fx.Vector(r.load(row_r + off, vec_size=CHUNK)).to(fx.Float32)
            yv = fx.Vector(y.load(row_y + off, vec_size=CHUNK)).to(fx.Float32)
            comb = [rv[i] + yv[i] * gate for i in range_constexpr(CHUNK)]
            comb_bf16 = fx.Vector.from_elements(comb, dtype=fx.Float32).to(fx.BFloat16)
            r2.store(row_r + off, comb_bf16)
            comb_r = comb_bf16.to(fx.Float32)
            for i in range_constexpr(CHUNK):
                sq = sq + comb_r[i] * comb_r[i]
        for sh in range_constexpr(log2_wave):
            sq = sq + fx.gpu.shuffle_xor(sq, WAVE // (2 << sh), WAVE)
        rstd = fmath.rsqrt(sq * fx.Float32(inv_hs) + fx.Float32(eps))
        # All lanes hold the same reduced rstd, so all 64 write the same value to the
        # same address -- harmless (hardware coalesces the identical writes). A lane-0
        # `if tid == 0:` guard was tried and reverted: a conditional store here needs
        # a local @flyc.jit dispatch (frontend side-effect-branch rule) -- not worth
        # it for one tiny [M, hc] f32 write.
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

    Forks :func:`_build_down_norm`: ``split_k`` workgroups per token block
    (``grid.y``) each reduce a disjoint K-slice of the ``hidden`` contraction,
    forming ``xn = r2 * rrms * (1+w)`` in registers per k-tile, and write a raw
    f32 partial ``[split_k*M, n_pad]``. SiLU / inject-split is deferred to the
    cross-K reduction (:func:`down_silu._build_reduce_silu`), which must follow the
    full reduction. This is the parallelism lever K1 lacked: it multiplies the
    workgroup count by ``split_k`` so the machine fills at small/mid M.

    N-tiled + LDS-rrms (§6.7a): the ``[n_pad]`` width is split into
    ``n_pad//block_n`` column blocks over ``grid.z``, and this M-tile's ``rrms``
    is staged in LDS once instead of a per-element global load in ``norm_A``.
    """
    assert n_pad % block_n == 0, f"n_pad={n_pad} must be a multiple of block_n={block_n}"
    n_cblocks = n_pad // block_n
    stream_dim = hidden // hc_count
    block_threads = m_waves * n_waves * 64
    assert block_m % (m_waves * mma_m) == 0
    assert block_n % (n_waves * mma_n) == 0
    assert block_k % mma_k == 0 and hidden % block_k == 0
    assert stream_dim % block_k == 0
    k_tiles = hidden // block_k
    assert k_tiles % split_k == 0, (
        f"k_tiles={k_tiles} must be a multiple of split_k={split_k}"
    )
    k_tiles_local = k_tiles // split_k
    k_group = mma_k // 4
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
        rrms_g = GTensor(rrms, fx.Float32, (1, hc_count))
        w_g = GTensor(w, fx.Float32, (1, w_len))

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
            fx.copy(copy_atom, thr_copy_A.partition_S(gA[None, None, k_base + kt]), frag_A_ret[stage])
            fx.copy(copy_atom, thr_copy_B.partition_S(gB[None, None, k_base + kt]), frag_B_ret[stage])
            norm_A(kt, stage)

        frag_C.fill(0.0)
        load_k(0, 0)
        for kt in range_constexpr(k_tiles_local):
            cur = kt % 2
            if kt + 1 < k_tiles_local:
                load_k(kt + 1, (kt + 1) % 2)
            fx.gemm(
                tiled_mma, frag_C, frag_A[cur], frag_B[cur], frag_C,
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


def _k1_auto_split_k(tokens: int, block_m: int, k_tiles: int) -> int:
    """Split-K factor from the token grid (§6.3, measured on the shipped shape).

    Split-K only helps while the ``M/block_m`` token grid can't fill the CUs. The
    swept optimum (``_tune_splitk.py`` / ``_bench_decouple.py``, block_m=64) is:
    aim for ~256 workgroups (``split_k ~= 256/grid_m``, capped at 16), snapped
    down to a divisor of ``k_tiles``; and once the plain grid is large enough
    (``grid_m >= 48``, i.e. ~3072+ tokens at block_m=64) split-K's extra partials
    traffic loses to the *decoupled pipelined* path (return 1 -> §6.4), which
    beats both the fused pipe and split-K there. E.g. block_m=64: 256->16,
    512->16, 1024->16, 2048->8, 4096->1 (decouple), 8192->1 (decouple).
    """
    grid_m = (tokens + block_m - 1) // block_m
    if grid_m >= 48:
        return 1
    target = max(1, min(16, 256 // grid_m))
    best = 1
    for d in range(1, target + 1):
        if k_tiles % d == 0:
            best = d
    return best


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
    stages: int = 2,
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
    """Fused K1 (SILOTIGER-1042): combine + grouped-RMSNorm + down.

    Combines ``block_output`` (gated by ``injection``) into ``residual``, forms
    the normalized ``xn`` in-flight (never materialized), and returns the packed
    down+inject output.

    ``split_k`` (default ``"auto"``) trades the single-launch fusion for the
    parallelism the fused body lacks at small/mid M (§6.3): a combine+RMS
    prologue emits ``r2`` + ``rrms``, ``split_k`` workgroups per token block each
    reduce a K-slice of a down-norm GEMM (``xn`` re-formed in the A-load, still
    never materialized), and a reduction sums the partials + SiLU. This is
    ~3-6x faster at M<=2048 but, because the f32 K-reduction order differs, is
    oracle-accurate rather than bit-exact vs the single-workgroup body. ``"auto"``
    uses split-K where it wins (small/mid M) and the decoupled async-LDS pipe
    (``split_k=1``, §6.4) elsewhere; explicit ``split_k>1``/``1`` force those
    paths. The bit-exact monolithic pipe (``split_k=0``) is not implemented in
    this package.

    Returns ``(r2, out)``: ``r2`` is the combined residual ``[M, hidden]`` (must
    be stored per the contract); ``out`` is the packed ``[M, n_pad]`` with
    columns ``[0, lowrank)`` the SiLU'd bottleneck and ``[lowrank, lowrank+hc)``
    the raw next-injection logits (slice with
    :func:`~...down_silu.split_down_inject`).
    """
    assert residual.dtype == torch.bfloat16 and residual.is_contiguous()
    assert block_output.dtype == torch.bfloat16 and block_output.is_contiguous()
    assert injection.dtype == torch.bfloat16 and injection.is_contiguous()
    assert w_dn_merged.dtype == torch.bfloat16 and w_dn_merged.is_contiguous()
    tokens, hidden = residual.shape
    # Default tile height. The async-LDS pipeline (stages>=2, §6.2) needs
    # block_m*block_k >= block_threads*async_vec (block_m>=32 here) for the async
    # loads to be whole-thread-covered, and pipe@32 beats the non-pipe body at
    # every M, so the pipe default is 32. The rolled single-buffer fallback
    # (stages<2) keeps the token-adaptive height (block_m=16 wins for small/mid
    # M, 32 for large; crossover between 4096 and 8192 tokens -- SILOTIGER-1042
    # §5e). An explicit block_m always wins.
    if block_m is None:
        block_m = 32 if stages >= 2 else (16 if tokens <= 4096 else 32)
    # Tail path (SILOTIGER-1042 low-M): when the true token count is not a tile
    # multiple, the GEMM stages run over a padded row count ``gemm_tokens`` while
    # the combine prologue still reads only the true ``tokens`` input rows (it is
    # one-workgroup-per-token, no block_m constraint) and writes r2/rrms[:tokens];
    # the zeroed pad rows [tokens:gemm_tokens] flow through down/K2 producing
    # discardable zero output. Removes the caller-side pad memcpy (§ low-M Phase 1).
    assert gemm_pad is None or gemm_pad >= tokens, "gemm_pad must be >= tokens"
    gemm_tokens = tokens if gemm_pad is None else gemm_pad
    # Data-driven config (tuned_configs.json "k1" table): fill any dimension the
    # caller left unset -- precedence explicit arg > tuned plan > heuristic. An
    # absent entry -> plan is None -> pure heuristic (behavior-preserving). Keyed
    # on gemm_tokens so a padded low-M tail resolves like its padded size.
    arch = arch_name(residual.device)
    plan = k1_plan(arch, gemm_tokens) if use_tuned else None
    # gfx942 (§13.9): the vendored async-LDS pipe now lowers on CDNA3 via the
    # 32-bit buffer_load...lds DMA (gemm_a16w16_gfx950 async width is arch-aware),
    # so the decouple/monolithic *pipe* is enabled (it wins at large M, 8192 K1
    # 711->602us). But the mid-M *split-K* partial still uses the register-prefetch
    # body (``_build_down_norm_partial``): the 32-bit split-K pipe's 4x load count
    # loses to it there (2048: 182 vs 237us). So only the split-K partial builder
    # is arch-gated; decouple/monolithic use the pipe on both archs.
    #
    # gfx950/CDNA4 (128-bit async DMA) is the only arch that prefers the split-K
    # pipe; gfx942 is the only other supported arch (see common.mfma_bf16), so gate
    # explicitly on gfx942 rather than "!= gfx950" -- a future arch then defaults
    # to the pipe consciously, not by omission. Set GR_FORCE_PIPE=1 to force the
    # split-K pipe on gfx942 (A/B testing the 32-bit pipe vs the register body).
    _gfx942_regp_partial = arch == "gfx942" and not os.environ.get("GR_FORCE_PIPE")
    _plan_sk_block_m = None
    if plan is not None:
        if split_k == "auto":
            split_k = 1 if plan.get("method") == "decouple" else int(plan.get("split_k", 1))
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
    # Resolve split-K (§6.3). Split-K trades the single-launch fusion for the
    # parallelism the fused body lacks at small/mid M: a combine+RMS prologue
    # emits r2 + rrms, then `split_k` workgroups per token block each reduce a
    # K-slice of a down-norm GEMM (xn re-formed in the A-load from r2+rrms, so xn
    # is still never materialized), and a reduction sums the partials + SiLU. NOT
    # bit-exact vs the single-workgroup body (f32 reduction order differs) --
    # validated against the oracle -- so it is opt-in (default off).
    k_tiles = hidden // block_k
    # Split-K partial tile height (§6.3 sweep), independent of the pipe's
    # block_m: block_m=64 wins at M>=1024 (and stretches the split-K advantage
    # out to 4096), but tiny M wants more/smaller blocks so 32 wins at M<=512.
    # Fall back to divisibility-preserving values when 64/32 don't tile tokens.
    if sk_block_m is None:
        sk_block_m = _plan_sk_block_m
    if sk_block_m is None:
        if gemm_tokens % 64 == 0 and gemm_tokens >= 1024:
            sk_block_m = 64
        elif gemm_tokens % 32 == 0:
            sk_block_m = 32
        else:
            sk_block_m = block_m
    if split_k == "auto":
        split_k = _k1_auto_split_k(gemm_tokens, sk_block_m, k_tiles)
    use_splitk = isinstance(split_k, int) and split_k > 1
    # split_k == 1 (explicit): decoupled path -- high-occupancy combine+RMS
    # prologue (r2+rrms) + a *pipelined* down-norm GEMM (§6.4). Decouples the
    # memory-bound prologue from the low-occupancy GEMM without the un-pipelined
    # partial's penalty. xn is still never materialized.
    use_decouple = isinstance(split_k, int) and split_k == 1
    if stages >= 2 and not use_splitk:
        assert block_m * block_k >= (m_waves * n_waves * 64) * 8, (
            f"async-LDS pipeline (stages={stages}) needs block_m*block_k >= "
            f"block_threads*async_vec; got block_m={block_m}, block_k={block_k}"
        )
    stream_dim = hidden // hc_count
    n_pad = w_dn_merged.shape[0]
    assert w_dn_merged.shape == (n_pad, hidden)
    assert block_output.shape == (tokens, stream_dim)
    assert injection.shape == (tokens, hc_count)
    w_len = norm_weight.numel()
    assert w_len in (stream_dim, hidden), (
        f"norm_weight must be [{stream_dim}] (shared) or [{hidden}], got {w_len}"
    )
    assert gemm_tokens % block_m == 0, (
        f"gemm_tokens={gemm_tokens} must be a multiple of block_m={block_m}"
    )
    # GEMM-facing buffers are sized to gemm_tokens; the true-M prologue fills rows
    # [:tokens] and the pad rows [tokens:gemm_tokens] stay uninitialized -- they
    # flow through down/K2 as garbage but every op here is per-token (RMS, GEMM
    # rows, gated mean over streams), so pad rows never contaminate a real row and
    # the caller slices them off. torch.empty (no memset) keeps decode M cheap.
    w = norm_weight.reshape(-1).float().contiguous()
    if r2_out is None:
        r2_out = torch.empty(gemm_tokens, hidden, dtype=residual.dtype, device=residual.device)
    if out is None:
        out = torch.empty(gemm_tokens, n_pad, dtype=torch.bfloat16, device=residual.device)
    if stream is None:
        stream = torch.cuda.current_stream()
    # rrms is the per-stream 1/rms K2 needs to re-form xn from r2 (see
    # flydsl_up_gate_mix_norm). The combine_rms-based paths (split-K / decouple,
    # which "auto" always selects) fill it; the single-kernel pipe path keeps
    # rrms in LDS only, so rrms_out is left untouched there.
    def _rrms_buf():
        if rrms_out is not None:
            assert rrms_out.shape == (gemm_tokens, hc_count) and rrms_out.dtype == torch.float32
            return rrms_out
        return torch.empty(gemm_tokens, hc_count, dtype=torch.float32, device=residual.device)

    mma = mfma_bf16(arch)

    if use_splitk:
        from aiter.ops.flydsl.kernels.hyper_connection_gated_residual.common import (
            _build_reduce_silu,
        )
        assert k_tiles % split_k == 0, (
            f"k_tiles={k_tiles} must be a multiple of split_k={split_k}"
        )
        rrms = _rrms_buf()
        partial = torch.empty(
            split_k * gemm_tokens, n_pad, dtype=torch.float32, device=residual.device
        )
        prologue = _build_combine_rms(hc_count, stream_dim, float(eps))
        # N-tile the split-K partial (§6.7a) only where it pays: at M>=1024
        # (sk_block_m=64, so LDS-rrms is active and the extra n-blocks don't
        # over-subscribe). Tiny M keeps the full-width tile (block_n=n_pad).
        # A tuned/explicit block_n comes from the with-inject shape (n_pad=384); it
        # may not divide the final-mixer n_pad (w_inject=None -> n_pad=lowrank=320),
        # so accept it only when it divides, else fall back to a divisor.
        if dn_block_n is not None and n_pad % dn_block_n == 0:
            sk_bn = dn_block_n
        elif gemm_tokens >= 1024 and n_pad % 128 == 0:
            sk_bn = 128
        else:
            sk_bn = n_pad
        # Wave layout: m_waves=2,n_waves=2 (§6.7d joint sweep) is ~20-26% faster
        # than the default 1x4 for the pipelined split-K partial at mid M.
        _sk_mw = dn_m_waves if dn_m_waves is not None else 2
        _sk_nw = dn_n_waves if dn_n_waves is not None else 2
        if _gfx942_regp_partial:
            part = _build_down_norm_partial(
                hidden, n_pad, sk_bn, hc_count, w_len, sk_block_m, block_k, split_k,
                _sk_mw, _sk_nw, mma.mma_m, mma.mma_n, mma.mma_k, fold_w=fold_w,
            )
        else:
            part = _build_down_norm_partial_pipe(
                hidden, n_pad, sk_bn, hc_count, w_len, sk_block_m, block_k, split_k,
                _sk_mw, _sk_nw, stages, mma.mma_m, mma.mma_n, mma.mma_k, fold_w=fold_w,
            )
        total = gemm_tokens * n_pad
        # f32 partials: buffer loads are 128-bit max, so vec<=4 (v8f32 won't isel).
        vec = 4 if total % (256 * 4) == 0 else 2
        reduce = _build_reduce_silu(total, split_k, lowrank, n_pad, hc_count, 256, vec)
        fx_stream = fx.Stream(stream)
        _run_compiled(prologue, residual, block_output, injection, r2_out, rrms, fx_stream)
        _run_compiled(part, r2_out, rrms, w, w_dn_merged, partial, fx_stream)
        _run_compiled(reduce, partial, out, fx_stream)
        return r2_out, out

    if use_decouple:
        rrms = _rrms_buf()
        prologue = _build_combine_rms(hc_count, stream_dim, float(eps))
        # N-tile the down GEMM (§6.7): block_n=64 (down_inject_best-style) shrinks
        # the B panel (higher occupancy) + adds workgroups -- a mid-M win. But at
        # large M the 6x n-blocks over-subscribe and re-read the A(r2) tile, so
        # widen block_n and raise block_m there to keep the workgroup count sane.
        # (dn_block_n/dn_block_m override the adaptive choice for tuning.)
        # Swept optimum for the decouple regime (grid_m>=48, i.e. ~4096+ tokens):
        # block_n=128 (3 n-tiles) + block_m=64 balances the workgroup count against
        # the per-n-tile A(r2) re-read -- beats both the full-width tile (bn=n_pad)
        # and the over-subscribed bn64 at every decouple size (§6.7).
        _dn_bn = dn_block_n
        _dn_bm = dn_block_m
        if _dn_bn is None:
            _dn_bn = 128 if n_pad % 128 == 0 else (64 if n_pad % 64 == 0 else n_pad)
        elif n_pad % _dn_bn:
            # tuned/explicit block_n (from the with-inject n_pad=384) may not divide
            # the final-mixer n_pad (w_inject=None -> n_pad=lowrank); use a divisor.
            _dn_bn = 64 if n_pad % 64 == 0 else n_pad
        if _dn_bm is None:
            _dn_bm = 64 if gemm_tokens % 64 == 0 else block_m
        # Wave layout: the joint sweep (§6.7d) found m_waves=2,n_waves=2 (same 256
        # threads, more balanced MMA tiling than the default 1x4) is ~10% faster
        # for the decouple down at large M -- a cross-dimension interaction the
        # per-dimension tuning missed. dn_m/n_waves override for tuning.
        _dn_mw = dn_m_waves if dn_m_waves is not None else 2
        _dn_nw = dn_n_waves if dn_n_waves is not None else 2
        downpipe = _build_down_norm_pipe(
            hidden, n_pad, _dn_bn, lowrank, hc_count, w_len, _dn_bm, block_k,
            _dn_mw, _dn_nw, stages, mma.mma_m, mma.mma_n, mma.mma_k,
            fold_w=fold_w,
        )
        fx_stream = fx.Stream(stream)
        _run_compiled(prologue, residual, block_output, injection, r2_out, rrms, fx_stream)
        _run_compiled(downpipe, r2_out, rrms, w, w_dn_merged, out, fx_stream)
        return r2_out, out

    # Only the auto split-K / decouple paths ship here. "auto" always resolves
    # split_k to >=1, so one of the two branches above returned. The bit-exact
    # monolithic K1 (split_k=0) is not implemented in this package.
    raise AssertionError(
        f"split_k={split_k!r} resolved to neither split-K nor decouple; this "
        "two-stage package supports split_k='auto'/1/>1 only "
        "(bit-exact split_k=0 is not implemented here)"
    )
