# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dense BF16/FP16 GEMM for gfx950 at one wave per SIMD: C[M, N] = A[M, K] @ W[N, K].T.

Four waves of 64x96 per 128x192x64 tile (a 256-CU grid at M=16384, N=384 has exactly one tile per CU),
fed by an S-stage direct-to-LDS ring with one barrier per K tile. Each K tile is two 32-wide MFMA chunks;
a chunk's MFMAs run under the LDS reads of the next chunk and under half of a later tile's DMA, placed by a
sched_group_barrier plan derived from the tile's instruction counts:

- tile t's DMA starts right after tile t-S's barrier, when its stage is free, and finishes in the first
  chunk of tile t-S+1, so ten DMA issues spread over two chunks instead of bunching behind one (a lone
  wave stalls on every DMA issue the memory path cannot take yet);
- LDS reads go first in each chunk: without alias information a DMA cannot be scheduled above an earlier
  LDS read, and a plan that asks for it is dropped as a whole;
- the K tile reaches the DMA through the buffer's scalar offset, so the loop has no address VALU.

N and K are compile-time constants; M is dynamic. There is no split-K, bias or fp32 output.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl

from .gemm_a16w16_gfx950 import (
    GEMM_A16W16_DTYPE_BF16,
    GEMM_A16W16_DTYPE_FP16,
    make_gemm_a16w16_gfx950_param,
    make_gemm_ab_lds_layouts,
    make_gemm_ab_load_context,
)
from .gemm_a16w16_gfx950_utils import (
    GFX950_DMA_BYTES,
    BlockSwizzle,
    get_wave_lds_offset,
    swizzled_col_idx,
)
from .kernels_common import run_cached, set_slp_vectorize_once_used
from .tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
)

RING_TILE = {
    "block_m": 128,
    "block_n": 192,
    "block_k": 64,
    "stages": 4,
    "m_waves": 2,
    "n_waves": 2,
}
# Of each chunk's MFMA slots, the LDS reads take the first READ_SPAN_8THS / 8; the DMA issues share the rest.
READ_SPAN_8THS = 4


def _plan_slots(count, n_slots, span_8ths):
    return [i * n_slots * span_8ths // (8 * count) for i in range(count)]


@fx.struct
class GemmA16W16RingParam:
    in_dtype_id: fx.Constexpr[int]
    block_m: fx.Constexpr[int]
    block_n: fx.Constexpr[int]
    block_k: fx.Constexpr[int]
    stages: fx.Constexpr[int]
    m_waves: fx.Constexpr[int]
    n_waves: fx.Constexpr[int]
    group_m: fx.Constexpr[int]
    a_is_transposed: fx.Constexpr[bool]
    b_is_transposed: fx.Constexpr[bool]
    n: fx.Constexpr[int]
    k: fx.Constexpr[int]
    mma_m: fx.Constexpr[int]
    mma_n: fx.Constexpr[int]
    mma_k: fx.Constexpr[int]
    async_load_bytes: fx.Constexpr[int]
    in_data_bytes: fx.Constexpr[int]
    cshuffle_r2g_vec_size: fx.Constexpr[int]
    ldg_x_threads: fx.Constexpr[int]
    block_threads: fx.Constexpr[int]
    ldg_a_iters: fx.Constexpr[int]
    ldg_b_iters: fx.Constexpr[int]


def ring_config_error(n, k, block_m, block_n, block_k, stages, m_waves, n_waves):
    """None when the ring kernel supports this N, K and tile, else the reason."""
    if block_k != 64:
        return "block_k must be 64 (two 32-wide MFMA chunks per K tile)"
    if stages < 2:
        return "stages must be at least 2"
    if k % block_k or k // block_k < stages + 1:
        return (
            f"K={k} must be a multiple of {block_k} with at least {stages + 1} K tiles"
        )
    if n % 8:
        return f"N={n} must be a multiple of 8"
    try:
        make_gemm_a16w16_gfx950_param(
            GEMM_A16W16_DTYPE_BF16,
            GEMM_A16W16_DTYPE_BF16,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            stages=stages,
            m_waves=m_waves,
            n_waves=n_waves,
            b_is_transposed=True,
        )
    except (AssertionError, ValueError) as e:
        return str(e) or "invalid tile"
    return None


@functools.cache
def make_gemm_a16w16_ring_param(
    in_dtype_id: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    m_waves: int,
    n_waves: int,
    group_m: int,
) -> GemmA16W16RingParam:
    error = ring_config_error(n, k, block_m, block_n, block_k, stages, m_waves, n_waves)
    if error is not None:
        raise ValueError(f"unsupported gemm_a16w16_ring configuration: {error}")
    base = make_gemm_a16w16_gfx950_param(
        in_dtype_id,
        in_dtype_id,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        stages=stages,
        m_waves=m_waves,
        n_waves=n_waves,
        group_m=group_m,
        b_is_transposed=True,
    )
    return GemmA16W16RingParam(
        in_dtype_id=in_dtype_id,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        stages=stages,
        m_waves=m_waves,
        n_waves=n_waves,
        group_m=group_m,
        a_is_transposed=False,
        b_is_transposed=True,
        n=n,
        k=k,
        mma_m=base.mma_m,
        mma_n=base.mma_n,
        mma_k=base.mma_k,
        async_load_bytes=base.async_load_bytes,
        in_data_bytes=base.in_data_bytes,
        cshuffle_r2g_vec_size=base.cshuffle_r2g_vec_size,
        ldg_x_threads=base.ldg_x_threads,
        block_threads=base.block_threads,
        ldg_a_iters=base.ldg_a_iters,
        ldg_b_iters=base.ldg_b_iters,
    )


def make_gemm_a16w16_ring_kernel_name(param: GemmA16W16RingParam):
    dtype = "fp16" if param.in_dtype_id == GEMM_A16W16_DTYPE_FP16 else "bf16"
    return (
        f"hgemm_ring_{dtype}_t{param.block_m}x{param.block_n}x{param.block_k}x{param.stages}"
        f"_w{param.m_waves}x{param.n_waves}_gm{param.group_m}_n{param.n}_k{param.k}"
    )


@flyc.kernel
def gemm_a16w16_ring_gfx950_kernel(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    m: fx.Int32,
    tiled_mma: fx.TiledMma,
    param: GemmA16W16RingParam,
):
    n = param.n
    k = param.k
    block_m = param.block_m
    block_n = param.block_n
    block_k = param.block_k
    stages = param.stages
    block_threads = param.block_threads
    vec = param.cshuffle_r2g_vec_size
    elem_bytes = param.in_data_bytes
    dma_vec = param.async_load_bytes // elem_bytes
    k_tiles = k // block_k
    wave_m = block_m // param.m_waves // param.mma_m
    wave_n = block_n // param.n_waves // param.mma_n
    n_reads = wave_m + wave_n
    n_mfma = wave_m * wave_n
    n_dma = param.ldg_a_iters + param.ldg_b_iters
    half_dma = n_dma // 2
    elem_dtype = (
        fx.Float16
        if const_expr(param.in_dtype_id == GEMM_A16W16_DTYPE_FP16)
        else fx.BFloat16
    )

    tid = fx.thread_idx.x
    num_pid_m = (m + block_m - 1) // block_m
    num_pid_n = (n + block_n - 1) // block_n
    bid_m, bid_n = BlockSwizzle(
        NUM_XCDS=8, NUM_PIDS_THRESHOLD=256, GROUP_M=param.group_m
    ).swizzle(num_pid_m, num_pid_n, fx.block_idx.x)
    block_m_offset = bid_m * block_m
    block_n_offset = bid_n * block_n

    @fx.struct
    class SharedABStorage:
        a: fx.Array[elem_dtype, stages * block_m * block_k, 16]
        b: fx.Array[elem_dtype, stages * block_n * block_k, 16]

    @fx.union
    class SharedStorage:
        ab: SharedABStorage
        c: fx.Array[elem_dtype, block_m * block_n, 16]

    storage = fx.SharedAllocator().allocate(SharedStorage)
    smem_a = storage.ab.a.peek().ptr
    smem_b = storage.ab.b.peek().ptr
    smem_c = storage.c.peek().ptr

    a_buf = fx.rocdl.make_buffer_tensor(a, max_size=True)
    b_buf = fx.rocdl.make_buffer_tensor(b, max_size=True)
    out_buf = fx.rocdl.make_buffer_tensor(out, max_size=True)

    ctx = make_gemm_ab_load_context(elem_dtype, load_tid=tid, ks_begin=0, param=param)
    thr_mma = tiled_mma.thr_slice(tid)
    thr_copy_A = fx.make_tiled_copy_A(ctx.a_tiled_copy_atom, tiled_mma).get_slice(tid)
    thr_copy_B = fx.make_tiled_copy_B(ctx.b_tiled_copy_atom, tiled_mma).get_slice(tid)
    a_lds_layout, b_lds_layout = make_gemm_ab_lds_layouts(
        block_m, block_n, block_k, False, True
    )

    gC = fx.flat_divide(out_buf, (block_m, block_n))[None, None, bid_m, bid_n]
    frag_A = thr_mma.make_fragment_A(fx.make_view(smem_a, a_lds_layout))
    frag_B = thr_mma.make_fragment_B(fx.make_view(smem_b, b_lds_layout))
    frag_C = thr_mma.make_fragment_C(gC)
    frag_A_retile = thr_copy_A.retile(frag_A)
    frag_B_retile = thr_copy_B.retile(frag_B)
    frag_C.fill(0.0)

    wave_lds_offset = (
        fx.Int32(get_wave_lds_offset(tid, param.async_load_bytes)) // elem_bytes
    )
    # (LDS base, global buffer, LDS layout, tile rows, row bound, first row, DMA issues): B's issues first.
    operands = (
        (smem_b, b_buf, b_lds_layout, block_n, n, block_n_offset, param.ldg_b_iters),
        (smem_a, a_buf, a_lds_layout, block_m, m, block_m_offset, param.ldg_a_iters),
    )

    def dma(k_tile, stage, lo, hi):
        """DMA issues [lo, hi) of one K tile into an LDS stage."""
        slot = 0
        for lds_base, buf, lds_layout, rows, bound, row_offset, iters in operands:
            for i in range_constexpr(iters):
                if const_expr(lo <= slot < hi):
                    gtid = block_threads * i + tid
                    row = gtid // param.ldg_x_threads
                    col = gtid % param.ldg_x_threads * dma_vec
                    grow = row_offset + row
                    grow = (grow < bound).select(grow, 0)
                    col = swizzled_col_idx(row, col, lds_layout, block_k)
                    lds_ptr = (
                        lds_base
                        + stage * rows * block_k
                        + wave_lds_offset
                        + i * block_threads * dma_vec
                    )
                    rocdl.raw_ptr_buffer_load_async_lds(
                        fx.rocdl.get_buffer_rsrc(fx.get_iter(buf)),
                        fx.to_llvm_ptr(lds_ptr),
                        fx.Int32(GFX950_DMA_BYTES).ir_value(),
                        fx.Int32((grow * k + col) * elem_bytes).ir_value(),
                        fx.Int32(k_tile * block_k * elem_bytes).ir_value(),
                        fx.Int32(0).ir_value(),
                        aux=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0),
                    )
                slot += 1

    def lds_read(stage, chunk):
        sA = thr_copy_A.partition_S(
            fx.make_view(smem_a + stage * block_m * block_k, a_lds_layout)
        )
        sB = thr_copy_B.partition_S(
            fx.make_view(smem_b + stage * block_n * block_k, b_lds_layout)
        )
        fx.copy(
            ctx.b_s2r_copy_atom, sB[None, None, chunk], frag_B_retile[None, None, chunk]
        )
        fx.copy(
            ctx.a_s2r_copy_atom, sA[None, None, chunk], frag_A_retile[None, None, chunk]
        )

    def mfma(chunk):
        fx.gemm(
            tiled_mma,
            frag_C,
            frag_A[None, None, chunk],
            frag_B[None, None, chunk],
            frag_C,
            traversal_order=fx.GemmTraversalOrder.KNM,
        )

    def plan(n_dma_issues):
        read_at = _plan_slots(n_reads, n_mfma, READ_SPAN_8THS)
        first_dma = read_at[-1] + 1
        dma_at = [
            first_dma + s for s in _plan_slots(n_dma_issues, n_mfma - first_dma, 8)
        ]
        pending = 0
        for s in range_constexpr(n_mfma):
            reads = read_at.count(s)
            dmas = dma_at.count(s)
            if const_expr(reads or dmas):
                if const_expr(pending):
                    rocdl.sched_mfma(pending)
                    pending = 0
                if const_expr(reads):
                    rocdl.sched_dsrd(reads)
                if const_expr(dmas):
                    rocdl.sched_vmem(dmas)
            pending += 1
        rocdl.sched_mfma(pending)

    def k_tile_body(kt, dma_first, dma_second, last):
        stage = kt % stages
        rocdl.sched_barrier(0)
        lds_read(stage, 1)
        if const_expr(dma_first):
            dma(kt + stages - 1, (kt + stages - 1) % stages, half_dma, n_dma)
        # Closes tile kt+S-1's async group, so group g always holds K tile g.
        rocdl.asyncmark()
        mfma(0)
        plan(n_dma - half_dma if dma_first else 0)
        rocdl.sched_barrier(0)
        if const_expr(last):
            mfma(1)
        else:
            # After it, tile kt+1 is in LDS and no wave reads `stage` again: the DMA may refill it.
            rocdl.wait_asyncmark(stages - 2)
            rocdl.s_waitcnt(lgkmcnt=0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)
            lds_read((kt + 1) % stages, 0)
            if const_expr(dma_second):
                dma(kt + stages, stage, 0, half_dma)
            mfma(1)
            plan(half_dma if dma_second else 0)
            rocdl.sched_barrier(0)

    for s in range_constexpr(stages - 1):
        dma(s, s, 0, n_dma)
        rocdl.asyncmark()
    rocdl.wait_asyncmark(stages - 2)
    rocdl.s_barrier()
    lds_read(0, 0)
    dma(stages - 1, stages - 1, 0, half_dma)
    kt_end = fx.Int32(k_tiles)
    for kt in range(kt_end - stages):
        k_tile_body(kt, True, True, False)
    k_tile_body(k_tiles - stages, True, False, False)
    for kt in range(kt_end - stages + 1, kt_end - 1):
        k_tile_body(kt, False, False, False)
    k_tile_body(k_tiles - 1, False, False, True)

    c_lds_layout = fx.make_layout((block_m, block_n), (block_n, 1))
    sC = fx.make_view(smem_c, c_lds_layout)
    thr_cRow = thr_mma.partition_C(
        fx.make_view(0, fx.make_layout((block_m, block_n), (1, 0)))
    )
    thr_cCol = thr_mma.partition_C(
        fx.make_view(0, fx.make_layout((block_m, block_n), (0, 1)))
    )
    frag_C_out = fx.make_fragment_like(frag_C, elem_dtype)
    frag_C_out.store(frag_C.load().to(elem_dtype))
    # C overlays the A/B stages: every wave must be done with its last fragment reads.
    gpu.barrier()
    for i in range_constexpr(fx.size(frag_C_out.shape).unpack()):
        sC[fx.get_scalar(thr_cRow[i]), fx.get_scalar(thr_cCol[i])] = frag_C_out[i]
    gpu.barrier()

    x_threads = block_n // vec
    vectors = block_m * block_n // vec
    for i in range_constexpr((vectors + block_threads - 1) // block_threads):
        vector_idx = block_threads * i + tid
        if vector_idx < vectors:
            local_row = vector_idx // x_threads
            local_col = vector_idx % x_threads * vec
            global_row = block_m_offset + local_row
            global_col = block_n_offset + local_col
            if (global_row < m) and (global_col < n):
                c_vec = fx.ptr_load(
                    smem_c + local_row * block_n + local_col,
                    result_type=fx.Vector.make_type(vec, elem_dtype),
                )
                fx.ptr_store(c_vec, fx.get_iter(out) + global_row * n + global_col)


@flyc.jit
def gemm_a16w16_ring_gfx950(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    param: GemmA16W16RingParam,
    stream: fx.Stream = fx.Stream(None),  # noqa: B008
):
    m = fx.Int32(fx.get_scalar(a.shape[0]))
    elem_dtype = (
        fx.Float16
        if const_expr(param.in_dtype_id == GEMM_A16W16_DTYPE_FP16)
        else fx.BFloat16
    )
    mma_atom = fx.make_mma_atom(
        fx.rocdl.MFMA(param.mma_m, param.mma_n, param.mma_k, elem_dtype)
    )
    k_per_mfma_group = param.mma_k // 4
    tiled_mma = fx.make_tiled_mma(
        mma_atom,
        fx.make_layout((param.m_waves, param.n_waves, 1), (param.n_waves, 1, 0)),
        fx.make_tile(
            None,
            None,
            fx.make_layout((k_per_mfma_group, 4), (1, k_per_mfma_group)),
        ),
    )
    num_pid_m = (m + param.block_m - 1) // param.block_m
    num_pid_n = (param.n + param.block_n - 1) // param.block_n
    kernel = gemm_a16w16_ring_gfx950_kernel
    kernel._known_block_size = [param.block_threads, 1, 1]
    kernel._func.__name__ = make_gemm_a16w16_ring_kernel_name(param)
    kernel(out, a, b, m, tiled_mma, param).launch(
        grid=(num_pid_m * num_pid_n, 1, 1),
        block=(param.block_threads, 1, 1),
        stream=stream,
    )


gemm_a16w16_ring_gfx950.compile_hints = {
    "llvm_options": {
        "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
        "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    },
}
set_slp_vectorize_once_used(gemm_a16w16_ring_gfx950, False)


def gemm_a16w16_ring(
    a: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    block_m: int = RING_TILE["block_m"],
    block_n: int = RING_TILE["block_n"],
    block_k: int = RING_TILE["block_k"],
    stages: int = RING_TILE["stages"],
    m_waves: int = RING_TILE["m_waves"],
    n_waves: int = RING_TILE["n_waves"],
    group_m: int = 1,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """C[M, N] = A[M, K] @ W[N, K].T for contiguous BF16 or FP16 A and W; C has the input dtype.

    K must be a multiple of 64 with at least ``stages + 1`` K tiles and N a multiple of 8. Each
    distinct (N, K, tile) compiles its own kernel.
    """
    if a.dtype not in (torch.bfloat16, torch.float16) or w.dtype != a.dtype:
        raise ValueError("A and W must both be bfloat16 or both float16")
    if a.ndim != 2 or w.ndim != 2 or not a.is_contiguous() or not w.is_contiguous():
        raise ValueError("A and W must be contiguous 2-D tensors")
    for t in (a, w):
        if t.data_ptr() % GFX950_DMA_BYTES:
            raise ValueError("A and W must be 16-byte aligned")
    m, k = a.shape
    n = w.shape[0]
    if w.shape[1] != k:
        raise ValueError(f"shape mismatch: A {tuple(a.shape)}, W {tuple(w.shape)}")
    if out is None:
        out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    elif out.shape != (m, n) or out.dtype != a.dtype or not out.is_contiguous():
        raise ValueError("out must be a contiguous [M, N] tensor of the input dtype")
    if m == 0:
        return out
    in_dtype_id = (
        GEMM_A16W16_DTYPE_FP16 if a.dtype is torch.float16 else GEMM_A16W16_DTYPE_BF16
    )
    param = make_gemm_a16w16_ring_param(
        in_dtype_id, n, k, block_m, block_n, block_k, stages, m_waves, n_waves, group_m
    )
    if stream is None:
        stream = torch.cuda.current_stream(a.device)
    # The kernel reads B as a logical [K, N]: W[N, K] is its column-major form.
    dispatch_args = (
        flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=1),
        flyc.from_dlpack(a).mark_layout_dynamic(leading_dim=1),
        flyc.from_dlpack(w.t()).mark_layout_dynamic(leading_dim=0),
        param,
        stream,
    )
    run_cached(
        gemm_a16w16_ring_gfx950,
        *dispatch_args,
        constexpr_param=param,
        compiler=flyc.compile,
        dispatch_args=dispatch_args,
    )
    return out
