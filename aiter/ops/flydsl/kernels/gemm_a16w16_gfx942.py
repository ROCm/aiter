# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Native bf16 (a16w16, f32-accumulate) GEMM for gfx942 / CDNA3 (MI300X).

C[M, N] = A[M, K] @ B[K, N], nn.Linear ``nt`` convention (A row-major [M,K],
B column-major [K,N] i.e. the ``w.t()`` view of a [N,K] weight). Only ``nt`` is
supported: the transposed-operand LDS reads need CDNA4's ``ds_read_tr``.

CDNA3 specifics:
- bf16 ``16x16x16`` MFMA (gfx950 is ``16x16x32``); the tiled-MMA K layout uses
  ``ab_k_perm`` so a 128-bit LDS->reg read pairs with two MFMA K-groups.
- 64 KB LDS budget.
- Global->LDS movement is either 32-bit async DMA (``async_load_bytes=4``) or
  ``buffered_load``: a 128-bit ``buffer_load`` to registers then ``ds_write``.

Main-loop pipelines (selected by the param, one per compiled kernel; they live
in ``gemm_a16w16_gfx942_mainloop``, shared with fused kernels):
- ``pingpong``: one LDS buffer with the MMA operand set streamed in two K
  halves; the default for most shapes.
- ``buffered``: buffered_load with one or two LDS buffers.
- ``async``: async DMA with ``stages`` LDS buffers; ``stages >= 3`` adds a
  register double buffer.
Epilogues: optional K-wave reduction, split-K (global atomics or a workspace
reduced by the last split), then a direct register->global store or an LDS
cshuffle store.

Configs come from the tuned table (``aiter/configs/flydsl_gemm_a16w16_gfx942_tuned.csv``)
and fall back to a cost-model search for shapes that are not in it.

Entry points are ``gemm_a16w16`` and ``flydsl_hgemm``; the table is tuned by
``op_tests/flydsl_tests/tune_gfx942_csv.py``. TunedGemm's FlyDSL catalog
(``gemm_a16w16_policy``) and AOT compilation do not cover gfx942: their
``flydsl_hgemm_*`` kernel names cannot encode the pipeline flags
(``pingpong``, ``buffered_load``, ``direct_store``, ``split_k_reduce``).
"""

import contextlib
import csv
import functools
import inspect
import itertools
import os
import types

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl

from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.utility.graph_alloc import persistent_alloc

from .gemm_a16w16_gfx942_mainloop import (
    GEMM_A16W16_DTYPE_BF16,
    GEMM_A16W16_DTYPE_FP32,
    GemmA16W16Gfx942Param,
    gfx942_ab_lds_elems,
    gfx942_k_wave_split,
    gfx942_red_lds_elems,
    make_gemm_a16w16_gfx942_param,
    make_gfx942_mainloop,
    make_gfx942_tiled_mma,
    run_gfx942_mainloop,
)
from .gemm_a16w16_gfx950 import write_cshuffle_vec_to_global
from .gemm_a16w16_gfx950_utils import BlockSwizzle, SplitKProtocol
from .kernels_common import run_cached

# The main loop (gemm_a16w16_gfx942_mainloop) documents its unstable rocdl
# calls. SplitKProtocol.zero_c (gfx950 helper, atomic split-K only) adds raw
# llvm.InlineAsmOp stores and tensor_shim.buf_base_i64.

GFX942_NUM_CUS = 304  # MI300X CUs: workgroups per round in the cost model
# CDNA3 buffer cache policy sc0 | sc1: system coherent, bypasses the per-XCD L2.
CPOL_SC0_SC1 = 1 | 16
# Split-K semaphore and signal slots per launch: one per output tile.
SPLIT_K_MAX_TILES = 1024
# AMD buffer resource descriptors encode the accessible byte count in 32 bits.
FLAT_BUFFER_LIMIT_BYTES = 1 << 32

# Config keys: the make_gemm_a16w16_gfx942_param arguments except the dtypes
# (set by gemm_a16w16) and the MFMA shape (fixed on gfx942), with their
# signature defaults. Unset keys of a config take these values.
CONFIG_DEFAULTS = {
    name: arg.default
    for name, arg in inspect.signature(make_gemm_a16w16_gfx942_param).parameters.items()
    if name not in ("in_dtype_id", "out_dtype_id", "mma_m", "mma_n", "mma_k")
}


def make_gemm_a16w16_gfx942_kernel_name(param: GemmA16W16Gfx942Param) -> str:
    out_suffix = "_fp32" if param.out_dtype_id == GEMM_A16W16_DTYPE_FP32 else ""
    sk = ("_ksr" if param.split_k_reduce else "_ksd") if param.is_split_k else "_ks1"
    dma = (
        f"buf{param.async_load_bytes}"
        if param.buffered_load
        else str(param.async_load_bytes)
    )
    store = (
        "_r2g"
        if param.direct_store
        else (f"_c{param.c_chunks}" if param.c_chunks > 1 else "")
    )
    pipe = "_pp" if param.pingpong else ""
    return (
        f"hgemm_bf16{out_suffix}_gfx942_"
        f"t{param.block_m}x{param.block_n}x{param.block_k}x{param.stages}{sk}_"
        f"w{param.m_waves}x{param.n_waves}x{param.k_waves}_"
        f"gm{param.group_m}_dma{dma}{store}{pipe}"
    )


@flyc.jit
def _reduce_split_k(
    frag_C,
    owns_tile,
    tid,
    tid_mn,
    threads_per_k_slice,
    ks_idx,
    split_k,
    num_tiles,
    semaphore,
    workspace,
    smem_flag,
):
    """Sum the split-K partial tiles in the split that arrives last.

    Each split stores its fp32 partial tile in fragment order (128-bit,
    lane-interleaved, coalesced), bumps the tile's arrival counter, and the
    last split to arrive sums every partial. Partials are written and read with
    sc0 sc1 so they are coherent across XCDs without agent-scope fences: those
    lower to a full L2 writeback/invalidate that would evict every workgroup's
    A/B reuse. Returns (owns_tile, is_last) for the epilogue.
    """
    n_vec4 = const_expr(fx.size(frag_C.shape).unpack() // 4)
    tile_id = fx.block_idx.x
    # (4, lane, chunk, split, tile): each lane's fragment as 128-bit chunks,
    # lane-interleaved so a chunk index is one coalesced store per wave.
    ws_stride = 4 * threads_per_k_slice * n_vec4
    ws = fx.make_view(
        fx.get_iter(fx.rocdl.make_buffer_tensor(workspace, max_size=True)),
        fx.make_layout(
            (4, threads_per_k_slice, n_vec4, split_k, num_tiles),
            (1, 4, 4 * threads_per_k_slice, ws_stride, ws_stride * split_k),
        ),
    )
    ws_atom = fx.make_copy_atom(
        fx.rocdl.BufferCopy128b(cache_modifier=CPOL_SC0_SC1), fx.Float32
    )
    frag_chunks = fx.make_layout((4, n_vec4), (1, 4))

    if owns_tile:
        fx.copy(
            ws_atom,
            fx.make_view(fx.get_iter(frag_C), frag_chunks),
            ws[None, tid_mn, None, ks_idx, tile_id],
        )
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid == 0:
        # Agent scope: the arrival counter is shared across XCDs.
        sem_ptr = fx.add_offset(fx.get_iter(semaphore), tile_id)
        arrived = fx.llvm.atomic_add(
            sem_ptr,
            fx.Int32(1),
            syncscope=fx.rocdl.SyncScope.Agent,
            ordering=fx.AtomicOrdering.Monotonic,
        )
        fx.ptr_store(arrived, smem_flag)
        if arrived == split_k - 1:
            fx.llvm.generic_store(
                sem_ptr,
                fx.Int32(0),
                memory_order=fx.AtomicOrdering.Monotonic,
                syncscope=fx.rocdl.SyncScope.Agent,
            )
    gpu.barrier()
    is_last = fx.Int32(fx.ptr_load(smem_flag)) == split_k - 1
    owns_tile = owns_tile & is_last
    if is_last:
        peer = fx.make_fragment_like(frag_C)
        peer_chunks = fx.make_view(fx.get_iter(peer), frag_chunks)
        frag_C.fill(0.0)
        for split in range(0, split_k, 1):
            fx.copy(ws_atom, ws[None, tid_mn, None, split, tile_id], peer_chunks)
            frag_C.store(frag_C.load() + peer.load())
    return owns_tile, is_last


@flyc.jit
def _store_direct(frag_C, gC, thr_copy_C, c_r2g_atom, owns_tile, gated, is_fp32_output):
    """Store the MMA fragment straight to global memory, skipping the LDS
    staging and its barriers. Needs N to be a block_n multiple."""
    if const_expr(is_fp32_output):
        frag_C_out = frag_C
    else:
        frag_C_out = fx.make_fragment_like(frag_C, fx.BFloat16)
        frag_C_out.store(frag_C.load().to(fx.BFloat16))
    if const_expr(gated):
        c_src = thr_copy_C.retile(frag_C_out)
        c_dst = thr_copy_C.partition_S(gC)
        if owns_tile:
            fx.copy(c_r2g_atom, c_src, c_dst)
    else:
        fx.copy(
            c_r2g_atom,
            thr_copy_C.retile(frag_C_out),
            thr_copy_C.partition_S(gC),
        )


@flyc.jit
def _store_cshuffle(
    param: GemmA16W16Gfx942Param,
    frag_C,
    thr_copy_C_lds,
    smem_c,
    out,
    out_buf,
    tid,
    m,
    n,
    block_m_offset,
    block_n_offset,
    owns_tile,
    is_last,
    gated,
    is_split_k,
    is_fp32_output,
    splitk_protocol,
    expected_signal,
):
    """Stage the C tile through LDS for coalesced, boundary-masked stores.

    Atomic split-K always uses it. The tile is staged in the output dtype, so
    fp32 output is not rounded to bf16. A tile over 64 KB goes through LDS in
    ``c_chunks`` row chunks.
    """
    block_m = param.block_m
    block_n = param.block_n
    block_threads = param.block_threads
    c_chunks = param.c_chunks
    cshuffle_r2g_vec_size = param.cshuffle_r2g_vec_size
    if const_expr(is_fp32_output):
        c_dtype = fx.Float32
        frag_C_out = frag_C
        c_s_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    else:
        c_dtype = fx.BFloat16
        frag_C_out = fx.make_fragment_like(frag_C, fx.BFloat16)
        frag_C_out.store(frag_C.load().to(fx.BFloat16))
        c_s_atom = fx.make_copy_atom(fx.UniversalCopy16b(), fx.BFloat16)
    # The (value, M repeat, N repeat) partitions put chunk c's rows
    # [c * chunk_rows, (c + 1) * chunk_rows) in M repeats
    # [c * chunk_reps, (c + 1) * chunk_reps).
    chunk_reps = block_m // (param.mma_m * param.m_waves) // c_chunks
    chunk_rows = block_m // c_chunks
    c_lds_layout = fx.make_layout((chunk_rows, block_n), (block_n, 1))
    out_layout = fx.make_layout((m, n), (n, 1))
    frag_src = thr_copy_C_lds.retile(frag_C_out)
    sC_dst = thr_copy_C_lds.partition_D(fx.make_view(smem_c, c_lds_layout))
    cshuffle_r2g_x_threads = block_n // cshuffle_r2g_vec_size
    cshuffle_vectors = chunk_rows * block_n // cshuffle_r2g_vec_size
    cshuffle_iters = (cshuffle_vectors + block_threads - 1) // block_threads

    # Called inside each store branch: one offset hoisted above both branches
    # schedules the epilogue worse (2-3% on small-M split-K shapes).
    def c_lds_ptr(row, col):
        return smem_c + fx.get_scalar(fx.crd2idx((row, col), c_lds_layout))

    for chunk in range_constexpr(c_chunks):
        gpu.barrier()
        if const_expr(gated):
            if owns_tile:
                for j in range_constexpr(chunk_reps):
                    fx.copy(
                        c_s_atom,
                        frag_src[None, chunk * chunk_reps + j, None],
                        sC_dst[None, j, None],
                    )
        else:
            for j in range_constexpr(chunk_reps):
                fx.copy(
                    c_s_atom,
                    frag_src[None, chunk * chunk_reps + j, None],
                    sC_dst[None, j, None],
                )

        if const_expr(is_split_k and chunk == 0):
            splitk_protocol.wait_until_initialized(expected_signal)
        else:
            gpu.barrier()

        # Hand-indexed: row-major vectors go round-robin over all threads.
        # No TV layout expresses that when block_n // vec does not divide the
        # thread count (block_n 48, 96, 112, 160, 192); a tiled copy would
        # idle threads and add a pass.
        for i in range_constexpr(cshuffle_iters):
            vector_idx = block_threads * i + tid
            if vector_idx < cshuffle_vectors:
                local_row = vector_idx // cshuffle_r2g_x_threads
                local_col = vector_idx % cshuffle_r2g_x_threads * cshuffle_r2g_vec_size
                global_row = block_m_offset + chunk * chunk_rows + local_row
                global_col = block_n_offset + local_col
                global_offset = fx.get_scalar(
                    fx.crd2idx((global_row, global_col), out_layout)
                )
                in_rows = (global_row < m) and is_last
                if in_rows and (global_col + cshuffle_r2g_vec_size <= n):
                    c_vec = fx.ptr_load(
                        c_lds_ptr(local_row, local_col),
                        result_type=fx.Vector.make_type(cshuffle_r2g_vec_size, c_dtype),
                    )
                    write_cshuffle_vec_to_global(
                        out,
                        out_buf,
                        global_offset,
                        c_vec,
                        is_split_k,
                        is_fp32_output,
                    )
                # Ragged N: the vector straddling column n is stored element by
                # element, else it would spill into the next row.
                if (
                    in_rows
                    and (global_col < n)
                    and (global_col + cshuffle_r2g_vec_size > n)
                ):
                    for e in range_constexpr(cshuffle_r2g_vec_size):
                        if global_col + e < n:
                            c_elem = fx.ptr_load(
                                c_lds_ptr(local_row, local_col) + e,
                                result_type=fx.Vector.make_type(1, c_dtype),
                            )
                            write_cshuffle_vec_to_global(
                                out,
                                out_buf,
                                global_offset + e,
                                c_elem,
                                is_split_k,
                                is_fp32_output,
                            )


def gemm_a16w16_gfx942_kernel(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    semaphore: fx.Tensor,
    signal: fx.Tensor,
    workspace: fx.Tensor,
    m: fx.Int32,
    n: fx.Int32,
    k: fx.Int32,
    split_k: fx.Int32,
    working_k: fx.Int32,
    a_leading_stride: fx.Int32,
    b_leading_stride: fx.Int32,
    tiled_mma: fx.TiledMma,
    param: GemmA16W16Gfx942Param,
):
    # split_k_reduce: partials go to a workspace and the last split of each
    # tile reduces them; otherwise split-K accumulates with global atomics.
    split_k_reduce = param.split_k_reduce
    is_split_k = param.is_split_k and not split_k_reduce
    block_m = param.block_m
    block_n = param.block_n
    block_k = param.block_k
    block_threads = param.block_threads
    cshuffle_r2g_vec_size = param.cshuffle_r2g_vec_size
    direct_store = param.direct_store
    c_chunks = param.c_chunks
    k_waves = param.k_waves
    elem_dtype = fx.BFloat16
    global_output_dtype = (
        fx.Float32
        if const_expr(param.out_dtype_id == GEMM_A16W16_DTYPE_FP32)
        else fx.BFloat16
    )
    is_fp32_output = param.out_dtype_id == GEMM_A16W16_DTYPE_FP32

    if const_expr(is_split_k):
        splitk_protocol = SplitKProtocol(
            block_m,
            block_n,
            cshuffle_r2g_vec_size,
            param.out_data_bytes,
            block_threads,
            False,
        )
    else:
        splitk_protocol = None

    tid = fx.thread_idx.x
    k_wave_split = gfx942_k_wave_split(param, tid)
    num_pid_m = (m + block_m - 1) // block_m
    num_pid_n = (n + block_n - 1) // block_n
    # XCD-aware block remap, active when group_m > 0. NUM_PIDS_THRESHOLD=0
    # applies it to grids of any size (the grid must still divide across the
    # 8 XCDs): small narrow-N grids such as 32x6 blocks need it for L2 reuse.
    block_swizzle = BlockSwizzle(
        NUM_XCDS=8, NUM_PIDS_THRESHOLD=0, GROUP_M=param.group_m
    )
    bid_m, bid_n = block_swizzle.swizzle(num_pid_m, num_pid_n, fx.block_idx.x)
    ks_idx = fx.block_idx.y
    ks_begin = ks_idx * working_k
    ks_end = ks_begin + working_k
    ks_end = fx.min(ks_end, k)
    k_tiles = (ks_end - ks_begin) // block_k
    block_m_offset = bid_m * block_m
    block_n_offset = bid_n * block_n

    a_lds_elems, b_lds_elems = gfx942_ab_lds_elems(param)

    @fx.struct
    class SharedABStorage:
        a: fx.Array[elem_dtype, a_lds_elems, 16]
        b: fx.Array[elem_dtype, b_lds_elems, 16]

    @fx.union
    class SharedStorage:
        ab: SharedABStorage
        c: fx.Array[
            global_output_dtype,
            1 if direct_store else block_m // c_chunks * block_n,
            16,
        ]
        red: fx.Array[fx.Float32, max(1, gfx942_red_lds_elems(param)), 16]
        flag: fx.Array[fx.Int32, 4, 16]

    storage = fx.SharedAllocator().allocate(SharedStorage)
    smem_c = storage.c.peek().ptr
    smem_flag = storage.flag.peek().ptr

    # Direct stores rely on the descriptor's range check: rows past M fall
    # beyond the buffer end and are dropped, but columns past N would wrap
    # into the next row, so direct_store needs N to be a block_n multiple.
    out_buf = fx.rocdl.make_buffer_tensor(out, max_size=not direct_store)

    if const_expr(is_split_k):
        splitk_protocol.init(
            semaphore,
            signal,
            out,
            None,
            tid,
            ks_idx,
            m,
            n,
            block_m_offset,
            block_n_offset,
            global_output_dtype,
            fx.block_idx.x,
            n,
        )

    gC = fx.flat_divide(out_buf, (block_m, block_n))[None, None, bid_m, bid_n]
    ml = make_gfx942_mainloop(
        param,
        tiled_mma,
        tid=tid,
        k_wave_split=k_wave_split,
        a=a,
        b=b,
        m=m,
        n=n,
        k=k,
        a_leading_stride=a_leading_stride,
        b_leading_stride=b_leading_stride,
        bid_m=bid_m,
        bid_n=bid_n,
        ks_begin=ks_begin,
        k_tiles=k_tiles,
        c_tile=gC,
        smem_a=storage.ab.a.peek().ptr,
        smem_b=storage.ab.b.peek().ptr,
        smem_red=storage.red.peek().ptr,
    )
    frag_C = ml.frag_C
    tid_mn = ml.tid_mn
    threads_per_k_slice = ml.threads_per_k_slice

    thr_copy_C_direct = None
    c_r2g_atom = None
    if const_expr(direct_store):
        if const_expr(is_fp32_output):
            c_r2g_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        else:
            c_r2g_atom = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), elem_dtype)
        thr_copy_C_direct = fx.make_tiled_copy_C(c_r2g_atom, tiled_mma).get_slice(
            tid_mn
        )
    thr_copy_C_lds = fx.make_tiled_copy_C(
        fx.make_copy_atom(
            (
                fx.UniversalCopy32b()
                if const_expr(is_fp32_output)
                else fx.UniversalCopy16b()
            ),
            global_output_dtype,
        ),
        tiled_mma,
    ).get_slice(tid_mn)

    expected_signal = None
    if const_expr(is_split_k):
        splitk_protocol.zero_c()
        expected_signal = splitk_protocol.take_ticket(split_k)
        splitk_protocol.publish(expected_signal)

    run_gfx942_mainloop(ml)

    # Lanes that own the final tile: K wave 0, and with split_k_reduce only in
    # the split that arrives last for this tile.
    gated_epilogue = const_expr(k_waves > 1 or split_k_reduce)
    if const_expr(k_waves > 1):
        owns_tile = ml.k_wave_idx == 0
    else:
        owns_tile = fx.Int32(1) == 1
    is_last = fx.Int32(1) == 1
    if const_expr(split_k_reduce):
        owns_tile, is_last = _reduce_split_k(
            frag_C,
            owns_tile,
            tid,
            tid_mn,
            threads_per_k_slice,
            ks_idx,
            split_k,
            num_pid_m * num_pid_n,
            semaphore,
            workspace,
            smem_flag,
        )

    if const_expr(direct_store):
        _store_direct(
            frag_C,
            gC,
            thr_copy_C_direct,
            c_r2g_atom,
            owns_tile,
            gated_epilogue,
            is_fp32_output,
        )
    else:
        _store_cshuffle(
            param,
            frag_C,
            thr_copy_C_lds,
            smem_c,
            out,
            out_buf,
            tid,
            m,
            n,
            block_m_offset,
            block_n_offset,
            owns_tile,
            is_last,
            gated_epilogue,
            is_split_k,
            is_fp32_output,
            splitk_protocol,
            expected_signal,
        )


def named_gemm_a16w16_gfx942_kernel(name: str):
    """``gemm_a16w16_gfx942_kernel`` as a ``flyc.kernel`` named ``name`` for
    profilers. Each wraps a fresh copy of the function, since ``flyc.kernel``
    rewrites the code of the function it decorates.

    A plain function, not cached: FlyDSL's JIT cache key only follows plain
    functions and rejects mutated globals. It runs once per compile.
    """
    body = gemm_a16w16_gfx942_kernel
    fresh = types.FunctionType(
        body.__code__, body.__globals__, body.__name__, body.__defaults__
    )
    return flyc.kernel(fresh, name=name)


@flyc.jit
def gemm_a16w16_gfx942(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    semaphore: fx.Tensor,
    signal: fx.Tensor,
    workspace: fx.Tensor,
    split_k: fx.Int32,
    param: GemmA16W16Gfx942Param,
    stream: fx.Stream = fx.Stream(None),  # noqa: B008
):
    m = fx.Int32(fx.get_scalar(a.shape[0]))
    n = fx.Int32(fx.get_scalar(b.shape[1]))
    k = fx.Int32(fx.get_scalar(a.shape[1]))
    a_leading_stride = fx.Int32(fx.get_scalar(a.stride[0]))
    b_leading_stride = fx.Int32(fx.get_scalar(b.stride[1]))
    tiled_mma = make_gfx942_tiled_mma(param)
    # Whole block_k tiles per split-K partition; the last one may be shorter.
    split_alignment = param.block_k
    working_k = (k + split_k - 1) // split_k
    working_k = (working_k + split_alignment - 1) // split_alignment * split_alignment
    num_pid_m = (m + param.block_m - 1) // param.block_m
    num_pid_n = (n + param.block_n - 1) // param.block_n
    named_gemm_a16w16_gfx942_kernel(make_gemm_a16w16_gfx942_kernel_name(param))(
        out,
        a,
        b,
        semaphore,
        signal,
        workspace,
        m,
        n,
        k,
        split_k,
        working_k,
        a_leading_stride,
        b_leading_stride,
        tiled_mma,
        param,
    ).launch(
        grid=(num_pid_m * num_pid_n, split_k, 1),
        block=(param.block_threads, 1, 1),
        stream=stream,
    )


def _dynamic_tensor_arg(tensor, leading_dim):
    return flyc.from_dlpack(tensor.detach()).mark_layout_dynamic(
        leading_dim=leading_dim
    )


def _tensor_span_bytes(tensor: torch.Tensor) -> int:
    if tensor.numel() == 0:
        return 0
    last_offset = sum(
        (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
    )
    return (last_offset + 1) * tensor.element_size()


def _check_flat_buffer_capacity(name: str, size: int) -> None:
    if size >= FLAT_BUFFER_LIMIT_BYTES:
        raise ValueError(
            f"gfx942 GEMM {name} buffer spans {size} bytes; "
            "flat buffer descriptors require fewer than 4 GiB"
        )


@functools.cache
def get_split_k_buffers(stream_handle: int, device, split_k, split_k_reduce):
    """Split-K tile semaphores and signals, one set per split_k and split_k_reduce.

    Also one per stream, since launches on two streams may overlap. Keyed by
    the raw stream handle (``cuda_stream``), so no stream object is kept alive.
    PyTorch pooled streams recycle a bounded set of handles; external streams
    can add one set per distinct handle. A reused handle inherits its idle set.
    Sets (8-12 KB each) are never evicted because a captured CUDA graph keeps
    using their addresses.

    Before graph capture, callers must run the operation once on the capture
    stream so allocation and the one-time zeroing happen outside the graph.

    Zeroed once, never reset by the host. Atomic split-K (``SplitKProtocol``)
    never resets its 64-bit tickets: each launch advances them by split_k,
    which is how a split derives the launch index, so a set shared across
    split_k values leaves splits waiting for a tag that never comes. The
    split_k_reduce arrival counters are 32-bit and reset by the last split,
    which would corrupt the tickets, so they get their own set.
    """
    sem_dtype = torch.int32 if split_k_reduce else torch.int64
    with persistent_alloc(torch.device(device)):
        semaphore = torch.zeros((SPLIT_K_MAX_TILES,), dtype=sem_dtype, device=device)
        signal = torch.zeros((SPLIT_K_MAX_TILES,), dtype=torch.int32, device=device)
    return semaphore, signal


# Search space of the fallback config and of the tuner's pingpong screen:
# tiles with their (m, n, k) wave layouts. The cost model cannot tell m/n
# wave splits, k_waves 2 from 4, or block_k apart, so earlier entries win
# its ties.
CANDIDATE_TILES_M = (32, 64, 128, 256)
CANDIDATE_TILES_N = (32, 48, 64, 96, 112, 128, 160, 192, 256)
CANDIDATE_WAVES = ((2, 2, 1), (2, 1, 2), (1, 2, 2), (1, 4, 1), (4, 1, 1), (1, 1, 4))
CANDIDATE_MAX_SPLIT_K = 8
PINGPONG_PIPELINE = {
    "stages": 1,
    "buffered_load": True,
    "async_load_bytes": 16,
    "pingpong": True,
}
# (block_m, block_n, block_k, split_k, m_waves, n_waves, k_waves).
# Bounding the runtime catalogue bounds JIT fan-out for dynamic shapes. The
# tuner still searches the complete candidate space above.
FALLBACK_CATALOG = (
    (32, 32, 64, 1, 2, 2, 1),
    (32, 32, 64, 4, 2, 2, 1),
    (32, 32, 128, 8, 2, 2, 1),
    (64, 64, 64, 1, 2, 2, 1),
    (64, 64, 128, 3, 2, 2, 1),
    (64, 64, 128, 4, 2, 2, 1),
    (64, 128, 64, 1, 2, 2, 1),
    (64, 128, 128, 1, 2, 2, 1),
    (64, 128, 128, 2, 2, 2, 1),
    (64, 128, 128, 4, 2, 2, 1),
    (128, 128, 64, 1, 2, 2, 1),
    (128, 128, 128, 1, 2, 2, 1),
    (128, 192, 64, 1, 2, 2, 1),
)


def fallback_catalog_configs() -> list[dict]:
    configs = []
    for bm, bn, bk, sk, mw, nw, kw in FALLBACK_CATALOG:
        configs.append(
            {
                "block_m": bm,
                "block_n": bn,
                "block_k": bk,
                "split_k": sk,
                "m_waves": mw,
                "n_waves": nw,
                "k_waves": kw,
                "group_m": 4,
                "split_k_reduce": sk > 1,
                "direct_store": False,
                **PINGPONG_PIPELINE,
            }
        )
    return configs


def pingpong_fill_cost_us(
    bm: int, bn: int, k_waves: int, split_k: int, k_tiles: int
) -> float:
    """Modelled runtime of one round of single-CU pingpong workgroups.

    Least-squares fit (14% rms) to ~1500 graph-timed pingpong configs on MI300X
    (N=256..2048, K=2048..10240): one bk128 tile costs 0.0205us per MFMA per
    SIMD plus 0.23us, and the split-K workspace reduction adds 2.86us plus
    54us per MB of fp32 partials. A tile is never faster than streaming its
    (bm + bn) x 128 bf16 operands at ~35 GB/s per CU; without that floor the
    fit prefers several rounds of 64x64 tiles over one of 128x128, which runs
    ~25% slower at M >= 768.
    """
    t_tile = max(0.0205 * (bm * bn / 128) + 0.23, (bm + bn) * 256 / 35e3)
    per_split = -(-k_tiles // split_k)
    if k_waves > 1 and split_k > 1 and bm * bn <= 64 * 64:
        # Small tiles with K waves save ~7% per K tile but pay a fixed LDS
        # reduction of ~1.7 tiles, so they win on long splits.
        cost = 1.8 + (0.93 * per_split + 1.7) * t_tile
    else:
        if k_waves > 1 and split_k > 1:
            # Larger tiles with K waves run ~11% slower per K tile.
            t_tile *= 1.107
        cost = 1.8 + per_split * t_tile
    if split_k > 1:
        cost += 2.86 + 54.25e-6 * split_k * bm * bn * 4
    return cost


def model_cost_us(m: int, n: int, k: int, cfg: dict) -> float:
    """Busiest-CU estimate: rounds of workgroups x the fitted per-round cost."""
    tiles = -(-m // cfg["block_m"]) * -(-n // cfg["block_n"])
    rounds = -(-(tiles * cfg["split_k"]) // GFX942_NUM_CUS)
    return rounds * pingpong_fill_cost_us(
        cfg["block_m"], cfg["block_n"], cfg["k_waves"], cfg["split_k"], max(k // 128, 1)
    )


def shape_error(m: int, n: int, k: int, cfg: dict) -> str | None:
    """Why ``cfg`` cannot run an (m, n, k) problem beyond the param checks, or None."""
    cfg = {**CONFIG_DEFAULTS, **cfg}
    block_k = cfg["block_k"]
    split_k = cfg["split_k"]
    if k % block_k:
        return f"K={k} must be divisible by block_k={block_k} (K-tail unsupported)"
    if cfg["direct_store"] and n % cfg["block_n"]:
        return "direct_store requires N to be a block_n multiple"
    # SplitKProtocol.zero_c clears C in whole fp32x4 vectors.
    if split_k > 1 and not cfg["split_k_reduce"] and n % 4:
        return f"atomic split-K requires N to be a multiple of 4 (got N={n})"
    k_tiles = k // block_k
    last_split_tiles = k_tiles - (split_k - 1) * -(-k_tiles // split_k)
    if last_split_tiles < 1:
        return (
            f"split_k={split_k} leaves an empty K partition "
            f"(K={k}, block_k={block_k})"
        )
    if split_k > 1:
        tiles = -(-m // cfg["block_m"]) * -(-n // cfg["block_n"])
        if tiles > SPLIT_K_MAX_TILES:
            return (
                f"split-K needs {tiles} tile semaphores, "
                f"more than {SPLIT_K_MAX_TILES}"
            )
    if cfg["pingpong"]:
        if last_split_tiles < 2:
            return (
                f"pingpong needs at least 2 K tiles per split "
                f"(K={k}, split_k={split_k}, block_k={block_k})"
            )
    elif not cfg["buffered_load"]:
        stages = cfg["stages"]
        if last_split_tiles < stages - 1:
            return (
                f"async stages={stages} needs at least {stages - 1} K tiles per "
                f"split (K={k}, split_k={split_k}, block_k={block_k})"
            )
    return None


def config_is_valid(
    m: int, n: int, k: int, cfg: dict, out_dtype_id: int = GEMM_A16W16_DTYPE_BF16
) -> bool:
    if shape_error(m, n, k, cfg) is not None:
        return False
    try:
        make_gemm_a16w16_gfx942_param(out_dtype_id=out_dtype_id, **cfg)
    except (ValueError, AssertionError, NotImplementedError):
        return False
    return True


def pingpong_candidates(m: int, n: int, k: int) -> list[tuple[float, dict]]:
    """Pingpong configs of a shape with their modelled cost.

    Sorted cheapest first and not yet checked by ``config_is_valid``. Split-K
    only fills one round of workgroups, never more, and leaves every split at
    least two K tiles.
    """
    block_ks = (128, 64, 256) if k >= 2048 else (64, 128, 256)
    ranked = []
    order = 0
    for bm, bn, (mw, nw, kw), bk, sk in itertools.product(
        CANDIDATE_TILES_M,
        CANDIDATE_TILES_N,
        CANDIDATE_WAVES,
        block_ks,
        range(1, CANDIDATE_MAX_SPLIT_K + 1),
    ):
        order += 1
        if bm % (mw * 16) or bn % (nw * 16) or k % bk or bk % (kw * 64):
            continue
        if bm > max(2 * m, 32) or bn > max(2 * n, 32):
            continue
        tiles = -(-m // bm) * -(-n // bn)
        if sk > 1 and tiles * sk > GFX942_NUM_CUS:
            continue
        k_tiles = k // bk
        if k_tiles - (sk - 1) * -(-k_tiles // sk) < 2:
            continue
        cfg = {
            "block_m": bm,
            "block_n": bn,
            "block_k": bk,
            "split_k": sk,
            "m_waves": mw,
            "n_waves": nw,
            "k_waves": kw,
            "group_m": 4,
            "split_k_reduce": sk > 1,
            "direct_store": False,
            **PINGPONG_PIPELINE,
        }
        ranked.append((model_cost_us(m, n, k, cfg), order, cfg))
    ranked.sort(key=lambda item: item[:2])
    return [(cost, cfg) for cost, _, cfg in ranked]


def fallback_eligible(n: int, cfg: dict) -> bool:
    """Candidates the cost model misranks, kept out of the fallback only.

    Measured on 160 untuned shapes: per-wave tiles above 64x112 spill
    accumulators, tiles more than 3x wider than tall (or 4x taller than wide)
    lose to squarer ones, and non-power-of-two block_n that leaves a partial
    N tile loses to a power of two.
    """
    bm, bn = cfg["block_m"], cfg["block_n"]
    if (bm // cfg["m_waves"]) * (bn // cfg["n_waves"]) > 64 * 112:
        return False
    if bn > 3 * bm or bm > 4 * bn:
        return False
    return not (bn & (bn - 1) and n % bn)


@functools.cache
def _fallback_config(m: int, n: int, k: int, out_dtype_id: int) -> dict:
    ranked = []
    for order, cfg in enumerate(fallback_catalog_configs()):
        tiles = -(-m // cfg["block_m"]) * -(-n // cfg["block_n"])
        if cfg["split_k"] > 1 and tiles * cfg["split_k"] > GFX942_NUM_CUS:
            continue
        if fallback_eligible(n, cfg) and config_is_valid(m, n, k, cfg, out_dtype_id):
            ranked.append((model_cost_us(m, n, k, cfg), order, cfg))
    if ranked:
        return min(ranked, key=lambda item: item[:2])[2]
    # Fewer than two block_k=64 tiles: the async pipeline, which needs one.
    return {
        "block_m": 32 if m <= 32 else 64,
        "block_n": 32 if n <= 32 else 64,
        "block_k": 64,
        "stages": 2,
        "split_k": 1,
        "m_waves": 2,
        "n_waves": 2,
        "k_waves": 1,
        "group_m": 0,
    }


def gfx942_fallback_config(
    m: int, n: int, k: int, out_dtype: torch.dtype = torch.bfloat16
) -> dict:
    """Cheapest valid config from the bounded runtime catalogue."""
    out_dtype_id = (
        GEMM_A16W16_DTYPE_FP32 if out_dtype == torch.float32 else GEMM_A16W16_DTYPE_BF16
    )
    return dict(_fallback_config(m, n, k, out_dtype_id))


_STRUCTURAL_KEYS = frozenset(
    ("block_m", "block_n", "block_k", "stages", "m_waves", "n_waves", "buffered_load")
)
_AUTO_ONLY_KEYS = (
    "split_k",
    "stages",
    "buffered_load",
    "async_load_bytes",
    "pingpong",
    "k_waves",
    "split_k_reduce",
    "direct_store",
)

# Config columns of the tuned table (aiter/configs/flydsl_gemm_a16w16_gfx942_tuned.csv).
TUNED_CONFIG_COLUMNS = tuple(CONFIG_DEFAULTS)
TUNED_BOOL_COLUMNS = frozenset(
    name for name, value in CONFIG_DEFAULTS.items() if isinstance(value, bool)
)
TUNED_KEY_COLUMNS = ("gfx", "cu_num", "M", "N", "K", "out_dtype")


def tuned_out_dtype_name(out_dtype: torch.dtype) -> str:
    return "fp32" if out_dtype == torch.float32 else "bf16"


def parse_tuned_config(row: dict) -> dict:
    """Config dict of a tuned-table row; blank cells keep the kernel default."""
    cfg = {}
    for name in TUNED_CONFIG_COLUMNS:
        value = (row.get(name) or "").strip()
        if value:
            cfg[name] = (
                value.lower() == "true" if name in TUNED_BOOL_COLUMNS else int(value)
            )
    return cfg


@functools.cache
def _load_tuned_table(path: str) -> dict:
    table = {}
    if not os.path.exists(path):
        return table
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["gfx"],
                int(row["cu_num"]),
                int(row["M"]),
                int(row["N"]),
                int(row["K"]),
                row["out_dtype"],
            )
            table[key] = parse_tuned_config(row)
    return table


@functools.cache
def tuned_table_path() -> str:
    """Resolved table path (env override + model_configs merge), cached per process."""
    from aiter.jit.core import AITER_CONFIGS

    return AITER_CONFIGS.AITER_CONFIG_FLYDSL_GEMM_A16W16_GFX942_FILE


def reload_tuned_table() -> None:
    """Re-read the table, e.g. after the tuner rewrote it or the env var changed."""
    tuned_table_path.cache_clear()
    _load_tuned_table.cache_clear()


@functools.cache
def _device_key() -> tuple[str, int]:
    gfx, cu_num = get_gfx_runtime(), get_cu_num()
    if gfx != "gfx942" or cu_num != GFX942_NUM_CUS:
        raise RuntimeError(
            "gfx942 GEMM requires a gfx942 device with 304 CUs; "
            f"got {gfx!r} with {cu_num} CUs"
        )
    return gfx, cu_num


def gfx942_tuned_config(
    m: int,
    n: int,
    k: int,
    out_dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
) -> dict | None:
    """The tuned-table config for this exact shape and device, or None."""
    if os.getenv("AITER_FLYDSL_GEMM_GFX942_SKIP_TABLE", "0") == "1":
        return None
    del device  # Aiter assumes a homogeneous node; shared helpers key the target.
    gfx, cu_num = _device_key()
    cfg = _load_tuned_table(tuned_table_path()).get(
        (gfx, cu_num, m, n, k, tuned_out_dtype_name(out_dtype))
    )
    return dict(cfg) if cfg is not None else None


def gfx942_default_config(
    m: int,
    n: int,
    k: int,
    out_dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
) -> dict:
    """What ``gemm_a16w16`` runs without overrides: the tuned row, else the
    cost-model fallback. ``AITER_FLYDSL_GEMM_GFX942_SKIP_TABLE=1`` skips the table.
    """
    cfg = gfx942_tuned_config(m, n, k, out_dtype, device)
    return cfg if cfg is not None else gfx942_fallback_config(m, n, k, out_dtype)


def gemm_a16w16(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    user_kwargs: dict | None = None,
    stream: torch.cuda.Stream | None = None,
    layout: str = "nt",
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Compute C[M, N] = A[M, K] @ B[K, N] on gfx942 (bf16 in, f32 accum).

    ``layout`` follows the a16w16 convention (N = row-major, T = column-major);
    only ``nt``, the nn.Linear call (A [M,K] row-major, B = w.t() column-major),
    is supported on gfx942.

    ``user_kwargs`` override the shape's default config (``gfx942_default_config``).
    Setting a tile or pipeline key (``_STRUCTURAL_KEYS``) also drops that
    config's pipeline flags and split-K, so the caller's tile runs the async
    pipeline without split-K unless ``user_kwargs`` sets them too.
    Padded row strides are supported when each operand's base address and row
    stride preserve the kernel's 16-byte global-load alignment.
    """
    if bias is not None:
        raise NotImplementedError("gfx942 GEMM bias fusion is not implemented yet")
    if layout.lower() != "nt":
        raise ValueError(
            f"gfx942 GEMM supports the 'nt' layout only (got {layout!r}); "
            "transposed operands need CDNA4 ds_read_tr"
        )

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"a and b must be 2-D (got {a.ndim}-D and {b.ndim}-D)")
    device = a.device
    if device.type != "cuda" or b.device != device:
        raise ValueError(f"a and b must be on one GPU (got {device} and {b.device})")
    _device_key()
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError(f"gfx942 kernel is bf16-only (got {a.dtype} and {b.dtype})")
    m, k = a.shape
    if b.shape[0] != k:
        raise ValueError(
            f"a is [M, K] = {list(a.shape)}, b is [K, N] = {list(b.shape)}"
        )
    n = b.shape[1]
    if a.stride(1) != 1 or b.stride(0) != 1:
        raise ValueError(
            "the nt layout needs a row-major (a.stride(1) == 1) and b column-major "
            f"(b.stride(0) == 1); got strides {a.stride()} and {b.stride()}"
        )
    for name, tensor, leading_dim in (("a", a, 0), ("b", b, 1)):
        row_stride_bytes = tensor.stride(leading_dim) * tensor.element_size()
        if tensor.data_ptr() % 16 != 0 or row_stride_bytes % 16 != 0:
            raise ValueError(
                f"{name} must have a 16-byte-aligned base address and row stride; "
                f"got address {tensor.data_ptr():#x} and {row_stride_bytes} bytes"
            )

    current = torch.cuda.current_stream()
    if stream is None:
        stream = (
            current if current.device == device else torch.cuda.current_stream(device)
        )
    elif stream.device != device:
        raise ValueError(f"stream is on {stream.device}, the operands on {device}")

    if out_dtype is None:
        out_dtype = a.dtype if out is None else out.dtype
    if out_dtype not in (a.dtype, torch.float32):
        raise ValueError(f"unsupported output dtype {out_dtype}")
    if out is not None:
        if out.dtype != out_dtype:
            raise TypeError(f"out is {out.dtype}, out_dtype is {out_dtype}")
        if out.device != device:
            raise ValueError(f"out is on {out.device}, the operands on {device}")
        if not out.is_contiguous() or out.numel() != m * n:
            raise ValueError(
                f"out must be contiguous with M * N = {m * n} elements "
                f"(got shape {list(out.shape)}, strides {out.stride()})"
            )

    on_stream = (
        contextlib.nullcontext() if stream == current else torch.cuda.stream(stream)
    )
    if m == 0 or n == 0:
        with on_stream:
            if out is None:
                out = torch.empty((m, n), dtype=out_dtype, device=device)
        return out.view(m, n)

    _check_flat_buffer_capacity("A", _tensor_span_bytes(a))
    _check_flat_buffer_capacity("B", _tensor_span_bytes(b))

    # Param defaults, overlaid by the shape's tuned-table row (else the
    # cost-model fallback), then by any caller overrides.
    kwargs = dict(CONFIG_DEFAULTS)
    auto = gfx942_default_config(m, n, k, out_dtype, device)
    user_kwargs = dict(user_kwargs or {})
    if _STRUCTURAL_KEYS & user_kwargs.keys():
        # A caller-chosen tile/pipeline must not inherit the pipeline flags or
        # split-K chosen for the default config's own tile/pipeline.
        for key in _AUTO_ONLY_KEYS:
            auto.pop(key, None)
    kwargs.update(auto)
    kwargs.update(user_kwargs)
    kwargs["in_dtype_id"] = GEMM_A16W16_DTYPE_BF16
    split_k = kwargs["split_k"]
    if split_k == 1:
        kwargs["split_k_reduce"] = False
    split_k_reduce = kwargs["split_k_reduce"]

    # gfx942/CDNA3 has no packed-bf16 buffer atomic add, so bf16-output atomic
    # split-K accumulates through an f32 scratch buffer and casts to bf16 at the
    # end. f32-output split-K accumulates in place with f32 atomics.
    # split_k_reduce writes the final dtype directly.
    f32_accum_for_bf16 = (
        split_k > 1 and not split_k_reduce and out_dtype is torch.bfloat16
    )
    kwargs["out_dtype_id"] = (
        GEMM_A16W16_DTYPE_FP32
        if f32_accum_for_bf16 or out_dtype is torch.float32
        else GEMM_A16W16_DTYPE_BF16
    )

    error = shape_error(m, n, k, kwargs)
    if error is not None:
        raise ValueError(error)
    param = make_gemm_a16w16_gfx942_param(**kwargs)

    kernel_out_bytes = (
        m * n * (4 if kwargs["out_dtype_id"] == GEMM_A16W16_DTYPE_FP32 else 2)
    )
    _check_flat_buffer_capacity("output", kernel_out_bytes)
    workspace_elems = None
    if split_k_reduce:
        num_tiles = ((m + param.block_m - 1) // param.block_m) * (
            (n + param.block_n - 1) // param.block_n
        )
        workspace_elems = num_tiles * split_k * param.block_m * param.block_n
        _check_flat_buffer_capacity("split-K workspace", workspace_elems * 4)

    # Allocate on the launch stream, so the caching allocator orders reuse of
    # these buffers after the kernel, and copy back on it. Entering the stream
    # context costs ~8 us of host time, so it is skipped when already current.
    with on_stream:
        if out is None:
            out = torch.empty((m, n), dtype=out_dtype, device=device)
        out = out.view(m, n)
        if f32_accum_for_bf16:
            kernel_out = torch.empty((m, n), dtype=torch.float32, device=device)
        else:
            kernel_out = out
        semaphore, signal = get_split_k_buffers(
            stream.cuda_stream, device, split_k, split_k_reduce
        )
        if split_k_reduce:
            # Partial tiles of every split; the last split of a tile reduces them.
            workspace = torch.empty(
                (workspace_elems,),
                dtype=torch.float32,
                device=device,
            )
        else:
            workspace = signal
        dispatch_args = (
            _dynamic_tensor_arg(kernel_out, 1),
            _dynamic_tensor_arg(a, 1),
            _dynamic_tensor_arg(b, 0),
            semaphore,
            signal,
            workspace,
            split_k,
            param,
            stream,
        )
        run_cached(
            gemm_a16w16_gfx942,
            *dispatch_args,
            constexpr_param=param,
            compiler=flyc.compile,
            dispatch_args=dispatch_args,
        )
        if f32_accum_for_bf16:
            out.copy_(kernel_out)
    return out
