# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Skinny A16W16 GEMM for gfx950 (the FlyDSL HGEMM ``sk`` policy).

    C[m, n] = A[m, k] @ B[n, k]^T,  bf16 in and out, fp32 accumulate, m <= block_m

A decode GEMM multiplies a few rows of A against a weight it reads exactly once,
so streaming the weight is the whole cost and any cross-workgroup reduction is
pure overhead. Each workgroup here owns one ``block_n``-column slab of C with the
full K range: no partial leaves the workgroup, so there is no split-K workspace,
semaphore or atomic, and a CUDA graph records nothing but the launch.

Inside the workgroup K is split over ``k_waves`` waves. Each wave streams its
slice in ``block_k``-element chunks through a wave-private ``stages``-deep LDS
ring filled by global->LDS DMA (``buffer_load ... lds``, 16 bytes per lane, so
every instruction covers whole 128-byte lines; XOR-swizzled per 16-byte chunk)
and multiplies them with 16x16x32 MFMAs. The ring is wave-private, so the K loop
has no workgroup barrier: ``vmcnt`` alone orders a stage's DMA before its reads.
The ``k_waves`` fp32 partials are summed once through LDS and stored once as
bf16. Rows of A at or above m read as zero and rows of C at or above m are
dropped, both by the buffer-descriptor bounds, so one compiled kernel serves
every m up to ``block_m``.

Fixed choices, each measured faster than its alternative on MI355X when the
kernel was tuned for the GLM-5.2 o_proj decode shape (N=6144, K=2048):
  * the weight is loaded non-temporal and the N slabs are dealt to the 8 XCDs
    in contiguous runs, so each XCD reads one dense region of the weight;
  * a stage is refilled right after its MFMA fragments are read, before the
    MFMAs run, so the refill DMA does not queue behind MFMA issue;
  * chunk-0 DMAs are issued as soon as their own offsets are computed;
  * for block_m <= 16 the MFMA runs operand-swapped (a C^T tile), so the
    partials go to LDS as one 16-byte write per fragment and the padding
    columns of a 24-wide slab are never reduced; above that the partials use
    a row-major fp32 layout with a 4-element row pad against bank conflicts.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.expr import arith, const_expr, gpu, ptrtoint, range_constexpr, rocdl
from flydsl.expr.typing import T

from aiter.jit.utils.chip_info import get_lds_capacity_bytes

from . import buffer_ops
from .kernels_common import create_llvm_ptr
from .tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    _run_compiled,
    _to_raw,
    ptr_arg,
)

WAVE_SIZE = 64
# bf16 elements in one 16-byte lane of a global->LDS DMA
DMA_VEC = 8
# MFMA 16x16x32: K elements consumed per step
MFMA_K = 32
GFX950_NUM_XCDS = 8
# fp32 row pad of the row-major partial reduce: puts the four MFMA lane groups
# in distinct LDS banks
REDUCE_ROW_PAD = 4


@functools.lru_cache(maxsize=1024)
def skinny_config_error(n, k, block_m, block_n, block_k, stages, k_waves):
    """Return why (N, K, config) cannot run on the skinny kernel, or None.

    M only has to satisfy 1 <= M <= block_m, which callers check per call.
    """
    if block_k < MFMA_K or block_k % MFMA_K or WAVE_SIZE % (block_k // DMA_VEC):
        return f"block_k={block_k} must be a multiple of {MFMA_K} dividing 512"
    rows_per_dma = WAVE_SIZE // (block_k // DMA_VEC)
    if block_m % rows_per_dma or block_n % rows_per_dma:
        return (
            f"block_m={block_m} and block_n={block_n} must be multiples of "
            f"{rows_per_dma}, the rows one DMA instruction covers at block_k={block_k}"
        )
    if block_n % 8:
        return f"block_n={block_n} must be a multiple of 8"
    if n % block_n:
        return f"N={n} must be a multiple of block_n={block_n}"
    if k_waves < 1 or k % (k_waves * block_k):
        return f"K={k} must be a multiple of k_waves*block_k={k_waves * block_k}"
    if stages < 1:
        return f"stages={stages} must be >= 1"
    chunks = k // k_waves // block_k
    st = min(stages, chunks)
    ipc = (block_m + block_n) // rows_per_dma
    if (st - 1) * ipc > 63:
        return f"stages={stages} needs more DMAs in flight than vmcnt can count"
    wave_elems = st * (block_m + block_n) * block_k
    mf, nf = (block_m + 15) // 16, (block_n + 15) // 16
    if block_m <= 16:
        reduce_elems = mf * nf * WAVE_SIZE * 4
    else:
        reduce_elems = mf * 16 * (nf * 16 + REDUCE_ROW_PAD)
    if reduce_elems * 2 > wave_elems:
        return "the fp32 partials do not fit in the wave's LDS ring; raise stages or block_k"
    if (mf * 16) * (block_n // DMA_VEC) > k_waves * WAVE_SIZE:
        return "the epilogue needs more threads than the workgroup has; raise k_waves"
    lds_bytes = k_waves * wave_elems * 2
    if lds_bytes > get_lds_capacity_bytes("gfx950"):
        return f"needs {lds_bytes} bytes of LDS, more than {get_lds_capacity_bytes('gfx950')}"
    return None


@functools.lru_cache(maxsize=256)
def compile_gemm_a16w16_skinny_gfx950(
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    k_waves: int,
):
    TA, TN, KC, K_WAVES = block_m, block_n, block_k, k_waves
    # TA = A rows staged; TM = MFMA rows (TA rounded up to 16). MFMA rows in
    # [TA, TM) read other LDS rows, but they only feed C rows >= m, which the
    # C descriptor bounds drop.
    MF = (TA + 15) // 16
    TM = MF * 16
    NF = (TN + 15) // 16
    TNP = NF * 16
    N_BLOCKS = n // TN
    KS = k // K_WAVES
    CHUNKS = KS // KC
    ST = min(stages, CHUNKS)
    CPR = KC // DMA_VEC  # 16-byte chunks per staged row
    ROWS_PER_INST = WAVE_SIZE // CPR
    R = TA + TN  # rows per staged chunk: A rows, then W rows
    IPC = R // ROWS_PER_INST  # DMA instructions per chunk
    A_INSTS = TA // ROWS_PER_INST
    STEPS = KC // MFMA_K
    STAGE_ELEMS = R * KC
    WAVE_ELEMS = ST * STAGE_ELEMS
    BLOCK_THREADS = K_WAVES * WAVE_SIZE
    LDS_ELEMS = K_WAVES * WAVE_ELEMS
    # Operand-swapped MFMA (C^T tile) with accumulator-order partials: one MFMA
    # row tile only.
    SWAPPED = TA <= 16
    TNR = TNP + REDUCE_ROW_PAD
    # (mf, 4-column group, row) output slots of the swapped epilogue, one per thread
    EPI_SLOTS = MF * (TN // 4) * 16
    VEC_PER_ROW = TN // DMA_VEC
    OUT_VECS = TM * VEC_PER_ROW
    ROW_GROUP = max(1, 16 // CPR)  # rows sharing one 256-byte bank line
    XCD = GFX950_NUM_XCDS if N_BLOCKS % GFX950_NUM_XCDS == 0 else 0

    def swz(r, c):
        return c ^ ((r // ROW_GROUP) % CPR)

    @fx.struct
    class SkinnyLdsShared:
        buf: fx.Array[fx.BFloat16, LDS_ELEMS, 16]

    KERNEL_NAME = f"hgemm_bf16_skinny_t{TA}x{TN}x{KC}x{ST}_w1x1x{K_WAVES}"

    def mfma(a_fr, b_fr, acc):
        return rocdl.mfma_f32_16x16x32_bf16(T.f32x4, [a_fr, b_fr, acc, 0, 0, 0])

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def skinny_kernel(
        C: fx.Pointer,
        A: fx.Pointer,
        B: fx.Pointer,
        m: fx.Int32,
    ):
        acc_init = arith.constant_vector(0.0, T.vec(4, T.f32))
        m_idx = fx.Index(m)
        a_bytes = arith.index_cast(T.i64, m_idx * fx.Index(k * 2))
        c_bytes = arith.index_cast(T.i64, m_idx * fx.Index(n * 2))
        a_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(ptrtoint(A)), num_records_bytes=a_bytes
        )
        b_rsrc = buffer_ops.create_buffer_resource_from_addr(fx.Int64(ptrtoint(B)))
        c_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(ptrtoint(C)), num_records_bytes=c_bytes
        )
        lds = fx.SharedAllocator().allocate(SkinnyLdsShared).peek()
        buf_ptr = lds.buf.ptr
        red_ptr = fx.recast_iter(fx.Float32, buf_ptr)

        tid = fx.Index(fx.thread_idx.x)
        wid = tid // WAVE_SIZE
        lane = tid % WAVE_SIZE
        ln = lane % 16
        lg = lane // 16
        if const_expr(XCD > 0):
            # XCD-contiguous slabs: workgroups are dealt round-robin over the
            # XCDs, so give each XCD a dense run of N slabs.
            bid = fx.Index(fx.block_idx.x)
            n0 = ((bid % XCD) * (N_BLOCKS // XCD) + bid // XCD) * TN
        else:
            n0 = fx.Index(fx.block_idx.x) * TN
        k0 = wid * KS

        buf_i64 = fx.Int64(fx.ptrtoint(buf_ptr))
        wave_off = rocdl.readfirstlane(
            T.i64,
            arith.index_cast(
                T.i64, fx.Index(wid) * arith.constant(WAVE_ELEMS * 2, index=True)
            ),
        )
        wave_lds = buffer_ops.get_element_ptr(create_llvm_ptr(buf_i64, 3), wave_off)

        # per-lane global byte offset of chunk 0, one per DMA instruction
        def chunk0_offset(i):
            p = lane + i * WAVE_SIZE
            r = p // CPR
            c = swz(r, p % CPR)
            if const_expr(i < A_INSTS):
                g = (r * k + k0 + c * DMA_VEC) * 2
            else:
                g = ((n0 + (r - TA)) * k + k0 + c * DMA_VEC) * 2
            return arith.index_cast(T.i32, g)

        vofs = [None] * IPC
        chunk_soff = [
            arith.constant(j * KC * 2, type=T.i32) for j in range_constexpr(CHUNKS)
        ]

        def issue_chunk(j):
            stage = j % ST
            for i in range_constexpr(IPC):
                lds_p = buffer_ops.get_element_ptr(
                    wave_lds,
                    static_byte_offset=stage * STAGE_ELEMS * 2 + i * WAVE_SIZE * 16,
                )
                rsrc = a_rsrc if const_expr(i < A_INSTS) else b_rsrc
                cpol = "" if const_expr(i < A_INSTS) else " nt"
                if const_expr(vofs[i] is None):
                    # each chunk-0 DMA goes out as soon as its own offset is
                    # ready; the sched_barrier below keeps the remaining
                    # address math from being hoisted in front of it
                    vofs[i] = chunk0_offset(i)
                # the chunk's K offset goes in soffset: an instruction offset
                # would also shift the LDS destination of the DMA
                llvm.InlineAsmOp(
                    None,
                    [lds_p, vofs[i], rsrc, chunk_soff[j]],
                    "s_mov_b32 m0, $0\n\tbuffer_load_dwordx4 $1, $2, $3 offen"
                    + cpol
                    + " lds",
                    "s,v,s,s",
                    has_side_effects=True,
                )
                if const_expr(j == 0):
                    rocdl.sched_barrier(0)

        def read_frag(stage, row, s):
            c = s * 4 + lg
            slot = row * CPR + swz(row, c)
            return fx.ptr_load(
                buf_ptr
                + fx.Int64(wid * WAVE_ELEMS + (stage * STAGE_ELEMS) + slot * DMA_VEC),
                result_type=fx.Vector.make_type(DMA_VEC, fx.BFloat16),
            )

        for j in range_constexpr(ST):
            issue_chunk(j)
        rocdl.sched_barrier(0)

        accs = [[acc_init] * NF for _ in range_constexpr(MF)]
        for j in range_constexpr(CHUNKS):
            stage = j % ST
            younger = min(CHUNKS, j + ST) - 1 - j
            llvm.InlineAsmOp(
                None, [], f"s_waitcnt vmcnt({younger * IPC})", "", has_side_effects=True
            )
            rocdl.sched_barrier(0)
            frags = []
            for s in range_constexpr(STEPS):
                a_fr = [read_frag(stage, ln + mf * 16, s) for mf in range_constexpr(MF)]
                b_fr = [
                    read_frag(stage, ln + (TA + nf * 16), s)
                    for nf in range_constexpr(NF)
                ]
                frags.append((a_fr, b_fr))
            if const_expr(j + ST < CHUNKS):
                # refill the stage just read once its ds_reads have landed
                llvm.InlineAsmOp(
                    None, [], "s_waitcnt lgkmcnt(0)", "", has_side_effects=True
                )
                issue_chunk(j + ST)
            rocdl.sched_barrier(0)
            for s in range_constexpr(STEPS):
                a_fr, b_fr = frags[s]
                for mf in range_constexpr(MF):
                    for nf in range_constexpr(NF):
                        if const_expr(SWAPPED):
                            # C^T tile: each lane holds 4 consecutive N columns of one M row
                            accs[mf][nf] = mfma(b_fr[nf], a_fr[mf], accs[mf][nf])
                        else:
                            accs[mf][nf] = mfma(a_fr[mf], b_fr[nf], accs[mf][nf])
            rocdl.sched_barrier(0)

        red_wave0 = wid * (WAVE_ELEMS // 2)
        if const_expr(SWAPPED):
            # partials in accumulator order, one 16-byte LDS write per fragment:
            # red[w][mf*NF+nf][lane][i] = C[mf*16 + lane%16][nf*16 + 4*(lane//16) + i]
            for mf in range_constexpr(MF):
                for nf in range_constexpr(NF):
                    if const_expr(nf * 16 < TN):
                        fx.ptr_store(
                            accs[mf][nf],
                            red_ptr
                            + fx.Int64(
                                red_wave0 + (mf * NF + nf) * (WAVE_SIZE * 4) + lane * 4
                            ),
                        )
            gpu.barrier()
            for base in range_constexpr(0, EPI_SLOTS, BLOCK_THREADS):
                sl = tid + base
                mf_s = sl // ((TN // 4) * 16)
                rem = sl % ((TN // 4) * 16)
                cg = rem // 16  # 4-column group
                rr = rem % 16

                def reduce_store(mf_s, cg, rr):
                    src = (mf_s * NF + cg // 4) * (WAVE_SIZE * 4) + (
                        (cg % 4) * 16 + rr
                    ) * 4
                    tot = None
                    for w in range_constexpr(K_WAVES):
                        v = fx.ptr_load(
                            red_ptr + fx.Int64(src + w * (WAVE_ELEMS // 2)),
                            result_type=fx.Vector.make_type(4, fx.Float32),
                        )
                        tot = v if const_expr(w == 0) else tot + v
                    out_v = arith.truncf(T.vec(4, T.bf16), _to_raw(tot))
                    c_off = arith.index_cast(T.i32, (mf_s * 16 + rr) * n + n0 + cg * 4)
                    buffer_ops.buffer_store(out_v, c_rsrc, c_off)

                if const_expr(base + BLOCK_THREADS <= EPI_SLOTS):
                    reduce_store(mf_s, cg, rr)
                else:
                    in_sl = arith.cmpi(arith.CmpIPredicate.ult, sl, fx.Index(EPI_SLOTS))
                    sl_if = scf.IfOp(in_sl, results_=[], has_else=False)
                    with ir.InsertionPoint(sl_if.then_block):
                        reduce_store(mf_s, cg, rr)
                        scf.YieldOp([])
            return

        # partials -> this wave's own LDS region, row-major fp32: red[w][row][col]
        for mf in range_constexpr(MF):
            for nf in range_constexpr(NF):
                acc_v = fx.Vector(accs[mf][nf])
                for i in range_constexpr(4):
                    row = lg * 4 + (mf * 16 + i)
                    col = ln + nf * 16
                    fx.ptr_store(
                        acc_v[i], red_ptr + fx.Int64(red_wave0 + row * TNR + col)
                    )
        gpu.barrier()

        is_out = arith.cmpi(arith.CmpIPredicate.ult, tid, fx.Index(OUT_VECS))
        out_if = scf.IfOp(is_out, results_=[], has_else=False)
        with ir.InsertionPoint(out_if.then_block):
            r = tid // VEC_PER_ROW
            c = (tid % VEC_PER_ROW) * DMA_VEC
            rc_off = r * TNR + c
            tot = None
            for w in range_constexpr(K_WAVES):
                v = fx.ptr_load(
                    red_ptr + fx.Int64(rc_off + w * (WAVE_ELEMS // 2)),
                    result_type=fx.Vector.make_type(DMA_VEC, fx.Float32),
                )
                tot = v if const_expr(w == 0) else tot + v
            out_v = arith.truncf(T.vec(DMA_VEC, T.bf16), _to_raw(tot))
            c_off = arith.index_cast(T.i32, r * n + n0 + c)
            buffer_ops.buffer_store(out_v, c_rsrc, c_off)
            scf.YieldOp([])

    @flyc.jit
    def launch_skinny_kernel(
        C: fx.Pointer,
        A: fx.Pointer,
        B: fx.Pointer,
        m: fx.Int32,
        stream: fx.Stream,
    ):
        skinny_kernel._func.__name__ = KERNEL_NAME
        skinny_kernel(C, A, B, m).launch(
            grid=(N_BLOCKS, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    # the ring-fill DMAs need the kernel args; preloaded into SGPRs they no
    # longer wait on the kernarg s_load round trip
    launch_skinny_kernel.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_skinny_kernel


def gemm_a16w16_skinny(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    k_waves: int,
    out: torch.Tensor | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """out[m, n] = a[m, k] @ b[n, k]^T on the skinny kernel; bf16 only, no bias."""
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise ValueError(f"skinny HGEMM supports bf16 only, got {a.dtype} x {b.dtype}")
    if a.dim() != 2 or b.dim() != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(
            f"expected a[m, k] and b[n, k], got {tuple(a.shape)} and {tuple(b.shape)}"
        )
    m, k = a.shape
    n = b.shape[0]
    err = skinny_config_error(n, k, block_m, block_n, block_k, stages, k_waves)
    if err is None and not 1 <= m <= block_m:
        err = f"M must be in [1, block_m={block_m}]"
    if err is not None:
        raise ValueError(f"skinny HGEMM cannot run M={m}, N={n}, K={k}: {err}")
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    elif out.shape != (m, n) or out.dtype != torch.bfloat16 or not out.is_contiguous():
        raise ValueError(f"out must be a contiguous bf16 [{m}, {n}] tensor")
    exe = compile_gemm_a16w16_skinny_gfx950(
        int(n),
        int(k),
        int(block_m),
        int(block_n),
        int(block_k),
        int(stages),
        int(k_waves),
    )
    launch_stream = (
        torch.cuda.current_stream(device=a.device) if stream is None else stream
    )
    _run_compiled(
        exe, ptr_arg(out), ptr_arg(a), ptr_arg(b), int(m), fx.Stream(launch_stream)
    )
    return out
