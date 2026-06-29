# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL implementation of the sparse paged-decode attention kernel for gfx942.

Ports _pa_decode_sparse from Triton (aiter/ops/triton/_triton_kernels/attention/
pa_decode_sparse.py) to FlyDSL. Both KV_SPLITS==1 (direct output) and KV_SPLITS>1
(partial emit + Triton reduce) paths are implemented.

Thread layout
-------------
BLOCK_THREADS = 64 (one wave). Each thread owns VEC = D // 64 contiguous elements of
the head dimension (e.g. VEC=9 for D=576). All 64 threads together cover all D
elements. The online-softmax accumulator (D fp32 values) lives in LDS; only the
running (m_i, l_i) scalars are carried as scf.ForOp iter-args.

KV_SPLITS > 1
-------------
The Triton _pa_decode_sparse_reduce kernel is reused for the combine step. The FlyDSL
kernel writes (m, l, acc) partials to intermediate fp32 buffers with the same layout
that the Triton reduce kernel expects:
  m_partial   [T, KV_SPLITS, H]    fp32
  l_partial   [T, KV_SPLITS, H]    fp32
  acc_partial [T, KV_SPLITS, H, D] fp32
"""

from __future__ import annotations

import functools
from contextlib import contextmanager

import torch
import triton

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as mlir_llvm, rocdl as mlir_rocdl, scf
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, vector
from flydsl.expr import rocdl
from flydsl.expr.arith import ArithValue, CmpFPredicate, CmpIPredicate
from flydsl.expr.typing import T, Int32, Stream as FlyStream
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from .tensor_shim import STensor, _to_raw

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLOCK_THREADS = 64   # one wave64
_LOG2E = 1.4426950408889634
_NEG_INF = float("-inf")

# ---------------------------------------------------------------------------
# LDS pointer helper (addr-space 3 = LDS on AMDGPU)
# ---------------------------------------------------------------------------

def _make_lds_ptr(byte_addr_i64):
    """Convert an i64 LDS byte address to !llvm.ptr<3> (LDS pointer)."""
    lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")
    return mlir_llvm.IntToPtrOp(lds_ptr_type, byte_addr_i64).result


# ---------------------------------------------------------------------------
# Helper: IfOp then-block context manager (mirrors fused_compress_attn.py)
# ---------------------------------------------------------------------------

@contextmanager
def _if_then(if_op):
    """SCF IfOp then-region context manager. Auto-yields empty if missing."""
    with ir.InsertionPoint(if_op.then_block):
        try:
            yield if_op.then_block
        finally:
            blk = if_op.then_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=256)
def _compile_pa_decode_sparse(
    *,
    H: int,
    D: int,
    BLOCK_K: int,
    KV_SPLITS: int,
    softmax_scale: float,
):
    """JIT-compile the FlyDSL kernel for the given constexpr configuration.

    Returns a flyc.jit launcher function that can be called with torch tensors.
    Cached on (H, D, BLOCK_K, KV_SPLITS, softmax_scale) — first call compiles,
    subsequent calls reuse.
    """
    assert D % BLOCK_THREADS == 0, f"D={D} must be divisible by BLOCK_THREADS={BLOCK_THREADS}"
    VEC = D // BLOCK_THREADS   # elements per thread (9 for D=576)

    fm = arith.FastMathFlags.fast

    # ---- LDS allocation ----
    # Region 0: acc accumulator — D fp32 = D*4 bytes
    # Region 1: KV staging      — BLOCK_K rows, row-major.
    # Region 2: scores          — BLOCK_K fp32 (written by all threads, same value)
    # Region 3: p_vals          — BLOCK_K fp32 (written by all threads, same value)
    # Region 4: slots           — BLOCK_K i32 (written by thread 0, read by all)
    # Region 5: valids          — BLOCK_K i32 (written by thread 0, read by all)
    #
    # buffer_load_to_lds uses size_bytes=4 (one dword per lane per call).
    # 64 lanes × 4 bytes = 256 bytes per call.  To preserve row-major layout,
    # the row stride must be a multiple of 256 bytes.
    # Row size (unpadded): D × sizeof(bf16) = D*2 bytes.
    # Padded row size:     ceil(D*2 / 256) × 256 bytes.
    _ROW_BYTES_RAW    = D * 2                         # bf16 row in bytes
    _N_DMA_CALLS      = (_ROW_BYTES_RAW + 255) // 256 # calls needed per row
    _ROW_BYTES_PADDED = _N_DMA_CALLS * 256            # padded to multiple of 256
    # Number of bf16 elements per padded row (may include padding elements at end)
    _ROW_ELEMS_PADDED = _ROW_BYTES_PADDED // 2

    GPU_ARCH = get_rocm_arch()
    allocator = SmemAllocator(
        None,
        arch=GPU_ARCH,
        global_sym_name=f"pa_decode_sparse_smem_D{D}_K{BLOCK_K}_S{KV_SPLITS}",
    )
    lds_acc_off   = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_acc_off + D * 4                          # D × sizeof(fp32)
    lds_kv_off    = allocator._align(allocator.ptr, 256)         # 256-byte align for DMA
    allocator.ptr = lds_kv_off + BLOCK_K * _ROW_BYTES_PADDED     # BLOCK_K padded rows
    lds_scores_off = allocator._align(allocator.ptr, 16)
    allocator.ptr  = lds_scores_off + BLOCK_K * 4               # BLOCK_K × sizeof(f32)
    lds_pvals_off  = allocator._align(allocator.ptr, 16)
    allocator.ptr  = lds_pvals_off + BLOCK_K * 4                # BLOCK_K × sizeof(f32)
    lds_slots_off  = allocator._align(allocator.ptr, 16)
    allocator.ptr  = lds_slots_off + BLOCK_K * 4                # BLOCK_K × sizeof(i32)
    lds_valids_off = allocator._align(allocator.ptr, 16)
    allocator.ptr  = lds_valids_off + BLOCK_K * 4               # BLOCK_K × sizeof(i32)

    # ---- Helpers (defined outside @flyc.kernel so they capture constants) ----

    def _fexp2(x):
        """exp2(x) via llvm.amdgcn.exp2.f32."""
        return mlir_llvm.call_intrinsic(T.f32, "llvm.amdgcn.exp2.f32", [x], [], [])

    def _wave_reduce_add(x):
        """Butterfly reduce-add across the 64-lane wave."""
        w = _to_raw(x)
        # 6 rounds: halves = 32, 16, 8, 4, 2, 1
        for sh in range_constexpr(6):
            half = 32 >> sh   # 32, 16, 8, 4, 2, 1
            peer = _to_raw(ArithValue(w).shuffle_xor(half, BLOCK_THREADS))
            w = arith.AddFOp(w, peer, fastmath=fm).result
        return w

    def _lds_ptr_i64(byte_addr_i32):
        """Extend i32 LDS byte address to i64, return !llvm.ptr<3>."""
        addr_i64 = arith.extui(T.i64, byte_addr_i32)
        return _make_lds_ptr(addr_i64)

    # ---- Kernel name ----
    _kname = f"pa_decode_sparse_H{H}_D{D}_K{BLOCK_K}_S{KV_SPLITS}_flydsl"

    @flyc.kernel(name=_kname, known_block_size=[BLOCK_THREADS, 1, 1])
    def _kernel(
        q:           fx.Tensor,   # [T, H, D]            bf16
        unified_kv:  fx.Tensor,   # [total_pages, D]     bf16
        kv_indices:  fx.Tensor,   # [total_indices]      i32
        kv_indptr:   fx.Tensor,   # [T+1]                i32
        attn_sink:   fx.Tensor,   # [H]                  f32
        m_partial:   fx.Tensor,   # [T, KV_SPLITS, H]    f32  (dummy when KV_SPLITS==1)
        l_partial:   fx.Tensor,   # [T, KV_SPLITS, H]    f32
        acc_partial: fx.Tensor,   # [T, KV_SPLITS, H, D] f32
        out:         fx.Tensor,   # [T, H, D]            bf16
    ):
        f32 = T.f32
        i32 = T.i32

        c_zero     = arith.constant(0.0, type=f32)
        c_one      = arith.constant(1.0, type=f32)
        c_neg_inf  = arith.constant(_NEG_INF, type=f32)
        c_eps      = arith.constant(1e-30, type=f32)
        c_log2e    = arith.constant(_LOG2E, type=f32)
        c_scale_l2 = arith.constant(softmax_scale * _LOG2E, type=f32)

        c0_i32 = arith.constant(0, type=i32)
        c1_i32 = arith.constant(1, type=i32)
        c2_i32 = arith.constant(2, type=i32)

        cH_i32   = arith.constant(H, type=i32)
        cD_i32   = arith.constant(D, type=i32)
        cHD_i32  = arith.constant(H * D, type=i32)
        cKVS_i32 = arith.constant(KV_SPLITS, type=i32)
        cBLK_i32 = arith.constant(BLOCK_K, type=i32)
        cVEC_i32 = arith.constant(VEC, type=i32)
        cKH_i32  = arith.constant(KV_SPLITS * H, type=i32)

        # block_idx / thread_idx return `index` type; cast to i32 for arithmetic.
        t     = arith.index_cast(i32, _to_raw(fx.block_idx.x))   # token index
        h     = arith.index_cast(i32, _to_raw(fx.block_idx.y))   # head index
        pid_k = arith.index_cast(i32, _to_raw(fx.block_idx.z))   # KV-split index
        tid   = arith.index_cast(i32, _to_raw(fx.thread_idx.x))  # 0..63

        # ---- 1. kv_indptr: read kv_start and kv_end ----
        indptr_rsrc = buffer_ops.create_buffer_resource(kv_indptr, max_size=True)
        kv_start_v  = buffer_ops.buffer_load(indptr_rsrc, t, vec_width=1, dtype=i32)
        kv_end_v    = buffer_ops.buffer_load(
            indptr_rsrc, ArithValue(t) + c1_i32, vec_width=1, dtype=i32
        )
        kv_start = rocdl.readfirstlane(i32, kv_start_v)
        kv_end   = rocdl.readfirstlane(i32, kv_end_v)
        kv_len   = arith.subi(kv_end, kv_start)

        # ---- 2. Compute tile range for this KV-split ----
        tiles_per_seg = arith.ceildivsi(kv_len, arith.muli(cKVS_i32, cBLK_i32))
        tile_start    = arith.muli(pid_k, tiles_per_seg)
        num_tiles     = arith.ceildivsi(kv_len, cBLK_i32)
        tile_end_raw  = arith.muli(arith.addi(pid_k, c1_i32), tiles_per_seg)
        tile_end      = arith.minsi(tile_end_raw, num_tiles)

        # ---- 3. Early-exit: nothing to do for this split ----
        has_work = arith.cmpi(CmpIPredicate.slt, tile_start, tile_end)
        _if_work = scf.IfOp(_to_raw(has_work), results_=[], has_else=False)
        with _if_then(_if_work):

            # ---- 4. Per-thread column range: [tid*VEC, tid*VEC+VEC) ----
            tid_x_vec = arith.muli(tid, cVEC_i32)

            # ---- 5. Load Q row: q[t, h, tid*VEC : tid*VEC+VEC] ----
            # Q is bf16; load as packed i32 dwords (2 bf16 per dword).
            # q_base is the flat element index of the first bf16 for this thread.
            q_rsrc = buffer_ops.create_buffer_resource(q, max_size=True)
            q_base = arith.addi(
                arith.muli(t, cHD_i32),
                arith.addi(arith.muli(h, cD_i32), tid_x_vec),
            )
            # Load VEC bf16 scalars, convert to fp32, scale by softmax_scale * LOG2E.
            # Since tid*VEC is always even (VEC=9 is odd → first element of each thread
            # may land on an odd bf16 index for some threads), handle per-element.
            q_lane = []
            for v in range_constexpr(VEC):
                off_bf16 = arith.addi(q_base, arith.constant(v, type=i32))
                # Dword index: floor(off_bf16 / 2); which half: off_bf16 % 2
                off_dw = arith.divsi(off_bf16, c2_i32)
                odd    = arith.remsi(off_bf16, c2_i32)
                dw_v   = buffer_ops.buffer_load(q_rsrc, off_dw, vec_width=1, dtype=i32)
                pair   = vector.bitcast(
                    T.vec(2, T.bf16),
                    vector.from_elements(T.vec(1, T.i32), [dw_v])
                )
                lo = vector.extract(pair, static_position=[0], dynamic_position=[])
                hi = vector.extract(pair, static_position=[1], dynamic_position=[])
                is_odd   = arith.cmpi(CmpIPredicate.eq, odd, c1_i32)
                q_bf16   = arith.select(is_odd, hi, lo)
                q_f32    = arith.extf(f32, q_bf16)
                q_lane.append(arith.mulf(q_f32, c_scale_l2, fastmath=fm))

            # ---- 6. Init m_i and l_i ----
            # KV_SPLITS is a Python compile-time constant (captured from factory closure).
            # The branch is resolved at kernel-factory time, so both paths are fine.
            # We pre-assign defaults to satisfy Python name analysis inside @flyc.kernel.
            m_init = c_neg_inf
            l_init = c_zero
            if KV_SPLITS == 1:
                # Fold attention sink as the virtual initial key (weight=1, value=0).
                # m_i = sink * LOG2E,  l_i = exp2(sink - m_i) = 1.0
                sink_rsrc = buffer_ops.create_buffer_resource(attn_sink, max_size=True)
                sink_v    = buffer_ops.buffer_load(sink_rsrc, h, vec_width=1, dtype=f32)
                sink_s    = rocdl.readfirstlane(f32, sink_v)
                m_init    = arith.mulf(sink_s, c_log2e, fastmath=fm)
                l_init    = c_one

            # ---- 7. Init LDS regions ----
            lds_base = allocator.get_base()
            # Acc accumulator: D fp32 values.
            lds_acc  = STensor(
                SmemPtr(lds_base, lds_acc_off, T.f32, shape=(D,)),
                dtype=T.f32,
                shape=(D,),
            )
            # KV staging: BLOCK_K rows, each _ROW_ELEMS_PADDED bf16 elements wide.
            # Layout: element (j, e) at flat index j*_ROW_ELEMS_PADDED + e.
            lds_kv = STensor(
                SmemPtr(lds_base, lds_kv_off, T.bf16,
                        shape=(BLOCK_K * _ROW_ELEMS_PADDED,)),
                dtype=T.bf16,
                shape=(BLOCK_K * _ROW_ELEMS_PADDED,),
            )
            # LDS-resident scores, p_vals, slots, valids — eliminates VGPR pressure
            # from holding BLOCK_K simultaneously-live values throughout the tile loop.
            lds_scores = STensor(
                SmemPtr(lds_base, lds_scores_off, T.f32, shape=(BLOCK_K,)),
                dtype=T.f32,
                shape=(BLOCK_K,),
            )
            lds_pvals = STensor(
                SmemPtr(lds_base, lds_pvals_off, T.f32, shape=(BLOCK_K,)),
                dtype=T.f32,
                shape=(BLOCK_K,),
            )
            lds_slots = STensor(
                SmemPtr(lds_base, lds_slots_off, T.i32, shape=(BLOCK_K,)),
                dtype=T.i32,
                shape=(BLOCK_K,),
            )
            lds_valids = STensor(
                SmemPtr(lds_base, lds_valids_off, T.i32, shape=(BLOCK_K,)),
                dtype=T.i32,
                shape=(BLOCK_K,),
            )
            for v in range_constexpr(VEC):
                idx = arith.addi(tid_x_vec, arith.constant(v, type=i32))
                lds_acc[fx.Index(idx)] = c_zero

            # Barrier: ensure all threads finish zeroing before the tile loop.
            gpu.barrier()

            # ---- 8. Buffer resources for kv_indices and unified_kv ----
            idx_rsrc = buffer_ops.create_buffer_resource(kv_indices, max_size=True)
            kv_rsrc  = buffer_ops.create_buffer_resource(unified_kv,  max_size=True)

            # Per-thread voffset contribution for DMA calls:
            #   tid * 4 bytes (each thread loads one dword per call)
            tid_dw_off = arith.muli(tid, arith.constant(4, type=i32))

            # ---- 9. KV tile loop ----
            # Carries (m_i, l_i) as ForOp iter-args; acc lives in LDS.
            for tile_idx, loop_state in range(
                _to_raw(tile_start), _to_raw(tile_end), 1,
                init=[m_init, l_init],
            ):
                m_i = loop_state[0]
                l_i = loop_state[1]

                # Loop induction variable tile_idx is index type; cast to i32.
                tile_i32    = arith.index_cast(i32, _to_raw(tile_idx))
                k_tile_base = arith.muli(tile_i32, cBLK_i32)

                # -- 9a. Thread 0 loads BLOCK_K slot indices into LDS ----
                # All threads read the same index values, so only thread 0 loads
                # and writes to lds_slots/lds_valids, eliminating 16 readfirstlane
                # calls (major SALU reduction).
                kv_end_m1 = arith.subi(arith.addi(kv_start, kv_len), c1_i32)
                is_t0 = arith.cmpi(CmpIPredicate.eq, tid, c0_i32)
                _if_load_slots = scf.IfOp(_to_raw(is_t0), results_=[], has_else=False)
                with _if_then(_if_load_slots):
                    for j in range_constexpr(BLOCK_K):
                        raw_pos = arith.addi(kv_start,
                                             arith.addi(k_tile_base,
                                                        arith.constant(j, type=i32)))
                        in_rng  = arith.cmpi(CmpIPredicate.slt, raw_pos,
                                              arith.addi(kv_start, kv_len))
                        pos     = arith.minsi(raw_pos, kv_end_m1)
                        slot_v  = buffer_ops.buffer_load(idx_rsrc, pos, vec_width=1, dtype=i32)
                        slot    = arith.select(in_rng, slot_v,
                                               arith.constant(-1, type=i32))
                        valid_i32 = arith.select(in_rng, c1_i32, c0_i32)
                        j_idx = arith.constant(j, type=i32)
                        lds_slots[fx.Index(j_idx)]  = slot
                        lds_valids[fx.Index(j_idx)] = valid_i32

                # Barrier: thread 0 slot writes visible to all threads before DMA.
                gpu.barrier()

                # -- 9b. Load BLOCK_K KV rows into LDS via buffer_load_to_lds --
                #
                # All threads read slot from LDS (no readfirstlane needed).
                # buffer_load_to_lds with size_bytes=4 (one dword per lane per call):
                #   64 lanes × 4 bytes = 256 bytes loaded per call.
                #   Each lane i loads VRAM[row_base + tid*4 + n*256] into
                #   LDS[lds_row_base + i*4 + n*256].
                #   → row-major layout preserved in LDS.
                for j in range_constexpr(BLOCK_K):
                    j_idx       = arith.constant(j, type=i32)
                    slot_j      = lds_slots[fx.Index(j_idx)]
                    safe_slot_j = arith.maxsi(slot_j, c0_i32)
                    # VRAM byte base for this row: safe_slot_j * D * 2 + tid*4
                    vram_row_base = arith.muli(
                        safe_slot_j, arith.constant(D * 2, type=i32)
                    )
                    vram_voff = arith.addi(vram_row_base, tid_dw_off)
                    # LDS base for row j (static, byte address)
                    lds_row_base_byte = arith.constant(
                        lds_kv_off + j * _ROW_BYTES_PADDED, type=i32
                    )
                    lds_ptr = _lds_ptr_i64(lds_row_base_byte)
                    for n in range_constexpr(_N_DMA_CALLS):
                        # Compute how many threads have a valid VRAM dword in chunk n.
                        # Chunk n covers VRAM bytes [n*256, n*256+256); valid range is
                        # [n*256, D*2). Number of valid dwords = (valid_bytes) // 4.
                        chunk_start = n * 256
                        chunk_valid_bytes = min(256, _ROW_BYTES_RAW - chunk_start)
                        n_valid_threads = chunk_valid_bytes // 4
                        if n_valid_threads == BLOCK_THREADS:
                            # All 64 threads have valid VRAM data: unconditional load.
                            rocdl.buffer_load_to_lds(
                                kv_rsrc, lds_ptr, vram_voff,
                                size_bytes=4,
                                offset=n * 256,
                            )
                        else:
                            # Partial chunk: only threads 0..n_valid_threads-1 load.
                            # Threads >= n_valid_threads would read past unified_kv.
                            in_valid = arith.cmpi(
                                CmpIPredicate.slt,
                                tid,
                                arith.constant(n_valid_threads, type=i32),
                            )
                            _if_dma = scf.IfOp(
                                _to_raw(in_valid), results_=[], has_else=False
                            )
                            with _if_then(_if_dma):
                                rocdl.buffer_load_to_lds(
                                    kv_rsrc, lds_ptr, vram_voff,
                                    size_bytes=4,
                                    offset=n * 256,
                                )

                # Barrier: all DMA writes complete before reading LDS.
                gpu.barrier()

                # -- 9c. Compute dot products: read KV from LDS, store score to LDS --
                # Pass 1: For each KV row j, compute partial dot, wave-reduce to score,
                # then write score to lds_scores[j]. This way scores[] never exist as
                # a VGPR list — only one score is live at a time.
                m_block = c_neg_inf
                for j in range_constexpr(BLOCK_K):
                    j_idx     = arith.constant(j, type=i32)
                    valid_i32 = lds_valids[fx.Index(j_idx)]
                    valid_j   = arith.cmpi(CmpIPredicate.ne, valid_i32, c0_i32)
                    partial_j = c_zero
                    for v in range_constexpr(VEC):
                        kv_lds_idx = arith.addi(
                            arith.constant(j * _ROW_ELEMS_PADDED, type=i32),
                            arith.addi(tid_x_vec, arith.constant(v, type=i32)),
                        )
                        kv_bf16 = lds_kv[fx.Index(kv_lds_idx)]
                        kv_f32  = arith.extf(f32, kv_bf16)
                        kv_safe = arith.select(valid_j, kv_f32, c_zero)
                        partial_j = arith.addf(
                            partial_j,
                            arith.mulf(q_lane[v], kv_safe, fastmath=fm),
                            fastmath=fm,
                        )
                    full_dot = _wave_reduce_add(partial_j)
                    score_j  = arith.select(valid_j, full_dot, c_neg_inf)
                    # All threads write same score (wave-broadcast value)
                    lds_scores[fx.Index(j_idx)] = score_j
                    m_block = arith.maximumf(m_block, score_j)

                m_new = arith.maximumf(m_i, m_block)

                # -- 9d. Online softmax update --

                # alpha = (m_new == -inf) ? 1.0 : exp2(m_i - m_new)
                is_all_inf = arith.cmpf(CmpFPredicate.OEQ, m_new, c_neg_inf)
                raw_alpha  = _fexp2(arith.subf(m_i, m_new))
                alpha      = arith.select(is_all_inf, c_one, raw_alpha)

                # Pass 2: Compute p[j] = exp2(score_j - m_new), accumulate sum_p,
                # write p_vals to LDS. Only one p_j live at a time.
                sum_p = c_zero
                for j in range_constexpr(BLOCK_K):
                    j_idx     = arith.constant(j, type=i32)
                    score_j   = lds_scores[fx.Index(j_idx)]
                    valid_i32 = lds_valids[fx.Index(j_idx)]
                    valid_j   = arith.cmpi(CmpIPredicate.ne, valid_i32, c0_i32)
                    pj        = _fexp2(arith.subf(score_j, m_new))
                    pj_safe   = arith.select(valid_j, pj, c_zero)
                    lds_pvals[fx.Index(j_idx)] = pj_safe
                    sum_p = arith.addf(sum_p, pj_safe, fastmath=fm)

                l_new = arith.addf(
                    arith.mulf(l_i, alpha, fastmath=fm), sum_p, fastmath=fm
                )

                # Pass 3: Acc update — read KV and p[j] from LDS row-by-row.
                # Apply alpha rescale once (row-independent), then accumulate
                # BLOCK_K weighted KV contributions. Inner order: v-outer, j-inner
                # so each thread's VEC elements of acc are updated together.
                for v in range_constexpr(VEC):
                    idx     = arith.addi(tid_x_vec, arith.constant(v, type=i32))
                    acc_old = lds_acc[fx.Index(idx)]
                    new_val = arith.mulf(acc_old, alpha, fastmath=fm)
                    for j in range_constexpr(BLOCK_K):
                        j_idx = arith.constant(j, type=i32)
                        kv_lds_idx = arith.addi(
                            arith.constant(j * _ROW_ELEMS_PADDED, type=i32),
                            arith.addi(tid_x_vec, arith.constant(v, type=i32)),
                        )
                        kv_bf16 = lds_kv[fx.Index(kv_lds_idx)]
                        kv_f32  = arith.extf(f32, kv_bf16)
                        pj_safe = lds_pvals[fx.Index(j_idx)]
                        new_val = arith.addf(
                            new_val,
                            arith.mulf(pj_safe, kv_f32, fastmath=fm),
                            fastmath=fm,
                        )
                    lds_acc[fx.Index(idx)] = new_val

                # Barrier before next tile's DMA overwrites lds_kv/lds_slots/lds_valids.
                gpu.barrier()

                loop_state = yield [m_new, l_new]

            m_final = loop_state[0]
            l_final = loop_state[1]

            # ---- 10. Output ----
            if KV_SPLITS == 1:
                # Direct write: out[t, h, :] = acc / max(l, eps)
                out_rsrc = buffer_ops.create_buffer_resource(out, max_size=True)
                l_safe   = arith.maximumf(l_final, c_eps)
                for v in range_constexpr(VEC):
                    idx      = arith.addi(tid_x_vec, arith.constant(v, type=i32))
                    acc_val  = lds_acc[fx.Index(idx)]
                    out_f32  = arith.divf(acc_val, l_safe)
                    out_bf16 = arith.truncf(T.bf16, out_f32)
                    out_flat = arith.addi(
                        arith.muli(t, cHD_i32),
                        arith.addi(arith.muli(h, cD_i32), idx),
                    )
                    buffer_ops.buffer_store(out_bf16, out_rsrc, out_flat)

            else:
                # Partial emit: write (m, l) from thread 0, acc from all threads.

                # m_partial[t, pid_k, h] and l_partial[t, pid_k, h]
                ml_flat = arith.addi(
                    arith.muli(t, cKH_i32),
                    arith.addi(arith.muli(pid_k, cH_i32), h),
                )

                # Only thread 0 writes m and l (they are wave-wide scalars)
                is_t0  = arith.cmpi(CmpIPredicate.eq, tid, c0_i32)
                _if_t0 = scf.IfOp(_to_raw(is_t0), results_=[], has_else=False)
                with _if_then(_if_t0):
                    mp_rsrc = buffer_ops.create_buffer_resource(m_partial, max_size=True)
                    lp_rsrc = buffer_ops.create_buffer_resource(l_partial, max_size=True)
                    buffer_ops.buffer_store(m_final, mp_rsrc, ml_flat)
                    buffer_ops.buffer_store(l_final, lp_rsrc, ml_flat)

                # All threads write their VEC elements of acc_partial[t, pid_k, h, :]
                ap_rsrc = buffer_ops.create_buffer_resource(acc_partial, max_size=True)
                for v in range_constexpr(VEC):
                    idx     = arith.addi(tid_x_vec, arith.constant(v, type=i32))
                    acc_val = lds_acc[fx.Index(idx)]
                    ap_flat = arith.addi(
                        arith.muli(ml_flat, cD_i32),
                        idx,
                    )
                    buffer_ops.buffer_store(acc_val, ap_rsrc, ap_flat)

    # ---- Launcher ----
    @flyc.jit
    def _launcher(
        q, unified_kv, kv_indices, kv_indptr, attn_sink,
        m_partial, l_partial, acc_partial, out,
        T_size: Int32,
        stream: FlyStream = fx.Stream(None),
    ):
        # Emit the LDS memref.global into the gpu.module body.
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        # Grid dimensions: (T, H, KV_SPLITS).
        # T_size is an Int32 scalar; H and KV_SPLITS are Python ints (compile-time consts).
        _kernel(
            q, unified_kv, kv_indices, kv_indptr, attn_sink,
            m_partial, l_partial, acc_partial, out,
        ).launch(
            grid=(T_size, H, KV_SPLITS),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return _launcher


# ---------------------------------------------------------------------------
# MFMA-based kernel (BLOCK_H=16, one MFMA tile per head-group per CTA)
# ---------------------------------------------------------------------------
#
# Architecture summary
# --------------------
# Grid : (T, ceil(H/BLOCK_H), KV_SPLITS)   BLOCK_H=16 heads per CTA
# Block: 64 threads (one wave64)
# MFMA : mfma_f32_16x16x16bf16_1k — 16×16×16 BF16 accumulate into 4×f32 per lane
#
# MFMA C-output layout (CONFIRMED from mfma_epilogues.py::default_epilog):
#   Lane l: c_frag[v] = C[M = (l//16)*4 + v,  N = l%16]
#   In other words: M-index = lane_cgrp*4+v,  N-index = lane_row
#
# LDS layout (BLOCK_H=16, D=576) — total 56448 bytes < 64 KB
# ----------------------------------------------------------------
#   Region 0: KV_lds  [BLOCK_K=16, D]        bf16 = 16*576*2 = 18432 bytes
#   Region 1: acc_lds [BLOCK_H=16, D]        fp32 = 16*576*4 = 36864 bytes
#   Region 2: p_lds   [BLOCK_H=16, BLOCK_K]  fp32 = 16*16*4  =  1024 bytes
#   Region 3: slots/valids [BLOCK_K*2]       i32  = 16*4*2   =   128 bytes
#
# Key design: KV tile loaded ONCE into KV_lds per tile (via buffer_load_to_lds),
# then reused for both QK B-frag and PV B-frag reads — matching Triton's pattern
# of using the kv register tile for both tl.dot(q, tl.trans(kv)) and
# tl.dot(p, kv). Q is loaded from VRAM per tile (hits L2 after first tile).
#
# QK step: S[BLOCK_H, BLOCK_K] = Q[BLOCK_H, D] × KV[BLOCK_K, D]^T
#   A-frag: Q[M=lane_row (head), K=lane_cgrp*4..+3 (D)]  — from VRAM (L2-cached)
#   B-frag: KV[N=lane_row (kv_slot), K=lane_cgrp*4..+3 (D)]  — from KV_lds
#   C-frag: c_frag[v] = S[head=lane_cgrp*4+v, kv=lane_row]
#
# Softmax: per-head max/exp across BLOCK_K KV columns.
#   Lane l holds scores for 4 heads (lane_cgrp*4..+3) vs 1 kv slot (lane_row).
#   Butterfly xor over lane_row dimension (xor 1,2,4,8) to get max/sum over all 16 kv.
#   Each lane carries 4 (m_i, l_i) pairs as ForOp iter-args.
#
# LDS transpose: write p[head=lane_cgrp*4+v, kv=lane_row]; barrier;
#   read as A-frag: p[head=lane_row, kv=lane_cgrp*4..+3]
#
# PV step: acc[BLOCK_H, D] += p[BLOCK_H, BLOCK_K] × KV[BLOCK_K, D]
#   A-frag: p[M=lane_row (head), K=lane_cgrp*4..+3 (kv)]  — from p_lds after transpose
#   B-frag: KV[N=lane_row (D_col), K=lane_cgrp*4..+3 (kv_slot)] — from KV_lds (same tile)
#   C-frag: c_frag[v] = acc[head=lane_cgrp*4+v, d=d_base+lane_row]
#
# Per-tile barrier sequence:
#   1. barrier after slot load    (lds_slots/lds_valids visible)
#   2. barrier after KV DMA       (KV_lds visible for QK + PV)
#   3. barrier after p write      (p_lds visible for PV A-frag transpose read)
#   4. barrier after acc write    (acc_lds visible before next tile overwrites KV_lds)

_MFMA_K = 16   # K-elements per MFMA call (gfx942 16×16×16 BF16)
_BLOCK_H = 16  # heads per CTA (matches MFMA M/N tile size)


def _get_mfma_bf16_k16():
    """Return the mfma_f32_16x16x16bf16_1k MLIR op, or None on non-gfx942."""
    fn = getattr(rocdl, "mfma_f32_16x16x16bf16_1k", None) or getattr(
        rocdl, "mfma_f32_16x16x16_bf16_1k", None
    )
    return fn


@functools.lru_cache(maxsize=256)
def _compile_pa_decode_sparse_mfma(
    *,
    H: int,          # actual number of heads (may be < _BLOCK_H)
    D: int,
    BLOCK_K: int,
    KV_SPLITS: int,
    softmax_scale: float,
):
    """JIT-compile the MFMA-based kernel: BLOCK_H=16 heads per CTA, 4 waves.

    Returns a flyc.jit launcher, cached on configuration.

    4-wave design (256 threads):
    - 4 waves × 64 lanes = 256 threads per CTA.
    - Each wave handles D_PER_WAVE = D//4 D-columns for both QK and PV.
    - QK b-frags (KV values) are retained as SSA VGPRs and reused for PV
      in the same tile loop body — no LDS round-trip for KV.
    - Cross-wave QK partial scores are accumulated via alias_lds (a sub-region
      of p_lds): each wave writes its partial QK c_frag to alias_lds, all waves
      barrier, then all read and sum the 4 wave partials to get full scores.
    - Softmax runs redundantly on all 4 waves (same butterfly over lane_row dim
      after alias_lds reduce), so all waves write the same p_lds values.
    - PV uses p from p_lds (transposed) and retained KV b-frags.
    """
    assert D % _MFMA_K == 0, f"D={D} must be divisible by MFMA_K={_MFMA_K}"
    assert D % 4 == 0, f"D={D} must be divisible by 4 waves"
    assert BLOCK_K == _BLOCK_H, (
        f"MFMA kernel requires BLOCK_K == BLOCK_H == {_BLOCK_H}; got BLOCK_K={BLOCK_K}"
    )

    mfma_fn = _get_mfma_bf16_k16()
    assert mfma_fn is not None, "mfma_f32_16x16x16bf16_1k not found in rocdl"

    fm = arith.FastMathFlags.fast
    N_WAVES       = 4
    D_PER_WAVE    = D // N_WAVES          # D-columns per wave (144 for D=576)
    D_CHUNKS      = D // _MFMA_K          # total MFMA K-chunks across all waves (36 for D=576)
    D_CHUNKS_W    = D_PER_WAVE // _MFMA_K  # K-chunks per wave (9 for D=576)
    BLOCK_THREADS_MFMA = 64 * N_WAVES     # 256

    GPU_ARCH = get_rocm_arch()

    # ---- LDS allocation ----
    # alias_lds  [N_WAVES, BLOCK_H, BLOCK_K] f32 = 4*16*16*4 = 4096 bytes
    #   — cross-wave partial QK scores; same base as p_lds (superset).
    # p_lds      [BLOCK_H, BLOCK_K] f32 = 16*16*4 = 1024 bytes
    #   — p-matrix for LDS transpose (QK→PV); first 1024 bytes of alias_lds region.
    # kv_lds     [BLOCK_K=16, D] bf16 = 16*D*2 bytes
    #   — KV tile loaded during QK loop, reused for PV B-frag (transposed access).
    # acc_lds    [BLOCK_H, D] f32 = 16*D*4 bytes
    # slots_lds  [BLOCK_K] i32 = 64 bytes
    # valids_lds [BLOCK_K] i32 = 64 bytes
    #
    # Total (D=576): 4096 + 18432 + 36864 + 128 = 59,520 bytes < 64 KB
    ALIAS_LDS_BYTES = N_WAVES * _BLOCK_H * BLOCK_K * 4   # 4096 bytes
    KV_LDS_BYTES    = BLOCK_K * D * 2                     # 18432 bytes for D=576 (bf16)
    ACC_LDS_BYTES   = _BLOCK_H * D * 4                    # 36864 bytes for D=576

    allocator = SmemAllocator(
        None,
        arch=GPU_ARCH,
        global_sym_name=f"pa_decode_sparse_mfma4w_smem_D{D}_K{BLOCK_K}_S{KV_SPLITS}",
    )
    # alias_lds and p_lds share the same base offset (alias_lds is a superset).
    lds_alias_off  = allocator._align(allocator.ptr, 16)
    allocator.ptr  = lds_alias_off + ALIAS_LDS_BYTES
    # kv_lds after alias_lds
    lds_kv_off     = allocator._align(allocator.ptr, 16)
    allocator.ptr  = lds_kv_off + KV_LDS_BYTES
    # acc_lds after kv_lds
    lds_acc_off    = allocator._align(allocator.ptr, 16)
    allocator.ptr  = lds_acc_off + ACC_LDS_BYTES
    lds_slots_off  = allocator._align(allocator.ptr, 16)
    allocator.ptr  = lds_slots_off + BLOCK_K * 4
    lds_valids_off = allocator._align(allocator.ptr, 16)
    allocator.ptr  = lds_valids_off + BLOCK_K * 4

    # p_lds is the first BLOCK_H*BLOCK_K elements of alias_lds (same base offset).
    lds_p_off = lds_alias_off

    # ---- Helpers ----
    def _fexp2(x):
        return mlir_llvm.call_intrinsic(T.f32, "llvm.amdgcn.exp2.f32", [x], [], [])

    def _lds_ptr_i64(byte_addr_i32):
        """Extend i32 LDS byte address to i64, return !llvm.ptr<3>."""
        addr_i64 = arith.extui(T.i64, byte_addr_i32)
        return _make_lds_ptr(addr_i64)

    def _mfma_bf16(a_v4bf16, b_v4bf16, acc_v4f32):
        """mfma_f32_16x16x16bf16_1k: A,B as v4i16 (bitcast from v4bf16)."""
        a_vi16 = vector.bitcast(T.vec(4, T.i16), a_v4bf16)
        b_vi16 = vector.bitcast(T.vec(4, T.i16), b_v4bf16)
        return mfma_fn(T.f32x4, [a_vi16, b_vi16, acc_v4f32, 0, 0, 0])

    _kname = f"pa_decode_sparse_mfma4w_H{H}_D{D}_K{BLOCK_K}_S{KV_SPLITS}"

    @flyc.kernel(name=_kname, known_block_size=[BLOCK_THREADS_MFMA, 1, 1])
    def _kernel(
        q:           fx.Tensor,   # [T, H, D]            bf16
        unified_kv:  fx.Tensor,   # [total_pages, D]     bf16
        kv_indices:  fx.Tensor,   # [total_indices]      i32
        kv_indptr:   fx.Tensor,   # [T+1]                i32
        attn_sink:   fx.Tensor,   # [H]                  f32
        m_partial:   fx.Tensor,   # [T, KV_SPLITS, H]    f32
        l_partial:   fx.Tensor,   # [T, KV_SPLITS, H]    f32
        acc_partial: fx.Tensor,   # [T, KV_SPLITS, H, D] f32
        out:         fx.Tensor,   # [T, H, D]            bf16
    ):
        f32 = T.f32
        i32 = T.i32

        c_zero_f32  = arith.constant(0.0, type=f32)
        c_one_f32   = arith.constant(1.0, type=f32)
        c_neg_inf   = arith.constant(_NEG_INF, type=f32)
        c_eps       = arith.constant(1e-30, type=f32)
        c_log2e     = arith.constant(_LOG2E, type=f32)
        c_scale_l2  = arith.constant(softmax_scale * _LOG2E, type=f32)
        c0_i32  = arith.constant(0, type=i32)
        c1_i32  = arith.constant(1, type=i32)
        c2_i32  = arith.constant(2, type=i32)
        c4_i32  = arith.constant(4, type=i32)
        c64_i32 = arith.constant(64, type=i32)
        cD_i32  = arith.constant(D, type=i32)
        cH_i32  = arith.constant(H, type=i32)
        cHD_i32 = arith.constant(H * D, type=i32)
        cBH_i32 = arith.constant(_BLOCK_H, type=i32)
        cBK_i32 = arith.constant(BLOCK_K, type=i32)
        cKH_i32 = arith.constant(KV_SPLITS * H, type=i32)
        cDPW_i32 = arith.constant(D_PER_WAVE, type=i32)

        # Block/thread indices
        t      = arith.index_cast(i32, _to_raw(fx.block_idx.x))
        bh_blk = arith.index_cast(i32, _to_raw(fx.block_idx.y))
        pid_k  = arith.index_cast(i32, _to_raw(fx.block_idx.z))
        tid    = arith.index_cast(i32, _to_raw(fx.thread_idx.x))  # 0..255

        # 4-wave decomposition
        wave_id  = arith.divsi(tid, c64_i32)   # 0..3
        lane     = arith.remsi(tid, c64_i32)   # 0..63
        lane_row = arith.remsi(lane, cBH_i32)  # 0..15 — MFMA M/N index (head/kv)
        lane_cgrp = arith.divsi(lane, cBH_i32) # 0..3  — column group within wave's D-slice

        # D-column base for this wave
        d_wave_base = arith.muli(wave_id, cDPW_i32)  # 0, D/4, D/2, 3*D/4

        h_base  = arith.muli(bh_blk, cBH_i32)
        h_lane  = arith.addi(h_base, lane_row)
        h_valid = arith.cmpi(CmpIPredicate.slt, h_lane, cH_i32)

        # ---- kv_indptr ----
        indptr_rsrc = buffer_ops.create_buffer_resource(kv_indptr, max_size=True)
        kv_start_v  = buffer_ops.buffer_load(indptr_rsrc, t, vec_width=1, dtype=i32)
        kv_end_v    = buffer_ops.buffer_load(
            indptr_rsrc, ArithValue(t) + c1_i32, vec_width=1, dtype=i32
        )
        kv_start = rocdl.readfirstlane(i32, kv_start_v)
        kv_end   = rocdl.readfirstlane(i32, kv_end_v)
        kv_len   = arith.subi(kv_end, kv_start)

        # ---- Tile range ----
        cKVS_i32      = arith.constant(KV_SPLITS, type=i32)
        tiles_per_seg = arith.ceildivsi(kv_len, arith.muli(cBK_i32, cKVS_i32))
        tile_start    = arith.muli(pid_k, tiles_per_seg)
        num_tiles     = arith.ceildivsi(kv_len, cBK_i32)
        tile_end_raw  = arith.muli(arith.addi(pid_k, c1_i32), tiles_per_seg)
        tile_end      = arith.minsi(tile_end_raw, num_tiles)

        has_work = arith.cmpi(CmpIPredicate.slt, tile_start, tile_end)
        _if_work = scf.IfOp(_to_raw(has_work), results_=[], has_else=False)
        with _if_then(_if_work):

            # ---- LDS regions ----
            lds_base = allocator.get_base()

            # alias_lds: [N_WAVES, BLOCK_H, BLOCK_K] fp32 — cross-wave partial QK scores
            # flat index: wave * BLOCK_H*BLOCK_K + head * BLOCK_K + kv
            lds_alias = STensor(
                SmemPtr(lds_base, lds_alias_off, T.f32, shape=(N_WAVES * _BLOCK_H * BLOCK_K,)),
                dtype=T.f32, shape=(N_WAVES * _BLOCK_H * BLOCK_K,),
            )
            # p_lds: first BLOCK_H*BLOCK_K elements of alias_lds — p-matrix after reduce
            lds_p = STensor(
                SmemPtr(lds_base, lds_p_off, T.f32, shape=(_BLOCK_H * BLOCK_K,)),
                dtype=T.f32, shape=(_BLOCK_H * BLOCK_K,),
            )
            # kv_lds: [BLOCK_K, D] bf16 — KV tile for QK+PV; written during QK loop
            # flat index: kv_row * D + d_col
            lds_kv = STensor(
                SmemPtr(lds_base, lds_kv_off, T.bf16, shape=(BLOCK_K * D,)),
                dtype=T.bf16, shape=(BLOCK_K * D,),
            )
            # acc_lds: [BLOCK_H, D] fp32 — accumulator
            lds_acc = STensor(
                SmemPtr(lds_base, lds_acc_off, T.f32, shape=(_BLOCK_H * D,)),
                dtype=T.f32, shape=(_BLOCK_H * D,),
            )
            lds_slots = STensor(
                SmemPtr(lds_base, lds_slots_off, T.i32, shape=(BLOCK_K,)),
                dtype=T.i32, shape=(BLOCK_K,),
            )
            lds_valids = STensor(
                SmemPtr(lds_base, lds_valids_off, T.i32, shape=(BLOCK_K,)),
                dtype=T.i32, shape=(BLOCK_K,),
            )

            # ---- Zero-initialize acc_lds ----
            # Each thread (tid) owns: D_CHUNKS (across ALL waves) d_chunks, and within
            # each chunk covers lane_row d-column. But with 4 waves each covering D/4 cols,
            # thread tid covers: d = d_wave_base + d_chunk*MFMA_K + lane_cgrp*4..+3 for
            # d_chunk in 0..D_CHUNKS_W-1. Each acc_lds slot holds acc[head, d]; head =
            # lane_cgrp*4+v. The 4 waves together cover all D columns, so all 256 threads
            # are needed to zero the full acc_lds.
            for dc in range_constexpr(D_CHUNKS_W):
                d_base_i32 = arith.addi(d_wave_base, arith.constant(dc * _MFMA_K, type=i32))
                d_col_i32  = arith.addi(d_base_i32, lane_row)
                for v in range_constexpr(4):
                    acc_head = arith.addi(arith.muli(lane_cgrp, c4_i32), arith.constant(v, type=i32))
                    lds_acc[fx.Index(arith.addi(arith.muli(acc_head, cD_i32), d_col_i32))] = c_zero_f32

            # ---- Init m_i[4], l_i[4] ----
            sink_rsrc_init = buffer_ops.create_buffer_resource(attn_sink, max_size=True) if KV_SPLITS == 1 else None
            m_inits = []
            l_inits = []
            for _v in range_constexpr(4):
                h_v = arith.addi(h_base, arith.addi(arith.muli(lane_cgrp, c4_i32), arith.constant(_v, type=i32)))
                h_v_valid = arith.cmpi(CmpIPredicate.slt, h_v, cH_i32)
                if KV_SPLITS == 1:
                    h_v_safe = arith.minsi(h_v, arith.subi(cH_i32, c1_i32))
                    sink_v = buffer_ops.buffer_load(sink_rsrc_init, h_v_safe, vec_width=1, dtype=f32)
                    sink_s = arith.select(h_v_valid, sink_v, c_neg_inf)
                    m_inits.append(arith.mulf(sink_s, c_log2e, fastmath=fm))
                    l_inits.append(c_one_f32)
                else:
                    m_inits.append(c_neg_inf)
                    l_inits.append(c_zero_f32)

            gpu.barrier()   # acc_lds zeroed

            # ---- Buffer resources ----
            idx_rsrc = buffer_ops.create_buffer_resource(kv_indices, max_size=True)
            kv_rsrc  = buffer_ops.create_buffer_resource(unified_kv,  max_size=True)
            q_rsrc   = buffer_ops.create_buffer_resource(q, max_size=True)

            kv_end_m1 = arith.subi(arith.addi(kv_start, kv_len), c1_i32)

            # ---- KV tile loop ----
            for tile_idx, loop_state in range(
                _to_raw(tile_start), _to_raw(tile_end), 1,
                init=m_inits + l_inits,
            ):
                m_i = [loop_state[v] for v in range_constexpr(4)]
                l_i = [loop_state[4 + v] for v in range_constexpr(4)]

                tile_i32    = arith.index_cast(i32, _to_raw(tile_idx))
                k_tile_base = arith.muli(tile_i32, cBK_i32)

                # -- Step 1: load BLOCK_K slot indices into lds_slots/lds_valids --
                # lane_cgrp==0 AND wave_id==0 lanes handle lane_row=0..15 (one slot each)
                is_cgrp0  = arith.cmpi(CmpIPredicate.eq, lane_cgrp, c0_i32)
                is_wave0  = arith.cmpi(CmpIPredicate.eq, wave_id, c0_i32)
                is_loader = arith.andi(is_cgrp0, is_wave0)
                _if_slot = scf.IfOp(_to_raw(is_loader), results_=[], has_else=False)
                with _if_then(_if_slot):
                    raw_pos   = arith.addi(kv_start, arith.addi(k_tile_base, lane_row))
                    in_rng    = arith.cmpi(CmpIPredicate.slt, raw_pos, arith.addi(kv_start, kv_len))
                    pos       = arith.minsi(raw_pos, kv_end_m1)
                    slot_v    = buffer_ops.buffer_load(idx_rsrc, pos, vec_width=1, dtype=i32)
                    slot      = arith.select(in_rng, slot_v, arith.constant(-1, type=i32))
                    valid_i32 = arith.select(in_rng, c1_i32, c0_i32)
                    lds_slots[fx.Index(lane_row)]  = slot
                    lds_valids[fx.Index(lane_row)] = valid_i32

                gpu.barrier()   # slots visible to all lanes

                # -- Step 2: QK MFMA (per wave, over this wave's D-slice) --
                # Each wave processes D_CHUNKS_W D-chunks.
                # kv slot for QK: lane_row selects the KV row (same across all waves).
                slot_j_qk    = lds_slots[fx.Index(lane_row)]
                safe_slot_qk = arith.maxsi(slot_j_qk, c0_i32)
                valid_kv_slot = arith.cmpi(CmpIPredicate.ne, lds_valids[fx.Index(lane_row)], c0_i32)

                c_zero_bf16  = arith.constant(0.0, type=T.bf16)
                c_zero_v4f32 = arith.constant_vector(0.0, T.f32x4)
                c_frag = c_zero_v4f32  # partial QK score for this wave's D-slice

                for dc in range_constexpr(D_CHUNKS_W):
                    d_base_const = dc * _MFMA_K   # local offset within wave's D-slice
                    # Global D-column = d_wave_base + d_base_const + lane_cgrp*4..+3
                    col_off = arith.addi(
                        d_wave_base,
                        arith.addi(arith.constant(d_base_const, type=i32),
                                   arith.muli(lane_cgrp, c4_i32)),
                    )

                    # Q A-frag: Q[h_lane, col_off..+3]
                    flat_q = arith.addi(
                        arith.muli(t, cHD_i32),
                        arith.addi(arith.muli(h_lane, cD_i32), col_off),
                    )
                    dw_q  = arith.divsi(flat_q, c2_i32)
                    raw_q = buffer_ops.buffer_load(q_rsrc, dw_q, vec_width=2, dtype=i32)
                    q_v4  = vector.bitcast(T.vec(4, T.bf16), raw_q)
                    q_vals = []
                    for v in range_constexpr(4):
                        elem = vector.extract(q_v4, static_position=[v], dynamic_position=[])
                        q_vals.append(arith.select(h_valid, elem, c_zero_bf16))
                    a_frag = vector.from_elements(T.vec(4, T.bf16), q_vals)

                    # KV B-frag for QK: KV[safe_slot_qk, col_off..+3]
                    flat_kv = arith.addi(arith.muli(safe_slot_qk, cD_i32), col_off)
                    dw_kv   = arith.divsi(flat_kv, c2_i32)
                    raw_kv  = buffer_ops.buffer_load(kv_rsrc, dw_kv, vec_width=2, dtype=i32)
                    kv_v4   = vector.bitcast(T.vec(4, T.bf16), raw_kv)

                    # Write KV into kv_lds[lane_row, col_off..+3] for PV reuse.
                    # kv_lds flat index: kv_row * D + d_col
                    for v in range_constexpr(4):
                        kv_elem  = vector.extract(kv_v4, static_position=[v], dynamic_position=[])
                        kv_flat  = arith.addi(
                            arith.muli(lane_row, cD_i32),
                            arith.addi(col_off, arith.constant(v, type=i32)),
                        )
                        lds_kv[fx.Index(kv_flat)] = kv_elem

                    c_frag = _mfma_bf16(a_frag, kv_v4, c_frag)

                # -- Step 3: Write partial QK scores to alias_lds --
                # c_frag[v] = partial S[head=lane_cgrp*4+v, kv=lane_row] over D/4 columns.
                # alias_lds flat: wave*BLOCK_H*BLOCK_K + head*BLOCK_K + kv
                for v in range_constexpr(4):
                    head_idx  = arith.addi(arith.muli(lane_cgrp, c4_i32), arith.constant(v, type=i32))
                    alias_flat = arith.addi(
                        arith.muli(wave_id, arith.constant(_BLOCK_H * BLOCK_K, type=i32)),
                        arith.addi(arith.muli(head_idx, arith.constant(BLOCK_K, type=i32)), lane_row),
                    )
                    s_v = vector.extract(c_frag, static_position=[v], dynamic_position=[])
                    lds_alias[fx.Index(alias_flat)] = s_v

                gpu.barrier()   # alias_lds written by all waves

                # -- Step 4: Cross-wave reduce + softmax (all waves, same result) --
                # Each lane reads the 4 wave partials for its (head, kv) combination and sums.
                # alias_lds[w, head, kv] for w=0..3 → full score[head, kv]
                # Then butterfly max/sum over lane_row dim (xor 1,2,4,8) as in current 1-wave.

                c_frag_scaled = []
                for v in range_constexpr(4):
                    head_idx = arith.addi(arith.muli(lane_cgrp, c4_i32), arith.constant(v, type=i32))
                    # Sum 4 wave partials for this (head, kv=lane_row)
                    partial_sum = c_zero_f32
                    for w in range_constexpr(N_WAVES):
                        alias_flat_w = arith.addi(
                            arith.constant(w * _BLOCK_H * BLOCK_K, type=i32),
                            arith.addi(arith.muli(head_idx, arith.constant(BLOCK_K, type=i32)), lane_row),
                        )
                        partial_sum = arith.addf(partial_sum, lds_alias[fx.Index(alias_flat_w)], fastmath=fm)
                    # Scale and gate by validity
                    partial_scaled = arith.mulf(partial_sum, c_scale_l2, fastmath=fm)
                    c_frag_scaled.append(arith.select(valid_kv_slot, partial_scaled, c_neg_inf))

                # Online softmax (same as 1-wave: butterfly over lane_row / kv dimension)
                m_new     = []
                alpha     = []
                p_vals_v4 = []
                sum_p_v   = []

                for v in range_constexpr(4):
                    s_v = c_frag_scaled[v]

                    # Butterfly max over lane_row (kv) dim within this wave's 64 lanes
                    s_max_v = s_v
                    for xor_off in [1, 2, 4, 8]:
                        peer    = _to_raw(ArithValue(s_max_v).shuffle_xor(xor_off, BLOCK_THREADS_MFMA))
                        s_max_v = arith.maximumf(s_max_v, peer)

                    m_new_v      = arith.maximumf(m_i[v], s_max_v)
                    is_all_inf_v = arith.cmpf(CmpFPredicate.OEQ, m_new_v, c_neg_inf)
                    raw_alpha_v  = _fexp2(arith.subf(m_i[v], m_new_v))
                    alpha_v      = arith.select(is_all_inf_v, c_one_f32, raw_alpha_v)

                    pj_v      = _fexp2(arith.subf(c_frag_scaled[v], m_new_v))
                    pj_safe_v = arith.select(valid_kv_slot, pj_v, c_zero_f32)

                    # Butterfly sum_p over lane_row dim
                    sum_p_vv = pj_safe_v
                    for xor_off in [1, 2, 4, 8]:
                        peer     = _to_raw(ArithValue(sum_p_vv).shuffle_xor(xor_off, BLOCK_THREADS_MFMA))
                        sum_p_vv = arith.addf(sum_p_vv, peer, fastmath=fm)

                    m_new.append(m_new_v)
                    alpha.append(alpha_v)
                    p_vals_v4.append(pj_safe_v)
                    sum_p_v.append(sum_p_vv)

                l_new = []
                for v in range_constexpr(4):
                    l_new.append(arith.addf(
                        arith.mulf(l_i[v], alpha[v], fastmath=fm), sum_p_v[v], fastmath=fm
                    ))

                # -- Step 5: Write p to p_lds (= alias_lds base) for LDS transpose --
                # All 4 waves compute the same p values (same reduce, same butterfly).
                # All waves write p_lds[head*BLOCK_K+kv]; redundant writes are harmless.
                for v in range_constexpr(4):
                    p_head = arith.addi(arith.muli(lane_cgrp, c4_i32), arith.constant(v, type=i32))
                    p_flat = arith.addi(
                        arith.muli(p_head, arith.constant(BLOCK_K, type=i32)), lane_row
                    )
                    lds_p[fx.Index(p_flat)] = p_vals_v4[v]

                gpu.barrier()   # p_lds visible for PV A-frag transposed read

                # -- Step 6: PV MFMA (per wave, using retained kv_bfrags) --
                # A-frag: p[M=lane_row (head), K=lane_cgrp*4..+3 (kv)] from p_lds (transposed)
                # B-frag: KV[N=lane_row (D-col), K=lane_cgrp*4..+3 (kv)] — from kv_lds
                # C-frag: acc[head=lane_cgrp*4+v, d=d_wave_base+dc*MFMA_K+lane_row]

                for dc in range_constexpr(D_CHUNKS_W):
                    d_base_i32 = arith.addi(d_wave_base, arith.constant(dc * _MFMA_K, type=i32))
                    d_col_i32  = arith.addi(d_base_i32, lane_row)

                    # Load C-frag from acc_lds
                    acc_vals = []
                    for v in range_constexpr(4):
                        acc_head = arith.addi(arith.muli(lane_cgrp, c4_i32), arith.constant(v, type=i32))
                        acc_flat = arith.addi(arith.muli(acc_head, cD_i32), d_col_i32)
                        acc_vals.append(lds_acc[fx.Index(acc_flat)])
                    acc_frag = vector.from_elements(T.f32x4, acc_vals)

                    # Rescale acc by per-head alpha
                    acc_scaled = []
                    for v in range_constexpr(4):
                        val = vector.extract(acc_frag, static_position=[v], dynamic_position=[])
                        acc_scaled.append(arith.mulf(val, alpha[v], fastmath=fm))
                    acc_frag_scaled = vector.from_elements(T.f32x4, acc_scaled)

                    # A-frag: p[head=lane_row, kv=lane_cgrp*4..+3] from p_lds (transposed)
                    a_vals_p = []
                    for v in range_constexpr(4):
                        kv_k     = arith.addi(arith.muli(lane_cgrp, c4_i32), arith.constant(v, type=i32))
                        p_flat_r = arith.addi(
                            arith.muli(lane_row, arith.constant(BLOCK_K, type=i32)), kv_k
                        )
                        a_vals_p.append(arith.truncf(T.bf16, lds_p[fx.Index(p_flat_r)]))
                    a_frag_pv = vector.from_elements(T.vec(4, T.bf16), a_vals_p)

                    # B-frag: KV[kv=lane_cgrp*4+v, d=d_wave_base+dc*16+lane_row] from kv_lds
                    # kv_lds flat index: kv_row * D + d_col
                    b_vals_pv = []
                    for v in range_constexpr(4):
                        kv_row_pv = arith.addi(arith.muli(lane_cgrp, c4_i32), arith.constant(v, type=i32))
                        kv_flat_pv = arith.addi(
                            arith.muli(kv_row_pv, cD_i32), d_col_i32
                        )
                        b_vals_pv.append(lds_kv[fx.Index(kv_flat_pv)])
                    b_frag_pv = vector.from_elements(T.vec(4, T.bf16), b_vals_pv)

                    # PV MFMA
                    new_acc_frag = _mfma_bf16(a_frag_pv, b_frag_pv, acc_frag_scaled)

                    # Write back to acc_lds
                    for v in range_constexpr(4):
                        val      = vector.extract(new_acc_frag, static_position=[v], dynamic_position=[])
                        acc_head = arith.addi(arith.muli(lane_cgrp, c4_i32), arith.constant(v, type=i32))
                        acc_flat = arith.addi(arith.muli(acc_head, cD_i32), d_col_i32)
                        lds_acc[fx.Index(acc_flat)] = val

                gpu.barrier()   # acc_lds written; alias_lds will be reused next tile

                loop_state = yield m_new + l_new

            m_final = [loop_state[v] for v in range_constexpr(4)]
            l_final = [loop_state[4 + v] for v in range_constexpr(4)]

            # ---- Output ----
            if KV_SPLITS == 1:
                out_rsrc = buffer_ops.create_buffer_resource(out, max_size=True)
                for v in range_constexpr(4):
                    h_v       = arith.addi(h_base, arith.addi(arith.muli(lane_cgrp, c4_i32), arith.constant(v, type=i32)))
                    h_v_valid = arith.cmpi(CmpIPredicate.slt, h_v, cH_i32)
                    l_safe_v  = arith.maximumf(l_final[v], c_eps)
                    for dc in range_constexpr(D_CHUNKS_W):
                        d_base_i32 = arith.addi(d_wave_base, arith.constant(dc * _MFMA_K, type=i32))
                        d_col_i32  = arith.addi(d_base_i32, lane_row)
                        acc_head   = arith.addi(arith.muli(lane_cgrp, c4_i32), arith.constant(v, type=i32))
                        acc_flat   = arith.addi(arith.muli(acc_head, cD_i32), d_col_i32)
                        acc_val    = lds_acc[fx.Index(acc_flat)]
                        out_f32    = arith.divf(acc_val, l_safe_v)
                        out_bf16   = arith.truncf(T.bf16, out_f32)
                        out_flat   = arith.addi(
                            arith.muli(t, cHD_i32),
                            arith.addi(arith.muli(h_v, cD_i32), d_col_i32),
                        )
                        _if_h = scf.IfOp(_to_raw(h_v_valid), results_=[], has_else=False)
                        with _if_then(_if_h):
                            buffer_ops.buffer_store(out_bf16, out_rsrc, out_flat)
            else:
                ap_rsrc = buffer_ops.create_buffer_resource(acc_partial, max_size=True)
                mp_rsrc = buffer_ops.create_buffer_resource(m_partial, max_size=True)
                lp_rsrc = buffer_ops.create_buffer_resource(l_partial, max_size=True)
                for v in range_constexpr(4):
                    h_v       = arith.addi(h_base, arith.addi(arith.muli(lane_cgrp, c4_i32), arith.constant(v, type=i32)))
                    h_v_valid = arith.cmpi(CmpIPredicate.slt, h_v, cH_i32)
                    _if_hv    = scf.IfOp(_to_raw(h_v_valid), results_=[], has_else=False)
                    with _if_then(_if_hv):
                        ml_flat = arith.addi(
                            arith.muli(t, cKH_i32),
                            arith.addi(arith.muli(pid_k, cH_i32), h_v),
                        )
                        is_row0 = arith.cmpi(CmpIPredicate.eq, lane_row, c0_i32)
                        _if_r0  = scf.IfOp(_to_raw(is_row0), results_=[], has_else=False)
                        with _if_then(_if_r0):
                            buffer_ops.buffer_store(m_final[v], mp_rsrc, ml_flat)
                            buffer_ops.buffer_store(l_final[v], lp_rsrc, ml_flat)

                        for dc in range_constexpr(D_CHUNKS_W):
                            d_base_i32  = arith.addi(d_wave_base, arith.constant(dc * _MFMA_K, type=i32))
                            d_col_i32   = arith.addi(d_base_i32, lane_row)
                            acc_head    = arith.addi(arith.muli(lane_cgrp, c4_i32), arith.constant(v, type=i32))
                            acc_flat_r  = arith.addi(arith.muli(acc_head, cD_i32), d_col_i32)
                            acc_val     = lds_acc[fx.Index(acc_flat_r)]
                            ap_flat     = arith.addi(arith.muli(ml_flat, cD_i32), d_col_i32)
                            buffer_ops.buffer_store(acc_val, ap_rsrc, ap_flat)

    @flyc.jit
    def _launcher(
        q, unified_kv, kv_indices, kv_indptr, attn_sink,
        m_partial, l_partial, acc_partial, out,
        T_size: Int32,
        stream: FlyStream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        n_head_blocks = (H + _BLOCK_H - 1) // _BLOCK_H
        _kernel(
            q, unified_kv, kv_indices, kv_indptr, attn_sink,
            m_partial, l_partial, acc_partial, out,
        ).launch(
            grid=(T_size, n_head_blocks, KV_SPLITS),
            block=(BLOCK_THREADS_MFMA, 1, 1),
            stream=stream,
        )

    return _launcher


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------

# Import the Triton reduce kernel for the KV_SPLITS>1 combine step.
# Imported lazily to avoid a circular dependency at module load time.
def _get_triton_reduce():
    from aiter.ops.triton._triton_kernels.attention.pa_decode_sparse import (
        _pa_decode_sparse_reduce,
    )
    return _pa_decode_sparse_reduce


def flydsl_pa_decode_sparse(
    q,
    unified_kv,
    kv_indices,
    kv_indptr,
    attn_sink,
    softmax_scale,
    *,
    kv_scales=None,
    block_h=None,    # unused (BLOCK_H=1 is fixed); kept for API parity
    kv_splits=None,
    has_invalid=False,   # sentinel support not yet implemented
    skip_reduce=False,
):
    """FlyDSL sparse paged-decode attention.

    Drop-in replacement for the Triton pa_decode_sparse wrapper. The FlyDSL
    main kernel runs the QK attention + online softmax; for KV_SPLITS>1 the
    existing Triton _pa_decode_sparse_reduce kernel is called to combine splits.

    Args
    ----
    q             : [T, H, D] bfloat16
    unified_kv    : [total_pages, D] bfloat16
    kv_indices    : [total_indices] int32 — flat scatter indices into unified_kv
    kv_indptr     : [T+1] int32 — CSR row pointers (kv_indptr[t]:kv_indptr[t+1]
                    gives token t's KV slice)
    attn_sink     : [H] float32 — per-head attention sink log-weight
    softmax_scale : float — 1/sqrt(D) scaling for QK dot products
    kv_scales     : not used (FP8 path not yet implemented)
    kv_splits     : int or None — number of K-splits (None = auto)
    has_invalid   : bool — whether kv_indices contains -1 sentinel values
    skip_reduce   : bool — if True return (acc_partial, m_partial, l_partial)
                    without running the reduce kernel (for testing)

    Returns
    -------
    Tensor [T, H, D] bfloat16, or tuple when skip_reduce=True.
    """
    if kv_scales is not None:
        raise NotImplementedError("FP8 KV cache not yet implemented in FlyDSL kernel")
    if has_invalid:
        raise NotImplementedError("has_invalid=True not yet implemented in FlyDSL kernel")

    T_val, H, D = q.shape
    device = q.device

    # ---- block_k selection ----
    # MFMA kernel: BLOCK_H=16, BLOCK_K=16. Requires H divisible by 16 and D divisible by 16.
    # Falls back to scalar kernel otherwise.
    _use_mfma = (
        D % _MFMA_K == 0
        and _get_mfma_bf16_k16() is not None
        and H % _BLOCK_H == 0
    )
    block_k = _BLOCK_H if _use_mfma else (16 if D >= 256 else 32)

    # ---- kv_splits auto-selection ----
    # Target: keep total CTAs near max_num_wg for good GPU utilization.
    # MFMA uses n_head_blocks = ceil(H/16) CTAs per token, vs H for scalar.
    # This reduces parallelism 16× → compensate with more kv_splits.
    max_num_wg = 256
    if kv_splits is None:
        max_kv_len    = kv_indices.shape[0]
        max_kv_splits = max(1, triton.cdiv(max_kv_len, block_k))
        n_head_blocks = (H + _BLOCK_H - 1) // _BLOCK_H if _use_mfma else H
        kv_splits     = max(1, max_num_wg // max(1, T_val * n_head_blocks))
        if _use_mfma:
            # MFMA groups BLOCK_H=16 heads per CTA, reducing head parallelism 16×.
            # Compensate with more kv_splits, but cap at 2 to avoid reduce overhead:
            # empirically kv_splits=2 is near-optimal for large T×H workloads.
            kv_splits = max(kv_splits, 2)
        kv_splits     = min(max_kv_splits, kv_splits)
        kv_splits     = triton.next_power_of_2(kv_splits)

    assert D % BLOCK_THREADS == 0, (
        f"D={D} must be divisible by {BLOCK_THREADS} (BLOCK_THREADS); "
        f"got D % {BLOCK_THREADS} = {D % BLOCK_THREADS}"
    )

    out = torch.zeros((T_val, H, D), dtype=torch.bfloat16, device=device)

    if kv_splits == 1:
        m_partial = l_partial = acc_partial = out  # dummy (not written by kernel)
    else:
        m_partial   = torch.empty((T_val, kv_splits, H), dtype=torch.float32, device=device)
        l_partial   = torch.empty_like(m_partial)
        acc_partial = torch.empty((T_val, kv_splits, H, D), dtype=torch.float32, device=device)

    # ---- Compile (or retrieve cached) kernel ----
    if _use_mfma:
        launcher = _compile_pa_decode_sparse_mfma(
            H=H,
            D=D,
            BLOCK_K=block_k,
            KV_SPLITS=kv_splits,
            softmax_scale=float(softmax_scale),
        )
    else:
        launcher = _compile_pa_decode_sparse(
            H=H,
            D=D,
            BLOCK_K=block_k,
            KV_SPLITS=kv_splits,
            softmax_scale=float(softmax_scale),
        )

    # ---- Launch FlyDSL kernel ----
    # T_size is annotated as fx.Int32 in the launcher, which accepts Python ints.
    # stream must be passed explicitly so that flyc.compile's CallState (built with
    # apply_defaults including stream) and subsequent cf(*args) calls agree on arg count.
    from .tensor_shim import _run_compiled
    fly_stream = fx.Stream(torch.cuda.current_stream())
    _run_compiled(
        launcher,
        q, unified_kv, kv_indices, kv_indptr, attn_sink,
        m_partial, l_partial, acc_partial, out,
        T_val,
        fly_stream,
    )

    if kv_splits == 1:
        return out

    if skip_reduce:
        return acc_partial, m_partial, l_partial

    # ---- Triton reduce kernel: combine KV_SPLITS partial states ----
    block_d      = triton.next_power_of_2(D)
    block_h_red  = 1
    grid_reduce  = (T_val, triton.cdiv(H, block_h_red))

    _pa_decode_sparse_reduce = _get_triton_reduce()
    _pa_decode_sparse_reduce[grid_reduce](
        m_partial,
        l_partial,
        acc_partial,
        attn_sink,
        kv_indptr,
        out,
        m_partial.stride(0),
        m_partial.stride(1),
        m_partial.stride(2),
        l_partial.stride(0),
        l_partial.stride(1),
        l_partial.stride(2),
        acc_partial.stride(0),
        acc_partial.stride(1),
        acc_partial.stride(2),
        acc_partial.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        H,
        D,
        kv_splits,
        BLOCK_H=block_h_red,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        USE_EXP2=True,   # FlyDSL kernel works in base-2 domain, same as Triton main
        num_warps=4,
    )
    return out
