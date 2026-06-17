# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""HCA-path FlyDSL compress + norm+rope+scatter kernels (2-kernel split).

Targeted optimization for V4-Pro HCA Main: D=512, RD=64, ratio=128, overlap=False.
Inspired by SGLang's `c128_v2.cuh`, adapted to AMD wave64 / flydsl with a
multi-wave LDS K-split (Phase 3 of the optimization series):

  Kernel A — flydsl_hca_compress_forward (multi-wave K-split)
    Grid:  (num_compress, NUM_SPLIT=head_dim/SLICE)
    Block: BLOCK_THREADS = 64 * k_split_num_waves (default 8 → 512 threads)
    Each block covers SLICE=64 head_dim elements of one boundary.
    K=128 split across NW waves (K_PER_WAVE = K/NW = 16). Per-wave local
    online-softmax + cross-wave LDS reduction. Each wave's K range splits
    at clamp(window_len, k_start, k_end) into a Phase 1 (state cache,
    padded softmax) sub-loop followed by a Phase 2 (ragged input) sub-loop.
    Output: kv_compressed[num_compress, head_dim] fp32 (compact, indexed by pid).

  Kernel B — flydsl_hca_norm_rope_scatter
    Grid:  (num_compress,)
    Block: BLOCK_THREADS=64 (1 wave)
    Each block reads one row of kv_compressed (full head_dim), does RMSNorm +
    GPT-J RoPE on the RD tail, scatters to paged kv_cache (BF16 only — HCA
    Main is the only HCA path that currently routes here; FP8 quant lives in
    the legacy single-kernel for now).

Why split into two kernels:
  Single-kernel HCA has 1 wave per boundary × K=128 serial chain = poor CU
  utilization at small N. Splitting head_dim into NUM_SPLIT=8 grid-Y blocks
  and parallelising K across NW=8 waves gives 1024 blocks at N=16, drastically
  cutting register pressure and shortening per-iter dependency chains.

Cost: extra HBM r/w of kv_compressed = num_compress * head_dim * 4 bytes.
For N=16384 D=512: 32 MB → ~4 us at 8 TB/s. Amortised by the compress kernel
speedup; after the ``slice_size`` + VEC=8 refactor the 2-kernel path beats
the legacy single-kernel at ALL N (1.06-3.7×, small N gets the largest win).

NOTE: HCA-only and BF16-only by design. CSA / Indexer / FP8 paths continue
to use the legacy single-kernel ``flydsl_fused_compress_attn``.
"""

import math
from contextlib import contextmanager
from functools import lru_cache
from typing import Optional

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, vector
from flydsl.expr import math as fmath
from flydsl.expr.arith import ArithValue, CmpFPredicate, CmpIPredicate
from flydsl.expr.typing import Int32, Stream, T
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from .tensor_shim import STensor, _to_raw, _run_compiled

# Force-bind LDS-related imports so isort/ruff/format hooks don't drop them
# (the multi-wave LDS kernel references CompilationContext, STensor,
# SmemAllocator, SmemPtr only inside @flyc.kernel / @flyc.jit closures,
# which formatters may not see).
_FORCE_BIND_LDS = (CompilationContext, STensor, SmemAllocator, SmemPtr, get_rocm_arch)

BLOCK_THREADS = 64  # 1 wave64
SLICE = 64  # head_dim elements per block (grid-Y split)
_NEG_INF = float("-inf")
_LOG2E = math.log2(math.e)


@contextmanager
def _if_then(if_op):
    with ir.InsertionPoint(if_op.then_block):
        try:
            yield if_op.then_block
        finally:
            blk = if_op.then_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])


# ============================================================================
# Kernel A: compress_forward with multi-wave LDS K-split
# ============================================================================


def _build_compress_forward_kernel(
    *,
    head_dim: int,
    ratio: int,
    state_size: int,
    k_split_num_waves: int = 8,
    slice_size: int = 64,
):
    """HCA compress_forward with K-axis parallelized across multiple waves.

    Architecture (multi-wave LDS K-split with per-thread VEC):
      - Grid:  (num_compress, NUM_SPLIT=head_dim/slice_size)
      - Block: BLOCK_THREADS = 64 * k_split_num_waves (8 waves on AMD).
      - Per block covers ``slice_size`` head_dim elements of one boundary.
      - Per thread owns ``VEC = slice_size / 64`` contiguous head_dim
        elements starting at lid*VEC within the block's slice.
      - K=ratio split across ``k_split_num_waves`` waves; each wave processes
        K_PER_WAVE = K/NW positions (= 16 for K=128, NW=8).
      - Per-wave local online-softmax → (m_local, kv_local, w_local) lists
        of VEC values per thread.
      - LDS cross-wave reduction: only wave 0 active; each thread reads
        NW*VEC values from LDS, computes VEC reduced compressed values,
        writes them out via vector buffer_store.

    Tuning knobs:
      - ``k_split_num_waves`` (= NW): trades K-serial chain length for LDS
        reduce cost. Small N → larger NW (more waves → more CU coverage);
        large N → smaller NW (less LDS overhead).
      - ``slice_size``: VEC width per thread. slice_size=64 → VEC=1 scalar
        (more blocks per boundary → small-N champion); slice_size=512 →
        VEC=8 (1 block per boundary, v1-like → large-N coalesced HBM).

    Phase 1 (state cache) is integrated by splitting each wave's K range at
    ``clamp(window_len, k_start, k_end)`` into a Phase 1 sub-loop reading
    kv_state + score_state (padded softmax when ``s < 0``) and a Phase 2
    sub-loop reading kv_in + score_in. Phase 2 in_row is clamped to ≥ 0
    so wasted reads in pure-Phase-1 iters stay in-bounds.
    """
    assert (
        head_dim % slice_size == 0
    ), f"head_dim={head_dim} must be divisible by slice_size={slice_size}"
    assert (
        slice_size % 64 == 0
    ), f"slice_size={slice_size} must be a multiple of 64 (wave width)"
    assert slice_size // 64 in (
        1,
        2,
        4,
        8,
    ), f"VEC={slice_size // 64} must be 1, 2, 4, or 8"
    assert (
        ratio % k_split_num_waves == 0
    ), f"K={ratio} must divide evenly across {k_split_num_waves} waves"
    assert state_size >= ratio, f"state_size={state_size} must be >= K={ratio}"
    D = head_dim
    K = ratio
    DIM_FULL = D
    SLICE_SZ = slice_size
    VEC = SLICE_SZ // 64  # per-lane head_dim element count
    NUM_SPLIT = D // SLICE_SZ
    NW = k_split_num_waves
    BLOCK_TH = 64 * NW
    K_PER_WAVE = K // NW

    # LDS layout: three independent fp32 arrays, each [NW * slice_size].
    LDS_M_ELEMS = NW * SLICE_SZ
    LDS_KV_ELEMS = NW * SLICE_SZ
    LDS_W_ELEMS = NW * SLICE_SZ
    LDS_M_BYTES = LDS_M_ELEMS * 4
    LDS_KV_BYTES = LDS_KV_ELEMS * 4
    LDS_W_BYTES = LDS_W_ELEMS * 4

    GPU_ARCH = get_rocm_arch()
    allocator = SmemAllocator(
        None,
        arch=GPU_ARCH,
        global_sym_name=(f"hca_compress_smem_D{D}_NW{NW}_SL{SLICE_SZ}_S{state_size}"),
    )
    lds_m_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_m_off + LDS_M_BYTES
    lds_kv_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_kv_off + LDS_KV_BYTES
    lds_w_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_w_off + LDS_W_BYTES

    _kname = (
        f"hca_compress_forward_D{D}_R{ratio}_NW{NW}_SL{SLICE_SZ}_S{state_size}_flydsl"
    )
    fm_fast = arith.FastMathFlags.fast

    @flyc.kernel(name=_kname, known_block_size=[BLOCK_TH, 1, 1])
    def kernel(
        kv_in: fx.Tensor,
        kv_in_row_stride: Int32,
        score_in: fx.Tensor,
        score_in_row_stride: Int32,
        plan: fx.Tensor,
        kv_state: fx.Tensor,  # [num_slots, STATE_SIZE, DIM_FULL] f32
        kv_state_slot_stride: Int32,  # f32 elements
        kv_state_pos_stride: Int32,
        score_state: fx.Tensor,
        score_state_slot_stride: Int32,
        score_state_pos_stride: Int32,
        state_slot_mapping: fx.Tensor,  # [bs] i32
        ape: fx.Tensor,
        kv_compressed: fx.Tensor,
        kv_compressed_row_stride: Int32,
    ):
        f32 = T.f32
        i32 = T.i32

        pid = fx.block_idx.x
        sid = fx.block_idx.y
        tid = fx.thread_idx.x  # 0..BLOCK_TH-1

        c_zero_i32 = arith.constant(0, type=i32)
        c_one_i32 = arith.constant(1, type=i32)
        c_64 = arith.constant(64, type=i32)
        c_neg_inf = arith.constant(_NEG_INF, type=f32)
        c_zero_f32 = arith.constant(0.0, type=f32)
        c_log2e = arith.constant(_LOG2E, type=f32)
        c_K_m1 = arith.constant(K - 1, type=i32)
        c_K_per_wave = arith.constant(K_PER_WAVE, type=i32)
        c_ratio = arith.constant(ratio, type=i32)
        c_DIM_FULL = arith.constant(DIM_FULL, type=i32)
        c_SLICE = arith.constant(SLICE_SZ, type=i32)
        c_VEC = arith.constant(VEC, type=i32)
        c_state_size = arith.constant(state_size, type=i32)

        def fexp_f32(x):
            return llvm.call_intrinsic(
                f32, "llvm.amdgcn.exp2.f32", [x * c_log2e], [], []
            )

        # Per-thread wave / lane (block-local).
        wid = arith.divsi(_to_raw(tid), c_64)  # ∈ [0, NW)
        lid = arith.remui(_to_raw(tid), c_64)  # ∈ [0, 64)

        # ── Load plan row ──────────────────────────────────────────────
        plan_rsrc = buffer_ops.create_buffer_resource(plan, max_size=True)
        plan_base = ArithValue(pid) * arith.constant(4, type=i32)
        plan_vec = buffer_ops.buffer_load(plan_rsrc, plan_base, vec_width=4, dtype=i32)
        ragged_id = vector.extract(plan_vec, static_position=[0], dynamic_position=[])
        batch_id = vector.extract(plan_vec, static_position=[1], dynamic_position=[])
        position = vector.extract(plan_vec, static_position=[2], dynamic_position=[])
        window_len = vector.extract(plan_vec, static_position=[3], dynamic_position=[])

        is_active = arith.cmpi(CmpIPredicate.sge, _to_raw(position), c_zero_i32)
        _if_active = scf.IfOp(is_active)
        with _if_then(_if_active):
            # Per-thread head_dim base: each thread owns VEC contiguous
            # elements starting at slice_base + lid * VEC.
            slice_base_i32 = ArithValue(sid) * c_SLICE
            col_off_base = slice_base_i32 + ArithValue(lid) * c_VEC

            slot_map_rsrc = buffer_ops.create_buffer_resource(
                state_slot_mapping, max_size=True
            )
            slot = buffer_ops.buffer_load(
                slot_map_rsrc, batch_id, vec_width=1, dtype=i32
            )

            kv_in_rsrc = buffer_ops.create_buffer_resource(kv_in, max_size=True)
            score_in_rsrc = buffer_ops.create_buffer_resource(score_in, max_size=True)
            kv_state_rsrc = buffer_ops.create_buffer_resource(kv_state, max_size=True)
            score_state_rsrc = buffer_ops.create_buffer_resource(
                score_state, max_size=True
            )
            ape_rsrc = buffer_ops.create_buffer_resource(ape, max_size=True)

            def _load_bf16_vec_to_f32(rsrc, base_off_elems_i32):
                """Load VEC contiguous bf16 elements starting at
                ``base_off_elems_i32`` → list of VEC f32 values.

                VEC=1: unaligned-safe scalar via dword + bit-extract.
                VEC>=2: vectorized i32 buffer_load + bitcast to bf16.
                """
                if const_expr(VEC == 1):
                    off_dw = ArithValue(base_off_elems_i32) >> c_one_i32
                    lane_in_dw = arith.andi(_to_raw(base_off_elems_i32), c_one_i32)
                    raw_s = buffer_ops.buffer_load(rsrc, off_dw, vec_width=1, dtype=i32)
                    hi = ArithValue(raw_s) >> arith.constant(16, type=i32)
                    lo_or_hi = arith.select(
                        arith.cmpi(CmpIPredicate.eq, lane_in_dw, c_zero_i32),
                        raw_s,
                        _to_raw(hi),
                    )
                    lo16 = arith.andi(lo_or_hi, arith.constant(0xFFFF, type=i32))
                    lo16_v = vector.from_elements(T.vec(1, T.i32), [lo16])
                    bf16_pair = vector.bitcast(T.vec(2, T.bf16), lo16_v)
                    bf16_v = vector.extract(
                        bf16_pair, static_position=[0], dynamic_position=[]
                    )
                    return [arith.extf(f32, bf16_v)]
                else:
                    # base must be VEC-aligned (caller guarantees by
                    # col_off_base = sid*SLICE + lid*VEC, both multiples of VEC).
                    off_dw = ArithValue(base_off_elems_i32) >> c_one_i32
                    dwords = VEC // 2  # VEC bf16 = VEC*2 bytes
                    if const_expr(dwords == 1):
                        # buffer_load(vec_width=1) returns scalar i32; wrap
                        # into vec<1xi32> before bitcast to vec<2xbf16>.
                        raw_s = buffer_ops.buffer_load(
                            rsrc, off_dw, vec_width=1, dtype=i32
                        )
                        raw = vector.from_elements(T.vec(1, T.i32), [raw_s])
                    else:
                        raw = buffer_ops.buffer_load(
                            rsrc, off_dw, vec_width=dwords, dtype=i32
                        )
                    vec_bf16 = vector.bitcast(T.vec(VEC, T.bf16), raw)
                    out = []
                    for i in range_constexpr(VEC):
                        bf16_v = vector.extract(
                            vec_bf16, static_position=[i], dynamic_position=[]
                        )
                        out.append(arith.extf(f32, bf16_v))
                    return out

            def _load_f32_vec(rsrc, base_off_elems_i32):
                """Load VEC f32 starting at base → list of VEC f32 values."""
                if const_expr(VEC <= 4):
                    raw = buffer_ops.buffer_load(
                        rsrc, base_off_elems_i32, vec_width=VEC, dtype=f32
                    )
                    if const_expr(VEC == 1):
                        # vec_width=1 returns scalar, not 1-vec.
                        return [raw]
                    return [
                        vector.extract(raw, static_position=[i], dynamic_position=[])
                        for i in range(VEC)
                    ]
                else:
                    # VEC == 8: AMD HW max is dwordx4 → 2 loads.
                    assert VEC == 8
                    half = VEC // 2
                    r0 = buffer_ops.buffer_load(
                        rsrc, base_off_elems_i32, vec_width=half, dtype=f32
                    )
                    r1 = buffer_ops.buffer_load(
                        rsrc,
                        ArithValue(base_off_elems_i32) + arith.constant(half, type=i32),
                        vec_width=half,
                        dtype=f32,
                    )
                    out = []
                    for i in range_constexpr(half):
                        out.append(
                            vector.extract(r0, static_position=[i], dynamic_position=[])
                        )
                    for i in range_constexpr(half):
                        out.append(
                            vector.extract(r1, static_position=[i], dynamic_position=[])
                        )
                    return out

            def _issue_phase2_loads(k_i32):
                """Phase 2 (ragged input) loads. Returns (kv_list, sc_list,
                ape_list) each of length VEC."""
                ape_row = arith.remui(k_i32, c_ratio)
                tmp = arith.subi(c_K_m1, k_i32)
                in_row_raw = arith.subi(_to_raw(ragged_id), tmp)
                in_row = arith.maxsi(in_row_raw, c_zero_i32)
                base_in_off = (
                    ArithValue(in_row) * ArithValue(kv_in_row_stride) + col_off_base
                )
                base_sc_off = (
                    ArithValue(in_row) * ArithValue(score_in_row_stride) + col_off_base
                )
                base_ape_off = ArithValue(ape_row) * c_DIM_FULL + col_off_base
                kv = _load_bf16_vec_to_f32(kv_in_rsrc, base_in_off)
                sc = _load_bf16_vec_to_f32(score_in_rsrc, base_sc_off)
                ape_v = _load_f32_vec(ape_rsrc, base_ape_off)
                return kv, sc, ape_v

            def _issue_phase1_loads(k_i32):
                """Phase 1 (state cache) loads. Returns (kv_list, sc_padded_list)
                each of length VEC. Score is -inf when s < 0."""
                s = arith.addi(
                    arith.subi(_to_raw(position), c_K_m1),
                    k_i32,
                )
                is_pad = arith.cmpi(CmpIPredicate.slt, s, c_zero_i32)
                s_safe = arith.select(is_pad, c_zero_i32, s)
                ring = arith.remui(s_safe, c_state_size)
                base_kv_off = (
                    ArithValue(slot) * ArithValue(kv_state_slot_stride)
                    + ArithValue(ring) * ArithValue(kv_state_pos_stride)
                    + col_off_base
                )
                base_sc_off = (
                    ArithValue(slot) * ArithValue(score_state_slot_stride)
                    + ArithValue(ring) * ArithValue(score_state_pos_stride)
                    + col_off_base
                )
                kv_list = _load_f32_vec(kv_state_rsrc, base_kv_off)
                sc_list = _load_f32_vec(score_state_rsrc, base_sc_off)
                sc_padded = [
                    arith.select(is_pad, c_neg_inf, sc_list[i]) for i in range(VEC)
                ]
                return kv_list, sc_padded

            def _softmax_step_padded(
                m_old_list, kv_old_list, w_old_list, score_k_list, kv_k_list
            ):
                """Padding-aware vector softmax step over VEC lanes. When
                score_k == -inf, w_k is forced to 0 (avoids NaN when m_old
                is also -inf). Safe in both Phase 1 (padding can occur) and
                Phase 2 (score finite → pad-select branch is dead code).
                """
                new_m, new_kv, new_w = [], [], []
                for i in range_constexpr(VEC):
                    m_old = m_old_list[i]
                    kv_old = kv_old_list[i]
                    w_old = w_old_list[i]
                    score_k = score_k_list[i]
                    kv_k = kv_k_list[i]
                    m_new = arith.maximumf(m_old, score_k)
                    is_first = arith.cmpf(CmpFPredicate.OEQ, m_old, c_neg_inf)
                    scale_active = fexp_f32(arith.subf(m_old, m_new))
                    scale_v = arith.select(is_first, c_zero_f32, scale_active)
                    wk_active = fexp_f32(arith.subf(score_k, m_new))
                    is_pad_score = arith.cmpf(CmpFPredicate.OEQ, score_k, c_neg_inf)
                    w_k = arith.select(is_pad_score, c_zero_f32, wk_active)
                    new_kv.append(
                        arith.AddFOp(
                            arith.MulFOp(kv_old, scale_v, fastmath=fm_fast).result,
                            arith.MulFOp(w_k, kv_k, fastmath=fm_fast).result,
                            fastmath=fm_fast,
                        ).result
                    )
                    new_w.append(
                        arith.AddFOp(
                            arith.MulFOp(w_old, scale_v, fastmath=fm_fast).result,
                            w_k,
                            fastmath=fm_fast,
                        ).result
                    )
                    new_m.append(m_new)
                return new_m, new_kv, new_w

            # ── Wave's K range: [wid * K_PER_WAVE, (wid+1) * K_PER_WAVE) ──
            k_start_i32 = ArithValue(wid) * c_K_per_wave
            k_end_i32 = k_start_i32 + c_K_per_wave

            # Split point inside this wave's K range. Each wave sees a
            # window_len-dependent slice of Phase 1 followed by Phase 2.
            # Cases (`wl = window_len`):
            #   wl ≤ k_start:  pure Phase 2 (entire wave is input)
            #   wl ≥ k_end:    pure Phase 1 (entire wave is state cache)
            #   else:          mixed (Phase 1 in [k_start, wl), Phase 2 in [wl, k_end))
            # ``split`` = clamp(wl, k_start, k_end) gives the boundary;
            # both sub-loops are empty when their bound collapses, so any
            # of the three cases naturally falls out.
            wl_i32 = _to_raw(window_len)
            split_lo = arith.maxsi(wl_i32, _to_raw(k_start_i32))
            split_i32 = arith.minsi(split_lo, _to_raw(k_end_i32))

            # State is 3*VEC scalars: m_lane[VEC] + kv_lane[VEC] + w_lane[VEC].
            init_m = [c_neg_inf for _ in range(VEC)]
            init_kv = [c_zero_f32 for _ in range(VEC)]
            init_w = [c_zero_f32 for _ in range(VEC)]
            init_state = init_m + init_kv + init_w

            # Sub-loop 1: Phase 1 sub-range [k_start, split). Reads state
            # cache; padded softmax (score can be -inf).
            phase1_local = init_state
            for k_static, state in range(
                _to_raw(k_start_i32), _to_raw(split_i32), 1, init=init_state
            ):
                m_lane = list(state[0:VEC])
                kv_lane = list(state[VEC : 2 * VEC])
                w_lane = list(state[2 * VEC : 3 * VEC])
                k_i32 = arith.index_cast(i32, _to_raw(k_static))
                kv_v, sc_v = _issue_phase1_loads(k_i32)
                new_m, new_kv, new_w = _softmax_step_padded(
                    m_lane, kv_lane, w_lane, sc_v, kv_v
                )
                phase1_local = yield list(new_m) + list(new_kv) + list(new_w)

            # Sub-loop 2: Phase 2 sub-range [split, k_end). Reads input;
            # uses padded softmax (the is-pad-score branch is dead code
            # since Phase 2 scores are always finite — compiler elides).
            # Carry Phase 1's accumulator through as init.
            final = phase1_local
            for k_static, state in range(
                _to_raw(split_i32), _to_raw(k_end_i32), 1, init=phase1_local
            ):
                m_lane = list(state[0:VEC])
                kv_lane = list(state[VEC : 2 * VEC])
                w_lane = list(state[2 * VEC : 3 * VEC])
                k_i32 = arith.index_cast(i32, _to_raw(k_static))
                p2_kv, p2_sc, p2_ape = _issue_phase2_loads(k_i32)
                p2_score = [
                    arith.AddFOp(p2_sc[i], p2_ape[i], fastmath=fm_fast).result
                    for i in range(VEC)
                ]
                new_m, new_kv, new_w = _softmax_step_padded(
                    m_lane, kv_lane, w_lane, p2_score, p2_kv
                )
                final = yield list(new_m) + list(new_kv) + list(new_w)

            m_local = list(final[0:VEC])
            kv_local = list(final[VEC : 2 * VEC])
            w_local = list(final[2 * VEC : 3 * VEC])

            # ── LDS write: each thread writes VEC entries per array ──
            # Layout: per array, NW * SLICE_SZ fp32 entries; per-thread
            # base = wid * SLICE_SZ + lid * VEC; thread writes VEC values
            # at base+0, base+1, ..., base+VEC-1.
            lds_base = allocator.get_base()
            lds_m = STensor(
                SmemPtr(lds_base, lds_m_off, T.f32, shape=(LDS_M_ELEMS,)),
                dtype=T.f32,
                shape=(LDS_M_ELEMS,),
            )
            lds_kv = STensor(
                SmemPtr(lds_base, lds_kv_off, T.f32, shape=(LDS_KV_ELEMS,)),
                dtype=T.f32,
                shape=(LDS_KV_ELEMS,),
            )
            lds_w = STensor(
                SmemPtr(lds_base, lds_w_off, T.f32, shape=(LDS_W_ELEMS,)),
                dtype=T.f32,
                shape=(LDS_W_ELEMS,),
            )
            lds_thread_base = ArithValue(wid) * c_SLICE + ArithValue(lid) * c_VEC
            for i in range_constexpr(VEC):
                idx_i = lds_thread_base + arith.constant(i, type=i32)
                lds_m[fx.Index(idx_i)] = m_local[i]
                lds_kv[fx.Index(idx_i)] = kv_local[i]
                lds_w[fx.Index(idx_i)] = w_local[i]

            gpu.barrier()

            # ── Cross-wave reduction: only wave 0 reads and reduces ──
            # Wave 0's 64 threads cover SLICE_SZ = 64 * VEC head_dim elements
            # (VEC elements per thread). For each owned element, the thread
            # reads NW values from LDS (one per K-split wave) and computes
            # the global online-softmax.
            is_wave0 = arith.cmpi(CmpIPredicate.eq, wid, c_zero_i32)
            _if_w0 = scf.IfOp(is_wave0)
            with _if_then(_if_w0):
                comp_list = []
                for i in range_constexpr(VEC):
                    lane_off = ArithValue(lid) * c_VEC + arith.constant(i, type=i32)
                    # Global max across NW waves for this element.
                    m_g = c_neg_inf
                    m_arr = []
                    for w in range_constexpr(NW):
                        idx_w = arith.constant(w * SLICE_SZ, type=i32) + lane_off
                        m_w = lds_m[fx.Index(idx_w)]
                        m_arr.append(m_w)
                        m_g = arith.maximumf(m_g, m_w)

                    # Weighted sums (kv * scale_w) and (w * scale_w).
                    kv_sum = c_zero_f32
                    w_sum = c_zero_f32
                    for w in range_constexpr(NW):
                        idx_w = arith.constant(w * SLICE_SZ, type=i32) + lane_off
                        kv_w = lds_kv[fx.Index(idx_w)]
                        w_w = lds_w[fx.Index(idx_w)]
                        m_w = m_arr[w]
                        scale_w = fexp_f32(arith.subf(m_w, m_g))
                        kv_sum = arith.AddFOp(
                            kv_sum,
                            arith.MulFOp(kv_w, scale_w, fastmath=fm_fast).result,
                            fastmath=fm_fast,
                        ).result
                        w_sum = arith.AddFOp(
                            w_sum,
                            arith.MulFOp(w_w, scale_w, fastmath=fm_fast).result,
                            fastmath=fm_fast,
                        ).result
                    rcp_w = llvm.call_intrinsic(
                        f32, "llvm.amdgcn.rcp.f32", [w_sum], [], []
                    )
                    comp_list.append(
                        arith.MulFOp(kv_sum, rcp_w, fastmath=fm_fast).result
                    )

                # ── Vectorized write of VEC f32 comp values ──
                out_rsrc = buffer_ops.create_buffer_resource(
                    kv_compressed, max_size=True
                )
                out_off = (
                    ArithValue(pid) * ArithValue(kv_compressed_row_stride)
                    + col_off_base
                )
                if const_expr(VEC == 1):
                    buffer_ops.buffer_store(comp_list[0], out_rsrc, out_off)
                elif const_expr(VEC <= 4):
                    out_vec = vector.from_elements(T.vec(VEC, T.f32), comp_list)
                    buffer_ops.buffer_store(out_vec, out_rsrc, out_off)
                else:
                    # VEC == 8: AMD HW max is dwordx4 → 2 stores.
                    assert VEC == 8
                    half = VEC // 2
                    v0 = vector.from_elements(T.vec(half, T.f32), comp_list[0:half])
                    v1 = vector.from_elements(T.vec(half, T.f32), comp_list[half:VEC])
                    buffer_ops.buffer_store(v0, out_rsrc, out_off)
                    buffer_ops.buffer_store(
                        v1,
                        out_rsrc,
                        ArithValue(out_off) + arith.constant(half, type=i32),
                    )

    @flyc.jit
    def launch_hca_compress_forward(
        kv_in: fx.Tensor,
        kv_in_row_stride: fx.Int32,
        score_in: fx.Tensor,
        score_in_row_stride: fx.Int32,
        plan: fx.Tensor,
        kv_state: fx.Tensor,
        kv_state_slot_stride: fx.Int32,
        kv_state_pos_stride: fx.Int32,
        score_state: fx.Tensor,
        score_state_slot_stride: fx.Int32,
        score_state_pos_stride: fx.Int32,
        state_slot_mapping: fx.Tensor,
        ape: fx.Tensor,
        kv_compressed: fx.Tensor,
        kv_compressed_row_stride: fx.Int32,
        plan_capacity: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        # Materialize the LDS global symbol inside the gpu_module body
        # (the SmemAllocator was declared outside the kernel decorator).
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        idx_p = arith.index_cast(T.index, _to_raw(plan_capacity))
        idx_s = arith.index_cast(T.index, arith.constant(NUM_SPLIT, type=T.i32))
        k = kernel(
            kv_in,
            kv_in_row_stride,
            score_in,
            score_in_row_stride,
            plan,
            kv_state,
            kv_state_slot_stride,
            kv_state_pos_stride,
            score_state,
            score_state_slot_stride,
            score_state_pos_stride,
            state_slot_mapping,
            ape,
            kv_compressed,
            kv_compressed_row_stride,
        )
        k.launch(
            grid=(idx_p, idx_s, 1),
            block=(BLOCK_TH, 1, 1),
            stream=stream,
        )

    return launch_hca_compress_forward


# ============================================================================
# Kernel B: norm + rope + scatter (BF16, per-row)
# ============================================================================


def _build_norm_rope_scatter_kernel(
    *,
    head_dim: int,
    rope_head_dim: int,
    ratio: int,
    k_per_block: int,
    rms_weight_is_bf16: bool,
    rms_eps: float,
):
    """Build per-row RMSNorm + GPT-J RoPE + BF16 paged scatter for HCA.

    Reads kv_compressed[num_compress, head_dim] fp32 and the plan; for each
    boundary, normalizes / rotates / scatters into kv_cache.
    """
    D = head_dim
    RD = rope_head_dim
    NOPE = D - RD
    VEC = D // BLOCK_THREADS  # 8 for D=512
    ROPE_THREAD_LO = NOPE // VEC
    PAIRS_PER_THREAD = VEC // 2

    assert D % BLOCK_THREADS == 0
    assert RD > 0 and RD % 2 == 0 and RD % VEC == 0

    _kname = (
        f"hca_norm_rope_scatter_D{D}_RD{RD}_R{ratio}_KB{k_per_block}"
        f"{'_rmsbf16' if rms_weight_is_bf16 else ''}_flydsl"
    )
    fm_fast = arith.FastMathFlags.fast
    log2_block = int(math.log2(BLOCK_THREADS))

    @flyc.kernel(name=_kname)
    def kernel(
        kv_compressed: fx.Tensor,  # [num_compress, head_dim] f32
        kv_compressed_row_stride: Int32,
        plan: fx.Tensor,  # [num_compress, 4] i32
        rms_weight: fx.Tensor,  # [head_dim] bf16 or f32
        cos_cache: fx.Tensor,  # [max_pos, RD/2] bf16
        sin_cache: fx.Tensor,
        kv_cache: fx.Tensor,  # [NB, k_per_block, D] bf16
        kv_cache_block_stride: Int32,  # bf16 elements
        kv_cache_token_stride: Int32,
        block_table: fx.Tensor,  # [bs, max_blocks_per_seq] i32
        block_table_seq_stride: Int32,
    ):
        f32 = T.f32
        i32 = T.i32
        vecVf32 = T.vec(VEC, T.f32)

        pid = fx.block_idx.x
        tid = fx.thread_idx.x

        c_zero_i32 = arith.constant(0, type=i32)
        c_one_i32 = arith.constant(1, type=i32)
        c_eps = arith.constant(rms_eps, type=f32)
        c_inv_D = arith.constant(1.0 / D, type=f32)
        c_ratio = arith.constant(ratio, type=i32)
        c_k_per_block = arith.constant(k_per_block, type=i32)

        def wave_reduce_add(x):
            w = _to_raw(x)
            for sh_exp in range_constexpr(log2_block):
                off = BLOCK_THREADS // (2 << sh_exp)
                peer = _to_raw(ArithValue(w).shuffle_xor(off, BLOCK_THREADS))
                w = arith.AddFOp(w, peer, fastmath=fm_fast).result
            return w

        # ── Load plan row ──
        plan_rsrc = buffer_ops.create_buffer_resource(plan, max_size=True)
        plan_base = ArithValue(pid) * arith.constant(4, type=i32)
        plan_vec = buffer_ops.buffer_load(plan_rsrc, plan_base, vec_width=4, dtype=i32)
        batch_id = vector.extract(plan_vec, static_position=[1], dynamic_position=[])
        position = vector.extract(plan_vec, static_position=[2], dynamic_position=[])

        is_active = arith.cmpi(CmpIPredicate.sge, _to_raw(position), c_zero_i32)
        _if_active = scf.IfOp(is_active)
        with _if_then(_if_active):
            tid_x_vec = ArithValue(tid) * arith.constant(VEC, type=i32)

            # ── Load kv_compressed[pid, tid*VEC : tid*VEC + VEC] ──
            kvc_rsrc = buffer_ops.create_buffer_resource(kv_compressed, max_size=True)
            base_off = (
                ArithValue(pid) * ArithValue(kv_compressed_row_stride) + tid_x_vec
            )
            # VEC ∈ {2, 4, 8}: VEC <= 4 → single dwordx{VEC}; VEC=8 → 2× dwordx4.
            if const_expr(VEC <= 4):
                raw = buffer_ops.buffer_load(
                    kvc_rsrc, base_off, vec_width=VEC, dtype=f32
                )
                comp_lane = [
                    vector.extract(raw, static_position=[i], dynamic_position=[])
                    for i in range(VEC)
                ]
            else:
                assert VEC == 8
                half = 4
                r0 = buffer_ops.buffer_load(
                    kvc_rsrc, base_off, vec_width=half, dtype=f32
                )
                r1 = buffer_ops.buffer_load(
                    kvc_rsrc,
                    ArithValue(base_off) + arith.constant(half, type=i32),
                    vec_width=half,
                    dtype=f32,
                )
                comp_lane = [
                    vector.extract(r0, static_position=[i], dynamic_position=[])
                    for i in range(half)
                ] + [
                    vector.extract(r1, static_position=[i], dynamic_position=[])
                    for i in range(half)
                ]

            # ── RMSNorm (wave reduce-add of squares / D + eps; rsqrt) ──
            sq_local = arith.constant(0.0, type=f32)
            for i in range_constexpr(VEC):
                sq_local = arith.AddFOp(
                    sq_local,
                    arith.MulFOp(comp_lane[i], comp_lane[i], fastmath=fm_fast).result,
                    fastmath=fm_fast,
                ).result
            sq_full = wave_reduce_add(sq_local)
            var = arith.MulFOp(sq_full, c_inv_D, fastmath=fm_fast).result
            rrms = fmath.rsqrt(
                arith.AddFOp(var, c_eps, fastmath=fm_fast).result, fastmath=fm_fast
            )

            # rms_weight load
            rmsw_rsrc = buffer_ops.create_buffer_resource(rms_weight, max_size=True)
            if const_expr(rms_weight_is_bf16):
                dwords = (VEC + 1) // 2
                off_dw = ArithValue(tid_x_vec) >> c_one_i32
                if const_expr(dwords == 1):
                    raw_s = buffer_ops.buffer_load(
                        rmsw_rsrc, off_dw, vec_width=1, dtype=i32
                    )
                    raw = vector.from_elements(T.vec(1, T.i32), [raw_s])
                else:
                    raw = buffer_ops.buffer_load(
                        rmsw_rsrc, off_dw, vec_width=dwords, dtype=i32
                    )
                vec_bf16 = vector.bitcast(T.vec(VEC, T.bf16), raw)
                rmsw_lane = []
                for i in range_constexpr(VEC):
                    bf16_v = vector.extract(
                        vec_bf16, static_position=[i], dynamic_position=[]
                    )
                    rmsw_lane.append(arith.extf(f32, bf16_v))
            else:
                if const_expr(VEC <= 4):
                    raw = buffer_ops.buffer_load(
                        rmsw_rsrc, tid_x_vec, vec_width=VEC, dtype=f32
                    )
                    rmsw_lane = [
                        vector.extract(raw, static_position=[i], dynamic_position=[])
                        for i in range(VEC)
                    ]
                else:
                    half = 4
                    r0 = buffer_ops.buffer_load(
                        rmsw_rsrc, tid_x_vec, vec_width=half, dtype=f32
                    )
                    r1 = buffer_ops.buffer_load(
                        rmsw_rsrc,
                        ArithValue(tid_x_vec) + arith.constant(half, type=i32),
                        vec_width=half,
                        dtype=f32,
                    )
                    rmsw_lane = [
                        vector.extract(r0, static_position=[i], dynamic_position=[])
                        for i in range(half)
                    ] + [
                        vector.extract(r1, static_position=[i], dynamic_position=[])
                        for i in range(half)
                    ]

            normed_lane = [
                arith.MulFOp(
                    arith.MulFOp(comp_lane[i], rrms, fastmath=fm_fast).result,
                    rmsw_lane[i],
                    fastmath=fm_fast,
                ).result
                for i in range(VEC)
            ]

            # ── GPT-J RoPE on RD tail ──
            comp_pos_i32 = arith.muli(arith.divsi(_to_raw(position), c_ratio), c_ratio)
            cos_rsrc = buffer_ops.create_buffer_resource(cos_cache, max_size=True)
            sin_rsrc = buffer_ops.create_buffer_resource(sin_cache, max_size=True)
            c_half_rd = arith.constant(RD // 2, type=i32)
            cos_row_base = ArithValue(comp_pos_i32) * c_half_rd

            is_rope_t = arith.cmpi(
                CmpIPredicate.sge,
                _to_raw(tid),
                arith.constant(ROPE_THREAD_LO, type=i32),
            )
            rope_rel_raw = ArithValue(tid) - arith.constant(ROPE_THREAD_LO, type=i32)
            rope_rel = arith.maxsi(rope_rel_raw, c_zero_i32)
            cs_lo = ArithValue(rope_rel) * arith.constant(PAIRS_PER_THREAD, type=i32)

            if const_expr(PAIRS_PER_THREAD == 1):
                cos_b = buffer_ops.buffer_load(
                    cos_rsrc, cos_row_base + cs_lo, vec_width=1, dtype=T.bf16
                )
                sin_b = buffer_ops.buffer_load(
                    sin_rsrc, cos_row_base + cs_lo, vec_width=1, dtype=T.bf16
                )
                cos_vals = [arith.extf(f32, cos_b)]
                sin_vals = [arith.extf(f32, sin_b)]
            else:
                cos_vec = buffer_ops.buffer_load(
                    cos_rsrc,
                    cos_row_base + cs_lo,
                    vec_width=PAIRS_PER_THREAD,
                    dtype=T.bf16,
                )
                sin_vec = buffer_ops.buffer_load(
                    sin_rsrc,
                    cos_row_base + cs_lo,
                    vec_width=PAIRS_PER_THREAD,
                    dtype=T.bf16,
                )
                cos_vals = [
                    arith.extf(
                        f32,
                        vector.extract(
                            cos_vec, static_position=[i], dynamic_position=[]
                        ),
                    )
                    for i in range(PAIRS_PER_THREAD)
                ]
                sin_vals = [
                    arith.extf(
                        f32,
                        vector.extract(
                            sin_vec, static_position=[i], dynamic_position=[]
                        ),
                    )
                    for i in range(PAIRS_PER_THREAD)
                ]

            rotated_lane = list(normed_lane)
            for k in range_constexpr(PAIRS_PER_THREAD):
                e = normed_lane[2 * k]
                o = normed_lane[2 * k + 1]
                c = cos_vals[k]
                s = sin_vals[k]
                new_e = arith.subf(
                    arith.MulFOp(e, c, fastmath=fm_fast).result,
                    arith.MulFOp(o, s, fastmath=fm_fast).result,
                )
                new_o = arith.AddFOp(
                    arith.MulFOp(e, s, fastmath=fm_fast).result,
                    arith.MulFOp(o, c, fastmath=fm_fast).result,
                    fastmath=fm_fast,
                ).result
                rotated_lane[2 * k] = new_e
                rotated_lane[2 * k + 1] = new_o

            out_lane = [
                arith.select(is_rope_t, rotated_lane[i], normed_lane[i])
                for i in range_constexpr(VEC)
            ]

            # ── Paged BF16 scatter ──
            ci = arith.divsi(_to_raw(position), c_ratio)
            block_in_seq = arith.divsi(ci, c_k_per_block)
            slot_in_block = arith.remui(ci, c_k_per_block)
            bt_rsrc = buffer_ops.create_buffer_resource(block_table, max_size=True)
            bt_off = ArithValue(batch_id) * ArithValue(
                block_table_seq_stride
            ) + ArithValue(block_in_seq)
            physical_block = buffer_ops.buffer_load(
                bt_rsrc, bt_off, vec_width=1, dtype=i32
            )

            cache_off = (
                ArithValue(physical_block) * ArithValue(kv_cache_block_stride)
                + ArithValue(slot_in_block) * ArithValue(kv_cache_token_stride)
                + tid_x_vec
            )
            out_vec_t = T.vec(VEC, T.bf16)
            raw_vec = vector.from_elements(vecVf32, out_lane)
            bf16_vec = raw_vec.truncf(out_vec_t)
            out_rsrc = buffer_ops.create_buffer_resource(kv_cache, max_size=True)
            cache_off_dw = ArithValue(cache_off) >> c_one_i32
            dwords = (VEC + 1) // 2
            bf16_as_i32 = vector.bitcast(T.vec(dwords, T.i32), bf16_vec)
            if const_expr(dwords == 1):
                scalar_i32 = vector.extract(
                    bf16_as_i32, static_position=[0], dynamic_position=[]
                )
                buffer_ops.buffer_store(scalar_i32, out_rsrc, cache_off_dw)
            else:
                buffer_ops.buffer_store(bf16_as_i32, out_rsrc, cache_off_dw)

    @flyc.jit
    def launch_hca_norm_rope_scatter(
        kv_compressed: fx.Tensor,
        kv_compressed_row_stride: fx.Int32,
        plan: fx.Tensor,
        rms_weight: fx.Tensor,
        cos_cache: fx.Tensor,
        sin_cache: fx.Tensor,
        kv_cache: fx.Tensor,
        kv_cache_block_stride: fx.Int32,
        kv_cache_token_stride: fx.Int32,
        block_table: fx.Tensor,
        block_table_seq_stride: fx.Int32,
        plan_capacity: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        idx_p = arith.index_cast(T.index, _to_raw(plan_capacity))
        k = kernel(
            kv_compressed,
            kv_compressed_row_stride,
            plan,
            rms_weight,
            cos_cache,
            sin_cache,
            kv_cache,
            kv_cache_block_stride,
            kv_cache_token_stride,
            block_table,
            block_table_seq_stride,
        )
        k.launch(
            grid=(idx_p, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_hca_norm_rope_scatter


# ============================================================================
# Cached compile + public API
# ============================================================================


_DEFAULT_COMPILE_HINTS = {
    "waves_per_eu": 8,
    "fast_fp_math": True,
    "unsafe_fp_math": True,
}


@lru_cache(maxsize=32)
def compile_hca_compress_forward(
    *,
    head_dim: int,
    ratio: int,
    state_size: int,
    k_split_num_waves: int = 8,
    slice_size: int = 64,
):
    """Build the HCA compress_forward launcher (multi-wave LDS K-split).

    Each wave handles K / ``k_split_num_waves`` K-positions; cross-wave LDS
    reduction merges per-wave softmax accumulators. Each iter selects
    between Phase 1 (state cache, ``k < window_len``) and Phase 2 (input)
    by splitting the wave's K range at ``clamp(window_len, k_start, k_end)``.

    ``slice_size`` controls per-thread vector width (VEC = slice_size / 64).
    Larger slice_size means each thread handles more head_dim elements per
    K-iter (wider buffer_load → better HBM coalescing), but fewer blocks
    per boundary (NUM_SPLIT = head_dim / slice_size). slice_size=64 → VEC=1
    (8 blocks/boundary, small-N champion); slice_size=512 → VEC=8
    (1 block/boundary, v1-like HBM access, large-N champion).

    ``state_size`` is the ring-buffer modulo of ``kv_state.shape[1]`` (≥ ratio).
    Cached per (head_dim, ratio, state_size, k_split_num_waves, slice_size) tuple.
    """
    launcher = _build_compress_forward_kernel(
        head_dim=head_dim,
        ratio=ratio,
        state_size=state_size,
        k_split_num_waves=k_split_num_waves,
        slice_size=slice_size,
    )
    launcher.compile_hints = dict(_DEFAULT_COMPILE_HINTS)
    return launcher


@lru_cache(maxsize=8)
def compile_hca_norm_rope_scatter(
    *,
    head_dim: int,
    rope_head_dim: int,
    ratio: int,
    k_per_block: int,
    rms_weight_is_bf16: bool,
    rms_eps: float,
):
    launcher = _build_norm_rope_scatter_kernel(
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        ratio=ratio,
        k_per_block=k_per_block,
        rms_weight_is_bf16=rms_weight_is_bf16,
        rms_eps=rms_eps,
    )
    launcher.compile_hints = dict(_DEFAULT_COMPILE_HINTS)
    return launcher


def flydsl_hca_compress_attn(
    *,
    kv_in: torch.Tensor,  # [num_q_tokens, head_dim] bf16
    score_in: torch.Tensor,  # [num_q_tokens, head_dim] bf16
    kv_state: torch.Tensor,  # [num_slots, STATE_SIZE, head_dim] f32
    score_state: torch.Tensor,  # same shape as kv_state
    state_slot_mapping: torch.Tensor,  # [bs] i32
    plan_gpu: torch.Tensor,  # [num_compress, 4] i32
    ape: torch.Tensor,  # [ratio, head_dim] f32
    rms_weight: torch.Tensor,  # [head_dim] f32 or bf16
    rms_eps: float,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    k_per_block: int,
    ratio: int,
    head_dim: int,
    rope_head_dim: int,
    kv_compressed_scratch: Optional[torch.Tensor] = None,
    k_split_num_waves: Optional[int] = None,
    slice_size: Optional[int] = None,
    stream: Optional[torch.cuda.Stream] = None,
) -> None:
    """HCA-only 2-kernel compress + norm+rope+scatter (V4-Pro Main path).

    Restrictions: ratio=128, overlap=False (implicit), head_dim=512 supported.
    BF16 cache scatter only.

    Phase 1 (state cache) is enabled by passing real ``kv_state`` /
    ``score_state`` / ``state_slot_mapping``. When ``window_len > 0`` in
    the plan, the corresponding K iters are sourced from the state cache
    ring buffer instead of kv_in / score_in.

    When ``k_split_num_waves`` / ``slice_size`` are ``None`` (the default),
    the launcher auto-picks via :func:`hca_per_n_config` keyed on
    ``plan_gpu.shape[0]`` (CUDAGraph-stable dispatch — see that function's
    docstring). Override only when bench-sweeping; the default matches the
    production tuning used by ATOM's compressor.
    """
    if k_split_num_waves is None or slice_size is None:
        # Local import to avoid a circular import between the two HCA modules
        # at package init time.
        from .fused_compress_attn import hca_per_n_config

        auto_slice, auto_kw = hca_per_n_config(plan_gpu.shape[0])
        if slice_size is None:
            slice_size = auto_slice
        if k_split_num_waves is None:
            k_split_num_waves = auto_kw
    # User-facing input validation — must be ``raise`` not ``assert`` (asserts
    # are stripped under ``python -O``, which would let invalid inputs reach
    # the kernel and silently corrupt outputs / fault the GPU).
    if head_dim != 512:
        raise ValueError(f"HCA 2-kernel only supports head_dim=512, got {head_dim}")
    if ratio != 128:
        raise ValueError(f"HCA 2-kernel only supports ratio=128, got {ratio}")
    if kv_in.dim() != 2 or kv_in.shape[1] != head_dim:
        raise ValueError(f"kv_in shape {tuple(kv_in.shape)} != [*, {head_dim}]")
    if score_in.shape != kv_in.shape:
        raise ValueError(f"score_in shape {tuple(score_in.shape)} != kv_in")
    if kv_in.dtype != torch.bfloat16 or score_in.dtype != torch.bfloat16:
        raise TypeError(
            f"kv_in/score_in must be bf16; got {kv_in.dtype}/{score_in.dtype}"
        )
    if kv_in.stride(-1) != 1 or score_in.stride(-1) != 1:
        raise ValueError("kv_in/score_in inner stride must be 1")
    if kv_in.stride(0) % 2 != 0 or score_in.stride(0) % 2 != 0:
        raise ValueError(
            "kv_in/score_in row strides (bf16 elem) must be even for dword bitcast"
        )

    plan_capacity = plan_gpu.shape[0]
    if plan_capacity == 0:
        return

    if ape.shape != (ratio, head_dim) or ape.dtype != torch.float32:
        raise ValueError(
            f"ape shape {tuple(ape.shape)} dtype {ape.dtype} != ({ratio}, {head_dim}) f32"
        )
    if not ape.is_contiguous():
        raise ValueError("ape must be contiguous")

    # State cache validation.
    if kv_state.dim() != 3 or kv_state.shape[2] != head_dim:
        raise ValueError(
            f"kv_state shape {tuple(kv_state.shape)} != [*, *, {head_dim}]"
        )
    state_size = kv_state.shape[1]
    if state_size < ratio:
        raise ValueError(f"state_size={state_size} must be >= K={ratio}")
    if score_state.shape != kv_state.shape:
        raise ValueError("score_state shape != kv_state")
    if kv_state.dtype != torch.float32 or score_state.dtype != torch.float32:
        raise TypeError("kv_state/score_state must be fp32")
    if not (kv_state.is_contiguous() and score_state.is_contiguous()):
        raise ValueError("kv_state/score_state must be contiguous")
    if state_slot_mapping.dim() != 1 or state_slot_mapping.dtype != torch.int32:
        raise ValueError("state_slot_mapping must be 1D int32")

    if kv_cache.dtype != torch.bfloat16:
        raise TypeError(f"HCA 2-kernel kv_cache must be bf16; got {kv_cache.dtype}")
    if block_tables.dtype != torch.int32:
        raise TypeError(f"block_tables must be int32; got {block_tables.dtype}")
    if not block_tables.is_contiguous():
        raise ValueError("block_tables must be contiguous")

    # Allocate kv_compressed scratch on demand.
    if kv_compressed_scratch is None:
        kv_compressed = torch.empty(
            (plan_capacity, head_dim),
            dtype=torch.float32,
            device=kv_in.device,
        )
    else:
        if kv_compressed_scratch.shape != (plan_capacity, head_dim):
            raise ValueError(
                f"kv_compressed_scratch shape {tuple(kv_compressed_scratch.shape)}"
                f" != ({plan_capacity}, {head_dim})"
            )
        if kv_compressed_scratch.dtype != torch.float32:
            raise TypeError("kv_compressed_scratch must be fp32")
        kv_compressed = kv_compressed_scratch

    # CRITICAL: must pass current_stream when stream is None. Stream(None) =
    # NULL/default stream, which during CUDA graph capture produces an empty
    # graph entry (kernel launches don't get recorded into the active graph),
    # so replay is a no-op → HCA boundaries silently never fire in decode CG.
    # Match v1 single-kernel pattern (fused_compress_attn.py:1381).
    if stream is None:
        stream = torch.cuda.current_stream()
    stream_obj = Stream(stream)

    compress_fn = compile_hca_compress_forward(
        head_dim=head_dim,
        ratio=ratio,
        state_size=int(state_size),
        k_split_num_waves=k_split_num_waves,
        slice_size=slice_size,
    )
    compress_args = (
        kv_in,
        int(kv_in.stride(0)),
        score_in,
        int(score_in.stride(0)),
        plan_gpu,
        kv_state,
        int(kv_state.stride(0)),
        int(kv_state.stride(1)),
        score_state,
        int(score_state.stride(0)),
        int(score_state.stride(1)),
        state_slot_mapping,
        ape,
        kv_compressed,
        int(kv_compressed.stride(0)),
        int(plan_capacity),
        stream_obj,
    )
    _run_compiled(compress_fn, *compress_args)

    rms_weight_is_bf16 = rms_weight.dtype == torch.bfloat16
    norm_fn = compile_hca_norm_rope_scatter(
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        ratio=ratio,
        k_per_block=k_per_block,
        rms_weight_is_bf16=rms_weight_is_bf16,
        rms_eps=rms_eps,
    )
    norm_args = (
        kv_compressed,
        int(kv_compressed.stride(0)),
        plan_gpu,
        rms_weight,
        cos_cache,
        sin_cache,
        kv_cache,
        int(kv_cache.stride(0)),
        int(kv_cache.stride(1)),
        block_tables,
        int(block_tables.stride(0)),
        int(plan_capacity),
        stream_obj,
    )
    _run_compiled(norm_fn, *norm_args)
