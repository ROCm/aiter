# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""HCA-path FlyDSL compress + norm+rope+scatter kernels (2-kernel split).

Targeted optimization for V4-Pro HCA Main: D=512, RD=64, ratio=128, overlap=False.
Inspired by SGLang's `c128_v2.cuh`, adapted to AMD wave64 / flydsl with a
multi-wave LDS K-split (Phase 3 of the optimization series):

  Kernel A -- flydsl_hca_compress_forward (multi-wave K-split)
    Grid:  (num_compress, NUM_SPLIT=head_dim/SLICE)
    Block: BLOCK_THREADS = 64 * k_split_num_waves (default 8 -> 512 threads)
    Each block covers SLICE=64 head_dim elements of one boundary.
    K=128 split across NW waves (K_PER_WAVE = K/NW = 16). Per-wave local
    online-softmax + cross-wave LDS reduction. Each wave's K range splits
    at clamp(window_len, k_start, k_end) into a Phase 1 (state cache,
    padded softmax) sub-loop followed by a Phase 2 (ragged input) sub-loop.
    Output: kv_compressed[num_compress, head_dim] fp32 (compact, indexed by pid).

  Kernel B -- flydsl_hca_norm_rope_scatter
    Grid:  (num_compress,)
    Block: BLOCK_THREADS=64 (1 wave)
    Each block reads one row of kv_compressed (full head_dim), does RMSNorm +
    GPT-J RoPE on the RD tail, scatters to paged kv_cache (BF16 only -- HCA
    Main is the only HCA path that currently routes here; FP8 quant lives in
    the legacy single-kernel for now).

Why split into two kernels:
  Single-kernel HCA has 1 wave per boundary x K=128 serial chain = poor CU
  utilization at small N. Splitting head_dim into NUM_SPLIT=8 grid-Y blocks
  and parallelising K across NW=8 waves gives 1024 blocks at N=16, drastically
  cutting register pressure and shortening per-iter dependency chains.

Cost: extra HBM r/w of kv_compressed = num_compress * head_dim * 4 bytes.
For N=16384 D=512: 32 MB -> ~4 us at 8 TB/s. Amortised by the compress kernel
speedup; after the ``slice_size`` + VEC=8 refactor the 2-kernel path beats
the legacy single-kernel at ALL N (1.06-3.7x, small N gets the largest win).

NOTE: HCA-only and BF16-only by design. CSA / Indexer / FP8 paths continue
to use the legacy single-kernel ``flydsl_fused_compress_attn``.
"""

import math
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.arith import CmpFPredicate
from flydsl.expr.typing import Int32, Stream, T

from aiter.ops.flydsl.kernels import buffer_ops

from .fused_compress_attn_common import emit_group_fp8_nm_asm_scatter
from .tensor_shim import _run_compiled, _to_raw


def _velem(vec, i):
    """Element *i* of *vec* as a raw ir.Value (this kernel feeds raw MLIR ops)."""
    return _to_raw(fx.Vector(vec)[i])


def _maxsi(a, b):
    """Signed integer max -> fx.Int32 (no fx signed-integer-max operator)."""
    # BOUNDARY: arith.maxsi (no fx signed-integer-max operator).
    return fx.Int32(arith.maxsi(_to_raw(fx.Int32(a)), _to_raw(fx.Int32(b))))


def _minsi(a, b):
    """Signed integer min -> fx.Int32 (no fx signed-integer-min operator)."""
    # BOUNDARY: arith.minsi (no fx signed-integer-min operator).
    return fx.Int32(arith.minsi(_to_raw(fx.Int32(a)), _to_raw(fx.Int32(b))))


def _oeq_raw(a, b):
    """Ordered fp equality (OEQ) as a fx.Boolean via RAW arith.cmpf so ambient
    fast_fp_math (ninf) does not fold the ``x == -inf`` softmax sentinel check."""
    from flydsl.expr.numeric import Boolean

    # BOUNDARY: non-fastmath OEQ against the -inf softmax sentinel.
    return Boolean(arith.cmpf(CmpFPredicate.OEQ, _to_raw(a), _to_raw(b)))


def _max_raw(a, b):
    """NaN/inf-safe fp maximum -> fx.Float32 via RAW arith.maximumf (fastmath
    None). The online-softmax running max operates on the -inf init sentinel;
    fast_fp_math ninf on fx `.maximumf` would drop that guard."""
    # BOUNDARY: non-fastmath maximumf over the -inf softmax accumulator.
    return fx.Float32(arith.maximumf(_to_raw(a), _to_raw(b)))


BLOCK_THREADS = 64  # 1 wave64
SLICE = 64  # head_dim elements per block (grid-Y split)
_NEG_INF = float("-inf")
_LOG2E = math.log2(math.e)


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
      - Per-wave local online-softmax -> (m_local, kv_local, w_local) lists
        of VEC values per thread.
      - LDS cross-wave reduction: only wave 0 active; each thread reads
        NW*VEC values from LDS, computes VEC reduced compressed values,
        writes them out via vector buffer_store.

    Tuning knobs:
      - ``k_split_num_waves`` (= NW): trades K-serial chain length for LDS
        reduce cost. Small N -> larger NW (more waves -> more CU coverage);
        large N -> smaller NW (less LDS overhead).
      - ``slice_size``: VEC width per thread. slice_size=64 -> VEC=1 scalar
        (more blocks per boundary -> small-N champion); slice_size=512 ->
        VEC=8 (1 block per boundary, v1-like -> large-N coalesced HBM).

    Phase 1 (state cache) is integrated by splitting each wave's K range at
    ``clamp(window_len, k_start, k_end)`` into a Phase 1 sub-loop reading
    kv_state + score_state (padded softmax when ``s < 0``) and a Phase 2
    sub-loop reading kv_in + score_in. Phase 2 in_row is clamped to >= 0
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

    @fx.struct
    class SharedStorage:
        lds_m: fx.Array[fx.Float32, LDS_M_ELEMS, 16]
        lds_kv: fx.Array[fx.Float32, LDS_KV_ELEMS, 16]
        lds_w: fx.Array[fx.Float32, LDS_W_ELEMS, 16]

    _kname = (
        f"hca_compress_forward_D{D}_R{ratio}_NW{NW}_SL{SLICE_SZ}_S{state_size}_flydsl"
    )

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

        c_neg_inf = arith.constant(_NEG_INF, type=f32)
        c_zero_f32 = arith.constant(0.0, type=f32)
        c_log2e = arith.constant(_LOG2E, type=f32)
        c_K_m1 = arith.constant(K - 1, type=i32)
        c_state_size = arith.constant(state_size, type=i32)

        def fexp_f32(x):
            return fx.rocdl.exp2(f32, x * c_log2e)

        # Per-thread wave / lane (block-local).
        wid = _to_raw(fx.Int32(tid) // fx.Int32(64))  # ? [0, NW)
        lid = _to_raw(fx.Int32(tid) % fx.Int32(64))  # ? [0, 64)

        # -- Load plan row ----------------------------------------------
        plan_rsrc = buffer_ops.create_buffer_resource(plan, max_size=True)
        plan_base = _to_raw(fx.Int32(pid) * fx.Int32(4))
        plan_vec = buffer_ops.buffer_load(plan_rsrc, plan_base, vec_width=4, dtype=i32)
        ragged_id = _velem(plan_vec, 0)
        batch_id = _velem(plan_vec, 1)
        position = _velem(plan_vec, 2)
        window_len = _velem(plan_vec, 3)

        # Sentinel-skip: run the whole body only for position >= 0, as a closure
        # under a runtime `if` (rewriter sees an opaque call -> scf.if).
        def _body():
            # Per-thread head_dim base: each thread owns VEC contiguous
            # elements starting at slice_base + lid * VEC.
            col_off_base = _to_raw(
                fx.Int32(sid) * fx.Int32(SLICE_SZ) + fx.Int32(lid) * fx.Int32(VEC)
            )

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
                ``base_off_elems_i32`` -> list of VEC f32 values.

                VEC=1: unaligned-safe scalar via dword + bit-extract.
                VEC>=2: vectorized i32 buffer_load + bitcast to bf16.
                """
                if const_expr(VEC == 1):
                    off_dw = _to_raw(fx.Int32(base_off_elems_i32) >> fx.Int32(1))
                    lane_in_dw = fx.Int32(base_off_elems_i32) & fx.Int32(1)
                    raw_s = buffer_ops.buffer_load(rsrc, off_dw, vec_width=1, dtype=i32)
                    hi = fx.Int32(raw_s) >> fx.Int32(16)
                    lo_or_hi = (lane_in_dw == fx.Int32(0)).select(raw_s, hi)
                    lo16 = fx.Int32(lo_or_hi) & fx.Int32(0xFFFF)
                    lo16_v = fx.Vector.from_elements([_to_raw(lo16)], fx.Int32)
                    bf16_pair = fx.Vector(lo16_v).bitcast(fx.BFloat16)
                    bf16_v = _velem(bf16_pair, 0)
                    return [_to_raw(fx.BFloat16(bf16_v).to(fx.Float32))]
                else:
                    # base must be VEC-aligned (caller guarantees by
                    # col_off_base = sid*SLICE + lid*VEC, both multiples of VEC).
                    off_dw = _to_raw(fx.Int32(base_off_elems_i32) >> fx.Int32(1))
                    dwords = VEC // 2  # VEC bf16 = VEC*2 bytes
                    if const_expr(dwords == 1):
                        # buffer_load(vec_width=1) returns scalar i32; wrap
                        # into vec<1xi32> before bitcast to vec<2xbf16>.
                        raw_s = buffer_ops.buffer_load(
                            rsrc, off_dw, vec_width=1, dtype=i32
                        )
                        raw = fx.Vector.from_elements([raw_s], fx.Int32)
                    else:
                        raw = buffer_ops.buffer_load(
                            rsrc, off_dw, vec_width=dwords, dtype=i32
                        )
                    vec_bf16 = fx.Vector(raw).bitcast(fx.BFloat16)
                    out = []
                    for i in range_constexpr(VEC):
                        bf16_v = _velem(vec_bf16, i)
                        out.append(_to_raw(fx.BFloat16(bf16_v).to(fx.Float32)))
                    return out

            def _load_f32_vec(rsrc, base_off_elems_i32):
                """Load VEC f32 starting at base -> list of VEC f32 values."""
                if const_expr(VEC <= 4):
                    raw = buffer_ops.buffer_load(
                        rsrc, base_off_elems_i32, vec_width=VEC, dtype=f32
                    )
                    if const_expr(VEC == 1):
                        # vec_width=1 returns scalar, not 1-vec.
                        return [raw]
                    return [_velem(raw, i) for i in range(VEC)]
                else:
                    # VEC == 8: AMD HW max is dwordx4 -> 2 loads.
                    assert VEC == 8
                    half = VEC // 2
                    r0 = buffer_ops.buffer_load(
                        rsrc, base_off_elems_i32, vec_width=half, dtype=f32
                    )
                    r1 = buffer_ops.buffer_load(
                        rsrc,
                        _to_raw(fx.Int32(base_off_elems_i32) + fx.Int32(half)),
                        vec_width=half,
                        dtype=f32,
                    )
                    out = []
                    for i in range_constexpr(half):
                        out.append(_velem(r0, i))
                    for i in range_constexpr(half):
                        out.append(_velem(r1, i))
                    return out

            def _issue_phase2_loads(k_i32):
                """Phase 2 (ragged input) loads. Returns (kv_list, sc_list,
                ape_list) each of length VEC."""
                ape_row = _to_raw(fx.Int32(k_i32) % fx.Int32(ratio))
                in_row = _maxsi(
                    fx.Int32(ragged_id) - (fx.Int32(c_K_m1) - fx.Int32(k_i32)),
                    fx.Int32(0),
                )
                base_in_off = _to_raw(
                    in_row * fx.Int32(kv_in_row_stride) + fx.Int32(col_off_base)
                )
                base_sc_off = _to_raw(
                    in_row * fx.Int32(score_in_row_stride) + fx.Int32(col_off_base)
                )
                base_ape_off = _to_raw(
                    fx.Int32(ape_row) * fx.Int32(DIM_FULL) + fx.Int32(col_off_base)
                )
                kv = _load_bf16_vec_to_f32(kv_in_rsrc, base_in_off)
                sc = _load_bf16_vec_to_f32(score_in_rsrc, base_sc_off)
                ape_v = _load_f32_vec(ape_rsrc, base_ape_off)
                return kv, sc, ape_v

            def _issue_phase1_loads(k_i32):
                """Phase 1 (state cache) loads. Returns (kv_list, sc_padded_list)
                each of length VEC. Score is -inf when s < 0."""
                s = fx.Int32(position) - fx.Int32(c_K_m1) + fx.Int32(k_i32)
                is_pad = s < fx.Int32(0)
                s_safe = is_pad.select(fx.Int32(0), s)
                ring = _to_raw(s_safe % fx.Int32(c_state_size))
                base_kv_off = _to_raw(
                    fx.Int32(slot) * fx.Int32(kv_state_slot_stride)
                    + fx.Int32(ring) * fx.Int32(kv_state_pos_stride)
                    + fx.Int32(col_off_base)
                )
                base_sc_off = _to_raw(
                    fx.Int32(slot) * fx.Int32(score_state_slot_stride)
                    + fx.Int32(ring) * fx.Int32(score_state_pos_stride)
                    + fx.Int32(col_off_base)
                )
                kv_list = _load_f32_vec(kv_state_rsrc, base_kv_off)
                sc_list = _load_f32_vec(score_state_rsrc, base_sc_off)
                sc_padded = [
                    _to_raw(is_pad.select(c_neg_inf, sc_list[i])) for i in range(VEC)
                ]
                return kv_list, sc_padded

            def _softmax_step_padded(
                m_old_list, kv_old_list, w_old_list, score_k_list, kv_k_list
            ):
                """Padding-aware vector softmax step over VEC lanes. When
                score_k == -inf, w_k is forced to 0 (avoids NaN when m_old
                is also -inf). Safe in both Phase 1 (padding can occur) and
                Phase 2 (score finite -> pad-select branch is dead code).
                """
                new_m, new_kv, new_w = [], [], []
                for i in range_constexpr(VEC):
                    m_old = m_old_list[i]
                    kv_old = kv_old_list[i]
                    w_old = w_old_list[i]
                    score_k = score_k_list[i]
                    kv_k = kv_k_list[i]
                    m_new = _max_raw(m_old, score_k)
                    is_first = _oeq_raw(m_old, c_neg_inf)
                    # BOUNDARY: non-fastmath subf by design.
                    scale_active = fexp_f32(arith.subf(m_old, _to_raw(m_new)))
                    scale_v = fx.Float32(is_first.select(c_zero_f32, scale_active))
                    # BOUNDARY: non-fastmath subf by design.
                    wk_active = fexp_f32(arith.subf(score_k, _to_raw(m_new)))
                    is_pad_score = _oeq_raw(score_k, c_neg_inf)
                    w_k = fx.Float32(is_pad_score.select(c_zero_f32, wk_active))
                    new_kv.append(
                        _to_raw(
                            fx.Float32(kv_old) * scale_v + w_k * fx.Float32(kv_k)
                        )
                    )
                    new_w.append(_to_raw(fx.Float32(w_old) * scale_v + w_k))
                    new_m.append(_to_raw(m_new))
                return new_m, new_kv, new_w

            # -- Wave's K range: [wid * K_PER_WAVE, (wid+1) * K_PER_WAVE) --
            k_start_i32 = _to_raw(fx.Int32(wid) * fx.Int32(K_PER_WAVE))
            k_end_i32 = _to_raw(fx.Int32(k_start_i32) + fx.Int32(K_PER_WAVE))

            # Split point inside this wave's K range. Each wave sees a
            # window_len-dependent slice of Phase 1 followed by Phase 2.
            # Cases (`wl = window_len`):
            #   wl <= k_start:  pure Phase 2 (entire wave is input)
            #   wl >= k_end:    pure Phase 1 (entire wave is state cache)
            #   else:          mixed (Phase 1 in [k_start, wl), Phase 2 in [wl, k_end))
            # ``split`` = clamp(wl, k_start, k_end) gives the boundary;
            # both sub-loops are empty when their bound collapses, so any
            # of the three cases naturally falls out.
            wl_i32 = _to_raw(window_len)
            split_lo = _maxsi(fx.Int32(wl_i32), fx.Int32(k_start_i32))
            split_i32 = _to_raw(_minsi(split_lo, fx.Int32(k_end_i32)))

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
                k_i32 = _to_raw(fx.Int32(k_static))
                kv_v, sc_v = _issue_phase1_loads(k_i32)
                new_m, new_kv, new_w = _softmax_step_padded(
                    m_lane, kv_lane, w_lane, sc_v, kv_v
                )
                phase1_local = yield list(new_m) + list(new_kv) + list(new_w)

            # Sub-loop 2: Phase 2 sub-range [split, k_end). Reads input;
            # uses padded softmax (the is-pad-score branch is dead code
            # since Phase 2 scores are always finite -- compiler elides).
            # Carry Phase 1's accumulator through as init.
            final = phase1_local
            for k_static, state in range(
                _to_raw(split_i32), _to_raw(k_end_i32), 1, init=phase1_local
            ):
                m_lane = list(state[0:VEC])
                kv_lane = list(state[VEC : 2 * VEC])
                w_lane = list(state[2 * VEC : 3 * VEC])
                k_i32 = _to_raw(fx.Int32(k_static))
                p2_kv, p2_sc, p2_ape = _issue_phase2_loads(k_i32)
                p2_score = [
                    _to_raw(fx.Float32(p2_sc[i]) + fx.Float32(p2_ape[i]))
                    for i in range(VEC)
                ]
                new_m, new_kv, new_w = _softmax_step_padded(
                    m_lane, kv_lane, w_lane, p2_score, p2_kv
                )
                final = yield list(new_m) + list(new_kv) + list(new_w)

            m_local = list(final[0:VEC])
            kv_local = list(final[VEC : 2 * VEC])
            w_local = list(final[2 * VEC : 3 * VEC])

            # -- LDS write: each thread writes VEC entries per array --
            # Layout: per array, NW * SLICE_SZ fp32 entries; per-thread
            # base = wid * SLICE_SZ + lid * VEC; thread writes VEC values
            # at base+0, base+1, ..., base+VEC-1.
            lds = fx.SharedAllocator().allocate(SharedStorage).peek()
            lds_m_ptr = lds.lds_m.ptr
            lds_kv_ptr = lds.lds_kv.ptr
            lds_w_ptr = lds.lds_w.ptr
            lds_thread_base = fx.Int32(wid) * fx.Int32(SLICE_SZ) + fx.Int32(
                lid
            ) * fx.Int32(VEC)
            for i in range_constexpr(VEC):
                idx_i = lds_thread_base + fx.Int32(i)
                fx.ptr_store(m_local[i], lds_m_ptr + idx_i)
                fx.ptr_store(kv_local[i], lds_kv_ptr + idx_i)
                fx.ptr_store(w_local[i], lds_w_ptr + idx_i)

            gpu.barrier()

            # -- Cross-wave reduction: only wave 0 reads and reduces --
            # Wave 0's 64 threads cover SLICE_SZ = 64 * VEC head_dim elements
            # (VEC elements per thread). For each owned element, the thread
            # reads NW values from LDS (one per K-split wave) and computes
            # the global online-softmax.
            def _wave0():
                comp_list = []
                for i in range_constexpr(VEC):
                    lane_off = fx.Int32(lid) * fx.Int32(VEC) + fx.Int32(i)
                    # Global max across NW waves for this element.
                    m_g = fx.Float32(c_neg_inf)
                    m_arr = []
                    for w in range_constexpr(NW):
                        idx_w = fx.Int32(w * SLICE_SZ) + lane_off
                        m_w = fx.ptr_load(lds_m_ptr + idx_w)
                        m_arr.append(m_w)
                        m_g = m_g.maximumf(m_w)

                    # Weighted sums (kv * scale_w) and (w * scale_w).
                    kv_sum = fx.Float32(0.0)
                    w_sum = fx.Float32(0.0)
                    for w in range_constexpr(NW):
                        idx_w = fx.Int32(w * SLICE_SZ) + lane_off
                        kv_w = fx.ptr_load(lds_kv_ptr + idx_w)
                        w_w = fx.ptr_load(lds_w_ptr + idx_w)
                        m_w = m_arr[w]
                        scale_w = fx.Float32(fexp_f32(_to_raw(m_w - m_g)))
                        kv_sum = kv_sum + kv_w * scale_w
                        w_sum = w_sum + w_w * scale_w
                    rcp_w = fx.Float32(fx.rocdl.rcp(f32, _to_raw(w_sum)))
                    comp_list.append(_to_raw(kv_sum * rcp_w))

                # -- Vectorized write of VEC f32 comp values --
                out_rsrc = buffer_ops.create_buffer_resource(
                    kv_compressed, max_size=True
                )
                out_off = _to_raw(
                    fx.Int32(pid) * fx.Int32(kv_compressed_row_stride)
                    + fx.Int32(col_off_base)
                )
                if const_expr(VEC == 1):
                    buffer_ops.buffer_store(comp_list[0], out_rsrc, out_off)
                elif const_expr(VEC <= 4):
                    out_vec = fx.Vector.from_elements(comp_list, fx.Float32)
                    buffer_ops.buffer_store(out_vec, out_rsrc, out_off)
                else:
                    # VEC == 8: AMD HW max is dwordx4 -> 2 stores.
                    assert VEC == 8
                    half = VEC // 2
                    v0 = fx.Vector.from_elements(comp_list[0:half], fx.Float32)
                    v1 = fx.Vector.from_elements(comp_list[half:VEC], fx.Float32)
                    buffer_ops.buffer_store(v0, out_rsrc, out_off)
                    buffer_ops.buffer_store(
                        v1,
                        out_rsrc,
                        _to_raw(fx.Int32(out_off) + fx.Int32(half)),
                    )

            if wid == 0:
                _wave0()

        if fx.Int32(position) >= 0:
            _body()

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
        stream: fx.Stream,
    ):
        idx_p = fx.Int64(plan_capacity)
        idx_s = fx.Int64(NUM_SPLIT)
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
    quant: bool = False,
    quant_group_size: int = 64,
    k_waves: int = 1,
):
    """Build per-row RMSNorm + GPT-J RoPE + paged scatter for HCA.

    Reads kv_compressed[num_compress, head_dim] fp32 and the plan; for each
    boundary, normalizes / rotates / scatters into kv_cache.

    quant=False: BF16 single-buffer scatter (nope + rope in one kv_cache row).
    quant=True : FP8 nope (1xG e8m0 group-quant) + inline duplicated e8m0 scale into
                 kv_cache (V4 nm asm layout), rotated PE bf16 into a SEPARATE k_rope_buff
                 -- byte-identical to the C++ k_wave / fused_kv_compress_scatter output.
    """
    D = head_dim
    RD = rope_head_dim
    NOPE = D - RD
    # WAVE lanes process one plan row (the reduce + group-amax shuffle_xor stay
    # within these 64 lanes). k_waves rows are packed into one block (BT threads)
    # purely to amortize block-launch/scheduling overhead -- waves are otherwise
    # independent (no cross-wave LDS). k_waves=1 reproduces the 1-wave/block path.
    WAVE = 64
    KW = k_waves
    BT = WAVE * KW  # threads per block
    VEC = D // WAVE  # 8 for D=512
    ROPE_THREAD_LO = NOPE // VEC
    PAIRS_PER_THREAD = VEC // 2

    assert D % WAVE == 0
    assert RD > 0 and RD % 2 == 0 and RD % VEC == 0

    # FP8 1xG e8m0 group-quant geometry (nope region only). GROUP_SIZE must divide
    # NOPE and be a multiple of VEC (a lane's VEC slice never crosses a group).
    GROUP_SIZE_Q = quant_group_size
    assert (not quant) or (
        NOPE % GROUP_SIZE_Q == 0 and GROUP_SIZE_Q % VEC == 0
    ), f"quant: NOPE={NOPE} must be divisible by group={GROUP_SIZE_Q}, group%VEC==0"
    RTS = GROUP_SIZE_Q // VEC if quant else 1  # threads per group (=8 for G=64,VEC=8)
    log2_rts = int(math.log2(RTS)) if quant else 0

    _kname = (
        f"hca_norm_rope_scatter_D{D}_RD{RD}_R{ratio}_KB{k_per_block}_KW{KW}"
        f"{'_rmsbf16' if rms_weight_is_bf16 else ''}{'_fp8' if quant else ''}_flydsl"
    )
    fm_fast = arith.FastMathFlags.fast
    log2_wave = int(math.log2(WAVE))

    @flyc.kernel(name=_kname)
    def kernel(
        kv_compressed: fx.Tensor,  # [num_compress, head_dim] f32
        kv_compressed_row_stride: Int32,
        plan: fx.Tensor,  # [num_compress, 4] i32
        rms_weight: fx.Tensor,  # [head_dim] bf16 or f32
        cos_cache: fx.Tensor,  # [max_pos, RD/2] bf16
        sin_cache: fx.Tensor,
        kv_cache: fx.Tensor,  # bf16: [NB,k_per_block,D]; fp8: [NB,k_per_block,entry] nope+scale
        kv_cache_block_stride: Int32,  # elements (bf16 or fp8/byte)
        kv_cache_token_stride: Int32,
        block_table: fx.Tensor,  # [bs, max_blocks_per_seq] i32
        block_table_seq_stride: Int32,
        k_rope_buff: fx.Tensor,  # fp8 only: paged [NB,k_per_block,RD] bf16 rope (dummy if !quant)
        krope_block_stride: Int32,
        krope_token_stride: Int32,
        plan_capacity: Int32,  # num plan rows (cap); tail waves with row>=cap bail
    ):
        f32 = T.f32
        i32 = T.i32
        vecVf32 = T.vec(VEC, T.f32)

        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        # Pack KW plan rows per block: wave_id picks the row, lane indexes head_dim.
        wave_id = fx.Int32(tid) >> fx.Int32(log2_wave)
        lane = _to_raw(fx.Int32(tid) & fx.Int32(WAVE - 1))
        pid = fx.Int32(bid) * fx.Int32(KW) + wave_id

        c_eps = arith.constant(rms_eps, type=f32)
        c_inv_D = arith.constant(1.0 / D, type=f32)

        def wave_reduce_add(x):
            w = fx.Float32(x)
            for sh_exp in range_constexpr(log2_wave):
                off = WAVE // (2 << sh_exp)
                w = w + fx.Float32(w.shuffle_xor(off, WAVE))
            return _to_raw(w)

        # -- Load plan row --
        plan_rsrc = buffer_ops.create_buffer_resource(plan, max_size=True)
        plan_base = _to_raw(pid * fx.Int32(4))
        plan_vec = buffer_ops.buffer_load(plan_rsrc, plan_base, vec_width=4, dtype=i32)
        batch_id = _velem(plan_vec, 1)
        position = _velem(plan_vec, 2)

        # active = real plan row (position>=0 sentinel) AND within capacity (tail
        # waves of the last block have pid>=cap and must bail; their plan load is
        # bounds-checked to 0 by the buffer resource, so guard explicitly here).
        is_active = (fx.Int32(position) >= 0) & (pid < plan_capacity)

        # Whole body as a closure under the runtime guard (opaque call -> scf.if).
        def _body():
            tid_x_vec = _to_raw(fx.Int32(lane) * fx.Int32(VEC))

            # -- Load kv_compressed[pid, tid*VEC : tid*VEC + VEC] --
            kvc_rsrc = buffer_ops.create_buffer_resource(kv_compressed, max_size=True)
            base_off = _to_raw(
                pid * fx.Int32(kv_compressed_row_stride) + fx.Int32(tid_x_vec)
            )
            # VEC ? {2, 4, 8}: VEC <= 4 -> single dwordx{VEC}; VEC=8 -> 2x dwordx4.
            if const_expr(VEC <= 4):
                raw = buffer_ops.buffer_load(
                    kvc_rsrc, base_off, vec_width=VEC, dtype=f32
                )
                comp_lane = [_velem(raw, i) for i in range(VEC)]
            else:
                assert VEC == 8
                half = 4
                r0 = buffer_ops.buffer_load(
                    kvc_rsrc, base_off, vec_width=half, dtype=f32
                )
                r1 = buffer_ops.buffer_load(
                    kvc_rsrc,
                    _to_raw(fx.Int32(base_off) + fx.Int32(half)),
                    vec_width=half,
                    dtype=f32,
                )
                comp_lane = [_velem(r0, i) for i in range(half)] + [
                    _velem(r1, i) for i in range(half)
                ]

            # -- RMSNorm (wave reduce-add of squares / D + eps; rsqrt) --
            sq_local = fx.Float32(0.0)
            for i in range_constexpr(VEC):
                cl = fx.Float32(comp_lane[i])
                sq_local = sq_local + cl * cl
            sq_full = wave_reduce_add(_to_raw(sq_local))
            var = fx.Float32(sq_full) * fx.Float32(c_inv_D)
            rrms = fmath.rsqrt(_to_raw(var + fx.Float32(c_eps)), fastmath=fm_fast)

            # rms_weight load
            rmsw_rsrc = buffer_ops.create_buffer_resource(rms_weight, max_size=True)
            if const_expr(rms_weight_is_bf16):
                dwords = (VEC + 1) // 2
                off_dw = _to_raw(fx.Int32(tid_x_vec) >> fx.Int32(1))
                if const_expr(dwords == 1):
                    raw_s = buffer_ops.buffer_load(
                        rmsw_rsrc, off_dw, vec_width=1, dtype=i32
                    )
                    raw = fx.Vector.from_elements([raw_s], fx.Int32)
                else:
                    raw = buffer_ops.buffer_load(
                        rmsw_rsrc, off_dw, vec_width=dwords, dtype=i32
                    )
                vec_bf16 = fx.Vector(raw).bitcast(fx.BFloat16)
                rmsw_lane = []
                for i in range_constexpr(VEC):
                    bf16_v = _velem(vec_bf16, i)
                    rmsw_lane.append(_to_raw(fx.BFloat16(bf16_v).to(fx.Float32)))
            else:
                if const_expr(VEC <= 4):
                    raw = buffer_ops.buffer_load(
                        rmsw_rsrc, tid_x_vec, vec_width=VEC, dtype=f32
                    )
                    rmsw_lane = [_velem(raw, i) for i in range(VEC)]
                else:
                    half = 4
                    r0 = buffer_ops.buffer_load(
                        rmsw_rsrc, tid_x_vec, vec_width=half, dtype=f32
                    )
                    r1 = buffer_ops.buffer_load(
                        rmsw_rsrc,
                        _to_raw(fx.Int32(tid_x_vec) + fx.Int32(half)),
                        vec_width=half,
                        dtype=f32,
                    )
                    rmsw_lane = [_velem(r0, i) for i in range(half)] + [
                        _velem(r1, i) for i in range(half)
                    ]

            rrms_f = fx.Float32(rrms)
            normed_lane = [
                _to_raw(fx.Float32(comp_lane[i]) * rrms_f * fx.Float32(rmsw_lane[i]))
                for i in range(VEC)
            ]

            # -- GPT-J RoPE on RD tail --
            comp_pos_i32 = _to_raw(
                (fx.Int32(position) // fx.Int32(ratio)) * fx.Int32(ratio)
            )
            cos_rsrc = buffer_ops.create_buffer_resource(cos_cache, max_size=True)
            sin_rsrc = buffer_ops.create_buffer_resource(sin_cache, max_size=True)
            cos_row_base = fx.Int32(comp_pos_i32) * fx.Int32(RD // 2)

            is_rope_t = fx.Int32(lane) >= fx.Int32(ROPE_THREAD_LO)
            rope_rel = _maxsi(
                fx.Int32(lane) - fx.Int32(ROPE_THREAD_LO), fx.Int32(0)
            )
            cs_off = _to_raw(cos_row_base + rope_rel * fx.Int32(PAIRS_PER_THREAD))

            if const_expr(PAIRS_PER_THREAD == 1):
                cos_b = buffer_ops.buffer_load(
                    cos_rsrc, cs_off, vec_width=1, dtype=T.bf16
                )
                sin_b = buffer_ops.buffer_load(
                    sin_rsrc, cs_off, vec_width=1, dtype=T.bf16
                )
                cos_vals = [_to_raw(fx.BFloat16(cos_b).to(fx.Float32))]
                sin_vals = [_to_raw(fx.BFloat16(sin_b).to(fx.Float32))]
            else:
                cos_vec = buffer_ops.buffer_load(
                    cos_rsrc,
                    cs_off,
                    vec_width=PAIRS_PER_THREAD,
                    dtype=T.bf16,
                )
                sin_vec = buffer_ops.buffer_load(
                    sin_rsrc,
                    cs_off,
                    vec_width=PAIRS_PER_THREAD,
                    dtype=T.bf16,
                )
                cos_vals = [
                    _to_raw(fx.BFloat16(_velem(cos_vec, i)).to(fx.Float32))
                    for i in range(PAIRS_PER_THREAD)
                ]
                sin_vals = [
                    _to_raw(fx.BFloat16(_velem(sin_vec, i)).to(fx.Float32))
                    for i in range(PAIRS_PER_THREAD)
                ]

            rotated_lane = list(normed_lane)
            for k in range_constexpr(PAIRS_PER_THREAD):
                e = normed_lane[2 * k]
                o = normed_lane[2 * k + 1]
                c = fx.Float32(cos_vals[k])
                s = fx.Float32(sin_vals[k])
                ef = fx.Float32(e)
                of = fx.Float32(o)
                # BOUNDARY: non-fastmath subf by design (no FMA contraction of
                # e*c - o*s; matches HEAD / fused_compress_attn_common).
                new_e = arith.subf(_to_raw(ef * c), _to_raw(of * s))
                new_o = _to_raw(ef * s + of * c)
                rotated_lane[2 * k] = new_e
                rotated_lane[2 * k + 1] = new_o

            # -- Paged scatter dest (shared by bf16 / fp8) --
            ci = fx.Int32(position) // fx.Int32(ratio)
            block_in_seq = _to_raw(ci // fx.Int32(k_per_block))
            slot_in_block = _to_raw(ci % fx.Int32(k_per_block))
            bt_rsrc = buffer_ops.create_buffer_resource(block_table, max_size=True)
            bt_off = _to_raw(
                fx.Int32(batch_id) * fx.Int32(block_table_seq_stride)
                + fx.Int32(block_in_seq)
            )
            physical_block = buffer_ops.buffer_load(
                bt_rsrc, bt_off, vec_width=1, dtype=i32
            )
            cache_base = _to_raw(
                fx.Int32(physical_block) * fx.Int32(kv_cache_block_stride)
                + fx.Int32(slot_in_block) * fx.Int32(kv_cache_token_stride)
            )
            out_rsrc = buffer_ops.create_buffer_resource(kv_cache, max_size=True)

            if const_expr(quant):
                # -- group_fp8 (V4 nm-asm) via shared emitter (single source of truth
                # shared with the CSA single-kernel; fp8 entry layout stays identical). --
                _krope_base = _to_raw(
                    fx.Int32(physical_block) * fx.Int32(krope_block_stride)
                    + fx.Int32(slot_in_block) * fx.Int32(krope_token_stride)
                )
                emit_group_fp8_nm_asm_scatter(
                    normed_lane=normed_lane,
                    rotated_lane=rotated_lane,
                    lane=lane,
                    is_rope_t=is_rope_t,
                    cache_base=cache_base,
                    out_rsrc=out_rsrc,
                    krope_base=_krope_base,
                    krope_rsrc=buffer_ops.create_buffer_resource(
                        k_rope_buff, max_size=True
                    ),
                    VEC=VEC,
                    NOPE=NOPE,
                    RTS=RTS,
                    log2_rts=log2_rts,
                    ROPE_THREAD_LO=ROPE_THREAD_LO,
                    wave_width=WAVE,
                    vecVf32=vecVf32,
                    fm_fast=fm_fast,
                )
            else:
                # ---- BF16 single-buffer scatter (nope + rope contiguous) ----
                out_lane = [
                    _to_raw(is_rope_t.select(rotated_lane[i], normed_lane[i]))
                    for i in range_constexpr(VEC)
                ]
                cache_off = fx.Int32(cache_base) + fx.Int32(tid_x_vec)
                out_vec_t = T.vec(VEC, T.bf16)
                raw_vec = fx.Vector.from_elements(out_lane, fx.Float32)
                bf16_vec = raw_vec.truncf(out_vec_t)
                cache_off_dw = _to_raw(cache_off >> fx.Int32(1))
                dwords = (VEC + 1) // 2
                bf16_as_i32 = fx.Vector(bf16_vec).bitcast(fx.Int32)
                if const_expr(dwords == 1):
                    scalar_i32 = _velem(bf16_as_i32, 0)
                    buffer_ops.buffer_store(scalar_i32, out_rsrc, cache_off_dw)
                else:
                    buffer_ops.buffer_store(bf16_as_i32, out_rsrc, cache_off_dw)

        if is_active:
            _body()

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
        k_rope_buff: fx.Tensor,
        krope_block_stride: fx.Int32,
        krope_token_stride: fx.Int32,
        plan_capacity: fx.Int32,
        stream: fx.Stream,
    ):
        # grid = ceil(cap / KW): KW plan rows packed per block.
        nblocks = _to_raw((fx.Int32(plan_capacity) + fx.Int32(KW - 1)) // fx.Int32(KW))
        idx_p = fx.Int64(nblocks)
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
            k_rope_buff,
            krope_block_stride,
            krope_token_stride,
            plan_capacity,
        )
        k.launch(
            grid=(idx_p, 1, 1),
            block=(BT, 1, 1),
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
    K-iter (wider buffer_load -> better HBM coalescing), but fewer blocks
    per boundary (NUM_SPLIT = head_dim / slice_size). slice_size=64 -> VEC=1
    (8 blocks/boundary, small-N champion); slice_size=512 -> VEC=8
    (1 block/boundary, v1-like HBM access, large-N champion).

    ``state_size`` is the ring-buffer modulo of ``kv_state.shape[1]`` (>= ratio).
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


@lru_cache(maxsize=16)
def compile_hca_norm_rope_scatter(
    *,
    head_dim: int,
    rope_head_dim: int,
    ratio: int,
    k_per_block: int,
    rms_weight_is_bf16: bool,
    rms_eps: float,
    quant: bool = False,
    quant_group_size: int = 64,
    k_waves: int = 1,
):
    launcher = _build_norm_rope_scatter_kernel(
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        ratio=ratio,
        k_per_block=k_per_block,
        rms_weight_is_bf16=rms_weight_is_bf16,
        rms_eps=rms_eps,
        quant=quant,
        quant_group_size=quant_group_size,
        k_waves=k_waves,
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
    kv_compressed_scratch: torch.Tensor | None = None,
    quant: bool = False,
    k_rope_cache: torch.Tensor | None = None,
    quant_group_size: int = 64,
    k_split_num_waves: int | None = None,
    slice_size: int | None = None,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """HCA-only 2-kernel compress + norm+rope+scatter (V4-Pro Main path).

    Restrictions: ratio=128, overlap=False (implicit), head_dim=512 supported.

    Cache scatter dtype:
      * ``quant=False`` (default): BF16 single-buffer scatter -- nope + rope written
        contiguously into ``kv_cache`` [NB, k_per_block, head_dim] bf16.
      * ``quant=True``: FP8 1xG e8m0 group-quant. ``kv_cache`` is fp8
        [NB, k_per_block, entry] holding nope fp8 + inline duplicated e8m0 scale
        (V4 nm asm layout); rotated PE bf16 goes to ``k_rope_cache``
        [NB, k_per_block, rope_head_dim] bf16. Byte-identical to the C++
        ``fused_kv_compress_scatter`` k_wave output.

    Phase 1 (state cache) is enabled by passing real ``kv_state`` /
    ``score_state`` / ``state_slot_mapping``. When ``window_len > 0`` in
    the plan, the corresponding K iters are sourced from the state cache
    ring buffer instead of kv_in / score_in.

    When ``k_split_num_waves`` / ``slice_size`` are ``None`` (the default),
    the launcher auto-picks via :func:`hca_per_n_config` keyed on
    ``plan_gpu.shape[0]`` (CUDAGraph-stable dispatch -- see that function's
    docstring). Override only when bench-sweeping; the default matches the
    production tuning used by ATOM's compressor.
    """
    # ---- gfx1250 dispatch (wave32) ----
    from aiter.jit.utils.chip_info import get_gfx as _get_gfx

    if _get_gfx() == "gfx1250":
        from .fused_compress_attn_hca_gfx1250 import flydsl_hca_compress_attn_gfx1250

        return flydsl_hca_compress_attn_gfx1250(
            kv_in=kv_in,
            score_in=score_in,
            kv_state=kv_state,
            score_state=score_state,
            state_slot_mapping=state_slot_mapping,
            plan_gpu=plan_gpu,
            ape=ape,
            rms_weight=rms_weight,
            rms_eps=rms_eps,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            kv_cache=kv_cache,
            block_tables=block_tables,
            k_per_block=k_per_block,
            ratio=ratio,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            kv_compressed_scratch=kv_compressed_scratch,
            quant=quant,
            k_rope_cache=k_rope_cache,
            quant_group_size=quant_group_size,
            k_split_num_waves=k_split_num_waves,
            slice_size=slice_size,
            stream=stream,
        )

    if k_split_num_waves is None or slice_size is None:
        # Local import to avoid a circular import between the two HCA modules
        # at package init time.
        from .fused_compress_attn import hca_per_n_config

        auto_slice, auto_kw = hca_per_n_config(plan_gpu.shape[0])
        if slice_size is None:
            slice_size = auto_slice
        if k_split_num_waves is None:
            k_split_num_waves = auto_kw
    # User-facing input validation -- must be ``raise`` not ``assert`` (asserts
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

    if quant:
        if kv_cache.dtype not in (torch.float8_e4m3fnuz, torch.float8_e4m3fn):
            raise TypeError(
                f"HCA fp8 kv_cache must be fp8 (e4m3fnuz/e4m3fn); got {kv_cache.dtype}"
            )
        if k_rope_cache is None:
            raise ValueError(
                "HCA fp8 path requires k_rope_cache (paged bf16 rope buffer)"
            )
        if k_rope_cache.dtype != torch.bfloat16:
            raise TypeError(f"k_rope_cache must be bf16; got {k_rope_cache.dtype}")
        if k_rope_cache.dim() != 3 or k_rope_cache.shape[2] != rope_head_dim:
            raise ValueError(
                f"k_rope_cache shape {tuple(k_rope_cache.shape)} != [NB, k_per_block, {rope_head_dim}]"
            )
        if not k_rope_cache.is_contiguous():
            raise ValueError("k_rope_cache must be contiguous")
    else:
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
    # so replay is a no-op -> HCA boundaries silently never fire in decode CG.
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
    # Kernel-B wave packing (k_waves rows/block): packing amortizes block-launch/
    # scheduling overhead and wins big at small N (launch-bound) and large N
    # (scheduling-bound), but slightly hurts the mid-range (64-512) where 1-wave
    # blocks spread wider across CUs. Pick by plan_capacity (fixed per graph ->
    # CG-safe; the chosen variant is a distinct lru_cache'd compile).
    norm_kw = 4 if (plan_capacity <= 32 or plan_capacity >= 1024) else 1
    norm_fn = compile_hca_norm_rope_scatter(
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        ratio=ratio,
        k_per_block=k_per_block,
        rms_weight_is_bf16=rms_weight_is_bf16,
        rms_eps=rms_eps,
        quant=quant,
        quant_group_size=quant_group_size,
        k_waves=norm_kw,
    )
    # k_rope_buff is referenced only on the quant path; pass kv_cache as a dummy
    # (valid tensor, never read) when bf16 so the launcher arity stays fixed.
    if quant:
        krope_buf = k_rope_cache
        krope_bs = int(k_rope_cache.stride(0))
        krope_ts = int(k_rope_cache.stride(1))
    else:
        krope_buf = kv_cache
        krope_bs = 0
        krope_ts = 0
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
        krope_buf,
        krope_bs,
        krope_ts,
        int(plan_capacity),
        stream_obj,
    )
    _run_compiled(norm_fn, *norm_args)


def flydsl_hca_compress_forward(
    *,
    kv_in: torch.Tensor,  # [num_q_tokens, head_dim] bf16
    score_in: torch.Tensor,  # [num_q_tokens, head_dim] bf16
    kv_state: torch.Tensor,  # [num_slots, STATE_SIZE, head_dim] f32
    score_state: torch.Tensor,  # same shape, ape pre-added
    state_slot_mapping: torch.Tensor,  # [bs] i32
    plan_gpu: torch.Tensor,  # [num_compress, 4] i32
    ape: torch.Tensor,  # [ratio, head_dim] f32
    ratio: int,
    head_dim: int,
    kv_compressed_out: torch.Tensor | None = None,
    k_split_num_waves: int | None = None,
    slice_size: int | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """HCA pool ONLY (Kernel A): softmax-pool ratio source positions (state-cache
    ring + ragged input + ape) -> ``kv_compressed[num_compress, head_dim]`` fp32.

    Split out of :func:`flydsl_hca_compress_attn` so the FP8 path can pool here
    and route the norm+rope+quant+scatter to the C++
    ``fused_kv_norm_rope_group_quant`` (cast the fp32 ``kv_compressed`` to bf16
    first; that bf16 round-trip is lossless relative to the final FP8 output).
    Returns the (allocated or caller-supplied) ``kv_compressed`` fp32 tensor.
    """
    from aiter.jit.utils.chip_info import get_gfx as _get_gfx

    if _get_gfx() == "gfx1250":
        raise NotImplementedError(
            "flydsl_hca_compress_forward standalone pool: gfx1250 path not wired"
        )
    if head_dim != 512 or ratio != 128:
        raise ValueError(
            f"HCA pool only supports head_dim=512, ratio=128; got {head_dim}/{ratio}"
        )
    if kv_in.dtype != torch.bfloat16 or score_in.dtype != torch.bfloat16:
        raise TypeError("kv_in/score_in must be bf16")
    if kv_state.dtype != torch.float32 or score_state.dtype != torch.float32:
        raise TypeError("kv_state/score_state must be fp32")
    if ape.shape != (ratio, head_dim) or ape.dtype != torch.float32:
        raise ValueError(f"ape must be ({ratio},{head_dim}) f32")

    plan_capacity = plan_gpu.shape[0]
    state_size = kv_state.shape[1]
    if k_split_num_waves is None or slice_size is None:
        from .fused_compress_attn import hca_per_n_config

        auto_slice, auto_kw = hca_per_n_config(plan_capacity)
        slice_size = slice_size if slice_size is not None else auto_slice
        k_split_num_waves = (
            k_split_num_waves if k_split_num_waves is not None else auto_kw
        )

    if kv_compressed_out is None:
        kv_compressed = torch.empty(
            (plan_capacity, head_dim), dtype=torch.float32, device=kv_in.device
        )
    else:
        if kv_compressed_out.shape != (plan_capacity, head_dim):
            raise ValueError("kv_compressed_out shape mismatch")
        kv_compressed = kv_compressed_out
    if plan_capacity == 0:
        return kv_compressed

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
    _run_compiled(
        compress_fn,
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
    return kv_compressed
