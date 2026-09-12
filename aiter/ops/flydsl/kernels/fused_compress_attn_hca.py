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
from flydsl.expr import arith, const_expr, fastmath, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Stream, T

from .fused_compress_attn_common import (
    block_base_bytes_i64,
    emit_group_fp8_nm_asm_scatter,
    state_slot_byte_offset,
)
from .tensor_shim import _run_compiled, _to_raw, ptr_buf_tensor

BLOCK_THREADS = 64  # 1 wave64
SLICE = 64  # head_dim elements per block (grid-Y split)
_NEG_INF = float("-inf")
_LOG2E = math.log2(math.e)


def _ptr_at_byte_off(tensor, base_i64):
    """Global byte pointer at ``tensor``'s base + ``base_i64`` (64-bit byte offset).

    Used to fold a slot/block rebase into the pointer handed to
    ``ptr_buf_tensor``, which re-derives the descriptor's element type -- so the
    i8 carrier type here only carries the address and the ptrtoint/inttoptr
    roundtrip folds away pre-ISA, keeping the emitted V# identical to the old
    ``buf_tensor(base_i64=...)`` descriptor.
    """
    pt = fx.PointerType.get(T.i8, address_space=fx.AddressSpace.Global, alignment=1)
    return fx.inttoptr(
        pt, fx.Int64(fx.ptrtoint(fx.get_iter(tensor))) + fx.Int64(base_i64)
    )


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

        # -inf sentinel + its 0.0 partner feed the softmax maximumf / == compare,
        # which must stay non-fast (a plain fx const would take ambient fast_fp_math
        # and corrupt the -inf guard) -- keep both raw.
        c_neg_inf = arith.constant(_NEG_INF, type=f32)
        c_zero_f32 = arith.constant(0.0, type=f32)
        c_log2e = fx.Float32(_LOG2E)

        def fexp_f32(x):
            # x is fx.Float32; exp2 needs a raw operand -> wrap once here.
            return fx.rocdl.exp2(f32, _to_raw(x * c_log2e))

        # Per-thread wave / lane (block-local).
        wid = fx.Int32(tid) // 64  # -> [0, NW)
        lid = fx.Int32(tid) % 64  # -> [0, 64)

        # -- Load plan row ----------------------------------------------
        plan_buf = ptr_buf_tensor(fx.get_iter(plan), fx.Int32)
        plan_vec = fx.Vector(
            fx.add_offset(fx.get_iter(plan_buf), fx.Int32(pid) * 4).load(T.vec(4, i32))
        )
        ragged_id = plan_vec[0]
        batch_id = plan_vec[1]
        position = plan_vec[2]
        window_len = plan_vec[3]

        # Sentinel-skip: run the whole body only for position >= 0, as a closure
        # under a runtime `if` (rewriter sees an opaque call -> scf.if).
        def _body():
            # Per-thread head_dim base: each thread owns VEC contiguous
            # elements starting at slice_base + lid * VEC.
            col_off_base = fx.Int32(sid) * SLICE_SZ + fx.Int32(lid) * VEC

            slot_map_buf = ptr_buf_tensor(fx.get_iter(state_slot_mapping), fx.Int32)
            slot = fx.add_offset(fx.get_iter(slot_map_buf), batch_id).load(i32)

            # bf16 inputs are read as i32 dwords (unaligned bit-extract) -> i32 buf.
            kv_in_buf = ptr_buf_tensor(fx.get_iter(kv_in), fx.Int32)
            score_in_buf = ptr_buf_tensor(fx.get_iter(score_in), fx.Int32)
            # Rebased onto this program's slot — the 64-bit slot byte offset is
            # folded into the descriptor base (see `state_slot_byte_offset`).
            kv_state_buf = ptr_buf_tensor(
                _ptr_at_byte_off(
                    kv_state, state_slot_byte_offset(slot, kv_state_slot_stride)
                ),
                fx.Float32,
            )
            score_state_buf = ptr_buf_tensor(
                _ptr_at_byte_off(
                    score_state, state_slot_byte_offset(slot, score_state_slot_stride)
                ),
                fx.Float32,
            )
            ape_buf = ptr_buf_tensor(fx.get_iter(ape), fx.Float32)

            def _load_bf16_vec_to_f32(buf, base_off_elems_i32):
                """Load VEC contiguous bf16 elements starting at
                ``base_off_elems_i32`` -> list of VEC f32 values.

                VEC=1: unaligned-safe scalar via dword + bit-extract.
                VEC>=2: vectorized i32 load + bitcast to bf16.
                """
                base_off = fx.Int32(base_off_elems_i32)
                if const_expr(VEC == 1):
                    off_dw = base_off >> 1
                    lane_in_dw = base_off & 1
                    raw_s = fx.Int32(fx.add_offset(fx.get_iter(buf), off_dw).load(i32))
                    # logical (unsigned) shift for the hi-word extract: fx Int32 >>
                    # is arithmetic -> use Uint32 to keep v_lshrrev_b32.
                    hi = fx.Int32((fx.Uint32(raw_s) >> 16).ir_value())
                    lo_or_hi = (lane_in_dw == fx.Int32(0)).select(raw_s, hi)
                    lo16 = lo_or_hi & 0xFFFF
                    lo16_v = fx.Vector.from_elements([lo16], dtype=fx.Int32)
                    bf16_pair = lo16_v.bitcast(fx.BFloat16)
                    return [bf16_pair[0].to(fx.Float32)]
                else:
                    # base must be VEC-aligned (caller guarantees by
                    # col_off_base = sid*SLICE + lid*VEC, both multiples of VEC).
                    off_dw = base_off >> 1
                    dwords = VEC // 2  # VEC bf16 = VEC*2 bytes
                    if const_expr(dwords == 1):
                        # vec_width=1 returns scalar i32; wrap into vec<1xi32>
                        # before bitcast to vec<2xbf16>.
                        raw = fx.Vector.from_elements(
                            [fx.add_offset(fx.get_iter(buf), off_dw).load(i32)],
                            dtype=fx.Int32,
                        )
                    else:
                        raw = fx.Vector(
                            fx.add_offset(fx.get_iter(buf), off_dw).load(
                                T.vec(dwords, i32)
                            )
                        )
                    vec_bf16 = raw.bitcast(fx.BFloat16)
                    return [vec_bf16[i].to(fx.Float32) for i in range_constexpr(VEC)]

            def _load_f32_vec(buf, base_off_elems_i32):
                """Load VEC f32 starting at base -> list of VEC f32 values."""
                if const_expr(VEC <= 4):
                    if const_expr(VEC == 1):
                        # width=1 returns scalar, not 1-vec.
                        return [
                            fx.Float32(
                                fx.add_offset(
                                    fx.get_iter(buf), base_off_elems_i32
                                ).load(f32)
                            )
                        ]
                    raw = fx.Vector(
                        fx.add_offset(fx.get_iter(buf), base_off_elems_i32).load(
                            T.vec(VEC, f32)
                        )
                    )
                    return [raw[i] for i in range(VEC)]
                else:
                    # VEC == 8: AMD HW max is dwordx4 -> 2 loads.
                    assert VEC == 8
                    half = VEC // 2
                    base = fx.Int32(base_off_elems_i32)
                    r0 = fx.Vector(
                        fx.add_offset(fx.get_iter(buf), base).load(T.vec(half, f32))
                    )
                    r1 = fx.Vector(
                        fx.add_offset(fx.get_iter(buf), base + half).load(
                            T.vec(half, f32)
                        )
                    )
                    return [r0[i] for i in range(half)] + [r1[i] for i in range(half)]

            def _issue_phase2_loads(k_i32):
                """Phase 2 (ragged input) loads. Returns (kv_list, sc_list,
                ape_list) each of length VEC."""
                k = fx.Int32(k_i32)
                ape_row = k % ratio
                in_row_raw = fx.Int32(ragged_id) - (fx.Int32(K - 1) - k)
                in_row = (in_row_raw > fx.Int32(0)).select(in_row_raw, fx.Int32(0))
                base_in_off = in_row * fx.Int32(kv_in_row_stride) + col_off_base
                base_sc_off = in_row * fx.Int32(score_in_row_stride) + col_off_base
                base_ape_off = ape_row * DIM_FULL + col_off_base
                kv = _load_bf16_vec_to_f32(kv_in_buf, base_in_off)
                sc = _load_bf16_vec_to_f32(score_in_buf, base_sc_off)
                ape_v = _load_f32_vec(ape_buf, base_ape_off)
                return kv, sc, ape_v

            def _issue_phase1_loads(k_i32):
                """Phase 1 (state cache) loads. Returns (kv_list, sc_padded_list)
                each of length VEC. Score is -inf when s < 0."""
                s = fx.Int32(position) - fx.Int32(K - 1) + fx.Int32(k_i32)
                is_pad = s < fx.Int32(0)
                s_safe = is_pad.select(fx.Int32(0), s)
                ring = s_safe % state_size
                # Slot term already folded into the descriptor base.
                base_kv_off = ring * fx.Int32(kv_state_pos_stride) + col_off_base
                base_sc_off = ring * fx.Int32(score_state_pos_stride) + col_off_base
                kv_list = _load_f32_vec(kv_state_buf, base_kv_off)
                sc_list = _load_f32_vec(score_state_buf, base_sc_off)
                neg_inf = fx.Float32(c_neg_inf)
                sc_padded = [is_pad.select(neg_inf, sc_list[i]) for i in range(VEC)]
                return kv_list, sc_padded

            def _softmax_step_padded(
                m_old_list, kv_old_list, w_old_list, score_k_list, kv_k_list
            ):
                """Padding-aware vector softmax step over VEC lanes. When
                score_k == -inf, w_k is forced to 0 (avoids NaN when m_old
                is also -inf). Safe in both Phase 1 (padding can occur) and
                Phase 2 (score finite -> pad-select branch is dead code).
                """
                neg_inf = fx.Float32(c_neg_inf)
                zero = fx.Float32(c_zero_f32)
                new_m, new_kv, new_w = [], [], []
                for i in range_constexpr(VEC):
                    m_old = fx.Float32(m_old_list[i])
                    kv_old = fx.Float32(kv_old_list[i])
                    w_old = fx.Float32(w_old_list[i])
                    score_k = fx.Float32(score_k_list[i])
                    kv_k = fx.Float32(kv_k_list[i])
                    # maximumf keeps the ambient fast_fp_math (matches the old
                    # raw maximumf, which resolved fastmath<fast>). The OEQ
                    # compares against -inf MUST emit fastmath<none> -- ambient
                    # fast carries ninf, which lets `x == -inf` fold to false
                    # and drops the sentinel select -- so scope only the `==`
                    # to fastmath(None) (== the old raw cmpf, no fastmath).
                    m_new = m_old.maximumf(score_k)
                    with fastmath(None):
                        is_first = m_old == neg_inf
                    scale_active = fx.Float32(fexp_f32(m_old - m_new))
                    scale_v = is_first.select(zero, scale_active)
                    wk_active = fx.Float32(fexp_f32(score_k - m_new))
                    with fastmath(None):
                        is_pad_score = score_k == neg_inf
                    w_k = is_pad_score.select(zero, wk_active)
                    new_kv.append((kv_old * scale_v + w_k * kv_k).ir_value())
                    new_w.append((w_old * scale_v + w_k).ir_value())
                    new_m.append(m_new.ir_value())
                return new_m, new_kv, new_w

            # -- Wave's K range: [wid * K_PER_WAVE, (wid+1) * K_PER_WAVE) --
            k_start_i32 = fx.Int32(wid) * K_PER_WAVE
            k_end_i32 = k_start_i32 + K_PER_WAVE

            # Split point inside this wave's K range. Each wave sees a
            # window_len-dependent slice of Phase 1 followed by Phase 2.
            # Cases (`wl = window_len`):
            #   wl <= k_start:  pure Phase 2 (entire wave is input)
            #   wl >= k_end:    pure Phase 1 (entire wave is state cache)
            #   else:          mixed (Phase 1 in [k_start, wl), Phase 2 in [wl, k_end))
            # ``split`` = clamp(wl, k_start, k_end) gives the boundary;
            # both sub-loops are empty when their bound collapses, so any
            # of the three cases naturally falls out.
            wl = fx.Int32(window_len)
            split_lo = (wl > k_start_i32).select(wl, k_start_i32)
            split_i32 = (split_lo < k_end_i32).select(split_lo, k_end_i32)

            # State is 3*VEC scalars: m_lane[VEC] + kv_lane[VEC] + w_lane[VEC].
            init_m = [c_neg_inf for _ in range(VEC)]
            init_kv = [c_zero_f32 for _ in range(VEC)]
            init_w = [c_zero_f32 for _ in range(VEC)]
            init_state = init_m + init_kv + init_w

            # Sub-loop 1: Phase 1 sub-range [k_start, split). Reads state
            # cache; padded softmax (score can be -inf).
            phase1_local = init_state
            for k_static, state in range(
                k_start_i32.ir_value(), split_i32.ir_value(), 1, init=init_state
            ):
                m_lane = list(state[0:VEC])
                kv_lane = list(state[VEC : 2 * VEC])
                w_lane = list(state[2 * VEC : 3 * VEC])
                k_i32 = fx.Int32(k_static)
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
                split_i32.ir_value(), k_end_i32.ir_value(), 1, init=phase1_local
            ):
                m_lane = list(state[0:VEC])
                kv_lane = list(state[VEC : 2 * VEC])
                w_lane = list(state[2 * VEC : 3 * VEC])
                k_i32 = fx.Int32(k_static)
                p2_kv, p2_sc, p2_ape = _issue_phase2_loads(k_i32)
                p2_score = [
                    (fx.Float32(p2_sc[i]) + p2_ape[i]).ir_value() for i in range(VEC)
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
            lds_thread_base = fx.Int32(wid) * SLICE_SZ + fx.Int32(lid) * VEC
            for i in range_constexpr(VEC):
                idx_i = lds_thread_base + i
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
                    lane_off = fx.Int32(lid) * VEC + i
                    # Global max across NW waves for this element.
                    m_g = fx.Float32(c_neg_inf)
                    m_arr = []
                    for w in range_constexpr(NW):
                        idx_w = w * SLICE_SZ + lane_off
                        m_w = fx.ptr_load(lds_m_ptr + idx_w)
                        m_arr.append(m_w)
                        m_g = m_g.maximumf(m_w)

                    # Weighted sums (kv * scale_w) and (w * scale_w).
                    kv_sum = fx.Float32(0.0)
                    w_sum = fx.Float32(0.0)
                    for w in range_constexpr(NW):
                        idx_w = w * SLICE_SZ + lane_off
                        kv_w = fx.ptr_load(lds_kv_ptr + idx_w)
                        w_w = fx.ptr_load(lds_w_ptr + idx_w)
                        m_w = m_arr[w]
                        scale_w = fx.Float32(fexp_f32(m_w - m_g))
                        kv_sum = kv_sum + kv_w * scale_w
                        w_sum = w_sum + w_w * scale_w
                    rcp_w = fx.Float32(fx.rocdl.rcp(f32, w_sum.ir_value()))
                    comp_list.append(kv_sum * rcp_w)

                # -- Vectorized write of VEC f32 comp values --
                out_buf = ptr_buf_tensor(fx.get_iter(kv_compressed), fx.Float32)
                out_off = (
                    fx.Int32(pid) * fx.Int32(kv_compressed_row_stride) + col_off_base
                )
                if const_expr(VEC == 1):
                    fx.add_offset(fx.get_iter(out_buf), out_off).store(comp_list[0])
                elif const_expr(VEC <= 4):
                    out_vec = fx.Vector.from_elements(comp_list, dtype=fx.Float32)
                    fx.add_offset(fx.get_iter(out_buf), out_off).store(out_vec)
                else:
                    # VEC == 8: AMD HW max is dwordx4 -> 2 stores.
                    assert VEC == 8
                    half = VEC // 2
                    v0 = fx.Vector.from_elements(comp_list[0:half], dtype=fx.Float32)
                    v1 = fx.Vector.from_elements(comp_list[half:VEC], dtype=fx.Float32)
                    fx.add_offset(fx.get_iter(out_buf), out_off).store(v0)
                    fx.add_offset(fx.get_iter(out_buf), out_off + half).store(v1)

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

        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        # Pack KW plan rows per block: wave_id picks the row, lane indexes head_dim.
        # logical shift (tid >= 0); fx Int32 >> is arithmetic -> use Uint32.
        wave_id = fx.Int32((fx.Uint32(tid) >> log2_wave).ir_value())
        lane = fx.Int32(tid) & (WAVE - 1)
        pid = fx.Int32(bid) * KW + wave_id

        c_eps = fx.Float32(rms_eps)
        c_inv_D = fx.Float32(1.0 / D)

        def wave_reduce_add(x):
            w = fx.Float32(x)
            for sh_exp in range_constexpr(log2_wave):
                off = WAVE // (2 << sh_exp)
                w = w + w.shuffle_xor(off, WAVE)
            return w

        # -- Load plan row --
        plan_buf = ptr_buf_tensor(fx.get_iter(plan), fx.Int32)
        plan_vec = fx.Vector(
            fx.add_offset(fx.get_iter(plan_buf), pid * 4).load(T.vec(4, i32))
        )
        batch_id = plan_vec[1]
        position = plan_vec[2]

        # active = real plan row (position>=0 sentinel) AND within capacity (tail
        # waves of the last block have pid>=cap and must bail; their plan load is
        # bounds-checked to 0 by the buffer resource, so guard explicitly here).
        is_active = (fx.Int32(position) >= 0) & (pid < plan_capacity)

        # Whole body as a closure under the runtime guard (opaque call -> scf.if).
        def _body():
            tid_x_vec = lane * VEC

            # -- Load kv_compressed[pid, tid*VEC : tid*VEC + VEC] --
            kvc_buf = ptr_buf_tensor(fx.get_iter(kv_compressed), fx.Float32)
            base_off = pid * fx.Int32(kv_compressed_row_stride) + tid_x_vec
            # VEC ? {2, 4, 8}: VEC <= 4 -> single dwordx{VEC}; VEC=8 -> 2x dwordx4.
            if const_expr(VEC <= 4):
                raw = fx.Vector(
                    fx.add_offset(fx.get_iter(kvc_buf), base_off).load(T.vec(VEC, f32))
                )
                comp_lane = [raw[i] for i in range(VEC)]
            else:
                assert VEC == 8
                half = 4
                r0 = fx.Vector(
                    fx.add_offset(fx.get_iter(kvc_buf), base_off).load(T.vec(half, f32))
                )
                r1 = fx.Vector(
                    fx.add_offset(fx.get_iter(kvc_buf), base_off + half).load(
                        T.vec(half, f32)
                    )
                )
                comp_lane = [r0[i] for i in range(half)] + [r1[i] for i in range(half)]

            # -- RMSNorm (wave reduce-add of squares / D + eps; rsqrt) --
            sq_local = fx.Float32(0.0)
            for i in range_constexpr(VEC):
                sq_local = sq_local + comp_lane[i] * comp_lane[i]
            sq_full = wave_reduce_add(sq_local)
            var = sq_full * c_inv_D
            rrms = fmath.rsqrt((var + c_eps).ir_value(), fastmath=fm_fast)

            # rms_weight load
            if const_expr(rms_weight_is_bf16):
                # bf16 weights read as i32 dwords -> i32 buf.
                rmsw_buf = ptr_buf_tensor(fx.get_iter(rms_weight), fx.Int32)
                dwords = (VEC + 1) // 2
                off_dw = fx.Int32((fx.Uint32(tid_x_vec) >> 1).ir_value())
                if const_expr(dwords == 1):
                    raw = fx.Vector.from_elements(
                        [fx.add_offset(fx.get_iter(rmsw_buf), off_dw).load(i32)],
                        dtype=fx.Int32,
                    )
                else:
                    raw = fx.Vector(
                        fx.add_offset(fx.get_iter(rmsw_buf), off_dw).load(
                            T.vec(dwords, i32)
                        )
                    )
                vec_bf16 = raw.bitcast(fx.BFloat16)
                rmsw_lane = [vec_bf16[i].to(fx.Float32) for i in range_constexpr(VEC)]
            else:
                rmsw_buf = ptr_buf_tensor(fx.get_iter(rms_weight), fx.Float32)
                if const_expr(VEC <= 4):
                    raw = fx.Vector(
                        fx.add_offset(fx.get_iter(rmsw_buf), tid_x_vec).load(
                            T.vec(VEC, f32)
                        )
                    )
                    rmsw_lane = [raw[i] for i in range(VEC)]
                else:
                    half = 4
                    r0 = fx.Vector(
                        fx.add_offset(fx.get_iter(rmsw_buf), tid_x_vec).load(
                            T.vec(half, f32)
                        )
                    )
                    r1 = fx.Vector(
                        fx.add_offset(fx.get_iter(rmsw_buf), tid_x_vec + half).load(
                            T.vec(half, f32)
                        )
                    )
                    rmsw_lane = [r0[i] for i in range(half)] + [
                        r1[i] for i in range(half)
                    ]

            normed_lane = [comp_lane[i] * rrms * rmsw_lane[i] for i in range(VEC)]

            # -- GPT-J RoPE on RD tail --
            comp_pos_i32 = (fx.Int32(position) // ratio) * ratio
            cos_buf = ptr_buf_tensor(fx.get_iter(cos_cache), fx.BFloat16)
            sin_buf = ptr_buf_tensor(fx.get_iter(sin_cache), fx.BFloat16)
            cos_row_base = comp_pos_i32 * (RD // 2)

            is_rope_t = lane >= fx.Int32(ROPE_THREAD_LO)
            rope_rel_raw = lane - ROPE_THREAD_LO
            rope_rel = (rope_rel_raw > fx.Int32(0)).select(rope_rel_raw, fx.Int32(0))
            cs_lo = rope_rel * PAIRS_PER_THREAD

            if const_expr(PAIRS_PER_THREAD == 1):
                cos_b = fx.add_offset(fx.get_iter(cos_buf), cos_row_base + cs_lo).load(
                    T.bf16
                )
                sin_b = fx.add_offset(fx.get_iter(sin_buf), cos_row_base + cs_lo).load(
                    T.bf16
                )
                cos_vals = [fx.BFloat16(cos_b).to(fx.Float32)]
                sin_vals = [fx.BFloat16(sin_b).to(fx.Float32)]
            else:
                cos_vec = fx.Vector(
                    fx.add_offset(fx.get_iter(cos_buf), cos_row_base + cs_lo).load(
                        T.vec(PAIRS_PER_THREAD, T.bf16)
                    )
                )
                sin_vec = fx.Vector(
                    fx.add_offset(fx.get_iter(sin_buf), cos_row_base + cs_lo).load(
                        T.vec(PAIRS_PER_THREAD, T.bf16)
                    )
                )
                cos_vals = [cos_vec[i].to(fx.Float32) for i in range(PAIRS_PER_THREAD)]
                sin_vals = [sin_vec[i].to(fx.Float32) for i in range(PAIRS_PER_THREAD)]

            rotated_lane = list(normed_lane)
            for k in range_constexpr(PAIRS_PER_THREAD):
                e = normed_lane[2 * k]
                o = normed_lane[2 * k + 1]
                c = cos_vals[k]
                s = sin_vals[k]
                rotated_lane[2 * k] = e * c - o * s
                rotated_lane[2 * k + 1] = e * s + o * c

            # -- Paged scatter dest (shared by bf16 / fp8) --
            ci = fx.Int32(position) // ratio
            block_in_seq = ci // k_per_block
            slot_in_block = ci % k_per_block
            bt_buf = ptr_buf_tensor(fx.get_iter(block_table), fx.Int32)
            bt_off = (
                fx.Int32(batch_id) * fx.Int32(block_table_seq_stride) + block_in_seq
            )
            physical_block = fx.add_offset(fx.get_iter(bt_buf), bt_off).load(i32)
            # The block term rides on the descriptor's base, not on the
            # 32-bit offset -- see `block_base_bytes_i64`. What is left is one
            # block's worth, which fits by construction.
            # The block term rides on the descriptor's base (i64 fold), so the
            # 32-bit voffset only spans one block. `block_base_bytes_i64` elem_bytes:
            # fp8 kv_cache=1, bf16 kv_cache=2.
            cache_block_base = block_base_bytes_i64(
                physical_block, kv_cache_block_stride, 1 if quant else 2
            )
            cache_base = slot_in_block * fx.Int32(kv_cache_token_stride)

            if const_expr(quant):
                # -- group_fp8 (V4 nm-asm) via shared emitter (single source of truth
                # shared with the CSA single-kernel; fp8 entry layout stays identical).
                # The emitter stores through direct global pointers built from the
                # i64 block base (kv_cache / k_rope) we pass below. --
                _krope_base = slot_in_block * fx.Int32(krope_token_stride)
                emit_group_fp8_nm_asm_scatter(
                    normed_lane=[v.ir_value() for v in normed_lane],
                    rotated_lane=[v.ir_value() for v in rotated_lane],
                    lane=lane.ir_value(),
                    is_rope_t=is_rope_t.ir_value(),
                    cache_base=cache_base.ir_value(),
                    out_base_i64=fx.Int64(fx.ptrtoint(fx.get_iter(kv_cache)))
                    + fx.Int64(cache_block_base),
                    krope_base=_krope_base.ir_value(),
                    krope_base_i64=fx.Int64(fx.ptrtoint(fx.get_iter(k_rope_buff)))
                    + fx.Int64(
                        block_base_bytes_i64(physical_block, krope_block_stride, 2)
                    ),
                    VEC=VEC,
                    NOPE=NOPE,
                    RTS=RTS,
                    log2_rts=log2_rts,
                    ROPE_THREAD_LO=ROPE_THREAD_LO,
                    wave_width=WAVE,
                )
            else:
                # ---- BF16 single-buffer scatter (nope + rope contiguous) ----
                # bf16 kv_cache written as i32 dwords -> i32 buf, block base folded.
                out_buf = ptr_buf_tensor(
                    _ptr_at_byte_off(kv_cache, cache_block_base),
                    fx.Int32,
                )
                out_lane = [
                    is_rope_t.select(rotated_lane[i], normed_lane[i])
                    for i in range_constexpr(VEC)
                ]
                cache_off = cache_base + tid_x_vec
                bf16_vec = fx.Vector.from_elements(out_lane, dtype=fx.Float32).to(
                    fx.BFloat16
                )
                bf16_as_i32 = bf16_vec.bitcast(fx.Int32)
                cache_off_dw = fx.Int32((fx.Uint32(cache_off) >> 1).ir_value())
                dwords = (VEC + 1) // 2
                if const_expr(dwords == 1):
                    fx.add_offset(fx.get_iter(out_buf), cache_off_dw).store(
                        bf16_as_i32[0]
                    )
                else:
                    fx.add_offset(fx.get_iter(out_buf), cache_off_dw).store(bf16_as_i32)

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
        nblocks = (fx.Int32(plan_capacity) + (KW - 1)) // KW
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
    # Slot and ring strides are passed to the kernel and the descriptor is
    # rebased per slot, so the states may be strided views — a per-request
    # arena hands out a view whose slot stride is a whole entry. Only the
    # innermost dim must be unit stride: the kernel addresses it as
    # `col_off + lane`.
    if kv_state.stride(-1) != 1 or score_state.stride(-1) != 1:
        raise ValueError("kv_state/score_state inner stride must be 1")
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
        if k_rope_cache.stride(2) != 1:
            raise ValueError("k_rope_cache must be dense in the last dim")
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
