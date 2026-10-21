# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused Compressor boundary kernel for V4 attention -- **gfx1250 (RDNA4, wave32)**.

Port of ``fused_compress_attn.py`` (wave64) to gfx1250 wave32.
Key differences from the wave64 version:
  - BLOCK_THREADS = 32 (wave32)
  - VEC = D / 32 (D=512 -> VEC=16; D=128 -> VEC=4)
  - VEC=16 load/store paths (4x dwordx4 for f32, 2x dwordx4 for bf16)
  - preshuffle forced False (MFMA preshuffle is gfx9-only)
  - Kernel names suffixed with "w32" to avoid JIT cache collision
  - FP8 VEC=4 packing path (no pair-coop shuffle needed)

Three kernel families:
  - Single-wave (``_build_kernel``): 1 wave32 per boundary
  - K-split multi-wave (``_build_kernel_ksplit``): K split across NW waves,
    block = 32*NW
  - FP4 CSA indexer (``_build_kernel_fp4``): 1 wave32 per boundary, the design of
    the Gluon kernel; quant_mode="fp4" always uses it

See ``fused_compress_attn.py`` for the original wave64 documentation.
"""

# NOTE: do NOT add `from __future__ import annotations` (see qk_norm_rope_quant
# header note -- PEP 563 breaks flydsl's runtime/constexpr param detection,
# triggering a JIT recompile per dynamic-arg value).

import math
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, const_expr, fastmath, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.arith import CmpFPredicate
from flydsl.expr.typing import Int32, Stream, T

from aiter.ops.flydsl.kernels import buffer_ops

from .fused_compress_attn_common import (
    _NEG_INF,
    _PRESHUFFLE_TILE,
    _fexp_f32,
    _fp8_const,
    _split_state,
    block_base_bytes_i64,
    emit_group_fp8_nm_asm_scatter,
    state_slot_byte_offset,
)
from .kernels_common import LOG2E as _LOG2E
from .tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    _run_compiled,
    _to_raw,
)

# --- shape constants --------------------------------------------------------
BLOCK_THREADS = 32  # 1 wave32 (RDNA4 / gfx1250); D must be a multiple
_FP4_GROUP_SIZE = 32


# ============================================================================
# Kernel builder
# ============================================================================


def _load_f32_vec(rsrc, off_elems_i32, vec_width):
    """Load vec_width fp32 from byte-aligned stream -> list of vec_width fp32 scalars.

    For vec_width=2 -> dwordx2; vec_width=4 -> dwordx4; vec_width=8 -> 2x dwordx4;
    vec_width=16 -> 4x dwordx4 (HW max is dwordx4).
    """
    f32 = T.f32
    if const_expr(vec_width <= 4):
        raw = fx.Vector(
            buffer_ops.buffer_load(rsrc, off_elems_i32, vec_width=vec_width, dtype=f32)
        )
        return [_to_raw(raw[i]) for i in range(vec_width)]
    else:
        # vec_width in {8, 16} -> split into quarter=4 chunks
        quarter = 4
        n_chunks = vec_width // quarter
        out = []
        for q in range_constexpr(n_chunks):
            r = fx.Vector(
                buffer_ops.buffer_load(
                    rsrc,
                    fx.Int32(off_elems_i32) + q * quarter,
                    vec_width=quarter,
                    dtype=f32,
                )
            )
            for i in range_constexpr(quarter):
                out.append(_to_raw(r[i]))
        return out


def _load_bf16_vec_then_f32(rsrc, off_elems_i32, vec_width):
    """Load vec_width bf16 from byte-aligned dword stream -> fp32 vec_width scalars.

    Returns a list of vec_width fp32 MLIR values.
    vec_width=16 -> dwords=8 -> 2x dwordx4 loads, bitcast each to vec<8,bf16>.
    """
    i32 = T.i32
    f32 = T.f32
    off_dw = fx.Uint32(off_elems_i32) >> 1
    # bf16 vec_width = vec_width * 2 bytes; dwords = vec_width/2.
    dwords = (vec_width + 1) // 2  # ceil(vec_width*2 / 4)
    if const_expr(dwords == 1):
        # buffer_load(vec_width=1) returns a scalar i32; wrap into
        # vec<1xi32> before bitcasting to vec<2xbf16>.
        raw_s = buffer_ops.buffer_load(rsrc, off_dw, vec_width=1, dtype=i32)
        raw = fx.Vector.from_elements([raw_s], dtype=fx.Int32)
        vec_bf16 = raw.bitcast(fx.BFloat16)
        # raw extf: .to() picks up the ambient fast_fp_math and would
        # tag the widen fastmath<fast> (IR drift vs the plain extend).
        return [arith.extf(f32, _to_raw(vec_bf16[i])) for i in range(vec_width)]
    elif const_expr(dwords <= 4):
        raw = fx.Vector(
            buffer_ops.buffer_load(rsrc, off_dw, vec_width=dwords, dtype=i32)
        )
        vec_bf16 = raw.bitcast(fx.BFloat16)
        return [arith.extf(f32, _to_raw(vec_bf16[i])) for i in range(vec_width)]
    else:
        # dwords > 4 (vec_width=16 -> dwords=8): split into 2x dwordx4
        half_dw = 4
        half_bf16 = half_dw * 2  # 8 bf16 per chunk
        out = []
        for chunk in range_constexpr(dwords // half_dw):
            r = fx.Vector(
                buffer_ops.buffer_load(
                    rsrc,
                    off_dw + chunk * half_dw,
                    vec_width=half_dw,
                    dtype=i32,
                )
            )
            vbf16 = r.bitcast(fx.BFloat16)
            for i in range_constexpr(half_bf16):
                out.append(arith.extf(f32, _to_raw(vbf16[i])))
        return out


def _wave_reduce_add(x, log2_block, fm_fast):
    """Butterfly sum across wave32."""
    w = _to_raw(x)
    for sh_exp in range_constexpr(log2_block):
        off = BLOCK_THREADS // (2 << sh_exp)
        peer = _to_raw(fx.Float32(w).shuffle_xor(off, BLOCK_THREADS))
        w = arith.AddFOp(w, peer, fastmath=fm_fast).result
    return w


def _wave_reduce_max(x, log2_block):
    """Butterfly max across wave32 (used by quant path)."""
    w = _to_raw(x)
    for sh_exp in range_constexpr(log2_block):
        off = BLOCK_THREADS // (2 << sh_exp)
        peer = _to_raw(fx.Float32(w).shuffle_xor(off, BLOCK_THREADS))
        w = fx.max(fx.Float32(w), fx.Float32(peer)).ir_value()
    return w


def _build_kernel(
    *,
    head_dim: int,
    rope_head_dim: int,
    ratio: int,
    overlap: bool,
    state_size: int,
    k_per_block: int,
    has_block_table: bool,
    quant: bool,
    use_ue8m0: bool,
    preshuffle: bool,
    rms_weight_is_bf16: bool,
    rms_eps: float,
    enable_prefetch_input: bool = False,
    quant_mode: str = "per_row_fp8",  # "per_row_fp8" (indexer) | "group_fp8" (CSA/HCA Main, nm-asm) | (future) "fp4"
    quant_group_size: int = 64,
):
    """Build the @flyc.kernel + @flyc.jit launcher for a given config.

    All shape / mode constants are captured via closure. Two launchers with
    different configs coexist safely. Returns the launcher.

    Constexpr knobs:
      - head_dim, rope_head_dim: V4-Pro Main = (512, 64); Indexer = (128, 64)
      - ratio: compression ratio (typ 4)
      - overlap: True -> K = 2*RATIO (CSA), False -> K = RATIO (HCA, no overlap)
      - state_size: ring-buffer modulo of kv_state.shape[1] (>= K)
      - k_per_block: paged cache tokens per block (= block_size // ratio)
      - has_block_table: False -> skip cache scatter (warmup path)
      - quant: True -> fp8 (Indexer-inner); False -> bf16 (Main)
      - use_ue8m0: only when quant=True (round scale to power-of-2)
      - preshuffle: only when quant=True (MFMA 16x16 tile layout)
      - enable_prefetch_input: True -> Phase 2 carries k+1 loads through
        scf.for iter-args so the buffer_load issue overlaps current iter's
        softmax compute. Helps long K (HCA K=128). Larger VEC pays a register
        cost (loop-carry grows by 3*VEC fp32) -- gate off if it regresses.
    """
    D = head_dim
    RD = rope_head_dim
    NOPE = D - RD
    VEC = D // BLOCK_THREADS
    K = (2 if overlap else 1) * ratio
    DIM_FULL = (2 if overlap else 1) * D

    # --- per-thread vec layout ----
    ROPE_THREAD_LO = NOPE // VEC  # first rope-thread tid
    PAIRS_PER_THREAD = VEC // 2  # GPT-J pairs each rope-thread owns
    # For Main (D=512, VEC=16) -> 28 .. 31 are rope threads, 8 pairs each (=64 total).
    # For Indexer (D=128, VEC=4) -> 16 .. 31 are rope threads, 2 pairs each (=64=2RD/2).
    # The RD%(2*VEC) == 0 invariant means rope threads cleanly own whole pairs.

    # FP8 1xG e8m0 group-quant geometry (quant_mode=="group_fp8" only): wave32 ->
    # VEC=16, RTS = G/VEC = 4, N_GROUPS = NOPE/G = 7. Byte-identical to wave64.
    nm_asm = quant_mode == "group_fp8"
    GROUP_SIZE_Q = quant_group_size
    RTS = (GROUP_SIZE_Q // VEC) if nm_asm else 1
    log2_rts = int(math.log2(RTS)) if nm_asm else 0
    if nm_asm:
        assert (
            quant and not preshuffle
        ), "group_fp8: requires quant=True, preshuffle=False"
        assert (
            NOPE % GROUP_SIZE_Q == 0 and GROUP_SIZE_Q % VEC == 0
        ), f"group_fp8: NOPE={NOPE} % G={GROUP_SIZE_Q} and G % VEC={VEC} must be 0"

    assert D % BLOCK_THREADS == 0, f"D={D} must divide BLOCK_THREADS={BLOCK_THREADS}"
    assert VEC in (2, 4, 8, 16), f"VEC={VEC} (D/{BLOCK_THREADS}) outside supported set"
    assert NOPE >= 0 and NOPE % VEC == 0
    assert RD > 0 and RD % 2 == 0 and RD % VEC == 0
    assert state_size >= K, f"state_size={state_size} < K={K}"
    if quant and preshuffle:
        assert D % _PRESHUFFLE_TILE == 0
        assert k_per_block % _PRESHUFFLE_TILE == 0
    if quant and not has_block_table:
        # quant=True with no scatter is meaningless (the scale write is what
        # the FP8 cache reader consumes). Reject early.
        raise ValueError("quant=True requires has_block_table=True")

    # --- kernel name ----
    _name_parts = [
        "fused_compress_attn_w32",
        f"D{D}",
        f"RD{RD}",
        f"R{ratio}",
        ("OVL" if overlap else "NOOVL"),
        f"SS{state_size}",
    ]
    if has_block_table:
        _name_parts.append(f"KB{k_per_block}")
        if quant:
            _name_parts.append("Q")
            if nm_asm:
                _name_parts.append(f"nmasm{GROUP_SIZE_Q}")
            else:
                if use_ue8m0:
                    _name_parts.append("ue8m0")
                if preshuffle:
                    _name_parts.append("psh")
    else:
        _name_parts.append("noBT")
    if rms_weight_is_bf16:
        _name_parts.append("rmsbf16")
    if enable_prefetch_input:
        _name_parts.append("pf")
    _name_parts.append("flydsl")
    _kname = "_".join(_name_parts)

    fm_fast = arith.FastMathFlags.fast

    # --- compile-time scalars used by emitters ----
    log2_block = int(math.log2(BLOCK_THREADS))

    @flyc.kernel(name=_kname)
    def kernel(
        kv_in: fx.Tensor,  # [num_q_tokens, DIM_FULL] bf16, strided
        kv_in_row_stride: Int32,  # bf16-elements
        score_in: fx.Tensor,  # [num_q_tokens, DIM_FULL] bf16, strided
        score_in_row_stride: Int32,
        plan: fx.Tensor,  # [num_compress, 4] i32 (ragged_id, batch_id, position, window_len)
        kv_state: fx.Tensor,  # [num_slots, STATE_SIZE, DIM_FULL] f32
        kv_state_slot_stride: Int32,  # f32-elements
        kv_state_pos_stride: Int32,  # f32-elements
        score_state: fx.Tensor,  # same shape as kv_state
        score_state_slot_stride: Int32,
        score_state_pos_stride: Int32,
        state_slot_mapping: fx.Tensor,  # [bs] i32
        ape: fx.Tensor,  # [ratio, DIM_FULL] f32
        rms_weight: fx.Tensor,  # [D] f32
        cos_cache: fx.Tensor,  # [max_pos, RD/2] bf16
        sin_cache: fx.Tensor,  # [max_pos, RD/2] bf16
        kv_cache: fx.Tensor,  # bf16 OR fp8 [NB, k_per_block, D]
        kv_cache_block_stride: Int32,  # elements (bf16 or fp8 -- caller's responsibility)
        kv_cache_token_stride: Int32,
        cache_scale: fx.Tensor,  # [NB, k_per_block] f32 (dummy if not quant / group_fp8)
        cache_scale_block_stride: Int32,
        k_rope_buff: fx.Tensor,  # group_fp8 only: paged [NB, k_per_block, RD] bf16 rope (dummy otherwise)
        krope_block_stride: Int32,
        krope_token_stride: Int32,
        block_table: fx.Tensor,  # [bs, max_blocks_per_seq] i32 (dummy if not has_bt)
        block_table_seq_stride: Int32,
    ):
        f32 = T.f32
        i32 = T.i32

        # --- thread / block ids ---
        pid = fx.block_idx.x  # one program per plan row
        tid = fx.thread_idx.x  # 0 .. 31

        # --- constants ---
        c_neg_inf = arith.constant(_NEG_INF, type=f32)
        c_zero_f32 = arith.constant(0.0, type=f32)
        c_eps = arith.constant(rms_eps, type=f32)
        c_inv_D = arith.constant(1.0 / D, type=f32)
        c_log2e = arith.constant(_LOG2E, type=f32)

        # ---- Step 1: load plan row (single dwordx4) ----
        # plan layout: each row = 4 contiguous i32 [ragged_id, batch_id, position, window_len].
        # Fuse the 4 scalar loads into one buffer_load_dwordx4 + 4 extracts --
        # saves 3 buffer-load instructions per program (visible at small N
        # where total program count is low).
        plan_rsrc = buffer_ops.create_buffer_resource(plan, max_size=True)
        plan_base = _to_raw(fx.Int32(pid) * 4)
        plan_vec = fx.Vector(
            buffer_ops.buffer_load(plan_rsrc, plan_base, vec_width=4, dtype=i32)
        )
        ragged_id = _to_raw(plan_vec[0])
        batch_id = _to_raw(plan_vec[1])
        position = _to_raw(plan_vec[2])
        window_len = _to_raw(plan_vec[3])

        # ---- Step 2: sentinel-skip ----
        # Guard the entire body on position >= 0. A bare `if cond: return`
        # would NOT early-exit (the tail kernel body still runs with stale
        # values -> OOB faults); an `if cond:` block scoping the whole body
        # lowers to the same scf.if guard the raw IfOp built.
        if fx.Int32(position) >= 0:
            # ---- Step 3: per-seq state slot ----
            slot_map_rsrc = buffer_ops.create_buffer_resource(
                state_slot_mapping, max_size=True
            )
            slot = buffer_ops.buffer_load(
                slot_map_rsrc, batch_id, vec_width=1, dtype=i32
            )

            # ---- Step 4: per-thread element-range bookkeeping ----
            # This thread owns columns [tid*VEC, tid*VEC+VEC) of BLOCK_D.
            tid_x_vec = fx.Int32(tid) * VEC

            # ---- Step 5: online-softmax accumulator init ----
            # 3 * VEC fp32 scalars carried across K iters.
            init_m = [c_neg_inf for _ in range(VEC)]
            init_kv = [c_zero_f32 for _ in range(VEC)]
            init_w = [c_zero_f32 for _ in range(VEC)]
            init_state = list(init_m) + list(init_kv) + list(init_w)

            def _online_softmax_update(
                m_lane,
                kv_lane,
                w_lane,
                score_lane,
                kv_v_lane,
                score_can_be_neg_inf=True,
            ):
                """Per-lane max-rescale update. All inputs/outputs are VEC-long
                lists of fp32 scalars.

                ``score_can_be_neg_inf`` (constexpr): True for Phase 1 (state
                cache may have padding=-inf rows); False for Phase 2 (input
                phase has no padding by construction) -- skips the cmp+select
                guard around exp(score - m_new).
                """
                new_m = []
                new_kv = []
                new_w = []
                for i in range_constexpr(VEC):
                    m_old = m_lane[i]
                    score = score_lane[i]
                    kv_v = kv_v_lane[i]
                    w_old = w_lane[i]
                    kv_old = kv_lane[i]

                    m_old_f = fx.Float32(m_old)
                    score_f = fx.Float32(score)
                    neg_inf_f = fx.Float32(c_neg_inf)
                    m_new = fx.max(m_old_f, score_f).ir_value()
                    # Keep the -inf sentinel comparisons outside ambient fastmath.
                    with fastmath(None):
                        is_first = m_old_f == neg_inf_f
                    scale_active = _fexp_f32(m_old_f - m_new, c_log2e)
                    scale_v = is_first.select(c_zero_f32, scale_active)
                    wk_active = _fexp_f32(score_f - m_new, c_log2e)
                    if const_expr(score_can_be_neg_inf):
                        with fastmath(None):
                            is_pad_score = score_f == neg_inf_f
                        w_k = is_pad_score.select(c_zero_f32, wk_active)
                    else:
                        w_k = wk_active
                    new_m.append(m_new)
                    new_kv.append(
                        _to_raw(
                            fx.Float32(kv_old) * fx.Float32(scale_v)
                            + fx.Float32(w_k) * fx.Float32(kv_v)
                        )
                    )
                    new_w.append(
                        _to_raw(
                            fx.Float32(w_old) * fx.Float32(scale_v) + fx.Float32(w_k)
                        )
                    )
                return new_m, new_kv, new_w

            # Buffer resources reused across K iters.
            kv_in_rsrc = buffer_ops.create_buffer_resource(kv_in, max_size=True)
            score_in_rsrc = buffer_ops.create_buffer_resource(score_in, max_size=True)
            # Rebased onto this program's slot — see `state_slot_byte_offset`.
            kv_state_rsrc = buffer_ops.create_buffer_resource(
                kv_state,
                max_size=True,
                base_byte_offset=state_slot_byte_offset(slot, kv_state_slot_stride),
            )
            score_state_rsrc = buffer_ops.create_buffer_resource(
                score_state,
                max_size=True,
                base_byte_offset=state_slot_byte_offset(slot, score_state_slot_stride),
            )
            ape_rsrc = buffer_ops.create_buffer_resource(ape, max_size=True)

            def _col_off_for_k(k_static_val):
                """Compute col_off ? {0, D} for OVERLAP (==head_dim when k >= RATIO),
                or constant 0 for HCA (no overlap).

                ``k_static_val`` may be a Python int (constexpr) or an MLIR i32 value.
                """
                if const_expr(not overlap):
                    return arith.constant(0, type=i32)
                if const_expr(isinstance(k_static_val, int)):
                    return arith.constant(D if k_static_val >= ratio else 0, type=i32)
                # Dynamic: (k >= RATIO) ? D : 0  via select
                is_b = fx.Int32(k_static_val) >= ratio
                return is_b.select(fx.Int32(D), fx.Int32(0))

            # ---- Step 6: Phase 1 -- state cache loop (dynamic bound = window_len) ----
            # window_len ? [0, K]. When 0, the loop is a no-op.

            for k_static, state in range(0, _to_raw(window_len), 1, init=init_state):
                m_lane, kv_lane, w_lane = _split_state(state, VEC)

                k_i32 = _to_raw(fx.Int32(k_static))
                s_fx = fx.Int32(position) - (K - 1) + fx.Int32(k_i32)
                is_pad_b = s_fx < 0
                is_pad = is_pad_b.ir_value()
                s_safe = is_pad_b.select(fx.Int32(0), s_fx)
                ring = fx.Uint32(s_safe) % state_size
                # Slot term already folded into the descriptor base.
                col_off_fx = fx.Int32(_col_off_for_k(k_i32))
                base_kv_off = (
                    fx.Int32(ring) * kv_state_pos_stride + col_off_fx + tid_x_vec
                )
                base_sc_off = (
                    fx.Int32(ring) * score_state_pos_stride + col_off_fx + tid_x_vec
                )

                kv_v_lane = _load_f32_vec(kv_state_rsrc, base_kv_off, VEC)
                sc_v_lane = _load_f32_vec(score_state_rsrc, base_sc_off, VEC)

                sc_pad_lane = []
                for i in range_constexpr(VEC):
                    sc_pad_lane.append(arith.select(is_pad, c_neg_inf, sc_v_lane[i]))

                new_m, new_kv, new_w = _online_softmax_update(
                    m_lane, kv_lane, w_lane, sc_pad_lane, kv_v_lane
                )
                final_state = yield (list(new_m) + list(new_kv) + list(new_w))

            phase1_state = final_state

            # ---- Step 7: Phase 2 -- ragged input loop (k ? [window_len, K)) ----
            # No padding in input phase by construction (window_len absorbs all
            # leading state-cache rows). Two code paths:
            #
            #   enable_prefetch_input=False (legacy): straight per-iter load
            #     + compute. Issue and compute are serialized within each iter.
            #
            #   enable_prefetch_input=True (default): manual single-iter
            #     prefetch. Prologue issues k=window_len's loads; each loop
            #     iter consumes the prefetched values and issues k+1's loads
            #     so the issue overlaps current iter's softmax compute. Helps
            #     long K (HCA K=128) latency-bound chains. Loop carry grows
            #     by 3*VEC fp32; gate off if VGPR spill regresses small VEC
            #     configs.

            def _phase2_offsets(k_i32):
                """Compute (col_off, in_row, ape_row) for Phase 2 iter k."""
                col_off = _col_off_for_k(k_i32)
                # k_i32 >= 0 -> unsigned rem.
                ape_row = fx.Uint32(k_i32) % ratio
                in_row = fx.Int32(ragged_id) - (fx.Int32(K - 1) - fx.Int32(k_i32))
                return col_off, in_row, ape_row

            def _phase2_issue_loads(k_i32):
                """Issue kv_in / score_in / ape loads for Phase 2 iter k.

                Returns (kv_lane, score_a_lane, ape_v_lane) -- three lists of
                VEC fp32 scalars. buffer_load with max_size resources is OOB-
                safe (returns 0), so callers may speculatively issue at
                k = K (one past the last legal iter) for prefetch tails.
                """
                col_off, in_row, ape_row = _phase2_offsets(k_i32)
                col_off_fx = fx.Int32(col_off)
                base_in_off = (
                    fx.Int32(in_row) * kv_in_row_stride + col_off_fx + tid_x_vec
                )
                base_sc_off = (
                    fx.Int32(in_row) * score_in_row_stride + col_off_fx + tid_x_vec
                )
                base_ape_off = fx.Int32(ape_row) * DIM_FULL + col_off_fx + tid_x_vec
                kv = _load_bf16_vec_then_f32(kv_in_rsrc, base_in_off, VEC)
                sc = _load_bf16_vec_then_f32(score_in_rsrc, base_sc_off, VEC)
                ape = _load_f32_vec(ape_rsrc, base_ape_off, VEC)
                return kv, sc, ape

            if const_expr(not enable_prefetch_input):
                for k_static, state in range(
                    _to_raw(window_len), K, 1, init=phase1_state
                ):
                    m_lane, kv_lane, w_lane = _split_state(state, VEC)
                    k_i32 = _to_raw(fx.Int32(k_static))
                    kv_a_lane, score_a_lane, ape_v_lane = _phase2_issue_loads(k_i32)
                    score_k_lane = [
                        arith.AddFOp(
                            score_a_lane[i], ape_v_lane[i], fastmath=fm_fast
                        ).result
                        for i in range(VEC)
                    ]
                    new_m, new_kv, new_w = _online_softmax_update(
                        m_lane,
                        kv_lane,
                        w_lane,
                        score_k_lane,
                        kv_a_lane,
                        score_can_be_neg_inf=False,
                    )
                    phase2_state = yield (list(new_m) + list(new_kv) + list(new_w))

                _m_final, kv_final, w_final = _split_state(phase2_state, VEC)
            else:
                # Phase 2 with single-iter prefetch, restructured to avoid a
                # per-iter clamp on the speculative k+1 load.
                #
                # Why the restructure: a naive `for k ? [window_len, K)` body
                # that issues at k+1 OOBs on the last iter (k+1 = K) -- and
                # AMD CDNA's buffer_load(max_size=True) does NOT reliably
                # return 0 for OOB (decode-time real workload faults). The
                # obvious fix `k_next = min(k+1, K-1)` works but costs ~27%
                # on v1 mid-N (an extra ``arith.minsi`` inside the K=128
                # loop body trashes scheduling / VGPR pressure).
                #
                # Restructure: peel the last iter outside the loop.
                #   prologue   : prefetch at min(window_len, K-1) -- clamps the
                #                window_len==K edge case (Phase 2 empty);
                #                otherwise loads the first real Phase 2 iter.
                #   main loop  : k ? [window_len, K-1); k+1 <= K-1 is *always*
                #                in-bounds -> no clamp inside the loop.
                #   tail iter  : k = K-1, consumes prefetched values, issues
                #                no new prefetch. Gated by window_len < K so
                #                that wl==K skips Phase 2 entirely.
                k_prologue = fx.min(fx.Int32(window_len), fx.Int32(K - 1))
                pre_kv0, pre_sc0, pre_ape0 = _phase2_issue_loads(k_prologue)
                init_pf_state = (
                    list(phase1_state) + list(pre_kv0) + list(pre_sc0) + list(pre_ape0)
                )

                loop_final = init_pf_state
                for k_static, state in range(
                    _to_raw(window_len), K - 1, 1, init=init_pf_state
                ):
                    m_lane = list(state[0:VEC])
                    kv_lane = list(state[VEC : 2 * VEC])
                    w_lane = list(state[2 * VEC : 3 * VEC])
                    pre_kv = list(state[3 * VEC : 4 * VEC])
                    pre_sc = list(state[4 * VEC : 5 * VEC])
                    pre_ape = list(state[5 * VEC : 6 * VEC])

                    k_i32 = _to_raw(fx.Int32(k_static))
                    # k+1 ? [window_len+1, K-1]: always in-bounds, no clamp.
                    k_next = fx.Int32(k_i32) + 1
                    nxt_kv, nxt_sc, nxt_ape = _phase2_issue_loads(k_next)

                    score_k_lane = [
                        arith.AddFOp(pre_sc[i], pre_ape[i], fastmath=fm_fast).result
                        for i in range(VEC)
                    ]
                    new_m, new_kv, new_w = _online_softmax_update(
                        m_lane,
                        kv_lane,
                        w_lane,
                        score_k_lane,
                        pre_kv,
                        score_can_be_neg_inf=False,
                    )
                    new_state = (
                        list(new_m)
                        + list(new_kv)
                        + list(new_w)
                        + list(nxt_kv)
                        + list(nxt_sc)
                        + list(nxt_ape)
                    )
                    loop_final = yield new_state

                # Skip the peeled tail when window_len == K: phase 2 is empty,
                # so its speculative prefetch values are invalid. Preserve
                # phase1_state without evaluating the tail update.
                tail_state = list(phase1_state)
                if fx.Int32(window_len) < fx.Int32(K):
                    m_lane_t = list(loop_final[0:VEC])
                    kv_lane_t = list(loop_final[VEC : 2 * VEC])
                    w_lane_t = list(loop_final[2 * VEC : 3 * VEC])
                    pre_kv_t = list(loop_final[3 * VEC : 4 * VEC])
                    pre_sc_t = list(loop_final[4 * VEC : 5 * VEC])
                    pre_ape_t = list(loop_final[5 * VEC : 6 * VEC])
                    score_k_lane_t = [
                        arith.AddFOp(pre_sc_t[i], pre_ape_t[i], fastmath=fm_fast).result
                        for i in range(VEC)
                    ]
                    new_m_t, new_kv_t, new_w_t = _online_softmax_update(
                        m_lane_t,
                        kv_lane_t,
                        w_lane_t,
                        score_k_lane_t,
                        pre_kv_t,
                        score_can_be_neg_inf=False,
                    )
                    tail_state = list(new_m_t) + list(new_kv_t) + list(new_w_t)

                kv_final = [_to_raw(v) for v in tail_state[VEC : 2 * VEC]]
                w_final = [_to_raw(v) for v in tail_state[2 * VEC : 3 * VEC]]

            # ---- Step 8: compressed = kv_acc / w_acc (per-lane) ----
            comp_lane = []
            for i in range_constexpr(VEC):
                rcp_w = rocdl.rcp(f32, w_final[i])
                comp_lane.append(
                    arith.MulFOp(kv_final[i], rcp_w, fastmath=fm_fast).result
                )

            # ---- Step 9: RMSNorm (fp32) -- sum-of-squares across wave ----
            sq_local = arith.constant(0.0, type=f32)
            for i in range_constexpr(VEC):
                sq_local = arith.AddFOp(
                    sq_local,
                    arith.MulFOp(comp_lane[i], comp_lane[i], fastmath=fm_fast).result,
                    fastmath=fm_fast,
                ).result
            sq_full = _wave_reduce_add(sq_local, log2_block, fm_fast)
            var = arith.MulFOp(sq_full, c_inv_D, fastmath=fm_fast).result
            rrms = fmath.rsqrt(
                arith.AddFOp(var, c_eps, fastmath=fm_fast).result, fastmath=fm_fast
            )

            # rms_weight: per-channel; this thread loads VEC values at tid*VEC.
            # Production atom passes bf16 (the param is cast at model load);
            # tests may pass fp32. Constexpr branch picks the right load.
            rmsw_rsrc = buffer_ops.create_buffer_resource(rms_weight, max_size=True)
            rmsw_lane = (
                _load_bf16_vec_then_f32(rmsw_rsrc, tid_x_vec, VEC)
                if rms_weight_is_bf16
                else _load_f32_vec(rmsw_rsrc, tid_x_vec, VEC)
            )

            normed_lane = [
                arith.MulFOp(
                    arith.MulFOp(comp_lane[i], rrms, fastmath=fm_fast).result,
                    rmsw_lane[i],
                    fastmath=fm_fast,
                ).result
                for i in range(VEC)
            ]

            # ---- Step 10: GPT-J RoPE on RD tail ----
            # is_rope = tid >= ROPE_THREAD_LO. RoPE applies only to those threads.
            # position >= 0 (guarded by the sentinel-skip IfOp) -> unsigned div.
            comp_pos_i32 = (fx.Uint32(position) // ratio) * ratio

            # Always compute the rotated/passthrough values per-lane, then
            # store. ROPE-only threads load cos/sin; NOPE threads use the
            # pass-through value. We branch via Python-level `if` because the
            # tid range is static (ROPE_THREAD_LO is constexpr).
            #
            # Always compute the rotated values per-lane, then per-lane
            # select(is_rope, rotated, normed). Avoids a scf.if whose body
            # mutates `out_lane` (the mutated values would not dominate the
            # outer scope -- MLIR verification fails).
            #
            # cos/sin loads for NOPE threads are safe because we clamp the
            # row-relative index to 0 (a valid in-bounds position).
            cos_rsrc = buffer_ops.create_buffer_resource(cos_cache, max_size=True)
            sin_rsrc = buffer_ops.create_buffer_resource(sin_cache, max_size=True)
            cos_row_base = fx.Int32(comp_pos_i32) * (RD // 2)

            is_rope_t = (fx.Int32(tid) >= ROPE_THREAD_LO).ir_value()
            # rope_rel may be negative for NOPE threads; clamp to 0 so the
            # cos/sin load address is in-bounds (the loaded value is unused
            # because is_rope_t = false).
            rope_rel_raw = fx.Int32(tid) - ROPE_THREAD_LO
            rope_rel = fx.max(rope_rel_raw, fx.Int32(0))
            cs_lo = fx.Int32(rope_rel) * PAIRS_PER_THREAD

            if const_expr(PAIRS_PER_THREAD == 1):
                cos_b = buffer_ops.buffer_load(
                    cos_rsrc,
                    cos_row_base + cs_lo,
                    vec_width=1,
                    dtype=T.bf16,
                )
                sin_b = buffer_ops.buffer_load(
                    sin_rsrc,
                    cos_row_base + cs_lo,
                    vec_width=1,
                    dtype=T.bf16,
                )
                cos_vals = [arith.extf(f32, cos_b)]
                sin_vals = [arith.extf(f32, sin_b)]
            else:
                cos_vec = fx.Vector(
                    buffer_ops.buffer_load(
                        cos_rsrc,
                        cos_row_base + cs_lo,
                        vec_width=PAIRS_PER_THREAD,
                        dtype=T.bf16,
                    )
                )
                sin_vec = fx.Vector(
                    buffer_ops.buffer_load(
                        sin_rsrc,
                        cos_row_base + cs_lo,
                        vec_width=PAIRS_PER_THREAD,
                        dtype=T.bf16,
                    )
                )
                cos_vals = [
                    arith.extf(f32, _to_raw(cos_vec[i]))
                    for i in range(PAIRS_PER_THREAD)
                ]
                sin_vals = [
                    arith.extf(f32, _to_raw(sin_vec[i]))
                    for i in range(PAIRS_PER_THREAD)
                ]

            # GPT-J pair rotation per VEC pair, then select rotated vs pass-through.
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

            # ---- Step 11: Scatter (only when has_block_table) ----
            if const_expr(has_block_table):
                # ci = position // ratio; block_in_seq = ci // k_per_block;
                # slot_in_block = ci % k_per_block. position >= 0 -> unsigned.
                ci = fx.Uint32(position) // ratio
                block_in_seq = _to_raw(ci // k_per_block)
                slot_in_block = _to_raw(ci % k_per_block)
                ci = _to_raw(ci)

                # physical_block = block_table[batch_id, block_in_seq]
                bt_rsrc = buffer_ops.create_buffer_resource(block_table, max_size=True)
                bt_off = _to_raw(
                    fx.Int32(batch_id) * block_table_seq_stride + fx.Int32(block_in_seq)
                )
                physical_block = buffer_ops.buffer_load(
                    bt_rsrc, bt_off, vec_width=1, dtype=i32
                )

                if const_expr(not quant):
                    # BF16 paged write. kv_cache layout: [NB, k_per_block, D].
                    # cache_addr = physical_block * block_stride + slot_in_block * token_stride + tid*VEC
                    # (strides are in bf16 elements; caller passes elements.)
                    # The block term rides on the descriptor's base, not on
                    # the 32-bit offset -- see `block_base_bytes_i64`.
                    cache_off = (
                        fx.Int32(slot_in_block) * kv_cache_token_stride + tid_x_vec
                    )
                    # Build a per-block GTensor and store VEC bf16 via dword path.
                    # bf16 VEC ? {2, 4, 8, 16} = {4, 8, 16, 32} bytes = {1, 2, 4, 8} dwords.
                    out_vec_t = T.vec(VEC, T.bf16)
                    bf16_vec = fx.Vector.from_elements(
                        out_lane, dtype=fx.Float32
                    ).truncf(out_vec_t)
                    out_rsrc = buffer_ops.create_buffer_resource(
                        kv_cache,
                        max_size=True,
                        base_byte_offset=block_base_bytes_i64(
                            physical_block, kv_cache_block_stride, 2
                        ),
                    )
                    # cache_off is in bf16 elements; convert to dword for the i32-vec store.
                    cache_off_dw = fx.Uint32(cache_off) >> 1
                    dwords = (VEC + 1) // 2
                    bf16_as_i32 = fx.Vector(bf16_vec).bitcast(fx.Int32)
                    if const_expr(dwords == 1):
                        # vec<1xi32> -> scalar i32 store
                        buffer_ops.buffer_store(
                            _to_raw(bf16_as_i32[0]), out_rsrc, cache_off_dw
                        )
                    elif const_expr(dwords <= 4):
                        buffer_ops.buffer_store(
                            _to_raw(bf16_as_i32), out_rsrc, cache_off_dw
                        )
                    else:
                        # dwords > 4 (VEC=16 -> dwords=8): split into 2x dwordx4.
                        lo = _to_raw(
                            fx.Vector.from_elements(
                                [bf16_as_i32[i] for i in range(4)], dtype=fx.Int32
                            )
                        )
                        hi = _to_raw(
                            fx.Vector.from_elements(
                                [bf16_as_i32[i] for i in range(4, 8)], dtype=fx.Int32
                            )
                        )
                        buffer_ops.buffer_store(lo, out_rsrc, cache_off_dw)
                        buffer_ops.buffer_store(hi, out_rsrc, cache_off_dw + 4)
                elif const_expr(nm_asm):
                    # -- group_fp8 (V4 nm-asm) via shared emitter (wave32; same layout
                    # as wave64 -- single source of truth). --
                    # The block term rides on each descriptor's base, not on
                    # the 32-bit offset -- see `block_base_bytes_i64`.
                    _nm_cache_base = fx.Int32(slot_in_block) * kv_cache_token_stride
                    _nm_krope_base = fx.Int32(slot_in_block) * krope_token_stride
                    emit_group_fp8_nm_asm_scatter(
                        normed_lane=normed_lane,
                        rotated_lane=rotated_lane,
                        lane=tid,
                        is_rope_t=is_rope_t,
                        cache_base=_to_raw(_nm_cache_base),
                        out_base_i64=fx.Int64(fx.ptrtoint(fx.get_iter(kv_cache)))
                        + fx.Int64(
                            block_base_bytes_i64(
                                physical_block, kv_cache_block_stride, 1
                            )
                        ),
                        krope_base=_to_raw(_nm_krope_base),
                        krope_base_i64=fx.Int64(fx.ptrtoint(fx.get_iter(k_rope_buff)))
                        + fx.Int64(
                            block_base_bytes_i64(physical_block, krope_block_stride, 2)
                        ),
                        VEC=VEC,
                        NOPE=NOPE,
                        RTS=RTS,
                        log2_rts=log2_rts,
                        ROPE_THREAD_LO=ROPE_THREAD_LO,
                        wave_width=BLOCK_THREADS,
                    )
                else:
                    # -- QUANT=1: FP8 per-row scaled write + fp32 scale --
                    # Steps:
                    #   (a) per-lane amax over VEC values, wave-reduce-max
                    #   (b) scale = amax / FP8_MAX (with safety floor); for
                    #       use_ue8m0=True, round UP to nearest power-of-2
                    #       (bit trick: (s_i32 + 0x7FFFFF) & 0xFF800000).
                    #   (c) inv_scale = 1.0 / scale (rcp.f32)
                    #   (d) per-lane fp8 cast with clamp + fnuz NaN guard
                    #   (e) pair-coop dword store: even tid combines own 2 fp8
                    #       with peer 2 fp8 (shuffle_xor 1) and stores 4 bytes.
                    #       Supports both PRESHUFFLE (16x16 tile) and linear
                    #       layouts via the offset formula.
                    #   (f) lane-0 writes fp32 scale at cache_scale[phys, slot].
                    #
                    # VEC=2 (Indexer D=128) -> 4 bytes per tid-pair (1 dword).
                    # VEC=8 (would-be D=512 quant; not used in V4-Pro but
                    # supported for symmetry) -> 8 bytes per thread alone (2
                    # dwords); pair cooperation collapses to no-op for VEC>=4
                    # since a single thread already has dword-aligned data.

                    _, fp8_max = _fp8_const()
                    c_fp8_max = arith.constant(fp8_max, type=f32)
                    c_neg_fp8_max = arith.constant(-fp8_max, type=f32)
                    c_safety_floor = arith.constant(1e-4, type=f32)
                    c_inv_fp8_max = arith.constant(1.0 / fp8_max, type=f32)

                    # (a) per-lane amax
                    am_local = arith.constant(0.0, type=f32)
                    for i in range_constexpr(VEC):
                        abs_v = fmath.absf(out_lane[i])
                        am_local = arith.maximumf(am_local, abs_v)
                    amax = _wave_reduce_max(am_local, log2_block)
                    am_safe = arith.maximumf(amax, c_safety_floor)

                    # (b) scale = am_safe / FP8_MAX, optionally ceil-pow2
                    scale_raw = arith.MulFOp(
                        am_safe, c_inv_fp8_max, fastmath=fm_fast
                    ).result
                    if const_expr(use_ue8m0):
                        # ceil-to-pow2 via bit trick: add 0x7FFFFF to mantissa,
                        # mask off mantissa. If mantissa was 0, exp unchanged;
                        # else exp += 1.
                        scale_i32 = scale_raw.bitcast(i32)
                        bits_up = (
                            scale_i32 + arith.constant(0x7FFFFF, type=i32)
                        ) & arith.constant(0xFF800000, type=i32)
                        scale_v = bits_up.bitcast(f32)
                    else:
                        scale_v = scale_raw

                    # (c) inv_scale via rcp.f32
                    inv_scale = rocdl.rcp(f32, scale_v)

                    # (d) per-lane fp8 cast: clamp + NaN guard
                    #     NaN guard: cvt_pk_fp8_f32 on fnuz returns 0x80 (NaN)
                    #     for inputs that round to negative zero. Clamp small
                    #     negatives v ? (-2^-8, 0) to +0 first. Matches
                    #     _store_fp8_packed in qk_norm_rope_quant.
                    c_neg_uf = arith.constant(-(2.0**-8), type=f32)
                    c_zero = arith.constant(0.0, type=f32)
                    fp8_inputs = []
                    for i in range_constexpr(VEC):
                        v = arith.MulFOp(
                            out_lane[i], inv_scale, fastmath=fm_fast
                        ).result
                        # clamp to [-FP8_MAX, +FP8_MAX]
                        v = _to_raw(fx.min(fx.max(v, c_neg_fp8_max), c_fp8_max))
                        # NaN guard
                        is_tn = arith.andi(
                            arith.cmpf(CmpFPredicate.OLT, v, c_zero),
                            arith.cmpf(CmpFPredicate.OGT, v, c_neg_uf),
                        )
                        v_safe = arith.select(is_tn, c_zero, v)
                        fp8_inputs.append(v_safe)

                    # (e) pack VEC fp32 -> VEC fp8 bytes inside i32 seed
                    # VEC=2: 1 cvt_pk_fp8_f32 call (places 2 bytes at index 0)
                    # VEC=4: 2 calls (places 4 bytes at indices 0, 1)
                    # VEC=8: 4 calls (places 8 bytes at indices 0..3 of 2 i32s)
                    c_p0 = arith.constant(0, type=i32)
                    if const_expr(VEC == 2):
                        # Result in low 16 bits of i32
                        pk = rocdl.cvt_pk_fp8_f32(
                            i32, fp8_inputs[0], fp8_inputs[1], c_p0, 0
                        )
                        # Pair cooperation: even tid stores dword with peer.
                        # peer_pack (in low 16 bits) shifted to high 16 bits.
                        peer_pk = fx.Int32(pk).shuffle_xor(1, BLOCK_THREADS)
                        dword = fx.Int32(pk) | (peer_pk << 16)
                    elif const_expr(VEC == 4):
                        # 4 bytes -> single i32, all in one thread. No coop.
                        pk = rocdl.cvt_pk_fp8_f32(
                            i32, fp8_inputs[0], fp8_inputs[1], c_p0, 0
                        )
                        pk = rocdl.cvt_pk_fp8_f32(
                            i32, fp8_inputs[2], fp8_inputs[3], pk, 1
                        )
                        dword = pk
                    else:
                        # VEC in {8, 16}: VEC bytes = VEC/4 dwords. Build per-dword.
                        assert VEC in (8, 16)
                        n_dwords = VEC // 4
                        dword_list = []
                        for dw_idx in range_constexpr(n_dwords):
                            base = dw_idx * 4
                            pk = rocdl.cvt_pk_fp8_f32(
                                i32,
                                fp8_inputs[base + 0],
                                fp8_inputs[base + 1],
                                c_p0,
                                0,
                            )
                            pk = rocdl.cvt_pk_fp8_f32(
                                i32,
                                fp8_inputs[base + 2],
                                fp8_inputs[base + 3],
                                pk,
                                1,
                            )
                            dword_list.append(pk)
                        dword = tuple(dword_list)

                    # Compute store address (in BYTES from kv_cache base).
                    # Both layouts share the same block base, folded into the
                    # descriptor above; only the in-block offset differs.
                    # Block base on the descriptor, not on the 32-bit offset
                    # -- see `block_base_bytes_i64`.
                    out_rsrc = buffer_ops.create_buffer_resource(
                        kv_cache,
                        max_size=True,
                        base_byte_offset=block_base_bytes_i64(
                            physical_block, kv_cache_block_stride, 1
                        ),
                    )

                    if const_expr(preshuffle):
                        # MFMA 16x16 tile layout
                        # offset = token_tile_id * (TILE * D)
                        #        + col_tile_id * (TILE * TILE)
                        #        + token_in_tile * TILE
                        #        + col_in_tile
                        # slot_in_block, d are non-negative -> unsigned div/rem.
                        TILE = _PRESHUFFLE_TILE
                        slot_u = fx.Uint32(slot_in_block)
                        token_tile_id = slot_u // TILE
                        token_in_tile = slot_u % TILE
                        # d = tid * VEC; col_tile_id = d // TILE; col_in_tile = d % TILE
                        d_for_tid = fx.Uint32(tid) * VEC
                        col_tile_id = d_for_tid // TILE
                        col_in_tile = d_for_tid % TILE
                        in_block_off = _to_raw(
                            token_tile_id * (TILE * D)
                            + col_tile_id * (TILE * TILE)
                            + token_in_tile * TILE
                            + col_in_tile
                        )
                    else:
                        # Linear layout: slot * D + tid * VEC
                        in_block_off = _to_raw(
                            fx.Int32(slot_in_block) * D + fx.Int32(tid) * VEC
                        )

                    byte_off = in_block_off

                    if const_expr(VEC == 2):
                        # Only even tid stores (its dword covers peer's bytes too).
                        if (fx.Int32(tid) & 1) == 0:
                            buffer_ops.buffer_store(
                                dword,
                                out_rsrc,
                                byte_off,
                                offset_is_bytes=True,
                            )
                    elif const_expr(VEC == 4):
                        buffer_ops.buffer_store(
                            dword, out_rsrc, byte_off, offset_is_bytes=True
                        )
                    else:
                        # VEC in {8, 16}: store n_dwords via dwordx4 chunks
                        n_dw = VEC // 4
                        if const_expr(n_dw <= 4):
                            store_vec = fx.Vector.from_elements(
                                list(dword), dtype=fx.Int32
                            )
                            buffer_ops.buffer_store(
                                _to_raw(store_vec),
                                out_rsrc,
                                byte_off,
                                offset_is_bytes=True,
                            )
                        else:
                            # n_dw > 4 (VEC=16 -> n_dw=4, actually fits dwordx4;
                            # kept for future-proofing)
                            for chunk_start in range_constexpr(n_dw // 4):
                                base = chunk_start * 4
                                sv = fx.Vector.from_elements(
                                    list(dword[base : base + 4]), dtype=fx.Int32
                                )
                                buffer_ops.buffer_store(
                                    _to_raw(sv),
                                    out_rsrc,
                                    fx.Int32(byte_off) + base * 4,
                                    offset_is_bytes=True,
                                )

                    # (f) lane-0 writes fp32 scale at cache_scale[phys, slot]
                    if fx.Int32(tid) == 0:
                        cs_rsrc = buffer_ops.create_buffer_resource(
                            cache_scale,
                            max_size=True,
                            base_byte_offset=block_base_bytes_i64(
                                physical_block, cache_scale_block_stride, 4
                            ),
                        )
                        buffer_ops.buffer_store(scale_v, cs_rsrc, slot_in_block)
            # else: warmup -- no scatter, just consume compute.

    @flyc.jit
    def launch_fused_compress_attn(
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
        rms_weight: fx.Tensor,
        cos_cache: fx.Tensor,
        sin_cache: fx.Tensor,
        kv_cache: fx.Tensor,
        kv_cache_block_stride: fx.Int32,
        kv_cache_token_stride: fx.Int32,
        cache_scale: fx.Tensor,
        cache_scale_block_stride: fx.Int32,
        k_rope_buff: fx.Tensor,
        krope_block_stride: fx.Int32,
        krope_token_stride: fx.Int32,
        block_table: fx.Tensor,
        block_table_seq_stride: fx.Int32,
        plan_capacity: fx.Int32,
        stream: fx.Stream,
    ):
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
            rms_weight,
            cos_cache,
            sin_cache,
            kv_cache,
            kv_cache_block_stride,
            kv_cache_token_stride,
            cache_scale,
            cache_scale_block_stride,
            k_rope_buff,
            krope_block_stride,
            krope_token_stride,
            block_table,
            block_table_seq_stride,
        )
        k.launch(
            grid=(plan_capacity, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_fused_compress_attn


# ============================================================================
# K-split single-kernel builder (multi-wave LDS reduce)
# ============================================================================
#
# Why this exists: the legacy single-wave kernel above runs ONE wave32 per
# boundary, serializing K iters of online-softmax. PMC on CSA Main (D=512,
# K=8) showed VALU IPC ~0.33 with 53% of cycles in SQ_WAIT_ANY and only 128
# VMEM insts -- i.e. the wave is stalled on the *serial dependency chain*
# (each iter's m/kv/w accumulator + 2x exp2 transcendental per lane), not on
# memory. At decode bs=1-32 each CU holds a single wave, so nothing hides the
# chain latency.
#
# Fix: split K across NW waves in ONE workgroup (block = 32*NW), grid stays
# = plan_capacity (single dispatch -> no extra ~2.2us launch floor). Each wave
# runs K/NW iters; LDS cross-wave online-softmax merges the per-wave
# accumulators; wave 0 then does RMSNorm + GPT-J RoPE + BF16 scatter inline
# (same tail as the legacy kernel). NW sibling waves on one CU hide each
# other's exp2 / dependency latency.
#
# BF16 scatter only (the V4-Pro CSA Main path the user cares about). FP8 /
# quant / preshuffle continue to use the legacy single-wave kernel.


def _build_kernel_ksplit(
    *,
    head_dim: int,
    rope_head_dim: int,
    ratio: int,
    overlap: bool,
    state_size: int,
    k_per_block: int,
    k_split_num_waves: int,
    quant: bool,
    use_ue8m0: bool,
    preshuffle: bool,
    rms_weight_is_bf16: bool,
    rms_eps: float,
    quant_mode: str = "per_row_fp8",
):
    """K-split single-kernel (wave32 / gfx1250): NW-wave LDS-reduced compress +
    norm + rope + scatter (BF16 or FP8).

    Layout:
      - Grid:  (plan_capacity, 1, 1)  -- one workgroup per plan row.
      - Block: 32 * NW threads (NW waves).
      - VEC = D / 32: each lane owns VEC contiguous D-columns. One wave's 32
        lanes cover the full head_dim.
      - K = (2 if overlap else 1) * ratio, split into NW slices of K_PER_WAVE.
      - LDS: 3 fp32 arrays of NW*D (m, kv, w).
    """
    D = head_dim
    RD = rope_head_dim
    NOPE = D - RD
    VEC = D // BLOCK_THREADS
    K = (2 if overlap else 1) * ratio
    DIM_FULL = (2 if overlap else 1) * D
    NW = k_split_num_waves
    BLOCK_TH = BLOCK_THREADS * NW
    K_PER_WAVE = K // NW

    ROPE_THREAD_LO = NOPE // VEC
    PAIRS_PER_THREAD = VEC // 2

    assert D % BLOCK_THREADS == 0, f"D={D} must divide {BLOCK_THREADS}"
    assert VEC in (2, 4, 8, 16), f"VEC={VEC} outside supported set"
    assert NOPE >= 0 and NOPE % VEC == 0
    assert RD > 0 and RD % 2 == 0 and RD % VEC == 0
    assert state_size >= K, f"state_size={state_size} < K={K}"
    assert K % NW == 0, f"K={K} must divide evenly across NW={NW} waves"
    if quant and preshuffle:
        assert D % _PRESHUFFLE_TILE == 0
        assert k_per_block % _PRESHUFFLE_TILE == 0

    # LDS: 3 fp32 arrays, each NW * D entries.
    LDS_ELEMS = NW * D

    @fx.struct
    class SharedStorage:
        lds_m: fx.Array[fx.Float32, LDS_ELEMS, 16]
        lds_kv: fx.Array[fx.Float32, LDS_ELEMS, 16]
        lds_w: fx.Array[fx.Float32, LDS_ELEMS, 16]

    _name_parts = [
        "fused_compress_attn_w32",
        f"D{D}",
        f"RD{RD}",
        f"R{ratio}",
        ("OVL" if overlap else "NOOVL"),
        f"SS{state_size}",
        f"KB{k_per_block}",
        f"KS{NW}",
    ]
    if quant:
        _name_parts.append("Q")
        if use_ue8m0:
            _name_parts.append("ue8m0")
        if preshuffle:
            _name_parts.append("psh")
    if rms_weight_is_bf16:
        _name_parts.append("rmsbf16")
    _name_parts.append("flydsl")
    _kname = "_".join(_name_parts)

    fm_fast = arith.FastMathFlags.fast
    log2_block = int(math.log2(BLOCK_THREADS))

    @flyc.kernel(name=_kname, known_block_size=[BLOCK_TH, 1, 1])
    def kernel(
        kv_in: fx.Tensor,
        kv_in_row_stride: Int32,
        score_in: fx.Tensor,
        score_in_row_stride: Int32,
        plan: fx.Tensor,
        kv_state: fx.Tensor,
        kv_state_slot_stride: Int32,
        kv_state_pos_stride: Int32,
        score_state: fx.Tensor,
        score_state_slot_stride: Int32,
        score_state_pos_stride: Int32,
        state_slot_mapping: fx.Tensor,
        ape: fx.Tensor,
        rms_weight: fx.Tensor,
        cos_cache: fx.Tensor,
        sin_cache: fx.Tensor,
        kv_cache: fx.Tensor,
        kv_cache_block_stride: Int32,
        kv_cache_token_stride: Int32,
        cache_scale: fx.Tensor,  # [NB, k_per_block] f32 (dummy if not quant)
        cache_scale_block_stride: Int32,
        block_table: fx.Tensor,
        block_table_seq_stride: Int32,
    ):
        f32 = T.f32
        i32 = T.i32

        pid = fx.block_idx.x
        tid = fx.thread_idx.x  # 0 .. BLOCK_TH-1

        c_neg_inf = arith.constant(_NEG_INF, type=f32)
        c_zero_f32 = arith.constant(0.0, type=f32)
        c_zero_i32 = arith.constant(0, type=i32)
        c_eps = arith.constant(rms_eps, type=f32)
        c_inv_D = arith.constant(1.0 / D, type=f32)
        c_log2e = arith.constant(_LOG2E, type=f32)

        wid = fx.Uint32(tid) // BLOCK_THREADS  # -> [0, NW)
        lid = fx.Uint32(tid) % BLOCK_THREADS  # -> [0, 32)

        # ---- plan row (single dwordx4) ----
        plan_rsrc = buffer_ops.create_buffer_resource(plan, max_size=True)
        plan_base = _to_raw(fx.Int32(pid) * 4)
        plan_vec = buffer_ops.buffer_load(plan_rsrc, plan_base, vec_width=4, dtype=i32)
        plan_vec = fx.Vector(plan_vec)
        ragged_id = _to_raw(plan_vec[0])
        batch_id = _to_raw(plan_vec[1])
        position = _to_raw(plan_vec[2])
        window_len = _to_raw(plan_vec[3])

        # Sentinel-skip: guard the whole body on position >= 0 (same scf.if
        # the raw IfOp built; a bare `if cond: return` would not early-exit).
        if fx.Int32(position) >= 0:
            slot_map_rsrc = buffer_ops.create_buffer_resource(
                state_slot_mapping, max_size=True
            )
            slot = buffer_ops.buffer_load(
                slot_map_rsrc, batch_id, vec_width=1, dtype=i32
            )

            # This lane owns columns [lid*VEC, lid*VEC+VEC) of head_dim.
            lid_x_vec = fx.Int32(lid) * VEC

            kv_in_rsrc = buffer_ops.create_buffer_resource(kv_in, max_size=True)
            score_in_rsrc = buffer_ops.create_buffer_resource(score_in, max_size=True)
            # Rebased onto this program's slot — see `state_slot_byte_offset`.
            kv_state_rsrc = buffer_ops.create_buffer_resource(
                kv_state,
                max_size=True,
                base_byte_offset=state_slot_byte_offset(slot, kv_state_slot_stride),
            )
            score_state_rsrc = buffer_ops.create_buffer_resource(
                score_state,
                max_size=True,
                base_byte_offset=state_slot_byte_offset(slot, score_state_slot_stride),
            )
            ape_rsrc = buffer_ops.create_buffer_resource(ape, max_size=True)

            def _col_off_for_k(k_i32):
                if const_expr(not overlap):
                    return c_zero_i32
                is_b = fx.Int32(k_i32) >= ratio
                return is_b.select(fx.Int32(D), fx.Int32(0))

            def _softmax_step(m_lane, kv_lane, w_lane, score_lane, kv_v_lane):
                """Padding-aware per-lane online-softmax update. Phase 2 scores
                are finite, so the is-pad branch is dead code there (compiler
                elides)."""
                new_m, new_kv, new_w = [], [], []
                for i in range_constexpr(VEC):
                    m_old = m_lane[i]
                    score = score_lane[i]
                    m_old_f = fx.Float32(m_old)
                    score_f = fx.Float32(score)
                    neg_inf_f = fx.Float32(c_neg_inf)
                    m_new = fx.max(m_old_f, score_f).ir_value()
                    # Keep the -inf sentinel comparisons outside ambient fastmath.
                    with fastmath(None):
                        is_first = m_old_f == neg_inf_f
                    scale_active = _fexp_f32(m_old_f - m_new, c_log2e)
                    scale_v = is_first.select(c_zero_f32, scale_active)
                    wk_active = _fexp_f32(score_f - m_new, c_log2e)
                    with fastmath(None):
                        is_pad = score_f == neg_inf_f
                    w_k = is_pad.select(c_zero_f32, wk_active)
                    new_m.append(m_new)
                    new_kv.append(
                        _to_raw(
                            fx.Float32(kv_lane[i]) * fx.Float32(scale_v)
                            + fx.Float32(w_k) * fx.Float32(kv_v_lane[i])
                        )
                    )
                    new_w.append(
                        _to_raw(
                            fx.Float32(w_lane[i]) * fx.Float32(scale_v)
                            + fx.Float32(w_k)
                        )
                    )
                return new_m, new_kv, new_w

            def _phase1_loads(k_i32):
                s_fx = fx.Int32(position) - (K - 1) + fx.Int32(k_i32)
                is_pad_b = s_fx < 0
                is_pad = is_pad_b.ir_value()
                s_safe = is_pad_b.select(fx.Int32(0), s_fx)
                ring = fx.Uint32(s_safe) % state_size
                col_off_fx = fx.Int32(_col_off_for_k(k_i32))
                # Slot term already folded into the descriptor base.
                base_kv = fx.Int32(ring) * kv_state_pos_stride + col_off_fx + lid_x_vec
                base_sc = (
                    fx.Int32(ring) * score_state_pos_stride + col_off_fx + lid_x_vec
                )
                kv_v = _load_f32_vec(kv_state_rsrc, base_kv, VEC)
                sc_v = _load_f32_vec(score_state_rsrc, base_sc, VEC)
                sc_pad = [arith.select(is_pad, c_neg_inf, sc_v[i]) for i in range(VEC)]
                return kv_v, sc_pad

            def _phase2_loads(k_i32):
                col_off_fx = fx.Int32(_col_off_for_k(k_i32))
                # k_i32 >= 0 -> unsigned rem.
                ape_row = fx.Uint32(k_i32) % ratio
                in_row_raw = fx.Int32(ragged_id) - (fx.Int32(K - 1) - fx.Int32(k_i32))
                in_row = fx.max(in_row_raw, fx.Int32(0))
                base_in = fx.Int32(in_row) * kv_in_row_stride + col_off_fx + lid_x_vec
                base_sc = (
                    fx.Int32(in_row) * score_in_row_stride + col_off_fx + lid_x_vec
                )
                base_ape = fx.Int32(ape_row) * DIM_FULL + col_off_fx + lid_x_vec
                kv = _load_bf16_vec_then_f32(kv_in_rsrc, base_in, VEC)
                sc = _load_bf16_vec_then_f32(score_in_rsrc, base_sc, VEC)
                ape_v = _load_f32_vec(ape_rsrc, base_ape, VEC)
                score = [
                    arith.AddFOp(sc[i], ape_v[i], fastmath=fm_fast).result
                    for i in range(VEC)
                ]
                return kv, score

            # ---- this wave's K range [wid*KPW, (wid+1)*KPW), split at window_len ----
            k_start = fx.Int32(wid) * K_PER_WAVE
            k_end = k_start + K_PER_WAVE
            split = fx.min(fx.max(fx.Int32(window_len), k_start), k_end)

            init_m = [c_neg_inf for _ in range(VEC)]
            init_kv = [c_zero_f32 for _ in range(VEC)]
            init_w = [c_zero_f32 for _ in range(VEC)]
            init_state = init_m + init_kv + init_w

            # Phase 1 sub-loop [k_start, split): state cache.
            p1 = init_state
            for k_static, state in range(
                _to_raw(k_start), _to_raw(split), 1, init=init_state
            ):
                m_lane = list(state[0:VEC])
                kv_lane = list(state[VEC : 2 * VEC])
                w_lane = list(state[2 * VEC : 3 * VEC])
                k_i32 = _to_raw(fx.Int32(k_static))
                kv_v, sc_v = _phase1_loads(k_i32)
                nm, nkv, nw = _softmax_step(m_lane, kv_lane, w_lane, sc_v, kv_v)
                p1 = yield list(nm) + list(nkv) + list(nw)

            # Phase 2 sub-loop [split, k_end): ragged input.
            final = p1
            for k_static, state in range(_to_raw(split), _to_raw(k_end), 1, init=p1):
                m_lane = list(state[0:VEC])
                kv_lane = list(state[VEC : 2 * VEC])
                w_lane = list(state[2 * VEC : 3 * VEC])
                k_i32 = _to_raw(fx.Int32(k_static))
                kv_v, score = _phase2_loads(k_i32)
                nm, nkv, nw = _softmax_step(m_lane, kv_lane, w_lane, score, kv_v)
                final = yield list(nm) + list(nkv) + list(nw)

            m_local = list(final[0:VEC])
            kv_local = list(final[VEC : 2 * VEC])
            w_local = list(final[2 * VEC : 3 * VEC])

            # ---- LDS write: each lane writes VEC entries at wid*D + lid*VEC ----
            lds = fx.SharedAllocator().allocate(SharedStorage).peek()
            lds_m_ptr = lds.lds_m.ptr
            lds_kv_ptr = lds.lds_kv.ptr
            lds_w_ptr = lds.lds_w.ptr
            lds_thread_base = fx.Int32(wid) * D + lid_x_vec
            for i in range_constexpr(VEC):
                idx_i = lds_thread_base + i
                fx.ptr_store(m_local[i], lds_m_ptr + idx_i)
                fx.ptr_store(kv_local[i], lds_kv_ptr + idx_i)
                fx.ptr_store(w_local[i], lds_w_ptr + idx_i)

            gpu.barrier()

            # ---- wave 0: cross-wave reduce + norm + rope + scatter ----
            if fx.Int32(wid) == 0:
                comp_lane = []
                for i in range_constexpr(VEC):
                    lane_off = lid_x_vec + i
                    m_g = fx.Float32(c_neg_inf)
                    m_arr = []
                    for w in range_constexpr(NW):
                        idx_w = lane_off + (w * D)
                        m_w = fx.ptr_load(lds_m_ptr + idx_w)
                        m_arr.append(m_w)
                        m_g = m_g.maximumf(m_w)
                    kv_sum = fx.Float32(0.0)
                    w_sum = fx.Float32(0.0)
                    for w in range_constexpr(NW):
                        idx_w = lane_off + (w * D)
                        kv_w = fx.ptr_load(lds_kv_ptr + idx_w)
                        w_w = fx.ptr_load(lds_w_ptr + idx_w)
                        scale_w = fx.Float32(
                            _fexp_f32(_to_raw(m_arr[w] - m_g), c_log2e)
                        )
                        kv_sum = kv_sum + kv_w * scale_w
                        w_sum = w_sum + w_w * scale_w
                    rcp_w = fx.Float32(rocdl.rcp(f32, _to_raw(w_sum)))
                    comp_lane.append(_to_raw(kv_sum * rcp_w))

                # ---- RMSNorm (wave-reduce sum-of-squares over wave 0) ----
                sq_local = arith.constant(0.0, type=f32)
                for i in range_constexpr(VEC):
                    sq_local = arith.AddFOp(
                        sq_local,
                        arith.MulFOp(
                            comp_lane[i], comp_lane[i], fastmath=fm_fast
                        ).result,
                        fastmath=fm_fast,
                    ).result
                sq_full = _wave_reduce_add(sq_local, log2_block, fm_fast)
                var = arith.MulFOp(sq_full, c_inv_D, fastmath=fm_fast).result
                rrms = fmath.rsqrt(
                    arith.AddFOp(var, c_eps, fastmath=fm_fast).result, fastmath=fm_fast
                )

                rmsw_rsrc = buffer_ops.create_buffer_resource(rms_weight, max_size=True)
                rmsw_lane = (
                    _load_bf16_vec_then_f32(rmsw_rsrc, lid_x_vec, VEC)
                    if rms_weight_is_bf16
                    else _load_f32_vec(rmsw_rsrc, lid_x_vec, VEC)
                )

                normed_lane = [
                    arith.MulFOp(
                        arith.MulFOp(comp_lane[i], rrms, fastmath=fm_fast).result,
                        rmsw_lane[i],
                        fastmath=fm_fast,
                    ).result
                    for i in range(VEC)
                ]

                # ---- GPT-J RoPE on RD tail ----
                # position >= 0 (sentinel-skip guard) -> unsigned div.
                comp_pos_i32 = (fx.Uint32(position) // ratio) * ratio
                cos_rsrc = buffer_ops.create_buffer_resource(cos_cache, max_size=True)
                sin_rsrc = buffer_ops.create_buffer_resource(sin_cache, max_size=True)
                cos_row_base = fx.Int32(comp_pos_i32) * (RD // 2)

                is_rope_t = (fx.Int32(lid) >= ROPE_THREAD_LO).ir_value()
                rope_rel_raw = fx.Int32(lid) - ROPE_THREAD_LO
                rope_rel = fx.max(rope_rel_raw, fx.Int32(0))
                cs_lo = fx.Int32(rope_rel) * PAIRS_PER_THREAD

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
                    cos_vec = fx.Vector(
                        buffer_ops.buffer_load(
                            cos_rsrc,
                            cos_row_base + cs_lo,
                            vec_width=PAIRS_PER_THREAD,
                            dtype=T.bf16,
                        )
                    )
                    sin_vec = fx.Vector(
                        buffer_ops.buffer_load(
                            sin_rsrc,
                            cos_row_base + cs_lo,
                            vec_width=PAIRS_PER_THREAD,
                            dtype=T.bf16,
                        )
                    )
                    cos_vals = [
                        arith.extf(f32, _to_raw(cos_vec[i]))
                        for i in range(PAIRS_PER_THREAD)
                    ]
                    sin_vals = [
                        arith.extf(f32, _to_raw(sin_vec[i]))
                        for i in range(PAIRS_PER_THREAD)
                    ]

                rotated_lane = list(normed_lane)
                for kk in range_constexpr(PAIRS_PER_THREAD):
                    e = normed_lane[2 * kk]
                    o = normed_lane[2 * kk + 1]
                    cc = cos_vals[kk]
                    ss = sin_vals[kk]
                    new_e = arith.subf(
                        arith.MulFOp(e, cc, fastmath=fm_fast).result,
                        arith.MulFOp(o, ss, fastmath=fm_fast).result,
                    )
                    new_o = arith.AddFOp(
                        arith.MulFOp(e, ss, fastmath=fm_fast).result,
                        arith.MulFOp(o, cc, fastmath=fm_fast).result,
                        fastmath=fm_fast,
                    ).result
                    rotated_lane[2 * kk] = new_e
                    rotated_lane[2 * kk + 1] = new_o

                out_lane = [
                    arith.select(is_rope_t, rotated_lane[i], normed_lane[i])
                    for i in range_constexpr(VEC)
                ]

                # ---- paged scatter (BF16 or FP8). Emitted in wave 0; ``lid``
                # (0..63) is the single-wave ``tid`` equivalent. ----
                # position >= 0 (sentinel-skip guard) -> unsigned div/rem.
                ci = fx.Uint32(position) // ratio
                block_in_seq = _to_raw(ci // k_per_block)
                slot_in_block = _to_raw(ci % k_per_block)
                ci = _to_raw(ci)
                bt_rsrc = buffer_ops.create_buffer_resource(block_table, max_size=True)
                bt_off = _to_raw(
                    fx.Int32(batch_id) * block_table_seq_stride + fx.Int32(block_in_seq)
                )
                physical_block = buffer_ops.buffer_load(
                    bt_rsrc, bt_off, vec_width=1, dtype=i32
                )

                if const_expr(not quant):
                    # The block term rides on the descriptor's base, not on
                    # the 32-bit offset -- see `block_base_bytes_i64`.
                    cache_off = (
                        fx.Int32(slot_in_block) * kv_cache_token_stride + lid_x_vec
                    )
                    out_vec_t = T.vec(VEC, T.bf16)
                    bf16_vec = fx.Vector.from_elements(
                        out_lane, dtype=fx.Float32
                    ).truncf(out_vec_t)
                    out_rsrc = buffer_ops.create_buffer_resource(
                        kv_cache,
                        max_size=True,
                        base_byte_offset=block_base_bytes_i64(
                            physical_block, kv_cache_block_stride, 2
                        ),
                    )
                    cache_off_dw = fx.Uint32(cache_off) >> 1
                    dwords = (VEC + 1) // 2
                    bf16_as_i32 = fx.Vector(bf16_vec).bitcast(fx.Int32)
                    if const_expr(dwords == 1):
                        buffer_ops.buffer_store(
                            _to_raw(bf16_as_i32[0]), out_rsrc, cache_off_dw
                        )
                    elif const_expr(dwords <= 4):
                        buffer_ops.buffer_store(
                            _to_raw(bf16_as_i32), out_rsrc, cache_off_dw
                        )
                    else:
                        # dwords > 4 (VEC=16 -> dwords=8): split into 2x dwordx4.
                        lo = _to_raw(
                            fx.Vector.from_elements(
                                [bf16_as_i32[i] for i in range(4)], dtype=fx.Int32
                            )
                        )
                        hi = _to_raw(
                            fx.Vector.from_elements(
                                [bf16_as_i32[i] for i in range(4, 8)], dtype=fx.Int32
                            )
                        )
                        buffer_ops.buffer_store(lo, out_rsrc, cache_off_dw)
                        buffer_ops.buffer_store(hi, out_rsrc, cache_off_dw + 4)
                else:
                    # -- FP8 per-row scaled write + fp32 scale (mirror legacy) --
                    # Wave-reduce-max over wave 0's 64 lanes; pair-coop dword
                    # store via shuffle_xor(1) within the wave.
                    _, fp8_max = _fp8_const()
                    c_fp8_max = arith.constant(fp8_max, type=f32)
                    c_neg_fp8_max = arith.constant(-fp8_max, type=f32)
                    c_safety_floor = arith.constant(1e-4, type=f32)
                    c_inv_fp8_max = arith.constant(1.0 / fp8_max, type=f32)

                    # (a) per-lane amax -> wave-reduce-max
                    am_local = arith.constant(0.0, type=f32)
                    for i in range_constexpr(VEC):
                        abs_v = fmath.absf(out_lane[i])
                        am_local = arith.maximumf(am_local, abs_v)
                    amax = _wave_reduce_max(am_local, log2_block)
                    am_safe = arith.maximumf(amax, c_safety_floor)

                    # (b) scale = am_safe / FP8_MAX, optionally ceil-pow2
                    scale_raw = arith.MulFOp(
                        am_safe, c_inv_fp8_max, fastmath=fm_fast
                    ).result
                    if const_expr(use_ue8m0):
                        scale_i32 = scale_raw.bitcast(i32)
                        bits_up = (
                            scale_i32 + arith.constant(0x7FFFFF, type=i32)
                        ) & arith.constant(0xFF800000, type=i32)
                        scale_v = bits_up.bitcast(f32)
                    else:
                        scale_v = scale_raw

                    # (c) inv_scale via rcp.f32
                    inv_scale = rocdl.rcp(f32, scale_v)

                    # (d) per-lane fp8 cast: clamp + fnuz NaN guard
                    c_neg_uf = arith.constant(-(2.0**-8), type=f32)
                    c_zero = arith.constant(0.0, type=f32)
                    fp8_inputs = []
                    for i in range_constexpr(VEC):
                        v = arith.MulFOp(
                            out_lane[i], inv_scale, fastmath=fm_fast
                        ).result
                        v = _to_raw(fx.min(fx.max(v, c_neg_fp8_max), c_fp8_max))
                        is_tn = arith.andi(
                            arith.cmpf(CmpFPredicate.OLT, v, c_zero),
                            arith.cmpf(CmpFPredicate.OGT, v, c_neg_uf),
                        )
                        v_safe = arith.select(is_tn, c_zero, v)
                        fp8_inputs.append(v_safe)

                    # (e) pack VEC fp32 -> VEC fp8 bytes
                    c_p0 = arith.constant(0, type=i32)
                    if const_expr(VEC == 2):
                        pk = rocdl.cvt_pk_fp8_f32(
                            i32, fp8_inputs[0], fp8_inputs[1], c_p0, 0
                        )
                        peer_pk = fx.Int32(pk).shuffle_xor(1, BLOCK_THREADS)
                        dword = fx.Int32(pk) | (peer_pk << 16)
                    elif const_expr(VEC == 4):
                        pk = rocdl.cvt_pk_fp8_f32(
                            i32, fp8_inputs[0], fp8_inputs[1], c_p0, 0
                        )
                        pk = rocdl.cvt_pk_fp8_f32(
                            i32, fp8_inputs[2], fp8_inputs[3], pk, 1
                        )
                        dword = pk
                    else:
                        # VEC in {8, 16}: VEC bytes = VEC/4 dwords
                        assert VEC in (8, 16)
                        n_dwords = VEC // 4
                        dword_list = []
                        for dw_idx in range_constexpr(n_dwords):
                            base = dw_idx * 4
                            pk = rocdl.cvt_pk_fp8_f32(
                                i32,
                                fp8_inputs[base + 0],
                                fp8_inputs[base + 1],
                                c_p0,
                                0,
                            )
                            pk = rocdl.cvt_pk_fp8_f32(
                                i32,
                                fp8_inputs[base + 2],
                                fp8_inputs[base + 3],
                                pk,
                                1,
                            )
                            dword_list.append(pk)
                        dword = tuple(dword_list)

                    # Block base on the descriptor, not on the 32-bit offset
                    # -- see `block_base_bytes_i64`.
                    out_rsrc = buffer_ops.create_buffer_resource(
                        kv_cache,
                        max_size=True,
                        base_byte_offset=block_base_bytes_i64(
                            physical_block, kv_cache_block_stride, 1
                        ),
                    )

                    if const_expr(preshuffle):
                        # slot_in_block, d are non-negative -> unsigned div/rem.
                        TILE = _PRESHUFFLE_TILE
                        slot_u = fx.Uint32(slot_in_block)
                        token_tile_id = slot_u // TILE
                        token_in_tile = slot_u % TILE
                        d_for_tid = lid * VEC
                        col_tile_id = d_for_tid // TILE
                        col_in_tile = d_for_tid % TILE
                        in_block_off = _to_raw(
                            token_tile_id * (TILE * D)
                            + col_tile_id * (TILE * TILE)
                            + token_in_tile * TILE
                            + col_in_tile
                        )
                    else:
                        in_block_off = _to_raw(
                            fx.Int32(slot_in_block) * D + fx.Int32(lid) * VEC
                        )

                    byte_off = in_block_off

                    if const_expr(VEC == 2):
                        # Only even lid stores (its dword covers peer's bytes too).
                        if (fx.Int32(lid) & 1) == 0:
                            buffer_ops.buffer_store(
                                dword, out_rsrc, byte_off, offset_is_bytes=True
                            )
                    elif const_expr(VEC == 4):
                        buffer_ops.buffer_store(
                            dword, out_rsrc, byte_off, offset_is_bytes=True
                        )
                    else:
                        # VEC in {8, 16}: store n_dwords via dwordx4 chunks
                        n_dw = VEC // 4
                        if const_expr(n_dw <= 4):
                            store_vec = fx.Vector.from_elements(
                                list(dword), dtype=fx.Int32
                            )
                            buffer_ops.buffer_store(
                                _to_raw(store_vec),
                                out_rsrc,
                                byte_off,
                                offset_is_bytes=True,
                            )
                        else:
                            for chunk_start in range_constexpr(n_dw // 4):
                                base = chunk_start * 4
                                sv = fx.Vector.from_elements(
                                    list(dword[base : base + 4]), dtype=fx.Int32
                                )
                                buffer_ops.buffer_store(
                                    _to_raw(sv),
                                    out_rsrc,
                                    fx.Int32(byte_off) + base * 4,
                                    offset_is_bytes=True,
                                )

                    # (f) lane-0 writes fp32 scale at cache_scale[phys, slot]
                    if fx.Int32(lid) == 0:
                        cs_rsrc = buffer_ops.create_buffer_resource(
                            cache_scale,
                            max_size=True,
                            base_byte_offset=block_base_bytes_i64(
                                physical_block, cache_scale_block_stride, 4
                            ),
                        )
                        buffer_ops.buffer_store(scale_v, cs_rsrc, slot_in_block)

    @flyc.jit
    def launch_fused_compress_attn_ksplit(
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
        rms_weight: fx.Tensor,
        cos_cache: fx.Tensor,
        sin_cache: fx.Tensor,
        kv_cache: fx.Tensor,
        kv_cache_block_stride: fx.Int32,
        kv_cache_token_stride: fx.Int32,
        cache_scale: fx.Tensor,
        cache_scale_block_stride: fx.Int32,
        block_table: fx.Tensor,
        block_table_seq_stride: fx.Int32,
        plan_capacity: fx.Int32,
        stream: fx.Stream,
    ):
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
            rms_weight,
            cos_cache,
            sin_cache,
            kv_cache,
            kv_cache_block_stride,
            kv_cache_token_stride,
            cache_scale,
            cache_scale_block_stride,
            block_table,
            block_table_seq_stride,
        )
        k.launch(
            grid=(plan_capacity, 1, 1),
            block=(BLOCK_TH, 1, 1),
            stream=stream,
        )

    return launch_fused_compress_attn_ksplit


# ============================================================================
# FP4 kernel (CSA indexer): one wave per plan row
# ============================================================================
# Same design and the same float operations as the Gluon kernel
# (aiter/ops/triton/_gluon_kernels/gfx1250/fusions/fused_compress_attn.py), so both
# write the same bytes. Lane l holds window tokens 4*(l//16) .. +3 and columns
# 8*(l%16) .. +7: lanes l and l^16 hold the same columns. Every window load starts
# before the first wait, the softmax runs in registers, and one hardware convert
# writes the 8 e2m1 values of a lane.

_FP4_HEAD = 128
_FP4_ROPE = 64
_FP4_RATIO = 4
_FP4_WINDOW = 8
_FP4_ROW = 256  # values in one state, input or ape row: the previous group, then the current one
_FP4_K_PER_BLOCK = 64
_FP4_LOG2E = 0x3FB8AA3B  # f32 log2(e)
_FP4_INV_SIX = 0x3E2AAAAB  # f32 1/6
_FP4_SCALE_FLOOR = 6.0 * 2.0**-126  # an all-zero group gets scale byte 1
_FP4_DPP_ROW_XMASK = 0x160  # DPP control row_xmask:m (0x160 + m): lane l reads lane l^m
# A masked lane reads at byte offset 2**31. The buffer bound is below that, so the
# lane gets 0 and reads no memory. The default bound (2**32 - 1) would let it read.
_FP4_NUM_RECORDS = 0x7FFFFFFE
_FP4_MASKED_OFFSET = 0x80000000


def _fp4_f32(bits):
    return fx.Int32(bits).bitcast(fx.Float32)


def _fp4_exp2(x):
    """2^x as one v_exp_f32."""
    return fx.Float32(rocdl.exp2(T.f32, x.ir_value()))


def _fp4_rcp(x):
    """1/x as one v_rcp_f32. The compiler pads the TRANS hazard."""
    return fx.Float32(rocdl.rcp(T.f32, x.ir_value()))


def _fp4_rsq(x):
    """1/sqrt(x) as one v_rsq_f32."""
    return fx.Float32(rocdl.rsq(T.f32, x.ir_value()))


def _fp4_xor16(v):
    """The value of lane l^16: one v_permlanex16_b32."""
    moved = rocdl.permlanex16(
        T.i32,
        _llvm.mlir_poison(T.i32),
        v.bitcast(fx.Int32).ir_value(),
        fx.Int32(0x76543210).ir_value(),
        fx.Int32(-0x01234568).ir_value(),  # 0xFEDCBA98: lane l reads lane l^16
        True,
        False,
    )
    return fx.Int32(moved).bitcast(fx.Float32)


def _fp4_xor_row(v, mask):
    """The value of lane l^mask, mask in {1, 2, 4, 8}: one DPP move."""
    moved = rocdl.update_dpp(
        T.i32,
        _llvm.mlir_poison(T.i32),
        v.bitcast(fx.Int32).ir_value(),
        _FP4_DPP_ROW_XMASK + mask,
        0xF,
        0xF,
        True,
    )
    return fx.Int32(moved).bitcast(fx.Float32)


def _fp4_buffer(ptr, elems_per_load):
    """Buffer view of ptr for loads of elems_per_load elements at an element offset."""
    view = fx.make_view(ptr, fx.make_layout((0x3FFFFFFF, elems_per_load), (1, 1)))
    return fx.rocdl.make_buffer_tensor(
        view, num_records_bytes=fx.Int64(_FP4_NUM_RECORDS)
    )


def _fp4_load_f32(buf, index):
    """4 f32 at element offset index: one 16-byte buffer load."""
    frag = fx.make_rmem_tensor(4, fx.Float32)
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
    fx.copy(atom, fx.slice(buf, (index, None)), frag)
    v = fx.Vector(frag.load())
    return [v[i] for i in range(4)]


def _fp4_load_bf16(buf, count, index):
    """count (4 or 8) bf16 at element offset index as exact f32: one 8- or 16-byte
    buffer load."""
    frag = fx.make_rmem_tensor(count, fx.BFloat16)
    copy = fx.rocdl.BufferCopy128b() if count == 8 else fx.rocdl.BufferCopy64b()
    fx.copy(fx.make_copy_atom(copy, fx.BFloat16), fx.slice(buf, (index, None)), frag)
    words = fx.Vector(frag.load()).bitcast(fx.Int32)
    low = (words << 16).bitcast(fx.Float32)
    high = (words & fx.Int32(-65536)).bitcast(fx.Float32)
    return [low[i // 2] if i % 2 == 0 else high[i // 2] for i in range(count)]


def _fp4_masked(mask, index, elem_bytes):
    """The element offset for a load, or the masked offset when mask is false."""
    return mask.select(index, fx.Int32(_FP4_MASKED_OFFSET // elem_bytes))


def _fp4_softmax_pool(values, scores):
    """values, scores: [4 tokens][8 columns] of the lane. Returns the 8 pooled values."""
    log2e = _fp4_f32(_FP4_LOG2E)
    pooled = []
    for j in range(8):
        s = [scores[t][j] for t in range(4)]
        v = [values[t][j] for t in range(4)]
        top = fx.maxnumf(fx.maxnumf(fx.maxnumf(s[0], s[1]), s[2]), s[3])
        top = fx.maxnumf(top, _fp4_xor16(top))
        w = [_fp4_exp2((s[t] - top) * log2e) for t in range(4)]
        sum_w = (w[0] + w[1]) + (w[2] + w[3])
        sum_w = sum_w + _fp4_xor16(sum_w)
        sum_wv = fx.fma(v[0], w[0], v[1] * w[1]) + fx.fma(v[2], w[2], v[3] * w[3])
        sum_wv = sum_wv + _fp4_xor16(sum_wv)
        pooled.append(sum_wv * _fp4_rcp(sum_w))
    return pooled


def _fp4_rmsnorm(x, weight, eps):
    """x, weight: the 8 values of the lane. The 16 lanes of a row hold the 128 values."""
    a0 = fx.fma(x[0], x[0], x[2] * x[2])
    a1 = fx.fma(x[1], x[1], x[3] * x[3])
    b0 = fx.fma(x[4], x[4], x[6] * x[6])
    b1 = fx.fma(x[5], x[5], x[7] * x[7])
    total = (a0 + b0) + (a1 + b1)
    for mask in (8, 4, 2, 1):
        total = total + _fp4_xor_row(total, mask)
    inv_rms = _fp4_rsq(fx.fma(total, fx.Float32(1.0 / _FP4_HEAD), fx.Float32(eps)))
    return [(x[j] * inv_rms) * weight[j] for j in range(8)]


def _fp4_interleaved_rope(x, cos, sin, is_rope):
    """Adjacent pairs (x[2k], x[2k+1]) rotate together on the RoPE columns."""
    out = []
    for k in range(4):
        x0, x1 = x[2 * k], x[2 * k + 1]
        r0 = fx.fma(cos[2 * k], x0, sin[2 * k] * (-x1))
        r1 = fx.fma(cos[2 * k + 1], x1, sin[2 * k + 1] * x0)
        out += [is_rope.select(r0, x0), is_rope.select(r1, x1)]
    return out


def _fp4_quant(x):
    """MXFP4 of the 8 values of the lane; 4 adjacent lanes form one 32-value group.

    Returns the 8 e2m1 values packed in an i32 (value 0 in bits 3:0) and the e8m0
    scale byte: ceil(log2(max(amax, 6 * 2^-126) / 6)) + 127.
    """
    amax = fx.absf(x[0])
    for j in range(1, 8):
        amax = fx.maxnumf(amax, fx.absf(x[j]))
    amax = fx.maxnumf(amax, _fp4_xor_row(amax, 2))
    amax = fx.maxnumf(amax, _fp4_xor_row(amax, 1))
    ratio = fx.maxnumf(amax, fx.Float32(_FP4_SCALE_FLOOR)) * _fp4_f32(_FP4_INV_SIX)
    # Adding 0x7FFFFF carries into the exponent unless the mantissa is 0.
    carry = ratio.bitcast(fx.Int32) + 0x7FFFFF
    scale_byte = fx.Int32(fx.Uint32(carry) >> 23) & 0xFF
    # The convert reads only the exponent field of its f32 scale.
    scale = (carry & 0x7F800000).bitcast(fx.Float32)
    vec = fx.Vector.from_elements(x, fx.Float32)
    packed = rocdl.cvt_scalef32_pk8_fp4_f32(T.i32, vec.ir_value(), scale.ir_value())
    return fx.Int32(packed), scale_byte


def _build_kernel_fp4(
    *,
    state_size: int,
    kv_state_slot_stride: int,
    score_state_slot_stride: int,
    rms_weight_is_bf16: bool,
    rms_eps: float,
):
    """CSA indexer FP4 compressor: one wave32 per plan row (head_dim 128, ratio 4,
    overlap). The state slot strides are compile-time constants."""
    HEAD = _FP4_HEAD
    ROPE = _FP4_ROPE
    RATIO = _FP4_RATIO
    WINDOW = _FP4_WINDOW
    ROW = _FP4_ROW
    KPB = _FP4_K_PER_BLOCK
    name = (
        f"fused_compress_attn_w32_D128_RD64_R4_OVL_SS{state_size}_KB64_Q_fp4nat_1w"
        f"{'_rmsbf16' if rms_weight_is_bf16 else ''}_flydsl"
    )

    @flyc.kernel(name=name, known_block_size=[32, 1, 1])
    def fused_compress_attn_fp4_kernel(
        plan: fx.Pointer,
        state_slots: fx.Pointer,
        block_tables: fx.Pointer,
        kv_state: fx.Pointer,
        score_state: fx.Pointer,
        kv_in: fx.Pointer,
        score_in: fx.Pointer,
        ape: fx.Pointer,
        rms_weight: fx.Pointer,
        cos: fx.Pointer,
        sin: fx.Pointer,
        kv_cache: fx.Pointer,
        cache_scale: fx.Pointer,
        num_seqs: fx.Int32,
        stride_block_tables: fx.Int32,
        stride_kv_in: fx.Int32,
        stride_score_in: fx.Int32,
    ):
        # The 13 pointers and 4 ints fill 30 kernarg dwords, which the wave gets
        # preloaded in SGPRs.

        # --- 1. Read the plan row ---
        plan_row = fx.block_idx.x
        lane = fx.thread_idx.x
        # Each plan row is [last_token_row, seq, last_pos, num_from_state].
        row = plan + plan_row * 4
        last_token_row, seq, last_pos, num_from_state = row[0], row[1], row[2], row[3]
        # Row i of a decode plan is usually sequence i, so load that slot together
        # with the plan row. Clamp, because a plan can have more rows than sequences.
        state_slot = state_slots[fx.min(plan_row, num_seqs - 1)]

        # An empty plan row has a negative position. The slot test keeps the slot
        # load next to the plan-row load.
        if (last_pos >= 0) & ((state_slot >= 0) | (seq != plan_row)):
            # A row of another sequence loads its own slot.
            if seq != plan_row:
                state_slot = state_slots[seq]
            entry = last_pos // RATIO
            group = lane // 16  # window tokens 4*group .. 4*group + 3
            col = (lane % 16) * 8  # the first of the 8 columns of the lane
            # The previous group is in the first half of a row, the current group
            # in the second half: tokens 0..3 and 4..7.
            half = group * HEAD + col

            # --- 2. Load the window, the RMSNorm weight and the RoPE values ---
            if const_expr(rms_weight_is_bf16):
                weight = _fp4_load_bf16(_fp4_buffer(rms_weight, 8), 8, col)
            else:
                weight_buf = _fp4_buffer(rms_weight, 4)
                weight = _fp4_load_f32(weight_buf, col) + _fp4_load_f32(
                    weight_buf, col + 4
                )
            # Slots can be far apart: the slot offset is 64-bit.
            slot = fx.Int64(state_slot)
            kv_ring = _fp4_buffer(kv_state + slot * kv_state_slot_stride, 4)
            score_ring = _fp4_buffer(score_state + slot * score_state_slot_stride, 4)
            kv_rows = _fp4_buffer(kv_in, 8)
            score_rows = _fp4_buffer(score_in, 8)
            ape_rows = _fp4_buffer(ape, 4)
            neg_inf = fx.Float32(float("-inf"))
            values, scores = [], []
            for j in range_constexpr(4):
                token = group * 4 + j
                token_pos = last_pos - (WINDOW - 1) + token
                is_real = token_pos >= 0
                from_state = is_real & (token < num_from_state)
                from_input = is_real & (token >= num_from_state)
                ring_start = _fp4_masked(
                    from_state, (token_pos % state_size) * ROW + half, 4
                )
                in_row = last_token_row - (WINDOW - 1) + token
                kv_start = _fp4_masked(from_input, in_row * stride_kv_in + half, 2)
                score_start = _fp4_masked(
                    from_input, in_row * stride_score_in + half, 2
                )
                state_v = _fp4_load_f32(kv_ring, ring_start) + _fp4_load_f32(
                    kv_ring, ring_start + 4
                )
                state_s = _fp4_load_f32(score_ring, ring_start) + _fp4_load_f32(
                    score_ring, ring_start + 4
                )
                input_v = _fp4_load_bf16(kv_rows, 8, kv_start)
                input_s = _fp4_load_bf16(score_rows, 8, score_start)
                # The state ring stores scores with ape added. Add ape to the input
                # scores only. The other tokens have an input score of -inf, so the
                # ape load needs no mask.
                ape_start = j * ROW + half
                ape_v = _fp4_load_f32(ape_rows, ape_start) + _fp4_load_f32(
                    ape_rows, ape_start + 4
                )
                # Each token comes from at most one source. A masked source gives 0,
                # so add the values; its score becomes -inf.
                values.append([state_v[e] + input_v[e] for e in range(8)])
                scores.append(
                    [
                        fx.maxnumf(
                            from_state.select(state_s[e], neg_inf),
                            from_input.select(input_s[e], neg_inf) + ape_v[e],
                        )
                        for e in range(8)
                    ]
                )
            # The cache row uses the RoPE angle of the first position in its group.
            # Columns 2i and 2i + 1 of the RoPE part use table entry i.
            is_rope = col >= HEAD - ROPE
            table_start = _fp4_masked(
                is_rope, entry * RATIO * (ROPE // 2) + (col - (HEAD - ROPE)) // 2, 2
            )
            cos4 = _fp4_load_bf16(_fp4_buffer(cos, 4), 4, table_start)
            sin4 = _fp4_load_bf16(_fp4_buffer(sin, 4), 4, table_start)
            cos8 = [cos4[e // 2] for e in range(8)]
            sin8 = [sin4[e // 2] for e in range(8)]
            # The block is needed only by the stores. Its load waits under the window
            # loads.
            block = block_tables[seq * stride_block_tables + entry // KPB]
            cache_row = fx.Int64(block * KPB + entry % KPB)

            # --- 3. Softmax pool, RMSNorm, RoPE and MXFP4 ---
            pooled = _fp4_softmax_pool(values, scores)
            normed = _fp4_rmsnorm(pooled, weight, rms_eps)
            rotated = _fp4_interleaved_rope(normed, cos8, sin8, is_rope)
            packed, scale_byte = _fp4_quant(rotated)

            # --- 4. Write the cache row ---
            # Lanes l and l^16 write the same bytes, and the 4 lanes of a group the same
            # scale byte. A store from every lane needs no branch.
            row_words = fx.inttoptr(
                fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4),
                fx.ptrtoint(kv_cache + cache_row * (HEAD // 2)),
            )
            (row_words + lane % 16).store(packed)
            group_scale = cache_scale + cache_row * (HEAD // _FP4_GROUP_SIZE)
            (group_scale + (lane % 16) // 4).store(fx.Int8(scale_byte))

    @flyc.jit
    def launch_fused_compress_attn_fp4(
        plan: fx.Pointer,
        state_slots: fx.Pointer,
        block_tables: fx.Pointer,
        kv_state: fx.Pointer,
        score_state: fx.Pointer,
        kv_in: fx.Pointer,
        score_in: fx.Pointer,
        ape: fx.Pointer,
        rms_weight: fx.Pointer,
        cos: fx.Pointer,
        sin: fx.Pointer,
        kv_cache: fx.Pointer,
        cache_scale: fx.Pointer,
        num_seqs: fx.Int32,
        stride_block_tables: fx.Int32,
        stride_kv_in: fx.Int32,
        stride_score_in: fx.Int32,
        plan_capacity: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        # cluster_dims 1,1,1 reads the block index from ttmp9, without the cluster
        # prologue.
        fused_compress_attn_fp4_kernel(
            plan,
            state_slots,
            block_tables,
            kv_state,
            score_state,
            kv_in,
            score_in,
            ape,
            rms_weight,
            cos,
            sin,
            kv_cache,
            cache_scale,
            num_seqs,
            stride_block_tables,
            stride_kv_in,
            stride_score_in,
            value_attrs={"rocdl.cluster_dims": "1,1,1"},
        ).launch(grid=(plan_capacity, 1, 1), block=(32, 1, 1), stream=stream)

    # No fast-math: the float operations must match the Gluon kernel. coexec is the
    # scheduler that Triton uses on gfx1250.
    launch_fused_compress_attn_fp4.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
            "amdgpu-sched-strategy": "coexec",
        },
    }
    return launch_fused_compress_attn_fp4


# ============================================================================
# Cached compile + public API
# ============================================================================


_DEFAULT_COMPILE_HINTS = {
    "waves_per_eu": 8,
    "fast_fp_math": True,
    "unsafe_fp_math": True,
}


def hca_per_n_config_gfx1250(
    plan_capacity: int, num_input_tokens: int | None = None
) -> tuple[int, int]:
    """Return ``(slice_size, k_split_num_waves)`` for gfx1250 HCA.

    Capacity alone does not determine the best config, because prefill and
    decode want opposite things at the same capacity and both occur there.
    Prefill leaves ``window_len == 0`` on all but each sequence's first
    boundary, so nearly the whole K range is the statically unrolled Phase 2,
    and few long-running waves beat many short ones: halving the waves doubles
    the unrolled trip count, which lifts the loads the scheduler keeps in
    flight from 22 to 27 and shrinks the cross-wave LDS reduce. Decode sources
    most of K from the state cache through the runtime-bounded Phase 1 loop,
    which does not unroll, so it wants waves instead. At capacity 144 the
    prefill optimum (64,4) costs decode 47% (16.6 -> 24.4us) and the decode
    optimum (128,16) costs prefill 6%.

    ``num_input_tokens`` (``kv_in.shape[0]``) separates them: a prefill
    boundary streams ~``ratio`` tokens out of ``kv_in`` while a decode boundary
    streams ~1, so tokens-per-boundary measures ~86-114 for prefill and ~1.1-2.8
    for decode over the whole tuned range, and 16 sits in the empty middle.
    Small capacities are the one place the two get close (~16 vs ~8), which is
    why the ``<= 32`` tier is decided before the mode test -- both modes want
    the same config there, so the signal is only consulted where it is
    unambiguous. Both arguments are host-side shapes, so dispatch stays
    CUDAGraph-stable (the plan contents that would name the mode directly live
    on the device). Callers that omit it keep decode-safe behaviour.

    Tuned on hardware (median of 3 perftest runs per point, D=512 ratio=128):

        prefill  capacity   18     24     48     80    144
          32,8              6.68   6.86   7.10   7.89  11.25
          32,16             6.56   6.70   7.27   8.52  12.48
          64,4              8.30   8.43   8.77   8.72  10.04
          64,8              7.30   7.43   7.88   8.31  10.48
          128,16            7.60   7.79   8.14   9.35  10.59

        decode   capacity   20     80    144    216    256   1040
          32,16            10.21  13.50  19.86  27.61  30.27 100.01
          64,4             22.74  23.64  24.39  26.21  27.15  70.92
          64,8             14.53  15.57  16.64  25.92  27.28  73.92
          128,8            15.99  16.32  17.19  18.79  19.70  64.93
          128,16           11.70  12.62  13.22  20.64  21.68  72.43

    ``slice_size`` 256 was 2-3x off the best at every capacity measured and
    512 does not build at all (VEC=16 exceeds what the Phase 1 f32 loader
    handles), so neither is selectable here.

    Folding several boundaries into one block to share their ape loads was
    tried and dropped: ape is boundary-independent so P=2 does cut the compress
    kernel's loads 12%, but it also halves the grid, and at 576 blocks over 256
    CUs the load imbalance costs more than the traffic saves (compress kernel
    7.28 -> 7.90us at capacity 144, and P=2 lost at every capacity measured).
    """
    if plan_capacity <= 32:
        return 32, 16
    is_prefill = num_input_tokens is not None and num_input_tokens >= 16 * plan_capacity
    if is_prefill:
        if plan_capacity <= 96:
            return 32, 8
        return 64, 4
    if plan_capacity <= 160:
        return 128, 16
    return 128, 8


@lru_cache(maxsize=32)
def compile_flydsl_fused_compress_attn_gfx1250(
    *,
    head_dim: int,
    rope_head_dim: int,
    ratio: int,
    overlap: bool,
    state_size: int,
    k_per_block: int,
    has_block_table: bool,
    quant: bool,
    use_ue8m0: bool,
    preshuffle: bool,
    rms_weight_is_bf16: bool,
    rms_eps: float,
    enable_prefetch_input: bool = False,
    quant_mode: str = "per_row_fp8",  # "per_row_fp8" (indexer) | "group_fp8" (CSA/HCA Main, nm-asm) | (future) "fp4"
    quant_group_size: int = 64,
):
    launcher = _build_kernel(
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        ratio=ratio,
        overlap=overlap,
        state_size=state_size,
        k_per_block=k_per_block,
        has_block_table=has_block_table,
        quant=quant,
        use_ue8m0=use_ue8m0,
        preshuffle=preshuffle,
        rms_weight_is_bf16=rms_weight_is_bf16,
        rms_eps=rms_eps,
        enable_prefetch_input=enable_prefetch_input,
        quant_mode=quant_mode,
        quant_group_size=quant_group_size,
    )
    launcher.compile_hints = dict(_DEFAULT_COMPILE_HINTS)
    return launcher


def csa_ksplit_num_waves_gfx1250(plan_capacity: int) -> int:
    """Auto-pick ``k_split_num_waves`` for gfx1250 (wave32).

    Initial values -- need hardware tuning.
    """
    if plan_capacity <= 512:
        return 4
    if plan_capacity <= 1024:
        return 2
    return 1  # legacy single-wave


@lru_cache(maxsize=32)
def compile_flydsl_fused_compress_attn_ksplit_gfx1250(
    *,
    head_dim: int,
    rope_head_dim: int,
    ratio: int,
    overlap: bool,
    state_size: int,
    k_per_block: int,
    k_split_num_waves: int,
    quant: bool,
    use_ue8m0: bool,
    preshuffle: bool,
    rms_weight_is_bf16: bool,
    rms_eps: float,
    quant_mode: str = "per_row_fp8",
):
    launcher = _build_kernel_ksplit(
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        ratio=ratio,
        overlap=overlap,
        state_size=state_size,
        k_per_block=k_per_block,
        k_split_num_waves=k_split_num_waves,
        quant=quant,
        use_ue8m0=use_ue8m0,
        preshuffle=preshuffle,
        rms_weight_is_bf16=rms_weight_is_bf16,
        rms_eps=rms_eps,
        quant_mode=quant_mode,
    )
    launcher.compile_hints = dict(_DEFAULT_COMPILE_HINTS)
    return launcher


@lru_cache(maxsize=32)
def compile_flydsl_fused_compress_attn_fp4_gfx1250(
    *,
    state_size: int,
    kv_state_slot_stride: int,
    score_state_slot_stride: int,
    rms_weight_is_bf16: bool,
    rms_eps: float,
):
    return _build_kernel_fp4(
        state_size=state_size,
        kv_state_slot_stride=kv_state_slot_stride,
        score_state_slot_stride=score_state_slot_stride,
        rms_weight_is_bf16=rms_weight_is_bf16,
        rms_eps=rms_eps,
    )


def _fp4_pointer(t: torch.Tensor, dtype):
    """A 16-byte aligned tensor as a raw pointer kernel argument (8 kernarg bytes)."""
    if t.data_ptr() % 16 != 0:
        raise ValueError("gfx1250 FP4 compressor needs 16-byte aligned tensors")
    return flyc.from_c_void_p(dtype, t.data_ptr(), assumed_align=16)


def _launch_fp4(
    *,
    kv_in,
    score_in,
    kv_state,
    score_state,
    plan_gpu,
    state_slot_mapping,
    ape,
    rms_weight,
    rms_eps,
    cos_2d,
    sin_2d,
    kv_cache,
    cache_scale,
    block_tables,
    stream,
):
    """Checks the layouts that the one-wave FP4 kernel reads and launches it."""
    if kv_in.stride(0) % 8 != 0 or score_in.stride(0) % 8 != 0:
        raise ValueError(
            "gfx1250 FP4 compressor needs kv_in/score_in row strides % 8 == 0"
        )
    for state in (kv_state, score_state):
        if state.stride(1) != state.shape[2] or state.stride(0) % 4 != 0:
            raise ValueError(
                "gfx1250 FP4 compressor needs kv_state/score_state rows of 256 "
                "contiguous values and a slot stride % 4 == 0"
            )
    if not (ape.is_contiguous() and rms_weight.is_contiguous()):
        raise ValueError("gfx1250 FP4 compressor needs contiguous ape and rms_weight")
    if not (plan_gpu.is_contiguous() and state_slot_mapping.is_contiguous()):
        raise ValueError("gfx1250 FP4 compressor needs contiguous plan and slot map")
    if state_slot_mapping.numel() == 0:
        # Without sequences every plan row is empty, but the kernel loads one slot.
        state_slot_mapping = torch.full(
            (1,), -1, dtype=torch.int32, device=state_slot_mapping.device
        )
    rms_weight_is_bf16 = rms_weight.dtype == torch.bfloat16
    launcher = compile_flydsl_fused_compress_attn_fp4_gfx1250(
        state_size=kv_state.shape[1],
        kv_state_slot_stride=kv_state.stride(0),
        score_state_slot_stride=score_state.stride(0),
        rms_weight_is_bf16=rms_weight_is_bf16,
        rms_eps=float(rms_eps),
    )
    if stream is None:
        stream = torch.cuda.current_stream()
    _run_compiled(
        launcher,
        _fp4_pointer(plan_gpu, fx.Int32),
        _fp4_pointer(state_slot_mapping, fx.Int32),
        _fp4_pointer(block_tables, fx.Int32),
        _fp4_pointer(kv_state, fx.Float32),
        _fp4_pointer(score_state, fx.Float32),
        _fp4_pointer(kv_in, fx.BFloat16),
        _fp4_pointer(score_in, fx.BFloat16),
        _fp4_pointer(ape, fx.Float32),
        _fp4_pointer(rms_weight, fx.BFloat16 if rms_weight_is_bf16 else fx.Float32),
        _fp4_pointer(cos_2d, fx.BFloat16),
        _fp4_pointer(sin_2d, fx.BFloat16),
        _fp4_pointer(kv_cache, fx.Int8),
        _fp4_pointer(cache_scale, fx.Int8),
        state_slot_mapping.numel(),
        block_tables.stride(0),
        kv_in.stride(0),
        score_in.stride(0),
        plan_gpu.shape[0],
        Stream(stream),
    )


def flydsl_fused_compress_attn_gfx1250(
    *,
    kv_in: torch.Tensor,  # [num_q_tokens, DIM_FULL] bf16
    score_in: torch.Tensor,  # [num_q_tokens, DIM_FULL] bf16
    kv_state: torch.Tensor,  # [num_slots, STATE_SIZE, DIM_FULL] f32
    score_state: torch.Tensor,  # same
    plan_gpu: torch.Tensor,  # [plan_capacity, 4] i32
    state_slot_mapping: torch.Tensor,  # [bs] i32
    ape: torch.Tensor,  # [ratio, DIM_FULL] f32
    rms_weight: torch.Tensor,  # [head_dim] f32
    rms_eps: float,
    cos_cache: torch.Tensor,  # [max_pos, ..., RD/2] bf16
    sin_cache: torch.Tensor,
    kv_cache: torch.Tensor | None,  # bf16 or fp8; None ? no scatter
    block_tables: torch.Tensor | None,  # [bs, max_blocks_per_seq] i32
    k_per_block: int,
    overlap: bool,
    ratio: int,
    head_dim: int,
    rope_head_dim: int,
    quant: bool = False,
    cache_scale: torch.Tensor | None = None,  # fp32 [NB, k_per_block]
    use_ue8m0: bool = True,
    preshuffle: bool = True,
    k_split_num_waves: int | None = None,
    quant_mode: str = "per_row_fp8",  # per_row_fp8 | group_fp8 | fp4
    k_rope_cache: (
        torch.Tensor | None
    ) = None,  # group_fp8 only: paged [NB, k_per_block, RD] bf16 rope
    stream: torch.cuda.Stream | None = None,
) -> None:
    """gfx1250 (wave32) drop-in for ``flydsl_fused_compress_attn``.

    Forces ``preshuffle=False`` (MFMA preshuffle is gfx9-only; gfx1250 uses
    linear FP8 layout) and ``enable_prefetch_input=False`` (VEC=16 register
    pressure on RDNA4's 512 VGPRs/wave).
    """
    # gfx1250 overrides
    preshuffle = False  # MFMA preshuffle is gfx9-only; force linear layout
    enable_prefetch_input = False  # noqa: F841  # VEC=16 VGPR pressure
    nm_asm = quant_mode == "group_fp8"  # V4 nm-asm group-quant (separate rope buf)
    quant_fp4 = quant_mode == "fp4"
    # ---- input validation ----
    plan_capacity = plan_gpu.shape[0]
    if plan_capacity == 0:
        return

    dim_full = (2 if overlap else 1) * head_dim
    if kv_in.dim() != 2 or kv_in.shape[1] != dim_full:
        raise ValueError(f"kv_in shape {tuple(kv_in.shape)} != [*, {dim_full}]")
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

    state_size = kv_state.shape[1]
    K_pool = (2 if overlap else 1) * ratio
    if state_size < K_pool or kv_state.shape[2] != dim_full:
        raise ValueError(
            f"kv_state {tuple(kv_state.shape)} expected [*, >={K_pool}, {dim_full}]"
        )
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
    if ape.shape != (ratio, dim_full) or ape.dtype != torch.float32:
        raise ValueError(
            f"ape shape {tuple(ape.shape)} dtype {ape.dtype} != ({ratio}, {dim_full}) f32"
        )
    if rms_weight.shape != (head_dim,):
        raise ValueError(f"rms_weight shape {tuple(rms_weight.shape)} != ({head_dim},)")
    if rms_weight.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(f"rms_weight must be fp32 or bf16, got {rms_weight.dtype}")
    _rms_weight_is_bf16 = rms_weight.dtype == torch.bfloat16
    if plan_gpu.dim() != 2 or plan_gpu.shape[1] != 4 or plan_gpu.dtype != torch.int32:
        raise ValueError(
            f"plan_gpu shape {tuple(plan_gpu.shape)} dtype {plan_gpu.dtype}"
            f" != [P, 4] i32"
        )
    if state_slot_mapping.dim() != 1 or state_slot_mapping.dtype != torch.int32:
        raise ValueError("state_slot_mapping must be 1D int32")
    if cos_cache.shape[-1] != rope_head_dim // 2:
        raise ValueError(
            f"cos_cache last dim {cos_cache.shape[-1]} != RD/2 {rope_head_dim // 2}"
        )
    if sin_cache.shape != cos_cache.shape:
        raise ValueError("cos/sin shape mismatch")
    if not (cos_cache.is_contiguous() and sin_cache.is_contiguous()):
        raise ValueError("cos/sin must be contiguous")

    has_bt = block_tables is not None and kv_cache is not None
    if has_bt:
        if kv_cache.dim() != 3:
            raise ValueError(f"kv_cache must be 3D, got {kv_cache.shape}")
        if block_tables.dim() != 2 or block_tables.dtype != torch.int32:
            raise ValueError("block_tables must be 2D int32")
        if not block_tables.is_contiguous():
            raise ValueError("block_tables must be contiguous")
    if quant:
        if not has_bt:
            raise ValueError("quant=True requires block_tables")
        if quant_fp4:
            if head_dim != 128 or k_per_block != 64:
                raise ValueError(
                    "gfx1250 FP4 compressor requires head_dim=128 and k_per_block=64"
                )
            if (
                kv_cache.dtype != torch.uint8
                or tuple(kv_cache.shape[1:]) != (k_per_block, head_dim // 2)
                or kv_cache.stride(-1) != 1
                or kv_cache.stride(0) != k_per_block * (head_dim // 2)
            ):
                raise ValueError(
                    "gfx1250 FP4 kv_cache must be contiguous uint8 "
                    f"[NB, {k_per_block}, {head_dim // 2}], got "
                    f"shape={tuple(kv_cache.shape)} stride={kv_cache.stride()} "
                    f"dtype={kv_cache.dtype}"
                )
            if (
                cache_scale is None
                or cache_scale.dtype != torch.uint8
                or tuple(cache_scale.shape[1:])
                != (k_per_block, head_dim // _FP4_GROUP_SIZE)
                or cache_scale.stride(-1) != 1
                or cache_scale.stride(0) != k_per_block * (head_dim // _FP4_GROUP_SIZE)
            ):
                raise ValueError(
                    "gfx1250 FP4 cache_scale must be contiguous uint8 "
                    f"[NB, {k_per_block}, {head_dim // _FP4_GROUP_SIZE}]"
                )
        else:
            if kv_cache.dtype == torch.bfloat16:
                raise TypeError("quant=True needs fp8/uint8 kv_cache")
            if not nm_asm and (
                cache_scale is None
                or cache_scale.dtype != torch.float32
                or cache_scale.dim() != 2
                or cache_scale.shape[0] != kv_cache.shape[0]
            ):
                raise ValueError(
                    "per-row FP8 requires fp32 [NB, k_per_block] cache_scale"
                )
        if preshuffle:
            if head_dim % _PRESHUFFLE_TILE != 0:
                raise ValueError(f"preshuffle requires head_dim%16==0, got {head_dim}")
            if k_per_block % _PRESHUFFLE_TILE != 0:
                raise ValueError(
                    f"preshuffle requires k_per_block%16==0, got {k_per_block}"
                )

    # cos/sin row stride must equal RD/2 (caller's [max_pos, ..., RD/2] view).
    cos_2d = cos_cache.view(cos_cache.shape[0], rope_head_dim // 2)
    sin_2d = sin_cache.view(sin_cache.shape[0], rope_head_dim // 2)

    # dummy placeholders for unused inputs so the kernel arg binding always has
    # valid tensors (matches qk_norm_rope_quant pattern).
    if has_bt:
        bt_arg = block_tables
        bt_seq_stride = block_tables.stride(0)
        kv_cache_arg = kv_cache
        kv_cache_block_stride = kv_cache.stride(0)
        kv_cache_token_stride = kv_cache.stride(1)
    else:
        bt_arg = state_slot_mapping  # int32 dummy
        bt_seq_stride = 0
        kv_cache_arg = cos_2d  # bf16 dummy
        kv_cache_block_stride = 0
        kv_cache_token_stride = 0

    if quant and not nm_asm:
        cs_arg = cache_scale
        cs_block_stride = cache_scale.stride(0)
    else:
        cs_arg = rms_weight  # fp32 dummy (group_fp8: e8m0 inline, no separate scale)
        cs_block_stride = 0

    # group_fp8: rotated PE bf16 -> separate paged k_rope_cache (V4 nm layout); else dummy.
    if nm_asm:
        if not quant:
            raise ValueError(
                "quant_mode='group_fp8' requires quant=True (fp8 kv_cache)"
            )
        if (
            k_rope_cache is None
            or k_rope_cache.dtype != torch.bfloat16
            or k_rope_cache.dim() != 3
        ):
            raise ValueError(
                "quant_mode='group_fp8' requires bf16 [NB, k_per_block, RD] k_rope_cache"
            )
        krope_arg = k_rope_cache
        krope_block_stride = k_rope_cache.stride(0)
        krope_token_stride = k_rope_cache.stride(1)
    else:
        krope_arg = cos_2d  # bf16 dummy
        krope_block_stride = 0
        krope_token_stride = 0

    # ---- FP4 (CSA indexer): one wave per plan row, for every plan size ----
    if quant_fp4:
        if rope_head_dim != 64 or ratio != 4 or not overlap:
            raise ValueError(
                "gfx1250 FP4 compressor requires rope_head_dim=64, ratio=4 and overlap"
            )
        _launch_fp4(
            kv_in=kv_in,
            score_in=score_in,
            kv_state=kv_state,
            score_state=score_state,
            plan_gpu=plan_gpu,
            state_slot_mapping=state_slot_mapping,
            ape=ape,
            rms_weight=rms_weight,
            rms_eps=rms_eps,
            cos_2d=cos_2d,
            sin_2d=sin_2d,
            kv_cache=kv_cache,
            cache_scale=cache_scale,
            block_tables=block_tables,
            stream=stream,
        )
        return

    # ---- K-split fast path (BF16 + FP8 scatter) ----
    # k_split_num_waves: None ? auto-pick (tuned geometries only); int>1 ?
    # forced NW; 1 ? forced legacy. Auto triggers for the CSA Main (BF16) and
    # CSA Indexer (FP8) shapes the K-split kernel was tuned for; other shapes
    # fall through to the legacy single-wave kernel.
    _is_csa_main = (
        head_dim == 512 and rope_head_dim == 64 and ratio == 4 and overlap and not quant
    )
    _is_csa_indexer = (
        head_dim == 128
        and rope_head_dim == 64
        and ratio == 4
        and overlap
        and quant_mode == "per_row_fp8"
    )
    if k_split_num_waves is None and has_bt and (_is_csa_main or _is_csa_indexer):
        nw_eff = csa_ksplit_num_waves_gfx1250(plan_capacity)
    else:
        nw_eff = k_split_num_waves if k_split_num_waves is not None else 1
    use_ksplit = (
        nw_eff > 1 and has_bt and not nm_asm
    )  # group_fp8: legacy only (for now)
    if use_ksplit:
        k_split_num_waves = nw_eff
        if K_pool % k_split_num_waves != 0:
            raise ValueError(
                f"k_split_num_waves={k_split_num_waves} must divide K={K_pool}"
            )
        ks_launcher = compile_flydsl_fused_compress_attn_ksplit_gfx1250(
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            ratio=ratio,
            overlap=overlap,
            state_size=state_size,
            k_per_block=k_per_block,
            k_split_num_waves=int(k_split_num_waves),
            quant=quant,
            use_ue8m0=use_ue8m0,
            preshuffle=preshuffle,
            rms_weight_is_bf16=_rms_weight_is_bf16,
            rms_eps=float(rms_eps),
            quant_mode=quant_mode,
        )
        if stream is None:
            stream = torch.cuda.current_stream()
        fx_stream = Stream(stream)
        ks_args = (
            kv_in,
            kv_in.stride(0),
            score_in,
            score_in.stride(0),
            plan_gpu,
            kv_state,
            kv_state.stride(0),
            kv_state.stride(1),
            score_state,
            score_state.stride(0),
            score_state.stride(1),
            state_slot_mapping,
            ape,
            rms_weight,
            cos_2d,
            sin_2d,
            kv_cache_arg,
            kv_cache_block_stride,
            kv_cache_token_stride,
            cs_arg,
            cs_block_stride,
            bt_arg,
            bt_seq_stride,
            plan_capacity,
            fx_stream,
        )
        _run_compiled(ks_launcher, *ks_args)
        return

    launcher = compile_flydsl_fused_compress_attn_gfx1250(
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        ratio=ratio,
        overlap=overlap,
        state_size=state_size,
        k_per_block=k_per_block,
        has_block_table=has_bt,
        quant=quant,
        use_ue8m0=use_ue8m0,
        preshuffle=preshuffle,
        rms_weight_is_bf16=_rms_weight_is_bf16,
        rms_eps=float(rms_eps),
        quant_mode=quant_mode,
        quant_group_size=64,
    )

    if stream is None:
        stream = torch.cuda.current_stream()
    fx_stream = Stream(stream)

    args = (
        kv_in,
        kv_in.stride(0),
        score_in,
        score_in.stride(0),
        plan_gpu,
        kv_state,
        kv_state.stride(0),
        kv_state.stride(1),
        score_state,
        score_state.stride(0),
        score_state.stride(1),
        state_slot_mapping,
        ape,
        rms_weight,
        cos_2d,
        sin_2d,
        kv_cache_arg,
        kv_cache_block_stride,
        kv_cache_token_stride,
        cs_arg,
        cs_block_stride,
        krope_arg,
        krope_block_stride,
        krope_token_stride,
        bt_arg,
        bt_seq_stride,
        plan_capacity,
        fx_stream,
    )
    _run_compiled(launcher, *args)
