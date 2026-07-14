# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused Compressor boundary kernel for V4 attention (FlyDSL).

Drop-in replacement for the Triton ``_fused_compress_attn_kernel`` in
``atom/model_ops/v4_kernels/fused_compress.py``. Per-boundary online-softmax
pool over K=2*RATIO source positions, RMSNorm (fp32), GPT-J interleaved RoPE,
optional FP8 (ue8m0 scale + MFMA 16x16 preshuffle) cache scatter.

Two kernel families share this file:
  - Legacy single-wave (``_build_kernel``): 1 wave64 per boundary, K iters
    serialized. Handles all shapes + the FP8/quant/preshuffle family.
  - K-split multi-wave (``_build_kernel_ksplit``, BF16 + FP8 scatter): K split
    across NW waves in one workgroup (block = 64*NW), LDS cross-wave
    online-softmax reduce, single dispatch. Parallelizes the serial softmax
    chain that bottlenecks the latency-bound small-N decode regime. On the
    CSA Main (D=512, BF16) and CSA Indexer (D=128, FP8) shapes it auto-engages
    via ``csa_ksplit_num_waves(plan_capacity)`` and wins ~1.3-1.4x (BF16) /
    ~1.15-1.24x (FP8) at decode bs=1-32; it falls back to legacy at high N
    where CU occupancy already saturates. See
    ``flydsl_fused_compress_attn``'s ``k_split_num_waves`` arg.

Grid: ``(plan_capacity, 1, 1)`` -- one program per packed plan row
``[ragged_id, batch_id, position, window_len]``. Position == -1 rows
sentinel-skip; sentinel-skip is implemented as an scf.IfOp that wraps the
entire body (flydsl ``if cond: return`` does NOT actually early-exit inside
a @flyc.kernel -- same trap as ``qk_norm_rope_quant``'s grid-Y chunk loop).

Per-thread layout (single wave64, BLOCK_THREADS=64):
  - VEC = D / 64 (V4-Pro Main: D=512 -> VEC=8; Indexer: D=128 -> VEC=2)
  - Thread t owns ``D`` elements ``[t*VEC, t*VEC+VEC)`` of the BLOCK_D vector.
  - Online-softmax accumulators ``m_acc/kv_acc/w_acc`` are per-thread fp32
    arrays carried across K iterations via scf.for loop-carried state.

Two-phase K loop (matches Triton perf split):
  Phase 1 (state cache): k ? [0, window_len) -- dynamic bound, scf.for with
    loop-carry. Each iter loads kv_state/score_state at ``s = pos - K + 1 + k``,
    masking padding (s < 0) via ``select(is_padding, NEG_INF, score_b)``.
  Phase 2 (ragged input): k ? [window_len, K) -- dynamic start, constexpr end.
    Each iter loads kv_in/score_in/ape, then ``score_k = score_a + ape_v``.

Both phases share the max-rescale update:
  m_new = max(m_acc, score_k)
  scale = (m_acc == -inf) ? 0 : exp(m_acc - m_new)
  w_k   = (score_k == -inf) ? 0 : exp(score_k - m_new)
  kv_acc = kv_acc * scale + w_k * kv_k
  w_acc  = w_acc  * scale + w_k
  m_acc  = m_new

After the K loop:
  compressed = kv_acc / w_acc                              # fp32 per-lane
  rstd       = rsqrt(sum_wave(compressed^2) / D + eps)     # fp32 scalar
  normed     = compressed * rstd * rms_weight              # fp32 per-lane
  rotated    = GPT-J RoPE on rope_head_dim tail            # fp32 per-lane

Scatter (only when block_table is non-null, i.e. not warmup):
  QUANT=0: bf16 paged write to kv_cache[physical_block, slot_in_block, :]
  QUANT=1: per-row amax -> ue8m0 scale (silu_and_mul_fq encoding) -> fp8 cast
           -> MFMA 16x16 preshuffled write + fp32 scale write into cache_scale.

Correctness invariant (caller-side): kernel reads state cache as-of-end-of-
previous-fwd. Caller MUST invoke BEFORE ``update_compressor_states``.
"""

# NOTE: do NOT add `from __future__ import annotations` (see qk_norm_rope_quant
# header note -- PEP 563 breaks flydsl's runtime/constexpr param detection,
# triggering a JIT recompile per dynamic-arg value).

import math
from contextlib import contextmanager
from functools import lru_cache
from typing import Optional

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr, vector, buffer_ops
from flydsl.expr import math as fmath
from flydsl.expr.arith import ArithValue, CmpFPredicate, CmpIPredicate
from flydsl.expr.typing import T, Int32, Stream
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, rocdl, scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from .tensor_shim import STensor, _to_raw, _run_compiled

# Shared FP8 group_fp8 (V4 nm-asm) scatter emitter (single source of truth across
# the CSA single-kernel + HCA 2-kernel paths). See fused_compress_attn_common.
from .fused_compress_attn_common import emit_group_fp8_nm_asm_scatter

# Force-bind LDS-related imports so isort/ruff/format hooks don't drop them
# (the K-split LDS path references these only inside @flyc.kernel / @flyc.jit
# closures, which formatters may not see).
_FORCE_BIND_LDS = (
    CompilationContext,
    STensor,
    SmemAllocator,
    SmemPtr,
    get_rocm_arch,
    gpu,
)

# --- shape constants --------------------------------------------------------
BLOCK_THREADS = 64  # 1 wave64; D must be a multiple

# --- fp8 + e8m0 constants ---------------------------------------------------
# Defer ``aiter.utility.dtypes`` import to first call (matches
# qk_norm_rope_quant pattern). The aiter package is walked by setup.py's AOT
# compile pass while its top-level ``__init__`` is still executing, and
# ``aiter.utility.dtypes`` transitively triggers a JIT call into
# ``module_aiter_core`` (not yet built at that point). Resolving the dtype
# constants lazily sidesteps both ordering hazards.
_E8M0_HEADROOM = 7  # silu_and_mul_fq / qk_norm_rope_quant convention


@lru_cache(maxsize=1)
def _fp8_const():
    from aiter.utility import dtypes as aiter_dtypes

    fp8_dtype = aiter_dtypes.fp8
    fp8_max = float(torch.finfo(fp8_dtype).max)
    return fp8_dtype, fp8_max


# --- math constants ---------------------------------------------------------
_NEG_INF = float("-inf")
_LOG2E = math.log2(math.e)  # exp(x) = exp2(x * log2e) -> single v_exp_f32

# Preshuffle MFMA tile (gfx9/gfx94/gfx95 16x16 layout used by aiter scaled GEMM).
_PRESHUFFLE_TILE = 16


# ============================================================================
# scf helpers (copied verbatim from moe_gemm_2stage.py -- too small to share)
# ============================================================================


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


# ============================================================================
# Kernel builder
# ============================================================================


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
    enable_prefetch_input: bool = True,
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
    # For Main (D=512, VEC=8) -> 56 .. 63 are rope threads, 4 pairs each (=64 total).
    # For Indexer (D=128, VEC=2) -> 32 .. 63 are rope threads, 1 pair each (=64=2RD/2).
    # The RD%(2*VEC) == 0 invariant means rope threads cleanly own whole pairs.

    # FP8 1xG e8m0 group-quant geometry (quant_mode=="nm_asm" only): nope region split
    # into N_GROUPS groups of quant_group_size; RTS lanes per group cooperate on amax via
    # shuffle_xor. Byte-identical to HCA Kernel B / C++ k_wave fp8.
    nm_asm = quant_mode == "group_fp8"
    GROUP_SIZE_Q = quant_group_size
    RTS = (GROUP_SIZE_Q // VEC) if nm_asm else 1  # threads/group (=8 for G=64,VEC=8)
    log2_rts = int(math.log2(RTS)) if nm_asm else 0
    if nm_asm:
        assert quant and not preshuffle, "nm_asm: requires quant=True, preshuffle=False"
        assert (
            NOPE % GROUP_SIZE_Q == 0 and GROUP_SIZE_Q % VEC == 0
        ), f"nm_asm: NOPE={NOPE} % G={GROUP_SIZE_Q} and G % VEC={VEC} must be 0"

    assert D % BLOCK_THREADS == 0, f"D={D} must divide BLOCK_THREADS={BLOCK_THREADS}"
    assert VEC in (2, 4, 8), f"VEC={VEC} (D/{BLOCK_THREADS}) outside supported set"
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
        "fused_compress_attn",
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
        cache_scale: fx.Tensor,  # [NB, k_per_block] f32 (dummy if not quant / nm_asm)
        cache_scale_block_stride: Int32,
        k_rope_buff: fx.Tensor,  # nm_asm fp8 only: paged [NB, k_per_block, RD] bf16 rope (dummy otherwise)
        krope_block_stride: Int32,
        krope_token_stride: Int32,
        block_table: fx.Tensor,  # [bs, max_blocks_per_seq] i32 (dummy if not has_bt)
        block_table_seq_stride: Int32,
    ):
        f32 = T.f32
        i32 = T.i32
        vecVf32 = T.vec(VEC, T.f32)

        # --- thread / block ids ---
        pid = fx.block_idx.x  # one program per plan row
        tid = fx.thread_idx.x  # 0 .. 63

        # --- constants ---
        c_neg_inf = arith.constant(_NEG_INF, type=f32)
        c_zero_f32 = arith.constant(0.0, type=f32)
        c_eps = arith.constant(rms_eps, type=f32)
        c_inv_D = arith.constant(1.0 / D, type=f32)
        c_log2e = arith.constant(_LOG2E, type=f32)

        def fexp_f32(x):
            """exp(x) via exp2(x * log2e). Single v_exp_f32 on AMD."""
            return llvm.call_intrinsic(
                f32, "llvm.amdgcn.exp2.f32", [x * c_log2e], [], []
            )

        def wave_reduce_add(x):
            """Butterfly sum across wave64."""
            w = _to_raw(x)
            for sh_exp in range_constexpr(log2_block):
                off = BLOCK_THREADS // (2 << sh_exp)
                peer = _to_raw(ArithValue(w).shuffle_xor(off, BLOCK_THREADS))
                w = arith.AddFOp(w, peer, fastmath=fm_fast).result
            return w

        def wave_reduce_max(x):
            """Butterfly max across wave64 (used by quant path)."""
            w = _to_raw(x)
            for sh_exp in range_constexpr(log2_block):
                off = BLOCK_THREADS // (2 << sh_exp)
                peer = _to_raw(ArithValue(w).shuffle_xor(off, BLOCK_THREADS))
                w = arith.maximumf(w, peer)
            return w

        # ---- Step 1: load plan row (single dwordx4) ----
        # plan layout: each row = 4 contiguous i32 [ragged_id, batch_id, position, window_len].
        # Fuse the 4 scalar loads into one buffer_load_dwordx4 + 4 extracts --
        # saves 3 buffer-load instructions per program (visible at small N
        # where total program count is low).
        plan_rsrc = buffer_ops.create_buffer_resource(plan, max_size=True)
        plan_base = ArithValue(pid) * arith.constant(4, type=i32)
        plan_vec = buffer_ops.buffer_load(plan_rsrc, plan_base, vec_width=4, dtype=i32)
        ragged_id = vector.extract(plan_vec, static_position=[0], dynamic_position=[])
        batch_id = vector.extract(plan_vec, static_position=[1], dynamic_position=[])
        position = vector.extract(plan_vec, static_position=[2], dynamic_position=[])
        window_len = vector.extract(plan_vec, static_position=[3], dynamic_position=[])

        # ---- Step 2: sentinel-skip ----
        # Wrap the entire body in scf.IfOp(position >= 0). flydsl's
        # `if cond: return` does NOT actually early-exit (tail kernel body
        # still runs with stale values, OOB faults). The IfOp does.
        is_active = arith.cmpi(
            CmpIPredicate.sge, _to_raw(position), arith.constant(0, type=i32)
        )
        _if_active = scf.IfOp(is_active)
        with _if_then(_if_active):
            # ---- Step 3: per-seq state slot ----
            slot_map_rsrc = buffer_ops.create_buffer_resource(
                state_slot_mapping, max_size=True
            )
            slot = buffer_ops.buffer_load(
                slot_map_rsrc, batch_id, vec_width=1, dtype=i32
            )

            # ---- Step 4: per-thread element-range bookkeeping ----
            # This thread owns columns [tid*VEC, tid*VEC+VEC) of BLOCK_D.
            tid_x_vec = ArithValue(tid) * arith.constant(VEC, type=i32)

            # ---- Step 5: online-softmax accumulator init ----
            # 3 * VEC fp32 scalars carried across K iters.
            init_m = [c_neg_inf for _ in range(VEC)]
            init_kv = [c_zero_f32 for _ in range(VEC)]
            init_w = [c_zero_f32 for _ in range(VEC)]
            init_state = list(init_m) + list(init_kv) + list(init_w)

            def _split_state(state):
                m_lane = list(state[:VEC])
                kv_lane = list(state[VEC : 2 * VEC])
                w_lane = list(state[2 * VEC : 3 * VEC])
                return m_lane, kv_lane, w_lane

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

                    m_new = arith.maximumf(m_old, score)
                    is_first = arith.cmpf(CmpFPredicate.OEQ, m_old, c_neg_inf)
                    scale_active = fexp_f32(arith.subf(m_old, m_new))
                    scale_v = arith.select(is_first, c_zero_f32, scale_active)
                    wk_active = fexp_f32(arith.subf(score, m_new))
                    if const_expr(score_can_be_neg_inf):
                        is_pad_score = arith.cmpf(CmpFPredicate.OEQ, score, c_neg_inf)
                        w_k = arith.select(is_pad_score, c_zero_f32, wk_active)
                    else:
                        w_k = wk_active
                    new_m.append(m_new)
                    new_kv.append(
                        arith.AddFOp(
                            arith.MulFOp(kv_old, scale_v, fastmath=fm_fast).result,
                            arith.MulFOp(w_k, kv_v, fastmath=fm_fast).result,
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
                return new_m, new_kv, new_w

            def _load_bf16_vec_then_f32(rsrc, off_elems_i32):
                """Load VEC bf16 from byte-aligned dword stream -> fp32 VEC scalars.

                Returns a list of VEC fp32 MLIR values.
                """
                off_dw = ArithValue(off_elems_i32) >> arith.constant(1, type=i32)
                # bf16 VEC = VEC * 2 bytes; for VEC ? {2, 4, 8} that's
                # {4, 8, 16} bytes = {1, 2, 4} dwords.
                dwords = (VEC + 1) // 2  # ceil(VEC*2 / 4)
                if const_expr(dwords == 1):
                    # buffer_load(vec_width=1) returns a scalar i32; wrap into
                    # vec<1xi32> before bitcasting to vec<2xbf16>.
                    raw_s = buffer_ops.buffer_load(rsrc, off_dw, vec_width=1, dtype=i32)
                    raw = vector.from_elements(T.vec(1, T.i32), [raw_s])
                else:
                    raw = buffer_ops.buffer_load(
                        rsrc, off_dw, vec_width=dwords, dtype=i32
                    )
                vec_bf16 = vector.bitcast(T.vec(VEC, T.bf16), raw)
                out = []
                for i in range_constexpr(VEC):
                    bf16_v = vector.extract(
                        vec_bf16,
                        static_position=[i],
                        dynamic_position=[],
                    )
                    f32_v = arith.extf(f32, bf16_v)
                    out.append(f32_v)
                return out

            def _load_f32_vec(rsrc, off_elems_i32):
                """Load VEC fp32 from byte-aligned stream -> list of VEC fp32 scalars.

                For VEC=2 -> dwordx2; VEC=4 -> dwordx4; VEC=8 -> 2x dwordx4 (HW max).
                """
                if const_expr(VEC <= 4):
                    vw = VEC
                    raw = buffer_ops.buffer_load(
                        rsrc, off_elems_i32, vec_width=vw, dtype=f32
                    )
                    return [
                        vector.extract(raw, static_position=[i], dynamic_position=[])
                        for i in range(VEC)
                    ]
                else:
                    # VEC == 8 -> 2x dwordx4
                    assert VEC == 8
                    half = VEC // 2
                    r0 = buffer_ops.buffer_load(
                        rsrc, off_elems_i32, vec_width=half, dtype=f32
                    )
                    r1 = buffer_ops.buffer_load(
                        rsrc,
                        ArithValue(off_elems_i32) + arith.constant(half, type=i32),
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

            # Buffer resources reused across K iters.
            kv_in_rsrc = buffer_ops.create_buffer_resource(kv_in, max_size=True)
            score_in_rsrc = buffer_ops.create_buffer_resource(score_in, max_size=True)
            kv_state_rsrc = buffer_ops.create_buffer_resource(kv_state, max_size=True)
            score_state_rsrc = buffer_ops.create_buffer_resource(
                score_state, max_size=True
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
                is_b = arith.cmpi(
                    CmpIPredicate.sge,
                    k_static_val,
                    arith.constant(ratio, type=i32),
                )
                return arith.select(
                    is_b,
                    arith.constant(D, type=i32),
                    arith.constant(0, type=i32),
                )

            # ---- Step 6: Phase 1 -- state cache loop (dynamic bound = window_len) ----
            # window_len ? [0, K]. When 0, the loop is a no-op.
            c_K_m1 = arith.constant(K - 1, type=i32)
            c_state_size = arith.constant(state_size, type=i32)

            for k_static, state in range(0, _to_raw(window_len), 1, init=init_state):
                m_lane, kv_lane, w_lane = _split_state(state)

                k_i32 = arith.index_cast(i32, _to_raw(k_static))
                s = arith.subi(
                    arith.addi(
                        arith.subi(_to_raw(position), c_K_m1),
                        k_i32,
                    ),
                    arith.constant(0, type=i32),
                )
                is_pad = arith.cmpi(CmpIPredicate.slt, s, arith.constant(0, type=i32))
                s_safe = arith.select(is_pad, arith.constant(0, type=i32), s)
                ring = arith.remui(s_safe, c_state_size)
                col_off = _col_off_for_k(k_i32)

                base_kv_off = (
                    ArithValue(slot) * ArithValue(kv_state_slot_stride)
                    + ArithValue(ring) * ArithValue(kv_state_pos_stride)
                    + ArithValue(col_off)
                    + tid_x_vec
                )
                base_sc_off = (
                    ArithValue(slot) * ArithValue(score_state_slot_stride)
                    + ArithValue(ring) * ArithValue(score_state_pos_stride)
                    + ArithValue(col_off)
                    + tid_x_vec
                )

                kv_v_lane = _load_f32_vec(kv_state_rsrc, base_kv_off)
                sc_v_lane = _load_f32_vec(score_state_rsrc, base_sc_off)

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
                ape_row = arith.remui(k_i32, arith.constant(ratio, type=i32))
                tmp = arith.subi(c_K_m1, k_i32)
                in_row = arith.subi(_to_raw(ragged_id), tmp)
                return col_off, in_row, ape_row

            def _phase2_issue_loads(k_i32):
                """Issue kv_in / score_in / ape loads for Phase 2 iter k.

                Returns (kv_lane, score_a_lane, ape_v_lane) -- three lists of
                VEC fp32 scalars. buffer_load with max_size resources is OOB-
                safe (returns 0), so callers may speculatively issue at
                k = K (one past the last legal iter) for prefetch tails.
                """
                col_off, in_row, ape_row = _phase2_offsets(k_i32)
                base_in_off = (
                    ArithValue(in_row) * ArithValue(kv_in_row_stride)
                    + ArithValue(col_off)
                    + tid_x_vec
                )
                base_sc_off = (
                    ArithValue(in_row) * ArithValue(score_in_row_stride)
                    + ArithValue(col_off)
                    + tid_x_vec
                )
                base_ape_off = (
                    ArithValue(ape_row) * arith.constant(DIM_FULL, type=i32)
                    + ArithValue(col_off)
                    + tid_x_vec
                )
                kv = _load_bf16_vec_then_f32(kv_in_rsrc, base_in_off)
                sc = _load_bf16_vec_then_f32(score_in_rsrc, base_sc_off)
                ape = _load_f32_vec(ape_rsrc, base_ape_off)
                return kv, sc, ape

            if const_expr(not enable_prefetch_input):
                for k_static, state in range(
                    _to_raw(window_len), K, 1, init=phase1_state
                ):
                    m_lane, kv_lane, w_lane = _split_state(state)
                    k_i32 = arith.index_cast(i32, _to_raw(k_static))
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

                m_final, kv_final, w_final = _split_state(phase2_state)
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
                c_K_m1_i32 = arith.constant(K - 1, type=i32)
                k_prologue = arith.minsi(_to_raw(window_len), c_K_m1_i32)
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

                    k_i32 = arith.index_cast(i32, _to_raw(k_static))
                    # k+1 ? [window_len+1, K-1]: always in-bounds, no clamp.
                    k_next = arith.addi(k_i32, arith.constant(1, type=i32))
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

                # Tail iter at k=K-1. Gated by `window_len < K`: when wl==K
                # Phase 2 is empty and the IfOp returns phase1_state.
                is_phase2_nonempty = arith.cmpi(
                    CmpIPredicate.slt,
                    _to_raw(window_len),
                    arith.constant(K, type=i32),
                )
                tail_result_types = [f32] * (3 * VEC)
                _if_tail = scf.IfOp(
                    is_phase2_nonempty, tail_result_types, has_else=True
                )
                with ir.InsertionPoint(_if_tail.then_block):
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
                    scf.YieldOp(list(new_m_t) + list(new_kv_t) + list(new_w_t))
                with ir.InsertionPoint(_if_tail.else_block):
                    # Phase 2 empty (window_len == K): pass through phase1
                    # accumulator unchanged.
                    m_p1 = list(phase1_state[0:VEC])
                    kv_p1 = list(phase1_state[VEC : 2 * VEC])
                    w_p1 = list(phase1_state[2 * VEC : 3 * VEC])
                    scf.YieldOp(list(m_p1) + list(kv_p1) + list(w_p1))

                kv_final = list(_if_tail.results[VEC : 2 * VEC])
                w_final = list(_if_tail.results[2 * VEC : 3 * VEC])

            # ---- Step 8: compressed = kv_acc / w_acc (per-lane) ----
            comp_lane = []
            for i in range_constexpr(VEC):
                rcp_w = llvm.call_intrinsic(
                    f32, "llvm.amdgcn.rcp.f32", [w_final[i]], [], []
                )
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
            sq_full = wave_reduce_add(sq_local)
            var = arith.MulFOp(sq_full, c_inv_D, fastmath=fm_fast).result
            rrms = fmath.rsqrt(
                arith.AddFOp(var, c_eps, fastmath=fm_fast).result, fastmath=fm_fast
            )

            # rms_weight: per-channel; this thread loads VEC values at tid*VEC.
            # Production atom passes bf16 (the param is cast at model load);
            # tests may pass fp32. Constexpr branch picks the right load.
            rmsw_rsrc = buffer_ops.create_buffer_resource(rms_weight, max_size=True)
            if const_expr(rms_weight_is_bf16):
                rmsw_lane = _load_bf16_vec_then_f32(rmsw_rsrc, tid_x_vec)
            else:
                rmsw_lane = _load_f32_vec(rmsw_rsrc, tid_x_vec)

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
            comp_pos_i32 = arith.muli(
                arith.divsi(_to_raw(position), arith.constant(ratio, type=i32)),
                arith.constant(ratio, type=i32),
            )

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
            c_half_rd = arith.constant(RD // 2, type=i32)
            cos_row_base = ArithValue(comp_pos_i32) * c_half_rd

            is_rope_t = arith.cmpi(
                CmpIPredicate.sge,
                _to_raw(tid),
                arith.constant(ROPE_THREAD_LO, type=i32),
            )
            # rope_rel may be negative for NOPE threads; clamp to 0 so the
            # cos/sin load address is in-bounds (the loaded value is unused
            # because is_rope_t = false).
            rope_rel_raw = ArithValue(tid) - arith.constant(ROPE_THREAD_LO, type=i32)
            rope_rel = arith.maxsi(rope_rel_raw, arith.constant(0, type=i32))
            cs_lo = ArithValue(rope_rel) * arith.constant(PAIRS_PER_THREAD, type=i32)

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
                # slot_in_block = ci % k_per_block.
                ci = arith.divsi(_to_raw(position), arith.constant(ratio, type=i32))
                block_in_seq = arith.divsi(ci, arith.constant(k_per_block, type=i32))
                slot_in_block = arith.remui(ci, arith.constant(k_per_block, type=i32))

                # physical_block = block_table[batch_id, block_in_seq]
                bt_rsrc = buffer_ops.create_buffer_resource(block_table, max_size=True)
                bt_off = ArithValue(batch_id) * ArithValue(
                    block_table_seq_stride
                ) + ArithValue(block_in_seq)
                physical_block = buffer_ops.buffer_load(
                    bt_rsrc, bt_off, vec_width=1, dtype=i32
                )

                if const_expr(not quant):
                    # BF16 paged write. kv_cache layout: [NB, k_per_block, D].
                    # cache_addr = physical_block * block_stride + slot_in_block * token_stride + tid*VEC
                    # (strides are in bf16 elements; caller passes elements.)
                    cache_off = (
                        ArithValue(physical_block) * ArithValue(kv_cache_block_stride)
                        + ArithValue(slot_in_block) * ArithValue(kv_cache_token_stride)
                        + tid_x_vec
                    )
                    # Build a per-block GTensor and store VEC bf16 via dword path.
                    # bf16 VEC ? {2, 4, 8} = {4, 8, 16} bytes = {1, 2, 4} dwords.
                    out_vec_t = T.vec(VEC, T.bf16)
                    raw_vec = vector.from_elements(vecVf32, out_lane)
                    bf16_vec = raw_vec.truncf(out_vec_t)
                    out_rsrc = buffer_ops.create_buffer_resource(
                        kv_cache, max_size=True
                    )
                    # cache_off is in bf16 elements; convert to dword for the i32-vec store.
                    cache_off_dw = ArithValue(cache_off) >> arith.constant(1, type=i32)
                    dwords = (VEC + 1) // 2
                    bf16_as_i32 = vector.bitcast(T.vec(dwords, T.i32), bf16_vec)
                    if const_expr(dwords == 1):
                        # vec<1xi32> -> scalar i32 store
                        scalar_i32 = vector.extract(
                            bf16_as_i32, static_position=[0], dynamic_position=[]
                        )
                        buffer_ops.buffer_store(scalar_i32, out_rsrc, cache_off_dw)
                    else:
                        buffer_ops.buffer_store(bf16_as_i32, out_rsrc, cache_off_dw)
                elif const_expr(nm_asm):
                    # -- group_fp8 (V4 nm-asm): nope fp8 + inline dup e8m0; rope bf16
                    # -> separate k_rope_buff. Shared emitter (byte-identical to HCA). --
                    _nm_cache_base = ArithValue(physical_block) * ArithValue(
                        kv_cache_block_stride
                    ) + ArithValue(slot_in_block) * ArithValue(kv_cache_token_stride)
                    _nm_krope_base = ArithValue(physical_block) * ArithValue(
                        krope_block_stride
                    ) + ArithValue(slot_in_block) * ArithValue(krope_token_stride)
                    emit_group_fp8_nm_asm_scatter(
                        normed_lane=normed_lane,
                        rotated_lane=rotated_lane,
                        lane=tid,
                        is_rope_t=is_rope_t,
                        cache_base=_to_raw(_nm_cache_base),
                        out_rsrc=buffer_ops.create_buffer_resource(
                            kv_cache, max_size=True
                        ),
                        krope_base=_to_raw(_nm_krope_base),
                        krope_rsrc=buffer_ops.create_buffer_resource(
                            k_rope_buff, max_size=True
                        ),
                        VEC=VEC,
                        NOPE=NOPE,
                        RTS=RTS,
                        log2_rts=log2_rts,
                        ROPE_THREAD_LO=ROPE_THREAD_LO,
                        wave_width=BLOCK_THREADS,
                        vecVf32=vecVf32,
                        fm_fast=fm_fast,
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
                    amax = wave_reduce_max(am_local)
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
                    inv_scale = llvm.call_intrinsic(
                        f32, "llvm.amdgcn.rcp.f32", [scale_v], [], []
                    )

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
                        v = arith.minimumf(arith.maximumf(v, c_neg_fp8_max), c_fp8_max)
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
                        peer_pk = ArithValue(pk).shuffle_xor(1, BLOCK_THREADS)
                        dword = ArithValue(pk) | (
                            ArithValue(peer_pk) << arith.constant(16, type=i32)
                        )
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
                        # VEC == 8: 8 bytes = 2 dwords. Build separately.
                        assert VEC == 8
                        p0 = rocdl.cvt_pk_fp8_f32(
                            i32, fp8_inputs[0], fp8_inputs[1], c_p0, 0
                        )
                        p0 = rocdl.cvt_pk_fp8_f32(
                            i32, fp8_inputs[2], fp8_inputs[3], p0, 1
                        )
                        p1 = rocdl.cvt_pk_fp8_f32(
                            i32, fp8_inputs[4], fp8_inputs[5], c_p0, 0
                        )
                        p1 = rocdl.cvt_pk_fp8_f32(
                            i32, fp8_inputs[6], fp8_inputs[7], p1, 1
                        )
                        dword = (p0, p1)

                    # Compute store address (in BYTES from kv_cache base).
                    # Both layouts use the same base (= phys * block_stride);
                    # the offset within the block differs.
                    out_rsrc = buffer_ops.create_buffer_resource(
                        kv_cache, max_size=True
                    )
                    block_byte_base = ArithValue(physical_block) * ArithValue(
                        kv_cache_block_stride
                    )

                    if const_expr(preshuffle):
                        # MFMA 16x16 tile layout
                        # offset = block_base
                        #        + token_tile_id * (TILE * D)
                        #        + col_tile_id * (TILE * TILE)
                        #        + token_in_tile * TILE
                        #        + col_in_tile
                        c_TILE = arith.constant(_PRESHUFFLE_TILE, type=i32)
                        c_TILE_D = arith.constant(_PRESHUFFLE_TILE * D, type=i32)
                        c_TILE_TILE = arith.constant(
                            _PRESHUFFLE_TILE * _PRESHUFFLE_TILE, type=i32
                        )
                        token_tile_id = arith.divsi(slot_in_block, c_TILE)
                        token_in_tile = arith.remui(slot_in_block, c_TILE)
                        # d = tid * VEC; col_tile_id = d // TILE; col_in_tile = d % TILE
                        d_for_tid = ArithValue(tid) * arith.constant(VEC, type=i32)
                        col_tile_id = arith.divsi(d_for_tid, c_TILE)
                        col_in_tile = arith.remui(d_for_tid, c_TILE)
                        in_block_off = (
                            ArithValue(token_tile_id) * c_TILE_D
                            + ArithValue(col_tile_id) * c_TILE_TILE
                            + ArithValue(token_in_tile) * c_TILE
                            + ArithValue(col_in_tile)
                        )
                    else:
                        # Linear layout: phys * block_stride + slot * D + tid * VEC
                        in_block_off = ArithValue(slot_in_block) * arith.constant(
                            D, type=i32
                        ) + ArithValue(tid) * arith.constant(VEC, type=i32)

                    byte_off = block_byte_base + in_block_off

                    if const_expr(VEC == 2):
                        # Only even tid stores (its dword covers peer's bytes too).
                        is_even = arith.cmpi(
                            CmpIPredicate.eq,
                            arith.andi(_to_raw(tid), arith.constant(1, type=i32)),
                            arith.constant(0, type=i32),
                        )
                        _if_even = scf.IfOp(is_even)
                        with _if_then(_if_even):
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
                        # VEC == 8: store 2 dwords (8 bytes) via vec<2xi32>
                        store_vec = vector.from_elements(
                            T.vec(2, i32), [dword[0], dword[1]]
                        )
                        buffer_ops.buffer_store(
                            store_vec,
                            out_rsrc,
                            byte_off,
                            offset_is_bytes=True,
                        )

                    # (f) lane-0 writes fp32 scale at cache_scale[phys, slot]
                    is_lane0 = arith.cmpi(
                        CmpIPredicate.eq,
                        _to_raw(tid),
                        arith.constant(0, type=i32),
                    )
                    _if_l0 = scf.IfOp(is_lane0)
                    with _if_then(_if_l0):
                        cs_rsrc = buffer_ops.create_buffer_resource(
                            cache_scale, max_size=True
                        )
                        cs_off = ArithValue(physical_block) * ArithValue(
                            cache_scale_block_stride
                        ) + ArithValue(slot_in_block)
                        buffer_ops.buffer_store(scale_v, cs_rsrc, cs_off)
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
        stream: fx.Stream = fx.Stream(None),
    ):
        idx_p = arith.index_cast(T.index, _to_raw(plan_capacity))
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
            grid=(idx_p, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_fused_compress_attn


# ============================================================================
# K-split single-kernel builder (multi-wave LDS reduce)
# ============================================================================
#
# Why this exists: the legacy single-wave kernel above runs ONE wave64 per
# boundary, serializing K iters of online-softmax. PMC on CSA Main (D=512,
# K=8) showed VALU IPC ~0.33 with 53% of cycles in SQ_WAIT_ANY and only 128
# VMEM insts -- i.e. the wave is stalled on the *serial dependency chain*
# (each iter's m/kv/w accumulator + 2x exp2 transcendental per lane), not on
# memory. At decode bs=1-32 each CU holds a single wave, so nothing hides the
# chain latency.
#
# Fix: split K across NW waves in ONE workgroup (block = 64*NW), grid stays
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
    quant_mode: str = "per_row_fp8",  # "per_row_fp8" | "group_fp8" (nm-asm)
    quant_group_size: int = 64,
):
    """K-split single-kernel: NW-wave LDS-reduced compress + norm + rope +
    scatter (BF16 or FP8). Constexpr knobs mirror :func:`_build_kernel` minus
    ``enable_prefetch_input`` (each wave runs so few iters that prefetch is
    moot). The FP8 / ue8m0 / preshuffle scatter is emitted in wave 0, where
    ``lid`` (0..63) plays the single-wave ``tid`` role -- pair-coop
    shuffle_xor and wave_reduce_max stay within wave 0's 64 lanes, identical
    to the legacy kernel's semantics.

    Layout:
      - Grid:  (plan_capacity, 1, 1)  -- one workgroup per plan row.
      - Block: 64 * NW threads (NW waves).
      - VEC = D / 64: each lane owns VEC contiguous D-columns. One wave's 64
        lanes cover the full head_dim.
      - K = (2 if overlap else 1) * ratio, split into NW slices of K_PER_WAVE.
      - LDS: 3 fp32 arrays of NW*D (m, kv, w). Each lane writes its VEC
        accumulator values at wid*D + lid*VEC + i; wave 0 reads NW values per
        owned element and folds them with a global-max online-softmax.
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

    # FP8 group_fp8 (V4 nm-asm) geometry: nope split into groups of
    # GROUP_SIZE_Q; RTS lanes/group cooperate on amax. Scatter reuses the
    # shared emitter (byte-identical to legacy / HCA). See _build_kernel.
    nm_asm = quant_mode == "group_fp8"
    GROUP_SIZE_Q = quant_group_size
    RTS = (GROUP_SIZE_Q // VEC) if nm_asm else 1
    log2_rts = int(math.log2(RTS)) if nm_asm else 0

    assert D % BLOCK_THREADS == 0, f"D={D} must divide {BLOCK_THREADS}"
    assert VEC in (2, 4, 8), f"VEC={VEC} outside supported set"
    assert NOPE >= 0 and NOPE % VEC == 0
    assert RD > 0 and RD % 2 == 0 and RD % VEC == 0
    assert state_size >= K, f"state_size={state_size} < K={K}"
    assert K % NW == 0, f"K={K} must divide evenly across NW={NW} waves"
    if quant and preshuffle:
        assert D % _PRESHUFFLE_TILE == 0
        assert k_per_block % _PRESHUFFLE_TILE == 0
    if nm_asm:
        assert quant and not preshuffle, "nm_asm: requires quant=True, preshuffle=False"
        assert (
            NOPE % GROUP_SIZE_Q == 0 and GROUP_SIZE_Q % VEC == 0
        ), f"nm_asm: NOPE={NOPE} % G={GROUP_SIZE_Q} and G % VEC={VEC} must be 0"

    # LDS: 3 fp32 arrays, each NW * D entries.
    LDS_ELEMS = NW * D
    LDS_BYTES = LDS_ELEMS * 4

    GPU_ARCH = get_rocm_arch()
    allocator = SmemAllocator(
        None,
        arch=GPU_ARCH,
        global_sym_name=(
            f"csa_ksplit_smem_D{D}_R{ratio}_O{int(overlap)}_NW{NW}_S{state_size}"
        ),
    )
    lds_m_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_m_off + LDS_BYTES
    lds_kv_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_kv_off + LDS_BYTES
    lds_w_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_w_off + LDS_BYTES

    _name_parts = [
        "fused_compress_attn",
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
        if nm_asm:
            _name_parts.append(f"nmasm{GROUP_SIZE_Q}")
        else:
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
        k_rope_buff: fx.Tensor,  # nm_asm only: paged [NB, k_per_block, RD] bf16 rope (dummy otherwise)
        krope_block_stride: Int32,
        krope_token_stride: Int32,
        block_table: fx.Tensor,
        block_table_seq_stride: Int32,
    ):
        f32 = T.f32
        i32 = T.i32
        vecVf32 = T.vec(VEC, T.f32)

        pid = fx.block_idx.x
        tid = fx.thread_idx.x  # 0 .. BLOCK_TH-1

        c_neg_inf = arith.constant(_NEG_INF, type=f32)
        c_zero_f32 = arith.constant(0.0, type=f32)
        c_zero_i32 = arith.constant(0, type=i32)
        c_one_i32 = arith.constant(1, type=i32)
        c_64 = arith.constant(BLOCK_THREADS, type=i32)
        c_eps = arith.constant(rms_eps, type=f32)
        c_inv_D = arith.constant(1.0 / D, type=f32)
        c_log2e = arith.constant(_LOG2E, type=f32)
        c_K_m1 = arith.constant(K - 1, type=i32)
        c_K_per_wave = arith.constant(K_PER_WAVE, type=i32)
        c_state_size = arith.constant(state_size, type=i32)
        c_VEC = arith.constant(VEC, type=i32)
        c_D = arith.constant(D, type=i32)

        def fexp_f32(x):
            return llvm.call_intrinsic(
                f32, "llvm.amdgcn.exp2.f32", [x * c_log2e], [], []
            )

        wid = arith.divsi(_to_raw(tid), c_64)  # ? [0, NW)
        lid = arith.remui(_to_raw(tid), c_64)  # ? [0, 64)

        # ---- plan row (single dwordx4) ----
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
            slot_map_rsrc = buffer_ops.create_buffer_resource(
                state_slot_mapping, max_size=True
            )
            slot = buffer_ops.buffer_load(
                slot_map_rsrc, batch_id, vec_width=1, dtype=i32
            )

            # This lane owns columns [lid*VEC, lid*VEC+VEC) of head_dim.
            lid_x_vec = ArithValue(lid) * c_VEC

            kv_in_rsrc = buffer_ops.create_buffer_resource(kv_in, max_size=True)
            score_in_rsrc = buffer_ops.create_buffer_resource(score_in, max_size=True)
            kv_state_rsrc = buffer_ops.create_buffer_resource(kv_state, max_size=True)
            score_state_rsrc = buffer_ops.create_buffer_resource(
                score_state, max_size=True
            )
            ape_rsrc = buffer_ops.create_buffer_resource(ape, max_size=True)

            def _col_off_for_k(k_i32):
                if const_expr(not overlap):
                    return c_zero_i32
                is_b = arith.cmpi(
                    CmpIPredicate.sge, k_i32, arith.constant(ratio, type=i32)
                )
                return arith.select(is_b, c_D, c_zero_i32)

            def _load_f32_vec(rsrc, off_elems_i32):
                if const_expr(VEC <= 4):
                    raw = buffer_ops.buffer_load(
                        rsrc, off_elems_i32, vec_width=VEC, dtype=f32
                    )
                    return [
                        vector.extract(raw, static_position=[i], dynamic_position=[])
                        for i in range(VEC)
                    ]
                else:
                    assert VEC == 8
                    half = VEC // 2
                    r0 = buffer_ops.buffer_load(
                        rsrc, off_elems_i32, vec_width=half, dtype=f32
                    )
                    r1 = buffer_ops.buffer_load(
                        rsrc,
                        ArithValue(off_elems_i32) + arith.constant(half, type=i32),
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

            def _load_bf16_vec_then_f32(rsrc, off_elems_i32):
                off_dw = ArithValue(off_elems_i32) >> c_one_i32
                dwords = (VEC + 1) // 2
                if const_expr(dwords == 1):
                    raw_s = buffer_ops.buffer_load(rsrc, off_dw, vec_width=1, dtype=i32)
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

            def _softmax_step(m_lane, kv_lane, w_lane, score_lane, kv_v_lane):
                """Padding-aware per-lane online-softmax update. Phase 2 scores
                are finite, so the is-pad branch is dead code there (compiler
                elides)."""
                new_m, new_kv, new_w = [], [], []
                for i in range_constexpr(VEC):
                    m_old = m_lane[i]
                    score = score_lane[i]
                    m_new = arith.maximumf(m_old, score)
                    is_first = arith.cmpf(CmpFPredicate.OEQ, m_old, c_neg_inf)
                    scale_active = fexp_f32(arith.subf(m_old, m_new))
                    scale_v = arith.select(is_first, c_zero_f32, scale_active)
                    wk_active = fexp_f32(arith.subf(score, m_new))
                    is_pad = arith.cmpf(CmpFPredicate.OEQ, score, c_neg_inf)
                    w_k = arith.select(is_pad, c_zero_f32, wk_active)
                    new_m.append(m_new)
                    new_kv.append(
                        arith.AddFOp(
                            arith.MulFOp(kv_lane[i], scale_v, fastmath=fm_fast).result,
                            arith.MulFOp(w_k, kv_v_lane[i], fastmath=fm_fast).result,
                            fastmath=fm_fast,
                        ).result
                    )
                    new_w.append(
                        arith.AddFOp(
                            arith.MulFOp(w_lane[i], scale_v, fastmath=fm_fast).result,
                            w_k,
                            fastmath=fm_fast,
                        ).result
                    )
                return new_m, new_kv, new_w

            def _phase1_loads(k_i32):
                s = arith.addi(arith.subi(_to_raw(position), c_K_m1), k_i32)
                is_pad = arith.cmpi(CmpIPredicate.slt, s, c_zero_i32)
                s_safe = arith.select(is_pad, c_zero_i32, s)
                ring = arith.remui(s_safe, c_state_size)
                col_off = _col_off_for_k(k_i32)
                base_kv = (
                    ArithValue(slot) * ArithValue(kv_state_slot_stride)
                    + ArithValue(ring) * ArithValue(kv_state_pos_stride)
                    + ArithValue(col_off)
                    + lid_x_vec
                )
                base_sc = (
                    ArithValue(slot) * ArithValue(score_state_slot_stride)
                    + ArithValue(ring) * ArithValue(score_state_pos_stride)
                    + ArithValue(col_off)
                    + lid_x_vec
                )
                kv_v = _load_f32_vec(kv_state_rsrc, base_kv)
                sc_v = _load_f32_vec(score_state_rsrc, base_sc)
                sc_pad = [arith.select(is_pad, c_neg_inf, sc_v[i]) for i in range(VEC)]
                return kv_v, sc_pad

            def _phase2_loads(k_i32):
                col_off = _col_off_for_k(k_i32)
                ape_row = arith.remui(k_i32, arith.constant(ratio, type=i32))
                in_row_raw = arith.subi(_to_raw(ragged_id), arith.subi(c_K_m1, k_i32))
                in_row = arith.maxsi(in_row_raw, c_zero_i32)
                base_in = (
                    ArithValue(in_row) * ArithValue(kv_in_row_stride)
                    + ArithValue(col_off)
                    + lid_x_vec
                )
                base_sc = (
                    ArithValue(in_row) * ArithValue(score_in_row_stride)
                    + ArithValue(col_off)
                    + lid_x_vec
                )
                base_ape = (
                    ArithValue(ape_row) * arith.constant(DIM_FULL, type=i32)
                    + ArithValue(col_off)
                    + lid_x_vec
                )
                kv = _load_bf16_vec_then_f32(kv_in_rsrc, base_in)
                sc = _load_bf16_vec_then_f32(score_in_rsrc, base_sc)
                ape_v = _load_f32_vec(ape_rsrc, base_ape)
                score = [
                    arith.AddFOp(sc[i], ape_v[i], fastmath=fm_fast).result
                    for i in range(VEC)
                ]
                return kv, score

            # ---- this wave's K range [wid*KPW, (wid+1)*KPW), split at window_len ----
            k_start = ArithValue(wid) * c_K_per_wave
            k_end = k_start + c_K_per_wave
            wl = _to_raw(window_len)
            split_lo = arith.maxsi(wl, _to_raw(k_start))
            split = arith.minsi(split_lo, _to_raw(k_end))

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
                k_i32 = arith.index_cast(i32, _to_raw(k_static))
                kv_v, sc_v = _phase1_loads(k_i32)
                nm, nkv, nw = _softmax_step(m_lane, kv_lane, w_lane, sc_v, kv_v)
                p1 = yield list(nm) + list(nkv) + list(nw)

            # Phase 2 sub-loop [split, k_end): ragged input.
            final = p1
            for k_static, state in range(_to_raw(split), _to_raw(k_end), 1, init=p1):
                m_lane = list(state[0:VEC])
                kv_lane = list(state[VEC : 2 * VEC])
                w_lane = list(state[2 * VEC : 3 * VEC])
                k_i32 = arith.index_cast(i32, _to_raw(k_static))
                kv_v, score = _phase2_loads(k_i32)
                nm, nkv, nw = _softmax_step(m_lane, kv_lane, w_lane, score, kv_v)
                final = yield list(nm) + list(nkv) + list(nw)

            m_local = list(final[0:VEC])
            kv_local = list(final[VEC : 2 * VEC])
            w_local = list(final[2 * VEC : 3 * VEC])

            # ---- LDS write: each lane writes VEC entries at wid*D + lid*VEC ----
            lds_base = allocator.get_base()
            lds_m = STensor(
                SmemPtr(lds_base, lds_m_off, T.f32, shape=(LDS_ELEMS,)),
                dtype=T.f32,
                shape=(LDS_ELEMS,),
            )
            lds_kv = STensor(
                SmemPtr(lds_base, lds_kv_off, T.f32, shape=(LDS_ELEMS,)),
                dtype=T.f32,
                shape=(LDS_ELEMS,),
            )
            lds_w = STensor(
                SmemPtr(lds_base, lds_w_off, T.f32, shape=(LDS_ELEMS,)),
                dtype=T.f32,
                shape=(LDS_ELEMS,),
            )
            lds_thread_base = ArithValue(wid) * c_D + lid_x_vec
            for i in range_constexpr(VEC):
                idx_i = lds_thread_base + arith.constant(i, type=i32)
                lds_m[fx.Index(idx_i)] = m_local[i]
                lds_kv[fx.Index(idx_i)] = kv_local[i]
                lds_w[fx.Index(idx_i)] = w_local[i]

            gpu.barrier()

            # ---- wave 0: cross-wave reduce + norm + rope + scatter ----
            is_wave0 = arith.cmpi(CmpIPredicate.eq, wid, c_zero_i32)
            _if_w0 = scf.IfOp(is_wave0)
            with _if_then(_if_w0):
                comp_lane = []
                for i in range_constexpr(VEC):
                    lane_off = lid_x_vec + arith.constant(i, type=i32)
                    m_g = c_neg_inf
                    m_arr = []
                    for w in range_constexpr(NW):
                        idx_w = arith.constant(w * D, type=i32) + lane_off
                        m_w = lds_m[fx.Index(idx_w)]
                        m_arr.append(m_w)
                        m_g = arith.maximumf(m_g, m_w)
                    kv_sum = c_zero_f32
                    w_sum = c_zero_f32
                    for w in range_constexpr(NW):
                        idx_w = arith.constant(w * D, type=i32) + lane_off
                        kv_w = lds_kv[fx.Index(idx_w)]
                        w_w = lds_w[fx.Index(idx_w)]
                        scale_w = fexp_f32(arith.subf(m_arr[w], m_g))
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
                    comp_lane.append(
                        arith.MulFOp(kv_sum, rcp_w, fastmath=fm_fast).result
                    )

                # ---- RMSNorm (wave-reduce sum-of-squares over wave 0) ----
                def wave_reduce_add(x):
                    w = _to_raw(x)
                    for sh_exp in range_constexpr(log2_block):
                        off = BLOCK_THREADS // (2 << sh_exp)
                        peer = _to_raw(ArithValue(w).shuffle_xor(off, BLOCK_THREADS))
                        w = arith.AddFOp(w, peer, fastmath=fm_fast).result
                    return w

                sq_local = arith.constant(0.0, type=f32)
                for i in range_constexpr(VEC):
                    sq_local = arith.AddFOp(
                        sq_local,
                        arith.MulFOp(
                            comp_lane[i], comp_lane[i], fastmath=fm_fast
                        ).result,
                        fastmath=fm_fast,
                    ).result
                sq_full = wave_reduce_add(sq_local)
                var = arith.MulFOp(sq_full, c_inv_D, fastmath=fm_fast).result
                rrms = fmath.rsqrt(
                    arith.AddFOp(var, c_eps, fastmath=fm_fast).result, fastmath=fm_fast
                )

                rmsw_rsrc = buffer_ops.create_buffer_resource(rms_weight, max_size=True)
                if const_expr(rms_weight_is_bf16):
                    rmsw_lane = _load_bf16_vec_then_f32(rmsw_rsrc, lid_x_vec)
                else:
                    rmsw_lane = _load_f32_vec(rmsw_rsrc, lid_x_vec)

                normed_lane = [
                    arith.MulFOp(
                        arith.MulFOp(comp_lane[i], rrms, fastmath=fm_fast).result,
                        rmsw_lane[i],
                        fastmath=fm_fast,
                    ).result
                    for i in range(VEC)
                ]

                # ---- GPT-J RoPE on RD tail ----
                comp_pos_i32 = arith.muli(
                    arith.divsi(_to_raw(position), arith.constant(ratio, type=i32)),
                    arith.constant(ratio, type=i32),
                )
                cos_rsrc = buffer_ops.create_buffer_resource(cos_cache, max_size=True)
                sin_rsrc = buffer_ops.create_buffer_resource(sin_cache, max_size=True)
                c_half_rd = arith.constant(RD // 2, type=i32)
                cos_row_base = ArithValue(comp_pos_i32) * c_half_rd

                is_rope_t = arith.cmpi(
                    CmpIPredicate.sge,
                    _to_raw(lid),
                    arith.constant(ROPE_THREAD_LO, type=i32),
                )
                rope_rel_raw = ArithValue(lid) - arith.constant(
                    ROPE_THREAD_LO, type=i32
                )
                rope_rel = arith.maxsi(rope_rel_raw, c_zero_i32)
                cs_lo = ArithValue(rope_rel) * arith.constant(
                    PAIRS_PER_THREAD, type=i32
                )

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
                ci = arith.divsi(_to_raw(position), arith.constant(ratio, type=i32))
                block_in_seq = arith.divsi(ci, arith.constant(k_per_block, type=i32))
                slot_in_block = arith.remui(ci, arith.constant(k_per_block, type=i32))
                bt_rsrc = buffer_ops.create_buffer_resource(block_table, max_size=True)
                bt_off = ArithValue(batch_id) * ArithValue(
                    block_table_seq_stride
                ) + ArithValue(block_in_seq)
                physical_block = buffer_ops.buffer_load(
                    bt_rsrc, bt_off, vec_width=1, dtype=i32
                )

                if const_expr(not quant):
                    cache_off = (
                        ArithValue(physical_block) * ArithValue(kv_cache_block_stride)
                        + ArithValue(slot_in_block) * ArithValue(kv_cache_token_stride)
                        + lid_x_vec
                    )
                    out_vec_t = T.vec(VEC, T.bf16)
                    raw_vec = vector.from_elements(vecVf32, out_lane)
                    bf16_vec = raw_vec.truncf(out_vec_t)
                    out_rsrc = buffer_ops.create_buffer_resource(
                        kv_cache, max_size=True
                    )
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
                elif const_expr(nm_asm):
                    # -- group_fp8 (V4 nm-asm): nope fp8 + inline dup e8m0; rope
                    # bf16 -> separate k_rope_buff. Shared emitter, byte-identical
                    # to the legacy single-wave / HCA paths (lane == wave-0 lid). --
                    _nm_cache_base = ArithValue(physical_block) * ArithValue(
                        kv_cache_block_stride
                    ) + ArithValue(slot_in_block) * ArithValue(kv_cache_token_stride)
                    _nm_krope_base = ArithValue(physical_block) * ArithValue(
                        krope_block_stride
                    ) + ArithValue(slot_in_block) * ArithValue(krope_token_stride)
                    emit_group_fp8_nm_asm_scatter(
                        normed_lane=normed_lane,
                        rotated_lane=rotated_lane,
                        lane=lid,
                        is_rope_t=is_rope_t,
                        cache_base=_to_raw(_nm_cache_base),
                        out_rsrc=buffer_ops.create_buffer_resource(
                            kv_cache, max_size=True
                        ),
                        krope_base=_to_raw(_nm_krope_base),
                        krope_rsrc=buffer_ops.create_buffer_resource(
                            k_rope_buff, max_size=True
                        ),
                        VEC=VEC,
                        NOPE=NOPE,
                        RTS=RTS,
                        log2_rts=log2_rts,
                        ROPE_THREAD_LO=ROPE_THREAD_LO,
                        wave_width=BLOCK_THREADS,
                        vecVf32=vecVf32,
                        fm_fast=fm_fast,
                    )
                else:
                    # -- FP8 per-row scaled write + fp32 scale (mirror legacy) --
                    # Wave-reduce-max over wave 0's 64 lanes; pair-coop dword
                    # store via shuffle_xor(1) within the wave.
                    def wave_reduce_max(x):
                        w = _to_raw(x)
                        for sh_exp in range_constexpr(log2_block):
                            off = BLOCK_THREADS // (2 << sh_exp)
                            peer = _to_raw(
                                ArithValue(w).shuffle_xor(off, BLOCK_THREADS)
                            )
                            w = arith.maximumf(w, peer)
                        return w

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
                    amax = wave_reduce_max(am_local)
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
                    inv_scale = llvm.call_intrinsic(
                        f32, "llvm.amdgcn.rcp.f32", [scale_v], [], []
                    )

                    # (d) per-lane fp8 cast: clamp + fnuz NaN guard
                    c_neg_uf = arith.constant(-(2.0**-8), type=f32)
                    c_zero = arith.constant(0.0, type=f32)
                    fp8_inputs = []
                    for i in range_constexpr(VEC):
                        v = arith.MulFOp(
                            out_lane[i], inv_scale, fastmath=fm_fast
                        ).result
                        v = arith.minimumf(arith.maximumf(v, c_neg_fp8_max), c_fp8_max)
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
                        peer_pk = ArithValue(pk).shuffle_xor(1, BLOCK_THREADS)
                        dword = ArithValue(pk) | (
                            ArithValue(peer_pk) << arith.constant(16, type=i32)
                        )
                    elif const_expr(VEC == 4):
                        pk = rocdl.cvt_pk_fp8_f32(
                            i32, fp8_inputs[0], fp8_inputs[1], c_p0, 0
                        )
                        pk = rocdl.cvt_pk_fp8_f32(
                            i32, fp8_inputs[2], fp8_inputs[3], pk, 1
                        )
                        dword = pk
                    else:
                        assert VEC == 8
                        p0 = rocdl.cvt_pk_fp8_f32(
                            i32, fp8_inputs[0], fp8_inputs[1], c_p0, 0
                        )
                        p0 = rocdl.cvt_pk_fp8_f32(
                            i32, fp8_inputs[2], fp8_inputs[3], p0, 1
                        )
                        p1 = rocdl.cvt_pk_fp8_f32(
                            i32, fp8_inputs[4], fp8_inputs[5], c_p0, 0
                        )
                        p1 = rocdl.cvt_pk_fp8_f32(
                            i32, fp8_inputs[6], fp8_inputs[7], p1, 1
                        )
                        dword = (p0, p1)

                    out_rsrc = buffer_ops.create_buffer_resource(
                        kv_cache, max_size=True
                    )
                    block_byte_base = ArithValue(physical_block) * ArithValue(
                        kv_cache_block_stride
                    )

                    if const_expr(preshuffle):
                        c_TILE = arith.constant(_PRESHUFFLE_TILE, type=i32)
                        c_TILE_D = arith.constant(_PRESHUFFLE_TILE * D, type=i32)
                        c_TILE_TILE = arith.constant(
                            _PRESHUFFLE_TILE * _PRESHUFFLE_TILE, type=i32
                        )
                        token_tile_id = arith.divsi(slot_in_block, c_TILE)
                        token_in_tile = arith.remui(slot_in_block, c_TILE)
                        d_for_tid = ArithValue(lid) * arith.constant(VEC, type=i32)
                        col_tile_id = arith.divsi(d_for_tid, c_TILE)
                        col_in_tile = arith.remui(d_for_tid, c_TILE)
                        in_block_off = (
                            ArithValue(token_tile_id) * c_TILE_D
                            + ArithValue(col_tile_id) * c_TILE_TILE
                            + ArithValue(token_in_tile) * c_TILE
                            + ArithValue(col_in_tile)
                        )
                    else:
                        in_block_off = ArithValue(slot_in_block) * arith.constant(
                            D, type=i32
                        ) + ArithValue(lid) * arith.constant(VEC, type=i32)

                    byte_off = block_byte_base + in_block_off

                    if const_expr(VEC == 2):
                        is_even = arith.cmpi(
                            CmpIPredicate.eq,
                            arith.andi(_to_raw(lid), arith.constant(1, type=i32)),
                            arith.constant(0, type=i32),
                        )
                        _if_even = scf.IfOp(is_even)
                        with _if_then(_if_even):
                            buffer_ops.buffer_store(
                                dword, out_rsrc, byte_off, offset_is_bytes=True
                            )
                    elif const_expr(VEC == 4):
                        buffer_ops.buffer_store(
                            dword, out_rsrc, byte_off, offset_is_bytes=True
                        )
                    else:
                        store_vec = vector.from_elements(
                            T.vec(2, i32), [dword[0], dword[1]]
                        )
                        buffer_ops.buffer_store(
                            store_vec, out_rsrc, byte_off, offset_is_bytes=True
                        )

                    # (f) lane-0 writes fp32 scale at cache_scale[phys, slot]
                    is_lane0 = arith.cmpi(
                        CmpIPredicate.eq,
                        _to_raw(lid),
                        arith.constant(0, type=i32),
                    )
                    _if_l0 = scf.IfOp(is_lane0)
                    with _if_then(_if_l0):
                        cs_rsrc = buffer_ops.create_buffer_resource(
                            cache_scale, max_size=True
                        )
                        cs_off = ArithValue(physical_block) * ArithValue(
                            cache_scale_block_stride
                        ) + ArithValue(slot_in_block)
                        buffer_ops.buffer_store(scale_v, cs_rsrc, cs_off)

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
        k_rope_buff: fx.Tensor,
        krope_block_stride: fx.Int32,
        krope_token_stride: fx.Int32,
        block_table: fx.Tensor,
        block_table_seq_stride: fx.Int32,
        plan_capacity: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        idx_p = arith.index_cast(T.index, _to_raw(plan_capacity))
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
            grid=(idx_p, 1, 1),
            block=(BLOCK_TH, 1, 1),
            stream=stream,
        )

    return launch_fused_compress_attn_ksplit


# ============================================================================
# Cached compile + public API
# ============================================================================


_DEFAULT_COMPILE_HINTS = {
    "waves_per_eu": 8,
    "fast_fp_math": True,
    "unsafe_fp_math": True,
}


def hca_per_n_config(plan_capacity: int) -> tuple[int, int]:
    """Return ``(slice_size, k_split_num_waves)`` for ``flydsl_hca_compress_attn``
    tuned for V4-Pro HCA Main (D=512, ratio=128, overlap=False) on MI355X.

    Dispatch on ``plan_capacity`` (NOT ``num_compress``): the HCA kernel's
    grid is ``plan_capacity`` (sentinel rows bail inside), and only
    ``plan_capacity`` is CUDAGraph-stable (it's fixed per (ratio,
    batch_bucket) and baked into captured launch args). Dispatching on the
    runtime-varying ``num_compress`` would silently mis-select kernels for
    captured graphs.
    """
    if plan_capacity <= 64:
        return 64, 8
    if plan_capacity <= 256:
        return 128, 8
    if plan_capacity <= 384:
        return 256, 8
    if plan_capacity <= 768:
        # HCA decode bs=512 (plan_cap = bs * ceil((1+MTP)/ratio) = 512 for
        # ratio=128, MTP<=3) prefers k_split=4 over 8: 30.4 vs 34.9 us on
        # MI355X (~14% win, ~5 us per HCA layer).
        return 256, 4
    if plan_capacity <= 1024:
        return 512, 8
    if plan_capacity <= 8192:
        return 512, 4
    return 512, 1


@lru_cache(maxsize=32)
def compile_flydsl_fused_compress_attn(
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
    enable_prefetch_input: bool = True,
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


def csa_ksplit_num_waves(plan_capacity: int) -> int:
    """Auto-pick ``k_split_num_waves`` for the CSA Main BF16 path
    (D=512, RD=64, ratio=4, overlap, K=8) and the CSA Indexer FP8 path
    (D=128, RD=64, ratio=4, overlap, K=8) on MI355X. Returns 1 to mean
    "use the legacy single-wave kernel".

    Rationale (measured, MI355X): the multi-wave LDS K-split parallelizes the
    serial online-softmax chain that bottlenecks the latency-bound small-N
    regime (decode: plan_capacity <= ~528 always). It wins ~22-30% (BF16) /
    ~15-24% (FP8) there. At high N the per-boundary CU occupancy is already
    saturated, so the extra NWx blocks + LDS reduce become pure overhead and
    the kernel regresses past plan_capacity ? 1300. NW=4 beats NW=8 (8-wave
    LDS fold costs more than the 2 extra K-iters it removes for K=8) and
    beats NW=2 across both shapes' decode range.

    Dispatch on ``plan_capacity`` (NOT num_compress): it's the grid size and
    is CUDAGraph-stable (baked into captured launch args), so captured graphs
    select the same kernel they were traced with.
    """
    if plan_capacity <= 768:
        return 4
    if plan_capacity <= 1280:
        return 2
    return 1  # legacy single-wave


@lru_cache(maxsize=32)
def compile_flydsl_fused_compress_attn_ksplit(
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
    quant_group_size: int = 64,
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
        quant_group_size=quant_group_size,
    )
    launcher.compile_hints = dict(_DEFAULT_COMPILE_HINTS)
    return launcher


def flydsl_fused_compress_attn(
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
    kv_cache: Optional[torch.Tensor],  # bf16 or fp8; None ? no scatter
    block_tables: Optional[torch.Tensor],  # [bs, max_blocks_per_seq] i32
    k_per_block: int,
    overlap: bool,
    ratio: int,
    head_dim: int,
    rope_head_dim: int,
    quant: bool = False,
    cache_scale: Optional[torch.Tensor] = None,  # fp32 [NB, k_per_block]
    use_ue8m0: bool = True,
    preshuffle: bool = True,
    k_split_num_waves: Optional[int] = None,
    quant_mode: str = "per_row_fp8",  # "per_row_fp8" (indexer) | "group_fp8" (CSA/HCA Main, nm-asm) | (future) "fp4"
    k_rope_cache: Optional[
        torch.Tensor
    ] = None,  # group_fp8 only: paged [NB, k_per_block, RD] bf16 rope
    stream: Optional[torch.cuda.Stream] = None,
) -> None:
    """FlyDSL drop-in replacement for ``fused_compress_attn`` (Triton).

    Same side-effecting semantics: cache scatter IS the output. Caller MUST
    invoke BEFORE ``update_compressor_states`` (state cache reads must see
    previous-fwd data).

    ``k_split_num_waves`` (BF16 or FP8 scatter): when set to NW > 1, routes to
    the multi-wave LDS K-split kernel (block = 64*NW, K split across NW waves,
    single dispatch). Speeds up the latency-bound decode regime (small N,
    1 wave/CU) where the legacy single-wave serial K-chain stalls. Requires a
    real ``block_tables`` (has_block_table). When None, auto-picks NW for the
    tuned CSA Main (BF16) and CSA Indexer (FP8) shapes via
    :func:`csa_ksplit_num_waves` and uses legacy elsewhere; when 1, forces
    legacy. K must be divisible by NW.
    """
    # ---- gfx1250 dispatch (wave32) ----
    from aiter.jit.utils.chip_info import get_gfx as _get_gfx

    if _get_gfx() == "gfx1250":
        from .fused_compress_attn_gfx1250 import flydsl_fused_compress_attn_gfx1250

        return flydsl_fused_compress_attn_gfx1250(
            kv_in=kv_in,
            score_in=score_in,
            kv_state=kv_state,
            score_state=score_state,
            plan_gpu=plan_gpu,
            state_slot_mapping=state_slot_mapping,
            ape=ape,
            rms_weight=rms_weight,
            rms_eps=rms_eps,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            kv_cache=kv_cache,
            block_tables=block_tables,
            k_per_block=k_per_block,
            overlap=overlap,
            ratio=ratio,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            quant=quant,
            cache_scale=cache_scale,
            use_ue8m0=use_ue8m0,
            preshuffle=preshuffle,
            k_split_num_waves=k_split_num_waves,
            quant_mode=quant_mode,
            k_rope_cache=k_rope_cache,
            stream=stream,
        )

    # ---- input validation ----
    plan_capacity = plan_gpu.shape[0]
    if plan_capacity == 0:
        return

    # group_fp8 (V4 nm-asm group-quant): non-preshuffle, separate bf16 rope buffer.
    nm_asm = quant_mode == "group_fp8"
    if nm_asm:
        preshuffle = False

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
    if not (kv_state.is_contiguous() and score_state.is_contiguous()):
        raise ValueError("kv_state/score_state must be contiguous")
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
        if kv_cache.dtype == torch.bfloat16:
            raise TypeError("quant=True needs fp8 kv_cache")
        if not nm_asm and (
            cache_scale is None
            or cache_scale.dtype != torch.float32
            or cache_scale.dim() != 2
            or cache_scale.shape[0] != kv_cache.shape[0]
        ):
            raise ValueError("quant=True requires fp32 [NB, k_per_block] cache_scale")
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
        # nm_asm has no separate scale tensor (e8m0 inline in the fp8 entry); bf16/
        # non-quant paths don't use cache_scale either -> fp32 dummy.
        cs_arg = rms_weight  # fp32 dummy
        cs_block_stride = 0

    # nm_asm: rotated PE bf16 -> separate paged k_rope_cache (V4 nm layout). For all
    # other modes pass a bf16 dummy so the kernel arg binding stays valid.
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

    # ---- K-split fast path (BF16 + FP8 scatter) ----
    # k_split_num_waves: None ? auto-pick (tuned geometries only); int>1 ?
    # forced NW; 1 ? forced legacy. Auto triggers for the CSA Main (BF16) and
    # CSA Indexer (FP8) shapes the K-split kernel was tuned for; other shapes
    # fall through to the legacy single-wave kernel.
    _is_csa_main = (
        head_dim == 512 and rope_head_dim == 64 and ratio == 4 and overlap and not quant
    )
    _is_csa_indexer = (
        head_dim == 128 and rope_head_dim == 64 and ratio == 4 and overlap and quant
    )
    # group_fp8 (nm-asm) CSA Main shares the CSA Main geometry (D=512, K=8); it is
    # ported to the K-split kernel via the shared nm-asm scatter emitter.
    _is_csa_main_nm = (
        head_dim == 512
        and rope_head_dim == 64
        and ratio == 4
        and overlap
        and quant
        and nm_asm
    )
    if (
        k_split_num_waves is None
        and has_bt
        and (_is_csa_main or _is_csa_indexer or _is_csa_main_nm)
    ):
        nw_eff = csa_ksplit_num_waves(plan_capacity)
    else:
        nw_eff = k_split_num_waves if k_split_num_waves is not None else 1
    # nm_asm now K-split-capable only on the validated CSA Main shape; all other
    # nm_asm shapes still fall back to the legacy single-wave kernel.
    use_ksplit = nw_eff > 1 and has_bt and (not nm_asm or _is_csa_main_nm)
    if use_ksplit:
        k_split_num_waves = nw_eff
        if K_pool % k_split_num_waves != 0:
            raise ValueError(
                f"k_split_num_waves={k_split_num_waves} must divide K={K_pool}"
            )
        ks_launcher = compile_flydsl_fused_compress_attn_ksplit(
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
            quant_group_size=64,
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
            krope_arg,
            krope_block_stride,
            krope_token_stride,
            bt_arg,
            bt_seq_stride,
            plan_capacity,
            fx_stream,
        )
        _run_compiled(ks_launcher, *ks_args)
        return

    launcher = compile_flydsl_fused_compress_attn(
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
