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
  - K-split multi-wave (``_build_kernel_ksplit``, BF16 + FP8 + FP4 scatter):
    K split across NW waves in one workgroup (block = 64*NW), LDS cross-wave
    online-softmax reduce, single dispatch. Parallelizes the serial softmax
    chain that bottlenecks the latency-bound small-N decode regime. On the
    CSA Main (D=512, BF16) and CSA Indexer (D=128, FP8/FP4) shapes it
    auto-engages via ``csa_ksplit_num_waves(plan_capacity)`` and wins ~1.3-1.4×
    (BF16) / ~1.15-1.24× (FP8) at decode bs=1-32; it falls back to legacy at
    high N where CU occupancy already saturates. The K-split win comes from
    parallelizing the dtype-agnostic online-softmax pool, so FP4 reuses the
    FP8 wave-count heuristic. See ``flydsl_fused_compress_attn``'s
    ``k_split_num_waves`` arg.

Grid: ``(plan_capacity, 1, 1)`` -- one program per packed plan row
``[ragged_id, batch_id, position, window_len]``. Position == -1 rows
sentinel-skip; the whole body is a closure invoked under a runtime ``if``
(flydsl ``if cond: return`` does NOT actually early-exit inside a
@flyc.kernel, but ``if cond: _body()`` lowers to scf.if and guards it).

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
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import rocdl
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.arith import ArithValue, CmpFPredicate
from flydsl.expr.typing import Int32, Stream, T

from aiter.utility.mx_types import (
    MxDtypeInt as _MxDtypeInt,
)
from aiter.utility.mx_types import (
    MxScaleRoundModeInt as _MxRoundInt,
)

# Shared FP8 group_fp8 (V4 nm-asm) scatter emitter (single source of truth across
# the CSA single-kernel + HCA 2-kernel paths). See fused_compress_attn_common.
from .fused_compress_attn_common import emit_group_fp8_nm_asm_scatter
from .quant_utils import emit_f32_to_e2m1, emit_mx_e8m0_scale
from .tensor_shim import _run_compiled, _to_raw


def _velem(vec, i):
    """Element *i* of *vec* as a raw ir.Value (this kernel feeds raw MLIR ops)."""
    return _to_raw(fx.Vector(vec)[i])


def _minsi(a, b):
    """Signed integer min -> fx.Int32. No fx int-min surface exists, so this is
    the one raw arith primitive kept for it."""
    # BOUNDARY: arith.minsi (no fx signed-integer-min operator).
    return fx.Int32(arith.minsi(_to_raw(fx.Int32(a)), _to_raw(fx.Int32(b))))


def _maxsi(a, b):
    """Signed integer max -> fx.Int32 (no fx int-max surface exists)."""
    # BOUNDARY: arith.maxsi (no fx signed-integer-max operator).
    return fx.Int32(arith.maxsi(_to_raw(fx.Int32(a)), _to_raw(fx.Int32(b))))


def _oeq_raw(a, b):
    """Ordered fp equality (OEQ) as a fx.Boolean, built with a RAW arith.cmpf so
    the ambient fast_fp_math (nnan/ninf) does NOT fold the ``x == -inf`` sentinel
    check to a constant. The softmax first/pad guards compare against -inf, which
    fast math would otherwise assume unreachable -> corrupt results."""
    from flydsl.expr.numeric import Boolean

    # BOUNDARY: non-fastmath OEQ against the -inf softmax sentinel.
    return Boolean(arith.cmpf(CmpFPredicate.OEQ, _to_raw(a), _to_raw(b)))


def _max_raw(a, b):
    """NaN/inf-safe fp maximum -> fx.Float32, built with a RAW arith.maximumf
    (fastmath=None). The online-softmax running max operates on the -inf init
    sentinel; the ambient fast_fp_math ninf flag on fx `.maximumf` would let the
    backend assume -inf never occurs and drop the guard -> corrupt results."""
    # BOUNDARY: non-fastmath maximumf over the -inf softmax accumulator.
    return fx.Float32(arith.maximumf(_to_raw(a), _to_raw(b)))


# ---------------------------------------------------------------------------
# Buffer helpers (FlyDSL layout API).
#
# These replace the vendored (rsrc, element_offset) buffer shim with a typed
# buffer tensor built from the tensor's base pointer. The layout uses stride
# (1, 1) so `group_index` counts ELEMENTS (matching the shim's element-offset
# calling convention): fx.slice on the last coord reads/writes `width` contiguous
# elements starting at `group_index`. An i8-typed buffer therefore makes
# `group_index` a byte offset, covering the old offset_is_bytes=True stores.
#
# A buffer bakes in (elem_ty, width); build one per (tensor, dtype, width) used.
# Loads/stores return / accept raw ir.Values, matching the old shim contract so
# the raw-MLIR call sites in this kernel need no rewrapping.
# ---------------------------------------------------------------------------
def _make_buffer(tensor, elem_ty, width=1, *, max_size=True, num_records_bytes=None):
    alignment = max(1, elem_ty.width * width // 8)
    ptr_ty = fx.PointerType.get(elem_ty.ir_type, fx.AddressSpace.Global, alignment)
    base = fx.inttoptr(ptr_ty, fx.Int64(fx.ptrtoint(fx.get_iter(tensor))))
    view = fx.Tensor(fx.make_view(base, fx.make_layout((width, 1), (1, 1))))
    return fx.rocdl.make_buffer_tensor(
        view, max_size=max_size, num_records_bytes=num_records_bytes
    )


def _buffer_load(buffer, group_index, elem_ty, width=1, cache_modifier=0):
    atom = fx.make_copy_atom(
        fx.rocdl.BufferCopy(elem_ty.width * width, cache_modifier), elem_ty
    )
    fragment = fx.make_rmem_tensor(width, elem_ty)
    fx.copy(atom, fx.slice(buffer, (None, group_index)), fragment)
    value = fx.Vector(fragment.load())
    return _to_raw(value[0]) if width == 1 else _to_raw(value)


def _buffer_store(buffer, group_index, value, elem_ty, width=1, cache_modifier=0):
    atom = fx.make_copy_atom(
        fx.rocdl.BufferCopy(elem_ty.width * width, cache_modifier), elem_ty
    )
    fragment = fx.make_rmem_tensor(width, elem_ty)
    fragment.store(
        fx.Vector.from_elements([value], elem_ty)
        if width == 1
        else fx.Vector(value)
    )
    fx.copy(atom, fragment, fx.slice(buffer, (None, group_index)))


def _make_rsrc(tensor, max_size=True):
    """Raw AMD buffer descriptor (``!llvm.ptr<8>``) for a tensor, dtype-agnostic.

    The shared nm-asm scatter emitter (byte-identical across CSA / HCA /
    gfx1250) takes a bare rsrc and drives the ROCDL raw-buffer store itself, so
    that path stays on the descriptor form rather than the fx.copy surface.
    """
    return fx.rocdl.get_buffer_rsrc(
        fx.get_iter(fx.rocdl.make_buffer_tensor(tensor, max_size))
    )


def _store_dwords_at_byte(byte_buf, byte_off, dwords):
    """Store the i32 values ``dwords`` (a single ir.Value or a list) at
    ``byte_off`` (in BYTES) through an i8 buffer, reinterpreting the payload as an
    i8 vector.

    Replaces the old offset_is_bytes=True dword stores: the i8 buffer's element
    index IS the byte offset (no scaling), while the BufferCopy width stays a
    whole number of dwords (emits buffer_store_dword{,xN}).
    """
    dw_list = dwords if isinstance(dwords, (list, tuple)) else [dwords]
    v_i8 = fx.Vector.from_elements(list(dw_list), fx.Int32).bitcast(fx.Int8)
    _buffer_store(byte_buf, byte_off, v_i8, fx.Int8, 4 * len(dw_list))


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

# FP4 (E2M1) MX block-scale group size: 32 elements share one e8m0 scale byte.
# Matches the DSv4 KV path (dsv4_rotate_quant.cu) and the MXFP4 / mfma_scale
# f8f6f4 convention (4 e8m0 bytes cover 4x32=128 FP4 elements per k_tile).
_FP4_GROUP_SIZE = 32
# FP4 preshuffle k_tile = 128 elements; each holds 4 groups of 32 (= 16 bytes/group).
_FP4_K_TILE = 128


# ============================================================================
# scf helpers (copied verbatim from moe_gemm_2stage.py -- too small to share)
# ============================================================================


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
    use_ue8m0: bool,
    preshuffle: bool,
    rms_weight_is_bf16: bool,
    rms_eps: float,
    enable_prefetch_input: bool = True,
    # Single quant source of truth: "none" | "per_row_fp8" (indexer) |
    # "group_fp8" (CSA/HCA Main nm-asm) | "fp4". `quant`/`quant_fp4`/`nm_asm`
    # are derived from it below.
    quant_mode: str = "none",
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
      - has_block_table: False → skip cache scatter (warmup path)
      - quant_mode: single quant selector (the booleans below are derived):
          "none"        → bf16 paged write (Main)            → quant=False
          "per_row_fp8" → FP8 e4m3 per-row scale (Indexer)   → quant=True
          "group_fp8"   → FP8 1xG group scale (Main nm-asm)  → quant=True, nm_asm
          "fp4"         → FP4 (E2M1) per-group(32) e8m0 scale → quant=True, quant_fp4
      - use_ue8m0: only for fp8 (round scale to power-of-2); the FP4 path
        always uses the MX RoundUp e8m0 scale regardless.
      - preshuffle: only when quant (MFMA 16x16 tile / FP4 KV tile layout)
      - enable_prefetch_input: True → Phase 2 carries k+1 loads through
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

    # Derive the quant booleans from the single `quant_mode` source of truth.
    quant = quant_mode != "none"
    quant_fp4 = quant_mode == "fp4"
    # FP8 1xG e8m0 group-quant geometry (quant_mode=="group_fp8" only): nope region split
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
    if quant_fp4:
        # FP4 KV preshuffle: k_tile = 128 elems → 4 groups of 32; data tile is
        # [..., kv_block_size, 16] bytes (16 bytes = 32 fp4). Require D a
        # multiple of 128 and k_per_block a multiple of the 16-token tile.
        assert not (quant and not quant_fp4), "internal: fp4/fp8 are exclusive"
        assert D % _FP4_K_TILE == 0, f"FP4 requires D%128==0, got D={D}"
        assert D % _FP4_GROUP_SIZE == 0, f"FP4 requires D%32==0, got D={D}"
        assert (
            _FP4_GROUP_SIZE % VEC == 0
        ), f"FP4 group {_FP4_GROUP_SIZE} must be a multiple of VEC={VEC}"
        if preshuffle:
            assert k_per_block % _PRESHUFFLE_TILE == 0

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
            elif quant_fp4:
                _name_parts.append("fp4")
            elif use_ue8m0:
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
            return fx.rocdl.exp2(f32, x * c_log2e)

        def wave_reduce_add(x):
            """Butterfly sum across wave64."""
            w = fx.Float32(x)
            for sh_exp in range_constexpr(log2_block):
                off = BLOCK_THREADS // (2 << sh_exp)
                w = w + fx.Float32(w.shuffle_xor(off, BLOCK_THREADS))
            return _to_raw(w)

        def wave_reduce_max(x):
            """Butterfly max across wave64 (used by quant path)."""
            w = fx.Float32(x)
            for sh_exp in range_constexpr(log2_block):
                off = BLOCK_THREADS // (2 << sh_exp)
                w = w.maximumf(fx.Float32(w.shuffle_xor(off, BLOCK_THREADS)))
            return _to_raw(w)

        # ---- Step 1: load plan row (single dwordx4) ----
        # plan layout: each row = 4 contiguous i32 [ragged_id, batch_id, position, window_len].
        # Fuse the 4 scalar loads into one buffer_load_dwordx4 + 4 extracts --
        # saves 3 buffer-load instructions per program (visible at small N
        # where total program count is low).
        plan_buf = _make_buffer(plan, fx.Int32, 4)
        plan_base = _to_raw(fx.Int32(pid) * fx.Int32(4))
        plan_vec = _buffer_load(plan_buf, plan_base, fx.Int32, 4)
        ragged_id = _velem(plan_vec, 0)
        batch_id = _velem(plan_vec, 1)
        position = _velem(plan_vec, 2)
        window_len = _velem(plan_vec, 3)

        # ---- Step 2: sentinel-skip ----
        # Sentinel-skip: run the whole body only for position >= 0. A bare
        # `if cond: return` does NOT early-exit under the tracer (tail still
        # runs with stale values -> OOB), so the body is a closure invoked
        # under a runtime `if`: the rewriter sees an opaque call (no state to
        # thread) and lowers it to scf.if, guarding every store inside.
        def _body():
            # ---- Step 3: per-seq state slot ----
            slot_map_buf = _make_buffer(state_slot_mapping, fx.Int32)
            slot = _buffer_load(slot_map_buf, batch_id, fx.Int32)

            # ---- Step 4: per-thread element-range bookkeeping ----
            # This thread owns columns [tid*VEC, tid*VEC+VEC) of BLOCK_D.
            tid_x_vec = _to_raw(fx.Int32(tid) * fx.Int32(VEC))

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

                    m_old_f = fx.Float32(m_old)
                    m_new = _max_raw(m_old_f, score)
                    is_first = _oeq_raw(m_old_f, c_neg_inf)
                    # BOUNDARY: non-fastmath subf by design (fx `-` would promote
                    # to fast and let the backend contract exp args into FMAs).
                    scale_active = fexp_f32(arith.subf(m_old, _to_raw(m_new)))
                    scale_v = fx.Float32(is_first.select(c_zero_f32, scale_active))
                    # BOUNDARY: non-fastmath subf by design.
                    wk_active = fexp_f32(arith.subf(score, _to_raw(m_new)))
                    if const_expr(score_can_be_neg_inf):
                        is_pad_score = _oeq_raw(score, c_neg_inf)
                        w_k = fx.Float32(is_pad_score.select(c_zero_f32, wk_active))
                    else:
                        w_k = fx.Float32(wk_active)
                    new_m.append(_to_raw(m_new))
                    new_kv.append(
                        _to_raw(
                            fx.Float32(kv_old) * scale_v + w_k * fx.Float32(kv_v)
                        )
                    )
                    new_w.append(_to_raw(fx.Float32(w_old) * scale_v + w_k))
                return new_m, new_kv, new_w

            def _load_bf16_vec_then_f32(buf, off_elems_i32):
                """Load VEC bf16 from byte-aligned dword stream -> fp32 VEC scalars.

                ``buf`` is an i32 buffer over the bf16 tensor (built at width
                BF16_DW). Returns a list of VEC fp32 MLIR values.
                """
                off_dw = fx.Int32(off_elems_i32) >> fx.Int32(1)
                # bf16 VEC = VEC * 2 bytes; for VEC ? {2, 4, 8} that's
                # {4, 8, 16} bytes = {1, 2, 4} dwords.
                dwords = (VEC + 1) // 2  # ceil(VEC*2 / 4)
                if const_expr(dwords == 1):
                    # width-1 load returns a scalar i32; wrap into vec<1xi32>
                    # before bitcasting to vec<2xbf16>.
                    raw_s = _buffer_load(buf, off_dw, fx.Int32)
                    raw = fx.Vector.from_elements([raw_s], fx.Int32)
                else:
                    raw = _buffer_load(buf, off_dw, fx.Int32, dwords)
                vec_bf16 = fx.Vector(raw).bitcast(fx.BFloat16)
                out = []
                for i in range_constexpr(VEC):
                    out.append(_to_raw(fx.BFloat16(_velem(vec_bf16, i)).to(fx.Float32)))
                return out

            def _load_f32_vec(buf, off_elems_i32):
                """Load VEC fp32 from byte-aligned stream -> list of VEC fp32 scalars.

                ``buf`` is an f32 buffer built at width F32_VW. For VEC=2 ->
                dwordx2; VEC=4 -> dwordx4; VEC=8 -> 2x dwordx4 (HW max).
                """
                if const_expr(VEC <= 4):
                    raw = _buffer_load(buf, off_elems_i32, fx.Float32, VEC)
                    return [_to_raw(_velem(raw, i)) for i in range(VEC)]
                else:
                    # VEC == 8 -> 2x dwordx4
                    assert VEC == 8
                    half = VEC // 2
                    r0 = _buffer_load(buf, off_elems_i32, fx.Float32, half)
                    r1 = _buffer_load(
                        buf,
                        fx.Int32(off_elems_i32) + fx.Int32(half),
                        fx.Float32,
                        half,
                    )
                    v0, v1 = fx.Vector(r0), fx.Vector(r1)
                    return [_to_raw(v0[i]) for i in range_constexpr(half)] + [
                        _to_raw(v1[i]) for i in range_constexpr(half)
                    ]

            # Buffers reused across K iters. bf16 tensors are read as i32 dword
            # streams (width BF16_DW); f32 tensors at width F32_VW.
            F32_VW = VEC if VEC <= 4 else VEC // 2
            BF16_DW = (VEC + 1) // 2
            kv_in_buf = _make_buffer(kv_in, fx.Int32, BF16_DW)
            score_in_buf = _make_buffer(score_in, fx.Int32, BF16_DW)
            kv_state_buf = _make_buffer(kv_state, fx.Float32, F32_VW)
            score_state_buf = _make_buffer(score_state, fx.Float32, F32_VW)
            ape_buf = _make_buffer(ape, fx.Float32, F32_VW)

            def _col_off_for_k(k_static_val):
                """Compute col_off ? {0, D} for OVERLAP (==head_dim when k >= RATIO),
                or constant 0 for HCA (no overlap).

                ``k_static_val`` may be a Python int (constexpr) or an MLIR i32 value.
                """
                if const_expr(not overlap):
                    return _to_raw(fx.Int32(0))
                if const_expr(isinstance(k_static_val, int)):
                    return _to_raw(fx.Int32(D if k_static_val >= ratio else 0))
                # Dynamic: (k >= RATIO) ? D : 0  via select
                is_b = fx.Int32(k_static_val) >= fx.Int32(ratio)
                return _to_raw(is_b.select(fx.Int32(D), fx.Int32(0)))

            # ---- Step 6: Phase 1 -- state cache loop (dynamic bound = window_len) ----
            # window_len ? [0, K]. When 0, the loop is a no-op.
            c_K_m1 = arith.constant(K - 1, type=i32)
            c_state_size = arith.constant(state_size, type=i32)

            for k_static, state in range(0, _to_raw(window_len), 1, init=init_state):
                m_lane, kv_lane, w_lane = _split_state(state)

                k_i32 = _to_raw(fx.Int32(k_static))
                s = fx.Int32(position) - fx.Int32(c_K_m1) + fx.Int32(k_i32)
                is_pad = s < fx.Int32(0)
                s_safe = is_pad.select(fx.Int32(0), s)
                ring = _to_raw(s_safe % fx.Int32(c_state_size))
                col_off = _col_off_for_k(k_i32)

                base_kv_off = _to_raw(
                    fx.Int32(slot) * fx.Int32(kv_state_slot_stride)
                    + fx.Int32(ring) * fx.Int32(kv_state_pos_stride)
                    + fx.Int32(col_off)
                    + fx.Int32(tid_x_vec)
                )
                base_sc_off = _to_raw(
                    fx.Int32(slot) * fx.Int32(score_state_slot_stride)
                    + fx.Int32(ring) * fx.Int32(score_state_pos_stride)
                    + fx.Int32(col_off)
                    + fx.Int32(tid_x_vec)
                )

                kv_v_lane = _load_f32_vec(kv_state_buf, base_kv_off)
                sc_v_lane = _load_f32_vec(score_state_buf, base_sc_off)

                sc_pad_lane = []
                for i in range_constexpr(VEC):
                    sc_pad_lane.append(
                        _to_raw(is_pad.select(c_neg_inf, sc_v_lane[i]))
                    )

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
                ape_row = _to_raw(fx.Int32(k_i32) % fx.Int32(ratio))
                in_row = _to_raw(
                    fx.Int32(ragged_id) - (fx.Int32(c_K_m1) - fx.Int32(k_i32))
                )
                return col_off, in_row, ape_row

            def _phase2_issue_loads(k_i32):
                """Issue kv_in / score_in / ape loads for Phase 2 iter k.

                Returns (kv_lane, score_a_lane, ape_v_lane) -- three lists of
                VEC fp32 scalars. buffer_load with max_size resources is OOB-
                safe (returns 0), so callers may speculatively issue at
                k = K (one past the last legal iter) for prefetch tails.
                """
                col_off, in_row, ape_row = _phase2_offsets(k_i32)
                base_in_off = _to_raw(
                    fx.Int32(in_row) * fx.Int32(kv_in_row_stride)
                    + fx.Int32(col_off)
                    + fx.Int32(tid_x_vec)
                )
                base_sc_off = _to_raw(
                    fx.Int32(in_row) * fx.Int32(score_in_row_stride)
                    + fx.Int32(col_off)
                    + fx.Int32(tid_x_vec)
                )
                base_ape_off = _to_raw(
                    fx.Int32(ape_row) * fx.Int32(DIM_FULL)
                    + fx.Int32(col_off)
                    + fx.Int32(tid_x_vec)
                )
                kv = _load_bf16_vec_then_f32(kv_in_buf, base_in_off)
                sc = _load_bf16_vec_then_f32(score_in_buf, base_sc_off)
                ape = _load_f32_vec(ape_buf, base_ape_off)
                return kv, sc, ape

            if const_expr(not enable_prefetch_input):
                for k_static, state in range(
                    _to_raw(window_len), K, 1, init=phase1_state
                ):
                    m_lane, kv_lane, w_lane = _split_state(state)
                    k_i32 = _to_raw(fx.Int32(k_static))
                    kv_a_lane, score_a_lane, ape_v_lane = _phase2_issue_loads(k_i32)
                    score_k_lane = [
                        _to_raw(fx.Float32(score_a_lane[i]) + fx.Float32(ape_v_lane[i]))
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

                _m_final, kv_final, w_final = _split_state(phase2_state)
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
                k_prologue = _to_raw(_minsi(fx.Int32(window_len), fx.Int32(K - 1)))
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
                    k_next = _to_raw(fx.Int32(k_i32) + fx.Int32(1))
                    nxt_kv, nxt_sc, nxt_ape = _phase2_issue_loads(k_next)

                    score_k_lane = [
                        _to_raw(fx.Float32(pre_sc[i]) + fx.Float32(pre_ape[i]))
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

                # Tail iter at k=K-1. When window_len == K phase 2 is empty and
                # the accumulator passes through phase1_state. Both arms are pure
                # arithmetic (no loads), so compute the update unconditionally and
                # per-lane select -- bit-identical to the old value-yielding IfOp.
                is_phase2_nonempty = fx.Int32(window_len) < K
                m_lane_t = list(loop_final[0:VEC])
                kv_lane_t = list(loop_final[VEC : 2 * VEC])
                w_lane_t = list(loop_final[2 * VEC : 3 * VEC])
                pre_kv_t = list(loop_final[3 * VEC : 4 * VEC])
                pre_sc_t = list(loop_final[4 * VEC : 5 * VEC])
                pre_ape_t = list(loop_final[5 * VEC : 6 * VEC])
                score_k_lane_t = [
                    _to_raw(fx.Float32(pre_sc_t[i]) + fx.Float32(pre_ape_t[i]))
                    for i in range(VEC)
                ]
                _, new_kv_t, new_w_t = _online_softmax_update(
                    m_lane_t,
                    kv_lane_t,
                    w_lane_t,
                    score_k_lane_t,
                    pre_kv_t,
                    score_can_be_neg_inf=False,
                )
                kv_p1 = list(phase1_state[VEC : 2 * VEC])
                w_p1 = list(phase1_state[2 * VEC : 3 * VEC])
                kv_final = [
                    is_phase2_nonempty.select(new_kv_t[i], kv_p1[i]).ir_value()
                    for i in range(VEC)
                ]
                w_final = [
                    is_phase2_nonempty.select(new_w_t[i], w_p1[i]).ir_value()
                    for i in range(VEC)
                ]

            # ---- Step 8: compressed = kv_acc / w_acc (per-lane) ----
            comp_lane = []
            for i in range_constexpr(VEC):
                rcp_w = fx.rocdl.rcp(f32, w_final[i])
                comp_lane.append(_to_raw(fx.Float32(kv_final[i]) * fx.Float32(rcp_w)))

            # ---- Step 9: RMSNorm (fp32) -- sum-of-squares across wave ----
            sq_local = fx.Float32(0.0)
            for i in range_constexpr(VEC):
                cl = fx.Float32(comp_lane[i])
                sq_local = sq_local + cl * cl
            sq_full = wave_reduce_add(_to_raw(sq_local))
            var = fx.Float32(sq_full) * fx.Float32(c_inv_D)
            rrms = fmath.rsqrt(_to_raw(var + fx.Float32(c_eps)), fastmath=fm_fast)

            # rms_weight: per-channel; this thread loads VEC values at tid*VEC.
            # Production atom passes bf16 (the param is cast at model load);
            # tests may pass fp32. Constexpr branch picks the right load.
            if const_expr(rms_weight_is_bf16):
                rmsw_buf = _make_buffer(rms_weight, fx.Int32, BF16_DW)
                rmsw_lane = _load_bf16_vec_then_f32(rmsw_buf, tid_x_vec)
            else:
                rmsw_buf = _make_buffer(rms_weight, fx.Float32, F32_VW)
                rmsw_lane = _load_f32_vec(rmsw_buf, tid_x_vec)

            rrms_f = fx.Float32(rrms)
            normed_lane = [
                _to_raw(fx.Float32(comp_lane[i]) * rrms_f * fx.Float32(rmsw_lane[i]))
                for i in range(VEC)
            ]

            # ---- Step 10: GPT-J RoPE on RD tail ----
            # is_rope = tid >= ROPE_THREAD_LO. RoPE applies only to those threads.
            comp_pos_i32 = _to_raw(
                (fx.Int32(position) // fx.Int32(ratio)) * fx.Int32(ratio)
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
            cos_buf = _make_buffer(cos_cache, fx.BFloat16, PAIRS_PER_THREAD)
            sin_buf = _make_buffer(sin_cache, fx.BFloat16, PAIRS_PER_THREAD)
            cos_row_base = _to_raw(fx.Int32(comp_pos_i32) * fx.Int32(RD // 2))

            is_rope_t = fx.Int32(tid) >= fx.Int32(ROPE_THREAD_LO)
            # rope_rel may be negative for NOPE threads; clamp to 0 so the
            # cos/sin load address is in-bounds (the loaded value is unused
            # because is_rope_t = false).
            rope_rel = _maxsi(fx.Int32(tid) - fx.Int32(ROPE_THREAD_LO), fx.Int32(0))
            cs_lo = _to_raw(rope_rel * fx.Int32(PAIRS_PER_THREAD))
            cs_off = _to_raw(fx.Int32(cos_row_base) + fx.Int32(cs_lo))

            if const_expr(PAIRS_PER_THREAD == 1):
                cos_b = _buffer_load(cos_buf, cs_off, fx.BFloat16)
                sin_b = _buffer_load(sin_buf, cs_off, fx.BFloat16)
                cos_vals = [_to_raw(fx.BFloat16(cos_b).to(fx.Float32))]
                sin_vals = [_to_raw(fx.BFloat16(sin_b).to(fx.Float32))]
            else:
                cos_vec = _buffer_load(cos_buf, cs_off, fx.BFloat16, PAIRS_PER_THREAD)
                sin_vec = _buffer_load(sin_buf, cs_off, fx.BFloat16, PAIRS_PER_THREAD)
                cos_vals = [
                    _to_raw(fx.BFloat16(_velem(cos_vec, i)).to(fx.Float32))
                    for i in range(PAIRS_PER_THREAD)
                ]
                sin_vals = [
                    _to_raw(fx.BFloat16(_velem(sin_vec, i)).to(fx.Float32))
                    for i in range(PAIRS_PER_THREAD)
                ]

            # GPT-J pair rotation per VEC pair, then select rotated vs pass-through.
            rotated_lane = list(normed_lane)
            for k in range_constexpr(PAIRS_PER_THREAD):
                e = fx.Float32(normed_lane[2 * k])
                o = fx.Float32(normed_lane[2 * k + 1])
                c = fx.Float32(cos_vals[k])
                s = fx.Float32(sin_vals[k])
                # BOUNDARY: non-fastmath subf by design -- fx `-` would promote to
                # fast and let the backend contract e*c - o*s into an FMA (matches
                # HEAD / fused_compress_attn_common).
                new_e = arith.subf(_to_raw(e * c), _to_raw(o * s))
                new_o = _to_raw(e * s + o * c)
                rotated_lane[2 * k] = new_e
                rotated_lane[2 * k + 1] = new_o

            out_lane = [
                _to_raw(is_rope_t.select(rotated_lane[i], normed_lane[i]))
                for i in range_constexpr(VEC)
            ]

            # ---- Step 11: Scatter (only when has_block_table) ----
            if const_expr(has_block_table):
                # ci = position // ratio; block_in_seq = ci // k_per_block;
                # slot_in_block = ci % k_per_block.
                ci = fx.Int32(position) // fx.Int32(ratio)
                block_in_seq = _to_raw(ci // fx.Int32(k_per_block))
                slot_in_block = _to_raw(ci % fx.Int32(k_per_block))

                # physical_block = block_table[batch_id, block_in_seq]
                bt_buf = _make_buffer(block_table, fx.Int32)
                bt_off = _to_raw(
                    fx.Int32(batch_id) * fx.Int32(block_table_seq_stride)
                    + fx.Int32(block_in_seq)
                )
                physical_block = _buffer_load(bt_buf, bt_off, fx.Int32)

                if const_expr(not quant):
                    # BF16 paged write. kv_cache layout: [NB, k_per_block, D].
                    # cache_addr = physical_block * block_stride + slot_in_block * token_stride + tid*VEC
                    # (strides are in bf16 elements; caller passes elements.)
                    cache_off = (
                        fx.Int32(physical_block) * fx.Int32(kv_cache_block_stride)
                        + fx.Int32(slot_in_block) * fx.Int32(kv_cache_token_stride)
                        + fx.Int32(tid_x_vec)
                    )
                    # Build a per-block GTensor and store VEC bf16 via dword path.
                    # bf16 VEC ? {2, 4, 8} = {4, 8, 16} bytes = {1, 2, 4} dwords.
                    out_vec_t = T.vec(VEC, T.bf16)
                    raw_vec = fx.Vector.from_elements(out_lane, fx.Float32)
                    bf16_vec = raw_vec.truncf(out_vec_t)
                    # cache_off is in bf16 elements; convert to dword for the i32-vec store.
                    cache_off_dw = _to_raw(cache_off >> fx.Int32(1))
                    dwords = (VEC + 1) // 2
                    bf16_as_i32 = fx.Vector(bf16_vec).bitcast(fx.Int32)
                    if const_expr(dwords == 1):
                        # vec<1xi32> -> scalar i32 store
                        scalar_i32 = _velem(bf16_as_i32, 0)
                        out_buf = _make_buffer(kv_cache, fx.Int32)
                        _buffer_store(out_buf, cache_off_dw, scalar_i32, fx.Int32)
                    else:
                        out_buf = _make_buffer(kv_cache, fx.Int32, dwords)
                        _buffer_store(
                            out_buf, cache_off_dw, bf16_as_i32, fx.Int32, dwords
                        )
                elif const_expr(nm_asm):
                    # -- group_fp8 (V4 nm-asm): nope fp8 + inline dup e8m0; rope bf16
                    # -> separate k_rope_buff. Shared emitter (byte-identical to HCA). --
                    _nm_cache_base = _to_raw(
                        fx.Int32(physical_block) * fx.Int32(kv_cache_block_stride)
                        + fx.Int32(slot_in_block) * fx.Int32(kv_cache_token_stride)
                    )
                    _nm_krope_base = _to_raw(
                        fx.Int32(physical_block) * fx.Int32(krope_block_stride)
                        + fx.Int32(slot_in_block) * fx.Int32(krope_token_stride)
                    )
                    emit_group_fp8_nm_asm_scatter(
                        normed_lane=normed_lane,
                        rotated_lane=rotated_lane,
                        lane=tid,
                        is_rope_t=is_rope_t,
                        cache_base=_nm_cache_base,
                        out_rsrc=_make_rsrc(kv_cache),
                        krope_base=_nm_krope_base,
                        krope_rsrc=_make_rsrc(k_rope_buff),
                        VEC=VEC,
                        NOPE=NOPE,
                        RTS=RTS,
                        log2_rts=log2_rts,
                        ROPE_THREAD_LO=ROPE_THREAD_LO,
                        wave_width=BLOCK_THREADS,
                        vecVf32=vecVf32,
                        fm_fast=fm_fast,
                    )
                elif const_expr(not quant_fp4):
                    # ── QUANT=1: FP8 per-row scaled write + fp32 scale ──
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
                    c_fp8_max = fx.Float32(fp8_max)
                    c_neg_fp8_max = fx.Float32(-fp8_max)
                    c_safety_floor = fx.Float32(1e-4)
                    c_inv_fp8_max = fx.Float32(1.0 / fp8_max)

                    # (a) per-lane amax
                    am_local = fx.Float32(0.0)
                    for i in range_constexpr(VEC):
                        am_local = am_local.maximumf(fx.Float32(fmath.absf(out_lane[i])))
                    amax = fx.Float32(wave_reduce_max(_to_raw(am_local)))
                    am_safe = amax.maximumf(c_safety_floor)

                    # (b) scale = am_safe / FP8_MAX, optionally ceil-pow2
                    scale_raw = am_safe * c_inv_fp8_max
                    if const_expr(use_ue8m0):
                        # ceil-to-pow2 via bit trick: add 0x7FFFFF to mantissa,
                        # mask off mantissa. If mantissa was 0, exp unchanged;
                        # else exp += 1.
                        scale_i32 = scale_raw.bitcast(fx.Int32)
                        bits_up = (scale_i32 + fx.Int32(0x7FFFFF)) & fx.Int32(0xFF800000)
                        scale_v = bits_up.bitcast(fx.Float32)
                    else:
                        scale_v = scale_raw

                    # (c) inv_scale via rcp.f32
                    inv_scale = fx.Float32(fx.rocdl.rcp(f32, _to_raw(scale_v)))

                    # (d) per-lane fp8 cast: clamp + NaN guard
                    #     NaN guard: cvt_pk_fp8_f32 on fnuz returns 0x80 (NaN)
                    #     for inputs that round to negative zero. Clamp small
                    #     negatives v ? (-2^-8, 0) to +0 first. Matches
                    #     _store_fp8_packed in qk_norm_rope_quant.
                    c_neg_uf = fx.Float32(-(2.0**-8))
                    c_zero = fx.Float32(0.0)
                    fp8_inputs = []
                    for i in range_constexpr(VEC):
                        v = fx.Float32(out_lane[i]) * inv_scale
                        # clamp to [-FP8_MAX, +FP8_MAX]. arith.minimumf kept raw:
                        # ArithValue has no .minimumf at this flydsl version.
                        v = fx.Float32(
                            arith.minimumf(
                                _to_raw(v.maximumf(c_neg_fp8_max)), _to_raw(c_fp8_max)
                            )
                        )
                        # NaN guard
                        is_tn = (v < c_zero) & (v > c_neg_uf)
                        fp8_inputs.append(_to_raw(is_tn.select(c_zero, v)))

                    # (e) pack VEC fp32 -> VEC fp8 bytes inside i32 seed
                    # VEC=2: 1 cvt_pk_fp8_f32 call (places 2 bytes at index 0)
                    # VEC=4: 2 calls (places 4 bytes at indices 0, 1)
                    # VEC=8: 4 calls (places 8 bytes at indices 0..3 of 2 i32s)
                    c_p0 = _to_raw(fx.Int32(0))
                    if const_expr(VEC == 2):
                        # Result in low 16 bits of i32
                        # BOUNDARY: hand-packed rocdl.cvt_pk_fp8_f32 (raw operands).
                        pk = rocdl.cvt_pk_fp8_f32(
                            i32, fp8_inputs[0], fp8_inputs[1], c_p0, 0
                        )
                        # Pair cooperation: even tid stores dword with peer.
                        # peer_pack (in low 16 bits) shifted to high 16 bits.
                        pk = fx.Int32(pk)
                        peer_pk = fx.Int32(pk.shuffle_xor(1, BLOCK_THREADS))
                        dword = pk | (peer_pk << fx.Int32(16))
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
                        # BOUNDARY: hand-packed rocdl.cvt_pk_fp8_f32 (raw operands).
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
                    dwords_per_store = 1 if VEC != 8 else 2
                    out_byte_buf = _make_buffer(
                        kv_cache, fx.Int8, 4 * dwords_per_store
                    )
                    block_byte_base = fx.Int32(physical_block) * fx.Int32(
                        kv_cache_block_stride
                    )

                    if const_expr(preshuffle):
                        # MFMA 16x16 tile layout
                        # offset = block_base
                        #        + token_tile_id * (TILE * D)
                        #        + col_tile_id * (TILE * TILE)
                        #        + token_in_tile * TILE
                        #        + col_in_tile
                        sib = fx.Int32(slot_in_block)
                        token_tile_id = sib // fx.Int32(_PRESHUFFLE_TILE)
                        token_in_tile = sib % fx.Int32(_PRESHUFFLE_TILE)
                        # d = tid * VEC; col_tile_id = d // TILE; col_in_tile = d % TILE
                        d_for_tid = fx.Int32(tid) * fx.Int32(VEC)
                        col_tile_id = d_for_tid // fx.Int32(_PRESHUFFLE_TILE)
                        col_in_tile = d_for_tid % fx.Int32(_PRESHUFFLE_TILE)
                        in_block_off = (
                            token_tile_id * fx.Int32(_PRESHUFFLE_TILE * D)
                            + col_tile_id * fx.Int32(_PRESHUFFLE_TILE * _PRESHUFFLE_TILE)
                            + token_in_tile * fx.Int32(_PRESHUFFLE_TILE)
                            + col_in_tile
                        )
                    else:
                        # Linear layout: phys * block_stride + slot * D + tid * VEC
                        in_block_off = fx.Int32(slot_in_block) * fx.Int32(
                            D
                        ) + fx.Int32(tid) * fx.Int32(VEC)

                    byte_off = _to_raw(block_byte_base + in_block_off)

                    if const_expr(VEC == 2):
                        # Only even tid stores (its dword covers peer's bytes too).
                        if (tid & 1) == 0:
                            _store_dwords_at_byte(out_byte_buf, byte_off, _to_raw(dword))
                    elif const_expr(VEC == 4):
                        _store_dwords_at_byte(out_byte_buf, byte_off, dword)
                    else:
                        # VEC == 8: store 2 dwords (8 bytes) via vec<2xi32>
                        _store_dwords_at_byte(
                            out_byte_buf, byte_off, [dword[0], dword[1]]
                        )

                    # (f) lane-0 writes fp32 scale at cache_scale[phys, slot]
                    if tid == 0:
                        cs_buf = _make_buffer(cache_scale, fx.Float32)
                        cs_off = _to_raw(
                            fx.Int32(physical_block) * fx.Int32(cache_scale_block_stride)
                            + fx.Int32(slot_in_block)
                        )
                        _buffer_store(cs_buf, cs_off, scale_v, fx.Float32)
                else:
                    # ── QUANT=1, FP4: per-group(32) e8m0 scale + E2M1 write ──
                    # Mirrors dsv4_rotate_quant.cu's FP4 KV writer + the shared
                    # FlyDSL IR builders (emit_mx_e8m0_scale / emit_f32_to_e2m1,
                    # used by silu_and_mul_fq). Each group of 32 elements shares
                    # one e8m0 byte; NTG = 32//VEC lanes cooperate per group.
                    #   (a) per-lane amax over VEC → group-reduce-max over NTG
                    #   (b) e8m0 = ceil_pow2(amax/6) (MX RoundUp); quant_scale =
                    #       (254 - e8m0) << 23
                    #   (c) per-element E2M1 nibble, pack VEC/2 bytes
                    #   (d) preshuffle (FP4 KV tile) or linear byte write
                    #   (e) group-rep lane writes the e8m0 scale byte
                    NTG = _FP4_GROUP_SIZE // VEC
                    LOG2_NTG = int(math.log2(NTG))
                    PACKED_BYTES = VEC // 2
                    K_TILES = D // _FP4_K_TILE
                    KVBS = k_per_block
                    # smallest-normal * fp4_max floor — guards all-zero groups,
                    # matches dsv4_rotate_quant.cu eps_amax (bit-exact w/ ref).
                    c_eps_amax = fx.Float32(6.0 * float.fromhex("0x1p-126"))

                    # (a) per-lane amax, then butterfly group-reduce over NTG lanes.
                    am_grp = fx.Float32(0.0)
                    for i in range_constexpr(VEC):
                        am_grp = am_grp.maximumf(fx.Float32(fmath.absf(out_lane[i])))
                    for sh_exp in range_constexpr(LOG2_NTG):
                        off = NTG // (2 << sh_exp)
                        am_grp = am_grp.maximumf(
                            fx.Float32(am_grp.shuffle_xor(off, BLOCK_THREADS))
                        )
                    am_safe = am_grp.maximumf(c_eps_amax)

                    # (b) MX RoundUp e8m0 + multiplicative quant scale.
                    # BOUNDARY: emit_mx_e8m0_scale is a raw IR builder; feed a
                    # raw ArithValue.
                    e8m0 = emit_mx_e8m0_scale(
                        ArithValue(am_safe.ir_value()),
                        mode=_MxRoundInt.RoundUp,
                        dtype=_MxDtypeInt.FP4_E2M1,
                    )
                    quant_exp = fx.Int32(254) - fx.Int32(e8m0)
                    quant_scale = (quant_exp << fx.Int32(23)).bitcast(fx.Float32)

                    # (c) per-element E2M1 nibble, pack VEC/2 bytes.
                    # BOUNDARY: emit_f32_to_e2m1 is a raw IR builder (raw operand).
                    nibs = [
                        emit_f32_to_e2m1(_to_raw(fx.Float32(out_lane[i]) * quant_scale))
                        for i in range_constexpr(VEC)
                    ]

                    out_byte_buf = _make_buffer(kv_cache, fx.Int8)
                    # packed byte index within the row = (tid*VEC) / 2.
                    packed_start = fx.Int32(tid_x_vec) >> fx.Int32(1)
                    # flat paged slot (for the linear fallback).
                    flat_slot = fx.Int32(physical_block) * fx.Int32(
                        k_per_block
                    ) + fx.Int32(slot_in_block)
                    for b in range_constexpr(PACKED_BYTES):
                        byte_val = fx.Int32(nibs[2 * b]) | (
                            fx.Int32(nibs[2 * b + 1]) << fx.Int32(4)
                        )
                        packed_idx = packed_start + fx.Int32(b)
                        if const_expr(preshuffle):
                            # FP4 KV preshuffle [NB, k_tiles, 4, kvbs, 16] u8.
                            k_tile = packed_idx // fx.Int32(64)
                            rem = packed_idx % fx.Int32(64)
                            group4 = rem // fx.Int32(16)
                            sub16 = rem % fx.Int32(16)
                            byte_off = _to_raw(
                                fx.Int32(physical_block) * fx.Int32(K_TILES * 4 * KVBS * 16)
                                + k_tile * fx.Int32(4 * KVBS * 16)
                                + group4 * fx.Int32(KVBS * 16)
                                + fx.Int32(slot_in_block) * fx.Int32(16)
                                + sub16
                            )
                        else:
                            byte_off = _to_raw(
                                flat_slot * fx.Int32(D // 2) + packed_idx
                            )
                        # i8 nibble byte: element offset == byte offset.
                        _buffer_store(
                            out_byte_buf,
                            byte_off,
                            _to_raw(byte_val.to(fx.Int8)),
                            fx.Int8,
                        )

                    # (e) group-rep lane writes the e8m0 scale byte.
                    scale_group_idx = fx.Int32(tid_x_vec) // fx.Int32(_FP4_GROUP_SIZE)
                    if tid % NTG == 0:
                        cs_buf = _make_buffer(cache_scale, fx.Int8)
                        if const_expr(preshuffle):
                            # scale [NB, k_tiles, 4, kvbs] u8, with the slot axis
                            # INTERLEAVED so the mqa-logits reader's packed-dword
                            # load (4 nt-bytes adjacent) is contiguous:
                            #   sflat = (slot % 16) * KVS_NTPW + (slot // 16)
                            # (KVS_NTPW == 4). Matches the op-test reference
                            # writer `indexer_k_fp4_paged_preshuffle` and the
                            # packed N_PHYS==1 readers in pa_mqa_logits_fp4*.
                            k_tile_s = scale_group_idx // fx.Int32(4)
                            group4_s = scale_group_idx % fx.Int32(4)
                            sflat = (fx.Int32(slot_in_block) % fx.Int32(16)) * fx.Int32(
                                4
                            ) + (fx.Int32(slot_in_block) // fx.Int32(16))
                            cs_off = _to_raw(
                                fx.Int32(physical_block) * fx.Int32(K_TILES * 4 * KVBS)
                                + k_tile_s * fx.Int32(4 * KVBS)
                                + group4_s * fx.Int32(KVBS)
                                + sflat
                            )
                        else:
                            cs_off = _to_raw(
                                flat_slot * fx.Int32(D // _FP4_GROUP_SIZE)
                                + scale_group_idx
                            )
                        _buffer_store(
                            cs_buf,
                            cs_off,
                            _to_raw(fx.Int32(e8m0).to(fx.Int8)),
                            fx.Int8,
                        )  # e8m0 uint8
            # else: warmup — no scatter, just consume compute.

        if fx.Int32(position) >= 0:
            _body()

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
        idx_p = fx.Int64(plan_capacity)
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
    use_ue8m0: bool,
    preshuffle: bool,
    rms_weight_is_bf16: bool,
    rms_eps: float,
    # Single quant source of truth: "none" | "per_row_fp8" | "group_fp8" | "fp4".
    # `quant`/`quant_fp4`/`nm_asm` are derived below. group_fp8 (nm-asm) and fp4
    # are both K-split-capable (nm-asm on the CSA Main shape; fp4 on the indexer).
    quant_mode: str = "none",
    quant_group_size: int = 64,
):
    """K-split single-kernel: NW-wave LDS-reduced compress + norm + rope +
    scatter (BF16, FP8, or FP4). Constexpr knobs mirror :func:`_build_kernel`
    minus ``enable_prefetch_input`` (each wave runs so few iters that prefetch
    is moot). The FP8 / FP4 / ue8m0 / preshuffle scatter is emitted in wave 0,
    where ``lid`` (0..63) plays the single-wave ``tid`` role — pair-coop
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

    # Derive the quant booleans from the single `quant_mode` source of truth.
    quant = quant_mode != "none"
    quant_fp4 = quant_mode == "fp4"

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
    if quant_fp4:
        assert quant, "internal: quant_fp4 requires quant=True"
        assert D % _FP4_K_TILE == 0, f"FP4 requires D%128==0, got D={D}"
        assert D % _FP4_GROUP_SIZE == 0, f"FP4 requires D%32==0, got D={D}"
        assert (
            _FP4_GROUP_SIZE % VEC == 0
        ), f"FP4 group {_FP4_GROUP_SIZE} must be a multiple of VEC={VEC}"
        if preshuffle:
            assert k_per_block % _PRESHUFFLE_TILE == 0
    elif quant and preshuffle:
        assert D % _PRESHUFFLE_TILE == 0
        assert k_per_block % _PRESHUFFLE_TILE == 0
    if nm_asm:
        assert quant and not preshuffle, "nm_asm: requires quant=True, preshuffle=False"
        assert (
            NOPE % GROUP_SIZE_Q == 0 and GROUP_SIZE_Q % VEC == 0
        ), f"nm_asm: NOPE={NOPE} % G={GROUP_SIZE_Q} and G % VEC={VEC} must be 0"

    # LDS: 3 fp32 arrays, each NW * D entries.
    LDS_ELEMS = NW * D

    @fx.struct
    class SharedStorage:
        lds_m: fx.Array[fx.Float32, LDS_ELEMS, 16]
        lds_kv: fx.Array[fx.Float32, LDS_ELEMS, 16]
        lds_w: fx.Array[fx.Float32, LDS_ELEMS, 16]

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
        elif quant_fp4:
            _name_parts.append("fp4")
        elif use_ue8m0:
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
        c_eps = arith.constant(rms_eps, type=f32)
        c_inv_D = arith.constant(1.0 / D, type=f32)
        c_log2e = arith.constant(_LOG2E, type=f32)
        c_K_m1 = arith.constant(K - 1, type=i32)
        c_state_size = arith.constant(state_size, type=i32)
        c_D = arith.constant(D, type=i32)

        def fexp_f32(x):
            return fx.rocdl.exp2(f32, x * c_log2e)

        wid = _to_raw(fx.Int32(tid) // fx.Int32(BLOCK_THREADS))  # ? [0, NW)
        lid = _to_raw(fx.Int32(tid) % fx.Int32(BLOCK_THREADS))  # ? [0, 64)

        # ---- plan row (single dwordx4) ----
        plan_buf = _make_buffer(plan, fx.Int32, 4)
        plan_base = _to_raw(fx.Int32(pid) * fx.Int32(4))
        plan_vec = _buffer_load(plan_buf, plan_base, fx.Int32, 4)
        ragged_id = _velem(plan_vec, 0)
        batch_id = _velem(plan_vec, 1)
        position = _velem(plan_vec, 2)
        window_len = _velem(plan_vec, 3)

        # Sentinel-skip: whole body runs only for position >= 0, as a closure
        # under a runtime `if` (see the CSA kernel above for the rationale).
        def _body():
            slot_map_buf = _make_buffer(state_slot_mapping, fx.Int32)
            slot = _buffer_load(slot_map_buf, batch_id, fx.Int32)

            # This lane owns columns [lid*VEC, lid*VEC+VEC) of head_dim.
            lid_x_vec = _to_raw(fx.Int32(lid) * fx.Int32(VEC))

            # Buffers reused across K iters. bf16 tensors are read as i32 dword
            # streams (width BF16_DW); f32 tensors at width F32_VW.
            F32_VW = VEC if VEC <= 4 else VEC // 2
            BF16_DW = (VEC + 1) // 2
            kv_in_buf = _make_buffer(kv_in, fx.Int32, BF16_DW)
            score_in_buf = _make_buffer(score_in, fx.Int32, BF16_DW)
            kv_state_buf = _make_buffer(kv_state, fx.Float32, F32_VW)
            score_state_buf = _make_buffer(score_state, fx.Float32, F32_VW)
            ape_buf = _make_buffer(ape, fx.Float32, F32_VW)

            def _col_off_for_k(k_i32):
                if const_expr(not overlap):
                    return c_zero_i32
                is_b = fx.Int32(k_i32) >= fx.Int32(ratio)
                return _to_raw(is_b.select(c_D, c_zero_i32))

            def _load_f32_vec(buf, off_elems_i32):
                if const_expr(VEC <= 4):
                    raw = _buffer_load(buf, off_elems_i32, fx.Float32, VEC)
                    return [_to_raw(_velem(raw, i)) for i in range(VEC)]
                else:
                    assert VEC == 8
                    half = VEC // 2
                    r0 = _buffer_load(buf, off_elems_i32, fx.Float32, half)
                    r1 = _buffer_load(
                        buf,
                        fx.Int32(off_elems_i32) + fx.Int32(half),
                        fx.Float32,
                        half,
                    )
                    v0, v1 = fx.Vector(r0), fx.Vector(r1)
                    return [_to_raw(v0[i]) for i in range_constexpr(half)] + [
                        _to_raw(v1[i]) for i in range_constexpr(half)
                    ]

            def _load_bf16_vec_then_f32(buf, off_elems_i32):
                off_dw = _to_raw(fx.Int32(off_elems_i32) >> fx.Int32(1))
                dwords = (VEC + 1) // 2
                if const_expr(dwords == 1):
                    raw_s = _buffer_load(buf, off_dw, fx.Int32)
                    raw = fx.Vector.from_elements([raw_s], fx.Int32)
                else:
                    raw = _buffer_load(buf, off_dw, fx.Int32, dwords)
                vec_bf16 = fx.Vector(raw).bitcast(fx.BFloat16)
                out = []
                for i in range_constexpr(VEC):
                    out.append(_to_raw(fx.BFloat16(_velem(vec_bf16, i)).to(fx.Float32)))
                return out

            def _softmax_step(m_lane, kv_lane, w_lane, score_lane, kv_v_lane):
                """Padding-aware per-lane online-softmax update. Phase 2 scores
                are finite, so the is-pad branch is dead code there (compiler
                elides)."""
                new_m, new_kv, new_w = [], [], []
                for i in range_constexpr(VEC):
                    m_old = m_lane[i]
                    score = score_lane[i]
                    m_old_f = fx.Float32(m_old)
                    m_new = _max_raw(m_old_f, score)
                    is_first = _oeq_raw(m_old_f, c_neg_inf)
                    # BOUNDARY: non-fastmath subf by design.
                    scale_active = fexp_f32(arith.subf(m_old, _to_raw(m_new)))
                    scale_v = fx.Float32(is_first.select(c_zero_f32, scale_active))
                    # BOUNDARY: non-fastmath subf by design.
                    wk_active = fexp_f32(arith.subf(score, _to_raw(m_new)))
                    is_pad = _oeq_raw(score, c_neg_inf)
                    w_k = fx.Float32(is_pad.select(c_zero_f32, wk_active))
                    new_m.append(_to_raw(m_new))
                    new_kv.append(
                        _to_raw(
                            fx.Float32(kv_lane[i]) * scale_v
                            + w_k * fx.Float32(kv_v_lane[i])
                        )
                    )
                    new_w.append(_to_raw(fx.Float32(w_lane[i]) * scale_v + w_k))
                return new_m, new_kv, new_w

            def _phase1_loads(k_i32):
                s = fx.Int32(position) - fx.Int32(c_K_m1) + fx.Int32(k_i32)
                is_pad = s < fx.Int32(0)
                s_safe = is_pad.select(fx.Int32(0), s)
                ring = _to_raw(s_safe % fx.Int32(c_state_size))
                col_off = _col_off_for_k(k_i32)
                base_kv = _to_raw(
                    fx.Int32(slot) * fx.Int32(kv_state_slot_stride)
                    + fx.Int32(ring) * fx.Int32(kv_state_pos_stride)
                    + fx.Int32(col_off)
                    + fx.Int32(lid_x_vec)
                )
                base_sc = _to_raw(
                    fx.Int32(slot) * fx.Int32(score_state_slot_stride)
                    + fx.Int32(ring) * fx.Int32(score_state_pos_stride)
                    + fx.Int32(col_off)
                    + fx.Int32(lid_x_vec)
                )
                kv_v = _load_f32_vec(kv_state_buf, base_kv)
                sc_v = _load_f32_vec(score_state_buf, base_sc)
                sc_pad = [
                    _to_raw(is_pad.select(c_neg_inf, sc_v[i])) for i in range(VEC)
                ]
                return kv_v, sc_pad

            def _phase2_loads(k_i32):
                col_off = _col_off_for_k(k_i32)
                ape_row = _to_raw(fx.Int32(k_i32) % fx.Int32(ratio))
                in_row = _maxsi(
                    fx.Int32(ragged_id) - (fx.Int32(c_K_m1) - fx.Int32(k_i32)),
                    fx.Int32(0),
                )
                base_in = _to_raw(
                    in_row * fx.Int32(kv_in_row_stride)
                    + fx.Int32(col_off)
                    + fx.Int32(lid_x_vec)
                )
                base_sc = _to_raw(
                    in_row * fx.Int32(score_in_row_stride)
                    + fx.Int32(col_off)
                    + fx.Int32(lid_x_vec)
                )
                base_ape = _to_raw(
                    fx.Int32(ape_row) * fx.Int32(DIM_FULL)
                    + fx.Int32(col_off)
                    + fx.Int32(lid_x_vec)
                )
                kv = _load_bf16_vec_then_f32(kv_in_buf, base_in)
                sc = _load_bf16_vec_then_f32(score_in_buf, base_sc)
                ape_v = _load_f32_vec(ape_buf, base_ape)
                score = [
                    _to_raw(fx.Float32(sc[i]) + fx.Float32(ape_v[i]))
                    for i in range(VEC)
                ]
                return kv, score

            # ---- this wave's K range [wid*KPW, (wid+1)*KPW), split at window_len ----
            k_start = _to_raw(fx.Int32(wid) * fx.Int32(K_PER_WAVE))
            k_end = _to_raw(fx.Int32(k_start) + fx.Int32(K_PER_WAVE))
            wl = _to_raw(window_len)
            split_lo = _maxsi(fx.Int32(wl), fx.Int32(k_start))
            split = _to_raw(_minsi(split_lo, fx.Int32(k_end)))

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
            lds_thread_base = fx.Int32(wid) * fx.Int32(D) + fx.Int32(lid_x_vec)
            for i in range_constexpr(VEC):
                idx_i = lds_thread_base + fx.Int32(i)
                fx.ptr_store(m_local[i], lds_m_ptr + idx_i)
                fx.ptr_store(kv_local[i], lds_kv_ptr + idx_i)
                fx.ptr_store(w_local[i], lds_w_ptr + idx_i)

            gpu.barrier()

            # ---- wave 0: cross-wave reduce + norm + rope + scatter ----
            def _wave0():
                comp_lane = []
                for i in range_constexpr(VEC):
                    lane_off = fx.Int32(lid_x_vec) + fx.Int32(i)
                    m_g = fx.Float32(c_neg_inf)
                    m_arr = []
                    for w in range_constexpr(NW):
                        idx_w = fx.Int32(w * D) + lane_off
                        m_w = fx.ptr_load(lds_m_ptr + idx_w)
                        m_arr.append(m_w)
                        m_g = m_g.maximumf(m_w)
                    kv_sum = fx.Float32(0.0)
                    w_sum = fx.Float32(0.0)
                    for w in range_constexpr(NW):
                        idx_w = fx.Int32(w * D) + lane_off
                        kv_w = fx.ptr_load(lds_kv_ptr + idx_w)
                        w_w = fx.ptr_load(lds_w_ptr + idx_w)
                        scale_w = fx.Float32(fexp_f32(_to_raw(m_arr[w] - m_g)))
                        kv_sum = kv_sum + kv_w * scale_w
                        w_sum = w_sum + w_w * scale_w
                    rcp_w = fx.Float32(fx.rocdl.rcp(f32, _to_raw(w_sum)))
                    comp_lane.append(_to_raw(kv_sum * rcp_w))

                # ---- RMSNorm (wave-reduce sum-of-squares over wave 0) ----
                def wave_reduce_add(x):
                    w = fx.Float32(x)
                    for sh_exp in range_constexpr(log2_block):
                        off = BLOCK_THREADS // (2 << sh_exp)
                        w = w + fx.Float32(w.shuffle_xor(off, BLOCK_THREADS))
                    return _to_raw(w)

                sq_local = fx.Float32(0.0)
                for i in range_constexpr(VEC):
                    cl = fx.Float32(comp_lane[i])
                    sq_local = sq_local + cl * cl
                sq_full = wave_reduce_add(_to_raw(sq_local))
                var = fx.Float32(sq_full) * fx.Float32(c_inv_D)
                rrms = fmath.rsqrt(_to_raw(var + fx.Float32(c_eps)), fastmath=fm_fast)

                if const_expr(rms_weight_is_bf16):
                    rmsw_buf = _make_buffer(rms_weight, fx.Int32, BF16_DW)
                    rmsw_lane = _load_bf16_vec_then_f32(rmsw_buf, lid_x_vec)
                else:
                    rmsw_buf = _make_buffer(rms_weight, fx.Float32, F32_VW)
                    rmsw_lane = _load_f32_vec(rmsw_buf, lid_x_vec)

                rrms_f = fx.Float32(rrms)
                normed_lane = [
                    _to_raw(fx.Float32(comp_lane[i]) * rrms_f * fx.Float32(rmsw_lane[i]))
                    for i in range(VEC)
                ]

                # ---- GPT-J RoPE on RD tail ----
                comp_pos_i32 = _to_raw(
                    (fx.Int32(position) // fx.Int32(ratio)) * fx.Int32(ratio)
                )
                cos_buf = _make_buffer(cos_cache, fx.BFloat16, PAIRS_PER_THREAD)
                sin_buf = _make_buffer(sin_cache, fx.BFloat16, PAIRS_PER_THREAD)
                cos_row_base = fx.Int32(comp_pos_i32) * fx.Int32(RD // 2)

                is_rope_t = fx.Int32(lid) >= fx.Int32(ROPE_THREAD_LO)
                rope_rel = _maxsi(
                    fx.Int32(lid) - fx.Int32(ROPE_THREAD_LO), fx.Int32(0)
                )
                cs_off = _to_raw(cos_row_base + rope_rel * fx.Int32(PAIRS_PER_THREAD))

                if const_expr(PAIRS_PER_THREAD == 1):
                    cos_b = _buffer_load(cos_buf, cs_off, fx.BFloat16)
                    sin_b = _buffer_load(sin_buf, cs_off, fx.BFloat16)
                    cos_vals = [_to_raw(fx.BFloat16(cos_b).to(fx.Float32))]
                    sin_vals = [_to_raw(fx.BFloat16(sin_b).to(fx.Float32))]
                else:
                    cos_vec = _buffer_load(cos_buf, cs_off, fx.BFloat16, PAIRS_PER_THREAD)
                    sin_vec = _buffer_load(sin_buf, cs_off, fx.BFloat16, PAIRS_PER_THREAD)
                    cos_vals = [
                        _to_raw(fx.BFloat16(_velem(cos_vec, i)).to(fx.Float32))
                        for i in range(PAIRS_PER_THREAD)
                    ]
                    sin_vals = [
                        _to_raw(fx.BFloat16(_velem(sin_vec, i)).to(fx.Float32))
                        for i in range(PAIRS_PER_THREAD)
                    ]

                rotated_lane = list(normed_lane)
                for kk in range_constexpr(PAIRS_PER_THREAD):
                    e = fx.Float32(normed_lane[2 * kk])
                    o = fx.Float32(normed_lane[2 * kk + 1])
                    cc = fx.Float32(cos_vals[kk])
                    ss = fx.Float32(sin_vals[kk])
                    # BOUNDARY: non-fastmath subf by design (no FMA contraction).
                    new_e = arith.subf(_to_raw(e * cc), _to_raw(o * ss))
                    new_o = _to_raw(e * ss + o * cc)
                    rotated_lane[2 * kk] = new_e
                    rotated_lane[2 * kk + 1] = new_o

                out_lane = [
                    _to_raw(is_rope_t.select(rotated_lane[i], normed_lane[i]))
                    for i in range_constexpr(VEC)
                ]

                # ---- paged scatter (BF16 or FP8). Emitted in wave 0; ``lid``
                # (0..63) is the single-wave ``tid`` equivalent. ----
                ci = fx.Int32(position) // fx.Int32(ratio)
                block_in_seq = _to_raw(ci // fx.Int32(k_per_block))
                slot_in_block = _to_raw(ci % fx.Int32(k_per_block))
                bt_buf = _make_buffer(block_table, fx.Int32)
                bt_off = _to_raw(
                    fx.Int32(batch_id) * fx.Int32(block_table_seq_stride)
                    + fx.Int32(block_in_seq)
                )
                physical_block = _buffer_load(bt_buf, bt_off, fx.Int32)

                if const_expr(not quant):
                    cache_off = (
                        fx.Int32(physical_block) * fx.Int32(kv_cache_block_stride)
                        + fx.Int32(slot_in_block) * fx.Int32(kv_cache_token_stride)
                        + fx.Int32(lid_x_vec)
                    )
                    out_vec_t = T.vec(VEC, T.bf16)
                    raw_vec = fx.Vector.from_elements(out_lane, fx.Float32)
                    bf16_vec = raw_vec.truncf(out_vec_t)
                    cache_off_dw = _to_raw(cache_off >> fx.Int32(1))
                    dwords = (VEC + 1) // 2
                    bf16_as_i32 = fx.Vector(bf16_vec).bitcast(fx.Int32)
                    if const_expr(dwords == 1):
                        scalar_i32 = _velem(bf16_as_i32, 0)
                        out_buf = _make_buffer(kv_cache, fx.Int32)
                        _buffer_store(out_buf, cache_off_dw, scalar_i32, fx.Int32)
                    else:
                        out_buf = _make_buffer(kv_cache, fx.Int32, dwords)
                        _buffer_store(
                            out_buf, cache_off_dw, bf16_as_i32, fx.Int32, dwords
                        )
                elif const_expr(nm_asm):
                    # -- group_fp8 (V4 nm-asm): nope fp8 + inline dup e8m0; rope
                    # bf16 -> separate k_rope_buff. Shared emitter, byte-identical
                    # to the legacy single-wave / HCA paths (lane == wave-0 lid). --
                    _nm_cache_base = _to_raw(
                        fx.Int32(physical_block) * fx.Int32(kv_cache_block_stride)
                        + fx.Int32(slot_in_block) * fx.Int32(kv_cache_token_stride)
                    )
                    _nm_krope_base = _to_raw(
                        fx.Int32(physical_block) * fx.Int32(krope_block_stride)
                        + fx.Int32(slot_in_block) * fx.Int32(krope_token_stride)
                    )
                    emit_group_fp8_nm_asm_scatter(
                        normed_lane=normed_lane,
                        rotated_lane=rotated_lane,
                        lane=lid,
                        is_rope_t=is_rope_t,
                        cache_base=_nm_cache_base,
                        out_rsrc=_make_rsrc(kv_cache),
                        krope_base=_nm_krope_base,
                        krope_rsrc=_make_rsrc(k_rope_buff),
                        VEC=VEC,
                        NOPE=NOPE,
                        RTS=RTS,
                        log2_rts=log2_rts,
                        ROPE_THREAD_LO=ROPE_THREAD_LO,
                        wave_width=BLOCK_THREADS,
                        vecVf32=vecVf32,
                        fm_fast=fm_fast,
                    )
                elif const_expr(not quant_fp4):
                    # ── FP8 per-row scaled write + fp32 scale (mirror legacy) ──
                    # Wave-reduce-max over wave 0's 64 lanes; pair-coop dword
                    # store via shuffle_xor(1) within the wave.
                    def wave_reduce_max(x):
                        w = fx.Float32(x)
                        for sh_exp in range_constexpr(log2_block):
                            off = BLOCK_THREADS // (2 << sh_exp)
                            w = w.maximumf(
                                fx.Float32(w.shuffle_xor(off, BLOCK_THREADS))
                            )
                        return _to_raw(w)

                    _, fp8_max = _fp8_const()
                    c_fp8_max = fx.Float32(fp8_max)
                    c_neg_fp8_max = fx.Float32(-fp8_max)
                    c_safety_floor = fx.Float32(1e-4)
                    c_inv_fp8_max = fx.Float32(1.0 / fp8_max)

                    # (a) per-lane amax -> wave-reduce-max
                    am_local = fx.Float32(0.0)
                    for i in range_constexpr(VEC):
                        am_local = am_local.maximumf(fx.Float32(fmath.absf(out_lane[i])))
                    amax = fx.Float32(wave_reduce_max(_to_raw(am_local)))
                    am_safe = amax.maximumf(c_safety_floor)

                    # (b) scale = am_safe / FP8_MAX, optionally ceil-pow2
                    scale_raw = am_safe * c_inv_fp8_max
                    if const_expr(use_ue8m0):
                        scale_i32 = scale_raw.bitcast(fx.Int32)
                        bits_up = (scale_i32 + fx.Int32(0x7FFFFF)) & fx.Int32(0xFF800000)
                        scale_v = bits_up.bitcast(fx.Float32)
                    else:
                        scale_v = scale_raw

                    # (c) inv_scale via rcp.f32
                    inv_scale = fx.Float32(fx.rocdl.rcp(f32, _to_raw(scale_v)))

                    # (d) per-lane fp8 cast: clamp + fnuz NaN guard
                    c_neg_uf = fx.Float32(-(2.0**-8))
                    c_zero = fx.Float32(0.0)
                    fp8_inputs = []
                    for i in range_constexpr(VEC):
                        v = fx.Float32(out_lane[i]) * inv_scale
                        # arith.minimumf kept raw (no ArithValue.minimumf).
                        v = fx.Float32(
                            arith.minimumf(
                                _to_raw(v.maximumf(c_neg_fp8_max)), _to_raw(c_fp8_max)
                            )
                        )
                        is_tn = (v < c_zero) & (v > c_neg_uf)
                        fp8_inputs.append(_to_raw(is_tn.select(c_zero, v)))

                    # (e) pack VEC fp32 -> VEC fp8 bytes
                    c_p0 = _to_raw(fx.Int32(0))
                    if const_expr(VEC == 2):
                        # BOUNDARY: hand-packed rocdl.cvt_pk_fp8_f32 (raw operands).
                        pk = rocdl.cvt_pk_fp8_f32(
                            i32, fp8_inputs[0], fp8_inputs[1], c_p0, 0
                        )
                        pk = fx.Int32(pk)
                        peer_pk = fx.Int32(pk.shuffle_xor(1, BLOCK_THREADS))
                        dword = pk | (peer_pk << fx.Int32(16))
                    elif const_expr(VEC == 4):
                        # BOUNDARY: hand-packed rocdl.cvt_pk_fp8_f32 (raw operands).
                        pk = rocdl.cvt_pk_fp8_f32(
                            i32, fp8_inputs[0], fp8_inputs[1], c_p0, 0
                        )
                        pk = rocdl.cvt_pk_fp8_f32(
                            i32, fp8_inputs[2], fp8_inputs[3], pk, 1
                        )
                        dword = pk
                    else:
                        assert VEC == 8
                        # BOUNDARY: hand-packed rocdl.cvt_pk_fp8_f32 (raw operands).
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

                    dwords_per_store = 1 if VEC != 8 else 2
                    out_byte_buf = _make_buffer(
                        kv_cache, fx.Int8, 4 * dwords_per_store
                    )
                    block_byte_base = fx.Int32(physical_block) * fx.Int32(
                        kv_cache_block_stride
                    )

                    if const_expr(preshuffle):
                        sib = fx.Int32(slot_in_block)
                        token_tile_id = sib // fx.Int32(_PRESHUFFLE_TILE)
                        token_in_tile = sib % fx.Int32(_PRESHUFFLE_TILE)
                        d_for_tid = fx.Int32(lid) * fx.Int32(VEC)
                        col_tile_id = d_for_tid // fx.Int32(_PRESHUFFLE_TILE)
                        col_in_tile = d_for_tid % fx.Int32(_PRESHUFFLE_TILE)
                        in_block_off = (
                            token_tile_id * fx.Int32(_PRESHUFFLE_TILE * D)
                            + col_tile_id * fx.Int32(_PRESHUFFLE_TILE * _PRESHUFFLE_TILE)
                            + token_in_tile * fx.Int32(_PRESHUFFLE_TILE)
                            + col_in_tile
                        )
                    else:
                        in_block_off = fx.Int32(slot_in_block) * fx.Int32(
                            D
                        ) + fx.Int32(lid) * fx.Int32(VEC)

                    byte_off = _to_raw(block_byte_base + in_block_off)

                    if const_expr(VEC == 2):
                        if (lid & 1) == 0:
                            _store_dwords_at_byte(out_byte_buf, byte_off, _to_raw(dword))
                    elif const_expr(VEC == 4):
                        _store_dwords_at_byte(out_byte_buf, byte_off, dword)
                    else:
                        _store_dwords_at_byte(
                            out_byte_buf, byte_off, [dword[0], dword[1]]
                        )

                    # (f) lane-0 writes fp32 scale at cache_scale[phys, slot]
                    if lid == 0:
                        cs_buf = _make_buffer(cache_scale, fx.Float32)
                        cs_off = _to_raw(
                            fx.Int32(physical_block) * fx.Int32(cache_scale_block_stride)
                            + fx.Int32(slot_in_block)
                        )
                        _buffer_store(cs_buf, cs_off, scale_v, fx.Float32)
                else:
                    # ── FP4: per-group(32) e8m0 scale + E2M1 write (mirror
                    # legacy _build_kernel). Emitted in wave 0 where ``lid``
                    # (0..63) is the single-wave ``tid`` equivalent; the
                    # butterfly group-reduce over NTG lanes stays within wave
                    # 0's 64 physical lanes. ``lid_x_vec`` replaces the legacy
                    # ``tid_x_vec``. See _build_kernel for the full rationale. ──
                    lid_x_vec_i = lid_x_vec
                    NTG = _FP4_GROUP_SIZE // VEC
                    LOG2_NTG = int(math.log2(NTG))
                    PACKED_BYTES = VEC // 2
                    K_TILES = D // _FP4_K_TILE
                    KVBS = k_per_block
                    c_eps_amax = fx.Float32(6.0 * float.fromhex("0x1p-126"))

                    # (a) per-lane amax, then butterfly group-reduce over NTG lanes.
                    am_grp = fx.Float32(0.0)
                    for i in range_constexpr(VEC):
                        am_grp = am_grp.maximumf(fx.Float32(fmath.absf(out_lane[i])))
                    for sh_exp in range_constexpr(LOG2_NTG):
                        off = NTG // (2 << sh_exp)
                        am_grp = am_grp.maximumf(
                            fx.Float32(am_grp.shuffle_xor(off, BLOCK_THREADS))
                        )
                    am_safe = am_grp.maximumf(c_eps_amax)

                    # (b) MX RoundUp e8m0 + multiplicative quant scale.
                    # BOUNDARY: emit_mx_e8m0_scale is a raw IR builder.
                    e8m0 = emit_mx_e8m0_scale(
                        ArithValue(am_safe.ir_value()),
                        mode=_MxRoundInt.RoundUp,
                        dtype=_MxDtypeInt.FP4_E2M1,
                    )
                    quant_exp = fx.Int32(254) - fx.Int32(e8m0)
                    quant_scale = (quant_exp << fx.Int32(23)).bitcast(fx.Float32)

                    # (c) per-element E2M1 nibble, pack VEC/2 bytes.
                    # BOUNDARY: emit_f32_to_e2m1 is a raw IR builder (raw operand).
                    nibs = [
                        emit_f32_to_e2m1(_to_raw(fx.Float32(out_lane[i]) * quant_scale))
                        for i in range_constexpr(VEC)
                    ]

                    out_byte_buf = _make_buffer(kv_cache, fx.Int8)
                    packed_start = fx.Int32(lid_x_vec_i) >> fx.Int32(1)
                    flat_slot = fx.Int32(physical_block) * fx.Int32(
                        k_per_block
                    ) + fx.Int32(slot_in_block)
                    for b in range_constexpr(PACKED_BYTES):
                        byte_val = fx.Int32(nibs[2 * b]) | (
                            fx.Int32(nibs[2 * b + 1]) << fx.Int32(4)
                        )
                        packed_idx = packed_start + fx.Int32(b)
                        if const_expr(preshuffle):
                            k_tile = packed_idx // fx.Int32(64)
                            rem = packed_idx % fx.Int32(64)
                            group4 = rem // fx.Int32(16)
                            sub16 = rem % fx.Int32(16)
                            byte_off = _to_raw(
                                fx.Int32(physical_block) * fx.Int32(K_TILES * 4 * KVBS * 16)
                                + k_tile * fx.Int32(4 * KVBS * 16)
                                + group4 * fx.Int32(KVBS * 16)
                                + fx.Int32(slot_in_block) * fx.Int32(16)
                                + sub16
                            )
                        else:
                            byte_off = _to_raw(
                                flat_slot * fx.Int32(D // 2) + packed_idx
                            )
                        # i8 nibble byte: element offset == byte offset.
                        _buffer_store(
                            out_byte_buf,
                            byte_off,
                            _to_raw(byte_val.to(fx.Int8)),
                            fx.Int8,
                        )

                    # (e) group-rep lane writes the e8m0 scale byte.
                    scale_group_idx = fx.Int32(lid_x_vec_i) // fx.Int32(_FP4_GROUP_SIZE)
                    if lid % NTG == 0:
                        cs_buf = _make_buffer(cache_scale, fx.Int8)
                        if const_expr(preshuffle):
                            # scale [NB, k_tiles, 4, kvbs] u8, slot axis
                            # INTERLEAVED: sflat = (slot%16)*4 + (slot//16)
                            # (KVS_NTPW==4). Matches the legacy writer, the
                            # op-test reference, and the packed N_PHYS==1
                            # readers in pa_mqa_logits_fp4*.
                            k_tile_s = scale_group_idx // fx.Int32(4)
                            group4_s = scale_group_idx % fx.Int32(4)
                            sflat = (fx.Int32(slot_in_block) % fx.Int32(16)) * fx.Int32(
                                4
                            ) + (fx.Int32(slot_in_block) // fx.Int32(16))
                            cs_off = _to_raw(
                                fx.Int32(physical_block) * fx.Int32(K_TILES * 4 * KVBS)
                                + k_tile_s * fx.Int32(4 * KVBS)
                                + group4_s * fx.Int32(KVBS)
                                + sflat
                            )
                        else:
                            cs_off = _to_raw(
                                flat_slot * fx.Int32(D // _FP4_GROUP_SIZE)
                                + scale_group_idx
                            )
                        _buffer_store(
                            cs_buf,
                            cs_off,
                            _to_raw(fx.Int32(e8m0).to(fx.Int8)),
                            fx.Int8,
                        )  # e8m0 uint8

            if wid == 0:
                _wave0()

        if fx.Int32(position) >= 0:
            _body()

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
        stream: fx.Stream,
    ):
        idx_p = fx.Int64(plan_capacity)
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
    use_ue8m0: bool,
    preshuffle: bool,
    rms_weight_is_bf16: bool,
    rms_eps: float,
    enable_prefetch_input: bool = True,
    # "none" | "per_row_fp8" | "group_fp8" | "fp4" (single quant selector).
    quant_mode: str = "none",
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
    use_ue8m0: bool,
    preshuffle: bool,
    rms_weight_is_bf16: bool,
    rms_eps: float,
    # "none" | "per_row_fp8" | "group_fp8" | "fp4" (single quant selector).
    quant_mode: str = "none",
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
    # Master quant selector; overrides the legacy `quant` bool. Accepts:
    #   "none" | "fp8"/"per_row_fp8" (indexer) | "group_fp8" (CSA/HCA Main
    #   nm-asm) | "fp4". None -> derive from `quant` ("per_row_fp8" if quant).
    quant_mode: str | None = None,
    k_split_num_waves: int | None = None,
    k_rope_cache: (
        torch.Tensor | None
    ) = None,  # group_fp8 only: paged [NB, k_per_block, RD] bf16 rope
    stream: torch.cuda.Stream | None = None,
) -> None:
    """FlyDSL drop-in replacement for ``fused_compress_attn`` (Triton).

    Same side-effecting semantics: cache scatter IS the output. Caller MUST
    invoke BEFORE ``update_compressor_states`` (state cache reads must see
    previous-fwd data).

    ``quant_mode`` selects the scatter quantization (default derived from the
    legacy ``quant`` bool: ``"fp8" if quant else "none"``):
      - ``"none"`` → BF16 paged write (CSA / HCA Main).
      - ``"fp8"``  → FP8 e4m3 per-row e8m0 scale + MFMA 16x16 preshuffle.
      - ``"fp4"``  → FP4 (E2M1) per-group(32) e8m0 scale + FP4 KV preshuffle
        (``kv_cache`` uint8 [NB, k_tiles, 4, k_per_block, 16];
        ``cache_scale`` uint8 [NB, k_tiles, 4, k_per_block]).

    ``k_split_num_waves`` (BF16, FP8, or FP4 scatter): when set to NW > 1, routes
    to the multi-wave LDS K-split kernel (block = 64*NW, K split across NW waves,
    single dispatch). Speeds up the latency-bound decode regime (small N,
    1 wave/CU) where the legacy single-wave serial K-chain stalls. Requires a
    real ``block_tables`` (has_block_table). When None, auto-picks NW for the
    tuned CSA Main (BF16), CSA Indexer (FP8), and CSA Indexer (FP4) shapes via
    :func:`csa_ksplit_num_waves` and uses legacy elsewhere; when 1, forces
    legacy. K must be divisible by NW. (The K-split win comes from parallelizing
    the dtype-agnostic online-softmax pool, so FP4 reuses the FP8 wave-count
    heuristic on the shared CSA Indexer geometry.)
    """
    # ---- resolve quant mode (quant_mode overrides the legacy `quant` bool) ----
    #   "none"        -> bf16 paged write
    #   "per_row_fp8" -> FP8 e4m3 per-row scale (indexer)        [alias "fp8"]
    #   "group_fp8"   -> FP8 1xG group scale (CSA/HCA Main nm-asm)
    #   "fp4"         -> FP4 (E2M1) per-group(32) e8m0 scale
    _mode = (
        quant_mode if quant_mode is not None else ("per_row_fp8" if quant else "none")
    )
    if _mode == "fp8":
        _mode = "per_row_fp8"  # back-compat alias
    if _mode not in ("none", "per_row_fp8", "group_fp8", "fp4"):
        raise ValueError(
            f"quant_mode must be none|fp8|per_row_fp8|group_fp8|fp4, got {_mode!r}"
        )
    _quant = _mode != "none"
    _fp4 = _mode == "fp4"
    # `_mode` (none|per_row_fp8|group_fp8|fp4) is the single selector passed
    # straight to the builders, which derive quant/quant_fp4/nm_asm from it.

    # ---- gfx1250 dispatch (wave32) ----
    from aiter.jit.utils.chip_info import get_gfx as _get_gfx

    if _get_gfx() == "gfx1250":
        if _fp4:
            raise NotImplementedError(
                "fused_compress_attn FP4 path is not implemented for gfx1250"
            )
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
    nm_asm = _mode == "group_fp8"
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
        # FP8/BF16 caches are 3D [NB, k_per_block, D]; the FP4 preshuffle cache
        # is 5D [NB, k_tiles, 4, k_per_block, 16] uint8.
        if not _fp4 and kv_cache.dim() != 3:
            raise ValueError(f"kv_cache must be 3D, got {kv_cache.shape}")
        if block_tables.dim() != 2 or block_tables.dtype != torch.int32:
            raise ValueError("block_tables must be 2D int32")
        if not block_tables.is_contiguous():
            raise ValueError("block_tables must be contiguous")
    if _quant:
        if not has_bt:
            raise ValueError("quant requires block_tables")
        if not _fp4:
            # FP8 path (per_row_fp8, or group_fp8/nm_asm which carries its own
            # inline e8m0 scale layout -> no separate 2D cache_scale).
            if kv_cache.dtype == torch.bfloat16:
                raise TypeError("fp8 quant needs fp8 kv_cache")
            if not nm_asm and (
                cache_scale is None
                or cache_scale.dtype != torch.float32
                or cache_scale.dim() != 2
                or cache_scale.shape[0] != kv_cache.shape[0]
            ):
                raise ValueError(
                    "fp8 quant requires fp32 [NB, k_per_block] cache_scale"
                )
            if preshuffle:
                if head_dim % _PRESHUFFLE_TILE != 0:
                    raise ValueError(
                        f"preshuffle requires head_dim%16==0, got {head_dim}"
                    )
                if k_per_block % _PRESHUFFLE_TILE != 0:
                    raise ValueError(
                        f"preshuffle requires k_per_block%16==0, got {k_per_block}"
                    )
        else:
            # FP4 path: uint8 packed cache + uint8 e8m0 scale.
            if kv_cache.dtype != torch.uint8:
                raise TypeError(
                    f"fp4 quant needs uint8 (fp4x2) kv_cache, got {kv_cache.dtype}"
                )
            if cache_scale is None or cache_scale.dtype != torch.uint8:
                raise ValueError("fp4 quant requires uint8 e8m0 cache_scale")
            if cache_scale.shape[0] != kv_cache.shape[0]:
                raise ValueError("fp4 cache_scale NB must match kv_cache NB")
            if head_dim % _FP4_K_TILE != 0:
                raise ValueError(f"fp4 requires head_dim%128==0, got {head_dim}")
            if head_dim % _FP4_GROUP_SIZE != 0:
                raise ValueError(f"fp4 requires head_dim%32==0, got {head_dim}")
            if preshuffle and k_per_block % _PRESHUFFLE_TILE != 0:
                raise ValueError(
                    f"fp4 preshuffle requires k_per_block%16==0, got {k_per_block}"
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
        # FP4 store derives byte offsets from constants (k_tiles/group/tile),
        # not these strides — bind the outer block stride for completeness.
        kv_cache_block_stride = kv_cache.stride(0)
        kv_cache_token_stride = kv_cache.stride(1) if not _fp4 else 0
    else:
        bt_arg = state_slot_mapping  # int32 dummy
        bt_seq_stride = 0
        kv_cache_arg = cos_2d  # bf16 dummy
        kv_cache_block_stride = 0
        kv_cache_token_stride = 0

    if _quant and not nm_asm:
        cs_arg = cache_scale
        # FP4 scale store derives its offset from constants; FP8 uses this.
        cs_block_stride = cache_scale.stride(0) if not _fp4 else 0
    else:
        # nm_asm has no separate scale tensor (e8m0 inline in the fp8 entry); bf16/
        # non-quant paths don't use cache_scale either -> fp32 dummy.
        cs_arg = rms_weight  # fp32 dummy
        cs_block_stride = 0

    # nm_asm: rotated PE bf16 -> separate paged k_rope_cache (V4 nm layout). For all
    # other modes pass a bf16 dummy so the kernel arg binding stays valid.
    if nm_asm:
        if not _quant:
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

    # ---- K-split fast path (BF16 + FP8 + FP4 scatter) ----
    # k_split_num_waves: None ⟹ auto-pick (tuned geometries only); int>1 ⟹
    # forced NW; 1 ⟹ forced legacy. Auto triggers for the CSA Main (BF16),
    # CSA Indexer (FP8), and CSA Indexer (FP4) shapes the K-split kernel
    # supports; other shapes fall through to the legacy single-wave kernel.
    _is_csa_main = (
        head_dim == 512
        and rope_head_dim == 64
        and ratio == 4
        and overlap
        and _mode == "none"
    )
    _is_csa_indexer = (
        head_dim == 128
        and rope_head_dim == 64
        and ratio == 4
        and overlap
        and _mode == "per_row_fp8"
    )
    # FP4 indexer: same compress geometry as the FP8 indexer (D=128, RD=64,
    # ratio=4, overlap), only the scatter dtype differs. The K-split win comes
    # from parallelizing the dtype-agnostic online-softmax pool, so the FP8
    # wave-count heuristic carries over.
    _is_csa_indexer_fp4 = (
        head_dim == 128
        and rope_head_dim == 64
        and ratio == 4
        and overlap
        and _mode == "fp4"
    )
    # group_fp8 (nm-asm) CSA Main shares the CSA Main geometry (D=512, K=8); it is
    # ported to the K-split kernel via the shared nm-asm scatter emitter.
    _is_csa_main_nm = (
        head_dim == 512
        and rope_head_dim == 64
        and ratio == 4
        and overlap
        and _quant
        and nm_asm
    )
    if (
        k_split_num_waves is None
        and has_bt
        and (_is_csa_main or _is_csa_indexer or _is_csa_indexer_fp4 or _is_csa_main_nm)
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
            use_ue8m0=use_ue8m0,
            preshuffle=preshuffle,
            rms_weight_is_bf16=_rms_weight_is_bf16,
            rms_eps=float(rms_eps),
            quant_mode=_mode,  # resolved: none|per_row_fp8|group_fp8|fp4
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
        use_ue8m0=use_ue8m0,
        preshuffle=preshuffle,
        rms_weight_is_bf16=_rms_weight_is_bf16,
        rms_eps=float(rms_eps),
        quant_mode=_mode,  # single selector: none|per_row_fp8|group_fp8|fp4
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
