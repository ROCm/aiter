# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""GDN K5 inter-chunk state scan (+ optional fused K6 output) — gfx942 FlyDSL.

For each chunk t (serial over NT chunks):
  1. Store h snapshot for downstream K6
  2. v_new = u - w @ h   (delta correction via MFMA)
  3. Gated decay + state update:
       v_new *= exp(g_last - g_cumsum)
       h = h * exp(g_last) + k^T @ v_new

``COMPUTE_OUTPUT=True`` additionally fuses the K6 inter/intra-chunk output into
the same dispatch, so ``h`` and the gated ``v_new`` stay resident in LDS instead
of round-tripping through HBM. The fused body appends, per chunk:
  4. o  = q @ h^T                              (GEMM3, inter-chunk)
     A  = tril(q @ k^T)                        (GEMM4a + causal mask)
     o  = scale * (exp(g_i) * o + exp(g_i - g_last) * (A @ v_new_gated))
     store o -> HBM [T_flat, H, V]
The fused path pairs with ``STORE_H=False`` / ``SAVE_NEW_VALUE=False``, which
elide the two HBM drains that only the separate K6 kernel consumes.
"""

import math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import vector as _vector
from flydsl.expr import as_ir_value, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .k5_variants import _bv_of_variant, _legal_bv_candidates, _variant_tag

_LOG2E = math.log2(math.e)  # 1.4426950408889634


# --------------------------------------------------------------------------- #
# gfx942 variant selection -- one H*N rule for both the K5-only and the fused
# K5+K6 kernel.
#
# The measured-best BV/wave tile is a function of the grid-size product ``H*N``:
# larger tiles win as there is more parallel work to fill the device.
#
#     H*N        K5 winner     fused K5+K6 winner
#     4..32      bv16          bv16
#     48..72     bv32          bv32
#     96..768    bv64w8        bv64w8
#
# Caveat: every sweep shape has V=128. ``H*N`` is a proxy for the CTA count
# ``H*N*ceil(V/BV)``; a future V!=128 workload may need the fill formulation.
# --------------------------------------------------------------------------- #
_HN_BV32 = 32  # H*N above this prefers bv32 over bv16
_HN_BV64W8 = 80  # H*N above this prefers bv64w8 (wave-widened BV=64)


def _hn_variant(*, H: int, N: int, V: int) -> str | None:
    """The ``H*N`` tile rule, or None when its BV is illegal for ``V``."""
    hn = H * max(1, N)
    if hn <= _HN_BV32:
        tag = "bv16"
    elif hn <= _HN_BV64W8:
        tag = "bv32"
    else:
        tag = "bv64w8"
    if _bv_of_variant(tag) not in _legal_bv_candidates(V):
        return None
    return tag


def select_variant(*, H: int, N: int, V: int) -> str | None:
    """gfx942 best K5 variant tag from the ``H*N`` grid-size rule.

    Returns None only when the rule's BV is illegal for ``V``, and the caller
    falls back to its cross-arch grid-fill heuristic.
    """
    return _hn_variant(H=H, N=N, V=V)


def select_fused_variant(*, H: int, N: int, V: int) -> str | None:
    """gfx942 best fused K5+K6 variant tag from the ``H*N`` grid-size rule."""
    return _hn_variant(H=H, N=N, V=V)


def _make_fast_exp(g_is_log2_scaled: bool):
    """Return the ``exp`` helper, pre-specialised on the gate's log2 scaling.

    ``as_ir_value`` is required: ``rocdl.exp2`` takes a raw ``ir.Value``, but a
    re-typed loop-carried value arrives as a FlyDSL ``Float32`` wrapper.
    """
    if g_is_log2_scaled:

        def _fast_exp(x):
            return rocdl.exp2(T.f32, as_ir_value(x))

    else:

        def _fast_exp(x):
            return rocdl.exp2(T.f32, as_ir_value(x * _LOG2E))

    return _fast_exp


# -- f32 -> bf16 conversion ----------------------------------------------------
# Two variants, selected by the ``FAST_FP32_TO_FP16`` environment variable.
#
# DEFAULT (unset) -- ``arith.truncf``: plain IEEE round-to-nearest-EVEN. No
#   custom code, unbiased rounding, and NaN/Inf classification falls out of the
#   semantic instead of needing a guard. gfx942 has no ``v_cvt_pk_bf16_f32`` so
#   the backend expands RNE inline.
#
# FAST_FP32_TO_FP16=1 -- ``(bitcast<u32>(x) + 0x8000) >> 16``, round-half-AWAY
#   from zero. 2 VALU/element. **UNSAFE: it does not preserve NaN.** This is the
#   original pre-review conversion, kept as an opt-in performance escape hatch.
#
# What "unsafe" means concretely, verified by bit math over the pattern space:
#
#   input                       default (truncf)   FAST_FP32_TO_FP16=1
#   0x7F800001 low-payload NaN        NaN                +Inf
#   0xFF800001 negative NaN           NaN                -Inf
#   0x7FFFFFFF all-ones mantissa      NaN                -0.0   <-- finite!
#   +-Inf                             +-Inf              +-Inf
#   finite                            finite             finite
#
_FAST_BF16_ENV = "FAST_FP32_TO_FP16"


def _fast_bf16_from_env() -> bool:
    """Read ``FAST_FP32_TO_FP16`` once, at import.

    Read at import rather than per call so the value cannot change midway
    through a process and leave some kernels built one way and some the other.
    Changing it therefore needs a fresh process, not just a fresh compile.
    """
    raw = os.environ.get(_FAST_BF16_ENV)
    if raw is None:
        return False
    val = raw.strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("", "0", "false", "no", "off"):
        return False
    raise ValueError(
        f"{_FAST_BF16_ENV}={raw!r} is not a boolean; use 1/0, true/false, "
        "yes/no or on/off."
    )


USE_FAST_BF16 = _fast_bf16_from_env()

# Folded into the kernel name below. An env var does not change any source file,
# so it does not invalidate the FlyDSL disk cache on its own.
_BF16_KERNEL_SUFFIX = "_fastcvt" if USE_FAST_BF16 else ""


def _to_bf16(val, n=1, fast=None):
    """f32 -> bf16. Which variant is used is decided by :data:`USE_FAST_BF16`.

    ``n`` is the element count: 1 for a scalar ``Float32``, N for an f32xN
    ``Vector``. Returns a raw ``ir.Value``.

    ``fast`` selects the variant; ``None`` (the default) means "whatever
    :data:`USE_FAST_BF16` says", i.e. the ``FAST_FP32_TO_FP16`` environment
    variable.

    Default: one ``arith.truncf`` -- IEEE round-to-nearest-even, NaN and Inf
    carried through by construction.

    ``FAST_FP32_TO_FP16=1``: ``(bitcast<u32>(x) + 0x8000) >> 16``, i.e.
    round-half-AWAY-from-zero. Values are sign-magnitude so one bias serves both
    signs, and for finite values the carry can only perturb the exponent within
    ~1 ulp of FLT_MAX. It does not preserve NaN.
    """
    if fast is None:
        fast = USE_FAST_BF16

    is_vec = n > 1
    bf16_ty = T.vec(n, T.bf16) if is_vec else T.bf16

    if not fast:
        return _arith.truncf(bf16_ty, as_ir_value(val))

    i32_ty = T.vec(n, T.i32) if is_vec else T.i32
    i16_ty = T.vec(n, T.i16) if is_vec else T.i16

    def _splat(c):
        return as_ir_value(fx.full(n, c, fx.Int32) if is_vec else fx.Int32(c))

    bits = _arith.bitcast(i32_ty, as_ir_value(val))
    # The shift may be signed or unsigned: the trunci below keeps only the low
    # 16 bits, which are bits 16..31 of the input either way.
    hi = _arith.shrui(_arith.addi(bits, _splat(0x8000)), _splat(16))
    narrowed = _arith.trunci(i16_ty, hi)
    cast = _vector.bitcast if is_vec else _arith.bitcast
    return cast(bf16_ty, narrowed)


def _buffer_copy_atoms():
    """bf16 element count -> the buffer-copy atom of that width."""
    return {
        2: fx.rocdl.BufferCopy32b,
        4: fx.rocdl.BufferCopy64b,
        8: fx.rocdl.BufferCopy128b,
    }


def compile_chunk_gated_delta_h_gfx942(
    *,
    K: int,
    V: int,
    BT: int = 64,
    BV: int = 32,
    H: int,
    Hg: int,
    USE_G: bool = True,
    USE_GK: bool = False,
    USE_INITIAL_STATE: bool = True,
    STORE_FINAL_STATE: bool = True,
    SAVE_NEW_VALUE: bool = True,
    IS_VARLEN: bool = True,
    WU_CONTIGUOUS: bool = True,
    STATE_DTYPE_BF16: bool = False,
    G_IS_LOG2_SCALED: bool = False,
    NR_SPLIT: int = 1,
    COMPUTE_OUTPUT: bool = False,
    STORE_H: bool = True,
    SCALE: float | None = None,
):
    """Build the gfx942 GDN K5 launcher for one compile-time configuration.

    The K5-only defaults keep the signature compatible with
    ``compile_chunk_gated_delta_h`` so ``_get_or_compile`` in
    ``linear_attention_prefill_kernels`` can call either implementation without
    modification.

    The three trailing flags select the fused K5+K6 build:
      COMPUTE_OUTPUT: emit the K6 output stage (GEMM3/4a/4b + the ``o`` store).
        Requires ``SCALE`` and uses the ``q_tensor`` / ``o_tensor`` params.
      STORE_H: drain the per-chunk ``h`` snapshot to HBM. Only the *separate* K6
        kernel reads it, so the fused build sets this False.
      SCALE: K6 query scale. Required iff COMPUTE_OUTPUT, ignored otherwise.
    """
    assert K <= 256
    assert K % 64 == 0
    assert BV % 16 == 0
    # gfx942 LDS budget: BV=64 is largest V-tile size that fits under the LDS budget.
    assert BV <= 64, (
        f"gfx942 LDS budget caps BV at 64 (got BV={BV}); "
        "BV>64 overflows the 64 KiB/CU LDS limit at K=128, BT=64."
    )
    assert (SCALE is not None) == COMPUTE_OUTPUT, (
        "SCALE is required iff COMPUTE_OUTPUT (the K6 query scale); got "
        f"COMPUTE_OUTPUT={COMPUTE_OUTPUT}, SCALE={SCALE}."
    )
    assert V % BV == 0, "BV must tile V; got BV={BV}, V={V}."
    NUM_K_BLOCKS = K // 64

    # -- Chiplet (XCD) remap --
    # A head's GRID_V V-tiles all read the same k/w/g/gk slices. Under the HW
    # default (flat block `xy` runs on XCD `xy % NXCD`) they scatter across XCDs,
    # so those slices land in up to NXCD separate private L2s. The remap below
    # co-locates each head's whole V-tile run on one XCD, so they are fetched
    # once and reused. GRID_V is compile-time, so it is all integer math on the
    # flat block id.
    GRID_V = (V + BV - 1) // BV
    NXCD = 8

    _fast_exp = _make_fast_exp(G_IS_LOG2_SCALED)

    WARP_SIZE = 64

    MFMA_M = 16
    MFMA_N = 16
    MFMA_K = 16  # gfx942: K=16
    N_REPEAT = BV // MFMA_N

    # -- Wave decomposition (NR_SPLIT: the "wave widening" axis) --
    # A wave is (wid_m, wid_n):
    #   wid_m in [0, M_WAVES)  -- the BT tile for GEMM1 / the K tile for GEMM2.
    #   wid_n in [0, NR_SPLIT) -- this wave's slice of the N_REPEAT (V) axis; it
    #                             owns N_REPEAT_LOCAL column tiles rather than
    #                             looping over all of them.
    #
    # LDS pins gfx942 to one workgroup per CU, so a CU holds exactly NUM_WARPS
    # waves. At NUM_WARPS=4 that is 1 wave/SIMD -- too little to hide HBM latency
    # in a memory-bound kernel. Splitting V across waves multiplies resident
    # waves by NR_SPLIT for free: LDS depends only on BV, and each wave's share
    # of the h accumulators (its VGPR footprint) shrinks by the same factor.
    # NR_SPLIT=1 is the plain 4-wave kernel.
    M_WAVES = BT // 16
    assert N_REPEAT % NR_SPLIT == 0, (
        f"NR_SPLIT={NR_SPLIT} must divide N_REPEAT={N_REPEAT} (=BV/16); "
        f"BV={BV} supports NR_SPLIT in "
        f"{[s for s in (1, 2, 4, 8) if N_REPEAT % s == 0]}"
    )
    N_REPEAT_LOCAL = N_REPEAT // NR_SPLIT

    # Splitting V so finely that a wave owns a single 16-wide tile is broken.
    # One of the legal configurations, BV=32 across 8 waves,
    # nondeterministically drains stale LDS into the h snapshot.
    # It is the lds_h -> HBM drain, where this shape degenerates:
    # H_DRAIN_ROWS lands on BV and the drain collapses to a single rest pass.
    # Hence, we assert here that we don't build such variants.
    assert not (NR_SPLIT > 1 and N_REPEAT_LOCAL == 1), (
        f"BV={BV} across {M_WAVES * NR_SPLIT} waves (NR_SPLIT={NR_SPLIT}) leaves "
        f"one 16-wide V tile per wave (N_REPEAT_LOCAL=1), which corrupts the h "
        f"snapshot; use NR_SPLIT=1 or a BV with N_REPEAT // NR_SPLIT >= 2."
    )
    NUM_WARPS = M_WAVES * NR_SPLIT
    BLOCK_THREADS = NUM_WARPS * WARP_SIZE
    assert BLOCK_THREADS <= 1024, (
        f"BLOCK_THREADS={BLOCK_THREADS} exceeds the gfx942 workgroup limit "
        f"(1024); reduce NR_SPLIT."
    )
    # K6 only: b_A (GEMM4a) is V-independent, so rather than have every wid_n
    # wave recompute all of A, each owns BT_STEPS_LOCAL of the BT // WMMA_K
    # key-column tiles and writes its slice into the shared lds_A.
    if COMPUTE_OUTPUT:
        assert (BT // MFMA_K) % NR_SPLIT == 0, (
            f"NR_SPLIT={NR_SPLIT} must divide BT_STEPS={BT // MFMA_K} to split "
            f"b_A across the V-split waves"
        )
    BT_STEPS_LOCAL = (BT // MFMA_K) // NR_SPLIT

    # -- LDS layout --
    # Every buffer uses the same GROUP-MAJOR + XOR scheme (see _grp_idx): a
    # logical [R, C] tile is stored as [R][C/4][4] with the group index
    # XOR-swizzled by the row. 4 bf16 = 8 B = one MFMA fragment, so each fragment
    # access is a single conflict-free ds_read_b64/ds_write_b64, no padding.
    assert BT % 4 == 0 and K % 4 == 0
    # The XOR is a bank bijection only if a row has >= 16 groups (one per lane of
    # an MFMA fragment).
    assert K // 4 >= 16 and BT // 4 >= 16, "group-XOR needs >=16 groups per row"

    # lds_w: w tile [BT, K] (A-frag for GEMM1). Single stage. Plain row-major
    # would give a 256 B pitch == 0 (mod 32 banks), putting all 16 lanes of an
    # A-frag on ONE bank -- a 16-way conflict on the hottest read in the kernel.
    LDS_W_NG = K // 4
    LDS_W_ELEMS = BT * K

    # -- lds_kt orientation --
    # The two tensor k MFMA consumers contract over opposite axes:
    #   GEMM2  h += kᵀ @ v_new   contracts over BT -> wants k as [K, BT]
    #   GEMM4a A   = q  @ kᵀ     contracts over K  -> wants k as [BT, K]
    #
    #   K5-only: GEMM2 is the only reader -> [K, BT], nothing gathers.
    #   fused:   GEMM4a issues 4x the fragment reads GEMM2 does (BT_STEPS_LOCAL x
    #            NUM_K_BLOCKS x K_STEPS_PER_BLOCK vs NUM_K_BLOCKS x BT_TILES).
    #            Hence, store k as [BT, K] and GEMM2 takes the strided read instead.
    KT_TRANSPOSED = not COMPUTE_OUTPUT
    # Groups per row of whichever logical layout is in use
    # ([K,BT] -> BT/4; [BT,K] -> K/4).
    LDS_KT_NG = (BT // 4) if KT_TRANSPOSED else (K // 4)
    LDS_KT_COLS = BT if KT_TRANSPOSED else K
    LDS_KT_ELEMS = K * BT

    # lds_vn: v_new stored TRANSPOSED as [BV, BT] -> GEMM2 B-frag = run over BT
    # (contraction) at fixed V. Each CTA only handles a BV-wide V-slice, and every
    # vnt access uses v_local = nr*16 + lane_n in [0, BV) -- NOT the full V, so the
    # buffer is sized to BV rows (this is what lets BV=64 fit the 64 KiB budget).
    LDS_VNT_NG = BT // 4
    LDS_VNT_ELEMS = BV * BT

    # lds_h: h snapshot, logically [BV, K] (v_local, k). Both consumers want
    # K-major: GEMM1's B-frag is a run over K (the contraction) at fixed V, and
    # the HBM snapshot wants K contiguous. Sized to BV rows, not V, for the same
    # reason as lds_vnt.
    LDS_H_NG = K // 4
    LDS_H_ELEMS = BV * K  # BV rows of V per CTA, no padding

    # lds_A (K6 only): the [BT, BT] attention matrix, staged between GEMM4a and
    # GEMM4b because GEMM4a's accumulator layout is the TRANSPOSE of the A-operand
    # GEMM4b needs. Same group-major + XOR scheme as the others.
    LDS_A_NG = BT // 4
    LDS_A_ELEMS = BT * BT if COMPUTE_OUTPUT else 0

    # LDS budget with the K6 buffer. At BV=64 / K=128 the five buffers come to
    # exactly 64 KiB, so aliasing is currently inactive -- but the path is kept
    # because GEMM4a (the lds_A writer) runs strictly after GEMM3 (the last lds_h
    # reader), so lds_A may reuse lds_h's storage under a barrier whenever the
    # total does overflow. Aliasing needs lds_A (BT*BT) <= lds_h (BV*K), i.e.
    # BV >= 32; BV=16 never overflows, so it keeps the buffers distinct.
    _lds_total_kib = (
        (LDS_W_ELEMS + LDS_KT_ELEMS + LDS_VNT_ELEMS + LDS_H_ELEMS + LDS_A_ELEMS)
        * 2
        / 1024
    )
    ALIAS_A_ONTO_H = (
        COMPUTE_OUTPUT and _lds_total_kib > 64.0 and LDS_A_ELEMS <= LDS_H_ELEMS
    )
    assert not (_lds_total_kib > 64.0 and not ALIAS_A_ONTO_H), (
        f"LDS {_lds_total_kib:.0f} KiB > 64 KiB and lds_A ({LDS_A_ELEMS}) does "
        f"not fit lds_h ({LDS_H_ELEMS}) to alias (BV={BV}, K={K})"
    )

    # The drain walks pairs of adjacent k-groups (one 16 B store per thread), so
    # the block must tile the pair count and a row must hold an even number of
    # groups. This holds for every legal NR_SPLIT, but assert rather than rely on
    # the algebra.
    if STORE_H:
        assert LDS_H_NG % 2 == 0, f"h drain pairs k-groups; K/4={LDS_H_NG} must be even"
        assert (BV * (LDS_H_NG // 2)) % BLOCK_THREADS == 0, (
            f"h snapshot drain ({BV * LDS_H_NG // 2} pairs) must tile "
            f"BLOCK_THREADS={BLOCK_THREADS}"
        )

    # lds_A gets its own allocation only when it cannot be aliased onto lds_h;
    # the K5 build and the aliased fused build want the identical 4-buffer layout.
    if COMPUTE_OUTPUT and not ALIAS_A_ONTO_H:

        @fx.struct
        class SharedStorage:
            lds_w: fx.Array[fx.BFloat16, LDS_W_ELEMS, 16]
            lds_kt: fx.Array[fx.BFloat16, LDS_KT_ELEMS, 16]
            lds_vnt: fx.Array[fx.BFloat16, LDS_VNT_ELEMS, 16]
            lds_h: fx.Array[fx.BFloat16, LDS_H_ELEMS, 16]
            lds_A: fx.Array[fx.BFloat16, LDS_A_ELEMS, 16]

    else:

        @fx.struct
        class SharedStorage:
            lds_w: fx.Array[fx.BFloat16, LDS_W_ELEMS, 16]
            lds_kt: fx.Array[fx.BFloat16, LDS_KT_ELEMS, 16]
            lds_vnt: fx.Array[fx.BFloat16, LDS_VNT_ELEMS, 16]
            lds_h: fx.Array[fx.BFloat16, LDS_H_ELEMS, 16]

    # Cooperative load parameters (bf16x8 = dwordx4). Two decompositions of the
    # [BT, K] w tile:
    #  * BATCHED: thread -> (k-block, row-batch). The default wherever valid.
    #  * LINEAR: slot s = i*BLOCK_THREADS + tid over the whole tile. Needed once
    #    BLOCK_THREADS > BT * THREADS_PER_ROW_64, where the batched form would
    #    divide BT by a rows-per-batch larger than BT and yield 0 batches.
    #
    # Both give W_THREADS_PER_ROW-consecutive tids one contiguous row segment, so
    # global coalescing is equivalent.
    LOAD_VEC_WIDTH = 8
    THREADS_PER_ROW_64 = 64 // LOAD_VEC_WIDTH  # 8
    ROWS_PER_BATCH_64 = BLOCK_THREADS // THREADS_PER_ROW_64
    W_BATCHED = ROWS_PER_BATCH_64 <= BT and BT % ROWS_PER_BATCH_64 == 0
    assert W_BATCHED, "Must have batched w-tensor load"
    NUM_LOAD_BATCHES_64 = BT // ROWS_PER_BATCH_64

    STRIDE_U_C = V if WU_CONTIGUOUS else H * V
    STRIDE_W_C = K if WU_CONTIGUOUS else H * K
    STRIDE_Q_C = Hg * K
    # Half a LOAD_VEC_WIDTH run: the XOR swizzle splits a thread's bf16x8 into two
    # 4-element LDS groups that are not adjacent (see the g2s block in the kernel).
    LDS_HALF = LOAD_VEC_WIDTH // 2

    W_THREADS_PER_ROW = K // LOAD_VEC_WIDTH  # 16 for K=128
    W_SLOTS = BT * W_THREADS_PER_ROW  # 1024 for BT=64, K=128
    assert W_SLOTS % BLOCK_THREADS == 0, (
        f"w tile ({W_SLOTS} vec{LOAD_VEC_WIDTH} slots) must tile "
        f"BLOCK_THREADS={BLOCK_THREADS}"
    )
    W_LOADS_PER_THREAD = (
        NUM_K_BLOCKS * NUM_LOAD_BATCHES_64 if W_BATCHED else W_SLOTS // BLOCK_THREADS
    )

    BT_STEPS = BT // MFMA_K  # 4

    # -- k store-transpose decomposition --
    # k arrives from HBM as runs along K but lds_kt wants runs along BT, so the
    # store is a genuine transpose. Giving a thread one row x 8 k-cols would
    # scatter its 8 elements over 8 lds_kt rows (8 scalar ds_write_b16). Instead
    # each thread takes 4 BT-CONSECUTIVE rows at the same k-cols, so per k-col
    # the 4 values form one bt-group and an in-register transpose turns those
    # writes into one packed ds_write_b64 each. A "slot" is one (row-quad,
    # k-col-group) pair. The ROW QUAD stays 4 regardless -- that is what makes
    # the store packed -- so a wider block costs load width, not LDS width:
    #   256 thr -> vec8 (dwordx4) | 512 thr -> vec4 (dwordx2) | 1024 thr -> vec2
    K_VEC_WIDTH = min(LOAD_VEC_WIDTH, max(2, (BT // 4) * K // BLOCK_THREADS))
    K_COL_GROUPS = K // K_VEC_WIDTH
    K_ROW_QUADS = BT // 4
    K_XPOSE_SLOTS = K_ROW_QUADS * K_COL_GROUPS
    K_SLOTS_PER_THREAD = K_XPOSE_SLOTS // BLOCK_THREADS
    # Row-quads covered per pass.
    K_ROW_QUAD_STRIDE = BLOCK_THREADS // K_COL_GROUPS
    STRIDE_K_C = Hg * K

    assert K & (K - 1) == 0, (
        f"K={K} must be a power of two: the k transpose staging needs a "
        f"power-of-two K_VEC_WIDTH (got {K_VEC_WIDTH}) and needs "
        f"K_XPOSE_SLOTS={K_XPOSE_SLOTS} to tile BLOCK_THREADS={BLOCK_THREADS}"
    )
    assert K_XPOSE_SLOTS % BLOCK_THREADS == 0, (
        f"k transpose slots ({K_XPOSE_SLOTS}) must tile "
        f"BLOCK_THREADS={BLOCK_THREADS}"
    )
    assert (
        4 * K_ROW_QUAD_STRIDE <= BT and BT % (4 * K_ROW_QUAD_STRIDE) == 0
    ), f"k transpose store tile ({4 * K_ROW_QUAD_STRIDE} rows) must tile BT={BT}"

    _kernel_deco_kwargs = (
        {} if BLOCK_THREADS == 256 else {"known_block_size": [BLOCK_THREADS, 1, 1]}
    )

    _kernel_name = (
        (
            "chunk_gdn_fwd_h_o_flydsl_vk"
            if COMPUTE_OUTPUT
            else "chunk_gdn_fwd_h_flydsl_vk"
        )
        + f"_{_variant_tag(BV, NUM_WARPS)}"
        + _BF16_KERNEL_SUFFIX
    )

    @flyc.kernel(name=_kernel_name, **_kernel_deco_kwargs)
    def gdn_h_kernel(
        k_tensor: fx.Tensor,
        u_tensor: fx.Tensor,
        w_tensor: fx.Tensor,
        v_new_tensor: fx.Tensor,
        g_tensor: fx.Tensor,
        gk_tensor: fx.Tensor,
        h_tensor: fx.Tensor,
        h0_tensor: fx.Tensor,
        ht_tensor: fx.Tensor,
        cu_seqlens_tensor: fx.Tensor,
        chunk_offsets_tensor: fx.Tensor,
        q_tensor: fx.Tensor,
        o_tensor: fx.Tensor,
        T_val: fx.Int32,
        T_flat: fx.Int32,
        N_val: fx.Int32,
    ):
        if const_expr(NXCD > 0):
            # Flatten the 2D block id the way the dispatcher does
            # (xy = x + gridDim.x * y), invert the round-robin so that runs of
            # GRID_V consecutive logical ids land on one XCD, then unflatten back
            # to (i_v, i_nh). Head runs are already GRID_V-contiguous in logical
            # order, so this co-locates each head's V-tiles.
            grid_nh_rt = N_val * fx.Int32(H)
            grid_total = fx.Int32(GRID_V) * grid_nh_rt
            xy = fx.block_idx.x + fx.Int32(GRID_V) * fx.block_idx.y
            xcd = xy % fx.Int32(NXCD)
            # Tail guard: ids past the last full NXCD*GRID_V cycle pass through
            # unchanged (remapping them would collide / go out of range). So any
            # grid is at least as good as the round-robin baseline.
            cycle = fx.Int32(NXCD) * fx.Int32(GRID_V)
            last_full = (grid_total // cycle) * cycle
            local_id = xy // fx.Int32(NXCD)
            chunk_idx = local_id // fx.Int32(GRID_V)
            pos = local_id % fx.Int32(GRID_V)
            remapped = chunk_idx * cycle + xcd * fx.Int32(GRID_V) + pos
            logical = (xy < last_full).select(remapped, xy)
            i_v = logical % fx.Int32(GRID_V)
            i_nh = logical // fx.Int32(GRID_V)
        else:
            i_v = fx.block_idx.x
            i_nh = fx.block_idx.y

        i_n = i_nh // fx.Int32(H)
        i_h = i_nh % fx.Int32(H)

        tid = fx.thread_idx.x
        wid = tid // fx.Int32(WARP_SIZE)
        lane = tid % fx.Int32(WARP_SIZE)

        # Wave split (see NR_SPLIT above). At NR_SPLIT == 1, wid_n is identically
        # 0 and wid_m is wid, so the generated code collapses to the 4-wave form.
        if const_expr(NR_SPLIT == 1):
            wid_m = wid
        else:
            wid_m = wid % fx.Int32(M_WAVES)
        wid_n = wid // fx.Int32(M_WAVES)

        def _nr_v(nr_local):
            """V-offset (elements) of this wave's local column tile ``nr_local``.

            The global tile index is ``nr_local * NR_SPLIT + wid_n``, times 16.
            """
            if const_expr(NR_SPLIT == 1):
                return fx.Int32(nr_local * 16)
            return (fx.Int32(nr_local * NR_SPLIT) + wid_n) * fx.Int32(16)

        def _elems(tensor):
            """Element count of ``tensor``'s whole footprint."""
            return fx.get_scalar(fx.cosize(fx.get_layout(tensor)))

        def _seq_view(tensor, base_elems, rows, row_stride, shape, stride):
            """Buffer view rooted at ``base_elems``, bounded to ``rows`` rows.

            ``add_offset`` runs on the raw pointer, so the sequence/head base
            lands in the descriptor's base word and ``num_records`` is measured from it.
            This lets the bound to become per-sequence instead of per-tensor.

            The hardware test is ``offset >= num_records``, so a bound of
            ``rows * row_stride`` elements excludes row ``rows`` exactly while
            admitting every column of rows ``0 .. rows-1`` (every tensor here
            has ``row_stride >= innermost extent``). A read past the sequence
            therefore returns a hardware zero rather than the neighbouring
            sequence's live data. Writes are bounded the same way: a store past
            ``num_records`` is discarded.

            ``num_records`` is in bytes, and the width is taken from the tensor
            rather than assumed: k/w/u/q are bf16 but the gates are f32, and
            hard-coding 2 here silently halves every gate descriptor -- which
            zeroes the back half of every sequence, not just its tail.
            """
            elem_bytes = tensor.element_type.width // 8
            assert elem_bytes * 8 == tensor.element_type.width, (
                "_seq_view needs a byte-sized element type; got "
                f"{tensor.element_type.width} bits"
            )
            it = fx.add_offset(fx.get_iter(tensor), base_elems)
            view = fx.make_view(it, fx.make_layout(shape, stride))
            return fx.Tensor(
                fx.rocdl.make_buffer_tensor(
                    view,
                    num_records_bytes=fx.Int64(rows * row_stride)
                    * fx.Int64(elem_bytes),
                )
            )

        if const_expr(IS_VARLEN):
            cu_buf = fx.rocdl.make_buffer_tensor(cu_seqlens_tensor, max_size=False)
            co_buf = fx.rocdl.make_buffer_tensor(chunk_offsets_tensor, max_size=False)

        # -- MMA atom --
        # One 16x16x16 bf16 MFMA per wave, replicated 1x1x1.
        mma_atom_bf16_16x16x16 = fx.make_mma_atom(
            fx.rocdl.MFMA(MFMA_M, MFMA_N, MFMA_K, fx.BFloat16)
        )
        # -- Multi-tile 16x16x16 MMA shared by all GEMMs --
        # The wave grid is (M_WAVES over the M tile, NR_SPLIT over the N=V tile,
        # 1 over K), with wave numbering wave = wid_m + M_WAVES*wid_n -> stride
        # (1, M_WAVES, 0). Reused by GEMM1 (M=BT), GEMM2 (M=64 K-rows/block), and
        # the later native GEMMs -- they share the same wave/atom decomposition,
        # only the M/N/K tile *sizes* passed to make_fragment/partition differ.
        mma_16x16x16_mnk = fx.make_tiled_mma(
            mma_atom_bf16_16x16x16,
            fx.make_layout((M_WAVES, NR_SPLIT, 1), (1, M_WAVES, 0)),
        )

        # LDS -> register copy atoms.
        cp_lds_x4 = fx.make_copy_atom(fx.UniversalCopy64b(), fx.BFloat16)
        cp_lds_x1 = fx.make_copy_atom(fx.UniversalCopy16b(), fx.BFloat16)

        # C-operand tiled copies, for spilling accumulators to LDS.
        tc_c_x4 = fx.make_tiled_copy_C(cp_lds_x4, mma_16x16x16_mnk).get_slice(tid)
        if const_expr(USE_INITIAL_STATE or STORE_FINAL_STATE):
            if const_expr(STATE_DTYPE_BF16):
                cp_state = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)
            else:
                cp_state = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
            tc_c_state = fx.make_tiled_copy_C(cp_state, mma_16x16x16_mnk).get_slice(tid)
        if const_expr(COMPUTE_OUTPUT):
            tc_c_x1 = fx.make_tiled_copy_C(cp_lds_x1, mma_16x16x16_mnk).get_slice(tid)

        # u is consumed as GEMM1's C operand (v_new = u - w @ h), so it loads
        # straight into the C-fragment layout: 4 BT rows x 1 V column per tile.
        # Those 4 rows are stride_v apart in HBM, so 16b is the widest atom
        # available.
        cp_u_g2r = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        tc_c_u = fx.make_tiled_copy_C(cp_u_g2r, mma_16x16x16_mnk).get_slice(tid)

        def _cfrag_to_lds(atom, tc, dst_view, acc_vec, n):
            """Convert an f32 accumulator to bf16 and copy it into ``dst_view``."""
            pD = tc.partition_D(dst_view)
            frag = fx.make_fragment_like(pD)
            frag.store(_to_bf16(acc_vec, n))
            fx.copy(atom, frag, pD)

        # -- LDS views --
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_w_ptr = lds.lds_w.ptr
        lds_kt_ptr = lds.lds_kt.ptr
        lds_vnt_ptr = lds.lds_vnt.ptr
        lds_h_ptr = lds.lds_h.ptr
        if const_expr(COMPUTE_OUTPUT):
            # lds_A aliases lds_h when LDS is tight: GEMM3, the last lds_h
            # reader, runs before GEMM4a, the lds_A writer, with a barrier
            # between (see ALIAS_A_ONTO_H above).
            if const_expr(ALIAS_A_ONTO_H):
                lds_A_ptr = lds_h_ptr
            else:
                lds_A_ptr = lds.lds_A.ptr

        # Every LDS buffer is a group-major [R][C/4][4] tile whose group index is
        # XOR-swizzled by the row, so an MFMA fragment's 16 lanes cover all 32 banks.
        def _single_xor_swz(cols, ng):
            return fx.SwizzleType.get(int(math.log2(ng)), 2, int(math.log2(cols)) - 2)

        # -- Hand-addressed group-major + XOR (K5 k^T / v_new^T only) --
        # The composed-layout swizzle above expresses only a single XOR fold
        # (group ^ (row & (ng-1))). That is correct for every buffer whose hot
        # site reads/writes an MFMA fragment, whose 16 rows vary in the low bits.
        # The k store-transpose is different: it writes rows (tid%16)*8 + e, and
        # across the 16 lanes of one fragment those low 4 bits take only two
        # values -> 8-way bank multiplicity. flydsl's layout-lowering drops the
        # nested/second XOR needed to key the swizzle on the bits that site does
        # vary (row >> 3), so lds_kt (and lds_vnt, stored the same transposed
        # way) are addressed by hand here. _grp_idx folds row bits 3+ onto the
        # low bits before the XOR, which puts both the transpose write and the
        # GEMM2 fragment read at their conflict floor. The XOR is a bijection on
        # the group index, so store and load -- deriving the mask from the same
        # row -- stay consistent.
        def _grp_idx(row, grp, cols, ng):
            mask = (row ^ (row >> fx.Int32(3))) & fx.Int32(ng - 1)
            return row * fx.Int32(cols) + ((grp ^ mask) * fx.Int32(4))

        def _lds_kt_idx(k_row, bt_grp):
            return _grp_idx(k_row, bt_grp, BT, LDS_KT_NG)

        def _lds_vnt_idx(v_local, bt_grp):
            return _grp_idx(v_local, bt_grp, BT, LDS_VNT_NG)

        # 4 bf16 = 8 B: one ds_read_b64 / ds_write_b64, one MFMA A/B fragment.
        v4bf16_type = T.vec(4, T.bf16)

        # -- C-fragment destinations --
        # An MFMA C fragment holds 4 consecutive M values at one N. For every
        # accumulator, the M axis is the fast axis of the destination buffer
        # and the 4 values are one packed b64 transaction.
        # Hence, we need a view that presents the buffer in (M, N) order:
        #   h    (GEMM2, M=k-row) -> lds_h  is [BV, K]  -> swap strides -> (K, BV)
        #   vnt  (GEMM1, M=BT)    -> lds_vnt is [BV,BT] -> swap strides -> (BT, BV)
        # lds_A is the exception: GEMM4a's M is the query row while lds_A is
        # [BT, BT] with the key column contiguous, so its natural orientation is
        # already (M, N) and no swap is needed. However, the 4 values are BT
        # apart and it must store through cp_lds_x1 (4 copy atoms).
        swz_h = fx.static(_single_xor_swz(K, LDS_H_NG))
        swz_vnt = fx.static(_single_xor_swz(BT, LDS_VNT_NG))
        sH_C = fx.make_view(
            lds_h_ptr, fx.make_composed_layout(swz_h, fx.make_layout((K, BV), (1, K)))
        )
        sVNT_C = fx.make_view(
            lds_vnt_ptr,
            fx.make_composed_layout(swz_vnt, fx.make_layout((BT, BV), (1, BT))),
        )
        if const_expr(COMPUTE_OUTPUT):
            swz_A = fx.static(_single_xor_swz(BT, LDS_A_NG))
            # Natural (M=query row, N=key col); also GEMM4b's A-operand view.
            sA = fx.make_view(
                lds_A_ptr,
                fx.make_composed_layout(
                    swz_A, fx.make_ordered_layout((BT, BT), (1, 0))
                ),
            )

        # -- Prologue: compute bos, T_local, NT, boh --
        # boh (the chunk-offset base) only addresses the h snapshot, so it -- and
        # the chunk_offsets read that produces it -- is skipped when not STORE_H.
        if const_expr(IS_VARLEN):
            bos = cu_buf[(i_n,)]
            eos = cu_buf[(i_n + fx.Int32(1),)]
            T_local = eos - bos
            NT = (T_local + fx.Int32(BT - 1)) // fx.Int32(BT)
            if const_expr(STORE_H):
                boh = co_buf[(i_n,)]
        else:
            bos = i_n * T_val
            T_local = T_val
            NT = (T_local + fx.Int32(BT - 1)) // fx.Int32(BT)
            if const_expr(STORE_H):
                boh = i_n * NT

        # -- Base pointer offsets (element counts) --
        if const_expr(STORE_H):
            h_base = (boh * fx.Int32(H) + i_h) * fx.Int32(V * K)
            stride_h = fx.Int32(H * V * K)

        gqa_ratio = H // Hg
        k_base = (bos * fx.Int32(Hg) + i_h // fx.Int32(gqa_ratio)) * fx.Int32(K)
        stride_k = fx.Int32(Hg * K)
        if const_expr(COMPUTE_OUTPUT):
            # q shares k's [B, T, Hg, K] layout.
            q_base = k_base
            stride_q = stride_k

        if const_expr(WU_CONTIGUOUS):
            if const_expr(IS_VARLEN):
                v_base = (i_h * T_flat + bos) * fx.Int32(V)
                w_base = (i_h * T_flat + bos) * fx.Int32(K)
            else:
                v_base = ((i_n * fx.Int32(H) + i_h) * T_flat) * fx.Int32(V)
                w_base = ((i_n * fx.Int32(H) + i_h) * T_flat) * fx.Int32(K)
            stride_v = fx.Int32(V)
            stride_w = fx.Int32(K)
        else:
            v_base = (bos * fx.Int32(H) + i_h) * fx.Int32(V)
            w_base = (bos * fx.Int32(H) + i_h) * fx.Int32(K)
            stride_v = fx.Int32(H * V)
            stride_w = fx.Int32(H * K)

        if const_expr(USE_G):
            # ``g`` is head-major [B, H, T_flat], so its batch stride is
            # H*T_flat -- unlike the token-major tensors, ``bos`` cannot be
            # folded into the row index. It goes into the descriptor base
            # instead, leaving the in-loop index sequence-relative.
            if const_expr(IS_VARLEN):
                g_base = i_h * T_flat + bos
            else:
                g_base = (i_n * fx.Int32(H) + i_h) * T_flat

        # This CTA owns the [i_v*BV, (i_v+1)*BV) column window of u, so the
        # column offset folds into the base and gU is a [BT, BV] view.
        u_base = v_base + i_v * fx.Int32(BV)

        # -- Gate buffers: rooted here, not at the top, because the base and the
        # -- bound both need bos / T_local from the prologue above.
        # The layout shape is only the addressing DOMAIN -- what actually limits
        # a read is num_records -- so it stays whole-tensor-sized, exactly as it
        # was before, and only the base and the bound change.
        if const_expr(USE_G):
            # Head-major [B, H, T_flat], so this head's run of this sequence's
            # tokens is contiguous and the bound is simply T_local elements.
            g_buf = _seq_view(
                g_tensor, g_base, T_local, fx.Int32(1), (_elems(g_tensor),), (1,)
            )
        if const_expr(USE_GK):
            # [n/4, 4]: one row is gk's 4-wide contiguous quad. The element
            # offset is always 4-aligned -- K % 4 == 0 (asserted above) and
            # every addend (H*K, i_h*K, kb*64, wid_m*16, lane_m_base*4) is a
            # multiple of 4 -- so quad // 4 is exact.
            gk_base = bos * fx.Int32(H * K) + i_h * fx.Int32(K)
            gk_buf = _seq_view(
                gk_tensor,
                gk_base,
                T_local,
                fx.Int32(H * K),
                (_elems(gk_tensor) // 4, 4),
                (4, 1),
            )

        # -- Tiled w staging: HBM [BT, K] tile -> lds_w --
        # Two destination views:
        # sW_hi is sW_lo composed with a + LDS_HALF offset, which addresses the hi group
        # from the LO tile coordinate.
        w_inner_layout = fx.make_ordered_layout((BT, K), (1, 0))
        w_swz = fx.static(_single_xor_swz(K, LDS_W_NG))
        sW_lo = fx.make_view(lds_w_ptr, fx.make_composed_layout(w_swz, w_inner_layout))
        sW_hi = fx.make_view(
            lds_w_ptr, fx.make_composed_layout(w_swz, LDS_HALF, w_inner_layout)
        )

        # General tiled copy definitions.
        copy_tile = fx.make_tile(ROWS_PER_BATCH_64, LOAD_VEC_WIDTH * THREADS_PER_ROW_64)
        _tv_thr = (THREADS_PER_ROW_64, ROWS_PER_BATCH_64)
        _tv_thr_stride = (ROWS_PER_BATCH_64 * LOAD_VEC_WIDTH, 1)
        tv_load = fx.make_layout(
            (_tv_thr, (1, LOAD_VEC_WIDTH)),
            (_tv_thr_stride, (1, ROWS_PER_BATCH_64)),
        )
        tv_store = fx.make_layout(
            (_tv_thr, (1, LDS_HALF)),
            (_tv_thr_stride, (1, ROWS_PER_BATCH_64)),
        )
        buf_cp_g2r_128b_bf16 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        univ_cp_r2s_64b_bf16 = fx.make_copy_atom(fx.UniversalCopy64b(), fx.BFloat16)
        tiled_cp_g2r = fx.make_tiled_copy(
            buf_cp_g2r_128b_bf16, tv_load, copy_tile
        ).get_slice(tid)
        tiled_cp_r2s = fx.make_tiled_copy(
            univ_cp_r2s_64b_bf16, tv_store, copy_tile
        ).get_slice(tid)

        # WU decomposition layout definitions. Both are rooted at this CTA's
        # (sequence, head) -- and for u, its V-column window -- and bounded to
        # T_local rows, so the chunk offset is all that rides in soffset and a
        # tail row reads a hardware zero.
        gW = _seq_view(w_tensor, w_base, T_local, stride_w, (BT, K), (STRIDE_W_C, 1))
        gU = _seq_view(u_tensor, u_base, T_local, stride_v, (BT, BV), (STRIDE_U_C, 1))
        pS_w = tiled_cp_g2r.partition_S(gW)
        # u goes through the MMA's C partitioning, not tiled_cp_g2r: its
        # consumer wants the C-fragment layout, not a row-contiguous tile.
        pS_u = tc_c_u.partition_S(gU)
        pD_w_lo = tiled_cp_r2s.partition_D(sW_lo)
        pD_w_hi = tiled_cp_r2s.partition_D(sW_hi)

        if const_expr(COMPUTE_OUTPUT):
            gQ = _seq_view(
                q_tensor, q_base, T_local, stride_q, (BT, K), (STRIDE_Q_C, 1)
            )
            pS_q = tiled_cp_g2r.partition_S(gQ)

        # -- Tiled k staging --
        swz_kt = fx.static(_single_xor_swz(LDS_KT_COLS, LDS_KT_NG))
        if const_expr(KT_TRANSPOSED):
            # HBM [BT, K] -> lds_kt [K, BT]. This is a genuine transpose and its
            # LDS write is hand-addressed (see _grp_idx above): the composed
            # swizzle cannot express the row>>3 fold the transpose store needs.
            # Thread -> (row-quad, k-col-group): consecutive tids walk k-col
            # groups, so a full K row is covered by K_COL_GROUPS consecutive
            # threads (contiguous HBM). Each thread owns K_SLOTS_PER_THREAD
            # slots; a slot is 4 BT-consecutive rows at one K_VEC_WIDTH-wide
            # k-col group, so the 4 rows form one bt-group -> one ds_write_b64.
            # [n/K_VEC_WIDTH, K_VEC_WIDTH]: one row is one contiguous k-col
            # group. The offset is always K_VEC_WIDTH-aligned (stride_k is a
            # multiple of K and kx_col_base is a multiple of K_VEC_WIDTH), so
            # off // K_VEC_WIDTH is exact. Rooted at (sequence, head) and
            # bounded to T_local rows, so a tail row reads a hardware zero.
            k_kt_buf = _seq_view(
                k_tensor,
                k_base,
                T_local,
                stride_k,
                (_elems(k_tensor) // fx.Int32(K_VEC_WIDTH), K_VEC_WIDTH),
                (K_VEC_WIDTH, 1),
            )
            kx_col_base = (tid % fx.Int32(K_COL_GROUPS)) * fx.Int32(K_VEC_WIDTH)
            kx_row_quad = tid // fx.Int32(K_COL_GROUPS)
        else:
            # HBM [BT, K] -> lds_kt [BT, K]: source and destination agree.
            sK_lo = fx.make_view(
                lds_kt_ptr, fx.make_composed_layout(swz_kt, w_inner_layout)
            )
            sK_hi = fx.make_view(
                lds_kt_ptr,
                fx.make_composed_layout(swz_kt, LDS_HALF, w_inner_layout),
            )
            gK = _seq_view(
                k_tensor, k_base, T_local, stride_k, (BT, K), (STRIDE_K_C, 1)
            )
            pS_k = tiled_cp_g2r.partition_S(gK)
            pD_k_lo = tiled_cp_r2s.partition_D(sK_lo)
            pD_k_hi = tiled_cp_r2s.partition_D(sK_hi)

        if const_expr(STORE_H):
            # -- h snapshot drain: lds_h -> HBM  --
            # Thread -> (k-group pair, v row): TPR pairs cover one v row, so the
            # block covers H_DRAIN_ROWS rows per pass and the rest modes walk v.
            H_DRAIN_TPR = LDS_H_NG // 2  # k-group pairs per v row = K // 8
            H_DRAIN_ROWS = BLOCK_THREADS // H_DRAIN_TPR
            H_DRAIN_RESTS = BV // H_DRAIN_ROWS
            assert BV % H_DRAIN_ROWS == 0 and LDS_H_NG % 2 == 0, (
                f"h drain: BV={BV} must tile {H_DRAIN_ROWS} rows/pass and K/4="
                f"{LDS_H_NG} must be even"
            )
            _h_tv = lambda vw: fx.make_layout(
                ((H_DRAIN_TPR, H_DRAIN_ROWS), (1, vw)),
                ((H_DRAIN_ROWS * LOAD_VEC_WIDTH, 1), (1, H_DRAIN_ROWS)),
            )
            _h_tile = fx.make_tile(H_DRAIN_ROWS, K)
            _h_inner = fx.make_ordered_layout((BV, K), (1, 0))
            sH_lo = fx.make_view(lds_h_ptr, fx.make_composed_layout(swz_h, _h_inner))
            sH_hi = fx.make_view(
                lds_h_ptr, fx.make_composed_layout(swz_h, LDS_HALF, _h_inner)
            )
            tc_h_s2r = fx.make_tiled_copy(
                cp_lds_x4, _h_tv(LDS_HALF), _h_tile
            ).get_slice(tid)
            cp_h_r2g = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
            tc_h_r2g = fx.make_tiled_copy(
                cp_h_r2g, _h_tv(LOAD_VEC_WIDTH), _h_tile
            ).get_slice(tid)
            pS_h_lo = tc_h_s2r.partition_S(sH_lo)
            pS_h_hi = tc_h_s2r.partition_S(sH_hi)
            h_git = fx.get_iter(fx.rocdl.make_buffer_tensor(h_tensor, max_size=False))
            pD_h_g = tc_h_r2g.partition_D(
                fx.Tensor(fx.make_view(h_git, fx.make_layout((BV, K), (K, 1))))
            )

        def _stage_g2r(atom, pS, stride, it_i32):
            """Issue chunk ``it_i32``'s [BT, *] tile into a register fragment.

            ``atom`` MUST be the atom ``pS`` was partitioned with. ``stride`` is
            the row pitch, so the chunk offset is ``it_i32 * BT`` rows. The
            offset is sequence-relative: the (sequence, head) base already sits
            in the descriptor (see ``_seq_view``), which is what makes the
            hardware bound per-sequence.
            """
            frag = fx.make_fragment_like(pS)
            fx.copy(
                atom,
                pS,
                frag,
                soffset=it_i32 * fx.Int32(BT) * stride,
            )
            return frag.load()

        def _stage_r2s(w_vec, dst_lo, dst_hi):
            """Store a staged [BT, K] fragment into LDS as two 4-element halves."""
            v = fx.Vector(w_vec)
            halves = []
            for off in (0, LDS_HALF):
                halves.append(
                    fx.Vector.from_elements(
                        [
                            v[j * LOAD_VEC_WIDTH + off + e]
                            for j in range_constexpr(W_LOADS_PER_THREAD)
                            for e in range_constexpr(LDS_HALF)
                        ],
                        dtype=fx.BFloat16,
                    )
                )
            f_lo = fx.make_fragment_like(dst_lo)
            f_hi = fx.make_fragment_like(dst_hi)
            f_lo.store(halves[0])
            f_hi.store(halves[1])
            fx.copy(univ_cp_r2s_64b_bf16, f_lo, dst_lo)
            fx.copy(univ_cp_r2s_64b_bf16, f_hi, dst_hi)

        # -- Output buffers: rooted per-sequence, so the hardware write drops the tail --
        # A buffer_store whose offset is >= num_records is discarded by the hardware.
        # Rooting the descriptor at the sequence therefore makes the padding rows of the
        # final chunk store nowhere.
        #
        # Offsets at the store sites are consequently sequence-relative, not
        # absolute, the base now lives in the descriptor.
        if const_expr(SAVE_NEW_VALUE):
            # v_new is head-major, one row per token, so the row stride is V and
            # the bound is T_local rows measured from this head's row 0.
            if const_expr(IS_VARLEN):
                vn_base = (i_h * T_flat + bos) * fx.Int32(V)
            else:
                vn_base = ((i_n * fx.Int32(H) + i_h) * T_flat) * fx.Int32(V)
            vn_buf = _seq_view(
                v_new_tensor,
                vn_base,
                T_local,
                fx.Int32(V),
                (_elems(v_new_tensor),),
                (1,),
            )

        if const_expr(COMPUTE_OUTPUT):
            # o is token-major [B, T_flat, H, V] (matches the Triton K6 output).
            # The base carries a column term (i_h*V) as well as the row term, so
            # the descriptor is rooted mid-row. That is still exact: a store at
            # row r < T_local, column c < V lands at r*H*V + c <= (T_local-1)*H*V
            # + V-1 < T_local*H*V, while row T_local starts at exactly
            # T_local*H*V and is dropped. No generated offset falls in the gap
            # between those two, so the bound separates the two cases cleanly.
            o_base = (bos * fx.Int32(H) + i_h) * fx.Int32(V)
            stride_o = fx.Int32(H * V)
            o_buf = _seq_view(
                o_tensor, o_base, T_local, stride_o, (_elems(o_tensor),), (1,)
            )

        if const_expr(USE_INITIAL_STATE):
            h0_base = i_nh * fx.Int32(V * K)
        if const_expr(STORE_FINAL_STATE):
            ht_base = i_nh * fx.Int32(V * K)

        # -- MFMA lane mapping for 16x16 tiles --
        lane_n = lane % fx.Int32(16)
        lane_m_base = lane // fx.Int32(16)

        # -- Initialize h accumulators  --
        h_accs_c = [
            fx.make_rmem_tensor(
                fx.tiled_mma_partition_shape(
                    fx.MmaOperand.C, mma_16x16x16_mnk, (64, BV)
                ),
                fx.Float32,
            )
            for _ in range_constexpr(NUM_K_BLOCKS)
        ]
        for frag in h_accs_c:
            frag.store(fx.Vector.filled(N_REPEAT_LOCAL * 4, 0.0, fx.Float32))

        # -- Load initial state if provided --
        # h0 is [V, K] with k contiguous, and the C fragment's 4 values run along k.
        # The (n, h) slice and this CTA's V window ride soffset.
        if const_expr(USE_INITIAL_STATE):
            gH0 = fx.Tensor(
                fx.make_view(
                    fx.get_iter(fx.rocdl.make_buffer_tensor(h0_tensor, max_size=False)),
                    fx.make_layout((K, BV), (1, K)),
                )
            )
            for kb in range_constexpr(NUM_K_BLOCKS):
                pS_h0 = tc_c_state.partition_S(
                    fx.slice(fx.zipped_divide(gH0, (64, BV)), (None, (kb, 0)))
                )
                f_h0 = fx.make_fragment_like(pS_h0)
                fx.copy(
                    cp_state,
                    pS_h0,
                    f_h0,
                    soffset=h0_base + i_v * fx.Int32(BV * K),
                )
                add_whole = fx.Vector(f_h0.load())
                if const_expr(STATE_DTYPE_BF16):
                    add_whole = add_whole.to(fx.Float32)
                cur = fx.Vector(h_accs_c[kb].load(), (N_REPEAT_LOCAL * 4,), fx.Float32)
                h_accs_c[kb].store(cur + add_whole)

        # -- Loop-carried prefetch --
        # ``_stage_prefetch`` and ``_unpack_prefetch`` are structural inverses:
        # the unpack walks the flat list in issue order, so neither side carries
        # an index into the other.
        def _stage_prefetch(it_i32):
            """Issue every chunk-``it_i32`` load that rides the carried state.

            Order: w tile, u tile, then g_last / 4 g_row / one gk quad per
            64-wide K block.

            Raw loads only: no exp, no in-bounds select, no vector packing.
            Nothing in the issuing iteration depends on the results and the
            loads never force a wait where they are issued.
            """
            out = [
                _stage_g2r(buf_cp_g2r_128b_bf16, pS_w, stride_w, it_i32),
                _stage_g2r(cp_u_g2r, pS_u, stride_v, it_i32),
            ]
            next_end = (it_i32 + fx.Int32(1)) * fx.Int32(BT)
            # Semantic clamp, NOT a bounds guard: the decay factor is the LAST
            # VALID token's gate, so a hardware zero here would be wrong.
            last_idx = (next_end < T_local).select(next_end, T_local) - fx.Int32(1)
            row_base = (
                it_i32 * fx.Int32(BT) + wid_m * fx.Int32(16) + lane_m_base * fx.Int32(4)
            )
            if const_expr(USE_G):
                out.append(g_buf[(last_idx,)])
                for elem_i in range_constexpr(4):
                    # No address clamp: g_buf is bounded to this sequence's
                    # T_local rows, so a row past the sequence end reads a
                    # hardware zero -- and the gate mask discards it anyway
                    # (exp(g_last - 0) is not 0, so that mask has to stay).
                    out.append(g_buf[(row_base + fx.Int32(elem_i),)])
            if const_expr(USE_GK):
                for kb in range_constexpr(NUM_K_BLOCKS):
                    quad = (
                        last_idx * fx.Int32(H * K)
                        + fx.Int32(kb * 64)
                        + wid_m * fx.Int32(16)
                        + lane_m_base * fx.Int32(4)
                    )
                    out.append(gk_buf[(quad // fx.Int32(4), None)].load())
            return out

        def _unpack_prefetch(carried):
            """Inverse of ``_stage_prefetch``; returns (w, u, g_last, g_row, gk)."""
            vals = iter(carried)
            w_frag = next(vals)
            u_frag = next(vals)
            g_last = next(vals) if const_expr(USE_G) else None
            g_row = (
                [next(vals) for _ in range_constexpr(4)] if const_expr(USE_G) else None
            )
            gk = (
                [next(vals) for _ in range_constexpr(NUM_K_BLOCKS)]
                if const_expr(USE_GK)
                else None
            )
            return w_frag, u_frag, g_last, g_row, gk

        # -- Prologue: pre-load the first chunk --
        init_state = _stage_prefetch(fx.Int32(0))

        for i_t, state in range(
            fx.Int64(0), fx.Int64(NT), fx.Int64(1), init=init_state
        ):
            (
                w_frag,
                u_frag,
                g_last_carried,
                g_row_carried,
                gk_quads,
            ) = _unpack_prefetch(state)
            i_t_i32 = fx.Int32(i_t)

            # -- C-fragment row coordinates --
            # The MMA's C fragment gives each thread 4 rows of the BT tile:
            # wid_m*16 + lane_m_base*4 + e. All our MFMA output have this form.
            frag_row_local = [
                wid_m * fx.Int32(16) + lane_m_base * fx.Int32(4) + fx.Int32(e)
                for e in range_constexpr(4)
            ]
            frag_row = [i_t_i32 * fx.Int32(BT) + r for r in frag_row_local]

            # -- Store h snapshot to LDS (group-major [BV][K/4][4] + XOR) --
            # h_accs element e = h[v_local = nr*16 + lane_n, k = kb*64 + wid*16 +
            #                      lane_m_base*4 + e].  The four e's are one
            # k-group, so the whole f32x4 accumulator packs into a single bf16x4
            # (one ds_write_b64) instead of 4 scalar ds_write_b16.
            for kb in range_constexpr(NUM_K_BLOCKS):
                _cfrag_to_lds(
                    cp_lds_x4,
                    tc_c_x4,
                    fx.slice(fx.zipped_divide(sH_C, (64, BV)), (None, (kb, 0))),
                    fx.Vector(h_accs_c[kb].load(), (N_REPEAT_LOCAL * 4,), fx.Float32),
                    N_REPEAT_LOCAL * 4,
                )

            # -- Store prefetched w to LDS (two b64 halves per bf16x8) --
            # Staged BEFORE the barrier so one barrier publishes both lds_h and
            # lds_w. w_frag is loop-carried, so this has no dependency on the
            # drain below and moving it up only shortens its live range. The
            # WAR against the previous chunk's GEMM1 reads of lds_w is covered
            # by the post-lds_kt/lds_vnt barrier at the end of the body.
            _stage_r2s(w_frag, pD_w_lo, pD_w_hi)

            gpu.barrier()

            # The drain reads lds_h groups written by other threads, so it has
            # to follow a barrier -- but not its own. Sitting after the shared
            # barrier instead of between two of them lets its LDS reads and the
            # HBM store overlap the GEMM1 MFMA chain, and costs two barriers per
            # chunk rather than three. (The fused build has no drain at all.)
            if const_expr(STORE_H):
                # -- LDS -> HBM h snapshot --
                # 8 contiguous k per thread is one 16 B HBM store,
                # but two non-adjacent b64 LDS reads, because
                # the XOR swizzle puts adjacent k-groups at non-adjacent slots.
                # The two 4-wide reads (lo/hi views) are reassembled into one
                # 8-wide global store.
                f_lo = fx.make_fragment_like(pS_h_lo)
                f_hi = fx.make_fragment_like(pS_h_hi)
                fx.copy(cp_lds_x4, pS_h_lo, f_lo)
                fx.copy(cp_lds_x4, pS_h_hi, f_hi)
                v_lo = fx.Vector(f_lo.load())
                v_hi = fx.Vector(f_hi.load())
                f_g = fx.make_fragment_like(pD_h_g)
                f_g.store(
                    fx.Vector.from_elements(
                        [
                            (v_lo if e < LDS_HALF else v_hi)[
                                j * LDS_HALF + (e % LDS_HALF)
                            ]
                            for j in range_constexpr(H_DRAIN_RESTS)
                            for e in range_constexpr(LOAD_VEC_WIDTH)
                        ],
                        dtype=fx.BFloat16,
                    )
                )
                offset = h_base + i_t_i32 * stride_h + i_v * fx.Int32(BV * K)
                fx.copy(
                    cp_h_r2g,
                    f_g,
                    pD_h_g,
                    soffset=offset,
                )

            # -- k prefetch (issued now, stored to LDS after GEMM1) --
            if const_expr(KT_TRANSPOSED):
                # Each thread owns K_SLOTS_PER_THREAD slots; a slot is 4
                # BT-consecutive rows at one K_VEC_WIDTH-wide k-col group. Rows
                # are gathered here so the transpose store below is a packed
                # ds_write_b64 with no cross-lane movement. k_prefetch[s][j] is
                # k[row = (kx_row_quad + s*K_ROW_QUAD_STRIDE)*4 + j,
                #   col = kx_col_base + (0..K_VEC_WIDTH-1)].
                #
                # No row clamp and no value mask. k_kt_buf is rooted at this
                # (sequence, head) and bounded to T_local rows, so a row past
                # the sequence end is out of range of the descriptor and the
                # hardware returns zero -- it can no longer reach the next
                # head/sequence's live data, which is what used to force the
                # clamp, and a hardware zero cannot be the Inf/NaN that used to
                # force the value mask.
                k_prefetch = []
                k_prefetch_lds_t = []  # row-quad per slot -> lds_kt[k, bt] group
                for s in range_constexpr(K_SLOTS_PER_THREAD):
                    row_quad = kx_row_quad + fx.Int32(s * K_ROW_QUAD_STRIDE)
                    quad_rows = []
                    for j in range_constexpr(4):
                        row = row_quad * fx.Int32(4) + fx.Int32(j)
                        abs_row = i_t_i32 * fx.Int32(BT) + row
                        k_off = abs_row * stride_k + kx_col_base
                        quad_rows.append(
                            fx.Vector(
                                k_kt_buf[(k_off // fx.Int32(K_VEC_WIDTH), None)].load()
                            )
                        )
                    k_prefetch.append(quad_rows)
                    k_prefetch_lds_t.append(row_quad)
            else:
                k_prefetch = _stage_g2r(buf_cp_g2r_128b_bf16, pS_k, stride_k, i_t_i32)

            # -- g / gk: type this chunk's values, prefetched last iter --
            # They come off the loop-carried state as bare IR values.
            def _as_f32(v):
                return fx.Float32(v)

            if const_expr(USE_G):
                g_last_val = _as_f32(g_last_carried)
                g_row_raw = [_as_f32(v) for v in g_row_carried]
            u_prefetch = fx.Vector(u_frag)

            # -- GEMM1: bv = w @ h  (contraction over K) --
            # A operand reads the very view the g2s copy wrote (see sW_lo).
            sW = sW_lo
            sH = fx.make_view(
                lds_h_ptr,
                fx.make_composed_layout(
                    fx.static(_single_xor_swz(K, LDS_H_NG)),
                    fx.make_ordered_layout((BV, K), (1, 0)),
                ),
            )
            g1_cp_a = fx.make_tiled_copy_A(cp_lds_x4, mma_16x16x16_mnk).get_slice(tid)
            g1_cp_b = fx.make_tiled_copy_B(cp_lds_x4, mma_16x16x16_mnk).get_slice(tid)
            g1_pS_w = g1_cp_a.partition_S(sW)
            g1_pS_h = g1_cp_b.partition_S(sH)
            g1_fw = mma_16x16x16_mnk.make_fragment_A(sW)
            g1_fh = mma_16x16x16_mnk.make_fragment_B(sH)
            g1_fw_rt = g1_cp_a.retile(g1_fw)
            g1_fh_rt = g1_cp_b.retile(g1_fh)
            frag_bv = fx.make_rmem_tensor(
                fx.tiled_mma_partition_shape(
                    fx.MmaOperand.C, mma_16x16x16_mnk, (BT, BV)
                ),
                fx.Float32,
            )
            frag_bv.fill(0.0)

            K_TILES = K // MFMA_K
            for kt in range_constexpr(K_TILES):
                fx.copy(cp_lds_x4, g1_pS_w[None, None, kt], g1_fw_rt[None, None, kt])
                fx.copy(cp_lds_x4, g1_pS_h[None, None, kt], g1_fh_rt[None, None, kt])
                fx.gemm(
                    mma_16x16x16_mnk,
                    frag_bv,
                    g1_fw[None, None, kt],
                    g1_fh[None, None, kt],
                    frag_bv,
                )

            # -- v_new = u - bv --
            # Consume the wide C-fragment frag_bv directly per V-tile.
            # frag_bv C-shape is ((4,1), 1, N_REPEAT_LOCAL).
            vn_frags = []
            for nr in range_constexpr(N_REPEAT_LOCAL):
                bv_val = fx.Vector(frag_bv[None, None, nr].load())
                u_f32_elems = []
                for elem_i in range_constexpr(4):
                    u_bf16 = fx.BFloat16(u_prefetch[nr * 4 + elem_i])
                    u_f32_elems.append(u_bf16.to(fx.Float32))
                u_f32 = fx.Vector.from_elements(u_f32_elems, dtype=fx.Float32)
                vn_frags.append(u_f32 - bv_val)

            # No tail-chunk row mask on v_new. On the final chunk the BT rows
            # beyond T_local are padding, and w / u / k are all bounded to
            # T_local rows (see _seq_view), so those rows load a hardware zero
            # and v_new = u - w @ h is already exactly zero there -- nothing
            # reaches the k^T @ v_new state update. The mask that used to live
            # here existed because the descriptors spanned the whole tensor and
            # padding rows returned a neighbour's live (possibly non-finite)
            # data; a hardware zero cannot be Inf or NaN, so the argument for it
            # is gone with the descriptor change.

            # -- next iteration's w/u + gate prefetch (K5 build) --
            # Issued here, ahead of the v_new store, for the same reason the
            # fused build issues its copy ahead of the o store: the stores merge
            # into one cluster that the scheduler places before the second barrier.
            # Ahead of the store, the distance is a property of the source, and
            # the store plus GEMM2's MFMA chain both sit between the loads and their
            # consumption at the top of the next iteration.
            #
            # Safe to hoist: _stage_prefetch reads only HBM w/u/g for chunk
            # i_t+1 and depends on nothing this chunk computes. On the last
            # iteration it reads chunk NT, which is out of range and
            # address-clamped; the values are discarded.
            if const_expr(not COMPUTE_OUTPUT):
                next_prefetch = _stage_prefetch(i_t_i32 + fx.Int32(1))

            # -- 2b. Store v_new for output --
            if const_expr(SAVE_NEW_VALUE):
                for nr in range_constexpr(N_REPEAT_LOCAL):
                    vn_val = vn_frags[nr]
                    vn_col = i_v * fx.Int32(BV) + _nr_v(nr) + lane_n
                    for elem_i in range_constexpr(4):
                        bf16_v = _to_bf16(vn_val[elem_i])
                        vn_off = frag_row[elem_i] * fx.Int32(V) + vn_col
                        vn_buf[(vn_off,)] = bf16_v

            # -- 3. Gating --
            # K6 note: GEMM4b (the intra-chunk term) reuses the gated v_new because
            #   o_intra[i] = sum_j exp(g_i - g_j) (q k^T)[i,j] v_ungated[j]
            # with v_ungated[j] = v_gated[j] exp(g_j - g_last) telescopes to
            #   o_intra[i] = exp(g_i - g_last) sum_j (q k^T)[i,j] v_gated[j],
            # i.e. an ungated causal A' @ v_gated scaled by a per-query-row
            # factor. So GEMM4a needs no column gate and no ungated snapshot.
            if const_expr(USE_G):
                exp_g_last = _fast_exp(g_last_val)
                gate_elems = []
                for elem_i in range_constexpr(4):
                    # No row mask on the gate. v_new is already EXACTLY zero on
                    # padding rows (w and u are bounded to T_local, so both read
                    # a hardware zero and 0 - 0*h == 0), and the gate is finite
                    # there -- g_row reads a hardware zero and g is a cumsum of
                    # non-positive values, so exp(g_last - 0) = exp(g_last) is
                    # in (0, 1]. 0 * finite == 0, so the mask changes nothing.
                    gate_elems.append(_fast_exp(g_last_val - g_row_raw[elem_i]))
                gate_vec = fx.Vector.from_elements(gate_elems, dtype=fx.Float32)
                for nr in range_constexpr(N_REPEAT_LOCAL):
                    vn_frags[nr] = vn_frags[nr] * gate_vec
                # Whole-fragment gate: exp_g_last is uniform across V-tiles.
                exp_g_last_vec = fx.full(
                    N_REPEAT_LOCAL * 4, fx.Float32(exp_g_last), fx.Float32
                )
                for kb in range_constexpr(NUM_K_BLOCKS):
                    cur = fx.Vector(
                        h_accs_c[kb].load(), (N_REPEAT_LOCAL * 4,), fx.Float32
                    )
                    h_accs_c[kb].store(cur * exp_g_last_vec)

            if const_expr(USE_GK):
                for kb in range_constexpr(NUM_K_BLOCKS):
                    # exp() applied here, not at load time, so the prefetch has
                    # no arithmetic depending on the loads.
                    gk_q = fx.Vector(gk_quads[kb])
                    # gk_vec is per-kb, same across V-tiles: tile it to whole width.
                    gk_vec = fx.Vector.from_elements(
                        [
                            _fast_exp(gk_q[elem_i % 4])
                            for elem_i in range_constexpr(N_REPEAT_LOCAL * 4)
                        ],
                        dtype=fx.Float32,
                    )
                    cur = fx.Vector(
                        h_accs_c[kb].load(), (N_REPEAT_LOCAL * 4,), fx.Float32
                    )
                    h_accs_c[kb].store(cur * gk_vec)

            # -- 4. State update: h += k^T @ v_new_gated --
            # Store gated v_new transposed as [V, BT] so GEMM2 B-frag (run over
            # BT for fixed V) is contiguous. On the K5 path GEMM2 reads v_new^T
            # by hand from the double-XOR-folded layout, so the store must use
            # the SAME _grp_idx addressing (a single ds_write_b64 per nr: the 4
            # C-fragment BT rows are one bt-group). The fused path reads it back
            # over the single-XOR composed view, so it keeps the tiled-copy store.
            if const_expr(KT_TRANSPOSED):
                for nr in range_constexpr(N_REPEAT_LOCAL):
                    vnt_v = _nr_v(nr) + lane_n
                    vnt_g = wid_m * fx.Int32(4) + lane_m_base
                    fx.ptr_store(
                        _to_bf16(vn_frags[nr], 4),
                        lds_vnt_ptr + _lds_vnt_idx(vnt_v, vnt_g),
                    )
            else:
                _cfrag_to_lds(
                    cp_lds_x4,
                    tc_c_x4,
                    sVNT_C,
                    fx.Vector.from_elements(
                        [
                            vn_frags[nr][e]
                            for nr in range_constexpr(N_REPEAT_LOCAL)
                            for e in range_constexpr(4)
                        ],
                        dtype=fx.Float32,
                    ),
                    N_REPEAT_LOCAL * 4,
                )

            if const_expr(KT_TRANSPOSED):
                # Store k transposed as [K, BT], hand-addressed through _grp_idx
                # (the swizzle lowering cannot express the row>>3 fold this
                # transpose write needs -- see _grp_idx above). In-register
                # transpose: for each k-col, the 4 BT-consecutive rows this
                # thread loaded form one bt-group -> one ds_write_b64. No
                # cross-lane movement is needed; the 4 rows are already local.
                for s in range_constexpr(K_SLOTS_PER_THREAD):
                    quad_rows = k_prefetch[s]
                    row_quad = k_prefetch_lds_t[s]
                    for e in range_constexpr(K_VEC_WIDTH):
                        bt_grp = fx.Vector.from_elements(
                            [quad_rows[j][e] for j in range_constexpr(4)],
                            dtype=fx.BFloat16,
                        )
                        fx.ptr_store(
                            bt_grp,
                            lds_kt_ptr
                            + _lds_kt_idx(kx_col_base + fx.Int32(e), row_quad),
                        )
            else:
                _stage_r2s(k_prefetch, pD_k_lo, pD_k_hi)

            gpu.barrier()

            if const_expr(COMPUTE_OUTPUT):
                # Issue q's HBM loads before GEMM2 so their latency hides behind
                # GEMM2's MFMA chain. q is independent of GEMM2; only the lds_q
                # store must wait for GEMM1's lds_w readers.
                q_prefetch = _stage_g2r(buf_cp_g2r_128b_bf16, pS_q, stride_q, i_t_i32)

            # -- GEMM2: h += k^T @ v_new  (contraction over BT) --
            # Output h is [K, V], contraction BT.
            BT_TILES = BT // MFMA_K
            if const_expr(KT_TRANSPOSED):
                # K5 path: both operands live in the double-XOR-folded LDS
                # layout _grp_idx produces (lds_kt and lds_vnt are both stored
                # transposed and hand-addressed -- see the store sites above),
                # so their fragments are read by hand with matching addressing
                # and fed to a scalar MFMA. The composed-view read below cannot
                # reproduce the row>>3 fold and would read the wrong slots.
                #
                # A-frag k: m = K row = kb*64 + wid_m*16 + lane_n,
                #           contraction bt = bt_s*4 + lane_m_base (group).
                # B-frag v: n = V = nr*16 + lane_n, same bt group.
                # The operands are hand-loaded into MMA A/B fragments (same shapes
                # the fused path builds via make_fragment_A/B) and consumed by the
                # tiled-MMA atom, so the state update runs the standard
                # ``fx.gemm(mma_16x16x16_mnk, ...)`` -- only the LDS read stays
                # hand-addressed (the ptr_load) because the composed-view read
                # cannot reproduce the row>>3 fold.
                sKT = fx.make_view(
                    lds_kt_ptr,
                    fx.make_composed_layout(swz_kt, fx.make_layout((K, BT), (1, K))),
                )
                sVNT = fx.make_view(
                    lds_vnt_ptr,
                    fx.make_composed_layout(
                        fx.static(_single_xor_swz(BT, LDS_VNT_NG)),
                        fx.make_ordered_layout((BV, BT), (1, 0)),
                    ),
                )
                g2_fk = mma_16x16x16_mnk.make_fragment_A(
                    fx.slice(fx.zipped_divide(sKT, (64, BT)), (None, (0, 0)))
                )
                g2_fv = mma_16x16x16_mnk.make_fragment_B(sVNT)
                for kb in range_constexpr(NUM_K_BLOCKS):
                    for bt_s in range_constexpr(BT_TILES):
                        k_g = fx.Int32(bt_s * (MFMA_K // 4)) + lane_m_base
                        k_m = fx.Int32(kb * 64) + wid_m * fx.Int32(16) + lane_n
                        g2_fk[None, None, bt_s].store(
                            fx.Vector(
                                fx.ptr_load(
                                    lds_kt_ptr + _lds_kt_idx(k_m, k_g),
                                    result_type=v4bf16_type,
                                )
                            )
                        )
                        # One B tile carries all N_REPEAT_LOCAL v-fragments; gather
                        # them (nr-major, matching the fragment's N layout) and
                        # store the whole bt_s slice in one shot.
                        g2_fv[None, None, bt_s].store(
                            fx.Vector.from_elements(
                                [
                                    fx.Vector(
                                        fx.ptr_load(
                                            lds_vnt_ptr
                                            + _lds_vnt_idx(_nr_v(nr) + lane_n, k_g),
                                            result_type=v4bf16_type,
                                        )
                                    )[e]
                                    for nr in range_constexpr(N_REPEAT_LOCAL)
                                    for e in range_constexpr(4)
                                ],
                                dtype=fx.BFloat16,
                            )
                        )
                        fx.gemm(
                            mma_16x16x16_mnk,
                            h_accs_c[kb],
                            g2_fk[None, None, bt_s],
                            g2_fv[None, None, bt_s],
                            h_accs_c[kb],
                        )
            else:
                # Fused path: lds_kt is stored [BT, K] (no transpose) and read
                # over single-XOR composed views -- the plain fold is correct
                # here. B (lds_vnt) is a [BV, BT] view; A (kᵀ) is a per-64-K-block
                # [64, BT] view whose contraction (bt) is K apart, so it reads
                # through cp_lds_x1 (4 atom calls per fragment).
                sVNT = fx.make_view(
                    lds_vnt_ptr,
                    fx.make_composed_layout(
                        fx.static(_single_xor_swz(BT, LDS_VNT_NG)),
                        fx.make_ordered_layout((BV, BT), (1, 0)),
                    ),
                )
                g2_cp_b = fx.make_tiled_copy_B(cp_lds_x4, mma_16x16x16_mnk).get_slice(
                    tid
                )
                g2_pS_v = g2_cp_b.partition_S(sVNT)
                g2_fv = mma_16x16x16_mnk.make_fragment_B(sVNT)
                g2_fv_rt = g2_cp_b.retile(g2_fv)
                sKT = fx.make_view(
                    lds_kt_ptr,
                    fx.make_composed_layout(swz_kt, fx.make_layout((K, BT), (1, K))),
                )
                cp_g2_a = cp_lds_x1
                g2_cp_a = fx.make_tiled_copy_A(cp_g2_a, mma_16x16x16_mnk).get_slice(tid)
                g2_fk = mma_16x16x16_mnk.make_fragment_A(
                    fx.slice(fx.zipped_divide(sKT, (64, BT)), (None, (0, 0)))
                )
                g2_fk_rt = g2_cp_a.retile(g2_fk)
                for kb in range_constexpr(NUM_K_BLOCKS):
                    sKT_kb = fx.slice(fx.zipped_divide(sKT, (64, BT)), (None, (kb, 0)))
                    g2_pS_k = g2_cp_a.partition_S(sKT_kb)
                    for bt_s in range_constexpr(BT_TILES):
                        fx.copy(
                            cp_g2_a,
                            g2_pS_k[None, None, bt_s],
                            g2_fk_rt[None, None, bt_s],
                        )
                        fx.copy(
                            cp_lds_x4,
                            g2_pS_v[None, None, bt_s],
                            g2_fv_rt[None, None, bt_s],
                        )
                        fx.gemm(
                            mma_16x16x16_mnk,
                            h_accs_c[kb],
                            g2_fk[None, None, bt_s],
                            g2_fv[None, None, bt_s],
                            h_accs_c[kb],
                        )

            # =============================================================== #
            # K6 output stage. h[t] is still resident in lds_h (the snapshot,
            # NOT the GEMM2-updated h_accs); lds_kt holds k^T; lds_vnt holds
            # the gated v_new -- exactly the operands GEMM3/GEMM4 need.
            # =============================================================== #
            if const_expr(COMPUTE_OUTPUT):
                # -- Store prefetched q into lds_q (aliases lds_w, dead after
                #    GEMM1) --
                _stage_r2s(q_prefetch, pD_w_lo, pD_w_hi)

                gpu.barrier()

                # -- GEMM3: o = q @ h^T  (contraction over K) --
                sQ = sW_lo
                sH3 = fx.make_view(
                    lds_h_ptr,
                    fx.make_composed_layout(
                        fx.static(_single_xor_swz(K, LDS_H_NG)),
                        fx.make_ordered_layout((BV, K), (1, 0)),
                    ),
                )
                g3_cp_a = fx.make_tiled_copy_A(cp_lds_x4, mma_16x16x16_mnk).get_slice(
                    tid
                )
                g3_cp_b = fx.make_tiled_copy_B(cp_lds_x4, mma_16x16x16_mnk).get_slice(
                    tid
                )
                g3_pS_q = g3_cp_a.partition_S(sQ)
                g3_pS_h = g3_cp_b.partition_S(sH3)
                g3_fq = mma_16x16x16_mnk.make_fragment_A(sQ)
                g3_fh = mma_16x16x16_mnk.make_fragment_B(sH3)
                g3_fq_rt = g3_cp_a.retile(g3_fq)
                g3_fh_rt = g3_cp_b.retile(g3_fh)
                frag_o = fx.make_rmem_tensor(
                    fx.tiled_mma_partition_shape(
                        fx.MmaOperand.C, mma_16x16x16_mnk, (BT, BV)
                    ),
                    fx.Float32,
                )
                frag_o.fill(0.0)
                for kt in range_constexpr(K // MFMA_K):
                    fx.copy(
                        cp_lds_x4, g3_pS_q[None, None, kt], g3_fq_rt[None, None, kt]
                    )
                    fx.copy(
                        cp_lds_x4, g3_pS_h[None, None, kt], g3_fh_rt[None, None, kt]
                    )
                    fx.gemm(
                        mma_16x16x16_mnk,
                        frag_o,
                        g3_fq[None, None, kt],
                        g3_fh[None, None, kt],
                        frag_o,
                    )

                # When lds_A aliases lds_h, all waves must finish reading lds_h
                # (GEMM1 + GEMM3) before GEMM4a overwrites it as lds_A.
                if const_expr(ALIAS_A_ONTO_H):
                    gpu.barrier()

                # -- GEMM4a: A = q @ k^T  (contraction over K) --
                # A[i,j] = sum_k q[i,k]*k[j,k]: M = query row, N = key row,
                # contraction = K.
                #
                # B wants k as (n=bt, contraction=k), and in the fused build
                # that is exactly how lds_kt is stored.
                #
                # Wave split (NR_SPLIT>1): b_A is V-independent, so each wid_n
                # owns BT_STEPS_LOCAL of the BT_STEPS key-column tiles and writes
                # its slice into the shared lds_A -- no redundant compute. A
                # barrier below precedes GEMM4b's full read.
                g4_cp_a = fx.make_tiled_copy_A(cp_lds_x4, mma_16x16x16_mnk).get_slice(
                    tid
                )
                g4_cp_b = fx.make_tiled_copy_B(cp_lds_x4, mma_16x16x16_mnk).get_slice(
                    tid
                )
                g4_pS_q = g4_cp_a.partition_S(sQ)
                g4_pS_k = g4_cp_b.partition_S(sK_lo)
                g4_fq = mma_16x16x16_mnk.make_fragment_A(sQ)
                g4_fk = mma_16x16x16_mnk.make_fragment_B(sK_lo)
                g4_fq_rt = g4_cp_a.retile(g4_fq)
                g4_fk_rt = g4_cp_b.retile(g4_fk)
                frag_a = fx.make_rmem_tensor(
                    fx.tiled_mma_partition_shape(
                        fx.MmaOperand.C, mma_16x16x16_mnk, (BT, BT)
                    ),
                    fx.Float32,
                )
                frag_a.fill(0.0)
                for kt in range_constexpr(K_TILES):
                    fx.copy(
                        cp_lds_x4, g4_pS_q[None, None, kt], g4_fq_rt[None, None, kt]
                    )
                    fx.copy(
                        cp_lds_x4, g4_pS_k[None, None, kt], g4_fk_rt[None, None, kt]
                    )
                    fx.gemm(
                        mma_16x16x16_mnk,
                        frag_a,
                        g4_fq[None, None, kt],
                        g4_fk[None, None, kt],
                        frag_a,
                    )

                # Causal mask only (no per-column gate).
                # The mask is applied to the whole accumulator first so the store
                # can be one tiled copy.
                masked = []
                for nr in range_constexpr(BT_STEPS_LOCAL):
                    bt_col = _nr_v(nr) + lane_n
                    a_acc = fx.Vector(frag_a[None, None, nr].load())
                    for e in range_constexpr(4):
                        # Triangle only. The two in-bounds terms this used to
                        # AND in (query row < T_local, key col < T_local) are
                        # redundant now that q and k are bounded to T_local:
                        # a padding row or column reads a hardware zero, so
                        # A = q @ k^T is already exactly zero there and cannot
                        # carry the NaN/Inf the select was guarding against.
                        # Dropping them also collapses a 3-term lane-mask AND
                        # chain into one compare, which is why the saving shows
                        # up mostly in SALU.
                        causal = frag_row_local[e] >= bt_col
                        masked.append(causal.select(a_acc[e], fx.Float32(0.0)))
                # lds_A's 4 C values are BT apart so this is goes out through the scalar copy atom.
                _cfrag_to_lds(
                    cp_lds_x1,
                    tc_c_x1,
                    sA,
                    fx.Vector.from_elements(masked, dtype=fx.Float32),
                    BT_STEPS_LOCAL * 4,
                )

                gpu.barrier()

                # -- GEMM4b: o_intra = A' @ v_gated  (contraction over BT) --
                # A-frag: A'[m=query row, contraction=key BT] from lds_A
                # (ungated, causal-masked). B-frag: gated v_new[contraction=key
                # BT, n=V] from lds_vnt. The intra term accumulates separately so
                # it can take the per-query-row factor exp(g_i - g_last) at store
                # time while the inter term takes exp(g_i).
                sVN4 = fx.make_view(
                    lds_vnt_ptr,
                    fx.make_composed_layout(
                        fx.static(_single_xor_swz(BT, LDS_VNT_NG)),
                        fx.make_ordered_layout((BV, BT), (1, 0)),
                    ),
                )
                g4_cp_a = fx.make_tiled_copy_A(cp_lds_x4, mma_16x16x16_mnk).get_slice(
                    tid
                )
                g4_cp_b = fx.make_tiled_copy_B(cp_lds_x4, mma_16x16x16_mnk).get_slice(
                    tid
                )
                g4_pS_a = g4_cp_a.partition_S(sA)
                g4_pS_v = g4_cp_b.partition_S(sVN4)
                g4_fa = mma_16x16x16_mnk.make_fragment_A(sA)
                g4_fv = mma_16x16x16_mnk.make_fragment_B(sVN4)
                g4_fa_rt = g4_cp_a.retile(g4_fa)
                g4_fv_rt = g4_cp_b.retile(g4_fv)
                frag_oi = fx.make_rmem_tensor(
                    fx.tiled_mma_partition_shape(
                        fx.MmaOperand.C, mma_16x16x16_mnk, (BT, BV)
                    ),
                    fx.Float32,
                )
                frag_oi.fill(0.0)
                for bt_s in range_constexpr(BT_STEPS):
                    fx.copy(
                        cp_lds_x4, g4_pS_a[None, None, bt_s], g4_fa_rt[None, None, bt_s]
                    )
                    fx.copy(
                        cp_lds_x4, g4_pS_v[None, None, bt_s], g4_fv_rt[None, None, bt_s]
                    )
                    fx.gemm(
                        mma_16x16x16_mnk,
                        frag_oi,
                        g4_fa[None, None, bt_s],
                        g4_fv[None, None, bt_s],
                        frag_oi,
                    )
                # frag_oi consumed directly at the combine below (no bridge).

                # -- Combine inter + intra with their per-query-row gates --
                # USE_G:  o = scale * (exp(g_i)*o_inter + exp(g_i-g_last)*o_intra)
                # USE_GK: o = scale * (o_inter + o_intra) -- the K6 output is
                #         ungated on the gk path (the per-K decay is already
                #         folded into h/v_new, and v_gated == v_ungated there).
                # Consume the wide C-fragments (frag_o from GEMM3, frag_oi from
                # GEMM4b) directly per V-tile. o_out[nr] is a plain f32x4 vector per V-tile.
                o_out = [None] * N_REPEAT_LOCAL
                if const_expr(USE_G):
                    exp_gi = [_fast_exp(g_row_raw[e]) for e in range_constexpr(4)]
                    exp_gi_vec = fx.Vector.from_elements(exp_gi, dtype=fx.Float32)
                    exp_gi_gl = [
                        _fast_exp(g_row_raw[e] - g_last_val) for e in range_constexpr(4)
                    ]
                    exp_gi_gl_vec = fx.Vector.from_elements(exp_gi_gl, dtype=fx.Float32)
                    for nr in range_constexpr(N_REPEAT_LOCAL):
                        o_out[nr] = (
                            fx.Vector(frag_o[None, None, nr].load()) * exp_gi_vec
                            + fx.Vector(frag_oi[None, None, nr].load()) * exp_gi_gl_vec
                        )
                else:
                    for nr in range_constexpr(N_REPEAT_LOCAL):
                        o_out[nr] = fx.Vector(
                            frag_o[None, None, nr].load()
                        ) + fx.Vector(frag_oi[None, None, nr].load())

                # Issued before the o store, not after. Issuing them here makes the distance a
                # property of the source instead of the scheduler's pressure
                # heuristic; the o store below now covers the loads.
                next_prefetch = _stage_prefetch(i_t_i32 + fx.Int32(1))

                # -- Scale and store o -> HBM [T_flat, H, V] token-major --
                scale_vec = fx.Vector.filled(4, SCALE, fx.Float32)
                for nr in range_constexpr(N_REPEAT_LOCAL):
                    o_scaled = o_out[nr] * scale_vec
                    o_col = i_v * fx.Int32(BV) + _nr_v(nr) + lane_n
                    for elem_i in range_constexpr(4):
                        o_off = frag_row[elem_i] * stride_o + o_col
                        o_buf[(o_off,)] = _to_bf16(o_scaled[elem_i])

            yield next_prefetch

        # -- Epilogue: store final state --
        if const_expr(STORE_FINAL_STATE):
            gHT = fx.Tensor(
                fx.make_view(
                    fx.get_iter(fx.rocdl.make_buffer_tensor(ht_tensor, max_size=False)),
                    fx.make_layout((K, BV), (1, K)),
                )
            )
            for kb in range_constexpr(NUM_K_BLOCKS):
                pD_ht = tc_c_state.partition_D(
                    fx.slice(fx.zipped_divide(gHT, (64, BV)), (None, (kb, 0)))
                )
                f_ht = fx.make_fragment_like(pD_ht)
                acc_whole = fx.Vector(
                    h_accs_c[kb].load(), (N_REPEAT_LOCAL * 4,), fx.Float32
                )
                if const_expr(STATE_DTYPE_BF16):
                    f_ht.store(_to_bf16(acc_whole, N_REPEAT_LOCAL * 4))
                else:
                    f_ht.store(acc_whole)
                fx.copy(
                    cp_state,
                    f_ht,
                    pD_ht,
                    soffset=ht_base + i_v * fx.Int32(BV * K),
                )

    # -- Host launcher ------------------------------------------------------
    @flyc.jit
    def launch_gdn_h(
        k_tensor: fx.Tensor,
        u_tensor: fx.Tensor,
        w_tensor: fx.Tensor,
        v_new_tensor: fx.Tensor,
        g_tensor: fx.Tensor,
        gk_tensor: fx.Tensor,
        h_tensor: fx.Tensor,
        h0_tensor: fx.Tensor,
        ht_tensor: fx.Tensor,
        cu_seqlens_tensor: fx.Tensor,
        chunk_offsets_tensor: fx.Tensor,
        q_tensor: fx.Tensor,
        o_tensor: fx.Tensor,
        T_val: fx.Int32,
        T_flat: fx.Int32,
        N_val: fx.Int32,
        grid_v: fx.Int32,
        grid_nh: fx.Int32,
        stream: fx.Stream,
    ):
        launcher = gdn_h_kernel(
            k_tensor,
            u_tensor,
            w_tensor,
            v_new_tensor,
            g_tensor,
            gk_tensor,
            h_tensor,
            h0_tensor,
            ht_tensor,
            cu_seqlens_tensor,
            chunk_offsets_tensor,
            q_tensor,
            o_tensor,
            T_val,
            T_flat,
            N_val,
        )
        launcher.launch(
            grid=(grid_v, grid_nh, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_gdn_h


__all__ = [
    "compile_chunk_gated_delta_h_gfx942",
    "select_variant",
]
