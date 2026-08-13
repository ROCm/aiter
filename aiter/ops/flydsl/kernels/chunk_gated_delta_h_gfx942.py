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
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import as_ir_value, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import vector as _vector

from .tensor_shim import GTensor, _to_raw
from .k5_variants import _bv_of_variant, _legal_bv_candidates, _variant_tag

_LOG2E = math.log2(math.e)  # 1.4426950408889634


# -- gfx942 tuned variant table ------------------------------------------
# Data-driven K5 variant selection for gfx942, from the MI325X graph-mode sweep.
# The host wrapper's grid-fill heuristic never emits a wave-widened tag and mispicks
# BV for many shapes; this table encodes the best variant per selection signature and
# is consulted first for gfx942 (the wrapper falls back to the heuristic on a miss).
#
# Key = (gate, H, _n_bucket(N), is_varlen). Findings from the full sweep:
#   * gate x H x N together decide the tile. gate alone is not enough: scalar-g is
#     bv16 only at small H*N; by H>=8,N>=4 the larger tiles win (g,8,8->bv32;
#     g,16,4->bv32; g,16/32,>=8->bv64w8). KDA scales the same way -- the N=1 pick
#     grows with H (gk,12/24,1->bv16; gk,48,1->bv32; gk,96,1->bv64w8), and every KDA
#     N>=4 at H>=24 is bv64w8.
#   * T_local (=T_flat/N) is never discriminative: each (gate,H,N-bucket) group
#     picks the same variant at T_flat 8192 and 32768, so T is not in the key.
#   * N-buckets are {1, 2, 4:(3-4), 8:(>=5)}. The >=5 bucket is flat (measured
#     N=6/8/12/16 for g H16/H32 and gk H24 all agree). N=2 is its own bucket: it
#     halves the grid and drops the tile one regime below N=4 for the mid-H groups
#     (g H16 & gk H12: N2->bv16 vs N4->bv32; g H32 & gk H24: N2->bv32 vs N4->bv64w8);
#     flat at the extremes (small-H floor bv16, gk H48/H96 ceiling bv64w8). N=3 has
#     no data for the groups that matter and folds into bucket 4 (larger-tile side).
#   * Tie-break objective is MIN-MEAN loss across the shapes sharing a signature,
#     including the varlen skew/skew_last patterns the selector cannot see (only
#     cu_seqlens exists). Rationale: real prefill batches are equal/ragged/bimodal;
#     the "skew" pattern (one seq holds ~half the tokens) is an adversarial corner,
#     so favour the common case and let the rare skew batch degrade.


@dataclass(frozen=True)
class _K5TunedEntry:
    gate: str  # "g" (scalar) | "gk" (per-channel)
    H: int
    n_bucket: int  # output of _n_bucket(N): 1 | 2 | 4 | 8
    is_varlen: bool
    variant: str  # registered K5 variant tag, e.g. "bv64w8"


# is_varlen is (N > 1) at runtime -- the wrapper sets it from cu_seqlens, which the
# host builds for every N>1 batch regardless of length distribution. So N=1 shapes
# are is_varlen=False and all N>1 shapes are is_varlen=True (redundant with the
# n_bucket, but kept explicit to mirror the selection signature).
_K5_TUNED_ROWS_GFX942: tuple[_K5TunedEntry, ...] = (
    # KDA (per-channel gate). N=1 pick grows with H; N>=4 at H>=24 is bv64w8.
    # N=2 is one tile below N=4 for the mid-H groups (H12/H24), flat at the extremes.
    _K5TunedEntry("gk", 12, 1, False, "bv16"),
    _K5TunedEntry("gk", 12, 2, True, "bv16"),
    _K5TunedEntry("gk", 12, 4, True, "bv32"),
    _K5TunedEntry("gk", 12, 8, True, "bv64w8"),
    _K5TunedEntry("gk", 24, 1, False, "bv16"),
    _K5TunedEntry("gk", 24, 2, True, "bv32"),
    _K5TunedEntry("gk", 24, 4, True, "bv64w8"),
    _K5TunedEntry("gk", 24, 8, True, "bv64w8"),
    _K5TunedEntry("gk", 48, 1, False, "bv32"),
    _K5TunedEntry("gk", 48, 2, True, "bv64w8"),
    _K5TunedEntry("gk", 48, 4, True, "bv64w8"),
    _K5TunedEntry("gk", 48, 8, True, "bv64w8"),
    _K5TunedEntry("gk", 96, 1, False, "bv64w8"),
    _K5TunedEntry("gk", 96, 2, True, "bv64w8"),
    _K5TunedEntry("gk", 96, 4, True, "bv64w8"),
    _K5TunedEntry("gk", 96, 8, True, "bv64w8"),
    # GDN (scalar gate). bv16 only at small H*N; larger tiles win as H*N grows.
    # N=2 is one tile below N=4 for the mid-H groups (H16/H32), flat at small H.
    _K5TunedEntry("g", 4, 1, False, "bv16"),
    _K5TunedEntry("g", 4, 2, True, "bv16"),
    _K5TunedEntry("g", 4, 4, True, "bv16"),
    _K5TunedEntry("g", 4, 8, True, "bv16"),
    _K5TunedEntry("g", 8, 1, False, "bv16"),
    _K5TunedEntry("g", 8, 2, True, "bv16"),
    _K5TunedEntry("g", 8, 4, True, "bv16"),
    _K5TunedEntry("g", 8, 8, True, "bv32"), 
    _K5TunedEntry("g", 16, 1, False, "bv16"),
    _K5TunedEntry("g", 16, 2, True, "bv16"),  # N=2 halves grid -> bv16 (N4 is bv32)
    _K5TunedEntry("g", 16, 4, True, "bv32"),
    _K5TunedEntry("g", 16, 8, True, "bv64w8"), 
    _K5TunedEntry("g", 32, 1, False, "bv16"),
    _K5TunedEntry("g", 32, 2, True, "bv32"),  # N=2 halves grid -> bv32 (N4 is bv64w8)
    _K5TunedEntry("g", 32, 4, True, "bv64w8"),
    _K5TunedEntry("g", 32, 8, True, "bv64w8"),
)

_K5_TUNED_TABLE_GFX942: dict[tuple, str] = {
    (e.gate, e.H, e.n_bucket, e.is_varlen): e.variant for e in _K5_TUNED_ROWS_GFX942
}


def _n_bucket(N: int) -> int:
    """Normalize the batch/sequence count into the measured grid-size regimes.

    Four buckets: {1: N==1, 2: N==2, 4: 3<=N<=4, 8: N>=5}. N=2 is its own bucket
    because halving the grid drops the optimal tile one regime below N=4 for the
    mid-H groups (measured: g H16/gk H12 N2->bv16 vs N4->bv32; g H32/gk H24 N2->bv32
    vs N4->bv64w8). N=3 has no measurement for the groups that matter and folds into
    bucket 4 (the larger-tile side). The >=5 bucket was validated flat (N=6/8/12/16).
    """
    if N <= 1:
        return 1
    if N == 2:
        return 2
    if N <= 4:
        return 4
    return 8


def select_variant(
    *, gate: str, H: int, N: int, V: int, is_varlen: bool
) -> str | None:
    """gfx942 measured-best K5 variant tag for this shape, or None on a table miss.

    Returns a tag from ``K5_VARIANTS`` (e.g. ``"bv64w8"``) when the
    ``(gate, H, _n_bucket(N), is_varlen)`` signature is tabled and the tabled BV is
    legal for ``V``; otherwise None (the host wrapper then falls back to its
    cross-arch grid-fill heuristic). ``T_flat``/``Hg`` are intentionally not part of
    the key -- see the table note above.
    """
    tag = _K5_TUNED_TABLE_GFX942.get((gate, H, _n_bucket(N), is_varlen))
    if tag is None:
        return None
    if _bv_of_variant(tag) not in _legal_bv_candidates(V):
        return None
    return tag


def _make_fast_exp(g_is_log2_scaled: bool):
    """Return the ``exp`` helper (see gfx950 kernel for the rationale).

    ``rocdl.exp2`` requires a raw ``ir.Value``; a FlyDSL ``Float32`` wrapper (as
    produced by re-typing a loop-carried value) is not one, and the arithmetic
    below may hand back either. ``as_ir_value`` normalises both.
    """
    if g_is_log2_scaled:

        def _fast_exp(x):
            return rocdl.exp2(T.f32, as_ir_value(x))

    else:

        def _fast_exp(x):
            return rocdl.exp2(T.f32, as_ir_value(x * _LOG2E))

    return _fast_exp

def _to_bf16_fast(val, n=1):
    """f32 -> bf16 as ``(bitcast<u32>(x) + 0x8000) >> 16`` (round-half-away).

    ``n`` is the element count: 1 for a scalar ``Float32``, N for an f32xN
    ``Vector``. Returns a raw ``ir.Value`` (accepted by ``fx.ptr_store`` and by
    the ``GTensor`` store paths).

    Why not ``.truncf()`` / ``.to(BFloat16)``: those emit ``arith.truncf`` with
    no rounding-mode attribute, which MLIR defines as IEEE round-to-nearest-EVEN.
    gfx942 has no ``v_cvt_pk_bf16_f32`` (gfx950-only), so the backend expands RNE
    into ~6 VALU per element -- extract lsb, add the 0x7FFF bias, ``v_cmp_u_f32``
    NaN test, ``v_cndmask``, then pack with ``v_perm_b32``. At 64 conversions per
    chunk that was ~90 of the ~193 non-MFMA VALU instructions in the chunk loop,
    the single largest term. Asking for ``rounding_mode=toward_zero`` instead is
    NOT an option: it hard-aborts inside MLIR (uncatchable) on this path.

    Pure truncation (what the HIP reference does --
    ``bit_cast<uint32_t>(x) >> 16``, csrc/kernels/chunk_gated_delta_rule_fwd_h.cu:63)
    is cheaper still at ~2 VALU, but it was measured to be marginally too lossy
    here: its one-sided bias, accumulated over the serial chunk scan, put 13 of
    25.4M h elements past the 5e-2 tolerance (max_abs 0.104). Adding the 0x8000
    bias first costs one more add and restores <=0.5 ulp symmetric error, which
    matches RNE except on exact ties.

    Ties round away from zero rather than to even. Values are sign-magnitude, so
    the same bias works for both signs; the carry can only perturb the exponent
    within ~1 ulp of FLT_MAX, far outside the range this kernel produces.
    """
    is_vec = n > 1
    i32_ty = T.vec(n, T.i32) if is_vec else T.i32
    i16_ty = T.vec(n, T.i16) if is_vec else T.i16
    bf16_ty = T.vec(n, T.bf16) if is_vec else T.bf16

    def _splat(c):
        return as_ir_value(fx.full(n, c, fx.Int32) if is_vec else fx.Int32(c))

    bits = _arith.bitcast(i32_ty, as_ir_value(val))
    # The shift may be signed or unsigned: the following trunci keeps only the
    # low 16 bits, which are bits 16..31 of the input either way.
    hi = _arith.shrui(_arith.addi(bits, _splat(0x8000)), _splat(16))
    narrowed = _arith.trunci(i16_ty, hi)
    cast = _vector.bitcast if is_vec else _arith.bitcast
    return cast(bf16_ty, narrowed)


def _mfma_bf16_16x16x16(a_bf16x4, b_bf16x4, acc_f32x4):
    """Single ``mfma_f32_16x16x16bf16_1k`` (gfx942 bf16 K=16 MFMA).

    The MFMA fragment ABI:
    * A operand (16x16x16): lane holds bf16x4 with element e = A[m=lane_n, k=grp*4+e],
        where grp = lane_m_base (lane//16).
    * B operand: lane holds bf16x4 with element e = B[k=grp*4+e, n=lane_n].
    * C/D accumulator: lane holds f32x4 with element e = C[m=grp*4+e, n=lane_n].
    
    Operands are bitcast bf16x4 -> vec<4xi16> (the intrinsic's operand type).
    """
    a_i16 = _vector.bitcast(T.vec(4, T.i16), as_ir_value(a_bf16x4))
    b_i16 = _vector.bitcast(T.vec(4, T.i16), as_ir_value(b_bf16x4))
    return rocdl.mfma_f32_16x16x16bf16_1k(T.f32x4, [a_i16, b_i16, acc_f32x4, 0, 0, 0])

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
    NUM_K_BLOCKS = K // 64

    # -- Chiplet (XCD) remap --------------------------------------
    # The grid is (GRID_V, grid_nh) with grid_nh = N*H. For a fixed head (i_nh)
    # the GRID_V V-tiles all read the same k/w/g/gk slices (only v/v_new differ
    # by V-tile). Under the HW default (flat block `xy` runs on XCD `xy % 8`)
    # those V-tiles scatter across XCDs, so k/w/g get pulled into up to nXCD
    # separate private L2 caches. The chiplet re-map applies the HipKittens
    # inverse shuffle with chunk size = GRID_V, forcing each head's whole
    # V-tile run on a single XCD. Hence, k/w/g are fetched into one L2 and reused.
    #
    # GRID_V is a compile-time constant (V and BV are both build-time known), so
    # the whole remap collapses to integer math on the flat block id.
    GRID_V = (V + BV - 1) // BV
    NXCD = 8

    _fast_exp = _make_fast_exp(G_IS_LOG2_SCALED)

    WARP_SIZE = 64

    WMMA_N = 16
    WMMA_K = 16  # gfx942: K=16
    N_REPEAT = BV // WMMA_N

    # -- Wave decomposition (NR_SPLIT: the "wave widening" axis) --
    # A wave is identified by (wid_m, wid_n):
    #   wid_m in [0, M_WAVES)  -- the BT tile for GEMM1 / the K tile for GEMM2.
    #                             This is the ONLY axis the original kernel had,
    #                             and BT=64 pins it to 4.
    #   wid_n in [0, NR_SPLIT) -- a slice of the N_REPEAT (V) axis. Each wave
    #                             owns N_REPEAT_LOCAL of the N_REPEAT column
    #                             tiles instead of looping over all of them.
    #
    # Why: LDS (38-56 KiB) allows only one workgroup per CU on gfx942's 64 KiB,
    # so a CU holds NUM_WARPS waves and no more. With NUM_WARPS=4 that is 1
    # wave/SIMD, far too little to hide HBM latency in a kernel that is
    # memory-bound at ~33% of peak. Splitting the V axis across waves multiplies
    # resident waves by NR_SPLIT at zero extra HBM traffic and identical CU
    # coverage: LDS depends only on BV, and each wave's share of the h
    # accumulators (and hence its VGPR footprint) shrinks by the same factor.
    #
    # NR_SPLIT=1 reproduces the original 4-wave kernel exactly.
    M_WAVES = BT // 16
    assert N_REPEAT % NR_SPLIT == 0, (
        f"NR_SPLIT={NR_SPLIT} must divide N_REPEAT={N_REPEAT} (=BV/16); "
        f"BV={BV} supports NR_SPLIT in "
        f"{[s for s in (1, 2, 4, 8) if N_REPEAT % s == 0]}"
    )
    N_REPEAT_LOCAL = N_REPEAT // NR_SPLIT
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
        assert (BT // WMMA_K) % NR_SPLIT == 0, (
            f"NR_SPLIT={NR_SPLIT} must divide BT_STEPS={BT // WMMA_K} to split "
            f"b_A across the V-split waves"
        )
    BT_STEPS_LOCAL = (BT // WMMA_K) // NR_SPLIT

    # Per-wave accumulators: only this wave's slice of the N_REPEAT axis.
    NUM_H_ACCS = NUM_K_BLOCKS * N_REPEAT_LOCAL

    # -- Loop-carried gate/u prefetch --
    # g/gk/u for chunk i+1 depend on nothing produced by chunk i, so they can be
    # issued a full iteration ahead.
    #
    # The carried values are raw loads only -- exp()/in-bounds selects stay at
    # the use site, so the prefetch block has no arithmetic hanging off the
    # loads and nothing forces a wait in the issuing iteration.
    N_GATE_G = 5 if USE_G else 0  # g_last + 4 g_row
    # gk is loaded as ONE dwordx4 per 64-wide K block, not four scalar dwords:
    # the 4 elements are consecutive k (kb*64 + wid_m*16 + lane_m_base*4 + 0..3)
    # and -- unlike g_row -- carry no per-element clamp, so the quad is
    # contiguous and 16 B aligned by construction (kb*64, wid_m*16,
    # lane_m_base*4, i_h*K and (bos+last_idx)*H*K are all multiples of 4).
    # Measured: -6 buffer_load and -13% SQ_WAIT_ANY per dispatch on shape 10.
    N_GATE_GK = NUM_K_BLOCKS if USE_GK else 0  # entries, each an f32x4
    N_U = N_REPEAT_LOCAL * 4
    N_GU = N_GATE_G + N_GATE_GK + N_U

    # -- LDS layout --
    # All four buffers use the same GROUP-MAJOR + XOR scheme (see _grp_idx in the
    # kernel body): a logical [R, C] tile is stored as [R][C/4][4], and the group
    # index is XOR-swizzled by the row. 4 bf16 = 8 B = one MFMA fragment, so every
    # fragment access is a single conflict-free ds_read_b64/ds_write_b64 and no
    # buffer needs padding.
    assert BT % 4 == 0 and K % 4 == 0
    # The XOR is a bank bijection only if a row has >= 16 groups (one per lane of
    # an MFMA fragment); both K/4 and BT/4 are >= 16 for the supported shapes.
    assert K // 4 >= 16 and BT // 4 >= 16, "group-XOR needs >=16 groups per row"

    # lds_w: w tile [BT, K] (A-frag for GEMM1). Single stage.
    # The old row-major [BT, K] pitch was 256 B == 0 (mod 32 banks), so all 16
    # lanes of an A-frag hit the SAME bank -- a 16-way conflict on the highest
    # traffic read in the kernel. The XOR fixes exactly that.
    LDS_W_NG = K // 4
    LDS_W_ELEMS = BT * K

    # lds_k: k tile stored TRANSPOSED as [K, BT] so GEMM2's k A-frag (a run over BT
    # for fixed K) is one group.
    LDS_KT_NG = BT // 4
    LDS_KT_ELEMS = K * BT

    # lds_vn: v_new stored TRANSPOSED as [BV, BT] -> GEMM2 B-frag = run over BT
    # (contraction) at fixed V. Each CTA only handles a BV-wide V-slice, and every
    # vnt access uses v_local = nr*16 + lane_n in [0, BV) -- NOT the full V, so the
    # buffer is sized to BV rows (this is what lets BV=64 fit the 64 KiB budget).
    LDS_VNT_NG = BT // 4
    LDS_VNT_ELEMS = BV * BT

    # lds_h: h snapshot, logically [BV, K] (v_local, k) -- GEMM1's B-frag is a run
    # over K (the contraction) at fixed V, and the HBM snapshot wants K contiguous
    # too, so both consumers want the same K-major order.
    #
    # Layout is GROUP-MAJOR + XOR (see _grp_idx below): the row is split into
    # NG = K/4 groups of 4 bf16, and the group index is XOR-swizzled by the row.
    # 4 bf16 = 8 bytes = exactly one MFMA fragment, so every access is a single
    # aligned ds_read_b64/ds_write_b64 instead of 4 scalar ds_*_u16.
    #
    # Why this replaces the old ``[BV, K+4]`` padded layout: with a 132-element
    # pitch the row stride is 264 B = 66 dwords == 2 (mod 32 banks), so the 16
    # lanes of a fragment land on only the 16 EVEN banks -- 2-way conflicted no
    # matter how wide the access is. That is why widening the old layout to vec4
    # and adding an XOR on top of the pitch both measured as no-ops. Here the row
    # term is a multiple of 128 B, so the bank pair is decided purely by the
    # swizzled group index, and XOR-ing by the row makes it a bijection across the
    # 16 lanes -> all 32 banks hit exactly once. Padding is no longer needed.
    LDS_H_NG = K // 4
    LDS_H_ELEMS = BV * K  # BV rows of V per CTA, no padding

    # lds_A (K6 only): the [BT, BT] attention matrix, staged between GEMM4a and
    # GEMM4b because GEMM4a's accumulator layout is the TRANSPOSE of the A-operand
    # GEMM4b needs. Same group-major + XOR scheme as the others.
    LDS_A_NG = BT // 4
    LDS_A_ELEMS = BT * BT if COMPUTE_OUTPUT else 0

    # LDS budget with the K6 buffer. At BV=64 (K=128) the five buffers come to
    # exactly 64 KiB (lds_w 16 + lds_kt 16 + lds_vnt 8 + lds_h 16 + lds_A 8), so
    # aliasing is not needed there any more -- but the path is kept because
    # GEMM4a (the lds_A writer) runs strictly after GEMM3 (the last lds_h reader),
    # so lds_A may reuse lds_h's storage under a barrier whenever the total does
    # overflow. Aliasing needs lds_A (BT*BT) <= lds_h (BV*K), i.e. BV >= 32;
    # BV=16 never overflows, so it keeps the buffers distinct.
    _lds_total_kib = (
        LDS_W_ELEMS + LDS_KT_ELEMS + LDS_VNT_ELEMS + LDS_H_ELEMS + LDS_A_ELEMS
    ) * 2 / 1024
    ALIAS_A_ONTO_H = (
        COMPUTE_OUTPUT and _lds_total_kib > 64.0 and LDS_A_ELEMS <= LDS_H_ELEMS
    )
    assert not (_lds_total_kib > 64.0 and not ALIAS_A_ONTO_H), (
        f"LDS {_lds_total_kib:.0f} KiB > 64 KiB and lds_A ({LDS_A_ELEMS}) does "
        f"not fit lds_h ({LDS_H_ELEMS}) to alias (BV={BV}, K={K})"
    )

    # The LDS->HBM snapshot drain walks BV*LDS_H_NG groups BLOCK_THREADS at a
    # time, so the block must tile it. This holds for every legal NR_SPLIT
    # (BLOCK_THREADS = 256*NR_SPLIT <= 16*BV, and BV*LDS_H_NG = 32*BV), but
    # assert it rather than rely on the algebra.
    # The drain walks pairs of adjacent k-groups (one 16 B store per thread), so
    # the block must tile the pair count and a row must hold an even number of
    # groups.
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

    # Cooperative load parameters (bf16x8 = dwordx4)
    #
    # Two decompositions of the [BT, K] w tile:
    #
    #  * BATCHED (the historical one): thread -> (k-block, row-batch), giving
    #    each thread NUM_K_BLOCKS x NUM_LOAD_BATCHES_64 slots that pair 2 rows
    #    with 2 k-blocks. Kept verbatim wherever it is still valid: switching
    #    BV=64 to the linear mapping below measured 19% slower on shape 6
    #    (283 -> 340 us) at an essentially unchanged VGPR count (240 vs 242).
    #    The mapping decides which (row, grp) each thread writes to lds_w, and
    #    hence the bank pattern of the XOR swizzle that 3e tuned to the conflict
    #    floor -- so a "harmless" reindex silently reintroduces bank conflicts.
    #    Any change here must be re-checked against the swizzle, not just the
    #    coalescing.
    #  * LINEAR: slot s = i*BLOCK_THREADS + tid over the whole tile. Needed once
    #    BLOCK_THREADS > BT * THREADS_PER_ROW_64, where the batched form divides
    #    BT by a rows-per-batch larger than BT and silently yields 0 batches.
    #
    # Both give W_THREADS_PER_ROW-consecutive tids one contiguous row segment,
    # so global coalescing is equivalent; only the register live-ranges differ.
    LOAD_VEC_WIDTH = 8
    THREADS_PER_ROW_64 = 64 // LOAD_VEC_WIDTH  # 8
    ROWS_PER_BATCH_64 = BLOCK_THREADS // THREADS_PER_ROW_64
    W_BATCHED = ROWS_PER_BATCH_64 <= BT and BT % ROWS_PER_BATCH_64 == 0
    NUM_LOAD_BATCHES_64 = BT // ROWS_PER_BATCH_64 if W_BATCHED else 0

    W_THREADS_PER_ROW = K // LOAD_VEC_WIDTH  # 16 for K=128
    W_SLOTS = BT * W_THREADS_PER_ROW  # 1024 for BT=64, K=128
    assert W_SLOTS % BLOCK_THREADS == 0, (
        f"w tile ({W_SLOTS} vec{LOAD_VEC_WIDTH} slots) must tile "
        f"BLOCK_THREADS={BLOCK_THREADS}"
    )
    W_LOADS_PER_THREAD = (
        NUM_K_BLOCKS * NUM_LOAD_BATCHES_64 if W_BATCHED else W_SLOTS // BLOCK_THREADS
    )

    K_STEPS_PER_BLOCK = 64 // WMMA_K  # 4
    BT_STEPS = BT // WMMA_K  # 4

    # -- k store-transpose decomposition --
    # k arrives from HBM as runs along K, but lds_kt wants runs along BT, so the
    # store is a genuine transpose. With the default load mapping (1 row x 8 k-cols
    # per thread) the 8 elements land in 8 different lds_kt rows -> 8 scalar
    # ds_write_b16. Instead give each thread 4 BT-CONSECUTIVE rows at the same 8
    # k-cols: then for each k-col the 4 values are one bt-group, so an in-register
    # transpose turns 8 scalar writes into one packed ds_write_b64 each.
    # A "slot" is one (row-quad, k-col-group) pair; each slot is 4 vec8 loads.
    # The per-thread k vector width shrinks as the block widens, so that the
    # (row-quad, k-col-group) slots keep tiling the block exactly. The ROW QUAD
    # stays 4 regardless -- that is what makes the transposed store a packed
    # ds_write_b64 -- so widening the block costs load width, not LDS width.
    #   256 thr -> vec8 (dwordx4) | 512 thr -> vec4 (dwordx2) | 1024 thr -> vec2
    K_VEC_WIDTH = min(LOAD_VEC_WIDTH, max(2, (BT // 4) * K // BLOCK_THREADS))
    K_COL_GROUPS = K // K_VEC_WIDTH
    K_ROW_QUADS = BT // 4
    K_XPOSE_SLOTS = K_ROW_QUADS * K_COL_GROUPS
    # Only take this path when the slots tile the block exactly (true for K=128
    # and K=256); otherwise fall back to the scalar transpose store.
    K_PACKED_XPOSE = K_XPOSE_SLOTS % BLOCK_THREADS == 0
    K_SLOTS_PER_THREAD = K_XPOSE_SLOTS // BLOCK_THREADS if K_PACKED_XPOSE else 0
    K_ROW_QUAD_STRIDE = BLOCK_THREADS // K_COL_GROUPS if K_PACKED_XPOSE else 0

    # known_block_size is REQUIRED once BLOCK_THREADS > 256 (the AMDGPU default
    # max_flat_workgroup_size), but must NOT be declared at 256: it raises the
    # backend's per-wave register budget, and at BV=64 that lets the allocator
    # go from 152 to 242 VGPRs, which measured 19% slower on shape 6 (286 ->
    # 340 us). Leaving the default in place at 4 waves keeps the historical
    # codegen bit-for-bit.
    _kernel_deco_kwargs = (
        {} if BLOCK_THREADS == 256 else {"known_block_size": [BLOCK_THREADS, 1, 1]}
    )

    # One kernel serves both builds, so the parameter list is the union of what
    # each needs. Params the active config does not use are never dereferenced
    # (no GTensor view is created for them) and so cost nothing beyond a wider
    # kernarg segment -- the same treatment gk/h0/ht already get when disabled.
    # The host wrapper passes a dummy tensor for each unused slot.
    _kernel_name = (
        ("chunk_gdn_fwd_h_o_flydsl_vk" if COMPUTE_OUTPUT else "chunk_gdn_fwd_h_flydsl_vk")
        + f"_{_variant_tag(BV, NUM_WARPS)}"
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
            # HipKittens remap algorithm: Flatten the HW-assigned 2D block id the way 
            # the AMD dispatcher does (xy = x + gridDim.x * y), invert the round-robin
            # such that runs of CHUNK = GRID_V consecutive logical ids land on one XCD, 
            # then unflatten back to (i_v, i_nh). Head runs are already GRID_V-contiguous
            # in logical order, so CHUNK = GRID_V co-locates each head's V-tiles.
            #
            # Enabled by default on gfx942 (nXCD=8). 
            # The tail guard below makes any grid_nh better or equal to the round-robin 
            # baseline: every full nXCD*CHUNK cycle is cleanly co-located, and the remaining
            # tail (when grid_nh is not a multiple of nXCD) passes through unchanged, i.e., 
            # it gets the default mapping.
            grid_nh_rt = N_val * fx.Int32(H)
            grid_total = fx.Int32(GRID_V) * grid_nh_rt
            xy = fx.block_idx.x + fx.Int32(GRID_V) * fx.block_idx.y
            xcd = xy % fx.Int32(NXCD)
            # Tail guard: ids past the last full nXCD*CHUNK cycle pass through
            # unchanged (remapping them would collide / go out of range).
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

        # Wave split (see NR_SPLIT above). wid_m keeps the original role (BT tile
        # for GEMM1, K tile for GEMM2); wid_n selects this wave's slice of the
        # N_REPEAT (V) axis. At NR_SPLIT == 1 wid_n is identically 0 and wid_m is
        # wid, so the generated code is unchanged from the 4-wave kernel.
        if const_expr(NR_SPLIT == 1):
            wid_m = wid
        else:
            wid_m = wid % fx.Int32(M_WAVES)
        wid_n = wid // fx.Int32(M_WAVES)

        def _nr_v(nr_local):
            """V-offset (in elements) of this wave's local column tile ``nr_local``.

            Global tile index is ``wid_n * N_REPEAT_LOCAL + nr_local``; returns
            that times 16. Compile-time constant when NR_SPLIT == 1.
            """
            if const_expr(NR_SPLIT == 1):
                return fx.Int32(nr_local * 16)
            return wid_n * fx.Int32(N_REPEAT_LOCAL * 16) + fx.Int32(nr_local * 16)

        k_ = GTensor(k_tensor, dtype=T.bf16, shape=(-1,))
        u_ = GTensor(u_tensor, dtype=T.bf16, shape=(-1,))
        w_ = GTensor(w_tensor, dtype=T.bf16, shape=(-1,))
        g_ = GTensor(g_tensor, dtype=T.f32, shape=(-1,))
        if const_expr(USE_GK):
            gk_ = GTensor(gk_tensor, dtype=T.f32, shape=(-1,))

        if const_expr(STORE_H):
            h_ = GTensor(h_tensor, dtype=T.bf16, shape=(-1,))
        if const_expr(SAVE_NEW_VALUE):
            vn_ = GTensor(v_new_tensor, dtype=T.bf16, shape=(-1,))
        if const_expr(COMPUTE_OUTPUT):
            q_ = GTensor(q_tensor, dtype=T.bf16, shape=(-1,))
            o_ = GTensor(o_tensor, dtype=T.bf16, shape=(-1,))
        state_t = T.bf16 if STATE_DTYPE_BF16 else T.f32
        if const_expr(USE_INITIAL_STATE):
            h0_ = GTensor(h0_tensor, dtype=state_t, shape=(-1,))
        if const_expr(STORE_FINAL_STATE):
            ht_ = GTensor(ht_tensor, dtype=state_t, shape=(-1,))

        if const_expr(IS_VARLEN):
            cu_ = GTensor(cu_seqlens_tensor, dtype=T.i32, shape=(-1,))
            co_ = GTensor(chunk_offsets_tensor, dtype=T.i32, shape=(-1,))

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
            # q aliases w: w is dead after GEMM1, q is only loaded after GEMM2.
            lds_q_ptr = lds_w_ptr

        # -- Group-major + XOR LDS addressing --
        # A buffer of R rows x C columns is stored as [R][C/4][4]: each row is
        # NG = C/4 groups of 4 bf16 (8 B = one MFMA fragment), and the group index
        # is XOR-swizzled by the row so that the 16 lanes of a fragment (whose row
        # indices are 16 consecutive values) map to 16 distinct groups -- covering
        # all 32 banks exactly once. Returns the element index of the group base.
        def _grp_idx(row, grp, cols, ng):
            # The mask folds the row's bits 3+ down onto its low bits before the
            # XOR. Most sites vary the row's low bits across lanes (all four MFMA
            # fragment reads use row = ...*16 + lane_n), so a plain ``row & (ng-1)``
            # spreads them sufficiently. But the k store-transpose writes rows
            # ``(tid%16)*8 + e`` -- across its 16 lanes ``row & 15`` takes only two
            # distinct values, so a plain mask cannot spread it and that causes
            # 8-way bank multiplicity.
            # Folding in ``row >> 3`` keys the swizzle on the bits that site does
            # vary, and a sweep over all nine LDS sites puts every one at minimum 
            # bank conflict rate.
            #
            # Safe by construction: this only permutes group slots within a row, XOR
            # by a fixed per-row value is a bijection on the group index, and writes
            # and reads derive the mask from the same row. Each (row, grp) still
            # maps to a unique slot.
            mask = (row ^ (row >> fx.Int32(3))) & fx.Int32(ng - 1)
            return row * fx.Int32(cols) + ((grp ^ mask) * fx.Int32(4))

        # 4 bf16 = 8 B: one ds_read_b64 / ds_write_b64, and one MFMA A/B fragment.
        v4bf16_type = T.vec(4, T.bf16)

        def _lds_h_idx(v_local, k_grp):
            return _grp_idx(v_local, k_grp, K, LDS_H_NG)

        def _lds_w_idx(bt_row, k_grp):
            return _grp_idx(bt_row, k_grp, K, LDS_W_NG)

        def _lds_kt_idx(k_row, bt_grp):
            return _grp_idx(k_row, bt_grp, BT, LDS_KT_NG)

        def _lds_vnt_idx(v_local, bt_grp):
            return _grp_idx(v_local, bt_grp, BT, LDS_VNT_NG)

        def _lds_A_idx(bt_row, bt_grp):
            return _grp_idx(bt_row, bt_grp, BT, LDS_A_NG)

        # -- Cooperative load decomposition --
        # See W_BATCHED above: the batched form is the historical mapping and is
        # preserved wherever valid; the linear form only kicks in at 1024 threads.
        if const_expr(W_BATCHED):
            load_row_in_batch = tid // fx.Int32(THREADS_PER_ROW_64)
            load_col_base = (tid % fx.Int32(THREADS_PER_ROW_64)) * fx.Int32(
                LOAD_VEC_WIDTH
            )

            def _w_slot(i_load):
                kb, batch = divmod(i_load, NUM_LOAD_BATCHES_64)
                row = fx.Int32(batch * ROWS_PER_BATCH_64) + load_row_in_batch
                return row, fx.Int32(kb * 64) + load_col_base

        else:

            def _w_slot(i_load):
                s = fx.Int32(i_load * BLOCK_THREADS) + tid
                row = s // fx.Int32(W_THREADS_PER_ROW)
                col_grp = s % fx.Int32(W_THREADS_PER_ROW)
                return row, col_grp * fx.Int32(LOAD_VEC_WIDTH)

        # k uses its own mapping so the transpose store can be packed: thread ->
        # (row-quad, k-col-group). Consecutive tids walk k-col groups, so a full
        # K-row is covered by K_COL_GROUPS consecutive threads (contiguous HBM).
        if const_expr(K_PACKED_XPOSE):
            kx_col_base = (tid % fx.Int32(K_COL_GROUPS)) * fx.Int32(K_VEC_WIDTH)
            kx_row_quad = tid // fx.Int32(K_COL_GROUPS)

        # -- Prologue: compute bos, T_local, NT, boh --
        # boh (the chunk-offset base) only addresses the h snapshot, so it -- and
        # the chunk_offsets read that produces it -- is skipped when not STORE_H.
        if const_expr(IS_VARLEN):
            bos = cu_[fx.Int64(i_n)]
            eos = cu_[fx.Int64(i_n) + fx.Int64(1)]
            T_local = eos - bos
            NT = (T_local + fx.Int32(BT - 1)) // fx.Int32(BT)
            if const_expr(STORE_H):
                boh = co_[fx.Int64(i_n)]
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

        if const_expr(SAVE_NEW_VALUE):
            if const_expr(IS_VARLEN):
                vn_base = (i_h * T_flat + bos) * fx.Int32(V)
            else:
                vn_base = ((i_n * fx.Int32(H) + i_h) * T_flat) * fx.Int32(V)

        if const_expr(COMPUTE_OUTPUT):
            # o is token-major [B, T_flat, H, V] (matches the Triton K6 output).
            o_base = (bos * fx.Int32(H) + i_h) * fx.Int32(V)
            stride_o = fx.Int32(H * V)

        if const_expr(USE_INITIAL_STATE):
            h0_base = i_nh * fx.Int32(V * K)
        if const_expr(STORE_FINAL_STATE):
            ht_base = i_nh * fx.Int32(V * K)

        # -- MFMA lane mapping for 16x16 tiles --
        lane_n = lane % fx.Int32(16)
        lane_m_base = lane // fx.Int32(16)

        # -- Initialize h accumulators --
        acc_zero = fx.full(4, 0.0, fx.Float32)
        h_accs = []
        for _kb in range_constexpr(NUM_K_BLOCKS):
            for _nr in range_constexpr(N_REPEAT_LOCAL):
                h_accs.append(acc_zero)

        # -- Load initial state if provided --
        # h_accs[kb][nr] element e = h[v = i_v*BV + nr*16 + lane_n,
        #                              k = kb*64 + wid*16 + lane_m_base*4 + e]
        if const_expr(USE_INITIAL_STATE):
            for kb in range_constexpr(NUM_K_BLOCKS):
                for nr in range_constexpr(N_REPEAT_LOCAL):
                    h0_col = i_v * fx.Int32(BV) + _nr_v(nr) + lane_n
                    h0_row_base = (
                        fx.Int32(kb * 64)
                        + wid_m * fx.Int32(16)
                        + lane_m_base * fx.Int32(4)
                    )
                    h0_off_base = h0_base + h0_col * fx.Int32(K) + h0_row_base
                    loaded_vec = h0_.vec_load((fx.Int64(h0_off_base),), 4)
                    if const_expr(STATE_DTYPE_BF16):
                        loaded_vec = loaded_vec.extf(T.f32x4)
                    acc_idx = kb * N_REPEAT_LOCAL + nr
                    h_accs[acc_idx] = h_accs[acc_idx] + loaded_vec

        NUM_W_LOADS = W_LOADS_PER_THREAD

        def _load_gate_u(it_i32):
            """Issue chunk ``it_i32``'s g/gk/u loads; return them as a flat list.

            Pure loads -- no exp, no in-bounds select -- so nothing in the
            issuing iteration depends on the results. Order must match the
            N_GATE_G / N_GATE_GK / N_U unpacking in the loop body.

            Out-of-range rows on the tail chunk (and the whole speculative
            chunk NT) are address-clamped to row 0 exactly as the w prefetch
            already is; the values are masked at the use site, so loading
            garbage here is harmless.
            """
            out = []
            next_end = (it_i32 + fx.Int32(1)) * fx.Int32(BT)
            last_idx = (next_end < T_local).select(
                next_end, T_local
            ) - fx.Int32(1)
            row_base = (
                it_i32 * fx.Int32(BT)
                + wid_m * fx.Int32(16)
                + lane_m_base * fx.Int32(4)
            )
            if const_expr(USE_G):
                out.append(g_[fx.Int64(i_h * T_flat + (bos + last_idx))])
                for elem_i in range_constexpr(4):
                    abs_row = row_base + fx.Int32(elem_i)
                    safe_row = (abs_row < T_local).select(abs_row, fx.Int32(0))
                    out.append(g_[fx.Int64(i_h * T_flat + (bos + safe_row))])
            if const_expr(USE_GK):
                gk_chunk_base = (bos + last_idx) * fx.Int32(H * K) + i_h * fx.Int32(K)
                for kb in range_constexpr(NUM_K_BLOCKS):
                    quad = (
                        gk_chunk_base
                        + fx.Int32(kb * 64)
                        + wid_m * fx.Int32(16)
                        + lane_m_base * fx.Int32(4)
                    )
                    out.append(gk_.vec_load((fx.Int64(quad),), 4))
            for nr in range_constexpr(N_REPEAT_LOCAL):
                u_col = i_v * fx.Int32(BV) + _nr_v(nr) + lane_n
                for elem_i in range_constexpr(4):
                    abs_row = row_base + fx.Int32(elem_i)
                    safe_row = (abs_row < T_local).select(abs_row, fx.Int32(0))
                    u_off = v_base + safe_row * stride_v + u_col
                    out.append(u_.vec_load((fx.Int64(u_off),), 1))
            return out

        # -- Prologue: pre-load first chunk's w + gate/u data --
        i_t0_i32 = fx.Int32(0)
        w_prefetch_init = []
        for i_load in range_constexpr(W_LOADS_PER_THREAD):
            row, col = _w_slot(i_load)
            abs_row = i_t0_i32 * fx.Int32(BT) + row
            safe_row = (abs_row < T_local).select(abs_row, fx.Int32(0))
            g_off = w_base + safe_row * stride_w + col
            w_prefetch_init.append(w_.vec_load((fx.Int64(g_off),), LOAD_VEC_WIDTH))

        gu_prefetch_init = _load_gate_u(i_t0_i32)

        init_state = (
            [_to_raw(v) for v in h_accs]
            + [_to_raw(v) for v in w_prefetch_init]
            + [_to_raw(v) for v in gu_prefetch_init]
        )
        c_zero = fx.Int64(0)
        c_one = fx.Int64(1)
        nt_idx = fx.Int64(NT)

        for i_t, state in range(c_zero, nt_idx, c_one, init=init_state):
            h_accs_in = list(state[:NUM_H_ACCS])
            w_prefetch_all = list(state[NUM_H_ACCS : NUM_H_ACCS + NUM_W_LOADS])
            gu_prefetch_all = list(state[NUM_H_ACCS + NUM_W_LOADS :])
            i_t_i32 = fx.Int32(i_t)

            # -- w LDS write offsets (group-major [BT][K/4][4] + XOR) --
            # Each thread holds a bf16x8 run = two adjacent k-groups, whose
            # swizzled positions are NOT adjacent, so the single ds_write_b128
            # becomes two ds_write_b64. That is the price of making the far
            # hotter A-frag read (below) conflict-free.
            w_prefetch_lds_all = []
            for i_load in range_constexpr(W_LOADS_PER_THREAD):
                row, col = _w_slot(i_load)
                grp = col // fx.Int32(4)
                w_prefetch_lds_all.append(
                    (_lds_w_idx(row, grp), _lds_w_idx(row, grp + fx.Int32(1)))
                )

            # -- Store h snapshot to LDS (group-major [BV][K/4][4] + XOR) --
            # h_accs element e = h[v_local = nr*16 + lane_n, k = kb*64 + wid*16 +
            #                      lane_m_base*4 + e].  The four e's are one
            # k-group, so the whole f32x4 accumulator packs into a single bf16x4
            # (one ds_write_b64) instead of 4 scalar ds_write_b16.
            for kb in range_constexpr(NUM_K_BLOCKS):
                for nr in range_constexpr(N_REPEAT_LOCAL):
                    acc_idx = kb * N_REPEAT_LOCAL + nr
                    acc_val = h_accs_in[acc_idx]
                    lds_h_v = _nr_v(nr) + lane_n
                    lds_h_g = fx.Int32(kb * 16) + wid_m * fx.Int32(4) + lane_m_base
                    fx.ptr_store(
                        _to_bf16_fast(acc_val, 4),
                        lds_h_ptr + _lds_h_idx(lds_h_v, lds_h_g),
                    )

            # The drain below reads lds_h groups written by other threads; the
            # fused build has no drain, so it needs no barrier here (its lds_h
            # readers are GEMM1/GEMM3, both after the post-lds_w barrier).
            if const_expr(STORE_H):
                gpu.barrier()

                # -- LDS -> HBM h snapshot, one k-group (4 bf16) per thread. --
                # Consecutive tids walk consecutive k-groups at fixed v, so the
                # XOR term is constant across the wave (conflict-free 8 B/lane)
                # and the HBM side is both coalesced and vectorized (it was
                # scalar bf16). Each thread drains two adjacent k-groups as one
                # 16 B store. The groups are consecutive k and h is [v, k] with
                # k contiguous, so the pair is contiguous in HBM and 16 B
                # aligned (g0 is even, K and stride_h are multiples of 8
                # elements). That halves the store count and doubles the
                # transaction width: 1024 contiguous B per wave instruction
                # instead of 512.
                #
                # The LDS side stays two b64 reads. The XOR swizzle puts
                # adjacent groups at non-adjacent slots on purpose, so this is a
                # store-side widening only.
                VG_PAIRS = BV * (LDS_H_NG // 2)
                for vp_base in range_constexpr(0, VG_PAIRS, BLOCK_THREADS):
                    linear = fx.Int32(vp_base) + tid
                    pair = linear % fx.Int32(LDS_H_NG // 2)
                    v_loc = linear // fx.Int32(LDS_H_NG // 2)
                    g0 = pair * fx.Int32(2)
                    lo = fx.ptr_load(
                        lds_h_ptr + _lds_h_idx(v_loc, g0), result_type=v4bf16_type
                    )
                    hi = fx.ptr_load(
                        lds_h_ptr + _lds_h_idx(v_loc, g0 + fx.Int32(1)),
                        result_type=v4bf16_type,
                    )
                    bf16_pair = _vector.shuffle(
                        as_ir_value(lo), as_ir_value(hi), list(range(8))
                    )
                    v_global = i_v * fx.Int32(BV) + v_loc
                    h_off = (
                        h_base
                        + i_t_i32 * stride_h
                        + v_global * fx.Int32(K)
                        + g0 * fx.Int32(4)
                    )
                    h_.vec_store((fx.Int64(h_off),), bf16_pair, 8)

            # -- Store prefetched w to LDS (two b64 halves per bf16x8) --
            for i_wp in range_constexpr(NUM_W_LOADS):
                wvec = w_prefetch_all[i_wp]
                off_lo, off_hi = w_prefetch_lds_all[i_wp]
                lo = fx.Vector.from_elements(
                    [wvec[e] for e in range_constexpr(4)], dtype=fx.BFloat16
                )
                hi = fx.Vector.from_elements(
                    [wvec[4 + e] for e in range_constexpr(4)], dtype=fx.BFloat16
                )
                fx.ptr_store(lo, lds_w_ptr + off_lo)
                fx.ptr_store(hi, lds_w_ptr + off_hi)

            gpu.barrier()

            # -- k prefetch (issued now, stored transposed after GEMM1) --
            k_prefetch = []
            k_prefetch_lds_t = []  # transposed store offsets: lds_kt[k, bt]
            if const_expr(K_PACKED_XPOSE):
                # Each thread owns K_SLOTS_PER_THREAD slots; a slot is 4
                # BT-consecutive rows at one 8-wide k-col group.
                for s in range_constexpr(K_SLOTS_PER_THREAD):
                    row_quad = kx_row_quad + fx.Int32(s * K_ROW_QUAD_STRIDE)
                    quad_rows = []
                    for j in range_constexpr(4):
                        row = row_quad * fx.Int32(4) + fx.Int32(j)
                        abs_row = i_t_i32 * fx.Int32(BT) + row
                        safe_row = (abs_row < T_local).select(abs_row, fx.Int32(0))
                        k_off = k_base + safe_row * stride_k + kx_col_base
                        quad_rows.append(
                            k_.vec_load((fx.Int64(k_off),), K_VEC_WIDTH)
                        )
                    k_prefetch.append(quad_rows)
                    k_prefetch_lds_t.append(row_quad)
            else:
                for i_load in range_constexpr(W_LOADS_PER_THREAD):
                    row, col = _w_slot(i_load)
                    abs_row = i_t_i32 * fx.Int32(BT) + row
                    safe_row = (abs_row < T_local).select(abs_row, fx.Int32(0))
                    k_off = k_base + safe_row * stride_k + col
                    k_prefetch.append(
                        k_.vec_load((fx.Int64(k_off),), LOAD_VEC_WIDTH)
                    )
                    # this vec holds k[row, col + (0..7)]; store each element
                    # transposed to lds_kt[kcol, row].
                    k_prefetch_lds_t.append((row, col))

            # -- g / gk / u: unpack this chunk's values, prefetched last iter --
            # Values come back off the loop-carried state as bare IR values; the
            # f32 gates must be re-wrapped as Float32 before they can feed
            # rocdl.exp2 (which needs a Value, not a raw carried operand). u is
            # left alone -- it is bf16 and its use site already wraps it.
            gu_all = list(gu_prefetch_all)

            def _as_f32(v):
                return fx.Float32(as_ir_value(v))

            if const_expr(USE_G):
                g_last_val = _as_f32(gu_all[0])
                g_row_raw = [_as_f32(v) for v in gu_all[1:5]]
            if const_expr(USE_GK):
                # one f32x4 per 64-wide K block (see N_GATE_GK)
                gk_quads = gu_all[N_GATE_G : N_GATE_G + N_GATE_GK]
            u_prefetch = gu_all[N_GATE_G + N_GATE_GK :]

            # -- GEMM1: bv = w @ h  (contraction over K) --
            # A-frag (w): lane holds w[m=BT row, k]; plain read of lds_w.
            # B-frag (h): lane holds h[k, n=V]; read from lds_h[v, k] with the
            #   transposed access = 4 contiguous k for fixed v (since lds_h is
            #   [v, k] with k contiguous, a run over k IS contiguous).
            bv_accs = []
            for _nr in range_constexpr(N_REPEAT_LOCAL):
                bv_accs.append(fx.full(4, 0.0, fx.Float32))

            for kb in range_constexpr(NUM_K_BLOCKS):
                for ks in range_constexpr(K_STEPS_PER_BLOCK):
                    # w A-frag: 4 bf16 K-elems for this lane's BT row.
                    # A[m=BT row=wid*16+lane_n, k=kb*64+ks*16 + lane_m_base*4 + e]
                    w_row = wid_m * fx.Int32(16) + lane_n
                    w_g = fx.Int32(kb * 16 + ks * (WMMA_K // 4)) + lane_m_base
                    a_frag = fx.ptr_load(
                        lds_w_ptr + _lds_w_idx(w_row, w_g), result_type=v4bf16_type
                    )

                    for nr in range_constexpr(N_REPEAT_LOCAL):
                        # h B-frag: B[k=kb*64+ks*16 + lane_m_base*4 + e, n=V=nr*16+lane_n]
                        # The 4 k-elements are exactly one k-group, so the whole
                        # fragment is a single ds_read_b64.
                        h_v = _nr_v(nr) + lane_n
                        h_g = fx.Int32(kb * 16 + ks * (WMMA_K // 4)) + lane_m_base
                        b_frag = fx.ptr_load(
                            lds_h_ptr + _lds_h_idx(h_v, h_g), result_type=v4bf16_type
                        )
                        bv_accs[nr] = _mfma_bf16_16x16x16(a_frag, b_frag, bv_accs[nr])

            # -- v_new = u - bv --
            vn_frags = []
            for nr in range_constexpr(N_REPEAT_LOCAL):
                bv_val = bv_accs[nr]
                u_f32_elems = []
                for elem_i in range_constexpr(4):
                    u_bf16 = fx.BFloat16(u_prefetch[nr * 4 + elem_i])
                    u_f32_elems.append(u_bf16.to(fx.Float32))
                u_f32 = fx.Vector.from_elements(u_f32_elems, dtype=fx.Float32)
                vn_frags.append(u_f32 - bv_val)

            # -- Tail-chunk row mask --
            # On the final chunk, BT rows beyond T_local are padding whose w/u/k
            # loads were clamped to row 0 (garbage). They must be zeroed in v_new
            # before the k^T @ v_new state update, or ``final_state`` is corrupted.
            # The USE_G gate below already zeroes out-of-range rows, but the
            # USE_GK path does no v_new gating -- so mask here unconditionally so
            # both gate ranks are correct. Each lane's f32x4 spans 4 BT rows (one
            # per elem_i); the row is the same across all nr.
            row_mask_elems = []
            for elem_i in range_constexpr(4):
                bt_row = (
                    i_t_i32 * fx.Int32(BT)
                    + wid_m * fx.Int32(16)
                    + lane_m_base * fx.Int32(4)
                    + fx.Int32(elem_i)
                )
                in_bounds = bt_row < T_local
                row_mask_elems.append(
                    in_bounds.select(fx.Float32(1.0), fx.Float32(0.0))
                )
            row_mask_vec = fx.Vector.from_elements(row_mask_elems, dtype=fx.Float32)
            for nr in range_constexpr(N_REPEAT_LOCAL):
                vn_frags[nr] = vn_frags[nr] * row_mask_vec

            # -- 2b. Store v_new (pre-gating) for output --
            if const_expr(SAVE_NEW_VALUE):

                def _emit_vn_store(off, value):
                    vn_[fx.Int64(off)] = value

                for nr in range_constexpr(N_REPEAT_LOCAL):
                    vn_val = vn_frags[nr]
                    vn_col = i_v * fx.Int32(BV) + _nr_v(nr) + lane_n
                    for elem_i in range_constexpr(4):
                        vn_bt_row = (
                            i_t_i32 * fx.Int32(BT)
                            + wid_m * fx.Int32(16)
                            + lane_m_base * fx.Int32(4)
                            + fx.Int32(elem_i)
                        )
                        if (vn_bt_row < T_local).ir_value():
                            f32_v = vn_val[elem_i]
                            bf16_v = _to_bf16_fast(f32_v)
                            vn_off = vn_base + vn_bt_row * fx.Int32(V) + vn_col
                            _emit_vn_store(vn_off, bf16_v)

            # -- 3. Gating --
            # K6 note: GEMM4b (the intra-chunk term) reuses this GATED v_new
            # rather than keeping a second ungated copy, because
            #   o_intra[i] = sum_j exp(g_i - g_j) (q k^T)[i,j] v_ungated[j]
            # with v_ungated[j] = v_gated[j] exp(g_j - g_last) telescopes to
            #   o_intra[i] = exp(g_i - g_last) sum_j (q k^T)[i,j] v_gated[j],
            # i.e. an ungated causal A' @ v_gated scaled by a per-query-row
            # factor. So GEMM4a needs no column gate and no ungated snapshot.
            if const_expr(USE_G):
                exp_g_last = _fast_exp(g_last_val)
                gate_elems = []
                for elem_i in range_constexpr(4):
                    # in_bounds is recomputed here rather than carried: it is a
                    # couple of VALU ops and keeping it off the prefetch keeps
                    # the loop-carried set to raw loads only.
                    abs_row = (
                        i_t_i32 * fx.Int32(BT)
                        + wid_m * fx.Int32(16)
                        + lane_m_base * fx.Int32(4)
                        + fx.Int32(elem_i)
                    )
                    in_bounds = abs_row < T_local
                    gate = _fast_exp(g_last_val - g_row_raw[elem_i])
                    gate_elems.append(in_bounds.select(gate, fx.Float32(0.0)))
                gate_vec = fx.Vector.from_elements(gate_elems, dtype=fx.Float32)
                for nr in range_constexpr(N_REPEAT_LOCAL):
                    vn_frags[nr] = vn_frags[nr] * gate_vec
                exp_g_last_vec = fx.full(4, fx.Float32(exp_g_last), fx.Float32)
                for kb in range_constexpr(NUM_K_BLOCKS):
                    for nr in range_constexpr(N_REPEAT_LOCAL):
                        acc_idx = kb * N_REPEAT_LOCAL + nr
                        h_accs_in[acc_idx] = h_accs_in[acc_idx] * exp_g_last_vec

            if const_expr(USE_GK):
                for kb in range_constexpr(NUM_K_BLOCKS):
                    # exp() applied here, not at load time, so the prefetch has
                    # no arithmetic depending on the loads.
                    gk_q = fx.Vector(as_ir_value(gk_quads[kb]))
                    gk_vec = fx.Vector.from_elements(
                        [
                            _fast_exp(gk_q[elem_i])
                            for elem_i in range_constexpr(4)
                        ],
                        dtype=fx.Float32,
                    )
                    for nr in range_constexpr(N_REPEAT_LOCAL):
                        acc_idx = kb * N_REPEAT_LOCAL + nr
                        h_accs_in[acc_idx] = h_accs_in[acc_idx] * gk_vec

            # -- 4. State update: h += k^T @ v_new_gated --
            # Store gated v_new transposed as [V, BT] so GEMM2 B-frag (run over
            # BT for fixed V) is contiguous. v_new element e is at
            # BT row = wid*16 + lane_m_base*4 + e, V col = nr*16 + lane_n.
            # The 4 accumulator elements are 4 consecutive BT = one bt-group, so
            # the fragment packs into a single ds_write_b64.
            for nr in range_constexpr(N_REPEAT_LOCAL):
                vnt_v = _nr_v(nr) + lane_n
                vnt_g = wid_m * fx.Int32(4) + lane_m_base
                fx.ptr_store(
                    _to_bf16_fast(vn_frags[nr], 4),
                    lds_vnt_ptr + _lds_vnt_idx(vnt_v, vnt_g),
                )

            # Store k transposed as [K, BT].
            if const_expr(K_PACKED_XPOSE):
                # In-register transpose: for each k-col, gather the 4 BT-consecutive
                # rows this thread loaded into one bt-group -> one ds_write_b64.
                # No cross-lane movement is needed; the 4 rows are already local.
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
                # k_prefetch[i] holds k[row, kcol+(0..7)]; scatter each element to
                # lds_kt[kcol+e, row] (scalar b16 writes).
                for i_kp in range_constexpr(NUM_W_LOADS):
                    kvec = k_prefetch[i_kp]
                    row, kcol = k_prefetch_lds_t[i_kp]
                    row_g = row // fx.Int32(4)
                    row_e = row % fx.Int32(4)
                    for e in range_constexpr(LOAD_VEC_WIDTH):
                        kt_idx = _lds_kt_idx(kcol + fx.Int32(e), row_g) + row_e
                        fx.ptr_store(kvec[e], lds_kt_ptr + kt_idx)

            gpu.barrier()

            # -- next iteration's w + gate/u prefetch (batched) --
            # On the last iteration these read chunk NT, which is out of range
            # and address-clamped; the values are discarded.
            #
            # WHERE this is issued differs between the two builds and is a
            # deliberate scheduling choice in each -- do not unify:
            #   K5:    before GEMM2, so the whole MFMA chain sits between the
            #          loads and their consumption at the top of the next iter.
            #   fused: after the o store, because that slot is taken by the q
            #          load, whose latency hides behind GEMM2's MFMA chain
            #          (measured ~+2%); the K6 stage then covers these loads.
            def _issue_next_prefetch():
                next_i_t_i32 = i_t_i32 + fx.Int32(1)
                w_next = []
                for i_load in range_constexpr(W_LOADS_PER_THREAD):
                    row, col = _w_slot(i_load)
                    abs_row = next_i_t_i32 * fx.Int32(BT) + row
                    safe_row = (abs_row < T_local).select(abs_row, fx.Int32(0))
                    g_off = w_base + safe_row * stride_w + col
                    w_next.append(w_.vec_load((fx.Int64(g_off),), LOAD_VEC_WIDTH))
                return w_next, _load_gate_u(next_i_t_i32)

            if const_expr(not COMPUTE_OUTPUT):
                w_next_prefetch, gu_next_prefetch = _issue_next_prefetch()
            else:
                # Issue q's HBM loads before GEMM2 so their latency hides behind
                # GEMM2's MFMA chain. q is independent of GEMM2; only the lds_q
                # store must wait for GEMM1's lds_w readers (barrier below).
                q_prefetch = []
                for i_load in range_constexpr(W_LOADS_PER_THREAD):
                    row, col = _w_slot(i_load)
                    abs_row = i_t_i32 * fx.Int32(BT) + row
                    safe_row = (abs_row < T_local).select(abs_row, fx.Int32(0))
                    qoff = q_base + safe_row * stride_q + col
                    q_prefetch.append(q_.vec_load((fx.Int64(qoff),), LOAD_VEC_WIDTH))

            # -- GEMM2: h += k^T @ v_new  (contraction over BT) --
            # A-frag (k): lane holds k[m=V head dim? no] -> k is [BT, K]; we want
            #   k^T = [K, BT] as A so output is [K, V]. A[m=K, contraction=BT].
            #   lds_kt[k, bt]: read 4 contiguous BT for fixed k.
            #   A[m=K=wid*16+lane_n? ] -- see ABI: A[m=lane_n(+row grp), k=grp*4+e].
            #   Here MFMA "m" = K output row, "contraction" = BT.
            for kb in range_constexpr(NUM_K_BLOCKS):
                for bt_s in range_constexpr(BT_STEPS):
                    # A-frag k: m = K row = kb*64 + wid*16 + lane_n,
                    #           contraction bt = bt_s*16 + lane_m_base*4 + e
                    k_m = fx.Int32(kb * 64) + wid_m * fx.Int32(16) + lane_n
                    k_g = fx.Int32(bt_s * (WMMA_K // 4)) + lane_m_base
                    k_a_frag = fx.ptr_load(
                        lds_kt_ptr + _lds_kt_idx(k_m, k_g), result_type=v4bf16_type
                    )

                    for nr in range_constexpr(N_REPEAT_LOCAL):
                        # B-frag v_new: n = V = nr*16 + lane_n,
                        #               contraction bt = bt_s*16 + lane_m_base*4 + e
                        vn_v = _nr_v(nr) + lane_n
                        vn_g = fx.Int32(bt_s * (WMMA_K // 4)) + lane_m_base
                        vn_b_frag = fx.ptr_load(
                            lds_vnt_ptr + _lds_vnt_idx(vn_v, vn_g),
                            result_type=v4bf16_type,
                        )
                        acc_idx = kb * N_REPEAT_LOCAL + nr
                        h_accs_in[acc_idx] = _mfma_bf16_16x16x16(
                            k_a_frag, vn_b_frag, h_accs_in[acc_idx]
                        )

            # =============================================================== #
            # K6 output stage. h[t] is still resident in lds_h (the snapshot,
            # NOT the GEMM2-updated h_accs_in); lds_kt holds k^T; lds_vnt holds
            # the gated v_new -- exactly the operands GEMM3/GEMM4 need.
            # =============================================================== #
            if const_expr(COMPUTE_OUTPUT):
                # -- Store prefetched q into lds_q (aliases lds_w, dead after
                #    GEMM1) --
                for i_qp in range_constexpr(NUM_W_LOADS):
                    qvec = q_prefetch[i_qp]
                    row, col = _w_slot(i_qp)
                    grp = col // fx.Int32(4)
                    lo = fx.Vector.from_elements(
                        [qvec[e] for e in range_constexpr(4)], dtype=fx.BFloat16
                    )
                    hi = fx.Vector.from_elements(
                        [qvec[4 + e] for e in range_constexpr(4)], dtype=fx.BFloat16
                    )
                    fx.ptr_store(lo, lds_q_ptr + _lds_w_idx(row, grp))
                    fx.ptr_store(hi, lds_q_ptr + _lds_w_idx(row, grp + fx.Int32(1)))

                gpu.barrier()

                # -- GEMM3: o = q @ h^T  (contraction over K) --
                # Run FIRST so lds_h is fully consumed before GEMM4a writes
                # lds_A; that is what lets lds_A alias the now-dead lds_h. Same
                # fragment structure as GEMM1: A-frag = q, B-frag = h. Output
                # o_accs[nr][e] = o[m=wid_m*16+lane_m_base*4+e, n=nr*16+lane_n].
                o_accs = []
                for _nr in range_constexpr(N_REPEAT_LOCAL):
                    o_accs.append(fx.full(4, 0.0, fx.Float32))
                for kb in range_constexpr(NUM_K_BLOCKS):
                    for ks in range_constexpr(K_STEPS_PER_BLOCK):
                        q_row = wid_m * fx.Int32(16) + lane_n
                        q_g = fx.Int32(kb * 16 + ks * (WMMA_K // 4)) + lane_m_base
                        qa_frag = fx.ptr_load(
                            lds_q_ptr + _lds_w_idx(q_row, q_g),
                            result_type=v4bf16_type,
                        )
                        for nr in range_constexpr(N_REPEAT_LOCAL):
                            h_v = _nr_v(nr) + lane_n
                            h_g = fx.Int32(kb * 16 + ks * (WMMA_K // 4)) + lane_m_base
                            hb_frag = fx.ptr_load(
                                lds_h_ptr + _lds_h_idx(h_v, h_g),
                                result_type=v4bf16_type,
                            )
                            o_accs[nr] = _mfma_bf16_16x16x16(
                                qa_frag, hb_frag, o_accs[nr]
                            )

                # When lds_A aliases lds_h, all waves must finish reading lds_h
                # (GEMM1 + GEMM3) before GEMM4a overwrites it as lds_A.
                if const_expr(ALIAS_A_ONTO_H):
                    gpu.barrier()

                # -- GEMM4a: A = q @ k^T, fused per key-tile (compute ->
                #    causal-mask -> store to lds_A -> free), so at most one f32x4
                #    A tile is live. --
                # A[i,j] = sum_k q[i,k]*k[j,k]. MFMA m=i (query row), n=j (key
                # row), contraction=k. A-op = q[i,k] (lds_q); B-op element e =
                # k[j=lane_n, k=kb*64+ks*16+lane_m_base*4+e]. lds_kt is [K, BT]
                # (BT contiguous), so the 4 contraction elems are one row apart
                # -> 4 scalar reads.
                #
                # Wave split (NR_SPLIT>1): b_A is V-independent, so each wid_n
                # owns BT_STEPS_LOCAL of the BT_STEPS key-column tiles and writes
                # its slice into the shared lds_A -- no redundant compute. A
                # barrier below precedes GEMM4b's full read.
                def _kt_scalar(k_row, bt):
                    bt_g = bt // fx.Int32(4)
                    bt_e = bt % fx.Int32(4)
                    return fx.ptr_load(
                        lds_kt_ptr + _lds_kt_idx(k_row, bt_g) + bt_e,
                        result_type=T.bf16,
                    )

                for bt_l in range_constexpr(BT_STEPS_LOCAL):
                    a_acc = fx.full(4, 0.0, fx.Float32)
                    # Runtime key-tile index for this wave (compile-time when
                    # NR_SPLIT==1, since wid_n is then identically 0).
                    if const_expr(NR_SPLIT == 1):
                        bt_base = fx.Int32(bt_l * 16)
                    else:
                        bt_base = (
                            wid_n * fx.Int32(BT_STEPS_LOCAL * 16)
                            + fx.Int32(bt_l * 16)
                        )
                    bt_col = bt_base + lane_n  # n = key row j
                    for kb in range_constexpr(NUM_K_BLOCKS):
                        for ks in range_constexpr(K_STEPS_PER_BLOCK):
                            q_row = wid_m * fx.Int32(16) + lane_n
                            q_g = (
                                fx.Int32(kb * 16 + ks * (WMMA_K // 4)) + lane_m_base
                            )
                            qa_frag = fx.ptr_load(
                                lds_q_ptr + _lds_w_idx(q_row, q_g),
                                result_type=v4bf16_type,
                            )
                            kb_frag = fx.Vector.from_elements(
                                [
                                    _kt_scalar(
                                        fx.Int32(kb * 64 + ks * 16)
                                        + lane_m_base * fx.Int32(4)
                                        + fx.Int32(e),
                                        bt_col,
                                    )
                                    for e in range_constexpr(4)
                                ],
                                dtype=fx.BFloat16,
                            )
                            a_acc = _mfma_bf16_16x16x16(qa_frag, kb_frag, a_acc)

                    # Causal mask only (no per-column gate) -- see the gating
                    # note above for why the column decay cancels. acc element e
                    # is query row i = wid_m*16+lane_m_base*4+e; column j=bt_col.
                    col_abs = i_t_i32 * fx.Int32(BT) + bt_col
                    for e in range_constexpr(4):
                        row_tok = (
                            wid_m * fx.Int32(16)
                            + lane_m_base * fx.Int32(4)
                            + fx.Int32(e)
                        )
                        row_abs = i_t_i32 * fx.Int32(BT) + row_tok
                        causal = (
                            (row_tok >= bt_col)
                            & (row_abs < T_local)
                            & (col_abs < T_local)
                        )
                        a_val = a_acc[e] * causal.select(
                            fx.Float32(1.0), fx.Float32(0.0)
                        )
                        a_row = row_tok
                        a_g = bt_col // fx.Int32(4)
                        a_e = bt_col % fx.Int32(4)
                        fx.ptr_store(
                            _to_bf16_fast(a_val),
                            lds_A_ptr + _lds_A_idx(a_row, a_g) + a_e,
                        )

                gpu.barrier()

                # -- GEMM4b: o_intra = A' @ v_gated  (contraction over BT) --
                # A-frag: A'[m=query row, contraction=key BT] from lds_A
                # (ungated, causal-masked). B-frag: gated v_new[contraction=key
                # BT, n=V] from lds_vnt. The intra term accumulates separately so
                # it can take the per-query-row factor exp(g_i - g_last) at store
                # time while the inter term takes exp(g_i).
                o_intra_accs = []
                for _nr in range_constexpr(N_REPEAT_LOCAL):
                    o_intra_accs.append(fx.full(4, 0.0, fx.Float32))
                for bt_s in range_constexpr(BT_STEPS):
                    a_m = wid_m * fx.Int32(16) + lane_n
                    a_g = fx.Int32(bt_s * (WMMA_K // 4)) + lane_m_base
                    a_frag = fx.ptr_load(
                        lds_A_ptr + _lds_A_idx(a_m, a_g), result_type=v4bf16_type
                    )
                    for nr in range_constexpr(N_REPEAT_LOCAL):
                        vn_v = _nr_v(nr) + lane_n
                        vn_g = fx.Int32(bt_s * (WMMA_K // 4)) + lane_m_base
                        vn_b_frag = fx.ptr_load(
                            lds_vnt_ptr + _lds_vnt_idx(vn_v, vn_g),
                            result_type=v4bf16_type,
                        )
                        o_intra_accs[nr] = _mfma_bf16_16x16x16(
                            a_frag, vn_b_frag, o_intra_accs[nr]
                        )

                # -- Combine inter + intra with their per-query-row gates --
                # USE_G:  o = scale * (exp(g_i)*o_inter + exp(g_i-g_last)*o_intra)
                # USE_GK: o = scale * (o_inter + o_intra) -- the K6 output is
                #         ungated on the gk path (the per-K decay is already
                #         folded into h/v_new, and v_gated == v_ungated there).
                if const_expr(USE_G):
                    exp_gi = [_fast_exp(g_row_raw[e]) for e in range_constexpr(4)]
                    exp_gi_vec = fx.Vector.from_elements(exp_gi, dtype=fx.Float32)
                    exp_gi_gl = [
                        _fast_exp(g_row_raw[e] - g_last_val)
                        for e in range_constexpr(4)
                    ]
                    exp_gi_gl_vec = fx.Vector.from_elements(
                        exp_gi_gl, dtype=fx.Float32
                    )
                    for nr in range_constexpr(N_REPEAT_LOCAL):
                        o_accs[nr] = (
                            o_accs[nr] * exp_gi_vec
                            + o_intra_accs[nr] * exp_gi_gl_vec
                        )
                else:
                    for nr in range_constexpr(N_REPEAT_LOCAL):
                        o_accs[nr] = o_accs[nr] + o_intra_accs[nr]

                # -- Scale and store o -> HBM [T_flat, H, V] token-major --
                # The store must go through a helper: a bare ``o_[off] = ...``
                # inside ``if (...).ir_value():`` makes the scf.if try to yield
                # the GTensor (TypeError). Same pattern as the v_new store.
                def _emit_o_store(off, value):
                    o_[fx.Int64(off)] = value

                scale_vec = fx.full(4, fx.Float32(SCALE), fx.Float32)
                for nr in range_constexpr(N_REPEAT_LOCAL):
                    o_scaled = o_accs[nr] * scale_vec
                    o_col = i_v * fx.Int32(BV) + _nr_v(nr) + lane_n
                    for elem_i in range_constexpr(4):
                        o_bt_row = (
                            i_t_i32 * fx.Int32(BT)
                            + wid_m * fx.Int32(16)
                            + lane_m_base * fx.Int32(4)
                            + fx.Int32(elem_i)
                        )
                        if (o_bt_row < T_local).ir_value():
                            o_off = o_base + o_bt_row * stride_o + o_col
                            _emit_o_store(o_off, _to_bf16_fast(o_scaled[elem_i]))

                w_next_prefetch, gu_next_prefetch = _issue_next_prefetch()

            results = (
                yield [_to_raw(v) for v in h_accs_in]
                + [_to_raw(v) for v in w_next_prefetch]
                + [_to_raw(v) for v in gu_next_prefetch]
            )

        h_accs_final = list(results[:NUM_H_ACCS])

        # -- Epilogue: store final state --
        if const_expr(STORE_FINAL_STATE):
            for kb in range_constexpr(NUM_K_BLOCKS):
                for nr in range_constexpr(N_REPEAT_LOCAL):
                    acc_idx = kb * N_REPEAT_LOCAL + nr
                    acc_val = h_accs_final[acc_idx]
                    ht_col = i_v * fx.Int32(BV) + _nr_v(nr) + lane_n
                    ht_row_base = (
                        fx.Int32(kb * 64)
                        + wid_m * fx.Int32(16)
                        + lane_m_base * fx.Int32(4)
                    )
                    ht_off_base = ht_base + ht_col * fx.Int32(K) + ht_row_base
                    if const_expr(STATE_DTYPE_BF16):
                        out_vec = _to_bf16_fast(acc_val, 4)
                    else:
                        out_vec = acc_val
                    ht_.vec_store((fx.Int64(ht_off_base),), out_vec, 4)

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
