# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Kimi Delta Attention (KDA) chunkwise forward kernel for gfx950 / CDNA4.

KDA (Kimi Linear, arXiv:2510.26692) is a gated delta-rule linear attention with a
*channel-wise* forget gate.  Per head the recurrence is

    S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
    o_t = S_t^T q_t                                      S_t in R^{dk x dv}

which this kernel evaluates with the chunkwise-parallel formulation (Eq. 6-9 of the
paper).  For a chunk of C tokens with cumulative log-decay Gamma (gamma^r = prod_{i<=r}
alpha_i) the chunk body is

    T'    = StrictTril(Diag(beta) (Gamma . K)(K / Gamma)^T)      C x C
    A     = (I + T')^-1 Diag(beta)                               C x C, unit-triangular solve
    V~    = A (V - (Gamma . K) S)                                pseudo-values
    O     = (Gamma . Q) S + Tril((Gamma . Q)(K / Gamma)^T) V~
    S    <- Diag(gamma^C) S + (K . (gamma^C / Gamma))^T V~

Two algebraic points shape the implementation:

* The reference formulation computes ``W = A(Gamma.K)``, ``U = AV`` and then
  ``V~ = U - W S``.  Because ``A`` left-multiplies both, ``V~ = A(V - (Gamma.K)S)``,
  so ``W`` is never materialized -- one C x C x dk matmul and 16 KB of LDS saved.
* ``(K / Gamma)`` is never formed.  Dividing by the cumulative decay overflows
  (cumulative log-decay reaches -500 and below within a single chunk for strong
  gates).  Instead both C x C matrices are factored against a per-chunk *reference*
  row so that each factor's exponent is bounded by the decay across half a chunk,
  and the three Gamma-scaled tiles use only non-positive exponents.

  Only the lower triangle of those matrices is used, where the *product* of the two
  factors always has a non-positive exponent; individual factors, however, reach
  ``+|Gamma_C| / 2``.  The construction is therefore exact while

      max within-chunk cumulative |log decay| <= 2 * 126 / log2(e)  ~=  175 nats

  which is roughly 20x beyond any trained gate (a chunk that decays by e^-175 has
  already forgotten its entire state).  Past that point the ``EXP2_CLAMP`` keeps
  results finite but accuracy degrades smoothly.  Extending the range would mean
  building the C x C tiles in sub-chunk blocks with a per-block reference: for an
  off-diagonal block, a reference on the block boundary makes *both* factors
  non-positive, so only the diagonal blocks would bound the exponent, at
  ``|Gamma_C| / (2 * nblocks)``.

Layout / parallelization
------------------------
One workgroup owns (batch*head, v-channel-split) and walks the chunks of that head
sequentially, since the state carries a serial dependency.  The state is held
*transposed* (S^T, dv x dk) so that both matmuls that read it have the contracted
index innermost; it lives in an fp32 MFMA accumulator that is never rounded across
chunks, with a bf16 mirror in LDS for MFMA operand reads.

Every matmul is bf16 ``MFMA 16x16x32`` under one 2x2-wave TiledMma, with extents
M, N in {C, EV, DK} and K in {C, DK} -- so a single TiledMma drives the whole kernel.

Chunk length and value-channel split are tuning knobs, and the defaults are set by
*occupancy* rather than by matmul shape.  The paper uses C=64, but that needs 158 KB
of LDS, which admits only one workgroup per CU: with 4 waves on 4 SIMDs nothing hides
LDS or MFMA latency.  ``C=32`` with ``DV_SPLIT=4`` fits in ~71 KB, so two workgroups
per CU overlap, and it also quarters the O(C^2) sequential triangular solve.  Together
that is ~2.9x faster than C=64 on gfx950 despite doubling the chunk count.

Host-side tensor shapes (all contiguous, see :mod:`kda_interface` for the wrapper):
    q, k    [BH*NC, C*DK]                   bf16
    g       [BH*NC, C*DK]                   fp32, log decay (<= 0), NOT pre-summed
    beta    [BH*NC, C]                      fp32
    v, o    [BH*NC, C, DV_SPLIT, EV]        bf16 / out dtype
    h0t,htt [BH, DV_SPLIT, EV, DK]          fp32, transposed state (S^T)
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import BFloat16, Float32

LOG2E = 1.4426950408889634
# exp2 argument clamp.  |exp2(+-126)| stays inside fp32/bf16 range (both have 8
# exponent bits), so the factored C x C construction can never produce inf/nan.
EXP2_CLAMP = 126.0

# gfx950 / CDNA4 gives each CU 160 KB of LDS, so ``160*1024 // lds_bytes`` workgroups
# of this kernel are resident per CU.  Exported so the host-side dispatch rule in
# kda_interface can turn a candidate geometry into an occupancy without building it.
LDS_PER_CU = 160 * 1024

# LDS is handed out in 512 B granules, so that -- not the raw struct size -- is what
# divides into LDS_PER_CU (73 344 B of struct reports as 73 728 B of allocation).
LDS_GRANULE = 512

# Overlay the query tile on the cumulative-decay tile once it is dead (see N_ARENA in
# the builder).  Module-level so an A/B of the two layouts needs no rebuild of the
# caller; there is no reason to turn either off outside such a measurement.
#
# ALIAS_VN would fold V~ into the same arena for another EV*(C+8)*2 B, but it measures
# ~10 % SLOWER at EV=64 (0.1659 vs 0.1494 ms on `small`) and is never needed to reach
# an occupancy step, so it stays off.  The overlay is applied only where it lands the
# workgroup at >= 2 WG/CU (see ``fwd_lds_mode``): at EV=128 the footprint is over
# 100 KB either way, one workgroup per CU is one wave per SIMD, and with nothing
# co-resident to hide it the deferred query store costs ~2.3 %.
ALIAS_LDS = True
ALIAS_VN = False

# Keep the chunk's cumulative log-decay in registers instead of LDS.  The scan already
# gives thread ``tid`` a whole C/2-row slice of one channel, which is exactly the set of
# cells every later [C, DK] tile build asks it for, so with ``cache_g`` the [C, DK] fp32
# tile never has to exist: g is read straight from global into a register accumulator and
# the only shared traffic left is one float per channel.  Supersedes ALIAS_LDS when on --
# there is no g_cum left to overlay.
REG_SCAN = True

# Keep the chunk's queries in registers instead of LDS.  Under REG_SCAN a thread owns a
# whole C/2-row slice of one channel, and BOTH passes that read q ask for exactly that
# set of cells, so the [C, DK] bf16 query tile never has to exist: q is read straight
# from global into a register file (lanes of a wave walk consecutive channels, so each
# scalar load is still one coalesced burst).  Removes 8 KB of LDS, the eight
# ds_write_b128 of ``store_q`` and 32 ds_read_u16 per thread per chunk.
REG_Q = True

# ...but only up to this value-channel split.  Every split re-reads the WHOLE query
# tile, and the register form trades ``NQI`` wide dwordx4 loads for ``C/2`` scalar
# ushort loads, so the extra issue+latency cost scales with DV_SPLIT while the 8 KB
# LDS saving per workgroup does not.  Measured: DV_SPLIT=2 gains (large 1.227 -> 1.265,
# xlarge 1.915 -> 1.977, longT 1.935 -> 1.996) and DV_SPLIT=4 loses (mid 1.113 -> 0.921,
# small 1.117 -> 1.070, tiny 1.369 -> 1.313).
#
# RE-SWEPT on the integrated tree (EARLY_YT / MERGE_TILES / MASK_IN_REG + WY solve):
# the DV_SPLIT=4 loss above was measured at the OLD, larger LDS footprint and no
# longer holds -- raising the gate 2 -> 4 is now a win on exactly the three cases
# that dispatch to DS=4 (tiny 1.554 -> 1.562, small 1.272 -> 1.281, mid unchanged;
# 6-case geomean 1.5988 -> 1.6157 before the WY solve was stacked on top).
REG_Q_MAX_SPLIT = 4

# Same treatment for the key tile.  k is read by THREE passes rather than two, so the
# LDS traffic removed is larger, but the global-side cost is identical, so it carries
# the same split gate.
REG_K = True

# ...but with its OWN split gate, because the two do not break even at the same
# split: k's extra pass also costs an extra global read per split.
#
# HISTORY, because this constant has now inverted TWICE and the stale comment is
# the trap.  On the round-3 tree the sweep read (REG_Q, REG_K) = (2,2) 1.5988,
# (2,4) 1.5978, (4,4) 1.6153, (4,2) 1.6157, and after the WY solve was stacked
# (4,4) 1.7879 vs (4,2) 1.7922 -- so the key gate was pinned at 2.
#
# RE-SWEPT after the exp2/decay rework (EXP2_AFN + DECAY_SHARE + FMA_CONTRACT),
# which cut SQ_INSTS_VALU by 41.8%, and it FLIPPED BACK: (4,4) 2.1431 / 2.1475 /
# 2.1490 (median 2.1475) against (4,2) 2.1264 / 2.1238 / 2.1212 (median 2.1238),
# three runs each and the two distributions do not overlap.  The mechanism is
# visible per case: the three DS=4 cases the gate actually opens all gain
# (tiny 2.169 -> 2.217, small 1.810 -> 1.847, mid 1.693 -> 1.737) while the DS=2
# cases give back 0.3-1.4%.  Opening the key gate costs an extra global read per
# split, and that cost is unchanged -- what moved is the VALU it competes with,
# which is now 42% smaller, so the register form's issue cost no longer dominates.
# GENERALISABLE: this gate tracks the VALU/global balance, so re-sweep it after
# ANY change to the arithmetic in the tile builds, not just after an LDS change.
REG_K_MAX_SPLIT = 4

# Give (K.gamma^C/Gamma)^T its own LDS buffer instead of overlaying it on the K/Gamma
# tile, so it can be built in the SAME pass as K/Gamma and Gamma.K at the top of the
# chunk rather than in a serial pass at the very end.
#
# The occupancy argument for keeping the arena small no longer applies: the kernel is
# VGPR-limited, not LDS-limited.  Measured .vgpr_count is 214 (DV_SPLIT=4) / 240
# (DV_SPLIT=2) against 512 VGPRs per SIMD, so 512//vgpr == 2 waves/SIMD -- exactly the
# 2 WG/CU that 58 KB of LDS already buys.  Reaching 3 WG/CU would need lds <= 54272 B
# AND vgpr <= 168 simultaneously; shaving LDS alone buys nothing.  Conversely there are
# ~23 KB of LDS per workgroup that are *free* (81920 B is the 2 WG/CU ceiling), and
# spending DK*PC*2 of them here deletes two barriers from the chunk's fully exposed
# tail and folds one tile build into an existing loop.
EARLY_YT = True

# Second [C, DK] A-operand buffer, funded from the same free LDS as EARLY_YT.
#
# x_s is written four times per chunk -- (Gamma.K)/gamma^ref, Gamma.K, (Gamma.Q)/gamma^ref,
# Gamma.Q -- and each rewrite has to be fenced against the matmul that read the previous
# one, so the chunk is a strict build/barrier/gemm/barrier ladder.  The four tiles pair up
# naturally: the two K tiles share their k and decay operands, and so do the two Q tiles.
# With a second buffer each pair is built in ONE pass (one read of k / of q instead of two)
# and the two matmuls of the pair issue back to back with no barrier between them.
# Removes three barriers per chunk from the head and one redundant operand read.
MERGE_TILES = True

# Apply the two triangular masks (and Diag(beta)) inside the MFMA accumulator instead of
# in a separate LDS pass.  Both masks are chunk-invariant, so they are materialized once
# into C-fragment registers before the chunk loop; beta is read per chunk through a
# stride-0 [C, C] view of beta_s.  For Tril(Aqk) this also lets the bf16 result go
# straight from the accumulator to vAq, deleting an fp32 store, a barrier and an fp32
# read of the whole [C, C] tile.
MASK_IN_REG = True
# Drop the barrier that follows publish_state (see the call site for why it is dead).
DROP_PUBLISH_BARRIER = True
# ...and the same for T' (which saves no barrier, only LDS traffic, and costs two more
# accumulator-sized register files).  Split from MASK_IN_REG so the two can be A/B'd.
TPRIME_IN_REG = False

# ── the exp2 / decay stream ─────────────────────────────────────────────────
# MEASURED on the shipped ISA (gfx950, DV_SPLIT=2): the kernel emits 65
# ``v_exp_f32``, and each one is the head of a NINE-instruction sequence --
#
#   v_sub_f32                 (form the exponent)
#   v_minimum3_f32 / v_maximum3_f32   (the EXP2_CLAMP to [-126, 126])
#   v_cmp_gt_f32 / v_cndmask_b32 / v_add_f32      (denormal pre-scale)
#   v_exp_f32
#   v_cndmask_b32 / v_ldexp_f32                   (denormal post-scale)
#
# Four of those sequences run per row of the [C, DK] factored tile build, 16
# rows per thread per chunk = ~576 VALU instructions, against ~1120 VALU
# instructions per wave per chunk from the hardware counters.  The exp2 stream
# is therefore roughly HALF the VALU work, and VALU is the top unit
# (VALUBusy 45.0% vs MfmaUtil 5.5%).  Two independent levers attack it.

# 1) The six-instruction denormal scaffolding is PROVABLY DEAD.  LLVM emits it
# because llvm.exp2.f32 must return a denormal for arguments below -126, but
# every argument here has already been clamped to [-EXP2_CLAMP, +EXP2_CLAMP] =
# [-126, 126], so the result is always a normal fp32 and the pre/post-scale
# multiply by 2^k is always by 2^0.  The ``afn`` fast-math flag tells the
# backend it may lower straight to v_exp_f32.  This does NOT change
# EXP2_CLAMP and does not widen the representable range; it only deletes the
# branch that the clamp already makes unreachable.  9 instructions -> 3.
EXP2_AFN = True

# 2) Halve the number of exp2 calls.  The tile-build pass needs four
# exponentials of the same cumulative decay ``gc`` per cell:
#
#   exp2(gc - gref)   (Gamma.K)/gamma^ref   ->  vX
#   exp2(gref - gc)   K/Gamma               ->  vY
#   exp2(gc)          Gamma.K               ->  vX2   and cached in egr
#   exp2(gl - gc)     K.gamma^C/Gamma       ->  vYt
#
# Only the first is irreducible.  ``gref`` and ``gl`` are per-CHANNEL constants
# -- one value per thread per chunk, not per row -- so with
#     gref_e = exp2(gref),  dref = exp2(gl - gref)
# hoisted out of the row loop, the other three follow by strength reduction:
#
#   exp2(gref - gc) = exp2(-clamp(gc - gref))     : reuse the clamped exponent,
#                                                   negate it, one more v_exp.
#                                                   BIT-EXACT (the clamp is
#                                                   symmetric about 0).
#   exp2(gc)        = exp2(gc - gref) * gref_e    : one v_mul, no exp2.
#   exp2(gl - gc)   = exp2(gref - gc) * dref      : one v_mul, no exp2, and its
#                                                   operand kv*exp2(gref-gc) is
#                                                   already formed for vY.
#
# 4 exp2 + 4 clamps + 3 subtracts per row become 2 exp2 + 1 clamp + 1 subtract
# + 1 negate + 2 multiplies.  The two derived-by-multiply values differ from a
# direct clamped exp2 only in the saturated regime (|exponent| > 126, i.e. past
# the ~175-nat within-chunk decay the module docstring already documents as
# "accuracy degrades smoothly"), where the product underflows toward the true
# zero instead of resting on the clamp floor.
DECAY_SHARE = True

# 3) Contract the WY solve's multiply-accumulates into real FMAs.
#
# MEASURED on the shipped ISA: the kernel emits ZERO v_fma / v_fmac / v_pk_fma.
# Every ``acc = acc + a * b`` in the explicit-inverse solve becomes a separate
# v_mul_f32 and v_add_f32, because MLIR emits arith.mulf / arith.addf with no
# ``contract`` fast-math flag and LLVM will not fuse without it.  The solve does
# 76 such MACs per thread per chunk (28 in the 8x8 register inversion, 16 in the
# 8->16 block recursion, 32 in the 16->32 one) = 152 of the 689 VALU
# instructions inside the chunk loop.
#
# Emitting ``math.fma`` directly halves that, and -- more importantly -- halves
# the LATENCY of the 28-deep dependent chain in the 8x8 inversion, which is the
# longest serial run left in the chunk and is what binds the small cases.  An
# FMA rounds once instead of twice, so this is strictly MORE accurate than the
# code it replaces; the reason to keep it behind a flag is A/B-ability, not
# numerical doubt.
FMA_CONTRACT = True

# 3b) Fold the (Gamma.Q)/gamma^ref tile build INTO the WY solve's starved barrier
# region.  The 8x8 diagonal-block inversion occupies 32 of 256 threads with a deep
# dependent chain while 224 threads sit at the next barrier doing nothing; the
# GQ tile build is fully independent of the solve (it reads vG/vQ and writes
# vX/vX2, none of which the solve touches -- the solve owns vM/vZ/vVn/z_buf) and
# vX has been dead since the merged (Gamma.K)S matmul, several barriers earlier.
# Executing it in the same barrier region turns sum(inversion, tile) into
# max(inversion, tile).  GQ_IN_SOLVE_TAG
def _tree_sum(terms, fx):
    """Balanced pairwise sum of a python list of fx scalars (host-side unroll)."""
    if not terms:
        return fx.Float32(0.0)
    while len(terms) > 1:
        nxt = [terms[i] + terms[i + 1] for i in range(0, len(terms) - 1, 2)]
        if len(terms) % 2:
            nxt.append(terms[-1])
        terms = nxt
    return terms[0]


GQ_IN_SOLVE = True
# How many of the tile's rows-per-thread stay in the first (8x8-inversion) barrier
# region; the remainder move to the 8->16 region.  16 = all of it there.
GQ_ROW_CUT = 16
QK_MFMA_EARLY = True
# At DV_SPLIT=2 (EV=64) the accumulator fragment held across the solve costs more
# than the overlap buys; measured DS=4-only is the better setting.
QK_EARLY_MIN_SPLIT = 4

# 3c) Re-associate the 8x8 register inversion so its dependent chain is 7 FMAs
# instead of 28.  Forward substitution computes
#     x[i] = -( sum_{j<i-1} M[i,j] x[j]  +  M[i,i-1] x[i-1] )
# and the shipped left-to-right accumulation makes ALL i(i-1)/2 multiply-adds one
# serial chain (28 deep at WYB=8).  Every term with j < i-1 is ready strictly
# before x[i-1] is, so summing those in a balanced tree and folding x[i-1] in with
# a single trailing FMA leaves a critical path of exactly one FMA per row -- 7 for
# the whole block -- while the tree work fills the same idle issue slots.  The FLOP
# count is unchanged; only the summation order moves, which is a rounding-order
# change of the same magnitude the solve already carries.  WY_TREE_TAG
WY_TREE = False  # MEASURED: -0.6% (2.3164 vs 2.3300).  Once GQ_IN_SOLVE fills the
# region with independent work the 28-deep chain is fully hidden, so buying depth
# with extra live values is a net loss.  Kept for the record; do not re-derive.

# Hold the persistent state accumulator in the TRANSPOSED MMA orientation, S [DK, EV],
# instead of S^T [EV, DK].
#
# publish_state() writes the whole EV x DK bf16 state mirror out of an MMA C-fragment
# every chunk.  With M = EV the fragment's four values per lane are four consecutive EV
# rows of a DK-contiguous tile, i.e. four addresses PDK*2 bytes apart, so the store has
# to be four single-element ``UniversalCopy(16)`` ds_write_b16 -- 32 of them per thread
# per chunk at EV=64 -- and the four quarter-waves land on only two disjoint 8-bank
# groups (2*PDK == 272 B == 16 banks mod 32), a 4-way bank conflict.
#
# The same product is available with the operands swapped: S = (K.gamma^C/Gamma)^T V~,
# i.e. gemm_lds(vYt, vVn, .) instead of gemm_lds(vVn, vYt, .).  Both operands already
# exist in LDS, the MFMA count is identical, and the accumulator comes out as [DK, EV]
# -- so now the four values per lane are four CONSECUTIVE DK channels, which in the same
# DK-contiguous st_b tile are 8 contiguous bytes: ONE ds_write_b64.  4x fewer LDS store
# instructions, and the quarter-waves spread over all 32 banks (lane stride 2 banks,
# 16 disjoint 2-bank slots), which is the hardware floor for a 512 B wave store.
#
# Everything else that touches the fragment just needs the transposed view of the same
# memory: the bf16 mirror (st_b, strides (1, PDK)), the per-channel chunk decay
# broadcast (dec_s, strides (1, 0)), the initial-state staging tile (strides (1, DK))
# and the final fp32 state store (a [BH, DV_SPLIT, DK, EV] view of the same contiguous
# [BH, DV_SPLIT, EV, DK] buffer).  No LDS is added or removed.
ST_TRANSPOSE = True

# vStb is written exactly once per chunk -- by publish_state(), the LAST thing the
# chunk does -- and read twice: by the (Gamma.K) S matmul in the chunk head and by
# the (Gamma.Q) S matmul in the tail.  Both reads want the *same* B fragment, and the
# value is already final the moment the chunk starts.  SHARE_STATE_B fetches it once,
# right after the chunk's first barrier, and reuses the registers for both matmuls:
# the fetch moves off the state recurrence and the second fetch disappears.
SHARE_STATE_B = True

# ...enabled only while the hoisted B fragment is at most this many bf16 elements per
# lane.  EV*DK/128 = 32 at EV=32 (16 VGPRs) and 64 at EV=64 (32 VGPRs, which crosses
# the 2-waves/SIMD register ceiling and costs ~39% on the deep grids).
SHARE_STATE_B_MAX_FRAG = 32

# MEASURED NEGATIVE, recorded so it is not re-tried: the EARLY_YT store into yt_s is a
# transposed single-element write whose 64 lanes hit only 8 banks (row stride
# PC*2 = 80 B = 20 dwords, and 20*gcol % 32 has period 8), i.e. an 8-way bank conflict
# 16 times per chunk -- but under the register scan a thread's 16 rows are CONTIGUOUS in
# yt_s and the AMDGPU backend ALREADY merges those 16 ds_write_b16 into 2 ds_write_b128.
# Hand-vectorizing it produced a byte-identical instruction mix (SQ_INSTS_LDS 16,900,096
# before and after) and no measurable time change.  Only STRIDED transposed stores (see
# VN_VEC) are worth vectorizing by hand.

# The same treatment for the WY solve's rhs^T -> vVn transpose (see the call site).
# Unlike the yt_s case above this one is NOT already done by the backend, because
# tile_iter gives each thread a STRIDED set of rows, so the ds_write_b16 cannot merge.
VN_VEC = True
# Elements per vector store.  MEASURED: at EV=64 a full 8-element (b128) flush costs
# 16 VGPRs (.vgpr_count 244 -> 260, read off the final ISA) and crosses the 256 ceiling
# for 2 waves/SIMD -- xlarge 2.43x -> 1.53x, longT 2.46x -> 1.57x.  Four elements (b64)
# keeps the register budget; note the backend re-merges the two b64 into one b128
# anyway, so the ISA is identical and only the live range shrinks.
VN_VEC_WIDTH = 4
# Fuse the rhs = Diag(beta)(V - Z) elementwise pass into that transposed store.  They
# share the tile_iter(C, EV) mapping and every cell is produced and consumed by the same
# thread, so fusing deletes an fp32 [C, EV] LDS write, the matching read and a barrier.
VN_FUSE_RHS = True


# ── per-buffer LDS row padding ──────────────────────────────────────────────
# Historically every bf16 MFMA tile shared one "+8" row pad and every fp32
# C-fragment tile one "+4".  Neither number was ever swept per buffer, and the
# buffers have very different access patterns (b128 A/B reads vs 4-value-per-lane
# C-fragment stores), so they are split out here.  Defaults reproduce the old
# geometry exactly.  Each is overridable from the environment so a sweep needs no
# source edit; ``_fwd_pads`` is the SINGLE source of truth shared by the host-side
# occupancy estimator and the kernel body (they used to be two copies that had to
# be kept in sync by hand).
def _padenv(name, default):
    import os
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


PAD_STB = _padenv("KDA_PAD_STB", 8)   # bf16 st_b   [EV, DK] row pad (C-fragment store)
PAD_X = _padenv("KDA_PAD_X", 16)       # bf16 x_s / x2_s [C, DK] row pad (MFMA A)
PAD_Y = _padenv("KDA_PAD_Y", 16)       # bf16 y_s    [C, DK] row pad (MFMA B)
PAD_YT = _padenv("KDA_PAD_YT", 8)     # bf16 yt_s   [DK, C] row pad
PAD_VN = _padenv("KDA_PAD_VN", 8)     # bf16 vn_s   [EV, C] row pad
PAD_AQ = _padenv("KDA_PAD_AQ", 16)     # bf16 vAq    [C, C]  row pad (overlays z_buf)
PAD_M = _padenv("KDA_PAD_M", 4)       # fp32 m_mat  [C, C]  row pad
PAD_Z = _padenv("KDA_PAD_Z", 4)       # fp32 z_buf  [C, EV] row pad


def _fwd_pads(DK, C, EV, pad_f32):
    """Row strides for every padded LDS tile.  Single source of truth."""
    return dict(
        PSB=DK + PAD_STB,
        PXK=DK + PAD_X,
        PYK=DK + PAD_Y,
        PYT=C + PAD_YT,
        PVN=C + PAD_VN,
        PAQ=C + PAD_AQ,
        PM=(C + PAD_M) if pad_f32 else C,
        PZ=(EV + PAD_Z) if pad_f32 else EV,
    )


def _fwd_lds_terms(DK, DV, C, DV_SPLIT, solve, pad_f32, has_initial_state=False):
    EV = DV // DV_SPLIT
    p = _fwd_pads(DK, C, EV, pad_f32)
    PSB, PXK, PYK = p["PSB"], p["PXK"], p["PYK"]
    PYT, PVN, PAQ = p["PYT"], p["PVN"], p["PAQ"]
    PM, PZ = p["PM"], p["PZ"]
    return dict(
        N_G=C * DK,
        N_STB=EV * PSB,
        N_K=C * DK,
        N_Q=C * DK,
        N_X=C * PXK,
        N_X2=(C * PXK) if MERGE_TILES else 8,
        N_Y=(C * PYK) if EARLY_YT else max(C * PYK, DK * PYT),
        N_YT=(DK * PYT) if EARLY_YT else 8,
        N_M=C * PM,
        N_Z=max((C + (1 if solve == "right" else 0)) * PZ, (C * PAQ + 1) // 2),
        N_VN=EV * PVN,
        REGQ_OK=DV_SPLIT <= REG_Q_MAX_SPLIT,
        REGK_OK=DV_SPLIT <= REG_K_MAX_SPLIT,
        N_FIX=C + DK + DK,
        # register-scan extras: one lower-half total per thread, one reference row per
        # channel, and (only with an initial state) the fp32 staging tile g_cum used to
        # lend out
        N_REG=2 * DK + DK + (EV * DK if has_initial_state else 8) + 8,  # BLOCK == 2*DK
    )


def _fwd_lds_size(t, mode, alias_vn=False):
    """Bytes for one layout choice; ``t`` is a :func:`_fwd_lds_terms` dict.

    ``mode`` is ``"plain"`` (every tile stand-alone), ``"alias"`` (queries, and with
    ``alias_vn`` also V~, overlaid on the dead g_cum tile) or ``"reg"`` (no g_cum at
    all -- the cumulative decay lives in registers).
    """
    n_g, n_q, n_vn, extra = t["N_G"], t["N_Q"], t["N_VN"], 0
    if mode == "alias":
        used = n_q + (n_vn if alias_vn else 0)
        n_g = max(n_g, (2 * used + 3) // 4)
        n_q = 64  # placeholder field, sized to keep later fields 128 B aligned
        if alias_vn:
            n_vn = 64
    elif mode == "reg":
        n_g, extra = 8, t["N_REG"]
        if REG_Q and t["REGQ_OK"]:
            n_q = 64  # placeholder field; the queries live in registers
        if REG_K and t["REGK_OK"]:
            t = dict(t, N_K=64)  # keys live in registers too
    return (
        4 * (n_g + t["N_M"] + t["N_Z"] + extra)
        + 2 * (t["N_STB"] + t["N_K"] + n_q + t["N_X"] + t["N_X2"] + t["N_Y"] + t["N_YT"] + n_vn)
        + 4 * t["N_FIX"]
    )


def fwd_lds_mode(*, DK=128, DV=128, C=32, DV_SPLIT=4, solve="blocked", pad_f32=True,
                 cache_g=True, specialize=False, has_initial_state=False):
    """Which shared-memory layout :func:`build_kda_fwd_module` will use.

    ``cache_g=False`` re-reads the decay from LDS in the four later tile builds and
    ``specialize`` re-reads it in the solve-shadowed one, so neither can give up the
    g_cum tile; both stay on ``"plain"``.  Otherwise prefer ``"reg"``, which drops the
    tile outright.  ``"alias"`` is the weaker fallback (it only overlays the tile) and
    is taken only when the shrunk footprint leaves two or more workgroups resident per
    CU -- that is where the co-resident workgroup hides the deferred query store; at
    one workgroup per CU (one wave per SIMD) it costs ~2.3 % instead.
    """
    if not (cache_g and not specialize):
        return "plain"
    t = _fwd_lds_terms(DK, DV, C, DV_SPLIT, solve, pad_f32, has_initial_state)
    if REG_SCAN:
        return "reg"
    if not ALIAS_LDS:
        return "plain"
    small = _fwd_lds_size(t, "alias", ALIAS_VN)
    occ = LDS_PER_CU // (-(-small // LDS_GRANULE) * LDS_GRANULE)
    return "alias" if occ >= 2 else "plain"


def fwd_solve(BH, DV_SPLIT, num_cus, C=32, BLOCK=256):
    """Triangular-solve strategy for a launch shape.

    ``"wy"`` (explicit inverse -> MFMA) has by far the shorter dependent chain but
    needs ~264 VGPRs against ``"blocked"``'s 240.  256 is the ceiling for two waves
    per SIMD, so in isolation WY cost an occupancy step at large grids and had to be
    gated to ``WGs <= NUM_CUS`` (measured alone: large 1.33x -> 1.46x at 256 WGs but
    xlarge 2.08x -> 1.35x at 1024 WGs).

    THAT GATE IS NOW OBSOLETE.  Once the register-caching gate (REG_Q_MAX_SPLIT=4)
    and the LDS/barrier rework (EARLY_YT / MERGE_TILES / MASK_IN_REG) are also in,
    the register pressure and the exposed-latency balance both move and WY wins
    EVERYWHERE.  Re-swept on the integrated tree, gfx950/256 CU, 6-case geomean:
      WGs <= NUM_CUS   1.7504   (xlarge 2.192x, longT 2.211x -- both on "blocked")
      WGs <= 2*NUM_CUS 1.7691   (longT flips to WY: 2.211x -> 2.372x)
      always WY        1.7922   (xlarge 2.356x, longT 2.375x)
    So the rule is now unconditional for the C=32 / BLOCK=256 geometry.

    Attempts to get WY under 256 that did NOT work, do not re-try: dropping the
    [EV, C] accumulator and its bf16 mirror in favour of reusing frag_Z (still 260,
    and 1 % slower where WY is taken); replacing the 8x8 register-array bottom with
    a 1x1 doubling recursion that holds no register array at all (still 260, and
    2-3 % slower); ``--amdgpu-num-vgpr`` / ``amdgpu-waves-per-eu`` compile hints
    (accepted by ``flyc.compile[...]`` but not honoured by this backend).
    """
    if C != 32 or BLOCK != 256:
        return "blocked"
    return "wy"


def fwd_lds_bytes(
    *, DK=128, DV=128, C=32, DV_SPLIT=4, solve="blocked", pad_f32=True, cache_g=True,
    specialize=False, has_initial_state=False,
):
    """LDS footprint of :func:`build_kda_fwd_module` for a geometry, without building it.

    Building a module to read ``.lds_bytes`` costs ~37 ms, which is far too much for a
    dispatch rule that has to run before the module is chosen.  The builder derives its
    own ``lds_bytes`` from this function, so the two cannot drift; keep ``_fwd_lds_terms``
    in sync if the shared-memory struct changes.
    """
    kw = dict(DK=DK, DV=DV, C=C, DV_SPLIT=DV_SPLIT, solve=solve, pad_f32=pad_f32,
              cache_g=cache_g, specialize=specialize, has_initial_state=has_initial_state)
    t = _fwd_lds_terms(DK, DV, C, DV_SPLIT, solve, pad_f32, has_initial_state)
    return _fwd_lds_size(t, fwd_lds_mode(**kw), ALIAS_VN)


def build_kda_fwd_module(
    *,
    BH: int,
    T: int,
    DK: int = 128,
    DV: int = 128,
    C: int = 32,
    DV_SPLIT: int = 4,
    BLOCK: int = 256,
    has_initial_state: bool = False,
    store_final_state: bool = True,
    out_dtype=BFloat16,
    debug_stage: str = "",
    ablate_solve: bool = False,
    ablate_gq: bool = False,
    solve: str = "blocked",
    solve_bs: int = 8,
    pad_f32: bool = True,
    specialize: bool = False,
    cache_g: bool = True,
    cache_kq: bool = False,
):
    """Build the KDA chunkwise forward kernel.

    Parameters
    ----------
    BH : batch * heads
    T  : sequence length (multiple of C)
    DK, DV : head dims for keys/queries and values
    C  : chunk length.  The paper uses 64; 32 is faster here (see module docstring).
    DV_SPLIT : how many workgroups split the value channels of one head.  Each
        workgroup owns EV = DV // DV_SPLIT value channels; the key-side work
        (T', A, Tril(...)) is replicated across the split.
    debug_stage : when set, the named intermediate is written to ``arg_ht`` in fp32
        instead of the final state.  The tap is unguarded, so with several chunks the
        surviving value is the last one -- drive a single chunk to inspect the first.
        See ``STAGES`` in ``tests/kernels/kda_stages.py`` for the names.
    ablate_solve : timing only.  Skips the triangular solve while leaving the rest of
        the pipeline intact, to attribute its cost.  Produces wrong results.
    ablate_gq : timing only.  Skips the two Gamma.Q elementwise passes, which are the
        only solve-independent work with no MMA in it, to size how much a
        wave-specialized overlap could hide.  Produces wrong results.
    solve : triangular solve strategy, all producing the same result.  ``"blocked"``
        alternates small register-resident diagonal solves with fully parallel
        rank-BS updates, ``"reg"`` keeps a thread's whole value channel in registers,
        ``"left"`` re-reads it from LDS, ``"right"`` fans each pivot out over all
        threads.  Kept selectable because they trade dependent-FMA depth against
        barriers and LDS traffic differently.
    solve_bs : diagonal block size for ``solve="blocked"``.  Smaller shortens the
        dependent chain but adds barriers and redundant update work.
    pad_f32 : pad the row stride of the fp32 tiles that receive an MMA C-fragment.
    specialize : hand the (Gamma.Q)/gamma^ref tile to the half of the block that the
        triangular solve leaves idle, so the two run concurrently.  Off by default:
        it measures 1.4x *slower*.  The tile build is LDS-latency-bound rather than
        issue-bound, so running it on half the waves costs ~3.9x rather than the 2x
        the thread count suggests -- losing the co-issued waves loses the latency
        hiding that made it cheap -- and it then no longer fits in the solve's
        shadow.  Kept so the measurement is reproducible.
    cache_g : keep this thread's slice of the cumulative decay, and the two reused
        exponentials         of it, in registers across the five [C, DK] tile builds instead
        of re-reading and recomputing them in each.
    cache_kq : likewise keep this thread's k and q values in registers rather than
        re-reading them from LDS in each pass that needs them.  Off by default: it
        helps on its own but spills when combined with ``cache_g``, and the pair
        measures 1.3x slower than ``cache_g`` alone.
    """
    if solve not in ("reg", "blocked", "left", "right", "wy"):
        raise ValueError(f"solve must be one of reg/blocked/left/right/wy; got {solve!r}")
    if solve == "wy" and not (C == 32 and BLOCK == 256):
        raise ValueError(f"solve='wy' needs C=32 and BLOCK=256; got C={C} BLOCK={BLOCK}")
    if T % C != 0:
        raise ValueError(f"T={T} must be a multiple of C={C}")
    if DV % DV_SPLIT != 0:
        raise ValueError(f"DV={DV} must be divisible by DV_SPLIT={DV_SPLIT}")
    if BLOCK != 2 * DK:
        raise ValueError(f"BLOCK must be 2*DK for the in-place cumulative sum; got {BLOCK}, DK={DK}")
    # C is the contracted extent of the Tril(Aqk) V~ matmul, so it has to be a
    # whole number of MFMA K steps
    if C % 32 != 0:
        raise ValueError(f"C must be a multiple of 32 (the MFMA K step); got {C}")
    if has_initial_state and DV // DV_SPLIT > C:
        # the initial state is staged through the (C x DK) g_cum buffer
        raise ValueError(f"has_initial_state needs DV//DV_SPLIT <= C; got {DV // DV_SPLIT} > {C}")

    EV = DV // DV_SPLIT
    # EV is an MMA M/N extent that the 2x2 wave grid halves, and each half must
    # still cover a full 16-wide MFMA tile
    if EV < 32:
        raise ValueError(f"DV//DV_SPLIT must be >= 32 for the 2x2 wave grid; got {EV}")
    NC = T // C
    CREF = C // 2  # reference row for the factored C x C construction

    # ── LDS geometry ────────────────────────────────────────────────────────
    # Only tiles that feed MFMA operands are padded; padding a bf16 tile's row
    # stride to 136 (68 dwords, 68 % 32 == 4) makes the 8-lane ds_read_b128
    # phases hit 8 distinct bank quads instead of all landing on bank 0.
    # ...and the fp32 tiles that receive an MMA C-fragment need padding too: an
    # accumulator holds 4 values per lane strided by the row pitch, and lane groups
    # differ only in which 4-row block they own, so an unpadded 32-dword pitch puts
    # every group on the same banks.  A pitch of 4 (mod 32) staggers rows instead.
    #
    # Every pad is now per-buffer -- see ``_fwd_pads``, which is the SINGLE source
    # shared with the host-side occupancy estimator.
    _p = _fwd_pads(DK, C, EV, pad_f32)
    PSB, PXK, PYK = _p["PSB"], _p["PXK"], _p["PYK"]
    PYT, PVN, PAQ = _p["PYT"], _p["PVN"], _p["PAQ"]
    PM, PZ = _p["PM"], _p["PZ"]

    N_G = C * DK  # fp32  cumulative log decay, unpadded (scalar access only)
    N_STB = EV * PSB  # bf16  S^T mirror                    (MFMA B)
    N_K = C * DK  # bf16  keys, unpadded (scalar access only)
    N_Q = C * DK  # bf16  queries, unpadded  (aliased into the g_cum arena)
    N_X = C * PXK  # bf16  A-operand scratch              (MFMA A)
    N_X2 = (C * PXK) if MERGE_TILES else 8  # second A-operand tile (see MERGE_TILES)
    # bf16 B-operand scratch (MFMA B).  With EARLY_YT the transposed view moves to its
    # own buffer (see the flag) and y_s carries only the [C, DK] K/Gamma tile.
    N_Y = (C * PYK) if EARLY_YT else max(C * PYK, DK * PYT)
    N_YT = (DK * PYT) if EARLY_YT else 8
    N_M = C * PM  # fp32  T' then raw Tril(QK)
    # fp32 rhs / solve workspace, also reused as the bf16 masked Tril tile, so it
    # must be large enough for whichever of the two views is bigger.  Row C is a
    # write-only sink that absorbs the masked lanes of the right-looking solve, so
    # every lane can store unconditionally.
    N_Z = max((C + (1 if solve == "right" else 0)) * PZ, (C * PAQ + 1) // 2)
    # thread mapping for the right-looking solve: a fixed value channel per thread,
    # with RSTEP rows updated concurrently
    RSTEP = BLOCK // EV
    # The solve-shadowed tile build runs on threads [GQ_BASE, BLOCK), which must be
    # disjoint from the solve's [0, EV) and must still tile a full [C, DK] block.
    # blocked solve geometry: BS-row diagonal blocks, each followed by a rank-BS
    # update of the rows below.  The update is required to tile BLOCK exactly so no
    # lane needs predicating.
    # WY solve geometry: 8x8 register-inverted diagonal blocks, then two levels of
    # 2x2 block-inverse recursion (8 -> 16 -> 32).
    WYB = 8
    WYNB = C // WYB
    WYH = C // 2
    # GQ_IN_SOLVE_TAG: fold the (Gamma.Q) tile build into the WY solve's most
    # starved barrier region (the 32-of-256-thread 8x8 diagonal-block inversion).
    GQ_EARLY = GQ_IN_SOLVE and solve == "wy" and not ablate_gq
    # ... and, one barrier region later, the Tril(QK) matmul that consumes the tile
    # GQ_EARLY just produced.  It is pure matrix-core work into an accumulator
    # fragment, and the profile has MFMA at 2.78% of peak, so it hides completely
    # behind the VALU-only 8x8->16x16 block-inverse recursion.  QK_MFMA_EARLY_TAG
    # rows of the (Gamma.Q) tile handed to the FIRST starved region; the rest go to
    # the second one, alongside the Tril(QK) matmul.
    GQ_NROW = C * DK // BLOCK
    GQ_CUT = min(GQ_ROW_CUT, GQ_NROW)
    QK_EARLY = QK_MFMA_EARLY and GQ_EARLY and DV_SPLIT >= QK_EARLY_MIN_SPLIT
    BS = min(solve_bs, C)
    NB = C // BS
    if solve == "blocked":
        if C % BS != 0:
            raise ValueError(f"blocked solve needs BS | C; got BS={BS} C={C}")
        for _b in range(NB - 1):
            if ((C - _b * BS - BS) * EV) % BLOCK != 0:
                raise ValueError(
                    f"blocked solve needs BLOCK | (C-r0-BS)*EV for every block; "
                    f"got C={C} BS={BS} EV={EV} BLOCK={BLOCK}"
                )
    GQ_BASE = BLOCK // 2
    if specialize and solve in ("reg", "left"):
        if GQ_BASE < EV:
            raise ValueError(f"specialize needs BLOCK//2 >= EV; got {GQ_BASE} < {EV}")
        if (BLOCK - GQ_BASE) % DK != 0 or C % ((BLOCK - GQ_BASE) // DK) != 0:
            raise ValueError(
                f"specialize needs DK | BLOCK-BLOCK//2 and C | their ratio; "
                f"got BLOCK={BLOCK} DK={DK} C={C}"
            )
    else:
        specialize = False
    N_VN = EV * PVN  # bf16  V~^T                           (MFMA A and B)
    # ── g_cum / q_s / vn_s share one arena ──────────────────────────────────
    # The cumulative log-decay tile is read only until the chunk's first pair of
    # factored [C, DK] tiles has been built -- ``cache_g`` already holds everything
    # the four later tile builds need in registers -- and the barrier that ends that
    # build is also the point before which nothing reads the query tile or V~.  So
    # ``g_cum`` overlays the two of them, and the chunk-loop back edge is safe because
    # the last reader of either (the O and S^T matmuls) is separated from the next
    # chunk's g load by the publish_state barrier pair.
    #
    #   [ 0 .............................. N_Q ) [ N_Q .......... N_Q + N_VN )
    #   |<---------------- q_s ----------------->|<---------- vn_s ---------->|
    #   |<------------------------- g_cum (fp32) ------------------ ... ----->|
    #
    # 16-byte alignment of the vn_s slice holds because N_Q = C*DK is a multiple of 8.
    #
    # See ``fwd_lds_mode`` for which layout is taken, and why.
    MODE = fwd_lds_mode(
        DK=DK, DV=DV, C=C, DV_SPLIT=DV_SPLIT, solve=solve, pad_f32=pad_f32,
        cache_g=cache_g, specialize=specialize, has_initial_state=has_initial_state,
    )
    REG_G = MODE == "reg"
    ALIAS = MODE == "alias"
    ALIASV = ALIAS and ALIAS_VN
    N_GX = BLOCK  # one lower-half decay total per thread
    N_SF = EV * DK if (REG_G and has_initial_state) else 8
    N_ARENA, NQ_F, NVN_F = N_G, N_Q, N_VN
    # queries stay in registers only where every reader uses the register-scan cell
    # mapping (grow0/gcol); ``specialize`` hands the Gamma.Q build to a different
    # thread partition and therefore still needs the shared tile.
    REGQ = REG_G and REG_Q and not specialize and DV_SPLIT <= REG_Q_MAX_SPLIT
    REGK = REG_G and REG_K and cache_g and DV_SPLIT <= REG_K_MAX_SPLIT
    # the vectorized yt_s flush needs the register scan's contiguous row slice, a
    # 16 B-aligned row base and a whole number of 8-element groups per thread
    # contiguous rows of vVn per thread for the WY rhs transpose
    NRV = C // (BLOCK // EV) if (BLOCK % EV == 0 and C % (BLOCK // EV) == 0) else 0
    VNW = min(NRV, VN_VEC_WIDTH) if NRV else 0
    VNVEC = (VN_VEC and solve == "wy" and NRV in (2, 4, 8) and VNW in (2, 4, 8)
             and NRV % VNW == 0 and PVN % VNW == 0)
    # the fused form additionally needs the fp32 rhs in vZ to have no other reader
    VNFUSE = (VN_FUSE_RHS and solve == "wy" and not ablate_solve and debug_stage == "")
    # closed-form register-budget gate, not a shape test: see SHARE_STATE_B_MAX_FRAG
    SHARE_B = SHARE_STATE_B and (EV * DK // 128) <= SHARE_STATE_B_MAX_FRAG
    N_K_F = 64 if REGK else N_K
    if REG_G:
        N_ARENA = 8  # g_cum is gone; the decay lives in registers
        if REGQ:
            NQ_F = 64  # placeholder; the queries live in registers
    if ALIAS:
        N_ARENA = max(N_G, (2 * (N_Q + (N_VN if ALIASV else 0)) + 3) // 4)  # fp32 elements
        NQ_F = 64  # placeholder; storage lives in the arena, size keeps 128 B alignment
        if ALIASV:
            NVN_F = 64

    # single source of truth, so the host-side occupancy rule sees the same number
    lds_bytes = fwd_lds_bytes(
        DK=DK, DV=DV, C=C, DV_SPLIT=DV_SPLIT, solve=solve, pad_f32=pad_f32,
        cache_g=cache_g, specialize=specialize, has_initial_state=has_initial_state,
    )
    if lds_bytes > LDS_PER_CU:
        raise ValueError(f"LDS request {lds_bytes} B exceeds the 160 KB gfx950 budget")

    @fx.struct
    class SharedStorage:
        g_cum: fx.Array[Float32, N_ARENA, 16]
        m_mat: fx.Array[Float32, N_M, 16]
        z_buf: fx.Array[Float32, N_Z, 16]
        st_b: fx.Array[BFloat16, N_STB, 16]
        k_s: fx.Array[BFloat16, N_K_F, 16]
        q_s: fx.Array[BFloat16, NQ_F, 16]
        x_s: fx.Array[BFloat16, N_X, 16]
        x2_s: fx.Array[BFloat16, N_X2, 16]
        y_s: fx.Array[BFloat16, N_Y, 16]
        yt_s: fx.Array[BFloat16, N_YT, 16]
        vn_s: fx.Array[BFloat16, NVN_F, 16]
        beta_s: fx.Array[Float32, C, 16]
        gl_s: fx.Array[Float32, DK, 16]
        dec_s: fx.Array[Float32, DK, 16]
        gx_s: fx.Array[Float32, N_GX, 16]
        gr_s: fx.Array[Float32, DK, 16]
        sf_s: fx.Array[Float32, N_SF, 16]

    @flyc.kernel
    def kda_fwd_kernel(
        arg_q: fx.Tensor,  # [BH*NC, C*DK] bf16
        arg_k: fx.Tensor,  # [BH*NC, C*DK] bf16
        arg_g: fx.Tensor,  # [BH*NC, C*DK] f32
        arg_beta: fx.Tensor,  # [BH*NC, C]    f32
        arg_v: fx.Tensor,  # [BH*NC, C, DV_SPLIT, EV] bf16
        arg_o: fx.Tensor,  # [BH*NC, C, DV_SPLIT, EV] out
        arg_h0: fx.Tensor,  # [BH, DV_SPLIT, EV, DK] f32
        arg_ht: fx.Tensor,  # [BH, DV_SPLIT, EV, DK] f32
        n_chunks: fx.Int32,
        q_scale: fx.Float32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        bh = bid // DV_SPLIT
        sp = bid % DV_SPLIT

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        # ── LDS views ───────────────────────────────────────────────────────
        vG = lds.g_cum.view(fx.make_layout((C, DK), (DK, 1)))
        vM = lds.m_mat.view(fx.make_layout((C, C), (PM, 1)))
        vZ = lds.z_buf.view(fx.make_layout((C, EV), (PZ, 1)))
        vStb = lds.st_b.view(fx.make_layout((EV, DK), (PSB, 1)))
        # same bytes, MMA-C-native orientation: M = DK is the contiguous mode, so a
        # C-fragment's four values per lane are four adjacent bf16 (see ST_TRANSPOSE)
        vStT = fx.make_view(lds.st_b.ptr, fx.make_layout((DK, EV), (1, PSB)))
        vK = None if REGK else lds.k_s.view(fx.make_layout((C, DK), (DK, 1)))
        vX = lds.x_s.view(fx.make_layout((C, DK), (PXK, 1)))
        vX2 = (lds.x2_s if const_expr(MERGE_TILES) else lds.x_s).view(
            fx.make_layout((C, DK), (PXK, 1)))
        vY = lds.y_s.view(fx.make_layout((C, DK), (PYK, 1)))  # K/Gamma tile   [j, d]
        vYt = (lds.yt_s if const_expr(EARLY_YT) else lds.y_s).view(
            fx.make_layout((DK, C), (PYT, 1)))  # (K.gamma^C/Gamma)^T [d, c]
        # queries and V~ live inside the g_cum arena (see N_ARENA): q_s first, then
        # vn_s.  Both are written only after the barrier that retires g_cum.
        qBase = fx.recast_iter(BFloat16, lds.g_cum.ptr) if const_expr(ALIAS) else lds.q_s.ptr
        vnBase = fx.add_offset(qBase, N_Q) if const_expr(ALIASV) else lds.vn_s.ptr
        vQ = fx.make_view(qBase, fx.make_layout((C, DK), (DK, 1)))
        vVn = fx.make_view(vnBase, fx.make_layout((EV, C), (PVN, 1)))
        # the fp32 solve workspace is reused as the bf16 masked Tril tile once the
        # solve has consumed it (C*PAQ*2 <= C*EV*4 bytes)
        vAq = fx.make_view(fx.recast_iter(BFloat16, lds.z_buf.ptr), fx.make_layout((C, C), (PAQ, 1)))
        # fp32 [C, C] scratch for the WY (explicit-inverse) solve.  It overlays the
        # same z_buf arena: the rhs is transposed into vVn (bf16) before the inverse
        # is built, so the fp32 solve workspace is dead from that point on.
        # N_Z = C*PZ >= C*(EV+4) >= C*36 > C*C, so the view always fits.
        vW = lds.z_buf.view(fx.make_layout((C, C), (C, 1)))
        # fp32 staging for the initial state.  Without a register scan this borrows
        # g_cum (dead until the first chunk's decay load); with one there is no g_cum
        # left, so sf_s carries it -- sized to nothing unless an initial state is given.
        vSf = fx.make_view(
            lds.sf_s.ptr if const_expr(REG_G) else lds.g_cum.ptr,
            fx.make_layout((EV, DK), (DK, 1)),
        )
        # ...and its MMA-C-native transpose, for the accumulator load (see ST_TRANSPOSE)
        vSfT = fx.make_view(
            lds.sf_s.ptr if const_expr(REG_G) else lds.g_cum.ptr,
            fx.make_layout((DK, EV), (1, DK)),
        )
        # register scan scratch: one lower-half decay total per thread, then one
        # reference-row value per channel
        vGx = lds.gx_s.view(fx.make_layout(N_GX, 1))
        vGr = lds.gr_s.view(fx.make_layout(DK, 1))
        vBeta = lds.beta_s.view(fx.make_layout(C, 1))
        # beta broadcast along columns (stride 0), so Diag(beta) can be applied to
        # the [C, C] accumulator in registers instead of in a separate LDS pass
        vBetaC = lds.beta_s.view(fx.make_layout((C, C), (1, 0)))
        vGl = lds.gl_s.view(fx.make_layout(DK, 1))
        vDec = lds.dec_s.view(fx.make_layout(DK, 1))
        # stride-0 broadcast of the per-channel chunk decay over the S^T tile,
        # so it can be partitioned exactly like the state accumulator.
        vDecC = lds.dec_s.view(
            fx.make_layout((DK, EV), (1, 0)) if const_expr(ST_TRANSPOSE)
            else fx.make_layout((EV, DK), (0, 1)))

        # flat views for the vectorized global -> LDS chunk loads
        fG = fx.logical_divide(lds.g_cum.view(fx.make_layout(N_G, 1)), fx.make_layout(4, 1))
        fK = None if REGK else fx.logical_divide(
            lds.k_s.view(fx.make_layout(N_K, 1)), fx.make_layout(8, 1))
        fQ = fx.logical_divide(fx.make_view(qBase, fx.make_layout(N_Q, 1)), fx.make_layout(8, 1))
        fVn = fx.logical_divide(
            fx.make_view(vnBase, fx.make_layout(N_VN, 1)), fx.make_layout(VNW, 1)
        ) if const_expr(VNVEC) else None

        # ── global tensors ──────────────────────────────────────────────────
        gQ = fx.rocdl.make_buffer_tensor(arg_q)
        gK = fx.rocdl.make_buffer_tensor(arg_k)
        gG = fx.rocdl.make_buffer_tensor(arg_g)
        gBeta = fx.rocdl.make_buffer_tensor(arg_beta)
        gV = fx.rocdl.make_buffer_tensor(arg_v)
        gO = fx.rocdl.make_buffer_tensor(arg_o)

        cp128_bf = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), BFloat16)
        cp128_f = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), Float32)
        cp16 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), BFloat16)
        cp32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), Float32)
        uni128 = fx.make_copy_atom(fx.UniversalCopy128b(), BFloat16)
        uni32 = fx.make_copy_atom(fx.UniversalCopy(32), Float32)
        # C-fragment stores must be single-element: an MMA accumulator holds its
        # 4 values per lane strided by the destination row pitch, not contiguous,
        # so a wider vector store would scatter them.
        uni16 = fx.make_copy_atom(fx.UniversalCopy(16), BFloat16)

        # ── MMA: one 2x2-wave TiledMma drives every matmul in the kernel ────
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, BFloat16))
        tmma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))
        thr_mma = tmma.thr_slice(tid)
        thr_a = fx.make_tiled_copy_A(uni128, tmma).get_slice(tid)
        thr_b = fx.make_tiled_copy_B(uni128, tmma).get_slice(tid)
        thr_c32 = fx.make_tiled_copy_C(uni32, tmma).get_slice(tid)

        # NOTE: every ``thr_*.method()`` call must stay inside one of these
        # closures.  The AST rewriter turns any object whose method is invoked
        # directly in a dynamic loop body into an scf.for iter_arg, and ThrMma /
        # ThrCopy cannot be rebuilt from IR values.
        def gemm_lds(sA, sB, frag_C):
            """frag_C += sA @ sB^T with both operands read from LDS."""
            fA = thr_mma.make_fragment_A(sA)
            fB = thr_mma.make_fragment_B(sB)
            fx.copy(uni128, thr_a.partition_S(sA), thr_a.retile(fA))
            fx.copy(uni128, thr_b.partition_S(sB), thr_b.retile(fB))
            fx.gemm(tmma, frag_C, fA, fB, frag_C)

        def gemm_lds_bs(sA, frag_C):
            """frag_C += sA @ vStb^T, reusing the hoisted state B fragment."""
            fA = thr_mma.make_fragment_A(sA)
            fx.copy(uni128, thr_a.partition_S(sA), thr_a.retile(fA))
            fx.gemm(tmma, frag_C, fA, frag_Bs, frag_C)

        def store_acc_f32(frag, dst_view):
            fx.copy(uni32, thr_c32.retile(frag), thr_c32.partition_S(dst_view))

        def load_acc_f32(frag, src_view):
            fx.copy(uni32, thr_c32.partition_S(src_view), thr_c32.retile(frag))

        def make_acc(shape_view):
            return thr_mma.make_fragment_C(shape_view)

        def zero(frag):
            frag.fill(0)

        # ── elementwise iteration helpers ───────────────────────────────────
        # A [R, COLS] tile is walked so that each thread keeps a *fixed* column
        # (tid % COLS) and strides over rows.  Consecutive lanes then touch
        # consecutive LDS addresses (conflict-free) and per-column constants such
        # as the reference row are loaded once per thread.
        def tile_iter(R, COLS):
            col = tid % COLS
            row0 = tid // COLS
            rstep = BLOCK // COLS
            return col, row0, rstep, R // rstep

        def tile_iter_sub(R, COLS, base, nthr):
            """tile_iter restricted to threads [base, base + nthr)."""
            t = tid - base
            return t % COLS, t // COLS, nthr // COLS, R // (nthr // COLS)

        # ── right-looking solve mapping ─────────────────────────────────────
        # Each thread owns one value channel and a slice of the rows below the
        # pivot.  Lanes whose row runs past the tile are folded onto the sink row
        # so the trailing update needs no predication.
        e_ch = tid % EV
        r_ch = tid // EV

        def row_or_sink(rr, exact):
            if const_expr(exact):
                return rr
            return (fx.Int32(rr) < fx.Int32(C)).select(fx.Int32(rr), fx.Int32(C))

        c_nclamp = fx.Float32(-EXP2_CLAMP)
        def clamp2(x):
            # Numeric.minimumf is broken upstream (ArithValue only defines
            # maximumf), so the upper clamp is written as min(x,C) = -max(-x,-C).
            return (-((-x).maximumf(c_nclamp))).maximumf(c_nclamp)

        def ex2c(x):
            """exp2 of an argument already inside [-EXP2_CLAMP, EXP2_CLAMP].

            ``Numeric.exp2`` lowers to ``math.exp2``, which this toolchain turns
            into a call to ``__ocml_exp2_f32``.  OCML's exp2 spends six extra VALU
            instructions (v_cmp_gt / v_cndmask / v_add / v_cndmask / v_ldexp plus
            the branch) pre- and post-scaling by a power of two so that arguments
            below -126 still return a denormal.  Every argument here is clamped to
            [-EXP2_CLAMP, EXP2_CLAMP] first, so that path is unreachable and the
            hardware v_exp_f32 is exact-equivalent.  MEASURED on the shipped ISA:
            65 v_exp_f32 came with 65 v_ldexp_f32 + 151 v_cndmask.  The MLIR
            ``fastmath<afn>`` flag does NOT help -- it survives to stage 09 and is
            then dropped at the OCML call boundary -- so go straight to the ROCDL
            intrinsic.
            """
            if const_expr(EXP2_AFN):
                return fx.Float32(fx.rocdl.exp2(Float32.ir_type, x.ir_value()))
            return x.exp2()

        def ex2(x):
            return ex2c(clamp2(x))

        def fma(a, b, c):
            """a * b + c as a single instruction.  See FMA_CONTRACT."""
            if const_expr(FMA_CONTRACT):
                return fx.Float32(fx.math.fma(a.ir_value(), b.ir_value(), c.ir_value()))
            return c + a * b

        # ── state accumulator: S^T, fp32, persistent across chunks ──────────
        # (with ST_TRANSPOSE the accumulator is S [DK, EV] over the same bytes)
        frag_S = thr_mma.make_fragment_C(vStT if const_expr(ST_TRANSPOSE) else vStb)
        frag_S.fill(0)
        if const_expr(has_initial_state):
            # stage S^T through LDS: a C-layout partition of a *global* tensor is
            # not a supported copy source, so load tile-wise and re-read as an
            # accumulator-shaped view.
            gH0 = fx.rocdl.make_buffer_tensor(arg_h0)
            gh = fx.zipped_divide(fx.slice(gH0, (bh, sp, None, None)), fx.make_tile(1, 1))
            col, row0, rstep, nrow = tile_iter(EV, DK)
            for i in range_constexpr(nrow):
                rr = row0 + i * rstep
                rh = fx.make_rmem_tensor(1, Float32)
                fx.copy_atom_call(cp32, fx.slice(gh, (None, (rr, col))), rh)
                fx.memref_store(fx.memref_load_vec(rh)[0], vSf, (rr, col))
            gpu.barrier()
            load_acc_f32(frag_S, vSfT if const_expr(ST_TRANSPOSE) else vSf)
            gpu.barrier()

        # bf16 mirror of the state for MFMA reads
        frag_Sb = fx.make_fragment_like(frag_S, BFloat16.ir_type)
        thr_cb = fx.make_tiled_copy_C(uni16, tmma).get_slice(tid)
        # ST_TRANSPOSE makes the C-fragment's four values per lane contiguous in the
        # destination, so the mirror goes out as one ds_write_b64 instead of four b16.
        uni64 = fx.make_copy_atom(fx.UniversalCopy(64), BFloat16)
        pub_atom = uni64 if const_expr(ST_TRANSPOSE) else uni16
        thr_cbp = fx.make_tiled_copy_C(pub_atom, tmma).get_slice(tid)
        pub_dst = vStT if const_expr(ST_TRANSPOSE) else vStb

        # hoisted, shared B operand for the two matmuls against the state (see
        # SHARE_STATE_B).  Allocated out here for the same reason as frag_S.
        frag_Bs = thr_mma.make_fragment_B(vStb) if const_expr(SHARE_B) else None

        def fetch_state_b():
            fx.copy(uni128, thr_b.partition_S(vStb), thr_b.retile(frag_Bs))

        def publish_state():
            fx.memref_store_vec(fx.memref_load_vec(frag_S).to(BFloat16), frag_Sb)
            fx.copy(pub_atom, thr_cbp.retile(frag_Sb), thr_cbp.partition_S(pub_dst))

        # q is read from global at the top of the chunk but cannot be written to LDS
        # until the factored tile build has retired g_cum (they share an arena), so it
        # is parked in these registers in between.  Allocated once outside the loop:
        # a list comprehension in the dynamic loop body would be traced as a loop.
        NQI = N_Q // (8 * BLOCK)
        qg = [fx.make_rmem_tensor(8, BFloat16) for _ in range(NQI)]
        vng = [fx.make_rmem_tensor(VNW, BFloat16) for _ in range(NRV // VNW)] if VNVEC else []
        vn_atom = fx.make_copy_atom(fx.UniversalCopy(16 * VNW), BFloat16) if VNVEC else None

        def store_q():
            for i in range_constexpr(NQI):
                fx.copy_atom_call(uni128, qg[i], fx.slice(fQ, (None, tid + i * BLOCK)))

        # per-chunk accumulators, allocated once outside the loop
        frag_M = make_acc(vM)  # C x C   : raw K/Q x (K/Gamma)^T
        frag_Z = make_acc(vZ)  # C x EV  : (Gamma.K) S, then reused
        frag_O = make_acc(vZ)  # C x EV  : output tile
        frag_dec = fx.make_fragment_like(frag_S, Float32.ir_type)
        if const_expr(solve == "wy"):
            # EV x C : V~^T = rhs^T (M^-1)^T, one MFMA instead of a substitution.
            frag_V = make_acc(vVn)
            frag_Vb = fx.make_fragment_like(frag_V, BFloat16.ir_type)

            def solve_mfma():
                """vVn <- vVn @ vAq^T.  The barrier between operand fetch and use is
                what lets the result be written back over the A operand."""
                fA = thr_mma.make_fragment_A(vVn)
                fB = thr_mma.make_fragment_B(vAq)
                fx.copy(uni128, thr_a.partition_S(vVn), thr_a.retile(fA))
                fx.copy(uni128, thr_b.partition_S(vAq), thr_b.retile(fB))
                gpu.barrier()
                frag_V.fill(0)
                fx.gemm(tmma, frag_V, fA, fB, frag_V)
                fx.memref_store_vec(fx.memref_load_vec(frag_V).to(BFloat16), frag_Vb)
                fx.copy(uni16, thr_cb.retile(frag_Vb), thr_cb.partition_S(vVn))

        cp_out = fx.make_copy_atom(
            fx.rocdl.BufferCopy32b() if const_expr(out_dtype is Float32) else fx.rocdl.BufferCopy16b(),
            out_dtype,
        )
        thr_co = fx.make_tiled_copy_C(cp_out, tmma).get_slice(tid)
        frag_Oo = fx.make_fragment_like(frag_O, out_dtype.ir_type)

        def apply_chunk_decay():
            """S^T <- Diag(gamma^C) S^T, broadcasting the per-channel decay."""
            fx.copy(uni32, thr_c32.partition_S(vDecC), thr_c32.retile(frag_dec))
            fx.memref_store_vec(fx.memref_load_vec(frag_S) * fx.memref_load_vec(frag_dec), frag_S)

        def store_output(gview, scale):
            fx.memref_store_vec((fx.memref_load_vec(frag_O) * scale).to(out_dtype), frag_Oo)
            fx.copy(cp_out, thr_co.retile(frag_Oo), thr_co.partition_S(gview))

        # ── debug tap: dump a [C, <=DK] LDS tile into arg_ht as fp32 ────────
        if const_expr(debug_stage != ""):
            gHtD = fx.rocdl.make_buffer_tensor(arg_ht)

            def dump(src, ncol, is_bf16=False):
                gd = fx.zipped_divide(fx.slice(gHtD, (bh, sp, None, None)), fx.make_tile(1, 1))
                col, row0, rstep, nrow = tile_iter(C, ncol)
                for i in range_constexpr(nrow):
                    rr = row0 + i * rstep
                    rw = fx.make_rmem_tensor(1, Float32)
                    val = fx.memref_load(src, (rr, col))
                    fx.memref_store(fx.Float32(val) if const_expr(not is_bf16) else val.to(Float32), rw, 0)
                    fx.copy_atom_call(cp32, rw, fx.slice(gd, (None, (rr, col))))

        # ── loop-invariant triangular masks, kept in accumulator registers ──
        # Both [C, C] tiles the chunk produces are masked immediately after their
        # matmul, which used to mean store-fp32 / barrier / read-mask-write in LDS.
        # The masks are chunk-invariant, so they are built once here (through the
        # otherwise-idle vM tile) into C-fragment registers; the mask then costs one
        # VALU multiply on a value already in the accumulator.
        frag_stril = make_acc(vM)
        frag_tril = make_acc(vM)
        if const_expr(MASK_IN_REG or TPRIME_IN_REG):
            mcol, mrow0, mrstep, mnrow = tile_iter(C, C)
            for i in range_constexpr(mnrow):
                rr = mrow0 + i * mrstep
                fx.memref_store(
                    (fx.Int32(rr) > fx.Int32(mcol)).select(fx.Float32(1.0), fx.Float32(0.0)),
                    vM, (rr, mcol))
            gpu.barrier()
            # frag_stril has exactly one reader, the TPRIME_IN_REG branch; building it
            # unconditionally leaves a dead C-fragment register file (and two barriers)
            # in the prologue whenever that branch is off, which it is by default.
            if const_expr(TPRIME_IN_REG):
                load_acc_f32(frag_stril, vM)
            gpu.barrier()
            for i in range_constexpr(mnrow):
                rr = mrow0 + i * mrstep
                fx.memref_store(
                    (fx.Int32(rr) >= fx.Int32(mcol)).select(fx.Float32(1.0), fx.Float32(0.0)),
                    vM, (rr, mcol))
            gpu.barrier()
            load_acc_f32(frag_tril, vM)
            gpu.barrier()
        frag_beta = make_acc(vBetaC)
        frag_aqb = fx.make_fragment_like(frag_tril, BFloat16.ir_type)

        def store_aq():
            """vAq <- Tril(frag_M) in bf16, straight out of the accumulator."""
            fx.memref_store_vec(
                (fx.memref_load_vec(frag_M) * fx.memref_load_vec(frag_tril)).to(BFloat16),
                frag_aqb)
            fx.copy(uni16, thr_cb.retile(frag_aqb), thr_cb.partition_S(vAq))

        publish_state()
        gpu.barrier()

        # ====================================================================
        # chunk loop
        # ====================================================================
        for n in range(n_chunks):
            tile = bh * NC + n
            if const_expr(debug_stage == "state_in"):
                dump(vStb, DK, is_bf16=True)

            # ---- load g, k, q for this chunk -------------------------------
            gg = fx.logical_divide(fx.slice(gG, (tile, None)), fx.make_layout(4, 1))
            gk = fx.logical_divide(fx.slice(gK, (tile, None)), fx.make_layout(8, 1))
            gq = fx.logical_divide(fx.slice(gQ, (tile, None)), fx.make_layout(8, 1))
            # [C, DK] cell -> thread.  With the register scan a thread owns a whole
            # C/2-row slice of one channel (that is what the scan produces); otherwise
            # the rows are interleaved.  Either way it is 16 cells of one column, and
            # lanes of a wave still walk consecutive columns of one row.
            if const_expr(REG_G):
                gcol = tid % DK
                ghalf = tid // DK
                grow0 = ghalf * (C // 2)
                grstep, gnrow = 1, C // 2
            else:
                gcol, grow0, grstep, gnrow = tile_iter(C, DK)
            gcr = fx.make_rmem_tensor(gnrow, Float32)
            egr = fx.make_rmem_tensor(gnrow, Float32)  # exp2(gcum)
            eir = fx.make_rmem_tensor(gnrow, Float32)  # exp2(gcum - gref)
            # k and q are read by three and two of the passes respectively; they come
            # from bf16 tiles, so caching them costs nothing in precision.
            kvr = fx.make_rmem_tensor(gnrow, BFloat16)
            qvr = fx.make_rmem_tensor(gnrow, BFloat16)

            if const_expr(REG_G):
                # The cumulative log-decay never reaches LDS: this thread reads its own
                # C/2 rows of one channel straight from global (lanes of a wave cover
                # consecutive channels, so each load is still one coalesced burst) and
                # scans them in a register.  That removes the 16 KB tile, ~80 LDS ops
                # per thread per chunk and two of the chunk's barriers; the only shared
                # traffic left is the lower-half total each channel hands to its
                # upper-half partner.
                gg1 = fx.logical_divide(fx.slice(gG, (tile, None)), fx.make_layout(1, 1))
                acc = fx.Float32(0.0)
                for i in range_constexpr(gnrow):
                    rg = fx.make_rmem_tensor(1, Float32)
                    fx.copy_atom_call(cp32, fx.slice(gg1, (None, (grow0 + i) * DK + gcol)), rg)
                    acc = acc + fx.memref_load_vec(rg)[0]
                    fx.memref_store(acc * LOG2E, gcr, i)
                fx.memref_store(acc * LOG2E, vGx, tid)
            else:
                for i in range_constexpr(N_G // (4 * BLOCK)):
                    idx = tid + i * BLOCK
                    r = fx.make_rmem_tensor(4, Float32)
                    fx.copy_atom_call(cp128_f, fx.slice(gg, (None, idx)), r)
                    fx.copy_atom_call(uni128, r, fx.slice(fG, (None, idx)))
            # q shares storage with g_cum under ALIAS, so it can only be *issued* here
            # -- the global read goes to registers and is parked there until the
            # factored tile build has retired g_cum a few hundred cycles later.
            if const_expr(REGK):
                gk1 = fx.logical_divide(fx.slice(gK, (tile, None)), fx.make_layout(1, 1))
                for i in range_constexpr(gnrow):
                    rk = fx.make_rmem_tensor(1, BFloat16)
                    fx.copy_atom_call(
                        cp16, fx.slice(gk1, (None, (grow0 + i * grstep) * DK + gcol)), rk
                    )
                    fx.memref_store(fx.memref_load_vec(rk)[0], kvr, i)
            else:
                for i in range_constexpr(N_K // (8 * BLOCK)):
                    idx = tid + i * BLOCK
                    r = fx.make_rmem_tensor(8, BFloat16)
                    fx.copy_atom_call(cp128_bf, fx.slice(gk, (None, idx)), r)
                    fx.copy_atom_call(uni128, r, fx.slice(fK, (None, idx)))
            if const_expr(REGQ):
                # one bf16 cell per row of this thread's channel slice; consecutive
                # lanes hold consecutive channels, so each of these is a coalesced
                # wave-wide burst, exactly like the register decay scan above.
                gq1 = fx.logical_divide(fx.slice(gQ, (tile, None)), fx.make_layout(1, 1))
                for i in range_constexpr(gnrow):
                    rq = fx.make_rmem_tensor(1, BFloat16)
                    fx.copy_atom_call(
                        cp16, fx.slice(gq1, (None, (grow0 + i * grstep) * DK + gcol)), rq
                    )
                    fx.memref_store(fx.memref_load_vec(rq)[0], qvr, i)
            else:
                for i in range_constexpr(NQI):
                    fx.copy_atom_call(cp128_bf, fx.slice(gq, (None, tid + i * BLOCK)), qg[i])
                if const_expr(not ALIAS):
                    store_q()

            # beta: replicated across thread groups (same value, write-only)
            gb = fx.logical_divide(fx.slice(gBeta, (tile, None)), fx.make_layout(1, 1))
            rb = fx.make_rmem_tensor(1, Float32)
            fx.copy_atom_call(cp32, fx.slice(gb, (None, tid % C)), rb)
            fx.memref_store(fx.memref_load_vec(rb)[0], vBeta, tid % C)
            gpu.barrier()
            # ---- state B operand: final since the previous chunk's publish_state,
            # and the barrier immediately above is the ONLY RAW fence against it.
            # DO NOT remove, move or const_expr-gate that barrier: with SHARE_B on,
            # this is the sole LDS read of vStb, so that barrier alone carries the
            # entire cross-chunk state recurrence (DROP_PUBLISH_BARRIER means there
            # is no fence after publish_state).  Margin here is exactly zero.
            # Fetching here puts the ds_reads in flight under the decay scan and the
            # tile build instead of leaving them exposed on the recurrence, and the
            # same registers then serve the tail's (Gamma.Q) S matmul.
            if const_expr(SHARE_B):
                fetch_state_b()
            if const_expr(TPRIME_IN_REG):
                load_acc_f32(frag_beta, vBetaC)

            # ---- cumulative log-decay, scaled to log2 ----------------------
            # Two halves scan concurrently (all BLOCK threads busy), then the
            # upper half is offset by the lower half's total.
            if const_expr(REG_G):
                # branch-free offset: every thread reads the total its channel's
                # lower-half owner (thread gcol) published, and the lower half adds 0
                addv = (fx.Int32(ghalf) > fx.Int32(0)).select(
                    fx.Float32(fx.memref_load(vGx, gcol)), fx.Float32(0.0)
                )
                for i in range_constexpr(gnrow):
                    fx.memref_store(fx.Float32(fx.memref_load(gcr, i)) + addv, gcr, i)
                if tid >= DK:
                    # this thread holds rows [C/2, C): its first is the reference row
                    # CREF and its last is the chunk total
                    fx.memref_store(fx.Float32(fx.memref_load(gcr, 0)), vGr, gcol)
                    gl = fx.Float32(fx.memref_load(gcr, gnrow - 1))
                    fx.memref_store(gl, vGl, gcol)
                    fx.memref_store(ex2(gl), vDec, gcol)
                gpu.barrier()
                gref = fx.Float32(fx.memref_load(vGr, gcol))
            else:
                d_c = tid % DK
                half = tid // DK
                r0 = half * (C // 2)
                acc = fx.Float32(0.0)
                for i in range_constexpr(C // 2):
                    acc = acc + fx.Float32(fx.memref_load(vG, (r0 + i, d_c)))
                    fx.memref_store(acc * LOG2E, vG, (r0 + i, d_c))
                gpu.barrier()
                col, row0, rstep, nrow = tile_iter(C // 2, DK)
                for i in range_constexpr(nrow):
                    rr = C // 2 + row0 + i * rstep
                    base = fx.Float32(fx.memref_load(vG, (C // 2 - 1, col)))
                    fx.memref_store(fx.Float32(fx.memref_load(vG, (rr, col))) + base, vG, (rr, col))
                gpu.barrier()
                dd = tid % DK
                gl = fx.Float32(fx.memref_load(vG, (C - 1, dd)))
                fx.memref_store(gl, vGl, dd)
                fx.memref_store(ex2(gl), vDec, dd)
                gpu.barrier()
                gref = fx.Float32(fx.memref_load(vG, (CREF, gcol)))

            # ---- factored C x C operands ----------------------------------
            #   vY[j,d] = k[j,d] * exp2(gref[d] - gcum[j,d])     (K / Gamma)
            #   vX[c,d] = k[c,d] * exp2(gcum[c,d] - gref[d])     (Gamma . K)
            # Both exponents are bounded by the decay across half a chunk, and
            # their product reconstructs exp2(gcum[c,d] - gcum[j,d]) exactly.
            # All five [C, DK] tile builds in a chunk walk exactly the coordinates
            # tile_iter assigns this thread, and each needs some exponential of the
            # same cumulative decay.  Caching the decay and the two reused
            # exponentials here turns 5 vG reads and 6 exp2 per cell into 1 and 4.
            # Dedicated names: the shared [C, DK] mapping has to survive the
            # tile_iter(C, C) and tile_iter(C, EV) walks that run between these
            # passes, which would otherwise rebind it.
            # the chunk's total log-decay for this channel, needed by the early
            # (K.gamma^C/Gamma)^T build folded into this same pass
            gl_e = fx.Float32(fx.memref_load(vGl, gcol)) if const_expr(EARLY_YT) else fx.Float32(0.0)
            # Per-CHANNEL exponentials, hoisted out of the row loop: one pair of
            # exp2 per thread per chunk instead of two per row.  See DECAY_SHARE.
            if const_expr(DECAY_SHARE):
                gref_e = ex2(gref)  # exp2(gref)       -> recovers exp2(gc)
                dref_e = ex2(gl_e - gref) if const_expr(EARLY_YT) else fx.Float32(0.0)
            for i in range_constexpr(gnrow):
                rr = grow0 + i * grstep
                gc = (
                    fx.Float32(fx.memref_load(gcr, i))
                    if const_expr(REG_G)
                    else fx.Float32(fx.memref_load(vG, (rr, gcol)))
                )
                if const_expr(DECAY_SHARE):
                    # ONE clamped exponent drives the whole cell.  exp2(-u) is
                    # bit-exact against ex2(gref - gc) because the clamp is
                    # symmetric; exp2(gc) and exp2(gl - gc) are strength-reduced
                    # to a multiply by the per-channel constants above.
                    u = clamp2(gc - gref)
                    e_i = ex2c(u)  # exp2(gc - gref)
                    e_v = ex2c(-u)  # exp2(gref - gc)
                    e_g = e_i * gref_e  # exp2(gc)
                else:
                    e_i = ex2(gc - gref)
                    e_v = ex2(gref - gc)
                    e_g = ex2(gc)
                if const_expr(cache_g):
                    fx.memref_store(gc, gcr, i)
                    fx.memref_store(e_g, egr, i)
                    fx.memref_store(e_i, eir, i)
                if const_expr(REGK):
                    kvb = fx.memref_load(kvr, i)
                else:
                    kvb = fx.memref_load(vK, (rr, gcol))
                    if const_expr(cache_kq):
                        fx.memref_store(kvb, kvr, i)
                kv = kvb.to(Float32)
                kvy = kv * e_v
                fx.memref_store(kvy.to(BFloat16), vY, (rr, gcol))
                fx.memref_store((kv * e_i).to(BFloat16), vX, (rr, gcol))
                if const_expr(MERGE_TILES):
                    # Gamma . K (all exponents <= 0) -- same kv and same decay as the
                    # two stores above, so building it here costs one store instead of
                    # a second full pass over k behind two barriers.
                    fx.memref_store((kv * e_g).to(BFloat16), vX2, (rr, gcol))
                if const_expr(EARLY_YT):
                    # (K . gamma^C / Gamma)^T, exponents <= 0.  Same operands as the two
                    # stores above, so it is one extra exp2 + one ds_write here instead
                    # of a whole serial pass (two barriers, a reload of k and of the
                    # decay) at the end of the chunk.  Under DECAY_SHARE it is not even
                    # an exp2: k*exp2(gl-gc) = (k*exp2(gref-gc)) * exp2(gl-gref), and the
                    # left factor is the vY value that was just formed.
                    yt_v = (
                        kvy * dref_e
                        if const_expr(DECAY_SHARE)
                        else kv * ex2(gl_e - gc)
                    )
                    fx.memref_store(yt_v.to(BFloat16), vYt, (gcol, rr))
            gpu.barrier()

            zero(frag_M)
            gemm_lds(vX, vY, frag_M)
            if const_expr(MERGE_TILES):
                # (Gamma.K) S -- its A operand was built in the pass above, so this
                # independent MFMA chain issues straight after the one before it and
                # the two overlap instead of being separated by a build and a barrier.
                zero(frag_Z)
                if const_expr(SHARE_B):
                    gemm_lds_bs(vX2, frag_Z)
                else:
                    gemm_lds(vX2, vStb, frag_Z)
                store_acc_f32(frag_Z, vZ)
            if const_expr(TPRIME_IN_REG):
                # T' = StrictTril(Diag(beta) Akk), straight in the accumulator
                fx.memref_store_vec(
                    fx.memref_load_vec(frag_M)
                    * fx.memref_load_vec(frag_beta)
                    * fx.memref_load_vec(frag_stril),
                    frag_M)
            store_acc_f32(frag_M, vM)
            gpu.barrier()
            # g_cum was fully consumed by the tile build above and two barriers now
            # separate its last reader from this writer, so the queries parked in
            # registers since the top of the chunk can land on top of it.  Placed here
            # rather than right after the first barrier so the eight ds_write_b128 fall
            # into the T' mask (pure VALU on vM) instead of competing with the
            # ds_read_b128 stream of the matmul above: worth ~3 % at EV=64.
            if const_expr(ALIAS and not REGQ):
                store_q()
            if const_expr(debug_stage == "akk"):
                dump(vM, C)

            # ---- T' = StrictTril(Diag(beta) Akk) --------------------------
            # Applied in the accumulator above when MASK_IN_REG.
            if const_expr(not TPRIME_IN_REG):
                col, row0, rstep, nrow = tile_iter(C, C)
                for i in range_constexpr(nrow):
                    rr = row0 + i * rstep
                    a = fx.Float32(fx.memref_load(vM, (rr, col)))
                    b = fx.Float32(fx.memref_load(vBeta, rr))
                    keep = fx.Int32(rr) > fx.Int32(col)
                    fx.memref_store(keep.select(b * a, fx.Float32(0.0)), vM, (rr, col))
            if const_expr(debug_stage == "tprime"):
                gpu.barrier()
                dump(vM, C)

            # ---- Gamma . K  (all exponents <= 0) --------------------------
            # Under MERGE_TILES this tile and its matmul already ran above, fused into
            # the first build pass and the first matmul pair.
            if const_expr(not MERGE_TILES):
                for i in range_constexpr(gnrow):
                    rr = grow0 + i * grstep
                    if const_expr(cache_g):
                        e_g = fx.Float32(fx.memref_load(egr, i))
                    else:
                        e_g = ex2(fx.Float32(fx.memref_load(vG, (rr, gcol))))
                    if const_expr(cache_kq or REGK):
                        kv = fx.memref_load(kvr, i).to(Float32)
                    else:
                        kv = fx.memref_load(vK, (rr, gcol)).to(Float32)
                    fx.memref_store((kv * e_g).to(BFloat16), vX, (rr, gcol))
                gpu.barrier()

                # ---- Z = (Gamma . K) S ; rhs = Diag(beta) (V - Z) ---------
                zero(frag_Z)
                if const_expr(SHARE_B):
                    gemm_lds_bs(vX, frag_Z)
                else:
                    gemm_lds(vX, vStb, frag_Z)
                store_acc_f32(frag_Z, vZ)
                gpu.barrier()

            gv = fx.zipped_divide(fx.slice(gV, (tile, None, sp, None)), fx.make_tile(1, 1))
            if const_expr(VNFUSE):
                # rhs = Diag(beta)(V - Z) and the WY solve's rhs^T -> vVn transpose walk
                # the SAME tile_iter(C, EV) cell -> thread map, and every cell is written
                # and re-read by its own thread, so the fp32 write-back to vZ, the
                # barrier and the whole second pass are pure overhead.  Fused, the rhs
                # never touches LDS in fp32 at all: it is produced in registers and
                # lands directly in the bf16 transposed operand.
                # The row slice per thread is contiguous (see VN_VEC), so the
                # transposed store is one aligned vector write instead of NRV
                # 8-way-bank-conflicting single-element writes.
                if const_expr(VNVEC):
                    col = tid % EV
                    row0 = (tid // EV) * NRV
                    for j in range_constexpr(NRV // VNW):
                        for i in range_constexpr(VNW):
                            rr = row0 + j * VNW + i
                            rv = fx.make_rmem_tensor(1, BFloat16)
                            fx.copy_atom_call(cp16, fx.slice(gv, (None, (rr, col))), rv)
                            vval = fx.memref_load_vec(rv)[0].to(Float32)
                            zval = fx.Float32(fx.memref_load(vZ, (rr, col)))
                            fx.memref_store(
                                (fx.Float32(fx.memref_load(vBeta, rr)) * (vval - zval)).to(BFloat16),
                                vng[j], i)
                        fx.copy_atom_call(
                            vn_atom, vng[j],
                            fx.slice(fVn, (None, (col * PVN + row0) // VNW + j)))
                else:
                    # fusion only, keeping tile_iter's strided rows: the store stays a
                    # scattered 8-way-conflicting ds_write_b16 (A/B control for VN_VEC)
                    col, row0, rstep, nrow = tile_iter(C, EV)
                    for i in range_constexpr(nrow):
                        rr = row0 + i * rstep
                        rv = fx.make_rmem_tensor(1, BFloat16)
                        fx.copy_atom_call(cp16, fx.slice(gv, (None, (rr, col))), rv)
                        vval = fx.memref_load_vec(rv)[0].to(Float32)
                        zval = fx.Float32(fx.memref_load(vZ, (rr, col)))
                        fx.memref_store(
                            (fx.Float32(fx.memref_load(vBeta, rr)) * (vval - zval)).to(BFloat16),
                            vVn, (col, rr))
            else:
                col, row0, rstep, nrow = tile_iter(C, EV)
                for i in range_constexpr(nrow):
                    rr = row0 + i * rstep
                    rv = fx.make_rmem_tensor(1, BFloat16)
                    fx.copy_atom_call(cp16, fx.slice(gv, (None, (rr, col))), rv)
                    vval = fx.memref_load_vec(rv)[0].to(Float32)
                    zval = fx.Float32(fx.memref_load(vZ, (rr, col)))
                    fx.memref_store(fx.Float32(fx.memref_load(vBeta, rr)) * (vval - zval), vZ, (rr, col))
            gpu.barrier()
            if const_expr(debug_stage == "rhs"):
                dump(vZ, EV)

            def gq_tile(lo=0, hi=None):
                """(Gamma . Q) / gamma^ref  (+ Gamma . Q under MERGE_TILES).

                Reads vG / vQ (+ the eir/egr/qvr register caches), writes vX / vX2.
                Disjoint from everything the triangular solve touches, and vX has
                been dead since the merged (Gamma.K)S matmul, so this may be issued
                anywhere from that matmul's barrier up to the tile's own consumer.
                """
                if const_expr(not ablate_gq):
                    if const_expr(specialize):
                        if tid >= GQ_BASE:
                            col, row0, rstep, nrow = tile_iter_sub(
                                C, DK, GQ_BASE, BLOCK - GQ_BASE)
                            gref_s = fx.Float32(fx.memref_load(vG, (CREF, col)))
                            for i in range_constexpr(nrow):
                                rr = row0 + i * rstep
                                gc = fx.Float32(fx.memref_load(vG, (rr, col)))
                                qv = fx.Float32(fx.memref_load(vQ, (rr, col)))
                                fx.memref_store(
                                    (qv * ex2(gc - gref_s)).to(BFloat16), vX, (rr, col))
                    else:
                        _hi = gnrow if hi is None else hi
                        for i in range_constexpr(_hi - lo):
                            i = i + lo
                            rr = grow0 + i * grstep
                            if const_expr(cache_g):
                                e_i = fx.Float32(fx.memref_load(eir, i))
                            else:
                                e_i = ex2(fx.Float32(fx.memref_load(vG, (rr, gcol))) - gref)
                            # this is the first pass that reads q, so it is also where
                            # the cache_kq register copy is taken (q only reaches LDS
                            # after the factored tile build retires the shared arena)
                            if const_expr(REGQ):
                                qvb = fx.memref_load(qvr, i)
                            else:
                                qvb = fx.memref_load(vQ, (rr, gcol))
                                if const_expr(cache_kq):
                                    fx.memref_store(qvb, qvr, i)
                            qvf = qvb.to(Float32)
                            fx.memref_store((qvf * e_i).to(BFloat16), vX, (rr, gcol))
                            if const_expr(MERGE_TILES):
                                # Gamma . Q (all exponents <= 0), from the same q cell
                                # that was just read -- the second q pass disappears.
                                e_g = (
                                    fx.Float32(fx.memref_load(egr, i))
                                    if const_expr(cache_g)
                                    else ex2(fx.Float32(fx.memref_load(vG, (rr, gcol))))
                                )
                                fx.memref_store((qvf * e_g).to(BFloat16), vX2, (rr, gcol))

            # ---- unit lower-triangular solve (I + T') V~ = rhs ------------
            # Solved in place in vZ so the substitution reads fp32 partials.
            if const_expr(ablate_solve):
                # timing-only: skip the substitution, keeping its stores so the
                # rest of the pipeline still runs.  Produces wrong results.
                if tid < EV:
                    e = tid
                    for c in range_constexpr(C):
                        a = fx.Float32(fx.memref_load(vZ, (c, e)))
                        fx.memref_store(a.to(BFloat16), vVn, (e, c))
            elif const_expr(solve == "reg"):
                # A thread owns its whole value channel, so the substitution needs no
                # cross-thread traffic at all: hold the column in registers and the
                # dependent chain reads registers instead of paying LDS latency on
                # every one of the C(C-1)/2 updates.
                if tid < EV:
                    zr = fx.make_rmem_tensor(C, Float32)
                    for c in range_constexpr(C):
                        fx.memref_store(fx.Float32(fx.memref_load(vZ, (c, e_ch))), zr, c)
                    for c in range_constexpr(C):
                        a = fx.Float32(fx.memref_load(zr, c))
                        for j in range_constexpr(c):
                            a = a - fx.Float32(fx.memref_load(vM, (c, j))) * fx.Float32(
                                fx.memref_load(zr, j)
                            )
                        fx.memref_store(a, zr, c)
                        # vZ writeback keeps the "vnew" debug tap meaningful; only
                        # vVn is consumed by the rest of the pipeline.
                        fx.memref_store(a, vZ, (c, e_ch))
                        fx.memref_store(a.to(BFloat16), vVn, (e_ch, c))
            elif const_expr(solve == "wy"):
                # ---- WY / explicit-inverse solve -------------------------
                # T' is STRICTLY lower triangular at C=32, so (I+T')^-1 is exact
                # and unit lower triangular.  Build it in place in vM with a two
                # level 2x2 block recursion
                #     [[A,0],[L,B]]^-1 = [[A^-1,0],[-B^-1 L A^-1, B^-1]]
                # bottoming out at 8x8 diagonal blocks that are inverted in
                # registers, then apply the inverse to the rhs with ONE MFMA.
                # This trades the 4 x 28 barrier-separated dependent FMAs of the
                # blocked substitution (a single wave, EV lanes wide) for a
                # 28-deep chain that runs on all four diagonal blocks at once plus
                # two fully parallel rank-8/rank-16 products and a matrix-core
                # matmul, which the profile shows sitting 97% idle.
                #
                # rhs^T -> vVn (bf16).  This retires the fp32 z_buf, freeing it as
                # the vW scratch and, later, as the bf16 Minv operand (vAq).
                if const_expr(VNFUSE):
                    pass  # already produced directly into vVn by the fused rhs pass
                elif const_expr(VNVEC):
                                    # Same transposed-store bank pathology as yt_s, and here the
                    # backend cannot rescue it: tile_iter hands a thread rows
                    # row0 + i*rstep, so its C/(BLOCK/EV) cells of column `col` are
                    # STRIDED in vVn and stay 8-way-conflicting single-element writes.
                    # Give the thread a CONTIGUOUS row slice instead -- same cells, same
                    # count, every cell still written exactly once -- and the slice
                    # becomes one aligned vector store at the bank floor.
                    col = tid % EV
                    row0 = (tid // EV) * NRV
                    for j in range_constexpr(NRV // VNW):
                        for i in range_constexpr(VNW):
                            fx.memref_store(
                                fx.Float32(
                                    fx.memref_load(vZ, (row0 + j * VNW + i, col))
                                ).to(BFloat16), vng[j], i)
                        fx.copy_atom_call(
                            vn_atom, vng[j],
                            fx.slice(fVn, (None, (col * PVN + row0) // VNW + j)))
                else:
                    col, row0, rstep, nrow = tile_iter(C, EV)
                    for i in range_constexpr(nrow):
                        rr = row0 + i * rstep
                        fx.memref_store(
                            fx.Float32(fx.memref_load(vZ, (rr, col))).to(BFloat16), vVn, (col, rr)
                        )
                gpu.barrier()

                # 1) invert the four 8x8 unit-lower-triangular diagonal blocks.
                # 32 lanes = one wavefront, so program order alone orders the
                # column reads before the column writes; no barrier is needed
                # inside the branch.
                if tid < WYNB * WYB:
                    wb = tid // WYB
                    wc = tid % WYB
                    wr = wb * WYB
                    xr = fx.make_rmem_tensor(WYB, Float32)
                    for i in range_constexpr(WYB):
                        if const_expr(WY_TREE):
                            # balanced tree over the terms that do NOT depend on
                            # x[i-1], then one trailing FMA that does
                            terms = []
                            for j in range_constexpr(max(i - 1, 0)):
                                terms.append(
                                    fx.Float32(fx.memref_load(vM, (wr + i, wr + j)))
                                    * fx.Float32(fx.memref_load(xr, j))
                                )
                            acc = _tree_sum(terms, fx)
                            if const_expr(i >= 1):
                                acc = fma(
                                    fx.Float32(fx.memref_load(vM, (wr + i, wr + i - 1))),
                                    fx.Float32(fx.memref_load(xr, i - 1)),
                                    acc,
                                )
                        else:
                            acc = fx.Float32(0.0)
                            for j in range_constexpr(i):
                                acc = fma(
                                    fx.Float32(fx.memref_load(vM, (wr + i, wr + j))),
                                    fx.Float32(fx.memref_load(xr, j)),
                                    acc,
                                )
                        fx.memref_store(
                            (fx.Int32(i) == wc).select(fx.Float32(1.0), -acc), xr, i
                        )
                    for i in range_constexpr(WYB):
                        fx.memref_store(
                            fx.Float32(fx.memref_load(xr, i)), vM, (wr + i, wr + wc)
                        )
                # 224 of the 256 threads are idle above; give them the entirely
                # independent (Gamma.Q) tile build to chew on in the same barrier
                # region instead of parking them at the next barrier.
                if const_expr(GQ_EARLY):
                    gq_tile(0, GQ_CUT)
                gpu.barrier()

                # The Tril(QK) matmul only needs vX (published at the barrier just
                # above) and vY; nothing in the rest of the solve writes either, and
                # its result lands in an accumulator fragment, not LDS.  Issue it
                # here so the matrix cores run under the VALU-only recursion below.
                if const_expr(QK_EARLY):
                    zero(frag_M)
                    gemm_lds(vX, vY, frag_M)
                if const_expr(GQ_EARLY and GQ_CUT < gnrow):
                    gq_tile(GQ_CUT, gnrow)

                # 2) 8x8 -> 16x16: off-diagonal block <- -A1i (L10 A0i)
                if tid < 2 * WYB * WYB:
                    wp = tid // (WYB * WYB)
                    wi = (tid // WYB) % WYB
                    wj = tid % WYB
                    wa = (2 * wp + 1) * WYB
                    wz = (2 * wp) * WYB
                    acc = fx.Float32(0.0)
                    for k in range_constexpr(WYB):
                        acc = fma(
                            fx.Float32(fx.memref_load(vM, (wa + wi, wz + k))),
                            fx.Float32(fx.memref_load(vM, (wz + k, wz + wj))),
                            acc,
                        )
                    fx.memref_store(acc, vW, (wp * WYB + wi, wj))
                gpu.barrier()
                if tid < 2 * WYB * WYB:
                    wp = tid // (WYB * WYB)
                    wi = (tid // WYB) % WYB
                    wj = tid % WYB
                    wa = (2 * wp + 1) * WYB
                    wz = (2 * wp) * WYB
                    acc = fx.Float32(0.0)
                    for k in range_constexpr(WYB):
                        acc = fma(
                            fx.Float32(fx.memref_load(vM, (wa + wi, wa + k))),
                            fx.Float32(fx.memref_load(vW, (wp * WYB + k, wj))),
                            acc,
                        )
                    fx.memref_store(-acc, vM, (wa + wi, wz + wj))
                gpu.barrier()

                # 3) 16x16 -> 32x32, one cell per thread
                wi = tid // WYH
                wj = tid % WYH
                acc = fx.Float32(0.0)
                for k in range_constexpr(WYH):
                    acc = fma(
                        fx.Float32(fx.memref_load(vM, (WYH + wi, k))),
                        fx.Float32(fx.memref_load(vM, (k, wj))),
                        acc,
                    )
                fx.memref_store(acc, vW, (wi, wj))
                gpu.barrier()
                acc = fx.Float32(0.0)
                for k in range_constexpr(WYH):
                    acc = fma(
                        fx.Float32(fx.memref_load(vM, (WYH + wi, WYH + k))),
                        fx.Float32(fx.memref_load(vW, (k, wj))),
                        acc,
                    )
                fx.memref_store(-acc, vM, (WYH + wi, wj))
                gpu.barrier()

                # 4) bf16 mirror of Minv on top of the (now dead) vW scratch
                col, row0, rstep, nrow = tile_iter(C, C)
                for i in range_constexpr(nrow):
                    rr = row0 + i * rstep
                    fx.memref_store(
                        fx.Float32(fx.memref_load(vM, (rr, col))).to(BFloat16), vAq, (rr, col)
                    )
                gpu.barrier()

                # 5) V~^T = rhs^T (M^-1)^T on the matrix cores, which the profile
                # shows sitting 97% idle.
                solve_mfma()
            elif const_expr(solve == "blocked"):
                # Block the substitution so the dependent chain shrinks from
                # C(C-1)/2 to NB * BS(BS-1)/2 FMAs.  The cross-block coupling is
                # then a rank-BS update of the remaining rows, which is fully
                # parallel over (row, channel) and so runs on every wave -- unlike
                # wave specialization, this trades chain depth for work that keeps
                # all four SIMDs co-issuing.
                for b in range_constexpr(NB):
                    r0 = b * BS
                    if tid < EV:
                        zb = fx.make_rmem_tensor(BS, Float32)
                        for c in range_constexpr(BS):
                            fx.memref_store(
                                fx.Float32(fx.memref_load(vZ, (r0 + c, e_ch))), zb, c
                            )
                        for c in range_constexpr(BS):
                            a = fx.Float32(fx.memref_load(zb, c))
                            for j in range_constexpr(c):
                                a = a - fx.Float32(
                                    fx.memref_load(vM, (r0 + c, r0 + j))
                                ) * fx.Float32(fx.memref_load(zb, j))
                            fx.memref_store(a, zb, c)
                            fx.memref_store(a, vZ, (r0 + c, e_ch))
                            fx.memref_store(a.to(BFloat16), vVn, (e_ch, r0 + c))
                    gpu.barrier()
                    if const_expr(b < NB - 1):
                        # rows below this block absorb its contribution
                        for i in range_constexpr((C - r0 - BS) * EV // BLOCK):
                            n = tid + i * BLOCK
                            rr = r0 + BS + n // EV
                            ee = n % EV
                            acc = fx.Float32(0.0)
                            for j in range_constexpr(BS):
                                acc = acc + fx.Float32(
                                    fx.memref_load(vM, (rr, r0 + j))
                                ) * fx.Float32(fx.memref_load(vZ, (r0 + j, ee)))
                            fx.memref_store(
                                fx.Float32(fx.memref_load(vZ, (rr, ee))) - acc, vZ, (rr, ee)
                            )
                        gpu.barrier()
            elif const_expr(solve == "left"):
                # sequential in rows, parallel over the EV value channels only
                if tid < EV:
                    e = tid
                    for c in range_constexpr(C):
                        a = fx.Float32(fx.memref_load(vZ, (c, e)))
                        for j in range_constexpr(c):
                            a = a - fx.Float32(fx.memref_load(vM, (c, j))) * fx.Float32(
                                fx.memref_load(vZ, (j, e))
                            )
                        fx.memref_store(a, vZ, (c, e))
                        fx.memref_store(a.to(BFloat16), vVn, (e, c))
            else:
                # Right-looking substitution.  Row c is final once every earlier row
                # has subtracted its contribution, so publish it and fan that
                # subtraction out over all (row, channel) cells below it.  The serial
                # depth becomes C barriers instead of the C(C-1)/2 dependent FMAs of
                # the left-looking form, and the trailing update keeps every thread
                # busy rather than EV of them.
                for c in range_constexpr(C):
                    if tid < EV:
                        # vZ[c, :] is final here, and the update below only touches
                        # rows > c, so publishing needs no barrier of its own.
                        fx.memref_store(
                            fx.Float32(fx.memref_load(vZ, (c, tid))).to(BFloat16), vVn, (tid, c)
                        )
                    if const_expr(c < C - 1):
                        zc = fx.Float32(fx.memref_load(vZ, (c, e_ch)))
                        for i in range_constexpr((C - 1 - c + RSTEP - 1) // RSTEP):
                            rr = row_or_sink(c + 1 + r_ch + i * RSTEP, (C - 1 - c) % RSTEP == 0)
                            upd = fx.Float32(fx.memref_load(vZ, (rr, e_ch))) - fx.Float32(
                                fx.memref_load(vM, (rr, c))
                            ) * zc
                            fx.memref_store(upd, vZ, (rr, e_ch))
                    gpu.barrier()

            # ---- (Gamma . Q) / gamma^ref ----------------------------------
            # Deliberately not separated from the solve above by a barrier: the solve
            # touches only vM / vZ / vVn and this only vG / vQ / vX, with vX free
            # since the (Gamma.K) S matmul consumed it, so the waves that hold no
            # active solve lane can start here immediately.  That overlap is partial
            # -- the solving wave still does its share of this tile -- but dropping
            # the barrier is free.  See ``specialize`` for why handing the tile to
            # the idle waves outright is a pessimization.
            if const_expr(not GQ_EARLY):
                gq_tile()
            gpu.barrier()
            if const_expr(debug_stage == "vnew"):
                dump(vZ, EV)
                gpu.barrier()

            # ---- Tril((Gamma . Q)(K / Gamma)^T) --------------------------

            if const_expr(not QK_EARLY):
                zero(frag_M)
                gemm_lds(vX, vY, frag_M)
            if const_expr(MASK_IN_REG):
                # Tril(Aqk) -> bf16 without ever touching the fp32 vM tile: the mask is
                # a register multiply and the bf16 result goes straight to vAq, which
                # removes an fp32 store pass, a barrier and an fp32 read pass.
                store_aq()
            else:
                store_acc_f32(frag_M, vM)
                gpu.barrier()
                if const_expr(debug_stage == "aqk_raw"):
                    dump(vM, C)

                col, row0, rstep, nrow = tile_iter(C, C)
                for i in range_constexpr(nrow):
                    rr = row0 + i * rstep
                    a = fx.Float32(fx.memref_load(vM, (rr, col)))
                    keep = fx.Int32(rr) >= fx.Int32(col)
                    fx.memref_store(keep.select(a, fx.Float32(0.0)).to(BFloat16), vAq, (rr, col))
            if const_expr(debug_stage == "aqk"):
                gpu.barrier()
                dump(vAq, C, is_bf16=True)

            # ---- Gamma . Q  (all exponents <= 0) --------------------------
            # Under MERGE_TILES this tile was built alongside (Gamma.Q)/gamma^ref above.
            if const_expr(not MERGE_TILES):
                if const_expr(not ablate_gq):
                    for i in range_constexpr(gnrow):
                        rr = grow0 + i * grstep
                        if const_expr(cache_g):
                            e_g = fx.Float32(fx.memref_load(egr, i))
                        else:
                            e_g = ex2(fx.Float32(fx.memref_load(vG, (rr, gcol))))
                        if const_expr(cache_kq or REGQ):
                            qv = fx.memref_load(qvr, i).to(Float32)
                        else:
                            qv = fx.memref_load(vQ, (rr, gcol)).to(Float32)
                        fx.memref_store((qv * e_g).to(BFloat16), vX, (rr, gcol))
            gpu.barrier()

            # ---- O = (Gamma . Q) S + Tril(...) V~ -------------------------
            if const_expr(debug_stage in ("o_state", "o_attn")):
                # dump one output term in isolation; clobbers vZ/vAq, which the
                # state update does not read (it uses vVn and vYt)
                zero(frag_O)
                if const_expr(debug_stage == "o_state"):
                    gemm_lds(vX, vStb, frag_O)
                else:
                    gemm_lds(vAq, vVn, frag_O)
                gpu.barrier()
                store_acc_f32(frag_O, vZ)
                gpu.barrier()
                dump(vZ, EV)
                gpu.barrier()
            zero(frag_O)
            if const_expr(SHARE_B):
                gemm_lds_bs(vX2 if const_expr(MERGE_TILES) else vX, frag_O)
            else:
                gemm_lds(vX2 if const_expr(MERGE_TILES) else vX, vStb, frag_O)
            gemm_lds(vAq, vVn, frag_O)
            store_output(fx.slice(gO, (tile, None, sp, None)), q_scale)

            # ---- (K . gamma^C / Gamma)^T, exponents <= 0 -----------------
            # Under EARLY_YT this tile was already built at the top of the chunk into
            # its own buffer, so the two barriers that bracketed this pass are gone:
            # nothing between that build and here writes vYt, and the state matmul
            # below only reads it.
            if const_expr(not EARLY_YT):
                gpu.barrier()
                gl = fx.Float32(fx.memref_load(vGl, gcol))
                for i in range_constexpr(gnrow):
                    rr = grow0 + i * grstep
                    if const_expr(cache_g):
                        gc = fx.Float32(fx.memref_load(gcr, i))
                    else:
                        gc = fx.Float32(fx.memref_load(vG, (rr, gcol)))
                    if const_expr(cache_kq or REGK):
                        kv = fx.memref_load(kvr, i).to(Float32)
                    else:
                        kv = fx.memref_load(vK, (rr, gcol)).to(Float32)
                    fx.memref_store((kv * ex2(gl - gc)).to(BFloat16), vYt, (gcol, rr))
                gpu.barrier()

            # ---- S^T <- Diag(gamma^C) S^T + V~^T (K . gamma^C/Gamma) -----
            apply_chunk_decay()
            # ST_TRANSPOSE: S += (K.gamma^C/Gamma)^T V~ instead of S^T += V~^T (K...),
            # i.e. the same product with A and B swapped -- identical MFMA count, but
            # the accumulator comes out [DK, EV] so publish_state can store b64.
            if const_expr(ST_TRANSPOSE):
                gemm_lds(vYt, vVn, frag_S)
            else:
                gemm_lds(vVn, vYt, frag_S)

            # PRE-PUBLISH FENCE.  This barrier exists for exactly one hazard: the WAR
            # between the tail (Gamma.Q) S matmul's LDS read of vStb and publish_state's
            # overwrite of the same bytes.  Under the 2x2 wave TiledMma the reader of
            # EV-half wn is BOTH waves (0,wn) and (1,wn) while the writer of
            # DK-half wm x EV-half wn is wave (wm,wn), so the hazard is genuinely
            # cross-wave and the fence is mandatory -- WHENEVER THAT LDS READ EXISTS.
            # With SHARE_B on (EV<=32, i.e. DV_SPLIT=4) it does NOT exist: the sole LDS
            # reader of vStb in the whole chunk is fetch_state_b() at the chunk head,
            # ~18 unconditional barriers upstream, so this barrier fences nothing.
            # Everything else the tail touches is disjoint from st_b (st_b is aliased
            # with nothing -- see the SharedStorage layout), and the next chunk's first
            # LDS write of any tile the tail reads (vYt, vVn) sits behind the chunk-head
            # barrier, which all four waves must still reach.  See report for the full
            # access-set argument.
            if const_expr(not SHARE_B):
                gpu.barrier()
            publish_state()
            # No trailing barrier: the next chunk's first reader of vStb is the
            # (Gamma.K) S matmul, which sits behind three barriers (post-load,
            # post-scan, post-tile-build), and after the last chunk the final state is
            # written from frag_S, not from the mirror.
            if const_expr(not DROP_PUBLISH_BARRIER):
                gpu.barrier()

        if const_expr(store_final_state and debug_stage == ""):
            # arg_ht is a contiguous [BH, DV_SPLIT, EV, DK] fp32 buffer.  With
            # ST_TRANSPOSE the accumulator is [DK, EV], so take a transposed *view* of
            # the same bytes -- no extra traffic, and the host-side layout is unchanged.
            if const_expr(ST_TRANSPOSE):
                gHt = fx.rocdl.make_buffer_tensor(fx.make_view(
                    fx.get_iter(arg_ht),
                    fx.make_layout((BH, DV_SPLIT, DK, EV),
                                   (DV_SPLIT * EV * DK, EV * DK, 1, DK))))
            else:
                gHt = fx.rocdl.make_buffer_tensor(arg_ht)
            thr_c32g = fx.make_tiled_copy_C(cp32, tmma).get_slice(tid)
            fx.copy(cp32, thr_c32g.retile(frag_S), thr_c32g.partition_S(fx.slice(gHt, (bh, sp, None, None))))

    @flyc.jit
    def launch_kda_fwd(
        arg_q: fx.Tensor,
        arg_k: fx.Tensor,
        arg_g: fx.Tensor,
        arg_beta: fx.Tensor,
        arg_v: fx.Tensor,
        arg_o: fx.Tensor,
        arg_h0: fx.Tensor,
        arg_ht: fx.Tensor,
        n_chunks: fx.Int32,
        q_scale: fx.Float32,
        stream: fx.Stream = fx.Stream(None),
    ):
        kda_fwd_kernel(
            arg_q, arg_k, arg_g, arg_beta, arg_v, arg_o, arg_h0, arg_ht, n_chunks, q_scale
        ).launch(grid=(BH * DV_SPLIT, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    launch_kda_fwd.lds_bytes = lds_bytes
    launch_kda_fwd.config = dict(BH=BH, T=T, DK=DK, DV=DV, C=C, DV_SPLIT=DV_SPLIT, BLOCK=BLOCK, EV=EV, NC=NC)
    return launch_kda_fwd
