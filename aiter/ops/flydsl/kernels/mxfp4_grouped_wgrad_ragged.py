# SPDX-License-Identifier: MIT
#
# RAGGED MXFP4 WGRAD grouped GEMM, gfx950 (CDNA4).
#
#     grad_W[g] = go[group_g]^T @ ia[group_g]
#
# Here the groups partition the CONTRACTION, not the output rows -- the opposite
# of the sibling ``mxfp4_grouped_gemm_ragged``, and the reason this body needs
# its own ragged machinery rather than reusing that one's.
#
# The pipeline is the tuned equal-groups shape -- 4-wave 2x2 geometry, AGPR-pinned
# scaled MFMA, 8-buffer LDS ping-pong with a depth-2 K pipeline, cooperative scale
# load + ds_bpermute + a one-body scale prefetch riding the scf.for state -- driven
# from device-resident ragged group sizes.
#
# Why the tuned body and not a simpler ragged one (E=8, N=1408, K=2048, 24576
# tok/expert, even groups so both see identical work): a non-pipelined ragged body
# runs at 630 TF/s (13.6% of peak) against this pipeline's 1349 TF/s (29.2%). The
# gap is the body, not the raggedness.
#
# ── The three things that made that pipeline equal-groups, and how each is paid ──
#
# 1. CONSTEXPR CONTRACTION LENGTH. The equal-groups form keys its compile on M_G
#    and derives
#    K_ITERS = M_G/128 as a Python int, which drives the prologue/rolled-loop/tail
#    split and the constexpr peel of the odd remainder.
#
#    Here the group length is a device value. Two moves make the same schedule
#    work with a runtime trip count:
#      * The window is EXTENDED, never shortened, to an even number of K-steps
#        >= 4 (`K_eff`). MAIN = K_eff - 2 is then even, so `N_ROLLED == MAIN`
#        and the constexpr odd-step peel disappears entirely -- one fewer
#        moving part than the equal-groups form, not one more.
#      * Everything outside the group is masked (see 3), so extending is free of
#        correctness consequences and costs at most 3 dead K-steps per block.
#
# 2. 128-ALIGNED GROUP STARTS. The cooperative scale load is a dwordx2 whose
#    element index is (half_base + lane)*SCALE_I32_ROW + m_start_i32 + k. It needs
#    that index EVEN, and it needs the lane's e8m0 to sit at byte k_grp of the
#    loaded i32 -- both of which hold only when the group starts on a 128 (really
#    256) boundary. Ragged group boundaries are only 32-aligned, which is why a
#    naive ragged body carries per-lane `a_e8`/`k_carry` index arithmetic and
#    cannot use the wide cooperative load at all.
#
#    Rather than widen the load (3 distinct i32 per unrolled body once a carry is
#    possible => dwordx4 and +8 live VGPRs against 12 of headroom), the window
#    BASE is rounded DOWN to 256: `lo0 = (start/256)*256`. Then m_start_i32 is
#    even and the per-lane byte is k_grp again, so **the entire equal-groups
#    scale path is reused verbatim** -- cooperative load, ds_bpermute, the
#    one-body prefetch and its vmcnt accounting. The up-to-224 leading elements
#    that rounding pulls in belong to the previous group and are masked off by
#    (3). Requires SCALE_I32_ROW even, i.e. M_ROW % 256 == 0 (see wgrad_ragged).
#
# 3. NO PARTIAL K-STEPS. An equal-groups body contracts exactly M_G elements,
#    all of them real.
#
#    Here a block's window [lo0, lo0 + K_eff*128) both starts before the group
#    (leading round-down) and ends after it (trailing partial + the extension
#    from 1). Both, plus empty groups, plus the dead extension steps, are handled
#    by ONE predicate on the A-operand scale:
#
#        e  = k*128 + k_grp*32          (this lane's 32-element sub-block)
#        ok = (e >= head) & (e < span)
#
#    where `head = start - lo0` and `span = end - lo0` are block-uniform. Forcing
#    the A e8m0 to 0x00 (= 2**-127, NOT NaN) makes a*scale*b underflow to ~1e-33
#    against accumulators of order 1e2. Masking rides the SCALE and not the MFMA
#    fragment because the scale path maps k_grp -> sub-block directly, while the
#    fragment's K layout is opaque and not lane//16.
#
#    Applied to A only: a == 0 kills the product for any FINITE b. That "finite"
#    is load-bearing and is why both scale descriptors are bounded below -- an
#    out-of-range e8m0 read of 0xFF is NaN, and 0 * NaN = NaN would survive the
#    mask. `max_size=False` ALONE DOES NOT BOUND a dynamic memref; it silently
#    falls back to 0xFFFFFFFF. Verify in the generated IR, not in this source.
#
# ── What this grid choice gains over a chunked ragged schedule ──
#
# The grid is E * n_r_tiles * n_c_tiles: exactly one block per (group, output
# tile), each walking its whole group. There is no split of a group across
# chunks, and therefore **no atomic epilogue** -- the block owns its output tile
# outright and stores it directly onto an uninitialised buffer, as the
# equal-groups form does. That also makes this body bit-reproducible at a fixed
# stride, which a chunked schedule is not, and removes the zero-init memset of
# the (E, R, C) f32 output.
#
# The cost of that choice is load balance: 384 blocks over 256 CUs means a skewed
# group is a straggler, where a chunked schedule would spread the work over
# ~5000 blocks. Equal groups accept the same 384-block grid.
#
# ── What the straggler actually costs, measured (2026-07-31) ──
#
# It is a SCHEDULING problem, not a chunking one, at the skew that occurs:
#
#   * The block -> group map IS the dispatch order (the hardware issues
#     workgroups in increasing block id). Ordering the groups longest-first --
#     `slot_to_group` below, computed on device from the offsets every block
#     already reads -- is worth a MEDIAN +10.8% (range +4.5%..+22.9%) over seven
#     group-size lists taken from real routed DSV3-16B steps. No atomics, no
#     extra launch, no host sync.
#   * With the remap on, the ragged path at those real sizes runs at 27.2-29.5%
#     of peak and is FASTER than equal groups at the same total work (0.88-0.96x)
#     -- so at real skew there is no straggler deficit left to chunk away.
#   * A synthetic ragged draw (flat Dirichlet split) has max/mean K_eff of
#     2.4-3.6. Real routing lands at 1.17-1.69. In the synthetic regime one group
#     IS the critical path, reordering can only reach +1.2% median, and chunking
#     would be the only remaining lever -- but that regime is an artifact of the
#     draw, not something routing produces.
#
# So a chunked variant that re-introduces the atomic epilogue is NOT worth
# building for this workload.
#
# NAMING / stride split:
#     A := go_t (R, M_ROW)   B := ia_t (C, M_ROW)   out[g] = A[:,s:e] @ B[:,s:e]^T
# M is the contraction (tokens); M_ROW is the HBM row stride and is a RUNTIME
# argument, never a constexpr: under real routing the routed token count changes
# every step, and a constexpr stride keys a fresh JIT compile -- and retains a
# fresh GPU module -- per step per layer. The compile key is (R, C, E) only.

# NOTE: no `from __future__ import annotations` -- fx.struct Storable needs real
# types at class-definition time.

import functools
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from . import buffer_ops

# These six are the FlyDSL fp8-GEMM helper set, already vendored into aiter by
# the 8-wave CDNA4 GEMM. They address BYTES, not elements, which is exactly why
# an fp4 kernel can reuse them unchanged -- see the LDS K-step note in the module
# docstring above.
from .gemm_a8w8_8wave import (
    G2SLoader,
    S2RLoader,
    compute_global_swizzle,
    make_fp8_buffer_tensor,
    swizzle_128,
    wait_barrier,
)

SCALE_BLOCK = 32

BLOCK_R = 256  # output rows per tile
BLOCK_C = 256  # output cols per tile
# One pipeline step stages 128 BYTES of the contraction per operand row -- the
# same LDS geometry the MXFP8 body uses. At 2 fp4 per byte that is 256 tokens,
# i.e. TWO K=128 MFMA sub-blocks (ksub 0/1). See the fp4 note in the header.
BLOCK_K_BYTES = 128
FP4_PER_BYTE = 2
BLOCK_K_ELEMS = BLOCK_K_BYTES * FP4_PER_BYTE  # 256 tokens
KSUBS = 2
BLOCK_M = 128  # elements covered by ONE scale i32 / one MFMA K-sub-block
SCALE_BLOCK = 32
# Window base rounding. Unchanged at 256: it is set by the scale index needing
# to be even (lo0/128 even), which is the same constraint at fp4, and 256 is
# also exactly one fp4 K-step so the step addressing lands on a row boundary.
WINDOW_ALIGN = 256
LPT_MAX_E = 32  # above this the dispatch remap falls back to `id`

# Block -> (group, bounds) mappings. Only "lpt" ships; the other two are here to
# be timed against it, because the difference between them splits `lpt`'s gain
# into a scheduling part and a codegen part. See slot_to_group.
#   id     -- group = dispatch slot, offsets read at a COMPUTED index (original)
#   idsel  -- group = dispatch slot, offsets read at constant indices
#   lpt    -- slot d takes the d-th longest group, constant-index reads
ORDER_MODES = ("id", "idsel", "lpt")


# Interleave each quadrant's 16 MFMAs with the g2s/s2r loads of the NEXT
# fragment so they co-issue in the MFMA execute shadow. Worth +6.6% here
# (and +5.1% on the sibling forward kernel) -- but only AFTER the scale
# delivery came off the L1 critical path; it measured neutral before that.
_INTERLEAVE = os.environ.get("AITER_FLYDSL_MXFP4_WGRAD_INTERLEAVE", "1") != "0"


def _mfma_scale_agpr(a, b, sa, sb, acc):
    """fp4 16x16x128 scaled MFMA, accumulator pinned in AGPR (=a,...,0) so the
    f32x4 acc accumulates in place and the compiler does not shuffle it
    between AGPR slots.

    a/b are i32x4 -- 16 bytes = 32 fp4 per lane = the K=128 operand. The operand
    format field cbsz/blgp (0 = fp8 e4m3, 4 = fp4 e2m1) is what selects the
    element width; the opcode is the same either way.

    sa/sb are broadcast-i32 E8M0 scales (one e8m0 replicated to 4 bytes), so
    op_sel is a don't-care and is left at its default -- which is why this kernel
    needs neither preshuffled scales nor op_sel bookkeeping."""
    asm = "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, $0, $3, $4 cbsz:4 blgp:4"
    return _llvm.inline_asm(
        Vec.make_type(4, fx.Float32),
        [
            arith._to_raw(a),
            arith._to_raw(b),
            arith._to_raw(sa),
            arith._to_raw(sb),
            arith._to_raw(acc),
        ],
        asm,
        "=a,v,v,v,v,0",
        has_side_effects=True,
    )


def _compile(R: int, C: int, E: int, sc_pair: bool, order: str):
    """Build the launch fn. Compile-time: (R, C, E, sc_pair, order) ONLY.

    Deliberately absent from the key: the row stride, the group sizes, the
    contraction length and the number of K iterations -- all runtime. Under
    real routing every one of those varies per step, and anything shape-keyed
    here becomes a JIT compile plus a retained GPU module per step per layer.

    `order` picks the block->group mapping (see slot_to_group). Like
    `sc_pair` it is a key, not a runtime branch, because it decides which code
    exists at all. Only the default is ever compiled in production; the other
    two exist to be timed against it in the same pass.
    """
    assert order in ORDER_MODES, f"unknown order {order!r}"
    ORDER = order
    N_TILES_A = BLOCK_R // 4 // 16
    N_TILES_B = BLOCK_C // 4 // 16
    N_ACCUMS = N_TILES_A * N_TILES_B
    N_LDS_ROUNDS = max(N_TILES_A, N_TILES_B)

    # The cooperative dwordx2 scale load needs its element index even. The
    # index is (half_base + lane)*SCALE_I32_ROW + m_start_i32 + k, with k
    # always even (chunks start on the unrolled body's first K-step), and
    # m_start_i32 even because the window base is WINDOW_ALIGN(=256)-aligned.
    # `lane` varies, so evenness reduces to SCALE_I32_ROW = M_ROW/128 being
    # even, i.e. M_ROW % 256 == 0.
    #
    # When it is odd the pair load would be misaligned for odd lanes, so fall
    # back to TWO dword loads -- same bytes and same cache lines, one extra
    # instruction per half. Requiring 256 instead would be an integration
    # trap: the MoE token dispatcher pads per-expert counts to 32, so M_ROW
    # is a multiple of 32 and only 1 in 8 such values is a multiple of 256.
    # 128 is the alignment the sibling even-groups body already contracts for.
    #
    # `sc_pair` is a compile-key bit rather than a runtime branch: it changes
    # the vmcnt barrier constants below, which must be compile-time. It adds
    # at most ONE extra variant, so the compile set stays bounded -- the
    # property that keeping M_ROW out of the key exists to protect.
    SC_PAIR = sc_pair
    N_SC_OPS = 4 if SC_PAIR else 8  # scale vm ops per unrolled body
    N_CHUNK = 8  # raw i32 carried per chunk (4 halves x 2 K-steps)

    # ── vmcnt accounting, unchanged from the equal-groups body ──
    # An unrolled body is 2 K-steps and issues, between barriers:
    #   R1a = N_SC_OPS scale loads (next chunk) + (NA+NB) g2s copies
    #   R2a = (NA+NB) g2s copies
    #   R1b = (NA+NB) g2s copies      R2b = (NA+NB) g2s copies
    # Barrier 1 of a K-step must retire that step's minus-2 R2; barrier 2 the
    # minus-1 R1. SAFE UNDER REORDERING within a region: the only region
    # holding more than one thing is R1a, and if the scheduler sinks the scale
    # loads below the g2s copies there, the barrier retires more than
    # required, never less. A count that is too LARGE under-waits and reads
    # unfilled LDS -- and that failure is silent.
    # fp4 re-derivation: an fp4 K-step needs a whole chunk (its two scale
    # i32 are ksub 0 and 1), so BOTH half-bodies now issue one, where the
    # MXFP8 body's single chunk covered both of its steps. R1b gains
    # N_SC_OPS, which lifts W1A and W2B by the same amount and leaves W2A
    # and W1B -- already counting a chunk -- unchanged. All four coincide.
    W1A = 2 * (N_TILES_A + N_TILES_B) + N_SC_OPS
    W2A = 2 * (N_TILES_A + N_TILES_B) + N_SC_OPS
    W1B = 2 * (N_TILES_A + N_TILES_B) + N_SC_OPS
    W2B = 2 * (N_TILES_A + N_TILES_B) + N_SC_OPS

    LDS_BLOCK_R = BLOCK_R // 2  # 128: each wave-dim half owns 128 rows
    LDS_BLOCK_C = BLOCK_C // 2

    def _interleave_plan(n_mfma, n_g2s, n_tiles):
        """Where to slot each load into the cluster's MFMA stream.

        Plain Python, run before tracing: FlyDSL's AST rewriter turns an
        `if` inside a nested kernel-body function into a separate `__then_N`
        that cannot see the enclosing closure, so schedule logic cannot live
        in the body. The body then walks the plan with no branches at all.
        """
        acts = []
        for r in range(max(n_g2s, n_tiles)):
            if r < n_g2s:
                acts.append(("g", r))
            if r < n_tiles:
                acts.append(("s0", r))
                acts.append(("s1", r))
        g_at = [[] for _ in range(n_mfma)]
        s0_at = [[] for _ in range(n_mfma)]
        s1_at = [[] for _ in range(n_mfma)]
        for n_, act in enumerate(acts):
            pos = int(round((n_ + 1) * n_mfma / (len(acts) + 1)))
            {"g": g_at, "s0": s0_at, "s1": s1_at}[act[0]][pos].append(act[1])
        return g_at, s0_at, s1_at

    # (tile_i, tile_j, ksub); ksub outermost so an accumulator's two MFMAs
    # are not back-to-back on the same AGPR.
    MFMA_SEQ = [
        (i, j, ks)
        for ks in range(KSUBS)
        for i in range(N_TILES_A)
        for j in range(N_TILES_B)
    ]
    PLANS = {
        (N_TILES_A, N_TILES_B): _interleave_plan(len(MFMA_SEQ), N_TILES_A, N_TILES_B),
        (N_TILES_B, N_TILES_A): _interleave_plan(len(MFMA_SEQ), N_TILES_B, N_TILES_A),
    }

    a_lds_size = LDS_BLOCK_R * BLOCK_K_BYTES
    b_lds_size = LDS_BLOCK_C * BLOCK_K_BYTES

    def slot_to_group(offs_rsrc, slot):
        """Dispatch slot -> (group, m_start, m_end), read on device.

        Defined OUTSIDE the kernel body on purpose: FlyDSL's AST rewriter
        rewrites the kernel function only, and it turns a Python `if` into an
        scf.if region (assignments inside do not escape) and a Python `range`
        into a dynamic scf.for (the induction variable stops being an int and
        cannot index a Python list). Out here both are ordinary Python again,
        so `LPT` is a real compile-time branch and the loops are unrolled by
        construction.

        ── LPT (longest processing time first) ──
        With the identity mapping `g = slot`, the group order IS the dispatch
        order, because the hardware issues workgroups in increasing block id.
        A large group sitting at a high `g` therefore starts in the last wave
        and sets the makespan by itself. Here slot d takes the d-th LONGEST
        group instead.

        MEASURED (E=8, N=1408, K=2048, 24576 tok/expert, 4 draws), sizes
        sorted descending vs ascending -- same multiset, so identical sum AND
        max K_eff, i.e. pure schedule: largest-first wins by 3.0-21.8%.

        Every block recomputes the whole permutation from the E offsets it
        already reads: E scalar loads and E*E uniform compares, tens of SALU
        ops against a 200+ iteration K-walk. No host sort, no second launch,
        no device scratch, and `offs` stays the tensor the caller passed.

        The sort key is `end - floor(start/256)*256`, the window the block
        will actually walk (header note 2), not the group size -- the
        round-down means two equal-sized groups can differ by a K-step, and
        the walk is what costs. Ties break on index, so the ranks are a
        permutation of 0..E-1 for any input, empty groups included.

        ── Two effects, not one ──
        `lpt` gains ~5% even on EQUAL groups, where its ranking provably
        returns the identity permutation, so part of the win is not the
        schedule at all. Mode `idsel` isolates it: identity mapping, but the
        offsets read the way `lpt` reads them. See ORDER_MODES.
        """
        if ORDER == "id":
            # The original mapping. Both loads are at a COMPUTED index, and
            # the g-1 one has to wait on a select, so a block cannot know its
            # window until a dependent scalar load returns. Bounded, not
            # max_size: the g==0 read is clamped to index 0 and then discarded
            # by the select, so it must not touch memory beyond OFFS.
            g = slot
            g_prev = arith.select(g > fx.Int32(0), g - fx.Int32(1), fx.Int32(0))
            prev_raw = buffer_ops.buffer_load(
                offs_rsrc, g_prev, vec_width=1, dtype=T.i32, is_scalar=True
            )
            end_raw = buffer_ops.buffer_load(
                offs_rsrc, g, vec_width=1, dtype=T.i32, is_scalar=True
            )
            m_start = arith.select(g > fx.Int32(0), ArithValue(prev_raw), fx.Int32(0))
            return g, ArithValue(m_start), ArithValue(end_raw)

        # All E offsets at CONSTANT indices: no address depends on anything,
        # so they issue together at block entry instead of serialising behind
        # a computed index. E=8 int32 is one 32-byte line either way.
        ends = [
            ArithValue(
                buffer_ops.buffer_load(
                    offs_rsrc, fx.Int32(i), vec_width=1, dtype=T.i32, is_scalar=True
                )
            )
            for i in range(E)
        ]
        starts = [ArithValue(fx.Int32(0))] + ends[:-1]

        if ORDER == "idsel":
            # Identity mapping, `lpt`'s load pattern: the control for how much
            # of `lpt`'s gain is scheduling and how much is just this.
            g = slot
            m_start = ArithValue(fx.Int32(0))
            m_end = ends[0]
            for i in range(E):
                mine = g == fx.Int32(i)
                m_start = arith.select(mine, starts[i], m_start)
                m_end = arith.select(mine, ends[i], m_end)
            return g, ArithValue(m_start), ArithValue(m_end)

        cost = [
            ends[i] - (starts[i] // fx.Int32(WINDOW_ALIGN)) * fx.Int32(WINDOW_ALIGN)
            for i in range(E)
        ]

        g = ArithValue(fx.Int32(0))
        m_start = ArithValue(fx.Int32(0))
        m_end = ends[0]
        for i in range(E):
            rank = ArithValue(fx.Int32(0))
            for j in range(E):
                # j < i wins ties, so the order is strict and total; j == i
                # takes the `>` branch and contributes 0.
                ahead = (cost[j] >= cost[i]) if j < i else (cost[j] > cost[i])
                rank = rank + arith.select(ahead, fx.Int32(1), fx.Int32(0))
            mine = rank == slot
            g = arith.select(mine, fx.Int32(i), g)
            m_start = arith.select(mine, starts[i], m_start)
            m_end = arith.select(mine, ends[i], m_end)
        return ArithValue(g), ArithValue(m_start), ArithValue(m_end)

    @fx.struct
    class SharedStorage:
        A_lds_cur_0: fx.Array[fx.Int8, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.Int8, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.Int8, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.Int8, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.Int8, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.Int8, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.Int8, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.Int8, b_lds_size, 16]

    @flyc.kernel(known_block_size=[256, 1, 1])
    def kernel_wgrad(
        A: fx.Tensor,
        B: fx.Tensor,
        OUT: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        OFFS: fx.Tensor,
        n_r_tiles: fx.Int32,
        n_c_tiles: fx.Int32,
        out_r: fx.Int32,
        out_c: fx.Int32,
        m_row: fx.Int32,
    ):
        I8_IR_t = fx.Int8.ir_type

        # Runtime row stride. One e8m0 per 32 contracting elements, 4 packed
        # per i32, so a row is m_row/128 i32 and m_row/32 bytes.
        sc_i32_row = ArithValue(m_row) // fx.Int32(BLOCK_M)
        sc_e8_row = ArithValue(m_row) // fx.Int32(SCALE_BLOCK)

        # ── Block -> (group, output tile). One block per (g, tile). ──
        bid = fx.block_idx.x
        tiles_per_g = n_r_tiles * n_c_tiles
        slot = ArithValue(bid) // tiles_per_g
        ttile = ArithValue(bid) % tiles_per_g
        r_base = (ttile // n_c_tiles) * fx.Int32(BLOCK_R)
        c_base = (ttile % n_c_tiles) * fx.Int32(BLOCK_C)

        # ── Group bounds, read ON DEVICE (uniform s.buffer.load, no sync) ──
        offs_rsrc = buffer_ops.create_buffer_resource(
            OFFS, max_size=False, num_records_bytes=E * 4
        )

        g, m_start, m_end = slot_to_group(offs_rsrc, slot)

        # ── The window: base rounded DOWN to 256, length rounded UP to an
        # even number of K-steps >= 4. See notes 1-3 in the file header. ──
        lo0 = (m_start // fx.Int32(WINDOW_ALIGN)) * fx.Int32(WINDOW_ALIGN)
        head = m_start - lo0  # 0..224, multiple of 32
        span = m_end - lo0  # real elements from the window base
        k_need = (span + fx.Int32(BLOCK_K_ELEMS - 1)) // fx.Int32(BLOCK_K_ELEMS)
        k_even = ((k_need + fx.Int32(1)) // fx.Int32(2)) * fx.Int32(2)
        K_eff = arith.select(k_even < fx.Int32(4), fx.Int32(4), k_even)
        N_ROLLED = K_eff - fx.Int32(2)  # even, >= 2: no constexpr peel needed
        # Scale-i32 index of the window base. Still lo0/128 -- an i32 is 4
        # e8m0 = 128 elements whatever the element format -- and still even,
        # because lo0 is 256-aligned. Our K-step spans i32 (2s, 2s+1).
        m_start_i32 = lo0 // fx.Int32(BLOCK_M)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_cur0, a_cur1 = lds.A_lds_cur_0, lds.A_lds_cur_1
        a_next0, a_next1 = lds.A_lds_next_0, lds.A_lds_next_1
        b_cur0, b_cur1 = lds.B_lds_cur_0, lds.B_lds_cur_1
        b_next0, b_next1 = lds.B_lds_next_0, lds.B_lds_next_1

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_i = wave_id // 2
        wave_j = wave_id % 2

        m_lane = lane_id % fx.Int32(16)
        k_grp = lane_id // fx.Int32(16)
        kshift = k_grp * fx.Int32(8)
        kgrp32 = k_grp * fx.Int32(SCALE_BLOCK)  # this lane's sub-block offset
        mask_ff = fx.Int32(0xFF)
        bcast = fx.Int32(0x01010101)
        zero_i32 = fx.Int32(0)

        # Contraction (K-walk) global offsets, in BYTES: an fp4 operand row
        # is m_row/2 bytes and the window base lo0 sits at lo0/2. lo0 is
        # 256-aligned so lo0/2 is a whole 128-byte LDS row.
        m_row_bytes = ArithValue(m_row) // fx.Int32(FP4_PER_BYTE)
        lo0_bytes = lo0 // fx.Int32(FP4_PER_BYTE)
        A0_gl = r_base * m_row_bytes + lo0_bytes
        A1_gl = (r_base + fx.Int32(LDS_BLOCK_R)) * m_row_bytes + lo0_bytes
        B0_gl = c_base * m_row_bytes + lo0_bytes
        B1_gl = (c_base + fx.Int32(LDS_BLOCK_C)) * m_row_bytes + lo0_bytes

        gA = make_fp8_buffer_tensor(A, I8_IR_t)
        gB = make_fp8_buffer_tensor(B, I8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        # BOUNDED scale descriptors. Two independent reasons, both real:
        #   * the cooperative load has lane L take row (half_base + L), so the
        #     last row tile of an operand whose R (or C) is not a multiple of
        #     256 addresses rows that do not exist -- at R=1408 = 5.5 tiles,
        #     128 of the last tile's 256 rows;
        #   * this kernel's window runs PAST the group and past the operand's
        #     last row on the final K-steps by construction.
        # Both must read 0, not memory: an 0xFF byte is e8m0 NaN and the A-side
        # mask does not neutralise it (0 * NaN = NaN). `max_size=False` alone
        # does NOT do this on a dynamic memref -- it silently yields
        # 0xFFFFFFFF. (The data descriptors get a real bound because
        # make_fp8_buffer_tensor emits a runtime fly.cosize instead.)
        sa_bytes = arith.index_cast(
            T.i64, arith.index_cast(T.index, fx.Int32(R) * sc_e8_row)
        )
        sb_bytes = arith.index_cast(
            T.i64, arith.index_cast(T.index, fx.Int32(C) * sc_e8_row)
        )
        sa_rsrc = buffer_ops.create_buffer_resource(
            A_scale, max_size=False, num_records_bytes=sa_bytes
        )
        sb_rsrc = buffer_ops.create_buffer_resource(
            B_scale, max_size=False, num_records_bytes=sb_bytes
        )

        # BYTE row stride -- the swizzle walks bytes.
        gl_off_a = compute_global_swizzle(
            lane_id, wave_id, m_row_bytes, N_LDS_ROUNDS, preshuffled=False
        )
        gl_off_b = compute_global_swizzle(
            lane_id, wave_id, m_row_bytes, N_LDS_ROUNDS, preshuffled=False
        )

        a_g2s = G2SLoader(a_div, gl_off_a, N_TILES_A, I8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_TILES_B, I8_IR_t, wave_id)
        a_s2r = S2RLoader(wave_i, N_TILES_A)
        b_s2r = S2RLoader(wave_j, N_TILES_B)

        # ── MXFP8 scales: the equal-groups body's cooperative wide load, unchanged ──
        # ONE dwordx2 per operand-half per unrolled body: lane L takes row
        # (half_base + L), so 64 lanes cover the half's 64 rows with no
        # duplicate address, and the pair spans the body's TWO K-steps
        # (adjacent i32 in the same row). 16 dword loads/K-step become 2 -- an
        # 8x cut in L1 requests, which is what actually set the pace: 16 loads
        # x 16 distinct cache lines per K-step equalled ALL of the step's data
        # traffic, so the address path, not the matrix core, was the limit.
        # The tile each lane needs is recovered with ds_bpermute (lane L pulls
        # tile t from lane t*16 + L%16) -- LDS crossbar, not HBM.
        assert (
            N_TILES_A == 4 and N_TILES_B == 4
        ), "cooperative scale load assumes 4x16=64 rows/half"

        sa0_base = r_base + wave_i * fx.Int32(N_TILES_A * 16)
        sa1_base = r_base + fx.Int32(LDS_BLOCK_R) + wave_i * fx.Int32(N_TILES_A * 16)
        sb0_base = c_base + wave_j * fx.Int32(N_TILES_B * 16)
        sb1_base = c_base + fx.Int32(LDS_BLOCK_C) + wave_j * fx.Int32(N_TILES_B * 16)

        # Only the row term is per-lane; the half base and the K index are
        # wave-uniform, so they ride the buffer instruction's SGPR soffset and
        # the address costs no per-step VALU and exactly one VGPR. (On gfx950
        # soffset IS included in the num_records bounds check, so the bound
        # above really does clamp these.)
        sc_voff = lane_id * sc_i32_row
        sc_sbase = [
            (_rs, (_base * sc_i32_row + m_start_i32) * fx.Int32(4))
            for _rs, _base in (
                (sa_rsrc, sa0_base),
                (sa_rsrc, sa1_base),
                (sb_rsrc, sb0_base),
                (sb_rsrc, sb1_base),
            )
        ]
        sc_perm = [
            fx.Int32(t_ * 64) + m_lane * fx.Int32(4)
            for t_ in range_constexpr(N_TILES_A)
        ]

        def issue_chunk(k_i32):
            """Cooperative scale loads for K-steps (k_i32, k_i32+1).

            Returns 2 raw i32 per operand-half, flat: [a0_lo, a0_hi, a1_lo, ...].
            """
            out = []
            for _rs, _sb in sc_sbase:
                soff = _sb + k_i32 * fx.Int32(4)
                # const_expr, not a bare `if`: 0.3.0's AST rewriter
                # routes an `if` in a kernel body through scf_if_dispatch.
                # For a Python-constant condition that still calls only the
                # taken branch -- so this is about intent, not correctness:
                # const_expr says "compile-time" at the call site instead of
                # leaving a reader to check _is_dynamic.
                if const_expr(SC_PAIR):
                    v = fx.Vector(
                        buffer_ops.buffer_load(
                            _rs, sc_voff, vec_width=2, dtype=T.i32, soffset_bytes=soff
                        )
                    )
                    out.append(v[0])
                    out.append(v[1])
                else:
                    # Two dwords: a dword needs only 4-byte alignment, so this
                    # is valid for any SCALE_I32_ROW parity.
                    out.append(
                        buffer_ops.buffer_load(
                            _rs, sc_voff, vec_width=1, dtype=T.i32, soffset_bytes=soff
                        )
                    )
                    out.append(
                        buffer_ops.buffer_load(
                            _rs,
                            sc_voff,
                            vec_width=1,
                            dtype=T.i32,
                            soffset_bytes=soff + fx.Int32(4),
                        )
                    )
            return out

        def window_ok(sub_idx):
            """Is this lane's 32-element sub-block inside the real group?

            ``sub_idx`` is the MFMA K-sub-block index -- equivalently the
            scale-i32 index -- so ``e = sub_idx*128 + k_grp*32`` relative to
            the window base. IDENTICAL to the MXFP8 formula: there it was the
            K-step index because a step was one sub-block; here a step is two,
            and the caller passes 2*step + ksub. Covers, with one predicate:
            the leading round-down to 256, the trailing partial K-step, the
            even/>=4 extension, and empty groups.
            """
            e = sub_idx * fx.Int32(BLOCK_M) + kgrp32
            return (e >= head) & (e < span)

        def use_chunk(chunk, p, k_idx):
            """Redistribute + consume the chunk's K-step `p` -> (sa0,sa1,sb0,sb1),
            masked to e8m0 0x00 outside the group.

            BOTH operands are masked, not just A. Masking A alone is the
            cheaper argument -- a == 0 kills the product -- but it is only
            valid while b is finite, and an e8m0 byte of 0xFF is NaN, for
            which 0 * NaN = NaN. Out-of-*allocation* reads are already
            clamped to 0 by the bounded descriptors above, but the pad
            columns this window deliberately reads are INSIDE the
            allocation, so nothing but this select covers them. 8 extra
            per-lane selects per K-step against 64 MFMAs.
            """
            ok = window_ok(k_idx)
            groups = []
            for h_ in range_constexpr(4):
                src = chunk[2 * h_ + p]
                grp = []
                for t_ in range_constexpr(N_TILES_A):
                    v = rocdl.ds_bpermute(res=T.i32, index=sc_perm[t_], src=src)
                    sc = ((ArithValue(v) >> kshift) & mask_ff) * bcast
                    grp.append(arith.select(ok, sc, zero_i32))
                groups.append(grp)
            return groups

        def use_step(chunk, k_step):
            """Both sub-blocks of K-step ``k_step``: sa[ksub][tile] etc."""
            sa0, sa1, sb0, sb1 = [], [], [], []
            for ks in range_constexpr(KSUBS):
                g0, g1, g2, g3 = use_chunk(
                    chunk, ks, k_step * fx.Int32(KSUBS) + fx.Int32(ks)
                )
                sa0.append(g0)
                sa1.append(g1)
                sb0.append(g2)
                sb1.append(g3)
            return sa0, sa1, sb0, sb1

        def mma(a, b, c, sa, sb):
            for ks in range_constexpr(KSUBS):
                for i in range_constexpr(N_TILES_A):
                    for j in range_constexpr(N_TILES_B):
                        idx = i * N_TILES_B + j
                        c[idx] = _mfma_scale_agpr(
                            a[i][ks], b[j][ks], sa[ks][i], sb[ks][j], c[idx]
                        )
            return c

        zero = Vec.filled(4, 0.0, fx.Float32)
        c00 = [zero] * N_ACCUMS
        c01 = [zero] * N_ACCUMS
        c10 = [zero] * N_ACCUMS
        c11 = [zero] * N_ACCUMS

        def _lds_swizzle(s2r):
            """Byte offsets of a tile's two K=128 sub-blocks in the staged
            128-byte row. Same formula as the MXFP8 body's -- the LDS
            geometry is in bytes -- but these are now two OPERANDS, not two
            halves of one."""
            out = []
            for row_off in range_constexpr(s2r.n_tiles):
                row = (
                    s2r.wave_idx * fx.Int32(s2r.n_tiles * 16)
                    + fx.Int32(row_off * 16)
                    + (lane_id % fx.Int32(16))
                )
                swz = []
                for ii in range_constexpr(KSUBS):
                    col = (lane_id // fx.Int32(16)) * fx.Int32(16) + fx.Int32(ii * 64)
                    r_, c_ = swizzle_128(row, col)
                    swz.append(r_ * fx.Int32(BLOCK_K_BYTES) + c_)
                out.append(swz)
            return out

        def s2r_load_fp4(s2r, lds_src):
            """Per tile: ``[i32x4_ksub0, i32x4_ksub1]``, NOT packed -- an fp4
            K=128 operand is 16 bytes, so each half already IS an operand."""
            swz = _lds_swizzle(s2r)
            return [
                [s2r.load_one(lds_src, swz[t_][ks]) for ks in range_constexpr(KSUBS)]
                for t_ in range_constexpr(s2r.n_tiles)
            ]

        # ── Prologue: 8-buffer LDS pipeline pre-fill ──
        # Chunk 0 is issued FIRST, ahead of every data copy, so it is the
        # oldest thing in flight: the two prologue barriers keep their counts
        # (an older op never changes how many ops trail a newer one), they
        # retire the chunk for free, and the wave enters the loop in exactly
        # the steady state W1A assumes. K_eff >= 4 guarantees steps 0 and 1
        # exist, so this needs no guard.
        # TWO chunks: an fp4 K-step consumes a whole chunk (its two i32 are
        # ksub 0 and 1), and the unrolled body is two K-steps. Both are issued
        # ahead of every data copy so they stay the oldest ops in flight and
        # the prologue barrier counts below are unchanged -- an older op never
        # changes how many ops trail a newer one.
        chunk0 = issue_chunk(fx.Int32(0))
        chunk1 = issue_chunk(fx.Int32(KSUBS))
        a_g2s.load(a_cur0, A0_gl + 0 * BLOCK_K_BYTES)
        b_g2s.load(b_cur0, B0_gl + 0 * BLOCK_K_BYTES)
        b_g2s.load(b_cur1, B1_gl + 0 * BLOCK_K_BYTES)
        a_g2s.load(a_cur1, A1_gl + 0 * BLOCK_K_BYTES)

        a_g2s.load(a_next0, A0_gl + 1 * BLOCK_K_BYTES)
        b_g2s.load(b_next0, B0_gl + 1 * BLOCK_K_BYTES)
        b_g2s.load(b_next1, B1_gl + 1 * BLOCK_K_BYTES)
        a_g2s.load(a_next1, A1_gl + 1 * BLOCK_K_BYTES)

        wait_barrier((3 * N_TILES_A) + (4 * N_TILES_B))
        a0 = s2r_load_fp4(a_s2r, a_cur0)
        wait_barrier((3 * N_TILES_A) + (3 * N_TILES_B))
        b0 = s2r_load_fp4(b_s2r, b_cur0)

        def _cluster_plain(lds_dst, g2s, k_off, s2r, lds_src, a, b, c, sa, sb):
            g2s.load(lds_dst, k_off)
            rt = s2r_load_fp4(s2r, lds_src)
            c = mma(a, b, c, sa, sb)
            return c, rt

        def _cluster_il(lds_dst, g2s, k_off, s2r, lds_src, a, b, c, sa, sb):
            # Interleave this quadrant's 32 scaled MFMAs (4x4 tiles x 2 ksub)
            # with the g2s (4 steps) and s2r (4 tiles x 2 sub-blocks) loads of
            # the NEXT fragment, so the loads co-issue in the MFMA execute
            # shadow.
            #
            # The MXFP8 body spelled this schedule out by hand for its 16
            # MFMAs. At 32 it is generated instead, by the same rule the
            # forward body uses: spread the 12 loads evenly across the MFMA
            # stream. fp4 doubles the MFMAs per cluster without changing the
            # load count, so every load lands in a wider shadow than before.
            swz = _lds_swizzle(s2r)
            rt = [[None, None] for _ in range(s2r.n_tiles)]
            g_at, s0_at, s1_at = PLANS[(g2s.n_load_steps, s2r.n_tiles)]

            for idx in range_constexpr(len(MFMA_SEQ)):
                for gi in range_constexpr(len(g_at[idx])):
                    g2s.load_one(lds_dst, k_off, g_at[idx][gi])
                for si in range_constexpr(len(s0_at[idx])):
                    t_ = s0_at[idx][si]
                    rt[t_][0] = s2r.load_one(lds_src, swz[t_][0])
                for si in range_constexpr(len(s1_at[idx])):
                    t_ = s1_at[idx][si]
                    rt[t_][1] = s2r.load_one(lds_src, swz[t_][1])
                i_, j_, ks_ = MFMA_SEQ[idx]
                c[i_ * N_TILES_B + j_] = _mfma_scale_agpr(
                    a[i_][ks_],
                    b[j_][ks_],
                    sa[ks_][i_],
                    sb[ks_][j_],
                    c[i_ * N_TILES_B + j_],
                )
            return c, rt

        _cluster = _cluster_il if _INTERLEAVE else _cluster_plain

        # ── Main K-loop (scf.for, unroll-2, RUNTIME trip count) ──
        # One body = 2 K-steps: the ping-pong pointer swap is period-2 so LDS
        # pointers are not loop-carried; carried state is the 4 accumulator
        # groups PLUS the a0/b0 fragments (this schedule prefetches the next
        # fragment during compute) PLUS the prefetched scale chunk.
        # N_ROLLED = K_eff - 2 is even by construction, so the constexpr
        # odd-remainder peel is not needed here at all.
        _R = arith._to_raw
        NA, NB = N_TILES_A, N_TILES_B

        def _one_step(kk_i, a0, b0, accs, bufs, chunk, p, issue_k):
            """One fp4 K-step (256 tokens = two K=128 MFMA sub-blocks).

            `p` is the step's slot in the unrolled body (0 or 1). BOTH slots
            issue a chunk now -- each fp4 K-step needs its own pair of scale
            i32 -- where the MXFP8 body issued one per body. `issue_k` is the
            scale-i32 index of the step being prefetched.
            """
            c00, c01, c10, c11 = accs
            ac0, ac1, an0, an1, bc0, bc1, bn0, bn1 = bufs
            k2 = (kk_i + fx.Int32(2)) * fx.Int32(BLOCK_K_BYTES)
            wait_barrier(W1A if p == 0 else W1B)
            # The scales this step consumes came off HBM a whole body ago and
            # the barrier above already retired them; the only HBM touch here
            # is the prefetch for the body two K-steps out -- the same horizon
            # the data ping-pong runs at.
            ch_new = issue_chunk(issue_k)
            sa0, sa1, sb0, sb1 = use_step(chunk, kk_i)
            c00, b1 = _cluster(
                ac0, a_g2s, A0_gl + k2, b_s2r, bc1, a0, b0, c00, sa0, sb0
            )
            c01, a1 = _cluster(
                bc0, b_g2s, B0_gl + k2, a_s2r, ac1, a0, b1, c01, sa0, sb1
            )
            wait_barrier(W2A if p == 0 else W2B)
            c10, a0n = _cluster(
                bc1, b_g2s, B1_gl + k2, a_s2r, an0, a1, b0, c10, sa1, sb0
            )
            c11, b0n = _cluster(
                ac1, a_g2s, A1_gl + k2, b_s2r, bn0, a1, b1, c11, sa1, sb1
            )
            new_bufs = (an0, an1, ac0, ac1, bn0, bn1, bc0, bc1)
            return a0n, b0n, (c00, c01, c10, c11), new_bufs, ch_new

        bufs0 = (a_cur0, a_cur1, a_next0, a_next1, b_cur0, b_cur1, b_next0, b_next1)

        # a0/b0 are now [tile][ksub] nests; flatten for the scf.for state.
        def _flat_frag(frag):
            out = []
            for t_ in frag:
                out.append(_R(t_[0]))
                out.append(_R(t_[1]))
            return out

        def _unflat_frag(flat, n_tiles):
            return [
                [flat[KSUBS * i + ks] for ks in range(KSUBS)] for i in range(n_tiles)
            ]

        # Two chunks carried, not one: each fp4 K-step of the body owns a
        # pair of scale i32, so the body prefetches two. 8 more live i32 than
        # the MXFP8 body -- paid back by the fragments, which stay the same
        # width (4 tiles x 2 x i32x4 == 4 tiles x i32x8).
        NA_F = KSUBS * NA
        NB_F = KSUBS * NB

        def _pack(a0, b0, accs, ch0, ch1):
            c00, c01, c10, c11 = accs
            return (
                _flat_frag(a0)
                + _flat_frag(b0)
                + [_R(x) for x in c00]
                + [_R(x) for x in c01]
                + [_R(x) for x in c10]
                + [_R(x) for x in c11]
                + [_R(x) for x in ch0]
                + [_R(x) for x in ch1]
            )

        def _unpack(state):
            o = 0
            a0 = _unflat_frag(state[o : o + NA_F], NA)
            o += NA_F
            b0 = _unflat_frag(state[o : o + NB_F], NB)
            o += NB_F
            c00 = list(state[o : o + N_ACCUMS])
            o += N_ACCUMS
            c01 = list(state[o : o + N_ACCUMS])
            o += N_ACCUMS
            c10 = list(state[o : o + N_ACCUMS])
            o += N_ACCUMS
            c11 = list(state[o : o + N_ACCUMS])
            o += N_ACCUMS
            ch0 = list(state[o : o + N_CHUNK])
            o += N_CHUNK
            ch1 = list(state[o : o + N_CHUNK])
            o += N_CHUNK
            return a0, b0, (c00, c01, c10, c11), ch0, ch1

        init_state = _pack(a0, b0, (c00, c01, c10, c11), chunk0, chunk1)
        state = init_state
        for kk, state in range(0, N_ROLLED, 2, init=init_state):
            a0, b0, accs, ch0, ch1 = _unpack(state)
            kk_i = fx.Int32(kk)
            # Step kk consumes ch0 and prefetches step kk+2's pair; step kk+1
            # consumes ch1 and prefetches step kk+3's. Scale-i32 index of
            # step s is 2s.
            a0, b0, accs, bufs, ch0n = _one_step(
                kk_i,
                a0,
                b0,
                accs,
                bufs0,
                ch0,
                0,
                (kk_i + fx.Int32(2)) * fx.Int32(KSUBS),
            )
            a0, b0, accs, bufs, ch1n = _one_step(
                kk_i + fx.Int32(1),
                a0,
                b0,
                accs,
                bufs,
                ch1,
                1,
                (kk_i + fx.Int32(3)) * fx.Int32(KSUBS),
            )
            state = yield _pack(a0, b0, accs, ch0n, ch1n)

        a0, b0, accs, chunk0, chunk1 = _unpack(state)
        c00, c01, c10, c11 = accs
        a_cur0, a_cur1, a_next0, a_next1, b_cur0, b_cur1, b_next0, b_next1 = bufs0

        # ── Tail steps K_eff-2 and K_eff-1 ──
        # Neither touches HBM for scales: the last loop body prefetched the
        # chunk covering exactly these two steps. The barrier counts stay at
        # the steady-state values -- the tails issue no scale loads, and a count that is
        # too SMALL only retires more than needed, never less.
        t0 = K_eff - fx.Int32(2)
        t1 = K_eff - fx.Int32(1)

        wait_barrier((2 * N_TILES_A) + (2 * N_TILES_B))
        sa0, sa1, sb0, sb1 = use_step(chunk0, t0)
        b1 = s2r_load_fp4(b_s2r, b_cur1)
        c00 = mma(a0, b0, c00, sa0, sb0)
        a1 = s2r_load_fp4(a_s2r, a_cur1)
        c01 = mma(a0, b1, c01, sa0, sb1)
        wait_barrier((1 * N_TILES_A) + (1 * N_TILES_B))
        a0 = s2r_load_fp4(a_s2r, a_next0)
        c10 = mma(a1, b0, c10, sa1, sb0)
        b0 = s2r_load_fp4(b_s2r, b_next0)
        c11 = mma(a1, b1, c11, sa1, sb1)

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        wait_barrier(0)
        sa0, sa1, sb0, sb1 = use_step(chunk1, t1)
        b1 = s2r_load_fp4(b_s2r, b_cur1)
        a1 = s2r_load_fp4(a_s2r, a_cur1)
        c00 = mma(a0, b0, c00, sa0, sb0)
        c01 = mma(a0, b1, c01, sa0, sb1)
        c10 = mma(a1, b0, c10, sa1, sb0)
        c11 = mma(a1, b1, c11, sa1, sb1)

        # ── Epilogue: DIRECT f32 store into out[g] (no atomics) ──
        # Exactly one block owns each (group, output tile), so every valid
        # (row, col) is written once and the output needs no zero-init.
        #
        # EMPTY GROUPS need one explicit correction. The A-scale mask forces
        # e8m0 to 0x00, which is 2**-127 and NOT zero, so a fully-masked block
        # accumulates a*2**-127*b ~ 1e-36 per step rather than exactly 0. In a
        # non-empty group that is 38 orders of magnitude under the real
        # entries and irrelevant, but an expert that received no tokens must
        # get an exactly-zero gradient plane: 1e-36 is a normal bf16, and an
        # Adam-style optimiser divides by sqrt(v), so a consistently tiny
        # gradient is not automatically a tiny update. A chunked body gets this by
        # giving empty groups no block at all; here every (g, tile) has one,
        # so it is a select on the stored value.
        nonempty = m_end > m_start
        f32_zero = fx.Float32(0.0)
        out_r_i = arith.index_cast(T.index, out_r)
        out_c_i = arith.index_cast(T.index, out_c)
        n_experts = fx.Int32(E)
        nbytes = arith.index_cast(
            T.i64,
            out_r_i * out_c_i * arith.index_cast(T.index, n_experts) * fx.Index(4),
        )
        o_rsrc = buffer_ops.create_buffer_resource(
            OUT, max_size=False, num_records_bytes=nbytes
        )
        base_row = r_base + wave_i * fx.Int32(N_TILES_A * 16)
        base_col = c_base + wave_j * fx.Int32(N_TILES_B * 16)
        g_off = g * out_r * out_c
        oob = out_r * out_c * n_experts

        def store_group(frag, br, bc):
            for ti in range_constexpr(N_TILES_A):
                row = br + fx.Int32(ti * 16) + k_grp * fx.Int32(4)
                for tj in range_constexpr(N_TILES_B):
                    col = bc + fx.Int32(tj * 16) + m_lane
                    col_ok = col < out_c
                    vec = frag[ti * N_TILES_B + tj]
                    for e in range_constexpr(4):
                        r_ = row + fx.Int32(e)
                        ok = (r_ < out_r) & col_ok
                        off = arith.select(ok, g_off + r_ * out_c + col, oob)
                        val = fx.Vector(vec)[e]
                        val = arith.select(nonempty, val, f32_zero)
                        buffer_ops.buffer_store(val, o_rsrc, off)

        store_group(c00, base_row, base_col)
        store_group(c01, base_row, base_col + fx.Int32(LDS_BLOCK_C))
        store_group(c10, base_row + fx.Int32(LDS_BLOCK_R), base_col)
        store_group(
            c11, base_row + fx.Int32(LDS_BLOCK_R), base_col + fx.Int32(LDS_BLOCK_C)
        )

    @flyc.jit
    def launch_wgrad(
        A,
        B,
        OUT,
        A_scale,
        B_scale,
        OFFS,
        n_blocks: fx.Int32,
        n_r_tiles: fx.Int32,
        n_c_tiles: fx.Int32,
        out_r: fx.Int32,
        out_c: fx.Int32,
        m_row: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_wgrad(
            A,
            B,
            OUT,
            A_scale,
            B_scale,
            OFFS,
            n_r_tiles,
            n_c_tiles,
            out_r,
            out_c,
            m_row,
            value_attrs={
                "rocdl.waves_per_eu": 1,
                "rocdl.flat_work_group_size": "256,256",
            },
        ).launch(grid=(n_blocks, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_wgrad


@functools.lru_cache(maxsize=None)
def cached_launch(R: int, C: int, E: int, sc_pair: bool, order: str):
    return _compile(R, C, E, sc_pair, order)
