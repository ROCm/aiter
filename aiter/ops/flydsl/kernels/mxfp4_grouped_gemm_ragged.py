# SPDX-License-Identifier: MIT
#
# RAGGED MXFP4 FORWARD/DGRAD grouped GEMM, 4-wave AGPR body, gfx950 (CDNA4).
#
#     out[group_g] = X[group_g] @ W[g]^T        X(M,K), W(E,N,K) -> (M,N)
#
# ---------------------------------------------------------------------------
# The ragged machinery -- device-resident offsets, the block -> (group, row
# tile, col tile) walk, the overhang mask, the over-provisioned grid -- does not
# depend on the element type, so it is identical to the MXFP8 form of this body.
#
# WHAT AN FP4 ELEMENT TYPE CHANGES, and why each change is what it is:
#
#   1. **The LDS K-step stays 128 BYTES and becomes 256 ELEMENTS.** Every piece
#      of the data path -- ``compute_global_swizzle``, ``swizzle_128``, the
#      ``BufferCopyLDS128b`` G2S atom, the S2R column formula -- addresses
#      bytes, not elements, so all of it is reused VERBATIM. What changes is
#      that one LDS row now feeds TWO K=128 MFMA sub-blocks (``ksub`` 0 and 1)
#      instead of one. The K-loop trip count halves and the MFMA count per step
#      doubles, which raises the MFMA-per-ds_read ratio from 2.0 to 4.0.
#
#   2. **S2R keeps the two 16-byte halves separate.** The MXFP8 loader packs
#      them with ``pack_i32x4_i32x8`` because an fp8 K=128 operand is 32 bytes.
#      An fp4 K=128 operand is 16 bytes, so each half IS an operand. Packing and
#      re-splitting would cost the round-trip FlyDSL's own fp4 reference calls
#      out by name (``fp4_gemm_4wave.py:45``): ~64 VGPR of split temporaries on
#      top of the i32x8 fragments, pushing arch VGPR to 256 and spilling scale.
#
#   3. **The MFMA is the same instruction with cbsz:4 blgp:4.**
#      ``v_mfma_scale_f32_16x16x128_f8f6f4`` selects fp4 e2m1 by operand-format
#      field, not by opcode. The accumulator stays pinned in AGPR.
#
#   4. **The scale plane is UNCHANGED and stays broadcast.** One E8M0 per 32
#      elements either way, so ``SCALE_I32_ROW = K // 128`` still holds: an i32
#      is 4 e8m0 = 128 elements = one MFMA K-sub-block. K-step ``s`` consumes
#      scale i32 ``2s + ksub``, which means the MXFP8 kernel's paired dwordx2
#      chunk load -- written to cover two of ITS K-steps -- now covers exactly
#      one of ours. The cooperative load + ds_bpermute redistribution and the
#      one-e8m0-broadcast-to-4-bytes trick carry over untouched, so this kernel
#      needs NO ``shuffle_scale_w4`` preshuffle and NO opsel plumbing, unlike
#      FlyDSL's dense fp4 reference.
#
#   5. **Byte offsets halve.** A row is K//2 bytes. Every global offset, buffer
#      bound and ``num_records_bytes`` is in bytes and was audited for this.
#
#   6. **K need not be a multiple of 256.** At K % 256 == 128 (DSV3's 1408) the
#      last K-step is a HALF step: the G2S still copies a full 128-byte row --
#      harmless, it reads into the next row and the bounded buffer resource
#      clamps the final one -- but only ksub 0 is issued to the MFMA.
# ---------------------------------------------------------------------------
#
# This body and its sibling ``mxfp4_grouped_wgrad_ragged`` are the same pipeline
# pointed in two directions, because the two problems have the SAME operand
# layout: both are `C = A @ B^T` with A and B row-major and the contraction
# running along the contiguous axis. What differs is only what the axes mean:
#
#                    wgrad                           forward (here)
#   A                go_t   (R=N, M_TOTAL)           X      (M_tok, K)
#   B                ia_t   (C=K, M_TOTAL)           W[g]   (N, K)
#   contraction      tokens, length m_g              K, length K (fixed)
#   groups partition the CONTRACTION                 the OUTPUT ROWS
#   per-group offset a column offset (m_start)       a row offset on A, a plane on B
#   output           (E, R, C) f32                   (M_tok, N) bf16
#
# So the group index moves from the K-walk into the tile's base addresses, the
# contraction becomes short and uniform (K/128 = 16 or 11 steps, versus the
# wgrad's 192), and the epilogue writes bf16 into one 2-D matrix instead of f32
# into a per-expert stack.
#
# Two structural consequences of the short contraction:
#   * The K-loop is fully unrolled at compile time. wgrad needed `scf.for`
#     because a 192-step constexpr loop hung the JIT; at 6-8 fp4 steps the
#     rolled form only costs scheduling freedom, and the loop-carried a0/b0
#     fragments and scale chunk disappear.
#   * Pipeline fill and drain are a much larger fraction of the kernel, and MORE
#     so at fp4 than at fp8: halving the step count to 6-8 leaves 2 prologue
#     steps and 2 tail steps out of 6, so half the kernel is fill/drain at
#     K=1408. That is the main structural risk in this port.
#
# The vmcnt accounting is derived, not transcribed. A hand-computed set of four
# constants (W1A/W2A/W1B/W2B) covers a periodic steady state; with the loop
# unrolled the issue pattern is not perfectly periodic (the last steps stop
# prefetching scales), so `_VmCounter` tracks issued vector-memory ops and each
# barrier waits for exactly "everything up to the op I depend on". It reproduces
# the hand-computed constants in the steady state.

# NOTE: no `from __future__ import annotations` -- fx.struct needs real types.

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

BLOCK_R = 256  # output rows (tokens) per tile
BLOCK_C_DEFAULT = 256  # output cols (N) per tile; 128 also supported, see pick_block_c
# One pipeline step stages 128 BYTES of K per operand row -- the same LDS
# geometry the MXFP8 body uses, and the reason its whole data path is reusable.
# At 2 fp4 per byte that is 256 elements, i.e. two K=128 MFMA sub-blocks.
BLOCK_K_BYTES = 128
FP4_PER_BYTE = 2
BLOCK_K_ELEMS = BLOCK_K_BYTES * FP4_PER_BYTE  # 256
KSUBS = 2  # MFMA K=128 sub-blocks per staged LDS row


def pick_block_c(N: int) -> int:
    """Column tile width. Always 256 -- the narrow tile was measured and lost.

    The motivation for a narrow tile is real: a tile that overhangs N is not
    free, because the MFMAs run on the overhang and the results are discarded at
    the store. At K=2048, M=196608, N=1408 (6x256 = 1536 columns) and N=1536 take
    the SAME 0.83 ms, so the 9.1% overhang is paid in full, and the kernel's
    29.8% of peak on useful FLOPs is really 32.3% of the hardware.

    1408 = 2^7 x 11 and a legal BLOCK_C is 64 x N_TILES_B, so 128 is the only
    sane width that divides it. Measured (median of 3, interleaved):

        K=2048 N=1408   256: 0.831 ms 1365 TF/s   128: 0.979 ms 1158   0.849x
        K=1408 N=2048   256: 0.833 ms 1361 TF/s   128: 1.034 ms 1097   0.806x
        K=2048 N=2048   256: 1.105 ms 1493 TF/s   128: 1.326 ms 1244   0.833x

    So the narrow tile removes the whole overhang at N=1408 and is still 15%
    slower. Why: per K-step a wave issues 4*NA*NB MFMAs against 4*(NA+NB) LDS
    reads, so the MFMA-per-ds_read ratio is NA*NB/(NA+NB) -- 2.0 at 4x4, 1.33 at
    4x2. Hardware efficiency drops 32.3% -> 25.1%, a 22% loss to save 9% of work.

    There is no better tile available: the accumulators already fill all 256
    AGPRs at NA*NB = 16 per quadrant, so no shape with a higher ratio fits. The
    NA=8, NB=2 corner (ratio 1.6, and exactly 160 KB of LDS) interpolates to
    ~28% hardware, still below the 29.8% useful the wide tile delivers.

    Conclusion: the overhang at N=1408 is not recoverable by retiling. Pass
    ``block_c=128`` explicitly to re-run the comparison.

    fp4 note: those numbers are the MXFP8 body's. fp4 doubles the MFMAs per LDS
    read (each staged row feeds two K=128 sub-blocks), so the ratio argument
    moves 2.0 -> 4.0 and the narrow tile's 1.33 -> 2.67. Whether that changes
    the verdict is a measurement, not an inference -- re-run it with block_c.
    """
    return BLOCK_C_DEFAULT


# Interleave each quadrant's MFMAs with the g2s/s2r loads of the NEXT
# fragment so they co-issue in the MFMA execute shadow. Worth +6.6% in wgrad
# once its scale loads stopped saturating the L1 address path. Set
# AITER_FLYDSL_MXFP4_GEMM_INTERLEAVE=0 for the plain cluster.
_INTERLEAVE = os.environ.get("AITER_FLYDSL_MXFP4_GEMM_INTERLEAVE", "1") != "0"


def _mfma_scale_agpr(a, b, sa, sb, acc):
    """fp4 16x16x128 scaled MFMA with the accumulator pinned in AGPR (=a,...,0)
    so the f32x4 accumulates in place and the compiler does not shuffle it
    between AGPR slots.

    ``a``/``b`` are i32x4 -- 16 bytes = 32 fp4 per lane = the K=128 operand.
    The MXFP8 body passes i32x8 here; same instruction, the operand format
    field ``cbsz``/``blgp`` (0 = fp8 e4m3, 4 = fp4 e2m1) is what changes the
    element width, not the opcode.

    ``sa``/``sb`` are broadcast-i32 E8M0 scales (one e8m0 replicated to 4
    bytes), so ``op_sel`` is a don't-care and is left at its default. That is
    why this kernel needs neither preshuffled scales nor opsel bookkeeping."""
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


class _VmCounter:
    """Issue-order bookkeeping for `s_waitcnt vmcnt(n)`.

    vmcnt retires IN ORDER, so "wait until op X has landed" is spelled
    "allow at most (ops issued after X) to be outstanding". Every vector
    memory op the kernel issues is counted here; `mark()` records a
    watermark right after the ops a later barrier will depend on, and
    `wait(mark)` emits the barrier with the exact count.

    Getting this wrong in the unsafe direction is silent: too LARGE a count
    under-waits and reads LDS the copy has not filled yet. Deriving it from
    the issue order removes the chance to mis-transcribe a constant.
    """

    def __init__(self):
        self.n = 0

    def issue(self, k):
        self.n += k
        return self.n

    def mark(self):
        return self.n

    def wait(self, mark):
        wait_barrier(self.n - mark)


def _compile(K: int, N: int, E: int, BLOCK_C: int):
    """Compile a RAGGED forward grouped GEMM for one (K, N, E) shape.

    Deliberately absent from the key: the token count and every group size.
    Both are runtime. Under real MoE routing they change every step, and a
    shape-keyed compile leaks a JIT compile plus a retained GPU module per
    step per layer -- the same defect the dim1 cast had, where it cost 618x.
    The even-groups parent keyed on (K, N, M_G, M_TOTAL, BLOCK_C) and so
    recompiled per token count; only balanced routing hid it.
    """
    assert K % 128 == 0, f"K ({K}) must be a multiple of 128 (MFMA K-sub-block)"
    assert BLOCK_C in (128, 256), f"BLOCK_C {BLOCK_C} not supported"

    K_BYTES = K // FP4_PER_BYTE
    # Steps of 256 fp4. A trailing 128 (K % 256 == 128, e.g. DSV3's 1408)
    # becomes a HALF step: the G2S still stages a whole 128-byte row and
    # only ksub 0 reaches the MFMA.
    K_ITERS = (K + BLOCK_K_ELEMS - 1) // BLOCK_K_ELEMS
    HALF_TAIL = (K % BLOCK_K_ELEMS) != 0
    assert (
        K_ITERS >= 4
    ), f"K {K} gives {K_ITERS} steps; need >= 4 (2 prologue + 2 tails)"

    # i32 of e8m0 per operand row. One i32 is 4 E8M0 = 128 elements = one
    # MFMA K-sub-block, so this count is IDENTICAL to the MXFP8 body's even
    # though the step size doubled: K-step s consumes i32 2s and 2s+1.
    SCALE_I32_ROW = K // 128

    N_TILES_A = BLOCK_R // 4 // 16
    N_TILES_B = BLOCK_C // 4 // 16
    N_ACCUMS = N_TILES_A * N_TILES_B
    N_LDS_ROUNDS = max(N_TILES_A, N_TILES_B)

    # A dwordx2 scale load needs an even element index. The index is
    # lane*SCALE_I32_ROW + row_base*SCALE_I32_ROW + 2s; the 2s term is always
    # even, so the row stride alone decides. K=2048 -> 16 (paired);
    # K=1408 -> 11 (two dword loads instead; same cache lines, one extra
    # instruction).
    SC_PAIR = SCALE_I32_ROW % 2 == 0
    N_SC_OPS = 4 if SC_PAIR else 8  # scale vm ops per chunk (= one K-step)
    # One chunk per K-step now, where the MXFP8 body needed one per two: its
    # paired dwordx2 covered two of its 128-element steps, and covers exactly
    # one of our 256-element ones.
    N_CHUNKS = K_ITERS

    LDS_BLOCK_R = BLOCK_R // 2  # each wave-dim half owns 128 rows
    LDS_BLOCK_C = BLOCK_C // 2

    def _interleave_plan(n_mfma, n_g2s, n_tiles):
        """Where to slot each load into the cluster's MFMA stream.

        Returns three lists indexed by MFMA position: the g2s steps, the
        first-half s2r tiles and the second-half s2r tiles to issue just
        before that MFMA. Generated rather than written out because the
        counts move with the tile -- BLOCK_C=256 is 16 MFMAs against 4 g2s
        steps and 4x2 s2r reads, BLOCK_C=128 is 8 against 4 and 2x2.

        Computed HERE, outside the kernel body, on purpose: FlyDSL's AST
        rewriter turns an `if` inside a nested kernel-body function into a
        separate `__then_N` function that cannot see the enclosing closure,
        so schedule logic has to be plain Python that runs before tracing.
        The kernel body then walks the plan with no branches at all.
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
            # Strictly < n_mfma for every n_, so no load is ever dropped.
            pos = int(round((n_ + 1) * n_mfma / (len(acts) + 1)))
            {"g": g_at, "s0": s0_at, "s1": s1_at}[act[0]][pos].append(act[1])
        return g_at, s0_at, s1_at

    # (tile_i, tile_j, ksub). Two MFMAs per accumulator per step, because a
    # staged 128-byte row is two K=128 sub-blocks. ksub is the OUTER loop so
    # both MFMAs of an accumulator are not back-to-back on the same AGPR.
    MFMA_SEQ = [
        (i, j, ks)
        for ks in range(KSUBS)
        for i in range(N_TILES_A)
        for j in range(N_TILES_B)
    ]
    # Keyed by (g2s steps, s2r tiles): clusters 1 and 4 push A and pull B,
    # clusters 2 and 3 the other way round. Keys collide harmlessly when the
    # tile is square.
    PLANS = {
        (N_TILES_A, N_TILES_B): _interleave_plan(len(MFMA_SEQ), N_TILES_A, N_TILES_B),
        (N_TILES_B, N_TILES_A): _interleave_plan(len(MFMA_SEQ), N_TILES_B, N_TILES_A),
    }
    a_lds_size = LDS_BLOCK_R * BLOCK_K_BYTES
    b_lds_size = LDS_BLOCK_C * BLOCK_K_BYTES

    # Int8, not Float8E4M3FN: LDS holds packed fp4 BYTES and nothing in the
    # data path interprets them until the MFMA does.
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
    def kernel_fwd(
        A: fx.Tensor,
        B: fx.Tensor,
        OUT: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        OFFS: fx.Tensor,
        n_c_tiles: fx.Int32,
        out_m: fx.Int32,
        out_n: fx.Int32,
    ):
        I8_IR_t = fx.Int8.ir_type

        # ── Block -> (group, row tile, col tile), resolved ON DEVICE ──
        # Groups partition the OUTPUT ROWS here, not the contraction, so
        # unlike the wgrad kernel nothing about the K-walk changes: K is
        # uniform across experts. Only this mapping and the epilogue mask
        # care that the groups are ragged.
        bid = fx.block_idx.x
        slot = ArithValue(bid) // n_c_tiles  # which row tile overall
        c_base = (ArithValue(bid) % n_c_tiles) * fx.Int32(BLOCK_C)

        offs_rsrc = buffer_ops.create_buffer_resource(
            OFFS, max_size=False, num_records_bytes=E * 4
        )
        # Uniform (SGPR) loads: every lane in the block wants the same
        # boundaries, so this is E scalar loads, not E per-lane loads.
        ends = [
            ArithValue(
                buffer_ops.buffer_load(
                    offs_rsrc, fx.Int32(i), vec_width=1, dtype=T.i32, is_scalar=True
                )
            )
            for i in range(E)
        ]
        starts = [ArithValue(fx.Int32(0))] + ends[:-1]

        # Walk the groups accumulating row tiles until the slot lands in one.
        # E is constexpr, so this unrolls to E selects on the SALU against a
        # K-walk of many iterations -- the same O(E) prologue the wgrad
        # kernel pays. `active` is false for slots past the last group's
        # tiles, which predicates those blocks off in the epilogue.
        g = ArithValue(fx.Int32(0))
        r_base = ArithValue(fx.Int32(0))
        m_start = ArithValue(fx.Int32(0))
        m_end = ArithValue(fx.Int32(0))
        active = fx.Int32(0) > fx.Int32(0)  # false
        cum = ArithValue(fx.Int32(0))
        # range_constexpr, not range: the AST rewriter turns a plain `range`
        # into a device-side scf.for, whose induction variable is an
        # ArithValue and cannot index the `ends`/`starts` Python lists. This
        # loop must unroll at trace time.
        for i in range_constexpr(E):
            m_i = ends[i] - starts[i]
            t_i = (m_i + fx.Int32(BLOCK_R - 1)) // fx.Int32(BLOCK_R)
            hit = (slot >= cum) & (slot < cum + t_i)
            g = arith.select(hit, fx.Int32(i), g)
            r_base = arith.select(hit, (slot - cum) * fx.Int32(BLOCK_R), r_base)
            m_start = arith.select(hit, starts[i], m_start)
            m_end = arith.select(hit, ends[i], m_end)
            active = active | hit
            cum = cum + t_i

        # A is offset by the group's first token row, B by the expert's
        # weight plane. Both are plain row offsets at the same stride K --
        # the "dynamic per-group stride" problem the MXFP8-MoE post
        # describes does not arise on this axis.
        row0 = m_start + r_base
        wrow0 = ArithValue(g) * fx.Int32(N) + c_base

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
        mask_ff = fx.Int32(0xFF)
        bcast = fx.Int32(0x01010101)

        # BYTES, not elements: an fp4 row is K//2 bytes wide.
        A0_gl = row0 * fx.Int32(K_BYTES)
        A1_gl = (row0 + fx.Int32(LDS_BLOCK_R)) * fx.Int32(K_BYTES)
        B0_gl = wrow0 * fx.Int32(K_BYTES)
        B1_gl = (wrow0 + fx.Int32(LDS_BLOCK_C)) * fx.Int32(K_BYTES)

        gA = make_fp8_buffer_tensor(A, I8_IR_t)
        gB = make_fp8_buffer_tensor(B, I8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        sa_rsrc = buffer_ops.create_buffer_resource(A_scale, max_size=True)
        sb_rsrc = buffer_ops.create_buffer_resource(B_scale, max_size=True)

        # K_BYTES: this is the global ROW STRIDE the swizzle walks, in bytes.
        gl_off_a = compute_global_swizzle(
            lane_id, wave_id, K_BYTES, N_LDS_ROUNDS, preshuffled=False
        )
        gl_off_b = compute_global_swizzle(
            lane_id, wave_id, K_BYTES, N_LDS_ROUNDS, preshuffled=False
        )

        a_g2s = G2SLoader(a_div, gl_off_a, N_TILES_A, I8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_TILES_B, I8_IR_t, wave_id)
        a_s2r = S2RLoader(wave_i, N_TILES_A)
        b_s2r = S2RLoader(wave_j, N_TILES_B)

        vm = _VmCounter()

        # ── MXFP4 scales: cooperative wide load + ds_bpermute ──
        # UNCHANGED from the MXFP8 body, and it costs nothing to keep: the MX
        # scale plane does not depend on the element format. One i32 per row
        # is 4 E8M0 = 128 elements = exactly one MFMA K-sub-block, fp8 or fp4
        # alike. What moved is only the INDEX: our K-step spans two i32
        # (2s, 2s+1) where the MXFP8 body's spanned one, so its dwordx2 pair
        # -- built to cover two of its steps -- covers exactly one of ours.
        #
        # Each operand-half needs the e8m0 byte of 64 consecutive rows (4 MFMA
        # tiles x 16 rows) at this sub-block; lane group k_grp takes byte
        # k_grp and broadcasts it to all 4 bytes, which is why op_sel can stay
        # at its default and no scale preshuffle is needed.
        #
        # The naive form -- one buffer_load per (half, tile) -- issues 16 dword
        # loads per K-step that touch the SAME 16 cache lines four times over,
        # because lanes m_lane..m_lane+48 share an address. That is the same L1
        # request count as all of the step's data traffic, and it, not the
        # matrix core, sets the pace (measured in wgrad: stubbing the scales
        # out took it 822 -> 1486 TFLOP/s).
        #
        # Instead: ONE cooperative load per half, lane L taking row
        # (half_base + L), so 64 lanes cover 64 rows with no duplicate address,
        # widened to a dwordx2 spanning two K-steps. 8x fewer scale L1
        # requests. Each lane then pulls the tile it actually needs with
        # ds_bpermute (lane L from lane t*16 + L%16) -- LDS crossbar, not HBM.
        #
        # A half spans N_TILES*16 rows, so 64 lanes cover it exactly at
        # N_TILES=4 and cover it twice over at N_TILES=2 (BLOCK_C=128). The
        # duplicate half of that load is wasted address bandwidth but stays
        # ONE instruction, and the rows it touches are the next 32 of the same
        # weight plane -- already-warm cache lines, or an in-bounds read the
        # bpermute never selects.
        assert N_TILES_A <= 4 and N_TILES_B <= 4, "a half must fit in 64 lanes"

        sa0_base = row0 + wave_i * fx.Int32(N_TILES_A * 16)
        sa1_base = row0 + fx.Int32(LDS_BLOCK_R) + wave_i * fx.Int32(N_TILES_A * 16)
        sb0_base = wrow0 + wave_j * fx.Int32(N_TILES_B * 16)
        sb1_base = wrow0 + fx.Int32(LDS_BLOCK_C) + wave_j * fx.Int32(N_TILES_B * 16)

        # Only the row term is per-lane; the half base and the K index are
        # wave-uniform and ride the buffer instruction's SGPR soffset, so the
        # address costs no per-step VALU and exactly one VGPR.
        sc_voff = lane_id * fx.Int32(SCALE_I32_ROW)
        sc_sbase = [
            (_rs, _base * fx.Int32(SCALE_I32_ROW * 4))
            for _rs, _base in (
                (sa_rsrc, sa0_base),
                (sa_rsrc, sa1_base),
                (sb_rsrc, sb0_base),
                (sb_rsrc, sb1_base),
            )
        ]
        sc_perm = [
            fx.Int32(t_ * 64) + m_lane * fx.Int32(4)
            for t_ in range_constexpr(max(N_TILES_A, N_TILES_B))
        ]
        # Tiles per half, in the order the chunk stores them: A0, A1, B0, B1.
        sc_tiles = (N_TILES_A, N_TILES_A, N_TILES_B, N_TILES_B)

        def issue_chunk(k_i32):
            """Cooperative scale loads for scale-i32 pair (k_i32, k_i32+1),
            i.e. the two K=128 sub-blocks of ONE 256-element K-step.

            Returns 2 raw i32 per operand-half, flat: [a0_lo, a0_hi, a1_lo, ...].
            A chunk for a K-step past the end reads the next row's scales (or
            out of bounds, which the buffer resource returns as 0) and is never
            consumed -- cheaper than predicating the last prefetch.
            """
            out = []
            for _rs, _sb in sc_sbase:
                soff = _sb + fx.Int32(k_i32 * 4)
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
            vm.issue(N_SC_OPS)
            return out

        def use_chunk(chunk, p):
            """Redistribute + consume the chunk's sub-block `p` (= ksub)
            -> (sa0, sa1, sb0, sb1), each a list of broadcast-i32 per tile."""
            groups = []
            for h_ in range_constexpr(4):
                src = chunk[2 * h_ + p]
                grp = []
                for t_ in range_constexpr(sc_tiles[h_]):
                    v = rocdl.ds_bpermute(res=T.i32, index=sc_perm[t_], src=src)
                    grp.append(((ArithValue(v) >> kshift) & mask_ff) * bcast)
                groups.append(grp)
            return groups

        def use_step(chunk, n_ksub=KSUBS):
            """Both sub-blocks of one K-step: sa[ks][tile], sb[ks][tile]."""
            sa0, sa1, sb0, sb1 = [], [], [], []
            for ks in range_constexpr(n_ksub):
                g0, g1, g2, g3 = use_chunk(chunk, ks)
                sa0.append(g0)
                sa1.append(g1)
                sb0.append(g2)
                sb1.append(g3)
            return sa0, sa1, sb0, sb1

        def mma(a, b, c, sa, sb, n_ksub=KSUBS):
            """``a[tile][ksub]`` / ``b[tile][ksub]`` are i32x4 K=128 operands;
            ``sa[ksub][tile]`` / ``sb[ksub][tile]`` the matching scales."""
            for ks in range_constexpr(n_ksub):
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
            128-byte row. Identical formula to the MXFP8 body's -- the LDS
            geometry is in bytes and did not change; what changed is that
            these two offsets are now two OPERANDS rather than two halves of
            one."""
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
            """Per tile: ``[i32x4_ksub0, i32x4_ksub1]``, NOT packed."""
            swz = _lds_swizzle(s2r)
            return [
                [s2r.load_one(lds_src, swz[t_][ks]) for ks in range_constexpr(KSUBS)]
                for t_ in range_constexpr(s2r.n_tiles)
            ]

        def _cluster_plain(lds_dst, g2s, k_off, s2r, lds_src, a, b, c, sa, sb):
            g2s.load(lds_dst, k_off)
            vm.issue(g2s.n_load_steps)
            rt = s2r_load_fp4(s2r, lds_src)
            c = mma(a, b, c, sa, sb)
            return c, rt

        def _cluster_il(lds_dst, g2s, k_off, s2r, lds_src, a, b, c, sa, sb):
            # Interleave this quadrant's MFMAs with the g2s and s2r loads of
            # the NEXT fragment, so the loads co-issue in the MFMA execute
            # shadow. Mirrors fp8_gemm_4wave's _interleaved_cluster; the MFMA
            # is scaled + AGPR-pinned here. The schedule comes from
            # `_interleave_plan` (computed before tracing), so this walks it
            # with no branches -- see that function on why.
            #
            # fp4 has 2x the MFMAs per cluster and the same number of loads,
            # so every load lands in a wider shadow than it did at fp8.
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

            vm.issue(g2s.n_load_steps)
            return c, rt

        _cluster = _cluster_il if _INTERLEAVE else _cluster_plain

        # ── Prologue: pre-fill the 8-buffer LDS pipeline (2 K-steps) ──
        # Chunks 0 and 1 go first so they are the oldest things in flight and
        # the prologue barriers retire them for free. TWO chunks, where the
        # MXFP8 body needed one: a chunk is one K-step's scales here, and the
        # prefetch horizon is two steps.
        chunks = [None] * N_CHUNKS
        chunks[0] = issue_chunk(0)
        if N_CHUNKS > 1:
            chunks[1] = issue_chunk(2)

        a_g2s.load(a_cur0, A0_gl + 0 * BLOCK_K_BYTES)
        mk_ac0 = vm.issue(N_TILES_A)
        b_g2s.load(b_cur0, B0_gl + 0 * BLOCK_K_BYTES)
        mk_bc0 = vm.issue(N_TILES_B)
        b_g2s.load(b_cur1, B1_gl + 0 * BLOCK_K_BYTES)
        vm.issue(N_TILES_B)
        a_g2s.load(a_cur1, A1_gl + 0 * BLOCK_K_BYTES)
        mk_a = vm.issue(N_TILES_A)

        a_g2s.load(a_next0, A0_gl + 1 * BLOCK_K_BYTES)
        vm.issue(N_TILES_A)
        b_g2s.load(b_next0, B0_gl + 1 * BLOCK_K_BYTES)
        mk_b = vm.issue(N_TILES_B)
        b_g2s.load(b_next1, B1_gl + 1 * BLOCK_K_BYTES)
        vm.issue(N_TILES_B)
        a_g2s.load(a_next1, A1_gl + 1 * BLOCK_K_BYTES)
        mk_c = vm.issue(N_TILES_A)

        vm.wait(mk_ac0)
        a0 = s2r_load_fp4(a_s2r, a_cur0)
        vm.wait(mk_bc0)
        b0 = s2r_load_fp4(b_s2r, b_cur0)

        # ── Main K-steps 0 .. K_ITERS-3, fully unrolled ──
        # Each step consumes the LDS buffers holding step k, loads step k+2
        # into them, and reads step k+1's leading fragments. Which watermark
        # each barrier waits on follows from the ping-pong depth:
        #   * clusters 1-2 read the halves written by step k-2's SECOND half,
        #   * clusters 3-4 read the halves written by step k-1's FIRST half.
        # The two prologue groups stand in for steps -2 and -1: `mk_a` closes
        # the k=0 group (step 0's second-half operands), `mk_b` sits right
        # after b_next0 (step 1's leading fragments) and `mk_c` closes the
        # k=1 group (step 1's second-half operands).
        bufs = (a_cur0, a_cur1, a_next0, a_next1, b_cur0, b_cur1, b_next0, b_next1)
        sec = {-2: mk_a, -1: mk_c}  # second-half watermark, by step index
        fst = {-1: mk_b}  # first-half watermark, by step index

        for kk in range_constexpr(K_ITERS - 2):
            ac0, ac1, an0, an1, bc0, bc1, bn0, bn1 = bufs
            k2 = (kk + 2) * BLOCK_K_BYTES

            # The scales for this step came off HBM two steps ago; the only
            # HBM touch is the prefetch for the step two out -- the same
            # horizon the data ping-pong runs at.
            vm.wait(sec[kk - 2])
            if kk + 2 < N_CHUNKS:
                chunks[kk + 2] = issue_chunk(2 * (kk + 2))
            sa0, sa1, sb0, sb1 = use_step(chunks[kk])

            c00, b1 = _cluster(
                ac0, a_g2s, A0_gl + k2, b_s2r, bc1, a0, b0, c00, sa0, sb0
            )
            c01, a1 = _cluster(
                bc0, b_g2s, B0_gl + k2, a_s2r, ac1, a0, b1, c01, sa0, sb1
            )
            fst[kk] = vm.mark()

            vm.wait(fst[kk - 1])
            c10, a0 = _cluster(
                bc1, b_g2s, B1_gl + k2, a_s2r, an0, a1, b0, c10, sa1, sb0
            )
            c11, b0 = _cluster(
                ac1, a_g2s, A1_gl + k2, b_s2r, bn0, a1, b1, c11, sa1, sb1
            )
            sec[kk] = vm.mark()

            bufs = (an0, an1, ac0, ac1, bn0, bn1, bc0, bc1)

        a_cur0, a_cur1, a_next0, a_next1, b_cur0, b_cur1, b_next0, b_next1 = bufs

        # ── Tail step K_ITERS-2: drain, no more g2s ──
        k_tail0 = K_ITERS - 2
        vm.wait(sec[k_tail0 - 2])
        sa0, sa1, sb0, sb1 = use_step(chunks[k_tail0])
        b1 = s2r_load_fp4(b_s2r, b_cur1)
        c00 = mma(a0, b0, c00, sa0, sb0)
        a1 = s2r_load_fp4(a_s2r, a_cur1)
        c01 = mma(a0, b1, c01, sa0, sb1)
        vm.wait(fst[k_tail0 - 1])
        a0 = s2r_load_fp4(a_s2r, a_next0)
        c10 = mma(a1, b0, c10, sa1, sb0)
        b0 = s2r_load_fp4(b_s2r, b_next0)
        c11 = mma(a1, b1, c11, sa1, sb1)

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # ── Tail step K_ITERS-1 ──
        # When K % 256 == 128 this is a HALF step: the G2S staged a whole
        # 128-byte row (reading 64 bytes into the next row, which the bounded
        # buffer resource clamps on the last one), but only ksub 0 -- the
        # first 64 bytes, K elements 0..127 -- is real, so only it is issued.
        NK_TAIL = 1 if HALF_TAIL else KSUBS
        k_tail1 = K_ITERS - 1
        vm.wait(sec[k_tail1 - 2])
        sa0, sa1, sb0, sb1 = use_step(chunks[k_tail1], NK_TAIL)
        b1 = s2r_load_fp4(b_s2r, b_cur1)
        a1 = s2r_load_fp4(a_s2r, a_cur1)
        c00 = mma(a0, b0, c00, sa0, sb0, NK_TAIL)
        c01 = mma(a0, b1, c01, sa0, sb1, NK_TAIL)
        c10 = mma(a1, b0, c10, sa1, sb0, NK_TAIL)
        c11 = mma(a1, b1, c11, sa1, sb1, NK_TAIL)

        # ── Epilogue: bf16 into out(M, N); the MFMA already applied the scales ──
        # Storing f32 and converting outside would cost a full extra pass over
        # M*N (553 MB at the e2e forward shape) -- more than half the kernel.
        out_m_i = arith.index_cast(T.index, out_m)
        out_n_i = arith.index_cast(T.index, out_n)
        nbytes = arith.index_cast(T.i64, out_m_i * out_n_i * fx.Index(2))
        o_rsrc = buffer_ops.create_buffer_resource(
            OUT, max_size=False, num_records_bytes=nbytes
        )
        base_row = row0 + wave_i * fx.Int32(N_TILES_A * 16)
        base_col = c_base + wave_j * fx.Int32(N_TILES_B * 16)
        oob = out_m * out_n

        def store_group(frag, br, bc):
            for ti in range_constexpr(N_TILES_A):
                row = br + fx.Int32(ti * 16) + k_grp * fx.Int32(4)
                for tj in range_constexpr(N_TILES_B):
                    col = bc + fx.Int32(tj * 16) + m_lane
                    col_ok = col < out_n
                    vec = Vec(frag[ti * N_TILES_B + tj])
                    for e in range_constexpr(4):
                        r_ = row + fx.Int32(e)
                        # `r_ < m_end` is the ragged part. A row tile may
                        # overhang its group, in which case the overhanging
                        # rows hold this expert's weights applied to the
                        # NEXT group's tokens -- correct arithmetic, wrong
                        # expert -- so they must not be stored. `active`
                        # kills whole slots past the last group's tiles.
                        # Reading those A rows is harmless: each output
                        # element is one A row dotted with one B column, so
                        # nothing leaks across rows, and any fp8 NaN in the
                        # pad region past m_total lands only in rows we drop.
                        ok = active & (r_ < m_end) & (r_ < out_m) & col_ok
                        off = arith.select(ok, r_ * out_n + col, oob)
                        buffer_ops.buffer_store(vec[e].to(fx.BFloat16), o_rsrc, off)

        store_group(c00, base_row, base_col)
        store_group(c01, base_row, base_col + fx.Int32(LDS_BLOCK_C))
        store_group(c10, base_row + fx.Int32(LDS_BLOCK_R), base_col)
        store_group(
            c11, base_row + fx.Int32(LDS_BLOCK_R), base_col + fx.Int32(LDS_BLOCK_C)
        )

    @flyc.jit
    def launch_fwd(
        A,
        B,
        OUT,
        A_scale,
        B_scale,
        OFFS,
        n_blocks: fx.Int32,
        n_c_tiles: fx.Int32,
        out_m: fx.Int32,
        out_n: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_fwd(
            A,
            B,
            OUT,
            A_scale,
            B_scale,
            OFFS,
            n_c_tiles,
            out_m,
            out_n,
            value_attrs={
                "rocdl.waves_per_eu": 1,
                "rocdl.flat_work_group_size": "256,256",
            },
        ).launch(grid=(n_blocks, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_fwd


@functools.lru_cache(maxsize=None)
def cached_launch(K: int, N: int, E: int, BLOCK_C: int):
    return _compile(K, N, E, BLOCK_C)
