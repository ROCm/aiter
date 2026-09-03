# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL FMHA backward for MLA-style d_qk=192 / d_v=128, varlen THD, causal, bf16, gfx942.

The MFMA primitive used everywhere is ``v_mfma_f32_16x16x16_bf16_1k``:

    C[16m, 16n] += A[16m, 16k] . B[16n, 16k]^T

with BOTH operands "row-major, K contiguous" (lane l holds row = l%16, k = (l//16)*4 + 0..3)
and the accumulator laid out as (n = l%16, m = (l//16)*4 + 0..3).

That asymmetry is the whole design driver: an accumulator fragment of ``C`` is *bit-identical*
to an A/B operand fragment of ``C^T``.  So a score fragment can be fed straight back into the
next MFMA as its own transpose, for free, and only the operands whose contraction index is the
token index need a real transpose through LDS.

TWO JOB TYPES, ONE LAUNCH (``k_bwd``) -- no atomics, fully deterministic:

  job A  (grid.y < nb1)   one workgroup per (key block of 128, sequence, head); streams query
         tiles of 32.  Computes  S[i,j] = Q.K^T  and  dP[i,j] = dO.V^T  (both contract over d,
         so both operands come straight out of memory), then reuses the S / dS fragments as the
         transposed operands for  dV = P^T.dO  and  dK = dS^T.Q  (which contract over i, so
         dO^T and Q^T are staged transposed in LDS).

  job B  (grid.y >= nb1)  one workgroup per (query block of 128, sequence, head); streams key
         tiles of 32.  Computes the scores TRANSPOSED (S^T[j,i] = K.Q^T) so that the dS^T
         fragment is, for free, the dS[i,j] operand that  dQ = dS.K  needs; K^T staged in LDS.

  pre    ``k_delta``  D = rowsum(dO * O), fp32, laid out [H, T] like lse.

WHY ONE LAUNCH AND WHY THIS GRID ORDER -- the two structural levers of this file:

 1. ``grid = (nseq*Hn, nb1 + nb2, 1)`` with the (sequence, head) pair on the FASTEST axis.  The
    command processor dispatches x fastest and is work-conserving, so emitting every
    (seq, head)'s block 0 -- the longest causal runs -- before any block 1 turns the hardware
    dispatcher into a largest-processing-time-first scheduler.  The original
    ``(block, seq, head)`` order emitted sequence 0's 4-tile tail before sequence 1's 114-tile
    head and the causal triangle set the makespan.

 2. Both job types share ONE grid.  Each job type alone has a ~25-30 % drain: its longest
    workgroup streams 114 / 146 tiles while the balanced average is 85 / 95, and measured wave
    residency was 72 % / 77 % of the 2-waves-per-SIMD ceiling.  Two separate dispatches cannot
    overlap (same queue = implicit barrier), and putting them on two streams does NOT help
    either -- both drains are perfectly correlated, so they just share the idle.  In one grid
    the dQ jobs are dispatched AFTER the dK/dV jobs, so they land exactly in the dK/dV drain:
    makespan becomes max(total_work/CUs, longest_single_job) instead of the sum of the two
    maxima.  The dQ block index is reversed (job B's work grows with the query block index) so
    both halves of the list stay descending.

The LDS arena is shared by the two branches (they are mutually exclusive), so the merged kernel
costs the same 44.3 KB / workgroup as job A alone.

Bounds: every global tensor is wrapped in a buffer resource with exact ``num_records``, so
out-of-range loads return 0 and masked stores are dropped by the hardware.
"""

import functools
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu, range_constexpr, rocdl
from flydsl.expr.arith import CmpIPredicate
from flydsl.expr.primitive import const_expr
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops, vector

DQK = 192
DV = 128
LOG2E = 1.4426950408889634

ROWS_DELTA = 32  # rows reduced per delta workgroup (2 unrolled 16-row passes)
NT = 512  # threads per workgroup (8 waves -> 2 per SIMD; LDS caps the CU at one WG)

BN1 = 128  # job A: keys owned by a workgroup (16 per wave)
BM1 = 32  # job A: queries streamed per iteration
LD1 = BM1 + 4  # padded leading dim of the transposed LDS tiles

BM2 = 128  # job B: queries owned by a workgroup (16 per wave)
BN2 = 32  # job B: keys streamed per iteration
LD2 = BN2 + 4

# Row strides of the "natural" LDS tiles.  They must NOT be a multiple of 32 dwords: an MFMA
# A-operand read has the 16 lanes of a row group reading 16 consecutive ROWS at the same column,
# so a stride of 192 elements (= 96 dwords = 3*32) puts all 16 lanes on one bank -- a 16-way
# conflict that dominated the whole kernel.  +4 elements makes the dword stride = 2 (mod 4),
# which is conflict-optimal for a 64-lane b64 read while keeping the 8-byte alignment b64 needs.
LDQ = DQK + 4
LDVN = DV + 4

NK_D = DQK // 16  # 12 MFMA tiles along d_qk
NV_D = DV // 16  # 8 MFMA tiles along d_v

# ---- shared bf16 LDS arena; job A and job B are mutually exclusive so they alias -------------
A_QN = 0  # job A: Q natural   [BM1, LDQ]
A_QT = A_QN + BM1 * LDQ  # job A: Q^T         [DQK, LD1]
A_ON = A_QT + DQK * LD1  # job A: dO natural  [BM1, LDVN]
A_OT = A_ON + BM1 * LDVN  # job A: dO^T        [DV, LD1]
A_END = A_OT + DV * LD1

B_KN = 0  # job B: K natural   [BN2, LDQ]
B_KT = B_KN + BN2 * LDQ  # job B: K^T         [DQK, LD2]
B_VN = B_KT + DQK * LD2  # job B: V natural   [BN2, LDVN]
B_END = B_VN + BN2 * LDVN

ARENA = max(A_END, B_END)

# ---- VALU-reduction ablation knobs (the measured winner is the default) -----------------------
# Measured (main case, 10 warmup / 100 repeats, 3 interleaved runs each, same session):
#   none 646.0/632.5/633.9  amask 636.3/638.7/642.2  bpeel 646.7/638.2/650.4
#   amask+bpeel 622.9/620.9/618.0  <-- every run below every run of every other config
# NEITHER half wins alone; together they do (-2.7 % on k_bwd's main-case time).
# `ascale` (LOG2E onto the LDS lse + `scale` onto the dK epilogue) is a MEASURED LOSS
# (+3.6 % on the main case) -- it trades 8 loop muls for 48 epilogue muls in the serial tail
# and LLVM had already packed the loop muls into v_pk_mul_f32.  Kept as a re-checkable
# ablation; do not enable.
_OPTS = {
    x
    for x in os.environ.get("AITER_FMHA_BWD_FLYDSL_OPT", "amask,bpeel").split(",")
    if x
}
OPT_AMASK = "amask" in _OPTS  # job A: single unsigned compare instead of 2 signed + and
OPT_ASCALE = (
    "ascale" in _OPTS
)  # job A: LOG2E onto the LDS lse, `scale` onto the dK epilogue
OPT_BPEEL = (
    "bpeel" in _OPTS
)  # job B: peel the causal-diagonal trips out of the streaming loop


# --------------------------------------------------------------------------- small helpers
def _mfma(a, b, acc):
    ai = vector.bitcast(T.vec(4, T.i16), a)
    bi = vector.bitcast(T.vec(4, T.i16), b)
    return rocdl.mfma_f32_16x16x16bf16_1k(T.vec(4, T.f32), [ai, bi, acc, 0, 0, 0])


def _zero4():
    return arith.constant_vector(0.0, T.vec(4, T.f32))


def _get(vec4, i):
    return fx.Float32(vector.extract(vec4, static_position=[i], dynamic_position=[]))


def _pack4(vals):
    """4 fp32 -> one v4bf16 MFMA operand fragment.

    Truncating pack (drop the low mantissa half), the same trick the production FlyDSL flash
    attention forward uses for its P operand: ~1.5 VALU ops per element instead of the ~6 that
    a generic ``arith.truncf f32 -> bf16`` expands to on gfx942 (shift / round / NaN fixup).
    """
    c_mask = fx.Int32(0xFFFF0000)

    def _pair(a, b):
        ai = fx.Float32(a).bitcast(fx.Int32)
        bi = fx.Float32(b).bitcast(fx.Int32)
        return ((bi & c_mask) | ai.shrui(fx.Int32(16))).ir_value()

    packed = vector.from_elements(
        T.vec(2, T.i32), [_pair(vals[0], vals[1]), _pair(vals[2], vals[3])]
    )
    return vector.bitcast(T.vec(4, T.bf16), packed)


# A byte offset far past any buffer's num_records but nowhere near the i32 wrap, so a masked
# store stays out of range even after the constant `soffset` of the widest d-tile is added.
_OOB = 0x40000000


def _cvt1(v):
    """fp32 -> bf16 in TWO VALU ops (add + shift), for the epilogue's scalar stores.

    ``fx.Float32(...).to(fx.BFloat16)`` lowers to the generic ``arith.truncf f32->bf16``, which
    gfx942 expands into round-to-nearest-even PLUS a NaN fixup branch (``v_bfe_u32`` +
    ``v_add3_u32`` + ``v_cmp_u_f32`` + 2x ``v_cndmask_b32``), ~8.5 VALU per output element.  The
    epilogue paid that 80x (job A) / 48x (job B) per thread, after the last MFMA, with nothing to
    overlap it: 677 and 407 VALU of pure serial tail per workgroup.

    Here we round-to-nearest by adding 0x8000 to the fp32 bit pattern -- the sign bit survives the
    truncation untouched, so a magnitude carry is exactly round-half-away-from-zero -- and drop
    the NaN fixup (these accumulators are finite by construction).
    """
    bits = fx.Float32(v).bitcast(fx.Int32) + fx.Int32(0x00008000)
    hi = bits.shrui(fx.Int32(16)).ir_value()
    return arith.bitcast(T.bf16, arith.trunci(T.i16, hi))


def _rsrc(tensor, nbytes):
    """Buffer resource with an EXACT num_records: OOB loads return 0, masked stores are dropped.

    (The JIT gives tensors a dynamic-shaped memref, so ``max_size=False`` cannot infer the size
    on its own -- we hand it the byte count computed from the runtime dims.)"""
    return buffer_ops.create_buffer_resource(
        tensor, num_records_bytes=fx.Int32(nbytes).ir_value()
    )


def _bf16x4(ptr):
    return fx.ptr_load(ptr, result_type=fx.Vector.make_type(4, fx.BFloat16))


def _packNr(vals):
    """2N fp32 -> one v(2N)bf16, round-to-nearest (add 0x8000 before the truncating pack).

    Same 2-op-per-element trick as ``_cvt1`` but producing a wide packed vector, for the split-K
    reduction kernel's vectorised bf16 stores.
    """
    c_mask = fx.Int32(0xFFFF0000)
    rnd = fx.Int32(0x00008000)

    def _pair(a, b):
        ai = fx.Float32(a).bitcast(fx.Int32) + rnd
        bi = fx.Float32(b).bitcast(fx.Int32) + rnd
        return ((bi & c_mask) | ai.shrui(fx.Int32(16))).ir_value()

    npair = len(vals) // 2
    packed = vector.from_elements(
        T.vec(npair, T.i32),
        [_pair(vals[2 * i], vals[2 * i + 1]) for i in range(npair)],
    )
    return vector.bitcast(T.vec(2 * npair, T.bf16), packed)


NT_RED = 256  # threads per split-K reduction workgroup
VEC_RED = 8  # bf16 lanes per thread (one dwordx4 per partial slab)


@functools.lru_cache(maxsize=4)
def build(nsp=1):
    """Compile the kernel set.  ``nsp`` = split-K factor along the STREAMED index.

    ``nsp == 1`` is the two-kernel path (k_delta + k_bwd writing bf16 straight into dq/dk/dv)
    and every Python-level branch below is written so that this variant emits exactly the code
    it emitted before split-K existed.

    ``nsp > 1`` compiles a SECOND, independently cached variant used only for small workloads
    (see ``_split()`` in fmha_bwd_kernel.py).  WHY: with 272 real workgroups on 304 CUs every
    workgroup is co-resident, so the kernel's makespan is the LONGEST SINGLE WORKGROUP, not the
    total work.  On a T=8192 / H=2 / 5-sequence case that is job A's block 0 of the 2200-token
    sequence -- 69 streamed query tiles -- while the work-balanced average is only 23.8 tiles
    per CU: a 2.9x imbalance that no dispatch ordering and no tile-size change can touch (the
    block that owns the FIRST keys must contract over every query, whatever its width).  The
    only way to shorten it is to cut the contraction itself: split the streamed range into
    ``nsp`` chunks handled by ``nsp`` different workgroups, each writing a partial, and add a
    trivial fully-coalesced elementwise reduction kernel.  ``k_bwd``'s signature is unchanged --
    the partial workspaces are handed in through the dq/dk/dv slots and the per-split stride is
    derived from ``Tlen*Hn``, so no extra kernel argument perturbs the nsp == 1 variant.
    """

    # ----------------------------------------------------------------- LDS storage layouts
    @fx.struct
    class SmemBwd:
        arena: fx.Array[fx.BFloat16, ARENA, 16]
        lse_s: fx.Array[fx.Float32, BM1, 16]
        del_s: fx.Array[fx.Float32, BM1, 16]

    # ================================================================= delta = rowsum(dO*O)
    # 16 threads cooperate on one 128-wide row so the global reads stay fully coalesced
    # (256 threads x 16 B = one contiguous 4 KB run per tensor per block).
    NT_D = 256
    ROWS_D = (
        NT_D // 16
    )  # rows covered by one 256-thread pass (16 threads per 128-wide row)
    UNR_D = (
        ROWS_DELTA // ROWS_D
    )  # passes per workgroup -> ROWS_DELTA rows per workgroup

    @flyc.kernel(known_block_size=[NT_D, 1, 1])
    def k_delta(
        DO: fx.Tensor, O: fx.Tensor, DEL: fx.Tensor, Tlen: fx.Int32, Hn: fx.Int32
    ):
        """D = rowsum(dO * O).

        The 16 partial sums of a row live in 16 CONSECUTIVE LANES of one wave, so the reduction
        is a pure cross-lane XOR butterfly (4 shuffles): no LDS array, no ``gpu.barrier()``, and
        none of the 73.8 % bank-conflict rate the LDS round-trip used to carry.  The row groups
        are also unrolled UNR_D-deep with every global load issued up front, so each thread keeps
        2*UNR_D VMEM requests in flight -- this kernel is pure streaming bandwidth and was 89 %
        dependency-wait.
        """
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        nrow = Tlen * Hn
        rdo = _rsrc(DO, nrow * (DV * 2))
        ro = _rsrc(O, nrow * (DV * 2))
        rd = _rsrc(DEL, nrow * 4)

        off = bid * (ROWS_DELTA * DV) + tid * 8
        av = [
            buffer_ops.buffer_load(
                rdo, off, vec_width=8, dtype=T.bf16, soffset_bytes=u * ROWS_D * DV * 2
            )
            for u in range_constexpr(UNR_D)
        ]
        bv = [
            buffer_ops.buffer_load(
                ro, off, vec_width=8, dtype=T.bf16, soffset_bytes=u * ROWS_D * DV * 2
            )
            for u in range_constexpr(UNR_D)
        ]

        lsub = tid % 16
        grow = tid // 16
        for u in range_constexpr(UNR_D):
            e0 = fx.Float32(0.0)
            e1 = fx.Float32(0.0)
            for c in range_constexpr(4):
                a0 = fx.Float32(
                    vector.extract(av[u], static_position=[2 * c], dynamic_position=[])
                )
                b0 = fx.Float32(
                    vector.extract(bv[u], static_position=[2 * c], dynamic_position=[])
                )
                a1 = fx.Float32(
                    vector.extract(
                        av[u], static_position=[2 * c + 1], dynamic_position=[]
                    )
                )
                b1 = fx.Float32(
                    vector.extract(
                        bv[u], static_position=[2 * c + 1], dynamic_position=[]
                    )
                )
                e0 = e0 + a0 * b0
                e1 = e1 + a1 * b1
            acc = e0 + e1
            for s in range_constexpr(4):
                acc = acc + acc.shuffle_xor(1 << s, 64)
            idx = bid * ROWS_DELTA + u * ROWS_D + grow
            ok = (lsub == 0) & (idx < nrow)
            t = idx // Hn
            h = idx % Hn
            buffer_ops.buffer_store(
                acc.ir_value(), rd, h * Tlen + t, mask=ok.ir_value()
            )

    @flyc.jit
    def launch_delta(
        DO: fx.Tensor,
        O: fx.Tensor,
        DEL: fx.Tensor,
        Tlen: fx.Int32,
        Hn: fx.Int32,
        nblk: fx.Int32,
        stream: fx.Stream,
    ):
        k_delta(DO, O, DEL, Tlen, Hn).launch(
            grid=(nblk, 1, 1), block=(NT_D, 1, 1), stream=stream
        )

    # ==================================================== the fused-schedule backward kernel
    @flyc.kernel(known_block_size=[NT, 1, 1])
    def k_bwd(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        DO: fx.Tensor,
        LSE: fx.Tensor,
        DEL: fx.Tensor,
        CU: fx.Tensor,
        DQO: fx.Tensor,
        DKO: fx.Tensor,
        DVO: fx.Tensor,
        Tlen: fx.Int32,
        Hn: fx.Int32,
        nseq: fx.Int32,
        nb1: fx.Int32,
        nbtot: fx.Int32,
        scale: fx.Float32,
        ilv: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        lane = tid % 64
        wv = tid // 64
        lm = lane % 16
        lk = (lane // 16) * 4

        shx = fx.Int32(fx.block_idx.x)
        sq = shx % nseq
        hh = shx // nseq
        yy = fx.Int32(fx.block_idx.y)

        nrow = Tlen * Hn
        rq = _rsrc(Q, nrow * (DQK * 2))
        rk = _rsrc(K, nrow * (DQK * 2))
        rv = _rsrc(V, nrow * (DV * 2))
        rdo = _rsrc(DO, nrow * (DV * 2))
        rlse = _rsrc(LSE, nrow * 4)
        rdel = _rsrc(DEL, nrow * 4)
        rcu = _rsrc(CU, (nseq + 1) * 4)
        # For nsp > 1 the three output slots are PARTIAL workspaces: nsp slabs of dq / dk / dv.
        # The partials are bf16, not fp32.  Measured: fp32 slabs cost 20.8 us in the reduction
        # (84 MB at 4.0 TB/s -- already at the bandwidth roof) plus ~9 us of extra store traffic
        # inside k_bwd; halving the element size halves both.  The accuracy cost is one extra
        # bf16 rounding on nsp values that are then summed in fp32: measured mean_rel vs the
        # fp32 reference stays at 3.4e-3 (gate 1.5e-2, and the frozen ASM baseline itself is at
        # 2.2e-3).
        rdq = _rsrc(DQO, nrow * (DQK * 2 * nsp))
        rdk = _rsrc(DKO, nrow * (DQK * 2 * nsp))
        rdv = _rsrc(DVO, nrow * (DV * 2 * nsp))
        if const_expr(nsp > 1):
            wst_q = nrow * DQK  # elements per partial slab of dq / dk
            wst_v = nrow * DV  # elements per partial slab of dv

        lo = fx.Int32(buffer_ops.buffer_load(rcu, sq, vec_width=1, dtype=T.i32))
        hi = fx.Int32(buffer_ops.buffer_load(rcu, sq + 1, vec_width=1, dtype=T.i32))
        ln = hi - lo

        sclog = scale * fx.Float32(LOG2E)

        lds = fx.SharedAllocator().allocate(SmemBwd).peek()
        p_ar = lds.arena.ptr
        p_ls = lds.lse_s.ptr
        p_de = lds.del_s.ptr

        srow = tid // 16
        ssub = (tid % 16) * 4
        lrow = tid % BM1

        # ---- XOR swizzle of the TRANSPOSED LDS tiles (Q^T, dO^T, K^T) ---------------------
        # The transposed tiles are FILLED by scalar ds_write_b16: adjacent lanes step the tile
        # ROW index d by 4, so the lane->dword stride is 2*LD1 = 72 == 8 (mod 32) and the 16
        # lanes of a row group land on only FOUR banks -- a 4-way conflict paid 12x (Q^T) + 8x
        # (dO^T) per thread per job-A iteration, 12x (K^T) in job B.  No padding can fix it (the
        # stride is 2*LD1 dwords, hence always even), which is why three rounds of +4 tweaks
        # moved SQ_LDS_BANK_CONFLICT by exactly zero bits.
        #
        # So permute the TOKEN axis instead, at the 4-element granularity of the ds_read_b64
        # that consumes the tile (the read granule stays contiguous and 8-byte aligned):
        #
        #     element(d, tok)  ->  d*LD1 + ((SW(d) ^ (tok>>2)) << 2) + (tok & 3)
        #     SW(d) = (d >> 4) & 7
        #
        # SW keys on bit 4 and up of d ON PURPOSE.  A read fragment covers d = dt*16 + lm with
        # lm = lane%16, so SW is CONSTANT across the 16 lanes of a read -- the transform is a
        # pure permutation of which lane reads which of the four 4-token groups, the ADDRESS
        # MULTISET of every ds_read_b64 is bit-identical to the unswizzled one, and the reads
        # therefore cannot regress.  A store, in contrast, covers d = ssub + ac*64 + ae with
        # ssub = (tid%16)*4, so SW = ((tid%16)>>2 + 4*ac) & 7 varies across the storing lanes
        # and spreads them over 16 banks (conflict-free) instead of 4.
        #
        # (Swizzling on (d>>2)&7 instead -- the "obvious" choice, which also de-conflicts the
        # stores -- was built and measured: SQ_LDS_BANK_CONFLICT/SQ_LDS_IDX_ACTIVE went UP,
        # 21.113% -> 26.30%, and the kernel got 3% slower.  It varies SW within a read fragment,
        # which breaks the reads by more than it fixes the writes.  Do not re-try it.)
        #
        # Every operand of the transform is a per-thread constant, hoisted here:
        _sp = (tid % 16) >> 2
        swz_w0 = ((_sp ^ (srow >> 2)) << 2) + (
            srow & 3
        )  # store column, even d-chunk (ac)
        swz_w1 = swz_w0 ^ 16  # store column, odd  d-chunk (ac)
        _lu4 = (lane >> 4) << 2
        _swz_r = [_lu4 ^ (k << 2) for k in range_constexpr(4)]

        def _wcol(c):
            """Swizzled token-column of a staging store of d-chunk `c` (c = the ac/bc index)."""
            return swz_w1 if (c & 1) else swz_w0

        def _tcol(dt, tt):
            """Swizzled token-column of an MFMA read of tile row block `dt`, token block `tt`."""
            return _swz_r[dt & 3] + 16 * (((dt >> 2) & 1) ^ tt)

        # ---- GRID DECODE: MERGED-LPT interleave of the two job types -------------------------
        # The alternative decode is a CONCATENATION: yy in [0, nb1) = job A descending, then
        # yy in [nb1, nbtot) = job B descending.  Because the command processor dispatches in
        # linear order and only ~304 workgroups are resident, that means job A runs essentially
        # to completion before job B starts.  Measured on the main case (nb1=nb2=29, 1508 WGs):
        # job A alone 366.3 us, job B alone 287.8 us, both together 646.2 us == the SUM.  The
        # "dQ fills the dK/dV drain" effect the concatenated order was designed for is worth
        # only 1.2 %: the drain is tiny next to 2.5 dispatch rounds of job A.
        #
        # Each job type's per-block work is an arithmetic sequence with the SAME slope:
        #   job A block b   -> (ln - b*BN1)/BM1     iterations
        #   job B block bb  -> min(bb*BM2+BM2, ln)/BN2 iterations
        # both decreasing by BN1/BM1 == BM2/BN2 == 4 per index step.  So the true
        # longest-processing-time-first order over the UNION is a 1:1 interleave of the two
        # descending lists, which is exactly what this decode emits:
        #   yy even -> job A block yy/2 ; yy odd -> job B block (nb2-1 - yy/2)
        # The machine then always holds a mix of both job types, and the makespan drops from
        # (max_A + max_B) towards max(total_work/CUs, longest_single_WG).
        #
        # (The tail handles nb1 != nb2 -- not reachable with max_seqlen_q == max_seqlen_k, but
        # the kernel must stay general.)
        # The interleave is HOST-GATED (`ilv`): it wins only when the whole grid is resident in
        # ~1 dispatch round, where the order fixes which workgroups are co-resident.  Once the
        # grid is several rounds deep the kernel is work-bound and mixing the two job types just
        # widens the working set -- measured on a uniform 32K case (1024 WGs): 724.5 -> 738.4 us
        # with the interleave forced on, vs 217.0 -> 205.8 us on the small case.  See
        # `_interleave()` in fmha_bwd_kernel.py for the CU-count-based predicate.
        nb2v = nbtot - nb1
        mm = (nb1 < nb2v).select(nb1, nb2v)
        half = yy // 2
        tail = yy - mm * 2
        inpair = yy < mm * 2
        on = ilv > 0
        is_a = on.select(
            inpair.select(
                ((yy % 2) == 0).select(fx.Int32(1), fx.Int32(0)),
                (nb1 > nb2v).select(fx.Int32(1), fx.Int32(0)),
            ),
            (yy < nb1).select(fx.Int32(1), fx.Int32(0)),
        )
        ablk = on.select(inpair.select(half, mm + tail), yy)
        bblk = on.select(
            inpair.select(nb2v - 1 - half, nb2v - 1 - mm - tail), (nbtot - 1) - yy
        )

        if const_expr(nsp > 1):
            # The host multiplied nb1/nbtot by nsp, so the decoded list index carries the split
            # index in its low digit.  All nsp splits of one block cost the same, so `blk // nsp`
            # keeps both job lists in exactly the descending order the merged-LPT decode assumes.
            asp = ablk % nsp
            ablk = ablk // nsp
            bsp = bblk % nsp
            bblk = bblk // nsp

        if is_a > 0:
            # ---------------------------------------------------------------- job A: dK / dV
            p_qn = p_ar + A_QN
            p_qt = p_ar + A_QT
            p_on = p_ar + A_ON
            p_ot = p_ar + A_OT

            aj0 = ablk * BN1

            # EMPTY-WORKGROUP EARLY EXIT.  The grid is sized on max_seqlen, so 30 % (main case)
            # and 57.4 % (uniform case) of the launched workgroups own a tile range that lies
            # past the end of their sequence: they run ZERO inner-loop iterations and every
            # epilogue store is masked off, yet they still paid the full prologue operand loads
            # and the whole serial epilogue.  Skip all of it.
            if aj0 < ln:
                # --- K / V operand fragments for this wave's 16 keys, register-resident -------
                ajl = wv * 16 + lm  # key row inside the block
                akrow = lo + aj0 + ajl
                # The 12 K / 8 V fragment loads differ only by a CONSTANT d-tile stride, so the
                # whole vgpr address is computed once and the stride rides in the scalar
                # `soffset` field -- 20 v_add_u32 of prologue address math deleted.
                akb = (akrow * Hn + hh) * DQK + lk
                avb = (akrow * Hn + hh) * DV + lk
                akf = []
                for ac in range_constexpr(NK_D):
                    akf.append(
                        buffer_ops.buffer_load(
                            rk, akb, vec_width=4, dtype=T.bf16, soffset_bytes=ac * 32
                        )
                    )
                avf = []
                for ac in range_constexpr(NV_D):
                    avf.append(
                        buffer_ops.buffer_load(
                            rv, avb, vec_width=4, dtype=T.bf16, soffset_bytes=ac * 32
                        )
                    )

                # ---- software-pipelined staging ----------------------------------------------
                # The five global tiles for the NEXT query block are issued right after the LDS
                # barrier of the current one, so their ~600-cycle VMEM latency is covered by the
                # current block's 1280 cycles of MFMA instead of being exposed at the top of the
                # loop.  Dependency-wait was 38 % of wave time before this.
                def _ld_a(ii0):
                    r = []
                    for ac in range_constexpr(DQK // 64):
                        r.append(
                            buffer_ops.buffer_load(
                                rq,
                                ((lo + ii0 + srow) * Hn + hh) * DQK + ssub + ac * 64,
                                vec_width=4,
                                dtype=T.bf16,
                            )
                        )
                    for ac in range_constexpr(DV // 64):
                        r.append(
                            buffer_ops.buffer_load(
                                rdo,
                                ((lo + ii0 + srow) * Hn + hh) * DV + ssub + ac * 64,
                                vec_width=4,
                                dtype=T.bf16,
                            )
                        )
                    r.append(
                        buffer_ops.buffer_load(
                            rlse,
                            hh * Tlen + lo + ii0 + lrow,
                            vec_width=1,
                            dtype=T.f32,
                        )
                    )
                    r.append(
                        buffer_ops.buffer_load(
                            rdel,
                            hh * Tlen + lo + ii0 + lrow,
                            vec_width=1,
                            dtype=T.f32,
                        )
                    )
                    return r

                NQC = DQK // 64
                NOC = DV // 64
                NPRE = NQC + NOC + 2
                if const_expr(nsp == 1):
                    a_lo = aj0
                    a_hi = ln
                else:
                    # this block's full streamed range is [aj0, ln); take chunk `asp` of nsp,
                    # cut on BM1-tile boundaries so the causal mask logic is untouched.
                    a_nt = (ln - aj0 + (BM1 - 1)) // BM1
                    a_lo = aj0 + ((a_nt * asp) // nsp) * BM1
                    a_hx = aj0 + ((a_nt * (asp + 1)) // nsp) * BM1
                    a_hi = (a_hx < ln).select(a_hx, ln)
                ainit = [_zero4() for _ in range_constexpr(NK_D + NV_D)] + _ld_a(a_lo)
                a_out = ainit
                for ai0_, ast in range(a_lo, a_hi, BM1, init=ainit):
                    ai0 = fx.Int32(ai0_)
                    dk_acc = [ast[ac] for ac in range_constexpr(NK_D)]
                    dv_acc = [ast[NK_D + ac] for ac in range_constexpr(NV_D)]
                    apre = [ast[NK_D + NV_D + ac] for ac in range_constexpr(NPRE)]

                    gpu.barrier()
                    for ac in range_constexpr(NQC):
                        adc = ssub + ac * 64
                        avq = apre[ac]
                        fx.ptr_store(avq, p_qn + (srow * LDQ + adc))
                        for ae in range_constexpr(4):
                            fx.ptr_store(
                                vector.extract(
                                    avq, static_position=[ae], dynamic_position=[]
                                ),
                                p_qt + ((adc + ae) * LD1 + _wcol(ac)),
                            )
                    for ac in range_constexpr(NOC):
                        adc = ssub + ac * 64
                        avo = apre[NQC + ac]
                        fx.ptr_store(avo, p_on + (srow * LDVN + adc))
                        for ae in range_constexpr(4):
                            fx.ptr_store(
                                vector.extract(
                                    avo, static_position=[ae], dynamic_position=[]
                                ),
                                p_ot + ((adc + ae) * LD1 + _wcol(ac)),
                            )
                    if const_expr(OPT_ASCALE):
                        # LOG2E folded onto lse ONCE per staged row (1 VALU/thread/iteration)
                        # instead of once per score element (8/thread/iteration).
                        fx.ptr_store(
                            (
                                fx.Float32(apre[NQC + NOC]) * fx.Float32(LOG2E)
                            ).ir_value(),
                            p_ls + lrow,
                        )
                    else:
                        fx.ptr_store(apre[NQC + NOC], p_ls + lrow)
                    fx.ptr_store(apre[NQC + NOC + 1], p_de + lrow)
                    gpu.barrier()
                    anxt = _ld_a(ai0 + BM1)  # prefetch for the next iteration

                    ajg = aj0 + ajl  # this lane's key column (C-frag n index)
                    # per-iteration base of the unsigned causal+tail test (see OPT_AMASK)
                    adbase = (ai0 + lk) - ajg
                    alnd = ln - ajg  # loop-invariant; LICM hoists it
                    for ait in range_constexpr(BM1 // 16):
                        s_acc = _zero4()
                        for ac in range_constexpr(NK_D):
                            s_acc = _mfma(
                                _bf16x4(p_qn + ((ait * 16 + lm) * LDQ + ac * 16 + lk)),
                                akf[ac],
                                s_acc,
                            )
                        p_acc = _zero4()
                        for ac in range_constexpr(NV_D):
                            p_acc = _mfma(
                                _bf16x4(p_on + ((ait * 16 + lm) * LDVN + ac * 16 + lk)),
                                avf[ac],
                                p_acc,
                            )

                        alse4 = fx.ptr_load(
                            p_ls + (ait * 16 + lk),
                            result_type=fx.Vector.make_type(4, fx.Float32),
                        )
                        adel4 = fx.ptr_load(
                            p_de + (ait * 16 + lk),
                            result_type=fx.Vector.make_type(4, fx.Float32),
                        )

                        apv = []
                        adsv = []
                        for aii in range_constexpr(4):
                            if const_expr(OPT_AMASK):
                                # ONE unsigned compare replaces (aig < ln) & (ajg <= aig):
                                #   u32(aig - ajg) < u32(ln - ajg)
                                # is exactly `ajg <= aig < ln` whenever ajg <= ln.  For ajg > ln
                                # the identity can fire spuriously, but that lane's key row is
                                # past the sequence and its dK/dV store is dropped by the
                                # buffer resource in the epilogue, so the value is never read.
                                adg = arith.cmpi(
                                    CmpIPredicate.ult,
                                    (adbase + (ait * 16 + aii)).ir_value(),
                                    alnd.ir_value(),
                                )
                            else:
                                aig = ai0 + ait * 16 + lk + aii
                                adg = (aig < ln) & (ajg <= aig)
                            if const_expr(OPT_ASCALE):
                                # lse already carries LOG2E (folded at the LDS store): one v_fma.
                                ax = _get(s_acc, aii) * sclog - _get(alse4, aii)
                            else:
                                ax = (
                                    _get(s_acc, aii) * scale - _get(alse4, aii)
                                ) * fx.Float32(LOG2E)
                            apx = fx.Float32(rocdl.exp2(T.f32, ax.ir_value()))
                            if const_expr(OPT_AMASK):
                                ap = fx.Float32(
                                    arith.select(
                                        adg, apx.ir_value(), fx.Float32(0.0).ir_value()
                                    )
                                )
                            else:
                                ap = fx.Float32(adg.select(apx, fx.Float32(0.0)))
                            if const_expr(OPT_ASCALE):
                                # `scale` rides on the dK epilogue instead (dK = dS^T.Q is
                                # linear in it; dV = P^T.dO must NOT be scaled).
                                ads = ap * (_get(p_acc, aii) - _get(adel4, aii))
                            else:
                                ads = ap * (_get(p_acc, aii) - _get(adel4, aii)) * scale
                            apv.append(ap)
                            adsv.append(ads)
                        apf = _pack4(apv)  # == P^T operand (row = j, k = i)
                        adsf = _pack4(adsv)  # == dS^T operand (row = j, k = i)

                        for adt in range_constexpr(NV_D):
                            ab = _bf16x4(
                                p_ot + ((adt * 16 + lm) * LD1 + _tcol(adt, ait))
                            )
                            dv_acc[adt] = _mfma(apf, ab, dv_acc[adt])
                        for adt in range_constexpr(NK_D):
                            ab = _bf16x4(
                                p_qt + ((adt * 16 + lm) * LD1 + _tcol(adt, ait))
                            )
                            dk_acc[adt] = _mfma(adsf, ab, dk_acc[adt])

                    a_out = yield dk_acc + dv_acc + anxt

                # --- epilogue: C fragments are (m = key row, n = d) --------------------------
                # The 20 stores of one `aii` differ ONLY by the constant d-tile stride
                # (16 elements = 32 B), so the whole address goes in ONE vgpr and the stride
                # rides in the scalar `soffset` field: no per-store v_add and no per-store
                # v_cndmask (the bounds mask is folded into that single vgpr up front by
                # steering it to an address 1 GB past num_records, which the buffer resource
                # drops in hardware).  That removes 160 of the epilogue's VALU ops.
                for aii in range_constexpr(4):
                    ajgs = aj0 + wv * 16 + lk + aii
                    aok = ajgs < ln
                    aorow = lo + ajgs
                    if const_expr(nsp == 1):
                        adkb = aok.select(((aorow * Hn + hh) * DQK + lm) * 2, _OOB)
                        advb = aok.select(((aorow * Hn + hh) * DV + lm) * 2, _OOB)
                    else:
                        # Partial into slab `asp`.  Every (token, d, split) slot is written by
                        # exactly one workgroup, so the workspace needs no zeroing.
                        adkb = aok.select(
                            (asp * wst_q + (aorow * Hn + hh) * DQK + lm) * 2, _OOB
                        )
                        advb = aok.select(
                            (asp * wst_v + (aorow * Hn + hh) * DV + lm) * 2, _OOB
                        )
                    for adt in range_constexpr(NK_D):
                        buffer_ops.buffer_store(
                            _cvt1(_get(a_out[adt], aii)),
                            rdk,
                            adkb,
                            offset_is_bytes=True,
                            soffset_bytes=adt * 32,
                        )
                    for adt in range_constexpr(NV_D):
                        buffer_ops.buffer_store(
                            _cvt1(_get(a_out[NK_D + adt], aii)),
                            rdv,
                            advb,
                            offset_is_bytes=True,
                            soffset_bytes=adt * 32,
                        )
        else:
            # ------------------------------------------------------------------- job B: dQ
            p_kn = p_ar + B_KN
            p_kt = p_ar + B_KT
            p_vn = p_ar + B_VN

            bbx = bblk  # reversed: job B's cost grows with the query block index
            bi0 = bbx * BM2
            # EMPTY-WORKGROUP EARLY EXIT -- see the job-A note.
            if bi0 < ln:

                # --- Q / dO operand fragments for this wave's 16 queries, register-resident ---
                bil = wv * 16 + lm
                big = bi0 + bil
                bqrow = lo + big
                bqb = (bqrow * Hn + hh) * DQK + lk
                bob = (bqrow * Hn + hh) * DV + lk
                bqf = []
                for bc in range_constexpr(NK_D):
                    bqf.append(
                        buffer_ops.buffer_load(
                            rq, bqb, vec_width=4, dtype=T.bf16, soffset_bytes=bc * 32
                        )
                    )
                bof = []
                for bc in range_constexpr(NV_D):
                    bof.append(
                        buffer_ops.buffer_load(
                            rdo, bob, vec_width=4, dtype=T.bf16, soffset_bytes=bc * 32
                        )
                    )
                # pre-scaled once per workgroup (see the job-A note): the inner loop then costs
                # one v_fma per score element instead of mul+sub+mul.
                blse_i = fx.Float32(
                    buffer_ops.buffer_load(
                        rlse, hh * Tlen + bqrow, vec_width=1, dtype=T.f32
                    )
                ) * fx.Float32(LOG2E)
                bdel_i = (
                    fx.Float32(
                        buffer_ops.buffer_load(
                            rdel, hh * Tlen + bqrow, vec_width=1, dtype=T.f32
                        )
                    )
                    * scale
                )
                bi_ok = big < ln

                # keys j <= i for some i in [i0, i0+BM2) => j < i0 + BM2, and j < ln
                bi_hi = bi0 + BM2
                bjend_full = (bi_hi < ln).select(bi_hi, ln)
                bjend = (bi0 < ln).select(bjend_full, fx.Int32(0))

                def _ld_b(jj0):
                    r = []
                    for bc in range_constexpr(DQK // 64):
                        r.append(
                            buffer_ops.buffer_load(
                                rk,
                                ((lo + jj0 + srow) * Hn + hh) * DQK + ssub + bc * 64,
                                vec_width=4,
                                dtype=T.bf16,
                            )
                        )
                    for bc in range_constexpr(DV // 64):
                        r.append(
                            buffer_ops.buffer_load(
                                rv,
                                ((lo + jj0 + srow) * Hn + hh) * DV + ssub + bc * 64,
                                vec_width=4,
                                dtype=T.bf16,
                            )
                        )
                    return r

                NKC = DQK // 64
                NVC = DV // 64
                NPRB = NKC + NVC
                if const_expr(nsp == 1):
                    b_lo = fx.Int32(0)
                    b_hi = bjend
                else:
                    b_nt = (bjend + (BN2 - 1)) // BN2
                    b_lo = ((b_nt * bsp) // nsp) * BN2
                    b_hx = ((b_nt * (bsp + 1)) // nsp) * BN2
                    b_hi = (b_hx < bjend).select(b_hx, bjend)
                binit = [_zero4() for _ in range_constexpr(NK_D)] + _ld_b(b_lo)

                def _bbody(bst, bj0, masked):
                    """One streamed key tile.  `masked` selects the causal-diagonal variant.

                    PEELED CAUSAL GEOMETRY (OPT_BPEEL): job B owns queries [bi0, bi0+BM2) and
                    streams keys j upward over this split's range [b_lo, b_hi).  Every key tile
                    that ends at or before bi0 satisfies j <= bi0-1 <= big unconditionally, so
                    the mask is dead for the whole [b_lo, min(bi0, b_hi)) prefix -- which is ALL
                    BUT the last <=4 trips.  Only the suffix straddles the diagonal.  The prefix
                    therefore drops 4 VALU per score element (add / cmp / and / cndmask).

                    The `big < ln` half of the old predicate is dropped in BOTH variants: a lane
                    whose query row is past the sequence writes only ITS OWN dQ rows (the MFMA A
                    operand row index is the query index), and the epilogue's buffer resource
                    drops exactly those stores.
                    """
                    dq_acc = [bst[bc] for bc in range_constexpr(NK_D)]
                    bpre = [bst[NK_D + bc] for bc in range_constexpr(NPRB)]

                    gpu.barrier()
                    for bc in range_constexpr(NKC):
                        bdc = ssub + bc * 64
                        bvk = bpre[bc]
                        fx.ptr_store(bvk, p_kn + (srow * LDQ + bdc))
                        for be in range_constexpr(4):
                            fx.ptr_store(
                                vector.extract(
                                    bvk, static_position=[be], dynamic_position=[]
                                ),
                                p_kt + ((bdc + be) * LD2 + _wcol(bc)),
                            )
                    for bc in range_constexpr(NVC):
                        bdc = ssub + bc * 64
                        fx.ptr_store(bpre[NKC + bc], p_vn + (srow * LDVN + bdc))
                    gpu.barrier()
                    bnxt = _ld_b(bj0 + BN2)  # prefetch for the next iteration

                    for bjt in range_constexpr(BN2 // 16):
                        st_acc = _zero4()
                        for bc in range_constexpr(NK_D):
                            st_acc = _mfma(
                                _bf16x4(p_kn + ((bjt * 16 + lm) * LDQ + bc * 16 + lk)),
                                bqf[bc],
                                st_acc,
                            )
                        pt_acc = _zero4()
                        for bc in range_constexpr(NV_D):
                            pt_acc = _mfma(
                                _bf16x4(p_vn + ((bjt * 16 + lm) * LDVN + bc * 16 + lk)),
                                bof[bc],
                                pt_acc,
                            )

                        bdsv = []
                        for bii in range_constexpr(4):
                            bx_ = _get(st_acc, bii) * sclog - blse_i
                            bpx = fx.Float32(rocdl.exp2(T.f32, bx_.ir_value()))
                            if const_expr(masked):
                                bjgg = bj0 + bjt * 16 + lk + bii
                                if const_expr(OPT_BPEEL):
                                    bgood = bjgg <= big
                                else:
                                    bgood = bi_ok & (bjgg <= big)
                                bp = fx.Float32(bgood.select(bpx, fx.Float32(0.0)))
                            else:
                                bp = bpx
                            bdsv.append(bp * (_get(pt_acc, bii) * scale - bdel_i))
                        bdsf = _pack4(bdsv)  # == dS operand (row = i, k = j)

                        for bdt in range_constexpr(NK_D):
                            bb = _bf16x4(
                                p_kt + ((bdt * 16 + lm) * LD2 + _tcol(bdt, bjt))
                            )
                            dq_acc[bdt] = _mfma(bdsf, bb, dq_acc[bdt])
                    return dq_acc + bnxt

                b_out = binit
                if const_expr(OPT_BPEEL):
                    # peel point = bi0 clamped into this split's [b_lo, b_hi)
                    bcut = (bi0 < b_hi).select(bi0, b_hi)
                    bcut = (bcut > b_lo).select(bcut, b_lo)
                    # mask-free prefix: key tiles wholly below the diagonal
                    for bj0_, bst in range(b_lo, bcut, BN2, init=binit):
                        b_out = yield _bbody(bst, fx.Int32(bj0_), False)
                    # <=4 diagonal-straddling trips keep the full predicate
                    bmid = b_out
                    for bj0_, bst in range(bcut, b_hi, BN2, init=bmid):
                        b_out = yield _bbody(bst, fx.Int32(bj0_), True)
                else:
                    for bj0_, bst in range(b_lo, b_hi, BN2, init=binit):
                        b_out = yield _bbody(bst, fx.Int32(bj0_), True)

                for bii in range_constexpr(4):
                    bigs = bi0 + wv * 16 + lk + bii
                    bok = bigs < ln
                    borow = lo + bigs
                    if const_expr(nsp == 1):
                        bqbase = bok.select(((borow * Hn + hh) * DQK + lm) * 2, _OOB)
                    else:
                        bqbase = bok.select(
                            (bsp * wst_q + (borow * Hn + hh) * DQK + lm) * 2, _OOB
                        )
                    for bdt in range_constexpr(NK_D):
                        buffer_ops.buffer_store(
                            _cvt1(_get(b_out[bdt], bii)),
                            rdq,
                            bqbase,
                            offset_is_bytes=True,
                            soffset_bytes=bdt * 32,
                        )

    @flyc.jit
    def launch_bwd(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        DO: fx.Tensor,
        LSE: fx.Tensor,
        DEL: fx.Tensor,
        CU: fx.Tensor,
        DQO: fx.Tensor,
        DKO: fx.Tensor,
        DVO: fx.Tensor,
        Tlen: fx.Int32,
        Hn: fx.Int32,
        scale: fx.Float32,
        nb1: fx.Int32,
        nbtot: fx.Int32,
        nseq: fx.Int32,
        ilv: fx.Int32,
        stream: fx.Stream,
    ):
        k_bwd(
            Q,
            K,
            V,
            DO,
            LSE,
            DEL,
            CU,
            DQO,
            DKO,
            DVO,
            Tlen,
            Hn,
            nseq,
            nb1,
            nbtot,
            scale,
            ilv,
        ).launch(grid=(nseq * Hn, nbtot, 1), block=(NT, 1, 1), stream=stream)

    if const_expr(nsp == 1):
        return launch_delta, launch_bwd

    # ------------------------------------------------- split-K reduction (nsp > 1 variant only)
    @flyc.kernel(known_block_size=[NT_RED, 1, 1])
    def k_red(WS: fx.Tensor, OUT: fx.Tensor, n: fx.Int32):
        """OUT[i] = bf16( sum_p WS[p*n + i] ), i over one whole gradient tensor.

        Perfectly coalesced dwordx4 streams; n is always a multiple of VEC_RED (it is
        T*H*192 or T*H*128), so a thread's 8-element group is either wholly inside
        num_records or wholly outside and the hardware bounds check makes the tail branchless.
        """
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        rws = _rsrc(WS, n * (2 * nsp))
        rout = _rsrc(OUT, n * 2)
        idx = (bid * NT_RED + tid) * VEC_RED
        nb = n * 2  # byte stride between partial slabs
        pv = []
        for p in range_constexpr(nsp):
            pv.append(
                buffer_ops.buffer_load(
                    rws,
                    idx,
                    vec_width=VEC_RED,
                    dtype=T.bf16,
                    soffset_bytes=(nb * p).ir_value(),
                )
            )
        vals = []
        for e in range_constexpr(VEC_RED):
            acc = fx.Float32(
                vector.extract(pv[0], static_position=[e], dynamic_position=[])
            )
            for p in range_constexpr(1, nsp):
                acc = acc + fx.Float32(
                    vector.extract(pv[p], static_position=[e], dynamic_position=[])
                )
            vals.append(acc)
        buffer_ops.buffer_store(_packNr(vals), rout, idx)

    @flyc.jit
    def launch_red(
        WS: fx.Tensor,
        OUT: fx.Tensor,
        n: fx.Int32,
        nblk: fx.Int32,
        stream: fx.Stream,
    ):
        k_red(WS, OUT, n).launch(grid=(nblk, 1, 1), block=(NT_RED, 1, 1), stream=stream)

    return launch_delta, launch_bwd, launch_red
