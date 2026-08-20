# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import math as host_math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as _fly
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, as_dsl_value
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from aiter.ops.flydsl.rocdl_mfma_fp8 import Mfma32x32x64

_LOG2E = host_math.log2(host_math.e)

# Atom geometry (32x32x64 fp8).
MFMA_M = 32
MFMA_N = 32
MFMA_K = 64
WARP_SIZE = 64
A_FP8_PER_LANE = 32  # vec<8xi32>
C_F32_PER_LANE = 16  # vec<16xf32>


def _llvm_value(value):
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    return value


def _extract_aligned_pointer(tensor, address_space=None) -> ir.Value:
    ptr_type = ir.Type.parse(
        "!llvm.ptr" if address_space is None else f"!llvm.ptr<{address_space}>"
    )
    return _fly.extract_aligned_pointer_as_index(ptr_type, _llvm_value(tensor))


def _pointer_load(result_type: ir.Type, ptr: ir.Value) -> ir.Value:
    return llvm.LoadOp(result_type, _llvm_value(ptr)).result


def _pointer_store(value: ir.Value, ptr: ir.Value):
    return llvm.StoreOp(_llvm_value(value), _llvm_value(ptr))


def build_flash_attn_fp8_module(
    num_heads,
    head_dim=128,
    softmax_scale=None,
    waves_per_eu=2,
):
    gpu_arch = get_hip_arch()
    assert gpu_arch.startswith(
        "gfx95"
    ), f"fp8 32x32x64 MFMA requires CDNA4, got {gpu_arch}"
    assert head_dim == 128, "v1 only supports head_dim=128"

    BLOCK_M = 256
    BLOCK_N = 128
    HEAD_DIM = head_dim
    NUM_HEADS = num_heads
    NUM_WAVES = 8
    BLOCK_SIZE = NUM_WAVES * WARP_SIZE  # 512
    ROWS_PER_WAVE = BLOCK_M // NUM_WAVES  # 32
    STRIDE_TOKEN = NUM_HEADS * HEAD_DIM

    K_STEPS = HEAD_DIM // MFMA_K  # 2 head-dim chunks of 64 (QK contraction)
    N_KV_TILES = BLOCK_N // MFMA_N  # 4 kv sub-tiles of 32 (GEMM1 N)
    D_TILES = HEAD_DIM // MFMA_N  # 4 output d sub-tiles of 32
    PV_K_STEPS = BLOCK_N // MFMA_K  # 2 kv K-steps of 64 (PV contraction)

    if softmax_scale is None:
        softmax_scale = 1.0 / host_math.sqrt(head_dim)

    K_STRIDE = HEAD_DIM
    V_KV_STRIDE = 16  # bytes per kv within a 16-wide d block
    N_DBLOCKS = HEAD_DIM // 16  # 8

    PAD_K = 16
    PAD_V = 16
    K_UNIT_ROWS = 8  # one wave writes 8 contiguous kv rows per DMA pass
    K_DATA = BLOCK_N * K_STRIDE  # 16384 unpadded K tile bytes
    K_UNIT_STRIDE = K_UNIT_ROWS * K_STRIDE + PAD_K  # 1040
    N_K_UNITS = BLOCK_N // K_UNIT_ROWS  # 16
    V_DBLOCK_STRIDE = BLOCK_N * V_KV_STRIDE + PAD_V  # 2064 (padded)

    NUM_BUF_K = 2
    NUM_BUF_V = 3
    LDS_K_TILE = N_K_UNITS * K_UNIT_STRIDE  # 16640 (padded)
    LDS_V_TILE = N_DBLOCKS * V_DBLOCK_STRIDE  # 16512 (padded)
    LDS_K_SIZE = NUM_BUF_K * LDS_K_TILE
    LDS_V_SIZE = NUM_BUF_V * LDS_V_TILE
    LDS_K_OFF = 0
    LDS_V_OFF = LDS_K_OFF + LDS_K_SIZE
    LDS_TOTAL = LDS_V_OFF + LDS_V_SIZE

    allocator = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name="flash_attn_fp8_smem",
    )
    lds_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_offset + LDS_TOTAL

    bf16_dtype = fx.BFloat16

    USE_MANUAL_SCHED = True
    USE_IGLP = False
    IGLP_VARIANT = 1
    # s_setprio bias for the softmax-role (transcendental) wave; 0 disables.
    SOFTMAX_PRIO = 1

    # USE_SCHRAUDOLPH: replace the per-element quarter-rate v_exp_f32 in the P
    # exp2 with the Schraudolph linear-mantissa bit trick (full-rate VALU).  The
    # P output is quantized to fp8 E4M3 (~6% precision) so the ~2% approx error
    # is below the quantization floor.  corr (the O-rescale) stays exact.

    # USE_ROLLBACK: skip the per-tile O*=corr (64 fmul) + L*=corr when no lane's
    # running max grew this tile (corr==exp2(0)==1.0 exactly).  Wave-uniform
    # scf.if predicate (64-lane OR of corr<1) so the wave never diverges; EXACT,
    # not approximate.  After softmax warmup most tiles skip the rescale.
    USE_SCHRAUDOLPH = True
    USE_ROLLBACK = False

    # USE_QK_SCALE_FOLD (Trick 1): the QK GEMM's output multiplier
    #   A = q_descale * k_descale * softmax_scale * log2(e) * 2^23
    # is loop-invariant, so instead of paying one multiply per accumulator
    # element per tile we split A = 2^E * m and push each half somewhere free:
    # 2^E into the scaled-MFMA's E8M0 per-operand exponent (verified on gfx950:
    # byte b applies an exact 2^(b-127)), m into Q once in the prologue.  S then
    # emerges from the GEMM already in Schraudolph units and the exp2's FMA
    # multiplier degenerates to 1.0.
    #
    # Note A ~ 1e6, so the fallback of folding *all* of A into Q is not open to
    # us here: Q is fp8 E4M3 with a max of 448.  The exponent split is required,
    # not a refinement.
    USE_QK_SCALE_FOLD = True

    # USE_FROZEN_MAX (Trick 2): compute the row max on tile 0 only and declare
    # it final.  With m constant, exp2(m_old - m_new) == 1, so every tile after
    # the first drops its 64-element max reduction, its cross-lane exchange, and
    # the whole O/L rescale.  m also becomes loop-invariant, which is what makes
    # Trick 4 possible at all.
    #
    # Freezing is a bet: a later tile may hold a score above tile 0's max.
    # USE_WATCHDOG is the (exact) way to detect and undo that -- see below.  The
    # two ship together; frozen-without-watchdog silently overflows fp8.
    USE_FROZEN_MAX = True

    # USE_WATCHDOG (Trick 5): per-tile overflow detection + exact rollback for
    # the frozen max.  Detection is a max over the tile's exp2 arguments against
    # a fixed cap; the repair is a rare, wave-uniform branch.
    USE_WATCHDOG = True

    # USE_BIAS_FOLD (Trick 4): seed the QK accumulator with the Schraudolph bias
    # (C - m) instead of zero, so the GEMM performs the subtraction as part of
    # work it was doing anyway and S emerges already *being* the exp2 pattern.
    # Removes 4x vec<16xf32> of adds per tile.
    #
    # Two of the doc's three complications do not arise in this layout.  The
    # accumulator fragment is C_frag[L][v] = C[row, col=lo], so a lane's 16 slots
    # all belong to one query row and the bias it needs is already a per-lane
    # scalar -- no rank-1 outer product to broadcast it, and hence no bf16
    # residual split (the seed stays fp32-exact).  Complication 3 does apply and
    # is handled the doc's way: one materialized bias tile is routed as the C
    # operand of all four blocks' first MFMA rather than materialized four times.
    #
    # What this *does* cost is the watchdog's repair route.  Tile i's seed is
    # fixed before tile i's overflow is known, so the correction can no longer
    # ride on the scalar m_term; subtracting it from the accumulators instead
    # would cost back exactly the adds this trick removes.  It goes on the fp8
    # convert instead -- see SCALED_CVT below.
    USE_BIAS_FOLD = True

    # L_SPLIT_TILES (Trick 6): how many of the four 32-KV blocks have their
    # contribution to the denominator L summed on the VALU in fp32, before the
    # fp8 pack, instead of by a ones-column MFMA after it.  The remaining blocks
    # keep the MFMA.  0 = all matrix (the pre-trick behaviour), 4 = all vector.
    #
    # Only multiples of 2 are meaningful: each ones-column MFMA contracts a
    # K=64 slab, which is exactly two blocks (p_words[r*8:(r+1)*8] covers
    # nt = 2r, 2r+1), so a split that is not slab-aligned would have to keep the
    # MFMA it was trying to remove.  This is pure pipe rebalancing -- MFMA issue
    # slots against VALU slots -- so the best value is schedule-dependent and was
    # picked by measurement, not derivation.
    #
    # Measured on this schedule (B=1 S=65536 H=5 D=128, 3 runs each, median):
    #   L_SPLIT_TILES = 0  ->  1880 TFLOPS   (all matrix)
    #   L_SPLIT_TILES = 2  ->  1819 TFLOPS
    #   L_SPLIT_TILES = 4  ->  1760 TFLOPS   (all vector)
    # Monotonically worse, which is the opposite of the ASM's result (it uses
    # 44 of 64 elements on the VALU).  The reason is the ping-pong structure:
    # the two wave groups already interleave so that one group's ones-column
    # MFMA issues under the other group's softmax, so those MFMAs are not on
    # anyone's critical path and there is no MFMA pressure to relieve -- while
    # the VALU, which the Schraudolph convert already saturates, is exactly the
    # pipe that is contended.  The trick moves work from a free pipe to a busy
    # one.  In a single-role schedule (the ASM's) the same move is a win.
    #
    # Kept in and defaulted to the best-measured split rather than removed: the
    # accuracy side is a genuine (if small) improvement -- the vector half is
    # taken pre-pack, so its share of L never sees the E4M3 rounding -- and the
    # right value is a property of the schedule, so a future retune of the
    # ping-pong phases can revisit it by changing one integer.
    #
    # The two partials are NOT symmetric, which is the trick's one real trap.
    # The matrix partial is already contracted across the whole row by the MFMA.
    # The vector partial covers only this lane's 16 KV positions; the row's other
    # half lives in lane^32 (the accumulator's row index is hi*4 + ..., hi =
    # lane//32), so it needs one cross-lane exchange to complete.  Shuffling the
    # matrix partial would double-count; not shuffling the vector partial would
    # halve L.  Both mistakes produce plausible output, so the disjointness is
    # asserted structurally below rather than trusted.
    L_SPLIT_TILES = 2

    # ---- Appendix A constants ----
    SCHRAUDOLPH_2P23 = 1 << 23  # 8388608
    # C = 127*2^23 - 486411; the offset minimizes worst-case relative error of
    # the linear-mantissa interpolation over one octave.
    SCHRAUDOLPH_C = 127 * SCHRAUDOLPH_2P23 - 486411  # 1064866805
    E8M0_ONE = 0x7F7F7F7F  # identity operand scale (four E8M0 bytes of 2^0)
    # 0x4E87C000 as f32 == 1138753536.0.  Decoded: (that - C)/2^23 = 8.808 and
    # 2^8.808 = 448.2 -- the largest finite fp8 E4M3.  The cap is not a heuristic
    # margin, it is the storage format's saturation point transcribed into the
    # Schraudolph pattern domain, where m has already been subtracted, so it
    # tests the gap S - m no matter where m sits.
    SCHRAUDOLPH_CAP = 1138753536.0

    if USE_QK_SCALE_FOLD:
        assert USE_SCHRAUDOLPH, "the 2^23 in A only makes sense for Schraudolph"
    if USE_WATCHDOG:
        assert USE_SCHRAUDOLPH, "the cap is expressed in the Schraudolph domain"
    if USE_FROZEN_MAX:
        assert USE_WATCHDOG, "a frozen max without a watchdog can overflow fp8"
    assert L_SPLIT_TILES in (0, 2, 4), "the ones-column MFMA contracts two blocks"
    if L_SPLIT_TILES:
        assert USE_SCHRAUDOLPH, "the vector partial reads the Schraudolph pattern"
    if USE_BIAS_FOLD:
        # The seed *is* the Schraudolph bias, and it has to be loop-invariant to
        # be seeded before the GEMM that determines it -- see the circularity
        # note in Trick 4.
        assert USE_SCHRAUDOLPH, "the seed is the Schraudolph bias"
        assert USE_FROZEN_MAX, "an unfrozen m cannot be seeded before its own GEMM"
    # EXP2_SHIFT: scale all Schraudolph P up by 2^SHIFT.  A global P scale cancels
    # in the softmax normalization (O and L both scale), so this only repositions
    # P (range [2^-9,1]) within fp8 E4M3 -- a few bits up lifts small P out of the
    # subnormal range and improves quantization accuracy for free.
    EXP2_SHIFT = 4.0

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def fp8_attn_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        q_descale: fx.Float32,
        k_descale: fx.Float32,
        v_descale: fx.Float32,
        seq_len: fx.Int32,
    ):
        i8_dtype = fx.Int8
        i8_type = i8_dtype.ir_type
        bf16_type = bf16_dtype.ir_type
        v_i8x16 = Vec.make_type(16, i8_dtype)  # 16 bytes for coop load
        v_i8x32 = Vec.make_type(A_FP8_PER_LANE, i8_dtype)  # MFMA operand bytes

        fm_fast = fx.arith.FastMathFlags.fast

        # rocdl.s_setprio(0)

        def _fadd(a, b):
            return arith.addf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fsub(a, b):
            return arith.subf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmul(a, b):
            return arith.mulf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmax(a, b):
            return arith.MaxNumFOp(_raw(a), _raw(b), fastmath=fm_fast).result

        def _fmin(a, b):
            return arith.MinNumFOp(_raw(a), _raw(b), fastmath=fm_fast).result

        def _wave_or(pred_i1):
            # OR a per-lane i1 predicate across all 64 lanes -> wave-uniform i1.
            # ballot is the whole reduction in one instruction: the v_cmp lands
            # the 64 lane bits straight in an SGPR pair, so "did anyone" is a
            # scalar compare against 0.  The portable form (6 shuffle_xor + 6
            # or, 12 VALU ops on the critical path) is kept below for reference.
            mask = rocdl.ballot(T.i64, _raw(pred_i1))
            return arith.cmpi(
                arith.CmpIPredicate.ne, _raw(mask), _raw(fx.Int64(0))
            )

        def _sched_barrier():
            if USE_MANUAL_SCHED:
                rocdl.sched_barrier(0)

        def _iglp():
            if USE_IGLP:
                rocdl.iglp_opt(IGLP_VARIANT)

        def _f32_to_fp8_byte(f):
            # arith.truncf cannot lower f32 -> fp8 on this target; use the
            # rocdl pack intrinsic and keep the low fp8 byte.  Returns an i8.
            packed = rocdl.cvt_pk_fp8_f32(
                T.i32, _raw(f), _raw(c_zero_f), fx.Int32(0), False
            )
            return arith.trunci(T.i8, _raw(packed))

        v2i16_ty = Vec.make_type(2, fx.Int16)
        _zero_2xi16 = Vec.filled(2, 0, fx.Int16)

        def _f32x4_to_fp8_word(f0, f1, f2, f3, rscale=None):
            # Pack 4 f32 -> one i32 (4 contiguous fp8 bytes [f0,f1,f2,f3]) using
            # two cvt_pk_fp8_f32: low word = (f0,f1), high word = (f2,f3).
            # Halves both the cvt count and the LDS store count vs per-byte.
            #
            # rscale (Trick 4 + Trick 5): the scaled form of the same convert
            # divides by a runtime f32 before rounding, at no extra instruction.
            # Verified on gfx950: dst == fp8(src / scale), and scale == 1.0 is
            # bit-identical to the unscaled op -- so the watchdog's repair is
            # free on the overwhelmingly common no-overflow path, which is what
            # lets Trick 4 coexist with a frozen max whose bias is already baked
            # into the GEMM seed and can no longer absorb a correction.
            if const_expr(rscale is None):
                w0 = rocdl.cvt_pk_fp8_f32(T.i32, _raw(f0), _raw(f1), fx.Int32(0), False)
                w1 = rocdl.cvt_pk_fp8_f32(T.i32, _raw(f2), _raw(f3), _raw(w0), True)
                return w1
            lo = rocdl.cvt_scalef32_pk_fp8_f32(
                v2i16_ty, _raw(_zero_2xi16), _raw(f0), _raw(f1), _raw(rscale), False
            )
            hi = rocdl.cvt_scalef32_pk_fp8_f32(
                v2i16_ty, _raw(_zero_2xi16), _raw(f2), _raw(f3), _raw(rscale), False
            )
            lo32 = Vec(Vec(lo).bitcast(fx.Int32))[0]
            hi32 = Vec(Vec(hi).bitcast(fx.Int32))[0]
            return arith.ori(
                _raw(arith.andi(_raw(lo32), _raw(fx.Int32(0xFFFF)))),
                _raw(arith.shli(_raw(hi32), _raw(fx.Int32(16)))),
            )

        mfma = Mfma32x32x64()

        q_ptr = _extract_aligned_pointer(Q)
        k_ptr = _extract_aligned_pointer(K)
        v_ptr = _extract_aligned_pointer(V)
        o_ptr = _extract_aligned_pointer(O)

        seq_len_v = fx.Index(seq_len)

        # ---- LDS pointers (int8 storage) ----
        base_ptr = allocator.get_base()
        lds = SmemPtr(base_ptr, lds_offset, i8_type, shape=(LDS_TOTAL,)).get()

        # ---- Thread / block indices ----
        block_id = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)
        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lo = lane % 32
        hi = lane // 32

        wave_q_offset = wave_id * ROWS_PER_WAVE

        # ---- Decompose block_id -> (head, batch, q_tile) ----
        head_idx = block_id % NUM_HEADS
        batch_q_tile_id = block_id // NUM_HEADS
        num_q_tiles = (seq_len_v + BLOCK_M - 1) // BLOCK_M
        q_tile_idx = batch_q_tile_id % num_q_tiles
        batch_idx = batch_q_tile_id // num_q_tiles
        q_start = q_tile_idx * BLOCK_M

        def global_idx(token_idx, col):
            token = batch_idx * seq_len_v + token_idx
            return token * STRIDE_TOKEN + head_idx * HEAD_DIM + col

        # ---- Scales (log2 domain) ----
        c_log2e = fx.Float32(_LOG2E)
        qk_scale = _fmul(_fmul(q_descale, k_descale), fx.Float32(softmax_scale))
        scale_log2e = _fmul(qk_scale, c_log2e)
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_f = fx.Float32(0.0)

        # Schraudolph base-2 exp constants (Appendix A): reading the bits of
        # N = x*2^23 + C as an fp32 decodes to 2^E * (1+f) where x = E+f, i.e.
        # the exponent is exact and only the mantissa is linearly interpolated.
        # C = 127*2^23 - 486411 re-centers that interpolation error, halving the
        # worst case from ~6% to ~3% -- well under the ~6% fp8 E4M3 floor that P
        # is quantized to immediately afterwards.
        c_2p23 = fx.Float32(float(SCHRAUDOLPH_2P23))
        c_inv_2p23 = fx.Float32(2.0**-23)
        # EXP2_SHIFT lifts all P up by 2^SHIFT (global scale cancels in softmax
        # normalization; only repositions P within fp8 E4M3, out of subnormals).
        c_exp2_bias = fx.Float32(
            float(SCHRAUDOLPH_C) + EXP2_SHIFT * float(SCHRAUDOLPH_2P23)
        )

        def _i32c(v):
            # fx.Int32 wants a signed value; wrap bit patterns with bit 31 set.
            return fx.Int32(v - (1 << 32) if v >= (1 << 31) else v)

        e8m0_ident = _i32c(E8M0_ONE)
        qk_scale_e8m0 = e8m0_ident
        q_mant = None
        if const_expr(USE_QK_SCALE_FOLD):
            # Trick 1.  A = q_descale*k_descale*softmax_scale*log2(e)*2^23 is
            # loop-invariant; split A = 2^E * m and pay neither factor in the
            # loop.  Following the reference kernel we normalize m into [0.5,1)
            # by forcing the fp32 exponent field to 0x3F000000 and compensating
            # with +1 on the E8M0 byte -- so 2^(e+1-127) * 0.5*(1+f) == A
            # exactly, with no rounding in the power-of-two half.
            #
            # A ~ 1e6 puts e ~ 146, nowhere near the E8M0 edges (0x00 is a zero
            # sentinel, 0xFF saturates), so the +1 is always safe here.
            a_full = _fmul(scale_log2e, c_2p23)
            a_bits = arith.bitcast(T.i32, _raw(a_full))
            e_byte = arith.addi(
                arith.andi(
                    arith.shrui(_raw(a_bits), _raw(fx.Int32(23))),
                    _raw(fx.Int32(0xFF)),
                ),
                _raw(fx.Int32(1)),
            )
            # One E8M0 byte per 32-element scale block; broadcast to all four.
            qk_scale_e8m0 = arith.muli(_raw(e_byte), _raw(fx.Int32(0x01010101)))
            q_mant = arith.bitcast(
                T.f32,
                arith.ori(
                    arith.andi(_raw(a_bits), _raw(_i32c(0x807FFFFF))),
                    _raw(fx.Int32(0x3F000000)),
                ),
            )

        def qk_mfma(a, b, c):
            # QK GEMM only.  The PV and ones-column MFMAs keep the atom wrapper
            # (identity scales); only this one carries Trick 1's 2^E.  A = K,
            # B = Q, and the mantissa half of A lives in Q, so the exponent
            # rides on scaleB to keep the two halves on the same operand.
            if const_expr(not USE_QK_SCALE_FOLD):
                return mfma.call(a, b, c)
            return rocdl.mfma_scale_f32_32x32x64_f8f6f4_(
                res=mfma.accum_type,
                a=_raw(a),
                b=_raw(b),
                c=_raw(c),
                cbsz=0,
                blgp=0,
                opsel_a=0,
                scale_a=_raw(e8m0_ident),
                opsel_b=0,
                scale_b=_raw(qk_scale_e8m0),
            )

        def _schraudolph(s, scale_x2p23, m_term):
            # 2^(s*scale_log2e - m_new*scale_log2e) via the Schraudolph bit trick,
            # fused: biased = s*(scale_log2e*2^23) + (-m_new*scale_log2e*2^23 + C).
            # The mul+add contracts to a single FMA under fastmath, so the whole
            # exp2 is FMA + max + cvt_u32 + (free) bitcast == 3 full-rate ops,
            # vs the original mul+add+v_exp_f32 (~6 incl. the quarter-rate exp).
            # The convert is *unsigned* (v_cvt_u32_f32) to match the bit layout;
            # the max clamps deep-negative exponents (P underflow) to +0.0 rather
            # than relying on fptoui's poison-on-negative semantics.
            biased = _fmax(_fadd(_fmul(s, scale_x2p23), m_term), c_zero_f)
            return arith.bitcast(T.f32, arith.fptoui(T.i32, _raw(biased)))

        DMA_BYTES = 16
        DMA_LANES = (NUM_WAVES // 2) * WARP_SIZE  # 256 (one 4-wave group)
        DMA_PASSES = K_DATA // (DMA_LANES * DMA_BYTES)  # 4 (unpadded data size)
        WAVE_DMA_STRIDE = WARP_SIZE * DMA_BYTES  # 1024: one wave's 64-cell span

        head_base_elem = batch_idx * seq_len_v * fx.Index(
            STRIDE_TOKEN
        ) + head_idx * fx.Index(HEAD_DIM)

        def _rsrc(ptr):
            base_i64 = llvm.PtrToIntOp(T.i64, ptr).result
            off_i64 = arith.index_cast(T.i64, _raw(head_base_elem))
            addr_i64 = arith.addi(base_i64, off_i64)
            return fx.buffer_ops.create_buffer_resource_from_addr(addr_i64)

        k_rsrc = _rsrc(k_ptr)
        v_rsrc = _rsrc(v_ptr)

        _dma_size = arith.constant(DMA_BYTES, type=T.i32)
        _dma_zero = arith.constant(0, type=T.i32)
        _dma_aux = arith.constant(1, type=T.i32)

        _lds_ptr_ty = ir.Type.parse("!llvm.ptr<3>")

        def _dma_issue(rsrc, lds_byte_off, voffset_idx):
            voff_i32 = arith.index_cast(T.i32, voffset_idx)
            lds_addr = rocdl.readfirstlane(T.i64, arith.index_cast(T.i64, lds_byte_off))
            lds_ptr = llvm.inttoptr(_lds_ptr_ty, lds_addr)
            rocdl.raw_ptr_buffer_load_lds(
                rsrc, lds_ptr, _dma_size, voff_i32, _dma_zero, _dma_zero, _dma_aux
            )

        # ---- Precompute loop-invariant per-pass DMA addressing (depends only on
        # tid/wave, not on kv_start/buf) once, so it is hoisted out of the main
        # loop region instead of recomputed every iteration.  Perf-neutral on the
        # bench shape (DMA address VALU is overlap-hidden / LLVM already hoists);
        # kept as code hygiene -- the loop-invariant math is now explicit. ----
        _ltid = tid - fx.Index(DMA_LANES)  # 0..255 within G1
        _lwave = wave_id - fx.Index(NUM_WAVES // 2)  # 0..3 within G1
        _dma_k_inv = []
        for p in range_constexpr(DMA_PASSES):
            c = fx.Index(p * DMA_LANES) + _ltid
            kv = c // fx.Index(8)
            d = (c % fx.Index(8)) * fx.Index(16)
            lds_perm = (fx.Index(p * (NUM_WAVES // 2)) + _lwave) * fx.Index(
                K_UNIT_STRIDE
            )
            _dma_k_inv.append((kv, d, lds_perm))

        _half = wave_id % fx.Index(2)
        _dma_v_inv = []
        for p in range_constexpr(DMA_PASSES):
            c = fx.Index(p * DMA_LANES) + tid
            d_block = c // fx.Index(BLOCK_N)
            kv_lds_pos = c % fx.Index(BLOCK_N)
            # Inverse permutation: given LDS position, find actual kv row.
            blk = kv_lds_pos // fx.Index(32)
            rem = kv_lds_pos % fx.Index(32)
            hi_group = (rem // fx.Index(16)) % fx.Index(2)
            grp = (rem // fx.Index(4)) % fx.Index(4)
            fine = rem % fx.Index(4)
            kv = blk * fx.Index(32) + hi_group * fx.Index(4) + grp * fx.Index(8) + fine
            d_voff = d_block * fx.Index(16)
            # Each wave-pass writes a 64-kv half of one d-block; HW adds lane*16
            # within the chosen 64-kv (1024B) half.
            lds_perm = d_block * fx.Index(V_DBLOCK_STRIDE) + _half * fx.Index(
                WAVE_DMA_STRIDE
            )
            _dma_v_inv.append((kv, d_voff, lds_perm))

        def dma_k(buf, kv_start):
            base_lds = (
                fx.Index(lds_offset) + fx.Index(LDS_K_OFF) + buf * fx.Index(LDS_K_TILE)
            )
            for p in range_constexpr(DMA_PASSES):
                kv, d, lds_perm = _dma_k_inv[p]
                kv_abs = kv_start + kv
                in_b = kv_abs < seq_len_v
                kv_safe = fx.Index(ArithValue(in_b).select(kv_abs, fx.Index(0)))
                voff = kv_safe * fx.Index(STRIDE_TOKEN) + d
                _dma_issue(k_rsrc, base_lds + lds_perm, voff)

        def dma_v(buf, kv_start):
            base_lds = (
                fx.Index(lds_offset) + fx.Index(LDS_V_OFF) + buf * fx.Index(LDS_V_TILE)
            )
            for p in range_constexpr(DMA_PASSES):
                kv, d_voff, lds_perm = _dma_v_inv[p]
                kv_abs = kv_start + kv
                in_b = kv_abs < seq_len_v
                kv_safe = fx.Index(ArithValue(in_b).select(kv_abs, fx.Index(0)))
                voff = kv_safe * fx.Index(STRIDE_TOKEN) + d_voff
                _dma_issue(v_rsrc, base_lds + lds_perm, voff)

        def _wait_lgkmcnt(count=0):
            llvm.InlineAsmOp(
                None, [], f"s_waitcnt lgkmcnt({count})", "", has_side_effects=True
            )

        def _wait_vmcnt(count=0):
            llvm.InlineAsmOp(
                None, [], f"s_waitcnt vmcnt({count})", "", has_side_effects=True
            )

        def _gpu_barrier():
            llvm.InlineAsmOp(None, [], "s_barrier", "", has_side_effects=True)

        q_row = q_start + wave_q_offset + lo
        q_in_bounds = q_row < seq_len_v
        q_row_safe = fx.Index(ArithValue(q_in_bounds).select(q_row, fx.Index(0)))
        zero_qpack = Vec.filled(A_FP8_PER_LANE, 0, i8_dtype)

        q_packs = []
        for ks in range_constexpr(K_STEPS):
            d_col = fx.Index(ks * MFMA_K) + hi * 32
            g_idx = global_idx(q_row_safe, d_col)
            # 32 contiguous fp8 bytes along d (hi*32 + v, v in [0,32)).
            gep = fx.buffer_ops.get_element_ptr(
                q_ptr, fx.Int64(g_idx), elem_type=i8_type
            )
            raw = _pointer_load(v_i8x32, gep)
            raw = ArithValue(q_in_bounds).select(raw, zero_qpack.ir_value())
            qw = Vec(raw).bitcast(fx.Int32)  # vec<8xi32>
            if const_expr(USE_QK_SCALE_FOLD):
                # Trick 1, mantissa half: Q *= m once, here, instead of S *= A
                # on 64 accumulator elements per tile forever.  Round-trips
                # through f32 because there is no fp8 multiply; the re-quantize
                # costs at most one E4M3 ULP, far below the format's own ~6%.
                scaled = []
                for w in range_constexpr(A_FP8_PER_LANE // 4):
                    word = qw[w]
                    lo2 = Vec(
                        rocdl.cvt_pk_f32_fp8(
                            Vec.make_type(2, fx.Float32), _raw(word), False
                        )
                    )
                    hi2 = Vec(
                        rocdl.cvt_pk_f32_fp8(
                            Vec.make_type(2, fx.Float32), _raw(word), True
                        )
                    )
                    f = [
                        _fmul(lo2[0], q_mant),
                        _fmul(lo2[1], q_mant),
                        _fmul(hi2[0], q_mant),
                        _fmul(hi2[1], q_mant),
                    ]
                    scaled.append(_f32x4_to_fp8_word(*f))
                qw = Vec.from_elements(scaled, fx.Int32)
            q_packs.append(qw)

        # ===================================================================
        # Online-softmax loop carried state, per wave (one 32-wide tile of q).
        # The C-layout puts q in the lane (lo) and kv in the value index, so
        # each lane independently owns the running stats for q = lo.  We keep
        # m / l as scalars per lane and O as 4 vec<16xf32> (one per d-tile).
        # ===================================================================
        # L[q] = sum_kv P[kv,q] is a VALU row-sum (computed in do_softmax over the
        # 64 exact f32 exp values this lane holds, then combined with the hi-peer
        # half via shuffle_xor(32) -> full 128-kv sum).  Carried as a scalar f32
        # per lane (q = lo) and rescaled by corr each online-softmax step.

        # ---- V HW-transpose read + PV accumulate helper ----
        # Factored out so both the deferred in-loop PV (tile i-1) and the
        # epilogue PV (tile N-1) share one code path.
        v_tr8_ty = Vec.make_type(2, fx.Int32)
        lo_in_grp = lo % fx.Index(16)

        def read_v_pack(v_off, dt, ks):
            d_block = (lo // fx.Index(16)) + fx.Index(2 * dt)
            grp_db = fx.Index(lds_offset) + v_off + d_block * fx.Index(V_DBLOCK_STRIDE)
            reads = []
            for kc in range_constexpr(4):
                kv0 = hi * fx.Index(16) + fx.Index(
                    ks * 64 + (kc // 2) * 32 + (kc % 2) * 8
                )
                byte_off = (
                    grp_db + kv0 * fx.Index(V_KV_STRIDE) + lo_in_grp * fx.Index(8)
                )
                ptr = fx.buffer_ops.create_llvm_ptr(fx.Int64(byte_off), address_space=3)
                reads.append(Vec(rocdl.ds_read_tr8_b64(v_tr8_ty, ptr).result))
            ab = reads[0].shuffle(reads[1], list(range(4)))
            cd = reads[2].shuffle(reads[3], list(range(4)))
            return ab.shuffle(cd, list(range(8)))

        def apply_pv(
            o_accs,
            l_acc,
            p_pack,
            p_rowsum,
            corr,
            v_off,
            preloaded_vw,
            k_buf_off,
        ):
            corr_vec = Vec.from_elements([corr], fx.Float32).broadcast_to(
                C_F32_PER_LANE
            )
            if const_expr(USE_ROLLBACK or USE_FROZEN_MAX):
                # Speculative rollback: O*=corr and L=L*corr+rowsum only matter
                # when some lane's running max grew this tile (corr<1).
                # corr==exp2(0)==1.0 exactly when no lane grew, so skipping the
                # 64-fmul O-rescale + the L-mul is EXACT (not approximate).
                # Branch on a wave-uniform OR of the per-lane (corr<1) predicate
                # so the wave never diverges.
                #
                # This is also how Trick 2 collects its main prize.  With the max
                # frozen, corr is the literal 1.0 on every tile, so the branch is
                # never taken and the rescale is gone from the steady state; the
                # only thing that revives it is a watchdog repair, which is what
                # makes that repair exact rather than a downgrade.
                grew = arith.cmpf(
                    arith.CmpFPredicate.OLT, _raw(corr), _raw(fx.Float32(1.0))
                )
                any_grew = _wave_or(grew)
                o_vt = Vec.make_type(C_F32_PER_LANE, fx.Float32)
                resc_if = scf.IfOp(
                    any_grew, results_=[o_vt, o_vt, o_vt, o_vt, T.f32], has_else=True
                )
                with ir.InsertionPoint(resc_if.then_block):
                    _ro = [
                        _fmul(Vec(o_accs[dt]), corr_vec)
                        for dt in range_constexpr(D_TILES)
                    ]
                    _rl = _fadd(_fmul(l_acc, corr), p_rowsum)
                    scf.YieldOp(
                        [
                            _raw(_ro[0]),
                            _raw(_ro[1]),
                            _raw(_ro[2]),
                            _raw(_ro[3]),
                            _raw(_rl),
                        ]
                    )
                with ir.InsertionPoint(resc_if.else_block):
                    _nl = _fadd(_raw(l_acc), _raw(p_rowsum))
                    scf.YieldOp(
                        [
                            _raw(o_accs[0]),
                            _raw(o_accs[1]),
                            _raw(o_accs[2]),
                            _raw(o_accs[3]),
                            _raw(_nl),
                        ]
                    )
                o_accs = [
                    as_dsl_value(resc_if.results[dt], o_accs[dt])
                    for dt in range_constexpr(D_TILES)
                ]
                l2 = as_dsl_value(resc_if.results[4], l_acc)
            else:
                l2 = _fadd(_fmul(l_acc, corr), p_rowsum)
            p_ks_list = [
                Vec(p_pack).shuffle(Vec(p_pack), list(range(r * 8, r * 8 + 8)))
                for r in range_constexpr(PV_K_STEPS)
            ]
            PV_UNITS = D_TILES * PV_K_STEPS  # 8
            vw = [None] * 4
            kw_prime = [None] * PREFETCH_DEPTH
            for u in range_constexpr(PREFETCH_DEPTH):
                vw[u] = preloaded_vw[u]
            o = list(o_accs)
            for u in range_constexpr(PV_UNITS):
                dt = u // PV_K_STEPS
                ks = u % PV_K_STEPS
                if const_expr(not (USE_ROLLBACK or USE_FROZEN_MAX) and ks == 0):
                    o[dt] = _fmul(Vec(o[dt]), corr_vec)
                if const_expr(u + PREFETCH_DEPTH < PV_UNITS):
                    un = u + PREFETCH_DEPTH
                    vw[(u + PREFETCH_DEPTH) % 4] = read_v_pack(
                        v_off, un // PV_K_STEPS, un % PV_K_STEPS
                    )
                else:
                    ki = u - (PV_UNITS - PREFETCH_DEPTH)
                    kw_prime[ki] = _load_k_unit(k_buf_off, ki // K_STEPS, ki % K_STEPS)
                _sched_barrier()
                o[dt] = mfma.call(vw[u % 4], p_ks_list[ks], o[dt])
            return o, l2, kw_prime

        QK_UNITS = N_KV_TILES * K_STEPS  # 8 (nt outer, ks inner)

        PREFETCH_DEPTH = 1

        def _load_k_unit(k_buf_off, nt, ks):
            kv_row = lo + fx.Index(nt * 32)
            d_base = fx.Index(ks * MFMA_K) + hi * 32
            row_off = kv_row * K_STRIDE + (kv_row // fx.Index(K_UNIT_ROWS)) * fx.Index(
                PAD_K
            )
            blk_lo = Vec(
                Vec.load(v_i8x16, lds, [k_buf_off + row_off + d_base])
            ).bitcast(fx.Int32)
            blk_hi = Vec(
                Vec.load(v_i8x16, lds, [k_buf_off + row_off + d_base + fx.Index(16)])
            ).bitcast(fx.Int32)
            return blk_lo.shuffle(blk_hi, list(range(8)))

        def _load_k_unit_global(kv_start, nt, ks):
            # Same MFMA A-operand (vec<8xi32> = K[kv_row, d_base:d_base+32]) as
            # _load_k_unit, but read straight from global K into VGPR
            # (global_load_dwordx4), skipping the global->LDS->VGPR round-trip.
            kv_row = kv_start + lo + fx.Index(nt * 32)
            k_in_b = kv_row < seq_len_v
            kv_safe = fx.Index(ArithValue(k_in_b).select(kv_row, fx.Index(0)))
            d_base = fx.Index(ks * MFMA_K) + hi * 32
            g_idx = global_idx(kv_safe, d_base)
            gep = fx.buffer_ops.get_element_ptr(
                k_ptr, fx.Int64(g_idx), elem_type=i8_type
            )
            raw = _pointer_load(v_i8x32, gep)
            raw = ArithValue(k_in_b).select(raw, zero_qpack.ir_value())
            return Vec(raw).bitcast(fx.Int32)

        def do_qk(k_buf_off, preloaded_kw, v_off, seed=None):
            kw = [None] * 4
            vw_prime = [None] * PREFETCH_DEPTH
            for u in range_constexpr(PREFETCH_DEPTH):
                kw[u] = preloaded_kw[u]
            # Trick 4: seeding C with the bias makes the GEMM do the subtraction.
            # One bias tile feeds all four blocks (the doc's complication 3): D
            # and C need not alias, so the same value is simply the C operand of
            # each block's first MFMA -- no copies, and it also replaces the 4x16
            # accumulator clears the zero seed would have needed.
            s_accs = [
                (mfma.zero_value if const_expr(seed is None) else seed)
                for _ in range_constexpr(N_KV_TILES)
            ]
            for u in range_constexpr(QK_UNITS):
                nt = u // K_STEPS
                ks = u % K_STEPS
                if const_expr(u + PREFETCH_DEPTH < QK_UNITS):
                    un = u + PREFETCH_DEPTH
                    kw[(u + PREFETCH_DEPTH) % 4] = _load_k_unit(
                        k_buf_off, un // K_STEPS, un % K_STEPS
                    )
                    # rocdl.sched_group_barrier(rocdl.mask_dsrd, 2, 0)
                else:
                    vi = u - (QK_UNITS - PREFETCH_DEPTH)
                    vw_prime[vi] = read_v_pack(v_off, vi // PV_K_STEPS, vi % PV_K_STEPS)
                _sched_barrier()
                s_accs[nt] = qk_mfma(kw[u % 4], q_packs[ks], s_accs[nt])
                # rocdl.sched_group_barrier(rocdl.mask_mfma, 1, 0)
            return s_accs, vw_prime

        def _lane_max(vecs):
            # Max over this lane's 64 scores, reduced packed: a tree over the
            # four vec<16xf32> (v_pk_max, 2 elements per slot) then a halving
            # tree inside the survivor via shuffles.  ~32 VALU slots against the
            # 63 a scalar chain would take.
            v = _fmax(_fmax(Vec(vecs[0]), Vec(vecs[1])), _fmax(Vec(vecs[2]), Vec(vecs[3])))
            w = C_F32_PER_LANE
            while const_expr(w > 1):
                h = w // 2
                a = Vec(v).shuffle(Vec(v), list(range(h)))
                b = Vec(v).shuffle(Vec(v), list(range(h, w)))
                v = _fmax(a, b)
                w = h
            return Vec(v)[0]

        def _lane_sum(v):
            # Sum a vec<16xf32> to a scalar by a halving shuffle tree: 4 packed
            # v_pk_add steps rather than the 15 scalar adds a chain would take.
            w = C_F32_PER_LANE
            while const_expr(w > 1):
                h = w // 2
                a = Vec(v).shuffle(Vec(v), list(range(h)))
                b = Vec(v).shuffle(Vec(v), list(range(h, w)))
                v = _fadd(a, b)
                w = h
            return Vec(v)[0]

        def _rowmax(s_accs, m_running):
            # Full row max: this lane's 64 scores, then the peer half of the row
            # (lane^32), then the running value.  Trick 2 pays this exactly once.
            local_max = Vec(s_accs[0])[0]
            for nt in range_constexpr(N_KV_TILES):
                for r in range_constexpr(C_F32_PER_LANE):
                    if const_expr(nt == 0 and r == 0):
                        continue
                    local_max = _fmax(local_max, Vec(s_accs[nt])[r])
            peer_max = fx.Float32(local_max).shuffle_xor(
                fx.Int32(32), fx.Int32(WARP_SIZE)
            )
            return _fmax(m_running, _fmax(local_max, peer_max))

        def _bias_seed(m):
            # Trick 4's accumulator seed: the Schraudolph bias C - m, which in
            # this fragment layout is a per-lane scalar splatted over the lane's
            # 16 slots (all one query row).  Recomputed per tile rather than
            # carried: it is two scalar ops off a value already in the loop's
            # carry, against 16 extra VGPRs of carry pressure.
            return Vec.from_elements(
                [_fadd(_fsub(c_zero_f, m), c_exp2_bias)], fx.Float32
            ).broadcast_to(C_F32_PER_LANE)

        def _watchdog_seeded(s_accs, m_frozen):
            # Trick 5 against a folded bias.  Detection is unchanged in spirit
            # but cheaper: S is already the pattern (the GEMM subtracted m), so
            # the m_term add before the compare is gone too.
            #
            # The repair cannot move the bias any more -- tile i's seed was fixed
            # before tile i's overflow was known -- so instead of lowering the
            # pattern it divides P down on the fp8 convert, which takes a runtime
            # scale operand for free.  Equivalent by construction: scaling P by
            # 2^(-e/2^23) is what lowering every pattern by e would have done.
            # m still rises by e so later tiles seed against the corrected
            # reference (the doc's step 7, "recompute the cached bias from m",
            # falls out of _bias_seed being recomputed per tile).
            t = _lane_max(s_accs)
            peer = fx.Float32(t).shuffle_xor(fx.Int32(32), fx.Int32(WARP_SIZE))
            e = _fmax(_fsub(_fmax(t, peer), fx.Float32(SCHRAUDOLPH_CAP)), c_zero_f)
            eu = _fmul(e, c_inv_2p23)
            # cvt_scalef32 divides by its scale operand (verified on gfx950), so
            # the reciprocal direction is the one that shrinks P.  e == 0 gives
            # exactly 1.0, which the same probe showed is bit-identical to the
            # unscaled convert -- the common path pays nothing at all.
            rscale = rocdl.exp2(T.f32, _raw(eu))
            corr = rocdl.exp2(T.f32, _raw(_fsub(c_zero_f, eu)))
            if const_expr(USE_QK_SCALE_FOLD):
                delta = e  # m is tracked in accumulator == pattern units
            else:
                delta = arith.divf(_raw(eu), _raw(scale_log2e))
            return _fadd(_raw(m_frozen), _raw(delta)), corr, rscale

        def _watchdog(s_accs, m_term, m_frozen):
            # Trick 5, detection.  The Schraudolph pattern is aff = S + m_term
            # with m_term uniform across the tile, so max(aff) = max(S) + m_term
            # and the whole test can run in the S domain -- no need to
            # materialize aff first.  SCHRAUDOLPH_CAP decodes to exactly 448.2,
            # the largest finite fp8 E4M3, so testing against it is testing
            # whether P is about to leave the storage format, wherever m sits.
            #
            # The repair is branchless.  An scf.IfOp here would make m a region
            # result, i.e. a loop-carried value produced by a branch, which
            # serializes tile i's softmax against tile i+1's and costs far more
            # than the arithmetic it skips (measured: ~170 TFLOPS).  Everything
            # the repair needs is six scalar VALU ops and one cross-lane swap,
            # so it is cheaper to always compute than to predicate.
            #
            # The repair works in the pattern domain rather than rescaling P in
            # float.  aff is affine in m, so lowering every pattern by a constant
            # e is *identical* to having frozen a max larger by e/2^23 -- the same
            # softmax against a larger reference, exact by construction, and it
            # folds into the scalar bias instead of touching 64 accumulators.
            #
            # e is the true row max above the cap: the row's other half lives in
            # lane^32, hence the one swap.  Clamped at 0, so in the common case
            # (no overflow) e is exactly 0 -- m is unmoved, m_term is unmoved,
            # and corr is exactly 1.0, which is what lets apply_pv skip the O/L
            # rescale.  SCHRAUDOLPH_CAP decodes to 448.2, the largest finite fp8
            # E4M3, so this is the storage format's saturation point, not a
            # heuristic margin.
            t = _fadd(_lane_max(s_accs), m_term)
            peer = fx.Float32(t).shuffle_xor(fx.Int32(32), fx.Int32(WARP_SIZE))
            # max(t,peer) - cap, floored at 0.  NaN propagates through subf and
            # loses the MaxNumF against 0, so a NaN row degrades to e == 0 rather
            # than poisoning m for every later tile.
            e = _fmax(_fsub(_fmax(t, peer), fx.Float32(SCHRAUDOLPH_CAP)), c_zero_f)

            if const_expr(USE_QK_SCALE_FOLD):
                delta = e  # m is tracked in accumulator == pattern units
            else:
                delta = arith.divf(
                    _raw(_fmul(e, c_inv_2p23)), _raw(scale_log2e)
                )
            corr = rocdl.exp2(T.f32, _raw(_fsub(c_zero_f, _fmul(e, c_inv_2p23))))
            return _fsub(m_term, e), _fadd(_raw(m_frozen), _raw(delta)), corr

        def do_softmax(s_accs, m_running, set_prio=False, first_tile=True, seeded=False):
            if const_expr(SOFTMAX_PRIO != 0 and set_prio):
                rocdl.s_setprio(SOFTMAX_PRIO)
            # With Trick 1 the accumulator already carries A = scale*log2e*2^23,
            # so m is tracked in those same units and the conversion back out is
            # the reciprocal of A's 2^23 half rather than scale_log2e.
            if const_expr(USE_QK_SCALE_FOLD):
                to_log2 = c_inv_2p23  # accumulator units -> log2 units
            else:
                to_log2 = scale_log2e

            # Trick 2: m is decided on tile 0 and never moves again.  Later tiles
            # skip the 64-element max chain and its cross-lane exchange, and corr
            # is exactly 1.0 -- which also deletes the O/L rescale in apply_pv.
            # If the bet loses, the watchdog below repairs it exactly.
            #
            # Both wave groups seed m from S(0) in the prologue, so `frozen` is a
            # compile-time fact everywhere inside the loop and the reduction is
            # not merely predicated but absent.
            frozen = const_expr(USE_FROZEN_MAX and not first_tile)
            if const_expr(frozen):
                m_new = m_running
                corr = fx.Float32(1.0)
            else:
                m_new = _rowmax(s_accs, m_running)
                corr = (
                    fx.Float32(1.0)
                    if const_expr(USE_FROZEN_MAX)
                    else rocdl.exp2(
                        T.f32, _raw(_fmul(_fsub(m_running, m_new), to_log2))
                    )
                )

            n_groups = C_F32_PER_LANE // 4

            p_words = []
            v_partial = None  # Trick 6's VALU-side partial denominator
            rscale = None
            if const_expr(USE_SCHRAUDOLPH):
                # Fold scale_log2e and the 2^23 / C affine into per-tile coeffs so
                # each element's exp2 is one FMA + max + cvt_u32 (all full-rate).
                # Done packed over the whole vec<16xf32> (v_pk_fma / v_pk_max), so
                # the per-element cost is half a VALU slot for the affine plus one
                # convert -- vs the packed affine + quarter-rate v_exp_f32 before.
                if const_expr(USE_QK_SCALE_FOLD):
                    # Trick 1 has already applied A to S inside the GEMM, so the
                    # FMA's multiplier is literally 1.0 and drops out: the exp2
                    # is now add + max + cvt_u32.
                    m_term = _fadd(_fsub(c_zero_f, m_new), c_exp2_bias)
                    sx_vec = None
                else:
                    neg_scaled_m_new = _fsub(c_zero_f, _fmul(scale_log2e, m_new))
                    scale_x2p23 = _fmul(scale_log2e, c_2p23)
                    m_term = _fadd(_fmul(neg_scaled_m_new, c_2p23), c_exp2_bias)
                    sx_vec = Vec.from_elements([scale_x2p23], fx.Float32).broadcast_to(
                        C_F32_PER_LANE
                    )
                if const_expr(seeded):
                    # Trick 4: the GEMM already added (C - m), so S *is* the
                    # pattern.  The affine collapses to nothing and the softmax
                    # is down to max + convert.
                    if const_expr(USE_WATCHDOG):
                        m_new, corr, rscale = _watchdog_seeded(s_accs, m_new)
                elif const_expr(frozen and USE_WATCHDOG):
                    # Runs before the affine so the repair can fold into the
                    # scalar m_term -- one f32 across the branch instead of four
                    # vec<16xf32>, and no rewrite of the accumulators.
                    m_term, m_new, corr = _watchdog(s_accs, m_term, m_new)

                mt_vec = Vec.from_elements([m_term], fx.Float32).broadcast_to(
                    C_F32_PER_LANE
                )
                zero_vec = Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)
                cap_vec = Vec.filled(C_F32_PER_LANE, SCHRAUDOLPH_CAP, fx.Float32)
                i32_vec_ty = Vec.make_type(C_F32_PER_LANE, fx.Int32)
                f32_vec_ty = Vec.make_type(C_F32_PER_LANE, fx.Float32)
                for nt in range_constexpr(N_KV_TILES):
                    if const_expr(seeded):
                        # Trick 4 leaves the pattern in the accumulator, so the
                        # watchdog's repair rides on the convert (rscale) and
                        # cannot lower aff itself the way the unseeded path
                        # lowers m_term.  But the bitcast below reinterprets the
                        # *pattern* as f32: once aff reaches 0x7F800000 the
                        # result is an Inf/NaN bit pattern, and rscale -- applied
                        # after, at the convert -- can no longer rescue it.
                        # So clamp here, exactly as the ASM's v_min_u32 does.
                        # The ceiling is the cap itself: the watchdog has already
                        # guaranteed 2^(-e/2^23) brings the true row max back to
                        # it, so this only bites on the elements the repair was
                        # going to squash anyway.
                        aff = _fmin(Vec(s_accs[nt]), cap_vec)
                    elif const_expr(USE_QK_SCALE_FOLD):
                        aff = _fadd(Vec(s_accs[nt]), mt_vec)
                    else:
                        aff = _fadd(_fmul(Vec(s_accs[nt]), sx_vec), mt_vec)
                    biased = _fmax(aff, zero_vec)
                    ps_v = Vec(
                        arith.bitcast(
                            f32_vec_ty, arith.fptoui(i32_vec_ty, _raw(biased))
                        )
                    )
                    if const_expr(nt < L_SPLIT_TILES):
                        # Trick 6, vector half.  Taken pre-pack, off the f32
                        # pattern rather than the fp8 bytes the MFMA would have
                        # read -- so this half of L skips the E4M3 quantization
                        # error entirely, on top of moving off the MFMA pipe.
                        v_partial = (
                            Vec(ps_v)
                            if const_expr(v_partial is None)
                            else _fadd(Vec(v_partial), Vec(ps_v))
                        )
                    for rg in range_constexpr(n_groups):
                        p_words.append(
                            _f32x4_to_fp8_word(
                                *[ps_v[rg * 4 + i] for i in range_constexpr(4)],
                                rscale=rscale,
                            )
                        )
            else:
                # Affine scale*S - scale*m computed packed (vector mulf/addf ->
                # v_pk_*) over the whole vec<16xf32>; exp2 stays per-scalar
                # (quarter-rate v_exp_f32, exact).
                # (unreachable with USE_QK_SCALE_FOLD, which asserts Schraudolph)
                neg_scaled_m_new = _fsub(c_zero_f, _fmul(scale_log2e, m_new))
                scale_vec = Vec.from_elements([scale_log2e], fx.Float32).broadcast_to(
                    C_F32_PER_LANE
                )
                neg_m_vec = Vec.from_elements(
                    [neg_scaled_m_new], fx.Float32
                ).broadcast_to(C_F32_PER_LANE)
                for nt in range_constexpr(N_KV_TILES):
                    aff = Vec(_fadd(_fmul(Vec(s_accs[nt]), scale_vec), neg_m_vec))
                    for rg in range_constexpr(n_groups):
                        ps = [
                            rocdl.exp2(T.f32, _raw(aff[rg * 4 + i]))
                            for i in range_constexpr(4)
                        ]
                        p_words.append(_f32x4_to_fp8_word(*ps))
            p_pack = Vec.from_elements(p_words, fx.Int32)
            # Trick 6: ones-column MFMA r contracts p_words[r*8:(r+1)*8], which
            # is exactly blocks nt = 2r and 2r+1.  So the slabs the vector half
            # already covered are the ones with 2r < L_SPLIT_TILES, and dropping
            # precisely those keeps the two partials disjoint and covering.
            m_first_slab = const_expr(L_SPLIT_TILES // 2)
            p_ks_list = [
                Vec(p_pack).shuffle(Vec(p_pack), list(range(r * 8, r * 8 + 8)))
                for r in range_constexpr(PV_K_STEPS)
            ]
            m_partial = None
            for ks in range_constexpr(m_first_slab, PV_K_STEPS):
                m_partial = mfma.call(
                    ones_pack,
                    p_ks_list[ks],
                    mfma.zero_value if const_expr(m_partial is None) else m_partial,
                )

            if const_expr(v_partial is None):
                p_rowsum = Vec(m_partial)[0]
            else:
                # The asymmetry the doc warns about: the vector partial holds
                # only this lane's KV positions, so it needs the lane^32 exchange
                # to complete the row; the matrix partial was already contracted
                # across lanes by the MFMA and must NOT be shuffled.
                v_sum = _lane_sum(v_partial)
                v_sum = _fadd(
                    v_sum,
                    fx.Float32(v_sum).shuffle_xor(fx.Int32(32), fx.Int32(WARP_SIZE)),
                )
                if const_expr(rscale is not None):
                    # The vector half is read before the convert, so it has not
                    # seen the watchdog's rscale division that the fp8 P (and
                    # hence the matrix half and PV) did.  corr == 1/rscale, and
                    # it is exactly 1.0 whenever no overflow fired, so the common
                    # path is one fmul the scheduler can hoist away.
                    v_sum = _fmul(v_sum, corr)
                p_rowsum = (
                    v_sum
                    if const_expr(m_partial is None)
                    else _fadd(v_sum, Vec(m_partial)[0])
                )
            if const_expr(SOFTMAX_PRIO != 0 and set_prio):
                rocdl.s_setprio(0)
            return m_new, corr, p_pack, p_rowsum

        # All-ones FP8 E4M3 A-operand for the ones-column row-sum MFMAs.
        # 1.0 in E4M3 = 0x38; packed 4 per i32 word -> 0x38383838.
        ones_pack = Vec.filled(A_FP8_PER_LANE // 4, 0x38383838, fx.Int32)

        m_init = c_neg_inf
        l_init = c_zero_f  # L is a scalar VALU accumulator now
        o_init = [
            Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)
            for _ in range_constexpr(D_TILES)
        ]

        def _bufs(kv_start):
            i = kv_start // fx.Index(BLOCK_N)
            is_first = i < fx.Index(1)
            k_cur = i % fx.Index(NUM_BUF_K)
            k_buf_off = fx.Index(LDS_K_OFF) + k_cur * fx.Index(LDS_K_TILE)
            v_cur = i % fx.Index(NUM_BUF_V)
            v_prev = (i + fx.Index(NUM_BUF_V - 1)) % fx.Index(NUM_BUF_V)
            v_prev_sel = fx.Index(ArithValue(is_first).select(v_cur, v_prev))
            v_prev_off = fx.Index(LDS_V_OFF) + v_prev_sel * fx.Index(LDS_V_TILE)
            k_next = (i + fx.Index(1)) % fx.Index(NUM_BUF_K)
            v_next = (i + fx.Index(1)) % fx.Index(NUM_BUF_V)
            return is_first, k_buf_off, v_prev_off, k_next, v_next

        is_g0 = wave_id < fx.Index(NUM_WAVES // 2)
        _, _, _, _k_next0, _v_next0 = _bufs(fx.Index(0))

        # Issue all prologue memory ops up front, back-to-back: the K-unit
        # global prefetch (global -> VGPR) then both DMA calls (tile 0 + tile 1).
        # The Q load (q_packs) is already issued above.
        kvw = [None] * PREFETCH_DEPTH
        for u in range_constexpr(PREFETCH_DEPTH):
            kvw[u] = _load_k_unit_global(fx.Index(0), u // K_STEPS, u % K_STEPS)

        if is_g0:
            dma_v(fx.Index(0), fx.Index(0))
            dma_v(_v_next0, fx.Index(BLOCK_N))
        else:
            dma_k(fx.Index(0), fx.Index(0))
            dma_k(_k_next0, fx.Index(BLOCK_N))

        _wait_vmcnt()
        _wait_lgkmcnt()
        _gpu_barrier()

        loop_step = fx.Int32(BLOCK_N)
        num_iters = (seq_len_v + fx.Index(BLOCK_N - 1)) // fx.Index(BLOCK_N)
        last_i = num_iters - fx.Index(1)
        v_last_buf = last_i % fx.Index(NUM_BUF_V)
        v_last_off = fx.Index(LDS_V_OFF) + v_last_buf * fx.Index(LDS_V_TILE)
        of0 = Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)
        of1 = Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)
        of2 = Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)
        of3 = Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)
        lf = c_zero_f

        if is_g0:

            def g0_iter0():
                _, k_buf_off, _, _, _ = _bufs(fx.Index(0))

                v0_off = fx.Index(LDS_V_OFF)  # V^0 buf = LDSV[0]
                sA, vwp = do_qk(k_buf_off, preloaded_kw=kvw, v_off=v0_off)
                m_new, corr_new, p_new, prowsum_new = do_softmax(sA, m_init)
                _wait_lgkmcnt()
                _wait_vmcnt()
                _gpu_barrier()
                return (
                    m_new,
                    l_init,
                    o_init[0],
                    o_init[1],
                    o_init[2],
                    o_init[3],
                    p_new,
                    prowsum_new,
                    corr_new,
                    *vwp[:PREFETCH_DEPTH],
                )

            # ---- Epilogue: apply deferred PV(N-1). ----
            def g0_epilogue(m_r, l_a, oo0, oo1, oo2, oo3, p_c, prowsum_c, corr_c, *vwp):
                _, k_buf_off_e, _, _, _ = _bufs(last_i * fx.Index(BLOCK_N))
                o, l2, _ = apply_pv(
                    [oo0, oo1, oo2, oo3],
                    l_a,
                    p_c,
                    prowsum_c,
                    corr_c,
                    v_last_off,
                    list(vwp),
                    k_buf_off_e,
                )
                return o, l2

            # ---- Main loop: tiles 1..N-1 (apply_pv + QK + softmax + DMA). ----
            g0_carry = list(g0_iter0())
            for_op = scf.ForOp(
                _raw(loop_step),
                _raw(seq_len),
                _raw(loop_step),
                [_raw(v) for v in g0_carry],
            )
            with ir.InsertionPoint(for_op.body):
                iv = for_op.induction_variable
                _iglp()
                _g0_args = [
                    as_dsl_value(a, ex)
                    for a, ex in zip(for_op.inner_iter_args, g0_carry)
                ]
                m_r, l_a, oo0, oo1, oo2, oo3, p_c, prowsum_c, corr_c = _g0_args[:9]
                vwp = _g0_args[9:]
                kv_start = fx.Index(iv)
                _, k_buf_off, v_prev_off, _, v_next = _bufs(kv_start)
                next_kv = kv_start + fx.Index(BLOCK_N)
                i_cur = kv_start // fx.Index(BLOCK_N)
                v_cur_off = fx.Index(LDS_V_OFF) + (
                    i_cur % fx.Index(NUM_BUF_V)
                ) * fx.Index(LDS_V_TILE)
                # PHASE 1 (mfma phase): deferred PV(i-1) then QK(i).
                oA, lA, kw_prime = apply_pv(
                    [oo0, oo1, oo2, oo3],
                    l_a,
                    p_c,
                    prowsum_c,
                    corr_c,
                    v_prev_off,
                    preloaded_vw=vwp,
                    k_buf_off=k_buf_off,
                )
                sA, vwp_new = do_qk(
                    k_buf_off,
                    preloaded_kw=kw_prime,
                    v_off=v_cur_off,
                    seed=(_bias_seed(m_r) if const_expr(USE_BIAS_FOLD) else None),
                )
                # PHASE 2 (softmax phase): softmax(i) | DMA V^{i+1}.
                dma_v(v_next, next_kv)
                m_new, corr_new, p_new, prowsum_new = do_softmax(
                    sA, m_r, first_tile=False, seeded=const_expr(USE_BIAS_FOLD)
                )
                rocdl.sched_barrier(0)
                _wait_lgkmcnt()
                _wait_vmcnt()
                _gpu_barrier()
                scf.YieldOp(
                    [
                        _raw(m_new),
                        _raw(lA),
                        _raw(oA[0]),
                        _raw(oA[1]),
                        _raw(oA[2]),
                        _raw(oA[3]),
                        _raw(p_new),
                        _raw(prowsum_new),
                        _raw(corr_new),
                        *[_raw(w) for w in vwp_new[:PREFETCH_DEPTH]],
                    ]
                )

            _g0_res = [as_dsl_value(r, ex) for r, ex in zip(for_op.results, g0_carry)]
            o_fin, l_fin_g = g0_epilogue(*_g0_res)
            of0 = o_fin[0]
            of1 = o_fin[1]
            of2 = o_fin[2]
            of3 = o_fin[3]
            lf = l_fin_g
        else:

            def g1_iter0():
                _, k_buf_off, _, _, _ = _bufs(fx.Index(0))

                v0_off = fx.Index(LDS_V_OFF)  # V^0 buf = LDSV[0]
                sB, vwp = do_qk(k_buf_off, preloaded_kw=kvw, v_off=v0_off)
                # Trick 2: G1 defers softmax by one tile, but S(0) is available
                # right here -- so freeze m in the prologue and let every
                # in-loop softmax be unconditionally frozen, rather than
                # carrying a predicated first-tile reduction through the loop.
                m_frozen = (
                    _rowmax(sB, m_init) if const_expr(USE_FROZEN_MAX) else m_init
                )
                if const_expr(USE_BIAS_FOLD):
                    # Tile 0's GEMM could not be seeded -- it is the GEMM that
                    # decides m.  So apply the bias here instead, once, in the
                    # prologue, and hand the loop an S in the same seeded form
                    # every later tile produces.  Without this the first
                    # iteration's softmax reads a raw S as if it were a pattern.
                    bseed = _bias_seed(m_frozen)
                    sB = [_fadd(Vec(x), bseed) for x in sB]
                _wait_lgkmcnt()
                _wait_vmcnt()
                _gpu_barrier()
                return (
                    m_frozen,
                    l_init,
                    o_init[0],
                    o_init[1],
                    o_init[2],
                    o_init[3],
                    sB[0],
                    sB[1],
                    sB[2],
                    sB[3],
                    *vwp[:PREFETCH_DEPTH],
                )

            # ---- Epilogue: softmax(N-1) then apply deferred PV(N-1). ----
            def g1_epilogue(m_e, l_e, oe0, oe1, oe2, oe3, sce0, sce1, sce2, sce3, *vwp):
                # rocdl.s_setprio(1)
                # G1 defers every softmax by one tile, so if the loop ran zero
                # iterations this call *is* tile 0's -- hence "maybe".
                _m_e, corrf, pf, prowsumf = do_softmax(
                    [sce0, sce1, sce2, sce3],
                    m_e,
                    first_tile=False,
                    seeded=const_expr(USE_BIAS_FOLD),
                )
                _, k_buf_off_e, _, _, _ = _bufs(last_i * fx.Index(BLOCK_N))
                o, l2, _ = apply_pv(
                    [oe0, oe1, oe2, oe3],
                    l_e,
                    pf,
                    prowsumf,
                    corrf,
                    v_last_off,
                    list(vwp),
                    k_buf_off_e,
                )
                # rocdl.s_setprio(0)
                return o, l2

            # ---- Main loop: tiles 1..N-1 (softmax + DMA K + apply_pv + QK). ----
            g1_carry = list(g1_iter0())
            for_op = scf.ForOp(
                _raw(loop_step),
                _raw(seq_len),
                _raw(loop_step),
                [_raw(v) for v in g1_carry],
            )
            with ir.InsertionPoint(for_op.body):
                rocdl.s_setprio(1)
                iv = for_op.induction_variable
                _iglp()
                _g1_args = [
                    as_dsl_value(a, ex)
                    for a, ex in zip(for_op.inner_iter_args, g1_carry)
                ]
                m_r, l_a, oo0, oo1, oo2, oo3, ss0, ss1, ss2, ss3 = _g1_args[:10]
                vwp = _g1_args[10:]
                kv_start = fx.Index(iv)
                _, k_buf_off, v_prev_off, k_next, _ = _bufs(kv_start)
                next_kv = kv_start + fx.Index(BLOCK_N)
                dma_k(k_next, next_kv)
                i_cur = kv_start // fx.Index(BLOCK_N)
                v_cur_off = fx.Index(LDS_V_OFF) + (
                    i_cur % fx.Index(NUM_BUF_V)
                ) * fx.Index(LDS_V_TILE)
                # PHASE 1 (mfma phase): softmax(i-1) | DMA K^{i+1}.
                m_sm, corr_sm, p_sm, prowsum_sm = do_softmax(
                    [ss0, ss1, ss2, ss3],
                    m_r,
                    first_tile=False,
                    seeded=const_expr(USE_BIAS_FOLD),
                )
                # PHASE 2 (softmax phase): deferred PV(i-1) + QK(i)->S.
                oB, lB, kw_prime = apply_pv(
                    [oo0, oo1, oo2, oo3],
                    l_a,
                    p_sm,
                    prowsum_sm,
                    corr_sm,
                    v_prev_off,
                    preloaded_vw=vwp,
                    k_buf_off=k_buf_off,
                )
                rocdl.s_setprio(0)
                sB, vwp_new = do_qk(
                    k_buf_off,
                    preloaded_kw=kw_prime,
                    v_off=v_cur_off,
                    # m_sm, not m_r: if this tile's softmax just repaired an
                    # overflow, the seed must carry the raised reference.
                    seed=(_bias_seed(m_sm) if const_expr(USE_BIAS_FOLD) else None),
                )
                rocdl.sched_barrier(0)
                _wait_lgkmcnt()
                _wait_vmcnt()
                _gpu_barrier()
                scf.YieldOp(
                    [
                        _raw(m_sm),
                        _raw(lB),
                        _raw(oB[0]),
                        _raw(oB[1]),
                        _raw(oB[2]),
                        _raw(oB[3]),
                        _raw(sB[0]),
                        _raw(sB[1]),
                        _raw(sB[2]),
                        _raw(sB[3]),
                        *[_raw(w) for w in vwp_new[:PREFETCH_DEPTH]],
                    ]
                )

            _g1_res = [as_dsl_value(r, ex) for r, ex in zip(for_op.results, g1_carry)]
            o_fin, l_fin_g = g1_epilogue(*_g1_res)
            of0 = o_fin[0]
            of1 = o_fin[1]
            of2 = o_fin[2]
            of3 = o_fin[3]
            lf = l_fin_g

        o_finals = [of0, of1, of2, of3]
        # L is the scalar VALU row-sum normalizer for this lane (q = lo).
        l_final = lf

        # Appendix D: guard 1/L.  A row that saw no in-bounds kv (or whose every
        # score underflowed) has L == 0; rcp would yield +inf and inf*0 -> NaN.
        # Substitute 0 so the row stores zeros.  v_descale is folded into the
        # reciprocal so the dequant costs one scalar mul, not 64 vector muls.
        inv_l = rocdl.rcp(T.f32, l_final)
        inv_l_v = _fmul(inv_l, v_descale)
        l_is_zero = arith.cmpf(arith.CmpFPredicate.OEQ, _raw(l_final), _raw(c_zero_f))
        inv_l_v = ArithValue(l_is_zero).select(c_zero_f, fx.Float32(inv_l_v))
        inv_l_vec = Vec.from_elements([inv_l_v], fx.Float32).broadcast_to(
            C_F32_PER_LANE
        )

        if q_in_bounds:
            for dt in range_constexpr(D_TILES):
                o_norm = Vec(o_finals[dt]) * inv_l_vec
                # O C-layout: row = d = hi*4 + (r%4) + 8*(r//4) (+ dt*32);
                # col = q = lo.  But this wave's q = q_row (= lo-based).
                # For fixed rg = r//4 the four r%4 lanes are contiguous in d, so
                # a group of 4 packs into one 8-byte bf16x4 store (16 stores per
                # wave instead of 64 scalar ones).
                for rg in range_constexpr(C_F32_PER_LANE // 4):
                    w0 = rocdl.cvt_pk_bf16_f32(
                        _raw(Vec(o_norm)[rg * 4 + 0]), _raw(Vec(o_norm)[rg * 4 + 1])
                    )
                    w1 = rocdl.cvt_pk_bf16_f32(
                        _raw(Vec(o_norm)[rg * 4 + 2]), _raw(Vec(o_norm)[rg * 4 + 3])
                    )
                    packed = Vec.from_elements([w0, w1], fx.Int32).bitcast(bf16_dtype)
                    d_row = hi * 4 + 8 * rg + dt * 32
                    o_global = global_idx(q_row, fx.Index(d_row))
                    gep = fx.buffer_ops.get_element_ptr(
                        o_ptr, fx.Int64(o_global), elem_type=bf16_type
                    )
                    llvm.StoreOp(
                        _llvm_value(packed.ir_value()),
                        _llvm_value(gep),
                        alignment=8,
                    )

    @flyc.jit
    def launch_fp8_attn(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        q_descale: fx.Float32,
        k_descale: fx.Float32,
        v_descale: fx.Float32,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        bs_idx = fx.Index(batch_size)
        sl_idx = fx.Index(seq_len)
        num_q_tiles = (sl_idx + BLOCK_M - 1) // BLOCK_M
        grid_x = bs_idx * num_q_tiles * NUM_HEADS

        fp8_attn_kernel(
            Q,
            K,
            V,
            O,
            q_descale,
            k_descale,
            v_descale,
            seq_len,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu,
                "rocdl.flat_work_group_size": f"{BLOCK_SIZE},{BLOCK_SIZE}",
            },
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    def _compile(  # noqa: E741
        Q,
        K,
        V,
        O,  # noqa: E741
        q_descale,
        k_descale,
        v_descale,
        batch_size,
        seq_len,
        stream=None,
    ):
        return flyc.compile(
            launch_fp8_attn,
            Q,
            K,
            V,
            O,
            q_descale,
            k_descale,
            v_descale,
            batch_size,
            seq_len,
            fx.Stream(stream),
        )

    launch_fp8_attn.compile = _compile
    return launch_fp8_attn


compile_flash_attn_fp8 = build_flash_attn_fp8_module
