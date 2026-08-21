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

MFMA_M = 32
MFMA_N = 32
MFMA_K = 64
WARP_SIZE = 64
A_FP8_PER_LANE = 32
C_F32_PER_LANE = 16

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
    BLOCK_SIZE = NUM_WAVES * WARP_SIZE
    ROWS_PER_WAVE = BLOCK_M // NUM_WAVES
    STRIDE_TOKEN = NUM_HEADS * HEAD_DIM

    K_STEPS = HEAD_DIM // MFMA_K
    N_KV_TILES = BLOCK_N // MFMA_N
    D_TILES = HEAD_DIM // MFMA_N
    PV_K_STEPS = BLOCK_N // MFMA_K

    if softmax_scale is None:
        softmax_scale = 1.0 / host_math.sqrt(head_dim)

    K_STRIDE = HEAD_DIM
    V_KV_STRIDE = 16
    N_DBLOCKS = HEAD_DIM // 16

    PAD_K = 16
    PAD_V = 16
    K_UNIT_ROWS = 8
    K_DATA = BLOCK_N * K_STRIDE
    K_UNIT_STRIDE = K_UNIT_ROWS * K_STRIDE + PAD_K
    N_K_UNITS = BLOCK_N // K_UNIT_ROWS
    V_DBLOCK_STRIDE = BLOCK_N * V_KV_STRIDE + PAD_V

    NUM_BUF_K = 2
    # V stays 3-deep. The asm gets away with 2 (kLdsSizeV2) because its loop
    # is cut QK -> softmax -> PV, so the PV consumes the very tile the QK tail
    # just prefetched and only two V tiles are live per iteration. This loop is
    # cut PV -> QK -> softmax, so tile i-1 (PV), tile i (QK prefetch) and tile
    # i+1 (DMA) are all live at once. Dropping to 2 needs the loop re-cut, not
    # just a smaller buffer -- see the note in do_qk about the tail barrier.
    NUM_BUF_V = 3
    LDS_K_TILE = N_K_UNITS * K_UNIT_STRIDE
    LDS_V_TILE = N_DBLOCKS * V_DBLOCK_STRIDE
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
    SOFTMAX_PRIO = 1

    USE_SCHRAUDOLPH = True
    USE_ROLLBACK = False

    USE_QK_SCALE_FOLD = True

    USE_FROZEN_MAX = True

    USE_WATCHDOG = True

    USE_BIAS_FOLD = True

    # How many of the four KV tiles reduce their L denominator on the VALU
    # pipe; the rest go through the ones-column MFMA, which is issued at the
    # head of the *next* MFMA phase (in apply_pv) rather than at the tail of
    # the softmax. 2 splits it exactly like the asm: the first half is
    # _L_k0_sum_fp32() on the VALU during softmax, the second half is
    # _matmul_l_issue() at the top of gemm_PV.
    L_SPLIT_TILES = 2
    # p_pack is contracted in PV_K_STEPS slabs of two KV tiles each, so the
    # VALU covers slabs [0, L_SPLIT_TILES//2) and the MFMA the rest.
    L_MFMA_SLABS = PV_K_STEPS - L_SPLIT_TILES // 2

    SCHRAUDOLPH_2P23 = 1 << 23
    SCHRAUDOLPH_C = 127 * SCHRAUDOLPH_2P23 - 486411
    E8M0_ONE = 0x7F7F7F7F
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
        assert USE_SCHRAUDOLPH, "the seed is the Schraudolph bias"
        assert USE_FROZEN_MAX, "an unfrozen m cannot be seeded before its own GEMM"
    EXP2_SHIFT = 4.0

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def fp8_attn_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,
        q_descale: fx.Float32,
        k_descale: fx.Float32,
        v_descale: fx.Float32,
        seq_len: fx.Int32,
    ):
        i8_dtype = fx.Int8
        i8_type = i8_dtype.ir_type
        bf16_type = bf16_dtype.ir_type
        v_i8x16 = Vec.make_type(16, i8_dtype)
        v_i8x32 = Vec.make_type(A_FP8_PER_LANE, i8_dtype)

        fm_fast = fx.arith.FastMathFlags.fast

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
            packed = rocdl.cvt_pk_fp8_f32(
                T.i32, _raw(f), _raw(c_zero_f), fx.Int32(0), False
            )
            return arith.trunci(T.i8, _raw(packed))

        v2i16_ty = Vec.make_type(2, fx.Int16)
        _zero_2xi16 = Vec.filled(2, 0, fx.Int16)

        def _f32x4_to_fp8_word(f0, f1, f2, f3, rscale=None):
            if const_expr(rscale is None):
                w0 = rocdl.cvt_pk_fp8_f32(T.i32, _raw(f0), _raw(f1), fx.Int32(0), False)
                w1 = rocdl.cvt_pk_fp8_f32(T.i32, _raw(f2), _raw(f3), _raw(w0), True)
                return w1
            # Chain through the `old` operand with opsel, exactly as the
            # unscaled path above: the second convert writes the high half of
            # the register the first one produced, so the word needs no
            # and/or/shl to assemble.
            lo = rocdl.cvt_scalef32_pk_fp8_f32(
                v2i16_ty, _raw(_zero_2xi16), _raw(f0), _raw(f1), _raw(rscale), False
            )
            hi = rocdl.cvt_scalef32_pk_fp8_f32(
                v2i16_ty, _raw(lo), _raw(f2), _raw(f3), _raw(rscale), True
            )
            return Vec(Vec(hi).bitcast(fx.Int32))[0]

        mfma = Mfma32x32x64()

        q_ptr = _extract_aligned_pointer(Q)
        k_ptr = _extract_aligned_pointer(K)
        v_ptr = _extract_aligned_pointer(V)
        o_ptr = _extract_aligned_pointer(O)

        seq_len_v = fx.Index(seq_len)

        base_ptr = allocator.get_base()
        lds = SmemPtr(base_ptr, lds_offset, i8_type, shape=(LDS_TOTAL,)).get()

        block_id = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)
        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lo = lane % 32
        hi = lane // 32

        wave_q_offset = wave_id * ROWS_PER_WAVE

        head_idx = block_id % NUM_HEADS
        batch_q_tile_id = block_id // NUM_HEADS
        num_q_tiles = (seq_len_v + BLOCK_M - 1) // BLOCK_M
        q_tile_idx = batch_q_tile_id % num_q_tiles
        batch_idx = batch_q_tile_id // num_q_tiles
        q_start = q_tile_idx * BLOCK_M

        def global_idx(token_idx, col):
            token = batch_idx * seq_len_v + token_idx
            return token * STRIDE_TOKEN + head_idx * HEAD_DIM + col

        c_log2e = fx.Float32(_LOG2E)
        qk_scale = _fmul(_fmul(q_descale, k_descale), fx.Float32(softmax_scale))
        scale_log2e = _fmul(qk_scale, c_log2e)
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_f = fx.Float32(0.0)

        c_2p23 = fx.Float32(float(SCHRAUDOLPH_2P23))
        c_inv_2p23 = fx.Float32(2.0**-23)
        c_exp2_bias = fx.Float32(
            float(SCHRAUDOLPH_C) + EXP2_SHIFT * float(SCHRAUDOLPH_2P23)
        )
        c_sch_cap = fx.Float32(SCHRAUDOLPH_CAP)

        def _i32c(v):
            return fx.Int32(v - (1 << 32) if v >= (1 << 31) else v)

        e8m0_ident = _i32c(E8M0_ONE)
        qk_scale_e8m0 = e8m0_ident
        q_mant = None
        if const_expr(USE_QK_SCALE_FOLD):
            a_full = _fmul(scale_log2e, c_2p23)
            a_bits = arith.bitcast(T.i32, _raw(a_full))
            e_byte = arith.addi(
                arith.andi(
                    arith.shrui(_raw(a_bits), _raw(fx.Int32(23))),
                    _raw(fx.Int32(0xFF)),
                ),
                _raw(fx.Int32(1)),
            )
            qk_scale_e8m0 = arith.muli(_raw(e_byte), _raw(fx.Int32(0x01010101)))
            q_mant = arith.bitcast(
                T.f32,
                arith.ori(
                    arith.andi(_raw(a_bits), _raw(_i32c(0x807FFFFF))),
                    _raw(fx.Int32(0x3F000000)),
                ),
            )

        def qk_mfma(a, b, c):
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
            biased = _fmax(_fadd(_fmul(s, scale_x2p23), m_term), c_zero_f)
            return arith.bitcast(T.f32, arith.fptoui(T.i32, _raw(biased)))

        DMA_BYTES = 16
        DMA_LANES = (NUM_WAVES // 2) * WARP_SIZE
        DMA_PASSES = K_DATA // (DMA_LANES * DMA_BYTES)
        WAVE_DMA_STRIDE = WARP_SIZE * DMA_BYTES

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

        def _dma_lds_ptr(lds_byte_off):
            """SALU-only: the LDS destination is wave-uniform."""
            lds_addr = rocdl.readfirstlane(T.i64, arith.index_cast(T.i64, lds_byte_off))
            return llvm.inttoptr(_lds_ptr_ty, lds_addr)

        def _dma_fire(rsrc, lds_ptr, voff_i32):
            rocdl.raw_ptr_buffer_load_lds(
                rsrc, lds_ptr, _dma_size, voff_i32, _dma_zero, _dma_zero, _dma_aux
            )

        def _dma_issue(rsrc, lds_byte_off, voffset_idx):
            _dma_fire(
                rsrc,
                _dma_lds_ptr(lds_byte_off),
                arith.index_cast(T.i32, voffset_idx),
            )

        _ltid = tid - fx.Index(DMA_LANES)
        _lwave = wave_id - fx.Index(NUM_WAVES // 2)
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
            blk = kv_lds_pos // fx.Index(32)
            rem = kv_lds_pos % fx.Index(32)
            hi_group = (rem // fx.Index(16)) % fx.Index(2)
            grp = (rem // fx.Index(4)) % fx.Index(4)
            fine = rem % fx.Index(4)
            kv = blk * fx.Index(32) + hi_group * fx.Index(4) + grp * fx.Index(8) + fine
            d_voff = d_block * fx.Index(16)
            lds_perm = d_block * fx.Index(V_DBLOCK_STRIDE) + _half * fx.Index(
                WAVE_DMA_STRIDE
            )
            _dma_v_inv.append((kv, d_voff, lds_perm))

        def dma_k_voffs(kv_start):
            """Per-pass voffsets for the K tile DMA.

            The asm emits the address setup once (emit_next_K_mem_addr) and
            then only the buffer_loads inside the QK loop. These are the VALU
            half of that; they are loop-carried so they get computed during the
            softmax phase, one iteration ahead of the loads that use them.
            """
            out = []
            for p in range_constexpr(DMA_PASSES):
                kv, d, _ = _dma_k_inv[p]
                kv_abs = kv_start + kv
                in_b = kv_abs < seq_len_v
                kv_safe = fx.Index(ArithValue(in_b).select(kv_abs, fx.Index(0)))
                out.append(
                    arith.index_cast(
                        T.i32, _raw(kv_safe * fx.Index(STRIDE_TOKEN) + d)
                    )
                )
            return out

        def dma_k_lds(k_off):
            base_lds = fx.Index(lds_offset) + k_off
            return [
                _dma_lds_ptr(base_lds + _dma_k_inv[p][2])
                for p in range_constexpr(DMA_PASSES)
            ]

        def dma_k_off(k_off, kv_start):
            """dma_k addressed by a precomputed LDS byte offset."""
            voffs = dma_k_voffs(kv_start)
            for p, ptr in enumerate(dma_k_lds(k_off)):
                _dma_fire(k_rsrc, ptr, voffs[p])

        def dma_k(buf, kv_start):
            dma_k_off(fx.Index(LDS_K_OFF) + buf * fx.Index(LDS_K_TILE), kv_start)

        def dma_v_off(v_off, kv_start):
            """dma_v addressed by a precomputed LDS byte offset.

            The hot loops carry the V buffer offset and rotate it, so they never
            need i % NUM_BUF_V (=3), which lowers to a 64-bit magic-multiply
            divide -- ~40 SALU emitted three times per iteration.
            """
            voffs = dma_v_voffs(kv_start)
            for p, ptr in enumerate(dma_v_lds(v_off)):
                _dma_fire(v_rsrc, ptr, voffs[p])

        def dma_v_voffs(kv_start):
            """Per-pass voffsets for the V tile DMA; see dma_k_voffs."""
            out = []
            for p in range_constexpr(DMA_PASSES):
                kv, d_voff, _ = _dma_v_inv[p]
                kv_abs = kv_start + kv
                in_b = kv_abs < seq_len_v
                kv_safe = fx.Index(ArithValue(in_b).select(kv_abs, fx.Index(0)))
                out.append(
                    arith.index_cast(
                        T.i32, _raw(kv_safe * fx.Index(STRIDE_TOKEN) + d_voff)
                    )
                )
            return out

        def dma_v_lds(v_off):
            base_lds = fx.Index(lds_offset) + v_off
            return [
                _dma_lds_ptr(base_lds + _dma_v_inv[p][2])
                for p in range_constexpr(DMA_PASSES)
            ]

        def dma_v(buf, kv_start):
            dma_v_off(fx.Index(LDS_V_OFF) + buf * fx.Index(LDS_V_TILE), kv_start)

        def _v_slot(b):
            return fx.Index(LDS_V_OFF + b * LDS_V_TILE)

        def _k_slot(b):
            return fx.Index(LDS_K_OFF + b * LDS_K_TILE)

        def _wait_lgkmcnt(count=0):
            llvm.InlineAsmOp(
                None, [], f"s_waitcnt lgkmcnt({count})", "", has_side_effects=True
            )

        def _wait_vmcnt(count=0):
            llvm.InlineAsmOp(
                None, [], f"s_waitcnt vmcnt({count})", "", has_side_effects=True
            )

        def _pin(v):
            """Opaque identity: v is unchanged but its def cannot be moved."""
            return llvm.InlineAsmOp(
                T.i32, [_raw(v)], "", "=v,0", has_side_effects=False
            ).res

        def _gpu_barrier():
            # sched_barrier(0) on both sides: the inline-asm s_barrier has
            # side effects but no memory clobber, so the machine scheduler is
            # free to sink pure VALU across it. Without these fences half of
            # g0's softmax (the med3/cvt_u32/cvt_fp8 chain) migrates into the
            # following MFMA phase, which is exactly what this kernel is
            # trying to avoid.
            rocdl.sched_barrier(0)
            llvm.InlineAsmOp(None, [], "s_barrier", "", has_side_effects=True)
            rocdl.sched_barrier(0)

        q_row = q_start + wave_q_offset + lo
        q_in_bounds = q_row < seq_len_v
        q_row_safe = fx.Index(ArithValue(q_in_bounds).select(q_row, fx.Index(0)))
        zero_qpack = Vec.filled(A_FP8_PER_LANE, 0, i8_dtype)

        q_packs = []
        for ks in range_constexpr(K_STEPS):
            d_col = fx.Index(ks * MFMA_K) + hi * 32
            g_idx = global_idx(q_row_safe, d_col)
            gep = fx.buffer_ops.get_element_ptr(
                q_ptr, fx.Int64(g_idx), elem_type=i8_type
            )
            raw = _pointer_load(v_i8x32, gep)
            raw = ArithValue(q_in_bounds).select(raw, zero_qpack.ir_value())
            qw = Vec(raw).bitcast(fx.Int32)
            if const_expr(USE_QK_SCALE_FOLD):
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

        v_tr8_ty = Vec.make_type(2, fx.Int32)
        lo_in_grp = lo % fx.Index(16)

        def _v_base_ptr(v_off):
            byte_off = (
                fx.Index(lds_offset)
                + v_off
                + (lo // fx.Index(16)) * fx.Index(V_DBLOCK_STRIDE)
                + hi * fx.Index(16 * V_KV_STRIDE)
                + lo_in_grp * fx.Index(8)
            )
            return fx.buffer_ops.create_llvm_ptr(fx.Int64(byte_off), address_space=3)

        def read_v_pack(v_off, dt, ks):
            base = _v_base_ptr(v_off)
            reads = []
            for kc in range_constexpr(4):
                imm = (2 * dt) * V_DBLOCK_STRIDE + (
                    ks * 64 + (kc // 2) * 32 + (kc % 2) * 8
                ) * V_KV_STRIDE
                ptr = fx.buffer_ops.get_element_ptr(
                    base, static_byte_offset=imm, elem_type=i8_type
                )
                reads.append(Vec(rocdl.ds_read_tr8_b64(v_tr8_ty, ptr).result))
            ab = reads[0].shuffle(reads[1], list(range(4)))
            cd = reads[2].shuffle(reads[3], list(range(4)))
            return ab.shuffle(cd, list(range(8)))

        def apply_pv(
            o_accs,
            l_acc,
            lr_acc,
            p_pack,
            p_rowsum,
            corr,
            v_off,
            preloaded_vw,
            k_buf_off,
            prio=False,
        ):
            if const_expr(prio):
                # asm: s_setprio(1) at the head of gemm_PV -- the MFMA phase
                # outranks the peer half's softmax until gemm_QK drops it.
                rocdl.s_setprio(1)
            corr_vec = Vec.from_elements([corr], fx.Float32).broadcast_to(
                C_F32_PER_LANE
            )
            if const_expr(USE_ROLLBACK or USE_FROZEN_MAX):
                grew = arith.cmpf(
                    arith.CmpFPredicate.OLT, _raw(corr), _raw(fx.Float32(1.0))
                )
                any_grew = _wave_or(grew)
                o_vt = Vec.make_type(C_F32_PER_LANE, fx.Float32)
                _res_tys = [o_vt, o_vt, o_vt, o_vt, T.f32]
                if const_expr(L_MFMA_SLABS):
                    _res_tys.append(o_vt)
                resc_if = scf.IfOp(any_grew, results_=_res_tys, has_else=True)
                with ir.InsertionPoint(resc_if.then_block):
                    _ro = [
                        _fmul(Vec(o_accs[dt]), corr_vec)
                        for dt in range_constexpr(D_TILES)
                    ]
                    _rl = _fadd(_fmul(l_acc, corr), p_rowsum)
                    _y = [
                        _raw(_ro[0]),
                        _raw(_ro[1]),
                        _raw(_ro[2]),
                        _raw(_ro[3]),
                        _raw(_rl),
                    ]
                    if const_expr(L_MFMA_SLABS):
                        _y.append(_raw(_fmul(Vec(lr_acc), corr_vec)))
                    scf.YieldOp(_y)
                with ir.InsertionPoint(resc_if.else_block):
                    _nl = _fadd(_raw(l_acc), _raw(p_rowsum))
                    _y = [
                        _raw(o_accs[0]),
                        _raw(o_accs[1]),
                        _raw(o_accs[2]),
                        _raw(o_accs[3]),
                        _raw(_nl),
                    ]
                    if const_expr(L_MFMA_SLABS):
                        _y.append(_raw(lr_acc))
                    scf.YieldOp(_y)
                o_accs = [
                    as_dsl_value(resc_if.results[dt], o_accs[dt])
                    for dt in range_constexpr(D_TILES)
                ]
                l2 = as_dsl_value(resc_if.results[4], l_acc)
                if const_expr(L_MFMA_SLABS):
                    lr_acc = as_dsl_value(resc_if.results[5], lr_acc)
            else:
                l2 = _fadd(_fmul(l_acc, corr), p_rowsum)
            p_ks_list = [
                Vec(p_pack).shuffle(Vec(p_pack), list(range(r * 8, r * 8 + 8)))
                for r in range_constexpr(PV_K_STEPS)
            ]
            # The tail of the L denominator rides the MFMA pipe at the head of
            # the MFMA phase (asm: _matmul_l_issue at the top of gemm_PV); the
            # head of it was summed on the VALU during the softmax phase.
            for ks in range_constexpr(PV_K_STEPS - L_MFMA_SLABS, PV_K_STEPS):
                lr_acc = mfma.call(ones_pack, p_ks_list[ks], lr_acc)
            PV_UNITS = D_TILES * PV_K_STEPS
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
                _sched_barrier()
                o[dt] = mfma.call(vw[u % 4], p_ks_list[ks], o[dt])
                # MFMA first, LDS after -- the asm issues the ds_read for the
                # operands of a later MFMA behind the one that is already
                # runnable, so the MFMA pipe never waits on lgkmcnt.
                if const_expr(u + PREFETCH_DEPTH < PV_UNITS):
                    un = u + PREFETCH_DEPTH
                    vw[un % 4] = read_v_pack(
                        v_off, un // PV_K_STEPS, un % PV_K_STEPS
                    )
                else:
                    ki = u - (PV_UNITS - PREFETCH_DEPTH)
                    kw_prime[ki] = _load_k_unit(k_buf_off, ki // K_STEPS, ki % K_STEPS)
                _sched_barrier()
            return o, l2, lr_acc, kw_prime

        QK_UNITS = N_KV_TILES * K_STEPS

        # Two units deep, like the asm (k_lds_unit_idx starts at 2 in gemm_QK
        # and v_lds_unit_idx at 2 in gemm_PV): the MFMA at index u consumes
        # operands read two MFMAs ago, so one full ds_read latency is hidden.
        PREFETCH_DEPTH = 2

        K_NT_STRIDE = 32 * K_STRIDE + (32 // K_UNIT_ROWS) * PAD_K

        def _k_base_row(k_buf_off):
            return (
                k_buf_off
                + lo * K_STRIDE
                + (lo // fx.Index(K_UNIT_ROWS)) * fx.Index(PAD_K)
                + hi * 32
            )

        def _load_k_unit(k_buf_off, nt, ks, base=None):
            if const_expr(base is None):
                base = _k_base_row(k_buf_off)
            imm = nt * K_NT_STRIDE + ks * MFMA_K
            blk_lo = Vec(
                Vec.load(v_i8x16, lds, [base + fx.Index(imm)])
            ).bitcast(fx.Int32)
            blk_hi = Vec(
                Vec.load(v_i8x16, lds, [base + fx.Index(imm + 16)])
            ).bitcast(fx.Int32)
            return blk_lo.shuffle(blk_hi, list(range(8)))

        def _load_k_unit_global(kv_start, nt, ks):
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

        def do_qk(k_buf_off, preloaded_kw, v_off, seed=None, dma=None):
            """QK GEMM, scheduled like the asm's ``gemm_QK``.

            Each unit issues the MFMA first, then the LDS read feeding the
            MFMA two units later. ``dma`` is ``(rsrc, lds_ptrs, voffs)``; the
            voffs were computed in the previous iteration's softmax phase, so
            the MFMA phase pays only the buffer_load itself.

            The DMA passes are fired as a block *after* the last ds_read
            rather than interleaved the way the asm does it. The asm can
            interleave because it writes its own s_waitcnt; here the backend's
            waitcnt inserter cannot prove that a ``buffer_load ... lds`` and a
            following ``ds_read`` touch disjoint LDS, so every ds_read placed
            after a DMA gets an ``s_waitcnt vmcnt(0)`` in front of it -- a full
            HBM round trip, four of them per phase. Issuing the DMAs last
            keeps the loads in flight across the whole softmax phase, which is
            the point of the prefetch.
            """
            kw = [None] * 4
            vw_prime = [None] * PREFETCH_DEPTH
            for u in range_constexpr(PREFETCH_DEPTH):
                kw[u] = preloaded_kw[u]
            s_accs = [
                (mfma.zero_value if const_expr(seed is None) else seed)
                for _ in range_constexpr(N_KV_TILES)
            ]
            dma_rsrc, dma_ptrs, dma_voffs = (
                (None, [], []) if const_expr(dma is None) else dma
            )
            for u in range_constexpr(QK_UNITS):
                nt = u // K_STEPS
                ks = u % K_STEPS
                _sched_barrier()
                s_accs[nt] = qk_mfma(kw[u % 4], q_packs[ks], s_accs[nt])
                if const_expr(u + PREFETCH_DEPTH < QK_UNITS):
                    un = u + PREFETCH_DEPTH
                    kw[un % 4] = _load_k_unit(
                        k_buf_off, un // K_STEPS, un % K_STEPS
                    )
                else:
                    vi = u - (QK_UNITS - PREFETCH_DEPTH)
                    vw_prime[vi] = read_v_pack(v_off, vi // PV_K_STEPS, vi % PV_K_STEPS)
                _sched_barrier()
            # Every ds_read of this phase has now been issued, so the DMA can
            # no longer force a vmcnt drain in front of one.
            for _dp in range_constexpr(len(dma_ptrs)):
                _dma_fire(dma_rsrc, dma_ptrs[_dp], dma_voffs[_dp])
            _sched_barrier()
            # asm: the s_barrier + s_setprio(0) that close gemm_QK. This is the
            # rendezvous with the peer half, which is closing its softmax --
            # the two groups run opposite phases, so one barrier pairs them.
            #
            # No vmcnt drain here: the DMA this GEMM just issued is not read
            # until the *next* iteration's GEMM, so it drains at the end of the
            # softmax phase instead. Draining it here would stall on a load
            # issued a few instructions ago and cost the whole tile latency.
            _gpu_barrier()
            rocdl.s_setprio(0)
            return s_accs, vw_prime

        def _lane_max(vecs):
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
            w = C_F32_PER_LANE
            while const_expr(w > 1):
                h = w // 2
                a = Vec(v).shuffle(Vec(v), list(range(h)))
                b = Vec(v).shuffle(Vec(v), list(range(h, w)))
                v = _fadd(a, b)
                w = h
            return Vec(v)[0]

        def _rowmax(s_accs, m_running):
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
            return Vec.from_elements(
                [_fadd(_fsub(c_zero_f, m), c_exp2_bias)], fx.Float32
            ).broadcast_to(C_F32_PER_LANE)

        # ---- MFMA bias prefill (asm: _qk_bias_split + _qk_bias_prefill) ----
        #
        # Broadcasting the seed into a 16-wide MFMA accumulator costs 16
        # v_mov_b32 inside the MFMA phase. Instead split the seed into three
        # bf16 chunks on the VALU during softmax and let one
        # mfma_f32_32x32x8_bf16 splat it across the accumulator: A is a column
        # of ones so every row gets chunk0+chunk1+chunk2, which reconstructs
        # the f32 seed to ~24 bits.
        def _f32_of_bf16_hi(x_i32):
            """f32 value of the bf16 in the low half of x_i32."""
            return arith.bitcast(
                T.f32, arith.shli(_raw(x_i32), _raw(fx.Int32(16)))
            )

        def _bias_split(m):
            """Three-chunk bf16 decomposition of the Schraudolph seed."""
            d = _fadd(_fsub(c_zero_f, m), c_exp2_bias)
            c0 = rocdl.cvt_pk_bf16_f32(_raw(d), _raw(d))
            r1 = _fsub(d, _f32_of_bf16_hi(c0))
            c1 = rocdl.cvt_pk_bf16_f32(_raw(r1), _raw(r1))
            r2 = _fsub(r1, _f32_of_bf16_hi(c1))
            return (
                rocdl.cvt_pk_bf16_f32(_raw(d), _raw(r1)),
                rocdl.cvt_pk_bf16_f32(_raw(r2), _raw(c_zero_f)),
            )

        # A operand: bf16 [1, 1, 1, 0] on the k=0..3 half, zero on k=4..7 so
        # only the three chunks contribute.
        _hi0 = hi == fx.Index(0)
        _bias_ones = Vec(
            Vec.from_elements(
                [
                    ArithValue(_hi0).select(fx.Int32(_i32c(0x3F803F80)), fx.Int32(0)),
                    ArithValue(_hi0).select(fx.Int32(0x00003F80), fx.Int32(0)),
                ],
                fx.Int32,
            )
        ).bitcast(fx.Int16)

        def _bias_prefill(chunks):
            b = Vec(Vec.from_elements(list(chunks), fx.Int32)).bitcast(fx.Int16)
            return rocdl.mfma_f32_32x32x8bf16_1k(
                mfma.accum_type,
                [_raw(_bias_ones), _raw(b), _raw(mfma.zero_value), 0, 0, 0],
            )

        def _watchdog_seeded(s_accs, m_frozen):
            t = _lane_max(s_accs)
            peer = fx.Float32(t).shuffle_xor(fx.Int32(32), fx.Int32(WARP_SIZE))
            e = _fmax(_fsub(_fmax(t, peer), fx.Float32(SCHRAUDOLPH_CAP)), c_zero_f)
            eu = _fmul(e, c_inv_2p23)
            rscale = rocdl.exp2(T.f32, _raw(eu))
            corr = rocdl.exp2(T.f32, _raw(_fsub(c_zero_f, eu)))
            if const_expr(USE_QK_SCALE_FOLD):
                delta = e
            else:
                delta = arith.divf(_raw(eu), _raw(scale_log2e))
            return _fadd(_raw(m_frozen), _raw(delta)), corr, rscale

        def _watchdog(s_accs, m_term, m_frozen):
            t = _fadd(_lane_max(s_accs), m_term)
            peer = fx.Float32(t).shuffle_xor(fx.Int32(32), fx.Int32(WARP_SIZE))
            e = _fmax(_fsub(_fmax(t, peer), fx.Float32(SCHRAUDOLPH_CAP)), c_zero_f)

            if const_expr(USE_QK_SCALE_FOLD):
                delta = e
            else:
                delta = arith.divf(
                    _raw(_fmul(e, c_inv_2p23)), _raw(scale_log2e)
                )
            corr = rocdl.exp2(T.f32, _raw(_fsub(c_zero_f, _fmul(e, c_inv_2p23))))
            return _fsub(m_term, e), _fadd(_raw(m_frozen), _raw(delta)), corr

        def do_softmax(
            s_accs,
            m_running,
            set_prio=False,
            first_tile=True,
            seeded=False,
            barrier=False,
        ):
            if const_expr(SOFTMAX_PRIO != 0 and set_prio):
                rocdl.s_setprio(SOFTMAX_PRIO)
            if const_expr(USE_QK_SCALE_FOLD):
                to_log2 = c_inv_2p23
            else:
                to_log2 = scale_log2e

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
            v_partial = None
            rscale = None
            if const_expr(USE_SCHRAUDOLPH):
                if const_expr(USE_QK_SCALE_FOLD):
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
                    if const_expr(USE_WATCHDOG):
                        m_new, corr, rscale = _watchdog_seeded(s_accs, m_new)
                elif const_expr(frozen and USE_WATCHDOG):
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
                        # clamp to [0, cap] -- v_med3_f32 does both ends in
                        # one instruction, halving the VALU here. FMED3 has no
                        # vector pattern, so build the vector element-wise.
                        _s = Vec(s_accs[nt])
                        biased = Vec.from_elements(
                            [
                                rocdl.fmed3(
                                    T.f32,
                                    _raw(_s[r]),
                                    _raw(c_zero_f),
                                    _raw(c_sch_cap),
                                )
                                for r in range_constexpr(C_F32_PER_LANE)
                            ],
                            fx.Float32,
                        )
                    else:
                        if const_expr(USE_QK_SCALE_FOLD):
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
            # Pin the P words to this block. g0 loop-carries p_pack across the
            # back-edge, so without this the scheduler sinks half the
            # producer chain (32 med3 + 32 cvt_u32 + 32 cvt_fp8) into the
            # consumer -- i.e. straight into the next MFMA phase. sched_barrier
            # is intra-block and cannot stop that; an opaque asm identity can,
            # because the value is only materialised once it is read out.
            p_words = [_pin(w) for w in p_words]
            p_pack = Vec.from_elements(p_words, fx.Int32)
            # The MFMA half of the L reduction has moved to the head of the
            # next MFMA phase (apply_pv); the softmax phase is VALU-only now.
            if const_expr(v_partial is None):
                p_rowsum = c_zero_f
            else:
                v_sum = _lane_sum(v_partial)
                v_sum = _fadd(
                    v_sum,
                    fx.Float32(v_sum).shuffle_xor(fx.Int32(32), fx.Int32(WARP_SIZE)),
                )
                if const_expr(rscale is not None):
                    v_sum = _fmul(v_sum, corr)
                p_rowsum = v_sum
            if const_expr(SOFTMAX_PRIO != 0 and set_prio):
                rocdl.s_setprio(0)
            # Split the next tile's QK seed into bf16 chunks here, on the VALU,
            # so the MFMA phase only pays the one prefill MFMA.
            bias_chunks = _bias_split(m_new) if const_expr(USE_BIAS_FOLD) else None
            if const_expr(barrier):
                # asm: the trailing s_barrier of _softmax_rescale_R_frozen,
                # preceded by the vmcnt(0) that publishes the tile the previous
                # GEMM DMA'd. A whole phase of VALU has run since the loads
                # issued, so this drain is nearly free.
                _wait_vmcnt()
                _wait_lgkmcnt()
                _gpu_barrier()
            return m_new, corr, p_pack, p_rowsum, bias_chunks

        ones_pack = Vec.filled(A_FP8_PER_LANE // 4, 0x38383838, fx.Int32)

        m_init = c_neg_inf
        l_init = c_zero_f
        lr_init = mfma.zero_value
        o_init = [
            Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)
            for _ in range_constexpr(D_TILES)
        ]
        N_BIAS_CHUNKS = 2 if USE_BIAS_FOLD else 0

        is_g0 = wave_id < fx.Index(NUM_WAVES // 2)

        kvw = [None] * PREFETCH_DEPTH
        for u in range_constexpr(PREFETCH_DEPTH):
            kvw[u] = _load_k_unit_global(fx.Index(0), u // K_STEPS, u % K_STEPS)

        # Tiles 0 and 1 are prefetched into buffers 0 and 1; from there on the
        # loops rotate their carried LDS byte offsets.
        if is_g0:
            dma_v(fx.Index(0), fx.Index(0))
            dma_v(fx.Index(1), fx.Index(BLOCK_N))
        else:
            dma_k(fx.Index(0), fx.Index(0))
            dma_k(fx.Index(1), fx.Index(BLOCK_N))

        _wait_vmcnt()
        _wait_lgkmcnt()
        _gpu_barrier()

        loop_step = fx.Int32(BLOCK_N)
        of0 = Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)
        of1 = Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)
        of2 = Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)
        of3 = Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)
        lf = c_zero_f
        lrf = lr_init

        if is_g0:

            def g0_iter0():
                k_buf_off = fx.Index(LDS_K_OFF)

                v0_off = fx.Index(LDS_V_OFF)
                sA, vwp = do_qk(k_buf_off, preloaded_kw=kvw, v_off=v0_off)
                m_new, corr_new, p_new, prowsum_new, bc_new = do_softmax(sA, m_init)
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
                    lr_init,
                    p_new,
                    prowsum_new,
                    corr_new,
                    *(bc_new if const_expr(USE_BIAS_FOLD) else ()),
                    # Carries hold the LDS byte offsets for the *first loop
                    # body*, which runs i=1 (iter0 above handled i=0). Rather
                    # than deriving them from iv -- i % NUM_BUF_V with
                    # NUM_BUF_V=3 lowers to a 64-bit magic-multiply divide --
                    # the loop permutes them at the yield, which costs nothing
                    # beyond a register rotation.
                    #   (v_prev, v_cur, v_next) = buffers (0, 1, 2)
                    #   (k_cur, k_other)        = buffers (1, 0)
                    _v_slot(0),
                    _v_slot(1),
                    _v_slot(2),
                    _k_slot(1),
                    _k_slot(0),
                    # voffs for the DMA the first loop body will fire (tile 2).
                    *dma_v_voffs(fx.Index(2 * BLOCK_N)),
                    *vwp[:PREFETCH_DEPTH],
                )

            def g0_epilogue(*carry):
                (m_r, l_a, oo0, oo1, oo2, oo3, lr_a, p_c, prowsum_c, corr_c) = carry[
                    :10
                ]
                rest = carry[10 + N_BIAS_CHUNKS :]
                v_prev_off_e = rest[0]
                k_buf_off_e = rest[4]
                vwp = rest[5 + DMA_PASSES :]
                # After the final yield the carries describe the iteration that
                # never ran, so the last tile's K buffer is the second slot.
                o, l2, lr2, _ = apply_pv(
                    [oo0, oo1, oo2, oo3],
                    l_a,
                    lr_a,
                    p_c,
                    prowsum_c,
                    corr_c,
                    v_prev_off_e,
                    list(vwp),
                    k_buf_off_e,
                )
                return o, l2, lr2

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
                m_r, l_a, oo0, oo1, oo2, oo3, lr_a = _g0_args[:7]
                p_c, prowsum_c, corr_c = _g0_args[7:10]
                bc_c = _g0_args[10 : 10 + N_BIAS_CHUNKS]
                _rest = _g0_args[10 + N_BIAS_CHUNKS :]
                v_prev_off, v_cur_off, v_next_off = _rest[0:3]
                k_buf_off, k_next_off = _rest[3:5]
                dv_voffs = _rest[5 : 5 + DMA_PASSES]
                vwp = _rest[5 + DMA_PASSES :]
                kv_start = fx.Index(iv)
                next_kv = kv_start + fx.Index(BLOCK_N)
                oA, lA, lrA, kw_prime = apply_pv(
                    [oo0, oo1, oo2, oo3],
                    l_a,
                    lr_a,
                    p_c,
                    prowsum_c,
                    corr_c,
                    v_prev_off,
                    preloaded_vw=vwp,
                    k_buf_off=k_buf_off,
                    prio=True,
                )
                sA, vwp_new = do_qk(
                    k_buf_off,
                    preloaded_kw=kw_prime,
                    v_off=v_cur_off,
                    seed=(_bias_prefill(bc_c) if const_expr(USE_BIAS_FOLD) else None),
                    dma=(v_rsrc, dma_v_lds(v_next_off), dv_voffs),
                )
                m_new, corr_new, p_new, prowsum_new, bc_new = do_softmax(
                    sA,
                    m_r,
                    first_tile=False,
                    seeded=const_expr(USE_BIAS_FOLD),
                    barrier=True,
                )
                # Address math for the *next* tile's DMA is computed here, in
                # the VALU phase; do_qk only fires the buffer_loads.
                dv_voffs_new = dma_v_voffs(next_kv + fx.Index(BLOCK_N))
                rocdl.sched_barrier(0)
                scf.YieldOp(
                    [
                        _raw(m_new),
                        _raw(lA),
                        _raw(oA[0]),
                        _raw(oA[1]),
                        _raw(oA[2]),
                        _raw(oA[3]),
                        _raw(lrA),
                        _raw(p_new),
                        _raw(prowsum_new),
                        _raw(corr_new),
                        *[_raw(c) for c in (bc_new or ())],
                        # Rotate the V triple and swap the K pair.
                        _raw(v_cur_off),
                        _raw(v_next_off),
                        _raw(v_prev_off),
                        _raw(k_next_off),
                        _raw(k_buf_off),
                        *[_raw(w) for w in dv_voffs_new],
                        *[_raw(w) for w in vwp_new[:PREFETCH_DEPTH]],
                    ]
                )

            _g0_res = [as_dsl_value(r, ex) for r, ex in zip(for_op.results, g0_carry)]
            o_fin, l_fin_g, lr_fin_g = g0_epilogue(*_g0_res)
            of0 = o_fin[0]
            of1 = o_fin[1]
            of2 = o_fin[2]
            of3 = o_fin[3]
            lf = l_fin_g
            lrf = lr_fin_g
        else:

            def g1_iter0():
                k_buf_off = fx.Index(LDS_K_OFF)

                v0_off = fx.Index(LDS_V_OFF)
                sB, vwp = do_qk(k_buf_off, preloaded_kw=kvw, v_off=v0_off)
                m_frozen = (
                    _rowmax(sB, m_init) if const_expr(USE_FROZEN_MAX) else m_init
                )
                if const_expr(USE_BIAS_FOLD):
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
                    lr_init,
                    sB[0],
                    sB[1],
                    sB[2],
                    sB[3],
                    # Offsets for the first loop body (i=1); see g0_iter0.
                    _v_slot(0),
                    _v_slot(1),
                    _v_slot(2),
                    _k_slot(1),
                    _k_slot(0),
                    *vwp[:PREFETCH_DEPTH],
                )

            def g1_epilogue(*carry):
                m_e, l_e, oe0, oe1, oe2, oe3, lr_e = carry[:7]
                sce0, sce1, sce2, sce3 = carry[7:11]
                rest = carry[11:]
                v_prev_off_e = rest[0]
                k_buf_off_e = rest[4]
                vwp = rest[5:]
                _m_e, corrf, pf, prowsumf, _bc = do_softmax(
                    [sce0, sce1, sce2, sce3],
                    m_e,
                    first_tile=False,
                    seeded=const_expr(USE_BIAS_FOLD),
                )
                o, l2, lr2, _ = apply_pv(
                    [oe0, oe1, oe2, oe3],
                    l_e,
                    lr_e,
                    pf,
                    prowsumf,
                    corrf,
                    v_prev_off_e,
                    list(vwp),
                    k_buf_off_e,
                )
                return o, l2, lr2

            g1_carry = list(g1_iter0())
            for_op = scf.ForOp(
                _raw(loop_step),
                _raw(seq_len),
                _raw(loop_step),
                [_raw(v) for v in g1_carry],
            )
            with ir.InsertionPoint(for_op.body):
                iv = for_op.induction_variable
                _iglp()
                _g1_args = [
                    as_dsl_value(a, ex)
                    for a, ex in zip(for_op.inner_iter_args, g1_carry)
                ]
                m_r, l_a, oo0, oo1, oo2, oo3, lr_a = _g1_args[:7]
                ss0, ss1, ss2, ss3 = _g1_args[7:11]
                _rest = _g1_args[11:]
                v_prev_off, v_cur_off, v_next_off = _rest[0:3]
                k_buf_off, k_next_off = _rest[3:5]
                vwp = _rest[5:]
                kv_start = fx.Index(iv)
                next_kv = kv_start + fx.Index(BLOCK_N)
                m_sm, corr_sm, p_sm, prowsum_sm, bc_sm = do_softmax(
                    [ss0, ss1, ss2, ss3],
                    m_r,
                    first_tile=False,
                    seeded=const_expr(USE_BIAS_FOLD),
                    barrier=True,
                )
                # The K DMA's address math belongs to this (VALU) phase; the
                # buffer_loads themselves are interleaved into do_qk below.
                dk_voffs = dma_k_voffs(next_kv)
                oB, lB, lrB, kw_prime = apply_pv(
                    [oo0, oo1, oo2, oo3],
                    l_a,
                    lr_a,
                    p_sm,
                    prowsum_sm,
                    corr_sm,
                    v_prev_off,
                    preloaded_vw=vwp,
                    k_buf_off=k_buf_off,
                    prio=True,
                )
                sB, vwp_new = do_qk(
                    k_buf_off,
                    preloaded_kw=kw_prime,
                    v_off=v_cur_off,
                    seed=(_bias_prefill(bc_sm) if const_expr(USE_BIAS_FOLD) else None),
                    dma=(k_rsrc, dma_k_lds(k_next_off), dk_voffs),
                )
                rocdl.sched_barrier(0)
                scf.YieldOp(
                    [
                        _raw(m_sm),
                        _raw(lB),
                        _raw(oB[0]),
                        _raw(oB[1]),
                        _raw(oB[2]),
                        _raw(oB[3]),
                        _raw(lrB),
                        _raw(sB[0]),
                        _raw(sB[1]),
                        _raw(sB[2]),
                        _raw(sB[3]),
                        # Rotate the V triple and swap the K pair.
                        _raw(v_cur_off),
                        _raw(v_next_off),
                        _raw(v_prev_off),
                        _raw(k_next_off),
                        _raw(k_buf_off),
                        *[_raw(w) for w in vwp_new[:PREFETCH_DEPTH]],
                    ]
                )

            _g1_res = [as_dsl_value(r, ex) for r, ex in zip(for_op.results, g1_carry)]
            o_fin, l_fin_g, lr_fin_g = g1_epilogue(*_g1_res)
            of0 = o_fin[0]
            of1 = o_fin[1]
            of2 = o_fin[2]
            of3 = o_fin[3]
            lf = l_fin_g
            lrf = lr_fin_g

        o_finals = [of0, of1, of2, of3]
        # asm: output_complete_results folds the MFMA half of L (_v_LR) into
        # the VALU half (_v_L) right before the reciprocal.
        l_final = _fadd(lf, Vec(lrf)[0]) if const_expr(L_MFMA_SLABS) else lf

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
        O: fx.Tensor,
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

    def _compile(
        Q,
        K,
        V,
        O,
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
