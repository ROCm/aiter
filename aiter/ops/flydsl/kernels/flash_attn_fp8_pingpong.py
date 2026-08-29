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

from aiter.ops.flydsl.rocdl_mfma_fp8 import (
    C16_F32_PER_LANE,
    ONES_ROW_LANES,
    Mfma16x16x128,
    Mfma32x32x64,
)

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

    L_SPLIT_TILES = 0
    L_MFMA_SLABS = PV_K_STEPS - L_SPLIT_TILES // 2

    SCHRAUDOLPH_2P23 = 1 << 23
    SCHRAUDOLPH_C = 127 * SCHRAUDOLPH_2P23 - 486411
    E8M0_ONE = 0x7F7F7F7F
    SCHRAUDOLPH_CAP = 1138753536.0

    assert L_SPLIT_TILES in (0, 2, 4), "the ones-column MFMA contracts two blocks"
    EXP2_SHIFT = 0.0

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
                # The first convert has opsel=False, so it writes *both* halves
                # of the result and the `old` operand is dead on arrival.
                # Passing a literal 0 makes the backend materialise a
                # `v_mov_b32 vX, 0` into a fresh VGPR ahead of every convert --
                # 16 dead VALU per softmax phase, ~7% of that phase. undef
                # lets the register allocator pick whatever is already live.
                w0 = rocdl.cvt_pk_fp8_f32(
                    T.i32, _raw(f0), _raw(f1), llvm.mlir_undef(T.i32), False
                )
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
        # The L denominator is a pure ones-column reduction, so it does not
        # need the 32x32 atom's output tile -- only one column of it.  The
        # 16x16x128 atom does the same reduction with a vec<4xf32>
        # accumulator instead of vec<16xf32>, which is what makes two
        # independent L accumulators affordable (8 VGPRs, not 32).
        l_mfma_atom = Mfma16x16x128()

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
        # Wave-uniform copy of the wave index, pulled into an SGPR once.  Every
        # LDS-DMA destination is derived from this instead of from `tid`, so the
        # address math lands on the SALU and `_dma_lds_ptr` needs no per-use
        # v_readfirstlane_b32.
        wave_id_s = fx.Index(
            arith.index_cast(
                T.index, rocdl.readfirstlane(T.i32, arith.index_cast(T.i32, _raw(wave_id)))
            )
        )
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

        # Byte extent of this (batch, head) slice measured from the resource
        # base.  Handing it to the descriptor lets the hardware do the KV
        # bounds check for free, so the per-tile voffsets need no software
        # clamp -- see get_hbm_koffs.
        _rsrc_num_records = (seq_len_v - fx.Index(1)) * fx.Index(
            STRIDE_TOKEN
        ) + fx.Index(HEAD_DIM)

        def _rsrc(ptr):
            base_i64 = llvm.PtrToIntOp(T.i64, ptr).result
            off_i64 = arith.index_cast(T.i64, _raw(head_base_elem))
            addr_i64 = arith.addi(base_i64, off_i64)
            return fx.buffer_ops.create_buffer_resource_from_addr(
                addr_i64, num_records_bytes=_raw(_rsrc_num_records)
            )

        k_rsrc = _rsrc(k_ptr)
        v_rsrc = _rsrc(v_ptr)

        _dma_size = arith.constant(DMA_BYTES, type=T.i32)
        _dma_zero = arith.constant(0, type=T.i32)
        _dma_aux = arith.constant(1, type=T.i32)

        _lds_ptr_ty = ir.Type.parse("!llvm.ptr<3>")

        def _dma_lds_ptr(lds_byte_off):
            """SALU-only: the LDS destination is wave-uniform.

            Every input is either a constant or derived from ``wave_id_s``,
            which was pulled into an SGPR once at kernel entry, so the whole
            expression is uniform and the backend keeps it on the SALU.  Do
            *not* readfirstlane here: that would force the address into a VGPR
            first, costing a v_add_u32 + v_readfirstlane_b32 per DMA pass.
            """
            lds_addr = arith.index_cast(T.i64, lds_byte_off)
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

        _dma_k_inv = []
        for p in range_constexpr(DMA_PASSES):
            c = fx.Index(p * DMA_LANES) + tid
            kv = c // fx.Index(8)
            d = (c % fx.Index(8)) * fx.Index(16)
            lds_perm = (fx.Index(p * (NUM_WAVES // 2)) + wave_id_s) * fx.Index(
                K_UNIT_STRIDE
            )
            _dma_k_inv.append((kv, d, lds_perm))

        _ltid = tid - fx.Index(DMA_LANES)
        _lwave = wave_id_s - fx.Index(NUM_WAVES // 2)
        _half = _lwave % fx.Index(2)
        _dma_v_inv = []
        for p in range_constexpr(DMA_PASSES):
            c = fx.Index(p * DMA_LANES) + _ltid
            # d_block is wave-uniform by construction (a wave covers 64
            # consecutive lanes and BLOCK_N is a multiple of 64), but written
            # as c // BLOCK_N the backend cannot see that and puts it in a
            # VGPR.  Rederive it from the uniform wave index instead so the
            # LDS destination stays on the SALU.
            d_block = fx.Index(p * (DMA_LANES // BLOCK_N)) + _lwave // fx.Index(
                BLOCK_N // WARP_SIZE
            )
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

        # One KV tile step, in elements of the voffset. Because the per-lane
        # part of a voffset is loop-invariant, advancing a tile is a single
        # add of this constant -- the FlyDSL analogue of the asm's
        # emit_next_K_mem_addr (persistent VGPR base + scalar delta).
        KV_TILE_DELTA = BLOCK_N * STRIDE_TOKEN
        _c_kv_delta = arith.constant(KV_TILE_DELTA, type=T.i32)

        def advance_hbm_offs(offs):
            """Bump a carried set of voffsets by one KV tile: 1 v_add_u32 each."""
            return [arith.addi(o, _c_kv_delta) for o in offs]

        def get_hbm_koffs(kv_start):
            """Per-pass voffsets for the K tile DMA.

            Only used to *seed* the carry (prologue / iteration 0, where
            kv_start is a constant); inside the loop the offsets are advanced
            with advance_hbm_offs instead of being rebuilt.

            There is no software bounds clamp: the buffer resource carries a
            num_records covering exactly this (batch, head) slice, so a tile
            prefetched past the end of the sequence is dropped by the hardware.
            """
            out = []
            for p in range_constexpr(DMA_PASSES):
                kv, d, _ = _dma_k_inv[p]
                out.append(
                    arith.index_cast(
                        T.i32, _raw((kv_start + kv) * fx.Index(STRIDE_TOKEN) + d)
                    )
                )
            return out

        def get_lds_koffs(k_off):
            base_lds = fx.Index(lds_offset) + k_off
            return [
                _dma_lds_ptr(base_lds + _dma_k_inv[p][2])
                for p in range_constexpr(DMA_PASSES)
            ]

        def dma_k(buf, kv_start):
            g_koffs = get_hbm_koffs(kv_start)
            l_koffs = get_lds_koffs(_k_slot(buf))
            for p, ptr in enumerate(l_koffs):
                _dma_fire(k_rsrc, ptr, g_koffs[p])

        def get_hbm_voffs(kv_start):
            """Per-pass voffsets for the V tile DMA; see get_hbm_koffs."""
            out = []
            for p in range_constexpr(DMA_PASSES):
                kv, d_voff, _ = _dma_v_inv[p]
                out.append(
                    arith.index_cast(
                        T.i32, _raw((kv_start + kv) * fx.Index(STRIDE_TOKEN) + d_voff)
                    )
                )
            return out

        def get_lds_voffs(v_off):
            base_lds = fx.Index(lds_offset) + v_off
            return [
                _dma_lds_ptr(base_lds + _dma_v_inv[p][2])
                for p in range_constexpr(DMA_PASSES)
            ]

        def dma_v(buf, kv_start):
            g_voffs = get_hbm_voffs(kv_start)
            l_voffs = get_lds_voffs(_v_slot(buf))
            for p, ptr in enumerate(l_voffs):
                _dma_fire(v_rsrc, ptr, g_voffs[p])

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

        # Lane-varying but loop-invariant part of the V read address.  Folding
        # the slot offset in used to happen inside the MFMA phase, where the
        # resulting v_add_u32 sat on the critical path: every ds_read_b64_tr_b8
        # of the phase depends on it, so the first V read issued ~250-370 cycles
        # after the barrier instead of the ~48 the asm kernel manages.  Hoisting
        # this out and carrying the finished per-slot addresses turns the
        # rotation into a register permute.
        v_lane_base = (
            fx.Index(lds_offset)
            + (lo // fx.Index(16)) * fx.Index(V_DBLOCK_STRIDE)
            + hi * fx.Index(16 * V_KV_STRIDE)
            + lo_in_grp * fx.Index(8)
        )

        def _v_addr(v_off):
            """Per-lane V base address for an LDS slot byte offset."""
            return v_lane_base + v_off

        def read_v_pack(v_addr, dt, ks):
            base = fx.buffer_ops.create_llvm_ptr(
                fx.Int64(v_addr), address_space=3
            )
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
            l_mfma,
            p_pack,
            v_addr,
            v_preloaded,
            k_base,
        ):
            p_ks_list = [
                Vec(p_pack).shuffle(Vec(p_pack), list(range(r * 8, r * 8 + 8)))
                for r in range_constexpr(PV_K_STEPS)
            ]
            # The tail of the L denominator from MFMA.  One accumulator per
            # slab rather than one chained accumulator: a single register
            # would serialise the slabs on the MFMA's C operand, and the
            # trace showed the second issuing 72 cycles after the first for
            # exactly that reason.  Independent C registers let both go
            # back-to-back at the pipe's issue rate; the partial sums are
            # folded together once, in the epilogue.
            # 16x16x128 rather than 32x32x64: the reduction only ever reads
            # one output column, so the wider atom's extra 12 accumulator
            # registers per slab were pure cost.
            l_mfma = [
                l_mfma_atom.call(ones_pack, p_ks_list[ks], l_mfma[i])
                for i, ks in enumerate(
                    range_constexpr(PV_K_STEPS - L_MFMA_SLABS, PV_K_STEPS)
                )
            ]
            PV_UNITS = D_TILES * PV_K_STEPS
            kv_windows = [None] * 4
            k_preloaded = [None] * PREFETCH_DEPTH
            for u in range_constexpr(PREFETCH_DEPTH):
                kv_windows[u] = v_preloaded[u]
            for u in range_constexpr(PV_UNITS):
                dt = u // PV_K_STEPS
                ks = u % PV_K_STEPS
                _sched_barrier()
                # if const_expr(u < 7):
                #     _wait_lgkmcnt(4)
                # else:
                #     _wait_lgkmcnt(2)
                o_accs[dt] = mfma.call(kv_windows[u % 4], p_ks_list[ks], o_accs[dt])
                un = u + PREFETCH_DEPTH
                if const_expr(un < PV_UNITS):
                    kv_windows[un % 4] = read_v_pack(
                        v_addr, un // PV_K_STEPS, un % PV_K_STEPS
                    )
                else:
                    ki = u - (PV_UNITS - PREFETCH_DEPTH)
                    kv_windows[un % 4] = _load_k_unit(
                        None, ki // K_STEPS, ki % K_STEPS, base=k_base
                    )
                    k_preloaded[ki] = kv_windows[un % 4]
                _sched_barrier()
            return o_accs, l_mfma, k_preloaded

        QK_UNITS = N_KV_TILES * K_STEPS

        # Two units deep, like the asm (k_lds_unit_idx starts at 2 in gemm_QK
        # and v_lds_unit_idx at 2 in gemm_PV): the MFMA at index u consumes
        # operands read two MFMAs ago, so one full ds_read latency is hidden.
        PREFETCH_DEPTH = 2

        K_NT_STRIDE = 32 * K_STRIDE + (32 // K_UNIT_ROWS) * PAD_K

        # Same hoist as the V side: everything but the slot offset is
        # loop-invariant, so the per-slot row address is built once outside the
        # loop and carried, not re-added inside the MFMA phase.
        k_lane_base = (
            lo * K_STRIDE
            + (lo // fx.Index(K_UNIT_ROWS)) * fx.Index(PAD_K)
            + hi * 32
        )

        def _k_base_row(k_buff_off):
            return k_lane_base + k_buff_off

        def _load_k_unit(k_buff_off, nt, ks, base=None):
            if const_expr(base is None):
                base = _k_base_row(k_buff_off)
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

        def do_qk(
            k_base,
            k_preloaded,
            v_addr,
            seed=None,
            dma=None
        ):
            kv_windows = [None] * 4
            v_preloaded = [None] * PREFETCH_DEPTH
            for u in range_constexpr(PREFETCH_DEPTH):
                kv_windows[u] = k_preloaded[u]
            s_accs = [
                (mfma.zero_value if const_expr(seed is None) else seed)
                for _ in range_constexpr(N_KV_TILES)
            ]
            dma_rsrc, l_ptrs, g_ptrs = (
                (None, [], []) if const_expr(dma is None) else dma
            )
            for u in range_constexpr(QK_UNITS):
                nt = u // K_STEPS
                ks = u % K_STEPS
                _sched_barrier()
                # if const_expr(u < 7):
                #     _wait_lgkmcnt(2)
                # elif const_expr(u == 7):
                #     _wait_lgkmcnt(4)

                s_accs[nt] = qk_mfma(kv_windows[u % 4], q_packs[ks], s_accs[nt])
                un = u + PREFETCH_DEPTH
                if const_expr(un < QK_UNITS):
                    kv_windows[un % 4] = _load_k_unit(
                        None, un // K_STEPS, un % K_STEPS, base=k_base
                    )
                else:
                    vi = u - (QK_UNITS - PREFETCH_DEPTH)
                    kv_windows[un % 4] = read_v_pack(
                        v_addr, vi // PV_K_STEPS, vi % PV_K_STEPS
                    )
                    v_preloaded[vi] = kv_windows[un % 4]
                _sched_barrier()
            for _dp in range_constexpr(len(l_ptrs)):
                _dma_fire(dma_rsrc, l_ptrs[_dp], g_ptrs[_dp])
            return s_accs, v_preloaded

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

        def _bias_split(m_term):
            """Three-chunk bf16 decomposition of the Schraudolph seed."""
            c0 = rocdl.cvt_pk_bf16_f32(_raw(m_term), _raw(m_term))
            r1 = _fsub(m_term, _f32_of_bf16_hi(c0))
            c1 = rocdl.cvt_pk_bf16_f32(_raw(r1), _raw(r1))
            r2 = _fsub(r1, _f32_of_bf16_hi(c1))
            return (
                rocdl.cvt_pk_bf16_f32(_raw(m_term), _raw(r1)),
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

        def do_softmax_prepare(
            s_accs,
            m_init,
        ):
            m_frozen = _rowmax(s_accs, m_init)
            m_term = _fadd(_fsub(c_zero_f, m_frozen), c_exp2_bias)
            bias_chunks = _bias_split(m_term)

            mt_vec = Vec.from_elements([m_term], fx.Float32).broadcast_to(
                C_F32_PER_LANE
            )
            for nt in range_constexpr(N_KV_TILES):
                s_accs[nt] = _fadd(Vec(s_accs[nt]), mt_vec)

            return m_frozen, bias_chunks, s_accs

        # ---- overflow watchdog + rollback (see tmp/watchdog_rollback_report.md)
        #
        # The frozen max is a bet: m is decided on tile 0 and never moves, so a
        # later tile whose scores climb past it produces P > 1 and can walk off
        # the top of E4M3 (448).  The watchdog is the hot-path half of the bet:
        # an in-lane max over this lane's 64 Schraudolph patterns plus one
        # ballot.  No cross-lane exchange -- the wave-wide OR already unions
        # every lane, so the exchange only matters once we are inside the
        # repair.  The compare is written NOT(CAP >= m) rather than m > CAP so
        # a NaN pattern also fires.
        #
        # The rollback is the cold half, and it is exact rather than a clamp.
        # Everything lives in Schraudolph pattern units, where the exponent is
        # affine: lowering every pattern of this tile by d is *identical* to
        # having frozen a max that was larger by d, so P needs no float rescale
        # and the repair folds into the scalar bias.  The already-accumulated
        # O / L / LR were built against the old max and are rescaled by
        # f = 2^(-d/2^23) -- uniform in a lane, so the O/L ratio is untouched
        # and the result is bit-exactly what a correctly-seeded run would give.
        # d is chosen to put the row max back at 2^EXP2_SHIFT, i.e. all the
        # headroom is restored, so the repair is self-healing and cannot
        # thrash.
        def _rollback(
            s_accs, o_accs, l_mfma, l_valu, m_frozen, bias_chunks, m_local, fired
        ):
            vt = Vec.make_type(C_F32_PER_LANE, fx.Float32)
            lvt = Vec.make_type(C16_F32_PER_LANE, fx.Float32)
            if_op = scf.IfOp(
                fired,
                results_=[vt] * N_KV_TILES
                + [vt] * D_TILES
                + [lvt] * L_MFMA_SLABS
                + [T.f32, T.f32, T.i32, T.i32],
                has_else=True,
            )
            with ir.InsertionPoint(if_op.then_block):
                # Complete the row: a row of S is one lane's 16 values in each
                # of the 4 tiles, unioned with its lane^32 peer.
                t = _fmax(
                    m_local,
                    fx.Float32(m_local).shuffle_xor(
                        fx.Int32(32), fx.Int32(WARP_SIZE)
                    ),
                )
                # Only rows that actually overflowed move; the ballot is
                # wave-wide, so most lanes inside the branch are innocent.
                hit = arith.cmpf(
                    arith.CmpFPredicate.OGT, _raw(t), _raw(c_sch_cap)
                )
                d = ArithValue(hit).select(
                    fx.Float32(_fsub(t, c_exp2_bias)), c_zero_f
                )
                f = rocdl.exp2(
                    T.f32, _raw(_fmul(_fsub(c_zero_f, d), c_inv_2p23))
                )
                d_vec = Vec.from_elements([d], fx.Float32).broadcast_to(
                    C_F32_PER_LANE
                )
                f_vec = Vec.from_elements([f], fx.Float32).broadcast_to(
                    C_F32_PER_LANE
                )
                _s = [
                    _fsub(Vec(s_accs[nt]), d_vec)
                    for nt in range_constexpr(N_KV_TILES)
                ]
                _o = [
                    _fmul(Vec(o_accs[dt]), f_vec)
                    for dt in range_constexpr(D_TILES)
                ]
                # The L accumulators are only vec<4xf32>, so they need their
                # own broadcast of the rescale factor.
                f_vec_l = Vec.from_elements([f], fx.Float32).broadcast_to(
                    C16_F32_PER_LANE
                )
                _lm = [_fmul(Vec(lm), f_vec_l) for lm in l_mfma]
                _lv = _fmul(l_valu, f)
                _m = _fadd(m_frozen, d)
                # Re-derive the seed: b = C - m.  Later tiles are prefilled
                # with the new b, so they agree with the rescaled past.
                _bc = _bias_split(_fsub(c_exp2_bias, _m))
                scf.YieldOp(
                    [_raw(v) for v in _s]
                    + [_raw(v) for v in _o]
                    + [_raw(v) for v in _lm]
                    + [_raw(_lv), _raw(_m)]
                    + [_raw(bc) for bc in _bc]
                )
            with ir.InsertionPoint(if_op.else_block):
                scf.YieldOp(
                    [_raw(v) for v in s_accs]
                    + [_raw(v) for v in o_accs]
                    + [_raw(v) for v in l_mfma]
                    + [_raw(l_valu), _raw(m_frozen)]
                    + [_raw(bc) for bc in bias_chunks]
                )
            r = if_op.results
            n = N_KV_TILES
            s_out = [
                as_dsl_value(r[i], s_accs[i]) for i in range_constexpr(n)
            ]
            o_out = [
                as_dsl_value(r[n + i], o_accs[i]) for i in range_constexpr(D_TILES)
            ]
            k = n + D_TILES
            lm_out = [
                as_dsl_value(r[k + i], l_mfma[i])
                for i in range_constexpr(L_MFMA_SLABS)
            ]
            k += L_MFMA_SLABS
            return (
                s_out,
                o_out,
                lm_out,
                as_dsl_value(r[k], l_valu),
                as_dsl_value(r[k + 1], m_frozen),
                tuple(
                    as_dsl_value(r[k + 2 + i], bias_chunks[i])
                    for i in range_constexpr(N_BIAS_CHUNKS)
                ),
            )

        def do_softmax(
            s_accs,
            m_frozen,
            l_valu,
            bias_chunks=None,
            seeded=False,
            o_accs=None,
            l_mfma=None,
        ):
            if const_expr(not seeded):
                m_frozen = _rowmax(s_accs, m_frozen)
                m_term = _fadd(_fsub(c_zero_f, m_frozen), c_exp2_bias)
                mt_vec = Vec.from_elements([m_term], fx.Float32).broadcast_to(
                    C_F32_PER_LANE
                )
                bias_chunks = _bias_split(m_term)
            else:
                # Watchdog: in-lane max of this lane's patterns, then one
                # ballot.  Cheap enough to sit on the critical path; the
                # branch is wave-uniform so the wave never diverges.
                m_local = _lane_max(s_accs)
                fired = _wave_or(
                    arith.cmpf(
                        arith.CmpFPredicate.UGT, _raw(m_local), _raw(c_sch_cap)
                    )
                )
                (
                    s_accs,
                    o_accs,
                    l_mfma,
                    l_valu,
                    m_frozen,
                    bias_chunks,
                ) = _rollback(
                    s_accs,
                    o_accs,
                    l_mfma,
                    l_valu,
                    m_frozen,
                    bias_chunks,
                    m_local,
                    fired,
                )

            n_groups = C_F32_PER_LANE // 4

            p_words = []
            v_partial = None
            zero_vec = Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)
            i32_vec_ty = Vec.make_type(C_F32_PER_LANE, fx.Int32)
            f32_vec_ty = Vec.make_type(C_F32_PER_LANE, fx.Float32)
            for nt in range_constexpr(N_KV_TILES):
                if const_expr(seeded):
                    # No clamp: the watchdog + rollback above subsume both
                    # ends of what the old med3 to [0, CAP] was doing.
                    #
                    # Ceiling: t is the max over *every* pattern this lane
                    # owns, so after the rollback each lane either had t <=
                    # CAP already (d = 0) or has been lowered by exactly
                    # t - C, which is bit-exact and lands its max on C.  The
                    # 0x7F800000 the bitcast would have to reach is 1.9x CAP,
                    # so the ceiling is unreachable, not merely unlikely.
                    #
                    # Floor: the saturating v_cvt_u32_f32 below already maps
                    # negatives and NaN to 0, which *is* the softmax mask.
                    #
                    # This is not belt-and-braces -- disabling detection here
                    # returns NaN at S=65536, so the rollback is load-bearing;
                    # it is the clamp that had become redundant.  Worth ~280
                    # TFLOPS (2436 -> 2714): 64 med3 per softmax phase, on the
                    # critical path of a VALU-only phase.
                    biased = Vec(s_accs[nt])
                else:
                    aff = _fadd(Vec(s_accs[nt]), mt_vec)
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
                            rscale=None,
                        )
                    )
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
            if const_expr(v_partial is not None):
                v_sum = _lane_sum(v_partial)
                v_sum = _fadd(
                    v_sum,
                    fx.Float32(v_sum).shuffle_xor(fx.Int32(32), fx.Int32(WARP_SIZE)),
                )
                l_valu = _fadd(l_valu, v_sum)
            # o_accs / l_mfma come back because the rollback may have rescaled
            # them; on the non-seeded path they are whatever was passed in.
            return m_frozen, bias_chunks, p_pack, l_valu, o_accs, l_mfma

        # Ones column for the L reduction.  0x38 is fp8 e4m3 1.0, so
        # 0x38383838 is four of them.  The 16x16x128 atom folds lanes
        # {n, n+16, n+32, n+48} into one output column, but a query row here
        # is lane % 32 -- so a uniform ones operand would add row n+16 into
        # row n.  Restricting the ones to lanes where `lo` is one of
        # ONES_ROW_LANES gives each 16-lane group its own output row and
        # makes C[lane][0] agree, bit for bit, with what the 32x32 atom
        # produced (checked against it on random inputs).  Same mask the
        # assembly kernel builds.
        _ones_lane_pred = None
        for _al in ONES_ROW_LANES:
            _p = arith.cmpi(
                arith.CmpIPredicate.eq, _raw(lo), _raw(fx.Index(_al))
            )
            _ones_lane_pred = _p if _ones_lane_pred is None else arith.ori(
                _ones_lane_pred, _p
            )
        ones_scalar = ArithValue(_ones_lane_pred).select(
            fx.Int32(0x38383838), fx.Int32(0)
        )
        ones_pack = Vec.from_elements(
            [ones_scalar] * (A_FP8_PER_LANE // 4), fx.Int32
        )

        o_init = [
            Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)
            for _ in range_constexpr(D_TILES)
        ]
        N_BIAS_CHUNKS = 2

        is_g0 = wave_id < fx.Index(NUM_WAVES // 2)

        k_preloaded_prologue = [None] * PREFETCH_DEPTH
        for u in range_constexpr(PREFETCH_DEPTH):
            k_preloaded_prologue[u] = _load_k_unit_global(fx.Index(0), u // K_STEPS, u % K_STEPS)

        # Tiles 0 and 1 are prefetched into buffers 0 and 1; from there on the
        # loops rotate their carried LDS byte offsets.
        if is_g0:
            dma_k(fx.Index(0), fx.Index(0))
        else:
            dma_v(fx.Index(0), fx.Index(0))
            dma_v(fx.Index(1), fx.Index(BLOCK_N))

        _wait_vmcnt()
        _wait_lgkmcnt()
        _gpu_barrier()

        loop_step = fx.Int32(BLOCK_N)
        l_valu_init = c_zero_f
        # One C register per L slab: the slabs are summed independently and
        # only merged in the epilogue, so nothing serialises them.
        l_mfma_init = [l_mfma_atom.zero_value for _ in range_constexpr(L_MFMA_SLABS)]
        # Carry layout.  Both groups start (m_frozen, l_valu, *l_mfma, o0..o3);
        # g0 then carries p_pack, g1 carries s_accs[0..3].
        G0_HEAD = 2 + L_MFMA_SLABS + 4 + 1
        G1_HEAD = 2 + L_MFMA_SLABS + 4
        G1_SACC = G1_HEAD + 4
        m_init = c_neg_inf

        ## for epilogue
        le_valu = c_zero_f
        # Flat names, not a list: `if is_g0` is a stateful dynamic scf.if and
        # every live state variable has to be an MLIR value.  PV_K_STEPS is 2,
        # so there are at most two slabs; an unused one stays zero and adds
        # nothing to l_final.
        le_mfma0 = l_mfma_atom.zero_value
        le_mfma1 = l_mfma_atom.zero_value
        oe0 = Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)
        oe1 = Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)
        oe2 = Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)
        oe3 = Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)

        if is_g0:

            def g0_iter0():
                k_buff_off = _k_slot(0)

                v_buff_off = _v_slot(0)

                ## LDS k offs
                l_koffs = get_lds_koffs(_k_slot(1))
                ## HBM k offs
                g_koffs = get_hbm_koffs(BLOCK_N)
                s_accs, v_preloaded = do_qk(
                    _k_base_row(k_buff_off),
                    k_preloaded_prologue,
                    _v_addr(v_buff_off),
                    dma=(k_rsrc, l_koffs, g_koffs),
                )
                m_frozen, bias_chunks, p_pack, l_valu, _, _ = do_softmax(
                    s_accs, m_init, l_valu_init
                )

                g_koffs = get_hbm_koffs(BLOCK_N * 2)
                return (
                    m_frozen,
                    l_valu,
                    *l_mfma_init,
                    o_init[0],
                    o_init[1],
                    o_init[2],
                    o_init[3],
                    p_pack,
                    *bias_chunks,
                    # Finished per-lane LDS addresses, rotated in lockstep with
                    # the raw slot offsets below.  The reads use these; only the
                    # DMA (wave-uniform SALU) still wants the raw offsets.
                    _v_addr(_v_slot(0)),
                    _v_addr(_v_slot(1)),
                    _v_addr(_v_slot(2)),
                    _k_base_row(_k_slot(1)),
                    _k_base_row(_k_slot(0)),
                    _v_slot(0),
                    _v_slot(1),
                    _v_slot(2),
                    _k_slot(1),
                    _k_slot(0),
                    *g_koffs,
                    *v_preloaded,
                )

            def g0_epilogue(*carry):
                m_frozen, l_valu = carry[:2]
                l_mfma = list(carry[2 : 2 + L_MFMA_SLABS])
                o0, o1, o2, o3, p_pack = carry[2 + L_MFMA_SLABS : G0_HEAD]
                _rest = carry[G0_HEAD + N_BIAS_CHUNKS :]
                v_addr = _rest[0]
                k_base = _rest[3]
                v_preloaded = _rest[10 + DMA_PASSES :]
                # After the final yield the carries describe the iteration that
                # never ran, so the last tile's K buffer is the second slot.
                o, l_mfma, _ = apply_pv(
                    [o0, o1, o2, o3],
                    l_mfma,
                    p_pack,
                    v_addr,
                    v_preloaded,
                    k_base,
                )
                return o, l_mfma, l_valu

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
                m_frozen, l_valu = _g0_args[:2]
                l_mfma = list(_g0_args[2 : 2 + L_MFMA_SLABS])
                o0, o1, o2, o3, p_pack = _g0_args[2 + L_MFMA_SLABS : G0_HEAD]
                bias_chunks = _g0_args[G0_HEAD : G0_HEAD + N_BIAS_CHUNKS]
                _rest = _g0_args[G0_HEAD + N_BIAS_CHUNKS :]
                v_addr, v_next_addr, v_further_addr = _rest[0:3]
                k_base, k_next_base = _rest[3:5]
                v_buff_off, v_next_off, v_further_off = _rest[5:8]
                k_buff_off, k_next_off = _rest[8:10]
                g_koffs = [_raw(x) for x in _rest[10 : 10 + DMA_PASSES]]
                # The DMA of this iteration lands in the buffer the *next* one
                # reads; its voffsets were computed in the previous softmax
                # phase, its LDS pointers are wave-uniform SALU rebuilt here.
                l_koffs = get_lds_koffs(k_next_off)

                v_preloaded = _rest[10 + DMA_PASSES :]

                _sched_barrier()
                rocdl.s_setprio(1)
                _gpu_barrier()
                o, l_mfma, k_preloaded = apply_pv(
                    [o0, o1, o2, o3],
                    l_mfma,
                    p_pack,
                    v_addr,
                    v_preloaded,
                    k_base,
                )
                _wait_vmcnt(0)
                seed = _bias_prefill(bias_chunks)
                s_accs, v_preloaded = do_qk(
                    k_base,
                    k_preloaded,
                    v_next_addr,
                    seed=seed,
                    dma=(k_rsrc, l_koffs, g_koffs),
                )

                _sched_barrier()
                rocdl.s_setprio(0)
                _gpu_barrier()
                # apply_pv has already folded tile i-1 into o / l_mfma, and
                # tile i's P is not built yet, so this is the one point where
                # the rollback can rescale the whole past in one shot.
                m_frozen, bias_chunks, p_pack, l_valu, o, l_mfma = do_softmax(
                    s_accs,
                    m_frozen,
                    l_valu,
                    bias_chunks,
                    seeded=True,
                    o_accs=o,
                    l_mfma=l_mfma,
                )
                # Address math for the *next* tile's DMA is computed here, in
                # the VALU phase; do_qk only fires the buffer_loads.  One add
                # per pass off the carried value -- nothing is rebuilt from
                # kv_start.
                g_koffs = advance_hbm_offs(g_koffs)
                scf.YieldOp(
                    [
                        _raw(m_frozen),
                        _raw(l_valu),
                        *[_raw(lm) for lm in l_mfma],
                        _raw(o[0]),
                        _raw(o[1]),
                        _raw(o[2]),
                        _raw(o[3]),
                        _raw(p_pack),
                        *[_raw(bc) for bc in bias_chunks],
                        # Rotate the V triple and swap the K pair -- addresses
                        # and raw offsets move together.
                        _raw(v_next_addr),
                        _raw(v_further_addr),
                        _raw(v_addr),
                        _raw(k_next_base),
                        _raw(k_base),
                        _raw(v_next_off),
                        _raw(v_further_off),
                        _raw(v_buff_off),
                        _raw(k_next_off),
                        _raw(k_buff_off),
                        *[_raw(gk) for gk in g_koffs],
                        *[_raw(vp) for vp in v_preloaded],
                    ]
                )

            _g0_res = [as_dsl_value(r, ex) for r, ex in zip(for_op.results, g0_carry)]
            oe, le_mfma, le_valu = g0_epilogue(*_g0_res)
            le_mfma0 = le_mfma[0]
            if const_expr(L_MFMA_SLABS > 1):
                le_mfma1 = le_mfma[1]
            oe0 = oe[0]
            oe1 = oe[1]
            oe2 = oe[2]
            oe3 = oe[3]
        else:

            def g1_iter0():
                k_buff_off = _k_slot(0)

                v_buff_off = _v_slot(0)

                ## LDS v offs
                l_voffs = get_lds_voffs(_v_slot(2))
                ## HBM v offs
                g_voffs = get_hbm_voffs(2 * BLOCK_N)
                s_accs, v_preloaded = do_qk(
                    _k_base_row(k_buff_off),
                    k_preloaded_prologue,
                    _v_addr(v_buff_off),
                    dma=(v_rsrc, l_voffs, g_voffs),
                )
                m_frozen, bias_chunks, s_accs = do_softmax_prepare(s_accs, m_init)
                _sched_barrier()
                _gpu_barrier()
                # Seed for the first loop body (iv == BLOCK_N), which loads the
                # tile at iv + 2 * BLOCK_N.
                g_voffs = advance_hbm_offs(g_voffs)
                return (
                    m_frozen,
                    l_valu_init,
                    *l_mfma_init,
                    o_init[0],
                    o_init[1],
                    o_init[2],
                    o_init[3],
                    s_accs[0],
                    s_accs[1],
                    s_accs[2],
                    s_accs[3],
                    *bias_chunks,
                    # Offsets for the first loop body (i=1); see g0_iter0.
                    _v_addr(_v_slot(0)),
                    _v_addr(_v_slot(1)),
                    _v_addr(_v_slot(2)),
                    _k_base_row(_k_slot(1)),
                    _k_base_row(_k_slot(0)),
                    _v_slot(0),
                    _v_slot(1),
                    _v_slot(2),
                    _k_slot(1),
                    _k_slot(0),
                    *g_voffs,
                    *v_preloaded,
                )

            def g1_epilogue(*carry):
                m_frozen, l_valu = carry[:2]
                l_mfma = list(carry[2 : 2 + L_MFMA_SLABS])
                o0, o1, o2, o3 = carry[2 + L_MFMA_SLABS : G1_HEAD]
                s0, s1, s2, s3 = carry[G1_HEAD : G1_HEAD + 4]
                _rest = carry[G1_SACC + N_BIAS_CHUNKS :]
                v_addr = _rest[0]
                k_base = _rest[3]
                v_preloaded = _rest[10 + DMA_PASSES :]
                bias_chunks = carry[G1_SACC : G1_SACC + N_BIAS_CHUNKS]
                m_frozen, _, p_pack, l_valu, o_r, l_mfma = do_softmax(
                    [s0, s1, s2, s3],
                    m_frozen,
                    l_valu,
                    bias_chunks,
                    seeded=True,
                    o_accs=[o0, o1, o2, o3],
                    l_mfma=l_mfma,
                )
                o, l_mfma, _ = apply_pv(
                    o_r,
                    l_mfma,
                    p_pack,
                    v_addr,
                    v_preloaded,
                    k_base,
                )
                return o, l_mfma, l_valu

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
                m_frozen, l_valu = _g1_args[:2]
                l_mfma = list(_g1_args[2 : 2 + L_MFMA_SLABS])
                o0, o1, o2, o3 = _g1_args[2 + L_MFMA_SLABS : G1_HEAD]
                s0, s1, s2, s3 = _g1_args[G1_HEAD : G1_HEAD + 4]
                bias_chunks = _g1_args[G1_SACC : G1_SACC + N_BIAS_CHUNKS]
                _rest = _g1_args[G1_SACC + N_BIAS_CHUNKS :]
                v_addr, v_next_addr, v_further_addr = _rest[0:3]
                k_base, k_next_base = _rest[3:5]
                v_buff_off, v_next_off, v_further_off = _rest[5:8]
                k_buff_off, k_next_off = _rest[8:10]
                g_voffs = [_raw(x) for x in _rest[10 : 10 + DMA_PASSES]]
                v_preloaded = _rest[10 + DMA_PASSES :]
                # o / l_mfma still hold only the tiles apply_pv has already
                # folded in -- i.e. strictly the past relative to the s_accs
                # being softmaxed here -- so the rescale is well-defined.
                (
                    m_frozen,
                    bias_chunks,
                    p_pack,
                    l_valu,
                    (o0, o1, o2, o3),
                    l_mfma,
                ) = do_softmax(
                    [s0, s1, s2, s3],
                    m_frozen,
                    l_valu,
                    bias_chunks,
                    seeded=True,
                    o_accs=[o0, o1, o2, o3],
                    l_mfma=l_mfma,
                )
                # The V DMA's address math belongs to this (VALU) phase; the
                # buffer_loads themselves are interleaved into do_qk below.
                l_voffs = get_lds_voffs(v_buff_off)
                _sched_barrier()
                rocdl.s_setprio(1)
                _gpu_barrier()

                o, l_mfma, k_preloaded = apply_pv(
                    [o0, o1, o2, o3],
                    l_mfma,
                    p_pack,
                    v_addr,
                    v_preloaded,
                    k_base,
                )
                _wait_vmcnt(0)
                seed = _bias_prefill(bias_chunks)
                s_accs, v_preloaded = do_qk(
                    k_base,
                    k_preloaded,
                    v_next_addr,
                    seed=seed,
                    dma=(v_rsrc, l_voffs, g_voffs),
                )
                _sched_barrier()
                rocdl.s_setprio(0)
                _gpu_barrier()
                g_voffs = advance_hbm_offs(g_voffs)
                scf.YieldOp(
                    [
                        _raw(m_frozen),
                        _raw(l_valu),
                        *[_raw(lm) for lm in l_mfma],
                        _raw(o[0]),
                        _raw(o[1]),
                        _raw(o[2]),
                        _raw(o[3]),
                        _raw(s_accs[0]),
                        _raw(s_accs[1]),
                        _raw(s_accs[2]),
                        _raw(s_accs[3]),
                        *[_raw(bc) for bc in bias_chunks],
                        # Rotate the V triple and swap the K pair -- addresses
                        # and raw offsets move together.
                        _raw(v_next_addr),
                        _raw(v_further_addr),
                        _raw(v_addr),
                        _raw(k_next_base),
                        _raw(k_base),
                        _raw(v_next_off),
                        _raw(v_further_off),
                        _raw(v_buff_off),
                        _raw(k_next_off),
                        _raw(k_buff_off),
                        *[_raw(gv) for gv in g_voffs],
                        *[_raw(vp) for vp in v_preloaded],
                    ]
                )

            _g1_res = [as_dsl_value(r, ex) for r, ex in zip(for_op.results, g1_carry)]
            oe, le_mfma, le_valu = g1_epilogue(*_g1_res)
            le_mfma0 = le_mfma[0]
            if const_expr(L_MFMA_SLABS > 1):
                le_mfma1 = le_mfma[1]
            oe0 = oe[0]
            oe1 = oe[1]
            oe2 = oe[2]
            oe3 = oe[3]

        o_finals = [oe0, oe1, oe2, oe3]
        # asm: output_complete_results folds the MFMA half of L (_v_LR) into
        # the VALU half (_v_L) right before the reciprocal.
        l_final = le_valu
        if const_expr(L_MFMA_SLABS > 0):
            l_final = _fadd(l_final, Vec(le_mfma0)[0])
        if const_expr(L_MFMA_SLABS > 1):
            l_final = _fadd(l_final, Vec(le_mfma1)[0])

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
