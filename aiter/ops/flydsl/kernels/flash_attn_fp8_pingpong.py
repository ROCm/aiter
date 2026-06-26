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

        def _f32x4_to_fp8_word(f0, f1, f2, f3):
            # Pack 4 f32 -> one i32 (4 contiguous fp8 bytes [f0,f1,f2,f3]) using
            # two cvt_pk_fp8_f32: low word = (f0,f1), high word = (f2,f3).
            # Halves both the cvt count and the LDS store count vs per-byte.
            w0 = rocdl.cvt_pk_fp8_f32(T.i32, _raw(f0), _raw(f1), fx.Int32(0), False)
            w1 = rocdl.cvt_pk_fp8_f32(T.i32, _raw(f2), _raw(f3), _raw(w0), True)
            return w1

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

        def dma_k(buf, kv_start):
            k_buf_off = fx.Index(LDS_K_OFF) + buf * fx.Index(LDS_K_TILE)
            ltid = tid - fx.Index(DMA_LANES)  # 0..255 within G1
            lwave = wave_id - fx.Index(NUM_WAVES // 2)  # 0..3 within G1
            for p in range_constexpr(DMA_PASSES):
                c = fx.Index(p * DMA_LANES) + ltid
                kv = c // fx.Index(8)
                d = (c % fx.Index(8)) * fx.Index(16)
                kv_abs = kv_start + kv
                in_b = kv_abs < seq_len_v
                kv_safe = fx.Index(ArithValue(in_b).select(kv_abs, fx.Index(0)))
                voff = kv_safe * fx.Index(STRIDE_TOKEN) + d
                lds_byte = (
                    fx.Index(lds_offset)
                    + k_buf_off
                    + (fx.Index(p * (NUM_WAVES // 2)) + lwave) * fx.Index(K_UNIT_STRIDE)
                )
                _dma_issue(k_rsrc, lds_byte, voff)

        def dma_v(buf, kv_start):
            v_buf_off = fx.Index(LDS_V_OFF) + buf * fx.Index(LDS_V_TILE)
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
                kv = (
                    blk * fx.Index(32)
                    + hi_group * fx.Index(4)
                    + grp * fx.Index(8)
                    + fine
                )
                kv_abs = kv_start + kv
                in_b = kv_abs < seq_len_v
                kv_safe = fx.Index(ArithValue(in_b).select(kv_abs, fx.Index(0)))
                voff = kv_safe * fx.Index(STRIDE_TOKEN) + d_block * fx.Index(16)
                # Each wave-pass writes a 64-kv half of one d-block.  The d-block
                # data span (2048B) is padded to V_DBLOCK_STRIDE; HW adds lane*16
                # within the chosen 64-kv (1024B) half.
                half = wave_id % fx.Index(2)
                lds_byte = (
                    fx.Index(lds_offset)
                    + v_buf_off
                    + d_block * fx.Index(V_DBLOCK_STRIDE)
                    + half * fx.Index(WAVE_DMA_STRIDE)
                )
                _dma_issue(v_rsrc, lds_byte, voff)

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
            q_packs.append(Vec(raw).bitcast(fx.Int32))  # vec<8xi32>

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
                if const_expr(ks == 0):
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

        def do_qk(k_buf_off, preloaded_kw, v_off):
            kw = [None] * 4
            vw_prime = [None] * PREFETCH_DEPTH
            for u in range_constexpr(PREFETCH_DEPTH):
                kw[u] = preloaded_kw[u]
            s_accs = [mfma.zero_value for _ in range_constexpr(N_KV_TILES)]
            for u in range_constexpr(QK_UNITS):
                nt = u // K_STEPS
                ks = u % K_STEPS
                if const_expr(u + PREFETCH_DEPTH < QK_UNITS):
                    un = u + PREFETCH_DEPTH
                    kw[(u + PREFETCH_DEPTH) % 4] = _load_k_unit(
                        k_buf_off, un // K_STEPS, un % K_STEPS
                    )
                else:
                    vi = u - (QK_UNITS - PREFETCH_DEPTH)
                    vw_prime[vi] = read_v_pack(v_off, vi // PV_K_STEPS, vi % PV_K_STEPS)
                _sched_barrier()
                s_accs[nt] = mfma.call(kw[u % 4], q_packs[ks], s_accs[nt])
            return s_accs, vw_prime

        def do_softmax(s_accs, m_running, set_prio=False):
            if const_expr(SOFTMAX_PRIO != 0 and set_prio):
                rocdl.s_setprio(SOFTMAX_PRIO)
            local_max = Vec(s_accs[0])[0]
            for nt in range_constexpr(N_KV_TILES):
                for r in range_constexpr(C_F32_PER_LANE):
                    if const_expr(nt == 0 and r == 0):
                        continue
                    local_max = _fmax(local_max, Vec(s_accs[nt])[r])
            peer_max = fx.Float32(local_max).shuffle_xor(
                fx.Int32(32), fx.Int32(WARP_SIZE)
            )
            row_max_int = _fmax(local_max, peer_max)
            m_new = _fmax(m_running, row_max_int)
            corr = ArithValue(_fmul(_fsub(m_running, m_new), scale_log2e)).exp2(
                fastmath=fm_fast
            )

            neg_scaled_m_new = _fsub(c_zero_f, _fmul(scale_log2e, m_new))
            n_groups = C_F32_PER_LANE // 4

            p_words = []
            for nt in range_constexpr(N_KV_TILES):
                for rg in range_constexpr(n_groups):
                    ps = [
                        ArithValue(
                            _fadd(
                                _fmul(Vec(s_accs[nt])[rg * 4 + i], scale_log2e),
                                neg_scaled_m_new,
                            )
                        ).exp2(fastmath=fm_fast)
                        for i in range_constexpr(4)
                    ]
                    p_words.append(_f32x4_to_fp8_word(*ps))
            p_pack = Vec.from_elements(p_words, fx.Int32)
            p_ks_list = [
                Vec(p_pack).shuffle(Vec(p_pack), list(range(r * 8, r * 8 + 8)))
                for r in range_constexpr(PV_K_STEPS)
            ]
            lr = mfma.zero_value
            for ks in range_constexpr(PV_K_STEPS):
                lr = mfma.call(ones_pack, p_ks_list[ks], lr)
            p_rowsum = Vec(lr)[0]
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
        if is_g0:
            dma_v(fx.Index(0), fx.Index(0))
        else:
            dma_k(fx.Index(0), fx.Index(0))

        _wait_vmcnt()
        _gpu_barrier()

        kvw = [None] * PREFETCH_DEPTH
        for u in range_constexpr(PREFETCH_DEPTH):
            kvw[u] = _load_k_unit(fx.Index(0), u // K_STEPS, u % K_STEPS)

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

        _wait_lgkmcnt()
        if is_g0:

            def g0_iter0():
                _, k_buf_off, _, _, v_next = _bufs(fx.Index(0))
                dma_v(v_next, fx.Index(BLOCK_N))

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
                sA, vwp_new = do_qk(k_buf_off, preloaded_kw=kw_prime, v_off=v_cur_off)
                # PHASE 2 (softmax phase): softmax(i) | DMA V^{i+1}.
                dma_v(v_next, next_kv)
                m_new, corr_new, p_new, prowsum_new = do_softmax(sA, m_r)
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
                _, k_buf_off, _, k_next, _ = _bufs(fx.Index(0))
                dma_k(k_next, fx.Index(BLOCK_N))

                v0_off = fx.Index(LDS_V_OFF)  # V^0 buf = LDSV[0]
                sB, vwp = do_qk(k_buf_off, preloaded_kw=kvw, v_off=v0_off)
                _wait_lgkmcnt()
                _wait_vmcnt()
                _gpu_barrier()
                return (
                    m_init,
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
                _m_e, corrf, pf, prowsumf = do_softmax([sce0, sce1, sce2, sce3], m_e)
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
                    [ss0, ss1, ss2, ss3], m_r, True
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
                sB, vwp_new = do_qk(k_buf_off, preloaded_kw=kw_prime, v_off=v_cur_off)
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

        inv_l = rocdl.rcp(T.f32, l_final)
        inv_l_v = _fmul(inv_l, v_descale)
        inv_l_vec = Vec.from_elements([inv_l_v], fx.Float32).broadcast_to(
            C_F32_PER_LANE
        )

        if q_in_bounds:
            for dt in range_constexpr(D_TILES):
                o_norm = Vec(o_finals[dt]) * inv_l_vec
                for r in range_constexpr(C_F32_PER_LANE):
                    o_val = Vec(o_norm)[r]
                    o_bf16 = fx.Float32(o_val).to(bf16_dtype)
                    # O C-layout: row = d = hi*4 + (r%4) + 8*(r//4) (+ dt*32);
                    # col = q = lo.  But this wave's q = q_row (= lo-based).
                    d_row = hi * 4 + (r % 4) + 8 * (r // 4) + dt * 32
                    o_global = global_idx(q_row, fx.Index(d_row))
                    gep = fx.buffer_ops.get_element_ptr(
                        o_ptr, fx.Int64(o_global), elem_type=bf16_type
                    )
                    _pointer_store(o_bf16, gep)

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
