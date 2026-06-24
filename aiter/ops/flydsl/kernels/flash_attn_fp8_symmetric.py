# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Symmetric-compute, split-DMA fp8 E4M3 flash-attention forward (gfx950).

A variant of :mod:`aiter.ops.flydsl.kernels.flash_attn_fp8_pingpong` with a
DIFFERENT ping-pong structure:

- All 8 waves run the IDENTICAL ``QK -> softmax -> PV`` sequence on the SAME
  KV tile each iteration (no wave-role swap, no deferred PV).
- The only divergence between the two 4-wave groups is the next-tile prefetch:
  G0 issues ``dma_k`` (next K tile), G1 issues ``dma_v`` (next V tile).
- Both K and V are DOUBLE-buffered (buf = i % 2).  The next-tile global->LDS
  DMA overlaps each group's own compute; one ``vmcnt(0)+s_barrier`` per
  iteration publishes K^{i+1}/V^{i+1} and keeps all 8 waves in lockstep.

Everything else (the MFMA atom, LDS padding, HW-transpose V read, register-
resident fp8 P, ones-column row-sum, the 4-window ds_read prefetch in do_qk /
apply_pv, and the SageAttention log2-domain online softmax) is carried over
unchanged from the role-ping-pong kernel.  See that module's docstring for the
MFMA fragment layouts and LDS layout details.

Config (v1): HEAD_DIM=128, BLOCK_M=256 (8 waves x 32 q-rows), BLOCK_N=128,
non-causal, no GQA, fp8 E4M3 in, bf16 out.
"""

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
    """Build the symmetric-compute split-DMA fp8 flash-attention launcher.

    Returns a callable ``launch(Q, K, V, O, q_descale, k_descale,
    v_descale, batch, seq_len)`` (plus ``.compile`` for explicit AOT
    compilation).  Q/K/V are flat fp8 (viewed as int8), O flat bf16.
    """
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

    # ---- LDS layout (fp8 element type) ----
    # K tile : [BLOCK_N kv][HEAD_DIM d]          row-major
    # V tile : [d_block(8)][BLOCK_N kv_perm][d_in(16)]  kv-permuted
    # P is register-resident (not in LDS).  See the role-ping-pong kernel
    # docstring for the V permutation / read_v_pack derivation.
    K_STRIDE = HEAD_DIM
    V_KV_STRIDE = 16  # bytes per kv within a 16-wide d block
    N_DBLOCKS = HEAD_DIM // 16  # 8
    # ---- LDS bank-conflict padding (asm-style, mi350_fmha_hd128_fp8) ----
    PAD_K = 16
    PAD_V = 16
    K_UNIT_ROWS = 8  # one wave writes 8 contiguous kv rows per DMA pass
    K_DATA = BLOCK_N * K_STRIDE  # 16384 unpadded K tile bytes
    K_UNIT_STRIDE = K_UNIT_ROWS * K_STRIDE + PAD_K  # 1040
    N_K_UNITS = BLOCK_N // K_UNIT_ROWS  # 16
    V_DBLOCK_STRIDE = BLOCK_N * V_KV_STRIDE + PAD_V  # 2064 (padded)
    # Symmetric pipeline: K^i AND V^i are both read in iteration i; the next
    # tile (i+1) is DMA'd this iteration into the OTHER buffer -> both K and V
    # double-buffer (i % 2).  No deferred PV, so V needs no third buffer.
    NUM_BUF_K = 2
    NUM_BUF_V = 2
    LDS_K_TILE = N_K_UNITS * K_UNIT_STRIDE  # 16640 (padded)
    LDS_V_TILE = N_DBLOCKS * V_DBLOCK_STRIDE  # 16512 (padded)
    LDS_K_SIZE = NUM_BUF_K * LDS_K_TILE
    LDS_V_SIZE = NUM_BUF_V * LDS_V_TILE
    LDS_K_OFF = 0
    LDS_V_OFF = LDS_K_OFF + LDS_K_SIZE
    LDS_TOTAL = LDS_V_OFF + LDS_V_SIZE  # ~64.8 KB

    allocator = SmemAllocator(
        None,
        arch=gpu_arch,
        global_sym_name="flash_attn_fp8_smem",
    )
    lds_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_offset + LDS_TOTAL

    bf16_dtype = fx.BFloat16

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
        # Storage is int8 (fp8 == 1 byte); bitcast to fp8 / i32 only at the
        # compute boundary (the LLVM dialect path can't lower fp8 vector types).
        i8_dtype = fx.Int8
        i8_type = i8_dtype.ir_type
        bf16_type = bf16_dtype.ir_type
        v_i8x16 = Vec.make_type(16, i8_dtype)  # 16 bytes for coop load
        v_i8x32 = Vec.make_type(A_FP8_PER_LANE, i8_dtype)  # MFMA operand bytes

        fm_fast = fx.arith.FastMathFlags.fast

        def _fadd(a, b):
            return arith.addf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fsub(a, b):
            return arith.subf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmul(a, b):
            return arith.mulf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmax(a, b):
            return arith.MaxNumFOp(_raw(a), _raw(b), fastmath=fm_fast).result

        def _f32_to_fp8_byte(f):
            packed = rocdl.cvt_pk_fp8_f32(
                T.i32, _raw(f), _raw(c_zero_f), fx.Int32(0), False
            )
            return arith.trunci(T.i8, _raw(packed))

        def _f32x4_to_fp8_word(f0, f1, f2, f3):
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

        # ===================================================================
        # DMA global->LDS (raw_ptr_buffer_load_lds): each DMA is issued by ONE
        # 4-wave group (G0 -> K, G1 -> V), 256 lanes x 16B = 4096 B/pass; a
        # 128x128 fp8 tile is 16384 B -> DMA_PASSES = 4.
        # ===================================================================
        DMA_BYTES = 16
        DMA_LANES = (NUM_WAVES // 2) * WARP_SIZE  # 256 (one 4-wave group)
        DMA_PASSES = K_DATA // (DMA_LANES * DMA_BYTES)  # 4 (unpadded data size)
        WAVE_DMA_STRIDE = WARP_SIZE * DMA_BYTES  # 1024: one wave's 64-cell span
        G1_TID0 = DMA_LANES  # first lane of G1 (waves 4-7)

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
            # K row-major into LDSK[buf], cooperatively by G0 (waves 0-3).
            # group-local cell c = p*256 + ltid maps to kv = c//8, d = (c%8)*16.
            k_buf_off = fx.Index(LDS_K_OFF) + buf * fx.Index(LDS_K_TILE)
            ltid = tid  # 0..255 within G0
            lwave = wave_id  # 0..3 within G0
            for p in range_constexpr(DMA_PASSES):
                c = fx.Index(p * DMA_LANES) + ltid
                kv = c // fx.Index(8)
                d = (c % fx.Index(8)) * fx.Index(16)
                kv_abs = kv_start + kv
                in_b = kv_abs < seq_len_v
                kv_safe = fx.Index(ArithValue(in_b).select(kv_abs, fx.Index(0)))
                voff = kv_safe * fx.Index(STRIDE_TOKEN) + d
                # One wave-pass writes 8 contiguous kv rows == one padded K unit;
                # unit index = p*4 + lwave.  HW adds lane*16 within the 1024B data.
                lds_byte = (
                    fx.Index(lds_offset)
                    + k_buf_off
                    + (fx.Index(p * (NUM_WAVES // 2)) + lwave) * fx.Index(K_UNIT_STRIDE)
                )
                _dma_issue(k_rsrc, lds_byte, voff)

        def dma_v(buf, kv_start):
            # V into the C-layout-permuted d-block LDS layout in LDSV[buf],
            # cooperatively by G1 (waves 4-7).  LDS position p = c%BLOCK_N maps
            # to actual kv row kv_actual(p) = blk*32 + hi_group*4 + grp*8 + fine.
            v_buf_off = fx.Index(LDS_V_OFF) + buf * fx.Index(LDS_V_TILE)
            ltid = tid - fx.Index(G1_TID0)  # 0..255 within G1
            lwave = wave_id - fx.Index(NUM_WAVES // 2)  # 0..3 within G1
            for p in range_constexpr(DMA_PASSES):
                c = fx.Index(p * DMA_LANES) + ltid
                d_block = c // fx.Index(BLOCK_N)
                kv_lds_pos = c % fx.Index(BLOCK_N)
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
                # Each wave-pass writes a 64-kv half of one d-block.
                half = lwave % fx.Index(2)
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

        def publish_barrier():
            # Mid-iteration (between softmax and apply_pv): drain this
            # iteration's next-tile DMA (vmcnt) and make it visible to all waves
            # (s_barrier) so apply_pv's K^{i+1} preload -- and the next
            # iteration's reads -- see the just-published K^{i+1}/V^{i+1}.
            llvm.InlineAsmOp(
                None,
                [],
                "s_waitcnt vmcnt(0)\n\ts_barrier",
                "",
                has_side_effects=True,
            )

        def war_barrier():
            # Loop top: drain the PRIOR iteration's apply_pv LDS reads -- the
            # V^{i-1} reads of buf (i+1)%2 and the carried K-unit preload
            # (lgkmcnt) -- then s_barrier so NO wave is still reading buf
            # (i+1)%2 when this iteration's DMA overwrites it (cross-wave WAR).
            llvm.InlineAsmOp(
                None,
                [],
                "s_waitcnt lgkmcnt(0)\n\ts_barrier",
                "",
                has_side_effects=True,
            )

        # ---- Preload Q packs (register resident) for this wave ----
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
            q_packs.append(Vec(raw).bitcast(fx.Int32))  # vec<8xi32>

        # ---- V HW-transpose read + PV accumulate helper ----
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

        # Prefetch depth for the register-window software pipeline.
        PREFETCH_DEPTH = 2
        QK_UNITS = N_KV_TILES * K_STEPS  # 8 (nt outer, ks inner)

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
            # GEMM1: S[kv,q] = K @ Q^T over N_KV_TILES kv subtiles, 4-window
            # rotating register buffer: kw[u%4] consumed at MFMA u.
            # preloaded_kw[0..1] seed from the loop-top K preload.
            # Last 2 dead windows (u=6,7) preload V^i units 0,1 for this
            # iteration's apply_pv.
            kw = [None] * 4
            vw_prime = [None] * PREFETCH_DEPTH
            for u in range_constexpr(PREFETCH_DEPTH):
                kw[u] = preloaded_kw[u]
            s_accs = [mfma.zero_value for _ in range_constexpr(N_KV_TILES)]
            for u in range_constexpr(QK_UNITS):
                nt = u // K_STEPS
                if const_expr(u + PREFETCH_DEPTH < QK_UNITS):
                    un = u + PREFETCH_DEPTH
                    kw[(u + PREFETCH_DEPTH) % 4] = _load_k_unit(
                        k_buf_off, un // K_STEPS, un % K_STEPS
                    )
                    rocdl.sched_group_barrier(rocdl.mask_dsrd, 2, 0)
                else:
                    vi = u - (QK_UNITS - PREFETCH_DEPTH)
                    vw_prime[vi] = read_v_pack(v_off, vi // PV_K_STEPS, vi % PV_K_STEPS)
                    rocdl.sched_group_barrier(rocdl.mask_dsrd, 4, 0)
                s_accs[nt] = mfma.call(kw[u % 4], q_packs[u % K_STEPS], s_accs[nt])
                rocdl.sched_group_barrier(rocdl.mask_mfma, 1, 0)
            return s_accs, vw_prime

        def do_softmax(s_accs, m_running):
            # Online softmax (VALU/transcendental only): returns (m_new, corr,
            # p_pack, p_rowsum).  p_pack is the PV B-operand fp8 words.
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
            # Row sum via ones-column MFMAs: ones(A) @ P(B) -> lr(C).
            p_ks_list = [
                Vec(p_pack).shuffle(Vec(p_pack), list(range(r * 8, r * 8 + 8)))
                for r in range_constexpr(PV_K_STEPS)
            ]
            lr = mfma.zero_value
            for ks in range_constexpr(PV_K_STEPS):
                lr = mfma.call(ones_pack, p_ks_list[ks], lr)
            p_rowsum = Vec(lr)[0]
            return m_new, corr, p_pack, p_rowsum

        # All-ones FP8 E4M3 A-operand for the ones-column row-sum MFMAs.
        ones_pack = Vec.filled(A_FP8_PER_LANE // 4, 0x38383838, fx.Int32)

        def apply_pv(
            o_accs, l_acc, p_pack, p_rowsum, corr, v_off, preloaded_vw, k_buf_off
        ):
            # O *= corr ; then O += V^T@P over the PV_K_STEPS 64-kv K-steps.
            # L is a scalar VALU online update L = L*corr + rowsum.
            # 4-window rotating buffer: vw[u%4] consumed at MFMA u.
            # preloaded_vw[0..1] seed from the preceding do_qk dead windows.
            # Last 2 dead windows (u=6,7) preload K^{i+1} units 0,1 (published
            # by this iteration's publish barrier) for the NEXT iteration's
            # do_qk -- returned as kw_prime and threaded through the carry.
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
                    rocdl.sched_group_barrier(rocdl.mask_dsrd, 4, 0)
                else:
                    ki = u - (PV_UNITS - PREFETCH_DEPTH)
                    kw_prime[ki] = _load_k_unit(k_buf_off, ki // K_STEPS, ki % K_STEPS)
                    rocdl.sched_group_barrier(rocdl.mask_dsrd, 2, 0)
                o[dt] = mfma.call(vw[u % 4], p_ks_list[ks], o[dt])
                rocdl.sched_group_barrier(rocdl.mask_mfma, 1, 0)
            return o, l2, kw_prime

        # ---- Shared init values ----
        m_init = c_neg_inf
        l_init = c_zero_f
        o_init = [
            Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)
            for _ in range_constexpr(D_TILES)
        ]

        # Prologue: DMA K^0 -> LDSK0 (by G0) and V^0 -> LDSV0 (by G1), then
        # publish.  All 8 waves cross the barrier.
        is_g0 = wave_id < fx.Index(NUM_WAVES // 2)
        if is_g0:
            dma_k(fx.Index(0), fx.Index(0))
        else:
            dma_v(fx.Index(0), fx.Index(0))

        _wait_vmcnt()

        # Prologue K preload: prime the first 2 K^0 units 0,1 (buf 0, just
        # published) for the first do_qk.  Subsequent iterations get their K
        # units from the preceding apply_pv (threaded via the carry).
        kvw0 = [
            _load_k_unit(fx.Index(LDS_K_OFF), u // K_STEPS, u % K_STEPS)
            for u in range_constexpr(PREFETCH_DEPTH)
        ]

        of0 = Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)
        of1 = Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)
        of2 = Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)
        of3 = Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)
        lf = c_zero_f

        loop_step = fx.Int32(BLOCK_N)
        zero_i32 = fx.Int32(0)
        init_carry = [
            m_init,
            l_init,
            o_init[0],
            o_init[1],
            o_init[2],
            o_init[3],
            kvw0[0],
            kvw0[1],
        ]
        gpu.barrier()

        def loop_body(iv, iter_args, is_g0):
            # Symmetric per-iteration pipeline run by ALL waves; only the
            # next-tile DMA call differs between groups.  Carry threads the
            # next do_qk's first 2 K units (kwp), preloaded by the prior
            # apply_pv (or the prologue for iteration 0).
            m_r, l_a, oo0, oo1, oo2, oo3, kwp0, kwp1 = iter_args
            kv_start = fx.Index(iv)
            i = kv_start // fx.Index(BLOCK_N)
            k_cur = i % fx.Index(NUM_BUF_K)
            v_cur = i % fx.Index(NUM_BUF_V)
            k_nxt = (i + fx.Index(1)) % fx.Index(NUM_BUF_K)
            v_nxt = (i + fx.Index(1)) % fx.Index(NUM_BUF_V)
            # do_qk / apply_pv consume byte OFFSETS; dma_k / dma_v take buffer
            # INDICES (they multiply by the tile size internally).
            k_buf = fx.Index(LDS_K_OFF) + k_cur * fx.Index(LDS_K_TILE)
            v_buf = fx.Index(LDS_V_OFF) + v_cur * fx.Index(LDS_V_TILE)
            k_next_buf = fx.Index(LDS_K_OFF) + k_nxt * fx.Index(LDS_K_TILE)
            next_kv = kv_start + fx.Index(BLOCK_N)
            # WAR guard: drain the prior iteration's apply_pv reads (incl. the
            # carried K preload) and cross-wave sync before the DMA overwrites
            # buf (i+1)%2.
            # war_barrier()
            if is_g0:
                dma_k(k_nxt, next_kv)
            else:
                dma_v(v_nxt, next_kv)
            _wait_lgkmcnt()
            sA, vwp = do_qk(k_buf, preloaded_kw=[kwp0, kwp1], v_off=v_buf)
            if not is_g0 or not is_first:
                gpu.barrier()
            m_new, corr_new, p_new, prowsum_new = do_softmax(sA, m_r)
            gpu.barrier()
            # Publish K^{i+1}/V^{i+1} before apply_pv preloads K^{i+1}.
            # publish_barrier()
            _wait_lgkmcnt()
            oA, lA, kwp_new = apply_pv(
                [oo0, oo1, oo2, oo3],
                l_a,
                p_new,
                prowsum_new,
                corr_new,
                v_buf,
                vwp,
                k_next_buf,
            )
            # gpu.barrier()
            return [m_new, lA, oA[0], oA[1], oA[2], oA[3], kwp_new[0], kwp_new[1]]

        if is_g0:
            for_op = scf.ForOp(
                _raw(zero_i32),
                _raw(seq_len),
                _raw(loop_step),
                [_raw(v) for v in init_carry],
            )
            with ir.InsertionPoint(for_op.body):
                iv = for_op.induction_variable
                args = [
                    as_dsl_value(a, ex)
                    for a, ex in zip(for_op.inner_iter_args, init_carry)
                ]
                res = loop_body(iv, args, is_g0=True)
                scf.YieldOp([_raw(r) for r in res])
            fin = [as_dsl_value(r, ex) for r, ex in zip(for_op.results, init_carry)]
            of0, of1, of2, of3 = fin[2], fin[3], fin[4], fin[5]
            lf = fin[1]
        else:
            for_op = scf.ForOp(
                _raw(zero_i32),
                _raw(seq_len),
                _raw(loop_step),
                [_raw(v) for v in init_carry],
            )
            with ir.InsertionPoint(for_op.body):
                iv = for_op.induction_variable
                args = [
                    as_dsl_value(a, ex)
                    for a, ex in zip(for_op.inner_iter_args, init_carry)
                ]
                res = loop_body(iv, args, is_g0=False)
                scf.YieldOp([_raw(r) for r in res])
            fin = [as_dsl_value(r, ex) for r, ex in zip(for_op.results, init_carry)]
            of0, of1, of2, of3 = fin[2], fin[3], fin[4], fin[5]
            lf = fin[1]

        o_finals = [of0, of1, of2, of3]
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
                    d_row = hi * 4 + (r % 4) + 8 * (r // 4) + dt * 32
                    o_global = global_idx(q_row, fx.Index(d_row))
                    gep = fx.buffer_ops.get_element_ptr(
                        o_ptr, fx.Int64(o_global), elem_type=bf16_type
                    )
                    _pointer_store(o_bf16, gep)

    @flyc.jit
    def launch_fp8_attn_v2(
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
            launch_fp8_attn_v2,
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

    launch_fp8_attn_v2.compile = _compile
    return launch_fp8_attn_v2


compile_flash_attn_fp8 = build_flash_attn_fp8_module
