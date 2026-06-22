# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Correct (un-tuned) fp8 E4M3 flash-attention forward for CDNA4 (gfx950).

This is the *correctness* skeleton requested as a precursor to a tuned,
ping-pong fp8 FMHA kernel.  It deliberately keeps things simple:

- All 8 waves run the identical ``QK -> softmax -> PV`` sequence, each on
  its own 32 query rows (no wave-role ping-pong).
- One CDNA4 ``mfma_scale_f32_32x32x64_f8f6f4`` atom (fp8 E4M3, f32 accum),
  reused from :mod:`aiter.ops.flydsl.rocdl_mfma_fp8`.
- SageAttention per-tensor recipe: exact softmax in the log2 domain
  (``exp2``), online over KV tiles, with scalar Q/K/V descales.
- ``P`` is register-resident (Step 1): the GEMM1 C-layout is reshaped to
  the GEMM2 B-operand via an intra-wave hi-peer ``shuffle_xor`` (no LDS).
- ``V`` is stored row-major in 16-wide d-blocks and read with HW transpose
  ``ds_read_tr8_b64`` (Step 2) for the PV A-operand (no scatter store).

Layout: Q/K/V/O are 1D flattened from BSHD (batch, seq, heads, head_dim).
Grid:   (num_q_tiles * batch * num_heads,)  with num_q_tiles = seq/BLOCK_M.
Block:  (512,) == 8 waves of 64 lanes.

Config (v1): HEAD_DIM=128, BLOCK_M=256 (8 waves x 32 q-rows), BLOCK_N=64,
non-causal, no GQA, fp8 E4M3 in, bf16 out.

The MFMA fragment layouts (lane L, lo=L%32, hi=L//32, verified in
``aiter/ops/flydsl/rocdl_mfma_fp8.py``):

- A (M=32 x K=64): ``A_frag[L][v] = A[row=lo, col=hi*32+v]``      v in [0,32)
- B (K=64 x N=32): ``B_frag[L][v] = B[row(K)=hi*32+v, col(N)=lo]``  v in [0,32)
- C (M=32 x N=32): ``C_frag[L][v] = C[row=hi*4+(v%4)+8*(v//4), col=lo]`` v in [0,16)

GEMM1 computes ``S = K @ Q^T`` so the C accumulator has M=kv (in the
value index) and N=q (in the lane), matching the bf16 reference's online
softmax which reduces over the value index (cheap, no cross-lane max).

GEMM2 computes ``O = V^T @ P`` so the O accumulator has M=d (value) and
N=q (lane) -- the store layout used by the bf16 reference.
"""

import math as host_math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as _fly
from flydsl._mlir.dialects import llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
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
    """Build the fp8 flash-attention launcher (correctness variant).

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
    P_PACK_WORDS = PV_K_STEPS * (A_FP8_PER_LANE // 4)  # 16 fp8 words for P B-operand

    if softmax_scale is None:
        softmax_scale = 1.0 / host_math.sqrt(head_dim)

    # ---- LDS layout (fp8 element type) ----
    # K tile : [BLOCK_N kv][HEAD_DIM d]          row-major
    # V tile : [d_block(8)][BLOCK_N kv][d_in(16)] row-major (Step 2)
    #
    # P is NOT in LDS (Step 1, register-resident): the GEMM1 C-layout P is
    # packed to fp8 in registers and reshaped into the GEMM2 B-operand via an
    # intra-wave cross-lane shuffle (hi-peer exchange), so no LDS round-trip.
    #
    # V layout (Step 2, HW-transpose read): V is stored row-major in 16-wide
    # d blocks so the cooperative load is a clean 16-byte vector store (no
    # scatter).  The GEMM2 A-operand
    #   A_frag[L][8*kc + r] = V[kv = hi*32 + 8*kc + r, d = lo + dt*32]
    # is produced by ds_read_tr8_b64 (HW transpose-on-read): for a 16-lane
    # group the atom returns result[lane][r] = LDS[group_base + lane%16 + 16*r]
    # (an 8x16 byte transpose).  With the d-block layout the per-(dt,kc) group
    # base = d_block*(BLOCK_N*16) + (hi*32+8*kc)*16 and d_block = lo//16 + 2*dt,
    # giving exactly V[kv, d] for the A fragment.  4 reads (kc=0..3) per d-tile.
    K_STRIDE = HEAD_DIM
    V_KV_STRIDE = 16  # bytes per kv within a 16-wide d block
    V_DBLOCK_STRIDE = BLOCK_N * V_KV_STRIDE
    N_DBLOCKS = HEAD_DIM // 16  # 8
    # Triple-buffer K/V (Step 7): with deferred PV the buffer overwritten in
    # iteration i (tile i-3) was last read two iterations earlier, so a single
    # barrier/iteration suffices (no extra hazard barrier) while PV(i-1) still
    # reads tile i-1 from a live buffer.
    NUM_BUF = 3
    LDS_K_TILE = BLOCK_N * K_STRIDE
    LDS_V_TILE = N_DBLOCKS * V_DBLOCK_STRIDE  # == HEAD_DIM * BLOCK_N
    LDS_K_SIZE = NUM_BUF * LDS_K_TILE
    LDS_V_SIZE = NUM_BUF * LDS_V_TILE
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
        # IR types must be materialized inside the kernel (MLIR context).
        #
        # NOTE: the raw LLVM dialect load/store path does not lower fp8
        # vector types ("unknown LLVM dialect type" -> crash), so all
        # global/LDS *storage* is done as int8 (1 byte, same as fp8) and we
        # bitcast to fp8 / i32 only at the compute boundary.  The MFMA atom
        # consumes vec<8xi32> operands, which we obtain by bitcasting i8x32.
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
        # qk_scale = q_descale * k_descale * softmax_scale
        c_log2e = fx.Float32(_LOG2E)
        qk_scale = _fmul(_fmul(q_descale, k_descale), fx.Float32(softmax_scale))
        scale_log2e = _fmul(qk_scale, c_log2e)
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_f = fx.Float32(0.0)

        # ===================================================================
        # Cooperative loads.  BLOCK_SIZE=512 lanes, each lane loads 16 fp8
        # (1 row chunk).  HEAD_DIM=128 -> 8 lanes/row, 512/8 = 64 rows/pass.
        # K/V tiles are BLOCK_N=64 rows -> exactly one pass.
        # ===================================================================
        VEC = 16
        THREADS_PER_ROW = HEAD_DIM // VEC  # 8
        load_row = tid // THREADS_PER_ROW  # 0..63 (one pass)
        load_col = (tid % THREADS_PER_ROW) * VEC  # 0,16,...,112
        # BLOCK_SIZE=512 lanes cover 64 rows/pass; BLOCK_N=128 -> 2 passes.
        ROWS_PER_PASS = BLOCK_SIZE // THREADS_PER_ROW  # 64
        N_LOAD_PASSES = BLOCK_N // ROWS_PER_PASS  # 2

        def _load_global_i8x16(ptr, base_idx):
            gep = fx.buffer_ops.get_element_ptr(
                ptr, fx.Int64(base_idx), elem_type=i8_type
            )
            return _pointer_load(v_i8x16, gep)

        zero_vec16 = Vec.filled(VEC, 0, i8_dtype)

        def global_load_kv(kv_start, src_ptr):
            # Issue the (latency-bound) global loads for this lane's row chunks
            # (N_LOAD_PASSES rows, 16 bytes each).  Returns a register
            # vec<(16*N_LOAD_PASSES)xi8> = pass chunks concatenated; OOB zeroed.
            chunks = []
            for p in range_constexpr(N_LOAD_PASSES):
                row_idx = kv_start + load_row + fx.Index(p * ROWS_PER_PASS)
                in_bounds = row_idx < seq_len_v
                safe_row = fx.Index(ArithValue(in_bounds).select(row_idx, fx.Index(0)))
                g_idx = global_idx(safe_row, load_col)
                vec = _load_global_i8x16(src_ptr, g_idx)
                vec = ArithValue(in_bounds).select(vec, zero_vec16.ir_value())
                chunks.append(Vec(vec))
            out = chunks[0]
            for p in range_constexpr(1, N_LOAD_PASSES):
                out = out.shuffle(chunks[p], list(range(16 * (p + 1))))
            return out

        def _k_swizzle(row_idx, col_idx):
            # Step 3: XOR swizzle at 16-element granularity, col ^ ((row&7)<<4),
            # to break the 32-way LDS bank conflict on the GEMM1 K read (32 lanes
            # sharing the same lo land on identical banks without it).
            mask = (row_idx & fx.Index(0x7)) << fx.Index(4)
            return col_idx ^ mask

        def _pass_chunk(vec, p):
            return Vec(vec).shuffle(Vec(vec), list(range(p * 16, p * 16 + 16)))

        def store_k_lds(vec, lds_off):
            # One 16-element block per pass; the swizzle relocates each whole
            # block (no intra-block split needed).
            for p in range_constexpr(N_LOAD_PASSES):
                row = load_row + fx.Index(p * ROWS_PER_PASS)
                swz_col = _k_swizzle(row, load_col)
                lds_idx = fx.Index(lds_off) + row * K_STRIDE + swz_col
                Vec(_pass_chunk(vec, p)).store(lds, [lds_idx])

        def store_v_lds_dblocked(vec, lds_off):
            # Step 2: store the lane's 16 contiguous d-values (one kv row) as a
            # single 16-byte vector into the d-block layout
            #   Vlds[d_block][kv][d_in] : d_block = load_col//16, kv = row.
            d_block = load_col // fx.Index(16)
            for p in range_constexpr(N_LOAD_PASSES):
                row = load_row + fx.Index(p * ROWS_PER_PASS)
                v_idx = (
                    fx.Index(lds_off)
                    + d_block * fx.Index(V_DBLOCK_STRIDE)
                    + row * fx.Index(V_KV_STRIDE)
                )
                Vec(_pass_chunk(vec, p)).store(lds, [v_idx])

        # ---- Preload Q packs (register resident) for this wave ----
        # A-operand for GEMM1 is K; B-operand is Q (Q^T).  But we keep Q in
        # registers as the B-operand pack:
        #   B_frag[L][v] = Q[q = lo, d = ks*64 + hi*32 + v]
        # The wave owns q-rows [q_start + wave_q_offset, +32).  In the B
        # fragment the q index is the lane lo, so each lane reads its own
        # q-row = q_start + wave_q_offset + lo.
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
        # Step 5: ones-column L.  L[q] = sum_kv P[kv,q] is computed by an MFMA
        # (A = all-ones fp8, B = P) instead of a VALU rowsum + cross-lane add.
        # The MFMA K-dim sums all 64 kv (both hi halves), so no peer shuffle is
        # needed, and L lands in the same C-layout as O -> it rides the corr
        # rescale.  fp8 E4M3 1.0 == 0x38, so each i32 lane word is 0x38383838.
        ones_pack = Vec.filled(A_FP8_PER_LANE // 4, 0x38383838, fx.Int32)

        # ---- V HW-transpose read + PV/L accumulate helpers (Step 7) ----
        # Factored out so both the deferred in-loop PV (tile i-1) and the
        # epilogue PV (tile N-1) share one code path.
        v_tr8_ty = Vec.make_type(2, fx.Int32)
        lo_in_grp = lo % fx.Index(16)

        def read_v_pack(v_off, dt, ks):
            d_block = (lo // fx.Index(16)) + fx.Index(2 * dt)
            grp_db = fx.Index(lds_offset) + v_off + d_block * fx.Index(V_DBLOCK_STRIDE)
            reads = []
            for kc in range_constexpr(4):
                kv0 = hi * fx.Index(32) + fx.Index(ks * 64 + 8 * kc)
                byte_off = (
                    grp_db + kv0 * fx.Index(V_KV_STRIDE) + lo_in_grp * fx.Index(8)
                )
                ptr = fx.buffer_ops.create_llvm_ptr(fx.Int64(byte_off), address_space=3)
                reads.append(Vec(rocdl.ds_read_tr8_b64(v_tr8_ty, ptr).result))
            ab = reads[0].shuffle(reads[1], list(range(4)))
            cd = reads[2].shuffle(reads[3], list(range(4)))
            return ab.shuffle(cd, list(range(8)))

        def apply_pv(o_accs, l_acc, p_pack, corr, v_off):
            # O,L *= corr ; then O += V^T@P and L += ones@P, summed over the
            # PV_K_STEPS 64-kv K-steps that tile BLOCK_N.  The PV/L MFMAs (matrix
            # pipe) are issued next to the following tile's softmax exp2
            # (transcendental pipe) so the two pipes overlap (deferred PV).
            corr_vec = Vec.from_elements([corr], fx.Float32).broadcast_to(
                C_F32_PER_LANE
            )
            # Scale O/L by corr in place on the first K-step (fold into the slot
            # the MFMA accumulates into) so the rescaled and pre-rescale copies
            # never co-exist as two full 64-VGPR lists.
            o = list(o_accs)
            l2 = l_acc
            for ks in range_constexpr(PV_K_STEPS):
                p_ks = Vec(p_pack).shuffle(Vec(p_pack), list(range(ks * 8, ks * 8 + 8)))
                if const_expr(ks == 0):
                    l2 = _fmul(Vec(l2), corr_vec)
                l2 = mfma.call(ones_pack, p_ks, l2)
                for dt in range_constexpr(D_TILES):
                    v_pack = read_v_pack(v_off, dt, ks)
                    if const_expr(ks == 0):
                        o[dt] = _fmul(Vec(o[dt]), corr_vec)
                    o[dt] = mfma.call(v_pack, p_ks, o[dt])
            return o, l2

        def do_qk(k_buf_off):
            # GEMM1: S[kv,q] = K @ Q^T over N_KV_TILES kv subtiles.  Returns the
            # list of N_KV_TILES accumulators (each vec<16xf32>).
            s_accs = []
            for nt in range_constexpr(N_KV_TILES):
                s_acc = mfma.zero_value
                for ks in range_constexpr(K_STEPS):
                    kv_row = lo + fx.Index(nt * 32)
                    d_base = fx.Index(ks * MFMA_K) + hi * 32
                    blk_lo = Vec(
                        Vec.load(
                            v_i8x16,
                            lds,
                            [
                                k_buf_off
                                + kv_row * K_STRIDE
                                + _k_swizzle(kv_row, d_base)
                            ],
                        )
                    ).bitcast(fx.Int32)
                    blk_hi = Vec(
                        Vec.load(
                            v_i8x16,
                            lds,
                            [
                                k_buf_off
                                + kv_row * K_STRIDE
                                + _k_swizzle(kv_row, d_base + fx.Index(16))
                            ],
                        )
                    ).bitcast(fx.Int32)
                    k_pack = blk_lo.shuffle(blk_hi, list(range(8)))
                    s_acc = mfma.call(k_pack, q_packs[ks], s_acc)
                s_accs.append(s_acc)
            return s_accs

        def do_softmax(s_accs, m_running):
            # Online softmax (VALU/transcendental only): returns (m_new, corr,
            # p_pack) where p_pack is the PV B-operand (P_PACK_WORDS fp8 words).
            # Pass 1: row max read directly off the carried S accumulators -- no
            # 64-element s_raw list, so the extracted S scalars never co-exist
            # with the 64 exp2 outputs (that overlap was the spill driver).
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

            # Pass 2: exp2 -> P packed 4-at-a-time, so at most 4 exp outputs are
            # live before they collapse into one fp8 i32 word.
            def _p(nt, r):
                e = _fadd(_fmul(Vec(s_accs[nt])[r], scale_log2e), neg_scaled_m_new)
                return ArithValue(e).exp2(fastmath=fm_fast)

            myword = [
                [
                    _f32x4_to_fp8_word(
                        _p(nt, rg * 4 + 0),
                        _p(nt, rg * 4 + 1),
                        _p(nt, rg * 4 + 2),
                        _p(nt, rg * 4 + 3),
                    )
                    for rg in range_constexpr(n_groups)
                ]
                for nt in range_constexpr(N_KV_TILES)
            ]
            peerword = [
                [
                    fx.Int32(myword[nt][rg]).shuffle_xor(
                        fx.Int32(32), fx.Int32(WARP_SIZE)
                    )
                    for rg in range_constexpr(n_groups)
                ]
                for nt in range_constexpr(N_KV_TILES)
            ]
            hi_is0 = hi < fx.Index(1)
            p_words = []
            for ks in range_constexpr(PV_K_STEPS):
                for w in range_constexpr(8):
                    rg = w // 2
                    if const_expr(w % 2 == 0):
                        sel = ArithValue(hi_is0).select(
                            _raw(myword[2 * ks + 0][rg]), _raw(peerword[2 * ks + 1][rg])
                        )
                    else:
                        sel = ArithValue(hi_is0).select(
                            _raw(peerword[2 * ks + 0][rg]), _raw(myword[2 * ks + 1][rg])
                        )
                    p_words.append(fx.Int32(sel))
            return m_new, corr, Vec.from_elements(p_words, fx.Int32)

        # ---- Shared init values (computed by all 512 lanes before the split) --
        m_init = c_neg_inf
        l_init = Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)
        o_init = [
            Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)
            for _ in range_constexpr(D_TILES)
        ]
        buf_init = fx.Index(0)
        k_reg0 = global_load_kv(fx.Index(0), k_ptr)  # K reg, tile 0
        v_reg0 = global_load_kv(fx.Index(0), v_ptr)  # V reg, tile 0
        p_init = Vec.filled(P_PACK_WORDS, 0, fx.Int32)  # p_carry=0
        corr_init = fx.Float32(1.0)  # no-op rescale at i=0
        s_init = [
            Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)
            for _ in range_constexpr(N_KV_TILES)
        ]
        zero_p = Vec.filled(P_PACK_WORDS, 0, fx.Int32)

        # Shared per-iteration prologue (BOTH groups, all 512 lanes): publish
        # tile i to LDS, barrier, then prefetch tile i+1's global loads.  The
        # cooperative load/store is workgroup-wide (each lane owns half the rows),
        # so both groups must run it AND both must hit the publish barrier -- this
        # keeps the per-group barrier count identical (2/iter), the invariant that
        # makes the split safe against deadlock.
        def _iter_prologue(kv_start, buf, k_reg, v_reg):
            k_buf_off = fx.Index(LDS_K_OFF) + buf * fx.Index(LDS_K_TILE)
            v_buf_off = fx.Index(LDS_V_OFF) + buf * fx.Index(LDS_V_TILE)
            # tile i-1 buffer.  At iteration 0 there is no tile i-1: P_prev=0
            # nullifies the contribution mathematically, but the PV MFMA still
            # reads V -- and 0*NaN=NaN if that buffer is uninitialized LDS.  So at
            # i=0 point the read at the just-written current buffer (finite).
            is_first = kv_start < fx.Index(BLOCK_N)
            prev_buf = fx.Index(
                ArithValue(is_first).select(
                    buf, (buf + fx.Index(2)) % fx.Index(NUM_BUF)
                )
            )
            v_prev_off = fx.Index(LDS_V_OFF) + prev_buf * fx.Index(LDS_V_TILE)
            next_buf = (buf + fx.Index(1)) % fx.Index(NUM_BUF)
            next_kv = kv_start + fx.Index(BLOCK_N)
            # Triple-buffered: one barrier publishes tile i; the buffer written
            # here (tile i-3) was last read in iteration i-2, no hazard barrier.
            store_k_lds(k_reg, k_buf_off)
            store_v_lds_dblocked(v_reg, v_buf_off)
            gpu.barrier()
            k_next = global_load_kv(next_kv, k_ptr)
            v_next = global_load_kv(next_kv, v_ptr)
            return k_buf_off, v_prev_off, next_buf, is_first, k_next, v_next

        # Cross-wave role ping-pong, hoisted into TWO specialized loop bodies
        # (option a).  The role split lives OUTSIDE the loop: G0 and G1 each run
        # their own scf.for (built via scf_for_dispatch -- the lowering API the
        # `for ... yield` sugar compiles to; we call it directly because a
        # yield-loop cannot be nested in a runtime scf.if).  Because the two
        # loops sit in mutually-exclusive scf.if regions, each group's S tile has
        # a single live range and the allocator coalesces G0.S with G1.S (64
        # VGPR, not 64+64).  G0 carries P across iterations (deferred PV) and
        # produces/consumes S within one iteration; G1 carries S across
        # iterations (its phase-B QK feeds the next phase-A softmax) with P
        # intra-iteration.  Both bodies keep the identical 2-barriers/iter rhythm
        # (one in the shared prologue, one between phases), so all 8 waves stay
        # barrier-lockstep (s_barrier counts waves, not PCs).
        is_g0 = wave_id < fx.Index(NUM_WAVES // 2)
        loop_lo = fx.Int32(0)
        loop_step = fx.Int32(BLOCK_N)
        of0 = Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)
        of1 = Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)
        of2 = Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)
        of3 = Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)
        lf = Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32)

        if is_g0:
            g0_names = [
                "m_r",
                "l_a",
                "oo0",
                "oo1",
                "oo2",
                "oo3",
                "buf",
                "k_reg",
                "v_reg",
                "p_c",
                "corr_c",
            ]
            g0_init = [
                m_init,
                l_init,
                o_init[0],
                o_init[1],
                o_init[2],
                o_init[3],
                buf_init,
                k_reg0,
                v_reg0,
                p_init,
                corr_init,
            ]

            def g0_body(
                iv, _names, m_r, l_a, oo0, oo1, oo2, oo3, buf, k_reg, v_reg, p_c, corr_c
            ):
                kv_start = fx.Index(iv)
                (
                    k_buf_off,
                    v_prev_off,
                    next_buf,
                    _isf,
                    k_next,
                    v_next,
                ) = _iter_prologue(kv_start, buf, k_reg, v_reg)
                # PHASE A: matrix -- deferred PV(i-1) then QK(i).
                oA, lA = apply_pv([oo0, oo1, oo2, oo3], l_a, p_c, corr_c, v_prev_off)
                sA = do_qk(k_buf_off)
                gpu.barrier()
                # PHASE B: softmax(i) -> P(i) carried to next iter's PV.
                m_new, corr_new, p_new = do_softmax([sA[0], sA[1], sA[2], sA[3]], m_r)
                return [
                    m_new,
                    lA,
                    oA[0],
                    oA[1],
                    oA[2],
                    oA[3],
                    next_buf,
                    k_next,
                    v_next,
                    p_new,
                    corr_new,
                ]

            res = scf_for_dispatch(  # noqa: F821
                loop_lo,
                seq_len,
                loop_step,
                g0_body,
                result_names=g0_names,
                result_values=g0_init,
            )
            # G0 epilogue: still owes PV(N-1) (softmax(N-1) ran in last phase B).
            m_e = res[0]
            l_e = res[1]
            oe0 = res[2]
            oe1 = res[3]
            oe2 = res[4]
            oe3 = res[5]
            buf_final = res[6]
            p_e = res[9]
            corr_e = res[10]
            last_buf = (buf_final + fx.Index(NUM_BUF - 1)) % fx.Index(NUM_BUF)
            v_last_off = fx.Index(LDS_V_OFF) + last_buf * fx.Index(LDS_V_TILE)
            o_fin, l_vec_g = apply_pv(
                [oe0, oe1, oe2, oe3], l_e, p_e, corr_e, v_last_off
            )
            of0 = o_fin[0]
            of1 = o_fin[1]
            of2 = o_fin[2]
            of3 = o_fin[3]
            lf = l_vec_g
        else:
            g1_names = [
                "m_r",
                "l_a",
                "oo0",
                "oo1",
                "oo2",
                "oo3",
                "buf",
                "k_reg",
                "v_reg",
                "ss0",
                "ss1",
                "ss2",
                "ss3",
            ]
            g1_init = [
                m_init,
                l_init,
                o_init[0],
                o_init[1],
                o_init[2],
                o_init[3],
                buf_init,
                k_reg0,
                v_reg0,
                s_init[0],
                s_init[1],
                s_init[2],
                s_init[3],
            ]

            def g1_body(
                iv,
                _names,
                m_r,
                l_a,
                oo0,
                oo1,
                oo2,
                oo3,
                buf,
                k_reg,
                v_reg,
                ss0,
                ss1,
                ss2,
                ss3,
            ):
                kv_start = fx.Index(iv)
                (
                    k_buf_off,
                    v_prev_off,
                    next_buf,
                    is_first,
                    k_next,
                    v_next,
                ) = _iter_prologue(kv_start, buf, k_reg, v_reg)
                # PHASE A: softmax(i-1) on carried S (iter 0 has none -> no-op).
                m_sm, corr_sm, p_sm = do_softmax([ss0, ss1, ss2, ss3], m_r)
                mA = ArithValue(is_first).select(m_r, m_sm)
                corrA = ArithValue(is_first).select(fx.Float32(1.0), corr_sm)
                pA = ArithValue(is_first).select(zero_p.ir_value(), _raw(p_sm))
                gpu.barrier()
                # PHASE B: matrix -- deferred PV(i-1) then QK(i) -> S(i) carried.
                oB, lB = apply_pv([oo0, oo1, oo2, oo3], l_a, pA, corrA, v_prev_off)
                sB = do_qk(k_buf_off)
                return [
                    mA,
                    lB,
                    oB[0],
                    oB[1],
                    oB[2],
                    oB[3],
                    next_buf,
                    k_next,
                    v_next,
                    sB[0],
                    sB[1],
                    sB[2],
                    sB[3],
                ]

            res = scf_for_dispatch(  # noqa: F821
                loop_lo,
                seq_len,
                loop_step,
                g1_body,
                result_names=g1_names,
                result_values=g1_init,
            )
            # G1 epilogue: owes BOTH softmax(N-1) (carried S) and PV(N-1).
            m_e = res[0]
            l_e = res[1]
            oe0 = res[2]
            oe1 = res[3]
            oe2 = res[4]
            oe3 = res[5]
            buf_final = res[6]
            sce0 = res[9]
            sce1 = res[10]
            sce2 = res[11]
            sce3 = res[12]
            last_buf = (buf_final + fx.Index(NUM_BUF - 1)) % fx.Index(NUM_BUF)
            v_last_off = fx.Index(LDS_V_OFF) + last_buf * fx.Index(LDS_V_TILE)
            _m_e, corrf, pf = do_softmax([sce0, sce1, sce2, sce3], m_e)
            o_fin, l_vec_g = apply_pv([oe0, oe1, oe2, oe3], l_e, pf, corrf, v_last_off)
            of0 = o_fin[0]
            of1 = o_fin[1]
            of2 = o_fin[2]
            of3 = o_fin[3]
            lf = l_vec_g

        o_finals = [of0, of1, of2, of3]
        l_vec = lf

        # l is the ones-column MFMA accumulator (Step 5): a vec<16xf32> whose
        # every value == L[q=lo] (the row sum is replicated over the dummy d
        # rows), so any element is the per-q normalizer.
        l_final = Vec(l_vec)[0]

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
