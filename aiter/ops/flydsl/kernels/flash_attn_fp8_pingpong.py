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
- The softmax output ``P`` and the value tile ``V`` are round-tripped
  through LDS so the QK accumulator C-layout can be re-read in the MFMA
  A/B fragment layouts for the PV stage (no register shuffle tricks).

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
    BLOCK_N = 64
    HEAD_DIM = head_dim
    NUM_HEADS = num_heads
    NUM_WAVES = 8
    BLOCK_SIZE = NUM_WAVES * WARP_SIZE  # 512
    ROWS_PER_WAVE = BLOCK_M // NUM_WAVES  # 32
    STRIDE_TOKEN = NUM_HEADS * HEAD_DIM

    K_STEPS = HEAD_DIM // MFMA_K  # 2 head-dim chunks of 64
    N_KV_TILES = BLOCK_N // MFMA_N  # 2 kv sub-tiles of 32
    D_TILES = HEAD_DIM // MFMA_M  # 4 output d sub-tiles of 32

    if softmax_scale is None:
        softmax_scale = 1.0 / host_math.sqrt(head_dim)

    # ---- LDS layout (fp8 element type) ----
    # K tile : [BLOCK_N kv][HEAD_DIM d]          row-major
    # V tile : [HEAD_DIM d][BLOCK_N kv] (fp8)    *transposed* (Step 1)
    # P tile : [BLOCK_M q][BLOCK_N kv] (fp8)     row-major, per workgroup
    #
    # V is laid out transposed so the GEMM2 A-operand fragment
    #   A_frag[L][v] = V[kv = hi*32 + v, d = lo + dt*32]
    #             = Vt[d = lo + dt*32][kv = hi*32 + v]
    # is a single contiguous 32-byte (v_i8x32) load along kv, replacing the
    # old 32 single-byte strided gathers.
    K_STRIDE = HEAD_DIM
    V_STRIDE = BLOCK_N  # transposed: stride over kv per d-row
    P_STRIDE = BLOCK_N
    # Step 2: double-buffer K and V (ping-pong by KV-tile parity) so the
    # global load for tile i+1 overlaps the MFMA compute of tile i.
    NUM_BUF = 2
    LDS_K_TILE = BLOCK_N * K_STRIDE
    LDS_V_TILE = HEAD_DIM * V_STRIDE
    LDS_K_SIZE = NUM_BUF * LDS_K_TILE
    LDS_V_SIZE = NUM_BUF * LDS_V_TILE
    LDS_P_SIZE = BLOCK_M * P_STRIDE
    LDS_K_OFF = 0
    LDS_V_OFF = LDS_K_OFF + LDS_K_SIZE
    LDS_P_OFF = LDS_V_OFF + LDS_V_SIZE
    LDS_TOTAL = LDS_P_OFF + LDS_P_SIZE

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
        load_row = tid // THREADS_PER_ROW  # 0..63
        load_col = (tid % THREADS_PER_ROW) * VEC  # 0,16,...,112

        def _load_global_i8x16(ptr, base_idx):
            gep = fx.buffer_ops.get_element_ptr(
                ptr, fx.Int64(base_idx), elem_type=i8_type
            )
            return _pointer_load(v_i8x16, gep)

        zero_vec16 = Vec.filled(VEC, 0, i8_dtype)

        def global_load_kv(kv_start, src_ptr):
            # Issue the (latency-bound) global load for this lane's row chunk.
            # Returns a register vec<16xi8>; OOB rows zeroed.
            row_idx = kv_start + load_row
            in_bounds = row_idx < seq_len_v
            safe_row = fx.Index(ArithValue(in_bounds).select(row_idx, fx.Index(0)))
            g_idx = global_idx(safe_row, load_col)
            vec = _load_global_i8x16(src_ptr, g_idx)
            return ArithValue(in_bounds).select(vec, zero_vec16.ir_value())

        def store_k_lds(vec, lds_off):
            lds_idx = fx.Index(lds_off) + load_row * K_STRIDE + load_col
            Vec(vec).store(lds, [lds_idx])

        def store_v_lds_transposed(vec, lds_off):
            # Scatter the 16 contiguous d-values into transposed Vt[d][kv].
            vv = Vec(vec)
            for e in range_constexpr(VEC):
                d_row = load_col + fx.Index(e)
                vt_idx = fx.Index(lds_off) + d_row * V_STRIDE + load_row
                Vec.from_elements([vv[e]], i8_dtype).store(lds, [vt_idx])

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
        init_args = [c_neg_inf, c_zero_f]
        for _ in range_constexpr(D_TILES):
            init_args.append(Vec.filled(C_F32_PER_LANE, 0.0, fx.Float32))
        # Double-buffer carried state: buffer parity + prefetched K/V register
        # vectors for the *current* tile.  Prefetch tile 0 before the loop.
        # The global loads (the latency-bound part) are issued one iteration
        # ahead so they overlap the previous tile's MFMA compute; the SSA
        # dependency from the LDS store auto-inserts the consume waitcnt.
        init_args.append(fx.Index(0))  # buf parity
        init_args.append(global_load_kv(fx.Index(0), k_ptr))  # K reg, tile 0
        init_args.append(global_load_kv(fx.Index(0), v_ptr))  # V reg, tile 0

        loop_results = init_args
        for kv_start, iter_args in range(0, seq_len_v, BLOCK_N, init=init_args):
            m_running = iter_args[0]
            l_running = iter_args[1]
            o_accs = [iter_args[2 + i] for i in range_constexpr(D_TILES)]
            buf = iter_args[2 + D_TILES]
            k_reg = iter_args[3 + D_TILES]
            v_reg = iter_args[4 + D_TILES]

            k_buf_off = fx.Index(LDS_K_OFF) + buf * fx.Index(LDS_K_TILE)
            v_buf_off = fx.Index(LDS_V_OFF) + buf * fx.Index(LDS_V_TILE)
            next_buf = fx.Index(1) - buf
            next_kv = kv_start + fx.Index(BLOCK_N)

            # ---- Commit the prefetched current tile into its LDS buffer ----
            store_k_lds(k_reg, k_buf_off)
            store_v_lds_transposed(v_reg, v_buf_off)
            gpu.barrier()

            # ---- Prefetch the NEXT tile's global loads (latency hides under
            #      this tile's MFMA compute below).  Carried to next iter. ----
            k_next = global_load_kv(next_kv, k_ptr)
            v_next = global_load_kv(next_kv, v_ptr)

            # ===============================================================
            # GEMM1: S[kv,q] = K @ Q^T  for each of the N_KV_TILES kv subtiles
            #   A = K   : A_frag[L][v] = K[kv = lo (+ nt*32), d = ks*64+hi*32+v]
            #   B = Q   : q_packs[ks]  (preloaded)
            # ===============================================================
            s_accs = []
            for nt in range_constexpr(N_KV_TILES):
                s_acc = mfma.zero_value
                for ks in range_constexpr(K_STEPS):
                    # K A-operand: kv-row = lo + nt*32, d = ks*64 + hi*32 + v
                    kv_row = lo + fx.Index(nt * 32)
                    d_base = fx.Index(ks * MFMA_K) + hi * 32
                    lds_idx = k_buf_off + kv_row * K_STRIDE + d_base
                    k_pack = Vec.load(v_i8x32, lds, [lds_idx])
                    k_pack = Vec(k_pack).bitcast(fx.Int32)
                    s_acc = mfma.call(k_pack, q_packs[ks], s_acc)
                s_accs.append(s_acc)

            # ===============================================================
            # Online softmax over the 64 kv positions of this tile.
            # C-layout: value index v -> kv-row hi*4+(v%4)+8*(v//4); lane lo
            # -> q.  s_accs[nt] covers kv in [nt*32, nt*32+32).  Reduce over
            # the 16 values + the hi half (xor-32) -> per-q (lane) row max.
            # ===============================================================
            s_raw = [[None] * C_F32_PER_LANE for _ in range_constexpr(N_KV_TILES)]
            for nt in range_constexpr(N_KV_TILES):
                for r in range_constexpr(C_F32_PER_LANE):
                    s_raw[nt][r] = Vec(s_accs[nt])[r]

            local_max = s_raw[0][0]
            for nt in range_constexpr(N_KV_TILES):
                for r in range_constexpr(C_F32_PER_LANE):
                    if const_expr(nt == 0 and r == 0):
                        continue
                    local_max = _fmax(local_max, s_raw[nt][r])
            # Combine the hi half (kv rows differ by hi) across the 32-lane split.
            peer_max = fx.Float32(local_max).shuffle_xor(
                fx.Int32(32), fx.Int32(WARP_SIZE)
            )
            row_max_int = _fmax(local_max, peer_max)

            # m / corr in *real* score domain handled via scale_log2e.
            m_new = _fmax(m_running, row_max_int)
            corr = ArithValue(_fmul(_fsub(m_running, m_new), scale_log2e)).exp2(
                fastmath=fm_fast
            )

            scaled_m_new = _fmul(scale_log2e, m_new)
            neg_scaled_m_new = _fsub(c_zero_f, scaled_m_new)

            # P = exp2(scale_log2e * S_int - scale_log2e * m_new), packed fp8.
            p_vals = [[None] * C_F32_PER_LANE for _ in range_constexpr(N_KV_TILES)]
            local_sum = c_zero_f
            for nt in range_constexpr(N_KV_TILES):
                for r in range_constexpr(C_F32_PER_LANE):
                    e = _fadd(_fmul(s_raw[nt][r], scale_log2e), neg_scaled_m_new)
                    p = ArithValue(e).exp2(fastmath=fm_fast)
                    p_vals[nt][r] = p
                    local_sum = _fadd(local_sum, p)
            peer_sum = fx.Float32(local_sum).shuffle_xor(
                fx.Int32(32), fx.Int32(WARP_SIZE)
            )
            tile_sum = _fadd(local_sum, peer_sum)
            l_new = _fadd(_fmul(corr, l_running), tile_sum)

            # ---- Write P to LDS in [q, kv] layout (fp8 stored as i8) ----
            # C-layout coord: kv = hi*4 + (r%4) + 8*(r//4) (+ nt*32); q = lo.
            # P_lds is shared by all 8 waves, so the row index must be the
            # block-local q = wave_q_offset + lo (NOT just lo, which would
            # make every wave collide on the same 32 rows).
            p_q_local = wave_q_offset + lo
            for nt in range_constexpr(N_KV_TILES):
                for r in range_constexpr(C_F32_PER_LANE):
                    kv_row = hi * 4 + (r % 4) + 8 * (r // 4) + nt * 32
                    p_i8 = _f32_to_fp8_byte(p_vals[nt][r])
                    p_idx = (
                        fx.Index(LDS_P_OFF) + p_q_local * P_STRIDE + fx.Index(kv_row)
                    )
                    Vec.from_elements([fx.Int8(p_i8)], i8_dtype).store(lds, [p_idx])
            gpu.barrier()

            # ===============================================================
            # GEMM2: O[d,q] += V^T @ P
            #   A = V^T : A_frag[L][v] = V[kv = hi*32+v, d = lo + dt*32]
            #   B = P   : B_frag[L][v] = P[kv = hi*32+v, q = lo]
            # both read along kv = hi*32+v (contiguous 32 kv).
            # ===============================================================
            # B (P) packs: read 32 contiguous kv for q = lo from LDS P[q, kv].
            # P_lds is [q, kv] row-major so P[lo, hi*32 : hi*32+32] is
            # contiguous.
            p_idx_b = fx.Index(LDS_P_OFF) + p_q_local * P_STRIDE + hi * 32
            p_pack = Vec.load(v_i8x32, lds, [p_idx_b])
            p_pack = Vec(p_pack).bitcast(fx.Int32)

            # Rescale O accumulators by corr before adding this tile.
            corr_vec = Vec.from_elements([corr], fx.Float32).broadcast_to(
                C_F32_PER_LANE
            )
            for dt in range_constexpr(D_TILES):
                o_accs[dt] = _fmul(Vec(o_accs[dt]), corr_vec)

            for dt in range_constexpr(D_TILES):
                # V A-operand: kv = hi*32+v, d = lo + dt*32.  V_lds is now
                # transposed Vt[d][kv] so the 32 kv values for a fixed
                # d = lo + dt*32 are contiguous -> single v_i8x32 load.
                d_row = lo + fx.Index(dt * 32)
                vlds = v_buf_off + d_row * V_STRIDE + hi * 32
                v_pack = Vec.load(v_i8x32, lds, [vlds])
                v_pack = Vec(v_pack).bitcast(fx.Int32)
                o_accs[dt] = mfma.call(v_pack, p_pack, o_accs[dt])

            m_running = m_new
            l_running = l_new

            loop_results = yield (
                [m_running, l_running] + o_accs + [next_buf, k_next, v_next]
            )

        # ---- Epilogue: O = (O / l) * v_descale ; store bf16 ----
        l_final = loop_results[1]
        o_finals = [loop_results[2 + dt] for dt in range_constexpr(D_TILES)]

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
