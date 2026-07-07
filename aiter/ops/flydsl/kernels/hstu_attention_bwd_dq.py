"""hstu_attention_bwd_dq - FlyDSL kernel (causal-only; computes dQ)

Companion to hstu_attention_bwd.py. That kernel is KV-owned and produces dV/dK
(both reduce over the query index). dQ instead reduces over the **key** index:

    dQ[q, hc] = alpha * sum_kv dS[q, kv] * K[kv, hc]
    dS[q, kv] = mask .* (1/N) * silu'(alpha*S) * (dO * V^T)[q, kv],  S = alpha*Q*K^T

so it wants the opposite orientation: each program **owns a query tile** (BLOCK_M q
rows) and **streams KV tiles** (BLOCK_N) -- exactly the forward's layout. dQ rows are
owned by a single program, so this is a lock-free single-writer accumulator.

Pipeline per streamed KV tile (mirrors the forward's K DMA + V register-prefetch):
  - K staged global->LDS (swizzled); V register-prefetched then published to LDS.
  - GEMM1 S^T[kv, q] = K * Q^T (A = K from LDS, B = Q resident) -- same as the forward.
  - retain the SiLU-derivative gate silu'(alpha*S) and the causal mask.
  - dA[kv, q] = V * dO^T (A = V from LDS, B = dO resident); dS = mask .* (1/N) * silu' .* dA.
  - dQ[q, hc] += dS[q, kv] * K[kv, hc] (dS frag reused as A-operand; K re-read from LDS as B),
    with alpha applied once in the epilogue.

Constraints match hstu_attention_bwd.py (causal only; {f16,bf16}; the divisibility /
arch contracts of the forward).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl
from flydsl.expr.arith import ArithValue
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from aiter.ops.flydsl.kernels.hstu_attention_bwd import (
    WARP_SIZE,
    NUM_GRID_GROUPS,
    MFMA_M,
    MFMA_N,
    MFMA_K,
    MFMA_LANE_K,
    MFMA_ELEMS_PER_LANE,
    _LOG2E,
    _arch_dma_params,
    _dtype_to_elem_type,
    _waitcnt_vm_n,
    lds_cap_bytes,
    validate_hstu_attention_bwd,
)


# Reuse the exact same validation contract as the dV/dK kernel.
validate_hstu_attention_bwd_dq = validate_hstu_attention_bwd


@functools.lru_cache(maxsize=16384)
def build_hstu_attention_bwd_dq(
    num_heads: int,
    head_dim: int,
    hidden_dim: int,
    batch: int,
    causal: bool,
    max_attn_len: int,
    contextual_seq_len: int,
    has_targets: bool,
    alpha: float,
    dtype_str: str,
    max_seq_len: int,
    *,
    block_m: int = 64,
    block_n: int = 16,
    num_waves: int = 2,
    waves_per_eu: int = 0,
):
    validate_hstu_attention_bwd_dq(
        num_heads,
        head_dim,
        hidden_dim,
        batch,
        causal,
        max_attn_len,
        contextual_seq_len,
        has_targets,
        alpha,
        dtype_str,
        max_seq_len,
        block_m=block_m,
        block_n=block_n,
        num_waves=num_waves,
        waves_per_eu=waves_per_eu,
    )

    # BLOCK_M = owned query tile (dQ output rows). BLOCK_N = streamed KV tile.
    BLOCK_M = block_m
    BLOCK_N = block_n
    NUM_WAVES = num_waves
    BLOCK_THREADS = NUM_WAVES * WARP_SIZE
    ROWS_PER_WAVE = BLOCK_M // NUM_WAVES
    Q_SUBTILES = ROWS_PER_WAVE // MFMA_M  # owned query sub-tiles per wave
    KV_SUBTILES = BLOCK_N // MFMA_N  # streamed KV sub-tiles
    WAVES_PER_EU = waves_per_eu

    DMA_BYTES, DMA_ELEMS, K_SWZ_ROWS, K_SWZ_SHIFT = _arch_dma_params()

    elem_dtype = _dtype_to_elem_type(dtype_str)
    is_bf16 = dtype_str == "bf16"

    K_STEPS = head_dim // MFMA_K  # real contraction steps (Q side)
    HEAD_DIM_K = ((head_dim + 63) // 64) * 64
    K_STEPS_K = HEAD_DIM_K // MFMA_K  # padded steps (K side)
    DK_STEPS = hidden_dim // MFMA_K  # dA contraction steps (over hidden d)
    HC_CHUNKS = head_dim // MFMA_M  # dQ accumulator chunks (over head_dim)

    num_q_tiles = (max_seq_len + BLOCK_M - 1) // BLOCK_M
    hz_per_group = (batch * num_heads) // NUM_GRID_GROUPS

    stride_qk_n = num_heads * head_dim

    K_STRIDE = HEAD_DIM_K
    k_lds_bytes = BLOCK_N * K_STRIDE * 2
    V_STRIDE = hidden_dim
    v_lds_bytes = BLOCK_N * V_STRIDE * 2

    k_tile_elems = BLOCK_N * K_STRIDE
    elems_per_dma_pass = BLOCK_THREADS * DMA_ELEMS
    assert k_tile_elems % elems_per_dma_pass == 0
    NUM_DMA_K = k_tile_elems // elems_per_dma_pass
    PAIRS_PER_ROW_K = K_STRIDE // DMA_ELEMS

    v_tile_elems = BLOCK_N * hidden_dim
    assert v_tile_elems % elems_per_dma_pass == 0

    VEC_V = 8 if (hidden_dim % 8 == 0 and (BLOCK_N * hidden_dim) % (BLOCK_THREADS * 8) == 0) else DMA_ELEMS
    THREADS_PER_ROW_V = hidden_dim // VEC_V
    assert BLOCK_THREADS % THREADS_PER_ROW_V == 0
    ROWS_PER_BATCH_V = BLOCK_THREADS // THREADS_PER_ROW_V
    assert BLOCK_N % ROWS_PER_BATCH_V == 0 or ROWS_PER_BATCH_V > BLOCK_N
    NUM_BATCHES_V = max(1, BLOCK_N // ROWS_PER_BATCH_V)
    V_NEEDS_GUARD = ROWS_PER_BATCH_V > BLOCK_N

    allocator = SmemAllocator(None, global_sym_name="hstu_attention_bwd_dq_smem")
    k_lds_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = k_lds_offset + k_lds_bytes
    v_lds_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = v_lds_offset + v_lds_bytes

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def hstu_attention_bwd_dq(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        do: fx.Tensor,
        seq_offsets: fx.Tensor,
        num_targets: fx.Tensor,
        dq: fx.Tensor,
    ) -> None:
        elem_type = elem_dtype.ir_type
        compute_type = fx.Float32.ir_type
        v4f32_type = Vec.make_type(MFMA_ELEMS_PER_LANE, fx.Float32)
        mfma_pack_type = Vec.make_type(MFMA_LANE_K, elem_dtype)
        c_zero_mfma_pack = Vec.filled(MFMA_LANE_K, 0.0, elem_dtype).ir_value()

        _mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(MFMA_M, MFMA_M, MFMA_K, elem_dtype))

        def mfma_acc(a_pack, b_pack, c):
            return fly.mma_atom_call_ssa([v4f32_type], _mma_atom, a_pack, b_pack, c)

        tid = fx.Int32(gpu.thread_idx.x)
        wave_id = tid // fx.Int32(WARP_SIZE)
        lane = tid % fx.Int32(WARP_SIZE)
        lane_mod_16 = lane % fx.Int32(MFMA_N)
        lane_div_16 = lane // fx.Int32(MFMA_N)

        # ---- Group-major grid decode -> (batch_idx, head_idx, q_tile_idx) ----
        block_id = fx.Int32(gpu.block_idx.x)
        grid_group = block_id % fx.Int32(NUM_GRID_GROUPS)
        pos_in_group = block_id // fx.Int32(NUM_GRID_GROUPS)
        local_hz_idx = pos_in_group // fx.Int32(num_q_tiles)
        q_tile_idx = pos_in_group % fx.Int32(num_q_tiles)
        hz_idx = grid_group * fx.Int32(hz_per_group) + local_hz_idx
        batch_idx = hz_idx // fx.Int32(num_heads)
        head_idx = hz_idx % fx.Int32(num_heads)

        seq_start = fx.Int32(seq_offsets[batch_idx])
        seq_len = fx.Int32(seq_offsets[batch_idx + fx.Int32(1)]) - seq_start

        def grouped_loader(t, dim, g):
            in_row = fx.make_layout((dim // g, g), (g, 1))

            def load(row_i64, head_val, colgrp):
                sub = t[row_i64, head_val, None]
                return fx.make_view(fx.get_iter(sub), in_row)[colgrp, None].load()

            return load

        q_load = grouped_loader(q, head_dim, MFMA_LANE_K)  # resident Q (B-operand for S)
        do_load = grouped_loader(do, hidden_dim, MFMA_LANE_K)  # resident dO (B-operand for dA)

        q_head_offset = head_idx * fx.Int32(head_dim)

        # ---- Streamed K DMA buffer resource ----
        k_base_byte_offset = (fx.Int64(seq_start) * fx.Int64(stride_qk_n) + fx.Int64(q_head_offset)) * fx.Int64(2)
        k_rsrc = buffer_ops.create_buffer_resource(k, max_size=True, base_byte_offset=k_base_byte_offset)

        lds_base = allocator.get_base()
        k_smem = SmemPtr(lds_base, k_lds_offset, elem_type, shape=(BLOCK_N, K_STRIDE))
        v_smem = SmemPtr(lds_base, v_lds_offset, elem_type, shape=(BLOCK_N, V_STRIDE))
        k_lds_byte_base = buffer_ops.extract_base_index(k_smem.get(), address_space=3)

        def k_swz_col(tile_row, col):
            return col ^ ((tile_row & fx.Int32(K_SWZ_ROWS - 1)) << fx.Int32(K_SWZ_SHIFT))

        q_wave_base = q_tile_idx * fx.Int32(BLOCK_M) + wave_id * fx.Int32(ROWS_PER_WAVE)

        # ---- Owned Q rows / bounds per query sub-tile ----
        q_rows = []
        q_in_bounds = []
        for qg in range_constexpr(Q_SUBTILES):
            local = q_wave_base + fx.Int32(qg * MFMA_M) + lane_mod_16
            q_rows.append(local)
            q_in_bounds.append(local < seq_len)

        # ---- Resident Q B-operand packs (GEMM1 S^T = K*Q^T) ----
        q_packs = []  # q_packs[ks][qg]
        for ks in range_constexpr(K_STEPS):
            q_col = fx.Int32(ks * MFMA_K) + lane_div_16 * fx.Int32(MFMA_LANE_K)
            per_qg = []
            for qg in range_constexpr(Q_SUBTILES):
                safe = q_in_bounds[qg].select(seq_start + q_rows[qg], seq_start)
                raw = q_load(fx.Int64(safe), head_idx, q_col // fx.Int32(MFMA_LANE_K)).ir_value()
                per_qg.append(q_in_bounds[qg].select(raw, c_zero_mfma_pack))
            q_packs.append(per_qg)

        # ---- Resident dO B-operand packs (dA = V*dO^T) ----
        # b_pack[i] = dO[q = lane_mod_16, d = ks*16 + lane_div_16*4 + i]; contraction over d.
        do_packs = []  # do_packs[ks][qg]
        for ks in range_constexpr(DK_STEPS):
            d_col = fx.Int32(ks * MFMA_K) + lane_div_16 * fx.Int32(MFMA_LANE_K)
            per_qg = []
            for qg in range_constexpr(Q_SUBTILES):
                safe = q_in_bounds[qg].select(seq_start + q_rows[qg], seq_start)
                raw = do_load(fx.Int64(safe), head_idx, d_col // fx.Int32(MFMA_LANE_K)).ir_value()
                per_qg.append(q_in_bounds[qg].select(raw, c_zero_mfma_pack))
            do_packs.append(per_qg)

        def _fadd(a, b):
            return arith.addf(_raw(a), _raw(b), fastmath=arith.FastMathFlags.fast)

        def _fmul(a, b):
            return arith.mulf(_raw(a), _raw(b), fastmath=arith.FastMathFlags.fast)

        c_alpha = fx.Float32(alpha)
        c_inv_n = fx.Float32(1.0 / max_seq_len)
        c_neg_log2e = fx.Float32(-_LOG2E)
        c_one_f = fx.Float32(1.0)
        c_neg_one_f = fx.Float32(-1.0)
        c_zero_f = fx.Float32(0.0)

        def silu_grad_batch(s_list):
            """silu'(alpha*s) = sigma*(1 + alpha*s*(1-sigma)); same fast sigmoid as forward."""
            sc = [_fmul(s, c_alpha) for s in s_list]
            tt = [_fmul(s, c_neg_log2e) for s in sc]
            emu = [llvm.call_intrinsic(compute_type, "llvm.amdgcn.exp2.f32", [t], [], []) for t in tt]
            den = [_fadd(c_one_f, e) for e in emu]
            sig = [llvm.call_intrinsic(compute_type, "llvm.amdgcn.rcp.f32", [d], [], []) for d in den]
            return [
                _fmul(sig[i], _fadd(c_one_f, _fmul(sc[i], _fadd(c_one_f, _fmul(c_neg_one_f, sig[i])))))
                for i in range(len(s_list))
            ]

        def pack_frag(vals):
            if is_bf16:
                c16 = fx.Int32(16)
                cmask = fx.Int32(0xFFFF0000)

                def bf16_pair(lo_f32, hi_f32):
                    lo_i32 = fx.Int32(ArithValue(lo_f32).bitcast(fx.Int32.ir_type))
                    hi_i32 = fx.Int32(ArithValue(hi_f32).bitcast(fx.Int32.ir_type))
                    return (hi_i32 & cmask) | lo_i32.shrui(c16)

                pairs = [bf16_pair(vals[0], vals[1]), bf16_pair(vals[2], vals[3])]
                return Vec.from_elements(pairs, fx.Int32).bitcast(elem_dtype).ir_value()
            elems = [fx.Float32(v).to(elem_dtype) for v in vals]
            return Vec.from_elements(elems, elem_dtype).ir_value()

        q_row_ids = q_rows  # causal-only: id == raw position

        # ---- Streamed KV range: causal upper bound (kv <= q) ----
        q_start = q_tile_idx * fx.Int32(BLOCK_M)
        q_end = q_start + fx.Int32(BLOCK_M)
        active = q_start < seq_len
        clamped = (q_end < seq_len).select(q_end, seq_len)
        kv_upper = active.select(clamped, fx.Int32(0))
        n_tiles = (kv_upper + fx.Int32(BLOCK_N - 1)) // fx.Int32(BLOCK_N)

        N_ACC = HC_CHUNKS * Q_SUBTILES
        c_zero_v4f32 = Vec.filled(MFMA_ELEMS_PER_LANE, 0.0, fx.Float32).ir_value()

        # ---- Streamed K DMA: global -> LDS (dword, swizzled) ----
        dma_size = fx.Int32(DMA_BYTES)
        dma_soff = fx.Int32(0)
        dma_off = fx.Int32(0)
        dma_aux = fx.Int32(1)
        c_dma_elems = fx.Int32(DMA_ELEMS)
        c_pairs_per_row_k = fx.Int32(PAIRS_PER_ROW_K)

        wave_lds_base_k = k_lds_byte_base + fx.Index(wave_id) * fx.Index(WARP_SIZE * DMA_BYTES)
        wave_lds_lane0_k = rocdl.readfirstlane(fx.Int64.ir_type, fx.Int64(wave_lds_base_k))
        k_dma_rows = []
        k_dma_gcols = []
        for d in range_constexpr(NUM_DMA_K):
            pair = tid + fx.Int32(d * BLOCK_THREADS)
            row = pair // c_pairs_per_row_k
            col_pair = pair % c_pairs_per_row_k
            col = col_pair * c_dma_elems
            k_dma_rows.append(row)
            k_dma_gcols.append(k_swz_col(row, col))

        c_stride_qk_n = fx.Int32(stride_qk_n)

        def async_load_k(kv_start):
            for d in range_constexpr(NUM_DMA_K):
                row = k_dma_rows[d]
                in_bounds = (kv_start + row) < seq_len
                local_tok = in_bounds.select(kv_start + row, fx.Int32(0))
                g_elem = local_tok * c_stride_qk_n + k_dma_gcols[d]
                voffset = g_elem * fx.Int32(2)
                lds_ptr = buffer_ops.create_llvm_ptr(wave_lds_lane0_k + fx.Int64(d * BLOCK_THREADS * DMA_BYTES), address_space=3)
                rocdl.raw_ptr_buffer_load_lds(k_rsrc, lds_ptr, dma_size, voffset, dma_soff, dma_off, dma_aux)

        # ---- Streamed V register prefetch -> LDS ----
        v_load = grouped_loader(v, hidden_dim, VEC_V)
        v_load_row_in_batch = tid // fx.Int32(THREADS_PER_ROW_V)
        v_load_lane_in_row = tid % fx.Int32(THREADS_PER_ROW_V)
        v_load_col = v_load_lane_in_row * fx.Int32(VEC_V)

        def async_load_v_regs(kv_start):
            vecs = []
            for b in range_constexpr(NUM_BATCHES_V):
                row = v_load_row_in_batch + fx.Int32(b * ROWS_PER_BATCH_V)
                tok = kv_start + row
                in_bounds = tok < seq_len
                if V_NEEDS_GUARD:
                    in_bounds = in_bounds & (row < fx.Int32(BLOCK_N))
                safe_tok = in_bounds.select(seq_start + tok, seq_start)
                raw = v_load(fx.Int64(safe_tok), head_idx, v_load_col // fx.Int32(VEC_V)).ir_value()
                vecs.append(in_bounds.select(raw, Vec.filled(VEC_V, 0.0, elem_dtype).ir_value()))
            return vecs

        def store_v_regs_to_lds(vecs):
            for b in range_constexpr(NUM_BATCHES_V):
                row = v_load_row_in_batch + fx.Int32(b * ROWS_PER_BATCH_V)
                Vec.store(Vec(vecs[b]), v_smem.get(), [fx.Index(row), fx.Index(v_load_col)])

        # ==== GEMM1: K(streamed)*Q^T(owned) -> S^T[kv, q]; retain silu' gate + mask ====
        def read_k_a_packs(ng):
            """LDS-read K A-operand packs for KV sub-tile ng (GEMM1)."""
            k_row = fx.Int32(ng * MFMA_M) + lane_mod_16
            packs = []
            for ks in range_constexpr(K_STEPS_K):
                k_col = fx.Int32(ks * MFMA_K) + lane_div_16 * fx.Int32(MFMA_LANE_K)
                packs.append(Vec.load(mfma_pack_type, k_smem.get(), [fx.Index(k_row), fx.Index(k_swz_col(k_row, k_col))]))
            return packs

        def compute_gate_tile(kv_start, k_packs_by_ng):
            """GEMM1 S^T[kv,q]; returns g_meta[ng][qg] = (grad_vals[4], keep[4]).

            C fragment C[m=kv, n=q]: value[i] -> (kv = ng*16 + lane_div_16*4 + i,
            q = qg*16 + lane_mod_16). Mask keeps (q >= kv) or diagonal.
            """
            g_meta = [[None for _ in range_constexpr(Q_SUBTILES)] for _ in range_constexpr(KV_SUBTILES)]
            for ng in range_constexpr(KV_SUBTILES):
                k_packs = [Vec(k_packs_by_ng[ng][ks]) for ks in range_constexpr(K_STEPS_K)]
                kv_base = kv_start + fx.Int32(ng * MFMA_M) + lane_div_16 * fx.Int32(MFMA_LANE_K)
                kv_raw = [kv_base + fx.Int32(i) for i in range_constexpr(MFMA_ELEMS_PER_LANE)]
                kv_in_seq = [kv_raw[i] < seq_len for i in range_constexpr(MFMA_ELEMS_PER_LANE)]
                for qg in range_constexpr(Q_SUBTILES):
                    cur = Vec.filled(MFMA_ELEMS_PER_LANE, 0.0, fx.Float32).ir_value()
                    for ks in range_constexpr(K_STEPS_K):
                        q_op = q_packs[ks][qg] if ks < K_STEPS else c_zero_mfma_pack
                        cur = mfma_acc(k_packs[ks].ir_value(), q_op, cur)
                    s_vals = [Vec(cur)[i] for i in range_constexpr(MFMA_ELEMS_PER_LANE)]

                    def keep_col(i):
                        """causal mask for (owned q = q_row_ids[qg], streamed kv = kv_raw[i])."""
                        dist = q_row_ids[qg] - kv_raw[i]
                        keep = (q_rows[qg] == kv_raw[i]) | (dist > fx.Int32(0))
                        keep = keep & kv_in_seq[i] & q_in_bounds[qg]
                        return keep

                    keep = [keep_col(i) for i in range_constexpr(MFMA_ELEMS_PER_LANE)]
                    grad_vals = silu_grad_batch(s_vals)
                    g_meta[ng][qg] = (grad_vals, keep)
            return g_meta

        # ==== dA = V*dO^T; dS = keep .* (1/N) .* silu' .* dA -> packed dS frag ====
        def read_v_a_packs(ng):
            """LDS-read V A-operand packs for KV sub-tile ng (dA GEMM).
            a_pack[i] = V[kv = ng*16 + lane_mod_16, d = ks*16 + lane_div_16*4 + i]."""
            v_row = fx.Int32(ng * MFMA_M) + lane_mod_16
            packs = []
            for ks in range_constexpr(DK_STEPS):
                d_col = fx.Int32(ks * MFMA_K) + lane_div_16 * fx.Int32(MFMA_LANE_K)
                packs.append(Vec.load(mfma_pack_type, v_smem.get(), [fx.Index(v_row), fx.Index(d_col)]))
            return packs

        def compute_ds_packs(g_meta):
            ds_packs = [[None for _ in range_constexpr(Q_SUBTILES)] for _ in range_constexpr(KV_SUBTILES)]
            for ng in range_constexpr(KV_SUBTILES):
                v_a = read_v_a_packs(ng)
                for qg in range_constexpr(Q_SUBTILES):
                    cur = Vec.filled(MFMA_ELEMS_PER_LANE, 0.0, fx.Float32).ir_value()
                    for ks in range_constexpr(DK_STEPS):
                        cur = mfma_acc(v_a[ks].ir_value(), do_packs[ks][qg], cur)
                    da_vals = [Vec(cur)[i] for i in range_constexpr(MFMA_ELEMS_PER_LANE)]
                    grad_vals, keep = g_meta[ng][qg]
                    ds_vals = []
                    for i in range_constexpr(MFMA_ELEMS_PER_LANE):
                        gated = _fmul(_fmul(c_inv_n, grad_vals[i]), da_vals[i])
                        ds_vals.append(keep[i].select(gated, c_zero_f))
                    ds_packs[ng][qg] = pack_frag(ds_vals)
            return ds_packs

        # ==== GEMM (dQ): dS(reused frag) * K -> dQ (alpha applied at epilogue) ====
        def accum_dq_tile(dq_acc, ds_packs):
            """dQ[q, hc] += dS[q, kv] * K[kv, hc]. A = dS frag, B = K[kv, hc] from LDS."""
            for c in range_constexpr(HC_CHUNKS):
                kb_packs = []
                for ng in range_constexpr(KV_SUBTILES):
                    hc_col = fx.Int32(c * MFMA_M) + lane_mod_16
                    kv_lane = fx.Int32(ng * MFMA_M) + lane_div_16 * fx.Int32(MFMA_LANE_K)
                    elems = []
                    for i in range_constexpr(MFMA_LANE_K):
                        kv_row = kv_lane + fx.Int32(i)
                        elems.append(Vec.load(Vec.make_type(1, elem_dtype), k_smem.get(), [fx.Index(kv_row), fx.Index(k_swz_col(kv_row, hc_col))]))
                    kb_packs.append(Vec.from_elements([Vec(e)[0] for e in elems], elem_dtype).ir_value())
                for qg in range_constexpr(Q_SUBTILES):
                    acc_off = c * Q_SUBTILES + qg
                    cur = dq_acc[acc_off]
                    for ng in range_constexpr(KV_SUBTILES):
                        cur = mfma_acc(ds_packs[ng][qg], kb_packs[ng], cur)
                    dq_acc[acc_off] = cur
            return dq_acc

        v_reg_outstanding = NUM_BATCHES_V

        def run_kv_tile(dq_acc, kv_start):
            async_load_k(kv_start)
            v_vecs = async_load_v_regs(kv_start)
            _waitcnt_vm_n(v_reg_outstanding)
            gpu.barrier()
            k_packs = [read_k_a_packs(ng) for ng in range_constexpr(KV_SUBTILES)]
            g_meta = compute_gate_tile(kv_start, k_packs)
            _waitcnt_vm_n(0)
            store_v_regs_to_lds(v_vecs)
            rocdl.sched_group_barrier(rocdl.mask_dswr, NUM_BATCHES_V, 0)
            gpu.barrier()  # V published; K still resident in LDS for dQ's B-operand
            ds_packs = compute_ds_packs(g_meta)
            dq_acc = accum_dq_tile(dq_acc, ds_packs)
            return dq_acc

        if active:
            acc_init = [c_zero_v4f32 for _ in range(N_ACC)]
            loop_results = acc_init
            for kv_tile, it in range(fx.Index(0), fx.Index(n_tiles), fx.Index(1), init=acc_init):  # ty: ignore
                it_list = list(it) if isinstance(it, (list, tuple)) else [it]
                dq_acc = [it_list[i] for i in range(N_ACC)]
                kv_start = fx.Int32(kv_tile) * fx.Int32(BLOCK_N)
                dq_acc = run_kv_tile(dq_acc, kv_start)
                loop_results = yield dq_acc

            # ---- Epilogue: store dQ (alpha applied here) ----
            # dQ C[m=q, n=hc]: q row = q_wave_base + qg*16 + lane_div_16*4 + e; hc col = c*16 + lane_mod_16.
            results = list(loop_results) if isinstance(loop_results, (list, tuple)) else [loop_results]
            for qg in range_constexpr(Q_SUBTILES):
                q_row_base = q_wave_base + fx.Int32(qg * MFMA_M) + lane_div_16 * fx.Int32(MFMA_LANE_K)
                for e in range_constexpr(MFMA_ELEMS_PER_LANE):
                    q_row_e = q_row_base + fx.Int32(e)
                    if q_row_e < seq_len:
                        for c in range_constexpr(HC_CHUNKS):
                            ov = results[c * Q_SUBTILES + qg]
                            hc_col = fx.Int32(c * MFMA_M) + lane_mod_16
                            val = fx.Float32(_fmul(Vec(ov)[e], c_alpha)).to(elem_dtype)
                            dq[fx.Int64(seq_start + q_row_e), head_idx, hc_col] = val

    _hstu_compile_hints = {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
    }

    @flyc.jit
    def launch_hstu_attention_bwd_dq(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        do: fx.Tensor,
        seq_offsets: fx.Tensor,
        num_targets: fx.Tensor,
        dq: fx.Tensor,
        stream: fx.Stream,
    ) -> None:
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        grid = num_q_tiles * batch * num_heads
        hstu_attention_bwd_dq(
            q,
            k,
            v,
            do,
            seq_offsets,
            num_targets,
            dq,
            value_attrs={
                "passthrough": [
                    ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
                    ["no-nans-fp-math", "true"],
                    ["unsafe-fp-math", "true"],
                ],
                "rocdl.waves_per_eu": WAVES_PER_EU,
                "rocdl.flat_work_group_size": f"{BLOCK_THREADS},{BLOCK_THREADS}",
            },
        ).launch(
            grid=grid,
            block=BLOCK_THREADS,
            smem=0,
            stream=stream,
        )

    launch_hstu_attention_bwd_dq.compile_hints = _hstu_compile_hints
    return launch_hstu_attention_bwd_dq
