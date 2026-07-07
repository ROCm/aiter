"""hstu_attention_bwd - FlyDSL kernel (Phase 1: dV only, causal-only)

Backward of HSTU attention. Given dO, recompute S = alpha*Q*K^T and sigma from
Q,K (nothing is stashed by the forward), form the masked, silu-gated attention
weights, and produce gradients. This first phase implements **dV only**:

    dV[kv, d] = (1/N) * sum_q P[q, kv] * dO[q, d],   P = mask .* silu(alpha*Q*K^T)

i.e. dV = A^T dO with A = mask .* silu(S)/N (the 1/N is hoisted to the epilogue,
exactly as the forward hoists it to the O epilogue).

Relationship to the forward kernel (hstu_attention_fwd.py): the two are the same
tiled MFMA pipeline with roles swapped. dV reduces over the **query** index, so:
  - the program owns a KV tile (BLOCK_M kv rows) and streams query tiles (BLOCK_N),
  - the streamed head-dim operand staged through LDS is **Q** (forward staged K),
  - the resident register operand is **K** (forward held Q),
  - **dO** takes the place of V (streamed, hidden_dim, register-prefetched -> LDS),
  - GEMM1 = A*B with A = Q (streamed) and B = K (resident) => C[m=q, n=kv] = S[q,kv];
    reusing that C fragment as the GEMM2 A-operand gives P^T[kv, q], contracting q,
  - the causal bound becomes a **lower** bound on the streamed q tiles (q >= kv),
  - dV is a single-writer accumulator (each KV row owned by one program) -> no atomics.

Constraints (Phase 1):
  - causal only; no num_targets / max_attn_len / contextual_seq_len yet.
  - dtype in {f16, bf16}; accumulate in fp32.
  - head_dim % 16 == 0, hidden_dim % 16 == 0; (batch*num_heads) % 8 == 0.
  - fast/unsafe FP math: not strict IEEE-754 (mirrors the forward's SiLU).
"""

import functools
import math as host_math

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
from flydsl.utils.smem_allocator import SMEM_CAPACITY_MAP, SmemAllocator, SmemPtr


def _dtype_to_elem_type(dtype_str: str):
    if dtype_str == "f16":
        return fx.Float16
    if dtype_str == "bf16":
        return fx.BFloat16
    raise ValueError(f"unsupported dtype: {dtype_str!r} (expected 'f16' or 'bf16')")


# ---- Kernel Geometry Constants (shared with the forward) ----

WARP_SIZE = 64
NUM_GRID_GROUPS = 8
MFMA_M = 16
MFMA_N = 16
MFMA_K = 16
MFMA_LANE_K = 4
MFMA_ELEMS_PER_LANE = (MFMA_M * MFMA_N) // WARP_SIZE  # 4 f32 per lane


def _arch_dma_params(arch: str | None = None):
    if arch is None:
        arch = get_rocm_arch()
    if (arch or "").startswith("gfx942"):
        dma_bytes = 4  # CDNA3 dword
        k_swz_rows, k_swz_shift = 16, 2
    else:
        dma_bytes = 16  # CDNA4 dwordx4
        k_swz_rows, k_swz_shift = 8, 3
    return dma_bytes, dma_bytes // 2, k_swz_rows, k_swz_shift


@functools.lru_cache(maxsize=16384)
def lds_cap_bytes(arch: str | None = None) -> int:
    default_lds_cap_bytes = 65536
    if arch is None:
        arch = get_rocm_arch()
    return SMEM_CAPACITY_MAP.get(arch, default_lds_cap_bytes)


_LOG2E = host_math.log2(host_math.e)


def _waitcnt_vm_n(n: int):
    """Emit s_waitcnt vmcnt(n) only (lgkmcnt=63, expcnt=7)."""
    vmcnt_lo_mask = 0xF
    lgkmcnt_expcnt_base = 0x3F70
    vmcnt_hi_shift = 14
    vmcnt_hi_mask = 0x3
    val = (n & vmcnt_lo_mask) | lgkmcnt_expcnt_base | (((n >> 4) & vmcnt_hi_mask) << vmcnt_hi_shift)
    rocdl.s_waitcnt(val)


def validate_hstu_attention_bwd(
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
    block_m: int,
    block_n: int,
    num_waves: int,
    waves_per_eu: int,
    arch: str | None = None,
) -> None:
    if arch is None:
        arch = get_rocm_arch()
    if not arch.startswith("gfx942") and not arch.startswith("gfx950"):
        raise ValueError(f"hstu attention bwd unsupported arch: {arch!r} (expected 'gfx942' or 'gfx950')")

    if dtype_str not in {"f16", "bf16"}:
        raise ValueError(f"unsupported dtype: {dtype_str!r} (expected 'f16' or 'bf16')")
    if not causal:
        raise ValueError("hstu_attention_bwd only supports causal attention")

    # Phase 1 restrictions: masking variants land in Phase 4.
    if has_targets:
        raise ValueError("hstu_attention_bwd Phase 1 does not support num_targets yet")
    if max_attn_len != 0:
        raise ValueError("hstu_attention_bwd Phase 1 does not support max_attn_len yet")
    if contextual_seq_len != 0:
        raise ValueError("hstu_attention_bwd Phase 1 does not support contextual_seq_len yet")

    if batch <= 0:
        raise ValueError(f"batch must be positive, got {batch}")
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
    if not host_math.isfinite(alpha):
        raise ValueError(f"alpha must be finite, got {alpha}")
    if head_dim <= 0 or head_dim % MFMA_K != 0:
        raise ValueError(f"head_dim must be positive and a multiple of MFMA_K={MFMA_K}, got {head_dim}")
    if hidden_dim <= 0 or hidden_dim % MFMA_M != 0:
        raise ValueError(f"hidden_dim must be positive and a multiple of MFMA_M={MFMA_M}, got {hidden_dim}")
    if (batch * num_heads) % NUM_GRID_GROUPS != 0:
        raise ValueError(f"require (batch*num_heads) % {NUM_GRID_GROUPS} == 0, got {batch * num_heads}")

    if block_m <= 0:
        raise ValueError(f"block_m must be positive, got {block_m}")
    if block_n <= 0:
        raise ValueError(f"block_n must be positive, got {block_n}")
    if num_waves <= 0:
        raise ValueError(f"num_waves must be positive, got {num_waves}")
    if waves_per_eu < 0:
        raise ValueError(f"waves_per_eu must be non-negative, got {waves_per_eu}")
    if block_m % (num_waves * MFMA_M) != 0:
        raise ValueError(f"block_m {block_m} must be a multiple of num_waves*MFMA_M ({num_waves * MFMA_M})")
    if block_n % MFMA_M != 0:
        raise ValueError(f"block_n {block_n} must be a multiple of MFMA_M={MFMA_M}")

    _, dma_elems, _, _ = _arch_dma_params()
    block_threads = num_waves * WARP_SIZE
    elems_per_dma_pass = block_threads * dma_elems
    head_dim_k = ((head_dim + 63) // 64) * 64
    # Streamed Q tile staged through LDS: [BLOCK_N, head_dim_k].
    if (block_n * head_dim_k) % elems_per_dma_pass != 0:
        raise ValueError("Q DMA tile does not divide the dword DMA pass evenly")
    # Streamed dO tile: [BLOCK_N, hidden_dim].
    if (block_n * hidden_dim) % elems_per_dma_pass != 0:
        raise ValueError("dO DMA tile does not divide the dword DMA pass evenly")

    vec_v = 8 if (hidden_dim % 8 == 0 and (block_n * hidden_dim) % (block_threads * 8) == 0) else dma_elems
    threads_per_row_v = hidden_dim // vec_v
    if block_threads % threads_per_row_v != 0:
        raise ValueError(f"block_threads={block_threads} must be divisible by threads_per_row_v={threads_per_row_v}")
    rows_per_batch_v = block_threads // threads_per_row_v
    if not (block_n % rows_per_batch_v == 0 or rows_per_batch_v > block_n):
        raise ValueError(f"rows_per_batch_v={rows_per_batch_v} must divide block_n={block_n}, unless rows_per_batch_v > block_n")

    lds_cap = lds_cap_bytes()
    lds_bytes = block_n * head_dim_k * 2 + block_n * hidden_dim * 2
    if lds_bytes > lds_cap:
        raise ValueError(f"LDS tile {lds_bytes} B exceeds the {lds_cap} B budget")


@functools.lru_cache(maxsize=16384)
def build_hstu_attention_bwd(
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
    validate_hstu_attention_bwd(
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

    # BLOCK_M = owned KV tile (dV output rows). BLOCK_N = streamed query tile.
    BLOCK_M = block_m
    BLOCK_N = block_n
    NUM_WAVES = num_waves
    BLOCK_THREADS = NUM_WAVES * WARP_SIZE
    ROWS_PER_WAVE = BLOCK_M // NUM_WAVES
    KV_OWNED_SUBTILES = ROWS_PER_WAVE // MFMA_M  # owned KV sub-tiles per wave
    Q_STREAM_SUBTILES = BLOCK_N // MFMA_N  # streamed query sub-tiles
    WAVES_PER_EU = waves_per_eu

    DMA_BYTES, DMA_ELEMS, K_SWZ_ROWS, K_SWZ_SHIFT = _arch_dma_params()

    elem_dtype = _dtype_to_elem_type(dtype_str)
    is_bf16 = dtype_str == "bf16"

    K_STEPS = head_dim // MFMA_K  # real contraction steps (resident K side)
    HEAD_DIM_K = ((head_dim + 63) // 64) * 64
    K_STEPS_K = HEAD_DIM_K // MFMA_K  # padded steps (streamed Q side)
    D_CHUNKS = hidden_dim // MFMA_M  # dV accumulator / GEMM2 chunks
    DK_STEPS = hidden_dim // MFMA_K  # dA contraction steps (over hidden d)
    HC_CHUNKS = head_dim // MFMA_M  # dK accumulator chunks (over head_dim)

    num_kv_tiles = (max_seq_len + BLOCK_M - 1) // BLOCK_M
    hz_per_group = (batch * num_heads) // NUM_GRID_GROUPS

    stride_qk_n = num_heads * head_dim

    Q_STRIDE = HEAD_DIM_K  # streamed Q LDS stride (64-aligned, XOR-swizzled)
    q_lds_bytes = BLOCK_N * Q_STRIDE * 2

    DO_STRIDE = hidden_dim  # streamed dO LDS stride (row-major [q, d])
    do_lds_bytes = BLOCK_N * DO_STRIDE * 2

    q_tile_elems = BLOCK_N * Q_STRIDE
    elems_per_dma_pass = BLOCK_THREADS * DMA_ELEMS
    assert q_tile_elems % elems_per_dma_pass == 0
    NUM_DMA_Q = q_tile_elems // elems_per_dma_pass
    PAIRS_PER_ROW_Q = Q_STRIDE // DMA_ELEMS

    do_tile_elems = BLOCK_N * hidden_dim
    assert do_tile_elems % elems_per_dma_pass == 0

    VEC_DO = 8 if (hidden_dim % 8 == 0 and (BLOCK_N * hidden_dim) % (BLOCK_THREADS * 8) == 0) else DMA_ELEMS
    THREADS_PER_ROW_DO = hidden_dim // VEC_DO
    assert BLOCK_THREADS % THREADS_PER_ROW_DO == 0
    ROWS_PER_BATCH_DO = BLOCK_THREADS // THREADS_PER_ROW_DO
    assert BLOCK_N % ROWS_PER_BATCH_DO == 0 or ROWS_PER_BATCH_DO > BLOCK_N
    NUM_BATCHES_DO = max(1, BLOCK_N // ROWS_PER_BATCH_DO)
    DO_NEEDS_GUARD = ROWS_PER_BATCH_DO > BLOCK_N

    allocator = SmemAllocator(None, global_sym_name="hstu_attention_bwd_smem")
    q_lds_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = q_lds_offset + q_lds_bytes
    do_lds_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = do_lds_offset + do_lds_bytes

    # ---- Device Kernel ----
    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def hstu_attention_bwd(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        do: fx.Tensor,
        seq_offsets: fx.Tensor,
        num_targets: fx.Tensor,
        dv: fx.Tensor,
        dk: fx.Tensor,
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

        # ---- Group-major grid decode -> (batch_idx, head_idx, kv_tile_idx) ----
        block_id = fx.Int32(gpu.block_idx.x)
        grid_group = block_id % fx.Int32(NUM_GRID_GROUPS)
        pos_in_group = block_id // fx.Int32(NUM_GRID_GROUPS)
        local_hz_idx = pos_in_group // fx.Int32(num_kv_tiles)
        kv_tile_idx = pos_in_group % fx.Int32(num_kv_tiles)
        hz_idx = grid_group * fx.Int32(hz_per_group) + local_hz_idx
        batch_idx = hz_idx // fx.Int32(num_heads)
        head_idx = hz_idx % fx.Int32(num_heads)

        seq_start = fx.Int32(seq_offsets[batch_idx])
        seq_len = fx.Int32(seq_offsets[batch_idx + fx.Int32(1)]) - seq_start

        # ---- Global tensor views ----
        def grouped_loader(t, dim, g):
            in_row = fx.make_layout((dim // g, g), (g, 1))

            def load(row_i64, head_val, colgrp):
                sub = t[row_i64, head_val, None]
                return fx.make_view(fx.get_iter(sub), in_row)[colgrp, None].load()

            return load

        k_load = grouped_loader(k, head_dim, MFMA_LANE_K)  # resident K (B-operand for S)
        v_load = grouped_loader(v, hidden_dim, MFMA_LANE_K)  # resident V (B-operand for dA)
        do_load = grouped_loader(do, hidden_dim, VEC_DO)  # streamed dO register prefetch

        q_head_offset = head_idx * fx.Int32(head_dim)

        # ---- Streamed Q DMA buffer resource ----
        q_base_byte_offset = (fx.Int64(seq_start) * fx.Int64(stride_qk_n) + fx.Int64(q_head_offset)) * fx.Int64(2)
        q_rsrc = buffer_ops.create_buffer_resource(q, max_size=True, base_byte_offset=q_base_byte_offset)

        lds_base = allocator.get_base()
        q_smem = SmemPtr(lds_base, q_lds_offset, elem_type, shape=(BLOCK_N, Q_STRIDE))
        do_smem = SmemPtr(lds_base, do_lds_offset, elem_type, shape=(BLOCK_N, DO_STRIDE))
        q_lds_byte_base = buffer_ops.extract_base_index(q_smem.get(), address_space=3)

        def q_swz_col(tile_row, col):
            return col ^ ((tile_row & fx.Int32(K_SWZ_ROWS - 1)) << fx.Int32(K_SWZ_SHIFT))

        kv_wave_base = kv_tile_idx * fx.Int32(BLOCK_M) + wave_id * fx.Int32(ROWS_PER_WAVE)

        # ---- Owned KV rows / bounds per KV sub-tile ----
        kv_rows = []
        kv_in_bounds = []
        for og in range_constexpr(KV_OWNED_SUBTILES):
            local = kv_wave_base + fx.Int32(og * MFMA_M) + lane_mod_16
            kv_rows.append(local)
            kv_in_bounds.append(local < seq_len)

        # ---- Resident K B-operand packs (per owned KV sub-tile), for GEMM1 S = Q*K^T ----
        k_packs = []  # k_packs[ks][og]
        for ks in range_constexpr(K_STEPS):
            k_col = fx.Int32(ks * MFMA_K) + lane_div_16 * fx.Int32(MFMA_LANE_K)
            per_og = []
            for og in range_constexpr(KV_OWNED_SUBTILES):
                safe = kv_in_bounds[og].select(seq_start + kv_rows[og], seq_start)
                raw = k_load(fx.Int64(safe), head_idx, k_col // fx.Int32(MFMA_LANE_K)).ir_value()
                per_og.append(kv_in_bounds[og].select(raw, c_zero_mfma_pack))
            k_packs.append(per_og)

        # ---- Resident V B-operand packs (per owned KV sub-tile), for dA = dO*V^T ----
        # b_pack[i] = V[kv = lane_mod_16, d = ks*16 + lane_div_16*4 + i]; contraction over d.
        v_packs = []  # v_packs[ks][og]
        for ks in range_constexpr(DK_STEPS):
            v_col = fx.Int32(ks * MFMA_K) + lane_div_16 * fx.Int32(MFMA_LANE_K)
            per_og = []
            for og in range_constexpr(KV_OWNED_SUBTILES):
                safe = kv_in_bounds[og].select(seq_start + kv_rows[og], seq_start)
                raw = v_load(fx.Int64(safe), head_idx, v_col // fx.Int32(MFMA_LANE_K)).ir_value()
                per_og.append(kv_in_bounds[og].select(raw, c_zero_mfma_pack))
            v_packs.append(per_og)

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

        def silu_and_grad_batch(s_list):
            """Return (silu(alpha*s), silu'(alpha*s)) for a list of raw scores s.

            With sc = alpha*s, sig = sigmoid(sc):
              silu = sc*sig
              silu' = sig*(1 + sc*(1 - sig))     (the SiLU derivative w.r.t. sc)
            Uses the same fast exp2/rcp sigmoid as the forward so numerics match.
            1/N is applied later (dV epilogue for P; dS gate for silu').
            """
            sc = [_fmul(s, c_alpha) for s in s_list]
            tt = [_fmul(s, c_neg_log2e) for s in sc]
            emu = [llvm.call_intrinsic(compute_type, "llvm.amdgcn.exp2.f32", [t], [], []) for t in tt]
            den = [_fadd(c_one_f, e) for e in emu]
            sig = [llvm.call_intrinsic(compute_type, "llvm.amdgcn.rcp.f32", [d], [], []) for d in den]
            silu = [_fmul(sc[i], sig[i]) for i in range(len(s_list))]
            grad = [
                _fmul(sig[i], _fadd(c_one_f, _fmul(sc[i], _fadd(c_one_f, _fmul(c_neg_one_f, sig[i])))))
                for i in range(len(s_list))
            ]
            return silu, grad

        def pack_p(vals):
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

        kv_owned_ids = kv_rows  # causal-only: id == raw position

        # ---- Streamed query range: causal lower bound (q >= kv) ----
        kv_start_row = kv_tile_idx * fx.Int32(BLOCK_M)
        active = kv_start_row < seq_len
        q_upper = active.select(seq_len, fx.Int32(0))
        n_q_tiles = (q_upper + fx.Int32(BLOCK_N - 1)) // fx.Int32(BLOCK_N)
        q_tile_start = kv_start_row // fx.Int32(BLOCK_N)

        N_ACC_DV = D_CHUNKS * KV_OWNED_SUBTILES  # dV accumulators
        N_ACC_DK = HC_CHUNKS * KV_OWNED_SUBTILES  # dK accumulators
        N_ACC = N_ACC_DV + N_ACC_DK  # loop-carried [dV..., dK...]
        c_zero_v4f32 = Vec.filled(MFMA_ELEMS_PER_LANE, 0.0, fx.Float32).ir_value()

        # ---- Streamed Q DMA: global -> LDS (dword, swizzled) ----
        dma_size = fx.Int32(DMA_BYTES)
        dma_soff = fx.Int32(0)
        dma_off = fx.Int32(0)
        dma_aux = fx.Int32(1)
        c_dma_elems = fx.Int32(DMA_ELEMS)
        c_pairs_per_row_q = fx.Int32(PAIRS_PER_ROW_Q)

        wave_lds_base_q = q_lds_byte_base + fx.Index(wave_id) * fx.Index(WARP_SIZE * DMA_BYTES)
        wave_lds_lane0_q = rocdl.readfirstlane(fx.Int64.ir_type, fx.Int64(wave_lds_base_q))
        q_dma_rows = []
        q_dma_gcols = []
        for d in range_constexpr(NUM_DMA_Q):
            pair = tid + fx.Int32(d * BLOCK_THREADS)
            row = pair // c_pairs_per_row_q
            col_pair = pair % c_pairs_per_row_q
            col = col_pair * c_dma_elems
            q_dma_rows.append(row)
            q_dma_gcols.append(q_swz_col(row, col))

        c_stride_qk_n = fx.Int32(stride_qk_n)

        def async_load_q(q_start):
            for d in range_constexpr(NUM_DMA_Q):
                row = q_dma_rows[d]
                in_bounds = (q_start + row) < seq_len
                local_tok = in_bounds.select(q_start + row, fx.Int32(0))
                g_elem = local_tok * c_stride_qk_n + q_dma_gcols[d]
                voffset = g_elem * fx.Int32(2)
                lds_ptr = buffer_ops.create_llvm_ptr(wave_lds_lane0_q + fx.Int64(d * BLOCK_THREADS * DMA_BYTES), address_space=3)
                rocdl.raw_ptr_buffer_load_lds(q_rsrc, lds_ptr, dma_size, voffset, dma_soff, dma_off, dma_aux)

        # ---- Streamed dO register prefetch ----
        do_load_row_in_batch = tid // fx.Int32(THREADS_PER_ROW_DO)
        do_load_lane_in_row = tid % fx.Int32(THREADS_PER_ROW_DO)
        do_load_col = do_load_lane_in_row * fx.Int32(VEC_DO)

        def async_load_do_regs(q_start):
            vecs = []
            for b in range_constexpr(NUM_BATCHES_DO):
                row = do_load_row_in_batch + fx.Int32(b * ROWS_PER_BATCH_DO)
                tok = q_start + row
                in_bounds = tok < seq_len
                if DO_NEEDS_GUARD:
                    in_bounds = in_bounds & (row < fx.Int32(BLOCK_N))
                safe_tok = in_bounds.select(seq_start + tok, seq_start)
                raw = do_load(fx.Int64(safe_tok), head_idx, do_load_col // fx.Int32(VEC_DO)).ir_value()
                vecs.append(in_bounds.select(raw, Vec.filled(VEC_DO, 0.0, elem_dtype).ir_value()))
            return vecs

        def store_do_regs_to_lds(vecs):
            for b in range_constexpr(NUM_BATCHES_DO):
                row = do_load_row_in_batch + fx.Int32(b * ROWS_PER_BATCH_DO)
                Vec.store(Vec(vecs[b]), do_smem.get(), [fx.Index(row), fx.Index(do_load_col)])

        # ==== GEMM1: Q(streamed)*K^T(owned) -> S[q, kv]; mask + silu -> P frag ====
        def read_q_packs(ng):
            """LDS-read streamed-Q A-operand packs for query sub-tile ng."""
            q_row = fx.Int32(ng * MFMA_M) + lane_mod_16
            packs = []
            for ks in range_constexpr(K_STEPS_K):
                q_col = fx.Int32(ks * MFMA_K) + lane_div_16 * fx.Int32(MFMA_LANE_K)
                packs.append(Vec.load(mfma_pack_type, q_smem.get(), [fx.Index(q_row), fx.Index(q_swz_col(q_row, q_col))]))
            return packs

        def compute_s_tile(q_start, q_packs_by_ng):
            """GEMM1 for one streamed-query tile.

            Returns (p_packs, s_meta):
              - p_packs[ng][og]: packed P = mask .* silu(alpha*S)  (GEMM2 A-operand for dV)
              - s_meta[ng][og] = (grad_vals[4], keep[4]): the SiLU-derivative gate and mask,
                retained so dS = keep .* (1/N) .* grad .* dA can be formed after dO publishes.

            C fragment C[m=q, n=kv]: value[i] -> (q = ng*16 + lane_div_16*4 + i,
            kv = og*16 + lane_mod_16). Mask keeps (q >= kv) or diagonal.
            """
            p_packs = [[None for _ in range_constexpr(KV_OWNED_SUBTILES)] for _ in range_constexpr(Q_STREAM_SUBTILES)]
            s_meta = [[None for _ in range_constexpr(KV_OWNED_SUBTILES)] for _ in range_constexpr(Q_STREAM_SUBTILES)]
            for ng in range_constexpr(Q_STREAM_SUBTILES):
                q_packs = [Vec(q_packs_by_ng[ng][ks]) for ks in range_constexpr(K_STEPS_K)]
                q_base = q_start + fx.Int32(ng * MFMA_M) + lane_div_16 * fx.Int32(MFMA_LANE_K)
                q_raw = [q_base + fx.Int32(i) for i in range_constexpr(MFMA_ELEMS_PER_LANE)]
                q_in_seq = [q_raw[i] < seq_len for i in range_constexpr(MFMA_ELEMS_PER_LANE)]
                for og in range_constexpr(KV_OWNED_SUBTILES):
                    cur = Vec.filled(MFMA_ELEMS_PER_LANE, 0.0, fx.Float32).ir_value()
                    for ks in range_constexpr(K_STEPS_K):
                        k_op = k_packs[ks][og] if ks < K_STEPS else c_zero_mfma_pack
                        cur = mfma_acc(q_packs[ks].ir_value(), k_op, cur)
                    s_vals = [Vec(cur)[i] for i in range_constexpr(MFMA_ELEMS_PER_LANE)]

                    def keep_row(i):
                        """causal mask for (streamed q = q_raw[i], owned kv = kv_owned_ids[og])."""
                        dist = q_raw[i] - kv_owned_ids[og]
                        keep = (q_raw[i] == kv_rows[og]) | (dist > fx.Int32(0))
                        keep = keep & q_in_seq[i] & kv_in_bounds[og]
                        return keep

                    keep = [keep_row(i) for i in range_constexpr(MFMA_ELEMS_PER_LANE)]
                    silu_vals, grad_vals = silu_and_grad_batch(s_vals)
                    p_vals = [keep[i].select(silu_vals[i], c_zero_f) for i in range_constexpr(MFMA_ELEMS_PER_LANE)]
                    p_packs[ng][og] = pack_p(p_vals)
                    s_meta[ng][og] = (grad_vals, keep)
            return p_packs, s_meta

        # ==== GEMM2 (dV): P^T(reused frag) * dO -> dV ====
        def accum_dv_tile(dv_acc, p_packs):
            """dV[kv, d] += P^T[kv, q] * dO[q, d]. A = P frag, B = dO[q, d]."""
            for c in range_constexpr(D_CHUNKS):
                do_packs = []
                for ng in range_constexpr(Q_STREAM_SUBTILES):
                    d_col = fx.Int32(c * MFMA_M) + lane_mod_16
                    q_lane = fx.Int32(ng * MFMA_M) + lane_div_16 * fx.Int32(MFMA_LANE_K)
                    elems = []
                    for i in range_constexpr(MFMA_LANE_K):
                        elems.append(Vec.load(Vec.make_type(1, elem_dtype), do_smem.get(), [fx.Index(q_lane + fx.Int32(i)), fx.Index(d_col)]))
                    do_packs.append(Vec.from_elements([Vec(e)[0] for e in elems], elem_dtype).ir_value())
                for og in range_constexpr(KV_OWNED_SUBTILES):
                    acc_off = c * KV_OWNED_SUBTILES + og
                    cur = dv_acc[acc_off]
                    for ng in range_constexpr(Q_STREAM_SUBTILES):
                        cur = mfma_acc(p_packs[ng][og], do_packs[ng], cur)
                    dv_acc[acc_off] = cur
            return dv_acc

        # ==== dA = dO * V^T; dS = keep .* (1/N) .* silu' .* dA -> packed dS frag ====
        def read_do_a_packs(ng):
            """dO A-operand packs for dA: a_pack[i] = dO[q = ng*16 + lane_mod_16,
            d = ks*16 + lane_div_16*4 + i] (4 contiguous d)."""
            q_row = fx.Int32(ng * MFMA_M) + lane_mod_16
            packs = []
            for ks in range_constexpr(DK_STEPS):
                d_col = fx.Int32(ks * MFMA_K) + lane_div_16 * fx.Int32(MFMA_LANE_K)
                packs.append(Vec.load(mfma_pack_type, do_smem.get(), [fx.Index(q_row), fx.Index(d_col)]))
            return packs

        def compute_ds_packs(s_meta):
            """Form packed dS fragments [ng][og] from dA (=dO*V^T) and the retained gate/mask."""
            ds_packs = [[None for _ in range_constexpr(KV_OWNED_SUBTILES)] for _ in range_constexpr(Q_STREAM_SUBTILES)]
            for ng in range_constexpr(Q_STREAM_SUBTILES):
                do_a = read_do_a_packs(ng)
                for og in range_constexpr(KV_OWNED_SUBTILES):
                    cur = Vec.filled(MFMA_ELEMS_PER_LANE, 0.0, fx.Float32).ir_value()
                    for ks in range_constexpr(DK_STEPS):
                        cur = mfma_acc(do_a[ks].ir_value(), v_packs[ks][og], cur)
                    da_vals = [Vec(cur)[i] for i in range_constexpr(MFMA_ELEMS_PER_LANE)]
                    grad_vals, keep = s_meta[ng][og]
                    ds_vals = []
                    for i in range_constexpr(MFMA_ELEMS_PER_LANE):
                        gated = _fmul(_fmul(c_inv_n, grad_vals[i]), da_vals[i])
                        ds_vals.append(keep[i].select(gated, c_zero_f))
                    ds_packs[ng][og] = pack_p(ds_vals)
            return ds_packs

        # ==== GEMM (dK): dS^T(reused frag) * Q -> dK (alpha applied at epilogue) ====
        def accum_dk_tile(dk_acc, ds_packs):
            """dK[kv, hc] += dS^T[kv, q] * Q[q, hc]. A = dS frag, B = Q[q, hc]."""
            for c in range_constexpr(HC_CHUNKS):
                qb_packs = []
                for ng in range_constexpr(Q_STREAM_SUBTILES):
                    hc_col = fx.Int32(c * MFMA_M) + lane_mod_16
                    q_lane = fx.Int32(ng * MFMA_M) + lane_div_16 * fx.Int32(MFMA_LANE_K)
                    elems = []
                    for i in range_constexpr(MFMA_LANE_K):
                        q_row = q_lane + fx.Int32(i)
                        elems.append(Vec.load(Vec.make_type(1, elem_dtype), q_smem.get(), [fx.Index(q_row), fx.Index(q_swz_col(q_row, hc_col))]))
                    qb_packs.append(Vec.from_elements([Vec(e)[0] for e in elems], elem_dtype).ir_value())
                for og in range_constexpr(KV_OWNED_SUBTILES):
                    acc_off = c * KV_OWNED_SUBTILES + og
                    cur = dk_acc[acc_off]
                    for ng in range_constexpr(Q_STREAM_SUBTILES):
                        cur = mfma_acc(ds_packs[ng][og], qb_packs[ng], cur)
                    dk_acc[acc_off] = cur
            return dk_acc

        do_reg_outstanding = NUM_BATCHES_DO

        def run_q_tile(dv_acc, dk_acc, q_start):
            async_load_q(q_start)
            do_vecs = async_load_do_regs(q_start)
            _waitcnt_vm_n(do_reg_outstanding)
            gpu.barrier()
            q_packs = [read_q_packs(ng) for ng in range_constexpr(Q_STREAM_SUBTILES)]
            p_packs, s_meta = compute_s_tile(q_start, q_packs)
            _waitcnt_vm_n(0)
            store_do_regs_to_lds(do_vecs)
            rocdl.sched_group_barrier(rocdl.mask_dswr, NUM_BATCHES_DO, 0)
            gpu.barrier()  # dO published; Q still resident in LDS for dK's B-operand
            dv_acc = accum_dv_tile(dv_acc, p_packs)
            ds_packs = compute_ds_packs(s_meta)
            dk_acc = accum_dk_tile(dk_acc, ds_packs)
            return dv_acc, dk_acc

        if active:
            acc_init = [c_zero_v4f32 for _ in range(N_ACC)]
            loop_results = acc_init
            for q_tile, it in range(fx.Index(q_tile_start), fx.Index(n_q_tiles), fx.Index(1), init=acc_init):  # ty: ignore
                it_list = list(it) if isinstance(it, (list, tuple)) else [it]
                dv_acc = [it_list[i] for i in range(N_ACC_DV)]
                dk_acc = [it_list[N_ACC_DV + i] for i in range(N_ACC_DK)]
                q_start = fx.Int32(q_tile) * fx.Int32(BLOCK_N)
                dv_acc, dk_acc = run_q_tile(dv_acc, dk_acc, q_start)
                loop_results = yield dv_acc + dk_acc

            # ---- Epilogue: store dV (1/N hoisted here) and dK (alpha applied here) ----
            # dV C[m=kv, n=d]: kv row = kv_wave_base + og*16 + lane_div_16*4 + e; d col = c*16 + lane_mod_16.
            # dK C[m=kv, n=hc]: same kv row layout; hc col = c*16 + lane_mod_16; scaled by alpha.
            results = list(loop_results) if isinstance(loop_results, (list, tuple)) else [loop_results]
            dv_results = results[:N_ACC_DV]
            dk_results = results[N_ACC_DV:]
            for og in range_constexpr(KV_OWNED_SUBTILES):
                kv_row_base = kv_wave_base + fx.Int32(og * MFMA_M) + lane_div_16 * fx.Int32(MFMA_LANE_K)
                for e in range_constexpr(MFMA_ELEMS_PER_LANE):
                    kv_row_e = kv_row_base + fx.Int32(e)
                    if kv_row_e < seq_len:
                        for c in range_constexpr(D_CHUNKS):
                            ov = dv_results[c * KV_OWNED_SUBTILES + og]
                            d_col = fx.Int32(c * MFMA_M) + lane_mod_16
                            val = fx.Float32(_fmul(Vec(ov)[e], c_inv_n)).to(elem_dtype)
                            dv[fx.Int64(seq_start + kv_row_e), head_idx, d_col] = val
                        for c in range_constexpr(HC_CHUNKS):
                            kv_ = dk_results[c * KV_OWNED_SUBTILES + og]
                            hc_col = fx.Int32(c * MFMA_M) + lane_mod_16
                            valk = fx.Float32(_fmul(Vec(kv_)[e], c_alpha)).to(elem_dtype)
                            dk[fx.Int64(seq_start + kv_row_e), head_idx, hc_col] = valk

    _hstu_compile_hints = {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
    }

    @flyc.jit
    def launch_hstu_attention_bwd(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        do: fx.Tensor,
        seq_offsets: fx.Tensor,
        num_targets: fx.Tensor,
        dv: fx.Tensor,
        dk: fx.Tensor,
        stream: fx.Stream,
    ) -> None:
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        grid = num_kv_tiles * batch * num_heads
        hstu_attention_bwd(
            q,
            k,
            v,
            do,
            seq_offsets,
            num_targets,
            dv,
            dk,
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

    launch_hstu_attention_bwd.compile_hints = _hstu_compile_hints
    return launch_hstu_attention_bwd
