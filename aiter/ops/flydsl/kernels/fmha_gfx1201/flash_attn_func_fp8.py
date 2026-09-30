# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FP8-input Flash Attention kernel for gfx1201 (RDNA4).

Q, K, V arrive as fp8 (e4m3) in HBM with per-tensor scales q_scale, k_scale,
v_scale; output O is bf16. Taking fp8 directly (rather than converting bf16->fp8
inside the kernel) halves K/V HBM traffic and does the quantization once upstream
instead of on every q-tile re-stream. q_scale*k_scale folds into the softmax
scale; v_scale folds into the 1/l normalizer.

WMMA 16x16x16 register layout (wave32):
  - A/B operand: two i32 registers containing 8 packed fp8 values per lane
    (lane16 = row/col, klane*8 = K-offset)
  - C/D result: v8f32 per lane, element si = C[klane*8+si][lane16]

Layout: Q/K/V/O are 1D flattened from BSHD (batch, seq_len, num_heads, head_dim).
Grid:   (batch * num_q_tiles * num_heads,)
Block:  (BLOCK_M / 16) wave32 waves by default; flat_work_group_size may override.

Requires: head_dim % 32 == 0, head_dim >= 64. K is row-major in LDS; V is
transposed as V_T[d, kv] so GEMM2 consumes packed FP8 fragments contiguously.
WMMA uses FlyDSL's typed gfx1201 FP8 atom. The remaining packed probability
conversion is localized because the pinned compiler cannot lower vector FP8 casts.
"""

import math as host_math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import (
    const_expr,
    gpu,
    range_constexpr,
)
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw

from ..kernels_common import LOG2E as _LOG2E
from ..layout_utils import crd2idx, idx2crd
from ..tensor_shim import _run_compiled, buf_copy_load, buf_copy_store, ptr_buf_tensor
from .flash_attn_func_common import (
    flatten_scores,
    kv_load_schedule,
    mask_scores,
)

NUM_PREFETCH_K = 1
NUM_PREFETCH_V = 1


def _fmul_scalar(a, b):
    """Typed scalar FP math; launch compile hints carry the fast-math policy."""
    return fx.Float32(a) * fx.Float32(b)


def _fadd_scalar(a, b):
    return fx.Float32(a) + fx.Float32(b)


def _fsub_scalar(a, b):
    return fx.Float32(a) - fx.Float32(b)


def _fmax(a, b):
    return fx.maxnumf(a, b)


def _fmul_vec(a, b):
    return (Vec(a) * Vec(b)).ir_value()


def _wave32_peer(value):
    return fx.Float32(value).shuffle_xor(16, 32)


def _next_kv_tile_start(kv_block_start, kv_upper, block_n, zero):
    next_start = kv_block_start + block_n
    return (next_start < kv_upper).select(next_start, zero)


def _update_online_softmax(
    scores,
    m_running,
    l_running,
    o_accs,
    sm_scale_log2e,
    c_zero_f,
    *,
    num_scores,
    d_chunks,
):
    """Update FP8 online-softmax state with its explicit fast-math contract."""
    local_max = scores[0]
    for idx in range_constexpr(num_scores - 1):
        local_max = _fmax(local_max, scores[idx + 1])
    row_max = _fmax(local_max, _wave32_peer(local_max))
    m_new = _fmax(m_running, row_max)

    diff_m_scaled = _fmul_scalar(_fsub_scalar(m_running, m_new), sm_scale_log2e)
    corr = fx.exp2(diff_m_scaled, fastmath="fast")
    neg_scaled_max = _fsub_scalar(c_zero_f, _fmul_scalar(sm_scale_log2e, m_new))

    probabilities = []
    local_sum = _raw(c_zero_f)
    for idx in range_constexpr(num_scores):
        diff = fx.math.fma(scores[idx], _raw(sm_scale_log2e), neg_scaled_max)
        probability = fx.exp2(diff, fastmath="fast")
        probabilities.append(probability)
        local_sum = _fadd_scalar(local_sum, probability)

    tile_sum = _fadd_scalar(local_sum, _wave32_peer(local_sum))
    l_new = _fadd_scalar(_fmul_scalar(corr, l_running), tile_sum)
    corr_vec = fx.Vector.from_elements([corr], fx.Float32).broadcast_to(8).ir_value()
    for chunk in range_constexpr(d_chunks):
        o_accs[chunk] = _fmul_vec(o_accs[chunk], corr_vec)
    return probabilities, m_new, l_new, o_accs


def get_flash_attn_fp8_lds_bytes(head_dim: int, block_n: int) -> int:
    """Return the FP8 kernel's exact static LDS allocation."""
    k_bytes = NUM_PREFETCH_K * block_n * (head_dim + 4)
    v_bytes = NUM_PREFETCH_V * head_dim * (block_n + 4)
    return k_bytes + v_bytes


def build_flash_attn_func_module(
    num_heads,
    head_dim,
    causal=True,
    dtype_str="bf16",
    sm_scale=None,
    waves_per_eu=2,
    flat_work_group_size=None,
    block_m=None,
    block_n=None,
    tail_mask=False,
    cross_attn=False,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
):
    """Build shape-tiled gfx1201 FP8 attention with pipelined GEMM2/V loads."""
    # ---- WMMA / wave32 constants ----
    WARP_SIZE = 32
    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 16
    K_SUB_N = 32
    ROWS_PER_WAVE = WMMA_M

    BLOCK_M = block_m if block_m is not None else 128
    BLOCK_N = block_n if block_n is not None else 32

    assert (
        BLOCK_N % K_SUB_N == 0
    ), f"BLOCK_N ({BLOCK_N}) must be a multiple of K_SUB_N ({K_SUB_N})"
    assert (
        BLOCK_M % ROWS_PER_WAVE == 0
    ), f"BLOCK_M ({BLOCK_M}) must be a multiple of {ROWS_PER_WAVE}"

    N_SUB_TILES = BLOCK_N // K_SUB_N
    NUM_S_ACCS = N_SUB_TILES * 2
    NUM_S_VALS = NUM_S_ACCS * 8

    NUM_WAVES = BLOCK_M // ROWS_PER_WAVE
    if flat_work_group_size is None:
        flat_work_group_size = NUM_WAVES * WARP_SIZE
    BLOCK_SIZE = flat_work_group_size

    K_STEP_QK = WMMA_K
    K_STEPS_QK = head_dim // K_STEP_QK
    WMMA_LANE_K = 8

    D_CHUNK = WMMA_N
    D_CHUNKS = head_dim // D_CHUNK

    PV_K_STEP = WMMA_K
    PV_K_STEPS = K_SUB_N // PV_K_STEP

    assert BLOCK_M % NUM_WAVES == 0
    assert head_dim % 32 == 0
    assert head_dim >= 64
    assert dtype_str in ("f16", "bf16")

    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(head_dim)

    NUM_HEADS = num_heads
    HEAD_DIM = head_dim
    CAUSAL = causal
    TAIL_MASK = tail_mask
    CROSS_ATTN = cross_attn
    STRIDE_TOKEN = NUM_HEADS * HEAD_DIM

    # Descriptive per-variant symbol name (shows up in profiles / ISA dumps).
    _name_flags = (
        f"{'_causal' if causal else ''}"
        f"{'_cross' if cross_attn else ''}"
        f"{'_tail' if tail_mask else ''}"
    )
    KERNEL_NAME = (
        f"flash_attn_func_fp8_gfx1201"
        f"_h{num_heads}_d{head_dim}_m{BLOCK_M}n{BLOCK_N}{_name_flags}"
    )

    # LDS layout -- K is row-major; V is transposed with the KV row contiguous.
    K_STRIDE = HEAD_DIM + 4  # padding to reduce bank conflicts (no swizzle)
    K_STRIDE_I32 = K_STRIDE // 4  # K in i32 units (4 fp8 per i32)

    # FP8 cooperative loads require one 16-byte vector per lane for this
    # packing and transposed-V layout.
    VEC_WIDTH = 16
    (
        THREADS_PER_ROW_LOAD,
        NUM_BATCHES_KV,
        KV_NEEDS_GUARD,
    ) = kv_load_schedule(BLOCK_SIZE, HEAD_DIM, BLOCK_N, VEC_WIDTH)
    NUM_KV_CHUNKS = BLOCK_N * THREADS_PER_ROW_LOAD

    # One aligned i32 arena preserves the packed LDS reads required by WMMA. An
    # i8 alias is used only for the transposed V scatter; both addresses share
    # the same 16-byte-aligned allocation.
    LDS_K_TILE_BYTES = BLOCK_N * K_STRIDE
    LDS_K_TOTAL_BYTES = NUM_PREFETCH_K * LDS_K_TILE_BYTES
    # FP8 V is transposed in LDS (V_T[d][kv_row]) so GEMM2 reads contiguous
    # v2i32. All V sizing and addressing is expressed directly in bytes.
    V_T_STRIDE_BYTES = BLOCK_N + 4  # fp8 bytes per d-row (kv_row inner + pad)
    V_T_STRIDE_I32 = V_T_STRIDE_BYTES // 4  # i32 words per contiguous V_T row
    V_BYTE_BASE = LDS_K_TOTAL_BYTES
    LDS_V_TOTAL_BYTES = NUM_PREFETCH_V * HEAD_DIM * V_T_STRIDE_BYTES
    LDS_TOTAL_BYTES = LDS_K_TOTAL_BYTES + LDS_V_TOTAL_BYTES

    @fx.struct
    class SharedStorage:
        kv: fx.Array[fx.Int32, LDS_TOTAL_BYTES // 4, 16]

    # Map the BF16/F16 output selector to FlyDSL's scalar element type.
    _NUMERIC_MAP = {"f16": fx.Float16, "bf16": fx.BFloat16}
    elem_numeric_cls = _NUMERIC_MAP[dtype_str]

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def flash_attn_func_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,
        seq_len: fx.Int32,
        seq_len_kv_real: fx.Int32,
        seq_len_kv: fx.Int32,
        q_scale_ptr: fx.Tensor,
        k_scale_ptr: fx.Tensor,
        v_scale_ptr: fx.Tensor,
    ):
        elem_dtype = elem_numeric_cls
        # Typed V# views preserve the tuned vector widths while keeping global
        # memory accesses in FlyDSL's buffer-view/copy API. ``unit_stride=1``
        # permits a wide access at each dword/byte index used by this layout.
        q_buf = ptr_buf_tensor(Q, fx.Int32, unit_elems=2, unit_stride=1)
        k_buf = ptr_buf_tensor(K, fx.Int32, unit_elems=4, unit_stride=1)
        v_buf = ptr_buf_tensor(V, fx.Int32, unit_elems=4, unit_stride=1)
        o_buf = ptr_buf_tensor(O, elem_dtype, unit_elems=8, unit_stride=1)
        # Per-tensor descales are one-element device tensors produced upstream.
        # Load them in the prologue without a host `.item()` synchronization.
        q_scale = fx.Float32(ptr_buf_tensor(q_scale_ptr, fx.Float32)[0])
        k_scale = fx.Float32(ptr_buf_tensor(k_scale_ptr, fx.Float32)[0])
        v_scale = fx.Float32(ptr_buf_tensor(v_scale_ptr, fx.Float32)[0])

        wmma_atom = fx.make_mma_atom(
            fx.rocdl.WMMA(WMMA_M, WMMA_N, WMMA_K, fx.Float8E4M3FN, fx.Float32)
        )

        def wmma_acc_fp8(k_v2i32_raw, q_pk_pair, c_v8):
            """Execute the gfx1201 FP8 WMMA through FlyDSL's typed atom."""
            a_frag = fx.make_rmem_tensor(8, fx.Float8E4M3FN)
            b_frag = fx.make_rmem_tensor(8, fx.Float8E4M3FN)
            c_frag = fx.make_rmem_tensor(8, fx.Float32)
            a_frag.store(Vec(k_v2i32_raw).bitcast(fx.Float8E4M3FN))
            b_frag.store(
                Vec.from_elements(q_pk_pair, fx.Int32).bitcast(fx.Float8E4M3FN)
            )
            c_frag.store(Vec(c_v8))
            fx.gemm(wmma_atom, c_frag, a_frag, b_frag, c_frag)
            return c_frag.load().ir_value()

        def _pack_probabilities_fp8(p8):
            # FlyDSL 0.3.4.1 cannot lower the typed vector f32->fp8 cast:
            # it emits vector arith.truncf with no LLVM translation. Keep
            # this target packing intrinsic localized until that lowering
            # exists; it also produces the two i32 words required by WMMA.
            _i32ty = fx.Int32.ir_type
            _c0 = fx.Int32(0).ir_value()
            pk0 = fx.rocdl.cvt_pk_fp8_f32(_i32ty, p8[0], p8[1], _c0, 0)
            pk0 = fx.rocdl.cvt_pk_fp8_f32(_i32ty, p8[2], p8[3], pk0, 1)
            pk1 = fx.rocdl.cvt_pk_fp8_f32(_i32ty, p8[4], p8[5], _c0, 0)
            pk1 = fx.rocdl.cvt_pk_fp8_f32(_i32ty, p8[6], p8[7], pk1, 1)
            return [pk0, pk1]

        # Logical coordinates and HBM element offsets are explicit u64 values.
        # V# helpers narrow at their intrinsic boundary; the public wrapper
        # rejects padded tensors whose flattened offsets exceed signed Int32.
        seq_len_v = fx.Uint64(seq_len)
        seq_len_kv_real_v = fx.Uint64(seq_len_kv_real)
        if const_expr(CROSS_ATTN):
            seq_len_kv_v = fx.Uint64(seq_len_kv)
        else:
            # Self-attn: K/V share Q's sequence length. Aliasing to seq_len_v (not
            # the seq_len_kv arg) leaves the addressing math unchanged, so the
            # self-attn path pays nothing for the cross-attn arg.
            seq_len_kv_v = seq_len_v

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_i32_ptr = lds.kv.ptr
        lds_i8_ptr = fx.recast_iter(
            fx.PointerType.get(fx.Int8.ir_type, lds_i32_ptr.address_space),
            lds_i32_ptr,
        )

        def lds_i32_view(offset, width):
            return fx.make_view(
                lds_i32_ptr + fx.Int32(offset),
                fx.make_layout(width, 1),
            )

        block_id = fx.Uint64(gpu.block_idx.x)
        tid = fx.Uint64(gpu.thread_idx.x)

        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane16 = lane % 16
        klane = lane // 16

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

        def kv_global_idx(token_idx, col):
            token = batch_idx * seq_len_kv_v + token_idx
            return token * STRIDE_TOKEN + head_idx * HEAD_DIM + col

        def _store_output_vec(base_idx, val):
            buf_copy_store(
                o_buf,
                fx.Int32(base_idx),
                val,
                elem=elem_dtype,
                unit_elems=8,
            )

        def _load_global_fp8(buffer, base_idx, *, elem, unit_elems, dword_index):
            # FP8 indices are byte indices. Int32-backed Q/K accesses therefore
            # convert to a dword offset before the wide buffer load.
            index = base_idx // fx.Uint64(4) if dword_index else base_idx
            return buf_copy_load(
                buffer,
                fx.Int32(index),
                elem=elem,
                unit_elems=unit_elems,
            ).ir_value()

        # FlyDSL 0.3.4.1 layout helpers require index-typed coordinates; keep
        # Index localized to this cooperative K/V loader boundary so its static
        # layout lowering is preserved without making runtime arithmetic Index.
        k_load_layout = fx.make_layout(
            (BLOCK_N, THREADS_PER_ROW_LOAD),
            (THREADS_PER_ROW_LOAD, 1),
        )
        k_lds_i32_layout = fx.make_layout(
            (BLOCK_N, HEAD_DIM // 4),
            (K_STRIDE_I32, 1),
        )

        def kv_load_coords(batch):
            # FlyDSL 0.3.4.1's static idx2crd/crd2idx helpers require an MLIR
            # index coordinate. This is the only fx.Index bridge: Int64 would
            # hide the same cast inside the helper. Replacing the helper with
            # manual u64 div/rem was correct but regressed Flux-1536 by ~4.5%
            # in the five-shape benchmark, so retain this measured boundary.
            linear_chunk = fx.Index(tid) + batch * BLOCK_SIZE
            lds_row, load_lane = idx2crd(linear_chunk, k_load_layout)
            load_col_base = load_lane * VEC_WIDTH
            load_col_i32 = load_lane * (VEC_WIDTH // 4)
            return linear_chunk, lds_row, load_col_base, load_col_i32

        def _store_k_packed(lds_row, load_col_i32, v4):
            """Write one 16-byte K load as the four packed WMMA LDS words."""
            lds_i32_idx = crd2idx((lds_row, load_col_i32), k_lds_i32_layout)
            # Scalar stores preserve the tuned LDS instruction sequence.
            for wi in range_constexpr(4):
                fx.ptr_store(
                    Vec(v4)[wi],
                    lds_i32_ptr + fx.Int32(lds_i32_idx + wi),
                )

        def coop_load_k(tile_start):
            """Cooperatively stage one row-major K tile in packed i32 LDS."""
            # The frontend represents ``range`` induction variables as index;
            # normalize at this boundary before combining with u64 coordinates.
            tile_start = fx.Uint64(tile_start)
            for batch in range_constexpr(NUM_BATCHES_KV):
                linear_chunk, lds_row, load_col_base, load_col_i32 = kv_load_coords(
                    batch
                )
                row_idx = tile_start + fx.Uint64(lds_row)
                if const_expr(KV_NEEDS_GUARD):
                    chunk_valid = linear_chunk < NUM_KV_CHUNKS
                    if chunk_valid:
                        g_idx = kv_global_idx(row_idx, fx.Uint64(load_col_base))
                        v4 = _load_global_fp8(
                            k_buf,
                            g_idx,
                            elem=fx.Int32,
                            unit_elems=4,
                            dword_index=True,
                        )
                        _store_k_packed(lds_row, load_col_i32, v4)
                else:
                    g_idx = kv_global_idx(row_idx, fx.Uint64(load_col_base))
                    v4 = _load_global_fp8(
                        k_buf,
                        g_idx,
                        elem=fx.Int32,
                        unit_elems=4,
                        dword_index=True,
                    )
                    _store_k_packed(lds_row, load_col_i32, v4)

        def _store_v_transposed_fp8(lds_row, load_col_base, v_bytes):
            # fp8-input: V already fp8 bytes (v16i8) — no convert, just scatter-store
            # TRANSPOSED (V_T[d][kv_row]). 16 d-values of this lane land in 16 d-rows
            # at the same kv column (stride V_T_STRIDE_BYTES). Makes GEMM2 load contiguous.
            # The strided scatter has no equivalent contiguous vector-store form.
            lds_row = fx.Uint64(lds_row)
            load_col_base = fx.Uint64(load_col_base)
            for j in range_constexpr(VEC_WIDTH):
                d_col = load_col_base + fx.Uint64(j)
                byte_idx = (
                    fx.Uint64(V_BYTE_BASE)
                    + d_col * fx.Uint64(V_T_STRIDE_BYTES)
                    + lds_row
                )
                fx.ptr_store(v_bytes[j], lds_i8_ptr + fx.Int32(byte_idx))

        def _load_v_transposed(st_kv_base, pks, d_chunk):
            """Read one contiguous packed V_T fragment for GEMM2."""
            d_pos = fx.Uint64(d_chunk * D_CHUNK) + lane16
            v_i32_idx = (
                fx.Uint64(V_BYTE_BASE // 4)
                + d_pos * fx.Uint64(V_T_STRIDE_I32)
                + fx.Uint64((st_kv_base + pks * PV_K_STEP) // 4)
                + klane * fx.Uint64(WMMA_LANE_K // 4)
            )
            return lds_i32_view(v_i32_idx, 2).load().ir_value()

        def coop_load_v_global(tile_start):
            # See coop_load_k: range induction variables are index-typed.
            tile_start = fx.Uint64(tile_start)
            vecs = []
            for batch in range_constexpr(NUM_BATCHES_KV):
                _, lds_row, load_col_base, _ = kv_load_coords(batch)
                lds_row_u64 = fx.Uint64(lds_row)
                load_col_base_u64 = fx.Uint64(load_col_base)
                if const_expr(KV_NEEDS_GUARD):
                    # Guard OOB global read: with BLOCK_SIZE>256 the extra load
                    # rows, including a partial final load batch, would read
                    # past the tile. Wrap into its valid row range; the value is
                    # discarded by the guarded LDS store.
                    safe_row = lds_row_u64 % fx.Uint64(BLOCK_N)
                    row_idx = tile_start + safe_row
                else:
                    row_idx = tile_start + lds_row_u64
                g_idx = kv_global_idx(row_idx, load_col_base_u64)
                vecs.append(
                    _load_global_fp8(
                        v_buf,
                        g_idx,
                        elem=fx.Int32,
                        unit_elems=4,
                        dword_index=True,
                    ).bitcast(fx.Int8)
                )
            return vecs

        def coop_store_v_lds(vecs):
            for batch in range_constexpr(NUM_BATCHES_KV):
                linear_chunk, lds_row, load_col_base, _ = kv_load_coords(batch)
                if const_expr(KV_NEEDS_GUARD):
                    chunk_valid = linear_chunk < NUM_KV_CHUNKS
                    if chunk_valid:
                        _store_v_transposed_fp8(lds_row, load_col_base, vecs[batch])
                else:
                    _store_v_transposed_fp8(lds_row, load_col_base, vecs[batch])

        # ---- Q preload ----
        q_row = q_start + wave_q_offset + lane16
        q_row_i32 = fx.Int32(q_row)
        q_in_bounds = q_row < seq_len_v
        q_row_safe = q_in_bounds.select(q_row, fx.Uint64(0))

        c_zero_v2i32_vec = Vec.filled(2, 0, fx.Int32).ir_value()
        q_b_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            q_col = fx.Uint64(ks * K_STEP_QK) + klane * fx.Uint64(WMMA_LANE_K)
            g_idx = global_idx(q_row_safe, q_col)
            # fp8-input: Q already fp8 — load 8 fp8 bytes as v2i32 WMMA-B frag direct.
            raw = _load_global_fp8(
                q_buf,
                g_idx,
                elem=fx.Int32,
                unit_elems=2,
                dword_index=True,
            )
            raw_safe = q_in_bounds.select(raw, c_zero_v2i32_vec)
            q_b_packs.append([_raw(Vec(raw_safe)[0]), _raw(Vec(raw_safe)[1])])

        # ---- Constants ----
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_sm_scale_log2e = fx.Float32(sm_scale * _LOG2E)
        # fp8-input: fold per-tensor q_scale*k_scale into the softmax log2e scale
        # (S = q_scale*k_scale * (Q_fp8 . K_fp8^T)). Runtime scalar, computed once.
        c_sm_scale_log2e_rt = _fmul_scalar(
            _fmul_scalar(q_scale, k_scale), c_sm_scale_log2e
        )
        c_zero_v8f32 = Vec.filled(8, 0.0, fx.Float32)
        _q_end = q_start + BLOCK_M
        if const_expr(CAUSAL):
            kv_upper = (_q_end < seq_len_v).select(_q_end, seq_len_v)
        else:
            kv_upper = seq_len_kv_real_v

        # ---- Pre-issue first V global load before the loop ----
        _v_vecs_init = coop_load_v_global(fx.Uint64(0))

        init_args = [_raw(c_neg_inf), _raw(c_zero_f)]
        for _ in range_constexpr(D_CHUNKS):
            init_args.append(_raw(c_zero_v8f32))
        # Carry V prefetch vecs as loop-carried values
        for batch in range_constexpr(NUM_BATCHES_KV):
            init_args.append(_v_vecs_init[batch])

        loop_results = init_args
        for kv_block_start, inner_iter_args in range(
            fx.Uint64(0), kv_upper, fx.Uint64(BLOCK_N), init=init_args
        ):
            # Keep the loop's index-typed induction variable out of coordinate
            # arithmetic; all K/V offsets below remain explicitly u64.
            kv_block_start_u64 = fx.Uint64(kv_block_start)
            m_running = inner_iter_args[0]
            l_running = inner_iter_args[1]
            o_accs = [inner_iter_args[2 + i] for i in range_constexpr(D_CHUNKS)]
            _v_vecs_prefetch = [
                inner_iter_args[2 + D_CHUNKS + b]
                for b in range_constexpr(NUM_BATCHES_KV)
            ]

            coop_load_k(kv_block_start_u64)
            gpu.barrier()

            # ==== GEMM1: S = K @ Q^T (fp8 WMMA) ====
            # K is read as v2i32 (ds_read_b64) from packed LDS; Q is a v2i32
            # register fragment. The typed WMMA atom consumes both fragments.
            s_accs = [_raw(c_zero_v8f32) for _ in range(NUM_S_ACCS)]

            for ks in range_constexpr(K_STEPS_QK):
                # k_col_i32 in i32 units: ks*K_STEP_QK fp8 elements / 4 fp8 per i32
                # + klane * WMMA_LANE_K fp8 per lane / 4 = klane * 2
                k_col_i32 = fx.Uint64(ks * K_STEP_QK // 4) + klane * fx.Uint64(
                    WMMA_LANE_K // 4
                )

                for st_idx in range_constexpr(N_SUB_TILES):
                    st_base_row = st_idx * K_SUB_N

                    k_row_a = lane16 + fx.Uint64(st_base_row)
                    k_lds_a_i32 = k_row_a * fx.Uint64(K_STRIDE_I32) + k_col_i32
                    k_pack_a_v2i32 = lds_i32_view(k_lds_a_i32, 2).load()

                    k_row_b = lane16 + fx.Uint64(st_base_row + 16)
                    k_lds_b_i32 = k_row_b * fx.Uint64(K_STRIDE_I32) + k_col_i32
                    k_pack_b_v2i32 = lds_i32_view(k_lds_b_i32, 2).load()

                    acc_idx_a = st_idx * 2
                    acc_idx_b = st_idx * 2 + 1
                    s_accs[acc_idx_a] = wmma_acc_fp8(
                        Vec(k_pack_a_v2i32).ir_value(),
                        q_b_packs[ks],
                        s_accs[acc_idx_a],
                    )
                    s_accs[acc_idx_b] = wmma_acc_fp8(
                        Vec(k_pack_b_v2i32).ir_value(),
                        q_b_packs[ks],
                        s_accs[acc_idx_b],
                    )

            s_raw = flatten_scores(s_accs, num_s_accs=NUM_S_ACCS)
            if const_expr(CAUSAL or TAIL_MASK):
                s_raw = mask_scores(
                    s_raw,
                    kv_block_start_u64,
                    klane,
                    q_row_i32,
                    seq_len_kv_real,
                    c_neg_inf,
                    num_s_accs=NUM_S_ACCS,
                    causal=CAUSAL,
                )
            p_vals, m_new_raw, l_new, o_accs = _update_online_softmax(
                s_raw,
                m_running,
                l_running,
                o_accs,
                c_sm_scale_log2e_rt,
                c_zero_f,
                num_scores=NUM_S_VALS,
                d_chunks=D_CHUNKS,
            )

            # Store V transposed as V_T[d][kv_row] for contiguous GEMM2 loads.
            coop_store_v_lds(_v_vecs_prefetch)
            gpu.barrier()

            # ==== Build P packs (fp8, pair list for wmma_acc_fp8) ====
            p_packs_all = []
            for st_idx in range_constexpr(N_SUB_TILES):
                p_packs_st = []
                for pks in range_constexpr(PV_K_STEPS):
                    acc_idx = st_idx * 2 + pks
                    p_base = acc_idx * 8
                    p_slice = [p_vals[p_base + j] for j in range(8)]
                    p_packs_st.append(_pack_probabilities_fp8(p_slice))
                p_packs_all.append(p_packs_st)

            # ==== GEMM2: O += V^T @ P (software pipelined, transposed LDS V) ====
            # Prefetch the next V pack while the current WMMA executes.

            # Software pipeline: preload first V pack
            cur_v_packs = []
            for st_idx in range_constexpr(N_SUB_TILES):
                cur_v_packs.append(_load_v_transposed(st_idx * K_SUB_N, 0, 0))

            for pks in range_constexpr(PV_K_STEPS):
                for dc in range_constexpr(D_CHUNKS):
                    next_dc = dc + 1
                    next_pks = pks
                    if const_expr(next_dc >= D_CHUNKS):
                        next_dc = 0
                        next_pks = pks + 1
                    has_next = const_expr(next_pks < PV_K_STEPS)

                    # Prefetch next V while current WMMA runs
                    next_v_packs = []
                    if const_expr(has_next):
                        for st_idx in range_constexpr(N_SUB_TILES):
                            next_v_packs.append(
                                _load_v_transposed(st_idx * K_SUB_N, next_pks, next_dc)
                            )

                    for st_idx in range_constexpr(N_SUB_TILES):
                        o_accs[dc] = wmma_acc_fp8(
                            cur_v_packs[st_idx], p_packs_all[st_idx][pks], o_accs[dc]
                        )

                    if const_expr(has_next):
                        cur_v_packs = next_v_packs

            m_running = m_new_raw
            l_running = l_new

            # ---- Issue the next iteration's V global load ----
            safe_next_kv_start = _next_kv_tile_start(
                kv_block_start_u64,
                kv_upper,
                fx.Uint64(BLOCK_N),
                fx.Uint64(0),
            )
            _v_vecs_next = coop_load_v_global(safe_next_kv_start)

            _yield_args = [m_running, l_running] + o_accs
            for batch in range_constexpr(NUM_BATCHES_KV):
                _yield_args.append(_v_vecs_next[batch])
            loop_results = yield _yield_args

        # ---- Normalize and store O ----
        l_final = loop_results[1]
        o_finals = [loop_results[2 + dc] for dc in range_constexpr(D_CHUNKS)]

        # fp8-input: O = v_scale * (P . V_fp8); fold v_scale into the 1/l
        # normalizer. The launch compile hints provide the FP fast-math policy.
        inv_l = c_one_f / fx.Float32(l_final)
        inv_l = _fmul_scalar(inv_l, v_scale)
        inv_l_vec = Vec.from_elements([inv_l], fx.Float32).broadcast_to(8).ir_value()

        if q_in_bounds:
            for dc in range_constexpr(D_CHUNKS):
                o_norm_vec = _fmul_vec(o_finals[dc], inv_l_vec)
                o_trunc = Vec(o_norm_vec).to(elem_dtype).ir_value()
                d_col = fx.Uint64(dc * D_CHUNK) + klane * fx.Uint64(8)
                o_global = global_idx(q_row, d_col)
                _store_output_vec(o_global, o_trunc)

    @flyc.jit
    def launch_flash_attn_func(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        seq_len_kv_real: fx.Int32,
        seq_len_kv: fx.Int32,
        q_scale_ptr: fx.Tensor,
        k_scale_ptr: fx.Tensor,
        v_scale_ptr: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        batch_size_u64 = fx.Uint64(batch_size)
        seq_len_u64 = fx.Uint64(seq_len)
        num_q_tiles = (seq_len_u64 + BLOCK_M - 1) // BLOCK_M
        grid_x = batch_size_u64 * num_q_tiles * NUM_HEADS

        flash_attn_func_kernel._func.__name__ = KERNEL_NAME
        passthrough_entries = (
            [
                ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
                ["no-nans-fp-math", "true"],
                ["unsafe-fp-math", "true"],
            ]
            if const_expr(daz)
            else None
        )
        kernel_attrs = {
            "rocdl.waves_per_eu": waves_per_eu,
            "rocdl.flat_work_group_size": f"{flat_work_group_size},{flat_work_group_size}",
            "passthrough": passthrough_entries,
        }

        launcher = flash_attn_func_kernel(
            Q,
            K,
            V,
            O,
            seq_len,
            seq_len_kv_real,
            seq_len_kv,
            q_scale_ptr,
            k_scale_ptr,
            v_scale_ptr,
            value_attrs=kernel_attrs,
        )

        launcher.launch(grid=(grid_x, 1, 1), block=(BLOCK_SIZE, 1, 1), stream=stream)

    _fmha_compile_hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
        "llvm_options": {"enable-post-misched": False, "lsr-drop-solution": True},
    }

    launch_flash_attn_func.compile_hints = dict(_fmha_compile_hints)

    def _launch(*args, **kwargs):
        stream = kwargs.pop("stream", fx.Stream(None))
        _run_compiled(launch_flash_attn_func, *args, stream)

    def _compile(
        Q,
        K,
        V,
        O,
        batch_size,
        seq_len,
        seq_len_kv_real,
        seq_len_kv,
        q_scale=None,
        k_scale=None,
        v_scale=None,
        stream=None,
    ):
        # Scales are one-element device f32 tensors, not host floats.
        return flyc.compile(
            launch_flash_attn_func,
            Q,
            K,
            V,
            O,
            batch_size,
            seq_len,
            seq_len_kv_real,
            seq_len_kv,
            q_scale,
            k_scale,
            v_scale,
            fx.Stream(stream),
        )

    _launch.compile = _compile
    return _launch
