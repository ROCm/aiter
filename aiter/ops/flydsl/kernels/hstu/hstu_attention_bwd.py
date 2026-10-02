# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL KV-owned HSTU backward kernel for dV and dK.

Each program owns KV rows and streams query tiles. Both gradients reuse the same
score and gate fragments:

    Z = alpha * Q * K^T
    A = mask * silu(Z) / N
    dV = A^T * dO
    dZ = mask * silu'(Z) * (dO * V^T) / N
    dK = alpha * dZ^T * Q

Rows are single-writer. Inputs are f16 or bf16, accumulation is fp32, and score
math uses fast floating-point operations. `validate_hstu_attention_bwd` defines
the supported dimensions, tile shapes, masks, and architectures.
"""

import functools
import math as host_math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch

from aiter.ops.flydsl.kernels.hstu.hstu_attention_common import (
    _LOG2E,
    MFMA_ELEMS_PER_LANE,
    MFMA_K,
    MFMA_LANE_K,
    MFMA_M,
    MFMA_N,
    NUM_GRID_GROUPS,
    WARP_SIZE,
    _arch_dma_params,
    _dtype_to_elem_type,
    _mfma_params_for_dim,
    bind_mfma_accs,
    decode_lane,
    exp2_f32,
    grouped_loader,
    lds_cap_bytes,
    make_lds_dma,
    pack_mfma_frag,
    swz_col,
)


def validate_hstu_attention_bwd(
    num_heads: int,
    head_dim: int,
    hidden_dim: int,
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
        raise ValueError(
            f"hstu attention bwd unsupported arch: {arch!r} (expected 'gfx942' or 'gfx950')"
        )

    if dtype_str not in {"f16", "bf16"}:
        raise ValueError(f"unsupported dtype: {dtype_str!r} (expected 'f16' or 'bf16')")
    if not causal:
        raise ValueError("hstu_attention_bwd only supports causal attention")
    if contextual_seq_len < 0:
        raise ValueError(
            f"contextual_seq_len must be non-negative, got {contextual_seq_len}"
        )
    if max_attn_len < 0:
        raise ValueError(f"max_attn_len must be non-negative, got {max_attn_len}")

    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
    if not host_math.isfinite(alpha):
        raise ValueError(f"alpha must be finite, got {alpha}")
    mfma_qk_k, _, _ = _mfma_params_for_dim(head_dim, arch)
    if head_dim <= 0 or head_dim % mfma_qk_k != 0:
        raise ValueError(
            f"head_dim must be positive and a multiple of MFMA_K={mfma_qk_k}, got {head_dim}"
        )
    mfma_da_k, _, _ = _mfma_params_for_dim(hidden_dim, arch)
    if hidden_dim <= 0 or hidden_dim % mfma_da_k != 0:
        raise ValueError(
            f"hidden_dim must be positive and a multiple of MFMA_K={mfma_da_k}, got {hidden_dim}"
        )

    if block_m <= 0:
        raise ValueError(f"block_m must be positive, got {block_m}")
    if block_n <= 0:
        raise ValueError(f"block_n must be positive, got {block_n}")
    if num_waves <= 0:
        raise ValueError(f"num_waves must be positive, got {num_waves}")
    if waves_per_eu < 0:
        raise ValueError(f"waves_per_eu must be non-negative, got {waves_per_eu}")
    if block_m % (num_waves * MFMA_M) != 0:
        raise ValueError(
            f"block_m {block_m} must be a multiple of num_waves*MFMA_M ({num_waves * MFMA_M})"
        )
    if block_n % MFMA_M != 0:
        raise ValueError(f"block_n {block_n} must be a multiple of MFMA_M={MFMA_M}")

    _, dma_elems, _, _ = _arch_dma_params(arch)
    block_threads = num_waves * WARP_SIZE
    elems_per_dma_pass = block_threads * dma_elems
    head_dim_k = ((head_dim + 63) // 64) * 64
    # Streamed Q tile staged through LDS: [BLOCK_N, head_dim_k].
    if (block_n * head_dim_k) % elems_per_dma_pass != 0:
        raise ValueError("Q DMA tile does not divide the dword DMA pass evenly")
    # Streamed dO tile: [BLOCK_N, hidden_dim].
    if (block_n * hidden_dim) % elems_per_dma_pass != 0:
        raise ValueError("dO DMA tile does not divide the dword DMA pass evenly")

    vec_v = (
        8
        if (hidden_dim % 8 == 0 and (block_n * hidden_dim) % (block_threads * 8) == 0)
        else dma_elems
    )
    threads_per_row_v = hidden_dim // vec_v
    if block_threads % threads_per_row_v != 0:
        raise ValueError(
            f"block_threads={block_threads} must be divisible by threads_per_row_v={threads_per_row_v}"
        )
    rows_per_batch_v = block_threads // threads_per_row_v
    if not (block_n % rows_per_batch_v == 0 or rows_per_batch_v > block_n):
        raise ValueError(
            f"rows_per_batch_v={rows_per_batch_v} must divide block_n={block_n}, unless rows_per_batch_v > block_n"
        )

    lds_cap = lds_cap_bytes(arch)
    lds_bytes = block_n * head_dim_k * 2 + block_n * hidden_dim * 2
    if lds_bytes > lds_cap:
        raise ValueError(f"LDS tile {lds_bytes} B exceeds the {lds_cap} B budget")


@functools.lru_cache(maxsize=16384)
def build_hstu_attention_bwd_dvdk(
    num_heads: int,
    head_dim: int,
    hidden_dim: int,
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
    num_waves: int = 4,
    waves_per_eu: int = 0,
    has_perm: bool = False,
):
    validate_hstu_attention_bwd(
        num_heads,
        head_dim,
        hidden_dim,
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

    BLOCK_M = block_m
    BLOCK_N = block_n
    NUM_WAVES = num_waves
    BLOCK_THREADS = NUM_WAVES * WARP_SIZE
    ROWS_PER_WAVE = BLOCK_M // NUM_WAVES
    KV_OWNED_SUBTILES = ROWS_PER_WAVE // MFMA_M
    Q_STREAM_SUBTILES = BLOCK_N // MFMA_N
    WAVES_PER_EU = waves_per_eu
    MFMA_QK_K, MFMA_QK_LANE_K, _ = _mfma_params_for_dim(head_dim)
    MFMA_DA_K, MFMA_DA_LANE_K, _ = _mfma_params_for_dim(hidden_dim)

    DMA_BYTES, DMA_ELEMS, K_SWZ_ROWS, K_SWZ_SHIFT = _arch_dma_params()

    elem_dtype = _dtype_to_elem_type(dtype_str)
    is_bf16 = dtype_str == "bf16"
    has_window = max_attn_len > 0
    has_contextual = contextual_seq_len > 0

    K_STEPS = head_dim // MFMA_QK_K
    HEAD_DIM_K = ((head_dim + 63) // 64) * 64
    K_STEPS_K = HEAD_DIM_K // MFMA_QK_K
    D_CHUNKS = hidden_dim // MFMA_M
    DK_STEPS = hidden_dim // MFMA_DA_K
    HC_CHUNKS = head_dim // MFMA_M

    num_kv_tiles = (max_seq_len + BLOCK_M - 1) // BLOCK_M
    # `hz` is a flattened (batch, head) index. Its total and per-group ceiling
    # stay runtime values so one binary serves every batch size.
    stride_qk_n = num_heads * head_dim

    Q_STRIDE = HEAD_DIM_K
    # Columns in [head_dim, HEAD_DIM_K) have no backing element in this head, so the
    # Q DMA source needs clamping (see async_load_q). Only live when padded; a
    # 64-aligned head_dim makes it compile-time false and emits no compare/select.
    Q_COL_GUARD = head_dim < HEAD_DIM_K
    DO_STRIDE = hidden_dim

    q_tile_elems = BLOCK_N * Q_STRIDE
    elems_per_dma_pass = BLOCK_THREADS * DMA_ELEMS
    assert q_tile_elems % elems_per_dma_pass == 0
    NUM_DMA_Q = q_tile_elems // elems_per_dma_pass
    PAIRS_PER_ROW_Q = Q_STRIDE // DMA_ELEMS

    do_tile_elems = BLOCK_N * hidden_dim
    assert do_tile_elems % elems_per_dma_pass == 0
    # dO global->LDS DMA: row-major [q, d], no swizzle (matches the dO LDS read
    # layout). Mirrors the Q DMA sans swizzle.
    NUM_DMA_DO = do_tile_elems // elems_per_dma_pass
    PAIRS_PER_ROW_DO = DO_STRIDE // DMA_ELEMS

    N_ACC_DV = D_CHUNKS * KV_OWNED_SUBTILES
    N_ACC_DK = HC_CHUNKS * KV_OWNED_SUBTILES
    N_ACC = N_ACC_DV + N_ACC_DK

    # Row-major tiles feed the score and dA MFMAs. Transposed copies make the
    # query-reduction operands contiguous, replacing four scalar LDS reads with
    # one packed read. The workgroup builds each copy once after the async DMA.
    # Padding spreads adjacent transpose rows across LDS banks.
    T_PAD = MFMA_LANE_K
    DO_T_STRIDE = BLOCK_N + T_PAD
    Q_T_STRIDE = BLOCK_N + T_PAD

    # Keep total LDS at or below 32 KiB so two workgroups fit in a 64 KiB budget.
    # Larger tiles use scalar gathers instead of transposed copies.
    T_LDS_BUDGET = 32768
    _lds_row_major = (BLOCK_N * Q_STRIDE + BLOCK_N * DO_STRIDE) * 2
    _lds_transposed = (head_dim * Q_T_STRIDE + hidden_dim * DO_T_STRIDE) * 2
    USE_T_TILES = (_lds_row_major + _lds_transposed) <= T_LDS_BUDGET
    # A single-row placeholder keeps the LDS views valid when transposes are disabled.
    T_Q_ROWS = head_dim if USE_T_TILES else 1
    T_DO_ROWS = hidden_dim if USE_T_TILES else 1

    # Build work is one unit per (row, 4-wide q group), spread over the workgroup.
    assert BLOCK_N % MFMA_LANE_K == 0
    T_QGROUPS = BLOCK_N // MFMA_LANE_K
    DO_T_UNITS = hidden_dim * T_QGROUPS
    Q_T_UNITS = head_dim * T_QGROUPS
    NUM_DO_T_PASSES = (DO_T_UNITS + BLOCK_THREADS - 1) // BLOCK_THREADS
    NUM_Q_T_PASSES = (Q_T_UNITS + BLOCK_THREADS - 1) // BLOCK_THREADS
    DO_T_EXACT = DO_T_UNITS % BLOCK_THREADS == 0
    Q_T_EXACT = Q_T_UNITS % BLOCK_THREADS == 0

    # LDS stores row-major Q/dO for score and dA, plus unswizzled transposes for
    # dK/dV. Q is column-swizzled to distribute its row-major MFMA reads.
    @fx.struct
    class SharedStorage:
        q: fx.Array[elem_dtype, BLOCK_N * Q_STRIDE, 16]
        do: fx.Array[elem_dtype, BLOCK_N * DO_STRIDE, 16]
        q_t: fx.Array[elem_dtype, T_Q_ROWS * Q_T_STRIDE, 16]
        do_t: fx.Array[elem_dtype, T_DO_ROWS * DO_T_STRIDE, 16]

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def hstu_attention_bwd_dvdk(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        do: fx.Tensor,
        seq_offsets: fx.Tensor,
        num_targets: fx.Tensor,
        perm: fx.Tensor,
        out_dv: fx.Tensor,
        out_dk: fx.Tensor,
        hz_per_group: fx.Int32,
        hz_total: fx.Int32,
    ) -> None:
        elem_type = elem_dtype.ir_type
        c_zero_qk_pack = Vec.filled(MFMA_QK_LANE_K, 0.0, elem_dtype).ir_value()
        c_zero_da_pack = Vec.filled(MFMA_DA_LANE_K, 0.0, elem_dtype).ir_value()

        # QK and V*dO use architecture-native dimension-axis MFMAs. The dV/dK
        # reductions stay 16-deep to reuse score fragments without lane shuffles.
        # Matching MFMA shapes share one accumulator binding.
        qk_mfma_acc, da_mfma_acc, mfma_acc = bind_mfma_accs(
            elem_dtype,
            (MFMA_QK_K, MFMA_QK_LANE_K),
            (MFMA_DA_K, MFMA_DA_LANE_K),
            (MFMA_K, MFMA_LANE_K),
        )

        tid = fx.Int32(gpu.thread_idx.x)
        wave_id, _lane, lane_div_16, lane_mod_16 = decode_lane(
            tid, NUM_WAVES, WARP_SIZE, MFMA_N
        )

        block_id = fx.Int32(gpu.block_idx.x)
        grid_group = block_id % fx.Int32(NUM_GRID_GROUPS)
        pos_in_group = block_id // fx.Int32(NUM_GRID_GROUPS)
        local_hz_idx = pos_in_group // fx.Int32(num_kv_tiles)
        kv_tile_idx = pos_in_group % fx.Int32(num_kv_tiles)
        hz_idx = grid_group * hz_per_group + local_hz_idx
        # hz_per_group is a ceil, so the padded tail of the last group runs past
        # batch*num_heads. Those blocks clamp to hz_idx=0 to keep the seq_offsets /
        # perm / num_targets reads in bounds, then take seq_len=0 below: every KV tile
        # is then inactive, so they stream no query tiles and store nothing (rows past
        # seq_len belong to the next sequence in the packed layout).
        block_valid = hz_idx < hz_total
        hz_idx = block_valid.select(hz_idx, fx.Int32(0))
        batch_idx = hz_idx // fx.Int32(num_heads)
        head_idx = hz_idx % fx.Int32(num_heads)

        # Remap batches so each grid group receives balanced quadratic sequence work.
        if const_expr(has_perm):
            batch_idx = fx.Int32(perm[batch_idx])

        seq_start = fx.Int32(seq_offsets[batch_idx])
        seq_len = fx.Int32(seq_offsets[batch_idx + fx.Int32(1)]) - seq_start
        seq_len = block_valid.select(seq_len, fx.Int32(0))

        num_target = fx.Int32(0)
        if has_targets:
            num_target = fx.Int32(num_targets[batch_idx])
        max_id = seq_len
        if has_contextual:
            max_id = seq_len - fx.Int32(contextual_seq_len) + fx.Int32(1)
        if has_targets:
            max_id = (num_target > fx.Int32(0)).select(max_id - num_target, max_id)

        def to_id(x):
            xid = x
            if has_contextual:
                xid = xid - fx.Int32(contextual_seq_len - 1)
                xid = (xid < fx.Int32(0)).select(fx.Int32(0), xid)
            if has_targets:
                xid = (xid > max_id).select(max_id, xid)
            return xid

        k_load = grouped_loader(k, head_dim, MFMA_QK_LANE_K)
        v_load = grouped_loader(v, hidden_dim, MFMA_DA_LANE_K)

        q_head_offset = head_idx * fx.Int32(head_dim)
        q_base_byte_offset = (
            fx.Int64(seq_start) * fx.Int64(stride_qk_n) + fx.Int64(q_head_offset)
        ) * fx.Int64(2)

        # The trailing layout axis carries each row's packed MFMA operand.
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        q_view = lds.q.view(
            fx.make_layout(
                (BLOCK_N, Q_STRIDE // MFMA_QK_LANE_K, MFMA_QK_LANE_K),
                (Q_STRIDE, MFMA_QK_LANE_K, 1),
            )
        )
        do_view = lds.do.view(
            fx.make_layout(
                (BLOCK_N, DO_STRIDE // MFMA_DA_LANE_K, MFMA_DA_LANE_K),
                (DO_STRIDE, MFMA_DA_LANE_K, 1),
            )
        )
        # Transposed tiles, grouped along q by the accum-GEMM pack width so a
        # B-operand pack is view[row, q_grp, None].load() -- a single ds_read_b64.
        q_t_view = lds.q_t.view(
            fx.make_layout(
                (T_Q_ROWS, Q_T_STRIDE // MFMA_LANE_K, MFMA_LANE_K),
                (Q_T_STRIDE, MFMA_LANE_K, 1),
            )
        )
        do_t_view = lds.do_t.view(
            fx.make_layout(
                (T_DO_ROWS, DO_T_STRIDE // MFMA_LANE_K, MFMA_LANE_K),
                (DO_T_STRIDE, MFMA_LANE_K, 1),
            )
        )
        q_lds_byte_base = fx.ptrtoint(fx.get_iter(q_view))

        # Direct dO global->LDS DMA. dO is [L, H, hidden],
        # so the per-token stride is num_heads*hidden_dim; base at this head's slice.
        stride_do_n = num_heads * hidden_dim
        do_base_byte_offset = (
            fx.Int64(seq_start) * fx.Int64(stride_do_n)
            + fx.Int64(head_idx) * fx.Int64(hidden_dim)
        ) * fx.Int64(2)
        do_lds_byte_base = fx.ptrtoint(fx.get_iter(do_view))

        # ── Copy-atom global->LDS DMA (buffer_load_lds via fx.copy) ──
        _dma_atom, _lds_ptr_ty, _rebased_buffer_div = make_lds_dma(DMA_BYTES, elem_type)

        q_div = _rebased_buffer_div(
            fx.get_iter(q), q_base_byte_offset, max_seq_len * stride_qk_n
        )
        do_div = _rebased_buffer_div(
            fx.get_iter(do), do_base_byte_offset, max_seq_len * stride_do_n
        )

        def q_swz_col(tile_row, col):
            return swz_col(tile_row, col, K_SWZ_ROWS, K_SWZ_SHIFT)

        kv_wave_base = kv_tile_idx * fx.Int32(BLOCK_M) + wave_id * fx.Int32(
            ROWS_PER_WAVE
        )

        kv_rows = []
        kv_in_bounds = []
        for og in range_constexpr(KV_OWNED_SUBTILES):
            local = kv_wave_base + fx.Int32(og * MFMA_M) + lane_mod_16
            kv_rows.append(local)
            kv_in_bounds.append(local < seq_len)

        k_packs = []
        for ks in range_constexpr(K_STEPS):
            k_col = fx.Int32(ks * MFMA_QK_K) + lane_div_16 * fx.Int32(MFMA_QK_LANE_K)
            per_og = []
            for og in range_constexpr(KV_OWNED_SUBTILES):
                safe = kv_in_bounds[og].select(seq_start + kv_rows[og], seq_start)
                raw = k_load(
                    fx.Int64(safe), head_idx, k_col // fx.Int32(MFMA_QK_LANE_K)
                ).ir_value()
                per_og.append(kv_in_bounds[og].select(raw, c_zero_qk_pack))
            k_packs.append(per_og)

        v_packs = []
        for ks in range_constexpr(DK_STEPS):
            v_col = fx.Int32(ks * MFMA_DA_K) + lane_div_16 * fx.Int32(MFMA_DA_LANE_K)
            per_og = []
            for og in range_constexpr(KV_OWNED_SUBTILES):
                safe = kv_in_bounds[og].select(seq_start + kv_rows[og], seq_start)
                raw = v_load(
                    fx.Int64(safe), head_idx, v_col // fx.Int32(MFMA_DA_LANE_K)
                ).ir_value()
                per_og.append(kv_in_bounds[og].select(raw, c_zero_da_pack))
            v_packs.append(per_og)

        c_alpha = fx.Float32(alpha)
        c_inv_n = fx.Float32(1.0 / max_seq_len)
        c_neg_log2e = fx.Float32(-_LOG2E)
        c_one_f = fx.Float32(1.0)
        c_neg_one_f = fx.Float32(-1.0)
        c_zero_f = fx.Float32(0.0)

        def silu_and_grad_batch(s_list):
            # Fast (non-IEEE) SiLU + derivative on fp32 lanes. The fastmath context
            # gives every add/mul the `fast` flag and turns the reciprocal into
            # v_rcp_f32, so no rocdl builder is needed there.
            with arith.fastmath(arith.FastMathFlags.fast):
                sc = [s * c_alpha for s in s_list]
                tt = [s * c_neg_log2e for s in sc]
                emu = [exp2_f32(t) for t in tt]
                den = [c_one_f + e for e in emu]
                sig = [c_one_f / d for d in den]
                silu = [sc[i] * sig[i] for i in range(len(s_list))]
                grad = [
                    sig[i] * (c_one_f + sc[i] * (c_one_f + c_neg_one_f * sig[i]))
                    for i in range(len(s_list))
                ]
            return silu, grad

        kv_owned_ids = [to_id(kv_rows[og]) for og in range_constexpr(KV_OWNED_SUBTILES)]

        kv_start_row = kv_tile_idx * fx.Int32(BLOCK_M)
        kv_end_row = kv_start_row + fx.Int32(BLOCK_M)
        active = kv_start_row < seq_len

        # Causal queries start at the owned KV tile. A local window caps the upper
        # query bound. Target rows share `max_id`, so the cap reopens to `seq_len`
        # when that effective ID sees this tile. Contextual masks keep the full
        # range because raw positions do not describe contextual prefix rows.
        q_upper = seq_len
        if has_window and not has_contextual:
            win_upper = kv_end_row + fx.Int32(max_attn_len)
            if has_targets:
                win_upper = (win_upper <= max_id).select(win_upper, seq_len)
            q_upper = (win_upper < seq_len).select(win_upper, seq_len)
        q_upper = active.select(q_upper, fx.Int32(0))
        n_q_tiles = (q_upper + fx.Int32(BLOCK_N - 1)) // fx.Int32(BLOCK_N)
        q_tile_start = kv_start_row // fx.Int32(BLOCK_N)
        if has_contextual:
            q_tile_start = fx.Int32(0)

        c_zero_v4f32 = Vec.filled(MFMA_ELEMS_PER_LANE, 0.0, fx.Float32).ir_value()

        c_dma_elems = fx.Int32(DMA_ELEMS)
        c_pairs_per_row_q = fx.Int32(PAIRS_PER_ROW_Q)

        wave_lds_base_q = fx.Int32(q_lds_byte_base) + fx.Int32(wave_id) * fx.Int32(
            WARP_SIZE * DMA_BYTES
        )
        # DMA needs a wave-uniform LDS base in an SGPR. FlyDSL has no stable
        # exported wrapper for readfirstlane.
        wave_lds_lane0_q = rocdl.readfirstlane(fx.Int32.ir_type, wave_lds_base_q)
        q_dma_rows = []
        q_dma_gcols = []
        q_dma_col_ok = []
        for d in range_constexpr(NUM_DMA_Q):
            pair = tid + fx.Int32(d * BLOCK_THREADS)
            row = pair // c_pairs_per_row_q
            col_pair = pair % c_pairs_per_row_q
            col = col_pair * c_dma_elems
            row_gcol = q_swz_col(row, col)
            q_dma_rows.append(row)
            q_dma_gcols.append(row_gcol)
            if const_expr(Q_COL_GUARD):
                # The swizzle XORs within a 64-column block above the DMA granule, so
                # the fetched column stays DMA_ELEMS-aligned and head_dim % MFMA_K == 0
                # means an in-range start never straddles head_dim. Loop-invariant in
                # q_start, so hoisted out of the query sweep.
                q_dma_col_ok.append(row_gcol < fx.Int32(head_dim))

        c_stride_qk_n = fx.Int32(stride_qk_n)

        def async_load_q(q_start):
            for d in range_constexpr(NUM_DMA_Q):
                row = q_dma_rows[d]
                in_bounds = (q_start + row) < seq_len
                local_tok = in_bounds.select(q_start + row, fx.Int32(0))
                src_elem = local_tok * c_stride_qk_n + q_dma_gcols[d]
                if const_expr(Q_COL_GUARD):
                    # A pad column would index into the next head, and on the last
                    # token/head past the tensor entirely -- the rebased descriptor
                    # carries a 4 GiB bound, so the hardware does not clamp it. Fold
                    # those lanes onto element 0. compute_s_tile pairs every pad
                    # contraction step with a zero K operand, so the value never
                    # reaches an output; it only has to be finite, since 0 * NaN
                    # would poison S.
                    src_elem = q_dma_col_ok[d].select(src_elem, fx.Int32(0))
                lds_byte = fx.Int32(wave_lds_lane0_q) + fx.Int32(
                    d * BLOCK_THREADS * DMA_BYTES
                )
                dst = fx.make_view(
                    fx.inttoptr(_lds_ptr_ty, lds_byte), fx.make_layout(1, 1)
                )
                src = fx.slice(q_div, (None, fx.Int32(src_elem)))
                fx.copy(_dma_atom, src, dst)

        # Out-of-range rows read token zero; masking removes them before accumulation.
        c_stride_do_n = fx.Int32(stride_do_n)
        wave_lds_base_do = fx.Int32(do_lds_byte_base) + fx.Int32(wave_id) * fx.Int32(
            WARP_SIZE * DMA_BYTES
        )
        wave_lds_lane0_do = rocdl.readfirstlane(fx.Int32.ir_type, wave_lds_base_do)
        do_dma_rows = []
        do_dma_cols = []
        for d in range_constexpr(NUM_DMA_DO):
            pair = tid + fx.Int32(d * BLOCK_THREADS)
            do_dma_rows.append(pair // fx.Int32(PAIRS_PER_ROW_DO))
            do_dma_cols.append((pair % fx.Int32(PAIRS_PER_ROW_DO)) * c_dma_elems)

        def async_load_do_lds(q_start):
            for d in range_constexpr(NUM_DMA_DO):
                row = do_dma_rows[d]
                in_bounds = (q_start + row) < seq_len
                local_tok = in_bounds.select(q_start + row, fx.Int32(0))
                src_elem = local_tok * c_stride_do_n + do_dma_cols[d]
                lds_byte = fx.Int32(wave_lds_lane0_do) + fx.Int32(
                    d * BLOCK_THREADS * DMA_BYTES
                )
                dst = fx.make_view(
                    fx.inttoptr(_lds_ptr_ty, lds_byte), fx.make_layout(1, 1)
                )
                src = fx.slice(do_div, (None, fx.Int32(src_elem)))
                fx.copy(_dma_atom, src, dst)

        # One transpose unit covers one row and one MFMA-width query group.
        def _t_unit_decode(num_passes, units, exact):
            rows, qgs = [], []
            for p in range_constexpr(num_passes):
                unit = tid + fx.Int32(p * BLOCK_THREADS)
                if const_expr(not exact):
                    # Surplus lanes redo the last unit rather than branch: the
                    # write is idempotent (same address, same value).
                    unit = (unit >= fx.Int32(units)).select(fx.Int32(units - 1), unit)
                rows.append(unit // fx.Int32(T_QGROUPS))
                qgs.append(unit % fx.Int32(T_QGROUPS))
            return rows, qgs

        do_t_rows, do_t_qgs = (
            _t_unit_decode(NUM_DO_T_PASSES, DO_T_UNITS, DO_T_EXACT)
            if USE_T_TILES
            else ([], [])
        )
        q_t_rows, q_t_qgs = (
            _t_unit_decode(NUM_Q_T_PASSES, Q_T_UNITS, Q_T_EXACT)
            if USE_T_TILES
            else ([], [])
        )

        def build_transposed_tiles():
            """Build packed dO and Q transposes cooperatively in LDS."""
            for p in range_constexpr(NUM_DO_T_PASSES):
                d_row = do_t_rows[p]
                qg = do_t_qgs[p]
                q_base = qg * fx.Int32(MFMA_LANE_K)
                d_grp = d_row // fx.Int32(MFMA_DA_LANE_K)
                d_lane = d_row % fx.Int32(MFMA_DA_LANE_K)
                elems = [
                    do_view[q_base + fx.Int32(i), d_grp, d_lane]
                    for i in range_constexpr(MFMA_LANE_K)
                ]
                do_t_view[d_row, qg, None].store(Vec.from_elements(elems, elem_dtype))
            for p in range_constexpr(NUM_Q_T_PASSES):
                hc_row = q_t_rows[p]
                qg = q_t_qgs[p]
                q_base = qg * fx.Int32(MFMA_LANE_K)
                elems = []
                for i in range_constexpr(MFMA_LANE_K):
                    q_row = q_base + fx.Int32(i)
                    col = q_swz_col(q_row, hc_row)
                    elems.append(
                        q_view[
                            q_row,
                            col // fx.Int32(MFMA_QK_LANE_K),
                            col % fx.Int32(MFMA_QK_LANE_K),
                        ]
                    )
                q_t_view[hc_row, qg, None].store(Vec.from_elements(elems, elem_dtype))

        def read_q_packs(ng):
            q_row = fx.Int32(ng * MFMA_M) + lane_mod_16
            packs = []
            for ks in range_constexpr(K_STEPS_K):
                q_col = fx.Int32(ks * MFMA_QK_K) + lane_div_16 * fx.Int32(
                    MFMA_QK_LANE_K
                )
                # swz_col is MFMA_LANE_K-aligned, so //MFMA_LANE_K selects the packed
                # group and the trailing group axis carries the row stride.
                packs.append(
                    q_view[
                        q_row,
                        q_swz_col(q_row, q_col) // fx.Int32(MFMA_QK_LANE_K),
                        None,
                    ].load()
                )
            return packs

        def compute_s_tile(q_start, q_packs_by_ng):
            p_packs = [
                [None for _ in range_constexpr(KV_OWNED_SUBTILES)]
                for _ in range_constexpr(Q_STREAM_SUBTILES)
            ]
            s_meta = [
                [None for _ in range_constexpr(KV_OWNED_SUBTILES)]
                for _ in range_constexpr(Q_STREAM_SUBTILES)
            ]
            for ng in range_constexpr(Q_STREAM_SUBTILES):
                q_packs = [
                    Vec(q_packs_by_ng[ng][ks]) for ks in range_constexpr(K_STEPS_K)
                ]
                q_base = (
                    q_start
                    + fx.Int32(ng * MFMA_M)
                    + lane_div_16 * fx.Int32(MFMA_LANE_K)
                )
                q_raw = [
                    q_base + fx.Int32(i) for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                ]
                q_in_seq = [
                    q_raw[i] < seq_len for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                ]
                q_ids = [to_id(q_raw[i]) for i in range_constexpr(MFMA_ELEMS_PER_LANE)]
                for og in range_constexpr(KV_OWNED_SUBTILES):
                    cur = Vec.filled(MFMA_ELEMS_PER_LANE, 0.0, fx.Float32).ir_value()
                    for ks in range_constexpr(K_STEPS_K):
                        k_op = k_packs[ks][og] if ks < K_STEPS else c_zero_qk_pack
                        cur = qk_mfma_acc(q_packs[ks].ir_value(), k_op, cur)
                    s_vals = [Vec(cur)[i] for i in range_constexpr(MFMA_ELEMS_PER_LANE)]

                    def keep_row(i, og=og, q_ids=q_ids, q_raw=q_raw, q_in_seq=q_in_seq):
                        dist = q_ids[i] - kv_owned_ids[og]
                        keep = (q_raw[i] == kv_rows[og]) | (dist > fx.Int32(0))
                        if has_window:
                            keep = keep & (dist <= fx.Int32(max_attn_len))
                        if has_contextual:
                            ctx = (q_ids[i] == fx.Int32(0)) & (
                                kv_owned_ids[og] < max_id
                            )
                            keep = keep | ctx
                        keep = keep & q_in_seq[i] & kv_in_bounds[og]
                        return keep

                    keep = [keep_row(i) for i in range_constexpr(MFMA_ELEMS_PER_LANE)]
                    silu_vals, grad_vals = silu_and_grad_batch(s_vals)
                    p_vals = [
                        keep[i].select(silu_vals[i], c_zero_f)
                        for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                    ]
                    p_packs[ng][og] = pack_mfma_frag(p_vals, is_bf16, elem_dtype)
                    s_meta[ng][og] = (grad_vals, keep)
            return p_packs, s_meta

        def _dv_gather_t(c):
            # dO B-operand packs (4 adjacent q at a fixed d) for output chunk c,
            # one ds_read_b64 each out of dO^T.
            do_packs = []
            for ng in range_constexpr(Q_STREAM_SUBTILES):
                d_row = fx.Int32(c * MFMA_M) + lane_mod_16
                q_grp = fx.Int32(ng * (MFMA_M // MFMA_LANE_K)) + lane_div_16
                do_packs.append(do_t_view[d_row, q_grp, None].load().ir_value())
            return do_packs

        def _dv_gather_scalar(c):
            # Fallback when dO^T does not fit: 4x ds_read_u16 per pack.
            do_packs = []
            for ng in range_constexpr(Q_STREAM_SUBTILES):
                d_col = fx.Int32(c * MFMA_M) + lane_mod_16
                q_lane = fx.Int32(ng * MFMA_M) + lane_div_16 * fx.Int32(MFMA_LANE_K)
                d_grp = d_col // fx.Int32(MFMA_DA_LANE_K)
                d_lane = d_col % fx.Int32(MFMA_DA_LANE_K)
                elems = [
                    do_view[q_lane + fx.Int32(i), d_grp, d_lane]
                    for i in range_constexpr(MFMA_LANE_K)
                ]
                do_packs.append(Vec.from_elements(elems, elem_dtype).ir_value())
            return do_packs

        _dv_gather = _dv_gather_t if USE_T_TILES else _dv_gather_scalar

        def accum_dv_tile(dv_acc, p_packs):
            # Prefetch next chunk's B-operand gather before consuming the current
            # chunk's MFMAs, so the ds_read latency overlaps the MFMA chain.
            do_cur = _dv_gather(0)
            for c in range_constexpr(D_CHUNKS):
                if const_expr(c + 1 < D_CHUNKS):
                    do_next = _dv_gather(c + 1)
                for og in range_constexpr(KV_OWNED_SUBTILES):
                    acc_off = c * KV_OWNED_SUBTILES + og
                    cur = dv_acc[acc_off]
                    for ng in range_constexpr(Q_STREAM_SUBTILES):
                        cur = mfma_acc(p_packs[ng][og], do_cur[ng], cur)
                    dv_acc[acc_off] = cur
                if const_expr(c + 1 < D_CHUNKS):
                    do_cur = do_next
            return dv_acc

        def read_do_a_packs(ng):
            q_row = fx.Int32(ng * MFMA_M) + lane_mod_16
            packs = []
            for ks in range_constexpr(DK_STEPS):
                d_col = fx.Int32(ks * MFMA_DA_K) + lane_div_16 * fx.Int32(
                    MFMA_DA_LANE_K
                )
                packs.append(
                    do_view[q_row, d_col // fx.Int32(MFMA_DA_LANE_K), None].load()
                )
            return packs

        def compute_ds_packs(s_meta):
            ds_packs = [
                [None for _ in range_constexpr(KV_OWNED_SUBTILES)]
                for _ in range_constexpr(Q_STREAM_SUBTILES)
            ]
            for ng in range_constexpr(Q_STREAM_SUBTILES):
                do_a = read_do_a_packs(ng)
                for og in range_constexpr(KV_OWNED_SUBTILES):
                    cur = Vec.filled(MFMA_ELEMS_PER_LANE, 0.0, fx.Float32).ir_value()
                    for ks in range_constexpr(DK_STEPS):
                        cur = da_mfma_acc(do_a[ks].ir_value(), v_packs[ks][og], cur)
                    da_vals = [
                        Vec(cur)[i] for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                    ]
                    grad_vals, keep = s_meta[ng][og]
                    ds_vals = []
                    with arith.fastmath(arith.FastMathFlags.fast):
                        for i in range_constexpr(MFMA_ELEMS_PER_LANE):
                            gated = c_inv_n * grad_vals[i] * da_vals[i]
                            ds_vals.append(keep[i].select(gated, c_zero_f))
                    ds_packs[ng][og] = pack_mfma_frag(ds_vals, is_bf16, elem_dtype)
            return ds_packs

        def _dk_gather_t(c):
            # Q B-operand packs (4 adjacent q at a fixed hc) for output chunk c,
            # one ds_read_b64 each out of Q^T. The swizzle is absorbed into the
            # transpose build, so no q_swz_col here.
            qb_packs = []
            for ng in range_constexpr(Q_STREAM_SUBTILES):
                hc_row = fx.Int32(c * MFMA_M) + lane_mod_16
                q_grp = fx.Int32(ng * (MFMA_M // MFMA_LANE_K)) + lane_div_16
                qb_packs.append(q_t_view[hc_row, q_grp, None].load().ir_value())
            return qb_packs

        def _dk_gather_scalar(c):
            # Fallback when Q^T does not fit: scalar-gather from the *streamed*
            # swizzled Q LDS view (col -> group col//MFMA_LANE_K, lane
            # col%MFMA_LANE_K), reusing GEMM1's Q.
            qb_packs = []
            for ng in range_constexpr(Q_STREAM_SUBTILES):
                hc_col = fx.Int32(c * MFMA_M) + lane_mod_16
                q_lane = fx.Int32(ng * MFMA_M) + lane_div_16 * fx.Int32(MFMA_LANE_K)
                elems = []
                for i in range_constexpr(MFMA_LANE_K):
                    q_row = q_lane + fx.Int32(i)
                    col = q_swz_col(q_row, hc_col)
                    elems.append(
                        q_view[
                            q_row,
                            col // fx.Int32(MFMA_QK_LANE_K),
                            col % fx.Int32(MFMA_QK_LANE_K),
                        ]
                    )
                qb_packs.append(Vec.from_elements(elems, elem_dtype).ir_value())
            return qb_packs

        _dk_gather = _dk_gather_t if USE_T_TILES else _dk_gather_scalar

        def accum_dk_tile(dk_acc, ds_packs):
            qb_cur = _dk_gather(0)
            for c in range_constexpr(HC_CHUNKS):
                if const_expr(c + 1 < HC_CHUNKS):
                    qb_next = _dk_gather(c + 1)
                for og in range_constexpr(KV_OWNED_SUBTILES):
                    acc_off = c * KV_OWNED_SUBTILES + og
                    cur = dk_acc[acc_off]
                    for ng in range_constexpr(Q_STREAM_SUBTILES):
                        cur = mfma_acc(ds_packs[ng][og], qb_cur[ng], cur)
                    dk_acc[acc_off] = cur
                if const_expr(c + 1 < HC_CHUNKS):
                    qb_cur = qb_next
            return dk_acc

        def run_q_tile(acc, q_start):
            # DMA both Q and dO global->LDS, then one workgroup barrier publishes both.
            async_load_q(q_start)
            async_load_do_lds(q_start)
            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            # Defer the publish barrier so transpose writes overlap the score MFMAs.
            if const_expr(USE_T_TILES):
                build_transposed_tiles()
            q_packs = [read_q_packs(ng) for ng in range_constexpr(Q_STREAM_SUBTILES)]
            p_packs, s_meta = compute_s_tile(q_start, q_packs)
            dv_acc = [acc[i] for i in range(N_ACC_DV)]
            dk_acc = [acc[N_ACC_DV + i] for i in range(N_ACC_DK)]
            if const_expr(USE_T_TILES):
                gpu.barrier()  # dO^T / Q^T published
            dv_acc = accum_dv_tile(dv_acc, p_packs)
            dk_acc = accum_dk_tile(dk_acc, compute_ds_packs(s_meta))
            # Prevent the next DMA from overwriting Q LDS while another wave reads it.
            gpu.barrier()
            return dv_acc + dk_acc

        if active:
            acc_init = [c_zero_v4f32 for _ in range(N_ACC)]
            loop_results = acc_init
            for q_tile, it in range(
                fx.Int64(q_tile_start), fx.Int64(n_q_tiles), fx.Int64(1), init=acc_init
            ):  # ty: ignore
                it_list = list(it) if isinstance(it, (list, tuple)) else [it]
                acc = [it_list[i] for i in range(N_ACC)]
                q_start = fx.Int32(q_tile) * fx.Int32(BLOCK_N)
                acc = run_q_tile(acc, q_start)
                loop_results = yield acc

            results = (
                list(loop_results)
                if isinstance(loop_results, (list, tuple))
                else [loop_results]
            )
            with arith.fastmath(arith.FastMathFlags.fast):
                for og in range_constexpr(KV_OWNED_SUBTILES):
                    kv_row_base = (
                        kv_wave_base
                        + fx.Int32(og * MFMA_M)
                        + lane_div_16 * fx.Int32(MFMA_LANE_K)
                    )
                    for e in range_constexpr(MFMA_ELEMS_PER_LANE):
                        kv_row_e = kv_row_base + fx.Int32(e)
                        if kv_row_e < seq_len:
                            for c in range_constexpr(D_CHUNKS):
                                ov = results[c * KV_OWNED_SUBTILES + og]
                                col = fx.Int32(c * MFMA_M) + lane_mod_16
                                val = (Vec(ov)[e] * c_inv_n).to(elem_dtype)
                                out_dv[
                                    fx.Int64(seq_start + kv_row_e), head_idx, col
                                ] = val
                            for c in range_constexpr(HC_CHUNKS):
                                ov = results[N_ACC_DV + c * KV_OWNED_SUBTILES + og]
                                col = fx.Int32(c * MFMA_M) + lane_mod_16
                                val = (Vec(ov)[e] * c_alpha).to(elem_dtype)
                                out_dk[
                                    fx.Int64(seq_start + kv_row_e), head_idx, col
                                ] = val

    @flyc.jit
    def launch_hstu_attention_bwd_dvdk(
        batch: fx.Int32,
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        do: fx.Tensor,
        seq_offsets: fx.Tensor,
        num_targets: fx.Tensor,
        perm: fx.Tensor,
        out_dv: fx.Tensor,
        out_dk: fx.Tensor,
        stream: fx.Stream,
    ) -> None:
        c_ngg = fx.Int32(NUM_GRID_GROUPS)
        hz_total = batch * fx.Int32(num_heads)
        hz_per_group = (hz_total + fx.Int32(NUM_GRID_GROUPS - 1)) // c_ngg
        grid = fx.Int32(num_kv_tiles) * hz_per_group * c_ngg
        hstu_attention_bwd_dvdk(
            q,
            k,
            v,
            do,
            seq_offsets,
            num_targets,
            perm,
            out_dv,
            out_dk,
            hz_per_group,
            hz_total,
            value_attrs={
                "passthrough": [
                    ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
                    ["no-nans-fp-math", "true"],
                    ["unsafe-fp-math", "true"],
                ],
                "rocdl.waves_per_eu": WAVES_PER_EU,
                "rocdl.flat_work_group_size": f"{BLOCK_THREADS},{BLOCK_THREADS}",
            },
        ).launch(grid=grid, block=BLOCK_THREADS, smem=0, stream=stream)

    return launch_hstu_attention_bwd_dvdk
