# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Ping-pong-pipelined variant of the KV-owned HSTU backward dV/dK kernel.

Same mathematics as `hstu_attention_bwd.py` -- see that module's docstring --
but scheduled so the two waves resident on each SIMD are held in *opposite*
phases:

    phase M : the 48 MFMAs, their LDS operand feed, the Q/dO DMA, the
              transpose build, and the mask.
    phase V : the fused SiLU / SiLU' gate.  No memory operations at all.

One wave group is in M while the other is in V, so the matrix core and the VALU
pipe are both busy continuously.  Without this, the two resident waves sit in
*lockstep* on the shared `gpu.barrier()`: they idle together and they burst MFMA
together, contending for the one matrix core (measured: 27.9% matrix-core busy,
50.7% of all stall in `s_waitcnt`).

The schedule is ported from the gfx950 fp8 FMHA ping-pong kernel.  Only the
schedule -- no CDNA4 instruction is used here.

Three structural points, each load-bearing:

* **The chained GEMMs consume the previous tile's fragments.**  GEMM1/GEMM3 at
  trip `i` use `p_packs[i-1]`/`ds_packs[i-1]`, which is what merges GEMM0/2 of
  tile `i` and GEMM1/3 of tile `i-1` into one contiguous MFMA burst.  Seeding
  those packs to zero makes trip 0 self-starting and the extra trip the drain,
  so no iteration is peeled.
* **The transpose build lives in phase M, not phase V.**  With a half-iteration
  skew, a tile written in one group's V phase is published only at the next
  barrier, but the partner's consuming M phase overlaps that very V phase -- an
  unsynchronised cross-group RAW race.  Building in M puts one barrier between
  the write and the earliest partner read.
* **Row-major Q/dO are 3-deep and fired two tiles ahead.**  At depth 2 the load
  gets only the tail of one MFMA phase to return, which reinstates the very
  `vmcnt` stall this kernel exists to remove.

`validate_hstu_attention_bwd_dvdk_pp` hard-gates the supported shape: this
kernel is specialized for gfx942 / bf16 / head_dim == hidden_dim == 64 and the
`(384, 16, 8)` tile, and raises for anything else.
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
    make_lds_dma,
    pack_mfma_frag,
    phase_barrier,
    swz_col,
)

# Depth of the rolling LDS operand window, in operand groups.  The MFMA at group
# `g` consumes the pack read at `g - PREFETCH_W`, so each read has
# PREFETCH_W * KV_OWNED_SUBTILES MFMAs of cover.  The fp8 reference uses depth 2
# against a 32-cycle `32x32x64`; `16x16x16_bf16` is only ~16 cycles, so the
# depth has to be larger here to hide the same ~130-cycle ds_read.
PREFETCH_W = 3
# Operand groups per MFMA phase: 4 GEMMs x (head_dim / MFMA_M) chunks.
NUM_OPERAND_GROUPS = 16


def validate_hstu_attention_bwd_dvdk_pp(
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
    """Hard gate for the ping-pong kernel.

    Deliberately strict: this kernel hardcodes a single tuned shape, so anything
    outside it must fall back to `build_hstu_attention_bwd_dvdk` rather than
    silently compile a variant that was never measured.
    """
    if arch is None:
        arch = get_rocm_arch()
    if not (arch or "").startswith("gfx942"):
        raise ValueError(
            f"hstu bwd dvdk ping-pong requires gfx942, got {arch!r}. The schedule "
            "assumes 16x16x16 MFMA timing and a dword LDS DMA."
        )
    if dtype_str != "bf16":
        raise ValueError(f"hstu bwd dvdk ping-pong requires bf16, got {dtype_str!r}")
    if not causal:
        raise ValueError("hstu_attention_bwd only supports causal attention")
    if head_dim != 64 or hidden_dim != 64:
        raise ValueError(
            "hstu bwd dvdk ping-pong requires head_dim == hidden_dim == 64, got "
            f"{head_dim}/{hidden_dim}. Both groups share one duty code path, which "
            "relies on the Q and dO tiles being structurally identical."
        )
    if (block_m, block_n, num_waves) != (384, 16, 8):
        raise ValueError(
            "hstu bwd dvdk ping-pong requires (block_m, block_n, num_waves) = "
            f"(384, 16, 8), got ({block_m}, {block_n}, {num_waves}). 8 waves give "
            "2 per SIMD inside one barrier domain; 384 keeps 48 KV rows per wave, "
            "and smaller tiles lose more to reuse than they gain in residency."
        )
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
    if waves_per_eu < 0:
        raise ValueError(f"waves_per_eu must be non-negative, got {waves_per_eu}")
    if not host_math.isfinite(alpha):
        raise ValueError(f"alpha must be finite, got {alpha}")
    if contextual_seq_len < 0:
        raise ValueError(
            f"contextual_seq_len must be non-negative, got {contextual_seq_len}"
        )
    if max_attn_len < 0:
        raise ValueError(f"max_attn_len must be non-negative, got {max_attn_len}")


@functools.lru_cache(maxsize=16384)
def build_hstu_attention_bwd_dvdk_pp(
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
    block_m: int = 384,
    block_n: int = 16,
    num_waves: int = 8,
    waves_per_eu: int = 2,
    has_perm: bool = False,
):
    validate_hstu_attention_bwd_dvdk_pp(
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
    WAVES_PER_EU = waves_per_eu
    MFMA_QK_K, MFMA_QK_LANE_K, _ = _mfma_params_for_dim(head_dim)
    MFMA_DA_K, MFMA_DA_LANE_K, _ = _mfma_params_for_dim(hidden_dim)
    DMA_BYTES, DMA_ELEMS, K_SWZ_ROWS, K_SWZ_SHIFT = _arch_dma_params()

    elem_dtype = _dtype_to_elem_type(dtype_str)
    is_bf16 = True
    has_window = max_attn_len > 0
    has_contextual = contextual_seq_len > 0

    # head_dim == hidden_dim == 64 (gated above), so every per-dimension count
    # collapses to one value and the Q and dO tiles are structurally identical.
    K_STEPS = head_dim // MFMA_QK_K
    D_CHUNKS = hidden_dim // MFMA_M
    DK_STEPS = hidden_dim // MFMA_DA_K
    HC_CHUNKS = head_dim // MFMA_M
    STRIDE = head_dim
    assert K_STEPS == D_CHUNKS == DK_STEPS == HC_CHUNKS == 4
    assert MFMA_QK_LANE_K == MFMA_DA_LANE_K == MFMA_LANE_K
    assert BLOCK_N == MFMA_N, "Q_STREAM_SUBTILES must be 1"
    assert NUM_OPERAND_GROUPS == 4 * D_CHUNKS

    num_kv_tiles = (max_seq_len + BLOCK_M - 1) // BLOCK_M
    stride_qk_n = num_heads * head_dim
    stride_do_n = num_heads * hidden_dim

    # ---- ping-pong groups -------------------------------------------------
    # g0 = waves 0..3 stages Q and builds Q^T; g1 = waves 4..7 stages dO and
    # builds dO^T.  Wave i and wave i+4 share a SIMD, which is the pair the
    # phase alternation (and `s_setprio`) arbitrates.
    GRP_WAVES = NUM_WAVES // 2
    GRP_THREADS = GRP_WAVES * WARP_SIZE
    assert GRP_THREADS == 256

    # ---- LDS geometry -----------------------------------------------------
    # Both the row-major tiles and the transposes are 2-deep, and every slot is
    # its own `fx.Array` field -- see SharedStorage.
    RM_SLOTS = 2
    T_SLOTS = 2
    # The loop body is unrolled by the slot count so each sub-trip names a
    # *literal* slot. That is what keeps the slot index compile-time, which is
    # what keeps each access provably inside one LDS global.
    UNROLL = RM_SLOTS
    assert UNROLL % RM_SLOTS == 0 and UNROLL % T_SLOTS == 0
    T_PAD = MFMA_LANE_K
    T_STRIDE = BLOCK_N + T_PAD
    RM_TILE_ELEMS = BLOCK_N * STRIDE
    T_TILE_ELEMS = head_dim * T_STRIDE

    rm_tile_elems_per_pass = GRP_THREADS * DMA_ELEMS
    assert RM_TILE_ELEMS % rm_tile_elems_per_pass == 0
    NUM_DMA = RM_TILE_ELEMS // rm_tile_elems_per_pass
    PAIRS_PER_ROW = STRIDE // DMA_ELEMS

    # One transpose unit per (transposed row, 4-wide query group).
    assert BLOCK_N % MFMA_LANE_K == 0
    T_QGROUPS = BLOCK_N // MFMA_LANE_K
    T_UNITS = head_dim * T_QGROUPS
    assert T_UNITS == GRP_THREADS, "transpose must be exactly one pass per group"

    N_ACC_DV = D_CHUNKS * KV_OWNED_SUBTILES
    N_ACC_DK = HC_CHUNKS * KV_OWNED_SUBTILES
    N_ACC = N_ACC_DV + N_ACC_DK
    N_PACK = KV_OWNED_SUBTILES

    # One field per ROTATION SLOT, not one array per tensor with the slot as an
    # offset. Each `fx.Array` field becomes its own LDS global, and only then
    # can the backend prove that the `buffer_load_lds` filling slot 1 is
    # disjoint from the `ds_read`s feeding the MFMAs out of slot 0 -- which is
    # the entire point of double buffering: the write and the reads have to
    # proceed at the same time.
    #
    # With one fused array and a runtime slot offset the accesses are only
    # provably inside "the q array", so every read may-alias the in-flight DMA
    # and the backend serializes them. Measured: that cost ~8% here, on top of
    # the separate `inttoptr` provenance bug (see the LDS views below).
    @fx.struct
    class SharedStorage:
        q0: fx.Array[elem_dtype, RM_TILE_ELEMS, 16]
        q1: fx.Array[elem_dtype, RM_TILE_ELEMS, 16]
        do0: fx.Array[elem_dtype, RM_TILE_ELEMS, 16]
        do1: fx.Array[elem_dtype, RM_TILE_ELEMS, 16]
        q_t0: fx.Array[elem_dtype, T_TILE_ELEMS, 16]
        q_t1: fx.Array[elem_dtype, T_TILE_ELEMS, 16]
        do_t0: fx.Array[elem_dtype, T_TILE_ELEMS, 16]
        do_t1: fx.Array[elem_dtype, T_TILE_ELEMS, 16]

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def hstu_attention_bwd_dvdk_pp(
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

        # The group predicate must be provably uniform or the backend will not
        # select `buffer_load_lds` on a selected descriptor. readfirstlane makes
        # the divergence analysis treat it as scalar.
        tid_u = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, tid))
        is_g0 = tid_u < fx.Int32(GRP_THREADS)
        grp_tid = tid - is_g0.select(fx.Int32(0), fx.Int32(GRP_THREADS))
        grp_wave = grp_tid // fx.Int32(WARP_SIZE)

        block_id = fx.Int32(gpu.block_idx.x)
        grid_group = block_id % fx.Int32(NUM_GRID_GROUPS)
        pos_in_group = block_id // fx.Int32(NUM_GRID_GROUPS)
        local_hz_idx = pos_in_group // fx.Int32(num_kv_tiles)
        kv_tile_idx = pos_in_group % fx.Int32(num_kv_tiles)
        hz_idx = grid_group * hz_per_group + local_hz_idx
        block_valid = hz_idx < hz_total
        hz_idx = block_valid.select(hz_idx, fx.Int32(0))
        batch_idx = hz_idx // fx.Int32(num_heads)
        head_idx = hz_idx % fx.Int32(num_heads)

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
        do_base_byte_offset = (
            fx.Int64(seq_start) * fx.Int64(stride_do_n)
            + fx.Int64(head_idx) * fx.Int64(hidden_dim)
        ) * fx.Int64(2)

        # ---- LDS views ----------------------------------------------------
        # One view per (tensor, slot), each built straight off its own field's
        # pointer. The slot is a Python index resolved at trace time, so every
        # access is statically inside exactly one LDS global and the backend can
        # cross-noalias them -- that is what lets slot 1's DMA and slot 0's
        # ds_reads overlap instead of being serialized.
        #
        # Do NOT reintroduce a runtime slot offset here, and never form an LDS
        # address with `fx.inttoptr` or a `select` between two pointers: both
        # break the provenance walk, after which SIInsertWaitcnts cannot prove a
        # `ds_read` misses the in-flight `buffer_load_lds` and plants a
        # `vmcnt(0)` in front of every operand read (measured: 19.8 M stall
        # cycles, matrix-core duty down to 22%).
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        rm_layout = fx.make_layout(
            (BLOCK_N, STRIDE // MFMA_LANE_K, MFMA_LANE_K),
            (STRIDE, MFMA_LANE_K, 1),
        )
        t_layout = fx.make_layout(
            (head_dim, T_STRIDE // MFMA_LANE_K, MFMA_LANE_K),
            (T_STRIDE, MFMA_LANE_K, 1),
        )
        q_views = [lds.q0.view(rm_layout), lds.q1.view(rm_layout)]
        do_views = [lds.do0.view(rm_layout), lds.do1.view(rm_layout)]
        q_t_views = [lds.q_t0.view(t_layout), lds.q_t1.view(t_layout)]
        do_t_views = [lds.do_t0.view(t_layout), lds.do_t1.view(t_layout)]

        def q_swz_col(tile_row, col):
            return swz_col(tile_row, col, K_SWZ_ROWS, K_SWZ_SHIFT)

        # ---- K / V into registers, once ------------------------------------
        c_zero_qk_pack = Vec.filled(MFMA_QK_LANE_K, 0.0, elem_dtype).ir_value()
        c_zero_da_pack = Vec.filled(MFMA_DA_LANE_K, 0.0, elem_dtype).ir_value()

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
        c_zero_f = fx.Float32(0.0)
        c_zero_v4f32 = Vec.filled(MFMA_ELEMS_PER_LANE, 0.0, fx.Float32).ir_value()
        c_zero_pack = Vec.filled(MFMA_LANE_K, 0.0, elem_dtype).ir_value()

        kv_owned_ids = [to_id(kv_rows[og]) for og in range_constexpr(KV_OWNED_SUBTILES)]

        kv_start_row = kv_tile_idx * fx.Int32(BLOCK_M)
        kv_end_row = kv_start_row + fx.Int32(BLOCK_M)
        active = kv_start_row < seq_len

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

        # ---- staging duty: global -> LDS ----------------------------------
        # g0 stages Q, g1 stages dO. The two duties are built as separate
        # closures and dispatched on a wave-uniform `if`, rather than selecting
        # between two pointers -- see the provenance note above.
        _dma_atom, _dma_ptr_ty, _rebased_buffer_div = make_lds_dma(DMA_BYTES, elem_type)
        c_stride_n = fx.Int32(stride_qk_n)
        c_dma_elems = fx.Int32(DMA_ELEMS)
        c_pairs_per_row = fx.Int32(PAIRS_PER_ROW)

        def make_dma(tensor, base_byte_off, dst_views, swizzled):
            div = _rebased_buffer_div(
                fx.get_iter(tensor), base_byte_off, max_seq_len * stride_qk_n
            )
            rows, gcols = [], []
            for d in range_constexpr(NUM_DMA):
                pair = grp_tid + fx.Int32(d * GRP_THREADS)
                row = pair // c_pairs_per_row
                col = (pair % c_pairs_per_row) * c_dma_elems
                rows.append(row)
                gcols.append(q_swz_col(row, col) if const_expr(swizzled) else col)
            # The DMA destination must be wave-uniform in an SGPR, so it is the
            # one LDS address that still goes through `inttoptr` -- that is how
            # `m0` is written and it is what the shipping kernel does too. One
            # base per slot, so the slot never becomes a runtime offset.
            lane0 = []
            for v in dst_views:
                wave_byte = fx.Int32(fx.ptrtoint(fx.get_iter(v))) + grp_wave * fx.Int32(
                    WARP_SIZE * DMA_BYTES
                )
                lane0.append(rocdl.readfirstlane(fx.Int32.ir_type, wave_byte))

            def fire(slot, q_start, d):
                row = rows[d]
                in_bounds = (q_start + row) < seq_len
                local_tok = in_bounds.select(q_start + row, fx.Int32(0))
                src_elem = local_tok * c_stride_n + gcols[d]
                lds_byte = fx.Int32(lane0[slot]) + fx.Int32(
                    d * GRP_THREADS * DMA_BYTES
                )
                dst = fx.make_view(
                    fx.inttoptr(_dma_ptr_ty, lds_byte), fx.make_layout(1, 1)
                )
                fx.copy(_dma_atom, fx.slice(div, (None, fx.Int32(src_elem))), dst)

            return fire

        dma_q = make_dma(q, q_base_byte_offset, q_views, True)
        dma_do = make_dma(do, do_base_byte_offset, do_views, False)

        # ---- staging duty: LDS -> LDS transpose ----------------------------
        t_unit_row = grp_tid // fx.Int32(T_QGROUPS)
        t_unit_qg = grp_tid % fx.Int32(T_QGROUPS)
        t_unit_qbase = t_unit_qg * fx.Int32(MFMA_LANE_K)

        def make_transpose(rm_views_, t_views_, swizzled):
            def run(rm_slot, t_slot):
                view = rm_views_[rm_slot]
                elems = []
                for i in range_constexpr(MFMA_LANE_K):
                    row = t_unit_qbase + fx.Int32(i)
                    col = (
                        q_swz_col(row, t_unit_row)
                        if const_expr(swizzled)
                        else t_unit_row
                    )
                    elems.append(
                        view[
                            row,
                            col // fx.Int32(MFMA_LANE_K),
                            col % fx.Int32(MFMA_LANE_K),
                        ]
                    )
                t_views_[t_slot][t_unit_row, t_unit_qg, None].store(
                    Vec.from_elements(elems, elem_dtype)
                )

            return run

        transpose_q = make_transpose(q_views, q_t_views, True)
        transpose_do = make_transpose(do_views, do_t_views, False)

        def transpose_zero(t_slot):
            z = Vec(c_zero_pack)
            q_t_views[t_slot][t_unit_row, t_unit_qg, None].store(z)
            do_t_views[t_slot][t_unit_row, t_unit_qg, None].store(z)

        # ---- the 16 MFMA operand groups ------------------------------------
        # g 0..3   GEMM1 dV: B = dO^T[tile i-1]        A = p_prev   (register)
        # g 4..7   GEMM3 dK: B = Q^T [tile i-1]        A = ds_prev  (register)
        # g 8..11  GEMM0 S : A = Q    [tile i]         B = k_packs  (register)
        # g 12..15 GEMM2 dA: A = dO   [tile i]         B = v_packs  (register)
        def read_operand(g, rm_slot, t_slot):
            if const_expr(g < 2 * D_CHUNKS):
                views = do_t_views if const_expr(g < D_CHUNKS) else q_t_views
                c = g if const_expr(g < D_CHUNKS) else g - D_CHUNKS
                row = fx.Int32(c * MFMA_M) + lane_mod_16
                return views[t_slot][row, lane_div_16, None].load().ir_value()
            q_row = lane_mod_16
            if const_expr(g < 3 * D_CHUNKS):
                col = fx.Int32((g - 2 * D_CHUNKS) * MFMA_QK_K) + lane_div_16 * fx.Int32(
                    MFMA_QK_LANE_K
                )
                view = q_views[rm_slot]
                return (
                    view[q_row, q_swz_col(q_row, col) // fx.Int32(MFMA_QK_LANE_K), None]
                    .load()
                    .ir_value()
                )
            col = fx.Int32((g - 3 * D_CHUNKS) * MFMA_DA_K) + lane_div_16 * fx.Int32(
                MFMA_DA_LANE_K
            )
            view = do_views[rm_slot]
            return view[q_row, col // fx.Int32(MFMA_DA_LANE_K), None].load().ir_value()

        def keep_masks(q_start, og):
            q_base = q_start + lane_div_16 * fx.Int32(MFMA_LANE_K)
            out = []
            for i in range_constexpr(MFMA_ELEMS_PER_LANE):
                q_raw = q_base + fx.Int32(i)
                q_id = to_id(q_raw)
                dist = q_id - kv_owned_ids[og]
                keep = (q_raw == kv_rows[og]) | (dist > fx.Int32(0))
                if has_window:
                    keep = keep & (dist <= fx.Int32(max_attn_len))
                if has_contextual:
                    ctx = (q_id == fx.Int32(0)) & (kv_owned_ids[og] < max_id)
                    keep = keep | ctx
                out.append(keep & (q_raw < seq_len) & kv_in_bounds[og])
            return out

        # ---- phase M -------------------------------------------------------
        def mfma_phase(dv, dk, p_prev, ds_prev, q_tile, rm_cur, rm_fill, t_cur, t_prev):
            """Four GEMMs, the operand feed, this group's DMA and transpose, the mask."""
            rocdl.s_setprio(1)
            q_start = fx.Int32(q_tile) * fx.Int32(BLOCK_N)
            s_accs = [c_zero_v4f32 for _ in range_constexpr(KV_OWNED_SUBTILES)]
            da_accs = [c_zero_v4f32 for _ in range_constexpr(KV_OWNED_SUBTILES)]

            # Prime the rolling window before the first MFMA.  Group 0 is GEMM1,
            # whose dO^T operand was published at the earliest barrier.
            # Ring of PREFETCH_W + 1 slots: group g consumes win[g % N] and then
            # issues the read for g + PREFETCH_W into the slot it just freed.
            # The last slot is filled at g == 0, before its first use at
            # g == PREFETCH_W.
            win = [read_operand(g, rm_cur, t_prev) for g in range_constexpr(PREFETCH_W)]
            win.append(None)

            for g in range_constexpr(NUM_OPERAND_GROUPS):
                rocdl.sched_barrier(0)
                op = win[g % (PREFETCH_W + 1)]
                for og in range_constexpr(KV_OWNED_SUBTILES):
                    if const_expr(g < D_CHUNKS):
                        off = g * KV_OWNED_SUBTILES + og
                        dv[off] = mfma_acc(p_prev[og], op, dv[off])
                    elif const_expr(g < 2 * D_CHUNKS):
                        off = (g - D_CHUNKS) * KV_OWNED_SUBTILES + og
                        dk[off] = mfma_acc(ds_prev[og], op, dk[off])
                    elif const_expr(g < 3 * D_CHUNKS):
                        s_accs[og] = qk_mfma_acc(
                            op, k_packs[g - 2 * D_CHUNKS][og], s_accs[og]
                        )
                    else:
                        da_accs[og] = da_mfma_acc(
                            op, v_packs[g - 3 * D_CHUNKS][og], da_accs[og]
                        )

                gn = g + PREFETCH_W
                if const_expr(gn < NUM_OPERAND_GROUPS):
                    win[gn % (PREFETCH_W + 1)] = read_operand(gn, rm_cur, t_prev)

                # The DMA fire is spread through the burst's slack; it is VMEM,
                # so it does not interact with the LDS retirement order below.
                # The duty branch is wave-uniform (`is_g0` comes from
                # readfirstlane) and side-effect-only, so it lowers to a scalar
                # branch.
                if const_expr(g == 2):
                    if is_g0:
                        dma_q(rm_fill, q_start + fx.Int32(BLOCK_N), 0)
                    else:
                        dma_do(rm_fill, q_start + fx.Int32(BLOCK_N), 0)
                if const_expr(g == 6):
                    if is_g0:
                        dma_q(rm_fill, q_start + fx.Int32(BLOCK_N), 1)
                    else:
                        dma_do(rm_fill, q_start + fx.Int32(BLOCK_N), 1)
                rocdl.sched_barrier(0)

            # Build this tile's transpose AFTER every operand read has been
            # issued, never in the middle of the burst.
            #
            # `lgkmcnt` retires IN ORDER, so these 5 ops (4 scalar ds_read_u16 +
            # 1 ds_write_b64) sitting at group 3 put themselves on the critical
            # path of all ten operand reads issued behind them: each of those
            # could only retire once a 2-byte strided gather had. That is what
            # produced lgkmcnt(0)/(1) in front of MFMAs that needed only
            # lgkmcnt(3), and it is why deepening PREFETCH_W did nothing -- the
            # window was never the constraint, queue position was.
            #
            # Placing it here exposes its own latency instead of everyone
            # else's, and leaves slot lifetimes untouched (still: write t_cur,
            # read t_prev, both within phase M), so it adds no cross-group
            # hazard.
            rocdl.sched_barrier(0)
            if is_g0:
                transpose_q(rm_cur, t_cur)
            else:
                transpose_do(rm_cur, t_cur)
            rocdl.sched_barrier(0)

            rocdl.s_setprio(0)
            # The tile i+1 DMA fired above must land before the mid barrier, not
            # merely before this group's own next read: with the half-iteration
            # skew the PARTNER's M(i+1) runs in the interval this barrier opens,
            # and `vmcnt` is per-wave. Firing at groups 2 and 6 of 16 leaves it
            # most of the burst to return. `lgkmcnt(0)` publishes the transpose
            # write and retires this phase's reads before the next trip's DMA.
            rocdl.s_waitcnt(vmcnt=0)
            rocdl.s_waitcnt(lgkmcnt=0)
            return dv, dk, s_accs, da_accs, q_start

        # ---- phase V -------------------------------------------------------
        def gate_phase(s_accs, da_accs, q_start):
            """Fused SiLU / SiLU' gate: produces both P and dS from one sigmoid.

            GEMM0 and GEMM2 both finished in phase M, so `grad` never has to
            outlive the element it belongs to -- unlike the non-ping-pong kernel,
            which carries `s_meta` across a barrier and 24 MFMAs.
            """
            # Batched a stage at a time across ALL KV_OWNED_SUBTILES *
            # MFMA_ELEMS_PER_LANE elements, not one element at a time. The gate
            # is a ~10-deep serial dependency chain per element with two
            # quarter-rate transcendentals in it, so the only way to fill the
            # VALU pipe is to give the scheduler every independent chain at
            # once. Measured 10 cy per VALU op when this was written per
            # element -- latency-bound, not throughput-bound.
            n = KV_OWNED_SUBTILES * MFMA_ELEMS_PER_LANE
            rng = range_constexpr(n)
            # The mask needs no GEMM result, so it can sit in either phase --
            # but it must NOT go inside the fenced MFMA burst. Interleaved
            # between the groups, these ~33 VALU ops are pinned by the
            # sched_barriers and stretch the MFMA stream: measured +9%.
            keep = [keep_masks(q_start, og) for og in range_constexpr(KV_OWNED_SUBTILES)]
            flat_keep = [
                keep[j // MFMA_ELEMS_PER_LANE][j % MFMA_ELEMS_PER_LANE] for j in rng
            ]
            with arith.fastmath(arith.FastMathFlags.fast):
                s = [Vec(s_accs[j // MFMA_ELEMS_PER_LANE])[j % MFMA_ELEMS_PER_LANE]
                     for j in rng]
                da = [Vec(da_accs[j // MFMA_ELEMS_PER_LANE])[j % MFMA_ELEMS_PER_LANE]
                      for j in rng]
                sc = [s[j] * c_alpha for j in rng]
                emu = [exp2_f32(sc[j] * c_neg_log2e) for j in rng]
                sig = [c_one_f / (c_one_f + emu[j]) for j in rng]
                silu = [sc[j] * sig[j] for j in rng]
                grad = [sig[j] * (c_one_f + sc[j] * (c_one_f - sig[j])) for j in rng]
                gated = [c_inv_n * grad[j] * da[j] for j in rng]
                p_vals = [flat_keep[j].select(silu[j], c_zero_f) for j in rng]
                ds_vals = [flat_keep[j].select(gated[j], c_zero_f) for j in rng]
            e = MFMA_ELEMS_PER_LANE
            p_packs = [
                pack_mfma_frag(p_vals[og * e : og * e + e], is_bf16, elem_dtype)
                for og in range_constexpr(KV_OWNED_SUBTILES)
            ]
            ds_packs = [
                pack_mfma_frag(ds_vals[og * e : og * e + e], is_bf16, elem_dtype)
                for og in range_constexpr(KV_OWNED_SUBTILES)
            ]
            return p_packs, ds_packs

        # ---- the pipeline ---------------------------------------------------
        if active:
            # Trip 0's GEMM1/GEMM3 read the transpose of a tile that does not
            # exist. Their A operands are zero, but `0 * NaN` is NaN and would
            # poison the accumulators for the whole sweep, so zero that slot.
            transpose_zero(1)
            # Prime the pipeline with tile T0; the loop fires one tile ahead.
            _t0 = q_tile_start * fx.Int32(BLOCK_N)
            if is_g0:
                dma_q(0, _t0, 0)
                dma_q(0, _t0, 1)
            else:
                dma_do(0, _t0, 0)
                dma_do(0, _t0, 1)
            rocdl.s_waitcnt(vmcnt=0)
            rocdl.s_waitcnt(lgkmcnt=0)
            phase_barrier()

            # The skew: g0 burns one extra barrier so that from here on g0 is in
            # phase M exactly when g1 is in phase V, and vice versa. Barrier
            # counts stay equal overall -- g1 repays it after the loop.
            if is_g0:
                phase_barrier()

            acc_init = [c_zero_v4f32 for _ in range_constexpr(N_ACC)] + [
                c_zero_pack for _ in range_constexpr(2 * N_PACK)
            ]
            loop_results = acc_init
            # The body is unrolled UNROLL times so every slot index below is a
            # literal, which is what keeps each LDS access inside one global.
            # The trip count is rounded UP to a multiple of UNROLL rather than
            # peeled: a tile past `q_upper` has every `keep` false, so its gate
            # emits zero packs and the next trip's GEMM1/GEMM3 add exactly
            # nothing. That is the same property the drain trip already relies
            # on, so the padding is free of correctness risk and costs <1 trip.
            n_trips = n_q_tiles + fx.Int32(1) - q_tile_start
            n_outer = (n_trips + fx.Int32(UNROLL - 1)) // fx.Int32(UNROLL)
            for _outer, it in range(
                fx.Int64(0), fx.Int64(n_outer), fx.Int64(1), init=acc_init
            ):  # ty: ignore
                it_list = list(it) if isinstance(it, (list, tuple)) else [it]
                dv = [it_list[i] for i in range_constexpr(N_ACC_DV)]
                dk = [it_list[N_ACC_DV + i] for i in range_constexpr(N_ACC_DK)]
                p_prev = [it_list[N_ACC + i] for i in range_constexpr(N_PACK)]
                ds_prev = [
                    it_list[N_ACC + N_PACK + i] for i in range_constexpr(N_PACK)
                ]
                base_tile = q_tile_start + fx.Int32(_outer) * fx.Int32(UNROLL)

                for u in range_constexpr(UNROLL):
                    q_tile = base_tile + fx.Int32(u)
                    # Literal slots: tile i lives in i % RM_SLOTS, its transpose
                    # in i % T_SLOTS, and GEMM1/GEMM3 read tile i-1's.
                    rm_cur = u % RM_SLOTS
                    rm_fill = (u + 1) % RM_SLOTS
                    t_cur = u % T_SLOTS
                    t_prev = (u + 1) % T_SLOTS
                    dv, dk, s_accs, da_accs, q_start = mfma_phase(
                        dv, dk, p_prev, ds_prev, q_tile, rm_cur, rm_fill, t_cur, t_prev
                    )
                    phase_barrier()
                    p_prev, ds_prev = gate_phase(s_accs, da_accs, q_start)
                    phase_barrier()

                loop_results = yield (dv + dk + p_prev + ds_prev)

            if not is_g0:
                phase_barrier()

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
    def launch_hstu_attention_bwd_dvdk_pp(
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
        hstu_attention_bwd_dvdk_pp(
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

    return launch_hstu_attention_bwd_dvdk_pp
