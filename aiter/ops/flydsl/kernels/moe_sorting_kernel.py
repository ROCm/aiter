# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MoE token sorting kernel (FlyDSL).

Implements the MoE sorting operation used in DeepSeek R1 and similar MoE models.
Given router top-k selections (topk_ids, topk_weights), reorganizes tokens by expert
for efficient batched expert GEMM execution.

Algorithm: counting sort in LDS (histogram → prefix-sum → scatter).

Three paths (selected by T vs ONESHOT_MAX_T = min(sub_tokens, max(16, BLOCK_SIZE // max(topk, E//8)))):
  - Oneshot (T <= ONESHOT_MAX_T): single kernel, all phases in LDS.
  - Multiphase/2k (ONESHOT_MAX_T < T <= 2048): 2 kernels (fused P0v2 + P23) via HBM workspace.
  - Multiphase/4k (T > 2048): 4 kernels (ClearWS → P0 scatter → P1 count → P23) via HBM workspace.

Packed token ID format: (topk_position << 24) | token_id
  - Upper 8 bits: topk slot (0..topk-1)
  - Lower 24 bits: token index (0..M-1)
  - Padding sentinel: (topk << 24) | M
"""

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops, gpu, range_constexpr
from flydsl.expr import rocdl as fly_rocdl
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from .kernels_common import get_warp_size
from .tensor_shim import _run_compiled

BLOCK_SIZE = 256
UNIT_SIZE = 32  # GEMM tile-M, aka block_size in CK
WARP_SIZE = get_warp_size()

# P23 block-size policy: E in (256, 512] (e.g. DSV4 E=385) uses 512-thread blocks
# when T <= threshold to avoid per-block serial prefix extension (E > K4_BLOCK).
# Above threshold, 256-thread blocks match OPUS mesh-scan geometry at large T.
P23_LARGE_T_THRESHOLD = 8192

# DPP constants for prefix sum (used by oneshot and multiphase)
DPP_ROW_SHR_1 = 0x111
DPP_ROW_SHR_2 = 0x112
DPP_ROW_SHR_4 = 0x114
DPP_ROW_SHR_8 = 0x118
DPP_ROW_MASK = 0xF
DPP_BANK_MASK = 0xF


def _unwrap_val(v):
    """Unwrap DSL value to raw MLIR ir.Value."""
    return v.ir_value() if hasattr(v, "ir_value") else v


def _dpp_intra_wave_prefix_sum(val, lane, WARP_SIZE):
    """inclusive prefix sum within a single wave using DPP.

    Performs 4 DPP row_shr steps (1, 2, 4, 8) for intra-row scan, then
    2 ds_bpermute steps (16, 32) for cross-row accumulation within the wave.
    Returns the inclusive prefix sum value for each lane.

    Call inside @flyc.kernel only — emits MLIR ops during tracing.
    """
    val_raw = _unwrap_val(val)
    zero_raw = _unwrap_val(fx.Int32(0))

    for shift, dpp_op, threshold in [
        (1, DPP_ROW_SHR_1, 1),
        (2, DPP_ROW_SHR_2, 2),
        (4, DPP_ROW_SHR_4, 4),
        (8, DPP_ROW_SHR_8, 8),
    ]:
        remote = fly_rocdl.update_dpp(
            T.i32, zero_raw, val_raw, dpp_op, DPP_ROW_MASK, DPP_BANK_MASK, True
        )
        val = (lane >= fx.Int32(threshold)).select(val + fx.Int32(remote), val)
        val_raw = _unwrap_val(val)

    src_lane_16 = (lane & fx.Int32(0x30)) - fx.Int32(1)
    remote16 = fly_rocdl.ds_bpermute(T.i32, src_lane_16 * fx.Int32(4), val)
    val = (lane >= fx.Int32(16)).select(val + fx.Int32(remote16), val)

    if WARP_SIZE > 32:
        src_lane_32 = (lane & fx.Int32(0x30)) - fx.Int32(17)
        remote32 = fly_rocdl.ds_bpermute(T.i32, src_lane_32 * fx.Int32(4), val)
        val = (lane >= fx.Int32(32)).select(val + fx.Int32(remote32), val)

    return val


@flyc.jit
def _allwave_inclusive_prefix_sum(val, lane, wave, scratch_mr, NUM_WAVES, WARP_SIZE):
    """DPP intra-wave prefix sum + cross-wave LDS accumulation.

    Returns (intra_wave_val, inclusive) where intra_wave_val is the per-wave
    result (needed for total_padded computation) and inclusive is the full
    cross-wave inclusive prefix sum.
    """
    val = _dpp_intra_wave_prefix_sum(val, lane, WARP_SIZE)
    if lane == fx.Int32(WARP_SIZE - 1):
        _lds_store_raw(scratch_mr, val, wave)
    gpu.barrier()
    cross = fx.Int32(0)
    for _w in range_constexpr(NUM_WAVES - 1):
        wt = _lds_load_raw(scratch_mr, fx.Int32(_w))
        cross = (wave > fx.Int32(_w)).select(cross + wt, cross)
    return val, val + cross


@flyc.jit
def _zero_moe_buf_grid_stride(moe_buf_rsrc, gid_v4, stride_v4, total_v4, oob_idx):
    """Grid-stride loop zeroing moe_buf via vectorized buffer_store."""
    c_one = fx.Int32(1)
    niters = (total_v4 + stride_v4 - c_one) // stride_v4
    c_zero_v4 = fx.Vector.filled(4, 0, fx.Int32)
    c4 = fx.Int32(4)
    for _z in range(fx.Index(0), ArithValue(niters).index_cast(T.index), fx.Index(1)):
        idx = gid_v4 + fx.Int32(_z) * stride_v4
        valid = idx < total_v4
        buffer_ops.buffer_store(
            c_zero_v4, moe_buf_rsrc, valid.select(idx * c4, oob_idx)
        )


def _extend_prefix_sum_serial(mr, start_block, E, load_fn, store_fn):
    """Thread-0 serial extension of prefix sum for experts >= start_block.

    Reads mr[start_block], then accumulates mr[start_block+1..E] in place.
    Returns the final accumulated value (mr[E]).
    """
    prev = load_fn(mr, fx.Int32(start_block))
    for _ext in range_constexpr(start_block, E):
        cur = load_fn(mr, fx.Int32(_ext + 1))
        new_val = prev + cur
        store_fn(mr, new_val, fx.Int32(_ext + 1))
        prev = new_val
    return prev


@flyc.jit
def _write_expert_id_blocks(sorted_e_rsrc, local_eid, blk_start, n_blks):
    """Write local_eid to sorted_expert_ids[blk_start .. blk_start+n_blks)."""
    for _jb in range(fx.Index(0), ArithValue(n_blks).index_cast(T.index), fx.Index(1)):
        blk_idx = blk_start + fx.Int32(_jb)
        buffer_ops.buffer_store(local_eid, sorted_e_rsrc, blk_idx)


@flyc.jit
def _fill_sentinel_slots(
    sorted_ids_rsrc, sorted_w_rsrc, start, count, sentinel, block_size, tid, oob_idx
):
    """Cooperative sentinel fill: threads fill [start, start+count) with sentinels."""
    c_zero = fx.Int32(0)
    end = start + count
    niters = (count + fx.Int32(block_size) - fx.Int32(1)) // fx.Int32(block_size)
    for _p in range(fx.Index(0), ArithValue(niters).index_cast(T.index), fx.Index(1)):
        slot = start + fx.Int32(_p) * fx.Int32(block_size) + tid
        safe = (slot < end).select(slot, oob_idx)
        buffer_ops.buffer_store(sentinel, sorted_ids_rsrc, safe)
        buffer_ops.buffer_store(c_zero, sorted_w_rsrc, safe)


# ---------------------------------------------------------------------------
# LDS helpers for multiphase kernels (module-level, used inside @flyc.kernel)
# ---------------------------------------------------------------------------
def _lds_load_raw(raw_ptr, idx):
    """Load i32 from an LDS pointer at element offset `idx` (i32 or index)."""
    return fx.ptr_load(raw_ptr + fx.Int64(idx))


def _lds_store_raw(raw_ptr, val, idx):
    """Store i32 to an LDS pointer at element offset `idx` (i32 or index)."""
    fx.ptr_store(val, raw_ptr + fx.Int64(idx))


_dummy_mask_cache = {}  # device -> torch.Tensor(1, dtype=i32, value=1)


# ---------------------------------------------------------------------------
# FlyDSL GPU kernel — oneshot path (single kernel, all phases in LDS)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=256)
def _compile_moe_sorting_oneshot(
    *,
    num_experts: int,
    topk: int,
    max_tokens: int = 128,
    unit_size: int = UNIT_SIZE,
    has_mask: bool = False,
):
    """Compile the oneshot MoE sorting kernel (single kernel, all phases in LDS).

    Parameters
    ----------
    num_experts : int
        Number of routed experts (e.g. 256 for DeepSeek R1).
    topk : int
        Experts per token (e.g. 8 for DeepSeek R1).
    max_tokens : int
        Upper bound on T for LDS sizing. Actual T is passed at runtime.
    unit_size : int
        GEMM tile-M for padding alignment (default 32).
    """
    arch = get_hip_arch()
    E = num_experts
    # CDNA (warp64): 512 threads = 8 waves, affordable cross-wave reduction.
    max_oneshot_block = 512 if WARP_SIZE == 64 else 256
    ONESHOT_BLOCK = 256 if E <= 256 else min(512, max_oneshot_block)
    NUM_WAVES = ONESHOT_BLOCK // WARP_SIZE
    smem_cols = E + 1

    # LDS sizing: sub_tokens rows for the token×expert histogram
    # Match CK's sizing: total LDS / occupancy / smem_cols, rounded to 8
    if arch in ("gfx942",) or str(arch).startswith("gfx94"):
        lds_capacity_bytes = 65536
    elif str(arch).startswith("gfx95"):
        lds_capacity_bytes = 163840
    else:
        lds_capacity_bytes = 65536  # conservative default

    lds_capacity_ints = lds_capacity_bytes // 4
    target_occupancy = 2
    r = lds_capacity_ints // target_occupancy // smem_cols
    sub_unroll = 8
    cumsum_bufs = 2
    if r < (cumsum_bufs + sub_unroll):
        raise ValueError(
            f"LDS too small for E={E}: need at least {(cumsum_bufs + sub_unroll) * smem_cols * 4} bytes"
        )
    r_for_sub = ((r - cumsum_bufs) // sub_unroll) * sub_unroll
    r_token_min = ((max_tokens + sub_unroll - 1) // sub_unroll) * sub_unroll
    r_for_sub = min(r_for_sub, r_token_min)
    sub_tokens = r_for_sub

    # LDS regions for the oneshot kernel:
    #   cumsum[E+1]  exclusive prefix sums per expert
    #   cumdup[E+1]  duplicate of cumsum for scatter phase
    #   mesh[sub_tokens, smem_cols]  token mesh (row-major linear elements)
    #   scratch[NUM_WAVES]  cross-wave scratch for all-wave prefix sum
    @fx.struct
    class SharedStorage:
        cumsum: fx.Array[fx.Int32, smem_cols, 16]
        cumdup: fx.Array[fx.Int32, smem_cols, 16]
        mesh: fx.Array[fx.Int32, sub_tokens * smem_cols, 16]
        scratch: fx.Array[fx.Int32, NUM_WAVES, 16]

    @flyc.kernel(known_block_size=[ONESHOT_BLOCK, 1, 1])
    def moe_sorting_oneshot_kernel(
        topk_ids_tensor: fx.Tensor,
        topk_weights_tensor: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_weights_out: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        moe_buf: fx.Tensor,
        expert_mask_tensor: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_moe_buf_elems: fx.Int32,
    ):
        bid = gpu.block_idx.x
        tid = gpu.thread_idx.x
        lane = tid % WARP_SIZE
        wave = tid // WARP_SIZE
        tokens = i32_tokens
        c_zero_i32 = fx.Int32(0)
        c_one_i32 = fx.Int32(1)
        c_oob_idx = fx.Int32(0x7FFFFFFF)
        c4_i32 = fx.Int32(4)

        # Buffer resources (needed by both paths, defined at top level)
        moe_buf_rsrc = buffer_ops.create_buffer_resource(moe_buf, max_size=True)
        topk_ids_rsrc = buffer_ops.create_buffer_resource(
            topk_ids_tensor, max_size=True
        )
        weights_rsrc = buffer_ops.create_buffer_resource(
            topk_weights_tensor, max_size=True
        )
        sorted_ids_rsrc = buffer_ops.create_buffer_resource(
            sorted_token_ids, max_size=True
        )
        sorted_w_rsrc = buffer_ops.create_buffer_resource(
            sorted_weights_out, max_size=True
        )
        sorted_e_rsrc = buffer_ops.create_buffer_resource(
            sorted_expert_ids, max_size=True
        )
        nvalid_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)
        mask_rsrc = buffer_ops.create_buffer_resource(expert_mask_tensor, max_size=True)

        # LDS: capture field pointers ONCE — dominates all child scf.for/scf.if.
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        cumsum_mr = lds.cumsum.ptr
        cumdup_mr = lds.cumdup.ptr
        mesh_mr = lds.mesh.ptr
        scratch_mr = lds.scratch.ptr

        c_topk = fx.Int32(topk)
        c_E = fx.Int32(E)
        c_unit = fx.Int32(unit_size)
        c_sub_tokens = fx.Int32(sub_tokens)
        c_smem_cols = fx.Int32(smem_cols)
        c_sentinel = fx.Int32((topk << 24))

        # =================== MOE_BUF ZEROING (blocks > 0 only) ===============
        if bid != c_zero_i32:
            zero_gid_v4 = (bid - c_one_i32) * fx.Int32(ONESHOT_BLOCK) + tid
            num_zero_blocks = gpu.grid_dim.x - c_one_i32
            zero_stride_v4 = num_zero_blocks * fx.Int32(ONESHOT_BLOCK)
            _zero_moe_buf_grid_stride(
                moe_buf_rsrc,
                zero_gid_v4,
                zero_stride_v4,
                i32_moe_buf_elems >> fx.Int32(2),
                c_oob_idx,
            )

        # =================== SORTING (block 0 only) ==========================
        if bid == c_zero_i32:
            # ========================= PHASE 1: Histogram =========================
            # Clear mesh region — unconditional store to safe index when out of bounds
            for i_clear in range_constexpr(0, sub_tokens * smem_cols, ONESHOT_BLOCK):
                idx = fx.Int32(i_clear) + tid
                is_valid = idx < fx.Int32(sub_tokens * smem_cols)
                safe_idx = is_valid.select(idx, c_zero_i32)
                safe_idx_ix = ArithValue(safe_idx).index_cast(T.index)
                # Always store; out-of-bounds threads harmlessly write to index 0
                _lds_store_raw(mesh_mr, c_zero_i32, safe_idx_ix)
            gpu.barrier()

            # Fill mesh: for each (token, topk_slot), write topk_slot+1 to mesh[token, expert_id]
            total_assignments = tokens * c_topk
            for i_assign in range_constexpr(0, max_tokens * topk, ONESHOT_BLOCK):
                flat_idx = fx.Int32(i_assign) + tid
                is_valid = flat_idx < total_assignments
                safe_flat = is_valid.select(flat_idx, c_zero_i32)

                token_id = safe_flat // c_topk
                topk_slot = safe_flat % c_topk

                global_idx = token_id * c_topk + topk_slot
                eid = buffer_ops.buffer_load(
                    topk_ids_rsrc, global_idx, vec_width=1, dtype=T.i32
                )

                # mesh[token_id, eid] = topk_slot + 1 (valid threads only).
                # Invalid threads must NOT write to mesh[0] — that would race
                # with a valid write to (token=0, expert=0).
                mesh_addr = token_id * c_smem_cols + eid
                last_mesh_idx = fx.Int32(sub_tokens * smem_cols - 1)
                safe_mesh_addr = is_valid.select(mesh_addr, last_mesh_idx)
                safe_mesh_ix = ArithValue(safe_mesh_addr).index_cast(T.index)
                val = is_valid.select(topk_slot + c_one_i32, c_zero_i32)
                _lds_store_raw(mesh_mr, val, safe_mesh_ix)
            gpu.barrier()

            # ===================== PHASE 2: Count + Prefix Sum =====================
            c_lane_group_sz = fx.Int32(8)
            lane_group_id = tid // c_lane_group_sz
            lane_group_os = tid % c_lane_group_sz
            width8_i32 = fx.Int32(8)

            is_t0 = tid == c_zero_i32

            # Initialize cumsum[0] = 0.  All threads write 0 so there's no
            # read-modify-write race across waves.
            _lds_store_raw(cumsum_mr, c_zero_i32, c_zero_i32)
            gpu.barrier()

            for i_e in range_constexpr(0, E, ONESHOT_BLOCK // 8):
                eid_local = fx.Int32(i_e) + lane_group_id
                eid_valid = eid_local < c_E

                cnt = c_zero_i32
                for i_sub in range_constexpr(0, sub_tokens, 8):
                    sub_idx = fx.Int32(i_sub) + lane_group_os
                    sub_valid = sub_idx < c_sub_tokens
                    combined_valid = eid_valid & sub_valid

                    safe_sub = combined_valid.select(sub_idx, c_zero_i32)
                    safe_eid = combined_valid.select(eid_local, c_zero_i32)
                    mesh_rd_addr = safe_sub * c_smem_cols + safe_eid
                    mesh_rd_ix = ArithValue(mesh_rd_addr).index_cast(T.index)
                    mesh_val = _lds_load_raw(mesh_mr, mesh_rd_ix)

                    has_token = combined_valid.select(
                        (mesh_val != c_zero_i32).select(c_one_i32, c_zero_i32),
                        c_zero_i32,
                    )

                    # Reduce within lane-group of 8
                    reduced = has_token
                    for sh in range_constexpr(3):
                        off = fx.Int32(1 << sh)
                        peer = reduced.shuffle_xor(off, width8_i32)
                        reduced = reduced + peer
                    cnt = cnt + reduced

                # Only lane 0 of each valid lane-group writes the count to cumsum[eid+1].
                # Invalid threads: write_valid is false, cs_idx = 0, and we write 0 to
                # cumsum[0] which is harmless (cumsum[0] is always 0).
                write_valid = eid_valid & (lane_group_os == c_zero_i32)
                cs_idx = write_valid.select(eid_local + c_one_i32, c_zero_i32)
                cs_ix = ArithValue(cs_idx).index_cast(T.index)
                cs_val = write_valid.select(cnt, c_zero_i32)
                _lds_store_raw(cumsum_mr, cs_val, cs_ix)
            gpu.barrier()

            # Phase 2b: Prefix sum over expert counts.
            # Step 1: Each thread converts its expert's raw count → padded block size.
            for i_cvt in range_constexpr(0, E, ONESHOT_BLOCK):
                cvt_eid = fx.Int32(i_cvt) + tid
                cvt_valid = cvt_eid < c_E
                # Safe index: valid → cumsum[eid+1], invalid → cumsum[0] (write 0, harmless)
                safe_cvt_idx = cvt_valid.select(cvt_eid + c_one_i32, c_zero_i32)
                cvt_ix = ArithValue(safe_cvt_idx).index_cast(T.index)
                raw_cnt_cvt = _lds_load_raw(cumsum_mr, cvt_ix)
                blocks_cvt = (raw_cnt_cvt + c_unit - c_one_i32) // c_unit
                padded_cvt = (raw_cnt_cvt == c_zero_i32).select(
                    c_zero_i32, blocks_cvt * c_unit
                )
                # Valid threads write padded value; invalid threads write 0 to cumsum[0]
                _lds_store_raw(
                    cumsum_mr, cvt_valid.select(padded_cvt, c_zero_i32), cvt_ix
                )
            gpu.barrier()

            if has_mask:
                # EP: zero padded count for masked experts in a separate pass.
                # Loading from mask buffer inside the padded-count loop above interfered
                # with expert 0 (MLIR codegen issue). Separate pass avoids this.
                for i_ep in range_constexpr(0, E, ONESHOT_BLOCK):
                    ep_eid = fx.Int32(i_ep) + tid
                    ep_valid = ep_eid < c_E
                    ep_safe_eid = ep_valid.select(ep_eid, c_zero_i32)
                    ep_m = buffer_ops.buffer_load(
                        mask_rsrc, ep_safe_eid, vec_width=1, dtype=T.i32
                    )
                    should_zero = ep_valid & (ep_m == c_zero_i32)
                    ep_cs_ix = ArithValue(
                        ep_valid.select(ep_eid + c_one_i32, c_zero_i32)
                    ).index_cast(T.index)
                    _lds_store_raw(
                        cumsum_mr,
                        should_zero.select(
                            c_zero_i32, _lds_load_raw(cumsum_mr, ep_cs_ix)
                        ),
                        ep_cs_ix,
                    )
                gpu.barrier()

            # Step 2: All-wave parallel prefix sum (cumsum → cumdup).
            # All threads read cumsum[tid+1] (in chunks for E > ONESHOT_BLOCK)
            for _ps_chunk in range_constexpr(0, E, ONESHOT_BLOCK):
                ps_eid = fx.Int32(_ps_chunk) + tid
                ps_valid = ps_eid < c_E
                ps_safe_ix = ArithValue(
                    ps_valid.select(ps_eid + c_one_i32, c_zero_i32)
                ).index_cast(T.index)
                ps_val = ps_valid.select(
                    _lds_load_raw(cumsum_mr, ps_safe_ix), c_zero_i32
                )
                _lds_store_raw(cumdup_mr, ps_val, ps_safe_ix)
            _lds_store_raw(cumdup_mr, c_zero_i32, c_zero_i32)
            gpu.barrier()

            # DPP prefix sum — all NUM_WAVES waves active
            ps_tid_valid = tid < c_E
            val = ps_tid_valid.select(
                _lds_load_raw(cumdup_mr, tid + c_one_i32), c_zero_i32
            )
            _, inclusive_ps = _allwave_inclusive_prefix_sum(
                val, lane, wave, scratch_mr, NUM_WAVES, WARP_SIZE
            )
            _lds_store_raw(
                cumdup_mr,
                ps_tid_valid.select(inclusive_ps, c_zero_i32),
                ArithValue(ps_tid_valid.select(tid + c_one_i32, c_zero_i32)).index_cast(
                    T.index
                ),
            )
            gpu.barrier()

            # For E > ONESHOT_BLOCK: thread 0 serially extends
            if E > ONESHOT_BLOCK:
                if is_t0:
                    _extend_prefix_sum_serial(
                        cumdup_mr, ONESHOT_BLOCK, E, _lds_load_raw, _lds_store_raw
                    )
                gpu.barrier()

            # cumdup[0] = 0
            _lds_store_raw(cumdup_mr, c_zero_i32, c_zero_i32)
            gpu.barrier()

            # Write num_valid_ids from cumdup[E]
            cs_E_ix_ps = ArithValue(c_E).index_cast(T.index)
            total_padded = _lds_load_raw(cumdup_mr, cs_E_ix_ps)
            buffer_ops.buffer_store(total_padded, nvalid_rsrc, c_zero_i32)
            buffer_ops.buffer_store(tokens, nvalid_rsrc, c_one_i32)
            gpu.barrier()

            # Copy cumdup → cumsum (all threads, one expert per thread)
            for i_cp in range_constexpr(0, E + 1, ONESHOT_BLOCK):
                cp_idx = fx.Int32(i_cp) + tid
                cp_valid = cp_idx <= c_E
                safe_cp_idx = cp_valid.select(cp_idx, c_zero_i32)
                cp_ix = ArithValue(safe_cp_idx).index_cast(T.index)
                cp_val = _lds_load_raw(cumdup_mr, cp_ix)
                _lds_store_raw(cumsum_mr, cp_val, cp_ix)
            gpu.barrier()

            if has_mask:
                # EP: Compute mask cumsum in cumdup for local expert index mapping.
                # cumdup[eid] = exclusive prefix sum of mask[0..eid-1] = local expert index.
                for i_ml in range_constexpr(0, E, ONESHOT_BLOCK):
                    ml_eid = fx.Int32(i_ml) + tid
                    ml_valid = ml_eid < c_E
                    safe_ml_eid = ml_valid.select(ml_eid, c_zero_i32)
                    ml_mask = buffer_ops.buffer_load(
                        mask_rsrc, safe_ml_eid, vec_width=1, dtype=T.i32
                    )
                    ml_val = ml_valid.select(ml_mask, c_zero_i32)
                    ml_ix = ArithValue(
                        ml_valid.select(ml_eid + c_one_i32, c_zero_i32)
                    ).index_cast(T.index)
                    _lds_store_raw(cumdup_mr, ml_val, ml_ix)
                _lds_store_raw(cumdup_mr, c_zero_i32, c_zero_i32)
                gpu.barrier()

                # All-wave DPP prefix sum over mask values in cumdup
                m_tid_valid = tid < c_E
                mval = m_tid_valid.select(
                    _lds_load_raw(cumdup_mr, tid + c_one_i32), c_zero_i32
                )
                _, inclusive_m = _allwave_inclusive_prefix_sum(
                    mval, lane, wave, scratch_mr, NUM_WAVES, WARP_SIZE
                )
                _lds_store_raw(
                    cumdup_mr,
                    m_tid_valid.select(inclusive_m, c_zero_i32),
                    ArithValue(
                        m_tid_valid.select(tid + c_one_i32, c_zero_i32)
                    ).index_cast(T.index),
                )
                gpu.barrier()

                if E > ONESHOT_BLOCK:
                    if is_t0:
                        _extend_prefix_sum_serial(
                            cumdup_mr, ONESHOT_BLOCK, E, _lds_load_raw, _lds_store_raw
                        )
                    gpu.barrier()

                _lds_store_raw(cumdup_mr, c_zero_i32, c_zero_i32)
                gpu.barrier()
            else:
                # No mask: cumdup[eid] = eid (identity mapping)
                for i_ml in range_constexpr(0, E, ONESHOT_BLOCK):
                    ml_eid = fx.Int32(i_ml) + tid
                    ml_valid = ml_eid < c_E
                    safe_ml_eid = ml_valid.select(ml_eid, c_zero_i32)
                    ml_ix = ArithValue(safe_ml_eid).index_cast(T.index)
                    _lds_store_raw(
                        cumdup_mr, ml_valid.select(safe_ml_eid, c_zero_i32), ml_ix
                    )
                gpu.barrier()

            # Write sorted_expert_ids — predicated stores to buffer (safe: buffer_store ignores OOB)
            # EP: use cumdup[eid] as local expert index instead of global eid
            for i_eid in range_constexpr(0, E, ONESHOT_BLOCK):
                eid_wr = fx.Int32(i_eid) + tid
                eid_wr_valid = eid_wr < c_E
                safe_eid_wr = eid_wr_valid.select(eid_wr, c_zero_i32)

                cs_start_ix = ArithValue(safe_eid_wr).index_cast(T.index)
                cs_end_ix = ArithValue(safe_eid_wr + c_one_i32).index_cast(T.index)
                e_start = _lds_load_raw(cumsum_mr, cs_start_ix)
                e_end = eid_wr_valid.select(
                    _lds_load_raw(cumsum_mr, cs_end_ix), e_start
                )
                local_eid = _lds_load_raw(cumdup_mr, cs_start_ix)

                # Store cumdup: reuse cumdup for scatter phase position tracking.
                # Write e_start to cumdup[eid] (overwriting mask cumsum, no longer needed).
                _lds_store_raw(cumdup_mr, e_start, cs_start_ix)

                blk_start = e_start // c_unit
                blk_end = e_end // c_unit
                n_blks_wr = eid_wr_valid.select(blk_end - blk_start, c_zero_i32)
                _write_expert_id_blocks(sorted_e_rsrc, local_eid, blk_start, n_blks_wr)
            gpu.barrier()

            # Store cumdup[E] = cumsum[E].
            # All threads write cumE to cumdup[E] (all write the same value, no race).
            cs_E_ix = ArithValue(c_E).index_cast(T.index)
            cumE = _lds_load_raw(cumsum_mr, cs_E_ix)
            _lds_store_raw(cumdup_mr, cumE, cs_E_ix)
            gpu.barrier()

            # ====================== PRE-FILL: Sentinel fill (cooperative) ===========
            total_padded_pre = _lds_load_raw(
                cumdup_mr, ArithValue(c_E).index_cast(T.index)
            )
            _fill_sentinel_slots(
                sorted_ids_rsrc,
                sorted_w_rsrc,
                c_zero_i32,
                total_padded_pre,
                c_sentinel | tokens,
                ONESHOT_BLOCK,
                tid,
                c_oob_idx,
            )
            gpu.barrier()

            # ====================== PHASE 3: Scatter ==============================
            for i_e2 in range_constexpr(0, E, ONESHOT_BLOCK // 8):
                eid_sc = fx.Int32(i_e2) + lane_group_id
                eid_sc_valid = eid_sc < c_E
                # Invalid lane groups map to cumsum[E] (the total count) instead of
                # cumsum[0] to avoid racing with lane_group 0's position write-back.
                safe_eid_sc = eid_sc_valid.select(eid_sc, c_E)

                sc_expert_enabled = eid_sc_valid
                if has_mask:
                    # EP: check if this expert is masked (skip scatter for masked experts)
                    sc_mask_val = buffer_ops.buffer_load(
                        mask_rsrc,
                        eid_sc_valid.select(eid_sc, c_zero_i32),
                        vec_width=1,
                        dtype=T.i32,
                    )
                    sc_expert_enabled = eid_sc_valid & (sc_mask_val != c_zero_i32)

                cs_sc_ix = ArithValue(safe_eid_sc).index_cast(T.index)
                position = _lds_load_raw(cumsum_mr, cs_sc_ix)

                for i_sub2 in range_constexpr(0, sub_tokens, 8):
                    # This lane handles sub_token (i_sub2 + lane_group_os).
                    my_sub = fx.Int32(i_sub2) + lane_group_os
                    my_sub_valid = sc_expert_enabled & (my_sub < c_sub_tokens)
                    safe_my_sub = my_sub_valid.select(my_sub, c_zero_i32)
                    my_mesh_addr = safe_my_sub * c_smem_cols + safe_eid_sc
                    my_mesh_ix = ArithValue(my_mesh_addr).index_cast(T.index)
                    my_x = _lds_load_raw(mesh_mr, my_mesh_ix)
                    my_has_token = my_sub_valid & (my_x != c_zero_i32)
                    local_cnt = my_has_token.select(c_one_i32, c_zero_i32)

                    # 8-lane group prefix sum (NOT full-wave — uses lane_group_os,
                    # only shifts 1,2,4, no cross-row bpermute needed).
                    cnt_raw = _unwrap_val(local_cnt)
                    zero_raw = _unwrap_val(c_zero_i32)

                    # row_shr:1
                    remote = fly_rocdl.update_dpp(
                        T.i32,
                        zero_raw,
                        cnt_raw,
                        DPP_ROW_SHR_1,
                        DPP_ROW_MASK,
                        DPP_BANK_MASK,
                        True,
                    )
                    should_add = lane_group_os >= c_one_i32
                    local_cnt = should_add.select(
                        local_cnt + fx.Int32(remote), local_cnt
                    )

                    # row_shr:2
                    cnt_raw = _unwrap_val(local_cnt)
                    remote = fly_rocdl.update_dpp(
                        T.i32,
                        zero_raw,
                        cnt_raw,
                        DPP_ROW_SHR_2,
                        DPP_ROW_MASK,
                        DPP_BANK_MASK,
                        True,
                    )
                    should_add = lane_group_os >= fx.Int32(2)
                    local_cnt = should_add.select(
                        local_cnt + fx.Int32(remote), local_cnt
                    )

                    # row_shr:4
                    cnt_raw = _unwrap_val(local_cnt)
                    remote = fly_rocdl.update_dpp(
                        T.i32,
                        zero_raw,
                        cnt_raw,
                        DPP_ROW_SHR_4,
                        DPP_ROW_MASK,
                        DPP_BANK_MASK,
                        True,
                    )
                    should_add = lane_group_os >= fx.Int32(4)
                    local_cnt = should_add.select(
                        local_cnt + fx.Int32(remote), local_cnt
                    )

                    # Broadcast batch total from last lane of group via ds_bpermute
                    last_lane_of_group = tid | fx.Int32(7)  # tid with lower 3 bits set
                    last_addr = last_lane_of_group * c4_i32
                    batch_total = fly_rocdl.ds_bpermute(T.i32, last_addr, local_cnt)
                    batch_total = fx.Int32(batch_total)

                    # Scatter this lane's token
                    slot = position + local_cnt - c_one_i32
                    safe_x = my_has_token.select(my_x, c_one_i32)
                    topk_slot_sc = safe_x - c_one_i32
                    packed_id = (topk_slot_sc << fx.Int32(24)) | my_sub
                    safe_slot = my_has_token.select(slot, c_oob_idx)
                    buffer_ops.buffer_store(packed_id, sorted_ids_rsrc, safe_slot)

                    w_addr = my_has_token.select(
                        my_sub * c_topk + topk_slot_sc, c_zero_i32
                    )
                    w_val_i32 = buffer_ops.buffer_load(
                        weights_rsrc, w_addr, vec_width=1, dtype=T.i32
                    )
                    buffer_ops.buffer_store(w_val_i32, sorted_w_rsrc, safe_slot)

                    # Advance position by batch total
                    position = position + batch_total

                # Write back updated position (for padding phase).
                # Invalid lane groups write position (=0+0=0) to cumsum[0] which is harmless.
                _lds_store_raw(cumsum_mr, position, cs_sc_ix)
            gpu.barrier()

            # Padding already filled by PRE-FILL phase above (before scatter).

    @flyc.jit
    def launch_moe_sorting_oneshot(
        topk_ids_tensor: fx.Tensor,
        topk_weights_tensor: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_weights_out: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids_out: fx.Tensor,
        moe_buf: fx.Tensor,
        expert_mask_tensor: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_moe_buf_elems: fx.Int32,
        n_grid_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = moe_sorting_oneshot_kernel(
            topk_ids_tensor,
            topk_weights_tensor,
            sorted_token_ids,
            sorted_weights_out,
            sorted_expert_ids,
            num_valid_ids_out,
            moe_buf,
            expert_mask_tensor,
            i32_tokens,
            i32_moe_buf_elems,
        )
        launcher.launch(
            grid=(n_grid_blocks, 1, 1),
            block=(ONESHOT_BLOCK, 1, 1),
            stream=stream,
        )

    return launch_moe_sorting_oneshot


# ---------------------------------------------------------------------------
# FlyDSL GPU kernels — multiphase path (2 or 4 kernels, large T via HBM workspace)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=64)
def _p23_block_size(num_experts: int, tokens: int) -> int:
    """Runtime P23 block size: balance E>256 prefix path vs large-T mesh scan."""
    if num_experts <= 256:
        return 256
    if tokens > P23_LARGE_T_THRESHOLD:
        return 256
    if num_experts <= 512:
        return 512
    return 256


@functools.lru_cache(maxsize=256)
def _compile_moe_sorting_multiphase(
    *,
    num_experts: int,
    topk: int,
    unit_size: int = UNIT_SIZE,
    has_mask: bool = False,
    k4_block: int = 256,
):
    """Compile the multiphase MoE sorting kernels (2 or 4 kernels via HBM workspace).

    For token counts exceeding LDS capacity, uses HBM workspace:
      K1: ClearWorkspace — zero the workspace buffer
      K2: P0 scatter     — scatter topk_ids into expert mesh in HBM
      K3: P1 count       — one block per expert, count non-zero mesh cells
      K4: P23 prefix-sum + scatter — prefix-sum on counts, scatter tokens,
          fill sorted_expert_ids, zero moe_buf
      P0_v2: Fused clear+scatter+count — replaces K1+K2+K3 for T <= 2048

    Workspace layout (i32 elements):
      [0 .. ws_mesh_i32)                : uint8 expert mesh (E rows x mesh_stride bytes, packed into i32)
      [ws_mesh_i32 .. ws_mesh_i32 + E+1): expert_cumsum (E+1 i32 entries)

    Parameters
    ----------
    num_experts : int
        Number of routed experts (e.g. 256 for DeepSeek R1).
    topk : int
        Experts per token (e.g. 8).
    unit_size : int
        GEMM tile-M for padding alignment (default 32).
    """
    E = num_experts

    @flyc.jit
    def _extend_local_idx_for_extra_experts(
        cumsum_mr, mask_rsrc, K4_BLOCK, E, has_mask
    ):
        """Thread-0: write local expert indices for experts >= K4_BLOCK to cumsum_mr."""
        if has_mask:
            prev_local = _lds_load_raw(cumsum_mr, fx.Int32(K4_BLOCK - 1))
            prev_mask = buffer_ops.buffer_load(
                mask_rsrc, fx.Int32(K4_BLOCK - 1), vec_width=1, dtype=T.i32
            )
            prev_local = prev_local + prev_mask
            for _e3 in range_constexpr(K4_BLOCK, E):
                e3_mask = buffer_ops.buffer_load(
                    mask_rsrc, fx.Int32(_e3), vec_width=1, dtype=T.i32
                )
                _lds_store_raw(cumsum_mr, prev_local, fx.Int32(_e3))
                prev_local = prev_local + e3_mask
        else:
            for _e3 in range_constexpr(K4_BLOCK, E):
                _lds_store_raw(cumsum_mr, fx.Int32(_e3), fx.Int32(_e3))

    @flyc.jit
    def _p23_scatter_mesh(
        tid,
        scatter_mr,
        ws_rsrc,
        weights_rsrc,
        sorted_ids_rsrc,
        sorted_w_rsrc,
        mask_rsrc,
        my_expert,
        my_start,
        my_end,
        i32_mesh_stride,
        c_topk,
        K4_BLOCK,
        has_mask,
    ):
        """P23 Step 4: EP mask check, read uint8 mesh, DPP prefix sum, scatter tokens."""
        lane = tid % WARP_SIZE
        wave = tid // WARP_SIZE
        K4_NUM_WAVES = K4_BLOCK // WARP_SIZE
        c_zero, c_one, c4 = fx.Int32(0), fx.Int32(1), fx.Int32(4)
        c_ff, c_oob_idx = fx.Int32(0xFF), fx.Int32(0x7FFFFFFF)
        p23_bid_enabled = c_one != c_zero
        if has_mask:
            p23_bid_mask = buffer_ops.buffer_load(
                mask_rsrc, my_expert, vec_width=1, dtype=T.i32
            )
            p23_bid_enabled = p23_bid_mask != c_zero
        i32_words_per_row = i32_mesh_stride >> fx.Int32(2)
        n_mesh_iters = (my_start != my_end).select(
            (i32_words_per_row + fx.Int32(K4_BLOCK - 1)) // fx.Int32(K4_BLOCK), c_zero
        )
        mesh_row_i32_base = (my_expert * i32_mesh_stride) >> fx.Int32(2)
        for _si, state in range(
            fx.Index(0),
            ArithValue(n_mesh_iters).index_cast(T.index),
            fx.Index(1),
            init=[my_start],
        ):
            position = state[0]
            word_idx = fx.Int32(_si) * fx.Int32(K4_BLOCK) + tid
            col_valid = p23_bid_enabled & (word_idx < i32_words_per_row)
            safe_word_idx = col_valid.select(word_idx, c_zero)
            word = buffer_ops.buffer_load(
                ws_rsrc, mesh_row_i32_base + safe_word_idx, vec_width=1, dtype=T.i32
            )
            x0 = word & c_ff
            x1 = (word >> fx.Int32(8)) & c_ff
            x2 = (word >> fx.Int32(16)) & c_ff
            x3 = (word >> fx.Int32(24)) & c_ff
            base_col = word_idx * c4
            h0 = col_valid & (x0 != c_zero)
            h1 = col_valid & (x1 != c_zero)
            h2 = col_valid & (x2 != c_zero)
            h3 = col_valid & (x3 != c_zero)
            my_cnt = (
                h0.select(c_one, c_zero)
                + h1.select(c_one, c_zero)
                + h2.select(c_one, c_zero)
                + h3.select(c_one, c_zero)
            )
            my_pre_scan = my_cnt
            my_cnt, my_cnt_inclusive = _allwave_inclusive_prefix_sum(
                my_cnt, lane, wave, scatter_mr, K4_NUM_WAVES, WARP_SIZE
            )
            wave_offset = my_cnt_inclusive - my_cnt
            batch_total = c_zero
            for _w in range_constexpr(K4_NUM_WAVES):
                batch_total = batch_total + _lds_load_raw(scatter_mr, fx.Int32(_w))
            gpu.barrier()
            my_exclusive = my_cnt - my_pre_scan + wave_offset
            scatter_base = position + my_exclusive
            pid_0 = (h0.select(x0 - c_one, c_zero) << fx.Int32(24)) | base_col
            pid_1 = (h1.select(x1 - c_one, c_zero) << fx.Int32(24)) | (base_col + c_one)
            pid_2 = (h2.select(x2 - c_one, c_zero) << fx.Int32(24)) | (
                base_col + fx.Int32(2)
            )
            pid_3 = (h3.select(x3 - c_one, c_zero) << fx.Int32(24)) | (
                base_col + fx.Int32(3)
            )
            safe_slot_0 = h0.select(scatter_base, c_oob_idx)
            off1 = scatter_base + h0.select(c_one, c_zero)
            safe_slot_1 = h1.select(off1, c_oob_idx)
            off2 = off1 + h1.select(c_one, c_zero)
            safe_slot_2 = h2.select(off2, c_oob_idx)
            off3 = off2 + h2.select(c_one, c_zero)
            safe_slot_3 = h3.select(off3, c_oob_idx)
            w_val_0 = buffer_ops.buffer_load(
                weights_rsrc,
                h0.select(base_col * c_topk + h0.select(x0 - c_one, c_zero), c_zero),
                vec_width=1,
                dtype=T.i32,
            )
            w_val_1 = buffer_ops.buffer_load(
                weights_rsrc,
                h1.select(
                    (base_col + c_one) * c_topk + h1.select(x1 - c_one, c_zero), c_zero
                ),
                vec_width=1,
                dtype=T.i32,
            )
            w_val_2 = buffer_ops.buffer_load(
                weights_rsrc,
                h2.select(
                    (base_col + fx.Int32(2)) * c_topk + h2.select(x2 - c_one, c_zero),
                    c_zero,
                ),
                vec_width=1,
                dtype=T.i32,
            )
            w_val_3 = buffer_ops.buffer_load(
                weights_rsrc,
                h3.select(
                    (base_col + fx.Int32(3)) * c_topk + h3.select(x3 - c_one, c_zero),
                    c_zero,
                ),
                vec_width=1,
                dtype=T.i32,
            )
            buffer_ops.buffer_store(pid_0, sorted_ids_rsrc, safe_slot_0)
            buffer_ops.buffer_store(pid_1, sorted_ids_rsrc, safe_slot_1)
            buffer_ops.buffer_store(pid_2, sorted_ids_rsrc, safe_slot_2)
            buffer_ops.buffer_store(pid_3, sorted_ids_rsrc, safe_slot_3)
            buffer_ops.buffer_store(w_val_0, sorted_w_rsrc, safe_slot_0)
            buffer_ops.buffer_store(w_val_1, sorted_w_rsrc, safe_slot_1)
            buffer_ops.buffer_store(w_val_2, sorted_w_rsrc, safe_slot_2)
            buffer_ops.buffer_store(w_val_3, sorted_w_rsrc, safe_slot_3)
            pos_next = position + batch_total
            results = yield [pos_next]
        return results

    # --- K1: ClearWorkspace kernel -------------------------------------------
    # CK uses grid=262144, block=1024 (1 store per thread, no loop).
    # Match that: block=1024, grid=ceil(ws_total/1024).
    K1_BLOCK = 1024

    @flyc.kernel(known_block_size=[K1_BLOCK, 1, 1])
    def clear_workspace_kernel(
        workspace: fx.Tensor,
        i32_total_elems: fx.Int32,
    ):
        gid = gpu.block_idx.x * fx.Int32(K1_BLOCK) + gpu.thread_idx.x
        ws_rsrc = buffer_ops.create_buffer_resource(workspace, max_size=True)
        c_zero = fx.Int32(0)

        # Each thread stores exactly one element (no loop needed).
        valid = gid < i32_total_elems
        buffer_ops.buffer_store(c_zero, ws_rsrc, valid.select(gid, c_zero))

    @flyc.jit
    def launch_clear_ws(
        workspace: fx.Tensor,
        i32_total_elems: fx.Int32,
        n_grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = clear_workspace_kernel(workspace, i32_total_elems)
        launcher.launch(grid=(n_grid, 1, 1), block=(K1_BLOCK, 1, 1), stream=stream)

    # --- K2: P0 scatter kernel -----------------------------------------------
    # uint8 mesh: stores topk_slot+1 (max 9) as a single byte directly.
    # mesh_stride is in bytes; byte_offset = eid * mesh_stride + token_id.
    # No two threads write the same byte (unique experts per token).
    K2_BLOCK = 256

    @flyc.kernel
    def p0_scatter_kernel(
        topk_ids: fx.Tensor,
        workspace: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_mesh_stride: fx.Int32,
        i32_niters: fx.Int32,
    ):
        gid = gpu.block_idx.x * fx.Int32(K2_BLOCK) + gpu.thread_idx.x
        stride = gpu.grid_dim.x * fx.Int32(K2_BLOCK)
        topk_rsrc = buffer_ops.create_buffer_resource(topk_ids, max_size=True)
        ws_rsrc = buffer_ops.create_buffer_resource(workspace, max_size=True)
        c_zero = fx.Int32(0)
        c_topk = fx.Int32(topk)
        c_one = fx.Int32(1)

        total = i32_tokens * c_topk

        _s = fx.Index(0)
        _e = ArithValue(i32_niters).index_cast(T.index)
        _one = fx.Index(1)
        for _i in range(_s, _e, _one):
            flat = gid + fx.Int32(_i) * stride
            valid = flat < total
            safe_flat = valid.select(flat, c_zero)
            token_id = safe_flat // c_topk
            topk_slot = safe_flat % c_topk
            eid = buffer_ops.buffer_load(topk_rsrc, safe_flat, vec_width=1, dtype=T.i32)
            byte_offset = eid * i32_mesh_stride + token_id
            val_i8 = ArithValue(topk_slot + c_one).trunci(T.i8)
            if valid:
                buffer_ops.buffer_store(
                    val_i8, ws_rsrc, byte_offset, offset_is_bytes=True
                )

    @flyc.jit
    def launch_p0(
        topk_ids: fx.Tensor,
        workspace: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_mesh_stride: fx.Int32,
        i32_niters: fx.Int32,
        n_grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = p0_scatter_kernel(
            topk_ids, workspace, i32_tokens, i32_mesh_stride, i32_niters
        )
        launcher.launch(grid=(n_grid, 1, 1), block=(K2_BLOCK, 1, 1), stream=stream)

    # --- K3: P1 count kernel -------------------------------------------------
    # 256 threads (4 waves), vec_width=4: each thread loads 4 i32 words (16
    # mesh cells) per iteration.  4 waves provide 4x memory-level parallelism
    # vs the old 1-wave (64-thread) design, matching CK P1's block size.
    # Cross-warp reduction via LDS (4 partial sums, one per warp).
    K3_BLOCK = 256
    K3_NUM_WAVES = K3_BLOCK // WARP_SIZE
    K3_VEC_WIDTH = 4
    K3_WORDS_PER_ITER = K3_BLOCK * K3_VEC_WIDTH
    K3_WORDS_PER_ITER_LOG2 = (K3_WORDS_PER_ITER).bit_length() - 1

    @fx.struct
    class P1SharedStorage:
        reduce: fx.Array[fx.Int32, K3_NUM_WAVES, 16]

    @flyc.kernel
    def p1_count_kernel(
        workspace: fx.Tensor,
        expert_mask_tensor: fx.Tensor,
        i32_mesh_stride: fx.Int32,
        i32_mesh_size: fx.Int32,
    ):
        eid = gpu.block_idx.x
        tid = gpu.thread_idx.x
        lane = tid % WARP_SIZE
        wave = tid // WARP_SIZE

        ws_rsrc = buffer_ops.create_buffer_resource(workspace, max_size=True)
        c_zero = fx.Int32(0)
        c_one = fx.Int32(1)
        c_ff = fx.Int32(0xFF)

        reduce_mr = fx.SharedAllocator().allocate(P1SharedStorage).peek().reduce.ptr

        mesh_row_i32_base = (eid * i32_mesh_stride) >> fx.Int32(2)
        i32_words_per_row = i32_mesh_stride >> fx.Int32(2)
        n_iters = (i32_words_per_row + fx.Int32(K3_WORDS_PER_ITER - 1)) >> fx.Int32(
            K3_WORDS_PER_ITER_LOG2
        )

        if has_mask:
            mask_rsrc = buffer_ops.create_buffer_resource(
                expert_mask_tensor, max_size=True
            )
            p1_mask = buffer_ops.buffer_load(mask_rsrc, eid, vec_width=1, dtype=T.i32)
            p1_is_local = p1_mask != c_zero
            p1_should_zero = (~p1_is_local) & (tid == c_zero)
            buffer_ops.buffer_store(
                c_zero,
                ws_rsrc,
                p1_should_zero.select(i32_mesh_size + eid, fx.Int32(0x7FFFFFFF)),
            )
            n_iters = p1_is_local.select(n_iters, c_zero)

        for _i, state in range(
            fx.Index(0),
            ArithValue(n_iters).index_cast(T.index),
            fx.Index(1),
            init=[c_zero],
        ):
            cnt_so_far = state[0]

            word_base = fx.Int32(_i) * fx.Int32(K3_WORDS_PER_ITER) + tid * fx.Int32(
                K3_VEC_WIDTH
            )
            valid = word_base < i32_words_per_row
            safe_addr = mesh_row_i32_base + valid.select(word_base, c_zero)
            vec4 = buffer_ops.buffer_load(ws_rsrc, safe_addr, vec_width=4, dtype=T.i32)

            iter_cnt = c_zero
            for _wi in range_constexpr(K3_VEC_WIDTH):
                word = Vec(vec4)[_wi]
                word_valid = valid & ((word_base + fx.Int32(_wi)) < i32_words_per_row)
                b0 = word & c_ff
                b1 = (word >> fx.Int32(8)) & c_ff
                b2 = (word >> fx.Int32(16)) & c_ff
                b3 = (word >> fx.Int32(24)) & c_ff
                nz0 = word_valid.select((b0 != c_zero).select(c_one, c_zero), c_zero)
                nz1 = word_valid.select((b1 != c_zero).select(c_one, c_zero), c_zero)
                nz2 = word_valid.select((b2 != c_zero).select(c_one, c_zero), c_zero)
                nz3 = word_valid.select((b3 != c_zero).select(c_one, c_zero), c_zero)
                iter_cnt = iter_cnt + nz0 + nz1 + nz2 + nz3

            new_cnt = cnt_so_far + iter_cnt
            results = yield [new_cnt]
        cnt = results

        # Intra-warp reduce via shuffle_xor
        width_ws = fx.Int32(WARP_SIZE)
        for sh in range_constexpr(int.bit_length(WARP_SIZE) - 1):
            off = fx.Int32(1 << sh)
            peer = cnt.shuffle_xor(off, width_ws)
            cnt = cnt + peer

        # Cross-warp reduce via LDS: lane 0 of each warp writes partial sum
        is_lane0 = lane == c_zero
        if is_lane0:
            wave_ix = ArithValue(wave).index_cast(T.index)
            _lds_store_raw(reduce_mr, cnt, wave_ix)
        gpu.barrier()

        # Thread 0 sums all warp partials and writes to HBM
        is_t0 = tid == c_zero
        total = c_zero
        for _w in range_constexpr(K3_NUM_WAVES):
            total = total + _lds_load_raw(reduce_mr, fx.Int32(_w))

        cs_offset = i32_mesh_size + eid
        c_oob_idx = fx.Int32(0x7FFFFFFF)
        safe_cs = is_t0.select(cs_offset, c_oob_idx)
        buffer_ops.buffer_store(total, ws_rsrc, safe_cs)

    @flyc.jit
    def launch_p1(
        workspace: fx.Tensor,
        expert_mask_tensor: fx.Tensor,
        i32_mesh_stride: fx.Int32,
        i32_mesh_size: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = p1_count_kernel(
            workspace, expert_mask_tensor, i32_mesh_stride, i32_mesh_size
        )
        launcher.launch(grid=(E, 1, 1), block=(K3_BLOCK, 1, 1), stream=stream)

    # --- P0_v2: Fused clear+scatter+count kernel (for T <= 2048) --------------
    # Replaces K1+K2+K3 with a single kernel launch.
    # Grid: E blocks (one per expert), Block: 512 threads (matching CK P0_v2).
    # Phase 1: clear this expert's mesh row
    # Phase 2: scan all T*topk assignments, filter by expert, byte stores
    # Phase 3: popcount + warp reduce + cross-wave LDS reduce -> expert_cumsum
    P0V2_BLOCK = 512
    P0V2_NUM_WAVES = P0V2_BLOCK // WARP_SIZE

    # Power-of-2 topk: use shift to avoid division
    _p0v2_topk_is_po2 = (topk & (topk - 1)) == 0 and topk > 0
    _p0v2_topk_log2 = topk.bit_length() - 1 if _p0v2_topk_is_po2 else 0

    # LDS for cross-wave reduction (same layout as K3)
    @fx.struct
    class P0V2SharedStorage:
        reduce: fx.Array[fx.Int32, P0V2_NUM_WAVES, 16]

    @flyc.kernel(known_block_size=[P0V2_BLOCK, 1, 1])
    def p0v2_kernel(
        topk_ids: fx.Tensor,
        workspace: fx.Tensor,
        expert_mask_tensor: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_mesh_stride: fx.Int32,
        i32_mesh_size: fx.Int32,
    ):
        eid = gpu.block_idx.x
        tid = gpu.thread_idx.x
        lane = tid % WARP_SIZE
        wave = tid // WARP_SIZE

        ws_rsrc = buffer_ops.create_buffer_resource(workspace, max_size=True)
        mask_rsrc = buffer_ops.create_buffer_resource(expert_mask_tensor, max_size=True)
        topk_rsrc = buffer_ops.create_buffer_resource(topk_ids, max_size=True)
        c_zero = fx.Int32(0)
        c_oob = fx.Int32(0x7FFFFFFF)
        c_one = fx.Int32(1)
        c_ff = fx.Int32(0xFF)
        c_topk = fx.Int32(topk)
        c_block = fx.Int32(P0V2_BLOCK)

        reduce_mr = fx.SharedAllocator().allocate(P0V2SharedStorage).peek().reduce.ptr

        # Precompute mesh row base (in i32 words) and words per row
        mesh_row_i32_base = (eid * i32_mesh_stride) >> fx.Int32(2)
        i32_words_per_row = i32_mesh_stride >> fx.Int32(2)

        clear_niters = (i32_words_per_row + fx.Int32(P0V2_BLOCK - 1)) >> fx.Int32(9)
        total_assignments = i32_tokens * c_topk
        scatter_niters = (total_assignments + fx.Int32(P0V2_BLOCK - 1)) >> fx.Int32(9)

        # EP: load mask, write cumsum=0 for masked experts, set loop bounds to 0
        if has_mask:
            m_val = buffer_ops.buffer_load(mask_rsrc, eid, vec_width=1, dtype=T.i32)
            is_local_expert = m_val != c_zero
            should_write_zero = (~is_local_expert) & (tid == c_zero)
            buffer_ops.buffer_store(
                c_zero, ws_rsrc, should_write_zero.select(i32_mesh_size + eid, c_oob)
            )
            clear_niters = is_local_expert.select(clear_niters, c_zero)
            scatter_niters = is_local_expert.select(scatter_niters, c_zero)

        # ---- Phase 1: Clear this expert's mesh row ----
        for _ci in range(
            fx.Index(0), ArithValue(clear_niters).index_cast(T.index), fx.Index(1)
        ):
            word_idx = fx.Int32(_ci) * c_block + tid
            valid = word_idx < i32_words_per_row
            safe_idx = mesh_row_i32_base + valid.select(word_idx, c_zero)
            buffer_ops.buffer_store(c_zero, ws_rsrc, valid.select(safe_idx, c_oob))

        gpu.barrier()

        # ---- Phase 2: Scatter (scan all T*topk, filter by expert) ----
        for _si in range(
            fx.Index(0), ArithValue(scatter_niters).index_cast(T.index), fx.Index(1)
        ):
            flat = fx.Int32(_si) * c_block + tid
            valid = flat < total_assignments
            safe_flat = valid.select(flat, c_zero)

            token_id = (
                safe_flat >> fx.Int32(_p0v2_topk_log2)
                if _p0v2_topk_is_po2
                else safe_flat // c_topk
            )
            topk_slot = (
                safe_flat & fx.Int32(topk - 1)
                if _p0v2_topk_is_po2
                else safe_flat % c_topk
            )

            expert_id = buffer_ops.buffer_load(
                topk_rsrc, safe_flat, vec_width=1, dtype=T.i32
            )

            is_mine = valid & (expert_id == eid)
            byte_offset = eid * i32_mesh_stride + token_id
            val_i8 = ArithValue(is_mine.select(topk_slot + c_one, c_zero)).trunci(T.i8)
            # Byte-mode buffer_store with OOB offset crashes on AMD GPUs.
            # Use conditional branch to skip the store for non-matching threads.
            if is_mine:
                buffer_ops.buffer_store(
                    val_i8, ws_rsrc, byte_offset, offset_is_bytes=True
                )

        gpu.barrier()

        # ---- Phase 3: Count non-zero bytes + warp/cross-wave reduce ----
        count_niters = clear_niters  # same loop structure, reuse (already EP-gated)
        for _ki, state in range(
            fx.Index(0),
            ArithValue(count_niters).index_cast(T.index),
            fx.Index(1),
            init=[c_zero],
        ):
            cnt_so_far = state[0]

            word_base = fx.Int32(_ki) * c_block + tid
            valid = word_base < i32_words_per_row
            safe_addr = mesh_row_i32_base + valid.select(word_base, c_zero)
            word = buffer_ops.buffer_load(ws_rsrc, safe_addr, vec_width=1, dtype=T.i32)

            b0 = word & c_ff
            b1 = (word >> fx.Int32(8)) & c_ff
            b2 = (word >> fx.Int32(16)) & c_ff
            b3 = (word >> fx.Int32(24)) & c_ff
            nz0 = valid.select((b0 != c_zero).select(c_one, c_zero), c_zero)
            nz1 = valid.select((b1 != c_zero).select(c_one, c_zero), c_zero)
            nz2 = valid.select((b2 != c_zero).select(c_one, c_zero), c_zero)
            nz3 = valid.select((b3 != c_zero).select(c_one, c_zero), c_zero)
            iter_cnt = nz0 + nz1 + nz2 + nz3

            new_cnt = cnt_so_far + iter_cnt
            results = yield [new_cnt]
        cnt = results

        # Intra-warp reduce via shuffle_xor
        width_ws = fx.Int32(WARP_SIZE)
        for sh in range_constexpr(int.bit_length(WARP_SIZE) - 1):
            off = fx.Int32(1 << sh)
            peer = cnt.shuffle_xor(off, width_ws)
            cnt = cnt + peer

        # Cross-warp reduce via LDS: lane 0 of each warp writes partial sum
        is_lane0 = lane == c_zero
        if is_lane0:
            wave_ix = ArithValue(wave).index_cast(T.index)
            _lds_store_raw(reduce_mr, cnt, wave_ix)
        gpu.barrier()

        # Thread 0 sums all warp partials and writes to HBM
        is_t0 = tid == c_zero
        total = c_zero
        for _w in range_constexpr(P0V2_NUM_WAVES):
            total = total + _lds_load_raw(reduce_mr, fx.Int32(_w))

        cs_offset = i32_mesh_size + eid
        c_oob_idx = fx.Int32(0x7FFFFFFF)
        safe_cs = is_t0.select(cs_offset, c_oob_idx)
        buffer_ops.buffer_store(total, ws_rsrc, safe_cs)

    @flyc.jit
    def launch_p0v2(
        topk_ids: fx.Tensor,
        workspace: fx.Tensor,
        expert_mask_tensor: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_mesh_stride: fx.Int32,
        i32_mesh_size: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = p0v2_kernel(
            topk_ids,
            workspace,
            expert_mask_tensor,
            i32_tokens,
            i32_mesh_stride,
            i32_mesh_size,
        )
        launcher.launch(grid=(E, 1, 1), block=(P0V2_BLOCK, 1, 1), stream=stream)

    # --- K4: P23 prefix-sum + scatter + moe_buf zeroing ---------------------
    # Parallel design (matching CK P23): each block [0, E) independently
    # computes the SAME prefix sum, then scatters ONLY for expert blockIdx.x.
    # No inter-block barrier needed — redundant prefix sums are deterministic.
    # P23 thread-block width (256 or 512). See _p23_block_size().
    K4_BLOCK = k4_block

    # LDS: cumsum[E+1] for prefix sums + cross-wave scratch for DPP scan
    K4_NUM_WAVES = K4_BLOCK // WARP_SIZE
    k4_smem_cols = max(E + 1, K4_BLOCK + 1)

    @fx.struct
    class K4SharedStorage:
        cumsum: fx.Array[fx.Int32, k4_smem_cols, 16]
        scatter: fx.Array[fx.Int32, K4_NUM_WAVES, 16]

    @flyc.kernel(known_block_size=[K4_BLOCK, 1, 1])
    def p23_kernel(
        workspace: fx.Tensor,
        topk_weights_tensor: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_weights_out: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        moe_buf: fx.Tensor,
        expert_mask_tensor: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_mesh_stride: fx.Int32,
        i32_mesh_size: fx.Int32,
        i32_moe_buf_elems: fx.Int32,
    ):
        bid = gpu.block_idx.x
        tid = gpu.thread_idx.x
        lane = tid % WARP_SIZE
        wave = tid // WARP_SIZE
        c_zero = fx.Int32(0)
        c_one = fx.Int32(1)
        c_E = fx.Int32(E)
        c_unit = fx.Int32(unit_size)
        c_topk = fx.Int32(topk)
        c_sentinel = fx.Int32(topk << 24)
        c_oob_idx = fx.Int32(0x7FFFFFFF)

        # Buffer resources
        ws_rsrc = buffer_ops.create_buffer_resource(workspace, max_size=True)
        weights_rsrc = buffer_ops.create_buffer_resource(
            topk_weights_tensor, max_size=True
        )
        sorted_ids_rsrc = buffer_ops.create_buffer_resource(
            sorted_token_ids, max_size=True
        )
        sorted_w_rsrc = buffer_ops.create_buffer_resource(
            sorted_weights_out, max_size=True
        )
        mask_rsrc = buffer_ops.create_buffer_resource(expert_mask_tensor, max_size=True)

        # LDS: cumsum[E+1] for prefix sums + cross-wave scratch
        lds = fx.SharedAllocator().allocate(K4SharedStorage).peek()
        cumsum_mr = lds.cumsum.ptr
        scatter_mr = lds.scatter.ptr

        is_sort_block = bid < c_E
        is_zero_block = bid >= c_E

        # ================ MOE_BUF ZEROING (blocks >= E) ==================
        if is_zero_block:
            moe_buf_rsrc = buffer_ops.create_buffer_resource(moe_buf, max_size=True)
            zero_gid_v4 = (bid - c_E) * fx.Int32(K4_BLOCK) + tid
            zero_stride_v4 = (gpu.grid_dim.x - c_E) * fx.Int32(K4_BLOCK)
            _zero_moe_buf_grid_stride(
                moe_buf_rsrc,
                zero_gid_v4,
                zero_stride_v4,
                i32_moe_buf_elems >> fx.Int32(2),
                c_oob_idx,
            )

        # ================ PARALLEL PREFIX-SUM + MESH SCATTER (blocks 0..E-1) ==
        # Each block independently: prefix sum (redundant), scatter for its expert only.
        if is_sort_block:
            my_expert = bid

            # Step 1: Load expert counts from workspace -> pad to unit_size -> LDS cumsum
            # Process E experts in chunks of K4_BLOCK (256). Most models have
            # E <= 256, so the extra chunk is only needed for E > 256
            # (e.g. DeepSeek-R1 with 256 routed + 1 shared = 257).
            if tid == c_zero:
                _lds_store_raw(cumsum_mr, c_zero, c_zero)

            # EP: load this thread's own mask value BEFORE the chunked loop.
            # The chunked loop overwrites p23_mask_val in later chunks, so we
            # need a stable copy for the mask prefix sum computed after the loop.
            my_mask_val = c_one
            if has_mask:
                tid_has_expert = tid < c_E
                my_mask_val = buffer_ops.buffer_load(
                    mask_rsrc,
                    tid_has_expert.select(tid, c_zero),
                    vec_width=1,
                    dtype=T.i32,
                )
                my_mask_val = tid_has_expert.select(my_mask_val, c_zero)

            for _chunk in range_constexpr(0, E, K4_BLOCK):
                expert_idx = fx.Int32(_chunk) + tid
                tid_valid_expert = expert_idx < c_E
                ws_cs_addr = i32_mesh_size + tid_valid_expert.select(expert_idx, c_zero)
                raw_cnt = buffer_ops.buffer_load(
                    ws_rsrc, ws_cs_addr, vec_width=1, dtype=T.i32
                )
                raw_cnt = tid_valid_expert.select(raw_cnt, c_zero)
                blocks = (raw_cnt + c_unit - c_one) // c_unit
                padded = (raw_cnt == c_zero).select(c_zero, blocks * c_unit)
                if has_mask:
                    chunk_mask = buffer_ops.buffer_load(
                        mask_rsrc,
                        tid_valid_expert.select(expert_idx, c_zero),
                        vec_width=1,
                        dtype=T.i32,
                    )
                    chunk_mask = tid_valid_expert.select(chunk_mask, c_zero)
                    padded = (chunk_mask == c_zero).select(c_zero, padded)
                raw_store_idx = expert_idx + c_one
                oob = raw_store_idx >= fx.Int32(k4_smem_cols)
                safe_store_idx = oob.select(c_zero, raw_store_idx)
                safe_store_val = oob.select(c_zero, padded)
                _lds_store_raw(cumsum_mr, safe_store_val, safe_store_idx)
            gpu.barrier()

            # Step 2: Prefix sum over cumsum LDS. When E <= K4_BLOCK (256),
            # a single DPP pass covers all experts. When E > K4_BLOCK, we
            # do the DPP pass for the first K4_BLOCK elements, then serially
            # accumulate the remaining entries from thread 0.
            val = _lds_load_raw(cumsum_mr, tid + c_one)
            val, inclusive_prefix = _allwave_inclusive_prefix_sum(
                val, lane, wave, scatter_mr, K4_NUM_WAVES, WARP_SIZE
            )
            total_padded = c_zero
            for _w in range_constexpr(K4_NUM_WAVES):
                total_padded = total_padded + _lds_load_raw(scatter_mr, fx.Int32(_w))
            _lds_store_raw(cumsum_mr, inclusive_prefix, tid + c_one)
            gpu.barrier()

            # For E > K4_BLOCK: thread 0 serially extends the prefix sum
            if E > K4_BLOCK:
                if tid == c_zero:
                    total_padded = _extend_prefix_sum_serial(
                        cumsum_mr, K4_BLOCK, E, _lds_load_raw, _lds_store_raw
                    )
                gpu.barrier()
                total_padded = _lds_load_raw(cumsum_mr, c_E)

            # Read my_start and my_end from cumsum LDS
            my_start = _lds_load_raw(cumsum_mr, my_expert)
            my_end = _lds_load_raw(cumsum_mr, my_expert + c_one)

            # Hoist before if/else: AST rewriter extracts branches into
            # separate functions, so variables must be defined in outer scope.
            local_idx_p23 = tid
            if has_mask:
                _, p23_mask_inclusive = _allwave_inclusive_prefix_sum(
                    my_mask_val, lane, wave, scatter_mr, K4_NUM_WAVES, WARP_SIZE
                )
                local_idx_p23 = p23_mask_inclusive - my_mask_val

            # Block 0, thread 0 writes num_valid_ids
            if (bid == c_zero) & (tid == c_zero):
                nvalid_rsrc = buffer_ops.create_buffer_resource(
                    num_valid_ids, max_size=True
                )
                buffer_ops.buffer_store(total_padded, nvalid_rsrc, c_zero)
                buffer_ops.buffer_store(i32_tokens, nvalid_rsrc, c_one)

            # Step 3: Write sorted_expert_ids for THIS expert (using local_idx_p23 for EP)
            # Store local_idx to LDS cumsum[tid], barrier, read cumsum[my_expert]
            _lds_store_raw(cumsum_mr, local_idx_p23, tid)
            # For E > K4_BLOCK: thread 0 extends local_idx using cumsum[K4_BLOCK-1].
            # Barrier ensures all threads have written before thread 0 reads.
            if E > K4_BLOCK:
                gpu.barrier()
                if tid == c_zero:
                    _extend_local_idx_for_extra_experts(
                        cumsum_mr, mask_rsrc, K4_BLOCK, E, has_mask
                    )
            gpu.barrier()
            my_local_idx = _lds_load_raw(cumsum_mr, my_expert)

            sorted_e_rsrc = buffer_ops.create_buffer_resource(
                sorted_expert_ids, max_size=True
            )
            blk_start = my_start // c_unit
            blk_end = my_end // c_unit
            _write_expert_id_blocks(
                sorted_e_rsrc, my_local_idx, blk_start, blk_end - blk_start
            )

            # Step 4: Mesh-based scatter (EP mask + uint8 mesh read + DPP prefix sum + scatter)
            scatter_end_pos_t0 = _p23_scatter_mesh(
                tid,
                scatter_mr,
                ws_rsrc,
                weights_rsrc,
                sorted_ids_rsrc,
                sorted_w_rsrc,
                mask_rsrc,
                my_expert,
                my_start,
                my_end,
                i32_mesh_stride,
                c_topk,
                K4_BLOCK,
                has_mask,
            )

            # Step 5: Fill padding with sentinel for THIS expert (parallel)
            _fill_sentinel_slots(
                sorted_ids_rsrc,
                sorted_w_rsrc,
                scatter_end_pos_t0,
                my_end - scatter_end_pos_t0,
                c_sentinel | i32_tokens,
                K4_BLOCK,
                tid,
                c_oob_idx,
            )

    @flyc.jit
    def launch_p23(
        workspace: fx.Tensor,
        topk_weights_tensor: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_weights_out: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids_out: fx.Tensor,
        moe_buf: fx.Tensor,
        expert_mask_tensor: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_mesh_stride: fx.Int32,
        i32_mesh_size: fx.Int32,
        i32_moe_buf_elems: fx.Int32,
        n_grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = p23_kernel(
            workspace,
            topk_weights_tensor,
            sorted_token_ids,
            sorted_weights_out,
            sorted_expert_ids,
            num_valid_ids_out,
            moe_buf,
            expert_mask_tensor,
            i32_tokens,
            i32_mesh_stride,
            i32_mesh_size,
            i32_moe_buf_elems,
        )
        launcher.launch(grid=(n_grid, 1, 1), block=(K4_BLOCK, 1, 1), stream=stream)

    @flyc.jit
    def launch_p0v2_p23(
        topk_ids: fx.Tensor,
        workspace: fx.Tensor,
        topk_weights_tensor: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_weights_out: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids_out: fx.Tensor,
        moe_buf: fx.Tensor,
        expert_mask_tensor: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_mesh_stride: fx.Int32,
        i32_mesh_size: fx.Int32,
        i32_moe_buf_elems: fx.Int32,
        n_grid_p23: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        l1 = p0v2_kernel(
            topk_ids,
            workspace,
            expert_mask_tensor,
            i32_tokens,
            i32_mesh_stride,
            i32_mesh_size,
        )
        l1.launch(grid=(E, 1, 1), block=(P0V2_BLOCK, 1, 1), stream=stream)

        l2 = p23_kernel(
            workspace,
            topk_weights_tensor,
            sorted_token_ids,
            sorted_weights_out,
            sorted_expert_ids,
            num_valid_ids_out,
            moe_buf,
            expert_mask_tensor,
            i32_tokens,
            i32_mesh_stride,
            i32_mesh_size,
            i32_moe_buf_elems,
        )
        l2.launch(grid=(n_grid_p23, 1, 1), block=(K4_BLOCK, 1, 1), stream=stream)

    @flyc.jit
    def launch_4k_fused(
        topk_ids: fx.Tensor,
        workspace: fx.Tensor,
        topk_weights_tensor: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_weights_out: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids_out: fx.Tensor,
        moe_buf: fx.Tensor,
        expert_mask_tensor: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_mesh_stride: fx.Int32,
        i32_mesh_size: fx.Int32,
        i32_moe_buf_elems: fx.Int32,
        i32_ws_total: fx.Int32,
        i32_p0_niters: fx.Int32,
        n_grid_k1: fx.Int32,
        n_grid_k2: fx.Int32,
        n_grid_p23: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        l1 = clear_workspace_kernel(workspace, i32_ws_total)
        l1.launch(grid=(n_grid_k1, 1, 1), block=(K1_BLOCK, 1, 1), stream=stream)

        l2 = p0_scatter_kernel(
            topk_ids, workspace, i32_tokens, i32_mesh_stride, i32_p0_niters
        )
        l2.launch(grid=(n_grid_k2, 1, 1), block=(K2_BLOCK, 1, 1), stream=stream)

        l3 = p1_count_kernel(
            workspace, expert_mask_tensor, i32_mesh_stride, i32_mesh_size
        )
        l3.launch(grid=(E, 1, 1), block=(K3_BLOCK, 1, 1), stream=stream)

        l4 = p23_kernel(
            workspace,
            topk_weights_tensor,
            sorted_token_ids,
            sorted_weights_out,
            sorted_expert_ids,
            num_valid_ids_out,
            moe_buf,
            expert_mask_tensor,
            i32_tokens,
            i32_mesh_stride,
            i32_mesh_size,
            i32_moe_buf_elems,
        )
        l4.launch(grid=(n_grid_p23, 1, 1), block=(K4_BLOCK, 1, 1), stream=stream)

    return (
        launch_clear_ws,
        launch_p0,
        launch_p1,
        launch_p23,
        launch_p0v2,
        launch_p0v2_p23,
        launch_4k_fused,
    )


# Host-side entry point
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=64)
def _compute_sub_tokens(num_experts, arch=None):
    """Compute the LDS-capacity threshold (sub_tokens) for oneshot vs multiphase decision.

    Returns the max T that fits in LDS for the oneshot (single-kernel) path.
    Same formula as _compile_moe_sorting_oneshot.
    """
    if arch is None:
        arch = get_hip_arch()
    E = num_experts
    smem_cols = E + 1
    if arch in ("gfx942",) or str(arch).startswith("gfx94"):
        lds_capacity_bytes = 65536
    elif str(arch).startswith("gfx95"):
        lds_capacity_bytes = 163840
    else:
        lds_capacity_bytes = 65536
    lds_capacity_ints = lds_capacity_bytes // 4
    target_occupancy = 2
    r = lds_capacity_ints // target_occupancy // smem_cols
    sub_unroll = 8
    cumsum_bufs = 2
    if r < (cumsum_bufs + sub_unroll):
        return 0  # LDS too small — always use multiphase
    r_for_sub = ((r - cumsum_bufs) // sub_unroll) * sub_unroll
    return r_for_sub


def moe_sorting_get_workspace_size(M, num_experts, topk, unit_size=UNIT_SIZE):
    """Return workspace size (in i32 elements) needed for the multiphase path.
    Returns 0 if the oneshot path will be used."""
    sub_tokens = _compute_sub_tokens(num_experts)
    ONESHOT_MAX_T = min(sub_tokens, max(16, BLOCK_SIZE // max(topk, num_experts // 8)))
    if M <= min(sub_tokens, ONESHOT_MAX_T):
        return 0
    mesh_stride = ((M + unit_size - 1) // unit_size) * unit_size
    ws_mesh_bytes = num_experts * mesh_stride
    ws_mesh_i32 = (ws_mesh_bytes + 3) // 4
    return ws_mesh_i32 + (num_experts + 1)


def compile_moe_sorting(
    *,
    num_experts,
    topk,
    max_tokens=128,
    unit_size=UNIT_SIZE,
    has_mask=False,
    k4_block=256,
):
    """Compile MoE sorting kernels for all paths (oneshot + multiphase).

    Returns (launch_oneshot, launch_p0v2_p23, launch_4k_fused) covering all T ranges.
    Oneshot compilation depends on max_tokens (LDS sizing); multiphase is independent.
    """
    launch_oneshot = _compile_moe_sorting_oneshot(
        num_experts=num_experts,
        topk=topk,
        max_tokens=max_tokens,
        unit_size=unit_size,
        has_mask=has_mask,
    )
    _, _, _, _, _, launch_p0v2_p23, launch_4k_fused = _compile_moe_sorting_multiphase(
        num_experts=num_experts,
        topk=topk,
        unit_size=unit_size,
        has_mask=has_mask,
        k4_block=k4_block,
    )
    return launch_oneshot, launch_p0v2_p23, launch_4k_fused


def moe_sorting_flydsl(
    topk_ids,
    topk_weights,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    moe_buf,
    num_experts,
    unit_size=UNIT_SIZE,
    expert_mask=None,
    num_local_tokens=None,
    workspace=None,
):
    """MoE sorting using FlyDSL kernel (oneshot + multiphase paths).

    API matches aiter.moe_sorting_fwd for drop-in replacement:
        moe_sorting_flydsl(topk_ids, topk_weights,
                           sorted_ids, sorted_weights, sorted_expert_ids,
                           num_valid_ids, moe_buf,
                           num_experts, unit_size, expert_mask,
                           num_local_tokens, workspace)

    All output tensors (sorted_ids, sorted_weights, sorted_expert_ids,
    num_valid_ids, moe_buf) must be pre-allocated by the caller.

    Returns
    -------
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf
    """
    topk = topk_ids.shape[1]
    if num_local_tokens is not None:
        M = (
            num_local_tokens.item()
            if isinstance(num_local_tokens, torch.Tensor)
            else int(num_local_tokens)
        )
    else:
        M = topk_ids.shape[0]

    sub_tokens = _compute_sub_tokens(num_experts)

    device = topk_ids.device
    # An empty placeholder moe_buf (reduce-mode stage2: caller owns the
    # [M, topk, model_dim] intermediate) carries no zeroing work. Hand the
    # kernel a real 2-D 0-element int32 tensor: reinterpreting the (0,0) bf16
    # buffer would fail the stride-divisibility check, and a 1-D empty tensor
    # breaks the kernel arg's 2-D shape codec (pack_into expects 2 dims).
    if moe_buf.numel() == 0:
        moe_buf_i32 = torch.empty((0, 0), dtype=torch.int32, device=device)
        moe_buf_elems = 0
    else:
        moe_buf_i32 = moe_buf.view(torch.int32)
        moe_buf_elems = moe_buf_i32.numel()

    # EP: prepare mask tensor and flag.
    has_mask = expert_mask is not None
    if not has_mask:
        mask_tensor = _dummy_mask_cache.get(device)
        if mask_tensor is None:
            mask_tensor = torch.ones(1, dtype=torch.int32, device=device)
            _dummy_mask_cache[device] = mask_tensor
    else:
        mask_tensor = expert_mask

    ONESHOT_MAX_T = min(sub_tokens, max(16, BLOCK_SIZE // max(topk, num_experts // 8)))

    target_occupancy = 2
    num_cu = torch.cuda.get_device_properties(device).multi_processor_count

    if M <= min(sub_tokens, ONESHOT_MAX_T):
        max_tokens = max(M, 8)
        max_tokens = ((max_tokens + 7) // 8) * 8

        n_zero_blocks = min(
            (moe_buf_elems + BLOCK_SIZE - 1) // BLOCK_SIZE, num_cu * target_occupancy
        )
        n_grid_blocks = 1 + n_zero_blocks

        launch_oneshot, _, _ = compile_moe_sorting(
            num_experts=num_experts,
            topk=topk,
            max_tokens=max_tokens,
            unit_size=unit_size,
            has_mask=has_mask,
        )
        oneshot_args = (
            topk_ids,
            topk_weights,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf_i32,
            mask_tensor,
            M,
            moe_buf_elems,
            n_grid_blocks,
        )
        _run_compiled(
            launch_oneshot,
            *oneshot_args,
            fx.Stream(torch.cuda.current_stream(device)),
        )
    else:
        mesh_stride = ((M + unit_size - 1) // unit_size) * unit_size
        ws_mesh_bytes = num_experts * mesh_stride
        ws_mesh_i32 = (ws_mesh_bytes + 3) // 4
        ws_total = ws_mesh_i32 + (num_experts + 1)
        if workspace is None:
            workspace = torch.empty(ws_total, dtype=torch.int32, device=device)
        elif workspace.numel() < ws_total:
            raise ValueError(
                f"workspace too small: need {ws_total} i32 elements, got {workspace.numel()}"
            )

        k4_block = _p23_block_size(num_experts, M)
        _, launch_p0v2_p23, launch_4k_fused = compile_moe_sorting(
            num_experts=num_experts,
            topk=topk,
            unit_size=unit_size,
            has_mask=has_mask,
            k4_block=k4_block,
        )
        stream = torch.cuda.current_stream(device)
        n_zero_blocks = min(
            (moe_buf_elems + BLOCK_SIZE - 1) // BLOCK_SIZE, num_cu * target_occupancy
        )
        k4_grid = num_experts + n_zero_blocks
        if M <= 2048:
            p0v2_args = (
                topk_ids,
                workspace,
                topk_weights,
                sorted_ids,
                sorted_weights,
                sorted_expert_ids,
                num_valid_ids,
                moe_buf_i32,
                mask_tensor,
                M,
                mesh_stride,
                ws_mesh_i32,
                moe_buf_elems,
                k4_grid,
            )
            _run_compiled(launch_p0v2_p23, *p0v2_args, fx.Stream(stream))
        else:
            k1_grid = (ws_total + 1023) // 1024
            k2_grid = num_cu * target_occupancy
            k2_total = M * topk
            k2_stride = k2_grid * 256
            k2_niters = (k2_total + k2_stride - 1) // k2_stride
            k4_args = (
                topk_ids,
                workspace,
                topk_weights,
                sorted_ids,
                sorted_weights,
                sorted_expert_ids,
                num_valid_ids,
                moe_buf_i32,
                mask_tensor,
                M,
                mesh_stride,
                ws_mesh_i32,
                moe_buf_elems,
                ws_total,
                k2_niters,
                k1_grid,
                k2_grid,
                k4_grid,
            )
            _run_compiled(launch_4k_fused, *k4_args, fx.Stream(stream))

    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf
