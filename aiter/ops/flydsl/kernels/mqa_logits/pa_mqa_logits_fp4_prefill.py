# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import triton
import triton.language as tl
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, rocdl
from flydsl.expr.primitive import range_constexpr
from flydsl.expr.typing import Float4E2M1FN, Int32, T
from flydsl.expr.utils.arith import _to_raw as as_mlir_value

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

DEFAULT_HEADS = 64
DEFAULT_HEAD_DIM = 128
DEFAULT_NUM_WARPS = 4
DEFAULT_BLOCK_K = 256
DEFAULT_KV_BLOCK_SIZE = 64
ROWS_PER_CTA = 4
NON_WRITER_ELEMENT_OFFSET = 1 << 28
WARP_SIZE = 64
MFMA_M = 32
MFMA_N = 32
MFMA_K = 64
BLOCK_THREADS = ROWS_PER_CTA * WARP_SIZE
M_TILES = DEFAULT_HEADS // MFMA_M
N_TILES = DEFAULT_KV_BLOCK_SIZE // MFMA_N
K_STEPS = DEFAULT_HEAD_DIM // MFMA_K
PAGES_PER_CHUNK = DEFAULT_BLOCK_K // DEFAULT_KV_BLOCK_SIZE
KV_CHUNK_BYTES = 16
KV_PAGE_BYTES = 4 * DEFAULT_KV_BLOCK_SIZE * KV_CHUNK_BYTES
KV_SCALE_PAGE_BYTES = 4 * DEFAULT_KV_BLOCK_SIZE
LDS_WAVE_STEP_BYTES = WARP_SIZE * KV_CHUNK_BYTES
KV_STAGE_BYTES = PAGES_PER_CHUNK * KV_PAGE_BYTES
KV_SCALE_STAGE_BYTES = PAGES_PER_CHUNK * KV_SCALE_PAGE_BYTES
LDS_SCALE_BASE_BYTES = KV_STAGE_BYTES
LDS_TOTAL_BYTES = KV_STAGE_BYTES + KV_SCALE_STAGE_BYTES

# cta_info fields per wave: row, batch, chunk start/count, window start/end.
CTA_INFO_WIDTH = 6


def compute_prefill_schedule(
    row_to_batch,
    local_starts,
    local_ends,
    block_k,
    parallel_unit_num,
    max_seq_len,
    cta_info_out=None,
):
    """Build a four-row-per-CTA schedule for ragged prefill.

    Rows must be grouped by batch. ``parallel_unit_num`` remains the logical
    one-row CTA budget used to choose the chunk split size and is also the
    fixed physical launch bound. Slots past the device-computed CTA count are
    marked inactive, avoiding a GPU-to-CPU synchronization on every schedule.
    """
    device = local_ends.device
    P = parallel_unit_num
    T = local_ends.shape[0]

    if T < 1 or T > _ROW_PLAN_MAX_ROWS:
        raise ValueError(f"FP4 prefill supports 1..{_ROW_PLAN_MAX_ROWS} rows, got {T}.")
    if P < T:
        raise ValueError(f"parallel_unit_num={P} must be at least rows={T}.")
    if not (
        row_to_batch.dtype == local_starts.dtype == local_ends.dtype == torch.int32
    ):
        raise TypeError("row_to_batch, local_starts, and local_ends must be int32.")

    s_max = max(1, (max_seq_len + block_k - 1) // block_k)
    plan = _row_plan(row_to_batch, local_ends, block_k, P, s_max)
    record_count = P * ROWS_PER_CTA

    if cta_info_out is None:
        cta_info = torch.empty(
            record_count, CTA_INFO_WIDTH, dtype=torch.int32, device=device
        )
    else:
        cta_info = cta_info_out
        if cta_info.numel() < record_count * CTA_INFO_WIDTH:
            raise ValueError(
                f"cta_info_out has room for {cta_info.numel() // CTA_INFO_WIDTH} "
                f"records, but {record_count} are required."
            )
    block_p = 256
    _prefill_cta_info_kernel[(triton.cdiv(P, block_p),)](
        plan.incl,
        plan.excl,
        plan.chunks,
        row_to_batch,
        local_starts,
        local_ends,
        plan.safe,
        plan.total_splits,
        cta_info,
        T,
        P,
        BLOCK_P=block_p,
        ROWS_PER_CTA=ROWS_PER_CTA,
        BLOCK_K=DEFAULT_BLOCK_K,
        INFO_WIDTH=CTA_INFO_WIDTH,
    )
    return plan.safe, cta_info, P


class _RowPlan(NamedTuple):
    """Everything the emit kernel needs about the rows, from either producer.

    One carrier, so a field can only be added where both have to answer for it.
    """

    incl: torch.Tensor  # [T] inclusive prefix sum of per-group CTA counts
    excl: torch.Tensor  # [T] exclusive prefix sum
    chunks: torch.Tensor  # [T] max chunks in each four-row group, else 0
    safe: torch.Tensor  # [1] chunk-splits merged into one CTA
    total_splits: torch.Tensor  # [1] physical CTA count


_ROW_PLAN_MAX_ROWS = 16384

_I32_PER_16B = 4


def _row_plan(rb, le, block_k, P, s_max) -> _RowPlan:
    T = le.shape[0]
    stride = (T + _I32_PER_16B - 1) // _I32_PER_16B * _I32_PER_16B
    tail = 3 * stride
    work = torch.empty(tail + 2 * _I32_PER_16B, dtype=torch.int32, device=le.device)
    plan = _RowPlan(
        incl=work[:T],
        excl=work[stride : stride + T],
        chunks=work[2 * stride : 2 * stride + T],
        safe=work[tail : tail + 1],
        total_splits=work[tail + _I32_PER_16B : tail + _I32_PER_16B + 1],
    )
    _prefill_row_plan_kernel[(1,)](
        rb,
        le,
        plan.incl,
        plan.excl,
        plan.chunks,
        plan.safe,
        plan.total_splits,
        T,
        P,
        block_k,
        s_max,
        # One masked launch shape avoids a separate JIT specialization when a
        # serving batch crosses the old 4K-row threshold.
        BLOCK_T=_ROW_PLAN_MAX_ROWS,
        # Keep both binary-search bounds dynamic. Random prompt lengths change
        # these values even when the data layout is identical; specializing
        # either loop caused a fresh multi-second Triton compile.
        SEARCH_STEPS=max(1, (s_max - 1).bit_length() + 1),
        ROW_SEARCH_STEPS=max(1, (T - 1).bit_length() + 1),
        ROWS_PER_CTA=ROWS_PER_CTA,
    )
    return plan


@triton.jit(
    do_not_specialize=[
        "T",
        "P",
        "s_max",
        "SEARCH_STEPS",
        "ROW_SEARCH_STEPS",
    ]
)
def _prefill_row_plan_kernel(
    rb_ptr,
    le_ptr,  # [T] int32 local_ends
    incl_ptr,  # [T] int32 out
    excl_ptr,  # [T] int32 out
    chunks_ptr,  # [T] int32 out
    safe_ptr,  # [1] int32 out
    total_splits_ptr,  # [1] int32 out
    T,
    P,
    block_k,
    s_max,
    BLOCK_T: tl.constexpr,
    SEARCH_STEPS,
    ROW_SEARCH_STEPS,
    ROWS_PER_CTA: tl.constexpr,
):
    """Choose the row split and prefix-sum four-row physical CTAs."""
    t = tl.arange(0, BLOCK_T)
    mask = t < T
    le = tl.load(le_ptr + t, mask=mask, other=0)
    row_chunks = tl.where(mask, tl.maximum((le + block_k - 1) // block_k, 0), 0)
    max_chunks = tl.maximum(tl.max(row_chunks, axis=0), 1)

    lo = 1
    hi = s_max
    for _ in range(SEARCH_STEPS):
        mid = (lo + hi) // 2
        feasible = tl.sum((row_chunks + mid - 1) // mid, axis=0) <= P
        active = lo < hi
        hi = tl.where(active & feasible, mid, hi)
        lo = tl.where(active & (feasible == 0), mid + 1, lo)
    total_smax = tl.sum((row_chunks + s_max - 1) // s_max, axis=0)
    safe = tl.where(total_smax <= P, lo, max_chunks)

    batch = tl.load(rb_ptr + t, mask=mask, other=2147483647)
    row_lo = tl.zeros([BLOCK_T], tl.int32)
    row_hi = tl.where(mask, t, 0)
    for _ in range(ROW_SEARCH_STEPS):
        mid = (row_lo + row_hi) // 2
        mid_batch = tl.load(
            rb_ptr + tl.minimum(mid, T - 1), mask=mask, other=2147483647
        )
        go_right = mid_batch < batch
        active = row_lo < row_hi
        row_lo = tl.where(active & go_right, mid + 1, row_lo)
        row_hi = tl.where(active & (go_right == 0), mid, row_hi)
    leader = mask & (((t - row_lo) % ROWS_PER_CTA) == 0)

    group_chunks = tl.zeros([BLOCK_T], tl.int32)
    for lane in tl.static_range(ROWS_PER_CTA):
        row = t + lane
        row_valid = leader & (row < T)
        same_batch = (
            tl.load(
                rb_ptr + tl.minimum(row, T - 1),
                mask=row_valid,
                other=2147483647,
            )
            == batch
        )
        row_end = tl.load(
            le_ptr + tl.minimum(row, T - 1), mask=row_valid & same_batch, other=0
        )
        chunks = tl.maximum((row_end + block_k - 1) // block_k, 0)
        group_chunks = tl.maximum(group_chunks, chunks)

    ctas = tl.where(leader, (group_chunks + safe - 1) // safe, 0)
    incl = tl.cumsum(ctas, axis=0)
    tl.store(incl_ptr + t, incl, mask=mask)
    tl.store(excl_ptr + t, incl - ctas, mask=mask)
    tl.store(chunks_ptr + t, tl.where(leader, group_chunks, 0), mask=mask)
    tl.store(safe_ptr, safe)
    tl.store(total_splits_ptr, tl.sum(ctas, axis=0))


@triton.jit(do_not_specialize=["T", "P"])
def _prefill_cta_info_kernel(
    incl_ptr,  # [T] int32 inclusive prefix sum of per-row CTA counts
    excl_ptr,  # [T] int32 exclusive prefix sum
    chunks_ptr,  # [T] int32 chunks_per_row
    rb_ptr,  # [T] int32 row_to_batch
    ls_ptr,  # [T] int32 local_starts
    le_ptr,  # [T] int32 local_ends
    safe_ptr,  # [1] int32
    total_splits_ptr,  # [1] int32 physical CTA count
    cta_info_ptr,  # [P * ROWS_PER_CTA, 6] int32
    T,
    P,
    BLOCK_P: tl.constexpr,
    ROWS_PER_CTA: tl.constexpr,
    BLOCK_K: tl.constexpr,
    INFO_WIDTH: tl.constexpr,
):
    """Map each physical CTA to four same-batch query rows."""
    pid = tl.program_id(0)
    safe = tl.load(safe_ptr)
    total_splits = tl.load(total_splits_ptr)
    slot = pid * BLOCK_P + tl.arange(0, BLOCK_P)
    smask = slot < P
    slot_active = smask & (slot < total_splits)

    lo = tl.zeros([BLOCK_P], tl.int32)
    hi = tl.full([BLOCK_P], T, tl.int32)
    for _ in tl.static_range(32):
        mid = (lo + hi) // 2
        incl_mid = tl.load(
            incl_ptr + tl.minimum(mid, T - 1), mask=(mid < T), other=2147483647
        )
        go_right = incl_mid <= slot
        lo = tl.where(go_right, mid + 1, lo)
        hi = tl.where(go_right, hi, mid)
    leader = tl.minimum(lo, T - 1)
    batch = tl.load(rb_ptr + leader, mask=smask, other=0)
    leader_start = tl.load(ls_ptr + leader, mask=smask, other=0)
    leader_end = tl.load(le_ptr + leader, mask=smask, other=0)
    group_chunks = tl.load(chunks_ptr + leader, mask=smask, other=0)
    split_within = slot - tl.load(excl_ptr + leader, mask=smask, other=0)
    start = split_within * safe
    count = tl.maximum(tl.minimum(safe, group_chunks - start), 1)

    lane = tl.arange(0, ROWS_PER_CTA)
    row = leader[:, None] + lane[None, :]
    row_in_range = slot_active[:, None] & (row < T)
    row_batch = tl.load(rb_ptr + tl.minimum(row, T - 1), mask=row_in_range, other=-1)
    row_start = tl.load(ls_ptr + tl.minimum(row, T - 1), mask=row_in_range, other=0)
    row_end = tl.load(le_ptr + tl.minimum(row, T - 1), mask=row_in_range, other=0)
    row_chunks = tl.maximum((row_end + BLOCK_K - 1) // BLOCK_K, 0)
    active = (
        row_in_range & (row_batch == batch[:, None]) & (row_chunks > start[:, None])
    )
    encoded_row = tl.where(active, row, -leader[:, None] - 1)
    row_start = tl.where(active, row_start, leader_start[:, None])
    row_end = tl.where(active, row_end, leader_end[:, None])

    record = slot[:, None] * ROWS_PER_CTA + lane[None, :]
    base = record * INFO_WIDTH
    # Padding CTAs execute one safe dummy chunk from batch/row zero. Their rows
    # are encoded as inactive, so buffer stores are suppressed by
    # NON_WRITER_ELEMENT_OFFSET. Keeping a positive uniform count also
    # preserves the barrier contract of the four-wave compute kernel.
    stored_row = tl.where(slot_active[:, None], encoded_row, -1)
    stored_batch = tl.where(slot_active, batch, 0)
    stored_start = tl.where(slot_active, start, 0)
    stored_count = tl.where(slot_active, count, 1)
    stored_row_start = tl.where(slot_active[:, None], row_start, 0)
    stored_row_end = tl.where(slot_active[:, None], row_end, 0)
    tl.store(cta_info_ptr + base + 0, stored_row, mask=smask[:, None])
    tl.store(
        cta_info_ptr + base + 1, stored_batch[:, None], mask=smask[:, None]
    )
    tl.store(cta_info_ptr + base + 2, stored_start[:, None], mask=smask[:, None])
    tl.store(cta_info_ptr + base + 3, stored_count[:, None], mask=smask[:, None])
    tl.store(cta_info_ptr + base + 4, stored_row_start, mask=smask[:, None])
    tl.store(cta_info_ptr + base + 5, stored_row_end, mask=smask[:, None])


# Kernel
# ============================================================================


@lru_cache(maxsize=32)
def compile_pa_mqa_logits_fp4_prefill(
    *,
    kv_page_stride: int,
    kv_scale_page_stride: int,
    block_table_stride: int,
    weight_scale: float = 1.0,
):
    """Build the FP4 MQA prefill kernel.

    Each of the four waves owns one query row and loads one cache page into a
    shared-memory stage. The CTA records share one batch and chunk range;
    padded records use a negative encoded row id.
    """

    @flyc.kernel
    def pa_mqa_logits_fp4_prefill_kernel(
        out_logits_ptr: fx.Tensor,
        q_ptr: fx.Tensor,
        q_scale_ptr: fx.Tensor,
        kv_cache_ptr: fx.Tensor,
        kv_scale_ptr: fx.Tensor,
        kv_indices_ptr: fx.Tensor,
        weights_ptr: fx.Tensor,
        cta_info_ptr: fx.Tensor,
        stride_out_row: Int32,
    ):
        tid = gpu.thread_idx.x
        pid = gpu.block_idx.x
        warp_id = tid >> 6
        warp_id_uniform = fx.Int32(
            rocdl.readfirstlane(T.i32, fx.Int32(warp_id).ir_value())
        )
        lane_id = tid % WARP_SIZE
        lane_mod_32 = lane_id & 31
        lane_div_32 = lane_id >> 5 & 1
        lane_mod_16 = lane_id & 15
        cta_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(fx.ptrtoint(fx.get_iter(cta_info_ptr)))
        )
        cta_record = fx.Int32(pid) * fx.Int32(ROWS_PER_CTA) + warp_id_uniform
        cta_rsrc_v4i32 = llvm.bitcast(
            ir.VectorType.get([4], T.i32),
            llvm.ptrtoint(ir.IntegerType.get_signless(128), as_mlir_value(cta_rsrc)),
        )
        cta_base = cta_record * fx.Int32(CTA_INFO_WIDTH)
        cta_info_lo = fx.Vector(
            buffer_ops.buffer_load(
                cta_rsrc, cta_base, vec_width=4, dtype=fx.Int32, is_scalar=True
            )
        )
        cta_info_hi = fx.Vector(
            llvm.inline_asm(
                ir.VectorType.get([2], T.i32),
                [
                    cta_rsrc_v4i32,
                    as_mlir_value((cta_base + fx.Int32(4)) * fx.Int32(4)),
                ],
                "s_buffer_load_dwordx2 $0, $1, $2 offset:0",
                "=s,s,s",
                has_side_effects=False,
            )
        )
        encoded_row_id = cta_info_lo[0]
        batch_id = cta_info_lo[1]
        chunk_start = cta_info_lo[2]
        chunk_count = cta_info_lo[3]
        local_start = cta_info_hi[0]
        local_end = cta_info_hi[1]
        row_active = encoded_row_id >= fx.Int32(0)
        row_id = row_active.select(encoded_row_id, -encoded_row_id - fx.Int32(1))
        inactive_row_off = row_active.select(
            fx.Int32(0), fx.Int32(NON_WRITER_ELEMENT_OFFSET)
        )
        bt_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(fx.ptrtoint(fx.get_iter(kv_indices_ptr)))
        )
        zero_v16 = fx.Vector.filled(16, 0.0, fx.Float32)
        row_elems = fx.Int64(row_id) * fx.Int64(stride_out_row)
        out_win = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fx.recast_iter(
                    fx.PointerType.get(T.f32, out_logits_ptr.memspace, 4),
                    fx.add_offset(fx.get_iter(out_logits_ptr), row_elems),
                ),
                fx.make_layout((local_end, 1), (1, 1)),
            ),
            max_size=False,
            num_records_bytes=local_end * fx.Int32(4),
        )
        out_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 1)
        out_reg_ty = fx.MemRefType.get(
            T.f32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register
        )
        out_reg_lay = fx.make_layout(1, 1)
        lds_ptr = fx.SharedAllocator().allocate(LDS_TOTAL_BYTES, alignment=16)._ptr
        lds_base_i32 = fx.Int32(fx.ptrtoint(lds_ptr))
        lds_i32 = fx.recast_iter(fx.Int32, lds_ptr)
        dma_ptr_ty = fx.PointerType.get(T.i8, fx.AddressSpace.Shared, 16)

        def _issue_scale_dma(scale_rsrc, scale_dst_ptr):
            rocdl.raw_ptr_buffer_load_lds(
                scale_rsrc,
                scale_dst_ptr,
                fx.Int32(4),
                fx.Int32(lane_id) * fx.Int32(4),
                fx.Int32(0),
                fx.Int32(0),
                fx.Int32(0),
            )

        mfma_atoms = [
            [
                fx.make_mma_atom(
                    fx.rocdl.cdna4.MFMA_Scale(
                        MFMA_M,
                        MFMA_N,
                        MFMA_K,
                        Float4E2M1FN,
                        Float4E2M1FN,
                        opsel_a=mi * 2,
                        opsel_b=nt * 2,
                    )
                )
                for mi in range(M_TILES)
            ]
            for nt in range(N_TILES)
        ]
        q_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(fx.ptrtoint(fx.get_iter(q_ptr)))
        )
        q_a_ops = []
        for mi in range_constexpr(M_TILES):
            q_a_ops_mi = []
            for ks in range_constexpr(K_STEPS):
                q_dword_offset = (
                    row_id * fx.Int32(DEFAULT_HEADS * DEFAULT_HEAD_DIM // 8)
                    + lane_mod_32 * fx.Int32(DEFAULT_HEAD_DIM // 8)
                    + lane_div_32 * fx.Int32(4)
                )
                q_soffset_bytes = mi * MFMA_M * DEFAULT_HEAD_DIM // 2 + ks * 32
                q_4xi32 = fx.Vector(
                    buffer_ops.buffer_load(
                        q_rsrc,
                        q_dword_offset,
                        vec_width=4,
                        dtype=fx.Int32,
                        soffset_bytes=q_soffset_bytes,
                    )
                )
                a_frag = fx.make_rmem_tensor(4, fx.Int32)
                a_frag.store(q_4xi32)
                q_a_ops_mi.append(a_frag)
            q_a_ops.append(q_a_ops_mi)
        qs_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(fx.ptrtoint(fx.get_iter(q_scale_ptr)))
        )
        q_scale_words = []
        for ks in range_constexpr(K_STEPS):
            kc = fx.Int32(ks * 2) + lane_div_32
            qs_dword_offset = row_id * fx.Int32(64) + kc * fx.Int32(16) + lane_mod_16
            q_scale_words.append(
                fx.Int32(
                    buffer_ops.buffer_load(
                        qs_rsrc, qs_dword_offset, vec_width=1, dtype=fx.Int32
                    )
                )
            )
        w_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(fx.ptrtoint(fx.get_iter(weights_ptr)))
        )
        w_rsrc_v4i32 = llvm.bitcast(
            ir.VectorType.get([4], T.i32),
            llvm.ptrtoint(ir.IntegerType.get_signless(128), as_mlir_value(w_rsrc)),
        )

        def _load_weight_row_x16(mi):
            weight_row_base = row_id * fx.Int32(DEFAULT_HEADS // 2) + fx.Int32(mi * 16)
            return fx.Vector(
                llvm.call_intrinsic(
                    ir.VectorType.get([16], T.i32),
                    "llvm.amdgcn.s.buffer.load.v16i32",
                    [
                        w_rsrc_v4i32,
                        as_mlir_value(weight_row_base * fx.Int32(4)),
                        as_mlir_value(fx.Int32(0)),
                    ],
                    [],
                    [],
                )
            )

        def _select_weight_rows_half_x16(weight_rows):
            result_ty = ir.Type.parse("!llvm.struct<(" + ", ".join(["i32"] * 16) + ")>")
            low_word_indices = (0, 1, 4, 5, 8, 9, 12, 13)
            high_word_indices = (2, 3, 6, 7, 10, 11, 14, 15)
            asm_lines = ["s_mov_b64 vcc, exec", "s_mov_b32 exec_hi, 0"]
            for mi in range_constexpr(M_TILES):
                input_base = 16 + mi * 16
                output_base = mi * 8
                for output_idx, low_idx in enumerate(low_word_indices):
                    asm_lines.append(
                        f"v_mov_b32 ${{{output_base + output_idx}}}, ${{{input_base + low_idx}}}"
                    )
            asm_lines.append("s_xor_b64 exec, vcc, exec")
            for mi in range_constexpr(M_TILES):
                input_base = 16 + mi * 16
                output_base = mi * 8
                for output_idx, high_idx in enumerate(high_word_indices):
                    asm_lines.append(
                        f"v_mov_b32 ${{{output_base + output_idx}}}, ${{{input_base + high_idx}}}"
                    )
            asm_lines.append("s_mov_b64 exec, vcc")
            result = llvm.inline_asm(
                result_ty,
                [
                    as_mlir_value(weight_rows[mi][word_idx])
                    for mi in range_constexpr(M_TILES)
                    for word_idx in range_constexpr(16)
                ],
                "\n\t".join(asm_lines),
                ",".join(["=&v"] * 16 + ["s"] * 32 + ["~{vcc}"]),
                has_side_effects=True,
            )
            return [
                [
                    fx.Int32(llvm.extractvalue(T.i32, result, [mi * 8 + word_idx]))
                    for word_idx in range_constexpr(8)
                ]
                for mi in range_constexpr(M_TILES)
            ]

        selected_weight_rows = _select_weight_rows_half_x16(
            [_load_weight_row_x16(mi) for mi in range_constexpr(M_TILES)]
        )
        w_per_lane = []
        for mi in range_constexpr(M_TILES):
            selected_weight_words = selected_weight_rows[mi]
            w_groups = []
            for group in range_constexpr(4):
                word_base = group * 2
                selected_words = fx.Vector.from_elements(
                    [
                        selected_weight_words[word_base],
                        selected_weight_words[word_base + 1],
                    ],
                    dtype=fx.Int32,
                )
                w_bf16 = selected_words.bitcast(fx.BFloat16)
                w_groups.append(w_bf16)
            w_per_lane.append(w_groups)

        def _load_phys_pages(c_i32_arg):
            page0_in_seq = (chunk_start + c_i32_arg) * fx.Int32(PAGES_PER_CHUNK)
            bt_index = batch_id * fx.Int32(block_table_stride) + page0_in_seq
            phys = fx.Int32(
                buffer_ops.buffer_load(
                    bt_rsrc,
                    bt_index + warp_id_uniform,
                    vec_width=1,
                    dtype=fx.Int32,
                    is_scalar=True,
                )
            )
            return [phys for _ in range_constexpr(PAGES_PER_CHUNK)]

        def _issue_mi_mfmas(kv_list, kv_scale_words, nt, mi):
            acc = zero_v16
            head_half = lane_mod_32 >> 4
            scale_shift = head_half * fx.Int32(8)
            for ks in range_constexpr(K_STEPS):
                b_frag = fx.make_rmem_tensor(4, fx.Int32)
                b_frag.store(fx.Vector(kv_list[nt * K_STEPS + ks]))
                scale_a = fx.Int32(fx.Uint32(q_scale_words[ks]) >> scale_shift)
                c_frag = fx.make_rmem_tensor(16, fx.Float32)
                c_frag.store(fx.Vector(acc))
                fx.gemm(
                    mfma_atoms[nt][mi],
                    c_frag,
                    q_a_ops[mi][ks],
                    b_frag,
                    c_frag,
                    scale_a=scale_a,
                    scale_b=kv_scale_words[ks],
                )
                acc = c_frag.load()
            return acc

        def _finish_pair(sum0, sum1):
            pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
            pair = rocdl.permlane32_swap(
                pair_ty,
                as_mlir_value(sum0.bitcast(fx.Int32)),
                as_mlir_value(sum1.bitcast(fx.Int32)),
                False,
                True,
            )
            lhs = fx.Int32(llvm.extractvalue(T.i32, pair, [0]))
            rhs = fx.Int32(llvm.extractvalue(T.i32, pair, [1]))
            return lhs.bitcast(fx.Float32) + rhs.bitcast(fx.Float32)

        def _store_thread_sum(thread_sum, c_i32_arg, page_idx):
            token_base = (chunk_start + c_i32_arg) * fx.Int32(
                DEFAULT_BLOCK_K
            ) + fx.Int32(page_idx * DEFAULT_KV_BLOCK_SIZE)
            token = token_base + fx.Int32(lane_id)
            lower_bound_off = fx.Int32(0)
            if local_start != fx.Int32(0):
                lower_bound_off = (token < local_start).select(
                    fx.Int32(NON_WRITER_ELEMENT_OFFSET), fx.Int32(0)
                )
            out_reg = fx.memref_alloca(out_reg_ty, out_reg_lay)
            fx.memref_store_vec(
                fx.Vector.from_elements([thread_sum], dtype=fx.Float32), out_reg
            )
            fx.copy(
                out_atom,
                out_reg,
                fx.slice(
                    out_win,
                    (
                        token + lower_bound_off + inactive_row_off,
                        None,
                    ),
                ),
            )

        def _post_process_pair(sum0, sum1, c_i32_arg, page_idx):
            _store_thread_sum(
                _finish_pair(sum0, sum1) * fx.Float32(weight_scale),
                c_i32_arg,
                page_idx,
            )

        scale_shift = (lane_mod_32 >> fx.Int32(4)) * fx.Int32(8)

        def _prepare_scales(kv_scale_list):
            packed_scales = fx.Vector.from_elements(
                kv_scale_list, dtype=fx.Int32
            ).bitcast(fx.Uint64)[0]
            shifted_scales = fx.Vector.from_elements(
                [fx.Uint64(packed_scales) >> fx.Uint64(scale_shift)], dtype=fx.Uint64
            ).bitcast(fx.Int32)
            return [shifted_scales[ks] for ks in range_constexpr(K_STEPS)]

        def _issue_page_half(kv_list, kv_scale_words, mi):
            acc1 = _issue_mi_mfmas(kv_list, kv_scale_words, 1, mi)
            acc0 = _issue_mi_mfmas(kv_list, kv_scale_words, 0, mi)
            return (acc0, acc1)

        def _reduce_first_half(acc00, acc01):
            relu00 = fx.maxnumf(
                fx.Vector(acc00), zero_v16, fastmath=fx.arith.FastMathFlags.nnan
            )
            relu01 = fx.maxnumf(
                fx.Vector(acc01), zero_v16, fastmath=fx.arith.FastMathFlags.nnan
            )
            seed_weight = fx.Float32(w_per_lane[0][0][0])
            sum0 = relu00[0] * seed_weight
            sum1 = relu01[0] * seed_weight
            for elem in range_constexpr(1, 16):
                weight = fx.Float32(w_per_lane[0][elem // 4][elem % 4])
                sum0 = fx.fma(relu00[elem], weight, sum0)
                sum1 = fx.fma(relu01[elem], weight, sum1)
            return (sum0, sum1)

        def _reduce_second_half(acc10, acc11, sum0, sum1):
            relu10 = fx.maxnumf(
                fx.Vector(acc10), zero_v16, fastmath=fx.arith.FastMathFlags.nnan
            )
            relu11 = fx.maxnumf(
                fx.Vector(acc11), zero_v16, fastmath=fx.arith.FastMathFlags.nnan
            )
            for elem in range_constexpr(16):
                weight = fx.Float32(w_per_lane[1][elem // 4][elem % 4])
                sum0 = fx.fma(relu10[elem], weight, sum0)
                sum1 = fx.fma(relu11[elem], weight, sum1)
            return (sum0, sum1)

        def _compute_and_store_page(kv_list, kv_scale_list, c_i32_arg, page_idx):
            kv_scale_words = _prepare_scales(kv_scale_list)
            acc00, acc01 = _issue_page_half(kv_list, kv_scale_words, 0)
            rocdl.sched_barrier(0)
            sum0, sum1 = _reduce_first_half(acc00, acc01)
            rocdl.sched_barrier(0)
            acc10, acc11 = _issue_page_half(kv_list, kv_scale_words, 1)
            rocdl.sched_barrier(0)
            sum0, sum1 = _reduce_second_half(acc10, acc11, sum0, sum1)
            _post_process_pair(sum0, sum1, c_i32_arg, page_idx)

        lds_copy = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        lds_reg_lay = fx.make_layout(4, 1)
        lds_scale_copy = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
        lds_scale_reg_lay = fx.make_layout(1, 1)

        def _page_rsrc(phys):
            page_addr = fx.Int64(fx.ptrtoint(fx.get_iter(kv_cache_ptr))) + fx.Int64(
                phys
            ) * fx.Int64(kv_page_stride)
            return buffer_ops.create_buffer_resource_from_addr(
                page_addr, num_records_bytes=KV_PAGE_BYTES
            )

        def _scale_page_rsrc(phys):
            page_addr = fx.Int64(fx.ptrtoint(fx.get_iter(kv_scale_ptr))) + fx.Int64(
                phys
            ) * fx.Int64(kv_scale_page_stride)
            return buffer_ops.create_buffer_resource_from_addr(
                page_addr, num_records_bytes=KV_SCALE_PAGE_BYTES
            )

        def _issue_page_group_dma(phys_pages, parity):
            """Have each wave load one page and its scales into LDS."""
            page_idx = warp_id_uniform
            phys = phys_pages[0]
            slot_base = parity * fx.Int32(KV_STAGE_BYTES) + page_idx * fx.Int32(
                KV_PAGE_BYTES
            )
            page_rsrc = _page_rsrc(phys)
            for plane_idx in range_constexpr(4):
                dst_addr = (
                    lds_base_i32 + slot_base + fx.Int32(plane_idx * LDS_WAVE_STEP_BYTES)
                )
                dst_ptr = fx.to_llvm_ptr(fx.inttoptr(dma_ptr_ty, fx.Int64(dst_addr)))
                src_byte = fx.Int32(plane_idx * LDS_WAVE_STEP_BYTES) + fx.Int32(
                    lane_id
                ) * fx.Int32(KV_CHUNK_BYTES)
                rocdl.raw_ptr_buffer_load_lds(
                    page_rsrc,
                    dst_ptr,
                    fx.Int32(KV_CHUNK_BYTES),
                    src_byte,
                    fx.Int32(0),
                    fx.Int32(0),
                    fx.Int32(0),
                )
            scale_slot_base = (
                lds_base_i32
                + fx.Int32(LDS_SCALE_BASE_BYTES)
                + parity * fx.Int32(KV_SCALE_STAGE_BYTES)
                + page_idx * fx.Int32(KV_SCALE_PAGE_BYTES)
            )
            scale_dst_ptr = fx.to_llvm_ptr(
                fx.inttoptr(dma_ptr_ty, fx.Int64(scale_slot_base))
            )
            _issue_scale_dma(_scale_page_rsrc(phys), scale_dst_ptr)

        def _load_lds_page_range(parity, page_idx, load_begin, load_end):
            kv_page = []
            slot_base_dw = (
                parity * fx.Int32(KV_STAGE_BYTES) + fx.Int32(page_idx * KV_PAGE_BYTES)
            ) // fx.Int32(4)
            for load_idx in range_constexpr(load_begin, load_end):
                nt = load_idx // K_STEPS
                ks = load_idx % K_STEPS
                token = fx.Int32(nt * MFMA_N) + lane_mod_32
                kc = fx.Int32(ks * 2) + lane_div_32
                src = fx.make_view(
                    fx.add_offset(
                        lds_i32,
                        slot_base_dw
                        + kc * fx.Int32(LDS_WAVE_STEP_BYTES // 4)
                        + token * fx.Int32(KV_CHUNK_BYTES // 4),
                    ),
                    lds_reg_lay,
                )
                reg = fx.make_rmem_tensor(4, fx.Int32)
                fx.copy(lds_copy, src, reg)
                kv_page.append(fx.memref_load_vec(reg).ir_value())
            return kv_page

        def _load_lds_page(parity, page_idx):
            return _load_lds_page_range(parity, page_idx, 0, N_TILES * K_STEPS)

        def _load_lds_scales(parity, page_idx):
            scale_words = []
            scale_base_dw = (
                fx.Int32(LDS_SCALE_BASE_BYTES)
                + parity * fx.Int32(KV_SCALE_STAGE_BYTES)
                + fx.Int32(page_idx * KV_SCALE_PAGE_BYTES)
            ) // fx.Int32(4)
            for ks in range_constexpr(K_STEPS):
                kc = fx.Int32(ks * 2) + lane_div_32
                src = fx.make_view(
                    fx.add_offset(
                        lds_i32,
                        scale_base_dw
                        + kc * fx.Int32(DEFAULT_KV_BLOCK_SIZE // 4)
                        + lane_mod_16,
                    ),
                    lds_scale_reg_lay,
                )
                reg = fx.make_rmem_tensor(1, fx.Int32)
                fx.copy(lds_scale_copy, src, reg)
                scale_words.append(fx.memref_load_vec(reg)[0])
            return scale_words

        def _consume_and_refill(c_i32_arg, cur_stage, next_stage, phys_next):
            for page_idx in range_constexpr(PAGES_PER_CHUNK - 1):
                kv_page = _load_lds_page(cur_stage, page_idx)
                kvs_page = _load_lds_scales(cur_stage, page_idx)
                _compute_and_store_page(kv_page, kvs_page, c_i32_arg, page_idx)
            last_page_idx = PAGES_PER_CHUNK - 1
            kv_page_last = _load_lds_page(cur_stage, last_page_idx)
            kvs_page_last = _load_lds_scales(cur_stage, last_page_idx)
            rocdl.sched_barrier(0)
            # The next DMA reuses the single LDS stage. All waves must finish
            # their current reads before any wave starts overwriting it.
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()
            _issue_page_group_dma(phys_next, next_stage)
            kvs_page_last_words = _prepare_scales(kvs_page_last)
            acc00, acc01 = _issue_page_half(kv_page_last, kvs_page_last_words, 0)
            rocdl.sched_barrier(0)
            sum0, sum1 = _reduce_first_half(acc00, acc01)
            rocdl.sched_barrier(0)
            acc10, acc11 = _issue_page_half(kv_page_last, kvs_page_last_words, 1)
            rocdl.sched_barrier(0)
            sum0, sum1 = _reduce_second_half(acc10, acc11, sum0, sum1)
            _post_process_pair(sum0, sum1, c_i32_arg, last_page_idx)

        last_c_i32 = chunk_count - fx.Int32(1)
        phys_first = _load_phys_pages(fx.Int32(0))
        _issue_page_group_dma(phys_first, fx.Int32(0))
        phys_next_pre = _load_phys_pages(fx.Int32(1))
        rocdl.sched_barrier(0)
        rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()
        chunk_count_minus_1 = fx.Int64(chunk_count - fx.Int32(1))
        for c_idx, state in range(0, chunk_count_minus_1, 1, init=list(phys_next_pre)):
            phys_next = [
                state[page_idx] for page_idx in range_constexpr(PAGES_PER_CHUNK)
            ]
            c_i32 = fx.Int32(c_idx)
            cur_parity = fx.Int32(0)
            next_parity = fx.Int32(0)
            _consume_and_refill(c_i32, cur_parity, next_parity, phys_next)
            phys_next_next = _load_phys_pages(c_i32 + fx.Int32(2))
            rocdl.sched_barrier(0)
            rocdl.s_waitcnt(vmcnt=1)
            gpu.barrier()
            _ = yield list(phys_next_next)
        last_parity = fx.Int32(0)
        for page_idx in range_constexpr(PAGES_PER_CHUNK):
            kv_last = _load_lds_page(last_parity, page_idx)
            kvs_last = _load_lds_scales(last_parity, page_idx)
            _compute_and_store_page(kv_last, kvs_last, last_c_i32, page_idx)

    @flyc.jit
    def launch_pa_mqa_logits_fp4_prefill(
        out,
        q,
        qs,
        kv,
        kvs,
        bt,
        w,
        cta_info,
        stride_out: fx.Int32,
        gx: fx.Int32,
        stream: fx.Stream,
    ):
        pa_mqa_logits_fp4_prefill_kernel(
            out,
            q,
            qs,
            kv,
            kvs,
            bt,
            w,
            cta_info,
            stride_out,
            value_attrs=None,
        ).launch(grid=(fx.Int64(gx),), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch_pa_mqa_logits_fp4_prefill, BLOCK_THREADS


def flydsl_pa_mqa_logits_fp4_prefill(
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    weights: torch.Tensor,
    row_to_batch: torch.Tensor,
    local_starts: torch.Tensor,
    local_ends: torch.Tensor,
    max_seq_len: int,
    *,
    weight_scale: float = 1.0,
    block_k: int = 256,
    kv_block_size: int = 64,
    num_warps: int = DEFAULT_NUM_WARPS,
    parallel_unit_num: int = 512,
    out: torch.Tensor | None = None,
    cta_info: torch.Tensor | None = None,
    n_ctas: int | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Ragged-prefill FP4 paged MQA logits (gfx950)."""
    total_tokens, heads, head_dim_packed = q_fp4.shape
    head_dim = head_dim_packed * 2
    if (
        block_k != DEFAULT_BLOCK_K
        or kv_block_size != DEFAULT_KV_BLOCK_SIZE
        or num_warps != ROWS_PER_CTA
        or heads != DEFAULT_HEADS
        or head_dim != DEFAULT_HEAD_DIM
    ):
        raise ValueError(
            "FP4 prefill requires block_k=256, kv_block_size=64, "
            "num_warps=4, heads=64, and head_dim=128."
        )

    if (cta_info is None) != (n_ctas is None):
        raise ValueError("Pass both cta_info and n_ctas, or neither.")
    schedule_internal = cta_info is None
    if schedule_internal:
        _, cta_info, n_ctas = compute_prefill_schedule(
            row_to_batch,
            local_starts,
            local_ends,
            block_k,
            parallel_unit_num,
            max_seq_len,
        )

    if out is None:
        out = torch.full(
            (total_tokens, max_seq_len),
            float("-inf"),
            dtype=torch.float32,
            device=q_fp4.device,
        )
    elif schedule_internal:
        out.fill_(float("-inf"))

    launcher, _ = compile_pa_mqa_logits_fp4_prefill(
        kv_page_stride=kv_cache.stride(0),
        kv_scale_page_stride=kv_scale.stride(0),
        block_table_stride=block_tables.stride(0),
        weight_scale=float(weight_scale),
    )

    if stream is None:
        stream = torch.cuda.current_stream()

    if n_ctas:
        _run_compiled(
            launcher,
            out,
            q_fp4,
            q_scale,
            kv_cache,
            kv_scale,
            block_tables,
            weights,
            cta_info,
            out.stride(0),
            n_ctas,
            stream,
        )
    return out


@triton.jit
def _varqlen_windows_kernel(
    cu_ptr,  # [B+1] int32, prefix-sum of per-batch qlen
    ctx_ptr,  # [B] int32, per-batch KV length
    row_to_batch_ptr,  # [total_q] int32 (out)
    local_starts_ptr,  # [total_q] int32 (out)
    local_ends_ptr,  # [total_q] int32 (out)
    total_q,
    B,
    BLOCK: tl.constexpr,
):
    """Fused build of ragged-row metadata for per-batch variable qlen (MTP)."""
    pid = tl.program_id(0)
    r = pid * BLOCK + tl.arange(0, BLOCK)
    rmask = r < total_q

    lo = tl.zeros([BLOCK], tl.int32)
    hi = tl.full([BLOCK], B, tl.int32)
    for _ in tl.static_range(32):
        mid = (lo + hi) // 2
        cu_mid = tl.load(
            cu_ptr + 1 + tl.minimum(mid, B - 1), mask=(mid < B), other=2147483647
        )
        go_right = cu_mid <= r
        lo = tl.where(go_right, mid + 1, lo)
        hi = tl.where(go_right, hi, mid)
    b = tl.minimum(lo, B - 1)

    cu_b = tl.load(cu_ptr + b, mask=rmask, other=0)
    cu_b1 = tl.load(cu_ptr + b + 1, mask=rmask, other=0)
    ctx_b = tl.load(ctx_ptr + b, mask=rmask, other=0)
    n = r - cu_b
    qlen = cu_b1 - cu_b
    le = tl.maximum(ctx_b - qlen + n + 1, 0)
    # Rows beyond the real total Σ (cu[B]) are FLAT tail-padding — force an empty
    # window so the mqa kernel / top_k skip them (used when `total_q` is the padded
    # count, e.g. the CUDAGraph decode path scores all padded rows in one shot).
    real_total = tl.load(cu_ptr + B)
    le = tl.where(r < real_total, le, tl.zeros([BLOCK], tl.int32))

    tl.store(row_to_batch_ptr + r, b, mask=rmask)
    tl.store(local_starts_ptr + r, tl.zeros([BLOCK], tl.int32), mask=rmask)
    tl.store(local_ends_ptr + r, le, mask=rmask)


def compute_varqlen_windows(cu_seq_q, context_lens, total_q, *, out=None):
    """Build ragged-row metadata for per-batch variable query length (MTP).

    Pass `out=(row_to_batch, local_starts, local_ends)` (fixed int32 buffers each
    >= total_q long) to write into stable addresses — the CUDAGraph decode path
    scores all padded rows, so top_k replays from these window pointers while
    `build()` refreshes their contents. Rows past the real total (cu[B]) get an
    empty window (local_ends == 0) so they are skipped.
    """
    dev = cu_seq_q.device
    cu = cu_seq_q.to(torch.int32).contiguous()
    ctx = context_lens.to(torch.int32).contiguous()
    B = ctx.shape[0]
    if out is None:
        row_to_batch = torch.empty(total_q, dtype=torch.int32, device=dev)
        local_starts = torch.empty(total_q, dtype=torch.int32, device=dev)
        local_ends = torch.empty(total_q, dtype=torch.int32, device=dev)
    else:
        row_to_batch, local_starts, local_ends = out
    if total_q > 0:
        BLOCK = 256
        grid = (triton.cdiv(total_q, BLOCK),)
        _varqlen_windows_kernel[grid](
            cu,
            ctx,
            row_to_batch,
            local_starts,
            local_ends,
            total_q,
            B,
            BLOCK=BLOCK,
        )
    return row_to_batch, local_starts, local_ends


def flydsl_pa_mqa_logits_fp4_varqlen(
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    weights: torch.Tensor,
    max_seq_len: int,
    *,
    cu_seq_q: torch.Tensor | None = None,
    context_lens: torch.Tensor | None = None,
    windows: tuple | None = None,
    weight_scale: float = 1.0,
    block_k: int = 256,
    kv_block_size: int = 64,
    num_warps: int = DEFAULT_NUM_WARPS,
    parallel_unit_num: int | None = None,
    out: torch.Tensor | None = None,
    cta_info: torch.Tensor | None = None,
    n_ctas: int | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Variable-qlen (per-batch MTP) FP4 paged MQA logits (gfx950)."""
    total_q = q_fp4.shape[0]
    if windows is None:
        if cu_seq_q is None or context_lens is None:
            raise ValueError(
                "flydsl_pa_mqa_logits_fp4_varqlen: pass windows=(row_to_batch, "
                "local_starts, local_ends) built once via "
                "compute_varqlen_windows, or both cu_seq_q and context_lens "
                "to build them here."
            )
        windows = compute_varqlen_windows(cu_seq_q, context_lens, total_q)
    row_to_batch, local_starts, local_ends = windows
    if parallel_unit_num is None:
        chunks_per_seq = max(1, (max_seq_len + block_k - 1) // block_k)
        parallel_unit_num = total_q * chunks_per_seq
    return flydsl_pa_mqa_logits_fp4_prefill(
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        row_to_batch,
        local_starts,
        local_ends,
        max_seq_len,
        weight_scale=weight_scale,
        block_k=block_k,
        kv_block_size=kv_block_size,
        num_warps=num_warps,
        parallel_unit_num=parallel_unit_num,
        out=out,
        cta_info=cta_info,
        n_ctas=n_ctas,
        stream=stream,
    )
