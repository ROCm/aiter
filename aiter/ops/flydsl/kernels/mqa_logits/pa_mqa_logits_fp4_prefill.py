# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa: SIM102


from functools import cache
from typing import NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import triton
import triton.language as tl
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, as_ir_value, const_expr, gpu, rocdl
from flydsl.expr.primitive import range_constexpr
from flydsl.expr.rocdl import ballot, readlane
from flydsl.expr.typing import Float4E2M1FN, Int32, T

from ..kernels_common import atomic_add_i32
from ..tensor_shim import _run_compiled, ptr_buf_tensor
from ..topk_per_row_decode import _warp_inclusive_prefix_i32
from .pa_mqa_litetopk_fp4_common import (
    FP4_LITETOPK_SUPPORTED_TOPKS,
    LiteTopKScanStorage,
    LiteTopKSeedStorage,
)
from .pa_mqa_logits_fp4_common import (
    _NON_WRITER_LANE_OFF,
    _i32_buffer,
    _load_vec4_i32,
)

DEFAULT_HEADS = 64
DEFAULT_HEAD_DIM = 128
DEFAULT_NUM_WARPS = 4
MFMA_M = 16
MFMA_N = 16
WARP_SIZE = 64
DEFAULT_BLOCK_THREADS = DEFAULT_NUM_WARPS * WARP_SIZE  # 256

# cta_info packed fields per CTA.
CTA_INFO_WIDTH = 6


def _popcount_i64(value):
    return fx.Int32(
        llvm.call_intrinsic(
            T.i64,
            "llvm.ctpop.i64",
            [as_ir_value(value)],
            [],
            [],
        )
    )


def compute_prefill_schedule(
    row_to_batch,
    local_starts,
    local_ends,
    block_k,
    parallel_unit_num,
    max_seq_len,
    cta_info_out=None,
):
    """Compute the persistent-grid schedule for ragged-prefill MQA logits.

    Pass `cta_info_out` (a fixed [parallel_unit_num, CTA_INFO_WIDTH] int32 buffer)
    to write the schedule into a stable address (CUDAGraph decode: the captured
    kernel replays from this pointer while `build()` refreshes its contents).

    Returns `(safe, cta_info, parallel_unit_num)`, `safe` being the [1] int32
    split factor the schedule was built with.

    The row plan is every-lane work over one [T] vector, so up to
    `_ROW_PLAN_MAX_ROWS` rows it is one block instead of the ~25 torch ops
    `_row_plan_torch` spells it as. What that buys is HOST latency, not device
    time -- the torch form hides its own kernels behind its own dispatch, and a
    decode step calls this once per forward, in the host gap between two.
    """
    device = local_ends.device
    P = parallel_unit_num
    T = local_ends.shape[0]  # fixed total_tokens (rows)

    assert P >= T, (
        f"compute_prefill_schedule: parallel_unit_num={P} < rows={T} would "
        f"silently drop rows past slot {P} (logits stay at the caller's "
        f"pre-fill -> wrong top-k). Pass parallel_unit_num >= number of rows."
    )

    # Asserted, not coerced. Every caller already holds int32, so the three
    # `.to()` calls this replaces were no-ops -- but on a host-bound path a
    # no-op still costs its dispatch, and a caller that stopped honouring this
    # would silently change what the kernels below compile to.
    assert (
        row_to_batch.dtype == local_starts.dtype == local_ends.dtype == torch.int32
    ), (
        f"compute_prefill_schedule: row_to_batch/local_starts/local_ends must be "
        f"int32, got {row_to_batch.dtype}/{local_starts.dtype}/{local_ends.dtype}."
    )

    s_max = max(1, (max_seq_len + block_k - 1) // block_k)
    plan = _row_plan(local_ends, block_k, P, s_max)

    # ── map each fixed slot → (row, split) + emit cta_info in ONE kernel ──
    if cta_info_out is None:
        cta_info = torch.empty(P, CTA_INFO_WIDTH, dtype=torch.int32, device=device)
    else:
        cta_info = cta_info_out
    BLOCK_P = 256
    grid = (triton.cdiv(P, BLOCK_P),)
    _prefill_cta_info_kernel[grid](
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
        BLOCK_P=BLOCK_P,
    )
    return plan.safe, cta_info, P


class _RowPlan(NamedTuple):
    """Everything the emit kernel needs about the rows, from either producer.

    One carrier, so a field can only be added where both have to answer for it.
    """

    incl: torch.Tensor  # [T] inclusive prefix sum of per-row CTA counts
    excl: torch.Tensor  # [T] exclusive prefix sum
    chunks: torch.Tensor  # [T] ceil(local_end / block_k), 0 for an empty row
    safe: torch.Tensor  # [1] chunk-splits merged into one CTA
    total_splits: torch.Tensor  # [1] number of valid (row, split) slots


# Rows the fused arm plans, and the only two widths it plans them in. A block
# costs its WIDTH, not the rows that fill it -- 40.8us at 16384 lanes whether 300
# rows or 16384 arrive -- so the choice is which shapes share a width, and every
# distinct width is a kernel to compile:
#
#     block    device    cold compile   what reaches it
#      4096    13.3us          0.84s    every decode forward
#     16384    40.8us          1.86s    every prefill forward
#
# The FLOOR is what keeps decode off the wide block: `max_num_seqs * (1 + spec
# steps)` = 512 x 8 fits in one width, so decode compiles once and pays 13.3us.
# Sizing per shape below it saves 0.3us for nine more variants.
#
# NOTHING between them, though an 8192 rung measured 21.1us: on a 100k/10 conc-50
# trace 534 of 541 prefill forwards ran 8193-16384 rows and four ran 4097-8192,
# so that rung would cost a third of the ladder and a 1.13s compile to save 20us
# on 0.7% of forwards -- and a variant nothing exercises is one discovered
# mid-run.
#
# The CAP is where widening stops paying: 32768 is bit-exact and still beats the
# torch arm on both axes (87.1us device against 338.8us), but takes 4.98s to
# compile, and 65536 does not finish compiling at all.
_ROW_PLAN_BLOCK_FLOOR = 4096
_ROW_PLAN_MAX_ROWS = 16384

# int32s per 16 bytes -- the granularity Triton's pointer-alignment
# specialization works at.
_I32_PER_16B = 4


def _row_plan(le, block_k, P, s_max) -> _RowPlan:
    T = le.shape[0]
    if T > _ROW_PLAN_MAX_ROWS:
        return _row_plan_torch(le, block_k, P, s_max)
    # One allocation, sliced: five `torch.empty` calls would put four more
    # dispatches back on the path this exists to shorten.
    #
    # Every region starts on a 16-byte boundary, which is not cosmetic: Triton
    # specializes a kernel on whether each pointer is 16-byte aligned, so a
    # stride of exactly T forks a variant per `T % 4` and the JIT never stops
    # finding new ones mid-run. Rounding the stride up pins all six pointers to
    # one alignment signature.
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
    # Named, not `*plan`: field ORDER should not become load-bearing.
    _prefill_row_plan_kernel[(1,)](
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
        BLOCK_T=(
            _ROW_PLAN_BLOCK_FLOOR if T <= _ROW_PLAN_BLOCK_FLOOR else _ROW_PLAN_MAX_ROWS
        ),
        SEARCH_STEPS=max(1, (s_max - 1).bit_length() + 1),
    )
    return plan


def _row_plan_torch(le, block_k, P, s_max) -> _RowPlan:
    """The row plan as ~25 torch ops. Reference for `_prefill_row_plan_kernel`."""
    # chunk count per row = ceil(le / block_k); le<=0 → 0 chunks.
    chunks_per_row = torch.clamp((le + (block_k - 1)) // block_k, min=0)  # [T]

    s_cand = torch.arange(1, s_max + 1, device=le.device, dtype=torch.int32)  # [s_max]
    ctas_per_r_s = (chunks_per_row[None, :] + (s_cand[:, None] - 1)) // s_cand[
        :, None
    ]  # [s_max, T]
    total_ctas_s = ctas_per_r_s.sum(dim=1)  # [s_max]
    feasible = total_ctas_s <= P  # [s_max] bool, monotonic False..True
    max_chunks = torch.clamp(chunks_per_row.max(), min=1).to(torch.int32)
    # smallest feasible s, via arithmetic (no tensor gather → no capture sync).
    first_feasible_s = torch.clamp((~feasible).to(torch.int32).sum() + 1, max=s_max)
    safe = torch.where(feasible.any(), first_feasible_s, max_chunks).to(torch.int32)

    # ── per-row number of CTAs (chunk-splits); 0 for empty rows ──
    ctas_r = (chunks_per_row + (safe - 1)) // safe  # [T]
    incl = torch.cumsum(ctas_r, dim=0, dtype=torch.int32)  # [T] inclusive prefix sum
    return _RowPlan(
        incl=incl,
        excl=incl - ctas_r,  # exclusive prefix sum
        chunks=chunks_per_row.to(torch.int32),
        safe=safe.reshape(1).to(torch.int32),
        total_splits=incl[-1].reshape(1).to(torch.int32),
    )


# `T` and `P` are batch shape, and a decode step's row count changes every
# forward, so specializing on whether they divide 16 keeps finding new variants
# to compile in the middle of a run. Both kernels here: over 17 shapes it takes
# this pair from 9 and 6 variants down to 3 and 1, and what is left of the 3 is
# exactly the BLOCK_T ladder -- a set small enough to be compiled through.
#
# `T` alone is the whole of that halving AND the whole of its cost: dropping it
# gives up the divisibility hint the wide blocks vectorize on, 41.1us -> 52.0us
# at 16384 lanes. That lands on prefill, one call per ~500ms forward; the widths
# decode runs stay hidden behind the call's own dispatch either way.
@triton.jit(do_not_specialize=["T", "P"])
def _prefill_row_plan_kernel(
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
    SEARCH_STEPS: tl.constexpr,
):
    """Single-block row plan: chunk counts, split factor, and its prefix sums.

    `total_ctas(s) = sum ceil(chunks/s)` is non-increasing in s, so feasibility is
    monotone and the smallest feasible s is a binary search. `_row_plan_torch`
    instead materializes the whole [s_max, T] feasibility matrix and counts its
    False entries -- same answer, and the matrix is why that path's cost also
    tracks the model's context length.

    Masked-out lanes carry `chunks = 0`, which contributes 0 CTAs at every s, so
    no reduction below needs a second mask.

    `SEARCH_STEPS` is derived from `s_max`, not a generous constant: the range
    halves each step, so `(s_max - 1).bit_length()` converges it and the caller
    passes one more. Every surplus step is another reduction over all BLOCK_T
    lanes -- 0.74us each at 4096 rows -- and a fixed 32 was this kernel's ENTIRE
    growth with block width, 28.7us against 12.8us. The one spare step is not
    that: it is hidden behind the call's own dispatch, and coming up short
    returns a schedule that is merely wrong, with nothing downstream to fault.
    """
    t = tl.arange(0, BLOCK_T)
    mask = t < T
    le = tl.load(le_ptr + t, mask=mask, other=0)
    # The clamp puts `le <= 0` at 0 under either rounding, so this agrees with the
    # torch path's floor division without asking which one Triton does.
    chunks = tl.where(mask, tl.maximum((le + block_k - 1) // block_k, 0), 0)
    max_chunks = tl.maximum(tl.max(chunks, axis=0), 1)

    lo = 1
    hi = s_max
    for _ in tl.static_range(SEARCH_STEPS):
        mid = (lo + hi) // 2
        feasible = tl.sum((chunks + mid - 1) // mid, axis=0) <= P
        active = lo < hi
        hi = tl.where(active & feasible, mid, hi)
        lo = tl.where(active & (feasible == 0), mid + 1, lo)
    # No s in [1, s_max] fits: fall back to one CTA per row, as torch's
    # `feasible.any()` arm does.
    total_smax = tl.sum((chunks + s_max - 1) // s_max, axis=0)
    safe = tl.where(total_smax <= P, lo, max_chunks)

    ctas = tl.where(mask, (chunks + safe - 1) // safe, 0)
    incl = tl.cumsum(ctas, axis=0)
    tl.store(incl_ptr + t, incl, mask=mask)
    tl.store(excl_ptr + t, incl - ctas, mask=mask)
    tl.store(chunks_ptr + t, chunks, mask=mask)
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
    total_splits_ptr,  # [1] int32
    cta_info_ptr,  # [P, 6] int32
    T,
    P,
    BLOCK_P: tl.constexpr,
):
    """Single-kernel slot->row mapping + cta_info emit for ragged prefill."""
    pid = tl.program_id(0)
    safe = tl.load(safe_ptr)
    total_splits = tl.load(total_splits_ptr)
    slot = pid * BLOCK_P + tl.arange(0, BLOCK_P)  # [BLOCK_P]
    smask = slot < P
    valid = slot < total_splits

    # searchsorted(incl, slot, right=True) = count(incl <= slot): per-slot
    # binary search over incl[T] in global memory (~log2(T) iters).
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
    safe_row = tl.minimum(lo, T - 1)  # clamp for gather

    excl_r = tl.load(excl_ptr + safe_row, mask=smask, other=0)
    chunks_r = tl.load(chunks_ptr + safe_row, mask=smask, other=0)
    rb_r = tl.load(rb_ptr + safe_row, mask=smask, other=0)
    ls_r = tl.load(ls_ptr + safe_row, mask=smask, other=0)
    le_r = tl.load(le_ptr + safe_row, mask=smask, other=0)

    vi = valid.to(tl.int32)
    split_within = slot - excl_r
    start = split_within * safe  # pre-mask (count uses this)
    count = tl.maximum(tl.minimum(safe, chunks_r - start), 0)
    row_id = safe_row * vi
    batch_id = rb_r * vi
    start = start * vi
    count = tl.where(valid, count, 1)
    ls_out = ls_r * vi
    le_out = le_r * vi

    base = slot * 6
    tl.store(cta_info_ptr + base + 0, row_id, mask=smask)
    tl.store(cta_info_ptr + base + 1, batch_id, mask=smask)
    tl.store(cta_info_ptr + base + 2, start, mask=smask)
    tl.store(cta_info_ptr + base + 3, count, mask=smask)
    tl.store(cta_info_ptr + base + 4, ls_out, mask=smask)
    tl.store(cta_info_ptr + base + 5, le_out, mask=smask)


def build_pa_mqa_logits_fp4_prefill_module(
    block_k=256,
    kv_block_size=64,
    max_chunks_per_cta=16,
    num_warps=DEFAULT_NUM_WARPS,
    heads=DEFAULT_HEADS,
    head_dim=DEFAULT_HEAD_DIM,
    relative_output=False,
    litetopk=False,
    seed_calibration=False,
    seed_emit=False,
    seed_status_nonfinite=1 << 2,
    seed_status_candidate_overflow=1 << 0,
    seed_status_underfilled=1 << 1,
    litetopk_topk=512,
    litetopk_refresh_every=64,
    seed_report_page_errors=False,
):
    """Build the ragged-prefill FP4 MQA logits kernel."""
    block_threads_k = num_warps * WARP_SIZE
    m_tiles = heads // MFMA_M
    k_tiles = head_dim // 128  # outer K-loop iters (MFMA K=128)
    assert (
        head_dim % 128 == 0
    ), f"head_dim must be a multiple of 128 (MFMA K), got {head_dim}"
    assert heads % MFMA_M == 0, f"heads must be a multiple of {MFMA_M}, got {heads}"

    N_TILES = block_k // MFMA_N
    assert (
        N_TILES % num_warps == 0
    ), f"block_k={block_k} -> N_TILES={N_TILES} must be multiple of num_warps={num_warps}"
    N_TILES_PER_WARP = N_TILES // num_warps
    assert not (litetopk and seed_calibration)
    assert not seed_calibration or relative_output
    assert not seed_calibration or num_warps <= 8
    assert not seed_emit or seed_calibration
    assert not seed_report_page_errors or seed_emit

    assert (
        kv_block_size % MFMA_N == 0
    ), f"kv_block_size={kv_block_size} must be a multiple of MFMA_N={MFMA_N}"
    assert (
        block_k % kv_block_size == 0
    ), f"block_k={block_k} must be a multiple of kv_block_size={kv_block_size}"
    TILES_PER_BLOCK = kv_block_size // MFMA_N
    N_PHYS = (N_TILES_PER_WARP + TILES_PER_BLOCK - 1) // TILES_PER_BLOCK

    # KV preshuffle layout: [block_id, K_TILES, K_chunk=4, kv_block_size, 16] uint8.
    _kv_chunk_bytes = 16
    _stride_kv_ktile = 4 * kv_block_size * _kv_chunk_bytes
    _stride_kv_block = k_tiles * _stride_kv_ktile
    # byte stride between consecutive nt tiles inside one kv block (one MFMA_N
    # row of tokens); used as the per-nt constant `soffset` immediate delta.
    _stride_kv_ntile = MFMA_N * _kv_chunk_bytes
    # KV_scale: [block_id, K_TILES, K_chunks=4, kv_block_size]
    _stride_kvs_ktile = 4 * kv_block_size
    _stride_kvs_block = k_tiles * _stride_kvs_ktile

    _kb_is_pow2 = kv_block_size & (kv_block_size - 1) == 0
    _kb_log2 = kv_block_size.bit_length() - 1
    _kb_mask = kv_block_size - 1

    def _floordiv_kb(x):
        return (x >> fx.Int32(_kb_log2)) if _kb_is_pow2 else (x // kv_block_size)

    def _mod_kb(x):
        return (x & fx.Int32(_kb_mask)) if _kb_is_pow2 else (x % kv_block_size)

    QS_DW = (m_tiles + 3) // 4
    qs_pad = QS_DW * 4
    qs_pad_bits = qs_pad * 8

    def _make_qs_buf_copy():
        if qs_pad_bits == 32:
            return fx.rocdl.BufferCopy32b()
        elif qs_pad_bits == 64:
            return fx.rocdl.BufferCopy64b()
        elif qs_pad_bits == 128:
            return fx.rocdl.BufferCopy128b()
        else:
            raise ValueError(f"unsupported QS_DW={QS_DW} (qs_pad_bits={qs_pad_bits})")

    if N_PHYS == 1:

        def _phys_to_list(phys_v):
            return [phys_v] * N_TILES_PER_WARP

    else:

        def _phys_to_list(phys_v):
            return [
                fx.Vector(phys_v)[nt // TILES_PER_BLOCK]
                for nt in range(N_TILES_PER_WARP)
            ]

    @flyc.kernel
    def pa_mqa_logits_fp4_prefill_kernel(
        out_logits_ptr: fx.Tensor,
        q_ptr: fx.Tensor,
        q_scale_ptr: fx.Tensor,
        kv_cache_ptr: fx.Tensor,
        kv_scale_ptr: fx.Tensor,
        kv_indices_ptr: fx.Tensor,
        weights_ptr: fx.Tensor,
        cta_info_ptr: fx.Tensor,  # [n_ctas, 6] i32
        stride_out_row: Int32,
        stride_block_table: Int32,
        block_table_rows: Int32,
        weight_scale: fx.Float32,
        origin_ptr: fx.Tensor,
        inv_delta_ptr: fx.Tensor,
        threshold_ptr: fx.Tensor,
        histogram_ptr: fx.Tensor,
        candidate_values_ptr: fx.Tensor,
        candidate_indices_ptr: fx.Tensor,
        candidate_counts_ptr: fx.Tensor,
        page_errors_ptr: fx.Tensor,
        score_errors_ptr: fx.Tensor,
        status_ptr: fx.Tensor,
        candidate_stride: Int32,
        merge_cap: Int32,
        block_table_capacity: Int32,
        physical_page_capacity: Int32,
    ):
        tid = gpu.thread_idx.x
        pid = gpu.block_idx.x

        warp_id = tid >> 6
        lane_id = tid % WARP_SIZE
        lane_mod_16 = lane_id & 15
        lane_div_16 = (lane_id >> 4) & 3

        # Per-CTA assignment: pid*CTA_INFO_WIDTH (wave-uniform) folded into the V#
        # base pointer so the 4 fields load as a dwordx4 at row 0; the window
        # bounds (fields 4,5) read as scalars at row 1 cols 0,1.
        cta_src = fx.get_iter(cta_info_ptr)
        cta_it = fx.add_offset(
            fx.recast_iter(fx.PointerType.get(T.i32, cta_src.memspace, 4), cta_src),
            fx.Int64(pid) * fx.Int64(CTA_INFO_WIDTH),
        )
        cta_info_bt = fx.rocdl.make_buffer_tensor(
            fx.make_view(cta_it, fx.make_layout((1 << 28, 4), (4, 1)))
        )
        cta_info_vec = fx.Vector(_load_vec4_i32(cta_info_bt, fx.Int32(0)))

        # Wave-uniform by construction, but they arrive in VGPRs via the buffer
        # load; the V# below needs num_records in an SGPR or the store is wrapped
        # in a waterfall loop.
        def _uniform(v):
            return fx.Int32(fx.rocdl.readfirstlane(T.i32, v))

        local_start = _uniform(cta_info_bt[(fx.Int32(1), fx.Int32(0))])
        local_end = _uniform(cta_info_bt[(fx.Int32(1), fx.Int32(1))])

        ZERO_F = fx.Float32(0.0)
        c0_i32 = fx.Int32(0)

        row_id = _uniform(cta_info_vec[0])
        batch_id = _uniform(cta_info_vec[1])
        chunk_start = _uniform(cta_info_vec[2])
        chunk_count = _uniform(cta_info_vec[3])

        win_len = local_end - local_start
        if const_expr(litetopk):
            origin_buf = ptr_buf_tensor(fx.get_iter(origin_ptr), fx.Float32)
            inv_delta_buf = ptr_buf_tensor(fx.get_iter(inv_delta_ptr), fx.Float32)
            threshold_buf = ptr_buf_tensor(fx.get_iter(threshold_ptr), fx.Int32)
            histogram_buf = ptr_buf_tensor(fx.get_iter(histogram_ptr), fx.Int32)
            candidate_values_it = fx.add_offset(
                fx.recast_iter(
                    fx.PointerType.get(T.f32, candidate_values_ptr.memspace, 4),
                    fx.get_iter(candidate_values_ptr),
                ),
                fx.Int64(row_id) * fx.Int64(candidate_stride),
            )
            candidate_indices_it = fx.add_offset(
                fx.recast_iter(
                    fx.PointerType.get(T.i32, candidate_indices_ptr.memspace, 4),
                    fx.get_iter(candidate_indices_ptr),
                ),
                fx.Int64(row_id) * fx.Int64(candidate_stride),
            )
            candidate_values_buf = ptr_buf_tensor(candidate_values_it, fx.Float32)
            candidate_indices_buf = ptr_buf_tensor(candidate_indices_it, fx.Int32)
            candidate_counts_buf = ptr_buf_tensor(
                fx.get_iter(candidate_counts_ptr), fx.Int32
            )
            score_errors_buf = ptr_buf_tensor(fx.get_iter(score_errors_ptr), fx.Int32)
            row_origin = origin_buf[row_id]
            row_inv_delta = inv_delta_buf[row_id]
            row_threshold = threshold_buf[row_id]
            scan_storage = fx.SharedAllocator().allocate(LiteTopKScanStorage)
            scan_histogram = scan_storage.histogram.peek().view(fx.make_layout(256, 1))
            scan_prefix = scan_storage.scan.peek().view(
                fx.make_layout(num_warps + 1, 1)
            )
            scan_state = scan_storage.state.peek().view(fx.make_layout(3, 1))
            if tid < fx.Int32(256):
                scan_histogram[tid] = histogram_buf[row_id * fx.Int32(256) + tid]
            if tid == 0:
                scan_state[0] = candidate_counts_buf[row_id]
                scan_state[1] = row_threshold
                scan_state[2] = score_errors_buf[row_id]
            gpu.barrier()
        else:
            # A V# spanning exactly [local_start, local_end) of this row:
            # num_records is the window test in hardware. Non-writer lanes use
            # an offset outside the descriptor and are dropped by the same bound.
            if const_expr(not seed_emit):
                _row_elems = fx.Int64(row_id) * fx.Int64(stride_out_row)
                if const_expr(not relative_output):
                    _row_elems = _row_elems + fx.Int64(local_start)
                out_win = fx.rocdl.make_buffer_tensor(
                    fx.make_view(
                        fx.recast_iter(
                            fx.PointerType.get(T.f32, out_logits_ptr.memspace, 4),
                            fx.add_offset(fx.get_iter(out_logits_ptr), _row_elems),
                        ),
                        fx.make_layout((win_len, 1), (1, 1)),
                    ),
                    max_size=False,
                    num_records_bytes=win_len * fx.Int32(4),
                )
                out_lane_off = lane_mod_16 + (lane_div_16 > fx.Int32(0)).select(
                    fx.Int32(_NON_WRITER_LANE_OFF), fx.Int32(0)
                )
                out_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 1)
                out_reg_ty = fx.MemRefType.get(
                    T.f32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register
                )
                out_reg_lay = fx.make_layout(1, 1)
            if const_expr(seed_calibration):
                origin_buf = ptr_buf_tensor(fx.get_iter(origin_ptr), fx.Float32)
                inv_delta_buf = ptr_buf_tensor(fx.get_iter(inv_delta_ptr), fx.Float32)
                status_buf = ptr_buf_tensor(fx.get_iter(status_ptr), fx.Int32)
                seed_storage = fx.SharedAllocator().allocate(LiteTopKSeedStorage)
                seed_scores = seed_storage.scores.peek().view(fx.make_layout(8192, 1))
                seed_histogram = seed_storage.histogram.peek().view(
                    fx.make_layout(256, 1)
                )
                seed_scan = seed_storage.scan.peek().view(
                    fx.make_layout(num_warps + 1, 1)
                )
                seed_maxima = seed_storage.maxima.peek().view(fx.make_layout(8, 1))
                seed_neg_minima = seed_storage.neg_minima.peek().view(
                    fx.make_layout(8, 1)
                )
                seed_finite_counts = seed_storage.finite_counts.peek().view(
                    fx.make_layout(8, 1)
                )
                seed_nonfinite_counts = seed_storage.nonfinite_counts.peek().view(
                    fx.make_layout(8, 1)
                )
                seed_calibration_values = seed_storage.calibration.peek().view(
                    fx.make_layout(2, 1)
                )
                seed_state = seed_storage.state.peek().view(fx.make_layout(2, 1))
                if const_expr(seed_emit):
                    threshold_buf = ptr_buf_tensor(fx.get_iter(threshold_ptr), fx.Int32)
                    histogram_buf = ptr_buf_tensor(fx.get_iter(histogram_ptr), fx.Int32)
                    candidate_values_it = fx.add_offset(
                        fx.recast_iter(
                            fx.PointerType.get(T.f32, candidate_values_ptr.memspace, 4),
                            fx.get_iter(candidate_values_ptr),
                        ),
                        fx.Int64(row_id) * fx.Int64(candidate_stride),
                    )
                    candidate_indices_it = fx.add_offset(
                        fx.recast_iter(
                            fx.PointerType.get(
                                T.i32, candidate_indices_ptr.memspace, 4
                            ),
                            fx.get_iter(candidate_indices_ptr),
                        ),
                        fx.Int64(row_id) * fx.Int64(candidate_stride),
                    )
                    candidate_values_buf = ptr_buf_tensor(
                        candidate_values_it, fx.Float32
                    )
                    candidate_indices_buf = ptr_buf_tensor(
                        candidate_indices_it, fx.Int32
                    )
                    candidate_counts_buf = ptr_buf_tensor(
                        fx.get_iter(candidate_counts_ptr), fx.Int32
                    )
                    seed_select_starts_buf = ptr_buf_tensor(
                        fx.get_iter(out_logits_ptr), fx.Int32
                    )
                    seed_select_ends_buf = ptr_buf_tensor(
                        fx.get_iter(score_errors_ptr), fx.Int32
                    )

        # Q load (hoisted): per (k_tile, mi_idx) a thread loads its 16-byte FP4
        # chunk for head row mi_idx*16+lane_mod_16. Q: [total_tokens, H, D/2] uint8.
        # Scaled FP4 16x16x128 MMA; opsel_b selects the per-nt scale byte, so one
        # atom per nt (opsel_a stays 0 — Q scale is one byte per (k_tile, mi)).
        mfma_atoms = [
            fx.make_mma_atom(
                fx.rocdl.cdna4.MFMA_Scale(
                    16, 16, 128, Float4E2M1FN, Float4E2M1FN, opsel_a=0, opsel_b=nt
                )
            )
            for nt in range(N_TILES_PER_WARP)
        ]

        Q_buf = fx.rocdl.make_buffer_tensor(q_ptr)
        q_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 8)
        q_reg_ty = fx.MemRefType.get(
            T.i8, fx.LayoutType.get(16, 1), fx.AddressSpace.Register
        )
        q_reg_lay = fx.make_layout(16, 1)
        q_a_ops = []
        for k_tile in range_constexpr(k_tiles):
            q_a_ops_kt = []
            for mi_idx in range_constexpr(m_tiles):
                q_row = fx.Int32(mi_idx * MFMA_M) + lane_mod_16
                q_row_bytes = fx.slice(Q_buf, (row_id, q_row, None))
                q_row_div = fx.logical_divide(q_row_bytes, fx.make_layout(16, 1))
                col_idx = fx.Int32(k_tile * 4) + lane_div_16
                r = fx.memref_alloca(q_reg_ty, q_reg_lay)
                fx.copy(q_atom, fx.slice(q_row_div, (None, col_idx)), r)
                q_4xi32 = fx.Vector(fx.memref_load_vec(r)).bitcast(fx.Int32)
                a_frag = fx.make_rmem_tensor(4, fx.Int32)
                a_frag.store(q_4xi32)
                q_a_ops_kt.append(a_frag)
            q_a_ops.append(q_a_ops_kt)

        # Q scale: host-preshuffled [total_tokens, K_TILES, 4, 16, QS_PAD].
        assert m_tiles <= 8, f"m_tiles={m_tiles} > 8 not supported. Use heads <= 128."
        QS_buf = fx.rocdl.make_buffer_tensor(q_scale_ptr)
        qs_atom = fx.make_copy_atom(_make_qs_buf_copy(), 8)
        qs_reg_ty = fx.MemRefType.get(
            T.i8, fx.LayoutType.get(qs_pad, 1), fx.AddressSpace.Register
        )
        qs_reg_lay = fx.make_layout(qs_pad, 1)
        q_scale_ops = []
        for k_tile in range_constexpr(k_tiles):
            row = fx.slice(
                QS_buf, (row_id, fx.Int32(k_tile), lane_div_16, lane_mod_16, None)
            )
            r = fx.memref_alloca(qs_reg_ty, qs_reg_lay)
            fx.copy(qs_atom, row, r)
            qs_dws_vec = fx.Vector(fx.memref_load_vec(r)).bitcast(fx.Int32)
            qs_dws = [qs_dws_vec[i] for i in range(QS_DW)]
            q_scale_ops.append(
                [qs_dws[mi // 4] >> fx.Int32(8 * (mi % 4)) for mi in range(m_tiles)]
            )

        # Weights (hoisted): [total_tokens, H] bf16, addressed by row_id.
        # Loaded as bf16 then widened to f32 for the per-head weighting below.
        W_buf = fx.rocdl.make_buffer_tensor(weights_ptr)
        w_row = fx.slice(W_buf, (row_id, None))
        w_tiled_mi = fx.logical_divide(w_row, fx.make_layout(MFMA_M, 1))
        w_atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), 16)
        w_reg_ty = fx.MemRefType.get(
            T.bf16, fx.LayoutType.get(4, 1), fx.AddressSpace.Register
        )
        w_reg_lay = fx.make_layout(4, 1)
        ws_vec = fx.Vector.from_elements([weight_scale] * 4, dtype=fx.Float32)
        w_per_lane = []
        for mi_idx in range_constexpr(m_tiles):
            tile = fx.slice(w_tiled_mi, (None, fx.Int32(mi_idx)))
            tile_div = fx.logical_divide(tile, fx.make_layout(4, 1))
            r = fx.memref_alloca(w_reg_ty, w_reg_lay)
            fx.copy(w_atom, fx.slice(tile_div, (None, lane_div_16)), r)
            w_f32 = fx.Vector(fx.memref_load_vec(r).to(fx.Float32))
            w_per_lane.append(w_f32 * ws_vec)

        # ── prologue + N-1 prefetch loop + epilogue ──

        def _load_phys(c_i32_arg):
            ni_base = warp_id * fx.Int32(N_TILES_PER_WARP)
            token_local_base = (
                (chunk_start + c_i32_arg) * fx.Int32(block_k)
                + ni_base * fx.Int32(MFMA_N)
                + lane_mod_16
            )
            bi_base = _floordiv_kb(token_local_base)
            if const_expr(litetopk or relative_output):
                batch_valid = (batch_id >= fx.Int32(0)) & (batch_id < block_table_rows)
                page_valid = (bi_base >= fx.Int32(0)) & (bi_base < block_table_capacity)
                safe_batch = batch_valid.select(batch_id, fx.Int32(0))
                safe_page = page_valid.select(bi_base, fx.Int32(0))
                bt_row = _i32_buffer(
                    kv_indices_ptr,
                    width=1,
                    elem_offset=fx.Int64(safe_batch) * fx.Int64(stride_block_table),
                )
                phys_vec = bt_row[safe_page]
                physical_valid = (phys_vec >= fx.Int32(0)) & (
                    phys_vec < physical_page_capacity
                )
                token_page_start = (chunk_start + c_i32_arg) * fx.Int32(
                    block_k
                ) + ni_base * fx.Int32(MFMA_N)
                page_live = (token_page_start < local_end) & (
                    token_page_start + fx.Int32(kv_block_size) > local_start
                )
            else:
                bt_row = _i32_buffer(
                    kv_indices_ptr,
                    width=1,
                    elem_offset=fx.Int64(batch_id) * fx.Int64(stride_block_table),
                )
                phys_vec = bt_row[bi_base]
                physical_valid = fx.Boolean(True)
                safe_phys = phys_vec
            if const_expr(litetopk or seed_report_page_errors):
                if (
                    (lane_id == 0)
                    & page_live
                    & ~(batch_valid & page_valid & physical_valid)
                ):
                    atomic_add_i32(page_errors_ptr, 1, row_id, "agent")
            if const_expr(litetopk or relative_output):
                safe_phys = physical_valid.select(phys_vec, fx.Int32(0))
            return _phys_to_list(safe_phys)

        def _prefetch_chunk(c_i32_arg, phys_list):
            assert N_TILES_PER_WARP == 4, "packed kvs assumes NTPW=4"
            assert N_PHYS == 1, "packed kvs assumes N_PHYS=1 (NTPW nts share one phys)"

            kv_list = []
            kvs_packed_list = []

            phys_shared = _uniform(phys_list[0])
            kv_bt = _i32_buffer(
                kv_cache_ptr,
                width=4,
                elem_offset=fx.Int64(phys_shared) * fx.Int64(_stride_kv_block // 4),
            )
            kvs_bt = _i32_buffer(
                kv_scale_ptr,
                width=1,
                elem_offset=fx.Int64(phys_shared) * fx.Int64(_stride_kvs_block // 4),
            )
            kvs_base_off_elems = (
                lane_div_16 * kv_block_size + lane_mod_16 * fx.Int32(N_TILES_PER_WARP)
            ) >> fx.Int32(2)
            for k_tile in range_constexpr(k_tiles):
                kvs_packed = kvs_bt[
                    kvs_base_off_elems + fx.Int32(k_tile * _stride_kvs_ktile // 4)
                ]
                kvs_packed_list.append(kvs_packed)

            ni0 = warp_id * fx.Int32(N_TILES_PER_WARP)
            token_local0 = (
                (chunk_start + c_i32_arg) * fx.Int32(block_k)
                + ni0 * fx.Int32(MFMA_N)
                + lane_mod_16
            )
            token_in_block0 = _mod_kb(token_local0)
            kv_base_off_elems = (
                lane_div_16 * kv_block_size * _kv_chunk_bytes
                + token_in_block0 * _kv_chunk_bytes
            ) >> fx.Int32(2)
            for nt in range_constexpr(N_TILES_PER_WARP):
                for k_tile in range_constexpr(k_tiles):
                    kv_soffset = k_tile * _stride_kv_ktile + nt * _stride_kv_ntile
                    kv_c = _load_vec4_i32(
                        kv_bt, kv_base_off_elems + fx.Int32(kv_soffset // 4)
                    )
                    kv_list.append(kv_c)

            return kv_list, kvs_packed_list

        def _issue_nt_mfmas(kv_list_in, kvs_packed_per_kt, nt):
            zero = fx.Vector.filled(4, 0.0, fx.Float32)
            accs = [zero] * m_tiles
            # opsel_b=nt is baked into the atom; scale_b is the packed 4-nt word.
            atom = mfma_atoms[nt]
            for k_tile in range_constexpr(k_tiles):
                b_frag = fx.make_rmem_tensor(4, fx.Int32)
                b_frag.store(fx.Vector(kv_list_in[nt * k_tiles + k_tile]))
                kv_scale_packed = kvs_packed_per_kt[k_tile]
                for mi_idx in range_constexpr(m_tiles):
                    c_frag = fx.make_rmem_tensor(4, fx.Float32)
                    c_frag.store(fx.Vector(accs[mi_idx]))
                    fx.gemm(
                        atom,
                        c_frag,
                        q_a_ops[k_tile][mi_idx],
                        b_frag,
                        c_frag,
                        scale_a=q_scale_ops[k_tile][mi_idx],
                        scale_b=kv_scale_packed,
                    )
                    accs[mi_idx] = c_frag.load()
            return accs

        def _store_candidate(values_buf, indices_buf, offset, score, logical_index):
            values_buf[offset] = score
            indices_buf[offset] = logical_index

        def _block_exclusive_prefix_i32(value, scan):
            warp = tid // fx.Int32(WARP_SIZE)
            inclusive = _warp_inclusive_prefix_i32(value, lane_id, WARP_SIZE)
            exclusive = inclusive - value
            if lane_id == fx.Int32(WARP_SIZE - 1):
                scan[warp] = inclusive
            gpu.barrier()
            if warp == 0:
                wave_value = fx.Int32(0)
                if lane_id < fx.Int32(num_warps):
                    wave_value = scan[lane_id]
                wave_inclusive = _warp_inclusive_prefix_i32(
                    wave_value, lane_id, WARP_SIZE
                )
                if lane_id < fx.Int32(num_warps):
                    scan[lane_id] = wave_inclusive - wave_value
                if lane_id == fx.Int32(num_warps - 1):
                    scan[num_warps] = wave_inclusive
            gpu.barrier()
            result = scan[warp] + exclusive
            total = scan[num_warps]
            gpu.barrier()
            return result, total

        def _refresh_litetopk_threshold(histogram, state, scan):
            gpu.barrier()
            count = fx.Int32(0)
            if tid < fx.Int32(256):
                count = histogram[tid]
            before, _ = _block_exclusive_prefix_i32(count, scan)
            old_threshold = state[1]
            crosses_topk = (before < fx.Int32(litetopk_topk)) & (
                before + count >= fx.Int32(litetopk_topk)
            )
            if crosses_topk & (tid < old_threshold):
                state[1] = tid
            gpu.barrier()

        def _reduce_nt_score(accs):
            zero = fx.Vector.filled(4, 0.0, fx.Float32)
            thread_sum = ZERO_F
            for mi_idx in range_constexpr(m_tiles):
                relu_v = fx.Vector(accs[mi_idx]).maximumf(zero)
                w_v = fx.Vector(w_per_lane[mi_idx])
                for elem in [0, 1, 2, 3]:
                    thread_sum = fx.fma(relu_v[elem], w_v[elem], thread_sum)

            lane_i32 = fx.Int32(lane_id)

            def _bperm_xor_add(val, sh):
                peer_byte = (lane_i32 ^ fx.Int32(sh)) * fx.Int32(4)
                peer_i32 = fx.Int32(
                    rocdl.ds_bpermute(T.i32, peer_byte, val.bitcast(fx.Int32))
                )
                return val + peer_i32.bitcast(fx.Float32)

            thread_sum = _bperm_xor_add(thread_sum, 16)
            return _bperm_xor_add(thread_sum, 32)

        def _post_process_nt(accs, nt, c_i32_arg, chunk_threshold):
            """relu + per-head weight + per-thread sum + bperm + windowed store."""
            ni_warp = warp_id * fx.Int32(N_TILES_PER_WARP) + fx.Int32(nt)
            token_base = (chunk_start + c_i32_arg) * fx.Int32(
                block_k
            ) + ni_warp * fx.Int32(MFMA_N)
            thread_sum = _reduce_nt_score(accs)
            lane_i32 = fx.Int32(lane_id)
            # `weight_scale` already folded into `w_per_lane` (hoisted, once/wave).

            if const_expr(litetopk):
                logical_index = token_base + lane_mod_16
                is_writer = lane_div_16 == fx.Int32(0)
                in_window = (logical_index >= local_start) & (logical_index < local_end)
                score_bits = thread_sum.bitcast(fx.Int32) & fx.Int32(0x7FFFFFFF)
                finite = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    score_bits,
                    fx.Int32(0x7F800000),
                )
                genuine = is_writer & in_window & finite
                bucket_f = (-thread_sum - row_origin) * row_inv_delta
                bucket_f = finite.select(bucket_f, fx.Float32(255.0))
                bucket = fx.Int32(fx.clampf(bucket_f, 0.0, 255.0))
                keep = genuine & (bucket <= chunk_threshold)
                if keep:
                    atomic_add_i32(
                        scan_histogram,
                        1,
                        bucket,
                        "workgroup",
                    )
                nonfinite = is_writer & in_window & ~finite
                if nonfinite:
                    atomic_add_i32(scan_state, 1, 2, "workgroup")

                keep_mask = fx.Int64(ballot(T.i64, keep))
                if keep_mask != fx.Int64(0):
                    lower_mask = (fx.Int64(1) << fx.Int64(lane_i32)) - fx.Int64(1)
                    exclusive = _popcount_i64(keep_mask & lower_mask)
                    wave_total = _popcount_i64(keep_mask)
                    wave_base_lane0 = fx.Int32(0)
                    if lane_id == 0:
                        wave_base_lane0 = atomic_add_i32(
                            scan_state,
                            wave_total,
                            0,
                            "workgroup",
                        )
                    wave_base = fx.Int32(readlane(T.i32, wave_base_lane0.ir_value(), 0))
                    if keep:
                        candidate_slot = wave_base + exclusive
                        if candidate_slot < merge_cap:
                            _store_candidate(
                                candidate_values_buf,
                                candidate_indices_buf,
                                candidate_slot,
                                thread_sum,
                                logical_index,
                            )
            else:
                r_out = fx.memref_alloca(out_reg_ty, out_reg_lay)
                fx.memref_store_vec(
                    fx.Vector.from_elements([thread_sum], dtype=fx.Float32), r_out
                )
                fx.copy(
                    out_atom,
                    r_out,
                    fx.slice(out_win, (token_base - local_start + out_lane_off, None)),
                )

        def _post_process_litetopk_chunk(
            score_nt0,
            score_nt1,
            score_nt2,
            score_nt3,
            c_i32_arg,
            chunk_threshold,
        ):
            thread_sum = (lane_div_16 == fx.Int32(1)).select(score_nt1, score_nt0)
            thread_sum = (lane_div_16 == fx.Int32(2)).select(score_nt2, thread_sum)
            thread_sum = (lane_div_16 == fx.Int32(3)).select(score_nt3, thread_sum)
            lane_i32 = fx.Int32(lane_id)
            logical_index = (
                (chunk_start + c_i32_arg) * fx.Int32(block_k)
                + warp_id * fx.Int32(N_TILES_PER_WARP * MFMA_N)
                + lane_i32
            )
            in_window = (logical_index >= local_start) & (logical_index < local_end)
            score_bits = thread_sum.bitcast(fx.Int32) & fx.Int32(0x7FFFFFFF)
            finite = arith.cmpi(
                arith.CmpIPredicate.ult,
                score_bits,
                fx.Int32(0x7F800000),
            )
            genuine = in_window & finite
            bucket_f = (-thread_sum - row_origin) * row_inv_delta
            bucket_f = finite.select(bucket_f, fx.Float32(255.0))
            bucket = fx.Int32(fx.clampf(bucket_f, 0.0, 255.0))
            keep = genuine & (bucket <= chunk_threshold)
            if keep:
                atomic_add_i32(
                    scan_histogram,
                    1,
                    bucket,
                    "workgroup",
                )
            if in_window & ~finite:
                atomic_add_i32(scan_state, 1, 2, "workgroup")

            keep_mask = fx.Int64(ballot(T.i64, keep))
            if keep_mask != fx.Int64(0):
                lower_mask = (fx.Int64(1) << fx.Int64(lane_i32)) - fx.Int64(1)
                exclusive = _popcount_i64(keep_mask & lower_mask)
                wave_total = _popcount_i64(keep_mask)
                wave_base_lane0 = fx.Int32(0)
                if lane_id == 0:
                    wave_base_lane0 = atomic_add_i32(
                        scan_state,
                        wave_total,
                        0,
                        "workgroup",
                    )
                wave_base = fx.Int32(readlane(T.i32, wave_base_lane0.ir_value(), 0))
                if keep:
                    candidate_slot = wave_base + exclusive
                    if candidate_slot < merge_cap:
                        _store_candidate(
                            candidate_values_buf,
                            candidate_indices_buf,
                            candidate_slot,
                            thread_sum,
                            logical_index,
                        )

        def _post_process_seed_chunk(
            score_nt0,
            score_nt1,
            score_nt2,
            score_nt3,
            c_i32_arg,
            seed_scores_out,
            row_max,
            row_neg_min,
            finite_count,
            nonfinite_count,
        ):
            thread_sum = (lane_div_16 == fx.Int32(1)).select(score_nt1, score_nt0)
            thread_sum = (lane_div_16 == fx.Int32(2)).select(score_nt2, thread_sum)
            thread_sum = (lane_div_16 == fx.Int32(3)).select(score_nt3, thread_sum)
            logical_index = (
                (chunk_start + c_i32_arg) * fx.Int32(block_k)
                + warp_id * fx.Int32(N_TILES_PER_WARP * MFMA_N)
                + fx.Int32(lane_id)
            )
            in_window = (logical_index >= local_start) & (logical_index < local_end)

            if const_expr(seed_emit):
                if in_window:
                    seed_scores_out[logical_index - local_start] = thread_sum
            else:
                r_out = fx.memref_alloca(out_reg_ty, out_reg_lay)
                fx.memref_store_vec(
                    fx.Vector.from_elements([thread_sum], dtype=fx.Float32), r_out
                )
                fx.copy(
                    out_atom,
                    r_out,
                    fx.slice(out_win, (logical_index - local_start, None)),
                )

            score_bits = thread_sum.bitcast(fx.Int32) & fx.Int32(0x7FFFFFFF)
            finite = arith.cmpi(
                arith.CmpIPredicate.ult,
                score_bits,
                fx.Int32(0x7F800000),
            )
            genuine = in_window & finite
            neg_inf = fx.Float32(-float("inf"))
            row_max = row_max.maximumf(genuine.select(thread_sum, neg_inf))
            row_neg_min = row_neg_min.maximumf(genuine.select(-thread_sum, neg_inf))
            finite_count = finite_count + genuine.select(fx.Int32(1), fx.Int32(0))
            nonfinite_count = nonfinite_count + (in_window & ~finite).select(
                fx.Int32(1), fx.Int32(0)
            )
            return row_max, row_neg_min, finite_count, nonfinite_count

        def _compute_chunk(
            kv_list_in,
            kvs_packed_list_in,
            c_i32_arg,
            nt0_accs_in=None,
            seed_accs=None,
        ):
            assert (
                N_TILES_PER_WARP == 4
            ), "pipelined-nt structure currently hardcoded for NTPW=4"

            accs_nt0 = (
                _issue_nt_mfmas(kv_list_in, kvs_packed_list_in, 0)
                if nt0_accs_in is None
                else list(nt0_accs_in)
            )
            chunk_threshold = scan_state[1] if const_expr(litetopk) else c0_i32

            if const_expr(litetopk or seed_calibration):
                accs_nt1 = _issue_nt_mfmas(kv_list_in, kvs_packed_list_in, 1)
                score_nt0 = _reduce_nt_score(accs_nt0)
                accs_nt2 = _issue_nt_mfmas(kv_list_in, kvs_packed_list_in, 2)
                score_nt1 = _reduce_nt_score(accs_nt1)
                accs_nt3 = _issue_nt_mfmas(kv_list_in, kvs_packed_list_in, 3)
                score_nt2 = _reduce_nt_score(accs_nt2)
                score_nt3 = _reduce_nt_score(accs_nt3)
                if const_expr(litetopk):
                    _post_process_litetopk_chunk(
                        score_nt0,
                        score_nt1,
                        score_nt2,
                        score_nt3,
                        c_i32_arg,
                        chunk_threshold,
                    )
                    return []
                return list(
                    _post_process_seed_chunk(
                        score_nt0,
                        score_nt1,
                        score_nt2,
                        score_nt3,
                        c_i32_arg,
                        seed_scores,
                        *seed_accs,
                    )
                )
            else:
                accs_nt1 = _issue_nt_mfmas(kv_list_in, kvs_packed_list_in, 1)
                _post_process_nt(accs_nt0, 0, c_i32_arg, chunk_threshold)
                accs_nt2 = _issue_nt_mfmas(kv_list_in, kvs_packed_list_in, 2)
                _post_process_nt(accs_nt1, 1, c_i32_arg, chunk_threshold)
                accs_nt3 = _issue_nt_mfmas(kv_list_in, kvs_packed_list_in, 3)
                _post_process_nt(accs_nt2, 2, c_i32_arg, chunk_threshold)
                _post_process_nt(accs_nt3, 3, c_i32_arg, chunk_threshold)
                return []

        # === Prologue ===
        N_KV = k_tiles * N_TILES_PER_WARP
        last_c_i32 = chunk_count - fx.Int32(1)

        phys_pre = _load_phys(c0_i32)
        kv_pre, kvs_pre = _prefetch_chunk(c0_i32, phys_pre)
        phys_next_pre = _load_phys(fx.Int32(1))

        nt0_accs_init = _issue_nt_mfmas(list(kv_pre), list(kvs_pre), 0)
        nt0_init_scalars = []
        for v in nt0_accs_init:
            vv = fx.Vector(v)
            for i in range(4):
                nt0_init_scalars.append(vv[i])

        # === Main loop: chunk_count - 1 iterations ===
        N_KVS = k_tiles
        chunk_count_minus_1_i32 = chunk_count - fx.Int32(1)
        chunk_count_minus_1_idx = fx.Int64(chunk_count_minus_1_i32)
        init_args = (
            list(kv_pre) + list(kvs_pre) + list(phys_next_pre) + nt0_init_scalars
        )
        if const_expr(seed_calibration):
            init_args = init_args + [
                fx.Float32(-float("inf")),
                fx.Float32(-float("inf")),
                fx.Int32(0),
                fx.Int32(0),
            ]
        for c_idx, state in range(0, chunk_count_minus_1_idx, 1, init=init_args):
            kv_cur_list = [state[i] for i in range(N_KV)]
            kvs_cur_list = [state[N_KV + i] for i in range(N_KVS)]
            phys_next_list = [state[N_KV + N_KVS + i] for i in range(N_TILES_PER_WARP)]
            nt0_acc_base = N_KV + N_KVS + N_TILES_PER_WARP
            nt0_accs_cur = [
                fx.Vector.from_elements(
                    [state[nt0_acc_base + mi * 4 + i] for i in range(4)],
                    dtype=fx.Float32,
                )
                for mi in range(m_tiles)
            ]
            c_idx_i32 = fx.Int32(c_idx)
            c_next_i32 = c_idx_i32 + fx.Int32(1)
            c_next_next_i32 = c_next_i32 + fx.Int32(1)

            seed_acc_base = nt0_acc_base + m_tiles * 4
            seed_accs_cur = (
                [state[seed_acc_base + i] for i in range(4)]
                if const_expr(seed_calibration)
                else None
            )
            seed_accs_next = _compute_chunk(
                kv_cur_list,
                kvs_cur_list,
                c_idx_i32,
                nt0_accs_in=nt0_accs_cur,
                seed_accs=seed_accs_cur,
            )
            if const_expr(litetopk and litetopk_refresh_every > 0):
                if (c_next_i32 % fx.Int32(litetopk_refresh_every)) == 0:
                    _refresh_litetopk_threshold(scan_histogram, scan_state, scan_prefix)

            kv_next, kvs_next = _prefetch_chunk(c_next_i32, phys_next_list)

            phys_next_next_list = _load_phys(c_next_next_i32)

            nt0_accs_next = _issue_nt_mfmas(list(kv_next), list(kvs_next), 0)
            nt0_next_scalars = []
            for v in nt0_accs_next:
                vv = fx.Vector(v)
                for i in range(4):
                    nt0_next_scalars.append(vv[i])

            results = yield (
                list(kv_next)
                + list(kvs_next)
                + list(phys_next_next_list)
                + nt0_next_scalars
                + seed_accs_next
            )

        # === Epilogue: process last chunk (chunk_count - 1) ===
        kv_last_list = [results[i] for i in range(N_KV)]
        kvs_last_list = [results[N_KV + i] for i in range(N_KVS)]
        nt0_acc_base = N_KV + N_KVS + N_TILES_PER_WARP
        nt0_accs_last = [
            fx.Vector.from_elements(
                [results[nt0_acc_base + mi * 4 + i] for i in range(4)],
                dtype=fx.Float32,
            )
            for mi in range(m_tiles)
        ]
        seed_acc_base = nt0_acc_base + m_tiles * 4
        seed_accs_last = (
            [results[seed_acc_base + i] for i in range(4)]
            if const_expr(seed_calibration)
            else None
        )
        seed_accs_final = _compute_chunk(
            kv_last_list,
            kvs_last_list,
            last_c_i32,
            nt0_accs_in=nt0_accs_last,
            seed_accs=seed_accs_last,
        )
        if const_expr(litetopk):
            if const_expr(litetopk_refresh_every > 0):
                _refresh_litetopk_threshold(scan_histogram, scan_state, scan_prefix)
            gpu.barrier()
            if tid < fx.Int32(256):
                histogram_offset = row_id * fx.Int32(256) + tid
                histogram_buf[histogram_offset] = scan_histogram[tid]
            if tid == 0:
                candidate_counts_buf[row_id] = scan_state[0]
                threshold_buf[row_id] = scan_state[1]
                score_errors_buf[row_id] = scan_state[2]
        elif const_expr(seed_calibration):
            row_max, row_neg_min, finite_count, nonfinite_count = seed_accs_final

            def _wave_reduce_max_f32(value):
                value = fx.Float32(value)
                for distance in (1, 2, 4, 8, 16, 32):
                    peer_i32 = fx.Int32(
                        rocdl.ds_bpermute(
                            T.i32,
                            (fx.Int32(lane_id) ^ fx.Int32(distance)) * fx.Int32(4),
                            value.bitcast(fx.Int32),
                        )
                    )
                    value = value.maximumf(peer_i32.bitcast(fx.Float32))
                return value

            def _wave_reduce_add_i32(value):
                value = fx.Int32(value)
                for distance in (1, 2, 4, 8, 16, 32):
                    peer = fx.Int32(
                        rocdl.ds_bpermute(
                            T.i32,
                            (fx.Int32(lane_id) ^ fx.Int32(distance)) * fx.Int32(4),
                            value,
                        )
                    )
                    value = value + peer
                return value

            row_max = _wave_reduce_max_f32(row_max)
            row_neg_min = _wave_reduce_max_f32(row_neg_min)
            finite_count = _wave_reduce_add_i32(finite_count)
            nonfinite_count = _wave_reduce_add_i32(nonfinite_count)
            if lane_id == 0:
                seed_maxima[warp_id] = row_max
                seed_neg_minima[warp_id] = row_neg_min
                seed_finite_counts[warp_id] = finite_count
                seed_nonfinite_counts[warp_id] = nonfinite_count
            gpu.barrier()

            if tid == 0:
                block_max = fx.Float32(-float("inf"))
                block_neg_min = fx.Float32(-float("inf"))
                block_finite_count = fx.Int32(0)
                block_nonfinite_count = fx.Int32(0)
                for wave in range_constexpr(num_warps):
                    block_max = block_max.maximumf(seed_maxima[wave])
                    block_neg_min = block_neg_min.maximumf(seed_neg_minima[wave])
                    block_finite_count = block_finite_count + seed_finite_counts[wave]
                    block_nonfinite_count = (
                        block_nonfinite_count + seed_nonfinite_counts[wave]
                    )

                has_finite = block_finite_count > fx.Int32(0)
                safe_max = has_finite.select(block_max, fx.Float32(0.0))
                safe_min = has_finite.select(-block_neg_min, fx.Float32(0.0))
                seed_origin = -safe_max
                magnitude = safe_max.maximumf(-safe_max).maximumf(
                    safe_min.maximumf(-safe_min)
                )
                span = (safe_max - safe_min).maximumf(
                    magnitude * fx.Float32(1.0 / 256.0)
                )
                span = span.maximumf(fx.Float32(1.0e-6))
                seed_inv_delta = fx.Float32(255.0) / span
                origin_bits = seed_origin.bitcast(fx.Int32) & fx.Int32(0x7FFFFFFF)
                inv_delta_bits = seed_inv_delta.bitcast(fx.Int32) & fx.Int32(0x7FFFFFFF)
                affine_finite = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    origin_bits,
                    fx.Int32(0x7F800000),
                ) & arith.cmpi(
                    arith.CmpIPredicate.ult,
                    inv_delta_bits,
                    fx.Int32(0x7F800000),
                )
                bad_calibration = (
                    (block_nonfinite_count > fx.Int32(0))
                    | ((local_end > local_start) & ~has_finite)
                    | ~affine_finite
                )
                origin_buf[row_id] = affine_finite.select(seed_origin, fx.Float32(0.0))
                inv_delta_buf[row_id] = affine_finite.select(
                    seed_inv_delta, fx.Float32(1.0)
                )
                status_buf[row_id] = status_buf[row_id] | bad_calibration.select(
                    fx.Int32(seed_status_nonfinite), fx.Int32(0)
                )
                if const_expr(seed_emit):
                    seed_calibration_values[0] = affine_finite.select(
                        seed_origin, fx.Float32(0.0)
                    )
                    seed_calibration_values[1] = affine_finite.select(
                        seed_inv_delta, fx.Float32(1.0)
                    )

            if const_expr(seed_emit):
                if tid < fx.Int32(256):
                    seed_histogram[tid] = fx.Int32(0)
                if tid == 0:
                    required = (win_len < fx.Int32(litetopk_topk)).select(
                        win_len, fx.Int32(litetopk_topk)
                    )
                    seed_state[0] = (required > fx.Int32(0)).select(
                        fx.Int32(255), fx.Int32(0)
                    )
                    seed_state[1] = required
                gpu.barrier()

                seed_origin_shared = seed_calibration_values[0]
                seed_inv_delta_shared = seed_calibration_values[1]
                for seed_base in range(
                    fx.Int32(0), fx.Int32(8192), fx.Int32(block_threads_k)
                ):
                    seed_slot = fx.Int32(seed_base) + tid
                    seed_score = seed_scores[seed_slot]
                    seed_finite_bits = seed_score.bitcast(fx.Int32) & fx.Int32(
                        0x7FFFFFFF
                    )
                    seed_finite = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        seed_finite_bits,
                        fx.Int32(0x7F800000),
                    )
                    seed_genuine = (seed_slot < win_len) & seed_finite
                    seed_bucket_f = (
                        -seed_score - seed_origin_shared
                    ) * seed_inv_delta_shared
                    seed_bucket_f = seed_finite.select(seed_bucket_f, fx.Float32(255.0))
                    seed_bucket = fx.Int32(fx.clampf(seed_bucket_f, 0.0, 255.0))
                    if seed_genuine:
                        atomic_add_i32(
                            seed_histogram,
                            1,
                            seed_bucket,
                            "workgroup",
                        )

                gpu.barrier()
                seed_count = fx.Int32(0)
                if tid < fx.Int32(256):
                    seed_count = seed_histogram[tid]
                seed_before, _ = _block_exclusive_prefix_i32(seed_count, seed_scan)
                seed_required = seed_state[1]
                seed_crosses = (seed_before < seed_required) & (
                    seed_before + seed_count >= seed_required
                )
                if seed_crosses & (tid < fx.Int32(256)):
                    seed_state[0] = tid
                gpu.barrier()

                if tid < fx.Int32(256):
                    histogram_buf[row_id * fx.Int32(256) + tid] = seed_histogram[tid]
                if tid == 0:
                    threshold_buf[row_id] = seed_state[0]

                seed_emit_result = None
                for seed_base, seed_emit_state in range(
                    fx.Int32(0),
                    fx.Int32(8192),
                    fx.Int32(block_threads_k),
                    init=[fx.Int32(0)],
                ):
                    seed_attempted = fx.Int32(seed_emit_state[0])
                    seed_slot = fx.Int32(seed_base) + tid
                    seed_score = seed_scores[seed_slot]
                    seed_finite_bits = seed_score.bitcast(fx.Int32) & fx.Int32(
                        0x7FFFFFFF
                    )
                    seed_finite = arith.cmpi(
                        arith.CmpIPredicate.ult,
                        seed_finite_bits,
                        fx.Int32(0x7F800000),
                    )
                    seed_genuine = (seed_slot < win_len) & seed_finite
                    seed_bucket_f = (
                        -seed_score - seed_origin_shared
                    ) * seed_inv_delta_shared
                    seed_bucket_f = seed_finite.select(seed_bucket_f, fx.Float32(255.0))
                    seed_bucket = fx.Int32(fx.clampf(seed_bucket_f, 0.0, 255.0))
                    seed_keep = seed_genuine & (seed_bucket <= seed_state[0])
                    seed_exclusive, seed_total = _block_exclusive_prefix_i32(
                        seed_keep.select(fx.Int32(1), fx.Int32(0)), seed_scan
                    )
                    seed_candidate_slot = seed_attempted + seed_exclusive
                    if seed_keep & (seed_candidate_slot < merge_cap):
                        _store_candidate(
                            candidate_values_buf,
                            candidate_indices_buf,
                            seed_candidate_slot,
                            seed_score,
                            local_start + seed_slot,
                        )
                    seed_emit_result = yield [seed_attempted + seed_total]

                seed_attempted = fx.Int32(seed_emit_result)
                if tid == 0:
                    candidate_counts_buf[row_id] = seed_attempted
                    seed_select_starts_buf[row_id] = fx.Int32(0)
                    seed_select_ends_buf[row_id] = (seed_attempted < merge_cap).select(
                        seed_attempted, merge_cap
                    )
                    seed_status = status_buf[row_id]
                    seed_status = seed_status | (seed_attempted > merge_cap).select(
                        fx.Int32(seed_status_candidate_overflow), fx.Int32(0)
                    )
                    seed_status = seed_status | (seed_attempted < seed_required).select(
                        fx.Int32(seed_status_underfilled), fx.Int32(0)
                    )
                    status_buf[row_id] = seed_status

    return pa_mqa_logits_fp4_prefill_kernel, block_threads_k


# ============================================================================
# Cached compile + public host API
# ============================================================================


@cache
def compile_pa_mqa_logits_fp4_prefill(
    *,
    block_k: int = 256,
    kv_block_size: int = 64,
    num_warps: int = DEFAULT_NUM_WARPS,
    heads: int = DEFAULT_HEADS,
    head_dim: int = DEFAULT_HEAD_DIM,
    relative_output: bool = False,
):
    kfn, block_threads = build_pa_mqa_logits_fp4_prefill_module(
        block_k=block_k,
        kv_block_size=kv_block_size,
        num_warps=num_warps,
        heads=heads,
        head_dim=head_dim,
        relative_output=relative_output,
    )

    @flyc.jit
    def launch_pa_mqa_logits_fp4_prefill(
        out,
        q,
        qs,
        kv,
        kvs,
        bt,
        w,
        cta_info_,
        stride_out: fx.Int32,
        stride_bt: fx.Int32,
        block_table_rows: fx.Int32,
        block_table_capacity: fx.Int32,
        physical_page_capacity: fx.Int32,
        weight_scale: fx.Float32,
        gx: fx.Int32,
        stream: fx.Stream,
    ):
        gxi = fx.Int64(gx)
        kfn(
            out,
            q,
            qs,
            kv,
            kvs,
            bt,
            w,
            cta_info_,
            stride_out,
            stride_bt,
            block_table_rows,
            weight_scale,
            out,
            out,
            out,
            out,
            out,
            out,
            out,
            out,
            out,
            out,
            fx.Int32(0),
            fx.Int32(0),
            block_table_capacity,
            physical_page_capacity,
        ).launch(grid=(gxi,), block=(block_threads, 1, 1), stream=stream)

    return launch_pa_mqa_logits_fp4_prefill, block_threads


@cache
def compile_pa_mqa_litetopk_fp4_seed(
    *,
    block_k: int = 256,
    kv_block_size: int = 64,
    num_warps: int = DEFAULT_NUM_WARPS,
    heads: int = DEFAULT_HEADS,
    head_dim: int = DEFAULT_HEAD_DIM,
    topk: int = 512,
    status_candidate_overflow: int = 1 << 0,
    status_underfilled: int = 1 << 1,
    status_nonfinite: int = 1 << 2,
    report_page_errors: bool = False,
):
    kfn, block_threads = build_pa_mqa_logits_fp4_prefill_module(
        block_k=block_k,
        kv_block_size=kv_block_size,
        num_warps=num_warps,
        heads=heads,
        head_dim=head_dim,
        relative_output=True,
        seed_calibration=True,
        seed_emit=True,
        seed_report_page_errors=report_page_errors,
        seed_status_candidate_overflow=status_candidate_overflow,
        seed_status_underfilled=status_underfilled,
        seed_status_nonfinite=status_nonfinite,
        litetopk_topk=topk,
    )

    @flyc.jit
    def launch_pa_mqa_litetopk_fp4_seed(
        q,
        qs,
        kv,
        kvs,
        bt,
        w,
        cta_info_,
        origin,
        inv_delta,
        threshold,
        histogram,
        candidate_values,
        candidate_indices,
        candidate_counts,
        select_starts,
        select_ends,
        status,
        page_errors,
        candidate_stride: fx.Int32,
        merge_cap: fx.Int32,
        stride_bt: fx.Int32,
        block_table_rows: fx.Int32,
        block_table_capacity: fx.Int32,
        physical_page_capacity: fx.Int32,
        weight_scale: fx.Float32,
        gx: fx.Int32,
        stream: fx.Stream,
    ):
        gxi = fx.Int64(gx)
        kfn(
            select_starts,
            q,
            qs,
            kv,
            kvs,
            bt,
            w,
            cta_info_,
            fx.Int32(0),
            stride_bt,
            block_table_rows,
            weight_scale,
            origin,
            inv_delta,
            threshold,
            histogram,
            candidate_values,
            candidate_indices,
            candidate_counts,
            page_errors,
            select_ends,
            status,
            candidate_stride,
            merge_cap,
            block_table_capacity,
            physical_page_capacity,
        ).launch(grid=(gxi,), block=(block_threads, 1, 1), stream=stream)

    return launch_pa_mqa_litetopk_fp4_seed, block_threads


def flydsl_pa_mqa_litetopk_fp4_seed(
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    weights: torch.Tensor,
    cta_info: torch.Tensor,
    origin: torch.Tensor,
    inv_delta: torch.Tensor,
    threshold: torch.Tensor,
    histogram: torch.Tensor,
    candidate_values: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_counts: torch.Tensor,
    select_starts: torch.Tensor,
    select_ends: torch.Tensor,
    status: torch.Tensor,
    *,
    n_ctas: int,
    merge_cap: int,
    topk: int,
    status_candidate_overflow: int,
    status_underfilled: int,
    status_nonfinite: int,
    page_errors: torch.Tensor | None = None,
    weight_scale: float = 1.0,
    block_k: int = 256,
    kv_block_size: int = 64,
    num_warps: int = DEFAULT_NUM_WARPS,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Score and emit the complete bounded LiteTopK seed in one CTA per row."""
    rows, heads, head_dim_packed = q_fp4.shape
    if heads != DEFAULT_HEADS or head_dim_packed * 2 != DEFAULT_HEAD_DIM:
        raise ValueError("LiteTopK seed requires H=64 and D=128")
    if n_ctas != rows:
        raise ValueError("LiteTopK seed requires exactly one CTA per row")
    if cta_info.shape != (rows, CTA_INFO_WIDTH) or cta_info.dtype != torch.int32:
        raise ValueError("cta_info must be int32 [rows, 6]")
    if any(t.dtype != torch.float32 or t.shape != (rows,) for t in (origin, inv_delta)):
        raise ValueError("origin and inv_delta must be float32 [rows]")
    row_i32 = (threshold, candidate_counts, select_starts, select_ends, status)
    if any(t.dtype != torch.int32 or t.shape != (rows,) for t in row_i32):
        raise ValueError("seed row metadata must be int32 [rows]")
    if histogram.dtype != torch.int32 or histogram.shape != (rows, 256):
        raise ValueError("histogram must be int32 [rows, 256]")
    if candidate_values.dtype != torch.float32 or candidate_values.shape != (
        rows,
        merge_cap,
    ):
        raise ValueError("candidate_values shape does not match rows and merge_cap")
    if candidate_indices.dtype != torch.int32 or candidate_indices.shape != (
        rows,
        merge_cap,
    ):
        raise ValueError("candidate_indices shape does not match rows and merge_cap")
    if page_errors is not None and (
        page_errors.dtype != torch.int32 or page_errors.shape != (rows,)
    ):
        raise ValueError("page_errors must be int32 [rows]")
    tensors = (
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        cta_info,
        origin,
        inv_delta,
        histogram,
        candidate_values,
        candidate_indices,
        *row_i32,
        *((page_errors,) if page_errors is not None else ()),
    )
    if any(t.device != q_fp4.device for t in tensors):
        raise ValueError("all LiteTopK seed tensors must be on one device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("all LiteTopK seed tensors must be contiguous")
    if stream is None:
        stream = torch.cuda.current_stream(q_fp4.device)
    if torch.device(stream.device) != q_fp4.device:
        raise ValueError("stream and LiteTopK seed tensors must be on one device")

    with torch.cuda.device(q_fp4.device), torch.cuda.stream(stream):
        launcher, _ = compile_pa_mqa_litetopk_fp4_seed(
            block_k=block_k,
            kv_block_size=kv_block_size,
            num_warps=num_warps,
            heads=heads,
            head_dim=head_dim_packed * 2,
            topk=topk,
            status_candidate_overflow=status_candidate_overflow,
            status_underfilled=status_underfilled,
            status_nonfinite=status_nonfinite,
            report_page_errors=page_errors is not None,
        )
        _run_compiled(
            launcher,
            q_fp4,
            q_scale,
            kv_cache,
            kv_scale,
            block_tables,
            weights,
            cta_info,
            origin,
            inv_delta,
            threshold,
            histogram,
            candidate_values,
            candidate_indices,
            candidate_counts,
            select_starts,
            select_ends,
            status,
            page_errors if page_errors is not None else candidate_values,
            candidate_values.stride(0),
            merge_cap,
            block_tables.stride(0),
            block_tables.shape[0],
            block_tables.shape[1],
            kv_cache.shape[0],
            float(weight_scale),
            n_ctas,
            stream,
        )


@cache
def compile_pa_mqa_litetopk_fp4_prefill_scan(
    *,
    block_k: int = 256,
    kv_block_size: int = 64,
    num_warps: int = DEFAULT_NUM_WARPS,
    heads: int = DEFAULT_HEADS,
    head_dim: int = DEFAULT_HEAD_DIM,
    topk: int = 512,
    refresh_every: int = 64,
):
    kfn, block_threads = build_pa_mqa_logits_fp4_prefill_module(
        block_k=block_k,
        kv_block_size=kv_block_size,
        num_warps=num_warps,
        heads=heads,
        head_dim=head_dim,
        litetopk=True,
        litetopk_topk=topk,
        litetopk_refresh_every=refresh_every,
    )

    @flyc.jit
    def launch_pa_mqa_litetopk_fp4_prefill_scan(
        q,
        qs,
        kv,
        kvs,
        bt,
        w,
        cta_info_,
        origin,
        inv_delta,
        threshold,
        histogram,
        candidate_values,
        candidate_indices,
        candidate_counts,
        page_errors,
        score_errors,
        candidate_stride: fx.Int32,
        merge_cap: fx.Int32,
        block_table_capacity: fx.Int32,
        physical_page_capacity: fx.Int32,
        stride_bt: fx.Int32,
        block_table_rows: fx.Int32,
        weight_scale: fx.Float32,
        gx: fx.Int32,
        stream: fx.Stream,
    ):
        gxi = fx.Int64(gx)
        kfn(
            candidate_values,
            q,
            qs,
            kv,
            kvs,
            bt,
            w,
            cta_info_,
            fx.Int32(0),
            stride_bt,
            block_table_rows,
            weight_scale,
            origin,
            inv_delta,
            threshold,
            histogram,
            candidate_values,
            candidate_indices,
            candidate_counts,
            page_errors,
            score_errors,
            score_errors,
            candidate_stride,
            merge_cap,
            block_table_capacity,
            physical_page_capacity,
        ).launch(grid=(gxi,), block=(block_threads, 1, 1), stream=stream)

    return launch_pa_mqa_litetopk_fp4_prefill_scan, block_threads


def flydsl_pa_mqa_litetopk_fp4_prefill_scan(
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    weights: torch.Tensor,
    cta_info: torch.Tensor,
    origin: torch.Tensor,
    inv_delta: torch.Tensor,
    threshold: torch.Tensor,
    histogram: torch.Tensor,
    candidate_values: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_counts: torch.Tensor,
    page_errors: torch.Tensor,
    score_errors: torch.Tensor,
    *,
    n_ctas: int,
    merge_cap: int,
    topk: int = 512,
    refresh_every: int = 64,
    weight_scale: float = 1.0,
    block_k: int = 256,
    kv_block_size: int = 64,
    num_warps: int = DEFAULT_NUM_WARPS,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Run FP4 paged scoring with a conservative, monotonically-tightened gate."""
    rows, heads, head_dim_packed = q_fp4.shape
    if heads != DEFAULT_HEADS or head_dim_packed * 2 != DEFAULT_HEAD_DIM:
        raise ValueError("LiteTopK scan requires H=64 and D=128")
    if n_ctas != rows:
        raise ValueError("LiteTopK scan requires exactly one CTA per query row")
    if cta_info.shape != (rows, CTA_INFO_WIDTH) or cta_info.dtype != torch.int32:
        raise ValueError("cta_info must be int32 [rows, 6]")
    if histogram.shape != (rows, 256) or histogram.dtype != torch.int32:
        raise ValueError("histogram must be int32 [rows, 256]")
    if topk not in FP4_LITETOPK_SUPPORTED_TOPKS:
        raise ValueError(
            "the FP4 LiteTopK scan requires topk in "
            f"{FP4_LITETOPK_SUPPORTED_TOPKS}, got {topk}"
        )
    if refresh_every * block_k != 16_384:
        raise ValueError("the production LiteTopK scan requires a 16384-token refresh")
    if candidate_values.shape != (rows, merge_cap):
        raise ValueError("candidate_values shape does not match rows and merge_cap")
    if candidate_indices.shape != (rows, merge_cap):
        raise ValueError("candidate_indices shape does not match rows and merge_cap")
    if candidate_values.dtype != torch.float32:
        raise ValueError("candidate_values must be float32")
    if candidate_indices.dtype != torch.int32:
        raise ValueError("candidate_indices must be int32")
    row_vectors = (
        origin,
        inv_delta,
        threshold,
        candidate_counts,
        page_errors,
        score_errors,
    )
    if any(t.shape != (rows,) for t in row_vectors):
        raise ValueError("LiteTopK row metadata must have shape [rows]")
    if origin.dtype != torch.float32 or inv_delta.dtype != torch.float32:
        raise ValueError("origin and inv_delta must be float32")
    if threshold.dtype != torch.int32 or candidate_counts.dtype != torch.int32:
        raise ValueError("threshold and candidate_counts must be int32")
    tensors = (
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        cta_info,
        histogram,
        candidate_values,
        candidate_indices,
        *row_vectors,
    )
    if any(t.device != q_fp4.device for t in tensors):
        raise ValueError("all LiteTopK scan tensors must be on one device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("all LiteTopK scan tensors must be contiguous")
    if stream is None:
        stream = torch.cuda.current_stream(q_fp4.device)
    if torch.device(stream.device) != q_fp4.device:
        raise ValueError("stream and LiteTopK tensors must be on one device")

    with torch.cuda.device(q_fp4.device), torch.cuda.stream(stream):
        launcher, _ = compile_pa_mqa_litetopk_fp4_prefill_scan(
            block_k=block_k,
            kv_block_size=kv_block_size,
            num_warps=num_warps,
            heads=heads,
            head_dim=head_dim_packed * 2,
            topk=topk,
            refresh_every=refresh_every,
        )
        _run_compiled(
            launcher,
            q_fp4,
            q_scale,
            kv_cache,
            kv_scale,
            block_tables,
            weights,
            cta_info,
            origin,
            inv_delta,
            threshold,
            histogram,
            candidate_values,
            candidate_indices,
            candidate_counts,
            page_errors,
            score_errors,
            candidate_values.stride(0),
            merge_cap,
            block_tables.shape[1],
            kv_cache.shape[0],
            block_tables.stride(0),
            block_tables.shape[0],
            float(weight_scale),
            n_ctas,
            stream,
        )


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
    relative_output: bool = False,
) -> torch.Tensor:
    """Ragged-prefill FP4 paged MQA logits (gfx950)."""
    total_tokens, heads, head_dim_packed = q_fp4.shape
    head_dim = head_dim_packed * 2
    if (cta_info is None) != (n_ctas is None):
        raise ValueError("Pass both cta_info and n_ctas, or neither.")
    if stream is None:
        stream = torch.cuda.current_stream(q_fp4.device)
    if torch.device(stream.device) != q_fp4.device:
        raise ValueError("stream and FP4 prefill tensors must be on one device")

    with torch.cuda.device(q_fp4.device), torch.cuda.stream(stream):
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
            block_k=block_k,
            kv_block_size=kv_block_size,
            num_warps=num_warps,
            heads=heads,
            head_dim=head_dim,
            relative_output=relative_output,
        )

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
            block_tables.stride(0),
            block_tables.shape[0],
            block_tables.shape[1],
            kv_cache.shape[0],
            float(weight_scale),
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
