# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx942 (CDNA3) port of the fp4 ragged-prefill pa_mqa_logits kernel.

Same port recipe as ``pa_mqa_logits_fp4_gfx942`` (decode): gfx950 runs one
scaled fp4 MFMA (``MFMA_Scale(16,16,128)``, e2m1 operands, in-instruction e8m0
scales) per 16x16x128 step; gfx942 replaces it with 4x dense fp8
``MFMA(16,16,32)`` calls, decodes packed e2m1 to e4m3-FNUZ codes in pure
integer ALU (``v_perm`` is unreliable on this ROCm stack), and applies the
per-32-chunk e8m0 scales as exact power-of-two f32 multipliers on the fp32 D
partials. The accumulator/epilogue layout matches gfx950, so the weight
indexing, butterfly reduce and windowed stores are unchanged.

Scale addressing is bit-faithful to the gfx950 layouts:
  q_scale: [total_tokens, K_TILES, 4, 16, qs_pad] — byte [..][kk][j][mi] is
           the e8m0 of head (mi*16 + j) at 32-chunk kk (host-preshuffled).
  kv_scale: [num_blocks, K_TILES, 4, kv_block_size] — byte [..][kk][n*4+nt]
           is the e8m0 of token (nt*16 + n) at chunk kk.

The decode helpers mirror ``pa_mqa_logits_fp4_gfx942`` (duplicated, not
imported, so each slice stays cherry-pickable).
"""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import triton
import triton.language as tl
from flydsl.expr import gpu, rocdl
from flydsl.expr.primitive import range_constexpr
from flydsl.expr.typing import Int32, T

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

# One dense fp8 MFMA covers 32 fp4 elements (one 32-chunk) per K-step; a
# gfx950 fp4 k_tile (128 fp4) is 4 K-steps.
KK_PER_KTILE = 4

# cta_info packed fields per CTA.
CTA_INFO_WIDTH = 6


# ---------------------------------------------------------------------------
# fp4 e2m1 (packed) -> e4m3 FNUZ (byte-per-code) decode via v_perm_b32.
# Mirrors pa_mqa_logits_fp4_gfx942 (kept self-contained for cherry-picking).
# ---------------------------------------------------------------------------


def _decode_fp4_dword(d):
    """4 packed e2m1 bytes -> (p01, p23): two dwords of FNUZ codes, K ascending.

    v_perm_b32-based: 6 perms + 11 integer ops per dword (the ALU-only version
    was ~77 ops). Window order: perm_b32(src_hi, src_lo, sel) -- the first
    operand is the MSB half of the 8-byte permute window (CDNA3 BYTE_PERMUTE
    spec). FNUZ truth: x in 0..7 -> 0x00,0x38,0x40,0x44,0x48,0x4C,0x50,0x54,
    then |0x80 for signed codes with x != 0 (negative zero stays 0x00: FNUZ
    0x80 is a NaN).
    """
    even = d & fx.Int32(0x07070707)
    odd = d.shrui(fx.Int32(4)) & fx.Int32(0x07070707)
    mag_e = fx.Int32(rocdl.perm_b32(fx.Int32(0x54504C48), fx.Int32(0x44403800), even))
    mag_o = fx.Int32(rocdl.perm_b32(fx.Int32(0x54504C48), fx.Int32(0x44403800), odd))
    nz_e = fx.Int32(rocdl.perm_b32(fx.Int32(0xFFFFFFFF), fx.Int32(0xFFFFFF00), even))
    nz_o = fx.Int32(rocdl.perm_b32(fx.Int32(0xFFFFFFFF), fx.Int32(0xFFFFFF00), odd))
    dec_e = mag_e | (((d & fx.Int32(0x08080808)) << fx.Int32(4)) & nz_e)
    dec_o = mag_o | ((d & fx.Int32(0x80808080)) & nz_o)
    p01 = fx.Int32(rocdl.perm_b32(dec_o, dec_e, fx.Int32(0x05010400)))
    p23 = fx.Int32(rocdl.perm_b32(dec_o, dec_e, fx.Int32(0x07030602)))
    return p01, p23


def _frag_i64_from_pp(p01, p23):
    """One lane's 8-byte fp8 fragment (K ascending) as a rank-1 register tensor."""
    raw = fx.Int64(fx.Uint32(p01)) | (fx.Int64(fx.Uint32(p23)) << fx.Int32(32))
    frag = fx.make_rmem_tensor(1, fx.Int64)
    frag.store(fx.Vector.from_elements([fx.Int64(raw)]))
    return frag


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
    """
    device = local_ends.device
    P = parallel_unit_num
    T = local_ends.shape[0]  # fixed total_tokens (rows)

    assert P >= T, (
        f"compute_prefill_schedule: parallel_unit_num={P} < rows={T} would "
        f"silently drop rows past slot {P} (logits stay at the caller's "
        f"pre-fill -> wrong top-k). Pass parallel_unit_num >= number of rows."
    )

    assert (
        row_to_batch.dtype == local_starts.dtype == local_ends.dtype == torch.int32
    ), (
        f"compute_prefill_schedule: row_to_batch/local_starts/local_ends must be "
        f"int32, got {row_to_batch.dtype}/{local_starts.dtype}/{local_ends.dtype}."
    )

    s_max = max(1, (max_seq_len + block_k - 1) // block_k)
    plan = _row_plan(local_ends, block_k, P, s_max)

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
    """Everything the emit kernel needs about the rows, from either producer."""

    incl: torch.Tensor  # [T] inclusive prefix sum of per-row CTA counts
    excl: torch.Tensor  # [T] exclusive prefix sum
    chunks: torch.Tensor  # [T] ceil(local_end / block_k), 0 for an empty row
    safe: torch.Tensor  # [1] chunk-splits merged into one CTA
    total_splits: torch.Tensor  # [1] number of valid (row, split) slots


_ROW_PLAN_BLOCK_FLOOR = 4096
_ROW_PLAN_MAX_ROWS = 16384

_I32_PER_16B = 4


def _row_plan(le, block_k, P, s_max) -> _RowPlan:
    T = le.shape[0]
    if T > _ROW_PLAN_MAX_ROWS:
        return _row_plan_torch(le, block_k, P, s_max)
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
    chunks_per_row = torch.clamp((le + (block_k - 1)) // block_k, min=0)  # [T]

    s_cand = torch.arange(1, s_max + 1, device=le.device, dtype=torch.int32)  # [s_max]
    ctas_per_r_s = (chunks_per_row[None, :] + (s_cand[:, None] - 1)) // s_cand[
        :, None
    ]  # [s_max, T]
    total_ctas_s = ctas_per_r_s.sum(dim=1)  # [s_max]
    feasible = total_ctas_s <= P  # [s_max] bool, monotonic False..True
    max_chunks = torch.clamp(chunks_per_row.max(), min=1).to(torch.int32)
    first_feasible_s = torch.clamp((~feasible).to(torch.int32).sum() + 1, max=s_max)
    safe = torch.where(feasible.any(), first_feasible_s, max_chunks).to(torch.int32)

    ctas_r = (chunks_per_row + (safe - 1)) // safe  # [T]
    incl = torch.cumsum(ctas_r, dim=0, dtype=torch.int32)  # [T] inclusive prefix sum
    return _RowPlan(
        incl=incl,
        excl=incl - ctas_r,  # exclusive prefix sum
        chunks=chunks_per_row.to(torch.int32),
        safe=safe.reshape(1).to(torch.int32),
        total_splits=incl[-1].reshape(1).to(torch.int32),
    )


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
    """Single-block row plan: chunk counts, split factor, and its prefix sums."""
    t = tl.arange(0, BLOCK_T)
    mask = t < T
    le = tl.load(le_ptr + t, mask=mask, other=0)
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


def build_pa_mqa_logits_fp4_prefill_gfx942_module(
    block_k=256,
    kv_block_size=64,
    max_blocks_per_seq=256,
    num_warps=DEFAULT_NUM_WARPS,
    heads=DEFAULT_HEADS,
    head_dim=DEFAULT_HEAD_DIM,
    relu=True,
    kv_page_stride=None,
    kv_scale_page_stride=None,
    kv_scale_offset=0,
):
    """Build the ragged-prefill FP4 MQA logits kernel (gfx942)."""
    block_threads_k = num_warps * WARP_SIZE
    m_tiles = heads // MFMA_M
    k_tiles = head_dim // 128  # outer K-loop iters (128 fp4 each = 4 K-steps)
    assert (
        head_dim % 128 == 0
    ), f"head_dim must be a multiple of 128 (fp4 k_tile), got {head_dim}"
    assert heads % MFMA_M == 0, f"heads must be a multiple of {MFMA_M}, got {heads}"
    assert m_tiles <= 8, f"m_tiles={m_tiles} > 8 not supported. Use heads <= 128."

    N_TILES = block_k // MFMA_N
    assert (
        N_TILES % num_warps == 0
    ), f"block_k={block_k} -> N_TILES={N_TILES} must be multiple of num_warps={num_warps}"
    N_TILES_PER_WARP = N_TILES // num_warps

    assert (
        kv_block_size % MFMA_N == 0
    ), f"kv_block_size={kv_block_size} must be a multiple of MFMA_N={MFMA_N}"
    assert (
        block_k % kv_block_size == 0
    ), f"block_k={block_k} must be a multiple of kv_block_size={kv_block_size}"
    assert (
        kv_block_size == 64
    ), "gfx942 packed KV loads currently require kv_block_size=64"
    TILES_PER_BLOCK = kv_block_size // MFMA_N
    N_PHYS = (N_TILES_PER_WARP + TILES_PER_BLOCK - 1) // TILES_PER_BLOCK
    assert N_TILES_PER_WARP == 4 and N_PHYS == 1, (
        "gfx942 packed KV loads require four 16-token tiles per wave within "
        "one physical page; expected block_k == 64 * num_warps and "
        "kv_block_size >= 64"
    )

    # block_tables row stride (i32 elements).
    _stride_bt = max_blocks_per_seq

    # KV preshuffle layout: [block_id, K_TILES, K_chunk=4, kv_block_size, 16] uint8.
    _kv_chunk_bytes = 16
    _stride_kv_ktile = 4 * kv_block_size * _kv_chunk_bytes
    _stride_kv_block = k_tiles * _stride_kv_ktile
    # KV_scale: [block_id, K_TILES, K_chunks=4, kv_block_size]
    _stride_kvs_ktile = 4 * kv_block_size
    _stride_kvs_block = k_tiles * _stride_kvs_ktile
    if kv_page_stride is not None:
        _stride_kv_block = kv_page_stride
    if kv_scale_page_stride is not None:
        _stride_kvs_block = kv_scale_page_stride

    _kb_is_pow2 = kv_block_size & (kv_block_size - 1) == 0
    _kb_log2 = kv_block_size.bit_length() - 1
    _kb_mask = kv_block_size - 1

    def _floordiv_kb(x):
        return (x >> fx.Int32(_kb_log2)) if _kb_is_pow2 else (x // kv_block_size)

    def _mod_kb(x):
        return (x & fx.Int32(_kb_mask)) if _kb_is_pow2 else (x % kv_block_size)

    QS_DW = (m_tiles + 3) // 4
    qs_pad = QS_DW * 4

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
    def pa_mqa_logits_fp4_prefill_gfx942_kernel(
        out_logits_ptr: fx.Tensor,
        q_ptr: fx.Tensor,
        q_scale_ptr: fx.Tensor,
        kv_cache_ptr: fx.Tensor,
        kv_scale_ptr: fx.Tensor,
        kv_indices_ptr: fx.Tensor,
        weights_ptr: fx.Tensor,
        cta_info_ptr: fx.Tensor,  # [n_ctas, 6] i32
        stride_out_row: Int32,
        weight_scale: fx.Float32,
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
            return fx.Int32(fx.rocdl.readfirstlane(T.i32, v.ir_value()))

        local_start = _uniform(cta_info_bt[(fx.Int32(1), fx.Int32(0))])
        local_end = _uniform(cta_info_bt[(fx.Int32(1), fx.Int32(1))])

        kv_bt = _i32_buffer(kv_cache_ptr, width=1)
        kvs_bt = _i32_buffer(kv_scale_ptr, width=1)
        bt_bt = _i32_buffer(kv_indices_ptr, width=1)
        q_bt = _i32_buffer(q_ptr, width=1)
        qs_bt = _i32_buffer(q_scale_ptr, width=1)

        ZERO_F = fx.Float32(0.0)
        c0_i32 = fx.Int32(0)

        row_id = cta_info_vec[0]
        batch_id = cta_info_vec[1]
        chunk_start = cta_info_vec[2]
        chunk_count = cta_info_vec[3]

        # A V# spanning exactly [local_start, local_end) of this row: num_records
        # then IS the window test, in hardware. Lanes 16..63 hold redundant
        # copies of the butterfly result and must not write; a large CONSTANT
        # added to their offset puts them past num_records.
        win_len = local_end - local_start
        _row_elems = fx.Int64(row_id) * fx.Int64(stride_out_row) + fx.Int64(local_start)
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

        # Dense fp8 16x16x32 MMA (FNUZ); 4 K-steps replace one gfx950 fp4 step.
        mfma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.Float8E4M3FNUZ))

        def _e8m0_to_f32(b):
            v = fx.Vector.from_elements(
                [(b & fx.Int32(0xFF)) << fx.Int32(23)], dtype=fx.Int32
            ).bitcast(fx.Float32)
            return v[0]

        # ---- A-fragments: per (k_tile, mi, kk), one packed dword -> i64 frag ----
        # Q: [total_tokens, H, D/2] uint8. Byte offset =
        # row_id*(H*D/2) + q_row*(D/2) + k_tile*64 + kk*16 + (l//16)*4.
        q_base = row_id * fx.Int32(heads * (head_dim // 2))
        q_a_ops = []
        for k_tile in range_constexpr(k_tiles):
            q_a_ops_kt = []
            for mi_idx in range_constexpr(m_tiles):
                q_row = fx.Int32(mi_idx * MFMA_M) + lane_mod_16
                q_row_base = q_base + q_row * fx.Int32(head_dim // 2)
                frags = []
                for kk in range_constexpr(KK_PER_KTILE):
                    off = (
                        q_row_base
                        + fx.Int32(k_tile * 64 + kk * 16)
                        + lane_div_16 * fx.Int32(4)
                    )
                    d = fx.Int32(q_bt[off // fx.Int32(4)])
                    p01, p23 = _decode_fp4_dword(d)
                    frags.append(_frag_i64_from_pp(p01, p23))
                q_a_ops_kt.append(frags)
            q_a_ops.append(q_a_ops_kt)

        # ---- A-scales: e8m0 byte [row_id][k_tile][kk][lane_div_16*4+t][mi] ----
        # Q scale: host-preshuffled [total_tokens, K_TILES, 4, 16, qs_pad].
        qs_base = row_id * fx.Int32(k_tiles * 64 * qs_pad)
        sa_f32_tab = []
        for k_tile in range_constexpr(k_tiles):
            sa_kt = []
            for kk in range_constexpr(KK_PER_KTILE):
                sa_kk = []
                for t in range_constexpr(4):
                    row = lane_div_16 * fx.Int32(4) + fx.Int32(t)
                    off = (
                        qs_base
                        + fx.Int32(k_tile * 64 * qs_pad)
                        + fx.Int32(kk * 16 * qs_pad)
                        + row * fx.Int32(qs_pad)
                    )
                    sa_t = []
                    for mi_idx in range_constexpr(m_tiles):
                        dw = fx.Int32(
                            qs_bt[(off // fx.Int32(4)) + fx.Int32(mi_idx // 4)]
                        )
                        sa_byte = (dw >> fx.Int32(8 * (mi_idx % 4))) & fx.Int32(0xFF)
                        sa_t.append(_e8m0_to_f32(sa_byte))
                    sa_kk.append(sa_t)
                sa_kt.append(sa_kk)
            sa_f32_tab.append(sa_kt)

        # Weights (hoisted): [total_tokens, H] bf16, addressed by row_id.
        # Loaded as bf16 then widened to f32; `weight_scale` folded here once.
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
        # NOTE: flydsl's AST rewriter lowers in-body `if` statements to runtime
        # scf.if dispatches (they are NOT trace-time python conditionals), so
        # the load helpers stay monomorphic; the general address math below is
        # used for all configs.

        def _load_phys(c_i32_arg):
            ni_base = warp_id * fx.Int32(N_TILES_PER_WARP)
            token_local_base = (
                (chunk_start + c_i32_arg) * fx.Int32(block_k)
                + ni_base * fx.Int32(MFMA_N)
                + lane_mod_16
            )
            bi_base = _floordiv_kb(token_local_base)
            phys_vec = bt_bt[batch_id * _stride_bt + bi_base]
            return _phys_to_list(phys_vec)

        def _prefetch_chunk(c_i32_arg, phys_list):
            assert N_TILES_PER_WARP == 4, "packed kvs assumes NTPW=4"
            assert N_PHYS == 1, "packed kvs assumes N_PHYS=1 (NTPW nts share one phys)"

            kv_list = []  # per (nt, k_tile, kk): packed-fp4 dword for this lane
            kvs_list = []  # per (k_tile, kk): packed sb dword (4 nt bytes)

            phys_shared = phys_list[0]

            # ---- KVS packed load: 1 dword per (k_tile, kk) covering 4 nts ----
            # Byte layout: [phys][k_tile][kk][n*4 + nt]; 4 consecutive nt bytes.
            for k_tile in range_constexpr(k_tiles):
                for kk in range_constexpr(KK_PER_KTILE):
                    kvs_packed_off_bytes = (
                        fx.Int64(phys_shared) * fx.Int64(_stride_kvs_block)
                        + fx.Int64(kv_scale_offset)
                        + fx.Int64(k_tile * _stride_kvs_ktile)
                        + fx.Int64(kk * kv_block_size)
                        + fx.Int64(lane_mod_16) * fx.Int64(N_TILES_PER_WARP)
                    )
                    kvs_list.append(kvs_bt[kvs_packed_off_bytes // fx.Int64(4)])

            # ---- KV loads: contiguous + cross-lane permute. Lane l loads
            # flat dword (token-in-tile = l//4, group = l%4) of MFMA tile
            # nt's 16-token span, so the 64 lanes of one load instruction
            # cover 256B densely (the old mapping strided 16 dwords at 16B
            # token stride per lane-group = 4x sector over-request). The
            # B-fragment owner (token l%16, group l//16) then pulls its
            # dword from lane (l%16)*4 + (l//16) via ds_bpermute — same
            # dword values in the same order as the old mapping.
            token_in_block0 = _mod_kb(
                (chunk_start + c_i32_arg) * fx.Int32(block_k)
                + warp_id * fx.Int32(N_TILES_PER_WARP) * fx.Int32(MFMA_N)
            )
            perm_src_byte = (lane_mod_16 * fx.Int32(4) + lane_div_16) * fx.Int32(4)
            for nt in range_constexpr(N_TILES_PER_WARP):
                for k_tile in range_constexpr(k_tiles):
                    for kk in range_constexpr(KK_PER_KTILE):
                        token_in_block = _mod_kb(
                            token_in_block0
                            + fx.Int32(nt * MFMA_N)
                            + (lane_id // fx.Int32(4))
                        )
                        kv_off_bytes = (
                            fx.Int64(phys_shared) * fx.Int64(_stride_kv_block)
                            + fx.Int64(k_tile * _stride_kv_ktile)
                            + fx.Int64(kk * kv_block_size * _kv_chunk_bytes)
                            + fx.Int64(token_in_block) * fx.Int64(_kv_chunk_bytes)
                            + fx.Int64(lane_id % fx.Int32(4)) * fx.Int64(4)
                        )
                        raw_c = kv_bt[kv_off_bytes // fx.Int64(4)]
                        kv_list.append(
                            fx.Int32(rocdl.ds_bpermute(T.i32, perm_src_byte, raw_c))
                        )

            return kv_list, kvs_list

        def _frag_from_i64(v):
            """Rebuild the 1x i64 rmem MFMA fragment from a carried i64 value."""
            frag = fx.make_rmem_tensor(1, fx.Int64)
            frag.store(fx.Vector.from_elements([v], dtype=fx.Int64))
            return frag

        def _decode_frags(kv_list_in):
            """Decode N_KV packed-fp4 dwords into N_KV i64 e4m3-FNUZ fragments.

            Decode-in-prefetch: runs while the PREVIOUS chunk's MFMAs retire,
            taking the e2m1 decode off the per-chunk MFMA+scale critical path.
            Fragment bits identical to the inline decode.
            """
            frags = []
            for i in range_constexpr(N_KV):
                p01, p23 = _decode_fp4_dword(fx.Int32(kv_list_in[i]))
                frags.append(
                    fx.Int64(fx.Uint32(p01))
                    | (fx.Int64(fx.Uint32(p23)) << fx.Int32(32))
                )
            return frags

        def _issue_nt_mfmas_frags(frags_in, kvs_list_in, nt):
            acc_s = [[ZERO_F for _ in range(4)] for _ in range(m_tiles)]
            for k_tile in range_constexpr(k_tiles):
                for kk in range_constexpr(KK_PER_KTILE):
                    b_frag = _frag_from_i64(
                        frags_in[((nt * k_tiles) + k_tile) * KK_PER_KTILE + kk]
                    )
                    sb_packed = fx.Int32(kvs_list_in[k_tile * KK_PER_KTILE + kk])
                    sb_byte = (sb_packed >> fx.Int32(8 * nt)) & fx.Int32(0xFF)
                    sb_f32 = _e8m0_to_f32(sb_byte)
                    for mi_idx in range_constexpr(m_tiles):
                        c_frag = fx.make_rmem_tensor(4, fx.Float32)
                        c_frag.store(fx.Vector.filled(4, 0.0, fx.Float32))
                        fx.gemm(
                            mfma_atom,
                            c_frag,
                            q_a_ops[k_tile][mi_idx][kk],
                            b_frag,
                            c_frag,
                        )
                        partial = c_frag.load()
                        # e8m0 scale product is a power of two: the multiply is
                        # exact, so fma is bit-identical to the two-step
                        # mul-then-add while saving one VALU op per element.
                        for t in range_constexpr(4):
                            acc_s[mi_idx][t] = fx.fma(
                                partial[t],
                                sa_f32_tab[k_tile][kk][t][mi_idx] * sb_f32,
                                acc_s[mi_idx][t],
                            )
            return [
                fx.Vector.from_elements(acc_s[mi], dtype=fx.Float32)
                for mi in range(m_tiles)
            ]

        def _post_process_nt(accs, nt, c_i32_arg):
            """relu + per-head weight + per-thread sum + bperm + windowed store."""
            zero = fx.full(4, 0.0, fx.Float32)
            ni_warp = warp_id * fx.Int32(N_TILES_PER_WARP) + fx.Int32(nt)
            token_base = (chunk_start + c_i32_arg) * fx.Int32(
                block_k
            ) + ni_warp * fx.Int32(MFMA_N)

            thread_sum = ZERO_F
            for mi_idx in range_constexpr(m_tiles):
                relu_v = (
                    fx.Vector(accs[mi_idx]).maximumf(zero)
                    if relu
                    else fx.Vector(accs[mi_idx])
                )
                w_v = fx.Vector(w_per_lane[mi_idx])
                for elem in [0, 1, 2, 3]:
                    thread_sum = fx.fma(relu_v[elem], w_v[elem], thread_sum)

            lane_i32 = fx.Int32(lane_id)

            def _bperm_xor_add(val, sh):
                peer_byte = (lane_i32 ^ fx.Int32(sh)) * 4
                peer_i32 = fx.Int32(
                    rocdl.ds_bpermute(T.i32, peer_byte, val.bitcast(fx.Int32))
                )
                return val + peer_i32.bitcast(fx.Float32)

            thread_sum = _bperm_xor_add(thread_sum, 16)
            thread_sum = _bperm_xor_add(thread_sum, 32)
            # `weight_scale` already folded into `w_per_lane` (hoisted, once/wave).

            # Window and writer-lane guards are both in the V#; nothing is
            # tested here. Cells outside stay at the caller's -inf pre-fill.
            r_out = fx.memref_alloca(out_reg_ty, out_reg_lay)
            fx.memref_store_vec(
                fx.Vector.from_elements([thread_sum], dtype=fx.Float32), r_out
            )
            fx.copy(
                out_atom,
                r_out,
                fx.slice(out_win, (token_base - local_start + out_lane_off, None)),
            )

        def _compute_chunk_frags(frags_in, kvs_list_in, c_i32_arg):
            """Process chunk c from pre-decoded fragments."""
            assert (
                N_TILES_PER_WARP == 4
            ), "pipelined-nt structure currently hardcoded for NTPW=4"

            accs_nt0 = _issue_nt_mfmas_frags(frags_in, kvs_list_in, 0)

            accs_nt1 = _issue_nt_mfmas_frags(frags_in, kvs_list_in, 1)
            _post_process_nt(accs_nt0, 0, c_i32_arg)

            accs_nt2 = _issue_nt_mfmas_frags(frags_in, kvs_list_in, 2)
            _post_process_nt(accs_nt1, 1, c_i32_arg)

            accs_nt3 = _issue_nt_mfmas_frags(frags_in, kvs_list_in, 3)
            _post_process_nt(accs_nt2, 2, c_i32_arg)

            _post_process_nt(accs_nt3, 3, c_i32_arg)

        # === Prologue ===
        N_KV = k_tiles * N_TILES_PER_WARP * KK_PER_KTILE
        N_KVS = k_tiles * KK_PER_KTILE
        last_c_i32 = chunk_count - fx.Int32(1)

        phys_pre = _load_phys(c0_i32)
        kv_pre, kvs_pre = _prefetch_chunk(c0_i32, phys_pre)
        frags_pre = _decode_frags(kv_pre)
        phys_next_pre = _load_phys(fx.min(fx.Int32(1), last_c_i32))

        # === Main loop: chunk_count - 1 iterations ===
        # Loop state: pre-decoded B-fragments + packed scales for the CURRENT
        # chunk (decode-in-prefetch was done by the previous iteration, off the
        # MFMA critical path), plus the phys row for chunk c+1.
        chunk_count_minus_1_i32 = chunk_count - fx.Int32(1)
        chunk_count_minus_1_idx = fx.Int64(chunk_count_minus_1_i32)
        init_args = list(frags_pre) + list(kvs_pre) + list(phys_next_pre)
        for c_idx, state in range(0, chunk_count_minus_1_idx, 1, init=init_args):
            frags_cur = [state[i] for i in range(N_KV)]
            kvs_cur_list = [state[N_KV + i] for i in range(N_KVS)]
            phys_next_list = [state[N_KV + N_KVS + i] for i in range(N_TILES_PER_WARP)]
            c_idx_i32 = fx.Int32(c_idx)
            c_next_i32 = c_idx_i32 + 1
            c_next_next_i32 = fx.min(c_next_i32 + 1, last_c_i32)

            # Issue KV loads for chunk c+1 first so their latency overlaps the
            # current chunk's MFMA chain.
            kv_next, kvs_next = _prefetch_chunk(c_next_i32, phys_next_list)

            # Issue phys load for chunk c+2.
            phys_next_next_list = _load_phys(c_next_next_i32)

            # Compute current chunk from pre-decoded fragments.
            _compute_chunk_frags(frags_cur, kvs_cur_list, c_idx_i32)

            # Decode-in-prefetch: decode chunk c+1's B-fragments now — while
            # this chunk's MFMAs retire — instead of inside chunk c+1's
            # compute. Fragment bits identical to the inline decode.
            frags_next = _decode_frags(kv_next)

            results = yield (
                list(frags_next) + list(kvs_next) + list(phys_next_next_list)
            )

        # === Epilogue: process last chunk (chunk_count - 1) ===
        frags_last = [results[i] for i in range(N_KV)]
        kvs_last_list = [results[N_KV + i] for i in range(N_KVS)]
        _compute_chunk_frags(frags_last, kvs_last_list, last_c_i32)

    return pa_mqa_logits_fp4_prefill_gfx942_kernel, block_threads_k


# ============================================================================
# Cached compile + public host API
# ============================================================================


@lru_cache(maxsize=32)
def compile_pa_mqa_logits_fp4_prefill_gfx942(
    *,
    block_k: int = 256,
    kv_block_size: int = 64,
    max_blocks_per_seq: int = 256,
    num_warps: int = DEFAULT_NUM_WARPS,
    heads: int = DEFAULT_HEADS,
    head_dim: int = DEFAULT_HEAD_DIM,
    relu: bool = True,
    kv_page_stride: int | None = None,
    kv_scale_page_stride: int | None = None,
    kv_scale_offset: int = 0,
):
    kfn, block_threads = build_pa_mqa_logits_fp4_prefill_gfx942_module(
        block_k=block_k,
        kv_block_size=kv_block_size,
        max_blocks_per_seq=max_blocks_per_seq,
        num_warps=num_warps,
        heads=heads,
        head_dim=head_dim,
        relu=relu,
        kv_page_stride=kv_page_stride,
        kv_scale_page_stride=kv_scale_page_stride,
        kv_scale_offset=kv_scale_offset,
    )

    @flyc.jit
    def launch_pa_mqa_logits_fp4_prefill_gfx942(
        out,
        q,
        qs,
        kv,
        kvs,
        bt,
        w,
        cta_info_,
        stride_out: fx.Int32,
        weight_scale: fx.Float32,
        gx: fx.Int32,
        stream: fx.Stream,
    ):
        gxi = fx.Int64(gx)
        kfn(out, q, qs, kv, kvs, bt, w, cta_info_, stride_out, weight_scale).launch(
            grid=(gxi,), block=(block_threads, 1, 1), stream=stream
        )

    return launch_pa_mqa_logits_fp4_prefill_gfx942, block_threads


def flydsl_pa_mqa_logits_fp4_prefill_gfx942(
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
    kv_page_stride: int | None = None,
    kv_scale_page_stride: int | None = None,
    kv_scale_offset: int = 0,
    relu: bool = True,
    block_k: int = 256,
    kv_block_size: int = 64,
    num_warps: int = DEFAULT_NUM_WARPS,
    parallel_unit_num: int = 512,
    out: torch.Tensor | None = None,
    cta_info: torch.Tensor | None = None,
    n_ctas: int | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Ragged-prefill FP4 paged MQA logits (gfx942).

    ``relu`` and the ``kv_*page_stride`` / ``kv_scale_offset`` layout arguments
    are as in ``flydsl_pa_mqa_logits_fp4_gfx942``.
    """
    total_tokens, heads, head_dim_packed = q_fp4.shape
    head_dim = head_dim_packed * 2
    max_blocks_per_seq = block_tables.shape[1]
    if total_tokens == 0:
        if out is None:
            return torch.empty(
                (0, max_seq_len), dtype=torch.float32, device=q_fp4.device
            )
        return out

    if (cta_info is None) != (n_ctas is None):
        raise ValueError("Pass both cta_info and n_ctas, or neither.")
    schedule_internal = cta_info is None
    if (
        schedule_internal
        and stream is not None
        and stream != torch.cuda.current_stream()
    ):
        raise ValueError(
            "a non-current stream requires caller-provided cta_info and n_ctas; "
            "build the schedule on that stream before launch"
        )
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

    launcher, _ = compile_pa_mqa_logits_fp4_prefill_gfx942(
        block_k=block_k,
        kv_block_size=kv_block_size,
        max_blocks_per_seq=max_blocks_per_seq,
        num_warps=num_warps,
        heads=heads,
        head_dim=head_dim,
        relu=relu,
        kv_page_stride=kv_page_stride,
        kv_scale_page_stride=kv_scale_page_stride,
        kv_scale_offset=kv_scale_offset,
    )

    if stream is None:
        stream = torch.cuda.current_stream()

    launcher(
        out,
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        cta_info,
        out.stride(0),
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


def flydsl_pa_mqa_logits_fp4_varqlen_gfx942(
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
    kv_page_stride: int | None = None,
    kv_scale_page_stride: int | None = None,
    kv_scale_offset: int = 0,
    relu: bool = True,
    block_k: int = 256,
    kv_block_size: int = 64,
    num_warps: int = DEFAULT_NUM_WARPS,
    parallel_unit_num: int | None = None,
    out: torch.Tensor | None = None,
    cta_info: torch.Tensor | None = None,
    n_ctas: int | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Variable-qlen (per-batch MTP) FP4 paged MQA logits (gfx942)."""
    total_q = q_fp4.shape[0]
    if windows is None:
        if cu_seq_q is None or context_lens is None:
            raise ValueError(
                "flydsl_pa_mqa_logits_fp4_varqlen_gfx942: pass windows="
                "(row_to_batch, local_starts, local_ends) built once via "
                "compute_varqlen_windows, or both cu_seq_q and context_lens "
                "to build them here."
            )
        windows = compute_varqlen_windows(cu_seq_q, context_lens, total_q)
    row_to_batch, local_starts, local_ends = windows
    if parallel_unit_num is None:
        chunks_per_seq = max(1, (max_seq_len + block_k - 1) // block_k)
        parallel_unit_num = total_q * chunks_per_seq
    return flydsl_pa_mqa_logits_fp4_prefill_gfx942(
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
        relu=relu,
        kv_page_stride=kv_page_stride,
        kv_scale_page_stride=kv_scale_page_stride,
        kv_scale_offset=kv_scale_offset,
        block_k=block_k,
        kv_block_size=kv_block_size,
        num_warps=num_warps,
        parallel_unit_num=parallel_unit_num,
        out=out,
        cta_info=cta_info,
        n_ctas=n_ctas,
        stream=stream,
    )
