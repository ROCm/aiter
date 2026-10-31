# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Row-packed gfx942 FP4 prefill index scorer emitting 64-token page maxima.

For single-head indexers (MiniMax M3). The single-row prefill scorer fills one
of the MFMA's 16 M slots, so each e2m1 decode and scale multiply (VALU work on
gfx942) serves one query. Here the 16 slots hold 16 query rows of the same
sequence, so each decoded K tile serves 16 queries, and the epilogue scales,
masks each row to its causal window and reduces to per-page maxima in-kernel.

Scores are signed and scaled by ``weight_scale`` only. A page maximum is
bit-identical to ``amax`` over the single-row kernel's ``relu=False`` logits.

Layouts (G groups of 16 rows, S = 16 G slots):
  q_fp4:     [G, 16, D/2] uint8   — slot s of group g is packed row 16 g + s.
  q_scale:   [G, K_TILES, 4, 16, 4] uint8 — byte [g][kt][kk][s][0] is the e8m0
             of slot s at 32-chunk kk; bytes [..][1:] are padding.
  slot_meta: [3, S] int32 — output row (-1: empty slot), causal start, causal
             end of each slot.
  out:       [rows, >= ceil(max_seq_len / 64)] fp32, caller-filled with -inf;
             pages no CTA reaches keep that fill.
"""

from __future__ import annotations

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import gpu, rocdl
from flydsl.expr.primitive import range_constexpr
from flydsl.expr.typing import Int32, T

from .pa_mqa_logits_fp4_common import _i32_buffer, _load_vec4_i32
from .pa_mqa_logits_fp4_prefill_gfx942 import (
    CTA_INFO_WIDTH,
    DEFAULT_NUM_WARPS,
    KK_PER_KTILE,
    MFMA_N,
    WARP_SIZE,
    _decode_fp4_dword,
    _frag_i64_from_pp,
    compute_prefill_schedule,
)

ROWS_PER_GROUP = 16
PAGE_TOKENS = 64
DEFAULT_HEAD_DIM = 128
# fp32 -inf as a signed i32, for the bit-level mask select.
_NEG_INF_BITS = -8388608


def build_pa_mqa_page_max_fp4_rowpack_prefill_gfx942_module(
    block_k=256,
    kv_block_size=64,
    max_blocks_per_seq=256,
    num_warps=DEFAULT_NUM_WARPS,
    head_dim=DEFAULT_HEAD_DIM,
    kv_page_stride=None,
    kv_scale_page_stride=None,
    kv_scale_offset=0,
):
    heads = ROWS_PER_GROUP
    block_threads_k = num_warps * WARP_SIZE
    k_tiles = head_dim // 128
    assert head_dim % 128 == 0, f"head_dim must be a multiple of 128, got {head_dim}"
    N_TILES = block_k // MFMA_N
    assert N_TILES % num_warps == 0
    N_TILES_PER_WARP = N_TILES // num_warps
    assert kv_block_size == 64, "gfx942 packed KV loads require kv_block_size=64"
    assert (
        N_TILES_PER_WARP == 4
    ), "each wave covers one 64-token page per chunk: block_k == 64 * num_warps"
    pages_per_chunk = block_k // PAGE_TOKENS

    _stride_bt = max_blocks_per_seq
    _kv_chunk_bytes = 16
    _stride_kv_ktile = 4 * kv_block_size * _kv_chunk_bytes
    _stride_kv_block = k_tiles * _stride_kv_ktile
    _stride_kvs_ktile = 4 * kv_block_size
    _stride_kvs_block = k_tiles * _stride_kvs_ktile
    if kv_page_stride is not None:
        _stride_kv_block = kv_page_stride
    if kv_scale_page_stride is not None:
        _stride_kvs_block = kv_scale_page_stride
    qs_pad = 4
    N_KV = k_tiles * N_TILES_PER_WARP * KK_PER_KTILE
    N_KVS = k_tiles * KK_PER_KTILE

    @flyc.kernel
    def pa_mqa_page_max_fp4_rowpack_prefill_gfx942_kernel(
        out_ptr: fx.Tensor,
        q_ptr: fx.Tensor,
        q_scale_ptr: fx.Tensor,
        kv_cache_ptr: fx.Tensor,
        kv_scale_ptr: fx.Tensor,
        kv_indices_ptr: fx.Tensor,
        slot_meta_ptr: fx.Tensor,
        cta_info_ptr: fx.Tensor,
        stride_out_row: Int32,
        out_pages: Int32,
        slot_plane: Int32,
        out_elems: Int32,
        weight_scale: fx.Float32,
    ):
        tid = gpu.thread_idx.x
        pid = gpu.block_idx.x
        warp_id = tid >> 6
        lane_id = tid % WARP_SIZE
        lane_mod_16 = lane_id & 15
        lane_div_16 = (lane_id >> 4) & 3

        cta_src = fx.get_iter(cta_info_ptr)
        cta_it = fx.add_offset(
            fx.recast_iter(fx.PointerType.get(T.i32, cta_src.memspace, 4), cta_src),
            fx.Int64(pid) * fx.Int64(CTA_INFO_WIDTH),
        )
        cta_info_bt = fx.rocdl.make_buffer_tensor(
            fx.make_view(cta_it, fx.make_layout((1 << 28, 4), (4, 1)))
        )
        cta_info_vec = fx.Vector(_load_vec4_i32(cta_info_bt, fx.Int32(0)))
        group_id = cta_info_vec[0]
        batch_id = cta_info_vec[1]
        chunk_start = cta_info_vec[2]
        chunk_count = cta_info_vec[3]

        kv_bt = _i32_buffer(kv_cache_ptr, width=1)
        kvs_bt = _i32_buffer(kv_scale_ptr, width=1)
        bt_bt = _i32_buffer(kv_indices_ptr, width=1)
        q_bt = _i32_buffer(q_ptr, width=1)
        qs_bt = _i32_buffer(q_scale_ptr, width=1)

        # This lane's four MFMA output rows are slots 4 * (lane / 16) + t.
        meta_bt = _i32_buffer(slot_meta_ptr, width=4)
        slot0 = group_id * fx.Int32(ROWS_PER_GROUP) + lane_div_16 * fx.Int32(4)
        rows_v = fx.Vector(_load_vec4_i32(meta_bt, slot0))
        starts_v = fx.Vector(_load_vec4_i32(meta_bt, slot_plane + slot0))
        ends_v = fx.Vector(_load_vec4_i32(meta_bt, slot_plane * fx.Int32(2) + slot0))

        # Dropped stores use offset out_elems, discarded by the range check.
        out_buf = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fx.recast_iter(
                    fx.PointerType.get(T.f32, out_ptr.memspace, 4),
                    fx.get_iter(out_ptr),
                ),
                fx.make_layout((out_elems, 1), (1, 1)),
            ),
            max_size=False,
            num_records_bytes=out_elems * fx.Int32(4),
        )
        out_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 1)
        out_reg_ty = fx.MemRefType.get(
            T.f32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register
        )
        out_reg_lay = fx.make_layout(1, 1)

        ZERO_F = fx.Float32(0.0)
        c0_i32 = fx.Int32(0)
        NEG_INF = fx.Int32(_NEG_INF_BITS)
        mfma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.Float8E4M3FNUZ))

        def _e8m0_to_f32(b):
            v = fx.Vector.from_elements(
                [(b & fx.Int32(0xFF)) << fx.Int32(23)], dtype=fx.Int32
            ).bitcast(fx.Float32)
            return v[0]

        # ---- A-fragments: slot lane_mod_16 of this group, 4 K-steps per k_tile.
        q_base = group_id * fx.Int32(heads * (head_dim // 2))
        q_row_base = q_base + lane_mod_16 * fx.Int32(head_dim // 2)
        q_a_ops = []
        for k_tile in range_constexpr(k_tiles):
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
            q_a_ops.append(frags)

        # ---- A-scales of this lane's four output rows (slots 4*(l/16)+t). ----
        qs_base = group_id * fx.Int32(k_tiles * 64 * qs_pad)
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
                    dw = fx.Int32(qs_bt[off // fx.Int32(4)])
                    sa_kk.append(_e8m0_to_f32(dw & fx.Int32(0xFF)))
                sa_kt.append(sa_kk)
            sa_f32_tab.append(sa_kt)

        def _load_phys(c_i32_arg):
            token_local_base = (
                (chunk_start + c_i32_arg) * fx.Int32(block_k)
                + warp_id * fx.Int32(N_TILES_PER_WARP * MFMA_N)
                + lane_mod_16
            )
            phys_vec = bt_bt[batch_id * _stride_bt + (token_local_base >> fx.Int32(6))]
            return [phys_vec] * N_TILES_PER_WARP

        def _prefetch_chunk(c_i32_arg, phys_list):
            kv_list = []
            kvs_list = []
            phys_shared = phys_list[0]
            for k_tile in range_constexpr(k_tiles):
                for kk in range_constexpr(KK_PER_KTILE):
                    kvs_off = (
                        fx.Int64(phys_shared) * fx.Int64(_stride_kvs_block)
                        + fx.Int64(kv_scale_offset)
                        + fx.Int64(k_tile * _stride_kvs_ktile)
                        + fx.Int64(kk * kv_block_size)
                        + fx.Int64(lane_mod_16) * fx.Int64(N_TILES_PER_WARP)
                    )
                    kvs_list.append(kvs_bt[kvs_off // fx.Int64(4)])
            token_in_block0 = (
                (chunk_start + c_i32_arg) * fx.Int32(block_k)
                + warp_id * fx.Int32(N_TILES_PER_WARP * MFMA_N)
            ) & fx.Int32(63)
            perm_src_byte = (lane_mod_16 * fx.Int32(4) + lane_div_16) * fx.Int32(4)
            for nt in range_constexpr(N_TILES_PER_WARP):
                for k_tile in range_constexpr(k_tiles):
                    for kk in range_constexpr(KK_PER_KTILE):
                        token_in_block = (
                            token_in_block0
                            + fx.Int32(nt * MFMA_N)
                            + (lane_id // fx.Int32(4))
                        ) & fx.Int32(63)
                        kv_off = (
                            fx.Int64(phys_shared) * fx.Int64(_stride_kv_block)
                            + fx.Int64(k_tile * _stride_kv_ktile)
                            + fx.Int64(kk * kv_block_size * _kv_chunk_bytes)
                            + fx.Int64(token_in_block) * fx.Int64(_kv_chunk_bytes)
                            + fx.Int64(lane_id % fx.Int32(4)) * fx.Int64(4)
                        )
                        raw_c = kv_bt[kv_off // fx.Int64(4)]
                        kv_list.append(
                            fx.Int32(rocdl.ds_bpermute(T.i32, perm_src_byte, raw_c))
                        )
            return kv_list, kvs_list

        def _frag_from_i64(v):
            frag = fx.make_rmem_tensor(1, fx.Int64)
            frag.store(fx.Vector.from_elements([v], dtype=fx.Int64))
            return frag

        def _decode_frags(kv_list_in):
            frags = []
            for i in range_constexpr(N_KV):
                p01, p23 = _decode_fp4_dword(fx.Int32(kv_list_in[i]))
                frags.append(
                    fx.Int64(fx.Uint32(p01))
                    | (fx.Int64(fx.Uint32(p23)) << fx.Int32(32))
                )
            return frags

        def _issue_nt_mfmas_frags(frags_in, kvs_list_in, nt):
            acc = [ZERO_F for _ in range(4)]
            for k_tile in range_constexpr(k_tiles):
                for kk in range_constexpr(KK_PER_KTILE):
                    b_frag = _frag_from_i64(
                        frags_in[((nt * k_tiles) + k_tile) * KK_PER_KTILE + kk]
                    )
                    sb_packed = fx.Int32(kvs_list_in[k_tile * KK_PER_KTILE + kk])
                    sb_f32 = _e8m0_to_f32(
                        (sb_packed >> fx.Int32(8 * nt)) & fx.Int32(0xFF)
                    )
                    c_frag = fx.make_rmem_tensor(4, fx.Float32)
                    c_frag.store(fx.Vector.filled(4, 0.0, fx.Float32))
                    fx.gemm(mfma_atom, c_frag, q_a_ops[k_tile][kk], b_frag, c_frag)
                    partial = c_frag.load()
                    for t in range_constexpr(4):
                        acc[t] = fx.fma(
                            partial[t], sa_f32_tab[k_tile][kk][t] * sb_f32, acc[t]
                        )
            return acc

        def _masked_scaled(acc, nt, c_i32_arg):
            token = (
                (chunk_start + c_i32_arg) * fx.Int32(block_k)
                + (warp_id * fx.Int32(N_TILES_PER_WARP) + fx.Int32(nt))
                * fx.Int32(MFMA_N)
                + lane_mod_16
            )
            vals = []
            for t in range_constexpr(4):
                bits = (acc[t] * weight_scale).bitcast(fx.Int32)
                bits = (token < ends_v[t]).select(bits, NEG_INF)
                bits = (token < starts_v[t]).select(NEG_INF, bits)
                vals.append(bits.bitcast(fx.Float32))
            return fx.Vector.from_elements(vals, dtype=fx.Float32)

        lane_i32 = fx.Int32(lane_id)

        def _xor_max(vec, sh):
            peer_byte = (lane_i32 ^ fx.Int32(sh)) * fx.Int32(4)
            peers = []
            for t in range_constexpr(4):
                peer = fx.Int32(
                    rocdl.ds_bpermute(T.i32, peer_byte, vec[t].bitcast(fx.Int32))
                )
                peers.append(peer.bitcast(fx.Float32))
            return vec.maximumf(fx.Vector.from_elements(peers, dtype=fx.Float32))

        def _store_page_max(page_max, c_i32_arg):
            page = (chunk_start + c_i32_arg) * fx.Int32(pages_per_chunk) + warp_id
            # Pages past the last column would land in the next row.
            writer = (lane_mod_16 < fx.Int32(1)) & (page < out_pages)
            for t in range_constexpr(4):
                row = rows_v[t]
                off = row * stride_out_row + page
                off = (row < fx.Int32(0)).select(out_elems, off)
                off = writer.select(off, out_elems)
                r_out = fx.memref_alloca(out_reg_ty, out_reg_lay)
                fx.memref_store_vec(
                    fx.Vector.from_elements([page_max[t]], dtype=fx.Float32), r_out
                )
                fx.copy(out_atom, r_out, fx.slice(out_buf, (off, None)))

        def _compute_chunk_frags(frags_in, kvs_list_in, c_i32_arg):
            # Issue each tile's MFMAs before the previous tile's epilogue.
            acc0 = _issue_nt_mfmas_frags(frags_in, kvs_list_in, 0)
            acc1 = _issue_nt_mfmas_frags(frags_in, kvs_list_in, 1)
            page_max = _masked_scaled(acc0, 0, c_i32_arg)
            acc2 = _issue_nt_mfmas_frags(frags_in, kvs_list_in, 2)
            page_max = page_max.maximumf(_masked_scaled(acc1, 1, c_i32_arg))
            acc3 = _issue_nt_mfmas_frags(frags_in, kvs_list_in, 3)
            page_max = page_max.maximumf(_masked_scaled(acc2, 2, c_i32_arg))
            page_max = page_max.maximumf(_masked_scaled(acc3, 3, c_i32_arg))
            # Reduce over the 16 lanes holding one row's tokens of a tile.
            for sh in [1, 2, 4, 8]:
                page_max = _xor_max(page_max, sh)
            _store_page_max(page_max, c_i32_arg)

        last_c_i32 = chunk_count - fx.Int32(1)
        phys_pre = _load_phys(c0_i32)
        kv_pre, kvs_pre = _prefetch_chunk(c0_i32, phys_pre)
        frags_pre = _decode_frags(kv_pre)
        phys_next_pre = _load_phys(fx.min(fx.Int32(1), last_c_i32))

        chunk_count_minus_1_idx = fx.Int64(chunk_count - fx.Int32(1))
        init_args = list(frags_pre) + list(kvs_pre) + list(phys_next_pre)
        for c_idx, state in range(0, chunk_count_minus_1_idx, 1, init=init_args):
            frags_cur = [state[i] for i in range(N_KV)]
            kvs_cur_list = [state[N_KV + i] for i in range(N_KVS)]
            phys_next_list = [state[N_KV + N_KVS + i] for i in range(N_TILES_PER_WARP)]
            c_idx_i32 = fx.Int32(c_idx)
            c_next_i32 = c_idx_i32 + 1
            c_next_next_i32 = fx.min(c_next_i32 + 1, last_c_i32)
            kv_next, kvs_next = _prefetch_chunk(c_next_i32, phys_next_list)
            phys_next_next_list = _load_phys(c_next_next_i32)
            _compute_chunk_frags(frags_cur, kvs_cur_list, c_idx_i32)
            frags_next = _decode_frags(kv_next)
            results = yield (
                list(frags_next) + list(kvs_next) + list(phys_next_next_list)
            )

        frags_last = [results[i] for i in range(N_KV)]
        kvs_last_list = [results[N_KV + i] for i in range(N_KVS)]
        _compute_chunk_frags(frags_last, kvs_last_list, last_c_i32)

    return pa_mqa_page_max_fp4_rowpack_prefill_gfx942_kernel, block_threads_k


@lru_cache(maxsize=32)
def compile_pa_mqa_page_max_fp4_rowpack_prefill_gfx942(
    *,
    block_k: int = 256,
    kv_block_size: int = 64,
    max_blocks_per_seq: int = 256,
    num_warps: int = DEFAULT_NUM_WARPS,
    head_dim: int = DEFAULT_HEAD_DIM,
    kv_page_stride: int | None = None,
    kv_scale_page_stride: int | None = None,
    kv_scale_offset: int = 0,
):
    kfn, block_threads = build_pa_mqa_page_max_fp4_rowpack_prefill_gfx942_module(
        block_k=block_k,
        kv_block_size=kv_block_size,
        max_blocks_per_seq=max_blocks_per_seq,
        num_warps=num_warps,
        head_dim=head_dim,
        kv_page_stride=kv_page_stride,
        kv_scale_page_stride=kv_scale_page_stride,
        kv_scale_offset=kv_scale_offset,
    )

    @flyc.jit
    def launch_pa_mqa_page_max_fp4_rowpack_prefill_gfx942(
        out,
        q,
        qs,
        kv,
        kvs,
        bt,
        meta,
        cta_info_,
        stride_out: fx.Int32,
        out_pages: fx.Int32,
        slot_plane: fx.Int32,
        out_elems: fx.Int32,
        weight_scale: fx.Float32,
        gx: fx.Int32,
        stream: fx.Stream,
    ):
        kfn(
            out,
            q,
            qs,
            kv,
            kvs,
            bt,
            meta,
            cta_info_,
            stride_out,
            out_pages,
            slot_plane,
            out_elems,
            weight_scale,
        ).launch(grid=(fx.Int64(gx),), block=(block_threads, 1, 1), stream=stream)

    return launch_pa_mqa_page_max_fp4_rowpack_prefill_gfx942, block_threads


def rowpack_prefill_inputs_gfx942(
    q_rows_fp4: torch.Tensor,
    q_rows_scale: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    row_to_batch: torch.Tensor,
    local_starts: torch.Tensor,
    local_ends: torch.Tensor,
):
    """Pack ragged prefill rows into 16-row groups that never cross a sequence.

    Device-only, no host sync: there are ``rows // 16 + batches`` groups and
    the unused ones are empty. ``q_rows_scale`` is the ``[rows, D/32]`` e8m0 of
    ``dynamic_mxfp4_quant``.

    Returns:
        ``(q_fp4, q_scale, slot_meta, group_to_batch, group_starts,
        group_ends)`` in the kernel's layouts.
    """
    rows, packed_dim = q_rows_fp4.shape
    chunks = q_rows_scale.shape[1]
    batches = cu_seqlens_q.shape[0] - 1
    dev = q_rows_fp4.device
    groups = rows // ROWS_PER_GROUP + batches
    slots = groups * ROWS_PER_GROUP

    cu = cu_seqlens_q.to(torch.int64)
    per_batch = (cu[1:] - cu[:-1] + ROWS_PER_GROUP - 1) // ROWS_PER_GROUP
    group_base = torch.cumsum(per_batch, 0) - per_batch
    r = torch.arange(rows, device=dev)
    b = row_to_batch.to(torch.int64)
    n = r - cu[b]
    g = group_base[b] + n // ROWS_PER_GROUP
    slot = g * ROWS_PER_GROUP + n % ROWS_PER_GROUP

    slot_meta = torch.zeros((3, slots), dtype=torch.int32, device=dev)
    slot_meta[0].fill_(-1)
    slot_meta[0, slot] = r.to(torch.int32)
    slot_meta[1, slot] = local_starts.to(torch.int32)
    slot_meta[2, slot] = local_ends.to(torch.int32)

    q_fp4 = torch.zeros((slots, packed_dim), dtype=torch.uint8, device=dev)
    q_fp4[slot] = q_rows_fp4
    natural = torch.zeros((slots, chunks), dtype=torch.uint8, device=dev)
    natural[slot] = q_rows_scale
    # [g, slot, chunk] -> [g, k_tile, chunk-in-tile, slot, pad=4], byte 0 live.
    k_tiles = chunks // KK_PER_KTILE
    q_scale = torch.zeros(
        (groups, k_tiles, KK_PER_KTILE, ROWS_PER_GROUP, 4),
        dtype=torch.uint8,
        device=dev,
    )
    q_scale[..., 0] = natural.view(
        groups, ROWS_PER_GROUP, k_tiles, KK_PER_KTILE
    ).permute(0, 2, 3, 1)

    ends = local_ends.to(torch.int32)
    group_to_batch = torch.zeros(groups, dtype=torch.int32, device=dev)
    group_to_batch.scatter_(0, g, row_to_batch.to(torch.int32))
    group_ends = torch.zeros(groups, dtype=torch.int32, device=dev)
    group_ends.scatter_reduce_(0, g, ends, "amax")
    group_starts = torch.full((groups,), 2**31 - 1, dtype=torch.int32, device=dev)
    group_starts.scatter_reduce_(0, g, local_starts.to(torch.int32), "amin")
    group_starts = torch.where(group_ends > 0, group_starts, 0)
    return (
        q_fp4.view(groups, ROWS_PER_GROUP, packed_dim),
        q_scale,
        slot_meta,
        group_to_batch,
        group_starts,
        group_ends,
    )


def flydsl_pa_mqa_page_max_fp4_rowpack_prefill_gfx942(
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    slot_meta: torch.Tensor,
    group_to_batch: torch.Tensor,
    group_starts: torch.Tensor,
    group_ends: torch.Tensor,
    max_seq_len: int,
    out: torch.Tensor,
    *,
    weight_scale: float = 1.0,
    kv_page_stride: int | None = None,
    kv_scale_page_stride: int | None = None,
    kv_scale_offset: int = 0,
    block_k: int = 256,
    kv_block_size: int = 64,
    num_warps: int = DEFAULT_NUM_WARPS,
    parallel_unit_num: int | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Signed per-64-token-page maxima of row-packed prefill index scores.

    ``out`` must be ``-inf``-filled by the caller. ``kv_page_stride``,
    ``kv_scale_page_stride`` and ``kv_scale_offset`` (bytes) read a paged cache
    whose data and scales share one allocation. ``parallel_unit_num`` bounds
    the persistent grid (default four CTAs per CU, at least one per group).
    """
    groups = q_fp4.shape[0]
    if tuple(q_fp4.shape[1:]) != (ROWS_PER_GROUP, q_fp4.shape[2]):
        raise ValueError(f"q_fp4 must be [groups, 16, D/2], got {tuple(q_fp4.shape)}")
    head_dim = q_fp4.shape[2] * 2
    slots = groups * ROWS_PER_GROUP
    if slot_meta.dtype != torch.int32 or tuple(slot_meta.shape) != (3, slots):
        raise ValueError(f"slot_meta must be int32 [3, {slots}]")
    if not slot_meta.is_contiguous():
        raise ValueError("slot_meta must be contiguous")
    pages = (max_seq_len + PAGE_TOKENS - 1) // PAGE_TOKENS
    if out.dtype != torch.float32 or out.dim() != 2 or out.stride(1) != 1:
        raise ValueError("out must be a row-major fp32 [rows, pages] tensor")
    if out.shape[1] < pages:
        raise ValueError(f"out needs >= {pages} page columns, got {out.shape[1]}")
    if out.numel() >= 1 << 29:
        raise ValueError("out must stay below 2 GiB (one buffer descriptor)")
    if groups == 0:
        return out

    if parallel_unit_num is None:
        props = torch.cuda.get_device_properties(q_fp4.device)
        parallel_unit_num = 4 * props.multi_processor_count
    parallel_unit_num = max(parallel_unit_num, groups)
    _, cta_info, n_ctas = compute_prefill_schedule(
        group_to_batch,
        group_starts,
        group_ends,
        block_k,
        parallel_unit_num,
        max_seq_len,
    )
    launcher, _ = compile_pa_mqa_page_max_fp4_rowpack_prefill_gfx942(
        block_k=block_k,
        kv_block_size=kv_block_size,
        max_blocks_per_seq=block_tables.shape[1],
        num_warps=num_warps,
        head_dim=head_dim,
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
        slot_meta,
        cta_info,
        out.stride(0),
        out.shape[1],
        slots,
        out.numel(),
        float(weight_scale),
        n_ctas,
        stream,
    )
    return out
