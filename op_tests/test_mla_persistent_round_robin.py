# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Round-robin (interleave) context-parallel MLA decode test.

KV is round-robin sharded across `cp_world_size` ranks (global pos p -> rank
p % W); each rank runs aiter.mla_decode_fwd over its local shard with the
global-position causal mask (g(j)=j*W+r), and per-rank partials are merged via
online softmax and compared against the full-KV causal golden.

This was split out of test_mla_persistent.py into its own file.

Example:
    python3 op_tests/test_mla_persistent_round_robin.py -d bf16 -kvd bf16 \
        -n 16,4 -b 1 -c 13 -cpw 4
"""

import torch
import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import checkAllclose, benchmark, run_perftest
from aiter import dtypes
import random
import itertools
import argparse
import pandas as pd

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)


def check_support(dtype, kv_dtype, nhead):
    if dtype == dtypes.fp8 and kv_dtype == dtypes.bf16:
        return False
    if dtype == dtypes.fp8 and kv_dtype == dtypes.fp8:
        return False
    if get_gfx() == "gfx942":
        return False
    return True


def cal_diff(
    x: torch.Tensor, y: torch.Tensor, name: str, use_fp8: bool = False
) -> None:
    x, y = x.double(), y.double()
    # RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    amax_diff = (x - y).abs().max().item()
    thr = 3e-2 if use_fp8 else 1e-5
    flag = (
        "  <<< over-thr (bf16 merge noise; checkAllclose is authoritative)"
        if cos_diff >= thr
        else ""
    )
    print(
        f"[cal_diff] {name}: cos_diff={cos_diff:.3e} amax_diff={amax_diff:.3e} thr={thr:.0e}{flag}"
    )


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    dtype,
    is_causal=True,
    is_fp8_q=False,
    is_fp8_kvc=False,
    q_scale=None,
    kv_scale=None,
    causal_diagonal=None,
    attn_mask=None,
):
    if is_fp8_q and q_scale is not None:
        scale *= q_scale
    if is_fp8_kvc and kv_scale is not None:
        scale *= kv_scale
    attn_weights = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * scale

    if attn_mask is not None:
        # Explicit boolean visibility mask [s_q, s_k] (True == keep). Used by the
        # round-robin CP reference where the causal relation is on GLOBAL token
        # positions and therefore cannot be expressed as a single diagonal.
        attn_bias = torch.zeros_like(attn_weights)  # [h, q, k]
        attn_bias.masked_fill_(attn_mask[None].logical_not(), float("-inf"))
        attn_weights = attn_weights + attn_bias
    elif is_causal:
        s_q = query.shape[0]
        s_k = key.shape[0]
        diagonal = causal_diagonal if causal_diagonal is not None else s_k - s_q
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=diagonal)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weights += attn_bias

    lse = attn_weights.logsumexp(dim=-1)
    m = attn_weights.max(-1).values
    attn_weights_exp = torch.exp(attn_weights - m.unsqueeze(-1))
    l = attn_weights_exp.sum(-1)  # noqa: E741
    if is_fp8_q:
        attn_weights_fp8 = attn_weights_exp.to(dtypes.fp8)
        attn_weights_exp = attn_weights_fp8.to(torch.float)

    out = torch.einsum("hqk,khd->qhd", attn_weights_exp.float(), value.float())
    out = out / l.transpose(0, 1).unsqueeze(-1)
    if is_fp8_kvc and kv_scale is not None:
        out *= kv_scale

    if attn_mask is not None:
        # Query rows with no visible key (whole row masked) produce NaN above;
        # define them as out=0, lse=-inf (an empty CP shard for that token).
        invalid = attn_mask.any(dim=-1).logical_not()  # [s_q]
        if bool(invalid.any()):
            out = out.clone()
            out[invalid] = 0.0
            lse = lse.clone()
            lse[:, invalid] = float("-inf")

    return out.to(dtype), lse


def torch_mla_extend(
    q,  # [total_q, nheads, headdim_q]
    kvc_cache,  # [num_page, page_size, nhead_kv, qk_head_dim]
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    sm_scale,
    kv_lora_rank,
    qk_rope_head_dim,
    dtype,
    is_causal=True,
    q_scale=None,
    kv_scale=None,
):
    num_page, page_size, nhead_kv, _ = kvc_cache.shape
    is_fp8_q = q.dtype == dtypes.fp8
    is_fp8_kvc = kvc_cache.dtype == dtypes.fp8

    if is_fp8_q:
        q = q.to(torch.float)

    if is_fp8_kvc:
        kvc_cache = kvc_cache.to(torch.float)

    qs = torch.tensor_split(q, qo_indptr.tolist()[1:])
    kvc = torch.index_select(kvc_cache, 0, kv_indices)
    kvs = torch.tensor_split(kvc, kv_indptr.tolist()[1:])
    bs = qo_indptr.shape[0] - 1

    os = []
    lses = []
    for i in range(bs):
        cur_num_page = kvs[i].shape[0]
        real_kv_seq_len = (cur_num_page - 1) * page_size + kv_last_page_lens.tolist()[i]
        kvc = kvs[i].flatten(0, 1)[:real_kv_seq_len,]
        q = qs[i]
        k = kvc
        v, _ = torch.split(kvc, [kv_lora_rank, qk_rope_head_dim], dim=-1)
        o, lse = ref_masked_attention(
            q,
            k,
            v,
            sm_scale,
            dtype,
            is_causal=is_causal,
            is_fp8_q=is_fp8_q,
            is_fp8_kvc=is_fp8_kvc,
            q_scale=q_scale,
            kv_scale=kv_scale,
        )
        os.append(o)
        lses.append(lse)
    o = torch.concat(os)
    # Each lse is (nheads, seq_q_i); concatenate query positions along dim=1, then (total_q, nheads).
    lse = torch.concat(lses, dim=1).transpose(0, 1)
    return o, lse


def torch_mla_extend_round_robin(
    q,  # [total_q, nheads, headdim_q]
    kvc_cache,  # [num_page, page_size, nhead_kv, qk_head_dim]
    qo_indptr,
    kv_indptr_r,  # [batch+1] per-rank LOCAL page indptr (same as kernel input)
    kv_indices_r,  # [num_local_page] per-rank LOCAL physical pages (same as kernel input)
    g_kv_indptr,  # [batch+1] GLOBAL page indptr (per-request global KV length)
    page_size,
    sm_scale,
    kv_lora_rank,
    qk_rope_head_dim,
    dtype,
    cp_world_size,
    cp_rank,
    q_scale=None,
    kv_scale=None,
):
    """Round-robin (interleave) context-parallel reference for ONE rank.

    Consumes the SAME pre-computed per-rank indices the aiter kernel path uses:
    ``kv_indptr_r`` / ``kv_indices_r`` select this rank's local KV from the paged
    cache, and ``g_kv_indptr`` gives the per-request GLOBAL KV length. The shard
    positions are NOT recomputed here; the global position of local token ``j`` is
    simply ``g(j) = j * cp_world_size + cp_rank`` (round-robin), used to build the
    causal mask: query token ``i`` (global pos ``global_len - s_q + i``) attends to
    local token ``j`` iff ``g(j) <= q_global``.

    Assumes ``page_size == 1`` (round-robin is token-granular).

    Returns (out[total_q, nheads, kv_lora_rank], lse[total_q, nheads]); empty
    shards / fully-masked query rows yield out=0, lse=-inf.
    """
    dev = kvc_cache.device
    is_fp8_q = q.dtype == dtypes.fp8
    is_fp8_kvc = kvc_cache.dtype == dtypes.fp8
    if is_fp8_q:
        q = q.to(torch.float)
    if is_fp8_kvc:
        kvc_cache = kvc_cache.to(torch.float)

    qs = torch.tensor_split(q, qo_indptr.tolist()[1:])
    kvc = torch.index_select(kvc_cache, 0, kv_indices_r)  # local pages of this rank
    indptr_r = kv_indptr_r.tolist()
    g_indptr = g_kv_indptr.tolist()
    bs = qo_indptr.shape[0] - 1

    os = []
    lses = []
    for i in range(bs):
        q_i = qs[i]
        s_q, nheads, _ = q_i.shape

        # local KV pages of this request, sliced explicitly via kv_indptr_r
        p0, p1 = int(indptr_r[i]), int(indptr_r[i + 1])
        s_k = (p1 - p0) * page_size

        if s_k == 0:
            # empty local shard -> out=0, lse=-inf (matches kernel/merge contract)
            os.append(torch.zeros(s_q, nheads, kv_lora_rank, dtype=dtype, device=dev))
            lses.append(torch.full((nheads, s_q), float("-inf"), device=dev))
            continue

        local_kv = kvc[p0:p1].flatten(0, 1)[:s_k]  # [s_k, nhead_kv, qk_head_dim]
        k = local_kv
        v, _ = torch.split(local_kv, [kv_lora_rank, qk_rope_head_dim], dim=-1)

        # GLOBAL positions of local tokens (round-robin): g(j) = j*W + r
        local_global_pos = torch.arange(s_k, device=dev) * cp_world_size + cp_rank
        global_len = (int(g_indptr[i + 1]) - int(g_indptr[i])) * page_size
        q_global = (global_len - s_q) + torch.arange(s_q, device=dev)
        attn_mask = local_global_pos[None, :] <= q_global[:, None]  # [s_q, s_k]

        o, lse = ref_masked_attention(
            q_i,
            k,
            v,
            sm_scale,
            dtype,
            is_fp8_q=is_fp8_q,
            is_fp8_kvc=is_fp8_kvc,
            q_scale=q_scale,
            kv_scale=kv_scale,
            attn_mask=attn_mask,
        )
        os.append(o)
        lses.append(lse)

    o = torch.concat(os)
    lse = torch.concat(lses, dim=1).transpose(0, 1)
    return o, lse


def merge_cp_ranks(cp_outs, cp_lses, out_dtype=torch.bfloat16):
    """Online-softmax merge of per-rank CP partials.

    cp_outs[r]: [total_q, nheads, dv] (fp32-friendly)
    cp_lses[r]: [total_q, nheads]
    -> merged out [total_q, nheads, dv] (out_dtype), merged lse [total_q, nheads].
    """
    LS = torch.stack([lse.float() for lse in cp_lses], 0)  # [W, total_q, nheads]
    glse = torch.logsumexp(LS, 0)  # [total_q, nheads]
    w = torch.exp(LS - glse).nan_to_num_(0.0)  # [W, total_q, nheads]
    out = sum(w[r][..., None] * cp_outs[r].float() for r in range(len(cp_outs)))
    return out.to(out_dtype), glse


def aiter_cp_rank_decode(
    q,  # [total_q, nhead, qk_head_dim]
    kv_buffer,  # [num_page, page_size, nhead_kv, qk_head_dim] ORIGINAL paged KV (global)
    qo_indptr,  # [batch+1] int32
    kv_indptr_r,  # [batch+1] int32  per-rank LOCAL page indptr (metadata tiling + kernel split)
    kv_indices_r,  # [num_local_page] int32  physical pages of THIS rank's local KV
    g_kv_indptr,  # [batch+1] int32  GLOBAL page indptr (kernel global-pos causal)
    kv_last_page_lens,  # [batch] int32
    batch_size,
    max_seqlen_q,
    nhead,
    nhead_kv,
    kv_lora_rank,
    qk_head_dim,
    v_head_dim,
    sm_scale,
    dtype,
    kvtype,
    max_split_per_batch,
    is_causal,
    cp_world_size,
    cp_rank,
):
    """Run aiter persistent MLA decode for ONE CP rank.

    Keeps the ORIGINAL global ``kv_buffer`` and selects this rank's KV via
    ``kv_indices_r``. Metadata tiling/splitting is driven by the per-rank LOCAL
    ``kv_indptr_r``. ``g_kv_indptr`` together with ``(cp_world_size, cp_rank)``
    let the kernel map a local index ``j`` back to its global position
    ``g(j) = j * W + r`` for the round-robin causal mask.
    """
    dev = q.device
    total_q = q.shape[0]

    o = torch.zeros(total_q, nhead, v_head_dim, dtype=torch.bfloat16, device=dev)

    info = aiter.get_mla_metadata_info_v1(
        batch_size,
        max_seqlen_q,
        nhead,
        dtype,
        kvtype,
        is_sparse=False,
        fast_mode=True,
        num_kv_splits=max_split_per_batch,
        intra_batch_mode=False,
    )

    def _alloc(sz, ty):
        return torch.empty(sz, dtype=ty, device=dev)

    work_meta_data = _alloc(*info[0])
    work_indptr = _alloc(*info[1])
    work_info_set = _alloc(*info[2])
    reduce_indptr = _alloc(*info[3])
    reduce_final_map = _alloc(*info[4])
    reduce_partial_map = _alloc(*info[5])

    # per-rank LOCAL kv_indptr_r drives the work tiling / kv-split
    aiter.get_mla_metadata_v1(
        qo_indptr,
        kv_indptr_r,
        kv_last_page_lens,
        nhead // nhead_kv,
        nhead_kv,
        False,
        work_meta_data,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        page_size=1,
        kv_granularity=16,
        max_seqlen_qo=max_seqlen_q,
        uni_seqlen_qo=max_seqlen_q,
        fast_mode=True,
        max_split_per_batch=max_split_per_batch,
        intra_batch_mode=False,
        dtype_q=dtype,
        dtype_kv=kvtype,
        is_cp_round_robin=True,
    )

    (_, final_lse), us = run_perftest(
        aiter.mla.mla_decode_fwd,
        q,
        kv_buffer,
        o,
        qo_indptr,
        kv_indptr_r,
        kv_indices_r,
        kv_last_page_lens,
        max_seqlen_q=max_seqlen_q,
        page_size=1,
        nhead_kv=nhead_kv,
        sm_scale=sm_scale,
        num_kv_splits=max_split_per_batch,
        work_meta_data=work_meta_data,
        work_indptr=work_indptr,
        work_info_set=work_info_set,
        reduce_indptr=reduce_indptr,
        reduce_final_map=reduce_final_map,
        reduce_partial_map=reduce_partial_map,
        intra_batch_mode=False,
        return_lse=True,
        g_kv_indptr=g_kv_indptr,
        cp_world_size=cp_world_size,
        cp_rank=cp_rank,
    )
    lse = final_lse.float() if final_lse is not None else None
    return o.float(), lse, us


@benchmark()
def test_mla_cp(
    ctx_lens,
    batch_size,
    nhead,
    kv_lora_rank,
    qk_rope_head_dim,
    v_head_dim,
    dtype,
    kvtype,
    page_size,
    varlen,
    decode_qlen,
    cp_world_size,
    max_split_per_batch,
    return_lse=False,
):
    """Round-robin (interleave) context-parallel MLA decode test.

    Reuses the paged reference (`torch_mla_extend`) for the full-KV causal
    golden, and the page/batch-aware `torch_mla_extend_round_robin` for the
    per-rank CP reference. KV is round-robin sharded across `cp_world_size`
    ranks (global pos p -> rank p % W); the per-rank local KV is EXTRACTED by
    `shard_pos` from the paged cache. We then:
      1. merge the per-rank CP references (cp_outs/cp_lses) and assert it matches
         the golden (out_ref/lse_ref)  -> validates the CP reference itself;
      2. run aiter.mla_decode_fwd per rank, merge, and compare to the golden
         (and per-rank vs the CP reference) -> validates the kernel.
    """
    ret = {}
    W = cp_world_size
    dev = "cuda"
    out_dtype = torch.bfloat16
    nhead_kv = 1
    qlen = decode_qlen
    qk_head_dim = kv_lora_rank + qk_rope_head_dim
    sm_scale = 1.0 / (qk_head_dim**0.5)
    is_causal = qlen > 1
    # round-robin interleave is token-granular: global pos p == page index, so the
    # per-rank kv_indices_r extraction below requires page_size == 1.
    page_size = 1

    # ---- build paged KV / Q (mirrors test_mla decode setup) ---------------- #
    kv_block_nums = torch.empty(batch_size, dtype=torch.int)
    seq_lens_kv = torch.empty(batch_size, dtype=torch.int)
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
    if varlen:
        for i in range(batch_size):
            seq_lens_kv[i] = max(int(random.uniform(W, ctx_lens)), W)
            kv_block_nums[i] = (seq_lens_kv[i] + page_size - 1) // page_size
            kv_last_page_lens[i] = (
                page_size
                if seq_lens_kv[i] % page_size == 0
                else seq_lens_kv[i] % page_size
            )
    else:
        seq_lens_kv.fill_(ctx_lens)
        kv_block_nums.fill_((ctx_lens + page_size - 1) // page_size)
        kv_last_page_lens.fill_(
            page_size if ctx_lens % page_size == 0 else ctx_lens % page_size
        )

    assert (
        int(seq_lens_kv.min().item()) >= W
    ), f"every request kv_len must be >= cp_world_size({W})"

    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    kv_indptr[1:] = torch.cumsum(kv_block_nums, dim=0)
    num_page = int(kv_indptr[-1].item())
    kv_indices = torch.randperm(num_page, dtype=torch.int)

    seq_lens_qo = torch.full((batch_size,), qlen, dtype=torch.int)
    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    qo_indptr[1:] = torch.cumsum(seq_lens_qo, dim=0)
    total_q = int(qo_indptr[-1].item())

    kv_buffer = torch.randn(
        (num_page, page_size, 1, qk_head_dim), dtype=torch.bfloat16
    ).to(kvtype)
    q = torch.randn((total_q, nhead, qk_head_dim), dtype=torch.bfloat16).to(dtype)

    # ---- full-KV causal golden via the EXISTING reference ------------------ #
    out_ref, lse_ref = torch_mla_extend(
        q,
        kv_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        sm_scale,
        kv_lora_rank,
        qk_rope_head_dim,
        dtype=out_dtype,
        is_causal=True,
    )

    # ---- build per-rank LOCAL indices ONCE (shared by reference + kernel) --- #
    # page_size==1 so the GLOBAL page indptr == token indptr. For each rank build:
    #   kv_indices_r : physical pages of positions p with (p % W == r), taken from
    #                  the ORIGINAL kv_indices -> selects this rank's KV in-place;
    #   kv_indptr_r  : per-request local page counts (drives metadata tiling);
    #   g_kv_indptr  : the original GLOBAL page indptr (per-request global length).
    g_kv_indptr = kv_indptr.to(dev).to(torch.int32)
    kv_indices_dev = kv_indices.to(dev).to(torch.int32)
    qo_indptr_dev = qo_indptr.to(dev).to(torch.int32)
    kv_buffer_dev = kv_buffer.to(dev)

    rank_kv_indptr_r, rank_kv_indices_r, rank_kv_last_r = [], [], []
    for r in range(W):
        idx_r_list, local_lens = [], []
        for b in range(batch_size):
            real_kv = int(seq_lens_kv[b].item())
            start = int(kv_indptr[b].item())
            pos = torch.arange(real_kv, device=dev)
            pos = pos[pos % W == r]  # this rank's GLOBAL positions in request b
            idx_r_list.append(kv_indices_dev[start + pos])  # -> physical pages
            local_lens.append(int(pos.numel()))
        kv_indices_r = (
            torch.cat(idx_r_list).to(torch.int32)
            if sum(local_lens) > 0
            else torch.zeros(1, dtype=torch.int32, device=dev)
        )
        kv_indptr_r = torch.zeros(batch_size + 1, dtype=torch.int32, device=dev)
        kv_indptr_r[1:] = torch.cumsum(
            torch.tensor(local_lens, dtype=torch.int32, device=dev), dim=0
        )
        rank_kv_indptr_r.append(kv_indptr_r)
        rank_kv_indices_r.append(kv_indices_r)
        rank_kv_last_r.append(torch.ones(batch_size, dtype=torch.int32, device=dev))

    # ---- per rank: CP reference + aiter kernel, compared rank-by-rank ------- #
    cp_outs, cp_lses = [], []
    aiter_outs, aiter_lses, rank_us = [], [], []
    for r in range(W):
        kv_indptr_r = rank_kv_indptr_r[r]
        kv_indices_r = rank_kv_indices_r[r]
        kv_last_page_lens_r = rank_kv_last_r[r]
        # per-rank round-robin CP reference (consumes the same per-rank indices)
        o_r, l_r = torch_mla_extend_round_robin(
            q,
            kv_buffer_dev,
            qo_indptr_dev,
            kv_indptr_r,
            kv_indices_r,
            g_kv_indptr,
            page_size,
            sm_scale,
            kv_lora_rank,
            qk_rope_head_dim,
            dtype=out_dtype,
            cp_world_size=W,
            cp_rank=r,
        )
        cp_outs.append(o_r)
        cp_lses.append(l_r)

        # run_perftest is applied inside aiter_cp_rank_decode and times ONLY the
        # mla_decode_fwd call (metadata setup is excluded from the perf number).
        o_a, l_a, us = aiter_cp_rank_decode(
            q,
            kv_buffer_dev,
            qo_indptr_dev,
            kv_indptr_r,
            kv_indices_r,
            g_kv_indptr,
            kv_last_page_lens_r,
            batch_size,
            qlen,
            nhead,
            nhead_kv,
            kv_lora_rank,
            qk_head_dim,
            v_head_dim,
            sm_scale,
            dtype,
            kvtype,
            max_split_per_batch,
            is_causal,
            W,
            r,
        )
        rank_us.append(us)

        # per-rank comparison: aiter kernel vs CP reference (this rank's shard),
        # comparing the raw kernel outputs directly (no NaN sanitization).
        local_lens_r = (kv_indptr_r[1:] - kv_indptr_r[:-1]).tolist()
        checkAllclose(
            o_r.float(),
            o_a,
            msg=f"mla_cp_round_robin W={W} qlen={qlen} rank{r} "
            f"local_len={local_lens_r} has_NaN={bool(torch.isnan(o_a).any())} "
            f"[cp_ref vs aiter]:......",
        )
        if return_lse:
            checkAllclose(
                l_r.float(),
                l_a,
                msg=f"mla_cp_round_robin W={W} qlen={qlen} rank{r} "
                f"[cp_ref vs aiter lse]:......",
            )
        aiter_outs.append(o_a)
        aiter_lses.append(l_a)

    # ---- merged comparisons vs full-KV causal golden ----------------------- #
    cp_merged_out, cp_merged_lse = merge_cp_ranks(cp_outs, cp_lses, out_dtype)
    err_ref = checkAllclose(
        out_ref,
        cp_merged_out,
        msg=f"mla_cp_round_robin W={W} qlen={qlen} [golden vs cp_ref_merge out]:......",
    )
    checkAllclose(
        lse_ref,
        cp_merged_lse,
        msg=f"mla_cp_round_robin W={W} qlen={qlen} [golden vs cp_ref_merge lse]:......",
    )

    aiter_merged_out, aiter_merged_lse = merge_cp_ranks(
        aiter_outs, aiter_lses, out_dtype
    )
    err = checkAllclose(
        out_ref,
        aiter_merged_out,
        msg=f"mla_cp_round_robin W={W} qlen={qlen} [golden vs aiter_merge out]:......",
    )
    if return_lse:
        checkAllclose(
            lse_ref,
            aiter_merged_lse,
            msg=f"mla_cp_round_robin W={W} qlen={qlen} [golden vs aiter_merge lse]:......",
        )
    cal_diff(out_ref, aiter_merged_out, "out", False)
    ret["cp:err_ref"] = err_ref
    ret["cp:err_aiter"] = err
    ret["cp:world_size"] = W
    ret["cp:rank_us"] = sum(rank_us) / max(len(rank_us), 1)
    return ret


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="round-robin context-parallel MLA decode test",
)
parser.add_argument(
    "-k",
    "--kv_lora_rank",
    type=int,
    default=512,
    help="""kv lora rank.
    e.g.: -k 512""",
)
parser.add_argument(
    "-qr",
    "--qk_rope_head_dim",
    type=int,
    default=64,
    help="""qk rope head dim.
    e.g.: -qr 64""",
)
parser.add_argument(
    "-vh",
    "--v_head_dim",
    type=int,
    default=512,
    help="""v head dim.
    e.g.: -vh 512""",
)
parser.add_argument(
    "-blk",
    "--block_size",
    type=int,
    default=1,
    help="""Block size (round-robin requires 1).
    e.g.: -blk 1""",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    choices=[dtypes.d_dtypes["bf16"]],
    nargs="*",
    default=[dtypes.d_dtypes["bf16"]],
    metavar="{bf16, fp8}",
    help="""Data type of Q.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-kvd",
    "--kv_dtype",
    type=dtypes.str2Dtype,
    choices=[dtypes.d_dtypes["bf16"]],
    nargs="*",
    metavar="{bf16, fp8}",
    default=[dtypes.d_dtypes["bf16"]],
    help="""Data type of KV.
    e.g.: -kvd bf16""",
)
parser.add_argument(
    "-c",
    "--ctxLen",
    type=int,
    nargs="*",
    default=[13, 23, 64, 256, 512, 1200, 3200, 5200, 8192],
    help="""Context length (global KV length).
    e.g.: -c 13""",
)
parser.add_argument(
    "-b",
    "--batchSize",
    type=int,
    nargs="*",
    default=[1, 10, 32, 64, 128],
    help="""Batch size.
    e.g.: -b 1""",
)
parser.add_argument(
    "-n",
    "--nhead",
    type=dtypes.str2tuple,
    nargs="*",
    const=None,
    default=[(16, 2), (32, 3), (64, 1), (64, 2), (128, 2)],
    help="""Number of heads, decode_qlen pairs.
    e.g.: -n 16,4""",
)
parser.add_argument(
    "-ms",
    "--max_split_per_batch",
    type=int,
    nargs="*",
    default=[32],
    help="""kv seqlens max split num for per batch.
    e.g.: -ms 32""",
)
parser.add_argument(
    "--varlen",
    action="store_true",
    help="""variable kv seqlens per batch. Default: False.
    --varlen # True""",
)
parser.add_argument(
    "-cp",
    "--cp_round_robin",
    action="store_true",
    help="""accepted for backward compatibility; this file always runs the
    round-robin CP test.""",
)
parser.add_argument(
    "-cpw",
    "--cp_world_size",
    type=int,
    nargs="*",
    default=[2, 3, 4, 7, 8],
    help="""cp world size (number of round-robin ranks).
    e.g.: -cpw 4""",
)
parser.add_argument(
    "-lse",
    "--return_lse",
    action="store_true",
    help="""also compare LSE results (per-rank cp_ref vs aiter, and golden vs
    aiter_merge). Default: False.
    --lse # True""",
)

args = parser.parse_args()
for nhead, decode_qlen in args.nhead:
    df = []
    for (
        dtype,
        kvtype,
        ctx_len,
        batch_size,
        max_split_per_batch,
        cp_world_size,
    ) in itertools.product(
        args.dtype,
        args.kv_dtype,
        args.ctxLen,
        args.batchSize,
        args.max_split_per_batch,
        args.cp_world_size,
    ):
        if dtype != dtypes.bf16 or kvtype != dtypes.bf16:
            # CP round-robin path validated for bf16/bf16 only for now.
            continue
        if not check_support(dtype, kvtype, nhead):
            continue
        if ctx_len < cp_world_size:
            continue
        ret = test_mla_cp(
            ctx_len,
            batch_size,
            nhead,
            args.kv_lora_rank,
            args.qk_rope_head_dim,
            args.v_head_dim,
            dtype,
            kvtype,
            page_size=args.block_size,
            varlen=args.varlen,
            decode_qlen=decode_qlen,
            cp_world_size=cp_world_size,
            max_split_per_batch=max_split_per_batch,
            return_lse=args.return_lse,
        )
        df.append(ret)
    if df:
        df = pd.DataFrame(df)
        df_md = df.to_markdown(index=False)
        aiter.logger.info(
            "mla_persistent CP round-robin summary (markdown):\n%s", df_md
        )
