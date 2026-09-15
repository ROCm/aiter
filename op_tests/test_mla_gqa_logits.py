# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MLA decode GQA-head sweep for persistent LEGACY layout.

Sweeps nhead in {32, 64, 96, 128}. KV is a single latent head (nhead_kv=1), so
GQA ratio == nhead. Page tables are fragmented (randperm); each sequence owns
its pages. Prefill, non-persistent decode, 3BUFFER and DS32_OPUS are out of
scope. KV length is uniform across the batch.

Two phases (``-p``), matching the round-robin CP test in
op_tests/test_mla_persistent_round_robin.py:

  decode  aiter.mla_decode_fwd vs torch_mla_extend
  cp      round-robin context-parallel: per-rank shard + online-softmax merge

Examples:
    python op_tests/test_mla_gqa_logits.py
    python op_tests/test_mla_gqa_logits.py -p decode -n 32 64 -b 16 64 -c 4096
    python op_tests/test_mla_gqa_logits.py -p cp -cpw 4 -n 32 -mtp 1 -c 64 -b 1
    python op_tests/test_mla_gqa_logits.py -d fp8 -kvd fp8 -n 32 64 96 128
    python op_tests/test_mla_gqa_logits.py -p decode --ref
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)

SUPPORTED_GFX = ["gfx942", "gfx950"]


def check_support(dtype, kv_dtype, nhead):
    if dtype != kv_dtype:
        return False
    return not (dtype == dtypes.bf16 and nhead == 32 and get_gfx() == "gfx942")


def cal_diff(
    x: torch.Tensor, y: torch.Tensor, name: str, use_fp8: bool = False
) -> None:
    x, y = x.double(), y.double()
    # RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    if use_fp8:
        assert cos_diff < 3e-2
    else:
        assert cos_diff < 1e-5


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
        attn_bias = torch.zeros_like(attn_weights)
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
    l = attn_weights_exp.sum(-1)
    if is_fp8_q:
        attn_weights_fp8 = attn_weights_exp.to(dtypes.fp8)
        attn_weights_exp = attn_weights_fp8.to(torch.float)

    out = torch.einsum("hqk,khd->qhd", attn_weights_exp.float(), value.float())
    out = out / l.transpose(0, 1).unsqueeze(-1)
    if is_fp8_kvc and kv_scale is not None:
        out *= kv_scale

    if attn_mask is not None:
        invalid = attn_mask.any(dim=-1).logical_not()
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
    _num_page, page_size, _nhead_kv, _ = kvc_cache.shape
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
    q,
    kvc_cache,
    qo_indptr,
    kv_indptr_r,
    kv_indices_r,
    g_kv_indptr,
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
    """Round-robin CP reference for one rank. page_size must be 1."""
    dev = kvc_cache.device
    is_fp8_q = q.dtype == dtypes.fp8
    is_fp8_kvc = kvc_cache.dtype == dtypes.fp8
    if is_fp8_q:
        q = q.to(torch.float)
    if is_fp8_kvc:
        kvc_cache = kvc_cache.to(torch.float)

    qs = torch.tensor_split(q, qo_indptr.tolist()[1:])
    kvc = torch.index_select(kvc_cache, 0, kv_indices_r)
    indptr_r = kv_indptr_r.tolist()
    g_indptr = g_kv_indptr.tolist()
    bs = qo_indptr.shape[0] - 1

    os = []
    lses = []
    for i in range(bs):
        q_i = qs[i]
        s_q, nheads, _ = q_i.shape
        p0, p1 = int(indptr_r[i]), int(indptr_r[i + 1])
        s_k = (p1 - p0) * page_size

        if s_k == 0:
            os.append(torch.zeros(s_q, nheads, kv_lora_rank, dtype=dtype, device=dev))
            lses.append(torch.full((nheads, s_q), float("-inf"), device=dev))
            continue

        local_kv = kvc[p0:p1].flatten(0, 1)[:s_k]
        k = local_kv
        v, _ = torch.split(local_kv, [kv_lora_rank, qk_rope_head_dim], dim=-1)

        local_global_pos = torch.arange(s_k, device=dev) * cp_world_size + cp_rank
        global_len = (int(g_indptr[i + 1]) - int(g_indptr[i])) * page_size
        q_global = (global_len - s_q) + torch.arange(s_q, device=dev)
        attn_mask = local_global_pos[None, :] <= q_global[:, None]

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
    LS = torch.stack([lse.float() for lse in cp_lses], 0)
    glse = torch.logsumexp(LS, 0)
    w = torch.exp(LS - glse).nan_to_num_(0.0)
    out = sum(w[r][..., None] * cp_outs[r].float() for r in range(len(cp_outs)))
    return out.to(out_dtype), glse


def aiter_cp_rank_decode(
    q,
    kv_buffer,
    qo_indptr,
    kv_indptr_r,
    kv_indices_r,
    g_kv_indptr,
    kv_last_page_lens,
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
    q_scale=None,
    kv_scale=None,
):
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
        dtype_q_nope=dtype,
        dtype_kv_nope=kvtype,
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
        q_scale=q_scale,
        kv_scale=kv_scale,
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
def test_mla_gqa_decode(
    ctx_lens,
    batch_size,
    nhead,
    kv_lora_rank,
    qk_nope_head_dim,
    qk_rope_head_dim,
    v_head_dim,
    dtype,
    kvtype,
    page_size,
    decode_qlen,
    max_split_per_batch,
    return_lse,
    causal,
    check_ref=False,
):
    """Persistent LEGACY paged MLA decode for one (B, N, nhead) shape."""
    ret = {}
    out_dtype = torch.bfloat16

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    seq_lens_qo = torch.empty(batch_size, dtype=torch.int)
    seq_lens_kv = torch.empty(batch_size, dtype=torch.int)
    kv_block_nums = torch.empty(batch_size, dtype=torch.int)
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
    seq_lens_kv.fill_(ctx_lens)
    kv_block_nums.fill_((ctx_lens + page_size - 1) // page_size)
    if ctx_lens % page_size == 0:
        kv_last_page_lens.fill_(page_size)
    else:
        kv_last_page_lens.fill_(ctx_lens % page_size)

    kv_indptr[1 : batch_size + 1] = torch.cumsum(kv_block_nums, dim=0)
    num_page = kv_indptr[-1].item()
    kv_indices = torch.randperm(num_page, dtype=torch.int)
    total_kv = seq_lens_kv.sum().item()

    kv_buffer = torch.randn(
        (num_page, page_size, 1, kv_lora_rank + qk_rope_head_dim),
        dtype=torch.bfloat16,
    )

    qk_head_dim = kv_lora_rank + qk_rope_head_dim
    sm_scale = 1.0 / (qk_head_dim**0.5)
    torch.cuda.empty_cache()
    nhead_kv = 1

    seq_lens_qo.fill_(decode_qlen)
    max_seqlen_qo = seq_lens_qo.max().item()
    qo_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)
    total_q = qo_indptr[-1].item()
    q = torch.randn((total_q, nhead, qk_head_dim), dtype=torch.bfloat16)

    out_ref = lse_ref = None
    if check_ref:
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
            is_causal=causal,
            dtype=out_dtype,
        )

    if nhead >= 128:
        gpu = torch.cuda.current_device()
        device_properties = torch.cuda.get_device_properties(gpu)
        cu_num = device_properties.multi_processor_count
        max_split_per_batch = min(
            (cu_num + batch_size - 1) // batch_size, max_split_per_batch
        )

    (
        (work_meta_data_size, work_meta_data_type),
        (work_indptr_size, work_indptr_type),
        (work_info_set_size, work_info_set_type),
        (reduce_indptr_size, reduce_indptr_type),
        (reduce_final_map_size, reduce_final_map_type),
        (reduce_partial_map_size, reduce_partial_map_type),
    ) = aiter.get_mla_metadata_info_v1(
        batch_size,
        max_seqlen_qo,
        nhead,
        dtype,
        kvtype,
        is_sparse=False,
        fast_mode=True,
        num_kv_splits=max_split_per_batch,
        intra_batch_mode=False,
    )

    work_meta_data = torch.empty(
        work_meta_data_size, dtype=work_meta_data_type, device="cuda"
    )
    work_indptr = torch.empty(work_indptr_size, dtype=work_indptr_type, device="cuda")
    work_info_set = torch.empty(
        work_info_set_size,
        dtype=work_info_set_type,
        device="cuda",
    )
    reduce_indptr = torch.empty(
        reduce_indptr_size, dtype=reduce_indptr_type, device="cuda"
    )
    reduce_final_map = torch.empty(
        reduce_final_map_size, dtype=reduce_final_map_type, device="cuda"
    )
    reduce_partial_map = torch.empty(
        reduce_partial_map_size, dtype=reduce_partial_map_type, device="cuda"
    )

    aiter.get_mla_metadata_v1(
        qo_indptr,
        kv_indptr,
        kv_last_page_lens,
        nhead // nhead_kv,
        nhead_kv,
        causal,
        work_meta_data,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        page_size=page_size,
        kv_granularity=max(
            page_size,
            (
                32
                if (nhead == 64 and dtype == dtypes.fp8 and kvtype == dtypes.fp8)
                else 16
            ),
        ),
        max_seqlen_qo=int(max_seqlen_qo),
        uni_seqlen_qo=decode_qlen,
        fast_mode=True,
        max_split_per_batch=max_split_per_batch,
        intra_batch_mode=False,
        dtype_q_nope=dtype,
        dtype_kv_nope=kvtype,
    )

    def test_absorb_decode_bf16():
        out_asm = torch.empty((total_q, nhead, v_head_dim), dtype=out_dtype).fill_(-1)
        (_attn_logits, attn_lse), us_asm_decode = run_perftest(
            aiter.mla.mla_decode_fwd,
            q,
            kv_buffer.view(num_page, page_size, nhead_kv, qk_head_dim),
            out_asm,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            max_seqlen_qo,
            page_size,
            nhead_kv,
            sm_scale,
            num_kv_splits=max_split_per_batch,
            work_meta_data=work_meta_data,
            work_indptr=work_indptr,
            work_info_set=work_info_set,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
            intra_batch_mode=False,
            return_lse=return_lse,
            causal=causal,
        )

        err = None
        if check_ref:
            err = checkAllclose(
                out_ref,
                out_asm,
                msg=f"mla_decode-absorb    [golden vs aiter_asm]: {us_asm_decode:>8.2f} us......",
            )
            if return_lse:
                checkAllclose(
                    lse_ref,
                    attn_lse.reshape(total_q, nhead),
                    msg=f"mla_decode-absorb    [lse_ref vs attn_lse]: {us_asm_decode:>8.2f} us......",
                )
        else:
            aiter.logger.info("mla_decode-absorb    [no-ref] %8.2f us", us_asm_decode)
        return err, us_asm_decode

    def test_absorb_decode_fp8():
        out_asm = torch.empty((total_q, nhead, v_head_dim), dtype=out_dtype).fill_(-1)
        q_fp8 = q.to(dtypes.fp8)
        q_scale = torch.ones([1], dtype=torch.float, device="cuda")
        kv_buffer_fp8 = kv_buffer.to(dtypes.fp8)
        kv_scale = torch.ones([1], dtype=torch.float, device="cuda")

        out_ref_fp8 = None
        if check_ref:
            out_ref_fp8, _lse_ref_fp8 = torch_mla_extend(
                q_fp8,
                kv_buffer_fp8,
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_last_page_lens,
                sm_scale,
                kv_lora_rank,
                qk_rope_head_dim,
                dtype=out_dtype,
                is_causal=causal,
                q_scale=None,
                kv_scale=kv_scale,
            )

        (_attn_logits, attn_lse), us_asm_decode = run_perftest(
            aiter.mla.mla_decode_fwd,
            q_fp8,
            kv_buffer_fp8.view(num_page, page_size, nhead_kv, qk_head_dim),
            out_asm,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            max_seqlen_qo,
            page_size,
            nhead_kv,
            sm_scale,
            num_kv_splits=max_split_per_batch,
            q_scale=q_scale,
            kv_scale=kv_scale,
            work_meta_data=work_meta_data,
            work_indptr=work_indptr,
            work_info_set=work_info_set,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
            intra_batch_mode=False,
            return_lse=return_lse,
            causal=causal,
        )

        err = None
        if check_ref:
            err = checkAllclose(
                out_ref,
                out_asm,
                msg=f"mla_decode-absorb_fp8    [golden vs aiter_asm]: {us_asm_decode:>8.2f} us......",
            )
            if return_lse:
                err = checkAllclose(
                    lse_ref,
                    attn_lse.reshape(total_q, nhead),
                    msg=f"mla_decode-absorb_fp8    [lse_ref vs attn_lse]: {us_asm_decode:>8.2f} us......",
                )
            err = checkAllclose(
                out_ref_fp8,
                out_asm,
                msg=f"mla_decode-absorb_fp8    [golden fp8 vs aiter_asm]: {us_asm_decode:>8.2f} us......",
            )
            cal_diff(out_ref, out_asm, "out", True)
        else:
            aiter.logger.info(
                "mla_decode-absorb_fp8    [no-ref] %8.2f us", us_asm_decode
            )
        return err, us_asm_decode

    err = None
    us_asm_decode = 1e12
    if dtype == torch.bfloat16:
        err, us_asm_decode = test_absorb_decode_bf16()
    elif kvtype == dtypes.fp8:
        err, us_asm_decode = test_absorb_decode_fp8()

    ret["decode:err"] = err
    ret["decode:asm_576"] = us_asm_decode
    flops = decode_qlen * total_kv * nhead * (qk_head_dim + v_head_dim) * 2
    nbytes = (
        total_kv * nhead_kv * qk_head_dim * (torch.finfo(kvtype).bits // 8)
        + total_q * nhead * qk_head_dim * (torch.finfo(dtype).bits // 8)
        + total_q * nhead * v_head_dim * (torch.finfo(out_dtype).bits // 8)
    )
    ret["decode:flops"] = flops
    ret["decode:bytes"] = nbytes
    ret["decode:TFLOPS"] = flops / us_asm_decode / 1e6
    ret["decode:TB/s"] = nbytes / us_asm_decode / 1e6
    return ret


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
    decode_qlen,
    cp_world_size,
    max_split_per_batch,
    return_lse=False,
    check_ref=False,
):
    """Fixed-length round-robin CP decode. page_size is forced to 1."""
    ret = {}
    W = cp_world_size
    dev = "cuda"
    out_dtype = torch.bfloat16
    nhead_kv = 1
    qlen = decode_qlen
    qk_head_dim = kv_lora_rank + qk_rope_head_dim
    sm_scale = 1.0 / (qk_head_dim**0.5)
    is_causal = qlen > 1
    page_size = 1

    kv_block_nums = torch.empty(batch_size, dtype=torch.int)
    seq_lens_kv = torch.empty(batch_size, dtype=torch.int)
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
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

    q_scale = (
        torch.ones([1], dtype=torch.float, device=dev) if dtype == dtypes.fp8 else None
    )
    kv_scale = (
        torch.ones([1], dtype=torch.float, device=dev) if kvtype == dtypes.fp8 else None
    )

    out_ref = lse_ref = None
    if check_ref:
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
            q_scale=q_scale,
            kv_scale=kv_scale,
        )

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
            pos = pos[pos % W == r]
            idx_r_list.append(kv_indices_dev[start + pos])
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

    cp_outs, cp_lses = [], []
    aiter_outs, aiter_lses, rank_us = [], [], []
    for r in range(W):
        kv_indptr_r = rank_kv_indptr_r[r]
        kv_indices_r = rank_kv_indices_r[r]
        kv_last_page_lens_r = rank_kv_last_r[r]
        o_r = l_r = None
        if check_ref:
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
                q_scale=q_scale,
                kv_scale=kv_scale,
            )
            cp_outs.append(o_r)
            cp_lses.append(l_r)

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
            q_scale,
            kv_scale,
        )
        rank_us.append(us)

        if check_ref:
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

    err_ref = err = None
    if check_ref:
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
    else:
        aiter.logger.info(
            "mla_cp_round_robin W=%s qlen=%s [no-ref] mean rank_us=%.2f",
            W,
            qlen,
            sum(rank_us) / max(len(rank_us), 1),
        )
    ret["cp:err_ref"] = err_ref
    ret["cp:err_aiter"] = err
    ret["cp:world_size"] = W
    ret["cp:rank_us"] = sum(rank_us) / max(len(rank_us), 1)
    return ret


def _summarize(name, rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    keep = [
        c
        for c in (
            "nhead",
            "decode_qlen",
            "batch_size",
            "ctx_lens",
            "dtype",
            "kvtype",
            "gfx",
            "decode:err",
            "decode:asm_576",
            "decode:TFLOPS",
            "decode:TB/s",
            "cp:world_size",
            "cp:err_ref",
            "cp:err_aiter",
            "cp:rank_us",
        )
        if c in df.columns
    ]
    aiter.logger.info(
        "%s summary (markdown):\n%s", name, df[keep].to_markdown(index=False)
    )


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("mla_gqa_logits unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-n",
        "--nhead",
        type=int,
        nargs="*",
        default=[32, 64, 96, 128],
        help="""Q heads (GQA ratio vs nhead_kv=1). Persistent LEGACY only.
    e.g.: -n 32 96""",
    )
    parser.add_argument(
        "-c",
        "--ctxLen",
        type=int,
        nargs="*",
        default=[4096, 16384, 65536, 131072],
        help="""KV length N.
    e.g.: -c 8192 131072""",
    )
    parser.add_argument(
        "-b",
        "--batchSize",
        type=int,
        nargs="*",
        default=[16, 32, 64, 128],
        help="""Decode batch. Decode M = batch * decode_qlen.
    e.g.: -b 128 256""",
    )
    parser.add_argument(
        "-mtp",
        "--decode_qlen",
        type=int,
        nargs="*",
        default=[1, 2, 4, 8],
        help="""Decode speculative rows per sequence.
    e.g.: -mtp 1 2""",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp8"]],
        nargs="*",
        default=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp8"]],
        metavar="{bf16, fp8}",
        help="""Data type of Q.
    e.g.: -d bf16 fp8""",
    )
    parser.add_argument(
        "-kvd",
        "--kv_dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp8"]],
        nargs="*",
        default=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp8"]],
        metavar="{bf16, fp8}",
        help="""Data type of KV.
    e.g.: -kvd bf16 fp8""",
    )
    parser.add_argument(
        "-blk",
        "--block_size",
        type=int,
        default=1,
        help="""Paged KV page size (LEGACY layout).
    e.g.: -blk 1""",
    )
    parser.add_argument(
        "-ms",
        "--max_split_per_batch",
        type=int,
        default=32,
        help="""kv seqlens max split num per batch.
    e.g.: -ms 32""",
    )
    parser.add_argument(
        "-k",
        "--kv_lora_rank",
        type=int,
        default=512,
        help="kv lora rank.",
    )
    parser.add_argument(
        "-qn",
        "--qk_nope_head_dim",
        type=int,
        default=512,
        help="qk nope head dim.",
    )
    parser.add_argument(
        "-qr",
        "--qk_rope_head_dim",
        type=int,
        default=64,
        help="qk rope head dim.",
    )
    parser.add_argument(
        "-vh",
        "--v_head_dim",
        type=int,
        default=512,
        help="v head dim.",
    )
    parser.add_argument(
        "-lse",
        "--return_lse",
        action="store_true",
        help="return lse.",
    )
    parser.add_argument(
        "--causal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="causal mask across decode_qlen tokens. Default: True.",
    )
    parser.add_argument(
        "-p",
        "--phase",
        type=str,
        nargs="*",
        choices=["decode", "cp"],
        default=["decode", "cp"],
        help="""Which phases to run.
    e.g.: -p decode
    e.g.: -p cp""",
    )
    parser.add_argument(
        "-cpw",
        "--cp_world_size",
        type=int,
        nargs="*",
        default=[2, 3, 4, 7, 8],
        help="""CP ranks for the round-robin phase. Skipped when ctx < W.
    e.g.: -cpw 4""",
    )
    parser.add_argument(
        "--ref",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compare against torch golden. Default: False. Pass --ref to enable.",
    )
    args = parser.parse_args()

    def _as_list(x):
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x]

    dtypes_q = _as_list(args.dtype)
    dtypes_kv = _as_list(args.kv_dtype)

    gfx = get_gfx()
    if "decode" in args.phase:
        rows = []
        for nhead, decode_qlen, dtype, kvtype, ctx_len, batch_size in itertools.product(
            args.nhead,
            args.decode_qlen,
            dtypes_q,
            dtypes_kv,
            args.ctxLen,
            args.batchSize,
        ):
            if not check_support(dtype, kvtype, nhead):
                aiter.logger.warning(
                    "skip unsupported combo nhead=%s dtype=%s kvtype=%s",
                    nhead,
                    dtype,
                    kvtype,
                )
                continue
            try:
                ret = test_mla_gqa_decode(
                    ctx_len,
                    batch_size,
                    nhead,
                    args.kv_lora_rank,
                    args.qk_nope_head_dim,
                    args.qk_rope_head_dim,
                    args.v_head_dim,
                    dtype,
                    kvtype,
                    args.block_size,
                    decode_qlen=decode_qlen,
                    max_split_per_batch=args.max_split_per_batch,
                    return_lse=args.return_lse,
                    causal=args.causal,
                    check_ref=args.ref,
                )
            except (RuntimeError, AttributeError) as e:
                msg = str(e).lower()
                if not any(
                    s in msg
                    for s in (
                        "out of memory",
                        "hk_mla",
                        "heuristic",
                        "cannot get",
                    )
                ):
                    raise
                aiter.logger.warning(
                    "skip decode nhead=%s mtp=%s dtype=%s ctx=%s b=%s (%s)",
                    nhead,
                    decode_qlen,
                    dtype,
                    ctx_len,
                    batch_size,
                    e,
                )
                torch.cuda.empty_cache()
                continue
            ret["gfx"] = gfx
            rows.append(ret)
            torch.cuda.empty_cache()
        _summarize("mla_gqa_logits decode", rows)

    if "cp" in args.phase:
        if gfx != "gfx950":
            aiter.logger.warning("mla_gqa_logits cp unsupported on %s; skipping", gfx)
        else:
            rows = []
            for (
                nhead,
                decode_qlen,
                dtype,
                kvtype,
                ctx_len,
                batch_size,
                cp_world_size,
            ) in itertools.product(
                args.nhead,
                args.decode_qlen,
                dtypes_q,
                dtypes_kv,
                args.ctxLen,
                args.batchSize,
                args.cp_world_size,
            ):
                if not check_support(dtype, kvtype, nhead):
                    continue
                if ctx_len < cp_world_size:
                    continue
                if dtype == dtypes.fp8 or kvtype == dtypes.fp8:
                    aiter.logger.warning(
                        "skip cp fp8: no cprr heuristic kernel (gqa=%s qseqlen=%s)",
                        nhead,
                        decode_qlen,
                    )
                    continue
                try:
                    ret = test_mla_cp(
                        ctx_len,
                        batch_size,
                        nhead,
                        args.kv_lora_rank,
                        args.qk_rope_head_dim,
                        args.v_head_dim,
                        dtype,
                        kvtype,
                        decode_qlen=decode_qlen,
                        cp_world_size=cp_world_size,
                        max_split_per_batch=args.max_split_per_batch,
                        return_lse=args.return_lse,
                        check_ref=args.ref,
                    )
                except (RuntimeError, AttributeError) as e:
                    msg = str(e).lower()
                    if not any(
                        s in msg
                        for s in (
                            "out of memory",
                            "hk_mla",
                            "heuristic",
                            "cannot get",
                        )
                    ):
                        raise
                    aiter.logger.warning(
                        "skip cp nhead=%s mtp=%s dtype=%s ctx=%s b=%s W=%s (%s)",
                        nhead,
                        decode_qlen,
                        dtype,
                        ctx_len,
                        batch_size,
                        cp_world_size,
                        e,
                    )
                    torch.cuda.empty_cache()
                    continue
                ret["gfx"] = gfx
                rows.append(ret)
                torch.cuda.empty_cache()
            _summarize("mla_gqa_logits cp", rows)


if __name__ == "__main__":
    main()
