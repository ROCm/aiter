# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import os
import argparse
import random
from typing import List, Optional, Tuple, Union, Dict
import hashlib
import numpy as np

import pandas as pd
import torch
import triton

import aiter
from aiter import dtypes
from aiter import paged_attn as ops
from aiter import pertoken_quant
from aiter.test_common import benchmark, checkAllclose, perftest

from utils import compare_arrays
# from pa_decode_gluon import paged_attention_decode as paged_attention_decode_gluon
# from pa_decode_triton import paged_attention_decode as paged_attention_decode_triton
# from pa_decode_triton_fp8 import paged_attention_decode as paged_attention_decode_triton_fp8
from pa_decode_triton_fp8_2 import paged_attention_decode as paged_attention_decode_triton_fp8
# from pa_decode_triton_fp8_gluon import paged_attention_decode as paged_attention_decode_gluon_fp8
from pa_decode_triton_fp8_gluon_kv_loop import paged_attention_decode as paged_attention_decode_gluon_fp8
import triton.language as tl


# os.environ["TRITON_CACHE_DIR"] = "/home/sijieli2/gluon_cache"
# os.environ["AITER_LOG_MORE"] = "1"
# os.environ["USE_IR_LOC"] = "ttgir"

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)

uniform_range = (-1, 1)
STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
}

tl_to_torch_dtype = {tl.bfloat16: torch.bfloat16, tl.float16: torch.float16}
torch_to_tl_dtype = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}


def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(123)


def get_kv_cache_torch_dtype(
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
) -> torch.dtype:
    if isinstance(cache_dtype, str):
        if cache_dtype == "auto":
            if isinstance(model_dtype, str):
                torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
            elif isinstance(model_dtype, torch.dtype):
                torch_dtype = model_dtype
            else:
                raise ValueError(f"Invalid model dtype: {model_dtype}")
        elif cache_dtype in ["half", "bfloat16", "float"]:
            torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        elif cache_dtype == "fp8":
            torch_dtype = torch.uint8
        else:
            raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    elif isinstance(cache_dtype, torch.dtype):
        torch_dtype = cache_dtype
    else:
        raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    return torch_dtype


def kv_cache_factory(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
    seed: int = 0,
    device: Optional[str] = "cuda",
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

    if cache_dtype == "fp8" and head_size % 16:
        raise ValueError(
            f"Does not support key cache of type fp8 with head_size {head_size}"
        )

    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)

    # softmax_scale = head_size**-0.5
    x = 16 // torch_dtype.itemsize
    k_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    k_caches: List[torch.Tensor] = []
    for _ in range(num_layers):
        k_cache = torch.empty(size=k_cache_shape, dtype=torch_dtype, device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            k_cache.uniform_(*uniform_range)
        else:
            raise ValueError(f"Does not support key cache of type {cache_dtype}")
        k_caches.append(k_cache)

    v_cache_shape = (num_blocks, num_heads, head_size, block_size)
    v_caches: List[torch.Tensor] = []
    for _ in range(num_layers):
        v_cache = torch.empty(size=v_cache_shape, dtype=torch_dtype, device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            v_cache.uniform_(*uniform_range)
        else:
            raise ValueError(f"Does not support value cache of type {cache_dtype}")
        v_caches.append(v_cache)
    return k_caches, v_caches


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: float,
    dtype,
    is_causal=True,
) -> torch.Tensor:
    h_q = query.shape[1]
    h_kv = key.shape[1]
    key = key.repeat_interleave(h_q // h_kv, dim=1)
    value = value.repeat_interleave(h_q // h_kv, dim=1)
    attn_weights = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * softmax_scale
    if is_causal:
        s_q = query.shape[0]
        s_k = key.shape[0]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weights += attn_bias
    attn_weights = torch.softmax(attn_weights, dim=-1)

    out = torch.einsum("hqk,khd->qhd", attn_weights.float(), value.float())
    return out.to(dtype)


def torch_mha_extend(
    q,  # [total_q, nheads, headdim_q]
    k_cache,  # [num_blocks, num_heads, head_size // x, block_size, x]
    v_cache,  # [num_blocks, num_heads, head_size, block_size]
    block_tables,
    seq_lens,
    qo_indptr,
    k_scale=None,  # [num_heads, num_blocks * block_size]
    v_scale=None,  # [num_heads, num_blocks * block_size]
):
    num_blocks, num_heads, head_size, block_size = v_cache.shape
    sm_scale = 1.0 / (head_size**0.5)

    dtype = q.dtype
    kv_dtype = k_cache.dtype
    qs = torch.tensor_split(q, qo_indptr.tolist()[1:])

    # (num_blocks, num_heads, head_size // x, block_size, x)
    k_cache = k_cache.permute(0, 3, 1, 2, 4).contiguous().view(-1, num_heads, head_size)
    # (num_blocks, num_heads, head_size, block_size)
    v_cache = v_cache.permute(0, 3, 1, 2).contiguous().view(-1, num_heads, head_size)

    bs = qo_indptr.shape[0] - 1

    os = []
    for i in range(bs):
        q = qs[i]

        block_table = block_tables[i]
        ctx_len = seq_lens[i].item()

        idx = (
            block_table.repeat_interleave(block_size)[:ctx_len] * block_size
            + torch.arange(ctx_len, device=block_table.device) % block_size
        )

        k = k_cache.view(torch.int8)[idx].view(kv_dtype).to(torch.float)
        if k_scale is not None:
            k *= k_scale[:, idx].t().unsqueeze(-1)

        v = v_cache.view(torch.int8)[idx].view(kv_dtype).to(torch.float)
        if v_scale is not None:
            v *= v_scale[:, idx].t().unsqueeze(-1)
        # if i == 0:
        #     print(f"q.shape={q.shape}")
        #     print(f"k.shape={k.shape}")
        #     print(f"v.shape={v.shape}")
        o = ref_masked_attention(q, k, v, sm_scale, dtype, is_causal=True)
        # o = ref_masked_attention(q, k, v, sm_scale, dtype, is_causal=False)
        os.append(o)
    o = torch.concat(os)
    return o


def pertoken_quant_kvcache_symm(
    # [num_blocks, num_heads, head_size // x, block_size, x]
    k_cache: torch.Tensor,
    # [num_blocks, num_heads, head_size, block_size]
    v_cache: torch.Tensor,
    quant_dtype: torch.dtype,  # e.g. torch.float8_e4m3fnuz
    scale_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_blocks = k_cache.shape[0]
    num_heads = k_cache.shape[1]
    head_dim = v_cache.shape[2]
    block_size = v_cache.shape[3]
    # x          = k_cache.shape[4]
    total_tokens = num_blocks * block_size

    # print(f"{k_cache.shape=}{k_cache.stride()=}")
    # print(f"{v_cache.shape=}{v_cache.stride()=}")

    k_cache_permute = (
        k_cache.permute(0, 1, 3, 2, 4)
        .reshape(num_blocks, num_heads, block_size, -1)
        .contiguous()
    )
    v_cache_permute = (
        v_cache.permute(0, 1, 3, 2)
        .reshape(num_blocks, num_heads, block_size, -1)
        .contiguous()
    )

    k_quant, k_scale_asm = pertoken_quant(k_cache_permute, quant_dtype=quant_dtype)
    v_quant, v_scale_asm = pertoken_quant(v_cache_permute, quant_dtype=quant_dtype)

    # NOTE: quant_x and original x could be different
    quant_x = 16 // quant_dtype.itemsize

    k_quant = (
        k_quant.view(num_blocks, num_heads, block_size, head_dim // quant_x, quant_x)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    k_scale = k_scale_asm.permute(1, 0, 2, 3).contiguous().view(num_heads, total_tokens)
    v_quant = (
        v_quant.view(num_blocks, num_heads, block_size, head_dim)
        .permute(0, 1, 3, 2)
        .contiguous()
    )
    v_scale = v_scale_asm.permute(1, 0, 2, 3).contiguous().view(num_heads, total_tokens)

    # print(f"{k_quant.shape=}{k_quant.stride()=}")
    # print(f"{k_scale.shape=}{k_scale.stride()=}")
    # print(f"{v_quant.shape=}{v_quant.stride()=}")
    # print(f"{v_scale.shape=}{v_scale.stride()=}")
    # print(f"k_cache_permute:{k_cache_permute[0, :, :, :]}, k_quant:{k_quant[0, :, :, :, :]}, k_scale:{k_scale[:, 0]}")

    return k_quant, k_scale, v_quant, v_scale, k_scale_asm, v_scale_asm


@perftest()
def run_aiter_asm(
    query,
    k_cache,
    v_cache,
    block_tables,
    seq_lens,
    block_tables_stride0,
    max_qlen,
    k_scale=None,
    v_scale=None,
    qo_indptr=None,
):
    return aiter.pa_fwd_asm(
        query,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        block_tables_stride0,
        max_qlen,
        k_scale,
        v_scale,
        None,
        qo_indptr,
    )


@perftest()
def run_aiter_hip(
    query,
    k_cache,
    v_cache,
    block_tables,
    seq_lens,
    max_seq_len,
    max_qlen,
    kv_cache_dtype,
    num_kv_heads,
    softmax_scale,
    k_scale=None,
    v_scale=None,
    q_scale=None,
):
    return aiter.paged_attn.PagedAttention.forward_decode(
        query,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        max_seq_len,
        kv_cache_dtype,
        num_kv_heads,
        softmax_scale,
        None,
        k_scale,
        v_scale,
        # q_scale=q_scale,
        mtp=max_qlen,
    )


def asm_V_shuffle(VC):
    # [num_blocks, num_kv_heads, head_size, block_size]
    x = 16 // VC.element_size()
    num_blocks, num_kv_heads, head_size, block_size = VC.shape
    VC = VC.view(num_blocks, num_kv_heads, head_size, block_size // x, x)
    # [num_blocks, num_kv_heads, block_size/x, head_size, x]
    VC = VC.permute(0, 1, 3, 2, 4).contiguous()
    # VC = VC.transpose(2, 3).reshape(num_blocks, num_kv_heads, block_size // x, head_size, x).contiguous()
    # VC = VC.transpose(2, 3).contiguous()
    # print(f"VC.shape={VC.shape}")
    # print(f"VC.stride()={VC.stride()}")
    return VC


def run_triton(
    output: torch.Tensor,       # [num_seqs, num_kv_heads*query_grp_sz, head_sz]
    query: torch.Tensor,        # [num_seqs, num_kv_heads*query_grp_sz, head_sz]
    key_cache: torch.Tensor,    # [num_blks, num_kv_heads, head_sz/x, kv_blk_sz, x]
    value_cache: torch.Tensor,  # [num_blks, num_kv_heads, head_sz, kv_blk_sz]
    seq_lens: torch.Tensor,     # [num_seqs]
    block_tables: torch.Tensor, # [num_seqs, max_num_blks_per_seq]
    attn_scale: float,
    max_seq_len: int,
    compute_type,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    num_seq_partitions: int = 0,  # TODO use this below
    alibi_slopes: torch.Tensor = None,
    max_logits: torch.Tensor = None,
    exp_sums: torch.Tensor = None,
    tmp_output: torch.Tensor = None,
) -> None:
    result = paged_attention_decode_triton(
        output,
        query,
        key_cache,
        value_cache,
        seq_lens,
        block_tables,
        attn_scale,
        max_seq_len,
        compute_type,
        k_scale,
        v_scale,
        num_seq_partitions=0,
        alibi_slopes=None,
    )
    return output, result


def run_triton_fp8(
    output: torch.Tensor,       # [num_seqs, num_kv_heads*query_grp_sz, head_sz]
    query: torch.Tensor,        # [num_seqs, num_kv_heads*query_grp_sz, head_sz]
    key_cache: torch.Tensor,    # [num_blks, num_kv_heads, head_sz/x, kv_blk_sz, x]
    value_cache: torch.Tensor,  # [num_blks, num_kv_heads, head_sz, kv_blk_sz]
    seq_lens: torch.Tensor,     # [num_seqs]
    block_tables: torch.Tensor, # [num_seqs, max_num_blks_per_seq]
    attn_scale: float,
    max_seq_len: int,
    compute_type,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    num_seq_partitions: int = 0,  # TODO use this below
    alibi_slopes: torch.Tensor = None,
    max_logits: torch.Tensor = None,
    exp_sums: torch.Tensor = None,
    tmp_output: torch.Tensor = None,
) -> None:
    result = paged_attention_decode_triton_fp8(
        output,
        query,
        key_cache,
        value_cache,
        seq_lens,
        block_tables,
        attn_scale,
        max_seq_len,
        compute_type,
        q_scale,
        k_scale,
        v_scale,
        num_seq_partitions=0,
        alibi_slopes=None,
    )
    return output, result


def run_gluon_fp8(
    output: torch.Tensor,       # [num_seqs, num_kv_heads*query_grp_sz, head_sz]
    query: torch.Tensor,        # [num_seqs, num_kv_heads*query_grp_sz, head_sz]
    key_cache: torch.Tensor,    # [num_blks, num_kv_heads, head_sz/x, kv_blk_sz, x]
    value_cache: torch.Tensor,  # [num_blks, num_kv_heads, head_sz, kv_blk_sz]
    seq_lens: torch.Tensor,     # [num_seqs]
    block_tables: torch.Tensor, # [num_seqs, max_num_blks_per_seq]
    attn_scale: float,
    max_seq_len: int,
    compute_type,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    num_seq_partitions: int = 0,  # TODO use this below
    alibi_slopes: torch.Tensor = None,
    max_logits: torch.Tensor = None,
    exp_sums: torch.Tensor = None,
    tmp_output: torch.Tensor = None,
) -> None:
    result = paged_attention_decode_gluon_fp8(
        output,
        query,
        key_cache,
        value_cache,
        seq_lens,
        block_tables,
        attn_scale,
        max_seq_len,
        compute_type,
        q_scale,
        k_scale,
        v_scale,
        num_seq_partitions=0,
        alibi_slopes=None,
    )
    return output, result


@benchmark()
def test_pa_mtp(
    ctx_lens: int,
    batch_size: int,
    num_heads: Tuple[int, int],
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
    qlen,
    trans_v,
) -> dict:
    ret = {}
    seed = 0
    device = "cuda:0"
    torch.set_default_device(device)
    num_query_heads, num_kv_heads = num_heads

    assert num_query_heads % num_kv_heads == 0
    max_seq_len = 16384
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    num_blocks = max_num_blocks_per_seq * batch_size
    num_blocks_per_seq = (ctx_lens + block_size - 1) // block_size

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int, device=device)
    seq_lens_qo = torch.randint(
        1, 5, (batch_size,), dtype=torch.int, device=device
    ).fill_(qlen)
    # print(seq_lens_qo)
    qo_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)
    total_qo = qo_indptr[-1].item()
    max_qlen = seq_lens_qo.max().item()

    qkv = torch.randn(
        total_qo,
        num_query_heads + 2 * num_kv_heads,
        head_size,
        dtype=dtype,
    )
    query, key, value = torch.split(
        qkv, [num_query_heads, num_kv_heads, num_kv_heads], dim=1
    )
    query.uniform_(*uniform_range)

    # seq_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(batch_size)]
    seq_lens = [ctx_lens for _ in range(batch_size)]
    seq_lens = torch.tensor(seq_lens, dtype=torch.int)

    # Create the block tables.
    block_tables_lst: List[List[int]] = []
    for _ in range(batch_size):
        block_table = [
            random.randint(0, num_blocks - 1) for _ in range(num_blocks_per_seq)
        ]
        block_tables_lst.append(block_table)

    block_tables = torch.tensor(block_tables_lst, dtype=torch.int)

    # Create the KV caches.
    k_caches, v_caches = kv_cache_factory(
        num_blocks,
        block_size,
        1,
        num_kv_heads,
        head_size,
        "auto",
        dtype,
        seed,
        device,
    )
    k_cache, v_cache = k_caches[0], v_caches[0]
    softmax_scale = float(1.0 / (head_size ** 0.5))


    out_ref_noquant = torch_mha_extend(
        query,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        qo_indptr,
    )


    out_hip_noquant, us_hip = run_aiter_hip(
        query,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        ctx_lens,
        max_qlen,
        "auto",
        num_kv_heads,
        softmax_scale,
    )
    err_hip_noquant = checkAllclose(
        out_ref_noquant,
        out_hip_noquant,
        msg=f"[torch vs aiter_hip][No Quant]: {us_hip:>8.2f} us......",
    )
    compare_arrays(out_hip_noquant.to(torch.float32).detach().cpu().numpy(), out_ref_noquant.to(torch.float32).detach().cpu().numpy())
    ret["us_hip_bf16"] = us_hip
    # ret["err_hip_bf16"] = err_hip_noquant


    # triton_output = torch.empty_like(out_ref_noquant)
    # triton_output, us_triton = run_triton(
    #     triton_output,
    #     query,
    #     k_cache,
    #     v_cache,
    #     seq_lens,
    #     block_tables,
    #     softmax_scale,
    #     seq_lens.max().item(),
    #     torch_to_tl_dtype[dtype],
    #     k_scale=torch.tensor(1.0, device=query.device, dtype=torch.float32),
    #     v_scale=torch.tensor(1.0, device=query.device, dtype=torch.float32),
    #     num_seq_partitions=0,
    #     alibi_slopes=None,
    # )
    # us_triton = us_triton['triton']
    # err_triton_noquant = checkAllclose(
    #     out_ref_noquant,
    #     triton_output,
    #     msg=f"[torch vs triton][No Quant]: {us_triton:>8.2f} us......",
    # )
    # compare_arrays(triton_output.to(torch.float32).detach().cpu().numpy(), out_ref_noquant.to(torch.float32).detach().cpu().numpy())
    # ret["us_triton_bf16"] = us_triton
    # ret["err_triton_bf16"] = err_triton_noquant
    # out_ref_noquant_md5 = hashlib.md5(out_ref_noquant.contiguous().view(torch.uint8).detach().cpu().numpy().tobytes()).hexdigest()
    # triton_output_md5 = hashlib.md5(triton_output.contiguous().view(torch.uint8).detach().cpu().numpy().tobytes()).hexdigest()
    # print(f"out_ref_noquant_md5={out_ref_noquant_md5}")
    # print(f"triton_output_md5={triton_output_md5}")


    # ################## quant start ######################
    q_quant, q_scale = pertoken_quant(query, quant_dtype=aiter.dtypes.fp8)
    k_quant_, k_scale_, v_quant_, v_scale_, k_scale_asm, v_scale_asm = (
        pertoken_quant_kvcache_symm(k_cache, v_cache, quant_dtype=aiter.dtypes.fp8)
    )


    # print(f"batch_size={batch_size}")
    # print(f"seq_lens={seq_lens}")
    # print(f"dtype={dtype}")
    # print(f"qkv.dtype={qkv.dtype}")
    # print(f"k_cache.dtype={k_cache.dtype}")
    # print(f"q_quant.dtype={q_quant.dtype}")
    # print(f"k_quant_.dtype={k_quant_.dtype}")
    # print(f"k_scale_asm.dtype={k_scale_asm.dtype}")
    # print(f"qkv.shape={qkv.shape}")
    # print(f"query.shape={query.shape}")
    # print(f"k_cache.shape={k_cache.shape}")
    # print(f"v_cache.shape={v_cache.shape}")
    # print(f"q_quant.shape={q_quant.shape}")
    # print(f"q_scale.shape={q_scale.shape}")
    # print(f"k_quant_.shape={k_quant_.shape}")
    # print(f"k_scale_.shape={k_scale_.shape}")
    # print(f"v_quant_.shape={v_quant_.shape}")
    # print(f"v_scale_.shape={v_scale_.shape}")
    # print(f"k_scale_asm.shape={k_scale_asm.shape}")
    # print(f"v_scale_asm.shape={v_scale_asm.shape}")
    # print(f"out_ref_noquant.shape={out_ref_noquant.shape}")


    # q_cvted = q_scale * q_quant.to(torch.float32)
    # k_ref = k_cache.transpose(2, 3).reshape(num_blocks, num_kv_heads, block_size, -1).to(torch.float32)
    # v_ref = v_cache.transpose(2, 3).reshape(num_blocks, num_kv_heads, block_size, -1).to(torch.float32)
    # k_cvted = (k_scale_asm.unsqueeze(2) * k_quant_.to(torch.float32)).transpose(2, 3).reshape(num_blocks, num_kv_heads, block_size, -1)
    # v_cvted = (v_scale_asm.transpose(2, 3) * v_quant_.to(torch.float32)).transpose(2, 3).reshape(num_blocks, num_kv_heads, block_size, -1)
    # compare_arrays(q_cvted.to(torch.float32).detach().cpu().numpy(), query.to(torch.float32).detach().cpu().numpy())
    # compare_arrays(k_cvted.to(torch.float32).detach().cpu().numpy(), k_ref.to(torch.float32).detach().cpu().numpy())
    # compare_arrays(v_cvted.to(torch.float32).detach().cpu().numpy(), v_ref.to(torch.float32).detach().cpu().numpy())

    # query_md5 = hashlib.md5(query.contiguous().view(torch.uint8).detach().cpu().numpy().tobytes()).hexdigest()
    # q_cvted_md5 = hashlib.md5(q_cvted.contiguous().view(torch.uint8).detach().cpu().numpy().tobytes()).hexdigest()
    # k_ref_md5 = hashlib.md5(k_ref.contiguous().view(torch.uint8).detach().cpu().numpy().tobytes()).hexdigest()
    # v_ref_md5 = hashlib.md5(v_ref.contiguous().view(torch.uint8).detach().cpu().numpy().tobytes()).hexdigest()
    # k_cvted_md5 = hashlib.md5(k_cvted.contiguous().view(torch.uint8).detach().cpu().numpy().tobytes()).hexdigest()
    # v_cvted_md5 = hashlib.md5(v_cvted.contiguous().view(torch.uint8).detach().cpu().numpy().tobytes()).hexdigest()
    # print(f"query_md5={query_md5}")
    # print(f"q_cvted_md5={q_cvted_md5}")
    # print(f"k_ref_md5={k_ref_md5}")
    # print(f"v_ref_md5={v_ref_md5}")
    # print(f"k_cvted_md5={k_cvted_md5}")
    # print(f"v_cvted_md5={v_cvted_md5}")


    # q_scale1 = q_scale.clone()
    # k_scale_asm1 = k_scale_asm.clone()
    # v_scale_asm1 = v_scale_asm.clone()
    # q_scale[...] = q_scale1.reshape(-1)[0]
    # k_scale_asm[...] = k_scale_asm1.reshape(-1)[0]
    # v_scale_asm[...] = v_scale_asm1.reshape(-1)[0]
    # query = q_scale * q_quant.to(torch.float32)
    # query = query.to(dtype)

    # quant version torch ref
    out_ref = torch_mha_extend(
        query, k_quant_, v_quant_, block_tables, seq_lens, qo_indptr, k_scale_, v_scale_
    )
    # out_ref = out_ref_noquant
    if trans_v:
        v_quant_ = asm_V_shuffle(v_quant_)
        print(f"trans v_quant_.shape={v_quant_.shape}")
    fp8_diff_thr = 5e-2


    if qlen <= 2:
        triton_fp8_output = torch.empty_like(out_ref_noquant)
        triton_fp8_output, us_triton = run_triton_fp8(
            triton_fp8_output,
            # query,
            # k_cache,
            # v_cache,
            q_quant,
            k_quant_,
            v_quant_,

            seq_lens,
            block_tables,
            softmax_scale,
            seq_lens.max().item(),
            torch_to_tl_dtype[dtype],
            q_scale=q_scale,
            k_scale=k_scale_asm,
            v_scale=v_scale_asm,
            num_seq_partitions=0,
            alibi_slopes=None,
        )
        us_triton = us_triton['triton']
        err_triton_noquant = checkAllclose(
            out_ref,
            triton_fp8_output,
            atol=fp8_diff_thr,
            rtol=fp8_diff_thr,
            msg=f"[torch vs triton_fp8][   Quant]: {us_triton:>8.2f} us......",
        )
        compare_arrays(triton_fp8_output.to(torch.float32).detach().cpu().numpy(), out_ref.to(torch.float32).detach().cpu().numpy())
        # compare_arrays(triton_fp8_output.to(torch.float32).detach().cpu().numpy(), out_ref_noquant.to(torch.float32).detach().cpu().numpy())
        ret["us_triton_fp8"] = us_triton
        # ret["err_triton_fp8"] = err_triton_noquant
        out_ref_md5 = hashlib.md5(out_ref.contiguous().view(torch.uint8).detach().cpu().numpy().tobytes()).hexdigest()
        triton_fp8_output_md5 = hashlib.md5(triton_fp8_output.contiguous().view(torch.uint8).detach().cpu().numpy().tobytes()).hexdigest()
        print(f"out_ref_md5={out_ref_md5}")
        print(f"triton_fp8_output_md5={triton_fp8_output_md5}")


    gluon_fp8_output = torch.empty_like(out_ref_noquant)
    gluon_fp8_output, us_triton = run_gluon_fp8(
        gluon_fp8_output,
        # query,
        # k_cache,
        # v_cache,
        q_quant,
        k_quant_,
        v_quant_,

        seq_lens,
        block_tables,
        softmax_scale,
        seq_lens.max().item(),
        torch_to_tl_dtype[dtype],
        q_scale=q_scale,
        k_scale=k_scale_asm,
        v_scale=v_scale_asm,
        # q_scale=q_scale.reshape(-1)[0],
        # k_scale=k_scale_asm.reshape(-1)[0],
        # v_scale=v_scale_asm.reshape(-1)[0],
        num_seq_partitions=0,
        alibi_slopes=None,
    )
    us_triton = us_triton['triton']
    err_triton_noquant = checkAllclose(
        out_ref,
        gluon_fp8_output,
        atol=fp8_diff_thr,
        rtol=fp8_diff_thr,
        msg=f"[torch vs gluon_fp8][   Quant]: {us_triton:>8.2f} us......",
    )
    compare_arrays(gluon_fp8_output.to(torch.float32).detach().cpu().numpy(), out_ref.to(torch.float32).detach().cpu().numpy())
    ret["us_gluon_fp8"] = us_triton
    ret["err_gluon_fp8"] = err_triton_noquant
    out_ref_md5 = hashlib.md5(out_ref.contiguous().view(torch.uint8).detach().cpu().numpy().tobytes()).hexdigest()
    gluon_fp8_output_md5 = hashlib.md5(gluon_fp8_output.contiguous().view(torch.uint8).detach().cpu().numpy().tobytes()).hexdigest()
    print(f"out_ref_md5={out_ref_md5}")
    print(f"gluon_fp8_output_md5={gluon_fp8_output_md5}")
    kt_us = us_triton
    bandwith = batch_size * head_size * (2 * ctx_lens * num_kv_heads * k_quant_.dtype.itemsize + 2 * qlen * num_query_heads * q_quant.dtype.itemsize) / (kt_us * 1e6 * 1.024 ** 4)
    ret["gluon_fp8_bandwith(TB/s)"] = bandwith


    if not(block_size == 1024 and num_heads != (10, 1)) and not(block_size == 16 and num_heads == (8, 1) and qlen == 3) \
        and block_size != 64:
        out_aiter_asm, us_aiter_asm = run_aiter_asm(
            query,
            k_quant_,
            v_quant_,
            block_tables,
            seq_lens,
            block_tables.size(1),
            max_qlen,
            k_scale_asm,
            v_scale_asm,
            # k_scale_asm.reshape(-1)[0],
            # v_scale_asm.reshape(-1)[0],
            qo_indptr,
        )
        err = checkAllclose(
            out_ref,
            out_aiter_asm,
            atol=fp8_diff_thr,
            rtol=fp8_diff_thr,
            msg=f"[torch vs aiter_asm][   Quant]: {us_aiter_asm:>8.2f} us......",
        )
        compare_arrays(out_aiter_asm.to(torch.float32).detach().cpu().numpy(), out_ref.to(torch.float32).detach().cpu().numpy())
        ret["us_asm_fp8"] = us_aiter_asm
        # ret["err fp8"] = err
        kt_us = us_aiter_asm
        bandwith = batch_size * head_size * (2 * ctx_lens * num_kv_heads * k_quant_.dtype.itemsize + 2 * qlen * num_query_heads * query.dtype.itemsize) / (kt_us * 1e6 * 1.024 ** 4)
        ret["asm_fp8_bandwith(TB/s)"] = bandwith


    q_scale = q_scale.squeeze(-1)
    out_hip, us_hip = run_aiter_hip(
        # q_quant.to(torch.bfloat16),
        query,
        k_quant_,
        v_quant_,
        block_tables,
        seq_lens,
        ctx_lens,
        max_qlen,
        "fp8",
        num_kv_heads,
        softmax_scale,
        k_scale_asm,
        v_scale_asm,
        q_scale,
    )
    err = checkAllclose(
        out_ref,
        out_hip,
        atol=fp8_diff_thr,
        rtol=fp8_diff_thr,
        msg=f"[torch vs aiter_hip_fp8][   Quant]: {us_hip:>8.2f} us......",
    )
    compare_arrays(out_hip.to(torch.float32).detach().cpu().numpy(), out_ref.to(torch.float32).detach().cpu().numpy())
    ret["us_hip_fp8"] = us_hip
    # ret["err_hip_fp8"] = err


    if "us_hip_fp8" in ret:
        ret["perf_fp8_gluon_vs_hip"] = f'{ret["us_hip_fp8"] / ret["us_gluon_fp8"]:.0%}'
    else:
        ret["perf_fp8_gluon_vs_hip"] = 'NaN'
    if "us_asm_fp8" in ret:
        ret["perf_fp8_gluon_vs_asm"] = f'{ret["us_asm_fp8"] / ret["us_gluon_fp8"]:.0%}'
    else:
        ret["perf_fp8_gluon_vs_asm"] = 'NaN'
    print(f"triton={triton}")
    print(f"triton.version={triton.__version__}")

    return ret


head_dim = 128
# block_size_list = [16, 64, 128, 256, 512, 1024]
block_size_list = [16, 64, 1024]
l_dtype = ["bf16"]
# l_num_heads = [(5, 1), (8, 1), (10, 1), (16, 1), (64, 1), (8, 2), (64, 4)]
# l_num_heads = [(5, 1), (8, 1), (10, 1), (16, 1)]
l_num_heads = [(8, 1), (10, 1), (16, 1)]
l_qlen = [1, 2, 3, 4]
# l_qlen = [1, 2, 4]
# l_ctx_len = [7, 26, 57, 66, 109, 128, 256, 257, 282, 512, 513, 4096, 4097]
# l_ctx_len = [512, 2048, 4096, 8192, 4097, 8193]
l_ctx_len = [4096]
# l_batch_size = [4, 32, 80, 128]
l_batch_size = [80, 128]

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=l_dtype,
    nargs="?",
    const=None,
    default=None,
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-n",
    "--num_heads",
    type=dtypes.str2tuple,
    default=None,
    help="""Number of heads.
    e.g. -n 8,1""",
)
parser.add_argument(
    "-q",
    "--qlen",
    type=int,
    choices=l_qlen,
    default=None,
    help="""Query length.
    e.g. -q 1""",
)
parser.add_argument(
    "-c",
    "--ctx_len",
    type=int,
    choices=l_ctx_len,
    default=None,
    help="""Context length.
    e.g. -c 128""",
)
parser.add_argument(
    "-b",
    "--batch_size",
    type=int,
    choices=l_batch_size,
    default=None,
    help="""Batch size.
    e.g. -b 128""",
)
parser.add_argument(
    "--block_size",
    type=int,
    choices=block_size_list,
    default=None,
    help="""Batch size.
    e.g. --block_size 16""",
)
parser.add_argument(
    "--trans_v",
    action="store_true",
    help="""e.g. --trans_v""",
)

args = parser.parse_args()
if args.dtype is None:
    l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
else:
    l_dtype = [dtypes.d_dtypes[args.dtype]]
if args.num_heads is not None:
    l_num_heads = [args.num_heads]
if args.qlen is not None:
    l_qlen = [args.qlen]
if args.ctx_len is not None:
    l_ctx_len = [args.ctx_len]
if args.batch_size is not None:
    l_batch_size = [args.batch_size]
if args.block_size is not None:
    block_size_list = [args.block_size]

df = []
for dtype in l_dtype:
    for block_size in block_size_list:
        for num_heads in l_num_heads:
            for ctx_len in l_ctx_len:
                for batch_size in l_batch_size:
                    for qlen in l_qlen:
                        ret = test_pa_mtp(
                            ctx_len,
                            batch_size,
                            num_heads,
                            head_dim,
                            block_size,
                            dtype,
                            qlen,
                            args.trans_v,
                        )
                        df.append(ret)
df = pd.DataFrame(df)
aiter.logger.info(f"summary:\n{df}")
file_name = "pa_gluon_fp8.csv"
if args.trans_v:
    file_name = "pa_gluon_fp8_trans_v.csv"
df.to_csv(file_name)
