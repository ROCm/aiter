# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools
import random
from typing import List, Optional, Tuple, Union

import pandas as pd
import torch

import aiter
from aiter import dtypes, paged_attn as ops
from aiter import pertoken_quant
from aiter.test_common import benchmark, checkAllclose, perftest

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

    # scale = head_size**-0.5
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
    scale: float,
    dtype,
    is_causal=True,
) -> torch.Tensor:
    attn_weights = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * scale
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
        o = ref_masked_attention(q, k, v, sm_scale, dtype, is_causal=True)
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
def run_aiter_asm_ps(
    Q,
    K,
    V,
    output,
    max_qlen,
    qo_indptr,
    kv_indptr,
    kv_indices,
    context_lens,
    K_QScale,
    V_QScale,
    work_indptr,
    work_info,
    reduce_indptr,
    reduce_final_map,
    reduce_partial_map,
    softmax_scale,
    mask,
):
    return aiter.pa_persistent_fwd(
        Q=Q,
        K=K,
        V=V,
        output=output,
        max_qlen=max_qlen,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        context_lens=context_lens,
        K_QScale=K_QScale,
        V_QScale=V_QScale,
        work_indptr=work_indptr,
        work_info=work_info,
        reduce_indptr=reduce_indptr,
        reduce_final_map=reduce_final_map,
        reduce_partial_map=reduce_partial_map,
        softmax_scale=softmax_scale,
        mask=mask,
    )


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
        # kernelName="_ZN5aiter42pa_bf16_pertokenFp8_gqa10_1tg_4w_mtp3_msk1E",
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
    scale,
    k_scale=None,
    v_scale=None,
):
    return ops.PagedAttention.forward_decode(
        query,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        max_seq_len,
        kv_cache_dtype,
        num_kv_heads,
        scale,
        None,
        k_scale,
        v_scale,
        mtp=max_qlen,
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
    scale,
    k_scale=None,
    v_scale=None,
    q_scale=None,
    output_dtype=dtypes.bf16,
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
        scale,
        None,
        k_scale,
        v_scale,
        q_scale=q_scale,
        mtp=max_qlen,
        output_dtype=output_dtype,
    )


def asm_V_shuffle(VC):
    # [num_blocks, num_kv_heads, head_size, block_size]
    x = 16 // VC.element_size()
    num_blocks, num_kv_heads, head_size, block_size = VC.shape
    VC = VC.view(num_blocks, num_kv_heads, head_size, block_size // x, x)
    # [num_blocks, num_kv_heads, block_size/X, head_size, X]
    VC = VC.permute(0, 1, 3, 2, 4).contiguous()
    return VC


@benchmark()
def test_pa_mtp(
    ctx_lens: int,
    batch_size: int,
    num_heads: Tuple[int, int],
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
    qlen: int,
    varlen: bool,
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

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    seq_lens_kv = torch.empty(batch_size, dtype=torch.int)
    if varlen:
        for i in range(batch_size):
            # seq_lens_kv[i] = max(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens)
            seq_lens_kv[i] = random.uniform(5, ctx_lens)
    else:
        seq_lens_kv.fill_(ctx_lens)
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

    out_ref_noquant = torch_mha_extend(
        query,
        k_cache,
        v_cache,
        block_tables,
        seq_lens_kv,
        qo_indptr,
    )

    # out_asm_noquant, us_asm_noquant = run_aiter_asm(
    #     query,
    #     k_cache,
    #     asm_V_shuffle(v_cache),
    #     block_tables,
    #     seq_lens_kv,
    #     block_tables.size(1),
    #     max_qlen,
    #     qo_indptr=qo_indptr,
    # )
    # err_noquant = checkAllclose(
    #     out_ref_noquant,
    #     out_asm_noquant,
    #     msg=f"[torch vs  aiter_asm][No Quant]: {us_asm_noquant:>8.2f} us......",
    # )
    # ret["us_asm_bf16"] = us_asm_noquant
    # ret["err_asm_bf16"] = err_noquant

    scale = float(1.0 / (head_size**0.5))
    out_hip_noquant, us_hip = run_aiter_hip(
        query,
        k_cache,
        v_cache,
        block_tables,
        seq_lens_kv,
        ctx_lens,
        max_qlen,
        "auto",
        num_kv_heads,
        scale,
    )
    # err_noquant = checkAllclose(
    #     out_ref_noquant,
    #     out_hip_noquant,
    #     msg=f"[torch vs  aiter_hip][No Quant]: {us_hip:>8.2f} us......",
    # )
    # ret["us_hip_bf16"] = us_hip
    # ret["err_hip_bf16"] = err_noquant

    # ################## quant start ######################
    k_quant_, k_scale_, v_quant_, v_scale_, k_scale_asm, v_scale_asm = (
        pertoken_quant_kvcache_symm(k_cache, v_cache, quant_dtype=aiter.dtypes.fp8)
    )

    # torch ref
    out_ref = torch_mha_extend(
        query,
        k_quant_,
        v_quant_,
        block_tables,
        seq_lens_kv,
        qo_indptr,
        k_scale_,
        v_scale_,
    )

    # out_asm_noquant, us_asm_noquant = run_aiter_asm(
    #     query,
    #     k_quant_,
    #     asm_V_shuffle(v_quant_),
    #     block_tables,
    #     seq_lens_kv,
    #     block_tables.size(1),
    #     max_qlen,
    #     k_scale=k_scale_,
    #     v_scale=v_scale_,
    #     qo_indptr=qo_indptr,
    # )
    # err_noquant = checkAllclose(
    #     out_ref_noquant,
    #     out_asm_noquant,
    #     msg=f"[torch vs  aiter_asm][No Quant]: {us_asm_noquant:>8.2f} us......",
    # )
    # ret["us_asm_bf16"] = us_asm_noquant
    # ret["err_asm_bf16"] = err_noquant

    if True:
        (
            (work_meta_data_size, work_meta_data_type),
            (work_indptr_size, work_indptr_type),
            (work_info_set_size, work_info_set_type),
            (reduce_indptr_size, reduce_indptr_type),
            (reduce_final_map_size, reduce_final_map_type),
            (reduce_partial_map_size, reduce_partial_map_type),
        ) = aiter.get_pa_metadata_info_v1(
            batch_size,
            max_qlen,
            num_query_heads,
            query.dtype,
            k_quant_.dtype,
            is_sparse=False,
        )
        work_metadata_ptrs = torch.empty(work_meta_data_size, dtype=work_meta_data_type)
        work_indptr = torch.empty(work_indptr_size, dtype=work_indptr_type)
        work_info = torch.empty(work_info_set_size, dtype=work_info_set_type)
        reduce_indptr = torch.empty(reduce_indptr_size, dtype=reduce_indptr_type)
        reduce_final_map = torch.empty(
            reduce_final_map_size, dtype=reduce_final_map_type
        )
        reduce_partial_map = torch.empty(
            reduce_partial_map_size, dtype=reduce_partial_map_type
        )

        # FIX: kv_indptr: seq_lens prefix sum -> actual_blocks prefix sum
        # refer: https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flashinfer.py#L731-L735
        actual_blocks = (seq_lens_kv + block_size - 1) // block_size
        kv_indptr[1 : batch_size + 1] = torch.cumsum(actual_blocks, dim=0)
        # FIX: kv_indices: random -> pack block_table actual blocks
        # refer: https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flashinfer.py#L1545
        kv_indices_lst = []
        for i in range(0, batch_size):
            kv_indices_lst += block_tables_lst[i][: actual_blocks[i]]
        kv_indices = torch.tensor(kv_indices_lst, dtype=torch.int)

        aiter.get_pa_metadata_v1(
            qo_indptr,
            kv_indptr,
            num_query_heads // num_kv_heads,
            num_kv_heads,
            True,
            work_metadata_ptrs,
            work_indptr,
            work_info,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            kv_granularity=max(block_size, 16),
            max_seqlen_qo=int(max_qlen),
            uni_seqlen_qo=qlen,
            fast_mode=True,
            max_split_per_batch=-1,
        )
        print(
            f"batch_size: {batch_size}, num_query_heads: {num_query_heads}, num_kv_heads: {num_kv_heads}"
        )
        print(f"qo_indptr: {qo_indptr.tolist()}")
        print(f"kv_indptr: {kv_indptr.tolist()}")
        print(f"kv_indices: {kv_indices.tolist()}")
        print(f"seq_lens_kv: {seq_lens_kv.tolist()}")

        print(f"==>work_idptr: \n{work_indptr}")
        print(f"==>work_info: \n{work_info}")
        print(f"==>reduce_indptr: \n{reduce_indptr}")
        print(f"==>reduce_final_map: \n{reduce_final_map}")
        print(f"==>reduce_partial_map: \n{reduce_partial_map}")
        torch.set_printoptions(threshold=999999, linewidth=120)

    output = torch.empty_like(query)
    out_aiter_asm, us_aiter_asm = aiter.pa_persistent_fwd(
        Q=query,
        K=k_quant_,
        V=asm_V_shuffle(v_quant_),
        output=output,
        max_qlen=max_qlen,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        context_lens=seq_lens_kv,
        K_QScale=k_scale_,
        V_QScale=v_scale_,
        # work_meta_data=work_metadata_ptrs,
        work_indptr=work_indptr,
        work_info=work_info,
        reduce_indptr=reduce_indptr,
        reduce_final_map=reduce_final_map,
        reduce_partial_map=reduce_partial_map,
        softmax_scale=scale,
        mask=1,
    )
    print("run pa_persistent_fwd successfully")
    print(f"seq_lens_kv: {seq_lens_kv.tolist()}")
    err = checkAllclose(
        out_ref,
        output,
        msg="[torch vs  pa_persistent_fwd][   Quant]: us......",
    )
    ret["us_asm_fp8"] = us_aiter_asm
    ret["err fp8"] = err

    # q_quant, q_scale = pertoken_quant(query, quant_dtype=aiter.dtypes.fp8)
    # q_scale = q_scale.squeeze(-1)

    # out_hip, us_hip = run_aiter_hip(
    #     q_quant,
    #     k_quant_,
    #     asm_V_shuffle(v_quant_),
    #     block_tables,
    #     seq_lens_kv,
    #     ctx_lens,
    #     max_qlen,
    #     "fp8",
    #     num_kv_heads,
    #     scale,
    #     k_scale_asm,
    #     v_scale_asm,
    #     q_scale,
    #     output_dtype=dtype,
    # )
    # err = checkAllclose(
    #     out_ref,
    #     out_hip,
    #     msg=f"[torch vs  aiter_hip][   Quant]: {us_hip:>8.2f} us......",
    # )
    # ret["us_hip_fp8"] = us_hip
    # ret["err_hip_fp8"] = err

    return ret


head_dim = 128
l_block_size = [1024]
l_dtype = ["bf16"]
l_num_heads = [
    (16, 1)
]  # num_query_heads must be multiple of 16 for get_mla_metadata_info_v1
l_qlen = [1]
l_ctx_len = [7, 26, 57, 66, 109, 128, 257, 282, 4097, 16384]
# l_ctx_len = [1024]
l_batch_size = [2]

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
    default=None,
    help="""Context length.
    e.g. -c 128""",
)
parser.add_argument(
    "-b",
    "--batch_size",
    type=int,
    default=None,
    help="""Batch size.
    e.g. -b 128""",
)
parser.add_argument(
    "--block_size",
    type=int,
    nargs="*",
    default=l_block_size,
    help="""Batch size.
    e.g. -b 128""",
)
parser.add_argument(
    "--varlen",
    action="store_true",
    help="""variable kv seqlens per batch. Default: False.
    --varlen # True""",
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
l_block_size = args.block_size
l_varlen = args.varlen

for dtype in l_dtype:
    df = []
    for num_heads, qlen, ctx_len, batch_size, block_size in itertools.product(
        l_num_heads, l_qlen, l_ctx_len, l_batch_size, l_block_size
    ):
        ret = test_pa_mtp(
            ctx_len,
            batch_size,
            num_heads,
            head_dim,
            block_size,
            dtype,
            qlen,
            l_varlen,
        )
        df.append(ret)
    df = pd.DataFrame(df)
    aiter.logger.info(f"summary:\n{df}")
    df.to_csv("mla_prefill.csv")


def test_pa_ps_simple():
    """Test PA persistent scheduling with hardcoded dummy data"""
    print("\n" + "=" * 80)
    print("Testing PA Persistent Scheduling (PS) - Simple Test")
    print("=" * 80)

    device = "cuda:0"
    torch.set_default_device(device)

    # batch_size = 4
    # num_head_q = 16
    # num_head_kv = 1
    # head_dim = 128
    # block_size = 16
    # dtype = dtypes.bf16

    # qo_indptr = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
    # kv_indptr_tokens = torch.tensor([0, 16, 18, 30, 36], dtype=torch.int32)
    # seq_lens_kv = torch.tensor([16, 2, 12, 6], dtype=torch.int32)

    batch_size = 2
    num_head_q = 16
    num_head_kv = 1
    head_dim = 128
    block_size = 1024
    dtype = dtypes.bf16

    qo_indptr = torch.tensor([0, 1, 2], dtype=torch.int32)
    kv_indptr_tokens = torch.tensor([0, 5, 10], dtype=torch.int32)
    seq_lens_kv = torch.tensor([1025, 1025], dtype=torch.int32)
    # 2048 + 16 = 2064

    actual_blocks = (seq_lens_kv + block_size - 1) // block_size
    kv_indptr_blocks = torch.zeros(batch_size + 1, dtype=torch.int32)
    kv_indptr_blocks[1:] = torch.cumsum(actual_blocks, dim=0)

    num_blocks = kv_indptr_blocks[-1].item()
    kv_indices = torch.arange(num_blocks, dtype=torch.int32)

    # This is generate by yang gen metadata cpp
    # work_indptr = torch.tensor([0, 1, 2, 4, 6, 7], dtype=torch.int32)
    # work_info = torch.tensor([
    #     [0,  0, 0,  4,  0,  8, 655360,      0],
    #     [0, 16, 0,  4,  8, 16,      0, 655360],
    #     [1, -1, 4,  6, 16, 18,      0, 655360],
    #     [2, 32, 6, 12, 18, 24, 655360,      0],
    #     [2, 48, 6, 12, 24, 30,      0, 655360],
    #     [3, 64, 12, 15, 30, 32, 655360,      0],
    #     [3, 80, 12, 15, 32, 36,      0, 655360],
    # ], dtype=torch.int32)

    # 2 Not reduce: this is pass
    # work_indptr = torch.tensor([0, 1, 2], dtype=torch.int32)
    # # work_info: (2) [
    # # work[  0]:          0,4294967295,         0,         1,         0,         5,         0,    655360(0,10)
    # # work[  1]:          1,4294967295,         1,         2,         5,        10,         0,    655360(0,10)
    # work_info = torch.tensor([
    #     [0, -1, 0, 1, 0, 5, 0, 655360],
    #     [1, -1, 1, 2, 5, 10, 0, 655360],
    # ], dtype=torch.int32)

    # work_indptr:(6) [0,1,2,2,2,2]
    # work_info: (2) [
    # work[  0]:          0,4294967295,         0,         1,         0,         1,         0,   1048576(0,16)
    # work[  1]:          1,4294967295,         1,         2,         1,         2,         0,   1048576(0,16)
    # ]
    # work_indptr = torch.tensor([0, 1, 2, 2, 2, 2], dtype=torch.int32)
    # work_info = torch.tensor([
    #     [0, -1, 0, 1, 0, 1, 0, 1048576],
    #     [1, -1, 1, 2, 1, 2, 0, 1048576],
    # ], dtype=torch.int32)

    # 3 Reduce
    # work_indptr = torch.tensor([0,1,3,4,5], dtype=torch.int32)
    # work_info = torch.tensor([
    #     [0, 0, 0, 1, 0, 3, 0, 655360],
    #     [0, 1, 0, 1, 3, 5, 0, 655360],
    #     [1, 2, 1, 2, 5, 6, 0, 655360],
    #     [1, 3, 1, 2, 6, 9, 0, 655360],
    #     [1, 4, 1, 2, 9, 10, 0, 655360],
    # ], dtype=torch.int32)

    # 4 reduce
    # work_indptr = torch.tensor([0,1,2], dtype=torch.int32)
    # work_info = torch.tensor([
    #     [0, 0, 0, 1, 0, 1, 0, 1048576],
    #     [0, 1, 0, 1, 1, 2, 0, 1048576],
    # ], dtype=torch.int32)

    # 2
    # reduce_indptr = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.int32)
    # reduce_final_map = torch.tensor([
    #     [0, 4, 0], [0, 4, 655360],
    #     [12, 18, 0], [12, 18, 655360],
    #     [24, 27, 0], [24, 27, 655360],
    # ], dtype=torch.int32)
    # reduce_partial_map = torch.tensor([0, 16, 32, 48, 64, 80], dtype=torch.int32)

    # reduce_indptr = torch.tensor([0], dtype=torch.int32)
    # reduce_final_map = torch.tensor([0], dtype=torch.int32)
    # reduce_partial_map = torch.tensor([0], dtype=torch.int32)

    # 3----------reduce_indptr:
    # reduce_indptr = torch.tensor([0, 2, 5], dtype=torch.int32)
    # reduce_final_map = torch.tensor([
    #     [0, 1],
    #     [1, 2],
    # ], dtype=torch.int32)
    # reduce_partial_map = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)

    # 4 reduce
    # reduce_indptr = torch.tensor([0, 2], dtype=torch.int32)
    # reduce_final_map = torch.tensor([
    #     [0, 1],
    # ], dtype=torch.int32)
    # reduce_partial_map = torch.tensor([0, 1], dtype=torch.int32)

    total_qo = qo_indptr[-1].item()

    print(f"batch_size: {batch_size}, num_heads: {num_head_q}")
    print(f"qo_indptr: {qo_indptr.tolist()}")
    print(f"kv_indptr_blocks: {kv_indptr_blocks.tolist()}")
    print(f"seq_lens_kv: {seq_lens_kv.tolist()}")
    # print(f"work_indptr: {work_indptr.tolist()}")

    Q = torch.randn(total_qo, num_head_q, head_dim, dtype=dtype, device=device)

    k_caches, v_caches = kv_cache_factory(
        num_blocks, block_size, 1, num_head_kv, head_dim, "auto", dtype, 0, device
    )
    k_cache, v_cache = k_caches[0], v_caches[0]

    k_quant_, k_scale_, v_quant_, v_scale_, k_scale_asm, v_scale_asm = (
        pertoken_quant_kvcache_symm(k_cache, v_cache, quant_dtype=aiter.dtypes.fp8)
    )

    scale = head_dim**-0.5
    max_qlen = 1
    output = torch.empty_like(Q)

    # Create block_tables for reference implementation
    block_tables_lst = []
    for i in range(batch_size):
        # Get blocks for this batch
        start_idx = kv_indptr_blocks[i].item()
        end_idx = kv_indptr_blocks[i + 1].item()
        blocks = kv_indices[start_idx:end_idx].tolist()
        # Pad to max blocks per seq if needed
        max_blocks = max(actual_blocks).item()
        blocks += [0] * (max_blocks - len(blocks))
        block_tables_lst.append(blocks)

    block_tables = torch.tensor(block_tables_lst, dtype=torch.int32, device=device)

    print("\n=== Computing reference output ===")
    out_ref = torch_mha_extend(
        Q, k_quant_, v_quant_, block_tables, seq_lens_kv, qo_indptr, k_scale_, v_scale_
    )
    print(f"Reference output shape: {out_ref.shape}")

    print("\n=== Testing PA Persistent Forward ===")
    # out_aiter_asm_ps, us_aiter_asm_ps = run_aiter_asm_ps(
    #     Q,
    #     k_quant_,
    #     asm_V_shuffle(v_quant_),
    #     output,
    #     max_qlen,
    #     qo_indptr,
    #     kv_indptr_blocks,
    #     kv_indices,
    #     seq_lens_kv,
    #     k_scale_,
    #     v_scale_,
    #     work_indptr,
    #     work_info,
    #     reduce_indptr,
    #     reduce_final_map,
    #     reduce_partial_map,
    #     scale,
    # )

    # use work info:

    (
        (work_meta_data_size, work_meta_data_type),
        (work_indptr_size, work_indptr_type),
        (work_info_set_size, work_info_set_type),
        (reduce_indptr_size, reduce_indptr_type),
        (reduce_final_map_size, reduce_final_map_type),
        (reduce_partial_map_size, reduce_partial_map_type),
    ) = aiter.get_pa_metadata_info_v1(
        batch_size,
        max_qlen,
        num_head_q,
        Q.dtype,
        k_quant_.dtype,
        is_sparse=False,
    )
    work_metadata_ptrs = torch.empty(work_meta_data_size, dtype=work_meta_data_type)
    work_indptr = torch.empty(work_indptr_size, dtype=work_indptr_type)
    work_info = torch.empty(work_info_set_size, dtype=work_info_set_type)
    reduce_indptr = torch.empty(reduce_indptr_size, dtype=reduce_indptr_type)
    reduce_final_map = torch.empty(reduce_final_map_size, dtype=reduce_final_map_type)
    reduce_partial_map = torch.empty(
        reduce_partial_map_size, dtype=reduce_partial_map_type
    )

    # FIX: kv_indptr: seq_lens prefix sum -> actual_blocks prefix sum
    # refer: https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flashinfer.py#L731-L735
    actual_blocks = (seq_lens_kv + block_size - 1) // block_size
    kv_indptr_blocks[1 : batch_size + 1] = torch.cumsum(actual_blocks, dim=0)
    # FIX: kv_indices: random -> pack block_table actual blocks
    # refer: https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flashinfer.py#L1545
    kv_indices_lst = []
    for i in range(0, batch_size):
        kv_indices_lst += block_tables_lst[i][: actual_blocks[i]]
    kv_indices = torch.tensor(kv_indices_lst, dtype=torch.int)

    print(f"==> kv_indptr: {kv_indptr_blocks}")
    print(f"==> kv_indices: {kv_indices}")
    print(f"==> context_lens: {seq_lens_kv}")
    print(f"==> K shape: {k_quant_.shape}")

    aiter.get_pa_metadata_v1(
        qo_indptr,
        kv_indptr_blocks,
        num_head_q // num_head_kv,
        num_head_kv,
        True,
        work_metadata_ptrs,
        work_indptr,
        work_info,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        kv_granularity=max(block_size, 16),
        max_seqlen_qo=int(max_qlen),
        uni_seqlen_qo=1,
        fast_mode=True,
        max_split_per_batch=-1,
    )
    # print(f"==>work_idptr: {work_indptr}")
    # print(f"==>work_info: {work_info}")
    # print(f"==>reduce_indptr: {reduce_indptr}")
    # print(f"==>reduce_final_map: {reduce_final_map}")
    # print(f"==>reduce_partial_map: {reduce_partial_map}")
    torch.set_printoptions(threshold=999999, linewidth=120)
    print(f"==>work_idptr:\n {work_indptr}")
    print(f"==>work_info:\n {work_info}")
    print(f"==>reduce_indptr:\n {reduce_indptr}")
    print(f"==>reduce_final_map:\n {reduce_final_map}")
    print(f"==>reduce_partial_map:\n {reduce_partial_map}")
    actual_num_work = work_indptr.max().item()
    work_indptr = work_indptr[: actual_num_work + 1]
    work_info = work_info[:actual_num_work]

    _, us_aiter_asm_ps = run_aiter_asm_ps(
        Q=Q,
        K=k_quant_,
        V=asm_V_shuffle(v_quant_),
        output=output,
        max_qlen=max_qlen,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr_blocks,
        kv_indices=kv_indices,
        context_lens=seq_lens_kv,
        K_QScale=k_scale_,
        V_QScale=v_scale_,
        work_indptr=work_indptr,
        work_info=work_info,
        reduce_indptr=reduce_indptr,
        reduce_final_map=reduce_final_map,
        reduce_partial_map=reduce_partial_map,
        softmax_scale=scale,
        mask=1,
    )

    print("PA PS forward succeeded!")
    print(f"Output shape: {output.shape}")

    err = checkAllclose(
        out_ref,
        output,
        msg="PA PS [torch_ref vs pa_persistent_fwd]",
    )

    print("\n=== Accuracy Check ===")
    print(f"Error: {err}")

    # PA without persistent
    # out_asm_noquant, us_asm_noquant = run_aiter_asm(
    #     Q,
    #     k_quant_,
    #     asm_V_shuffle(v_quant_),
    #     block_tables,
    #     seq_lens_kv,
    #     block_tables.size(1),
    #     max_qlen,
    #     k_scale=k_scale_,
    #     v_scale=v_scale_,
    #     qo_indptr=qo_indptr,
    # )

    # print("Check performance:")
    # print(f"PA PS forward: {us_aiter_asm_ps:>8.2f} us")
    # print(f"PA without persistent: {us_asm_noquant:>8.2f} us")


if __name__ == "__main__":
    test_pa_ps_simple()
