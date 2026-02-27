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
    qlen,
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

    out_ref_noquant = torch_mha_extend(
        query,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        qo_indptr,
    )

    # out_asm_noquant, us_asm_noquant = run_aiter_asm(
    #     query,
    #     k_cache,
    #     asm_V_shuffle(v_cache),
    #     block_tables,
    #     seq_lens,
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
        seq_lens,
        ctx_lens,
        max_qlen,
        "auto",
        num_kv_heads,
        scale,
    )
    err_noquant = checkAllclose(
        out_ref_noquant,
        out_hip_noquant,
        msg=f"[torch vs  aiter_hip][No Quant]: {us_hip:>8.2f} us......",
    )
    ret["us_hip_bf16"] = us_hip
    ret["err_hip_bf16"] = err_noquant

    # ################## quant start ######################
    k_quant_, k_scale_, v_quant_, v_scale_, k_scale_asm, v_scale_asm = (
        pertoken_quant_kvcache_symm(k_cache, v_cache, quant_dtype=aiter.dtypes.fp8)
    )

    # torch ref
    out_ref = torch_mha_extend(
        query, k_quant_, v_quant_, block_tables, seq_lens, qo_indptr, k_scale_, v_scale_
    )
    out_aiter_asm, us_aiter_asm = run_aiter_asm(
        query,
        k_quant_,
        asm_V_shuffle(v_quant_),
        block_tables,
        seq_lens,
        block_tables.size(1),
        max_qlen,
        k_scale_asm,
        v_scale_asm,
        qo_indptr,
    )
    err = checkAllclose(
        out_ref,
        out_aiter_asm,
        msg=f"[torch vs  aiter_asm][   Quant]: {us_aiter_asm:>8.2f} us......",
    )
    ret["us_asm_fp8"] = us_aiter_asm
    ret["err fp8"] = err

    q_quant, q_scale = pertoken_quant(query, quant_dtype=aiter.dtypes.fp8)
    q_scale = q_scale.squeeze(-1)

    out_hip, us_hip = run_aiter_hip(
        q_quant,
        k_quant_,
        asm_V_shuffle(v_quant_),
        block_tables,
        seq_lens,
        ctx_lens,
        max_qlen,
        "fp8",
        num_kv_heads,
        scale,
        k_scale_asm,
        v_scale_asm,
        q_scale,
        output_dtype=dtype,
    )
    err = checkAllclose(
        out_ref,
        out_hip,
        msg=f"[torch vs  aiter_hip][   Quant]: {us_hip:>8.2f} us......",
    )
    ret["us_hip_fp8"] = us_hip
    ret["err_hip_fp8"] = err

    return ret


head_dim = 128
l_block_size = [16]
l_dtype = ["bf16"]
l_num_heads = [(5, 1), (8, 1), (10, 1)]
l_qlen = [1, 2, 3, 4]
l_ctx_len = [7, 26, 57, 66, 109, 128, 257, 282, 4097, 16384]
l_batch_size = [128]

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
        )
        df.append(ret)
    df = pd.DataFrame(df)
    df_md = df.to_markdown(index=False)
    aiter.logger.info("pa_mtp summary (markdown):\n%s", df_md)
    # df.to_csv("mla_prefill.csv")
