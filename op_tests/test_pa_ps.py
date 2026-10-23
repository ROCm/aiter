# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import hashlib
import itertools
import json
import random
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.profiler as tpf

import aiter
from aiter import dtypes, pertoken_quant
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.attention import _attention_ps_reduce_asm
from aiter.ops.enum import QuantType
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
    cache_dtype: str | torch.dtype | None,
    model_dtype: str | torch.dtype | None = None,
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
        raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")  # noqa: TRY004
    return torch_dtype


def kv_cache_factory(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: str | torch.dtype | None,
    model_dtype: str | torch.dtype | None = None,
    seed: int = 0,
    device: str | None = "cuda",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:

    if cache_dtype == "fp8" and head_size % 16:
        raise ValueError(
            f"Does not support key cache of type fp8 with head_size {head_size}"
        )

    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)

    # scale = head_size**-0.5
    x = 16 // torch_dtype.itemsize
    k_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    k_caches: list[torch.Tensor] = []
    for _ in range(num_layers):
        k_cache = torch.empty(size=k_cache_shape, dtype=torch_dtype, device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            k_cache.uniform_(*uniform_range)
        else:
            raise ValueError(f"Does not support key cache of type {cache_dtype}")
        k_caches.append(k_cache)

    v_cache_shape = (num_blocks, num_heads, head_size, block_size)
    v_caches: list[torch.Tensor] = []
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
    num_query_heads = query.shape[1]
    num_kv_heads = key.shape[1]

    if num_query_heads != num_kv_heads:
        assert num_query_heads % num_kv_heads == 0
        num_groups = num_query_heads // num_kv_heads
        key = key.repeat_interleave(num_groups, dim=1)
        value = value.repeat_interleave(num_groups, dim=1)

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
    k_scale=None,  # [num_heads, num_blocks * block_size] for per-token, [num_blocks, num_heads] for per-block
    v_scale=None,  # [num_heads, num_blocks * block_size] for per-token, [num_blocks, num_heads] for per-block
    is_causal=True,
    quant_type=QuantType.per_Token,
):
    _num_blocks, num_heads, head_size, block_size = v_cache.shape
    sm_scale = 1.0 / (head_size**0.5)

    per_block_quant = quant_type in (
        QuantType.per_1x128,
        QuantType.per_256x128,
        QuantType.per_1024x128,
    )

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
            if per_block_quant:
                # per-block: k_scale is [num_blocks, num_heads]
                block_idx = idx // block_size
                k *= k_scale[block_idx, :].unsqueeze(-1)  # [ctx_len, num_heads, 1]
            else:
                # per-token: k_scale is [num_heads, total_tokens]
                k *= k_scale[:, idx].t().unsqueeze(-1)

        v = v_cache.view(torch.int8)[idx].view(kv_dtype).to(torch.float)
        if v_scale is not None:
            if per_block_quant:
                # per-block: v_scale is [num_blocks, num_heads]
                block_idx = idx // block_size
                v *= v_scale[block_idx, :].unsqueeze(-1)  # [ctx_len, num_heads, 1]
            else:
                # per-token: v_scale is [num_heads, total_tokens]
                v *= v_scale[:, idx].t().unsqueeze(-1)
        o = ref_masked_attention(q, k, v, sm_scale, dtype, is_causal=is_causal)
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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


def perblock_quant_kvcache_symm(
    # [num_blocks, num_heads, head_size // x, block_size, x]
    k_cache: torch.Tensor,
    # [num_blocks, num_heads, head_size, block_size]
    v_cache: torch.Tensor,
    quant_dtype: torch.dtype,  # e.g. torch.float8_e4m3fnuz
    scale_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reshape to [num_blocks, num_heads, -1] and treat as per-token quant.
    """
    num_blocks = k_cache.shape[0]
    num_heads = k_cache.shape[1]
    head_dim = v_cache.shape[2]
    block_size = v_cache.shape[3]

    # [num_blocks, num_heads, head_size // x, block_size, x] -> [num_blocks, num_heads, -1]
    k_cache_flat = k_cache.view(num_blocks, num_heads, -1)
    # [num_blocks, num_heads, head_size, block_size] -> [num_blocks, num_heads, -1]
    v_cache_flat = v_cache.view(num_blocks, num_heads, -1)

    k_cache_flat = k_cache_flat.view(num_blocks * num_heads, -1)
    v_cache_flat = v_cache_flat.view(num_blocks * num_heads, -1)

    k_quant_flat, k_scale_flat = pertoken_quant(k_cache_flat, quant_dtype=quant_dtype)
    v_quant_flat, v_scale_flat = pertoken_quant(v_cache_flat, quant_dtype=quant_dtype)

    # NOTE: quant_x and original x could be different
    quant_x = 16 // quant_dtype.itemsize

    k_quant = k_quant_flat.view(
        num_blocks, num_heads, head_dim // quant_x, block_size, quant_x
    )
    v_quant = v_quant_flat.view(num_blocks, num_heads, head_dim, block_size)

    k_scale_asm = k_scale_flat.view(num_blocks, num_heads).to(scale_dtype)
    v_scale_asm = v_scale_flat.view(num_blocks, num_heads).to(scale_dtype)

    return k_quant, k_scale_asm, v_quant, v_scale_asm


@perftest(num_rotate_args=20)
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
    quant_type: QuantType = QuantType.per_Token,
    high_precision: int = 1,
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
        quant_type=quant_type,
        high_precision=high_precision,
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
    high_precision=1,
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
        high_precision=high_precision,
    )


def asm_V_shuffle(VC):
    # [num_blocks, num_kv_heads, head_size, block_size]
    x = 16 // VC.element_size()
    num_blocks, num_kv_heads, head_size, block_size = VC.shape
    VC = VC.view(num_blocks, num_kv_heads, head_size, block_size // x, x)
    # [num_blocks, num_kv_heads, block_size/X, head_size, X]
    VC = VC.permute(0, 1, 3, 2, 4).contiguous()
    return VC


def profile_kernel_breakdown(func, num_iters=100, num_warmup=10):
    """
    Profile a function and extract PA kernel and reduce kernel time breakdown.
    """
    schedule = tpf.schedule(
        wait=0,
        warmup=num_warmup,
        active=num_iters,
        repeat=1,
    )

    pa_time = 0.0
    reduce_time = 0.0
    total_time = 0.0

    with tpf.profile(
        activities=[tpf.ProfilerActivity.CUDA],
        schedule=schedule,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for step in range(num_warmup + num_iters):
            func()
            prof.step()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Analyze kernel times
    for event in prof.events():
        if event.device_type.name == "CUDA":
            time_us = event.device_time_total
            total_time += time_us
            name = event.name.lower()
            if "reduce" in name:
                reduce_time += time_us
            elif "pa_" in name or "pa " in name:
                pa_time += time_us

    avg_total = total_time / num_iters if num_iters > 0 else 0
    pa_ratio = pa_time / total_time if total_time > 0 else 0
    reduce_ratio = reduce_time / total_time if total_time > 0 else 0
    avg_pa_time = pa_time / num_iters if num_iters > 0 else 0
    avg_reduce_time = reduce_time / num_iters if num_iters > 0 else 0

    return avg_total, pa_ratio, reduce_ratio, avg_pa_time, avg_reduce_time


@benchmark()
def test_pa_ps(
    ctx_lens: int,
    batch_size: int,
    num_heads: tuple[int, int],
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
    qlen: int,
    varlen: bool = False,
    load_metadata: bool = False,
    dump_metadata: bool = False,
    profile_ps: bool = False,
    quant_type: QuantType = QuantType.per_Token,
    kv_dtype: torch.dtype = dtypes.fp8,
    is_causal: bool = True,
    high_precision: int = 1,
    metadata_mode: str = "v1_2",
) -> dict:
    ret = {}
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    device = "cuda:0"
    torch.set_default_device(device)
    num_query_heads, num_kv_heads = num_heads

    assert num_query_heads % num_kv_heads == 0
    assert 1 <= qlen <= ctx_lens
    if (
        kv_dtype in (torch.bfloat16, torch.float16)
        and quant_type != QuantType.per_Token
    ):
        raise ValueError("unquantized KV does not support per-block quantization")

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    seq_lens_kv = torch.empty(batch_size, dtype=torch.int)
    if varlen:
        for i in range(batch_size):
            seq_lens_kv[i] = random.randint(qlen, ctx_lens)
    else:
        seq_lens_kv.fill_(ctx_lens)
    seq_lens_qo = torch.randint(
        1, 5, (batch_size,), dtype=torch.int, device=device
    ).fill_(qlen)
    # seq_lens_kv = torch.tensor([10240] * (batch_size - 1) + [30720], dtype=torch.int, device=device)
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
    query, _key, _value = torch.split(
        qkv, [num_query_heads, num_kv_heads, num_kv_heads], dim=1
    )
    query.uniform_(*uniform_range)

    # Create the block tables.
    max_kv_len = seq_lens_kv.max().item()
    max_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    max_blocks = max_blocks_per_seq * batch_size
    block_tables = torch.randperm(max_blocks, dtype=torch.int).reshape(batch_size, -1)
    block_tables_lst = block_tables.tolist()

    # convert input for pa persistent interface
    actual_blocks = (seq_lens_kv + block_size - 1) // block_size
    kv_indptr[1 : batch_size + 1] = torch.cumsum(actual_blocks, dim=0)
    kv_indices_lst = []
    for i in range(batch_size):
        kv_indices_lst += block_tables_lst[i][: actual_blocks[i]]
    kv_indices = torch.tensor(kv_indices_lst, dtype=torch.int)

    # Create the KV caches.
    k_caches, v_caches = kv_cache_factory(
        max_blocks,
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

    torch_mha_extend(
        query,
        k_cache,
        v_cache,
        block_tables,
        seq_lens_kv,
        qo_indptr,
        is_causal=is_causal,
    )

    scale = float(1.0 / (head_size**0.5))

    # ################## quant start ######################
    if kv_dtype in (torch.bfloat16, torch.float16):
        k_quant_, v_quant_ = k_cache.to(kv_dtype), v_cache.to(kv_dtype)
        k_scale_ = v_scale_ = k_scale_asm = v_scale_asm = None
    elif quant_type in [
        QuantType.per_1x128,
        QuantType.per_256x128,
        QuantType.per_1024x128,
    ]:
        k_quant_, k_scale_asm, v_quant_, v_scale_asm = perblock_quant_kvcache_symm(
            k_cache, v_cache, quant_dtype=kv_dtype
        )
        # For per-block, k_scale_asm is [num_blocks, num_heads]
        k_scale_ = k_scale_asm
        v_scale_ = v_scale_asm
    else:
        (
            k_quant_,
            k_scale_,
            v_quant_,
            v_scale_,
            k_scale_asm,
            v_scale_asm,
        ) = pertoken_quant_kvcache_symm(k_cache, v_cache, quant_dtype=kv_dtype)

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
        is_causal=is_causal,
        quant_type=quant_type,
    )

    (
        (work_meta_data_size, work_meta_data_type),
        (work_indptr_size, work_indptr_type),
        (work_info_set_size, work_info_set_type),
        (reduce_indptr_size, reduce_indptr_type),
        (reduce_final_map_size, reduce_final_map_type),
        (reduce_partial_map_size, reduce_partial_map_type),
    ) = aiter.get_pa_metadata_info_v1(
        batch_size, num_kv_heads, fast_mode=metadata_mode == "v1_2"
    )
    work_metadata_ptrs = torch.empty(work_meta_data_size, dtype=work_meta_data_type)
    work_indptr = torch.empty(work_indptr_size, dtype=work_indptr_type)
    work_info = torch.empty(work_info_set_size, dtype=work_info_set_type)
    reduce_indptr = torch.empty(reduce_indptr_size, dtype=reduce_indptr_type)
    reduce_final_map = torch.empty(reduce_final_map_size, dtype=reduce_final_map_type)
    reduce_partial_map = torch.empty(
        reduce_partial_map_size, dtype=reduce_partial_map_type
    )

    metadata_map = {
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "seq_lens_kv": seq_lens_kv,
        "work_indptr": work_indptr,
        "work_info": work_info,
        "reduce_indptr": reduce_indptr,
        "reduce_final_map": reduce_final_map,
        "reduce_partial_map": reduce_partial_map,
    }

    us_metadata = 0.0
    if load_metadata:
        for name, meta in metadata_map.items():
            file_name = f"{name}.bin"
            shape = meta.shape
            array = np.fromfile(file_name, dtype=np.uint32)
            meta = torch.from_numpy(array).reshape(shape)
            torch.set_printoptions(threshold=999999, linewidth=120)
            print(f"==>load {name} from {file_name}:\n{meta}")
    else:
        # warmup for get_pa_metadata_v1
        aiter.get_pa_metadata_v1(
            (
                qo_indptr
                if metadata_mode == "v1_1"
                else torch.tensor([0], dtype=torch.int32)
            ),
            (
                kv_indptr
                if metadata_mode == "v1_1"
                else torch.tensor([0], dtype=torch.int32)
            ),
            (
                seq_lens_kv
                if metadata_mode == "v1_1"
                else torch.tensor([0], dtype=torch.int32)
            ),
            num_query_heads // num_kv_heads if metadata_mode == "v1_1" else 1,
            num_kv_heads if metadata_mode == "v1_1" else 1,
            is_causal,
            work_metadata_ptrs,
            work_indptr,
            work_info,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            kv_granularity=max(block_size, 16),
            block_size=block_size,
            max_seqlen_qo=int(max_qlen) if metadata_mode == "v1_1" else 1,
            uni_seqlen_qo=qlen if metadata_mode == "v1_1" else 1,
            fast_mode=metadata_mode == "v1_2",
            max_split_per_batch=-1,
        )
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        aiter.get_pa_metadata_v1(
            qo_indptr,
            kv_indptr,
            seq_lens_kv,
            num_query_heads // num_kv_heads,
            num_kv_heads,
            is_causal,
            work_metadata_ptrs,
            work_indptr,
            work_info,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            kv_granularity=max(block_size, 16),
            block_size=block_size,
            max_seqlen_qo=int(max_qlen),
            uni_seqlen_qo=qlen,
            fast_mode=metadata_mode == "v1_2",
            max_split_per_batch=-1,
        )
        end_event.record()
        end_event.synchronize()
        us_metadata = start_event.elapsed_time(end_event) * 1000  # ms to us

    if dump_metadata:
        for name, meta in metadata_map.items():
            file_name = f"{name}.bin"
            torch.set_printoptions(threshold=999999, linewidth=120)
            print(f"==>dump {name} shape {meta.shape} to {file_name}:\n{meta}")
            meta.cpu().numpy().astype(np.uint32).tofile(file_name)

    # Benchmark PA Persistent Scheduling
    output = torch.empty_like(query)
    v_shuffled = asm_V_shuffle(v_quant_)
    ps_kwargs = {
        "Q": query,
        "K": k_quant_,
        "V": v_shuffled,
        "output": output,
        "max_qlen": max_qlen,
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "context_lens": seq_lens_kv,
        "K_QScale": k_scale_asm,
        "V_QScale": v_scale_asm,
        "work_indptr": work_indptr,
        "work_info": work_info,
        "reduce_indptr": reduce_indptr,
        "reduce_final_map": reduce_final_map,
        "reduce_partial_map": reduce_partial_map,
        "softmax_scale": scale,
        "mask": int(is_causal),
        "quant_type": quant_type,
        "high_precision": high_precision,
    }
    _, us_pa_ps = run_aiter_asm_ps(**ps_kwargs)
    err = checkAllclose(out_ref, output, msg="[torch vs pa_persistent_fwd]")
    assert err <= 0.05, f"PA_PS output mismatch ratio: {err:.2%}"
    ret["us_metadata"] = us_metadata
    ret["us_pa_ps"] = us_pa_ps
    ret["err"] = err

    if profile_ps:
        (
            _,
            pa_ps_ratio,
            reduce_ratio,
            us_pa_ps_kernel,
            us_reduce_kernel,
        ) = profile_kernel_breakdown(lambda: aiter.pa_persistent_fwd(**ps_kwargs))
        ret["us_pa_ps_kernel"] = us_pa_ps_kernel
        ret["pa_ps_ratio"] = pa_ps_ratio
        ret["us_reduce_kernel"] = us_reduce_kernel
        ret["reduce_ratio"] = reduce_ratio
        if quant_type == QuantType.per_Token and (is_causal or max_qlen == 1):
            nops_args = (
                query,
                k_quant_,
                v_shuffled,
                block_tables,
                seq_lens_kv,
                block_tables.size(1),
                max_qlen,
                k_scale_asm,
                v_scale_asm,
            )
            _, us_pa_nops = run_aiter_asm(
                *nops_args, qo_indptr=qo_indptr, high_precision=high_precision
            )
            _, _, _, us_pa_nops_kernel, _ = profile_kernel_breakdown(
                lambda: aiter.pa_fwd_asm(
                    *nops_args, qo_indptr=qo_indptr, high_precision=high_precision
                )
            )
            ret["us_pa_nops"] = us_pa_nops
            ret["speedup"] = us_pa_nops / us_pa_ps if us_pa_ps > 0 else 0
            ret["speedup(pa)"] = (
                us_pa_nops_kernel / us_pa_ps_kernel if us_pa_ps_kernel > 0 else 0
            )

    return ret


def test_pa_metadata_v1_1():
    workers = torch.cuda.get_device_properties(0).multi_processor_count
    cases = [
        ([257] * 32, [4] * 32, 256, 8, -1, 256),
        ([257] * 31 + [70612], [4] * 32, 1024, 8, -1, 1024),
        ([70612] + [257] * 31, [4] * 32, 1024, 8, -1, 1024),
        ([16, 32, 1024], [1] * 3, 16, 1, -1, 16),
        ([257] * 7 + [200003], [4] * 8, 128, 8, 32, 128),
        ([200003] + [257] * 7, [1, 4, 2, 3] * 2, 16, 3, 3, 16),
        ([15, 17, 33] * 43, [1, 2, 4] * 43, 16, 8, -1, 16),
        ([0, 16, 32], [1] * 3, 16, 1, -1, 16),
        ([0] * 5, [1] * 5, 16, 3, 3, 16),
        ([2**30 + 17] * 65, [16] * 65, 256, 1, -1, 256),
        ([31, 513, 65537], [4] * 3, 16, 8, 1, 16),
        ([257, 8193, 65537], [4, 2, 1], 16, 3, 3, 64),
    ]
    for lengths, qo_lengths, page_size, heads, cap, granularity in cases:
        batch = len(lengths)
        qo_offsets = [0] + list(itertools.accumulate(qo_lengths))
        page_counts = [(length + page_size - 1) // page_size for length in lengths]
        page_offsets = [0] + list(itertools.accumulate(page_counts))
        context = torch.tensor(lengths, dtype=torch.int32, device="cuda")
        qo = torch.tensor(qo_offsets, dtype=torch.int32, device="cuda")
        kv = torch.tensor(page_offsets, dtype=torch.int32, device="cuda")
        buffers = []
        guards = []
        for shape, dtype in aiter.get_pa_metadata_info_v1(batch, heads, False, cap):
            size = int(np.prod(shape))
            backing = torch.full((size + 16,), 777, dtype=dtype, device="cuda")
            buffers.append(backing[:size].view(shape))
            guards.append(backing[size:])

        def generate(
            *,
            qo=qo,
            kv=kv,
            context=context,
            heads=heads,
            buffers=buffers,
            granularity=granularity,
            page_size=page_size,
            qo_lengths=qo_lengths,
            cap=cap,
        ):
            aiter.get_pa_metadata_v1(
                qo,
                kv,
                context,
                8,
                heads,
                True,
                *buffers,
                kv_granularity=granularity,
                block_size=page_size,
                max_seqlen_qo=max(qo_lengths),
                uni_seqlen_qo=-1,
                fast_mode=False,
                max_split_per_batch=cap,
            )

        def verify(
            current_lengths,
            *,
            granularity=granularity,
            page_size=page_size,
            qo_lengths=qo_lengths,
            buffers=buffers,
            heads=heads,
            cap=cap,
            qo_offsets=qo_offsets,
            page_offsets=page_offsets,
            batch=batch,
            guards=guards,
        ):
            tiles = [
                (length + granularity - 1) // granularity for length in current_lengths
            ]
            costs = [count * length for count, length in zip(tiles, qo_lengths)]
            total_cost = max(sum(costs), 1)
            capacity = buffers[5].numel()
            workers_per_head = workers // heads
            max_parts = workers if cap < 0 else min(workers, cap)
            cost_before = 0
            partial_before = 0
            reduce_offsets = [0]
            partial_map = []
            sequence_works = []
            sequence_workers = []
            for sequence, (tile_count, qo_length, cost) in enumerate(
                zip(tiles, qo_lengths, costs)
            ):
                cuts = {0, tile_count}
                if tile_count and qo_length:
                    for worker in range(1, workers_per_head):
                        boundary = worker * total_cost // workers_per_head
                        cut = (boundary - cost_before + qo_length - 1) // qo_length
                        cuts.add(max(0, min(tile_count, cut)))
                ordered_cuts = sorted(cuts)
                fragments = len(ordered_cuts) - 1
                parts = min(max_parts, fragments) if qo_length else 0
                for part in range(parts):
                    begin = ordered_cuts[part * fragments // parts]
                    end = ordered_cuts[(part + 1) * fragments // parts]
                    partial = partial_before + part * qo_length if parts > 1 else -1
                    sequence_works.append(
                        [
                            sequence,
                            partial,
                            qo_offsets[sequence],
                            qo_offsets[sequence + 1],
                            min(
                                page_offsets[sequence]
                                + begin * (granularity // page_size),
                                page_offsets[sequence + 1],
                            ),
                            min(
                                page_offsets[sequence]
                                + end * (granularity // page_size),
                                page_offsets[sequence + 1],
                            ),
                            0,
                            0,
                        ]
                    )
                    sequence_workers.append(
                        min(
                            (
                                (cost_before + begin * qo_length + 1) * workers_per_head
                                - 1
                            )
                            // total_cost,
                            workers_per_head - 1,
                        )
                    )
                    if parts > 1:
                        partial_map.append(partial)
                if parts > 1:
                    partial_before += parts * qo_length
                reduce_offsets.append(len(partial_map))
                cost_before += cost
            expected_works = []
            assigned_workers = []
            for head in range(heads):
                worker_begin = head * (workers // heads) + min(head, workers % heads)
                for work, worker in zip(sequence_works, sequence_workers):
                    expected_works.append(
                        work[:-1] + [((head + 1) * 8 << 16) | (head * 8)]
                    )
                    assigned_workers.append(worker_begin + worker)
            assert assigned_workers == sorted(assigned_workers)
            expected_indptr = [
                sum(worker < boundary for worker in assigned_workers)
                for boundary in range(workers + 1)
            ]
            actual_works = buffers[2][: len(expected_works)].cpu().tolist()
            if (
                batch == 32
                and heads == 8
                and workers == 256
                and max(current_lengths) == 257
            ):
                assert (
                    int(buffers[1][-1].item()) == 256
                ), "uniform workload must not be oversplit"
                assert (
                    buffers[3][-1].item() == 0
                ), "uniform workload should write output directly"
            if (
                batch == 32
                and heads == 8
                and workers == 256
                and max(current_lengths) == 70612
            ):
                actual_offsets = buffers[1].cpu().tolist()
                worker_pages = [
                    sum(work[5] - work[4] for work in actual_works[begin:end])
                    for begin, end in itertools.pairwise(actual_offsets)
                ]
                assert max(worker_pages) <= 4, "long-tail worker exceeds its KV budget"
            assert actual_works == expected_works, (batch, heads, cap, "works")
            assert buffers[1].cpu().tolist() == expected_indptr, (
                batch,
                heads,
                cap,
                "workers",
            )
            assert buffers[3].cpu().tolist() == reduce_offsets, (
                batch,
                heads,
                cap,
                "reduce",
            )
            assert buffers[4].cpu().tolist() == list(
                map(list, itertools.pairwise(qo_offsets))
            )
            assert buffers[5][: len(partial_map)].cpu().tolist() == partial_map
            assert buffers[0].cpu().tolist() == [
                buffers[1].data_ptr(),
                buffers[2].data_ptr(),
            ]
            assert partial_before <= capacity * max(qo_lengths)
            assert all(torch.all(guard == 777).item() for guard in guards)
            assert all(work[6] == 0 for work in actual_works)

        generate()
        verify(lengths)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            generate()
        shorter = [length // 2 for length in lengths]
        context.copy_(torch.tensor(shorter, dtype=torch.int32, device="cuda"))
        graph.replay()
        verify(shorter)
    print(f"PASS PA v1_1 metadata: {len(cases)} cases and {len(cases)} graph refreshes")


def benchmark_pa_metadata(args):
    if args.benchmark_rounds < 2 or args.benchmark_replays < 1:
        raise ValueError("benchmark requires at least two rounds and one replay")
    device = torch.device("cuda:0")
    torch.set_default_device(device)
    root = Path(__file__).resolve().parents[1]
    output_path = Path(args.benchmark_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_files = [
        "csrc/kernels/mla/metadata/v1_1_pa_device.cuh",
        "csrc/kernels/mla/metadata/v1_2_pa_device.cuh",
        "csrc/kernels/mla/metadata.cu",
        "aiter/ops/attention.py",
        "op_tests/test_pa_ps.py",
    ]
    report = {
        "gpu": get_gfx(),
        "torch": torch.__version__,
        "rounds": args.benchmark_rounds,
        "replays": args.benchmark_replays,
        "calls_per_graph": 32,
        "timing": "CUDA Graph, preallocated fixed buffers, warm cache, alternating mode order",
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in source_files
        },
        "cases": [],
    }
    configurations = itertools.product(
        args.batch_size,
        args.qlen,
        args.num_heads,
        args.block_size,
        args.ctx_len,
        args.benchmark_scenarios,
        args.quant_type,
    )
    for (
        batch,
        qlen,
        heads,
        page_size,
        max_context,
        scenario,
        quant_name,
    ) in configurations:
        if page_size == 16 and get_gfx() != "gfx950":
            continue
        quant_type = getattr(QuantType, quant_name)
        use_asm_reduce = (
            page_size == 16 and args.head_dim == 128 and get_gfx() == "gfx950"
        )
        reduce_fn = _attention_ps_reduce_asm if use_asm_reduce else aiter.pa_reduce_v1
        query_heads, kv_heads = heads
        if query_heads % kv_heads or qlen * (query_heads // kv_heads) > 128:
            raise ValueError("benchmark requires GQA * qlen <= 128")
        if max_context < 257 or qlen < 1:
            raise ValueError("benchmark requires max context >= 257 and positive qlen")
        if scenario == "uniform-short":
            lengths = [257] * batch
        elif scenario == "uniform-long":
            lengths = [max_context] * batch
        elif scenario == "one-long":
            lengths = [max_context] + [257] * (batch - 1)
        else:
            lengths = [
                max(
                    257, round(257 * (max_context / 257) ** (index / max(batch - 1, 1)))
                )
                for index in range(batch)
            ]
        random.Random(42).shuffle(lengths)
        torch.manual_seed(42)
        page_counts = [(length + page_size - 1) // page_size for length in lengths]
        offsets = [0] + list(itertools.accumulate(page_counts))
        pool_blocks = offsets[-1] + kv_heads
        context = torch.tensor(lengths, dtype=torch.int32, device=device)
        qo = torch.arange(batch + 1, dtype=torch.int32, device=device) * qlen
        kv = torch.tensor(offsets, dtype=torch.int32, device=device)
        indices = torch.randperm(pool_blocks, dtype=torch.int32, device=device)[
            : offsets[-1]
        ]
        block_tables = torch.zeros(
            (batch, max(page_counts)), dtype=torch.int32, device=device
        )
        for sequence in range(batch):
            block_tables[sequence, : page_counts[sequence]] = indices[
                offsets[sequence] : offsets[sequence + 1]
            ]
        query = torch.empty(
            (batch * qlen, query_heads, args.head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        query.uniform_(-1, 1)
        keys, values = kv_cache_factory(
            pool_blocks,
            page_size,
            1,
            kv_heads,
            args.head_dim,
            "auto",
            torch.bfloat16,
            42,
            device,
        )
        if quant_type == QuantType.per_Token:
            key, key_ref_scale, value, value_ref_scale, key_scale, value_scale = (
                pertoken_quant_kvcache_symm(
                    keys[0], values[0], quant_dtype=aiter.dtypes.fp8
                )
            )
        else:
            key, key_scale, value, value_scale = perblock_quant_kvcache_symm(
                keys[0], values[0], quant_dtype=aiter.dtypes.fp8
            )
            key_ref_scale, value_ref_scale = key_scale, value_scale
        del keys, values
        scale = args.head_dim**-0.5
        reference = torch_mha_extend(
            query,
            key,
            value,
            block_tables,
            context,
            qo,
            key_ref_scale,
            value_ref_scale,
            is_causal=True,
            quant_type=quant_type,
        )
        value = asm_V_shuffle(value)
        graphs = {}
        states = {}
        metrics = ("metadata", "attention_reduce", "total")
        modes = ("v1_2", "v1_1")
        if args.benchmark_diagnose:
            metrics += ("attention", "reduce")
            modes += ("v1_1_repacked",)
        samples = {
            mode: {
                metric: []
                for metric in metrics
                if mode != "v1_1_repacked" or metric not in ("metadata", "total")
            }
            for mode in modes
        }
        for mode, mode_samples in samples.items():
            fast_mode = mode == "v1_2"
            buffers = [
                torch.empty(shape, dtype=dtype, device=device)
                for shape, dtype in aiter.get_pa_metadata_info_v1(
                    batch, kv_heads, fast_mode
                )
            ]
            output = torch.empty_like(query)
            partial = torch.empty(
                (buffers[5].numel() * qlen, 1, query_heads, args.head_dim),
                dtype=torch.float32,
                device=device,
            )
            partial_lse = torch.empty(
                (buffers[5].numel() * qlen, 1, query_heads, 1),
                dtype=torch.float32,
                device=device,
            )
            final_lse = torch.empty(
                (batch * qlen, query_heads), dtype=torch.float32, device=device
            )

            def generate(
                *,
                qo=qo,
                kv=kv,
                context=context,
                query_heads=query_heads,
                kv_heads=kv_heads,
                buffers=buffers,
                page_size=page_size,
                qlen=qlen,
                fast_mode=fast_mode,
            ):
                aiter.get_pa_metadata_v1(
                    qo,
                    kv,
                    context,
                    query_heads // kv_heads,
                    kv_heads,
                    True,
                    *buffers,
                    kv_granularity=page_size,
                    block_size=page_size,
                    max_seqlen_qo=qlen,
                    uni_seqlen_qo=qlen,
                    fast_mode=fast_mode,
                )

            def attention(
                *,
                query=query,
                key=key,
                value=value,
                kv=kv,
                indices=indices,
                context=context,
                scale=scale,
                qlen=qlen,
                key_scale=key_scale,
                value_scale=value_scale,
                output=output,
                qo=qo,
                buffers=buffers,
                partial=partial,
                partial_lse=partial_lse,
                quant_type=quant_type,
            ):
                aiter.pa_ps_fwd_asm(
                    query,
                    key,
                    value,
                    kv,
                    indices,
                    context,
                    scale,
                    max_qlen=qlen,
                    K_QScale=key_scale,
                    V_QScale=value_scale,
                    out_=output,
                    qo_indptr=qo,
                    work_indptr=buffers[1],
                    work_info=buffers[2],
                    splitData=partial,
                    splitLse=partial_lse,
                    mask=1,
                    quant_type=quant_type,
                )

            def reduce(
                *,
                reduce_fn=reduce_fn,
                partial=partial,
                partial_lse=partial_lse,
                buffers=buffers,
                qlen=qlen,
                output=output,
                final_lse=final_lse,
            ):
                reduce_fn(
                    partial,
                    partial_lse,
                    buffers[3],
                    buffers[4],
                    buffers[5],
                    qlen,
                    output,
                    final_lse,
                )

            def calculate():
                attention()
                reduce()

            def total():
                generate()
                calculate()

            for _ in range(3):
                total()
            torch.cuda.synchronize()
            task_count = int(buffers[1][-1].item())
            work_records = buffers[2][:task_count].cpu().tolist()
            if mode == "v1_1_repacked":
                worker_count = buffers[1].numel() - 1
                buckets = [[] for _ in range(worker_count)]
                loads = [0] * worker_count

                def work_cost(work):
                    return (work[5] - work[4]) * (work[3] - work[2])

                for work in sorted(work_records, key=work_cost, reverse=True):
                    head = (work[7] & 65535) // (query_heads // kv_heads)
                    head_begin = head * (worker_count // kv_heads) + min(
                        head, worker_count % kv_heads
                    )
                    head_workers = worker_count // kv_heads + int(
                        head < worker_count % kv_heads
                    )
                    worker = min(
                        range(head_begin, head_begin + head_workers),
                        key=lambda index: (loads[index], len(buckets[index]), index),
                    )
                    buckets[worker].append(work)
                    loads[worker] += work_cost(work)
                repacked = list(itertools.chain.from_iterable(buckets))
                assert sorted(repacked) == sorted(work_records)
                buffers[2][:task_count].copy_(
                    torch.tensor(repacked, dtype=torch.int32, device=device)
                )
                buffers[1].copy_(
                    torch.tensor(
                        [0]
                        + list(itertools.accumulate(len(bucket) for bucket in buckets)),
                        dtype=torch.int32,
                        device=device,
                    )
                )
                work_records = repacked
                for _ in range(3):
                    calculate()
                torch.cuda.synchronize()
            torch.testing.assert_close(output, reference, atol=0.01, rtol=0.01)
            work_offsets = buffers[1].cpu().tolist()
            worker_pages = [
                sum(work[5] - work[4] for work in work_records[begin:end])
                for begin, end in itertools.pairwise(work_offsets)
            ]
            worker_works = [
                end - begin for begin, end in itertools.pairwise(work_offsets)
            ]
            split_pages = [work[5] - work[4] for work in work_records if work[1] >= 0]
            states[mode] = {
                "buffers": buffers,
                "output": output,
                "partial": partial,
                "partial_lse": partial_lse,
                "final_lse": final_lse,
                "tasks": task_count,
                "reduce_partials": int(buffers[3][-1].item()),
                "work_layout": {
                    "worker_pages": worker_pages,
                    "worker_works": worker_works,
                    "active_workers": sum(pages > 0 for pages in worker_pages),
                    "max_worker_pages": max(worker_pages),
                    "max_worker_works": max(worker_works),
                    "direct_tasks": sum(work[1] < 0 for work in work_records),
                    "split_tasks": len(split_pages),
                    "min_split_pages": min(split_pages, default=0),
                    "max_split_pages": max(split_pages, default=0),
                },
                "max_abs_error": float(
                    (output.float() - reference.float()).abs().max().item()
                ),
            }
            graphs[mode] = {}
            functions = {
                "metadata": generate,
                "attention_reduce": calculate,
                "total": total,
                "attention": attention,
                "reduce": reduce,
            }
            for metric in mode_samples:
                function = functions[metric]
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(report["calls_per_graph"]):
                        function()
                graphs[mode][metric] = graph
                graph.replay()
            torch.cuda.synchronize()

        for round_index in range(args.benchmark_rounds):
            order = modes if round_index % 2 == 0 else tuple(reversed(modes))
            for metric in metrics:
                for mode in order:
                    if metric not in graphs[mode]:
                        continue
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.benchmark_replays):
                        graphs[mode][metric].replay()
                    end.record()
                    end.synchronize()
                    elapsed = (
                        start.elapsed_time(end)
                        * 1000
                        / (args.benchmark_replays * report["calls_per_graph"])
                    )
                    samples[mode][metric].append(elapsed)
        for mode in states:
            torch.testing.assert_close(
                states[mode]["output"], reference, atol=0.01, rtol=0.01
            )
        medians = {
            mode: {
                metric: statistics.median(times)
                for metric, times in mode_samples.items()
            }
            for mode, mode_samples in samples.items()
        }
        case = {
            "scenario": scenario,
            "batch": batch,
            "qlen": qlen,
            "query_heads": query_heads,
            "kv_heads": kv_heads,
            "head_dim": args.head_dim,
            "page_size": page_size,
            "quant_type": quant_name,
            "reduce_backend": "asm" if use_asm_reduce else "hip",
            "context_lengths": lengths,
            "correctness": "PASS atol=rtol=0.01 before and after timing",
            "modes": {
                mode: {
                    "median_us": medians[mode],
                    "samples_us": samples[mode],
                    "tasks": states[mode]["tasks"],
                    "reduce_partials": states[mode]["reduce_partials"],
                    "work_layout": states[mode]["work_layout"],
                    "max_abs_error": states[mode]["max_abs_error"],
                }
                for mode in states
            },
            "total_change_pct": 100
            * (medians["v1_1"]["total"] / medians["v1_2"]["total"] - 1),
            "paired_total_change_pct": [
                100 * (new / old - 1)
                for old, new in zip(samples["v1_2"]["total"], samples["v1_1"]["total"])
            ],
        }
        report["cases"].append(case)
        output_path.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"PERF {scenario} B={batch} Q={qlen} H={query_heads}/{kv_heads} page={page_size}: "
            f"v1_2={medians['v1_2']['total']:.3f} us v1_1={medians['v1_1']['total']:.3f} us "
            f"change={case['total_change_pct']:+.2f}%",
            flush=True,
        )
        del graphs, states
    print(f"Performance results: {output_path}")


gpu = torch.cuda.current_device()
device_properties = torch.cuda.get_device_properties(gpu)
cu_num = device_properties.multi_processor_count
gpu_arch = get_gfx()

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="PA_PS correctness and performance tests. Page16 COs currently "
    "support gfx950 (MI350 series) only; other page sizes are unchanged.",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    nargs="*",
    default=[dtypes.d_dtypes["bf16"]],
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-kvd",
    "--kv_dtype",
    type=dtypes.str2Dtype,
    choices=[dtypes.fp8, dtypes.i8, dtypes.bf16, dtypes.fp16],
    nargs="+",
    default=[dtypes.fp8],
    metavar="{fp8,i8,bf16,fp16}",
    help="KV cache dtype; bf16/fp16 use unquantized KV.",
)
parser.add_argument(
    "-n",
    "--num_heads",
    type=dtypes.str2tuple,
    nargs="*",
    default=[(10, 1), (16, 2)],
    help="""Number of heads.
    e.g. -n 8,1""",
)
parser.add_argument(
    "-hd",
    "--head_dim",
    type=int,
    default=128,
    help="""Head dimension.
    e.g. -hd 128""",
)
parser.add_argument(
    "-q",
    "--qlen",
    type=int,
    nargs="*",
    default=[1, 2, 3, 4],
    help="""Query length.
    e.g. -q 1""",
)
parser.add_argument(
    "-c",
    "--ctx_len",
    type=int,
    nargs="*",
    default=None,
    help="""Context length.
    Defaults to the correctness grid, or 8192 in benchmark mode.
    e.g. -c 128""",
)
parser.add_argument(
    "-b",
    "--batch_size",
    type=int,
    nargs="*",
    default=sorted([64, 158, cu_num // 2]),
    help="""Batch size.
    e.g. -b 128""",
)
parser.add_argument(
    "--block_size",
    type=int,
    nargs="*",
    default=[1024],
    help="""KV page size. Page16 requires gfx950 (MI350 series);
    page16 cases are skipped on other architectures, including gfx942 (MI300 series).
    e.g. --block_size 1024""",
)
parser.add_argument(
    "--causal",
    type=dtypes.str2bool,
    nargs="+",
    default=[True],
    help="Causal mask, e.g. --causal true false.",
)
parser.add_argument(
    "--high_precision",
    type=int,
    choices=[0, 1, 2],
    nargs="+",
    default=[1],
    help="Kernel precision: 0=normal, 1=prefer high, 2=ultra.",
)
parser.add_argument(
    "--varlen",
    action="store_true",
    help="""variable kv seqlens per batch. Default: False.
    --varlen # True""",
)
parser.add_argument(
    "--load_metadata",
    action="store_true",
    help="""load metadata by metadata_map Default: False.
    --load_metadata # True""",
)
parser.add_argument(
    "--dump_metadata",
    action="store_true",
    help="""dump metadata by metadata_map. Default: False.
    --dump_metadata # True""",
)
parser.add_argument(
    "--profile",
    action="store_true",
    help="""Enable performance profiling. Default: False (single run).
    --profile # Enable detailed performance stats""",
)
parser.add_argument(
    "--quant_type",
    type=str,
    choices=["per_Token", "per_256x128", "per_1024x128"],
    nargs="+",
    default=["per_Token"],
    help="Quantization granularity, e.g. --quant_type per_Token per_256x128.",
)
parser.add_argument(
    "--output", default="pa_ps.csv", help="Path for the benchmark CSV output."
)
parser.add_argument(
    "--metadata-mode",
    choices=["v1_2", "v1_1"],
    default="v1_2",
    help="PA metadata strategy; v1_1 uses prefix-budget GPU planning (dense, one Q tile).",
)
parser.add_argument(
    "--test-metadata-v1-1",
    action="store_true",
    help="Run proportional metadata coverage and CUDA Graph checks only.",
)
parser.add_argument("--benchmark-metadata", action="store_true")
parser.add_argument(
    "--benchmark-diagnose",
    action="store_true",
    help="Separate attention/reduce timings and test CPU repacking of identical v1_1 tasks.",
)
parser.add_argument(
    "--benchmark-scenarios",
    nargs="+",
    choices=["uniform-short", "uniform-long", "geometric", "one-long"],
    default=["uniform-short", "uniform-long", "geometric", "one-long"],
)
parser.add_argument("--benchmark-rounds", type=int, default=9)
parser.add_argument("--benchmark-replays", type=int, default=8)
parser.add_argument("--benchmark-output", default="pa_metadata_performance.json")
args = parser.parse_args()
if args.ctx_len is None:
    args.ctx_len = (
        [8192]
        if args.benchmark_metadata
        else [7, 109, 256, 282, 1024, 2580, 4097, 8192, 10240]
    )
if args.benchmark_metadata:
    benchmark_pa_metadata(args)
    raise SystemExit(0)
if args.test_metadata_v1_1:
    test_pa_metadata_v1_1()
    raise SystemExit(0)

if gpu_arch != "gfx950" and 16 in args.block_size:
    aiter.logger.info(
        "Skipping page16 PA_PS on %s: page16 COs are only provided for "
        "gfx950 (MI350 series). Other requested page sizes will still run.",
        gpu_arch,
    )
    args.block_size = [page_size for page_size in args.block_size if page_size != 16]
    if not args.block_size:
        aiter.logger.info(
            "PA_PS skipped: no supported page sizes remain; "
            "no cases ran and no result CSV was written."
        )
        raise SystemExit(0)

df = []
for dtype in args.dtype:
    for (
        num_heads,
        qlen,
        ctx_len,
        batch_size,
        block_size,
        kv_dtype,
        is_causal,
        high_precision,
        quant_type,
    ) in itertools.product(
        args.num_heads,
        args.qlen,
        args.ctx_len,
        args.batch_size,
        args.block_size,
        args.kv_dtype,
        args.causal,
        args.high_precision,
        args.quant_type,
    ):
        if qlen > ctx_len:
            continue
        ret = test_pa_ps(
            ctx_len,
            batch_size,
            num_heads,
            args.head_dim,
            block_size,
            dtype,
            qlen,
            args.varlen,
            args.load_metadata,
            args.dump_metadata,
            args.profile,
            getattr(QuantType, quant_type),
            kv_dtype=kv_dtype,
            is_causal=is_causal,
            high_precision=high_precision,
            metadata_mode=args.metadata_mode,
        )
        df.append(ret)
df = pd.DataFrame(df)
df = df.drop(columns=["load_metadata", "dump_metadata", "profile_ps"], errors="ignore")
df_md = df.to_markdown(index=False)
aiter.logger.info("pa_ps summary (markdown):\n%s", df_md)
df.to_csv(args.output)
