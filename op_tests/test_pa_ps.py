# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools
from aiter.ops.enum import QuantType
import numpy as np
import random
from typing import List, Optional, Tuple, Union

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter import pertoken_quant
from aiter.test_common import benchmark, checkAllclose, perftest
import torch.profiler as tpf

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
):
    num_blocks, num_heads, head_size, block_size = v_cache.shape
    sm_scale = 1.0 / (head_size**0.5)

    # Detect per-block vs per-token based on scale shape
    # per-token: [num_heads, total_tokens], per-block: [num_blocks, num_heads]
    per_block_quant = k_scale is not None and k_scale.shape[0] == num_blocks

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


def perblock_quant_kvcache_symm(
    # [num_blocks, num_heads, head_size // x, block_size, x]
    k_cache: torch.Tensor,
    # [num_blocks, num_heads, head_size, block_size]
    v_cache: torch.Tensor,
    quant_dtype: torch.dtype,  # e.g. torch.float8_e4m3fnuz
    scale_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    num_heads: Tuple[int, int],
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
    qlen: int,
    varlen: bool = False,
    load_metadata: bool = False,
    dump_metadata: bool = False,
    profile_ps: bool = False,
    quant_type: QuantType = QuantType.per_Token,
) -> dict:
    ret = {}
    seed = 0
    device = "cuda:0"
    torch.set_default_device(device)
    num_query_heads, num_kv_heads = num_heads

    assert num_query_heads % num_kv_heads == 0

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
    query, key, value = torch.split(
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
    for i in range(0, batch_size):
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
    )

    scale = float(1.0 / (head_size**0.5))

    # ################## quant start ######################
    if quant_type in [
        QuantType.per_1x128,
        QuantType.per_256x128,
        QuantType.per_1024x128,
    ]:
        k_quant_, k_scale_asm, v_quant_, v_scale_asm = perblock_quant_kvcache_symm(
            k_cache, v_cache, quant_dtype=aiter.dtypes.fp8
        )
        # For per-block, k_scale_asm is [num_blocks, num_heads]
        k_scale_ = k_scale_asm
        v_scale_ = v_scale_asm
    else:
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

    (
        (work_meta_data_size, work_meta_data_type),
        (work_indptr_size, work_indptr_type),
        (work_info_set_size, work_info_set_type),
        (reduce_indptr_size, reduce_indptr_type),
        (reduce_final_map_size, reduce_final_map_type),
        (reduce_partial_map_size, reduce_partial_map_type),
    ) = aiter.get_pa_metadata_info_v1(batch_size, num_kv_heads)
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
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            1,
            1,
            True,
            work_metadata_ptrs,
            work_indptr,
            work_info,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            kv_granularity=max(block_size, 16),
            block_size=block_size,
            max_seqlen_qo=1,
            uni_seqlen_qo=1,
            fast_mode=True,
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
            True,
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
            fast_mode=True,
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

    if profile_ps:
        # Prepare V cache for both methods
        v_shuffled = asm_V_shuffle(v_quant_)

        out_aiter_asm, us_pa_ps = run_aiter_asm_ps(
            Q=query,
            K=k_quant_,
            V=v_shuffled,
            output=output,
            max_qlen=max_qlen,
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            context_lens=seq_lens_kv,
            K_QScale=k_scale_asm,
            V_QScale=v_scale_asm,
            work_indptr=work_indptr,
            work_info=work_info,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
            softmax_scale=scale,
            mask=1,
        )

        _, us_pa_nops = run_aiter_asm(
            query,
            k_quant_,
            v_shuffled,
            block_tables,
            seq_lens_kv,
            block_tables.size(1),
            max_qlen,
            k_scale=k_scale_asm,
            v_scale=v_scale_asm,
            qo_indptr=qo_indptr,
        )

        # Profile kernel breakdown for PA PS
        _, pa_ps_ratio, reduce_ratio, us_pa_ps_kernel, us_reduce_kernel = (
            profile_kernel_breakdown(
                lambda: aiter.pa_persistent_fwd(
                    Q=query,
                    K=k_quant_,
                    V=v_shuffled,
                    output=output,
                    max_qlen=max_qlen,
                    qo_indptr=qo_indptr,
                    kv_indptr=kv_indptr,
                    kv_indices=kv_indices,
                    context_lens=seq_lens_kv,
                    K_QScale=k_scale_asm,
                    V_QScale=v_scale_asm,
                    work_indptr=work_indptr,
                    work_info=work_info,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    softmax_scale=scale,
                    mask=1,
                )
            )
        )

        # Profile kernel breakdown for PA NOPS (no reduce kernel)
        _, _, _, us_pa_nops_kernel, _ = profile_kernel_breakdown(
            lambda: aiter.pa_fwd_asm(
                query,
                k_quant_,
                v_shuffled,
                block_tables,
                seq_lens_kv,
                block_tables.size(1),
                max_qlen,
                k_scale_asm,
                v_scale_asm,
                None,
                qo_indptr,
            )
        )

        # Verify correctness
        err = checkAllclose(
            out_ref,
            output,
            msg="[torch vs  pa_persistent_fwd][   Quant]: us......",
        )

        # Store results
        ret["us_metadata"] = us_metadata
        ret["us_pa_nops"] = us_pa_nops
        ret["us_pa_ps"] = us_pa_ps
        ret["us_pa_ps_kernel"] = us_pa_ps_kernel
        ret["pa_ps_ratio"] = pa_ps_ratio
        ret["us_reduce_kernel"] = us_reduce_kernel
        ret["reduce_ratio"] = reduce_ratio
        ret["speedup"] = us_pa_nops / us_pa_ps if us_pa_ps > 0 else 0
        ret["speedup(pa)"] = (
            us_pa_nops_kernel / us_pa_ps_kernel if us_pa_ps_kernel > 0 else 0
        )
        ret["err fp8"] = err
    else:
        out_aiter_asm, us_aiter_asm = run_aiter_asm_ps(
            Q=query,
            K=k_quant_,
            V=asm_V_shuffle(v_quant_),
            output=output,
            max_qlen=max_qlen,
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            context_lens=seq_lens_kv,
            K_QScale=k_scale_asm,
            V_QScale=v_scale_asm,
            work_indptr=work_indptr,
            work_info=work_info,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
            softmax_scale=scale,
            mask=1,
            quant_type=quant_type,
        )

        # _, us_asm_noquant = run_aiter_asm(
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

        err = checkAllclose(
            out_ref,
            output,
            msg="[torch vs  pa_persistent_fwd][   Quant]: us......",
        )

        # if isinstance(us_aiter_asm, torch.Tensor):
        #     us_aiter_asm = us_aiter_asm.item()
        # if isinstance(us_asm_noquant, torch.Tensor):
        #     us_asm_noquant = us_asm_noquant.item()

        # print(f"seq_lens_kv: {seq_lens_kv.tolist()[0]}")
        # print(f"batch_size: {batch_size}")
        # print(f"PA PS forward: {us_aiter_asm:>8.2f} ms")
        # print(f"PA without persistent: {us_asm_noquant:>8.2f} ms")
        # print(
        #     f"Speedup: {us_asm_noquant / us_aiter_asm if us_aiter_asm > 0 else 0:.2f}x"
        # )

        ret["us_metadata"] = us_metadata
        ret["us_asm_fp8"] = us_aiter_asm
        ret["err fp8"] = err

    return ret


head_dim = 128
l_block_size = [1024]
l_dtype = ["bf16"]
l_num_heads = [
    # (10, 1),
    # (16, 2),
    (16, 1),
]
l_qlen = [3]
# l_qlen = [4]
l_ctx_len = [
    7,
    109,
    256,
    282,
    1024,
    2580,
    4097,
    8192,
    10240,
]
gpu = torch.cuda.current_device()
device_properties = torch.cuda.get_device_properties(gpu)
cu_num = device_properties.multi_processor_count
l_batch_size = [64, 158, cu_num // 2]
l_batch_size.sort()

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
    default="per_Token",
    help="""Use per-block quantization instead of per-token. Default: False.
    --per_block_quant # Enable per-block quantization""",
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

# Convert string to QuantType enum
quant_type_map = {
    "per_Token": QuantType.per_Token,
    "per_256x128": QuantType.per_256x128,
    "per_1024x128": QuantType.per_1024x128,
}
l_quant_type = quant_type_map.get(args.quant_type, QuantType.per_Token)

for dtype in l_dtype:
    df = []
    for num_heads, qlen, ctx_len, batch_size, block_size in itertools.product(
        l_num_heads, l_qlen, l_ctx_len, l_batch_size, l_block_size
    ):
        ret = test_pa_ps(
            ctx_len,
            batch_size,
            num_heads,
            head_dim,
            block_size,
            dtype,
            qlen,
            l_varlen,
            args.load_metadata,
            args.dump_metadata,
            args.profile,
            l_quant_type,
        )
        df.append(ret)
    df = pd.DataFrame(df)
    df_md = df.to_markdown(index=False)
    aiter.logger.info("pa_ps summary (markdown):\n%s", df_md)
    df.to_csv("pa_ps.csv")
