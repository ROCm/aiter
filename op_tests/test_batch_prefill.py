# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import itertools
from typing import Optional
import math
import os
import sys
import pytest
import torch
import json
import numpy as np
import argparse
import triton
import triton.language as tl

import aiter
from einops import rearrange, repeat
from aiter.test_common import perftest

TEST_NUM_ITERS = 100
# TEST_NUM_ITERS = 2


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: Optional[int] = None,
    soft_cap: Optional[float] = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx:start_idx + query_len]
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = torch.triu(empty_mask,
                                             diagonal=kv_len -
                                             (query_len + sliding_window) +
                                             1).bool().logical_not()
            mask |= sliding_window_mask
        if soft_cap is not None:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


@triton.jit
def _vllm_layout_trans_kernel(
    k_buffer_ptr,
    v_buffer_ptr,
    k_values_ptr,
    v_values_ptr,
    b_seq_lens_loc,
    block_table,
    block_table_stride_0,
    E_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    batch_token_indexes = tl.load(b_seq_lens_loc + batch_idx +
                                    tl.arange(0, 2))
    batch_token_start, batch_token_end = tl.split(batch_token_indexes)
    seq_len = batch_token_end - batch_token_start
    if block_idx * BLOCK_SIZE < seq_len:
        block_mask = (block_idx * BLOCK_SIZE +
                        tl.arange(0, BLOCK_SIZE)[:, None]) < seq_len

        kv_idx = tl.load(block_table + batch_idx * block_table_stride_0 +
                            block_idx)

        kv_buffer_off = kv_idx * BLOCK_SIZE * E_DIM + tl.arange(
            0, BLOCK_SIZE)[:, None] * E_DIM + tl.arange(0, E_DIM)[None, :]
        k_vals = tl.load(k_buffer_ptr + kv_buffer_off,
                            mask=block_mask,
                            other=0.0)
        v_vals = tl.load(v_buffer_ptr + kv_buffer_off,
                            mask=block_mask,
                            other=0.0)

        kv_values_off = batch_token_start * E_DIM + \
            block_idx * BLOCK_SIZE * E_DIM + \
            tl.arange(0, BLOCK_SIZE)[:, None] * E_DIM + \
            tl.arange(0, E_DIM)[None, :]
        tl.store(k_values_ptr + kv_values_off, k_vals, mask=block_mask)
        tl.store(v_values_ptr + kv_values_off, v_vals, mask=block_mask)


def vllm_layout_trans(b_seq_lens_loc, block_table, k_buffer, v_buffer,
                        max_seq_len, total_tokens):
    H_KV = v_buffer.shape[2]
    D = v_buffer.shape[3]
    BLOCK_SIZE = v_buffer.shape[1]
    dtype = k_buffer.dtype
    k_values = torch.empty((total_tokens, H_KV, D),
                            dtype=dtype,
                            device="cuda")
    v_values = torch.empty((total_tokens, H_KV, D),
                            dtype=dtype,
                            device="cuda")

    grid = (block_table.shape[0],
            (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)

    _vllm_layout_trans_kernel[grid](k_buffer,
                                    v_buffer,
                                    k_values,
                                    v_values,
                                    b_seq_lens_loc,
                                    block_table,
                                    block_table.stride(0),
                                    E_DIM=H_KV * D,
                                    BLOCK_SIZE=BLOCK_SIZE)

    return k_values, v_values


def load_function_inputs(dump_dir, skip_data_tensor_name_list):
    """
    Loads function inputs from the specified directory,
    fully restoring tensor layouts (including non-contiguous ones).
    """

    def create_tensor(min, max, *args, **kwargs):
        # print(args)
        # print(kwargs)
        x = torch.randn(*args, **kwargs)
        x = (x - x.min()) / (x.max() - x.min())
        return min + (max - min) * x

    # Read metadata file
    metadata_path = os.path.join(dump_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    params = {}

    # Load non-tensor parameters
    for name, value in metadata['non_tensors'].items():
        # Special handling for window_size tuple
        params[name] = value

    # Load tensor parameters
    for name, tensor_meta in metadata['tensors'].items():
        contiguous_tensor = None
        if name in skip_data_tensor_name_list:
            print(f"tensor {name} create from scratch")
            shape = tensor_meta['shape']
            dtype = tensor_meta['dtype']
            device = tensor_meta['device']
            dtype = getattr(torch, dtype.split('.')[1])
            # print(shape)
            # print(type(dtype))
            # print(type(device))
            init_min = None
            init_max = None
            if name in "kv":
                init_min, init_max = (-5, 5)
            else:
                init_min, init_max = (-10, 10)
            contiguous_tensor = create_tensor(
                init_min, init_max, shape, dtype=dtype, device=device)
        else:
            print(f"tensor {name} load data from disk")
            file_path = os.path.join(dump_dir, tensor_meta['file'])
            data = np.load(file_path)
            # Create contiguous PyTorch tensor from numpy data
            contiguous_tensor = torch.tensor(data)
            # Restore original data type
            dtype_str = tensor_meta['dtype'].replace("torch.", "")
            dtype = getattr(torch, dtype_str)
            contiguous_tensor = contiguous_tensor.to(dtype)
            # Restore original device placement
            if tensor_meta['device'] != 'cpu':
                device = torch.device(tensor_meta['device'])
                contiguous_tensor = contiguous_tensor.to(device)

        params[name] = contiguous_tensor

    return params


def dump_function_inputs(dump_dir, skip_data_tensor_name_list, **kwargs):
    """
    """
    os.makedirs(dump_dir, exist_ok=True)
    metadata = {
        'tensors': {},
        'non_tensors': {}
    }

    log_file = os.path.join(dump_dir, 'params_info.txt')
    with open(log_file, 'w') as log:
        for name, value in kwargs.items():
            log.write(f"Parameter: {name}\n")

            if isinstance(value, torch.Tensor):
                log.write(f"  Type: torch.Tensor\n")
                log.write(f"  StorageDataPtr: {value.storage().data_ptr()}\n")
                log.write(f"  DataPtr       : {value.data_ptr()}\n")
                log.write(f"  Shape: {tuple(value.shape)}\n")
                log.write(f"  Dtype: {value.dtype}\n")
                log.write(f"  Device: {value.device}\n")
                log.write(f"  Is contiguous: {value.is_contiguous()}\n")
                log.write(f"  Stride: {value.stride()}\n")

                tensor_np = None
                if name not in skip_data_tensor_name_list:
                    print(f"tensor {name} save meta info to txt and data to disk")
                    file_path = os.path.join(dump_dir, f"{name}.npy")
                    tensor_np = value.detach().cpu().numpy()
                    np.save(file_path, tensor_np)
                else:
                    print(f"tensor {name} only save meta info to txt")

                num_elements = value.numel()
                log.write(f"  Num elements: {num_elements}\n")

                if num_elements <= 10:
                    log.write(f"  Data: {tensor_np.tolist()}\n")
                else:
                    if name not in skip_data_tensor_name_list:
                        flat_data = tensor_np.ravel()
                        head = flat_data[:5].tolist()
                        tail = flat_data[-5:].tolist()
                        log.write(f"  Data (partial): head={head}, tail={tail}\n")
                        log.write(f"  Data range: min={np.min(tensor_np):.4f}, max={np.max(tensor_np):.4f}, mean={np.mean(tensor_np):.4f}\n")

                metadata['tensors'][name] = {
                    'shape': list(value.shape),
                    'dtype': str(value.dtype),
                    'device': str(value.device),
                    'file': f"{name}.npy",
                    'stride': value.stride(),
                    'is_contiguous': value.is_contiguous()
                }
            else:
                log.write(f"  Type: {type(value).__name__}\n")

                if value is None:
                    log.write(f"  Value: None\n")
                    metadata['non_tensors'][name] = None
                elif isinstance(value, tuple) or isinstance(value, list):
                    log.write(f"  Value[0] type: {type(value[0]).__name__}\n")
                    log.write(f"  Value: {value}\n")
                    metadata['non_tensors'][name] = list(value)
                else:
                    log.write(f"  Value: {value}\n")
                    metadata['non_tensors'][name] = value
        log.flush()

    with open(os.path.join(dump_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Parameter exported to: {dump_dir}")
    print(f"Detailed information refer to: {log_file}")
    sys.stdout.flush()


def load_input_and_run_test(use_ori_qkv_shape=True, use_ori_vllm_impl=False):
    def create_tensor(min, max, *args, **kwargs):
        x = torch.randn(*args, **kwargs)
        x = (x - x.min()) / (x.max() - x.min())
        return min + (max - min) * x

    # request_id_list = list(range(0, 3))
    # request_id_list = list(range(0, 111))
    # request_id_list = list(range(0, 108))
    request_id_list = list(range(0, 135))

    selected_layer_id = [0, 31]

    # skip_data_tensor_name_list = []
    skip_data_tensor_name_list = ["q", "k", "v", "out"]

    root_data_dir = "/data/heyanguang/code/vllm_fa_batch_prefill/aiter/log_run/flash_attn_varlen_func_impl_dump_data/Llama-3.1-8B-Instruct-FP8-KV/v2_format_prompts_1000_part1/"
    # root_data_dir = "/data/heyanguang/code/vllm_fa_batch_prefill/aiter/log_run/flash_attn_varlen_func_impl_dump_data/Llama-3.1-8B-Instruct-FP8-KV/v1_format_prompts_1000_part1/"
    for request_id in request_id_list:
        for layer_id in selected_layer_id:
            # root_data_dir = "/data/heyanguang/code/vllm_fa_batch_prefill/aiter/log_run/"
            # dump_dir = root_data_dir + "function_inputs_dump"

            dump_dir = root_data_dir + "prompt_" + str(request_id) + "/layer_" + str(layer_id)
            print(f"dump_dir={dump_dir}")
            params = load_function_inputs(dump_dir, skip_data_tensor_name_list)

            # Print loaded parameters for verification
            print("Loaded parameters summary:")
            for name, value in params.items():
                if isinstance(value, torch.Tensor):
                    print(f"{name}: tensor(shape={value.shape}, dtype={value.dtype}, device={value.device}, "
                        f"contiguous={value.is_contiguous()}, stride={value.stride()})")
                elif isinstance(value, tuple) or isinstance(value, list):
                    print(f"{name}: {type(value).__name__} = {value}")
                    print(f"value[0] type: {type(value[0]).__name__}")
                else:
                    print(f"{name}: {type(value).__name__} = {value}")

            q = None
            k_cache = None
            k_cache = None
            kv_indices_gpu = None

            # Reconstruct the original function call
            q_read=params['q']
            k_read=params['k']
            v_read=params['v']
            q_indptr_gpu=params['cu_seqlens_q']
            kv_indptr_gpu=params['kv_indptr']
            block_table_read=params['kv_page_indices']
            kv_page_indices_vllm=params['kv_page_indices_vllm']
            alibi_slopes=params['alibi_slopes']
            out=params['out']
            total_tokens=params['total_tokens']
            max_seqlen_q=params['max_seqlen_q']
            max_seqlen_k=params['max_seqlen_k']
            softmax_scale=params['softmax_scale']
            window_size=params['window_size']
            causal=params['causal']
            is_chunked_prefill=params['is_chunked_prefill']
            dtype = k_read.dtype
            device = k_read.device

            q_indptr_cpu = q_indptr_gpu.detach().cpu()
            kv_indptr_cpu = kv_indptr_gpu.detach().cpu()

            q_lens = []
            for idx in range(1, len(q_indptr_cpu)):
                q_len = q_indptr_cpu[idx] - q_indptr_cpu[idx - 1]
                q_len = q_len.item()
                q_lens.append(q_len)
            kv_lens = []
            for idx in range(1, len(kv_indptr_cpu)):
                kv_len = kv_indptr_cpu[idx] - kv_indptr_cpu[idx - 1]
                kv_len = kv_len.item()
                kv_lens.append(kv_len)
            kv_lens = torch.tensor(kv_lens, dtype=torch.int32)

            batch_size = block_table_read.shape[0]
            page_size = k_read.shape[1]
            num_q_heads = q_read.shape[1]
            num_kv_heads, head_dim = k_read.shape[-2:]
            kv_last_page_len_cpu = ((kv_lens - 1) % page_size + 1).int()

            print(f"use_ori_qkv_shape={use_ori_qkv_shape}")
            if use_ori_qkv_shape:
                q = q_read
                k_cache = k_read
                v_cache = v_read
                kv_indices_gpu = block_table_read
            else:
                max_q_len = max(q_lens)
                max_kv_len = max(kv_lens)
                q_init_min, q_init_max = (-10, 10)
                kv_init_min, kv_init_max = (-5, 5)
                max_num_kv_pages_per_seq = (max_kv_len + page_size - 1) // page_size
                total_num_pages = batch_size * max_num_kv_pages_per_seq
                kv_shape = [2, total_num_pages, page_size, num_kv_heads, head_dim]
                q_create = create_tensor(
                    q_init_min, q_init_max, batch_size * max_q_len, num_q_heads, head_dim, dtype=dtype, device=device)
                kv_create = create_tensor(
                    kv_init_min, kv_init_max, kv_shape, dtype=dtype, device=device)
                chunks = torch.chunk(kv_create, 2, dim=0)
                k_create = chunks[0].squeeze(0)
                v_create = chunks[1].squeeze(0)
                q = q_create
                k_cache = k_create
                v_cache = v_create

                kv_indices_pad_len = 0
                block_table_create = torch.nn.functional.pad(
                    torch.randperm(total_num_pages).int().view(batch_size, max_num_kv_pages_per_seq), (0, kv_indices_pad_len), value=0
                ).to(device)
                kv_indices_gpu = block_table_create
            kv_indices_cpu = kv_indices_gpu.detach().cpu()

            o_ck_flash_attn, avg_kt = run_mha_batch_prefill_func(
                q,
                k_cache,
                v_cache,
                q_indptr_gpu,
                kv_indptr_gpu,
                kv_indices_gpu,
                max_seqlen_q,
                max_seqlen_k,
                total_tokens=total_tokens,
                causal=causal,
                is_chunked_prefill=is_chunked_prefill,
                use_ori_vllm_impl=use_ori_vllm_impl,
            )

            q_storage_data_ptr = q.storage().data_ptr()
            k_storage_data_ptr = k_cache.storage().data_ptr()
            v_storage_data_ptr = v_cache.storage().data_ptr()
            q_data_ptr = q.data_ptr()
            k_data_ptr = k_cache.data_ptr()
            v_data_ptr = v_cache.data_ptr()
            print(f"q_storage_data_ptr={q_storage_data_ptr}, q_data_ptr={q_data_ptr}, k_storage_data_ptr={k_storage_data_ptr}, k_data_ptr={k_data_ptr}, v_storage_data_ptr={v_storage_data_ptr}, v_data_ptr={v_data_ptr}")
            msg = f"\n[perf] q_shape: {str(q.shape):<20} kv_shape: {str(k_cache.shape):<20} q_dtype: {q.dtype} kv_dtype: {k_cache.dtype}, avg_kt_base: {avg_kt:<8.2f} us, avg_kt: {avg_kt:<8.2f} us, uplift: {avg_kt / avg_kt - 1:<5.1%}"
            print(msg)


            # k, v = vllm_layout_trans(kv_indptr_gpu, kv_indices_gpu, k_cache, v_cache,
            #                             max_seqlen_k, total_tokens)
            # ref_output = aiter.flash_attn_varlen_func(
            #     q=q,
            #     k=k,
            #     v=v,
            #     cu_seqlens_q=q_indptr_gpu,
            #     cu_seqlens_k=kv_indptr_gpu,
            #     max_seqlen_q=max_seqlen_q,
            #     max_seqlen_k=max_seqlen_k,
            #     softmax_scale=softmax_scale,
            #     causal=causal,
            #     alibi_slopes=alibi_slopes,
            #     window_size=window_size,
            #     out=out,
            # )

            # o_ck_flash_attn = o_ck_flash_attn.cpu()
            # head_dim = q.shape[2]
            # ref_output = ref_paged_attn(
            #     query=q.cpu(),
            #     key_cache=k_cache.cpu(),
            #     value_cache=v_cache.cpu(),
            #     query_lens=q_lens,
            #     kv_lens=kv_lens,
            #     block_tables=kv_indices_gpu.cpu(),
            #     scale=head_dim**-0.5,
            # )

            # watch_loc = [0, -1]
            # print(f"o_ck_flash_attn={o_ck_flash_attn[watch_loc]}")
            # print(f"ref_output={ref_output[watch_loc]}")
            # print(f"kv_indices_gpu={kv_indices_gpu}")
            # print(f"kv_page_indices_vllm={kv_page_indices_vllm}")
            print(f"k_cache.shape={k_cache.shape}")
            print(f"q_indptr_cpu={q_indptr_cpu}")
            print(f"kv_indptr_cpu={kv_indptr_cpu}")
            print(f"q_lens={q_lens}")
            print(f"kv_lens={kv_lens}")
            print(f"page_size={page_size}")
            # print(f"kv_last_page_len_cpu={kv_last_page_len_cpu}")

            # rtol, atol = (1e-3, 1e-3) if dtype == torch.float16 else (2e-2, 1e-2)
            # # rtol, atol = (1e-6, 1e-6) if dtype == torch.float16 else (2e-2, 1e-2)
            # torch.testing.assert_close(o_ck_flash_attn, ref_output, rtol=rtol, atol=atol)
            # print(f"PASSED")
            # continue


            for i in range(batch_size):
                if q_lens[i] == 1:
                    continue
                qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
                end_page_id = (kv_lens[i] + page_size - 1) // page_size
                used_kv_indices = kv_indices_cpu[i, : end_page_id]
                # print(f"used_kv_indices_extra={torch.cat([used_kv_indices, kv_indices_cpu[i, end_page_id : end_page_id + 6]], dim=-1)}")

                if len(used_kv_indices) == 1:
                    ki = torch.cat(
                        [
                            k_cache[used_kv_indices[-1], : kv_last_page_len_cpu[i]].reshape(-1, num_kv_heads, head_dim),
                        ],
                        dim=0,
                    )
                elif len(used_kv_indices) >= 2:
                    ki = torch.cat(
                        [
                            k_cache[used_kv_indices[:-1]].reshape(-1, num_kv_heads, head_dim),
                            k_cache[used_kv_indices[-1], : kv_last_page_len_cpu[i]].reshape(-1, num_kv_heads, head_dim),
                        ],
                        dim=0,
                    )
                else:
                    raise RuntimeError(f"len(used_kv_indices)={len(used_kv_indices)}, it must >= 1")

                if len(used_kv_indices) == 1:
                    vi = torch.cat(
                        [
                            v_cache[used_kv_indices[-1], : kv_last_page_len_cpu[i]].reshape(-1, num_kv_heads, head_dim),
                        ],
                        dim=0,
                    )
                elif len(used_kv_indices) >= 2:
                    vi = torch.cat(
                        [
                            v_cache[used_kv_indices[:-1]].reshape(-1, num_kv_heads, head_dim),
                            v_cache[used_kv_indices[-1], : kv_last_page_len_cpu[i]].reshape(-1, num_kv_heads, head_dim),
                        ],
                        dim=0,
                    )
                else:
                    raise RuntimeError(f"len(used_kv_indices)={len(used_kv_indices)}, it must >= 1")

                # print(f"qi.shape={qi.shape}")
                # print(f"ki.shape={ki.shape}")
                # print(f"vi.shape={vi.shape}")
                # print(f"ki.is_contiguous()={ki.is_contiguous()}")
                # print(f"qi={qi}")
                # print(f"ki={ki}")
                # print(f"vi={vi}")

                # enlarge rtol for bf16 to allow passing very few numeric errors
                rtol, atol = (1e-3, 1e-3) if dtype == torch.float16 else (2e-2, 1e-2)
                # rtol, atol = (1e-6, 1e-6) if dtype == torch.float16 else (2e-2, 1e-2)

                o_ref_i = ref_masked_attention(
                    qi, ki, vi, causal=causal,
                )
                o_i = o_ck_flash_attn[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
                # print(f"o_ref_i.shape={o_ref_i.shape}")
                # print(f"o_ref_i={o_ref_i}")
                # print(f"o_i={o_i}")

                torch.testing.assert_close(o_i, o_ref_i, rtol=rtol, atol=atol)
            print(f"PASSED")


@perftest(num_iters=TEST_NUM_ITERS)
def run_mha_batch_prefill_func(
    q,
    k,
    v,
    cu_seqlens_q,
    kv_indptr,
    kv_page_indices,
    max_seqlen_q,
    max_seqlen_k,
    total_tokens=-1,
    dropout_p=0.0,
    softmax_scale=None,
    logits_soft_cap=0.0,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_lse=False,
    return_attn_probs=False,
    is_chunked_prefill=False,
    use_ori_vllm_impl=False,
    out=None,
):
    output = None
    if use_ori_vllm_impl:
        k_trans, v_trans = vllm_layout_trans(kv_indptr, kv_page_indices, k, v,
                                    max_seqlen_k, total_tokens)
        output = aiter.flash_attn_varlen_func(
            q=q,
            k=k_trans,
            v=v_trans,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_k=kv_indptr,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            alibi_slopes=alibi_slopes,
            window_size=window_size,
            out=out,
        )
    else:
        output = aiter.mha_batch_prefill_func(
        q,
        k,
        v,
        cu_seqlens_q,
        kv_indptr,
        kv_page_indices,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        logits_soft_cap,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_lse,
        return_attn_probs,
        is_chunked_prefill,
        out)

    return output


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
    key_leftpad=None,
):
    row_idx = rearrange(
        torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1"
    )
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool = False,
    window_left: int = -1,
    logits_soft_cap: float = 0.0,
) -> torch.Tensor:
    if causal:
        window_size = (window_left, 0)
    else:
        window_size = (-1, -1)

    head_dim = query.shape[2]
    seqlen_q = query.shape[0]
    seqlen_k = key.shape[0]
    scale = 1.0 / math.sqrt(head_dim)
    # head_num = query.shape[1]
    head_num_kv = key.shape[1]
    # kv_repeat_count = head_num // head_num_kv
    # key = key.repeat(1, kv_repeat_count, 1)
    # value = value.repeat(1, kv_repeat_count, 1)

    # [seqlen_q, head_num, head_dim] --> [seqlen_q, head_num_kv, head_num / head_num_kv, head_dim]
    query = query.reshape(seqlen_q, head_num_kv, -1, head_dim)
    # [seqlen_k, head_num_kv, head_dim] --> [seqlen_k, head_num_kv, 1, head_dim]
    key = key.unsqueeze(2)
    value = value.unsqueeze(2)

    # [head_num_kv, head_num / head_num_kv, seqlen_q, seqlen_k]
    attn_weights = scale * torch.einsum("qmid,kmjd->miqk", query.float(), key.float())
    if 0 < logits_soft_cap:
        mode = int(os.environ.get("CK_TILE_ATTENTION_LOGITS_SOFT_CAP_DEFAULT", 0))
        if mode == 0:
            attn_weights = logits_soft_cap * torch.tanh(attn_weights / logits_soft_cap)
        else:
            attn_weights = attn_weights / (
                1.0 + torch.abs(attn_weights / logits_soft_cap)
            )

    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            device=query.device,
        )
        attn_weights.masked_fill_(local_mask, float("-inf"))
    attn_weights = torch.softmax(attn_weights, dim=-1)
    if window_size[0] >= 0 or window_size[1] >= 0:
        attn_weights = attn_weights.masked_fill(
            torch.all(local_mask, dim=-1, keepdim=True), 0.0
        )
    # [seqlen_q, head_num_kv, head_num / head_num_kv, head_dim]
    out = torch.einsum("miqk,kmjd->qmid", attn_weights, value.float())
    out = out.reshape(seqlen_q, -1, head_dim)
    return out.to(query)


@pytest.mark.parametrize("is_sglang_layout", [True, False])
@pytest.mark.parametrize("batch_size", [1, 3, 7])
@pytest.mark.parametrize(
    "qo_len,kv_len",
    [
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
        (1023, 1025),
    ],
)
@pytest.mark.parametrize("page_size", [1, 16, 32, 64])
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(6, 1), (3, 1)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("is_chunked_prefill", [False, True])
@pytest.mark.parametrize("kv_layout", ["NHD"])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
@pytest.mark.parametrize("contiguous_kv", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("q_init_min,q_init_max", [(-10, 10)])
@pytest.mark.parametrize("kv_init_min,kv_init_max", [(-5, 5)])
@pytest.mark.parametrize("seed", [19378])
def test_batch_prefill_with_paged_kv_cache(
    batch_size,
    kv_len,
    qo_len,
    page_size,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    causal,
    is_chunked_prefill,
    kv_layout,
    logits_soft_cap,
    contiguous_kv,
    dtype,
    q_init_min,
    q_init_max,
    kv_init_min,
    kv_init_max,
    seed,
    is_sglang_layout,
):
    if seed is not None:
        torch.manual_seed(seed)

    if causal and kv_len < qo_len:
        pytest.skip("kv_len < qo_len is not allowed if causal=True")

    if head_dim == 64 and qo_len <= 64:
        pytest.skip("Unsupported configuration")

    def create_tensor(min, max, *args, **kwargs):
        x = torch.randn(*args, **kwargs)
        x = (x - x.min()) / (x.max() - x.min())
        return min + (max - min) * x

    def convert_lens_to_indtpr(lens):
        return torch.cumsum(torch.cat((torch.tensor([0]), lens)), dim=0).int()
    print()

    q = create_tensor(
        q_init_min, q_init_max, batch_size * qo_len, num_qo_heads, head_dim, dtype=dtype
    ).to(0)
    if 1 < batch_size:
        qo_lens = torch.randint(1, qo_len + 1, (batch_size,)).int()
    else:
        qo_lens = torch.full((batch_size,), qo_len).int()
    q_indptr_cpu = convert_lens_to_indtpr(qo_lens)
    max_num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = max_num_pages_per_seq * batch_size
    # kv_shape = [total_num_pages, 2, num_kv_heads, page_size, head_dim]
    kv_shape = [2, total_num_pages, page_size, num_kv_heads, head_dim]
    # print(f"\ncontiguous_kv={contiguous_kv}")
    if not contiguous_kv:
        tmp = [kv_shape[0]]
        for v in kv_shape[1:]:
            tmp.append(2)
            tmp.append(v)
        kv_shape = tmp
        kv_data_fp32 = create_tensor(
            kv_init_min, kv_init_max, *kv_shape, dtype=torch.float32
        ).to(0)
        kv_data = kv_data_fp32.to(dtype)
        print(f"kv_data.shape={kv_data.shape}")
        kv_data = kv_data[:, 1, :, 1, :, 1, :, 1, :]
        print(f"kv_data.shape={kv_data.shape}")
        kv_data_fp32 = kv_data_fp32[:, 1, :, 1, :, 1, :, 1, :]
        # actual data is stored in non-contiguous memory
        assert (
            kv_data.stride(-4)
            != kv_data.shape[-3] * kv_data.shape[-2] * kv_data.shape[-1]
        )
    else:
        kv_data_fp32 = create_tensor(
            kv_init_min, kv_init_max, *kv_shape, dtype=torch.float32
        ).to(0)
        kv_data = kv_data_fp32.to(dtype)
    if 1 < batch_size:
        kv_lens = torch.maximum(
            qo_lens, torch.randint(1, kv_len + 1, (batch_size,))
        ).int()
    else:
        kv_lens = torch.full((batch_size,), kv_len).int()
    # kv_num_used_pages = (kv_lens + page_size - 1) // page_size
    # kv_indptr_cpu = convert_lens_to_indtpr(kv_num_used_pages)
    kv_indptr_cpu = convert_lens_to_indtpr(kv_lens)

    # kv_indices_pad_len = 128
    kv_indices_pad_len = 0
    kv_indices_cpu = None
    # kv_num_used_pages = (kv_lens + page_size - 1) // page_size
    # kv_indices_cpu = convert_lens_to_indtpr(kv_num_used_pages)
    if is_sglang_layout:
        kv_indices_cpu = torch.nn.functional.pad(
            torch.randperm(total_num_pages).int(), (0, kv_indices_pad_len), value=0
        )
    else:
        kv_indices_cpu = torch.nn.functional.pad(
            torch.randperm(total_num_pages).int().view(batch_size, max_num_pages_per_seq), (0, kv_indices_pad_len), value=0
        )

    kv_last_page_len_cpu = ((kv_lens - 1) % page_size + 1).int()
    # print(f"qo_lens={qo_lens}")
    # print(f"kv_lens={kv_lens}")
    # print(f"kv_last_page_len_cpu={kv_last_page_len_cpu}")
    # print(f"q_indptr_cpu={q_indptr_cpu}")
    # print(f"kv_indptr_cpu={kv_indptr_cpu}")
    # print(f"kv_indices_cpu.shape={kv_indices_cpu.shape}")
    # print(f"kv_data.shape={kv_data.shape}")

    q_indptr_gpu = q_indptr_cpu.to(0)
    kv_indptr_gpu = kv_indptr_cpu.to(0)
    kv_indices_gpu = kv_indices_cpu.to(0)

    # chunks = torch.chunk(kv_data, 2, dim=1)
    # k_cache = chunks[0].squeeze(2).squeeze(2)
    # v_cache = chunks[1].squeeze(2).squeeze(2)

    chunks = torch.chunk(kv_data, 2, dim=0)
    k_cache = chunks[0].squeeze(0)
    v_cache = chunks[1].squeeze(0)

    print(f"kv_data_fp32.shape={kv_data_fp32.shape}")
    # # print(f"type(chunks)={type(chunks)}")
    # print(f"chunks[0].shape={chunks[0].shape}")
    # print(f"q.shape={q.shape}")
    # print(f"k_cache.shape={k_cache.shape}")

    # if batch_size == 1:
    #     o_ck_flash_attn_vllm = aiter.mha_batch_prefill_func(
    #         q,
    #         k_cache,
    #         v_cache,
    #         q_indptr_gpu,
    #         kv_indptr_gpu,
    #         kv_indices_gpu.unsqueeze(0),
    #         torch.max(qo_lens).item(),
    #         torch.max(kv_lens).item(),
    #         causal=causal,
    #         is_chunked_prefill=is_chunked_prefill,
    #         logits_soft_cap=logits_soft_cap,
    #     )
    # o_ck_flash_attn = aiter.mha_batch_prefill_func(
    #     q,
    #     k_cache,
    #     v_cache,
    #     q_indptr_gpu,
    #     kv_indptr_gpu,
    #     kv_indices_gpu,
    #     torch.max(qo_lens).item(),
    #     torch.max(kv_lens).item(),
    #     causal=causal,
    #     is_chunked_prefill=is_chunked_prefill,
    #     logits_soft_cap=logits_soft_cap,
    # )

    o_ck_flash_attn, avg_kt = run_mha_batch_prefill_func(
        q,
        k_cache,
        v_cache,
        q_indptr_gpu,
        kv_indptr_gpu,
        kv_indices_gpu,
        torch.max(qo_lens).item(),
        torch.max(kv_lens).item(),
        logits_soft_cap=logits_soft_cap,
        causal=causal,
        is_chunked_prefill=is_chunked_prefill,
        use_ori_vllm_impl=False,
    )

    # o_ck_flash_attn = q.clone()
    # avg_kt = 1.0
    msg = f"\n[perf] q_shape: {str(q.shape):<20} kv_shape: {str(k_cache.shape):<20} q_dtype: {q.dtype} kv_dtype: {k_cache.dtype}, avg_kt_base: {avg_kt:<8.2f} us, avg_kt: {avg_kt:<8.2f} us, uplift: {avg_kt / avg_kt - 1:<5.1%}"
    print(msg)

    for i in range(batch_size):
        perm_dims = [0, 2, 1, 3] if kv_layout == "HND" else [0, 1, 2, 3]
        perm_dims_last = [1, 0, 2] if kv_layout == "HND" else [0, 1, 2]
        qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        end_page_id = (kv_lens[i] + page_size - 1) // page_size
        if is_sglang_layout:
            used_kv_indices = kv_indices_cpu[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1]]
        else:
            used_kv_indices = kv_indices_cpu[i, : end_page_id]
        # print(f"end_page_id={end_page_id}")
        # # print(f"used_kv_indices={used_kv_indices}")
        # print(f"len(used_kv_indices)={len(used_kv_indices)}")

        # ki = torch.cat(
        #     [
        #         kv_data_fp32[used_kv_indices[:-1], 0]
        #         .permute(*perm_dims)
        #         .reshape(-1, num_kv_heads, head_dim),
        #         (
        #             kv_data_fp32[used_kv_indices[-1], 0, :, : kv_last_page_len_cpu[i]]
        #             if kv_layout == "HND"
        #             else kv_data_fp32[
        #                 used_kv_indices[-1], 0, : kv_last_page_len_cpu[i], :
        #             ]
        #         )
        #         .permute(*perm_dims_last)
        #         .reshape(-1, num_kv_heads, head_dim),
        #     ],
        #     dim=0,
        # ).to(dtype)
        # vi = torch.cat(
        #     [
        #         kv_data_fp32[used_kv_indices[:-1], 1]
        #         .permute(*perm_dims)
        #         .reshape(-1, num_kv_heads, head_dim),
        #         (
        #             kv_data_fp32[used_kv_indices[-1], 1, :, : kv_last_page_len_cpu[i]]
        #             if kv_layout == "HND"
        #             else kv_data_fp32[
        #                 used_kv_indices[-1], 1, : kv_last_page_len_cpu[i], :
        #             ]
        #         )
        #         .permute(*perm_dims_last)
        #         .reshape(-1, num_kv_heads, head_dim),
        #     ],
        #     dim=0,
        # ).to(dtype)

        if len(used_kv_indices) == 1:
            ki = torch.cat(
                [
                    kv_data_fp32[0, used_kv_indices[-1], : kv_last_page_len_cpu[i]].reshape(-1, num_kv_heads, head_dim),
                ],
                dim=0,
            ).to(dtype)
        elif len(used_kv_indices) >= 2:
            ki = torch.cat(
                [
                    kv_data_fp32[0, used_kv_indices[:-1]].reshape(-1, num_kv_heads, head_dim),
                    kv_data_fp32[0, used_kv_indices[-1], : kv_last_page_len_cpu[i]].reshape(-1, num_kv_heads, head_dim),
                ],
                dim=0,
            ).to(dtype)
        else:
            raise RuntimeError(f"len(used_kv_indices)={len(used_kv_indices)}, it must >= 1")

        if len(used_kv_indices) == 1:
            vi = torch.cat(
                [
                    kv_data_fp32[1, used_kv_indices[-1], : kv_last_page_len_cpu[i]].reshape(-1, num_kv_heads, head_dim),
                ],
                dim=0,
            ).to(dtype)
        elif len(used_kv_indices) >= 2:
            vi = torch.cat(
                [
                    kv_data_fp32[1, used_kv_indices[:-1]].reshape(-1, num_kv_heads, head_dim),
                    kv_data_fp32[1, used_kv_indices[-1], : kv_last_page_len_cpu[i]].reshape(-1, num_kv_heads, head_dim),
                ],
                dim=0,
            ).to(dtype)
        else:
            raise RuntimeError(f"len(used_kv_indices)={len(used_kv_indices)}, it must >= 1")

        # print(f"qi.shape={qi.shape}")
        # print(f"ki.shape={ki.shape}")
        # print(f"vi.shape={vi.shape}")
        # print(f"ki.is_contiguous()={ki.is_contiguous()}")

        # enlarge rtol for bf16 to allow passing very few numeric errors
        rtol, atol = (1e-3, 1e-3) if dtype == torch.float16 else (2e-2, 1e-2)

        o_ref_i = ref_masked_attention(
            qi, ki, vi, causal=causal, logits_soft_cap=logits_soft_cap
        )
        o_i = o_ck_flash_attn[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]

        # import pdb
        # pdb.set_trace()
        # print(f"o_ref_i.shape={o_ref_i.shape}")
        # print(f"o_i.shape={o_i.shape}")
        # print(f"o_ref_i={o_ref_i[-4:, 0, :4]}")
        # print(f"o_i={o_i[-4:, 0, :4]}")
        torch.testing.assert_close(o_i, o_ref_i, rtol=rtol, atol=atol)

@pytest.mark.parametrize(
    "qo_len,kv_len",
    [
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (8192, 8192),
    ],
)
@pytest.mark.parametrize("head_dim", [128,])
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(8, 1), (8, 2), (32, 8)])
@pytest.mark.parametrize("batch_size", [1, 7])
# @pytest.mark.parametrize("batch_size", [1, 3, 7])
@pytest.mark.parametrize("use_ori_vllm_impl", [True, False])
@pytest.mark.parametrize("is_sglang_layout", [True, False])
# @pytest.mark.parametrize(
#     "qo_len,kv_len",
#     [
#         (1024, 1024),
#         (1023, 1024),
#         (1024, 1023),
#         (2048, 2048),
#         (1023, 1025),
#         (8192, 8192),
#     ],
# )
@pytest.mark.parametrize("page_size", [1, 16, 32, 64])
# @pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(6, 1), (3, 1)])
# @pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(6, 1), (3, 1), (32, 8), (6, 2)])
# @pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("is_chunked_prefill", [False, True])
@pytest.mark.parametrize("kv_layout", ["NHD"])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
@pytest.mark.parametrize("contiguous_kv", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("q_init_min,q_init_max", [(-10, 10)])
@pytest.mark.parametrize("kv_init_min,kv_init_max", [(-5, 5)])
@pytest.mark.parametrize("seed", [19378])
def test_batch_prefill_with_paged_kv_cache_v2(
    batch_size,
    kv_len,
    qo_len,
    page_size,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    causal,
    is_chunked_prefill,
    kv_layout,
    logits_soft_cap,
    contiguous_kv,
    dtype,
    q_init_min,
    q_init_max,
    kv_init_min,
    kv_init_max,
    seed,
    is_sglang_layout,
    use_ori_vllm_impl,
):
    if seed is not None:
        torch.manual_seed(seed)

    if causal and kv_len < qo_len:
        pytest.skip("kv_len < qo_len is not allowed if causal=True")

    if head_dim == 64 and qo_len <= 64:
        pytest.skip("Unsupported configuration")

    def create_tensor(min, max, *args, **kwargs):
        x = torch.randn(*args, **kwargs)
        x = (x - x.min()) / (x.max() - x.min())
        return min + (max - min) * x

    def convert_lens_to_indtpr(lens):
        return torch.cumsum(torch.cat((torch.tensor([0]), lens)), dim=0).int()
    print()

    q = create_tensor(
        q_init_min, q_init_max, batch_size * qo_len, num_qo_heads, head_dim, dtype=dtype
    ).to(0)
    if 1 < batch_size:
        qo_lens = torch.randint(1, qo_len + 1, (batch_size,)).int()
    else:
        qo_lens = torch.full((batch_size,), qo_len).int()
    q_indptr_cpu = convert_lens_to_indtpr(qo_lens)
    max_num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = max_num_pages_per_seq * batch_size
    kv_shape = [2, total_num_pages, page_size, num_kv_heads, head_dim]
    # print(f"\ncontiguous_kv={contiguous_kv}")
    if not contiguous_kv:
        tmp = [kv_shape[0]]
        for v in kv_shape[1:]:
            tmp.append(2)
            tmp.append(v)
        kv_shape = tmp
        kv_data_fp32 = create_tensor(
            kv_init_min, kv_init_max, *kv_shape, dtype=torch.float32
        ).to(0)
        kv_data = kv_data_fp32.to(dtype)
        print(f"kv_data.shape={kv_data.shape}")
        kv_data = kv_data[:, 1, :, 1, :, 1, :, 1, :]
        print(f"kv_data.shape={kv_data.shape}")
        kv_data_fp32 = kv_data_fp32[:, 1, :, 1, :, 1, :, 1, :]
        # actual data is stored in non-contiguous memory
        assert (
            kv_data.stride(-4)
            != kv_data.shape[-3] * kv_data.shape[-2] * kv_data.shape[-1]
        )
    else:
        kv_data_fp32 = create_tensor(
            kv_init_min, kv_init_max, *kv_shape, dtype=torch.float32
        ).to(0)
        kv_data = kv_data_fp32.to(dtype)
    if 1 < batch_size:
        kv_lens = torch.maximum(
            qo_lens, torch.randint(1, kv_len + 1, (batch_size,))
        ).int()
    else:
        kv_lens = torch.full((batch_size,), kv_len).int()
    kv_indptr_cpu = convert_lens_to_indtpr(kv_lens)

    # kv_indices_pad_len = 128
    kv_indices_pad_len = 0
    kv_indices_cpu = None
    if is_sglang_layout:
        kv_indices_cpu = torch.nn.functional.pad(
            torch.randperm(total_num_pages).int(), (0, kv_indices_pad_len), value=0
        )
    else:
        kv_indices_cpu = torch.nn.functional.pad(
            torch.randperm(total_num_pages).int().view(batch_size, max_num_pages_per_seq), (0, kv_indices_pad_len), value=0
        )

    kv_last_page_len_cpu = ((kv_lens - 1) % page_size + 1).int()
    q_indptr_gpu = q_indptr_cpu.to(0)
    kv_indptr_gpu = kv_indptr_cpu.to(0)
    kv_indices_gpu = kv_indices_cpu.to(0)

    chunks = torch.chunk(kv_data, 2, dim=0)
    k_cache = chunks[0].squeeze(0)
    v_cache = chunks[1].squeeze(0)

    # print(f"kv_data_fp32.shape={kv_data_fp32.shape}")
    # print(f"q.shape={q.shape}")
    # print(f"k_cache.shape={k_cache.shape}")

    # root_data_dir = "/data/heyanguang/code/vllm_fa_batch_prefill/aiter/log_run/"
    # dump_dir = root_data_dir + "function_inputs_dump"
    # dump_function_inputs(
    #     dump_dir,
    #     q=q,
    #     k=k_cache,
    #     v=v_cache,
    #     cu_seqlens_q=q_indptr_gpu,
    #     kv_indptr=kv_indptr_gpu,
    #     kv_page_indices=kv_indices_gpu,
    #     max_seqlen_q=torch.max(qo_lens).item(),
    #     max_seqlen_k=torch.max(kv_lens).item(),
    #     causal=causal,
    #     is_chunked_prefill=is_chunked_prefill,
    #     # logits_soft_cap=logits_soft_cap,
    # )

    o_ck_flash_attn, avg_kt = run_mha_batch_prefill_func(
        q,
        k_cache,
        v_cache,
        q_indptr_gpu,
        kv_indptr_gpu,
        kv_indices_gpu,
        torch.max(qo_lens).item(),
        torch.max(kv_lens).item(),
        total_tokens=kv_indptr_cpu[-1],
        logits_soft_cap=logits_soft_cap,
        causal=causal,
        is_chunked_prefill=is_chunked_prefill,
        use_ori_vllm_impl=use_ori_vllm_impl,
    )

    q_storage_data_ptr = q.storage().data_ptr()
    k_storage_data_ptr = k_cache.storage().data_ptr()
    v_storage_data_ptr = v_cache.storage().data_ptr()
    q_data_ptr = q.data_ptr()
    k_data_ptr = k_cache.data_ptr()
    v_data_ptr = v_cache.data_ptr()
    print(f"q_storage_data_ptr={q_storage_data_ptr}, q_data_ptr={q_data_ptr}, k_storage_data_ptr={k_storage_data_ptr}, k_data_ptr={k_data_ptr}, v_storage_data_ptr={v_storage_data_ptr}, v_data_ptr={v_data_ptr}")
    msg = f"\n[perf] q_shape: {str(q.shape):<20} kv_shape: {str(k_cache.shape):<20} q_dtype: {q.dtype} kv_dtype: {k_cache.dtype}, avg_kt_base: {avg_kt:<8.2f} us, avg_kt: {avg_kt:<8.2f} us, uplift: {avg_kt / avg_kt - 1:<5.1%}"
    print(msg)

    # print(f"qo_lens={qo_lens}")
    # print(f"kv_lens={kv_lens}")
    # print(f"page_size={page_size}")
    # print(f"kv_last_page_len_cpu={kv_last_page_len_cpu}")

    for i in range(batch_size):
        qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        end_page_id = (kv_lens[i] + page_size - 1) // page_size
        if is_sglang_layout:
            used_kv_indices = kv_indices_cpu[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1]]
        else:
            used_kv_indices = kv_indices_cpu[i, : end_page_id]
        # print(f"end_page_id={end_page_id}")
        # print(f"len(used_kv_indices)={len(used_kv_indices)}")
        # print(f"used_kv_indices={used_kv_indices}")

        if len(used_kv_indices) == 1:
            ki = torch.cat(
                [
                    kv_data_fp32[0, used_kv_indices[-1], : kv_last_page_len_cpu[i]].reshape(-1, num_kv_heads, head_dim),
                ],
                dim=0,
            ).to(dtype)
        elif len(used_kv_indices) >= 2:
            ki = torch.cat(
                [
                    kv_data_fp32[0, used_kv_indices[:-1]].reshape(-1, num_kv_heads, head_dim),
                    kv_data_fp32[0, used_kv_indices[-1], : kv_last_page_len_cpu[i]].reshape(-1, num_kv_heads, head_dim),
                ],
                dim=0,
            ).to(dtype)
        else:
            raise RuntimeError(f"len(used_kv_indices)={len(used_kv_indices)}, it must >= 1")

        if len(used_kv_indices) == 1:
            vi = torch.cat(
                [
                    kv_data_fp32[1, used_kv_indices[-1], : kv_last_page_len_cpu[i]].reshape(-1, num_kv_heads, head_dim),
                ],
                dim=0,
            ).to(dtype)
        elif len(used_kv_indices) >= 2:
            vi = torch.cat(
                [
                    kv_data_fp32[1, used_kv_indices[:-1]].reshape(-1, num_kv_heads, head_dim),
                    kv_data_fp32[1, used_kv_indices[-1], : kv_last_page_len_cpu[i]].reshape(-1, num_kv_heads, head_dim),
                ],
                dim=0,
            ).to(dtype)
        else:
            raise RuntimeError(f"len(used_kv_indices)={len(used_kv_indices)}, it must >= 1")

        # print(f"qi.shape={qi.shape}")
        # print(f"ki.shape={ki.shape}")
        # print(f"vi.shape={vi.shape}")
        # print(f"ki.is_contiguous()={ki.is_contiguous()}")

        # enlarge rtol for bf16 to allow passing very few numeric errors
        rtol, atol = (1e-3, 1e-3) if dtype == torch.float16 else (2e-2, 1e-2)
        # rtol, atol = (1e-6, 1e-6) if dtype == torch.float16 else (2e-2, 1e-2)

        o_ref_i = ref_masked_attention(
            qi, ki, vi, causal=causal, logits_soft_cap=logits_soft_cap
        )
        o_i = o_ck_flash_attn[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        # print(f"o_ref_i.shape={o_ref_i.shape}")

        torch.testing.assert_close(o_i, o_ref_i, rtol=rtol, atol=atol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--use_ori_qkv_shape',
        action='store_true',
        default=False,
        help='use_ori_qkv_shape'
    )
    parser.add_argument(
        '--use_ori_vllm_impl',
        action='store_true',
        default=False,
        help='use_ori_vllm_impl'
    )

    args = parser.parse_args()
    use_ori_qkv_shape = args.use_ori_qkv_shape
    use_ori_vllm_impl = args.use_ori_vllm_impl

    # for (
    #     causal,
    #     logits_soft_cap,
    #     dtype,
    #     is_chunked_prefill,
    # ) in itertools.product([True, False], [0.0, 30.0], [torch.float16, torch.bfloat16], [True, False]):
    #     test_batch_prefill_with_paged_kv_cache(
    #         batch_size=1,
    #         kv_len=8192,
    #         qo_len=8192,
    #         page_size=1,
    #         num_qo_heads=6,
    #         num_kv_heads=1,
    #         head_dim=128,
    #         causal=causal,
    #         is_chunked_prefill=is_chunked_prefill,
    #         kv_layout="NHD",
    #         logits_soft_cap=logits_soft_cap,
    #         contiguous_kv=True,
    #         dtype=dtype,
    #         q_init_min=-10,
    #         q_init_max=10,
    #         kv_init_min=-5,
    #         kv_init_max=5,
    #         seed=19378,
    #     )

    load_input_and_run_test(use_ori_qkv_shape=use_ori_qkv_shape, use_ori_vllm_impl=use_ori_vllm_impl)
    pass

