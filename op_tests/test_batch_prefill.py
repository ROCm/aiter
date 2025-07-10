# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import itertools
import math
import os
import pytest
import torch
import json
import numpy as np

import aiter
from einops import rearrange, repeat
from aiter.test_common import perftest

TEST_NUM_ITERS = 100
# TEST_NUM_ITERS = 2


def load_function_inputs(dump_dir):
    """
    Loads function inputs from the specified directory,
    fully restoring tensor layouts (including non-contiguous ones).
    """
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


def dump_function_inputs(dump_dir, **kwargs):
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

                file_path = os.path.join(dump_dir, f"{name}.npy")
                tensor_np = value.detach().cpu().numpy()
                np.save(file_path, tensor_np)

                num_elements = value.numel()
                log.write(f"  Num elements: {num_elements}\n")

                if num_elements <= 10:
                    log.write(f"  Data: {tensor_np.tolist()}\n")
                else:
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

    with open(os.path.join(dump_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Parameter exported to: {dump_dir}")
    print(f"Detailed information refer to: {log_file}")


def load_input_and_run_test():
    # root_data_dir = "/data/heyanguang/code/vllm_fa_batch_prefill/aiter/log_run/"
    # dump_dir = root_data_dir + "function_inputs_dump"

    # request_id_list = list(range(0, 27))
    request_id_list = list(range(0, 6))
    selected_layer_id = [0, 31]
    root_data_dir = "/data/heyanguang/code/vllm_fa_batch_prefill/aiter/log_run/flash_attn_varlen_func_impl_dump_data/Llama-3.1-8B-Instruct-FP8-KV/"
    for request_id in request_id_list:
        for layer_id in selected_layer_id:
            dump_dir = root_data_dir + "prompt_" + str(request_id) + "/layer_" + str(layer_id)
            params = load_function_inputs(dump_dir)

            # Print loaded parameters for verification
            print(f"dump_dir={dump_dir}")
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

            # Reconstruct the original function call
            q=params['q']
            k_cache=params['k']
            v_cache=params['v']
            q_indptr_gpu=params['cu_seqlens_q']
            kv_indptr_gpu=params['kv_indptr']
            kv_indices_gpu=params['kv_page_indices']
            max_seqlen_q=params['max_seqlen_q']
            max_seqlen_k=params['max_seqlen_k']
            causal=params['causal']
            is_chunked_prefill=params['is_chunked_prefill']

            o_ck_flash_attn, avg_kt = run_mha_batch_prefill_func(
                q,
                k_cache,
                v_cache,
                q_indptr_gpu,
                kv_indptr_gpu,
                kv_indices_gpu,
                max_seqlen_q,
                max_seqlen_k,
                causal=causal,
                is_chunked_prefill=is_chunked_prefill,
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

            q_indptr_cpu = q_indptr_gpu.detach().cpu()
            kv_indptr_cpu = kv_indptr_gpu.detach().cpu()
            kv_indices_cpu = kv_indices_gpu.detach().cpu()
            kv_lens = []
            for idx in range(1, len(kv_indptr_cpu)):
                kv_len = kv_indptr_cpu[idx] - kv_indptr_cpu[idx - 1]
                kv_lens.append(kv_len)
            kv_lens = torch.tensor(kv_lens, dtype=torch.int32)

            batch_size = kv_indices_gpu.shape[0]
            page_size = k_cache.shape[1]
            num_kv_heads, head_dim = k_cache.shape[-2:]
            kv_last_page_len_cpu = ((kv_lens - 1) % page_size + 1).int()
            dtype = k_cache.dtype

            # print(f"kv_indices_gpu={kv_indices_gpu}")
            # print(f"k_cache.shape={k_cache.shape}")
            # print(f"q_indptr_cpu={q_indptr_cpu}")
            # print(f"kv_indptr_cpu={kv_indptr_cpu}")
            print(f"kv_lens={kv_lens}")
            print(f"page_size={page_size}")
            # print(f"kv_last_page_len_cpu={kv_last_page_len_cpu}")

            for i in range(batch_size):
                qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
                end_page_id = (kv_lens[i] + page_size - 1) // page_size
                used_kv_indices = kv_indices_cpu[i, : end_page_id]
                print(f"used_kv_indices_extra={torch.cat([used_kv_indices, kv_indices_cpu[i, end_page_id : end_page_id + 6]], dim=-1)}")

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

                # print(f"qi={qi}")
                # print(f"ki={ki}")
                # print(f"vi={vi}")
                # print(f"qi.shape={qi.shape}")
                # print(f"ki.shape={ki.shape}")
                # print(f"vi.shape={vi.shape}")
                # print(f"ki.is_contiguous()={ki.is_contiguous()}")

                # enlarge rtol for bf16 to allow passing very few numeric errors
                rtol, atol = (1e-3, 1e-3) if dtype == torch.float16 else (2e-2, 1e-2)
                # rtol, atol = (1e-6, 1e-6) if dtype == torch.float16 else (2e-2, 1e-2)

                o_ref_i = ref_masked_attention(
                    qi, ki, vi, causal=causal,
                )
                o_i = o_ck_flash_attn[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
                # print(f"o_ref_i.shape={o_ref_i.shape}")

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
    out=None,
):
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
        causal=causal,
        is_chunked_prefill=is_chunked_prefill,
        logits_soft_cap=logits_soft_cap)
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
        (8192, 8192),
    ],
)
@pytest.mark.parametrize("page_size", [1, 16, 32, 64])
# @pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(6, 1), (3, 1)])
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(6, 1), (3, 1), (32, 8), (6, 2)])
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

    root_data_dir = "/data/heyanguang/code/vllm_fa_batch_prefill/aiter/log_run/"
    dump_dir = root_data_dir + "function_inputs_dump"
    dump_function_inputs(
        dump_dir,
        q=q,
        k=k_cache,
        v=v_cache,
        cu_seqlens_q=q_indptr_gpu,
        kv_indptr=kv_indptr_gpu,
        kv_page_indices=kv_indices_gpu,
        max_seqlen_q=torch.max(qo_lens).item(),
        max_seqlen_k=torch.max(kv_lens).item(),
        causal=causal,
        is_chunked_prefill=is_chunked_prefill,
        # logits_soft_cap=logits_soft_cap,
    )

    o_ck_flash_attn, avg_kt = run_mha_batch_prefill_func(
        q,
        k_cache,
        v_cache,
        q_indptr_gpu,
        kv_indptr_gpu,
        kv_indices_gpu,
        torch.max(qo_lens).item(),
        torch.max(kv_lens).item(),
        causal=causal,
        is_chunked_prefill=is_chunked_prefill,
        logits_soft_cap=logits_soft_cap)

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

    load_input_and_run_test()

