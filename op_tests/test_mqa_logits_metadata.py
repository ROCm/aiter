# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import random

import aiter
from aiter.test_common import checkAllclose, benchmark, perftest, run_perftest
from aiter.ops.triton.pa_mqa_logits import (
    # deepgemm_fp8_paged_mqa_logits_ragged_k,
    deepgemm_fp8_paged_mqa_logits,
)


def cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


def kv_cache_cast_to_fp8(x: torch.Tensor, padding=False) -> torch.Tensor:
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 240.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fnuz)

    padding_size = 0 if not padding else (16 - (block_size * 4) % 16) % 16
    x_fp8 = torch.empty(
        (num_blocks, block_size * (head_dim + 4 + padding_size)),
        device=x.device,
        dtype=torch.uint8,
    )
    x_fp8[:, : block_size * head_dim] = x_scaled.view(
        num_blocks, block_size * head_dim
    ).view(dtype=torch.uint8)
    x_fp8[:, block_size * head_dim : block_size * head_dim + 4 * block_size] = sf.view(
        num_blocks, block_size
    ).view(dtype=torch.uint8)
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4 + padding_size)


def count_bytes(*tensors):
    total = 0
    for t in tensors:
        if isinstance(t, (tuple, list)):
            total += count_bytes(*t)
        elif t is not None:
            total += t.numel() * t.element_size()
    return total


def ref_fp8_paged_mqa_logits_ragged(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    prefix_sum_context_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    max_model_len: int,
):
    batch_size, next_n, heads, dim = q.size()
    seq_kv, _, dim = kv_cache.size()  # 3d
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    prefix_sum_context_lens = prefix_sum_context_lens.tolist()
    for i in range(batch_size):
        context_len = prefix_sum_context_lens[i + 1] - prefix_sum_context_lens[i]
        q_offsets = torch.arange(context_len - next_n, context_len, device="cuda")
        qx, kx = (
            q[i],
            kv_cache[kv_indices[prefix_sum_context_lens[i] : prefix_sum_context_lens[i + 1]]],
        )
        k_offsets = torch.arange(0, context_len, device="cuda")
        mask = (k_offsets[None, :] < context_len) & (k_offsets[None, :] <= q_offsets[:, None])
        s = torch.where(
            mask[None, :, :],
            (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(logits.dtype),
            float("-inf"),
        )
        weight_slice = weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        s = torch.relu(s) * weight_slice[..., None]
        s = s.sum(dim=0)
        logits[i * next_n : (i + 1) * next_n, :context_len] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float("-inf"))

    return logits


def ref_fp8_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
):
    batch_size, next_n, heads, dim = q.size()
    num_block, block_size, _, dim = kv_cache.size()
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    context_lens = context_lens.tolist()
    for i in range(batch_size):
        context_len = context_lens[i]
        q_offsets = torch.arange(context_len - next_n, context_len, device="cuda")
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        for block_rk in range(cdiv(context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_cache[block_idx]
            k_offsets = torch.arange(
                block_rk * block_size, (block_rk + 1) * block_size, device="cuda"
            )
            mask = (k_offsets[None, :] < context_len) & (
                k_offsets[None, :] <= q_offsets[:, None]
            )
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
                    logits.dtype
                ),
                float("-inf"),
            )
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)
            logits[
                i * next_n : (i + 1) * next_n,
                block_rk * block_size : (block_rk + 1) * block_size,
            ] = torch.where(
                k_offsets[None, None, :] <= q_offsets[None, :, None], s, float("-inf")
            )
    return logits


ChunkK = 256
max_model_len = 111 * 1000
WavePerEU = 2
# for batch_size, next_n in [(64, 1), (64, 2), (128, 1)]:
for batch_size, next_n in [(1, 1), (1, 2), (2, 1), (2, 2), (4, 1), (4, 2), (8, 1), (8, 2)]:
    for heads, index_dim in [(64, 128)]:
        for avg_kv in (8192, 65536):
            # max_model_len = 2 * avg_kv
            num_blocks, blocksize = max_model_len * 3, 1

            q = torch.randn((batch_size, next_n, heads, index_dim), device='cuda', dtype=torch.bfloat16)
            kv_cache = torch.randn((num_blocks, blocksize, 1, index_dim), device='cuda', dtype=torch.bfloat16)
            weights = torch.randn((batch_size * next_n, heads), device='cuda', dtype=torch.float32)

            # context_lens = torch.randint(int(0.7 * avg_kv), int(1.3 * avg_kv), (batch_size, )).cuda().to(torch.int32)
            context_lens = torch.randint(int(1.0 * avg_kv), int(1.0 * avg_kv) + 1, (batch_size, )).cuda().to(torch.int32)
            max_block_len = (context_lens.max().item() + blocksize - 1) // blocksize * blocksize
            block_tables = torch.zeros((batch_size, max_block_len), device='cuda', dtype=torch.int32)

            prefix_sum_context_lens = torch.zeros((batch_size + 1,), device="cuda", dtype=torch.int32)
            prefix_sum_context_lens[1:] = torch.cumsum(context_lens, dim=0)

            counter = 0
            block_idx_pool = list(range(num_blocks))
            random.shuffle(block_idx_pool)
            for i in range(batch_size):
                ctx_len = context_lens[i].item()
                for j in range(cdiv(ctx_len, blocksize)):
                    block_tables[i][j]
                    block_tables[i][j] = block_idx_pool[counter % num_blocks]
                    counter += 1

            q_fp8 = q.to(aiter.dtypes.fp8)
            kv_cache_fp8 = kv_cache_cast_to_fp8(kv_cache, padding=True)

            kv_indices = torch.zeros(
                prefix_sum_context_lens[-1], device="cuda", dtype=torch.int32
            )
            for i in range(batch_size):
                ctx_len = int(context_lens[i].item())
                kv_indices[prefix_sum_context_lens[i] : prefix_sum_context_lens[i + 1]] = (
                    torch.randperm(max_model_len, device="cuda")[:ctx_len]
                )

            ref_logits = ref_fp8_paged_mqa_logits(
                q, kv_cache, weights, context_lens, block_tables, max_model_len
            )
            # ref_logits = ref_fp8_paged_mqa_logits_ragged(
            #     q,
            #     kv_cache.view([num_blocks, blocksize, index_dim]),
            #     weights,
            #     prefix_sum_context_lens,
            #     kv_indices,
            #     max_model_len,
            # )

            out_logits = torch.full(
                (batch_size * next_n, max_model_len),
                float("-inf"),
                device="cuda",
                dtype=torch.float32,
            )

            gpu = torch.cuda.current_device()
            device_properties = torch.cuda.get_device_properties(gpu)
            cu_num = device_properties.multi_processor_count

            schedule_metadata = aiter.get_paged_mqa_logits_metadata(context_lens, blocksize, cu_num)
            # print(schedule_metadata)
            _, elapsed_us = run_perftest(
                deepgemm_fp8_paged_mqa_logits,
                q_fp8,
                kv_cache_fp8,
                weights,
                out_logits,
                context_lens,
                block_tables,
                max_model_len,
                ChunkK=ChunkK,
                Preshuffle=False,
                KVBlockSize=1,
                WavePerEU=WavePerEU,
                TotalCuCount=cu_num, 
                mqa_schedule_metadata=schedule_metadata,
            )

            positions = torch.arange(max_model_len, device="cuda").unsqueeze(0).expand(batch_size * next_n, -1)
            row_indices = torch.arange(batch_size * next_n, device="cuda") // next_n
            next_n_offset = torch.arange(batch_size * next_n, device="cuda") % next_n
            mask = positions <= (context_lens[row_indices] - next_n + next_n_offset).unsqueeze(1)

            def calc_diff(x: torch.Tensor, y: torch.Tensor):
                x, y = x.double(), y.double()
                denominator = (x * x + y * y).sum()
                sim = 2 * (x * y).sum() / denominator
                return 1 - sim

            out_logits = out_logits.masked_fill(~mask, 0)
            ref_logits = ref_logits.masked_fill(~mask, 0)


            logits_diff = calc_diff(out_logits, ref_logits)
            print(logits_diff)

            # import pdb;pdb.set_trace()
            assert logits_diff < 1e-3, f"{logits_diff=}"

            sum_lens = context_lens.float().sum().item()
            total_float_operations = 2 * next_n * heads * index_dim * sum_lens
            flops = total_float_operations / (elapsed_us + 1e-6) * 1e-6

            # ctx_list = context_lens.tolist()
            # total_memcpyA_bytes = batch_size * next_n * SplitKV * heads * index_dim
            # total_memcpyB_bytes = sum([cdiv(ctx, ChunkK) * ChunkK * index_dim for ctx in ctx_list]) * next_n
            # bandwidth_gbps = (total_memcpyA_bytes + total_memcpyB_bytes) / elapsed_us * 1e-3

            input_bytes = count_bytes(q_fp8, weights, context_lens) + sum_lens * (index_dim + 4) + (sum_lens / blocksize) * 4
            output_bytes = sum_lens * next_n * 4
            bandwidth_gbps = (input_bytes + output_bytes) / (elapsed_us + 1e-6) * 1e-3

            print(
                "ragged_k",
                " time elapsed: ",
                elapsed_us,
                "us   bandwidth (GB/s): ",
                bandwidth_gbps,
                "   TFlops: ",
                flops,
            )
