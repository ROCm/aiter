# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import random
import argparse

import torch
import os

import triton

from aiter.test_common import run_perftest
from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits
from aiter.ops.shuffle import shuffle_weight


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


def ref_fp8_paged_mqa_logits_ragged(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    prefix_sum_context_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    max_model_len: int,
):
    batch_size, next_n, heads, dim = q.size()
    seq_kv, block_size, dim = kv_cache.size()  # 3d
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
            kv_cache[
                kv_indices[prefix_sum_context_lens[i] : prefix_sum_context_lens[i + 1]]
            ],
        )
        k_offsets = torch.arange(0, context_len, device="cuda")
        mask = (k_offsets[None, :] < context_len) & (
            k_offsets[None, :] <= q_offsets[:, None]
        )
        s = torch.where(
            mask[None, :, :],
            (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(logits.dtype),
            float("-inf"),
        )
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        s = torch.relu(s) * weight_slice[..., None]
        s = s.sum(dim=0)
        logits[i * next_n : (i + 1) * next_n, :context_len] = torch.where(
            k_offsets[None, :] <= q_offsets[:, None], s, float("-inf")
        )

    return logits


def create_paged_mqa_logits_configs(args: argparse.Namespace):
    x_names = ["batch_size", "next_n", "heads", "index_dim", "avg_kv_length"]
    line_names = ["non_ragged_k"]
    line_args = "kv_storage_kind"

    if args.perf:
        x_vals_list = [
            (1, 2, 64, 128, 16384),
            (1, 2, 64, 128, 32768),
            (1, 2, 64, 128, 65536),
            (2, 2, 64, 128, 16384),
            (2, 2, 64, 128, 32768),
            (2, 2, 64, 128, 65536),
            (4, 2, 64, 128, 16384),
            (4, 2, 64, 128, 32768),
            (4, 2, 64, 128, 65536),
            (1, 1, 64, 128, 65536),
            (2, 1, 64, 128, 65536),
            (4, 1, 64, 128, 65536),
            (8, 1, 64, 128, 65536),
        ]
    else:
        x_vals_list = [
            (args.batch, args.mtp + 1, args.heads, args.index_dim, args.kv_length)
        ]

    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=x_names,
            x_vals=x_vals_list,
            line_arg=line_args,
            line_vals=line_names,
            line_names=line_names,
            styles=[("red", "-"), ("green", "-")],
            ylabel="TFLOPS",
            plot_name="paged_mqa_logits",
            args={},
        )
    )

    return configs


def run_benchmark(args: argparse.Namespace):
    ChunkK = 256
    WavePerEU = 2

    @triton.testing.perf_report(create_paged_mqa_logits_configs(args))
    def test_deepgemm_fp8_paged_mqa_logits(
        batch_size, next_n, heads, index_dim, avg_kv_length, kv_storage_kind
    ):
        torch.manual_seed(0)
        random.seed(0)

        max_model_len = 2 * avg_kv_length
        num_blocks = max_model_len
        blocksize = args.blocksize if args.kv_preshuffle else 1

        assert blocksize == 1 or args.kv_preshuffle and blocksize % 16 == 0

        var_ratio = 0.0
        context_lens = (
            torch.randint(
                int((1 - var_ratio) * avg_kv_length),
                int(((1 + var_ratio)) * avg_kv_length) + 1,
                (batch_size,),
            )
            .cuda()
            .to(torch.int32)
        )
        prefix_sum_context_lens = torch.zeros(
            (batch_size + 1,), device="cuda", dtype=torch.int32
        )
        prefix_sum_context_lens[1:] = torch.cumsum(context_lens, dim=0)

        q = torch.randn(
            (batch_size, next_n, heads, index_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
        kv_cache = torch.randn(
            (num_blocks, blocksize, 1, index_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
        weights = torch.randn(
            (batch_size * next_n, heads),
            device="cuda",
            dtype=torch.float32,
        )

        qk_datatype = torch.float8_e4m3fnuz
        max_block_len = (
            (context_lens.max().item() + blocksize - 1) // blocksize * blocksize
        )
        block_tables = torch.zeros(
            (batch_size, max_block_len), device="cuda", dtype=torch.int32
        )

        counter = 0
        block_idx_pool = list(range(num_blocks))
        random.shuffle(block_idx_pool)
        for i in range(batch_size):
            ctx_len = context_lens[i].item()
            for j in range(cdiv(ctx_len, blocksize)):
                block_tables[i][j] = block_idx_pool[counter % num_blocks]
                counter += 1

        q_fp8 = q.to(qk_datatype)
        kv_cache_fp8 = kv_cache_cast_to_fp8(kv_cache, padding=args.padding)

        kv_indices = torch.zeros(
            prefix_sum_context_lens[-1], device="cuda", dtype=torch.int32
        )
        for i in range(batch_size):
            ctx_len = int(context_lens[i].item())
            kv_indices[prefix_sum_context_lens[i] : prefix_sum_context_lens[i + 1]] = (
                torch.randperm(max_model_len, device="cuda")[:ctx_len]
            )

        if kv_storage_kind == "non_ragged_k":
            ref_logits = ref_fp8_paged_mqa_logits(
                q, kv_cache, weights, context_lens, block_tables, max_model_len
            )
        else:
            ref_logits = ref_fp8_paged_mqa_logits_ragged(
                q,
                kv_cache.view([num_blocks, blocksize, index_dim]),
                weights,
                prefix_sum_context_lens,
                kv_indices,
                max_model_len,
            )

        out_logits = torch.full(
            (batch_size * next_n, max_model_len),
            float("-inf"),
            device="cuda",
            dtype=torch.float32,
        )

        if kv_storage_kind == "non_ragged_k":
            Preshuffle = blocksize % 16 == 0

            if Preshuffle:
                kv_num_block, kv_block_Size, _, kv_index_dim = kv_cache_fp8.size()

                split_kv_cache = kv_cache_fp8.view(-1, blocksize * kv_index_dim)
                split_kv_cache_data = shuffle_weight(
                    split_kv_cache[..., : kv_block_Size * index_dim]
                    .contiguous()
                    .view([kv_num_block, kv_block_Size, index_dim])
                )
                split_kv_cache[..., : kv_block_Size * index_dim] = (
                    split_kv_cache_data.view(kv_num_block, kv_block_Size * index_dim)
                )

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
                Preshuffle=Preshuffle,
                KVBlockSize=blocksize,
                WavePerEU=WavePerEU,
            )
            cache_key = deepgemm_fp8_paged_mqa_logits(
                q_fp8,
                kv_cache_fp8,
                weights,
                out_logits,
                context_lens,
                block_tables,
                max_model_len,
                ChunkK=ChunkK,
                Preshuffle=Preshuffle,
                KVBlockSize=blocksize,
                WavePerEU=WavePerEU,
            )

            print(">>> ", cache_key)

        positions = (
            torch.arange(max_model_len, device="cuda")
            .unsqueeze(0)
            .expand(batch_size * next_n, -1)
        )
        row_indices = torch.arange(batch_size * next_n, device="cuda") // next_n
        next_n_offset = torch.arange(batch_size * next_n, device="cuda") % next_n
        mask = positions <= (
            context_lens[row_indices] - next_n + next_n_offset
        ).unsqueeze(1)

        def calc_diff(x: torch.Tensor, y: torch.Tensor):
            x, y = x.double(), y.double()
            denominator = (x * x + y * y).sum()
            sim = 2 * (x * y).sum() / denominator
            return 1 - sim

        out_logits = out_logits.masked_fill(~mask, 0)
        ref_logits = ref_logits.masked_fill(~mask, 0)

        logits_diff = calc_diff(out_logits, ref_logits)

        print(">>>! logits_diff = ", logits_diff)
        # assert logits_diff < 1e-3

        total_float_operations = (
            2 * next_n * heads * index_dim * context_lens.float().sum().item()
        )
        flops = total_float_operations / elapsed_us * 1e-6

        print(
            kv_storage_kind,
            " time elapsed: ",
            elapsed_us,
        )

        if args.aot:
            triton_cache_dir = str(triton.knobs.cache.dir)
            aot_kernel_dir = f"./paged_mqa_logits/aot"

            padded_str = "T" if args.padding else "F"
            os.makedirs(aot_kernel_dir, exist_ok=True)
            aot_name = f"paged_mqa_logits{"_preshuffle" if args.kv_preshuffle else ""}_{heads}x{ChunkK}x{index_dim}_B{blocksize}P{padded_str}W{WavePerEU}"

            src = os.path.join(triton_cache_dir, cache_key)
            dst = os.path.join(aot_kernel_dir, aot_name)
            if os.path.exists(dst):
                os.system(f"rm -rf {dst}")
            os.system(f"mv {src} {dst}")
            print(f"Moved cache from {src} to {dst}")

            os.system(f"zip -r paged_mqa_logits_aot_kernel paged_mqa_logits")

        return flops

    test_deepgemm_fp8_paged_mqa_logits.run(print_data=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-B", "--batch", type=int, default=128, help="Batch size.")
    parser.add_argument(
        "-hq",
        "--heads",
        type=int,
        default=64,
        help="Number of query heads (equal to number of key/value heads)",
    )
    parser.add_argument(
        "--index_dim",
        type=int,
        default=128,
        help="Head dimension (dimension of query/key/value vectors)",
    )
    parser.add_argument(
        "-kv_length",
        type=int,
        default=4096,
        help="Sequence length (since this is decode, this is the length of the key/value sequence)",
    )
    parser.add_argument(
        "-mtp",
        type=int,
        default=0,
        help="Q sequence length (mtp + 1 == qo_len) in MTP mode",
    )
    parser.add_argument(
        "-p",
        "--padding",
        action="store_true",
        help="Padding the contiguous dimension of KVCache to multiple of 16 Bytes",
    )
    parser.add_argument(
        "-aot",
        action="store_true",
        help="Save compiled triton kernel for later AOT use",
    )
    parser.add_argument(
        "--perf",
        action="store_true",
    )
    parser.add_argument(
        "--kv_preshuffle",
        action="store_true",
        help="Enable KV cache preshuffle, also change blocksize to 16",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=16,
        help="KVCache block size, only used when kv_preshuffle is enabled, must be multiple of 16",
    )

    args = parser.parse_args()
    run_benchmark(args)
