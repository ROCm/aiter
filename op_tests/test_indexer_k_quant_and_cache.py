# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter.test_common import checkAllclose, run_perftest, benchmark
from aiter import dtypes
from aiter import pertoken_quant, dtypes, indexer_k_quant_and_cache
import argparse
import pandas as pd

MAX_TOKEN_SUPPORTED = 16384
torch.set_default_device("cuda")


def run_torch(k, kv_cache, slot_mapping, quant_block_size, scale_fmt):
    num_token, head_dim = k.shape
    block_size = kv_cache.shape[1]
    per_token_amax, _ = torch.max(
        input=torch.abs(k.view(-1, quant_block_size)), dim=-1, keepdim=True
    )
    scale = per_token_amax / torch.finfo(dtypes.fp8).max
    if scale_fmt == "ue8m0":
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    k_fp8, scale = pertoken_quant(
        k.view(-1, quant_block_size), quant_dtype=dtypes.fp8, scale=scale
    )
    k_fp8 = k_fp8.view(num_token, head_dim)
    for i in range(num_token):
        slot = slot_mapping[i].item()
        blockId = slot // block_size
        block_offset = slot % block_size
        kv_cache[blockId, block_offset, :head_dim] = k_fp8[i]
        kv_cache[blockId, block_offset, head_dim:] = scale[i].view(dtypes.fp8)


@benchmark()
def test_indexer_k_quant_and_cache(
    num_token, block_size, quant_block_size, head_dim=128
):
    assert (
        num_token <= MAX_TOKEN_SUPPORTED
    ), f"test only support max_token={MAX_TOKEN_SUPPORTED}"
    block_num = (num_token + block_size - 1) // block_size
    k = torch.randn((num_token, head_dim), dtype=dtypes.bf16)
    slot_mapping = torch.arange(0, num_token, 1, dtype=torch.int64)
    scale_fmt = "ue8m0"
    kv_cache = torch.empty((block_num, block_size, head_dim + 4), dtype=dtypes.fp8)
    run_torch(k, kv_cache, slot_mapping, quant_block_size, scale_fmt)
    kv_cache2 = torch.empty((block_num, block_size, head_dim + 4), dtype=dtypes.fp8)
    _, us = run_perftest(
        indexer_k_quant_and_cache,
        k,
        kv_cache2,
        slot_mapping,
        quant_block_size,
        scale_fmt,
    )
    err = checkAllclose(
        kv_cache.view(-1, head_dim + 4)[:num_token].to(torch.float),
        kv_cache2.view(-1, head_dim + 4)[:num_token].to(torch.float),
    )
    # scale = kv_cache[:, :, head_dim:].view(torch.float)
    # scale2 = kv_cache2[:, :, head_dim:].view(torch.float)
    ret = {"aiter us": us, "aiter err": err}
    try:
        from vllm import _custom_ops as ops

        kv_cache3 = torch.empty((block_num, block_size, head_dim + 4), dtype=dtypes.fp8)
        _, us2 = run_perftest(
            ops.indexer_k_quant_and_cache,
            k,
            kv_cache3,
            slot_mapping,
            quant_block_size,
            scale_fmt,
        )
        err2 = checkAllclose(
            kv_cache.view(-1, head_dim + 4)[:num_token].to(torch.float),
            kv_cache3.view(-1, head_dim + 4)[:num_token].to(torch.float),
        )
        ret.update({"vllm us": us2, "vllm err": err2})
    except Exception:
        # Ignore all exceptions here because vllm._custom_ops is optional and may not be available.
        pass
    return ret


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Test indexer_k_quant_and_cache.",
)
parser.add_argument(
    "-m",
    type=int,
    nargs="*",
    default=[1, 64, 128, 257, 1028, 16384],
    help="""token num""",
)
parser.add_argument(
    "-b",
    "--block_size",
    type=int,
    nargs="*",
    default=[1],
    help="""block_size, default: 1""",
)

args = parser.parse_args()
df = []
for m in args.m:
    for block_size in args.block_size:
        ret = test_indexer_k_quant_and_cache(m, block_size, 128, 128)
        df.append(ret)
df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("indexer_k_quant_and_cache summary (markdown):\n%s", df_md)
