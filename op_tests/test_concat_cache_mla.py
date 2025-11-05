import torch
import aiter
from aiter.test_common import checkAllclose, perftest, benchmark
from aiter import dtypes
from typing import Tuple
import argparse
import itertools
import pandas as pd
import random
import time


@perftest()
def run_aiter(
    kv_c,
    k_pe,
    kv_cache,
    slot_mapping,
    kv_cache_dtype: str,
    scale,
):
    aiter.concat_and_cache_mla(
        kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale
    )
    return kv_cache


@perftest(3)
def run_torch(
    kv_c,
    k_pe,
    kv_cache,
    slot_mapping,
    kv_cache_dtype: str,
    scale,
    dtype,
):

    block_size = kv_cache.shape[1]
    num_tokens = kv_c.shape[0]
    kv_lora_rank = kv_c.shape[-1]

    for i in range(num_tokens):
        slot = slot_mapping[i].item()
        block_idx = slot // block_size
        block_offset = slot % block_size
        kv_cache[block_idx, block_offset, :kv_lora_rank] = kv_c[i]
        kv_cache[block_idx, block_offset, kv_lora_rank:] = k_pe[i]

    if kv_cache_dtype == "fp8":
        ref_kv_cache = (kv_cache.to(torch.float32) / scale.item()).to(dtype)
    else:
        ref_kv_cache = kv_cache
    return ref_kv_cache


## compare with vllm impl
# from vllm import _custom_ops as ops
# @perftest()
# def run_vllm(
#    kv_c,
#    k_pe,
#    kv_cache,
#    slot_mapping,
#    kv_cache_dtype: str,
#    scale,
# ):
#    ops.concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale)
#    return kv_cache


@benchmark()
def test_concat_and_cache_mla(
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    num_tokens: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    device: str,
    kv_cache_dtype: str,
) -> None:
    ret = {}
    torch.set_default_device(device)

    total_slots = num_blocks * block_size
    slot_mapping_lst = random.sample(range(total_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long, device=device)

    kv_c = torch.randn(num_tokens, kv_lora_rank, dtype=dtype, device=device)
    k_pe = torch.randn(num_tokens, qk_rope_head_dim, dtype=dtype, device=device)
    entry_size = kv_lora_rank + qk_rope_head_dim

    scale = torch.tensor(0.1, dtype=torch.float32, device=device)
    cache_dtype = dtypes.fp8 if kv_cache_dtype == "fp8" else dtype
    kv_cache = torch.zeros(
        num_blocks, block_size, entry_size, dtype=cache_dtype, device=device
    )

    kv_cache, avg_us = run_aiter(
        kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale
    )
    ref_temp = torch.zeros(*kv_cache.shape, dtype=dtype, device=device)
    ref_kv_cache, ref_us = run_torch(
        kv_c, k_pe, ref_temp, slot_mapping, kv_cache_dtype, scale, kv_cache.dtype
    )

    # vllm_temp = torch.zeros(*kv_cache.shape, dtype=cache_dtype, device=device)
    # vllm_kv_cache, vllm_us = run_vllm(
    #    kv_c, k_pe, vllm_temp, slot_mapping, kv_cache_dtype, scale
    # )

    if kv_cache_dtype == "fp8":
        result_temp = kv_cache.to(torch.float32) * scale
        expected_temp = ref_kv_cache.to(torch.float32) * scale
        # result_temp = torch.empty_like(kv_cache, dtype=torch.float32)
        # ops.convert_fp8(result_temp, kv_cache, scale.item(), kv_dtype=kv_cache_dtype)
        # expected_vllm = torch.empty_like(vllm_kv_cache, dtype=torch.float32)
        # ops.convert_fp8(
        #    expected_vllm, vllm_kv_cache, scale.item(), kv_dtype=kv_cache_dtype
        # )
        checkAllclose(result_temp, expected_temp, atol=0.01, rtol=0.01)
    else:
        checkAllclose(kv_cache, ref_kv_cache)
    ret["aiter_us"] = avg_us
    ret["torch_us"] = ref_us
    # ret["vllm_us"] = vllm_us
    ret["aiter_bw(TB/s)"] = (
        num_tokens
        * (kv_lora_rank + qk_rope_head_dim)
        * 2
        * (torch.finfo(dtype).bits // 8)
        / (avg_us * 1e6)
    )
    return ret


df = []
kv_lora_rank = 128
qk_rope_head_dim = 64
l_num_tokens = [128, 256, 512, 1024, 2048, 4096]  # , 8192, 16384
block_size = 64
dtype = torch.float16
device = "cuda"
l_kv_cache_dtypes = ["auto", "fp8"]

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-k",
    "--kv_lora_rank",
    type=int,
    default=512,
    help="""kv lora rank.
    e.g.: -k 512""",
)
parser.add_argument(
    "-qr",
    "--qk_rope_head_dim",
    type=int,
    default=64,
    help="""qk rope head dim.
    e.g.: -qr 64""",
)

parser.add_argument(
    "-blk",
    "--block_size",
    type=int,
    default=64,
    help="""Block size.
    e.g.: -blk 1""",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    default="bf16",
    help="""Data type of input.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-kvd",
    "--kv_dtype",
    type=str,
    choices=["auto", "fp8"],
    nargs="*",
    default=["auto", "fp8"],
    help="""Data type of KV cache.
    e.g.: -kvd auto""",
)
parser.add_argument(
    "-t",
    "--token",
    type=int,
    nargs="*",
    default=l_num_tokens,
    help="""token nums.
    e.g.: -t 128""",
)


args = parser.parse_args()
if args.dtype is not None:
    dtype = dtypes.d_dtypes[args.dtype]
if args.token is not None:
    l_num_tokens = args.token
if args.kv_dtype is not None:
    l_kv_cache_dtypes = args.kv_dtype
if args.block_size is not None:
    block_size = args.block_size
if args.qk_rope_head_dim is not None:
    qk_rope_head_dim = args.qk_rope_head_dim
if args.kv_lora_rank is not None:
    kv_lora_rank = args.kv_lora_rank

for num_token in l_num_tokens:
    num_blocks = num_token // block_size
    for kv_cache_dtype in l_kv_cache_dtypes:
        ret = test_concat_and_cache_mla(
            kv_lora_rank,
            qk_rope_head_dim,
            num_token,
            block_size,
            num_blocks,
            dtype,
            device,
            kv_cache_dtype,
        )

        df.append(ret)
df = pd.DataFrame(df)
aiter.logger.info(f"summary:\n{df}")
