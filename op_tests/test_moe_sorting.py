import torch
from typing import Tuple
import aiter
from aiter.test_common import checkAllclose, perftest, benchmark
from aiter.fused_moe import moe_sorting, fused_topk
from aiter import dtypes
import argparse

BLOCK_SIZE_M = 32


@perftest(num_iters=3, num_warmup=0)
def test_moe_sorting_naive(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    expert_mask=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    block_size = BLOCK_SIZE_M

    device = topk_ids.device
    M, topk = topk_ids.shape
    topk = topk_ids.shape[1]
    max_num_tokens_padded = topk_ids.numel() + num_experts * block_size - topk
    max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)
    init_val = topk << 24 | M
    sorted_ids = torch.full(
        (max_num_tokens_padded,), init_val, dtype=dtypes.i32, device=device
    )
    sorted_weights = torch.empty(
        (max_num_tokens_padded,), dtype=dtypes.fp32, device=device
    )
    sorted_expert_ids = torch.full(
        (max_num_m_blocks,), -1, dtype=dtypes.i32, device=device
    )
    num_tokens_post_pad = torch.empty((2), dtype=dtypes.i32, device=device)

    sorted_ids_begin = 0
    sorted_expert_ids_begin = 0
    skip_expert_num = 0
    for expertId in range(num_experts):
        if expert_mask != None and expert_mask[expertId] == 0:
            skip_expert_num += 1
            continue
        token_id, topk_id = torch.where(topk_ids == expertId)
        tokensNum = token_id.numel()
        sorted_expert_ids_num = (tokensNum + block_size - 1) // block_size
        tokensNumPad = sorted_expert_ids_num * block_size
        sorted_ids[sorted_ids_begin : sorted_ids_begin + tokensNum] = (
            topk_id << 24 | token_id
        )
        sorted_weights[sorted_ids_begin : sorted_ids_begin + tokensNum] = topk_weights[
            token_id, topk_id
        ]
        sorted_ids_begin = sorted_ids_begin + tokensNumPad
        sorted_expert_ids[
            sorted_expert_ids_begin : sorted_expert_ids_begin + sorted_expert_ids_num
        ] = (expertId - skip_expert_num)
        sorted_expert_ids_begin = sorted_expert_ids_begin + sorted_expert_ids_num

    num_tokens_post_pad[0] = sorted_ids_begin
    num_tokens_post_pad[1] = topk_ids.shape[0]

    return sorted_ids, sorted_weights, sorted_expert_ids, num_tokens_post_pad


@perftest()
def test_moe_sorting_ck(
    topk_ids, topk_weights, num_experts, model_dim, moebuf_dtype, expert_mask=None
):
    return moe_sorting(
        topk_ids,
        topk_weights,
        num_experts,
        model_dim,
        moebuf_dtype,
        expert_mask=expert_mask,
    )


@benchmark()
def test_moe_sorting(
    dtype, token, model_dim, inter_dim, E, topk, has_expert_mask=False
):
    dim = (token, model_dim, inter_dim)
    input = torch.randn((token, model_dim), dtype=dtype, device="cuda")
    score = torch.rand((token, E), device="cuda", dtype=dtype)

    topk_weights, topk_ids = fused_topk(input, score, topk, True)

    expert_mask = (
        torch.randint(0, 2, (E,), dtype=topk_ids.dtype, device="cuda")
        if has_expert_mask
        else None
    )

    (
        sorted_ids_a,
        sorted_weights_a,
        sorted_expert_ids_a,
        num_tokens_post_padded_a,
    ), avg_a = test_moe_sorting_naive(topk_ids, topk_weights, E, expert_mask)

    (
        sorted_ids_b,
        sorted_weights_b,
        sorted_expert_ids_b,
        num_tokens_post_padded_b,
        moe_buf,
    ), avg_b = test_moe_sorting_ck(
        topk_ids, topk_weights, E, model_dim, dtype, expert_mask
    )

    print(
        f"[perf] {token=}, {model_dim=}, {inter_dim=}, {E=}, {topk=}, dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
    )
    checkAllclose(
        num_tokens_post_padded_a,
        num_tokens_post_padded_b,
        atol=0,
        msg="num_tokens_post_padded",
    )
    mask = sorted_ids_a != (topk << 24 | token)
    num_tokens_post_pad = num_tokens_post_padded_a[0].item()
    checkAllclose(
        sorted_ids_a[:num_tokens_post_pad],
        sorted_ids_b[:num_tokens_post_pad],
        msg="sorted_ids",
    )
    checkAllclose(sorted_weights_a[mask], sorted_weights_b[mask], msg="sorted_weights")

    expert_mask = sorted_expert_ids_a != -1
    checkAllclose(
        sorted_expert_ids_a[expert_mask],
        sorted_expert_ids_b[expert_mask],
        msg="sorted_expert_ids",
    )
    return {"us": avg_b}


import pandas as pd

l_dtype = ["bf16"]
l_m = [1, 7, 31, 64, 128, 256, 163840]
l_expert = [32, 256]
l_topk = [5, 8]
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
    "-m",
    type=int,
    default=None,
    help="""M of mnk.
    e.g.: -m 64""",
)
parser.add_argument(
    "-e",
    "--expert",
    type=int,
    choices=l_expert,
    nargs="?",
    const=None,
    default=None,
    help="""Number of experts.
    e.g.: -e 32""",
)
parser.add_argument(
    "-t",
    "--topk",
    type=int,
    choices=l_topk,
    nargs="?",
    const=None,
    default=None,
    help="""Number of top experts.
    e.g.: -t 5""",
)

args = parser.parse_args()
if args.dtype is None:
    l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
else:
    l_dtype = [dtypes.d_dtypes[args.dtype]]
if args.m is not None:
    l_m = [args.m]
if args.expert is not None:
    l_expert = [args.expert]
if args.topk is not None:
    l_topk = [args.topk]

df = []
print("test test_moe_sorting, no expert mask")
for dtype in l_dtype:
    for m in l_m:
        for E in l_expert:
            for top in l_topk:
                ret = test_moe_sorting(dtype, m, 7168, 4096, E, top)
                df.append(ret)
df = pd.DataFrame(df)
aiter.logger.info(f"summary:\n{df}")


df = []
print("test test_moe_sorting, with expert mask")
for dtype in l_dtype:
    for m in l_m:
        for E in l_expert:
            for top in l_topk:
                ret = test_moe_sorting(
                    dtype, m, 4096, 4096, E, top, has_expert_mask=True
                )
                df.append(ret)
df = pd.DataFrame(df)
aiter.logger.info(f"summary:\n{df}")
