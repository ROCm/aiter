# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import torch
from typing import Tuple
import aiter
import aiter.fused_moe as fm
from aiter.test_common import checkAllclose, run_perftest, benchmark
from aiter.fused_moe import moe_sorting, fused_topk
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter import dtypes
import argparse
import pandas as pd

BLOCK_SIZE_M = 32


def set_moe_sorting_backend(backend: str) -> None:
    """Force which moe_sorting backend `moe_sorting()` dispatches to.

    The backend is selected by module-level flags in aiter.fused_moe that are
    read at call time, so flipping them here is sufficient for this
    single-process test.
    """
    if backend == "flydsl":
        if not is_flydsl_available():
            raise RuntimeError(
                "backend=flydsl requested but FlyDSL is not available in this build"
            )
        fm._USE_CK_MOE_SORTING = False
        fm._USE_FLYDSL_MOE_SORTING = True
    elif backend == "opus":
        fm._USE_CK_MOE_SORTING = False
        fm._USE_FLYDSL_MOE_SORTING = False
    elif backend == "ck":
        fm._USE_CK_MOE_SORTING = True
        fm._USE_FLYDSL_MOE_SORTING = False
    elif backend == "auto":
        pass  # leave module defaults as imported
    else:
        raise ValueError(f"unknown backend: {backend}")


def moe_sorting_native(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    block_size=BLOCK_SIZE_M,
    expert_mask=None,
    num_local_tokens=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

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

    if num_local_tokens is not None:
        topk_ids = topk_ids[: num_local_tokens.item()]

    sorted_ids_begin = 0
    sorted_expert_ids_begin = 0
    skip_expert_num = 0
    for expertId in range(num_experts):
        if expert_mask is not None and expert_mask[expertId] == 0:
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


@benchmark()
def test_moe_sorting(
    dtype,
    token,
    model_dim,
    inter_dim,
    E,
    topk,
    has_expert_mask=False,
    padding_token=False,
    dispatch_policy=0,
    backend="auto",
):
    set_moe_sorting_backend(backend)
    input = torch.randn((token, model_dim), dtype=dtype, device="cuda")
    score = torch.rand((token, E), device="cuda", dtype=dtype)

    topk_weights, topk_ids = fused_topk(input, score, topk, True)

    expert_mask = (
        torch.randint(0, 2, (E,), dtype=topk_ids.dtype, device="cuda")
        if has_expert_mask
        else None
    )
    if padding_token:
        num_local_tokens = torch.tensor([token], dtype=topk_ids.dtype, device="cuda")
        topk_ids_pad = torch.empty(
            [token + 1000, topk], dtype=topk_ids.dtype, device="cuda"
        )
        topk_ids_pad[:token, :] = topk_ids
        topk_ids = topk_ids_pad
    else:
        num_local_tokens = None

    (
        (
            sorted_ids_a,
            sorted_weights_a,
            sorted_expert_ids_a,
            num_tokens_post_padded_a,
        ),
        avg_a,
    ) = run_perftest(
        moe_sorting_native,
        topk_ids,
        topk_weights,
        E,
        BLOCK_SIZE_M,
        expert_mask,
        num_local_tokens,
        num_warmup=1,
        num_iters=2,
    )

    (
        (
            sorted_ids_b,
            sorted_weights_b,
            sorted_expert_ids_b,
            num_tokens_post_padded_b,
            moe_buf,
        ),
        avg_b,
    ) = run_perftest(
        moe_sorting,
        topk_ids,
        topk_weights,
        E,
        model_dim,
        dtype,
        BLOCK_SIZE_M,
        expert_mask,
        num_local_tokens,
        dispatch_policy,
    )

    print(
        f"[perf] backend={backend}, {token=}, {model_dim=}, {inter_dim=}, {E=}, {topk=}, dtype: {dtype}, torch avg: {avg_a:<8.2f} us, kernel avg: {avg_b:<8.2f} us, uplift: {avg_a / avg_b - 1:<5.1%}"
    )
    # checkAllclose only logs; it returns the mismatch fraction (0 == all close)
    # and never raises. Collect the fractions and assert so a regression in any
    # backend actually fails the process / CI rather than passing silently.
    errs = {}
    errs["num_tokens_post_padded"] = checkAllclose(
        num_tokens_post_padded_a,
        num_tokens_post_padded_b,
        atol=0,
        msg="num_tokens_post_padded",
    )
    mask = sorted_ids_a != (topk << 24 | topk_ids.shape[0])
    num_tokens_post_pad = num_tokens_post_padded_a[0].item()
    errs["sorted_ids"] = checkAllclose(
        sorted_ids_a[:num_tokens_post_pad],
        sorted_ids_b[:num_tokens_post_pad],
        msg="sorted_ids",
    )
    errs["sorted_weights"] = checkAllclose(
        sorted_weights_a[mask],
        sorted_weights_b[mask],
        msg="sorted_weights",
    )

    expert_mask = sorted_expert_ids_a != -1
    errs["sorted_expert_ids"] = checkAllclose(
        sorted_expert_ids_a[expert_mask],
        sorted_expert_ids_b[expert_mask],
        msg="sorted_expert_ids",
    )
    bad = {k: v for k, v in errs.items() if v}
    assert not bad, (
        f"moe_sorting backend={backend} mismatch vs CPU reference at "
        f"token={token}, E={E}, topk={topk}, has_expert_mask={has_expert_mask}, "
        f"padding_token={padding_token}, dispatch_policy={dispatch_policy}: "
        f"mismatch fractions {bad}"
    )
    return {"us": avg_b}


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    choices=[dtypes.d_dtypes["bf16"]],
    nargs="*",
    default=[dtypes.d_dtypes["bf16"]],
    metavar="{bf16}",
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-m",
    type=int,
    nargs="*",
    default=[1, 7, 31, 64, 128, 256, 163840],
    help="""Number of tokens.
    e.g.: -m 64""",
)
parser.add_argument(
    "-e",
    "--expert",
    type=int,
    nargs="*",
    default=[32, 256],
    help="""Number of experts.
    e.g.: -e 32""",
)
parser.add_argument(
    "-md",
    "--model_dim",
    type=int,
    default=4096,
    help="""Model dimension.
    e.g.: -md 4096""",
)
parser.add_argument(
    "-id",
    "--inter_dim",
    type=int,
    default=4096,
    help="""Intermediate dimension.
    e.g.: -id 4096""",
)
parser.add_argument(
    "-t",
    "--topk",
    type=int,
    nargs="*",
    default=[5, 8],
    help="""Number of top experts.
    e.g.: -t 5""",
)
parser.add_argument(
    "-p",
    "--padding",
    type=int,
    nargs="*",
    default=[0, 1000],
    help="""Padding token.
    e.g.: -p 0""",
)
parser.add_argument(
    "-dp",
    "--dispatch_policy",
    type=int,
    nargs="*",
    default=[0, 1],
    help="""Dispatch policy.
    e.g.: -dp 0""",
)
parser.add_argument(
    "-em",
    "--expert_mask",
    type=dtypes.str2bool,
    nargs="*",
    default=[True, False],
    help="""Expert mask default is [True, False].
    e.g.: -em f    # false
          -em t    # true""",
)
parser.add_argument(
    "-b",
    "--backend",
    type=str,
    nargs="*",
    choices=["auto", "flydsl", "opus", "ck"],
    default=None,
    help="""moe_sorting backend(s) to validate against the CPU reference.
    Each is checked independently so coverage is explicit per backend and does
    not depend on which backend the build happens to default to.
    Default: validate FlyDSL and OPUS when FlyDSL is available in the build
    (FlyDSL skipped otherwise; OPUS always validated).
    e.g.: -b flydsl opus    # validate both
          -b auto           # whatever the build defaults to""",
)

args = parser.parse_args()

if args.backend is None:
    if is_flydsl_available():
        backends = ["flydsl", "opus"]
    else:
        backends = ["auto"]
        print("FlyDSL not available in this build; validating default backend (auto)")
else:
    backends = args.backend

for backend in backends:
    if backend == "flydsl" and not is_flydsl_available():
        print("skipping backend=flydsl: not available in this build")
        continue
    for padding_token in args.padding:
        for expert_mask in args.expert_mask:
            for dispatch_policy in args.dispatch_policy:
                # dispatch_policy is not plumbed into the FlyDSL kernel; its
                # guard falls back to opus/ck for dp != 0, so skip the redundant
                # (and mislabeled) FlyDSL run there.
                if backend == "flydsl" and dispatch_policy != 0:
                    continue
                df = []
                print(
                    f"test test_moe_sorting, backend:{backend}, "
                    f"expert mask:{bool(expert_mask)}, "
                    f"padding_token:{padding_token}, "
                    f"dispatch_policy={dispatch_policy}"
                )
                for dtype in args.dtype:
                    for m in args.m:
                        for E, top in zip(args.expert, args.topk):
                            ret = test_moe_sorting(
                                dtype,
                                m,
                                args.model_dim,
                                args.inter_dim,
                                E,
                                top,
                                has_expert_mask=expert_mask,
                                padding_token=padding_token,
                                dispatch_policy=dispatch_policy,
                                backend=backend,
                            )
                            df.append(ret)
                df = pd.DataFrame(df)
                df_md = df.to_markdown(index=False)
                aiter.logger.info(
                    "moe_sorting summary (backend=%s, markdown):\n%s", backend, df_md
                )
