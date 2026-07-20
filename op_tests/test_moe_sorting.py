# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools

import pandas as pd
import torch

import aiter
import aiter.fused_moe as fm
from aiter import dtypes
from aiter.fused_moe import fused_topk, moe_sorting
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

BLOCK_SIZE_M = 32
SUPPORTED_GFX = ["gfx942", "gfx950", "gfx1250"]


def set_moe_sorting_backend(backend: str) -> None:
    """Force which moe_sorting backend `moe_sorting()` dispatches to."""
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
        pass
    else:
        raise ValueError(f"unknown backend: {backend}")


def run_torch_moe_sorting(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    block_size: int = BLOCK_SIZE_M,
    expert_mask=None,
    num_local_tokens=None,
):
    """CPU/torch reference for moe_sorting outputs (not timed, not in the table)."""
    device = topk_ids.device
    m, topk = topk_ids.shape
    max_num_tokens_padded = topk_ids.numel() + num_experts * block_size - topk
    max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)
    init_val = topk << 24 | m
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
    for expert_id in range(num_experts):
        if expert_mask is not None and expert_mask[expert_id] == 0:
            skip_expert_num += 1
            continue
        token_id, topk_id = torch.where(topk_ids == expert_id)
        tokens_num = token_id.numel()
        sorted_expert_ids_num = (tokens_num + block_size - 1) // block_size
        tokens_num_pad = sorted_expert_ids_num * block_size
        sorted_ids[sorted_ids_begin : sorted_ids_begin + tokens_num] = (
            topk_id << 24 | token_id
        )
        sorted_weights[sorted_ids_begin : sorted_ids_begin + tokens_num] = topk_weights[
            token_id, topk_id
        ]
        sorted_ids_begin = sorted_ids_begin + tokens_num_pad
        sorted_expert_ids[
            sorted_expert_ids_begin : sorted_expert_ids_begin + sorted_expert_ids_num
        ] = (expert_id - skip_expert_num)
        sorted_expert_ids_begin = sorted_expert_ids_begin + sorted_expert_ids_num

    num_tokens_post_pad[0] = sorted_ids_begin
    num_tokens_post_pad[1] = topk_ids.shape[0]

    return sorted_ids, sorted_weights, sorted_expert_ids, num_tokens_post_pad


def _moe_sorting_roofline(token, topk, E, model_dim, dtype):
    """Crude roofline estimates for a memory-bound sort (not exact kernel traffic)."""
    max_num_tokens_padded = token * topk + E * BLOCK_SIZE_M - topk
    max_num_m_blocks = (max_num_tokens_padded + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    elem_bytes = torch.empty((), dtype=dtype).element_size()
    nbytes = (
        token * topk * 4
        + token * topk * 4
        + max_num_tokens_padded * 4
        + max_num_tokens_padded * 4
        + max_num_m_blocks * 4
        + token * model_dim * elem_bytes
    )
    flops = 2 * max_num_tokens_padded
    return flops, nbytes


def _compare_moe_sorting_outputs(ref, out, topk, num_rows):
    (
        sorted_ids_a,
        sorted_weights_a,
        sorted_expert_ids_a,
        num_tokens_post_padded_a,
    ) = ref
    (
        sorted_ids_b,
        sorted_weights_b,
        sorted_expert_ids_b,
        num_tokens_post_padded_b,
        _moe_buf,
    ) = out

    errs = {}
    errs["num_tokens_post_padded"] = checkAllclose(
        num_tokens_post_padded_a,
        num_tokens_post_padded_b,
        atol=0,
        msg="num_tokens_post_padded",
    )
    weight_mask = sorted_ids_a != (topk << 24 | num_rows)
    num_tokens_post_pad = num_tokens_post_padded_a[0].item()
    errs["sorted_ids"] = checkAllclose(
        sorted_ids_a[:num_tokens_post_pad],
        sorted_ids_b[:num_tokens_post_pad],
        msg="sorted_ids",
    )
    errs["sorted_weights"] = checkAllclose(
        sorted_weights_a[weight_mask],
        sorted_weights_b[weight_mask],
        msg="sorted_weights",
    )
    expert_mask = sorted_expert_ids_a != -1
    errs["sorted_expert_ids"] = checkAllclose(
        sorted_expert_ids_a[expert_mask],
        sorted_expert_ids_b[expert_mask],
        msg="sorted_expert_ids",
    )
    return errs


def _build_moe_sorting_inputs(
    token,
    model_dim,
    E,
    topk,
    dtype,
    has_expert_mask,
    padding_extra,
):
    input_tensor = torch.randn((token, model_dim), dtype=dtype, device="cuda")
    score = torch.rand((token, E), device="cuda", dtype=dtype)
    topk_weights, topk_ids = fused_topk(input_tensor, score, topk, True)

    expert_mask = (
        torch.randint(0, 2, (E,), dtype=topk_ids.dtype, device="cuda")
        if has_expert_mask
        else None
    )
    if padding_extra:
        num_local_tokens = torch.tensor([token], dtype=topk_ids.dtype, device="cuda")
        topk_ids_pad = torch.empty(
            [token + padding_extra, topk], dtype=topk_ids.dtype, device="cuda"
        )
        topk_ids_pad[:token, :] = topk_ids
        topk_ids = topk_ids_pad
    else:
        num_local_tokens = None

    return topk_ids, topk_weights, expert_mask, num_local_tokens


@benchmark()
def test_moe_sorting(
    dtype,
    token,
    model_dim,
    inter_dim,
    E,
    topk,
    has_expert_mask=False,
    padding_extra=0,
    dispatch_policy=0,
):
    topk_ids, topk_weights, expert_mask, num_local_tokens = _build_moe_sorting_inputs(
        token,
        model_dim,
        E,
        topk,
        dtype,
        has_expert_mask,
        padding_extra,
    )

    ref = run_torch_moe_sorting(
        topk_ids,
        topk_weights,
        E,
        BLOCK_SIZE_M,
        expert_mask,
        num_local_tokens,
    )

    candidates = {
        "opus": lambda: moe_sorting(
            topk_ids,
            topk_weights,
            E,
            model_dim,
            dtype,
            BLOCK_SIZE_M,
            expert_mask,
            num_local_tokens,
            dispatch_policy,
        ),
        "ck": lambda: moe_sorting(
            topk_ids,
            topk_weights,
            E,
            model_dim,
            dtype,
            BLOCK_SIZE_M,
            expert_mask,
            num_local_tokens,
            dispatch_policy,
        ),
    }
    # FlyDSL kernel only supports dispatch_policy=0 today.
    if is_flydsl_available() and dispatch_policy == 0:
        candidates["flydsl"] = lambda: moe_sorting(
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

    flops, nbytes = _moe_sorting_roofline(token, topk, E, model_dim, dtype)
    ret = {"gfx": get_gfx()}
    failures = {}

    for name, fn in candidates.items():
        set_moe_sorting_backend(name)
        out, us = run_perftest(
            fn,
            num_warmup=1,
            num_iters=2,
        )
        errs = _compare_moe_sorting_outputs(ref, out, topk, topk_ids.shape[0])
        err = max(errs.values()) if errs else 0.0
        bad = {k: v for k, v in errs.items() if v}
        if bad:
            failures[name] = bad

        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err

    if failures:
        raise AssertionError(
            f"moe_sorting mismatch vs CPU reference at token={token}, E={E}, "
            f"topk={topk}, has_expert_mask={has_expert_mask}, "
            f"padding_extra={padding_extra}, dispatch_policy={dispatch_policy}: "
            f"{failures}"
        )
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("moe_sorting unsupported on %s; skipping", get_gfx())
        return

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
        help="Data type.\n    e.g.: -d bf16",
    )
    parser.add_argument(
        "-m",
        type=int,
        nargs="*",
        default=[1, 7, 31, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        help="Number of tokens.\n    e.g.: -m 64",
    )
    parser.add_argument(
        "-e",
        "--expert",
        type=int,
        nargs="*",
        default=[32, 256, 40, 385],
        help="Number of experts (paired with -t).\n    e.g.: -e 32 385",
    )
    parser.add_argument(
        "-md",
        "--model_dim",
        type=int,
        default=4096,
        help="Model dimension.\n    e.g.: -md 7168",
    )
    parser.add_argument(
        "-id",
        "--inter_dim",
        type=int,
        default=4096,
        help="Intermediate dimension (table column only).\n    e.g.: -id 4096",
    )
    parser.add_argument(
        "-t",
        "--topk",
        type=int,
        nargs="*",
        default=[5, 8, 6, 7],
        help="Top-k experts per token (paired with -e).\n    e.g.: -t 5 7",
    )
    parser.add_argument(
        "-p",
        "--padding",
        type=int,
        nargs="*",
        default=[0, 1000],
        help="Extra padding rows in topk_ids (0 = none).\n    e.g.: -p 0",
    )
    parser.add_argument(
        "-dp",
        "--dispatch_policy",
        type=int,
        nargs="*",
        default=[0, 1],
        help="Dispatch policy.\n    e.g.: -dp 0",
    )
    parser.add_argument(
        "-em",
        "--expert_mask",
        type=dtypes.str2bool,
        nargs="*",
        default=[True, False],
        help="Expert mask.\n    e.g.: -em f",
    )
    args = parser.parse_args()

    if len(args.expert) != len(args.topk):
        parser.error("-e/--expert and -t/--topk must have the same length")

    model_configs = list(zip(args.expert, args.topk))

    for dtype in args.dtype:
        df = []
        for padding_extra, expert_mask, dispatch_policy, m in itertools.product(
            args.padding,
            args.expert_mask,
            args.dispatch_policy,
            args.m,
        ):
            for E, topk in model_configs:
                df.append(
                    test_moe_sorting(
                        dtype,
                        m,
                        args.model_dim,
                        args.inter_dim,
                        E,
                        topk,
                        has_expert_mask=expert_mask,
                        padding_extra=padding_extra,
                        dispatch_policy=dispatch_policy,
                    )
                )
        df = pd.DataFrame(df)
        aiter.logger.info(
            "moe_sorting summary (markdown):\n%s", df.to_markdown(index=False)
        )


if __name__ == "__main__":
    main()
