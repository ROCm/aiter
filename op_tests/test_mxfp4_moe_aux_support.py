# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Direct generated-sort correctness and perf for MXMOE retune shapes."""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.moe_mxfp4_aux import mxfp4_moe_sort, prepare_mxfp4_moe_aux
from aiter.test_common import benchmark, checkAllclose, run_perftest

# (expert, model_dim, inter_dim, topk), from the A8W4 implementation spec.
MODEL_SHAPES = [
    (24, 7168, 3072, 6),
    (256, 4096, 2048, 6),
    (896, 3584, 3072, 16),
]
SUPPORTED_GFX = ["gfx950"]


def run_torch(topk_ids, topk_weights, expert, block_m):
    token, topk = topk_ids.shape
    device = topk_ids.device
    counts = torch.bincount(topk_ids.flatten().long(), minlength=expert)
    blocks = (counts + block_m - 1) // block_m
    token_ids = torch.arange(token, device=device).repeat_interleave(topk)
    slots = torch.arange(topk, device=device).repeat(token)
    return {
        "num_valid_ids": torch.tensor(
            [blocks.sum().item() * block_m, token], device=device
        ),
        "sorted_expert_ids": torch.repeat_interleave(
            torch.arange(expert, device=device), blocks
        ),
        "route_ids": token_ids | (slots << 24),
        "token_ids": token_ids,
        "weights": topk_weights.flatten(),
    }


@benchmark()
def test_mxfp4_moe_aux(dtype, token, expert, model_dim, inter_dim, topk):
    block_m = 16
    device = "cuda"
    token_ids = torch.arange(token, device=device, dtype=dtypes.i32).unsqueeze(1)
    slots = torch.arange(topk, device=device, dtype=dtypes.i32).unsqueeze(0)
    # Distinct experts within each token, including high expert IDs. The token
    # offset also makes repeated experts span more than one BM16 block.
    topk_ids = ((token_ids * 7 + slots * (expert // topk)) % expert).contiguous()
    topk_weights = (
        torch.arange(1, token * topk + 1, device=device, dtype=dtypes.fp32)
        .view(token, topk)
        .div(token * topk)
    )
    ref = run_torch(topk_ids, topk_weights, expert, block_m)
    num_sorted = int(ref["num_valid_ids"][0].item())
    num_blocks = num_sorted // block_m

    # Match the adaptive sort's caller-provided buffers, including empty zero
    # workspace and sort3stage scratch for the BM16 prologue.
    active = min(expert, token * topk)
    max_sorted = (
        (token * topk + active * (block_m - 1) + block_m - 1) // block_m
    ) * block_m
    sorted_ids = torch.empty(max_sorted, dtype=dtypes.i32, device=device)
    sorted_experts = torch.empty(max_sorted // block_m, dtype=dtypes.i32, device=device)
    num_valid_ids = torch.empty(2, dtype=dtypes.i32, device=device)
    reverse_sorted = torch.empty(token * topk, dtype=dtypes.i32, device=device)
    sorted_weights = torch.empty(max_sorted, dtype=dtypes.fp32, device=device)
    m_indices = torch.empty(max_sorted, dtype=dtypes.i32, device=device)
    zero_out = torch.empty((token, model_dim), dtype=dtype, device=device)
    empty_bf16 = torch.empty(0, dtype=dtype, device=device)
    empty_i32 = torch.empty(0, dtype=dtypes.i32, device=device)

    def launch(output):
        mxfp4_moe_sort(
            topk_ids=topk_ids,
            topk_weight=topk_weights,
            sorted_token_ids=sorted_ids,
            sorted_expert_ids=sorted_experts,
            cumsum_tensor=num_valid_ids,
            reverse_sorted=reverse_sorted,
            sorted_weights=sorted_weights,
            m_indices=m_indices,
            bf16_zero_out=output,
            bf16_zero_workspace=empty_bf16,
            sort3stage_ws=empty_i32,
            M_logical=token,
            NE=expert,
            TOPK=topk,
            D_HIDDEN=model_dim,
            D_INTER=inter_dim,
            MB=block_m,
            prologue=0,
        )

    def check_outputs(name, zero_init):
        def compare(expected, actual, label):
            if not expected.is_floating_point():
                # FP32 cannot represent the low token bits of every encoded
                # top-k ID. Require exact integer equality before the metric.
                assert torch.equal(expected, actual), f"{name}: {label} mismatch"
            err = checkAllclose(
                expected.to(dtypes.fp32),
                actual.to(dtypes.fp32),
                rtol=0,
                atol=0,
                tol_err_ratio=0,
                msg=f"{name}: {label}",
            )
            assert err == 0, f"{name}: {label} mismatch ({err})"
            return err

        compare(ref["num_valid_ids"], num_valid_ids, "valid rows and tokens")
        compare(ref["sorted_expert_ids"], sorted_experts[:num_blocks], "expert blocks")
        positions = reverse_sorted.long()
        assert ((positions >= 0) & (positions < num_sorted)).all()
        assert positions.unique().numel() == token * topk
        compare(ref["route_ids"], sorted_ids[positions], "reverse route IDs")
        compare(
            topk_ids.flatten(), sorted_experts[positions // block_m], "route experts"
        )
        compare(ref["weights"], sorted_weights[positions], "route weights")
        compare(ref["token_ids"], m_indices[positions], "GEMM1 token indices")
        padding = torch.ones(num_sorted, dtype=torch.bool, device=device)
        padding[positions] = False
        if padding.any():
            compare(
                torch.full_like(sorted_ids[:num_sorted][padding], token),
                sorted_ids[:num_sorted][padding],
                "padding IDs",
            )
            compare(
                torch.full_like(m_indices[:num_sorted][padding], token),
                m_indices[:num_sorted][padding],
                "padding token indices",
            )
            compare(
                torch.zeros_like(sorted_weights[:num_sorted][padding]),
                sorted_weights[:num_sorted][padding],
                "padding weights",
            )
        if zero_init:
            compare(torch.zeros_like(zero_out), zero_out, "atomic output zero init")
        return 0

    candidates = {
        "sort_only": lambda: launch(empty_bf16),
        "sort_zero_init": lambda: launch(zero_out),
    }
    # As in the common sorting test, these are crude memory-bound roofline
    # estimates. The reference and buffer poisoning stay outside timing.
    flops = 2 * token * topk
    sort_bytes = token * topk * 12 + num_sorted * 12 + num_blocks * 4 + 8
    ret = {"gfx": get_gfx(), "block_m": block_m}
    for name, fn in candidates.items():
        zero_init = name == "sort_zero_init"
        for poison in (float("nan"), 13.0):
            for output in (
                sorted_ids,
                sorted_experts,
                num_valid_ids,
                reverse_sorted,
                m_indices,
            ):
                output.fill_(-99)
            sorted_weights.fill_(float("nan"))
            zero_out.fill_(poison)
            fn()
            check_outputs(name, zero_init)
        _, us = run_perftest(fn)
        err = check_outputs(name, zero_init)
        nbytes = sort_bytes + (
            zero_out.numel() * zero_out.element_size() if zero_init else 0
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "generated MXMOE auxiliary unsupported on %s; skipping", get_gfx()
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Generated MXMOE auxiliary sort and zero-init validation",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="+",
        choices=[dtypes.bf16],
        default=[dtypes.bf16],
    )
    parser.add_argument("-b", "--batch", type=int, nargs="+", default=[3, 17, 64])
    parser.add_argument(
        "-s",
        "--shape",
        type=dtypes.str2tuple,
        nargs="+",
        default=MODEL_SHAPES,
        help="(expert, model_dim, inter_dim, topk), e.g. -s 24,7168,3072,6",
    )
    args = parser.parse_args()
    prepare_mxfp4_moe_aux(args.shape)
    rows = []
    for dtype, token, (expert, hidden, inter, topk) in itertools.product(
        args.dtype, args.batch, args.shape
    ):
        rows.append(test_mxfp4_moe_aux(dtype, token, expert, hidden, inter, topk))
    aiter.logger.info(
        "generated MXMOE auxiliary summary (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
