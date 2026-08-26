# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools

import aiter
import pandas as pd
import torch
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]


def run_torch(
    logits: torch.Tensor,
    context_len: int,
    next_n: int,
    top_k: int,
    stable: bool,
) -> torch.Tensor:
    rows = []
    for row in range(logits.shape[0]):
        row_len = context_len - next_n + row % next_n + 1
        if stable:
            selected = torch.argsort(
                logits[row, :row_len], descending=True, stable=True
            )[:top_k]
            rows.append(torch.sort(selected).values)
        else:
            rows.append(torch.topk(logits[row, :row_len], top_k, sorted=False).indices)
    return torch.stack(rows).to(torch.int32)


def _run_flydsl(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    indices: torch.Tensor,
    top_k: int,
    stable: bool,
) -> torch.Tensor:
    aiter.flydsl_top_k_per_row_decode(
        logits,
        next_n,
        seq_lens,
        indices,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        top_k,
        stable,
    )
    return indices


def _run_hip(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    indices: torch.Tensor,
    top_k: int,
    stable: bool,
) -> torch.Tensor:
    aiter.top_k_per_row_decode(
        logits,
        next_n,
        seq_lens,
        indices,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        top_k,
        stable,
    )
    return indices


@benchmark()
def test_flydsl_topk_decode(
    batch_size: int,
    context_len: int,
    top_k: int,
    next_n: int,
    row_padding: int,
    stable: bool,
) -> dict:
    if context_len - next_n + 1 < top_k:
        raise ValueError("every effective decode row must contain at least top_k items")

    num_rows = batch_size * next_n
    torch.manual_seed(42)
    if stable:
        logits_storage = torch.randint(
            -8,
            9,
            (num_rows, context_len + row_padding),
            dtype=torch.int32,
        ).to(torch.float32)
    else:
        logits_storage = torch.randn(
            num_rows, context_len + row_padding, dtype=torch.float32
        )
    logits = logits_storage[:, :context_len]
    seq_lens = torch.full((batch_size,), context_len, dtype=torch.int32)
    ref = run_torch(logits, context_len, next_n, top_k, stable)

    outputs = {
        "flydsl": torch.empty(num_rows, top_k, dtype=torch.int32),
        "hip": torch.empty(num_rows, top_k, dtype=torch.int32),
    }
    candidates = {
        "flydsl": lambda: _run_flydsl(
            logits, next_n, seq_lens, outputs["flydsl"], top_k, stable
        ),
        "hip": lambda: _run_hip(
            logits, next_n, seq_lens, outputs["hip"], top_k, stable
        ),
    }

    # Stable mode adds deterministic count and ordered-write passes.
    flops = 0
    nbytes = (6 if stable else 5) * logits.numel() * logits.element_size() + outputs[
        "flydsl"
    ].numel() * outputs["flydsl"].element_size()
    ref_compare = (ref if stable else torch.sort(ref, dim=-1).values).to(dtypes.fp32)
    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        out_compare = (out if stable else torch.sort(out, dim=-1).values).to(
            dtypes.fp32
        )
        err = checkAllclose(
            ref_compare,
            out_compare,
            rtol=0,
            atol=0,
            msg=f"{name}: top_k_per_row_decode",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "FlyDSL top_k_per_row_decode unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL per-row decode TopK correctness and performance",
    )
    parser.add_argument("-b", "--batch-size", type=int, nargs="*", default=[4])
    parser.add_argument("-c", "--context-len", type=int, nargs="*", default=[131072])
    parser.add_argument("-k", "--top-k", type=int, nargs="*", default=[2048])
    parser.add_argument("-n", "--next-n", type=int, nargs="*", default=[1, 4])
    parser.add_argument("--row-padding", type=int, nargs="*", default=[0, 17])
    parser.add_argument(
        "--stable", type=dtypes.str2bool, nargs="*", default=[False, True]
    )
    args = parser.parse_args()

    rows = []
    for (
        batch_size,
        context_len,
        top_k,
        next_n,
        row_padding,
        stable,
    ) in itertools.product(
        args.batch_size,
        args.context_len,
        args.top_k,
        args.next_n,
        args.row_padding,
        args.stable,
    ):
        rows.append(
            test_flydsl_topk_decode(
                batch_size, context_len, top_k, next_n, row_padding, stable
            )
        )
    df = pd.DataFrame(rows)
    aiter.logger.info(
        "FlyDSL top_k_per_row_decode summary (markdown):\n%s",
        df.to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
