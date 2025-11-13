# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from aiter.test_common import (
    checkAllclose,
    benchmark,
    run_perftest,
)
from aiter import dtypes, logger
from aiter.ops.triton.topk import topk as triton_topk
from aiter.ops.topk_plain import topk_plain
import pandas as pd

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)


@benchmark()
def test_topk(
    batch_size,
    hiddensize,
    topk,
    largest,
    dtype,
):
    output = torch.randn((batch_size, hiddensize), dtype=dtype)
    device = output.device

    topk_ids = torch.zeros((batch_size, topk), dtype=dtypes.i32, device=device)

    x = torch.arange(hiddensize, dtype=dtype).repeat(batch_size, 1)
    for b in range(batch_size):
        x[b] = x[b, torch.randperm(hiddensize)]

    (ref_value, ref_index), us_ref = run_perftest(
        torch.topk,
        x,
        topk,
        largest=largest,
        num_iters=1000,
        num_warmup=100,
    )

    (res_triton_value, res_triton_index), us_triton = run_perftest(
        triton_topk,
        x,
        topk,
        largest=largest,
        num_iters=1000,
        num_warmup=100,
    )

    id_ref, _ref = torch.sort(ref_index)
    id_triton, _triton = torch.sort(res_triton_index)
    err = checkAllclose(
        ref_value.gather(1, _ref),
        res_triton_value.gather(1, _triton),
        msg="topk_values [golden vs triton]",
    )
    checkAllclose(
        id_ref,
        id_triton,
        msg=(
            f"topk_ids Performance Comparison:\n"
            f"  {'Method':<10} {'Time (us)':>12}\n"
            f"  {'-'*10} {'-'*12}\n"
            f"  {'golden':<10} {us_ref:>12.2f}\n"
            f"  {'triton':<10} {us_triton:>12.2f}\n"
        ),
    )

    _, us_aiter = run_perftest(
        topk_plain,
        x,
        topk_ids,
        topk,
        largest,
    )

    id_aiter, _aiter = torch.sort(topk_ids.to(torch.long))
    checkAllclose(
        id_ref,
        id_aiter,
        msg=(
            f"topk_ids Performance Comparison:\n"
            f"  {'Method':<10} {'Time (us)':>12}\n"
            f"  {'-'*10} {'-'*12}\n"
            f"  {'golden':<10} {us_ref:>12.2f}\n"
            f"  {'triton':<10} {us_triton:>12.2f}\n"
            f"  {'aiter':<10} {us_aiter:>12.2f}\n"
        ),
    )

    return {
        "err": err,
        "us_aiter": us_aiter,
        "us_torch": us_ref,
        "us_triton": us_triton,
    }


BATCH_SIZES = [100, 1000, 10000]
HIDDENSIZES = [10000, 100000]
topk = 64
largest = True

df = []
for batch_size in BATCH_SIZES:
    for hiddensize in HIDDENSIZES:
        print(f"\n{'='*60}")
        print(f"Testing: batch_size={batch_size}, hiddensize={hiddensize}, topk={topk}")
        print(f"{'='*60}")
        ret = test_topk(
            batch_size,
            hiddensize,
            topk,
            largest,
            dtypes.fp32,
        )
        df.append(
            {
                "batch_size": batch_size,
                "hiddensize": hiddensize,
                "topk": topk,
                "error": ret["err"],
                "time_us (aiter)": ret["us_aiter"],
                "time_us (torch)": ret["us_torch"],
                "time_us (triton)": ret["us_triton"],
            }
        )

df = pd.DataFrame(df)

# Add speedup columns
df["speedup (aiter vs torch)"] = df["time_us (torch)"] / df["time_us (aiter)"]
df["speedup (aiter vs triton)"] = df["time_us (triton)"] / df["time_us (aiter)"]

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
logger.info(f"\n{df.to_string(index=False)}")
