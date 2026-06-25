# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

import gc

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.ops.topk_plain import topk_plain
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)

# Correctness sweep: a few timed iters is plenty (was 1000). This is a
# correctness check, not a perf gate; the high iter count made the file the
# slowest in its CI shard for no benefit.
NUM_ITERS = 100
NUM_WARMUP = 10

# checkAllclose returns the fraction of mismatching elements (0 == exact) and
# does NOT raise; assert on it so an incorrect topk_plain actually fails CI.
TOL_ERR_RATIO = 0.05


@benchmark()
def run_topk_case(batch_size, hiddensize, topk, largest, dtype):
    device = "cuda"
    # Each row is a permutation of [0, hiddensize) -> distinct values for topk.
    # Vectorised; replaces the per-row Python randperm loop (batch_size iters).
    x = torch.rand(batch_size, hiddensize, device=device).argsort(dim=1).to(dtype)

    topk_ids = torch.zeros((batch_size, topk), dtype=dtypes.i32, device=device)
    topk_value = torch.zeros((batch_size, topk), dtype=dtype, device=device)

    (ref_value, ref_index), us_ref = run_perftest(
        torch.topk,
        x,
        topk,
        largest=largest,
        num_iters=NUM_ITERS,
        num_warmup=NUM_WARMUP,
    )
    id_ref, _ref = torch.sort(ref_index)

    # TODO: re-enable triton topk comparison when it returns in a reasonable time.
    us_triton = 0.0

    _, us_aiter = run_perftest(
        topk_plain,
        x,
        topk_ids,
        topk_value,
        topk,
        largest,
        torch.tensor([], dtype=torch.int32, device=device),  # rowStarts
        torch.tensor([], dtype=torch.int32, device=device),  # rowEnds
        -1,
        1,  # stride0, stride1
        num_iters=NUM_ITERS,
        num_warmup=NUM_WARMUP,
    )
    id_aiter, _aiter = torch.sort(topk_ids.to(torch.long))

    if dtype not in (torch.float16, torch.bfloat16):
        err = checkAllclose(id_ref, id_aiter, msg="topk_ids [golden vs aiter]")
    else:
        # fp16/bf16 can tie within the top-k -> compare values, not indices.
        err = checkAllclose(ref_value, topk_value, msg="topk_values [golden vs aiter]")
    assert err <= TOL_ERR_RATIO, (
        f"topk_plain mismatch: err ratio {err:.4f} > {TOL_ERR_RATIO} "
        f"(batch_size={batch_size}, hiddensize={hiddensize}, topk={topk}, dtype={dtype})"
    )

    # Release this case's buffers before the next (larger) case allocates, so a
    # memory-pressured runner does not OOM accumulating the whole sweep.
    del x, topk_ids, topk_value, ref_value, ref_index, id_ref, id_aiter, _ref, _aiter
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "err": err,
        "us_aiter": us_aiter,
        "us_torch": us_ref,
        "us_triton": us_triton,
    }


BATCH_SIZES = [3072]
HIDDENSIZES = [3072, 4096, 8192, 16384, 32768, 65536, 131072]
TOPKS = [2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
largest = True


def main():
    rows = []
    for batch_size in BATCH_SIZES:
        for hiddensize in HIDDENSIZES:
            for topk in TOPKS:
                if topk > hiddensize:
                    continue
                print(f"\n{'='*60}")
                print(
                    f"Testing: batch_size={batch_size}, hiddensize={hiddensize}, topk={topk}"
                )
                print(f"{'='*60}")
                ret = run_topk_case(batch_size, hiddensize, topk, largest, dtypes.fp32)
                rows.append(
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

    df = pd.DataFrame(rows)
    df["speedup (aiter vs torch)"] = df["time_us (torch)"] / df["time_us (aiter)"]
    df["speedup (aiter vs triton)"] = df["time_us (triton)"] / df["time_us (aiter)"]
    # Summary is informational only -- never let rendering fail the run.
    try:
        table = df.to_markdown(index=False)
    except ImportError:
        table = df.to_string(index=False)  # `tabulate` not installed
    aiter.logger.info("topk_plain summary:\n%s", table)


if __name__ == "__main__":
    main()
