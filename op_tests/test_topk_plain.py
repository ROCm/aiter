# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter.test_common import (
    checkAllclose,
    benchmark,
    run_perftest,
    perftest,
)
from aiter import dtypes, get_gfx
from aiter.ops.triton.topk import topk as triton_topk
import pandas as pd
import argparse

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

    row = torch.arange(
        hiddensize, dtype=dtypes.i32, device=device
    )  # [0, 1, ..., length-1]
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
            f"  {'Method':<10} {'Time (μs)':>12}\n"
            f"  {'-'*10} {'-'*12}\n"
            f"  {'golden':<10} {us_ref:>12.2f}\n"
            f"  {'triton':<10} {us_triton:>12.2f}\n"
        ),
    )

    _, us_aiter = run_perftest(
        aiter.topk_plain,
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
            f"  {'Method':<10} {'Time (μs)':>12}\n"
            f"  {'-'*10} {'-'*12}\n"
            f"  {'golden':<10} {us_ref:>12.2f}\n"
            f"  {'triton':<10} {us_triton:>12.2f}\n"
            f"  {'aiter':<10} {us_aiter:>12.2f}\n"
        ),
    )

    return {"err": err, "us": us_aiter}


# BATCH_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 16, 1335]
# DIM2 = [16, 128256]
# K = [2, 8]

batch_size = 1000
hiddensize = 100000
topk = 64
largest = True

df = []
ret = test_topk(
    batch_size,
    hiddensize,
    topk,
    largest,
    dtypes.fp32,
)
df = pd.DataFrame(df)

aiter.logger.info(f"summary:\n{df}")
