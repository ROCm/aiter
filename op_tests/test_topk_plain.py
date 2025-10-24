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

    row = torch.arange(hiddensize, dtype=dtypes.i32, device=device)  # [0, 1, ..., length-1]
    topk_ids = row.unsqueeze(0).expand(batch_size, -1).clone()

    x = torch.arange(hiddensize, dtype=dtype).repeat(batch_size, 1)
    for b in range(batch_size):
        x[b] = x[b, torch.randperm(hiddensize)]

    _, us_aiter = run_perftest(
        aiter.topk_plain,
        x,
        topk_ids,
        topk,
        largest,
    )
    
    err = 0
    return {"err": err, "us": us_aiter}

# BATCH_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 16, 1335]
# DIM2 = [16, 128256]
# K = [2, 8]

batch_size = 400
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
