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
    batch,
    length,
    topk_num,
    dtype,
):
    output = torch.randn((batch, length), dtype=dtype)
    device = output.device

    row = torch.arange(length, dtype=dtypes.i32, device=device)  # [0, 1, ..., length-1]
    topk_ids = row.unsqueeze(0).expand(batch, -1).clone()

    _, us_aiter = run_perftest(
        aiter.topk_plain,
        output,
        topk_ids,
        topk_num,
    )
    
    
    err = 0
    return {"err": err, "us": us_aiter}


batch = 400
length = 100000
topk_num = 64


df = []
ret = test_topk(
    batch,
    length,
    topk_num,
    dtypes.fp32,
)
df = pd.DataFrame(df)

aiter.logger.info(f"summary:\n{df}")
