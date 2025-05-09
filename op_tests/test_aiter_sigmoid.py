# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter

from torch.profiler import profile, ProfilerActivity
shape0 = (4096, 880)
stride0 = (880, 1)

# from ater.test_common import checkAllclose, perftest
from torch.profiler import profile, record_function, ProfilerActivity

# input shape: torch.Size([4096, 64, 160]) (20480, 1, 128)
# other shape: torch.Size([4096, 64, 160]) (10240, 160, 1)

class TestAiterSigmoid:
    def __init__(self):
        self.collections = []


    def exec(self):
        tensor0 = torch.empty_strided(shape0, stride0, dtype=torch.float16, device='cuda')
        tensor0.copy_(torch.rand(shape0))
        ret = ""

        ret += f"shape0: {shape0}, stride0: {stride0}\n"

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True,
            with_stack=True, with_modules=True, record_shapes = True) as torch_prof:
            for _ in range(100):
                torch_out = torch.sigmoid(tensor0)
        ret += torch_prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
        ret += "\n"


        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True,
            with_stack=True, with_modules=True, record_shapes = True) as aiter_prof:
            for _ in range(100):
                aiter_out = aiter.sigmoid(tensor0)

        ret += aiter_prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
        ret += "\n"

        self.collections.append([shape0, stride0, torch.equal(torch_out, aiter_out)])

    def analysis(self):
        for shape, stride, is_equal in self.collections:
            if not is_equal:
                print(f"shape: {shape}, stride: {stride} " + \
                    "=> torch output is different from aiter output")


if __name__ == "__main__":
    aiter_sigmoid = TestAiterSigmoid()
    aiter_sigmoid.exec()
    aiter_sigmoid.analysis()
    