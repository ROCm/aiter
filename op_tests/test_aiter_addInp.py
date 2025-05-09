# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from torch.profiler import profile, ProfilerActivity

shapes = [(512,), (1280, 232, 256), (256, 256), (256, 8192), (256,), (1280, 32, 256), (384, 256), (384,), (65536,), (65536, 256), (1, 8, 256), (512, 256), (1280, 532, 256)]
strides = [(1,), (59392, 256, 1), (256, 1), (8192, 1), (1, ), (8192, 256, 1), (256, 1), (1,), (1,), (256, 1), (2048, 256, 1), (256, 1), (136192, 256, 1)]


class TestAiterAddInp:
    def __init__(self):
        self.collections = []

    def _init_tensors(self, shapes, strides):
        tensors = [torch.empty_strided(shape, stride, dtype=aiter.dtypes.bf16, device="cuda") for shape, stride in zip(shapes, strides)]
        for tensor in tensors:
            tensor.copy_(torch.rand_like(tensor))
        return tensors

    def func(self, tensor_pair):
        tensor0, tensor1 = tensor_pair
        ret = ""
        ret += f"Size: {tensor0.size()}, Stride: {tensor0.stride()}\n"
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True,
            with_stack=True, with_modules=True, record_shapes=True) as prof_torch:

            torch_add_a_ = tensor0.clone()
            torch_add_b_ = tensor1.clone()
            for _ in range(100):
                torch_add_a_.add_(torch_add_b_)
                torch_out = torch_add_a_
        ret += prof_torch.key_averages().table(sort_by="cuda_time_total", row_limit=10)
        ret += "\n"


        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True,
            with_stack=True, with_modules=True, record_shapes=True) as prof_aiter:
            aiter_add_a_ = tensor0.clone()
            aiter_add_b_ = tensor1.clone()
            for _ in range(100):
                aiter_out = aiter.add_(aiter_add_a_, aiter_add_b_)
        ret += prof_aiter.key_averages().table(sort_by="cuda_time_total", row_limit=10)
        ret += "\n"

        self.collections.append([tensor0.size(), tensor1.size(), torch.equal(torch_out, aiter_out)])

        print(ret)

    def exec(self):
        tensors0 = self._init_tensors(shapes, strides)
        tensors1 = self._init_tensors(shapes, strides)
        for tensor0, tensor1 in zip(tensors0, tensors1):
            self.func((tensor0, tensor1))

    def analysis(self):
        for t0_size, t1_size, is_equal in self.collections:
            if not is_equal:
                print(f"tensor0 shape: {t0_size}, tensor1 shape: {t1_size} " + \
                    "=> torch output is different from aiter output")

if __name__ == "__main__":
    aiter_add_inp = TestAiterAddInp()
    aiter_add_inp.exec()
    aiter_add_inp.analysis()
