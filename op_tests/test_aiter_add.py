# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from torch.profiler import profile, ProfilerActivity



shapes0 = [
        (512,), (1280, 232, 256), (256, 256), (256, 8192), (256,), (1280, 32, 256),
        (384, 256), (384,), (65536,), (65536, 256), (1, 8, 256), (512, 256),
        (1280, 532, 256),
        (6144, 100, 96), (1, 100, 96),
        (6144, 16, 96), (6144, 1, 96),
        (6144, 289, 96), (289, 1),
        (6144, 16, 192), (192,),
]
strides0 = [
        (1,), (59392, 256, 1), (256, 1), (8192, 1), (1, ), (8192, 256, 1),
        (256, 1), (1,), (1,), (256, 1), (2048, 256, 1), (256, 1),
        (136192, 256, 1),
        (9600, 96, 1), (9600, 96, 1),
        (16*96, 96, 1), (96, 96, 1),
        (289*96, 96, 1), (1, 1),
        (16*192, 192, 1), (1,),
        (8, 1, 1), (1,),
]

shapes1 = [
        (512,), (1280, 232, 256), (256, 256), (256, 8192), (256,), (1280, 32, 256),
        (384, 256), (384,), (65536,), (65536, 256), (1, 8, 256), (512, 256),
        (1280, 532, 256),
        (1, 100, 96), (6144, 100, 96),
        (6144, 1, 96), (6144, 16, 96),
        (289, 1), (6144, 289, 96),
        (192,), (6144, 16, 192),
        (1,), (6144, 8, 1),
]
strides1 = [
        (1,), (59392, 256, 1), (256, 1), (8192, 1), (1, ), (8192, 256, 1),
        (256, 1), (1,), (1,), (256, 1), (2048, 256, 1), (256, 1),
        (136192, 256, 1),
        (9600, 96, 1), (9600, 96, 1),
        (96, 96, 1), (16*96, 96, 1),
        (1, 1), (289*96, 96, 1),
        (1,), (16*192, 192, 1),
        (1,), (8, 1, 1),
]

class TestAiterAdd:
    def __init__(self):
        self.collections = []

    def _init_tensors(self, shapes, strides, dtype=torch.bfloat16):
        tensors = [torch.empty_strided(shape, stride, dtype=dtype, device="cuda") for shape, stride in zip(shapes, strides)]
        for tensor in tensors:
            tensor.copy_(torch.rand_like(tensor))
        return tensors

    def func(self, tensor_pair):
        tensor0, tensor1 = tensor_pair
        ret = ""
        ret += f"Size: {tensor0.size()}, Stride: {tensor0.stride()}\n"

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True,
            with_stack=True, with_modules=True, record_shapes=True) as prof_torch:
            for _ in range(100):
                torch_out = torch.add(tensor0, tensor1)
        ret += prof_torch.key_averages().table(sort_by="cuda_time_total", row_limit=10)
        ret += "\n"


        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True,
            with_stack=True, with_modules=True, record_shapes=True) as prof_aiter:
            for _ in range(100):
                aiter_out = aiter.add(tensor0, tensor1)
        ret += prof_aiter.key_averages().table(sort_by="cuda_time_total", row_limit=10)
        ret += "\n"

        self.collections.append([tensor0.size(), tensor1.size(), torch.equal(torch_out, aiter_out)])
        print(ret)

    def exec(self):
        tensors0 = self._init_tensors(shapes0, strides0)
        tensors1 = self._init_tensors(shapes1, strides1)
        for tensor0, tensor1 in zip(tensors0, tensors1):
            self.func((tensor0, tensor1))

    def analysis(self):
        for t0_size, t1_size, is_equal in self.collections:
            if not is_equal:
                print(f"tensor0 shape: {t0_size}, tensor1 shape: {t1_size} " + \
                    "=> torch output is different from aiter output")

if __name__ == "__main__":
    aiter_add = TestAiterAdd()
    aiter_add.exec()
    aiter_add.analysis()
