# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import aiter
from aiter.test_common import checkAllclose, perftest
from torch.profiler import profile, record_function, ProfilerActivity

input_shapes = [
        (512,), (1280, 232, 256), (256, 256), (256, 8192), (256,), (1280, 32, 256), 
        (384, 256), (384,), (65536,), (65536, 256), (1, 8, 256), (512, 256), 
        (1280, 532, 256),
        (6144, 100, 96), (1, 100, 96),
        (6144, 16, 96), (6144, 1, 96),
        (6144, 289, 96), (289, 1),
        (6144, 16, 192), (192,),
        (6144, 8, 1), (1,),
        ]  
input_strides = [
        (1,), (59392, 256, 1), (256, 1), (8192, 1), (1, ), (8192, 256, 1), 
        (256, 1), (1,), (1,), (256, 1), (2048, 256, 1), (256, 1), 
        (136192, 256, 1),
        (9600, 96, 1), (9600, 96, 1),
        (16*96, 96, 1), (96, 96, 1),
        (289*96, 96, 1), (1, 1),
        (16*192, 192, 1), (1,),
        (8, 1, 1), (1,),
        ]

other_shapes = [
        (512,), (1280, 232, 256), (256, 256), (256, 8192), (256,), (1280, 32, 256), 
        (384, 256), (384,), (65536,), (65536, 256), (1, 8, 256), (512, 256), 
        (1280, 532, 256),
        (1, 100, 96), (6144, 100, 96),
        (6144, 1, 96), (6144, 16, 96),
        (289, 1), (6144, 289, 96),
        (192,), (6144, 16, 192),
        (1,), (6144, 8, 1),
        ]  
other_strides = [
        (1,), (59392, 256, 1), (256, 1), (8192, 1), (1, ), (8192, 256, 1), 
        (256, 1), (1,), (1,), (256, 1), (2048, 256, 1), (256, 1), 
        (136192, 256, 1),
        (9600, 96, 1), (9600, 96, 1),
        (96, 96, 1), (16*96, 96, 1),
        (1, 1), (289*96, 96, 1),
        (1,), (16*192, 192, 1),
        (1,), (8, 1, 1),
        ]

tensors0 = [torch.empty_strided(shape, stride, dtype=torch.bfloat16, device='cuda') for shape, stride in zip(input_shapes, input_strides)]  
tensors1 = [torch.empty_strided(shape, stride, dtype=torch.bfloat16, device='cuda') for shape, stride in zip(other_shapes, other_strides)]
for tensor in tensors0:
    tensor.copy_(torch.rand_like(tensor))  
    # tensor.fill_(1)
for tensor in tensors1:
    tensor.copy_(torch.rand_like(tensor))  
    # tensor.fill_(1)

# tensor0 = torch.empty_strided(shape0, stride0, dtype=torch.bfloat16, device='cuda')
# tensor1 = torch.empty_strided(shape1, stride1, dtype=torch.bfloat16, device='cuda')
# # tensor0 = torch.empty_strided(shape0, stride0, dtype=torch.float32, device='cuda')
# # tensor1 = torch.empty_strided(shape1, stride1, dtype=torch.float32, device='cuda')
# random_data0 = torch.rand(shape0)
# # tensor0.copy_(random_data0)
# tensor0.fill_(0)
# random_data1 = torch.rand(shape1)
# # tensor1.copy_(random_data1)
# tensor1.fill_(2)

for tensor0, tensor1 in zip(tensors0, tensors1):
    print("shape:", tensor0.size())  
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True,
        with_stack=True, with_modules=True, record_shapes = True) as prof:
        for j in range(100):
            #cache_flush1 = torch.randn(10000, 10000, requires_grad=True, device="cuda", dtype=torch.float32).to(torch.int32)
            result = torch.add(tensor0, tensor1)
            # result_con = result.contiguous()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True,
        with_stack=True, with_modules=True, record_shapes = True) as prof:
        for j in range(100):
            #cache_flush1 = torch.randn(10000, 10000, requires_grad=True, device="cuda", dtype=torch.float32).to(torch.int32)
            # output = torch.empty_like(tensor1)
            output = aiter.add(tensor0, tensor1)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print(torch.equal(result, output))
# print("result:", result)
# print("output:", output)
