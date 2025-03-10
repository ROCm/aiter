# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import aiter
from aiter.test_common import checkAllclose, perftest
from torch.profiler import profile, record_function, ProfilerActivity

# input shape: torch.Size([4096, 64, 160]) (20480, 1, 128) 
# other shape: torch.Size([4096, 64, 160]) (10240, 160, 1)

# input shape: torch.Size([4096, 64, 160]) (47360, 1, 296)
# other shape: torch.Size([4096, 64, 160]) (10240, 160, 1)

# shape0 = (4096, 64, 160)
# shape1 = (4096, 64, 160)
# stride0 = (47360, 1, 296)
# # stride1 = (20480, 1, 128)
# stride1 = (10240, 160, 1)

# shape1 = (4096, 200, 64)
# shape0 = (1, 200, 64)
# stride0 = (12800, 64, 1)
# stride1 = (12800, 64, 1)

# shape1 = (144, 1, 160)
# shape0 = (144, 4096, 160)
# stride1 = (160, 160, 1)
# stride0 = (655360, 160, 1)

shapes = [(512,), (1280, 232, 256), (256, 256), (256, 8192), (256,), (1280, 32, 256), (384, 256), (384,), (65536,), (65536, 256), (1, 8, 256), (512, 256), (1280, 532, 256)]  
strides = [(1,), (59392, 256, 1), (256, 1), (8192, 1), (1, ), (8192, 256, 1), (256, 1), (1,), (1,), (256, 1), (2048, 256, 1), (256, 1), (136192, 256, 1)]

#shapes = [(32,)]
#strides = [(1,)]

tensors0 = [torch.empty_strided(shape, stride, dtype=torch.bfloat16, device='cuda') for shape, stride in zip(shapes, strides)]  
tensors1 = [torch.empty_strided(shape, stride, dtype=torch.bfloat16, device='cuda') for shape, stride in zip(shapes, strides)]
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
            # result = torch.add_(tensor0, tensor1)
            torch_add_a_ = tensor0.clone()
            torch_add_b_ = tensor1.clone()
            torch_add_a_.add_(torch_add_b_)
            result = torch_add_a_
            # result_con = result.contiguous()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True,
        with_stack=True, with_modules=True, record_shapes = True) as prof:
        for j in range(100):
            #cache_flush1 = torch.randn(10000, 10000, requires_grad=True, device="cuda", dtype=torch.float32).to(torch.int32)
            # output = torch.empty_like(tensor1)
            aiter_add_a_ = tensor0.clone()
            aiter_add_b_ = tensor1.clone()
            output = aiter.add_(aiter_add_a_, aiter_add_b_)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print(torch.equal(result, output))
    #print("result:", result)
    #print("output:", output)
