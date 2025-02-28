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

shape0 = (512,) #501.482 -->213.721us(1)
shape1 = (512,)
stride0 = (1,)
stride1 = (1,)

# shape0 = (1280, 232, 256) #13.906ms(1) ->  22.850ms
# shape1 = (1280, 232, 256)
# stride0 = (59392, 256, 1)
# stride1 = (59392, 256, 1)

# shape0 = (256, 256) #647.601us --> 361.121us
# shape1 = (256, 256)
# stride0 = (256, 1)
# stride1 = (256, 1)

# shape0 = (256, 8192) #822.003us -->710.601us(1)
# shape1 = (256, 8192)
# stride0 = (8192, 1)
# stride1 = (8192, 1)

# shape0 = (256,) #492.081us -->218.041us(1)
# shape1 = (256,)
# stride0 = (1,)
# stride1 = (1,)

# shape0 = (1280, 32, 256) #2.004ms(1) -->3.103ms
# shape1 = (1280, 32, 256)
# stride0 = (8192, 256, 1)
# stride1 = (8192, 256, 1)

# shape0 = (384, 256) #670.403us --> 362.761us
# shape1 = (384, 256)
# stride0 = (256, 1)
# stride1 = (256, 1)

# shape0 = (384,) #2.075ms -->210.200us
# shape1 = (384,)
# stride0 = (1,)
# stride1 = (1,)


# shape0 = (65536,) #2.468ms -->348.240us
# shape1 = (65536,)
# stride0 = (1,)
# stride1 = (1,)

# shape0 = (65536, 256) #30.225ms(1) -->4.069ms
# shape1 = (65536, 256)
# stride0 = (256, 1)
# stride1 = (256, 1)

# shape0 = (1, 8, 256) #510.240us -->296.640us
# shape1 = (1, 8, 256)
# stride0 = (2048, 256, 1)
# stride1 = (2048, 256, 1)

# shape0 = (512, 256) #2.480ms -->378.400us
# shape1 = (512, 256)
# stride0 = (256, 1)
# stride1 = (256, 1)

# shape0 = (1280, 532, 256) #48.577ms(1) -->47.973ms
# shape1 = (1280, 532, 256)
# stride0 = (136192, 256, 1)
# stride1 = (136192, 256, 1)

# shape1 = (4096, 200, 64)
# shape0 = (1, 200, 64)
# stride0 = (12800, 64, 1)
# stride1 = (12800, 64, 1)

# shape1 = (144, 1, 160)
# shape0 = (144, 4096, 160)
# stride1 = (160, 160, 1)
# stride0 = (655360, 160, 1)

tensor0 = torch.empty_strided(shape0, stride0, dtype=torch.bfloat16, device='cuda')
tensor1 = torch.empty_strided(shape1, stride1, dtype=torch.bfloat16, device='cuda')
# tensor0 = torch.empty_strided(shape0, stride0, dtype=torch.float32, device='cuda')
# tensor1 = torch.empty_strided(shape1, stride1, dtype=torch.float32, device='cuda')
random_data0 = torch.rand(shape0)
# tensor0.copy_(random_data0)
tensor0.fill_(0)
random_data1 = torch.rand(shape1)
# tensor1.copy_(random_data1)
tensor1.fill_(2)

print("shape0", shape0)
print("shape1", shape1)
print("strride0:", stride0)
print("strride1:", stride1)

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
