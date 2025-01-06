import torch
import torch.nn.functional as F
import ater
from ater.test_common import checkAllclose, perftest
from torch.profiler import profile, record_function, ProfilerActivity

# input shape: torch.Size([4096, 64, 160]) (20480, 1, 128) 
# other shape: torch.Size([4096, 64, 160]) (10240, 160, 1)

# input shape: torch.Size([4096, 64, 160]) (47360, 1, 296)
# other shape: torch.Size([4096, 64, 160]) (10240, 160, 1)

# shape = (4096, 64, 160)
# stride1 = (47360, 1, 296)
# # stride1 = (20480, 1, 128)
# stride2 = (10240, 160, 1)

# shape0 = (4096, 200, 64)
# shape1 = (1, 200, 64)
# stride0 = (12800, 64, 1)
# stride1 = (12800, 64, 1)

# [[144, 1, 160], [144, 4096, 160], []]
# [[160, 160, 1], [655360, 160, 1], []]


shape0 = (144, 1, 160)
shape1 = (144, 4096, 160)
stride0 = (160, 160, 1)
stride1 = (655360, 160, 1)

# shape0 = (2, 1, 32)
# shape1 = (2, 64, 32)
# stride0 = (32, 32, 1)
# stride1 = (64*32, 32, 1)


# shape0 = (1, 1, 16)
# shape1 = (1, 18, 16)
# stride0 = (16, 16, 1)
# stride1 = (16*18, 16, 1)

tensor0 = torch.empty_strided(shape0, stride0, dtype=torch.float16, device='cuda')
tensor1 = torch.empty_strided(shape1, stride1, dtype=torch.float16, device='cuda')
# tensor0 = torch.empty_strided(shape, stride1, dtype=torch.float32, device='cuda')
# tensor1 = torch.empty_strided(shape, stride2, dtype=torch.float32, device='cuda')
random_data0 = torch.rand(shape0)
tensor0.copy_(random_data0)
# tensor0.fill_(0)
random_data1 = torch.rand(shape1)
tensor1.copy_(random_data1)
# tensor1.fill_(2)

print("shape0", shape0)
print("shape1", shape1)
print("strride0:", stride0)
print("strride1:", stride1)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True,
    with_stack=True, with_modules=True, record_shapes = True) as prof:
    for j in range(100):
        #cache_flush1 = torch.randn(10000, 10000, requires_grad=True, device="cuda", dtype=torch.float32).to(torch.int32)
        result = torch.add(tensor0, tensor1)
        result_con = result.contiguous()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True,
    with_stack=True, with_modules=True, record_shapes = True) as prof:
    for j in range(100):
        #cache_flush1 = torch.randn(10000, 10000, requires_grad=True, device="cuda", dtype=torch.float32).to(torch.int32)
        # output = torch.empty_like(tensor1)
        output = ater.transpose_add(tensor0, tensor1)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("result_con", result_con)
print("output", output)
print(torch.equal(result_con, output))
