import torch
import torch.nn.functional as F
import aiter
#from ater.test_common import checkAllclose, perftest
from torch.profiler import profile, record_function, ProfilerActivity

# input shape: torch.Size([4096, 64, 160]) (20480, 1, 128) 
# other shape: torch.Size([4096, 64, 160]) (10240, 160, 1)

# input shape: torch.Size([4096, 64, 160]) (47360, 1, 296)
# other shape: torch.Size([4096, 64, 160]) (10240, 160, 1)

shape0 = (4096,880)
stride0 = (880, 1)

# shape0 = (16,16)
# stride0 = (16, 1)

tensor0 = torch.empty_strided(shape0, stride0, dtype=torch.float16, device='cuda')
random_data0 = torch.rand(shape0)
tensor0.copy_(random_data0)
# tensor0.fill_(1)

print("shape0", shape0)
print("strride0:", stride0)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True,
    with_stack=True, with_modules=True, record_shapes = True) as prof:
    for j in range(100):
        #cache_flush1 = torch.randn(10000, 10000, requires_grad=True, device="cuda", dtype=torch.float32).to(torch.int32)
        result = torch.sigmoid(tensor0)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True,
    with_stack=True, with_modules=True, record_shapes = True) as prof:
    for j in range(100):
        #cache_flush1 = torch.randn(10000, 10000, requires_grad=True, device="cuda", dtype=torch.float32).to(torch.int32)
        output = aiter.sigmoid(tensor0)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print(torch.equal(result, output))
print("result:", result)
print("output:", output)

# 逐元素比较
# elementwise_close = torch.isclose(result, output, atol=1e-2)

# 判断是否所有元素都满足条件
# print(torch.all(elementwise_close))  # True