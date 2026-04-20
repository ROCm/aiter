from aiter.fused_moe_bf16_asm import asm_moe
from aiter import ActivationType
import torch
from aiter.test_common import run_perftest
from aiter.ops.shuffle import shuffle_weight
import os

os.environ["AITER_LOG_MORE"] = "1"

num_experts = 400
seq_len = 12765
topk = 20
dim = 4096
inter_dim = 1536


seq_len = 20480
x0 = torch.randn(seq_len, 4096, dtype=torch.bfloat16, device='cuda:0')
x1 = torch.randn(seq_len, 20, dtype=torch.float32, device='cuda:0')
w1 = torch.randint(low=-128, high=128, size=(400, 1536, 4096), dtype=torch.int8, device='cuda:0')
w2 = torch.randint(low=-128, high=128, size=(400, 4096, 1536), dtype=torch.int8, device='cuda:0')
x2 = torch.randint(low=0, high=num_experts, size=(seq_len, 20), dtype=torch.int32, device='cuda:0')
fc1_scale = torch.randn(400, 1536, 1, dtype=torch.float32, device='cuda:0')
fc2_scale = torch.randn(400, 4096, 1, dtype=torch.float32, device='cuda:0')
fc1_smooth_scale = torch.randn(400, 4096, dtype=torch.float32, device='cuda:0')
fc2_smooth_scale = torch.randn(400, 1536, dtype=torch.float32, device='cuda:0')
activation = ActivationType.Gelu

w1s = shuffle_weight(w1, (16, 16))
w2s = shuffle_weight(w2, (16, 16))

from aiter.fused_moe import fused_moe
from aiter import QuantType

out, us_aiter = run_perftest(
    fused_moe,
    x0,
    w1s,
    w2s,
    x1,
    x2,
    None,
    ActivationType.Gelu,
    QuantType.per_Token,
    False,
    fc1_scale,
    fc2_scale,
    None,
    None,
    None,
    None,
    0,
    torch.bfloat16,
    0,
    0,
    None,
    None,
    0,
    num_iters=100,
    num_warmup=10,
    testGraph=False,
    num_rotate_args=0,
    needTrace=False,
)
print(f"us_aiter: {us_aiter}")

# y_chunk = asm_moe(
#     x0,
#     w1,
#     w2,
#     x1,
#     x2,
#     fc1_scale,
#     fc2_scale,
#     fc1_smooth_scale,
#     fc2_smooth_scale,
#     True, #False,
#     None,
#     None,
#     None,
#     activation,
# )


# repeat=100
# warmup=1

# start = torch.cuda.Event(enable_timing=True)
# end = torch.cuda.Event(enable_timing=True)


# start.record()
# for i in range(repeat):
#     y_chunk = asm_moe(
#         x0,
#         w1,
#         w2,
#         x1,
#         x2,
#         fc1_scale,
#         fc2_scale,
#         fc1_smooth_scale,
#         fc2_smooth_scale,
#         True, #False,
#         None,
#         None,
#         None,
#         activation,
#     )

# end.record()
# torch.cuda.synchronize()

# avg_ms = start.elapsed_time(end) / repeat
# print(f"fused_moe avg time: {avg_ms:.3f} ms  ({repeat} iters, {warmup} warmup)")

# out, us_aiter = run_perftest(
#     asm_moe,
#     x0,
#     w1,
#     w2,
#     x1,
#     x2,
#     fc1_scale,
#     fc2_scale,
#     fc1_smooth_scale,
#     fc2_smooth_scale,
#     True, #False,
#     None,
#     None,
#     None,
#     activation,
#     None,
#     num_iters=100,
#     num_warmup=2,
#     testGraph=False,
#     num_rotate_args=0,
#     needTrace=False,
# )
# print(f"us_aiter: {us_aiter}")