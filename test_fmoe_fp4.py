import torch
import os
from aiter.fused_moe import fused_moe, fused_topk
from aiter import ActivationType, QuantType
from aiter.ops.quant import get_torch_quant
from aiter.test_common import run_perftest

os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["AITER_LOG_MORE"] = "1"

device = "cuda"
seq_len = 20480 #19147
num_experts = 400
model_dim = 4096
inter_dim = 1536
topk = 20
dtype = torch.bfloat16

torch.manual_seed(0)

x0 = torch.randn(seq_len, model_dim, dtype=dtype, device=device)

w1_bf16 = torch.randn(num_experts, inter_dim, model_dim, dtype=dtype, device=device) / 10
w2_bf16 = torch.randn(num_experts, model_dim, inter_dim, dtype=dtype, device=device) / 10

fp4_quant = get_torch_quant(QuantType.per_1x32)
w1_fp4, w1_scale = fp4_quant(w1_bf16)
w1_fp4 = w1_fp4.view(num_experts, inter_dim, model_dim // 2)
w2_fp4, w2_scale = fp4_quant(w2_bf16)
w2_fp4 = w2_fp4.view(num_experts, model_dim, inter_dim // 2)

score = torch.randn(seq_len, num_experts, dtype=dtype, device=device)
topk_weight, topk_ids = fused_topk(x0, score, topk, True)

out, us_aiter = run_perftest(
    fused_moe,
    x0,
    w1_fp4,
    w2_fp4,
    topk_weight,
    topk_ids,
    None,
    ActivationType.Gelu,
    QuantType.per_1x32,
    False,
    w1_scale,
    w2_scale,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    0,
    dtype,
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
