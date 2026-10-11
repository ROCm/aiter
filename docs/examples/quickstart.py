# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Small reference checks for a ROCm installation; run from the repository root."""

# BEGIN environment
from importlib.metadata import version

import torch
import torch.nn.functional as F

from aiter.fused_moe import fused_moe, fused_topk, torch_moe
from aiter.ops.activation import silu_and_mul
from aiter.ops.mha import flash_attn_func, flash_attn_varlen_func
from aiter.ops.rmsnorm import rms_norm
from aiter.ops.shuffle import shuffle_weight

assert torch.version.hip, "Install a ROCm build of PyTorch"
assert torch.cuda.is_available(), "No accessible GPU"
print("amd-aiter:", version("amd-aiter"))
print("PyTorch:", torch.__version__, "HIP:", torch.version.hip)
print("GPU:", torch.cuda.get_device_name(0))
print("Architecture:", torch.cuda.get_device_properties(0).gcnArchName)
# END environment

torch.manual_seed(0)

# BEGIN attention
# AITER: [batch, sequence, heads, dimension]. PyTorch SDPA: [batch, heads, sequence, dimension].
q, k, v = [
    torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.float16) for _ in range(3)
]
out = flash_attn_func(q, k, v, causal=True)
# FP32 math reference avoids selecting another fused half-precision attention kernel.
scores = q.float().transpose(1, 2) @ k.float().transpose(1, 2).transpose(-2, -1)
scores /= q.shape[-1] ** 0.5
mask = torch.ones(128, 128, device="cuda", dtype=torch.bool).triu(1)
scores.masked_fill_(mask, float("-inf"))
reference = (scores.softmax(-1) @ v.float().transpose(1, 2)).transpose(1, 2)
torch.testing.assert_close(out.float(), reference, rtol=1e-2, atol=2e-3)
print("attention:", tuple(out.shape), "reference check passed")
# END attention

# BEGIN varlen
# Packed sequences of lengths 64 and 128; no page table in this example.
qv, kv, vv = [
    torch.randn(192, 8, 64, device="cuda", dtype=torch.float16) for _ in range(3)
]
cu = torch.tensor([0, 64, 192], device="cuda", dtype=torch.int32)
packed = flash_attn_varlen_func(qv, kv, vv, cu, cu, 128, 128, causal=True)
for start, end in [(0, 64), (64, 192)]:
    single = flash_attn_func(
        qv[start:end][None], kv[start:end][None], vv[start:end][None], causal=True
    )
    torch.testing.assert_close(packed[start:end], single[0], rtol=1e-2, atol=2e-3)
print("packed attention:", tuple(packed.shape), "sequence isolation check passed")
# END varlen

# BEGIN rmsnorm
x = torch.randn(32, 4096, device="cuda", dtype=torch.bfloat16)
weight = torch.ones(4096, device="cuda", dtype=torch.bfloat16)
epsilon = 1e-6
normalized = rms_norm(x, weight, epsilon=epsilon)
reference = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + epsilon)
reference *= weight.float()
torch.testing.assert_close(normalized.float(), reference, rtol=1e-2, atol=2e-2)
print("rmsnorm:", tuple(normalized.shape), "reference check passed")
# END rmsnorm

# BEGIN moe
# BF16, SiLU gate/up, no quantization, local experts. Layout follows test_moe_2stage.py.
tokens, hidden, intermediate, experts, top_k = 32, 512, 256, 8, 2
x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) / 10
w1 = (
    torch.randn(experts, 2 * intermediate, hidden, device="cuda", dtype=torch.bfloat16)
    / 10
)
w2 = (
    torch.randn(experts, hidden, intermediate, device="cuda", dtype=torch.bfloat16) / 10
)
logits = torch.randn(tokens, experts, device="cuda", dtype=torch.float32)
topk_weights, topk_ids = fused_topk(x, logits, top_k, renormalize=True)
reference = torch_moe(x, w1, w2, topk_weights, topk_ids)
# Preserve raw weights for the reference; the kernel consumes its own shuffled layout.
output = fused_moe(x, shuffle_weight(w1), shuffle_weight(w2), topk_weights, topk_ids)
torch.testing.assert_close(output, reference, rtol=5e-2, atol=2e-2)
print("moe:", tuple(output.shape), "reference check passed")
# END moe

# BEGIN activation
activation_input = torch.randn(32, 512, device="cuda", dtype=torch.float16)
activation_out = torch.empty(32, 256, device="cuda", dtype=torch.float16)
silu_and_mul(activation_out, activation_input)
gate, up = activation_input.float().chunk(2, dim=-1)
torch.testing.assert_close(
    activation_out.float(), F.silu(gate) * up, rtol=1e-2, atol=2e-3
)
print("activation:", tuple(activation_out.shape), "reference check passed")
# END activation
