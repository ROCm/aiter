#!/usr/bin/env python3
"""Minimal debug test for the flat FMoE kernel.

Prepares data the same way dev-platform's fmoe_dispatch.cpp does,
calls the kernel directly, and compares against a torch reference.
"""
import torch
import aiter
from aiter import dtypes, pertoken_quant
from aiter.fused_moe import moe_sorting, fused_topk

torch.set_default_device("cuda")
torch.manual_seed(42)

# GLM5.2-FP8 params
token = 1
d_model = 6144
d_ff = 2048
experts = 256
topk = 8
blk_n, blk_k = 128, 128

print(f"=== Flat kernel debug test ===")
print(f"token={token} d_model={d_model} d_ff={d_ff} experts={experts} topk={topk}")

# --- Create random data ---
input_bf16 = torch.randn((token, d_model), dtype=torch.bfloat16)
w1_f32 = torch.randn((experts, d_ff * 2, d_model), dtype=torch.float32) / 10
w2_f32 = torch.randn((experts, d_model, d_ff), dtype=torch.float32) / 10
score = torch.randn((token, experts), dtype=torch.bfloat16)
topk_weights, topk_ids = fused_topk(input_bf16, score, topk, True)

# --- Block-quantize weights (same as dev-platform) ---
def block_quant_weight(w, blk_n, blk_k):
    """Quantize weight to FP8 with per-block scaling, matching dev-platform."""
    E, N, K = w.shape
    w_blocks = w.view(E, N // blk_n, blk_n, K // blk_k, blk_k)
    w_blocks = w_blocks.permute(0, 1, 3, 2, 4).contiguous()  # [E, nbn, nbk, blk_n, blk_k]
    # Per-block max for scale
    amax = w_blocks.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
    scale = amax / 240.0  # FP8 E4M3FNUZ max = 240
    w_q = (w_blocks / scale).clamp(-240, 240).to(torch.float8_e4m3fnuz)
    scale = scale.squeeze(-1).squeeze(-1)  # [E, nbn, nbk]
    # Reshape back to [E, N, K]
    w_q = w_q.permute(0, 1, 3, 2, 4).contiguous().view(E, N, K)
    return w_q, scale

w1_q, w1_scale = block_quant_weight(w1_f32, blk_n, blk_k)
w2_q, w2_scale = block_quant_weight(w2_f32, blk_n, blk_k)

print(f"w1_q: {w1_q.shape} {w1_q.dtype}")
print(f"w1_scale: {w1_scale.shape} {w1_scale.dtype}")
print(f"w2_q: {w2_q.shape} {w2_q.dtype}")
print(f"w2_scale: {w2_scale.shape} {w2_scale.dtype}")

# --- Dev-platform 16x16 tile shuffle ---
def devplat_shuffle(x):
    E, N, K = x.shape
    x = x.view(E, N // 16, 16, K // 16, 16)
    x = x.permute(0, 1, 3, 2, 4).contiguous()
    return x.view(E, N, K)

w1_shuffled = devplat_shuffle(w1_q)
w2_shuffled = devplat_shuffle(w2_q)

# --- Torch reference (BF16 input, dequantized FP8 weights) ---
def torch_moe_ref(input_bf16, w1_q, w2_q, w1_scale, w2_scale, topk_weights, topk_ids,
                  d_ff, blk_n, blk_k):
    """Simple torch MoE reference with block-scaled FP8 weights."""
    E, N1, K1 = w1_q.shape
    token_num = input_bf16.shape[0]
    topk = topk_ids.shape[1]

    # Dequantize weights
    nbn1 = N1 // blk_n
    nbk1 = K1 // blk_k
    w1_f = w1_q.float().view(E, nbn1, blk_n, nbk1, blk_k) * \
           w1_scale.view(E, nbn1, 1, nbk1, 1)
    w1_f = w1_f.view(E, N1, K1)

    E2, N2, K2 = w2_q.shape
    nbn2 = N2 // blk_n
    nbk2 = K2 // blk_k
    w2_f = w2_q.float().view(E2, nbn2, blk_n, nbk2, blk_k) * \
           w2_scale.view(E2, nbn2, 1, nbk2, 1)
    w2_f = w2_f.view(E2, N2, K2)

    input_f = input_bf16.float()
    out = torch.zeros((token_num, K1), dtype=torch.float32, device="cuda")

    for t in range(token_num):
        for k in range(topk):
            eid = topk_ids[t, k].item()
            w = topk_weights[t, k].item()
            x = input_f[t]
            # Stage 1: gate + up + SiLU
            gu = x @ w1_f[eid].T  # [d_ff*2]
            gate = gu[:d_ff]
            up = gu[d_ff:]
            act = torch.nn.functional.silu(gate) * up
            # Stage 2: down
            down = act @ w2_f[eid].T  # [d_model]
            out[t] += w * down

    return out.to(torch.bfloat16)

print("\n--- Computing torch reference ---")
ref = torch_moe_ref(input_bf16, w1_q, w2_q, w1_scale, w2_scale,
                     topk_weights, topk_ids, d_ff, blk_n, blk_k)
print(f"ref: {ref.shape} first 4 values: {ref[0,:4]}")

# --- Call kernel via fmoe_fp8_blockscale_g1u1 ---
# Flat: pass raw topk_ids/weights, no sorting
print("\n--- Setting up flat kernel call ---")
topk_ids_i32 = topk_ids.to(torch.int32).contiguous()
topk_weights_f32 = topk_weights.to(torch.float32).contiguous()

# Output buffer with 8-byte flag region for MARKER_ZERO
elem_size = 2  # bf16
row_bytes = token * d_model * elem_size
flat_buf = torch.empty(row_bytes + 8, dtype=torch.uint8, device="cuda")
moe_buf = flat_buf[:row_bytes].view(torch.bfloat16).view(token, d_model)

# Flatten scales for kernel
w1_scale_flat = w1_scale.view(experts, -1).contiguous()
w2_scale_flat = w2_scale.view(experts, -1).contiguous()

# Empty input scale (xbf16 kernel computes its own)
a1_scale = torch.empty(0, device="cuda")

kernel_name = "_ZN5aiter56fmoe_bf16_a16_blockscaleFp8_g1u1_vs_silu_1tg_16x128_flatE"

print(f"moe_buf: {moe_buf.shape} {moe_buf.dtype}")
print(f"input_bf16: {input_bf16.shape} {input_bf16.dtype}")
print(f"w1_shuffled: {w1_shuffled.shape} {w1_shuffled.dtype}")
print(f"w2_shuffled: {w2_shuffled.shape} {w2_shuffled.dtype}")
print(f"topk_ids_i32: {topk_ids_i32.shape} {topk_ids_i32.dtype}")
print(f"topk_weights_f32: {topk_weights_f32.shape} {topk_weights_f32.dtype}")
print(f"w1_scale_flat: {w1_scale_flat.shape} {w1_scale_flat.dtype}")
print(f"w2_scale_flat: {w2_scale_flat.shape} {w2_scale_flat.dtype}")
print(f"a1_scale: {a1_scale.shape}")

print("\n--- Calling kernel ---")
try:
    aiter.fmoe_fp8_blockscale_g1u1(
        moe_buf,           # output
        input_bf16,        # input (BF16)
        w1_shuffled,       # gate+up weights (FP8, shuffled)
        w2_shuffled,       # down weights (FP8, shuffled)
        topk_ids_i32,      # sorted_token_ids (raw topk_ids for flat)
        topk_weights_f32,  # sorted_weights (raw topk_weights for flat)
        topk_ids_i32,      # sorted_expert_ids (dummy, unused by flat)
        topk_ids_i32,      # num_valid_ids (dummy, unused by flat)
        topk,
        a1_scale,          # input scale (empty for xbf16)
        w1_scale_flat,     # fc1 scale
        w2_scale_flat,     # fc2 scale
        kernel_name,
        128,               # fc_scale_blkn
        128,               # fc_scale_blkk
        None,              # fc2_smooth_scale
    )
    torch.cuda.synchronize()
    print(f"kernel output: {moe_buf.shape} first 4 values: {moe_buf[0,:4]}")
    print(f"ref first 4:    {ref[0,:4]}")

    # Compare
    diff = (moe_buf.float() - ref.float()).abs()
    max_diff = diff.max().item()
    rel_err = diff.sum().item() / ref.float().abs().sum().item()
    print(f"\nmax_abs_diff: {max_diff}")
    print(f"relative_err: {rel_err:.6f}")
    print(f"elements > 0.01: {(diff > 0.01).sum().item()} / {diff.numel()}")

except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
