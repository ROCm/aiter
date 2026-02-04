#!/usr/bin/env python3
"""
Minimal test to verify moe_smoothquant_fwd and smooth_per_token_scaled_quant equivalence.
"""

import torch
import aiter


def test_conversion(m=128, n=4096, topk=3, experts=8, dtype=torch.bfloat16):
    """Test that both functions produce identical results."""

    print(f"Testing: m={m}, n={n}, topk={topk}, experts={experts}, dtype={dtype}")

    # Setup inputs
    device = "cuda"
    input = torch.randn((m, n), dtype=dtype, device=device)
    x_scale = torch.randn((experts, n), dtype=torch.float32, device=device)
    topk_ids = torch.randint(0, experts, (m, topk), dtype=torch.int32, device=device)

    # Method 1: Original moe_smoothquant_fwd
    output1 = torch.empty((m, topk, n), dtype=torch.int8, device=device)
    y_scale1 = torch.empty((m, topk, 1), dtype=torch.float32, device=device)
    aiter.moe_smoothquant_fwd(output1, input, x_scale, topk_ids, y_scale1)

    # Method 2: Replacement with smooth_per_token_scaled_quant
    output2_flat = torch.empty((m * topk, n), dtype=torch.int8, device=device)
    y_scale2_flat = torch.empty((m * topk), dtype=torch.float32, device=device)

    aiter.smooth_per_token_scaled_quant(
        output2_flat.view(topk, m, n).transpose(0, 1),  # [m, topk, n]
        input.view(m, 1, n).expand(-1, topk, -1),  # [m, topk, n]
        y_scale2_flat,  # [m*topk] -> kernel reshapes
        x_scale,  # [experts, n]
        topk_ids,  # [m, topk]
        enable_ps=True,
    )

    output2 = output2_flat.view(m, topk, n)
    y_scale2 = y_scale2_flat.view(m, topk, 1)

    # Compare outputs
    output_match = torch.allclose(output1.float(), output2.float(), rtol=0.01, atol=1.0)
    scale_match = torch.allclose(y_scale1, y_scale2, rtol=1e-3, atol=1e-3)

    if output_match and scale_match:
        print("? PASS: Outputs match!")
        print(
            f"  Output max diff: {(output1.float() - output2.float()).abs().max().item():.6f}"
        )
        print(f"  Scale max diff: {(y_scale1 - y_scale2).abs().max().item():.6f}")
        return True
    else:
        print("? FAIL: Outputs do not match!")
        print(f"  Output match: {output_match}")
        print(f"  Scale match: {scale_match}")
        if not output_match:
            print(
                f"  Output max diff: {(output1.float() - output2.float()).abs().max().item():.6f}"
            )
        if not scale_match:
            print(f"  Scale max diff: {(y_scale1 - y_scale2).abs().max().item():.6f}")
        return False


if __name__ == "__main__":
    # Test various configurations
    configs = [
        (16, 4096, 2, 8, torch.bfloat16),
        (128, 4096, 3, 16, torch.bfloat16),
        (256, 5120, 4, 128, torch.bfloat16),
        (64, 3072, 6, 64, torch.float16),
        (1024, 4096, 8, 256, torch.bfloat16),
    ]

    results = []
    for m, n, topk, experts, dtype in configs:
        passed = test_conversion(m, n, topk, experts, dtype)
        results.append(passed)
        print()

    print("=" * 60)
    print(f"Summary: {sum(results)}/{len(results)} tests passed")
    if all(results):
        print("? All tests PASSED")
        exit(0)
    else:
        print("? Some tests FAILED")
        exit(1)
