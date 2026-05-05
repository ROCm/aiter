#!/usr/bin/env python
"""
End-to-end smoke test for the SplitK zero-init fusion mechanism on the
bpreshuffle CKTile FP8 a8w8 blockscale GEMM path.

Compares three configurations (all CKTile, bpreshuffle layout):
  config1: splitK=0, no fusion, kernel does its own Y allocation.
  config2: splitK>0, kernel does Y.zero_() before atomic_add (default).
  config3: splitK>0, producer per_group_quant_hip pre-zeroes Y, GEMM
           skips Y.zero_() via y_is_zeroed=True.

All three must produce bit-similar outputs (modulo bf16 atomic-add ordering
in SplitK, which adds at most ~1 LSB of noise).
"""
import torch
import aiter
from aiter import dtypes
from aiter.ops.quant import per_group_quant_hip
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle_cktile
from aiter.ops.shuffle import shuffle_weight


def shuffle_weights(w):
    return shuffle_weight(w, layout=(16, 16))


def run_gemm(x_bf16, w_q, w_scale, splitK, fused_zero_init=False, dtype=torch.bfloat16):
    M, K = x_bf16.shape
    N = w_q.shape[0]

    Y = torch.empty(M, N, dtype=dtype, device=x_bf16.device)
    if fused_zero_init:
        zero_init = Y
    else:
        zero_init = None

    x_q, x_scale = per_group_quant_hip(
        x_bf16,
        quant_dtype=dtypes.fp8,
        group_size=128,
        transpose_scale=True,
        gemm_out_zero_init=zero_init,
    )

    gemm_a8w8_blockscale_bpreshuffle_cktile(
        x_q, w_q, x_scale, w_scale, Y,
        True,             # preshuffleB
        splitK,           # splitK
        fused_zero_init,  # y_is_zeroed
    )
    torch.cuda.synchronize()
    return Y


def main():
    torch.manual_seed(0)
    device = "cuda"

    test_shapes = [
        (8, 5120, 2048),
        (8, 2048, 4096),
        (8, 1024, 2048),
    ]
    for M, N, K in test_shapes:
        print(f"\n=== M={M}, N={N}, K={K} ===")
        x_bf16 = torch.randn(M, K, device=device, dtype=torch.bfloat16) * 0.1

        # FP8 weight
        w_bf16 = torch.randn(N, K, device=device, dtype=torch.bfloat16) * 0.1
        w_q = w_bf16.to(dtypes.fp8)
        w_q_shuffled = shuffle_weights(w_q.clone())
        w_scale = torch.ones(
            (N + 127) // 128, (K + 127) // 128, device=device, dtype=torch.float32
        )

        Y_c1 = run_gemm(x_bf16, w_q_shuffled, w_scale, splitK=0)
        Y_c2 = run_gemm(x_bf16, w_q_shuffled, w_scale, splitK=2, fused_zero_init=False)
        Y_c3 = run_gemm(x_bf16, w_q_shuffled, w_scale, splitK=2, fused_zero_init=True)

        d12 = (Y_c1 - Y_c2).abs().max().item()
        d23 = (Y_c2 - Y_c3).abs().max().item()
        ymax = Y_c1.abs().max().item()
        print(f"  config1 vs config2 max diff: {d12:.4f} (rel {d12/(ymax+1e-9):.2e})")
        print(f"  config2 vs config3 max diff: {d23:.4f} (rel {d23/(ymax+1e-9):.2e})")
        print(f"  Y_c1 sample: {Y_c1[0, :4].tolist()}")
        print(f"  Y_c3 sample: {Y_c3[0, :4].tolist()}")
        ok12 = d12 / (ymax + 1e-9) < 0.05
        ok23 = d23 / (ymax + 1e-9) < 0.01
        print("  config1<->config2:", "PASS" if ok12 else "FAIL")
        print("  config2<->config3:", "PASS" if ok23 else "FAIL")


if __name__ == "__main__":
    main()
