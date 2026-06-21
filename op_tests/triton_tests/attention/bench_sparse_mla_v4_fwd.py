"""
Benchmark `sparse_mla_fwd_v4` with real DeepSeek V4 paper configs.

Layer types (from ATOM #664 / vLLM blog):
  - CSA Flash:  SK = 128 + min(512,  T/4)   H=64
  - CSA Pro:    SK = 128 + min(1024, T/4)   H=128
  - HCA:        SK = 128 + T/128            (both variants)
  - SWA-only:   SK = 128                    (first 2 layers + MTP head)

Compute ranking per paper: HCA >> CSA Pro > CSA Flash >> SWA-only.

For each (variant, layer type, T): runs 50 iterations after 10 warmup,
reports wall-time ms and TFLOPS. Compares V4(sink) vs V3.2 path (sink=None)
to confirm sink overhead is negligible.
"""
import math
import os
import sys
sys.path.insert(0, "/workspace/aiter_main_7890e4b")

import torch

from aiter.ops.triton._triton_kernels.attention.deepseek_sparse_attention import (
    sparse_mla_fwd as sparse_mla_fwd_v3,
)
from aiter.ops.triton._triton_kernels.attention.deepseek_sparse_attention_v4 import (
    sparse_mla_fwd_v4,
)


D_V = 512
D_ROPE = 64
D_QK = D_V + D_ROPE


def topk_for_layer(layer: str, variant: str, T: int) -> int:
    if layer == "csa":
        sparse_topk = 512 if variant == "flash" else 1024
        return 128 + min(sparse_topk, T // 4)
    if layer == "hca":
        return 128 + max(1, T // 128)
    if layer == "swa-only":
        return 128
    raise ValueError(f"unknown layer {layer}")


def num_heads(variant: str) -> int:
    return 64 if variant == "flash" else 128


def flops(T: int, H: int, TOPK: int) -> float:
    # QK^T:  2 * T * H * TOPK * D_QK
    # P @ V: 2 * T * H * TOPK * D_V
    return float(2 * T * H * TOPK * (D_QK + D_V))


def bench_one(label, T, H, TOPK, use_sink, warmup=10, steps=50):
    torch.manual_seed(0)
    total = T
    q = torch.randn(total, H, D_QK, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(total, 1, D_QK, dtype=torch.bfloat16, device="cuda")
    topk_indices = torch.randint(0, total, (total, TOPK), dtype=torch.int32, device="cuda")
    attn_sink = torch.randn(H, dtype=torch.float32, device="cuda") if use_sink else None

    def call():
        if use_sink:
            return sparse_mla_fwd_v4(q, kv, topk_indices, attn_sink=attn_sink, kv_lora_rank=D_V)
        else:
            return sparse_mla_fwd_v4(q, kv, topk_indices, attn_sink=None, kv_lora_rank=D_V)

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        call()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / steps
    tflops = flops(T, H, TOPK) / ms / 1e9
    sink_tag = "sink" if use_sink else "no-sink"
    print(f"  {label:38s}  TOPK={TOPK:4d}  {sink_tag:8s}  {ms:7.3f} ms  {tflops:6.1f} TFLOPS")
    return ms, tflops


def main():
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Arch:   {torch.cuda.get_device_properties(0).gcnArchName}")
    print()

    # Context lengths to sweep
    SEQ_LENS = [4096, 8192, 16384, 32768]

    for variant in ["flash", "pro"]:
        H = num_heads(variant)
        print("=" * 90)
        print(f"V4-{variant.upper()}  (H={H}, D_V={D_V}, D_ROPE={D_ROPE})")
        print("=" * 90)

        for T in SEQ_LENS:
            print(f"\n  T = {T}")
            for layer in ["csa", "hca", "swa-only"]:
                TOPK = topk_for_layer(layer, variant, T)
                label = f"{layer:8s}"
                # Compare with vs without sink to gauge epilogue overhead
                ms_ns, tf_ns = bench_one(label, T, H, TOPK, use_sink=False)
                ms_s,  tf_s  = bench_one(label, T, H, TOPK, use_sink=True)
                overhead_pct = (ms_s / ms_ns - 1) * 100
                print(f"    sink overhead: {overhead_pct:+.2f}%")
        print()


if __name__ == "__main__":
    main()
