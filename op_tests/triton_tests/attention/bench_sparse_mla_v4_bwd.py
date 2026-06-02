"""
Benchmark `sparse_mla_bwd_v4` (chunked_gather + sink) with real DeepSeek V4 configs.

Same layer matrix as bench_sparse_mla_v4_fwd.py:
  - CSA Flash:  SK = 128 + min(512,  T/4)   H=64
  - CSA Pro:    SK = 128 + min(1024, T/4)   H=128
  - HCA:        SK = 128 + T/128            (both variants)
  - SWA-only:   SK = 128

Reports:
  - Bwd wall-time ms for each configuration
  - V4 with sink vs V4 without sink (verifies sink reduction is cheap)
  - Bwd/Fwd ratio (rule of thumb: ~2.5x for FlashAttention-style)
"""
import sys
sys.path.insert(0, "/workspace/aiter_main_7890e4b")

import torch

from aiter.ops.triton._triton_kernels.attention.deepseek_sparse_attention_v4 import (
    sparse_mla_fwd_v4, sparse_mla_bwd_v4,
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


def bench_one(label, T, H, TOPK, use_sink, warmup=5, steps=20):
    torch.manual_seed(0)
    total = T
    q = torch.randn(total, H, D_QK, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(total, 1, D_QK, dtype=torch.bfloat16, device="cuda")
    topk_indices = torch.randint(0, total, (total, TOPK), dtype=torch.int32, device="cuda")
    do = torch.randn(total, H, D_V, dtype=torch.bfloat16, device="cuda")
    attn_sink = torch.randn(H, dtype=torch.float32, device="cuda") if use_sink else None

    o, lse = sparse_mla_fwd_v4(q, kv, topk_indices, attn_sink=attn_sink, kv_lora_rank=D_V)

    def fwd_call():
        return sparse_mla_fwd_v4(q, kv, topk_indices, attn_sink=attn_sink, kv_lora_rank=D_V)

    def bwd_call():
        return sparse_mla_bwd_v4(q, kv, o, do, topk_indices, lse,
                                  attn_sink=attn_sink, kv_lora_rank=D_V)

    # Warmup
    for _ in range(warmup):
        fwd_call()
        bwd_call()
    torch.cuda.synchronize()

    def timeit(fn, n=steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(n):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / n

    fwd_ms = timeit(fwd_call)
    bwd_ms = timeit(bwd_call)
    ratio = bwd_ms / fwd_ms if fwd_ms > 0 else 0
    sink_tag = "sink" if use_sink else "no-sink"
    print(f"  {label:38s} TOPK={TOPK:4d}  {sink_tag:8s}  "
          f"fwd {fwd_ms:7.3f} ms  bwd {bwd_ms:8.3f} ms  bwd/fwd {ratio:4.2f}x")
    return fwd_ms, bwd_ms


def main():
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Arch:   {torch.cuda.get_device_properties(0).gcnArchName}")
    print()

    SEQ_LENS = [4096, 8192, 16384]   # 32K bwd is slow; do later if needed

    for variant in ["flash", "pro"]:
        H = num_heads(variant)
        print("=" * 100)
        print(f"V4-{variant.upper()}  (H={H}, D_V={D_V}, D_ROPE={D_ROPE})")
        print("=" * 100)

        for T in SEQ_LENS:
            print(f"\n  T = {T}")
            for layer in ["csa", "hca", "swa-only"]:
                TOPK = topk_for_layer(layer, variant, T)
                bench_one(layer, T, H, TOPK, use_sink=False)
                bench_one(layer, T, H, TOPK, use_sink=True)
        print()


if __name__ == "__main__":
    main()
