"""Quick benchmark for FP8 PER_TOKEN_HEAD batch_prefill on MI308X.

Compares against:
  - bf16 baseline (same shapes, no quant)
  - kv_blockscale FP8 (existing path)
"""

import torch
from op_tests.test_batch_prefill import (
    run_batch_prefill_per_token_head,
    run_batch_prefill_kv_blockscale,
)


# (batch, qo_len, kv_len, nhq, nhk, head_dim, causal, soft_cap)
SHAPES = [
    (1, 1024, 1024, 32, 8, 128, False, 0.0),
    (1, 2048, 2048, 32, 8, 128, False, 0.0),
    (1, 4096, 4096, 32, 8, 128, False, 0.0),
    (4, 1024, 4096, 32, 8, 128, False, 0.0),
    (1, 1024, 1024, 32, 8, 128, True, 30.0),
    (1, 2048, 2048, 32, 8, 128, True, 30.0),
    (1, 4096, 4096, 32, 8, 128, True, 30.0),
    (4, 1024, 4096, 32, 8, 128, True, 30.0),
    (1, 1024, 1024, 16, 16, 128, False, 0.0),
]


def fmt(r):
    if r.get("status") != "passed":
        return "skip"
    return f"{r['time_us']:8.2f} us  {r['tflops']:7.2f} TFLOPS"


print(
    f"{'batch':>5} {'qo':>5} {'kv':>5} {'nhq':>4} {'nhk':>4} {'hd':>4} {'causal':>6} "
    f"{'soft':>5} | {'PER_TOKEN_HEAD':>30} | {'KV_BLOCKSCALE':>30}"
)
print("-" * 130)

for b, qo, kv, nhq, nhk, hd, c, sc in SHAPES:
    common = dict(
        kvcache_layout="linear",
        table_layout="sglang",
        batch_size=b,
        qo_len=qo,
        kv_len=kv,
        page_size=1024,
        num_qo_heads=nhq,
        num_kv_heads=nhk,
        head_dim=hd,
        causal=c,
        logits_soft_cap=sc,
        dtype=torch.bfloat16,
        contiguous_kv=True,
        seed=42,
        profile=True,
    )
    pth = run_batch_prefill_per_token_head(**common)
    kvb = run_batch_prefill_kv_blockscale(**common)
    print(
        f"{b:>5} {qo:>5} {kv:>5} {nhq:>4} {nhk:>4} {hd:>4} {str(c):>6} {sc:>5.1f} | "
        f"{fmt(pth):>30} | {fmt(kvb):>30}"
    )
