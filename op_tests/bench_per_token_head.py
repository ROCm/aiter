"""Quick benchmark for FP8 PER_TOKEN_HEAD batch_prefill on MI308X.

Compares against:
  - bf16 baseline (same shapes, no quant)
  - kv_blockscale FP8 (existing path)
"""

import os
import sys

import torch

# Allow running as `python op_tests/bench_per_token_head.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from op_tests.test_batch_prefill import (
    run_batch_prefill_per_token_head,
    run_batch_prefill_kv_blockscale,
)


# (batch, qo_len, kv_len, nhq, nhk, head_dim, causal, soft_cap)
#
# H20 reference config (from ticket "H20 FP8+MTP Attention Kernel_perf.pdf"):
#   q_heads=8, kv_heads=1 (GQA ratio=8), head_dim=128, block_size=64,
#   quant_type=QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD,
#   q_len = kv_len, full causal prefill, batch (num_seqs) in {2, 4, 6, 8}.
#
# NOTE: H20 uses block_size=64; our CK pipeline currently supports page_size=1024.
# Comparison is not strictly apples-to-apples but reflects end-to-end kernel perf.
#
# H20 TFLOPS reference table (rows = seq_len, cols = batch):
#                  batch=2   batch=4   batch=6   batch=8
#   seq_len=1024     146.2    169.4    186.7    188.7
#   seq_len=16384    266.5    269.8    270.1    268.0
#   seq_len=32768    269.6    272.3    272.5    270.3
#   seq_len=65536    270.8    273.6    273.6    271.5
#   seq_len=131072   272.8    274.2    272.1    272.1
H20_REF = {
    (2,   1024): 146.2, (4,   1024): 169.4, (6,   1024): 186.7, (8,   1024): 188.7,
    (2,  16384): 266.5, (4,  16384): 269.8, (6,  16384): 270.1, (8,  16384): 268.0,
    (2,  32768): 269.6, (4,  32768): 272.3, (6,  32768): 272.5, (8,  32768): 270.3,
    (2,  65536): 270.8, (4,  65536): 273.6, (6,  65536): 273.6, (8,  65536): 271.5,
    (2, 131072): 272.8, (4, 131072): 274.2, (6, 131072): 272.1, (8, 131072): 272.1,
}

SHAPES = [
    (b, s, s, 8, 1, 128, True, 0.0)
    for s in (1024, 16384, 32768, 65536, 131072)
    for b in (2, 4, 6, 8)
]


def fmt(r):
    if r.get("status") != "passed":
        return "skip"
    return f"{r['time_us']:8.2f} us  {r['tflops']:7.2f} TFLOPS"


print(
    f"{'batch':>5} {'seq':>6} {'nhq':>4} {'nhk':>4} {'hd':>4} | "
    f"{'PER_TOKEN_HEAD':>26} | {'KV_BLOCKSCALE':>26} | "
    f"{'H20 ref':>9} | {'MI300/H20':>9}"
)
print("-" * 120)

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
        skip_reference=True,
    )
    pth = run_batch_prefill_per_token_head(**common)
    kvb = run_batch_prefill_kv_blockscale(**common)
    h20 = H20_REF.get((b, kv))
    if pth.get("status") == "passed" and h20:
        ratio = f"{pth['tflops'] / h20:>8.2f}x"
    else:
        ratio = "  -"
    h20_str = f"{h20:>9.1f}" if h20 else f"{'-':>9}"
    print(
        f"{b:>5} {kv:>6} {nhq:>4} {nhk:>4} {hd:>4} | "
        f"{fmt(pth):>26} | {fmt(kvb):>26} | "
        f"{h20_str} | {ratio:>9}"
    )
