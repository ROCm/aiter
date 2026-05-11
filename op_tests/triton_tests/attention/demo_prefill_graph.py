#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Is UA's prefill speedup real or just Python-side launch overhead?

We time `unified_attention_fwd` and `mha_fwd` for prefill shapes two ways:

  (A) per-call torch.cuda.Event -- includes Python wrapper + dispatch +
      kernel launch + kernel time on every iteration.
  (B) HIP graph capture & replay -- the kernel-launch sequence is recorded
      once, then replayed.  Per-iteration cost is graph-launch (~2-5 us)
      plus the kernel.  Python-side wrapper overhead is paid only at
      capture time, not at replay time.

If UA's measured advantage *shrinks* under HIP graphs, the gap was
Python-overhead.  If it stays the same, UA's kernel is genuinely faster.
"""

from __future__ import annotations

import math

import torch

from aiter.ops.mha import mha_fwd
from aiter.ops.unified_attention import unified_attention_fwd

DEVICE = "cuda"
DTYPE = torch.bfloat16
WARMUP, ITERS = 10, 50

CASES = [
    # (label,  num_kv_heads, gqa, hdim, page_size, q_lens, ctx_lens)
    ("prefill h8x8 d64  q=512 ",  1, 8,  64, 64, [512],  [512]),
    ("prefill h8x8 d64  q=1024",  1, 8,  64, 64, [1024], [1024]),
    ("prefill h8x8 d64  q=2048",  1, 8,  64, 64, [2048], [2048]),
    ("prefill h8x8 d64  q=4096",  1, 8,  64, 64, [4096], [4096]),
    ("prefill h8x1 d128 q=512 ",  8, 1, 128, 64, [512],  [512]),
    ("prefill h8x1 d128 q=1024",  8, 1, 128, 64, [1024], [1024]),
    ("prefill h8x1 d128 q=2048",  8, 1, 128, 64, [2048], [2048]),
    ("prefill h8x1 d128 q=4096",  8, 1, 128, 64, [4096], [4096]),
]


def make_inputs(num_kv_heads, gqa, hdim, page_size, q_lens, ctx_lens):
    h_q = num_kv_heads * gqa
    B = len(q_lens)
    total_q = sum(q_lens)
    max_kv = max(ctx_lens)
    max_pages = (max_kv + page_size - 1) // page_size

    torch.manual_seed(0)

    # 4D batched (mha_fwd) -- requires uniform shapes; here all q_lens equal.
    s = q_lens[0]
    q4 = torch.randn(B, s, h_q, hdim, dtype=DTYPE, device=DEVICE)
    k4 = torch.randn(B, max_kv, num_kv_heads, hdim, dtype=DTYPE, device=DEVICE)
    v4 = torch.randn_like(k4)

    # 3D varlen Q (UA)
    q3 = q4.reshape(total_q, h_q, hdim).contiguous()

    # Paged KV (UA)
    num_blocks = B * max_pages + 4
    k_paged = torch.randn(num_blocks, page_size, num_kv_heads, hdim,
                          dtype=DTYPE, device=DEVICE)
    v_paged = torch.randn_like(k_paged)
    block_table = torch.zeros(B, max_pages, dtype=torch.int32, device=DEVICE)
    for i in range(B):
        base = i * max_pages
        n = (ctx_lens[i] + page_size - 1) // page_size
        block_table[i, :n] = torch.arange(base, base + n, dtype=torch.int32,
                                          device=DEVICE)
    cu_q = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()],
                        dtype=torch.int32, device=DEVICE)
    seq_lens = torch.tensor(ctx_lens, dtype=torch.int32, device=DEVICE)

    out_ua = torch.empty_like(q3)
    return dict(q3=q3, q4=q4, k4=k4, v4=v4,
                k_paged=k_paged, v_paged=v_paged,
                block_table=block_table, cu_q=cu_q, seq_lens=seq_lens,
                out_ua=out_ua, scale=1.0 / math.sqrt(hdim),
                hdim=hdim)


def make_callables(t):
    def _ua():
        unified_attention_fwd(
            t["out_ua"], t["q3"], t["k_paged"], t["v_paged"],
            t["block_table"], t["seq_lens"], t["cu_q"],
            mask_type=2, scale_s=t["scale"],
            scale=1.0, scale_k=1.0, scale_v=1.0, scale_out=1.0,
        )

    def _mha():
        mha_fwd(
            t["q4"], t["k4"], t["v4"],
            dropout_p=0.0, softmax_scale=t["scale"],
            is_causal=True, window_size_left=-1, window_size_right=-1,
            sink_size=0,
            return_softmax_lse=False, return_dropout_randval=False,
        )

    return _ua, _mha


def time_event(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(ITERS):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / ITERS


def time_graph(fn) -> float:
    # Warm up (also forces JIT / kernel selection before capture).
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()

    # Capture on a side stream (PyTorch convention for cuda graphs on ROCm).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        # extra warmup on the capture stream
        for _ in range(3):
            fn()
        torch.cuda.current_stream().synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(ITERS):
                fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    g.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / ITERS


print(f"warmup={WARMUP} iters={ITERS} dtype=bf16  (timing in ms/call)")
print(f"{'case':<26s} {'UA evt':>8s} {'mha evt':>8s} {'UA gph':>8s} {'mha gph':>8s} "
      f"{'evt UA/mha':>12s} {'gph UA/mha':>12s}")
print("-" * 90)

for label, num_kv, gqa, hdim, page, qls, kls in CASES:
    t = make_inputs(num_kv, gqa, hdim, page, qls, kls)
    ua, mha = make_callables(t)

    try:
        ua_evt = time_event(ua)
    except Exception as e:  # noqa: BLE001
        ua_evt = float("nan")
        print(f"{label:<26s}  UA event err: {type(e).__name__}: {str(e).splitlines()[0]}")
        continue
    mha_evt = time_event(mha)

    try:
        ua_gph = time_graph(ua)
    except Exception as e:  # noqa: BLE001
        ua_gph = float("nan")
    try:
        mha_gph = time_graph(mha)
    except Exception as e:  # noqa: BLE001
        mha_gph = float("nan")

    evt_ratio = ua_evt / mha_evt
    gph_ratio = ua_gph / mha_gph if mha_gph and not math.isnan(mha_gph) else float("nan")
    print(f"{label:<26s} {ua_evt:8.4f} {mha_evt:8.4f} {ua_gph:8.4f} {mha_gph:8.4f} "
          f"{evt_ratio:12.3f} {gph_ratio:12.3f}")

print()
print("Reading the table:")
print("  evt UA/mha < 1  -> UA looks faster under per-call event timing")
print("  gph UA/mha < 1  -> UA is *actually* faster (Python overhead removed)")
print("  if evt UA/mha << 1 but gph UA/mha ~= 1  -> the speedup was launch overhead.")
