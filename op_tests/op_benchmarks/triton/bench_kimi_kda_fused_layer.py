# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Latency benchmark for the fused Kimi-K3 KDA whole-layer ops (gfx950, TP8).

Times the decode op across the real cuda-graph batch tiers (dispatched to the
m64 / m256 schedules) and the prefill op across a few (tokens, sequences)
shapes. Requires a gfx950 device with Triton >= 3.8.

    python op_tests/op_benchmarks/triton/bench_kimi_kda_fused_layer.py
"""

import torch

from aiter import logger
from aiter.ops.triton.kimi_delta_attn import (
    kimi_kda_fused_layer_decode,
    kimi_kda_fused_layer_prefill,
)

BF16, F32, I32 = torch.bfloat16, torch.float32, torch.int32
DEV = "cuda"


def _rnd(shape, dtype):
    if dtype == I32:
        return torch.zeros(shape, dtype=I32, device=DEV)
    return (torch.randn(shape, device=DEV, dtype=torch.float32) * 0.05).to(dtype)


def _time(fn, *, warmup=5, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3  # microseconds


def _decode_call(m):
    slots = max(m, 64)
    x = _rnd((m, 7168), BF16)
    args = (
        x,
        _rnd((6144, 7168), BF16),
        _rnd((144, 7168), BF16),
        _rnd((7168, 1536), BF16),
        _rnd((1536, 128), BF16),
        _rnd((4608, 4), F32),
        _rnd((12,), F32),
        _rnd((1536,), F32),
        _rnd((128,), BF16),
        _rnd((slots, 3, 4608), BF16),
        _rnd((slots, 12, 128, 128), F32),
        torch.arange(m, dtype=I32, device=DEV),
    )
    out = torch.empty((m, 7168), dtype=BF16, device=DEV)
    return lambda: kimi_kda_fused_layer_decode(
        *args, lower_bound=-5.0, norm_eps=1e-5, output_tensor=out
    )


def _prefill_call(m, n):
    slots = max(n, 4)
    base = m // n
    lens = [base] * n
    lens[-1] += m - base * n
    cu = torch.tensor(
        [0, *torch.cumsum(torch.tensor(lens), 0).tolist()], dtype=I32, device=DEV
    )
    args = (
        _rnd((m, 4608), BF16),
        _rnd((m, 1536), BF16),
        _rnd((m, 128), BF16),
        _rnd((m, 12), BF16),
        _rnd((1536, 128), BF16),
        _rnd((4608, 4), F32),
        _rnd((12,), F32),
        _rnd((1536,), F32),
        _rnd((128,), BF16),
        _rnd((slots, 3, 4608), BF16),
        _rnd((slots, 12, 128, 128), F32),
        torch.arange(n, dtype=I32, device=DEV),
        cu,
        torch.zeros(n, dtype=torch.bool, device=DEV),
    )
    return lambda: kimi_kda_fused_layer_prefill(*args, lower_bound=-5.0, norm_eps=1e-5)


def main():
    logger.info("%-10s%-18s%10s", "op", "shape", "us/call")
    for m in (1, 8, 32, 64, 96, 128, 192, 256):
        shape = f"m={m}"
        logger.info("%-10s%-18s%10.1f", "decode", shape, _time(_decode_call(m)))
    for m, n in ((1024, 1), (2048, 4), (4096, 8), (8192, 32)):
        shape = f"m={m} n={n}"
        logger.info("%-10s%-18s%10.1f", "prefill", shape, _time(_prefill_call(m, n)))


if __name__ == "__main__":
    main()
