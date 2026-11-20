# SPDX-License-Identifier: MIT
"""Measure the fused routing wrapper with graph replay on gfx1250."""

from functools import partial

import torch
from triton.testing import do_bench_cudagraph

from aiter import logger
from aiter.ops.triton.moe.fused_topk_dispatch import fused_topk_dispatch


def main():
    for n in (1, 4, 128, 512, 2048):
        logits = torch.randn((n, 256), device="cuda", dtype=torch.bfloat16)
        weights = torch.empty((n, 8), device="cuda", dtype=torch.float32)
        ids = torch.empty((n, 8), device="cuda", dtype=torch.int32)
        cu = torch.arange(n + 1, device="cuda", dtype=torch.int32)
        requests = torch.empty((), device="cuda", dtype=torch.int64)
        pad = torch.empty(n, device="cuda", dtype=torch.bool)
        recv = torch.empty((), device="cuda", dtype=torch.int32)
        ms = do_bench_cudagraph(
            partial(
                fused_topk_dispatch,
                logits,
                weights,
                ids,
                cu,
                requests,
                pad,
                recv,
                batch_size=n,
                is_draft=False,
                write_requests=True,
            )
        )
        logger.info("tokens=%d latency_us=%.3f", n, ms * 1000)


if __name__ == "__main__":
    main()
