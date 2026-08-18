# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import time

import torch

from aiter.ops.flydsl.kernels.megamoe_tile.kernels import build_copy_put_signal_module
from aiter.ops.flydsl.kernels.megamoe_tile.markers import profiler_pause, profiler_resume, roctx_range


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bytes", type=int, default=64 * 1024)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--profile-once", action="store_true")
    args = parser.parse_args()

    torch.cuda.set_device(0)
    src = torch.arange(args.bytes, dtype=torch.int32, device="cuda").remainder(251).to(torch.uint8)
    dst = torch.zeros_like(src)
    signal = torch.zeros(1, dtype=torch.int64, device="cuda")
    launch = build_copy_put_signal_module()
    stream = torch.cuda.current_stream()

    profiler_pause()
    for i in range(20):
        launch(src, dst, signal, args.bytes, i + 1, stream=stream)
    torch.cuda.synchronize()

    iters = 1 if args.profile_once else args.iters
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    profiler_resume()
    with roctx_range("MEGAMOE_TILE/rdma_copy_stub"):
        start.record()
        for i in range(iters):
            launch(src, dst, signal, args.bytes, 100 + i, stream=stream)
        end.record()
        torch.cuda.synchronize()
    profiler_pause()

    elapsed_ms = start.elapsed_time(end)
    assert torch.equal(src, dst)
    print(
        f"COPY_STUB bytes={args.bytes} iters={iters} "
        f"total_ms={elapsed_ms:.6f} us_per_put={elapsed_ms * 1000 / iters:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
