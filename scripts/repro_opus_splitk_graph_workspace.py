#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reproduce/guard Opus splitK graph replay after workspace grow.

The graph captures a small splitK GEMM, then an eager larger GEMM grows the
same stream's workspace. Replaying the old graph must still be correct after
the workspace pointer changes; raw-pointer kernel args would bake the old
workspace address into the graph.
"""

import argparse
from typing import Tuple

import torch

from aiter.ops.opus.gemm_op_a16w16 import (
    opus_gemm_a16w16_tune,
    opus_gemm_workspace_init,
)


def _shape(text: str) -> Tuple[int, int, int]:
    parts = text.lower().replace(",", "x").split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected MxNxK, got {text!r}")
    return tuple(int(p) for p in parts)


def _make_tensors(shape: Tuple[int, int, int], seed: int):
    m, n, k = shape
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    x = torch.randn((1, m, k), device="cuda", dtype=torch.bfloat16, generator=gen)
    w = torch.randn((1, n, k), device="cuda", dtype=torch.bfloat16, generator=gen)
    y = torch.empty((1, m, n), device="cuda", dtype=torch.bfloat16)
    return x, w, y


def _run(x, w, y, kid: int, splitk: int):
    opus_gemm_a16w16_tune(x, w, y, None, kid, splitk)
    return y


def _aligned_workspace_bytes(
    shape: Tuple[int, int, int],
    splitk: int,
    block_m: int,
    block_n: int,
    dtype_bytes: int,
) -> int:
    m, n, _ = shape
    padded_m = ((m + block_m - 1) // block_m) * block_m
    padded_n = ((n + block_n - 1) // block_n) * block_n
    raw = splitk * padded_m * padded_n * dtype_bytes
    align = 4 * 1024 * 1024
    return ((raw + align - 1) // align) * align


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kid", type=int, default=10210)
    parser.add_argument("--splitk", type=int, default=16)
    parser.add_argument("--small", type=_shape, default=_shape("512x128x4096"))
    parser.add_argument("--large", type=_shape, default=_shape("1024x2048x4096"))
    parser.add_argument("--block-m", type=int, default=512)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--workspace-dtype-bytes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260616)
    args = parser.parse_args()

    small_ws = _aligned_workspace_bytes(
        args.small, args.splitk, args.block_m, args.block_n, args.workspace_dtype_bytes
    )
    large_ws = _aligned_workspace_bytes(
        args.large, args.splitk, args.block_m, args.block_n, args.workspace_dtype_bytes
    )
    if large_ws <= small_ws:
        raise SystemExit(
            "large shape does not force workspace grow under the supplied tile parameters"
        )

    torch.cuda.set_device(0)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())

    xs, ws, ys = _make_tensors(args.small, args.seed)
    xg, wg, yg = _make_tensors(args.large, args.seed + 1)

    with torch.cuda.stream(side):
        opus_gemm_workspace_init()

        _run(xs, ws, ys, args.kid, args.splitk)
        side.synchronize()
        golden = ys.detach().clone()
        side.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            _run(xs, ws, ys, args.kid, args.splitk)

        _run(xg, wg, yg, args.kid, args.splitk)
        side.synchronize()

        ys.zero_()
        graph.replay()
        side.synchronize()

    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    max_delta = (ys.float() - golden.float()).abs().max().item()
    exact = torch.equal(ys, golden)

    print(f"expected_small_workspace_bytes={small_ws}")
    print(f"expected_large_workspace_bytes={large_ws}")
    print("workspace_grow_forced=True")
    print(f"graph_replay_exact={exact} max_delta={max_delta}")

    if not exact:
        raise SystemExit("graph replay output changed after workspace grow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
