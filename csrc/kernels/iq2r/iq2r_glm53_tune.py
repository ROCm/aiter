# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tune GLM-5.3 IQ2R MoE launch choices per token count.

Reads shapes from ``aiter/configs/iq2r_glm53_untuned.csv``, times every
candidate (gate kernel and grid, down kernel and grid, prefill down chunks)
on synthetic packed weights, and writes the fastest matching choice to the
tuned CSV consumed by :func:`aiter.iq2r_glm53.iq2r_glm53_config`.

    python3 csrc/kernels/iq2r/iq2r_glm53_tune.py \\
        -i aiter/configs/iq2r_glm53_untuned.csv \\
        -o aiter/configs/iq2r_glm53_tuned.csv
"""

from __future__ import annotations

import argparse
import itertools

import pandas as pd
import torch

from aiter.iq2r_glm53 import (
    EXPERTS,
    HIDDEN,
    MAX_DECODE_TOKENS,
    TOPK,
    IQ2RGlm53Config,
    IQ2RGlm53Workspace,
    iq2r_glm53_moe_out,
    iq2r_glm53_pack,
)
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.iq2r_format import iq2r_packed_sizes

KEYS = ["gfx", "cu_num", "token", "model_dim", "inter_dim", "expert", "topk"]
RESULTS = ["gate_kernel", "gate_grid", "down_kernel", "down_grid", "down_chunks", "us"]


def synthetic_weights(intermediate_size: int, device: str):
    """Random IQ2R records with finite FP8 codebooks and unit base exponents."""

    def random_bytes(columns: int):
        return torch.randint(
            0, 256, (EXPERTS, columns), dtype=torch.uint8, device=device
        )

    def matrix(n: int, k: int):
        data_bytes, aux_bytes = iq2r_packed_sizes(n, k)
        data = random_bytes(data_bytes)
        aux = torch.full((EXPERTS, aux_bytes), 127, dtype=torch.uint8, device=device)
        # E4M3 magnitudes <= 1.0 keep the synthetic output finite.
        book = random_bytes(4096)
        aux[:, :4096] = (book & 0x80) | (book & 0x37)
        aux[:, :8] = 0
        return data, aux

    gate, gate_aux = matrix(2 * intermediate_size, HIDDEN)
    down, down_aux = matrix(HIDDEN, intermediate_size)
    return iq2r_glm53_pack(
        gate, gate_aux, down, down_aux, intermediate_size=intermediate_size
    )


def candidates(tokens: int, intermediate_size: int):
    tp8 = intermediate_size == 256
    if tokens == 1:
        yield IQ2RGlm53Config("decode", 2, "route9", 4, 1)
        return
    if tokens <= MAX_DECODE_TOKENS:
        gates = ("decode", "nobarrier") if tp8 else ("decode",)
        downs = [("route9", 4)] if tokens in (2, 4) else []
        down_kernels = ("packed", "ordered") if tp8 else ("packed",)
        downs += list(itertools.product(down_kernels, (2, 4, 8)))
        for gate, grid in itertools.product(gates, (1, 2, 4)):
            for down, down_grid in downs:
                yield IQ2RGlm53Config(gate, grid, down, down_grid, 1)
    if tokens > 16:
        for grid, chunks in itertools.product((1, 2, 4), (1, 2)):
            yield IQ2RGlm53Config("prefill", grid, "prefill", 0, chunks)


def time_us(fn, warmup: int, iters: int) -> float:
    """Replay time of a CUDA graph of ``fn``: serving captures decode batches,
    so host launch overhead (several launches on the prefill path) is excluded."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            fn()
    for _ in range(max(warmup // 10, 1)):
        graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 100.0 / iters


def tune_shape(tokens: int, intermediate_size: int, weights, args) -> dict:
    device = "cuda"
    hidden = torch.randn((tokens, HIDDEN), dtype=torch.bfloat16, device=device)
    routed = torch.rand((tokens, EXPERTS - 1), device=device).topk(TOPK - 1).indices
    shared = torch.full((tokens, 1), EXPERTS - 1, dtype=routed.dtype, device=device)
    ids = torch.cat((routed, shared), 1).to(torch.int32).contiguous()
    weights_topk = torch.rand((tokens, TOPK), dtype=torch.float32, device=device)
    workspace = IQ2RGlm53Workspace.allocate(tokens, intermediate_size, device=device)

    def run(config, out):
        iq2r_glm53_moe_out(hidden, *weights, weights_topk, ids, out, workspace, config)

    options = list(candidates(tokens, intermediate_size))
    reference = torch.empty_like(hidden)
    run(options[0], reference)
    tolerance = 1e-2 * reference.float().abs().max().item()
    best = None
    for config in options:
        out = torch.empty_like(hidden)
        try:
            run(config, out)
            torch.cuda.synchronize()
        except RuntimeError as error:
            print(f"M={tokens} I={intermediate_size} {config}: skipped ({error})")
            continue
        # Down variants may sum the nine routes in a different order.
        if not torch.allclose(out, reference, rtol=1e-2, atol=tolerance):
            print(f"M={tokens} I={intermediate_size} {config}: mismatch, skipped")
            continue
        us = time_us(lambda: run(config, out), args.warmup, args.iters)
        print(f"M={tokens} I={intermediate_size} {config}: {us:.2f} us")
        if best is None or us < best[1]:
            best = (config, us)
    config, us = best
    return {
        "gate_kernel": config.gate_kernel,
        "gate_grid": config.gate_grid,
        "down_kernel": config.down_kernel,
        "down_grid": config.down_grid,
        "down_chunks": config.down_chunks,
        "us": round(us, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-i", "--untune_file", default="aiter/configs/iq2r_glm53_untuned.csv"
    )
    parser.add_argument(
        "-o", "--tune_file", default="aiter/configs/iq2r_glm53_tuned.csv"
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(0)
    gfx, cu_num = get_gfx_runtime(), get_cu_num()
    shapes = pd.read_csv(args.untune_file)
    shapes = shapes[(shapes.gfx == gfx) & (shapes.cu_num == cu_num)]
    unsupported = (
        (shapes.model_dim != HIDDEN)
        | (shapes.expert != EXPERTS)
        | (shapes.topk != TOPK)
    )
    if unsupported.any():
        raise ValueError("the GLM-5.3 tuner covers hidden 6144, 257 experts, top-9")

    rows = []
    for intermediate_size, group in shapes.groupby("inter_dim", sort=False):
        weights = synthetic_weights(int(intermediate_size), "cuda")
        for shape in group.itertuples(index=False):
            tokens = int(shape.token)
            result = tune_shape(tokens, int(intermediate_size), weights, args)
            rows.append({**shape._asdict(), **result})
        del weights
        torch.cuda.empty_cache()

    tuned = pd.DataFrame(rows, columns=KEYS + RESULTS)
    try:
        previous = pd.read_csv(args.tune_file)
        tuned = pd.concat([previous, tuned]).drop_duplicates(KEYS, keep="last")
    except FileNotFoundError:
        pass
    tuned.to_csv(args.tune_file, index=False)
    print(f"wrote {len(tuned)} rows to {args.tune_file}")


if __name__ == "__main__":
    main()
