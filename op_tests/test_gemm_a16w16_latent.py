# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""BF16 latent-projection GEMMs: caller buffers and changed graph inputs.

The defaults exercise the three retuned cells and padded-M boundary consumers,
without bias, scaling, or preshuffled weights. Torch FP32 linear is an untimed reference.
"""

import argparse
import itertools
import statistics

import pandas as pd
import torch
import torch.nn.functional as F

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.tuned_gemm import gemm_a16w16, gemm_a16w16_out

# Exact keys plus representative first/last ragged-M consumers of those keys.
# The lookup tries exact M, fine padding, then coarse padding; the CSV does
# not specialize dispatch to the exact three M values alone.
SHAPES = (
    (128, 7168, 3584),
    (16384, 3584, 7168),
    (16384, 7168, 3584),
    (65, 7168, 3584),
    (127, 7168, 3584),
    (8193, 3584, 7168),
    (16383, 3584, 7168),
    (16257, 7168, 3584),
    (16383, 7168, 3584),
)


def run_torch(x, weight):
    return F.linear(x.float(), weight.float()).to(x.dtype)


def assert_result(expected, actual, label):
    assert expected.shape == actual.shape and expected.dtype == actual.dtype
    error = checkAllclose(
        expected.float(),
        actual.float(),
        rtol=1e-2,
        atol=1e-2,
        tol_err_ratio=0,
        msg=label,
    )
    assert error == 0, (label, error)
    return error


@benchmark()
def test_latent_gemm(m, n, k, seed):
    torch.manual_seed(seed)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    # Exact trailing capacities for inputs; nonzero output halos expose overwrites.
    storage = torch.full((m * n + 32,), 19.0, device=x.device, dtype=x.dtype)
    out = storage[16:-16].view(m, n)
    expected = run_torch(x, weight)

    def into():
        assert gemm_a16w16_out(x, weight, out) is None
        return out

    candidates = {"functional": lambda: gemm_a16w16(x, weight), "out": into}
    for name, fn in candidates.items():
        result = fn()
        assert_result(expected, result, name)

    graphs = []
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    others = [torch.empty_like(out) for _ in streams]
    for stream, other in zip(streams, others):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            gemm_a16w16_out(x, weight, other)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                gemm_a16w16_out(x, weight, other)
        stream.synchronize()
        graphs.append(graph)

    for pattern in ("small", "dyadic"):
        if pattern == "small":
            x.mul_(0.125)
            weight.mul_(0.125)
        else:
            x.copy_(torch.randint(-4, 5, x.shape, device=x.device).float() / 16)
            weight.copy_(
                torch.randint(-4, 5, weight.shape, device=x.device).float() / 16
            )
        before = (x.clone(), weight.clone())
        expected = run_torch(x, weight)
        for stream, graph in zip(streams, graphs):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph.replay()
                graph.replay()
        for stream, other in zip(streams, others):
            stream.synchronize()
            assert_result(expected, other, pattern)
        assert torch.equal(x, before[0]) and torch.equal(weight, before[1])
        into()
        assert_result(expected, out, "out " + pattern)
        assert (storage[:16] == 19).all().item()
        assert (storage[-16:] == 19).all().item()

    # Restore a random distribution before timing. Include allocation for the
    # functional call; the out= variant reuses the supplied output.
    x.normal_()
    weight.normal_()
    expected = run_torch(x, weight)
    # Amortize Python launch gaps when comparing GPU kernel configurations.
    graph = torch.cuda.CUDAGraph()
    into()
    with torch.cuda.graph(graph):
        for _ in range(4):
            into()

    def replay():
        graph.replay()
        return out

    candidates["out_graph"] = replay
    ret = {"gfx": get_gfx_runtime()}
    flops = 2 * m * n * k
    nbytes = 2 * (m * k + n * k + m * n)
    for name, fn in candidates.items():
        samples = []
        for _ in range(3):
            result, us = run_perftest(
                fn,
                num_iters=30,
                num_warmup=5,
                num_rotate_args=1,
                use_cuda_event=True,
            )
            error = assert_result(expected, result, name + " measured")
            samples.append(us)
        us = statistics.median(samples) / (4 if name == "out_graph" else 1)
        ret.update(
            {
                name + " us": us,
                name + " TFLOPS": flops / us / 1e6,
                name + " TB/s": nbytes / us / 1e6,
                name + " err": error,
            }
        )
    return ret


def main():
    if get_gfx_runtime() != "gfx950":
        aiter.logger.warning("This tuning qualification targets gfx950; skipping")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-s", "--mnk", nargs="+", type=aiter.dtypes.str2tuple, default=SHAPES
    )
    parser.add_argument("--seed", nargs="+", type=int, default=[912, 913])
    args = parser.parse_args()
    rows = [
        test_latent_gemm(*shape, seed)
        for shape, seed in itertools.product(args.mnk, args.seed)
    ]
    aiter.logger.info(
        "BF16 latent projection GEMMs:\n%s", pd.DataFrame(rows).to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
