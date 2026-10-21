# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and performance coverage for FlyDSL HC mix (Qwen hyper-connections).

    x   : [M(tokens), K]  bf16, K = HC * hidden (HC = 4 branches)
    down: [R, K]          bf16
    up  : [K, R]          bf16
    y   : [M, K / 4] = mean_hc(sigmoid(silu(x @ down.T / 4) @ up.T) * x)

Weights are packed once by ``pack_hc_weights`` after loading (not timed); each
forward calls ``hc_mix(x, down_packed, up_packed)``, which allocates its output.
Production shape is K=10240, R=320 with 1 <= M <= 16 decode tokens.

Run:
    python op_tests/test_flydsl_hc_mix.py
    python op_tests/test_flydsl_hc_mix.py -m 1 4 -s 10240,320
    pytest op_tests/test_flydsl_hc_mix.py -q
"""

import argparse
import itertools
import sys

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.hc_mix import hc_mix, pack_hc_weights
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

# CDNA4 16x16x32 BF16 MFMA only.
SUPPORTED_GFX = ["gfx950"]
pytestmark = pytest.mark.skipif(
    get_gfx() not in SUPPORTED_GFX, reason="FlyDSL hc_mix requires gfx950"
)

SEED = 11
ATOL = 0.004
RTOL = 0.01


def make_inputs(m, k=10240, r=320, scale=0.02):
    x = torch.randn(m, k, dtype=dtypes.bf16)
    d = torch.randn(r, k, dtype=x.dtype) * scale
    u = torch.randn(k, r, dtype=x.dtype) * scale
    return x, d, u


def run_torch(x, d, u):
    # CuTe casts after SiLU, then keeps logits, sigmoid and weighted sum FP32.
    t = F.silu(x.double() @ d.double().T / 4).bfloat16().double()
    return (
        ((t @ u.double().T).sigmoid() * x.double())
        .view(x.shape[0], 4, -1)
        .mean(1)
        .bfloat16()
    )


def check(x, d, u, y, atol=ATOL, msg="hc_mix"):
    return checkAllclose(
        run_torch(x, d, u).to(dtypes.fp32),
        y.to(dtypes.fp32),
        rtol=RTOL,
        atol=atol,
        msg=msg,
    )


@benchmark()
def test_hc_mix(m, k, r):
    torch.manual_seed(SEED)
    x, d, u = make_inputs(m, k, r)
    dp, up = pack_hc_weights(d, u)

    candidates = {"flydsl": hc_mix}

    flops = 2 * m * k * r * 2  # down [M,K]x[K,R] + up [M,R]x[R,K]
    nbytes = (m * k + 2 * k * r + m * k // 4) * x.element_size()  # x + d + u + y

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        # Positional tensors so run_perftest rotates x and the weights, matching
        # the model's per-layer weights arriving cold from HBM.
        y, us = run_perftest(fn, x, dp, up)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = check(x, d, u, y, msg=f"{name}: m={m} k={k} r={r}")
    return ret


test_hc_mix.__test__ = False


@pytest.mark.parametrize("m", [1, 2, 4, 7, 15, 16, 17, 32, 33, 64, 100, 128])
@pytest.mark.parametrize("k,r", [(2048, 32), (8192, 128), (10240, 320)])
def test_correctness(m, k, r):
    torch.manual_seed(SEED)
    x, d, u = make_inputs(m, k, r)
    assert check(x, d, u, hc_mix(x, *pack_hc_weights(d, u))) == 0


@pytest.mark.parametrize("m", [5, 128])
def test_unstaged_operand(monkeypatch, m):
    # Tiles too large for LDS make the up projection read T from global memory.
    import aiter.ops.flydsl.hc_mix as mod

    monkeypatch.setattr(mod, "_LDS_BYTES", 0)
    torch.manual_seed(SEED)
    x, d, u = make_inputs(m)
    assert check(x, d, u, hc_mix(x, *pack_hc_weights(d, u))) == 0


@pytest.mark.parametrize("m", [1, 7, 16])
@pytest.mark.parametrize("k,r", [(2048, 32), (10240, 320)])
def test_fused_reduce(m, k, r):
    torch.manual_seed(SEED)
    x, d, u = make_inputs(m, k, r)
    assert check(x, d, u, hc_mix(x, *pack_hc_weights(d, u), fuse_reduce=True)) == 0


@pytest.mark.parametrize("scale", [0.0, 0.1, 1.0])
def test_gate_range(scale):
    x, d, u = make_inputs(4, scale=scale)
    assert check(x, d, u, hc_mix(x, *pack_hc_weights(d, u)), atol=0.005) == 0


@pytest.mark.parametrize("m", [4, 64])
def test_graph_and_stream_isolation(m):
    # Different concurrent calls must never share split-K scratch or counters.
    data = [make_inputs(m), make_inputs(m)]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    graphs, outputs, packed = [], [], []
    for (x, d, u), stream in zip(data, streams):
        dp, up = pack_hc_weights(d, u)
        # Graphs don't own their inputs: keep the packed weights alive for replay.
        packed.append((dp, up))
        hc_mix(x, dp, up)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs.append(hc_mix(x, dp, up))
        graphs.append(graph)
    for _ in range(12):
        for (x, _, _), stream, graph in zip(data, streams, graphs):
            with torch.cuda.stream(stream):
                x.normal_()
                graph.replay()
        torch.cuda.synchronize()
        for (x, d, u), y in zip(data, outputs):
            assert check(x, d, u, y, msg="hc_mix graph replay") == 0
    snapshots = [y.clone() for y in outputs]
    for graph in graphs:
        graph.replay()
    torch.cuda.synchronize()
    for a, b in zip(snapshots, outputs):
        assert torch.equal(a, b)


@pytest.mark.parametrize("m", [0, 129])
def test_reject_rows(m):
    x, d, u = make_inputs(m)
    with pytest.raises(ValueError):
        hc_mix(x, *pack_hc_weights(d, u))


def test_reject_invalid_arguments():
    x, d, u = make_inputs(1)
    packed = pack_hc_weights(d, u)
    for split in [0, -1, 3]:
        with pytest.raises(ValueError):
            hc_mix(x, *packed, split_k=split)
    with pytest.raises(ValueError):
        hc_mix(x.half(), *packed)
    with pytest.raises(ValueError):
        hc_mix(x, packed[0], packed[1][:-1])
    with pytest.raises(ValueError):
        hc_mix(torch.cat([x] * 17), *packed, fuse_reduce=True)
    with pytest.raises(ValueError):
        pack_hc_weights(d, u, hc=5)


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("flydsl hc_mix unsupported on %s; skipping", get_gfx())
        return 0

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL hc_mix correctness and perf",
    )
    parser.add_argument(
        "-m",
        "--m",
        type=int,
        nargs="*",
        default=[1, 2, 4, 8, 16],
        help="""Tokens (1..128).
    e.g.: -m 1 4""",
    )
    parser.add_argument(
        "-s",
        "--kr",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(2048, 32), (8192, 128), (10240, 320)],
        help="""K (HC * hidden), R (HC rank).
    e.g.: -s 10240,320""",
    )
    args = parser.parse_args()

    aiter.logger.info("hc_mix: running correctness checks...")
    status = pytest.main([__file__, "-q", "-p", "no:cacheprovider"])
    if status != pytest.ExitCode.OK:
        aiter.logger.error(
            "hc_mix: checks failed (pytest exit %s); skipping perf sweep", int(status)
        )
        return 1

    df = []
    for (k, r), m in itertools.product(args.kr, args.m):
        df.append(test_hc_mix(m, k, r))
    df = pd.DataFrame(df)
    aiter.logger.info(
        "flydsl hc_mix summary (markdown):\n%s", df.to_markdown(index=False)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
