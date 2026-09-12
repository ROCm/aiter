# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 fused split-K reduction, including every tuned split-K profile."""

import csv
from pathlib import Path
from unittest import mock

import pytest
import torch

from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import mxfp8_128_bpreshuffle_gemm_gfx1250 as backend
from aiter.ops.shuffle import shuffle_mxfp8fp4_a, shuffle_weight

pytestmark = [
    pytest.mark.skipif(get_gfx() != "gfx1250", reason="requires gfx1250"),
]


def _splitk_configs():
    config_dir = Path(__file__).resolve().parents[1] / "aiter/configs/model_configs"
    rows = []
    for prefix in ("bpreshuffle", "abpreshuffle"):
        path = config_dir / f"dsv4_a8w8_blockscale_{prefix}_tuned_gemm.csv"
        with path.open() as source:
            for row in csv.DictReader(source):
                name = row["kernelName"]
                if (
                    row["gfx"] == "gfx1250"
                    and name.startswith(backend.COMPUTE_WMMA_NAME_PREFIX)
                    and "_sk1_" not in name
                ):
                    rows.append(
                        pytest.param(
                            int(row["M"]),
                            int(row["N"]),
                            int(row["K"]),
                            name,
                            id=f"{prefix}-{row['M']}x{row['N']}x{row['K']}",
                        )
                    )
    return rows


def _inputs(m, n, k, a_preshuffle=False, strided_scale=False):
    a = (torch.randn((m, k), device="cuda") * 0.1).to(dtypes.fp8)
    b = (torch.randn((n, k), device="cuda") * 0.1).to(dtypes.fp8)
    sa = torch.randint(124, 128, (k // 128, m), device="cuda", dtype=torch.uint8)
    sa = sa.view(dtypes.fp8_e8m0)
    sa = sa.T if strided_scale else sa.view(m, k // 128)
    sb = torch.randint(124, 128, (n // 128, k // 128), device="cuda", dtype=torch.uint8)
    sb = sb.view(dtypes.fp8_e8m0)
    if a_preshuffle:
        if m % 2:
            a = torch.cat((a, torch.zeros_like(a[:1])))
        a = shuffle_mxfp8fp4_a(a)
    return a, shuffle_weight(b, layout=(16, 16)), sa, sb


def _run(inputs, out, name):
    return backend.run_gemm_a8w8_mxfp8_128_bpreshuffle_gfx1250(
        *inputs, out, name, a_is_preshuffled="_apre" in name
    )


def _reference(inputs, out, name):
    with mock.patch.object(
        backend, "splitk_epilogue_flags", return_value=(False, True)
    ):
        return _run(inputs, out, name)


@pytest.mark.parametrize("m,n,k,name", _splitk_configs())
def test_tuned_splitk(m, n, k, name):
    """Fused and separate reduction agree bitwise for the CSV-selected kernels."""
    pytest.importorskip("flydsl")
    torch.manual_seed(42)
    inputs = _inputs(m, n, k, a_preshuffle="_apre" in name)
    expected = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    actual = torch.full_like(expected, float("nan"))
    _reference(inputs, expected, name)
    with mock.patch.object(
        backend, "_compile_splitk_reduce", side_effect=AssertionError("extra kernel")
    ):
        _run(inputs, actual, name)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    flags = backend.get_split_k_flags(
        torch.cuda.current_stream().cuda_stream, inputs[0].device
    )
    assert flags.count_nonzero().item() == 0


@pytest.mark.parametrize("split_k", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("a_preshuffle", [False, True])
def test_splitk_tail_and_graph(split_k, dtype, a_preshuffle):
    """Check partial M tiles, padded output, strided scales, and graph replay."""
    pytest.importorskip("flydsl")
    torch.manual_seed(43)
    m, n, k = 129, 512, 4096
    name = (
        "flydsl_mxfp8_128_bpreshuffle_compute_wmma_t128x128x128_"
        f"mw2_nw2_nb4_sk{split_k}_cm1_cn2" + ("_apre" if a_preshuffle else "")
    )
    inputs = _inputs(m, n, k, a_preshuffle, strided_scale=True)
    expected = torch.empty((m, n), dtype=dtype, device="cuda")
    _reference(inputs, expected, name)
    outputs, graphs = [], []
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            storage = torch.full((m, n + 8), 17.0, dtype=dtype, device="cuda")
            out = storage[:, :n]
            _run(inputs, out, name)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                _run(inputs, out, name)
                _run(inputs, out, name)
            outputs.append(storage)
            graphs.append(graph)
    for _ in range(5):
        # Change the partials between replays to expose stale cache lines.
        inputs[0].view(torch.uint8).bitwise_xor_(0x80)
        _reference(inputs, expected, name)
        for stream, graph in zip(streams, graphs):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph.replay()
        torch.cuda.synchronize()
    for stream, storage in zip(streams, outputs):
        torch.testing.assert_close(storage[:, :n], expected, rtol=0, atol=0)
        assert torch.all(storage[:, n:] == 17.0).item()
        flags = backend.get_split_k_flags(stream.cuda_stream, inputs[0].device)
        assert flags.count_nonzero().item() == 0
