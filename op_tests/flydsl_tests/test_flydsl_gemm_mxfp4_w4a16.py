# SPDX-License-Identifier: MIT

"""Config, layout and gfx1151 correctness coverage for dense MXFP4 GEMM."""

from dataclasses import replace

import pytest
import torch

pytest.importorskip("flydsl")

from aiter.jit.utils.chip_info import get_gfx_runtime_for_device
from aiter.ops.flydsl import (
    flydsl_gemm_mxfp4_w4a16,
    prepare_gemm_mxfp4_w4a16_weight,
)
from aiter.ops.flydsl.gemm_mxfp4_w4a16_config import (
    _read_tuned_configs,
    select_mxfp4_w4a16_config,
    validate_config,
)

_GFX1151 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is None
    or get_gfx_runtime_for_device() != "gfx1151",
    reason="Requires gfx1151",
)

_QWEN_PROJECTIONS = (
    (34816, 5120),
    (5120, 17408),
    (16384, 5120),
    (96, 5120),
    (14336, 5120),
    (5120, 6144),
)
_TUNED_M = {1, 8, 32, 128, 512}


@pytest.mark.parametrize("n,k", _QWEN_PROJECTIONS)
def test_select_qwen_config(n, k):
    rows = _read_tuned_configs("gfx1151", n, k)
    assert set(rows) == _TUNED_M
    for m, config in rows.items():
        assert select_mxfp4_w4a16_config("gfx1151", m, n, k, 40) is config
        assert select_mxfp4_w4a16_config("gfx1151", m, n, k, 32) is not config

    # Both an unmatched M and an unmatched CU count use legal fallback configs.
    fallback = select_mxfp4_w4a16_config("gfx1151", 2, n, k, 40)
    assert fallback is validate_config(fallback, (2, n, k))
    assert all(fallback is not config for config in rows.values())


def test_prepare_odd_n():
    packed = torch.tensor([[0x10, 0x43], [0x65, 0x87], [0xAB, 0xCD]], dtype=torch.uint8)
    scales = torch.tensor([[1], [2], [3]], dtype=torch.uint8)
    prepared = prepare_gemm_mxfp4_w4a16_weight(packed, scales)
    assert (prepared.n, prepared.k) == (3, 4)
    assert prepared.packed.tolist() == [0x50, 0x1B, 0xA6, 0x73, 0x4D, 0xC8]
    assert prepared.scales.tolist() == [[1, 2, 3]]


def _inputs(m, n, k):
    generator = torch.Generator().manual_seed(m + n + k)
    packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=generator)
    scales = torch.randint(
        123, 127, (n, (k + 31) // 32), dtype=torch.uint8, generator=generator
    )
    a = (torch.randn((m, k), generator=generator) * k**-0.5).bfloat16()
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).view(n, k)
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
    weights = levels[(codes & 7).long()]
    weights = torch.where(codes & 8 != 0, -weights, weights)
    weights *= (scales.float() - 127).exp2().repeat_interleave(32, dim=1)[:, :k]
    reference = (a.float() @ weights.T).bfloat16().cuda()
    return (
        a.cuda(),
        prepare_gemm_mxfp4_w4a16_weight(packed.cuda(), scales.cuda()),
        reference,
    )


@_GFX1151
@pytest.mark.parametrize("m,n,k", [(1, 96, 5120), (3, 33, 38), (29, 127, 250)])
def test_dense_gemm(m, n, k):
    a, prepared, reference = _inputs(m, n, k)
    buffer = torch.full((m * n + 16,), -99.0, dtype=a.dtype, device=a.device)
    out = buffer[:-16].view(m, n)
    assert flydsl_gemm_mxfp4_w4a16(a, prepared, out=out) is out
    torch.testing.assert_close(out, reference, rtol=0.016, atol=0.03125)
    prepared.packed.bitwise_xor_(0x88)
    assert flydsl_gemm_mxfp4_w4a16(a, prepared, out=out) is out
    torch.testing.assert_close(out, -reference, rtol=0.016, atol=0.03125)
    assert torch.all(buffer[-16:] == -99).item()


@_GFX1151
def test_split_k_graph_replays():
    shape = (3, 33, 70)
    a, prepared, reference = _inputs(*shape)
    config = replace(select_mxfp4_w4a16_config("gfx1151", *shape, 40), split_k=4)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graphs = []
    with torch.cuda.stream(stream):
        # Compile before capture. The two graphs and eager use independent scratch.
        flydsl_gemm_mxfp4_w4a16(a, prepared, config=config)
        for _ in range(2):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                result = flydsl_gemm_mxfp4_w4a16(a, prepared, config=config)
            graphs.append((graph, result))
    torch.cuda.current_stream().wait_stream(stream)
    for sign in (-1, 1):
        prepared.packed.bitwise_xor_(0x88)
        torch.testing.assert_close(
            flydsl_gemm_mxfp4_w4a16(a, prepared, config=config),
            sign * reference,
            rtol=0.016,
            atol=0.03125,
        )
        for graph, out in reversed(graphs):
            graph.replay()
            torch.testing.assert_close(out, sign * reference, rtol=0.016, atol=0.03125)
