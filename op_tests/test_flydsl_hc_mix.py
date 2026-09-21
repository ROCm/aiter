# SPDX-License-Identifier: MIT
import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available() or torch.version.hip is None:
    pytest.skip("Requires gfx950", allow_module_level=True)
if torch.cuda.get_device_properties().gcnArchName.split(":")[0] != "gfx950":
    pytest.skip("Requires gfx950", allow_module_level=True)

from aiter.ops.flydsl.hc_mix import hc_mix, pack_hc_weights


def inputs(m, k=10240, r=320, scale=0.02):
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    d = torch.randn(r, k, device="cuda", dtype=x.dtype) * scale
    u = torch.randn(k, r, device="cuda", dtype=x.dtype) * scale
    return x, d, u


def reference(x, d, u):
    # CuTe casts after SiLU, then keeps logits, sigmoid and weighted sum FP32.
    t = F.silu(x.double() @ d.double().T / 4).bfloat16().double()
    return ((t @ u.double().T).sigmoid() * x.double()).view(x.shape[0], 4, -1).mean(1).bfloat16()


@pytest.mark.parametrize("m", [1, 2, 4, 7, 15, 16])
@pytest.mark.parametrize("k,r", [(2048, 32), (8192, 128), (10240, 320)])
def test_correctness(m, k, r):
    torch.manual_seed(11)
    x, d, u = inputs(m, k, r)
    dp, up = pack_hc_weights(d, u)
    y = hc_mix(x, dp, up)
    torch.testing.assert_close(y, reference(x, d, u), atol=0.004, rtol=0.01)


@pytest.mark.parametrize("scale", [0.0, 0.1, 1.0])
def test_gate_range(scale):
    x, d, u = inputs(4, scale=scale)
    dp, up = pack_hc_weights(d, u)
    torch.testing.assert_close(hc_mix(x, dp, up), reference(x, d, u), atol=0.005, rtol=0.01)


def test_graph_and_stream_isolation():
    # Different concurrent calls must never share split-K scratch or counters.
    data = [inputs(4), inputs(4)]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    graphs, outputs, packed = [], [], []
    for (x, d, u), stream in zip(data, streams):
        dp, up = pack_hc_weights(d, u)
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
            torch.testing.assert_close(y, reference(x, d, u), atol=0.004, rtol=0.01)
    snapshots = [y.clone() for y in outputs]
    for graph in graphs:
        graph.replay()
    torch.cuda.synchronize()
    for a, b in zip(snapshots, outputs):
        assert torch.equal(a, b)


@pytest.mark.parametrize("m", [0, 17])
def test_reject_rows(m):
    x, d, u = inputs(m)
    with pytest.raises(ValueError):
        hc_mix(x, *pack_hc_weights(d, u))


def test_reject_invalid_arguments():
    x, d, u = inputs(1)
    packed = pack_hc_weights(d, u)
    for split in [0, -1, 3]:
        with pytest.raises(ValueError):
            hc_mix(x, *packed, split_k=split)
    with pytest.raises(ValueError):
        hc_mix(x.half(), *packed)
    with pytest.raises(ValueError):
        hc_mix(x, packed[0], packed[1][:-1])
    with pytest.raises(ValueError):
        pack_hc_weights(d, u, hc=5)
