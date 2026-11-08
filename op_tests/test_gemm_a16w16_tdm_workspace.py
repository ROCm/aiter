# SPDX-License-Identifier: MIT
"""FP32 workspace reduction, policy cache separation and stream lifetime tests."""

import pytest
import torch


def require_gpu():
    if not torch.cuda.is_available():
        pytest.skip("ROCm GPU required")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1250":
        pytest.skip("gfx1250 required")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("split", range(2, 10))
@pytest.mark.parametrize("interleave", [False, True])
def test_workspace_graph_odd_n(dtype, split, interleave):
    require_gpu()
    from aiter.ops.flydsl import flydsl_gemm_a16w16_tdm as gemm

    torch.set_num_threads(1)
    torch.manual_seed(split)
    m, n, k = 137, 139, split * 128 * 5
    a = torch.randn(m, k, device="cuda", dtype=dtype) * 0.1
    b = (torch.randn(n, k, device="cuda", dtype=dtype) * 0.1).t()
    storage = torch.full((m + 2, 160), 17.0, device="cuda", dtype=dtype)
    out = storage[1 : m + 1, 8 : n + 8]
    bias = torch.randn(n, device="cuda", dtype=dtype) * 0.1
    kw = {
        "split_k": split,
        "split_k_policy": "workspace",
        "interleave": interleave,
        "group_m": 3,
    }
    gemm(a, b, out, bias, **kw)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gemm(a, b, out, bias, **kw)
    try:
        for rep in range(5):
            a.neg_()
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            ref = (a.float() @ b.float()).to(dtype)
            ref = (ref.float() + bias.float()).to(dtype)
            assert torch.isfinite(out).all()
            err = (out.float() - ref.float()).norm() / ref.float().norm()
            assert err < (0.004 if dtype == torch.bfloat16 else 0.0006), err
            assert torch.all(storage[0] == 17) and torch.all(storage[-1] == 17)
            assert torch.all(storage[:, :8] == 17) and torch.all(storage[:, 147:] == 17)
            if rep == 0:
                first = out.clone()
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(out, first)
    finally:
        graph.reset()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_workspace_cancellation_and_determinism(dtype):
    require_gpu()
    from aiter.ops.flydsl import flydsl_gemm_a16w16_tdm as gemm

    big = 512.0 if dtype == torch.bfloat16 else 4096.0
    a = torch.zeros(128, 512, device="cuda", dtype=dtype)
    b = torch.ones(128, 512, device="cuda", dtype=dtype).t()
    a[:, 0] = big
    a[:, 1] = 1
    a[:, 256] = -big
    atomic = gemm(a, b, split_k=2, split_k_policy="atomic")
    workspace = gemm(a, b, split_k=2, split_k_policy="workspace")
    torch.cuda.synchronize()
    assert torch.count_nonzero(atomic).item() == 0
    assert torch.equal(workspace, torch.ones_like(workspace))
    old = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        y = gemm(a, b, split_k=2, split_k_policy="workspace")
        torch.cuda.synchronize()
        assert torch.equal(y, workspace)
        with pytest.raises(RuntimeError, match="nondeterministic"):
            gemm(a, b, split_k=2, split_k_policy="atomic")
    finally:
        torch.use_deterministic_algorithms(old)
    # Half partial overflow must not be introduced by the FP32 workspace.
    a[:, :256] = 512
    a[:, 256:] = -512
    y = gemm(a, b, split_k=2, split_k_policy="workspace")
    torch.cuda.synchronize()
    assert torch.count_nonzero(y).item() == 0


def test_workspace_concurrent_streams_and_nn():
    require_gpu()
    from aiter.ops.flydsl import flydsl_gemm_a16w16_tdm as gemm

    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    data = []
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            a = torch.randn(65, 1024, device="cuda", dtype=torch.float16) * 0.1
            b = torch.randn(1024, 80, device="cuda", dtype=torch.float16) * 0.1
            out = gemm(
                a,
                b,
                layout="nn",
                interleave=False,
                split_k=4,
                split_k_policy="workspace",
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                gemm(
                    a,
                    b,
                    out=out,
                    layout="nn",
                    interleave=False,
                    split_k=4,
                    split_k_policy="workspace",
                )
            data.append((a, b, out, graph))
    for _ in range(10):
        for stream, (a, b, out, g) in zip(streams, data):
            with torch.cuda.stream(stream):
                a.neg_()
                g.replay()
    for stream, (a, b, out, g) in zip(streams, data):
        stream.synchronize()
        torch.testing.assert_close(
            out.float(), a.float() @ b.float(), atol=0.002, rtol=0.002
        )
        g.reset()


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q", __file__]))
