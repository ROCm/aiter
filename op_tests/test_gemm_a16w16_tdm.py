# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 unified TDM GEMM: graph liveness, cluster init, strides, and precision."""

import pytest
import torch


def require_gpu():
    if not torch.cuda.is_available():
        pytest.skip("ROCm GPU required")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1250":
        pytest.skip("gfx1250 required")


def matrix(m, n, dtype, transpose=False):
    # Pointer offset and padded leading strides; guard rows/columns not written.
    rows, cols = (n, m) if transpose else (m, n)
    buf = torch.full(
        (rows + 2, (cols + 15) // 8 * 8 + 8), 17.0, device="cuda", dtype=dtype
    )
    view = buf[1 : rows + 1, 8 : cols + 8]
    view.copy_(torch.randn_like(view) * 0.2)
    return (view.t() if transpose else view), buf


def check(y, a, b, bias, sk):
    # GPU FP32 matmul oracle, not another instance of the tested kernel.
    ref = a.float() @ b.float()
    ref = ref.to(y.dtype)
    if bias is not None:
        ref = (ref.float() + bias.float()).to(y.dtype)
    error = (y.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12)
    limit = (
        (0.006 if y.dtype == torch.bfloat16 else 0.0008) * sk**0.5 if sk > 1 else 0.004
    )
    assert torch.isfinite(y).all()
    assert error.item() < limit, (error.item(), limit)
    # Global error alone could hide an isolated bad tile/row.
    rms = ref.float().square().mean().sqrt().item()
    assert (y.float() - ref.float()).abs().max().item() < rms * (
        0.04 if y.dtype == torch.bfloat16 else 0.008
    ) * max(1, sk) + 0.002


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("sk", range(1, 10))
@pytest.mark.parametrize("interleave", [False, True])
def test_split_graph(dtype, sk, interleave):
    require_gpu()
    from aiter.ops.flydsl.gemm_a16w16_tdm import gemm_a16w16_tdm as gemm

    torch.manual_seed(177 + sk)
    torch.set_num_threads(1)
    # Odd K tile count tests unroll2 remainder and drain for every split rank.
    a, _ = matrix(137, sk * 128 * 5, dtype)
    b, _ = matrix(sk * 128 * 5, 146, dtype, True)
    y, guard = matrix(137, 146, dtype)
    bias = torch.randn(152, device="cuda", dtype=dtype)[:146] * 0.1
    kw = {"split_k": sk, "interleave": interleave, "group_m": 3}
    gemm(a, b, y, bias, **kw)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        gemm(a, b, y, bias, **kw)
    for i in range(5):
        a.neg_()
        # Input events are on the capture stream; stale output must never survive.
        y.fill_(float("nan") if i % 2 else 99)
        g.replay()
        torch.cuda.synchronize()
        check(y, a, b, bias, sk)
        assert torch.all(guard[0] == 17) and torch.all(guard[-1] == 17)
        assert torch.all(guard[:, :8] == 17) and torch.all(guard[:, 154:] == 17)
    g.reset()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "stages,tiles", [(2, 2), (2, 3), (3, 3), (3, 4), (3, 7), (4, 4), (4, 9), (5, 11)]
)
@pytest.mark.parametrize("interleave", [False, True])
def test_pipeline_tail(dtype, stages, tiles, interleave):
    require_gpu()
    from aiter.ops.flydsl.gemm_a16w16_tdm import gemm_a16w16_tdm as gemm

    bk = 64 if stages == 5 else 128
    a, _ = matrix(133, tiles * bk, dtype)
    b, _ = matrix(tiles * bk, 139, dtype, True)
    out = gemm(a, b, block_k=bk, stages=stages, interleave=interleave)
    torch.cuda.synchronize()
    check(out, a, b, None, 1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sk", [1, 3])
def test_nn_small_tile(dtype, sk):
    require_gpu()
    from aiter.ops.flydsl.gemm_a16w16_tdm import gemm_a16w16_tdm as gemm

    a, _ = matrix(79, sk * 384, dtype)
    b, _ = matrix(sk * 384, 82, dtype)
    y = gemm(
        a,
        b,
        block_m=64,
        block_n=64,
        m_waves=2,
        n_waves=2,
        layout="nn",
        interleave=False,
        split_k=sk,
    )
    torch.cuda.synchronize()
    check(y, a, b, None, sk)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_other_stream(dtype):
    require_gpu()
    from aiter.ops.flydsl.gemm_a16w16_tdm import gemm_a16w16_tdm as gemm

    a, _ = matrix(137, 1024, dtype)
    b, _ = matrix(1024, 144, dtype, True)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    y = gemm(a, b, split_k=4, stream=s)
    torch.cuda.current_stream().wait_stream(s)
    check(y, a, b, None, 4)


def test_invalid_output_alias_and_odd_atomic():
    require_gpu()
    from aiter.ops.flydsl.gemm_a16w16_tdm import gemm_a16w16_tdm as gemm

    a = torch.ones(256, 256, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="overlap"):
        gemm(a, a.t(), out=a)
    with pytest.raises(ValueError, match="even N"):
        gemm(a, a[:129].t(), block_k=32, split_k=2)
    with pytest.raises(ValueError, match="K divisible"):
        gemm(a, a.t(), split_k=3)
    old = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with pytest.raises(RuntimeError, match="nondeterministic"):
            gemm(a, a.t(), block_k=32, split_k=2)
    finally:
        torch.use_deterministic_algorithms(old)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cluster_grouped_swizzle_and_exact_init(dtype):
    require_gpu()
    from aiter.ops.flydsl.gemm_a16w16_tdm import gemm_a16w16_tdm as gemm

    # Ragged group of output tiles, both full and partial C tiles, exact sums.
    a = torch.ones(385, 1536, device="cuda", dtype=dtype)
    b = torch.ones(264, 1536, device="cuda", dtype=dtype).t()
    out = torch.full((385, 272), 37.0, device="cuda", dtype=dtype)[:, :264]
    for sk in (1, 2, 3, 6):
        gemm(a, b, out=out, split_k=sk, group_m=3)
        torch.cuda.synchronize()
        assert torch.equal(out, torch.full_like(out, 1536.0))
        a.zero_()
        gemm(a, b, out=out, split_k=sk, group_m=3)
        torch.cuda.synchronize()
        assert torch.count_nonzero(out).item() == 0
        a.fill_(1)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_atomic_partial_rounding_contract(dtype):
    require_gpu()
    from aiter.ops.flydsl.gemm_a16w16_tdm import gemm_a16w16_tdm as gemm

    # FP32 sum is 1, but the first split's sum loses the unit when converted
    # to the requested low-precision atomic input. This is intentional.
    big = 512.0 if dtype == torch.bfloat16 else 4096.0
    a = torch.zeros(128, 512, device="cuda", dtype=dtype)
    a[:, 0] = big
    a[:, 1] = 1
    a[:, 256] = -big
    b = torch.ones(128, 512, device="cuda", dtype=dtype).t()
    exact = gemm(a, b, split_k=1)
    rounded = gemm(a, b, split_k=2)
    torch.cuda.synchronize()
    assert torch.equal(exact, torch.ones_like(exact))
    assert torch.count_nonzero(rounded).item() == 0


def test_fp16_atomic_partial_overflow_contract():
    require_gpu()
    from aiter.ops.flydsl.gemm_a16w16_tdm import gemm_a16w16_tdm as gemm

    a = torch.empty(128, 512, device="cuda", dtype=torch.float16)
    a[:, :256] = 512
    a[:, 256:] = -512
    b = torch.ones(128, 512, device="cuda", dtype=torch.float16).t()
    whole = gemm(a, b, split_k=1)
    split = gemm(a, b, split_k=2)
    torch.cuda.synchronize()
    assert torch.count_nonzero(whole).item() == 0
    assert torch.isnan(split).all()  # +inf + -inf, NOT an FP32 cancellation.


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_schedule_identity_and_empty_k_guard(dtype):
    require_gpu()
    from aiter.ops.flydsl.gemm_a16w16_tdm import gemm_a16w16_tdm as gemm

    a = torch.randn(151, 896, device="cuda", dtype=dtype)
    b = torch.randn(152, 896, device="cuda", dtype=dtype).t()
    simple = gemm(a, b, interleave=False, stages=3)
    interleaved = gemm(a, b, interleave=True, stages=3)
    torch.cuda.synchronize()
    assert torch.equal(simple, interleaved)


def test_concurrent_streams_and_public_export():
    require_gpu()
    from aiter.ops.flydsl import flydsl_gemm_a16w16_tdm as gemm

    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    data = []
    for s, sk in zip(streams, (2, 4)):
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            a = torch.randn(137, 1024, device="cuda", dtype=torch.bfloat16) * 0.2
            b = torch.randn(144, 1024, device="cuda", dtype=torch.bfloat16).t() * 0.2
            y = gemm(a, b, split_k=sk)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=s):
                gemm(a, b, out=y, split_k=sk)
            data.append((a, b, y, sk, graph))
    for rep in range(12):
        for s, (a, b, y, sk, g) in zip(streams, data):
            with torch.cuda.stream(s):
                a.neg_()
                y.fill_(float("nan"))
                g.replay()
    for s, (a, b, y, sk, g) in zip(streams, data):
        s.synchronize()
        check(y, a, b, None, sk)
        g.reset()


@pytest.mark.parametrize(
    "changes",
    [
        {"bm": 0},
        {"bk": 48},
        {"stages": 1},
        {"split_k": 10},
        {"split_k": True},
        {"mw": 64},
        {"group_m": -1},
        {"layout": "tt"},
        {"bm": 48},
        {"bn": 16},
        {"stages": 16},
        {"interleave": True, "layout": "nn"},
    ],
)
def test_config_guards(changes):
    from aiter.ops.flydsl.kernels.gemm_a16w16_tdm_gfx1250 import validate_config

    args = {
        "bm": 128,
        "bn": 128,
        "bk": 128,
        "stages": 2,
        "mw": 2,
        "nw": 2,
        "split_k": 1,
        "group_m": 0,
        "layout": "nt",
        "interleave": True,
    }
    args.update(changes)
    with pytest.raises(ValueError):
        validate_config(**args)


@pytest.mark.parametrize(
    "bm,bn,mw,nw", [(128, 256, 2, 2), (256, 128, 2, 2), (128, 256, 1, 2)]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_rectangular_serpentine(bm, bn, mw, nw, dtype):
    require_gpu()
    from aiter.ops.flydsl.gemm_a16w16_tdm import gemm_a16w16_tdm as gemm

    a, _ = matrix(137, 768, dtype)
    b, _ = matrix(768, 266, dtype, True)
    kw = {"block_m": bm, "block_n": bn, "m_waves": mw, "n_waves": nw, "split_k": 3}
    out = gemm(a, b, **kw)
    torch.cuda.synchronize()
    check(out, a, b, None, 3)


def test_deep_small_pipeline():
    require_gpu()
    from aiter.ops.flydsl.gemm_a16w16_tdm import gemm_a16w16_tdm as gemm

    a, _ = matrix(65, 32 * 17, torch.bfloat16)
    b, _ = matrix(32 * 17, 72, torch.bfloat16, True)
    out = gemm(a, b, block_m=64, block_n=64, block_k=32, stages=16, interleave=False)
    torch.cuda.synchronize()
    check(out, a, b, None, 1)


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q", __file__]))
