# SPDX-License-Identifier: MIT

import argparse
import statistics

import pandas as pd
import pytest
import torch

import aiter
from aiter.ops.mhc import (
    MHC_PRE_RMSNORM_QUANT_GFX950_M,
    get_mhc_pre_splitk,
    mhc_fused_post_pre_quant,
    mhc_post,
    mhc_pre,
    mhc_pre_big_fuse_rmsnorm,
    mhc_pre_big_fuse_rmsnorm_quant,
    mhc_shuffle_fn,
)


@pytest.mark.parametrize("m", [512, 1025])
def test_mhc_pre_quant_rejects_unsupported_m(m: int):
    assert m not in MHC_PRE_RMSNORM_QUANT_GFX950_M


@pytest.mark.parametrize("m", [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
@pytest.mark.parametrize("seed", [7, 19])
def test_mhc_pre_big_fuse_rmsnorm_quant(m: int, seed: int):
    if not torch.cuda.is_available():
        pytest.skip("requires a ROCm GPU")
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx950")

    torch.manual_seed(seed)
    hidden_size = 4096
    hc_mult = 4
    hc_mult3 = hc_mult * hc_mult + 2 * hc_mult
    n_splits = get_mhc_pre_splitk(m, hc_mult * hidden_size)[0]

    gemm_out_mul = torch.randn(
        n_splits, m, hc_mult3, dtype=torch.float32, device="cuda"
    )
    gemm_out_sqrsum = torch.rand(n_splits, m, dtype=torch.float32, device="cuda") + 1
    hc_scale = torch.randn(3, dtype=torch.float32, device="cuda") * 0.1
    hc_base = torch.randn(hc_mult3, dtype=torch.float32, device="cuda")
    residual = torch.randn(m, hc_mult, hidden_size, dtype=torch.bfloat16, device="cuda")
    norm_weight = torch.randn(hidden_size, dtype=torch.bfloat16, device="cuda")

    post_ref = torch.empty(m, hc_mult, dtype=torch.float32, device="cuda")
    comb_ref = torch.empty(m, hc_mult * hc_mult, dtype=torch.float32, device="cuda")
    out_ref = torch.empty(m, hidden_size, dtype=torch.bfloat16, device="cuda")
    mhc_pre_big_fuse_rmsnorm(
        post_ref,
        comb_ref,
        out_ref,
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        norm_weight,
    )
    quant_ref, scale_ref = aiter.per_token_quant_hip(
        out_ref, quant_dtype=aiter.dtypes.fp8
    )

    post = torch.empty_like(post_ref)
    comb = torch.empty_like(comb_ref)
    out = torch.empty_like(out_ref)
    quant = torch.empty_like(quant_ref)
    scale = torch.empty_like(scale_ref)
    mhc_pre_big_fuse_rmsnorm_quant(
        post,
        comb,
        out,
        quant,
        scale,
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        norm_weight,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(post, post_ref, rtol=0, atol=0)
    torch.testing.assert_close(comb, comb_ref, rtol=0, atol=0)
    torch.testing.assert_close(out, out_ref, rtol=0, atol=0)
    torch.testing.assert_close(quant.float(), quant_ref.float(), rtol=0, atol=0)
    torch.testing.assert_close(scale, scale_ref, rtol=0, atol=0)
    assert torch.isfinite(out).all()
    assert torch.isfinite(scale).all()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mhc_pre_big_fuse_rmsnorm_quant(
            post,
            comb,
            out,
            quant,
            scale,
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual,
            norm_weight,
        )

    for pattern in ("zeros", "signed_extremes"):
        if pattern == "zeros":
            gemm_out_mul.zero_()
            gemm_out_sqrsum.fill_(1)
            hc_scale.zero_()
            hc_base.zero_()
            residual.zero_()
            norm_weight.fill_(1)
        else:
            gemm_out_mul.fill_(1)
            gemm_out_mul.view(-1)[1::2].neg_()
            gemm_out_sqrsum.fill_(4)
            hc_scale.fill_(0.1)
            hc_base.fill_(0.5)
            residual.fill_(448)
            residual.view(-1)[1::2].neg_()
            norm_weight.fill_(2)
            norm_weight[1::2].neg_()

        mhc_pre_big_fuse_rmsnorm(
            post_ref,
            comb_ref,
            out_ref,
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual,
            norm_weight,
        )
        quant_ref, scale_ref = aiter.per_token_quant_hip(
            out_ref, quant_dtype=aiter.dtypes.fp8
        )
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(post, post_ref, rtol=0, atol=0)
        torch.testing.assert_close(comb, comb_ref, rtol=0, atol=0)
        torch.testing.assert_close(out, out_ref, rtol=0, atol=0)
        torch.testing.assert_close(quant.float(), quant_ref.float(), rtol=0, atol=0)
        torch.testing.assert_close(scale, scale_ref, rtol=0, atol=0)
        assert torch.isfinite(out).all(), pattern
        assert torch.isfinite(scale).all(), pattern


@pytest.mark.parametrize("m", [4096, 8192, 16384])
def test_mhc_fused_post_pre_quant_matches_fallback(m: int):
    if not torch.cuda.is_available():
        pytest.skip("requires a ROCm GPU")
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx950")

    torch.manual_seed(31 + m)
    hidden_size = 4096
    hc_mult = 4
    hc_mult3 = hc_mult * hc_mult + 2 * hc_mult
    layer_input = (
        torch.randn(m, hidden_size, dtype=torch.bfloat16, device="cuda") * 0.02
    )
    residual = (
        torch.randn(m, hc_mult, hidden_size, dtype=torch.bfloat16, device="cuda") * 0.02
    )
    post = torch.sigmoid(torch.randn(m, hc_mult, device="cuda"))
    comb = torch.softmax(torch.randn(m, hc_mult, hc_mult, device="cuda"), dim=-1)
    fn = (
        torch.randn(
            hc_mult3,
            hc_mult * hidden_size,
            dtype=torch.float32,
            device="cuda",
        )
        * 0.01
    )
    packed_fn = mhc_shuffle_fn(fn)
    hc_scale = torch.tensor([0.5, 0.25, 0.25], dtype=torch.float32, device="cuda")
    hc_base = torch.zeros(hc_mult3, dtype=torch.float32, device="cuda")
    norm_weight = torch.linspace(
        0.75, 1.25, hidden_size, dtype=torch.bfloat16, device="cuda"
    )

    next_ref = torch.empty_like(residual)
    mhc_post(next_ref, layer_input, residual, post, comb)
    post_ref, comb_ref, out_ref = mhc_pre(
        next_ref,
        packed_fn,
        hc_scale,
        hc_base,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=20,
        norm_weight=norm_weight,
        w_preshuffle_bf16=1,
    )
    actual = mhc_fused_post_pre_quant(
        layer_input,
        residual,
        post,
        comb,
        packed_fn,
        hc_scale,
        hc_base,
        norm_weight,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=20,
    )
    post_actual, comb_actual, out_actual, quant_actual, scale_actual, next_actual = (
        actual
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(next_actual, next_ref, atol=0, rtol=0)
    torch.testing.assert_close(post_actual, post_ref, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(comb_actual, comb_ref, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(out_actual, out_ref, atol=6.25e-2, rtol=3e-2)
    quant_ref, scale_ref = aiter.per_token_quant_hip(
        out_actual, quant_dtype=aiter.dtypes.fp8
    )
    torch.testing.assert_close(quant_actual.float(), quant_ref.float(), atol=0, rtol=0)
    torch.testing.assert_close(scale_actual, scale_ref, atol=0, rtol=0)

    if m == 8192:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = mhc_fused_post_pre_quant(
                layer_input,
                residual,
                post,
                comb,
                packed_fn,
                hc_scale,
                hc_base,
                norm_weight,
                hc_post_mult_value=2.0,
                sinkhorn_repeat=20,
            )
        layer_input.copy_(torch.randn_like(layer_input) * 0.02)
        residual.copy_(torch.randn_like(residual) * 0.02)
        post.copy_(torch.sigmoid(torch.randn_like(post)))
        comb.copy_(torch.softmax(torch.randn_like(comb), dim=-1))
        graph.replay()
        torch.cuda.synchronize()

        captured_out, captured_quant, captured_scale, captured_next = (
            captured[2],
            captured[3],
            captured[4],
            captured[5],
        )
        mhc_post(next_ref, layer_input, residual, post, comb)
        torch.testing.assert_close(captured_next, next_ref, atol=0, rtol=0)
        quant_ref, scale_ref = aiter.per_token_quant_hip(
            captured_out, quant_dtype=aiter.dtypes.fp8
        )
        torch.testing.assert_close(
            captured_quant.float(), quant_ref.float(), atol=0, rtol=0
        )
        torch.testing.assert_close(captured_scale, scale_ref, atol=0, rtol=0)


def _bench_batched(fn, warmup: int = 10, iters: int = 100, reps: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    timings = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        stop.record()
        stop.synchronize()
        timings.append(start.elapsed_time(stop) * 1000 / iters)
    return statistics.median(timings)


def _bench_synchronized(fn, warmup: int = 10, iters: int = 20, reps: int = 3) -> float:
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()
    timings = []
    for _ in range(reps):
        samples = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            stop.record()
            stop.synchronize()
            samples.append(start.elapsed_time(stop) * 1000)
        timings.append(statistics.median(samples))
    return statistics.median(timings)


def benchmark_mhc_pre_big_fuse_rmsnorm_quant(m: int) -> dict:
    torch.manual_seed(20260929 + m)
    hidden_size = 4096
    hc_mult = 4
    hc_mult3 = hc_mult * hc_mult + 2 * hc_mult
    n_splits = get_mhc_pre_splitk(m, hc_mult * hidden_size)[0]
    gemm_out_mul = torch.randn(
        n_splits, m, hc_mult3, dtype=torch.float32, device="cuda"
    )
    gemm_out_sqrsum = torch.rand(n_splits, m, dtype=torch.float32, device="cuda") + 1
    hc_scale = torch.randn(3, dtype=torch.float32, device="cuda") * 0.1
    hc_base = torch.randn(hc_mult3, dtype=torch.float32, device="cuda")
    residual = torch.randn(m, hc_mult, hidden_size, dtype=torch.bfloat16, device="cuda")
    norm_weight = torch.randn(hidden_size, dtype=torch.bfloat16, device="cuda")
    post = torch.empty(m, hc_mult, dtype=torch.float32, device="cuda")
    comb = torch.empty(m, hc_mult * hc_mult, dtype=torch.float32, device="cuda")
    out = torch.empty(m, hidden_size, dtype=torch.bfloat16, device="cuda")
    quant = torch.empty(m, hidden_size, dtype=aiter.dtypes.fp8, device="cuda")
    scale = torch.empty(m, 1, dtype=torch.float32, device="cuda")

    def baseline():
        mhc_pre_big_fuse_rmsnorm(
            post,
            comb,
            out,
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual,
            norm_weight,
        )
        return aiter.per_token_quant_hip(out, quant_dtype=aiter.dtypes.fp8)

    def candidate():
        mhc_pre_big_fuse_rmsnorm_quant(
            post,
            comb,
            out,
            quant,
            scale,
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual,
            norm_weight,
        )
        return quant, scale

    baseline_us = _bench_batched(baseline)
    fused_us = _bench_batched(candidate)
    baseline_eager_us = _bench_synchronized(baseline)
    fused_eager_us = _bench_synchronized(candidate)

    baseline_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(baseline_graph):
        baseline()
    candidate_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(candidate_graph):
        candidate()
    baseline_graph_us = _bench_batched(baseline_graph.replay)
    fused_graph_us = _bench_batched(candidate_graph.replay)
    return {
        "m": m,
        "hidden_size": hidden_size,
        "n_splits": n_splits,
        "baseline_us": baseline_us,
        "fused_us": fused_us,
        "delta_pct": (fused_us / baseline_us - 1) * 100,
        "baseline_eager_us": baseline_eager_us,
        "fused_eager_us": fused_eager_us,
        "eager_delta_pct": (fused_eager_us / baseline_eager_us - 1) * 100,
        "baseline_graph_us": baseline_graph_us,
        "fused_graph_us": fused_graph_us,
        "graph_delta_pct": (fused_graph_us / baseline_graph_us - 1) * 100,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate and benchmark fused mHC-pre RMSNorm + FP8 quant."
    )
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument(
        "-m",
        type=int,
        nargs="*",
        default=[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
    )
    args = parser.parse_args()
    if args.benchmark:
        rows = [benchmark_mhc_pre_big_fuse_rmsnorm_quant(m) for m in args.m]
        print(pd.DataFrame(rows).to_markdown(index=False))
    else:
        raise SystemExit(pytest.main([__file__, "-q"]))
