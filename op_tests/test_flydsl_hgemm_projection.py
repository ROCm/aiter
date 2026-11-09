# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools

import aiter
import pandas as pd
import torch
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.gemm_a16w16_policy import get_flydsl_a16w16_configs
from aiter.ops.flydsl.gemm_a16w16_projection import projection_config
from aiter.ops.flydsl.gemm_kernels import (
    flydsl_hgemm,
    flydsl_hgemm_kernel_name,
    get_flydsl_hgemm_kernel_params,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.tuned_gemm import flydsl_gemm

# M/N/K are the supplied dense benchmark's latent_down, latent_up and fused
# router/latent_down projections. A[M,K], W[N,K], model-owned Y[M,N].
_BASELINES = {
    (256, 3584, 7168): (128, 64, 64, 3, 4, 2, 1, 4, 0),
    (256, 7168, 3584): (128, 64, 64, 4, 4, 2, 1, 1, 4),
    (256, 4480, 7168): (64, 80, 128, 2, 2, 1, 2, 2, 0),
}


def run_torch(a, w, dtype):
    return (a.float() @ w.float().t()).to(dtype)


def _check(ref, actual, label):
    err = checkAllclose(
        ref.float(),
        actual.float(),
        rtol=0.05,
        atol=0.05,
        tol_err_ratio=0.05,
        msg=label,
    )
    assert err <= (0.05 if label == "flydsl_ft" else 0.0)
    assert torch.isfinite(actual).all()
    rms = (
        (actual.float() - ref.float()).square().mean() / ref.float().square().mean()
    ).sqrt()
    assert rms < 0.008
    return err


@benchmark()
def test_projection(m, n, k, dtype, out_dtype):
    config = projection_config(m, n, k)
    assert config is not None
    catalog = get_flydsl_a16w16_configs(m, n, k, dtype, out_dtype, False)
    assert config in catalog
    name = flydsl_hgemm_kernel_name(
        dtype=dtype,
        out_dtype=out_dtype,
        config=config,
        has_bias=False,
        target_gfx="gfx950",
    )
    parsed = get_flydsl_hgemm_kernel_params(name)
    assert parsed is not None and parsed["policy"] == "pr"
    kwargs = {
        key: value
        for key, value in config.items()
        if key != "use_half_tile_interleaved"
    }
    assert all(parsed[key] == value for key, value in kwargs.items())

    torch.manual_seed(2026)
    a = torch.randn((m, k), device="cuda", dtype=dtype)
    w = torch.randn((n, k), device="cuda", dtype=dtype)
    guard = torch.full((m * n + 32,), 8192, device="cuda", dtype=out_dtype)
    y = guard[16:-16].view(m, n)
    ref = run_torch(a, w, out_dtype)
    fields = (
        "block_m",
        "block_n",
        "block_k",
        "stages",
        "m_waves",
        "n_waves",
        "k_waves",
        "split_k",
        "k_stagger",
    )
    old = dict(zip(fields, _BASELINES[m, n, k]))
    # Exercise the same tuned dispatcher the model uses, not only the kernel API.
    dispatched = flydsl_gemm(
        a, w, 0, otype=out_dtype, out=y, config={"kernelName": name}
    )
    _check(ref, dispatched, "tuned projection dispatcher")
    candidates = {
        "flydsl_ft": lambda: flydsl_hgemm(a, w, out=y, out_dtype=out_dtype, **old),
        "flydsl_projection": lambda: flydsl_hgemm(
            a, w, out=y, out_dtype=out_dtype, **kwargs
        ),
    }
    for _ in range(20):
        for fn in candidates.values():
            fn()
    torch.cuda.synchronize()
    flops = 2 * m * n * k
    nbytes = (m * k + n * k) * a.element_size() + m * n * y.element_size()
    ret = {"gfx": get_gfx()}
    for label, fn in candidates.items():
        y.fill_(float("nan"))
        out, us = run_perftest(fn)
        err = _check(ref, out, label)
        assert out.data_ptr() == y.data_ptr()
        assert (guard[:16] == 8192).all() and (guard[-16:] == 8192).all()
        ret[f"{label} us"] = us
        ret[f"{label} TFLOPS"] = flops / us / 1e6
        ret[f"{label} TB/s"] = nbytes / us / 1e6
        ret[f"{label} err"] = err

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        candidates["flydsl_projection"]()
    for seed in (37, 123, 593):
        torch.manual_seed(seed)
        a.normal_()
        w.normal_()
        ref = run_torch(a, w, out_dtype)
        y.fill_(float("nan"))
        for _ in range(5):
            graph.replay()
        _check(ref, y, "changed-input graph")
        assert (guard[:16] == 8192).all() and (guard[-16:] == 8192).all()

    # Explicit stream and concurrent graphs must not alias split-K partials.
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    inputs = [(a.clone(), w.clone()), (-a, w.clone())]
    outputs = [torch.empty_like(y), torch.empty_like(y)]
    graphs = []
    for st, (aa, ww), yy in zip(streams, inputs, outputs):
        st.wait_stream(torch.cuda.current_stream())
        flydsl_hgemm(aa, ww, out=yy, stream=st, **kwargs)
        st.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=st):
            flydsl_hgemm(aa, ww, out=yy, stream=st, **kwargs)
        graphs.append(g)
    for _ in range(20):
        for st, g in zip(streams, graphs):
            with torch.cuda.stream(st):
                g.replay()
    torch.cuda.synchronize()
    for (aa, ww), yy in zip(inputs, outputs):
        _check(run_torch(aa, ww, out_dtype), yy, "concurrent stream graphs")

    cold_stream = torch.cuda.Stream()
    cold_stream.wait_stream(torch.cuda.current_stream())
    cold_y = torch.empty_like(y)
    cold_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cold_graph, stream=cold_stream):
        # The executable is warm, but this stream's scratch is first allocated
        # during capture and must not alias graph-pool intermediates.
        flydsl_hgemm(a, w, out=cold_y, stream=cold_stream, **kwargs)
    with torch.cuda.stream(cold_stream):
        cold_graph.replay()
    cold_stream.synchronize()
    _check(ref, cold_y, "capture-first workspace")

    # Preserve public HGEMM strided-input normalization, not an idealized layout.
    aa = a.t().contiguous().t()
    ww = w.t().contiguous().t()
    flydsl_hgemm(aa, ww, out=y, **kwargs)
    _check(ref, y, "strided inputs")
    return ret


def main():
    if (
        get_gfx() != "gfx950"
        or torch.cuda.get_device_properties(0).multi_processor_count != 256
    ):
        aiter.logger.warning("projection catalog requires gfx950 with 256 CUs; skip")
        return
    parser = argparse.ArgumentParser(description="FlyDSL BF16 dense projection")
    parser.add_argument(
        "-d", "--dtype", type=dtypes.str2Dtype, nargs="+", default=[dtypes.bf16]
    )
    parser.add_argument(
        "--out-dtype", type=dtypes.str2Dtype, nargs="+", default=[dtypes.bf16]
    )
    parser.add_argument(
        "-s", "--mnk", type=dtypes.str2tuple, nargs="+", default=list(_BASELINES)
    )
    args = parser.parse_args()
    rows = []
    for shape, dtype, out_dtype in itertools.product(
        args.mnk, args.dtype, args.out_dtype
    ):
        if dtype != dtypes.bf16 or out_dtype != dtypes.bf16 or shape not in _BASELINES:
            parser.error(f"unsupported projection case: {shape} {dtype} {out_dtype}")
        rows.append(test_projection(*shape, dtype, out_dtype))
    aiter.logger.info(
        "FlyDSL projection summary (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
