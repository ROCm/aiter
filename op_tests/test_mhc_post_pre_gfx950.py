# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Check mHC post/pre values, current inputs, output ownership and graph replay."""

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import statistics
from unittest.mock import patch

import torch

from aiter.ops.mhc import (
    _mhc_post_pre_gfx950_supported,
    mhc_fused_post_pre,
    mhc_pack_fn_gfx950,
    mhc_shuffle_fn,
)
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import checkAllclose, run_perftest


def reference(
    x,
    residual,
    post,
    comb,
    fn,
    scale,
    base,
    norm_weight=None,
    rms_eps=1e-6,
    hc_pre_eps=1e-6,
    hc_sinkhorn_eps=1e-6,
    hc_post_mult_value=1.0,
    sinkhorn_repeat=20,
    norm_eps=1e-6,
):
    """Unfused PyTorch post -> projection -> mixes -> fold -> RMSNorm."""
    next_residual = (
        x.float().unsqueeze(1) * post.reshape(-1, 4, 1)
        + torch.bmm(comb.mT, residual.float())
    ).to(x.dtype)
    flat = next_residual.flatten(1).float()
    mix = (flat @ fn.T) * (flat.square().mean(-1, keepdim=True) + rms_eps).rsqrt()
    mix = (
        mix * torch.cat((scale[0].expand(4), scale[1].expand(4), scale[2].expand(16)))
        + base
    )
    pre = (mix[:, :4].sigmoid() + hc_pre_eps).unsqueeze(-1)
    out_post = (mix[:, 4:8].sigmoid() * hc_post_mult_value).unsqueeze(-1)
    out_comb = mix[:, 8:].reshape(-1, 4, 4).softmax(-1) + hc_sinkhorn_eps
    out_comb = out_comb / (out_comb.sum(-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        out_comb = out_comb / (out_comb.sum(-1, keepdim=True) + hc_sinkhorn_eps)
        out_comb = out_comb / (out_comb.sum(-2, keepdim=True) + hc_sinkhorn_eps)
    out = (next_residual * pre).sum(1)
    if norm_weight is not None:
        out = out * (out.square().mean(-1, keepdim=True) + norm_eps).rsqrt()
        out = out * norm_weight.float()
    # The public HIP API returns BF16 layer output even for FP16 residual input.
    return out_post, out_comb, out.bfloat16(), next_residual


def inputs(m, hidden=7168, dtype=torch.bfloat16, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def rand(shape, dt=torch.float32):
        return torch.randn(shape, device="cuda", dtype=dt, generator=gen)

    args = (
        rand((m, hidden), dtype),
        rand((m, 4, hidden), dtype),
        rand((m, 4, 1)),
        rand((m, 4, 4)),
        rand((24, 4 * hidden)),
        rand((3,)) * 0.1,
        rand((24,)) * 0.1,
    )
    return args, {"norm_weight": rand((hidden,), dtype)}


def check(actual, expected):
    assert len(actual) == len(expected) == 4
    for i, (got, want) in enumerate(zip(actual, expected)):
        assert torch.isfinite(got).all(), f"non-finite output {i}"
        assert got.shape == want.shape and got.dtype == want.dtype
        # AITER's 1e-2 defaults, with zero elements allowed outside tolerance.
        checkAllclose(
            want, got, rtol=1e-2, atol=1e-2, tol_err_ratio=0, msg=f"output {i}"
        )
        torch.testing.assert_close(got, want, rtol=1e-2, atol=1e-2)


def test_case(m, seed):
    args, kwargs = inputs(m, seed=seed)
    assert _mhc_post_pre_gfx950_supported(*args, kwargs["norm_weight"])
    packed = torch.empty_like(args[4], dtype=torch.int32)
    mhc_pack_fn_gfx950(packed, args[4])
    torch.testing.assert_close(
        packed, mhc_shuffle_fn(args[4], arch="gfx950"), rtol=0, atol=0
    )
    saved_inputs = tuple(t.clone() for t in args)
    saved_norm = kwargs["norm_weight"].clone()
    out = mhc_fused_post_pre(*args, **kwargs)
    check(out, reference(*args, **kwargs))
    saved_out = tuple(t.clone() for t in out)
    for got, want in zip(args, saved_inputs):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    torch.testing.assert_close(kwargs["norm_weight"], saved_norm, rtol=0, atol=0)

    # New weights in the same allocation must be read on the next call.
    args[4].mul_(0.5)
    args[0].add_(0.1)
    for repeat in (1, 7, 20):
        options = dict(
            kwargs,
            sinkhorn_repeat=repeat,
            hc_post_mult_value=2.0,
            rms_eps=3e-6,
            norm_eps=2e-6,
        )
        check(mhc_fused_post_pre(*args, **options), reference(*args, **options))
    for got, want in zip(out, saved_out):
        torch.testing.assert_close(got, want, rtol=0, atol=0)

    # Capture only after all JIT compilation has finished.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mhc_fused_post_pre(*args, **kwargs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = mhc_fused_post_pre(*args, **kwargs)
    args[4].mul_(0.75)
    args[1].add_(0.05)
    graph.replay()
    torch.cuda.synchronize()
    check(captured, reference(*args, **kwargs))
    print(f"m={m}, seed={seed}: correctness/state/graph PASS")


def check_fallback(args, kwargs):
    assert not _mhc_post_pre_gfx950_supported(*args, kwargs.get("norm_weight"))
    # Disable the new branch to call the unmodified upstream dispatch. Check its
    # existing behavior rather than impose a new layout or precision contract.
    with patch(
        "aiter.ops.mhc._mhc_post_pre_gfx950_supported",
        new=lambda *args, **kwargs: False,
    ):
        expected = mhc_fused_post_pre(*args, **kwargs)
    actual = mhc_fused_post_pre(*args, **kwargs)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_fallbacks(seed):
    for m, hidden, dtype in (
        (63, 7168, torch.bfloat16),
        (64, 4096, torch.bfloat16),
        (64, 7168, torch.float16),
    ):
        args, kwargs = inputs(m, hidden, dtype, seed)
        check_fallback(args, kwargs)
    args, kwargs = inputs(64, seed=seed)
    # Native post/pre uses x.stride(0); the inner stride remains one.
    x = torch.empty((64, 7168 + 8), dtype=args[0].dtype, device="cuda")[:, :7168]
    x.copy_(args[0])
    check_fallback((x, *args[1:]), kwargs)
    check_fallback(args, {})
    print("fallback shape/dtype/layout/no-norm equivalence PASS")


def test_small_values(seed):
    for amplitude in (0.0, 1e-5):
        args, kwargs = inputs(64, seed=seed)
        args[0].mul_(amplitude)
        args[1].mul_(amplitude)
        args[5].fill_(0.1 if amplitude else 0.0)
        args[6].fill_(0.01 if amplitude else 0.0)
        check(mhc_fused_post_pre(*args, **kwargs), reference(*args, **kwargs))
    print("zero/near-zero finite RMSNorm and epsilon checks PASS")


def benchmark_case(m, seed):
    args, kwargs = inputs(m, seed=seed)
    expected = reference(*args, **kwargs)
    rows = []
    for cycle in range(3):
        order = (
            ("upstream", "optimized") if cycle % 2 == 0 else ("optimized", "upstream")
        )
        for position, implementation in enumerate(order):
            guard = (
                patch(
                    "aiter.ops.mhc._mhc_post_pre_gfx950_supported",
                    new=lambda *args, **kwargs: False,
                )
                if implementation == "upstream"
                else nullcontext()
            )
            with guard:
                for _ in range(2):
                    output = mhc_fused_post_pre(*args, **kwargs)
                torch.cuda.synchronize()
                check(output, expected)
                if m == 64:
                    # One captured public call includes all kernels, weight
                    # packing where applicable, and the four returned outputs.
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = mhc_fused_post_pre(*args, **kwargs)
                    _, us = run_perftest(
                        graph.replay,
                        num_iters=101,
                        num_warmup=2,
                        num_rotate_args=1,
                        use_cuda_event=True,
                    )
                    method = "cuda_graph_events"
                else:
                    output, us = run_perftest(
                        mhc_fused_post_pre,
                        *args,
                        **kwargs,
                        num_iters=101,
                        num_warmup=2,
                        num_rotate_args=1,
                        use_cuda_event=True,
                    )
                    method = "cuda_events_graph_disabled"
                torch.cuda.synchronize()
                check(output, expected)
            rows.append(
                {
                    "m": m,
                    "seed": seed,
                    "cycle": cycle + 1,
                    "position": position + 1,
                    "implementation": implementation,
                    "method": method,
                    "mean_us": float(us),
                }
            )
    medians = {
        name: statistics.median(
            row["mean_us"] for row in rows if row["implementation"] == name
        )
        for name in ("upstream", "optimized")
    }
    return {
        "m": m,
        "seed": seed,
        "cycles": rows,
        "median_us": medians,
        "speedup": medians["upstream"] / medians["optimized"],
        "latency_reduction_pct": 100 * (1 - medians["optimized"] / medians["upstream"]),
    }


def benchmark_cases(ms, seed, output_path):
    records = [benchmark_case(m, seed) for m in ms]
    repo = Path(__file__).resolve().parents[1]
    paths = (
        "aiter/ops/mhc.py",
        "csrc/kernels/mhc_post_pre_gfx950.cu",
        "csrc/kernels/mhc_kernels.cu",
        "csrc/include/mhc.h",
        "csrc/pybind/mhc_pybind.cu",
        "aiter/jit/optCompilerConfig.json",
        "op_tests/test_mhc_post_pre_gfx950.py",
    )
    result = {
        "gpu": torch.cuda.get_device_name(),
        "gfx": get_gfx_runtime(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "scope": "same-process current upstream dispatch vs optimized full public call",
        "input_distribution": "seeded normal synthetic inputs; FP32 scale/base multiplied by 0.1",
        "timing": "3 alternating-order cycles; 101 event-timed calls per cycle; 2 warmups; fixed inputs",
        "files_sha256": {
            name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
            for name in paths
        },
        "results": records,
    }
    Path(output_path).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", type=int, nargs="+", default=[64, 8192])
    parser.add_argument("--seed", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--benchmark-json",
        type=Path,
        help="also run the paired benchmark and write its raw results",
    )
    args = parser.parse_args()
    if get_gfx_runtime() != "gfx950":
        print("SKIP: optimized cases require gfx950")
    else:
        for seed in args.seed:
            for m in args.m:
                test_case(m, seed)
        test_fallbacks(args.seed[0])
        test_small_values(args.seed[0])
        if args.benchmark_json:
            benchmark_cases(args.m, args.seed[0], args.benchmark_json)
