# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Check mHC post/pre values, current inputs, output ownership and graph replay."""

import argparse
import hashlib
import json
import os
import statistics
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch

from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops import mhc as mhc_api
from aiter.ops.mhc import (
    _mhc_post_pre_gfx950_supported,
    mhc_fused_post_pre,
    mhc_shuffle_fn,
)
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


def call_inputs(args, kwargs, packed):
    call_args = (*args[:4], mhc_shuffle_fn(args[4]) if packed else args[4], *args[5:])
    return call_args, dict(kwargs, w_preshuffle_bf16=packed)


def upstream_dispatch():
    return patch(
        "aiter.ops.mhc._mhc_post_pre_gfx950_supported",
        new=lambda *args, **kwargs: False,
    )


def test_case(m, seed, packed):
    args, kwargs = inputs(m, seed=seed)
    call_args, options = call_inputs(args, kwargs, packed)
    assert _mhc_post_pre_gfx950_supported(*call_args, kwargs["norm_weight"], packed)
    saved_inputs = tuple(t.clone() for t in call_args)
    saved_norm = kwargs["norm_weight"].clone()
    out = mhc_fused_post_pre(*call_args, **options)
    check(out, reference(*args, **kwargs))
    saved_out = tuple(t.clone() for t in out)
    for got, want in zip(call_args, saved_inputs):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    torch.testing.assert_close(kwargs["norm_weight"], saved_norm, rtol=0, atol=0)

    # New weights in the same allocation must be read on the next call.
    args[4].mul_(0.5)
    args[0].add_(0.1)
    if packed:
        call_args[4].copy_(mhc_shuffle_fn(args[4]))
    for repeat in (1, 7, 20):
        variants = {
            "sinkhorn_repeat": repeat,
            "hc_post_mult_value": 2.0,
            "rms_eps": 3e-6,
            "norm_eps": 2e-6,
        }
        check(
            mhc_fused_post_pre(*call_args, **options, **variants),
            reference(*args, **kwargs, **variants),
        )
    # Both supported post_mix ranks describe the same values.
    post_2d = (*call_args[:2], call_args[2].squeeze(-1), *call_args[3:])
    check(mhc_fused_post_pre(*post_2d, **options), reference(*args, **kwargs))
    for got, want in zip(out, saved_out):
        torch.testing.assert_close(got, want, rtol=0, atol=0)

    # Capture only after all JIT compilation has finished. Refresh caller-packed
    # weights outside capture, in their original allocation, before replay.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mhc_fused_post_pre(*call_args, **options)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = mhc_fused_post_pre(*call_args, **options)
    args[4].mul_(0.75)
    args[1].add_(0.05)
    if packed:
        call_args[4].copy_(mhc_shuffle_fn(args[4]))
    graph.replay()
    torch.cuda.synchronize()
    check(captured, reference(*args, **kwargs))
    print(f"m={m}, seed={seed}, packed={packed}: correctness/state/graph PASS")


def check_fallback(args, kwargs):
    # Compare existing outputs or ValueError behavior exactly; unsupported Gluon
    # requests can raise, while large-M unfused dispatch precedes backend choice.
    with upstream_dispatch():
        try:
            expected = mhc_fused_post_pre(*args, **kwargs)
        except ValueError as error:
            expected_error = str(error)
        else:
            expected_error = None
    if expected_error is not None:
        try:
            mhc_fused_post_pre(*args, **kwargs)
        except ValueError as error:
            assert str(error) == expected_error
        else:
            raise AssertionError("expected the upstream ValueError")
    else:
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
        assert not _mhc_post_pre_gfx950_supported(*args, kwargs["norm_weight"])
        check_fallback(args, kwargs)
    args, kwargs = inputs(64, seed=seed)
    # Native post/pre uses x.stride(0); the inner stride remains one.
    x = torch.empty((64, 7168 + 8), dtype=args[0].dtype, device="cuda")[:, :7168]
    x.copy_(args[0])
    assert not _mhc_post_pre_gfx950_supported(x, *args[1:], kwargs["norm_weight"])
    check_fallback((x, *args[1:]), kwargs)
    check_fallback(args, {})
    print("fallback shape/dtype/layout/no-norm equivalence PASS")


def test_routing(m, seed):
    args, kwargs = inputs(m, seed=seed)
    for packed in (False, True):
        call_args, options = call_inputs(args, kwargs, packed)
        for force_fused in (False, True):
            for backend in (None, "hip", "gluon"):
                config = dict(options, force_fused=force_fused, backend=backend)
                selected = (
                    not force_fused and backend != "gluon" and (m == 64 or packed)
                )
                with patch(
                    "aiter.ops.mhc._mhc_post_pre_gfx950",
                    wraps=mhc_api._mhc_post_pre_gfx950,
                ) as implementation:
                    if selected:
                        check(
                            mhc_fused_post_pre(*call_args, **config),
                            reference(*args, **kwargs),
                        )
                    else:
                        check_fallback(call_args, config)
                    assert implementation.call_count == int(selected), config
    print(f"m={m}: precision/force_fused/backend routing PASS")


def test_hardware_exclusions(seed):
    args, kwargs = inputs(64, seed=seed)
    # Probe the eligibility predicate only: no unsupported launch is executed.
    with patch("aiter.ops.mhc.get_cu_num", return_value=128):
        assert not _mhc_post_pre_gfx950_supported(*args, kwargs["norm_weight"])
    with patch("aiter.ops.mhc.get_cu_num", return_value=256):
        for name in ("HSA_CU_MASK", "ROC_GLOBAL_CU_MASK"):
            for value in ("", "0x1"):
                with patch.dict(os.environ, {name: value}):
                    assert not _mhc_post_pre_gfx950_supported(
                        *args, kwargs["norm_weight"]
                    )
    print("CU count and mask eligibility exclusions PASS")


def test_small_values(seed):
    for packed in (False, True):
        for amplitude in (0.0, 1e-5):
            args, kwargs = inputs(64, seed=seed)
            args[0].mul_(amplitude)
            args[1].mul_(amplitude)
            args[5].fill_(0.1 if amplitude else 0.0)
            args[6].fill_(0.01 if amplitude else 0.0)
            call_args, options = call_inputs(args, kwargs, packed)
            check(mhc_fused_post_pre(*call_args, **options), reference(*args, **kwargs))
    print("zero/near-zero finite RMSNorm and epsilon checks PASS")


def error_metrics(actual, expected):
    result = {}
    for name, got, want in zip(
        ("post", "comb", "layer_output", "next_residual"), actual, expected
    ):
        delta = (got.float() - want.float()).abs()
        result[name] = {
            "max_abs": delta.max().item(),
            "rms_abs": delta.square().mean().sqrt().item(),
            "max_relative_floor_1e_minus_6": (
                delta / want.float().abs().clamp_min(1e-6)
            )
            .max()
            .item(),
            "outside_public_tolerance": int(
                (delta > 1e-2 + 1e-2 * want.float().abs()).sum().item()
            ),
        }
    return result


def benchmark_case(m, seed, packed, force_fused):
    args, kwargs = inputs(m, seed=seed)
    call_args, options = call_inputs(args, kwargs, packed)
    options.update(force_fused=force_fused, backend="hip")
    expected = reference(*args, **kwargs)
    rows = []
    outputs = {}
    for cycle in range(3):
        order = (
            ("upstream", "optimized") if cycle % 2 == 0 else ("optimized", "upstream")
        )
        for position, implementation in enumerate(order):
            guard = (
                upstream_dispatch() if implementation == "upstream" else nullcontext()
            )
            with guard:
                for _ in range(2):
                    output = mhc_fused_post_pre(*call_args, **options)
                torch.cuda.synchronize()
                check(output, expected)
                if m == 64:
                    # One captured complete public call. Explicit caller weight
                    # prepacking is outside timing for both implementations.
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = mhc_fused_post_pre(*call_args, **options)
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
                        *call_args,
                        **options,
                        num_iters=101,
                        num_warmup=2,
                        num_rotate_args=1,
                        use_cuda_event=True,
                    )
                    method = "cuda_events_graph_disabled"
                torch.cuda.synchronize()
                check(output, expected)
                if cycle == 0:
                    outputs[implementation] = tuple(t.clone() for t in output)
            rows.append(
                {
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
        "mode": ("prepacked_hilo" if packed else "fp32")
        + ("_force_fused" if force_fused else "_default"),
        "flags": {
            "w_preshuffle_bf16": packed,
            "force_fused": force_fused,
            "backend": "hip",
            "res_preshuffle": False,
        },
        "optimized_specialization": not force_fused and (m == 64 or packed),
        "cycles": rows,
        "median_us": medians,
        "speedup": medians["upstream"] / medians["optimized"],
        "latency_reduction_pct": 100 * (1 - medians["optimized"] / medians["upstream"]),
        "reference_errors": {
            name: error_metrics(out, expected) for name, out in outputs.items()
        },
        "optimized_vs_upstream_errors": error_metrics(
            outputs["optimized"], outputs["upstream"]
        ),
    }


def benchmark_cases(ms, seed, output_path):
    records = [
        benchmark_case(m, seed, packed, force_fused)
        for m in ms
        for packed in (False, True)
        for force_fused in ((False, True) if m == 8192 else (False,))
    ]
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
        "cu_count": get_cu_num(),
        "cu_masks": {
            name: os.environ.get(name) for name in ("HSA_CU_MASK", "ROC_GLOBAL_CU_MASK")
        },
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "scope": "same-process current upstream dispatch vs adapted full public call, matched flags per row",
        "input_distribution": "seeded normal synthetic inputs; FP32 scale/base multiplied by 0.1",
        "reference": "PyTorch FP32 math with contract BF16 residual/output rounding",
        "prepacking": "mhc_shuffle_fn outside timing for both implementations in explicit packed mode",
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
    parser.add_argument(
        "-m", type=int, nargs="+", choices=(64, 8192), default=[64, 8192]
    )
    parser.add_argument("--seed", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--benchmark-json",
        type=Path,
        help="also run the paired benchmark and write its raw results",
    )
    args = parser.parse_args()
    if (
        get_gfx_runtime() != "gfx950"
        or get_cu_num() != 256
        or any(
            os.environ.get(name) is not None
            for name in ("HSA_CU_MASK", "ROC_GLOBAL_CU_MASK")
        )
    ):
        print("SKIP: optimized cases require an unmasked 256-CU gfx950 device")
    else:
        for seed in args.seed:
            for m in args.m:
                for packed in ((False, True) if m == 64 else (True,)):
                    test_case(m, seed, packed)
        test_fallbacks(args.seed[0])
        test_hardware_exclusions(args.seed[0])
        for m in args.m:
            test_routing(m, args.seed[0])
        test_small_values(args.seed[0])
        if args.benchmark_json:
            benchmark_cases(args.m, args.seed[0], args.benchmark_json)
