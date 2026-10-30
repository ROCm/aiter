# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Check Gram decode, unchanged fallbacks, and complete-call graph timing."""

import argparse
import gc
import json
import os
import statistics
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
from test_mhc_post_pre_gfx950 import error_metrics, inputs, reference

from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops import mhc as mhc_api
from aiter.ops.mhc import mhc_fused_post_pre, mhc_shuffle_fn
from aiter.test_common import assertAllclose, run_perftest


def check(actual, expected):
    """Assert the pinned native ratio policy; retain strict errors as diagnostics."""
    assert len(actual) == len(expected) == 4
    for i, (got, want) in enumerate(zip(actual, expected)):
        assert torch.isfinite(got).all(), f"non-finite output {i}"
        assert torch.isfinite(want).all(), f"non-finite reference output {i}"
        assert got.shape == want.shape and got.dtype == want.dtype
        # Use the native mHC tolerance, including its 5% outlier allowance.
        assertAllclose(
            want, got, rtol=1e-2, atol=1e-2, tol_err_ratio=0.05, msg=f"output {i}"
        )
    errors = error_metrics(actual, expected)
    if any(item["all_element_mismatches_actual_expected"] for item in errors.values()):
        print("reference_all_element_diagnostic=" + json.dumps(errors))


def parent_dispatch():
    return patch("aiter.ops.mhc._mhc_gram_gfx950_supported", return_value=False)


def check_with_parent(actual, expected, args, kwargs, eligible=True):
    with parent_dispatch():
        parent = mhc_fused_post_pre(*args, **kwargs)
    # Retain all error fields before a possible assertion failure.
    print(
        "mhc_gram_numerical="
        + json.dumps(
            {
                "candidate_vs_reference": error_metrics(actual, expected),
                "parent_vs_reference": error_metrics(parent, expected),
                "candidate_vs_parent": error_metrics(actual, parent),
            }
        ),
        flush=True,
    )
    for got, want in zip(actual, parent):
        assert torch.isfinite(got).all() and torch.isfinite(want).all()
        assert got.shape == want.shape and got.dtype == want.dtype
    if eligible:
        check(actual, expected)
    else:
        for got, want in zip(actual, parent):
            torch.testing.assert_close(got, want, rtol=0, atol=0)


def qualify_benchmark_case(m, seed=0):
    for scaled in (False, True):
        args, kwargs = inputs(m, seed=seed)
        if scaled:
            args[4].mul_((4 * args[0].shape[-1]) ** -0.5)
            args[5].mul_(10.0)
        check_with_parent(
            mhc_fused_post_pre(*args, **kwargs),
            reference(*args, **kwargs),
            args,
            kwargs,
            eligible=(m == 64),
        )


def test_case(seed, scaled):
    args, kwargs = inputs(64, seed=seed)
    if scaled:
        args[4].mul_((4 * args[0].shape[-1]) ** -0.5)
        args[5].mul_(10.0)
    assert mhc_api._mhc_gram_gfx950_supported(*args, kwargs["norm_weight"])
    saved = [t.clone() for t in (*args, kwargs["norm_weight"])]
    with patch(
        "aiter.ops.mhc._mhc_gram_gfx950", wraps=mhc_api._mhc_gram_gfx950
    ) as route:
        output = mhc_fused_post_pre(*args, **kwargs)
        assert route.call_count == 1
    check_with_parent(output, reference(*args, **kwargs), args, kwargs)
    for actual, expected in zip((*args, kwargs["norm_weight"]), saved):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    held = [t.clone() for t in output]
    args[4].mul_(0.5)
    args[0].add_(0.1)
    changed = dict(
        kwargs,
        hc_post_mult_value=2.0,
        rms_eps=3e-6,
        hc_pre_eps=4e-6,
        hc_sinkhorn_eps=5e-6,
        norm_eps=2e-6,
    )
    check_with_parent(
        mhc_fused_post_pre(*args, **changed), reference(*args, **changed), args, changed
    )
    post_2d = (*args[:2], args[2].squeeze(-1), *args[3:])
    check_with_parent(
        mhc_fused_post_pre(*post_2d, **kwargs),
        reference(*args, **kwargs),
        post_2d,
        kwargs,
    )
    for actual, expected in zip(output, held):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
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
    check_with_parent(captured, reference(*args, **kwargs), args, kwargs)
    print(f"seed={seed}, scaled={scaled}: correctness/state/graph PASS")


def check_fallback(args, kwargs):
    with parent_dispatch():
        try:
            expected = mhc_fused_post_pre(*args, **kwargs)
        except ValueError as error:
            expected_error = str(error)
        else:
            expected_error = None
    with patch(
        "aiter.ops.mhc._mhc_gram_gfx950", wraps=mhc_api._mhc_gram_gfx950
    ) as route:
        if expected_error is not None:
            try:
                mhc_fused_post_pre(*args, **kwargs)
            except ValueError as error:
                assert str(error) == expected_error
            else:
                raise AssertionError("expected the original ValueError")
        else:
            actual = mhc_fused_post_pre(*args, **kwargs)
            for got, want in zip(actual, expected):
                assert torch.isfinite(got).all() and torch.isfinite(want).all()
                torch.testing.assert_close(got, want, rtol=0, atol=0)
        assert route.call_count == 0


def test_fallbacks():
    args, kwargs = inputs(64)
    for updates in (
        {"sinkhorn_repeat": 1},
        {"sinkhorn_repeat": 7},
        {"force_fused": True},
        {"norm_weight": None},
        {"backend": "gluon"},
    ):
        check_fallback(args, dict(kwargs, **updates))
    packed = (*args[:4], mhc_shuffle_fn(args[4]), *args[5:])
    check_fallback(packed, dict(kwargs, w_preshuffle_bf16=True))
    strided_fn = args[4].T.contiguous().T
    # Upstream's fused producer requires row-contiguous fn and aborts on this
    # layout. Check only that the new selector rejects it; do not call parent.
    assert not mhc_api._mhc_gram_gfx950_supported(
        *args[:4], strided_fn, *args[5:], kwargs["norm_weight"]
    )
    for m, hidden, dtype in (
        (16, 7168, torch.bfloat16),
        (64, 4096, torch.bfloat16),
        (64, 7168, torch.float16),
        (8192, 7168, torch.bfloat16),
    ):
        fallback_args, fallback_kwargs = inputs(m, hidden, dtype)
        check_fallback(fallback_args, fallback_kwargs)
    for count in (255, 304):
        with patch("aiter.ops.mhc.get_cu_num", return_value=count):
            assert not mhc_api._mhc_gram_gfx950_supported(*args, kwargs["norm_weight"])
    for name in ("HSA_CU_MASK", "ROC_GLOBAL_CU_MASK"):
        with patch.dict(os.environ, {name: ""}):
            assert not mhc_api._mhc_gram_gfx950_supported(*args, kwargs["norm_weight"])
    print("fallback shape/dtype/layout/precision/repeat/no-norm/hardware PASS")


def test_native_input_checks():
    args, kwargs = inputs(64)
    device = args[0].device
    native = [
        torch.empty((64, 4, 1), device=device),
        torch.empty((64, 4, 4), device=device),
        torch.empty_like(args[0]),
        torch.empty_like(args[1]),
        torch.empty((56, 64, 48), device=device),
        args[0],
        args[1],
        args[2].squeeze(-1),
        args[3],
        args[4],
        args[5],
        args[6],
        kwargs["norm_weight"],
    ]
    for index, wrong in (
        (5, torch.empty((64, 7168), dtype=torch.bfloat16)),
        (10, torch.empty((4,), device=device)),
        (4, torch.empty((56, 64, 47), device=device)),
    ):
        invalid = list(native)
        invalid[index] = wrong
        try:
            mhc_api.mhc_post_pre_gram_gfx950(*invalid, 1e-6, 1e-6, 1e-6, 1e-6, 1.0)
        except RuntimeError:
            pass
        else:
            raise AssertionError("invalid direct binding input was accepted")
    print("direct binding invalid device/shape exceptions PASS")


def test_small_values():
    for value in (0.0, 1e-6):
        args, kwargs = inputs(64)
        args[0].fill_(value)
        args[1].fill_(value)
        kwargs.update(rms_eps=3e-6, norm_eps=2e-6)
        check_with_parent(
            mhc_fused_post_pre(*args, **kwargs),
            reference(*args, **kwargs),
            args,
            kwargs,
        )
    print("zero/near-zero finite RMSNorm PASS")


def test_cancellation(seed):
    for amplitude in (1.0, 16.0, 256.0):
        for gate_logits in (
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 1e-4, 0.0, -1e-4),
            (0.0, 0.0, 0.0, -1e-6),
        ):
            for perturb in (False, True):
                args, kwargs = inputs(64, seed=seed)
                x, residual, post, comb, fn, scale, base = args
                x.zero_()
                post.zero_()
                comb.copy_(torch.eye(4, device=comb.device).expand_as(comb))
                fn.zero_()
                scale.zero_()
                base.zero_()
                base[:4].copy_(base.new_tensor(gate_logits))
                kwargs["norm_weight"].fill_(1.0)
                residual[:, 0].mul_(amplitude)
                residual[:, 1].copy_(-residual[:, 0])
                residual[:, 2].mul_(amplitude)
                residual[:, 3].copy_(-residual[:, 2])
                if perturb:
                    residual[:, 3, ::257].add_(amplitude / 128)
                expected = reference(*args, **kwargs)
                for packed in (False, True):
                    call_args = (
                        *args[:4],
                        mhc_shuffle_fn(fn) if packed else fn,
                        *args[5:],
                    )
                    options = dict(kwargs, backend="hip", w_preshuffle_bf16=packed)
                    with patch(
                        "aiter.ops.mhc._mhc_gram_gfx950", wraps=mhc_api._mhc_gram_gfx950
                    ) as route:
                        actual = mhc_fused_post_pre(*call_args, **options)
                        assert route.call_count == int(not packed)
                    print(
                        json.dumps(
                            {
                                "seed": seed,
                                "amplitude": amplitude,
                                "gate_logits": gate_logits,
                                "perturb": perturb,
                                "packed": packed,
                            }
                        ),
                        flush=True,
                    )
                    check_with_parent(
                        actual, expected, call_args, options, eligible=not packed
                    )
    print("cancellation oracle and packed fallback checks PASS")


def test_cancellation_boundary(seed):
    """Check near the detector threshold, real projections and finite-input overflow."""
    cases = (
        [
            (amplitude, delta, eps, False)
            for amplitude in (1.0, 256.0)
            for delta in (1 / 32, 3 / 64, 1 / 16, 5 / 64, 1 / 8)
            for eps in (1e-12, 1e-6)
        ]
        + [
            (amplitude, delta, 1e-6, True)
            for amplitude in (1.0, 16.0)
            for delta in (0.0, 1 / 32, 1 / 16, 1 / 8)
        ]
        + [(2.0**64, 0.0, 1e-6, False), (2.0**70, 0.0, 1e-6, False)]
    )
    for amplitude, delta, eps, projection in cases:
        args, kwargs = inputs(64, seed=seed)
        x, residual, post, comb, fn, scale, base = args
        x.zero_()
        post.zero_()
        comb.copy_(torch.eye(4, device=comb.device).expand_as(comb))
        scale.fill_(1.0 if projection else 0.0)
        base.zero_()
        if projection:
            fn.mul_((4 * x.shape[-1]) ** -0.5)
            fn[1].copy_(fn[0])
            fn[3].copy_(fn[2])
        else:
            fn.zero_()
        # Exact BF16 amplitudes with alternating signs. delta straddles the
        # q/diagonal=1e-3 threshold; tied gate pairs retain a real projection.
        residual[:, 0].fill_(amplitude)
        residual[:, 0, 1::2].neg_()
        residual[:, 1].copy_(-residual[:, 0])
        residual[:, 2].copy_(residual[:, 0])
        residual[:, 3].copy_(-residual[:, 0] * (1.0 - delta))
        kwargs["norm_weight"].fill_(1.0)
        kwargs["norm_eps"] = eps
        options = dict(kwargs, backend="hip", force_fused=False)
        with patch(
            "aiter.ops.mhc._mhc_gram_gfx950", wraps=mhc_api._mhc_gram_gfx950
        ) as route:
            actual = mhc_fused_post_pre(*args, **options)
            assert route.call_count == 1
        print(
            json.dumps(
                {
                    "seed": seed,
                    "amplitude": amplitude,
                    "residual_delta": delta,
                    "norm_eps": eps,
                    "nonzero_projection": projection,
                }
            ),
            flush=True,
        )
        check_with_parent(actual, reference(*args, **kwargs), args, options)
    print("cancellation boundary/projection/overflow numerical checks PASS")


def benchmark_cases(seeds, output_path):
    """Use AITER's event timer on a graph of 101 complete public calls."""
    records = []
    summaries = []
    for m in (64, 8192):
        for scaled in (False, True):
            paired = []
            for seed in seeds:
                args, kwargs = inputs(m, seed=seed)
                if scaled:
                    args[4].mul_((4 * args[0].shape[-1]) ** -0.5)
                    args[5].mul_(10.0)
                expected = reference(*args, **kwargs)
                with parent_dispatch():
                    parent_output = mhc_fused_post_pre(*args, **kwargs)
                check_with_parent(
                    mhc_fused_post_pre(*args, **kwargs),
                    expected,
                    args,
                    kwargs,
                    eligible=(m == 64),
                )
                for round_id in range(3):
                    timings = {"parent": [], "gram": []}
                    for position, arm in enumerate(
                        ("parent", "gram", "gram", "parent")
                    ):
                        guard = parent_dispatch() if arm == "parent" else nullcontext()
                        gc.collect()
                        was_enabled = gc.isenabled()
                        gc.disable()
                        try:
                            with guard:
                                mhc_fused_post_pre(*args, **kwargs)
                                torch.cuda.synchronize()
                                graph = torch.cuda.CUDAGraph()
                                with torch.cuda.graph(graph):
                                    for _ in range(101):
                                        output = mhc_fused_post_pre(*args, **kwargs)
                                _, batch_us = run_perftest(
                                    graph.replay,
                                    num_iters=101,
                                    num_warmup=2,
                                    num_rotate_args=1,
                                    use_cuda_event=True,
                                )
                                torch.cuda.synchronize()
                                if arm == "gram" and m == 64:
                                    check(output, expected)
                                else:
                                    for got, want in zip(output, parent_output):
                                        assert torch.isfinite(got).all()
                                        torch.testing.assert_close(
                                            got, want, rtol=0, atol=0
                                        )
                                del graph
                        finally:
                            if was_enabled:
                                gc.enable()
                        us = float(batch_us) / 101
                        assert us > 0 and us < float("inf")
                        timings[arm].append(us)
                        records.append(
                            {
                                "m": m,
                                "scaled": scaled,
                                "seed": seed,
                                "round": round_id,
                                "position": position,
                                "arm": arm,
                                "mean_us": us,
                            }
                        )
                    arm_us = {
                        arm: statistics.median(values)
                        for arm, values in timings.items()
                    }
                    paired.append(
                        dict(**arm_us, ratio=arm_us["parent"] / arm_us["gram"])
                    )
            summary = {
                "m": m,
                "scaled": scaled,
                "rmsnorm": True,
                "parent_us": statistics.median(p["parent"] for p in paired),
                "gram_us": statistics.median(p["gram"] for p in paired),
                "paired_ratio": statistics.median(p["ratio"] for p in paired),
                "ratio_min": min(p["ratio"] for p in paired),
                "ratio_max": max(p["ratio"] for p in paired),
            }
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    output_path.write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "hip": torch.version.hip,
                "comparator": "same compiled stack with only Gram dispatch disabled",
                "timer": "mean of 101 device-event graph replays, each containing 101 calls",
                "samples": records,
                "summaries": summaries,
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--benchmark-json", type=Path, help="also run the paired graph benchmark"
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
        print("SKIP: requires an unmasked 256-CU gfx950 device")
        return
    for seed in args.seed:
        for scaled in (False, True):
            test_case(seed, scaled)
    test_fallbacks()
    test_small_values()
    for seed in args.seed:
        test_cancellation(seed)
        test_cancellation_boundary(seed)
    test_native_input_checks()
    qualify_benchmark_case(8192)
    if args.benchmark_json:
        benchmark_cases(args.seed, args.benchmark_json)


if __name__ == "__main__":
    main()
