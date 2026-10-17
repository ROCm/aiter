# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.quant import dynamic_per_tensor_quant, static_per_tensor_quant
from aiter.test_common import benchmark, checkAllclose, run_perftest


def make_output(shape, dtype, layout):
    count = shape[0] * shape[1]
    storage = torch.full((2 * count + 64,), 16, dtype=dtype, device="cuda")
    offset = 17 if layout == "unaligned" else 16
    flat = storage[offset : offset + count]
    if layout == "strided":
        out = storage[offset : offset + 2 * count : 2].view(shape)
    elif layout == "transposed":
        out = flat.view(shape[1], shape[0]).t()
    else:
        out = flat.view(shape)
    scale = torch.full((1,), 99.0, dtype=torch.float32, device="cuda")
    return storage, out, scale, offset


def make_input(shape, dtype, pattern):
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=dtype, device="cuda")
    if pattern == "finite":
        x[-1, -1] = -20.0
    elif pattern == "zero":
        x.zero_()
    elif pattern == "all_nan":
        x.fill_(float("nan"))
    elif pattern == "nan_first":
        x[0, 0] = float("nan")
    elif pattern == "nan_last":
        x[-1, -1] = float("nan")
    elif pattern == "positive_inf":
        x[-1, -1] = float("inf")
    elif pattern == "negative_inf":
        x[-1, -1] = -float("inf")
    elif pattern == "mixed_inf":
        x[0, 0], x[-1, -1] = float("inf"), -float("inf")
    else:
        raise ValueError(f"Unknown input pattern: {pattern}")
    return x


def assert_exact(actual, reference, count):
    storage, _, scale, offset = actual
    ref_storage, _, ref_scale, ref_offset = reference
    assert torch.equal(scale.view(torch.int32), ref_scale.view(torch.int32))
    # The legacy conversion writes flat bytes even for noncontiguous outputs.
    # Compare backing storage, without claiming general strided-output support.
    actual_bytes = storage[offset : offset + 2 * count + 16].view(torch.uint8)
    reference_bytes = ref_storage[ref_offset : ref_offset + 2 * count + 16].view(
        torch.uint8
    )
    assert torch.equal(actual_bytes, reference_bytes)
    assert bool((storage[:offset].float() == 16).all())
    assert bool((storage[offset + count :].float() == 16).all())


def assert_dispatch(x, out, scale, partial):
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        dynamic_per_tensor_quant(out, x, scale)
        torch.cuda.synchronize()
    names = [
        event.name
        for event in profile.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    assert any("data_to_scale_kernel" in name for name in names), names
    assert sum("reduce_per_tensor_scale" in name for name in names) == int(
        partial
    ), names
    assert sum("initializeScale" in name for name in names) == int(not partial), names


def update_reference(x, pattern, reference):
    storage, ref_out, ref_scale, _ = reference
    storage.fill_(16)
    ref_scale.fill_(99.0)
    if pattern in ("finite", "zero"):
        output_dtype = ref_out.dtype
        limit = (
            torch.finfo(output_dtype).max
            if output_dtype.is_floating_point
            else torch.iinfo(output_dtype).max
        )
        inverse_limit = torch.tensor(1.0 / limit, dtype=torch.float32, device="cuda")
        ref_scale.copy_((x.abs().amax().float() * inverse_limit).reshape(1))
        static_per_tensor_quant(ref_out, x, ref_scale)
    else:
        # NaN/Inf checks establish parity with the forced atomic scale path.
        # They do not define an independent numerical policy for nonfinite input.
        dynamic_per_tensor_quant(ref_out, x, ref_scale)


def assert_changed_input_graph(x, actual, reference):
    storage, out, scale, _ = actual
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            dynamic_per_tensor_quant(out, x, scale)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        dynamic_per_tensor_quant(out, x, scale)
    torch.cuda.current_stream().wait_stream(stream)
    for iteration in range(5):
        x.mul_(0.875)
        x[-1, -1] = -21.0 - iteration
        update_reference(x, "finite", reference)
        storage.fill_(16)
        scale.fill_(99.0)
        graph.replay()
        torch.cuda.synchronize()
        assert_exact(actual, reference, x.numel())


@benchmark()
def test_per_tensor_quant_exact(m, n, input_dtype, output_dtype, pattern, layout):
    x = make_input((m, n), input_dtype, pattern)
    reference = make_output(x.shape, output_dtype, "unaligned")
    update_reference(x, pattern, reference)
    if pattern not in ("finite", "zero"):
        assert_dispatch(x, reference[1], reference[2], partial=False)

    actual = make_output(x.shape, output_dtype, layout)
    storage, out, scale, _ = actual
    candidates = {"hip": lambda: dynamic_per_tensor_quant(out, x, scale)}
    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        for _ in range(3):
            storage.fill_(16)
            scale.fill_(99.0)
            fn()
            assert_exact(actual, reference, x.numel())
        _, us = run_perftest(fn, use_cuda_event=True)
        assert_exact(actual, reference, x.numel())
        # Compare byte values as fp32 so NaN encodings also have a strict check.
        offset, ref_offset = actual[3], reference[3]
        err = checkAllclose(
            reference[0][ref_offset : ref_offset + x.numel()].view(torch.uint8).float(),
            storage[offset : offset + x.numel()].view(torch.uint8).float(),
            rtol=0,
            atol=0,
            tol_err_ratio=0,
        )
        assert err == 0
        ret[f"{name} us"] = us
        # Effective algorithm rates: abs, max and scale multiply per element;
        # two input reads and one output write. Excludes scratch and scalar work.
        ret[f"{name} TFLOPS"] = 3 * x.numel() / us / 1e6
        ret[f"{name} TB/s"] = (
            x.numel() * (2 * x.element_size() + out.element_size()) / us / 1e6
        )
        ret[f"{name} err"] = err

    if pattern == "finite":
        partial = (
            m >= 1024 and n % 4 == 0 and out.is_contiguous() and out.data_ptr() % 4 == 0
        )
        assert_dispatch(x, out, scale, partial)
        if m == 1024 and n in (20, 520):
            assert_changed_input_graph(x, actual, reference)
    return ret


def main():
    parser = argparse.ArgumentParser(
        description="Exact dynamic per-tensor quantization"
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="+",
        default=[torch.bfloat16, torch.float16],
    )
    parser.add_argument(
        "-q",
        "--quant-dtype",
        choices=["fp8", "int8"],
        nargs="+",
        default=["fp8", "int8"],
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="+",
        default=[
            (21, 3072),
            (1023, 3072),
            (1024, 3072),
            (1025, 3072),
            (511, 3072),
            (512, 3072),
            (513, 3072),
            (4096, 3072),
            (512, 3088),
            (1024, 4),
            (1024, 20),
            (1024, 520),
            (512, 20),
            (512, 520),
        ],
    )
    args = parser.parse_args()
    if get_gfx() not in ("gfx942", "gfx950"):
        aiter.logger.warning(
            "Per-tensor exactness test unsupported on %s; skipping", get_gfx()
        )
        return
    if any(dtype not in (torch.bfloat16, torch.float16) for dtype in args.dtype):
        parser.error("input dtype must be bf16 or fp16")
    if any(len(shape) != 2 or min(shape) <= 0 for shape in args.mnk):
        parser.error("each shape must contain two positive integers")

    output_dtypes = {"fp8": dtypes.fp8, "int8": torch.int8}
    special_patterns = [
        "nan_first",
        "nan_last",
        "all_nan",
        "positive_inf",
        "negative_inf",
        "mixed_inf",
    ]
    suites = {
        "finite": (args.mnk, ["finite", "zero"], ["contiguous", "unaligned"]),
        "nonfinite": (
            [(1024, 20), (1024, 32), (1025, 3088)],
            special_patterns,
            ["contiguous"],
        ),
        "output_layout": (
            [(1024, 20), (1024, 32)],
            ["finite"],
            ["strided", "transposed"],
        ),
    }
    for suite, (shapes, patterns, layouts) in suites.items():
        rows = []
        for (m, n), input_dtype, quant_dtype, pattern, layout in itertools.product(
            shapes, args.dtype, args.quant_dtype, patterns, layouts
        ):
            rows.append(
                test_per_tensor_quant_exact(
                    m, n, input_dtype, output_dtypes[quant_dtype], pattern, layout
                )
            )
        aiter.logger.info(
            "%s exactness summary (%d cases):\n%s",
            suite,
            len(rows),
            pd.DataFrame(rows).to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
