# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""G2L correctness + perf using the actual grouped-MoE helper and caller.

    python op_tests/test_moe_g2l_lut.py --shapes 512,128 513,128 8192,1024 8193,1024
    python op_tests/test_moe_g2l_lut.py --shapes 8192,1024 8193,8192 \
        -d i32 i64 bool fp32 --stride 1 2

N is expert_mask.numel(), E is the number of local experts/counter slots.
8192 global experts plus a trailing dropped sentinel means N=8193.
The shapes cover the requested 8192-expert workload, representative EP shard
sizes, and synthetic dispatch boundaries; they are not a full-model benchmark.

run_torch is an untimed oracle. The timed torch_fallback candidate is the
production fallback implementation, not the oracle. TFLOPS is zero for this
integer scan. TB/s is effective *logical* input/output traffic, not measured
DRAM bandwidth (nor the sum of intermediates in the multi-kernel fallback).
Timing uses run_perftest's profiler-based kernel durations, with optional graph
replay; it is not directly comparable to whole-graph CUDA-event wall time.
Specialized pytest tests cover both backends, reset ownership, exceptions,
and dynamic graphs:
    op_tests/triton_tests/moe/test_g2l_lut_large.py
"""

import argparse
import functools
import itertools
import os
from contextlib import contextmanager

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest

# Same auxiliary-kernel architectures as test_flydsl_moe_aux.py. This is one
# helper-level test, not a per-arch GEMM test; the wrapper chooses the backend.
SUPPORTED_GFX = ("gfx942", "gfx950", "gfx1250")
MASK_DTYPES = (torch.int32, torch.int64, torch.bool, torch.float32)
DEFAULT_SHAPES = [
    (32, 8),
    (64, 16),
    (128, 32),
    (256, 64),
    (511, 128),
    (512, 128),
    (513, 128),
    (640, 160),
    (768, 192),
    (1023, 256),
    (1024, 256),
    (1025, 256),
    (2048, 512),
    (4096, 1024),
    (8191, 1024),
    (8192, 128),
    (8192, 256),
    (8192, 1024),
    (8192, 8192),
    (8193, 128),
    (8193, 256),
    (8193, 1024),
    (8193, 8192),
    (16383, 1024),
    (16384, 1024),
    (16385, 1024),
]


def run_torch(mask, E, nvt, topk):
    """Untimed torch reference; compare !=0 before casting the mask."""
    enabled = mask.reshape(-1) != 0
    prefix = torch.cumsum(enabled.to(torch.int32), dim=0, dtype=torch.int32)
    lut = torch.where(enabled, prefix - 1, E).to(torch.int32)
    counter = torch.zeros(E, dtype=torch.int32, device=mask.device)
    nvr = nvt.reshape(-1)[:1].to(torch.int32) * topk
    return lut, counter, nvr


def check_outputs(ref, out, name):
    """Standard err report AND bit-exact int32 checks, including large nvr."""
    assert len(ref) == len(out) == 3, f"{name}: expected lut, counter, nvr"
    errors = []
    for field, expected, actual in zip(("lut", "counter", "nvr"), ref, out):
        assert actual.dtype == torch.int32 and actual.is_contiguous(), field
        assert actual.shape == expected.shape, field
        # fp32 cannot distinguish neighboring large int32 values. Keep this
        # assertion on the original integers even though the report uses fp32.
        assert torch.equal(expected, actual), f"{name}: {field} int32 mismatch"
        err = checkAllclose(
            expected.to(dtypes.fp32),
            actual.to(dtypes.fp32),
            rtol=0,
            atol=0,
            tol_err_ratio=0,
            msg=f"{name}: {field}",
        )
        assert err == 0, f"{name}: {field} err={err}"
        errors.append(err)
    return max(errors)


@contextmanager
def _backend_env(force_torch):
    """Change dispatch only for this candidate; preserve the caller's env."""
    values = {
        "AITER_G2L_TORCH": "1" if force_torch else "0",
        "AITER_G2L_TRITON": "1",
    }
    saved = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_grouped(mask, E, nvt, topk, *, expect_fused):
    """The production helper plus the nvr/counter handling in its callers."""
    from aiter.ops.flydsl import grouped_moe_gfx1250 as grouped

    lut, counter, built_nvr = grouped._build_g2l_lut(
        mask, E, mask.device, nvt=nvt, topk=topk
    )
    # Do not report a failed fused kernel as a successful torch performance row.
    assert (built_nvr is not None) == expect_fused, "unexpected G2L nvr fallback"
    assert (counter is not None) == expect_fused, "unexpected G2L counter fallback"
    nvr = built_nvr if built_nvr is not None else (nvt * topk).contiguous()
    if counter is None:
        counter = torch.zeros(E, dtype=torch.int32, device=mask.device)
    return lut, counter, nvr


@benchmark()
def test_moe_g2l_lut(N, E, dtype, stride, tokens, topk, execution):
    from aiter.ops.flydsl import grouped_moe_gfx1250 as grouped
    from aiter.ops.triton.moe.g2l_lut import MAX_G2L_EXPERTS

    assert dtype in MASK_DTYPES and stride >= 1
    assert 1 <= E <= N
    assert 0 <= tokens <= 2**31 - 1 and 0 <= topk <= 2**31 - 1
    assert tokens * topk <= 2**31 - 1
    assert execution in ("eager", "graph")
    assert grouped._flydsl_dispatch_context() is None, "run outside a live EP dispatch"

    # Create the model-like 0/1 mask as a real strided view. E is exactly the
    # enabled count; leave the final entry as a dropped sentinel when E < N.
    host = torch.zeros(N, dtype=dtype, device="cpu")
    step = max(1, (N - 1) // E)
    host[torch.arange(E, device="cpu") * step] = 1
    backing = torch.empty(N * stride, dtype=dtype, device="cuda")
    mask = backing[::stride]
    mask.copy_(host)
    nvt = torch.tensor([tokens], dtype=torch.int32, device=mask.device)
    ref = run_torch(mask, E, nvt, topk)
    extent = max(N, E)
    auto_backend = (
        "flydsl"
        if extent <= grouped._G2L_MAX_N
        else "triton" if extent <= MAX_G2L_EXPERTS else "torch"
    )
    candidates = {
        "auto": (
            functools.partial(
                run_grouped,
                mask,
                E,
                nvt,
                topk,
                expect_fused=auto_backend != "torch",
            ),
            False,
        ),
        # This is the actual legacy fallback under test, not run_torch().
        "torch_fallback": (
            functools.partial(run_grouped, mask, E, nvt, topk, expect_fused=False),
            True,
        ),
    }
    # Integer scan, stores, and an integer multiply: no floating-point FLOPs.
    flops = 0
    nbytes = (
        mask.numel() * mask.element_size() + sum(x.nbytes for x in ref) + nvt.nbytes
    )
    ret = {"gfx": get_gfx(), "auto_backend": auto_backend}
    for name, (fn, force_torch) in candidates.items():
        with _backend_env(force_torch):
            # Compile/warm before graph capture; check the real route eagerly.
            check_outputs(ref, fn(), f"{name} warmup")
            # Bounded graph size for these very small kernels. Use AITER's
            # standard profiler-based timing, not an ad-hoc event benchmark.
            out, us = run_perftest(
                fn, num_iters=33, num_rotate_args=1, testGraph=execution == "graph"
            )
            assert us > 0, f"{name}: missing GPU timing"
            err = check_outputs(ref, out, name)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


# The CLI supplies this function's shape arguments; it is not a pytest fixture
# test. Existing pytest modules retain graph, fallback, and precise edge checks.
test_moe_g2l_lut.__test__ = False


def _mask_dtype(value):
    # dtypes.str2Dtype has no bool spelling; all other types use its parser.
    dtype = torch.bool if value == "bool" else dtypes.str2Dtype(value)
    if dtype not in MASK_DTYPES:
        raise argparse.ArgumentTypeError("mask dtype must be i32, i64, bool, or fp32")
    return dtype


def _shape(value):
    shape = dtypes.str2tuple(value)
    if not isinstance(shape, tuple) or len(shape) != 2 or not 1 <= shape[1] <= shape[0]:
        raise argparse.ArgumentTypeError("shape must be N,E with 1 <= E <= N")
    return shape


def main():
    # Gate before the sweep, not inside @benchmark (which would emit empty rows).
    if not torch.cuda.is_available():
        aiter.logger.warning("G2L requires a GPU; skipping")
        return
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("G2L unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    parser.add_argument(
        "-d", "--dtype", type=_mask_dtype, nargs="+", default=[dtypes.i32]
    )
    parser.add_argument(
        "-s",
        "--shapes",
        type=_shape,
        nargs="+",
        default=DEFAULT_SHAPES,
        help="N,E pairs: N=global mask length (incl. sentinel), E=local experts",
    )
    parser.add_argument("--stride", type=int, nargs="+", default=[1])
    parser.add_argument("--tokens", type=int, nargs="+", default=[4096])
    parser.add_argument("--topk", type=int, nargs="+", default=[8])
    parser.add_argument(
        "--execution", choices=["eager", "graph"], nargs="+", default=["graph"]
    )
    args = parser.parse_args()
    if any(s < 1 for s in args.stride):
        parser.error("stride must be >= 1")
    if any(t < 0 for t in args.tokens) or any(
        k < 0 or k > 2**31 - 1 for k in args.topk
    ):
        parser.error("tokens/topk must be nonnegative; topk must fit int32")
    if any(
        t > 2**31 - 1 or t * k > 2**31 - 1
        for t, k in itertools.product(args.tokens, args.topk)
    ):
        parser.error("tokens and tokens*topk must fit int32")
    rows = [
        test_moe_g2l_lut(N, E, dtype, stride, tokens, topk, execution)
        for (N, E), dtype, stride, tokens, topk, execution in itertools.product(
            args.shapes, args.dtype, args.stride, args.tokens, args.topk, args.execution
        )
    ]
    aiter.logger.info(
        "moe_g2l_lut summary (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
