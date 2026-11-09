# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness + perf for the FlyDSL FP8 block-scale B-preshuffle GEMM (gfx942).

``aiter.gemm_a8w8_blockscale_bpreshuffle`` runs this kernel when a tuned CSV row has
``libtype`` flydsl:

    x        : [M, K]        fp8
    w        : [N, K]        fp8, shuffle_weight(w, layout=(16, 16))
    x_scale  : [K/128, M]    fp32 bytes, K-major (shape [M, K/128] view)
    w_scale  : [ceil(N/128), K/128] fp32
    y        : [M, N]        bf16 / fp16

Each case is a kernel on a shape, run through
``flydsl_gemm_a8w8_blockscale_bpreshuffle`` three ways: K-major scale bytes viewed as
[M, K/128] (``wrapper``), the scales as a stride-(1, M) view (``wrapper strided
scale``), and fp8 passed as uint8 (``wrapper uint8``). It also runs through the
public op (``dispatch``), with a tuned row naming the kernel injected in place of the
shipped CSV. Each candidate writes into its own dirty output buffer, so unwritten
output is caught. The first case is also captured into a CUDA graph and replayed
twice, each replay into a dirty output. The run fails if any result differs from the
reference.

Run:
    python op_tests/test_gemm_a8w8_blockscale_bpreshuffle_flydsl.py
    python op_tests/test_gemm_a8w8_blockscale_bpreshuffle_flydsl.py -s 4,2048,5120 -d bf16
"""

import argparse
from unittest import mock

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops import gemm_op_a8w8
from aiter.ops.flydsl.gemm_kernels import flydsl_gemm_a8w8_blockscale_bpreshuffle
from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_blockscale_bpreshuffle_common import (
    OFFSET_LIMIT,
    kernelInstance,
    offsets_fit,
    parse_kernel_name,
)
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942"]
# Arches whose dispatch reaches the FlyDSL branch, so the CK-fallback check has a path
# to test; the check mocks the arch test itself.
DISPATCH_GFX = ["gfx942", "gfx950"]
SEED = 0
BLOCK = 128

# (m, n, k, kernel).
_CASES = [
    # Decode split-K; each split runs two k tiles, so one step and no loop.
    (4, 2048, 5120, kernelInstance(16, 64, 256, split_k=10)),
    (32, 4352, 5120, kernelInstance(16, 128, 256, split_k=4)),
    # The unsplit 16-row tile, which issues its loads beside the first MFMAs.
    (4, 5120, 768, kernelInstance(16, 64, 256)),
    # Async copy with the A scales staged in LDS.
    (256, 5120, 768, kernelInstance(32, 64, 256, use_async_copy=True)),
    (2048, 5120, 768, kernelInstance(64, 128, 128, use_cshuffle_epilog=True)),
    # M not a multiple of the tile.
    (3307, 5120, 2176, kernelInstance(64, 256, 128, use_cshuffle_epilog=True)),
    (2048, 4352, 5120, kernelInstance(64, 256, 128)),
    (512, 2048, 5120, kernelInstance(64, 256, 128, num_waves=8)),
    # Split-K on a tile taller than 16 rows: k offsets via soffset from a nonzero
    # k_start, and an odd step count so the loop and the peeled step both run.
    (256, 2048, 5120, kernelInstance(64, 128, 128, split_k=4)),
    # A single k tile: no k loop.
    (64, 2048, 128, kernelInstance(64, 256, 128)),
    # N not a multiple of 128: the last N tile reads a half-used B scale block.
    (4, 2112, 1024, kernelInstance(16, 64, 256, split_k=2)),
    (96, 2112, 1024, kernelInstance(32, 64, 128, use_cshuffle_epilog=True)),
]


def run_torch(x, w, x_scale, w_scale, dtype):
    # x_scale is [M, K/128] here; w_scale is [ceil(N/128), K/128].
    n = w.shape[0]
    xf = x.to(dtypes.fp32) * x_scale.repeat_interleave(BLOCK, dim=1)
    ws = w_scale.repeat_interleave(BLOCK, dim=0)[:n].repeat_interleave(BLOCK, dim=1)
    return (xf @ (w.to(dtypes.fp32) * ws).t()).to(dtype)


def _make_inputs(m, n, k):
    torch.manual_seed(SEED)
    x = (torch.randn(m, k) * 0.5).to(dtypes.fp8)
    w = (torch.randn(n, k) * 0.5).to(dtypes.fp8)
    x_scale = torch.rand(m, k // BLOCK) + 0.5
    w_scale = torch.rand(-(-n // BLOCK), k // BLOCK) + 0.5
    return x, w, x_scale, w_scale


@benchmark()
def test_gemm(m, n, k, dtype, kernel_name):
    ki = parse_kernel_name(kernel_name)

    x, w, x_scale, w_scale = _make_inputs(m, n, k)
    ref = run_torch(x, w, x_scale, w_scale, dtype)

    w_shuf = shuffle_weight(w, layout=(16, 16))
    # K-major bytes seen as [M, K/128], as the model hands them over.
    x_scale_km = x_scale.t().contiguous().view(m, k // BLOCK)
    # The same bytes as a stride-(1, M) [M, K/128] view.
    x_scale_strided = x_scale.t().contiguous().t()

    def _dirty_out():
        # Dirty preallocated output, as the model passes a buffer it owns.
        return torch.randn(m, n).to(dtype)

    def _wrapper(xq, wq, xs):
        out = _dirty_out()
        # run_perftest reuses this buffer every iteration and the check reads the last
        # launch, so split-K counters left non-zero by any launch give a wrong result.
        return lambda: flydsl_gemm_a8w8_blockscale_bpreshuffle(
            xq,
            wq,
            xs,
            w_scale,
            out,
            ki.tile_m,
            ki.tile_n,
            ki.tile_k,
            use_cshuffle_epilog=ki.use_cshuffle_epilog,
            use_async_copy=ki.use_async_copy,
            num_waves=ki.num_waves,
            split_k=ki.split_k,
        )

    candidates = {
        "wrapper": _wrapper(x, w_shuf, x_scale_km),
        "wrapper strided scale": _wrapper(x, w_shuf, x_scale_strided),
        "wrapper uint8": _wrapper(
            x.view(torch.uint8), w_shuf.view(torch.uint8), x_scale_km
        ),
    }
    dispatch_row = {"libtype": "flydsl", "kernelName": kernel_name, "splitK": 0}
    dispatch_out = _dirty_out()

    def _dispatch():
        with mock.patch.object(
            gemm_op_a8w8, "get_CKGEMM_config", return_value=dispatch_row
        ):
            return aiter.gemm_a8w8_blockscale_bpreshuffle(
                x, w_shuf, x_scale_km, w_scale, dtype, out=dispatch_out
            )

    candidates["dispatch"] = _dispatch

    flops = 2 * m * n * k
    # Input bytes only, as test_gemm_a8w8_blockscale.py counts them.
    nbytes = x.nbytes + w.nbytes
    ret = {
        "tile": f"{ki.tile_m}x{ki.tile_n}x{ki.tile_k}",
        "split_k": ki.split_k,
    }
    for name, fn in candidates.items():
        y, us = run_perftest(fn)
        err = checkAllclose(
            ref.to(dtypes.fp32),
            y.to(dtypes.fp32),
            rtol=2e-2,
            atol=2e-2,
            msg=f"{name}: flydsl blockscale bpreshuffle",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def check_split_k_graph_replay(dtype=dtypes.bf16):
    m, n, k, ki = _CASES[0]
    x, w, x_scale, w_scale = _make_inputs(m, n, k)
    ref = run_torch(x, w, x_scale, w_scale, dtype)
    w_shuf = shuffle_weight(w, layout=(16, 16))
    x_scale_km = x_scale.t().contiguous().view(m, k // BLOCK)
    out = torch.empty(m, n, dtype=dtype)

    def _launch():
        flydsl_gemm_a8w8_blockscale_bpreshuffle(
            x,
            w_shuf,
            x_scale_km,
            w_scale,
            out,
            ki.tile_m,
            ki.tile_n,
            ki.tile_k,
            split_k=ki.split_k,
        )

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        _launch()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        _launch()
    for replay in range(2):
        out.copy_(torch.randn(m, n).to(dtype))
        graph.replay()
        torch.cuda.synchronize()
        err = checkAllclose(
            ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=2e-2,
            atol=2e-2,
            msg=f"split-K graph replay {replay}: flydsl blockscale bpreshuffle",
        )
        assert err == 0, f"split-K graph replay {replay} differs from the reference"


def check_split_k_first_call_in_capture_raises():
    """A stream's first split-K call inside a capture raises."""
    m, n, k, ki = _CASES[0]
    x, w, x_scale, w_scale = _make_inputs(m, n, k)
    w_shuf = shuffle_weight(w, layout=(16, 16))
    x_scale_km = x_scale.t().contiguous().view(m, k // BLOCK)
    out = torch.empty(m, n, dtype=dtypes.bf16)
    stream = torch.cuda.Stream()
    try:
        with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=stream):
            flydsl_gemm_a8w8_blockscale_bpreshuffle(
                x,
                w_shuf,
                x_scale_km,
                w_scale,
                out,
                ki.tile_m,
                ki.tile_n,
                ki.tile_k,
                split_k=ki.split_k,
            )
    except RuntimeError as e:
        assert "graph capture" in str(e), f"unexpected error: {e}"
    else:
        raise AssertionError("first split-K call inside a capture did not raise")


def check_unservable_flydsl_rows_fall_back_to_ck():
    """A flydsl row the FlyDSL kernels cannot serve goes to CK, not to a FlyDSL function.

    Unservable: a kernelName that is neither ``flydsl_blockscale_8w_`` nor
    ``flydsl_blockscale_bpreshuffle_``, the latter on an arch other than gfx942, or a
    shape past the kernel's 32-bit offsets, which the padded lookup can map onto a
    FlyDSL row.
    """
    n, k = 128, 128
    kernel = _CASES[0][3]
    tail_m = OFFSET_LIMIT // n - 1
    assert offsets_fit(tail_m, n, k, 1) and not offsets_fit(tail_m, n, k, kernel.tile_m)
    # label: (M, kernelName, arch supported)
    cases = {
        "unknown name": (16, "flydsl_something_else_16x64x256", True),
        "bpreshuffle name on another arch": (16, kernel.name, False),
        "tail rows past 32-bit offsets": (tail_m, kernel.name, True),
    }
    w = torch.empty(n, k, dtype=dtypes.fp8)
    w_scale = torch.empty(n // BLOCK, k // BLOCK)
    for label, (m, kernel_name, arch_supported) in cases.items():
        # Stride-0 rows: the full shape reaches dispatch without the memory.
        x = torch.empty(1, k, dtype=dtypes.fp8).expand(m, k)
        x_scale = torch.empty(1, k // BLOCK).expand(m, k // BLOCK)
        out = torch.empty(1, n, dtype=dtypes.bf16).expand(m, n)
        row = {"libtype": "flydsl", "kernelName": kernel_name, "splitK": 0}
        fail = mock.Mock(side_effect=AssertionError(f"{label}: went to FlyDSL"))
        ck = mock.Mock(side_effect=lambda x, w, xs, ws, y, **_: y)
        with (
            mock.patch.object(gemm_op_a8w8, "get_CKGEMM_config", return_value=row),
            mock.patch.object(
                gemm_op_a8w8,
                "_flydsl_blockscale_supported",
                mock.Mock(return_value=arch_supported),
            ),
            mock.patch.object(gemm_op_a8w8, "gemm_a8w8_blockscale_flydsl", fail),
            mock.patch.object(
                gemm_op_a8w8, "gemm_a8w8_blockscale_bpreshuffle_flydsl", fail
            ),
            mock.patch.object(gemm_op_a8w8, "gemm_a8w8_blockscale_bpreshuffle_ck", ck),
        ):
            aiter.gemm_a8w8_blockscale_bpreshuffle(
                x, w, x_scale, w_scale, dtypes.bf16, out=out
            )
        assert ck.call_count == 1, f"{label}: CK was called {ck.call_count} times"


def main():
    case_shapes = sorted({case[:3] for case in _CASES})
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp16"]],
        nargs="*",
        default="bf16,fp16",
        metavar="{bf16, fp16}",
        help="""Output data type.
        e.g.: -d bf16""",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=case_shapes,
        help=f"""Run only the cases on these shapes, from: {case_shapes}
        e.g.:   -s 4,2048,5120""",
    )
    args = parser.parse_args()
    wanted = set(args.mnk)
    if wanted - set(case_shapes):
        parser.error(f"no case on {sorted(wanted - set(case_shapes))}")

    if get_gfx() in DISPATCH_GFX:
        check_unservable_flydsl_rows_fall_back_to_ck()
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "flydsl blockscale bpreshuffle unsupported on %s; skipping", get_gfx()
        )
        return
    check_split_k_graph_replay()
    check_split_k_first_call_in_capture_raises()

    failures = []
    for dtype in args.dtype:
        rows = [
            test_gemm(m, n, k, dtype, ki.name)
            for m, n, k, ki in _CASES
            if (m, n, k) in wanted
        ]
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "flydsl blockscale bpreshuffle %s summary (markdown):\n%s",
            dtype,
            df.to_markdown(index=False),
        )
        err_cols = [c for c in df.columns if c.endswith(" err")]
        bad = df[(df[err_cols].fillna(0) != 0).any(axis=1)]
        failures += [
            f"{dtype} {r.m}x{r.n}x{r.k} {r.kernel_name}" for r in bad.itertuples()
        ]
    assert not failures, "results differ from the reference:\n" + "\n".join(failures)


if __name__ == "__main__":
    main()
