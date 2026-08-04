# SPDX-License-Identifier: MIT

"""Correctness and timing harness for persistent BF16 decode GEMM."""

from __future__ import annotations

import argparse
import re
import statistics
from dataclasses import replace
from pathlib import Path

import pytest
import torch

pytest.importorskip("flydsl")

import flydsl.expr as fx

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.gemm_decode import gemm_decode_bf16
from aiter.ops.flydsl.kernels.gemm_decode_persistent import (
    PersistentDecodeConfig,
    compile_gemm_decode_persistent_bf16,
    gemm_decode_persistent_bf16,
    select_persistent_decode_config,
)

pytestmark = pytest.mark.skipif(
    get_gfx() != "gfx950",
    reason="persistent BF16 decode requires gfx950",
)

ATOL = 0.125
RTOL = 0.01
CORRECTNESS_CASES = (
    (1, 1, 1),
    (2, 17, 255),
    (3, 65, 257),
    (4, 128, 896),
    (1, 16384, 7168),
    (4, 16384, 7168),
    (1, 7168, 896),
    (4, 7168, 896),
)
PROVIDERS = ("persistent", "direct", "reference")


def _make_tensors(m: int, n: int, k: int, seed: int = 0):
    torch.manual_seed(seed)
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    c = torch.full((m, n), torch.nan, dtype=torch.bfloat16, device="cuda")
    reference = (a.float() @ b.float().T).bfloat16()
    return a, b, c, reference


def _assert_correct(output, reference):
    assert torch.isfinite(output).all(), "output sentinel was not fully overwritten"
    torch.testing.assert_close(output, reference, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(("m", "n", "k"), CORRECTNESS_CASES)
def test_persistent_bf16_decode(m, n, k):
    a, b, c, reference = _make_tensors(m, n, k)
    num_cus = torch.cuda.get_device_properties(0).multi_processor_count
    gemm_decode_persistent_bf16(a, b, c, m, n, k, num_cus)
    torch.cuda.synchronize()
    _assert_correct(c, reference)


def test_persistent_bf16_decode_is_deterministic():
    m, n, k = 4, 65, 257
    a, b, c, reference = _make_tensors(m, n, k)
    num_cus = torch.cuda.get_device_properties(0).multi_processor_count
    first = None
    for _ in range(3):
        c.fill_(torch.nan)
        gemm_decode_persistent_bf16(a, b, c, m, n, k, num_cus)
        torch.cuda.synchronize()
        _assert_correct(c, reference)
        if first is None:
            first = c.clone()
        else:
            assert torch.equal(c, first)


def test_persistent_config_rejects_unsafe_lds_residency():
    config = PersistentDecodeConfig(workgroups_per_cu=4)
    with pytest.raises(ValueError, match="LDS capacity"):
        config.validate(m=4, k=7168)


def test_persistent_wrapper_validates_tensor_contract():
    m, n, k = 1, 33, 128
    a, b, c, _ = _make_tensors(m, n, k)
    num_cus = torch.cuda.get_device_properties(0).multi_processor_count
    with pytest.raises(ValueError, match="B must have shape"):
        gemm_decode_persistent_bf16(
            a,
            b[:-1],
            c,
            m,
            n,
            k,
            num_cus,
        )


@pytest.mark.parametrize("cache_modifier", [0, 0x2000])
def test_persistent_b_cache_modifier(cache_modifier):
    m, n, k = 1, 33, 128
    a, b, c, reference = _make_tensors(m, n, k)
    num_cus = torch.cuda.get_device_properties(0).multi_processor_count
    config = PersistentDecodeConfig(b_cache_modifier=cache_modifier)
    gemm_decode_persistent_bf16(
        a,
        b,
        c,
        m,
        n,
        k,
        num_cus,
        config=config,
    )
    torch.cuda.synchronize()
    _assert_correct(c, reference)


def test_persistent_geometry_graph_stream_and_ir_contracts():
    m, n, k = 1, 33, 256
    a, b, c, reference = _make_tensors(m, n, k)
    num_cus = torch.cuda.get_device_properties(0).multi_processor_count
    config = PersistentDecodeConfig(
        waves_per_workgroup=4,
        columns_per_wave=2,
        workgroups_per_cu=4,
        waves_per_eu=1,
        b_cache_modifier=0x2000,
    )
    launcher = compile_gemm_decode_persistent_bf16(
        m,
        n,
        k,
        num_cus,
        config,
    )
    current = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    side.wait_stream(current)

    launcher(a, b, c, stream=fx.Stream(side))
    side.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        launcher(a, b, c, stream=fx.Stream(side))
    side.synchronize()

    c.fill_(torch.nan)
    side.wait_stream(current)
    with torch.cuda.stream(side):
        graph.replay()
    current.wait_stream(side)
    current.synchronize()
    _assert_correct(c, reference)

    artifact = launcher._last_compiled[1]
    source_ir = artifact.source_ir
    assert launcher.kernel_name in source_ir
    assert source_ir.count("rocdl.waves_per_eu") == 1
    assert "rocdl.flat_work_group_size" not in source_ir
    attr_offset = source_ir.index("rocdl.waves_per_eu")
    assert source_ir.rfind("gpu.func", 0, attr_offset) > source_ir.rfind(
        "gpu.launch_func",
        0,
        attr_offset,
    )
    assert re.search(
        r"rocdl\.raw\.ptr\.buffer\.load[^\n]*%c8192",
        source_ir,
    )


def _load_reference_op(library):
    if not hasattr(torch.ops, "_rocm_C") or not hasattr(torch.ops._rocm_C, "wvSplitK"):
        if library is None:
            raise ValueError("reference provider requires --reference-library")
        path = Path(library)
        if not path.is_file():
            raise ValueError(f"reference library does not exist: {path}")
        torch.ops.load_library(str(path))
    return torch.ops._rocm_C.wvSplitK


def _prepare_provider(provider, a, b, c, m, n, k, config, reference_library):
    if provider == "persistent":
        num_cus = torch.cuda.get_device_properties(0).multi_processor_count
        launcher = compile_gemm_decode_persistent_bf16(m, n, k, num_cus, config)

        def run(stream=None):
            launcher(a, b, c, stream=fx.Stream(stream))
            return c

        return run, launcher
    if provider == "direct":

        def run(stream=None):
            gemm_decode_bf16(a, b, c, m, n, k, stream=fx.Stream(stream))
            return c

        return run, None
    if provider == "reference":
        op = _load_reference_op(reference_library)
        num_cus = torch.cuda.get_device_properties(0).multi_processor_count

        def run(_stream=None):
            return op(b, a, None, num_cus)

        return run, None
    raise ValueError(f"unknown provider: {provider}")


def _event_samples(run, warmup, repeat, batches=5):
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    samples = []
    for _ in range(batches):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(repeat):
            run()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000.0 / repeat)
    return samples


def _graph_samples(run, warmup, repeat, batches=5):
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(side):
        run(side)
        side.synchronize()
        with torch.cuda.graph(graph, stream=side):
            for _ in range(repeat):
                run(side)
    torch.cuda.current_stream().wait_stream(side)
    samples = []
    for _ in range(batches):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000.0 / repeat)
    return samples


def _format_samples(samples):
    return (
        f"median={statistics.median(samples):.3f} us "
        f"spread=[{min(samples):.3f}, {max(samples):.3f}] us"
    )


def _benchmark(args):
    for n, k in args.shape:
        for m in args.m:
            config = (
                select_persistent_decode_config(m, n, k)
                if args.auto_config
                else PersistentDecodeConfig(
                    waves_per_workgroup=args.waves,
                    columns_per_wave=args.columns,
                    workgroups_per_cu=args.workgroups_per_cu,
                    waves_per_eu=args.waves_per_eu,
                    b_cache_modifier=(
                        0x2000
                        if args.b_cache_modifier is None
                        else args.b_cache_modifier
                    ),
                )
            )
            if args.auto_config and args.b_cache_modifier is not None:
                config = replace(
                    config,
                    b_cache_modifier=args.b_cache_modifier,
                )
            a, b, c, reference = _make_tensors(m, n, k)
            for provider in args.providers:
                c.fill_(torch.nan)
                run, launcher = _prepare_provider(
                    provider,
                    a,
                    b,
                    c,
                    m,
                    n,
                    k,
                    config,
                    args.reference_library,
                )
                output = run()
                torch.cuda.synchronize()
                _assert_correct(output, reference)
                eager = _event_samples(run, args.warmup, args.repeat)
                graph_text = ""
                if args.graph:
                    graph_samples = _graph_samples(run, args.warmup, args.repeat)
                    graph_text = f" graph_{_format_samples(graph_samples)}"
                metadata = ""
                if launcher is not None:
                    metadata = (
                        f" grid={launcher.grid_workgroups}"
                        f" lds={launcher.lds_bytes}"
                        f" kernel={launcher.kernel_name}"
                    )
                print(
                    f"m={m} n={n} k={k} provider={provider} "
                    f"eager_{_format_samples(eager)}{graph_text}{metadata}"
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-M", dest="m", type=int, nargs="+", default=[1, 4])
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        action="append",
        default=[],
        metavar=("N", "K"),
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=PROVIDERS,
        default=list(PROVIDERS),
    )
    parser.add_argument("--waves", type=int, choices=(4, 8, 16), default=16)
    parser.add_argument("--columns", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument(
        "--auto-config",
        action="store_true",
        help="use the measured per-shape persistent config selector",
    )
    parser.add_argument(
        "--workgroups-per-cu",
        type=int,
        choices=(1, 2, 4),
        default=1,
    )
    parser.add_argument("--waves-per-eu", type=int, default=0)
    parser.add_argument(
        "--b-cache-modifier",
        type=lambda value: int(value, 0),
        default=None,
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--reference-library")
    args = parser.parse_args()
    if not args.shape:
        args.shape = [(7168, 896)]
    _benchmark(args)


if __name__ == "__main__":
    main()
