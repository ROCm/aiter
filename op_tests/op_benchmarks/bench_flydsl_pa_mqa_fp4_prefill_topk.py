# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MI355X benchmark for the three FP4 paged-MQA prefill TopK paths.

The benchmark creates the production packed Q/K ABI directly. It never creates
a context-sized BF16/FP32 K tensor or a per-head score reference. Each provider
owns preallocated output and scratch buffers, and latency is measured with HIP
events after JIT preflight.

Examples:
    python op_tests/op_benchmarks/bench_flydsl_pa_mqa_fp4_prefill_topk.py \
        benchmark --rows 32
    python op_tests/op_benchmarks/bench_flydsl_pa_mqa_fp4_prefill_topk.py \
        benchmark --rows 1 8 32 --c4-context-width 65536 196608 262144 \
        --topk 512 1024 --parallel-unit-num 512 --score-batch-chunks 4
    python op_tests/op_benchmarks/bench_flydsl_pa_mqa_fp4_prefill_topk.py \
        correctness --topk 512 1024

Peak memory is the steady-state PyTorch allocator peak while one provider is
active, including the shared packed inputs. ``*_over_inputs`` subtracts the
input-only baseline. JIT compilation and input generation are outside both the
latency and steady-state memory windows.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import math
import statistics
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, fields
from functools import partial
from typing import NamedTuple

import torch

HEADS = 64
HEAD_DIM = 128
KV_BLOCK_SIZE = 64
BLOCK_K = 256
DEFAULT_TILE_TOKENS = 4096
DEFAULT_C4_CONTEXT_WIDTHS = (65536, 196608, 262144)
SUPPORTED_TOPK = (512, 1024)
SUPPORTED_SCORE_BATCH_CHUNKS = (1, 2, 4)
WEIGHT_SCALE = 1.25


@dataclass
class PackedCase:
    q_fp4: torch.Tensor
    q_scale: torch.Tensor
    kv_cache: torch.Tensor
    kv_scale: torch.Tensor
    block_tables: torch.Tensor
    weights: torch.Tensor
    row_to_batch: torch.Tensor
    local_starts: torch.Tensor
    local_ends: torch.Tensor
    context_width: int
    batch_size: int

    @property
    def rows(self) -> int:
        return self.q_fp4.shape[0]

    @property
    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.q_fp4,
            self.q_scale,
            self.kv_cache,
            self.kv_scale,
            self.block_tables,
            self.weights,
            self.row_to_batch,
            self.local_starts,
            self.local_ends,
        )

    @property
    def operands(self) -> tuple[object, ...]:
        return (*self.tensors, self.context_width)


class LegacyResult(NamedTuple):
    values: torch.Tensor
    raw_indices: torch.Tensor


@dataclass
class PreparedPath:
    name: str
    run: Callable[[], object]
    resources: tuple[torch.Tensor, ...]
    scratch_override: torch.Tensor | None = None

    @property
    def nbytes(self) -> int:
        return _tensor_bytes(self.resources)


@dataclass(frozen=True)
class Measurement:
    name: str
    latency_p50_us: float
    latency_mean_us: float
    peak_allocated: int
    peak_reserved: int
    allocated_over_inputs: int
    reserved_over_inputs: int
    transient_allocated: int
    preallocated_path_bytes: int


def _tensor_bytes(tensors: tuple[torch.Tensor, ...]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative integer, got {value}"
        )
    return parsed


def _parallel_unit_num(requested: int, rows: int) -> int:
    parallel_unit_num = requested or max(512, rows)
    if parallel_unit_num < rows:
        raise ValueError(
            f"parallel_unit_num={parallel_unit_num} must be at least rows={rows}"
        )
    return parallel_unit_num


def _require_mi355x() -> torch.device:
    if (
        not torch.cuda.is_available()
        or torch.version.hip is None
        or torch.cuda.device_count() == 0
    ):
        raise SystemExit("This benchmark requires a ROCm MI355X GPU.")
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    arch = str(properties.gcnArchName).split(":")[0]
    if arch != "gfx950":
        raise SystemExit(
            f"This benchmark requires MI355X/gfx950; current device is {arch}."
        )
    print(
        f"device={torch.cuda.get_device_name(device)} arch={arch} "
        f"rocm={torch.version.hip}"
    )
    return device


def _random_u8(
    shape: tuple[int, ...],
    device: torch.device,
    generator: torch.Generator,
    low: int = 0,
    high: int = 256,
) -> torch.Tensor:
    output = torch.empty(shape, dtype=torch.uint8, device=device)
    return output.random_(low, high, generator=generator)


def _make_packed_case(
    rows: int,
    context_width: int,
    batch_size: int,
    device: torch.device,
    seed: int,
    *,
    windows: tuple[list[int], list[int], list[int]] | None = None,
) -> PackedCase:
    if batch_size > rows:
        raise ValueError(f"batch_size={batch_size} must not exceed rows={rows}")

    generator = torch.Generator(device=device).manual_seed(seed)
    blocks_per_sequence = math.ceil(context_width / KV_BLOCK_SIZE)
    num_blocks = batch_size * blocks_per_sequence

    # Every byte is two valid E2M1 nibbles. Scale exponents stay near the E8M0
    # bias so random data remains finite while still exercising packed loads.
    q_fp4 = _random_u8((rows, HEADS, HEAD_DIM // 2), device, generator)
    q_scale = _random_u8((rows, 1, 4, 16, 4), device, generator, 124, 131)
    kv_cache = _random_u8(
        (num_blocks, 1, 4, KV_BLOCK_SIZE, 16),
        device,
        generator,
    )
    kv_scale = _random_u8(
        (num_blocks, 1, 4, KV_BLOCK_SIZE),
        device,
        generator,
        124,
        131,
    )
    block_tables = torch.randperm(
        num_blocks,
        dtype=torch.int32,
        device=device,
        generator=generator,
    ).reshape(batch_size, blocks_per_sequence)
    weights = torch.empty(
        (rows, HEADS),
        dtype=torch.bfloat16,
        device=device,
    ).normal_(mean=0.0, std=0.1, generator=generator)

    if windows is None:
        row_to_batch = torch.arange(rows, dtype=torch.int32, device=device) % batch_size
        local_starts = torch.zeros(rows, dtype=torch.int32, device=device)
        local_ends = torch.full(
            (rows,),
            context_width,
            dtype=torch.int32,
            device=device,
        )
    else:
        batches, starts, ends = windows
        if not (len(batches) == len(starts) == len(ends) == rows):
            raise ValueError("correctness windows must contain one entry per row")
        row_to_batch = torch.tensor(batches, dtype=torch.int32, device=device)
        local_starts = torch.tensor(starts, dtype=torch.int32, device=device)
        local_ends = torch.tensor(ends, dtype=torch.int32, device=device)

    return PackedCase(
        q_fp4=q_fp4,
        q_scale=q_scale,
        kv_cache=kv_cache,
        kv_scale=kv_scale,
        block_tables=block_tables.contiguous(),
        weights=weights,
        row_to_batch=row_to_batch.contiguous(),
        local_starts=local_starts.contiguous(),
        local_ends=local_ends.contiguous(),
        context_width=context_width,
        batch_size=batch_size,
    )


def _prepare_legacy(
    case: PackedCase,
    topk: int,
    parallel_unit_num: int,
) -> PreparedPath:
    from aiter.ops.flydsl import flydsl_pa_mqa_logits_fp4_prefill
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
        CTA_INFO_WIDTH,
        compute_prefill_schedule,
    )
    from aiter.ops.topk import _top_k_per_row_prefill, topk_ob_workspace_size

    device = case.q_fp4.device
    cta_info = torch.empty(
        (parallel_unit_num, CTA_INFO_WIDTH),
        dtype=torch.int32,
        device=device,
    )
    compute_prefill_schedule(
        case.row_to_batch,
        case.local_starts,
        case.local_ends,
        BLOCK_K,
        parallel_unit_num,
        case.context_width,
        cta_info_out=cta_info,
    )
    logits = torch.empty(
        (case.rows, case.context_width),
        dtype=torch.float32,
        device=device,
    )
    indices = torch.empty((case.rows, topk), dtype=torch.int32, device=device)
    values = torch.empty((case.rows, topk), dtype=torch.float32, device=device)
    scratch_size = topk_ob_workspace_size(
        case.rows,
        logits.stride(0),
        topk,
        False,
    )
    topk_scratch = torch.empty(
        max(1, scratch_size),
        dtype=torch.uint8,
        device=device,
    )

    def run() -> LegacyResult:
        flydsl_pa_mqa_logits_fp4_prefill(
            *case.operands,
            weight_scale=WEIGHT_SCALE,
            block_k=BLOCK_K,
            kv_block_size=KV_BLOCK_SIZE,
            parallel_unit_num=parallel_unit_num,
            out=logits,
            cta_info=cta_info,
            n_ctas=parallel_unit_num,
        )
        _top_k_per_row_prefill(
            logits,
            case.local_starts,
            case.local_ends,
            indices,
            values,
            case.rows,
            logits.stride(0),
            logits.stride(1),
            topk,
            topk_scratch,
            True,
        )
        return LegacyResult(values, indices)

    return PreparedPath(
        name="legacy-full-logits",
        run=run,
        resources=(cta_info, logits, indices, values, topk_scratch),
    )


def _prepare_bounded(
    case: PackedCase,
    topk: int,
    tile_tokens: int,
) -> PreparedPath:
    from aiter.ops.flydsl import (
        allocate_fp4_bounded_prefill_topk_workspace,
        flydsl_pa_mqa_topk_fp4_prefill_tiled,
    )
    from aiter.ops.topk import topk_ob_workspace_size

    workspace = allocate_fp4_bounded_prefill_topk_workspace(
        case.rows,
        topk,
        case.q_fp4.device,
        tile_tokens=tile_tokens,
    )
    scratch_size = max(
        topk_ob_workspace_size(case.rows, tile_tokens, topk, False),
        topk_ob_workspace_size(case.rows, 2 * topk, topk, False),
    )
    topk_scratch = torch.empty(
        max(1, scratch_size),
        dtype=torch.uint8,
        device=case.q_fp4.device,
    )

    def run() -> object:
        return flydsl_pa_mqa_topk_fp4_prefill_tiled(
            *case.operands,
            k=topk,
            tile_tokens=tile_tokens,
            weight_scale=WEIGHT_SCALE,
            workspace=workspace,
        )

    workspace_tensors = tuple(
        value
        for field in fields(workspace)
        if isinstance((value := getattr(workspace, field.name)), torch.Tensor)
    )
    return PreparedPath(
        name="bounded-tiled",
        run=run,
        resources=(*workspace_tensors, topk_scratch),
        scratch_override=topk_scratch,
    )


def _prepare_fused(
    case: PackedCase,
    topk: int,
    parallel_unit_num: int,
    score_batch_chunks: int,
) -> PreparedPath:
    from aiter.ops.flydsl import (
        FP4PrefillTopKResult,
        allocate_fp4_prefill_topk_workspace,
        flydsl_candidate_topk_merge,
    )
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
        compile_pa_mqa_logits_fp4_prefill_topk,
        compute_prefill_schedule,
    )
    from aiter.ops.flydsl.mqa_topk_finalize import order_and_map_mqa_topk

    device = case.q_fp4.device
    workspace = allocate_fp4_prefill_topk_workspace(
        case.rows,
        parallel_unit_num,
        topk,
        device,
    )
    output = FP4PrefillTopKResult(
        values=torch.empty((case.rows, topk), dtype=torch.float32, device=device),
        raw_indices=torch.empty(
            (case.rows, topk),
            dtype=torch.int32,
            device=device,
        ),
        physical_indices=torch.empty(
            (case.rows, topk),
            dtype=torch.int32,
            device=device,
        ),
        counts=torch.empty(case.rows, dtype=torch.int32, device=device),
    )

    # Keep schedule construction out of the kernel-only timing, just as the
    # legacy scorer does. Both schedule outputs live in the fused workspace.
    compute_prefill_schedule(
        case.row_to_batch,
        case.local_starts,
        case.local_ends,
        BLOCK_K,
        parallel_unit_num,
        case.context_width,
        cta_info_out=workspace.cta_info,
        row_offsets_out=workspace.row_offsets,
    )
    launcher, _ = compile_pa_mqa_logits_fp4_prefill_topk(
        topk=topk,
        block_k=BLOCK_K,
        kv_block_size=KV_BLOCK_SIZE,
        max_blocks_per_seq=case.block_tables.shape[1],
        num_warps=4,
        heads=HEADS,
        head_dim=HEAD_DIM,
        score_batch_chunks=score_batch_chunks,
    )
    stream = torch.cuda.current_stream(device)

    def run() -> object:
        with torch.cuda.stream(stream):
            launcher(
                workspace.candidate_values,
                workspace.candidate_indices,
                workspace.candidate_counts,
                case.q_fp4,
                case.q_scale,
                case.kv_cache,
                case.kv_scale,
                case.block_tables,
                case.weights,
                workspace.cta_info,
                float(WEIGHT_SCALE),
                parallel_unit_num,
                stream,
            )
            flydsl_candidate_topk_merge(
                workspace.candidate_values,
                workspace.candidate_indices,
                workspace.candidate_counts,
                workspace.row_offsets,
                case.row_to_batch,
                case.block_tables,
                workspace.merge_values,
                workspace.merge_indices,
                workspace.merge_physical_indices,
                output.counts,
                KV_BLOCK_SIZE,
                stream=stream,
            )
            order_and_map_mqa_topk(
                workspace.merge_values,
                workspace.merge_indices,
                output.counts,
                case.local_starts,
                case.local_ends,
                case.row_to_batch,
                case.block_tables,
                output.values,
                output.raw_indices,
                output.physical_indices,
                case.context_width,
                topk,
                KV_BLOCK_SIZE,
            )
        return output

    return PreparedPath(
        name="true-fused",
        run=run,
        resources=(*tuple(workspace), *tuple(output)),
    )


@contextmanager
def _use_preallocated_topk_scratch(
    scratch: torch.Tensor | None,
) -> Iterator[None]:
    if scratch is None:
        yield
        return

    topk_module = importlib.import_module("aiter.ops.topk")
    original = topk_module.get_topk_scratch_workspace

    def get_workspace(device: torch.device, size: int) -> torch.Tensor:
        if torch.device(device) != scratch.device:
            raise ValueError("preallocated TopK scratch is on the wrong device")
        if size > scratch.numel():
            raise ValueError(
                f"TopK requested {size} scratch bytes, only "
                f"{scratch.numel()} were preallocated"
            )
        return scratch

    topk_module.get_topk_scratch_workspace = get_workspace
    try:
        yield
    finally:
        topk_module.get_topk_scratch_workspace = original


def _cleanup_cuda() -> None:
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


def _event_latency(
    run: Callable[[], object],
    iters: int,
) -> tuple[float, float]:
    event_pairs = [
        (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for _ in range(iters)
    ]
    for start, end in event_pairs:
        start.record()
        run()
        end.record()
    torch.cuda.synchronize()
    samples_us = [start.elapsed_time(end) * 1000.0 for start, end in event_pairs]
    return statistics.median(samples_us), statistics.fmean(samples_us)


def _measure_path(
    builder: Callable[[], PreparedPath],
    warmup: int,
    iters: int,
) -> Measurement:
    _cleanup_cuda()
    input_allocated = torch.cuda.memory_allocated()
    input_reserved = torch.cuda.memory_reserved()
    path = builder()

    with _use_preallocated_topk_scratch(path.scratch_override):
        # One explicit preflight absorbs all shape-specific JIT compilation.
        path.run()
        for _ in range(warmup):
            path.run()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        steady_allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        latency_p50_us, latency_mean_us = _event_latency(path.run, iters)
        torch.cuda.synchronize()
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()

    measurement = Measurement(
        name=path.name,
        latency_p50_us=latency_p50_us,
        latency_mean_us=latency_mean_us,
        peak_allocated=peak_allocated,
        peak_reserved=peak_reserved,
        allocated_over_inputs=peak_allocated - input_allocated,
        reserved_over_inputs=peak_reserved - input_reserved,
        transient_allocated=max(0, peak_allocated - steady_allocated),
        preallocated_path_bytes=path.nbytes,
    )
    del path
    _cleanup_cuda()
    return measurement


def _mib(value: int) -> float:
    return value / 2**20


def _print_measurement(
    rows: int,
    context_width: int,
    topk: int,
    parallel_unit_num: int,
    score_batch_chunks: int,
    measurement: Measurement,
    legacy_latency_us: float,
) -> None:
    speedup = legacy_latency_us / measurement.latency_p50_us
    print(
        "RESULT"
        f",rows={rows}"
        f",c4_context_width={context_width}"
        f",topk={topk}"
        f",parallel_unit_num={parallel_unit_num}"
        f",score_batch_chunks={score_batch_chunks}"
        f",provider={measurement.name}"
        f",latency_p50_us={measurement.latency_p50_us:.3f}"
        f",latency_mean_us={measurement.latency_mean_us:.3f}"
        f",speedup_vs_legacy={speedup:.4f}"
        f",peak_allocated_mib={_mib(measurement.peak_allocated):.3f}"
        f",peak_reserved_mib={_mib(measurement.peak_reserved):.3f}"
        f",allocated_over_inputs_mib="
        f"{_mib(measurement.allocated_over_inputs):.3f}"
        f",reserved_over_inputs_mib="
        f"{_mib(measurement.reserved_over_inputs):.3f}"
        f",transient_allocated_mib="
        f"{_mib(measurement.transient_allocated):.3f}"
        f",preallocated_path_mib="
        f"{_mib(measurement.preallocated_path_bytes):.3f}"
    )


def _run_benchmark(args: argparse.Namespace, device: torch.device) -> None:
    print(
        "timing=queued-per-invocation HIP-event p50 "
        "schedule=precomputed-for-legacy-and-true-fused"
    )
    case_index = 0
    for rows in args.rows:
        parallel_unit_num = _parallel_unit_num(args.parallel_unit_num, rows)
        if args.batch_size > rows:
            raise ValueError(
                f"batch_size={args.batch_size} must not exceed rows={rows}"
            )
        for context_width in args.c4_context_width:
            case = _make_packed_case(
                rows,
                context_width,
                args.batch_size,
                device,
                args.seed + case_index,
            )
            case_index += 1
            torch.cuda.synchronize()
            print(
                f"\nshape rows={rows} c4_context_width={context_width} "
                f"batch_size={args.batch_size} pages={case.kv_cache.shape[0]} "
                f"packed_inputs_mib={_mib(_tensor_bytes(case.tensors)):.3f}"
            )

            for topk in args.topk:
                builders = (
                    partial(_prepare_legacy, case, topk, parallel_unit_num),
                    partial(_prepare_bounded, case, topk, args.tile_tokens),
                    partial(
                        _prepare_fused,
                        case,
                        topk,
                        parallel_unit_num,
                        args.score_batch_chunks,
                    ),
                )
                measurements = [
                    _measure_path(builder, args.warmup, args.iters)
                    for builder in builders
                ]
                legacy_latency_us = measurements[0].latency_p50_us
                for measurement in measurements:
                    _print_measurement(
                        rows,
                        context_width,
                        topk,
                        parallel_unit_num,
                        args.score_batch_chunks,
                        measurement,
                        legacy_latency_us,
                    )
                del builders

            del case
            _cleanup_cuda()


def _expected_physical_indices(
    case: PackedCase,
    raw_indices: torch.Tensor,
) -> torch.Tensor:
    valid = raw_indices >= 0
    safe_raw = raw_indices.clamp_min(0)
    batches = case.row_to_batch[:, None].to(torch.int64)
    physical_pages = case.block_tables[
        batches,
        (safe_raw // KV_BLOCK_SIZE).to(torch.int64),
    ]
    physical = physical_pages * KV_BLOCK_SIZE + safe_raw % KV_BLOCK_SIZE
    return torch.where(valid, physical, -1).to(torch.int32)


def _assert_result(
    label: str,
    values: torch.Tensor,
    raw_indices: torch.Tensor,
    physical_indices: torch.Tensor,
    counts: torch.Tensor,
    expected_values: torch.Tensor,
    expected_raw_indices: torch.Tensor,
    expected_physical_indices: torch.Tensor,
    expected_counts: torch.Tensor,
) -> None:
    torch.testing.assert_close(values, expected_values, rtol=0, atol=0)
    torch.testing.assert_close(raw_indices, expected_raw_indices, rtol=0, atol=0)
    torch.testing.assert_close(
        physical_indices,
        expected_physical_indices,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(counts, expected_counts, rtol=0, atol=0)
    print(f"[pass] {label}")


def _run_correctness(args: argparse.Namespace, device: torch.device) -> None:
    rows = 4
    context_width = 5120
    batch_size = 2
    windows = (
        [0, 1, 0, 1],
        [0, 37, 4011, 1700],
        [5120, 777, 5011, 1700],
    )
    parallel_unit_num = _parallel_unit_num(args.parallel_unit_num, rows)
    case = _make_packed_case(
        rows,
        context_width,
        batch_size,
        device,
        args.seed,
        windows=windows,
    )
    print(
        f"correctness rows={rows} context_width={context_width} "
        "reference=small-full-logits (no full-precision Q/K)"
    )

    for topk in args.topk:
        legacy_path = _prepare_legacy(case, topk, parallel_unit_num)
        legacy = legacy_path.run()
        torch.cuda.synchronize()
        expected_values = legacy.values.clone()
        expected_raw_indices = legacy.raw_indices.clone()
        expected_physical_indices = _expected_physical_indices(
            case,
            expected_raw_indices,
        )
        expected_counts = torch.clamp(
            case.local_ends - case.local_starts,
            min=0,
            max=topk,
        )
        del legacy, legacy_path
        _cleanup_cuda()

        bounded_path = _prepare_bounded(case, topk, args.tile_tokens)
        with _use_preallocated_topk_scratch(bounded_path.scratch_override):
            bounded = bounded_path.run()
            torch.cuda.synchronize()
            _assert_result(
                f"bounded-tiled topk={topk}",
                bounded.values,
                bounded.raw_indices,
                bounded.kv_indices,
                bounded.valid_counts,
                expected_values,
                expected_raw_indices,
                expected_physical_indices,
                expected_counts,
            )
        del bounded, bounded_path
        _cleanup_cuda()

        fused_path = _prepare_fused(
            case,
            topk,
            parallel_unit_num,
            args.score_batch_chunks,
        )
        fused = fused_path.run()
        torch.cuda.synchronize()
        _assert_result(
            f"true-fused topk={topk}",
            fused.values,
            fused.raw_indices,
            fused.physical_indices,
            fused.counts,
            expected_values,
            expected_raw_indices,
            expected_physical_indices,
            expected_counts,
        )
        del fused, fused_path
        del expected_values, expected_raw_indices
        del expected_physical_indices, expected_counts
        _cleanup_cuda()


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--topk",
        type=int,
        nargs="+",
        choices=SUPPORTED_TOPK,
        default=list(SUPPORTED_TOPK),
        help="TopK widths to run",
    )
    parser.add_argument(
        "--parallel-unit-num",
        "--parallel_unit_num",
        type=_nonnegative_int,
        default=0,
        help="persistent fused/legacy CTA slots; 0 selects max(512, rows)",
    )
    parser.add_argument(
        "--score-batch-chunks",
        "--score_batch_chunks",
        type=int,
        choices=SUPPORTED_SCORE_BATCH_CHUNKS,
        default=4,
        help="chunks retained per in-kernel fused TopK score batch",
    )
    parser.add_argument(
        "--tile-tokens",
        type=_positive_int,
        default=DEFAULT_TILE_TOKENS,
        help="bounded tiled-path score width",
    )
    parser.add_argument("--seed", type=int, default=2026)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser(
        "benchmark",
        help="compare latency and memory at long C4 widths",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_arguments(benchmark)
    benchmark.add_argument(
        "--rows",
        type=_positive_int,
        nargs="+",
        default=[32],
        help="query-row counts to sweep",
    )
    benchmark.add_argument(
        "--c4-context-width",
        "--context-width",
        "--context_width",
        type=_positive_int,
        nargs="+",
        default=list(DEFAULT_C4_CONTEXT_WIDTHS),
        help="logical C4 context widths to sweep",
    )
    benchmark.add_argument(
        "--batch-size",
        type=_positive_int,
        default=1,
        help="page-table rows; query rows map round-robin",
    )
    benchmark.add_argument(
        "--warmup",
        type=_nonnegative_int,
        default=5,
        help="untimed invocations after one JIT preflight",
    )
    benchmark.add_argument(
        "--iters",
        type=_positive_int,
        default=20,
        help="queued per-invocation HIP-event measurements",
    )

    correctness = subparsers.add_parser(
        "correctness",
        help="compare bounded and fused outputs on a small packed case",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_arguments(correctness)

    args = parser.parse_args()
    args.topk = list(dict.fromkeys(args.topk))
    if args.tile_tokens < max(args.topk) or args.tile_tokens % BLOCK_K:
        parser.error(f"--tile-tokens must be >= max(topk) and divisible by {BLOCK_K}")
    if args.command == "benchmark":
        args.rows = list(dict.fromkeys(args.rows))
        args.c4_context_width = list(dict.fromkeys(args.c4_context_width))
    return args


def main() -> None:
    args = _parse_args()
    device = _require_mi355x()
    with torch.inference_mode():
        if args.command == "benchmark":
            _run_benchmark(args, device)
        else:
            _run_correctness(args, device)


if __name__ == "__main__":
    main()
