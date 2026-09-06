# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark legacy full-logits TopK against fused FP4 paged-MQA TopK."""

from __future__ import annotations

import argparse
import math
import statistics
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch

from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
    DEFAULT_SCORE_BATCH_CHUNKS,
    SUPPORTED_SCORE_BATCH_CHUNKS,
)

HEADS = 64
HEAD_DIM = 128
KV_BLOCK_SIZE = 64
BLOCK_K = 256
WEIGHT_SCALE = 1.25
WIDTHS = (65536, 196608, 262144)
TOPKS = (512, 1024)


@dataclass(frozen=True)
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
    width: int

    @property
    def operands(self) -> tuple[object, ...]:
        return (*self.tensors, self.width)

    @property
    def tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(v for v in vars(self).values() if torch.is_tensor(v))


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def _tensor_bytes(tensors: Iterable[torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _preflight(rows: int, width: int, topks: Iterable[int], device) -> None:
    from aiter.ops import topk as topk_ops

    pages = math.ceil(width / KV_BLOCK_SIZE)
    input_bytes = rows * (HEADS * HEAD_DIM // 2 + 256 + HEADS * 2 + 12)
    input_bytes += pages * (4096 + 256 + 4)
    parallel_units = max(512, rows)
    required = 0
    for topk in topks:
        output_bytes = rows * (topk * 12 + 4)
        topk_scratch = max(
            1, int(topk_ops.topk_ob_workspace_size(rows, width, topk, False))
        )
        legacy_scratch = rows * width * 4 + topk_scratch
        fused_scratch = (
            parallel_units * (28 + topk * 8) + (rows + 1) * 4 + rows * topk * 8
        )
        required = max(
            required, input_bytes + legacy_scratch + fused_scratch + 2 * output_bytes
        )
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if required > free_bytes:
        raise SystemExit(
            "Insufficient free GPU memory for explicit inputs, "
            "scratch, and outputs: "
            f"required={required} bytes, free={free_bytes} bytes."
        )


def _require_gfx950() -> torch.device:
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise SystemExit("This benchmark requires ROCm on a gfx950 GPU.")
    device = torch.device("cuda", torch.cuda.current_device())
    arch = str(torch.cuda.get_device_properties(device).gcnArchName).split(":")[0]
    if arch != "gfx950":
        raise SystemExit(f"This benchmark requires gfx950; current device is {arch}.")
    return device


def _random_u8(shape, device, generator, low=0, high=256) -> torch.Tensor:
    return torch.empty(shape, dtype=torch.uint8, device=device).random_(
        low, high, generator=generator
    )


def _make_case(rows: int, width: int, device: torch.device, seed: int) -> PackedCase:
    generator = torch.Generator(device=device).manual_seed(seed)
    pages = math.ceil(width / KV_BLOCK_SIZE)
    q_fp4 = _random_u8((rows, HEADS, HEAD_DIM // 2), device, generator)
    q_scale = _random_u8((rows, 1, 4, 16, 4), device, generator, 124, 131)
    kv_cache = _random_u8((pages, 1, 4, 64, 16), device, generator)
    kv_scale = _random_u8((pages, 1, 4, 64), device, generator, 124, 131)
    block_tables = torch.randperm(
        pages, dtype=torch.int32, device=device, generator=generator
    )[None]
    weights = torch.empty((rows, HEADS), dtype=torch.bfloat16, device=device)
    weights.normal_(0.0, 0.1, generator=generator)
    row_to_batch = torch.zeros(rows, dtype=torch.int32, device=device)
    local_starts = torch.zeros(rows, dtype=torch.int32, device=device)
    local_ends = torch.full((rows,), width, dtype=torch.int32, device=device)
    return PackedCase(
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables.contiguous(),
        weights.contiguous(),
        row_to_batch,
        local_starts,
        local_ends,
        width,
    )


def _prepare_legacy(case: PackedCase, topk: int, parallel_units: int):
    from aiter.ops import topk as topk_ops
    from aiter.ops.flydsl import flydsl_pa_mqa_logits_fp4_prefill
    from aiter.ops.flydsl.mqa_topk_finalize import order_and_map_mqa_topk

    rows = case.q_fp4.shape[0]
    device = case.q_fp4.device
    logits = torch.empty((rows, case.width), dtype=torch.float32, device=device)
    selected_raw = torch.empty((rows, topk), dtype=torch.int32, device=device)
    selected_values = torch.empty((rows, topk), dtype=torch.float32, device=device)
    values = torch.empty_like(selected_values)
    raw_indices = torch.empty_like(selected_raw)
    physical_indices = torch.empty_like(selected_raw)
    counts = torch.full((rows,), topk, dtype=torch.int32, device=device)
    scratch_size = topk_ops.topk_ob_workspace_size(rows, case.width, topk, False)
    scratch = topk_ops.get_topk_scratch_workspace(device, scratch_size)
    allocate_scratch = topk_ops.get_topk_scratch_workspace

    def supplied_scratch(request_device, requested_size):
        assert torch.device(request_device) == device
        assert requested_size <= scratch.numel()
        return scratch

    def run() -> None:
        flydsl_pa_mqa_logits_fp4_prefill(
            *case.operands,
            weight_scale=WEIGHT_SCALE,
            block_k=BLOCK_K,
            kv_block_size=KV_BLOCK_SIZE,
            parallel_unit_num=parallel_units,
            out=logits,
        )
        # Public TopK has no scratch argument; this benchmark is isolated-process.
        assert topk_ops.get_topk_scratch_workspace is allocate_scratch
        topk_ops.get_topk_scratch_workspace = supplied_scratch
        try:
            topk_ops.top_k_per_row_prefill(
                logits,
                case.local_starts,
                case.local_ends,
                selected_raw,
                selected_values,
                rows,
                logits.stride(0),
                logits.stride(1),
                topk,
                stable=True,
            )
        finally:
            unchanged = topk_ops.get_topk_scratch_workspace is supplied_scratch
            topk_ops.get_topk_scratch_workspace = allocate_scratch
            assert unchanged, "TopK scratch allocator changed during benchmark"
        order_and_map_mqa_topk(
            selected_values,
            selected_raw,
            counts,
            case.local_starts,
            case.local_ends,
            case.row_to_batch,
            case.block_tables,
            values,
            raw_indices,
            physical_indices,
            case.width,
            topk,
            KV_BLOCK_SIZE,
        )

    output = (values, raw_indices, physical_indices, counts)
    scratch_bytes = _tensor_bytes((logits, selected_values, selected_raw, scratch))
    return run, output, scratch_bytes, _tensor_bytes(output)


def _prepare_fused(case: PackedCase, topk: int, parallel_units: int, chunks: int):
    from aiter.ops.flydsl import (
        FP4PrefillTopKResult,
        allocate_fp4_prefill_topk_workspace,
        flydsl_pa_mqa_topk_fp4_prefill,
    )

    rows = case.q_fp4.shape[0]
    workspace = allocate_fp4_prefill_topk_workspace(
        rows, parallel_units, topk, case.q_fp4.device
    )
    device = case.q_fp4.device
    out = FP4PrefillTopKResult(
        torch.empty((rows, topk), dtype=torch.float32, device=device),
        torch.empty((rows, topk), dtype=torch.int32, device=device),
        torch.empty((rows, topk), dtype=torch.int32, device=device),
        torch.empty(rows, dtype=torch.int32, device=device),
    )

    def run() -> None:
        flydsl_pa_mqa_topk_fp4_prefill(
            *case.operands,
            topk=topk,
            weight_scale=WEIGHT_SCALE,
            block_k=BLOCK_K,
            kv_block_size=KV_BLOCK_SIZE,
            parallel_unit_num=parallel_units,
            score_batch_chunks=chunks,
            workspace=workspace,
            out=out,
        )

    return run, out, _tensor_bytes(workspace), _tensor_bytes(out)


def _time(run: Callable[[], None], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    pairs = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iterations)
    ]
    for start, end in pairs:
        start.record()
        run()
        end.record()
    torch.cuda.synchronize()
    return statistics.median(start.elapsed_time(end) * 1000.0 for start, end in pairs)


def _check_exact(case: PackedCase, topk: int, legacy, fused) -> None:
    names = ("values", "raw_indices", "physical_indices", "counts")
    for name, wanted, actual in zip(names, legacy, fused, strict=True):
        expected = wanted.view(torch.int32) if wanted.is_floating_point() else wanted
        observed = actual.view(torch.int32) if actual.is_floating_point() else actual
        if not torch.equal(expected, observed):
            mismatch = torch.nonzero(expected != observed)[0].tolist()
            index = tuple(mismatch)
            detail = ""
            if name == "physical_indices":
                row, slot = index
                raw = int(legacy[1][row, slot])
                batch = int(case.row_to_batch[row])
                logical_page = raw // KV_BLOCK_SIZE
                physical_page = int(case.block_tables[batch, logical_page])
                detail = (
                    f", raw={raw}, batch={batch}, logical_page={logical_page}, "
                    f"physical_page={physical_page}"
                )
            raise AssertionError(
                f"exact {name} mismatch for topk={topk} at {index}: "
                f"expected={int(expected[index])}, actual={int(observed[index])}"
                f"{detail}"
            )


def _result(
    provider, region, args, latency, legacy, scratch, output, baseline_scratch
) -> None:
    rows, width, topk, chunks, parallel_units = args
    reduction = 100.0 * (baseline_scratch - scratch) / baseline_scratch
    print(
        f"RESULT,provider={provider},rows={rows},context_width={width},topk={topk},"
        f"score_batch_chunks={chunks},parallel_unit_num={parallel_units},"
        f"timed_region={region},"
        f"latency_p50_us={latency:.3f},speedup_vs_legacy={legacy / latency:.4f},"
        f"scratch_bytes={scratch},output_bytes={output},"
        f"workspace_reduction_pct={reduction:.3f}"
    )


def _benchmark(case, topk, chunks, warmup, iterations) -> None:
    rows = case.q_fp4.shape[0]
    parallel_units = max(512, rows)
    legacy_run, legacy, legacy_scratch, legacy_output = _prepare_legacy(
        case, topk, parallel_units
    )
    fused_run, fused, fused_scratch, fused_output = _prepare_fused(
        case, topk, parallel_units, chunks
    )
    legacy_run()
    fused_run()
    torch.cuda.synchronize()
    _check_exact(case, topk, legacy, fused)
    legacy_us = _time(legacy_run, warmup, iterations)
    fused_us = _time(fused_run, warmup, iterations)
    config = (rows, case.width, topk, chunks, parallel_units)
    _result(
        "legacy",
        "scorer+stable_topk+ordering+physical_mapping",
        config,
        legacy_us,
        legacy_us,
        legacy_scratch,
        legacy_output,
        legacy_scratch,
    )
    _result(
        "fused",
        "public_scorer+selection+ordering+physical_mapping",
        config,
        fused_us,
        legacy_us,
        fused_scratch,
        fused_output,
        legacy_scratch,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=_positive, nargs="+", default=[1024])
    parser.add_argument("--widths", type=_positive, nargs="+", default=list(WIDTHS))
    parser.add_argument(
        "--topks", type=int, nargs="+", choices=TOPKS, default=list(TOPKS)
    )
    parser.add_argument(
        "--score-batch-chunks",
        type=int,
        nargs="+",
        choices=SUPPORTED_SCORE_BATCH_CHUNKS,
        default=[DEFAULT_SCORE_BATCH_CHUNKS],
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=_positive, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if min(args.widths) < max(args.topks):
        parser.error("every --widths value must be at least max(--topks)")
    return args


def main() -> None:
    args = _parse_args()
    device = _require_gfx950()
    print(
        "TIMING,process_model=isolated-only,"
        "legacy=scorer+stable_topk+ordering+physical_mapping,"
        "fused=public_scorer+selection+ordering+physical_mapping,"
        "equivalent_postprocessing=true"
    )
    with torch.inference_mode():
        case_number = 0
        for rows in dict.fromkeys(args.rows):
            for width in dict.fromkeys(args.widths):
                _preflight(rows, width, dict.fromkeys(args.topks), device)
                case = _make_case(rows, width, device, args.seed + case_number)
                case_number += 1
                for topk in dict.fromkeys(args.topks):
                    for chunks in dict.fromkeys(args.score_batch_chunks):
                        _benchmark(case, topk, chunks, args.warmup, args.iterations)


if __name__ == "__main__":
    main()
