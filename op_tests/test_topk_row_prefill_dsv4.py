# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Broad gfx1250 prefill TopK comparison between HIP and FlyDSL.

Each shape/configuration is appended to a JSONL checkpoint immediately after it
finishes, so completed measurements survive a later process or container crash.

Run the default sweep with:

    python -m op_tests.test_topk_row_prefill_dsv4
"""

import argparse
import gc
import itertools
import json
import os
import time
from functools import lru_cache
from pathlib import Path

import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.topk import (
    _top_k_per_row_prefill,
    flydsl_radix_topk_one_block_gfx1250,
    topk_mb_workspace_size,
    topk_ob_workspace_size,
    topk_use_mulblocks,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx1250"]
_CORRECTNESS_ROWS = 16
_VALIDATION_CHUNK_ROWS = 256
_WARMUP = 5
_ITERATIONS = 20

# shape = (pattern, rows, physical_width, max_effective_row_len, compression_ratio)
DEFAULT_BATCHES = [1 << exponent for exponent in range(15)]
DEFAULT_LENGTHS = [1 << exponent for exponent in range(7, 21)]
DEFAULT_SHAPES = [
    ("full", batch, length, length, 1)
    for batch, length in itertools.product(
        DEFAULT_BATCHES,
        DEFAULT_LENGTHS,
    )
]


def _parse_shape(text: str) -> tuple[str, int, int, int, int]:
    parts = text.split(":")
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "shape must be pattern:rows:physical_width:max_seq_len:compression_ratio"
        )
    pattern = parts[0]
    if pattern not in ("full", "padded", "fixed", "ragged", "compressed"):
        raise argparse.ArgumentTypeError(
            "pattern must be full, padded, fixed, ragged, or compressed"
        )
    try:
        rows, width, max_seq_len, compression_ratio = map(int, parts[1:])
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape dimensions must be integers") from error
    return pattern, rows, width, max_seq_len, compression_ratio


def _make_boundaries(
    pattern: str,
    num_rows: int,
    physical_width: int,
    max_seq_len: int,
    compression_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.arange(num_rows, dtype=torch.int64, device="cuda")
    if pattern == "full":
        starts = torch.zeros_like(rows)
        ends = torch.full_like(rows, physical_width)
    elif pattern == "compressed":
        starts = torch.zeros_like(rows)
        ends = rows // compression_ratio
    elif pattern == "padded":
        if num_rows == 1:
            lengths = torch.full_like(rows, max_seq_len)
            starts = torch.full_like(rows, (physical_width - max_seq_len) // 2)
        else:
            lengths = rows * max_seq_len // (num_rows - 1)
            starts = rows * (physical_width - max_seq_len) // (num_rows - 1)
        ends = starts + lengths
    elif pattern == "fixed":
        lengths = torch.full_like(rows, max_seq_len)
        denominator = max(1, num_rows - 1)
        starts = rows * (physical_width - max_seq_len) // denominator
        ends = starts + lengths
    else:
        lengths = (
            rows * 1_103_515_245 + 12_345
        ) % (max_seq_len + 1)
        room = physical_width - lengths
        starts = (
            (rows * 2_654_435_761 + 1_013_904_223) % (room + 1)
        )
        ends = starts + lengths
    return starts.to(torch.int32), ends.to(torch.int32)


@lru_cache(maxsize=1)
def _make_inputs(
    pattern: str,
    num_rows: int,
    physical_width: int,
    max_seq_len: int,
    compression_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    row_starts, row_ends = _make_boundaries(
        pattern,
        num_rows,
        physical_width,
        max_seq_len,
        compression_ratio,
    )
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260903)
    logits = torch.randn(
        (num_rows, physical_width),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    return logits, row_starts, row_ends


def _hip_path(stable: bool, num_rows: int, physical_width: int) -> str:
    if stable or not topk_use_mulblocks(num_rows, physical_width):
        return "one_block"
    return "multi_block"


def _hip_workspace_bytes(
    path: str,
    num_rows: int,
    physical_width: int,
    k: int,
) -> int:
    if path == "multi_block":
        size = int(
            topk_mb_workspace_size(num_rows, physical_width, k, False)
        )
        return 1 if size <= 1 else 1 << (size - 1).bit_length()
    return max(
        1,
        int(topk_ob_workspace_size(num_rows, physical_width, k, False)),
    )


def _required_device_bytes(
    num_rows: int,
    physical_width: int,
    k: int,
    stable: bool,
    write_values: bool,
) -> int:
    path = _hip_path(stable, num_rows, physical_width)
    workspace_bytes = _hip_workspace_bytes(
        path, num_rows, physical_width, k
    )
    logits_bytes = num_rows * physical_width * torch.float32.itemsize
    one_candidate_output = num_rows * k * torch.int32.itemsize
    if write_values:
        one_candidate_output *= 2
    return logits_bytes + 2 * one_candidate_output + workspace_bytes


def _sample_rows(num_rows: int) -> list[int]:
    count = min(num_rows, _CORRECTNESS_ROWS)
    if count == 1:
        return [0]
    return (
        torch.linspace(0, num_rows - 1, count, device="cpu")
        .round()
        .to(torch.int64)
        .tolist()
    )


def _reference_values(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    k: int,
    sample_rows: list[int],
) -> torch.Tensor:
    chunks = []
    for row in sample_rows:
        start = int(row_starts[row].item())
        end = int(row_ends[row].item())
        count = min(k, end - start)
        if count:
            chunks.append(
                torch.topk(
                    logits[row, start:end],
                    count,
                    largest=True,
                    sorted=True,
                ).values
            )
    if chunks:
        return torch.cat(chunks)
    return torch.empty(0, dtype=logits.dtype, device=logits.device)


def _all_rows_valid(
    indices: torch.Tensor,
    values: torch.Tensor | None,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    k: int,
) -> bool:
    positions = torch.arange(k, dtype=torch.int32, device="cuda")[None, :]
    for begin in range(0, indices.shape[0], _VALIDATION_CHUNK_ROWS):
        end = min(begin + _VALIDATION_CHUNK_ROWS, indices.shape[0])
        chunk_indices = indices[begin:end]
        starts = row_starts[begin:end, None]
        ends = row_ends[begin:end, None]
        counts = torch.minimum(
            ends - starts,
            torch.full_like(ends, k),
        )
        valid_mask = positions < counts
        in_range = (
            (chunk_indices >= starts)
            & (chunk_indices < ends)
        )
        if not bool(torch.all(in_range[valid_mask])):
            return False
        if not bool(torch.all(chunk_indices[~valid_mask] == -1)):
            return False
        if values is not None and not bool(
            torch.all(torch.isneginf(values[begin:end][~valid_mask]))
        ):
            return False
    return True


def _check_output(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor | None,
    k: int,
    stable: bool,
    sample_rows: list[int],
    reference: torch.Tensor,
    name: str,
) -> float:
    valid = _all_rows_valid(indices, values, row_starts, row_ends, k)
    selected_chunks = []
    values_match = True
    unique = True
    stable_order = True
    for row in sample_rows:
        start = int(row_starts[row].item())
        end = int(row_ends[row].item())
        count = min(k, end - start)
        selected_indices = indices[row, :count]
        unique &= selected_indices.unique().numel() == count
        if stable and count > 1:
            stable_order &= bool(
                torch.all(selected_indices[1:] >= selected_indices[:-1])
            )
        if count:
            safe_indices = selected_indices.clamp(start, end - 1)
            gathered = logits[row, safe_indices.to(torch.int64)]
            selected_chunks.append(
                torch.sort(gathered, descending=True).values
            )
            if values is not None:
                values_match &= torch.equal(values[row, :count], gathered)

    selected = (
        torch.cat(selected_chunks)
        if selected_chunks
        else torch.empty(0, dtype=logits.dtype, device=logits.device)
    )
    err = checkAllclose(
        reference.to(torch.float32),
        selected.to(torch.float32),
        rtol=0,
        atol=0,
        tol_err_ratio=0,
        msg=f"{name}: prefill TopK",
    )
    if not (valid and unique and stable_order and values_match):
        return 1.0
    return float(err)


@benchmark()
def test_prefill_topk_case(
    pattern: str,
    num_rows: int,
    physical_width: int,
    max_seq_len: int,
    compression_ratio: int,
    k: int,
    stable: bool,
    write_values: bool,
) -> dict:
    logits, row_starts, row_ends = _make_inputs(
        pattern,
        num_rows,
        physical_width,
        max_seq_len,
        compression_ratio,
    )
    effective_lengths = row_ends - row_starts
    max_effective_row_len = int(effective_lengths.max().item())
    if max_effective_row_len > physical_width:
        raise ValueError("effective row length exceeds physical width")

    hip_indices = torch.full(
        (num_rows, k), -123456789, dtype=torch.int32, device="cuda"
    )
    flydsl_indices = torch.full_like(hip_indices, -123456789)
    hip_values = (
        torch.empty((num_rows, k), dtype=torch.float32, device="cuda")
        if write_values
        else None
    )
    flydsl_values = (
        torch.empty_like(hip_values) if hip_values is not None else None
    )

    hip_path = _hip_path(stable, num_rows, physical_width)
    workspace_bytes = _hip_workspace_bytes(
        hip_path, num_rows, physical_width, k
    )
    hip_workspace = (
        torch.zeros(workspace_bytes, dtype=torch.uint8, device="cuda")
        if hip_path == "multi_block"
        else torch.empty(workspace_bytes, dtype=torch.uint8, device="cuda")
    )

    def run_hip() -> torch.Tensor:
        _top_k_per_row_prefill(
            logits,
            row_starts,
            row_ends,
            hip_indices,
            hip_values,
            num_rows,
            logits.stride(0),
            logits.stride(1),
            k,
            hip_workspace,
            stable,
        )
        return hip_indices

    def run_flydsl() -> torch.Tensor:
        flydsl_radix_topk_one_block_gfx1250(
            logits,
            row_starts,
            row_ends,
            flydsl_indices,
            flydsl_values,
            num_rows,
            logits.stride(0),
            logits.stride(1),
            k=k,
            stable=stable,
        )
        return flydsl_indices

    hip_name = f"hip_{hip_path}"
    candidates = {
        hip_name: (run_hip, hip_values),
        "flydsl": (run_flydsl, flydsl_values),
    }
    sample_rows = _sample_rows(num_rows)
    reference = _reference_values(
        logits, row_starts, row_ends, k, sample_rows
    )

    visible_scores = int(effective_lengths.to(torch.int64).sum().item())
    output_bytes = hip_indices.nbytes
    if write_values:
        output_bytes *= 2
    logical_flops = visible_scores
    logical_bytes = visible_scores * logits.element_size() + output_bytes
    ret = {
        "gfx": get_gfx(),
        "hip_path": hip_path,
        "actual_min": int(effective_lengths.min().item()),
        "actual_max": max_effective_row_len,
        "logits_MiB": logits.nbytes / 2**20,
        "visible_MiB": visible_scores * logits.element_size() / 2**20,
        "workspace_MiB": workspace_bytes / 2**20,
    }

    for name, (candidate, candidate_values) in candidates.items():
        output, us = run_perftest(
            candidate,
            num_iters=_ITERATIONS,
            num_warmup=_WARMUP,
            use_cuda_event=True,
        )
        err = _check_output(
            logits,
            row_starts,
            row_ends,
            output,
            candidate_values,
            k,
            stable,
            sample_rows,
            reference,
            name,
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = logical_flops / us / 1e6
        ret[f"{name} TB/s"] = logical_bytes / us / 1e6
        ret[f"{name} err"] = err

    return ret


def _append_checkpoint(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(
            json.dumps(
                record,
                sort_keys=True,
                default=lambda value: value.item(),
            )
            + "\n"
        )
        output.flush()
        os.fsync(output.fileno())


def main() -> None:
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "prefill TopK HIP/FlyDSL comparison unsupported on %s; skipping",
            get_gfx(),
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Broad prefill TopK HIP/FlyDSL comparison",
    )
    parser.add_argument(
        "-s",
        "--shape",
        type=_parse_shape,
        nargs="*",
        default=DEFAULT_SHAPES,
        help=(
            "pattern:rows:physical_width:max_seq_len:compression_ratio\n"
            "Example: -s padded:16:200000:4096:1"
        ),
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        nargs="*",
        default=[1 << exponent for exponent in range(1, 13)],
        help="Top-k values. Example: -k 512 1024",
    )
    parser.add_argument(
        "--stable",
        type=int,
        choices=(0, 1),
        nargs="*",
        default=[0, 1],
        help="Stable modes. Example: --stable 0 1",
    )
    parser.add_argument(
        "--write-values",
        type=int,
        choices=(0, 1),
        nargs="*",
        default=[0, 1],
        help="Value-output modes. Example: --write-values 0 1",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("aiter_logs/topk_prefill_gfx1250_results.jsonl"),
        help="Append-only per-case JSONL checkpoint path",
    )
    args = parser.parse_args()

    run_id = time.strftime("%Y%m%d-%H%M%S")
    completed = []
    for (
        (pattern, num_rows, physical_width, max_seq_len, compression_ratio),
        k,
        stable_int,
        write_values_int,
    ) in itertools.product(
        args.shape,
        args.top_k,
        args.stable,
        args.write_values,
    ):
        stable = bool(stable_int)
        write_values = bool(write_values_int)
        case = {
            "run_id": run_id,
            "pattern": pattern,
            "num_rows": num_rows,
            "physical_width": physical_width,
            "max_seq_len": max_seq_len,
            "compression_ratio": compression_ratio,
            "k": k,
            "stable": stable,
            "write_values": write_values,
        }
        _append_checkpoint(
            args.results_file,
            {**case, "status": "started"},
        )
        try:
            if min(
                num_rows,
                physical_width,
                max_seq_len,
                compression_ratio,
                k,
            ) <= 0:
                raise ValueError("all shape values must be positive")
            if max_seq_len > physical_width:
                raise ValueError("max_seq_len exceeds physical_width")

            required_bytes = _required_device_bytes(
                num_rows,
                physical_width,
                k,
                stable,
                write_values,
            )
            free_bytes, _ = torch.cuda.mem_get_info()
            if required_bytes > free_bytes * 0.8:
                record = {
                    **case,
                    "status": "skipped_memory",
                    "required_GiB": required_bytes / 2**30,
                    "free_GiB": free_bytes / 2**30,
                }
            else:
                record = test_prefill_topk_case(
                    pattern,
                    num_rows,
                    physical_width,
                    max_seq_len,
                    compression_ratio,
                    k,
                    stable,
                    write_values,
                )
                record.update(run_id=run_id, status="ok")
        except Exception as error:
            record = {
                **case,
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
            }
            aiter.logger.exception("prefill TopK case failed")

        _append_checkpoint(args.results_file, record)
        completed.append(record)
        aiter.logger.info(
            "checkpointed case %s to %s",
            len(completed),
            args.results_file,
        )
        gc.collect()
        torch.cuda.empty_cache()

    df = pd.DataFrame(completed)
    aiter.logger.info(
        "Prefill TopK HIP/FlyDSL summary (markdown):\n%s",
        df.to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
