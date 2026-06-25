# SPDX-License-Identifier: MIT

"""Run focused FlyDSL cooperative TopK primitive diagnostics."""

from __future__ import annotations

import argparse
import math
import sys

import torch

from aiter.ops.flydsl.utils import is_flydsl_available


def _load_diagnostics():
    from aiter.ops.flydsl.kernels.topk_per_row_decode_coop_diag import (
        COVERAGE_MODES,
        barrier_workspace_slots,
        run_coop_arrival_diagnostic,
        run_coop_barrier_diagnostic,
        run_coop_coverage_diagnostic,
    )

    return (
        COVERAGE_MODES,
        barrier_workspace_slots,
        run_coop_arrival_diagnostic,
        run_coop_barrier_diagnostic,
        run_coop_coverage_diagnostic,
    )


def _row_lengths(num_rows: int, max_len: int) -> list[int]:
    if num_rows == 1:
        return [max_len]
    variants = [max_len, max(1, max_len - 17), max(1, max_len // 2), min(max_len, 4096)]
    return [variants[i % len(variants)] for i in range(num_rows)]


def _validate_coverage(
    counts: torch.Tensor,
    owner_sums: torch.Tensor,
    lengths: list[int],
    max_vec_blocks: int,
) -> tuple[bool, str]:
    counts_cpu = counts.cpu()
    owners_cpu = owner_sums.cpu()
    for row, row_len in enumerate(lengths):
        vec_blocks = math.ceil(max(row_len, 0) / 4)
        expected = torch.zeros(max_vec_blocks, dtype=torch.int32)
        expected[:vec_blocks] = 1
        actual = counts_cpu[row]
        if not torch.equal(actual, expected):
            bad = torch.nonzero(actual != expected, as_tuple=False).flatten()
            first = int(bad[0].item())
            skipped = int(((actual[:vec_blocks]) == 0).sum().item())
            duplicate = int(((actual[:vec_blocks]) > 1).sum().item())
            tail = int((actual[vec_blocks:] != 0).sum().item())
            owner = int(owners_cpu[row, first].item())
            return (
                False,
                (
                    f"row={row} vblk={first} actual={int(actual[first].item())} "
                    f"expected={int(expected[first].item())} owner_sum={owner} "
                    f"skipped={skipped} duplicate={duplicate} tail={tail}"
                ),
            )
    return True, "ok"


def run_coverage_matrix(
    *,
    partitions: list[int],
    row_counts: list[int],
    lengths: list[int],
    modes: tuple[str, ...],
) -> bool:
    _, _, _, _, run_coop_coverage_diagnostic = _load_diagnostics()
    all_ok = True
    for p in partitions:
        for mode in modes:
            for num_rows in row_counts:
                for max_len in lengths:
                    host_lengths = _row_lengths(num_rows, max_len)
                    max_vec_blocks = math.ceil(max(host_lengths) / 4)
                    row_lengths = torch.tensor(host_lengths, device="cuda", dtype=torch.int32)
                    counts = torch.zeros(
                        (num_rows, max_vec_blocks), device="cuda", dtype=torch.int32
                    )
                    owner_sums = torch.zeros_like(counts)
                    run_coop_coverage_diagnostic(
                        row_lengths,
                        counts,
                        owner_sums,
                        partitions_per_row=p,
                        max_vec_blocks=max_vec_blocks,
                        mode=mode,
                    )
                    torch.cuda.synchronize()
                    ok, detail = _validate_coverage(
                        counts, owner_sums, host_lengths, max_vec_blocks
                    )
                    status = "PASS" if ok else "FAIL"
                    print(
                        f"coverage {status}: P={p} mode={mode} rows={num_rows} "
                        f"max_len={max_len} detail={detail}"
                    )
                    all_ok = all_ok and ok
    return all_ok


def run_arrival_matrix(
    *,
    partitions: list[int],
    row_counts: list[int],
) -> bool:
    _, barrier_workspace_slots, run_coop_arrival_diagnostic, _, _ = _load_diagnostics()
    all_ok = True
    for p in partitions:
        expected_values = torch.tensor(list(range(1, p + 1)), dtype=torch.int32)
        for num_rows in row_counts:
            values = torch.zeros((num_rows, p), device="cuda", dtype=torch.int32)
            workspace = torch.zeros(
                barrier_workspace_slots(num_rows), device="cuda", dtype=torch.int32
            )
            run_coop_arrival_diagnostic(
                values,
                workspace,
                partitions_per_row=p,
            )
            torch.cuda.synchronize()
            values_cpu = values.cpu()
            workspace_cpu = workspace.cpu().view(num_rows, -1)
            values_ok = bool(torch.all(values_cpu == expected_values).item())
            arrive_ok = bool(torch.all(workspace_cpu[:, 0] == p).item())
            done_ok = bool(torch.all(workspace_cpu[:, 1] == 1).item())
            ok = values_ok and arrive_ok and done_ok
            detail = "ok"
            if not ok:
                detail = f"values={values_cpu.tolist()} workspace={workspace_cpu.tolist()}"
            status = "PASS" if ok else "FAIL"
            print(f"arrival {status}: P={p} rows={num_rows} detail={detail}")
            all_ok = all_ok and ok
    return all_ok


def run_barrier_matrix(
    *,
    partitions: list[int],
    row_counts: list[int],
) -> bool:
    _, barrier_workspace_slots, _, run_coop_barrier_diagnostic, _ = _load_diagnostics()
    all_ok = True
    epoch = 1
    for p in partitions:
        expected = p * (p + 1) // 2
        for num_rows in row_counts:
            values = torch.zeros((num_rows, p), device="cuda", dtype=torch.int32)
            observed = torch.zeros_like(values)
            workspace = torch.zeros(
                barrier_workspace_slots(num_rows), device="cuda", dtype=torch.int32
            )
            run_coop_barrier_diagnostic(
                values,
                observed,
                workspace,
                partitions_per_row=p,
                epoch=epoch,
            )
            epoch += 1
            torch.cuda.synchronize()
            observed_cpu = observed.cpu()
            ok = bool(torch.all(observed_cpu == expected).item())
            detail = "ok"
            if not ok:
                bad = torch.nonzero(observed_cpu != expected, as_tuple=False)[0]
                row = int(bad[0].item())
                part = int(bad[1].item())
                detail = (
                    f"row={row} part={part} actual={int(observed_cpu[row, part].item())} "
                    f"expected={expected}"
                )
            status = "PASS" if ok else "FAIL"
            print(f"barrier {status}: P={p} rows={num_rows} detail={detail}")
            all_ok = all_ok and ok
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-blocking-barrier",
        action="store_true",
        help=(
            "Also run the device-side spin barrier diagnostic. This is opt-in "
            "because the current primitive is known to hang on failure."
        ),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=None,
        help="Coverage modes to run. Defaults to all diagnostic modes.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("SKIP: CUDA/ROCm is not available")
        return 0
    if not is_flydsl_available():
        print("SKIP: FlyDSL is not available")
        return 0

    coverage_modes, _, _, _, _ = _load_diagnostics()
    modes = tuple(args.modes) if args.modes is not None else coverage_modes
    unknown = sorted(set(modes) - set(coverage_modes))
    if unknown:
        raise ValueError(f"unknown coverage modes: {unknown}")

    partitions = [4, 8, 16]
    row_counts = [1, 4, 16]
    lengths = [4096, 32768, 120000]

    ok = run_coverage_matrix(
        partitions=partitions,
        row_counts=row_counts,
        lengths=lengths,
        modes=modes,
    )
    ok = run_arrival_matrix(partitions=partitions, row_counts=row_counts) and ok
    if args.run_blocking_barrier:
        ok = run_barrier_matrix(partitions=partitions, row_counts=row_counts) and ok

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
