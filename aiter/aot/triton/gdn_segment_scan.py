#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compile the segmented GDN K5 Triton kernels into the Triton cache.

The segmented path takes over from ``chunk_gdn_fwd_h_flydsl_opt`` on exactly
the shapes the FlyDSL dispatch gate hands it, and that kernel is already
covered by ``aiter.aot.flydsl.chunk_gdn_h``. Without this module those shapes
trade an AOT cache hit for a cold Triton compile on the first long prefill,
which lands in server warmup where no benchmark looks.

Unlike the other modules under ``aiter/aot/triton`` this does not emit hsaco
plus a C stub through ``triton.tools.compile``: nothing calls these kernels
from C++, the caller is the Python FlyDSL wrapper. What is wanted here is the
FlyDSL AOT contract instead -- fill the cache the JIT path already reads -- so
this warms the Triton cache via ``JITFunction.warmup``.

Coverage resolves through the same merged tuned table as the FlyDSL K5 AOT
(``AITER_CONFIGS.AITER_CONFIG_GDN_K5_OPT_FILE``) so the two track each other.

That default only reaches as far as the tuning does, and the kernels
specialise on ``H // Hg``. The shipped tables cover a ratio of 2
(``qwen3_5_35b``) and 4 (``qwen3_5_397b``), so a model at any other ratio
gets nothing from them -- Qwen3.8 is 128 value heads to 16 key heads, which
is a ratio of 8 at every TP degree, and a CSV-driven run compiles artifacts
its server will never ask for. Pass ``--heads`` to compile the deployed
split, and read the shape line the header prints before trusting a cache.

Only ``H``/``Hg``/``dtype`` are read from a row, because nothing else these
kernels specialise on is a tuned dimension: snapshot and state dtype, the
indexed state pool, the final-state write and the incoming state of the replay
pass are all resolved per request by the caller. Those are fanned out in full,
the same way ``chunk_gdn_h`` fans out ``BV``. State dtype in particular is
worth the extra artifacts -- the tuned rows all carry ``state_bf16=False``
while vLLM runs the pool at bf16, so trusting the column would miss every
shape that matters in serving.

Two differences from the FlyDSL AOT are worth knowing:

* This has to run on a GPU of the target architecture. Triton compiles for the
  active driver's target, so there is no ``FLYDSL_GPU_ARCH`` equivalent to
  cross-compile with. No kernel is launched -- pointer arguments are dtype
  placeholders that Triton wraps as ``MockTensor`` -- so one idle GPU is enough.
* The cache is Triton's, not FlyDSL's, so the build step and the serving
  process have to agree on ``TRITON_CACHE_DIR`` or the runtime sees a miss.

Usage:
    python -m aiter.aot.triton.gdn_segment_scan
    python -m aiter.aot.triton.gdn_segment_scan --csv /path/to/tuned.csv
    # Qwen3.8 (128 value / 16 key heads) at TP8:
    python -m aiter.aot.triton.gdn_segment_scan --heads 16:2

Environment variables:
    TRITON_CACHE_DIR  Cache directory (default: ~/.triton/cache)
    AITER_CONFIGS     Resolves the default CSV lookup path (same as the runtime)
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time
from typing import Any

import torch
from triton.runtime.jit import MockTensor

from aiter.aot.flydsl.common import collect_aot_jobs, dedupe_jobs
from aiter.jit.core import AITER_CONFIGS
from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.gdn_segment_scan import (
    _gdn_segment_kernel,
    _gdn_segment_scan_kernel,
)

# Imported rather than restated: BV and the warp counts are tuning results, and
# an AOT list that disagrees with the wrapper compiles artifacts the runtime
# will never ask for while missing the ones it will.
from aiter.ops.triton.gated_delta_net.gdn_segment_scan import (
    _K,
    _SCAN_BV,
    _SCAN_WARPS,
    _SEGMENT_BV,
    _SEGMENT_WARPS,
    _V,
)

DEFAULT_CSVS = [AITER_CONFIGS.AITER_CONFIG_GDN_K5_OPT_FILE]

_TORCH_DTYPE = {
    "torch.bfloat16": torch.bfloat16,
    "torch.float16": torch.float16,
}

_F32 = "torch.float32"
_BF16 = "torch.bfloat16"

# T_FLAT is a runtime argument, but Triton still specialises integer arguments
# on ``== 1`` and divisibility by 16. The kernel carries
# ``do_not_specialize=["T_FLAT"]`` for that reason, which is what lets one
# arbitrary token count here stand in for every prefill length.
_T_FLAT = 8192

# Neither kernel's cache key depends on the grid, and nothing is launched.
_GRID = (1, 1)


def _shape_rows(csv_path: str) -> list[dict[str, Any]]:
    """Read the (H, Hg, dtype) shapes the segmented path can be handed.

    Rows the dispatch gate could never route here are dropped rather than
    compiled: the gate requires K=V=128, so a row at any other head dimension
    describes a shape that stays on FlyDSL.
    """
    rows: list[dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(line for line in f if not line.lstrip().startswith("#"))
        for row in reader:
            dtype = (row.get("dtype") or _BF16).strip()
            if dtype not in _TORCH_DTYPE:
                print(f"  [WARN] Unsupported dtype {dtype!r}, skipping")
                continue
            try:
                K, V = int(row["K"]), int(row["V"])
                H, HG = int(row["H"]), int(row["Hg"])
            except (KeyError, TypeError, ValueError) as e:
                print(f"  [WARN] malformed row in {csv_path}: {e}")
                continue
            if (K, V) != (_K, _V):
                continue
            rows.append({"H": H, "HG": HG, "dtype": dtype})
    return rows


def _launches_for_shape(H: int, HG: int, dtype: str) -> list[dict[str, Any]]:
    """Expand one shape into every launch ``gdn_segment_scan_fwd`` can issue.

    The wrapper issues three: a dual-summary pass, the scan that composes the
    summaries, and a replay pass. Only the replay varies much -- it can receive
    its incoming state three ways (no state at all, the float32 buffer the scan
    wrote, or a gathered entry state when every sequence fits one segment), and
    each forks the compiled artifact.
    """
    jobs: list[dict[str, Any]] = []
    for state_bf16, use_state_indices in itertools.product((False, True), repeat=2):
        state_dtype = _BF16 if state_bf16 else _F32
        common = {
            "H": H,
            "HG": HG,
            "dtype": dtype,
            "state_bf16": state_bf16,
            "use_state_indices": use_state_indices,
        }

        jobs.append(
            {
                **common,
                "kernel": "summary",
                "snapshot_dtype": None,
                "h_in_dtype": None,
                "store_final": False,
            }
        )

        # The scan touches neither the inputs nor the key/value head split, so
        # it carries no HG or input dtype: leaving them out lets dedupe collapse
        # the copy every (HG, dtype) row would otherwise contribute.
        for has_h0 in (False, True):
            jobs.append(
                {
                    "kernel": "scan",
                    "H": H,
                    "state_bf16": state_bf16,
                    "use_state_indices": use_state_indices,
                    "has_h0": has_h0,
                }
            )

        h_in_dtypes = (None, _F32, state_dtype)
        for snapshot_bf16, store_final, h_in_dtype in itertools.product(
            (True, False), (False, True), h_in_dtypes
        ):
            jobs.append(
                {
                    **common,
                    "kernel": "replay",
                    "snapshot_dtype": _BF16 if snapshot_bf16 else _F32,
                    "h_in_dtype": h_in_dtype,
                    "store_final": store_final,
                }
            )
    return jobs


def parse_csv(csv_path: str) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for row in _shape_rows(csv_path):
        jobs.extend(_launches_for_shape(row["H"], row["HG"], row["dtype"]))
    return jobs


def _ptr(dtype: str | torch.dtype | None):
    """Pointer placeholder.

    ``MockTensor`` is constructed here rather than left to ``warmup``, which
    only wraps bare dtypes it receives positionally -- every argument below is
    passed by keyword so it can be read against the wrapper's launch.

    ``None`` is passed through: the runtime passes ``None`` for the buffers a
    pass does not use and Triton specialises on that, so substituting a
    placeholder here would compile an artifact the runtime never asks for.
    """
    if dtype is None:
        return None
    if isinstance(dtype, str):
        dtype = _TORCH_DTYPE.get(dtype, torch.float32)
    return MockTensor(dtype)


def _warmup_segment(job: dict[str, Any]) -> None:
    summary = job["kernel"] == "summary"
    state_dtype = _BF16 if job["state_bf16"] else _F32
    in_dtype = _ptr(job["dtype"])
    i32 = _ptr(torch.int32)
    _gdn_segment_kernel.warmup(
        k=in_dtype,
        u=in_dtype,
        w=in_dtype,
        g=_ptr(torch.float32),
        h_in=_ptr(job["h_in_dtype"]),
        h_out=_ptr(torch.float32) if summary else None,
        a_out=_ptr(torch.bfloat16) if summary else None,
        h_snapshots=None if summary else _ptr(job["snapshot_dtype"]),
        v_new=None if summary else in_dtype,
        final_state=_ptr(state_dtype) if job["store_final"] else None,
        state_indices=i32 if job["use_state_indices"] else None,
        seg_chunk_base=i32,
        seg_nchunks=i32,
        seg_tok_base=i32,
        seg_tok_end=i32,
        seg_seq=i32,
        seg_is_last=i32,
        T_FLAT=_T_FLAT,
        H=job["H"],
        HG=job["HG"],
        K=_K,
        V=_V,
        BV=_SEGMENT_BV,
        DUAL_SUMMARY=summary,
        HAS_H_IN=job["h_in_dtype"] is not None,
        WRITE_OUTPUTS=not summary,
        STORE_H_OUT=summary,
        STORE_FINAL=job["store_final"],
        USE_STATE_INDICES=job["use_state_indices"],
        STATE_BF16=job["state_bf16"],
        num_warps=_SEGMENT_WARPS,
        num_stages=1,
        grid=_GRID,
    )


def _warmup_scan(job: dict[str, Any]) -> None:
    state_dtype = _BF16 if job["state_bf16"] else _F32
    _gdn_segment_scan_kernel.warmup(
        a_seg=_ptr(torch.bfloat16),
        b_seg=_ptr(torch.float32),
        h_in=_ptr(torch.float32),
        h0=_ptr(state_dtype) if job["has_h0"] else None,
        state_indices=_ptr(torch.int32) if job["use_state_indices"] else None,
        seq_seg_offsets=_ptr(torch.int32),
        H=job["H"],
        K=_K,
        V=_V,
        BV=_SCAN_BV,
        HAS_H0=job["has_h0"],
        USE_STATE_INDICES=job["use_state_indices"],
        num_warps=_SCAN_WARPS,
        num_stages=1,
        grid=_GRID,
    )


def _format_shape_str(job: dict[str, Any]) -> str:
    if job["kernel"] == "scan":
        return (
            f"gdn_segment_scan  H={job['H']} "
            f"has_h0={job['has_h0']} "
            f"state_bf16={job['state_bf16']} "
            f"use_state_indices={job['use_state_indices']}"
        )
    return (
        f"gdn_segment_{job['kernel']:<7} H={job['H']} HG={job['HG']} "
        f"dtype={job['dtype']} snapshot={job['snapshot_dtype']} "
        f"h_in={job['h_in_dtype']} store_final={job['store_final']} "
        f"state_bf16={job['state_bf16']} "
        f"use_state_indices={job['use_state_indices']}"
    )


def compile_one_config(job: dict[str, Any]) -> dict[str, Any]:
    shape_str = _format_shape_str(job)
    result = {"shape": shape_str, "compile_time": None}
    t0 = time.time()
    try:
        if job["kernel"] == "scan":
            _warmup_scan(job)
        else:
            _warmup_segment(job)
        result["compile_time"] = time.time() - t0
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] compile  {shape_str}: {e}")
    return result


def _parse_heads(spec: str) -> tuple[int, int]:
    try:
        H, HG = spec.split(":")
        return int(H), int(HG)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--heads expects H:HG (e.g. 16:8), got {spec!r}"
        ) from None


def main():
    parser = argparse.ArgumentParser(
        description="AOT pre-compile the segmented GDN K5 Triton kernels "
        "from aiter CSV config",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=str,
        nargs="+",
        default=DEFAULT_CSVS,
        help="Path(s) to tuned CSV config file(s); defaults come from AITER_CONFIGS",
    )
    parser.add_argument(
        "--heads",
        type=_parse_heads,
        nargs="+",
        default=None,
        help="Compile exactly these H:HG pairs instead of the CSV shapes.\n"
        "Needed whenever the deployed model's head split is not in the\n"
        "tuned tables: they currently cover H//Hg of 2 and 4 only, so a\n"
        "model at any other ratio gets no coverage from the CSVs at all.",
    )
    args = parser.parse_args()

    csv_paths = [os.path.abspath(p) for p in args.csv]
    for csv_path in csv_paths:
        if not os.path.isfile(csv_path):
            print(f"Error: CSV file not found: {csv_path}")
            sys.exit(1)

    if args.heads:
        # Injected, not filtered. The CSVs are a tuning artifact and only
        # describe the models that have been tuned, so filtering them can
        # never produce a shape they do not already contain -- which is the
        # case that matters, because an untuned model is exactly the one
        # whose kernels are still on the JIT path.
        dtypes = sorted({row["dtype"] for p in csv_paths for row in _shape_rows(p)})
        jobs = dedupe_jobs(
            [
                job
                for (H, HG), dtype in itertools.product(
                    dict.fromkeys(args.heads), dtypes or [_BF16]
                )
                for job in _launches_for_shape(H, HG, dtype)
            ]
        )
    else:
        jobs = collect_aot_jobs(csv_paths, parse_csv)

    shapes = sorted({(j["H"], j["HG"]) for j in jobs if "HG" in j})

    cache_dir = os.path.expanduser(
        os.environ.get("TRITON_CACHE_DIR", "~/.triton/cache")
    )

    print("=" * 72)
    print("GDN segmented K5 (Triton) AOT Pre-compilation")
    print("=" * 72)
    for csv_path in csv_paths:
        print(f"  CSV:          {csv_path}")
    print(f"  Source:       {'--heads' if args.heads else 'CSV shapes'}")
    # Printed because the failure this guards against is silent: a cache full
    # of the wrong head split looks exactly like a cache that works, right up
    # until the server JITs anyway.
    print(
        "  Shapes:       "
        + (", ".join(f"H={H} Hg={HG}" for H, HG in shapes) or "(none)")
    )
    print(f"  Total jobs:   {len(jobs)}")
    print(f"  Cache dir:    {cache_dir}")
    print("=" * 72)

    if not jobs:
        print(
            "\nNo shapes to compile. The tuned CSVs cover no K=V=128 row, so "
            "pass the deployed model's head split with --heads H:HG."
        )
        sys.exit(1)

    total_t0 = time.time()
    print(f"\n--- Compiling {len(jobs)} kernels ---")
    # Serial on purpose. The FlyDSL AOT forks a worker pool because each of its
    # compiles costs seconds and gigabytes; Triton's are cheap, and a forked
    # process that has already touched the GPU is a liability for no gain.
    results = [compile_one_config(job) for job in jobs]
    total_elapsed = time.time() - total_t0

    ok = sum(1 for r in results if r["compile_time"] is not None)
    fail = sum(1 for r in results if r["compile_time"] is None)

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"  Total time:   {total_elapsed:.1f}s")
    print(f"  Compiled:     {ok} ok, {fail} failed")
    print(f"  Cache dir:    {cache_dir}")
    print()

    if fail > 0:
        print("Some compilations failed. Check output above for details.")
        sys.exit(1)
    print("All compilations succeeded. Cache is ready.")
    sys.exit(0)


if __name__ == "__main__":
    main()
