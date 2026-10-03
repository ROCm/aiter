# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Small helpers shared by the tuning scripts. No torch/aiter imports: the driver imports this."""

import glob
import json
import os

FINAL_STATUSES = ("ok", "error", "crashed", "hung")
TOP_BUCKET_M = 4096  # --all-buckets sweeps the family's M_LEQ buckets up to here


def cdiv(a, b):
    return -(-a // b)


def next_pow2(n):
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def config_key(config):
    """Identity of a candidate in plans, result files and the installer (None is the baseline)."""
    return "baseline" if config is None else json.dumps(config, sort_keys=True)


def shape_tag(shape):
    return "-".join(f"{d}={shape[d]}" for d in ("B", "M", "N", "K") if d in shape)


def results_path(runs_dir, arch, backend, kernel, shape):
    return os.path.join(
        runs_dir, f"sweep-{arch}-{backend}-{kernel}-{shape_tag(shape)}.jsonl"
    )


def results_files(runs_dir, arch, backend, kernel, shape_nk):
    """All per-M result files of one (kernel, backend, N, K[, B])."""
    pattern = results_path(runs_dir, arch, backend, kernel, dict(shape_nk, M="*"))
    return sorted(glob.glob(pattern))


def specialized_filename(config_name, shape):
    """<CONFIG>-[B=..-]N=..-K=..json, the file get_gemm_config() looks up for this shape."""
    dims = "-".join(f"{d}={shape[d]}" for d in ("B", "N", "K") if d in shape)
    return f"{config_name}-{dims}.json"


def bucket_for(M, bounds):
    """M_LEQ_<smallest bound >= M>, the bucket get_gemm_config() picks; None above the largest bound."""
    for bound in bounds:
        if M <= bound:
            return f"M_LEQ_{bound}"
    return None


def append_record(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_records(path):
    """Last record per candidate, and how many 'ready' markers the file holds.

    A 'start' marker never replaces a final record, so a candidate the worker died on keeps
    status 'start'. A truncated last line (killed worker) is skipped.
    """
    records, n_ready = {}, 0
    if not os.path.exists(path):
        return records, n_ready
    with open(path) as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record["status"] == "ready":
                n_ready += 1
                continue
            key = "baseline" if record.get("baseline") else config_key(record["config"])
            is_final = record["status"] != "start"
            if is_final or records.get(key, {}).get("status") in (None, "start"):
                records[key] = record
    return records, n_ready


def add_shape_args(parser, with_m=True, multi_m=False):
    parser.add_argument("--B", type=int, help="batch, batched kernels only")
    if with_m:
        parser.add_argument(
            "--M", type=int, required=True, nargs="+" if multi_m else None
        )
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument(
        "--K",
        type=int,
        required=True,
        help="logical K, as the input generators take it",
    )


def shape_from_args(args, dims, M=None):
    if ("B" in dims) != (args.B is not None):
        raise SystemExit(
            "--B is required for batched kernels and not accepted otherwise"
        )
    shape = {"B": args.B} if "B" in dims else {}
    if M is not None:
        shape["M"] = M
    elif getattr(args, "M", None) is not None:
        shape["M"] = args.M
    shape.update(N=args.N, K=args.K)
    return shape
