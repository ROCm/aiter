# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Small helpers shared by the tuning scripts. No torch/aiter imports: the driver imports this."""

import glob
import json
import os

FINAL_STATUSES = ("ok", "error", "crashed", "hung")
TOP_BUCKET_M = 8192  # --all-buckets sweeps the family's M_LEQ buckets up to here


def cdiv(a, b):
    return -(-a // b)


def next_pow2(n):
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def config_key(config):
    """Identity of a candidate in plans, result files and the installer (None is the baseline)."""
    return "baseline" if config is None else json.dumps(config, sort_keys=True)


GEMM_DIMS = ("B", "M", "N", "K")


def shape_tag(shape):
    """GEMM dims in B, M, N, K order, then any other family's dims by name."""
    dims = [d for d in GEMM_DIMS if d in shape]
    dims += sorted(d for d in shape if d not in GEMM_DIMS)
    return "-".join(f"{d}={shape[d]}" for d in dims)


def results_path(runs_dir, arch, backend, kernel, shape):
    return os.path.join(
        runs_dir, f"sweep-{arch}-{backend}-{kernel}-{shape_tag(shape)}.jsonl"
    )


def plan_path(runs_dir, arch, backend, kernel, shape):
    return os.path.join(
        runs_dir, f"plan-{arch}-{backend}-{kernel}-{shape_tag(shape)}.json"
    )


def final_results_path(out):
    """Same-device re-timing of the fastest candidates when several GPUs swept an M."""
    return out[: -len(".jsonl")] + ".final.jsonl"


def results_files(runs_dir, arch, backend, kernel, shape_nk, bucket_dim="M"):
    """All per-bucket-value result files of one (kernel, backend, shape without the bucket dim)."""
    pattern = results_path(
        runs_dir, arch, backend, kernel, dict(shape_nk, **{bucket_dim: "*"})
    )
    return sorted(glob.glob(pattern))


def specialized_filename(config_name, shape, file_dims=("B", "N", "K")):
    """<CONFIG>-[B=..-]N=..-K=..json, the file get_gemm_config() looks up for this shape."""
    dims = "-".join(f"{d}={shape[d]}" for d in file_dims if d in shape)
    return f"{config_name}-{dims}.json"


def bucket_for(value, bounds, bucket_dim="M"):
    """<dim>_LEQ_<smallest bound >= value>, the bucket the loader picks; None above the largest."""
    for bound in bounds:
        if value <= bound:
            return f"{bucket_dim}_LEQ_{bound}"
    return None


def append_record(path, record):
    with open(path, "a+") as f:
        f.seek(0, 2)
        if f.tell() and not _ends_with_newline(path):
            # a worker died mid-line; terminate the fragment so this record stays parsable
            f.write("\n")
        f.write(json.dumps(record) + "\n")


def _ends_with_newline(path):
    with open(path, "rb") as f:
        f.seek(-1, 2)
        return f.read(1) == b"\n"


def iter_records(path):
    """The parsed JSON lines of a results file; a truncated line (killed worker) is skipped."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def read_records(path):
    """Last record per candidate. A 'start' marker never replaces a final record, so a candidate
    the worker died on keeps status 'start'."""
    records = {}
    for record in iter_records(path):
        if record["status"] == "ready":
            continue
        key = "baseline" if record.get("baseline") else config_key(record["config"])
        is_final = record["status"] != "start"
        if is_final or records.get(key, {}).get("status") in (None, "start"):
            records[key] = record
    return records


DIM_HELP = {
    "B": "batch, batched kernels only",
    "K": "logical K, as the input generators take it",
}


def add_shape_args(parser, specs, with_bucket=True, multi_bucket=False):
    """--<dim> for every dim of every kernel; shape_from_args checks the kernel's own.

    A bucket dim (M for GEMM) is left out without with_bucket and takes several values
    with multi_bucket.
    """
    bucket_dims = {spec.bucket_dim for spec in specs}
    for dim in dict.fromkeys(d for spec in specs for d in spec.dims):
        if dim in bucket_dims and not with_bucket:
            continue
        nargs = "+" if dim in bucket_dims and multi_bucket else None
        parser.add_argument(f"--{dim}", type=int, nargs=nargs, help=DIM_HELP.get(dim))


def shape_from_args(args, spec, value=None, with_bucket=True):
    """The kernel's shape in its dims order; value replaces the bucket dim.

    Without with_bucket the bucket dim is left out (the installer's shape).
    """
    stray = sorted(
        d
        for d, v in vars(args).items()
        if d.isupper() and v is not None and d not in spec.dims
    )
    if stray:
        raise SystemExit(
            f"{spec.name} takes --{' --'.join(spec.dims)}, not --{' --'.join(stray)}"
        )
    shape = {}
    for dim in spec.dims:
        given = getattr(args, dim, None)
        if dim == spec.bucket_dim:
            if not with_bucket:
                continue
            given = value if value is not None else given
        if given is None:
            raise SystemExit(f"{spec.name} needs --{dim}")
        shape[dim] = given
    return shape
