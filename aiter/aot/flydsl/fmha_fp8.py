#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for the gfx950 FlyDSL FP8 flash-attention forward.

``flydsl_flash_attn_fp8_func`` picks its compile-time configuration per call:
the rescale threshold from the KV length, BLOCK_M from the KV tile count and
occupancy, and the split-K factor from a makespan model. Every new combination
is a JIT compile on the serving path, where it stalls a request for seconds.

The launcher's cache key is fixed by the ``_build_fp8`` parameters alone:
runtime ints/floats (batch, sequence lengths, strides, scale) enter it by type
only and the flattened tensors by dtype only. So the full set a model can hit is
finite and is enumerated here from the heuristics' own module constants, which
keeps the job list in step with them.

Head shapes come from ``AITER_CONFIGS.AITER_CONFIG_FMHA_FP8_AOT_FILE``: the
header-only ``aiter/configs/fmha_fp8_aot.csv`` merged with every
``model_configs/*_fmha_fp8_aot.csv`` (columns ``cu_num, num_heads,
num_kv_heads, head_dim, head_dim_v, layout, model``), so ``setup.py`` builds
them with the other FlyDSL AOT kinds. ``model`` is informational, but it is
part of the merge key: two models sharing a head shape stay distinct rows
(the merge rejects exact duplicates) and their jobs dedupe by kernel name.

    python -m aiter.aot.flydsl.fmha_fp8                          # config CSVs
    python -m aiter.aot.flydsl.fmha_fp8 --shape 12:12:192:128    # ad hoc

``layout`` is ``varlen_cross`` (packed Q/KV with independent lengths, the
chunked-prefill serving path), ``varlen``, ``dense`` or ``dense_cross``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

from aiter.aot.flydsl.common import (
    _CU_NUM_TO_ARCH,
    collect_aot_jobs,
    compile_only_env,
    cu_num_to_arch,
    dedupe_jobs,
    override_env,
    run_jobs_parallel,
)
from aiter.jit.core import AITER_CONFIGS

DEFAULT_CSVS = [AITER_CONFIGS.AITER_CONFIG_FMHA_FP8_AOT_FILE]
AOT_ARCH = "gfx950"
LAYOUTS = {
    "varlen_cross": (True, True),
    "varlen": (True, False),
    "dense": (False, False),
    "dense_cross": (False, True),
}
# Flags the public wrapper defaults to; they are part of the compile key.
_DEFAULT_FLAGS = {
    "daz": True,
    "lazy_rescale": True,
    "setprio": True,
    "enable_stagger": True,
}


def _layout(value) -> str:
    # get_config_file fills a column a model file lacks with 0, and pandas writes
    # a blank cell back as "" or "nan": all of those mean "not given".
    layout = str(value if value is not None else "").strip()
    layout = "varlen_cross" if layout in ("", "0", "nan") else layout
    if layout not in LAYOUTS:
        raise ValueError(
            f"unknown layout {layout!r}; expected one of {sorted(LAYOUTS)}"
        )
    return layout


def parse_shape(spec: str) -> tuple[int, int, int, int, str]:
    """``"H:Hkv:D:Dv[@layout]"`` -> ``(H, Hkv, D, Dv, layout)``."""
    dims, _, layout = spec.strip().partition("@")
    h, hkv, d, dv = (int(x) for x in dims.split(":"))
    return h, hkv, d, dv, _layout(layout)


def unsupported_reason(h: int, hkv: int, d: int, dv: int) -> str | None:
    """Why the kernel cannot serve this head shape, or None. The runtime gate
    (flydsl_flash_attn_fp8_supported) uses the same check, so a rejected shape
    is never dispatched and its kernels would only fail the build."""
    from aiter.ops.flydsl.kernels.flash_attn_func_fp8_gfx950 import (
        _fp8_config_reason,
    )

    return _fp8_config_reason(h, hkv, d, dv)


def _variant_space(causal: bool, cross: bool) -> list[tuple[float, int, int, int]]:
    """Every ``(rescale_threshold, block_m, num_kv_splits, batch_interleave_group)``
    the wrapper can pick, as an upper bound derived from its heuristics.

    - threshold: 6.0 while seqlen_kv <= _FP8_LONG_SEQ, else 4.0.
    - block_m: the narrow tile only while kv_tiles <= _FP8_NARROW_MAX_KV_TILES.
    - splits: a candidate s > 1 needs kv_tiles // s >= _FP8_AUTOSPLIT_MIN_TILES,
      so the largest kv_tiles a (threshold, block_m) bucket admits bounds s.
      Causal self-attention always lands in the skewed branch (kept < 0.75 once
      split-K is allowed), which only returns 1 or 2.
    - interleave group: > 1 only for causal self-attention without split-K.
    """
    from aiter.ops.flydsl.kernels import flash_attn_func_fp8_gfx950 as fa

    wide = fa.DUALWAVE_SWP_BLOCK_M
    narrow = wide // 2
    long_tiles = fa._FP8_LONG_SEQ // fa._FP8_BLOCK_N
    narrow_tiles = fa._FP8_NARROW_MAX_KV_TILES
    unbounded = None

    buckets = []  # (threshold, block_m, max kv_tiles in the bucket or None)
    buckets.append((6.0, wide, long_tiles))
    buckets.append((6.0, narrow, min(narrow_tiles, long_tiles)))
    buckets.append((4.0, wide, unbounded))
    if narrow_tiles > long_tiles:
        buckets.append((4.0, narrow, narrow_tiles))

    max_splits = 2 if causal and not cross else max(fa._FP8_AUTOSPLIT_CANDIDATES)
    out = []
    for thr, bm, max_tiles in buckets:
        for s in fa._FP8_AUTOSPLIT_CANDIDATES:
            if s > max_splits:
                continue
            if (
                s > 1
                and max_tiles is not unbounded
                and max_tiles // s < fa._FP8_AUTOSPLIT_MIN_TILES
            ):
                continue
            groups = {1}
            if causal and not cross and s == 1:
                groups.add(fa._FP8_BATCH_INTERLEAVE_GROUP)
            for g in sorted(groups):
                out.append((thr, bm, s, g))
    return out


def jobs_for_shape(h: int, hkv: int, d: int, dv: int, layout: str) -> list[dict]:
    varlen, cross = LAYOUTS[layout]
    jobs = []
    for causal in (True, False):
        for return_lse in (False, True):
            for thr, bm, s, g in _variant_space(causal, cross):
                jobs.append(
                    {
                        "kernel_name": (
                            f"fmha_fp8_h{h}_hkv{hkv}_d{d}_dv{dv}_{layout}"
                            f"_c{int(causal)}_lse{int(return_lse)}"
                            f"_thr{thr:g}_bm{bm}_s{s}_g{g}"
                        ),
                        "num_heads": h,
                        "num_kv_heads": hkv,
                        "head_dim": d,
                        "head_dim_v": dv,
                        "varlen": varlen,
                        "cross_seqlen": cross,
                        "causal": causal,
                        "return_lse": return_lse,
                        "rescale_threshold": thr,
                        "block_m": bm,
                        "num_kv_splits": s,
                        "batch_interleave_group": g,
                    }
                )
    return jobs


def parse_csv(csv_path: str) -> list[dict]:
    """One row per head shape -> every variant that shape can reach.

    A row that is malformed, not gfx950, or a shape the kernel rejects is
    skipped with a warning rather than failing the whole build.
    """
    jobs = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = csv.DictReader(line for line in f if not line.lstrip().startswith("#"))
        for row in rows:
            try:
                cu_num = int(float(row["cu_num"]))
                h, hkv, d, dv = (
                    int(float(row[k]))
                    for k in ("num_heads", "num_kv_heads", "head_dim", "head_dim_v")
                )
                layout = _layout(row.get("layout"))
            except (KeyError, TypeError, ValueError) as error:
                print(f"  [WARN] {csv_path}: malformed row {dict(row)}: {error}")
                continue
            if cu_num not in _CU_NUM_TO_ARCH:
                print(
                    f"  [WARN] {csv_path}: unknown cu_num={cu_num}, assuming {AOT_ARCH}"
                )
            arch = cu_num_to_arch(cu_num)
            if arch != AOT_ARCH:
                print(
                    f"  [WARN] {csv_path}: cu_num={cu_num} is {arch}; "
                    f"the FP8 FMHA kernel is {AOT_ARCH}-only, skipping"
                )
                continue
            reason = unsupported_reason(h, hkv, d, dv)
            if reason:
                print(
                    f"  [WARN] {csv_path}: {h}:{hkv}:{d}:{dv} unsupported ({reason}), skipping"
                )
                continue
            jobs.extend(jobs_for_shape(h, hkv, d, dv, layout))
    return jobs


def _compile_to_cache(
    *,
    num_heads,
    num_kv_heads,
    head_dim,
    head_dim_v,
    varlen,
    cross_seqlen,
    causal,
    return_lse,
    rescale_threshold,
    block_m,
    num_kv_splits,
    batch_interleave_group,
):
    import torch

    from aiter.ops.flydsl.kernels.flash_attn_func_fp8_gfx950 import _build_fp8

    exe = _build_fp8(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        causal=causal,
        rescale_threshold=rescale_threshold,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        varlen=varlen,
        cross_seqlen=cross_seqlen,
        num_kv_splits=num_kv_splits,
        block_m=block_m,
        batch_interleave_group=batch_interleave_group,
        return_lse=return_lse,
        **_DEFAULT_FLAGS,
    )

    # Mirror the runtime call exactly (flash_attn_func_fp8_gfx950.py: the exe(...)
    # at the end of flydsl_flash_attn_fp8_func): flat tensors, the same optional
    # kwargs present or absent, the same dtypes. Values and sizes never reach
    # the key; a missing or extra kwarg, or a wrong dtype, silently would.
    cpu = torch.device("cpu")

    def t(n, dtype):
        return torch.empty(n, dtype=dtype, device=cpu)

    f8, bf16, f32, i32 = torch.float8_e4m3fn, torch.bfloat16, torch.float32, torch.int32
    batch, seq = batch_interleave_group, 512
    kwargs = {
        "stream": None,
        "softmax_scale": head_dim**-0.5,
        "q_descale": t(1, f32),
        "k_descale": t(1, f32),
        "v_descale": t(1, f32),
    }
    if return_lse:
        kwargs["lse"] = t(num_heads * seq, f32)
        kwargs["lse_stride_h"] = seq
    if num_kv_splits > 1:
        kwargs["workspace"] = t(1024, f32)
    if varlen:
        kwargs.update(cu_seqlens_q=t(batch + 1, i32), cu_seqlens_kv=t(batch + 1, i32))
    if cross_seqlen:
        kwargs["seq_len_kv"] = seq
    exe.compile(
        t(seq * num_heads * head_dim, f8),
        t(seq * num_kv_heads * head_dim, f8),
        t(seq * num_kv_heads * head_dim_v, f8),
        t(seq * num_heads * head_dim_v, bf16),
        batch,
        seq,
        **kwargs,
    )


def compile_one_config(**job) -> dict:
    result = {**job, "compile_time": None}
    params = {k: v for k, v in job.items() if k != "kernel_name"}
    from torch._subclasses.fake_tensor import FakeTensorMode

    started = time.time()
    try:
        # COMPILE_ONLY is mandatory: without it flyc.compile also launches.
        with (
            override_env("FLYDSL_GPU_ARCH", AOT_ARCH),
            compile_only_env(),
            FakeTensorMode(),
        ):
            _compile_to_cache(**params)
        result["compile_time"] = time.time() - started
    except Exception as error:  # noqa: BLE001
        print(f"  [FAIL] {job['kernel_name']}: {error}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--csv",
        nargs="+",
        default=DEFAULT_CSVS,
        help="head-shape CSV(s) (default: AITER_CONFIG_FMHA_FP8_AOT_FILE)",
    )
    parser.add_argument(
        "--shape",
        action="append",
        default=[],
        help="H:Hkv:D:Dv[@layout], repeatable; replaces --csv",
    )
    parser.add_argument("--list", action="store_true", help="print the jobs and exit")
    args = parser.parse_args()
    if args.shape:
        jobs = []
        for spec in args.shape:
            h, hkv, d, dv, layout = parse_shape(spec)
            reason = unsupported_reason(h, hkv, d, dv)
            if reason:
                print(f"Error: {spec} is unsupported by the FP8 FMHA kernel: {reason}")
                sys.exit(1)
            jobs.extend(jobs_for_shape(h, hkv, d, dv, layout))
        jobs = dedupe_jobs(jobs)
    else:
        for csv_path in args.csv:
            if not os.path.isfile(csv_path):
                print(f"Error: CSV file not found: {csv_path}")
                sys.exit(1)
        jobs = collect_aot_jobs(args.csv, parse_csv)
    if args.list:
        for job in jobs:
            print(job["kernel_name"])
        print(f"{len(jobs)} jobs")
        return
    if not jobs:
        print("no head shapes configured; nothing to compile")
        return
    results = run_jobs_parallel(compile_one_config, jobs)
    failed = sum(result["compile_time"] is None for result in results)
    print(f"Compiled: {len(results) - failed} ok, {failed} failed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
