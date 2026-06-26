#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for the FlyDSL PS-reduce kernel used by pa_decode_gluon.

The set of compile-time constant combinations to precompile is hand-specified as
a cartesian product of axes in ``PA_DECODE_AOT_SPEC``. Each combination is
compiled under ``COMPILE_ONLY=1`` with CPU dummy tensors, calling the same
``compile_pa_decode_ps_reduce_flydsl`` builder the runtime uses, so the AOT cache
key equals the runtime lookup.

Runs on a GPU host (importing pa_decode_gluon needs a GPU); arch is auto-detected
by FlyDSL and matches the runtime arch on the same host.

Usage:
    python -m aiter.aot.flydsl.pa_decode
"""

import argparse
import itertools
import time

import torch

from aiter.aot.flydsl.common import compile_only_env
from aiter.ops.triton.gluon.pa_decode_gluon import (
    FLYDSL_PS_REDUCE_AVAILABLE,
    PA_DECODE_MAX_SPLITS,
    _flydsl_dtype_str,
    compile_pa_decode_ps_reduce_flydsl,
)

# Hand-edited cartesian-product spec. Replace these starter values with the real
# target model configs. Axes map 1:1 onto the kernel's compile-time constants
# (sink_dtype is pinned to output_dtype when use_sinks is False; see _expand_spec).
PA_DECODE_AOT_SPEC = {
    "head_size": [64],
    "output_dtype": [torch.bfloat16],
    "logits_dtype": [torch.bfloat16],
    "use_sinks": [True],
    "sink_dtype": [torch.bfloat16],
    "max_context_partition_num": list(range(1, PA_DECODE_MAX_SPLITS + 1)),
}

_AXES = (
    "head_size",
    "output_dtype",
    "logits_dtype",
    "use_sinks",
    "sink_dtype",
    "max_context_partition_num",
)

# query_seq_len / query_group_size are passed at launch time and no longer affect
# the compiled kernel's cache key, so they are not AOT axes. Fixed dummy values
# are used only to shape the COMPILE_ONLY dummy tensors.
_DUMMY_QUERY_SEQ_LEN = 1
_DUMMY_QUERY_GROUP_SIZE = 8


def _job_kernel_name(job: dict) -> str:
    return (
        "pa_decode_ps_reduce "
        f"hs={job['head_size']} out={job['output_dtype']} "
        f"logits={job['logits_dtype']} sink={job['use_sinks']}/{job['sink_dtype']} "
        f"n={job['max_context_partition_num']}"
    )


def _expand_spec(spec: dict) -> list[dict]:
    """Cartesian product of the spec axes into a deduped list of job dicts.

    When use_sinks is False, sink_dtype is pinned to output_dtype to match the
    runtime funnel (which sets sink_dtype = output dtype when no sink is passed),
    then duplicates are removed.
    """
    value_lists = [spec[axis] for axis in _AXES]
    seen = set()
    jobs = []
    for combo in itertools.product(*value_lists):
        job = dict(zip(_AXES, combo))
        if not job["use_sinks"]:
            job["sink_dtype"] = job["output_dtype"]
        key = tuple(job[axis] for axis in _AXES)
        if key in seen:
            continue
        seen.add(key)
        job["kernel_name"] = _job_kernel_name(job)
        jobs.append(job)
    return jobs


def collect_jobs() -> list[dict]:
    """Jobs for the build-time AOT harness. Empty when FlyDSL is unavailable."""
    if not FLYDSL_PS_REDUCE_AVAILABLE:
        return []
    return _expand_spec(PA_DECODE_AOT_SPEC)


def compile_one_config(
    *,
    head_size: int,
    output_dtype: torch.dtype,
    logits_dtype: torch.dtype,
    use_sinks: bool,
    sink_dtype: torch.dtype,
    max_context_partition_num: int,
    kernel_name: str = "",
) -> dict:
    """Compile one PS-reduce variant to the FlyDSL cache (COMPILE_ONLY, no exec).

    Uses CPU dummy tensors and the inner ``compiled["launch"]`` (default stream)
    so it is safe inside a forked process pool and needs no live CUDA context.
    Only dtype/rank/contiguity of the dummy tensors matter for the cache key, so
    sizes are minimal (batch=1, num_kv_heads=1). query_seq_len/query_group_size
    are launch-time args (not cache keys), so fixed dummy values are used here.
    """
    result = {"kernel_name": kernel_name, "compile_time": None}
    n = max_context_partition_num
    query_seq_len = _DUMMY_QUERY_SEQ_LEN
    query_group_size = _DUMMY_QUERY_GROUP_SIZE
    eq_group = query_seq_len * query_group_size
    dev = torch.device("cpu")

    output = torch.empty(
        1, query_seq_len, 1, query_group_size, head_size, device=dev, dtype=output_dtype
    )
    exp_sums = torch.zeros(1, 1, n, eq_group, device=dev, dtype=torch.float32)
    max_logits = torch.zeros(1, 1, n, eq_group, device=dev, dtype=torch.float32)
    temporary_output = torch.zeros(
        1, 1, n, eq_group, head_size, device=dev, dtype=logits_dtype
    )
    if use_sinks:
        # num_query_heads = num_kv_heads(1) * query_group_size
        sink = torch.empty(query_group_size, device=dev, dtype=sink_dtype)
        sink_dtype_str = _flydsl_dtype_str(sink_dtype)
    else:
        sink = torch.empty(0, device=dev, dtype=output_dtype)
        sink_dtype_str = _flydsl_dtype_str(output_dtype)

    t0 = time.time()
    try:
        with compile_only_env():
            compiled = compile_pa_decode_ps_reduce_flydsl(
                max_context_partition_num=n,
                head_size=head_size,
                output_dtype_str=_flydsl_dtype_str(output_dtype),
                logits_dtype_str=_flydsl_dtype_str(logits_dtype),
                sink_dtype_str=sink_dtype_str,
                use_sinks=use_sinks,
            )
            # Inner launch: omit the stream arg (defaults to fx.Stream(None)).
            compiled["launch"](
                output,
                exp_sums,
                max_logits,
                temporary_output,
                sink,
                output.stride(0),
                output.stride(1),
                output.stride(2),
                output.stride(3),
                exp_sums.stride(0),
                exp_sums.stride(1),
                exp_sums.stride(2),
                temporary_output.stride(0),
                temporary_output.stride(1),
                temporary_output.stride(2),
                temporary_output.stride(3),
                query_seq_len,
                query_group_size,
                output.shape[0],
                output.shape[2],
            )
        result["compile_time"] = time.time() - t0
        print(f"  [OK] compile  {result['compile_time']:6.1f}s  {kernel_name}")
    except Exception as e:
        print(f"  [FAIL] compile  {kernel_name}: {e}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="AOT pre-compile FlyDSL PS-reduce kernels from PA_DECODE_AOT_SPEC",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.parse_args()

    jobs = collect_jobs()
    print("=" * 72)
    print("FlyDSL pa_decode PS-reduce AOT Pre-compilation")
    print(f"  Total jobs: {len(jobs)}")
    print("=" * 72)

    t0 = time.time()
    ok = 0
    for i, job in enumerate(jobs, 1):
        res = compile_one_config(**job)
        if res["compile_time"] is not None:
            ok += 1
        print(f"  ... {i}/{len(jobs)} complete")
    elapsed = time.time() - t0

    print("=" * 72)
    print(f"  compiled {ok} ok, {len(jobs) - ok} failed in {elapsed:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
