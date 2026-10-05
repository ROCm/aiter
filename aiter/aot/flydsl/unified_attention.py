#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for the FlyDSL fp8 unified-attention kernel family.

Unlike the GEMM/MoE families, unified attention has no tuning CSV: the served
config space is a fixed cross-product of build-time specializations for the
Qwen3-VL production shape (num_heads=64, num_kv_heads=4 => GQA-16, head_dim=128,
fp8 QKV, paged + varlen). ``DEFAULT_CSVS = [None]`` is a sentinel so the shared
``collect_aot_jobs`` machinery iterates once and ``parse_csv`` emits the whole
hardcoded job list regardless of its argument.

One builder backs the family (gfx950-only, forced via FLYDSL_GPU_ARCH):
  - prefill: ``_get_kernel`` -> ``build_flash_attn_dualwave_swp_fp8_module``

Each job is compiled by fetching its launcher from the runtime adapter's own
``_get_kernel`` (so the AOT binary is keyed identically
to what dispatch will look up), then invoking it under ``FakeTensorMode`` +
``COMPILE_ONLY=1`` with geometry-only fake tensors -- no GPU, no real memory.

Usage:
    python -m aiter.aot.flydsl.unified_attention
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

from aiter.aot.flydsl.common import (
    collect_aot_jobs,
    compile_only_env,
    override_env,
    run_jobs_parallel,
)

# The whole family is pinned to the gfx950 production config; attention has no
# CSV cu_num column, so the target arch is hardcoded per job.
UNIFIED_ATTENTION_AOT_ARCH = "gfx950"

# Sentinel: no CSV. collect_aot_jobs iterates DEFAULT_CSVS and calls
# parse_csv(None) once, which returns the full hardcoded job list.
DEFAULT_CSVS = [None]

# Pinned production geometry (Qwen3-VL fp8 unified attention on gfx950).
_HEAD_DIM = 128
_NUM_HEADS = 64
_NUM_KV_HEADS = 4  # Qwen3-VL production GQA-16

_OUT_DTYPES = ("bf16", "f16")


def _prefill_jobs() -> list[dict]:
    return [
        {
            "path": "prefill",
            "causal": causal,
            "out_dtype_str": out_dtype,
            "shuffled_kv_cache": shuffled,
            "rescale_threshold": threshold,
            "block_m": block_m,
            "packed_bn64": packed,
        }
        for causal in (True, False)
        for out_dtype in _OUT_DTYPES
        for shuffled in (False, True)
        for threshold in (6.0, 4.0)
        # The adapter routes this GQA-16 shape to packed BN64 (fixed block_m=128).
        for block_m, packed in ((128, True),)
    ]


def parse_csv(_csv=None) -> list[dict]:
    """Return the full unified-attention job list. The CSV argument is ignored
    (this family has no tuning CSV); ``_csv`` is present only to satisfy the
    ``collect_aot_jobs`` / ``_collect_aot_jobs_for`` contract."""
    jobs = _prefill_jobs()
    common = {
        "num_heads": _NUM_HEADS,
        "num_kv_heads": _NUM_KV_HEADS,
        "head_dim": _HEAD_DIM,
        "dtype_str": "fp8",
        "varlen": True,
        "paged": True,
        "arch": UNIFIED_ATTENTION_AOT_ARCH,
    }
    for job in jobs:
        for key, val in common.items():
            job.setdefault(key, val)
        job["kernel_name"] = _kernel_name(job)
    print(
        f"[aiter] FlyDSL unified-attention AOT: {len(jobs)} jobs "
        f"({sum(j['path'] == 'prefill' for j in jobs)} prefill)"
    )
    return jobs


def _kernel_name(job: dict) -> str:
    return (
        f"flydsl_unified_attn_{job['path']}"
        f"_c{int(job['causal'])}_{job['out_dtype_str']}"
        f"_shuf{int(job['shuffled_kv_cache'])}"
        f"_thr{job['rescale_threshold']:g}_bm{job['block_m']}"
        f"{'_bn64pack' if job['packed_bn64'] else ''}"
    )


# Small geometry-only fake shapes (no data, no real memory -- only the C-ABI
# int32 shape fields matter to the compile).
_FAKE_NUM_SEQS = 4
_FAKE_NUM_BLOCKS = 8
_FAKE_MAX_BLOCKS = 4
_FAKE_MAX_SEQLEN_Q = 4  # queries per prefill sequence
_PAGE = 64
_VEC = 16  # 5D-shuffled vectorization width


def _out_torch_dtype(out_dtype_str: str):
    import torch

    return torch.bfloat16 if out_dtype_str == "bf16" else torch.float16


def _fake_kv(num_kv_heads: int, shuffled_kv_cache: bool):
    """Build fake paged K/V pool tensors in the layout the builder expects."""
    import torch

    fp8 = torch.float8_e4m3fn
    d = _HEAD_DIM
    if shuffled_kv_cache:
        # 5D shuffled: K [nb, Hkv, D//x, PAGE, x]; V [nb, Hkv, PAGE//x, D, x].
        k = torch.empty(
            (_FAKE_NUM_BLOCKS, num_kv_heads, d // _VEC, _PAGE, _VEC), dtype=fp8
        )
        v = torch.empty(
            (_FAKE_NUM_BLOCKS, num_kv_heads, _PAGE // _VEC, d, _VEC), dtype=fp8
        )
    else:
        # Linear 4D pool [nb, PAGE, Hkv, D].
        k = torch.empty((_FAKE_NUM_BLOCKS, _PAGE, num_kv_heads, d), dtype=fp8)
        v = torch.empty((_FAKE_NUM_BLOCKS, _PAGE, num_kv_heads, d), dtype=fp8)
    return k, v


def _compile_prefill(job: dict) -> None:
    import torch

    from aiter.ops.flydsl.unified_attention_kernels import _as_i8, _get_kernel

    num_heads = job["num_heads"]
    num_kv_heads = job["num_kv_heads"]
    out_dtype_str = job["out_dtype_str"]
    shuffled_kv_cache = job["shuffled_kv_cache"]
    kernel = _get_kernel(
        num_heads,
        num_kv_heads,
        bool(job["causal"]),
        out_dtype_str,
        shuffled_kv_cache,
        job["rescale_threshold"],
        job["block_m"],
        packed_bn64=job["packed_bn64"],
    )

    fp8 = torch.float8_e4m3fn
    d = _HEAD_DIM
    num_seqs = _FAKE_NUM_SEQS
    max_seqlen_q = _FAKE_MAX_SEQLEN_Q
    total_q = num_seqs * max_seqlen_q

    q = torch.empty((total_q, num_heads, d), dtype=fp8)
    out = torch.empty((total_q, num_heads, d), dtype=_out_torch_dtype(out_dtype_str))
    k, v = _fake_kv(num_kv_heads, shuffled_kv_cache)
    cu_seqlens_q = torch.empty((num_seqs + 1,), dtype=torch.int32)
    cu_seqlens_kv = torch.empty((num_seqs,), dtype=torch.int32)
    block_table = torch.empty((num_seqs, _FAKE_MAX_BLOCKS), dtype=torch.int32)
    q_descale = torch.empty((1,), dtype=torch.float32)
    k_descale = torch.empty((1,), dtype=torch.float32)
    v_descale = torch.empty((1,), dtype=torch.float32)
    stride_kv_n = num_kv_heads * d

    with compile_only_env():
        kernel(
            _as_i8(q).reshape(-1),
            _as_i8(k).reshape(k.shape[0], -1),
            _as_i8(v).reshape(v.shape[0], -1),
            out.reshape(-1),
            num_seqs,
            int(max_seqlen_q),
            stride_kv_n,
            q.stride(0),
            softmax_scale=d**-0.5,
            seq_len_kv=_FAKE_MAX_BLOCKS * _PAGE,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            block_table=block_table.reshape(-1),
            block_table_stride=int(block_table.stride(0)),
            stream=0,
        )


def compile_one_config(**job) -> dict:
    """Compile one unified-attention variant and save it to the FlyDSL cache."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    aot_arch = job.get("arch", UNIFIED_ATTENTION_AOT_ARCH)
    kernel_name = job.get("kernel_name", "flydsl_unified_attn")
    shape_str = (
        f"{kernel_name}  path={job.get('path')} causal={job.get('causal')} "
        f"out={job.get('out_dtype_str')} shuffled={job.get('shuffled_kv_cache')} "
        f"threshold={job.get('rescale_threshold')} block_m={job.get('block_m')}"
    )

    t0 = time.time()
    try:
        with (
            override_env("FLYDSL_GPU_ARCH", aot_arch),
            FakeTensorMode(),
        ):
            if job["path"] == "prefill":
                _compile_prefill(job)
            else:
                raise ValueError(f"Unknown unified-attention path: {job['path']!r}")
        elapsed = time.time() - t0
        return {**job, "compile_time": elapsed, "compile_arch": aot_arch}
    except Exception as e:  # noqa: BLE001
        # Return cleanly (compile_time=None) so the AOT pool marks it "produced no
        # kernel" and does NOT retry -- an escaping exception crashes the worker
        # (exitcode != 0), which the pool misreads as transient -> deadlock.
        print(f"  [FAIL] compile  {shape_str}  arch={aot_arch}: {e}")
        traceback.print_exc()
        return {**job, "compile_time": None, "compile_arch": aot_arch}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    cache_dir = os.path.expanduser(
        os.environ.get("FLYDSL_RUNTIME_CACHE_DIR", "~/.flydsl/cache")
    )
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = cache_dir

    # No CSV: the None sentinel in DEFAULT_CSVS routes collect_aot_jobs straight
    # to parse_csv, which ignores its arg and emits the hardcoded job list.
    jobs = collect_aot_jobs(DEFAULT_CSVS, parse_csv)

    total_t0 = time.time()
    print(f"--- Compiling {len(jobs)} unified-attention kernels ---")
    print(f"  Cache dir: {cache_dir}")
    results = run_jobs_parallel(compile_one_config, jobs)
    total_elapsed = time.time() - total_t0

    ok = sum(1 for r in results if r.get("compile_time") is not None)
    fail = len(results) - ok
    print(f"\nSummary: {ok} ok, {fail} failed in {total_elapsed:.1f}s")
    sys.exit(1 if fail > 0 else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
