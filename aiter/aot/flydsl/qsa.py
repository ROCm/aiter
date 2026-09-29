# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT jobs for the family A QSA compiles this campaign already launches.

K1 is the page-16 emit kernel, the one-row long-row scorer (decode
M=1 and M=8 share it), and the 16-row prefill scorer. K2 is the three
bar launches. Family B, H=8, and the other sweep values are not here.
The JIT wrappers stay the runtime path; this module is only imported
by the AOT collector.
"""

from __future__ import annotations

import time

import torch

from aiter.aot.flydsl.common import compile_only_env

# (kernel_name, op, m, seq_len). K2 width and head counts are family A.
_FAMILY_A_LAUNCHES = (
    ("qsa_k1_emit_family_a", "k1", 1, 512),
    ("qsa_k1_long_row_family_a_decode", "k1", 1, 32768),
    ("qsa_k1_long_row_family_a_prefill", "k1", 512, 8192),
    ("qsa_k2_family_a_m1_l32768", "k2", 1, 32768),
    ("qsa_k2_family_a_m8_l32768", "k2", 8, 32768),
    ("qsa_k2_family_a_m512_l8192", "k2", 512, 8192),
)


def default_jobs(launches=_FAMILY_A_LAUNCHES):
    """One job per launch. An empty launch list is an empty job list."""
    jobs = []
    for kernel_name, op, rows, seq_len in launches:
        if op == "k1":
            jobs.append(
                {
                    "kernel_name": kernel_name,
                    "op": "k1",
                    "m": rows,
                    "seq_len": seq_len,
                    "heads": 4,
                    "head_dim": 128,
                    "kv_heads": 1,
                    "page_size": 16,
                    "compress_ratio": 4,
                }
            )
        elif op == "k2":
            jobs.append(
                {
                    "kernel_name": kernel_name,
                    "op": "k2",
                    "m": rows,
                    "seq_len": seq_len,
                    "hq": 24,
                    "hkv": 2,
                    "head_dim": 256,
                    "page_size": 16,
                    "width": 2051,
                }
            )
        else:
            raise ValueError(f"unknown QSA AOT op {op!r}")
    return jobs


def _compile_k1(job):
    from aiter import dtypes
    from aiter.ops.flydsl.qsa import qsa_k1_block_ids

    device = torch.device("cuda")
    rows = job["m"]
    seq_len = job["seq_len"]
    page = job["page_size"]
    heads = job["heads"]
    head_dim = job["head_dim"]
    n_blocks = seq_len // job["compress_ratio"]
    n_pages = n_blocks // page
    q = torch.empty(rows, heads, head_dim, dtype=dtypes.bf16, device=device)
    k_cache = torch.empty(
        n_pages, page, job["kv_heads"], head_dim, dtype=dtypes.bf16, device=device
    )
    table = torch.zeros(1, n_pages, dtype=dtypes.i32, device=device)
    qpos = torch.zeros(rows, dtype=dtypes.i32, device=device)
    slen = torch.full((1,), seq_len, dtype=dtypes.i32, device=device)
    token_to_req = torch.zeros(rows, dtype=dtypes.i32, device=device)
    qsa_k1_block_ids(q, k_cache, table, token_to_req, qpos, slen, heads=(heads,))


def _compile_k2(job):
    from aiter import dtypes
    from aiter.ops.flydsl.qsa import qsa_k2

    device = torch.device("cuda")
    rows = job["m"]
    seq_len = job["seq_len"]
    page = job["page_size"]
    hq, hkv, head_dim = job["hq"], job["hkv"], job["head_dim"]
    width = job["width"]
    n_pages = (seq_len + page - 1) // page
    q = torch.empty(rows, hq, head_dim, dtype=dtypes.bf16, device=device)
    k_cache = torch.empty(
        n_pages, page, hkv, head_dim, dtype=dtypes.bf16, device=device
    )
    v_cache = torch.empty_like(k_cache)
    table = torch.zeros(1, n_pages, dtype=dtypes.i32, device=device)
    indices = torch.zeros(rows, width, dtype=dtypes.i32, device=device)
    indices[:, -1] = -1
    token_to_req = torch.zeros(rows, dtype=dtypes.i32, device=device)
    qsa_k2(q, k_cache, v_cache, indices, table, token_to_req)


def compile_one_config(**job) -> dict:
    """Compile one family A launch. ``COMPILE_ONLY`` keeps the wrapper from launching."""
    result = {**job, "compile_time": None}
    started = time.time()
    try:
        with compile_only_env():
            if job["op"] == "k1":
                _compile_k1(job)
            elif job["op"] == "k2":
                _compile_k2(job)
            else:
                raise ValueError(f"unknown QSA AOT op {job['op']!r}")
        result["compile_time"] = time.time() - started
    except Exception as error:  # noqa: BLE001
        print(f"  [FAIL] {job['kernel_name']}: {error}")
    return result
