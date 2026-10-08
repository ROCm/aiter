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

import os
import time
from contextlib import contextmanager

import torch

from aiter.aot.flydsl.common import compile_only_env, override_env

_QSA_ARCHS = ("gfx942", "gfx950")

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


def _qsa_aot_targets() -> list[tuple[str, int]]:
    """Archs this job compiles, with the CU count that selects the tile.

    ``GPU_ARCHS`` is the wheel list. CI sets ``gfx942;gfx950`` and the
    container has no device, so that list is the whole answer. Without
    it, a visible GPU compiles its own arch and CU count. With neither,
    both supported archs use the full-chip counts.
    """
    from aiter.jit.utils.build_targets import GFX_CU_NUM_MAP

    raw = os.environ.get("GPU_ARCHS", "").strip()
    if raw and raw.lower() != "native":
        archs = [part.strip() for part in raw.split(";") if part.strip() in _QSA_ARCHS]
        if not archs:
            raise RuntimeError(
                f"GPU_ARCHS={raw!r} names no QSA arch ({', '.join(_QSA_ARCHS)})"
            )
        if len(archs) == 1 and os.environ.get("CU_NUM"):
            return [(archs[0], int(os.environ["CU_NUM"]))]
        return [(arch, GFX_CU_NUM_MAP[arch]) for arch in archs]
    if torch.cuda.is_available():
        from aiter.jit.utils.chip_info import get_cu_num
        from aiter.ops.flydsl.kernels.qsa.arch import qsa_device_arch

        arch = qsa_device_arch(torch.cuda.get_device_properties(0).gcnArchName)
        return [(arch, int(get_cu_num()))]
    return [(arch, GFX_CU_NUM_MAP[arch]) for arch in _QSA_ARCHS]


@contextmanager
def _qsa_compile_device(arch: str, cu_num: int):
    """Present ``arch`` to the QSA wrappers without allocating a device buffer.

    The wrappers read ``gcnArchName`` and, on gfx942, the CU count. Fake
    tensors satisfy the CUDA checks. ``COMPILE_ONLY`` never launches, so
    the stream is only a value the launcher records.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode

    from aiter.jit.utils.chip_info import get_cu_num

    class _Props:
        gcnArchName = arch

    class _Stream:
        cuda_stream = 0

    saved = (
        torch.cuda.get_device_properties,
        torch.cuda.current_stream,
        torch.cuda.is_current_stream_capturing,
    )
    torch.cuda.get_device_properties = lambda device=None: _Props()
    torch.cuda.current_stream = lambda device=None: _Stream()
    torch.cuda.is_current_stream_capturing = lambda: False
    try:
        with (
            FakeTensorMode(),
            override_env("ARCH", arch),
            override_env("FLYDSL_GPU_ARCH", arch),
            override_env("CU_NUM", str(cu_num)),
        ):
            get_cu_num.cache_clear()
            try:
                yield
            finally:
                get_cu_num.cache_clear()
    finally:
        (
            torch.cuda.get_device_properties,
            torch.cuda.current_stream,
            torch.cuda.is_current_stream_capturing,
        ) = saved


def _compile_k1(job):
    from aiter.ops.flydsl.qsa import qsa_k1_block_ids

    bf16, i32 = torch.bfloat16, torch.int32
    device = torch.device("cuda")
    rows = job["m"]
    seq_len = job["seq_len"]
    page = job["page_size"]
    heads = job["heads"]
    head_dim = job["head_dim"]
    n_blocks = seq_len // job["compress_ratio"]
    n_pages = n_blocks // page
    for arch, cu_num in _qsa_aot_targets():
        with _qsa_compile_device(arch, cu_num):
            q = torch.empty(rows, heads, head_dim, dtype=bf16, device=device)
            k_cache = torch.empty(
                n_pages,
                page,
                job["kv_heads"],
                head_dim,
                dtype=bf16,
                device=device,
            )
            table = torch.zeros(1, n_pages, dtype=i32, device=device)
            qpos = torch.zeros(rows, dtype=i32, device=device)
            slen = torch.full((1,), seq_len, dtype=i32, device=device)
            token_to_req = torch.zeros(rows, dtype=i32, device=device)
            qsa_k1_block_ids(
                q, k_cache, table, token_to_req, qpos, slen, heads=(heads,)
            )


def _compile_k2(job):
    from aiter.ops.flydsl.qsa import qsa_k2

    bf16, i32 = torch.bfloat16, torch.int32
    device = torch.device("cuda")
    rows = job["m"]
    seq_len = job["seq_len"]
    page = job["page_size"]
    hq, hkv, head_dim = job["hq"], job["hkv"], job["head_dim"]
    width = job["width"]
    n_pages = (seq_len + page - 1) // page
    for arch, cu_num in _qsa_aot_targets():
        with _qsa_compile_device(arch, cu_num):
            q = torch.empty(rows, hq, head_dim, dtype=bf16, device=device)
            k_cache = torch.empty(
                n_pages, page, hkv, head_dim, dtype=bf16, device=device
            )
            v_cache = torch.empty_like(k_cache)
            table = torch.zeros(1, n_pages, dtype=i32, device=device)
            indices = torch.zeros(rows, width, dtype=i32, device=device)
            indices[:, -1] = -1
            token_to_req = torch.zeros(rows, dtype=i32, device=device)
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
