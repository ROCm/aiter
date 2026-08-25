#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for FlyDSL GEMM kernels from aiter tuned CSV configs.

Reads tuned GEMM CSV config files, extracts all unique FlyDSL kernel entries,
and pre-compiles them into the FlyDSL cache. The default CSV set is resolved
through ``AITER_CONFIGS`` so model-specific tuned CSVs can be merged the same
way as runtime JIT config lookup.

Supported kernel families:
  - ``flydsl_gemm2_*``                        split-K HGEMM kernels
  - ``flydsl_bpreshuflle_*``                  a8w8 preshuffle GEMM kernels
  - ``flydsl_bpreshuffle_8w_*``               gfx950 8-wave a8w8 ptpc GEMM kernels
  - ``flydsl_bpreshuffle_wmma_*``             gfx1250 a8w8 ptpc GEMM kernels
  - ``flydsl_mxfp8_128_bpreshuffle_wmma_*``   gfx1250 mxfp8_128 GEMM kernels
  - ``flydsl_decode_*``                       exact-shape BF16 decode GEMM kernels
  - ``flydsl_mxfp8_128_bpreshuffle_compute_wmma_*`` gfx1250 compute-bound mxfp8_128 kernels

Usage:
    # Compile all unique FlyDSL GEMM kernels from default CSVs
    python -m aiter.aot.flydsl.gemm

    # Custom CSV file(s)
    python -m aiter.aot.flydsl.gemm --csv /path/to/config1.csv /path/to/config2.csv

Environment variables:
    FLYDSL_RUNTIME_CACHE_DIR  Cache directory (default: ~/.flydsl/cache)
    GPU_ARCHS / ARCH          Target GPU architecture information for logging.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time

import flydsl.expr as fx

from aiter.aot.flydsl.common import (
    collect_aot_jobs,
    compile_only_env,
    cu_num_to_arch,
    job_identity,
    override_env,
    run_jobs_parallel,
)
from aiter.jit.core import AITER_CONFIGS
from aiter.ops.flydsl.bpreshuffle_gemm_gfx1250 import (
    parse_wmma_kernel_name as parse_ptpc_wmma_kernel_name,
)
from aiter.ops.flydsl.gemm_a8w8_bpreshuffle_8wave import (
    compile_8wave_gemm,
    parse_8wave_kernel_name,
)
from aiter.ops.flydsl.gemm_kernels import (
    SPLIT_K_SEMAPHORE_MAX_LEN,
    compile_gemm_decode_bf16,
    get_flydsl_splitk_hgemm_kernel_params,
    parse_gemm_decode_kernel_name,
)
from aiter.ops.flydsl.kernels.hgemm_dispatch import compile_flydsl_hgemm_kernel
from aiter.ops.flydsl.kernels.preshuffle_gemm import compile_preshuffle_gemm
from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg, unused_tensor_arg
from aiter.ops.flydsl.mxfp8_128_bpreshuffle_gemm_gfx1250 import (
    BLOCK_K as SCALE_BLOCK_SIZE,
)
from aiter.ops.flydsl.mxfp8_128_bpreshuffle_gemm_gfx1250 import (
    COMPUTE_WMMA_NAME_PREFIX as MXFP8_128_COMPUTE_WMMA_PREFIX,
)
from aiter.ops.flydsl.mxfp8_128_bpreshuffle_gemm_gfx1250 import (
    WMMA_NAME_PREFIX as MXFP8_128_WMMA_PREFIX,
)
from aiter.ops.flydsl.mxfp8_128_bpreshuffle_gemm_gfx1250 import (
    is_compute_wmma_kernel_name,
)
from aiter.ops.flydsl.mxfp8_128_bpreshuffle_gemm_gfx1250 import (
    parse_wmma_kernel_name as parse_mxfp8_128_wmma_kernel_name,
)

# Keep the default AOT coverage aligned with runtime config resolution.
DEFAULT_CSVS = [
    AITER_CONFIGS.AITER_CONFIG_GEMM_A4W4_FILE,
    AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_FILE,
    AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE,
    AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE,
    AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE,
    AITER_CONFIGS.AITER_CONFIG_A8W8_BATCHED_GEMM_FILE,
    AITER_CONFIGS.AITER_CONFIG_BF16_BATCHED_GEMM_FILE,
    AITER_CONFIGS.AITER_CONFIG_GEMM_BF16_FILE,
]
GEMM_AOT_ARCH_DEFAULT = "gfx950"

_PRESHUFFLE_RE = re.compile(
    r"^flydsl_bpreshuflle_"
    r"(?P<tile_m>\d+)x(?P<tile_n>\d+)x(?P<tile_k>\d+)_"
    r"(?P<qa>[A-Z0-9]+)_(?P<qw>[A-Z0-9]+)_(?P<out>[A-Z0-9]+)_"
    r"(?P<async_copy>\d+)x(?P<waves_per_eu>\d+)(?:x(?P<xcd_swizzle>\d+))?(?:x(?P<lds_stage>\d+))?_"
    r"(?P<scheduler>[A-Za-z][A-Za-z0-9]*)$"
)
_SHORT_DTYPE = {
    "F8": "fp8",
    "I8": "int8",
    "B16": "bf16",
    "F16": "fp16",
}


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    normalized = value.strip().lower()
    if normalized == "":
        return False
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"Expected True/False, got {value!r}")


def _parse_preshuffle_kernel_name(name: str) -> dict | None:
    m = _PRESHUFFLE_RE.fullmatch(name)
    if m is None:
        return None

    qa = _SHORT_DTYPE.get(m.group("qa"))
    qw = _SHORT_DTYPE.get(m.group("qw"))
    out = _SHORT_DTYPE.get(m.group("out"))
    if qa is None or qw is None or out is None:
        return None
    if qa != qw:
        raise ValueError(
            f"Unsupported mixed preshuffle input dtypes in {name!r}: {qa} vs {qw}"
        )

    return {
        "kind": "preshuffle",
        "tile_m": int(m.group("tile_m")),
        "tile_n": int(m.group("tile_n")),
        "tile_k": int(m.group("tile_k")),
        "in_dtype": qa,
        "out_dtype": out,
        "use_async_copy": int(m.group("async_copy")),
        "waves_per_eu": int(m.group("waves_per_eu")),
        "xcd_swizzle": int(m.group("xcd_swizzle")) if m.group("xcd_swizzle") else 0,
        "lds_stage": int(m.group("lds_stage")) if m.group("lds_stage") else 2,
        "scheduler": m.group("scheduler"),
    }


def _parse_decode_row(row: dict[str, str | None], kernel_name: str) -> dict:
    m = int(row["M"])
    n = int(row["N"])
    k = int(row["K"])
    cu_num = int(row["cu_num"])
    csv_arch = (row.get("gfx") or "").strip()
    name_arch, name_m, name_n, name_k, config, name_has_bias = (
        parse_gemm_decode_kernel_name(kernel_name)
    )
    if (name_m, name_n, name_k) != (m, n, k):
        raise ValueError(
            "FlyDSL decode kernel name shape does not match CSV row: "
            f"name={(name_m, name_n, name_k)}, row={(m, n, k)}"
        )
    if csv_arch and csv_arch != name_arch:
        raise ValueError(
            f"FlyDSL decode architecture mismatch: name={name_arch}, csv={csv_arch}"
        )
    has_bias = _parse_bool(row.get("bias"))
    if name_has_bias != has_bias:
        raise ValueError("FlyDSL decode CSV bias metadata does not match kernel name")
    if (row.get("dtype") or "").strip() != "torch.bfloat16":
        raise ValueError("FlyDSL decode AOT requires BF16 input dtype")
    if (row.get("outdtype") or "").strip() != "torch.bfloat16":
        raise ValueError("FlyDSL decode AOT requires BF16 output dtype")
    if _parse_bool(row.get("scaleAB")):
        raise ValueError("FlyDSL decode AOT does not support scaling")
    if _parse_bool(row.get("bpreshuffle")):
        raise ValueError("FlyDSL decode AOT does not support preshuffled weights")

    return {
        "kind": "decode",
        "config": config,
        "m": m,
        "n": n,
        "k": k,
        "cu_num": cu_num,
        "gfx": csv_arch or name_arch,
        "has_bias": has_bias,
    }


def parse_csv(csv_path: str):
    """Parse a GEMM tuned CSV and return a list of unique FlyDSL compile jobs."""
    jobs = []
    seen = set()

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kernel_name = (row.get("kernelName") or "").strip()
            libtype = (row.get("libtype") or "").strip()
            if libtype == "flydsl_decode":
                if not kernel_name:
                    raise ValueError("FlyDSL decode CSV row requires kernelName")
                params = _parse_decode_row(row, kernel_name)
                job = {
                    "kernel_name": kernel_name,
                    **params,
                }
                key = job_identity(job)
                if key not in seen:
                    seen.add(key)
                    jobs.append(job)
                continue
            if libtype != "flydsl" or not kernel_name.startswith("flydsl_"):
                continue

            m = int(row["M"])
            n = int(row["N"])
            k = int(row["K"])
            cu_num = int(row.get("cu_num", "0"))
            gfx = row.get("gfx", "").strip()

            if kernel_name.startswith("flydsl_bpreshuflle_"):
                params = _parse_preshuffle_kernel_name(kernel_name)
            elif kernel_name.startswith("flydsl_bpreshuffle_8w_"):
                params = parse_8wave_kernel_name(kernel_name)
                if params is not None:
                    params = dict(params)
                    params["kind"] = "8wave"
            elif kernel_name.startswith(
                (
                    f"{MXFP8_128_WMMA_PREFIX}_",
                    f"{MXFP8_128_COMPUTE_WMMA_PREFIX}_",
                )
            ):
                params = parse_mxfp8_128_wmma_kernel_name(kernel_name)
                if params is not None:
                    params = dict(params)
                    params["kind"] = "mxfp8_128_wmma"
            elif kernel_name.startswith("flydsl_bpreshuffle_wmma_"):
                params = parse_ptpc_wmma_kernel_name(kernel_name)
                if params is not None:
                    params = dict(params)
                    params["kind"] = "ptpc_wmma"
            elif kernel_name.startswith("flydsl_gemm"):
                params = get_flydsl_splitk_hgemm_kernel_params(kernel_name)
                if params is not None:
                    params = dict(params)
                    params["kind"] = "hgemm"
            else:
                params = None

            if params is None:
                print(
                    f"  [WARN] Unknown FlyDSL GEMM kernel name: {kernel_name}, skipping"
                )
                continue
            if (
                params.get("kind") == "hgemm"
                and int(row.get("splitK", "0")) != params["split_k"]
            ):
                raise ValueError("FlyDSL HGEMM CSV splitK does not match kernel name")
            if params.get("kind") == "hgemm" and params.get("target_gfx") != gfx:
                raise ValueError(
                    "FlyDSL HGEMM CSV architecture does not match kernel name"
                )
            if (
                params.get("kind") == "hgemm"
                and params.get("n") is not None
                and params.get("k") is not None
                and (params.get("n"), params.get("k")) != (n, k)
            ):
                raise ValueError("FlyDSL HGEMM CSV N/K does not match kernel name")

            job = {
                **params,
                "kernel_name": kernel_name,
                "m": m,
                "n": n,
                "k": k,
                "cu_num": cu_num,
                "gfx": gfx,
                "has_bias": _parse_bool(row.get("bias")),
            }
            key = job_identity(job)
            if key in seen:
                continue
            seen.add(key)

            jobs.append(job)

    return jobs


def _torch_dtype_for_kernel(dtype_name: str):
    import torch

    mapping = {
        "bf16": torch.bfloat16,
        "f16": torch.float16,
        "fp16": torch.float16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch dtype name for GEMM AOT: {dtype_name!r}")
    return mapping[dtype_name]


def _compile_executable_to_cache(exe, *args) -> None:
    with compile_only_env():
        exe(*args)


def _ptr_view_safe(t):
    from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

    return ptr_arg(t)


def _compile_hgemm_to_cache(
    *,
    m: int,
    n: int,
    k: int,
    dtype: str,
    out_dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    stages: int,
    split_k: int,
    block_m_warps: int,
    block_n_warps: int,
    block_k_warps: int,
    async_copy: bool,
    b_to_lds: bool,
    b_preshuffle: bool,
    c_to_lds: bool,
    target_gfx: str,
    has_bias: bool = False,
):
    del out_dtype

    import torch

    dev = torch.device("cpu")
    torch_dtype = _torch_dtype_for_kernel(dtype)

    out = torch.empty((m, n), device=dev, dtype=torch_dtype)
    a = torch.empty((m, k), device=dev, dtype=torch_dtype)
    b = torch.empty((n, k), device=dev, dtype=torch_dtype)
    bias = torch.empty((n,), device=dev, dtype=torch_dtype)
    semaphore = torch.zeros(
        (SPLIT_K_SEMAPHORE_MAX_LEN,),
        device=dev,
        dtype=torch.int32,
    )
    signal = torch.zeros(
        (SPLIT_K_SEMAPHORE_MAX_LEN,),
        device=dev,
        dtype=torch.int32,
    )
    stream = fx.Stream(0)

    del target_gfx
    exe = compile_flydsl_hgemm_kernel(
        dtype,
        n,
        k,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        stages=stages,
        split_k=split_k,
        block_m_warps=block_m_warps,
        block_n_warps=block_n_warps,
        block_k_warps=block_k_warps,
        async_copy=async_copy,
        b_to_lds=b_to_lds,
        b_preshuffle=b_preshuffle,
        c_to_lds=c_to_lds,
        has_bias=has_bias,
    )
    # FlyDSL JIT does not accept None for tensor slots; pass real buffers for
    # optional bias and split-K sync tensors.
    launch_bias = unused_tensor_arg(bias if has_bias else None, b)
    _compile_executable_to_cache(
        exe,
        ptr_arg(out),
        ptr_arg(a),
        ptr_arg(b),
        ptr_arg(launch_bias),
        m,
        ptr_arg(semaphore),
        ptr_arg(signal),
        stream,
    )


def _compile_preshuffle_to_cache(
    *,
    m: int,
    n: int,
    k: int,
    in_dtype: str,
    out_dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    use_async_copy: int,
    waves_per_eu: int,
    xcd_swizzle: int = 0,
    lds_stage: int = 2,
    scheduler: str = "Default",
    **kwargs,
):
    del kwargs
    enable_scheduler = str(scheduler).lower() != "off"

    import torch

    dev = torch.device("cpu")
    out_torch_dtype = _torch_dtype_for_kernel(out_dtype)

    # FlyDSL preshuffle kernels consume raw quantized bytes for fp8/int8 paths.
    a = torch.empty((m * k,), device=dev, dtype=torch.int8)
    b = torch.empty((n * k,), device=dev, dtype=torch.int8)
    out = torch.empty((m * n,), device=dev, dtype=out_torch_dtype)
    scale_a = torch.empty((max(m, 1),), device=dev, dtype=torch.float32)
    scale_b = torch.empty((max(n, 1),), device=dev, dtype=torch.float32)
    bias = unused_tensor_arg(None, torch.empty(0, device=dev, dtype=out_torch_dtype))
    stream = fx.Stream(0)

    exe = compile_preshuffle_gemm(
        N=n,
        K=k,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        in_dtype=in_dtype,
        out_dtype="bf16" if out_torch_dtype == torch.bfloat16 else "fp16",
        use_async_copy=bool(use_async_copy),
        waves_per_eu=None if waves_per_eu <= 0 else waves_per_eu,
        enable_scheduler=enable_scheduler,
        xcd_swizzle=xcd_swizzle,
        lds_stage=lds_stage,
    )
    # The layout-API launcher uses fx.Tensor args (it builds views via
    # fx.get_iter/make_view), so pass flat torch tensors directly rather
    # than raw pointers (pointer args would fail GetIterOp type checks).
    _compile_executable_to_cache(
        exe,
        out,
        a,
        b,
        scale_a,
        scale_b,
        bias,
        m,
        n,
        stream,
    )


def _compile_8wave_to_cache(
    *,
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    waves_per_eu: int,
    xcd_swizzle: int,
    **kwargs,
):
    del kwargs

    import torch

    dev = torch.device("cpu")
    a = torch.empty((m * k,), device=dev, dtype=torch.int8)
    b = torch.empty((n * k,), device=dev, dtype=torch.int8)
    out = torch.empty((m * n,), device=dev, dtype=torch.bfloat16)
    scale_a = torch.empty((max(m, 1),), device=dev, dtype=torch.float32)
    scale_b = torch.empty((max(n, 1),), device=dev, dtype=torch.float32)

    exe = compile_8wave_gemm(
        K=k,
        block_m=block_m,
        block_n=block_n,
        waves_per_eu=waves_per_eu,
        xcd_swizzle=int(xcd_swizzle),
    )
    # NOTE: the 8-wave launcher takes (A, B, C, ...), not the preshuffle
    # launcher's (C, A, B, ...).
    _compile_executable_to_cache(exe, a, b, out, scale_a, scale_b, m, n, fx.Stream(0))


def _compile_mxfp8_128_wmma_to_cache(
    *,
    kernel_name: str,
    m: int,
    n: int,
    k: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    m_warp: int,
    n_warp: int,
    num_buffers: int,
    split_k: int,
    cluster_m: int,
    cluster_n: int,
    **kwargs,
):
    del kwargs

    import torch

    from aiter.ops.flydsl.kernels.gemm_a8w8_256x256_gfx1250 import (
        launch_gemm_a8w8_256x256,
    )
    from aiter.ops.flydsl.kernels.gemm_a8w8_gfx1250 import launch_gemm_a8w8
    from aiter.ops.flydsl.kernels.gemm_a8w8_splitk_reduce_gfx1250 import (
        compile_gemm_a8w8_splitk_reduce,
    )

    dev = torch.device("cpu")
    k_blocks = (k + 127) // 128
    xq = torch.empty((m, k), device=dev, dtype=torch.uint8)
    wq = torch.empty((n, k), device=dev, dtype=torch.uint8)
    a_scale = torch.empty((m, k_blocks), device=dev, dtype=torch.uint8)
    b_scale = torch.empty(((n + 127) // 128, k_blocks), device=dev, dtype=torch.uint8)
    out = torch.empty((m, n), device=dev, dtype=torch.bfloat16)
    stream = fx.Stream(0)

    with compile_only_env():
        launch_args = (
            _ptr_view_safe(out),
            _ptr_view_safe(xq),
            _ptr_view_safe(wq),
            _ptr_view_safe(a_scale),
            _ptr_view_safe(b_scale),
            m,
            stream,
            n,
            k,
            a_scale.numel() // a_scale.stride(0),
            xq.stride(0),
            out.stride(0),
            tile_m,
            tile_n,
            tile_k,
            m_warp,
            n_warp,
            0,
            num_buffers,
            cluster_m,
            cluster_n,
            True,
        )
        launch = (
            launch_gemm_a8w8_256x256
            if is_compute_wmma_kernel_name(kernel_name)
            else launch_gemm_a8w8
        )
        launch(*launch_args, SCALE_BLOCK_SIZE, split_k)
        if split_k > 1:
            compile_gemm_a8w8_splitk_reduce(split_k=split_k, out_dtype_str="bf16")(
                _ptr_view_safe(out),
                _ptr_view_safe(out),
                m * n,
                1,
                n,
                m * n * 2,
                stream,
            )


def _compile_ptpc_wmma_to_cache(
    *,
    m: int,
    n: int,
    k: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    m_warp: int,
    n_warp: int,
    num_buffers: int,
    split_k: int,
    cluster_m: int,
    cluster_n: int,
    **kwargs,
):
    del kwargs

    import torch

    from aiter.ops.flydsl.kernels.gemm_a8w8_gfx1250 import launch_gemm_a8w8
    from aiter.ops.flydsl.kernels.gemm_a8w8_splitk_reduce_gfx1250 import (
        compile_gemm_a8w8_splitk_reduce,
    )

    dev = torch.device("cpu")
    xq = torch.empty((m, k), device=dev, dtype=torch.uint8)
    wq = torch.empty((n, k), device=dev, dtype=torch.uint8)
    scale_a = torch.empty((max(m, 1),), device=dev, dtype=torch.float32)
    scale_b = torch.empty((max(n, 1),), device=dev, dtype=torch.float32)
    out = torch.empty((m, n), device=dev, dtype=torch.bfloat16)
    stream = fx.Stream(0)

    with compile_only_env():
        launch_gemm_a8w8(
            ptr_arg(out),
            ptr_arg(xq),
            ptr_arg(wq),
            ptr_arg(scale_a),
            ptr_arg(scale_b),
            m,
            stream,
            n,
            k,
            0,
            xq.stride(0),
            out.stride(0),
            tile_m,
            tile_n,
            tile_k,
            m_warp,
            n_warp,
            0,
            num_buffers,
            cluster_m,
            cluster_n,
            False,
            SCALE_BLOCK_SIZE,
            split_k,
        )
        if split_k > 1:
            compile_gemm_a8w8_splitk_reduce(split_k=split_k, out_dtype_str="bf16")(
                _ptr_view_safe(out),
                _ptr_view_safe(out),
                m * n,
                1,
                n,
                m * n * 2,
                stream,
            )


def job_arch(cu_num: int = 0, gfx: str = "") -> str:
    """Target arch a job would compile for -- shared by dispatch and ARCH filtering."""
    return gfx or cu_num_to_arch(cu_num, default=GEMM_AOT_ARCH_DEFAULT)


def _compile_decode_to_cache(
    *,
    m: int,
    n: int,
    k: int,
    arch: str,
    cu_num: int,
    config,
    has_bias: bool = False,
    **kwargs,
) -> None:
    del kwargs
    import torch

    device = torch.device("cpu")
    a = torch.empty((m, k), device=device, dtype=torch.bfloat16)
    b = torch.empty((n, k), device=device, dtype=torch.bfloat16)
    c = torch.empty((m, n), device=device, dtype=torch.bfloat16)
    bias = torch.empty((n,), device=device, dtype=torch.bfloat16)
    launcher = compile_gemm_decode_bf16(
        m,
        n,
        k,
        config,
        arch=arch,
        num_cus=cu_num,
        has_bias=has_bias,
    )
    _compile_executable_to_cache(
        launcher,
        a,
        b,
        c,
        unused_tensor_arg(bias if has_bias else None, b),
        fx.Stream(0),
    )


def compile_one_config(
    kernel_name: str,
    kind: str,
    m: int,
    n: int,
    k: int,
    cu_num: int = 0,
    gfx: str = "",
    **kwargs,
) -> dict:
    """Compile one GEMM kernel configuration and save it to cache."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    aot_arch = job_arch(cu_num, gfx)
    shape_str = f"{kernel_name}  M={m} N={n} K={k}"
    result = {
        "kernel_name": kernel_name,
        "kind": kind,
        "shape": shape_str,
        "compile_time": None,
        "compile_arch": aot_arch,
    }

    t0 = time.time()
    try:
        with (
            override_env("FLYDSL_GPU_ARCH", aot_arch),
            FakeTensorMode(),
        ):
            if kind == "hgemm":
                kwargs.pop("kernel_family", None)
                _compile_hgemm_to_cache(m=m, n=n, k=k, **kwargs)
            elif kind == "preshuffle":
                _compile_preshuffle_to_cache(m=m, n=n, k=k, **kwargs)
            elif kind == "8wave":
                _compile_8wave_to_cache(m=m, n=n, k=k, **kwargs)
            elif kind == "mxfp8_128_wmma":
                _compile_mxfp8_128_wmma_to_cache(
                    kernel_name=kernel_name,
                    m=m,
                    n=n,
                    k=k,
                    **kwargs,
                )
            elif kind == "ptpc_wmma":
                _compile_ptpc_wmma_to_cache(m=m, n=n, k=k, **kwargs)
            elif kind == "decode":
                _compile_decode_to_cache(
                    m=m,
                    n=n,
                    k=k,
                    arch=aot_arch,
                    cu_num=cu_num,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown GEMM AOT kind: {kind}")

        elapsed = time.time() - t0
        result["compile_time"] = elapsed
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] compile  {shape_str}  arch={aot_arch}: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="AOT pre-compile FlyDSL GEMM kernels from aiter CSV config",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=str,
        nargs="+",
        default=DEFAULT_CSVS,
        help="Path(s) to tuned CSV config file(s); defaults come from AITER_CONFIGS",
    )
    args = parser.parse_args()

    csv_paths = [os.path.abspath(p) for p in args.csv]
    for csv_path in csv_paths:
        if not os.path.isfile(csv_path):
            print(f"Error: CSV file not found: {csv_path}")
            sys.exit(1)

    cache_dir = os.path.expanduser(
        os.environ.get("FLYDSL_RUNTIME_CACHE_DIR", "~/.flydsl/cache")
    )
    arch = os.environ.get("ARCH") or os.environ.get("GPU_ARCHS")

    all_jobs = collect_aot_jobs(csv_paths, parse_csv)
    if arch:
        # GPU_ARCHS may be a ';'- or ','-separated list (e.g. "gfx942;gfx950").
        arch_set = {a.strip() for a in re.split(r"[;,]", arch) if a.strip()}
        n_before = len(all_jobs)
        all_jobs = [
            j for j in all_jobs if job_arch(j["cu_num"], j.get("gfx", "")) in arch_set
        ]
        print(f"[aiter] ARCH={arch}: {len(all_jobs)}/{n_before} jobs match")

    hgemm_jobs = [j for j in all_jobs if j["kind"] == "hgemm"]
    preshuffle_jobs = [j for j in all_jobs if j["kind"] == "preshuffle"]
    eightwave_jobs = [j for j in all_jobs if j["kind"] == "8wave"]
    mxfp8_128_wmma_jobs = [j for j in all_jobs if j["kind"] == "mxfp8_128_wmma"]
    ptpc_wmma_jobs = [j for j in all_jobs if j["kind"] == "ptpc_wmma"]
    decode_jobs = [j for j in all_jobs if j["kind"] == "decode"]

    print("=" * 72)
    print("FlyDSL GEMM AOT Pre-compilation")
    print("=" * 72)
    for csv_path in csv_paths:
        print(f"  CSV:              {csv_path}")
    print(f"  HGEMM jobs:       {len(hgemm_jobs)}")
    print(f"  Preshuffle jobs:  {len(preshuffle_jobs)}")
    print(f"  8wave jobs:       {len(eightwave_jobs)}")
    print(f"  MXFP8_128 wmma jobs: {len(mxfp8_128_wmma_jobs)}")
    print(f"  PTPC wmma jobs:   {len(ptpc_wmma_jobs)}")
    print(f"  Decode jobs:      {len(decode_jobs)}")
    print(f"  Total jobs:       {len(all_jobs)}")
    print(f"  Cache dir:        {cache_dir}")
    print(f"  Target arch:      {arch or '(all archs found in CSVs)'}")
    print("=" * 72)

    total_t0 = time.time()

    # Independent compiles that share one pool for maximum fan-out instead of
    # separate serial passes per kind.
    print(f"\n--- Compiling {len(all_jobs)} kernels ---")
    results = run_jobs_parallel(
        compile_one_config,
        hgemm_jobs
        + preshuffle_jobs
        + eightwave_jobs
        + mxfp8_128_wmma_jobs
        + ptpc_wmma_jobs
        + decode_jobs,
    )

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

    exit_code = 0
    if fail > 0:
        print("Some compilations failed. Check output above for details.")
        exit_code = 1
    else:
        print("All compilations succeeded. Cache is ready.")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
