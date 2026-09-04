#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for FlyDSL GEMM kernels from aiter tuned CSV configs.

Reads tuned GEMM CSV config files, extracts all unique FlyDSL kernel entries,
and pre-compiles them into the FlyDSL cache. The default CSV set is resolved
through ``AITER_CONFIGS`` so model-specific tuned CSVs can be merged the same
way as runtime JIT config lookup.

Supported kernel families:
  - ``flydsl_hgemm_*``                        gfx950 A16W16 GEMM kernels
  - ``flydsl_bpreshuflle_*``                  a8w8 preshuffle GEMM kernels
  - ``flydsl_bpreshuffle_8w_*``               gfx950 8-wave a8w8 ptpc GEMM kernels
  - ``flydsl_bpreshuffle_wmma_*``             gfx1250 a8w8 ptpc GEMM kernels
  - ``flydsl_mxfp8_128_bpreshuffle_wmma_*``   gfx1250 mxfp8_128 GEMM kernels
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
from contextlib import nullcontext

import flydsl.compiler as flyc
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
    get_flydsl_hgemm_kernel_params,
)
from aiter.ops.flydsl.kernels.common import run_cached
from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 import (
    GEMM_A16W16_DTYPE_BF16,
    GEMM_A16W16_DTYPE_FP16,
    GEMM_A16W16_DTYPE_FP32,
    _dynamic_tensor_arg,
    gemm_a16w16_gfx950,
    make_gemm_a16w16_param_and_validate,
)
from aiter.ops.flydsl.kernels.preshuffle_gemm import compile_preshuffle_gemm
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
    r"(?!ks\d+$)(?P<scheduler>[A-Za-z][A-Za-z0-9]*)"
    # Trailing _ksN, emitted only for k_split > 1, so pre-split-K names still
    # match. Without it they fail fullmatch and drop out of the AOT build.
    r"(?:_ks(?P<k_split>\d+))?$"
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
        "k_split": int(m.group("k_split")) if m.group("k_split") else 1,
    }


def parse_csv(csv_path: str):
    """Parse a GEMM tuned CSV and return a list of unique FlyDSL compile jobs."""
    jobs = []
    seen = set()

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kernel_name = row.get("kernelName", "").strip()
            libtype = row.get("libtype", "").strip()
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
            elif kernel_name.startswith("flydsl_hgemm"):
                params = get_flydsl_hgemm_kernel_params(kernel_name)
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

            job = {
                "kernel_name": kernel_name,
                "m": m,
                "n": n,
                "k": k,
                "cu_num": cu_num,
                "gfx": gfx,
                "has_bias": _parse_bool(row.get("bias")),
                **params,
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
        "f32": torch.float32,
        "fp32": torch.float32,
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
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    split_k: int,
    m_waves: int,
    n_waves: int,
    k_waves: int,
    group_m: int,
    use_half_tile_interleaved: bool,
    has_bias: bool,
    target_gfx: str,
    **kwargs,
):
    del kwargs

    import torch

    if target_gfx != "gfx950":
        raise ValueError(
            f"The FlyDSL A16W16 kernel only supports gfx950, got {target_gfx}"
        )

    torch_dtype = _torch_dtype_for_kernel(dtype)
    torch_out_dtype = _torch_dtype_for_kernel(out_dtype)
    if torch_out_dtype not in (torch_dtype, torch.float32):
        raise ValueError(
            f"Unsupported output dtype {out_dtype!r} for input dtype {dtype!r}"
        )
    in_dtype_id = (
        GEMM_A16W16_DTYPE_FP16
        if torch_dtype == torch.float16
        else GEMM_A16W16_DTYPE_BF16
    )
    out_dtype_id = (
        GEMM_A16W16_DTYPE_FP32 if torch_out_dtype == torch.float32 else in_dtype_id
    )
    config = {
        "in_dtype_id": in_dtype_id,
        "out_dtype_id": out_dtype_id,
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "stages": stages,
        "split_k": split_k,
        "m_waves": m_waves,
        "n_waves": n_waves,
        "k_waves": k_waves,
        "group_m": group_m,
        "use_half_tile_interleaved": use_half_tile_interleaved,
        "a_is_transposed": False,
        "b_is_transposed": True,
        "has_bias": has_bias,
    }
    param = make_gemm_a16w16_param_and_validate(m, n, k, config)
    if param is None:
        raise ValueError(
            f"Invalid FlyDSL A16W16 config for M={m}, N={n}, K={k}: {config}"
        )

    # Layout-dynamic arguments make this compile independent of M/N/K. Small
    # real CPU tensors avoid materializing model-sized buffers during AOT.
    with compile_only_env():
        dev = torch.device("cpu")
        representative_extent = 8
        a = torch.empty((1, representative_extent), device=dev, dtype=torch_dtype)
        b = torch.empty(
            (representative_extent, representative_extent),
            device=dev,
            dtype=torch_dtype,
        ).t()
        out = torch.empty((1, representative_extent), device=dev, dtype=torch_out_dtype)
        bias = torch.empty((representative_extent,), device=dev, dtype=torch_dtype)
        semaphore = torch.zeros(
            (SPLIT_K_SEMAPHORE_MAX_LEN,), device=dev, dtype=torch.int32
        )
        signal = torch.zeros(
            (SPLIT_K_SEMAPHORE_MAX_LEN,), device=dev, dtype=torch.int32
        )
        stream = fx.Stream(0)
        a_arg = _dynamic_tensor_arg(a, 1)
        b_arg = _dynamic_tensor_arg(b, 0)
        out_arg = _dynamic_tensor_arg(out, 1)
        bias_arg = a_arg if not has_bias else _dynamic_tensor_arg(bias, 0)
        dispatch_args = (
            out_arg,
            a_arg,
            b_arg,
            bias_arg,
            semaphore,
            signal,
            split_k,
            param,
            stream,
        )
        run_cached(
            gemm_a16w16_gfx950,
            *dispatch_args,
            constexpr_param=param,
            compiler=flyc.compile,
            dispatch_args=dispatch_args,
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
    k_split: int = 1,
    **kwargs,
):
    del kwargs
    enable_scheduler = str(scheduler).lower() != "off"
    k_split = int(k_split)

    import torch

    dev = torch.device("cpu")
    out_torch_dtype = _torch_dtype_for_kernel(out_dtype)

    # FlyDSL preshuffle kernels consume raw quantized bytes for fp8/int8 paths.
    a = torch.empty((m * k,), device=dev, dtype=torch.int8)
    b = torch.empty((n * k,), device=dev, dtype=torch.int8)
    from aiter.ops.flydsl.gemm_kernels import (
        PRESHUFFLE_SPLIT_K_MAX_TILES,
        PRESHUFFLE_SPLIT_K_WORKSPACE_ELEMS,
    )

    # Sized to the same bounds the runtime uses, so the signatures match.
    out = torch.empty((m * n,), device=dev, dtype=out_torch_dtype)
    workspace = (
        torch.empty(PRESHUFFLE_SPLIT_K_WORKSPACE_ELEMS, device=dev, dtype=torch.float32)
        if k_split > 1
        else out
    )
    semaphore = torch.zeros(
        PRESHUFFLE_SPLIT_K_MAX_TILES if k_split > 1 else 0,
        device=dev,
        dtype=torch.int32,
    )
    scale_a = torch.empty((max(m, 1),), device=dev, dtype=torch.float32)
    scale_b = torch.empty((max(n, 1),), device=dev, dtype=torch.float32)
    bias = torch.empty(0, device=dev, dtype=out_torch_dtype)
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
        split_k=k_split,
    )
    # The layout-API launcher uses fx.Tensor args (it builds views via
    # fx.get_iter/make_view), so pass flat torch tensors directly rather
    # than raw pointers (pointer args would fail GetIterOp type checks).
    _compile_executable_to_cache(
        exe,
        workspace,
        out,
        semaphore,
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
            _ptr_view_safe(out),
            _ptr_view_safe(xq),
            _ptr_view_safe(wq),
            _ptr_view_safe(scale_a),
            _ptr_view_safe(scale_b),
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
        tensor_context = nullcontext() if kind == "hgemm" else FakeTensorMode()
        with (
            override_env("FLYDSL_GPU_ARCH", aot_arch),
            tensor_context,
        ):
            if kind == "hgemm":
                hgemm_kwargs = dict(kwargs)
                hgemm_kwargs["target_gfx"] = aot_arch
                _compile_hgemm_to_cache(m=m, n=n, k=k, **hgemm_kwargs)
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
        + ptpc_wmma_jobs,
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
