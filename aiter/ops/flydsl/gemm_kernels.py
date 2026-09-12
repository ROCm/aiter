# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL GEMM APIs."""

from __future__ import annotations

import functools
import re

import flydsl.expr as fx
import torch
from torch import Tensor

from aiter import logger
from aiter.jit.utils.chip_info import get_gfx

from .kernels.gemm_a16w16_gfx950 import (
    SPLIT_K_SEMAPHORE_MAX_LEN,
    gemm_a16w16,
)
from .kernels.tensor_shim import _run_compiled

__all__ = [
    "SPLIT_K_SEMAPHORE_MAX_LEN",
    "flydsl_hgemm",
    "flydsl_hgemm_kernel_name",
    "flydsl_preshuffle_gemm_a8",
    "get_flydsl_hgemm_kernel_params",
]


def _get_dtypes():
    from aiter.utility import dtypes

    return dtypes


_HGEMM_KERNEL_RE = re.compile(
    r"^flydsl_hgemm_"
    r"a(?P<dtype>bf16|fp16|f16)_w(?P=dtype)"
    r"_(?P<out_dtype>bf16|fp16|f16|fp32)_"
    r"t(?P<block_m>\d+)x(?P<block_n>\d+)x(?P<block_k>\d+)"
    r"x(?P<stages>\d+)_ks(?P<split_k>\d+)_"
    r"w(?P<m_waves>\d+)x(?P<n_waves>\d+)x(?P<k_waves>\d+)_"
    r"bias(?P<has_bias>[01])_ktail(?P<has_k_tail>[01])_"
    r"gm(?P<group_m>\d+)_p(?P<policy>ft|ht|hti)_"
    r"(?P<target_gfx>gfx[0-9a-z]+)$"
)


def _normalize_dtype_name(dtype: str | torch.dtype) -> str:
    if dtype in ("f16", "fp16", torch.float16):
        return "fp16"
    if dtype in ("bf16", torch.bfloat16):
        return "bf16"
    if dtype in ("f32", "fp32", torch.float32):
        return "fp32"
    raise ValueError(f"Unsupported FlyDSL HGEMM dtype: {dtype!r}")


def flydsl_hgemm_kernel_name(
    *,
    dtype: str | torch.dtype,
    out_dtype: str | torch.dtype,
    config: dict,
    has_bias: bool,
    target_gfx: str | None = None,
) -> str:
    """Build the stable kernel name persisted in tuned GEMM CSV files."""

    dtype_name = _normalize_dtype_name(dtype)
    out_dtype_name = _normalize_dtype_name(out_dtype)
    if dtype_name not in ("bf16", "fp16"):
        raise ValueError(f"Unsupported input dtype for HGEMM: {dtype_name}")
    if out_dtype_name not in (dtype_name, "fp32"):
        raise ValueError(
            f"Unsupported output dtype {out_dtype_name} for input {dtype_name}"
        )

    policy = "ht" if config["use_half_tile_interleaved"] else "ft"
    name = (
        f"flydsl_hgemm_a{dtype_name}_w{dtype_name}_{out_dtype_name}_"
        f"t{config['block_m']}x{config['block_n']}x{config['block_k']}"
        f"x{config['stages']}_ks{config['split_k']}_"
        f"w{config['m_waves']}x{config['n_waves']}x{config['k_waves']}_"
        f"bias{int(has_bias)}_ktail0_gm{config['group_m']}_p{policy}_"
        f"{target_gfx or get_gfx()}"
    )
    return name


def get_flydsl_hgemm_kernel_params(name: str) -> dict | None:
    """Parse a tuned FlyDSL HGEMM kernel name into the current config schema."""

    match = _HGEMM_KERNEL_RE.fullmatch(name)
    if match is None or match.group("has_k_tail") != "0":
        return None

    dtype = _normalize_dtype_name(match.group("dtype"))
    out_dtype = _normalize_dtype_name(match.group("out_dtype"))
    if out_dtype not in (dtype, "fp32"):
        return None

    return {
        "block_m": int(match.group("block_m")),
        "block_n": int(match.group("block_n")),
        "block_k": int(match.group("block_k")),
        "stages": int(match.group("stages")),
        "split_k": int(match.group("split_k")),
        "m_waves": int(match.group("m_waves")),
        "n_waves": int(match.group("n_waves")),
        "k_waves": int(match.group("k_waves")),
        "group_m": int(match.group("group_m")),
        "use_half_tile_interleaved": match.group("policy") in ("ht", "hti"),
        "has_bias": match.group("has_bias") == "1",
        "dtype": dtype,
        "out_dtype": out_dtype,
        "target_gfx": match.group("target_gfx"),
    }


def flydsl_hgemm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 64,
    stages: int = 2,
    split_k: int = 1,
    m_waves: int = 2,
    n_waves: int = 2,
    k_waves: int = 1,
    group_m: int = 0,
    policy: str = "ft",
    out_dtype: torch.dtype | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Run the gfx950 A16W16 kernel with AITER's ``B[N, K]`` convention."""

    if get_gfx() != "gfx950":
        raise RuntimeError("The FlyDSL A16W16 kernel currently supports gfx950 only")
    if policy not in ("ft", "ht", "hti"):
        raise ValueError(f"Unsupported FlyDSL HGEMM policy: {policy!r}")
    launch_stream = (
        torch.cuda.current_stream(device=a.device) if stream is None else stream
    )
    if launch_stream.device != a.device:
        raise ValueError(f"`stream` must be on {a.device}, got {launch_stream.device}")

    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    user_kwargs = {
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "stages": stages,
        "split_k": split_k,
        "m_waves": m_waves,
        "n_waves": n_waves,
        "k_waves": k_waves,
        "group_m": group_m,
        "use_half_tile_interleaved": policy in ("ht", "hti"),
    }
    return gemm_a16w16(
        a,
        b.t(),
        out=out,
        bias=bias,
        user_kwargs=user_kwargs,
        stream=launch_stream,
        layout="nt",
        out_dtype=out_dtype,
    )


# ---------------------------------------------------------------------------
# FlyDSL preshuffle GEMM kernel management
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _get_compile_fn():
    """Import the preshuffle compiler on first use."""
    from .kernels.preshuffle_gemm import compile_preshuffle_gemm

    logger.info("[FlyDSL] loaded preshuffle GEMM compiler")
    return compile_preshuffle_gemm


# Fixed size rather than one buffer per shape: a shape-keyed cache grows without
# limit and can evict a buffer a captured CUDA graph still points at. The bounds
# come from k_split_candidates, which keeps tile_count under CU_NUM and
# k_split * tile_count at four per CU.
# Mirrors preshuffle_gemm.PRESHUFFLE_M_MAX; duplicated to avoid importing the
# compiler module before the preshuffle path is selected.
PRESHUFFLE_M_MAX = 65536

PRESHUFFLE_SPLIT_K_MAX_TILES = 256
PRESHUFFLE_SPLIT_K_MAX_TILE_ELEMS = 32 * 128
PRESHUFFLE_SPLIT_K_WORKSPACE_ELEMS = (
    4 * PRESHUFFLE_SPLIT_K_MAX_TILES * PRESHUFFLE_SPLIT_K_MAX_TILE_ELEMS
)


@functools.lru_cache(maxsize=128)
def _get_preshuffle_split_buffers(
    device: torch.device,
    stream: torch.cuda.Stream,
) -> tuple[Tensor, Tensor]:
    # Safe to reuse: launches on a stream are ordered and the reduction hands
    # the semaphore back zeroed.
    workspace = torch.empty(
        PRESHUFFLE_SPLIT_K_WORKSPACE_ELEMS, dtype=torch.float32, device=device
    )
    semaphore = torch.zeros(
        PRESHUFFLE_SPLIT_K_MAX_TILES, dtype=torch.int32, device=device
    )
    return workspace, semaphore


def _check_preshuffle_split_capacity(
    m: int, n: int, tile_m: int, tile_n: int, split_k: int
) -> None:
    tiles = ((m + tile_m - 1) // tile_m) * (n // tile_n)
    if tiles > PRESHUFFLE_SPLIT_K_MAX_TILES:
        raise RuntimeError(
            f"[FlyDSL] split_k needs {tiles} tile semaphores, "
            f"more than {PRESHUFFLE_SPLIT_K_MAX_TILES}"
        )
    elems = split_k * m * n
    if elems > PRESHUFFLE_SPLIT_K_WORKSPACE_ELEMS:
        raise RuntimeError(
            f"[FlyDSL] split_k needs a {elems}-element fp32 workspace, "
            f"more than {PRESHUFFLE_SPLIT_K_WORKSPACE_ELEMS}"
        )


def flydsl_preshuffle_gemm_a8(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    use_async_copy: int = 0,
    waves_per_eu: int = 0,
    xcd_swizzle: int = 0,
    lds_stage: int = 2,
    enable_scheduler: bool = True,
    split_k: int = 1,
) -> Tensor:
    """Compile and run FlyDSL preshuffle GEMM, optionally with fp32 split-K."""
    compile_fn = _get_compile_fn()
    dtypes = _get_dtypes()

    m, k = XQ.shape[0], XQ.shape[-1]
    n = WQ.shape[0]

    if m > PRESHUFFLE_M_MAX:
        raise RuntimeError(
            f"[FlyDSL] M ({m}) exceeds {PRESHUFFLE_M_MAX}; the preshuffle kernel "
            f"views A and C through a layout bounded by that many rows."
        )
    if n % tile_n != 0:
        raise RuntimeError(
            f"[FlyDSL] N ({n}) is not a multiple of tile_n ({tile_n}). "
            f"Arguments not supported! Skipping gemm!"
        )
    if split_k < 1 or k % split_k != 0:
        raise RuntimeError(
            f"[FlyDSL] K ({k}) must be divisible by split_k ({split_k})."
        )
    if (k // split_k) % tile_k != 0:
        raise RuntimeError(
            f"[FlyDSL] K/split_k ({k // split_k}) is not a multiple of "
            f"tile_k ({tile_k}). "
            f"Arguments not supported! Skipping gemm!"
        )

    if XQ.dtype == dtypes.fp8:
        in_dtype = "fp8"
    elif XQ.dtype == torch.int8:
        in_dtype = "int8"
    else:
        raise ValueError(f"[FlyDSL] unsupported input dtype {XQ.dtype}")

    wpe = None if waves_per_eu <= 0 else waves_per_eu

    if Out.dtype == torch.bfloat16:
        out_dtype = "bf16"
    elif Out.dtype == torch.float16:
        out_dtype = "fp16"
    else:
        raise ValueError(
            f"[FlyDSL] unsupported output dtype {Out.dtype}; "
            "expected torch.bfloat16 or torch.float16"
        )

    exe = compile_fn(
        N=n,
        K=k,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        use_async_copy=bool(use_async_copy),
        waves_per_eu=wpe,
        enable_scheduler=bool(enable_scheduler),
        xcd_swizzle=int(xcd_swizzle),
        lds_stage=int(lds_stage),
        split_k=int(split_k),
    )

    def _as_i8(t):
        return t.view(torch.int8) if "float8" in str(t.dtype) else t

    out_contig = Out.contiguous()
    # FlyDSL's preshuffle kernel requires an arg_bias slot (used only when
    # epilogue != "none"). Pass an empty tensor as a placeholder for the
    # default epilogue="none" path.
    dummy_bias = torch.empty(0, dtype=Out.dtype, device=Out.device)
    if split_k > 1:
        _check_preshuffle_split_capacity(m, n, tile_m, tile_n, split_k)
        workspace, semaphore = _get_preshuffle_split_buffers(
            Out.device, torch.cuda.current_stream(device=Out.device)
        )
    else:
        workspace = out_contig
        # dtype is part of the executable's cache signature, so this must match
        # what the AOT pre-compile passes or every non-split-K kernel misses it.
        semaphore = torch.empty(0, dtype=torch.int32, device=Out.device)
    # The layout-API launcher takes fx.Tensor args, so pass flat tensors
    # directly rather than raw pointers.
    _run_compiled(
        exe,
        workspace.view(-1),
        out_contig.view(-1),
        semaphore,
        _as_i8(XQ.contiguous()).view(-1),
        _as_i8(WQ.contiguous()).view(-1),
        x_scale.contiguous().view(-1),
        w_scale.contiguous().view(-1),
        dummy_bias,
        m,
        n,
        fx.Stream(torch.cuda.current_stream()),
    )
    if out_contig is not Out:
        Out.copy_(out_contig)

    return Out
