# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL HGEMM APIs."""

from __future__ import annotations

import functools
import re
from itertools import product

import flydsl.expr as fx
import torch
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SMEM_CAPACITY_MAP
from torch import Tensor

from aiter import logger
from aiter.jit.utils.chip_info import get_gfx, get_gfx_runtime
from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

from .kernels.gemm_a16w16_gfx950 import (
    GEMM_A16W16_DTYPE_BF16,
    GEMM_A16W16_DTYPE_FP16,
    GEMM_A16W16_DTYPE_FP32,
    make_gemm_a16w16_param_and_validate,
)
from .kernels.gemm_a16w16_gfx950 import (
    gemm_a16w16 as gemm_a16w16_gfx950,
)
from .kernels.hgemm_dispatch import compile_flydsl_hgemm_kernel

# from .kernels.small_m_hgemm import iter_small_m_registry_configs
from .kernels.tensor_shim import _run_compiled
from .utils import get_shared_memory_per_block, is_flydsl_available

__all__ = [
    "flydsl_gfx950_hgemm",
    "flydsl_gfx950_kernel_name",
    "flydsl_hgemm",
    "get_flydsl_hgemm_kernel_params",
    "get_flydsl_hgemm_kernels",
]


def _get_dtypes():
    from aiter.utility import dtypes

    return dtypes


SPLIT_K_SEMAPHORE_MAX_LEN = 256
FIXED_STAGE = 2
FIXED_C_TO_LDS = False
KERNEL_ASYNC_COPY = get_rocm_arch() != "gfx942"
KERNEL_FAMILY_HGEMM = "hgemm"
KERNEL_FAMILY_SMALL_M = "small_m"
KERNEL_FAMILY_GFX950 = "gfx950_a16w16"
_HGEMM_KERNEL_RE = re.compile(
    r"^flydsl_gemm(?P<stages>\d+)_"
    r"a(?P<a_dtype>[a-z0-9]+)_w(?P<w_dtype>[a-z0-9]+)_(?P<out_dtype>[a-z0-9]+)_"
    r"t(?P<tile_m>\d+)x(?P<tile_n>\d+)x(?P<tile_k>\d+)_"
    r"split_k(?P<split_k>\d+)_"
    r"block_m_warp(?P<block_m_warps>\d+)_"
    r"block_n_warp(?P<block_n_warps>\d+)_"
    r"block_k_warp(?P<block_k_warps>\d+)_"
    r"async_copy(?P<async_copy>True|False)_"
    r"b_to_lds(?P<b_to_lds>True|False)_"
    r"b_preshuffle(?P<b_preshuffle>True|False)_"
    r"c_to_lds(?P<c_to_lds>True|False)"
    r"(?P<small_m_suffix>"
    r"(?:_small_m)"
    r"(?:_nr(?P<n_tile_repeat>\d+))?"
    r"(?:_pn(?P<persistent_n_tiles>\d+))?"
    r"(?:_wpe(?P<waves_per_eu>\d+))?"
    r"(?:_ur(?P<b_to_lds_unroll>\d+))?"
    r")?"
    r"_(?P<target_gfx>gfx[0-9a-z]+)$"
)
_GFX950_HGEMM_KERNEL_RE = re.compile(
    r"^flydsl_hgemm_"
    r"a(?P<a_dtype>bf16|f16|fp16)_"
    r"w(?P<w_dtype>bf16|f16|fp16)_"
    r"(?P<out_dtype>bf16|f16|fp16|f32|fp32)_"
    r"t(?P<tile_m>\d+)x(?P<tile_n>\d+)x(?P<tile_k>\d+)x(?P<stages>\d+)_"
    r"ks(?P<split_k>\d+)_"
    r"w(?P<m_waves>\d+)x(?P<n_waves>\d+)x(?P<k_waves>\d+)_"
    r"bias(?P<has_bias>[01])_"
    r"gm(?P<group_m>\d+)_"
    r"p(?P<policy>ft|ht|hti)_"
    r"(?P<target_gfx>gfx[0-9a-z]+)$"
)

SplitKStreamKey = tuple[int, int]
SPLIT_K_GLOBAL_SEMAPHORE: dict[SplitKStreamKey, torch.Tensor] = {}
SPLIT_K_GLOBAL_SIGNAL: dict[SplitKStreamKey, torch.Tensor] = {}


# Keep the generic auto-generated catalog aligned with the upstream FlyDSL
# reference tuning space. The wider local one-off search space introduced
# gfx950-faulting candidates (for example tile_k=160 and tile_n=160/192),
# and higher split-K values are now capped at 8 for better accuracy.
HGEMM_TILE_N_OPTIONS = (64, 128, 256)
HGEMM_TILE_K_OPTIONS = (64, 128, 256)
HGEMM_TILE_M_OPTIONS = (16, 32, 48, 64, 80, 96, 128, 256)
HGEMM_STAGE_OPTIONS = tuple([i for i in range(2, 9)])
HGEMM_BASE_SPLIT_K_OPTIONS = tuple(range(1, 14))
HGEMM_MAX_SPLIT_K = 13
HGEMM_WARP_SHAPE_OPTIONS = [
    (wm, wn, wk) for wm, wn, wk in product([1, 2, 4], repeat=3) if wm * wn * wk <= 16
]
KERNEL_CONFIG_VARIANTS = [
    {
        "block_m_warps": wm,
        "block_n_warps": wn,
        "block_k_warps": wk,
        "b_to_lds": True,
    }
    for wm, wn, wk in HGEMM_WARP_SHAPE_OPTIONS
]

_SPLITK_HGEMM_KERNELS: dict[str, dict] = {}


def _normalize_supported_kernel_metadata(
    *,
    async_copy: bool,
    c_to_lds: bool,
) -> tuple[int, bool, bool]:
    # Latest `hgemm.py` fixes these choices internally instead of exposing
    # multiple codegen variants to the wrapper layer.
    if async_copy != KERNEL_ASYNC_COPY:
        raise ValueError(
            "Current kernel fixes async_copy from the active GPU architecture; "
            f"got async_copy={async_copy}, expected {KERNEL_ASYNC_COPY}"
        )
    if c_to_lds != FIXED_C_TO_LDS:
        raise ValueError(
            f"Current kernel only supports c_to_lds={FIXED_C_TO_LDS}; "
            f"got c_to_lds={c_to_lds}"
        )
    return KERNEL_ASYNC_COPY, FIXED_C_TO_LDS


def _normalize_gfx950_dtype_name(dtype: str, *, allow_fp32: bool = False) -> str:
    normalized = dtype.lower()
    aliases = {
        "fp16": "f16",
        "float16": "f16",
        "bfloat16": "bf16",
        "fp32": "f32",
        "float32": "f32",
    }
    normalized = aliases.get(normalized, normalized)
    supported = {"bf16", "f16"}
    if allow_fp32:
        supported.add("f32")
    if normalized not in supported:
        expected = "bf16/f16/f32" if allow_fp32 else "bf16/f16"
        raise ValueError(f"Unsupported dtype {dtype!r}; expected {expected}")
    return normalized


def _get_runtime_gfx(device: torch.device | None = None) -> str:
    """Return the live architecture, preferring the selected torch device."""

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(
            torch.cuda.current_device() if device is None else device
        )
        arch = str(getattr(props, "gcnArchName", "")).split(":")[0]
        if not arch:
            raise RuntimeError(
                "Unable to determine the active ROCm device architecture"
            )
        return arch
    return get_gfx_runtime()


def flydsl_gfx950_kernel_name(
    *,
    dtype: str,
    out_dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    stages: int,
    split_k: int,
    m_waves: int,
    n_waves: int,
    k_waves: int,
    has_bias: bool,
    group_m: int,
    use_half_tile_interleaved: bool,
    target_gfx: str = "gfx950",
) -> str:
    """Return the stable tuned-config name for the layout-based gfx950 kernel."""

    dtype = _normalize_gfx950_dtype_name(dtype)
    out_dtype = _normalize_gfx950_dtype_name(out_dtype, allow_fp32=True)
    policy = "hti" if use_half_tile_interleaved else "ft"
    return (
        f"flydsl_hgemm_a{dtype}_w{dtype}_{out_dtype}_"
        f"t{tile_m}x{tile_n}x{tile_k}x{stages}_ks{split_k}_"
        f"w{m_waves}x{n_waves}x{k_waves}_bias{int(has_bias)}_"
        f"gm{group_m}_p{policy}_{target_gfx}"
    )


def _parse_gfx950_hgemm_kernel_params(name: str) -> dict | None:
    match = _GFX950_HGEMM_KERNEL_RE.fullmatch(name)
    if match is None or match.group("a_dtype") != match.group("w_dtype"):
        return None

    dtype = _normalize_gfx950_dtype_name(match.group("a_dtype"))
    out_dtype = _normalize_gfx950_dtype_name(match.group("out_dtype"), allow_fp32=True)
    policy = match.group("policy")
    return {
        "kernel_family": KERNEL_FAMILY_GFX950,
        "dtype": dtype,
        "out_dtype": out_dtype,
        "tile_m": int(match.group("tile_m")),
        "tile_n": int(match.group("tile_n")),
        "tile_k": int(match.group("tile_k")),
        "stages": int(match.group("stages")),
        "split_k": int(match.group("split_k")),
        "block_m_warps": int(match.group("m_waves")),
        "block_n_warps": int(match.group("n_waves")),
        "block_k_warps": int(match.group("k_waves")),
        "has_bias": match.group("has_bias") == "1",
        "group_m": int(match.group("group_m")),
        "policy": "hti" if policy in {"ht", "hti"} else "ft",
        "use_half_tile_interleaved": policy in {"ht", "hti"},
        "target_gfx": match.group("target_gfx"),
        # Generic metadata consumers use these fields when filtering catalogs.
        "async_copy": True,
        "b_to_lds": True,
        "b_preshuffle": False,
        "c_to_lds": False,
    }


def get_flydsl_hgemm_kernel_params(name: str) -> dict | None:
    """Look up either the new gfx950 family or a legacy generic HGEMM name."""

    config = _parse_gfx950_hgemm_kernel_params(name)
    if config is not None:
        return config
    return get_flydsl_splitk_hgemm_kernel_params(name)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _gfx950_split_k_padded(k: int, tile_k: int, split_k: int) -> int:
    working_k = _ceil_div(k, split_k)
    padded_k = 0
    for split_idx in range(split_k):
        remaining_k = max(k - split_idx * working_k, 0)
        part_k = min(working_k, remaining_k)
        if part_k > 0:
            padded_k += _ceil_div(part_k, tile_k) * tile_k
    return padded_k


def _gfx950_config_iou(
    m: int,
    n: int,
    k: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    split_k: int,
) -> float:
    padded_m = _ceil_div(m, tile_m) * tile_m
    padded_n = _ceil_div(n, tile_n) * tile_n
    padded_k = _gfx950_split_k_padded(k, tile_k, split_k)
    return (m * n * k) / (padded_m * padded_n * padded_k)


def get_flydsl_hgemm_kernels(
    dtype: str,
    out_dtype: str,
    has_bias: bool,
    *,
    m: int,
    n: int,
    k: int,
) -> dict[str, dict]:
    """Enumerate validated gfx950 mixed-policy HGEMM tuning candidates."""

    target_gfx = _get_runtime_gfx()
    if target_gfx != "gfx950":
        return {}

    dtype = _normalize_gfx950_dtype_name(dtype)
    out_dtype = _normalize_gfx950_dtype_name(out_dtype, allow_fp32=True)
    if out_dtype not in {dtype, "f32"}:
        return {}

    in_dtype_id = GEMM_A16W16_DTYPE_FP16 if dtype == "f16" else GEMM_A16W16_DTYPE_BF16
    out_dtype_id = GEMM_A16W16_DTYPE_FP32 if out_dtype == "f32" else in_dtype_id
    split_k_options = (1,) + tuple(
        split_k for split_k in range(2, 10) if k % split_k == 0
    )
    tile_m_options = (16, 32, 48, 64, 80, 96, 128, 256)
    tile_n_options = (16, 32, 64, 80, 96, 128, 256)
    tile_k_options = (64, 128, 256)
    stage_options = tuple(range(2, 10))
    wave_options = (1, 2, 4)
    k_wave_options = (1, 2)
    group_m_options = (0, 4)
    policy_options = (False, True)

    base_ious = {
        (tile_m, tile_n, tile_k, split_k): _gfx950_config_iou(
            m, n, k, tile_m, tile_n, tile_k, split_k
        )
        for tile_m, tile_n, tile_k, split_k in product(
            tile_m_options,
            tile_n_options,
            tile_k_options,
            split_k_options,
        )
    }
    keep_ratio = 0.75 if m <= 32 else 0.85 if m <= 128 else 0.95
    iou_threshold = max(base_ious.values()) * keep_ratio
    is_large_gemm = m >= 4096 and n >= 4096 and k >= 4096

    kernels: dict[str, dict] = {}
    for (
        tile_m,
        tile_n,
        tile_k,
        stages,
        split_k,
        m_waves,
        n_waves,
        k_waves,
        group_m,
        use_half_tile_interleaved,
    ) in product(
        tile_m_options,
        tile_n_options,
        tile_k_options,
        stage_options,
        split_k_options,
        wave_options,
        wave_options,
        k_wave_options,
        group_m_options,
        policy_options,
    ):
        if base_ious[(tile_m, tile_n, tile_k, split_k)] < iou_threshold:
            continue
        if is_large_gemm:
            if not (
                use_half_tile_interleaved
                and tile_m == 256
                and tile_n == 256
                and tile_k == 64
                and stages == 2
                and split_k == 1
                and m_waves == 2
                and n_waves == 4
                and k_waves == 1
            ):
                continue
        elif not use_half_tile_interleaved:
            mma_m_iters = tile_m // m_waves // 16
            mma_n_iters = tile_n // n_waves // 16
            if mma_m_iters > 4 or mma_n_iters > 4:
                continue

        validation_config = {
            "in_dtype_id": in_dtype_id,
            "out_dtype_id": out_dtype_id,
            "block_m": tile_m,
            "block_n": tile_n,
            "block_k": tile_k,
            "stages": stages,
            "split_k": split_k,
            "m_waves": m_waves,
            "n_waves": n_waves,
            "k_waves": k_waves,
            "group_m": group_m,
            "use_half_tile_interleaved": use_half_tile_interleaved,
            "has_bias": bool(has_bias),
        }
        param = make_gemm_a16w16_param_and_validate(m, n, k, validation_config)
        if param is None:
            continue

        config = {
            "kernel_family": KERNEL_FAMILY_GFX950,
            "dtype": dtype,
            "out_dtype": out_dtype,
            "tile_m": tile_m,
            "tile_n": tile_n,
            "tile_k": tile_k,
            "stages": stages,
            "split_k": split_k,
            "block_m_warps": m_waves,
            "block_n_warps": n_waves,
            "block_k_warps": k_waves,
            "has_bias": bool(has_bias),
            "group_m": group_m,
            "policy": "hti" if use_half_tile_interleaved else "ft",
            "use_half_tile_interleaved": use_half_tile_interleaved,
            "target_gfx": target_gfx,
            "async_copy": True,
            "b_to_lds": True,
            "b_preshuffle": False,
            "c_to_lds": False,
        }
        name = flydsl_gfx950_kernel_name(
            dtype=dtype,
            out_dtype=out_dtype,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            stages=stages,
            split_k=split_k,
            m_waves=m_waves,
            n_waves=n_waves,
            k_waves=k_waves,
            has_bias=has_bias,
            group_m=group_m,
            use_half_tile_interleaved=use_half_tile_interleaved,
            target_gfx=target_gfx,
        )
        kernels[name] = config
    return kernels


def flydsl_kernel_name(
    stages: int,
    dtype: str,
    out_dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    split_k: int,
    block_m_warp: int,
    block_n_warp: int,
    block_k_warp: int,
    async_copy: bool,
    b_to_lds: bool,
    b_preshuffle: bool = False,
    c_to_lds: bool = False,
    kernel_family: str = KERNEL_FAMILY_HGEMM,
    n_tile_repeat: int = 1,
    persistent_n_tiles: int = 1,
    waves_per_eu: int = 0,
    b_to_lds_unroll: int = 0,
) -> str:
    async_copy, c_to_lds = _normalize_supported_kernel_metadata(
        async_copy=async_copy,
        c_to_lds=c_to_lds,
    )
    if b_preshuffle and b_to_lds:
        raise ValueError(
            "Current kernel requires b_to_lds=False when b_preshuffle=True"
        )
    if kernel_family == KERNEL_FAMILY_HGEMM and b_preshuffle:
        raise ValueError("Current generic kernel only supports `b_preshuffle=False`")
    if kernel_family == KERNEL_FAMILY_SMALL_M and b_preshuffle:
        raise ValueError("small-M kernel only supports `b_preshuffle=False`")
    name = (
        f"flydsl_gemm{stages}_a{dtype}_w{dtype}_{out_dtype}_t{tile_m}x{tile_n}x{tile_k}"
    )
    name += f"_split_k{split_k}_block_m_warp{block_m_warp}_block_n_warp{block_n_warp}_block_k_warp{block_k_warp}"
    name += (
        f"_async_copy{async_copy}_b_to_lds{b_to_lds}_b_preshuffle{b_preshuffle}"
        f"_c_to_lds{c_to_lds}"
    )
    if kernel_family == KERNEL_FAMILY_SMALL_M:
        name += "_small_m"
        if n_tile_repeat > 1:
            name += f"_nr{n_tile_repeat}"
        if persistent_n_tiles > 1:
            name += f"_pn{persistent_n_tiles}"
        if waves_per_eu > 0:
            name += f"_wpe{waves_per_eu}"
        if b_to_lds_unroll > 0:
            name += f"_ur{b_to_lds_unroll}"
    elif kernel_family != KERNEL_FAMILY_HGEMM:
        raise ValueError(
            f"Unsupported kernel_family={kernel_family!r}; expected "
            f"{KERNEL_FAMILY_HGEMM!r} or {KERNEL_FAMILY_SMALL_M!r}"
        )
    name += f"_{get_gfx()}"
    return name


def _stream_cache_key(stream: torch.cuda.Stream) -> SplitKStreamKey:
    device_index = stream.device.index
    if device_index is None:
        raise ValueError(f"Unable to determine device index for stream {stream!r}")
    return (device_index, int(stream.cuda_stream))


def _normalize_launch_stream(
    device: torch.device,
    stream: torch.cuda.Stream | None,
) -> torch.cuda.Stream:
    launch_stream = (
        torch.cuda.current_stream(device=device) if stream is None else stream
    )
    if launch_stream.device != device:
        raise ValueError(f"`stream` must be on {device}, got {launch_stream.device}")
    return launch_stream


def _to_kernel_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "f16"
    if dtype == torch.bfloat16:
        return "bf16"
    raise ValueError(f"Only fp16/bf16 are supported, got {dtype!r}")


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _hgemm_tile_m_options(m: int | None) -> tuple[int, ...]:
    if m is None:
        return HGEMM_TILE_M_OPTIONS
    max_tile_m = max(96, _align_up(max(1, m) * 2, 16))
    return tuple(tile_m for tile_m in HGEMM_TILE_M_OPTIONS if tile_m <= max_tile_m)


def _hgemm_split_k_options(k: int | None, tile_k: int) -> tuple[int, ...]:
    if k is None:
        return HGEMM_BASE_SPLIT_K_OPTIONS
    return tuple(
        split_k
        for split_k in HGEMM_BASE_SPLIT_K_OPTIONS
        if split_k <= HGEMM_MAX_SPLIT_K
        and k % split_k == 0
        and (k // split_k) % tile_k == 0
    )


def _estimate_hgemm_lds_bytes(
    *,
    dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    stages: int,
    block_k_warps: int,
    b_to_lds: bool,
) -> int:
    if dtype not in {"f16", "bf16"}:
        raise ValueError(f"`dtype` must be 'f16' or 'bf16', got {dtype!r}")

    dtype_bytes = 2
    pipeline_lds_bytes = stages * tile_m * tile_k * dtype_bytes
    if b_to_lds:
        pipeline_lds_bytes += stages * tile_n * tile_k * dtype_bytes
    c_lds_bytes = block_k_warps * tile_m * tile_n * dtype_bytes
    return max(pipeline_lds_bytes, c_lds_bytes)


def _validate_hgemm_inputs(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor | None,
    bias: torch.Tensor | None,
) -> tuple[int, int, int]:
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(
            f"`flydsl_hgemm` expects 2D inputs, got a.dim={a.dim()} b.dim={b.dim()}"
        )
    if a.device.type != "cuda" or b.device.type != "cuda":
        raise ValueError("`flydsl_hgemm` only supports CUDA/ROCm tensors")
    if a.device != b.device:
        raise ValueError(
            f"`a` and `b` must be on the same device, got {a.device=} {b.device=}"
        )
    if a.dtype != b.dtype:
        raise ValueError(
            f"`a` and `b` must have the same dtype, got {a.dtype=} {b.dtype=}"
        )

    m, k = a.shape
    n, bk = b.shape
    if k != bk:
        raise ValueError(
            f"Incompatible GEMM shapes: a={tuple(a.shape)} b={tuple(b.shape)}"
        )

    if out is not None:
        if out.shape != (m, n):
            raise ValueError(f"`out` must have shape {(m, n)}, got {tuple(out.shape)}")
        if out.dtype != a.dtype:
            raise ValueError(
                f"`out` dtype must match input dtype, got {out.dtype=} {a.dtype=}"
            )
        if out.device != a.device:
            raise ValueError(f"`out` must be on {a.device}, got {out.device}")
        if not out.is_contiguous():
            raise ValueError("`out` must be contiguous")

    if bias is not None:
        if bias.dim() != 1:
            raise ValueError(f"`bias` must be 1D, got bias.dim={bias.dim()}")
        if bias.shape != (n,):
            raise ValueError(f"`bias` must have shape {(n,)}, got {tuple(bias.shape)}")
        if bias.dtype != a.dtype:
            raise ValueError(
                f"`bias` dtype must match input dtype, got {bias.dtype=} {a.dtype=}"
            )
        if bias.device != a.device:
            raise ValueError(f"`bias` must be on {a.device}, got {bias.device}")

    return m, n, k


def selection_filter(m, n, k, kwargs):
    TILE_M = kwargs["TILE_M"]
    TILE_N = kwargs["TILE_N"]
    TILE_K = kwargs["TILE_K"]
    STAGES = kwargs["STAGES"]
    SPLIT_K = kwargs["SPLIT_K"]
    BLOCK_M_WARPS = kwargs["BLOCK_M_WARPS"]
    BLOCK_N_WARPS = kwargs["BLOCK_N_WARPS"]
    BLOCK_K_WARPS = kwargs["BLOCK_K_WARPS"]
    B_TO_LDS = kwargs.get("B_TO_LDS", True)
    GPU_ARCH = get_rocm_arch()
    DTYPE_BYTES = 2

    def get_stage_smem_use(stages_):
        SMEM_USE = stages_ * TILE_M * TILE_K * DTYPE_BYTES
        if B_TO_LDS:
            SMEM_USE += stages_ * TILE_N * TILE_K * DTYPE_BYTES
        SMEM_USE = max(SMEM_USE, BLOCK_K_WARPS * TILE_M * TILE_N * DTYPE_BYTES)
        return SMEM_USE

    smem_use_s0 = get_stage_smem_use(STAGES)
    # smem_use_s1 = get_stage_smem_use(STAGES + 3)
    smem_cap = SMEM_CAPACITY_MAP[GPU_ARCH]
    if not (smem_use_s0 <= smem_cap):
        return False
    return not (
        m >= 4096
        and n >= 4096
        and k >= 4096
        and not (
            TILE_M == 256
            and TILE_N == 256
            and TILE_K == 64
            and STAGES == 2
            and SPLIT_K == 1
            and BLOCK_M_WARPS == 2
            and BLOCK_N_WARPS == 4
            and BLOCK_K_WARPS == 1
        )
    )


def _validate_hgemm_tiling(
    m: int,
    n: int,
    k: int,
    *,
    dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    pack_n: int,
    split_k: int,
    stages: int,
    block_m_warps: int,
    block_n_warps: int,
    block_k_warps: int,
    b_to_lds: bool,
) -> None:
    config = {
        "TILE_M": tile_m,
        "TILE_N": tile_n,
        "TILE_K": tile_k,
        "STAGES": stages,
        "SPLIT_K": split_k,
        "BLOCK_M_WARPS": block_m_warps,
        "BLOCK_N_WARPS": block_n_warps,
        "BLOCK_K_WARPS": block_k_warps,
        "B_TO_LDS": b_to_lds,
    }
    if not selection_filter(m, n, k, config):
        raise ValueError(
            f"Invalid tiling configuration for m={m} n={n} k={k}: {config}"
        )

    if tile_m < 1 or tile_n < 1 or tile_k < 1:
        raise ValueError(
            f"Tile sizes must be positive, got tile_m={tile_m}, tile_n={tile_n}, tile_k={tile_k}"
        )
    if block_m_warps < 1 or block_n_warps < 1 or block_k_warps < 1:
        raise ValueError(
            "Warp tiling must be positive, got "
            f"block_m_warps={block_m_warps}, block_n_warps={block_n_warps}, block_k_warps={block_k_warps}"
        )
    if tile_k < 32:
        raise ValueError(
            f"Invalid tile_k={tile_k}; latest kernel requires tile_k >= 32"
        )
    if tile_k % 32 != 0:
        raise ValueError(
            f"Invalid tile_k={tile_k}; latest kernel requires tile_k % 32 == 0"
        )
    if split_k < 1:
        raise ValueError(f"Invalid split_k={split_k}; split_k must be >= 1")
    if pack_n != 1:
        raise ValueError(
            f"Current kernel only supports `pack_n=1`; got pack_n={pack_n}"
        )

    warp_atom_m = 16
    warp_atom_n = 16

    if tile_m % (block_m_warps * warp_atom_m) != 0:
        raise ValueError(
            f"Invalid tiling: tile_m={tile_m} must be divisible by "
            f"block_m_warps * 16 = {block_m_warps * warp_atom_m}"
        )
    if tile_n % (block_n_warps * warp_atom_n) != 0:
        raise ValueError(
            f"Invalid tiling: tile_n={tile_n} must be divisible by "
            f"block_n_warps * 16 = {block_n_warps * warp_atom_n}"
        )

    block_n = tile_n
    if n < block_n or n % block_n != 0:
        raise ValueError(
            f"Invalid N for this kernel: N={n} must satisfy N >= {block_n} and N % {block_n} == 0"
        )

    if k % split_k != 0:
        raise ValueError(
            f"Invalid split-K: K={k} must be divisible by split_k={split_k}"
        )

    ks = k // split_k
    if ks < tile_k or ks % tile_k != 0:
        raise ValueError(
            f"Invalid K for this kernel: K/split_k={ks} must satisfy "
            f">= tile_k={tile_k} and % tile_k == 0"
        )

    block_threads = block_m_warps * block_n_warps * block_k_warps * 64
    ldg_vec_size = 8
    block_vecs = ldg_vec_size * block_threads
    block_mk_size = tile_m * tile_k
    block_nk_size = tile_n * tile_k
    block_mn_size = tile_m * tile_n
    if block_mk_size % block_vecs != 0:
        raise ValueError(
            "Invalid tile combination: tile_m * tile_k must be divisible by "
            f"ldg_vec_size * block_threads = {block_vecs}; got {block_mk_size}"
        )
    if block_nk_size % block_vecs != 0:
        raise ValueError(
            "Invalid tile combination: tile_n * tile_k must be divisible by "
            f"ldg_vec_size * block_threads = {block_vecs}; got {block_nk_size}"
        )
    if block_mn_size % block_vecs != 0:
        raise ValueError(
            "Invalid tile combination: tile_m * tile_n must be divisible by "
            f"ldg_vec_size * block_threads = {block_vecs}; got {block_mn_size}"
        )
    ldg_reg_a_count = block_mk_size // block_vecs
    ldg_reg_b_count = block_nk_size // block_vecs
    ldg_reg_c_count = block_mn_size // block_vecs
    if ldg_reg_a_count < 1 or ldg_reg_b_count < 1:
        raise ValueError(
            "Invalid tile combination: requires at least one vectorized global load per thread "
            f"(got ldg_reg_a_count={ldg_reg_a_count}, ldg_reg_b_count={ldg_reg_b_count})"
        )
    if ldg_reg_c_count < 1:
        raise ValueError(
            "Invalid tile combination: requires at least one vectorized C load/store per thread "
            f"(got ldg_reg_c_count={ldg_reg_c_count})"
        )

    lds_bytes = _estimate_hgemm_lds_bytes(
        dtype=dtype,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        stages=stages,
        block_k_warps=block_k_warps,
        b_to_lds=b_to_lds,
    )
    lds_limit = get_shared_memory_per_block(fallback_gfx=get_gfx())
    if lds_bytes > lds_limit:
        raise ValueError(
            "Invalid tile combination: estimated LDS usage "
            f"{lds_bytes} exceeds the hardware limit {lds_limit}"
        )


def _normalize_registry_config(
    *,
    dtype: str,
    stages: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    split_k: int,
    block_m_warps: int,
    block_n_warps: int,
    block_k_warps: int,
    b_to_lds: bool,
) -> dict | None:
    config = {
        "kernel_family": KERNEL_FAMILY_HGEMM,
        "stages": int(stages),
        "tile_m": int(tile_m),
        "tile_n": int(tile_n),
        "tile_k": int(tile_k),
        "split_k": int(split_k),
        "block_m_warps": int(block_m_warps),
        "block_n_warps": int(block_n_warps),
        "block_k_warps": int(block_k_warps),
        "async_copy": KERNEL_ASYNC_COPY,
        "b_to_lds": bool(b_to_lds),
        "b_preshuffle": False,
        "c_to_lds": FIXED_C_TO_LDS,
    }

    try:
        _validate_hgemm_tiling(
            1,
            config["tile_n"],
            config["tile_k"] * config["split_k"],
            dtype=dtype,
            tile_m=config["tile_m"],
            tile_n=config["tile_n"],
            tile_k=config["tile_k"],
            pack_n=1,
            split_k=config["split_k"],
            stages=config["stages"],
            block_m_warps=config["block_m_warps"],
            block_n_warps=config["block_n_warps"],
            block_k_warps=config["block_k_warps"],
            b_to_lds=config["b_to_lds"],
        )
    except ValueError:
        return None

    return config


def _parse_hgemm_kernel_params(name: str) -> dict | None:
    m = _HGEMM_KERNEL_RE.fullmatch(name)
    if m is None:
        return None
    if m.group("a_dtype") != m.group("w_dtype"):
        return None

    kernel_family = (
        KERNEL_FAMILY_SMALL_M
        if m.group("small_m_suffix") is not None
        else KERNEL_FAMILY_HGEMM
    )
    block_k_warps = m.group("block_k_warps")
    block_k_warps = int(block_k_warps) if block_k_warps else 1
    config: dict[str, object] = {
        "kernel_family": kernel_family,
        "stages": int(m.group("stages")),
        "tile_m": int(m.group("tile_m")),
        "tile_n": int(m.group("tile_n")),
        "tile_k": int(m.group("tile_k")),
        "split_k": int(m.group("split_k")),
        "block_m_warps": int(m.group("block_m_warps")),
        "block_n_warps": int(m.group("block_n_warps")),
        "block_k_warps": block_k_warps,
        "async_copy": m.group("async_copy") == "True",
        "b_to_lds": m.group("b_to_lds") == "True",
        "b_preshuffle": m.group("b_preshuffle") == "True",
        "c_to_lds": m.group("c_to_lds") == "True",
        "dtype": m.group("a_dtype"),
        "out_dtype": m.group("out_dtype"),
        "target_gfx": m.group("target_gfx"),
    }
    if kernel_family == KERNEL_FAMILY_SMALL_M:
        config["n_tile_repeat"] = int(m.group("n_tile_repeat") or 1)
        config["persistent_n_tiles"] = int(m.group("persistent_n_tiles") or 1)
        config["waves_per_eu"] = int(m.group("waves_per_eu") or 0)
        config["b_to_lds_unroll"] = int(m.group("b_to_lds_unroll") or 0)
    return config


def get_flydsl_splitk_hgemm_kernel_params(name: str) -> dict | None:
    config = _SPLITK_HGEMM_KERNELS.get(name)
    if config is not None:
        return dict(config)
    config = _parse_hgemm_kernel_params(name)
    if config is not None:
        return dict(config)
    return None


def get_flydsl_splitk_hgemm_kernels(
    dtype: str,
    out_dtype: str,
    *,
    m: int | None = None,
    n: int | None = None,
    k: int | None = None,
) -> dict[str, dict]:
    kernels = {}
    if any(dim is None for dim in (m, n, k)) and any(
        dim is not None for dim in (m, n, k)
    ):
        raise ValueError(
            "m, n, k must be provided together when requesting shape-aware kernels"
        )
    tile_ms = _hgemm_tile_m_options(m)
    for tile_m, tile_n, tile_k, stages, variant in product(
        tile_ms,
        HGEMM_TILE_N_OPTIONS,
        HGEMM_TILE_K_OPTIONS,
        HGEMM_STAGE_OPTIONS,
        KERNEL_CONFIG_VARIANTS,
    ):
        if n is not None and (n < tile_n or n % tile_n != 0):
            continue
        split_k_options = _hgemm_split_k_options(k, tile_k)
        if not split_k_options:
            continue
        for split_k in split_k_options:
            config = _normalize_registry_config(
                dtype=dtype,
                stages=stages,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                split_k=split_k,
                block_m_warps=variant["block_m_warps"],
                block_n_warps=variant["block_n_warps"],
                block_k_warps=variant["block_k_warps"],
                b_to_lds=variant["b_to_lds"],
            )
            if config is None:
                continue
            config["dtype"] = dtype
            config["out_dtype"] = out_dtype
            config["target_gfx"] = get_gfx()
            name = flydsl_kernel_name(
                config["stages"],
                dtype,
                out_dtype,
                config["tile_m"],
                config["tile_n"],
                config["tile_k"],
                config["split_k"],
                config["block_m_warps"],
                config["block_n_warps"],
                config["block_k_warps"],
                config["async_copy"],
                config["b_to_lds"],
                config["b_preshuffle"],
                config["c_to_lds"],
            )
            kernels[name] = config
    # NOTE: Keep the old small_m registry generation here for now, but leave it
    # disabled so shape-aware FlyDSL catalog/tuning only enumerates generic HGEMM.
    #
    # if m is not None and n is not None and k is not None:
    #     for config in (
    #         iter_small_m_registry_configs(
    #             dtype,
    #             out_dtype,
    #             m=m,
    #             n=n,
    #             k=k,
    #         )
    #         or ()
    #     ):
    #         name = flydsl_kernel_name(
    #             config["stage"],
    #             dtype,
    #             out_dtype,
    #             config["tile_m"],
    #             config["tile_n"],
    #             config["tile_k"],
    #             config["split_k"],
    #             config["block_m_warps"],
    #             config["block_n_warps"],
    #             config["async_copy"],
    #             config["b_to_lds"],
    #             c_to_lds=config["c_to_lds"],
    #             kernel_family=KERNEL_FAMILY_SMALL_M,
    #             n_tile_repeat=config["n_tile_repeat"],
    #             persistent_n_tiles=config["persistent_n_tiles"],
    #             waves_per_eu=config["waves_per_eu"],
    #             b_to_lds_unroll=config["b_to_lds_unroll"],
    #         )
    #         kernels[name] = config
    return kernels


def _register_all_configs():
    for dtype in ("bf16", "f16"):
        for out_dtype in ("f16", "bf16"):
            _SPLITK_HGEMM_KERNELS.update(
                get_flydsl_splitk_hgemm_kernels(dtype, out_dtype)
            )


_register_all_configs()


@functools.lru_cache(maxsize=128)
def _get_split_k_tensors(
    device: torch.device,
    stream: torch.cuda.Stream,
) -> tuple[torch.Tensor, torch.Tensor]:
    semaphore = torch.zeros(
        (SPLIT_K_SEMAPHORE_MAX_LEN,), dtype=torch.int32, device=device
    )
    signal = torch.zeros((SPLIT_K_SEMAPHORE_MAX_LEN,), dtype=torch.int32, device=device)
    return semaphore, signal


def _check_split_k_semaphore_capacity(
    m: int, n: int, tile_m: int, tile_n: int, split_k: int
) -> None:
    if split_k <= 1:
        return
    bm = (m + tile_m - 1) // tile_m
    bn = n // tile_n
    required = bm * bn
    if required > SPLIT_K_SEMAPHORE_MAX_LEN:
        raise ValueError(
            "Split-K semaphore capacity exceeded: "
            f"requires {required} counters, max supported is {SPLIT_K_SEMAPHORE_MAX_LEN}"
        )


@functools.lru_cache(maxsize=16384)
def _compile_flydsl_hgemm(
    dtype: str,
    m: int,
    n: int,
    k: int,
    *,
    tile_k: int = 64,
    block_m_warps: int = 2,
    block_n_warps: int = 2,
    block_k_warps: int = 1,
    tile_m: int = 128,
    tile_n: int = 128,
    pack_n: int = 1,
    n_tile_repeat: int = 1,
    persistent_n_tiles: int = 1,
    waves_per_eu: int = 0,
    b_to_lds_unroll: int = 0,
    stages: int = FIXED_STAGE,
    async_copy: bool = False,
    b_to_lds: bool = False,
    b_preshuffle: bool = False,
    split_k: int = 1,
    c_to_lds: bool = False,
    kernel_family: str = KERNEL_FAMILY_HGEMM,
    has_bias: bool = False,
):
    if dtype not in {"f16", "bf16"}:
        raise ValueError(f"`dtype` must be 'f16' or 'bf16', got {dtype!r}")
    if c_to_lds:
        raise ValueError("Current kernel does not support `c_to_lds=True`")

    if kernel_family == KERNEL_FAMILY_HGEMM:
        if b_preshuffle:
            raise ValueError(
                "Current generic kernel only supports `b_preshuffle=False`"
            )
        _validate_hgemm_tiling(
            m,
            n,
            k,
            dtype=dtype,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            pack_n=pack_n,
            split_k=split_k,
            stages=stages,
            block_m_warps=block_m_warps,
            block_n_warps=block_n_warps,
            block_k_warps=block_k_warps,
            b_to_lds=b_to_lds,
        )
    elif kernel_family == KERNEL_FAMILY_SMALL_M:
        if dtype != "bf16":
            raise ValueError(f"small-M kernel only supports `bf16`, got {dtype!r}")
        if b_preshuffle:
            raise ValueError("small-M kernel only supports `b_preshuffle=False`")
        if tile_m != 16:
            raise ValueError(f"small-M kernel fixes tile_m=16; got tile_m={tile_m}")
        if block_m_warps != 1:
            raise ValueError(
                "small-M kernel fixes block_m_warps=1; "
                f"got block_m_warps={block_m_warps}"
            )
    else:
        raise ValueError(
            f"Unsupported kernel_family={kernel_family!r}; expected "
            f"{KERNEL_FAMILY_HGEMM!r} or {KERNEL_FAMILY_SMALL_M!r}"
        )

    kernel = compile_flydsl_hgemm_kernel(
        dtype,
        n,
        k,
        kernel_family=kernel_family,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        pack_n=pack_n,
        split_k=split_k,
        block_m_warps=block_m_warps,
        block_n_warps=block_n_warps,
        block_k_warps=block_k_warps,
        n_tile_repeat=n_tile_repeat,
        persistent_n_tiles=persistent_n_tiles,
        waves_per_eu=waves_per_eu,
        b_to_lds_unroll=b_to_lds_unroll,
        stages=stages,
        async_copy=async_copy,
        b_to_lds=b_to_lds,
        b_preshuffle=b_preshuffle,
        c_to_lds=c_to_lds,
        has_bias=has_bias,
    )

    def launcher(
        out: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        bias: torch.Tensor | None = None,
        stream: torch.cuda.Stream | None = None,
    ):
        if has_bias and bias is None:
            raise ValueError(
                "This launcher was compiled with bias support and requires `bias`."
            )
        if not has_bias and bias is not None:
            raise ValueError(
                "This launcher was compiled without bias support; "
                "recompile with `has_bias=True`."
            )
        launch_bias = b if bias is None else bias
        runtime_m = int(a.shape[0])
        launch_stream = _normalize_launch_stream(a.device, stream)
        _check_split_k_semaphore_capacity(runtime_m, n, tile_m, tile_n, split_k)
        semaphore, signal = _get_split_k_tensors(a.device, launch_stream)
        return _run_compiled(
            kernel,
            ptr_arg(out),
            ptr_arg(a),
            ptr_arg(b),
            ptr_arg(launch_bias),
            runtime_m,
            ptr_arg(semaphore),
            ptr_arg(signal),
            fx.Stream(launch_stream),
        )

    return launcher


def flydsl_gfx950_hgemm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    bias: torch.Tensor | None = None,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 64,
    split_k: int = 1,
    block_m_warps: int = 2,
    block_n_warps: int = 2,
    block_k_warps: int = 1,
    stages: int = 2,
    group_m: int = 0,
    policy: str = "ft",
    out_dtype: torch.dtype | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Run the layout-based mixed-policy gfx950 HGEMM.

    The public aiter convention is preserved: ``a`` is ``[M, K]`` and ``b`` is
    an unshuffled weight tensor with shape ``[N, K]``. The kernel consumes the
    column-major logical ``[K, N]`` view produced by ``b.t()``.
    """

    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(f"Expected 2D inputs, got a.dim={a.dim()} and b.dim={b.dim()}")
    if a.device.type != "cuda" or b.device.type != "cuda":
        raise ValueError("FlyDSL gfx950 HGEMM only supports CUDA/ROCm tensors")
    if a.device != b.device:
        raise ValueError(
            f"`a` and `b` must share a device, got {a.device} and {b.device}"
        )
    target_gfx = _get_runtime_gfx(a.device)
    if target_gfx != "gfx950":
        raise RuntimeError(
            f"The mixed-policy FlyDSL HGEMM only supports gfx950, got {target_gfx}"
        )
    launch_stream = (
        torch.cuda.current_stream(device=a.device) if stream is None else stream
    )
    if launch_stream.device != a.device:
        raise ValueError(f"`stream` must be on {a.device}, got {launch_stream.device}")
    if a.dtype != b.dtype or a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            "FlyDSL gfx950 HGEMM requires matching fp16/bf16 inputs, "
            f"got a.dtype={a.dtype} and b.dtype={b.dtype}"
        )

    m, k = a.shape
    n, weight_k = b.shape
    if weight_k != k:
        raise ValueError(
            f"Incompatible GEMM shapes: a={tuple(a.shape)} b={tuple(b.shape)}"
        )

    if out_dtype is None:
        out_dtype = a.dtype if out is None else out.dtype
    if out_dtype not in (a.dtype, torch.float32):
        raise ValueError(
            f"`out_dtype` must be {a.dtype} or torch.float32, got {out_dtype}"
        )
    if out is not None:
        if out.shape != (m, n):
            raise ValueError(f"`out` must have shape {(m, n)}, got {tuple(out.shape)}")
        if out.dtype != out_dtype:
            raise ValueError(
                f"`out` dtype must match out_dtype={out_dtype}, got {out.dtype}"
            )
        if out.device != a.device:
            raise ValueError(f"`out` must be on {a.device}, got {out.device}")
        if not out.is_contiguous():
            raise ValueError("`out` must be contiguous")

    if bias is not None:
        if bias.shape != (n,):
            raise ValueError(f"`bias` must have shape {(n,)}, got {tuple(bias.shape)}")
        if bias.dtype != a.dtype or bias.device != a.device:
            raise ValueError(
                f"`bias` must have dtype={a.dtype} on {a.device}, "
                f"got dtype={bias.dtype} device={bias.device}"
            )

    normalized_policy = policy.lower()
    if normalized_policy not in {"ft", "ht", "hti"}:
        raise ValueError(
            f"Unsupported gfx950 HGEMM policy {policy!r}; expected 'ft' or 'hti'"
        )
    use_half_tile_interleaved = normalized_policy in {"ht", "hti"}
    user_kwargs = {
        "block_m": tile_m,
        "block_n": tile_n,
        "block_k": tile_k,
        "stages": stages,
        "split_k": split_k,
        "m_waves": block_m_warps,
        "n_waves": block_n_warps,
        "k_waves": block_k_warps,
        "group_m": group_m,
        "use_half_tile_interleaved": use_half_tile_interleaved,
    }

    with torch.cuda.stream(launch_stream):
        if not a.is_contiguous():
            a = a.contiguous()
        if not b.is_contiguous():
            b = b.contiguous()
        if bias is not None and not bias.is_contiguous():
            bias = bias.contiguous()

        return gemm_a16w16_gfx950(
            a,
            b.t(),
            out=out,
            bias=bias,
            user_kwargs=user_kwargs,
            stream=launch_stream,
            layout="nt",
            out_dtype=out_dtype,
        )


def flydsl_hgemm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    bias: torch.Tensor | None = None,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 64,
    pack_n: int = 1,
    split_k: int = 1,
    block_m_warps: int = 2,
    block_n_warps: int = 2,
    block_k_warps: int = 1,
    n_tile_repeat: int = 1,
    persistent_n_tiles: int = 1,
    waves_per_eu: int = 0,
    b_to_lds_unroll: int = 0,
    stages: int = 2,
    async_copy: bool = False,
    b_to_lds: bool = False,
    b_preshuffle: bool = False,
    auto_shuffle_b: bool = False,
    c_to_lds: bool = False,
    kernel_family: str | None = None,
    group_m: int = 0,
    policy: str | None = None,
    out_dtype: torch.dtype | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Run FlyDSL HGEMM."""

    if kernel_family == KERNEL_FAMILY_GFX950 or policy is not None:
        if b_preshuffle or auto_shuffle_b:
            raise ValueError(
                "gfx950 mixed-policy FlyDSL HGEMM requires unshuffled weights"
            )
        return flydsl_gfx950_hgemm(
            a,
            b,
            out=out,
            bias=bias,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            split_k=split_k,
            block_m_warps=block_m_warps,
            block_n_warps=block_n_warps,
            block_k_warps=block_k_warps,
            stages=stages,
            group_m=group_m,
            policy="ft" if policy is None else policy,
            out_dtype=out_dtype,
            stream=stream,
        )

    m, n, k = _validate_hgemm_inputs(a, b, out, bias)
    kernel_dtype = _to_kernel_dtype(a.dtype)

    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    if auto_shuffle_b:
        raise ValueError(
            "`auto_shuffle_b=True` is unsupported because `b_preshuffle=True` "
            "is not supported for generic FlyDSL HGEMM"
        )
    if b_preshuffle:
        raise ValueError(
            "`b_preshuffle=True` is not supported for generic FlyDSL HGEMM"
        )

    if out is None:
        out = torch.empty((m, n), dtype=a.dtype, device=a.device)

    launch_stream = _normalize_launch_stream(a.device, stream)
    resolved_kernel_family = (
        KERNEL_FAMILY_HGEMM if kernel_family is None else kernel_family
    )
    launcher = _compile_flydsl_hgemm(
        kernel_dtype,
        m,
        n,
        k,
        tile_k=tile_k,
        block_m_warps=block_m_warps,
        block_n_warps=block_n_warps,
        block_k_warps=block_k_warps,
        tile_m=tile_m,
        tile_n=tile_n,
        pack_n=pack_n,
        n_tile_repeat=n_tile_repeat,
        persistent_n_tiles=persistent_n_tiles,
        waves_per_eu=waves_per_eu,
        b_to_lds_unroll=b_to_lds_unroll,
        stages=stages,
        async_copy=async_copy,
        b_to_lds=b_to_lds,
        b_preshuffle=b_preshuffle,
        split_k=split_k,
        c_to_lds=c_to_lds,
        kernel_family=resolved_kernel_family,
        has_bias=bias is not None,
    )

    launcher(out, a, b, bias=bias, stream=launch_stream)
    return out


# ---------------------------------------------------------------------------
# FlyDSL preshuffle GEMM kernel management
# ---------------------------------------------------------------------------

_flydsl_compile_fn = None
_flydsl_import_done = False


def _get_compile_fn():
    """Lazy-import compile_preshuffle_gemm so the module loads even without FlyDSL."""
    global _flydsl_compile_fn, _flydsl_import_done
    if _flydsl_import_done:
        return _flydsl_compile_fn
    _flydsl_import_done = True
    if not is_flydsl_available():
        logger.info("[FlyDSL] not available, will fall back to CK/CKTile")
        return None
    try:
        from .kernels.preshuffle_gemm import compile_preshuffle_gemm

        _flydsl_compile_fn = compile_preshuffle_gemm
        logger.info("[FlyDSL] loaded preshuffle GEMM compiler")
    except Exception as e:  # noqa: BLE001
        logger.info(
            f"[FlyDSL] preshuffle GEMM not available, will fall back to CK/CKTile: {e}"
        )
    return _flydsl_compile_fn


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
) -> Tensor:
    """Compile (cached via lru_cache) and run a FlyDSL preshuffle GEMM kernel."""
    compile_fn = _get_compile_fn()
    if compile_fn is None:
        raise RuntimeError("[FlyDSL] compile function not available")
    dtypes = _get_dtypes()

    m, k = XQ.shape[0], XQ.shape[-1]
    n = WQ.shape[0]

    if n % tile_n != 0:
        raise RuntimeError(
            f"[FlyDSL] N ({n}) is not a multiple of tile_n ({tile_n}). "
            f"Arguments not supported! Skipping gemm!"
        )
    if k % tile_k != 0:
        raise RuntimeError(
            f"[FlyDSL] K ({k}) is not a multiple of tile_k ({tile_k}). "
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
            f"[FlyDSL] unsupported output dtype {Out.dtype}; expected torch.bfloat16 or torch.float16"
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
    )

    def _as_i8(t):
        return t.view(torch.int8) if "float8" in str(t.dtype) else t

    out_contig = Out.contiguous()
    # FlyDSL's preshuffle kernel requires an arg_bias slot (used only when
    # epilogue != "none"). Pass an empty tensor as a placeholder for the
    # default epilogue="none" path.
    _dummy_bias = torch.empty(0, dtype=Out.dtype, device=Out.device)
    # The layout-API launcher (PR #754) takes fx.Tensor args (it builds views via
    # fx.get_iter/make_view), so pass flat torch tensors directly rather than raw
    # pointers.
    _run_compiled(
        exe,
        out_contig.view(-1),
        _as_i8(XQ.contiguous()).view(-1),
        _as_i8(WQ.contiguous()).view(-1),
        x_scale.contiguous().view(-1),
        w_scale.contiguous().view(-1),
        _dummy_bias,
        m,
        n,
        fx.Stream(torch.cuda.current_stream()),
    )
    if out_contig is not Out:
        Out.copy_(out_contig)

    return Out
