# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL HGEMM APIs."""

from __future__ import annotations

import re
from itertools import product
from typing import Dict, Optional

import torch
from torch import Tensor

from aiter import logger
from flydsl.runtime.device import get_rocm_arch

from aiter.jit.utils.chip_info import get_gfx

from ..shuffle import shuffle_weight
from .kernels.hgemm_dispatch import compile_flydsl_hgemm_kernel
from .kernels.small_m_hgemm import iter_small_m_registry_configs
from .kernels.tensor_shim import _run_compiled
from .utils import get_shared_memory_per_block, is_flydsl_available

__all__ = [
    "flydsl_hgemm",
]


def _get_dtypes():
    from aiter.utility import dtypes

    return dtypes

SPLIT_K_COUNTER_MAX_LEN = 128
SPLIT_K_SIGNAL_STATE_COUNT = 3
FIXED_STAGE = 2
FIXED_C_TO_LDS = False
KERNEL_ASYNC_COPY = get_rocm_arch() != "gfx942"
KERNEL_FAMILY_HGEMM = "hgemm"
KERNEL_FAMILY_SMALL_M = "small_m"
_HGEMM_KERNEL_RE = re.compile(
    r"^flydsl_gemm(?P<stage>\d+)_"
    r"a(?P<a_dtype>[a-z0-9]+)_w(?P<w_dtype>[a-z0-9]+)_(?P<out_dtype>[a-z0-9]+)_"
    r"t(?P<tile_m>\d+)x(?P<tile_n>\d+)x(?P<tile_k>\d+)_"
    r"split_k(?P<split_k>\d+)_"
    r"block_m_warp(?P<block_m_warps>\d+)_"
    r"block_n_warp(?P<block_n_warps>\d+)_"
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

SplitKStreamKey = tuple[int, int]
SPLIT_K_GLOBAL_SEMAPHORE: dict[SplitKStreamKey, torch.Tensor] = {}
SPLIT_K_GLOBAL_SEMAPHORE_STATE: dict[SplitKStreamKey, int] = {}

# Expand the original default HGEMM catalog with the extra cases that proved
# useful during the wider one-off search, instead of maintaining separate
# search-space modes.
HGEMM_TILE_N_OPTIONS = (64, 128, 160, 192, 256)
HGEMM_TILE_K_OPTIONS = (64, 96, 128, 160, 256)
HGEMM_TILE_M_OPTIONS = (16, 32, 48, 64, 80, 96, 112, 128, 160, 256)
HGEMM_BASE_SPLIT_K_OPTIONS = (1, 2, 4, 8, 16)
HGEMM_MAX_SPLIT_K = 32
HGEMM_EXTRA_BLOCK_K_LOOPS_MIN = 2
HGEMM_EXTRA_BLOCK_K_LOOPS_MAX = 8
KERNEL_CONFIG_VARIANTS = (
    {
        "block_m_warps": 1,
        "block_n_warps": 2,
        "b_to_lds": False,
    },
    {
        "block_m_warps": 1,
        "block_n_warps": 4,
        "b_to_lds": False,
    },
    {
        "block_m_warps": 2,
        "block_n_warps": 2,
        "b_to_lds": False,
    },
    {
        "block_m_warps": 1,
        "block_n_warps": 4,
        "b_to_lds": True,
    },
    {
        "block_m_warps": 2,
        "block_n_warps": 2,
        "b_to_lds": True,
    },
)

_SPLITK_HGEMM_KERNELS: Dict[str, Dict] = {}


def _normalize_supported_kernel_metadata(
    *,
    stage: int,
    async_copy: bool,
    c_to_lds: bool,
) -> tuple[int, bool, bool]:
    # Latest `hgemm.py` fixes these choices internally instead of exposing
    # multiple codegen variants to the wrapper layer.
    if stage != FIXED_STAGE:
        raise ValueError(
            f"Current kernel only supports stage={FIXED_STAGE}; got stage={stage}"
        )
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
    return FIXED_STAGE, KERNEL_ASYNC_COPY, FIXED_C_TO_LDS


def flydsl_kernel_name(
    stage: int,
    dtype: str,
    out_dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    split_k: int,
    block_m_warp: int,
    block_n_warp: int,
    async_copy: bool,
    b_to_lds: bool,
    b_preshuffle: bool,
    c_to_lds: bool,
    kernel_family: str = KERNEL_FAMILY_HGEMM,
    n_tile_repeat: int = 1,
    persistent_n_tiles: int = 1,
    waves_per_eu: int = 0,
    b_to_lds_unroll: int = 0,
) -> str:
    stage, async_copy, c_to_lds = _normalize_supported_kernel_metadata(
        stage=stage,
        async_copy=async_copy,
        c_to_lds=c_to_lds,
    )
    if b_preshuffle and b_to_lds:
        raise ValueError(
            "Current kernel requires b_to_lds=False when b_preshuffle=True"
        )
    name = (
        f"flydsl_gemm{stage}_a{dtype}_w{dtype}_{out_dtype}_t{tile_m}x{tile_n}x{tile_k}"
    )
    name += f"_split_k{split_k}_block_m_warp{block_m_warp}_block_n_warp{block_n_warp}"
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
    stream: Optional[torch.cuda.Stream],
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


def _hgemm_tile_m_options(m: Optional[int]) -> tuple[int, ...]:
    if m is None:
        return HGEMM_TILE_M_OPTIONS
    max_tile_m = max(96, _align_up(max(1, m) * 2, 16))
    return tuple(tile_m for tile_m in HGEMM_TILE_M_OPTIONS if tile_m <= max_tile_m)


def _hgemm_split_k_options(k: Optional[int], tile_k: int) -> tuple[int, ...]:
    if k is None:
        return HGEMM_BASE_SPLIT_K_OPTIONS
    options = set()
    for split_k in range(1, HGEMM_MAX_SPLIT_K + 1):
        if k % split_k != 0 or (k // split_k) % tile_k != 0:
            continue
        if split_k in HGEMM_BASE_SPLIT_K_OPTIONS:
            options.add(split_k)
            continue
        block_k_loops = k // (split_k * tile_k)
        if (
            HGEMM_EXTRA_BLOCK_K_LOOPS_MIN
            <= block_k_loops
            <= HGEMM_EXTRA_BLOCK_K_LOOPS_MAX
        ):
            options.add(split_k)
    return tuple(sorted(options))


def _estimate_hgemm_lds_bytes(
    *,
    dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    stages: int,
    b_to_lds: bool,
) -> int:
    if dtype not in {"f16", "bf16"}:
        raise ValueError(f"`dtype` must be 'f16' or 'bf16', got {dtype!r}")

    dtype_bytes = 2
    a_lds_bytes = max(
        stages * tile_m * tile_k * dtype_bytes,
        tile_m * tile_n * dtype_bytes,
    )
    if not b_to_lds:
        return a_lds_bytes
    return _align_up(a_lds_bytes, 16) + stages * tile_n * tile_k * dtype_bytes


def _get_flydsl_shuffle_layout(pack_n: int) -> tuple[int, int]:
    return (16 * pack_n, 16)


def _validate_hgemm_inputs(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
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
    b_to_lds: bool,
) -> None:
    del m

    if tile_m < 1 or tile_n < 1 or tile_k < 1:
        raise ValueError(
            f"Tile sizes must be positive, got tile_m={tile_m}, tile_n={tile_n}, tile_k={tile_k}"
        )
    if block_m_warps < 1 or block_n_warps < 1:
        raise ValueError(
            "Warp tiling must be positive, got "
            f"block_m_warps={block_m_warps}, block_n_warps={block_n_warps}"
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
    if stages != FIXED_STAGE:
        raise ValueError(
            f"Invalid stages={stages}; current kernel always compiles a {FIXED_STAGE}-stage kernel"
        )
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

    block_threads = block_m_warps * block_n_warps * 64
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
    stage: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    split_k: int,
    block_m_warps: int,
    block_n_warps: int,
    b_to_lds: bool,
    b_preshuffle: bool,
) -> Optional[Dict]:
    config = {
        "kernel_family": KERNEL_FAMILY_HGEMM,
        "stage": FIXED_STAGE,
        "tile_m": int(tile_m),
        "tile_n": int(tile_n),
        "tile_k": int(tile_k),
        "split_k": int(split_k),
        "block_m_warps": int(block_m_warps),
        "block_n_warps": int(block_n_warps),
        "async_copy": KERNEL_ASYNC_COPY,
        "b_to_lds": bool(b_to_lds),
        "b_preshuffle": bool(b_preshuffle),
        "c_to_lds": FIXED_C_TO_LDS,
    }
    if stage != FIXED_STAGE:
        return None
    if config["b_preshuffle"] and config["b_to_lds"]:
        return None

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
            stages=FIXED_STAGE,
            block_m_warps=config["block_m_warps"],
            block_n_warps=config["block_n_warps"],
            b_to_lds=config["b_to_lds"],
        )
    except ValueError:
        return None

    return config


def _parse_hgemm_kernel_params(name: str) -> Optional[Dict]:
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
    config: Dict[str, object] = {
        "kernel_family": kernel_family,
        "stage": int(m.group("stage")),
        "tile_m": int(m.group("tile_m")),
        "tile_n": int(m.group("tile_n")),
        "tile_k": int(m.group("tile_k")),
        "split_k": int(m.group("split_k")),
        "block_m_warps": int(m.group("block_m_warps")),
        "block_n_warps": int(m.group("block_n_warps")),
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


def get_flydsl_splitk_hgemm_kernel_params(name: str) -> Optional[Dict]:
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
    m: Optional[int] = None,
    n: Optional[int] = None,
    k: Optional[int] = None,
) -> Dict[str, Dict]:
    kernels = {}
    if any(dim is None for dim in (m, n, k)) and any(
        dim is not None for dim in (m, n, k)
    ):
        raise ValueError(
            "m, n, k must be provided together when requesting shape-aware kernels"
        )
    b_preshuffles = [False, True]
    tile_ms = _hgemm_tile_m_options(m)
    for tile_m, tile_n, tile_k, b_preshuffle, variant in product(
        tile_ms,
        HGEMM_TILE_N_OPTIONS,
        HGEMM_TILE_K_OPTIONS,
        b_preshuffles,
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
                stage=FIXED_STAGE,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                split_k=split_k,
                block_m_warps=variant["block_m_warps"],
                block_n_warps=variant["block_n_warps"],
                b_to_lds=variant["b_to_lds"],
                b_preshuffle=b_preshuffle,
            )
            if config is None:
                continue
            config["dtype"] = dtype
            config["out_dtype"] = out_dtype
            config["target_gfx"] = get_gfx()
            name = flydsl_kernel_name(
                config["stage"],
                dtype,
                out_dtype,
                config["tile_m"],
                config["tile_n"],
                config["tile_k"],
                config["split_k"],
                config["block_m_warps"],
                config["block_n_warps"],
                config["async_copy"],
                config["b_to_lds"],
                config["b_preshuffle"],
                config["c_to_lds"],
            )
            kernels[name] = config
    if m is not None and n is not None and k is not None:
        for config in (
            iter_small_m_registry_configs(
                dtype,
                out_dtype,
                m=m,
                n=n,
                k=k,
            )
            or ()
        ):
            name = flydsl_kernel_name(
                config["stage"],
                dtype,
                out_dtype,
                config["tile_m"],
                config["tile_n"],
                config["tile_k"],
                config["split_k"],
                config["block_m_warps"],
                config["block_n_warps"],
                config["async_copy"],
                config["b_to_lds"],
                config["b_preshuffle"],
                config["c_to_lds"],
                kernel_family=KERNEL_FAMILY_SMALL_M,
                n_tile_repeat=config["n_tile_repeat"],
                persistent_n_tiles=config["persistent_n_tiles"],
                waves_per_eu=config["waves_per_eu"],
                b_to_lds_unroll=config["b_to_lds_unroll"],
            )
            kernels[name] = config
    return kernels


def _register_all_configs():
    for dtype in ("bf16", "f16"):
        for out_dtype in ("f16", "bf16"):
            _SPLITK_HGEMM_KERNELS.update(
                get_flydsl_splitk_hgemm_kernels(dtype, out_dtype)
            )


_register_all_configs()


def _get_split_k_global_semaphore(stream: torch.cuda.Stream) -> torch.Tensor:
    key = _stream_cache_key(stream)
    semaphore = SPLIT_K_GLOBAL_SEMAPHORE.get(key)
    if semaphore is None:
        semaphore = torch.zeros(
            (SPLIT_K_SIGNAL_STATE_COUNT * SPLIT_K_COUNTER_MAX_LEN,),
            dtype=torch.int32,
            device=stream.device,
        )
        SPLIT_K_GLOBAL_SEMAPHORE[key] = semaphore
        SPLIT_K_GLOBAL_SEMAPHORE_STATE[key] = int(0)
    return semaphore


def _get_split_k_signal_state(stream: torch.cuda.Stream) -> int:
    _get_split_k_global_semaphore(stream)
    return SPLIT_K_GLOBAL_SEMAPHORE_STATE[_stream_cache_key(stream)]


def _advance_split_k_signal_state(stream: torch.cuda.Stream) -> None:
    key = _stream_cache_key(stream)
    SPLIT_K_GLOBAL_SEMAPHORE_STATE[key] = (
        _get_split_k_signal_state(stream) + 1
    ) % SPLIT_K_SIGNAL_STATE_COUNT


def _check_split_k_counter_capacity(
    m: int, n: int, tile_m: int, tile_n: int, split_k: int
) -> None:
    if split_k <= 1:
        return
    bm = (m + tile_m - 1) // tile_m
    bn = n // tile_n
    required = bm * bn
    if required > SPLIT_K_COUNTER_MAX_LEN:
        raise ValueError(
            "Split-K counter capacity exceeded: "
            f"requires {required} counters, max supported is {SPLIT_K_COUNTER_MAX_LEN}"
        )


def _compile_flydsl_hgemm(
    dtype: str,
    m: int,
    n: int,
    k: int,
    *,
    tile_k: int = 64,
    block_m_warps: int = 1,
    block_n_warps: int = 4,
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
    b_preshuffle: bool = True,
    split_k: int = 1,
    c_to_lds: bool = False,
    kernel_family: str = KERNEL_FAMILY_HGEMM,
    has_bias: bool = False,
):
    if dtype not in {"f16", "bf16"}:
        raise ValueError(f"`dtype` must be 'f16' or 'bf16', got {dtype!r}")
    if b_preshuffle and b_to_lds:
        raise ValueError(
            "Current kernel requires b_to_lds=False when b_preshuffle=True"
        )
    if c_to_lds:
        raise ValueError("Current kernel does not support `c_to_lds=True`")

    if kernel_family == KERNEL_FAMILY_HGEMM:
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
            b_to_lds=b_to_lds,
        )
    elif kernel_family == KERNEL_FAMILY_SMALL_M:
        if dtype != "bf16":
            raise ValueError(f"small-M kernel only supports `bf16`, got {dtype!r}")
        if stages != FIXED_STAGE:
            raise ValueError(
                f"small-M kernel only supports stage={FIXED_STAGE}; got stage={stages}"
            )
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
        signal_state: int,
        bias: Optional[torch.Tensor] = None,
        stream: Optional[torch.cuda.Stream] = None,
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
        _check_split_k_counter_capacity(runtime_m, n, tile_m, tile_n, split_k)
        launch_stream = _normalize_launch_stream(a.device, stream)
        semaphore = _get_split_k_global_semaphore(launch_stream)
        return _run_compiled(
            kernel,
            out,
            a,
            b,
            launch_bias,
            runtime_m,
            semaphore,
            signal_state,
            launch_stream,
        )

    return launcher


def flydsl_hgemm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    bias: Optional[torch.Tensor] = None,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 64,
    pack_n: int = 1,
    split_k: int = 1,
    block_m_warps: int = 1,
    block_n_warps: int = 4,
    n_tile_repeat: int = 1,
    persistent_n_tiles: int = 1,
    waves_per_eu: int = 0,
    b_to_lds_unroll: int = 0,
    stages: int = FIXED_STAGE,
    async_copy: bool = False,
    b_to_lds: bool = False,
    b_preshuffle: bool = True,
    auto_shuffle_b: bool = False,
    c_to_lds: bool = False,
    kernel_family: Optional[str] = None,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """Run FlyDSL HGEMM."""

    m, n, k = _validate_hgemm_inputs(a, b, out, bias)
    kernel_dtype = _to_kernel_dtype(a.dtype)

    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    if b_preshuffle and not getattr(b, "is_shuffled", False):
        if auto_shuffle_b:
            b = shuffle_weight(b, layout=_get_flydsl_shuffle_layout(pack_n))
        else:
            raise ValueError(
                "`b_preshuffle=True` expects `b` to be pre-shuffled. "
                f"Use `shuffle_weight(b, layout={_get_flydsl_shuffle_layout(pack_n)})` "
                "first or pass `auto_shuffle_b=True`."
            )

    if out is None:
        out = torch.empty((m, n), dtype=a.dtype, device=a.device)

    launch_stream = _normalize_launch_stream(a.device, stream)
    signal_state = _get_split_k_signal_state(launch_stream)
    launcher = _compile_flydsl_hgemm(
        kernel_dtype,
        m,
        n,
        k,
        tile_k=tile_k,
        block_m_warps=block_m_warps,
        block_n_warps=block_n_warps,
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
        kernel_family=(KERNEL_FAMILY_HGEMM if kernel_family is None else kernel_family),
        has_bias=bias is not None,
    )

    launcher(out, a, b, signal_state=signal_state, bias=bias, stream=launch_stream)
    if split_k > 1:
        _advance_split_k_signal_state(launch_stream)
    return out


# ---------------------------------------------------------------------------
# FlyDSL preshuffle GEMM kernel management
# ---------------------------------------------------------------------------

_flydsl_compile_fn = None
_flydsl_import_done = False


def _get_compile_fn():
    """Lazy-import compile_preshuffle_gemm_a8 so the module loads even without FlyDSL."""
    global _flydsl_compile_fn, _flydsl_import_done
    if _flydsl_import_done:
        return _flydsl_compile_fn
    _flydsl_import_done = True
    if not is_flydsl_available():
        logger.info("[FlyDSL] not available, will fall back to CK/CKTile")
        return None
    try:
        from .kernels.preshuffle_gemm import compile_preshuffle_gemm_a8

        _flydsl_compile_fn = compile_preshuffle_gemm_a8
        logger.info("[FlyDSL] loaded preshuffle GEMM compiler")
    except Exception as e:
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
    lds_stage: int = 2,
    use_cshuffle_epilog: int = 0,
    use_async_copy: int = 0,
    waves_per_eu: int = 0,
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
        lds_stage=lds_stage,
        use_cshuffle_epilog=bool(use_cshuffle_epilog),
        use_async_copy=bool(use_async_copy),
        waves_per_eu=wpe,
    )

    def _as_i8(t):
        return t.view(torch.int8) if "float8" in str(t.dtype) else t

    out_contig = Out.contiguous()
    _run_compiled(
        exe,
        out_contig.view(-1),
        _as_i8(XQ.contiguous()).view(-1),
        _as_i8(WQ.contiguous()).view(-1),
        x_scale.contiguous().view(-1),
        w_scale.contiguous().view(-1),
        m,
        n,
        torch.cuda.current_stream(),
    )
    if out_contig is not Out:
        Out.copy_(out_contig)

    return Out
