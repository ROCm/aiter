# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level gfx950 dense BF16 x MXFP4 FlyDSL API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

from .kernels.gemm_a16wfp4 import compile_gemm_a16wfp4
from .kernels.tensor_shim import _run_compiled

__all__ = [
    "DenseGemmConfig",
    "PreshuffledA16WFP4Weight",
    "a16wfp4_config_legal",
    "a16wfp4_shape_supported",
    "flydsl_gemm_a16wfp4",
    "prepare_gemm_a16wfp4_weight",
]

_TILE_K = 256
_N_ALIGN = 256
_K_ALIGN = 256
_MAX_LDS_BYTES = 163840


class DenseGemmConfig(NamedTuple):
    """Tile and occupancy knobs for one dense A16WFP4 launch."""

    block_m: int
    tile_n: int
    tile_k: int
    k_wave: int
    waves_per_eu: int | None


@dataclass(frozen=True)
class PreshuffledA16WFP4Weight:
    """One-time prepared dense MXFP4 weight and E8M0 scale buffers."""

    weight: torch.Tensor
    scale: torch.Tensor
    n: int
    k: int


def _require_gfx950() -> None:
    arch = str(get_gfx()).split(":", 1)[0]
    if arch != "gfx950":
        raise RuntimeError(f"flydsl_gemm_a16wfp4 requires gfx950, got {arch!r}")


def a16wfp4_shape_supported(n: int, k: int) -> bool:
    """Return whether packed ``[N, K]`` can run the fused gfx950 kernel."""
    return n > 0 and k > 0 and n % _N_ALIGN == 0 and k % _K_ALIGN == 0


def a16wfp4_config_legal(n: int, k: int, cfg: DenseGemmConfig) -> bool:
    """Return whether ``cfg`` can compile for packed ``[N, K]`` without overflowing LDS."""
    if cfg.block_m <= 0 or cfg.block_m % 16:
        return False
    if cfg.tile_k <= 0 or cfg.tile_k % 128 or k % cfg.tile_k:
        return False
    if cfg.k_wave not in (1, 2, 4):
        return False
    if k % (cfg.k_wave * cfg.tile_k):
        return False
    n_tile_align = 16 * (4 // cfg.k_wave)
    if cfg.tile_n < n_tile_align or cfg.tile_n % n_tile_align or n % cfg.tile_n:
        return False
    from .kernels.gemm_a16wfp4_helpers import DenseGemmTraits

    try:
        traits = DenseGemmTraits(
            n,
            k,
            cfg.block_m,
            cfg.tile_n,
            cfg.tile_k,
            2,
            k_wave=cfg.k_wave,
            waves_per_eu=cfg.waves_per_eu,
        )
    except ValueError:
        return False
    return traits.lds_bytes <= _MAX_LDS_BYTES


def _select_tile_n(n: int, prefer: int) -> int:
    if n % prefer == 0:
        return prefer
    if n % 256 == 0:
        return 256
    if n % 128 == 0:
        return 128
    raise ValueError(f"N={n} is not divisible by 128 or 256")


def _select_k_wave(k: int, tile_k: int, max_k_wave: int = 4) -> int:
    for k_wave in (4, 2, 1):
        if k_wave <= max_k_wave and k % (k_wave * tile_k) == 0:
            return k_wave
    return 1


_TARGET_BLOCKS = 256  # gfx950 CU count


def _decode_tile_n(m: int, n: int, k_wave: int) -> int:
    """Widest N tile that still puts a workgroup on every CU."""
    align = 16 * (4 // k_wave)
    grid_m = (m + 15) // 16
    legal = [
        tile_n
        for tile_n in (256, 128, 64, 32, 16)
        if tile_n >= align and tile_n % align == 0 and n % tile_n == 0
    ]
    for tile_n in legal:
        if grid_m * (n // tile_n) >= _TARGET_BLOCKS:
            return tile_n
    return legal[-1]


def _decode_config(m: int, n: int, k: int) -> DenseGemmConfig:
    """Decode is grid-bound, not bandwidth-bound: size the tile to fill the GPU.

    Measured on MI355X at M=1: the fused kernel sustains ~17 GB/s per resident
    workgroup, so time scales with 1/workgroups until every CU has one. Halving
    TILE_N from the compute-optimal 128 down to the grid-optimal width measured
    1.56x on (8448,7168) and 3.09x on (1536,7168).
    """
    k_wave = _select_k_wave(k, _TILE_K, max_k_wave=4)
    if k_wave == 1:
        # K too short for split-K: its K loop cannot hide the extra dispatch of
        # a narrower tile, so keep the wide one.
        return DenseGemmConfig(16, _select_tile_n(n, 128), _TILE_K, 1, 2)
    return DenseGemmConfig(
        block_m=16,
        tile_n=_decode_tile_n(m, n, k_wave),
        tile_k=_TILE_K,
        k_wave=k_wave,
        waves_per_eu=4 if n >= 4096 else 1,
    )


# Measured gfx950 winners for Kimi-K3 production (N, K), as ascending M
# breakpoints; the last entry with ``m_min <= M`` wins. A wider tile decodes
# fewer MXFP4 blocks per MFMA but launches fewer workgroups, and that trade only
# pays when the resulting tile count still fills whole 256-CU waves, so the
# winner moves with M rather than with a single size class.
_K3_MEASURED: dict[tuple[int, int], tuple[tuple[int, DenseGemmConfig], ...]] = {
    (8448, 7168): (
        (64, DenseGemmConfig(32, 128, 256, 1, 1)),
        (512, DenseGemmConfig(48, 192, 256, 1, 2)),
        (1024, DenseGemmConfig(64, 192, 128, 1, 2)),
    ),
    (1536, 7168): (
        (64, DenseGemmConfig(16, 96, 128, 4, 2)),
        (768, DenseGemmConfig(32, 128, 256, 1, 2)),
        (2048, DenseGemmConfig(32, 256, 128, 1, 1)),
    ),
    (7168, 768): (
        (64, DenseGemmConfig(32, 128, 256, 1, 2)),
        (1024, DenseGemmConfig(64, 256, 128, 1, 2)),
    ),
    (3584, 512): (
        (64, DenseGemmConfig(32, 128, 256, 1, 2)),
        (1024, DenseGemmConfig(32, 256, 128, 1, 1)),
    ),
}


def _select_gemm_config(m: int, n: int, k: int) -> DenseGemmConfig:
    """Grid-filling decode rule, then measured K3 breakpoints, then a heuristic."""
    if m < 64:
        return _decode_config(m, n, k)
    measured = _K3_MEASURED.get((n, k))
    if measured is not None:
        return next(cfg for m_min, cfg in reversed(measured) if m >= m_min)

    tile_k = _TILE_K
    if m >= 4096 or (k <= 512 and m >= 2048):
        return DenseGemmConfig(
            block_m=32,
            tile_n=_select_tile_n(n, 256),
            tile_k=tile_k,
            k_wave=1,
            waves_per_eu=1,
        )
    return DenseGemmConfig(
        block_m=16,
        tile_n=_select_tile_n(n, 128),
        tile_k=tile_k,
        k_wave=1,
        waves_per_eu=2,
    )


def prepare_gemm_a16wfp4_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> PreshuffledA16WFP4Weight:
    """Preshuffle packed E2M1 ``[N,K/2]`` and E8M0 ``[N,K/32]`` once."""
    _require_gfx950()
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError(
            "weight and scale must be 2D; "
            f"got weight.ndim={weight.ndim}, scale.ndim={scale.ndim}"
        )
    if weight.device.type != "cuda" or scale.device.type != "cuda":
        raise ValueError("weight and scale must be CUDA/ROCm tensors")
    if weight.device != scale.device:
        raise ValueError(
            f"weight and scale must be on the same device, got {weight.device} and {scale.device}"
        )
    if weight.element_size() != 1 or scale.element_size() != 1:
        raise TypeError(
            "weight must be byte-packed E2M1 and scale must be byte E8M0; "
            f"got element sizes {weight.element_size()} and {scale.element_size()}"
        )

    n, packed_k = weight.shape
    k = packed_k * 2
    if n % 256:
        raise ValueError(f"N must be divisible by 256, got {n}")
    if k % 256:
        raise ValueError(f"K must be divisible by 256, got {k}")
    if tuple(scale.shape) != (n, k // 32):
        raise ValueError(
            f"scale must have shape {(n, k // 32)}, got {tuple(scale.shape)}"
        )

    weight_u8 = weight.contiguous().view(torch.uint8)
    scale_u8 = scale.contiguous().view(torch.uint8)
    shuffled_weight = shuffle_weight_a16w4(
        weight_u8.unsqueeze(0), NLane=16, gate_up=False
    ).squeeze(0)
    shuffled_scale = shuffle_scale_a16w4(scale_u8, experts_cnt=1, gate_up=False)
    return PreshuffledA16WFP4Weight(
        weight=shuffled_weight,
        scale=shuffled_scale,
        n=n,
        k=k,
    )


def flydsl_gemm_a16wfp4(
    a: torch.Tensor,
    prepared_weight: PreshuffledA16WFP4Weight,
    *,
    out: torch.Tensor | None = None,
    stream: torch.cuda.Stream | None = None,
    config: DenseGemmConfig | None = None,
) -> torch.Tensor:
    """Compute ``A @ W.T`` without quantizing or otherwise transforming ``A``."""
    _require_gfx950()
    if not isinstance(prepared_weight, PreshuffledA16WFP4Weight):
        raise TypeError("prepared_weight must come from prepare_gemm_a16wfp4_weight()")
    if a.ndim != 2:
        raise ValueError(f"a must be 2D, got a.ndim={a.ndim}")
    if a.dtype != torch.bfloat16:
        raise TypeError(f"a must be BF16, got {a.dtype}")
    if a.device.type != "cuda":
        raise ValueError("a must be a CUDA/ROCm tensor")
    if not a.is_contiguous():
        raise ValueError("a must be contiguous")

    m, k = a.shape
    if m <= 0:
        raise ValueError(f"M must be positive, got {m}")
    if k != prepared_weight.k:
        raise ValueError(
            f"a K={k} does not match prepared weight K={prepared_weight.k}"
        )
    if a.device != prepared_weight.weight.device:
        raise ValueError(
            f"a and prepared weight must share a device, got {a.device} and "
            f"{prepared_weight.weight.device}"
        )

    expected_shape = (m, prepared_weight.n)
    if out is None:
        out = torch.empty(expected_shape, dtype=torch.bfloat16, device=a.device)
    else:
        if tuple(out.shape) != expected_shape:
            raise ValueError(
                f"out must have shape {expected_shape}, got {tuple(out.shape)}"
            )
        if out.dtype != torch.bfloat16 or out.device != a.device:
            raise ValueError(
                f"out must be BF16 on {a.device}, got dtype={out.dtype}, device={out.device}"
            )
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")

    launch_stream = torch.cuda.current_stream(a.device) if stream is None else stream
    if launch_stream.device != a.device:
        raise ValueError(
            f"stream must be on {a.device}, got stream device {launch_stream.device}"
        )

    cfg = _select_gemm_config(m, prepared_weight.n, k) if config is None else config
    if not isinstance(cfg, DenseGemmConfig):
        raise TypeError(f"config must be DenseGemmConfig, got {type(cfg)!r}")
    launcher = compile_gemm_a16wfp4(
        N=prepared_weight.n,
        K=prepared_weight.k,
        BM=cfg.block_m,
        TILE_N=cfg.tile_n,
        TILE_K=cfg.tile_k,
        k_wave=cfg.k_wave,
        waves_per_eu=cfg.waves_per_eu,
    )
    _run_compiled(
        launcher,
        a.data_ptr(),
        prepared_weight.weight.data_ptr(),
        prepared_weight.scale.data_ptr(),
        out.data_ptr(),
        int(m),
        launch_stream,
    )
    return out
