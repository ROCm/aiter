# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Grouped/masked MoE MXScale GEMM helpers for gfx1250.

Initial A8W4 grouped support reuses the tuned gemm_fp8fp4_gfx1250
compile_a8w4_gemm schedule per expert.  The wrapper keeps the grouped/masked
calling convention while the underlying A8W4 GEMM owns TDM/WMMA_SCALE codegen.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Optional

import torch

from aiter.ops.flydsl.kernels.gemm_fp8fp4_gfx1250 import compile_a8w4_gemm, compile_mxfp4_gemm


@dataclass(frozen=True)
class _GroupedA8W4Config:
    model_dim: int
    inter_dim: int
    experts: int
    max_m: int
    tile_m: int
    tile_n: int
    tile_k: int
    m_warp: int
    n_warp: int
    num_buffers: int
    waves_per_eu: Optional[int]
    out_dtype: str
    use_tdm_store: bool
    inst_prefetch: bool
    wave_specialized_tdm: bool
    cluster_m: int
    cluster_n: int
    use_scale_opsel: bool
    expert_sched_mode: bool
    grouped_persistent_m: bool = True
    persistent_workers: Optional[int] = None
    data_format: str = "a8w4"
    act: str = "silu"


def _validate_common(cfg: _GroupedA8W4Config) -> None:
    if cfg.out_dtype not in ("f16", "bf16"):
        raise ValueError(f"out_dtype must be 'f16' or 'bf16', got {cfg.out_dtype!r}")
    if cfg.num_buffers not in (2, 3, 4):
        raise ValueError(f"num_buffers must be 2, 3 or 4, got {cfg.num_buffers}")
    if cfg.data_format not in ("a8w4", "fp4"):
        raise ValueError(f"data_format must be 'a8w4' or 'fp4', got {cfg.data_format!r}")
    if cfg.model_dim % 32 != 0:
        raise ValueError(f"model_dim must be divisible by 32 for MXScale scales, got {cfg.model_dim}")
    if cfg.inter_dim % 32 != 0:
        raise ValueError(f"inter_dim must be divisible by 32 for MXScale scales, got {cfg.inter_dim}")
    if cfg.tile_k % 128 != 0:
        raise ValueError(f"tile_k must be a multiple of 128 for MXScale WMMA_SCALE, got {cfg.tile_k}")
    if cfg.act not in ("silu", "swiglu"):
        raise ValueError(f"act must be 'silu' or 'swiglu', got {cfg.act!r}")
    if cfg.grouped_persistent_m and (cfg.cluster_m != 1 or cfg.cluster_n != 1):
        raise ValueError("grouped_persistent_m currently requires cluster_m=cluster_n=1")


def _to_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _make_m_tile_prefix(masked_m: torch.Tensor, cfg: _GroupedA8W4Config) -> torch.Tensor:
    valid_m = masked_m[:cfg.experts].to(dtype=torch.int32)
    valid_m = valid_m.clamp(min=0, max=cfg.max_m)
    valid_tiles = torch.div(
        valid_m + (cfg.tile_m - 1),
        cfg.tile_m,
        rounding_mode="floor",
    )
    prefix = torch.empty((cfg.experts + 1,), device=masked_m.device, dtype=torch.int32)
    prefix[0].zero_()
    torch.cumsum(valid_tiles, dim=0, out=prefix[1:])
    return prefix


def _check_rank(name: str, tensor: torch.Tensor, rank: int) -> None:
    if tensor.dim() != rank:
        raise ValueError(f"{name} must be rank-{rank}, got shape={tuple(tensor.shape)}")


def _pack_factors(cfg: _GroupedA8W4Config) -> tuple[int, int]:
    if cfg.data_format == "fp4":
        return 2, 2
    return 1, 2


def _preshuffled_scale_shape(rows: int, k_dim: int, warp_tile: int, tile_k: int) -> tuple[int, int]:
    # Matches tests.kernels.test_gemm_fp8fp4_gfx1250.preshuffle_e8m0_scale.
    k_scale = int(k_dim) // 32
    scale_k_per_tile = int(tile_k) // 32
    if k_scale % scale_k_per_tile != 0:
        raise ValueError(f"K scale columns must be divisible by tile_k/32, got {k_scale} and {scale_k_per_tile}")
    wmma_rep = int(warp_tile) // 16
    if wmma_rep < 1:
        raise ValueError(f"warp_tile must be >= 16, got {warp_tile}")
    if int(rows) % wmma_rep != 0:
        raise ValueError(f"scale rows must be divisible by wmma_rep={wmma_rep}, got {rows}")
    return int(rows) // wmma_rep, k_scale * wmma_rep


def _check_stage1_args(y, x, w, scale_x, scale_w, masked_m, cfg: _GroupedA8W4Config) -> None:
    _check_rank("y", y, 3)
    _check_rank("x", x, 3)
    _check_rank("w", w, 3)
    _check_rank("scale_x", scale_x, 3)
    _check_rank("scale_w", scale_w, 3)
    if tuple(y.shape) != (cfg.experts, cfg.max_m, cfg.inter_dim):
        raise ValueError(f"y shape must be {(cfg.experts, cfg.max_m, cfg.inter_dim)}, got {tuple(y.shape)}")
    pack_a, pack_b = _pack_factors(cfg)
    if tuple(x.shape) != (cfg.experts, cfg.max_m, cfg.model_dim // pack_a):
        raise ValueError(f"x shape must be {(cfg.experts, cfg.max_m, cfg.model_dim // pack_a)}, got {tuple(x.shape)}")
    if tuple(w.shape) != (cfg.experts, 2 * cfg.inter_dim, cfg.model_dim // pack_b):
        raise ValueError(f"w shape must be {(cfg.experts, 2 * cfg.inter_dim, cfg.model_dim // pack_b)}, got {tuple(w.shape)}")
    warp_tile_m = cfg.tile_m // cfg.m_warp
    warp_tile_n = cfg.tile_n // cfg.n_warp
    scale_x_shape = _preshuffled_scale_shape(cfg.max_m, cfg.model_dim, warp_tile_m, cfg.tile_k)
    scale_w_shape = _preshuffled_scale_shape(2 * cfg.inter_dim, cfg.model_dim, warp_tile_n, cfg.tile_k)
    if tuple(scale_x.shape) != (cfg.experts, *scale_x_shape):
        raise ValueError(f"scale_x shape must be {(cfg.experts, *scale_x_shape)}, got {tuple(scale_x.shape)}")
    if tuple(scale_w.shape) != (cfg.experts, *scale_w_shape):
        raise ValueError(f"scale_w shape must be {(cfg.experts, *scale_w_shape)}, got {tuple(scale_w.shape)}")
    if masked_m.numel() < cfg.experts:
        raise ValueError(f"masked_m must contain at least {cfg.experts} entries, got {masked_m.numel()}")


def _apply_gate_up(gate: torch.Tensor, up: torch.Tensor, act: str) -> torch.Tensor:
    if act == "swiglu":
        gate = gate.clamp(max=7.0)
        up = up.clamp(min=-7.0, max=7.0)
        return gate * torch.sigmoid(1.702 * gate) * (up + 1.0)
    return torch.nn.functional.silu(gate) * up


def _check_stage2_args(y, x, w, scale_x, scale_w, masked_m, cfg: _GroupedA8W4Config) -> None:
    _check_rank("y", y, 3)
    _check_rank("x", x, 3)
    _check_rank("w", w, 3)
    _check_rank("scale_x", scale_x, 3)
    _check_rank("scale_w", scale_w, 3)
    if tuple(y.shape) != (cfg.experts, cfg.max_m, cfg.model_dim):
        raise ValueError(f"y shape must be {(cfg.experts, cfg.max_m, cfg.model_dim)}, got {tuple(y.shape)}")
    pack_a, pack_b = _pack_factors(cfg)
    if tuple(x.shape) != (cfg.experts, cfg.max_m, cfg.inter_dim // pack_a):
        raise ValueError(f"x shape must be {(cfg.experts, cfg.max_m, cfg.inter_dim // pack_a)}, got {tuple(x.shape)}")
    if tuple(w.shape) != (cfg.experts, cfg.model_dim, cfg.inter_dim // pack_b):
        raise ValueError(f"w shape must be {(cfg.experts, cfg.model_dim, cfg.inter_dim // pack_b)}, got {tuple(w.shape)}")
    warp_tile_m = cfg.tile_m // cfg.m_warp
    warp_tile_n = cfg.tile_n // cfg.n_warp
    scale_x_shape = _preshuffled_scale_shape(cfg.max_m, cfg.inter_dim, warp_tile_m, cfg.tile_k)
    scale_w_shape = _preshuffled_scale_shape(cfg.model_dim, cfg.inter_dim, warp_tile_n, cfg.tile_k)
    if tuple(scale_x.shape) != (cfg.experts, *scale_x_shape):
        raise ValueError(f"scale_x shape must be {(cfg.experts, *scale_x_shape)}, got {tuple(scale_x.shape)}")
    if tuple(scale_w.shape) != (cfg.experts, *scale_w_shape):
        raise ValueError(f"scale_w shape must be {(cfg.experts, *scale_w_shape)}, got {tuple(scale_w.shape)}")
    if masked_m.numel() < cfg.experts:
        raise ValueError(f"masked_m must contain at least {cfg.experts} entries, got {masked_m.numel()}")

def _compile_base_a8w4_gemm(*, K: int, N: int, cfg: _GroupedA8W4Config):
    compiler = compile_mxfp4_gemm if cfg.data_format == "fp4" else compile_a8w4_gemm
    return compiler(
        M=cfg.max_m,
        N=N,
        K=K,
        tile_m=cfg.tile_m,
        tile_n=cfg.tile_n,
        tile_k=cfg.tile_k,
        m_warp=cfg.m_warp,
        n_warp=cfg.n_warp,
        num_buffers=cfg.num_buffers,
        waves_per_eu=cfg.waves_per_eu,
        out_dtype=cfg.out_dtype,
        use_tdm_store=cfg.use_tdm_store,
        inst_prefetch=cfg.inst_prefetch,
        wave_specialized_tdm=cfg.wave_specialized_tdm,
        cluster_m=cfg.cluster_m,
        cluster_n=cfg.cluster_n,
        use_scale_opsel=cfg.use_scale_opsel,
        expert_sched_mode=cfg.expert_sched_mode,
        batch_count=cfg.experts,
        grouped_masked_m=True,
        grouped_persistent_m=cfg.grouped_persistent_m,
        persistent_workers=cfg.persistent_workers,
    )


@functools.lru_cache(maxsize=128)
def compile_moe_grouped_gemm1_a8w4_masked(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    max_m: int,
    tile_m: int = 16,
    tile_n: int = 64,
    tile_k: int = 128,
    m_warp: int = 1,
    n_warp: int = 2,
    out_dtype: str = "f16",
    num_buffers: int = 2,
    waves_per_eu: int | None = None,
    use_tdm_store: bool = True,
    inst_prefetch: bool = False,
    wave_specialized_tdm: bool = False,
    cluster_m: int = 1,
    cluster_n: int = 1,
    use_scale_opsel: bool = False,
    expert_sched_mode: bool = True,
    grouped_persistent_m: bool = True,
    persistent_workers: int | None = None,
    act: str = "silu",
    data_format: str = "a8w4",
):
    cfg = _GroupedA8W4Config(
        model_dim=int(model_dim), inter_dim=int(inter_dim), experts=int(experts),
        max_m=int(max_m), tile_m=int(tile_m), tile_n=int(tile_n), tile_k=int(tile_k),
        m_warp=int(m_warp), n_warp=int(n_warp), num_buffers=int(num_buffers),
        waves_per_eu=waves_per_eu, out_dtype=str(out_dtype), use_tdm_store=bool(use_tdm_store),
        inst_prefetch=bool(inst_prefetch), wave_specialized_tdm=bool(wave_specialized_tdm),
        cluster_m=int(cluster_m), cluster_n=int(cluster_n), use_scale_opsel=bool(use_scale_opsel),
        expert_sched_mode=bool(expert_sched_mode), grouped_persistent_m=bool(grouped_persistent_m),
        persistent_workers=persistent_workers, data_format=str(data_format), act=str(act),
    )
    _validate_common(cfg)
    base = _compile_base_a8w4_gemm(K=cfg.model_dim, N=2 * cfg.inter_dim, cfg=cfg)

    def launch(y, x, w, scale_x, scale_w, masked_m, max_m_arg, inter_dim_arg,
               model_dim_arg, experts_arg, *, stream=None):
        if int(max_m_arg) != cfg.max_m or int(inter_dim_arg) != cfg.inter_dim or int(model_dim_arg) != cfg.model_dim or int(experts_arg) != cfg.experts:
            raise ValueError("runtime dimensions must match compile-time grouped A8W4 stage1 config")
        _check_stage1_args(y, x, w, scale_x, scale_w, masked_m, cfg)
        if stream is None:
            stream = torch.cuda.current_stream()
        tmp = torch.empty((cfg.experts, cfg.max_m, 2 * cfg.inter_dim), device=y.device, dtype=y.dtype)
        if cfg.grouped_persistent_m:
            base(tmp, x, w, scale_x, scale_w, masked_m, masked_m,
                 cfg.max_m, 2 * cfg.inter_dim, stream=stream)
        else:
            base(tmp, x, w, scale_x, scale_w, masked_m,
                 cfg.max_m, 2 * cfg.inter_dim, stream=stream)
        for e in range(cfg.experts):
            valid = _to_int(masked_m[e])
            if valid <= 0:
                continue
            if valid > cfg.max_m:
                raise ValueError(f"masked_m[{e}]={valid} exceeds max_m={cfg.max_m}")
            gate = tmp[e, :valid, :cfg.inter_dim].float()
            up = tmp[e, :valid, cfg.inter_dim:2 * cfg.inter_dim].float()
            y[e, :valid].copy_(_apply_gate_up(gate, up, cfg.act).to(y.dtype))
        return y

    return launch


@functools.lru_cache(maxsize=128)
def compile_moe_grouped_gemm2_a8w4_masked(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    max_m: int,
    tile_m: int = 16,
    tile_n: int = 64,
    tile_k: int = 128,
    m_warp: int = 1,
    n_warp: int = 2,
    out_dtype: str = "f16",
    num_buffers: int = 2,
    waves_per_eu: int | None = None,
    use_tdm_store: bool = True,
    inst_prefetch: bool = False,
    wave_specialized_tdm: bool = False,
    cluster_m: int = 1,
    cluster_n: int = 1,
    use_scale_opsel: bool = False,
    expert_sched_mode: bool = True,
    grouped_persistent_m: bool = True,
    persistent_workers: int | None = None,
    data_format: str = "a8w4",
):
    cfg = _GroupedA8W4Config(
        model_dim=int(model_dim), inter_dim=int(inter_dim), experts=int(experts),
        max_m=int(max_m), tile_m=int(tile_m), tile_n=int(tile_n), tile_k=int(tile_k),
        m_warp=int(m_warp), n_warp=int(n_warp), num_buffers=int(num_buffers),
        waves_per_eu=waves_per_eu, out_dtype=str(out_dtype), use_tdm_store=bool(use_tdm_store),
        inst_prefetch=bool(inst_prefetch), wave_specialized_tdm=bool(wave_specialized_tdm),
        cluster_m=int(cluster_m), cluster_n=int(cluster_n), use_scale_opsel=bool(use_scale_opsel),
        expert_sched_mode=bool(expert_sched_mode), grouped_persistent_m=bool(grouped_persistent_m),
        persistent_workers=persistent_workers, data_format=str(data_format),
    )
    _validate_common(cfg)
    base = _compile_base_a8w4_gemm(K=cfg.inter_dim, N=cfg.model_dim, cfg=cfg)

    def launch(y, x, w, scale_x, scale_w, masked_m, max_m_arg, model_dim_arg,
               inter_dim_arg, experts_arg, *, stream=None):
        if int(max_m_arg) != cfg.max_m or int(model_dim_arg) != cfg.model_dim or int(inter_dim_arg) != cfg.inter_dim or int(experts_arg) != cfg.experts:
            raise ValueError("runtime dimensions must match compile-time grouped A8W4 stage2 config")
        _check_stage2_args(y, x, w, scale_x, scale_w, masked_m, cfg)
        if stream is None:
            stream = torch.cuda.current_stream()
        if cfg.grouped_persistent_m:
            base(y, x, w, scale_x, scale_w, masked_m, masked_m,
                 cfg.max_m, cfg.model_dim, stream=stream)
        else:
            base(y, x, w, scale_x, scale_w, masked_m,
                 cfg.max_m, cfg.model_dim, stream=stream)
        for e in range(cfg.experts):
            valid = _to_int(masked_m[e])
            if valid <= 0:
                continue
            if valid > cfg.max_m:
                raise ValueError(f"masked_m[{e}]={valid} exceeds max_m={cfg.max_m}")
        return y

    return launch


def compile_moe_grouped_gemm1_mxfp4_masked(**kwargs):
    return compile_moe_grouped_gemm1_a8w4_masked(data_format="fp4", **kwargs)


def compile_moe_grouped_gemm2_mxfp4_masked(**kwargs):
    return compile_moe_grouped_gemm2_a8w4_masked(data_format="fp4", **kwargs)


__all__ = [
    "compile_moe_grouped_gemm1_a8w4_masked",
    "compile_moe_grouped_gemm2_a8w4_masked",
    "compile_moe_grouped_gemm1_mxfp4_masked",
    "compile_moe_grouped_gemm2_mxfp4_masked",
]
