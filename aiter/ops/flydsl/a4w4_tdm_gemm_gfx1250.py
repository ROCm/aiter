# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dense MXFP4 (a4w4) GEMM on the gfx1250 grouped-MoE TDM pipeline.

``Out[M, N] = dequant(A) @ dequant(B).T`` runs ``launch_gemm_a8w4_tdm`` as a
one-expert grouped GEMM, i.e. exactly the kernel the grouped MoE gemm1 uses
(no activation, bf16/fp16 out). Its operand layouts differ from the dense
``gemm_a4w4`` path, so the prepare helpers convert them once:

* B: ``[N, K / 2]`` fp4x2, 16x16 preshuffled (``shuffle_weight``), the same
  bytes the dense path takes.
* B scale: e8m0 in the grouped n32k4 layout (:func:`prepare_a4w4_tdm_b_scale`
  from the raw ``[N, K / 32]`` scale).
* A and A scale: quantized from bf16 by the grouped quant kernel
  (:func:`prepare_a4w4_tdm_a`), payload A-preshuffled or row-major and the
  scale interleaved over the wave's M rows (M16K4).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class TdmGemmConfig:
    """Launch configuration; defaults mirror the tuned 16384-token gemm1 row."""

    tile_m: int = 256
    tile_n: int = 256
    tile_k: int = 256
    m_warp: int = 2
    n_warp: int = 2
    num_buffers: int = 4
    cluster_m: int = 4
    cluster_n: int = 4
    waves_per_tensor_tdm: int = 1
    next_stage_prefetch: int = 1
    tdm_as_in_prologue: int = 0
    tdm_b_th: int = 0
    lds_soa_load_interleave: int = 1
    rolled_drain: int = 0
    a_preshuffled: int = 1

    @property
    def wmma_rep(self) -> int:
        return self.tile_m // self.m_warp // 16


DEFAULT_CONFIG = TdmGemmConfig()
_tile_maps: dict = {}


def tdm_launch_error(M: int, N: int, K: int, cfg: TdmGemmConfig = DEFAULT_CONFIG):
    """Why (M, N, K) cannot run on ``cfg``, None when it can."""
    rows = cfg.wmma_rep * 16
    if M % rows:
        return f"M={M} must be a multiple of wmma_rep*16={rows}"
    if N % (cfg.tile_n * cfg.cluster_n):
        return f"N={N} must be a multiple of tile_n*cluster_n={cfg.tile_n * cfg.cluster_n}"
    if K % cfg.tile_k:
        return f"K={K} must be a multiple of tile_k={cfg.tile_k}"
    if cfg.a_preshuffled and (K % 1024 or K > 16384):
        return f"A-preshuffled quant needs K % 1024 == 0 and K <= 16384, got K={K}"
    return None


def prepare_a4w4_tdm_a(x: Tensor, cfg: TdmGemmConfig = DEFAULT_CONFIG):
    """Quantize bf16 ``x [M, K]`` into the TDM GEMM's A payload and A scale."""
    from .moe_kernels import flydsl_moe_fused_quant_preshuffle

    M, K = x.shape
    payload, scale = flydsl_moe_fused_quant_preshuffle(
        x.contiguous().view(1, M, K),
        1,
        M,
        wmma_rep=cfg.wmma_rep,
        quant_mode="fp4",
        masked_m=None,
        topids_to_rows=None,
        payload_preshuffled=bool(cfg.a_preshuffled),
    )
    return payload, scale


def prepare_a4w4_tdm_b_scale(b_scale_raw: Tensor) -> Tensor:
    """Raw e8m0 ``[N, K / 32]`` B scale -> the grouped n32k4 layout."""
    from aiter.ops.shuffle import shuffle_scale_n32k4

    N, kb = b_scale_raw.shape
    return shuffle_scale_n32k4(b_scale_raw.contiguous().view(1, N, kb), 1)


def _tile_map(M: int, device) -> Tensor:
    # A per-call H2D copy would land in the timed loop, so keep one per (M, device).
    key = (M, str(device))
    t = _tile_maps.get(key)
    if t is None:
        t = torch.tensor([M], dtype=torch.int32, device=device)
        _tile_maps[key] = t
    return t


def gemm_a4w4_tdm(
    A: Tensor,
    A_scale: Tensor,
    B: Tensor,
    B_scale: Tensor,
    out: Tensor | None = None,
    dtype: torch.dtype = torch.bfloat16,
    cfg: TdmGemmConfig = DEFAULT_CONFIG,
) -> Tensor:
    """Run the TDM GEMM on operands from the prepare helpers; returns ``[M, N]``."""
    from .grouped_gemm_mxfp4 import flydsl_grouped_gemm_a8w4_masked

    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"out dtype must be bf16 or fp16, got {dtype}")
    M = A.shape[-2]
    K = A.shape[-1] * 2
    N = B.shape[0]
    if B.shape[-1] * 2 != K:
        raise ValueError(f"K mismatch: A has K={K}, B has K={B.shape[-1] * 2}")
    err = tdm_launch_error(M, N, K, cfg)
    if err is not None:
        raise ValueError(f"[a4w4 tdm] {err}")
    if out is None:
        out = torch.empty((M, N), dtype=dtype, device=A.device)
    flydsl_grouped_gemm_a8w4_masked(
        out.view(1, M, N),
        A.view(torch.uint8),
        B.view(torch.uint8),
        A_scale,
        B_scale.reshape(-1).view(torch.int32),
        _tile_map(M, A.device),
        n_experts=1,
        contiguous_m=M,
        N=N,
        K=K,
        tile_m=cfg.tile_m,
        tile_n=cfg.tile_n,
        tile_k=cfg.tile_k,
        m_warp=cfg.m_warp,
        n_warp=cfg.n_warp,
        num_buffers=cfg.num_buffers,
        out_is_f16=int(dtype == torch.float16),
        a_is_fp4=1,
        a_preshuffled=cfg.a_preshuffled,
        stage1_act=0,
        gemm_stage=1,
        cluster_m=cfg.cluster_m,
        cluster_n=cfg.cluster_n,
        waves_per_tensor_tdm=cfg.waves_per_tensor_tdm,
        next_stage_prefetch=cfg.next_stage_prefetch,
        tdm_as_in_prologue=cfg.tdm_as_in_prologue,
        tdm_b_th=cfg.tdm_b_th,
        lds_soa_load_interleave=cfg.lds_soa_load_interleave,
        rolled_drain=cfg.rolled_drain,
    )
    return out


__all__ = [
    "DEFAULT_CONFIG",
    "TdmGemmConfig",
    "gemm_a4w4_tdm",
    "prepare_a4w4_tdm_a",
    "prepare_a4w4_tdm_b_scale",
    "tdm_launch_error",
]
