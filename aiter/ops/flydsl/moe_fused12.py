# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused grouped MoE stage 1 + stage 2 on gfx1250 (one launch; see kernels/moe_fused12_gfx1250_tdm.py).

Opt-in: AITER_MOE_FUSED12=1 (default off). Bitwise with the two-launch path by construction (the same tile bodies);
eligible only for the plain TP decode shape: fused act+requant stage 1, equal tile_m, no clusters / persistent workers
/ EP / shared expert / bias, and at most AITER_MOE_FUSED12_MAX_TOKENS (default 64) tokens.
"""

from __future__ import annotations

import functools
import os

import torch

from .grouped_gemm_mxfp4 import (
    _gather_scale_prologue,
    _select_bool_env,
    _select_cluster_m,
    _select_cluster_n,
    _select_num_waves_per_tensor_tdm,
    _select_tdm_b_th,
)
from .kernels.tensor_shim import fast_jit, ptr_arg


def fused12_enabled() -> bool:
    return os.environ.get("AITER_MOE_FUSED12", "0") == "1"


def fused12_max_tokens() -> int:
    return int(os.environ.get("AITER_MOE_FUSED12_MAX_TOKENS", "64"))


_CNT = {}
# m-tiles a launch may address: contiguous_m / tile_m. Decode at tile_m 16 and <= 64 tokens needs
# (64*8 + 256*16) / 16 = 288; keep headroom.
_CNT_LEN = 1 << 16


def fused12_counters(device) -> torch.Tensor | None:
    """Persistent zeroed per-m-tile counters (self-resetting in the kernel). Allocated on the first eager call; None
    while a CUDA graph is being captured before any eager call (the caller then keeps the two-launch path)."""
    device = torch.device(device)
    if device.index is None:
        device = torch.device(device.type, torch.cuda.current_device())
    key = str(device)
    buf = _CNT.get(key)
    if buf is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        buf = torch.zeros(_CNT_LEN, dtype=torch.int32, device=device)
        _CNT[key] = buf
    return buf


# Fused-launch tile overrides per token bucket (the CSV row's padded token count). Stage-1 / stage-2 tile and wave
# choices only move work between workgroups (no split-K), so they keep the results bitwise; checked by bench_moe.py.
# AITER_MOE_FUSED12_CFG=<json file> {"<bucket>": {"num_buffers": 2, "tile_n2": 128, ...}} replaces this table.
_FUSED12_CFG_DEFAULT: dict = {}


@functools.cache
def _fused12_cfg_table(path: str) -> dict:
    if not path:
        return _FUSED12_CFG_DEFAULT
    import json

    with open(path) as f:
        return {int(k): v for k, v in json.load(f).items()}


def fused12_overrides(padded_m: int) -> dict | None:
    """Overrides for this bucket; None if a table is given and does not list the bucket (-> keep two launches)."""
    path = os.environ.get("AITER_MOE_FUSED12_CFG", "")
    table = _fused12_cfg_table(path)
    if path and int(padded_m) not in table:
        return None
    return dict(table.get(int(padded_m), {}))


@functools.cache
def _launcher():
    from .kernels.moe_fused12_gfx1250_tdm import launch_moe_fused12_a8w4_tdm

    return fast_jit(launch_moe_fused12_a8w4_tdm)


def fused12_stage2_waves(m_warp, n_warp, tile_n2, m_warp2, n_warp2):
    """Stage-2 wave grid for the fused launch: one workgroup size for both stages. Stage 2 has no split-K, so its
    tile / wave grid does not change its results; keep m_warp2 and change n_warp2 to match stage 1's wave count."""
    w1 = int(m_warp) * int(n_warp)
    if int(m_warp2) * int(n_warp2) == w1:
        return int(m_warp2), int(n_warp2)
    if w1 % int(m_warp2):
        return None
    nw2 = w1 // int(m_warp2)
    if int(tile_n2) % (nw2 * 32):
        return None
    return int(m_warp2), nw2


def fused12_eligible(*, token_num, tile_m, tile_m2, cluster_m, cluster_n, cluster_m2, cluster_n2,
                     persistent_workers, persistent_workers2, n_tiles1, n_tiles2, device=None) -> bool:
    if not fused12_enabled() or int(token_num) > fused12_max_tokens() or int(tile_m) != int(tile_m2):
        return False
    if device is not None and fused12_counters(device) is None:
        return False
    if _select_cluster_n(n_tiles1, cluster_n) != 1 or _select_cluster_n(n_tiles2, cluster_n2) != 1:
        return False
    if _select_cluster_m(cluster_m, 1, 1) != 1 or _select_cluster_m(cluster_m2, 1, 0) != 1:
        return False
    for stage, pw in ((1, persistent_workers), (2, persistent_workers2)):
        v = int(os.environ.get(f"AITER_FLYDSL_PERSISTENT_WORKERS{stage}",
                               os.environ.get("AITER_FLYDSL_PERSISTENT_WORKERS", pw)))
        if v:
            return False
    return True


def flydsl_moe_fused12_a8w4(
    out2, a1, w1, a1_scale, w1_scale, a2, a2_scale, w2, w2_scale, m_tile_map, *,
    n_experts, contiguous_m, N1, K1, N2, K2,
    tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers,
    tile_n2, tile_k2, m_warp2, n_warp2, num_buffers2,
    out_is_f16=0, a_is_fp4=0, stage1_act=1, swiglu_limit=7.0, quant_wmma_rep=1,
    waves_per_tensor_tdm=-1, waves_per_tensor_tdm2=-1, next_stage_prefetch=0, tdm_as_in_prologue=0,
    tdm_b_th=0, lds_soa_load_interleave=0, rolled_drain=0, row_major_ascale=0, a_row_stride_bytes=0,
    a_scale_row_stride_bytes=0, a_row_map=None, situ_beta=1.0, situ_linear_beta=1.0,
    stream=None, sync_mode=None, max_routes=None,
):
    launch = _launcher()
    if stream is None:
        stream = torch.cuda.current_stream()
    if sync_mode is None:
        sync_mode = int(os.environ.get("AITER_MOE_FUSED12_SYNC", "1"))
    num_buffers = min(int(num_buffers), max(1, (K1 + tile_k - 1) // tile_k))
    num_buffers2 = min(int(num_buffers2), max(1, (K2 + tile_k2 - 1) // tile_k2))
    n_tiles1 = (N1 + tile_n - 1) // tile_n
    n_tiles2 = (N2 + tile_n2 - 1) // tile_n2
    assert ((contiguous_m + tile_m - 1) // tile_m + 1) * int(os.environ.get("AITER_MOE_FUSED12_CNT_STRIDE", "32")) < _CNT_LEN
    cnt = fused12_counters(out2.device)
    assert cnt is not None, "fused12: counters must be allocated before graph capture (fused12_eligible checks it)"
    # Live m-tiles <= routes (every live tile holds >= 1 route row): trim the CUDA-graph-safe worst-case grid.
    m_tiles_all = (contiguous_m + tile_m - 1) // tile_m
    m_tiles_grid = m_tiles_all
    if max_routes is not None and os.environ.get("AITER_MOE_FUSED12_TRIM", "1") == "1":
        m_tiles_grid = max(1, min(m_tiles_all, int(max_routes)))
    row_map = a_row_map if a_row_map is not None else out2
    launch(
        out2,
        ptr_arg(a1),
        ptr_arg(w1),
        a1_scale.view(torch.int32),
        w1_scale.view(torch.int32),
        a2,
        a2_scale.view(torch.uint8),
        a2_scale.view(torch.int32),
        ptr_arg(w2),
        w2_scale.view(torch.int32),
        ptr_arg(m_tile_map),
        ptr_arg(cnt),
        row_map,
        int(contiguous_m),
        int(m_tiles_grid * tile_m),
        stream,
        int(N1),
        int(N2),
        int(K1),
        int(K2),
        n_tiles1,
        n_tiles2,
        int(tile_m), int(tile_n), int(tile_k), int(m_warp), int(n_warp), num_buffers,
        int(tile_n2), int(tile_k2), int(m_warp2), int(n_warp2), num_buffers2,
        int(out_is_f16), int(a_is_fp4), int(n_experts), int(stage1_act), float(swiglu_limit), int(quant_wmma_rep),
        _select_num_waves_per_tensor_tdm(waves_per_tensor_tdm),
        _select_num_waves_per_tensor_tdm(waves_per_tensor_tdm2),
        next_stage_prefetch=_select_bool_env("AITER_TDM_NEXT_STAGE_PREFETCH", next_stage_prefetch),
        tdm_as_in_prologue=_select_bool_env("AITER_GROUPED_GEMM_AS_PROLOGUE", tdm_as_in_prologue),
        tdm_b_th=_select_tdm_b_th(tdm_b_th),
        lds_soa_load_interleave=int(lds_soa_load_interleave),
        rolled_drain=int(rolled_drain),
        row_major_ascale=int(row_major_ascale),
        a_row_stride_bytes=int(a_row_stride_bytes),
        a_scale_row_stride_bytes=int(a_scale_row_stride_bytes),
        gather_a=int(a_row_map is not None),
        scale_prologue=(_gather_scale_prologue() if a_row_map is not None else 0),
        f32_situ_beta=float(situ_beta),
        f32_situ_linear_beta=float(situ_linear_beta),
        sync_mode=int(sync_mode),
        # the kernel module reads these at import; pass them so FlyDSL's cache key sees them
        variant=",".join(
            f"{k}={os.environ.get('AITER_MOE_FUSED12_' + k, '')}"
            for k in ("TRACE", "BFIRST", "PREFETCH_B2", "SPIN_SLEEP", "SPIN_MAX", "POLL_ONE_WAVE", "POLL_L2", "CNT_STRIDE", "FENCE")
        ),
    )
    return out2
