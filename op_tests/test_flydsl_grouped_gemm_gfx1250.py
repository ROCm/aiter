#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 masked grouped MoE GEMM tests for FlyDSL-backed aiter wrappers.

Pytest runs a small correctness suite.  Direct execution provides a lightweight
shape runner for DeepSeek-style grouped MoE GEMMs, e.g.::

    python op_tests/test_flydsl_grouped_gemm_gfx1250.py \
      --stage both --model-dim 7168 --inter-dim 256 --experts 256 --max-m 32
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pytest
import torch

_LOCAL_DEPS = ("/root/data/aiter", "/root/data/triton/python")
for _dep in reversed(_LOCAL_DEPS):
    if os.path.exists(_dep) and _dep not in sys.path:
        sys.path.insert(0, _dep)

from aiter.ops.flydsl.kernels.moe_grouped_gemm_mxscale_gfx1250 import (  # noqa: E402
    compile_moe_grouped_gemm1_mxfp4_masked,
    compile_moe_grouped_gemm2_mxfp4_masked,
)
from aiter.test_common import run_perftest  # noqa: E402
from aiter.utility import dtypes  # noqa: E402
from flydsl.runtime.device import get_rocm_arch  # noqa: E402

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]
SCALE_BLOCK = 32
DEFAULT_SCALE_U8 = 120
_START_TIME = time.perf_counter()


def _log(message: str) -> None:
    print(f"[masked-grouped-moe-gemm][{time.perf_counter() - _START_TIME:.2f}s] {message}", flush=True)


def _require_gfx1250() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm not available")
    arch = str(get_rocm_arch())
    if not arch.startswith("gfx1250"):
        pytest.skip(f"FlyDSL masked grouped MoE GEMM requires gfx1250, got {arch}")


def _shape(
    *,
    experts: int = 3,
    max_m: int = 32,
    model_dim: int = 256,
    inter_dim: int = 256,
    tile_m: int = 16,
    tile_n: int = 64,
    tile_k: int = 128,
    m_warp: int = 1,
    n_warp: int = 2,
    num_buffers: int = 2,
    waves_per_eu: int | None = None,
    inst_prefetch: bool = False,
    use_tdm_store: bool = True,
    use_scale_opsel: bool = False,
    wave_specialized_tdm: bool = False,
    expert_sched_mode: bool = False,
    persistent_workers: int | None = None,
    cluster_m: int = 1,
    cluster_n: int = 1,
) -> dict:
    return dict(
        experts=experts,
        max_m=max_m,
        model_dim=model_dim,
        inter_dim=inter_dim,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=m_warp,
        n_warp=n_warp,
        num_buffers=num_buffers,
        waves_per_eu=waves_per_eu,
        inst_prefetch=inst_prefetch,
        use_tdm_store=use_tdm_store,
        use_scale_opsel=use_scale_opsel,
        wave_specialized_tdm=wave_specialized_tdm,
        expert_sched_mode=expert_sched_mode,
        persistent_workers=persistent_workers,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
    )


_KERNEL_OPTION_KEYS = (
    "tile_m", "tile_n", "tile_k", "m_warp", "n_warp",
    "num_buffers", "waves_per_eu", "inst_prefetch", "use_tdm_store",
    "use_scale_opsel", "wave_specialized_tdm", "expert_sched_mode",
    "persistent_workers", "cluster_m", "cluster_n",
)


def _kernel_kwargs(s: dict) -> dict:
    """Extract kernel-tuning kwargs to forward to compile_moe_grouped_gemm*."""
    return {k: s[k] for k in _KERNEL_OPTION_KEYS if k in s}


def _masked_m(experts: int, max_m: int, *, mode: str = "mixed", override: int | None = None) -> torch.Tensor:
    if override is not None:
        if override < 0 or override > max_m:
            raise ValueError(f"masked_m override={override} must be in [0, max_m={max_m}]")
        vals = [int(override)] * experts
    elif mode == "full":
        vals = [max_m] * experts
    elif mode == "descending":
        vals = [max(0, max_m - (idx % 4) * max(1, max_m // 4)) for idx in range(experts)]
    else:
        base = [max(1, max_m // 2 + 1), max_m, 0, max(1, max_m // 4)]
        vals = [min(max_m, base[idx % len(base)]) for idx in range(experts)]
    return torch.tensor(vals, dtype=torch.int32, device="cuda")


def _pack_uint4(unpacked: torch.Tensor) -> torch.Tensor:
    shape = unpacked.shape
    assert shape[-1] % 2 == 0
    flat = unpacked.contiguous().view(-1)
    return ((flat[1::2] << 4) | flat[::2]).view(*shape[:-1], shape[-1] // 2)


def _pattern_mxfp4(rows: int, cols: int, *, seed: int) -> torch.Tensor:
    vals = torch.tensor([0, 1, 8, 9], dtype=torch.uint8)
    idx = (torch.arange(rows * cols, dtype=torch.long) + seed) % vals.numel()
    return _pack_uint4(vals[idx].view(rows, cols))


def _constant_mxfp4(rows: int, cols: int, byte: int = 0x11) -> torch.Tensor:
    assert cols % 2 == 0
    return torch.full((rows, cols // 2), byte, dtype=torch.uint8)


def _mxfp4_batch(experts: int, rows: int, cols: int, *, mode: str, seed: int) -> torch.Tensor:
    if mode == "pattern":
        # Vectorized: build the (rows*cols) index pattern once on GPU, broadcast
        # to (experts, rows*cols) with per-expert seed offset, then pack. This
        # replaces a 256-step CPU loop (which dominated build-inputs latency).
        device = "cuda" if torch.cuda.is_available() else "cpu"
        vals = torch.tensor([0, 1, 8, 9], dtype=torch.uint8, device=device)
        base = torch.arange(rows * cols, dtype=torch.long, device=device)  # (R*C,)
        seeds = torch.arange(experts, dtype=torch.long, device=device) + seed
        idx = (base.view(1, -1) + seeds.view(-1, 1)) % vals.numel()       # (E, R*C)
        u4 = vals[idx].view(experts, rows, cols)                           # (E, R, C)  uint8 with values in {0,1,8,9}
        # Pack two nibbles per byte along the last dim.
        if cols % 2 != 0:
            raise ValueError(f"cols must be even, got {cols}")
        low = u4[..., 0::2] & 0x0F
        high = (u4[..., 1::2] & 0x0F) << 4
        packed = (low | high).contiguous()
        return packed.cpu()
    assert cols % 2 == 0
    return torch.full((experts, rows, cols // 2), 0x11, dtype=torch.uint8)


def _mock_topk(tokens: int, topk: int, experts: int, *, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    token_idx = torch.arange(tokens, dtype=torch.int64).view(tokens, 1)
    rank_idx = torch.arange(topk, dtype=torch.int64).view(1, topk)
    if mode == "hot":
        topk_ids = torch.remainder(rank_idx, experts).expand(tokens, topk).clone()
    elif mode == "stride":
        topk_ids = torch.remainder(token_idx * max(1, topk) + rank_idx * 3, experts)
    elif mode == "expert_balance":
        topk_ids = torch.remainder(token_idx * topk + rank_idx, experts)
    else:
        topk_ids = torch.remainder(token_idx + rank_idx, experts)
    topk_weights = torch.full((tokens, topk), 1.0 / max(1, topk), dtype=torch.float32)
    return topk_ids.to(torch.int32), topk_weights


def _mock_group_slots(
    topk_ids: torch.Tensor,
    *,
    experts: int,
    max_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens, topk = topk_ids.shape
    counts = torch.zeros((experts,), dtype=torch.int32)
    slot_token_ids = torch.full((experts, max_m), -1, dtype=torch.int32)
    slot_rank_ids = torch.full((experts, max_m), -1, dtype=torch.int32)
    for token in range(tokens):
        for rank in range(topk):
            expert = int(topk_ids[token, rank].item())
            row = int(counts[expert].item())
            if row >= max_m:
                raise ValueError(
                    f"mock route overflow: expert={expert} needs > max_m={max_m}. "
                    "Increase --max-m or reduce --tokens/--topk."
                )
            slot_token_ids[expert, row] = token
            slot_rank_ids[expert, row] = rank
            counts[expert] += 1
    return counts, slot_token_ids, slot_rank_ids


def _mock_grouped_mxfp4(
    slot_token_ids: torch.Tensor,
    slot_rank_ids: torch.Tensor,
    cols: int,
    *,
    mode: str,
    seed: int,
) -> torch.Tensor:
    experts, max_m = slot_token_ids.shape
    if mode == "constant":
        return torch.full((experts, max_m, cols // 2), 0x11, dtype=torch.uint8)
    if cols % 2 != 0:
        raise ValueError(f"cols must be even, got {cols}")

    # Vectorized version: build per-(expert,row) seed and pattern indices on GPU
    # in one shot rather than 256x32=8192 host-side .item()/_pattern_mxfp4 calls.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vals = torch.tensor([0, 1, 8, 9], dtype=torch.uint8, device=device)
    slot_tok = slot_token_ids.to(device, dtype=torch.long)               # (E, max_m)
    slot_rnk = slot_rank_ids.to(device, dtype=torch.long)
    expert_idx = torch.arange(experts, dtype=torch.long, device=device).view(experts, 1)
    valid = slot_tok >= 0                                                # (E, max_m) bool
    base_seed = (
        seed
        + slot_tok.clamp(min=0) * 17
        + slot_rnk.clamp(min=0) * 3
        + expert_idx
    )                                                                     # (E, max_m)
    base_idx = torch.arange(cols, dtype=torch.long, device=device).view(1, 1, cols)  # (1, 1, cols)
    idx = (base_idx + base_seed.unsqueeze(-1)) % vals.numel()             # (E, max_m, cols)
    u4 = vals[idx]                                                        # (E, max_m, cols) uint8 in {0,1,8,9}
    low = u4[..., 0::2] & 0x0F
    high = (u4[..., 1::2] & 0x0F) << 4
    packed = (low | high)                                                  # (E, max_m, cols//2)
    # Zero out rows where slot_token_id < 0 (no token assigned).
    packed = packed * valid.unsqueeze(-1)
    return packed.contiguous().cpu()


def _mxfp4_to_f32(x: torch.Tensor) -> torch.Tensor:
    x = x.view(torch.uint8)
    x = x.repeat_interleave(2, dim=-1)
    x[..., ::2] = x[..., ::2] & 0xF
    x[..., 1::2] = x[..., 1::2] >> 4
    values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=x.device,
    )
    return values[x.long()]


def _e8m0_to_f32(scale: torch.Tensor) -> torch.Tensor:
    scale_u8 = scale.view(torch.uint8)
    scale_i32 = scale_u8.to(torch.int32) << 23
    scale_i32 = torch.where(scale_u8 == 0, torch.full_like(scale_i32, 0x00400000), scale_i32)
    scale_i32 = torch.where(scale_u8 == 0xFF, torch.full_like(scale_i32, 0x7F800001), scale_i32)
    return scale_i32.view(torch.float32)


def _raw_scale(rows: int, k_dim: int, value: int = DEFAULT_SCALE_U8) -> torch.Tensor:
    return torch.full((rows, k_dim // SCALE_BLOCK), value, dtype=torch.uint8)


def _preshuffle_e8m0_scale(scale: torch.Tensor, *, warp_tile: int, tile_k: int, wmma_dim: int = 16) -> torch.Tensor:
    _, k_scale = scale.shape
    scales_per_wmma = 4
    scale_k_per_tile = tile_k // SCALE_BLOCK
    assert k_scale % scale_k_per_tile == 0
    assert scale_k_per_tile % scales_per_wmma == 0
    wmma_rep = warp_tile // wmma_dim
    k_groups = k_scale // scale_k_per_tile
    k_wmma_steps = scale_k_per_tile // scales_per_wmma
    g = scale.view(-1, wmma_rep, wmma_dim, k_groups, k_wmma_steps, scales_per_wmma)
    g = g.permute(0, 2, 3, 4, 1, 5).contiguous()
    return g.reshape(-1, k_groups * k_wmma_steps * wmma_rep * scales_per_wmma)


def _prep_scale_batch(scales: torch.Tensor, *, warp_tile: int, tile_k: int) -> torch.Tensor:
    return torch.stack([
        _preshuffle_e8m0_scale(scales[e], warp_tile=warp_tile, tile_k=tile_k)
        for e in range(scales.shape[0])
    ]).cuda()


def _reference_mxfp4(a_fp4: torch.Tensor, b_fp4: torch.Tensor, a_scale: torch.Tensor, b_scale: torch.Tensor, m: int, n: int, k: int) -> torch.Tensor:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a_fp4 = a_fp4.to(device, non_blocking=True)
    b_fp4 = b_fp4.to(device, non_blocking=True)
    a_scale = a_scale.to(device, non_blocking=True)
    b_scale = b_scale.to(device, non_blocking=True)
    a_f32 = _mxfp4_to_f32(a_fp4.view(torch.uint8))[:m, :k]
    b_f32 = _mxfp4_to_f32(b_fp4.view(torch.uint8))[:n, :k]
    a_sc = _e8m0_to_f32(a_scale.view(torch.uint8)).repeat_interleave(SCALE_BLOCK, dim=-1)[:m, :k]
    b_sc = _e8m0_to_f32(b_scale.view(torch.uint8)).repeat_interleave(SCALE_BLOCK, dim=-1)[:n, :k]
    return torch.matmul(a_f32 * a_sc, (b_f32 * b_sc).T)


def _reference_mxfp4_batched(
    a_fp4: torch.Tensor,        # (B, M_pad, K_pack)  uint8
    b_fp4: torch.Tensor,        # (B, N,     K_pack)  uint8
    a_scale: torch.Tensor,      # (B, M_pad, K/32)    uint8
    b_scale: torch.Tensor,      # (B, N,     K/32)    uint8
    *,
    k: int,
) -> torch.Tensor:
    """Batched FP4 GEMM reference -- single GPU call per intermediate, no host loop.

    Returns (B, M_pad, N) float32. Caller must select rows [:valid] per expert.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a_fp4 = a_fp4.to(device, non_blocking=True).contiguous()
    b_fp4 = b_fp4.to(device, non_blocking=True).contiguous()
    a_scale = a_scale.to(device, non_blocking=True).contiguous()
    b_scale = b_scale.to(device, non_blocking=True).contiguous()
    a_f32 = _mxfp4_to_f32(a_fp4.view(torch.uint8))[..., :k]
    b_f32 = _mxfp4_to_f32(b_fp4.view(torch.uint8))[..., :k]
    a_sc = _e8m0_to_f32(a_scale.view(torch.uint8)).repeat_interleave(SCALE_BLOCK, dim=-1)[..., :k]
    b_sc = _e8m0_to_f32(b_scale.view(torch.uint8)).repeat_interleave(SCALE_BLOCK, dim=-1)[..., :k]
    a_scaled = a_f32 * a_sc                                    # (B, M, K)
    b_scaled = (b_f32 * b_sc).transpose(-1, -2).contiguous()    # (B, K, N)
    return torch.matmul(a_scaled, b_scaled)                     # (B, M, N)


def _assert_valid_rows_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual = actual.float()
    expected = expected.to(actual.device).float()
    diff = (actual - expected).abs()
    print(f"[masked-grouped-moe-gemm] {name}: shape={tuple(actual.shape)} max_abs={float(diff.max()):.8e} mean_abs={float(diff.mean()):.8e}", flush=True)
    torch.testing.assert_close(actual, expected, rtol=0.25, atol=0.25)


def _assert_batched_close(
    name: str,
    actual: torch.Tensor,        # (B, M_pad, N)
    expected: torch.Tensor,      # (B, M_pad, N)
    masked_m_cpu: torch.Tensor,  # (B,) int
) -> None:
    """Compare only the [:valid_m[e]] rows per expert in a single GPU op.

    Builds a row mask of shape (B, M_pad, 1) on device, broadcasts, and asserts
    once. Pad rows are forced to match (0 == 0) so they don't trigger failures.
    """
    actual = actual.float()
    expected = expected.to(actual.device).float()
    B, M_pad, _ = actual.shape
    row_idx = torch.arange(M_pad, device=actual.device).view(1, M_pad)
    masked_m_dev = masked_m_cpu.to(actual.device, dtype=torch.int32).view(B, 1)
    row_mask = (row_idx < masked_m_dev).unsqueeze(-1).float()  # (B, M_pad, 1)
    actual_masked = actual * row_mask
    expected_masked = expected * row_mask
    diff = (actual_masked - expected_masked).abs()
    n_valid = float(row_mask.sum()) * actual.shape[-1]
    max_abs = float(diff.max())
    mean_abs = float(diff.sum() / max(n_valid, 1.0))
    print(
        f"[masked-grouped-moe-gemm] {name} batched: shape={tuple(actual.shape)} "
        f"active_experts={int((masked_m_cpu > 0).sum().item())} "
        f"max_abs={max_abs:.8e} mean_abs={mean_abs:.8e}",
        flush=True,
    )
    torch.testing.assert_close(actual_masked, expected_masked, rtol=0.25, atol=0.25)


def _run_stage1(s: dict[str, int], *, persistent: bool, verify: bool, data: str, masked_mode: str, warmup: int, iters: int, masked_m_override: int | None = None) -> float:
    E, max_m = s["experts"], s["max_m"]
    model_dim, inter_dim = s["model_dim"], s["inter_dim"]
    _log(f"stage1 persistent={persistent}: build inputs")
    masked_m = _masked_m(E, max_m, mode=masked_mode, override=masked_m_override)
    x_raw = _mxfp4_batch(E, max_m, model_dim, mode=data, seed=0)
    w_raw = _mxfp4_batch(E, 2 * inter_dim, model_dim, mode=data, seed=17)
    x_scale_raw = torch.full((E, max_m, model_dim // SCALE_BLOCK), DEFAULT_SCALE_U8, dtype=torch.uint8)
    w_scale_raw = torch.full((E, 2 * inter_dim, model_dim // SCALE_BLOCK), DEFAULT_SCALE_U8, dtype=torch.uint8)
    _log(f"stage1 persistent={persistent}: preshuffle scales")
    x_scale = _prep_scale_batch(x_scale_raw, warp_tile=s["tile_m"] // s["m_warp"], tile_k=s["tile_k"])
    w_scale = _prep_scale_batch(w_scale_raw, warp_tile=s["tile_n"] // s["n_warp"], tile_k=s["tile_k"])
    y = torch.empty((E, max_m, inter_dim), device="cuda", dtype=torch.float16)
    _log(f"stage1 persistent={persistent}: compile")
    compile_start = time.perf_counter()
    kernel = compile_moe_grouped_gemm1_mxfp4_masked(
        model_dim=model_dim, inter_dim=inter_dim, experts=E, max_m=max_m,
        out_dtype="f16", grouped_persistent_m=persistent,
        **_kernel_kwargs(s),
    )
    _log(f"stage1 persistent={persistent}: compile done in {time.perf_counter() - compile_start:.2f}s")
    x = x_raw.cuda()
    w = w_raw.cuda()

    def launch():
        kernel(y, x, w, x_scale, w_scale, masked_m, max_m, inter_dim, model_dim, E, stream=torch.cuda.current_stream())

    _log(f"stage1 persistent={persistent}: launch warmup={warmup} iters={iters}")
    _, us = run_perftest(launch, num_warmup=warmup, num_iters=iters, testGraph=False)
    _log(f"stage1 persistent={persistent}: launch done us={us:.2f}")
    if verify:
        masked_m_cpu_full = masked_m.cpu()
        active_idx = torch.nonzero(masked_m_cpu_full > 0, as_tuple=False).flatten().tolist()
        if active_idx:
            active_t = torch.tensor(active_idx, dtype=torch.long)
            gate_up_all = _reference_mxfp4_batched(
                x_raw[active_t], w_raw[active_t],
                x_scale_raw[active_t], w_scale_raw[active_t],
                k=model_dim,
            )  # (A, max_m, 2*inter_dim)
            gate = gate_up_all[..., :inter_dim]
            up = gate_up_all[..., inter_dim:]
            expected_active = (torch.nn.functional.silu(gate) * up).to(torch.float16).float()
            actual_active = y[active_t].float()
            _assert_batched_close(
                f"stage1 persistent={persistent}",
                actual_active, expected_active,
                masked_m_cpu_full[active_t],
            )
    return us


def _run_stage2(s: dict[str, int], *, persistent: bool, verify: bool, data: str, masked_mode: str, warmup: int, iters: int, masked_m_override: int | None = None) -> float:
    E, max_m = s["experts"], s["max_m"]
    model_dim, inter_dim = s["model_dim"], s["inter_dim"]
    _log(f"stage2 persistent={persistent}: build inputs")
    masked_m = _masked_m(E, max_m, mode=masked_mode, override=masked_m_override)
    x_raw = _mxfp4_batch(E, max_m, inter_dim, mode=data, seed=31)
    w_raw = _mxfp4_batch(E, model_dim, inter_dim, mode=data, seed=47)
    x_scale_raw = torch.full((E, max_m, inter_dim // SCALE_BLOCK), DEFAULT_SCALE_U8, dtype=torch.uint8)
    w_scale_raw = torch.full((E, model_dim, inter_dim // SCALE_BLOCK), DEFAULT_SCALE_U8, dtype=torch.uint8)
    _log(f"stage2 persistent={persistent}: preshuffle scales")
    x_scale = _prep_scale_batch(x_scale_raw, warp_tile=s["tile_m"] // s["m_warp"], tile_k=s["tile_k"])
    w_scale = _prep_scale_batch(w_scale_raw, warp_tile=s["tile_n"] // s["n_warp"], tile_k=s["tile_k"])
    y = torch.empty((E, max_m, model_dim), device="cuda", dtype=torch.float16)
    _log(f"stage2 persistent={persistent}: compile")
    compile_start = time.perf_counter()
    kernel = compile_moe_grouped_gemm2_mxfp4_masked(
        model_dim=model_dim, inter_dim=inter_dim, experts=E, max_m=max_m,
        out_dtype="f16", grouped_persistent_m=persistent,
        **_kernel_kwargs(s),
    )
    _log(f"stage2 persistent={persistent}: compile done in {time.perf_counter() - compile_start:.2f}s")
    x = x_raw.cuda()
    w = w_raw.cuda()

    def launch():
        kernel(y, x, w, x_scale, w_scale, masked_m, max_m, model_dim, inter_dim, E, stream=torch.cuda.current_stream())

    _log(f"stage2 persistent={persistent}: launch warmup={warmup} iters={iters}")
    _, us = run_perftest(launch, num_warmup=warmup, num_iters=iters, testGraph=False)
    _log(f"stage2 persistent={persistent}: launch done us={us:.2f}")
    if verify:
        masked_m_cpu_full = masked_m.cpu()
        active_idx = torch.nonzero(masked_m_cpu_full > 0, as_tuple=False).flatten().tolist()
        if active_idx:
            active_t = torch.tensor(active_idx, dtype=torch.long)
            expected_active = _reference_mxfp4_batched(
                x_raw[active_t], w_raw[active_t],
                x_scale_raw[active_t], w_scale_raw[active_t],
                k=inter_dim,
            ).to(torch.float16).float()
            actual_active = y[active_t].float()
            _assert_batched_close(
                f"stage2 persistent={persistent}",
                actual_active, expected_active,
                masked_m_cpu_full[active_t],
            )
    return us


def _run_mock_moe(
    s: dict[str, int],
    *,
    tokens: int,
    topk: int,
    route_mode: str,
    persistent: bool,
    verify: bool,
    data: str,
    warmup: int,
    iters: int,
) -> tuple[float, float]:
    E, max_m = s["experts"], s["max_m"]
    model_dim, inter_dim = s["model_dim"], s["inter_dim"]
    topk_ids, topk_weights = _mock_topk(tokens, topk, E, mode=route_mode)
    masked_m_cpu, slot_token_ids, slot_rank_ids = _mock_group_slots(topk_ids, experts=E, max_m=max_m)
    masked_m = masked_m_cpu.cuda()
    _log(
        "mock moe route: "
        f"tokens={tokens} topk={topk} mode={route_mode} "
        f"max_expert_m={int(masked_m_cpu.max().item())} non_empty={int((masked_m_cpu > 0).sum().item())}/{E}"
    )

    _log(f"mock moe stage1 persistent={persistent}: build inputs")
    x1_raw = _mock_grouped_mxfp4(slot_token_ids, slot_rank_ids, model_dim, mode=data, seed=0)
    w1_raw = _mxfp4_batch(E, 2 * inter_dim, model_dim, mode=data, seed=17)
    x1_scale_raw = torch.full((E, max_m, model_dim // SCALE_BLOCK), DEFAULT_SCALE_U8, dtype=torch.uint8)
    w1_scale_raw = torch.full((E, 2 * inter_dim, model_dim // SCALE_BLOCK), DEFAULT_SCALE_U8, dtype=torch.uint8)
    x1_scale = _prep_scale_batch(x1_scale_raw, warp_tile=s["tile_m"] // s["m_warp"], tile_k=s["tile_k"])
    w1_scale = _prep_scale_batch(w1_scale_raw, warp_tile=s["tile_n"] // s["n_warp"], tile_k=s["tile_k"])
    y1 = torch.empty((E, max_m, inter_dim), device="cuda", dtype=torch.float16)
    _log(f"mock moe stage1 persistent={persistent}: compile")
    compile_start = time.perf_counter()
    k1 = compile_moe_grouped_gemm1_mxfp4_masked(
        model_dim=model_dim, inter_dim=inter_dim, experts=E, max_m=max_m,
        out_dtype="f16", grouped_persistent_m=persistent,
        **_kernel_kwargs(s),
    )
    _log(f"mock moe stage1 persistent={persistent}: compile done in {time.perf_counter() - compile_start:.2f}s")
    x1 = x1_raw.cuda()
    w1 = w1_raw.cuda()

    def launch_stage1():
        k1(y1, x1, w1, x1_scale, w1_scale, masked_m, max_m, inter_dim, model_dim, E, stream=torch.cuda.current_stream())

    _log(f"mock moe stage1 persistent={persistent}: launch warmup={warmup} iters={iters}")
    _, us1 = run_perftest(launch_stage1, num_warmup=warmup, num_iters=iters, testGraph=False)
    _log(f"mock moe stage1 persistent={persistent}: launch done us={us1:.2f}")
    if verify:
        active_idx = torch.nonzero(masked_m_cpu > 0, as_tuple=False).flatten().tolist()
        if active_idx:
            active_t = torch.tensor(active_idx, dtype=torch.long)
            gate_up_all = _reference_mxfp4_batched(
                x1_raw[active_t], w1_raw[active_t],
                x1_scale_raw[active_t], w1_scale_raw[active_t],
                k=model_dim,
            )
            gate = gate_up_all[..., :inter_dim]
            up = gate_up_all[..., inter_dim:]
            expected_active = (torch.nn.functional.silu(gate) * up).to(torch.float16).float()
            actual_active = y1[active_t].float()
            _assert_batched_close(
                f"mock stage1 persistent={persistent}",
                actual_active, expected_active,
                masked_m_cpu[active_t],
            )

    # Future integration point: replace this mocked quantized intermediate with
    # quantization of y1 once the end-to-end grouped MoE path is wired in.
    _log(f"mock moe stage2 persistent={persistent}: build inputs")
    x2_raw = _mock_grouped_mxfp4(slot_token_ids, slot_rank_ids, inter_dim, mode=data, seed=31)
    w2_raw = _mxfp4_batch(E, model_dim, inter_dim, mode=data, seed=47)
    x2_scale_raw = torch.full((E, max_m, inter_dim // SCALE_BLOCK), DEFAULT_SCALE_U8, dtype=torch.uint8)
    w2_scale_raw = torch.full((E, model_dim, inter_dim // SCALE_BLOCK), DEFAULT_SCALE_U8, dtype=torch.uint8)
    x2_scale = _prep_scale_batch(x2_scale_raw, warp_tile=s["tile_m"] // s["m_warp"], tile_k=s["tile_k"])
    w2_scale = _prep_scale_batch(w2_scale_raw, warp_tile=s["tile_n"] // s["n_warp"], tile_k=s["tile_k"])
    y2 = torch.empty((E, max_m, model_dim), device="cuda", dtype=torch.float16)
    _log(f"mock moe stage2 persistent={persistent}: compile")
    compile_start = time.perf_counter()
    k2 = compile_moe_grouped_gemm2_mxfp4_masked(
        model_dim=model_dim, inter_dim=inter_dim, experts=E, max_m=max_m,
        out_dtype="f16", grouped_persistent_m=persistent,
        **_kernel_kwargs(s),
    )
    _log(f"mock moe stage2 persistent={persistent}: compile done in {time.perf_counter() - compile_start:.2f}s")
    x2 = x2_raw.cuda()
    w2 = w2_raw.cuda()

    def launch_stage2():
        k2(y2, x2, w2, x2_scale, w2_scale, masked_m, max_m, model_dim, inter_dim, E, stream=torch.cuda.current_stream())

    _log(f"mock moe stage2 persistent={persistent}: launch warmup={warmup} iters={iters}")
    _, us2 = run_perftest(launch_stage2, num_warmup=warmup, num_iters=iters, testGraph=False)
    _log(f"mock moe stage2 persistent={persistent}: launch done us={us2:.2f}")
    if verify:
        expected_topk_out = torch.zeros((tokens, topk, model_dim), device="cuda", dtype=torch.float32)
        active_idx = torch.nonzero(masked_m_cpu > 0, as_tuple=False).flatten().tolist()
        if active_idx:
            active_t = torch.tensor(active_idx, dtype=torch.long)
            expected_all = _reference_mxfp4_batched(
                x2_raw[active_t], w2_raw[active_t],
                x2_scale_raw[active_t], w2_scale_raw[active_t],
                k=inter_dim,
            ).to(torch.float16).float()  # (A, max_m, model_dim)
            actual_active = y2[active_t].float()
            _assert_batched_close(
                f"mock stage2 persistent={persistent}",
                actual_active, expected_all,
                masked_m_cpu[active_t],
            )
            # Scatter expected rows into (tokens, topk, model_dim). Active loop
            # bound is small (= number of routed experts), so a per-row Python
            # loop here is cheap compared to the 256-expert version.
            for ai, e in enumerate(active_idx):
                valid = int(masked_m_cpu[e].item())
                for row in range(valid):
                    token = int(slot_token_ids[e, row].item())
                    rank = int(slot_rank_ids[e, row].item())
                    expected_topk_out[token, rank].copy_(expected_all[ai, row])

    # Vectorized scatter: gather rows that have a valid token/rank assignment
    # into (tokens, topk, model_dim) without iterating experts on the host.
    topk_out = torch.zeros((tokens, topk, model_dim), device="cuda", dtype=torch.float16)
    row_idx = torch.arange(max_m, device="cuda").view(1, max_m)
    valid_mask = (row_idx < masked_m.view(E, 1))                      # (E, max_m)  bool, GPU
    flat_mask = valid_mask.view(-1)                                    # (E*max_m,)
    if flat_mask.any():
        flat_tokens = slot_token_ids.to("cuda").view(-1)               # (E*max_m,)
        flat_ranks = slot_rank_ids.to("cuda").view(-1)
        flat_y = y2.view(E * max_m, model_dim)                         # (E*max_m, model_dim)
        sel = torch.nonzero(flat_mask, as_tuple=False).flatten()       # (R,) indices
        topk_out[flat_tokens[sel], flat_ranks[sel]] = flat_y[sel]
    out = (topk_out.float() * topk_weights.cuda().unsqueeze(-1)).sum(dim=1).to(torch.float16)
    if verify:
        expected_out = (
            expected_topk_out * topk_weights.cuda().unsqueeze(-1)
        ).sum(dim=1).to(torch.float16).float()
        _assert_valid_rows_close(f"mock moe final persistent={persistent}", out, expected_out)
    _log(f"mock moe scatter done: topk_out={tuple(topk_out.shape)} out={tuple(out.shape)}")
    return us1, us2


@pytest.mark.parametrize("persistent", [False, True])
def test_masked_grouped_moe_gemm_stage1_mxfp4_silu(persistent: bool):
    _require_gfx1250()
    _run_stage1(_shape(), persistent=persistent, verify=True, data="pattern", masked_mode="mixed", warmup=0, iters=1)


@pytest.mark.parametrize("persistent", [False, True])
def test_masked_grouped_moe_gemm_stage2_mxfp4(persistent: bool):
    _require_gfx1250()
    _run_stage2(_shape(), persistent=persistent, verify=True, data="pattern", masked_mode="mixed", warmup=0, iters=1)


def test_mock_moe_usage_mxfp4():
    _require_gfx1250()
    _run_mock_moe(
        _shape(experts=3, max_m=16),
        tokens=8,
        topk=2,
        route_mode="balanced",
        persistent=True,
        verify=True,
        data="pattern",
        warmup=0,
        iters=1,
    )


def test_mock_moe_expert_balance_mxfp4():
    _require_gfx1250()
    _run_mock_moe(
        _shape(experts=8, max_m=16),
        tokens=16,
        topk=4,
        route_mode="expert_balance",
        persistent=True,
        verify=True,
        data="pattern",
        warmup=0,
        iters=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("grouped", "moe"), default="grouped")
    parser.add_argument("--stage", choices=("1", "2", "both"), default="both")
    parser.add_argument("--model-dim", type=int, default=7168)
    parser.add_argument("--inter-dim", type=int, default=256)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--max-m", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--route-mode", choices=("balanced", "stride", "hot", "expert_balance"), default="balanced")
    parser.add_argument("--tile-m", type=int, default=16)
    parser.add_argument("--tile-n", type=int, default=64)
    parser.add_argument("--tile-k", type=int, default=128)
    parser.add_argument("--m-warp", type=int, default=1)
    parser.add_argument("--n-warp", type=int, default=2)
    parser.add_argument("--num-buffers", type=int, default=2,
                        help="K-tile pipeline depth. Must satisfy K/tile_k >= num_buffers.")
    parser.add_argument("--waves-per-eu", type=int, default=None,
                        help="Override waves-per-EU (default: kernel decides).")
    parser.add_argument("--persistent-workers", type=int, default=None,
                        help="Override persistent grid size (default: get_cu_num()).")
    parser.add_argument("--cluster-m", type=int, default=1,
                        help="Workgroup cluster M (mutually exclusive with --persistent).")
    parser.add_argument("--cluster-n", type=int, default=1,
                        help="Workgroup cluster N (mutually exclusive with --persistent).")
    parser.add_argument("--inst-prefetch", action=argparse.BooleanOptionalAction, default=False,
                        help="Inject s_prefetch_inst_pc_rel hints (--inst-prefetch / --no-inst-prefetch).")
    parser.add_argument("--use-tdm-store", action=argparse.BooleanOptionalAction, default=True,
                        help="Use LDS+TDM tensor_store_2d for C writeback.")
    parser.add_argument("--use-scale-opsel", action=argparse.BooleanOptionalAction, default=False,
                        help="Use WMMA scale op_sel to reduce scale loads.")
    parser.add_argument("--wave-specialized-tdm", action=argparse.BooleanOptionalAction, default=False,
                        help="Dedicate one warp to TDM issue (requires 4-warp configs).")
    parser.add_argument("--expert-sched-mode", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable LLVM amdgpu-expert-scheduling-mode.")
    parser.add_argument("--persistent", action="store_true", help="Use persistent grouped scheduling.")
    parser.add_argument("--non-persistent", action="store_true", help="Also run the non-persistent variant.")
    parser.add_argument("--verify", action="store_true", help="Run torch reference checks. Expensive for DeepSeek shapes.")
    parser.add_argument("--data", choices=("constant", "pattern"), default="constant")
    parser.add_argument("--masked-mode", choices=("mixed", "full", "descending"), default="mixed")
    parser.add_argument("--masked-m-override", type=int, default=None,
                        help="Force every expert's valid_m to this value (0..max_m). "
                             "Useful for decode (M=1) cases where buffer max_m must be >= tile_m.")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=101)
    args = parser.parse_args()

    try:
        _require_gfx1250()
    except pytest.skip.Exception as exc:
        raise SystemExit(str(exc)) from None
    s = _shape(
        experts=args.experts,
        max_m=args.max_m,
        model_dim=args.model_dim,
        inter_dim=args.inter_dim,
        tile_m=args.tile_m,
        tile_n=args.tile_n,
        tile_k=args.tile_k,
        m_warp=args.m_warp,
        n_warp=args.n_warp,
        num_buffers=args.num_buffers,
        waves_per_eu=args.waves_per_eu,
        inst_prefetch=args.inst_prefetch,
        use_tdm_store=args.use_tdm_store,
        use_scale_opsel=args.use_scale_opsel,
        wave_specialized_tdm=args.wave_specialized_tdm,
        expert_sched_mode=args.expert_sched_mode,
        persistent_workers=args.persistent_workers,
        cluster_m=args.cluster_m,
        cluster_n=args.cluster_n,
    )
    persist_modes = [args.persistent]
    if args.non_persistent:
        persist_modes = [False, True] if args.persistent else [False]

    print(
        f"[masked-grouped-moe-gemm] scenario={args.scenario} shape={s} stage={args.stage} "
        f"verify={args.verify} data={args.data} masked_mode={args.masked_mode}",
        flush=True,
    )
    for persistent in persist_modes:
        if args.scenario == "moe":
            us1, us2 = _run_mock_moe(
                s,
                tokens=args.tokens,
                topk=args.topk,
                route_mode=args.route_mode,
                persistent=persistent,
                verify=args.verify,
                data=args.data,
                warmup=args.warmup,
                iters=args.iters,
            )
            print(f"[masked-grouped-moe-gemm] mock_moe persistent={persistent} stage1_us={us1:.2f} stage2_us={us2:.2f}", flush=True)
            continue
        if args.stage in ("1", "both"):
            us = _run_stage1(s, persistent=persistent, verify=args.verify, data=args.data, masked_mode=args.masked_mode, warmup=args.warmup, iters=args.iters, masked_m_override=args.masked_m_override)
            print(f"[masked-grouped-moe-gemm] stage1 persistent={persistent} us={us:.2f}", flush=True)
        if args.stage in ("2", "both"):
            us = _run_stage2(s, persistent=persistent, verify=args.verify, data=args.data, masked_mode=args.masked_mode, warmup=args.warmup, iters=args.iters, masked_m_override=args.masked_m_override)
            print(f"[masked-grouped-moe-gemm] stage2 persistent={persistent} us={us:.2f}", flush=True)


if __name__ == "__main__":
    main()
