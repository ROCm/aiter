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
) -> dict[str, int]:
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
    )


def _masked_m(experts: int, max_m: int, *, mode: str = "mixed") -> torch.Tensor:
    if mode == "full":
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
        return torch.stack([_pattern_mxfp4(rows, cols, seed=seed + e) for e in range(experts)])
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

    grouped = torch.zeros((experts, max_m, cols // 2), dtype=torch.uint8)
    for expert in range(experts):
        for row in range(max_m):
            token = int(slot_token_ids[expert, row].item())
            if token < 0:
                continue
            rank = int(slot_rank_ids[expert, row].item())
            grouped[expert, row] = _pattern_mxfp4(1, cols, seed=seed + token * 17 + rank * 3 + expert)[0]
    return grouped


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
    a_f32 = _mxfp4_to_f32(a_fp4.view(torch.uint8))[:m, :k]
    b_f32 = _mxfp4_to_f32(b_fp4.view(torch.uint8))[:n, :k]
    a_sc = _e8m0_to_f32(a_scale.view(torch.uint8)).repeat_interleave(SCALE_BLOCK, dim=-1)[:m, :k]
    b_sc = _e8m0_to_f32(b_scale.view(torch.uint8)).repeat_interleave(SCALE_BLOCK, dim=-1)[:n, :k]
    return torch.matmul(a_f32 * a_sc, (b_f32 * b_sc).T)


def _assert_valid_rows_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual = actual.float()
    expected = expected.to(actual.device).float()
    diff = (actual - expected).abs()
    print(f"[masked-grouped-moe-gemm] {name}: shape={tuple(actual.shape)} max_abs={float(diff.max()):.8e} mean_abs={float(diff.mean()):.8e}", flush=True)
    torch.testing.assert_close(actual, expected, rtol=0.25, atol=0.25)


def _time_kernel(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / max(1, iters)


def _run_stage1(s: dict[str, int], *, persistent: bool, verify: bool, data: str, masked_mode: str, warmup: int, iters: int) -> float:
    E, max_m = s["experts"], s["max_m"]
    model_dim, inter_dim = s["model_dim"], s["inter_dim"]
    _log(f"stage1 persistent={persistent}: build inputs")
    masked_m = _masked_m(E, max_m, mode=masked_mode)
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
        tile_m=s["tile_m"], tile_n=s["tile_n"], tile_k=s["tile_k"], m_warp=s["m_warp"], n_warp=s["n_warp"],
        out_dtype="f16", num_buffers=2, expert_sched_mode=False, grouped_persistent_m=persistent,
    )
    _log(f"stage1 persistent={persistent}: compile done in {time.perf_counter() - compile_start:.2f}s")
    x = x_raw.cuda()
    w = w_raw.cuda()

    def launch():
        kernel(y, x, w, x_scale, w_scale, masked_m, max_m, inter_dim, model_dim, E, stream=torch.cuda.current_stream())

    _log(f"stage1 persistent={persistent}: launch warmup={warmup} iters={iters}")
    us = _time_kernel(launch, warmup=warmup, iters=iters)
    _log(f"stage1 persistent={persistent}: launch done")
    if verify:
        for e in range(E):
            valid = int(masked_m[e].item())
            if valid == 0:
                continue
            gate_up = _reference_mxfp4(x_raw[e], w_raw[e], x_scale_raw[e], w_scale_raw[e], valid, 2 * inter_dim, model_dim)
            expected = (torch.nn.functional.silu(gate_up[:, :inter_dim]) * gate_up[:, inter_dim:]).to(torch.float16).float()
            _assert_valid_rows_close(f"stage1 persistent={persistent} expert={e}", y[e, :valid], expected)
    return us


def _run_stage2(s: dict[str, int], *, persistent: bool, verify: bool, data: str, masked_mode: str, warmup: int, iters: int) -> float:
    E, max_m = s["experts"], s["max_m"]
    model_dim, inter_dim = s["model_dim"], s["inter_dim"]
    _log(f"stage2 persistent={persistent}: build inputs")
    masked_m = _masked_m(E, max_m, mode=masked_mode)
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
        tile_m=s["tile_m"], tile_n=s["tile_n"], tile_k=s["tile_k"], m_warp=s["m_warp"], n_warp=s["n_warp"],
        out_dtype="f16", num_buffers=2, expert_sched_mode=False, grouped_persistent_m=persistent,
    )
    _log(f"stage2 persistent={persistent}: compile done in {time.perf_counter() - compile_start:.2f}s")
    x = x_raw.cuda()
    w = w_raw.cuda()

    def launch():
        kernel(y, x, w, x_scale, w_scale, masked_m, max_m, model_dim, inter_dim, E, stream=torch.cuda.current_stream())

    _log(f"stage2 persistent={persistent}: launch warmup={warmup} iters={iters}")
    us = _time_kernel(launch, warmup=warmup, iters=iters)
    _log(f"stage2 persistent={persistent}: launch done")
    if verify:
        for e in range(E):
            valid = int(masked_m[e].item())
            if valid == 0:
                continue
            expected = _reference_mxfp4(x_raw[e], w_raw[e], x_scale_raw[e], w_scale_raw[e], valid, model_dim, inter_dim).to(torch.float16).float()
            _assert_valid_rows_close(f"stage2 persistent={persistent} expert={e}", y[e, :valid], expected)
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
        tile_m=s["tile_m"], tile_n=s["tile_n"], tile_k=s["tile_k"], m_warp=s["m_warp"], n_warp=s["n_warp"],
        out_dtype="f16", num_buffers=2, expert_sched_mode=False, grouped_persistent_m=persistent,
    )
    _log(f"mock moe stage1 persistent={persistent}: compile done in {time.perf_counter() - compile_start:.2f}s")
    x1 = x1_raw.cuda()
    w1 = w1_raw.cuda()

    def launch_stage1():
        k1(y1, x1, w1, x1_scale, w1_scale, masked_m, max_m, inter_dim, model_dim, E, stream=torch.cuda.current_stream())

    _log(f"mock moe stage1 persistent={persistent}: launch warmup={warmup} iters={iters}")
    us1 = _time_kernel(launch_stage1, warmup=warmup, iters=iters)
    _log(f"mock moe stage1 persistent={persistent}: launch done")
    if verify:
        for e in range(E):
            valid = int(masked_m_cpu[e].item())
            if valid == 0:
                continue
            gate_up = _reference_mxfp4(x1_raw[e], w1_raw[e], x1_scale_raw[e], w1_scale_raw[e], valid, 2 * inter_dim, model_dim)
            expected = (torch.nn.functional.silu(gate_up[:, :inter_dim]) * gate_up[:, inter_dim:]).to(torch.float16).float()
            _assert_valid_rows_close(f"mock stage1 persistent={persistent} expert={e}", y1[e, :valid], expected)

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
        tile_m=s["tile_m"], tile_n=s["tile_n"], tile_k=s["tile_k"], m_warp=s["m_warp"], n_warp=s["n_warp"],
        out_dtype="f16", num_buffers=2, expert_sched_mode=False, grouped_persistent_m=persistent,
    )
    _log(f"mock moe stage2 persistent={persistent}: compile done in {time.perf_counter() - compile_start:.2f}s")
    x2 = x2_raw.cuda()
    w2 = w2_raw.cuda()

    def launch_stage2():
        k2(y2, x2, w2, x2_scale, w2_scale, masked_m, max_m, model_dim, inter_dim, E, stream=torch.cuda.current_stream())

    _log(f"mock moe stage2 persistent={persistent}: launch warmup={warmup} iters={iters}")
    us2 = _time_kernel(launch_stage2, warmup=warmup, iters=iters)
    _log(f"mock moe stage2 persistent={persistent}: launch done")
    if verify:
        expected_topk_out = torch.zeros((tokens, topk, model_dim), device="cuda", dtype=torch.float32)
        for e in range(E):
            valid = int(masked_m_cpu[e].item())
            if valid == 0:
                continue
            expected = _reference_mxfp4(x2_raw[e], w2_raw[e], x2_scale_raw[e], w2_scale_raw[e], valid, model_dim, inter_dim).to(torch.float16).float()
            _assert_valid_rows_close(f"mock stage2 persistent={persistent} expert={e}", y2[e, :valid], expected)
            for row in range(valid):
                token = int(slot_token_ids[e, row].item())
                rank = int(slot_rank_ids[e, row].item())
                expected_topk_out[token, rank].copy_(expected[row])

    topk_out = torch.zeros((tokens, topk, model_dim), device="cuda", dtype=torch.float16)
    for expert in range(E):
        valid = int(masked_m_cpu[expert].item())
        for row in range(valid):
            token = int(slot_token_ids[expert, row].item())
            rank = int(slot_rank_ids[expert, row].item())
            topk_out[token, rank].copy_(y2[expert, row])
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
    parser.add_argument("--persistent", action="store_true", help="Use persistent grouped scheduling.")
    parser.add_argument("--non-persistent", action="store_true", help="Also run the non-persistent variant.")
    parser.add_argument("--verify", action="store_true", help="Run torch reference checks. Expensive for DeepSeek shapes.")
    parser.add_argument("--data", choices=("constant", "pattern"), default="constant")
    parser.add_argument("--masked-mode", choices=("mixed", "full", "descending"), default="mixed")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
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
            us = _run_stage1(s, persistent=persistent, verify=args.verify, data=args.data, masked_mode=args.masked_mode, warmup=args.warmup, iters=args.iters)
            print(f"[masked-grouped-moe-gemm] stage1 persistent={persistent} us={us:.2f}", flush=True)
        if args.stage in ("2", "both"):
            us = _run_stage2(s, persistent=persistent, verify=args.verify, data=args.data, masked_mode=args.masked_mode, warmup=args.warmup, iters=args.iters)
            print(f"[masked-grouped-moe-gemm] stage2 persistent={persistent} us={us:.2f}", flush=True)


if __name__ == "__main__":
    main()
