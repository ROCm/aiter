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
from aiter.jit.utils.chip_info import get_cu_num  # noqa: E402
from flydsl.runtime.device import get_rocm_arch  # noqa: E402

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]
SCALE_BLOCK = 32
DEFAULT_SCALE_U8 = 120
_START_TIME = time.perf_counter()


def _log(message: str) -> None:
    print(f"[masked-grouped-moe-gemm][{time.perf_counter() - _START_TIME:.2f}s] {message}", flush=True)


def _auto_rotate_count(per_iter_bytes: int, requested: int, *, headroom: float = 0.85) -> int:
    """Pick the rotate count actually used.

    - requested == 0: rotate disabled (returns 1, the trivial single-buffer case).
    - requested  > 0: cap at the maximum number of buffers that fit into
      ``headroom * free_gpu_memory``. Returns at least 1.
    """
    if requested <= 0:
        return 1
    if not torch.cuda.is_available():
        return 1
    free_bytes, _total = torch.cuda.mem_get_info()
    budget = int(free_bytes * headroom)
    if per_iter_bytes <= 0:
        return max(1, requested)
    max_fit = max(1, budget // max(per_iter_bytes, 1))
    return max(1, min(requested, max_fit))


def _measure_gemm_event_us(
    launch_with_events,
    *,
    num_warmup: int,
    num_iters: int,
    rotate_select=None,
) -> float:
    """Measure pure GEMM device time per iter using start/end cuda.Event pairs
    that the kernel wrapper records around the GEMM launch only.

    `launch_with_events(start_event, end_event)` must invoke the wrapper with
    ``_gemm_events=(start_event, end_event)`` so that prefix-sum / epilogue ops
    are excluded.

    If `rotate_select` is given it is called as ``rotate_select(iter_idx)``
    before each launch (warmup included with negative ``iter_idx``) so the
    closure can swap to a different on-device buffer set. This forces L2/HBM
    cache misses across iters when the rotation set is larger than L2.

    Returns trimmed-mean us per iteration.
    """
    # Warm-up (no recording -- uses scratch events to avoid disturbing samples).
    warm_start = torch.cuda.Event(enable_timing=True)
    warm_end = torch.cuda.Event(enable_timing=True)
    for w in range(num_warmup):
        if rotate_select is not None:
            rotate_select(-1 - w)
        launch_with_events(warm_start, warm_end)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    for i in range(num_iters):
        if rotate_select is not None:
            rotate_select(i)
        launch_with_events(starts[i], ends[i])
    torch.cuda.synchronize()

    samples_us = [starts[i].elapsed_time(ends[i]) * 1e3 for i in range(num_iters)]
    samples_us.sort()
    trim = max(1, num_iters // 20)
    trimmed = samples_us[trim:num_iters - trim] if num_iters > 2 * trim else samples_us
    return sum(trimmed) / max(1, len(trimmed))


def _run_perftest_us(func, *args, num_warmup: int, num_iters: int, rotate: int) -> float:
    """Measure wrapper latency with aiter.test_common.run_perftest.

    Keep run_perftest's own rotate disabled. Its rotate path deep-copies every
    CUDA tensor argument, including outputs and metadata tensors; this script
    manages input rotation explicitly so persistent kernels see stable output
    buffers and only x/w/scale inputs rotate.
    """
    _out, us = run_perftest(
        func,
        *args,
        num_warmup=num_warmup,
        num_iters=num_iters,
        testGraph=False,
        num_rotate_args=1,
    )
    return float(us)


def _make_m_tile_prefix_for_shape(masked_m: torch.Tensor, s: dict[str, int]) -> torch.Tensor:
    valid_m = masked_m[:s["experts"]].to(dtype=torch.int32)
    valid_m = valid_m.clamp(min=0, max=s["max_m"])
    valid_tiles = torch.div(
        valid_m + (s["tile_m"] - 1),
        s["tile_m"],
        rounding_mode="floor",
    )
    prefix = torch.empty((s["experts"] + 1,), device=masked_m.device, dtype=torch.int32)
    prefix[0].zero_()
    torch.cumsum(valid_tiles, dim=0, out=prefix[1:])
    return prefix


def _make_m_tile_map_for_shape(masked_m: torch.Tensor, s: dict[str, int]) -> torch.Tensor:
    valid_m = masked_m[:s["experts"]].to(dtype=torch.int32)
    valid_m = valid_m.clamp(min=0, max=s["max_m"])
    valid_tiles = torch.div(
        valid_m + (s["tile_m"] - 1),
        s["tile_m"],
        rounding_mode="floor",
    ).cpu().tolist()
    max_m_tiles = (s["max_m"] + s["tile_m"] - 1) // s["tile_m"]
    packed = [
        expert * max_m_tiles + local_tile
        for expert, count in enumerate(valid_tiles)
        for local_tile in range(int(count))
    ]
    if not packed:
        packed = [0]
    return torch.tensor(packed, device=masked_m.device, dtype=torch.int32)


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
    split_k: int = 1,
    inst_prefetch: bool = False,
    use_tdm_store: bool = True,
    use_scale_opsel: bool = False,
    wave_specialized_tdm: bool = False,
    expert_sched_mode: bool = False,
    persistent_workers: int | None = None,
    cluster_m: int = 1,
    cluster_n: int = 1,
) -> dict:
    if split_k > 1:
        use_tdm_store = False
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
        split_k=split_k,
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
    "num_buffers", "waves_per_eu", "split_k", "inst_prefetch", "use_tdm_store",
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
        if cols % 2 != 0:
            raise ValueError(f"cols must be even, got {cols}")
        # Vectorized GPU build. The pattern values are {0,1,8,9} which equals
        # (idx & 1) | ((idx & 2) << 2) for idx in [0..3], so we can avoid a
        # 64-bit gather lookup entirely. We chunk over experts to bound peak
        # GPU memory (each chunk allocates an int32 index tile of size
        # chunk*R*C*4 bytes).
        device = "cuda" if torch.cuda.is_available() else "cpu"
        per_expert_rxc = rows * cols
        max_chunk_bytes = 256 * 1024 * 1024
        max_chunk = max(1, min(experts, max_chunk_bytes // max(per_expert_rxc * 4, 1)))
        out = torch.empty((experts, rows, cols // 2), dtype=torch.uint8, device=device)
        base = torch.arange(per_expert_rxc, dtype=torch.int32, device=device)
        for start in range(0, experts, max_chunk):
            end = min(experts, start + max_chunk)
            seeds = torch.arange(start + seed, end + seed,
                                 dtype=torch.int32, device=device)
            idx = (base.view(1, -1) + seeds.view(-1, 1)) & 3                   # (chunk, R*C) i32, in [0,3]
            # u4 = (idx & 1) | ((idx & 2) << 2). Compute as i32 then cast.
            u4_i32 = (idx & 1) | ((idx & 2) << 2)                              # (chunk, R*C) i32, in {0,1,8,9}
            u4 = u4_i32.to(torch.uint8).view(end - start, rows, cols)
            low = u4[..., 0::2] & 0x0F
            high = (u4[..., 1::2] & 0x0F) << 4
            out[start:end] = (low | high)
            del idx, u4_i32, u4, low, high
        return out.cpu()
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
        # Perfect coverage: assign (t*topk + r) mod E. With tokens*topk <= experts
        # this gives 1 token per expert (max_expert_m=1). When tokens*topk > experts
        # it wraps evenly so every expert sees the same load (+/-1).
        topk_ids = torch.remainder(token_idx * topk + rank_idx, experts)
    elif mode == "diagonal" or mode == "balanced":
        # Lower-triangular pattern: only experts in [0..tokens+topk-2] are
        # touched. The historical "balanced" name is kept as an alias for
        # backward compat with sweep scripts; prefer "diagonal" for clarity.
        topk_ids = torch.remainder(token_idx + rank_idx, experts)
    else:
        raise ValueError(
            f"unknown route mode {mode!r}; expected one of "
            "'expert_balance', 'diagonal', 'hot', 'stride' (or legacy 'balanced')"
        )
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

    # Vectorized: build per-(expert, row, col) pattern indices on GPU using
    # i32 arithmetic only (no 64-bit gather lookup), then bit-twiddle to the
    # {0,1,8,9} value set. Replaces 256*32=8192 host-side .item() loops.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    slot_tok = slot_token_ids.to(device, dtype=torch.int32)
    slot_rnk = slot_rank_ids.to(device, dtype=torch.int32)
    expert_idx = torch.arange(experts, dtype=torch.int32, device=device).view(experts, 1)
    valid = slot_tok >= 0
    base_seed = (
        seed
        + slot_tok.clamp(min=0) * 17
        + slot_rnk.clamp(min=0) * 3
        + expert_idx
    )                                                                     # (E, max_m) i32
    base_idx = torch.arange(cols, dtype=torch.int32, device=device).view(1, 1, cols)
    idx = (base_idx + base_seed.unsqueeze(-1)) & 3                        # (E, max_m, cols) i32 in [0,3]
    u4_i32 = (idx & 1) | ((idx & 2) << 2)                                 # values {0,1,8,9}
    u4 = u4_i32.to(torch.uint8)
    low = u4[..., 0::2] & 0x0F
    high = (u4[..., 1::2] & 0x0F) << 4
    packed = (low | high)
    packed = packed * valid.unsqueeze(-1).to(torch.uint8)
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
    rows, k_scale = scale.shape
    scales_per_wmma = 4
    scale_k_per_tile = tile_k // SCALE_BLOCK
    assert k_scale % scale_k_per_tile == 0
    assert scale_k_per_tile % scales_per_wmma == 0
    wmma_rep = warp_tile // wmma_dim
    k_groups = k_scale // scale_k_per_tile
    k_wmma_steps = scale_k_per_tile // scales_per_wmma
    if rows % (wmma_rep * wmma_dim) != 0:
        # Decode-tight buffers may have max_m=1 while the compute tile still
        # covers 16 rows. With one M-side WMMA repeat there is no row
        # interleave to perform, and the kernel's expected scale shape is the
        # raw (rows, k_scale) layout.
        if wmma_rep == 1:
            return scale.contiguous()
        raise ValueError(
            f"scale rows={rows} must be divisible by warp_tile={wmma_rep * wmma_dim} "
            f"for warp_tile={warp_tile}"
        )
    row_groups = rows // (wmma_rep * wmma_dim)
    g = scale.view(row_groups, wmma_rep, wmma_dim, k_groups, k_wmma_steps, scales_per_wmma)
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


# Relative L2 distance is the canonical metric for low-precision GEMM checks:
# entry-wise rtol/atol can flag legitimate per-element MXFP4 jitter while still
# missing systematic bias. We bound ||actual-expected||_2 / ||expected||_2 instead.
DEFAULT_L2_TOL = 0.05

# Sentinel value written into output buffers before the kernel launch when
# verify is on. After the launch, any cell still equal to this value within
# the masked-valid region means the kernel skipped writing it (early-quit
# bug, persistent scheduling miscount, masked_m mismatch, etc.).
_VERIFY_SENTINEL = -8192.0
_VERIFY_WARN_ONLY = False


def _sentinel_like(t: torch.Tensor) -> torch.Tensor:
    """Sentinel represented in the tensor's dtype.

    Keep the comparison in the original dtype so future dtype changes do not
    accidentally compare a rounded storage value against an fp32 literal.
    """
    return torch.tensor(_VERIFY_SENTINEL, dtype=t.dtype, device=t.device)


def _sentinel_count(t: torch.Tensor) -> int:
    if t.numel() == 0:
        return 0
    return int((t == _sentinel_like(t)).sum().item())


def _handle_verify_failure(name: str, exc: AssertionError) -> None:
    if not _VERIFY_WARN_ONLY:
        raise exc
    print(f"[masked-grouped-moe-gemm] {name}: VERIFY WARNING: {exc}", flush=True)


def _stat_summary(t: torch.Tensor) -> str:
    """Compact one-line stats: shape, min, max, abs_max, nonzero ratio."""
    if t.numel() == 0:
        return f"shape={tuple(t.shape)} EMPTY"
    flat = t.float().flatten()
    nz = float((flat != 0).sum()) / flat.numel()
    return (
        f"shape={tuple(t.shape)} "
        f"min={float(flat.min()):.4e} max={float(flat.max()):.4e} "
        f"abs_max={float(flat.abs().max()):.4e} nonzero={nz*100:.1f}%"
    )


def _kernel_ran_check(name: str, actual: torch.Tensor, masked_m_cpu: torch.Tensor) -> None:
    """Verify the kernel actually wrote to the masked-valid region."""
    B, M_pad = actual.shape[:2]
    row_idx = torch.arange(M_pad, device=actual.device).view(1, M_pad)
    masked_m_dev = masked_m_cpu.to(actual.device, dtype=torch.int32).view(B, 1)
    row_mask = (row_idx < masked_m_dev).unsqueeze(-1)  # (B, M_pad, 1) bool
    # Look at the masked-valid region only.
    valid = actual[row_mask.expand_as(actual)]
    if valid.numel() == 0:
        print(f"[masked-grouped-moe-gemm] {name} kernel-ran-check: NO valid cells", flush=True)
        return
    sentinels = _sentinel_count(valid)
    pct = 100.0 * sentinels / valid.numel()
    print(
        f"[masked-grouped-moe-gemm] {name} kernel-ran-check: "
        f"valid_cells={valid.numel()} sentinel_remaining={int(sentinels)} ({pct:.2f}%) "
        f"-- {_stat_summary(valid)}",
        flush=True,
    )
    if sentinels > 0:
        raise AssertionError(
            f"{name}: kernel left {int(sentinels)}/{valid.numel()} sentinel "
            f"values ({pct:.2f}%) in the masked-valid region -- the GEMM did "
            f"not write to those cells."
        )


# When ||expected||_2 is essentially zero (very few valid elements, MXFP4
# rounded silu(x)*up <<1, etc.) the relative L2 metric becomes unstable.
# In that regime we instead require ||actual - expected||_2 / sqrt(N) <= this
# absolute tolerance (RMS of element-wise residual). 1e-2 is comfortably
# above MXFP4 round-trip error when the result is fp16.
DEFAULT_L2_ABS_TOL = 1e-2


def _l2_metrics(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, bool]:
    """Return (metric, ref_norm, used_absolute).

    - If ||expected||_2 is meaningfully > 0, metric = relative L2.
    - Otherwise metric = RMS residual = ||actual-expected||_2 / sqrt(N).
    """
    diff_norm = float((actual - expected).norm(p=2))
    ref_norm = float(expected.norm(p=2))
    n = max(actual.numel(), 1)
    # "Meaningful" reference means the ref RMS is well above fp16 epsilon.
    if ref_norm <= 1e-6 * (n ** 0.5):
        rms = diff_norm / (n ** 0.5)
        return rms, ref_norm, True
    return diff_norm / ref_norm, ref_norm, False


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Backward-compatible scalar metric (relative L2, with absolute fallback)."""
    metric, _, _ = _l2_metrics(actual, expected)
    return metric


def _assert_l2(name: str, actual: torch.Tensor, expected: torch.Tensor, tol: float,
               *, abs_tol: float = DEFAULT_L2_ABS_TOL) -> None:
    metric, ref_norm, used_abs = _l2_metrics(actual, expected)
    eff_tol = abs_tol if used_abs else tol
    if metric > eff_tol:
        kind = "absolute RMS" if used_abs else "relative L2"
        raise AssertionError(
            f"{name}: {kind} {metric:.6e} exceeds tolerance {eff_tol:.6e} "
            f"(ref_norm={ref_norm:.3e})"
        )


def _assert_valid_rows_close(name: str, actual: torch.Tensor, expected: torch.Tensor,
                             *, l2_tol: float = DEFAULT_L2_TOL,
                             abs_tol: float = DEFAULT_L2_ABS_TOL) -> None:
    actual = actual.float()
    expected = expected.to(actual.device).float()
    diff = (actual - expected).abs()
    metric, ref_norm, used_abs = _l2_metrics(actual, expected)
    label = "abs_rms" if used_abs else "rel_l2"
    eff_tol = abs_tol if used_abs else l2_tol
    print(
        f"[masked-grouped-moe-gemm] {name}: shape={tuple(actual.shape)} "
        f"max_abs={float(diff.max()):.4e} mean_abs={float(diff.mean()):.4e} "
        f"{label}={metric:.4e} (tol={eff_tol:.2e}, ref_norm={ref_norm:.3e})",
        flush=True,
    )
    _assert_l2(name, actual, expected, l2_tol, abs_tol=abs_tol)


def _assert_batched_close(
    name: str,
    actual: torch.Tensor,        # (B, M_pad, N)
    expected: torch.Tensor,      # (B, M_pad, N)
    masked_m_cpu: torch.Tensor,  # (B,) int
    *,
    l2_tol: float = DEFAULT_L2_TOL,
    abs_tol: float = DEFAULT_L2_ABS_TOL,
) -> None:
    """Compare only the [:valid_m[e]] rows per expert in a single GPU op.

    Builds a row mask on device, broadcasts, and computes:
      - global metric (relative L2 if ref is meaningful, otherwise abs RMS)
      - per-expert worst-case (same fallback rule, evaluated per-expert).
    Per-expert outliers are flagged separately so a single bad expert can't
    hide inside the global average.
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

    metric, ref_norm, used_abs = _l2_metrics(actual_masked, expected_masked)
    label = "abs_rms" if used_abs else "rel_l2"
    eff_tol = abs_tol if used_abs else l2_tol

    # Diagnostic dump: stats of valid actual vs expected, and a 0th-active-
    # expert row spot-check. This makes "kernel did not run" immediately
    # visible (actual all near sentinel / all near zero / etc.).
    valid_mask = (row_mask > 0).expand_as(actual_masked)
    actual_valid = actual_masked[valid_mask]
    expected_valid = expected_masked[valid_mask]
    print(
        f"[masked-grouped-moe-gemm] {name} dump: "
        f"actual_valid {_stat_summary(actual_valid)} | "
        f"expected_valid {_stat_summary(expected_valid)}",
        flush=True,
    )
    active_b = torch.nonzero(masked_m_cpu > 0, as_tuple=False).flatten()
    if len(active_b) > 0:
        e0 = int(active_b[0].item())
        n_show = min(8, actual.shape[-1])
        a_row = actual[e0, 0, :n_show].tolist()
        e_row = expected[e0, 0, :n_show].tolist()
        print(
            f"[masked-grouped-moe-gemm] {name} expert{e0} row0[:{n_show}]: "
            f"actual={[f'{v:+.4e}' for v in a_row]} "
            f"expected={[f'{v:+.4e}' for v in e_row]}",
            flush=True,
        )

    # Per-expert worst-case with the same fallback rule.
    n_per_exp = float(actual.shape[-1] * M_pad)
    diff_norm_per_exp = (actual_masked - expected_masked).reshape(B, -1).norm(p=2, dim=1)
    ref_norm_per_exp = expected_masked.reshape(B, -1).norm(p=2, dim=1)
    threshold = 1e-6 * (n_per_exp ** 0.5)  # same scale rule as _l2_metrics
    has_signal = ref_norm_per_exp > threshold
    metric_per_exp = torch.where(
        has_signal,
        diff_norm_per_exp / ref_norm_per_exp.clamp_min(1e-30),
        diff_norm_per_exp / max(n_per_exp ** 0.5, 1.0),  # abs RMS fallback
    )
    eff_tol_per_exp = torch.where(
        has_signal,
        torch.full_like(metric_per_exp, l2_tol),
        torch.full_like(metric_per_exp, abs_tol),
    )
    # Worst is the expert with the largest metric/threshold ratio (so absolute
    # and relative regimes are comparable).
    severity = metric_per_exp / eff_tol_per_exp.clamp_min(1e-30)
    worst_idx = int(torch.argmax(severity).item())
    worst_metric = float(metric_per_exp[worst_idx].item())
    worst_used_abs = not bool(has_signal[worst_idx].item())
    worst_label = "abs_rms" if worst_used_abs else "rel_l2"
    worst_eff_tol = abs_tol if worst_used_abs else l2_tol

    print(
        f"[masked-grouped-moe-gemm] {name} batched: shape={tuple(actual.shape)} "
        f"active_experts={int((masked_m_cpu > 0).sum().item())} "
        f"max_abs={max_abs:.4e} mean_abs={mean_abs:.4e} "
        f"{label}={metric:.4e} worst_expert_{worst_label}={worst_metric:.4e}@e{worst_idx} "
        f"(tol={eff_tol:.2e}/worst={worst_eff_tol:.2e}, ref_norm={ref_norm:.3e})",
        flush=True,
    )
    if metric > eff_tol:
        kind = "global abs RMS" if used_abs else "global relative L2"
        raise AssertionError(
            f"{name}: {kind} {metric:.6e} exceeds tolerance {eff_tol:.6e} "
            f"(worst expert={worst_idx} {worst_label}={worst_metric:.6e})"
        )
    # Per-expert outlier guard: 4x the appropriate tolerance, but never less
    # than 0.10 relative or 4*abs_tol absolute.
    per_exp_outlier_tol = max(worst_eff_tol * 4, 0.10 if not worst_used_abs else worst_eff_tol * 4)
    if worst_metric > per_exp_outlier_tol:
        raise AssertionError(
            f"{name}: per-expert worst {worst_label}={worst_metric:.6e} "
            f"(expert {worst_idx}) exceeds 4x tolerance ({per_exp_outlier_tol:.6e})"
        )


def _run_stage1(
    s: dict[str, int],
    *,
    persistent: bool,
    verify: bool,
    data: str,
    masked_mode: str,
    warmup: int,
    iters: int,
    masked_m_override: int | None = None,
    bench_mode: str = "event",
    bench_scope: str = "wrapper",
    rotate: int = 0,
) -> dict[str, float]:
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

    per_iter_bytes = (
        w.numel() * w.element_size()
        + w_scale.numel() * w_scale.element_size()
        + x.numel() * x.element_size()
        + x_scale.numel() * x_scale.element_size()
    )
    rotate_n = _auto_rotate_count(per_iter_bytes, rotate)
    if rotate_n > 1:
        _log(
            f"stage1 persistent={persistent}: rotate=on N={rotate_n} "
            f"per_iter={per_iter_bytes/1e9:.2f}GB total={(per_iter_bytes*rotate_n)/1e9:.2f}GB"
        )
        x_list = [x] + [x.clone() for _ in range(rotate_n - 1)]
        w_list = [w] + [w.clone() for _ in range(rotate_n - 1)]
        x_scale_list = [x_scale] + [x_scale.clone() for _ in range(rotate_n - 1)]
        w_scale_list = [w_scale] + [w_scale.clone() for _ in range(rotate_n - 1)]
        for i in range(1, rotate_n):
            x_list[i].bitwise_xor_(torch.tensor(i & 0xFF, dtype=x.dtype, device=x.device))
            w_list[i].bitwise_xor_(torch.tensor((i * 17) & 0xFF, dtype=w.dtype, device=w.device))
        torch.cuda.synchronize()
        cur = [0]

        def rotate_select(iter_idx):
            cur[0] = iter_idx % rotate_n if iter_idx >= 0 else (-iter_idx - 1) % rotate_n

        def launch(start_ev=None, end_ev=None):
            i = cur[0]
            kernel(y, x_list[i], w_list[i], x_scale_list[i], w_scale_list[i],
                   masked_m, max_m, inter_dim, model_dim, E,
                   stream=torch.cuda.current_stream(),
                   _gemm_events=(start_ev, end_ev) if start_ev is not None else None)
    else:
        rotate_select = None

        def launch(start_ev=None, end_ev=None):
            kernel(y, x, w, x_scale, w_scale, masked_m, max_m, inter_dim, model_dim, E,
                   stream=torch.cuda.current_stream(),
                   _gemm_events=(start_ev, end_ev) if start_ev is not None else None)

    perf_iter = [0]
    bench_prefix = _make_m_tile_prefix_for_shape(masked_m, s) if persistent and bench_scope == "gemm" else None
    bench_map = _make_m_tile_map_for_shape(masked_m, s) if persistent and bench_scope == "gemm" else None
    bench_tmp = (
        torch.empty((E, max_m, 2 * inter_dim), device=y.device, dtype=y.dtype)
        if bench_scope == "gemm"
        else None
    )

    def launch_perftest():
        if rotate_select is not None:
            rotate_select(perf_iter[0])
            perf_iter[0] += 1
        if bench_scope == "gemm":
            if rotate_select is not None:
                i = perf_iter[0] - 1
            else:
                i = 0
            if rotate_select is not None:
                x_arg, w_arg = x_list[i % rotate_n], w_list[i % rotate_n]
                x_scale_arg, w_scale_arg = x_scale_list[i % rotate_n], w_scale_list[i % rotate_n]
            else:
                x_arg, w_arg = x, w
                x_scale_arg, w_scale_arg = x_scale, w_scale
            kernel(y, x_arg, w_arg, x_scale_arg, w_scale_arg, masked_m,
                   max_m, inter_dim, model_dim, E,
                   stream=torch.cuda.current_stream(),
                   _m_tile_prefix=bench_prefix,
                   _m_tile_map=bench_map,
                   _tmp=bench_tmp,
                   _skip_epilogue=True)
        else:
            launch()
        return y

    timings: dict[str, float] = {}
    if bench_mode in ("event", "both"):
        _log(f"stage1 persistent={persistent}: event launch warmup={warmup} iters={iters}"
             + (f" rotate=N{rotate_n}" if rotate_select is not None else ""))
        us = _measure_gemm_event_us(launch, num_warmup=warmup, num_iters=iters,
                                    rotate_select=rotate_select)
        timings["gemm_event_us"] = us
        _log(f"stage1 persistent={persistent}: event launch done gemm_event_us={us:.2f}")
    if bench_mode in ("run_perftest", "both"):
        _log(f"stage1 persistent={persistent}: run_perftest warmup={warmup} iters={iters} rotate={rotate_n}")
        us = _run_perftest_us(
            launch_perftest,
            num_warmup=warmup,
            num_iters=iters,
            rotate=rotate_n,
        )
        timings["run_perftest_us"] = us
        _log(f"stage1 persistent={persistent}: run_perftest done us={us:.2f}")
    if verify:
        if bench_scope == "gemm" or rotate_select is not None:
            if rotate_select is not None:
                rotate_select(0)
            launch()
            torch.cuda.synchronize()
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
    return timings


def _run_stage2(
    s: dict[str, int],
    *,
    persistent: bool,
    verify: bool,
    data: str,
    masked_mode: str,
    warmup: int,
    iters: int,
    masked_m_override: int | None = None,
    bench_mode: str = "event",
    bench_scope: str = "wrapper",
    rotate: int = 0,
) -> dict[str, float]:
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

    per_iter_bytes = (
        w.numel() * w.element_size()
        + w_scale.numel() * w_scale.element_size()
        + x.numel() * x.element_size()
        + x_scale.numel() * x_scale.element_size()
    )
    rotate_n = _auto_rotate_count(per_iter_bytes, rotate)
    if rotate_n > 1:
        _log(
            f"stage2 persistent={persistent}: rotate=on N={rotate_n} "
            f"per_iter={per_iter_bytes/1e9:.2f}GB total={(per_iter_bytes*rotate_n)/1e9:.2f}GB"
        )
        x_list = [x] + [x.clone() for _ in range(rotate_n - 1)]
        w_list = [w] + [w.clone() for _ in range(rotate_n - 1)]
        x_scale_list = [x_scale] + [x_scale.clone() for _ in range(rotate_n - 1)]
        w_scale_list = [w_scale] + [w_scale.clone() for _ in range(rotate_n - 1)]
        for i in range(1, rotate_n):
            x_list[i].bitwise_xor_(torch.tensor(i & 0xFF, dtype=x.dtype, device=x.device))
            w_list[i].bitwise_xor_(torch.tensor((i * 17) & 0xFF, dtype=w.dtype, device=w.device))
        torch.cuda.synchronize()
        cur = [0]

        def rotate_select(iter_idx):
            cur[0] = iter_idx % rotate_n if iter_idx >= 0 else (-iter_idx - 1) % rotate_n

        def launch(start_ev=None, end_ev=None):
            i = cur[0]
            kernel(y, x_list[i], w_list[i], x_scale_list[i], w_scale_list[i],
                   masked_m, max_m, model_dim, inter_dim, E,
                   stream=torch.cuda.current_stream(),
                   _gemm_events=(start_ev, end_ev) if start_ev is not None else None)
    else:
        rotate_select = None

        def launch(start_ev=None, end_ev=None):
            kernel(y, x, w, x_scale, w_scale, masked_m, max_m, model_dim, inter_dim, E,
                   stream=torch.cuda.current_stream(),
                   _gemm_events=(start_ev, end_ev) if start_ev is not None else None)

    perf_iter = [0]
    bench_prefix = _make_m_tile_prefix_for_shape(masked_m, s) if persistent and bench_scope == "gemm" else None
    bench_map = _make_m_tile_map_for_shape(masked_m, s) if persistent and bench_scope == "gemm" else None

    def launch_perftest():
        if rotate_select is not None:
            rotate_select(perf_iter[0])
            perf_iter[0] += 1
        if bench_scope == "gemm":
            if rotate_select is not None:
                i = perf_iter[0] - 1
                x_arg, w_arg = x_list[i % rotate_n], w_list[i % rotate_n]
                x_scale_arg, w_scale_arg = x_scale_list[i % rotate_n], w_scale_list[i % rotate_n]
            else:
                x_arg, w_arg = x, w
                x_scale_arg, w_scale_arg = x_scale, w_scale
            kernel(y, x_arg, w_arg, x_scale_arg, w_scale_arg, masked_m,
                   max_m, model_dim, inter_dim, E,
                   stream=torch.cuda.current_stream(),
                   _m_tile_prefix=bench_prefix,
                   _m_tile_map=bench_map)
        else:
            launch()
        return y

    timings: dict[str, float] = {}
    if bench_mode in ("event", "both"):
        _log(f"stage2 persistent={persistent}: event launch warmup={warmup} iters={iters}"
             + (f" rotate=N{rotate_n}" if rotate_select is not None else ""))
        us = _measure_gemm_event_us(launch, num_warmup=warmup, num_iters=iters,
                                    rotate_select=rotate_select)
        timings["gemm_event_us"] = us
        _log(f"stage2 persistent={persistent}: event launch done gemm_event_us={us:.2f}")
    if bench_mode in ("run_perftest", "both"):
        _log(f"stage2 persistent={persistent}: run_perftest warmup={warmup} iters={iters} rotate={rotate_n}")
        us = _run_perftest_us(
            launch_perftest,
            num_warmup=warmup,
            num_iters=iters,
            rotate=rotate_n,
        )
        timings["run_perftest_us"] = us
        _log(f"stage2 persistent={persistent}: run_perftest done us={us:.2f}")
    if verify:
        if bench_scope == "gemm" or rotate_select is not None:
            if rotate_select is not None:
                rotate_select(0)
            launch()
            torch.cuda.synchronize()
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
    return timings


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
    rotate: int = 0,
    bench_mode: str = "event",
    bench_scope: str = "wrapper",
) -> tuple[dict[str, float], dict[str, float]]:
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
    # CPU-side mirror of the persistent early-exit predicate. The kernel only
    # skips workers whose initial block_id >= total_tiles, where
    # total_tiles = sum_e(ceil(masked_m[e] / tile_m)) * ceil(N / tile_n).
    # This log lets us rule out a bad zero-tile path without synchronizing
    # device-side prefix buffers.
    m_tiles_cpu = torch.div(
        masked_m_cpu.clamp(min=0, max=max_m) + (s["tile_m"] - 1),
        s["tile_m"],
        rounding_mode="floor",
    )
    total_m_tiles = int(m_tiles_cpu.sum().item())
    stage1_n_tiles = (2 * inter_dim + s["tile_n"] - 1) // s["tile_n"]
    stage2_n_tiles = (model_dim + s["tile_n"] - 1) // s["tile_n"]
    _log(
        "mock moe early-exit mirror: "
        f"total_m_tiles={total_m_tiles} "
        f"stage1_total_tiles={total_m_tiles * stage1_n_tiles} "
        f"(n_tiles={stage1_n_tiles}) "
        f"stage2_total_tiles={total_m_tiles * stage2_n_tiles} "
        f"(n_tiles={stage2_n_tiles})"
    )
    if persistent and s["persistent_workers"] is None:
        s = dict(s)
        s["persistent_workers"] = max(1, min(int(get_cu_num()), max(total_m_tiles, 1)))
        _log(f"mock moe persistent auto-workers={s['persistent_workers']} (total_m_tiles={total_m_tiles})")

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
    x1_scale_dev = x1_scale  # already on-device after _prep_scale_batch
    w1_scale_dev = w1_scale

    # Per-iter on-device bytes that rotation actually displaces from cache.
    per_iter_bytes_s1 = (
        w1.numel() * w1.element_size()
        + w1_scale_dev.numel() * w1_scale_dev.element_size()
        + x1.numel() * x1.element_size()
        + x1_scale_dev.numel() * x1_scale_dev.element_size()
    )
    rotate_n_s1 = _auto_rotate_count(per_iter_bytes_s1, rotate)
    if rotate_n_s1 > 1:
        _log(
            f"mock moe stage1 persistent={persistent}: rotate=on N={rotate_n_s1} "
            f"per_iter={per_iter_bytes_s1/1e9:.2f}GB total={(per_iter_bytes_s1*rotate_n_s1)/1e9:.2f}GB"
        )
        # Build N-1 extra on-device copies. We mutate-in-place via add_(small)
        # so each replica has a different bit pattern (avoids any optimisation
        # that might dedup identical pages) but stays within the value range.
        x1_list = [x1] + [x1.clone() for _ in range(rotate_n_s1 - 1)]
        w1_list = [w1] + [w1.clone() for _ in range(rotate_n_s1 - 1)]
        x1_scale_list = [x1_scale_dev] + [x1_scale_dev.clone() for _ in range(rotate_n_s1 - 1)]
        w1_scale_list = [w1_scale_dev] + [w1_scale_dev.clone() for _ in range(rotate_n_s1 - 1)]
        for i in range(1, rotate_n_s1):
            # Toggle a single bit per byte so cache lines truly differ.
            x1_list[i].bitwise_xor_(torch.tensor(i & 0xFF, dtype=x1.dtype, device=x1.device))
            w1_list[i].bitwise_xor_(torch.tensor((i * 17) & 0xFF, dtype=w1.dtype, device=w1.device))
        torch.cuda.synchronize()
        cur = [0]

        def rotate_select_s1(iter_idx):
            cur[0] = iter_idx % rotate_n_s1 if iter_idx >= 0 else (-iter_idx - 1) % rotate_n_s1

        def launch_stage1(start_ev=None, end_ev=None, _debug_tmp_sentinel=None, _debug_tmp_out=None):
            i = cur[0]
            k1(y1, x1_list[i], w1_list[i], x1_scale_list[i], w1_scale_list[i],
               masked_m, max_m, inter_dim, model_dim, E,
               stream=torch.cuda.current_stream(),
               _gemm_events=(start_ev, end_ev) if start_ev is not None else None,
               _debug_tmp_sentinel=_debug_tmp_sentinel,
               _debug_tmp_out=_debug_tmp_out)
    else:
        rotate_select_s1 = None

        def launch_stage1(start_ev=None, end_ev=None, _debug_tmp_sentinel=None, _debug_tmp_out=None):
            k1(y1, x1, w1, x1_scale_dev, w1_scale_dev, masked_m, max_m, inter_dim, model_dim, E,
               stream=torch.cuda.current_stream(),
               _gemm_events=(start_ev, end_ev) if start_ev is not None else None,
               _debug_tmp_sentinel=_debug_tmp_sentinel,
               _debug_tmp_out=_debug_tmp_out)

    perf_iter_s1 = [0]
    bench_prefix_s1 = _make_m_tile_prefix_for_shape(masked_m, s) if persistent and bench_scope == "gemm" else None
    bench_map_s1 = _make_m_tile_map_for_shape(masked_m, s) if persistent and bench_scope == "gemm" else None
    bench_tmp_s1 = (
        torch.empty((E, max_m, 2 * inter_dim), device=y1.device, dtype=y1.dtype)
        if bench_scope == "gemm"
        else None
    )

    def launch_stage1_perftest():
        if rotate_select_s1 is not None:
            rotate_select_s1(perf_iter_s1[0])
            perf_iter_s1[0] += 1
        if bench_scope == "gemm":
            if rotate_select_s1 is not None:
                i = perf_iter_s1[0] - 1
                x_arg, w_arg = x1_list[i % rotate_n_s1], w1_list[i % rotate_n_s1]
                x_scale_arg = x1_scale_list[i % rotate_n_s1]
                w_scale_arg = w1_scale_list[i % rotate_n_s1]
            else:
                x_arg, w_arg = x1, w1
                x_scale_arg, w_scale_arg = x1_scale_dev, w1_scale_dev
            k1(y1, x_arg, w_arg, x_scale_arg, w_scale_arg,
               masked_m, max_m, inter_dim, model_dim, E,
               stream=torch.cuda.current_stream(),
               _m_tile_prefix=bench_prefix_s1,
               _m_tile_map=bench_map_s1,
               _tmp=bench_tmp_s1,
               _skip_epilogue=True)
        else:
            launch_stage1()
        return y1

    timings1: dict[str, float] = {}
    if bench_mode in ("event", "both"):
        _log(f"mock moe stage1 persistent={persistent}: event launch warmup={warmup} iters={iters}"
             + (f" rotate=N{rotate_n_s1}" if rotate_select_s1 is not None else ""))
        us1 = _measure_gemm_event_us(launch_stage1, num_warmup=warmup, num_iters=iters,
                                     rotate_select=rotate_select_s1)
        timings1["gemm_event_us"] = us1
        _log(f"mock moe stage1 persistent={persistent}: event launch done gemm_event_us={us1:.2f}"
             + (f" rotate=N{rotate_n_s1}" if rotate_select_s1 is not None else ""))
    if bench_mode in ("run_perftest", "both"):
        _log(f"mock moe stage1 persistent={persistent}: run_perftest warmup={warmup} iters={iters} rotate={rotate_n_s1}")
        us1 = _run_perftest_us(
            launch_stage1_perftest,
            num_warmup=warmup,
            num_iters=iters,
            rotate=rotate_n_s1,
        )
        timings1["run_perftest_us"] = us1
        _log(f"mock moe stage1 persistent={persistent}: run_perftest done us={us1:.2f}")
    if verify:
        # Fill y1 with a sentinel and run one uncounted launch on buffer 0
        # (matching the verify reference inputs). After this, any cell in the
        # masked-valid region still equal to the sentinel means the kernel
        # did not write that cell.
        y1.fill_(_VERIFY_SENTINEL)
        if rotate_select_s1 is not None:
            cur[0] = 0
        # Capture raw GEMM output (pre-silu*up) so we can tell whether the
        # GEMM kernel itself wrote anything, separate from the epilogue.
        _tmp_capture = []
        torch.cuda.synchronize()
        launch_stage1(_debug_tmp_sentinel=_VERIFY_SENTINEL, _debug_tmp_out=_tmp_capture)
        torch.cuda.synchronize()
        if _tmp_capture:
            tmp_dbg = _tmp_capture[0]
            tmp_flat = tmp_dbg.flatten()
            n_sent = _sentinel_count(tmp_flat)
            print(
                f"[masked-grouped-moe-gemm] mock stage1 persistent={persistent} "
                f"raw-tmp diag: shape={tuple(tmp_dbg.shape)} "
                f"sentinel_remaining={n_sent}/{tmp_flat.numel()} "
                f"({100.0*n_sent/tmp_flat.numel():.2f}%) -- {_stat_summary(tmp_dbg)}",
                flush=True,
            )
            # Also dump first active expert's pre-epilogue first 8 values.
            active_idx_dbg = torch.nonzero(masked_m_cpu > 0, as_tuple=False).flatten().tolist()
            if active_idx_dbg:
                e0 = int(active_idx_dbg[0])
                row = tmp_dbg[e0, 0, :8].float().tolist()
                print(
                    f"[masked-grouped-moe-gemm] mock stage1 persistent={persistent} "
                    f"raw-tmp expert{e0} row0[:8]: {[f'{v:+.4e}' for v in row]}",
                    flush=True,
                )

        active_idx = torch.nonzero(masked_m_cpu > 0, as_tuple=False).flatten().tolist()
        if active_idx:
            active_t = torch.tensor(active_idx, dtype=torch.long)
            actual_active_raw = y1[active_t]
            check_name = f"mock stage1 persistent={persistent}"
            try:
                _kernel_ran_check(
                    check_name,
                    actual_active_raw, masked_m_cpu[active_t],
                )
            except AssertionError as exc:
                _handle_verify_failure(check_name, exc)
            gate_up_all = _reference_mxfp4_batched(
                x1_raw[active_t], w1_raw[active_t],
                x1_scale_raw[active_t], w1_scale_raw[active_t],
                k=model_dim,
            )
            gate = gate_up_all[..., :inter_dim]
            up = gate_up_all[..., inter_dim:]
            expected_active = (torch.nn.functional.silu(gate) * up).to(torch.float16).float()
            actual_active = actual_active_raw.float()
            try:
                _assert_batched_close(
                    check_name,
                    actual_active, expected_active,
                    masked_m_cpu[active_t],
                )
            except AssertionError as exc:
                _handle_verify_failure(check_name, exc)

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
    x2_scale_dev = x2_scale
    w2_scale_dev = w2_scale

    per_iter_bytes_s2 = (
        w2.numel() * w2.element_size()
        + w2_scale_dev.numel() * w2_scale_dev.element_size()
        + x2.numel() * x2.element_size()
        + x2_scale_dev.numel() * x2_scale_dev.element_size()
    )
    rotate_n_s2 = _auto_rotate_count(per_iter_bytes_s2, rotate)
    if rotate_n_s2 > 1:
        _log(
            f"mock moe stage2 persistent={persistent}: rotate=on N={rotate_n_s2} "
            f"per_iter={per_iter_bytes_s2/1e9:.2f}GB total={(per_iter_bytes_s2*rotate_n_s2)/1e9:.2f}GB"
        )
        x2_list = [x2] + [x2.clone() for _ in range(rotate_n_s2 - 1)]
        w2_list = [w2] + [w2.clone() for _ in range(rotate_n_s2 - 1)]
        x2_scale_list = [x2_scale_dev] + [x2_scale_dev.clone() for _ in range(rotate_n_s2 - 1)]
        w2_scale_list = [w2_scale_dev] + [w2_scale_dev.clone() for _ in range(rotate_n_s2 - 1)]
        for i in range(1, rotate_n_s2):
            x2_list[i].bitwise_xor_(torch.tensor(i & 0xFF, dtype=x2.dtype, device=x2.device))
            w2_list[i].bitwise_xor_(torch.tensor((i * 17) & 0xFF, dtype=w2.dtype, device=w2.device))
        torch.cuda.synchronize()
        cur2 = [0]

        def rotate_select_s2(iter_idx):
            cur2[0] = iter_idx % rotate_n_s2 if iter_idx >= 0 else (-iter_idx - 1) % rotate_n_s2

        def launch_stage2(start_ev=None, end_ev=None):
            i = cur2[0]
            k2(y2, x2_list[i], w2_list[i], x2_scale_list[i], w2_scale_list[i],
               masked_m, max_m, model_dim, inter_dim, E,
               stream=torch.cuda.current_stream(),
               _gemm_events=(start_ev, end_ev) if start_ev is not None else None)
    else:
        rotate_select_s2 = None

        def launch_stage2(start_ev=None, end_ev=None):
            k2(y2, x2, w2, x2_scale_dev, w2_scale_dev, masked_m, max_m, model_dim, inter_dim, E,
               stream=torch.cuda.current_stream(),
               _gemm_events=(start_ev, end_ev) if start_ev is not None else None)

    perf_iter_s2 = [0]
    bench_prefix_s2 = _make_m_tile_prefix_for_shape(masked_m, s) if persistent and bench_scope == "gemm" else None
    bench_map_s2 = _make_m_tile_map_for_shape(masked_m, s) if persistent and bench_scope == "gemm" else None

    def launch_stage2_perftest():
        if rotate_select_s2 is not None:
            rotate_select_s2(perf_iter_s2[0])
            perf_iter_s2[0] += 1
        if bench_scope == "gemm":
            if rotate_select_s2 is not None:
                i = perf_iter_s2[0] - 1
                x_arg, w_arg = x2_list[i % rotate_n_s2], w2_list[i % rotate_n_s2]
                x_scale_arg = x2_scale_list[i % rotate_n_s2]
                w_scale_arg = w2_scale_list[i % rotate_n_s2]
            else:
                x_arg, w_arg = x2, w2
                x_scale_arg, w_scale_arg = x2_scale_dev, w2_scale_dev
            k2(y2, x_arg, w_arg, x_scale_arg, w_scale_arg,
               masked_m, max_m, model_dim, inter_dim, E,
               stream=torch.cuda.current_stream(),
               _m_tile_prefix=bench_prefix_s2,
               _m_tile_map=bench_map_s2)
        else:
            launch_stage2()
        return y2

    timings2: dict[str, float] = {}
    if bench_mode in ("event", "both"):
        _log(f"mock moe stage2 persistent={persistent}: event launch warmup={warmup} iters={iters}"
             + (f" rotate=N{rotate_n_s2}" if rotate_select_s2 is not None else ""))
        us2 = _measure_gemm_event_us(launch_stage2, num_warmup=warmup, num_iters=iters,
                                     rotate_select=rotate_select_s2)
        timings2["gemm_event_us"] = us2
        _log(f"mock moe stage2 persistent={persistent}: event launch done gemm_event_us={us2:.2f}"
             + (f" rotate=N{rotate_n_s2}" if rotate_select_s2 is not None else ""))
    if bench_mode in ("run_perftest", "both"):
        _log(f"mock moe stage2 persistent={persistent}: run_perftest warmup={warmup} iters={iters} rotate={rotate_n_s2}")
        us2 = _run_perftest_us(
            launch_stage2_perftest,
            num_warmup=warmup,
            num_iters=iters,
            rotate=rotate_n_s2,
        )
        timings2["run_perftest_us"] = us2
        _log(f"mock moe stage2 persistent={persistent}: run_perftest done us={us2:.2f}")
    if verify:
        y2.fill_(_VERIFY_SENTINEL)
        if rotate_select_s2 is not None:
            cur2[0] = 0
        torch.cuda.synchronize()
        launch_stage2()
        torch.cuda.synchronize()

        expected_topk_out = torch.zeros((tokens, topk, model_dim), device="cuda", dtype=torch.float32)
        active_idx = torch.nonzero(masked_m_cpu > 0, as_tuple=False).flatten().tolist()
        if active_idx:
            active_t = torch.tensor(active_idx, dtype=torch.long)
            actual_active_raw = y2[active_t]
            check_name = f"mock stage2 persistent={persistent}"
            try:
                _kernel_ran_check(
                    check_name,
                    actual_active_raw, masked_m_cpu[active_t],
                )
            except AssertionError as exc:
                _handle_verify_failure(check_name, exc)
            expected_all = _reference_mxfp4_batched(
                x2_raw[active_t], w2_raw[active_t],
                x2_scale_raw[active_t], w2_scale_raw[active_t],
                k=inter_dim,
            ).to(torch.float16).float()  # (A, max_m, model_dim)
            actual_active = actual_active_raw.float()
            try:
                _assert_batched_close(
                    check_name,
                    actual_active, expected_all,
                    masked_m_cpu[active_t],
                )
            except AssertionError as exc:
                _handle_verify_failure(check_name, exc)
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
        check_name = f"mock moe final persistent={persistent}"
        try:
            _assert_valid_rows_close(check_name, out, expected_out)
        except AssertionError as exc:
            _handle_verify_failure(check_name, exc)
    _log(f"mock moe scatter done: topk_out={tuple(topk_out.shape)} out={tuple(out.shape)}")
    return timings1, timings2


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
        route_mode="expert_balance",
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


def _format_timing_summary(timings: dict[str, float]) -> str:
    fields = []
    if "run_perftest_us" in timings:
        fields.append(f"run_perftest_us={timings['run_perftest_us']:.2f}")
    if "gemm_event_us" in timings:
        fields.append(f"gemm_event_us={timings['gemm_event_us']:.2f}")
    return " ".join(fields) if fields else "no_timing"


def main() -> None:
    global DEFAULT_L2_TOL, DEFAULT_L2_ABS_TOL, _VERIFY_WARN_ONLY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("grouped", "moe"), default="grouped")
    parser.add_argument("--stage", choices=("1", "2", "both"), default="both")
    parser.add_argument("--model-dim", type=int, default=7168)
    parser.add_argument("--inter-dim", type=int, default=256)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--max-m", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument(
        "--route-mode",
        choices=("expert_balance", "diagonal", "stride", "hot", "balanced"),
        default="expert_balance",
        help=(
            "expert_balance: perfect 1-to-1 mapping when tokens*topk<=experts "
            "(default). diagonal: lower-triangular -- touches only "
            "tokens+topk-1 experts. 'balanced' kept as deprecated alias of "
            "'diagonal'."
        ),
    )
    parser.add_argument("--tile-m", type=int, default=16)
    parser.add_argument("--tile-n", type=int, default=64)
    parser.add_argument("--tile-k", type=int, default=128)
    parser.add_argument("--m-warp", type=int, default=1)
    parser.add_argument("--n-warp", type=int, default=2)
    parser.add_argument("--num-buffers", type=int, default=2,
                        help="K-tile pipeline depth. Must satisfy K/tile_k >= num_buffers.")
    parser.add_argument("--waves-per-eu", type=int, default=None,
                        help="Override waves-per-EU (default: kernel decides).")
    parser.add_argument("--split-k", type=int, default=1,
                        help="Split K across grid.z and atomic-add partial sums (disables TDM store).")
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
    parser.add_argument("--verify-warn-only", action="store_true",
                        help="Print verify failures as warnings and continue. "
                             "Useful for collecting stage2 diagnostics when stage1 is known-bad.")
    parser.add_argument("--l2-tol", type=float, default=DEFAULT_L2_TOL,
                        help="Relative L2 tolerance for verify (default: %(default)s). "
                             "rel_l2 = ||actual - expected||_2 / ||expected||_2.")
    parser.add_argument("--l2-abs-tol", type=float, default=DEFAULT_L2_ABS_TOL,
                        help="Absolute L2 RMS tolerance used when ||expected||_2 is "
                             "near-zero (default: %(default)s). RMS = "
                             "||actual - expected||_2 / sqrt(N).")
    parser.add_argument("--data", choices=("constant", "pattern"), default="constant")
    parser.add_argument("--masked-mode", choices=("mixed", "full", "descending"), default="mixed")
    parser.add_argument("--masked-m-override", type=int, default=None,
                        help="Force every expert's valid_m to this value (0..max_m). "
                             "Useful for decode (M=1) cases where buffer max_m must be >= tile_m.")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=101)
    parser.add_argument("--bench-mode", choices=("run_perftest", "event", "both"), default="run_perftest",
                        help="Timing path. run_perftest measures wrapper/profile latency; "
                             "event measures GEMM-only cuda.Event timing inside the wrapper.")
    parser.add_argument("--bench-scope", choices=("gemm", "wrapper"), default="gemm",
                        help="Scope for run_perftest timing. gemm precomputes persistent "
                             "metadata and skips stage1 epilogue; wrapper times the full Python wrapper.")
    parser.add_argument("--rotate", type=int, default=0,
                        help="Rotate-buffer count (0=off). Each timed iter uses a "
                             "different on-device weight/activation buffer so L2/HBM "
                             "caches cannot inflate measured BW. Auto-capped to fit "
                             "in free GPU memory.")
    args = parser.parse_args()
    # Honor --l2-tol/--l2-abs-tol globally so existing _assert_* helpers pick
    # them up without threading the values through every wrapper.
    DEFAULT_L2_TOL = float(args.l2_tol)
    DEFAULT_L2_ABS_TOL = float(args.l2_abs_tol)
    _VERIFY_WARN_ONLY = bool(args.verify_warn_only)

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
        split_k=args.split_k,
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
        f"verify={args.verify} data={args.data} masked_mode={args.masked_mode} "
        f"bench_mode={args.bench_mode} bench_scope={args.bench_scope} rotate={args.rotate}",
        flush=True,
    )
    for persistent in persist_modes:
        if args.scenario == "moe":
            t1, t2 = _run_mock_moe(
                s,
                tokens=args.tokens,
                topk=args.topk,
                route_mode=args.route_mode,
                persistent=persistent,
                verify=args.verify,
                data=args.data,
                warmup=args.warmup,
                iters=args.iters,
                rotate=args.rotate,
                bench_mode=args.bench_mode,
                bench_scope=args.bench_scope,
            )
            print(
                f"[masked-grouped-moe-gemm] mock_moe persistent={persistent} "
                f"stage1_{_format_timing_summary(t1)} "
                f"stage2_{_format_timing_summary(t2)}",
                flush=True,
            )
            continue
        if args.stage in ("1", "both"):
            timings = _run_stage1(
                s,
                persistent=persistent,
                verify=args.verify,
                data=args.data,
                masked_mode=args.masked_mode,
                warmup=args.warmup,
                iters=args.iters,
                masked_m_override=args.masked_m_override,
                bench_mode=args.bench_mode,
                bench_scope=args.bench_scope,
                rotate=args.rotate,
            )
            print(f"[masked-grouped-moe-gemm] stage1 persistent={persistent} {_format_timing_summary(timings)}", flush=True)
        if args.stage in ("2", "both"):
            timings = _run_stage2(
                s,
                persistent=persistent,
                verify=args.verify,
                data=args.data,
                masked_mode=args.masked_mode,
                warmup=args.warmup,
                iters=args.iters,
                masked_m_override=args.masked_m_override,
                bench_mode=args.bench_mode,
                bench_scope=args.bench_scope,
                rotate=args.rotate,
            )
            print(f"[masked-grouped-moe-gemm] stage2 persistent={persistent} {_format_timing_summary(timings)}", flush=True)


if __name__ == "__main__":
    main()
