# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""AITER fused_moe vs FlyDSL warp-decode MoE benchmark (FP8 block-scale, gfx950).

Compares full MoE-block paths on shared weights/topk inputs:

  1. AITER       : fused_topk + aiter.fused_moe(QuantType.per_1x128)
     (internally: moe_sorting + per-1x128 FP8 activation quant + fmoe_fp8_blockscale_g1u1)

  2. FLYDSL-BF16 : fused_topk + flydsl_warp_decode_gate_up (block2d) +
                   flydsl_warp_decode_down_reduce (block2d)

  3. FLYDSL-FP8  : fused_topk + per-1x128 FP8 quant + flydsl_warp_decode_gate_up_fp8act
                   + flydsl_warp_decode_down_reduce (block2d)

  4. FLYDSL-FP4  : fused_topk + flydsl_warp_decode_gate_up_fp4 (E8M0 block 1x32) +
                   flydsl_warp_decode_down_reduce_fp4

Headline: AITER default vs FlyDSL. Default regime is COLD (disjoint-expert
router rotation). --regime warm keeps a fixed fused_topk router.

Sweeps DeepSeek-V3-like (HIDDEN=7168, INTER=2048) and MiniMax-like
(HIDDEN=3072, INTER=1536) shapes for B in {1,2,4,8,16,32,64} with E=256,
TOPK=8. Reports total us, per-stage us (topk, quant, gate_up, down_reduce),
and GB/s per path, plus a correctness column vs. torch_moe_blockscale.

Run (flydsl_venv, GPU 1):
    HIP_VISIBLE_DEVICES=1 ./flydsl_venv/bin/python tickets/667/bench/bench_moe_warp_decode.py
    HIP_VISIBLE_DEVICES=1 ./flydsl_venv/bin/python tickets/667/bench/bench_moe_warp_decode.py --iters 30 --warmup 3 --shapes deepseek --csv tickets/667/bench/bench.csv

Requires aiter, flydsl, and a gfx950 GPU.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from einops import rearrange  # noqa: E402

from aiter import QuantType, get_hip_quant, dtypes  # noqa: E402
from aiter.fused_moe import fused_topk, fused_moe  # noqa: E402
from aiter.test_common import run_perftest  # noqa: E402
from aiter import pertoken_quant  # noqa: E402
from aiter.ops.quant import per_1x32_f4_quant  # noqa: E402
from aiter.ops.shuffle import shuffle_weight  # noqa: E402
from aiter.jit.utils.chip_info import get_gfx  # noqa: E402
from einops import rearrange  # noqa: E402

from aiter.ops.flydsl import (  # noqa: E402
    flydsl_warp_decode_down_reduce,
    flydsl_warp_decode_down_reduce_fp4,
    flydsl_warp_decode_gate_up,
    flydsl_warp_decode_gate_up_fp4,
    flydsl_warp_decode_gate_up_fp8act,
)

# ---------------------------------------------------------------------------
# Shapes & config
# ---------------------------------------------------------------------------


@dataclass
class ShapeCfg:
    name: str
    HIDDEN: int
    INTER: int
    E: int = 256
    TOPK: int = 8


SHAPES = {
    "deepseek": ShapeCfg("deepseek-v3", HIDDEN=7168, INTER=2048),
    "minimax": ShapeCfg("minimax", HIDDEN=3072, INTER=1536),
}

DEFAULT_BATCHES = (1, 2, 4, 8, 16, 32, 64)


# ---------------------------------------------------------------------------
# Weight / scale preparation
# ---------------------------------------------------------------------------

BLOCK = 128
_FLYDSL_SCALE_BLOCK = (BLOCK, BLOCK)  # FlyDSL default-vs-default: w_scale_mode=block2d
_MXFP4_SCALE_BLOCK = (1, 32)  # MXFP4 spec; E8M0 convert operand (plan ?2)
HEADLINE_PATHS = (
    "aiter",
    "flydsl_bf16",
    "flydsl_fp8",
    "flydsl_fp4",
)
PATH_CONFIG = {
    "aiter": "fused_moe QuantType.per_1x128 shuffle_weight(16,16)",
    "flydsl_bf16": "flydsl_warp_decode_gate_up/down block2d (128,128) kernel defaults",
    "flydsl_fp8": "flydsl_warp_decode_gate_up_fp8act + down block2d (128,128) kernel defaults",
    "flydsl_fp4": "flydsl_*_fp4 scale_block=(1,32) E8M0 convert; kernel defaults (kvector=8)",
}
FP8_DTYPE = dtypes.fp8  # on gfx950 this is torch.float8_e4m3fn (OCP)
BF16 = torch.bfloat16
F32 = torch.float32
I32 = torch.int32


def _block_quant(w_bf16: torch.Tensor, block_n: int = BLOCK, block_k: int = BLOCK):
    """Block-[block_n x block_k] FP8 quantize a 3D weight tensor [E, N, K].

    Returns (w_fp8[E,N,K], w_scale[E, N/block_n, K/block_k]).
    """
    assert w_bf16.dim() == 3
    E, N, K = w_bf16.shape
    assert (
        N % block_n == 0 and K % block_k == 0
    ), f"weight shape {w_bf16.shape} not divisible by block {(block_n, block_k)}"
    tmp = rearrange(
        w_bf16.view(E, N // block_n, block_n, K // block_k, block_k),
        "e nbn bn nbk bk -> e nbn nbk (bn bk)",
    ).contiguous()
    w_q, w_scale = pertoken_quant(tmp, quant_dtype=FP8_DTYPE)
    w_q = rearrange(
        w_q.view(E, N // block_n, K // block_k, block_n, block_k),
        "e nbn nbk bn bk -> e (nbn bn) (nbk bk)",
    ).contiguous()
    w_scale = w_scale.view(E, N // block_n, K // block_k).contiguous()
    return w_q, w_scale


def _mxfp4_pack(w_bf16: torch.Tensor):
    """Pack [E, N, K] bf16 -> MXFP4 uint8 [E,N,K//2] + E8M0 [E*N, K/32], shuffle=False."""
    E, N, K = w_bf16.shape
    assert K % 32 == 0, f"K={K} must be divisible by 32 for MXFP4"
    q, s = per_1x32_f4_quant(w_bf16.reshape(E * N, K), shuffle=False)
    packed = q.view(torch.uint8).reshape(E, N, K // 2).contiguous()
    scale = s.view(torch.uint8).reshape(E * N, K // 32).contiguous()
    return packed, scale


@dataclass
class WeightPack:
    shape: ShapeCfg

    # BF16 weights (kept for reference / torch_moe_blockscale)
    w_gate_bf16: torch.Tensor  # [E, INTER, HIDDEN]
    w_up_bf16: torch.Tensor
    w_down_bf16: torch.Tensor  # [E, HIDDEN, INTER]

    # Per-block FP8 weights (gate/up split)
    w_gate_fp8: torch.Tensor  # [E, INTER, HIDDEN] fp8
    w_up_fp8: torch.Tensor
    w_down_fp8: torch.Tensor  # [E, HIDDEN, INTER] fp8

    # Warp-decode / FlyDSL block2d scales. Storage is row-major over
    # (row-block, K-block): sidx = (e*N + j)//128 * (K/128) + k//128, which is
    # exactly this 2D view flattened. FP32 (not E8M0) -- FlyDSL FP8 kernels
    # pass scale=1 to cvt_scalef32 and fold these after dot2.
    w_gate_scale_wd: torch.Tensor  # [E*INTER/128, HIDDEN/128] fp32
    w_up_scale_wd: torch.Tensor
    w_down_scale_wd: torch.Tensor  # [E*HIDDEN/128, INTER/128] fp32

    # AITER fused-MoE tensors (gate|up interleaved along N axis)
    w1_fp8: torch.Tensor  # [E, 2*INTER, HIDDEN] fp8
    w1_scale: torch.Tensor  # [E, 2*INTER/128, HIDDEN/128] fp32
    w2_fp8: torch.Tensor  # [E, HIDDEN, INTER] fp8
    w2_scale: torch.Tensor  # [E, HIDDEN/128, INTER/128] fp32

    # MXFP4 + E8M0 (FlyDSL fp4 path). Packed 2 FP4/byte; scales are E8M0 bytes
    # for convert's scale operand (plan ?2), Block2D<1,32>.
    w_gate_fp4: torch.Tensor  # [E, INTER, HIDDEN//2] uint8
    w_up_fp4: torch.Tensor
    w_down_fp4: torch.Tensor  # [E, HIDDEN, INTER//2] uint8
    w_gate_scale_fp4: torch.Tensor  # [E*INTER, HIDDEN/32] uint8 E8M0
    w_up_scale_fp4: torch.Tensor
    w_down_scale_fp4: torch.Tensor  # [E*HIDDEN, INTER/32] uint8 E8M0


def build_weights(shape: ShapeCfg, device: str = "cuda", seed: int = 123) -> WeightPack:
    torch.manual_seed(seed)
    HIDDEN, INTER, E = shape.HIDDEN, shape.INTER, shape.E
    assert (
        INTER % BLOCK == 0 and HIDDEN % BLOCK == 0
    ), f"INTER={INTER}, HIDDEN={HIDDEN} must both be divisible by {BLOCK}"

    w_gate_bf16 = torch.randn(E, INTER, HIDDEN, dtype=BF16, device=device) / 10.0
    w_up_bf16 = torch.randn(E, INTER, HIDDEN, dtype=BF16, device=device) / 10.0
    w_down_bf16 = torch.randn(E, HIDDEN, INTER, dtype=BF16, device=device) / 10.0

    w_gate_fp8, w_gate_scale = _block_quant(w_gate_bf16)
    w_up_fp8, w_up_scale = _block_quant(w_up_bf16)
    w_down_fp8, w_down_scale = _block_quant(w_down_bf16)

    # [E, N/128, K/128] -> [E*N/128, K/128]: FlyDSL block2d (BN,BK)=(128,128)
    # indexes this as a row-major (row-block, K-block) vector.
    w_gate_scale_wd = w_gate_scale.reshape(
        E * (INTER // BLOCK), HIDDEN // BLOCK
    ).contiguous()
    w_up_scale_wd = w_up_scale.reshape(
        E * (INTER // BLOCK), HIDDEN // BLOCK
    ).contiguous()
    w_down_scale_wd = w_down_scale.reshape(
        E * (HIDDEN // BLOCK), INTER // BLOCK
    ).contiguous()

    # AITER wants gate/up concatenated along N axis:
    #   w1[E, 2*INTER, HIDDEN] with matching w1_scale[E, 2*INTER/128, HIDDEN/128].
    # fused_moe's per_1x128 2-stage path expects shuffled weights (16,16 layout),
    # same as test_moe_blockscale.py::asm_moe_test.
    w1_unsh = torch.cat([w_gate_fp8, w_up_fp8], dim=1).contiguous()
    w1_fp8 = shuffle_weight(w1_unsh, (16, 16))
    w1_scale = torch.cat([w_gate_scale, w_up_scale], dim=1).contiguous()
    w2_fp8 = shuffle_weight(w_down_fp8.contiguous(), (16, 16))
    w2_scale = w_down_scale.contiguous()

    w_gate_fp4, w_gate_scale_fp4 = _mxfp4_pack(w_gate_bf16)
    w_up_fp4, w_up_scale_fp4 = _mxfp4_pack(w_up_bf16)
    w_down_fp4, w_down_scale_fp4 = _mxfp4_pack(w_down_bf16)

    return WeightPack(
        shape=shape,
        w_gate_bf16=w_gate_bf16,
        w_up_bf16=w_up_bf16,
        w_down_bf16=w_down_bf16,
        w_gate_fp8=w_gate_fp8,
        w_up_fp8=w_up_fp8,
        w_down_fp8=w_down_fp8,
        w_gate_scale_wd=w_gate_scale_wd,
        w_up_scale_wd=w_up_scale_wd,
        w_down_scale_wd=w_down_scale_wd,
        w1_fp8=w1_fp8,
        w1_scale=w1_scale,
        w2_fp8=w2_fp8,
        w2_scale=w2_scale,
        w_gate_fp4=w_gate_fp4,
        w_up_fp4=w_up_fp4,
        w_down_fp4=w_down_fp4,
        w_gate_scale_fp4=w_gate_scale_fp4,
        w_up_scale_fp4=w_up_scale_fp4,
        w_down_scale_fp4=w_down_scale_fp4,
    )


# ---------------------------------------------------------------------------
# Per-path wrappers
# ---------------------------------------------------------------------------

_hip_quant_per_1x128 = get_hip_quant(QuantType.per_1x128)


def aiter_moe_core(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    wp: WeightPack,
) -> torch.Tensor:
    return fused_moe(
        hidden_states,
        wp.w1_fp8,
        wp.w2_fp8,
        topk_weights,
        topk_ids,
        quant_type=QuantType.per_1x128,
        w1_scale=wp.w1_scale.view(wp.shape.E, -1),
        w2_scale=wp.w2_scale.view(wp.shape.E, -1),
    )


def aiter_moe_block(
    hidden_states: torch.Tensor, gating: torch.Tensor, wp: WeightPack
) -> torch.Tensor:
    topk_weights, topk_ids = fused_topk(hidden_states, gating, wp.shape.TOPK, True)
    return aiter_moe_core(hidden_states, topk_weights, topk_ids, wp)


def flydsl_fp8_moe_block(
    hidden_states: torch.Tensor,
    gating: torch.Tensor,
    wp: WeightPack,
) -> torch.Tensor:
    topk_weights, topk_ids = fused_topk(hidden_states, gating, wp.shape.TOPK, True)
    x_fp8, x_scale = _hip_quant_per_1x128(hidden_states, quant_dtype=FP8_DTYPE)

    B = hidden_states.shape[0]
    inter = torch.empty(
        (B, wp.shape.TOPK, wp.shape.INTER), dtype=BF16, device=hidden_states.device
    )
    router_ids = topk_ids.to(I32).contiguous()
    _flydsl_gate_up_fp8act(
        x_fp8,
        x_scale,
        wp.w_gate_fp8,
        wp.w_gate_scale_wd,
        wp.w_up_fp8,
        wp.w_up_scale_wd,
        router_ids,
        inter,
    )
    y = torch.empty((B, wp.shape.HIDDEN), dtype=BF16, device=hidden_states.device)
    _flydsl_down(
        inter,
        wp.w_down_fp8,
        wp.w_down_scale_wd,
        router_ids,
        topk_weights.to(F32).contiguous(),
        y,
    )
    return y


def flydsl_bf16_moe_block(
    hidden_states: torch.Tensor,
    gating: torch.Tensor,
    wp: WeightPack,
) -> torch.Tensor:
    topk_weights, topk_ids = fused_topk(hidden_states, gating, wp.shape.TOPK, True)

    B = hidden_states.shape[0]
    inter = torch.empty(
        (B, wp.shape.TOPK, wp.shape.INTER), dtype=BF16, device=hidden_states.device
    )
    router_ids = topk_ids.to(I32).contiguous()
    _flydsl_gate_up_bf16(
        hidden_states,
        wp.w_gate_fp8,
        wp.w_gate_scale_wd,
        wp.w_up_fp8,
        wp.w_up_scale_wd,
        router_ids,
        inter,
    )
    y = torch.empty((B, wp.shape.HIDDEN), dtype=BF16, device=hidden_states.device)
    _flydsl_down(
        inter,
        wp.w_down_fp8,
        wp.w_down_scale_wd,
        router_ids,
        topk_weights.to(F32).contiguous(),
        y,
    )
    return y


def _flydsl_block2d_scale(scale: torch.Tensor) -> torch.Tensor:
    """FlyDSL block2d scale pointer: 1D f32, row-major (row-block, K-block).

    ``w_*_scale_wd`` is already that layout as ``[E*N/128, K/128]``; flatten so
    the kernel's ``sidx = row_blk * (K/128) + k_blk`` matches storage. These
    values are arbitrary f32 from ``pertoken_quant``, not E8M0 -- the FP8
    kernels pass ``scale=1`` into ``cvt_scalef32_pk_bf16_fp8`` and fold *this*
    tensor after dot2 (plan ?2). Do not recode as E8M0 on this path.
    """
    assert scale.dtype == F32, f"block2d weight scale must be fp32, got {scale.dtype}"
    return scale.reshape(-1).contiguous()


def _flydsl_gate_up_bf16(x, w_gate, w_gate_scale, w_up, w_up_scale, router_ids, out):
    """Adapter for flydsl_warp_decode_gate_up (block2d)."""
    assert router_ids.dtype == I32, f"router_ids must be int32, got {router_ids.dtype}"
    return flydsl_warp_decode_gate_up(
        x,
        w_gate,
        w_up,
        router_ids,
        _flydsl_block2d_scale(w_gate_scale),
        _flydsl_block2d_scale(w_up_scale),
        w_scale_mode="block2d",
        scale_block=_FLYDSL_SCALE_BLOCK,
        out=out,
    )


def _flydsl_gate_up_fp8act(
    x_fp8, x_scale, w_gate, w_gate_scale, w_up, w_up_scale, router_ids, out
):
    """Adapter for flydsl_warp_decode_gate_up_fp8act."""
    assert router_ids.dtype == I32, f"router_ids must be int32, got {router_ids.dtype}"
    # hip per-1x128 scale is [B, HIDDEN/128]; FlyDSL wants [B * HIDDEN/128]
    # over (token, K-block). Same row-major flatten as weight block2d.
    return flydsl_warp_decode_gate_up_fp8act(
        x_fp8,
        w_gate,
        w_up,
        router_ids,
        x_scale.reshape(-1).contiguous(),
        _flydsl_block2d_scale(w_gate_scale),
        _flydsl_block2d_scale(w_up_scale),
        scale_block=_FLYDSL_SCALE_BLOCK,
        out=out,
    )


def _flydsl_down(inter, w_down, w_down_scale, router_ids, router_wts, out):
    """Adapter for flydsl_warp_decode_down_reduce (block2d)."""
    assert router_ids.dtype == I32, f"router_ids must be int32, got {router_ids.dtype}"
    assert router_wts.dtype == F32, f"router_wts must be fp32, got {router_wts.dtype}"
    return flydsl_warp_decode_down_reduce(
        inter,
        w_down,
        router_ids,
        router_wts,
        _flydsl_block2d_scale(w_down_scale),
        w_scale_mode="block2d",
        scale_block=_FLYDSL_SCALE_BLOCK,
        out=out,
    )


def _flydsl_gate_up_fp4(x, w_gate, w_gate_scale, w_up, w_up_scale, router_ids, out):
    """Adapter for flydsl_warp_decode_gate_up_fp4 (E8M0 Block2D<1,32>)."""
    assert router_ids.dtype == I32, f"router_ids must be int32, got {router_ids.dtype}"
    assert w_gate_scale.dtype == torch.uint8 and w_up_scale.dtype == torch.uint8
    return flydsl_warp_decode_gate_up_fp4(
        x,
        w_gate,
        w_up,
        router_ids,
        w_gate_scale.contiguous(),
        w_up_scale.contiguous(),
        scale_block=_MXFP4_SCALE_BLOCK,
        out=out,
    )


def _flydsl_down_fp4(inter, w_down, w_down_scale, router_ids, router_wts, out):
    """Adapter for flydsl_warp_decode_down_reduce_fp4 (E8M0 Block2D<1,32>)."""
    assert router_ids.dtype == I32, f"router_ids must be int32, got {router_ids.dtype}"
    assert router_wts.dtype == F32, f"router_wts must be fp32, got {router_wts.dtype}"
    assert w_down_scale.dtype == torch.uint8
    return flydsl_warp_decode_down_reduce_fp4(
        inter,
        w_down,
        router_ids,
        router_wts,
        w_down_scale.contiguous(),
        scale_block=_MXFP4_SCALE_BLOCK,
        out=out,
    )


def flydsl_fp4_moe_block(
    hidden_states: torch.Tensor, gating: torch.Tensor, wp: WeightPack
) -> torch.Tensor:
    topk_weights, topk_ids = fused_topk(hidden_states, gating, wp.shape.TOPK, True)
    B = hidden_states.shape[0]
    inter = torch.empty(
        (B, wp.shape.TOPK, wp.shape.INTER), dtype=BF16, device=hidden_states.device
    )
    router_ids = topk_ids.to(I32).contiguous()
    _flydsl_gate_up_fp4(
        hidden_states,
        wp.w_gate_fp4,
        wp.w_gate_scale_fp4,
        wp.w_up_fp4,
        wp.w_up_scale_fp4,
        router_ids,
        inter,
    )
    y = torch.empty((B, wp.shape.HIDDEN), dtype=BF16, device=hidden_states.device)
    _flydsl_down_fp4(
        inter,
        wp.w_down_fp4,
        wp.w_down_scale_fp4,
        router_ids,
        topk_weights.to(F32).contiguous(),
        y,
    )
    return y


# ---------------------------------------------------------------------------
# Torch reference (correctness)
# ---------------------------------------------------------------------------


def torch_moe_blockscale_ref(
    hidden_states: torch.Tensor, gating: torch.Tensor, wp: WeightPack
) -> torch.Tensor:
    topk_weights, topk_ids = fused_topk(hidden_states, gating, wp.shape.TOPK, True)
    E, INTER, HIDDEN = wp.shape.E, wp.shape.INTER, wp.shape.HIDDEN
    B = hidden_states.shape[0]
    compute = F32

    x = hidden_states.to(compute)
    w_gate = wp.w_gate_fp8.to(compute)
    w_up = wp.w_up_fp8.to(compute)
    w_down = wp.w_down_fp8.to(compute)

    def _apply_scale_3d(w_q: torch.Tensor, scale: torch.Tensor):
        # scale: [E, N/128, K/128] -> broadcast to [E, N, K]
        return w_q * scale.repeat_interleave(BLOCK, dim=1).repeat_interleave(
            BLOCK, dim=2
        )

    w_gate = _apply_scale_3d(
        w_gate, wp.w_gate_scale_wd.view(E, INTER // BLOCK, HIDDEN // BLOCK)
    )
    w_up = _apply_scale_3d(
        w_up, wp.w_up_scale_wd.view(E, INTER // BLOCK, HIDDEN // BLOCK)
    )
    w_down = _apply_scale_3d(
        w_down, wp.w_down_scale_wd.view(E, HIDDEN // BLOCK, INTER // BLOCK)
    )

    out = torch.zeros(B, wp.shape.TOPK, HIDDEN, dtype=compute, device=x.device)
    for b in range(B):
        for k in range(wp.shape.TOPK):
            e = topk_ids[b, k].item()
            g = x[b] @ w_gate[e].t()
            u = x[b] @ w_up[e].t()
            inter = F.silu(g) * u
            out[b, k] = inter @ w_down[e].t()
    return (out * topk_weights.view(B, -1, 1).to(compute)).sum(dim=1).to(BF16)


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------


@dataclass
class StageTimings:
    topk_us: float = 0.0
    quant_us: float = 0.0  # per-1x128 FP8 activation quant (AITER / FlyDSL-FP8)
    gate_up_us: float = 0.0
    down_us: float = 0.0
    total_us: float = 0.0  # full path incl topk
    core_us: float = 0.0  # full path excluding topk
    err: Optional[dict] = None  # correctness (optional)


def _router_group_list(B: int, E: int, TOPK: int, device) -> list[torch.Tensor]:
    """Disjoint-expert router groups (FlyDSL / G9 cold harness).

    ``rotate = ceil(E / (B*TOPK))`` groups tile the pool; each launch reads
    ``B*TOPK`` distinct experts (``rids[i] = i % E`` within a group).
    """
    bk = B * TOPK
    rotate = max(1, (E + bk - 1) // bk)
    rid_list = []
    for g in range(rotate):
        flat = (g * bk + torch.arange(bk, device=device)) % E
        rid_list.append(flat.to(I32).view(B, TOPK).contiguous())
    return rid_list


def _time(
    func: Callable, *args, iters: int, warmup: int, num_rotate_args: int = 0, **kwargs
) -> tuple:
    out, us = run_perftest(
        func,
        *args,
        num_iters=iters,
        num_warmup=warmup,
        num_rotate_args=num_rotate_args,
        **kwargs,
    )
    return out, us


def _time_rotated(entry_fn: Callable, rid_list: list[torch.Tensor], iters, warmup):
    """Rotate router ids in a closure; weights stay captured (no deep-copy)."""
    state = {"i": 0}

    def fn():
        rid = rid_list[state["i"] % len(rid_list)]
        state["i"] += 1
        return entry_fn(rid)

    return _time(fn, iters=iters, warmup=warmup, num_rotate_args=1)


def bench_aiter(
    hidden_states,
    gating,
    wp,
    iters,
    warmup,
    cold: bool = False,
    rid_list: Optional[list] = None,
    router_wts: Optional[torch.Tensor] = None,
) -> StageTimings:
    tt = StageTimings()

    _, tt.topk_us = _time(
        lambda h, g: fused_topk(h, g, wp.shape.TOPK, True),
        hidden_states,
        gating,
        iters=iters,
        warmup=warmup,
    )

    # Quant cost (what fused_moe does internally before the 1-stage ASM kernel).
    _, tt.quant_us = _time(
        lambda h: _hip_quant_per_1x128(h, quant_dtype=FP8_DTYPE),
        hidden_states,
        iters=iters,
        warmup=warmup,
    )

    if cold:
        wts = router_wts
        _, tt.core_us = _time_rotated(
            lambda rid: aiter_moe_core(hidden_states, wts, rid, wp),
            rid_list,
            iters,
            warmup,
        )
        tt.total_us = tt.core_us + tt.topk_us
    else:

        def path(h, g):
            return aiter_moe_block(h, g, wp)

        _, tt.total_us = _time(path, hidden_states, gating, iters=iters, warmup=warmup)
        tt.core_us = tt.total_us - tt.topk_us
    return tt


def bench_flydsl_fp8(
    hidden_states,
    gating,
    wp,
    iters,
    warmup,
    cold: bool = False,
    rid_list: Optional[list] = None,
    router_wts: Optional[torch.Tensor] = None,
) -> StageTimings:
    tt = StageTimings()
    gate_up_func = _flydsl_gate_up_fp8act
    down_func = _flydsl_down

    _, tt.topk_us = _time(
        lambda h, g: fused_topk(h, g, wp.shape.TOPK, True),
        hidden_states,
        gating,
        iters=iters,
        warmup=warmup,
    )

    _, tt.quant_us = _time(
        lambda h: _hip_quant_per_1x128(h, quant_dtype=FP8_DTYPE),
        hidden_states,
        iters=iters,
        warmup=warmup,
    )

    topk_weights, topk_ids = fused_topk(hidden_states, gating, wp.shape.TOPK, True)
    x_fp8, x_scale = _hip_quant_per_1x128(hidden_states, quant_dtype=FP8_DTYPE)
    B = hidden_states.shape[0]
    inter = torch.empty(
        (B, wp.shape.TOPK, wp.shape.INTER), dtype=BF16, device=hidden_states.device
    )
    y = torch.empty((B, wp.shape.HIDDEN), dtype=BF16, device=hidden_states.device)
    if router_wts is None:
        router_wts = topk_weights.to(F32).contiguous()
    router_ids = topk_ids.to(I32).contiguous()

    if cold:

        def core(rid):
            xq, xs = _hip_quant_per_1x128(hidden_states, quant_dtype=FP8_DTYPE)
            gate_up_func(
                xq,
                xs,
                wp.w_gate_fp8,
                wp.w_gate_scale_wd,
                wp.w_up_fp8,
                wp.w_up_scale_wd,
                rid,
                inter,
            )
            down_func(
                inter,
                wp.w_down_fp8,
                wp.w_down_scale_wd,
                rid,
                router_wts,
                y,
            )
            return y

        _, tt.core_us = _time_rotated(core, rid_list, iters, warmup)
        tt.total_us = tt.core_us + tt.topk_us
        _, tt.gate_up_us = _time_rotated(
            lambda rid: gate_up_func(
                x_fp8,
                x_scale,
                wp.w_gate_fp8,
                wp.w_gate_scale_wd,
                wp.w_up_fp8,
                wp.w_up_scale_wd,
                rid,
                inter,
            ),
            rid_list,
            iters,
            warmup,
        )
        _, tt.down_us = _time_rotated(
            lambda rid: down_func(
                inter,
                wp.w_down_fp8,
                wp.w_down_scale_wd,
                rid,
                router_wts,
                y,
            ),
            rid_list,
            iters,
            warmup,
        )
    else:

        def path(h, g):
            return flydsl_fp8_moe_block(h, g, wp)

        _, tt.total_us = _time(path, hidden_states, gating, iters=iters, warmup=warmup)
        _, tt.gate_up_us = _time(
            gate_up_func,
            x_fp8,
            x_scale,
            wp.w_gate_fp8,
            wp.w_gate_scale_wd,
            wp.w_up_fp8,
            wp.w_up_scale_wd,
            router_ids,
            inter,
            iters=iters,
            warmup=warmup,
        )
        _, tt.down_us = _time(
            down_func,
            inter,
            wp.w_down_fp8,
            wp.w_down_scale_wd,
            router_ids,
            router_wts,
            y,
            iters=iters,
            warmup=warmup,
        )
        tt.core_us = tt.total_us - tt.topk_us
    return tt


def bench_flydsl_bf16(
    hidden_states,
    gating,
    wp,
    iters,
    warmup,
    cold: bool = False,
    rid_list: Optional[list] = None,
    router_wts: Optional[torch.Tensor] = None,
) -> StageTimings:
    tt = StageTimings()
    gate_up_func = _flydsl_gate_up_bf16
    down_func = _flydsl_down

    _, tt.topk_us = _time(
        lambda h, g: fused_topk(h, g, wp.shape.TOPK, True),
        hidden_states,
        gating,
        iters=iters,
        warmup=warmup,
    )
    tt.quant_us = 0.0

    topk_weights, topk_ids = fused_topk(hidden_states, gating, wp.shape.TOPK, True)
    B = hidden_states.shape[0]
    inter = torch.empty(
        (B, wp.shape.TOPK, wp.shape.INTER), dtype=BF16, device=hidden_states.device
    )
    y = torch.empty((B, wp.shape.HIDDEN), dtype=BF16, device=hidden_states.device)
    if router_wts is None:
        router_wts = topk_weights.to(F32).contiguous()
    router_ids = topk_ids.to(I32).contiguous()

    if cold:

        def core(rid):
            gate_up_func(
                hidden_states,
                wp.w_gate_fp8,
                wp.w_gate_scale_wd,
                wp.w_up_fp8,
                wp.w_up_scale_wd,
                rid,
                inter,
            )
            down_func(
                inter,
                wp.w_down_fp8,
                wp.w_down_scale_wd,
                rid,
                router_wts,
                y,
            )
            return y

        _, tt.core_us = _time_rotated(core, rid_list, iters, warmup)
        tt.total_us = tt.core_us + tt.topk_us
        _, tt.gate_up_us = _time_rotated(
            lambda rid: gate_up_func(
                hidden_states,
                wp.w_gate_fp8,
                wp.w_gate_scale_wd,
                wp.w_up_fp8,
                wp.w_up_scale_wd,
                rid,
                inter,
            ),
            rid_list,
            iters,
            warmup,
        )
        _, tt.down_us = _time_rotated(
            lambda rid: down_func(
                inter,
                wp.w_down_fp8,
                wp.w_down_scale_wd,
                rid,
                router_wts,
                y,
            ),
            rid_list,
            iters,
            warmup,
        )
    else:

        def path(h, g):
            return flydsl_bf16_moe_block(h, g, wp)

        _, tt.total_us = _time(path, hidden_states, gating, iters=iters, warmup=warmup)
        _, tt.gate_up_us = _time(
            gate_up_func,
            hidden_states,
            wp.w_gate_fp8,
            wp.w_gate_scale_wd,
            wp.w_up_fp8,
            wp.w_up_scale_wd,
            router_ids,
            inter,
            iters=iters,
            warmup=warmup,
        )
        _, tt.down_us = _time(
            down_func,
            inter,
            wp.w_down_fp8,
            wp.w_down_scale_wd,
            router_ids,
            router_wts,
            y,
            iters=iters,
            warmup=warmup,
        )
        tt.core_us = tt.total_us - tt.topk_us
    return tt


def bench_flydsl_fp4(
    hidden_states,
    gating,
    wp,
    iters,
    warmup,
    cold: bool = False,
    rid_list: Optional[list] = None,
    router_wts: Optional[torch.Tensor] = None,
) -> StageTimings:
    tt = StageTimings()
    _, tt.topk_us = _time(
        lambda h, g: fused_topk(h, g, wp.shape.TOPK, True),
        hidden_states,
        gating,
        iters=iters,
        warmup=warmup,
    )
    tt.quant_us = 0.0
    topk_weights, topk_ids = fused_topk(hidden_states, gating, wp.shape.TOPK, True)
    B = hidden_states.shape[0]
    inter = torch.empty(
        (B, wp.shape.TOPK, wp.shape.INTER), dtype=BF16, device=hidden_states.device
    )
    y = torch.empty((B, wp.shape.HIDDEN), dtype=BF16, device=hidden_states.device)
    if router_wts is None:
        router_wts = topk_weights.to(F32).contiguous()
    router_ids = topk_ids.to(I32).contiguous()

    if cold:

        def core(rid):
            _flydsl_gate_up_fp4(
                hidden_states,
                wp.w_gate_fp4,
                wp.w_gate_scale_fp4,
                wp.w_up_fp4,
                wp.w_up_scale_fp4,
                rid,
                inter,
            )
            _flydsl_down_fp4(
                inter,
                wp.w_down_fp4,
                wp.w_down_scale_fp4,
                rid,
                router_wts,
                y,
            )
            return y

        _, tt.core_us = _time_rotated(core, rid_list, iters, warmup)
        tt.total_us = tt.core_us + tt.topk_us
        _, tt.gate_up_us = _time_rotated(
            lambda rid: _flydsl_gate_up_fp4(
                hidden_states,
                wp.w_gate_fp4,
                wp.w_gate_scale_fp4,
                wp.w_up_fp4,
                wp.w_up_scale_fp4,
                rid,
                inter,
            ),
            rid_list,
            iters,
            warmup,
        )
        _, tt.down_us = _time_rotated(
            lambda rid: _flydsl_down_fp4(
                inter,
                wp.w_down_fp4,
                wp.w_down_scale_fp4,
                rid,
                router_wts,
                y,
            ),
            rid_list,
            iters,
            warmup,
        )
    else:

        def path(h, g):
            return flydsl_fp4_moe_block(h, g, wp)

        _, tt.total_us = _time(path, hidden_states, gating, iters=iters, warmup=warmup)
        _, tt.gate_up_us = _time(
            _flydsl_gate_up_fp4,
            hidden_states,
            wp.w_gate_fp4,
            wp.w_gate_scale_fp4,
            wp.w_up_fp4,
            wp.w_up_scale_fp4,
            router_ids,
            inter,
            iters=iters,
            warmup=warmup,
        )
        _, tt.down_us = _time(
            _flydsl_down_fp4,
            inter,
            wp.w_down_fp4,
            wp.w_down_scale_fp4,
            router_ids,
            router_wts,
            y,
            iters=iters,
            warmup=warmup,
        )
        tt.core_us = tt.total_us - tt.topk_us
    return tt


def path_bytes(B: int, shape: ShapeCfg, path: str) -> float:
    """Approximate bytes moved for the full MoE block for a given path.

    Reads:
      - gate_up: B*TOPK*INTER activations read from hidden_states (x once per top_k
        output) + 2 * B*TOPK*INTER*HIDDEN weight bytes (gate & up)
      - down   : B*TOPK*INTER intermediate + B*TOPK*INTER*HIDDEN w_down bytes
    Writes:
      - intermediate [B,TOPK,INTER] bf16
      - y [B, HIDDEN] bf16

    Per-element bytes: fp8=1, bf16=2, fp32 scales ignored (negligible).
    """
    HIDDEN, INTER, TOPK = shape.HIDDEN, shape.INTER, shape.TOPK
    x_elem = 1 if path in ("flydsl_fp8", "aiter") else 2
    w_elem = 0.5 if path.startswith("flydsl_fp4") else 1
    inter_elem = 2
    y_elem = 2

    # gate_up: x read per output element (with kVector loads the compiler reuses
    # across the two matmuls, but it is still two reads effectively per hidden elt)
    gate_up_x = B * TOPK * INTER * HIDDEN * x_elem
    gate_up_w = 2.0 * B * TOPK * INTER * HIDDEN * w_elem
    gate_up_y = B * TOPK * INTER * inter_elem

    down_x = B * TOPK * INTER * inter_elem
    down_w = B * TOPK * INTER * HIDDEN * w_elem
    down_y = B * HIDDEN * y_elem

    return gate_up_x + gate_up_w + gate_up_y + down_x + down_w + down_y


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def _provenance(args) -> str:
    import flydsl

    cold = args.regime == "cold"
    regime = (
        "COLD (disjoint-expert router rotation, rotate=ceil(E/(B*TOPK)); "
        "num_rotate_args=1, weights captured in closure)"
        if cold
        else "WARM-ish (single WeightPack, fixed fused_topk router)"
    )
    return "\n".join(
        [
            (
                "SILOTIGER-667 full-MoE comparison (Phase E)"
                if cold
                else "SILOTIGER-667 full-MoE comparison (Phase D warm)"
            ),
            f"regime: {regime}",
            f"gfx: {get_gfx()}",
            f"HIP_VISIBLE_DEVICES: {os.environ.get('HIP_VISIBLE_DEVICES')}",
            f"python: {sys.executable}",
            f"flydsl: {getattr(flydsl, '__version__', '?')}",
            f"iters={args.iters} warmup={args.warmup} batches={list(args.batches)} shapes={list(args.shapes)}",
            "policy: default-vs-default (each path uses its own shipped defaults)",
            "path configs:",
            *[f"  {p}: {PATH_CONFIG[p]}" for p in HEADLINE_PATHS],
            "CK warp-decode backend: removed (AITER vs FlyDSL only)",
        ]
    )


def _write_headline_md(out: Path, provenance: str, rows: list[dict]) -> None:
    """Pivot headline paths: one row per shape x B."""
    keys = sorted({(r["shape"], r["B"]) for r in rows})
    by = {(r["shape"], r["B"], r["path"]): r for r in rows}
    cols = [p for p in HEADLINE_PATHS if any(r["path"] == p for r in rows)]
    lines = [
        "# Full-MoE AITER vs FlyDSL",
        "",
        "```",
        provenance,
        "```",
        "",
    ]
    hdr = ["shape", "B"] + [f"{p} core_us" for p in cols] + [f"{p} cos" for p in cols]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("| " + " | ".join(["---"] * len(hdr)) + " |")
    for shape, B in keys:
        rec = [str(shape), str(B)]
        for p in cols:
            r = by.get((shape, B, p))
            rec.append(f"{r['core_us']:.1f}" if r else "-")
        for p in cols:
            r = by.get((shape, B, p))
            rec.append(
                f"{r['cos_sim']:.4f}" if r and r.get("cos_sim") is not None else "-"
            )
        lines.append("| " + " | ".join(rec) + " |")
    lines.append("")
    lines.append(
        "**Notes:** Default regime is COLD (`rotate=ceil(E/(B*TOPK))`, closure-captured "
        "weights, `num_rotate_args=1`). `flydsl_fp4` has no FP8-torch cos (different quant). "
        "B=4 has no cos because correctness only runs for B in {1,2,8}."
    )
    out.write_text("\n".join(lines) + "\n")


def sweep(args):
    device = "cuda"
    shapes = [SHAPES[k] for k in args.shapes]
    batches = tuple(args.batches)

    rows = []
    prov = _provenance(args)
    print(prov)
    print()

    header = (
        f"{'shape':<14} {'B':>4} {'path':<12} "
        f"{'total_us':>10} {'topk_us':>9} {'quant_us':>10} "
        f"{'gate_up_us':>12} {'down_us':>10} {'core_us':>10} "
        f"{'GB/s':>8} {'ratio':>7} {'err/cos':>14}"
    )
    print(header)
    print("-" * len(header))

    for shape in shapes:
        wp = build_weights(shape, device=device)
        torch.cuda.synchronize()

        for B in batches:
            torch.manual_seed(1000 + B)
            hidden_states = (
                torch.randn(B, shape.HIDDEN, dtype=BF16, device=device) * 0.1
            )
            gating = torch.randn(B, shape.E, dtype=BF16, device=device)

            iters, warmup = args.iters, args.warmup
            cold = args.regime == "cold"
            rid_list = _router_group_list(B, shape.E, shape.TOPK, hidden_states.device)
            topk_weights, _topk_ids = fused_topk(
                hidden_states, gating, shape.TOPK, True
            )
            router_wts = topk_weights.to(F32).contiguous()
            cold_kw = dict(cold=cold, rid_list=rid_list, router_wts=router_wts)

            path_specs = {
                "flydsl_bf16": (
                    bench_flydsl_bf16,
                    {},
                    lambda h, g: flydsl_bf16_moe_block(h, g, wp),
                ),
                "flydsl_fp8": (
                    bench_flydsl_fp8,
                    {},
                    lambda h, g: flydsl_fp8_moe_block(h, g, wp),
                ),
                "flydsl_fp4": (
                    bench_flydsl_fp4,
                    {},
                    lambda h, g: flydsl_fp4_moe_block(h, g, wp),
                ),
            }

            path_results: dict[str, StageTimings] = {}

            path_results["aiter"] = bench_aiter(
                hidden_states, gating, wp, iters, warmup, **cold_kw
            )
            for name, (bench_func, kwargs, _) in path_specs.items():
                path_results[name] = bench_func(
                    hidden_states, gating, wp, iters, warmup, **kwargs, **cold_kw
                )

            # Correctness: only for the smaller batches (torch ref is O(B*TOPK*E))
            if args.correctness and B in (1, 2, 8):
                ref = torch_moe_blockscale_ref(hidden_states, gating, wp)
                y_a = aiter_moe_block(hidden_states, gating, wp)
                path_results["aiter"].err = _err_metrics(ref, y_a)
                for name, (_, _, path_func) in path_specs.items():
                    if name == "flydsl_fp4":
                        continue  # MXFP4 weights; not the FP8 torch_moe_blockscale ref
                    path_results[name].err = _err_metrics(
                        ref, path_func(hidden_states, gating)
                    )

            # Ratio = slowest total / current total (so 1.0 = slowest)
            slowest = max(p.total_us for p in path_results.values())

            for name, tt in path_results.items():
                bytes_ = path_bytes(B, shape, name)
                gbs = bytes_ / (tt.core_us * 1e3) if tt.core_us > 0 else 0.0
                ratio = slowest / tt.total_us if tt.total_us > 0 else 0.0
                if tt.err is not None:
                    err_str = (
                        f"{tt.err['err_ratio']:.3f}/" f"{tt.err['cosine_sim']:.4f}"
                    )
                else:
                    err_str = "     -"
                print(
                    f"{shape.name:<14} {B:>4} {name:<12} "
                    f"{tt.total_us:>10.1f} {tt.topk_us:>9.1f} {tt.quant_us:>10.1f} "
                    f"{tt.gate_up_us:>12.1f} {tt.down_us:>10.1f} {tt.core_us:>10.1f} "
                    f"{gbs:>8.1f} {ratio:>7.2f} {err_str:>14}"
                )
                rows.append(
                    {
                        "shape": shape.name,
                        "HIDDEN": shape.HIDDEN,
                        "INTER": shape.INTER,
                        "E": shape.E,
                        "TOPK": shape.TOPK,
                        "B": B,
                        "path": name,
                        "total_us": tt.total_us,
                        "topk_us": tt.topk_us,
                        "quant_us": tt.quant_us,
                        "gate_up_us": tt.gate_up_us,
                        "down_us": tt.down_us,
                        "core_us": tt.core_us,
                        "gb_per_s": gbs,
                        "ratio_vs_slowest": ratio,
                        "err_ratio": tt.err["err_ratio"] if tt.err else None,
                        "max_abs_err": tt.err["max_abs_err"] if tt.err else None,
                        "cos_sim": tt.err["cosine_sim"] if tt.err else None,
                        "regime": "cold" if cold else "warm-ish",
                        "rotate": len(rid_list),
                        "config": PATH_CONFIG.get(name, ""),
                    }
                )
            print()

    if args.csv:
        out = Path(args.csv)
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote CSV: {out}")
    if args.md:
        md = Path(args.md)
        _write_headline_md(md, prov, rows)
        print(f"Wrote markdown: {md}")


def _err_metrics(a: torch.Tensor, b: torch.Tensor, rtol=1e-2, atol=1e-2) -> dict:
    a_f = a.to(F32).flatten()
    b_f = b.to(F32).flatten()
    close = torch.isclose(a_f, b_f, rtol=rtol, atol=atol)
    err_ratio = 1.0 - close.float().mean().item()
    max_abs_err = (a_f - b_f).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        a_f.unsqueeze(0), b_f.unsqueeze(0)
    ).item()
    return dict(err_ratio=err_ratio, max_abs_err=max_abs_err, cosine_sim=cos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shapes", nargs="+", default=list(SHAPES.keys()), choices=list(SHAPES.keys())
    )
    ap.add_argument("--batches", type=int, nargs="+", default=list(DEFAULT_BATCHES))
    ap.add_argument(
        "--iters", type=int, default=50, help="perftest iterations per timed call"
    )
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--csv", type=str, default=None, help="optional CSV output path")
    ap.add_argument("--md", type=str, default=None, help="headline markdown table path")
    ap.add_argument(
        "--regime",
        choices=["cold", "warm"],
        default="cold",
        help="cold: disjoint-expert router rotation (decode HBM); warm: fixed fused_topk",
    )
    ap.add_argument(
        "--no-correctness",
        dest="correctness",
        action="store_false",
        default=True,
        help="skip torch_moe_blockscale correctness check",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No GPU available")

    sweep(args)


if __name__ == "__main__":
    main()
