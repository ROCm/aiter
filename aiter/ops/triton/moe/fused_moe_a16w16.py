# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""BF16 x BF16 Triton stages for aiter.fused_moe's two-stage path.

A tuned fmoe CSV selects these stages per token bucket with kernel names

    triton_moe1_a16w16_bm{BM}_bn{BN}_bk{BK}_gm{G}_w{WARPS}_s{STAGES}[_we{W}]
    triton_moe2_a16w16_bm{BM}_bn{BN}_bk{BK}_gm{G}_w{WARPS}_s{STAGES}[_we{W}]

BM must equal the row's block_m (the moe_sorting block). The stages read the
same plain or shuffle_weight((16, 16)) weights as the CK / FlyDSL a16w16 rows.
"""

import functools
import re

import torch
import triton

from aiter.ops.triton._triton_kernels.moe.fused_moe_a16w16 import (
    _fused_moe_a16w16_stage1_kernel,
    _fused_moe_a16w16_stage2_kernel,
)

_NAME_RE = re.compile(
    r"^triton_moe(?P<stage>[12])_a16w16"
    r"_bm(?P<bm>\d+)_bn(?P<bn>\d+)_bk(?P<bk>\d+)_gm(?P<gm>\d+)"
    r"_w(?P<warps>\d+)_s(?P<stages>\d+)(?:_we(?P<we>\d+))?$"
)


@functools.cache
def parse_triton_a16w16_kernel(name: str) -> dict:
    m = _NAME_RE.match(name)
    if m is None:
        raise ValueError(f"Invalid Triton a16w16 MoE kernel name: {name}")
    cfg = {k: int(v) for k, v in m.groupdict().items() if v is not None}
    cfg.setdefault("we", 0)
    return cfg


def triton_a16w16_kernel_name(stage: int, bm, bn, bk, gm, warps, stages, we=0):
    name = f"triton_moe{stage}_a16w16_bm{bm}_bn{bn}_bk{bk}_gm{gm}_w{warps}_s{stages}"
    return f"{name}_we{we}" if we else name


def _launch_kwargs(cfg):
    kw = {"num_warps": cfg["warps"], "num_stages": cfg["stages"]}
    if cfg["we"]:
        kw["waves_per_eu"] = cfg["we"]
    kw["matrix_instr_nonkdim"] = 16
    return kw


def _check(cfg, block_m, n_dim, k_dim, weight, shuffled):
    if cfg["bm"] != block_m:
        raise ValueError(
            f"Triton a16w16 MoE kernel block_m={cfg['bm']} does not match "
            f"moe_sorting block_m={block_m}"
        )
    if k_dim % cfg["bk"] or n_dim % 16:
        raise ValueError(
            f"Triton a16w16 MoE needs K % BLOCK_K == 0 and N % 16 == 0, got "
            f"N={n_dim} K={k_dim} BLOCK_K={cfg['bk']}"
        )
    if shuffled is None:
        shuffled = bool(getattr(weight, "is_shuffled", False))
    if shuffled and (cfg["bn"] % 16 or cfg["bk"] % 32):
        raise ValueError("shuffled weights need BLOCK_N % 16 == 0, BLOCK_K % 32 == 0")
    return shuffled


def fused_moe_a16w16_stage1(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    out: torch.Tensor,
    topk: int,
    block_m: int,
    kernel_name: str,
    shuffled: bool | None = None,
) -> torch.Tensor:
    """Gate-up GEMM + SiLU-and-mul into out[token, topk_slot, inter_dim].

    w1 is [E, 2 * inter_dim, model_dim] with the gate rows first. The sorted
    buffers come from moe_sorting at block_m, which must equal the BM of
    kernel_name. K must be a multiple of BLOCK_K. shuffled=None reads the
    layout from w1.is_shuffled.
    """
    cfg = parse_triton_a16w16_kernel(kernel_name)
    num_tokens, k_dim = hidden_states.shape
    inter_dim = w1.shape[1] // 2
    shuffled = _check(cfg, block_m, inter_dim, k_dim, w1, shuffled)
    out2d = out.view(-1, inter_dim)
    num_m_blocks = sorted_token_ids.numel() // block_m
    grid = (num_m_blocks * triton.cdiv(inter_dim, cfg["bn"]),)
    _fused_moe_a16w16_stage1_kernel[grid](
        hidden_states,
        w1,
        out2d,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        num_tokens,
        inter_dim,
        k_dim,
        hidden_states.stride(0),
        w1.stride(0),
        w1.stride(1),
        out2d.stride(0),
        TOPK=topk,
        BLOCK_M=block_m,
        BLOCK_N=cfg["bn"],
        BLOCK_K=cfg["bk"],
        GROUP_M=cfg["gm"],
        SHUFFLED=shuffled,
        NUM_M_BLOCKS=num_m_blocks,
        **_launch_kwargs(cfg),
    )
    return out


def fused_moe_a16w16_stage2(
    inter_states: torch.Tensor,
    w2: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    sorted_weights,
    num_valid_ids: torch.Tensor,
    out: torch.Tensor,
    topk: int,
    block_m: int,
    kernel_name: str,
    shuffled: bool | None = None,
) -> torch.Tensor:
    """Down GEMM, scaled by the route weight, atomically added into out[token].

    inter_states is the stage-1 output [token, topk, inter_dim]. out must be
    zeroed by the caller (moe_sorting does it). sorted_weights=None skips the
    route weight. shuffled=None reads the layout from w2.is_shuffled.
    """
    cfg = parse_triton_a16w16_kernel(kernel_name)
    model_dim, inter_dim = w2.shape[1], w2.shape[2]
    shuffled = _check(cfg, block_m, model_dim, inter_dim, w2, shuffled)
    a2d = inter_states.view(-1, inter_dim)
    num_tokens = out.shape[0]
    num_m_blocks = sorted_token_ids.numel() // block_m
    grid = (num_m_blocks * triton.cdiv(model_dim, cfg["bn"]),)
    _fused_moe_a16w16_stage2_kernel[grid](
        a2d,
        w2,
        out,
        sorted_token_ids,
        sorted_expert_ids,
        sorted_weights if sorted_weights is not None else sorted_token_ids,
        num_valid_ids,
        num_tokens,
        model_dim,
        inter_dim,
        a2d.stride(0),
        w2.stride(0),
        w2.stride(1),
        out.stride(0),
        TOPK=topk,
        BLOCK_M=block_m,
        BLOCK_N=cfg["bn"],
        BLOCK_K=cfg["bk"],
        GROUP_M=cfg["gm"],
        SHUFFLED=shuffled,
        MUL_ROUTED_WEIGHT=sorted_weights is not None,
        NUM_M_BLOCKS=num_m_blocks,
        **_launch_kwargs(cfg),
    )
    return out
