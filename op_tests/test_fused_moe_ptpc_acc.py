# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Accuracy test for Qwen3.8 FP8 PTPC ASM-JIT and FlyDSL MoE kernels."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import torch

import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_topk
from aiter.fused_moe_asmjit_aot import fused_moe_asmjit_aot
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.fused_moe_gfx942 import run_flydsl_moe_gfx942
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import checkAllclose


torch.set_default_device("cuda")

ACTIVATION = ActivationType.Silu
QUANT_TYPE = QuantType.per_Token
DTYPE = torch.bfloat16
FP8_DTYPE = torch.float8_e4m3fnuz
DIFF_THR = 0.02
CONFIG_CSV = (
    Path(__file__).resolve().parents[1]
    / "aiter/configs/model_configs/qwen3_8_flash_next_fp8_ptpc_tuned_fmoe.csv"
)

TP_CONFIGS = {
    640: "qwen3_8_flash_next_tp1",
    320: "qwen3_8_flash_next_tp2",
    192: "qwen3_8_flash_next_tp4",
}


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator.item() == 0:
        return 0.0
    similarity = 2 * (x * y).sum() / denominator
    return float(1 - similarity)


def load_selected_cases(
    backends: set[str],
    models: set[str] | None,
    tokens: set[int] | None,
) -> list[dict[str, Any]]:
    cases = []
    with CONFIG_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            inter_dim = int(row["inter_dim"])
            model = TP_CONFIGS.get(inter_dim)
            if model is None or (models is not None and model not in models):
                continue

            token = int(row["token"])
            if tokens is not None and token not in tokens:
                continue

            kernel_name = row["kernelName1"]
            if kernel_name.startswith("fused_moe_asmjit_aot__"):
                backend = "asmjit"
                config_string = kernel_name.split("__", 1)[1]
            elif kernel_name.startswith("impl__flydsl_gfx942__"):
                backend = "flydsl"
                config_string = kernel_name.split("__", 2)[2]
            else:
                continue
            if backend not in backends:
                continue

            cases.append(
                {
                    "model": model,
                    "backend": backend,
                    "config": config_string,
                    "token": token,
                    "hidden_size": int(row["model_dim"]),
                    "inter_dim": inter_dim,
                    "expert": int(row["expert"]),
                    "topk": int(row["topk"]),
                }
            )
    return sorted(
        cases,
        key=lambda case: (
            -case["inter_dim"],
            case["token"],
            case["backend"],
        ),
    )


def get_torch_ref(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    batch_size, hidden_size = hidden_states.shape
    num_experts, n1, _ = w1.shape
    inter_dim = n1 // 2
    output = torch.zeros(
        batch_size,
        hidden_size,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    expert_mask = torch.nn.functional.one_hot(
        topk_ids.to(torch.long), num_classes=num_experts
    ).permute(2, 1, 0)
    for expert_id in range(num_experts):
        topk_slot, token_id = torch.where(expert_mask[expert_id])
        if token_id.numel() == 0:
            continue
        gate_proj = w1[expert_id, :inter_dim].t()
        up_proj = w1[expert_id, inter_dim:].t()
        down_proj = w2[expert_id].t()
        hidden = hidden_states[token_id]
        expert_output = (
            torch.nn.functional.silu(hidden @ gate_proj) * (hidden @ up_proj)
        ) @ down_proj
        output.index_add_(
            0,
            token_id,
            (expert_output * topk_weight[token_id, topk_slot, None]).to(
                output.dtype
            ),
        )
    return output


def quant_expert_weights(w_bf16: torch.Tensor):
    quant = aiter.get_torch_quant(QUANT_TYPE)
    w_fp8, w_scale = quant(w_bf16, quant_dtype=FP8_DTYPE)
    w_ref = (w_fp8.to(DTYPE) * w_scale).to(DTYPE)
    w_kernel = shuffle_weight(w_fp8.clone(), layout=(16, 16))
    return w_kernel, w_scale, w_ref


def build_weights(expert: int, inter_dim: int, hidden_size: int):
    torch.manual_seed(42)
    w1_bf16 = torch.randn(expert, inter_dim * 2, hidden_size, dtype=DTYPE)
    w2_bf16 = torch.randn(expert, hidden_size, inter_dim, dtype=DTYPE)
    w1_kernel, w1_scale, w1_ref = quant_expert_weights(w1_bf16)
    w2_kernel, w2_scale, w2_ref = quant_expert_weights(w2_bf16)
    return w1_kernel, w2_kernel, w1_scale, w2_scale, w1_ref, w2_ref


def build_inputs(token: int, expert: int, topk: int, hidden_size: int):
    torch.manual_seed(token)
    hidden_states = (
        torch.randn(token, hidden_size, dtype=DTYPE, device="cuda") + 1
    ) * 0.001
    score = torch.randn(token, expert, dtype=DTYPE, device="cuda")
    topk_weight, topk_ids = fused_topk(
        hidden_states,
        score,
        topk,
        renormalize=True,
    )
    return hidden_states, topk_weight, topk_ids


def run_backend(
    case: dict[str, Any],
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> torch.Tensor:
    common_args = (
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        ACTIVATION,
        QUANT_TYPE,
        w1_scale,
        w2_scale,
        None,
        None,
        0,
    )
    if case["backend"] == "asmjit":
        return fused_moe_asmjit_aot(
            *common_args,
            config_string=case["config"],
        )
    return run_flydsl_moe_gfx942(
        *common_args,
        config_string=case["config"],
    )


def run_case(
    case: dict[str, Any],
    weights: tuple[torch.Tensor, ...],
) -> dict[str, Any]:
    w1, w2, w1_scale, w2_scale, w1_ref, w2_ref = weights
    hidden_states, topk_weight, topk_ids = build_inputs(
        case["token"],
        case["expert"],
        case["topk"],
        case["hidden_size"],
    )
    ref_out = get_torch_ref(
        hidden_states,
        w1_ref,
        w2_ref,
        topk_weight,
        topk_ids,
    )
    out = run_backend(
        case,
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        w1_scale,
        w2_scale,
    )
    torch.cuda.synchronize()

    assert out is not None, f"{case['backend']} returned no output"
    assert out.shape == ref_out.shape, f"{out.shape=} {ref_out.shape=}"
    assert torch.isfinite(out).all(), f"{case['backend']} output has NaN/Inf"

    mismatch_ratio = checkAllclose(
        ref_out,
        out,
        rtol=1e-2,
        atol=1e-2,
        msg=(
            f"{case['model']} token={case['token']} "
            f"{case['backend']} vs Torch"
        ),
    )
    diff = calc_diff(ref_out, out)
    assert diff <= DIFF_THR, (
        f"{case['model']} token={case['token']} {case['backend']} "
        f"diff={diff:.6f} > {DIFF_THR}"
    )
    aiter.logger.info(
        "%s token=%d backend=%s config=%s diff=%.6f mismatch_ratio=%.6f PASS",
        case["model"],
        case["token"],
        case["backend"],
        case["config"],
        diff,
        mismatch_ratio,
    )
    return {**case, "diff": diff, "mismatch_ratio": mismatch_ratio, "status": "PASS"}


parser = argparse.ArgumentParser(
    description="Qwen3.8 FP8 PTPC ASM-JIT/FlyDSL accuracy test"
)
parser.add_argument(
    "--backend",
    choices=["asmjit", "flydsl"],
    nargs="*",
    default=["asmjit", "flydsl"],
)
parser.add_argument(
    "-m",
    "--model",
    choices=list(TP_CONFIGS.values()),
    nargs="*",
    default=None,
)
parser.add_argument("-t", "--tokenNum", type=int, nargs="*", default=None)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("skip: CUDA is not available")
        sys.exit(0)
    if get_gfx() != "gfx942":
        print(f"skip: unsupported platform {get_gfx()!r}; expected gfx942")
        sys.exit(0)

    args = parser.parse_args()
    cases = load_selected_cases(
        set(args.backend),
        set(args.model) if args.model else None,
        set(args.tokenNum) if args.tokenNum else None,
    )
    if not cases:
        parser.error("no ASM-JIT or FlyDSL rows matched the requested filters")

    all_results = []
    for inter_dim in sorted({case["inter_dim"] for case in cases}, reverse=True):
        group = [case for case in cases if case["inter_dim"] == inter_dim]
        first = group[0]
        weights = build_weights(
            first["expert"], first["inter_dim"], first["hidden_size"]
        )
        for case in group:
            all_results.append(run_case(case, weights))
        del weights
        torch.cuda.empty_cache()

    import pandas as pd

    print(pd.DataFrame(all_results).to_markdown(index=False))
