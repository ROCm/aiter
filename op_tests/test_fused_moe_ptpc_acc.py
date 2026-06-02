# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""``fused_moe`` FP8 per-token (PTPC) accuracy vs torch MoE reference."""

from __future__ import annotations

import argparse
import sys
from typing import Any, Optional

import torch

import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
from aiter.ops.shuffle import shuffle_weight

torch.set_default_device("cuda")

ACTIVATION = ActivationType.Silu
QUANT_TYPE = QuantType.per_Token
DTYPE = torch.bfloat16
FP8_DTYPE = torch.float8_e4m3fnuz
DIFF_THR = 0.02

# Correctness scope: we only certify fused_moe paths backed by pyhip-compiled .co
# kernels (fmoe_asmjit). Numerical checks here are aligned with pyhip's MoE tests on
# branch moe_prefill_fp8_308:
#   https://github.com/tingqli/pyhip/tree/moe_prefill_fp8_308
# (see tests/contrib/moe/test_moe.py). Other kernel backends are out of scope.

# Qwen3.5 PTPC FP8 model shapes (TP-split inter_dim)
MOE_CONFIGS = [
    {"name": "qwen3_5_35b", "hidden_size": 2048, "inter_dim": 512 // 4, "expert": 257, "topk": 9},
    {"name": "qwen3_5_122b", "hidden_size": 3072, "inter_dim": 1024 // 8, "expert": 257, "topk": 9},
    {"name": "qwen3_5_397b", "hidden_size": 4096, "inter_dim": 128, "expert": 513, "topk": 11},
]

DEFAULT_TOKEN_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]


def div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator.item() == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return float(1 - sim)


def get_torch_ref(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """BF16 MoE reference: dequantized weights, no activation quant."""
    batch_size, hidden_dim = hidden_states.shape
    num_experts, n1, _ = w1.shape
    inter_size = n1 // 2
    out = torch.zeros(
        (batch_size, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )

    expert_mask = torch.nn.functional.one_hot(
        topk_ids.to(dtype=torch.long), num_classes=num_experts
    ).permute(2, 1, 0)

    for expert_idx in range(num_experts):
        idx, top_x = torch.where(expert_mask[expert_idx])
        if idx.numel() == 0:
            continue
        gate_proj = w1[expert_idx, :inter_size].t()
        up_proj = w1[expert_idx, inter_size:].t()
        down_proj = w2[expert_idx].t()
        x = hidden_states[top_x]
        y = (torch.nn.functional.silu(x @ gate_proj) * (x @ up_proj)) @ down_proj
        out.index_add_(0, top_x, (y * topk_weight[top_x, idx, None]).to(out.dtype))
    return out


def quant_expert_weights(w_bf16: torch.Tensor, dtype: torch.dtype):
    torch_quant = aiter.get_torch_quant(QuantType.per_Token)
    w_fp8, w_scale = torch_quant(w_bf16, quant_dtype=dtype)
    w_ref = (w_fp8.to(dtype=w_bf16.dtype) * w_scale).to(dtype=w_bf16.dtype)
    return w_fp8, w_scale, w_ref


def build_fp8_weights(
    expert: int,
    inter_dim: int,
    hidden_size: int,
    seed: int = 42,
):
    torch.manual_seed(seed)
    w1_bf16 = torch.randn(expert, inter_dim * 2, hidden_size, dtype=DTYPE)
    w1_fp8, w1_scale, w1_ref = quant_expert_weights(w1_bf16, FP8_DTYPE)
    w2_bf16 = torch.randn(expert, hidden_size, inter_dim, dtype=DTYPE)
    w2_fp8, w2_scale, w2_ref = quant_expert_weights(w2_bf16, FP8_DTYPE)
    w1_kernel = shuffle_weight(w1_fp8.clone(), layout=(16, 16))
    w2_kernel = shuffle_weight(w2_fp8.clone(), layout=(16, 16))
    return w1_kernel, w2_kernel, w1_scale, w2_scale, w1_ref, w2_ref


def build_inputs(
    token: int,
    expert: int,
    topk: int,
    hidden_size: int,
    seed: int = 0,
):
    torch.manual_seed(seed)
    hidden = (torch.randn(token, hidden_size, dtype=DTYPE) + 1) * 0.001
    topk_weight = torch.randn(token, topk, dtype=torch.float32)
    rep_e = div_up(token * topk, expert)
    topk_ids_1d = torch.ones(rep_e, expert, dtype=torch.int32)
    topk_ids_1d[:, :] = torch.randperm(expert, dtype=torch.int32)
    topk_ids = topk_ids_1d.reshape(-1)[: token * topk].reshape(token, topk)
    return hidden, topk_weight, topk_ids


def test_fused_moe_ptpc_acc(
    token: int,
    hidden_size: int,
    inter_dim: int,
    expert: int,
    topk: int,
    model_name: str,
    w1_kernel: torch.Tensor,
    w2_kernel: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w1_ref: torch.Tensor,
    w2_ref: torch.Tensor,
) -> dict[str, Any]:
    hidden, topk_weight, topk_ids = build_inputs(
        token, expert, topk, hidden_size, seed=token
    )
    ref_out = get_torch_ref(hidden, w1_ref, w2_ref, topk_weight, topk_ids)

    out = fused_moe(
        hidden,
        w1_kernel,
        w2_kernel,
        topk_weight,
        topk_ids,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        quant_type=QUANT_TYPE,
        activation=ACTIVATION,
    )
    torch.cuda.synchronize()

    diff = calc_diff(ref_out, out)
    ok = diff <= DIFF_THR
    status = "PASS" if ok else "FAIL"
    aiter.logger.info(
        "%s token=%d diff=%.6f %s",
        model_name,
        token,
        diff,
        status,
    )
    if not ok:
        aiter.logger.warning(
            "%s token=%d failed: diff=%.6f > %.2f",
            model_name,
            token,
            diff,
            DIFF_THR,
        )
    return {
        "model": model_name,
        "token": token,
        "hidden_size": hidden_size,
        "inter_dim": inter_dim,
        "expert": expert,
        "topk": topk,
        "diff": diff,
        "status": status,
    }


def run_model(cfg: dict[str, Any], token_counts: list[int]) -> list[dict[str, Any]]:
    name = cfg["name"]
    hidden_size = cfg["hidden_size"]
    inter_dim = cfg["inter_dim"]
    expert = cfg["expert"]
    topk = cfg["topk"]

    aiter.logger.info(
        "=== %s hidden=%d inter=%d E=%d topk=%d ===",
        name,
        hidden_size,
        inter_dim,
        expert,
        topk,
    )
    w1_k, w2_k, w1_s, w2_s, w1_ref, w2_ref = build_fp8_weights(
        expert, inter_dim, hidden_size
    )
    results = []
    for token in token_counts:
        try:
            ret = test_fused_moe_ptpc_acc(
                token,
                hidden_size,
                inter_dim,
                expert,
                topk,
                name,
                w1_k,
                w2_k,
                w1_s,
                w2_s,
                w1_ref,
                w2_ref,
            )
            results.append(ret)
            if ret["status"] != "PASS":
                raise AssertionError(
                    f"{name} token={token} diff={ret['diff']:.6f} > {DIFF_THR}"
                )
        except Exception as ex:
            aiter.logger.error("%s token=%d exception: %s", name, token, ex)
            results.append(
                {
                    "model": name,
                    "token": token,
                    "hidden_size": hidden_size,
                    "inter_dim": inter_dim,
                    "expert": expert,
                    "topk": topk,
                    "diff": float("nan"),
                    "status": "ERROR",
                }
            )
            raise
    return results


parser = argparse.ArgumentParser(description="fused_moe FP8 PTPC accuracy test")
parser.add_argument(
    "-t",
    "--tokenNum",
    type=int,
    nargs="*",
    default=DEFAULT_TOKEN_COUNTS,
    help="token batch sizes to test",
)
parser.add_argument(
    "-m",
    "--model",
    type=str,
    nargs="*",
    choices=[c["name"] for c in MOE_CONFIGS],
    default=None,
    help="model config name(s); default: all",
)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("skip: CUDA not available")
        sys.exit(0)

    args = parser.parse_args()
    configs = MOE_CONFIGS
    if args.model is not None:
        configs = [c for c in MOE_CONFIGS if c["name"] in args.model]

    torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=4)

    all_results: list[dict[str, Any]] = []
    failed = False
    for cfg in configs:
        try:
            all_results.extend(run_model(cfg, args.tokenNum))
        except AssertionError:
            failed = True

    if all_results:
        import pandas as pd

        df = pd.DataFrame(all_results)
        aiter.logger.info("summary:\n%s", df.to_markdown(index=False))

    if failed:
        sys.exit(1)
