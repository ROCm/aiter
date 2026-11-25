# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""ASM stage-1 kernels (`_ZN5aiter...fmoe_stage1...E`).

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
`asm_stage1` moved here from aiter/fused_moe.py and is shared with the legacy MoE path
(re-exported from aiter.fused_moe, run by legacy MoE calls): changes to it must keep
legacy behavior (`python tests/moe_dispatch/snapshot.py --check`). Please coordinate any
change in this module with the moev2 owners (@qli88). See aiter/moev2/README.md.
"""

import functools

import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.ops.moe_limits import normalize_swiglu_limit

from ..dispatch import kernel_names
from ..dispatch.buf import Buf
from ..dispatch.ops import Op
from ..problem import ScaleSpec
from ..shapes import get_inter_dim
from .base import Segment
from .ck import a8w8_per_tensor_caps


def asm_stage1(
    input,
    w1,
    w2,
    sorted_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,  # [token_num, topk, inter_dim]
    topk,
    block_m: int,
    kernelName: str = "",
    ksplit: int = 0,
    activation=ActivationType.Silu,
    quant_type=QuantType.No,
    a1_scale=None,
    w1_scale=None,
    sorted_weights=None,
    swiglu_limit=None,
):
    if swiglu_limit is not None and activation == ActivationType.Silu:
        swiglu_limit = normalize_swiglu_limit(swiglu_limit)
        if swiglu_limit is not None and ksplit <= 0:
            raise NotImplementedError(
                f"ASM MoE stage1 cannot apply swiglu_limit={swiglu_limit!r} without "
                "split-K; the fused SiLU epilogue does not clamp."
            )
    dtype = dtypes.bf16  # out.dtype, asm only support bf16
    if quant_type != QuantType.per_1x128:
        out = out.view(dtype)
    device = out.device
    token_num, _, _ = out.shape
    E, _model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)

    if quant_type == QuantType.per_Tensor:
        a1_scale = a1_scale.view(1, 1).repeat(token_num, 1)
        w1_scale = w1_scale.view(E, 1).repeat(1, w1.shape[1])
        quant_type = QuantType.per_Token

    tmp_out = out
    if ksplit > 0:
        tmp_out = torch.zeros(
            (token_num, topk, w1.shape[1]),
            dtype=dtypes.fp32,
            device=device,
        ).view(dtype)

    aiter.moe_stage1_g1u1(
        input,
        w1,
        w2,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        tmp_out,
        inter_dim,
        kernelName,
        block_m,
        ksplit=ksplit,
        activation=activation,
        quant_type=quant_type,
        a1_scale=a1_scale,
        w1_scale=w1_scale,
        sorted_weights=sorted_weights,
    )
    if ksplit > 0:
        if activation == ActivationType.Silu:
            # silu_and_mul takes a float32 limit. Keep a positive limit from
            # narrowing to 0 (no clamp) or inf, like the CK launcher does.
            limit = (
                min(max(swiglu_limit, 2.0**-149), torch.finfo(dtypes.fp32).max)
                if swiglu_limit
                else 0.0
            )
            aiter.silu_and_mul(out, tmp_out.view(dtypes.fp32), limit)
        elif activation == ActivationType.Swiglu:
            aiter.swiglu_and_mul(out, tmp_out.view(dtypes.fp32))
        elif activation == ActivationType.Gelu:
            aiter.gelu_and_mul(out, tmp_out.view(dtypes.fp32))
        else:
            raise ValueError(
                f"Unsupported activation for split-k post-activation: {activation}"
            )
    return out


def stage2_block_m(block_m):
    """Round a tuned block_m up to one CK stage 2 supports (16, 32, 64 or 128)."""
    # TODO: remove when stage2 support more size
    tmpList = [16, 32, 64, 128]
    if block_m not in tmpList:
        block_m = ([el for el in tmpList if block_m < el] + [128])[0]
    return block_m


class AsmStage1(Segment):
    name = "asm.stage1"
    prefixes = (kernel_names.ASM,)
    covers = (Op.GEMM1, Op.ACT)

    @property
    def capabilities(self):
        return a8w8_per_tensor_caps()

    def produces(self, p, params):
        return Buf("inter", ScaleSpec(dtypes.bf16))

    def build(self, p, params):
        return functools.partial(
            asm_stage1,
            kernelName=params.kernel,
            activation=p.activation.activation,
            quant_type=p.quant_type,
        )
