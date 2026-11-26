# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""HIP activation quantization before GEMM1 and between the GEMMs (`aiter.get_hip_quant`).

Covers the quant types whose activations are quantized by a standalone kernel and the
intermediate is quantized after GEMM1 (the generic branches of the legacy
`fused_moe_2stages`).

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
Please do not modify this code without coordinating with the moev2 owners (@qli88). See
aiter/moev2/README.md.
"""

from aiter import QuantType
from aiter import get_hip_quant as get_quant

from ..dispatch.buf import Buf
from ..dispatch.ops import Op
from .base import Cap, Segment

_FEATURES = frozenset(
    {"ep_mask", "num_local_tokens", "doweight_stage1", "act_scales", "a1_scale"}
)


class QuantA1(Segment):
    name = "quant.a1"
    covers = (Op.QUANT_A1, Op.QUANT_A1)
    capabilities = (Cap(features=_FEATURES),)

    def produces(self, p, params):
        return Buf("tokens", p.act_in)

    def run(self, quant_type, hidden_states, a1_scale, q_dtype_a, num_local_tokens):
        """`(a1, a1_scale)`: quantize unless the activations already are `q_dtype_a`."""
        if hidden_states.dtype != q_dtype_a:
            return get_quant(quant_type)(
                hidden_states,
                scale=a1_scale,
                quant_dtype=q_dtype_a,
                num_rows=num_local_tokens,
            )
        assert (
            a1_scale is not None or quant_type == QuantType.No
        ), "a1_scale must be provided for quantized input for fused_moe"
        return hidden_states, a1_scale


class QuantInter(Segment):
    name = "quant.inter"
    covers = (Op.QUANT_INTER, Op.QUANT_INTER)
    capabilities = (Cap(features=_FEATURES),)

    def produces(self, p, params):
        return Buf("inter", p.act_in)

    def run(
        self,
        quant_type,
        a2,
        a2_scale,
        q_dtype_a,
        num_local_tokens,
        token_num,
        topk,
        inter_dim,
    ):
        a2, a2_scale = get_quant(quant_type)(
            a2,
            scale=a2_scale,
            quant_dtype=q_dtype_a,
            num_rows=num_local_tokens,
            num_rows_factor=topk,
        )
        return a2.view(token_num, topk, inter_dim), a2_scale
