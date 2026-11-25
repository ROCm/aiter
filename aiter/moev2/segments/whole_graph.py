# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Whole-graph implementations named `impl__<name>__<config>` in tuned tables.

The implementation is looked up in `aiter.fused_moe_registry` (e.g. `flydsl_gfx942`,
registered by aiter.ops.flydsl) and runs the whole MoE from a `FusedMoeRequest`.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
Please do not modify this code without coordinating with the moev2 owners (@qli88). See
aiter/moev2/README.md.
"""

from aiter import ActivationType, dtypes
from aiter.fused_moe_registry import FusedMoeImplResolutionError, resolve_fused_moe_impl
from aiter.ops.flydsl.moe_common import GateMode

from ..dispatch import kernel_names
from ..dispatch.buf import Buf
from ..dispatch.ops import Op
from ..problem import ScaleSpec
from .base import RowRejection, Segment
from .ck import a8w8_per_tensor_caps

__all__ = ["FusedMoeImplResolutionError", "WholeGraph", "resolve_fused_moe_impl"]


class WholeGraph(Segment):
    name = "impl.whole_graph"
    prefixes = (kernel_names.IMPL,)
    covers = (Op.SORT, Op.REDUCE)

    @property
    def capabilities(self):
        return a8w8_per_tensor_caps()

    def produces(self, p, params):
        return Buf("out", ScaleSpec(p.out_dtype))

    def check_row(self, p, row, kernel_name1=""):
        if kernel_name1.startswith("impl__flydsl_"):
            unsupported = None
            if not p.fully_shuffled:
                unsupported = "both w1 and w2 must be marked is_shuffled=True"
            elif "bias1" in p.features or "bias2" in p.features:
                unsupported = "per-expert bias"
            elif "doweight_stage1" in p.features:
                unsupported = "doweight_stage1=True"
            elif p.in_dtype is not None and p.in_dtype != dtypes.bf16:
                unsupported = f"activation dtype {p.in_dtype}"
            elif "scatter" in p.features:
                unsupported = "stage2_scatter"
            elif "act_scales" in p.features:
                unsupported = "prequantized activations"
            elif "num_local_tokens" in p.features:
                unsupported = "num_local_tokens"
            elif p.layout.hidden_pad or p.layout.intermediate_pad:
                unsupported = "hidden/intermediate padding"
            elif p.gate_mode is not GateMode.SEPARATED:
                unsupported = f"gate mode {p.gate_mode.value!r}"
            elif p.activation.activation not in (
                ActivationType.Silu,
                ActivationType.Swiglu,
            ):
                unsupported = f"activation {p.activation.activation}"
            if unsupported is not None:
                return RowRejection(
                    f"[fused_moe] discarding FlyDSL whole-graph config for "
                    f"{p.log_keys()}: unsupported {unsupported}; using default heuristics"
                )
        if p.is_ep:
            return RowRejection(None)
        return None

    def build(self, p, params):
        return params.impl
