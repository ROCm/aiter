# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CK 2-stage kernels (`moe_ck2stages_gemm1_*` / `moe_ck2stages_gemm2_*`).

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
`ck_moe_stage1` and `_require_shuffled_gfx950_fp8` moved here from aiter/fused_moe.py
and are shared with the legacy MoE path (re-exported from aiter.fused_moe, run by legacy
MoE calls): changes to them must keep legacy behavior
(`python tests/moe_dispatch/snapshot.py --check`). Please coordinate any change in this
module with the moev2 owners (@qli88). See aiter/moev2/README.md.
"""

import functools

import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.ops.moe_limits import RUNTIME_SILU_LIMIT_QUANT_TYPES, normalize_swiglu_limit

from ..dispatch import kernel_names
from ..dispatch.buf import Buf
from ..dispatch.ops import Op
from ..policy import quant_remap
from ..problem import ScaleSpec, scale_spec
from .base import Cap, Segment


def ck_moe_stage1(
    hidden_states,
    w1,  # [E, inter_dim*2, model_dim]
    w2,  # [E, model_dim, inter_dim]
    sorted_token_ids,  # [max_num_tokens_padded]
    sorted_expert_ids,  # [max_num_m_blocks]
    num_valid_ids,  # [1]
    out,
    topk,
    block_m,
    a1_scale,
    w1_scale,
    kernelName="",
    sorted_weights=None,
    quant_type=aiter.QuantType.No,
    activation=ActivationType.Gelu,
    splitk=1,
    use_non_temporal_load=False,
    dtype=None,
    swiglu_limit=None,
):
    limit = None
    if swiglu_limit and activation == ActivationType.Silu:
        limit = normalize_swiglu_limit(swiglu_limit)
        if limit is not None:
            quant_type = quant_remap.get(quant_type, quant_type)
            w1_dtype = getattr(w1, "dtype", None)
            if quant_type not in RUNTIME_SILU_LIMIT_QUANT_TYPES or (
                w1_dtype != dtypes.fp8
            ):
                raise NotImplementedError(
                    f"CK MoE stage1 cannot apply swiglu_limit={swiglu_limit!r} for "
                    f"{quant_type} with weight dtype {w1_dtype}; only FP8 "
                    "per_1x128 / per_Token / per_Tensor have a clamped CK SiLU "
                    "epilogue (clamp before SiLU)."
                )
            if kernelName and ("Splitk" in kernelName or "_silu_" not in kernelName):
                raise ValueError(
                    "Runtime SiLU clamp requires a non-split CK SiLU kernel"
                )
            splitk = 0
    token_num = hidden_states.shape[0]
    is_splitk = quant_type == QuantType.per_1x128 and splitk > 1
    if is_splitk:
        # CK kernel zeros this buffer via hipMemsetAsync when KBatch > 1
        sorted_size = min(token_num * topk * block_m, sorted_token_ids.shape[0])
        tmp_out = torch.empty(
            (sorted_size, w1.shape[1]), dtype=dtypes.fp32, device=out.device
        )
    else:
        tmp_out = out
    aiter.ck_moe_stage1_fwd(
        hidden_states,
        w1,
        w2,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        tmp_out,
        topk,
        kernelName,
        w1_scale,
        a1_scale,
        block_m,
        sorted_weights,
        quant_type,
        activation,
        splitk if is_splitk else 0,
        use_non_temporal_load,
        out.dtype,
        limit,
    )
    if is_splitk:
        valid_out = tmp_out[: token_num * topk, :]
        if activation == ActivationType.Silu:
            aiter.silu_and_mul(out, valid_out.view(dtypes.fp32))
        elif activation == ActivationType.Gelu:
            aiter.gelu_and_mul(out, valid_out.view(dtypes.fp32))
        else:
            raise ValueError(
                f"Unsupported activation for split-k post-activation: {activation}"
            )
    return out


def _require_shuffled_gfx950_fp8(gfx, q_dtype_w, q_type, weights_shuffled):
    # is_shuffled is true when either weight is tagged. weights_shuffled is
    # the both-weights flag when the caller passed opus_weights_shuffled,
    # which fused_moe and fused_moe_2stages do. gfx is the live device;
    # get_gfx() follows GPU_ARCHS and can hide an actual gfx950 process.
    if (
        gfx == "gfx950"
        and q_dtype_w == dtypes.fp8
        and q_type in (QuantType.per_Tensor, QuantType.per_Token)
        and not weights_shuffled
    ):
        raise RuntimeError(
            "gfx950 fp8 CK MoE reads the shuffle_weight() layout on "
            "both w1 and w2; unshuffled per_Tensor and per_Token "
            "weights are unsupported"
        )


def per_tensor_fp8():
    """fp8 per-tensor `ScaleSpec` of the running arch (e4m3fnuz on gfx942)."""
    return scale_spec(QuantType.per_Tensor, dtypes.fp8)


def a8w8_per_tensor_caps():
    fp8 = frozenset({per_tensor_fp8()})
    return (
        Cap(
            arch=frozenset({"gfx942", "gfx950"}),
            act_in=fp8,
            w1=fp8,
            w2=fp8,
            out_dtype=frozenset({dtypes.bf16, dtypes.fp16}),
            activation=frozenset({"silu", "gelu"}),
            gate_mode=frozenset({"separated"}),
            features=frozenset(
                {
                    "ep_mask",
                    "num_local_tokens",
                    "doweight_stage1",
                    "act_scales",
                    "a1_scale",
                }
            ),
        ),
    )


class CkGemm1(Segment):
    name = "ck.gemm1"
    prefixes = (kernel_names.CK2STAGES,)
    covers = (Op.GEMM1, Op.ACT)

    @property
    def capabilities(self):
        return a8w8_per_tensor_caps()

    def owns(self, kernel) -> bool:
        """An empty kernelName1 runs the CK heuristic kernel."""
        return kernel is None or (
            kernel.family == kernel_names.CK2STAGES and kernel.stage == 1
        )

    def produces(self, p, params):
        return Buf("inter", ScaleSpec(p.out_dtype))

    def build(self, p, params):
        return functools.partial(
            ck_moe_stage1,
            kernelName=params.kernel,
            activation=p.activation.activation,
            quant_type=p.quant_type,
            dtype=p.out_dtype,
            splitk=int(params.ksplit),
            use_non_temporal_load=params.nt,
        )


class CkGemm2(Segment):
    name = "ck.gemm2"
    prefixes = (kernel_names.CK2STAGES,)
    covers = (Op.GEMM2, Op.REDUCE)

    @property
    def capabilities(self):
        return a8w8_per_tensor_caps()

    def owns(self, kernel) -> bool:
        return kernel is None or (
            kernel.family == kernel_names.CK2STAGES and kernel.stage == 2
        )

    def produces(self, p, params):
        return Buf("out", ScaleSpec(p.out_dtype))

    def build(self, p, params):
        return functools.partial(
            aiter.ck_moe_stage2_fwd,
            kernelName=params.kernel,
            activation=p.activation.activation,
            quant_type=p.quant_type,
            use_non_temporal_load=params.nt,
        )
