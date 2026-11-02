# SPDX-License-Identifier: MIT
"""Hybrid-only linear view: share loaded weights and preserve the native ABI."""

import torch
from torch import nn

from atom.models.deepseek_v4_mono.kernels.ops import projection


def linear_refusal(linear):
    from aiter import QuantType

    checks = [
        (linear.params_dtype == torch.float8_e4m3fn, "not FP8"),
        (linear.quant_type.value == QuantType.per_1x128.value, "not per-1x128"),
        (linear.blockscale_e8m0_scale, "not E8M0"),
        (linear.native_a8_group_rows is None, "native group layout"),
        (not getattr(linear, "use_mxscale_shuffle", False), "MX scale shuffle"),
        (linear.weight_scale.dtype == torch.float8_e8m0fnu, "weight scale dtype"),
    ]
    return next((why for ok, why in checks if not ok), None)


class HybridLinear(nn.Module):
    def __init__(self, native, profile, role):
        super().__init__()
        self.native, self.profile, self.role = native, profile, role

    def forward(self, x, x_scale=None, otype=torch.bfloat16, out=None):
        n = self.native
        if out is not None or otype != torch.bfloat16:
            return n(x, x_scale=x_scale, otype=otype, out=out)
        if x_scale is None:
            from atom.model_ops.linear import per_1x128_e8m0_quant

            x, x_scale = per_1x128_e8m0_quant(
                x,
                quant_dtype=n.params_dtype,
                transpose_scale=True,
                scale=getattr(n, "input_scale", None),
            )
        y = projection(x, n.weight, x_scale, n.weight_scale, self.profile, self.role)
        if n.bias is not None:
            y += n.bias
        if n.is_output_padded:
            y = y[..., : n._output_size_before_padding]
        if n.tp_dim == 1 and n.tp_size > 1 and n.reduce_results:
            from atom.model_ops.communication_op import tensor_model_parallel_all_reduce

            y = tensor_model_parallel_all_reduce(y)
        return y

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.native, name)
