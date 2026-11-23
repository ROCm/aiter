# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AttnRes for training: FlyDSL forward and backward on gfx1250.

``flydsl_attn_res_train(query, bank, prefix, rms_weight, output_rms_weight)``
returns the mixed residual of one AttnRes site and is differentiable in every
tensor argument. The candidates are the ``K`` banked block representations
``bank[:, 0..K)`` followed by the intra-block ``prefix``; ``bank`` may be a
``[:, :K]`` slice of a preallocated ``[N, K_max, H]`` buffer.
"""

from functools import cache

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

from .kernels.attn_res_train import build_attn_res_bwd, build_attn_res_fwd
from .kernels.tensor_shim import _run_compiled

SUPPORTED_GFX = ("gfx1250",)
# Buffer loads and stores take a 32-bit byte offset from the tensor's base.
_BUFFER_LIMIT_BYTES = 1 << 32
_BWD_GROUPS_PER_CU = 4

_launchers: dict = {}


def _launcher(key, build):
    launcher = _launchers.get(key)
    if launcher is None:
        launcher = _launchers[key] = build()
    return launcher


@cache
def _num_cus(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _check_rows(tensor: torch.Tensor, name: str, hidden_size: int) -> None:
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"{name} must be bfloat16, got {tensor.dtype}")
    if tensor.shape[-1] != hidden_size or tensor.stride(-1) != 1:
        raise ValueError(f"{name} rows must be contiguous with {hidden_size} elements")
    # Each row is read in 128-bit tiles.
    if tensor.data_ptr() % 16 or any(s % 8 for s in tensor.stride()[:-1]):
        raise ValueError(f"{name} rows must be 16-byte aligned")
    extent = sum((d - 1) * s for d, s in zip(tensor.shape, tensor.stride())) + 1
    if tensor.numel() and extent * tensor.element_size() > _BUFFER_LIMIT_BYTES:
        raise ValueError(f"{name} spans more than {_BUFFER_LIMIT_BYTES} bytes")


def _forward(query, bank, prefix, rms_weight, output_rms_weight, eps, out_eps, scale):
    num_tokens, hidden_size = prefix.shape
    num_bank = 0 if bank is None else bank.shape[1]
    device = prefix.device
    output_norm = output_rms_weight is not None
    qw = (query.reshape(-1).float() * rms_weight.reshape(-1).float()).contiguous()
    w_out = output_rms_weight.reshape(-1).float().contiguous() if output_norm else qw
    y = torch.empty((num_tokens, hidden_size), device=device, dtype=prefix.dtype)
    stats = torch.empty(
        (2, num_tokens, num_bank + 1), device=device, dtype=torch.float32
    )
    logit, rstd = stats[0], stats[1]
    lse = torch.empty(num_tokens, device=device, dtype=torch.float32)
    out_rstd = torch.empty_like(lse) if output_norm else lse
    bank_arg = bank if num_bank else prefix
    key = (
        "fwd",
        device.index,
        hidden_size,
        num_bank,
        output_norm,
        eps,
        out_eps,
        scale,
        bank_arg.stride(),
        prefix.stride(),
    )
    launcher = _launcher(
        key,
        lambda: build_attn_res_fwd(
            hidden_size, num_bank, output_norm, eps, out_eps, scale
        ),
    )
    if num_tokens:
        _run_compiled(
            launcher,
            bank_arg,
            prefix,
            qw,
            w_out,
            y,
            logit,
            rstd,
            lse,
            out_rstd,
            num_tokens,
            torch._C._cuda_getCurrentRawStream(device.index),
        )
    return y, qw, w_out, logit, rstd, lse, out_rstd


def _backward(
    dy, bank, prefix, qw, w_out, logit, rstd, lse, out_rstd, output_norm, scale
):
    num_tokens, hidden_size = prefix.shape
    num_bank = 0 if bank is None else bank.shape[1]
    device = prefix.device
    if dy.stride(-1) != 1 or dy.data_ptr() % 16 or dy.stride(0) % 8:
        dy = dy.contiguous()
    dbank = (
        torch.empty(
            (num_tokens, num_bank, hidden_size), device=device, dtype=bank.dtype
        )
        if num_bank
        else None
    )
    dprefix = torch.empty((num_tokens, hidden_size), device=device, dtype=prefix.dtype)
    groups = max(1, min(num_tokens, _BWD_GROUPS_PER_CU * _num_cus(device.index)))
    part_dqw = torch.empty((groups, hidden_size), device=device, dtype=torch.float32)
    part_dwo = torch.empty_like(part_dqw) if output_norm else part_dqw
    bank_arg = bank if num_bank else prefix
    dbank_arg = dbank if num_bank else dprefix
    key = (
        "bwd",
        device.index,
        hidden_size,
        num_bank,
        output_norm,
        scale,
        dy.stride(),
        bank_arg.stride(),
        prefix.stride(),
        dbank_arg.stride(),
    )
    launcher = _launcher(
        key, lambda: build_attn_res_bwd(hidden_size, num_bank, output_norm, scale)
    )
    if num_tokens:
        _run_compiled(
            launcher,
            dy,
            bank_arg,
            prefix,
            qw,
            w_out,
            logit,
            rstd,
            lse,
            out_rstd,
            dbank_arg,
            dprefix,
            part_dqw,
            part_dwo,
            num_tokens,
            groups,
            torch._C._cuda_getCurrentRawStream(device.index),
        )
    # With one candidate the scores get no gradient and the kernel writes no dqw.
    if num_tokens and num_bank:
        dqw = part_dqw.sum(0)
    else:
        dqw = torch.zeros(hidden_size, device=device, dtype=torch.float32)
    if output_norm:
        dwo = part_dwo.sum(0) if num_tokens else torch.zeros_like(dqw)
    else:
        dwo = None
    return dbank, dprefix, dqw, dwo


class _AttnResTrain(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, query, bank, prefix, rms_weight, output_rms_weight, eps, out_eps, scale
    ):
        y, qw, w_out, logit, rstd, lse, out_rstd = _forward(
            query, bank, prefix, rms_weight, output_rms_weight, eps, out_eps, scale
        )
        ctx.save_for_backward(
            query,
            bank,
            prefix,
            rms_weight,
            output_rms_weight,
            qw,
            w_out,
            logit,
            rstd,
            lse,
            out_rstd,
        )
        ctx.scale = scale
        return y

    @staticmethod
    def backward(ctx, dy):
        (
            query,
            bank,
            prefix,
            rms_weight,
            output_rms_weight,
            qw,
            w_out,
            logit,
            rstd,
            lse,
            out_rstd,
        ) = ctx.saved_tensors
        output_norm = output_rms_weight is not None
        dbank, dprefix, dqw, dwo = _backward(
            dy,
            bank,
            prefix,
            qw,
            w_out,
            logit,
            rstd,
            lse,
            out_rstd,
            output_norm,
            ctx.scale,
        )
        dquery = (
            (dqw * rms_weight.reshape(-1).float()).to(query.dtype).view(query.shape)
        )
        drms = (
            (dqw * query.reshape(-1).float())
            .to(rms_weight.dtype)
            .view(rms_weight.shape)
        )
        dout = (
            dwo.to(output_rms_weight.dtype).view(output_rms_weight.shape)
            if output_norm
            else None
        )
        return dquery, dbank, dprefix, drms, dout, None, None, None


def flydsl_attn_res_train(
    query: torch.Tensor,
    bank: torch.Tensor | None,
    prefix: torch.Tensor,
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor | None = None,
    eps: float = 1e-6,
    output_eps: float | None = None,
    scale: float = 1.0,
) -> torch.Tensor:
    """One AttnRes site with a FlyDSL forward and backward.

    Key parameters:
    - query: ``[H]`` or ``[1, H]`` pseudo-query (``*_res_proj.weight``).
    - bank: ``[N, K, H]`` banked candidates, or ``None`` for ``K = 0``.
    - prefix: ``[N, H]`` intra-block candidate, mixed last.
    - rms_weight: ``[H]`` candidate RMSNorm weight (``*_res_norm.weight``).
    - output_rms_weight: optional ``[H]`` weight of the RMSNorm that follows the
      site (``input_layernorm`` / ``post_attention_layernorm`` / ``norm``).
    - eps / output_eps: the two RMSNorm epsilons; ``output_eps`` defaults to ``eps``.
    - scale: multiplies the logits before the softmax.

    Returns ``y`` of shape ``[N, H]`` in the candidates' dtype.
    """
    if get_gfx_runtime() not in SUPPORTED_GFX:
        raise RuntimeError(
            f"flydsl_attn_res_train supports {SUPPORTED_GFX}, got {get_gfx_runtime()}"
        )
    if prefix.dim() != 2:
        raise ValueError(f"prefix must be [N, H], got {tuple(prefix.shape)}")
    num_tokens, hidden_size = prefix.shape
    _check_rows(prefix, "prefix", hidden_size)
    if bank is not None:
        if bank.dim() != 3 or bank.shape[0] != num_tokens:
            raise ValueError(
                f"bank must be [{num_tokens}, K, {hidden_size}], got {tuple(bank.shape)}"
            )
        if bank.dtype != prefix.dtype:
            raise TypeError("bank and prefix must share a dtype")
        _check_rows(bank, "bank", hidden_size)
        if bank.shape[1] == 0:
            bank = None
    for name, weight in (
        ("query", query),
        ("rms_weight", rms_weight),
        ("output_rms_weight", output_rms_weight),
    ):
        if weight is not None and weight.numel() != hidden_size:
            raise ValueError(
                f"{name} must have {hidden_size} elements, got {weight.numel()}"
            )
    out_eps = eps if output_eps is None else output_eps
    return _AttnResTrain.apply(
        query,
        bank,
        prefix,
        rms_weight,
        output_rms_weight,
        float(eps),
        float(out_eps),
        float(scale),
    )
