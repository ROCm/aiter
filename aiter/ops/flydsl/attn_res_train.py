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

import flydsl.compiler as flyc
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime, get_lds_capacity_bytes

from .kernels.attn_res_train import (
    build_attn_res_bwd,
    build_attn_res_fwd,
    build_attn_res_weight_grads,
    bwd_lds_bytes,
)
from .kernels.tensor_shim import _run_compiled

SUPPORTED_GFX = ("gfx1250",)
# Buffer loads and stores take a 32-bit byte offset from the tensor's base.
_BUFFER_LIMIT_BYTES = 1 << 32
# Both kernels are persistent and size their grid to the workgroups that fit at
# once. HIP cannot load a module to ask while a graph is captured; at Kimi-K3's
# width two workgroups fill a WGP.
_FALLBACK_BLOCKS_PER_CU = 2

_launchers: dict = {}


def _launcher(key, build):
    launcher = _launchers.get(key)
    if launcher is None:
        launcher = _launchers[key] = build()
    return launcher


@cache
def _num_cus(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _resident_groups(launcher, args, num_tokens: int, device_index: int) -> int:
    """Workgroups that run at once, cached on the launcher; at most one per token."""
    per_cu = getattr(launcher, "_blocks_per_cu", None)
    if per_cu is None:
        if torch.cuda.is_current_stream_capturing():
            per_cu = _FALLBACK_BLOCKS_PER_CU
        else:
            # Meta tensors make flyc.compile compile only, without launching.
            meta = [a.to("meta") if isinstance(a, torch.Tensor) else a for a in args]
            try:
                per_cu = flyc.compile(launcher, *meta).max_blocks_per_cu()
            except (RuntimeError, ValueError):
                per_cu = _FALLBACK_BLOCKS_PER_CU
            launcher._blocks_per_cu = per_cu
    return max(1, min(num_tokens, per_cu * _num_cus(device_index)))


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


def _weight_rows(query, rms_weight, output_rms_weight):
    """The weights as contiguous rows in bf16 or fp32, the two dtypes the kernels read.

    Without an output norm its slot holds ``rms_weight``, which the kernels ignore.
    """
    rows = []
    for weight in (query, rms_weight, output_rms_weight):
        if weight is None:
            rows.append(rows[1])
            continue
        row = weight.reshape(-1)
        if row.dtype not in (torch.bfloat16, torch.float32):
            row = row.float()
        rows.append(row.contiguous())
    return tuple(rows), tuple(row.dtype == torch.bfloat16 for row in rows)


def _forward(query, bank, prefix, rms_weight, output_rms_weight, eps, out_eps, scale):
    num_tokens, hidden_size = prefix.shape
    num_bank = 0 if bank is None else bank.shape[1]
    device = prefix.device
    output_norm = output_rms_weight is not None
    weights, weight_bf16 = _weight_rows(query, rms_weight, output_rms_weight)
    y = torch.empty((num_tokens, hidden_size), device=device, dtype=prefix.dtype)
    stats = torch.empty(
        (2, num_tokens, num_bank + 1), device=device, dtype=torch.float32
    )
    logit, rstd = stats[0], stats[1]
    lse = torch.empty(num_tokens, device=device, dtype=torch.float32)
    out_rstd = torch.empty_like(lse) if output_norm else lse
    bank_arg = bank if num_bank else prefix
    bank_strides = tuple(bank.stride()[:2]) if num_bank else (0, 0)
    key = (
        "fwd",
        device.index,
        hidden_size,
        num_bank,
        output_norm,
        eps,
        out_eps,
        scale,
        bank_strides,
        prefix.stride(),
        weight_bf16,
    )
    launcher = _launcher(
        key,
        lambda: build_attn_res_fwd(
            hidden_size,
            num_bank,
            output_norm,
            eps,
            out_eps,
            scale,
            bank_strides,
            prefix.stride(0),
            weight_bf16,
        ),
    )
    if num_tokens:
        args = (bank_arg, prefix, *weights, y, logit, rstd, lse, out_rstd)
        stream = torch._C._cuda_getCurrentRawStream(device.index)
        groups = _resident_groups(
            launcher, (*args, num_tokens, 1, stream), num_tokens, device.index
        )
        _run_compiled(launcher, *args, num_tokens, groups, stream)
    return y, logit, rstd, lse, out_rstd


def _backward(
    dy,
    bank,
    prefix,
    query,
    rms_weight,
    output_rms_weight,
    logit,
    rstd,
    lse,
    out_rstd,
    scale,
):
    """``dbank, dprefix`` and the three weight gradients as rows of the weight dtypes."""
    num_tokens, hidden_size = prefix.shape
    num_bank = 0 if bank is None else bank.shape[1]
    device = prefix.device
    output_norm = output_rms_weight is not None
    weights, weight_bf16 = _weight_rows(query, rms_weight, output_rms_weight)
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
    bank_arg = bank if num_bank else prefix
    dbank_arg = dbank if num_bank else dprefix
    bank_strides = tuple(bank.stride()[:2]) if num_bank else (0, 0)
    key = (
        "bwd",
        device.index,
        hidden_size,
        num_bank,
        output_norm,
        scale,
        dy.stride(),
        bank_strides,
        prefix.stride(),
        dbank_arg.stride(),
        weight_bf16,
    )
    launcher = _launcher(
        key,
        lambda: build_attn_res_bwd(
            hidden_size,
            num_bank,
            output_norm,
            scale,
            bank_strides,
            prefix.stride(0),
            dy.stride(0),
            weight_bf16,
        ),
    )
    args = (dy, bank_arg, prefix, *weights, logit, rstd, lse, out_rstd, dbank_arg)
    stream = torch._C._cuda_getCurrentRawStream(device.index)
    groups = 1
    if num_tokens:
        part = torch.empty((1, hidden_size), device="meta", dtype=torch.float32)
        groups = _resident_groups(
            launcher,
            (*args, dprefix, part, part, num_tokens, 1, stream),
            num_tokens,
            device.index,
        )
    part_dqw = torch.empty((groups, hidden_size), device=device, dtype=torch.float32)
    part_dwo = torch.empty_like(part_dqw) if output_norm else part_dqw
    if num_tokens:
        _run_compiled(
            launcher,
            *args,
            dprefix,
            part_dqw,
            part_dwo,
            num_tokens,
            groups,
            stream,
        )
    q_row, w_row, wo_row = weights
    dquery = torch.empty_like(q_row)
    drms = torch.empty_like(w_row)
    dwo = torch.empty_like(wo_row) if output_norm else dquery
    wgrad = _launcher(
        ("wgrad", device.index, hidden_size, num_bank, output_norm, weight_bf16),
        lambda: build_attn_res_weight_grads(
            hidden_size, num_bank, output_norm, weight_bf16
        ),
    )
    _run_compiled(
        wgrad,
        part_dqw,
        part_dwo,
        q_row,
        w_row,
        dquery,
        drms,
        dwo,
        groups if num_tokens else 0,
        stream,
    )
    return dbank, dprefix, dquery, drms, dwo if output_norm else None


class _AttnResTrain(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, query, bank, prefix, rms_weight, output_rms_weight, eps, out_eps, scale
    ):
        y, logit, rstd, lse, out_rstd = _forward(
            query, bank, prefix, rms_weight, output_rms_weight, eps, out_eps, scale
        )
        ctx.save_for_backward(
            query,
            bank,
            prefix,
            rms_weight,
            output_rms_weight,
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
            logit,
            rstd,
            lse,
            out_rstd,
        ) = ctx.saved_tensors
        dbank, dprefix, dquery, drms, dout = _backward(
            dy,
            bank,
            prefix,
            query,
            rms_weight,
            output_rms_weight,
            logit,
            rstd,
            lse,
            out_rstd,
            ctx.scale,
        )
        dquery = dquery.to(query.dtype).view(query.shape)
        drms = drms.to(rms_weight.dtype).view(rms_weight.shape)
        if dout is not None:
            dout = dout.to(output_rms_weight.dtype).view(output_rms_weight.shape)
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
    num_bank = 0 if bank is None else bank.shape[1]
    lds_bytes = bwd_lds_bytes(hidden_size, num_bank, output_rms_weight is not None)
    if lds_bytes > get_lds_capacity_bytes(get_gfx_runtime()):
        raise ValueError(
            f"the backward stages {num_bank + 1} rows of {hidden_size} in LDS, "
            f"{lds_bytes} bytes, over the {get_gfx_runtime()} capacity"
        )
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
