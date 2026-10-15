# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 MLA gather + FlyDSL ptpc projection."""

import functools

import torch
from torch import Tensor

_KV_C = 512
_KV_PE = 64
_NOPE = 128
_V_DIM = 128


@functools.lru_cache(maxsize=1)
def _projection_runtime():
    """Import the gfx1250 compiler path only when this specialization runs."""
    import flydsl.expr as fx

    from .kernels.gather_kv_b_proj_gfx1250 import (
        launch_gather_kv_b_proj_8wave_gfx1250,
    )
    from .kernels.tensor_shim import ptr_arg

    return launch_gather_kv_b_proj_8wave_gfx1250, ptr_arg, fx


def _cluster_n(m: int, n_heads: int) -> int:
    """Use A/RoPE multicast only after it amortizes cluster synchronization."""
    return 4 if m >= 3072 and n_heads >= 96 and n_heads % 4 == 0 else 1


def _run_projection(
    cache: Tensor,
    indices: Tensor,
    activation_scale: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    k_out: Tensor,
    v_out: Tensor,
    n_heads: int,
    m: int,
    cluster_n: int,
) -> None:
    """Launch the fixed-width indexed MLA WMMA kernel."""
    launch_projection, ptr_arg, fx = _projection_runtime()

    scale_is_ptr = activation_scale.device.type != "cpu"
    scale_ptr = activation_scale if scale_is_ptr else cache
    args = (
        ptr_arg(k_out),
        ptr_arg(v_out),
        ptr_arg(cache),
        ptr_arg(indices),
        ptr_arg(weight),
        ptr_arg(scale_ptr),
        ptr_arg(weight_scale.reshape(-1)),
        cache.shape[0],
        m,
        0.0 if scale_is_ptr else float(activation_scale),
        fx.Stream(torch.cuda.current_stream(device=cache.device)),
    )
    launch_projection(*args, n_heads, int(scale_is_ptr), cluster_n)


def unsupported_reason(
    k_buffer: Tensor,
    kv_proj_weight: Tensor,
    kv_proj_scale: Tensor | None,
    k_prefix: Tensor,
    v_prefix: Tensor,
    *,
    shuffled_kv_cache: bool = False,
    **_,
) -> str | None:
    if shuffled_kv_cache:
        return "shuffled_kv_cache is not supported"
    if k_buffer.dim() != 3 or tuple(k_buffer.shape[1:]) != (1, _KV_C + _KV_PE):
        return f"k_buffer must be [num_blocks, 1, {_KV_C + _KV_PE}]"
    if k_buffer.dtype != torch.float8_e4m3fn:
        return f"k_buffer must be torch.float8_e4m3fn, got {k_buffer.dtype}"
    if not k_buffer.is_contiguous():
        return "k_buffer must be contiguous"
    if kv_proj_weight.dtype != torch.float8_e4m3fn:
        return f"kv_proj_weight must be torch.float8_e4m3fn, got {kv_proj_weight.dtype}"
    if not getattr(kv_proj_weight, "is_shuffled", False):
        return "kv_proj_weight must use the 16x16 preshuffled layout"
    if not kv_proj_weight.is_contiguous():
        return "kv_proj_weight must be contiguous"
    if (
        k_prefix.dim() != 3
        or v_prefix.dim() != 3
        or k_prefix.shape[1] != v_prefix.shape[1]
        or k_prefix.shape[0] != v_prefix.shape[0]
    ):
        return "outputs must be matching [M, heads, width] tensors"
    n_heads = k_prefix.shape[1]
    if n_heads <= 0:
        return "outputs must have a positive head count"
    if k_prefix.shape[2] != _NOPE + _KV_PE or v_prefix.shape[2] != _V_DIM:
        return "output widths must be NoPE+PE=192 and V=128"
    weight_n = n_heads * (_NOPE + _V_DIM)
    if tuple(kv_proj_weight.shape) != (weight_n, _KV_C):
        return f"kv_proj_weight must be [{weight_n}, {_KV_C}]"
    if kv_proj_scale is None or kv_proj_scale.numel() != weight_n:
        return f"kv_proj_scale must have {weight_n} per-output-row elements"
    if kv_proj_scale.dtype != torch.float32 or not kv_proj_scale.is_contiguous():
        return "kv_proj_scale must be contiguous float32"
    if k_prefix.dtype != torch.bfloat16 or v_prefix.dtype != torch.bfloat16:
        return "outputs must both be bfloat16"
    if not k_prefix.is_contiguous() or not v_prefix.is_contiguous():
        return "outputs must be contiguous"
    tensors = (kv_proj_weight, kv_proj_scale, k_prefix, v_prefix)
    if any(t.device != k_buffer.device for t in tensors):
        return "all tensors must be on the cache device"
    return None


def gather_kv_b_proj_flydsl_gfx1250(
    k_buffer: Tensor,
    k_scale: Tensor,
    kv_indptr: Tensor,
    kv_indices: Tensor,
    kv_prefix_sum_context_lens: Tensor,
    kv_proj_weight: Tensor,
    kv_proj_scale: Tensor,
    k_prefix: Tensor,
    v_prefix: Tensor,
    *,
    num_tokens: int | None = None,
    weight_preshuffle: bool = True,
    shuffled_kv_cache: bool = False,
) -> None:
    """Use the dedicated gfx1250 FlyDSL WMMA path for 128+128 MLA prefixes."""
    if not weight_preshuffle:
        raise ValueError(
            "[gfx1250 FlyDSL gather_kv_b_proj] preshuffled weight required"
        )
    total_kv = k_prefix.shape[0]
    m = int(total_kv if num_tokens is None else num_tokens)
    if m < 0:
        raise ValueError(f"num_tokens must be >=0, got {m}")
    if m > total_kv:
        raise ValueError(f"num_tokens={m} exceeds the allocated {total_kv} output rows")
    if m == 0:
        return
    if kv_indices.numel() < m:
        raise ValueError(f"kv_indices has {kv_indices.numel()} entries, need {m}")
    if (
        kv_indices.dtype != torch.int32
        or kv_indices.device != k_buffer.device
        or not kv_indices.is_contiguous()
    ):
        raise ValueError("kv_indices must be contiguous int32 on the cache device")
    if k_scale.numel() != 1 or k_scale.device not in (
        k_buffer.device,
        torch.device("cpu"),
    ):
        raise ValueError("k_scale must be one element on the CPU or cache device")
    if k_scale.dtype != torch.float32:
        raise ValueError(f"k_scale must be float32, got {k_scale.dtype}")

    _run_projection(
        k_buffer,
        kv_indices,
        k_scale,
        kv_proj_weight,
        kv_proj_scale,
        k_prefix,
        v_prefix,
        k_prefix.shape[1],
        m,
        cluster_n=_cluster_n(m, k_prefix.shape[1]),
    )
