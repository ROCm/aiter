# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 H64 mixed FP8/NVFP4 sparse MLA with fused split-K reduction."""

import math
import struct

import torch

from ..jit.core import compile_ops
from ..jit.utils.asm_guard import require_gfx1250_asm
from ..jit.utils.chip_info import get_cu_num, get_gfx_runtime
from ..utility import dtypes


@compile_ops("module_mla_sparse_prefill_v41_asm", ffi_type="ctypes")
def asm_mla_sparse_prefill_v41_fwd(
    q: torch.Tensor,
    q_scale: torch.Tensor,
    unified_kv: torch.Tensor,
    unified_kv_scale: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_scale: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
    num_splits: int,
    partial_r: torch.Tensor | None,
    partial_lse: torch.Tensor | None,
) -> None:
    """Raw ctypes entry; the argument order must match the C++ entry exactly."""


def _check_tensor(name, value, shape, allowed_dtypes, device):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"mla_sparse_prefill_v41 asm: {name} must be a tensor")
    if value.device != device:
        raise ValueError(
            f"mla_sparse_prefill_v41 asm: {name} must be on {device}, got {value.device}"
        )
    if value.dtype not in allowed_dtypes:
        raise ValueError(
            f"mla_sparse_prefill_v41 asm: {name} must have dtype in "
            f"{allowed_dtypes}, got {value.dtype}"
        )
    if len(value.shape) != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(value.shape, shape)
    ):
        raise ValueError(
            f"mla_sparse_prefill_v41 asm: {name} must have shape {shape}, "
            f"got {tuple(value.shape)}"
        )
    if not value.is_contiguous():
        raise ValueError(
            f"mla_sparse_prefill_v41 asm: {name} must be contiguous; "
            "data and scales use separate dense arrays. Packed NVFP4 rows "
            "with a 288-byte stride require explicit unpacking before this call."
        )


def _auto_v41_asm_splits(n: int, nnz: int) -> int:
    # Import at call time: the public module also imports this backend.
    from .mla_sparse_prefill import _asm_v41_kv_splits

    return _asm_v41_kv_splits(n, nnz, get_cu_num())


def _asm_sparse_prefill_v41(
    q,
    q_scale,
    unified_kv,
    unified_kv_scale,
    kv_indices_prefix,
    kv_indptr_prefix,
    kv,
    kv_scale,
    kv_indices_extend,
    kv_indptr_extend,
    attn_sink,
    softmax_scale,
    out,
    num_kv_splits=None,
):
    """Run H64 attention; ``None`` selects splits with a shape-based heuristic.

    The caller supplies valid CSR indices, including causal visibility.
    Only the FP32 value of ``1 / sqrt(512)`` is supported.
    """
    splits = num_kv_splits
    if splits is not None and (type(splits) is not int or not 1 <= splits <= 16):
        raise ValueError(
            "mla_sparse_prefill_v41 asm: num_kv_splits must be an integer "
            f"in [1, 16] or None, got {num_kv_splits!r}"
        )
    if not isinstance(q, torch.Tensor) or q.dim() != 3:
        raise ValueError("mla_sparse_prefill_v41 asm: q must have shape [N, 64, 512]")
    if q.device.type != "cuda" or torch.version.hip is None:
        raise ValueError("mla_sparse_prefill_v41 asm: q must be a HIP device tensor")
    n = q.size(0)
    if n > 65536:
        raise ValueError("mla_sparse_prefill_v41 asm: N must be at most 65536")
    device = q.device
    fp8 = (torch.float8_e4m3fn,)
    e8m0 = (torch.uint8, dtypes.fp8_e8m0)
    specs = (
        ("q", q, (n, 64, 512), fp8),
        ("q_scale", q_scale, (n, 64, 16), e8m0),
        ("unified_kv", unified_kv, (None, 256), (torch.uint8, dtypes.fp4x2)),
        ("unified_kv_scale", unified_kv_scale, (None, 32), (torch.uint8, *fp8)),
        ("kv", kv, (None, 512), fp8),
        ("kv_scale", kv_scale, (None, 16), e8m0),
        ("kv_indices_prefix", kv_indices_prefix, (None,), (torch.int32,)),
        ("kv_indptr_prefix", kv_indptr_prefix, (n + 1,), (torch.int32,)),
        ("kv_indices_extend", kv_indices_extend, (None,), (torch.int32,)),
        ("kv_indptr_extend", kv_indptr_extend, (n + 1,), (torch.int32,)),
    )
    for name, value, shape, allowed_dtypes in specs:
        _check_tensor(name, value, shape, allowed_dtypes, device)
    if unified_kv.size(0) != unified_kv_scale.size(0):
        raise ValueError("mla_sparse_prefill_v41 asm: CSA data/scale row counts differ")
    if kv.size(0) != kv_scale.size(0):
        raise ValueError("mla_sparse_prefill_v41 asm: SWA data/scale row counts differ")
    for name, indices in (
        ("kv_indices_prefix", kv_indices_prefix),
        ("kv_indices_extend", kv_indices_extend),
    ):
        if indices.numel() > 2**31 - 1:
            raise ValueError(
                f"mla_sparse_prefill_v41 asm: {name} exceeds int32 CSR capacity"
            )
    if attn_sink is not None:
        _check_tensor("attn_sink", attn_sink, (64,), (torch.float32,), device)
    if out is None:
        out = torch.empty((n, 64, 512), dtype=torch.bfloat16, device=device)
    else:
        _check_tensor("out", out, (n, 64, 512), (torch.bfloat16,), device)
        if out.numel():
            out_start = out.data_ptr()
            out_end = out_start + out.numel() * out.element_size()
            inputs = [(name, value) for name, value, _, _ in specs]
            if attn_sink is not None:
                inputs.append(("attn_sink", attn_sink))
            for name, value in inputs:
                if value.numel():
                    start = value.data_ptr()
                    end = start + value.numel() * value.element_size()
                    if out_start < end and start < out_end:
                        raise ValueError(
                            f"mla_sparse_prefill_v41 asm: out must not overlap {name}"
                        )
    scale = float(softmax_scale)
    try:
        supported_scale = struct.pack("<f", scale) == struct.pack(
            "<f", 1 / math.sqrt(512)
        )
    except (OverflowError, struct.error):
        supported_scale = False
    if not supported_scale:
        raise ValueError(
            "mla_sparse_prefill_v41 asm only supports softmax_scale=1/sqrt(512) "
            "with the same FP32 representation; the fixed kernel's sink path "
            "does not support another scale"
        )
    if n == 0:
        return out

    with torch.cuda.device(device):
        gfx = get_gfx_runtime()
        if gfx != "gfx1250":
            raise RuntimeError(
                f"mla_sparse_prefill_v41 asm requires gfx1250, got {gfx}"
            )
        require_gfx1250_asm("mla_sparse_prefill_v41")
        if splits is None:
            nnz = kv_indices_prefix.numel() + kv_indices_extend.numel()
            splits = _auto_v41_asm_splits(n, nnz)
        if attn_sink is None:
            attn_sink = torch.full((64,), -math.inf, dtype=torch.float32, device=device)
        partial_r = partial_lse = None
        if splits > 1:
            partial_r = torch.empty(
                (n, splits, 65, 512), dtype=torch.float32, device=device
            )
            partial_lse = torch.empty(
                (n, splits, 64), dtype=torch.float32, device=device
            )
        asm_mla_sparse_prefill_v41_fwd(
            q,
            q_scale,
            unified_kv,
            unified_kv_scale,
            kv_indices_prefix,
            kv_indptr_prefix,
            kv,
            kv_scale,
            kv_indices_extend,
            kv_indptr_extend,
            attn_sink,
            out,
            scale,
            splits,
            partial_r,
            partial_lse,
        )
    return out
