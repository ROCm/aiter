# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 H64 BF16-Q RoPE/quant + mixed MLA + FP8-O inverse-RoPE/quant.

This replaces the old FP8-Q/BF16-O ASM contract. OPUS keeps its separate
attention-only interface in mla_sparse_prefill.py. The shipped CO consumes a
144-byte kernarg with a 120-byte preload window; argument order below follows it.
"""

import math
import struct

import torch

from ..jit.core import compile_ops
from ..jit.utils.asm_guard import require_gfx1250_asm
from ..jit.utils.build_targets import is_gfx1250_family
from ..jit.utils.chip_info import get_cu_num, get_gfx_runtime
from ..jit.utils.torch_guard import torch_compile_guard
from ..utility import dtypes


@compile_ops("module_mla_sparse_prefill_v41_asm", ffi_type="ctypes")
def asm_mla_sparse_prefill_v41_fwd(
    q_nope: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    unified_kv: torch.Tensor,
    unified_kv_scale: torch.Tensor,
    kv: torch.Tensor,
    kv_scale: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    num_splits: int,
    out: torch.Tensor,
    out_scale: torch.Tensor,
    partial_r: torch.Tensor | None,
    partial_lse: torch.Tensor | None,
) -> None:
    """Raw 19-argument ctypes entry, in the 144B GPU kernarg field order.

    Positions are fixed int64. out_scale is uint8 containing E8M0 bytes.
    The C++ FFI appends the current HIP stream, which is not a kernarg field.
    """


def _check_tensor(name, value, shape, allowed_dtypes, device):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"mla_sparse_prefill_v41 asm: {name} must be a tensor")
    if value.device != device:
        raise ValueError(f"mla_sparse_prefill_v41 asm: {name} must be on {device}")
    if value.dtype not in allowed_dtypes:
        raise ValueError(
            f"mla_sparse_prefill_v41 asm: {name} has unsupported dtype {value.dtype}"
        )
    if len(value.shape) != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(value.shape, shape)
    ):
        raise ValueError(f"mla_sparse_prefill_v41 asm: {name} must have shape {shape}")
    if not value.is_contiguous():
        raise ValueError(
            f"mla_sparse_prefill_v41 asm: {name} must be contiguous; "
            "prepare separate dense CSA data/scale planes outside this call"
        )


def _check_disjoint(reads, writes):
    for i, (name, value) in enumerate(writes):
        if value is None or not value.numel():
            continue
        lo = value.data_ptr()
        hi = lo + value.numel() * value.element_size()
        for other_name, other in [*reads, *writes[:i]]:
            if other is None or not other.numel():
                continue
            start = other.data_ptr()
            end = start + other.numel() * other.element_size()
            if lo < end and start < hi:
                raise ValueError(
                    f"mla_sparse_prefill_v41 asm: {name} must not overlap {other_name}"
                )


def _auto_v41_asm_splits(
    n: int,
    nnz: int,
    *,
    prefix_nnz: int | None = None,
    prefix_rows: int | None = None,
) -> int:
    if n == 0 or n > 64:
        return 1
    # Warm Q/O-kernel measurements cover these two physical CSA pools and
    # N=8/16/32/48/64. Do not extrapolate the winners to unmeasured batches
    # or pool sizes. Prefix cardinality is a metadata-only workload estimate;
    # SWA entries must not change the inferred CSA topk.
    if prefix_rows in (65536, 1048576):
        if prefix_nnz == n * 512:
            if n in (8, 16) or (n == 32 and prefix_rows == 1048576):
                return 4
        elif prefix_nnz == n * 1024:
            if n in (8, 16):
                return 8
            if n in (32, 48, 64):
                return 4
    # Retain the previous policy outside the measured configurations.
    if prefix_nnz == n * 512:
        return 1
    if prefix_nnz == n * 1024:
        return 4 if n <= 32 else 2
    from .mla_sparse_prefill import _asm_v41_kv_splits

    return _asm_v41_kv_splits(n, nnz, get_cu_num())


def _run_qo(
    q_nope,
    cos,
    sin,
    positions,
    unified_kv,
    unified_kv_scale,
    kv,
    kv_scale,
    kv_indptr_prefix,
    kv_indices_prefix,
    kv_indptr_extend,
    kv_indices_extend,
    attn_sink,
    softmax_scale,
    num_kv_splits,
    out=None,
    out_scale=None,
    partial_r=None,
    partial_lse=None,
):
    splits = num_kv_splits
    if splits is not None and (type(splits) is not int or not 1 <= splits <= 16):
        raise ValueError(
            "mla_sparse_prefill_v41 asm: num_kv_splits must be an integer in [1,16] or None"
        )
    if not isinstance(q_nope, torch.Tensor) or q_nope.dim() != 3:
        raise ValueError(
            "mla_sparse_prefill_v41 asm: q_nope must have shape [N,64,512]"
        )
    if q_nope.device.type != "cuda" or torch.version.hip is None:
        raise ValueError("mla_sparse_prefill_v41 asm: q_nope must be a HIP tensor")
    n, device = q_nope.size(0), q_nope.device
    if n > 65535:
        raise ValueError("mla_sparse_prefill_v41 asm: N must be at most 65535")
    fp8, e8m0 = (torch.float8_e4m3fn,), (torch.uint8, dtypes.fp8_e8m0)
    specs = (
        ("q_nope", q_nope, (n, 64, 512), (torch.bfloat16,)),
        ("cos", cos, (None, 32), (torch.bfloat16,)),
        ("sin", sin, (None, 32), (torch.bfloat16,)),
        ("positions", positions, (n,), (torch.int64,)),
        ("unified_kv", unified_kv, (None, 256), (torch.uint8, dtypes.fp4x2)),
        ("unified_kv_scale", unified_kv_scale, (None, 32), (torch.uint8, *fp8)),
        ("kv", kv, (None, 512), fp8),
        ("kv_scale", kv_scale, (None, 16), e8m0),
        ("kv_indptr_prefix", kv_indptr_prefix, (n + 1,), (torch.int32,)),
        ("kv_indices_prefix", kv_indices_prefix, (None,), (torch.int32,)),
        ("kv_indptr_extend", kv_indptr_extend, (n + 1,), (torch.int32,)),
        ("kv_indices_extend", kv_indices_extend, (None,), (torch.int32,)),
    )
    for name, value, shape, allowed in specs:
        _check_tensor(name, value, shape, allowed, device)
    if cos.shape != sin.shape or (n and cos.size(0) == 0):
        raise ValueError(
            "mla_sparse_prefill_v41 asm: cos/sin row counts must match and be nonempty for N>0"
        )
    if unified_kv.size(0) != unified_kv_scale.size(0) or kv.size(0) != kv_scale.size(0):
        raise ValueError("mla_sparse_prefill_v41 asm: KV data/scale row counts differ")
    if any(x.numel() > 2**31 - 1 for x in (kv_indices_prefix, kv_indices_extend)):
        raise ValueError(
            "mla_sparse_prefill_v41 asm: indices exceed int32 CSR capacity"
        )
    if attn_sink is not None:
        _check_tensor("attn_sink", attn_sink, (64,), (torch.float32,), device)
    try:
        scale = float(softmax_scale)
        supported = struct.pack("<f", scale) == struct.pack("<f", 1 / math.sqrt(512))
    except (TypeError, ValueError, OverflowError, struct.error):
        supported = False
    if not supported:
        raise ValueError(
            "mla_sparse_prefill_v41 asm only supports softmax_scale=1/sqrt(512) with the same FP32 representation"
        )
    if (partial_r is None) != (partial_lse is None):
        raise ValueError(
            "mla_sparse_prefill_v41 asm: partial_r and partial_lse must be supplied together"
        )
    if out is None:
        out = torch.empty((n, 64, 512), dtype=torch.float8_e4m3fn, device=device)
    if out_scale is None:
        out_scale = torch.empty((n, 64, 16), dtype=torch.uint8, device=device)
    _check_tensor("out", out, (n, 64, 512), fp8, device)
    _check_tensor("out_scale", out_scale, (n, 64, 16), (torch.uint8,), device)
    reads = [(name, value) for name, value, _, _ in specs] + [("attn_sink", attn_sink)]
    _check_disjoint(reads, [("out", out), ("out_scale", out_scale)])
    if n == 0:
        return out, out_scale

    with torch.cuda.device(device):
        gfx = get_gfx_runtime()
        if not is_gfx1250_family(gfx):
            raise RuntimeError(
                f"mla_sparse_prefill_v41 asm requires gfx1250, got {gfx}"
            )
        require_gfx1250_asm("mla_sparse_prefill_v41")
        if splits is None:
            prefix_nnz = kv_indices_prefix.numel()
            splits = _auto_v41_asm_splits(
                n,
                prefix_nnz + kv_indices_extend.numel(),
                prefix_nnz=prefix_nnz,
                prefix_rows=unified_kv.size(0),
            )
        if attn_sink is None:
            attn_sink = torch.full((64,), -math.inf, dtype=torch.float32, device=device)
        if splits > 1 or partial_r is not None:
            if partial_r is None:
                partial_r = torch.empty(
                    (n, splits, 65, 512), dtype=torch.float32, device=device
                )
                partial_lse = torch.empty(
                    (n, splits, 64), dtype=torch.float32, device=device
                )
            _check_tensor(
                "partial_r", partial_r, (n, splits, 65, 512), (torch.float32,), device
            )
            _check_tensor(
                "partial_lse", partial_lse, (n, splits, 64), (torch.float32,), device
            )
            _check_disjoint(
                reads,
                [
                    ("out", out),
                    ("out_scale", out_scale),
                    ("partial_r", partial_r),
                    ("partial_lse", partial_lse),
                ],
            )
        native = getattr(
            asm_mla_sparse_prefill_v41_fwd,
            "_eager_body",
            asm_mla_sparse_prefill_v41_fwd,
        )
        native(
            q_nope,
            cos,
            sin,
            positions,
            unified_kv,
            unified_kv_scale,
            kv,
            kv_scale,
            kv_indptr_prefix,
            kv_indices_prefix,
            kv_indptr_extend,
            kv_indices_extend,
            attn_sink,
            scale,
            splits,
            out,
            out_scale,
            partial_r if splits > 1 else None,
            partial_lse if splits > 1 else None,
        )
    return out, out_scale


def _alloc_fake(q_nope, *args, **kwargs):
    n = q_nope.shape[0]
    return (
        torch.empty((n, 64, 512), dtype=torch.float8_e4m3fn, device=q_nope.device),
        torch.empty((n, 64, 16), dtype=torch.uint8, device=q_nope.device),
    )


def _out_fake(*args, **kwargs):
    return None


@torch_compile_guard(mutates_args=[], gen_fake=_alloc_fake)
def _asm_sparse_prefill_v41_alloc(
    q_nope: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    unified_kv: torch.Tensor,
    unified_kv_scale: torch.Tensor,
    kv: torch.Tensor,
    kv_scale: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    num_kv_splits: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _run_qo(
        q_nope,
        cos,
        sin,
        positions,
        unified_kv,
        unified_kv_scale,
        kv,
        kv_scale,
        kv_indptr_prefix,
        kv_indices_prefix,
        kv_indptr_extend,
        kv_indices_extend,
        attn_sink,
        softmax_scale,
        num_kv_splits,
    )


@torch_compile_guard(
    mutates_args=["out", "out_scale", "partial_r", "partial_lse"], gen_fake=_out_fake
)
def _asm_sparse_prefill_v41_out(
    q_nope: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    unified_kv: torch.Tensor,
    unified_kv_scale: torch.Tensor,
    kv: torch.Tensor,
    kv_scale: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    num_kv_splits: int | None,
    out: torch.Tensor,
    out_scale: torch.Tensor,
    partial_r: torch.Tensor | None,
    partial_lse: torch.Tensor | None,
) -> None:
    _run_qo(
        q_nope,
        cos,
        sin,
        positions,
        unified_kv,
        unified_kv_scale,
        kv,
        kv_scale,
        kv_indptr_prefix,
        kv_indices_prefix,
        kv_indptr_extend,
        kv_indices_extend,
        attn_sink,
        softmax_scale,
        num_kv_splits,
        out,
        out_scale,
        partial_r,
        partial_lse,
    )


def _asm_sparse_prefill_v41(
    q_nope: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    unified_kv: torch.Tensor,
    unified_kv_scale: torch.Tensor,
    kv: torch.Tensor,
    kv_scale: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    num_kv_splits: int | None = None,
    out: torch.Tensor | None = None,
    out_scale: torch.Tensor | None = None,
    partial_r: torch.Tensor | None = None,
    partial_lse: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unrotated BF16 Q -> forward RoPE/quant -> A8W4 MLA -> inverse RoPE/quant.

    H64/D512, BF16 cos/sin [P,32], int64 positions[N], independent dense KV
    planes and int32 CSRs are required. Callers ensure valid CSR/position values
    and finite selected inputs; CSR determines all visibility. Scale is fixed
    to the FP32 representation of 1/sqrt(512). No RMSNorm or BF16 roundtrip.

    Returns FP8 E4M3 [N,64,512] and uint8 [N,64,16] E8M0 encodings. For a
    grouped GEMM, bit-view the latter as dtypes.fp8_e8m0, then reshape both
    by the model's group count. Outputs/workspace must not overlap any input
    or each other. S>1 uses FP32 R[N,S,65,512] and LSE[N,S,64], in one cluster
    launch; these are scratch buffers, not a final LSE return. Preallocated
    pairs are optional. Do not share scratch between concurrent launches.

    The allocating custom op returns new tensors. The buffer-writing custom
    op returns None; aliases are returned here, outside its schema.
    """
    # Validate before the custom-op schema can coerce bool/float to SymInt.
    if num_kv_splits is not None and (
        type(num_kv_splits) is not int or not 1 <= num_kv_splits <= 16
    ):
        raise ValueError(
            "mla_sparse_prefill_v41 asm: num_kv_splits must be an integer in [1,16] or None"
        )
    if isinstance(softmax_scale, torch.Tensor):
        raise TypeError(
            "mla_sparse_prefill_v41 asm: softmax_scale must be a host scalar"
        )
    if out is None and out_scale is None and partial_r is None and partial_lse is None:
        return _asm_sparse_prefill_v41_alloc(
            q_nope,
            cos,
            sin,
            positions,
            unified_kv,
            unified_kv_scale,
            kv,
            kv_scale,
            kv_indptr_prefix,
            kv_indices_prefix,
            kv_indptr_extend,
            kv_indices_extend,
            attn_sink,
            softmax_scale,
            num_kv_splits,
        )
    if not isinstance(q_nope, torch.Tensor) or q_nope.dim() != 3:
        raise ValueError(
            "mla_sparse_prefill_v41 asm: q_nope must have shape [N,64,512]"
        )
    n = q_nope.shape[0]
    if out is None:
        out = torch.empty((n, 64, 512), dtype=torch.float8_e4m3fn, device=q_nope.device)
    if out_scale is None:
        out_scale = torch.empty((n, 64, 16), dtype=torch.uint8, device=q_nope.device)
    _asm_sparse_prefill_v41_out(
        q_nope,
        cos,
        sin,
        positions,
        unified_kv,
        unified_kv_scale,
        kv,
        kv_scale,
        kv_indptr_prefix,
        kv_indices_prefix,
        kv_indptr_extend,
        kv_indices_extend,
        attn_sink,
        softmax_scale,
        num_kv_splits,
        out,
        out_scale,
        partial_r,
        partial_lse,
    )
    return out, out_scale
