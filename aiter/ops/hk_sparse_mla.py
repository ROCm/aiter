# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""HipKittens sparse MLA (DeepSeek Sparse Attention) forward for gfx950 (csrc/kernels/mla/hk_sparse_mla).

Every query token attends to its own list of selected KV slots; all heads share one latent row (MQA), which is both
K and V. Built for 32 heads per GPU x 512 latent dims (no RoPE part), fp8 q / kv without scales (P quantized as
fp8(P * 448), the TileLang kernel's numerics), up to 2304 indices per token.
"""

import math

import torch
import torch.nn.functional as F

from ..jit.core import compile_ops

HEADS, DIM, MAX_TOPK, KEY_BLOCK = 32, 512, 2304, 128


@compile_ops(
    "module_hk_sparse_mla", fc_name="aiter_hk_sparse_mla_fwd", ffi_type="ctypes"
)
def _hk_sparse_mla_fwd_raw(
    q: int,
    kv: int,
    indices: int,
    out: int,
    num_tokens: int,
    topk: int,
    kv_stride_bytes: int,
    num_slots: int,
    sm_scale: float,
) -> None: ...


def hk_sparse_mla_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sparse MLA forward: out[t, h] = softmax_k(q[t, h] . kv[idx[t, k]] * sm_scale) @ kv[idx[t, k]][:512].

    q [T, 32, 512] (cast to kv's fp8 dtype); kv [N, ..., >= 512] fp8 e4m3 whose rows are addressed by
    kv.stride(0) (the first 512 entries are the latent; rows 16-byte aligned); indices [T, topk] int32 slots,
    -1 or >= N masked, padded here to a multiple of 128. Returns bf16 [T, 32, 512].
    Uses one workgroup per query without a workspace. Optimized for T >= 512;
    callers should dispatch smaller batches to another backend. Warm the JIT before graph capture.
    """
    if not q.is_cuda:
        raise ValueError("hk_sparse_mla_fwd: q must be on a HIP device")
    for name, tensor in (("kv", kv), ("indices", indices), ("out", out)):
        if tensor is not None and tensor.device != q.device:
            raise ValueError(f"hk_sparse_mla_fwd: {name} must be on {q.device}")
    arch = getattr(
        torch.cuda.get_device_properties(q.device), "gcnArchName", "unknown"
    ).split(":")[0]
    if arch != "gfx950":
        raise RuntimeError(f"hk_sparse_mla_fwd requires gfx950, running on {arch}")
    if (
        kv.dtype != torch.float8_e4m3fn
        or kv.dim() < 2
        or kv.shape[-1] < DIM
        or kv.stride(-1) != 1
        or math.prod(kv.shape[1:-1]) != 1
        or not 1 <= kv.shape[0] < 2**31
        or kv.stride(0) < DIM
        or kv.stride(0) % 16
        or kv.data_ptr() % 16
    ):
        raise ValueError(
            f"hk_sparse_mla_fwd: kv must be OCP fp8 e4m3 [N, ..., >= {DIM}] with singleton middle dimensions, "
            "1 <= N < 2^31, contiguous latent rows and 16-byte-aligned base/row stride"
        )
    if not math.isfinite(sm_scale) or sm_scale <= 0:
        raise ValueError("hk_sparse_mla_fwd: sm_scale must be finite and positive")
    if not q.is_floating_point():
        raise ValueError("hk_sparse_mla_fwd: q must have a floating-point dtype")
    q = q.to(kv.dtype).contiguous()
    if q.data_ptr() % 16:
        q = q.clone()
    if q.dim() != 3 or q.shape[1] != HEADS or q.shape[2] != DIM:
        raise ValueError(
            f"hk_sparse_mla_fwd: q must be [T, {HEADS}, {DIM}], got {tuple(q.shape)}"
        )
    T = q.shape[0]
    if (
        indices.dim() != 2
        or indices.shape[0] != T
        or indices.dtype not in (torch.int32, torch.int64)
        or not 1 <= indices.shape[1] <= MAX_TOPK
    ):
        raise ValueError(
            f"hk_sparse_mla_fwd: indices must be int32/int64 [T, topk], 1 <= topk <= {MAX_TOPK}, got {tuple(indices.shape)}"
        )
    if out is None:
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    elif (
        out.shape != q.shape
        or out.dtype != torch.bfloat16
        or not out.is_contiguous()
        or out.data_ptr() % 16
    ):
        raise ValueError(
            f"hk_sparse_mla_fwd: out must be a contiguous bf16 [T, {HEADS}, {DIM}] tensor"
        )
    if out.numel() and any(
        torch._C._overlaps(out, tensor) for tensor in (q, kv, indices)
    ):
        raise ValueError("hk_sparse_mla_fwd: out must not overlap any input storage")
    if T == 0:
        return out
    # Preserve masking for int64 IDs outside the int32 address range.
    idx = (
        indices.clamp(-1, kv.shape[0]).to(torch.int32)
        if indices.dtype == torch.int64
        else indices
    )
    pad = -idx.shape[1] % KEY_BLOCK
    if pad:
        idx = F.pad(idx, (0, pad), value=-1)
    idx = idx.contiguous()
    topk = idx.shape[1]
    if topk > MAX_TOPK:
        raise ValueError(
            f"hk_sparse_mla_fwd: at most {MAX_TOPK} indices per token, got {topk}"
        )
    with torch.cuda.device(q.device):
        _hk_sparse_mla_fwd_raw(
            q.data_ptr(),
            kv.data_ptr(),
            idx.data_ptr(),
            out.data_ptr(),
            T,
            topk,
            kv.stride(0) * kv.element_size(),
            kv.shape[0],
            float(sm_scale),
        )
    return out
