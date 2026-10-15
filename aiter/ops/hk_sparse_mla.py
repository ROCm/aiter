# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""HipKittens sparse MLA (DeepSeek Sparse Attention) forward for gfx950 (csrc/kernels/mla/hk_sparse_mla).

Every query token attends to its own list of selected KV slots; all heads share one latent row (MQA), which is both
K and V. Built for 32 heads per GPU x 512 latent dims (no RoPE part), fp8 q / kv without scales (P quantized as
fp8(P * 448), the TileLang kernel's numerics), up to 2304 indices per token.
"""

import torch
import torch.nn.functional as F

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx

HEADS, DIM, MAX_TOPK, KEY_BLOCK = 32, 512, 2304, 128
_FP8 = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)


@compile_ops(
    "module_hk_sparse_mla",
    fc_name="aiter_hk_sparse_mla_workspace_units",
    ffi_type="ctypes",
)
def _hk_sparse_mla_workspace_units(num_tokens: int, topk: int) -> int: ...


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
    workspace: int,
    workspace_bytes: int,
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
    -1 or >= N masked, padded here to a multiple of 128. Returns bf16 [T, 32, 512]. Small T splits each token's
    keys and merges the partials; their workspace comes from torch's caching allocator, so the call is
    graph-capturable.
    """
    if get_gfx() != "gfx950":
        raise RuntimeError(f"hk_sparse_mla_fwd requires gfx950, running on {get_gfx()}")
    if kv.dtype not in _FP8 or kv.dim() < 2 or kv.shape[-1] < DIM or kv.stride(-1) != 1:
        raise ValueError(
            f"hk_sparse_mla_fwd: kv must be fp8 e4m3 [N, ..., >= {DIM}] with contiguous rows"
        )
    q = q.to(kv.dtype).contiguous()
    if q.dim() != 3 or q.shape[1] != HEADS or q.shape[2] != DIM:
        raise ValueError(
            f"hk_sparse_mla_fwd: q must be [T, {HEADS}, {DIM}], got {tuple(q.shape)}"
        )
    T = q.shape[0]
    if indices.dim() != 2 or indices.shape[0] != T:
        raise ValueError(
            f"hk_sparse_mla_fwd: indices must be [T, topk], got {tuple(indices.shape)}"
        )
    if out is None:
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    elif out.shape != q.shape or out.dtype != torch.bfloat16 or not out.is_contiguous():
        raise ValueError(
            f"hk_sparse_mla_fwd: out must be a contiguous bf16 [T, {HEADS}, {DIM}] tensor"
        )
    if T == 0:
        return out
    idx = indices.to(torch.int32)
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
        ws = torch.empty(
            max(_hk_sparse_mla_workspace_units(T, topk) * 256, 1),
            dtype=torch.uint8,
            device=q.device,
        )
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
            ws.data_ptr(),
            ws.numel(),
        )
    return out
