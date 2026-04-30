# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional

import torch
import triton
from torch import Tensor

from ..jit.core import compile_ops

MD_NAME = "module_chunk_gdr_fwd_h"

_BV_FIXED_LDS_BYTES = 32 * 1024
_BV_LDS_BYTES_PER_BV = 512
_BV_RESIDENT_WGS_CAP = 2
_BV_CANDIDATES = (64, 32, 16)
_BV_CACHE: dict[tuple[int, int, int, int], int] = {}


def _device_idx(device: torch.device) -> int:
    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def _get_shared_memory_per_cu(props: object) -> int:
    """Query per-CU shared memory with architecture-based fallback."""
    shared_per_cu = getattr(props, "shared_memory_per_multiprocessor", None)
    if shared_per_cu is not None:
        return int(shared_per_cu)
    arch = getattr(props, "gcnArchName", "")
    if arch:
        arch = arch.split(":")[0]
    _ARCH_LDS = {"gfx95": 128 * 1024, "gfx94": 64 * 1024}
    for prefix, size in _ARCH_LDS.items():
        if arch.startswith(prefix):
            return size
    shared_per_block = getattr(props, "shared_memory_per_block", None)
    if shared_per_block is not None:
        return int(shared_per_block)
    raise RuntimeError("Unable to determine shared memory per CU.")


def _compute_bv(
    device: torch.device,
    total_chunks: int,
    max_seq_chunks: int,
    num_heads: int,
) -> int:
    props = torch.cuda.get_device_properties(device)
    num_cus = props.multi_processor_count
    lds_per_cu = _get_shared_memory_per_cu(props)

    for bv in _BV_CANDIDATES:
        lds_per_wg = _BV_FIXED_LDS_BYTES + _BV_LDS_BYTES_PER_BV * bv
        resident = min(max(1, lds_per_cu // lds_per_wg), _BV_RESIDENT_WGS_CAP)
        total_wgs = (128 // bv) * num_heads * total_chunks
        threshold = max(1, (num_cus * resident) // 2) * max_seq_chunks
        if total_wgs >= threshold:
            return bv
    return 16


def _select_bv(
    device: torch.device, num_heads: int, total_chunks: int, max_seq_chunks: int
) -> int:
    key = (_device_idx(device), num_heads, total_chunks, max_seq_chunks)
    cached = _BV_CACHE.get(key)
    if cached is not None:
        return cached
    bv = _compute_bv(device, total_chunks, max_seq_chunks, num_heads)
    _BV_CACHE[key] = bv
    return bv


def _select_bv_for_dense(
    batch_size: int, seq_len: int, chunk_size: int, num_heads: int, device: torch.device
) -> int:
    nt = (seq_len + chunk_size - 1) // chunk_size
    return _select_bv(device, num_heads, batch_size * nt, nt)


def _select_bv_for_varlen(chunk_offsets: torch.Tensor, num_heads: int) -> int:
    offsets = chunk_offsets.tolist()
    total_chunks = offsets[-1]
    max_seq_chunks = max(offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1))
    return _select_bv(chunk_offsets.device, num_heads, total_chunks, max_seq_chunks)


@compile_ops(MD_NAME)
def chunk_gated_delta_rule_fwd_h_hip(
    k: Tensor,
    w: Tensor,
    u: Tensor,
    g: Tensor,
    gk: Tensor,
    initial_state: Tensor,
    cu_seqlens: Tensor,
    chunk_offsets: Tensor,
    selected_bv: int,
    has_initial_state: bool,
    output_final_state: bool,
    save_new_value: bool,
    use_exp2: bool,
) -> list[Tensor]: ...


def chunk_gated_delta_rule_fwd_h_hip_fn(
    k: Tensor,
    w: Tensor,
    u: Tensor,
    g: Optional[Tensor] = None,
    gk: Optional[Tensor] = None,
    initial_state: Optional[Tensor] = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: Optional[Tensor] = None,
    selected_bv: Optional[int] = None,
    use_exp2: bool = False,
) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    """
    HIP hidden state forward with h layout [V, K].

    Drop-in replacement for ``chunk_gated_delta_rule_fwd_h_opt_vk``
    when K=128, V=128, bf16.

    w, u: [B, H, T, K/V] head-major contiguous.
    initial_state/final_state: [N, H, V, K].
    h snapshots: [B, NT, H, V, K].
    v_new output: [B, H, T_flat, V].
    """
    assert chunk_size == 64
    assert k.shape[-1] == 128 and u.shape[-1] == 128

    B, T, Hg, K = k.shape
    H = w.shape[1]
    V = u.shape[-1]
    T_flat = w.shape[2]
    is_varlen = cu_seqlens is not None
    NT = triton.cdiv(T, chunk_size)

    _has_initial_state = initial_state is not None

    k_hip = k.contiguous()
    w_hip = w.contiguous()
    u_hip = u.contiguous()

    total_tokens = T_flat if is_varlen else B * T
    if g is not None:
        g_hip = g.reshape(total_tokens, H)
        if g_hip.dtype != torch.float32:
            g_hip = g_hip.to(torch.float32)
        g_hip = g_hip.contiguous()
    else:
        g_hip = torch.zeros(total_tokens, H, dtype=torch.float32, device=k.device)

    if is_varlen:
        from aiter.ops.triton._triton_kernels.gated_delta_rule.utils import (
            prepare_chunk_offsets,
        )

        assert B == 1, "Varlen mode expects B=1 (flattened input)."
        cu_seqlens_int32 = cu_seqlens.to(torch.int32)
        chunk_offsets_int32 = prepare_chunk_offsets(
            cu_seqlens_int32.to(torch.int64), chunk_size
        ).to(torch.int32)
    else:
        cu_seqlens_int32 = torch.empty(0, device=k.device, dtype=torch.int32)
        chunk_offsets_int32 = torch.arange(
            0, (B + 1) * NT, NT, dtype=torch.int32, device=k.device
        )

    if selected_bv is None:
        if is_varlen:
            selected_bv = _select_bv_for_varlen(chunk_offsets_int32, H)
        else:
            selected_bv = _select_bv_for_dense(B, T_flat, chunk_size, H, k.device)

    _initial_state = (
        initial_state.to(torch.float32)
        if initial_state is not None
        else torch.empty(0, device=k.device, dtype=torch.float32)
    )

    if gk is not None:
        gk_arg = gk.to(torch.float32).contiguous()
    else:
        gk_arg = torch.empty(0, device=k.device, dtype=torch.float32)

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h_hip(
        k_hip,
        w_hip,
        u_hip,
        g_hip,
        gk_arg,
        _initial_state,
        cu_seqlens_int32,
        chunk_offsets_int32,
        selected_bv,
        _has_initial_state,
        output_final_state,
        save_new_value,
        use_exp2,
    )

    if not is_varlen:
        h = h.view(B, NT, H, V, K)

    if not save_new_value:
        v_new = None

    if not output_final_state:
        final_state = None

    return h, v_new, final_state
