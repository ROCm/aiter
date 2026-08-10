# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import math
import weakref
from itertools import pairwise

import torch
import torch.nn.functional as F

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx

# Explicit W/U backend selection.  Keep the integer ABI stable so the adapter
# layer can select a path without relying on process-global environment state.
OPUS_GDN_K2_AUTO = 0
OPUS_GDN_K2_WU_FUSED = 1
OPUS_GDN_K2_SPLIT = 2
OPUS_GDN_SUPPORTED_K2_MODES = (
    OPUS_GDN_K2_AUTO,
    OPUS_GDN_K2_WU_FUSED,
    OPUS_GDN_K2_SPLIT,
)

_INT32_MAX = 2**31 - 1
_INFERENCE_TENSOR_VERSION = -1
_VarlenMetadata = tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, int]
_VarlenMetadataEntry = tuple[
    weakref.ReferenceType[torch.Tensor],
    int,
    _VarlenMetadata,
]
_VARLEN_METADATA_CACHE: dict[tuple[int, int], _VarlenMetadataEntry] = {}


def _tensor_version(tensor: torch.Tensor) -> int:
    """Return a cache version, treating inference metadata as immutable.

    PyTorch inference tensors deliberately omit the version counter. Packed
    sequence metadata created by serving schedulers is stable for its lifetime,
    so cache it by object identity. Callers that need in-place mutation must
    use a normal versioned tensor; those retain automatic invalidation.
    """
    try:
        return tensor._version
    except RuntimeError:
        return _INFERENCE_TENSOR_VERSION


def _prepare_opus_gdn_varlen_metadata(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> _VarlenMetadata:
    """Validate packed offsets and cache their native int32 chunk metadata.

    The weak-keyed multi-entry cache is indexed by Tensor identity, chunk size,
    and ``_version``. Schedulers may keep several static packed partitions hot;
    an in-place update still invalidates its derived chunk addressing. Entries
    disappear automatically when the corresponding offsets Tensor dies.
    """
    if not isinstance(cu_seqlens, torch.Tensor):
        raise TypeError("cu_seqlens must be a torch.Tensor")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must have shape [N + 1]")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("cu_seqlens must have dtype int32 or int64")
    if not cu_seqlens.is_contiguous():
        raise ValueError("cu_seqlens must be contiguous")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    cache_key = (id(cu_seqlens), chunk_size)
    cached_entry = _VARLEN_METADATA_CACHE.get(cache_key)
    if cached_entry is not None:
        cached_ref, cached_version, cached_result = cached_entry
        if cached_ref() is cu_seqlens:
            if cached_version == _INFERENCE_TENSOR_VERSION:
                return cached_result
            version = _tensor_version(cu_seqlens)
            if cached_version == version:
                return cached_result
        else:
            _VARLEN_METADATA_CACHE.pop(cache_key, None)
            version = _tensor_version(cu_seqlens)
    else:
        version = _tensor_version(cu_seqlens)

    cu_values = cu_seqlens.tolist()
    if cu_values[0] != 0:
        raise ValueError("cu_seqlens endpoints must start at 0")
    if any(b <= a for a, b in pairwise(cu_values)):
        raise ValueError("cu_seqlens must be strictly increasing")
    total_tokens = int(cu_values[-1])
    if total_tokens > _INT32_MAX:
        raise ValueError("total_tokens must fit in int32")

    chunk_indices_values: list[tuple[int, int]] = []
    chunk_offsets_values = [0]
    max_chunks = 0
    for sequence_id, (bos, eos) in enumerate(pairwise(cu_values)):
        num_chunks = (int(eos) - int(bos) + chunk_size - 1) // chunk_size
        max_chunks = max(max_chunks, num_chunks)
        chunk_indices_values.extend(
            (sequence_id, local_chunk) for local_chunk in range(num_chunks)
        )
        chunk_offsets_values.append(chunk_offsets_values[-1] + num_chunks)

    total_chunks = chunk_offsets_values[-1]
    if total_chunks > _INT32_MAX:
        raise ValueError("total_chunks must fit in int32")

    device = cu_seqlens.device
    cu_seqlens_i32 = torch.tensor(cu_values, dtype=torch.int32, device=device)
    chunk_indices = torch.tensor(
        chunk_indices_values, dtype=torch.int32, device=device
    ).reshape(total_chunks, 2)
    chunk_offsets = torch.tensor(chunk_offsets_values, dtype=torch.int32, device=device)
    result = (
        total_tokens,
        cu_seqlens_i32,
        chunk_indices,
        chunk_offsets,
        max_chunks,
    )

    version_after_build = _tensor_version(cu_seqlens)
    if version_after_build != version:
        raise ValueError("cu_seqlens was modified while preparing chunk metadata")

    def drop_cache_entry(
        dead_ref: weakref.ReferenceType[torch.Tensor],
        key: tuple[int, int] = cache_key,
    ) -> None:
        current = _VARLEN_METADATA_CACHE.get(key)
        if current is not None and current[0] is dead_ref:
            _VARLEN_METADATA_CACHE.pop(key, None)

    tensor_ref = weakref.ref(cu_seqlens, drop_cache_entry)
    _VARLEN_METADATA_CACHE[cache_key] = (
        tensor_ref,
        version_after_build,
        result,
    )
    return result


def _shares_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def _is_exact_contiguous_view(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.data_ptr() == b.data_ptr()
        and tuple(a.shape) == tuple(b.shape)
        and tuple(a.stride()) == tuple(b.stride())
    )


@compile_ops("module_opus_gdn_wu_prefill")
def _opus_gdn_wu_prefill_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    o: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    final_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    chunk_offsets: torch.Tensor,
    varlen_max_chunks: int,
    has_initial_state: bool,
    output_final_state: bool,
    BT: int,
    BV: int,
    num_warps: int,
    k1_algo: int,
    k2_mode: int,
    use_env_overrides: bool = True,
) -> None: ...


def opus_gdn_wu_prefill_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    BT: int = 64,
    BV: int = 64,
    num_warps: int | None = None,
    k1_algo: int = 1,
    k2_mode: int = 0,
    out: torch.Tensor | None = None,
    use_env_overrides: bool = True,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Opus HIP kernel for Gated DeltaNet prefill (forward only).

    Fuses all 4 chunkwise steps into specialized HIP kernels.
    K=V=128 specialized, gfx942/gfx950.

    Args:
        q: [B, T, H, K] bf16
        k: [B, T, H, K] bf16
        v: [B, T, H, V] bf16
        g: [B, T, H] — gate (log-space decay). BT64 Neumann kernels load
            bf16/fp32 directly; other dtypes are cast to fp32.
        beta: [B, T, H] — sigmoid gating. BT64 Neumann kernels load
            bf16/fp32 directly; other dtypes are cast to fp32.
        scale: 1/sqrt(K) if None
        initial_state: [B, H, V, K] fp32 for dense input, or
            [N, H, V, K] fp32 for packed varlen input.
        output_final_state: whether to return final hidden state
        BT: chunk size, 64 (tuned default), 32, 16, or 128 (gfx950 only)
        num_warps: K2 wave count. If omitted, uses 8 for the tuned BT64
            WS/WF paths and 4 for the legacy BT16/32/128 configurations.
        k2_mode: 0 selects between the W/U fused and split families using the
            measured gfx942 T/B*H/state envelope; 1 forces fused (WF); 2
            forces split scan/output (WS). Forced split requires BT=64.
        out: optional preallocated contiguous bf16 output tensor. Dense padded
            calls require an already BT-aligned sequence length; packed varlen
            accepts arbitrary positive sequence lengths.
        use_env_overrides: whether the native W/U launcher may read its
            OPUS_GDN_* benchmark overrides. Defaults to True for direct
            backend A/B compatibility; production adapters should pass False.
        cu_seqlens: optional cumulative sequence lengths [N+1]. When present,
            q/k/v use packed [1, total_tokens, H, 128] layout and the native
            BT64 Neumann + WS kernels reset recurrence at every sequence. HIP
            Graph capture requires a cache-prewarm call with the same Tensor;
            retain it and do not modify its offsets for any replay. Recapture
            the graph when the packed partition changes.

    Returns:
        (o, final_state): o is [B, T, H, V] bf16. final_state is
        [B, H, V, K] for dense input, [N, H, V, K] for packed varlen, or None.
    """
    if not isinstance(q, torch.Tensor) or q.ndim != 4:
        raise ValueError("q must have shape [B, T, H, 128]")
    B, T, H, K = q.shape
    if K != 128:
        raise ValueError("q feature size must be 128")
    expected_vector_shape = (B, T, H, 128)
    expected_scalar_shape = (B, T, H)
    for name, tensor in (("k", k), ("v", v)):
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape) != expected_vector_shape
        ):
            raise ValueError(f"{name} must have shape {expected_vector_shape}")
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on the same device as q")
    for name, tensor in (("g", g), ("beta", beta)):
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape) != expected_scalar_shape
        ):
            raise ValueError(f"{name} must have shape {expected_scalar_shape}")
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on the same device as q")
    V = v.shape[-1]
    is_varlen = cu_seqlens is not None

    if BT not in (16, 32, 64, 128):
        raise ValueError(f"Unsupported BT={BT}; expected 16, 32, 64, or 128")
    if BV not in (32, 64):
        raise ValueError(f"Unsupported BV={BV}; expected 32 or 64")

    if num_warps is None:
        num_warps = 8 if BT == 64 else 4
    if num_warps not in (2, 4, 8):
        raise ValueError(f"Unsupported num_warps={num_warps}; expected 2, 4, or 8")

    if k2_mode not in OPUS_GDN_SUPPORTED_K2_MODES:
        raise ValueError(
            f"Unsupported k2_mode={k2_mode}; expected one of "
            f"{OPUS_GDN_SUPPORTED_K2_MODES}"
        )
    if k2_mode == OPUS_GDN_K2_SPLIT and BT != 64:
        raise ValueError("k2_mode=2 (WS) requires BT=64")
    if is_varlen:
        if B != 1:
            raise ValueError("packed varlen expects q/k/v batch dimension B=1")
        if BT != 64 or k1_algo != 1:
            raise ValueError("packed varlen requires BT=64 and k1_algo=1")
        if k2_mode == OPUS_GDN_K2_WU_FUSED:
            raise ValueError("packed varlen currently supports the WS path only")
        k2_mode = OPUS_GDN_K2_SPLIT

    if scale is None:
        scale = K**-0.5
    if not math.isfinite(float(scale)):
        raise ValueError("scale must be finite")

    empty_meta = torch.empty(0, device=q.device, dtype=torch.int32)
    if is_varlen:
        if not isinstance(cu_seqlens, torch.Tensor):
            raise TypeError("cu_seqlens must be a torch.Tensor")
        if cu_seqlens.device != q.device:
            raise ValueError("cu_seqlens must be on the same device as q")
        total_tokens, cu_seqlens_i32, chunk_indices, chunk_offsets, max_chunks = (
            _prepare_opus_gdn_varlen_metadata(cu_seqlens, BT)
        )
        if total_tokens != T:
            raise ValueError(f"cu_seqlens endpoints must be 0 and total_tokens={T}")
        N = cu_seqlens.numel() - 1
    else:
        cu_seqlens_i32 = empty_meta
        chunk_indices = empty_meta
        chunk_offsets = empty_meta
        max_chunks = 0
        N = B

    q = q.contiguous().to(torch.bfloat16)
    k = k.contiguous().to(torch.bfloat16)
    v = v.contiguous().to(torch.bfloat16)
    native_scalar_loads = BT == 64 and k1_algo == 1
    g = g.contiguous()
    beta = beta.contiguous()
    if not native_scalar_loads or g.dtype not in (torch.bfloat16, torch.float32):
        g = g.float()
    if not native_scalar_loads or beta.dtype not in (
        torch.bfloat16,
        torch.float32,
    ):
        beta = beta.float()

    if BT == 128 and get_gfx() != "gfx950":
        raise ValueError(
            "BT=128 requires gfx950 (MI350) — LDS exceeds gfx942 64KB limit"
        )

    pad_len = 0 if is_varlen else (BT - T % BT) % BT
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, 0, 0, pad_len))
        g = F.pad(g, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, 0, 0, pad_len))

    if out is not None:
        if pad_len:
            raise ValueError("a preallocated output requires T to be BT-aligned")
        if (
            out.shape != v.shape
            or out.dtype != torch.bfloat16
            or out.device != v.device
            or not out.is_contiguous()
        ):
            raise ValueError(
                "out must be a contiguous bf16 tensor matching v shape/device"
            )
        for name, read_only in (
            ("q", q),
            ("k", k),
            ("g", g),
            ("beta", beta),
            ("initial_state", initial_state),
        ):
            if read_only is not None and _shares_storage(out, read_only):
                raise ValueError(f"out must not alias {name} storage")
        if _shares_storage(out, v) and not _is_exact_contiguous_view(out, v):
            raise ValueError("out may alias v only when it is exactly the same view")
        o = out
    else:
        o = torch.empty_like(v)

    has_init = initial_state is not None
    if is_varlen and has_init:
        expected_state_shape = (N, H, V, K)
        if (
            tuple(initial_state.shape) != expected_state_shape
            or initial_state.dtype != torch.float32
            or initial_state.device != q.device
            or not initial_state.is_contiguous()
        ):
            raise ValueError(
                "initial_state must be contiguous fp32 with shape "
                f"{expected_state_shape} on q.device"
            )
        init_st = initial_state
    else:
        init_st = (
            initial_state.contiguous().float()
            if has_init
            else torch.empty(0, device=q.device, dtype=torch.float32)
        )
    final_st = (
        torch.empty(N, H, V, K, dtype=torch.float32, device=q.device)
        if output_final_state
        else torch.empty(0, device=q.device, dtype=torch.float32)
    )

    _opus_gdn_wu_prefill_fwd(
        q,
        k,
        v,
        g,
        beta,
        o,
        scale,
        init_st,
        final_st,
        cu_seqlens_i32,
        chunk_indices,
        chunk_offsets,
        max_chunks,
        has_init,
        output_final_state,
        BT,
        BV,
        num_warps,
        k1_algo,
        k2_mode,
        use_env_overrides,
    )

    if pad_len > 0:
        o = o[:, :T]

    return o, final_st if output_final_state else None
