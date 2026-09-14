# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL backend for the DeepSeek MLA fused gather + kv_b_proj expansion (gfx950).

Supported: fp8 KV cache (OCP e4m3), fp8 weight in either row-major or
``shuffle_weight((16,16))`` layout, per-output-row *or* 128x128 block weight
scale, per-tensor activation scale, page_size 1, bf16 outputs, gfx950.

The cache has no size limit: up to 4 GiB it is reached through one buffer
descriptor, beyond that through 64-bit per-lane addresses. Output width is
bounded, though -- see ``m_rows`` in :func:`_validate`.
"""

import functools

import flydsl.expr as fx
import torch
from flydsl.runtime.device import get_rocm_arch
from torch import Tensor

from aiter.jit.utils.chip_info import get_lds_capacity_bytes

from .kernels.gather_gemm_8wave import compile_gather_kv_b_proj_8w
from .kernels.tensor_shim import _run_compiled, ptr_arg

# The MLA latent layout, fixed by the model.
KV_C_DIM = 512
KV_PE_DIM = 64
KV_ROW_ELEMS = KV_C_DIM + KV_PE_DIM

# LDS = 4 A buffers of (BM/2)x128 plus 4 B buffers of (BN/2)x128, 1 byte/elem.
_LDS_BYTES_PER_BLOCK_UNIT = 256
_I32_MAX = 2**31
# num_records is a 32-bit BYTE count, so one descriptor spans at most this.
_BUFFER_SPAN_MAX = 2**32


def lds_bytes(block_m: int, block_n: int) -> int:
    return _LDS_BYTES_PER_BLOCK_UNIT * (int(block_m) + int(block_n))


def _config_reason(
    *,
    n_heads: int,
    nope: int,
    v_dim: int,
    block_m: int,
    waves_per_eu: int,
    m_rows: int | None = None,
) -> str | None:
    """Why the kernel cannot serve this configuration, or None if it can.

    Every precondition is written once here and reaches callers two ways: as an
    exception via :func:`_validate`, as a bool via
    :func:`gather_kv_b_proj_flydsl_supported`. A second copy for the predicate
    would drift, and drift reads as "declines a shape it handles" or, worse,
    "accepts one it does not".
    """
    block_n = nope + v_dim  # BLOCK_N is one head
    if block_m < 128 or block_m % 128 != 0:
        return f"BLOCK_M must be >=128 and %128==0, got {block_m}"
    if nope != 128 or v_dim != 128:
        return (
            f"this backend requires qk_nope_head_dim == v_head_dim == 128 (the "
            f"k/v split is the MFMA accumulator-group boundary, not a runtime "
            f"offset), got nope={nope} v_head_dim={v_dim}"
        )
    if int(waves_per_eu) < 1:
        return (
            f"waves_per_eu must be >=1 (the kernel always emits the "
            f"rocdl.waves_per_eu attribute), got {waves_per_eu}"
        )
    need = lds_bytes(block_m, block_n)
    have = get_lds_capacity_bytes(get_rocm_arch().split(":", 1)[0])
    if need > have:
        return f"BLOCK_M={block_m} needs {need} B of LDS, limit is {have} B"
    # No num_blocks ceiling: past _BUFFER_SPAN_MAX the kernel addresses the
    # cache in 64 bits per lane instead (`wide_index`).
    if m_rows is not None:
        if m_rows < 0:
            return f"num_tokens must be >=0, got {m_rows}"
        if m_rows * n_heads * (nope + KV_PE_DIM) >= _I32_MAX:
            return (
                f"num_tokens={m_rows} x {n_heads} heads overflows 32-bit output "
                f"indexing"
            )
    return None


def _raise(reason: str) -> None:
    raise ValueError(f"[FlyDSL gather_kv_b_proj] {reason}")


def _validate(**kwargs) -> None:
    """:func:`_config_reason` for the callers with nowhere to fall back."""
    reason = _config_reason(**kwargs)
    if reason is not None:
        _raise(reason)


def _is_per_row_scale(kv_proj_scale: Tensor) -> bool:
    return kv_proj_scale.dim() == 1 or (
        kv_proj_scale.dim() == 2 and kv_proj_scale.shape[1] == 1
    )


def _unsupported_reason(
    k_buffer: Tensor,
    kv_proj_weight: Tensor,
    kv_proj_scale: Tensor | None,
    k_prefix: Tensor,
    v_prefix: Tensor,
    *,
    shuffled_kv_cache: bool = False,
    block_m: int = 256,
    waves_per_eu: int = 2,
) -> str | None:
    """Why this backend cannot serve these tensors, or None if it can.

    Everything here is fixed once the weights are loaded and the cache is
    allocated, so a caller that wants to route around this backend can ask once
    and keep the answer. What is left inside the op is per-call.
    """
    if shuffled_kv_cache:
        return (
            "shuffled_kv_cache is not supported (the gather assumes each slot's "
            f"{KV_ROW_ELEMS} latents are contiguous)"
        )
    if kv_proj_scale is None:
        return (
            "an unquantized weight (kv_proj_scale=None) is not supported; this "
            "backend is fp8 x per-output-row scale only"
        )
    if k_buffer.dim() != 3 or k_buffer.shape[1] != 1:
        return (
            f"k_buffer must be [num_blocks, 1, {KV_ROW_ELEMS}] (page_size 1), got "
            f"{tuple(k_buffer.shape)}"
        )
    if k_buffer.shape[2] != KV_ROW_ELEMS:
        return f"k_buffer last dim must be {KV_ROW_ELEMS}, got {k_buffer.shape[2]}"

    arch = _arch_of(k_buffer.device.index)
    if arch != "gfx950":
        return f"gfx950 only (OCP e4m3 + CDNA4 MFMA_Scale + 128 KB LDS), got {arch}"
    for name, t in (("k_buffer", k_buffer), ("kv_proj_weight", kv_proj_weight)):
        if t.dtype != torch.float8_e4m3fn:
            return f"{name} must be torch.float8_e4m3fn (OCP e4m3), got {t.dtype}"
    if k_prefix.dim() != 3 or v_prefix.dim() != 3:
        return (
            f"outputs must be 3-D, got {tuple(k_prefix.shape)}, "
            f"{tuple(v_prefix.shape)}"
        )
    if k_prefix.dtype != torch.bfloat16 or v_prefix.dtype != torch.bfloat16:
        return "outputs must be bf16"

    total_kv, n_heads, kp_dim = k_prefix.shape
    total_kv_v, n_heads_v, v_dim = v_prefix.shape
    if (total_kv, n_heads) != (total_kv_v, n_heads_v):
        return (
            f"k_prefix / v_prefix disagree: {tuple(k_prefix.shape)} vs "
            f"{tuple(v_prefix.shape)}"
        )
    nope = kp_dim - KV_PE_DIM
    weight_n, weight_k = kv_proj_weight.shape
    if weight_k != KV_C_DIM:
        return f"weight K must be {KV_C_DIM}, got {weight_k}"
    if weight_n != n_heads * (nope + v_dim):
        return (
            f"weight N={weight_n} != n_heads*(nope+v_dim)="
            f"{n_heads}*({nope}+{v_dim})"
        )

    if _is_per_row_scale(kv_proj_scale):
        if kv_proj_scale.numel() != weight_n:
            return (
                f"per-row kv_proj_scale must have {weight_n} elements, got "
                f"{tuple(kv_proj_scale.shape)}"
            )
    elif kv_proj_scale.dim() != 2:
        return (
            f"kv_proj_scale must be 1-D (per-row) or 2-D (block), got "
            f"{tuple(kv_proj_scale.shape)}"
        )
    elif (
        kv_proj_scale.shape[0] * 128 != weight_n
        or kv_proj_scale.shape[1] * 128 != KV_C_DIM
    ):
        return (
            f"block kv_proj_scale must be [{weight_n // 128}, "
            f"{KV_C_DIM // 128}] (128x128 granularity), got "
            f"{tuple(kv_proj_scale.shape)}"
        )

    return _config_reason(
        n_heads=n_heads,
        nope=nope,
        v_dim=v_dim,
        block_m=block_m,
        waves_per_eu=waves_per_eu,
    )


def gather_kv_b_proj_flydsl_supported(*args, **kwargs) -> bool:
    """Would :func:`gather_kv_b_proj_flydsl` serve these tensors?

    For callers holding a fallback -- the Triton op takes the same arguments and
    covers bf16 caches, unquantized and MXFP4 weights, page_size > 1 and every
    non-gfx950 arch. Ask once: the answer is fixed by the weights and the cache,
    not by the call. Arguments are :func:`_unsupported_reason`'s.
    """
    return _unsupported_reason(*args, **kwargs) is None


@functools.lru_cache(maxsize=64)
def compile_gather_kv_b_proj(
    *,
    n_heads: int,
    nope: int,
    v_dim: int,
    block_m: int,
    waves_per_eu: int,
    xcd_swizzle: int,
    weight_preshuffle: bool,
    per_row_scale: bool,
    wide_index: bool,
):
    """Compile (and memoize) a gather+proj launcher."""
    _validate(
        n_heads=n_heads,
        nope=nope,
        v_dim=v_dim,
        block_m=block_m,
        waves_per_eu=waves_per_eu,
    )
    return compile_gather_kv_b_proj_8w(
        n_heads=int(n_heads),
        nope=int(nope),
        v_dim=int(v_dim),
        BLOCK_M=int(block_m),
        waves_per_eu=int(waves_per_eu),
        xcd_swizzle=int(xcd_swizzle),
        weight_preshuffle=bool(weight_preshuffle),
        per_row_scale=bool(per_row_scale),
        wide_index=bool(wide_index),
    )


@functools.lru_cache(maxsize=16)
def _arch_of(device_index: int) -> str:
    """Memoized: ``get_device_properties`` is a per-call host cost of tens of us,
    which dwarfs the kernel itself at small M."""
    return torch.cuda.get_device_properties(device_index).gcnArchName.split(":")[0]


def _as_i8(t: Tensor) -> Tensor:
    """Bitcast fp8 storage to int8; the kernel recasts the iterator back."""
    return t.view(torch.int8) if "float8" in str(t.dtype) else t


_NUM_XCDS = 8


def _default_xcd(m_rows: int) -> int:
    """Pick xcd_swizzle -- the ``wgm`` of the XCD tile remap -- from the row count."""
    num_pid_m = -(-int(m_rows) // 256)
    if num_pid_m <= 8:
        return 0
    if num_pid_m <= 48:
        return 2
    if num_pid_m <= 64:
        return 4
    return _NUM_XCDS


def gather_kv_b_proj_flydsl(
    k_buffer: Tensor,  # [num_blocks, 1, 576] fp8
    k_scale: Tensor,  # [1] fp32, per-tensor activation scale
    kv_indptr: Tensor,  # unused, kept for signature parity with the Triton op
    kv_indices: Tensor,  # [total_kv] int32, one cache slot per token
    kv_prefix_sum_context_lens: Tensor,  # unused, see kv_indptr
    kv_proj_weight: Tensor,  # [n_heads*256, 512] fp8, shuffle_weight(w, (16,16))
    kv_proj_scale: Tensor,  # [n_heads*256] or [n_heads*256, 1] fp32, per-row
    k_prefix: Tensor,  # [total_kv, n_heads, 192] bf16, written in place
    v_prefix: Tensor,  # [total_kv, n_heads, 128] bf16, written in place
    *,
    num_tokens: int | None = None,
    weight_preshuffle: bool = True,
    shuffled_kv_cache: bool = False,
    block_m: int = 256,
    waves_per_eu: int = 2,
    xcd_swizzle: int | None = None,
) -> None:
    """Fused gather + kv_b_proj + rope copy. Writes k_prefix / v_prefix in place.

    ``kv_indptr`` and ``kv_prefix_sum_context_lens`` are accepted but unused --
    with page_size 1 the output row index *is* the token index, which is why the
    Triton flat kernel ignores them too. They stay in the signature so this is a
    positional drop-in for ``aiter.ops.triton.gather_kv_b_proj``.

    ``num_tokens`` is the live row count when the caller preallocates the chunk
    workspace at its maximum; it defaults to ``k_prefix.shape[0]``. Rows past it
    are neither read nor written -- their gathered indices are clamped by the
    kv_indices descriptor and their stores are dropped by the output descriptor.

    ``xcd_swizzle`` defaults to ``None``, which lets :func:`_default_xcd` pick
    it from the live row count; pass an int to pin it. It is a compile-time
    constant, so a workload spanning several row counts compiles one kernel per
    distinct value.
    """

    reason = _unsupported_reason(
        k_buffer,
        kv_proj_weight,
        kv_proj_scale,
        k_prefix,
        v_prefix,
        shuffled_kv_cache=shuffled_kv_cache,
        block_m=block_m,
        waves_per_eu=waves_per_eu,
    )
    if reason is not None:
        _raise(reason)

    num_blocks = k_buffer.shape[0]
    total_kv, n_heads, kp_dim = k_prefix.shape
    nope = kp_dim - KV_PE_DIM
    v_dim = v_prefix.shape[2]

    m_rows = int(total_kv if num_tokens is None else num_tokens)
    # Per-call, so not in `_unsupported_reason`: violating either is a caller
    # bug, not a shape this backend declines, and raising is the right answer.
    if m_rows > total_kv:
        _raise(f"num_tokens={m_rows} exceeds the allocated {total_kv} output rows")
    if m_rows == 0:
        return
    if kv_indices.numel() < m_rows:
        _raise(
            f"kv_indices has {kv_indices.numel()} entries, need at least "
            f"num_tokens={m_rows}"
        )

    per_row_scale = _is_per_row_scale(kv_proj_scale)
    scale = kv_proj_scale.reshape(-1)
    if scale.dtype != torch.float32:
        scale = scale.to(torch.float32)

    xcd_swizzle = _default_xcd(m_rows) if xcd_swizzle is None else int(xcd_swizzle)

    # The config half already ran in `_unsupported_reason`; this is here for
    # `m_rows`, which only exists once `num_tokens` is resolved.
    _validate(
        n_heads=n_heads,
        nope=nope,
        v_dim=v_dim,
        block_m=block_m,
        waves_per_eu=waves_per_eu,
        m_rows=m_rows,
    )

    exe = compile_gather_kv_b_proj(
        n_heads=int(n_heads),
        nope=int(nope),
        v_dim=int(v_dim),
        block_m=int(block_m),
        waves_per_eu=int(waves_per_eu),
        xcd_swizzle=int(xcd_swizzle),
        weight_preshuffle=bool(weight_preshuffle),
        per_row_scale=bool(per_row_scale),
        # This cache's own extent, as its own compile-time variant. The two
        # measure the same, so the split is not for speed: the descriptor form
        # keeps the hardware bounds check, which the wide form gives up.
        wide_index=(num_blocks * KV_ROW_ELEMS >= _BUFFER_SPAN_MAX),
    )

    # Local: `ptr_arg` keeps only the address, so a `.contiguous()` temporary
    # has to outlive the launch.
    kv_cache = k_buffer.contiguous()
    _run_compiled(
        exe,
        ptr_arg(kv_cache),
        num_blocks,
        kv_indices.contiguous().view(-1),
        _as_i8(kv_proj_weight.contiguous()).view(-1),
        scale.contiguous(),
        k_scale.reshape(-1).to(torch.float32).contiguous(),
        k_prefix.view(-1),
        v_prefix.view(-1),
        m_rows,
        fx.Stream(torch.cuda.current_stream(device=k_buffer.device)),
    )
