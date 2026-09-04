# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL backend for the DeepSeek MLA fused gather + kv_b_proj expansion (gfx950).

Supported: fp8 KV cache (OCP e4m3), fp8 weight in either row-major or
``shuffle_weight((16,16))`` layout, per-output-row *or* 128x128 block weight
scale, per-tensor activation scale, page_size 1, bf16 outputs, gfx950.
"""

import functools

import flydsl.expr as fx
import torch
from flydsl.runtime.device import get_rocm_arch
from torch import Tensor

from aiter.jit.utils.chip_info import get_lds_capacity_bytes

from .kernels.gather_kv_b_proj_8wave import compile_gather_kv_b_proj_8w
from .kernels.tensor_shim import _run_compiled

# The MLA latent layout, fixed by the model.
KV_C_DIM = 512
KV_PE_DIM = 64
KV_ROW_ELEMS = KV_C_DIM + KV_PE_DIM

# LDS = 4 A buffers of (BM/2)x128 plus 4 B buffers of (BN/2)x128, 1 byte/elem.
_LDS_BYTES_PER_BLOCK_UNIT = 256
_I32_MAX = 2**31


def lds_bytes(block_m: int, block_n: int) -> int:
    return _LDS_BYTES_PER_BLOCK_UNIT * (int(block_m) + int(block_n))


def _validate(
    *,
    n_heads: int,
    nope: int,
    v_dim: int,
    block_m: int,
    waves_per_eu: int,
    num_blocks: int | None = None,
    m_rows: int | None = None,
) -> None:
    """Re-check every kernel precondition as ValueError."""
    block_n = nope + v_dim  # BLOCK_N is one head
    if block_m < 128 or block_m % 128 != 0:
        raise ValueError(
            f"[FlyDSL gather_kv_b_proj] BLOCK_M must be >=128 and %128==0, got {block_m}"
        )
    if nope != 128 or v_dim != 128:
        raise ValueError(
            f"[FlyDSL gather_kv_b_proj] this backend requires qk_nope_head_dim == "
            f"v_head_dim == 128 (the k/v split is the MFMA accumulator-group "
            f"boundary, not a runtime offset), got nope={nope} v_head_dim={v_dim}. "
            f"Use the Triton op for other head dims."
        )
    if int(waves_per_eu) < 1:
        raise ValueError(
            f"[FlyDSL gather_kv_b_proj] waves_per_eu must be >=1 (the kernel always "
            f"emits the rocdl.waves_per_eu attribute), got {waves_per_eu}"
        )
    need = lds_bytes(block_m, block_n)
    have = get_lds_capacity_bytes(get_rocm_arch().split(":", 1)[0])
    if need > have:
        raise ValueError(
            f"[FlyDSL gather_kv_b_proj] BLOCK_M={block_m} needs {need} B of LDS, "
            f"limit is {have} B"
        )
    # Every gathered address and every output index is computed in 32-bit.
    if num_blocks is not None and num_blocks * KV_ROW_ELEMS >= _I32_MAX:
        raise ValueError(
            f"[FlyDSL gather_kv_b_proj] num_blocks={num_blocks} x {KV_ROW_ELEMS} "
            f"overflows 32-bit buffer indexing"
        )
    if m_rows is not None:
        if m_rows < 0:
            raise ValueError(
                f"[FlyDSL gather_kv_b_proj] num_tokens must be >=0, got {m_rows}"
            )
        if m_rows * n_heads * (nope + KV_PE_DIM) >= _I32_MAX:
            raise ValueError(
                f"[FlyDSL gather_kv_b_proj] num_tokens={m_rows} x {n_heads} heads "
                f"overflows 32-bit output indexing"
            )


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

    if shuffled_kv_cache:
        raise ValueError(
            "[FlyDSL gather_kv_b_proj] shuffled_kv_cache is not supported (the "
            "gather assumes each slot's 576 latents are contiguous). "
        )
    if kv_proj_scale is None:
        raise ValueError(
            "[FlyDSL gather_kv_b_proj] an unquantized weight (kv_proj_scale=None) "
            "is not supported; this backend is fp8 x per-output-row scale only. "
        )

    if k_buffer.dim() != 3 or k_buffer.shape[1] != 1:
        raise ValueError(
            f"[FlyDSL gather_kv_b_proj] k_buffer must be [num_blocks, 1, {KV_ROW_ELEMS}] "
            f"(page_size 1), got {tuple(k_buffer.shape)}. Use the Triton op for "
            f"page_size > 1."
        )
    num_blocks, _, hidden = k_buffer.shape
    if hidden != KV_ROW_ELEMS:
        raise ValueError(
            f"[FlyDSL gather_kv_b_proj] k_buffer last dim must be {KV_ROW_ELEMS}, got {hidden}"
        )

    arch = _arch_of(k_buffer.device.index)
    if arch != "gfx950":
        raise ValueError(
            f"[FlyDSL gather_kv_b_proj] gfx950 only (OCP e4m3 + CDNA4 MFMA_Scale + "
            f"128 KB LDS), got {arch}."
        )
    for name, t in (("k_buffer", k_buffer), ("kv_proj_weight", kv_proj_weight)):
        if t.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"[FlyDSL gather_kv_b_proj] {name} must be torch.float8_e4m3fn "
                f"(OCP e4m3), got {t.dtype}"
            )
    if k_prefix.dim() != 3 or v_prefix.dim() != 3:
        raise ValueError(
            f"[FlyDSL gather_kv_b_proj] outputs must be 3-D, got "
            f"{tuple(k_prefix.shape)}, {tuple(v_prefix.shape)}"
        )
    if k_prefix.dtype != torch.bfloat16 or v_prefix.dtype != torch.bfloat16:
        raise ValueError("[FlyDSL gather_kv_b_proj] outputs must be bf16")

    total_kv, n_heads, kp_dim = k_prefix.shape
    total_kv_v, n_heads_v, v_dim = v_prefix.shape
    if (total_kv, n_heads) != (total_kv_v, n_heads_v):
        raise ValueError(
            f"[FlyDSL gather_kv_b_proj] k_prefix / v_prefix disagree: "
            f"{tuple(k_prefix.shape)} vs {tuple(v_prefix.shape)}"
        )
    nope = kp_dim - KV_PE_DIM
    weight_n, weight_k = kv_proj_weight.shape
    if weight_k != KV_C_DIM:
        raise ValueError(
            f"[FlyDSL gather_kv_b_proj] weight K must be {KV_C_DIM}, got {weight_k}"
        )
    if weight_n != n_heads * (nope + v_dim):
        raise ValueError(
            f"[FlyDSL gather_kv_b_proj] weight N={weight_n} != n_heads*(nope+v_dim)="
            f"{n_heads}*({nope}+{v_dim})"
        )

    m_rows = int(total_kv if num_tokens is None else num_tokens)
    if m_rows > total_kv:
        raise ValueError(
            f"[FlyDSL gather_kv_b_proj] num_tokens={m_rows} exceeds the allocated "
            f"{total_kv} output rows"
        )
    if m_rows == 0:
        return
    if kv_indices.numel() < m_rows:
        raise ValueError(
            f"[FlyDSL gather_kv_b_proj] kv_indices has {kv_indices.numel()} entries, "
            f"need at least num_tokens={m_rows}"
        )

    per_row_scale = kv_proj_scale.dim() == 1 or (
        kv_proj_scale.dim() == 2 and kv_proj_scale.shape[1] == 1
    )
    if per_row_scale:
        if kv_proj_scale.numel() != weight_n:
            raise ValueError(
                f"[FlyDSL gather_kv_b_proj] per-row kv_proj_scale must have "
                f"{weight_n} elements, got {tuple(kv_proj_scale.shape)}"
            )
    else:
        if kv_proj_scale.dim() != 2:
            raise ValueError(
                f"[FlyDSL gather_kv_b_proj] kv_proj_scale must be 1-D (per-row) or "
                f"2-D (block), got {tuple(kv_proj_scale.shape)}"
            )
        scale_n, scale_k = kv_proj_scale.shape
        if scale_n * 128 != weight_n or scale_k * 128 != KV_C_DIM:
            raise ValueError(
                f"[FlyDSL gather_kv_b_proj] block kv_proj_scale must be "
                f"[{weight_n // 128}, {KV_C_DIM // 128}] (128x128 granularity), "
                f"got {tuple(kv_proj_scale.shape)}"
            )
    scale = kv_proj_scale.reshape(-1)
    if scale.dtype != torch.float32:
        scale = scale.to(torch.float32)

    xcd_swizzle = _default_xcd(m_rows) if xcd_swizzle is None else int(xcd_swizzle)

    _validate(
        n_heads=n_heads,
        nope=nope,
        v_dim=v_dim,
        block_m=block_m,
        waves_per_eu=waves_per_eu,
        num_blocks=num_blocks,
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
    )

    _run_compiled(
        exe,
        _as_i8(k_buffer.contiguous()).view(-1),
        kv_indices.contiguous().view(-1),
        _as_i8(kv_proj_weight.contiguous()).view(-1),
        scale.contiguous(),
        k_scale.reshape(-1).to(torch.float32).contiguous(),
        k_prefix.view(-1),
        v_prefix.view(-1),
        m_rows,
        fx.Stream(torch.cuda.current_stream(device=k_buffer.device)),
    )
