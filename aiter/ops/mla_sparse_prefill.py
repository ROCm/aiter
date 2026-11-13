# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MLA sparse prefill (DeepSeek DSA): one softmax per query token over its rows of a
paged prefix source (``unified_kv``) and of a flat extend source (``kv``), with a
per-head sink in the denominator only.

One public entry per KV layout: the name carries the layout (the model version that
introduced it) and the tensors' dtypes carry the precision, so a new precision of a
layout is a new dtype case, not a new entry.

* :func:`mla_sparse_prefill`     -- unscaled rows (bf16): opus.
* :func:`mla_sparse_prefill_v4`  -- v4: NoPE with its E8M0 scales in the row, plus bf16
  RoPE (NoPE fp8): asm on gfx1250 with H == 128 (opus if it declines), else opus.
* :func:`mla_sparse_prefill_v41` -- v4.1: rows with separate ``[*, 16]`` E8M0 scales
  (q, kv fp8), the prefix source fp8 or NVFP4 with ``[*, 32]`` e4m3 scales: opus on gfx950,
  and with an fp8 prefix on gfx1250; asm for H64 NVFP4 on gfx1250.

``backend`` forces one backend. A configuration no backend serves raises RuntimeError;
a bad argument raises ValueError.
"""

import functools
import math
import struct
from numbers import Real
from typing import NamedTuple

import torch

from .. import logger
from .. import plugin as _plugin
from ..jit.core import compile_ops
from ..jit.utils.asm_guard import is_gfx1250_asm_supported, require_gfx1250_asm
from ..jit.utils.chip_info import get_cu_num, get_gfx_runtime
from ..jit.utils.torch_guard import torch_compile_guard
from ..plugin import route as _route
from ..utility import dtypes
from .mla_sparse_prefill_v41_asm import _asm_sparse_prefill_v41
from .opus.mla import (
    opus_mla_v4_prefill_a8w8_gfx950_fwd,
    opus_mla_v4_prefill_a8w8_gfx1250_fwd,
    opus_mla_v4_prefill_a16w16_gfx950_fwd,
    opus_mla_v4_prefill_a16w16_gfx1250_fwd,
    opus_mla_v41_prefill_a8w4_gfx950_fwd,
    opus_mla_v41_prefill_a8w8_gfx950_fwd,
    opus_mla_v41_prefill_a8w8_gfx1250_fwd,
)

MD_NAME = "module_mla_sparse_prefill_asm"

# Baked into the asm code object.
_ASM_HEADS = 128


def _check_backend(op: str, backend: str | None, backends: tuple) -> None:
    if backend is not None and backend not in backends:
        raise ValueError(
            f"{op}: backend must be one of {backends} or None, got {backend!r}"
        )


def _out_bf16(t: int, h: int, device, out: torch.Tensor | None) -> torch.Tensor:
    if out is None:
        return torch.empty((t, h, 512), dtype=torch.bfloat16, device=device)
    if out.shape != (t, h, 512) or out.dtype != torch.bfloat16:
        raise RuntimeError(
            f"out shape/dtype mismatch: got shape={tuple(out.shape)} dtype={out.dtype}, "
            f"expected shape={(t, h, 512)} dtype={torch.bfloat16}"
        )
    return out


# ---------------------------------------------------------------------------
# asm (gfx1250)
# ---------------------------------------------------------------------------


# NOTE: ctypes binds positionally off this signature -- the argument order here
# must match `mla_sparse_prefill_fp8_asm_fwd` in
# csrc/py_itfs_cu/asm_mla_sparse_prefill.cu one for one. A silent reorder here
# is a silent wrong-pointer launch, not a compile error.
@compile_ops(MD_NAME, fc_name="mla_sparse_prefill_fp8_asm_fwd", ffi_type="ctypes")
def mla_sparse_prefill_fp8_asm_fwd(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    unified_kv_nope: torch.Tensor,
    unified_kv_rope: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
) -> None: ...


def _asm_sparse_prefill_v4(
    q_nope,
    q_rope,
    unified_kv_nope,
    unified_kv_rope,
    kv_indices_prefix,
    kv_indptr_prefix,
    kv_nope,
    kv_rope,
    kv_indices_extend,
    kv_indptr_extend,
    attn_sink,
    softmax_scale,
    out,
):
    gfx = get_gfx_runtime()
    if gfx != "gfx1250":
        raise RuntimeError(f"mla_sparse_prefill_fp8_asm requires gfx1250, got {gfx}")
    require_gfx1250_asm("mla_sparse_prefill_fp8_asm")
    if attn_sink is None:
        raise RuntimeError("mla_sparse_prefill_fp8_asm requires attn_sink")

    if q_nope.dtype != unified_kv_nope.dtype or q_nope.dtype != kv_nope.dtype:
        raise RuntimeError(
            f"NoPE dtype mismatch: q_nope={q_nope.dtype}, "
            f"unified_kv_nope={unified_kv_nope.dtype}, kv_nope={kv_nope.dtype}"
        )
    if q_rope.dtype != torch.bfloat16:
        raise RuntimeError(f"q_rope must be bf16, got {q_rope.dtype}")

    t, h = q_nope.shape[0], q_nope.shape[1]
    if h != _ASM_HEADS:
        # Hard constraint, not a dispatch miss.
        raise RuntimeError(f"mla_sparse_prefill_fp8_asm requires H == 128, got {h}")
    out = _out_bf16(t, h, q_nope.device, out)

    mla_sparse_prefill_fp8_asm_fwd(
        q_nope,
        q_rope,
        unified_kv_nope,
        unified_kv_rope,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv_nope,
        kv_rope,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        out,
        float(softmax_scale),
    )
    return out


def _mla_sparse_prefill_fp8_asm_fake(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    unified_kv_nope: torch.Tensor,
    unified_kv_rope: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is not None:
        return out
    t, h, _ = q_nope.shape
    return torch.empty((t, h, 512), dtype=torch.bfloat16, device=q_nope.device)


@torch_compile_guard(mutates_args=["out"], gen_fake=_mla_sparse_prefill_fp8_asm_fake)
def mla_sparse_prefill_fp8_asm(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    unified_kv_nope: torch.Tensor,
    unified_kv_rope: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sparse prefill attention with split fp8 NoPE and bf16 RoPE inputs (asm).

    Signature-compatible with :func:`mla_sparse_prefill_v4` and
    :func:`aiter.ops.pa_sparse_prefill_opus.pa_sparse_prefill_fp8_opus`.

    Args:
      q_nope:            ``[T, H, 512]`` fp8 query without positional encoding.
      q_rope:            ``[T, H, 64]`` bf16 query RoPE encoding part.
      unified_kv_nope:   ``[total_pages, 512]`` fp8 prefix KV NoPE source.
      unified_kv_rope:   ``[total_pages, 64]`` bf16 prefix KV RoPE source.
      kv_indices_prefix: ``[total_prefix]`` int32 row indices into the prefix
        sources, concatenated per token.
      kv_indptr_prefix:  ``[T+1]`` int32 CSR row pointers.
      kv_nope:           ``[total_tokens, 512]`` fp8 extend KV NoPE source.
      kv_rope:           ``[total_tokens, 64]`` bf16 extend KV RoPE source.
      kv_indices_extend: ``[total_extend]`` int32 row indices into the extend
        sources, concatenated per token.
      kv_indptr_extend:  ``[T+1]`` int32 CSR row pointers.
      attn_sink:         ``[H]`` fp32 per-head softmax-denom bias.
      softmax_scale:     float scalar applied to the combined QK^T scores.
      out:               Optional ``[T, H, 512]`` bf16 output buffer; allocated
        if ``None``.

    Returns:
      ``out`` (``[T, H, 512]`` bf16).
    """
    return _asm_sparse_prefill_v4(
        q_nope,
        q_rope,
        unified_kv_nope,
        unified_kv_rope,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv_nope,
        kv_rope,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        softmax_scale,
        out,
    )


# ---------------------------------------------------------------------------
# opus
# ---------------------------------------------------------------------------


def _opus_fwd(gfx: str, op_gfx950, op_gfx1250):
    """gfx950 compiles the kernel from source; gfx1250 loads a prebuilt code object,
    as its kernel needs a scheduler that release toolchains do not ship."""
    if gfx == "gfx1250":
        return op_gfx1250
    if gfx == "gfx950":
        return op_gfx950
    raise RuntimeError(
        f"opus MLA sparse prefill supports gfx950 and gfx1250, got {gfx}"
    )


# Auto split cost in keys: one workgroup's walk (a token's keys over S splits, in whole KV tiles),
# plus c0 + c1 * N * H * S to fold when split.
class _SplitFit(NamedTuple):
    heads_per_wg: int
    kv_tile: int
    wgs_per_cu: int  # workgroups per CU a split may fill
    counts: tuple  # the split counts auto picks from
    c0: float
    c1: float
    packed: float = 1.0  # walk factor past one workgroup per CU


# The reduce gives each split a lane of one wave, so a wave32 arch caps the count.
_MAX_KV_SPLITS = 32
_ANY_COUNT = tuple(range(2, _MAX_KV_SPLITS + 1))
# Split counts a gfx1250 v4.1 cluster folds in-kernel.
_CLUSTER_COUNT = (2, 4, 8, 16)
# Split counts the gfx1250 v4 16mx1 and fp8 32mx1 fits were measured on.
_FIT_COUNT = (2, 3, 4, 6, 8, 12, 16)

# (arch, kernel) -> fit over H 16-128, topk 384-4096 and N 8-768 on gfx950, topk 512-4096 and
# N 8-512 on gfx1250. v41_a8w4_16mx1_16nx4 reuses the v41_a8w8 fit.
_KV_SPLIT_FIT = {
    ("gfx950", "v4_a16w16_16mx1_16nx4"): _SplitFit(16, 64, 2, _ANY_COUNT, 90, 0.043),
    ("gfx950", "v4_a16w16_32mx4_32nx1"): _SplitFit(128, 32, 1, _ANY_COUNT, 150, 0.014),
    ("gfx950", "v4_a8w8_16mx1_16nx4"): _SplitFit(16, 64, 2, _ANY_COUNT, 55, 0.022),
    ("gfx950", "v4_a8w8_32mx4_32nx1"): _SplitFit(128, 32, 1, _ANY_COUNT, 150, 0.014),
    ("gfx950", "v41_a8w8_16mx1_16nx4"): _SplitFit(16, 64, 2, _ANY_COUNT, 80, 0.046),
    ("gfx950", "v41_a8w8_16mx4_64nx1"): _SplitFit(64, 64, 1, _ANY_COUNT, 90, 0.025),
    ("gfx950", "v41_a8w4_16mx1_16nx4"): _SplitFit(16, 64, 2, _ANY_COUNT, 80, 0.046),
    ("gfx950", "v41_a8w4_16mx4_64nx1"): _SplitFit(64, 64, 1, _ANY_COUNT, 250, 0.012),
    ("gfx1250", "v4_a16w16_16mx1_16nx4"): _SplitFit(16, 64, 1, _FIT_COUNT, 250, 0.08),
    ("gfx1250", "v4_a16w16_32mx1_16nx4"): _SplitFit(32, 64, 1, _ANY_COUNT, 200, 0.06),
    ("gfx1250", "v4_a16w16_16mx4_64nx1"): _SplitFit(64, 64, 1, _ANY_COUNT, 150, 0.06),
    ("gfx1250", "v4_a8w8_16mx1_16nx4"): _SplitFit(
        16, 64, 2, _FIT_COUNT, 350, 0.06, 1.25
    ),
    ("gfx1250", "v4_a8w8_32mx1_16nx4"): _SplitFit(
        32, 64, 2, _FIT_COUNT, 200, 0.04, 1.25
    ),
    ("gfx1250", "v4_a8w8_16mx4_64nx1"): _SplitFit(64, 64, 1, _ANY_COUNT, 200, 0.06),
    ("gfx1250", "v41_a8w8_16mx1_16nx4"): _SplitFit(
        16, 64, 2, _CLUSTER_COUNT, 275, 0.03, 1.75
    ),
    ("gfx1250", "v41_a8w8_16mx4_64nx1"): _SplitFit(64, 64, 1, _CLUSTER_COUNT, 0, 0.05),
}

# Fallback for prefix lengths outside the ASM short-query policy.
_ASM_V41_INITIAL_SPLIT_FIT = _SplitFit(64, 64, 1, tuple(range(2, 17)), 290, 0.045)


def _opus_split_fit(gfx: str, kernel: str, h: int) -> _SplitFit:
    """The split-K fit of the ``kernel`` (v4_a16w16, v4_a8w8, v41_a8w8, v41_a8w4) variant that
    serves ``h`` heads, as the launcher picks it."""
    if gfx == "gfx1250" and kernel == "v41_a8w8":
        tile = "16mx1_16nx4" if h <= 32 else "16mx4_64nx1"
    elif gfx == "gfx1250":
        tile = "16mx1_16nx4" if h <= 16 else "32mx1_16nx4" if h <= 32 else "16mx4_64nx1"
    elif kernel in ("v41_a8w8", "v41_a8w4"):
        tile = "16mx1_16nx4" if h <= 16 else "16mx4_64nx1"
    else:
        tile = "16mx1_16nx4" if h <= 32 else "32mx4_32nx1"
    return _KV_SPLIT_FIT[(gfx, f"{kernel}_{tile}")]


@functools.lru_cache(maxsize=1024)
def _auto_kv_splits(fit: _SplitFit, n: int, h: int, nnz: int, cu_num: int) -> int:
    if n == 0:
        return 1
    keys = nnz / n
    blocks = n * -(-h // fit.heads_per_wg)
    max_splits = fit.wgs_per_cu * cu_num // blocks

    def cost(s: int) -> float:
        walk = -(-math.ceil(keys / s) // fit.kv_tile) * fit.kv_tile
        if s > 1 and blocks * s > cu_num:
            walk *= fit.packed
        return walk + (fit.c0 + fit.c1 * n * h * s if s > 1 else 0)

    return min([1, *(s for s in fit.counts if s <= max_splits)], key=cost)


def _asm_v41_kv_splits(n: int, nnz: int, cu_num: int) -> int:
    return _auto_kv_splits(_ASM_V41_INITIAL_SPLIT_FIT, n, 64, nnz, cu_num)


def _kv_splits(
    num_kv_splits: int | None, fit: _SplitFit, n: int, h: int, nnz: int
) -> int:
    """The split count. Auto reads host scalars only (``nnz``: all tokens' rows, from the
    index tensors' sizes), so a captured graph's choice is its shape's."""
    if num_kv_splits is None:
        return _auto_kv_splits(fit, n, h, nnz, get_cu_num())
    if not 1 <= num_kv_splits <= _MAX_KV_SPLITS:
        # ValueError, a caller's error; RuntimeError would read as a missing kernel.
        raise ValueError(
            f"num_kv_splits must be in [1, {_MAX_KV_SPLITS}], got {num_kv_splits}"
        )
    return num_kv_splits


def _split_args(n: int, num_kv_splits: int, h: int, device) -> tuple:
    """The opus kernels' trailing split-K arguments, none without split-K: num_splits,
    partial_o [N, S, H, 512], partial_max and partial_sum [N, S, H], fp32."""
    if num_kv_splits == 1:
        return ()
    partial_o = torch.empty(
        n, num_kv_splits, h, 512, dtype=torch.float32, device=device
    )
    partial_max = torch.empty(n, num_kv_splits, h, dtype=torch.float32, device=device)
    return num_kv_splits, partial_o, partial_max, torch.empty_like(partial_max)


def _opus_sparse_prefill(
    q,
    unified_kv,
    kv_indices_prefix,
    kv_indptr_prefix,
    kv,
    kv_indices_extend,
    kv_indptr_extend,
    attn_sink,
    softmax_scale,
    out,
    num_kv_splits,
):
    gfx = get_gfx_runtime()
    fwd = _opus_fwd(
        gfx,
        opus_mla_v4_prefill_a16w16_gfx950_fwd,
        opus_mla_v4_prefill_a16w16_gfx1250_fwd,
    )

    if q.dtype != torch.bfloat16:
        raise RuntimeError(f"expects bf16 q, got {q.dtype}")
    if unified_kv.dtype != q.dtype:
        raise RuntimeError(
            f"unified_kv dtype mismatch: unified_kv={unified_kv.dtype}, q={q.dtype}"
        )
    if kv.dtype != q.dtype:
        raise RuntimeError(f"kv dtype mismatch: kv={kv.dtype}, q={q.dtype}")
    if unified_kv.size(-1) != kv.size(-1):
        raise RuntimeError(
            f"head_dim mismatch: unified_kv={unified_kv.size(-1)}, kv={kv.size(-1)}"
        )

    if out is None:
        out = torch.empty_like(q)
    elif out.shape != q.shape or out.dtype != q.dtype:
        raise RuntimeError(
            f"out shape/dtype mismatch: got shape={tuple(out.shape)} dtype={out.dtype}, "
            f"expected shape={tuple(q.shape)} dtype={q.dtype}"
        )

    n, h, _ = q.shape
    nnz = kv_indices_prefix.numel() + kv_indices_extend.numel()
    fit = _opus_split_fit(gfx, "v4_a16w16", h)
    num_kv_splits = _kv_splits(num_kv_splits, fit, n, h, nnz)
    fwd(
        q,
        unified_kv,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        out,
        float(softmax_scale),
        *_split_args(n, num_kv_splits, h, q.device),
    )
    return out


# NoPE dtype -> (gfx950, gfx1250) kernel.
_OPUS_V4_KERNELS = {
    dtypes.fp8: (
        opus_mla_v4_prefill_a8w8_gfx950_fwd,
        opus_mla_v4_prefill_a8w8_gfx1250_fwd,
    ),
}


def _opus_sparse_prefill_v4(
    q_nope,
    q_rope,
    unified_kv_nope,
    unified_kv_rope,
    kv_indices_prefix,
    kv_indptr_prefix,
    kv_nope,
    kv_rope,
    kv_indices_extend,
    kv_indptr_extend,
    attn_sink,
    softmax_scale,
    out,
    num_kv_splits=None,
):
    if q_nope.dtype != unified_kv_nope.dtype or q_nope.dtype != kv_nope.dtype:
        raise RuntimeError(
            f"NoPE dtype mismatch: q_nope={q_nope.dtype}, "
            f"unified_kv_nope={unified_kv_nope.dtype}, kv_nope={kv_nope.dtype}"
        )
    kernels = _OPUS_V4_KERNELS.get(q_nope.dtype)
    if kernels is None:
        raise RuntimeError(
            f"opus v4 sparse prefill supports NoPE dtypes {list(_OPUS_V4_KERNELS)}, "
            f"got {q_nope.dtype}"
        )
    gfx = get_gfx_runtime()
    fwd = _opus_fwd(gfx, *kernels)
    if q_rope.dtype != torch.bfloat16:
        raise RuntimeError(f"q_rope must be bf16, got {q_rope.dtype}")
    n, h, _ = q_nope.shape
    out = _out_bf16(n, h, q_nope.device, out)
    args = (
        q_nope,
        q_rope,
        unified_kv_nope,
        unified_kv_rope,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv_nope,
        kv_rope,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
    )
    nnz = kv_indices_prefix.numel() + kv_indices_extend.numel()
    fit = _opus_split_fit(gfx, "v4_a8w8", h)
    num_kv_splits = _kv_splits(num_kv_splits, fit, n, h, nnz)
    fwd(
        *args,
        out,
        float(softmax_scale),
        *_split_args(n, num_kv_splits, h, q_nope.device),
    )
    return out


_E8M0 = (torch.uint8, dtypes.fp8_e8m0)

# unified_kv dtype -> ((gfx950, gfx1250) kernel, split-fit variant, unified_kv_scale dtypes);
# q and kv are fp8.
_OPUS_V41_KERNELS = {
    dtypes.fp8: (
        (opus_mla_v41_prefill_a8w8_gfx950_fwd, opus_mla_v41_prefill_a8w8_gfx1250_fwd),
        "v41_a8w8",
        _E8M0,
    ),
    dtypes.fp4x2: (
        (opus_mla_v41_prefill_a8w4_gfx950_fwd, None),
        "v41_a8w4",
        (torch.uint8, dtypes.fp8),
    ),
}


def _v41_runtime_arch(device: torch.device | str) -> str | None:
    device = torch.device(device)
    if device.type != "cuda" or torch.version.hip is None:
        return None
    if not torch.cuda.is_available():
        return None
    index = torch.cuda.current_device() if device.index is None else device.index
    if not 0 <= index < torch.cuda.device_count():
        raise ValueError(f"CUDA device index is out of range: {index}")
    gfx = torch.cuda.get_device_properties(index).gcnArchName.split(":", 1)[0].lower()
    if gfx not in ("gfx950", "gfx1250"):
        return None
    # Native dispatch currently uses a node-wide cached architecture. Do not
    # promise support for a different target on a heterogeneous node.
    return gfx if gfx == get_gfx_runtime() else None


def _v41_asm_eligible(gfx, shape, dtypes_, layout, scale):
    n, h, d = shape
    (
        q_dtype,
        q_scale_dtype,
        prefix_dtype,
        prefix_scale_dtype,
        kv_dtype,
        kv_scale_dtype,
    ) = dtypes_
    return (
        gfx == "gfx1250"
        and h == 64
        and d == 512
        and n <= 65536
        and layout == "contiguous"
        and q_dtype == kv_dtype == dtypes.fp8
        and q_scale_dtype in _E8M0
        and kv_scale_dtype in _E8M0
        and prefix_dtype in (torch.uint8, dtypes.fp4x2)
        and prefix_scale_dtype in (torch.uint8, dtypes.fp8)
        and struct.pack("<f", scale) == struct.pack("<f", 1 / math.sqrt(512))
        and is_gfx1250_asm_supported()
    )


def _v41_opus_eligible(gfx, shape, dtypes_, layout):
    (
        q_dtype,
        q_scale_dtype,
        prefix_dtype,
        prefix_scale_dtype,
        kv_dtype,
        kv_scale_dtype,
    ) = dtypes_
    prefix_dtype = dtypes.fp4x2 if prefix_dtype == torch.uint8 else prefix_dtype
    entry = _OPUS_V41_KERNELS.get(prefix_dtype)
    return (
        gfx in ("gfx950", "gfx1250")
        and shape[2] == 512
        and layout in ("contiguous", "row_strided")
        and q_dtype == kv_dtype == dtypes.fp8
        and q_scale_dtype in _E8M0
        and kv_scale_dtype in _E8M0
        and entry is not None
        and prefix_scale_dtype in entry[2]
        and entry[0][gfx == "gfx1250"] is not None
    )


def get_mla_sparse_prefill_v41_backend(
    device: torch.device | str,
    q_shape: tuple[int, int, int],
    *,
    unified_kv_dtype: torch.dtype,
    softmax_scale: float = 1 / math.sqrt(512),
    q_dtype: torch.dtype = dtypes.fp8,
    q_scale_dtype: torch.dtype = torch.uint8,
    unified_kv_scale_dtype: torch.dtype = torch.uint8,
    kv_dtype: torch.dtype = dtypes.fp8,
    kv_scale_dtype: torch.dtype = torch.uint8,
    layout: str = "contiguous",
) -> str | None:
    """Query the native v4.1 backend before preparing/quantizing Q.

    Returns ``"asm"``, ``"opus"`` or ``None`` when the intended configuration
    is unsupported. ``q_shape`` is the concrete host shape ``(N, H, D)``;
    dtypes describe the intended quantized tensors, not the BF16 Q producer.
    No tensors are allocated, quantized or read, and no kernel is compiled.

    ``layout="contiguous"`` describes separate dense Q/KV/scale arrays.
    ``"row_strided"`` permits unit last-dimension strides with data bases/row
    strides aligned to 16 bytes and scales aligned to 4 bytes (including
    separate views of packed NVFP4 rows). Other layout strings return None.
    Actual tensor shapes, addresses, CSR, output and split workspaces remain
    subject to execution-time validation; this query does not catch launch
    errors or test whether a toolchain can successfully compile a kernel.

    The target must be a live HIP device matching native dispatch's runtime
    architecture. ASM also observes the existing B0+ runtime gate. Malformed
    metadata raises rather than being treated as a supported fallback.
    This query describes AITER's native fallback, not plugin providers that
    may need the actual prepared tensors to decide whether to serve a call.
    """
    if not isinstance(q_shape, (tuple, list)) or len(q_shape) != 3:
        raise ValueError("q_shape must be (N, H, D)")
    if any(type(x) is not int for x in q_shape):
        raise TypeError("q_shape must contain concrete integer dimensions")
    n, h, d = q_shape
    if n < 0 or h <= 0 or d <= 0:
        raise ValueError("q_shape requires N >= 0, H > 0 and D > 0")
    dtype_spec = (
        q_dtype,
        q_scale_dtype,
        unified_kv_dtype,
        unified_kv_scale_dtype,
        kv_dtype,
        kv_scale_dtype,
    )
    if any(not isinstance(dtype, torch.dtype) for dtype in dtype_spec):
        raise TypeError("quantized dtypes must be torch.dtype values")
    if not isinstance(layout, str):
        raise TypeError("layout must be a string")
    if isinstance(softmax_scale, bool) or not isinstance(softmax_scale, Real):
        raise TypeError("softmax_scale must be a host number, not a tensor")
    scale = float(softmax_scale)
    try:
        finite_fp32 = math.isfinite(struct.unpack("<f", struct.pack("<f", scale))[0])
    except (OverflowError, struct.error):
        finite_fp32 = False
    if not finite_fp32:
        raise ValueError("softmax_scale must have a finite FP32 representation")
    if max(n, h) > 2**31 - 1 or d != 512:
        return None
    gfx = _v41_runtime_arch(device)
    if _v41_asm_eligible(gfx, q_shape, dtype_spec, layout, scale):
        return "asm"
    if _v41_opus_eligible(gfx, q_shape, dtype_spec, layout):
        return "opus"
    return None


def _v41_tensor_layout(q, q_scale, unified_kv, unified_kv_scale, kv, kv_scale):
    tensors = (q, q_scale, unified_kv, unified_kv_scale, kv, kv_scale)
    for tensor, alignment in zip(tensors, (16, 4, 16, 4, 16, 4)):
        if tensor.dim() == 0 or tensor.stride(-1) != 1:
            return "unsupported"
        if tensor.data_ptr() % alignment or any(
            stride * tensor.element_size() % alignment
            for stride in tensor.stride()[:-1]
        ):
            return "unsupported"
    return "contiguous" if all(t.is_contiguous() for t in tensors) else "row_strided"


def _opus_sparse_prefill_v41(
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
    if q.dtype != dtypes.fp8 or kv.dtype != dtypes.fp8:
        raise RuntimeError(f"q and kv must be fp8, got q={q.dtype}, kv={kv.dtype}")
    prefix_dtype = dtypes.fp4x2 if unified_kv.dtype == torch.uint8 else unified_kv.dtype
    if prefix_dtype not in _OPUS_V41_KERNELS:
        raise RuntimeError(
            f"mla_sparse_prefill_v41 supports unified_kv dtypes "
            f"{list(_OPUS_V41_KERNELS)}, got {unified_kv.dtype}"
        )
    kernels, variant, prefix_scale_dtypes = _OPUS_V41_KERNELS[prefix_dtype]
    gfx = get_gfx_runtime()
    fwd = _opus_fwd(gfx, *kernels)
    if fwd is None:
        raise RuntimeError(
            f"mla_sparse_prefill_v41: the NVFP4 prefix requires gfx950, got {gfx}"
        )
    for name, scale, scale_dtypes in (
        ("q_scale", q_scale, _E8M0),
        ("unified_kv_scale", unified_kv_scale, prefix_scale_dtypes),
        ("kv_scale", kv_scale, _E8M0),
    ):
        if scale.dtype not in scale_dtypes:
            raise RuntimeError(
                f"{name} must be one of {scale_dtypes}, got {scale.dtype}"
            )
    n, h, _ = q.shape
    out = _out_bf16(n, h, q.device, out)
    args = (
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
    )
    nnz = kv_indices_prefix.numel() + kv_indices_extend.numel()
    fit = _opus_split_fit(gfx, variant, h)
    num_kv_splits = _kv_splits(num_kv_splits, fit, n, h, nnz)
    fwd(*args, out, float(softmax_scale), *_split_args(n, num_kv_splits, h, q.device))
    return out


# ---------------------------------------------------------------------------
# Public entries
# ---------------------------------------------------------------------------


@functools.cache
def _warn_asm_fallback(reason: str) -> None:
    logger.warning(
        f"mla_sparse_prefill_v4: the asm backend declined ({reason}); using opus"
    )


def _mla_sparse_prefill_fake(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    num_kv_splits: int | None = None,
    backend: str | None = None,
) -> torch.Tensor:
    return out if out is not None else torch.empty_like(q)


# Routed here rather than on either public spelling or on a backend binding.
# `mla_sparse_prefill` and `pa_sparse_prefill_opus` are two public names for one
# op and both call this directly, so this is where the op begins. Routing a
# spelling instead would cover only the callers who happen to use that one, and
# routing `_opus_sparse_prefill` would name a backend in the contract.
#
# `backend=` stays ABOVE this layer on purpose. It selects which implementation
# AITER would run natively, which is a question the dispatcher answers before
# the op begins; a provider asked to serve this contract is not offered it and
# so cannot be asked to honour a pin it has no kernel for.
@_route(op="mla_sparse_prefill", version=1)
def _mla_sparse_prefill_routed(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    num_kv_splits: int | None = None,
) -> torch.Tensor:
    """MLA sparse prefill over bf16/fp16 Q/KV (D = 512).

    Shapes and meanings are those of :func:`mla_sparse_prefill`, minus its
    ``backend`` argument. ``out`` is written in place when given and returned
    either way, so a provider that allocates must still return what it wrote.
    """
    return _opus_sparse_prefill(
        q,
        unified_kv,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        softmax_scale,
        out,
        num_kv_splits,
    )


@torch_compile_guard(mutates_args=["out"], gen_fake=_mla_sparse_prefill_fake)
def mla_sparse_prefill(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    num_kv_splits: int | None = None,
    backend: str | None = None,
) -> torch.Tensor:
    """MLA sparse prefill over bf16 Q/KV (D = 512).

    Args:
      q:                 ``[T, H, D]`` bf16 query.
      unified_kv:        ``[total_pages, D]`` prefix source, same dtype as ``q``.
      kv_indices_prefix: ``[total_prefix]`` int32 row indices into
        ``unified_kv``, concatenated per token.
      kv_indptr_prefix:  ``[T+1]`` int32 CSR row pointers.
      kv:                ``[total_tokens, D]`` extend source, same dtype and row
        stride as ``unified_kv``.
      kv_indices_extend: ``[total_extend]`` int32 row indices into ``kv``,
        concatenated per token.
      kv_indptr_extend:  ``[T+1]`` int32 CSR row pointers.
      attn_sink:         ``[H]`` fp32 per-head softmax-denom bias, or ``None``
        for no sink.
      softmax_scale:     float scalar applied to the QK^T scores.
      out:               Optional ``[T, H, D]`` output (``q``'s dtype);
        allocated if ``None``.
      num_kv_splits:     split each token's keys over this many workgroups and
        fold them in a second launch: 1 to 32. ``None`` picks it from the
        token, head and key counts.
      backend:           ``"opus"`` or ``None``.

    Returns:
      ``out``.
    """
    _check_backend("mla_sparse_prefill", backend, ("opus",))
    return _mla_sparse_prefill_routed(
        q,
        unified_kv,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        softmax_scale,
        out,
        num_kv_splits,
    )


def _mla_sparse_prefill_v4_fake(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    unified_kv_nope: torch.Tensor,
    unified_kv_rope: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    num_kv_splits: int | None = None,
    backend: str | None = None,
) -> torch.Tensor:
    if out is not None:
        return out
    t, h, _ = q_nope.shape
    return torch.empty((t, h, 512), dtype=torch.bfloat16, device=q_nope.device)


@torch_compile_guard(mutates_args=["out"], gen_fake=_mla_sparse_prefill_v4_fake)
def mla_sparse_prefill_v4(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    unified_kv_nope: torch.Tensor,
    unified_kv_rope: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    num_kv_splits: int | None = None,
    backend: str | None = None,
) -> torch.Tensor:
    """MLA sparse prefill over the v4 layout: NoPE rows holding their own E8M0
    scales, plus bf16 RoPE. Supported NoPE dtypes: fp8.

    Args:
      q_nope:            ``[T, H, 512]`` NoPE (fp8: 448 values, 14 E8M0 scales, pad).
      q_rope:            ``[T, H, 64]`` bf16.
      unified_kv_nope:   ``[total_pages, 512]`` prefix NoPE source, as ``q_nope``.
      unified_kv_rope:   ``[total_pages, 64]`` bf16 prefix RoPE source.
      kv_indices_prefix: ``[total_prefix]`` int32 row indices into the prefix
        sources, concatenated per token.
      kv_indptr_prefix:  ``[T+1]`` int32 CSR row pointers.
      kv_nope:           ``[total_tokens, 512]`` extend NoPE source, as ``q_nope``.
      kv_rope:           ``[total_tokens, 64]`` bf16 extend RoPE source.
      kv_indices_extend: ``[total_extend]`` int32 row indices into the extend
        sources, concatenated per token.
      kv_indptr_extend:  ``[T+1]`` int32 CSR row pointers.
      attn_sink:         ``[H]`` fp32 per-head softmax-denom bias, or ``None``
        for no sink (opus; ``None`` makes the default backend opus).
      softmax_scale:     float scalar applied to the QK^T scores.
      out:               Optional ``[T, H, 512]`` bf16 output; allocated if ``None``.
      num_kv_splits:     as in :func:`mla_sparse_prefill` (opus only).
      backend:           ``"asm"``, ``"opus"`` or ``None``.

    Returns:
      ``out``.
    """
    _check_backend("mla_sparse_prefill_v4", backend, ("asm", "opus"))
    args = (
        q_nope,
        q_rope,
        unified_kv_nope,
        unified_kv_rope,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv_nope,
        kv_rope,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        softmax_scale,
        out,
    )
    unsplit = num_kv_splits in (None, 1)
    if backend == "asm":
        if not unsplit:
            raise RuntimeError("the asm backend has no split-K")
        return _asm_sparse_prefill_v4(*args)
    if (
        backend is None
        and unsplit
        and attn_sink is not None
        and get_gfx_runtime() == "gfx1250"
        and q_nope.size(1) == _ASM_HEADS
    ):
        try:
            return _asm_sparse_prefill_v4(*args)
        except RuntimeError as e:  # asm is stricter about dtypes and shapes than opus
            _warn_asm_fallback(str(e))
    return _opus_sparse_prefill_v4(*args, num_kv_splits)


def _mla_sparse_prefill_v41_fake(
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
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    num_kv_splits: int | None = None,
    backend: str | None = None,
) -> torch.Tensor:
    if out is not None:
        return out
    t, h, _ = q.shape
    return torch.empty((t, h, 512), dtype=torch.bfloat16, device=q.device)


# Routed below the backend pin, as `mla_sparse_prefill` is. `backend` picks
# which implementation AITER runs -- a question the dispatcher settles before
# the op begins. A provider has its own kernel, not a choice of AITER's, so a
# contract carrying the pin would have it decline every call that passes one.
#
# Version 2: v1 was this signature WITH `backend` in it. A v1 provider was
# built for fifteen arguments and a v2 call offers fourteen, so v1 is not
# superseded -- the arity check rejecting it is the correct outcome.
@_route(op="mla_sparse_prefill_v41", version=2)
def _mla_sparse_prefill_v41_routed(
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
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    num_kv_splits: int | None = None,
) -> torch.Tensor:
    """MLA v4.1 sparse prefill, below the backend pin.

    Shapes and meanings are those of :func:`mla_sparse_prefill_v41`, minus its
    ``backend`` argument. ``out`` is written when given and returned either
    way, so a provider returns what it wrote.
    """
    return _opus_sparse_prefill_v41(
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
        num_kv_splits,
    )


@torch_compile_guard(mutates_args=["out"], gen_fake=_mla_sparse_prefill_v41_fake)
def mla_sparse_prefill_v41(
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
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    num_kv_splits: int | None = None,
    backend: str | None = None,
) -> torch.Tensor:
    """MLA v4.1 sparse prefill: rows of 512 values with separate E8M0 scales (uint8
    or fp8_e8m0, one per 32 values). Same argument order as
    :func:`mla_sparse_prefill_v4`, with the RoPE tensors replaced by the scales.
    Supported (q, unified_kv, kv) dtypes: (fp8, fp8, fp8) on gfx950 and gfx1250, and
    (fp8, fp4x2, fp8) on gfx950 (opus) and gfx1250 H64 (asm), with an NVFP4
    prefix source: e2m1 pairs (fp4x2 or uint8, the even value in the low nibble)
    with an e4m3 scale (fp8 or uint8) per 16 values. The asm backend requires
    contiguous, separate data and scale arrays; it does not unpack interleaved
    288-byte CSA rows. Its fixed sink path requires softmax_scale to have the
    same FP32 value as 1/sqrt(512). CSR indices define all visibility, including
    causal masks.

    Args:
      q:                 ``[T, H, 512]`` query.
      q_scale:           ``[T, H, 16]`` scales of ``q``.
      unified_kv:        ``[total_pages, 512]`` prefix KV source, or
        ``[total_pages, 256]`` if NVFP4.
      unified_kv_scale:  ``[total_pages, 16]`` scales of ``unified_kv``, or
        ``[total_pages, 32]`` if NVFP4.
      kv_indices_prefix: ``[total_prefix]`` int32 row indices into the prefix
        source, concatenated per token.
      kv_indptr_prefix:  ``[T+1]`` int32 CSR row pointers.
      kv:                ``[total_tokens, 512]`` extend KV source.
      kv_scale:          ``[total_tokens, 16]`` scales of ``kv``.
      kv_indices_extend: ``[total_extend]`` int32 row indices into the extend
        source, concatenated per token.
      kv_indptr_extend:  ``[T+1]`` int32 CSR row pointers.
      attn_sink:         ``[H]`` fp32 per-head softmax-denom bias, or ``None``
        for no sink.
      softmax_scale:     float scalar applied to the QK^T scores.
      out:               Optional ``[T, H, 512]`` bf16 output buffer; allocated
        if ``None``.
      num_kv_splits:     opus: as in :func:`mla_sparse_prefill`; on gfx1250,
        2, 4, 8 or 16 fold in the same launch. asm: an integer in [1, 16].
        With ``None``, asm uses 1 split for N > 64. For N <= 64, an average of
        512 prefix rows uses 1 split; 1024 prefix rows use 4 splits at N <= 32
        and 2 otherwise. Other prefix lengths use the ASM fallback heuristic.
        Extend rows do not count toward the prefix average. asm fuses reduction
        in the same launch and allocates workspace internally.
      backend:           ``"asm"``, ``"opus"`` or ``None``. Default selects asm
        for a supported gfx1250 H64 NVFP4 prefix, otherwise opus. Use
        :func:`get_mla_sparse_prefill_v41_backend` before quantizing Q to check
        native eligibility. Explicit backends retain their validation/errors.
        Enabled plugins are tried first and do not receive this native pin.

    Returns:
      ``out`` (``[T, H, 512]`` bf16).
    """
    _check_backend("mla_sparse_prefill_v41", backend, ("asm", "opus"))

    def native_fallback(*native_args, **native_kwargs):
        selected = backend
        if selected is None:
            selected = get_mla_sparse_prefill_v41_backend(
                q.device,
                tuple(q.shape),
                unified_kv_dtype=unified_kv.dtype,
                softmax_scale=softmax_scale,
                q_dtype=q.dtype,
                q_scale_dtype=q_scale.dtype,
                unified_kv_scale_dtype=unified_kv_scale.dtype,
                kv_dtype=kv.dtype,
                kv_scale_dtype=kv_scale.dtype,
                layout=_v41_tensor_layout(
                    q, q_scale, unified_kv, unified_kv_scale, kv, kv_scale
                ),
            )
            if selected is None:
                raise RuntimeError(
                    "mla_sparse_prefill_v41: no native backend supports the "
                    "requested device, shape, quantized dtypes, layout and scale"
                )
        implementation = (
            _asm_sparse_prefill_v41 if selected == "asm" else _opus_sparse_prefill_v41
        )
        return implementation(*native_args, **native_kwargs)

    args = (
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
        num_kv_splits,
    )
    operation = _mla_sparse_prefill_v41_routed.__aiter_plugin_op__
    registered = _plugin._routed_ops[operation.name]
    return _plugin._dispatch(
        operation, registered.canonicalize, native_fallback, args, {}
    )


__all__ = [
    "get_mla_sparse_prefill_v41_backend",
    "mla_sparse_prefill",
    "mla_sparse_prefill_fp8_asm",
    "mla_sparse_prefill_fp8_asm_fwd",
    "mla_sparse_prefill_v4",
    "mla_sparse_prefill_v41",
]
