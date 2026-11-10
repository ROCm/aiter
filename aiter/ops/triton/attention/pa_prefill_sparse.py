# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sparse paged-prefill attention over two KV sources (prefix + extend) with
per-head sink bias — gfx1250 (gluon) only.

Exposes ``pa_prefill_sparse`` — grid ``(T, cdiv(H, BLOCK_H))``, one token
and BLOCK_H heads per CTA. Same grid as the decode kernel. No split-K:
prefill fills the GPU via the token dimension.

Gfx950 (the single-pass gluon kernel, or `mla_gluon`) and others (Triton
`_sparse_attn_prefill_kernel`) take a 1-D ``(kv_indices, kv_indptr)`` pair
over one pool.

``pa_prefill_sparse`` — a single entry that dispatches on arch:

    gfx1250 -> gluon ``_pa_prefill_sparse``           (two sources, native)
    gfx950  -> gluon ``_pa_prefill_sparse``           (single source, H=32 or 64, D=512, bf16)
               else gluon wrapper ``mla_gluon``       (single source)
    else    -> triton ``_sparse_attn_prefill_kernel`` (single source)
"""

import functools

import numpy as np
import torch
import triton

from aiter.ops.triton._gluon_kernels.gfx950.attention.pa_prefill_sparse import (
    _pa_prefill_sparse as _pa_prefill_sparse_gfx950,
)
from aiter.ops.triton._gluon_kernels.gfx1250.attention.pa_prefill_sparse import (
    _pa_prefill_sparse as gluon_pa_prefill_sparse,
)
from aiter.ops.triton._triton_kernels.attention.sparse_attention_dsv4 import (
    _sparse_attn_prefill_kernel,
)
from aiter.ops.triton.gluon.mla_gluon import (
    mla_gluon as gluon_mla_sparse_prefill,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir
from aiter.ops.triton.utils.device_info import get_num_xcds
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.tuned_config_utils import get_tuned_kernel_config

DEVICE_ARCH = arch_info.get_arch()

_LOGGER = AiterTritonLogger()


# Returned by the lookup where nothing is published for the device and shape.
_NO_PINNED_CONFIG = triton.Config({})


@functools.lru_cache
def _pinned_prefill_config(num_heads: int, head_dim: int) -> triton.Config | None:
    """Published single-source prefill tile for this device and shape, if any.

    Entries are keyed ``_sparse_attn_prefill_kernel_H{num_heads}_D{head_dim}``
    in the arch's ``sparse_attention_dsv4/DEFAULT.json``. Devices and shapes
    without an entry return None, quietly, and use the autotuned launch.
    """
    cfg = get_tuned_kernel_config(
        "attention",
        "SPARSE_ATTENTION_DSV4",
        f"_sparse_attn_prefill_kernel_H{num_heads}_D{head_dim}",
        fallback=_NO_PINNED_CONFIG,
        warn_on_miss=False,
    )
    return None if cfg is _NO_PINNED_CONFIG else cfg


def pa_prefill_sparse(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor | None,
    kv_indices_extend: torch.Tensor | None,
    kv_indptr_extend: torch.Tensor | None,
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    has_invalid: bool | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sparse prefill attention over two KV sources with sink.

    Dispatches on arch: gfx1250 -> gluon (two sources), gfx950 -> a single-pass
    gluon kernel for bf16 q and KV with H=32 or 64 and D=512, else `mla_gluon`;
    otherwise the Triton kernel. Only gfx1250 reads a second KV source; the
    other two serve the prefix source alone and reject any extend source.

    Meant for prefill-sized calls. The grid is one program per query token and
    block of heads with no split over the KV slots, so a call with only a few
    query rows (a decode step) leaves most of the GPU idle; use a split-KV
    decode kernel for those.

    Args:
        q:                 [T, H, D] BF16/FP16 — queries.
        unified_kv:        [total_pages, D] — prefix KV source (paged).
        kv_indices_prefix: [total_prefix] int32 — flat per-token slot lists
            into unified_kv. The Triton branch also takes int64 as is; gfx950
            casts other dtypes to int32, so slots there must be below 2^31. Invalid-slot handling depends on the branch; see
            ``has_invalid``.
        kv_indptr_prefix:  [T+1] int32 — true prefix sum.
        kv:                [total_tokens, D] — extend KV source (this fwd's
            input K, not yet in paged buffer). ``None`` for no extend source.
        kv_indices_extend: [total_extend] int32 — flat per-token row idx lists
            into kv. ``-1`` sentinels skipped. ``None`` for no extend source.
        kv_indptr_extend:  [T+1] int32 — true prefix sum. ``None`` for none.
        attn_sink:         [H] fp32 — per-head softmax-denom bias.
        softmax_scale:     float.
        has_invalid:       gfx1250 only: whether index lists may hold ``-1``
            sentinels (``None`` picks a heuristic). The Triton fallback always
            skips ``-1`` and out-of-pool slots. gfx950 does not check slot
            values: pass only valid slots there. ``True`` keeps gfx950 on
            ``mla_gluon``.
        out:               optional contiguous [T, H, D] buffer on q's device,
            same dtype as q; written in place and returned.

    Returns:
        [T, H, D] attention output, same dtype as q.
    """
    if out is None:
        out = torch.empty_like(q)
    else:
        assert (
            out.shape == q.shape and out.dtype == q.dtype
        ), f"out {tuple(out.shape)} {out.dtype} != q {tuple(q.shape)} {q.dtype}"
        assert out.device == q.device, f"out on {out.device}, q on {q.device}"
        # Every branch writes [T, H, D] in place: overlapping views (e.g. an
        # expand() over heads) would race, and gfx950 ignores out.stride(2).
        assert (
            out.is_contiguous()
        ), f"out must be contiguous, got strides {out.stride()}"
    if DEVICE_ARCH in ("gfx1250", "gfx1250-strict"):
        if not q.is_cuda:
            raise RuntimeError("pa_prefill_sparse requires CUDA/HIP tensors")
        if q.dtype not in (torch.bfloat16, torch.float16):
            raise RuntimeError(f"pa_prefill_sparse expects fp16/bf16 q, got {q.dtype}")
        if unified_kv.dtype != q.dtype:
            raise RuntimeError(
                f"unified_kv dtype mismatch: kv={unified_kv.dtype}, q={q.dtype}"
            )
        if kv.dtype != q.dtype:
            raise RuntimeError(f"kv dtype mismatch: kv={kv.dtype}, q={q.dtype}")

        T, H, D = q.shape
        if has_invalid is None:
            avg_prefix_len = kv_indices_prefix.numel() / max(T, 1)
            has_invalid = not (0 < avg_prefix_len <= 16)
        _LOGGER.info(
            "PA_PREFILL_SPARSE T=%d H=%d D=%d prefix_indices=%d extend_indices=%d",
            T,
            H,
            D,
            kv_indices_prefix.shape[0],
            kv_indices_extend.shape[0],
        )

        assert (
            kv_indices_prefix.dtype == torch.int32 and kv_indices_prefix.is_contiguous()
        )
        assert (
            kv_indptr_prefix.dtype == torch.int32 and kv_indptr_prefix.is_contiguous()
        )
        assert (
            kv_indices_extend.dtype == torch.int32 and kv_indices_extend.is_contiguous()
        )
        assert (
            kv_indptr_extend.dtype == torch.int32 and kv_indptr_extend.is_contiguous()
        )

        total_prefix_pages = unified_kv.shape[0]
        total_extend_tokens = kv.shape[0]
        USE_EXP2 = True
        block_d = triton.next_power_of_2(D)
        assert block_d == D

        if H >= 64:
            block_h = 64
            block_k = 32
            num_warps = 4
            waves_per_eu = 1
        elif H >= 32:
            block_h = 32
            block_k = 32
            num_warps = 2
            waves_per_eu = 1
        else:
            block_h = max(triton.next_power_of_2(min(H, 16)), 16)
            block_k = 16
            num_warps = 1
            waves_per_eu = 1
        grid = (T, triton.cdiv(H, block_h))

        gluon_pa_prefill_sparse[grid](
            q,
            unified_kv,
            kv_indices_prefix,
            kv_indptr_prefix,
            kv,
            kv_indices_extend,
            kv_indptr_extend,
            attn_sink,
            out,
            total_prefix_pages,
            total_extend_tokens,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            unified_kv.stride(0),
            unified_kv.stride(1),
            kv.stride(0),
            kv.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            H,
            D,
            float(softmax_scale),
            BLOCK_H=block_h,
            BLOCK_D=block_d,
            BLOCK_K=block_k,
            HAS_INVALID=has_invalid,
            USE_EXP2=USE_EXP2,
            num_warps=num_warps,
            waves_per_eu=waves_per_eu,
        )
        return out

    elif DEVICE_ARCH == "gfx950":
        kv_indices_prefix, kv_indptr_prefix = _prep_single_source(
            kv_indices_prefix,
            kv_indptr_prefix,
            kv,
            kv_indices_extend,
            kv_indptr_extend,
        )
        # The single-pass kernel takes every call it serves, at any T:
        # `mla_gluon` adds the sink once per key split, leaves rows without
        # keys unwritten when it runs one split, and at H=64 asserts unless T
        # is a multiple of 64 and gets rows with few keys wrong.
        if _single_pass_unsupported(q, unified_kv, has_invalid) is None:
            _single_pass_prefill(
                q,
                unified_kv,
                kv_indices_prefix,
                kv_indptr_prefix,
                attn_sink,
                softmax_scale,
                out,
            )
            return out
        gluon_mla_sparse_prefill(
            q,  # q_nope = combined-D query (RoPE folded in)
            None,  # q_pe unused in prefill mode
            unified_kv,  # kv_c
            out,  # o (written in place)
            kv_indices_prefix,  # page_table = ragged kv_indices
            kv_indptr_prefix,  # seq_info = ragged kv_indptr
            float(softmax_scale),
            min_kv_seq_len=float("inf"),  # skip min_kv_seq_len check
            has_pe=False,
            attn_sink=attn_sink.contiguous() if attn_sink is not None else None,
        )
        return out

    else:
        # Portable Triton fallback.
        # Same up-front checks as gfx1250: an fp8 KV pool would otherwise fail
        # inside tl.dot at compile time with an opaque CompilationError.
        if q.dtype not in (torch.bfloat16, torch.float16):
            raise RuntimeError(f"pa_prefill_sparse expects fp16/bf16 q, got {q.dtype}")
        if unified_kv.dtype != q.dtype:
            raise RuntimeError(
                f"unified_kv dtype mismatch: kv={unified_kv.dtype}, q={q.dtype}"
            )
        # The Triton kernel takes int32 or int64 indices as passed; converting
        # int64 to int32 would silently wrap slots past 2^31 - 1.
        kv_indices_prefix, kv_indptr_prefix = _prep_single_source(
            kv_indices_prefix,
            kv_indptr_prefix,
            kv,
            kv_indices_extend,
            kv_indptr_extend,
            index_dtypes=(torch.int32, torch.int64),
        )
        has_attn_sink = attn_sink is not None
        if has_attn_sink:
            attn_sink = attn_sink.contiguous()
        else:
            attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)
        num_queries, num_heads, head_dim = q.shape
        block_d = triton.next_power_of_2(head_dim)
        args = (
            q,
            unified_kv,
            kv_indices_prefix,
            kv_indptr_prefix,
            attn_sink,
            out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            unified_kv.stride(0),
            unified_kv.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            num_heads,
            head_dim,
            unified_kv.shape[0],
            float(softmax_scale),
        )

        pinned = _pinned_prefill_config(num_heads, head_dim)
        if pinned is not None:
            # Published per-shape tile (configs/<arch>/.../sparse_attention_dsv4).
            block_h = pinned.kwargs["BLOCK_H"]
            _sparse_attn_prefill_kernel.fn[
                (num_queries, triton.cdiv(num_heads, block_h))
            ](
                *args,
                HAS_ATTN_SINK=has_attn_sink,
                BLOCK_D=block_d,
                # Drop the head/dim masks only; invalid KV rows are still
                # masked out of the gather (never loaded), so an empty pool
                # or a non-finite unused row is safe.
                EVEN_H=num_heads % block_h == 0,
                EVEN_D=block_d == head_dim,
                num_warps=pinned.num_warps,
                num_stages=pinned.num_stages,
                **pinned.kwargs,
            )
            return out

        grid = lambda META: (
            num_queries,
            triton.cdiv(num_heads, META["BLOCK_H"]),
        )
        _sparse_attn_prefill_kernel[grid](
            *args,
            HAS_ATTN_SINK=has_attn_sink,
            BLOCK_D=block_d,
        )
        return out


# ---------------------------------------------------------------------------
# gfx950 single-pass kernel
# ---------------------------------------------------------------------------

_LOG2E = 1.44269504089


@functools.lru_cache
def _single_pass_config() -> dict:
    """Launch config of the gfx950 single-pass kernel.

    Returns the cached dict: do not change it.
    """
    cfg_dir = resolve_config_dir("attention", "PA_PREFILL_SPARSE", backend="gluon")
    return load_config_json(f"{cfg_dir}/DEFAULT.json")["_pa_prefill_sparse"]


# The kernel computes 32 heads per program, so at H=64 two programs share each
# query row, one per half of the heads, in one launch. The kernel places a
# row's two programs on one XCD, so the second one's KV gathers can hit the
# first one's in L2: on random slots over a 1 GiB pool, half as many KV reads
# miss L2 as with one launch per half. One 64-head program measured slower.
_GROUP_HEADS = 32
_SINGLE_PASS_HEADS = (32, 64)


def _single_pass_unsupported(
    q: torch.Tensor, unified_kv: torch.Tensor, has_invalid: bool | None
) -> str | None:
    """Why the gfx950 single-pass kernel cannot serve the call, or None."""
    _, H, D = q.shape
    if H not in _SINGLE_PASS_HEADS or D != 512:
        return f"needs H in {_SINGLE_PASS_HEADS} and D=512, got H={H} D={D}"
    if q.dtype != torch.bfloat16 or unified_kv.dtype != torch.bfloat16:
        return f"needs bf16 q and unified_kv, got {q.dtype} and {unified_kv.dtype}"
    if unified_kv.dim() != 2 or unified_kv.shape[1] != D:
        return f"needs a [total_pages, {D}] unified_kv, got {tuple(unified_kv.shape)}"
    if q.stride(2) != 1 or unified_kv.stride(1) != 1:
        return (
            f"needs unit inner strides, got q {q.stride(2)} "
            f"and unified_kv {unified_kv.stride(1)}"
        )
    if has_invalid is True:
        return "reads valid slots only: has_invalid must not be True"
    return None


@functools.lru_cache(maxsize=64)
def _qk_scale(softmax_scale: float) -> float:
    """fp32(softmax_scale) * fp32(log2(e)), rounded once in fp32.

    A product in [0, the smallest normal fp32) becomes that value: the kernel
    masks the scores past a row's last key to -inf before scaling them, and
    -inf * 0 is NaN. Masked scores stay -inf, and for scores below 1e8 every
    other exponent s * qk_scale - m is within 1e-30 of 0: each key weighs 1, as
    at 0.
    """
    tiny = float(np.finfo(np.float32).tiny)
    qk_scale = float(np.float32(softmax_scale) * np.float32(_LOG2E))
    return tiny if 0.0 <= qk_scale < tiny else qk_scale


def _single_pass_prefill(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    out: torch.Tensor,
) -> None:
    """Launch the gfx950 single-pass kernel once: a program per query row and
    _GROUP_HEADS heads."""
    T, H, _ = q.shape
    groups = H // _GROUP_HEADS
    cfg = _single_pass_config()
    _LOGGER.info("PA_PREFILL_SPARSE gfx950 single-pass T=%d H=%d", T, H)
    if attn_sink is not None:
        attn_sink = attn_sink.to(torch.float32).contiguous()
    _pa_prefill_sparse_gfx950[(T * groups,)](
        q,
        unified_kv,
        kv_indices,
        kv_indptr,
        attn_sink,
        out,
        q.stride(0),
        q.stride(1),
        unified_kv.stride(0),
        out.stride(0),
        out.stride(1),
        _qk_scale(float(softmax_scale)),
        H=_GROUP_HEADS,
        HEAD_GROUPS=groups,
        # one compile for every device at H=32, where no row is shared
        NUM_XCDS=get_num_xcds() if groups > 1 else 1,
        BLOCK_K=cfg["BLOCK_K"],
        HAS_SINK=attn_sink is not None,
        num_warps=cfg["num_warps"],
    )


# ---------------------------------------------------------------------------
# Inputs preparation
# ---------------------------------------------------------------------------


def _prep_single_source(
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor | None,
    kv_indices_extend: torch.Tensor | None,
    kv_indptr_extend: torch.Tensor | None,
    index_dtypes: tuple[torch.dtype, ...] = (torch.int32,),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize the KV pool and indices for the gfx950 / Triton kernels.

    Indices and indptr keep their dtype if it is in ``index_dtypes`` and are
    converted to int32 otherwise.

    Rejects an extend KV source outright: only the gfx1250 gluon kernel reads a
    second pool.
    """
    extend = (
        ("kv", kv),
        ("kv_indices_extend", kv_indices_extend),
        ("kv_indptr_extend", kv_indptr_extend),
    )
    provided = [name for name, tensor in extend if tensor is not None]
    if provided:
        raise NotImplementedError(
            f"pa_prefill_sparse got an extend KV source ({', '.join(provided)}), "
            f"but {DEVICE_ARCH} is served by a single-source kernel that reads "
            f"only unified_kv. Two KV sources are supported on gfx1250 only; "
            f"merge them into unified_kv, or pass None for the extend trio."
        )

    return (
        _as_index_contiguous_1d(kv_indices_prefix, index_dtypes),
        _as_index_contiguous_1d(kv_indptr_prefix, index_dtypes),
    )


def _as_index_contiguous_1d(
    x: torch.Tensor, index_dtypes: tuple[torch.dtype, ...]
) -> torch.Tensor:
    if x.dtype not in index_dtypes:
        return _as_int32_contiguous_1d(x)
    return x.reshape(-1).contiguous()


def _as_int32_contiguous_1d(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.int32 and x.ndim == 1 and x.is_contiguous():
        return x
    return x.to(torch.int32).contiguous()
