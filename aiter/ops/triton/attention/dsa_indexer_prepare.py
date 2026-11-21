# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused DSA (DeepSeek Sparse Attention) indexer prepare for decode, with
block-FP8 indexer weights.

Two launches replace the decode chain a DSA indexer runs before its top-k
logits: three projections (``wq_b``, ``wk`` as block-FP8 GEMMs with the 1x128
activation quant, ``weights_proj`` in bf16) and the
``indexer_qk_rope_quant_and_cache`` epilogue -- K LayerNorm, RoPE, Hadamard(128),
ue8m0 FP8 quant, the preshuffled index-K cache write and the head gate weights.
In SGLang that is five to six kernels (two per_1x128 quants, two block-FP8
GEMMs, one bf16 GEMM, the fused writer); here the projections are one launch
of single-wave split-K tiles and the epilogue a second one.

Why a new op: no aiter op owns the projections together with the epilogue.
``indexer_qk_rope_quant_and_cache`` (``aiter/ops/cache.py``) takes the
projected q/k/gate and omits the Hadamard rotation; this op produces the
rotated index-K the stock gfx950 SGLang decode path reads back.

Support is narrow and is reported by :func:`dsa_indexer_prepare_supported`
rather than asserted, so callers fall back to the unfused chain. gfx950 only:
the kernels use ``mfma_scaled`` e4m3 MFMAs, CDNA4 buffer loads and explicit
register layouts, in the Gluon dialect of Triton 3.8.
"""

import functools
import re

import torch
import triton

from aiter.ops.triton.utils.config_utils import (
    load_config_json,
    resolve_config_dir,
    select_leq_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

HEAD_DIM = 128
ROPE_DIM = 64
# Index-K cache page: 64 tokens of 128 e4m3 bytes in 16x16 tiles, then 64 fp32 scales.
PAGE_SIZE = 64
CACHE_STRIDE = HEAD_DIM + 4
# SGLang's convention for a padded CUDA-graph row; the K write is skipped.
PAD_SLOT_ID = -1
# Decode rows the single-wave projection tiles are tuned for (see DEFAULT.json).
MAX_TOKENS = 32


def _arch_supported() -> tuple[bool, str]:
    """gfx950 probe, guarded so architecture detection never raises at import."""
    try:
        from aiter.ops.triton.utils._triton.arch_info import get_arch

        arch = get_arch()
    except Exception as exc:  # noqa: BLE001
        return False, f"architecture detection failed ({exc})"
    if arch != "gfx950":
        return False, f"gfx950 only, got {arch}"
    return True, ""


@functools.lru_cache(maxsize=1)
def _gluon_supported() -> tuple[bool, str]:
    """Cached probe: can this Triton compile the kernels' Gluon dialect?

    The kernels are written against the Gluon dialect of Triton **3.8**
    (``gl.amd.cdna4.mfma_scaled``, ``buffer_load``); ROCm backported an
    earlier, incompatible Gluon into some 3.7 builds where the import works
    but the layout ops fail, so the gate is a version check, not an import
    probe. Mirrors ``fused_gdn_decode_qkvz._gluon_supported``.
    """
    version = triton.__version__
    matched = re.match(r"(\d+)\.(\d+)", version or "")
    if matched is None or (int(matched.group(1)), int(matched.group(2))) < (3, 8):
        return False, f"Triton >= 3.8 required for this Gluon dialect, got {version}"
    try:
        from triton.experimental import gluon  # noqa: F401
    except ImportError:
        return False, f"triton.experimental.gluon unavailable (Triton {version})"
    return True, ""


@functools.cache
def _configs() -> dict:
    cfg_dir = resolve_config_dir("attention", "DSA_INDEXER_PREPARE", backend="gluon")
    return load_config_json(f"{cfg_dir}/DEFAULT.json")


def _full_config(tokens: int) -> dict:
    return select_leq_config(_configs()["full"], tokens, prefix="M_LEQ_")


def _key_only_config(tokens: int) -> dict:
    return select_leq_config(_configs()["key_only"], tokens, prefix="M_LEQ_")


def _is_shuffled_fp8(w: torch.Tensor, n: int, k: int) -> bool:
    return w.dtype is torch.float8_e4m3fn and w.shape == (n, k) and w.is_contiguous()


def dsa_indexer_prepare_supported(
    x: torch.Tensor,
    q_lora: torch.Tensor | None,
    wq_b: torch.Tensor | None,
    wq_b_scale: torch.Tensor | None,
    wk: torch.Tensor,
    wk_scale: torch.Tensor,
    weights_proj: torch.Tensor | None,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    key_only: bool = False,
) -> tuple[bool, str]:
    """Report whether this call is covered, and if not, why.

    Returns ``(True, "")`` or ``(False, reason)``. The reason is meant to be
    logged once by the caller on its fallback path.
    """
    ok, reason = _arch_supported()
    if not ok:
        return False, reason
    ok, reason = _gluon_supported()
    if not ok:
        return False, reason

    if x.ndim != 2 or x.dtype is not torch.bfloat16 or x.stride(1) != 1:
        return (
            False,
            f"x must be bf16 [T, K] with dense rows, got {x.dtype} {tuple(x.shape)}",
        )
    tokens, k = x.shape
    if not 0 < tokens <= MAX_TOKENS:
        return False, f"decode rows 1..{MAX_TOKENS} only, got {tokens}"
    if k % 512:
        return False, f"hidden size must be a multiple of 512, got {k}"
    if wk.shape != (HEAD_DIM, k) or not _is_shuffled_fp8(wk, HEAD_DIM, k):
        return (
            False,
            f"wk must be fp8 e4m3fn (16,16)-shuffled [{HEAD_DIM}, {k}], got {wk.dtype} {tuple(wk.shape)}",
        )
    if (
        wk_scale.shape != (1, k // 128)
        or wk_scale.dtype is not torch.float32
        or not wk_scale.is_contiguous()
    ):
        return (
            False,
            f"wk_scale must be fp32 [1, {k // 128}], got {wk_scale.dtype} {tuple(wk_scale.shape)}",
        )
    if not key_only:
        if q_lora is None or wq_b is None or wq_b_scale is None or weights_proj is None:
            return (
                False,
                "q_lora, wq_b, wq_b_scale and weights_proj are required unless key_only",
            )
        if (
            q_lora.ndim != 2
            or q_lora.shape[0] != tokens
            or q_lora.dtype is not torch.bfloat16
            or q_lora.stride(1) != 1
        ):
            return (
                False,
                f"q_lora must be bf16 [{tokens}, QK] with dense rows, got {q_lora.dtype} {tuple(q_lora.shape)}",
            )
        qk = q_lora.shape[1]
        if qk % 512:
            return False, f"q_lora_rank must be a multiple of 512, got {qk}"
        heads = weights_proj.shape[0]
        if heads % 32 or heads > 64:
            return False, f"index heads must be 32 or 64, got {heads}"
        if not _is_shuffled_fp8(wq_b, heads * HEAD_DIM, qk):
            return False, (
                f"wq_b must be fp8 e4m3fn (16,16)-shuffled [{heads * HEAD_DIM}, {qk}], "
                f"got {wq_b.dtype} {tuple(wq_b.shape)}"
            )
        if (
            wq_b_scale.shape != (heads, qk // 128)
            or wq_b_scale.dtype is not torch.float32
            or not wq_b_scale.is_contiguous()
        ):
            return (
                False,
                f"wq_b_scale must be fp32 [{heads}, {qk // 128}], got {wq_b_scale.dtype} {tuple(wq_b_scale.shape)}",
            )
        if (
            weights_proj.shape != (heads, k)
            or weights_proj.dtype is not torch.bfloat16
            or weights_proj.stride(1) != 1
        ):
            return (
                False,
                f"weights_proj must be bf16 [{heads}, {k}], got {weights_proj.dtype} {tuple(weights_proj.shape)}",
            )
    for name, t in (("norm_weight", norm_weight), ("norm_bias", norm_bias)):
        if (
            t.shape != (HEAD_DIM,)
            or t.dtype not in (torch.bfloat16, torch.float32)
            or not t.is_contiguous()
        ):
            return (
                False,
                f"{name} must be a contiguous bf16/fp32 [{HEAD_DIM}], got {t.dtype} {tuple(t.shape)}",
            )
    for name, t in (("cos_cache", cos_cache), ("sin_cache", sin_cache)):
        if (
            t.ndim != 2
            or t.shape[1] != ROPE_DIM // 2
            or t.dtype is not torch.bfloat16
            or t.stride(1) != 1
        ):
            return (
                False,
                f"{name} must be bf16 [max_pos, {ROPE_DIM // 2}] with dense rows, got {t.dtype} {tuple(t.shape)}",
            )
    for name, t in (("positions", positions), ("slot_mapping", slot_mapping)):
        if t.shape != (tokens,) or t.dtype is not torch.int64 or not t.is_contiguous():
            return (
                False,
                f"{name} must be a contiguous int64 [{tokens}], got {t.dtype} {tuple(t.shape)}",
            )
    if (
        kv_cache.ndim != 3
        or kv_cache.shape[1:] != (PAGE_SIZE, CACHE_STRIDE)
        or kv_cache.element_size() != 1
    ):
        return False, (
            f"kv_cache must be a 1-byte [pages, {PAGE_SIZE}, {CACHE_STRIDE}] index-K cache, "
            f"got {kv_cache.dtype} {tuple(kv_cache.shape)}"
        )
    if not kv_cache.is_contiguous():
        return False, "kv_cache must be contiguous"
    device = x.device
    tensors = [
        x,
        wk,
        wk_scale,
        norm_weight,
        norm_bias,
        positions,
        cos_cache,
        sin_cache,
        slot_mapping,
        kv_cache,
    ]
    if not key_only:
        tensors += [q_lora, wq_b, wq_b_scale, weights_proj]
    if not all(t.is_cuda and t.device == device for t in tensors):
        return False, "every operand must be a CUDA tensor on one device"
    return True, ""


def _fits_32bit(*tensors: torch.Tensor) -> bool:
    return all(t.stride(0) * t.shape[0] * t.element_size() < 1 << 31 for t in tensors)


def _p2(n: int) -> int:
    return triton.next_power_of_2(n)


def dsa_indexer_prepare(
    x: torch.Tensor,
    q_lora: torch.Tensor | None,
    wq_b: torch.Tensor | None,
    wq_b_scale: torch.Tensor | None,
    wk: torch.Tensor,
    wk_scale: torch.Tensor,
    weights_proj: torch.Tensor | None,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    *,
    epsilon: float,
    weights_scale: float,
    key_only: bool = False,
    q_out: torch.Tensor | None = None,
    weights_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Fused DSA indexer prepare (decode): projections + norm + RoPE + Hadamard + FP8 + cache.

    Args:
        x: [T, K] bf16 hidden states (K multiple of 512), T <= 32 decode rows.
        q_lora: [T, QK] bf16 compressed query (QK multiple of 512). ``None`` when
            ``key_only``.
        wq_b: [H*128, QK] fp8 e4m3fn, ``shuffle_weight(w, layout=(16, 16))``.
        wq_b_scale: [H, QK/128] fp32, the 128x128 block scales of ``wq_b``.
        wk: [128, K] fp8 e4m3fn, (16, 16)-shuffled. wk_scale: [1, K/128] fp32.
        weights_proj: [H, K] bf16 head-gate projection, H in {32, 64}.
        norm_weight, norm_bias: [128] bf16 or fp32, the K LayerNorm parameters.
        positions: [T] int64. cos_cache, sin_cache: [max_pos, 32] bf16 rotary
            tables (interleaved / non-NeoX RoPE on the first 64 lanes).
        slot_mapping: [T] int64 index-K cache slots; ``PAD_SLOT_ID`` rows skip the
            cache write (their q and weights are still produced).
        kv_cache: [pages, 64, 132] fp8/uint8 index-K cache in aiter's preshuffled
            layout (``indexer_k_quant_and_cache(..., preshuffle=True)``).
            **Mutated in place.**
        epsilon: LayerNorm epsilon.
        weights_scale: factor applied to the head gate together with the query
            scale, ``softmax_scale * H ** -0.5`` in SGLang.
        key_only: only project K and write the cache (returns ``(None, None)``).
        q_out, weights_out: optional [T, H, 128] fp8 and [T, H] fp32 outputs.

    Returns:
        ``(q_fp8, weights)``: [T, H, 128] fp8 e4m3fn ue8m0-quantized rotated
        queries and [T, H] fp32 ``gate * q_scale * weights_scale``.
    """
    ok, reason = dsa_indexer_prepare_supported(
        x,
        q_lora,
        wq_b,
        wq_b_scale,
        wk,
        wk_scale,
        weights_proj,
        norm_weight,
        norm_bias,
        positions,
        cos_cache,
        sin_cache,
        slot_mapping,
        kv_cache,
        key_only,
    )
    if not ok:
        raise ValueError(f"dsa_indexer_prepare does not support this call: {reason}")
    _LOGGER.info("DSA_INDEXER_PREPARE: x=%s key_only=%s", tuple(x.shape), key_only)

    from aiter.ops.triton._gluon_kernels.gfx950.attention.dsa_indexer_prepare import (
        _finish,
        _packed_key,
        _projections,
    )

    tokens, k = x.shape
    device = x.device
    cache = kv_cache.view(torch.uint8)
    page = kv_cache.shape[1]
    buffer = _fits_32bit(x, wk, cos_cache, sin_cache, cache)

    if key_only:
        cfg = _key_only_config(tokens)
        kpg, bn = cfg["KPG"], cfg["BN"]
        key_parts = triton.cdiv(k, 128 * kpg)
        key_partials = torch.empty(
            (tokens, key_parts, HEAD_DIM), dtype=torch.float32, device=device
        )
        # Rows per MFMA tile: a power of two so the diagonal reduce can reshape.
        rt = 4 if tokens >= 4 else (2 if tokens >= 2 else 1)
        row_first = rt == 4 and tokens % 4 == 0 and tokens >= 8
        _packed_key[(key_parts * (128 // bn) * triton.cdiv(tokens, rt),)](
            x,
            wk,
            wk_scale,
            key_partials,
            tokens,
            k,
            x.stride(0),
            wk.stride(0),
            True,
            rt,
            key_parts,
            buffer,
            bn,
            row_first,
            num_warps=1,
            enable_fp_fusion=False,
        )
        _finish[(tokens, 1)](
            key_partials,
            key_partials,
            key_partials,
            norm_weight,
            norm_bias,
            cos_cache,
            sin_cache,
            positions,
            slot_mapping,
            cache,
            cache,
            key_partials,
            tokens,
            1,
            key_parts,
            1,
            1,
            _p2(key_parts),
            1,
            1,
            cos_cache.stride(0),
            sin_cache.stride(0),
            epsilon,
            page,
            weights_scale,
            True,
            buffer,
            4,
            num_warps=1,
            enable_fp_fusion=False,
        )
        return None, None

    qk = q_lora.shape[1]
    heads = weights_proj.shape[0]
    cfg = _full_config(tokens)
    kpg, qpg, qn, gpg, kn = cfg["KPG"], cfg["QPG"], cfg["QN"], cfg["GPG"], cfg["KN"]
    kg, qg, gg = (
        triton.cdiv(k, 128 * kpg),
        triton.cdiv(qk, 128 * qpg),
        triton.cdiv(k, 128 * gpg),
    )
    buffer = (
        buffer
        and _fits_32bit(q_lora, wq_b, weights_proj)
        and max(
            tokens * kg * HEAD_DIM * 4,
            tokens * heads * qg * HEAD_DIM * 4,
            tokens * heads * gg * 4,
        )
        < 1 << 31
    )
    compact = tokens in (1, 2, 4, 8, 16)
    kp = torch.empty((tokens, kg, HEAD_DIM), dtype=torch.float32, device=device)
    qp = torch.empty((tokens, heads, qg, HEAD_DIM), dtype=torch.float32, device=device)
    gp = torch.empty((gg, tokens, heads), dtype=torch.float32, device=device)
    if q_out is None:
        q_out = torch.empty(
            (tokens, heads, HEAD_DIM), dtype=torch.float8_e4m3fn, device=device
        )
    if weights_out is None:
        weights_out = torch.empty((tokens, heads), dtype=torch.float32, device=device)
    tiles = triton.cdiv(tokens, 16)
    tasks = tiles * (
        (128 // kn) * kg
        + triton.cdiv(heads * HEAD_DIM, qn) * qg
        + triton.cdiv(heads, 32) * gg
    )
    _projections[(tasks,)](
        x,
        q_lora,
        wk,
        wq_b,
        weights_proj,
        wk_scale,
        wq_b_scale,
        kp,
        qp,
        gp,
        tokens,
        k,
        qk,
        heads,
        x.stride(0),
        q_lora.stride(0),
        wk.stride(0),
        wq_b.stride(0),
        weights_proj.stride(0),
        True,
        kpg,
        qpg,
        gpg,
        kn,
        qn,
        buffer,
        compact,
        num_warps=1,
        enable_fp_fusion=True,
    )
    _finish[(tokens, heads + 1)](
        kp,
        qp,
        gp,
        norm_weight,
        norm_bias,
        cos_cache,
        sin_cache,
        positions,
        slot_mapping,
        cache,
        q_out,
        weights_out,
        tokens,
        heads,
        kg,
        qg,
        gg,
        _p2(kg),
        _p2(qg),
        _p2(gg),
        cos_cache.stride(0),
        sin_cache.stride(0),
        epsilon,
        page,
        weights_scale,
        False,
        buffer,
        2,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return q_out, weights_out
