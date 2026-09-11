# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Engram: the conditional-memory block of DeepSeek-V4.1-Flash.

``engram_embedding_lookup`` gathers and dequantizes the fp8 n-gram rows;
``engram_gate_apply`` gates the resulting value against the residual stream
and injects it. Both live under ``fusions/`` because each collapses a chain
of eager ops -- gather + dequant + concat, and norm + dot + gate + add --
that would otherwise materialize large intermediates between them.

The n-gram hashing that produces ``hash_ids`` stays in framework code.
"""

from __future__ import annotations

import torch
import triton

from aiter.ops.triton._triton_kernels.fusions.engram import (
    _engram_embedding_lookup_kernel,
    _engram_gate_apply_kernel,
)
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

_LOOKUP_CONFIG_NAME = "ENGRAM-EMBEDDING-LOOKUP"
_GATE_CONFIG_NAME = "ENGRAM-GATE-APPLY"


def _token_bucketed_config(config_name: str, n_tokens: int) -> dict:
    """Pick the ``M_LEQ_<x>``/``any`` bucket for ``n_tokens``, ascending."""
    cfg_dir = resolve_config_dir("fusions", config_name, backend="triton")
    raw = load_config_json(f"{cfg_dir}/DEFAULT.json")
    for bound in sorted(
        int(key[len("M_LEQ_") :]) for key in raw if key.startswith("M_LEQ_")
    ):
        if n_tokens <= bound:
            return dict(raw[f"M_LEQ_{bound}"])
    return dict(raw["any"])


def _get_lookup_config(n_tokens: int) -> dict:
    return _token_bucketed_config(_LOOKUP_CONFIG_NAME, n_tokens)


def _get_gate_config(n_tokens: int) -> dict:
    return _token_bucketed_config(_GATE_CONFIG_NAME, n_tokens)


def engram_embedding_lookup(
    hash_ids: torch.Tensor,
    table: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor | None = None,
    row_offset: int = 0,
    num_rows: int | None = None,
) -> torch.Tensor:
    """Gather one fp8 embedding row per hash id, dequantize it, and concatenate.

        out[b, l, h * D : (h + 1) * D] = table[id - row_offset] * 2 ** (e - 127)

    where ``id = hash_ids[b, l, h]`` and ``e`` is the raw e8m0 byte of
    ``scale[id - row_offset, d // SCALE_BLOCK]``.

    Key parameters:
    - hash_ids: [B, L, H] int64 row ids in the global (unsharded) id space,
      contiguous. H is (max_ngram_size - 1) * n_hash_heads, e.g. 24
    - table: [R, D] float8_e4m3fn, the rows this rank owns
    - scale: [R, D // SCALE_BLOCK] float8_e8m0fnu (or the equivalent uint8),
      one exponent per SCALE_BLOCK columns
    - out: optional [B, L, H * D] bf16 destination; allocated when omitted
    - row_offset: global id of local row 0 -- this rank owns
      ``[row_offset, row_offset + num_rows)``
    - num_rows: rows this rank owns; defaults to ``table.shape[0]``

    Ids outside the window, including negative ones, yield exactly 0.0 and
    are never used to address the table, so a sharded run is finished by
    all-reducing the outputs. Row arithmetic is int64: R may exceed int32.

    Returns:
    - out: [B, L, H * D] bf16
    """
    _LOGGER.info(
        "ENGRAM_EMBEDDING_LOOKUP: ids=%s table=%s window=(%s, %s)",
        tuple(hash_ids.shape),
        tuple(table.shape),
        row_offset,
        num_rows,
    )

    assert (
        hash_ids.ndim == 3
    ), f"hash_ids must be [B, L, H], got {tuple(hash_ids.shape)}"
    assert (
        hash_ids.dtype == torch.int64
    ), f"hash_ids must be int64, got {hash_ids.dtype}"
    assert hash_ids.is_contiguous(), "hash_ids must be contiguous"
    assert table.ndim == 2, f"table must be [R, D], got {tuple(table.shape)}"
    assert (
        table.dtype == torch.float8_e4m3fn
    ), f"table must be fp8 e4m3, got {table.dtype}"
    assert (
        scale.ndim == 2
    ), f"scale must be [R, D // SCALE_BLOCK], got {tuple(scale.shape)}"
    assert scale.shape[0] == table.shape[0], "table and scale must have the same rows"

    B, L, H = hash_ids.shape
    R, D = table.shape
    assert D % scale.shape[1] == 0, f"D={D} must be a multiple of the scale count"
    SCALE_BLOCK = D // scale.shape[1]

    if num_rows is None:
        num_rows = R
    assert 0 <= num_rows <= R, f"num_rows must be in [0, {R}], got {num_rows}"

    if out is None:
        out = torch.empty((B, L, H * D), dtype=torch.bfloat16, device=hash_ids.device)
    else:
        assert out.shape == (B, L, H * D), f"out must be [{B}, {L}, {H * D}]"
        assert out.dtype == torch.bfloat16, f"out must be bf16, got {out.dtype}"
        assert out.is_contiguous(), "out must be contiguous"

    n_tokens = B * L
    if n_tokens == 0 or H == 0:
        return out

    # e8m0 carries no arithmetic in the kernel: the raw byte is the exponent.
    raw_scale = scale if scale.dtype == torch.uint8 else scale.view(torch.uint8)

    config = _get_lookup_config(n_tokens)
    BLOCK_H = min(config.pop("BLOCK_H"), triton.next_power_of_2(H))
    BLOCK_D = min(config.pop("BLOCK_D"), triton.next_power_of_2(D))
    BLOCK_D = max(BLOCK_D, SCALE_BLOCK)
    assert (
        BLOCK_D % SCALE_BLOCK == 0
    ), f"BLOCK_D={BLOCK_D} must be a multiple of SCALE_BLOCK={SCALE_BLOCK}"

    _engram_embedding_lookup_kernel[(n_tokens, triton.cdiv(H, BLOCK_H))](
        out,
        hash_ids,
        table,
        raw_scale,
        H,
        D,
        row_offset,
        num_rows,
        hash_ids.stride(1),
        out.stride(1),
        table.stride(0),
        raw_scale.stride(0),
        SCALE_BLOCK=SCALE_BLOCK,
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D,
        **config,
    )
    return out


def engram_gate_apply(
    h: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    weight: torch.Tensor,
    token_mask: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    eps: float = 1e-20,
    clamp_value: float = 1e-6,
) -> torch.Tensor:
    """Gate the Engram value against the residual stream and inject it.

        rstd = rsqrt(mean(h^2, -1) + eps) * rsqrt(mean(key^2, -1) + eps)
        dot  = sum(h * weight * key, -1) * rstd * dim ** -0.5
        gate = sigmoid(copysign(clamp(abs(dot), min=clamp_value).sqrt(), dot))
        out  = h + gate.unsqueeze(-1) * value.unsqueeze(-2)

    Key parameters:
    - h: [B, L, C, dim] bf16 residual stream, C mHC copies
    - key: [B, L, C, dim] fp32 Engram key
    - value: [B, L, dim], shared by the C copies
    - weight: [C, dim], the precomputed ``q_weight * k_weight`` product
    - token_mask: optional [B, L] bool; False shuts the gate, so those
      positions pass through untouched
    - out: optional [B, L, C, dim] bf16 destination; allocated when omitted
    - eps: added to both mean squares before the reciprocal square root
    - clamp_value: floor on abs(dot) before the signed square root

    Every reduction and the injection accumulate in fp32; only the store
    rounds. Fusing the two normalizations, the dot product and the add keeps
    h in cache between the reduction and the injection, and never
    materializes the [B, L, C] gate or the broadcast value.

    Returns:
    - out: [B, L, C, dim] bf16
    """
    _LOGGER.info(
        "ENGRAM_GATE_APPLY: h=%s masked=%s",
        tuple(h.shape),
        token_mask is not None,
    )

    assert h.ndim == 4, f"h must be [B, L, C, dim], got {tuple(h.shape)}"
    B, L, C, dim = h.shape
    assert key.shape == h.shape, f"key must be {tuple(h.shape)}, got {tuple(key.shape)}"
    assert value.shape == (B, L, dim), f"value must be [{B}, {L}, {dim}]"
    assert weight.shape == (C, dim), f"weight must be [{C}, {dim}]"
    for name, tensor in (("h", h), ("key", key), ("value", value), ("weight", weight)):
        assert tensor.is_contiguous(), f"{name} must be contiguous"
    if token_mask is not None:
        assert token_mask.shape == (B, L), f"token_mask must be [{B}, {L}]"
        assert token_mask.is_contiguous(), "token_mask must be contiguous"

    if out is None:
        out = torch.empty((B, L, C, dim), dtype=torch.bfloat16, device=h.device)
    else:
        assert out.shape == h.shape, f"out must be {tuple(h.shape)}"
        assert out.dtype == torch.bfloat16, f"out must be bf16, got {out.dtype}"
        assert out.is_contiguous(), "out must be contiguous"

    n_tokens = B * L
    if n_tokens == 0:
        return out

    config = _get_gate_config(n_tokens)
    BLOCK_D = min(config.pop("BLOCK_D"), triton.next_power_of_2(dim))

    _engram_gate_apply_kernel[(n_tokens,)](
        out,
        h,
        key,
        value,
        weight,
        token_mask,
        C,
        dim,
        h.stride(1),
        h.stride(2),
        key.stride(1),
        key.stride(2),
        value.stride(1),
        weight.stride(0),
        out.stride(1),
        out.stride(2),
        eps,
        clamp_value,
        dim**-0.5,
        HAS_MASK=token_mask is not None,
        BLOCK_C=triton.next_power_of_2(C),
        BLOCK_D=BLOCK_D,
        **config,
    )
    return out
