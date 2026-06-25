# SPDX-License-Identifier: MIT
"""Shared helpers for sparse MLA prefill tests and benchmarks (no vLLM dep)."""

from __future__ import annotations

import math

import torch

NOPE_HEAD_DIM = 448
ROPE_HEAD_DIM = 64
PACKED_HEAD_DIM = NOPE_HEAD_DIM + ROPE_HEAD_DIM
H_PROD = 128
_FNUZ = torch.float8_e4m3fnuz
_OCP = torch.float8_e4m3fn

# ---- GLM-5 / DeepSeek-V3.2 (ROCM_AITER_MLA_SPARSE) flat fp8 cache ----------
# Per docs/sparse-mla-prefill/01 + vLLM rocm_aiter_mla_sparse.get_kv_cache_shape:
# cache is a flat [num_blocks, block_size, 576] tensor, both the 512-d latent
# and the 64-d rope stored as fp8 with a single per-tensor scale (layer._k_scale,
# kept outside the cache).  The kernel reads the bytes straight to LDS and folds
# the per-tensor scale into the QK softmax scale and the output.
GLM_LATENT_DIM = 512
GLM_ROPE_DIM = 64
GLM_HEAD_DIM = GLM_LATENT_DIM + GLM_ROPE_DIM  # 576
GLM_V_DIM = GLM_LATENT_DIM  # 512


def pack_fp8_ds_mla_cache(
    kv: torch.Tensor,
    block_size: int,
    *,
    is_extra: bool = False,
    scale_byte: int | list[int] = 127,
) -> torch.Tensor:
    assert kv.shape[-1] == PACKED_HEAD_DIM
    num_tokens = kv.shape[0]
    num_blocks = (num_tokens + block_size - 1) // block_size
    cache = torch.zeros((num_blocks, block_size, 584), dtype=torch.uint8, device=kv.device)
    cache_flat = cache.view(torch.uint8).flatten()
    nope_dtype = _OCP if is_extra else _FNUZ
    kv_nope_fp8 = kv[:, :NOPE_HEAD_DIM].to(nope_dtype).view(torch.uint8)
    kv_rope_u8 = kv[:, NOPE_HEAD_DIM:].contiguous().view(torch.uint8)
    for slot in range(num_tokens):
        block_idx = slot // block_size
        pos = slot % block_size
        block_base = block_idx * cache.stride(0)
        token_base = block_base + pos * 576
        scale_base = block_base + block_size * 576 + pos * 8
        cache_flat[token_base : token_base + NOPE_HEAD_DIM].copy_(kv_nope_fp8[slot])
        cache_flat[
            token_base + NOPE_HEAD_DIM : token_base + NOPE_HEAD_DIM + ROPE_HEAD_DIM * 2
        ].copy_(kv_rope_u8[slot])
        if isinstance(scale_byte, int):
            cache_flat[scale_base : scale_base + 7].fill_(scale_byte)
        else:
            sb = torch.tensor(list(scale_byte), dtype=torch.uint8, device=kv.device)
            cache_flat[scale_base : scale_base + 7].copy_(sb)
    return cache


def ragged_from_rows(rows: list[list[int]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    flat = [slot for row in rows for slot in row]
    indptr = [0]
    for row in rows:
        indptr.append(indptr[-1] + len(row))
    return (
        torch.tensor(flat, dtype=torch.int32, device=device),
        torch.tensor(indptr, dtype=torch.int32, device=device),
    )


def identity_block_table(num_slots: int, block_size: int, device: torch.device) -> torch.Tensor:
    num_blocks = (num_slots + block_size - 1) // block_size
    return torch.arange(num_blocks, dtype=torch.int32, device=device).reshape(1, num_blocks)


def gen_kv(num_tokens: int, seed: int, device: str = "cuda", scale: float = 0.125) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    return (
        torch.randn(num_tokens, PACKED_HEAD_DIM, generator=g, dtype=torch.bfloat16, device=device)
        * scale
    )


def gen_q(sq: int, h: int, seed: int, device: str = "cuda", scale: float = 0.125) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(sq, h, PACKED_HEAD_DIM, generator=g, dtype=torch.bfloat16, device=device) * scale


def gen_ragged_rows(
    num_queries: int,
    topk: int,
    num_slots: int,
    seed: int,
    device: str = "cuda",
) -> list[list[int]]:
    g = torch.Generator(device=device).manual_seed(seed)
    rows: list[list[int]] = []
    for _ in range(num_queries):
        slots = torch.randint(0, num_slots, (topk,), generator=g, device=device)
        rows.append(slots.tolist())
    return rows


def merge_two_region_csrs(
    main_rows: list[list[int]],
    extra_rows: list[list[int]],
    extra_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge per-query main+extra slot lists into one flat CSR (main then extra)."""
    device = torch.device("cuda")
    merged: list[list[int]] = []
    for m_row, e_row in zip(main_rows, extra_rows):
        merged.append(list(m_row) + [int(s) + extra_offset for s in e_row])
    return ragged_from_rows(merged, device)


def default_scale() -> float:
    return PACKED_HEAD_DIM ** -0.5


# ---------------------------------------------------------------------------
# GLM-5 / DeepSeek-V3.2 helpers (flat 576 fp8 cache, per-tensor scale)
# ---------------------------------------------------------------------------
def pack_glm_fp8_cache(
    kv: torch.Tensor,
    block_size: int,
    *,
    kv_scale: float = 1.0,
) -> torch.Tensor:
    """Pack [num_tokens, 576] bf16 "true" KV into a flat fnuz fp8 cache.

    Stores ``f = round_fnuz(kv / kv_scale)`` so the kernel's effective value
    ``kv_scale * f`` reproduces ``kv`` (modulo fp8 rounding).  Returns an
    e4m3fnuz tensor ``[num_blocks, block_size, 576]`` (gfx942 native fp8).
    """
    assert kv.shape[-1] == GLM_HEAD_DIM
    num_tokens = kv.shape[0]
    num_blocks = (num_tokens + block_size - 1) // block_size
    cache = torch.zeros((num_blocks, block_size, GLM_HEAD_DIM), dtype=_FNUZ, device=kv.device)
    f = (kv.to(torch.float32) / float(kv_scale)).to(_FNUZ)
    cache.view(num_blocks * block_size, GLM_HEAD_DIM)[:num_tokens] = f
    return cache


def gen_kv_glm(num_tokens: int, seed: int, device: str = "cuda", scale: float = 0.125) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(num_tokens, GLM_HEAD_DIM, generator=g, dtype=torch.bfloat16, device=device) * scale


def gen_q_glm(sq: int, h: int, seed: int, device: str = "cuda", scale: float = 0.125) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(sq, h, GLM_HEAD_DIM, generator=g, dtype=torch.bfloat16, device=device) * scale


def default_scale_glm() -> float:
    return GLM_HEAD_DIM ** -0.5


def _glm_cache_as_fp8(cache: torch.Tensor) -> torch.Tensor:
    if cache.dtype == _FNUZ:
        return cache
    if cache.dtype == torch.uint8:
        return cache.view(_FNUZ)
    raise TypeError(f"glm cache must be fnuz fp8 or uint8, got {cache.dtype}")


def ref_prefill_glm(
    q: torch.Tensor,
    cache: torch.Tensor,
    rows: list[list[int]],
    scale: float,
    block_size: int,
    *,
    q_scale: float = 1.0,
    kv_scale: float = 1.0,
    attn_sink: torch.Tensor | None = None,
) -> torch.Tensor:
    """f32 oracle for the GLM flat fp8 cache, reproducing the kernel's fp8 rounding.

    q: [sq, H, 576] bf16. ``rows[qi]`` = valid slot ids for query qi (caller
    pre-filters invalid slots). Returns [sq, H, 512] bf16.
    """
    sq, H, D = q.shape
    assert D == GLM_HEAD_DIM
    q_eff = (q.to(torch.float32) / float(q_scale)).to(_FNUZ).to(torch.float32)
    qk_scale = scale * float(q_scale) * float(kv_scale)
    flat = _glm_cache_as_fp8(cache).reshape(-1, GLM_HEAD_DIM)
    out = torch.zeros(sq, H, GLM_V_DIM, dtype=torch.float32, device=q.device)
    for qi in range(sq):
        if not rows[qi]:
            continue
        kv = torch.stack([flat[int(s)].to(torch.float32) for s in rows[qi]])  # fp8 dequant, no kv_scale
        for h in range(H):
            scores = torch.mv(kv, q_eff[qi, h]) * qk_scale
            if attn_sink is not None:
                scores_s = torch.cat([scores, attn_sink[h].float().reshape(1)])
                probs = torch.softmax(scores_s, dim=0)[:-1]
            else:
                probs = torch.softmax(scores, dim=0)
            out[qi, h] = torch.sum(probs[:, None] * kv[:, :GLM_V_DIM], dim=0) * float(kv_scale)
    return out.to(torch.bfloat16)
