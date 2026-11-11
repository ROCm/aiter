# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pure-torch native implementation of the DeepSeek-V4.1 decode-layer contracts.

``deepseek_v41_decode_attn_pre`` / ``deepseek_v41_decode_layer_post`` v1 (see
``deepseek_v41_decode.md``). Correctness only: every weight is dequantized to
fp32 for the rows a call touches, every GEMM is an fp32 ``matmul``, and the
TP all-reduces are an all-gather summed in rank order. It runs on any device
torch runs on (CPU included), which is what the CPU tests use.

The numerics follow the model's reference path step by step -- where a value
is rounded to bf16, quantized to MXFP8 (and by which rule), and in what order a
fold is summed -- so a fused provider can be compared against it with the
tolerances of the contract sheet. It does not reproduce any kernel's reduction
order inside a GEMM, so it is not bit-exact with any of them.

Every dimension is read off the tensors (nothing below hard-codes 5120 or 384),
so the same code runs the full model and the reduced shapes of the CPU tests.
The only fixed constants are the format ones: 32-element MX groups, 64 RoPE
dimensions at the tail of a head, the FP8 range and the Sinkhorn schedule.

This module imports only torch (``torch.distributed`` on a multi-rank call).
"""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch

FP8 = torch.float8_e4m3fn
E8M0 = getattr(torch, "float8_e8m0fnu", None)
FP4X2 = getattr(torch, "float4_e2m1fn_x2", None)

FP8_MAX = 448.0
AMAX_FLOOR = 1e-10
GROUP = 32  # MX group (activations 1x32, dense weights 32x32)
ROPE_DIMS = 64  # rotated dims at the tail of every head (GPT-J pairs)
SINKHORN_ITERS = 20
POST_MULT = 2.0
LOG2E = 1.0 / math.log(2.0)
SOFTPLUS_THRESHOLD = 20.0
L_FLOOR = 1e-30
MAX_TOPK = 512
# the head-weight scale of the FP8 index query: index_dim^-0.5 * index_heads^-0.5
# (computed from the shapes below; 1/64 for 32 heads of 128)


class ContractError(ValueError):
    """A call outside the contract (shape, dtype, layout, argument combination)."""


# ---------------------------------------------------------------------------
# formats
# ---------------------------------------------------------------------------


def _is_e8m0(t: torch.Tensor) -> bool:
    return (E8M0 is not None and t.dtype == E8M0) or t.dtype == torch.uint8


def _codes(t: torch.Tensor) -> torch.Tensor:
    """E8M0 scales (``float8_e8m0fnu`` or their uint8 bytes) as int32 codes."""
    return t.view(torch.uint8).to(torch.int32)


def e8m0_to_f32(t: torch.Tensor) -> torch.Tensor:
    """E8M0 codes -> fp32 scales ``2^(code - 127)`` (code 255 is NaN)."""
    code = _codes(t)
    out = torch.ldexp(torch.ones_like(code, dtype=torch.float32), code - 127)
    return torch.where(code == 255, torch.full_like(out, float("nan")), out)


def _codes_to_e8m0(
    code: torch.Tensor, like: torch.Tensor | None = None
) -> torch.Tensor:
    u8 = code.to(torch.uint8)
    if like is not None and like.dtype == torch.uint8:
        return u8
    return u8.view(E8M0) if E8M0 is not None else u8


def _ceil_code(x: torch.Tensor) -> torch.Tensor:
    """Biased exponent of the smallest power of two >= x (x > 0, fp32): the
    ``ceil_pow2_code`` of the CEIL rule."""
    bits = x.contiguous().view(torch.int32)
    return (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(torch.int32)


def _groups(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % GROUP:
        raise ContractError(f"last dim {x.shape[-1]} is not a multiple of {GROUP}")
    return x.reshape(*x.shape[:-1], x.shape[-1] // GROUP, GROUP)


def mxfp8_ceil(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """CEIL-rule MXFP8 of ``x`` along its last dim: (e4m3 values, int32 codes).

    amax floored at 1e-10, ``code = ceil_pow2(amax / 448)``, values
    ``e4m3(clamp(x * 2^-(code - 127), +-448))`` (round to nearest even). This is
    the rule of ATOM's ``quantize_fp8`` and of every activation quant in the layer
    but the q latent.
    """
    g = _groups(x.float())
    amax = g.abs().amax(-1).clamp_min(AMAX_FLOOR)
    code = _ceil_code(amax * torch.tensor(1.0 / FP8_MAX, dtype=torch.float32))
    inv = torch.ldexp(torch.ones_like(amax), 127 - code)
    q = (g * inv[..., None]).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return q.reshape(x.shape), code


def mxfp8_round(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """ROUND-rule MXFP8 (aiter ``_mxfp8_quant_op``), the q latent's: (values, codes).

    amax floored at 1e-10 and rounded at the 3rd mantissa bit to a power of two
    ``p2``; scale ``2^(log2 p2 - 8)`` (unbiased exponent clamped to +-127); values
    ``e4m3(x / scale)``, no clamp.
    """
    g = _groups(x.float())
    amax = g.abs().amax(-1).clamp_min(AMAX_FLOOR)
    bits = amax.contiguous().view(torch.int32)
    p2 = (bits + 0x200000) & 0x7F800000
    unbiased = ((p2 >> 23) - 127 - 8).clamp(-127, 127)
    inv = torch.ldexp(torch.ones_like(amax), -unbiased)
    q = (g * inv[..., None]).to(FP8)
    return q.reshape(x.shape), unbiased + 127


def mx_dequant(values: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """fp32 of MX values ``[..., K]`` with per-32 codes ``[..., K/32]`` (int or e8m0)."""
    scale = codes if codes.dtype == torch.int32 else _codes(codes)
    scale = torch.ldexp(torch.ones_like(scale, dtype=torch.float32), scale - 127)
    g = _groups(values.float())
    return (g * scale[..., None]).reshape(values.shape)


def mxfp8_ceil_dq(x: torch.Tensor) -> torch.Tensor:
    """``x`` through the CEIL-rule MXFP8 round trip, fp32."""
    q, code = mxfp8_ceil(x)
    return mx_dequant(q, code)


_E2M1 = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
)


def fp4_unpack(packed: torch.Tensor) -> torch.Tensor:
    """E2M1 pairs (low nibble first) ``[..., K/2]`` bytes -> fp32 ``[..., K]``."""
    b = packed.view(torch.uint8).to(torch.int64)
    table = _E2M1.to(packed.device)
    lo, hi = table[b & 0xF], table[b >> 4]
    return torch.stack([lo, hi], dim=-1).reshape(*b.shape[:-1], b.shape[-1] * 2)


# ---------------------------------------------------------------------------
# weight layouts (inverses of aiter's preshuffles)
# ---------------------------------------------------------------------------


def unshuffle_16x16(w: torch.Tensor) -> torch.Tensor:
    """Inverse of aiter ``shuffle_weight(w, layout=(16, 16))`` for 1-byte elements
    (FP8 ``[..., N, K]``): the row-major weight's bytes."""
    b = w.view(torch.uint8)
    *lead, n, k = b.shape
    if n % 16 or k % 32:
        raise ContractError(f"preshuffled weight [{n}, {k}] is not a 16x32-byte tiling")
    v = b.reshape(-1, n // 16, k // 32, 2, 16, 16).permute(0, 1, 4, 2, 3, 5)
    return v.reshape(*lead, n, k)


def unshuffle_moe_weight(w: torch.Tensor, gate_up: bool) -> torch.Tensor:
    """Inverse of aiter ``shuffle_weight(w, (16, 16), is_guinterleave=True,
    gate_up=gate_up)`` (gfx950's ``moe_shuffle_weight``) on ``[E, N, K/2]`` bytes.
    The result is the row-major GGUU weight: gate rows, then up rows."""
    b = w.view(torch.uint8)
    e, n, kb = b.shape
    if kb % 64:
        raise ContractError(f"MXFP4 expert weight K bytes {kb} is not a multiple of 64")
    if gate_up:
        n0 = n // 2 // 16
        v = b.reshape(e, n0, 2, kb // 64, 4, 16, 16).permute(0, 2, 1, 5, 3, 4, 6)
    else:
        n0 = n // 16
        v = b.reshape(e, n0, kb // 64, 4, 16, 16).permute(0, 1, 4, 2, 3, 5)
    return v.reshape(e, n, kb)


def unshuffle_moe_scale(s: torch.Tensor, experts: int, gate_up: bool) -> torch.Tensor:
    """Inverse of aiter ``shuffle_scale(s, experts, is_guinterleave=True,
    gate_up=gate_up)`` on ``[E * N, K/32 padded to 8]`` E8M0: ``[E, N, K/32 padded]``.
    """
    b = s.view(torch.uint8)
    rows, kp = b.shape
    n = rows // experts
    if kp % 8 or n % 32:
        raise ContractError(f"expert scale [{rows}, {kp}] is not the shuffled tiling")
    k1, n1 = kp // 8, n // 32
    v = b.reshape(experts, n1, k1, 4, 16, 2, 2)
    if gate_up:
        v = v.permute(0, 6, 1, 4, 2, 5, 3)  # -> (E, 2, N1, 16, K1, 2, 4)
    else:
        v = v.permute(0, 1, 6, 4, 2, 5, 3)  # -> (E, N1, 2, 16, K1, 2, 4)
    return v.reshape(experts, n, kp)


def dense_fp8_weight(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """fp32 ``[N, K]`` of a (16, 16)-preshuffled FP8 weight with 32x32 E8M0 block
    scales ``[N/32, K/32]``."""
    n, k = w.shape
    if tuple(scale.shape) != (n // GROUP, k // GROUP):
        raise ContractError(f"scale {tuple(scale.shape)} for weight [{n}, {k}]")
    values = unshuffle_16x16(w).view(FP8).float()
    s = e8m0_to_f32(scale).repeat_interleave(GROUP, 0).repeat_interleave(GROUP, 1)
    return values * s


def expert_weight(w: torch.Tensor, scale_rows: torch.Tensor, e: int, gate_up: bool):
    """fp32 ``[N, K]`` of routed expert ``e`` (MXFP4, shuffled), from its rows of
    the shuffled weight and its block of the shuffled scale."""
    one = unshuffle_moe_weight(w[e : e + 1], gate_up)[0]
    k = one.shape[-1] * 2
    codes = unshuffle_moe_scale(scale_rows, 1, gate_up)[0][:, : k // GROUP]
    return mx_dequant(fp4_unpack(one), codes.to(torch.int32))


# ---------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------


def _bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16)


def mhc_seam(
    residual: torch.Tensor,
    pre_mix: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_eps: float,
    sublayer_output: torch.Tensor | None = None,
    post_mix: torch.Tensor | None = None,
    comb_mix: torch.Tensor | None = None,
):
    """A manifold-constrained hyper-connection seam (``pre_delayed``).

    ``residual`` [S, HC, H] bf16 and ``pre_mix`` [S, HC] (the incoming collapse
    weights). With a fold, the owed sublayer output is added first:
    ``R[j] = bf16(post[j] y + sum_i comb[i, j] R[i])``. Returns (R, normed, pre,
    post, comb): the seam's bf16 residual, the bf16 normed collapsed row, and the
    next gates (fp32).
    """
    s, hc, h = residual.shape
    if sublayer_output is not None:
        y = sublayer_output.float()
        r = residual.float()
        rn = post_mix.float()[:, :, None] * y[:, None, :]
        rn = rn + torch.einsum("sij,sih->sjh", comb_mix.float(), r)
        rn = _bf16(rn)
    else:
        rn = residual
    x = rn.float().reshape(s, hc * h)
    rstd = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    mix = (x @ hc_fn.float().t()) * rstd
    sc, base = hc_scale.float(), hc_base.float()
    pre = torch.sigmoid(mix[:, :hc] * sc[0] + base[:hc]) + hc_eps
    post = POST_MULT * torch.sigmoid(mix[:, hc : 2 * hc] * sc[1] + base[hc : 2 * hc])
    comb = mix[:, 2 * hc :].reshape(s, hc, hc) * sc[2] + base[2 * hc :].reshape(hc, hc)
    comb = torch.softmax(comb, dim=-1) + hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    for _ in range(SINKHORN_ITERS - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    collapsed = _bf16((pre_mix.float()[:, :, None] * rn.float()).sum(1)).float()
    r = torch.rsqrt(collapsed.pow(2).mean(-1, keepdim=True) + eps)
    normed = _bf16((collapsed * r) * norm_weight.float())
    return rn, normed, pre, post, comb


def rope_tail(x: torch.Tensor, cos_cache, sin_cache, positions, inverse=False):
    """GPT-J (interleaved pairs) RoPE on the last ``ROPE_DIMS`` of ``x`` [S, ..., D],
    fp32 result (the caller rounds). ``inverse`` applies the transpose rotation."""
    xf = x.float().clone()
    pos = positions.long()
    cos = cos_cache[pos].float()  # [S, 32]
    sin = sin_cache[pos].float()
    shape = (x.shape[0],) + (1,) * (x.dim() - 2) + (cos.shape[-1],)
    cos, sin = cos.reshape(shape), sin.reshape(shape)
    tail = xf[..., -ROPE_DIMS:]
    e, o = tail[..., 0::2], tail[..., 1::2]
    if inverse:
        e2, o2 = e * cos + o * sin, o * cos - e * sin
    else:
        e2, o2 = e * cos - o * sin, e * sin + o * cos
    xf[..., -ROPE_DIMS:] = torch.stack([e2, o2], dim=-1).reshape(tail.shape)
    return xf


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)) * w.float()


def _kv_row_format(kv_cache: torch.Tensor, head_dim: int) -> str:
    if kv_cache.dim() != 2:
        raise ContractError("kv_cache must be 2-D [rows, row]")
    if kv_cache.dtype == torch.bfloat16 and kv_cache.shape[1] == head_dim:
        return "bf16"
    if (
        kv_cache.dtype == torch.uint8
        and kv_cache.shape[1] == head_dim + head_dim // GROUP
    ):
        return "mxfp8"
    raise ContractError(
        f"kv_cache {kv_cache.dtype} {tuple(kv_cache.shape)}: expected bf16 [R, {head_dim}]"
        f" or uint8 [R, {head_dim + head_dim // GROUP}]"
    )


def _kv_rows(kv_cache: torch.Tensor, rows: torch.Tensor, head_dim: int) -> torch.Tensor:
    """fp32 key/value rows ``rows`` of the cache."""
    picked = kv_cache[rows.long()]
    if kv_cache.dtype == torch.bfloat16:
        return picked.float()
    return mx_dequant(
        picked[:, :head_dim].view(FP8), picked[:, head_dim:].to(torch.int32)
    )


# ---------------------------------------------------------------------------
# communication
# ---------------------------------------------------------------------------


def _device_group(group):
    """The torch ProcessGroup to communicate device tensors over: ``group`` itself
    or its ``device_group`` (a framework coordinator)."""
    return getattr(group, "device_group", group)


def world_of(group) -> tuple[int, int]:
    """(world size, rank) of ``group``; (1, 0) without one."""
    if group is None:
        return 1, 0
    import torch.distributed as dist

    dg = _device_group(group)
    return dist.get_world_size(dg), dist.get_rank(dg)


def all_reduce_rank_order(partial: torch.Tensor, group) -> torch.Tensor:
    """bf16 of the fp32 sum of every rank's ``partial`` in rank order 0..TP-1."""
    world, _ = world_of(group)
    if world == 1:
        return _bf16(partial.float())
    import torch.distributed as dist

    # fp32 on the wire: exact for a bf16 partial, and every backend carries it
    mine = partial.float().contiguous()
    parts = [torch.empty_like(mine) for _ in range(world)]
    dist.all_gather(parts, mine, group=_device_group(group))
    total = parts[0]
    for p in parts[1:]:
        total = total + p
    return _bf16(total)


def _stream_ctx(stream):
    return torch.cuda.stream(stream) if stream is not None else nullcontext()


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise ContractError(msg)


def _check(t, name, shape, dtypes):
    _need(t is not None, f"{name} is required")
    _need(
        tuple(t.shape) == tuple(shape),
        f"{name} shape {tuple(t.shape)} != {tuple(shape)}",
    )
    _need(t.dtype in dtypes, f"{name} dtype {t.dtype} not in {dtypes}")
    _need(t.is_contiguous(), f"{name} must be contiguous")


_SCALE = tuple(d for d in (E8M0, torch.uint8) if d is not None)
_FP4 = tuple(d for d in (FP4X2, torch.uint8) if d is not None)
_F32 = (torch.float32,)
_BF16 = (torch.bfloat16,)


def _all_or_none(names, values):
    given = [v is not None for v in values]
    _need(all(given) or not any(given), f"{', '.join(names)}: give all or none")
    return all(given)


# ---------------------------------------------------------------------------
# deepseek_v41_decode_attn_pre
# ---------------------------------------------------------------------------


def attn_pre(
    residual,
    pre_mix,
    hc_fn,
    hc_scale,
    hc_base,
    attn_norm_weight,
    wqkv_a,
    wqkv_a_scale,
    q_norm_weight,
    kv_norm_weight,
    wq_b,
    wq_b_scale,
    cos_cache,
    sin_cache,
    positions,
    kv_cache,
    kv_slots,
    pre_out,
    post_out,
    comb_out,
    q_out,
    *,
    layer_index,
    sublayer_output=None,
    post_mix=None,
    comb_mix=None,
    residual_out=None,
    normed_out=None,
    q_latent_out=None,
    q_latent_scale_out=None,
    index_wq_b=None,
    index_wq_b_scale=None,
    index_weights_proj=None,
    index_q_out=None,
    index_q_scale_out=None,
    index_weights_out=None,
    aux_out=None,
    eps=1e-20,
    hc_eps=1e-6,
    group=None,
    stream=None,
) -> None:
    """Native ``deepseek_v41_decode_attn_pre`` v1 (the contract sheet, section 4)."""

    _need(residual.dim() == 3, "residual must be [S, HC, H]")
    s, hc, h = residual.shape
    mix = hc * (hc + 2)
    q_rank, head_dim = q_norm_weight.shape[0], kv_norm_weight.shape[0]
    heads = q_out.shape[1] if q_out.dim() == 3 else -1
    _need(1 <= s, "a step has at least one row")
    _check(residual, "residual", (s, hc, h), _BF16)
    _check(pre_mix, "pre_mix", (s, hc), _F32)
    _check(hc_fn, "hc_fn", (mix, hc * h), _F32)
    _check(hc_scale, "hc_scale", (3,), _F32)
    _check(hc_base, "hc_base", (mix,), _F32)
    _check(attn_norm_weight, "attn_norm_weight", (h,), _BF16)
    _check(wqkv_a, "wqkv_a", (q_rank + head_dim, h), (FP8,))
    _check(
        wqkv_a_scale, "wqkv_a_scale", ((q_rank + head_dim) // GROUP, h // GROUP), _SCALE
    )
    _check(q_norm_weight, "q_norm_weight", (q_rank,), _BF16)
    _check(kv_norm_weight, "kv_norm_weight", (head_dim,), _BF16)
    _check(q_out, "q_out", (s, heads, head_dim), _BF16)
    _check(wq_b, "wq_b", (heads * head_dim, q_rank), (FP8,))
    _check(
        wq_b_scale, "wq_b_scale", (heads * head_dim // GROUP, q_rank // GROUP), _SCALE
    )
    _need(
        cos_cache.dim() == 2 and cos_cache.shape[1] == ROPE_DIMS // 2,
        "cos_cache [P, 32]",
    )
    _check(sin_cache, "sin_cache", tuple(cos_cache.shape), (cos_cache.dtype,))
    _check(positions, "positions", (s,), (torch.int64,))
    _kv_row_format(kv_cache, head_dim)
    _check(kv_slots, "kv_slots", (s,), (torch.int32,))
    _check(pre_out, "pre_out", (s, hc), _F32)
    _check(post_out, "post_out", (s, hc), _F32)
    _check(comb_out, "comb_out", (s, hc, hc), _F32)
    _need(isinstance(layer_index, int) and layer_index >= 0, "layer_index: an int >= 0")
    fold = _all_or_none(
        ("sublayer_output", "post_mix", "comb_mix"),
        (sublayer_output, post_mix, comb_mix),
    )
    if fold:
        _check(sublayer_output, "sublayer_output", (s, h), _BF16)
        _check(post_mix, "post_mix", (s, hc), _F32)
        _check(comb_mix, "comb_mix", (s, hc, hc), _F32)
        _check(residual_out, "residual_out", (s, hc, h), _BF16)
    else:
        _need(residual_out is None, "residual_out is written only by a folding call")
    index = _all_or_none(
        (
            "normed_out",
            "q_latent_out",
            "q_latent_scale_out",
            "index_wq_b",
            "index_wq_b_scale",
            "index_weights_proj",
            "index_q_out",
            "index_q_scale_out",
            "index_weights_out",
        ),
        (
            normed_out,
            q_latent_out,
            q_latent_scale_out,
            index_wq_b,
            index_wq_b_scale,
            index_weights_proj,
            index_q_out,
            index_q_scale_out,
            index_weights_out,
        ),
    )
    if index:
        _need(index_q_out.dim() == 3, "index_q_out must be [S, index_heads, index_dim]")
        ih, idim = index_q_out.shape[1], index_q_out.shape[2]
        _check(normed_out, "normed_out", (s, h), _BF16)
        _check(q_latent_out, "q_latent_out", (s, q_rank), (FP8,))
        _check(q_latent_scale_out, "q_latent_scale_out", (s, q_rank // GROUP), _SCALE)
        _check(index_wq_b, "index_wq_b", (ih * idim, q_rank), (FP8,))
        _check(
            index_wq_b_scale,
            "index_wq_b_scale",
            (ih * idim // GROUP, q_rank // GROUP),
            _SCALE,
        )
        _check(index_weights_proj, "index_weights_proj", (ih, h), _BF16)
        _check(index_q_out, "index_q_out", (s, ih, idim), (FP8,))
        _check(index_q_scale_out, "index_q_scale_out", (s, ih), _F32)
        _check(index_weights_out, "index_weights_out", (s, ih), _F32)
    if aux_out is not None:
        _check(aux_out, "aux_out", (s, h), _BF16)

    with _stream_ctx(stream), torch.no_grad():
        rn, normed, pre, post, comb = mhc_seam(
            residual,
            pre_mix,
            hc_fn,
            hc_scale,
            hc_base,
            attn_norm_weight,
            eps,
            hc_eps,
            sublayer_output,
            post_mix,
            comb_mix,
        )
        if fold:
            residual_out.copy_(rn)
        pre_out.copy_(pre)
        post_out.copy_(post)
        comb_out.copy_(comb)
        # wqkv_a on the CEIL-MXFP8 normed row
        qkv = _bf16(mxfp8_ceil_dq(normed) @ dense_fp8_weight(wqkv_a, wqkv_a_scale).t())
        q_lora, kv_pre = qkv[:, :q_rank], qkv[:, q_rank:]
        # q_norm in fp32 straight into the ROUND-rule MXFP8 q latent
        qr, qr_code = mxfp8_round(_rms(q_lora, q_norm_weight, eps))
        kv = _bf16(_rms(kv_pre, kv_norm_weight, eps))
        q = _bf16(mx_dequant(qr, qr_code) @ dense_fp8_weight(wq_b, wq_b_scale).t())
        q = _bf16(
            rope_tail(q.view(s, heads, head_dim), cos_cache, sin_cache, positions)
        )
        q_out.copy_(q)
        # the window row: RoPE, then the CEIL-rule MXFP8 QAT
        kv_r = _bf16(rope_tail(kv, cos_cache, sin_cache, positions))
        kv8, kv_code = mxfp8_ceil(kv_r)
        live = kv_slots >= 0
        if bool(live.any()):
            rows = kv_slots[live].long()
            if kv_cache.dtype == torch.bfloat16:
                kv_cache[rows] = _bf16(mx_dequant(kv8, kv_code))[live]
            else:
                packed = torch.cat(
                    [kv8.view(torch.uint8), kv_code.to(torch.uint8)], dim=-1
                )
                kv_cache[rows] = packed[live]
        if index:
            normed_out.copy_(normed)
            q_latent_out.copy_(qr)
            q_latent_scale_out.copy_(_codes_to_e8m0(qr_code, q_latent_scale_out))
            iq = _bf16(
                mx_dequant(qr, qr_code)
                @ dense_fp8_weight(index_wq_b, index_wq_b_scale).t()
            )
            iq = _bf16(rope_tail(iq.view(s, ih, idim), cos_cache, sin_cache, positions))
            w = _bf16(normed.float() @ index_weights_proj.float().t())
            qs = iq.float().abs().amax(-1).clamp_min(AMAX_FLOOR) / FP8_MAX
            index_q_out.copy_(
                (iq.float() / qs[..., None]).clamp(-FP8_MAX, FP8_MAX).to(FP8)
            )
            index_q_scale_out.copy_(qs)
            weights_scale = (idim**-0.5) * (ih**-0.5)
            index_weights_out.copy_((w.float() * qs) * weights_scale)
        if aux_out is not None:
            aux_out.copy_(_bf16(rn.float().mean(1)))


# ---------------------------------------------------------------------------
# deepseek_v41_decode_layer_post
# ---------------------------------------------------------------------------


def sparse_attention(q, kv_cache, keys, attn_sink, softmax_scale, head_dim):
    """Sparse MQA decode: ``q`` [S, H, D] bf16 over each token's key rows
    (``keys[t]``: pool rows, K = V), with an attention sink. bf16 [S, H, D]."""
    fp8_pool = kv_cache.dtype == torch.uint8
    qf = mxfp8_ceil_dq(q) if fp8_pool else q.float()
    sink = attn_sink.float() * LOG2E
    out = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    for t, rows in enumerate(keys):
        if rows.numel() == 0:
            continue
        kv = _kv_rows(kv_cache, rows, head_dim)  # [N, D]
        scores = (qf[t] @ kv.t()) * float(torch.tensor(softmax_scale * LOG2E).float())
        m = torch.maximum(scores.amax(-1), sink)
        p = torch.exp2(scores - m[:, None])
        denom = p.sum(-1) + torch.exp2(sink - m)
        out[t] = (_bf16(p).float() @ kv) / denom.clamp_min(L_FLOOR)[:, None]
    return _bf16(out)


def _keys_of(topk_rows, topk_count, window_meta, ring_slots):
    keys = []
    counts, meta = topk_count.tolist(), window_meta.tolist()
    for t, (n, (first, count, _, base)) in enumerate(zip(counts, meta)):
        sel = topk_rows[t, :n]
        win = base + (first + torch.arange(count, device=topk_rows.device)) % ring_slots
        keys.append(torch.cat([sel.long(), win.long()]))
    return keys


def moe_partial(x, gate_weight, gate_bias, w13, w13_scale, w2, w2_scale, shared, opts):
    """This rank's MoE output (routed share + shared share), bf16 [S, H]."""
    top_k, scaling, limit = opts
    s, h = x.shape
    experts = gate_weight.shape[0]
    logits = _bf16(x.float() @ gate_weight.float().t()).float()
    soft = torch.where(
        logits > SOFTPLUS_THRESHOLD, logits, torch.log1p(torch.exp(logits))
    )
    scores = torch.nan_to_num(torch.sqrt(soft), nan=0.0)
    choice = scores + gate_bias.float()
    ids = torch.sort(choice, dim=-1, descending=True, stable=True).indices[:, :top_k]
    picked = scores.gather(1, ids)
    weights = picked * scaling / picked.sum(-1, keepdim=True).clamp_min(1e-20)
    x8 = mxfp8_ceil_dq(x)
    inter2 = w13.shape[1]
    s13 = w13_scale.view(torch.uint8).reshape(experts, -1, w13_scale.shape[-1])
    s2 = w2_scale.view(torch.uint8).reshape(experts, -1, w2_scale.shape[-1])
    routed = torch.zeros(s, h, dtype=torch.float32, device=x.device)
    cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for t in range(s):
        for k in range(top_k):
            e = int(ids[t, k])
            if e not in cache:
                cache[e] = (
                    expert_weight(w13, s13[e], e, gate_up=True),
                    expert_weight(w2, s2[e], e, gate_up=False),
                )
            wgu, wd = cache[e]
            gu = x8[t] @ wgu.t()
            g = gu[: inter2 // 2].clamp(max=limit)
            u = gu[inter2 // 2 :].clamp(-limit, limit)
            hmid = _bf16(g * torch.sigmoid(g) * u)
            acc = mxfp8_ceil_dq(hmid) @ wd.t()
            routed[t] += _bf16(acc * weights[t, k]).float()
    sw13, sw13_s, sw2, sw2_s = shared
    gu = _bf16(x8 @ dense_fp8_weight(sw13, sw13_s).t())
    half = gu.shape[-1] // 2
    g = _bf16(gu[:, :half].float().clamp(max=limit))
    u = gu[:, half:].float().clamp(-limit, limit)
    mid = _bf16(torch.nn.functional.silu(g.float()) * u)
    shared_out = _bf16(mxfp8_ceil_dq(mid) @ dense_fp8_weight(sw2, sw2_s).t())
    return _bf16(routed + shared_out.float())


def layer_post(
    q,
    kv_cache,
    topk_rows,
    topk_count,
    window_meta,
    attn_sink,
    positions,
    cos_cache,
    sin_cache,
    wo_a,
    wo_a_scale,
    wo_b,
    wo_b_scale,
    residual,
    pre_mix,
    post_mix,
    comb_mix,
    hc_fn,
    hc_scale,
    hc_base,
    ffn_norm_weight,
    gate_weight,
    gate_bias,
    w13,
    w13_scale,
    w2,
    w2_scale,
    shared_w13,
    shared_w13_scale,
    shared_w2,
    shared_w2_scale,
    residual_out,
    pre_out,
    post_out,
    comb_out,
    ffn_out,
    *,
    softmax_scale,
    ring_slots,
    layer_index,
    ffn_normed_out=None,
    top_k=6,
    routed_scaling_factor=1.5,
    swiglu_limit=10.0,
    eps=1e-20,
    hc_eps=1e-6,
    group=None,
    stream=None,
) -> None:
    """Native ``deepseek_v41_decode_layer_post`` v1 (the contract sheet, section 5)."""

    _need(q.dim() == 3, "q must be [S, heads, head_dim]")
    s, heads, head_dim = q.shape
    _need(residual.dim() == 3 and residual.shape[0] == s, "residual must be [S, HC, H]")
    _, hc, h = residual.shape
    mix = hc * (hc + 2)
    experts = gate_weight.shape[0]
    _check(q, "q", (s, heads, head_dim), _BF16)
    _kv_row_format(kv_cache, head_dim)
    _need(topk_rows.dim() == 2 and topk_rows.shape[0] == s, "topk_rows must be [S, T]")
    width = topk_rows.shape[1]
    _need(1 <= width <= MAX_TOPK, f"topk_rows width {width} not in 1..{MAX_TOPK}")
    _check(topk_rows, "topk_rows", (s, width), (torch.int32,))
    _check(topk_count, "topk_count", (s,), (torch.int32,))
    _check(window_meta, "window_meta", (s, 4), (torch.int32,))
    _check(attn_sink, "attn_sink", (heads,), _F32)
    _check(positions, "positions", (s,), (torch.int64,))
    _need(
        cos_cache.dim() == 2 and cos_cache.shape[1] == ROPE_DIMS // 2,
        "cos_cache [P, 32]",
    )
    _check(sin_cache, "sin_cache", tuple(cos_cache.shape), (cos_cache.dtype,))
    _need(
        wo_a.dim() == 2 and wo_a.dtype == FP8, "wo_a must be FP8 [G * o_rank, group_k]"
    )
    group_k = wo_a.shape[1]
    _need(
        (heads * head_dim) % group_k == 0, "wo_a columns must divide heads * head_dim"
    )
    groups = heads * head_dim // group_k
    _need(wo_a.shape[0] % groups == 0, "wo_a rows must be groups * o_rank")
    o_rows = wo_a.shape[0]
    _check(wo_a, "wo_a", (o_rows, group_k), (FP8,))
    _check(wo_a_scale, "wo_a_scale", (o_rows // GROUP, group_k // GROUP), _SCALE)
    _check(wo_b, "wo_b", (h, o_rows), (FP8,))
    _check(wo_b_scale, "wo_b_scale", (h // GROUP, o_rows // GROUP), _SCALE)
    _check(residual, "residual", (s, hc, h), _BF16)
    _check(pre_mix, "pre_mix", (s, hc), _F32)
    _check(post_mix, "post_mix", (s, hc), _F32)
    _check(comb_mix, "comb_mix", (s, hc, hc), _F32)
    _check(hc_fn, "hc_fn", (mix, hc * h), _F32)
    _check(hc_scale, "hc_scale", (3,), _F32)
    _check(hc_base, "hc_base", (mix,), _F32)
    _check(ffn_norm_weight, "ffn_norm_weight", (h,), _BF16)
    _check(gate_weight, "gate_weight", (experts, h), _BF16)
    _check(gate_bias, "gate_bias", (experts,), _F32)
    _need(
        w13.dim() == 3 and w13.shape[0] == experts, "w13 must be [E, 2 * inter, H / 2]"
    )
    inter2 = w13.shape[1]
    _check(w13, "w13", (experts, inter2, h // 2), _FP4)
    _check(
        w13_scale, "w13_scale", (experts * inter2, -(-(h // GROUP) // 8) * 8), _SCALE
    )
    _check(w2, "w2", (experts, h, inter2 // 4), _FP4)
    _check(
        w2_scale, "w2_scale", (experts * h, -(-(inter2 // 2 // GROUP) // 8) * 8), _SCALE
    )
    _need(
        shared_w13.dim() == 2 and shared_w13.shape[1] == h,
        "shared_w13 [2 * sh_inter, H]",
    )
    sh2 = shared_w13.shape[0]
    _check(shared_w13, "shared_w13", (sh2, h), (FP8,))
    _check(shared_w13_scale, "shared_w13_scale", (sh2 // GROUP, h // GROUP), _SCALE)
    _check(shared_w2, "shared_w2", (h, sh2 // 2), (FP8,))
    _check(shared_w2_scale, "shared_w2_scale", (h // GROUP, sh2 // 2 // GROUP), _SCALE)
    _check(residual_out, "residual_out", (s, hc, h), _BF16)
    _check(pre_out, "pre_out", (s, hc), _F32)
    _check(post_out, "post_out", (s, hc), _F32)
    _check(comb_out, "comb_out", (s, hc, hc), _F32)
    _check(ffn_out, "ffn_out", (s, h), _BF16)
    if ffn_normed_out is not None:
        _check(ffn_normed_out, "ffn_normed_out", (s, h), _BF16)
    _need(isinstance(ring_slots, int) and ring_slots > 0, "ring_slots: an int > 0")
    _need(isinstance(layer_index, int) and layer_index >= 0, "layer_index: an int >= 0")
    _need(isinstance(top_k, int) and 1 <= top_k <= experts, "top_k: 1..experts")
    _need(isinstance(softmax_scale, (int, float)), "softmax_scale: a Python number")

    with _stream_ctx(stream), torch.no_grad():
        _need(bool((topk_count >= 0).all()), "topk_count must be >= 0")
        _need(
            bool((topk_count <= width).all()), "topk_count must be <= topk_rows width"
        )
        _need(bool((window_meta[:, 1] >= 0).all()), "window count must be >= 0")
        keys = _keys_of(topk_rows, topk_count, window_meta, ring_slots)
        attn = sparse_attention(q, kv_cache, keys, attn_sink, softmax_scale, head_dim)
        # the output projections: inverse RoPE (fp32), wo_a per group, wo_b
        o = rope_tail(attn, cos_cache, sin_cache, positions, inverse=True)
        x = mxfp8_ceil_dq(o.reshape(s, groups, group_k))
        wa = dense_fp8_weight(wo_a, wo_a_scale).view(groups, o_rows // groups, group_k)
        y = _bf16(torch.einsum("sgk,gnk->sgn", x, wa)).reshape(s, o_rows)
        part = _bf16(mxfp8_ceil_dq(y) @ dense_fp8_weight(wo_b, wo_b_scale).t())
        attn_out = all_reduce_rank_order(part, group)
        rn, normed, pre, post, comb = mhc_seam(
            residual,
            pre_mix,
            hc_fn,
            hc_scale,
            hc_base,
            ffn_norm_weight,
            eps,
            hc_eps,
            attn_out,
            post_mix,
            comb_mix,
        )
        residual_out.copy_(rn)
        pre_out.copy_(pre)
        post_out.copy_(post)
        comb_out.copy_(comb)
        if ffn_normed_out is not None:
            ffn_normed_out.copy_(normed)
        partial = moe_partial(
            normed,
            gate_weight,
            gate_bias,
            w13,
            w13_scale,
            w2,
            w2_scale,
            (shared_w13, shared_w13_scale, shared_w2, shared_w2_scale),
            (top_k, float(routed_scaling_factor), float(swiglu_limit)),
        )
        ffn_out.copy_(all_reduce_rank_order(partial, group))
