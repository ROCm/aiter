# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Numerical regression for gfx950 FP8 MLA decode softmax.

The FP8 P*V path must scale softmax numerators before converting them to E4M3.
Otherwise, individually small but collectively important probabilities round to
zero while the FP32 denominator still includes their mass.
"""

import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx

_HEADS = 16
_KV_LEN = 2048
_KV_LORA_RANK = 512
_ROPE_DIM = 64
_HEAD_DIM = _KV_LORA_RANK + _ROPE_DIM
_PAGE_SIZE = 1
_SOFTMAX_SCALE = 1.0 / 16.0
_FP8_MAX = 448.0


def _make_case(*, long_tail: bool):
    device = torch.device("cuda")

    # Q_nope is zero so the first 512 cache elements are free to act as V.
    # Q_rope is one, making a cache rope value of -1.75 produce logit -7:
    # 64 * 1 * -1.75 * (1 / 16) == -7.
    q = torch.zeros((1, _HEADS, _HEAD_DIM), dtype=torch.float32, device=device)
    q[..., _KV_LORA_RANK:] = 1.0

    kv = torch.ones(
        (_KV_LEN, _PAGE_SIZE, 1, _HEAD_DIM),
        dtype=torch.float32,
        device=device,
    )
    kv[..., _KV_LORA_RANK:] = -1.75 if long_tail else 0.0
    if long_tail:
        # Put the maximum first so every later online-softmax tile uses the same
        # running maximum. The exact probabilities are 1 and exp(-7).
        kv[0, ..., _KV_LORA_RANK:] = 0.0

    q = q.to(dtypes.fp8)
    kv = kv.to(dtypes.fp8)

    qo_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, _KV_LEN], dtype=torch.int32, device=device)
    kv_indices = torch.arange(_KV_LEN, dtype=torch.int32, device=device)
    kv_last_page_lens = torch.ones((1,), dtype=torch.int32, device=device)
    scale = torch.ones((1,), dtype=torch.float32, device=device)

    return q, kv, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens, scale


def _reference(q: torch.Tensor, kv: torch.Tensor):
    query = q[0].float()
    cache = kv[:, 0, 0].float()
    value = cache[:, :_KV_LORA_RANK]

    scores = torch.einsum("hd,kd->hk", query, cache) * _SOFTMAX_SCALE
    numerators = torch.exp(scores - scores.amax(dim=-1, keepdim=True))
    denominator = numerators.sum(dim=-1, keepdim=True)

    fp32 = torch.einsum("hk,kd->hd", numerators, value) / denominator
    unscaled_fp8 = (
        torch.einsum("hk,kd->hd", numerators.to(dtypes.fp8).float(), value)
        / denominator
    )
    scaled_fp8 = torch.einsum(
        "hk,kd->hd", (numerators * _FP8_MAX).to(dtypes.fp8).float(), value
    ) / (denominator * _FP8_MAX)
    return fp32, unscaled_fp8, scaled_fp8


def _run_aiter(case):
    q, kv, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens, scale = case
    out = torch.empty((1, _HEADS, _KV_LORA_RANK), dtype=torch.bfloat16, device=q.device)

    aiter.mla.mla_decode_fwd(
        q,
        kv,
        out,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        1,
        _PAGE_SIZE,
        1,
        _SOFTMAX_SCALE,
        q_scale=scale,
        kv_scale=scale,
        num_kv_splits=1,
        causal=True,
    )
    torch.cuda.synchronize()
    return out[0].float()


def _max_abs_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual - expected).abs().amax().item()


def test_mla_fp8_softmax_numerics() -> None:
    gfx = get_gfx()
    if gfx != "gfx950":
        print(f"Skipping: requires gfx950, got {gfx}")
        return

    # First prove the fixture and dispatch are sane with representable, uniform
    # probabilities. Then exercise the long tail that exposed the bug.
    uniform = _make_case(long_tail=False)
    uniform_out = _run_aiter(uniform)
    uniform_ref, _, _ = _reference(uniform[0], uniform[1])
    uniform_error = _max_abs_error(uniform_out, uniform_ref)
    assert uniform_error < 0.03, f"uniform control failed: {uniform_error=:g}"

    long_tail = _make_case(long_tail=True)
    out = _run_aiter(long_tail)
    fp32_ref, unscaled_fp8_ref, scaled_fp8_ref = _reference(long_tail[0], long_tail[1])

    fp32_error = _max_abs_error(out, fp32_ref)
    unscaled_fp8_error = _max_abs_error(out, unscaled_fp8_ref)
    scaled_fp8_error = _max_abs_error(out, scaled_fp8_ref)
    print(
        "FP8 MLA softmax means: "
        f"out={out.mean().item():.6f}, "
        f"fp32={fp32_ref.mean().item():.6f}, "
        f"unscaled_fp8={unscaled_fp8_ref.mean().item():.6f}, "
        f"scaled_fp8={scaled_fp8_ref.mean().item():.6f}; "
        "errors: "
        f"{fp32_error=:.6f}, "
        f"{unscaled_fp8_error=:.6f}, "
        f"{scaled_fp8_error=:.6f}"
    )

    assert _max_abs_error(scaled_fp8_ref, fp32_ref) < 0.03
    assert fp32_error < 0.03, (
        "FP8 MLA decode lost softmax numerator mass before P*V: "
        f"{fp32_error=:.6f}, {unscaled_fp8_error=:.6f}, "
        f"{scaled_fp8_error=:.6f}"
    )


if __name__ == "__main__":
    test_mla_fp8_softmax_numerics()
