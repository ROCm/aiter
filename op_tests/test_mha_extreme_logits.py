# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import math

import pytest
import torch


def _require_gfx(arch):
    if not torch.version.hip or not torch.cuda.is_available():
        pytest.skip(f"requires {arch}")

    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() != arch:
        pytest.skip(f"requires {arch}")


def _inputs(head_dim, nq, nk, score):
    # One active channel makes every raw QK dot product exact.
    q = torch.zeros((1, nq, 1, head_dim), device="cuda", dtype=torch.bfloat16)
    k = q.new_zeros((1, nk, 1, head_dim))
    v = q.new_ones((1, nk, 1, min(head_dim, 128)))
    q[..., 0] = 1
    k[..., 0] = score
    return q, k, v


def _causal_forward(q, k, v, scale, window_size_left=-1):
    from aiter.ops.mha import fmha_fwd_bf16_opus_fwd, mha_fwd_native_splitkv

    if q.size(-1) == 64:
        return mha_fwd_native_splitkv(q, k, v, None, scale, True, True, 1)
    return fmha_fwd_bf16_opus_fwd(
        q, k, v, scale, True, return_lse=True, window_size_left=window_size_left
    )


def test_mha_native_extreme_negative_logits():
    _require_gfx("gfx942")
    from aiter.ops.mha import mha_fwd_native_splitkv

    # All 64 products contribute to the raw score of -2**101.
    q = torch.ones((1, 64, 1, 64), device="cuda", dtype=torch.bfloat16)
    k = torch.full_like(q, -(2.0**101) / 64)
    v = torch.ones_like(q)
    out, lse = mha_fwd_native_splitkv(q, k, v, None, 64**-0.5, False, True, 1)
    torch.testing.assert_close(out, torch.ones_like(out), rtol=0, atol=0)
    torch.testing.assert_close(lse, torch.full_like(lse, -(2.0**98)), rtol=2e-7, atol=0)


@pytest.mark.parametrize("head_dim", [128, 192])
@pytest.mark.parametrize("default_scale", [True, False], ids=["default", "base2-unit"])
def test_fmha_opus_extreme_negative_logits(head_dim, default_scale):
    _require_gfx("gfx950")
    from aiter.ops.mha import fmha_fwd_bf16_opus_fwd

    # Both tile maxima stay below -1e30 even after D128 scales Q.
    # Using powers of two makes the scaled products exact.
    score = -(2.0**104)
    scale = head_dim**-0.5 if default_scale else math.log(2)
    q, k, v = _inputs(head_dim, 64, 128, 2 * score)
    k[:, 64:, :, 0] = score
    v[:, :64] = 0
    out, lse = fmha_fwd_bf16_opus_fwd(q, k, v, scale, False, return_lse=True)
    torch.testing.assert_close(out, torch.ones_like(out), rtol=0, atol=0)
    # D128 rounds scaled Q to BF16. At this magnitude, adding log(64) rounds away.
    rtol = 4e-3 if head_dim == 128 and default_scale else 2e-7
    torch.testing.assert_close(
        lse, torch.full_like(lse, scale * score), rtol=rtol, atol=0
    )


@pytest.mark.parametrize("head_dim", [64, 128, 192])
@pytest.mark.parametrize(
    "score",
    [-3.0 * 2**31, -5.0 * 2**31],
    ids=["negative-residual", "positive-residual"],
)
def test_equal_logits_with_inexact_base2_scale(head_dim, score):
    _require_gfx("gfx942" if head_dim == 64 else "gfx950")

    # With scale=1 the kernel multiplies by log2(e). A fused subtract gives
    # -256 or +256 at the maximum, where the exponent should be zero.
    q, k, v = _inputs(head_dim, 256, 256, score)
    for tile, value in enumerate([1, 3, 8, 10]):
        v[:, 64 * tile : 64 * (tile + 1)] = value
    window = 191 if head_dim == 128 else -1
    out, lse = _causal_forward(q, k, v, 1.0, window)

    # At each tile end the mean of the visible values is exact in BF16.
    # The last D128 window covers only the final three tiles.
    ends = out[:, 63::64]
    means = v.new_tensor([1, 2, 4, 7 if head_dim == 128 else 5.5])
    torch.testing.assert_close(
        ends, means.view(1, 4, 1, 1).expand_as(ends), rtol=0, atol=0
    )
    # Adding log(number of visible keys) rounds back to score in FP32.
    torch.testing.assert_close(lse, torch.full_like(lse, score), rtol=2e-7, atol=0)


@pytest.mark.parametrize("head_dim", [64, 128, 192])
@pytest.mark.parametrize(
    "scale,raw_scores",
    [
        pytest.param(1.0, None, id="adjacent"),
        # The raw gap overflows FP32. The natural-log gap is only 8.
        pytest.param(2.0**-125, (-(2.0**127), 2.0**127), id="opposite-sign"),
        # The scale fits FP32 but multiplying it by log2(e) overflows.
        pytest.param(1.5 * 2.0**127, (0.0, 2.0**-126), id="large-scale"),
        # Both scores are normal. Their raw difference can flush before scaling.
        pytest.param(2.0**126, (2.0**-126, 1.5 * 2.0**-126), id="subnormal-gap"),
        # The public Python float is normal in FP64 and subnormal in FP32.
        pytest.param(2.0**-127, (-(2.0**127), 2.0**127), id="subnormal-scale"),
    ],
)
def test_distinct_logits_within_and_across_tiles(head_dim, scale, raw_scores):
    _require_gfx("gfx942" if head_dim == 64 else "gfx950")

    # Tile 0 has only low scores, tile 1 only high scores. Later tiles mix
    # both. Giving the low keys V=1 exposes even a small lost weight in O.
    q, k, v = _inputs(head_dim, 256, 255, 0)
    key = torch.arange(k.size(1), device=k.device)
    high = (key >= 64) & ((key < 128) | (key % 2 == 1))
    v.copy_((~high).view(1, -1, 1, 1))
    if raw_scores is None:
        # 255 * -65536 + 255 * -256 - 255 = -16777215. All partial
        # sums, including the high score one greater, are exact in FP32.
        q[..., 0] = q[..., 16] = 255
        q[..., -1] = 1
        k[..., 0] = -65536
        k[..., 16] = -256
        k[0, :, 0, -1] = -255 + high.int()
        lse_atol = 1.0  # One FP32 ulp at this LSE magnitude.
    else:
        low_score, high_score = raw_scores
        k[0, :, 0, 0] = torch.where(high, high_score, low_score)
        lse_atol = 2e-6

    window = 63 if head_dim == 128 else -1
    out, lse = _causal_forward(q, k, v, scale, window)

    # Bottom-right causal alignment leaves row 0 empty. The D128 window
    # also masks whole tiles before the first visible key in later rows.
    assert (out[:, 0] == 0).all()
    assert torch.isneginf(lse[..., 0]).all()
    scores = (q[0, 1:, 0].double() @ k[0, :, 0].double().T) * scale
    last_key = torch.arange(scores.size(0), device=q.device)[:, None]
    scores.masked_fill_(key > last_key, -math.inf)
    if window >= 0:
        scores.masked_fill_(key < last_key - window, -math.inf)
    expected = scores.softmax(-1) @ v[0, :, 0].double()
    # Both P and the output pass through BF16 in the kernels.
    torch.testing.assert_close(out[0, 1:, 0].double(), expected, rtol=0.012, atol=0)
    torch.testing.assert_close(
        lse[0, 0, 1:], scores.logsumexp(-1).float(), rtol=0, atol=lse_atol
    )


@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_native_nonpositive_scale_with_causal_mask(scale):
    _require_gfx("gfx942")
    from aiter.ops.mha import ENABLE_CK, flash_attn_func

    if not ENABLE_CK:
        pytest.skip("native split-K dispatch requires ENABLE_CK")
    q, k, v = _inputs(64, 128, 128, 128)
    k[:, 1::2, :, 0] = -128
    v[:, ::2] = 2
    v[:, 1::2] = 4
    # Two splits force native dispatch. The second split is fully masked
    # for the first 64 rows. Row zero has only one visible key in split 0.
    out, lse = flash_attn_func(
        q, k, v, softmax_scale=scale, causal=True, return_lse=True, num_splits=2
    )
    rows = [0, 63, 127]
    end_value = 3 if scale == 0 else 4
    values = v.new_tensor([2, end_value, end_value]).view(1, 3, 1, 1)
    torch.testing.assert_close(
        out[:, rows], values.expand_as(out[:, rows]), rtol=0, atol=0
    )
    # At each tile end, half the keys have each score. For scale=-1 the
    # smaller weight is exp(-256), which rounds away in the sum.
    ends = [math.log(n) if scale == 0 else 128 + math.log(n // 2) for n in (64, 128)]
    expected_lse = lse.new_tensor([128 * scale, *ends])
    torch.testing.assert_close(lse[0, 0, rows], expected_lse, rtol=0, atol=2e-5)


@pytest.mark.parametrize("head_dim", [64, 128, 192])
def test_empty_causal_row_next_to_live_row(head_dim):
    _require_gfx("gfx942" if head_dim == 64 else "gfx950")

    # Bottom-right alignment masks row zero. Row one sees one zero score.
    q, k, v = _inputs(head_dim, 2, 1, 0)
    out, lse = _causal_forward(q, k, v, 1.0)
    torch.testing.assert_close(out[:, 0], torch.zeros_like(out[:, 0]), rtol=0, atol=0)
    torch.testing.assert_close(out[:, 1], torch.ones_like(out[:, 1]), rtol=0, atol=0)
    assert torch.isneginf(lse[..., 0]).all()
    torch.testing.assert_close(
        lse[..., 1], torch.zeros_like(lse[..., 1]), rtol=0, atol=0
    )


@pytest.mark.parametrize("packed", [False, True], ids=["dense", "packed"])
@pytest.mark.parametrize("scale", [math.log(2), 1.0], ids=["base2-unit", "unit"])
def test_opus_sliding_window_extreme_negative_logits(packed, scale):
    _require_gfx("gfx950")
    from aiter.ops.mha import fmha_fwd_bf16_opus_fwd, fmha_fwd_bf16_opus_varlen_fwd

    # Row i sees keys i and i+1. The second wave group starts at tile 2.
    # Give every KV tile a distinct value to detect a misplaced window.
    score = -(2.0**104)
    q, k, v = _inputs(128, 256, 257, score)
    values = (torch.arange(257, device=q.device) // 64 + 1).to(v.dtype)
    v.copy_(values.view(1, 257, 1, 1))
    expected = ((values[:-1] + values[1:]) / 2).view(256, 1, 1).expand_as(q[0])
    kwargs = dict(softmax_scale=scale, causal=True, return_lse=True, window_size_left=1)
    if packed:
        starts_q = torch.tensor([0, 256], device=q.device, dtype=torch.int32)
        starts_k = torch.tensor([0, 257], device=q.device, dtype=torch.int32)
        out, lse = fmha_fwd_bf16_opus_varlen_fwd(
            q[0],
            k[0],
            v[0],
            seqstart_q=starts_q,
            seqstart_k=starts_k,
            max_seqlen_q=256,
            max_seqlen_k=257,
            **kwargs,
        )
    else:
        out, lse = fmha_fwd_bf16_opus_fwd(q, k, v, **kwargs)
        out = out[0]
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        lse, torch.full_like(lse, scale * score), rtol=2e-7, atol=0
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
