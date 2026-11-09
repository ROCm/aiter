# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The a16w4 SiTUv2 preshuffle is the generic preshuffle.

SGLang's separated path calls ``shuffle_weight(w, (16, 16))`` and
``shuffle_scale(s, E, is_guinterleave=False)``. AITER's own a16w4 tests call
``shuffle_weight_a16w4(w, 16, False)`` and ``shuffle_scale_a16w4(s, E, False)``.
Those are different tile shapes in the source (16x64 bytes versus 16x32, or
16x16 on the packed fp4 view) whose destination offsets coincide, including
the 0x7F pad when a scale's group count is not a multiple of 8. ``gate_up=True``
is a different permutation, so a byte compare can tell the two layouts apart.

CPU only: no GPU, and no quantized kernel.
"""

import torch

from aiter.ops.shuffle import (
    shuffle_scale,
    shuffle_scale_a16w4,
    shuffle_weight,
    shuffle_weight_a16w4,
)

# Kimi-K3 MoE, TP-local intermediate sizes. E is small: the equality is per
# expert and does not depend on the full 896.
_E = 4
_SHAPES = ((3584, 384), (3584, 512))


def _bytes_equal(a, b):
    # torch.equal is not implemented for Float4_e2m1fn_x2. Both shuffles
    # restore the input dtype, so the uint8 view is the byte compare.
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _weight(dtype, experts, rows, packed_k):
    torch.manual_seed(0)
    raw = torch.randint(0, 256, (experts, rows, packed_k), dtype=torch.uint8)
    if dtype is torch.uint8:
        return raw
    return raw.view(dtype)


def test_a16w4_weights_match_generic_preshuffle():
    for model_dim, inter_dim in _SHAPES:
        for dtype in (torch.uint8, torch.float4_e2m1fn_x2):
            for rows, packed_k in (
                (inter_dim * 2, model_dim // 2),  # w13, GGUU
                (model_dim, inter_dim // 2),  # w2
            ):
                weight = _weight(dtype, _E, rows, packed_k)
                a16w4 = shuffle_weight_a16w4(weight, 16, False)
                generic = shuffle_weight(weight, (16, 16))
                assert _bytes_equal(a16w4, generic), (
                    f"a16w4 GGUU != generic preshuffle for {dtype} "
                    f"{(rows, packed_k)}"
                )
                gugu = shuffle_weight_a16w4(weight, 16, True)
                assert not _bytes_equal(gugu, generic), (
                    f"GUGU collapsed onto the generic preshuffle for {dtype} "
                    f"{(rows, packed_k)}"
                )


def test_a16w4_scales_match_generic_preshuffle_and_pad_with_e8m0_one():
    for model_dim, inter_dim in _SHAPES:
        for rows, groups in (
            (inter_dim * 2, model_dim // 32),  # w13 scale
            (model_dim, inter_dim // 32),  # w2 scale; 384 -> 12 groups
        ):
            torch.manual_seed(1)
            scale = torch.randint(0, 255, (_E * rows, groups), dtype=torch.uint8)
            a16w4 = shuffle_scale_a16w4(scale, _E, False)
            generic = shuffle_scale(scale, _E, is_guinterleave=False)
            assert torch.equal(a16w4, generic), (
                f"a16w4 scale != generic scale for {(rows, groups)}"
            )
            gugu = shuffle_scale_a16w4(scale, _E, True)
            assert not torch.equal(gugu, generic), (
                f"GUGU scale collapsed onto the generic preshuffle for {(rows, groups)}"
            )
            # A constant payload makes the 0x7F pad countable after permutation.
            # 384's w2 scale has 12 groups, which both functions pad to 16.
            padded = (groups + 7) // 8 * 8
            flat = torch.full((_E * rows, groups), 0x10, dtype=torch.uint8)
            padded_out = shuffle_scale_a16w4(flat, _E, False)
            assert padded_out.shape == (_E * rows, padded)
            assert torch.equal(
                padded_out, shuffle_scale(flat, _E, is_guinterleave=False)
            )
            pad_bytes = _E * rows * (padded - groups)
            assert int((padded_out == 0x7F).sum()) == pad_bytes
