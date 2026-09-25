# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Selector coverage for the FlyDSL fp8_mqa_logits kernel variants.

CI discovers ``op_tests/test_*.py``. These pin the RPB bands, the middle-band
step-down, and the WPB rule; they launch no kernel. The GPU sweep covers the
auto-selected ``r4`` path.
"""

import pytest

from aiter.ops.flydsl.fp8_mqa_logits_kernels import (
    _ARCH,
    KERNEL_VARIANTS,
    _auto_variant,
    _resolve_variant,
)

pytestmark = pytest.mark.skipif(
    _ARCH != "gfx942", reason="gfx942 variant selector and registry"
)

RPB2_MIN_ELEMS = 2**19
RPB4_MIN_ELEMS = 2**21
# The gfx942 rule ignores num_heads; it only has to be a multiple of MFMA_M=16.
NUM_HEADS = 32


def _rpb_wpb(seq_len, seq_len_kv):
    tag = _auto_variant(seq_len, seq_len_kv, NUM_HEADS)
    assert tag in KERNEL_VARIANTS, f"{tag} is not a registered variant"
    _, rpb, wpb = tag.split("_")
    return int(rpb[1:]), int(wpb[1:])


@pytest.mark.parametrize(
    "seq_len, seq_len_kv, expected_rpb",
    [
        (1, 1024, 1),  # below RPB2_MIN_ELEMS
        (64, 4096, 1),
        (256, 1024, 1),
        (64, 8192, 2),  # exactly RPB2_MIN_ELEMS
        (512, 1024, 2),
        (1024, 1560, 2),
        (513, 1024, 1),  # middle band, odd seq_len: step down
        (1, 2**20, 1),
        (2048, 1024, 4),  # exactly RPB4_MIN_ELEMS
        (1024, 131072, 4),  # vLLM long-context indexer prefill
        (1025, 131072, 4),  # odd seq_len: padding accepted up here
    ],
)
def test_auto_variant_rpb_bands(seq_len, seq_len_kv, expected_rpb):
    rpb, _ = _rpb_wpb(seq_len, seq_len_kv)
    assert rpb == expected_rpb


@pytest.mark.parametrize("seq_len_kv", [1024, 8192, 131072])
@pytest.mark.parametrize(
    "threshold, below, at", [(RPB2_MIN_ELEMS, 1, 2), (RPB4_MIN_ELEMS, 2, 4)]
)
def test_auto_variant_rpb_thresholds_track_element_count(
    seq_len_kv, threshold, below, at
):
    """Band edges land on ``seq_len * seq_len_kv``, not on ``seq_len`` alone."""
    seq_len = threshold // seq_len_kv
    assert seq_len % 2 == 0, "keep the step-down out of this assertion"
    assert _rpb_wpb(seq_len - 2, seq_len_kv)[0] == below
    assert _rpb_wpb(seq_len, seq_len_kv)[0] == at


@pytest.mark.parametrize(
    "seq_len_kv, odd_mid, even_mid, odd_above",
    [(1024, 513, 514, 2049), (8192, 65, 66, 257)],
)
def test_auto_variant_steps_down_only_in_middle_band(
    seq_len_kv, odd_mid, even_mid, odd_above
):
    """An indivisible seq_len costs four host-side cats, a fixed price the
    middle band cannot absorb and the top band can."""
    assert _rpb_wpb(odd_mid, seq_len_kv)[0] == 1
    assert _rpb_wpb(even_mid, seq_len_kv)[0] == 2
    assert _rpb_wpb(odd_above, seq_len_kv)[0] == 4


def test_auto_variant_reaches_every_rpb_family():
    seen = {
        _rpb_wpb(seq_len, seq_len_kv)[0]
        for seq_len in (1, 3, 512, 513, 1024, 2048)
        for seq_len_kv in (1024, 8192, 131072)
    }
    assert seen == {1, 2, 4}


@pytest.mark.parametrize(
    "seq_len, seq_len_kv, expected_wpb",
    [
        (2048, 8192, 2),
        (4096, 131072, 2),
        (2047, 8192, 4),
        (2048, 4096, 4),
        (1024, 131072, 4),
        (1, 1024, 4),
    ],
)
def test_auto_variant_wpb_rule_unchanged(seq_len, seq_len_kv, expected_wpb):
    _, wpb = _rpb_wpb(seq_len, seq_len_kv)
    assert wpb == expected_wpb


def test_resolve_variant_precedence(monkeypatch):
    """Explicit > env > shape-adaptive, so the auto path is what runs by default."""
    monkeypatch.delenv("FLYDSL_FP8_MQA_LOGITS_VARIANT", raising=False)
    assert _resolve_variant(None, 1024, 131072, NUM_HEADS) == "mfma_r4_w4"

    monkeypatch.setenv("FLYDSL_FP8_MQA_LOGITS_VARIANT", "mfma_r1_w1")
    assert _resolve_variant(None, 1024, 131072, NUM_HEADS) == "mfma_r1_w1"
    assert _resolve_variant("mfma_r2_w2", 1024, 131072, NUM_HEADS) == "mfma_r2_w2"


def test_resolve_variant_rejects_unknown_tag(monkeypatch):
    monkeypatch.delenv("FLYDSL_FP8_MQA_LOGITS_VARIANT", raising=False)
    with pytest.raises(ValueError, match="unknown fp8_mqa_logits variant"):
        _resolve_variant("mfma_r3_w4", 1024, 131072, NUM_HEADS)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
