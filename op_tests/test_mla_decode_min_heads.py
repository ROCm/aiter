# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``mla_decode_min_heads``: the query-head count a caller pads an MLA decode
to. Decision-level, so no GPU is needed; the agreement test checks it against
the C++ dispatch rules where the MLA modules are built.

CI runs this via ``python3 op_tests/test_mla_decode_min_heads.py`` (also
pytest-collectable).
"""

import sys

import pytest
import torch

from aiter import mla
from aiter.mla import mla_decode_min_heads

fp8 = torch.float8_e4m3fn
fp8_fnuz = torch.float8_e4m3fnuz
bf16 = torch.bfloat16


@pytest.mark.parametrize(
    "arch, q_dtype, kv_dtype, qlen, persistent, expected",
    [
        ("gfx950", fp8, fp8, 1, True, 12),
        ("gfx950", fp8_fnuz, fp8_fnuz, 1, True, 12),
        ("gfx950", fp8, fp8, 10, True, 12),  # 12 * 10 = 120
        ("gfx950", fp8, fp8, 11, True, 16),  # 12 * 11 = 132 > 128
        ("gfx950", fp8, fp8, 1, False, 16),  # only the persistent remap takes 12
        ("gfx950", bf16, bf16, 5, True, 16),  # 12 * 5 = 60 < 64
        ("gfx950", bf16, bf16, 6, True, 12),  # 12 * 6 = 72, the gqa=64 kernel
        ("gfx950", bf16, bf16, 11, True, 12),  # 12 * 11 = 132, the gqa=32 kernel
        ("gfx950", bf16, bf16, 8, False, 16),
        ("gfx950", bf16, fp8, 1, True, 16),
        ("gfx950", bf16, fp8, 8, True, 16),
        ("gfx950", fp8, bf16, 1, True, 16),
        ("gfx950", torch.uint8, torch.uint8, 1, True, 16),  # byte KV
        ("gfx942", fp8_fnuz, fp8_fnuz, 1, True, 16),
        ("gfx942", bf16, bf16, 8, True, 16),
        ("gfx1250", fp8, fp8, 1, True, 16),
    ],
)
def test_min_heads(arch, q_dtype, kv_dtype, qlen, persistent, expected):
    assert mla_decode_min_heads(q_dtype, kv_dtype, qlen, persistent, arch) == expected


def test_arch_defaults_to_the_device(monkeypatch):
    monkeypatch.setattr(mla, "get_gfx", lambda: "gfx950")
    assert mla_decode_min_heads(fp8, fp8, 1, True) == 12
    monkeypatch.setattr(mla, "get_gfx", lambda: "gfx942")
    assert mla_decode_min_heads(fp8, fp8, 1, True) == 16


def test_rejects_an_empty_query():
    with pytest.raises(ValueError):
        mla_decode_min_heads(fp8, fp8, 0, True, "gfx950")


@pytest.mark.parametrize("qlen", range(1, 17))
@pytest.mark.parametrize("persistent", [True, False])
@pytest.mark.parametrize(
    "arch, q_dtype, kv_dtype",
    [
        ("gfx950", fp8, fp8),
        ("gfx950", bf16, bf16),
        ("gfx950", bf16, fp8),
        ("gfx942", fp8_fnuz, fp8_fnuz),
        ("gfx942", bf16, bf16),
    ],
)
def test_agrees_with_the_dispatch_rules(arch, q_dtype, kv_dtype, qlen, persistent):
    """Where it answers 12, the planner and the asm table serve 12 heads; where
    it answers 16, they do not (so a caller must pad)."""
    try:
        from aiter.ops.attention import get_mla_decode_shape_support

        support = get_mla_decode_shape_support(
            12,
            qlen,
            q_dtype,
            kv_dtype,
            persistent=persistent,
            arch=arch,
            enable_experimental=False,
            flydsl_ps1=False,
        )
    except Exception as e:  # noqa: BLE001 - the MLA modules are not built
        pytest.skip(f"MLA dispatch rules unavailable: {e}")
    if support.supported is None:
        pytest.skip(f"{arch} asm table not built: {support.reasons}")
    narrow = mla_decode_min_heads(q_dtype, kv_dtype, qlen, persistent, arch) == 12
    assert support.supported is narrow, support.reasons


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
