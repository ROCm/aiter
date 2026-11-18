# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only tier selection of the FlyDSL decode top-k (no GPU needed)."""

import pytest

from aiter.ops import topk

select = topk._select_flydsl_topk_decode_path


@pytest.mark.parametrize(
    "args, kwargs, expected",
    [
        # (arch, stable, width, rows, next_n, k)
        # Exact sampled width/next_n gates.
        (("gfx950", False, 98_303, 128, 1, 2048), {}, "one_block"),
        (("gfx950", False, 98_304, 128, 1, 2048), {}, "sampled"),
        (("gfx950", False, 65_536, 128, 1, 2048), {}, "one_block"),
        (("gfx950", False, 65_536, 128, 2, 2048), {}, "sampled"),
        # Multi-CTA precedence boundary and sampled fallback on the next row.
        (("gfx950", False, 131_072, 32, 1, 2048), {}, "multi_cta"),
        (("gfx950", False, 131_072, 33, 1, 2048), {}, "sampled"),
        # Values/missing-lengths, packed rows, stable calls, arch, and k
        # eligibility fall back to the mutually exclusive tier.
        (("gfx950", False, 131_072, 128, 4, 2048), {"sampled_ok": False}, "one_block"),
        (("gfx950", False, 131_072, 64, 4, 2048), {"packed": True}, "multi_cta"),
        (("gfx950", True, 131_072, 32, 4, 2048), {}, "multi_cta"),
        (("gfx942", False, 131_072, 128, 4, 2048), {}, "one_block"),
        (("gfx950", False, 131_072, 4, 4, 256), {}, "one_block"),
    ],
)
def test_select_decode_path(args, kwargs, expected):
    assert select(*args, **kwargs) == expected


def test_sampled_runtime_opt_out(monkeypatch):
    monkeypatch.setattr(topk, "_FLYDSL_TOPK_SAMPLED_ENABLED", False)
    assert select("gfx950", False, 98_304, 128, 1, 2048) == "one_block"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-q", __file__]))
