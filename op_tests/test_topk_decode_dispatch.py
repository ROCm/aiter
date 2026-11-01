# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only tier selection of the FlyDSL decode top-k (no GPU needed)."""

import pytest

from aiter.ops.topk import _select_flydsl_topk_decode_path as select


@pytest.mark.parametrize(
    "args, kwargs, expected",
    [
        # (arch, stable, width, rows, next_n, k)
        (("gfx950", False, 131072, 128, 4, 2048), {}, "sampled"),
        # no seq_lens: sampled ineligible, previous multi-CTA / one-block routing
        (("gfx950", False, 131072, 4, 4, 2048), {"sampled_ok": False}, "multi_cta"),
        (("gfx950", False, 131072, 128, 4, 2048), {"sampled_ok": False}, "one_block"),
        (("gfx942", False, 131072, 128, 4, 2048), {}, "one_block"),
        (("gfx950", False, 131072, 32, 1, 2048), {}, "multi_cta"),
        # non-stable gfx950 multi-CTA row cap is 64 for both layouts
        (("gfx950", False, 131072, 48, 1, 2048), {"sampled_ok": False}, "multi_cta"),
        (("gfx950", False, 131072, 48, 1, 2048), {"packed": True}, "multi_cta"),
        (("gfx950", False, 131072, 64, 1, 2048), {"packed": True}, "multi_cta"),
        (("gfx950", False, 131072, 96, 1, 2048), {"sampled_ok": False}, "one_block"),
        (("gfx950", False, 131072, 96, 1, 2048), {"packed": True}, "one_block"),
        (("gfx950", False, 131072, 32, 1, 2048), {"packed": True}, "multi_cta"),
        (("gfx950", False, 131072, 64, 1, 2048), {"sampled_ok": False}, "multi_cta"),
        # multi-CTA outranks sampled only at next_n == 1 and rows <= 32
        (("gfx950", False, 131072, 48, 1, 2048), {}, "sampled"),
        (("gfx950", False, 131072, 64, 1, 2048), {}, "sampled"),
        (("gfx950", False, 131072, 96, 1, 2048), {}, "sampled"),
        (("gfx950", False, 131072, 16, 4, 2048), {}, "sampled"),
        (("gfx950", False, 131072, 32, 4, 2048), {}, "sampled"),
    ],
)
def test_select_decode_path(args, kwargs, expected):
    assert select(*args, **kwargs) == expected


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-q", __file__]))
