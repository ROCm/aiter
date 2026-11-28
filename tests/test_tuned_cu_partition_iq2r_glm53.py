# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""IQ2R GLM-5.3 tuned-row lookup on compute partitions, with the partition mocked."""

import pytest

from aiter import iq2r_glm53
from aiter.jit.utils import chip_info

MI355X_DPX = (128, 256)
MI350P = (128,)


def config(name):
    return iq2r_glm53.IQ2RGlm53Config(name, 2, "route9", 1, 1)


TABLE = {
    ("gfx950", 256, 256): [(2, config("full_m2")), (8, config("full_m8"))],
    ("gfx950", 128, 256): [(4, config("own_m4")), (8, config("own_m8"))],
}


@pytest.mark.parametrize(
    "cu_nums, tokens, expected",
    [
        (MI355X_DPX, 2, "full_m2"),
        (MI355X_DPX, 3, "own_m4"),
        (MI355X_DPX, 8, "own_m8"),  # the own row wins at the same M
        (MI350P, 2, "own_m4"),  # MI350P is no partition
    ],
)
def test_iq2r_glm53_lookup(monkeypatch, cu_nums, tokens, expected):
    monkeypatch.setattr(chip_info, "_tuned_cu_nums_of_device", lambda _: cu_nums)
    monkeypatch.setattr(chip_info, "_shared_tuned_cu_nums", lambda: cu_nums)
    monkeypatch.setattr(iq2r_glm53, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(iq2r_glm53, "get_cu_num", lambda: cu_nums[0])
    monkeypatch.setattr(iq2r_glm53, "_tuned_table", lambda _: TABLE)
    monkeypatch.setattr(iq2r_glm53, "_config_problem", lambda *_: None)
    iq2r_glm53.iq2r_glm53_config.cache_clear()
    picked = iq2r_glm53.iq2r_glm53_config(tokens, 256)
    iq2r_glm53.iq2r_glm53_config.cache_clear()
    assert picked.gate_kernel == expected
