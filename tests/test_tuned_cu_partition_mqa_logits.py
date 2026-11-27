# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MQA logits tuned-row lookups on compute partitions, with the partition mocked."""

import pytest

from aiter import mqa_logits, paged_mqa_logits
from aiter.jit.utils import chip_info

MI355X_DPX = (128, 256)
MI350P = (128,)


@pytest.fixture
def device(monkeypatch):
    def install(cu_nums, module):
        monkeypatch.setattr(chip_info, "_tuned_cu_nums_of_device", lambda _: cu_nums)
        monkeypatch.setattr(chip_info, "_shared_tuned_cu_nums", lambda: cu_nums)
        monkeypatch.setattr(module, "get_gfx_runtime", lambda: "gfx950")
        monkeypatch.setattr(module, "get_cu_num", lambda: cu_nums[0])

    return install


@pytest.mark.parametrize(
    "cu_nums, seq_len_kv, expected",
    [
        (MI355X_DPX, 4096, "flydsl"),
        (MI355X_DPX, 2048, "gluon"),
        (MI355X_DPX, 1024, "own"),  # own row at the same lengths wins
        (MI350P, 4096, "gluon"),  # MI350P is no partition: nearest own row
    ],
)
def test_mqa_logits_lookup(device, monkeypatch, cu_nums, seq_len_kv, expected):
    device(cu_nums, mqa_logits)
    table = {
        ("gfx950", 256, 32, 128): {
            (4096, 256): {"backend": "flydsl"},
            (1024, 256): {"backend": "full_chip"},
        },
        ("gfx950", 128, 32, 128): {
            (2048, 256): {"backend": "gluon"},
            (1024, 256): {"backend": "own"},
        },
    }
    monkeypatch.setattr(mqa_logits, "_load_tuned_table", lambda: table)
    mqa_logits.get_mqa_logits_config.cache_clear()
    config = mqa_logits.get_mqa_logits_config(256, seq_len_kv, 32, 128)
    mqa_logits.get_mqa_logits_config.cache_clear()
    assert config["backend"] == expected


@pytest.mark.parametrize(
    "cu_nums, batch, expected",
    [
        (MI355X_DPX, 8, "flydsl"),
        (MI355X_DPX, 2, "own"),
        (MI350P, 8, "own"),
    ],
)
def test_paged_mqa_logits_lookup(device, monkeypatch, cu_nums, batch, expected):
    device(cu_nums, paged_mqa_logits)
    shape = (1, 64, 128, 64, True)
    table = {
        ("gfx950", 256, *shape): {
            8: {"backend": "flydsl", "ChunkK": 256, "WavePerEU": 2},
            1: {"backend": "full_chip"},
        },
        ("gfx950", 128, *shape): {1: {"backend": "own"}},
    }
    monkeypatch.setattr(paged_mqa_logits, "_load_tuned_table", lambda: table)
    paged_mqa_logits.get_paged_mqa_logits_config.cache_clear()
    config = paged_mqa_logits.get_paged_mqa_logits_config(batch, *shape)
    paged_mqa_logits.get_paged_mqa_logits_config.cache_clear()
    assert config["backend"] == expected
