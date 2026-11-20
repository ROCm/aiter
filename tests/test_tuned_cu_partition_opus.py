# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""OPUS A16W16 tuned-row lookup on compute partitions, with the partition mocked.

HIP device 0 is an MI300X in SPX mode (304 CUs) and device 1 a partition of an
MI300X in DPX mode (152 CUs).
"""

import pytest
import torch

from aiter.jit.utils import chip_info
from aiter.ops.opus import policy

# The node has two kinds of GPUs. The current device is device 1.
DEVICES = {0: (304,), 1: (152, 304)}
BF16 = str(torch.bfloat16)


def key(cu, n):
    return ("gfx942", cu, 64, n, 4096, False, BF16, BF16, False, False)


TABLE = {
    key(304, 4096): {"solidx": 7},
    key(152, 1024): {"solidx": 3},
    key(304, 1024): {"solidx": 9},
}


@pytest.mark.parametrize(
    "n, cu_num, device, expected",
    [
        (4096, 152, None, 7),
        (1024, 152, None, 3),  # own row wins
        (4096, 152, 1, 7),
        (4096, 152, 0, None),  # device 0 is no partition
        (4096, 80, None, None),  # MI308X is no partition
    ],
)
def test_opus_a16w16_lookup(monkeypatch, n, cu_num, device, expected):
    monkeypatch.setattr(chip_info, "_tuned_cu_nums_of_device", DEVICES.get)
    monkeypatch.setattr(chip_info, "_shared_tuned_cu_nums", lambda: None)
    monkeypatch.setattr(chip_info, "_current_device", lambda: 1)
    monkeypatch.setattr(policy, "_load_a16w16_opus_tuned", lambda: TABLE)
    policy.lookup_a16w16_opus_config.cache_clear()
    config = policy.lookup_a16w16_opus_config(
        arch="gfx942",
        cu_num=cu_num,
        M=64,
        N=n,
        K=4096,
        has_bias=False,
        input_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        device=device,
    )
    policy.lookup_a16w16_opus_config.cache_clear()
    assert (config and config["solidx"]) == expected
