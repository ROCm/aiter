# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""BF16 GEMM tuned-row lookup on compute partitions, with the partition mocked.

HIP device 0 is an MI300X in SPX mode (304 CUs) and device 1 a partition of an
MI300X in DPX mode (152 CUs).
"""

import pytest
import torch

from aiter import tuned_gemm
from aiter.jit.utils import chip_info

BF16 = str(torch.bfloat16)
# The node has two kinds of GPUs. The current device is device 1.
DEVICES = {0: (304,), 1: (152, 304)}


def bf16_key(cu, m, n):
    return ("gfx942", cu, m, n, 4096, False, BF16, BF16, False, False)


TABLE = {
    bf16_key(304, 64, 4096): {"libtype": "asm", "kernelName": "full"},
    bf16_key(152, 64, 1024): {"libtype": "asm", "kernelName": "own"},
    bf16_key(304, 64, 1024): {"libtype": "asm", "kernelName": "full"},
    bf16_key(152, 64, 2048): {"libtype": "asm", "kernelName": "own_padded"},
    bf16_key(304, 50, 2048): {"libtype": "asm", "kernelName": "full_exact"},
}


@pytest.fixture(autouse=True)
def node(monkeypatch):
    monkeypatch.setattr(chip_info, "_tuned_cu_nums_of_device", DEVICES.get)
    monkeypatch.setattr(chip_info, "_shared_tuned_cu_nums", lambda: None)
    monkeypatch.setattr(chip_info, "_current_device", lambda: 1)
    monkeypatch.setattr(tuned_gemm, "get_GEMM_A16W16_config_", lambda: TABLE)
    monkeypatch.setattr(tuned_gemm, "get_padded_m", lambda m, n, k, gl: 64)


@pytest.mark.parametrize(
    "m, n, cu_num, device, expected",
    [
        (64, 4096, 152, None, "full"),
        (64, 1024, 152, None, "own"),
        (50, 2048, 152, None, "full_exact"),  # exact M before padded M
        (50, 4096, 152, None, "full"),
        (64, 4096, 152, 1, "full"),
        (64, 4096, 152, 0, None),  # device 0 is no partition
        (64, 4096, 80, None, None),  # MI308X is no partition
        (64, 4096, 228, None, None),  # MI300A is no partition
    ],
)
def test_bf16_lookup(m, n, cu_num, device, expected):
    config = tuned_gemm.get_GEMM_A16W16_tuned_config(
        m, n, 4096, False, BF16, BF16, gfx="gfx942", cu_num=cu_num, device=device
    )
    assert (config and config["kernelName"]) == expected
