# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
``padded_m_py`` (pure Python, used by the GPU-free AOT wheel build) matches the
compiled ``getPaddedM`` it mirrors.

Requires a GPU: ``get_padded_m`` is dispatched through torch_compile_guard.

Run:
    python3 -m pytest op_tests/tuning_tests/test_padded_m_parity.py -v
"""

import csv
import os

import pytest
import torch

from aiter.aot.flydsl.gemm import DECODE_MAX_M
from aiter.ops.gemm_op_common import get_padded_m, padded_m_py

CSV = os.path.join(
    os.path.dirname(__file__),
    "../../aiter/configs/model_configs/qwen38_27b_a4w4_blockscale_tuned_gemm.csv",
)
MAX_M = max(DECODE_MAX_M, 512)
GENERIC_NK = [(4096, 4096), (8192, 1024)]


def _nk_shapes():
    with open(CSV, newline="") as f:
        shapes = {(int(r["N"]), int(r["K"])) for r in csv.DictReader(f)}
    return sorted(shapes | set(GENERIC_NK))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("gl", [0, 1])
def test_padded_m_parity(gl):
    for n, k in _nk_shapes():
        for m in range(1, MAX_M + 1):
            assert padded_m_py(m, n, k, gl) == get_padded_m(m, n, k, gl), (m, n, k, gl)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
