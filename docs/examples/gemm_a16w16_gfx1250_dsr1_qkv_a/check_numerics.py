# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Numerics of the installed configs: the wrapper with config=None (the loader's pick) against
F.linear, at the unit test's tolerances (atol=1e-1, rtol=1e-2), bias=None as DSR1 calls it.

    HIP_VISIBLE_DEVICES=0 python3 check_numerics.py
"""

import sys

import torch
import torch.nn.functional as F
from common import K, N, make_inputs

from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config

MS = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]


def main():
    failed = []
    for M in MS:
        x, w, _ = make_inputs(M)
        config, is_tuned = get_gemm_config("GEMM-A16W16", M, N, K, backend="gluon")
        out, ref = gemm_a16w16(x, w), F.linear(x, w)
        err = (out.float() - ref.float()).abs().max().item()
        try:
            torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-2)
            status = "PASS"
        except AssertionError:
            status = "FAIL"
            failed.append(M)
        print(
            f"M={M:6d} {status} max_abs_err={err:.4f} bit_exact={torch.equal(out, ref)} "
            f"is_tuned={is_tuned} {config}",
            flush=True,
        )
    print("FAILED:", failed or "none")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
