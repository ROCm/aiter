# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Out-of-bounds write check for the old and new configs of every given M.

y is a view into a larger buffer whose guard bands (256 rows before and after y) hold a
sentinel; the kernel may only write y. Each config is launched REPS times with bias=None, then
the guard bands and y (bit-exact against F.linear) are checked. One process per M keeps a GPU
fault from hiding the other results.

    for M in 1 100 2047 2049; do HIP_VISIBLE_DEVICES=0 python3 guard_test.py $M; done
"""

import sys

import torch
import torch.nn.functional as F
from common import N, load_tables, make_inputs, resolve

from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16

REPS = 5
SENTINEL = -12345.0  # exactly representable in bf16


def main():
    tables = load_tables()
    bad = 0
    for M in map(int, sys.argv[1:]):
        x, w, _ = make_inputs(M)
        ref = F.linear(x, w)
        guard = 256 * N
        for label, table in tables.items():
            bucket, config = resolve(table, M)
            buf = torch.full(
                (guard + M * N + guard,), SENTINEL, dtype=torch.bfloat16, device="cuda"
            )
            y = buf[guard : guard + M * N].view(M, N)
            for _ in range(REPS):
                gemm_a16w16(
                    x, w, bias=None, dtype=torch.bfloat16, y=y, config=dict(config)
                )
            torch.cuda.synchronize()
            before = (buf[:guard] != SENTINEL).sum().item()
            after = (buf[guard + M * N :] != SENTINEL).sum().item()
            exact = torch.equal(y, ref)
            ok = before == 0 and after == 0 and exact
            bad += not ok
            print(
                f"M={M:6d} {label} {bucket:<10} {'OK ' if ok else 'BAD'} "
                f"guard_before={before} guard_after={after} y_exact={exact} {config}",
                flush=True,
            )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
