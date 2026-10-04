# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Repeat the timing of one config (old or new table) at one M; a GPU fault aborts the process.

HIP_VISIBLE_DEVICES=2 python3 stress.py 2048 new 50
"""

import sys

from common import (
    load_tables,
    make_cold_copies,
    make_inputs,
    resolve,
    time_with_cuda_graph,
)


def main():
    M, label, reps = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    copies = make_cold_copies(make_inputs(M))
    bucket, config = resolve(load_tables()[label], M)
    times = sorted(time_with_cuda_graph(config, copies)[0] for _ in range(reps))
    print(
        f"M={M} {label} {bucket} reps={reps} median={times[reps // 2]:.3f} us {config}"
    )


if __name__ == "__main__":
    main()
