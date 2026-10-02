# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Communication benchmarks for aiter's collectives.

``--operation`` picks the collective; every other option belongs to it:

* ``ar`` (default) -- all-reduce, and with ``--fusion ar_rmsnorm`` all-reduce +
  residual add + RMSNorm: custom all-reduce, quick-reduce, the FlyDSL
  schedules and RCCL. Implemented in ``bench_comm_ar.py``.
* ``a2a`` -- equal-split all-to-all: the FlyDSL mesh and ring against RCCL.
  Implemented in ``bench_comm_a2a.py``.

``bench_comm.py --operation ar --help`` and ``--operation a2a --help`` list
each operation's options.

Examples::

    python3 op_tests/op_benchmarks/flydsl/bench_comm.py -tp 4 8
    python3 op_tests/op_benchmarks/flydsl/bench_comm.py --operation a2a -tp 8 -s 4096,8192
"""

import argparse
import os
import sys
from multiprocessing import freeze_support

# The operation modules are siblings, not a package. Spawned ranks re-import
# them by module name, so the directory has to be on the path in every process.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OPERATIONS = ("ar", "a2a")


def main(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--operation", choices=OPERATIONS, default="ar")
    args, rest = pre.parse_known_args(argv)
    if args.operation == "a2a":
        import bench_comm_a2a as op
    else:
        import bench_comm_ar as op
    op.main(rest)


if __name__ == "__main__":
    freeze_support()
    main()
