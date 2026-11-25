# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Resolve the installed config for one shape through the real loader, then time it with the
kernel's benchmark and the final-round settings (TUNE_CONFIRM)."""

import argparse
import functools
import json
import statistics
import sys

from _utils import add_shape_args, shape_from_args
from kernels import (
    check_backend,
    ensure_repo_on_path,
    get_spec,
    resolve_installed,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", type=get_spec)
    add_shape_args(parser)
    parser.add_argument("--backend", choices=("triton", "gluon"))
    parser.add_argument(
        "--expect-tuned",
        action="store_true",
        help="exit 1 unless a specialized file served this shape",
    )
    args = parser.parse_args(argv)

    spec = args.kernel
    shape = shape_from_args(args, spec.dims)
    ensure_repo_on_path()
    from aiter.ops.triton.utils._triton import arch_info
    from aiter.ops.triton.utils.config_utils import (
        get_config_cu_num,
        resolve_config_dir,
    )
    from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
        TUNE_CONFIRM,
        RotatingBufferBench,
    )

    arch = arch_info.get_arch()
    backend = args.backend or spec.default_backend(arch)
    check_backend(spec, arch, backend)
    config, is_tuned = resolve_installed(spec, shape, backend)
    config_dir = resolve_config_dir("gemm", spec.config_name, backend=backend)
    print(f"config dir: {config_dir} (device: {get_config_cu_num()} CUs)")
    print(f"backend {backend}, arch {arch}, shape {shape}, is_tuned {is_tuned}")
    print(f"config: {json.dumps(config)}")

    make_case, _ = spec.setup(shape, backend)
    bencher = RotatingBufferBench()
    launch, rot_bufs, flops, _ = make_case(bencher)
    times_ms = bencher.bench_times(functools.partial(launch, None), **TUNE_CONFIRM)
    median = statistics.median(times_ms) * 1e3
    print(
        f"time: {median:.3f} us (min {min(times_ms) * 1e3:.3f}, max {max(times_ms) * 1e3:.3f}), "
        f"{flops / median / 1e6:.1f} TFLOPS, {len(rot_bufs)} rotating input sets"
    )
    if args.expect_tuned and not is_tuned:
        print("expected a tuned config but the loader returned DEFAULT.json or 'any'")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
