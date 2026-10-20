# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Resolve the installed config for one shape through the real loader and time it like the sweep does."""

import argparse
import json
import sys

from _utils import add_shape_args, shape_from_args
from harness import make_cold_copies, time_with_cuda_graph
from kernels import (
    check_backend,
    ensure_repo_on_path,
    flops,
    get_spec,
    resolve_installed,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", type=get_spec)
    add_shape_args(parser)
    parser.add_argument("--backend", choices=("triton", "gluon"))
    parser.add_argument("--replays", type=int, default=25)
    parser.add_argument("--calls", type=int, default=24)
    parser.add_argument("--cold-mb", type=int, default=1024)
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
    from aiter.ops.triton.utils.config_utils import resolve_config_dir

    arch = arch_info.get_arch()
    backend = args.backend or spec.default_backend(arch)
    check_backend(spec, arch, backend)
    config, is_tuned = resolve_installed(spec, shape, backend)
    config_dir = resolve_config_dir("gemm", spec.config_name, backend=backend)
    print(f"config dir: {config_dir}")
    print(f"backend {backend}, arch {arch}, shape {shape}, is_tuned {is_tuned}")
    print(f"config: {json.dumps(config)}")

    call, inputs, _ = spec.setup(shape, backend)
    copies = make_cold_copies(inputs, args.cold_mb)
    median, low, high = time_with_cuda_graph(
        call, None, copies, args.calls, args.replays
    )
    print(
        f"time: {median:.3f} us (min {low:.3f}, max {high:.3f}), {flops(shape) / median / 1e6:.1f} TFLOPS, "
        f"{len(copies)} cold copies"
    )
    if args.expect_tuned and not is_tuned:
        print("expected a tuned config but the loader returned DEFAULT.json or 'any'")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
