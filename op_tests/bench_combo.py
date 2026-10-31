# SPDX-License-Identifier: MIT
"""Architecture dispatcher for the existing uBench combo infrastructure.

The main-branch backend supports gfx950. Architecture-specific suites can be
added without importing their GPU modules on unsupported hardware.
"""

from __future__ import annotations

import argparse
import importlib
import sys

BACKENDS = {"gfx950": "bench_gfx950_combo"}


def backend_for(arch):
    try:
        return BACKENDS[arch]
    except KeyError:
        raise ValueError(
            f"Unsupported architecture {arch!r}; supported: {', '.join(BACKENDS)}"
        ) from None


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--arch", choices=list(BACKENDS))
    args, remaining = parser.parse_known_args(argv)
    if args.arch is None:
        if "--help" in remaining or "-h" in remaining:
            print(
                "Usage: bench_combo.py --arch {gfx950} [backend options]\n"
                "Omit --arch to detect the current GPU. Use --arch gfx950 --help for operator options."
            )
            return
        from aiter.jit.utils.chip_info import get_gfx

        arch = get_gfx()
    else:
        arch = args.arch
    backend = importlib.import_module(backend_for(arch))
    sys.argv = [sys.argv[0], *remaining]
    backend.main()


if __name__ == "__main__":
    main()
