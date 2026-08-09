#!/usr/bin/env python3
"""Device-free compile gate for FlyDSL kernels.

Compiles kernels with COMPILE_ONLY=1 and dumps final ISA, so a migration can be
checked by diffing ISA before/after with no GPU. Pointer args are faked via
from_c_void_p, which needs no device allocation.

Usage:
    PYTHONPATH=/tmp/envstub FLYDSL_GPU_ARCH=gfx950 COMPILE_ONLY=1 \
    FLYDSL_DEBUG_DUMP_ASM=1 FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=<dir> \
    python compile_gate.py [name ...]
"""
import hashlib
import os
import sys
import traceback

from flydsl.compiler.jit_argument import from_c_void_p
from flydsl.expr.typing import Int32

ADDR = 0x100000


def P(elem=Int32):
    return from_c_void_p(elem, ADDR)


def _route_maps():
    from aiter.ops.flydsl.kernels.moe_route_maps import build_moe_route_maps_module

    build_moe_route_maps_module()(P(), P(), P(), P(), 16, 2, 8, 1)


def _topids_to_rows():
    from aiter.ops.flydsl.kernels.moe_route_maps import (
        build_moe_topids_to_rows_module,
    )

    build_moe_topids_to_rows_module()(P(), P(), P(), 16, 8, 1)


# name -> zero-arg callable that triggers one compile
CASES = {
    "route_maps": _route_maps,
    "topids_to_rows": _topids_to_rows,
}


def main():
    wanted = sys.argv[1:] or sorted(CASES)
    rc = 0
    for name in wanted:
        try:
            CASES[name]()
            print(f"[ok]   {name}")
        except Exception as e:  # noqa: BLE001 - report and keep going
            rc = 1
            print(f"[FAIL] {name}: {type(e).__name__}: {str(e)[:300]}")
            if os.environ.get("GATE_TRACEBACK"):
                traceback.print_exc()

    dump = os.environ.get("FLYDSL_DUMP_DIR")
    if dump and os.path.isdir(dump):
        print("--- ISA ---")
        for root, _, files in sorted(os.walk(dump)):
            for f in sorted(files):
                if f.endswith("_final_isa.s"):
                    p = os.path.join(root, f)
                    with open(p, "rb") as fh:
                        h = hashlib.sha256(fh.read()).hexdigest()[:16]
                    print(f"{h}  {os.path.basename(root)}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
