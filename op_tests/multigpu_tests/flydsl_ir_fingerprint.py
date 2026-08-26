# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dump a path-independent fingerprint of every compiled FlyDSL kernel.

Used to prove that a pure code-movement refactor did not change the generated
code. Two facts make the naive approaches wrong:

  * ``CompiledArtifact._source_ir`` embeds ``loc("/abs/path.py":line:col)``, so it
    changes whenever a function moves file or line even if the emitted code is
    identical. Only ``_ir_text`` is path-free -- it starts at
    ``module attributes {gpu.container_module}`` and carries no source location.
  * The cache directory name is derived from the kernel's SOURCE TEXT
    (flydsl/compiler/jit_function.py:572-578), so directory names change too.
    Fingerprints must therefore be compared as a sorted multiset, not pairwise
    by directory.

Usage:

    rm -rf ~/.flydsl/cache
    <run the workload>
    python op_tests/multigpu_tests/flydsl_ir_fingerprint.py --out before.txt

    <apply the refactor>

    rm -rf ~/.flydsl/cache
    <run the same workload>
    python op_tests/multigpu_tests/flydsl_ir_fingerprint.py --out after.txt

    diff before.txt after.txt && echo "IR UNCHANGED"
"""

import argparse
import hashlib
import os
import pathlib
import pickle
import sys


def fingerprints(cache_root: pathlib.Path) -> list[str]:
    """Sorted sha256 of every artifact's _ir_text. Path- and order-independent."""
    out = []
    unreadable = []
    for pkl in sorted(cache_root.rglob("*.pkl")):
        try:
            artifact = pickle.loads(pkl.read_bytes())
        except Exception as exc:  # noqa: BLE001 - report, do not mask
            unreadable.append(f"{pkl}: {type(exc).__name__}: {exc}")
            continue
        ir = getattr(artifact, "_ir_text", None)
        if not ir:
            unreadable.append(f"{pkl}: no _ir_text")
            continue
        out.append(hashlib.sha256(ir.encode()).hexdigest())
    if unreadable:
        print(f"WARNING: {len(unreadable)} artifact(s) skipped:", file=sys.stderr)
        for line in unreadable[:10]:
            print(f"  {line}", file=sys.stderr)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cache",
        default=os.path.expanduser("~/.flydsl/cache"),
        help="FlyDSL JIT cache root",
    )
    ap.add_argument("--out", required=True, help="where to write the sorted hashes")
    args = ap.parse_args()

    root = pathlib.Path(args.cache)
    if not root.is_dir():
        raise SystemExit(f"cache root does not exist: {root}")
    hashes = fingerprints(root)
    if not hashes:
        raise SystemExit(
            f"no artifacts under {root}; did the workload actually compile anything?"
        )
    pathlib.Path(args.out).write_text("\n".join(hashes) + "\n")
    print(f"{len(hashes)} artifacts -> {args.out} ({len(set(hashes))} distinct)")


if __name__ == "__main__":
    main()
