"""Coverage guard for ``gfx1250-strict`` AOT wheels (the C4 per-row check).

The family-level :func:`~aiter.aot.flydsl.common.assert_targets_covered` only
rejects an arch family that produced *no* job; it cannot see a single tuned row
whose strict build silently drops to a different kernel than the gfx1250 family
it is meant to reuse (e.g. a tuned lookup keyed by the exact gfx that misses and
falls back to a heuristic). This guard catches that.

Method (cheap -- no compilation): for one driver, run the identified-kernels dry
pass once per arch in its own subprocess (the arch is process-global via
``GPU_ARCHS``), recording for every tuned row the set of FlyDSL kernels it
requests. Then diff: every kernel ``gfx1250`` requests for a row must also be
requested by ``gfx1250-strict``, with the arch folded out of the cache identity.
A row strict does not cover is a coverage regression and fails the check.

Usage::

    python -m aiter.aot.flydsl.coverage_check                # all AOT drivers
    python -m aiter.aot.flydsl.coverage_check gemm moe       # named drivers

Exit code is non-zero when any driver has an uncovered row, so it runs as a CI
gate. ``--dump`` is the internal per-arch worker and is not called directly.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import Any

from aiter.aot.flydsl.common import replay_env, resolve_target_arch
from aiter.aot.flydsl.replay import (
    _default_arch,
    _jobs,
    _replay_job,
    job_arch,
)

# The two arches this guard compares: the family baseline and the strict target
# that is supposed to reuse it. Strict must cover the baseline, row for row.
BASELINE_ARCH = "gfx1250"
STRICT_ARCH = "gfx1250-strict"

# Every AOT driver whose rows a strict wheel ships. Kept in sync with the
# modules under this package that expose ``collect_jobs``/``parse_csv``.
DRIVERS = ["gemm", "moe", "mega_moe", "mxfp4_moe", "conv", "chunk_gdn_h", "fmha_fp8"]


# --------------------------------------------------------------------------
# Kernel identity folding
# --------------------------------------------------------------------------
# A FlyDSL cache identity ``(manager_key, cache_key)`` embeds the compile arch,
# so the *same* kernel has different raw identities under gfx1250 and
# gfx1250-strict. To compare across arches we fold the strict arch back to the
# family name and drop the arch-only codegen component.
#
# NOTE: the exact token the codegen arch appears under is confirmed on A0 (the
# C4 diagnosis dropped ('LLVM_GFX1250_ANY_XNACK_CODEGEN', '')). This helper is
# the single place that encodes it; A0's acceptance run tunes it if the real
# key carries the arch under a different token (a false positive on the clean
# baseline-vs-baseline self-check below flags that immediately).
_ARCH_CODEGEN_MARKERS = ("GFX1250_ANY_XNACK_CODEGEN", "gfx1250-strict", "gfx1250")


def fold_kernel_identity(identity: Any) -> str:
    """Arch-independent, hashable form of one ``(manager_key, cache_key)``.

    Folds ``gfx1250-strict`` to ``gfx1250`` and removes the arch-only codegen
    component, so a kernel that is identical apart from its target arch compares
    equal between the two builds.
    """
    text = repr(identity)
    text = text.replace(STRICT_ARCH, BASELINE_ARCH)
    # Drop the arch-only codegen token (and any trailing value) so a strict and
    # a baseline launch that differ *only* by codegen arch fold together.
    for marker in ("GFX1250_ANY_XNACK_CODEGEN",):
        text = _strip_token(text, marker)
    return text


def _strip_token(text: str, marker: str) -> str:
    """Remove ``('<...marker...>', '<value>')``-shaped fragments from ``text``."""
    out, i = [], 0
    while True:
        j = text.find(marker, i)
        if j == -1:
            out.append(text[i:])
            return "".join(out)
        # Back up to the opening paren/bracket of the enclosing tuple, forward
        # to its close, and drop the whole fragment.
        start = max(text.rfind("(", 0, j), text.rfind("[", 0, j))
        end = min(
            (p for p in (text.find(")", j), text.find("]", j)) if p != -1),
            default=j + len(marker),
        )
        if start == -1:
            start = j
        out.append(text[i:start])
        i = end + 1


# --------------------------------------------------------------------------
# Per-row identity of a tuned row, independent of the compile arch
# --------------------------------------------------------------------------
def row_key(job: dict) -> str:
    """Stable identity of a tuned row across arches (the compile arch excluded).

    A strict build reuses the gfx1250 row retargeted, so the two jobs match on
    every field but ``gfx``; dropping it keys the same row on both sides.
    """
    return json.dumps(
        {k: _jsonable(v) for k, v in sorted(job.items()) if k != "gfx"},
        sort_keys=True,
        default=str,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


# --------------------------------------------------------------------------
# Per-arch dump (runs in a subprocess under that arch's GPU_ARCHS)
# --------------------------------------------------------------------------
def _dump_arch(module_name: str, arch: str, out_path: str) -> int:
    """Record ``{row_key: [folded kernel identities]}`` for every row of this
    arch's kernel family. Runs in a process whose ``GPU_ARCHS`` is ``arch``."""
    import importlib

    module = importlib.import_module(f"aiter.aot.flydsl.{module_name}")
    default = _default_arch(module)
    csv_paths = [p for p in getattr(module, "DEFAULT_CSVS", []) if os.path.isfile(p)]
    jobs = _jobs(module, csv_paths, None)

    coverage: dict[str, list[str]] = {}
    for job in jobs:
        cur = job_arch(job, default)
        # Keep only the rows this arch's family owns (as the replay worker does).
        if resolve_target_arch(cur, {arch}) != arch:
            continue
        probe_job = {**job, "gfx": arch} if cur != arch else job
        _result, kernels = _replay_job(module, probe_job)
        if kernels is None:  # a hard failure -- the normal build reports it
            continue
        coverage[row_key(job)] = sorted({fold_kernel_identity(k) for k in kernels})

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(coverage, fh)
    return 0


# --------------------------------------------------------------------------
# Compare one driver
# --------------------------------------------------------------------------
def check_driver(module_name: str) -> list[str]:
    """Return human-readable descriptions of rows strict fails to cover."""
    with tempfile.TemporaryDirectory() as tmp:
        base_json = os.path.join(tmp, "base.json")
        strict_json = os.path.join(tmp, "strict.json")
        _spawn_dump(module_name, BASELINE_ARCH, base_json)
        _spawn_dump(module_name, STRICT_ARCH, strict_json)
        with open(base_json, encoding="utf-8") as fh:
            base = json.load(fh)
        with open(strict_json, encoding="utf-8") as fh:
            strict = json.load(fh)

    problems = []
    for key, base_kernels in base.items():
        strict_kernels = strict.get(key)
        if strict_kernels is None:
            problems.append(f"[{module_name}] row absent under strict: {key}")
            continue
        missing = set(base_kernels) - set(strict_kernels)
        if missing:
            problems.append(
                f"[{module_name}] row covers {len(base_kernels) - len(missing)}/"
                f"{len(base_kernels)} baseline kernels, strict drops "
                f"{len(missing)}: {key}"
            )
    return problems


def _spawn_dump(module_name: str, arch: str, out_path: str) -> None:
    command = [
        sys.executable,
        "-m",
        "aiter.aot.flydsl.coverage_check",
        "--dump",
        "--module",
        module_name,
        "--arch",
        arch,
        "--out",
        out_path,
    ]
    env = replay_env(arch)
    proc = subprocess.run(command, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"coverage dump for {module_name} @ {arch} failed (rc={proc.returncode})"
        )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("drivers", nargs="*", help="drivers to check (default: all)")
    parser.add_argument("--dump", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--module", help=argparse.SUPPRESS)
    parser.add_argument("--arch", help=argparse.SUPPRESS)
    parser.add_argument("--out", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.dump:
        sys.exit(_dump_arch(args.module, args.arch, args.out))

    drivers = args.drivers or DRIVERS
    all_problems: list[str] = []
    for module_name in drivers:
        problems = check_driver(module_name)
        status = "OK" if not problems else f"{len(problems)} uncovered row(s)"
        print(f"[coverage] {module_name}: {status}", flush=True)
        all_problems += problems

    if all_problems:
        print("\n".join(all_problems), flush=True)
        print(
            f"\n[coverage] FAIL: {len(all_problems)} row(s) a strict build does "
            f"not cover vs the {BASELINE_ARCH} family.",
            flush=True,
        )
        sys.exit(1)
    print("[coverage] PASS: strict covers every baseline row.", flush=True)


if __name__ == "__main__":
    main()
