#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ISA-identity gate for the gfx950 dense FlyDSL FMHA path.

Compiles every dense config of a source tree (no GPU, COMPILE_ONLY) and dumps,
per compiled kernel, the disassembly and the code-object notes, so two trees
can be proven byte-identical in instruction stream and resource usage.

Usage:
    python scripts/fmha_isa_gate.py --src <tree> --out <dir> [--jobs N]
    python scripts/fmha_isa_gate.py --compare <dirA> <dirB> [--reverify <srcA> <srcB>]

Mode 1 compiles <tree> into <out>/artifacts/ (one subprocess per shard, each
with PYTHONPATH=<tree>, so the tree under test is pinned explicitly rather
than by cwd). Mode 2 diffs two such output dirs and exits non-zero on any
difference or on a job present on one side only.

Config set, enumerated from the tree under test (nothing hand-copied):
  - every job of aiter/aot/flydsl/fmha_fp8.py default_jobs();
  - every (heads, kv_heads, head_dim, head_dim_v) head shape reachable from the
    parametrize axes of the test_fp8_* tests in op_tests/test_flydsl_fmha.py,
    each expanded over all four layouts and the wrapper's full variant space
    (threshold, block_m, splits 1..16, batch-interleave, causal, return_lse)
    through the AOT module's own jobs_for_shape. Shapes the kernel's own gate
    rejects are skipped and listed in <out>/skipped_shapes.txt.
  Splits > 1 jobs include the combine kernel (it is part of the same launcher).
  Not covered: lazy_rescale=False / daz=False / setprio / stagger variants,
  which only appear in the test's softmax checks and are not AOT-reachable.

--reverify: the parallel compile is not bit-deterministic for a few jobs (the
generated ISA varies with load), so each job reported DIFFERENT is recompiled
serially, REVERIFY_N times per tree with a fresh cache dir, and classified:
  IDENTICAL-AFTER-REVERIFY  some serial A variant matches some serial B one,
                            one variant per side
  FLAKY-BASE                same match, but a side yields several variants
                            (passes; per-side variant counts are printed)
  DIFFERENT                 no A/B variant match (fails, flaky or not)
Only DIFFERENT or a missing job fails the exit code. Variants are written to
<dirB>/reverify/<job>/.

Normalizations (the only ones; instruction text is never altered):
  - objdump: addresses and raw encodings are not emitted (--no-addresses,
    --no-show-raw-insn) and the trailing AMDGPU "// addr: encoding" comment is
    stripped, since it repeats the address and encoding;
  - the objdump header line carrying the input filename is dropped;
  - readelf notes: none needed beyond the filename not being in the output.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import os
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

ARCH = "gfx950"
REVERIFY_N = 3
LLVM_BIN_CANDIDATES = (
    "/opt/rocm/llvm/bin",
    # pip-ROCm-SDK layout on this node (no /opt/rocm).
    *sorted(
        Path(sys.prefix).glob("lib/python*/site-packages/_rocm_sdk_devel/lib/llvm/bin")
    ),
)
TEST_FILE = "op_tests/test_flydsl_fmha.py"
ALIAS = {
    "num_heads": "H",
    "H": "H",
    "num_kv_heads": "Hkv",
    "H_KV": "Hkv",
    "head_dim": "D",
    "D": "D",
    "head_dim_v": "Dv",
    "Dv": "Dv",
}
# Defaults the tests' _run_fp8_shape uses when an axis is not parametrized.
DEFAULT_SHAPE = {"H": 12, "D": 192, "Dv": 128}


def llvm_tool(name: str) -> str:
    for d in LLVM_BIN_CANDIDATES:
        p = Path(d) / name
        if p.exists():
            return str(p)
    raise SystemExit(f"{name} not found under {LLVM_BIN_CANDIDATES}")


# ---------------------------------------------------------------- enumeration
def _test_grid_shapes(tree: Path) -> set[tuple[int, int, int, int]]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("fmha_test_grid", tree / TEST_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    shapes = set()
    for name, fn in vars(mod).items():
        if not name.startswith("test_fp8"):
            continue
        axes = []
        for mark in getattr(fn, "pytestmark", []):
            if mark.name != "parametrize":
                continue
            names = (
                [s.strip() for s in mark.args[0].split(",")]
                if isinstance(mark.args[0], str)
                else list(mark.args[0])
            )
            if not any(n in ALIAS for n in names):
                continue
            rows = []
            for v in mark.args[1]:
                v = getattr(v, "values", v)  # unwrap pytest.param
                if len(names) == 1:
                    v = (v[0] if isinstance(v, (tuple, list)) and len(v) == 1 else v,)
                rows.append({ALIAS[n]: x for n, x in zip(names, v) if n in ALIAS})
            axes.append(rows)
        for combo in itertools.product(*axes) if axes else ():
            d = dict(DEFAULT_SHAPE)
            for r in combo:
                d.update(r)
            shapes.add((d["H"], d.get("Hkv", d["H"]), d["D"], d["Dv"]))
    return shapes


def enumerate_jobs(tree: Path) -> tuple[list[dict], list[tuple]]:
    from aiter.aot.flydsl import fmha_fp8

    jobs = list(fmha_fp8.default_jobs())
    skipped = []
    for shape in sorted(_test_grid_shapes(tree)):
        if fmha_fp8.unsupported_reason(*shape):
            skipped.append(shape)
            continue
        for layout in fmha_fp8.LAYOUTS:
            jobs.extend(fmha_fp8.jobs_for_shape(*shape, layout))
    jobs = fmha_fp8.dedupe_jobs(jobs)
    keys = [j["kernel_name"] for j in jobs]
    assert len(set(keys)) == len(keys), "kernel_name is not a unique job identity"
    return sorted(jobs, key=lambda j: j["kernel_name"]), skipped


# ----------------------------------------------------------------- extraction
def _decode_mlir_bytes(s: str, i: int) -> bytes:
    """Decode the MLIR escaped string literal starting at s[i] (after the quote)."""
    out = bytearray()
    while s[i] != '"':
        c = s[i]
        if c == "\\":
            n = s[i + 1]
            if n in '"\\':
                out.append(ord(n))
                i += 2
            elif n == "n":
                out.append(10)
                i += 2
            elif n == "t":
                out.append(9)
                i += 2
            else:
                out.append(int(s[i + 1 : i + 3], 16))
                i += 3
        else:
            out += c.encode()
            i += 1
    return bytes(out)


def code_objects(cache_dir: Path) -> list[bytes]:
    """All gpu.binary ELF blobs in the compiled artifacts under cache_dir, in a
    deterministic order (sorted by launcher dir name is cache-hash based, so
    order by blob content hash instead)."""
    blobs = []
    for pkl in sorted(cache_dir.glob("*/*.pkl")):
        with open(pkl, "rb") as fh:
            art = pickle.load(fh)
        for m in re.finditer(r'bin = "', art.ir):
            blobs.append(_decode_mlir_bytes(art.ir, m.end()))
    return sorted(blobs, key=lambda b: hashlib.sha256(b).hexdigest())


def dump_one(elf: bytes, stem: Path, tools: tuple[str, str]) -> None:
    objdump, readelf = tools
    elf_path = stem.with_suffix(".elf.tmp")
    elf_path.write_bytes(elf)
    try:
        dis = subprocess.run(
            [
                objdump,
                "-d",
                f"--mcpu={ARCH}",
                "--no-show-raw-insn",
                "--no-addresses",
                str(elf_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        notes = subprocess.run(
            [readelf, "--notes", str(elf_path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    finally:
        elf_path.unlink()
    lines = []
    for ln in dis.splitlines():
        if ln.startswith(str(elf_path)):  # "<file>:\tfile format ..." header
            continue
        lines.append(re.sub(r"\s*//.*$", "", ln).rstrip())
    Path(f"{stem}.s").write_text("\n".join(lines) + "\n")
    Path(f"{stem}.notes").write_text(notes)


# ----------------------------------------------------------------- worker mode
def worker(tree: Path, out: Path, shard: int, nshards: int) -> int:
    from aiter.aot.flydsl.fmha_fp8 import compile_one_config

    jobs, _ = enumerate_jobs(tree)
    tools = (llvm_tool("llvm-objdump"), llvm_tool("llvm-readelf"))
    art = out / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    failed = 0
    for j in jobs[shard::nshards]:
        key = j["kernel_name"]
        cache = out / "cache" / key
        os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(cache)  # fresh per job
        res = compile_one_config(**j)
        if res["compile_time"] is None:
            (art / f"{key}.FAILED").write_text("compile failed\n")
            failed += 1
            continue
        blobs = code_objects(cache)
        if not blobs:
            (art / f"{key}.FAILED").write_text("no code object found\n")
            failed += 1
            continue
        for i, b in enumerate(blobs):
            dump_one(b, art / f"{key}__{i}", tools)
        # The cache dir is only a staging area; keep the output dir small.
        import shutil

        shutil.rmtree(cache, ignore_errors=True)
    return failed


def worker_one(tree: Path, out: Path, key: str) -> int:
    """Compile one job serially into <out>/<key>__i.{s,notes}."""
    from aiter.aot.flydsl.fmha_fp8 import compile_one_config

    job = next(j for j in enumerate_jobs(tree)[0] if j["kernel_name"] == key)
    tools = (llvm_tool("llvm-objdump"), llvm_tool("llvm-readelf"))
    out.mkdir(parents=True, exist_ok=True)
    cache = out / f"cache_{key}"
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(cache)
    res = compile_one_config(**job)
    blobs = code_objects(cache) if res["compile_time"] is not None else []
    import shutil

    shutil.rmtree(cache, ignore_errors=True)
    for i, b in enumerate(blobs):
        dump_one(b, out / f"{key}__{i}", tools)
    return 0 if blobs else 1


def _variant(d: Path, key: str) -> bytes:
    return b"\0".join(
        p.read_bytes()
        for p in sorted(d.glob(f"{key}__*"))
        if p.suffix in (".s", ".notes")
    )


def reverify(a: Path, b: Path, srcs: tuple[Path, Path], diffs: list[str]) -> int:
    """Serially recompile each differing job from both trees; return #DIFFERENT."""
    root = b / "reverify"
    run_dirs = (a / "artifacts", b / "artifacts")
    bad = 0
    for key in diffs:
        variants: dict[str, list[bytes]] = {"A": [], "B": []}
        for side, src in zip("AB", srcs):
            env = dict(
                os.environ,
                PYTHONPATH=str(src),
                FLYDSL_GPU_ARCH=ARCH,
                COMPILE_ONLY="1",
                ENABLE_CK="0",
            )
            for n in range(REVERIFY_N):
                d = root / key / f"{side}{n}"
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--_one",
                        str(src.resolve()),
                        str(d),
                        key,
                    ],
                    env=env,
                    cwd="/tmp",
                    check=True,
                )
                variants[side].append(_variant(d, key))
        distinct = {s: len(set(v)) for s, v in variants.items()}
        # A job passes only if some serial A variant byte-matches some serial B
        # variant; flakiness on a side never excuses a missing match.
        if not set(variants["A"]) & set(variants["B"]):
            cls = "DIFFERENT"
            bad += 1
        elif any(n > 1 for n in distinct.values()):
            cls = "FLAKY-BASE"
        else:
            cls = "IDENTICAL-AFTER-REVERIFY"
        # Which variant each serial compile produced, relative to the parallel runs.
        seen = [_variant(rd, key) for rd in run_dirs]
        tags = []
        for side in "AB":
            for n, v in enumerate(variants[side]):
                m = [f"run{i + 1}" for i, sv in enumerate(seen) if sv == v]
                tags.append(f"{side}{n}={'+'.join(m) or 'new'}")
        print(
            f"{cls} {key}  distinct A={distinct['A']} B={distinct['B']}  {' '.join(tags)}"
        )
    return bad


# ------------------------------------------------------------------ mode 1 / 2
def compile_tree(tree: Path, out: Path, nproc: int) -> int:
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(tree),
        FLYDSL_GPU_ARCH=ARCH,
        COMPILE_ONLY="1",
        ENABLE_CK="0",
    )
    # Listing in a subprocess too, so the parent never imports the tree.
    t0 = time.time()
    procs = [
        subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--_worker",
                str(tree),
                str(out),
                str(i),
                str(nproc),
            ],
            env=env,
            cwd="/tmp",
        )
        for i in range(nproc)
    ]
    rc = max(p.wait() for p in procs)
    # Skipped-shape manifest (same enumeration, cheap).
    lst = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--_skipped", str(tree)],
        env=env,
        cwd="/tmp",
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (out / "skipped_shapes.txt").write_text(lst)
    n = len(
        {
            p.name.split("__")[0].removesuffix(".FAILED")
            for p in (out / "artifacts").iterdir()
        }
    )
    print(
        f"compiled {tree} -> {out}: {n} jobs in {time.time() - t0:.0f}s (worker rc={rc})"
    )
    return rc


def compare(a: Path, b: Path, srcs: tuple[Path, Path] | None = None) -> int:
    def files(d: Path) -> dict[str, Path]:
        return {p.name: p for p in (d / "artifacts").iterdir()}

    fa, fb = files(a), files(b)
    jobs = lambda fs: {n.split("__")[0].removesuffix(".FAILED") for n in fs}
    ja, jb = jobs(fa), jobs(fb)
    bad = 0
    missing = 0
    diffs = []
    for job in sorted(ja | jb):
        names = sorted(
            n
            for n in set(fa) | set(fb)
            if n.split("__")[0].removesuffix(".FAILED") == job
        )
        if job not in ja or job not in jb:
            print(f"MISSING {job} (only in {'B' if job not in ja else 'A'})")
            bad += 1
            missing += 1
            continue
        same = all(
            n in fa and n in fb and fa[n].read_bytes() == fb[n].read_bytes()
            for n in names
        )
        failed = any(n.endswith(".FAILED") for n in names)
        print(
            f"{'identical' if same else 'DIFFERENT'} {job}"
            + (" (compile FAILED on both)" if failed and same else "")
        )
        if not same:
            diffs.append(job)
        bad += (not same) or failed
    print(
        f"jobs A={len(ja)} B={len(jb)}; identical={len(ja & jb) - bad}; problems={bad}"
    )
    if srcs is None:
        return 1 if bad or ja != jb else 0
    # Compile-failed jobs still count against the gate; only DIFFERENT-by-ISA
    # jobs are eligible for re-verification.
    final = missing + (bad - missing - len(diffs)) + reverify(a, b, srcs, diffs)
    print(f"after reverify: failing={final}")
    return 1 if final or ja != jb else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", help="source tree root to compile")
    ap.add_argument("--out", help="output directory for --src")
    ap.add_argument("--compare", nargs=2, metavar=("DIR_A", "DIR_B"))
    ap.add_argument(
        "--reverify",
        nargs=2,
        metavar=("SRC_A", "SRC_B"),
        help="with --compare: serially recompile differing jobs from these trees",
    )
    ap.add_argument(
        "--_one", nargs=3, metavar=("TREE", "OUT", "KEY"), help=argparse.SUPPRESS
    )
    ap.add_argument("--jobs", type=int, default=32, help="parallel worker processes")
    ap.add_argument(
        "--_worker", nargs=4, metavar=("TREE", "OUT", "I", "N"), help=argparse.SUPPRESS
    )
    ap.add_argument("--_skipped", metavar="TREE", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a._worker:
        tree, out, i, n = a._worker
        return 1 if worker(Path(tree), Path(out), int(i), int(n)) else 0
    if a._one:
        return worker_one(Path(a._one[0]), Path(a._one[1]), a._one[2])
    if a._skipped:
        _, sk = enumerate_jobs(Path(a._skipped))
        print("\n".join(map(str, sk)))
        return 0
    if a.compare:
        srcs = tuple(Path(x).resolve() for x in a.reverify) if a.reverify else None
        return compare(Path(a.compare[0]), Path(a.compare[1]), srcs)
    if a.src and a.out:
        return compile_tree(Path(a.src).resolve(), Path(a.out), a.jobs)
    ap.error("give --src and --out, or --compare")


if __name__ == "__main__":
    sys.exit(main())
