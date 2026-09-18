#!/usr/bin/env python
"""Select Triton unit tests to run for a PR, from its git diff."""

import argparse
import ast
import os
import subprocess
import sys
from collections import deque
from functools import cache
from pathlib import Path

SRC = "aiter/ops/triton/"
KERNELS = SRC + "_triton_kernels/"
GLUON_KERNELS = SRC + "_gluon_kernels/"
CONFIGS = SRC + "configs/"
TESTS = "op_tests/triton_tests/"
BENCH = "op_tests/op_benchmarks/triton/"

# A change here can affect anything: run the full suite.
GLOBAL_PREFIXES = (
    ".github/",
    SRC + "utils/",
    KERNELS + "common/",
    TESTS + "utils/",
)

# Directories under the source tree that are not op categories.
NON_CATEGORY_DIRS = {"utils", "configs", "_triton_kernels", "_gluon_kernels"}


def find_root():
    # Normally two levels above .github/scripts/; fall back to the current
    # directory so the script also works when run from a repo checkout.
    here = Path(__file__).resolve().parent.parent.parent
    return here if (here / TESTS).is_dir() else Path.cwd()


ROOT = find_root()


def log(msg):
    print(msg, file=sys.stderr)


def basename(path):
    return path.rsplit("/", 1)[-1]


def stem(path):
    return basename(path).rsplit(".", 1)[0]


def subjects(test_path):
    """What a test file is named after: `test_gemm_a16w16.py` -> gemm_a16w16.
    `torch_compile/test_compile_rmsnorm.py` also answers to `rmsnorm`, since
    those tests reach their op through a dynamic helper the import scan
    cannot see."""
    base = stem(test_path)[len("test_") :]
    found = {base}
    if base.startswith("compile_"):
        found.add(base[len("compile_") :])
    return found


def list_files(base, pattern):
    return sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / base).rglob(pattern))


def category_of(path):
    """Op category a source or test file belongs to, or None."""
    for base in (KERNELS, GLUON_KERNELS, SRC, TESTS):
        if not path.startswith(base):
            continue
        parts = path[len(base) :].split("/")
        # Gluon is a backend, not an op: _gluon_kernels/<arch>/<cat>/...
        if base == GLUON_KERNELS and parts[0].startswith("gfx"):
            parts = parts[1:]
        if len(parts) < 2 or parts[0] in NON_CATEGORY_DIRS:
            return None
        return parts[0]
    return None


# --- import graph -----------------------------------------------------------


# Roots the graph follows. Tests are in here as well as sources: the suite
# reuses reference implementations and input generators across test files, so
# a fused test often reaches the kernel it exercises only through another test.
#
# Invariant: a Triton test reaches the kernel it exercises through one of these
# roots, or is named after it. A test that gets there only through a module
# outside them (aiter.ops.shuffle, say) is invisible to the graph. If nothing
# under aiter/ops/triton is in its closure at all it lands in unmapped() and
# runs on every selection regardless; if something is, that kernel's changes
# will not select it. A sweep found no such test on 2026-09-18.
IMPORT_ROOTS = ("aiter.ops.triton", "op_tests.triton_tests")


@cache
def resolve_module(dotted):
    """aiter.ops.triton.x.y -> the repo file for that module, if it exists.
    Cached: the same names recur across hundreds of files and each miss costs
    two filesystem probes."""
    if not dotted.startswith(IMPORT_ROOTS):
        return None
    rel = dotted.replace(".", "/")
    for cand in (rel + ".py", rel + "/__init__.py"):
        if (ROOT / cand).is_file():
            return cand
    return None


def scan_imports(path):
    """The aiter.ops.triton modules `path` imports directly."""
    found = set()
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            # `from aiter.ops.triton.moe import moe_op_gemm_a8w4` — the
            # imported names may themselves be submodules.
            names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
        else:
            continue
        found.update(filter(None, map(resolve_module, names)))
    return found


def reachable(start, imports):
    """Transitive closure of `start`'s imports inside the triton tree."""
    seen = set()
    frontier = deque([start])
    while frontier:
        for dep in imports.get(frontier.popleft(), ()):
            if dep not in seen:
                seen.add(dep)
                frontier.append(dep)
    return seen


def changed_files(args):
    if args.merge_ref:
        # A PR merge ref: diff against its first parent (the base branch).
        cmd = ["git", "diff", "--name-only", args.merge_ref + "^1", args.merge_ref]
    else:
        cmd = ["git", "diff", "--name-only", f"{args.target}...{args.source}"]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
    return [line for line in out.stdout.splitlines() if line.strip()]


# --- selection --------------------------------------------------------------


def unmapped(tests, test_reach, sources):
    """Tests the selector cannot tie to any source: nothing under
    aiter/ops/triton in their import closure and no source named after them.
    Four torch_compile tests today, reaching their op through a dynamic helper.
    They run on every non-empty selection, since no diff can prove it did not
    touch them, and the list in the summary is how growth of this set gets
    noticed."""
    stems = {stem(s) for s in sources}
    return [
        t
        for t in tests
        if not any(p.startswith(SRC) for p in test_reach[t])
        and not (subjects(t) & stems)
    ]


def select(diff):
    """Map changed files to test files. Raises when a subset is not safe."""
    tests = list_files(TESTS, "test_*.py")
    sources = list_files(SRC, "*.py")
    # Helpers are parsed as well as test_*.py: a test can reach its kernel
    # through one, and a helper change has to find the tests behind it.
    imports = {f: scan_imports(f) for f in sources + list_files(TESTS, "*.py")}
    # Every module each test can reach, so a change anywhere in that set
    # selects the test -- this is what covers fused kernels without a map.
    test_reach = {t: reachable(t, imports) for t in tests}
    by_subject = {}
    for t in tests:
        for s in subjects(t):
            by_subject.setdefault(s, []).append(t)

    selected = set()
    reasons = []
    relevant = False

    def reached_by(f):
        """Tests whose import closure contains f."""
        return [t for t in tests if f in test_reach[t]]

    def folder_of(cat, changed):
        hits = [t for t in tests if t.startswith(TESTS + cat + "/")]
        if not hits:
            raise RuntimeError(f"{changed}: category '{cat}' has no tests")
        return hits

    def add_for_source(f):
        """Paired test by name, else the op-type folder, plus every test whose
        imports reach this module (the fused ones)."""
        paired = by_subject.get(stem(f), [])
        fused = [t for t in reached_by(f) if t not in paired]
        if paired:
            selected.update(paired)
            note = f"paired {len(paired)}"
        else:
            cat = category_of(f)
            if not cat:
                raise RuntimeError(f"{f}: no paired test and no category")
            selected.update(folder_of(cat, f))
            note = f"no paired test -> '{cat}' folder"
        selected.update(fused)
        reasons.append(f"{f}: {note}, fused {len(fused)}")

    for f in diff:
        if f.endswith(".md") or basename(f) == ".gitkeep":
            continue

        if f.startswith(GLOBAL_PREFIXES):
            raise RuntimeError(f"{f} is shared machinery/CI infra")

        if f.startswith(BENCH):
            reasons.append(f"{f}: benchmark — no unit tests selected")
            continue

        if f.startswith(TESTS):
            relevant = True
            if basename(f).startswith("test_") and f.endswith(".py"):
                if not (ROOT / f).is_file():
                    # Deleted, or renamed away. split_tests.sh refuses a
                    # selection naming a path that is not a test file, and the
                    # graph cannot find what imported a module that no longer
                    # exists -- so run the folder, which is where those
                    # importers live. A whole folder gone raises, to the full
                    # suite.
                    cat = category_of(f)
                    folder = folder_of(cat, f) if cat else []
                    selected.update(folder)
                    reasons.append(
                        f"{f}: deleted test — not run; '{cat}' folder ({len(folder)})"
                    )
                    continue
                importers = reached_by(f)
                selected.add(f)
                selected.update(importers)
                reasons.append(
                    f"{f}: changed test — runs itself + {len(importers)} importing"
                )
                continue
            cat = category_of(f)  # a test helper runs its whole folder
            if not cat:
                raise RuntimeError(f"{f} is a shared test helper")
            folder = set(folder_of(cat, f))
            outside = [t for t in reached_by(f) if t not in folder]
            selected.update(folder)
            selected.update(outside)
            reasons.append(
                f"{f}: test helper -> '{cat}' folder ({len(folder)})"
                f" + {len(outside)} importing test(s)"
            )
            continue

        if f.startswith(CONFIGS):
            relevant = True
            # Only the nested layout maps to an op:
            # configs/<arch>/<backend>/<op>/<d_type>/...
            parts = f[len(CONFIGS) :].split("/")
            if not (
                f.endswith(".json")
                and len(parts) >= 4
                and parts[1] in ("triton", "gluon")
            ):
                raise RuntimeError(f"{f}: config outside the nested layout")
            op, d_type = parts[2], parts[3]
            # A config is read by the wrapper at run time, never imported, so
            # the graph cannot say which variants consume it — a tuning
            # change for `attention/mha` also reaches test_mha_with_pe and
            # test_mha_with_sink. Run the whole op folder, plus any test
            # outside it that imports the module the family is named after.
            hits = {t for t in tests if t.startswith(TESTS + op + "/")}
            in_folder = len(hits)
            consumers = [s for s in sources if stem(s) == d_type]
            hits.update(t for t in tests if any(c in test_reach[t] for c in consumers))
            hits.update(by_subject.get(d_type, []))
            if not hits:
                raise RuntimeError(f"{f}: config maps to no tests")
            selected.update(hits)
            reasons.append(
                f"{f}: config -> '{op}' folder ({in_folder})"
                f" + {len(hits) - in_folder} importing test(s)"
            )
            continue

        if f.startswith(SRC):
            relevant = True
            if basename(f) == "__init__.py":
                raise RuntimeError(f"{f}: package __init__ changed")
            if not f.endswith(".py"):
                raise RuntimeError(f"{f}: non-Python file under triton sources")
            add_for_source(f)
            continue

        # Anything else (csrc/, other aiter/, ...) is covered by other CI jobs.

    if relevant and not selected:
        raise RuntimeError("relevant files changed but nothing was selected")
    if selected:
        extra = [t for t in unmapped(tests, test_reach, sources) if t not in selected]
        selected.update(extra)
        reasons.append(
            f"{len(extra)} test(s) no source maps to, run on every selection: "
            + ", ".join(basename(t) for t in extra)
        )
    return sorted(selected), reasons


# --- output -----------------------------------------------------------------


def write_outputs(tests, reasons, is_full, output):
    Path(output).write_text("".join(t + "\n" for t in tests), encoding="utf-8")
    if is_full:
        header = f"Triton test selection: FULL SUITE ({len(tests)} files)"
    else:
        header = f"Triton test selection: {len(tests)} of {len(list_files(TESTS, 'test_*.py'))} test files"
    log(header)
    for r in reasons:
        log(f"  - {r}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(f"### {header}\n\n")
            fh.writelines(f"- {r}\n" for r in reasons)
            if not is_full:
                fh.write("\n<details><summary>Selected tests</summary>\n\n")
                fh.writelines(f"- `{t}`\n" for t in tests)
                fh.write("\n</details>\n")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--merge-ref", help="PR merge ref; diff is taken against its first parent"
    )
    mode.add_argument("--source", help="source ref (with --target)")
    mode.add_argument("--all", action="store_true", help="select the full suite")
    ap.add_argument("--target", help="target ref for --source mode")
    ap.add_argument("--output", default="selected_triton_tests.list")
    args = ap.parse_args()
    if args.source and not args.target:
        ap.error("--source requires --target")
    return args


def main():
    args = parse_args()
    if args.all:
        write_outputs(
            list_files(TESTS, "test_*.py"), ["full suite requested"], True, args.output
        )
        return
    try:
        diff = changed_files(args)
        log(f"Changed files ({len(diff)}):")
        for f in diff:
            log(f"  {f}")
        tests, reasons = select(diff)
        is_full = False
    except Exception as why:  # noqa: BLE001 -- any failure falls open to a full run
        tests, reasons, is_full = (
            list_files(TESTS, "test_*.py"),
            [f"FULL SUITE: {why}"],
            True,
        )
    write_outputs(tests, reasons, is_full, args.output)


if __name__ == "__main__":
    main()
