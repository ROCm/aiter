#!/usr/bin/env python
"""Select Triton unit tests to run for a PR, from its git diff."""

import argparse
import ast
import os
import subprocess
import sys
from functools import cache
from pathlib import Path

SRC = "aiter/ops/triton/"
KERNELS = SRC + "_triton_kernels/"
GLUON_KERNELS = SRC + "_gluon_kernels/"
CONFIGS = SRC + "configs/"
TESTS = "op_tests/triton_tests/"
BENCH = "op_tests/op_benchmarks/triton/"
ROOT = Path(__file__).resolve().parents[2]
IMPORT_ROOTS = ("aiter.ops.triton", "op_tests.triton_tests")

# A change here can affect anything: run the full suite.
GLOBAL_PREFIXES = (
    ".github/",
    SRC + "utils/",
    KERNELS + "common/",
    TESTS + "utils/",
)

# Directories under the source tree that are not op categories.
NON_CATEGORY_DIRS = {"utils", "configs", "_triton_kernels", "_gluon_kernels"}

# Config categories whose source/test folders use a different layout.
CONFIG_CATEGORIES = {
    "attention": {"attention", "chunk_delta_attn"},
    "mhc": {"fusions"},
}


def basename(path):
    return path.rsplit("/", 1)[-1]


def stem(path):
    return basename(path).rsplit(".", 1)[0]


def subjects(test_path):
    """Source names covered by test_<op>.py or test_compile_<op>.py."""
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


@cache
def resolve_module(dotted, gone=frozenset()):
    """Resolve local modules, retaining edges to deleted files."""
    if not any(
        dotted == root or dotted.startswith(root + ".") for root in IMPORT_ROOTS
    ):
        return None
    rel = dotted.replace(".", "/")
    for cand in (rel + ".py", rel + "/__init__.py"):
        if cand in gone or (ROOT / cand).is_file():
            return cand
    return None


def scan_imports(path, gone=frozenset(), dynamic=None):
    """Collect local imports and flag loaders with unknown runtime targets."""
    found = set()
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    loaders = {"import_module", "__import__", "spec_from_file_location"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            loaders.update(a.asname or a.name for a in node.names if a.name in loaders)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                package = path.split("/")[:-1]
                if node.level > len(package):
                    raise RuntimeError(f"{path}: relative import escapes its package")
                package = package[: len(package) - node.level + 1]
                module = ".".join(package + ([module] if module else []))
            # Imported names can themselves be submodules.
            names = [module] + [f"{module}.{a.name}" for a in node.names]
        elif isinstance(node, ast.Call):
            name = getattr(node.func, "id", getattr(node.func, "attr", None))
            if dynamic is not None and name in loaders:
                dynamic.add(path)
            continue
        else:
            continue
        found.update(filter(None, (resolve_module(n, gone) for n in names)))
    return found


def reachable(start, imports):
    """Transitive closure of `start`'s imports inside the triton tree."""
    seen = set()
    frontier = [start]
    while frontier:
        for dep in imports.get(frontier.pop(), ()):
            if dep not in seen:
                seen.add(dep)
                frontier.append(dep)
    return seen


def changed_files(args):
    # Compare the checked-out PR merge to its base parent; retain both rename paths.
    refs = [args.merge_ref + "^1", args.merge_ref]
    cmd = ["git", "diff", "--name-only", "--no-renames", "-z", *refs]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
    return [path for path in out.stdout.split("\0") if path]


def select(diff):
    """Select affected test files, raising when a subset cannot be determined."""
    tests = list_files(TESTS, "test_*.py")
    sources = list_files(SRC, "*.py")
    gone = frozenset(
        f
        for f in diff
        if f.endswith(".py") and f.startswith((SRC, TESTS)) and not (ROOT / f).is_file()
    )
    dynamic = set()
    imports = {
        f: scan_imports(f, gone, dynamic) for f in sources + list_files(TESTS, "*.py")
    }
    # Name-only tests also inherit their wrapper's transitive dependencies.
    by_name = {}
    for source in set(sources) | {f for f in gone if f.startswith(SRC)}:
        by_name.setdefault(stem(source), set()).add(source)
    for test in tests:
        for name in subjects(test):
            imports[test].update(by_name.get(name, ()))
    test_reach = {test: reachable(test, imports) for test in tests}

    def reached_by(path):
        return {test for test in tests if path in test_reach[test]}

    def folder_tests(path):
        category = category_of(path)
        hits = {test for test in tests if category and category_of(test) == category}
        if not hits:
            raise RuntimeError(f"{path}: no mapped tests or op test folder")
        return hits

    selected = set()
    reasons = []
    relevant = False
    for path in diff:
        if path.endswith(".md") or basename(path) == ".gitkeep":
            continue
        if path.startswith(GLOBAL_PREFIXES):
            raise RuntimeError(f"{path}: shared machinery/CI changed")
        if not path.startswith((SRC, TESTS)):
            if path.startswith(BENCH):
                continue
            # A mixed PR may also change dependencies outside the import graph.
            if path.startswith(("aiter/", "op_tests/", "requirements")) or path in {
                "setup.py",
                "setup.cfg",
                "pyproject.toml",
                "pytest.ini",
                "conftest.py",
            }:
                raise RuntimeError(f"{path}: shared code outside the import graph")
            continue
        relevant = True
        if basename(path) == "__init__.py":
            raise RuntimeError(f"{path}: package initializer changed")

        if path.startswith(CONFIGS):
            # configs/<arch>/<backend>/<op>/<family>/<file>.json
            parts = path[len(CONFIGS) :].split("/")
            if not (
                path.endswith(".json")
                and len(parts) >= 5
                and parts[1] in ("triton", "gluon")
            ):
                raise RuntimeError(f"{path}: unknown config layout")
            op, family = parts[2:4]
            categories = CONFIG_CATEGORIES.get(op, {op})
            hits = {test for test in tests if category_of(test) in categories}
            if not hits and not any(
                category_of(source) in categories or stem(source) == op
                for source in sources
            ):
                raise RuntimeError(f"{path}: unknown config op '{op}'")
            # Family names need not match filenames; include the whole op and dependents.
            consumers = {
                source
                for source in sources
                if category_of(source) in categories or stem(source) in (op, family)
            }
            hits.update(test for test in tests if test_reach[test] & consumers)
            if not hits:
                raise RuntimeError(f"{path}: config maps to no tests")
            reason = f"'{op}' folder and dependents"
        elif path.startswith(TESTS):
            hits = reached_by(path)
            if basename(path).startswith("test_") and path.endswith(".py"):
                if path not in gone:
                    hits.add(path)
                reason = "changed test and importers"
            else:
                hits.update(folder_tests(path))
                reason = "test helper folder and importers"
        else:
            if not path.endswith(".py"):
                raise RuntimeError(f"{path}: non-Python source changed")
            hits = reached_by(path)
            reason = "paired and importing tests"
            if not hits:
                hits = folder_tests(path)
                reason = "unmapped source; whole op folder"
        selected.update(hits)
        reasons.append(f"{path}: {reason} ({len(hits)} tests)")

    if relevant and not selected:
        raise RuntimeError("relevant files changed but no tests were selected")
    if selected:
        # Static imports cannot rule out impact on unmapped or dynamic tests.
        extra = {
            test
            for test in tests
            if not any(path.startswith(SRC) for path in test_reach[test])
            or test in dynamic
            or test_reach[test] & dynamic
        } - selected
        selected.update(extra)
        if extra:
            reasons.append("unmapped/dynamic tests: " + ", ".join(sorted(extra)))
    return sorted(selected), reasons


def write_outputs(tests, reasons, is_full, output):
    Path(output).write_text("".join(t + "\n" for t in tests), encoding="utf-8")
    mode = "full suite" if is_full else "selected"
    report = f"Triton tests: {len(tests)} files ({mode})\n"
    report += "".join(f"- {reason}\n" for reason in reasons)
    print(report, file=sys.stderr)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(report + "\n")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--merge-ref", help="PR merge ref; diff is taken against its first parent"
    )
    mode.add_argument("--all", action="store_true", help="select the full suite")
    ap.add_argument("--output", default="selected_triton_tests.list")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.all:
        write_outputs(
            list_files(TESTS, "test_*.py"), ["full suite requested"], True, args.output
        )
        return
    try:
        diff = changed_files(args)
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
