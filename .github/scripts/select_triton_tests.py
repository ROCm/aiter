#!/usr/bin/env python
"""Select Triton unit tests to run for a PR, from its git diff."""

import argparse
import ast
import os
import subprocess
import sys
from collections import Counter
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
CI_FILES = {
    ".github/scripts/build_aiter_triton.sh",
    ".github/scripts/download_triton_wheel.sh",
    ".github/scripts/install_triton.sh",
    ".github/scripts/select_triton_tests.py",
    ".github/scripts/split_tests.sh",
    ".github/scripts/verify_triton_pin.py",
    ".github/scripts/ensure_zstd.sh",
    ".github/requirements/triton-test.txt",
    ".github/workflows/triton-test.yaml",
    ".github/workflows/prepare-triton-wheel.yaml",
    ".github/workflows/ci-config.yaml",
}

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
def compatibility_modules():
    """Read legacy module aliases without importing GPU code."""
    path = ROOT / (SRC + "__init__.py")
    if path.is_file():
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "_BACKWARD_COMPAT_MAP"
                for target in node.targets
            ):
                return {
                    f"aiter.ops.triton.{old}": f"aiter.ops.triton.{new}"
                    for old, new in ast.literal_eval(node.value).items()
                }
    return {}


@cache
def resolve_module(dotted, gone):
    """Resolve local modules, retaining edges to deleted files."""
    if not any(
        dotted == root or dotted.startswith(root + ".") for root in IMPORT_ROOTS
    ):
        return None
    rel = compatibility_modules().get(dotted, dotted).replace(".", "/")
    for cand in (rel + ".py", rel + "/__init__.py"):
        if cand in gone or (ROOT / cand).is_file():
            return cand
    return None


def scan_imports(path, gone, dynamic):
    """Collect local imports and flag loaders with unknown runtime targets."""
    found = set()
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    # Resolve constants such as PUBLIC_WRAPPER or a local legacy import path.
    writes = Counter(
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    )
    writes.update(node.arg for node in ast.walk(tree) if isinstance(node, ast.arg))
    constants = {
        target.id: node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        for target in node.targets
        if isinstance(target, ast.Name) and writes[target.id] == 1
    }
    loaders = {
        name: name
        for name in ("import_module", "__import__", "spec_from_file_location")
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            loaders.update(
                (a.asname or a.name, loaders[a.name])
                for a in node.names
                if a.name in loaders
            )
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
            names = [f"{module}.{a.name}" for a in node.names]
            if not all(name in compatibility_modules() for name in names):
                names.append(module)
        elif isinstance(node, ast.Call):
            name = getattr(node.func, "id", getattr(node.func, "attr", None))
            if name not in loaders:
                continue
            arg = node.args[0] if node.args else None
            target = (
                arg.value
                if isinstance(arg, ast.Constant)
                else constants.get(arg.id) if isinstance(arg, ast.Name) else None
            )
            if (
                loaders[name] == "spec_from_file_location"
                or not isinstance(target, str)
                or target.startswith(".")
                or (
                    target.startswith(IMPORT_ROOTS)
                    and resolve_module(target, gone) is None
                )
            ):
                dynamic.add(path)
                continue
            names = [target]
        else:
            continue
        found.update(filter(None, (resolve_module(n, gone) for n in names)))
    return found


def dependents(paths, importers):
    """Follow imports backwards from changed files to their consumers."""
    seen = set(paths)
    frontier = list(seen)
    while frontier:
        for path in importers.get(frontier.pop(), ()):
            if path not in seen:
                seen.add(path)
                frontier.append(path)
    return seen


def changed_files(merge_ref):
    # Compare the checked-out PR merge to its base parent; retain both rename paths.
    refs = [merge_ref + "^1", merge_ref]
    cmd = ["git", "diff", "--name-only", "--no-renames", "-z", *refs]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
    return [path for path in out.stdout.split("\0") if path]


def select(diff):
    """Select affected test files, raising when a subset cannot be determined."""
    diff = [p for p in diff if not p.endswith(".md") and Path(p).name != ".gitkeep"]
    if not any(
        p.startswith((SRC, TESTS, "aiter/aot/triton/", "aiter/utility/triton/"))
        or p in CI_FILES
        for p in diff
    ):
        return [], ["no Triton/Gluon changes"]
    for path in diff:
        if path.startswith(GLOBAL_PREFIXES):
            raise RuntimeError(f"{path}: shared machinery/CI changed")
        if not path.startswith((SRC, TESTS, BENCH)) and (
            path.startswith(("aiter/", "op_tests/", "requirements"))
            or path
            in {"setup.py", "setup.cfg", "pyproject.toml", "pytest.ini", "conftest.py"}
        ):
            raise RuntimeError(f"{path}: shared code outside the import graph")
    diff = [p for p in diff if p.startswith((SRC, TESTS))]

    tests = set(list_files(TESTS, "test_*.py"))
    sources = list_files(SRC, "*.py")
    gone = frozenset(
        f
        for f in diff
        if f.endswith(".py") and f.startswith((SRC, TESTS)) and not (ROOT / f).is_file()
    )
    all_sources = set(sources) | {f for f in gone if f.startswith(SRC)}
    by_name = {}
    for source in all_sources:
        by_name.setdefault(Path(source).stem, set()).add(source)

    dynamic, importers = set(), {}
    for path in sources + list_files(TESTS, "*.py"):
        dependencies = scan_imports(path, gone, dynamic)
        if path in tests:
            # Some tests select their wrapper by name instead of importing it.
            subject = Path(path).stem.removeprefix("test_")
            for name in (subject, subject.removeprefix("compile_")):
                dependencies.update(by_name.get(name, ()))
        for dependency in dependencies:
            importers.setdefault(dependency, set()).add(path)

    def affected(paths):
        return tests & dependents(paths, importers)

    def folder_tests(path):
        category = category_of(path)
        hits = {test for test in tests if category and category_of(test) == category}
        if not hits:
            raise RuntimeError(f"{path}: no mapped tests or op test folder")
        return hits

    selected = set()
    reasons = []
    for path in diff:
        if Path(path).name == "__init__.py":
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
                category_of(source) in categories or Path(source).stem == op
                for source in sources
            ):
                raise RuntimeError(f"{path}: unknown config op '{op}'")
            # Family names need not match filenames; include the whole op and dependents.
            consumers = {
                source
                for source in sources
                if category_of(source) in categories
                or Path(source).stem in (op, family)
            }
            hits.update(affected(consumers))
            if not hits:
                raise RuntimeError(f"{path}: config maps to no tests")
            reason = f"'{op}' folder and dependents"
        elif path.startswith(TESTS):
            hits = affected([path])
            if Path(path).name.startswith("test_") and path.endswith(".py"):
                reason = "changed test and importers"
            else:
                hits.update(folder_tests(path))
                reason = "test helper folder and importers"
        else:
            if not path.endswith(".py"):
                raise RuntimeError(f"{path}: non-Python source changed")
            hits = affected([path])
            reason = "paired and importing tests"
            if not hits:
                hits = folder_tests(path)
                reason = "unmapped source; whole op folder"
        selected.update(hits)
        reasons.append(f"{path}: {reason} ({len(hits)} tests)")

    if not selected:
        raise RuntimeError("relevant files changed but no tests were selected")
    # Only unresolved runtime imports need conservative extra coverage.
    extra = affected(dynamic) - selected
    selected.update(extra)
    if extra:
        reasons.append("unresolved dynamic imports: " + ", ".join(sorted(extra)))
    return sorted(selected), reasons


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--merge-ref", help="PR merge ref; diff is taken against its first parent"
    )
    mode.add_argument("--all", action="store_true", help="select the full suite")
    ap.add_argument("--output", default="selected_triton_tests.list")
    args = ap.parse_args()
    needs_mi300x = True
    try:
        if args.all:
            tests, reasons = list_files(TESTS, "test_*.py"), ["full suite requested"]
        else:
            diff = changed_files(args.merge_ref)
            needs_mi300x = any(
                path.startswith((CONFIGS + "gfx942/", GLUON_KERNELS + "gfx942/"))
                for path in diff
            )
            tests, reasons = select(diff)
    except Exception as why:  # noqa: BLE001 -- any failure falls open to a full run
        tests, reasons = list_files(TESTS, "test_*.py"), [f"FULL SUITE: {why}"]
    Path(args.output).write_text("".join(t + "\n" for t in tests), encoding="utf-8")
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a", encoding="utf-8") as fh:
            fh.write(
                f"selected_count={len(tests)}\nneeds_mi300x={str(needs_mi300x).lower()}\n"
            )
    report = f"Triton tests: {len(tests)} files\n"
    report += "".join(f"- {reason}\n" for reason in reasons)
    print(report, file=sys.stderr)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(report + "\n")


if __name__ == "__main__":
    main()
