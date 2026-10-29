#!/usr/bin/env python
"""Select aiter op_tests to run for a PR, from its git diff.

.github/workflows/aiter-test.yaml runs this as a dry run (--dry-run) while its
DRY_RUN switch is 'true': the selection is printed, and CI runs the full suite
it ran before test selection existed. The ci:aiter-full PR label runs the full
suite (--all). Any error falls back to the full suite.

Unlike the Triton selector, an import graph alone cannot carry this: aiter's
__init__.py re-exports ~60 modules with `import *`, so every test that merely
does `import aiter` would depend on all of aiter/ops. Four chains run instead,
and a test is selected when any of them reaches it:

  A  import graph   -- for tests that import a specific module. The bare
                       `import aiter` edge is dropped, or the graph collapses.
  B  ownership map  -- .github/aiter-test-map.yaml, for tests that only import
                       bare aiter. Hand-maintained; a bare-import test missing
                       from it is a hard error, not a silent miss.
  C  csrc chain     -- csrc file -> optCompilerConfig.json srcs (plus #include
                       closure) -> module_<x> -> @compile_ops("module_<x>") ->
                       aiter/ops/*.py -> chain A or B.
  D  hsa map        -- hsa/<arch>/<family>/ -> tests, from the same YAML. ASM
                       kernels appear in no import graph.

Anything a chain cannot explain raises, and the full suite runs.
"""

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from functools import cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

TESTS_DIR = "op_tests/"
MAP_FILE = ".github/aiter-test-map.yaml"
JIT_CONFIG = "aiter/jit/optCompilerConfig.json"
CSRC = "csrc/"
CONFIGS = "aiter/configs/"
HSA = "hsa/"

# split_tests.sh --test-type aiter collects exactly these. The selection must be
# a subset: it rejects any path it did not collect, so the two must agree.
EXTRA_TUNING_TESTS = (
    "op_tests/tuning_tests/test_csv_validation.py",
    "op_tests/tuning_tests/test_config_shape_collision.py",
    "op_tests/tuning_tests/test_fast_tuning.py",
    "op_tests/tuning_tests/test_fhmoe_tuner.py",
    "op_tests/tuning_tests/test_mixed_mxfp_tuning.py",
)

# A change here can affect anything: run the full suite.
GLOBAL_PREFIXES = (
    ".github/",
    "aiter/jit/",
    "3rdparty/",
    "aiter/test_common",
    "aiter/utility/",
)
GLOBAL_FILES = {
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "pytest.ini",
    "conftest.py",
    "MANIFEST.in",
}

# Changes that select nothing on their own. The op_tests subdirectories here
# hold suites this workflow never runs: split_tests.sh --test-type aiter
# collects op_tests/test_*.py plus EXTRA_TUNING_TESTS and nothing else.
IGNORED_SUFFIXES = (".md",)
IGNORED_PREFIXES = (
    "docs/",
    ".claude/",
    "op_tests/windows/",
    "op_tests/triton_tests/",
    "op_tests/op_benchmarks/",
    "op_tests/multigpu_tests/",
    "op_tests/flydsl_tests/",
    "op_tests/cpp/",
    "op_tests/opus/",
    "op_tests/tuners/",
)
IGNORED_FILES = {"LICENSE", ".gitignore"}

# Triton and Gluon kernels have their own workflow and their own selector, but
# some op_tests import them, so a change here still selects those tests -- and
# only those. A .py resolves through the import graph; a tuned config does not
# name its consumers, so it takes every test that reaches into the Triton tree.
TRITON_SRC = "aiter/ops/triton/"

# Headers are resolved against these, longest-prefix first, like the build does.
INCLUDE_DIRS = (
    "csrc/include",
    "csrc",
    "csrc/kernels",
    "csrc/py_itfs_ck",
    "csrc/cpp_itfs",
)


def list_tests():
    """Every test split_tests.sh would shard, as it would name it."""
    top = sorted(
        p.relative_to(ROOT).as_posix() for p in (ROOT / TESTS_DIR).glob("test_*.py")
    )
    extra = [t for t in EXTRA_TUNING_TESTS if (ROOT / t).is_file()]
    return sorted(set(top) | set(extra))


def source_files():
    """aiter Python sources reachable from this suite's tests.

    The Triton tree is included: it has its own workflow, but op_tests here
    import from it, and dropping it would make those edges invisible.
    """
    return [p.relative_to(ROOT).as_posix() for p in sorted(ROOT.glob("aiter/**/*.py"))]


@cache
def triton_dependent_tests():
    """Tests that reach into aiter/ops/triton, however deep the chain."""
    importers = build_importers(frozenset())
    triton = [p for p in source_files() if p.startswith(TRITON_SRC)]
    return set(list_tests()) & dependents(triton, importers)


# --------------------------------------------------------------------------
# Chain A: import graph
# --------------------------------------------------------------------------


@cache
def resolve_module(dotted, gone):
    """Resolve a dotted name to a repo file, retaining edges to deleted files.

    `aiter` itself resolves to nothing: aiter/__init__.py re-exports ~60 modules
    with `import *`, so honouring that edge makes every importer depend on all
    of aiter/ops and the selection collapses to the full suite.
    """
    if dotted == "aiter" or not dotted.startswith("aiter"):
        return None
    rel = dotted.replace(".", "/")
    for cand in (rel + ".py", rel + "/__init__.py"):
        if cand in gone or (ROOT / cand).is_file():
            return cand
    return None


def scan_imports(path, gone):
    """Local aiter imports of one file."""
    found = set()
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8", errors="ignore"))
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
            # An imported name can itself be a submodule.
            names = [f"{module}.{a.name}" for a in node.names] + [module]
        else:
            continue
        found.update(filter(None, (resolve_module(n, gone) for n in names)))
    return found


def build_importers(gone):
    """dependency -> files that import it."""
    importers = defaultdict(set)
    for path in source_files() + list_tests():
        if not (ROOT / path).is_file():
            continue
        for dep in scan_imports(path, gone):
            importers[dep].add(path)
    return importers


def dependents(paths, importers):
    seen = set(paths)
    frontier = list(seen)
    while frontier:
        for path in importers.get(frontier.pop(), ()):
            if path not in seen:
                seen.add(path)
                frontier.append(path)
    return seen


# --------------------------------------------------------------------------
# Chains B and D: the hand-maintained map
# --------------------------------------------------------------------------


@cache
def load_map():
    data = yaml.safe_load((ROOT / MAP_FILE).read_text(encoding="utf-8")) or {}
    return (
        data.get("tests") or {},
        data.get("hsa") or {},
        data.get("tuned_configs") or {},
    )


def bare_import_tests():
    """Tests whose only aiter import is the bare package (chain A can't place them)."""
    generic = {
        "aiter",
        "aiter.test_common",
        "aiter.test_mha_common",
        "aiter.jit",
        "aiter.jit.core",
        "aiter.jit.utils",
        "aiter.jit.utils.chip_info",
        "aiter.utility",
        "aiter.utility.dtypes",
        "aiter.utility.mx_types",
        "aiter.utility.fp4_utils",
        "aiter.benchmark_data_init",
        "aiter.benchmark_reporting",
    }
    out = set()
    for test in list_tests():
        text = (ROOT / test).read_text(encoding="utf-8", errors="ignore")
        mods = set(
            re.findall(r"^\s*(?:from|import)\s+(aiter[A-Za-z0-9_.]*)", text, re.M)
        )
        if not mods - generic:
            out.add(test)
    return out


# --------------------------------------------------------------------------
# Chain C: csrc -> JIT module -> aiter/ops wrapper
# --------------------------------------------------------------------------


@cache
def resolve_includes(path):
    """Repo-relative headers `path` includes directly."""
    found = set()
    full = ROOT / path
    if not full.is_file():
        return frozenset()
    text = full.read_text(encoding="utf-8", errors="ignore")
    for match in re.finditer(r'#\s*include\s+"([^"]+)"', text):
        target = match.group(1)
        for base in (str(Path(path).parent),) + INCLUDE_DIRS:
            cand = ROOT / base / target
            if cand.is_file():
                found.add(cand.resolve().relative_to(ROOT.resolve()).as_posix())
                break
    return frozenset(found)


@cache
def csrc_to_modules():
    """csrc file -> JIT modules that compile it, following #include edges."""
    config = json.loads((ROOT / JIT_CONFIG).read_text(encoding="utf-8"))
    owners = defaultdict(set)
    for module, cfg in config.items():
        declared = set()
        for src in cfg.get("srcs", []):
            match = re.search(r"\{AITER_CSRC_DIR\}/([^'\"]+)", src)
            if match:
                declared.add(CSRC + match.group(1))
        # A module owns its declared sources and everything they include.
        seen, frontier = set(declared), list(declared)
        while frontier:
            for dep in resolve_includes(frontier.pop()):
                if dep not in seen:
                    seen.add(dep)
                    frontier.append(dep)
        for path in seen:
            owners[path].add(module)
    return dict(owners)


@cache
def hsa_to_modules():
    """hsa/<arch>/<family>/ -> JIT modules, from each module's blob_gen_cmd.

    A module that compiles ASM kernels names its family on the codegen command
    line ("hsa/codegen.py -m mla"), which is the same declaration the build
    itself relies on. 24 of the 27 families on disk are covered; the rest fall
    back to the map's `hsa:` section.
    """
    config = json.loads((ROOT / JIT_CONFIG).read_text(encoding="utf-8"))
    owners = defaultdict(set)
    for module, cfg in config.items():
        for family in re.findall(
            r"codegen\.py\s+-m\s+([a-z0-9_]+)", str(cfg.get("blob_gen_cmd", ""))
        ):
            owners[family].add(module)
    return dict(owners)


@cache
def module_to_wrappers():
    """JIT module name -> the aiter Python files that drive it.

    @compile_ops("module_x") is the common declaration, but the ASM modules are
    loaded by name instead -- MD_NAME = "module_mla_asm", or a
    build_ctypes_module("module_mla_asm") call -- so any file quoting the name
    counts as a consumer.
    """
    names = set(json.loads((ROOT / JIT_CONFIG).read_text(encoding="utf-8")))
    owners = defaultdict(set)
    for path in source_files():
        text = (ROOT / path).read_text(encoding="utf-8", errors="ignore")
        for module in re.findall(r'["\']([A-Za-z0-9_]+)["\']', text):
            if module in names:
                owners[module].add(path)
    return dict(owners)


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


def changed_files(merge_ref):
    """Compare the checked-out PR merge to its base parent."""
    cmd = [
        "git",
        "diff",
        "--name-only",
        "--no-renames",
        "-z",
        merge_ref + "^1",
        merge_ref,
    ]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
    return [path for path in out.stdout.split("\0") if path]


def select(diff):
    """Affected tests, or raise when a subset cannot be determined."""
    diff = [
        p
        for p in diff
        if not p.endswith(IGNORED_SUFFIXES)
        and not p.startswith(IGNORED_PREFIXES)
        and p not in IGNORED_FILES
    ]
    if not diff:
        return [], ["only ignored files changed"]

    for path in diff:
        if path.startswith(GLOBAL_PREFIXES) or path in GLOBAL_FILES:
            raise RuntimeError(f"{path}: shared machinery changed")

    tests = set(list_tests())
    owns, hsa_map, config_map = load_map()
    bare = bare_import_tests()

    missing = sorted(bare - set(owns))
    if missing:
        raise RuntimeError(
            f"{MAP_FILE} is missing bare-import tests: {', '.join(missing)}"
        )

    gone = frozenset(p for p in diff if p.endswith(".py") and not (ROOT / p).is_file())
    importers = build_importers(gone)

    # Chain B is a reverse index: source path -> tests that declare it.
    owned_by = defaultdict(set)
    for test, entry in owns.items():
        for path in (entry or {}).get("owns") or []:
            owned_by[path].add(test)

    def reached(paths):
        """Tests chains A and B reach from these source paths."""
        hits = tests & dependents(paths, importers)
        for path in paths:
            hits |= owned_by.get(path, set())
        return hits

    selected = set()
    reasons = []
    for path in diff:
        if path.startswith(TESTS_DIR):
            if path in tests:
                hits, why = {path} | reached([path]), "changed test and importers"
            else:
                hits, why = reached([path]), "changed test helper and importers"
                if not hits:
                    raise RuntimeError(f"{path}: test helper maps to no tests")
        elif path.startswith(CSRC):
            modules = csrc_to_modules().get(path)
            if not modules:
                raise RuntimeError(f"{path}: not reachable from any JIT module's srcs")
            wrappers = set()
            for module in modules:
                wrappers |= module_to_wrappers().get(module, set())
            if not wrappers:
                raise RuntimeError(
                    f"{path}: modules {sorted(modules)} have no @compile_ops wrapper"
                )
            hits = reached(wrappers)
            why = f"JIT modules {sorted(modules)} and their tests"
        elif path.startswith(HSA):
            parts = path[len(HSA) :].split("/")
            family = parts[1] if len(parts) > 1 else None
            modules = hsa_to_modules().get(family, set())
            wrappers = set()
            for module in modules:
                wrappers |= module_to_wrappers().get(module, set())
            hits = reached(wrappers) if wrappers else set()
            why = f"hsa family '{family}' via {sorted(modules)}"
            if not hits:
                # No codegen declaration, or none of its modules has a wrapper.
                hits = set(hsa_map.get(family) or [])
                why = f"hsa family '{family}' from the map"
            if not hits:
                raise RuntimeError(f"{path}: hsa family '{family}' is not mapped")
        elif path.startswith(TRITON_SRC):
            # A .py resolves through the import graph; a tuned config names no
            # consumer, and a kernel no op_test imports directly is still
            # reachable at runtime, so both take the whole Triton surface.
            hits = reached([path]) if path.endswith(".py") else set()
            why = "importing tests"
            if not hits:
                hits = triton_dependent_tests()
                why = "every test that reaches the Triton tree"
            if not hits:
                raise RuntimeError(f"{path}: no op_test reaches the Triton tree")
        elif path.startswith(CONFIGS) and not path.endswith(".py"):
            hits = set()
            for pattern, names in config_map.items():
                if re.search(pattern, Path(path).name):
                    hits |= set(names)
            if not hits:
                raise RuntimeError(f"{path}: no tuned-config rule matches")
            why = "tuned config rule"
        elif path.endswith(".py"):
            hits = reached([path])
            why = "importing tests and ownership map"
            if not hits:
                raise RuntimeError(f"{path}: maps to no tests")
        else:
            raise RuntimeError(f"{path}: unclassified change")

        selected |= hits
        reasons.append(f"{path}: {why} ({len(hits)} tests)")

    if not selected:
        raise RuntimeError("relevant files changed but no tests were selected")
    return sorted(selected), reasons


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--merge-ref",
        help="PR merge ref; diff is taken against its first parent. Without it "
        "the full suite runs",
    )
    ap.add_argument("--all", action="store_true", help="select the full suite")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the selection, but run what CI ran before test selection",
    )
    ap.add_argument("--output", default="selected_aiter_tests.list")
    args = ap.parse_args()

    everything = list_tests()
    tests, reasons = everything, ["full suite requested"]
    try:
        if args.merge_ref and not args.all:
            tests, reasons = select(changed_files(args.merge_ref))
    except Exception as why:  # noqa: BLE001 -- any failure falls open to a full run
        tests, reasons = everything, [f"FULL SUITE: {why}"]

    report = f"Aiter tests: {len(tests)} of {len(everything)} files\n"
    report += "".join(f"- {reason}\n" for reason in reasons)
    if tests and len(tests) < len(everything):
        report += "".join(f"  {test}\n" for test in tests)
    if args.dry_run:
        # Print the selection, but keep the tests CI ran before selection.
        report = "DRY RUN, selection not applied.\n" + report
        tests = everything
        report += f"Running {len(tests)} files, as CI did before test selection.\n"

    Path(args.output).write_text("".join(t + "\n" for t in tests), encoding="utf-8")
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a", encoding="utf-8") as fh:
            fh.write(f"selected_count={len(tests)}\n")
    print(report, file=sys.stderr)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(report + "\n")


if __name__ == "__main__":
    main()
