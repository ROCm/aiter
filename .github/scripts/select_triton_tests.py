#!/usr/bin/env python
"""Select Triton unit tests to run for a PR, from its git diff, by OP category.

Eager, category-level selection:

  R1  A change in category C — wrapper, ``_triton_kernels``/``_gluon_kernels``
      body, or config — selects the whole folder ``op_tests/triton_tests/C/``.
  R2  Test files OUTSIDE C whose imports reach a module of C (AST import BFS,
      depth-limited, through wrappers into kernel bodies) are selected too —
      the fused tests — individually. This is how "rmsnorm change also runs
      the MoE test that fuses rmsnorm" works, without a hand-written map.

  Fail-open: shared machinery (``utils/``, ``_triton_kernels/common/``,
  ``__init__.py`` files, CI infra), unmappable files, git/AST errors, or an
  empty selection while triton sources changed all select the FULL suite.
  Benchmark-only diffs intentionally select nothing (benchmarks are not UTs).
  Doc files (*.md) never contribute.

Output: newline-separated test file paths (repo-relative) written to
``--output``; a human-readable reason report goes to stderr and, when
``$GITHUB_STEP_SUMMARY`` is set, to the job summary. Always exits 0 —
failure modes degrade to selecting the full suite, never to selecting less.
"""

import argparse
import ast
import os
import re
import subprocess
import sys
from collections import deque
from pathlib import Path, PurePosixPath

SRC = PurePosixPath("aiter/ops/triton")
KERNELS = SRC / "_triton_kernels"
GLUON_KERNELS = SRC / "_gluon_kernels"
CONFIGS = SRC / "configs"
TESTS = PurePosixPath("op_tests/triton_tests")
BENCH = PurePosixPath("op_tests/op_benchmarks/triton")


def _find_root() -> Path:
    """Repo root: normally two levels above .github/scripts/, with a cwd
    fallback so the script also works when invoked from a repo checkout."""
    cand = Path(__file__).resolve().parent.parent.parent
    if (cand / TESTS).is_dir():
        return cand
    return Path.cwd()


ROOT = _find_root()

NON_CATEGORY_DIRS = {"utils", "configs", "_triton_kernels", "_gluon_kernels", "__pycache__"}

# Any change here invalidates everything: run the full suite.
GLOBAL_PREFIXES = (
    SRC / "utils",
    KERNELS / "common",
    TESTS / "utils",
)
GLOBAL_FILES = {
    PurePosixPath(".github/scripts/select_triton_tests.py"),
    PurePosixPath(".github/scripts/split_tests.sh"),
    PurePosixPath(".github/scripts/build_aiter_triton.sh"),
    PurePosixPath(".github/scripts/install_triton.sh"),
    PurePosixPath(".github/scripts/download_triton_wheel.sh"),
    PurePosixPath(".github/scripts/verify_triton_pin.py"),
    PurePosixPath(".github/requirements/triton-test.txt"),
    PurePosixPath(".github/workflows/triton-test.yaml"),
    PurePosixPath(".github/workflows/prepare-triton-wheel.yaml"),
}

CONFIG_LOADER_FUNCS = {"get_gemm_config", "get_tuned_kernel_config", "get_moe_configs"}
IMPORT_BFS_DEPTH = 3


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# Repo scanning
# ---------------------------------------------------------------------------


def discover_categories() -> set[str]:
    cats = set()
    for base in (ROOT / SRC, ROOT / KERNELS, ROOT / GLUON_KERNELS, ROOT / TESTS):
        if not base.is_dir():
            continue
        for p in base.iterdir():
            if p.is_dir() and p.name not in NON_CATEGORY_DIRS and p.name != "common" and not p.name.startswith("__"):
                cats.add(p.name)
    return cats


def list_test_files() -> list[PurePosixPath]:
    return sorted(
        PurePosixPath(p.relative_to(ROOT).as_posix())
        for p in (ROOT / TESTS).rglob("test_*.py")
    )


def list_src_files() -> list[PurePosixPath]:
    return sorted(
        PurePosixPath(p.relative_to(ROOT).as_posix())
        for p in (ROOT / SRC).rglob("*.py")
    )


def category_of(path: PurePosixPath, categories: set[str]) -> str | None:
    """Category folder a source/test file belongs to, or None for loose files."""
    for base in (KERNELS, GLUON_KERNELS, SRC, TESTS):
        try:
            rel = path.relative_to(base)
        except ValueError:
            continue
        if len(rel.parts) >= 2 and rel.parts[0] in categories:
            return rel.parts[0]
        return None
    return None


# ---------------------------------------------------------------------------
# AST scan: imports + config references per module
# ---------------------------------------------------------------------------


class ModuleInfo:
    __slots__ = ("imports", "config_names")

    def __init__(self):
        self.imports: set[PurePosixPath] = set()
        self.config_names: set[str] = set()


def _resolve_module(dotted: str) -> list[PurePosixPath]:
    """aiter.ops.triton.x.y -> existing repo file(s) for that module."""
    if not dotted.startswith("aiter.ops.triton"):
        return []
    rel = PurePosixPath(*dotted.split("."))
    out = []
    for cand in (rel.with_suffix(".py"), rel / "__init__.py"):
        if (ROOT / cand).is_file():
            out.append(cand)
            break
    return out


def scan_module(path: PurePosixPath) -> ModuleInfo | None:
    info = ModuleInfo()
    try:
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                info.imports.update(_resolve_module(alias.name))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            hit = _resolve_module(node.module)
            info.imports.update(hit)
            # `from aiter.ops.triton.moe import moe_op_gemm_a8w4` style: the
            # names may themselves be submodules.
            for alias in node.names:
                info.imports.update(_resolve_module(f"{node.module}.{alias.name}"))
        elif isinstance(node, ast.Call):
            func = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if func in CONFIG_LOADER_FUNCS:
                for arg in list(node.args) + [kw.value for kw in node.keywords]:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        info.config_names.add(arg.value)
    return info


def build_module_infos(files: list[PurePosixPath]) -> dict[PurePosixPath, ModuleInfo]:
    infos = {}
    for f in files:
        info = scan_module(f)
        if info is None:
            raise RuntimeError(f"failed to parse {f}")
        infos[f] = info
    return infos


def reachable_from(
    start: PurePosixPath,
    infos: dict[PurePosixPath, ModuleInfo],
    depth: int = IMPORT_BFS_DEPTH,
) -> set[PurePosixPath]:
    seen: set[PurePosixPath] = set()
    frontier = deque([(start, 0)])
    while frontier:
        cur, d = frontier.popleft()
        if d >= depth:
            continue
        for dep in infos.get(cur, ModuleInfo()).imports:
            if dep not in seen:
                seen.add(dep)
                frontier.append((dep, d + 1))
    return seen


# ---------------------------------------------------------------------------
# Config-file attribution
# ---------------------------------------------------------------------------


def config_categories_and_owners(
    cfg: PurePosixPath,
    categories: set[str],
    src_infos: dict[PurePosixPath, ModuleInfo],
) -> tuple[set[str], set[PurePosixPath]] | None:
    """Attribute a nested-layout config file to categories/owner modules.

    Only the nested ``<arch>/<backend>/<op>/<d_type>/`` layout is supported;
    returns None for anything else (legacy flat layout) so the caller can
    fail open to the full suite.
    """
    rel = cfg.relative_to(CONFIGS)
    parts = rel.parts
    if len(parts) < 4 or parts[1] not in ("triton", "gluon"):
        return None  # legacy layout — not supported

    cats: set[str] = set()
    if parts[2] in categories:
        cats.add(parts[2])

    owners: set[PurePosixPath] = set()
    stem = parts[-1]
    for mod, info in src_infos.items():
        # Generic loaders under utils/ read every config file — only
        # kernel/wrapper modules count as owners.
        if is_under(mod, SRC / "utils"):
            continue
        for name in info.config_names:
            if name.lower().replace("-", "_") in parts or stem.startswith(name):
                owners.add(mod)
                break
    return cats, owners


# ---------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------


def changed_files(args: argparse.Namespace) -> list[PurePosixPath]:
    if args.merge_ref:
        cmd = ["git", "diff", "--name-only", f"{args.merge_ref}^1", args.merge_ref]
    else:
        cmd = ["git", "diff", "--name-only", f"{args.target}...{args.source}"]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
    return [PurePosixPath(line) for line in out.stdout.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def is_under(path: PurePosixPath, prefix: PurePosixPath) -> bool:
    try:
        path.relative_to(prefix)
        return True
    except ValueError:
        return False


def select(diff: list[PurePosixPath]) -> tuple[list[PurePosixPath], list[str], bool]:
    """Returns (selected test files, reasons, is_full_suite)."""
    categories = discover_categories()
    all_tests = list_test_files()
    src_files = list_src_files()
    infos = build_module_infos(src_files + all_tests)

    test_reach = {t: reachable_from(t, infos) for t in all_tests}
    test_cats: dict[PurePosixPath, set[str]] = {}
    for t in all_tests:
        cats = set()
        own = category_of(t, categories)
        if own:
            cats.add(own)
        for dep in test_reach[t]:
            c = category_of(dep, categories)
            if c:
                cats.add(c)
        test_cats[t] = cats

    reasons: list[str] = []
    selected: set[PurePosixPath] = set()
    selected_cats: set[str] = set()
    loose_sources: set[PurePosixPath] = set()
    saw_relevant_change = False

    def full(reason: str) -> tuple[list[PurePosixPath], list[str], bool]:
        reasons.append(f"FULL SUITE: {reason}")
        return all_tests, reasons, True

    for f in diff:
        if f.suffix == ".md":
            reasons.append(f"{f}: documentation — no tests selected for it")
            continue
        if f in GLOBAL_FILES or any(is_under(f, p) for p in GLOBAL_PREFIXES):
            return full(f"{f} is shared machinery/CI infra")
        if is_under(f, BENCH):
            reasons.append(f"{f}: benchmark change — no unit tests selected for it")
            continue
        if is_under(f, TESTS):
            saw_relevant_change = True
            if f.name.startswith("test_") and f.suffix == ".py":
                selected.add(f)
                reasons.append(f"{f}: changed test file — runs itself")
            else:
                cat = category_of(f, categories)
                if cat:
                    selected_cats.add(cat)
                    reasons.append(f"{f}: test helper — runs whole '{cat}' folder")
                else:
                    return full(f"{f} is a shared test helper outside any category")
            continue
        if is_under(f, CONFIGS):
            if f.name == ".gitkeep":
                continue
            saw_relevant_change = True
            if f.suffix != ".json":
                return full(f"{f}: non-JSON change under configs/")
            attributed = config_categories_and_owners(f, categories, infos)
            if attributed is None:
                return full(
                    f"{f}: legacy config layout — only the nested "
                    "<arch>/<backend>/<op>/ layout is supported by selection"
                )
            cats, owners = attributed
            for mod in owners:
                c = category_of(mod, categories)
                if c:
                    cats.add(c)
                else:
                    loose_sources.add(mod)
            if not cats and not owners:
                return full(f"{f}: config file not attributable to any kernel")
            selected_cats.update(cats)
            reasons.append(
                f"{f}: config — categories {sorted(cats) or '[]'}"
                + (f", owner modules {sorted(str(o) for o in owners)}" if owners else "")
            )
            continue
        if is_under(f, SRC):
            saw_relevant_change = True
            if f.name == "__init__.py":
                return full(f"{f}: package __init__ changed")
            if f.suffix != ".py":
                return full(f"{f}: non-Python file under triton sources")
            cat = category_of(f, categories)
            if cat:
                selected_cats.add(cat)
                reasons.append(f"{f}: source in category '{cat}'")
            else:
                loose_sources.add(f)
                reasons.append(f"{f}: loose module — selecting importers")
            continue
        # Outside triton scope (other aiter/, csrc/, ...): covered by other CI.

    # R1: whole test folder per selected category.
    for cat in sorted(selected_cats):
        folder = [t for t in all_tests if is_under(t, TESTS / cat)]
        selected.update(folder)
    # R2: fused tests elsewhere whose category-set intersects.
    for t in all_tests:
        if test_cats[t] & selected_cats:
            selected.add(t)
    # Loose modules: exact import reachability.
    for src in sorted(loose_sources):
        importers = [t for t in all_tests if src in test_reach[t]]
        if not importers:
            return full(f"{src}: loose module not reached by any test")
        selected.update(importers)

    if not selected and saw_relevant_change:
        return full("relevant files changed but nothing was selected")
    return sorted(selected), reasons, False


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_outputs(
    tests: list[PurePosixPath], reasons: list[str], is_full: bool, output: str
) -> None:
    Path(output).write_text("".join(f"{t}\n" for t in tests), encoding="utf-8")
    header = (
        f"Triton test selection: FULL SUITE ({len(tests)} files)"
        if is_full
        else f"Triton test selection: {len(tests)} test file(s)"
    )
    log(header)
    for r in reasons:
        log(f"  - {r}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(f"### {header}\n\n")
            for r in reasons:
                fh.write(f"- {r}\n")
            if not is_full:
                fh.write("\n<details><summary>Selected tests</summary>\n\n")
                for t in tests:
                    fh.write(f"- `{t}`\n")
                fh.write("\n</details>\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--merge-ref", help="PR merge ref; diff is taken against its first parent")
    mode.add_argument("--source", help="source ref (with --target)")
    mode.add_argument("--all", action="store_true", help="select the full suite")
    ap.add_argument("--target", help="target ref for --source mode")
    ap.add_argument("--output", default="selected_triton_tests.list")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.source and not args.target:
        print("--source requires --target", file=sys.stderr)
        sys.exit(2)
    if args.all:
        write_outputs(list_test_files(), ["full suite requested"], True, args.output)
        return
    try:
        diff = changed_files(args)
        log(f"Changed files ({len(diff)}):")
        for f in diff:
            log(f"  {f}")
        tests, reasons, is_full = select(diff)
    except Exception as exc:  # noqa: BLE001 — fail open, never fail closed
        tests, reasons, is_full = (
            list_test_files(),
            [f"FULL SUITE: selection error ({type(exc).__name__}: {exc})"],
            True,
        )
    write_outputs(tests, reasons, is_full, args.output)


if __name__ == "__main__":
    main()
