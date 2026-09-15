#!/usr/bin/env python3
"""Select the tests affected by a set of changed files.

Builds a module-level import graph over ``aiter/`` and ``op_tests/``, reverses
it, and walks it transitively from the changed files to every test that can
reach them. Non-Python changes under ``aiter/ops/triton/configs/`` are mapped to
their op by path. Anything that cannot be mapped escalates to the full suite --
selection is only ever allowed to be too broad, never too narrow.

Emits a newline-separated list of test files, which ``split_tests.sh`` consumes
via ``--select-from`` and bin-packs into shards as usual.
"""

from __future__ import annotations

import argparse
import ast
import collections
import fnmatch
import json
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPLIT_TESTS_SH = os.path.join(REPO_ROOT, ".github", "scripts", "split_tests.sh")

# Roots scanned when building the import graph.
SOURCE_ROOTS = ("aiter", "op_tests")

# A change to any of these invalidates the whole graph (or the runner itself),
# so they escalate to the full suite rather than being resolved through imports.
ALWAYS_FULL = (
    "conftest.py",
    "*/conftest.py",
    "pyproject.toml",
    "setup.py",
    "requirements*.txt",
    ".github/scripts/*",
    ".github/workflows/*",
    "aiter/jit/*",
    "aiter/__init__.py",
    "aiter/utility/*",
    "3rdparty/*",
)

# Changes here are inert: they cannot affect test outcomes.
IGNORED = (
    "*.md",
    "docs/*",
    "*.txt.license",
    "LICENSE",
    ".gitignore",
)

# Directories whose files MUST resolve to a seed. An unmappable path under one of
# these means the graph has drifted from the tree, so we escalate.
MUST_RESOLVE = (
    "aiter/ops/",
    "op_tests/",
)

CONFIG_PREFIX = "aiter/ops/triton/configs/"
# Depth of a config JSON path, in "/"-separated components:
#   aiter/ops/triton/configs/<arch>/<backend>/<category>/<op_name>/<SHAPE>.json
CONFIG_DEPTH = 9

# aiter/__init__.py deliberately re-exports the Iris comms helpers, so these are
# expected. Any OTHER eager triton import from the package root would make every
# triton change select the whole suite, so we flag those.
ROOT_TRITON_ALLOWED = ("aiter.ops.triton.comms",)

TEST_SCOPES = {
    # Mirrors the discovery globs in split_tests.sh.
    "triton": lambda p: (
        p.startswith("op_tests/triton_tests/")
        and os.path.basename(p).startswith("test_")
    ),
    "aiter": lambda p: (
        os.path.dirname(p) == "op_tests" and os.path.basename(p).startswith("test_")
    ),
    "all": lambda p: (
        p.startswith("op_tests/") and os.path.basename(p).startswith("test_")
    ),
}


class ImportGraph:
    """Module-level import graph over the repo, plus its reverse."""

    def __init__(self, root: str = REPO_ROOT, roots=SOURCE_ROOTS):
        self.root = root
        self.mod2file: dict[str, str] = {}
        self.file2mod: dict[str, str] = {}
        self.forward: dict[str, set[str]] = collections.defaultdict(set)
        self.reverse: dict[str, set[str]] = collections.defaultdict(set)
        self._scan(roots)
        self._link()

    def _scan(self, roots) -> None:
        for source_root in roots:
            for dirpath, dirnames, filenames in os.walk(
                os.path.join(self.root, source_root)
            ):
                dirnames[:] = [d for d in dirnames if d != "__pycache__"]
                for name in filenames:
                    if not name.endswith(".py"):
                        continue
                    abs_path = os.path.join(dirpath, name)
                    rel = os.path.relpath(abs_path, self.root)
                    module = rel[:-3].replace(os.sep, ".")
                    module = module.removesuffix(".__init__")
                    self.mod2file[module] = rel
                    self.file2mod[rel] = module

    def _resolve(self, target: str, importer: str) -> str | None:
        """Longest-prefix match of a dotted name onto a known module."""
        candidate = target
        while candidate:
            if candidate in self.mod2file and candidate != importer:
                return candidate
            candidate = candidate.rsplit(".", 1)[0] if "." in candidate else ""
        return None

    def _link(self) -> None:
        for module, rel in self.mod2file.items():
            try:
                with open(
                    os.path.join(self.root, rel), encoding="utf-8", errors="ignore"
                ) as fh:
                    tree = ast.parse(fh.read())
            except (SyntaxError, ValueError):
                continue
            package = module.rsplit(".", 1)[0] if "." in module else module
            for node in ast.walk(tree):
                for target in self._targets(node, package):
                    hit = self._resolve(target, module)
                    if hit:
                        self.forward[module].add(hit)
        for importer, imported in self.forward.items():
            for dep in imported:
                self.reverse[dep].add(importer)

    @staticmethod
    def _targets(node: ast.AST, package: str) -> list[str]:
        if isinstance(node, ast.Import):
            return [alias.name for alias in node.names]
        if not isinstance(node, ast.ImportFrom):
            return []
        if node.level:
            parts = package.split(".")
            if node.level > 1:
                parts = parts[: len(parts) - (node.level - 1)]
            base = ".".join(parts + ([node.module] if node.module else []))
        elif node.module:
            base = node.module
        else:
            return []
        # `from a.b import c` may name a module OR a symbol; try both.
        return [base] + [f"{base}.{alias.name}" for alias in node.names]

    def dependents(self, seeds) -> set[str]:
        """All modules transitively importing any seed, including the seeds."""
        seen: set[str] = set()
        stack = list(seeds)
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self.reverse.get(current, ()))
        return seen


def load_file_times(path: str = SPLIT_TESTS_SH) -> dict[str, int]:
    """Parse FILE_TIMES out of split_tests.sh so timings have one source."""
    times: dict[str, int] = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for match in re.finditer(r"FILE_TIMES\[([^\]]+)\]=(\d+)", fh.read()):
                times[match.group(1)] = int(match.group(2))
    except OSError:
        pass
    return times


class Selector:
    def __init__(self, graph: ImportGraph, scope: str = "triton"):
        self.graph = graph
        self.scope = scope
        in_scope = TEST_SCOPES[scope]
        self.tests = {m for m, p in graph.mod2file.items() if in_scope(p)}
        self.all_test_modules = {
            m for m, p in graph.mod2file.items() if TEST_SCOPES["all"](p)
        }
        self.cross_scope: list[str] = []
        # Index op modules by the category directory they live under, so a config
        # dir that names a tuning variant rather than a module still maps somewhere.
        self.by_category: dict[str, set[str]] = collections.defaultdict(set)
        for module, rel in graph.mod2file.items():
            if not rel.startswith("aiter/ops/triton/"):
                continue
            for part in rel.split("/")[3:-1]:
                if not part.startswith("_") and part != "configs":
                    self.by_category[part].add(module)
        # Index op/test basenames so config dirs can be mapped by name.
        self.by_name: dict[str, set[str]] = collections.defaultdict(set)
        for module, rel in graph.mod2file.items():
            base = os.path.basename(rel)[:-3]
            if rel.startswith("aiter/ops/triton/"):
                self.by_name[base].add(module)
            if base.startswith("test_"):
                self.by_name[base[5:]].add(module)

    def map_path(self, path: str) -> tuple[set[str], str]:
        """Map one changed path to seed modules. Returns (seeds, reason)."""
        path = path.replace(os.sep, "/")
        if any(fnmatch.fnmatch(path, pat) for pat in IGNORED):
            return set(), "ignored"
        if any(fnmatch.fnmatch(path, pat) for pat in ALWAYS_FULL):
            return set(), "escalate:always-full"
        if path in self.graph.file2mod:
            return {self.graph.file2mod[path]}, "import-graph"
        if path.startswith(CONFIG_PREFIX):
            # configs/<arch>/<backend>/<category>/<op_name>/<SHAPE>.json
            parts = path.split("/")
            if len(parts) >= 6:
                # Preferred: the <op_name> dir matches an op module one-for-one.
                seeds = set(self.by_name.get(parts[-2], ()))
                if seeds:
                    return seeds, f"config-op:{parts[-2]}"
                # Fallback: many config dirs name a tuning variant (a4w4,
                # rmsnorm_large_m_small_n) rather than a module. Widen to the
                # <category> dir, which does map onto aiter/ops/triton/<category>/.
                category = parts[-3]
                seeds = set(self.by_category.get(category, ()))
                if seeds:
                    return seeds, f"config-category:{category}"
            return set(), f"escalate:unmapped-config:{path}"
        if any(path.startswith(prefix) for prefix in MUST_RESOLVE):
            if path.endswith(".py"):
                # Deleted or renamed away; nothing left to select for it.
                return set(), "deleted"
            return set(), f"escalate:unresolvable:{path}"
        return set(), "out-of-tree"

    def select(self, changed):
        """Returns (test_files, escalated, reasons)."""
        seeds: set[str] = set()
        reasons: dict[str, str] = {}
        escalated: list[str] = []
        for path in changed:
            mapped, reason = self.map_path(path)
            reasons[path] = reason
            if reason.startswith("escalate:"):
                escalated.append(f"{path} ({reason.split(':', 1)[1]})")
            seeds |= mapped
        if escalated:
            return self.all_tests(), escalated, reasons
        reachable = self.graph.dependents(seeds)
        hits = reachable & self.tests
        # Tests that the change affects but that this suite does not run. They are
        # another workflow's responsibility; surface them so nothing is lost.
        other = reachable & self.all_test_modules - self.tests
        self.cross_scope = sorted(self.graph.mod2file[m] for m in other)
        return sorted(self.graph.mod2file[m] for m in hits), [], reasons

    def all_tests(self) -> list[str]:
        return sorted(self.graph.mod2file[m] for m in self.tests)

    def check_invariants(self) -> list[str]:
        """Guard the assumptions that keep selection narrow."""
        warnings = []
        unexpected = sorted(
            dep
            for dep in self.graph.forward.get("aiter", set())
            if dep.startswith("aiter.ops.triton")
            and not dep.startswith(ROOT_TRITON_ALLOWED)
        )
        if unexpected:
            warnings.append(
                "aiter/__init__.py eagerly imports "
                + ", ".join(unexpected)
                + "; changes there will select every test that imports aiter. "
                "Make the import lazy or add it to ROOT_TRITON_ALLOWED."
            )
        depths = set()
        config_root = os.path.join(self.graph.root, CONFIG_PREFIX)
        for dirpath, _, filenames in os.walk(config_root):
            for name in filenames:
                if name.endswith(".json"):
                    rel = os.path.relpath(os.path.join(dirpath, name), self.graph.root)
                    depths.add(len(rel.split("/")))
        if depths - {CONFIG_DEPTH}:
            warnings.append(
                f"config layout depth changed (saw {sorted(depths)}, expected "
                f"{CONFIG_DEPTH}); the <op_name> component of the config rule has moved."
            )
        return warnings


def changed_files(base: str | None, from_file: str | None) -> list[str]:
    if from_file:
        if from_file == "-":
            return [line.strip() for line in sys.stdin if line.strip()]
        with open(from_file, encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip()]
    ref = base or "HEAD~1"
    merge_base = (
        subprocess.run(
            ["git", "merge-base", ref, "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=False,
        ).stdout.strip()
        or ref
    )
    out = subprocess.run(
        ["git", "diff", "--name-only", merge_base, "HEAD"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=True,
    ).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--test-type",
        default="triton",
        choices=sorted(TEST_SCOPES),
        help="which suite to select from (default: triton)",
    )
    ap.add_argument("--base", help="git ref to diff against (default: HEAD~1)")
    ap.add_argument(
        "--changed-from",
        metavar="FILE",
        help="read changed paths from FILE, or '-' for stdin",
    )
    ap.add_argument(
        "--output", "-o", help="write the selected list here (default: stdout)"
    )
    ap.add_argument(
        "--shadow",
        action="store_true",
        help="emit the FULL suite but report what would have been selected",
    )
    ap.add_argument(
        "--explain", action="store_true", help="log per-path mapping decisions"
    )
    ap.add_argument("--json", action="store_true", help="emit a JSON report on stderr")
    args = ap.parse_args()

    graph = ImportGraph()
    selector = Selector(graph, args.test_type)
    changed = changed_files(args.base, args.changed_from)
    selected, escalated, reasons = selector.select(changed)
    every = selector.all_tests()
    times = load_file_times()

    def cost(files):
        return sum(times.get(f, 15) for f in files)

    for warning in selector.check_invariants():
        print(f"::warning::select_tests: {warning}", file=sys.stderr)

    total_cost, sel_cost = cost(every), cost(selected)
    report = {
        "test_type": args.test_type,
        "changed_files": len(changed),
        "selected": len(selected),
        "total": len(every),
        "escalated": escalated,
        "est_seconds_selected": sel_cost,
        "est_seconds_full": total_cost,
        "shadow": args.shadow,
        "cross_scope": selector.cross_scope,
    }

    print(
        f"select_tests: {len(selected)}/{len(every)} {args.test_type} tests "
        f"(~{sel_cost}s of ~{total_cost}s, "
        f"{100 * (1 - sel_cost / total_cost) if total_cost else 0:.0f}% saved)",
        file=sys.stderr,
    )
    if selector.cross_scope:
        print(
            f"select_tests: {len(selector.cross_scope)} affected test(s) live outside "
            f"the '{args.test_type}' suite and are not run here:",
            file=sys.stderr,
        )
        for path in selector.cross_scope[:5]:
            print(f"  - {path}", file=sys.stderr)
    if escalated:
        print("select_tests: ESCALATED to full suite:", file=sys.stderr)
        for item in escalated[:10]:
            print(f"  - {item}", file=sys.stderr)
    if args.explain:
        for path, reason in sorted(reasons.items()):
            print(f"  {reason:<28} {path}", file=sys.stderr)
    if args.json:
        print(json.dumps(report, indent=2), file=sys.stderr)

    emitted = every if args.shadow else selected
    payload = "\n".join(emitted) + ("\n" if emitted else "")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(payload)
        if args.shadow:
            with open(args.output + ".selected", "w", encoding="utf-8") as fh:
                fh.write("\n".join(selected) + ("\n" if selected else ""))
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
