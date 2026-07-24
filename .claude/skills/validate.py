#!/usr/bin/env python3
"""
Validate .claude/skills/*/SKILL.md files.

Layer 1 (format): every SKILL.md has a well-formed YAML front matter with at
                  least `name` and `description`, and `name` matches its
                  directory.

Layer 2 (facts):  every AITER-repo path referenced in backticks (inline code)
                  actually exists in the checkout.

Usage:
    python .claude/skills/validate.py            # run all checks
    python .claude/skills/validate.py --strict   # also treat warnings as errors

Exit code 0 on success, 1 otherwise.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from typing import Iterable

try:
    import yaml  # type: ignore
except ImportError:
    print(
        "ERROR: PyYAML is required. Install with: pip install pyyaml",
        file=sys.stderr,
    )
    sys.exit(2)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SKILLS_DIR = REPO_ROOT / ".claude" / "skills"

REQUIRED_META_KEYS = ("name", "description")
MIN_DESCRIPTION_LEN = 60

# Paths worth fact-checking live under these top-level AITER directories.
# Extend as the repo grows.
TRACKED_PREFIXES = (
    "aiter/",
    "csrc/",
    "op_tests/",
    "scripts/",
    ".github/",
    ".githooks/",
    "hsa_runtime/",
    "3rdparty/",
)

# Inline-code paths we want to verify. Matches `foo/bar.py`, `csrc/.../x.cpp`,
# etc. Stops at whitespace / backtick / closing punctuation.
PATH_RE = re.compile(r"`([A-Za-z0-9_./-]+)`")

# Characters that usually mean "placeholder, not a real path".
PLACEHOLDER_MARKERS = ("<", ">", "*", "?", "$", "{")

# Template tokens that appear in example paths in skills (e.g. `aiter/ops/my_op.py`).
# Any backtick path containing one of these is treated as illustrative and skipped.
TEMPLATE_TOKENS = (
    "my_op",
    "my_gemm",
    "my_add",
    "MY-OP",
    "module_xxx",
    "/category/",
    "_untuned_",
    "gemm_op_my_",
    "_triton_kernels/category",
    "triton_tests/category",
    "triton_tests//",
    "kernels//",
    "ops/.py",
    "test_.py",
)

# Runtime-only paths that don't exist until the user builds / runs something.
# We skip these rather than fail the check.
RUNTIME_PATHS = ("aiter/jit/build",)


def iter_skill_files() -> Iterable[pathlib.Path]:
    yield from sorted(SKILLS_DIR.rglob("SKILL.md"))


def check_front_matter(path: pathlib.Path) -> list[str]:
    errs: list[str] = []
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not m:
        errs.append("missing YAML front matter (--- ... ---)")
        return errs
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as exc:
        errs.append(f"YAML parse error: {exc}")
        return errs
    if not isinstance(meta, dict):
        errs.append("front matter is not a mapping")
        return errs
    for key in REQUIRED_META_KEYS:
        if not meta.get(key):
            errs.append(f"missing required key `{key}`")
    name = meta.get("name")
    if name and name != path.parent.name:
        errs.append(f"`name: {name}` != directory `{path.parent.name}`")
    desc = meta.get("description") or ""
    if len(desc.strip()) < MIN_DESCRIPTION_LEN:
        errs.append(
            f"`description` too short ({len(desc.strip())} < {MIN_DESCRIPTION_LEN}); "
            "it is what triggers the skill, keep it detailed"
        )
    return errs


def is_placeholder(p: str) -> bool:
    if any(mark in p for mark in PLACEHOLDER_MARKERS):
        return True
    if any(tok in p for tok in TEMPLATE_TOKENS):
        return True
    return False


def is_runtime_path(p: str) -> bool:
    return any(p.startswith(r) for r in RUNTIME_PATHS)


def check_paths(path: pathlib.Path) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Errors = nonexistent paths. Warnings = suspicious."""
    errors: list[str] = []
    warnings: list[str] = []
    text = path.read_text(encoding="utf-8")
    # Only verify paths inside fenced inline-code; ignore code blocks (```) to
    # avoid false positives on sample shell commands.
    text_wo_blocks = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    for m in PATH_RE.finditer(text_wo_blocks):
        raw = m.group(1).rstrip(".,):;")
        if is_placeholder(raw):
            continue
        if not raw.startswith(TRACKED_PREFIXES):
            continue
        if is_runtime_path(raw):
            continue
        # Strip any line-range suffixes like aiter/foo.py:12
        raw = raw.split(":", 1)[0]
        target = REPO_ROOT / raw
        if not target.exists():
            errors.append(f"missing path: `{raw}`")
    return errors, warnings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true", help="treat warnings as errors")
    args = ap.parse_args()

    skill_files = list(iter_skill_files())
    if not skill_files:
        print(f"No SKILL.md files under {SKILLS_DIR}", file=sys.stderr)
        return 1

    total_errors = 0
    total_warnings = 0

    for f in skill_files:
        rel = f.relative_to(REPO_ROOT)
        errs = check_front_matter(f)
        path_errs, path_warns = check_paths(f)
        errs.extend(path_errs)

        if errs or path_warns:
            print(f"\n=== {rel}")
        for e in errs:
            print(f"  [ERROR]   {e}")
            total_errors += 1
        for w in path_warns:
            print(f"  [WARNING] {w}")
            total_warnings += 1

    print()
    print(f"Checked {len(skill_files)} SKILL.md files")
    print(f"  errors   : {total_errors}")
    print(f"  warnings : {total_warnings}")

    if total_errors > 0:
        return 1
    if args.strict and total_warnings > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
