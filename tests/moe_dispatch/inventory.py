# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Inventory of every name used from the legacy MoE modules, and every monkeypatch.

    python tests/moe_dispatch/inventory.py --vllm /path/to/vllm > fused_moe_public_names.txt

Static (AST) scan, so imports inside functions count. For each tracked module it lists
the names other files use, who uses them (aiter package, aiter tests and tuners, vLLM
library, vLLM tests), and which tests monkeypatch them. The classification is a
starting point for review:

- public:    used by the vLLM library;
- internal:  used only by other modules of the aiter package;
- test-only: used only by tests, benchmarks or tuners (aiter or vLLM).

NOTE: part of the aiter.moev2 migration (under active development, not graduated).
Please coordinate changes with the moev2 owners (@qli88). See aiter/moev2/README.md.

Golden snapshots: after an intended change to legacy MoE dispatch, run
`python tests/moe_dispatch/snapshot.py --update` and paste the changes it prints into
the PR.
"""

import argparse
import ast
import collections
import pathlib
import sys

TRACKED = (
    "aiter.fused_moe",
    "aiter.fhmoe",
    "aiter.fhmoe_contract",
    "aiter.fused_moe_bf16_asm",
    "aiter.fused_moe_dp_shared_expert",
    "aiter.fused_moe_registry",
    "aiter.ops.comm_fused_moe_runtime",
    "aiter.ops.flydsl.comm_fused_moe_host",
    "aiter.ops.flydsl.mega_moe_tp",
    "aiter.ops.flydsl.kernels.mega_moe",
    "aiter.ops.flydsl.kernels.mega_moe.mega_moe_v2",
    "aiter.ops.flydsl.kernels.mega_moe.mega_moe_config",
    "aiter.ops.flydsl.kernels.mega_moe_gfx1250",
    "aiter.ops.flydsl.kernels.mega_moe_gfx1250.mega_moe",
)

_PATCHERS = {"setattr", "patch", "object", "delattr"}


def _module_of_file(path, root):
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


class _Scanner(ast.NodeVisitor):
    def __init__(self, module_name, is_package):
        self.module_name = module_name
        self.package = module_name if is_package else module_name.rpartition(".")[0]
        self.aliases = {}  # local name -> tracked module
        self.uses = collections.defaultdict(set)  # tracked module -> names
        self.patches = collections.defaultdict(set)

    def _resolve_from(self, node):
        if not node.level:
            return node.module or ""
        base = self.package.split(".")
        if node.level > 1:
            base = base[: -(node.level - 1)]
        return ".".join(base + ([node.module] if node.module else []))

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name in TRACKED and alias.asname:
                self.aliases[alias.asname] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        source = self._resolve_from(node)
        for alias in node.names:
            full = f"{source}.{alias.name}"
            if full in TRACKED:
                self.aliases[alias.asname or alias.name] = full
            elif source in TRACKED and alias.name != "*":
                self.uses[source].add(alias.name)
        self.generic_visit(node)

    def _dotted(self, node):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
            return list(reversed(parts))
        return None

    def _tracked_prefix(self, parts):
        """(module, next name) for a dotted access into a tracked module."""
        if parts[0] in self.aliases:
            module, rest = self.aliases[parts[0]], parts[1:]
            while rest and f"{module}.{rest[0]}" in TRACKED:
                module, rest = f"{module}.{rest[0]}", rest[1:]
            return (module, rest[0]) if rest else None
        for i in range(len(parts) - 1, 0, -1):
            module = ".".join(parts[:i])
            if module in TRACKED:
                return module, parts[i]
        return None

    def visit_Attribute(self, node):
        parts = self._dotted(node)
        if parts and len(parts) > 1:
            hit = self._tracked_prefix(parts)
            if hit:
                self.uses[hit[0]].add(hit[1])
                return
        self.generic_visit(node)

    def visit_Assign(self, node):
        for target in node.targets:
            parts = self._dotted(target) if isinstance(target, ast.Attribute) else None
            if parts:
                hit = self._tracked_prefix(parts)
                if hit:
                    self.patches[hit[0]].add(hit[1])
        self.generic_visit(node)

    def visit_Call(self, node):
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name in _PATCHERS and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                parts = first.value.split(".")
                hit = self._tracked_prefix(parts) if len(parts) > 1 else None
                if hit:
                    self.patches[hit[0]].add(hit[1])
            elif (
                len(node.args) > 1
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                parts = self._dotted(first)
                module = None
                if parts:
                    if len(parts) == 1 and parts[0] in self.aliases:
                        module = self.aliases[parts[0]]
                    elif ".".join(parts) in TRACKED:
                        module = ".".join(parts)
                if module:
                    self.patches[module].add(node.args[1].value)
        self.generic_visit(node)


def _package_root(module):
    """The outermost tracked module that contains `module` (itself if none)."""
    return min(
        (t for t in TRACKED if module == t or module.startswith(t + ".")),
        key=len,
    )


def _inside(user, module):
    root = _package_root(module)
    return user == root or user.startswith(root + ".")


def _area(path, aiter_root, vllm_root):
    if vllm_root and path.is_relative_to(vllm_root):
        rel = path.relative_to(vllm_root)
        return "vllm-tests" if rel.parts[0] in ("tests", "benchmarks") else "vllm"
    rel = path.relative_to(aiter_root)
    return "aiter" if rel.parts[0] == "aiter" else "aiter-tests"


def scan(aiter_root, vllm_root=None):
    roots = [aiter_root / d for d in ("aiter", "op_tests", "tests", "csrc")]
    if vllm_root:
        roots += [vllm_root / d for d in ("vllm", "tests", "benchmarks")]
    uses = collections.defaultdict(lambda: collections.defaultdict(set))
    patches = collections.defaultdict(lambda: collections.defaultdict(set))
    for root in roots:
        if not root.exists():
            continue
        base = vllm_root if vllm_root and root.is_relative_to(vllm_root) else aiter_root
        for path in sorted(root.rglob("*.py")):
            if "moe_dispatch" in path.parts and path.name == "inventory.py":
                continue
            module_name = _module_of_file(path, base)
            try:
                tree = ast.parse(path.read_text(errors="replace"), filename=str(path))
            except SyntaxError:
                continue
            scanner = _Scanner(module_name, path.name == "__init__.py")
            scanner.visit(tree)
            label = str(path.relative_to(base.parent))
            for module, names in scanner.uses.items():
                if _inside(module_name, module):
                    continue
                for name in names:
                    uses[module][name].add((_area(path, aiter_root, vllm_root), label))
            for module, names in scanner.patches.items():
                for name in names:
                    patches[module][name].add(label)
    return uses, patches


def classify(users):
    areas = {area for area, _ in users}
    if "vllm" in areas:
        return "public"
    if "aiter" in areas:
        return "internal"
    return "test-only"


def report(uses, patches, out=sys.stdout):
    w = out.write
    w(__doc__.split("\n\n", 2)[2].rstrip() + "\n\n")
    w("Regenerate: python tests/moe_dispatch/inventory.py --vllm <vllm checkout>\n")
    for module in TRACKED:
        names = sorted(set(uses.get(module, {})) | set(patches.get(module, {})))
        if not names:
            continue
        w(f"\n== {module} ({len(names)} names)\n")
        for name in names:
            users = uses.get(module, {}).get(name, set())
            counts = collections.Counter(area for area, _ in users)
            count_text = " ".join(f"{a}:{counts[a]}" for a in sorted(counts))
            patched = sorted(patches.get(module, {}).get(name, set()))
            kind = classify(users) if users else "test-only"
            w(f"{name:<48} {kind:<10} {count_text}\n")
            for area, label in sorted(users):
                if area in ("vllm", "aiter"):
                    w(f"    {label}\n")
            for label in patched:
                w(f"    patched by {label}\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--vllm", type=pathlib.Path, help="vLLM checkout to scan as well"
    )
    args = parser.parse_args(argv)
    aiter_root = pathlib.Path(__file__).resolve().parents[2]
    vllm_root = args.vllm.resolve() if args.vllm else None
    report(*scan(aiter_root, vllm_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
