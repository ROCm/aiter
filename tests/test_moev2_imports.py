# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""aiter.moev2 must not depend on the legacy MoE modules.

The legacy modules may import from moev2, never the reverse, and `import aiter` must not
load moev2.

NOTE: part of the aiter.moev2 migration (under active development, not graduated).
Please coordinate changes with the moev2 owners (@qli88). See aiter/moev2/README.md.

Golden snapshots: after an intended change to legacy MoE dispatch, run
`python tests/moe_dispatch/snapshot.py --update` and paste the changes it prints into
the PR.
"""

import ast
import json
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
MOEV2 = REPO / "aiter" / "moev2"
LEGACY = ("aiter.fused_moe", "aiter.fhmoe", "aiter.fused_moe_bf16_asm")


def _module_name(path):
    rel = path.relative_to(REPO).with_suffix("")
    parts = rel.parts[:-1] if rel.name == "__init__" else rel.parts
    return ".".join(parts)


def _imported_modules(path):
    """Absolute names of every module a file imports, including lazy and importlib imports."""
    tree = ast.parse(path.read_text(), filename=str(path))
    module = _module_name(path)
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[: len(base) - node.level + 1]
                parent = ".".join(base + ([node.module] if node.module else []))
            else:
                parent = node.module
            yield node.lineno, parent
            for alias in node.names:
                yield node.lineno, f"{parent}.{alias.name}"
        elif isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            if name in ("import_module", "__import__") and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    yield node.lineno, arg.value


def _is_legacy(name):
    return any(name == m or name.startswith(m + ".") for m in LEGACY)


def test_moev2_does_not_import_legacy_statically():
    bad = []
    for path in sorted(MOEV2.rglob("*.py")):
        for lineno, name in _imported_modules(path):
            if _is_legacy(name):
                bad.append(f"{path.relative_to(REPO)}:{lineno}: imports {name}")
    assert not bad, "aiter.moev2 must not import legacy MoE modules:\n" + "\n".join(bad)


def test_aiter_init_does_not_import_moev2():
    init = REPO / "aiter" / "__init__.py"
    bad = [
        f"aiter/__init__.py:{lineno}: imports {name}"
        for lineno, name in _imported_modules(init)
        if name == "aiter.moev2" or name.startswith("aiter.moev2.")
    ]
    assert not bad, "\n".join(bad)


_RUNTIME_CHECK = r"""
import importlib, json, pkgutil, sys
import aiter
before = "aiter.moev2" in sys.modules
import aiter.moev2 as pkg
names = [pkg.__name__] + [m.name for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + ".")]
for name in names:
    importlib.import_module(name)
print(json.dumps({"moev2_by_aiter": before, "modules": names, "loaded": sorted(sys.modules)}))
"""


def test_moev2_imports_on_cpu_without_legacy():
    env = dict(os.environ)
    env.update(
        HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="-1"
    )
    env.setdefault("AITER_LOG_LEVEL", "ERROR")
    proc = subprocess.run(
        [sys.executable, "-c", _RUNTIME_CHECK],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert not result["moev2_by_aiter"], "`import aiter` must not import aiter.moev2"
    loaded = [m for m in result["loaded"] if _is_legacy(m)]
    assert not loaded, f"importing aiter.moev2 loaded legacy modules: {loaded}"


_LEGACY_ENTRY_CHECK = "import aiter.fused_moe\nprint('ok')\n"


def test_legacy_fused_moe_imports_with_moev2():
    """`import aiter.fused_moe` must keep working now that it imports aiter.moev2.

    fused_moe.py pulls the moved helpers back out of aiter.moev2 at module level, so one
    unimportable moev2 module breaks `import aiter.fused_moe` for every downstream MoE
    consumer (vLLM, SGLang, ATOM). `import aiter` does not import fused_moe, and the
    moev2 runtime check above does not import the legacy module, so nothing else here
    covers that import graph.
    """
    env = dict(os.environ)
    env.setdefault("AITER_LOG_LEVEL", "ERROR")
    proc = subprocess.run(
        [sys.executable, "-c", _LEGACY_ENTRY_CHECK],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
