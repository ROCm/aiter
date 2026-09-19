# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Check quickstart syntax and direct AITER calls without importing GPU packages.

This is intentionally a source-contract check, not a runtime or dtype validator.
"""

import ast
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "docs/examples/quickstart.py"


def source_signature(function):
    args = function.args
    parameters = []
    positional = args.posonlyargs + args.args
    required = len(positional) - len(args.defaults)
    for index, arg in enumerate(positional):
        kind = (
            inspect.Parameter.POSITIONAL_ONLY
            if index < len(args.posonlyargs)
            else inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        default = inspect.Parameter.empty if index < required else None
        parameters.append(inspect.Parameter(arg.arg, kind, default=default))
    if args.vararg:
        parameters.append(
            inspect.Parameter(args.vararg.arg, inspect.Parameter.VAR_POSITIONAL)
        )
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        parameters.append(
            inspect.Parameter(
                arg.arg,
                inspect.Parameter.KEYWORD_ONLY,
                default=inspect.Parameter.empty if default is None else None,
            )
        )
    if args.kwarg:
        parameters.append(
            inspect.Parameter(args.kwarg.arg, inspect.Parameter.VAR_KEYWORD)
        )
    return inspect.Signature(parameters)


def check(path=EXAMPLE):
    tree = ast.parse(path.read_text(), filename=str(path))
    signatures = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not (node.module or "").startswith(
            "aiter."
        ):
            continue
        source = ROOT.joinpath(*node.module.split(".")).with_suffix(".py")
        source_tree = ast.parse(source.read_text(), filename=str(source))
        functions = {
            f.name: f for f in source_tree.body if isinstance(f, ast.FunctionDef)
        }
        for alias in node.names:
            if alias.name not in functions:
                raise ValueError(
                    f"{path}:{node.lineno}: missing source function {node.module}.{alias.name}"
                )
            signatures[alias.asname or alias.name] = source_signature(
                functions[alias.name]
            )
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in signatures:
            continue
        if any(isinstance(arg, ast.Starred) for arg in node.args) or any(
            k.arg is None for k in node.keywords
        ):
            raise ValueError(
                f"{path}:{node.lineno}: use explicit arguments for static validation"
            )
        try:
            signatures[node.func.id].bind(
                *[None for _ in node.args], **{k.arg: None for k in node.keywords}
            )
        except TypeError as error:
            raise ValueError(
                f"{path}:{node.lineno}: {node.func.id}: {error}"
            ) from error
        count += 1
    print(
        f"Validated {len(signatures)} AITER source imports and {count} calls in {path.name}"
    )


if __name__ == "__main__":
    check()
