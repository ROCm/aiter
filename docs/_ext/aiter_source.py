# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Render checked Python signatures without importing ROCm, torch or AITER."""

import ast
from pathlib import Path

from docutils import nodes
from sphinx.util.docutils import SphinxDirective


class AiterFunction(SphinxDirective):
    required_arguments = 1

    def run(self):
        module, name = self.arguments[0].rsplit(".", 1)
        root = Path(self.env.srcdir).parent
        path = root.joinpath(*module.split(".")).with_suffix(".py")
        if not path.is_relative_to(root / "aiter") or not path.is_file():
            raise self.error(f"Missing AITER source module: {module}")
        self.env.note_dependency(str(path))
        tree = ast.parse(path.read_text())
        function = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == name
            ),
            None,
        )
        if function is None:
            raise self.error(f"Missing AITER function: {module}.{name}")
        signature = f"{name}({ast.unparse(function.args)})"
        if function.returns:
            signature += f" -> {ast.unparse(function.returns)}"
        code = nodes.literal_block(signature, signature, language="python")
        revision = self.config.source_revision
        if revision == "unknown":
            revision = "main"
        url = (
            f"https://github.com/ROCm/aiter/blob/{revision}/"
            f"{path.relative_to(root).as_posix()}#L{function.lineno}"
        )
        source = nodes.paragraph()
        source += nodes.reference("", f"Source: {module}.{name}", refuri=url)
        return [code, source]


def setup(app):
    app.add_config_value("source_revision", "unknown", "html")
    app.add_directive("aiter-function", AiterFunction)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
