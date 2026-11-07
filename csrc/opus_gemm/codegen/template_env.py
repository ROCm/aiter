# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Jinja2 rendering for the OPUS GEMM codegen C++ emitters.

Mirrors ``csrc/ck_gemm_a8w8/template_env.py``: a single lazily-built
:class:`~jinja2.Environment` backed by a ``templates/`` directory, with
``stream`` (render straight to a file) and ``render`` (return the text)
helpers so the arch modules stop hand-assembling multi-hundred-line C++
f-strings.

The C++ these templates emit is brace-dense -- a launcher body opens blocks
with ``{{``/``}}`` (extra nested scopes) and launches kernels with
``<<<...>>>``. Jinja2's default ``{{ }}``/``{% %}``/``{# #}`` delimiters would
collide with that literal text, so we rebind them to ``[[ ]]`` / ``[% %]`` /
``[# #]``. Those three sequences never occur in any generated OPUS source, so
the templates can carry C++ through verbatim with no brace-escaping.
"""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from jinja2.environment import Template

_TEMPLATES_DIR = Path(__file__).parent / "templates"

template_env = None


def get_template(template_name: str) -> Template:
    global template_env
    if template_env is None:
        template_env = Environment(
            loader=FileSystemLoader(str(_TEMPLATES_DIR)),
            undefined=StrictUndefined,
            keep_trailing_newline=True,
            # Custom delimiters: C++ braces/angle-brackets stay literal.
            variable_start_string="[[",
            variable_end_string="]]",
            block_start_string="[%",
            block_end_string="%]",
            comment_start_string="[#",
            comment_end_string="#]",
        )
    return template_env.get_template(template_name)


def stream(template_name: str, output_file, **ctx) -> None:
    return get_template(template_name).stream(**ctx).dump(output_file)


def render(template_name: str, **ctx) -> str:
    return get_template(template_name).render(**ctx)
