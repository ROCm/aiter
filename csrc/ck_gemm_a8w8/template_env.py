# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from jinja2.environment import Template, TemplateStream

_TEMPLATES_DIR = Path(__file__).parent / "templates"

template_env = None


def get_template(template_name: str) -> Template:
    global template_env
    if template_env is None:
        template_env = Environment(
            loader=FileSystemLoader(str(_TEMPLATES_DIR)),
            undefined=StrictUndefined,
            keep_trailing_newline=True,
        )
    return template_env.get_template(template_name)


def stream(template_name: str, output_file, **ctx) -> None:
    return get_template(template_name).stream(**ctx).dump(output_file)


def render(template_name: str, **ctx) -> str:
    return get_template(Template_name).render(**ctx)
