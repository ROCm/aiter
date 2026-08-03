# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import re
import shlex
from typing import List


def windows_blob_gen_argv(
    python_executable: str, blob_gen_cmd: str, blob_dir: str
) -> List[str]:
    """Convert a blob generator command to argv without invoking cmd.exe.

    Blob generator commands are stored as shell-like strings whose first
    argument is always a Python script and whose remaining arguments are
    option/value pairs. On Windows, paths in those strings may contain spaces
    (including the checkout, Python installation, output, and tune-file paths).
    Group each option's value before passing the result to CreateProcess so no
    shell quoting is required.
    """
    command = blob_gen_cmd.format(blob_dir).strip()
    script_match = re.match(r"^(.*?\.py)(?:\s+(.*))?$", command, re.IGNORECASE)
    if script_match is None:
        raise ValueError(
            f"Blob generator command must start with a .py script: {command}"
        )

    script, raw_args = script_match.groups()
    argv = [python_executable, script]
    if not raw_args:
        return argv

    lexer = shlex.shlex(raw_args, posix=False)
    lexer.whitespace_split = True
    lexer.commenters = ""

    option = None
    values: List[str] = []

    def append_option():
        nonlocal option, values
        if option is None:
            return
        argv.append(option)
        if values:
            value = " ".join(values)
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            argv.append(value)
        option = None
        values = []

    for token in lexer:
        is_option = re.match(r"^-{1,2}[A-Za-z]", token) is not None
        if is_option:
            append_option()
            if "=" in token:
                option, first_value = token.split("=", 1)
                values.append(first_value)
            else:
                option = token
        elif option is None:
            raise ValueError(f"Unexpected positional blob generator argument: {token}")
        else:
            values.append(token)
    append_option()
    return argv
