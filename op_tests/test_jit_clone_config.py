# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only coverage for the JIT third-party Git command.

Load only the function under test so ROCm extension imports are not required.
"""

import ast
import os
import re
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


def _clone_function(tmp_path):
    source = Path(__file__).resolve().parents[1] / "aiter/jit/core.py"
    module = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "clone_3rdparty"
    )
    namespace = {
        "os": os,
        "HIP_KITTENS_DIR": str(tmp_path / "HipKittens"),
        "bd_dir": str(tmp_path),
        "logger": Mock(),
        "mp_lock": lambda *, lockPath, MainFunc: MainFunc(),
    }
    code = compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec")
    exec(code, namespace)
    return namespace["clone_3rdparty"]


def test_modern_git_clone_does_not_write_global_config(tmp_path):
    clone = _clone_function(tmp_path)
    with (
        patch.object(
            subprocess, "check_output", return_value="git version 2.53.0"
        ) as output,
        patch.object(subprocess, "call", return_value=0) as call,
    ):
        clone("HipKittens")

    output.assert_called_once_with(["git", "--version"], text=True)
    call.assert_called_once_with(
        [
            "git",
            "-c",
            "advice.detachedHead=false",
            "clone",
            "-q",
            "--revision=d3cd9b31cb0ff611ff64b5701f57ccdeb7712f39",
            "--depth=1",
            "--recurse-submodules",
            "https://github.com/HazyResearch/HipKittens.git",
            str(tmp_path / "HipKittens"),
        ]
    )


def test_older_git_keeps_existing_clone_path(tmp_path):
    clone = _clone_function(tmp_path)
    with (
        patch.object(subprocess, "check_output", return_value="git version 2.48.0"),
        patch.object(subprocess, "call", return_value=0) as call,
    ):
        clone("HipKittens")

    assert call.call_count == 3
    assert call.call_args_list[0].args[0][:3] == ["git", "clone", "-q"]
    assert call.call_args_list[1].args[0][3:5] == ["reset", "-q"]
    assert call.call_args_list[2].args[0][3:5] == ["submodule", "update"]


def test_command_local_config_with_real_git(tmp_path):
    version = subprocess.check_output(["git", "--version"], text=True)
    match = re.search(r"(\d+)\.(\d+)", version)
    if match is None or tuple(map(int, match.groups())) < (2, 49):
        pytest.skip("git clone --revision requires Git >= 2.49")

    source = tmp_path / "source"
    target = tmp_path / "target"
    global_config = tmp_path / "global.gitconfig"
    global_config.write_text("[advice]\n\tdetachedHead = true\n", encoding="utf-8")
    original_config = global_config.read_bytes()
    env = {**os.environ, "GIT_CONFIG_GLOBAL": str(global_config)}

    subprocess.run(["git", "init", "-q", str(source)], check=True, env=env)
    subprocess.run(
        ["git", "-C", str(source), "config", "user.name", "Test"], check=True, env=env
    )
    subprocess.run(
        ["git", "-C", str(source), "config", "user.email", "test@example.com"],
        check=True,
        env=env,
    )
    (source / "fixture.txt").write_text("clone fixture\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(source), "add", "fixture.txt"], check=True, env=env
    )
    subprocess.run(
        ["git", "-C", str(source), "commit", "-qm", "fixture"], check=True, env=env
    )
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True, env=env
    ).strip()

    subprocess.run(
        [
            "git",
            "-c",
            "advice.detachedHead=false",
            "clone",
            "-q",
            f"--revision={commit}",
            "--depth=1",
            "--recurse-submodules",
            str(source),
            str(target),
        ],
        check=True,
        env=env,
    )
    assert global_config.read_bytes() == original_config
    assert (
        subprocess.check_output(
            ["git", "-C", str(target), "rev-parse", "HEAD"], text=True, env=env
        ).strip()
        == commit
    )
