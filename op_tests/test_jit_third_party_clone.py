# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only regression tests for third-party JIT source acquisition."""

import ast
import logging
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


CORE_PATH = Path(__file__).resolve().parents[1] / "aiter" / "jit" / "core.py"


def _clone_function(destination):
    tree = ast.parse(CORE_PATH.read_text(encoding="utf-8"), filename=str(CORE_PATH))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "clone_3rdparty"
    )
    namespace = {
        "os": os,
        "shutil": shutil,
        "tempfile": tempfile,
        "logger": logging.getLogger(__name__),
        "bd_dir": str(destination.parent),
        "HIP_KITTENS_DIR": str(destination),
        "mp_lock": lambda lockPath, MainFunc, build_after_wait: MainFunc(),
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(CORE_PATH), "exec"),
        namespace,
    )
    return namespace["clone_3rdparty"]


class TestThirdPartyClone(unittest.TestCase):
    @unittest.skipIf(
        os.name == "nt", "Windows Git can hold submodule files during cleanup"
    )
    def test_local_clone_keeps_submodule_usable(self):
        real_run = subprocess.run

        def git(*args, cwd=None):
            return real_run(
                ["git", *args],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "GIT_ALLOW_PROTOCOL": "file"},
            )

        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            child = root_path / "child-source"
            source = root_path / "main-source"
            child.mkdir()
            source.mkdir()
            git("init", "-q", cwd=child)
            (child / "header.h").write_text("// child\n", encoding="utf-8")
            git("add", ".", cwd=child)
            git(
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-qm",
                "child",
                cwd=child,
            )
            git("init", "-q", cwd=source)
            git("submodule", "add", "-q", str(child), "child", cwd=source)
            git(
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-qam",
                "main",
                cwd=source,
            )
            commit = git("rev-parse", "HEAD", cwd=source).stdout.strip()

            def local_run(command, **kwargs):
                command = [
                    str(source)
                    if part == "https://github.com/HazyResearch/HipKittens.git"
                    else commit
                    if part == "d3cd9b31cb0ff611ff64b5701f57ccdeb7712f39"
                    else f"--revision={commit}"
                    if part == "--revision=d3cd9b31cb0ff611ff64b5701f57ccdeb7712f39"
                    else part
                    for part in command
                ]
                return real_run(
                    command,
                    **kwargs,
                    env={**os.environ, "GIT_ALLOW_PROTOCOL": "file"},
                    capture_output=True,
                    text=True,
                )

            git_version = real_run(
                ["git", "--version"], capture_output=True, text=True, check=True
            ).stdout
            match = re.search(r"(\d+)\.(\d+)", git_version)
            versions = ["2.48.0"]
            if match and tuple(map(int, match.groups())) >= (2, 49):
                versions.append("2.49.0")
            for version in versions:
                with self.subTest(version=version):
                    destination = root_path / version / "HipKittens"
                    clone = _clone_function(destination)
                    with (
                        mock.patch(
                            "subprocess.check_output",
                            return_value=f"git version {version}",
                        ),
                        mock.patch("subprocess.run", side_effect=local_run),
                    ):
                        clone("HipKittens")

                    self.assertEqual(
                        git("rev-parse", "HEAD", cwd=destination).stdout.strip(),
                        commit,
                    )
                    self.assertEqual(
                        (destination / "child" / "header.h").read_text(
                            encoding="utf-8"
                        ),
                        "// child\n",
                    )
                    self.assertEqual(
                        git(
                            "rev-parse",
                            "--is-inside-work-tree",
                            cwd=destination / "child",
                        ).stdout.strip(),
                        "true",
                    )

    def test_failed_reset_leaves_no_destination_and_retry_succeeds(self):
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / "HipKittens"
            clone = _clone_function(destination)
            attempts = 0

            def run(command, **kwargs):
                nonlocal attempts
                self.assertTrue(kwargs.get("check"))
                if "clone" in command:
                    Path(command[-1]).mkdir()
                elif "reset" in command:
                    attempts += 1
                    if attempts == 1:
                        raise subprocess.CalledProcessError(1, command)
                return subprocess.CompletedProcess(command, 0)

            with (
                mock.patch(
                    "subprocess.check_output", return_value="git version 2.48.0"
                ),
                mock.patch("subprocess.call", return_value=0),
                mock.patch("subprocess.run", side_effect=run),
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    clone("HipKittens")
                self.assertFalse(destination.exists())
                clone("HipKittens")

            self.assertEqual(attempts, 2)
            self.assertTrue(destination.is_dir())
            self.assertEqual(list(Path(root).glob(".HipKittens-*")), [])

    def test_fast_clone_failure_does_not_change_global_git_config(self):
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / "HipKittens"
            clone = _clone_function(destination)

            def run(command, **kwargs):
                self.assertTrue(kwargs.get("check"))
                if "clone" in command:
                    Path(command[-1]).mkdir()
                    raise subprocess.CalledProcessError(1, command)
                return subprocess.CompletedProcess(command, 0)

            with (
                mock.patch(
                    "subprocess.check_output", return_value="git version 2.49.0"
                ),
                mock.patch("subprocess.call") as legacy_call,
                mock.patch("subprocess.run", side_effect=run),
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    clone("HipKittens")

            legacy_call.assert_not_called()
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
