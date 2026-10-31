# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only tests for the ``hipconfig`` lookup in ``csrc.cpp_itfs.utils``."""

import pathlib
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from csrc.cpp_itfs import utils as cpp_itfs_utils  # noqa: E402


def _run_result(stdout="", stderr="", returncode=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def test_hip_version_comes_from_the_located_hipconfig(monkeypatch):
    monkeypatch.setattr(
        cpp_itfs_utils, "_rocm_executable", lambda name: f"/rocm/bin/{name}"
    )
    with mock.patch.object(
        cpp_itfs_utils.subprocess,
        "run",
        return_value=_run_result("7.12.60610-2bd1678d3d\n"),
    ) as run:
        version = cpp_itfs_utils.get_hip_version()

    assert version == cpp_itfs_utils.Version("7.12.60610+2bd1678d3d")
    assert run.call_args.args[0] == ["/rocm/bin/hipconfig", "--version"]
    assert not run.call_args.kwargs.get("shell")


def test_missing_hipconfig_raises_a_clear_error(monkeypatch):
    monkeypatch.setattr(cpp_itfs_utils, "_rocm_executable", lambda name: None)
    with mock.patch.object(cpp_itfs_utils.subprocess, "run") as run:
        with pytest.raises(RuntimeError, match=r"`hipconfig`.*ROCM_PATH"):
            cpp_itfs_utils.get_hip_version()
    run.assert_not_called()


def test_hipconfig_without_output_raises_a_clear_error(monkeypatch):
    monkeypatch.setattr(
        cpp_itfs_utils, "_rocm_executable", lambda name: f"/rocm/bin/{name}"
    )
    failed = _run_result(stderr="not found", returncode=127)
    with mock.patch.object(cpp_itfs_utils.subprocess, "run", return_value=failed):
        with pytest.raises(
            RuntimeError, match=r"/rocm/bin/hipconfig --version.*127.*not found"
        ):
            cpp_itfs_utils.get_hip_version()
