# SPDX-License-Identifier: MIT
"""CPU tests: unsupported architectures must not run or import GPU backends."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location("combo_dispatch", ROOT / "bench_combo.py")
combo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(combo)


@pytest.mark.parametrize(
    "arch,expected",
    [("gfx950", "bench_gfx950_combo")],
)
def test_dispatch_preserves_backend_arguments(arch, expected):
    backend = Mock()
    with (
        patch.object(combo.importlib, "import_module", return_value=backend) as load,
        patch.object(sys, "argv", ["bench_combo.py"]),
    ):
        combo.main(["--arch", arch, "--perf", "--ops", "a16w16"])
        load.assert_called_once_with(expected)
        backend.main.assert_called_once_with()
        assert sys.argv[1:] == ["--perf", "--ops", "a16w16"]


def test_unknown_arch_is_error():
    with pytest.raises(ValueError, match="Unsupported architecture"):
        combo.backend_for("gfx942")


def test_help_without_gpu_imports():
    result = subprocess.run(
        [sys.executable, str(ROOT / "bench_combo.py"), "--help"],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0
    assert "gfx950" in result.stdout


def test_gfx950_help_without_gpu_imports():
    result = subprocess.run(
        [sys.executable, str(ROOT / "bench_combo.py"), "--arch", "gfx950", "--help"],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "softmax" in result.stdout


def test_gfx950_rejects_unsupported_operator():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench_combo.py"),
            "--arch",
            "gfx950",
            "--ops",
            "mega_moe",
            "--output",
            "unused.json",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr
