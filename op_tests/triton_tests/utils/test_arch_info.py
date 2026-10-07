# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import os
import subprocess
import sys
from unittest.mock import MagicMock, patch
import pytest

from aiter.ops.triton.utils._triton import arch_info


# =========================================================================
# A. SUBPROCESS TESTS (Fresh Python interpreter, real import boundary)
# =========================================================================

def test_subprocess_import_arch_info_clean():
    """Verify arch_info imports cleanly in a fresh interpreter without blaming JAX."""
    cmd = [
        sys.executable,
        "-c",
        "import aiter.ops.triton.utils._triton.arch_info as ai; print(ai.get_arch())",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, (
        f"Import of arch_info failed.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    )
    assert "No module named 'jax'" not in res.stderr


def test_subprocess_import_types_clean():
    """Verify downstream types.py imports cleanly without blast-radius crash."""
    code = (
        "try:\n"
        "    import triton\n"
        "except ImportError:\n"
        "    import sys, unittest.mock\n"
        "    mock_t = unittest.mock.MagicMock()\n"
        "    mock_t.runtime.driver.active.get_current_target.side_effect = RuntimeError('No active target')\n"
        "    sys.modules['triton'] = mock_t\n"
        "    sys.modules['triton.language'] = unittest.mock.MagicMock()\n"
        "import aiter.ops.triton.utils.types as t\n"
        "assert t.e4m3_dtype is not None\n"
        "print('SUCCESS')\n"
    )
    env = {**os.environ, "AITER_TRITON_ONLY": "1"}
    cmd = [sys.executable, "-c", code]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert res.returncode == 0, (
        f"Import of types.py failed.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    )


def test_subprocess_missing_jax_does_not_blame_jax():
    """Verify that when JAX is absent, arch_info does not crash or raise ModuleNotFoundError."""
    cmd = [
        sys.executable,
        "-c",
        "import sys; sys.modules['jax'] = None; sys.modules['jax._src.lib'] = None; "
        "import aiter.ops.triton.utils._triton.arch_info as ai; "
        "print('SUCCESS')",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, (
        f"arch_info import failed when JAX is missing.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    )
    assert "No module named 'jax'" not in res.stderr


# =========================================================================
# B. IN-PROCESS UNIT TESTS (Targeted mocking, zero sys.modules reloads)
# =========================================================================

@pytest.mark.parametrize("mock_arch", ["gfx942", "gfx950", "gfx1250"])
def test_detect_arch_triton_success(mock_arch):
    """Verify _detect_arch() correctly extracts architecture when Triton succeeds."""
    mock_triton = MagicMock()
    mock_triton.runtime.driver.active.get_current_target.return_value.arch = mock_arch

    with patch.object(arch_info, "triton", mock_triton):
        assert arch_info._detect_arch() == mock_arch


def test_detect_arch_triton_failure_jax_success():
    """Verify _detect_arch() falls back to JAX when Triton driver fails."""
    mock_triton = MagicMock()
    mock_triton.runtime.driver.active.get_current_target.side_effect = RuntimeError("No active target")

    mock_jax = MagicMock()
    mock_jax_src = MagicMock()
    mock_jax_lib = MagicMock()
    mock_gpu_triton = MagicMock()
    mock_gpu_triton.get_arch_details.return_value = "gfx950:sramecc+:xnack-"
    mock_jax_lib.gpu_triton = mock_gpu_triton

    with patch.object(arch_info, "triton", mock_triton):
        with patch.dict(sys.modules, {
            "jax": mock_jax,
            "jax._src": mock_jax_src,
            "jax._src.lib": mock_jax_lib,
            "jax._src.lib.gpu_triton": mock_gpu_triton,
        }):
            assert arch_info._detect_arch() == "gfx950"


def test_detect_arch_both_fail_returns_none():
    """Verify _detect_arch() returns None when both Triton and JAX fail."""
    mock_triton = MagicMock()
    mock_triton.runtime.driver.active.get_current_target.side_effect = RuntimeError("hipErrorNoDevice")

    with patch.object(arch_info, "triton", mock_triton):
        with patch.dict(sys.modules, {"jax": None, "jax._src.lib": None, "jax._src.lib.gpu_triton": None}):
            assert arch_info._detect_arch() is None


@pytest.mark.parametrize("arch,expected_fp8,expected_gluon,expected_fp4,expected_tdm", [
    ("gfx942", True, False, False, False),
    ("gfx950", True, True, True, False),
    ("gfx1250", True, True, True, True),
    ("gfx1200", True, False, False, False),
    ("gfx1100", False, False, False, False),
    (None, False, False, False, False),
])
def test_capability_predicates(arch, expected_fp8, expected_gluon, expected_fp4, expected_tdm):
    """Verify capability predicates evaluate accurately without reloading modules."""
    with patch.object(arch_info, "_CACHED_ARCH", arch):
        assert arch_info.get_arch() == arch
        assert arch_info.is_fp8_avail() is expected_fp8
        assert arch_info.is_gluon_avail() is expected_gluon
        assert arch_info.is_fp4_avail() is expected_fp4
        assert arch_info.is_tdm_avail() is expected_tdm


def test_lds_cap_bytes_constant_unaltered():
    """Verify LDS capacity table constants are preserved."""
    assert arch_info._LDS_CAP_BYTES == {
        "gfx1250": 327680,
        "gfx950": 163840,
        "gfx942": 65536,
    }


# =========================================================================
# C. LIVE RUNNER CANARY (Real AMD ROCm runners)
# =========================================================================

def test_live_runner_canary():
    """Live GPU canary: verifies detected architecture on real ROCm runners.

    Skipped when PyTorch cannot establish ROCm GPU availability, avoiding
    the false assumption that CPU-only PyTorch implies Triton or JAX cannot
    detect a GPU.
    """
    try:
        import torch
        is_rocm = (
            torch.cuda.is_available()
            and getattr(torch.version, "hip", None) is not None
        )
    except ImportError:
        is_rocm = False

    if not is_rocm:
        pytest.skip(
            "PyTorch ROCm GPU is not available; skipping live GPU canary."
        )

    arch = arch_info.get_arch()
    assert isinstance(arch, str) and arch.startswith("gfx"), (
        f"Expected detected AMD GPU architecture on ROCm runner, got {arch!r}"
    )
