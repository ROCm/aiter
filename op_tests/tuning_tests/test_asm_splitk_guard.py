# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Level 1: Unit tests for the ASM SplitK semaphore grid guard in GemmA16W16Tuner.

The ASM SplitK kernels index into a semaphore workspace of size ASM_SPLITK_MAX_GRID
(defined in aiter.ops.gemm_op_a16w16).  Candidates where gdx*gdy exceeds that limit
must be filtered out by _get_asm_tasks() to avoid out-of-bounds writes.

No GPU required.  Run:
    python3 -m unittest op_tests.tuning_tests.test_asm_splitk_guard -v
"""

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Minimal stubs so gemm_a16w16_tune can be imported without a real ROCm stack.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ASM_SPLITK_MAX_GRID = 16 * 64


def _make_stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def _install_stubs():
    import logging

    class _DType:
        def __init__(self, name):
            self._name = name

        def __str__(self):
            return self._name

        def __repr__(self):
            return self._name

        def __eq__(self, other):
            return isinstance(other, _DType) and self._name == other._name

        def __hash__(self):
            return hash(self._name)

    dtypes_mod = _make_stub(
        "aiter.dtypes",
        bf16=_DType("bf16"),
        fp16=_DType("fp16"),
        fp32=_DType("fp32"),
        fp8=_DType("fp8"),
    )
    aiter_mod = _make_stub(
        "aiter", dtypes=dtypes_mod, logger=logging.getLogger("aiter")
    )

    stubs = {
        "aiter": aiter_mod,
        "aiter.dtypes": dtypes_mod,
        "aiter.jit": _make_stub("aiter.jit"),
        "aiter.jit.core": _make_stub(
            "aiter.jit.core",
            AITER_CONFIG_GEMM_BF16="",
            get_asm_dir=lambda: "/nonexistent",
        ),
        "aiter.jit.utils": _make_stub("aiter.jit.utils"),
        "aiter.jit.utils.chip_info": _make_stub(
            "aiter.jit.utils.chip_info",
            get_cu_num=lambda: 128,
            get_gfx=lambda: "gfx942",
            get_gfx_runtime=lambda: "gfx942",
        ),
        "aiter.ops": _make_stub("aiter.ops"),
        "aiter.ops.flydsl": _make_stub("aiter.ops.flydsl"),
        "aiter.ops.flydsl.utils": _make_stub(
            "aiter.ops.flydsl.utils",
            is_flydsl_available=lambda: False,
        ),
        "aiter.ops.gemm_op_a16w16": _make_stub(
            "aiter.ops.gemm_op_a16w16",
            ASM_SPLITK_MAX_GRID=_ASM_SPLITK_MAX_GRID,
        ),
        "aiter.ops.shuffle": _make_stub(
            "aiter.ops.shuffle",
            shuffle_weight=lambda *a, **kw: None,
        ),
        "aiter.ops.triton": _make_stub("aiter.ops.triton"),
        "aiter.ops.triton.gemm": _make_stub("aiter.ops.triton.gemm"),
        "aiter.ops.triton.gemm.basic": _make_stub("aiter.ops.triton.gemm.basic"),
        "aiter.ops.triton.gemm.basic.gemm_a16w16": _make_stub(
            "aiter.ops.triton.gemm.basic.gemm_a16w16",
            gemm_a16w16=lambda *a, **kw: None,
        ),
        "aiter.utility": _make_stub("aiter.utility"),
        "aiter.utility.base_tuner": _make_stub(
            "aiter.utility.base_tuner",
            GemmCommonTuner=type(
                "GemmCommonTuner",
                (),
                {
                    "ARG_DEFAULTS": {
                        "verbose": False,
                        "tune_file": "",
                        "untune_file": "",
                        "errRatio": 0.05,
                        "batch": 100,
                        "profile_file": "",
                        "timeout": None,
                        "warmup": 5,
                        "iters": 101,
                        "min_improvement_pct": 3.0,
                        "sort": True,
                    },
                    "__init__": lambda self, *a, **kw: None,
                },
            ),
        ),
        "aiter.utility.mp_tuner": _make_stub(
            "aiter.utility.mp_tuner",
            mp_tuner=lambda *a, **kw: [],
        ),
    }
    for name, mod in stubs.items():
        if name in sys.modules:
            continue
        try:
            importlib.import_module(name)
        except Exception:
            sys.modules[name] = mod


_install_stubs()

sys.path.insert(0, str(_REPO_ROOT / "csrc" / "gemm_a16w16"))
from gemm_a16w16_tune import GemmA16W16Tuner  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_kernels(tile_m, tile_n, splitK_flag, subK):
    """Return a minimal kernel dict as returned by get_asm_kernels."""
    key = (tile_m, tile_n, 1, splitK_flag, subK, 0, 0)
    return {key: ["fake_kernel"]}


def _make_tuner():
    """Create a GemmA16W16Tuner without argparse / GPU."""
    key = [
        "gfx",
        "cu_num",
        "M",
        "N",
        "K",
        "bias",
        "dtype",
        "outdtype",
        "scaleAB",
        "bpreshuffle",
    ]
    resultList = [
        "libtype",
        "solidx",
        "splitK",
        "us",
        "kernelName",
        "err_ratio",
        "tflops",
        "bw",
    ]
    tuner = GemmA16W16Tuner.__new__(GemmA16W16Tuner)
    tuner.keys = key
    tuner.columns = key + resultList
    return tuner


def _call_get_asm_tasks(tuner, m, n, k, asm_kernels):
    import aiter.dtypes as dtypes

    info_keys = (
        "gfx942",
        128,
        m,
        n,
        k,
        False,
        str(dtypes.bf16),
        str(dtypes.fp32),
        False,
        False,
    )
    run_kwargs = {"num_warmup": 0, "num_iters": 1}
    with patch("gemm_a16w16_tune.get_asm_kernels", return_value=asm_kernels):
        with patch("gemm_a16w16_tune.get_gfx", return_value="gfx942"):
            return tuner._get_asm_tasks(
                info_keys,
                False,
                dtypes.bf16,
                dtypes.fp32,
                False,
                False,
                run_kwargs,
            )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSplitKSemaphoreGuard(unittest.TestCase):

    def test_large_grid_candidates_are_skipped(self):
        """Candidates where gdx*gdy > ASM_SPLITK_MAX_GRID must not appear in the task list."""
        tuner = _make_tuner()
        tasks = _call_get_asm_tasks(
            tuner, 4096, 4096, 256, _fake_kernels(64, 64, 1, 64)
        )

        for task in tasks:
            info = task[0]
            shape = info[0]
            splitK = info[2]
            m, n = shape[2], shape[3]
            gdx = (n + 64 - 1) // 64
            gdy = (m + 64 - 1) // 64
            self.assertLessEqual(
                gdx * gdy,
                _ASM_SPLITK_MAX_GRID,
                f"Task with splitK={splitK} has grid {gdx}x{gdy}={gdx*gdy} > {_ASM_SPLITK_MAX_GRID}",
            )

    def test_small_grid_candidates_are_kept(self):
        """Candidates where gdx*gdy <= ASM_SPLITK_MAX_GRID must still be generated."""
        tuner = _make_tuner()
        tasks = _call_get_asm_tasks(
            tuner, 128, 128, 256, _fake_kernels(128, 128, 1, 64)
        )

        splitk_tasks = [t for t in tasks if t[0][2] > 1]
        self.assertGreater(
            len(splitk_tasks), 0, "Expected SplitK tasks for a small grid, got none"
        )

    def test_boundary_grid_exactly_max_is_kept(self):
        """A grid of exactly gdx*gdy == ASM_SPLITK_MAX_GRID must not be filtered."""
        tuner = _make_tuner()
        tasks = _call_get_asm_tasks(
            tuner, 2048, 2048, 256, _fake_kernels(64, 64, 1, 64)
        )

        splitk_tasks = [t for t in tasks if t[0][2] > 1]
        self.assertGreater(
            len(splitk_tasks),
            0,
            f"Grid of exactly {_ASM_SPLITK_MAX_GRID} should not be filtered",
        )


if __name__ == "__main__":
    unittest.main()
