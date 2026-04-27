# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Level 1: Unit tests for the ASM SplitK semaphore grid guard in GemmTuner.

The ASM SplitK kernels index into a semaphore workspace of size ASM_SPLITK_MAX_GRID
(defined in aiter.ops.gemm_op_a16w16).  Candidates where gdx*gdy exceeds that limit
must be filtered out by asm_gemm_all_solutions() to avoid out-of-bounds writes.

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
# Minimal stubs so GemmTuner can be imported without a real ROCm stack.
#
# GemmTuner.py executes nine "from aiter..." imports at module level, before
# any test code runs.  On a machine without ROCm every one fails, so we must
# pre-populate sys.modules with objects that satisfy those imports before the
# "from gradlib.GemmTuner import Gemm" line below.  That is why _install_stubs()
# is called unconditionally at module level rather than inside setUp().
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ASM_SPLITK_MAX_GRID = 16 * 64  # mirrors aiter.ops.gemm_op_a16w16.ASM_SPLITK_MAX_GRID


def _make_stub(name, **attrs):
    # types.ModuleType produces a bare module object — the same class "import"
    # normally creates — so "from stub_name import attr" works without real code.
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

    dtypes_mod = _make_stub("aiter.dtypes", bf16=_DType("bf16"), fp32=_DType("fp32"))
    aiter_mod = _make_stub("aiter", dtypes=dtypes_mod, logger=logging.getLogger("aiter"))

    stubs = {
        "aiter":                            aiter_mod,
        "aiter.dtypes":                     dtypes_mod,
        "aiter.jit":                        _make_stub("aiter.jit"),
        "aiter.jit.core":                   _make_stub(
            "aiter.jit.core",
            AITER_CONFIG_GEMM_BF16="",
            get_asm_dir=lambda: "/nonexistent",
        ),
        "aiter.jit.utils":                  _make_stub("aiter.jit.utils"),
        "aiter.jit.utils.chip_info":        _make_stub(
            "aiter.jit.utils.chip_info",
            get_cu_num=lambda: 128,
            get_gfx=lambda: "gfx942",
        ),
        "aiter.ops":                        _make_stub("aiter.ops"),
        "aiter.ops.flydsl":                 _make_stub("aiter.ops.flydsl"),
        "aiter.ops.flydsl.utils":           _make_stub(
            "aiter.ops.flydsl.utils",
            is_flydsl_available=lambda: False,
        ),
        "aiter.ops.gemm_op_a16w16":         _make_stub(
            "aiter.ops.gemm_op_a16w16",
            ASM_SPLITK_MAX_GRID=_ASM_SPLITK_MAX_GRID,
        ),
        "aiter.ops.shuffle":                _make_stub(
            "aiter.ops.shuffle",
            shuffle_weight=lambda *a, **kw: None,
        ),
        "aiter.ops.triton":                 _make_stub("aiter.ops.triton"),
        "aiter.ops.triton.gemm":            _make_stub("aiter.ops.triton.gemm"),
        "aiter.ops.triton.gemm.basic":      _make_stub("aiter.ops.triton.gemm.basic"),
        "aiter.ops.triton.gemm.basic.gemm_a16w16": _make_stub(
            "aiter.ops.triton.gemm.basic.gemm_a16w16",
            gemm_a16w16=lambda *a, **kw: None,
        ),
        "aiter.utility":                    _make_stub("aiter.utility"),
        "aiter.utility.base_tuner":         _make_stub(
            "aiter.utility.base_tuner",
            GemmCommonTuner=type(
                "GemmCommonTuner",
                (),
                {"ARG_DEFAULTS": {
                    "verbose": False, "tune_file": "", "untune_file": "",
                    "errRatio": 0.05, "batch": 100, "profile_file": "",
                    "timeout": None, "warmup": 5, "iters": 101,
                    "min_improvement_pct": 3.0,
                }},
            ),
        ),
        "aiter.utility.mp_tuner":           _make_stub(
            "aiter.utility.mp_tuner",
            mp_tuner=lambda *a, **kw: [],
        ),
    }
    for name, mod in stubs.items():
        if name in sys.modules:
            continue
        try:
            # Prefer the real module when it is importable (e.g. on a GPU CI
            # machine where aiter is installed).  Only fall back to the stub on
            # ImportError, so we don't corrupt other tests in the same session
            # that depend on the real implementation.
            importlib.import_module(name)
        except Exception:
            sys.modules[name] = mod


_install_stubs()

sys.path.insert(0, str(_REPO_ROOT / "gradlib"))
from gradlib.GemmTuner import Gemm  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gemm(m, n, k):
    import aiter.dtypes as dtypes
    # __new__ allocates the instance without calling __init__, which would
    # trigger GPU data generation.  We set the attributes that
    # asm_gemm_all_solutions() actually reads, and nothing more.
    gemm = Gemm.__new__(Gemm)
    gemm.m = m
    gemm.n = n
    gemm.k = k
    gemm.indtype = dtypes.bf16
    gemm.outdtype = dtypes.fp32
    gemm.scaleAB = False
    gemm.has_bias = False
    gemm.bias = None
    gemm.is_shuffle = False
    gemm.asm_map = {}
    gemm.num_warmup = 0
    gemm.rtol = 1e-2
    gemm.atol = 1e-2
    return gemm


def _fake_kernels(tile_m, tile_n, splitK_flag, subK):
    """Return a minimal kernel dict as returned by get_asm_kernels."""
    key = (tile_m, tile_n, 1, splitK_flag, subK, 0, 0)
    return {key: ["fake_kernel"]}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSplitKSemaphoreGuard(unittest.TestCase):

    # @patch target must name the lookup site, not the definition site.
    # GemmTuner does "from aiter.jit.utils.chip_info import get_gfx", so get_gfx
    # lives in the gradlib.GemmTuner namespace — patching the original module
    # would have no effect on the already-bound local reference.
    @patch("gradlib.GemmTuner.get_gfx", return_value="gfx942")
    @patch("gradlib.GemmTuner.generate_data", return_value=None)
    def test_large_grid_candidates_are_skipped(self, _gen, _gfx):
        """Candidates where gdx*gdy > ASM_SPLITK_MAX_GRID must not appear in the task list."""
        # tile 64x64 on a 4096x4096 grid => gdx=64, gdy=64 => 4096 > 1024
        gemm = _make_gemm(m=4096, n=4096, k=256)

        with patch.object(Gemm, "get_asm_kernels",
                          return_value=_fake_kernels(64, 64, 1, 64)):
            tasks = gemm.asm_gemm_all_solutions()

        for task in tasks:
            info = task[0]
            shape, splitK = info[0], info[2]
            m, n = shape[0], shape[1]
            gdx = (n + 64 - 1) // 64
            gdy = (m + 64 - 1) // 64
            self.assertLessEqual(
                gdx * gdy, _ASM_SPLITK_MAX_GRID,
                f"Task with splitK={splitK} has grid {gdx}x{gdy}={gdx*gdy} > {_ASM_SPLITK_MAX_GRID}",
            )

    @patch("gradlib.GemmTuner.get_gfx", return_value="gfx942")
    @patch("gradlib.GemmTuner.generate_data", return_value=None)
    def test_small_grid_candidates_are_kept(self, _gen, _gfx):
        """Candidates where gdx*gdy <= ASM_SPLITK_MAX_GRID must still be generated."""
        # tile 128x128 on 128x128 => gdx=1, gdy=1 => 1 <= 1024
        gemm = _make_gemm(m=128, n=128, k=256)

        with patch.object(Gemm, "get_asm_kernels",
                          return_value=_fake_kernels(128, 128, 1, 64)):
            tasks = gemm.asm_gemm_all_solutions()

        splitk_tasks = [t for t in tasks if t[0][2] > 1]
        self.assertGreater(len(splitk_tasks), 0,
                           "Expected SplitK tasks for a small grid, got none")

    @patch("gradlib.GemmTuner.get_gfx", return_value="gfx942")
    @patch("gradlib.GemmTuner.generate_data", return_value=None)
    def test_boundary_grid_exactly_max_is_kept(self, _gen, _gfx):
        """A grid of exactly gdx*gdy == ASM_SPLITK_MAX_GRID must not be filtered."""
        # tile=64, m=2048, n=2048 => gdx=32, gdy=32 => exactly 1024
        gemm = _make_gemm(m=2048, n=2048, k=256)

        with patch.object(Gemm, "get_asm_kernels",
                          return_value=_fake_kernels(64, 64, 1, 64)):
            tasks = gemm.asm_gemm_all_solutions()

        splitk_tasks = [t for t in tasks if t[0][2] > 1]
        self.assertGreater(len(splitk_tasks), 0,
                           f"Grid of exactly {_ASM_SPLITK_MAX_GRID} should not be filtered")


if __name__ == "__main__":
    unittest.main()
