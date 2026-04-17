"""
Tests for GemmTuner.asm_gemm_all_solutions SplitK semaphore guard.

The ASM SplitK kernels use a semaphore array of size gdx*gdy.  Candidates
where gdx*gdy > 1024 must be filtered out to avoid out-of-bounds writes.
"""
import sys
import types
import unittest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Minimal stubs so GemmTuner can be imported without a real ROCm stack
# ---------------------------------------------------------------------------

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
        sys.modules.setdefault(name, mod)


_install_stubs()

# Now import the module under test from the local gradlib tree
sys.path.insert(0, "gradlib")
from gradlib.GemmTuner import Gemm  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gemm(m, n, k):
    """Return a Gemm instance configured for the given shape, bypassing GPU calls."""
    import aiter.dtypes as dtypes
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

    @patch("gradlib.GemmTuner.get_gfx", return_value="gfx942")
    @patch("gradlib.GemmTuner.generate_data", return_value=None)
    def test_large_grid_candidates_are_skipped(self, _gen, _gfx):
        """Candidates where gdx*gdy > 1024 must not appear in the task list."""
        # tile 64x64 on a 4096x4096 grid => gdx=64, gdy=64 => 4096 > 1024
        gemm = _make_gemm(m=4096, n=4096, k=256)

        with patch.object(Gemm, "get_asm_kernels",
                          return_value=_fake_kernels(64, 64, 1, 64)):
            tasks = gemm.asm_gemm_all_solutions()

        for task in tasks:
            # info structure: ((m,n,k,...), solidx, splitK, libtype, kname)
            info = task[0]
            shape, splitK = info[0], info[2]
            m, n = shape[0], shape[1]
            gdx = (n + 64 - 1) // 64
            gdy = (m + 64 - 1) // 64
            self.assertLessEqual(
                gdx * gdy, 1024,
                f"Task with splitK={splitK} has grid {gdx}x{gdy}={gdx*gdy} > 1024",
            )

    @patch("gradlib.GemmTuner.get_gfx", return_value="gfx942")
    @patch("gradlib.GemmTuner.generate_data", return_value=None)
    def test_small_grid_candidates_are_kept(self, _gen, _gfx):
        """Candidates where gdx*gdy <= 1024 must still be generated."""
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
    def test_boundary_grid_exactly_1024_is_kept(self, _gen, _gfx):
        """A grid of exactly gdx*gdy == 1024 should not be filtered."""
        # tile=64, m=64*32=2048, n=64*32=2048 => gdx=32, gdy=32 => exactly 1024
        gemm = _make_gemm(m=2048, n=2048, k=256)

        with patch.object(Gemm, "get_asm_kernels",
                          return_value=_fake_kernels(64, 64, 1, 64)):
            tasks = gemm.asm_gemm_all_solutions()

        splitk_tasks = [t for t in tasks if t[0][2] > 1]
        self.assertGreater(len(splitk_tasks), 0,
                           "Grid of exactly 1024 should not be filtered")


if __name__ == "__main__":
    unittest.main()
