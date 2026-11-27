# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
No a16w16 dispatch path selects ``gemm_a16w16_asm`` on an arch that has no
``hsa/<gfx>/bf16gemm`` kernel table (e.g. gfx1250, #2519), and gfx942/gfx950
keep their selection. The arch is stubbed, so this runs on CPU without ROCm.

Run:
    python3 -m unittest op_tests.tuning_tests.test_a16w16_asm_arch_guard -v
"""

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_AITER = _REPO_ROOT / "aiter"
_STUBBED = (
    "aiter",
    "aiter.dtypes",
    "aiter.jit",
    "aiter.jit.core",
    "aiter.jit.utils",
    "aiter.jit.utils.asm_kernels",
    "aiter.jit.utils.build_targets",
    "aiter.jit.utils.chip_info",
    "aiter.jit.utils.torch_guard",
    "aiter.ops",
    "aiter.ops.gemm_op_common",
    "aiter.plugin",
    "aiter.tuned_gemm",
)


def _module(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_tuned_gemm():
    """aiter.tuned_gemm over stubbed native pieces; the arch helpers are real."""
    dtypes = _module(
        "aiter.dtypes",
        bf16=torch.bfloat16,
        fp16=torch.float16,
        fp32=torch.float32,
        fp8=torch.float8_e4m3fn,
    )
    stubs = {
        "aiter": _module(
            "aiter",
            dtypes=dtypes,
            gemm_a16w16_asm=mock.Mock(name="gemm_a16w16_asm"),
            hipb_create_extension=mock.Mock(),
            hipb_mm=mock.Mock(),
            logger=mock.Mock(),
        ),
        "aiter.dtypes": dtypes,
        "aiter.jit": _module("aiter.jit"),
        "aiter.jit.core": _module(
            "aiter.jit.core",
            AITER_CONFIGS=types.SimpleNamespace(AITER_CONFIG_GEMM_BF16_FILE=""),
            AITER_LOG_TUNED_CONFIG=False,
        ),
        "aiter.jit.utils": _module("aiter.jit.utils"),
        "aiter.jit.utils.chip_info": _module(
            "aiter.jit.utils.chip_info",
            get_cu_num=lambda: 256,
            get_gfx_runtime=lambda: "gfx950",
        ),
        "aiter.jit.utils.torch_guard": _module(
            "aiter.jit.utils.torch_guard",
            torch_compile_guard=lambda **_kw: (lambda fn: fn),
        ),
        "aiter.ops": _module("aiter.ops"),
        "aiter.ops.gemm_op_common": _module(
            "aiter.ops.gemm_op_common", get_padded_m=lambda m, _n, _k, _gl: m
        ),
        "aiter.plugin": _module("aiter.plugin", route=lambda **_kw: (lambda fn: fn)),
    }
    sys.modules.update(stubs)
    _load("aiter.jit.utils.asm_kernels", _AITER / "jit/utils/asm_kernels.py")
    _load("aiter.jit.utils.build_targets", _AITER / "jit/utils/build_targets.py")
    return _load("aiter.tuned_gemm", _AITER / "tuned_gemm.py")


_saved = {name: sys.modules.get(name) for name in _STUBBED}
try:
    tuned = _load_tuned_gemm()
finally:
    for _name, _mod in _saved.items():
        if _mod is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _mod

BF16 = str(torch.bfloat16)
FP32 = str(torch.float32)
CU_NUM = 256
M, N, K = 64, 256, 1024
NO_ASM_ARCHES = ("gfx1250", "gfx1250-strict", "gfx90a")


def _row_key(gfx, bpreshuffle):
    return (gfx, CU_NUM, M, N, K, False, BF16, BF16, False, bpreshuffle)


class _Base(unittest.TestCase):
    def _config(self, gfx, bpreshuffle=False, otype=BF16, table=None):
        patch = mock.patch.object(tuned, "get_GEMM_A16W16_config_", lambda: table or {})
        patch.start()
        self.addCleanup(patch.stop)
        tuned.get_GEMM_A16W16_config.cache_clear()
        self.addCleanup(tuned.get_GEMM_A16W16_config.cache_clear)
        return tuned.get_GEMM_A16W16_config(
            M, N, K, False, BF16, otype, False, bpreshuffle, gfx, CU_NUM
        )


class TestAsmKernelMetadata(unittest.TestCase):
    def test_capability_follows_hsa_metadata(self):
        for gfx in ("gfx942", "gfx950"):
            self.assertTrue(tuned.asm_a16w16_supported(gfx), gfx)
        for gfx in NO_ASM_ARCHES:
            self.assertFalse(tuned.asm_a16w16_supported(gfx), gfx)

    def test_any_arch_dir_with_a_kernel_table_counts(self):
        # The answer is cached per root, so each case gets its own root.
        has_asm_kernels = tuned.has_asm_kernels
        with tempfile.TemporaryDirectory() as empty, tempfile.TemporaryDirectory() as full:
            Path(empty, "gfx9999", "bf16gemm").mkdir(parents=True)
            Path(full, "gfx9999", "bf16gemm").mkdir(parents=True)
            Path(full, "gfx9999", "bf16gemm", "bf16gemm_fp32bf16.csv").write_text(
                "knl_name\n"
            )
            self.assertFalse(has_asm_kernels("bf16gemm", "gfx9999", empty))
            self.assertTrue(has_asm_kernels("bf16gemm", "gfx9999", full))
            self.assertFalse(has_asm_kernels("bf16gemm", "gfx1250", full))


class TestDefaultSelection(_Base):
    def test_bpreshuffle_without_asm_kernels_raises_before_launch(self):
        for gfx in NO_ASM_ARCHES:
            with self.subTest(gfx=gfx):
                with self.assertRaisesRegex(RuntimeError, "preshuffled B.*" + gfx):
                    self._config(gfx, bpreshuffle=True)

    def test_bpreshuffle_selection_is_unchanged_on_gfx942_and_gfx950(self):
        cfg = self._config("gfx942", bpreshuffle=True)
        self.assertEqual((cfg["libtype"], cfg["solidx"]), ("hipblaslt", -1))
        for otype in (BF16, FP32):
            cfg = self._config("gfx950", bpreshuffle=True, otype=otype)
            self.assertEqual(
                cfg,
                {
                    "bpreshuffle": True,
                    "libtype": "asm",
                    "solidx": 0,
                    "splitK": None,
                    "kernelName": None,
                },
            )

    def test_unshuffled_selection_is_unchanged(self):
        self.assertEqual(self._config("gfx1250")["libtype"], "triton")
        self.assertEqual(self._config("gfx950")["libtype"], "torch")
        self.assertEqual(self._config("gfx90a")["libtype"], "torch")


class TestTunedRows(_Base):
    ROW = {"libtype": "asm", "solidx": 3, "splitK": 1, "kernelName": "k"}

    def test_asm_row_on_an_arch_without_kernels_is_skipped(self):
        for gfx in NO_ASM_ARCHES:
            with self.subTest(gfx=gfx):
                table = {_row_key(gfx, False): self.ROW}
                cfg = self._config(gfx, table=table)
                self.assertNotEqual(cfg["libtype"], "asm")

    def test_preshuffled_asm_row_on_an_arch_without_kernels_raises(self):
        table = {_row_key("gfx1250", True): self.ROW}
        with self.assertRaisesRegex(RuntimeError, "preshuffled B"):
            self._config("gfx1250", bpreshuffle=True, table=table)

    def test_asm_row_on_gfx942_and_gfx950_is_returned(self):
        for gfx in ("gfx942", "gfx950"):
            with self.subTest(gfx=gfx):
                self.assertEqual(
                    self._config(gfx, table={_row_key(gfx, False): self.ROW}),
                    self.ROW,
                )


class TestAsmLaunch(unittest.TestCase):
    def _run(self, gfx):
        inp = torch.zeros(M, K, dtype=torch.bfloat16)
        weights = torch.zeros(N, K, dtype=torch.bfloat16)
        launch = mock.Mock(name="gemm_a16w16_asm")
        with mock.patch.object(
            tuned, "get_gfx_runtime", lambda: gfx
        ), mock.patch.object(tuned, "gemm_a16w16_asm", launch):
            tuned.asm_gemm(
                inp, weights, 0, otype=torch.bfloat16, config={"libtype": "asm"}
            )
        return launch

    def test_launch_is_refused_on_an_arch_without_kernels(self):
        for gfx in NO_ASM_ARCHES:
            with self.subTest(gfx=gfx):
                with mock.patch.object(tuned, "gemm_a16w16_asm") as launch:
                    with mock.patch.object(tuned, "get_gfx_runtime", lambda g=gfx: g):
                        with self.assertRaisesRegex(RuntimeError, "no bf16gemm"):
                            tuned.asm_gemm(
                                torch.zeros(M, K, dtype=torch.bfloat16),
                                torch.zeros(N, K, dtype=torch.bfloat16),
                                0,
                                otype=torch.bfloat16,
                                config={"libtype": "asm"},
                            )
                    launch.assert_not_called()

    def test_launch_proceeds_on_gfx942_and_gfx950(self):
        for gfx in ("gfx942", "gfx950"):
            with self.subTest(gfx=gfx):
                self._run(gfx).assert_called_once()


if __name__ == "__main__":
    unittest.main()
