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
from typing import ClassVar
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
        with (
            tempfile.TemporaryDirectory() as empty,
            tempfile.TemporaryDirectory() as full,
        ):
            Path(empty, "gfx9999", "bf16gemm").mkdir(parents=True)
            Path(full, "gfx9999", "bf16gemm").mkdir(parents=True)
            Path(full, "gfx9999", "bf16gemm", "bf16gemm_fp32bf16.csv").write_text(
                "knl_name\n"
            )
            self.assertFalse(has_asm_kernels("bf16gemm", "gfx9999", empty))
            self.assertTrue(has_asm_kernels("bf16gemm", "gfx9999", full))
            self.assertFalse(has_asm_kernels("bf16gemm", "gfx1250", full))

    def test_kernel_tables_in_subdirectories_count_like_in_codegen(self):
        # hsa/codegen.py globs `<gfx>/<family>/**/*.csv`.
        has_asm_kernels = tuned.has_asm_kernels
        self.assertTrue(has_asm_kernels("fmoe", "gfx950"))
        with tempfile.TemporaryDirectory() as root:
            nested = Path(root, "gfx9999", "fmoe", "silu")
            nested.mkdir(parents=True)
            Path(root, "gfx9999", "fmoe", "readme.txt").write_text("not a table\n")
            self.assertFalse(has_asm_kernels("fmoe", "gfx9999", root))
        with tempfile.TemporaryDirectory() as root:
            nested = Path(root, "gfx9999", "fmoe", "silu")
            nested.mkdir(parents=True)
            (nested / "fmoe_bf16.csv").write_text("knl_name\n")
            self.assertTrue(has_asm_kernels("fmoe", "gfx9999", root))


class TestDefaultSelection(_Base):
    def test_bpreshuffle_without_asm_kernels_raises_before_launch(self):
        for gfx in NO_ASM_ARCHES:
            with (
                self.subTest(gfx=gfx),
                self.assertRaisesRegex(RuntimeError, "preshuffled B.*" + gfx),
            ):
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
    ROW: ClassVar[dict] = {
        "libtype": "asm",
        "solidx": 3,
        "splitK": 1,
        "kernelName": "k",
    }

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
    """``asm_gemm`` checks the arch of ``inp.device``, which is where the C++
    launcher runs, not the node-wide ``get_gfx_runtime()``."""

    def _launch(self, device_arches, device, node_wide="gfx950"):
        """Run asm_gemm on ``device``; return the launcher mock."""
        arch_module = _module(
            "aiter.ops.opus._arch", _device_arch=lambda dev: device_arches[dev]
        )
        launch = mock.Mock(name="gemm_a16w16_asm")
        inp = types.SimpleNamespace(device=device, shape=(M, K))
        weights = types.SimpleNamespace(device=device, shape=(N, K))
        with (
            mock.patch.dict(sys.modules, {"aiter.ops.opus._arch": arch_module}),
            mock.patch.object(tuned, "get_gfx_runtime", lambda: node_wide),
            mock.patch.object(tuned, "gemm_a16w16_asm", launch),
            mock.patch.object(tuned, "_result_buffer", lambda *_a: object()),
        ):
            tuned.asm_gemm(
                inp, weights, 0, otype=torch.bfloat16, config={"libtype": "asm"}
            )
        return launch

    def test_launch_is_refused_on_a_device_without_kernels(self):
        dev = torch.device("cuda", 0)
        for gfx in NO_ASM_ARCHES:
            for node_wide in ("gfx950", gfx):
                with self.subTest(gfx=gfx, node_wide=node_wide):
                    launch = mock.Mock()
                    with (
                        mock.patch.object(tuned, "gemm_a16w16_asm", launch),
                        self.assertRaisesRegex(RuntimeError, "no bf16gemm.*" + gfx),
                    ):
                        self._launch({dev: gfx}, dev, node_wide)
                    launch.assert_not_called()

    def test_launch_proceeds_on_a_device_with_kernels(self):
        dev = torch.device("cuda", 0)
        for gfx in ("gfx942", "gfx950"):
            for node_wide in (gfx, "gfx1250"):
                with self.subTest(gfx=gfx, node_wide=node_wide):
                    self._launch({dev: gfx}, dev, node_wide).assert_called_once()

    def test_mixed_arch_node_follows_the_input_device(self):
        # cuda:0 is gfx950, cuda:1 is gfx1250; the node-wide arch is cuda:0's.
        dev0, dev1 = torch.device("cuda", 0), torch.device("cuda", 1)
        arches = {dev0: "gfx950", dev1: "gfx1250"}
        self._launch(arches, dev0).assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "no bf16gemm.*gfx1250.*cuda:1"):
            self._launch(arches, dev1)
        # And the other way round: the node-wide arch must not block cuda:0.
        swapped = {dev0: "gfx1250", dev1: "gfx950"}
        self._launch(swapped, dev1, node_wide="gfx1250").assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "no bf16gemm.*gfx1250.*cuda:0"):
            self._launch(swapped, dev0, node_wide="gfx1250")


def _load_tuner(build_arch):
    """csrc/gemm_a16w16/gemm_a16w16_tune.py over stubbed aiter, with the real
    ``asm_kernels``; ``build_arch`` is what ``get_gfx()`` (GPU_ARCHS) reports."""
    dtypes = _module("aiter.dtypes", bf16=torch.bfloat16, fp16=torch.float16)
    stubs = {
        "aiter": _module("aiter", dtypes=dtypes, logger=mock.Mock()),
        "aiter.dtypes": dtypes,
        "aiter.jit": _module("aiter.jit"),
        "aiter.jit.core": _module(
            "aiter.jit.core",
            AITER_ASM_DIR=str(_REPO_ROOT / "hsa") + "/",
            AITER_CONFIG_GEMM_BF16="",
        ),
        "aiter.jit.utils": _module("aiter.jit.utils"),
        "aiter.jit.utils.chip_info": _module(
            "aiter.jit.utils.chip_info",
            get_cu_num=lambda: 256,
            get_gfx=lambda: build_arch,
            get_gfx_runtime=lambda: build_arch,
        ),
        "aiter.ops": _module("aiter.ops"),
        "aiter.ops.flydsl": _module("aiter.ops.flydsl"),
        "aiter.ops.flydsl.gemm_a16w16_policy": _module(
            "aiter.ops.flydsl.gemm_a16w16_policy",
            get_flydsl_a16w16_configs=mock.Mock(),
        ),
        "aiter.ops.flydsl.gemm_kernels": _module(
            "aiter.ops.flydsl.gemm_kernels",
            flydsl_hgemm=mock.Mock(),
            flydsl_hgemm_kernel_name=mock.Mock(),
        ),
        "aiter.ops.gemm_op_a16w16": _module(
            "aiter.ops.gemm_op_a16w16", ASM_SPLITK_MAX_GRID=1 << 30
        ),
        "aiter.ops.opus": _module("aiter.ops.opus"),
        "aiter.ops.opus._arch": _module(
            "aiter.ops.opus._arch", _device_arch_and_cu=lambda _d: (build_arch, 256)
        ),
        "aiter.ops.triton": _module("aiter.ops.triton"),
        "aiter.ops.triton.gemm": _module("aiter.ops.triton.gemm"),
        "aiter.ops.triton.gemm.basic": _module("aiter.ops.triton.gemm.basic"),
        "aiter.ops.triton.gemm.basic.gemm_a16w16": _module(
            "aiter.ops.triton.gemm.basic.gemm_a16w16", gemm_a16w16=mock.Mock()
        ),
        "aiter.utility": _module("aiter.utility"),
        "aiter.utility.base_tuner": _module(
            "aiter.utility.base_tuner",
            GemmCommonTuner=type("GemmCommonTuner", (), {"ARG_DEFAULTS": {}}),
        ),
        "aiter.utility.mp_tuner": _module(
            "aiter.utility.mp_tuner", mp_tuner=mock.Mock()
        ),
    }
    saved_modules, saved_path = dict(sys.modules), list(sys.path)
    try:
        sys.modules.update(stubs)
        _load("aiter.jit.utils.asm_kernels", _AITER / "jit/utils/asm_kernels.py")
        return _load(
            "gemm_a16w16_tune_under_test",
            _REPO_ROOT / "csrc/gemm_a16w16/gemm_a16w16_tune.py",
        )
    finally:
        sys.modules.clear()
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path


class TestTunerAsmCandidates(unittest.TestCase):
    """The tuner keys rows by the live device arch (``info_keys[0]``); GPU_ARCHS,
    the build arch, must not decide whether ASM candidates exist."""

    KERNELS: ClassVar[dict] = {(64, 64, 0, 0, 64, 0, 0): ["knl"]}

    def _asm_tasks(self, device_arch, build_arch):
        tune = _load_tuner(build_arch)
        read = mock.Mock(return_value=self.KERNELS)
        info_keys = (device_arch, 256, M, N, K, False, BF16, BF16, False, False)
        with mock.patch.object(tune, "get_asm_kernels", read):
            tasks = tune.GemmA16W16Tuner._get_asm_tasks(
                None, info_keys, False, torch.bfloat16, torch.bfloat16, False, False, {}
            )
        return tasks, read

    def test_candidates_follow_the_device_arch_not_the_build_arch(self):
        tasks, read = self._asm_tasks("gfx950", build_arch="gfx1250")
        self.assertEqual(len(tasks), 1)
        csv_path = Path(read.call_args.args[0])
        self.assertEqual(
            csv_path.parts[-3:], ("gfx950", "bf16gemm", "bf16gemm_fp32bf16.csv")
        )
        self.assertTrue(csv_path.is_relative_to(_REPO_ROOT / "hsa"))

    def test_no_candidates_on_a_device_without_kernels_whatever_the_build_arch(self):
        for device_arch in NO_ASM_ARCHES:
            with self.subTest(device_arch=device_arch):
                tasks, read = self._asm_tasks(device_arch, build_arch="gfx950")
                self.assertEqual(tasks, [])
                read.assert_not_called()

    def test_arch_shape_conditions_use_the_device_arch(self):
        # gfx950 needs N % 64 == 0; the build arch (gfx942) must not relax it.
        tune = _load_tuner("gfx942")
        info_keys = ("gfx950", 256, M, 100, K, False, BF16, BF16, False, False)
        with mock.patch.object(tune, "get_asm_kernels", mock.Mock()) as read:
            tasks = tune.GemmA16W16Tuner._get_asm_tasks(
                None, info_keys, False, torch.bfloat16, torch.bfloat16, False, False, {}
            )
        self.assertEqual(tasks, [])
        read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
