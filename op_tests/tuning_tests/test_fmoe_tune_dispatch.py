# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU locks for FmoeTuner.tune() dispatch, generator filters, and CSV write.

The winner replay feeds post_process hand-made times. This file covers the
paths that replay never calls: which generators tune() runs, the opus and
FlyDSL shape filters, GroupedFmoeTuner and Mxfp4FlydslTuner candidate
selection, and result_to_csv. Generators are called for real where the check
is the task list. The GPU timing entry, mp_tuner, is replaced.
"""

import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Same order as the key list in aiter.tuning.moe.fused_moe.main().
KEYS = [
    "gfx",
    "cu_num",
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "use_g1u1",
    "doweight_stage1",
]
RESULT_LIST = [
    "block_m",
    "ksplit",
    "us1",
    "kernelName1",
    "err1",
    "us2",
    "kernelName2",
    "err2",
    "us",
    "run_1stage",
    "xbf16",
    "flat",
    "tflops",
    "bw",
]
GENERATORS = (
    "gen_2stages_asm1_task",
    "gen_2stages_task",
    "gen_flydsl_2stages_task",
    "gen_flydsl_v2_2stages_task",
    "gen_flydsl_i4_2stages_task",
    "gen_opus_2stages_task",
    "gen_1stage_asm_task",
)
_TUNER_ENV = (
    "TUNE_ONLY",
    "OPUS_ONLY",
    "OPUS_SKIP_CKTILE",
    "TUNE_MOE_KERNEL_REGEX",
    "AITER_FMOE_TUNE_EXCLUDE_KERNELS",
)
BLOCK_MS = [16, 32, 64, 128]


def _task(name):
    return (("info", "stage1", name, 32), None)


def _kernel_names(tasks):
    return [task[0][2] for task in tasks]


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None, "gemm_moe_tune.py needs torch"
)
class TestFmoeTuneDispatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        from aiter import ActivationType, QuantType, dtypes
        from aiter.fused_moe import AsmMoeMode
        from aiter.tuning.moe import fused_moe
        from aiter.tuning.moe.backends import (
            asm_1stage,
            asm_stage1,
            common,
            flydsl,
            opus,
        )
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        cls.ActivationType = ActivationType
        cls.AsmMoeMode = AsmMoeMode
        cls.QuantType = QuantType
        cls.dtypes = dtypes
        cls.torch = torch
        cls.gemm_moe_tune = gemm_moe_tune
        cls.fused_moe = fused_moe
        cls.backends_common = common
        cls.backend_asm_stage1 = asm_stage1
        cls.backend_asm_1stage = asm_1stage
        cls.backend_flydsl = flydsl
        cls.backend_opus = opus

    def setUp(self):
        m = self.gemm_moe_tune
        self.tuner = m.FmoeTuner(
            "dispatchFmoe", KEYS, RESULT_LIST + ["nt"], "tune dispatch"
        )
        with mock.patch.object(sys, "argv", ["gemm_moe_tune.py"]):
            self.args = self.tuner.parse_args()

    def untuned(self, **overrides):
        row = {
            "gfx": "gfx950",
            "cu_num": 256,
            "token": 32,
            "model_dim": 4096,
            "inter_dim": 1024,
            "expert": 8,
            "topk": 2,
            "act_type": "ActivationType.Silu",
            "dtype": "torch.bfloat16",
            "q_dtype_a": "torch.bfloat16",
            "q_dtype_w": "torch.bfloat16",
            "q_type": "QuantType.No",
            "use_g1u1": 1,
            "doweight_stage1": 0,
        }
        row.update(overrides)
        return pd.DataFrame([row])[KEYS]

    def run_tune(self, frame, env=None, gfx="gfx950", returns=None):
        """Call tune() with every generator replaced. returns maps a generator
        name to the tasks it yields; the rest yield nothing."""
        returns = {} if returns is None else returns
        calls = []
        got = {}

        def make(name):
            def fn(*args, **kwargs):
                calls.append(name)
                return list(returns.get(name, []))

            return fn

        def fake_mp(tasks, *args, **kwargs):
            got["tasks"] = list(tasks)
            got["default_device"] = kwargs.get("default_device")
            return []

        cleaned = {k: v for k, v in os.environ.items() if k not in _TUNER_ENV}
        cleaned.update(env or {})
        with (
            mock.patch.object(self.fused_moe, "get_gfx", return_value=gfx),
            mock.patch.object(self.fused_moe, "mp_tuner", fake_mp),
            mock.patch.dict(os.environ, cleaned, clear=True),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.ExitStack() as stack,
        ):
            for name in GENERATORS:
                stack.enter_context(mock.patch.object(self.tuner, name, make(name)))
            out = self.tuner.tune(frame, None, self.args)
        if "default_device" in got:
            self.assertEqual(got["default_device"], "cuda")
        return out, calls, got

    def info(
        self,
        *,
        token=1,
        model_dim=5120,
        inter_dim=1152,
        q_dtype_a=None,
        q_dtype_w=None,
        q_type=None,
        act_type=None,
        doweight_stage1=0,
        dtype=None,
        gfx="gfx950",
    ):
        return (
            gfx,
            256,
            token,
            model_dim,
            inter_dim,
            8,
            2,
            self.ActivationType.Silu if act_type is None else act_type,
            self.torch.bfloat16 if dtype is None else dtype,
            self.dtypes.fp8 if q_dtype_a is None else q_dtype_a,
            self.dtypes.fp4x2 if q_dtype_w is None else q_dtype_w,
            self.QuantType.per_1x32 if q_type is None else q_type,
            1,
            doweight_stage1,
        )

    def test_per_1x32_off_gfx950_returns_no_tasks(self):
        frame = self.untuned(q_type="QuantType.per_1x32")
        out, calls, got = self.run_tune(frame, gfx="gfx942")
        self.assertEqual(out, [])
        self.assertEqual(calls, [])
        self.assertNotIn("tasks", got)
        # The same shape on gfx950 still reaches the generators.
        out, calls, got = self.run_tune(frame, gfx="gfx950")
        self.assertEqual(out, [])
        self.assertIn("gen_flydsl_v2_2stages_task", calls)

    def test_g1u0_is_skipped(self):
        out, calls, got = self.run_tune(self.untuned(use_g1u1=0))
        self.assertEqual(out, [])
        self.assertEqual(calls, [])
        self.assertNotIn("tasks", got)

    def test_default_tune_only_stops_at_flydsl_v2_when_it_has_tasks(self):
        returns = {"gen_flydsl_v2_2stages_task": [_task("v2_kernel")]}
        out, calls, got = self.run_tune(self.untuned(), returns=returns)
        self.assertEqual(out, [])
        self.assertEqual(calls, ["gen_flydsl_v2_2stages_task"])
        self.assertEqual(_kernel_names(got["tasks"]), ["v2_kernel"])

    def test_default_tune_only_runs_every_generator_when_v2_is_empty(self):
        returns = {name: [_task(name)] for name in GENERATORS}
        returns["gen_flydsl_v2_2stages_task"] = []
        out, calls, got = self.run_tune(self.untuned(), returns=returns)
        self.assertEqual(out, [])
        # The v2 probe is the first call. An empty result is reused, not
        # generated a second time, and the other six generators follow.
        self.assertEqual(calls[0], "gen_flydsl_v2_2stages_task")
        self.assertEqual(calls.count("gen_flydsl_v2_2stages_task"), 1)
        self.assertEqual(set(calls), set(GENERATORS))
        self.assertEqual(
            _kernel_names(got["tasks"]),
            [name for name in GENERATORS if name != "gen_flydsl_v2_2stages_task"],
        )

    def test_opus_only_and_skip_cktile_select_generators(self):
        returns = {name: [_task(name)] for name in GENERATORS}
        with self.subTest("OPUS_ONLY"):
            out, calls, got = self.run_tune(
                self.untuned(),
                env={"TUNE_ONLY": "", "OPUS_ONLY": "1"},
                returns=returns,
            )
            self.assertEqual(calls, ["gen_opus_2stages_task"])
            self.assertEqual(_kernel_names(got["tasks"]), ["gen_opus_2stages_task"])
            self.assertEqual(out, [])
        with self.subTest("OPUS_SKIP_CKTILE skips gen_2stages_task"):
            out, calls, got = self.run_tune(
                self.untuned(),
                env={"TUNE_ONLY": "", "OPUS_SKIP_CKTILE": "1"},
                returns=returns,
            )
            self.assertNotIn("gen_2stages_task", calls)
            self.assertIn("gen_opus_2stages_task", calls)
            self.assertNotIn("gen_2stages_task", _kernel_names(got["tasks"]))

    def test_kernel_regex_filters_dispatched_tasks(self):
        returns = {name: [_task("drop_me")] for name in GENERATORS}
        returns["gen_opus_2stages_task"] = [_task("keep_me")]
        out, calls, got = self.run_tune(
            self.untuned(),
            env={"TUNE_ONLY": "", "TUNE_MOE_KERNEL_REGEX": "keep_me"},
            returns=returns,
        )
        self.assertEqual(out, [])
        self.assertEqual(set(calls), set(GENERATORS))
        self.assertEqual(_kernel_names(got["tasks"]), ["keep_me"])

    def test_opus_stage1_rejects_a_model_dim_that_misses_the_mfma_tile(self):
        # Stage1 needs model_dim % 128 == 0. 5184 % 128 == 64, and 1152 still
        # passes the inter_dim gate, so only stage1 disappears.
        bad = self.tuner.gen_opus_2stages_task(
            self.info(model_dim=5184, inter_dim=1152), BLOCK_MS
        )
        good = self.tuner.gen_opus_2stages_task(
            self.info(model_dim=5120, inter_dim=1152, token=1), BLOCK_MS
        )
        self.assertEqual([t[0][1] for t in bad if t[0][1] == "stage1"], [])
        self.assertTrue(any(t[0][1] == "stage1" for t in good))
        # inter_dim % 32 == 0 lives in the stage1 helper. 1168 % 32 == 16.
        # The tuner also rejects it earlier (inter_dim % 128), so both ends
        # of that chain have to stay.
        self.assertEqual(
            self.tuner.gen_opus_2stages_task(self.info(inter_dim=1168), BLOCK_MS),
            [],
        )
        self.assertEqual(
            self.backend_opus.opus_a8w4_stage1_instances_for_shape(
                model_dim=5120, logical_inter_dim=1168, inter_dim_pad=0
            ),
            (),
        )
        self.assertTrue(
            self.backend_opus.opus_a8w4_stage1_instances_for_shape(
                model_dim=5120, logical_inter_dim=1152, inter_dim_pad=0
            )
        )

    def test_opus_stage1_gates_leave_stage2_in_place(self):
        # stage1_supported only guards the stage1 loop. An unsupported
        # activation or doweight_stage1 drops stage1 and keeps every stage2.

        def by_stage(tasks):
            return {
                stage: sorted(t[0][2:4] for t in tasks if t[0][1] == stage)
                for stage in ("stage1", "stage2")
            }

        base = by_stage(self.tuner.gen_opus_2stages_task(self.info(), BLOCK_MS))
        self.assertTrue(base["stage1"])
        self.assertTrue(base["stage2"])
        cases = {
            "Gelu": self.info(act_type=self.ActivationType.Gelu),
            "doweight_stage1": self.info(doweight_stage1=1),
        }
        for name, info in cases.items():
            with self.subTest(name):
                got = by_stage(self.tuner.gen_opus_2stages_task(info, BLOCK_MS))
                self.assertEqual(got["stage1"], [])
                self.assertEqual(got["stage2"], base["stage2"])

    def test_opus_stage2_drops_a_block_n_that_misses_model_dim(self):
        # 5248 = 41 * 128: a multiple of block_n 64 and 128, not of 256.
        catalog = self.backend_opus.get_opus_a8w4_stage2_kernels(token=1)
        by_block_n = {}
        for name, kparams in catalog.items():
            by_block_n.setdefault(int(kparams["kernel_block_n"]), set()).add(name)
        self.assertEqual(set(by_block_n), {64, 128, 256})

        def stage2(model_dim):
            tasks = self.tuner.gen_opus_2stages_task(
                self.info(model_dim=model_dim), BLOCK_MS
            )
            return {t[0][2] for t in tasks if t[0][1] == "stage2" and len(t[0]) == 4}

        self.assertEqual(stage2(5120), set(catalog))
        self.assertEqual(stage2(5248), by_block_n[64] | by_block_n[128])

    def test_opus_stage2_kernels_depend_on_token(self):
        small = self.tuner.gen_opus_2stages_task(self.info(token=1), BLOCK_MS)
        large = self.tuner.gen_opus_2stages_task(self.info(token=2048), BLOCK_MS)

        def stage2(tasks):
            return {t[0][2] for t in tasks if t[0][1] == "stage2"}

        self.assertTrue(stage2(small))
        self.assertTrue(stage2(large))
        self.assertNotEqual(stage2(small), stage2(large))

    def test_flydsl_stage1_drops_kernels_whose_k_does_not_divide(self):
        kernels = self.backend_flydsl.get_flydsl_stage1_kernels("fp4", "fp4", "bf16")

        def divides(kparams, model_dim):
            k_batch = kparams.get("k_batch", 1) or 1
            tile_k = kparams["tile_k"]
            k_wave = kparams.get("k_wave", 1) or 1
            if model_dim % k_batch != 0:
                return False
            per_batch = model_dim // k_batch
            if per_batch % tile_k != 0:
                return False
            return not (
                k_wave > 1
                and (per_batch % k_wave != 0 or (per_batch // k_wave) % tile_k != 0)
            )

        model_dim = None
        rejected = []
        for candidate in (100, 96, 80, 48, 7):
            rejected = [
                name
                for name, kparams in kernels.items()
                if kparams.get("tile_m") in (16, 32, 64, 128)
                and not divides(kparams, candidate)
            ]
            if rejected:
                model_dim = candidate
                break
        self.assertIsNotNone(
            model_dim, "no catalog kernel misses these model_dim values"
        )
        tasks = self.tuner.gen_flydsl_2stages_task(
            self.info(
                model_dim=model_dim,
                inter_dim=256,
                q_dtype_a=self.dtypes.fp4x2,
                q_dtype_w=self.dtypes.fp4x2,
            ),
            BLOCK_MS,
        )
        emitted = {t[0][2] for t in tasks if t[0][1] == "stage1"}
        leaked = [
            name
            for name in rejected
            if name in emitted or f"{name}_fp4" in emitted or f"{name}_fp8" in emitted
        ]
        self.assertEqual(leaked, [])

    def a16w16(self, **overrides):
        bf16 = self.dtypes.bf16
        return self.info(
            q_dtype_a=bf16, q_dtype_w=bf16, q_type=self.QuantType.No, **overrides
        )

    def a16w4(self, **overrides):
        return self.info(
            model_dim=3072,
            q_dtype_a=self.dtypes.bf16,
            q_dtype_w=self.dtypes.fp4x2,
            **overrides,
        )

    def flydsl_stage2_params(self, info):
        tasks = self.tuner.gen_flydsl_2stages_task(info, BLOCK_MS)
        # Stage2 run args are (keys, dtype, topk, kparams, blockM, ...).
        return [t[4][3] for t in tasks if t[0][1] == "stage2"]

    def test_flydsl_a16w16_gate(self):
        gen = self.tuner.gen_flydsl_2stages_task
        self.assertTrue(gen(self.a16w16(), BLOCK_MS))
        cases = {
            "Gelu": self.a16w16(act_type=self.ActivationType.Gelu),
            "doweight_stage1": self.a16w16(doweight_stage1=1),
            "gfx90a": self.a16w16(gfx="gfx90a"),
        }
        for name, info in cases.items():
            with self.subTest(name):
                self.assertEqual(gen(info, BLOCK_MS), [])

    def test_flydsl_bf16_activation_stage2_tile_k_divides_inter_dim(self):
        # 384 = 3 * 128: tile_k 128 divides it, tile_k 256 does not.
        def tile_ks(inter_dim):
            params = self.flydsl_stage2_params(self.a16w4(inter_dim=inter_dim))
            return {p["tile_k"] for p in params}

        self.assertIn(256, tile_ks(3072))
        self.assertTrue(tile_ks(384))
        self.assertNotIn(256, tile_ks(384))

    def test_flydsl_bf16_activation_stage2_fits_lds(self):
        limit = 160 * 1024

        def lds(p):
            return p["tile_m"] * (p["tile_k"] * 2 + p["tile_n"] * 4)

        catalog = self.backend_flydsl.get_flydsl_stage2_kernels("bf16", "fp4", "bf16")
        self.assertTrue(
            any(lds(p) > limit for p in catalog.values()),
            "no catalog kernel exceeds the LDS limit; the filter is not exercised",
        )
        params = self.flydsl_stage2_params(self.a16w4(inter_dim=3072))
        self.assertTrue(params)
        self.assertEqual([p for p in params if lds(p) > limit], [])

    def test_flydsl_v2_skips_block_m_16(self):
        tasks = self.tuner.gen_flydsl_v2_2stages_task(
            self.info(
                model_dim=4096,
                inter_dim=1024,
                q_dtype_a=self.dtypes.fp4x2,
                q_dtype_w=self.dtypes.fp4x2,
            ),
            BLOCK_MS,
        )
        self.assertTrue(tasks)
        self.assertNotIn(16, {t[0][3] for t in tasks})

    def test_flydsl_v2_rejects_doweight_and_non_bf16_output(self):
        gen = self.tuner.gen_flydsl_v2_2stages_task
        self.assertTrue(gen(self.info(), BLOCK_MS))
        self.assertEqual(gen(self.info(doweight_stage1=1), BLOCK_MS), [])
        self.assertEqual(gen(self.info(dtype=self.torch.float16), BLOCK_MS), [])

    def test_flydsl_v2_stage2_tile_n_must_divide_model_dim(self):
        # 5248 = 41 * 128: tile_n 128 divides it, tile_n 256 does not.
        def tile_ns(model_dim):
            tasks = self.tuner.gen_flydsl_v2_2stages_task(
                self.info(model_dim=model_dim), BLOCK_MS
            )
            # Stage2 run args end with the kernel params.
            return {t[4][-1]["tile_n"] for t in tasks if t[0][1] == "stage2"}

        self.assertEqual(tile_ns(5120), {128, 256})
        self.assertEqual(tile_ns(5248), {128})

    def test_default_tune_only_keeps_v2_without_stage2(self):
        # Current behavior, not a requirement: v2 stage1 has no inter_dim
        # filter, so with inter_dim % 128 != 0 the default mode keeps v2 alone
        # and submits stage1 with no stage2 to pair. TUNE_ONLY="" still
        # pairs other backends.
        frame = self.untuned(
            token=32,
            model_dim=5120,
            inter_dim=1088,
            q_dtype_a=str(self.dtypes.fp8),
            q_dtype_w=str(self.dtypes.fp4x2),
            q_type="QuantType.per_1x32",
        )

        def block_ms(tasks, stage):
            return {t[0][3] for t in tasks if t[0][1] == stage}

        default = self.submitted(frame, {})
        self.assertTrue(default)
        self.assertEqual({t[0][1] for t in default}, {"stage1"})
        self.assertTrue(all(t[0][5] == 1 for t in default))
        every = self.submitted(frame, {"TUNE_ONLY": ""})
        self.assertTrue(block_ms(every, "stage1") & block_ms(every, "stage2"))

    def test_ck_rejects_bf16_activation_with_fp4_weights(self):
        # Silu, so the Swiglu + fp4 early return does not hide this one.
        gen = self.tuner.gen_2stages_task
        fp4 = self.dtypes.fp4x2
        self.assertTrue(gen(self.info(q_dtype_a=fp4, q_dtype_w=fp4), BLOCK_MS))
        self.assertEqual(
            gen(self.info(q_dtype_a=self.dtypes.bf16, q_dtype_w=fp4), BLOCK_MS), []
        )

    def test_exclude_kernels_filters_ck_stage2_only(self):
        # _TUNE_EXCLUDE_KERNEL_PATTERNS is read from the env at import, so the
        # test patches the parsed list.
        fp8 = self.dtypes.fp8
        info = self.info(
            inter_dim=2560,
            q_dtype_a=fp8,
            q_dtype_w=fp8,
            q_type=self.QuantType.per_Token,
        )

        def names(patterns):
            with mock.patch.object(
                self.backends_common, "_TUNE_EXCLUDE_KERNEL_PATTERNS", patterns
            ):
                tasks = self.tuner.gen_2stages_task(info, BLOCK_MS)
            return {
                stage: sorted(t[0][2] for t in tasks if t[0][1] == stage)
                for stage in ("stage1", "stage2")
            }

        base = names([])
        self.assertTrue(base["stage1"])
        self.assertTrue(base["stage2"])
        with self.subTest("a stage2 name is excluded"):
            pattern = base["stage2"][0]
            got = names([pattern])
            self.assertEqual(got["stage1"], base["stage1"])
            self.assertEqual(
                got["stage2"], [n for n in base["stage2"] if pattern not in n]
            )
        with self.subTest("a stage1 name is not"):
            pattern = base["stage1"][0]
            got = names([pattern])
            self.assertEqual(got["stage1"], base["stage1"])

    def test_tune_only_cktile_covers_ck_and_cktile(self):
        # TUNE_ONLY=cktile and OPUS_SKIP_CKTILE gate CK and CK-tile together.
        # Real generators, so this still holds once gen_2stages_task splits.
        fp8 = str(self.dtypes.fp8)
        frames = {
            "ck": self.untuned(
                model_dim=5120,
                inter_dim=2560,
                q_dtype_a=fp8,
                q_dtype_w=fp8,
                q_type="QuantType.per_Token",
            ),
            "cktile": self.untuned(
                model_dim=5120,
                inter_dim=1152,
                q_dtype_a=fp8,
                q_dtype_w=str(self.dtypes.fp4x2),
                q_type="QuantType.per_1x32",
            ),
        }
        for name, frame in frames.items():
            with self.subTest(name):
                kernels = {
                    t[0][2] for t in self.submitted(frame, {"TUNE_ONLY": "cktile"})
                }
                self.assertTrue(kernels)
                is_cktile = {k.startswith("cktile_") for k in kernels}
                self.assertEqual(is_cktile, {name == "cktile"})
                skipped = self.submitted(
                    frame, {"TUNE_ONLY": "cktile", "OPUS_SKIP_CKTILE": "1"}
                )
                self.assertEqual(skipped, [])

    @contextlib.contextmanager
    def asm_gfx950(self, module, *, pin_gfx):
        """ASM kernel lists come from the running GPU's arch; pin gfx950's in
        the module whose generator runs. pin_gfx for generators that call
        get_gfx() as well as get_asm_dir()."""
        asm_dir = str(REPO_ROOT / "hsa" / "gfx950")
        with contextlib.ExitStack() as stack:
            if pin_gfx:
                stack.enter_context(
                    mock.patch.object(module, "get_gfx", return_value="gfx950")
                )
            stack.enter_context(
                mock.patch.object(module, "get_asm_dir", return_value=asm_dir)
            )
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            yield

    def test_asm_stage1_reads_the_int8_list_and_honours_exclusion(self):
        i8 = self.dtypes.i8
        info = self.info(
            inter_dim=2560, q_dtype_a=i8, q_dtype_w=i8, q_type=self.QuantType.per_Token
        )
        listed = set(
            pd.read_csv(
                REPO_ROOT
                / "hsa/gfx950/fmoe_2stages/fmoe_stage1_bf16_pertokenInt8_g1u1.csv"
            )["knl_name"]
        )

        def names(patterns):
            with (
                self.asm_gfx950(self.backend_asm_stage1, pin_gfx=False),
                mock.patch.object(
                    self.backends_common, "_TUNE_EXCLUDE_KERNEL_PATTERNS", patterns
                ),
            ):
                tasks = self.tuner.gen_2stages_asm1_task(info, BLOCK_MS)
            return {t[0][2] for t in tasks}

        base = names([])
        self.assertTrue(base)
        self.assertLessEqual(base, listed)
        dropped = min(base)
        self.assertEqual(names([dropped]), {n for n in base if dropped not in n})

    def test_asm_1stage_per_1x32_keeps_only_post_reduce_at_inter_dim_tile_n(self):
        # per_1x32 has no 1-stage fmoe_func, so only POST_REDUCE kernels run,
        # and only for fp4 activations with inter_dim equal to their tile_n.
        fp4 = self.dtypes.fp4x2
        post_reduce = self.AsmMoeMode.POST_REDUCE.value

        def tags(**overrides):
            with self.asm_gfx950(self.backend_asm_1stage, pin_gfx=True):
                tasks = self.tuner.gen_1stage_asm_task(
                    self.info(model_dim=7168, **overrides)
                )
            return {(t[0][2], t[0][4]) for t in tasks}

        kept = tags(inter_dim=256, q_dtype_a=fp4, q_dtype_w=fp4)
        self.assertTrue(kept)
        self.assertEqual({flat for _, flat in kept}, {post_reduce})
        cases = {
            "inter_dim 512, tile_n 256": {
                "inter_dim": 512,
                "q_dtype_a": fp4,
                "q_dtype_w": fp4,
            },
            "fp8 activation": {"inter_dim": 256},
        }
        for name, overrides in cases.items():
            with self.subTest(name):
                self.assertEqual(tags(**overrides), set())

    def submitted(self, frame, env):
        """Tasks the real tune() hands to mp_tuner for frame under env."""
        got = {}

        def fake_mp(tasks, *args, **kwargs):
            got["tasks"] = list(tasks)
            got["default_device"] = kwargs.get("default_device")
            return []

        cleaned = {k: v for k, v in os.environ.items() if k not in _TUNER_ENV}
        cleaned.update(env)
        with (
            mock.patch.object(self.fused_moe, "get_gfx", return_value="gfx950"),
            mock.patch.object(
                self.backend_asm_1stage, "get_gfx", return_value="gfx950"
            ),
            mock.patch.object(self.fused_moe, "mp_tuner", fake_mp),
            mock.patch.dict(os.environ, cleaned, clear=True),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.tuner.tune(frame, None, self.args)
        if "default_device" in got:
            self.assertEqual(got["default_device"], "cuda")
        return got.get("tasks", [])

    def pre_processed(self, tuner, untuned_rows, tuned_rows, **flags):
        """Untuned rows tuner.pre_process keeps when tuned_rows are on file."""
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                all=False,
                last=False,
                verbose=False,
                untune_file=os.path.join(tmp, "untuned.csv"),
                tune_file=os.path.join(tmp, "tuned.csv"),
            )
            vars(args).update(flags)
            pd.DataFrame(untuned_rows).to_csv(args.untune_file, index=False)
            pd.DataFrame(tuned_rows).to_csv(args.tune_file, index=False)
            with (
                mock.patch.object(tuner, "get_gfx", return_value="gfx950"),
                mock.patch.object(tuner, "get_cu_num", return_value=256),
                mock.patch.dict(os.environ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                os.environ.pop("AITER_MXFP4_TUNE_SWIGLU_LIMIT", None)
                tuner.pre_process(args)
        return tuner.untunedf

    def test_fmoe_pre_process_requires_a_named_kernel_to_skip_a_shape(self):
        m = self.gemm_moe_tune
        grouped = m.GroupedFmoeTuner(
            "groupedFmoe",
            KEYS + ["gate_mode", "ep_fused"],
            ["kernelName1", "kernelName2", "us"],
            "grouped",
        )
        shape = {
            **self.untuned().iloc[0].to_dict(),
            "gate_mode": "GateMode.SEPARATED",
            "ep_fused": 0,
        }

        def row(token, **result):
            return {**shape, "token": token, **result}

        for tuner in (self.tuner, grouped):
            with self.subTest(tuner.name):
                left = self.pre_processed(
                    tuner,
                    [row(t) for t in range(1, 11)],
                    [
                        row(1, kernelName1="", kernelName2=""),
                        row(2, kernelName1=None, kernelName2=None),
                        row(3, kernelName1=" \t", kernelName2="  "),
                        row(4, kernelName1="k1", run_1stage=1),
                        row(5, kernelName2="k2", us=-1),
                        row(6, kernelName1="k1", kernelName2="k2", us=None),
                        row(7),
                        # No tuned row for token=8.
                        row(9, kernelName1="", kernelName2=""),
                        row(9, kernelName1="k1"),
                        row(10, gfx="gfx942", cu_num=304, kernelName1="k1"),
                    ],
                )
                self.assertEqual(list(left["token"]), [1, 2, 3, 7, 8, 10])

    def test_fmoe_pre_process_retunes_rows_with_no_kernel_columns(self):
        shape = self.untuned().iloc[0].to_dict()
        left = self.pre_processed(self.tuner, [shape], [shape])
        self.assertEqual(list(left["token"]), [shape["token"]])

    def test_fmoe_pre_process_all_and_last_still_retune_named_rows(self):
        shape = self.untuned().iloc[0].to_dict()
        rows = [{**shape, "token": t} for t in (1, 2)]
        tuned = [{**row, "kernelName1": "k1", "kernelName2": "k2"} for row in rows]
        for flags, expected in (({"all": True}, [1, 2]), ({"last": True}, [2])):
            with self.subTest(flags=flags):
                left = self.pre_processed(self.tuner, rows, tuned, **flags)
                self.assertEqual(list(left["token"]), expected)

    def test_grouped_pre_process_skips_shapes_tuned_for_the_same_gate_and_ep(self):
        m = self.gemm_moe_tune
        tuner = m.GroupedFmoeTuner(
            "groupedFmoe",
            KEYS + ["gate_mode", "ep_fused"],
            ["kernelName1", "kernelName2", "us"],
            "grouped",
        )
        shape = self.untuned(q_dtype_a="torch.float4_e2m1fn_x2").iloc[0].to_dict()

        def row(gate, ep_fused):
            return {**shape, "gate_mode": f"GateMode.{gate}", "ep_fused": ep_fused}

        tuned = {**row("INTERLEAVE", 0), "kernelName1": "k1", "kernelName2": "k2"}
        left = self.pre_processed(
            tuner,
            [row("INTERLEAVE", 0), row("INTERLEAVE", 1), row("SEPARATED", 0)],
            [tuned],
        )
        self.assertEqual(
            list(zip(left["gate_mode"], left["ep_fused"])),
            [("GateMode.INTERLEAVE", 1), ("GateMode.SEPARATED", 0)],
        )

    def test_mxfp4_pre_process_retunes_placeholders_and_unsupported_rows(self):
        # FmoeTuner.pre_process calls the _already_tuned_mask hook; MXFP4's
        # counts only tuned rows naming a kernel, and never unsupported ones.
        m = self.gemm_moe_tune
        tuner = m.Mxfp4FlydslTuner("mxfp4Fmoe", KEYS, RESULT_LIST, "mxfp4")
        fp4 = str(self.dtypes.fp4x2)
        supported = self.untuned(
            q_dtype_a=fp4, q_dtype_w=fp4, q_type=str(self.QuantType.per_1x32)
        )
        shape = supported.iloc[0].to_dict()
        tuned = {"kernelName1": "k1", "kernelName2": "k2"}
        unsupported = {**shape, "token": 4, "q_type": str(self.QuantType.per_Token)}
        left = self.pre_processed(
            tuner,
            [{**shape, "token": t} for t in (1, 2, 3)] + [unsupported],
            [
                {**shape, "token": 1, **tuned},
                {**shape, "token": 2, "kernelName1": "", "kernelName2": ""},
                {**unsupported, **tuned},
            ],
        )
        self.assertEqual(list(left["token"]), [2, 3, 4])

    def test_inherited_cli_checks_for_grouped_and_mxfp4(self):
        m = self.gemm_moe_tune
        generic = self.tuner
        grouped = m.GroupedFmoeTuner("groupedFmoe", KEYS, ["us"], "grouped")
        mxfp4 = m.Mxfp4FlydslTuner("mxfp4Fmoe", KEYS, RESULT_LIST, "mxfp4")

        def parse(tuner, *flags):
            argv = ["gemm_moe_tune.py", *flags]
            with (
                mock.patch.object(sys, "argv", argv),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                return tuner.parse_args()

        args = parse(mxfp4, "--mxfp4-flydsl")
        self.assertEqual(
            (args.mxfp4_search_mode, args.mxfp4_screen_topk), ("staged", 2)
        )
        self.assertTrue(parse(grouped, "--grouped-gemm").grouped_gemm)
        rejected = {
            "grouped with a search mode": (
                grouped,
                ("--grouped-gemm", "--mxfp4-search-mode", "full"),
            ),
            "generic tuner with the mxfp4 flag": (
                generic,
                ("--mxfp4-flydsl", "--mxfp4-search-mode", "full"),
            ),
            "screen top-k of 0": (
                mxfp4,
                ("--mxfp4-flydsl", "--mxfp4-screen-topk", "0"),
            ),
            "screen top-k with full search": (
                mxfp4,
                (
                    "--mxfp4-flydsl",
                    "--mxfp4-search-mode",
                    "full",
                    "--mxfp4-screen-topk",
                    "3",
                ),
            ),
        }
        for name, (tuner, flags) in rejected.items():
            with self.subTest(name), self.assertRaises(SystemExit):
                parse(tuner, *flags)

    def test_grouped_candidate_names_follow_gate_and_tile(self):
        tuner = self.gemm_moe_tune.GroupedFmoeTuner(
            "groupedFmoe", KEYS, ["kernelName1", "us"], "grouped"
        )
        row = {
            "token": 100,
            "expert": 4,
            "topk": 8,
            "q_dtype_a": "torch.float4_e2m1fn_x2",
            "gate_mode": "GateMode.INTERLEAVE",
        }
        rows = tuner._candidate_rows(row)
        self.assertEqual([item["tile_m"] for item in rows], [16, 32, 64, 128])
        # max_m_raw = (100 * 8 + 4 - 1) // 4 = 200; tile 64 rounds it to 256.
        self.assertEqual(rows[2]["max_m"], 256)
        self.assertEqual(rows[2]["kernelName1"], "grouped_gemm1_fp4_gugu")
        self.assertEqual(rows[2]["kernelName2"], "grouped_gemm2_fp4")
        separated = tuner._candidate_row(
            {
                **row,
                "q_dtype_a": "torch.float8_e4m3fn",
                "gate_mode": "GateMode.SEPARATED",
            },
            16,
        )
        self.assertEqual(separated["kernelName1"], "grouped_gemm1_a8w4_gguu")

    def test_grouped_and_mxfp4_post_process_keep_watts(self):
        m = self.gemm_moe_tune
        cases = (
            m.GroupedFmoeTuner("groupedFmoe", KEYS, RESULT_LIST, "grouped"),
            m.Mxfp4FlydslTuner("mxfp4Fmoe", KEYS, RESULT_LIST, "mxfp4"),
        )
        for tuner in cases:
            with self.subTest(tuner.name):
                kept = tuner.post_process(
                    [{"us": 1.0, "kernelName1": "k", "watts": 3.5}], None
                )
                self.assertAlmostEqual(float(kept.iloc[0]["watts"]), 3.5)
                plain = tuner.post_process([{"us": 1.0, "kernelName1": "k"}], None)
                self.assertNotIn("watts", plain.columns)

    def test_mxfp4_finalist_stays_inside_the_latency_band(self):
        from aiter.tuning.moe import mxfp4

        def cand(name, us, err):
            return {
                "kernelName1": name,
                "kernelName2": name + "_s2",
                "us": us,
                "_err": err,
            }

        fast = cand("fast", 100.0, 0.05)
        close = cand("close", 100.5, 0.01)  # 0.5% from the fastest, inside 1%
        far = cand("far", 102.0, 0.0)  # 2% out, but the most accurate
        self.assertEqual(
            mxfp4._select_mxfp4_finalist([far, fast, close])["kernelName1"], "close"
        )
        percent = {
            "kernelName1": "pct",
            "kernelName2": "pct_s2",
            "us": 100.0,
            "err1": "1.0%",
        }
        raw = cand("raw", 100.0, 0.02)
        self.assertEqual(
            mxfp4._select_mxfp4_finalist([raw, percent])["kernelName1"], "pct"
        )
        finalists = mxfp4._select_mxfp4_performance_finalists([far, fast, close], 1)
        self.assertEqual(finalists[0]["kernelName1"], "fast")
        # far is outside the band, so promotion is what keeps its accuracy.
        promoted = mxfp4._select_mxfp4_accuracy_promotion([fast, close, far], finalists)
        self.assertEqual(promoted["kernelName1"], "far")
        self.assertIsNone(mxfp4._select_mxfp4_accuracy_promotion([fast], finalists))

    def test_result_to_csv_tag_rules_differ_between_fmoe_and_mxfp4(self):
        # The runtime skips only flydsl_fallback rows (fused_moe.py). FmoeTuner
        # still drops every tagged row and the column; MXFP4 drops only those.
        m = self.gemm_moe_tune
        result = {
            "block_m": 32,
            "ksplit": 0,
            "us1": 1,
            "kernelName1": "k1",
            "err1": 0,
            "us2": 1,
            "kernelName2": "k2",
            "err2": 0,
            "us": 2,
            "run_1stage": 0,
            "xbf16": 0,
            "flat": 0,
            "tflops": 0,
            "bw": 0,
            "nt": 0,
        }
        shape = self.untuned().iloc[0].to_dict()
        on_file = [
            {**shape, **result, "token": token, "_tag": tag}
            for token, tag in ((1, ""), (2, "fused_expert"), (3, "flydsl_fallback"))
        ]
        cases = {
            "fmoe": (self.tuner, {"1", "99"}, None),
            "mxfp4": (
                m.Mxfp4FlydslTuner("mxfp4Fmoe", KEYS, RESULT_LIST, "mxfp4"),
                {"1", "2", "99"},
                "fused_expert",
            ),
        }
        for name, (tuner, tokens, tag_of_2) in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as root:
                path = str(Path(root) / "tuned.csv")
                pd.DataFrame(on_file).to_csv(path, index=False)
                update = pd.DataFrame([{**shape, **result, "token": 99}])
                with contextlib.redirect_stdout(io.StringIO()):
                    tuner.result_to_csv(update[tuner.columns], path)
                saved = pd.read_csv(path, dtype=str, keep_default_na=False)
                self.assertEqual(set(saved["token"]), tokens)
                if tag_of_2 is None:
                    self.assertNotIn("_tag", saved.columns)
                else:
                    tags = dict(zip(saved["token"], saved["_tag"]))
                    self.assertEqual(tags["2"], tag_of_2)

    def test_result_to_csv_migrates_gfx_drops_tags_and_keeps_the_last_row(self):
        columns = self.tuner.columns
        identity = {
            "cu_num": 304,
            "model_dim": 4096,
            "inter_dim": 1024,
            "expert": 8,
            "topk": 2,
            "act_type": "ActivationType.Silu",
            "dtype": "torch.bfloat16",
            "q_dtype_a": "torch.bfloat16",
            "q_dtype_w": "torch.bfloat16",
            "q_type": "QuantType.No",
            "use_g1u1": 1,
            "doweight_stage1": 0,
            "block_m": 32,
            "ksplit": 0,
            "us1": 1,
            "err1": "0.0%",
            "us2": 1,
            "kernelName2": "s2",
            "err2": "0.0%",
            "us": 2,
            "run_1stage": 0,
            "xbf16": 0,
            "tflops": 0,
            "bw": 0,
        }

        def legacy(token, kernel, tag):
            row = dict(identity)
            row.update(token=token, kernelName1=kernel, _tag=tag)
            return row

        updates = pd.DataFrame(
            [
                {
                    **identity,
                    "gfx": "gfx942",
                    "token": 32,
                    "kernelName1": "new_kernel",
                    "flat": 1,
                    "nt": 1,
                },
                {
                    **identity,
                    "gfx": "gfx950",
                    "cu_num": 256,
                    "token": 96,
                    "kernelName1": "added_first",
                    "flat": 0,
                    "nt": 0,
                },
                {
                    **identity,
                    "gfx": "gfx950",
                    "cu_num": 256,
                    "token": 96,
                    "kernelName1": "added_last",
                    "flat": 0,
                    "nt": 0,
                },
            ]
        )
        with tempfile.TemporaryDirectory() as root:
            path = str(Path(root) / "tuned.csv")
            pd.DataFrame(
                [
                    legacy(32, "old_kernel", ""),
                    legacy(64, "tagged_kernel", "stale"),
                    legacy(7, "untouched", ""),
                ]
            ).to_csv(path, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                self.tuner.result_to_csv(updates[columns], path)
            saved = pd.read_csv(path, dtype=str)
        self.assertEqual(next(iter(saved.columns)), "gfx")
        self.assertNotIn("_tag", saved.columns)
        by_token = {row["token"]: row for _, row in saved.iterrows()}
        self.assertEqual(set(by_token), {"32", "7", "96"})
        self.assertEqual(by_token["32"]["gfx"], "gfx942")
        self.assertEqual(by_token["32"]["kernelName1"], "new_kernel")
        self.assertEqual(by_token["32"]["flat"], "1")
        self.assertEqual(by_token["7"]["gfx"], "gfx942")
        self.assertEqual(by_token["7"]["kernelName1"], "untouched")
        self.assertEqual(by_token["7"]["nt"], "0")
        self.assertEqual(by_token["7"]["flat"], "0")
        self.assertEqual(by_token["96"]["kernelName1"], "added_last")
        self.assertEqual(by_token["96"]["gfx"], "gfx950")


if __name__ == "__main__":
    unittest.main(verbosity=2)
