# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Level 1: Unit tests for AITER_ONLINE_TUNE logic (no GPU required).

Tests the online tuning decision path in fused_moe.py / fused_moe_dp_shared_expert.py:
  - Config found -> skip online tune
  - Config missing + AITER_ONLINE_TUNE=0 -> skip online tune
  - Config missing + AITER_ONLINE_TUNE=1 -> trigger mp_lock, reload cfg
  - Tune succeeds -> new config used
  - Tune fails -> cfg stays None, warning logged
  - mp_lock synchronization: first acquirer runs MainFunc, waiters get None
  - MainFunc writes correct shape row to untuned CSV
"""

import importlib
import logging
import os
import tempfile
import unittest
from unittest.mock import patch

SAMPLE_KEYS = (
    256,
    16,
    7168,
    256,
    256,
    8,
    "ActivationType.Silu",
    "torch.bfloat16",
    "torch.float8_e4m3fnuz",
    "torch.float8_e4m3fnuz",
    "QuantType.per_Token",
    1,
    0,
)

SAMPLE_CFG = {
    "block_m": 64,
    "ksplit": 0,
    "kernelName1": "ck2stages_kernel_1",
    "kernelName2": "ck2stages_kernel_2",
    "run_1stage": False,
    "us": 42.0,
}


def _whole_graph_test_impl(request, config):
    return request.hidden_states


def simulate_online_tune_path(
    cfg_2stages,
    keys,
    env_online_tune,
    mp_lock_fn,
    reload_fn,
    logger_obj,
):
    """Simulate the AITER_ONLINE_TUNE decision logic from fused_moe.py:837-844.

    This mirrors the production code pattern:
        cfg = cfg_2stages.get(keys, None) if cfg_2stages else None
        if cfg is None and os.environ.get("AITER_ONLINE_TUNE", "0") == "1":
            lock_path = ...
            mp_lock(lock_path, MainFunc=MainFunc, FinalFunc=FinalFunc)
            cfg_2stages = get_cfg_2stages(tune_file)
            cfg = cfg_2stages.get(keys, None) if cfg_2stages else None
            if cfg is None:
                logger.warning(f"Fmoe tuning not support for {keys}")

    Returns (cfg, cfg_2stages, mp_lock_called).
    """
    mp_lock_called = False
    cfg = cfg_2stages.get(keys, None) if cfg_2stages else None
    if cfg is None and env_online_tune == "1":
        mp_lock_called = True
        mp_lock_fn()
        cfg_2stages = reload_fn()
        cfg = cfg_2stages.get(keys, None) if cfg_2stages else None
        if cfg is None:
            logger_obj.warning(f"Fmoe tuning not support for {keys}")
    return cfg, cfg_2stages, mp_lock_called


class TestOnlineTuneDecision(unittest.TestCase):
    """Tests for the AITER_ONLINE_TUNE decision logic."""

    def test_config_found_skips_online_tune(self):
        """When cfg is already in cfg_2stages, online tune should not trigger."""
        cfg_2stages = {SAMPLE_KEYS: SAMPLE_CFG}
        logger_obj = logging.getLogger("test_online_tune")

        cfg, _, mp_lock_called = simulate_online_tune_path(
            cfg_2stages,
            SAMPLE_KEYS,
            "1",
            mp_lock_fn=lambda: self.fail("mp_lock should not be called"),
            reload_fn=lambda: self.fail("reload should not be called"),
            logger_obj=logger_obj,
        )
        self.assertFalse(mp_lock_called)
        self.assertEqual(cfg, SAMPLE_CFG)

    def test_config_missing_env_off_skips_online_tune(self):
        """When AITER_ONLINE_TUNE=0, missing config should NOT trigger tuning."""
        cfg_2stages = {}
        logger_obj = logging.getLogger("test_online_tune")

        cfg, _, mp_lock_called = simulate_online_tune_path(
            cfg_2stages,
            SAMPLE_KEYS,
            "0",
            mp_lock_fn=lambda: self.fail("mp_lock should not be called"),
            reload_fn=lambda: self.fail("reload should not be called"),
            logger_obj=logger_obj,
        )
        self.assertFalse(mp_lock_called)
        self.assertIsNone(cfg)

    def test_config_missing_env_default_skips_online_tune(self):
        """Default env value (not set) should NOT trigger tuning."""
        cfg_2stages = {}
        logger_obj = logging.getLogger("test_online_tune")

        saved = os.environ.pop("AITER_ONLINE_TUNE", None)
        try:
            env_val = os.environ.get("AITER_ONLINE_TUNE", "0")
            cfg, _, mp_lock_called = simulate_online_tune_path(
                cfg_2stages,
                SAMPLE_KEYS,
                env_val,
                mp_lock_fn=lambda: self.fail("mp_lock should not be called"),
                reload_fn=lambda: self.fail("reload should not be called"),
                logger_obj=logger_obj,
            )
        finally:
            if saved is not None:
                os.environ["AITER_ONLINE_TUNE"] = saved

        self.assertFalse(mp_lock_called)
        self.assertIsNone(cfg)

    def test_config_missing_env_on_triggers_tune_success(self):
        """AITER_ONLINE_TUNE=1 + missing config -> tune, reload succeeds."""
        cfg_2stages = {}
        new_cfg_2stages = {SAMPLE_KEYS: SAMPLE_CFG}
        logger_obj = logging.getLogger("test_online_tune")

        cfg, updated_cfgs, mp_lock_called = simulate_online_tune_path(
            cfg_2stages,
            SAMPLE_KEYS,
            "1",
            mp_lock_fn=lambda: None,
            reload_fn=lambda: new_cfg_2stages,
            logger_obj=logger_obj,
        )
        self.assertTrue(mp_lock_called)
        self.assertEqual(cfg, SAMPLE_CFG)
        self.assertIs(updated_cfgs, new_cfg_2stages)

    def test_config_missing_env_on_tune_fails(self):
        """AITER_ONLINE_TUNE=1 + tune fails -> cfg stays None, warning logged."""
        cfg_2stages = {}
        logger_obj = logging.getLogger("test_online_tune")

        with self.assertLogs(logger_obj, level="WARNING") as cm:
            cfg, _, mp_lock_called = simulate_online_tune_path(
                cfg_2stages,
                SAMPLE_KEYS,
                "1",
                mp_lock_fn=lambda: None,
                reload_fn=dict,
                logger_obj=logger_obj,
            )
        self.assertTrue(mp_lock_called)
        self.assertIsNone(cfg)
        self.assertTrue(any("Fmoe tuning not support" in msg for msg in cm.output))

    def test_cfg_2stages_none_env_on(self):
        """When cfg_2stages is None (no CSV loaded at all), should still trigger tune."""
        new_cfg_2stages = {SAMPLE_KEYS: SAMPLE_CFG}
        logger_obj = logging.getLogger("test_online_tune")

        cfg, _updated_cfgs, mp_lock_called = simulate_online_tune_path(
            None,
            SAMPLE_KEYS,
            "1",
            mp_lock_fn=lambda: None,
            reload_fn=lambda: new_cfg_2stages,
            logger_obj=logger_obj,
        )
        self.assertTrue(mp_lock_called)
        self.assertEqual(cfg, SAMPLE_CFG)

    def test_cfg_2stages_none_env_off(self):
        """When cfg_2stages is None and env is off, no tuning."""
        logger_obj = logging.getLogger("test_online_tune")

        cfg, _, mp_lock_called = simulate_online_tune_path(
            None,
            SAMPLE_KEYS,
            "0",
            mp_lock_fn=lambda: self.fail("mp_lock should not be called"),
            reload_fn=lambda: self.fail("reload should not be called"),
            logger_obj=logger_obj,
        )
        self.assertFalse(mp_lock_called)
        self.assertIsNone(cfg)


class TestMpLock(unittest.TestCase):
    """Tests for the real mp_lock function from aiter.jit.core."""

    def test_first_acquirer_runs_main_func(self):
        from aiter.jit.core import mp_lock

        with tempfile.TemporaryDirectory() as tmp:
            lock_path = os.path.join(tmp, "test_lock")
            called = []
            mp_lock(
                lock_path,
                MainFunc=lambda: called.append("main"),
                FinalFunc=lambda: called.append("final"),
            )
            self.assertEqual(called, ["main", "final"])

    def test_main_func_return_value(self):
        from aiter.jit.core import mp_lock

        with tempfile.TemporaryDirectory() as tmp:
            lock_path = os.path.join(tmp, "test_lock_ret")
            result = mp_lock(lock_path, MainFunc=lambda: 42)
            self.assertEqual(result, 42)

    def test_final_func_called_on_exception(self):
        from aiter.jit.core import mp_lock

        with tempfile.TemporaryDirectory() as tmp:
            lock_path = os.path.join(tmp, "test_lock_exc")
            final_called = []

            with self.assertRaises(ValueError):

                def raise_err():
                    raise ValueError("boom")

                mp_lock(
                    lock_path,
                    MainFunc=raise_err,
                    FinalFunc=lambda: final_called.append(True),
                )
            self.assertEqual(
                len(final_called), 1, "FinalFunc should be called even on exception"
            )

    def test_lock_file_cleaned_up(self):
        from aiter.jit.core import mp_lock

        with tempfile.TemporaryDirectory() as tmp:
            lock_path = os.path.join(tmp, "test_lock_cleanup")
            mp_lock(lock_path, MainFunc=lambda: None)
            self.assertFalse(
                os.path.exists(lock_path),
                "Lock file should be removed after release",
            )


UNTUNED_HEADER = (
    "token,model_dim,inter_dim,expert,topk,act_type,dtype,"
    "q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1"
)


class TestOnlineTuneFmoe(unittest.TestCase):
    """Tests for the real _online_tune_fmoe shared by both online-tune paths."""

    def _run(self, tmp, returncode, untuned_content=""):
        """Call _online_tune_fmoe with the tuner subprocess faked to exit returncode.

        Returns (result, untuned lines, tuner command, aiter log lines).
        """
        import subprocess

        import torch

        fused_moe = importlib.import_module("aiter.fused_moe")
        untune_file = os.path.join(tmp, "untuned_fmoe.csv")
        tune_file = os.path.join(tmp, "tuned_fmoe.csv")
        profile_file = os.path.join(tmp, "profile_fmoe.csv")
        with open(untune_file, "w") as f:
            f.write(untuned_content)
        commands = []

        def fake_run(cmd, **kwargs):
            commands.append(cmd)
            return subprocess.CompletedProcess(cmd, returncode)

        with (
            patch.object(fused_moe.subprocess, "run", fake_run),
            self.assertLogs("aiter", level="INFO") as logs,
        ):
            result = fused_moe._online_tune_fmoe(
                untune_file,
                tune_file,
                profile_file,
                16,
                7168,
                256,
                256,
                8,
                "ActivationType.Silu",
                torch.bfloat16,
                torch.float8_e4m3fnuz,
                torch.uint32,
                "QuantType.per_Token",
                True,
                False,
            )
        with open(untune_file) as f:
            lines = f.read().split("\n")
        return result, lines, commands, logs.output

    def test_writes_header_when_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, lines, _, _ = self._run(tmp, returncode=0)
        self.assertEqual(lines[0], UNTUNED_HEADER)
        self.assertEqual(
            lines[1],
            "16,7168,256,256,8,ActivationType.Silu,torch.bfloat16,"
            "torch.float8_e4m3fnuz,torch.int4,QuantType.per_Token,1,0",
        )
        self.assertEqual(len(lines), 2)

    def test_appends_without_header_when_nonempty(self):
        existing = UNTUNED_HEADER + "\n512,6144,4096,8,2,ActivationType.Silu,..."
        with tempfile.TemporaryDirectory() as tmp:
            _, lines, _, _ = self._run(tmp, returncode=0, untuned_content=existing)
        self.assertEqual(sum(line == UNTUNED_HEADER for line in lines), 1)
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[2].startswith("16,7168,256,"))

    def test_runs_tuner_on_last_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, commands, _ = self._run(tmp, returncode=0)
        self.assertEqual(len(commands), 1)
        cmd = commands[0]
        self.assertTrue(cmd[1].endswith("gemm_moe_tune.py"))
        self.assertEqual(
            cmd[cmd.index("-i") + 1], os.path.join(tmp, "untuned_fmoe.csv")
        )
        self.assertEqual(cmd[cmd.index("-o") + 1], os.path.join(tmp, "tuned_fmoe.csv"))
        self.assertIn("--last", cmd)

    def test_success_returns_true_and_prints_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _, _, logs = self._run(tmp, returncode=0)
        self.assertTrue(result)
        self.assertTrue(any("[Hint] tuned configs are saved" in m for m in logs))

    def test_failure_returns_false_without_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _, _, logs = self._run(tmp, returncode=1)
        self.assertFalse(result)
        self.assertFalse(any("[Hint]" in m for m in logs))
        self.assertTrue(any("online tune exited 1" in m for m in logs))

    def test_shared_expert_path_uses_same_helper(self):
        fused_moe = importlib.import_module("aiter.fused_moe")
        shared = importlib.import_module("aiter.fused_moe_dp_shared_expert")
        self.assertIs(shared._online_tune_fmoe, fused_moe._online_tune_fmoe)


class TestGetCfg2stages(unittest.TestCase):
    """Test the CSV->dict loading logic used by get_cfg_2stages."""

    def test_loads_and_indexes_correctly(self):
        import pandas as pd

        _INDEX_COLS = [
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

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(
                "cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,"
                "q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,"
                "block_m,ksplit,kernelName1,kernelName2,us\n"
            )
            f.write(
                "256,16,7168,256,256,8,ActivationType.Silu,torch.bfloat16,"
                "torch.float8_e4m3fnuz,torch.float8_e4m3fnuz,QuantType.per_Token,"
                "1,0,64,0,ck2stages_k1,ck2stages_k2,42.0\n"
            )
            path = f.name

        try:
            df = pd.read_csv(path)
            cfg_2stages = df.set_index(_INDEX_COLS).to_dict("index")
            cfg = cfg_2stages.get(SAMPLE_KEYS)
            self.assertIsNotNone(cfg)
            self.assertEqual(cfg["block_m"], 64)
            self.assertEqual(cfg["kernelName1"], "ck2stages_k1")
        finally:
            os.unlink(path)

    def test_tag_filter(self):
        """Rows with non-empty _tag should be filtered out (fused_moe.py behavior)."""
        import pandas as pd

        _INDEX_COLS = [
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

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(
                "cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,"
                "q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,"
                "block_m,ksplit,kernelName1,kernelName2,us,_tag\n"
            )
            f.write(
                "256,16,7168,256,256,8,ActivationType.Silu,torch.bfloat16,"
                "torch.float8_e4m3fnuz,torch.float8_e4m3fnuz,QuantType.per_Token,"
                "1,0,64,0,ck_k1,ck_k2,42.0,\n"
            )
            f.write(
                "256,16,7168,256,256,8,ActivationType.Silu,torch.bfloat16,"
                "torch.float8_e4m3fnuz,torch.float8_e4m3fnuz,QuantType.per_Token,"
                "1,0,32,0,legacy_k1,legacy_k2,50.0,legacy_tag\n"
            )
            path = f.name

        try:
            df = pd.read_csv(path)
            if "_tag" in df.columns:
                df = df[df["_tag"].fillna("") == ""]
            cfg_2stages = df.set_index(_INDEX_COLS).to_dict("index")
            self.assertEqual(len(cfg_2stages), 1)
            cfg = cfg_2stages.get(SAMPLE_KEYS)
            self.assertIsNotNone(cfg)
            self.assertEqual(cfg["kernelName1"], "ck_k1")
        finally:
            os.unlink(path)

    def test_whole_graph_kernel_name_is_stripped(self):
        from aiter import ActivationType, QuantType, dtypes
        from aiter.fused_moe_registry import (
            make_fused_moe_impl_kernel_name,
            register_fused_moe_impl,
        )

        fused_moe = importlib.import_module("aiter.fused_moe")
        implementation_name = "test_whitespace"
        register_fused_moe_impl(
            implementation_name, f"{__name__}:_whole_graph_test_impl"
        )
        key = (
            "gfx942",
            80,
            1,
            4,
            8,
            2,
            1,
            str(ActivationType.Silu),
            str(dtypes.bf16),
            str(dtypes.fp8),
            str(dtypes.fp8),
            str(QuantType.per_Token),
            True,
            False,
        )
        kernel_name = make_fused_moe_impl_kernel_name(implementation_name, "16")
        previous_configs = fused_moe.cfg_2stages
        fused_moe.cfg_2stages = (
            {key: {"kernelName1": f"  {kernel_name}  ", "block_m": 64}},
            {},
        )
        fused_moe.get_2stage_cfgs.cache_clear()
        try:
            with (
                patch.object(fused_moe, "get_cu_num", return_value=80),
                patch.object(fused_moe, "get_gfx_runtime", return_value="gfx942"),
            ):
                metadata = fused_moe.get_2stage_cfgs(
                    1,
                    4,
                    8,
                    2,
                    1,
                    dtypes.bf16,
                    dtypes.fp8,
                    dtypes.fp8,
                    QuantType.per_Token,
                    True,
                    ActivationType.Silu,
                    False,
                    0,
                    0,
                )
        finally:
            fused_moe.cfg_2stages = previous_configs
            fused_moe.get_2stage_cfgs.cache_clear()

        self.assertIsNotNone(metadata.full_impl)
        self.assertEqual(metadata.block_m, 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
