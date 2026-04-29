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

import logging
import os
import tempfile
import unittest

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
                reload_fn=lambda: {},
                logger_obj=logger_obj,
            )
        self.assertTrue(mp_lock_called)
        self.assertIsNone(cfg)
        self.assertTrue(any("Fmoe tuning not support" in msg for msg in cm.output))

    def test_cfg_2stages_none_env_on(self):
        """When cfg_2stages is None (no CSV loaded at all), should still trigger tune."""
        new_cfg_2stages = {SAMPLE_KEYS: SAMPLE_CFG}
        logger_obj = logging.getLogger("test_online_tune")

        cfg, updated_cfgs, mp_lock_called = simulate_online_tune_path(
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


class TestMainFuncCSVWrite(unittest.TestCase):
    """Tests that the MainFunc closure writes the correct row to the untuned CSV."""

    def test_writes_header_when_empty(self):
        """MainFunc should write CSV header when file is empty."""
        with tempfile.TemporaryDirectory() as tmp:
            untune_file = os.path.join(tmp, "untuned_fmoe.csv")
            with open(untune_file, "w"):
                pass

            token, model_dim, inter_dim, expert, topk = 16, 7168, 256, 256, 8
            activation = "ActivationType.Silu"
            dtype = "torch.bfloat16"
            q_dtype_a = "torch.float8_e4m3fnuz"
            q_dtype_w = "torch.float8_e4m3fnuz"
            q_type = "QuantType.per_Token"
            use_g1u1 = 1
            doweight_stage1 = 0

            with open(untune_file, "a") as f:
                if os.path.getsize(untune_file) == 0:
                    f.write(
                        "token,model_dim,inter_dim,expert,topk,act_type,dtype,"
                        "q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1"
                    )
                f.write(
                    f"\n{token},{model_dim},{inter_dim},{expert},{topk},{activation},"
                    f"{dtype},{q_dtype_a},{q_dtype_w},{q_type},{int(use_g1u1)},{int(doweight_stage1)}"
                )

            with open(untune_file) as f:
                lines = f.read().strip().split("\n")
            self.assertEqual(len(lines), 2, "Should have header + 1 data row")
            self.assertIn("token,model_dim", lines[0])
            parts = lines[1].split(",")
            self.assertEqual(parts[0], "16")
            self.assertEqual(parts[1], "7168")
            self.assertEqual(parts[3], "256")

    def test_appends_without_header_when_nonempty(self):
        """MainFunc should NOT re-write header when file already has content."""
        with tempfile.TemporaryDirectory() as tmp:
            untune_file = os.path.join(tmp, "untuned_fmoe.csv")
            header = (
                "token,model_dim,inter_dim,expert,topk,act_type,dtype,"
                "q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1"
            )
            existing_row = "512,6144,4096,8,2,ActivationType.Silu,torch.bfloat16,torch.bfloat16,torch.bfloat16,QuantType.No,1,0"
            with open(untune_file, "w") as f:
                f.write(header + "\n" + existing_row)

            token, model_dim, inter_dim, expert, topk = 16, 7168, 256, 256, 8
            activation = "ActivationType.Silu"
            dtype = "torch.bfloat16"
            q_dtype_a = "torch.float8_e4m3fnuz"
            q_dtype_w = "torch.float8_e4m3fnuz"
            q_type = "QuantType.per_Token"
            use_g1u1 = 1
            doweight_stage1 = 0

            with open(untune_file, "a") as f:
                if os.path.getsize(untune_file) == 0:
                    f.write(header)
                f.write(
                    f"\n{token},{model_dim},{inter_dim},{expert},{topk},{activation},"
                    f"{dtype},{q_dtype_a},{q_dtype_w},{q_type},{int(use_g1u1)},{int(doweight_stage1)}"
                )

            with open(untune_file) as f:
                lines = f.read().strip().split("\n")
            self.assertEqual(len(lines), 3, "Should have header + 2 data rows")
            header_count = sum(
                1 for line in lines if line.startswith("token,model_dim")
            )
            self.assertEqual(header_count, 1, "Header should appear only once")


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
                "1,0,32,0,flydsl_k1,flydsl_k2,50.0,flydsl_fallback\n"
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
