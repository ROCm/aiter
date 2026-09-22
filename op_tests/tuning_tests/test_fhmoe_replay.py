# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Replay FHMoE's public operator against tuner-written config in a fresh process.

The tuner worker times `_fused_moe_impl`. This checks that `fused_moe(...)`
with shared-expert arguments, in a new interpreter, reads only
`AITER_CONFIG_FHMOE` and either raises on an uncovered DSV4 I384 token or
launches the kernel names from a CSV that `FhmoeTuner` actually wrote.
A planted row is not the producer contract.
"""

from __future__ import annotations

import csv
import glob
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

AITER_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_TUNE_SCRIPT = os.path.join(
    AITER_ROOT, "csrc", "ck_gemm_moe_2stages_codegen", "gemm_moe_tune.py"
)

# Two kn2 suffixes, both distinct from shipped token=1 (`…_atomic` with no suffix).
# TUNE_MOE_KERNEL_REGEX matches f"{kn1} {kn2}".
_TUNE_KN1 = "flydsl_moe1_afp8_wfp4_bf16_t32x64x256_w4_gui_kw4_fp8"
_TUNE_KN2_A = "flydsl_moe2_afp8_wfp4_bf16_t32x256x128_atomic_bnt2"
_TUNE_KN2_B = "flydsl_moe2_afp8_wfp4_bf16_t32x256x128_atomic_persist"
_TUNE_KN2S = frozenset((_TUNE_KN2_A, _TUNE_KN2_B))
_SHIPPED_TOKEN1_KN2 = "flydsl_moe2_afp8_wfp4_bf16_t32x256x128_atomic"
_KERNEL_REGEX = rf"{_TUNE_KN1} {_TUNE_KN2_A}$|{_TUNE_KN1} {_TUNE_KN2_B}$"
# fused_moe refuses M=16 unless the CSV has every power-of-two through 16.
_TUNE_TOKENS = (1, 2, 4, 8, 16)
_REPLAY_TOKENS = (1, 16)

_NATIVE_FIELDS = [
    "gfx",
    "cu_num",
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "shared_expert_id",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "use_g1u1",
    "doweight_stage1",
    "hidden_pad",
    "intermediate_pad",
    "gate_mode",
    "block_m",
    "ksplit",
    "kernelName1",
    "kernelName2",
]

_UNTUNED_FIELDS = [
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "shared_expert_id",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "use_g1u1",
    "doweight_stage1",
    "hidden_pad",
    "intermediate_pad",
    "gate_mode",
]


def _gpu_available():
    try:
        import torch

        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except ImportError:
        return False


def _write_header_only(path):
    with open(path, "w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=_NATIVE_FIELDS).writeheader()


def _write_untuned_rows(path, tokens):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_UNTUNED_FIELDS)
        writer.writeheader()
        for token in tokens:
            writer.writerow(
                {
                    "token": token,
                    "model_dim": 7168,
                    "inter_dim": 384,
                    "expert": 385,
                    "topk": 7,
                    "shared_expert_id": 384,
                    "act_type": "ActivationType.Silu",
                    "dtype": "torch.bfloat16",
                    "q_dtype_a": "torch.float8_e4m3fn",
                    "q_dtype_w": "torch.float4_e2m1fn_x2",
                    "q_type": "QuantType.per_1x32",
                    "use_g1u1": 1,
                    "doweight_stage1": 0,
                    "hidden_pad": 0,
                    "intermediate_pad": 0,
                    "gate_mode": "GateMode.INTERLEAVE",
                }
            )


def _cleanup_stale_lock_files():
    build_dir = os.path.join(AITER_ROOT, "aiter", "jit", "build")
    if not os.path.isdir(build_dir):
        return
    for pattern in (
        os.path.join(build_dir, "lock_*"),
        os.path.join(build_dir, "*", "build", "lock"),
        os.path.join(build_dir, "lock_3rdparty_*"),
    ):
        for lock_file in glob.glob(pattern):
            try:
                os.remove(lock_file)
            except OSError:
                pass


def _child_script(token):
    return textwrap.dedent(f"""\
        import sys
        sys.path.insert(
            0,
            {os.path.join(AITER_ROOT, "csrc", "ck_gemm_moe_2stages_codegen")!r},
        )
        import torch
        from aiter import ActivationType, QuantType, dtypes
        from aiter.fused_moe import fused_moe
        from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
        from aiter.ops.flydsl.moe_common import GateMode
        from gemm_moe_tune import FhmoeTuner

        shape = {{
            "gfx": get_gfx_runtime(),
            "cu_num": int(get_cu_num()),
            "token": {int(token)},
            "model_dim": 7168,
            "inter_dim": 384,
            "expert": 385,
            "topk": 7,
            "shared_expert_id": 384,
            "act_type": ActivationType.Silu,
            "dtype": torch.bfloat16,
            "q_dtype_a": dtypes.fp8,
            "q_dtype_w": dtypes.fp4x2,
            "q_type": QuantType.per_1x32,
            "use_g1u1": 1,
            "doweight_stage1": 0,
            "hidden_pad": 0,
            "intermediate_pad": 0,
            "gate_mode": GateMode.INTERLEAVE,
        }}
        data = FhmoeTuner.generate_fhmoe_data(shape)

        def _cuda(tensor):
            if tensor is None or not torch.is_tensor(tensor):
                return tensor
            return tensor.cuda()

        w1 = _cuda(data["w1"])
        w2 = _cuda(data["w2"])
        w1.is_shuffled = True
        w2.is_shuffled = True
        try:
            out = fused_moe(
                _cuda(data["hidden"]),
                w1,
                w2,
                _cuda(data["topk_weights"]),
                _cuda(data["topk_ids"]),
                activation=ActivationType.Silu,
                quant_type=QuantType.per_1x32,
                w1_scale=_cuda(data["w1_scale"]),
                w2_scale=_cuda(data["w2_scale"]),
                dtype=torch.bfloat16,
                hidden_pad=0,
                intermediate_pad=0,
                gate_mode=GateMode.INTERLEAVE.value,
                shared_w1=_cuda(data["shared_w1"]),
                shared_w2=_cuda(data["shared_w2"]),
                shared_w1_scale=_cuda(data["shared_w1_scale"]),
                shared_w2_scale=_cuda(data["shared_w2_scale"]),
                shared_expert_id=384,
            )
        except NotImplementedError as exc:
            print("REPLAY_RAISE", str(exc))
            raise SystemExit(0)
        print("REPLAY_OK", tuple(out.shape))
        """)


@unittest.skipUnless(_gpu_available(), "No GPU available")
class TestFhmoeReplay(unittest.TestCase):
    def setUp(self):
        from aiter.jit.utils.chip_info import get_gfx

        if get_gfx() != "gfx950":
            self.skipTest("FHMoE replay requires gfx950")

    def _run_public_op(self, csv_path, token=1, timeout=600):
        script_path = os.path.join(
            os.path.dirname(csv_path), f"fhmoe_replay_child_t{token}.py"
        )
        with open(script_path, "w") as handle:
            handle.write(_child_script(token))
        env = os.environ.copy()
        env["AITER_CONFIG_FHMOE"] = csv_path
        env["PYTHONPATH"] = AITER_ROOT + os.pathsep + env.get("PYTHONPATH", "")
        env.pop("AITER_BYPASS_TUNE_CONFIG", None)
        return subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=AITER_ROOT,
            env=env,
            check=False,
        )

    def _run_fhmoe_tuner(self, untuned_path, tuned_path, timeout=1800):
        _cleanup_stale_lock_files()
        env = os.environ.copy()
        script_dir = os.path.dirname(_TUNE_SCRIPT)
        env["PYTHONPATH"] = script_dir + os.pathsep + env.get("PYTHONPATH", "")
        env["TUNE_MOE_KERNEL_REGEX"] = _KERNEL_REGEX
        env.pop("AITER_BYPASS_TUNE_CONFIG", None)
        cmd = [
            sys.executable,
            _TUNE_SCRIPT,
            "--fhmoe",
            "-i",
            untuned_path,
            "-o",
            tuned_path,
            "--mp",
            "1",
        ]
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=AITER_ROOT,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired:
            _cleanup_stale_lock_files()
            raise

    def test_uncovered_token_raises_in_fresh_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "untuned_empty.csv")
            _write_header_only(csv_path)
            result = self._run_public_op(csv_path)
            output = result.stdout + result.stderr
            self.assertEqual(
                result.returncode,
                0,
                f"uncovered replay child failed\n{output[-4000:]}",
            )
            self.assertIn("REPLAY_RAISE", result.stdout)
            self.assertIn("does not cover this DSV4 I384", result.stdout)

    def test_tuner_written_csv_selected_by_public_fused_moe(self):
        with tempfile.TemporaryDirectory() as tmp:
            untuned_path = os.path.join(tmp, "untuned_fhmoe.csv")
            tuned_path = os.path.join(tmp, "tuned_fhmoe.csv")
            _write_untuned_rows(untuned_path, _TUNE_TOKENS)
            tune = self._run_fhmoe_tuner(untuned_path, tuned_path)
            tune_out = tune.stdout + tune.stderr
            self.assertEqual(
                tune.returncode,
                0,
                f"FhmoeTuner failed\n{tune_out[-4000:]}",
            )
            self.assertTrue(os.path.isfile(tuned_path), "tuner did not write -o")
            with open(tuned_path, newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows, f"tuner wrote no rows\n{tune_out[-4000:]}")
            self.assertIn("us", rows[0], "producer CSV must include timed us")
            by_token = {int(row["token"]): row for row in rows}
            self.assertEqual(
                set(by_token),
                set(_TUNE_TOKENS),
                f"tuner tokens {sorted(by_token)} != {_TUNE_TOKENS}",
            )
            for token, row in by_token.items():
                kn1, kn2 = row["kernelName1"], row["kernelName2"]
                self.assertEqual(kn1, _TUNE_KN1, f"token={token} kn1={kn1}")
                self.assertIn(kn2, _TUNE_KN2S, f"token={token} kn2={kn2}")
                self.assertGreater(float(row["us"]), 0.0)
                if token == 1:
                    self.assertNotEqual(
                        kn2,
                        _SHIPPED_TOKEN1_KN2,
                        "winner matches shipped token=1; cannot prove AITER_CONFIG_FHMOE",
                    )
            for token in _REPLAY_TOKENS:
                row = by_token[token]
                result = self._run_public_op(tuned_path, token=token)
                output = result.stdout + result.stderr
                self.assertEqual(
                    result.returncode,
                    0,
                    f"token={token} replay child failed\n{output[-4000:]}",
                )
                self.assertIn("REPLAY_OK", result.stdout)
                self.assertIn(row["kernelName1"], output)
                self.assertIn(row["kernelName2"], output)
                self.assertNotIn("REPLAY_RAISE", result.stdout)

    def test_run_config_times_public_fused_moe(self):
        from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

        shipped = os.path.join(AITER_ROOT, "aiter", "configs", "tuned_fhmoe.csv")
        if not os.path.isfile(shipped):
            self.skipTest("aiter/configs/tuned_fhmoe.csv is missing")
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "run_config_fhmoe.csv")
            with open(shipped, newline="") as handle:
                rows = list(csv.DictReader(handle))
            token1 = next((row for row in rows if int(row["token"]) == 1), None)
            if token1 is None:
                self.skipTest("shipped tuned_fhmoe.csv has no token=1 row")
            token1 = dict(token1)
            token1["gfx"] = get_gfx_runtime()
            token1["cu_num"] = str(int(get_cu_num()))
            with open(csv_path, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=token1.keys())
                writer.writeheader()
                writer.writerow(token1)
            _cleanup_stale_lock_files()
            env = os.environ.copy()
            script_dir = os.path.dirname(_TUNE_SCRIPT)
            env["PYTHONPATH"] = script_dir + os.pathsep + env.get("PYTHONPATH", "")
            env.pop("AITER_BYPASS_TUNE_CONFIG", None)
            result = subprocess.run(
                [
                    sys.executable,
                    _TUNE_SCRIPT,
                    "--fhmoe",
                    "--run_config",
                    csv_path,
                    "--warmup",
                    "1",
                    "--iters",
                    "2",
                    "--mp",
                    "1",
                ],
                capture_output=True,
                text=True,
                timeout=600,
                cwd=AITER_ROOT,
                env=env,
                check=False,
            )
            output = result.stdout + result.stderr
            self.assertEqual(
                result.returncode,
                0,
                f"--fhmoe --run_config failed\n{output[-4000:]}",
            )
            self.assertIn("shared=", output)
            lines = [line.strip() for line in output.splitlines()]
            self.assertTrue(
                any(line.endswith("OK") for line in lines),
                f"no OK status line\n{output[-4000:]}",
            )
            self.assertFalse(
                any(line.endswith("ERROR") for line in lines),
                f"run_config ERROR\n{output[-4000:]}",
            )
            self.assertFalse(
                any(line.endswith("MISMATCH") for line in lines),
                f"run_config MISMATCH\n{output[-4000:]}",
            )


if __name__ == "__main__":
    unittest.main()
