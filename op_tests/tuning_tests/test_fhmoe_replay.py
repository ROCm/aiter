# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Replay FHMoE's public operator against a generated config in a fresh process.

The tuner worker times `_fused_moe_impl`. This checks that `fused_moe(...)`
with shared-expert arguments, in a new interpreter, reads only
`AITER_CONFIG_FHMOE` and either raises on an uncovered DSV4 I384 token or
launches the kernel names from that file.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

AITER_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# Distinct from shipped tuned_fhmoe.csv token=1 (…_atomic without _persist).
_REPLAY_KN1 = "flydsl_moe1_afp8_wfp4_bf16_t32x64x256_w4_gui_kw4_fp8"
_REPLAY_KN2 = "flydsl_moe2_afp8_wfp4_bf16_t32x256x128_atomic_persist"

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


def _gpu_available():
    try:
        import torch

        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except ImportError:
        return False


def _write_header_only(path):
    with open(path, "w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=_NATIVE_FIELDS).writeheader()


def _write_generated_row(path, gfx, cu_num):
    row = {
        "gfx": gfx,
        "cu_num": cu_num,
        "token": 1,
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
        "block_m": 32,
        "ksplit": 0,
        "kernelName1": _REPLAY_KN1,
        "kernelName2": _REPLAY_KN2,
    }
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_NATIVE_FIELDS)
        writer.writeheader()
        writer.writerow(row)


def _child_script():
    return textwrap.dedent(
        f"""\
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
            "token": 1,
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
        """
    )


@unittest.skipUnless(_gpu_available(), "No GPU available")
class TestFhmoeReplay(unittest.TestCase):
    def setUp(self):
        from aiter.jit.utils.chip_info import get_gfx

        if get_gfx() != "gfx950":
            self.skipTest("FHMoE replay requires gfx950")

    def _run_public_op(self, csv_path, timeout=600):
        script_path = os.path.join(os.path.dirname(csv_path), "fhmoe_replay_child.py")
        with open(script_path, "w") as handle:
            handle.write(_child_script())
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

    def test_generated_csv_selected_by_public_fused_moe(self):
        from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "generated_fhmoe.csv")
            _write_generated_row(csv_path, get_gfx_runtime(), int(get_cu_num()))
            result = self._run_public_op(csv_path)
            output = result.stdout + result.stderr
            self.assertEqual(
                result.returncode,
                0,
                f"generated replay child failed\n{output[-4000:]}",
            )
            self.assertIn("REPLAY_OK", result.stdout)
            self.assertIn(_REPLAY_KN1, output)
            self.assertIn(_REPLAY_KN2, output)
            self.assertNotIn("REPLAY_RAISE", result.stdout)


if __name__ == "__main__":
    unittest.main()
