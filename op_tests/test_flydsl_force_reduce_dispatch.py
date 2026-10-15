# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression tests for AITER_FLYDSL_FORCE_REDUCE (Issue #5299).

Verifies:
1. Sibling kernel rewriting logic (_rewrite_flydsl_force_reduce) across modern V2,
   legacy V1, already-reduce, and native MXMOE kernel families.
2. Complete manifest audit verifying all unique FlyDSL stage2 kernels in
   aiter/configs/model_configs/*.csv rewrite cleanly.
3. Control Plane synchronization in get_2stage_cfgs():
   AITER_FLYDSL_FORCE_REDUCE=1 rewrites kernelName2 and synchronizes
   stage2_uses_route_reduce() and the upstream moe_sorting(accumulate=...) contract.
4. Correct behavior when AITER_FLYDSL_FORCE_REDUCE is unset or "0" (unchanged).
5. Native MXMOE stage2 wrapper support in stage2_uses_route_reduce().
6. Non-FlyDSL kernels (CK-Tile, Opus) are untouched.
7. Moe_kernels.py comment states disabled by default, matching code.
"""

from __future__ import annotations

import csv
import enum
import functools
import glob
import importlib.util
import os
import re
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import pytest

# Ensure mocked environment for CPU-only / zero-GPU execution if running without ROCm
for mod in [
    "aiter.fused_moe_registry",
    "aiter.jit.core",
    "aiter.jit.utils.chip_info",
    "aiter.jit.utils.torch_guard",
    "aiter.ops.flydsl.kernels.mega_moe_gfx1250.types",
    "aiter.ops.flydsl.moe_common",
    "aiter.ops.moe_mxfp4_aux",
    "aiter.ops.opus",
    "aiter.ops.opus.moe_stage1_a8w4",
    "aiter.ops.opus.moe_stage2_a8w4",
    "flydsl",
    "flydsl.runtime",
    "flydsl.compiler",
]:
    sys.modules.setdefault(mod, MagicMock())

mock_flydsl = MagicMock()
mock_flydsl.__version__ = "0.2.4"
sys.modules["flydsl"] = mock_flydsl

sys.modules["aiter.fused_moe_registry"].resolve_fused_moe_impl.return_value = None
sys.modules["aiter.jit.utils.chip_info"].get_cu_num = lambda: 256
sys.modules["aiter.jit.utils.chip_info"].get_gfx_runtime = lambda: "gfx942"
sys.modules["aiter.jit.utils.chip_info"].gfx_from_cu_num = lambda cu: "gfx942"


class ActivationType(enum.Enum):
    No = -1
    Silu = 0
    Gelu = 1
    Swiglu = 2
    Situv2 = 3
    GeluTanh = 4
    Relu2 = 5


class QuantType(enum.Enum):
    No = 0
    per_Tensor = 1
    per_Token = 2
    per_1x32 = 3
    per_1x128 = 4
    per_128x128 = 5
    per_256x128 = 6
    per_1024x128 = 7


class AutoMockModule(types.ModuleType):
    def __getattr__(self, name):
        return MagicMock()


if "aiter" not in sys.modules or not hasattr(sys.modules["aiter"], "fused_moe"):
    mock_aiter = AutoMockModule("aiter")
    mock_aiter.__path__ = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "aiter"))
    ]
    mock_aiter.ActivationType = ActivationType
    mock_aiter.QuantType = QuantType
    mock_aiter.dtypes = AutoMockModule("dtypes")
    mock_aiter.dtypes.bf16 = "torch.bfloat16"
    mock_aiter.dtypes.fp8 = "torch.float8"
    mock_aiter.dtypes.fp4x2 = "torch.float4_e2m1fn_x2"
    sys.modules["aiter"] = mock_aiter

class GateMode(str, enum.Enum):
    SEPARATED = "separated"
    INTERLEAVE = "interleave"


sys.modules["aiter.ops.flydsl.moe_common"].GateMode = GateMode

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FUSED_MOE_PATH = os.path.join(REPO_ROOT, "aiter", "fused_moe.py")

spec = importlib.util.spec_from_file_location("aiter.fused_moe", FUSED_MOE_PATH)
fused_moe = importlib.util.module_from_spec(spec)
sys.modules["aiter.fused_moe"] = fused_moe
spec.loader.exec_module(fused_moe)

from aiter.ops.flydsl.mxfp4_kname import parse_flydsl_v2_gemm2_kernel, parse_g2_kname_any


class TestFlydslForceReduceRewriting(unittest.TestCase):
    """Test sibling kernel name rewriting under Candidate A."""

    def test_modern_v2_rewriting(self):
        kname = "flydsl_moe2_layout_afp8_wfp4_bf16_t32x128x256_atomic_persist_nt_sbm32"
        rewritten = fused_moe._rewrite_flydsl_force_reduce(kname)
        expected = "flydsl_moe2_layout_afp8_wfp4_bf16_t32x128x256_reduce_persist_nt_sbm32"
        self.assertEqual(rewritten, expected)
        cfg = parse_flydsl_v2_gemm2_kernel(rewritten)
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg["epilog"], "reduce")

    def test_modern_v2_already_reduce_unchanged(self):
        kname = "flydsl_moe2_layout_afp8_wfp4_bf16_t32x128x256_reduce_persist_nt_sbm32"
        rewritten = fused_moe._rewrite_flydsl_force_reduce(kname)
        self.assertEqual(rewritten, kname)
        cfg = parse_flydsl_v2_gemm2_kernel(rewritten)
        self.assertEqual(cfg["epilog"], "reduce")

    def test_legacy_v1_rewriting(self):
        kname = "flydsl_moe2_fp8_fp4_bf16_32x128x256_atomic_bnt2"
        rewritten = fused_moe._rewrite_flydsl_force_reduce(kname)
        expected = "flydsl_moe2_fp8_fp4_bf16_32x128x256_reduce_bnt2"
        self.assertEqual(rewritten, expected)

    def test_native_mxmoe_rewriting(self):
        kname = "flydsl_mxmoe_g2_a4w4_16x128x256_atomic_nt"
        rewritten = fused_moe._rewrite_flydsl_force_reduce(kname)
        expected = "flydsl_mxmoe_g2_a4w4_16x128x256_nt"
        self.assertEqual(rewritten, expected)
        parsed = parse_g2_kname_any(rewritten)
        self.assertFalse(parsed["atomic"])

    def test_native_mxmoe_already_reduce_unchanged(self):
        kname = "flydsl_mxmoe_g2_a4w4_16x128x256_nt"
        rewritten = fused_moe._rewrite_flydsl_force_reduce(kname)
        self.assertEqual(rewritten, kname)
        parsed = parse_g2_kname_any(rewritten)
        self.assertFalse(parsed["atomic"])

    def test_non_flydsl_unchanged(self):
        ck = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1"
        opus = "opus_moe2_gemm_atomic"
        self.assertEqual(fused_moe._rewrite_flydsl_force_reduce(ck), ck)
        self.assertEqual(fused_moe._rewrite_flydsl_force_reduce(opus), opus)
        self.assertEqual(fused_moe._rewrite_flydsl_force_reduce(""), "")
        self.assertIsNone(fused_moe._rewrite_flydsl_force_reduce(None))

    def test_all_manifest_kernels_valid(self):
        csv_pattern = os.path.join(REPO_ROOT, "aiter", "configs", "model_configs", "*.csv")
        csv_files = glob.glob(csv_pattern)
        self.assertGreater(len(csv_files), 0, "No model CSV configs found")

        names = set()
        for f in csv_files:
            with open(f, encoding="utf-8") as fp:
                reader = csv.DictReader(fp)
                for row in reader:
                    kn2 = row.get("kernelName2", "")
                    if kn2 and kn2.startswith("flydsl_"):
                        names.add(kn2)

        self.assertGreater(len(names), 100, f"Expected >100 FlyDSL stage2 kernels, found {len(names)}")

        for name in sorted(names):
            rewritten = fused_moe._rewrite_flydsl_force_reduce(name)
            if name.startswith("flydsl_moe2_layout_"):
                cfg = parse_flydsl_v2_gemm2_kernel(rewritten)
                self.assertIsNotNone(cfg, f"Failed parsing {rewritten}")
                self.assertEqual(cfg["epilog"], "reduce", f"Expected reduce epilog in {rewritten}")
            elif name.startswith("flydsl_mxmoe_g2_"):
                parsed = parse_g2_kname_any(rewritten)
                self.assertIsNotNone(parsed, f"Failed parsing {rewritten}")
                self.assertFalse(parsed["atomic"], f"Expected non-atomic in {rewritten}")
            elif name.startswith("flydsl_moe2_"):
                self.assertIn("_reduce", rewritten, f"Expected _reduce in {rewritten}")


class TestFlydslForceReduceControlPlane(unittest.TestCase):
    """Test full Control Plane & Coordination Plane dispatch in get_2stage_cfgs()."""

    def setUp(self):
        self.config_file = os.path.join(
            REPO_ROOT, "aiter", "configs", "model_configs", "dsv3_fp4_tuned_fmoe.csv"
        )
        self.assertTrue(os.path.exists(self.config_file))
        fused_moe.AITER_CONFIGS.AITER_CONFIG_FMOE_FILE = self.config_file

    def test_modern_v2_default_and_force_reduce(self):
        # Token 32 in dsv3_fp4 has flydsl_moe2_layout_afp4_wfp4_bf16_t32x256x256_atomic_nt_sbm32

        # 1. Unset / default: stays atomic
        with patch.dict(os.environ, {"AITER_FLYDSL_FORCE_REDUCE": "0"}, clear=False):
            fused_moe.get_2stage_cfgs.cache_clear()
            fused_moe.cfg_2stages = None
            meta_default = fused_moe.get_2stage_cfgs(
                token=32,
                model_dim=7168,
                inter_dim=256,
                expert=257,
                topk=9,
                dtype="torch.bfloat16",
                q_dtype_a="torch.float4_e2m1fn_x2",
                q_dtype_w="torch.float4_e2m1fn_x2",
                q_type=QuantType.per_1x32,
                use_g1u1=1,
                activation=ActivationType.Silu,
                doweight_stage1=0,
                hidden_pad=0,
                intermediate_pad=0,
                is_shuffled=True,
                gate_mode=GateMode.SEPARATED,
            )
            kn2_default = meta_default.stage2.keywords["kernelName"]
            self.assertIn("_atomic", kn2_default)
            self.assertFalse(fused_moe.stage2_uses_route_reduce(meta_default.stage2))
            accumulate_contract = not fused_moe.stage2_uses_route_reduce(meta_default.stage2)
            self.assertTrue(accumulate_contract)

        # 2. Force reduce = 1: rewrites to reduce, upstream sorting accumulate becomes False
        with patch.dict(os.environ, {"AITER_FLYDSL_FORCE_REDUCE": "1"}, clear=False):
            fused_moe.get_2stage_cfgs.cache_clear()
            fused_moe.cfg_2stages = None
            meta_reduce = fused_moe.get_2stage_cfgs(
                token=32,
                model_dim=7168,
                inter_dim=256,
                expert=257,
                topk=9,
                dtype="torch.bfloat16",
                q_dtype_a="torch.float4_e2m1fn_x2",
                q_dtype_w="torch.float4_e2m1fn_x2",
                q_type=QuantType.per_1x32,
                use_g1u1=1,
                activation=ActivationType.Silu,
                doweight_stage1=0,
                hidden_pad=0,
                intermediate_pad=0,
                is_shuffled=True,
                gate_mode=GateMode.SEPARATED,
            )
            kn2_reduce = meta_reduce.stage2.keywords["kernelName"]
            self.assertIn("_reduce", kn2_reduce)
            self.assertNotIn("_atomic", kn2_reduce)
            self.assertTrue(fused_moe.stage2_uses_route_reduce(meta_reduce.stage2))
            accumulate_contract = not fused_moe.stage2_uses_route_reduce(meta_reduce.stage2)
            self.assertFalse(accumulate_contract)

    def test_already_reduce_config_preserved(self):
        # Token 4 in dsv3_fp4 is already reduce: flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_reduce_sbm32
        for env_val in ("0", "1"):
            with patch.dict(os.environ, {"AITER_FLYDSL_FORCE_REDUCE": env_val}, clear=False):
                fused_moe.get_2stage_cfgs.cache_clear()
                fused_moe.cfg_2stages = None
                meta = fused_moe.get_2stage_cfgs(
                    token=4,
                    model_dim=7168,
                    inter_dim=256,
                    expert=257,
                    topk=9,
                    dtype="torch.bfloat16",
                    q_dtype_a="torch.float4_e2m1fn_x2",
                    q_dtype_w="torch.float4_e2m1fn_x2",
                    q_type=QuantType.per_1x32,
                    use_g1u1=1,
                    activation=ActivationType.Silu,
                    doweight_stage1=0,
                    hidden_pad=0,
                    intermediate_pad=0,
                    is_shuffled=True,
                    gate_mode=GateMode.SEPARATED,
                )
                kn2 = meta.stage2.keywords["kernelName"]
                self.assertIn("_reduce", kn2)
                self.assertTrue(fused_moe.stage2_uses_route_reduce(meta.stage2))

    def test_mxfp4_stage2_wrapper_support(self):
        # Test stage2_uses_route_reduce directly with _mxfp4_a4w4_stage2_fw
        atomic_stage2 = functools.partial(
            fused_moe._mxfp4_a4w4_stage2_fw,
            kernelName2="flydsl_mxmoe_g2_a4w4_16x128x256_atomic_nt"
        )
        self.assertFalse(fused_moe.stage2_uses_route_reduce(atomic_stage2))

        reduce_stage2 = functools.partial(
            fused_moe._mxfp4_a4w4_stage2_fw,
            kernelName2="flydsl_mxmoe_g2_a4w4_16x128x256_nt"
        )
        self.assertTrue(fused_moe.stage2_uses_route_reduce(reduce_stage2))

    def test_mxfp4_atomic_to_reduce_detection(self):
        """Verify MXFP4 atomic=True -> False route-reduce detection and atomic=False -> True."""
        # 1. Native MXMOE kernel:
        kname_atomic = "flydsl_mxmoe_g2_a4w4_16x128x256_atomic_nt"
        parsed_atomic = parse_g2_kname_any(kname_atomic)
        self.assertTrue(parsed_atomic["atomic"])
        stage2_atomic = functools.partial(fused_moe._mxfp4_a4w4_stage2_fw, kernelName2=kname_atomic)
        self.assertFalse(fused_moe.stage2_uses_route_reduce(stage2_atomic))

        kname_reduce = "flydsl_mxmoe_g2_a4w4_16x128x256_nt"
        parsed_reduce = parse_g2_kname_any(kname_reduce)
        self.assertFalse(parsed_reduce["atomic"])
        stage2_reduce = functools.partial(fused_moe._mxfp4_a4w4_stage2_fw, kernelName2=kname_reduce)
        self.assertTrue(fused_moe.stage2_uses_route_reduce(stage2_reduce))

        # 2. Layout V2 kernel under _mxfp4_a4w4_stage2_fw:
        kname_v2_atomic = "flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_atomic_persist_nt_sbm32"
        parsed_v2_atomic = parse_g2_kname_any(kname_v2_atomic)
        self.assertTrue(parsed_v2_atomic["atomic"])
        stage2_v2_atomic = functools.partial(fused_moe._mxfp4_a4w4_stage2_fw, kernelName2=kname_v2_atomic)
        self.assertFalse(fused_moe.stage2_uses_route_reduce(stage2_v2_atomic))

        kname_v2_reduce = "flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_reduce_persist_nt_sbm32"
        parsed_v2_reduce = parse_g2_kname_any(kname_v2_reduce)
        self.assertFalse(parsed_v2_reduce["atomic"])
        stage2_v2_reduce = functools.partial(fused_moe._mxfp4_a4w4_stage2_fw, kernelName2=kname_v2_reduce)
        self.assertTrue(fused_moe.stage2_uses_route_reduce(stage2_v2_reduce))

    def test_kernel_name_and_kernel_name2_recovery_every_implementation(self):
        """Verify that kernelName and kernelName2 are correctly recovered for every Stage-2 implementation."""
        v2_reduce_kname = "flydsl_moe2_layout_afp4_wfp4_bf16_t32x128x256_reduce_persist_nt_sbm32"
        native_reduce_kname = "flydsl_mxmoe_g2_a4w4_16x128x256_nt"

        # 1. _flydsl_v2_stage2_wrapper bound via kernelName
        s2_v2_kn = functools.partial(fused_moe._flydsl_v2_stage2_wrapper, kernelName=v2_reduce_kname)
        self.assertTrue(fused_moe.stage2_uses_route_reduce(s2_v2_kn))

        # 2. _flydsl_v2_stage2_wrapper bound via kernelName2
        s2_v2_kn2 = functools.partial(fused_moe._flydsl_v2_stage2_wrapper, kernelName2=v2_reduce_kname)
        self.assertTrue(fused_moe.stage2_uses_route_reduce(s2_v2_kn2))

        # 3. _mxfp4_a4w4_stage2_fw bound via kernelName2
        s2_mx_kn2 = functools.partial(fused_moe._mxfp4_a4w4_stage2_fw, kernelName2=native_reduce_kname)
        self.assertTrue(fused_moe.stage2_uses_route_reduce(s2_mx_kn2))

        # 4. _mxfp4_a4w4_stage2_fw bound via kernelName
        s2_mx_kn = functools.partial(fused_moe._mxfp4_a4w4_stage2_fw, kernelName=native_reduce_kname)
        self.assertTrue(fused_moe.stage2_uses_route_reduce(s2_mx_kn))

        # 5. Non-FlyDSL / unknown wrappers return False without error
        def dummy_s2():
            pass
        self.assertFalse(fused_moe.stage2_uses_route_reduce(dummy_s2))

    def test_fallback_heuristic_configuration_path(self):
        """Verify fallback configuration path applies force-reduce rewrite correctly."""
        token = 32
        model_dim = 7168
        inter_dim = 256
        expert = 257
        topk = 9

        def mock_flydsl_kernel_name(stage, a_dtype, b_dtype, out_dtype, tile_m, tile_n, tile_k, mode="", sort_block_m=0):
            name = f"flydsl_moe{stage}_a{a_dtype}_w{b_dtype}_{out_dtype}_t{tile_m}x{tile_n}x{tile_k}"
            if mode:
                name += f"_{mode}"
            if sort_block_m > 0 and sort_block_m != tile_m:
                name += f"_sbm{sort_block_m}"
            return name

        def mock_pick_flydsl_stage2_tile_k(inter_dim):
            return 256 if (inter_dim % 256 == 0) else 128

        def mock_get_flydsl_kernel_params(kname):
            if "_atomic" in kname:
                return {"mode": "atomic", "a_dtype": "fp8", "b_dtype": "fp4"}
            if "_reduce" in kname:
                return {"mode": "reduce", "a_dtype": "fp8", "b_dtype": "fp4"}
            return None

        mock_moe_kernels = MagicMock()
        mock_moe_kernels.flydsl_kernel_name = mock_flydsl_kernel_name
        mock_moe_kernels.pick_flydsl_stage2_tile_k = mock_pick_flydsl_stage2_tile_k
        mock_moe_kernels.get_flydsl_kernel_params = mock_get_flydsl_kernel_params

        # Case 1: AITER_FLYDSL_FORCE_REDUCE unset / "0" -> fallback generates atomic kernel
        with patch.dict(os.environ, {"AITER_FLYDSL_FORCE_REDUCE": "0", "AITER_MXMOE_FALLBACK": "0"}, clear=False), \
             patch.dict(sys.modules, {"aiter.ops.flydsl.moe_kernels": mock_moe_kernels}):
            fused_moe.get_2stage_cfgs.cache_clear()
            fused_moe.cfg_2stages = {}
            with patch.object(fused_moe, "_get_flydsl_moe_kernels", return_value=mock_moe_kernels):
                meta_fallback_0 = fused_moe.get_2stage_cfgs(
                    token=token,
                    model_dim=model_dim,
                    inter_dim=inter_dim,
                    expert=expert,
                    topk=topk,
                    dtype="torch.bfloat16",
                    q_dtype_a="torch.float8",
                    q_dtype_w="torch.float4_e2m1fn_x2",
                    q_type=QuantType.per_1x32,
                    use_g1u1=1,
                    activation=ActivationType.Silu,
                    doweight_stage1=0,
                    hidden_pad=0,
                    intermediate_pad=0,
                    is_shuffled=True,
                    gate_mode=GateMode.INTERLEAVE,
                )
                kn2_0 = meta_fallback_0.stage2.keywords["kernelName"]
                self.assertIn("_atomic", kn2_0)
                self.assertNotIn("_reduce", kn2_0)
                self.assertFalse(fused_moe.stage2_uses_route_reduce(meta_fallback_0.stage2))

        # Case 2: AITER_FLYDSL_FORCE_REDUCE="1" -> fallback rewrites kn2 to reduce
        with patch.dict(os.environ, {"AITER_FLYDSL_FORCE_REDUCE": "1", "AITER_MXMOE_FALLBACK": "0"}, clear=False), \
             patch.dict(sys.modules, {"aiter.ops.flydsl.moe_kernels": mock_moe_kernels}):
            fused_moe.get_2stage_cfgs.cache_clear()
            fused_moe.cfg_2stages = {}
            with patch.object(fused_moe, "_get_flydsl_moe_kernels", return_value=mock_moe_kernels):
                meta_fallback_1 = fused_moe.get_2stage_cfgs(
                    token=token,
                    model_dim=model_dim,
                    inter_dim=inter_dim,
                    expert=expert,
                    topk=topk,
                    dtype="torch.bfloat16",
                    q_dtype_a="torch.float8",
                    q_dtype_w="torch.float4_e2m1fn_x2",
                    q_type=QuantType.per_1x32,
                    use_g1u1=1,
                    activation=ActivationType.Silu,
                    doweight_stage1=0,
                    hidden_pad=0,
                    intermediate_pad=0,
                    is_shuffled=True,
                    gate_mode=GateMode.INTERLEAVE,
                )
                kn2_1 = meta_fallback_1.stage2.keywords["kernelName"]
                self.assertIn("_reduce", kn2_1)
                self.assertNotIn("_atomic", kn2_1)
                self.assertTrue(fused_moe.stage2_uses_route_reduce(meta_fallback_1.stage2))



class TestFlydslForceReduceDocumentationAndComments(unittest.TestCase):
    """Test that comments in production source accurately state default behavior."""

    def test_moe_kernels_comment_not_inverted(self):
        moe_kernels_path = os.path.join(
            REPO_ROOT, "aiter", "ops", "flydsl", "moe_kernels.py"
        )
        with open(moe_kernels_path, encoding="utf-8") as f:
            content = f.read()

        self.assertIn(
            "Disabled by default; set AITER_FLYDSL_FORCE_REDUCE=1 to enable.",
            content,
            "Comment inversion defect still present in moe_kernels.py!",
        )
        self.assertNotIn(
            "Enabled by default; set AITER_FLYDSL_FORCE_REDUCE=0 to opt out.",
            content,
            "Old inverted comment still found in moe_kernels.py!",
        )


if __name__ == "__main__":
    unittest.main()
