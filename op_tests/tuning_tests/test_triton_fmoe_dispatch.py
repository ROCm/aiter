# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import os
import unittest
from unittest import mock

try:
    from aiter import ActivationType, QuantType, dtypes
    import aiter.fused_moe as fused_moe_mod
    from aiter.ops.flydsl.moe_common import GateMode

    _IMPORT_ERR = None
except Exception as e:  # noqa: BLE001
    ActivationType = None
    QuantType = None
    dtypes = None
    fused_moe_mod = None
    GateMode = None
    _IMPORT_ERR = e


@unittest.skipUnless(fused_moe_mod is not None, f"aiter imports unavailable: {_IMPORT_ERR}")
class TestTritonFmoeDispatch(unittest.TestCase):
    def setUp(self):
        fused_moe_mod._load_triton_fmoe_configs.cache_clear()

    def _call(self, **overrides):
        kwargs = dict(
            M=1152,
            model_dim=2048,
            inter_dim=128,
            expert=256,
            topk=8,
            activation=ActivationType.Silu,
            dtype=dtypes.bf16,
            q_dtype_a=dtypes.bf16,
            q_dtype_w=dtypes.bf16,
            quant_type=QuantType.No,
            is_g1u1=True,
            doweight_stage1=False,
            expert_mask=None,
            hidden_pad=0,
            intermediate_pad=0,
            bias1=None,
            bias2=None,
            w1_scale=None,
            w2_scale=None,
            a1_scale=None,
            a2_scale=None,
            num_local_tokens=None,
            gate_mode=GateMode.SEPARATED,
        )
        kwargs.update(overrides)
        with (
            mock.patch.object(fused_moe_mod, "get_gfx_runtime", return_value="gfx1201"),
            mock.patch.object(fused_moe_mod, "get_cu_num", return_value=64),
        ):
            return fused_moe_mod._get_gfx1201_triton_fmoe_config(**kwargs)

    def test_selected_exact_row_returns_config(self):
        cfg = self._call(M=1152)
        self.assertEqual(
            cfg,
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 4,
                "num_warps": 8,
                "num_stages": 2,
            },
        )

    def test_neighboring_unselected_row_returns_none(self):
        self.assertIsNone(self._call(M=1153))

    def test_inter_dim_256_is_not_enabled(self):
        self.assertIsNone(self._call(M=1088, inter_dim=256))

    def test_non_gfx1201_returns_none(self):
        with (
            mock.patch.object(fused_moe_mod, "get_gfx_runtime", return_value="gfx950"),
            mock.patch.object(fused_moe_mod, "get_cu_num", return_value=64),
        ):
            cfg = fused_moe_mod._get_gfx1201_triton_fmoe_config(
                M=1152,
                model_dim=2048,
                inter_dim=128,
                expert=256,
                topk=8,
                activation=ActivationType.Silu,
                dtype=dtypes.bf16,
                q_dtype_a=dtypes.bf16,
                q_dtype_w=dtypes.bf16,
                quant_type=QuantType.No,
                is_g1u1=True,
                doweight_stage1=False,
                expert_mask=None,
                hidden_pad=0,
                intermediate_pad=0,
                bias1=None,
                bias2=None,
                w1_scale=None,
                w2_scale=None,
                a1_scale=None,
                a2_scale=None,
                num_local_tokens=None,
                gate_mode=GateMode.SEPARATED,
            )
        self.assertIsNone(cfg)

    def test_non_eligible_requests_fall_through(self):
        self.assertIsNone(self._call(quant_type=QuantType.per_1x32))
        self.assertIsNone(self._call(expert_mask=object()))
        self.assertIsNone(self._call(bias1=object()))
        self.assertIsNone(self._call(hidden_pad=128))
        self.assertIsNone(self._call(w1_scale=object()))
        self.assertIsNone(self._call(num_local_tokens=object()))
        self.assertIsNone(self._call(gate_mode=GateMode.INTERLEAVE))

    def test_config_family_resolves(self):
        path = fused_moe_mod.AITER_CONFIGS.AITER_CONFIG_TRITON_FMOE_FILE
        self.assertTrue(os.path.exists(path), path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
