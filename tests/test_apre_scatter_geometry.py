# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only validation of the full-row A-preshuffle scatter's K contract."""

import os
import unittest
from itertools import product
from unittest.mock import patch

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from aiter.ops.flydsl import moe_kernels
from aiter.ops.flydsl.kernels.moe_apreshuffled_quant_scatter import (
    build_moe_apre_quant_module,
    build_moe_apre_scatter_module,
    validate_apre_k,
)


class ApreScatterGeometryTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, FLYDSL_GPU_ARCH="gfx1250"))

    def test_all_complete_slices_through_16384(self):
        for k in range(1024, 16385, 1024):
            with self.subTest(k=k):
                validate_apre_k(k)
                self.assertTrue(callable(build_moe_apre_quant_module(k, 8)))
                self.assertTrue(callable(build_moe_apre_scatter_module(k, 8)))

    def test_invalid_k_rejected_before_building_quant_layout(self):
        for k, builder in product(
            (-1024, 0, 128, 256, 512, 1536, 7169, 16385, 17408),
            (build_moe_apre_quant_module, build_moe_apre_scatter_module),
        ):
            with (
                self.subTest(k=k, builder=builder.__name__),
                patch(
                    "aiter.ops.flydsl.kernels.moe_apreshuffled_quant_scatter.route_quant._quant_layout"
                ) as layout,
                self.assertRaisesRegex(ValueError, "0 < K <= 16384"),
            ):
                builder(k, 8)
            layout.assert_not_called()

    def test_runtime_rejects_before_quant_launch(self):
        for k, prequantized in product((128, 256, 512, 1536, 17408), (False, True)):
            with (
                self.subTest(k=k, prequantized=prequantized),
                FakeTensorMode(),
                patch.object(moe_kernels, "_get_compiled_apre_quant") as quant,
                patch.object(moe_kernels, "_get_compiled_apre_scatter") as scatter,
            ):
                x = torch.empty(
                    (2, k // 2 if prequantized else k),
                    dtype=torch.uint8 if prequantized else torch.bfloat16,
                    device="cpu",
                )
                scales = (
                    torch.empty((2, k // 32), dtype=torch.uint8, device="cpu")
                    if prequantized
                    else None
                )
                with self.assertRaisesRegex(ValueError, "0 < K <= 16384"):
                    moe_kernels.flydsl_moe_fused_quant_preshuffle(
                        x,
                        1,
                        128,
                        wmma_rep=8,
                        topids_to_rows=torch.empty(4, dtype=torch.int32, device="cpu"),
                        source_topk=2,
                        row_to_token=torch.empty(128, dtype=torch.int32, device="cpu"),
                        prequantized_scale=scales,
                        payload_preshuffled=True,
                    )
                quant.assert_not_called()
                scatter.assert_not_called()

    def test_non_routed_preshuffle_uses_the_same_contract(self):
        with (
            FakeTensorMode(),
            patch.object(moe_kernels, "_get_compiled_fused_quant_preshuffle") as quant,
            self.assertRaisesRegex(ValueError, "K divisible by 1024"),
        ):
            moe_kernels.flydsl_moe_fused_quant_preshuffle(
                torch.empty((1, 128, 256), dtype=torch.bfloat16, device="cpu"),
                1,
                128,
                wmma_rep=8,
                payload_preshuffled=True,
            )
        quant.assert_not_called()

    def test_runtime_accepts_16384_bf16_and_packed_fp4(self):
        for scale_stride in (None, 512, 544):
            with (
                self.subTest(scale_stride=scale_stride),
                FakeTensorMode(),
                patch.object(moe_kernels, "_get_compiled_apre_quant") as quant,
                patch.object(moe_kernels, "_get_compiled_apre_scatter") as scatter,
                patch("torch.cuda.current_stream", return_value=0),
                patch("aiter.jit.utils.chip_info.get_cu_num", return_value=256),
            ):
                prequantized = scale_stride is not None
                x = torch.empty(
                    (2, 8192 if prequantized else 16384),
                    dtype=torch.uint8 if prequantized else torch.bfloat16,
                    device="cpu",
                )
                scales = (
                    torch.empty((2, scale_stride), dtype=torch.uint8, device="cpu")
                    if prequantized
                    else None
                )
                moe_kernels.flydsl_moe_fused_quant_preshuffle(
                    x,
                    1,
                    128,
                    wmma_rep=8,
                    topids_to_rows=torch.empty(4, dtype=torch.int32, device="cpu"),
                    source_topk=2,
                    row_to_token=torch.empty(128, dtype=torch.int32, device="cpu"),
                    prequantized_scale=scales,
                    payload_preshuffled=True,
                )
                if prequantized:
                    quant.assert_not_called()
                else:
                    quant.assert_called_once_with(16384, 8)
                    quant.return_value.assert_called_once()
                scatter.assert_called_once_with(
                    16384, 8, src_scale_stride_bytes=scale_stride or 512
                )
                scatter.return_value.assert_called_once()


if __name__ == "__main__":
    unittest.main()
