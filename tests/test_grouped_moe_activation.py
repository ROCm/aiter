# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only dispatch and buffer checks for standalone GEMM1 activation."""

import os
import unittest
from unittest.mock import patch

import torch

from aiter import ActivationType
from aiter.ops.flydsl import grouped_gemm_mxfp4, moe_kernels
from aiter.ops.flydsl import grouped_moe_gfx1250 as grouped


class GroupedActivationTest(unittest.TestCase):
    def setUp(self):
        # The merged runtime also uses a fused route/psum path. Stub dispatch
        # itself so CPU test pointers can never reach a GPU kernel.
        self.enterContext(patch.object(grouped, "_run_compiled"))
        self.enterContext(patch.object(moe_kernels, "_dispatch_compiled"))
        self.enterContext(patch("torch.cuda.current_stream", return_value=0))
        self.enterContext(
            patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU access"))
        )

    def test_dispatch_buffers_and_benchmark_callables(self):
        for flag in ("0", "1"):
            for apre in (0, 1):
                for act, code in (
                    (ActivationType.Silu, 1),
                    (ActivationType.Swiglu, 2),
                    (ActivationType.Situv2, 3),
                ):
                    captured = []

                    def quant(x, experts, rows, **kw):
                        k = x.shape[-1] // (2 if kw.get("activation") else 1)
                        return (
                            torch.empty((1, rows, k // 2), dtype=torch.uint8),
                            torch.empty((1, rows, k // 32), dtype=torch.uint8),
                        )

                    with (
                        self.subTest(flag=flag, apre=apre, act=act),
                        patch.dict(
                            os.environ,
                            {
                                "AITER_FLYDSL_DISABLE_GEMM1_ACT": flag,
                                "AITER_FLYDSL_DISABLE_GEMM1_REQUANT": "0",
                            },
                        ),
                        patch.object(
                            grouped, "_flydsl_dispatch_context", return_value=None
                        ),
                        patch.object(grouped, "get_wmma_m_rep", return_value=4),
                        patch.object(grouped, "kernel_bench_callable", captured),
                        patch.object(
                            grouped,
                            "contiguous_psum_remap",
                            return_value=(
                                None,
                                torch.empty(4, dtype=torch.int32),
                                None,
                            ),
                        ),
                        patch.object(grouped, "flydsl_moe_gather_reduce"),
                        patch.object(
                            moe_kernels,
                            "flydsl_moe_topids_to_rows",
                            return_value=(
                                torch.empty(4, dtype=torch.int32),
                                torch.empty(16, dtype=torch.int32),
                            ),
                        ),
                        patch.object(
                            moe_kernels,
                            "flydsl_moe_fused_quant_preshuffle",
                            side_effect=quant,
                        ) as q,
                        patch.object(
                            grouped_gemm_mxfp4, "flydsl_grouped_gemm_a8w4_masked"
                        ) as gemm,
                    ):
                        grouped._grouped_a8w4_tdm_moe(
                            torch.empty((8, 1024), dtype=torch.bfloat16),
                            torch.empty(4, dtype=torch.uint8),
                            torch.empty(4, dtype=torch.uint8),
                            torch.empty((8, 2)),
                            torch.empty((8, 2), dtype=torch.int32),
                            E=4,
                            model_dim=1024,
                            inter_dim=1024,
                            dtype=torch.bfloat16,
                            activation=act,
                            w1_scale=torch.empty(4, dtype=torch.uint8),
                            w2_scale=torch.empty(4, dtype=torch.uint8),
                            bias1=None,
                            bias2=None,
                            swiglu_limit=6.0,
                            doweight_stage1=False,
                            data_format="fp4",
                            a_preshuffle=apre,
                            situ_beta=2.0,
                            situ_linear_beta=3.0,
                        )
                    first, second = gemm.call_args_list
                    self.assertEqual(second.kwargs["K"], 1024)
                    self.assertEqual(second.kwargs["a_preshuffled"], apre)
                    if flag == "1":
                        self.assertEqual(first.args[0].shape[-1], 2048)
                        self.assertEqual(first.args[0].dtype, torch.bfloat16)
                        self.assertEqual(first.kwargs["stage1_act"], 0)
                        self.assertEqual(first.kwargs["gemm_stage"], 1)
                        self.assertEqual(first.kwargs.get("stage1_quant_out", 0), 0)
                        self.assertEqual(q.call_args.kwargs["activation"], code)
                        self.assertEqual(q.call_args.kwargs["swiglu_limit"], 6.0)
                        callables = dict(captured)
                        self.assertEqual(
                            callables["quant_a2"].keywords["activation"], code
                        )
                        self.assertEqual(callables["gemm1"].keywords["stage1_act"], 0)
                        self.assertEqual(callables["gemm1"].args[0].shape[-1], 2048)
                    else:
                        self.assertEqual(first.kwargs["stage1_act"], code)
                        self.assertEqual(first.kwargs["stage1_quant_out"], 1)
                        self.assertEqual(q.call_count, 1)

    def test_quant_uses_post_activation_shape(self):
        for apre in (False, True):
            with (
                patch("torch.cuda.current_stream", return_value=0),
                patch(
                    "aiter.ops.flydsl.kernels.kernels_common.get_warp_size",
                    return_value=32,
                ),
                patch.object(
                    moe_kernels, "_get_compiled_fused_quant_preshuffle"
                ) as build,
            ):
                payload, scale = moe_kernels.flydsl_moe_fused_quant_preshuffle(
                    torch.empty((1, 64, 2048), dtype=torch.bfloat16),
                    1,
                    64,
                    wmma_rep=4,
                    activation=1,
                    payload_preshuffled=apre,
                )
                self.assertEqual(payload.shape, (1, 64, 512))
                self.assertEqual(scale.numel(), 64 * 32)
                self.assertEqual(build.call_args.kwargs["feat_dim"], 1024)
                self.assertEqual(build.call_args.kwargs["activation"], 1)


if __name__ == "__main__":
    unittest.main()
