# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Marlowe AI
"""Prepared exact-M8 native MoE consumer route merge.

Opt-in only. Supplied native top-8 IDs are distinct within each token; column
eight is expert 256. The actual FP32 weight of that shared route is preserved.
No router or collective is owned here. Other shapes retain public fused_moe.
Handles own per-layer scratch and support serial eager/captured-graph reuse.
"""


def supported_geometry(rows, arch, w1_shape, w2_shape, fp4_weights, shuffled):
    return (
        rows == 8
        and arch.split(":", 1)[0] == "gfx950"
        and tuple(w1_shape) == (257, 512, 3072)
        and tuple(w2_shape) == (257, 6144, 128)
        and fp4_weights
        and shuffled
    )


class PreparedRouteMerge8:
    reference_contract = {
        "input_rounding": "flydsl_roundup_0x7fffff",
        "middle_rounding": "flydsl_roundup_0x7fffff",
        "fp32_middle": True,
        "output": "bf16_expert_operands",
    }

    def __init__(self, weights, rows, *, prefetch_hidden=False, use_nt_g2=True):
        import torch

        from .kernels.mxfp4_gemm_common import kas_per_chunk_dw_for

        self.weights, self.rows = dict(weights), rows
        self.device = weights["w1"].device
        arch = torch.cuda.get_device_properties(self.device).gcnArchName
        self.enabled = supported_geometry(
            rows,
            arch,
            weights["w1"].shape,
            weights["w2"].shape,
            all(weights[k].dtype == torch.float4_e2m1fn_x2 for k in ("w1", "w2")),
            all(getattr(weights[k], "is_shuffled", False) for k in ("w1", "w2")),
        ) and all(
            weights[k].is_contiguous() and weights[k].device == self.device
            for k in weights
        )
        self.enabled = self.enabled and all(
            weights[k].dtype == torch.float8_e8m0fnu and weights[k].numel() == size
            for k, size in (("w1_scale", 257 * 512 * 192), ("w2_scale", 257 * 6144 * 8))
        )
        self.info = {
            "backend": "native exact-M8 consumer ballot merge",
            "enabled": self.enabled,
            "rows": rows,
            "middle": "65 stable slot tiles x 16 rows, native packed MXFP4/scales",
            "routes": "supplied native8 distinct/token plus shared256 in column8 once",
            "reference_contract": self.reference_contract,
            "static_kwargs": {
                "prefetch_hidden": prefetch_hidden,
                "use_nt_g2": use_nt_g2,
            },
        }
        self.scratch = []
        if not self.enabled:
            return
        from .kernels.mxfp4_gemm1 import compile_gemm1_a4w4_port
        from .kernels.mxfp4_gemm2 import compile_gemm2_a4w4_port

        self.middle = torch.empty((65 * 16, 128), dtype=torch.uint8, device=self.device)
        self.scales = torch.empty(
            65 * kas_per_chunk_dw_for(256) * 4, dtype=torch.uint8, device=self.device
        )
        self.out = torch.empty((8, 6144), dtype=torch.bfloat16, device=self.device)
        self.scratch = [self.middle, self.scales, self.out]
        self.scratch_identity = tuple(
            (v.data_ptr(), tuple(v.shape), v.dtype, v.device) for v in self.scratch
        )
        self.g1 = compile_gemm1_a4w4_port(
            BM=16,
            use_nt=False,
            inline_quant=True,
            prefetch_hidden=prefetch_hidden,
            D_HIDDEN=6144,
            D_INTER=256,
            NE=257,
            native_scale_layout=True,
            merge_routes8=True,
        )
        self.g2 = compile_gemm2_a4w4_port(
            BM=16,
            use_nt=use_nt_g2,
            NE=257,
            N_OUT=6144,
            D_INTER=256,
            merge_routes8=True,
        )

    def run(self, x, ids, route_weights):
        import aiter
        import torch

        if not (
            self.enabled
            and tuple(x.shape) == (8, 6144)
            and tuple(ids.shape) == tuple(route_weights.shape) == (8, 9)
            and x.dtype == torch.bfloat16
            and ids.dtype == torch.int32
            and route_weights.dtype == torch.float32
            and all(
                v.is_contiguous() and v.device == self.device
                for v in (x, ids, route_weights)
            )
        ):
            from aiter.fused_moe import GateMode, fused_moe

            return fused_moe(
                x,
                **self.weights,
                topk_ids=ids,
                topk_weight=route_weights,
                activation=aiter.ActivationType.Silu,
                quant_type=aiter.QuantType.per_1x32,
                gate_mode=GateMode.SEPARATED.value,
            )
        if torch.cuda.current_device() != self.device.index:
            raise ValueError("M8 handle must execute on its weight device")
        stream = torch.cuda.current_stream(self.device)
        w = self.weights
        dummy = self.out.data_ptr()
        self.out.zero_()
        self.g1(
            dummy,
            dummy,
            w["w1"].data_ptr(),
            w["w1_scale"].data_ptr(),
            ids.data_ptr(),
            dummy,
            ids.data_ptr(),
            8,
            65 * 2,
            self.middle.data_ptr(),
            self.scales.data_ptr(),
            x.data_ptr(),
            dummy,
            stream,
        )
        self.g2(
            self.middle.data_ptr(),
            self.scales.data_ptr(),
            w["w2"].data_ptr(),
            w["w2_scale"].data_ptr(),
            ids.data_ptr(),
            dummy,
            ids.data_ptr(),
            route_weights.data_ptr(),
            8,
            65,
            dummy,
            dummy,
            stream,
        )
        return self.out

    def state_check(self):
        current = tuple(
            (v.data_ptr(), tuple(v.shape), v.dtype, v.device) for v in self.scratch
        )
        return {
            "passed": not self.enabled or current == self.scratch_identity,
            "scratch_contract": "private per-handle scratch; output zero each call; no counters",
            "accuracy_state_checks": "performed separately by alternating/zero graph qualification",
        }


def make_operator(*, weights, rows, prefetch_hidden=False, use_nt_g2=True):
    return PreparedRouteMerge8(
        weights, rows, prefetch_hidden=prefetch_hidden, use_nt_g2=use_nt_g2
    )
