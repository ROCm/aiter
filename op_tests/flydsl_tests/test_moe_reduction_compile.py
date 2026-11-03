# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""GPU-free compilation coverage for the stage2 reducer (requires ROCm lld)."""

import pytest
import torch


@pytest.mark.parametrize("dtype", ["f16", "bf16", "f32", "fp8"])
@pytest.mark.parametrize("guard", ["none", "ids", "mask", "padded"])
def test_moe_reduction_compiles_without_a_gpu(monkeypatch, dtype, guard):
    flyc = pytest.importorskip("flydsl.compiler")
    from aiter.ops.flydsl.kernels.moe_reduce import compile_moe_reduction
    from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_GPU_ARCH", "gfx950")
    monkeypatch.delenv("FLYDSL_RUNTIME_RUN_ONLY", raising=False)

    def forbid_gpu(*args, **kwargs):
        raise AssertionError("compile-only reduction touched the GPU")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbid_gpu)
    launcher = compile_moe_reduction(
        topk=4,
        model_dim=256,
        dtype_str=dtype,
        out_dtype_str="bf16" if dtype == "fp8" else dtype,
        use_mask=guard in ("mask", "padded"),
        # A standalone masked caller need not set skip_invalid_ids too.
        skip_invalid_ids=guard in ("ids", "padded"),
        num_experts=8 if guard in ("mask", "padded") else 0,
        valid_tokens=guard == "padded",
        use_weight=dtype == "fp8",
    )
    dummy = ptr_arg(torch.empty(4096, dtype=torch.uint8, device="cpu"))
    # Constructing the launcher is lazy: compile it to exercise DSL lowering,
    # code generation and linking, including every guarded load path.
    flyc.compile(launcher, dummy, dummy, dummy, dummy, dummy, dummy, 4, 0)
