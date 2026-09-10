# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""CPU contracts for FP8 gather output with descriptor and wide cache addressing."""

import importlib
from unittest.mock import Mock

import pytest
import torch

gather = importlib.import_module("aiter.ops.flydsl.gather_kv_b_proj")


@pytest.fixture
def launch(monkeypatch):
    # Meta tensors model multi-GiB caches without storage or GPU initialization.
    monkeypatch.setattr(gather, "_arch_of", lambda device: "gfx950")
    monkeypatch.setattr(gather, "get_rocm_arch", lambda: "gfx950")
    compile_op = Mock(return_value=object())
    run_op = Mock()
    monkeypatch.setattr(gather, "compile_gather_kv_b_proj", compile_op)
    monkeypatch.setattr(gather, "_run_compiled", run_op)
    monkeypatch.setattr(gather, "ptr_arg", lambda tensor: tensor)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: None)
    monkeypatch.setattr(gather.fx, "Stream", lambda stream: stream)
    return compile_op, run_op


def _case(num_blocks, dtype):
    def tensor(shape, dtype=torch.float32):
        return torch.empty(shape, dtype=dtype, device="meta")

    return {
        "k_buffer": tensor((num_blocks, 1, 576), torch.float8_e4m3fn),
        "k_scale": tensor((1,)),
        "kv_indptr": tensor((2,), torch.int32),
        "kv_indices": tensor((128,), torch.int32),
        "kv_prefix_sum_context_lens": tensor((2,), torch.int32),
        "kv_proj_weight": tensor((12 * 256, 512), torch.float8_e4m3fn),
        "kv_proj_scale": tensor((12 * 256, 1)),
        "k_prefix": tensor((128, 12, 192), dtype),
        "v_prefix": tensor((128, 12, 128), dtype),
    }


def _supported(case):
    return gather.gather_kv_b_proj_flydsl_supported(
        *(
            case[name]
            for name in (
                "k_buffer",
                "kv_proj_weight",
                "kv_proj_scale",
                "k_prefix",
                "v_prefix",
            )
        )
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    "num_blocks", [128, (2**32 - 1) // 576, (2**32 - 1) // 576 + 1]
)
def test_output_scales_and_cache_extent_reach_launcher(launch, dtype, num_blocks):
    compile_op, run_op = launch
    case = _case(num_blocks, dtype)
    assert _supported(case)
    fp8 = dtype == torch.float8_e4m3fn
    scales = (
        {
            name: torch.empty((1,), device="meta")
            for name in ("k_out_scale", "v_out_scale")
        }
        if fp8
        else {}
    )
    gather.gather_kv_b_proj_flydsl(**case, **scales, num_tokens=70)
    assert compile_op.call_args.kwargs["wide_index"] == (num_blocks * 576 >= 2**32)
    assert compile_op.call_args.kwargs["output_fp8"] == fp8
    args = run_op.call_args.args
    assert args[1] is case["k_buffer"]
    assert args[2] == num_blocks
    assert args[-2] == 70
    assert args[7].dtype == (torch.int8 if fp8 else torch.bfloat16)
    assert args[8].dtype == (torch.int8 if fp8 else torch.bfloat16)
    if fp8:
        assert args[9]._base is scales["k_out_scale"]
        assert args[10]._base is scales["v_out_scale"]
    else:
        assert args[9:11] == (1.0, 1.0)


@pytest.mark.parametrize(
    "invalid",
    [
        "missing_scale",
        "scale_dtype",
        "scale_device",
        "mixed_output",
        "strided_output",
        "output_device",
        "bf16_scale",
    ],
)
def test_invalid_outputs_are_rejected_before_compile(launch, invalid):
    compile_op, run_op = launch
    case = _case(128, torch.float8_e4m3fn)
    scales = {
        name: torch.empty((1,), device="meta")
        for name in ("k_out_scale", "v_out_scale")
    }
    if invalid == "missing_scale":
        scales.pop("k_out_scale")
    elif invalid == "scale_dtype":
        scales["k_out_scale"] = scales["k_out_scale"].to(torch.bfloat16)
    elif invalid == "scale_device":
        scales["k_out_scale"] = torch.empty((1,))
    elif invalid == "mixed_output":
        case["v_prefix"] = case["v_prefix"].to(torch.bfloat16)
    elif invalid == "strided_output":
        case["k_prefix"] = torch.empty(
            (128, 12, 384), dtype=torch.float8_e4m3fn, device="meta"
        )[..., ::2]
    elif invalid == "output_device":
        case["v_prefix"] = torch.empty((128, 12, 128), dtype=torch.float8_e4m3fn)
    else:
        case = _case(128, torch.bfloat16)
    if invalid in ("mixed_output", "strided_output", "output_device"):
        assert not _supported(case)
    with pytest.raises(ValueError):
        gather.gather_kv_b_proj_flydsl(**case, **scales)
    compile_op.assert_not_called()
    run_op.assert_not_called()
