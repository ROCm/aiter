# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Single-GPU correctness for the MiniMax-M3 sparse-PA FP8 epilogue.

Run later on gfx1250 (this test is intentionally not part of the CPU handoff):

    python -m pytest -q -s op_tests/test_flydsl_pa_decode_output_quant.py

The oracle is the exact unfused chain requested by F4:
FlyDSL PA BF16 output followed by
``get_hip_quant(QuantType.per_Token)`` over the flattened GQA16 x D128 row.
Both the decode cluster8 and prefill-style direct specializations are pinned.
"""

import importlib
import importlib.util

import pytest
import torch

from aiter import QuantType, dtypes, get_hip_quant, pertoken_quant
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.kernels.pa_decode_plan import _lds_max_batch


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("ROCm is not available")
    if get_gfx_runtime() != "gfx1250":
        pytest.skip("the fused output epilogue targets gfx1250")
    if importlib.util.find_spec("flydsl") is None:
        pytest.skip("FlyDSL is not installed")
    return torch.device("cuda")


@pytest.mark.parametrize(
    "mode,rows,max_partitions",
    [
        ("cluster8", 1, 8),
        ("cluster8", 4, 8),
        ("cluster8", 16, 8),  # bs=4, Eagle3 verifies four tokens per request
        ("cluster8", 64, 8),
        ("direct", 2517, 1),
        ("direct", 4536, 1),  # traced sparse-prefill token count
    ],
)
def test_sparse_pa_output_quant_matches_separate_kernel(
    device, monkeypatch, mode, rows, max_partitions
):
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    torch.manual_seed(393)
    page, head_dim, max_context = 16, 128, 2048
    pages = max_context // page
    query = torch.randn(
        (rows, 16, head_dim), dtype=torch.bfloat16, device=device
    )
    key = torch.randn(
        (pages, 1, page, head_dim), dtype=torch.bfloat16, device=device
    )
    value = torch.randn_like(key)
    table = (
        torch.arange(pages, dtype=torch.int32, device=device)
        .view(1, -1)
        .expand(rows, -1)
        .contiguous()
    )
    context = (
        torch.full((rows,), max_context, dtype=torch.int32, device=device)
        if mode == "cluster8"
        else torch.arange(rows, dtype=torch.int32, device=device)
        .remainder_(max_context)
        .add_(1)
    )
    key, key_scale = pertoken_quant(key, quant_dtype=dtypes.fp8)
    value, value_scale = pertoken_quant(value, quant_dtype=dtypes.fp8)
    key_cache = key.reshape(pages, 1, 16, 8, 16).permute(0, 1, 3, 2, 4).contiguous()
    value_cache = (
        value.reshape(pages, 1, 1, 16, 128).permute(0, 1, 2, 4, 3).contiguous()
    )
    # Production ATOM raises the default 4096 cap to the planner's gfx1250 LDS
    # limit for sparse prefill. Use the same explicit capacity so rows=4536
    # exercises the 8192 runtime/AOT bucket instead of shrinking the real shape.
    max_batch_size = _lds_max_batch(
        torch.cuda.get_device_properties(device).shared_memory_per_block
    )
    plan = pa.plan_pa_decode(
        context,
        1,
        max_partitions=max_partitions,
        max_batch_size=max_batch_size,
        query_length=1,
    )
    assert plan.max_batch_size == max_batch_size
    scalar_shape = (1, plan.capacity, 16)
    workspace = {
        "exp_sums": torch.empty(scalar_shape, dtype=torch.float32, device=device),
        "max_logits": torch.empty(scalar_shape, dtype=torch.float32, device=device),
        "temporary_output": torch.empty(
            (*scalar_shape, 128), dtype=query.dtype, device=device
        ),
    }
    reference_output = torch.empty_like(query)
    output = torch.empty_like(query)
    output_fp8 = torch.empty_like(query, dtype=dtypes.fp8)
    output_scale = torch.empty((query.shape[0], 1), dtype=torch.float32, device=device)

    compile_calls = []
    compile_tile = pa.compile_pa_decode_tile

    def spy_tile(**kwargs):
        compile_calls.append(kwargs)
        return compile_tile(**kwargs)

    monkeypatch.setattr(pa, "compile_pa_decode_tile", spy_tile)

    def launch(*, quantized):
        pa.pa_decode(
            output if quantized else reference_output,
            query,
            key_cache,
            value_cache,
            context,
            table,
            128**-0.5,
            1,
            compute_type=dtypes.fp8,
            key_scale=key_scale,
            value_scale=value_scale,
            **(
                {
                    "quantized_output": output_fp8,
                    "output_scale": output_scale,
                }
                if quantized
                else {}
            ),
            work_plan=plan,
            max_context_length=max_context,
            **workspace,
        )

    def check():
        assert torch.equal(output, reference_output)
        assert torch.isfinite(output).all()
        ref_fp8, ref_scale = get_hip_quant(QuantType.per_Token)(
            output.view(output.shape[0], -1), quant_dtype=dtypes.fp8
        )
        assert torch.equal(output_scale, ref_scale)
        assert torch.equal(
            output_fp8.view(torch.uint8),
            ref_fp8.view_as(output_fp8).view(torch.uint8),
        )
        torch.testing.assert_close(
            output_fp8.float().view(output.shape[0], -1) * output_scale,
            ref_fp8.float() * ref_scale,
            rtol=0,
            atol=0,
        )

    launch(quantized=False)
    launch(quantized=True)
    check()
    selected = compile_calls[-1]
    assert selected["quant_output"] is True
    if mode == "cluster8":
        assert selected["cluster_merge"] == 8
        assert selected["direct_output"] is False
    else:
        assert selected["cluster_merge"] == 0
        assert selected["direct_output"] is True

    # Replay must read new query values and must not retain scale/max scratch.
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        launch(quantized=True)
    torch.cuda.current_stream().wait_stream(capture_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        launch(quantized=True)
    query.neg_()
    output.fill_(float("nan"))
    output_fp8.view(torch.uint8).fill_(0x7F)
    output_scale.fill_(float("nan"))
    graph.replay()
    launch(quantized=False)
    check()
