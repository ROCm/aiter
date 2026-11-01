# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The advertised vec8 gather must use legal accesses, including output tails."""

import importlib

import pytest
import torch

pytest.importorskip("flydsl")
kernel = importlib.import_module("aiter.ops.flydsl.kernels.moe_gather_reduce")
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_arg

CASES = [
    (1, 2, 1, 1),
    (3, 6, 5, 1),
    (2, 14, 5, 2),
    (2, 16, 5, 1),
    (3, 18, 8, 3),
    (9, 128, 32, 2),
    (2, 2046, 5, 1),
    (2, 2048, 5, 2),
    (2, 2050, 5, 1),
    (2, 4094, 5, 2),
    (2, 4096, 5, 1),
    (3, 4098, 5, 3),
    (8, 6144, 5, 1),
    (2, 7168, 8, 2),
]


@pytest.mark.parametrize("geometry", CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_vec8_reference_guards_and_metadata_replay(geometry, dtype, weight_dtype):
    if not torch.cuda.is_available():
        pytest.skip("requires a ROCm GPU")
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() not in ("gfx1250", "gfx950", "gfx942"):
        pytest.skip("ROCm gather kernel")
    tokens, dim, topk, split = geometry
    torch.manual_seed(1048)
    source_rows = max(33, tokens * topk)
    grouped = torch.randn(split, source_rows, dim, device="cuda", dtype=dtype) * 0.2
    # A dropped route must not load row zero, even if its weight is nonzero.
    grouped[:, 0].fill_(float("nan"))
    ids = torch.arange(tokens * topk, device="cuda", dtype=torch.int32)
    ids = (ids % (source_rows - 1) + 1).reshape(tokens, topk)
    ids.reshape(-1)[::4] = -1
    weights = torch.randn(tokens, topk, device="cuda", dtype=weight_dtype) * 0.2
    valid = torch.tensor([tokens], device="cuda", dtype=torch.int32)
    # Two-element offset is 4-byte aligned but deliberately not 16-byte aligned.
    guard = [
        torch.full((tokens * dim + 10,), 73, device="cuda", dtype=dtype)
        for _ in range(2)
    ]
    outputs = [p[2:-8].view(tokens, dim) for p in guard]
    out_type = "bf16" if dtype == torch.bfloat16 else "f16"
    weight_type = {torch.float32: "f32", torch.bfloat16: "bf16", torch.float16: "f16"}[
        weight_dtype
    ]
    launchers = [
        kernel.build_moe_gather_reduce_module(
            dim, topk, out_type, split, v, weight_type
        )
        for v in (4, 8)
    ]

    def run(i):
        _run_compiled(
            launchers[i],
            ptr_arg(grouped),
            ptr_arg(ids),
            ptr_arg(weights),
            ptr_arg(outputs[i]),
            tokens,
            source_rows * dim // 2,
            ptr_arg(valid),
            torch.cuda.current_stream(),
        )

    def check():
        active = int(valid.item())
        expected = torch.full_like(outputs[0], 73)
        total = torch.zeros((active, dim), device="cuda")
        for k in range(topk):
            rows = ids[:active, k].long()
            mapped = rows >= 0
            partial = torch.zeros_like(total)
            for sk in range(split):
                values = grouped[sk, rows.clamp(min=1)].float()
                partial = partial + torch.where(mapped[:, None], values, 0)
            total = total + partial * weights[:active, k, None].float()
        expected[:active] = total.to(dtype)
        assert torch.equal(outputs[0], outputs[1])
        for out, backing in zip(outputs, guard):
            torch.testing.assert_close(out, expected, rtol=0.03, atol=0.03)
            assert (backing[:2] == 73).all() and (backing[-8:] == 73).all()
            assert (out[active:] == 73).all()

    run(0)
    run(1)
    check()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run(1)
    torch.cuda.current_stream().wait_stream(stream)
    for step in range(5):
        grouped[:, 1:].neg_()
        weights.mul_(-1.125)
        if step == 0:
            ids.fill_(-1)
        else:
            ids.copy_(
                (
                    torch.arange(
                        tokens * topk, device="cuda", dtype=torch.int32
                    ).reshape(tokens, topk)
                    * (step + 1)
                )
                % (source_rows - 1)
                + 1
            )
            ids.reshape(-1)[::7] = -1
        valid.fill_(0 if step == 1 else tokens // 2 if step == 2 else tokens)
        for out in outputs:
            out.fill_(73)
        graph.replay()
        run(0)
        check()
