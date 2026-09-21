# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.flydsl.kernels.mqa_logits import pa_mqa_logits_fp4 as decode
from aiter.ops.flydsl.kernels.mqa_logits import pa_mqa_logits_fp4_prefill as prefill
from op_tests import test_flydsl_pa_mqa_logits_fp4 as reference

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires an AMD gfx950 device"
)


def make_case(heads, dim, next_n, page, seed=43):
    reference.setup_seed(seed)
    batch, maximum, padded = 3, 513, 768
    context = torch.tensor([0, 65, maximum], dtype=torch.int32, device="cuda")
    q_dense = torch.randn(
        batch, next_n, heads, dim, dtype=torch.bfloat16, device="cuda"
    )
    kv_dense = torch.randn(batch, padded, dim, dtype=torch.bfloat16, device="cuda")
    weights = (
        torch.randn(batch * next_n, heads, dtype=torch.bfloat16, device="cuda") * 0.1
    )
    q, q_scale = reference.fp4_quant_e2m1_with_e8m0(q_dense.reshape(-1, dim))
    q = q.reshape(batch, next_n, heads, dim // 2)
    q_scale = q_scale.reshape(batch, next_n, heads, dim // 32)
    blocks = batch * padded // page
    table_storage = torch.full(
        (batch, padded // page + 3), -1, dtype=torch.int32, device="cuda"
    )
    table = table_storage[:, : padded // page]
    table.copy_(torch.randperm(blocks, device="cuda").int().reshape_as(table))
    kv, scales, dense_fp4, dense_scales = reference.create_paged_preshuffle_kv_fp4(
        kv_dense, page, blocks, table
    )

    def padded_pages(tensor, gap):
        width = tensor[0].numel()
        storage = torch.zeros((blocks, width + gap), dtype=tensor.dtype, device="cuda")
        view = storage[:, :width].view(tensor.shape)
        view.copy_(tensor)
        return view

    kv = padded_pages(kv, 256)
    scales = padded_pages(scales, 32)
    m = heads // 16
    shuffled = q_scale.reshape(batch, next_n, m, 16, dim // 128, 4)
    shuffled = shuffled.permute(0, 1, 4, 5, 3, 2).contiguous()
    shuffled = torch.nn.functional.pad(shuffled, (0, (m + 3) // 4 * 4 - m)).contiguous()
    storage = torch.full((batch * next_n, maximum + 7), 71.0, device="cuda")
    output = storage[:, :maximum]
    output.fill_(float("-inf"))

    def expected():
        return reference.ref_mqa_logits_mixed(
            q,
            q_scale,
            dense_fp4,
            dense_scales,
            weights,
            context,
            next_n=next_n,
            weight_scale=1.5,
        )[:, :maximum]

    return (
        {
            "q_fp4": q,
            "q_scale": shuffled,
            "kv_cache": kv,
            "kv_scale": scales,
            "block_tables": table,
            "weights": weights,
            "context_lens": context,
            "max_seq_len": maximum,
            "next_n": next_n,
            "kv_block_size": page,
            "weight_scale": 1.5,
            "out": output,
        },
        expected,
        storage,
    )


SHAPES = [
    (16, 128, 1, 64),
    (32, 128, 2, 64),
    (48, 128, 3, 64),
    (64, 128, 1, 64),
    (64, 128, 2, 64),
    (64, 128, 4, 64),
    (64, 128, 5, 64),
    (64, 128, 8, 64),
    (80, 128, 2, 64),
    (96, 128, 3, 64),
    (112, 128, 1, 64),
    (128, 128, 2, 64),
    (64, 256, 1, 64),
    (128, 256, 2, 64),
    (32, 128, 2, 128),
    (64, 128, 1, 128),
    (64, 128, 3, 128),
    (128, 256, 4, 128),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("mode", ["direct", "persistent", "external"])
def test_decode_pipeline(shape, mode, monkeypatch):
    heads, dim, next_n, page = shape
    data, expected, storage = make_case(*shape)

    def forbidden(*args, **kwargs):
        raise AssertionError("Decode must never dispatch the prefill kernel")

    monkeypatch.setattr(prefill, "compile_pa_mqa_logits_fp4_prefill", forbidden)
    rows = data["q_fp4"].shape[0] * next_n
    if mode == "persistent":
        data["parallel_unit_num"] = rows
    elif mode == "external":
        block, waves = decode._default_decode_config(3, next_n, heads, dim, 513, page)
        _, info, total = decode.compute_varctx_schedule(
            data["context_lens"], block, rows, 513, next_n=next_n
        )
        data.update(block_k=block, num_warps=waves, cta_info=info, total_ctas=total)
    actual = decode.flydsl_pa_mqa_logits_fp4(**data)
    torch.testing.assert_close(actual, expected(), rtol=1e-4, atol=5e-4)
    assert torch.all(storage[:, 513:] == 71.0)


@pytest.mark.parametrize("next_n", [1, 2, 3, 4, 5])
def test_decode_graph_updates_context(next_n):
    data, expected, storage = make_case(64, 128, next_n, 64)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        decode.flydsl_pa_mqa_logits_fp4(**data)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        decode.flydsl_pa_mqa_logits_fp4(**data)
    for lengths in ([129, 0, 257], [0, 0, 0], [513, 1, 65]):
        data["context_lens"].copy_(
            torch.tensor(lengths, dtype=torch.int32, device="cuda")
        )
        graph.replay()
        torch.testing.assert_close(data["out"], expected(), rtol=1e-4, atol=5e-4)
        assert torch.all(storage[:, 513:] == 71.0)
