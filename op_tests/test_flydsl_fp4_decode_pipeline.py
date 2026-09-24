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


def test_decode_reuses_compiled_launcher_with_new_tensors(monkeypatch):
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_decode import (
        compile_pa_mqa_logits_fp4_decode,
    )

    compile_pa_mqa_logits_fp4_decode.cache_clear()
    cases = [make_case(64, 128, 1, 64, seed=seed) for seed in (43, 44)]
    for data, _, _ in cases:
        rows = data["q_fp4"].shape[0]
        block, waves = decode._default_decode_config(3, 1, 64, 128, 513, 64)
        _, info, total = decode.compute_varctx_schedule(
            data["context_lens"], block, rows, 513
        )
        data.update(
            block_k=block,
            num_warps=waves,
            cta_info=info,
            total_ctas=total,
        )
    launches = []
    run_compiled = decode._run_compiled

    def tracked_run_compiled(launcher, *args):
        compiled_before = getattr(launcher, "_cf", None)
        run_compiled(launcher, *args)
        launches.append((launcher, compiled_before, getattr(launcher, "_cf", None)))

    monkeypatch.setattr(decode, "_run_compiled", tracked_run_compiled)
    for data, expected, storage in cases:
        actual = decode.flydsl_pa_mqa_logits_fp4(**data)
        torch.testing.assert_close(actual, expected(), rtol=1e-4, atol=5e-4)
        assert torch.all(storage[:, 513:] == 71.0)

    assert launches[0][0] is launches[1][0]
    assert launches[0][1] is None
    assert launches[0][2] is not None
    assert launches[1][1] is launches[0][2]
    assert launches[1][2] is launches[0][2]


@pytest.mark.parametrize(
    ("block_k", "num_warps"), [(None, None), (64, 2)], ids=["auto", "block64"]
)
def test_decode_direct_clears_caller_output_outside_context(block_k, num_warps):
    data, expected, storage = make_case(64, 128, 1, 64)
    data.update(block_k=block_k, num_warps=num_warps)
    output = data["out"]
    output.fill_(71.0)

    actual = decode.flydsl_pa_mqa_logits_fp4(**data)
    torch.testing.assert_close(actual, expected(), rtol=1e-4, atol=5e-4)
    assert torch.all(storage[:, 513:] == 71.0)


def test_decode_direct_can_leave_masked_tail_untouched():
    data, expected, storage = make_case(64, 128, 1, 64)
    output = data["out"]
    output.fill_(71.0)
    data["clean_logits"] = False

    actual = decode.flydsl_pa_mqa_logits_fp4(**data)
    reference_output = expected()
    for row, length in enumerate(data["context_lens"].tolist()):
        torch.testing.assert_close(
            actual[row, :length], reference_output[row, :length], rtol=1e-4, atol=5e-4
        )
        assert torch.all(actual[row, length:] == 71.0)
    assert torch.all(storage[:, 513:] == 71.0)


def test_prefill_schedule_does_not_read_back_gpu_count(monkeypatch):
    rows = 4
    launch_ctas = 8
    row_to_batch = torch.zeros(rows, dtype=torch.int32, device="cuda")
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.full((rows,), 256, dtype=torch.int32, device="cuda")

    # Compile before replacing Tensor.item so the assertion covers the steady
    # schedule path rather than unrelated compiler setup.
    prefill.compute_prefill_schedule(
        row_to_batch, starts, ends, 256, launch_ctas, 256
    )
    torch.cuda.synchronize()

    def fail_item(*args, **kwargs):
        raise AssertionError("compute_prefill_schedule must not read a GPU scalar")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)
    _, info, actual_launch_ctas = prefill.compute_prefill_schedule(
        row_to_batch, starts, ends, 256, launch_ctas, 256
    )
    torch.cuda.synchronize()

    assert actual_launch_ctas == launch_ctas
    assert info.shape == (launch_ctas * prefill.ROWS_PER_CTA, prefill.CTA_INFO_WIDTH)


def test_prefill_reuses_compiled_launcher_with_new_tensors(monkeypatch):
    prefill.compile_pa_mqa_logits_fp4_prefill.cache_clear()
    cases = [make_case(64, 128, 1, 64, seed=seed) for seed in (45, 46)]
    launches = []
    run_compiled = prefill._run_compiled

    def tracked_run_compiled(launcher, *args):
        compiled_before = getattr(launcher, "_cf", None)
        run_compiled(launcher, *args)
        launches.append((launcher, compiled_before, getattr(launcher, "_cf", None)))

    monkeypatch.setattr(prefill, "_run_compiled", tracked_run_compiled)
    for data, expected, storage in cases:
        q_fp4 = data["q_fp4"].squeeze(1)
        q_scale = data["q_scale"].squeeze(1)
        rows = q_fp4.shape[0]
        row_to_batch = torch.arange(rows, dtype=torch.int32, device="cuda")
        starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
        ends = data["context_lens"]
        _, info, n_ctas = prefill.compute_prefill_schedule(
            row_to_batch, starts, ends, 256, max(512, rows), data["max_seq_len"]
        )
        actual = prefill.flydsl_pa_mqa_logits_fp4_prefill(
            q_fp4,
            q_scale,
            data["kv_cache"],
            data["kv_scale"],
            data["block_tables"],
            data["weights"],
            row_to_batch,
            starts,
            ends,
            data["max_seq_len"],
            weight_scale=data["weight_scale"],
            out=data["out"],
            cta_info=info,
            n_ctas=n_ctas,
        )
        torch.testing.assert_close(actual, expected(), rtol=1e-4, atol=5e-4)
        assert torch.all(storage[:, 513:] == 71.0)

    assert launches[0][0] is launches[1][0]
    assert launches[0][1] is None
    assert launches[0][2] is not None
    assert launches[1][1] is launches[0][2]
    assert launches[1][2] is launches[0][2]


@pytest.mark.parametrize(
    ("next_n", "block_k", "num_warps"),
    [
        (1, None, None),
        (1, 64, 2),
        (2, None, None),
        (3, None, None),
        (4, None, None),
        (5, None, None),
    ],
    ids=["n1-auto", "n1-block64", "n2", "n3", "n4", "n5"],
)
def test_decode_graph_updates_context(next_n, block_k, num_warps):
    data, expected, storage = make_case(64, 128, next_n, 64)
    data.update(block_k=block_k, num_warps=num_warps)
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
