import math
from itertools import pairwise

import pytest
import torch

from aiter import dtypes
from aiter.ops.attention import (
    get_pa_metadata_info_v1,
    get_pa_metadata_v1,
    pa_persistent_fwd,
)
from aiter.ops.enum import QuantType
from aiter.ops.triton.attention.pa_ps_metadata import plan_pa_ps_metadata
from aiter.test_common import assertAllclose
from aiter.test_mha_common import attention_ref


@pytest.fixture(autouse=True)
def require_gfx950():
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    if not torch.cuda.get_device_properties().gcnArchName.startswith("gfx950"):
        pytest.skip("PA_PS planner configuration requires gfx950")


def _inputs(lengths, query_lengths, block_size):
    context = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    qo_indptr = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=context.device)
    kv_indptr = torch.zeros_like(qo_indptr)
    qo_indptr[1:] = torch.tensor(
        query_lengths, dtype=torch.int32, device=context.device
    ).cumsum(0)
    kv_indptr[1:] = ((context + block_size - 1) // block_size).cumsum(0)
    return qo_indptr, kv_indptr, context


def _check_metadata(plan, inputs, lengths, query_lengths):
    qo_indptr, kv_indptr, _ = inputs
    query_offsets = qo_indptr.cpu().tolist()
    page_offsets = kv_indptr.cpu().tolist()
    work_ptr = plan.work_indptr.cpu().tolist()
    work = plan.work_info[: work_ptr[-1]].cpu().tolist()
    reduce_ptr = plan.reduce_indptr.cpu().tolist()
    reduce_final = plan.reduce_final_map.cpu().tolist()
    reduce_partial = plan.reduce_partial_map[: reduce_ptr[-1]].cpu().tolist()
    num_cu = len(work_ptr) - 1
    groups = num_cu // plan.num_heads_k
    assert work_ptr[0] == reduce_ptr[0] == 0
    assert work_ptr == sorted(work_ptr)
    assert reduce_ptr == sorted(reduce_ptr)
    assert work_ptr[-1] <= plan.work_info.shape[0]
    assert reduce_ptr[-1] <= plan.reduce_partial_map.numel()
    assert plan.work_metadata_ptrs.cpu().tolist() == [
        plan.work_indptr.data_ptr(),
        plan.work_info.data_ptr(),
    ]
    total_tiles = sum(math.ceil(length / 256) for length in lengths)
    total_cost = total_tiles + len(lengths) * plan.work_overhead
    expected_counts = [
        max(
            1,
            min(
                math.ceil(length / 256),
                plan.max_partitions,
                math.ceil(math.ceil(length / 256) * groups / total_cost),
            ),
        )
        for length in lengths
    ]
    assert sum(count - 1 for count in expected_counts) <= groups - 1
    split_queries = (
        plan.max_qlen > 1
        and len(lengths) <= plan.scan_block_size
        and sum(
            count * queries for count, queries in zip(expected_counts, query_lengths)
        )
        <= groups
    )
    by_sequence = [[[] for _ in lengths] for _ in range(plan.num_heads_k)]
    for group in range(num_cu):
        head = group // groups
        for record in work[work_ptr[group] : work_ptr[group + 1]]:
            (
                sequence,
                partial,
                query_start,
                query_end,
                begin,
                end,
                offset,
                head_range,
            ) = record
            assert (
                query_offsets[sequence]
                <= query_start
                < query_end
                <= query_offsets[sequence + 1]
            )
            assert query_end - query_start == (
                1 if split_queries else query_lengths[sequence]
            )
            assert offset == 0
            assert head_range == (((head + 1) * plan.num_heads_per_head_k) << 16) | (
                head * plan.num_heads_per_head_k
            )
            assert page_offsets[sequence] <= begin <= end <= page_offsets[sequence + 1]
            assert (
                partial == -1
                or 0 <= partial < plan.reduce_partial_map.numel() * plan.max_qlen
            )
            by_sequence[head][sequence].append(record)
    for sequence, expected_count in enumerate(expected_counts):
        first_head = by_sequence[0][sequence]
        assert len(first_head) == expected_count * (
            query_lengths[sequence] if split_queries else 1
        )
        partials = reduce_partial[reduce_ptr[sequence] : reduce_ptr[sequence + 1]]
        for query in range(query_offsets[sequence], query_offsets[sequence + 1]):
            query_work = sorted(
                (record for record in first_head if record[2] <= query < record[3]),
                key=lambda record: record[4],
            )
            assert len(query_work) == expected_count
            expected_start = page_offsets[sequence]
            for part, record in enumerate(query_work):
                assert record[4] == expected_start
                expected_start = record[5]
                if part:
                    assert (
                        record[4] - page_offsets[sequence]
                    ) * plan.block_size % 256 == 0
                if expected_count == 1:
                    assert record[1] == -1
                else:
                    assert (
                        record[1] + query - record[2]
                        == partials[part] + query - query_offsets[sequence]
                    )
            assert expected_start == page_offsets[sequence + 1]
        if expected_count == 1:
            assert reduce_ptr[sequence] == reduce_ptr[sequence + 1]
        else:
            assert len(set(partials)) == expected_count
            assert reduce_final[sequence] == [
                query_offsets[sequence],
                query_offsets[sequence + 1],
            ]
        for head in range(1, plan.num_heads_k):
            assert [record[:7] for record in by_sequence[head][sequence]] == [
                record[:7] for record in first_head
            ]


@pytest.mark.parametrize("batch", [1, 8, 257, 32768])
@pytest.mark.parametrize("num_heads_k", [1, 2])
def test_pa_ps_metadata_coverage(batch, num_heads_k):
    lengths = [200003] + [257 + sequence % 1024 for sequence in range(batch - 1)]
    query_lengths = [1 + sequence % 4 for sequence in range(batch)]
    inputs = _inputs(lengths, query_lengths, 16)
    plan = plan_pa_ps_metadata(*inputs, 16, num_heads_k, max_qlen=4)
    _check_metadata(plan, inputs, lengths, query_lengths)


@pytest.mark.parametrize("batch", [8, 1025])
def test_pa_ps_metadata_graph_refresh(batch):
    block_size = 16
    lengths = [200003] + [257] * (batch - 1)
    query_lengths = [1 + sequence % 4 for sequence in range(batch)]
    inputs = _inputs(lengths, query_lengths, block_size)
    plan = plan_pa_ps_metadata(
        *inputs, 16, 2, max_qlen=4, block_size=block_size, max_partitions=7
    )
    pointers = {
        name: tensor.data_ptr()
        for name, tensor in vars(plan).items()
        if isinstance(tensor, torch.Tensor)
    }
    stream = torch.cuda.Stream(device=inputs[2].device)
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        plan_pa_ps_metadata(
            *inputs,
            16,
            2,
            max_qlen=4,
            block_size=block_size,
            max_partitions=7,
            plan=plan,
        )
    torch.cuda.current_stream().wait_stream(stream)
    for refreshed_lengths in (
        [4 + sequence % 16 for sequence in range(batch)],
        [33] * (batch - 1) + [131077],
        lengths,
    ):
        refreshed_queries = list(reversed(query_lengths))
        refreshed = _inputs(refreshed_lengths, refreshed_queries, block_size)
        for target, source in zip(inputs, refreshed):
            target.copy_(source)
        graph.replay()
        _check_metadata(plan, inputs, refreshed_lengths, refreshed_queries)
        assert all(
            getattr(plan, name).data_ptr() == pointer
            for name, pointer in pointers.items()
        )


@pytest.mark.parametrize("max_partitions", [1, 3, 7, 255, 256])
@pytest.mark.parametrize("batch", [1, 255, 256, 65536])
def test_pa_ps_metadata_capacity(batch, max_partitions):
    lengths = [131077] + [1 + sequence % 513 for sequence in range(batch - 1)]
    inputs = _inputs(lengths, [1] * batch, 16)
    plan = plan_pa_ps_metadata(
        *inputs,
        8,
        1,
        max_qlen=1,
        block_size=16,
        max_partitions=max_partitions,
    )
    _check_metadata(plan, inputs, lengths, [1] * batch)


def test_pa_ps_metadata_one_group_per_head():
    inputs = _inputs([16, 257, 8193], [1, 2, 4], 16)
    num_heads_k = torch.cuda.get_device_properties(
        inputs[2].device
    ).multi_processor_count
    plan = plan_pa_ps_metadata(*inputs, 8, num_heads_k, max_qlen=4)
    _check_metadata(plan, inputs, [16, 257, 8193], [1, 2, 4])


def test_pa_ps_metadata_query_parallelism():
    inputs = _inputs([1921], [4], 16)
    original_inputs = [tensor.clone() for tensor in inputs]
    plan = plan_pa_ps_metadata(*inputs, 16, 1, max_qlen=4, work_overhead=8)
    work_ptr = plan.work_indptr.cpu().tolist()
    work = plan.work_info[: work_ptr[-1]].cpu().tolist()
    assert len(work) == 32
    active_groups = sum(end > begin for begin, end in pairwise(work_ptr))
    assert active_groups == 32
    expected = [
        [
            0,
            part * 4 + query,
            query,
            query + 1,
            part * 16,
            min((part + 1) * 16, 121),
            0,
            16 << 16,
        ]
        for part in range(8)
        for query in range(4)
    ]
    assert sorted(work) == sorted(expected)
    assert plan.reduce_indptr.cpu().tolist() == [0, 8]
    assert plan.reduce_final_map.cpu().tolist() == [[0, 4]]
    assert plan.reduce_partial_map[:8].cpu().tolist() == list(range(0, 32, 4))
    for original, current in zip(original_inputs, inputs):
        torch.testing.assert_close(current, original, atol=0, rtol=0)


@pytest.mark.parametrize("num_heads_k", [1, 2])
@pytest.mark.parametrize("batch", [1, 4, 8, 9, 16, 32, 64])
def test_pa_ps_metadata_query_parallelism_graph(batch, num_heads_k):
    lengths = [1921 + sequence % 128 for sequence in range(batch)]
    queries = [4] * batch
    inputs = _inputs(lengths, queries, 16)
    plan = plan_pa_ps_metadata(*inputs, 16, num_heads_k, max_qlen=4, work_overhead=8)
    pointers = {
        name: tensor.data_ptr()
        for name, tensor in vars(plan).items()
        if isinstance(tensor, torch.Tensor)
    }
    _check_metadata(plan, inputs, lengths, queries)
    stream = torch.cuda.Stream(device=inputs[2].device)
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        plan_pa_ps_metadata(
            *inputs, 16, num_heads_k, max_qlen=4, work_overhead=8, plan=plan
        )
    torch.cuda.current_stream().wait_stream(stream)
    for new_lengths, new_queries in (
        ([131077] * batch, queries),
        (lengths, [1 + sequence % 4 for sequence in range(batch)]),
        ([4] * batch, queries),
        (lengths, queries),
    ):
        incoming = _inputs(new_lengths, new_queries, 16)
        for target, source in zip(inputs, incoming):
            target.copy_(source)
        graph.replay()
        _check_metadata(plan, inputs, new_lengths, new_queries)
        assert all(
            getattr(plan, name).data_ptr() == pointer
            for name, pointer in pointers.items()
        )
    graph.reset()


def test_pa_ps_metadata_policy_validation():
    inputs = _inputs([257, 513], [1, 2], 16)
    plan = plan_pa_ps_metadata(*inputs, 16, 2, max_qlen=2)
    with pytest.raises(ValueError, match="page16"):
        plan_pa_ps_metadata(*inputs, 16, 2, max_qlen=2, block_size=128)
    with pytest.raises(ValueError, match="geometry or policy"):
        plan_pa_ps_metadata(*inputs, 16, 2, max_qlen=4, plan=plan)
    with pytest.raises(ValueError, match="geometry or policy"):
        plan_pa_ps_metadata(*inputs, 16, 2, max_qlen=2, max_partitions=7, plan=plan)
    with pytest.raises(ValueError, match="work_overhead"):
        plan_pa_ps_metadata(*inputs, 16, 2, max_qlen=2, work_overhead=0)
    with pytest.raises(ValueError, match="contiguous int32"):
        plan_pa_ps_metadata(
            inputs[0], inputs[1], inputs[2].to(torch.int64), 16, 2, max_qlen=2
        )


@pytest.mark.parametrize("query_length", [9, 17])
@pytest.mark.parametrize("uniform_query", [False, True])
@pytest.mark.parametrize("is_causal", [False, True])
def test_pa_metadata_query_tile_buffers(query_length, uniform_query, is_causal):
    lengths = [32, 1040, 32]
    query_lengths = (
        [query_length] * 3
        if uniform_query
        else [query_length - 1, query_length, query_length - 2]
    )
    inputs = _inputs(lengths, query_lengths, 16)
    num_cu = torch.cuda.get_device_properties(inputs[2].device).multi_processor_count
    if num_cu < 69:
        pytest.skip("requires at least one workgroup per page")
    shapes = get_pa_metadata_info_v1(
        3, 1, max_seqlen_qo=query_length, num_heads_per_head_k=16
    )
    metadata, storage = [], []
    sentinel = 0x12345678
    for shape, dtype in shapes:
        dimensions = (shape,) if isinstance(shape, int) else shape
        buffer = torch.full(
            (math.prod(dimensions) + 128,),
            sentinel,
            dtype=dtype,
            device=inputs[2].device,
        )
        metadata.append(buffer[64:-64].view(dimensions))
        storage.append(buffer)
    get_pa_metadata_v1(
        *inputs,
        16,
        1,
        is_causal,
        *metadata,
        kv_granularity=16,
        block_size=16,
        max_seqlen_qo=query_length,
        uni_seqlen_qo=query_length if uniform_query else -1,
    )
    torch.cuda.synchronize()
    for buffer in storage:
        assert (buffer[:64] == sentinel).all()
        assert (buffer[-64:] == sentinel).all()
    _, work_indptr, work_info, reduce_indptr, final_map, partial_map = metadata
    offsets = work_indptr.cpu().tolist()
    assert offsets == sorted(offsets)
    assert offsets[-1] <= work_info.shape[0]
    records = work_info[: offsets[-1]].cpu().tolist()
    query_offsets = inputs[0].cpu().tolist()
    expected_indptr, expected_final, expected_partial = [0], [], []
    for sequence, length in enumerate(query_lengths):
        query_tiles = math.ceil(length * 16 / 128)
        tile_size = math.ceil(length / query_tiles)
        for tile in range(query_tiles):
            begin = query_offsets[sequence] + tile * tile_size
            end = min(begin + tile_size, query_offsets[sequence + 1])
            tile_work = [
                record
                for record in records
                if record[0] == sequence and record[2:4] == [begin, end]
            ]
            assert len(tile_work) == math.ceil(lengths[sequence] / 16)
            assert all(record[1] >= 0 for record in tile_work)
            expected_partial.extend(record[1] for record in tile_work)
            expected_indptr.append(len(expected_partial))
            expected_final.append([begin, end])
    assert reduce_indptr[: len(expected_indptr)].cpu().tolist() == expected_indptr
    assert (reduce_indptr[len(expected_indptr) :] == expected_indptr[-1]).all()
    assert final_map[: len(expected_final)].cpu().tolist() == expected_final
    assert partial_map[: len(expected_partial)].cpu().tolist() == expected_partial


def test_pa_metadata_rejects_undersized_query_tiles():
    inputs = _inputs([32, 1040, 32], [17, 17, 17], 16)
    metadata = [
        torch.empty(shape, dtype=dtype, device=inputs[2].device)
        for shape, dtype in get_pa_metadata_info_v1(3, 1)
    ]
    with pytest.raises(ValueError, match="metadata buffers are too small"):
        get_pa_metadata_v1(
            *inputs,
            16,
            1,
            True,
            *metadata,
            kv_granularity=16,
            block_size=16,
            max_seqlen_qo=17,
            uni_seqlen_qo=17,
        )


@pytest.mark.parametrize("query_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("gqa", [8, 16])
@pytest.mark.parametrize("kv_kind", ["noquant", "fp8", "int8"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("max_partitions", [1, 7])
def test_pa_ps_output_lse_graph(query_dtype, gqa, kv_kind, causal, max_partitions):
    if kv_kind == "noquant" and gqa == 16 and not causal:
        pytest.skip("GQA16 noquant has only causal code objects")
    torch.manual_seed(5692)
    device = torch.device("cuda", torch.cuda.current_device())
    kv_heads, dimension, max_pages = 2, 128, 66
    kv_dtype = {"noquant": query_dtype, "fp8": dtypes.fp8, "int8": torch.int8}[kv_kind]
    width = 16 // torch.empty((), dtype=kv_dtype).element_size()
    key = (
        torch.empty(
            (max_pages, kv_heads, dimension // width, 16, width),
            dtype=query_dtype,
            device=device,
        )
        .uniform_(-2, 2)
        .to(kv_dtype)
    )
    value = (
        torch.empty(
            (max_pages, kv_heads, 16 // width, dimension, width),
            dtype=query_dtype,
            device=device,
        )
        .uniform_(-2, 2)
        .to(kv_dtype)
    )
    key_scale = value_scale = None
    if kv_kind != "noquant":
        key_scale = torch.empty((max_pages, kv_heads, 16), device=device).uniform_(
            0.2, 0.5
        )
        value_scale = torch.empty_like(key_scale).uniform_(0.2, 0.5)
    query = torch.empty(
        (7, kv_heads * gqa, dimension), dtype=query_dtype, device=device
    ).uniform_(-1, 1)
    output = torch.empty_like(query)
    pages = torch.randperm(max_pages, dtype=torch.int32, device=device)
    inputs = _inputs([257, 17], [4, 3], 16)
    plan = plan_pa_ps_metadata(
        *inputs, gqa, kv_heads, max_qlen=4, max_partitions=max_partitions
    )

    def launch():
        plan_pa_ps_metadata(
            *inputs, gqa, kv_heads, max_qlen=4, max_partitions=max_partitions, plan=plan
        )
        return pa_persistent_fwd(
            query,
            key,
            value,
            output,
            4,
            inputs[0],
            inputs[1],
            pages,
            inputs[2],
            plan.work_indptr,
            plan.work_info,
            plan.reduce_indptr,
            plan.reduce_final_map,
            plan.reduce_partial_map,
            key_scale,
            value_scale,
            mask=int(causal),
            quant_type=QuantType.No if kv_kind == "noquant" else QuantType.per_Token,
        )[1]

    def check(final_lse):
        query_offsets = inputs[0].cpu().tolist()
        page_offsets = inputs[1].cpu().tolist()
        lengths = inputs[2].cpu().tolist()
        reduce_offsets = plan.reduce_indptr.cpu().tolist()
        for sequence, length in enumerate(lengths):
            page_ids = pages[page_offsets[sequence] : page_offsets[sequence + 1]].long()
            dense_key = (
                key[page_ids]
                .permute(0, 3, 1, 2, 4)
                .reshape(-1, kv_heads, dimension)
                .float()
            )
            dense_value = (
                value[page_ids]
                .permute(0, 2, 4, 1, 3)
                .reshape(-1, kv_heads, dimension)
                .float()
            )
            if key_scale is not None:
                dense_key *= (
                    key_scale[page_ids].permute(0, 2, 1).reshape(-1, kv_heads, 1)
                )
                dense_value *= (
                    value_scale[page_ids].permute(0, 2, 1).reshape(-1, kv_heads, 1)
                )
            begin, end = query_offsets[sequence : sequence + 2]
            expected, _, expected_lse = attention_ref(
                query[begin:end].unsqueeze(0).float(),
                dense_key[:length].unsqueeze(0),
                dense_value[:length].unsqueeze(0),
                causal=causal,
            )
            actual = output[begin:end].float()
            assert torch.isfinite(actual).all()
            assertAllclose(expected[0], actual)
            head_error = (actual - expected[0]).square().mean(dim=(0, 2)).sqrt()
            head_scale = expected[0].square().mean(dim=(0, 2)).sqrt().clamp_min(1e-8)
            assert (head_error / head_scale < 0.06).all()
            if reduce_offsets[sequence] == reduce_offsets[sequence + 1]:
                assert torch.isnan(final_lse[begin:end]).all()
            else:
                torch.testing.assert_close(
                    final_lse[begin:end], expected_lse[0].T, atol=0.03, rtol=0.005
                )

    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        final_lse = launch()
        stream.synchronize()
        check(final_lse)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            final_lse = launch()
        try:
            for lengths, queries in (
                ([17, 257], [3, 4]),
                ([17, 17], [4, 3]),
                ([513, 33], [3, 4]),
                ([257, 17], [4, 3]),
            ):
                incoming = _inputs(lengths, queries, 16)
                for target, source in zip(inputs, incoming):
                    target.copy_(source)
                output.fill_(torch.nan)
                graph.replay()
                stream.synchronize()
                check(final_lse)
        finally:
            graph.reset()
    torch.cuda.current_stream().wait_stream(stream)
