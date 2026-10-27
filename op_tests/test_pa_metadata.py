import itertools
import math
import subprocess
import sys
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.attention import _attention_ps_reduce_asm, pa_reduce_v1


def test_pa_metadata_v1_2_work_layout():
    cases = [
        ([257] * 8, 16, 1, 4),
        ([32768] * 16, 16, 1, 4),
        ([32768] * 32, 16, 1, 4),
        ([200003] + [257] * 199, 1024, 1, 4),
        ([257] * 513, 16, 8, 4),
        ([200003] + [257] * 32767, 16, 1, 4),
    ]
    cases.extend(
        ([context] * batch, 16, 1, query_length)
        for batch, context, query_length in itertools.product(
            (1, 2, 4, 8, 16, 32), (1024, 2048, 3072, 4096), (1, 2, 4)
        )
    )
    workers = torch.cuda.get_device_properties(0).multi_processor_count
    for lengths, page_size, heads, query_length in cases:
        batch = len(lengths)
        context = torch.tensor(lengths, dtype=torch.int32, device="cuda")
        qo = torch.arange(batch + 1, dtype=torch.int32, device="cuda") * query_length
        kv = torch.cat(
            (context.new_zeros(1), ((context + page_size - 1) // page_size).cumsum(0))
        ).to(torch.int32)
        layouts = []
        for fast in (True, False):
            info = aiter.get_pa_metadata_info_v1(batch, heads, fast)
            buffers = [
                torch.empty(shape, dtype=dtype, device="cuda") for shape, dtype in info
            ]
            if fast:
                buffers[3] = context.new_empty(batch + workers + 1)
                buffers[4] = context.new_empty((batch + workers, 2))
            aiter.get_pa_metadata_v1(
                qo,
                kv,
                context,
                16 if heads == 1 else 8,
                heads,
                True,
                *buffers,
                block_size=page_size,
                kv_granularity=page_size,
                max_seqlen_qo=query_length,
                uni_seqlen_qo=query_length,
                fast_mode=fast,
            )
            count = int(buffers[1][-1].item())
            layouts.append((buffers[1].cpu(), buffers[2][:count].cpu()))
        for expected, actual in zip(layouts[0], layouts[1]):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    print(f"PASS v1_2 work layout: {len(cases)} cases")


def test_pa_metadata_v1_1():
    workers = torch.cuda.get_device_properties(0).multi_processor_count
    for batch, heads, cap in ((4096, 4, 1), (32768, 1, 1), (8, workers, -1)):
        info = aiter.get_pa_metadata_info_v1(batch, heads, False, cap)
        assert info[-1][0] == 0, "unsplit work must not reserve reduction capacity"
    for heads in (1, 4, workers):
        info = aiter.get_pa_metadata_info_v1(32768, heads, False)
        assert info[-1][0] == 2 * (workers // heads - 1)
    cases = [
        ([257] * (workers + 1), [4] * (workers + 1), 1, 2, 16),
        (
            [256 * (workers // 4)] * (workers // 4 - 1),
            [4] * (workers // 4 - 1),
            4,
            2,
            16,
        ),
        ([200003] + [257] * 32767, [1] * 32768, 1, -1, 16),
        ([257] * 4097, [1] * 4097, 1, -1, 16),
        (
            [0, 16, 257, 2048] * 8191 + [0, 16, 257],
            [1, 4, 2, 3] * 8191 + [1, 4, 2],
            3,
            3,
            16,
        ),
        ([0] * 32768, [1] * 32768, 1, 1, 16),
        ([200003] + [257] * 63, [4] * 64, 8, -1, 16),
        ([200003] + [257] * 64, [4] * 65, 8, -1, 16),
        ([200003] + [257] * 199, [4] * 200, 8, -1, 16),
        ([200003] + [257] * 511, [4] * 512, 8, -1, 16),
        ([200003] + [257] * 512, [4] * 513, 8, -1, 16),
        ([200003] + [257] * 4095, [4] * 4096, 8, -1, 16),
        ([200003], [4], 8, -1, 16),
        ([200000] * 64, [4] * 64, 3, -1, 16),
        ([200003, 257, 8193, 65537] * 4, [1, 4, 2, 3] * 4, 3, -1, 16),
        ([1073741823] * 5, [16] * 5, 3, -1, 16),
        ([257] * 8, [4] * 8, 1, -1, 16),
        ([200003] + [257] * 7, [4] * 8, 1, -1, 16),
        ([200003] + [257] * 199, [4] * 200, 1, -1, 16),
        ([200000] * 200, [4] * 200, 1, -1, 16),
        ([200000] * 160, [1, 4, 2, 3] * 40, 2, 1, 16),
        ([16, 32, 1024], [1] * 3, 1, -1, 16),
        ([200003] + [257] * 7, [1, 4, 2, 3] * 2, 3, 3, 16),
        ([15, 17, 33] * 43, [1, 2, 4] * 43, 8, -1, 16),
        ([0, 16, 32], [1] * 3, 1, -1, 16),
        ([0] * 5, [1] * 5, 3, 3, 16),
        ([31, 513, 65537], [4] * 3, 8, 1, 16),
        ([257, 8193, 65537], [4, 2, 1], 3, 3, 64),
    ]
    for lengths, qo_lengths, heads, cap, granularity in cases:
        context = torch.tensor(lengths, dtype=torch.int32, device="cuda")
        qo = torch.tensor(
            [0] + list(itertools.accumulate(qo_lengths)),
            dtype=torch.int32,
            device="cuda",
        )
        kv = torch.tensor(
            [0] + list(itertools.accumulate((length + 15) // 16 for length in lengths)),
            dtype=torch.int32,
            device="cuda",
        )
        buffers, guards = [], []
        for shape, dtype in aiter.get_pa_metadata_info_v1(
            len(lengths), heads, False, cap
        ):
            size = shape if isinstance(shape, int) else math.prod(shape)
            backing = torch.full((size + 16,), 777, dtype=dtype, device="cuda")
            buffers.append(backing[:size].view(shape))
            guards.append(backing[size:])
        generate = partial(
            aiter.get_pa_metadata_v1,
            qo,
            kv,
            context,
            8,
            heads,
            True,
            *buffers,
            kv_granularity=granularity,
            block_size=16,
            max_seqlen_qo=max(qo_lengths),
            uni_seqlen_qo=qo_lengths[0] if min(qo_lengths) == max(qo_lengths) else -1,
            fast_mode=False,
            max_split_per_batch=cap,
        )
        for captured in (False, True):
            if captured:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    generate()
                context.div_(2, rounding_mode="floor")
                graph.replay()
            else:
                generate()
            torch.cuda.synchronize()
            for indptr, capacity in (
                (buffers[1], buffers[2].shape[0]),
                (buffers[3], buffers[5].numel()),
            ):
                assert indptr[0].item() == 0
                assert torch.all(torch.diff(indptr) >= 0).item()
                assert 0 <= indptr[-1].item() <= capacity
            max_parts = workers if cap < 0 else min(cap, workers)
            task_capacity = (
                min(len(lengths) * max_parts, len(lengths) + workers // heads - 1)
                * heads
            )
            assert buffers[1][-1].item() <= task_capacity
            works = buffers[2][: int(buffers[1][-1].item())]
            sequences = works[:, 0].long()
            assert torch.all((sequences >= 0) & (sequences < len(lengths))).item()
            assert torch.all(works[:, 4] >= kv[sequences]).item()
            assert torch.all(works[:, 4] < works[:, 5]).item()
            assert torch.all(
                works[:, 5] <= kv[sequences] + (context[sequences] + 15) // 16
            ).item()
            covered_pages = torch.zeros_like(context)
            covered_pages.scatter_add_(0, sequences, works[:, 5] - works[:, 4])
            torch.testing.assert_close(covered_pages, ((context + 15) // 16) * heads)
            if heads == 8 and cap < 0 and lengths[0] == 200003:
                pages = works[:, 5] - works[:, 4]
                prefix_pages = torch.cat((pages.new_zeros(1), pages.cumsum(0)))
                worker_pages = torch.diff(prefix_pages[buffers[1].long()])
                budget = math.ceil(pages.sum().item() / heads / (workers // heads))
                assert worker_pages.max().item() <= budget + 16, (
                    f"worker load {worker_pages.max().item()} exceeds page budget "
                    f"{budget} plus one tile"
                )
            work_heads = (works[:, 7] & 65535) // 8
            torch.testing.assert_close(
                torch.div(works[:, 7], 65536, rounding_mode="floor"),
                (work_heads + 1) * 8,
            )
            work_workers = torch.repeat_interleave(
                torch.arange(workers, device="cuda"), torch.diff(buffers[1]).long()
            )
            assignments = (work_workers * heads + work_heads) * len(lengths) + sequences
            assert assignments.unique().numel() == works.shape[0]
            first_head = works[work_heads == 0]
            parts = torch.bincount(first_head[:, 0].long(), minlength=len(lengths))
            reduce_counts = torch.where(parts > 1, parts, 0)
            split_sequences = parts > 1
            groups = int(split_sequences.sum().item())
            torch.testing.assert_close(
                torch.diff(buffers[3][: groups + 1]).long(),
                reduce_counts[split_sequences],
            )
            assert torch.all(buffers[3][groups:] == reduce_counts.sum()).item()
            torch.testing.assert_close(
                buffers[4][:groups],
                torch.stack((qo[:-1][split_sequences], qo[1:][split_sequences]), dim=1),
            )
            if heads == 4 and cap == 2 and not captured:
                assert buffers[3][-1].item() == buffers[5].numel()
            qo_sizes = torch.diff(qo).long()
            partial_counts = reduce_counts * qo_sizes
            partial_starts = partial_counts.cumsum(0) - partial_counts
            reduce_sequences = torch.repeat_interleave(
                torch.arange(len(lengths), device="cuda"), reduce_counts
            )
            reduce_ranks = torch.arange(
                reduce_sequences.numel(), device="cuda"
            ) - torch.repeat_interleave(
                reduce_counts.cumsum(0) - reduce_counts, reduce_counts
            )
            expected_partials = (
                partial_starts[reduce_sequences]
                + reduce_ranks * qo_sizes[reduce_sequences]
            )
            torch.testing.assert_close(
                buffers[5][: reduce_sequences.numel()].long(), expected_partials
            )
            torch.testing.assert_close(
                first_head[first_head[:, 1] >= 0, 1].long(), expected_partials
            )
            assert torch.all(
                first_head[parts[first_head[:, 0].long()] == 1, 1] == -1
            ).item()
            for head in range(heads):
                head_works = works[work_heads == head]
                torch.testing.assert_close(head_works[:, 1], first_head[:, 1])
                head_sequences = head_works[:, 0].long()
                head_pages = torch.zeros_like(context)
                head_pages.scatter_add_(
                    0, head_sequences, head_works[:, 5] - head_works[:, 4]
                )
                torch.testing.assert_close(head_pages, (context + 15) // 16)
                if head_works.shape[0]:
                    first = torch.cat(
                        (
                            head_sequences.new_ones(1, dtype=torch.bool),
                            head_sequences[1:] != head_sequences[:-1],
                        )
                    )
                    last = torch.cat((first[1:], first.new_ones(1)))
                    torch.testing.assert_close(
                        head_works[first, 4], kv[head_sequences[first]]
                    )
                    torch.testing.assert_close(
                        head_works[last, 5],
                        kv[head_sequences[last]]
                        + (context[head_sequences[last]] + 15) // 16,
                    )
                    torch.testing.assert_close(
                        head_works[~first, 4], head_works[~last, 5]
                    )
            if cap < 0:
                pages = (works[:, 5] - works[:, 4]).long()
                prefix_pages = torch.cat((pages.new_zeros(1), pages.cumsum(0)))
                worker_pages = torch.diff(prefix_pages[buffers[1].long()])
                units = (context.long() + granularity - 1) // granularity
                budget = math.ceil(units.sum().item() / (workers // heads))
                assert worker_pages.max().item() <= budget * (granularity // 16)
            if cap > 0:
                assert torch.all(
                    torch.bincount(sequences, minlength=len(lengths)) <= cap * heads
                ).item()
            partial_map = buffers[5][: int(buffers[3][-1].item())]
            assert torch.all(
                (partial_map >= 0)
                & (partial_map < buffers[5].numel() * max(qo_lengths))
            ).item()
            assert all(torch.all(guard == 777).item() for guard in guards)
    print(f"PASS page16 metadata: {len(cases)} cases and {len(cases)} graph refreshes")


def test_pa_metadata_graph_boundaries():
    page_sizes = (16, 1024) if get_gfx() == "gfx950" else (1024,)
    for batch, page_size, query_length in itertools.product(
        (512, 513), page_sizes, (1, 4)
    ):
        torch.manual_seed(42)
        query_heads = 8
        lengths = [64 * page_size + 1] + [max(257, page_size + 1)] * (batch - 1)
        context = torch.tensor(lengths, dtype=torch.int32, device="cuda")
        qo = torch.arange(batch + 1, dtype=torch.int32, device="cuda") * query_length
        pages = (context + page_size - 1) // page_size
        kv = torch.cat((pages.new_zeros(1), pages.cumsum(0, dtype=torch.int32)))
        pool = int(kv[-1].item())
        indices = torch.randperm(pool, dtype=torch.int32, device="cuda")
        query = torch.empty(
            (batch * query_length, query_heads, 128),
            dtype=torch.bfloat16,
            device="cuda",
        ).uniform_(-0.5, 0.5)
        key_tokens = (
            torch.empty((pool, page_size, 128), device="cuda")
            .uniform_(-0.5, 0.5)
            .to(aiter.dtypes.fp8)
        )
        value_tokens = (
            torch.empty_like(key_tokens, dtype=torch.float32)
            .uniform_(-0.5, 0.5)
            .to(aiter.dtypes.fp8)
        )
        key = (
            key_tokens.view(pool, page_size, 1, 8, 16)
            .permute(0, 2, 3, 1, 4)
            .contiguous()
        )
        value = (
            value_tokens.transpose(1, 2)
            .contiguous()
            .view(pool, 1, 128, page_size // 16, 16)
        )
        value = value.permute(0, 1, 3, 2, 4).contiguous()
        scale = torch.ones((pool, 1, page_size, 1), device="cuda")
        specs = list(aiter.get_pa_metadata_info_v1(batch, 1, False))
        capacity = specs[5][0]
        specs.extend(
            [
                (query.shape, query.dtype),
                ((capacity * query_length, 1, query_heads, 128), torch.float32),
                ((capacity * query_length, 1, query_heads, 1), torch.float32),
                ((batch * query_length, query_heads), torch.float32),
            ]
        )
        tensors, guards = [], []
        for shape, dtype in specs:
            size = shape if isinstance(shape, int) else math.prod(shape)
            backing = torch.full((size + 16,), 777, dtype=dtype, device="cuda")
            tensors.append(backing[:size].view(shape))
            guards.append(backing[size:])
        buffers = tensors[:6]
        output, partial_output, partial_lse, final_lse = tensors[6:]
        generate = partial(
            aiter.get_pa_metadata_v1,
            qo,
            kv,
            context,
            query_heads,
            1,
            True,
            *buffers,
            kv_granularity=max(16, page_size),
            block_size=page_size,
            max_seqlen_qo=query_length,
            uni_seqlen_qo=query_length,
            fast_mode=False,
        )
        attention = partial(
            aiter.pa_ps_fwd_asm,
            query,
            key,
            value,
            kv,
            indices,
            context,
            128**-0.5,
            max_qlen=query_length,
            K_QScale=scale,
            V_QScale=scale,
            out_=output,
            qo_indptr=qo,
            work_indptr=buffers[1],
            work_info=buffers[2],
            splitData=partial_output,
            splitLse=partial_lse,
            mask=1,
            high_precision=1,
        )
        reduce_function = _attention_ps_reduce_asm if page_size == 16 else pa_reduce_v1
        reduce = partial(
            reduce_function,
            partial_output,
            partial_lse,
            buffers[3],
            buffers[4],
            buffers[5],
            query_length,
            output,
            final_lse,
        )

        def forward(generate=generate, attention=attention, reduce=reduce):
            generate()
            attention()
            reduce()

        forward()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward()
        original_counts = None
        for shortened in (False, True):
            if shortened:
                context.fill_(max(16, page_size // 2))
            for tensor in (output, partial_output, partial_lse, final_lse):
                tensor.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            work_count = int(buffers[1][-1].item())
            reduce_count = int(buffers[3][-1].item())
            assert buffers[1][0].item() == buffers[3][0].item() == 0
            assert torch.all(torch.diff(buffers[1]) >= 0).item()
            assert torch.all(torch.diff(buffers[3]) >= 0).item()
            workers = buffers[1].numel() - 1
            assert batch <= work_count <= batch + workers - 1
            assert 0 <= reduce_count <= buffers[5].numel()
            works = buffers[2][:work_count]
            sequences = works[:, 0].long()
            assert torch.all((sequences >= 0) & (sequences < batch)).item()
            assert torch.all(torch.diff(sequences) >= 0).item()
            torch.testing.assert_close(works[:, 2], qo[sequences])
            torch.testing.assert_close(works[:, 3], qo[sequences + 1])
            assert torch.all(works[:, 4] < works[:, 5]).item()
            first = torch.cat(
                (
                    sequences.new_ones(1, dtype=torch.bool),
                    sequences[1:] != sequences[:-1],
                )
            )
            last = torch.cat((first[1:], first.new_ones(1)))
            torch.testing.assert_close(
                sequences[first], torch.arange(batch, device="cuda")
            )
            torch.testing.assert_close(works[first, 4], kv[:-1])
            torch.testing.assert_close(
                works[last, 5], kv[:-1] + (context + page_size - 1) // page_size
            )
            torch.testing.assert_close(works[~first, 4], works[~last, 5])
            parts = torch.bincount(sequences, minlength=batch)
            expected_reduce_counts = torch.where(parts > 1, parts, 0)
            split_sequences = parts > 1
            groups = int(split_sequences.sum().item())
            torch.testing.assert_close(
                torch.diff(buffers[3][: groups + 1]).long(),
                expected_reduce_counts[split_sequences],
            )
            assert torch.all(buffers[3][groups:] == reduce_count).item()
            expected_partials = torch.arange(reduce_count, device="cuda") * query_length
            torch.testing.assert_close(
                buffers[5][:reduce_count].long(), expected_partials
            )
            torch.testing.assert_close(
                works[works[:, 1] >= 0, 1].long(), expected_partials
            )
            assert torch.all(works[parts[sequences] == 1, 1] == -1).item()
            torch.testing.assert_close(
                buffers[4][:groups],
                torch.stack((qo[:-1][split_sequences], qo[1:][split_sequences]), dim=1),
            )
            if shortened:
                assert work_count < original_counts[0]
                assert reduce_count < original_counts[1]
                assert work_count == batch and reduce_count == 0
            else:
                original_counts = (work_count, reduce_count)
                assert work_count > batch and reduce_count > 0
            reference = torch.empty_like(output)
            for sequence, (length, page_begin) in enumerate(
                zip(context.cpu().tolist(), kv[:-1].cpu().tolist())
            ):
                tokens = torch.arange(length, device="cuda")
                physical_pages = indices[page_begin + tokens // page_size].long()
                key_ref = (
                    key_tokens[physical_pages, tokens % page_size]
                    .float()
                    .unsqueeze(0)
                    .expand(query_heads, -1, -1)
                )
                value_ref = (
                    value_tokens[physical_pages, tokens % page_size]
                    .float()
                    .unsqueeze(0)
                    .expand(query_heads, -1, -1)
                )
                query_ref = (
                    query[sequence * query_length : (sequence + 1) * query_length]
                    .transpose(0, 1)
                    .float()
                )
                allowed = tokens.unsqueeze(0) <= length - query_length + torch.arange(
                    query_length, device="cuda"
                ).unsqueeze(1)
                expected = torch.nn.functional.scaled_dot_product_attention(
                    query_ref,
                    key_ref,
                    value_ref,
                    attn_mask=allowed,
                )
                reference[sequence * query_length : (sequence + 1) * query_length] = (
                    expected.transpose(0, 1).to(output.dtype)
                )
            torch.testing.assert_close(output, reference, atol=0.01, rtol=0.01)
            assert all(torch.all(guard == 777).item() for guard in guards)
        print(
            f"PASS full graph boundary: batch={batch}, page={page_size}, Q={query_length}"
        )


def test_pa_metadata_no_splits():
    page_sizes = (16, 1024) if get_gfx() == "gfx950" else (1024,)
    for page_size, query_length in itertools.product(page_sizes, (1, 4)):
        heads = 4
        query_heads = heads * 16
        context = torch.tensor([17, 257, 513], dtype=torch.int32, device="cuda")
        qo = torch.arange(4, dtype=torch.int32, device="cuda") * query_length
        pages = (context + page_size - 1) // page_size
        kv = torch.cat((pages.new_zeros(1), pages.cumsum(0, dtype=torch.int32)))
        pool = int(kv[-1].item())
        indices = torch.arange(pool, dtype=torch.int32, device="cuda")
        query = torch.zeros(
            (3 * query_length, query_heads, 128), dtype=torch.bfloat16, device="cuda"
        )
        key = torch.zeros(
            (pool, heads, 8, page_size, 16), dtype=aiter.dtypes.fp8, device="cuda"
        )
        value = torch.ones(
            (pool, heads, 128, page_size), dtype=torch.bfloat16, device="cuda"
        ).to(aiter.dtypes.fp8)
        scale = torch.ones((pool, heads, page_size, 1), device="cuda")
        output = torch.full_like(query, float("nan"))
        buffers = [
            torch.empty(shape, dtype=dtype, device="cuda")
            for shape, dtype in aiter.get_pa_metadata_info_v1(3, heads, False, 1)
        ]
        assert buffers[5].numel() == 0

        def forward(
            *,
            qo=qo,
            kv=kv,
            context=context,
            heads=heads,
            buffers=buffers,
            page_size=page_size,
            query_length=query_length,
            query=query,
            key=key,
            value=value,
            output=output,
            indices=indices,
            scale=scale,
        ):
            aiter.get_pa_metadata_v1(
                qo,
                kv,
                context,
                16,
                heads,
                True,
                *buffers,
                kv_granularity=max(16, page_size),
                block_size=page_size,
                max_seqlen_qo=query_length,
                uni_seqlen_qo=query_length,
                fast_mode=False,
                max_split_per_batch=1,
            )
            partial_output, _ = aiter.pa_persistent_fwd(
                query,
                key,
                value,
                output,
                query_length,
                qo,
                kv,
                indices,
                context,
                buffers[1],
                buffers[2],
                buffers[3],
                buffers[4],
                buffers[5],
                K_QScale=scale,
                V_QScale=scale,
                mask=1,
            )
            assert partial_output.numel() == 0

        with (
            patch("aiter.ops.attention.pa_reduce_v1") as hip_reduce,
            patch("aiter.ops.attention._attention_ps_reduce_asm") as asm_reduce,
        ):
            forward()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                output, torch.ones_like(output), atol=0.01, rtol=0.01
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                forward()
            context.div_(2, rounding_mode="floor")
            output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                output, torch.ones_like(output), atol=0.01, rtol=0.01
            )
            hip_reduce.assert_not_called()
            asm_reduce.assert_not_called()
        assert torch.count_nonzero(buffers[3]).item() == 0
        works = buffers[2][: int(buffers[1][-1].item())]
        assert torch.all(works[:, 1] == -1).item()
    print(f"PASS no-split output: {2 * len(page_sizes)} cases and graph refreshes")


if __name__ == "__main__":
    test_pa_metadata_v1_2_work_layout()
    test_pa_metadata_v1_1()
    test_pa_metadata_no_splits()
    test_pa_metadata_graph_boundaries()
    if get_gfx() == "gfx950":
        with TemporaryDirectory() as output_dir:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("test_pa_ps.py")),
                    "--metadata-mode",
                    "v1_1",
                    "--block_size",
                    "16",
                    "-b",
                    "3",
                    "-n",
                    "16,2",
                    "32,2",
                    "-q",
                    "1",
                    "4",
                    "-c",
                    "4097",
                    "--varlen",
                    "--output",
                    str(Path(output_dir) / "pa.csv"),
                ],
                check=True,
            )
    else:
        print("SKIP page16 PA output checks: ASM kernels require gfx950")
