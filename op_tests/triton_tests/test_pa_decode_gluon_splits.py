# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Split-count heuristic of pa_decode_gluon.

The count tests run against a mocked 256-CU device. The occupancy table test
compiles the PS decode kernels and needs a gfx950 GPU.
"""

import itertools
import re
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

import aiter.ops.triton.gluon.pa_decode_gluon as pa
from aiter import dtypes

NUM_CUS = 256
BF16 = torch.bfloat16
FP8 = dtypes.fp8


def _is_gfx950():
    return torch.cuda.is_available() and pa.get_cdna_version() == 4


@pytest.fixture
def device(request):
    cdna = request.param
    props = SimpleNamespace(multi_processor_count=NUM_CUS)
    with (
        mock.patch.object(pa.torch.cuda, "get_device_properties", return_value=props),
        mock.patch.object(pa, "get_cdna_version", return_value=cdna),
    ):
        yield cdna


@pytest.mark.parametrize("device", [3, 4], indirect=True)
@pytest.mark.parametrize(
    "num_sequences,num_kv_heads,split_kv_blocks,expected",
    [(64, 8, 1, 1), (4, 1, 1, 8), (16, 2, 1, 8), (1, 1, 4, 8), (64, 1, 4, 8)],
)
def test_without_head_size_keeps_the_original_count(
    device, num_sequences, num_kv_heads, split_kv_blocks, expected
):
    assert (
        pa.get_recommended_splits(num_sequences, num_kv_heads, split_kv_blocks)
        == expected
    )


@pytest.mark.parametrize("device", [4], indirect=True)
@pytest.mark.parametrize(
    "num_sequences,num_kv_heads,head_size,kv_cache_dtype,expected",
    [
        (1, 1, 64, BF16, 32),  # capped at 32
        (4, 8, 64, BF16, 16),  # 128 B rows, 3 workgroups per CU: 768 // 32 = 24
        (24, 1, 64, BF16, 32),
        (24, 1, 96, BF16, 16),  # padded to 128, 256 B rows, 2 per CU: 512 // 24
        (24, 2, 128, BF16, 8),
        (16, 1, 256, BF16, 16),  # 512 B rows, 1 per CU: 256 // 16
        (16, 1, 512, BF16, 16),  # rows above 512 B use the 512 B entry
        (192, 1, 256, BF16, 4),  # 1 resident, original count 3 rounded up
        (96, 2, 128, BF16, 4),  # 2 resident, original count 3 rounded up
        (64, 8, 128, BF16, 1),
        (256, 8, 64, BF16, 1),  # more heads than resident workgroups
        (4, 8, 64, FP8, 32),  # 64 B rows, 4 per CU: 1024 // 32
        (24, 2, 128, FP8, 16),  # 128 B rows: 768 // 48
        (24, 2, 256, FP8, 8),  # 256 B rows: 512 // 48 = 10
    ],
)
def test_count_fills_the_resident_workgroups(
    device, num_sequences, num_kv_heads, head_size, kv_cache_dtype, expected
):
    splits = pa.get_recommended_splits(
        num_sequences,
        num_kv_heads,
        head_size=head_size,
        kv_cache_dtype=kv_cache_dtype,
    )
    assert splits == expected


@pytest.mark.parametrize("device", [4], indirect=True)
@pytest.mark.parametrize(
    "num_sequences,num_kv_heads,max_context_length,expected",
    [
        (1, 1, 1, 1),
        (1, 1, 300, 2),
        (1, 1, 1024, 4),
        (1, 1, 1500, 4),  # 6 partitions, rounded down to a power of two
        (4, 8, 65536, 16),
    ],
)
def test_count_stops_at_the_context_partitions(
    device, num_sequences, num_kv_heads, max_context_length, expected
):
    splits = pa.get_recommended_splits(
        num_sequences,
        num_kv_heads,
        head_size=128,
        kv_cache_dtype=BF16,
        max_context_length=max_context_length,
    )
    assert splits == expected


@pytest.mark.parametrize("device", [4], indirect=True)
@pytest.mark.parametrize(
    "num_sequences,head_size,max_context_length,expected",
    [
        (2, 64, None, 32),  # 8 sequence splits of 4 blocks
        (2, 64, 2048, 8),  # 2048 tokens fill 2 KV blocks of 1024
        (64, 128, None, 8),  # 512 // 256 = 2 sequence splits
    ],
)
def test_count_scales_with_split_kv_blocks(
    device, num_sequences, head_size, max_context_length, expected
):
    splits = pa.get_recommended_splits(
        num_sequences,
        1,
        4,
        head_size=head_size,
        kv_cache_dtype=BF16,
        max_context_length=max_context_length,
    )
    assert splits == expected


@pytest.mark.parametrize("device", [4], indirect=True)
def test_count_is_valid_for_the_kernel(device):
    for bs, hkv, skb, head_size, kv_cache_dtype, ctx in itertools.product(
        (1, 3, 7, 24, 100, 256),
        (1, 2, 8),
        (1, 4),
        (64, 128, 256),
        (BF16, FP8),
        (None, 1, 300, 4097, 131072),
    ):
        splits = pa.get_recommended_splits(
            bs,
            hkv,
            skb,
            head_size=head_size,
            kv_cache_dtype=kv_cache_dtype,
            max_context_length=ctx,
        )
        sequence_splits = splits // skb
        assert splits % skb == 0 and splits <= 32
        assert sequence_splits & (sequence_splits - 1) == 0
        if ctx is None:
            assert splits >= pa.get_recommended_splits(bs, hkv, skb)


@pytest.mark.parametrize("device", [4], indirect=True)
@pytest.mark.parametrize(
    "kwargs",
    [{"head_size": 128}, {"kv_cache_dtype": BF16}],
    ids=["no_dtype", "no_head"],
)
def test_head_size_and_dtype_go_together(device, kwargs):
    with pytest.raises(ValueError, match="together"):
        pa.get_recommended_splits(4, 8, **kwargs)


@pytest.mark.parametrize("device", [3], indirect=True)
def test_head_size_is_ignored_on_cdna3(device):
    assert pa.get_recommended_splits(
        1, 1, head_size=64, kv_cache_dtype=BF16, max_context_length=32768
    ) == pa.get_recommended_splits(1, 1)


class _LaunchRecorder:
    """Stands in for a Gluon kernel and keeps the compiled kernel of each launch."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.compiled = []

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(*args, **kwargs):
            compiled = launch(*args, **kwargs)
            self.compiled.append(compiled)
            return compiled

        return run


def _run_ps_decode(num_query_heads, num_kv_heads, head_size, kv_cache_dtype):
    batch_size, block_size, blocks_per_seq, splits = 2, 64, 8, 2
    elements_per_16b = 16 // kv_cache_dtype.itemsize
    num_blocks = batch_size * blocks_per_seq
    key_cache = torch.randn(
        num_blocks,
        num_kv_heads,
        head_size // elements_per_16b,
        block_size,
        elements_per_16b,
        device="cuda",
    ).to(kv_cache_dtype)
    value_cache = torch.randn(
        num_blocks,
        num_kv_heads,
        block_size // elements_per_16b,
        head_size,
        elements_per_16b,
        device="cuda",
    ).to(kv_cache_dtype)
    query = torch.randn(
        batch_size, num_query_heads, head_size, dtype=BF16, device="cuda"
    )
    scale = torch.ones(1, device="cuda") if kv_cache_dtype.itemsize == 1 else None
    pa.pa_decode_gluon(
        output=torch.empty_like(query),
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        context_lengths=torch.full(
            (batch_size,), blocks_per_seq * block_size, dtype=torch.int32, device="cuda"
        ),
        block_tables=torch.arange(num_blocks, dtype=torch.int32, device="cuda").view(
            batch_size, blocks_per_seq
        ),
        softmax_scale=head_size**-0.5,
        query_length=1,
        max_context_partition_num=splits,
        key_scale=scale,
        value_scale=scale,
        ps=True,
    )


@pytest.mark.skipif(not _is_gfx950(), reason="occupancy table is for gfx950")
@pytest.mark.parametrize(
    "head_size,kv_cache_dtype",
    [(64, BF16), (128, BF16), (256, BF16), (64, FP8), (128, FP8), (256, FP8)],
)
def test_occupancy_table_matches_the_compiled_kernels(
    monkeypatch, head_size, kv_cache_dtype
):
    recorders = []
    for name in (
        "paged_attention_decode_sliding_window",
        "paged_attention_decode_sliding_window_head_1",
    ):
        recorder = _LaunchRecorder(getattr(pa, name))
        monkeypatch.setattr(pa, name, recorder)
        recorders.append(recorder)
    # One KV head launches the head_1 kernel, several the general one.
    for num_query_heads, num_kv_heads in ((16, 1), (64, 8)):
        _run_ps_decode(num_query_heads, num_kv_heads, head_size, kv_cache_dtype)
    compiled = [kernel for recorder in recorders for kernel in recorder.compiled]
    assert len(compiled) == 2
    occupancy = min(
        int(re.search(r"; Occupancy: (\d+)", kernel.asm["amdgcn"]).group(1))
        for kernel in compiled
    )
    row_bytes = head_size * kv_cache_dtype.itemsize
    assert pa._CDNA4_PS_WORKGROUPS_PER_CU[row_bytes] == occupancy
