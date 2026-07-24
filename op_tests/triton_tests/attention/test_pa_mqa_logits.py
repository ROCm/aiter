import pytest

from aiter.ops.triton.attention.pa_mqa_logits import _stage1_chunkq_splitkv


@pytest.mark.parametrize("heads", [8, 16, 32])
def test_stage1_chunkq_splitkv_uses_small_head_count(heads):
    chunk_q, split_kv = _stage1_chunkq_splitkv(
        batch_size=1,
        next_n=1,
        heads=heads,
        ChunkQ=64,
        TotalCuCount=80,
        WavePerEU=2,
    )

    assert chunk_q == heads
    assert split_kv == 160


def test_stage1_chunkq_splitkv_keeps_supported_chunkq():
    chunk_q, split_kv = _stage1_chunkq_splitkv(
        batch_size=2,
        next_n=1,
        heads=128,
        ChunkQ=64,
        TotalCuCount=80,
        WavePerEU=2,
    )

    assert chunk_q == 64
    assert split_kv == 40


def test_stage1_chunkq_splitkv_rejects_non_divisible_heads():
    with pytest.raises(AssertionError, match="must be divisible"):
        _stage1_chunkq_splitkv(
            batch_size=1,
            next_n=1,
            heads=96,
            ChunkQ=64,
            TotalCuCount=80,
            WavePerEU=2,
        )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"batch_size": 0}, "batch_size must be positive"),
        ({"next_n": 0}, "next_n must be positive"),
        ({"heads": 0}, "heads must be positive"),
        ({"ChunkQ": 0}, "ChunkQ must be positive"),
        ({"TotalCuCount": 0}, "TotalCuCount must be positive"),
        ({"WavePerEU": 0}, "WavePerEU must be positive"),
    ],
)
def test_stage1_chunkq_splitkv_rejects_non_positive_inputs(kwargs, match):
    args = dict(
        batch_size=1,
        next_n=1,
        heads=64,
        ChunkQ=64,
        TotalCuCount=80,
        WavePerEU=2,
    )
    args.update(kwargs)

    with pytest.raises(AssertionError, match=match):
        _stage1_chunkq_splitkv(**args)
