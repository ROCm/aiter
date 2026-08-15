import pytest
from aiter.ops.triton.attention.pa_mqa_logits import (
    _resolve_paged_mqa_split_kv,
    deepgemm_fp8_paged_mqa_logits,
)


@pytest.mark.parametrize("split_kv", [0, -1, True, 1.5])
def test_paged_mqa_logits_rejects_invalid_split_kv(split_kv):
    with pytest.raises(ValueError, match="SplitKV must be a positive integer"):
        _resolve_paged_mqa_split_kv(split_kv, 256, 8192, 2, False)


@pytest.mark.parametrize(
    "total_cu_count,tile_q_count,wave_per_eu,is_gfx1250,expected",
    [
        (256, 8192, 2, False, 10),
        (256, 64, 2, False, 10),
        (256, 1, 2, False, 520),
        (256, 8192, 4, True, 40),
    ],
)
def test_paged_mqa_logits_legacy_split_kv(
    total_cu_count, tile_q_count, wave_per_eu, is_gfx1250, expected
):
    assert (
        _resolve_paged_mqa_split_kv(
            None,
            total_cu_count,
            tile_q_count,
            wave_per_eu,
            is_gfx1250,
        )
        == expected
    )


@pytest.mark.parametrize("split_kv", [1, 2, 7, 16])
def test_paged_mqa_logits_explicit_split_kv(split_kv):
    assert _resolve_paged_mqa_split_kv(split_kv, 256, 8192, 2, False) == split_kv


def test_paged_mqa_logits_rejects_split_kv_with_varctx():
    with pytest.raises(ValueError, match="SplitKV cannot be used with VarCtxSchedule"):
        deepgemm_fp8_paged_mqa_logits(
            None,
            None,
            None,
            None,
            None,
            None,
            1,
            SplitKV=2,
            VarCtxSchedule=object(),
        )
