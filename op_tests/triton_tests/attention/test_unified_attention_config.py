from types import SimpleNamespace

import torch

from aiter.ops.triton.utils.types import e4m3_dtype
from aiter.ops.triton.utils.unified_attention_utils import (
    get_unified_attention_config,
)


def _mimo_decode_params(*, sliding_window: int, **kwargs):
    params = dict(
        head_size=192,
        max_seqlen_q=1,
        max_seqlen_k=60000,
        sliding_window=sliding_window,
        shuffled_kv_cache=False,
        q_dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        block_size=1,
        num_sms=304,
        num_2d_prgms=16,
        num_queries_per_kv=16,
    )
    params.update(kwargs)
    return SimpleNamespace(**params)


def test_gfx950_mimo_decode_configs():
    sliding = _mimo_decode_params(sliding_window=128)
    full = _mimo_decode_params(sliding_window=0)

    assert get_unified_attention_config(
        "attn_2d", sliding, backend="triton", arch="gfx950"
    ) == {
        "BLOCK_M": 16,
        "num_warps": 4,
        "num_stages": 2,
        "waves_per_eu": 1,
        "TILE_SIZE": 32,
    }
    assert get_unified_attention_config(
        "attn_3d", full, backend="triton", arch="gfx950"
    ) == {
        "BLOCK_M": 16,
        "num_warps": 2,
        "num_stages": 1,
        "waves_per_eu": 2,
    }
    assert get_unified_attention_config(
        "kv_split", full, backend="triton", arch="gfx950"
    ) == {
        "NUM_SEGMENTS": 64,
        "TILE_SIZE": 64,
    }


def test_gfx950_mimo_verify_and_prefill_configs():
    def config(op, **kwargs):
        params = _mimo_decode_params(num_sms=256, **kwargs)
        return get_unified_attention_config(op, params, backend="triton", arch="gfx950")

    # DFLASH verify, 8 query tokens per sequence: 15 sequences at 60k context and
    # 4 at 128k (num_2d_prgms = tokens + sequences). The split follows the batch.
    verify_60k = dict(sliding_window=0, max_seqlen_q=8, num_2d_prgms=135)
    verify_128k = dict(
        sliding_window=0, max_seqlen_q=8, max_seqlen_k=131072, num_2d_prgms=36
    )
    for params in (verify_60k, verify_128k):
        assert config("attn_3d", **params) == {
            "BLOCK_M": 64,
            "num_warps": 2,
            "num_stages": 1,
            "waves_per_eu": 2,
        }
    assert config("kv_split", **verify_60k) == {"NUM_SEGMENTS": 32, "TILE_SIZE": 32}
    assert config("kv_split", **verify_128k) == {"NUM_SEGMENTS": 64, "TILE_SIZE": 32}

    # a 32k-token prefill chunk
    assert config("attn_2d", sliding_window=0, max_seqlen_q=32768) == {
        "BLOCK_M": 256,
        "num_warps": 8,
        "num_stages": 2,
        "waves_per_eu": 1,
        "SPLIT_UNMASKED_LOOP": True,
        "TILE_SIZE": 64,
    }

    # unchanged: decode, sliding-window prefill, and head sizes other than 192
    # (these configs are tuned for the 128 + 64 split, which needs exactly 192)
    assert config("attn_3d", sliding_window=0) == {
        "BLOCK_M": 16,
        "num_warps": 2,
        "num_stages": 1,
        "waves_per_eu": 2,
    }
    prefill_2d_default = {
        "BLOCK_M": 128,
        "num_warps": 4,
        "num_stages": 1,
        "waves_per_eu": 2,
        "TILE_SIZE": 64,
    }
    assert config("attn_2d", sliding_window=128, max_seqlen_q=32768) == (
        prefill_2d_default
    )
    # the prefill config is tuned for the split QK dot, which needs an unshuffled
    # BF16/FP16 cache; shuffled and FP8 caches keep the padded dot and old config
    assert (
        config(
            "attn_2d",
            sliding_window=0,
            max_seqlen_q=32768,
            shuffled_kv_cache=True,
            block_size=64,
        )
        == prefill_2d_default
    )
    assert (
        config(
            "attn_2d",
            sliding_window=0,
            max_seqlen_q=32768,
            q_dtype=e4m3_dtype,
            kv_cache_dtype=e4m3_dtype,
        )
        == prefill_2d_default
    )
    for head_size in (128, 160):
        assert (
            config("attn_2d", sliding_window=0, max_seqlen_q=32768, head_size=head_size)
            == prefill_2d_default
        )
        assert config(
            "attn_3d", sliding_window=0, max_seqlen_q=8, head_size=head_size
        ) == {
            "BLOCK_M": 16,
            "num_warps": 2,
            "num_stages": 1,
            "waves_per_eu": 2,
        }
