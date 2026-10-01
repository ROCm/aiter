from types import SimpleNamespace

import torch

from aiter.ops.triton.utils.unified_attention_utils import (
    get_unified_attention_config,
)


def _mimo_decode_params(*, sliding_window: int):
    return SimpleNamespace(
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
