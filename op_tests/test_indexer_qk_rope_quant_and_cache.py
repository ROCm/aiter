# SPDX-License-Identifier: MIT

import torch

from aiter import dtypes, indexer_qk_rope_quant_and_cache


def _run(
    q,
    weights,
    k,
    slot_mapping,
    positions,
    cos_cache,
    sin_cache,
    norm_weight,
    norm_bias,
    compute_all_q_rope,
):
    q_out = torch.zeros_like(q, dtype=dtypes.fp8)
    weights_out = torch.zeros_like(weights, dtype=torch.float32)
    kv_cache = torch.zeros((1, 16, 132), device=q.device, dtype=dtypes.fp8)
    extra = (
        {} if compute_all_q_rope is None else {"compute_all_q_rope": compute_all_q_rope}
    )
    indexer_qk_rope_quant_and_cache(
        q,
        q_out,
        weights,
        weights_out,
        k,
        kv_cache,
        slot_mapping,
        norm_weight,
        norm_bias,
        positions,
        cos_cache,
        sin_cache,
        1e-6,
        128,
        "ue8m0",
        128**-0.5 * 32**-0.5,
        preshuffle=False,
        is_neox=True,
        **extra,
    )
    torch.cuda.synchronize()
    return q_out, weights_out, kv_cache


def test_indexer_qk_compute_all_q_rope():
    torch.manual_seed(1)
    device = "cuda"
    num_tokens, num_heads, head_dim = 8, 32, 128
    q = torch.randn(
        num_tokens, num_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    weights = torch.randn(num_tokens, num_heads, device=device, dtype=torch.bfloat16)
    k = torch.randn(num_tokens, head_dim, device=device, dtype=torch.bfloat16)
    norm_weight = torch.ones(head_dim, device=device, dtype=torch.float32)
    norm_bias = torch.zeros(head_dim, device=device, dtype=torch.float32)
    angles = torch.randn(16, 32, device=device)
    cos_cache = angles.cos().bfloat16()
    sin_cache = angles.sin().bfloat16()

    clamped_positions = torch.tensor(
        [0, 1, 2, 3, 0, 15, 15, 15], device=device, dtype=torch.int64
    )
    valid_slots = torch.arange(num_tokens, device=device, dtype=torch.int64)
    q_ref, weights_ref, _ = _run(
        q,
        weights,
        k,
        valid_slots,
        clamped_positions,
        cos_cache,
        sin_cache,
        norm_weight,
        norm_bias,
        False,
    )
    q_default, weights_default, cache_default = _run(
        q,
        weights,
        k,
        valid_slots,
        clamped_positions,
        cos_cache,
        sin_cache,
        norm_weight,
        norm_bias,
        None,
    )
    q_explicit, weights_explicit, cache_explicit = _run(
        q,
        weights,
        k,
        valid_slots,
        clamped_positions,
        cos_cache,
        sin_cache,
        norm_weight,
        norm_bias,
        False,
    )
    torch.testing.assert_close(q_default.float(), q_explicit.float(), rtol=0, atol=0)
    torch.testing.assert_close(weights_default, weights_explicit, rtol=0, atol=0)
    torch.testing.assert_close(
        cache_default.float(), cache_explicit.float(), rtol=0, atol=0
    )

    mixed_slots = valid_slots.clone()
    mixed_slots[4:] = -1
    stale_positions = torch.tensor(
        [0, 1, 2, 3, -7, 16, 100, 1000], device=device, dtype=torch.int64
    )
    q_all, weights_all, kv_cache = _run(
        q,
        weights,
        k,
        mixed_slots,
        stale_positions,
        cos_cache,
        sin_cache,
        norm_weight,
        norm_bias,
        True,
    )

    torch.testing.assert_close(q_all.float(), q_ref.float(), rtol=0, atol=0)
    torch.testing.assert_close(weights_all, weights_ref, rtol=0, atol=0)

    cache_flat = kv_cache.view(-1)
    k_region = cache_flat[: 16 * head_dim]
    scale_region = cache_flat[16 * head_dim :]
    assert torch.count_nonzero(k_region[4 * head_dim :]).item() == 0
    assert torch.count_nonzero(scale_region[4 * 4 :]).item() == 0

    q_default, weights_default, _ = _run(
        q,
        weights,
        k,
        mixed_slots,
        stale_positions,
        cos_cache,
        sin_cache,
        norm_weight,
        norm_bias,
        False,
    )
    torch.testing.assert_close(q_default[:4].float(), q_ref[:4].float(), rtol=0, atol=0)
    torch.testing.assert_close(weights_default[:4], weights_ref[:4], rtol=0, atol=0)
    assert torch.count_nonzero(q_default[4:].float()).item() == 0
    assert torch.count_nonzero(weights_default[4:]).item() == 0


if __name__ == "__main__":
    test_indexer_qk_compute_all_q_rope()
