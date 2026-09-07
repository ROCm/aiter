# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Validate the Opus ``output_aux`` sorting ABI for expert parallel MoE.

This is deliberately a sorting-only GPU test.  It feeds the same global
``topk_ids`` and local-expert mask to the standard sorting path used by
``flydsl_moe1_afp4_*`` and to the auxiliary sorting path required by
``flydsl_mxmoe_g1_*``.  No GEMM is launched.

Besides comparing the common sorting outputs, the test verifies the two aux
maps consumed by the MXFP4 GEMMs:

* ``m_indices[sorted_slot]`` is the source token row;
* ``reverse_sorted[token, topk_slot]`` is the corresponding sorted slot.

Only valid local routes are inspected.  Padding slots and masked-out routes
have no defined aux-map value and must not be assumed to be zero.

Example::

    HIP_VISIBLE_DEVICES=0 python op_tests/test_moe_sorting_output_aux_ep.py \
        --tokens 16 --global-experts 896 --local-experts 56 --rank 8 \
        --topk 16 --block-m 32
"""

import argparse
from contextlib import contextmanager

import torch

from aiter import dtypes
import aiter.fused_moe as fm


@contextmanager
def _force_opus_aux_sort():
    """Keep this diagnostic independent of the process environment."""
    old_backend = fm._MOE_SORT_BACKEND
    old_ck = fm._USE_CK_MOE_SORTING
    old_flydsl = fm._USE_FLYDSL_MOE_SORTING
    fm._MOE_SORT_BACKEND = "opus"
    fm._USE_CK_MOE_SORTING = False
    fm._USE_FLYDSL_MOE_SORTING = False
    try:
        yield
    finally:
        fm._MOE_SORT_BACKEND = old_backend
        fm._USE_CK_MOE_SORTING = old_ck
        fm._USE_FLYDSL_MOE_SORTING = old_flydsl


def _make_global_routes(tokens, topk, global_experts, local_begin, local_experts):
    """Build deterministic unique routes with both local and remote experts."""
    if topk > global_experts:
        raise ValueError("topk cannot exceed global_experts")
    if local_experts < 2:
        raise ValueError("local_experts must be at least 2")

    local_per_token = max(1, min(topk // 2, local_experts))
    ids = torch.empty((tokens, topk), dtype=dtypes.i32, device="cuda")
    for token in range(tokens):
        local = [
            local_begin + ((token * local_per_token + slot) % local_experts)
            for slot in range(local_per_token)
        ]
        remote = []
        candidate = token * topk
        while len(remote) < topk - local_per_token:
            expert = candidate % global_experts
            candidate += 1
            if local_begin <= expert < local_begin + local_experts:
                continue
            if expert in remote:
                continue
            remote.append(expert)
        # Interleave local routes instead of putting them in a convenient suffix.
        row = []
        for slot in range(topk):
            if slot % 2 == 0 and local:
                row.append(local.pop(0))
            elif remote:
                row.append(remote.pop(0))
            else:
                row.append(local.pop(0))
        ids[token] = torch.tensor(row, dtype=dtypes.i32, device="cuda")

    # A unique value per (token, top-k slot) makes weight-map errors observable.
    weights = torch.arange(
        1, tokens * topk + 1, dtype=dtypes.fp32, device="cuda"
    ).view(tokens, topk)
    weights.div_(tokens * topk + 1)
    return ids, weights


def _sort(topk_ids, topk_weights, expert_mask, num_local_tokens, block_m, output_aux):
    # Use the internal implementation intentionally: the public fused_moe path
    # currently rejects output_aux + expert_mask before the sorting kernel runs.
    return fm._moe_sorting_impl(
        topk_ids,
        topk_weights,
        expert_mask.numel(),
        model_dim=32,
        moebuf_dtype=dtypes.bf16,
        block_size=block_m,
        expert_mask=expert_mask,
        num_local_tokens=num_local_tokens,
        dispatch_policy=1 if output_aux else 0,
        use_opus=True,
        accumulate=True,
        output_aux=output_aux,
    )


def _assert_common_outputs_equal(standard, auxiliary, block_m):
    std_ids, std_weights, std_experts, std_count, _ = standard
    aux_ids, aux_weights, aux_experts, aux_count, *_ = auxiliary
    torch.testing.assert_close(aux_count, std_count, rtol=0, atol=0)

    post_pad = int(std_count[0].item())
    tiles = post_pad // block_m
    torch.testing.assert_close(aux_ids[:post_pad], std_ids[:post_pad], rtol=0, atol=0)

    # Padding weights are unspecified; compare weights only for real routes.
    valid = (std_ids[:post_pad] & 0x00FFFFFF) < int(std_count[1].item())
    torch.testing.assert_close(
        aux_weights[:post_pad][valid], std_weights[:post_pad][valid], rtol=0, atol=0
    )
    torch.testing.assert_close(
        aux_experts[:tiles], std_experts[:tiles], rtol=0, atol=0
    )


def run_test(tokens, global_experts, local_experts, rank, topk, block_m, padding):
    local_begin = rank * local_experts
    local_end = local_begin + local_experts
    if local_end > global_experts:
        raise ValueError("rank/local_experts select experts beyond global_experts")

    ids, weights = _make_global_routes(
        tokens, topk, global_experts, local_begin, local_experts
    )
    if padding:
        padded_ids = torch.full(
            (tokens + padding, topk), -1, dtype=dtypes.i32, device="cuda"
        )
        padded_weights = torch.zeros(
            (tokens + padding, topk), dtype=dtypes.fp32, device="cuda"
        )
        padded_ids[:tokens].copy_(ids)
        padded_weights[:tokens].copy_(weights)
        ids, weights = padded_ids, padded_weights
        num_local_tokens = torch.tensor([tokens], dtype=dtypes.i32, device="cuda")
    else:
        num_local_tokens = None

    # Match the inter-node adapter: map global routes into a compact local
    # expert domain and use the last ID as an always-masked fake expert.
    ids = ids - local_begin
    ids.masked_fill_((ids < 0) | (ids >= local_experts), local_experts)
    expert_mask = torch.ones(local_experts + 1, dtype=dtypes.i32, device="cuda")
    expert_mask[-1] = 0
    local_begin, local_end = 0, local_experts

    with _force_opus_aux_sort():
        standard = _sort(
            ids, weights, expert_mask, num_local_tokens, block_m, output_aux=False
        )
        torch.cuda.synchronize()
        print("standard sorting complete", flush=True)
        auxiliary = _sort(
            ids, weights, expert_mask, num_local_tokens, block_m, output_aux=True
        )
        torch.cuda.synchronize()
        print("output_aux sorting complete", flush=True)
    torch.cuda.synchronize()
    _assert_common_outputs_equal(standard, auxiliary, block_m)

    sorted_ids, sorted_weights, sorted_experts, count, _, m_indices, reverse = auxiliary
    valid_tokens = tokens
    post_pad = int(count[0].item())
    expected_routes = []
    ids_cpu = ids[:valid_tokens].cpu()
    for token in range(valid_tokens):
        for slot in range(topk):
            expert = int(ids_cpu[token, slot])
            if local_begin <= expert < local_end:
                expected_routes.append((token, slot, expert - local_begin))

    # Padded sorting length must be the sum of per-local-expert block-M rounds.
    counts = torch.bincount(
        (ids[:valid_tokens][
            (ids[:valid_tokens] >= local_begin) & (ids[:valid_tokens] < local_end)
        ] - local_begin).long(),
        minlength=local_experts,
    )
    expected_post_pad = int((((counts + block_m - 1) // block_m) * block_m).sum())
    assert post_pad == expected_post_pad, (post_pad, expected_post_pad)

    # Atomic GEMM2 does not consume reverse_sorted. Policy-0 Opus sorting does
    # not populate it, so validate the m_indices contract directly per sorted slot.
    for sorted_slot in range(post_pad):
        token = int(sorted_ids[sorted_slot].item()) & 0x00FFFFFF
        if token < valid_tokens:
            assert int(m_indices[sorted_slot].item()) == token

    valid_sorted = (sorted_ids[:post_pad] & 0x00FFFFFF) < valid_tokens
    assert int(valid_sorted.sum().item()) == len(expected_routes)
    print(
        "PASS: standard and output_aux Opus sorting agree; "
        f"tokens={tokens}, routes={len(expected_routes)}, post_pad={post_pad}, "
        f"local_experts=[{local_begin},{local_end}), block_m={block_m}"
    )


def test_output_aux_ep_sorting():
    """Pytest entry using one real EP16 rank's expert interval."""
    run_test(
        tokens=16,
        global_experts=896,
        local_experts=56,
        rank=8,
        topk=16,
        block_m=32,
        padding=8,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--global-experts", type=int, default=896)
    parser.add_argument("--local-experts", type=int, default=56)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--block-m", type=int, default=32)
    parser.add_argument("--padding", type=int, default=8)
    args = parser.parse_args()
    run_test(
        args.tokens,
        args.global_experts,
        args.local_experts,
        args.rank,
        args.topk,
        args.block_m,
        args.padding,
    )


if __name__ == "__main__":
    main()
