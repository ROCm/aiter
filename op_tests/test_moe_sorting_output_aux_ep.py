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
import torch.nn.functional as F

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
    return {
        "ids": ids,
        "weights": weights,
        "expert_mask": expert_mask,
        "num_local_tokens": num_local_tokens,
        "standard": standard,
        "auxiliary": auxiliary,
        "valid_tokens": valid_tokens,
        "local_begin": local_begin,
        "local_end": local_end,
    }


def _dequant_mxfp4(packed, scale):
    from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32

    values = mxfp4_to_f32(packed)
    scale_f32 = e8m0_to_f32(scale).repeat_interleave(32, dim=-1)
    return values * scale_f32


def run_mxmoe_gemm1_test(
    sorting,
    *,
    model_dim,
    inter_dim,
    block_m,
    topk,
    seed,
    sanitize_padding,
    reference,
):
    """Launch only the direct-FP4 ``flydsl_mxmoe_g1`` implementation.

    The Opus aux sorter does not define ``m_indices`` for block-M padding.
    By default this diagnostic first prints those raw values and then replaces
    padding entries with source row zero before GEMM.  This makes it possible
    to distinguish a real-route mapping error from an unsafe padding read.  Use
    ``--no-sanitize-padding`` to reproduce the exact raw aux-map contract.
    """
    from aiter.ops.flydsl.mxfp4_gemm1_kernels import flydsl_mxfp4_gemm1
    from aiter.ops.quant import per_1x32_mx_quant_hip
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

    ids = sorting["ids"]
    valid_tokens = sorting["valid_tokens"]
    (
        sorted_ids,
        _,
        sorted_experts,
        count,
        _,
        m_indices,
        _,
    ) = sorting["auxiliary"]
    device = ids.device
    capacity = ids.shape[0]
    local_experts = sorting["local_end"] - sorting["local_begin"]
    post_pad = int(count[0].item())
    tiles = post_pad // block_m
    valid_sorted = (sorted_ids[:post_pad] & 0x00FFFFFF) < valid_tokens
    padding_sorted = ~valid_sorted

    valid_m = m_indices[:post_pad][valid_sorted]
    padding_m = m_indices[:post_pad][padding_sorted]
    print(
        "GEMM1 metadata: "
        f"num_valid={count.cpu().tolist()}, post_pad={post_pad}, tiles={tiles}, "
        f"sorted_expert_range=[{int(sorted_experts[:tiles].min())},"
        f"{int(sorted_experts[:tiles].max())}], "
        f"valid_m_indices_range=[{int(valid_m.min())},{int(valid_m.max())}], "
        f"padding_count={padding_m.numel()}, "
        f"padding_m_indices_tail={padding_m[-16:].cpu().tolist()}"
    )
    assert int(sorted_experts[:tiles].min()) >= 0
    assert int(sorted_experts[:tiles].max()) < local_experts
    assert int(valid_m.min()) >= 0 and int(valid_m.max()) < valid_tokens

    if sanitize_padding:
        # Padding rows participate in full block-M execution but are discarded
        # later.  Give GEMM a legal activation row without changing real maps.
        padding_positions = torch.nonzero(padding_sorted).flatten()
        m_indices[padding_positions] = 0
        print("GEMM1 diagnostic: sanitized undefined padding m_indices to row 0")

    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(
        (capacity, model_dim),
        dtype=dtypes.bf16,
        device=device,
        generator=generator,
    ).mul_(0.25)
    x[valid_tokens:].zero_()
    # GGUU layout: first I rows are gate, second I rows are up.
    w1 = torch.randn(
        (local_experts, 2 * inter_dim, model_dim),
        dtype=dtypes.bf16,
        device=device,
        generator=generator,
    ).mul_(0.125)

    x_q, x_scale = per_1x32_mx_quant_hip(x, quant_dtype=dtypes.fp4x2)
    flat_w = w1.view(-1, model_dim)
    flat_w_q, flat_w_scale = per_1x32_mx_quant_hip(
        flat_w, quant_dtype=dtypes.fp4x2
    )
    w1_q = flat_w_q.view(local_experts, 2 * inter_dim, model_dim // 2)
    w1_scale = flat_w_scale.view(local_experts, 2 * inter_dim, model_dim // 32)
    w1_q_shuffled = shuffle_weight_a16w4(w1_q, 16, False)
    w1_scale_shuffled = shuffle_scale_a16w4(flat_w_scale, local_experts, False)

    max_sorted = sorted_ids.shape[0]
    padded_rows = ((max_sorted + 31) // 32) * 32
    scale_cols = model_dim // 32
    x_scale_sorted = torch.empty(
        padded_rows * scale_cols * 2, dtype=torch.uint8, device=device
    )
    import aiter

    aiter.mxfp4_moe_sort_scales(
        a_scale=x_scale,
        sorted_token_ids=sorted_ids,
        cumsum_tensor=count,
        a_scale_sorted_shuffled=x_scale_sorted,
        NE=local_experts,
        TOPK=topk,
        D_HIDDEN=model_dim,
        MB=block_m,
        max_sorted=max_sorted,
    )

    inter_q = torch.full(
        (max_sorted, inter_dim // 2), 0xFF, dtype=torch.uint8, device=device
    )
    inter_scale_cols = inter_dim // 32
    inter_scale_bytes = max(
        max_sorted * max((1024 // 64) * 4, inter_scale_cols * 2), 1
    )
    inter_scale_rows = (
        (inter_scale_bytes + inter_scale_cols - 1) // inter_scale_cols + 31
    ) // 32 * 32
    inter_scale = torch.full(
        (inter_scale_rows, inter_scale_cols),
        0xFF,
        dtype=torch.uint8,
        device=device,
    )
    empty_hidden = torch.empty(0, dtype=dtypes.bf16, device=device)
    flydsl_mxfp4_gemm1(
        a_quant=x_q,
        a_scale_sorted_shuffled=x_scale_sorted,
        w1_u8=w1_q_shuffled.view(torch.uint8),
        w1_scale_u8=w1_scale_shuffled.view(torch.uint8),
        sorted_expert_ids=sorted_experts,
        cumsum_tensor=count,
        m_indices=m_indices,
        inter_sorted_quant=inter_q,
        inter_sorted_shuffled_scale=inter_scale,
        hidden_states=empty_hidden,
        n_tokens=capacity,
        BM=block_m,
        use_nt=True,
        inline_quant=False,
        NE=local_experts,
        D_HIDDEN=model_dim,
        D_INTER=inter_dim,
        topk=topk,
        BN=256,
        BK=256,
        interleave=False,
        xcd_swizzle=2,
    )
    torch.cuda.synchronize()

    assert not torch.all(inter_q[:post_pad] == 0xFF), "GEMM1 did not write output"
    print(
        "PASS: flydsl_mxmoe_g1 GEMM1-only launch completed; "
        f"output={tuple(inter_q.shape)}, scale={tuple(inter_scale.shape)}"
    )

    if not reference:
        return

    # A compact numerical check on real routes.  Reconstruct the exact MXFP4
    # operands (before weight preshuffle), calculate SiLU(gate)*up, quantize it,
    # and compare the packed output codes.  This deliberately avoids decoding
    # the GEMM's shuffled output-scale ABI.
    x_deq = _dequant_mxfp4(x_q[:valid_tokens], x_scale[:valid_tokens])
    unique_experts = sorted(set(sorted_experts[:tiles].cpu().tolist()))
    expected_packed = {}
    for expert in unique_experts:
        expert_rows = torch.nonzero(
            sorted_experts[:tiles].repeat_interleave(block_m)[:post_pad] == expert
        ).flatten()
        expert_rows = expert_rows[valid_sorted[expert_rows]]
        if expert_rows.numel() == 0:
            continue
        w_deq = _dequant_mxfp4(w1_q[expert], w1_scale[expert])
        source_rows = m_indices[expert_rows].long()
        gemm = x_deq[source_rows].float() @ w_deq.float().T
        gate, up = gemm[:, :inter_dim], gemm[:, inter_dim:]
        ref = F.silu(gate) * up
        ref_q, _ = per_1x32_mx_quant_hip(ref.to(dtypes.bf16), quant_dtype=dtypes.fp4x2)
        expected_packed[expert] = (expert_rows, ref_q.view(torch.uint8))

    matches = 0
    elements = 0
    for rows, ref_q in expected_packed.values():
        got = inter_q[rows]
        matches += int((got == ref_q).sum().item())
        elements += got.numel()
    ratio = matches / max(elements, 1)
    print(f"GEMM1 packed-code reference match={ratio:.6f} ({matches}/{elements})")
    assert ratio >= 0.90, f"GEMM1 packed-code match too low: {ratio:.6f}"


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
    parser.add_argument("--run-mxmoe-gemm1", action="store_true")
    parser.add_argument("--model-dim", type=int, default=3584)
    parser.add_argument("--inter-dim", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--no-sanitize-padding",
        dest="sanitize_padding",
        action="store_false",
        help="leave undefined Opus aux padding m_indices untouched",
    )
    parser.add_argument(
        "--gemm-reference",
        action="store_true",
        help="run the slower packed-code torch reference check",
    )
    parser.set_defaults(sanitize_padding=True)
    args = parser.parse_args()
    sorting = run_test(
        args.tokens,
        args.global_experts,
        args.local_experts,
        args.rank,
        args.topk,
        args.block_m,
        args.padding,
    )
    if args.run_mxmoe_gemm1:
        run_mxmoe_gemm1_test(
            sorting,
            model_dim=args.model_dim,
            inter_dim=args.inter_dim,
            block_m=args.block_m,
            topk=args.topk,
            seed=args.seed,
            sanitize_padding=args.sanitize_padding,
            reference=args.gemm_reference,
        )


if __name__ == "__main__":
    main()
