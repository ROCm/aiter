# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Numerical regression for gfx950 FP8 MLA decode softmax.

The FP8 P*V path must scale softmax numerators before converting them to E4M3.
Otherwise, individually small but collectively important probabilities round to
zero while the FP32 denominator still includes their mass.
"""

import argparse

import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx

_HEADS = 16
_KV_LEN = 2048
_KV_LORA_RANK = 512
_ROPE_DIM = 64
_HEAD_DIM = _KV_LORA_RANK + _ROPE_DIM
_PAGE_SIZE = 1
_SOFTMAX_SCALE = 1.0 / 16.0
_FP8_MAX = 448.0


def _make_case(*, long_tail: bool, nhead: int = _HEADS, q_len: int = 1):
    device = torch.device("cuda")

    # Q_nope is zero so the first 512 cache elements are free to act as V.
    # Q_rope is one, making a cache rope value of -1.75 produce logit -7:
    # 64 * 1 * -1.75 * (1 / 16) == -7.
    q = torch.zeros((q_len, nhead, _HEAD_DIM), dtype=torch.float32, device=device)
    q[..., _KV_LORA_RANK:] = 1.0

    kv = torch.ones(
        (_KV_LEN, _PAGE_SIZE, 1, _HEAD_DIM),
        dtype=torch.float32,
        device=device,
    )
    kv[..., _KV_LORA_RANK:] = -1.75 if long_tail else 0.0
    if long_tail:
        # Put the maximum first so every later online-softmax tile uses the same
        # running maximum. The exact probabilities are 1 and exp(-7).
        kv[0, ..., _KV_LORA_RANK:] = 0.0

    q = q.to(dtypes.fp8)
    kv = kv.to(dtypes.fp8)

    qo_indptr = torch.tensor([0, q_len], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, _KV_LEN], dtype=torch.int32, device=device)
    kv_indices = torch.arange(_KV_LEN, dtype=torch.int32, device=device)
    kv_last_page_lens = torch.ones((1,), dtype=torch.int32, device=device)
    scale = torch.ones((1,), dtype=torch.float32, device=device)

    return q, kv, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens, scale


def _reference(q: torch.Tensor, kv: torch.Tensor, *, cp_world_size: int = 1):
    query = q.flatten(0, 1).float()
    cache = kv[:, 0, 0].float()
    value = cache[:, :_KV_LORA_RANK]

    scores = torch.einsum("hd,kd->hk", query, cache) * _SOFTMAX_SCALE
    visible_kv = (
        torch.arange(q.shape[0], device=q.device) + kv.shape[0] - q.shape[0] + 1
    )
    visible_kv = visible_kv.repeat_interleave(q.shape[1])
    masked = torch.arange(kv.shape[0], device=q.device)[None, :] >= visible_kv[:, None]
    scores.masked_fill_(masked, float("-inf"))
    scores = scores.reshape(query.shape[0], -1, cp_world_size).permute(2, 0, 1)
    value = value.reshape(-1, cp_world_size, _KV_LORA_RANK).permute(1, 0, 2)
    maxima = scores.amax(dim=-1, keepdim=True)
    numerators = torch.exp(scores - maxima)
    denominator = numerators.sum(dim=-1, keepdim=True)
    rank_lse = maxima + denominator.log()
    weights = torch.softmax(rank_lse, dim=0)

    fp32 = torch.einsum("rhk,rkd->rhd", numerators, value) / denominator
    unscaled_fp8 = (
        torch.einsum("rhk,rkd->rhd", numerators.to(dtypes.fp8).float(), value)
        / denominator
    )
    scaled_fp8 = torch.einsum(
        "rhk,rkd->rhd", (numerators * _FP8_MAX).to(dtypes.fp8).float(), value
    ) / (denominator * _FP8_MAX)
    return (
        (fp32 * weights).sum(dim=0),
        (unscaled_fp8 * weights).sum(dim=0),
        (scaled_fp8 * weights).sum(dim=0),
        torch.logsumexp(rank_lse, dim=0).squeeze(-1),
    )


def _run_aiter_rank(case, *, persistent, return_lse, cp_world_size, cp_rank):
    q, kv, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens, scale = case
    g_kv_indptr = kv_indptr if cp_world_size > 1 else None
    if cp_world_size > 1:
        kv_indices = kv_indices[cp_rank::cp_world_size].contiguous()
        kv_indptr = torch.tensor(
            [0, kv_indices.numel()], dtype=torch.int32, device=q.device
        )
    q_len, nhead = q.shape[:2]
    out = torch.empty((q_len, nhead, _KV_LORA_RANK), dtype=torch.bfloat16, device=q.device)
    metadata_args = {}
    if persistent:
        metadata = [
            torch.empty(shape, dtype=dtype, device=q.device)
            for shape, dtype in aiter.get_mla_metadata_info_v1(
                1,
                q_len,
                nhead,
                q.dtype,
                kv.dtype,
                is_sparse=False,
                fast_mode=True,
                num_kv_splits=1,
                intra_batch_mode=False,
            )
        ]
        (
            work_meta_data,
            work_indptr,
            work_info_set,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
        ) = metadata
        aiter.get_mla_metadata_v1(
            qo_indptr,
            kv_indptr,
            kv_last_page_lens,
            nhead,
            1,
            cp_world_size == 1,
            work_meta_data,
            work_info_set,
            work_indptr,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            page_size=_PAGE_SIZE,
            kv_granularity=32 if nhead == 64 else 16,
            max_seqlen_qo=q_len,
            uni_seqlen_qo=q_len,
            fast_mode=True,
            max_split_per_batch=1,
            intra_batch_mode=False,
            dtype_q_nope=q.dtype,
            dtype_kv_nope=kv.dtype,
            is_cp_round_robin=cp_world_size > 1,
        )
        metadata_args = dict(
            work_meta_data=work_meta_data,
            work_indptr=work_indptr,
            work_info_set=work_info_set,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
        )

    _, final_lse = aiter.mla.mla_decode_fwd(
        q,
        kv,
        out,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        q_len,
        _PAGE_SIZE,
        1,
        _SOFTMAX_SCALE,
        q_scale=scale,
        kv_scale=scale,
        num_kv_splits=1,
        return_lse=return_lse,
        causal=True,
        g_kv_indptr=g_kv_indptr,
        cp_world_size=cp_world_size,
        cp_rank=cp_rank,
        **metadata_args,
    )
    torch.cuda.synchronize()
    return out.flatten(0, 1).float(), (
        final_lse.flatten().float() if final_lse is not None else None
    )


def _run_aiter(case, *, persistent, return_lse, cp_world_size):
    rank_results = [
        _run_aiter_rank(
            case,
            persistent=persistent,
            return_lse=return_lse,
            cp_world_size=cp_world_size,
            cp_rank=cp_rank,
        )
        for cp_rank in range(cp_world_size)
    ]
    if cp_world_size == 1:
        return rank_results[0]
    rank_lse = torch.stack([result[1] for result in rank_results])
    final_lse = torch.logsumexp(rank_lse, dim=0)
    weights = torch.exp(rank_lse - final_lse).unsqueeze(-1)
    out = (torch.stack([result[0] for result in rank_results]) * weights).sum(dim=0)
    return out, final_lse


def _max_abs_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual - expected).abs().amax().item()


def test_mla_fp8_softmax_underflow(
    nhead: int = _HEADS,
    q_len: int = 1,
    persistent: bool = False,
    return_lse: bool = False,
    cp_world_size: int = 1,
) -> None:
    gfx = get_gfx()
    if gfx != "gfx950":
        print(f"Skipping: requires gfx950, got {gfx}")
        return

    if cp_world_size > 1:
        persistent = True
        return_lse = True
    run_args = dict(
        persistent=persistent, return_lse=return_lse, cp_world_size=cp_world_size
    )
    print(
        f"Case: {nhead=}, {q_len=}, {persistent=}, {return_lse=}, {cp_world_size=}",
        flush=True,
    )
    # First prove the fixture and dispatch are sane with representable, uniform
    # probabilities. Then exercise the long tail that exposed the bug.
    uniform = _make_case(long_tail=False, nhead=nhead, q_len=q_len)
    uniform_out, uniform_lse = _run_aiter(uniform, **run_args)
    uniform_ref, _, _, uniform_lse_ref = _reference(
        uniform[0], uniform[1], cp_world_size=cp_world_size
    )
    uniform_error = _max_abs_error(uniform_out, uniform_ref)
    assert uniform_error < 0.03, f"uniform control failed: {uniform_error=:g}"
    if return_lse:
        uniform_lse_error = _max_abs_error(uniform_lse, uniform_lse_ref)
        assert uniform_lse_error < 0.03, f"uniform LSE failed: {uniform_lse_error=:g}"

    long_tail = _make_case(long_tail=True, nhead=nhead, q_len=q_len)
    out, lse = _run_aiter(long_tail, **run_args)
    fp32_ref, unscaled_fp8_ref, scaled_fp8_ref, lse_ref = _reference(
        long_tail[0], long_tail[1], cp_world_size=cp_world_size
    )

    fp32_error = _max_abs_error(out, fp32_ref)
    unscaled_fp8_error = _max_abs_error(out, unscaled_fp8_ref)
    scaled_fp8_error = _max_abs_error(out, scaled_fp8_ref)
    print(
        "FP8 MLA softmax means: "
        f"out={out.mean().item():.6f}, "
        f"fp32={fp32_ref.mean().item():.6f}, "
        f"unscaled_fp8={unscaled_fp8_ref.mean().item():.6f}, "
        f"scaled_fp8={scaled_fp8_ref.mean().item():.6f}; "
        "errors: "
        f"{fp32_error=:.6f}, "
        f"{unscaled_fp8_error=:.6f}, "
        f"{scaled_fp8_error=:.6f}"
    )

    if return_lse:
        lse_error = _max_abs_error(lse, lse_ref)
        print(f"LSE error: {lse_error=:.6f}")
        assert lse_error < 0.03, f"long-tail LSE failed: {lse_error=:g}"
    assert _max_abs_error(scaled_fp8_ref, fp32_ref) < 0.03
    assert _max_abs_error(unscaled_fp8_ref, fp32_ref) > 0.1
    assert fp32_error < 0.03, (
        "FP8 MLA decode lost softmax numerator mass before P*V: "
        f"{fp32_error=:.6f}, {unscaled_fp8_error=:.6f}, "
        f"{scaled_fp8_error=:.6f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--nhead", type=int, choices=[8, 16, 32, 64], default=_HEADS)
    parser.add_argument("-q", "--q-len", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument("--persistent", action="store_true")
    parser.add_argument("-lse", "--return-lse", action="store_true")
    parser.add_argument("--cp-world-size", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    test_mla_fp8_softmax_underflow(
        nhead=args.nhead,
        q_len=args.q_len,
        persistent=args.persistent,
        return_lse=args.return_lse,
        cp_world_size=args.cp_world_size,
    )