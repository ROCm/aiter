# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import math
from functools import partial

import torch
import torch.nn.functional as F
import triton

from aiter import logger
from aiter.ops.triton.attention.broadcast_q_causal import (
    broadcast_q_causal_varlen_func,
)
from aiter.ops.triton.attention.mha import flash_attn_varlen_func

BATCH = 768
HEADS = 4
Q_LEN = 128
CORRECTNESS_BATCH = 8


def _make_lengths(name):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1234)
    if name == "dense_512":
        return torch.full((BATCH,), 512, dtype=torch.int32)
    if name == "dense_1024":
        return torch.full((BATCH,), 1024, dtype=torch.int32)
    if name == "mixed_1024":
        return torch.randint(
            256, 1025, (BATCH,), generator=generator, dtype=torch.int32
        )
    if name == "skew_bimodal":
        lengths = torch.empty(BATCH, dtype=torch.int32)
        lengths[: BATCH // 2] = 32
        lengths[BATCH // 2 :] = 1024
        return lengths
    raise ValueError("Unknown length distribution")


def _make_inputs(lengths, head_dim):
    torch.manual_seed(1234)
    total_kv = int(lengths.sum())
    q = torch.randn((Q_LEN, HEADS, head_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((total_kv, HEADS, head_dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    cu_seqlens_k = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32),
            lengths.cumsum(dim=0, dtype=torch.int32),
        )
    ).cuda()
    grad_out = torch.randn(
        (BATCH * Q_LEN, HEADS, head_dim), device="cuda", dtype=torch.bfloat16
    )
    return q, k, v, cu_seqlens_k, grad_out


def _active_indices(lengths):
    active_lengths = lengths.clamp(max=Q_LEN)
    q_indices = []
    local_rows = []
    local_keys = []
    for batch, (start, active_len) in enumerate(
        zip(torch.cumsum(lengths, dim=0) - lengths, active_lengths)
    ):
        q_indices.append(torch.arange(int(active_len), dtype=torch.int64))
        local_rows.append(
            batch * Q_LEN + torch.arange(int(active_len), dtype=torch.int64)
        )
        local_keys.append(int(start) + torch.arange(int(active_len), dtype=torch.int64))
    row_indices = torch.cat(local_rows).cuda()
    key_indices = torch.cat(local_keys).cuda()
    cu_active = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32),
            active_lengths.cumsum(dim=0, dtype=torch.int32),
        )
    ).cuda()
    q_indices = torch.cat(q_indices).cuda()
    return active_lengths, q_indices, row_indices, key_indices, cu_active


def _expanded_inputs(q, k, v, q_indices, key_indices, cu_active):
    q_expanded = q.index_select(0, q_indices)
    k_expanded = k.index_select(0, key_indices)
    v_expanded = v.index_select(0, key_indices)
    return q_expanded, k_expanded, v_expanded, cu_active


def _aiter_call(q_expanded, k_expanded, v_expanded, cu_active, max_active):
    return flash_attn_varlen_func(
        q_expanded,
        k_expanded,
        v_expanded,
        cu_active,
        cu_active,
        Q_LEN,
        max_active,
        causal=True,
        return_lse=True,
    )


def _aiter_forward_no_grad(q_expanded, k_expanded, v_expanded, cu_active, max_active):
    with torch.no_grad():
        return _aiter_call(q_expanded, k_expanded, v_expanded, cu_active, max_active)


def _aiter_forward_bwd(
    q_expanded,
    k_expanded,
    v_expanded,
    cu_active,
    max_active,
    grad_out,
    row_indices,
):
    out_expanded, _ = _aiter_call(
        q_expanded, k_expanded, v_expanded, cu_active, max_active
    )
    do_expanded = grad_out.index_select(0, row_indices)
    torch.autograd.grad(out_expanded, (q_expanded, k_expanded, v_expanded), do_expanded)


def _direct_forward(q, k, v, cu_seqlens_k):
    with torch.no_grad():
        return broadcast_q_causal_varlen_func(q, k, v, cu_seqlens_k, return_lse=False)


def _direct_forward_bwd(q, k, v, cu_seqlens_k, grad_out):
    out = broadcast_q_causal_varlen_func(q, k, v, cu_seqlens_k, return_lse=False)
    return torch.autograd.grad(out, (q, k, v), grad_out)


def _expanded_forward(
    q, k, v, q_indices, row_indices, key_indices, cu_active, max_active
):
    with torch.no_grad():
        q_expanded, k_expanded, v_expanded, cu_active = _expanded_inputs(
            q, k, v, q_indices, key_indices, cu_active
        )
        out_expanded, _ = _aiter_call(
            q_expanded, k_expanded, v_expanded, cu_active, max_active
        )
        return out_expanded


def _expanded_forward_bwd(
    q, k, v, q_indices, row_indices, key_indices, cu_active, max_active, grad_out
):
    q_expanded, k_expanded, v_expanded, cu_active = _expanded_inputs(
        q, k, v, q_indices, key_indices, cu_active
    )
    out_expanded, _ = _aiter_call(
        q_expanded, k_expanded, v_expanded, cu_active, max_active
    )
    do_expanded = grad_out.index_select(0, row_indices)
    _, dk_expanded, dv_expanded = torch.autograd.grad(
        out_expanded, (q_expanded, k_expanded, v_expanded), do_expanded
    )
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    dk.index_add_(0, key_indices, dk_expanded)
    dv.index_add_(0, key_indices, dv_expanded)
    return out_expanded, dk, dv


def _reference_subset(head_dim):
    torch.manual_seed(1234)
    lengths = torch.full((CORRECTNESS_BATCH,), 512, dtype=torch.int32)
    total_kv = int(lengths.sum())
    q = torch.randn((Q_LEN, HEADS, head_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((total_kv, HEADS, head_dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    cu_seqlens_k = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32),
            lengths.cumsum(dim=0, dtype=torch.int32),
        )
    ).cuda()
    grad_out = torch.randn(
        (CORRECTNESS_BATCH * Q_LEN, HEADS, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )

    out, lse = broadcast_q_causal_varlen_func(q, k, v, cu_seqlens_k)
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), grad_out)

    q_ref = q.detach().float().requires_grad_()
    k_ref = k.detach().float().requires_grad_()
    v_ref = v.detach().float().requires_grad_()
    out_blocks = []
    lse_blocks = []
    scale = head_dim**-0.5
    for batch in range(CORRECTNESS_BATCH):
        start = batch * 512
        q_block = q_ref.transpose(0, 1).unsqueeze(0)
        k_block = k_ref[start : start + Q_LEN].transpose(0, 1).unsqueeze(0)
        v_block = v_ref[start : start + Q_LEN].transpose(0, 1).unsqueeze(0)
        mask = torch.tril(torch.ones((Q_LEN, Q_LEN), device="cuda", dtype=torch.bool))
        out_blocks.append(
            F.scaled_dot_product_attention(
                q_block,
                k_block,
                v_block,
                attn_mask=mask,
                scale=scale,
            )
            .squeeze(0)
            .transpose(0, 1)
        )
        scores = (
            torch.einsum("mhd,nhd->hmn", q_ref, k_ref[start : start + Q_LEN]) * scale
        )
        scores = scores.masked_fill(~mask, -torch.inf)
        lse_blocks.append(torch.logsumexp(scores, dim=-1).T)

    out_ref = torch.stack(out_blocks)
    lse_ref = torch.stack(lse_blocks)

    grad_out_ref = grad_out.float().reshape_as(out_ref)
    ref_grads = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), grad_out_ref)
    torch.testing.assert_close(
        out.float().reshape_as(out_ref), out_ref, atol=0.02, rtol=0.02
    )
    torch.testing.assert_close(lse.reshape_as(lse_ref), lse_ref, atol=0.02, rtol=0.02)
    for actual, expected in zip((dq, dk, dv), ref_grads):
        torch.testing.assert_close(actual.float(), expected, atol=0.05, rtol=0.05)


def _measure(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024**2)


def _geomean(values):
    return math.exp(sum(math.log(value) for value in values) / len(values))


def run(head_dim):
    _reference_subset(head_dim)
    results = []
    for name in ("dense_512", "dense_1024", "mixed_1024", "skew_bimodal"):
        lengths = _make_lengths(name)
        q, k, v, cu_seqlens_k, grad_out = _make_inputs(lengths, head_dim)
        active_lengths, q_indices, row_indices, key_indices, cu_active = (
            _active_indices(lengths)
        )
        max_active = int(active_lengths.max())
        q_expanded, k_expanded, v_expanded, _ = _expanded_inputs(
            q, k, v, q_indices, key_indices, cu_active
        )

        direct_fwd = triton.testing.do_bench(
            partial(_direct_forward, q, k, v, cu_seqlens_k), warmup=5, rep=20
        )
        direct_fwd_bwd = triton.testing.do_bench(
            partial(_direct_forward_bwd, q, k, v, cu_seqlens_k, grad_out),
            warmup=5,
            rep=20,
        )

        aiter_fwd = triton.testing.do_bench(
            partial(
                _aiter_forward_no_grad,
                q_expanded,
                k_expanded,
                v_expanded,
                cu_active,
                max_active,
            ),
            warmup=5,
            rep=20,
        )
        aiter_fwd_bwd_time = triton.testing.do_bench(
            partial(
                _aiter_forward_bwd,
                q_expanded,
                k_expanded,
                v_expanded,
                cu_active,
                max_active,
                grad_out,
                row_indices,
            ),
            warmup=5,
            rep=20,
        )
        expanded_fwd = triton.testing.do_bench(
            partial(
                _expanded_forward,
                q,
                k,
                v,
                q_indices,
                row_indices,
                key_indices,
                cu_active,
                max_active,
            ),
            warmup=5,
            rep=20,
        )
        expanded_fwd_bwd = triton.testing.do_bench(
            partial(
                _expanded_forward_bwd,
                q,
                k,
                v,
                q_indices,
                row_indices,
                key_indices,
                cu_active,
                max_active,
                grad_out,
            ),
            warmup=5,
            rep=20,
        )
        direct_peak = _measure(
            partial(_direct_forward_bwd, q, k, v, cu_seqlens_k, grad_out)
        )
        expanded_peak = _measure(
            partial(
                _expanded_forward_bwd,
                q,
                k,
                v,
                q_indices,
                row_indices,
                key_indices,
                cu_active,
                max_active,
                grad_out,
            )
        )
        results.append(
            {
                "name": name,
                "direct_fwd": direct_fwd,
                "direct_fwd_bwd": direct_fwd_bwd,
                "aiter_fwd": aiter_fwd,
                "aiter_fwd_bwd": aiter_fwd_bwd_time,
                "expanded_fwd": expanded_fwd,
                "expanded_fwd_bwd": expanded_fwd_bwd,
                "direct_peak": direct_peak,
                "expanded_peak": expanded_peak,
            }
        )
        logger.info(
            "%s: direct fwd=%.3fms fwd_bwd=%.3fms peak=%.1fMB peak_delta=%.1fMB | "
            "aiter-kernel fwd=%.3fms fwd_bwd=%.3fms | expanded-e2e fwd=%.3fms "
            "fwd_bwd=%.3fms peak=%.1fMB",
            name,
            direct_fwd,
            direct_fwd_bwd,
            direct_peak,
            expanded_peak - direct_peak,
            aiter_fwd,
            aiter_fwd_bwd_time,
            expanded_fwd,
            expanded_fwd_bwd,
            expanded_peak,
        )

    geomeans = {
        key: _geomean([result[key] for result in results])
        for key in (
            "direct_fwd",
            "direct_fwd_bwd",
            "aiter_fwd",
            "aiter_fwd_bwd",
            "expanded_fwd",
            "expanded_fwd_bwd",
            "direct_peak",
            "expanded_peak",
        )
    }
    geomeans["direct_peak"] = max(result["direct_peak"] for result in results)
    geomeans["expanded_peak"] = max(result["expanded_peak"] for result in results)
    logger.info(
        "geomeans: direct fwd=%.3fms fwd_bwd=%.3fms peak=%.1fMB | "
        "aiter-kernel fwd=%.3fms fwd_bwd=%.3fms | expanded-e2e fwd=%.3fms "
        "fwd_bwd=%.3fms peak=%.1fMB",
        geomeans["direct_fwd"],
        geomeans["direct_fwd_bwd"],
        geomeans["direct_peak"],
        geomeans["aiter_fwd"],
        geomeans["aiter_fwd_bwd"],
        geomeans["expanded_fwd"],
        geomeans["expanded_fwd_bwd"],
        geomeans["expanded_peak"],
    )
    logger.info(
        "ratios: direct/aiter-kernel fwd=%.3fx fwd_bwd=%.3fx | "
        "direct/expanded-e2e fwd=%.3fx fwd_bwd=%.3fx peak=%.3fx peak_delta=%.1fMB",
        geomeans["direct_fwd"] / geomeans["aiter_fwd"],
        geomeans["direct_fwd_bwd"] / geomeans["aiter_fwd_bwd"],
        geomeans["direct_fwd"] / geomeans["expanded_fwd"],
        geomeans["direct_fwd_bwd"] / geomeans["expanded_fwd_bwd"],
        geomeans["direct_peak"] / geomeans["expanded_peak"],
        geomeans["expanded_peak"] - geomeans["direct_peak"],
    )
    logger.info("torch=%s", torch.__version__)
    logger.info("triton=%s", triton.__version__)
    logger.info("device=%s", torch.cuda.get_device_name())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-dim", type=int, choices=(64, 128, 256), default=128)
    args = parser.parse_args()
    run(args.head_dim)


if __name__ == "__main__":
    main()
