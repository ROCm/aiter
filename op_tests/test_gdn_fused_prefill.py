# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and performance coverage for the single-kernel FlyDSL GDN prefill.

Each case runs ``gdn_fused_prefill_fwd`` on q/k/v taken as strided views of
one packed projection (the layout serving stacks pass) and compares the
output, the updated pool slots and the tracked chunk-boundary states against
an fp32 token-by-token recurrence. Rows with has_initial_state False start from
zero whatever their slot holds. Padding rows (slot -1) must write no pool slot
or track row, and slots outside the batch must stay untouched.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, run_perftest

torch.set_default_device("cuda")

BT = 64
K = V = 128
SLOTS = 16
# bf16 operands and a bf16 state copy feed the MFMAs. Against the fp32
# reference every sequence, from 1 token to 16k, sits at ~3.3e-3 rel L2 for o
# and ~3e-3 for its final state, with |o - ref| <= ~7e-4 where |ref| <= ~0.12.
# All bounds apply per sequence so one bad short sequence cannot hide in a
# long one.
SEQ_REL_L2_TOL = 1e-2
MAX_ABS_TOL = 2e-3
# Per sequence, the fused kernel must be at least as accurate as the
# multi-kernel FlyDSL path it replaces (~4.9e-3), up to rounding noise.
VS_MULTI_KERNEL_SLACK = 1.1


@dataclass(frozen=True)
class FusedShape:
    seq_lens: tuple[int, ...]
    hg: int
    h: int


SHAPES = {
    # Qwen3.8-Flash-Next at TP1 (Hg=16, H=48) and TP8 (Hg=2, H=6).
    "qwen38_dense_4k": FusedShape((4096,), 16, 48),
    "qwen38_dense_16k": FusedShape((16384,), 16, 48),
    "qwen38_varlen": FusedShape((1000, 64, 1, 2000, 333, 63, 65), 16, 48),
    "qwen38_tp8_16k": FusedShape((16384,), 2, 6),
    # Qwen3.5 397B at TP8 and 35B at TP2.
    "qwen35_tp8_tail": FusedShape((300, 129), 2, 8),
    "qwen35_tp8_16k": FusedShape((16384,), 2, 8),
    "qwen35_tp8_batch": FusedShape((2048,) * 8, 2, 8),
    "qwen35_tp2_varlen": FusedShape((128, 172, 300), 8, 16),
    "qwen35_tp2_long_mixed": FusedShape((8192, 512, 256), 8, 16),
}


def _load():
    from aiter.ops.flydsl.gdn_fused_prefill import (
        gdn_fused_prefill_fwd,
        gdn_fused_prefill_supported,
    )

    return gdn_fused_prefill_fwd, gdn_fused_prefill_supported


def _opt_vk_candidate(q, k, v, g, beta, cu, seq_lens, idx, backend):
    """The multi-kernel path serving stacks run today, on the same operands.

    ``backend`` picks the hidden-state stage: "triton", "flydsl" or "hip".
    """
    from aiter.ops.prefill_batch_metadata import (
        build_gated_delta_rule_prefill_metadata,
    )
    from aiter.ops.triton.gated_delta_net import chunk_gated_delta_rule_opt_vk

    # The metadata is bound to the exact offsets tensor passed at call time.
    cu64 = cu.long()
    meta = build_gated_delta_rule_prefill_metadata(list(seq_lens), cu_seqlens=cu64)

    def run(pool):
        o, _ = chunk_gated_delta_rule_opt_vk(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=pool,
            initial_state_indices=idx,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu64,
            use_chunk_flydsl=backend == "flydsl",
            use_chunk_hip=backend == "hip",
            use_prepare_flydsl=True,
            state_dtype=torch.bfloat16,
            prefill_metadata=meta,
        )
        return o

    return run


def _make_inputs(shape: FusedShape, seed: int = 2026):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    t, hg, h = sum(shape.seq_lens), shape.hg, shape.h
    mixed = torch.randn(t, 2 * hg * K + h * V, dtype=torch.bfloat16, generator=gen)
    q, k, v = torch.split(mixed, [hg * K, hg * K, h * V], dim=-1)
    q = q.view(1, t, hg, K)
    k = k.view(1, t, hg, K)
    v = v.view(1, t, h, V)
    g = (
        -torch.nn.functional.softplus(
            torch.randn(1, t, h, dtype=torch.float32, generator=gen)
        )
        * 0.5
    )
    beta = torch.randn(1, t, h, dtype=torch.float32, generator=gen).sigmoid()
    offsets = [0]
    for length in shape.seq_lens:
        offsets.append(offsets[-1] + length)
    cu = torch.tensor(offsets, dtype=torch.int32)
    pool = 0.05 * torch.randn(SLOTS, h, V, K, dtype=torch.bfloat16, generator=gen)
    n = len(shape.seq_lens)
    idx = (torch.arange(1, n + 1, dtype=torch.int32) * 2) % SLOTS
    nchunks = [(length + BT - 1) // BT for length in shape.seq_lens]
    # Middle chunk where there is one; even single-chunk rows track chunk 0,
    # odd ones opt out with -1.
    track_idx = torch.tensor(
        [c // 2 if c > 1 else (0 if i % 2 == 0 else -1) for i, c in enumerate(nchunks)],
        dtype=torch.int32,
    )
    return q, k, v, g, beta, cu, pool, idx, track_idx


@torch.no_grad()
def run_torch_reference(q, k, v, g, beta, cu, pool, idx, track_idx, has_init=None):
    """fp32 recurrence. Returns o, final states [N, H, V, K], tracked states.

    Rows with ``has_init[n]`` False start from a zero state.
    """
    h, rep = v.shape[2], v.shape[2] // q.shape[2]
    scale = K**-0.5

    def l2n(x):
        x = x.float()
        return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)

    qf = l2n(q[0]).repeat_interleave(rep, dim=1)
    kf = l2n(k[0]).repeat_interleave(rep, dim=1)
    vf, gf, bf = v[0].float(), g[0].float(), beta[0].float()
    out = torch.zeros(q.shape[1], h, V, dtype=torch.float32)
    finals, tracks = [], []
    for n in range(cu.numel() - 1):
        bos, eos = int(cu[n]), int(cu[n + 1])
        state = pool[int(idx[n])].float().clone()  # [H, V, K]
        if has_init is not None and not bool(has_init[n]):
            state = torch.zeros_like(state)
        tracked = None
        for t in range(bos, eos):
            if (t - bos) % BT == 0 and (t - bos) // BT == int(track_idx[n]):
                tracked = state.clone()
            state = state * gf[t].exp()[:, None, None]
            delta = vf[t] - torch.einsum("hvk,hk->hv", state, kf[t])
            state = state + (bf[t, :, None] * delta)[:, :, None] * kf[t][:, None, :]
            out[t] = torch.einsum("hvk,hk->hv", state, qf[t]) * scale
        finals.append(state)
        tracks.append(tracked)
    return out.unsqueeze(0), torch.stack(finals), tracks


def _rel_l2(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def _seq_errors(o, o_ref, pool, final_ref, cu, idx, rows):
    """Per-sequence o and final-state rel L2 for ``rows``, and the worst
    |o - ref| over their tokens."""
    o_errs, s_errs, max_abs = {}, {}, 0.0
    for i in rows:
        bos, eos = int(cu[i]), int(cu[i + 1])
        s_errs[i] = _rel_l2(pool[int(idx[i])], final_ref[i])
        if eos == bos:
            continue
        o_errs[i] = _rel_l2(o[0, bos:eos], o_ref[0, bos:eos])
        diff = (o[0, bos:eos].float() - o_ref[0, bos:eos]).abs().max().item()
        max_abs = max(max_abs, diff)
    return o_errs, s_errs, max_abs


def _assert_accurate(tag, o_errs, s_errs, max_abs):
    for what, errs in (("o", o_errs), ("state", s_errs)):
        worst = max(errs, key=errs.get)
        assert (
            errs[worst] < SEQ_REL_L2_TOL
        ), f"{tag}: {what} rel_l2 {errs[worst]:.3e} on sequence {worst}"
    assert max_abs < MAX_ABS_TOL, f"{tag}: o max abs error {max_abs:.3e}"


@benchmark()
def test_gdn_fused_prefill(name, seq_lens, hg, h):
    fwd, supported = _load()
    shape = FusedShape(tuple(seq_lens), hg, h)
    q, k, v, g, beta, cu, pool0, idx, track_idx = _make_inputs(shape)
    n = len(seq_lens)
    assert supported(q, k, v, g, beta, pool0), f"{name}: unexpectedly unsupported"

    # Even rows are new sequences (has_initial_state False). Their slots hold
    # NaN, which must be ignored rather than scaled to zero.
    has_init = torch.arange(n) % 2 == 1
    o_ref, final_ref, track_ref = run_torch_reference(
        q, k, v, g, beta, cu, pool0, idx, track_idx, has_init
    )

    pool = pool0.clone()
    pool[idx[~has_init].long()] = float("nan")
    track = torch.full((n, h, V, K), float("nan"), dtype=torch.float32)
    o = fwd(
        q,
        k,
        v,
        g,
        beta,
        ssm_states=pool,
        cache_indices=idx,
        cu_seqlens=cu,
        track_state=track,
        track_chunk_idx=track_idx,
        has_initial_state=has_init,
    )
    torch.cuda.synchronize()

    assert torch.isfinite(o).all(), f"{name}: non-finite output"
    o_errs, s_errs, max_abs = _seq_errors(o, o_ref, pool, final_ref, cu, idx, range(n))
    _assert_accurate(name, o_errs, s_errs, max_abs)
    track_errs = [
        _rel_l2(track[i], track_ref[i]) for i in range(n) if track_ref[i] is not None
    ]
    # NaN when no sequence tracks a chunk.
    err_track = max(track_errs, default=float("nan"))
    assert all(
        e < SEQ_REL_L2_TOL for e in track_errs
    ), f"{name}: track rel_l2 {err_track:.3e}"
    opted_out = [i for i in range(n) if track_ref[i] is None]
    assert torch.isnan(track[opted_out]).all(), f"{name}: wrote opted-out track rows"
    untouched = torch.ones(SLOTS, dtype=torch.bool)
    untouched[idx.long()] = False
    assert torch.equal(pool[untouched], pool0[untouched]), f"{name}: stray writes"

    # -1 (padding) rows are skipped. Pad the even rows so most shapes carry
    # several -1 entries; the live (odd, has_init) rows must still match on the
    # default path, which loads every slot.
    idx_pad = idx.clone()
    idx_pad[0::2] = -1
    live = idx_pad >= 0
    pool = pool0.clone()
    track = torch.full((n, h, V, K), float("nan"), dtype=torch.float32)
    o = fwd(
        q,
        k,
        v,
        g,
        beta,
        ssm_states=pool,
        cache_indices=idx_pad,
        cu_seqlens=cu,
        track_state=track,
        track_chunk_idx=track_idx,
    )
    torch.cuda.synchronize()
    touched = torch.zeros(SLOTS, dtype=torch.bool)
    touched[idx_pad[live].long()] = True
    assert torch.equal(pool[~touched], pool0[~touched]), f"{name}: padding leaked"
    assert torch.isnan(track[~live]).all(), f"{name}: padding wrote track rows"
    if live.any():
        rows = live.nonzero().flatten().tolist()
        _assert_accurate(
            f"{name} padded batch",
            *_seq_errors(o, o_ref, pool, final_ref, cu, idx_pad, rows),
        )

    # A/B against the multi-kernel path on the same operands and GPU. Fresh
    # (has_initial_state False) slots are zeroed up front, the contract that
    # path needs, so every candidate is checked against the same reference.
    pool_ab = pool0.clone()
    pool_ab[idx[~has_init].long()] = 0

    def fused(pool):
        return fwd(q, k, v, g, beta, ssm_states=pool, cache_indices=idx, cu_seqlens=cu)

    candidates = {"fused": fused}
    for backend in ("triton", "flydsl"):
        candidates[f"opt_vk_{backend}"] = _opt_vk_candidate(
            q, k, v, g, beta, cu, seq_lens, idx, backend
        )

    t = sum(seq_lens)
    # Per token and head: q k^T and k k^T, the two [BT, BT] x [BT, V] products,
    # and k S^T, q S^T and the state update.
    flops = t * h * (4 * BT * K + 4 * BT * V + 6 * K * V)
    nbytes = (
        (q[0].numel() + k[0].numel() + 2 * v[0].numel()) * q.element_size()
        + (g.numel() + beta.numel()) * g.element_size()
        + 2 * n * h * V * K * pool0.element_size()
    )
    ret = {
        "gfx": get_gfx(),
        "tokens": t,
        # What a caller passing max_seqlen would pick.
        "gate": supported(q, k, v, g, beta, pool0, max_seqlen=max(seq_lens)),
        # Worst sequence for each.
        "o_rel_l2": max(o_errs.values()),
        "state_rel_l2": max(s_errs.values()),
        "track_rel_l2": err_track,
        "o_max_abs": max_abs,
    }
    seq_errs = {}
    for cname, fn in candidates.items():
        pool = pool_ab.clone()
        errs = _seq_errors(fn(pool), o_ref, pool, final_ref, cu, idx, range(n))
        _assert_accurate(f"{name}: {cname}", *errs)
        seq_errs[cname] = errs
        _, us = run_perftest(fn, pool_ab.clone(), num_iters=21, num_warmup=3)
        ret[f"{cname} us"] = us
        ret[f"{cname} TFLOPS"] = flops / us / 1e6
        ret[f"{cname} TB/s"] = nbytes / us / 1e6
        ret[f"{cname} err"] = max(errs[0].values())

    # The bf16 state copy feeding k S^T / q S^T must not drift on long
    # sequences: per sequence, o and the final state stay as accurate as the
    # multi-kernel FlyDSL path.
    fused, multi = seq_errs["fused"], seq_errs["opt_vk_flydsl"]
    for what, part in (("o", 0), ("state", 1)):
        for i, err in fused[part].items():
            bound = VS_MULTI_KERNEL_SLACK * multi[part][i]
            assert err <= bound, (
                f"{name}: fused {what} rel_l2 {err:.3e} on sequence {i} exceeds "
                f"multi-kernel FlyDSL {multi[part][i]:.3e}"
            )
    return ret


def _run_contract_checks():
    fwd, supported = _load()
    q, k, v, g, beta, cu, pool, idx, _ = _make_inputs(SHAPES["qwen35_tp8_tail"])
    assert not supported(q.float(), k, v, g, beta, pool), "fp32 q accepted"
    assert not supported(q, k, v, g.bfloat16(), beta, pool), "bf16 g accepted"
    assert not supported(q, k, v, g, beta, pool.float()), "fp32 pool accepted"
    assert not supported(q, k, v, g, beta, pool.transpose(-1, -2)), "strided pool"
    assert not supported(q[..., :64], k[..., :64], v, g, beta, pool), "K=64"
    assert supported(q, k, v, g, beta, pool, max_seqlen=300), "short batch gated"
    try:
        fwd(q.float(), k, v, g, beta, ssm_states=pool, cache_indices=idx, cu_seqlens=cu)
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported operands did not raise")

    # Perf gate: a lone long sequence is rejected at H=8 but kept at H=48.
    for shape, expect in (("qwen35_tp8_16k", False), ("qwen38_dense_16k", True)):
        q, k, v, g, beta, _, pool, _, _ = _make_inputs(SHAPES[shape])
        assert supported(q, k, v, g, beta, pool), f"{shape}: operands rejected"
        got = supported(q, k, v, g, beta, pool, max_seqlen=q.shape[1])
        assert got == expect, f"{shape}: perf gate returned {got}"


def main():
    if get_gfx() != "gfx950":
        aiter.logger.warning("FlyDSL fused GDN prefill needs gfx950; skipping")
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Single-kernel FlyDSL GDN prefill correctness and performance",
    )
    parser.add_argument(
        "-s",
        "--shapes",
        type=str,
        nargs="*",
        choices=list(SHAPES),
        default=list(SHAPES),
        help="Shape labels.",
    )
    args = parser.parse_args()

    _run_contract_checks()
    rows = []
    for name in args.shapes:
        shape = SHAPES[name]
        rows.append(
            test_gdn_fused_prefill(name, list(shape.seq_lens), shape.hg, shape.h)
        )
    aiter.logger.info(
        "gdn_fused_prefill summary:\n%s", pd.DataFrame(rows).to_markdown()
    )


if __name__ == "__main__":
    main()
