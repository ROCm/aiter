# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and performance coverage for the single-kernel FlyDSL GDN prefill.

Each case runs ``gdn_fused_prefill_fwd`` on q/k/v taken as strided views of
one packed projection (the layout serving stacks pass) and compares the
output, the updated pool slots and the tracked chunk-boundary states against
an fp32 token-by-token recurrence. A padding row (slot -1) must touch slot 0
only, and slots outside the batch must stay untouched.
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
# bf16 operands and a bf16 state copy feed the MFMAs; the fp32 reference sits
# at ~4e-3 relative L2 on these shapes.
REL_L2_TOL = 2e-2


@dataclass(frozen=True)
class FusedShape:
    seq_lens: tuple[int, ...]
    hg: int
    h: int


SHAPES = {
    # Qwen3.8-Flash-Next at TP1: Hg=16, H=48.
    "qwen38_dense_4k": FusedShape((4096,), 16, 48),
    "qwen38_varlen": FusedShape((1000, 64, 1, 2000, 333, 63, 65), 16, 48),
    # Qwen3.5 397B at TP8 and 35B at TP2.
    "qwen35_tp8_tail": FusedShape((300, 129), 2, 8),
    "qwen35_tp2_varlen": FusedShape((128, 172, 300), 8, 16),
}


def _load():
    from aiter.ops.flydsl.gdn_fused_prefill import (
        gdn_fused_prefill_fwd,
        gdn_fused_prefill_supported,
    )

    return gdn_fused_prefill_fwd, gdn_fused_prefill_supported


def _make_inputs(shape: FusedShape, seed: int = 2026):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    t, hg, h = sum(shape.seq_lens), shape.hg, shape.h
    mixed = torch.randn(t, 2 * hg * K + h * V, dtype=torch.bfloat16, generator=gen)
    q, k, v = torch.split(mixed, [hg * K, hg * K, h * V], dim=-1)
    q = q.view(1, t, hg, K)
    k = k.view(1, t, hg, K)
    v = v.view(1, t, h, V)
    g = -torch.nn.functional.softplus(
        torch.randn(1, t, h, dtype=torch.float32, generator=gen)
    ) * 0.5
    beta = torch.randn(1, t, h, dtype=torch.float32, generator=gen).sigmoid()
    offsets = [0]
    for length in shape.seq_lens:
        offsets.append(offsets[-1] + length)
    cu = torch.tensor(offsets, dtype=torch.int32)
    pool = 0.05 * torch.randn(SLOTS, h, V, K, dtype=torch.bfloat16, generator=gen)
    pool[0].zero_()
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
def run_torch_reference(q, k, v, g, beta, cu, pool, idx, track_idx):
    """fp32 recurrence. Returns o, final states [N, H, V, K], tracked states."""
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
        state = pool[max(int(idx[n]), 0)].float().clone()  # [H, V, K]
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


@benchmark()
def test_gdn_fused_prefill(name, seq_lens, hg, h):
    fwd, supported = _load()
    shape = FusedShape(tuple(seq_lens), hg, h)
    q, k, v, g, beta, cu, pool0, idx, track_idx = _make_inputs(shape)
    n = len(seq_lens)
    assert supported(q, k, v, g, beta, pool0), f"{name}: unexpectedly unsupported"

    o_ref, final_ref, track_ref = run_torch_reference(
        q, k, v, g, beta, cu, pool0, idx, track_idx
    )

    pool = pool0.clone()
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
    )
    torch.cuda.synchronize()

    err_o = _rel_l2(o, o_ref)
    err_state = _rel_l2(pool[idx.long()], final_ref)
    rows = [i for i in range(n) if track_ref[i] is not None]
    err_track = _rel_l2(track[rows], torch.stack([track_ref[i] for i in rows]))
    assert torch.isfinite(o).all(), f"{name}: non-finite output"
    assert err_o < REL_L2_TOL, f"{name}: o rel_l2 {err_o:.3e}"
    assert err_state < REL_L2_TOL, f"{name}: state rel_l2 {err_state:.3e}"
    assert err_track < REL_L2_TOL, f"{name}: track rel_l2 {err_track:.3e}"
    opted_out = [i for i in range(n) if track_ref[i] is None]
    assert torch.isnan(track[opted_out]).all(), f"{name}: wrote opted-out track rows"
    untouched = torch.ones(SLOTS, dtype=torch.bool)
    untouched[idx.long()] = False
    assert torch.equal(pool[untouched], pool0[untouched]), f"{name}: stray writes"

    # A -1 (padding) slot runs against slot 0 and leaves every other slot alone.
    idx_pad = idx.clone()
    idx_pad[-1] = -1
    pool = pool0.clone()
    fwd(q, k, v, g, beta, ssm_states=pool, cache_indices=idx_pad, cu_seqlens=cu)
    torch.cuda.synchronize()
    touched = torch.zeros(SLOTS, dtype=torch.bool)
    touched[idx_pad.clamp(min=0).long()] = True
    assert torch.equal(pool[~touched], pool0[~touched]), f"{name}: padding leaked"

    _, us = run_perftest(
        fwd,
        q,
        k,
        v,
        g,
        beta,
        ssm_states=pool0.clone(),
        cache_indices=idx,
        cu_seqlens=cu,
        num_iters=21,
        num_warmup=3,
    )
    return {
        "tokens": sum(seq_lens),
        "o_rel_l2": err_o,
        "state_rel_l2": err_state,
        "track_rel_l2": err_track,
        "us": us,
    }


def _run_contract_checks():
    fwd, supported = _load()
    q, k, v, g, beta, cu, pool, idx, _ = _make_inputs(SHAPES["qwen35_tp8_tail"])
    assert not supported(q.float(), k, v, g, beta, pool), "fp32 q accepted"
    assert not supported(q, k, v, g.bfloat16(), beta, pool), "bf16 g accepted"
    assert not supported(q, k, v, g, beta, pool.float()), "fp32 pool accepted"
    assert not supported(q, k, v, g, beta, pool.transpose(-1, -2)), "strided pool"
    assert not supported(q[..., :64], k[..., :64], v, g, beta, pool), "K=64"
    try:
        fwd(q.float(), k, v, g, beta, ssm_states=pool, cache_indices=idx, cu_seqlens=cu)
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported operands did not raise")


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
    aiter.logger.info("gdn_fused_prefill summary:\n%s", pd.DataFrame(rows).to_markdown())


if __name__ == "__main__":
    main()
