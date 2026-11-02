# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Token-multidest prequantized MoE preshuffle against the route-indexed kernel.

Both scatter an fp4 EP-dispatch payload and its e8m0 row into the grouped GEMM1
layout. The payload must match byte for byte; the scales must match on every
mapped row (padding rows are scratch in both paths).
"""

import os

import pytest
import torch

from aiter.ops.flydsl.moe_kernels import flydsl_moe_fused_quant_preshuffle

E_GLOBAL, E_LOCAL, TOPK, D = 256, 64, 8, 2048
PB, WS = D // 2, D // 32


def _align_up(x, a):
    return (x + a - 1) // a * a


def _make_case(n_static, n_valid, align, src_ws):
    g = torch.Generator(device="cpu").manual_seed(0)
    # Received tokens have at least one route to this rank's experts [0, E_LOCAL).
    ids = []
    while len(ids) < n_valid:
        t = torch.rand(4 * n_valid, E_GLOBAL, generator=g).argsort(dim=1)[:, :TOPK]
        ids.extend(t[(t < E_LOCAL).any(dim=1)])
    ids = torch.stack(ids[:n_valid]).reshape(-1)
    local = ids < E_LOCAL
    experts = ids[local]
    counts = torch.bincount(experts, minlength=E_LOCAL)
    starts = torch.zeros(E_LOCAL, dtype=torch.int64)
    starts[1:] = torch.cumsum(_align_up(counts, align), 0)[:-1]
    slots = torch.zeros_like(experts)
    for e in range(E_LOCAL):
        m = experts == e
        slots[m] = torch.arange(int(m.sum()))
    rows = torch.full((n_static * TOPK,), -1, dtype=torch.int32)
    valid_rows = torch.full_like(ids, -1)
    valid_rows[local] = starts[experts] + slots
    rows[: n_valid * TOPK] = valid_rows.to(torch.int32)
    # The dead tail holds garbage rows the kernels must skip.
    rows[n_valid * TOPK :] = torch.randint(
        0, 1000, ((n_static - n_valid) * TOPK,), generator=g, dtype=torch.int32
    )
    contiguous_m = max(align, _align_up(n_static * TOPK + E_LOCAL * align - TOPK, align))
    hidden = torch.randint(0, 256, (n_static, PB), dtype=torch.uint8, generator=g)
    src_scale = torch.randint(0, 256, (n_static, src_ws), dtype=torch.uint8, generator=g)
    nvr = torch.tensor([n_valid * TOPK], dtype=torch.int32)
    return (
        hidden.cuda(),
        src_scale.cuda(),
        rows.cuda(),
        nvr.cuda(),
        contiguous_m,
    )


def _run(case, wmma_rep, row_major, multidest, monkeypatch):
    hidden, src_scale, rows, nvr, cm = case
    monkeypatch.setenv(
        "AITER_FLYDSL_TOKEN_MULTIDEST_PREQUANT", "1" if multidest else "0"
    )
    payload = torch.zeros((1, cm, PB), dtype=torch.uint8, device="cuda")
    if row_major:
        scale = torch.zeros((cm, WS), dtype=torch.uint8, device="cuda")
    else:
        scale = torch.zeros(
            (1, cm // wmma_rep, WS * wmma_rep), dtype=torch.uint8, device="cuda"
        )
    flydsl_moe_fused_quant_preshuffle(
        hidden.reshape(1, -1, PB),
        1,
        cm,
        wmma_rep=wmma_rep,
        quant_mode="fp4",
        topids_to_rows=rows,
        source_topk=TOPK,
        num_valid_routes=nvr,
        prequantized_scale=src_scale,
        out_payload=payload,
        out_scale=scale,
        row_major_scale=row_major,
    )
    return payload, scale


def _scale_rows(scale, wmma_rep, row_major):
    if row_major:
        return scale.view(-1, WS)
    t = scale.reshape(-1, WS // 4, wmma_rep, 16, 4).permute(0, 2, 3, 1, 4)
    return t.reshape(-1, WS)


@pytest.mark.parametrize(
    "n_static, n_valid, align, wmma_rep, row_major, src_ws",
    [
        # prefill: the two-pass scale path (>= 4096 received tokens)
        (8192, 7000, 64, 4, False, 128),
        (8192, 7000, 64, 4, True, 128),
        # decode: one pass
        (1024, 870, 256, 8, False, 128),
        (1024, 870, 256, 8, True, 64),
        (1024, 1024, 64, 4, False, 64),
    ],
)
def test_token_multidest_prequant_matches_route(
    n_static, n_valid, align, wmma_rep, row_major, src_ws, monkeypatch
):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    if "gfx1250" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("token multidest prequant is gfx1250 only")
    case = _make_case(n_static, n_valid, align, src_ws)
    ref = _run(case, wmma_rep, row_major, False, monkeypatch)
    new = _run(case, wmma_rep, row_major, True, monkeypatch)
    torch.testing.assert_close(new[0], ref[0], rtol=0, atol=0)
    mapped = case[2][: n_valid * TOPK]
    mapped = mapped[mapped >= 0].long()
    torch.testing.assert_close(
        _scale_rows(new[1], wmma_rep, row_major)[mapped],
        _scale_rows(ref[1], wmma_rep, row_major)[mapped],
        rtol=0,
        atol=0,
    )


if __name__ == "__main__":
    os.environ.setdefault("AITER_LOG_LEVEL", "WARNING")
    pytest.main([__file__, "-q"])
