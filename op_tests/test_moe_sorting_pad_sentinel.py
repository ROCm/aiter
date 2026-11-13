# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Padding entries of ``sorted_token_ids`` under a device ``num_local_tokens``.

Every MoE consumer of ``sorted_token_ids`` (the CK, CK-Tile, opus and FlyDSL GEMMs, the
sort-quant kernels) treats a token id >= its row count M as padding. With
``num_local_tokens`` = n < M the sorting kernels wrote the pad id as n, so every pad entry
passed those checks: the CK stage 2 computed a full GEMM row for each and accumulated
all of them with atomics into one output row (up to 28x slower), and a count of M - 1
made the pad row index run one row past the end of the stage-1 output. The pad id is
now the capacity M, as it already is without a local count.
"""

import numpy as np
import pytest
import torch

import aiter
import aiter.fused_moe as fused_moe_module
from aiter import dtypes
from aiter.fused_moe import AUX_SORT_OPUS, fused_topk, moe_sorting
from aiter.jit.utils.chip_info import get_gfx

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not torch.version.hip
    or get_gfx() not in ("gfx942", "gfx950"),
    reason="Needs gfx942 or gfx950",
)

D = 256
CASES = [
    # (M, experts, topk, local tokens): single-kernel and multi-phase sorts
    (64, 128, 8, 48),
    (64, 128, 8, 63),
    (64, 128, 8, 64),
    (64, 128, 8, 1),
    (1024, 128, 8, 128),
    (1024, 128, 8, 1023),
    (4096, 128, 8, 512),
    (4096, 257, 9, 2048),
]


def _routes(M, E, topk):
    torch.manual_seed(M * 131 + E)
    x = torch.randn(M, 64, dtype=dtypes.bf16, device="cuda")
    weights, ids = fused_topk(
        x, torch.randn(M, E, dtype=dtypes.bf16, device="cuda"), topk, True
    )
    return ids, weights


def _check(sorted_ids, num_valid, ids, M, topk, local):
    rows = int(num_valid[0])
    entries = sorted_ids[:rows].cpu().numpy().astype(np.int64)
    token, slot = entries & 0xFFFFFF, entries >> 24
    real = token < local
    in_gap = (token >= local) & (token < M)
    assert not in_gap.any(), (
        f"{int(in_gap.sum())} padding entries have an id in [{local}, {M}) "
        f"(first {int(token[in_gap][0])}): they pass every consumer's `id < M` check"
    )
    assert (token[~real] == M).all()
    assert (slot[real] < topk).all()
    expected = sorted(
        (int(t) * topk + int(s)) for t in range(local) for s in range(topk)
    )
    got = sorted(int(t) * topk + int(s) for t, s in zip(token[real], slot[real]))
    assert (
        got == expected
    ), "the real routes differ from the first `local` tokens' routes"
    assert int(num_valid[1]) == local


@pytest.mark.parametrize("dispatch_policy", [0, 1])
@pytest.mark.parametrize("M,E,topk,local", CASES)
def test_opus_sort_pads_with_the_capacity(M, E, topk, local, dispatch_policy):
    ids, weights = _routes(M, E, topk)
    nlt = torch.tensor([local], dtype=torch.int32, device="cuda")
    sorted_ids, _, _, num_valid, _ = moe_sorting(
        ids,
        weights,
        E,
        D,
        dtypes.bf16,
        32,
        None,
        nlt,
        dispatch_policy,
    )
    torch.cuda.synchronize()
    _check(sorted_ids, num_valid, ids, M, topk, local)


@pytest.mark.parametrize("M,E,topk,local", CASES[:6])
def test_opus_sort_m_indices_pad_with_the_capacity(M, E, topk, local):
    ids, weights = _routes(M, E, topk)
    nlt = torch.tensor([local], dtype=torch.int32, device="cuda")
    out = moe_sorting(
        ids,
        weights,
        E,
        D,
        dtypes.bf16,
        32,
        None,
        nlt,
        0,
        False,
        True,
        fused_moe_module.AsmMoeMode.SORTED,
        AUX_SORT_OPUS,
    )
    torch.cuda.synchronize()
    sorted_ids, num_valid, m_indices = out[0], out[3], out[5]
    rows = int(num_valid[0])
    entries = sorted_ids[:rows].cpu().numpy().astype(np.int64) & 0xFFFFFF
    m_idx = m_indices[:rows].cpu().numpy().astype(np.int64) & 0xFFFFFF
    pad = entries >= local
    assert (m_idx[pad] == M).all(), "m_indices pad must also be the capacity"
    assert (m_idx[~pad] == entries[~pad]).all()


@pytest.mark.parametrize("M,E,topk,local", CASES[:6])
def test_ck_tile_sort_pads_with_the_capacity(monkeypatch, M, E, topk, local):
    monkeypatch.setattr(fused_moe_module, "_USE_CK_MOE_SORTING", True)
    ids, weights = _routes(M, E, topk)
    nlt = torch.tensor([local], dtype=torch.int32, device="cuda")
    sorted_ids, _, _, num_valid, _ = moe_sorting(
        ids, weights, E, D, dtypes.bf16, 32, None, nlt, 0
    )
    torch.cuda.synchronize()
    _check(sorted_ids, num_valid, ids, M, topk, local)


@pytest.mark.skipif(get_gfx() != "gfx950", reason="fp4 convert needs gfx950")
@pytest.mark.parametrize("M,local", [(64, 48), (128, 127), (128, 128), (16, 5)])
def test_fused_sort_quant_pads_with_the_capacity(M, local):
    E, topk, H = 512, 10, 4096
    ids, weights = _routes(M, E, topk)
    hidden = torch.randn(M, H, dtype=dtypes.bf16, device="cuda")
    route_count = M * topk
    active = min(E, route_count)
    max_sorted = (route_count + active * 31 + 31) // 32 * 32
    sorted_ids = torch.empty(max_sorted, dtype=dtypes.i32, device="cuda")
    sorted_w = torch.empty(max_sorted, dtype=dtypes.fp32, device="cuda")
    sorted_eids = torch.empty(max_sorted // 32, dtype=dtypes.i32, device="cuda")
    num_valid = torch.empty(2, dtype=dtypes.i32, device="cuda")
    moe_buf = torch.empty(M, H, dtype=dtypes.bf16, device="cuda")
    quant = torch.empty(M, H // 2, dtype=dtypes.fp4x2, device="cuda")
    scale = torch.empty(M, H // 32, dtype=dtypes.fp8_e8m0, device="cuda")
    nlt = torch.tensor([local], dtype=torch.int32, device="cuda")
    aiter.mxfp4_moe_sort_quant_fwd(
        hidden,
        ids,
        weights,
        sorted_ids,
        sorted_w,
        sorted_eids,
        num_valid,
        moe_buf,
        quant,
        scale,
        E,
        nlt,
    )
    torch.cuda.synchronize()
    _check(sorted_ids, num_valid, ids, M, topk, local)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
