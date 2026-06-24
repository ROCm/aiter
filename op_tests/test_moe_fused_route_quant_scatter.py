# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness test for the fused MoE route+quant+scatter+preshuffle kernel.

Validates ``flydsl_moe_fused_route_quant_scatter`` against the reference
four-step pipeline it replaces:

    build_route_maps -> per_1x32_f4_quant -> scatter_copy_token ->
    scatter_preshuffle_scale

The within-expert row order is atomic-race (nondeterministic), so we neutralise
it by using the kernel's own ``topids_to_rows`` to scatter the reference
per-token quant into the same grouped rows, then compare payload + preshuffled
scale on the valid rows only (padding rows are intentionally uninitialised).

The kernel is arch-generic FlyDSL (buffer/atomic/shuffle ops only) so it runs on
gfx942 as well as gfx1250.

Run:  AITER_USE_SYSTEM_TRITON=1 python op_tests/test_moe_fused_route_quant_scatter.py
"""

import pytest
import torch

from aiter.ops.flydsl.moe_kernels import (
    flydsl_moe_fused_route_quant_scatter,
    flydsl_moe_fused_route_psum_quant_scatter,
    flydsl_moe_fused_quant_preshuffle,
)
from aiter.ops.flydsl.grouped_moe_gfx1250 import _grouped_a8w4_preshuffle_e8m0_scale
from aiter.ops.quant import per_1x32_f4_quant
from aiter.utility import dtypes, fp4_utils
from aiter.utility.mx_types import MxDtypeInt, MX_DEFAULT_ROUND_MODE
from flydsl.runtime.device import get_rocm_arch


def _fp8_variant():
    """(MxDtypeInt, torch fp8 dtype) matching the kernel's arch FP8 choice."""
    if str(get_rocm_arch()).startswith("gfx942"):
        return MxDtypeInt.FP8_E4M3_FNUZ, torch.float8_e4m3fnuz
    return MxDtypeInt.FP8_E4M3, torch.float8_e4m3fn


def _ref_token_quant(hidden, quant_mode):
    """Per-token (payload_u8 (T,Pb), e8m0_u8 (T,Ws)) reference matching the kernel."""
    token_num, model_dim = hidden.shape
    Ws = model_dim // 32
    if quant_mode == "fp4":
        # Torch reference MXFP4 quant (RoundUp, same contract as the kernel).
        a1q, a1s = per_1x32_f4_quant(hidden, quant_dtype=dtypes.fp4x2, shuffle=False)
        return (
            a1q.view(torch.uint8).contiguous().view(token_num, model_dim // 2),
            a1s.view(torch.uint8).contiguous().view(token_num, Ws),
        )
    # fp8: amax -> RoundUp e8m0 (arch FP8 dtype) -> divide -> cast to fp8.
    mxdt, fp8 = _fp8_variant()
    x = hidden.float().view(token_num, Ws, 32)
    amax = x.abs().amax(-1)
    e8 = fp4_utils.f32_to_mx_e8m0_scale(
        amax.reshape(-1), mode=MX_DEFAULT_ROUND_MODE, dtype=mxdt
    ).view(token_num, Ws)
    sc = fp4_utils.e8m0_to_f32(e8)
    y = (x / sc.unsqueeze(-1)).reshape(token_num, model_dim).to(fp8)
    return (
        y.view(torch.uint8).contiguous().view(token_num, model_dim),
        e8.view(torch.uint8).contiguous().view(token_num, Ws),
    )


def _ref_grouped(
    hidden, topk_ids, E, max_m, wmma_rep, topids_to_rows, quant_mode, scale_k_per_tile
):
    """Reference grouped payload + preshuffled e8m0 scale, scattered into the rows
    the kernel chose (topids_to_rows) so the atomic-race order matches.

    ``scale_k_per_tile`` is passed through to the reference preshuffle so the test
    validates the exact byte layout the GEMM consumes (tile_k//32), not just the
    helper's default of 4. (The in-kernel preshuffle is independent of this value;
    pinning it here guards against that assumption silently breaking.)"""
    token_num, topk = topk_ids.shape
    model_dim = hidden.shape[-1]
    Pb = model_dim if quant_mode == "fp8" else model_dim // 2
    Ws = model_dim // 32
    dev = hidden.device

    a1q_u8, a1s_u8 = _ref_token_quant(hidden, quant_mode)

    flat_rows = topids_to_rows.reshape(-1).to(torch.long)
    flat_tokens = torch.arange(token_num * topk, device=dev, dtype=torch.long) // topk

    ref_payload = torch.zeros(E * max_m, Pb, dtype=torch.uint8, device=dev)
    ref_payload[flat_rows] = a1q_u8[flat_tokens]

    ref_scale_rm = torch.zeros(E * max_m, Ws, dtype=torch.uint8, device=dev)
    ref_scale_rm[flat_rows] = a1s_u8[flat_tokens]
    ref_scale_pre = _grouped_a8w4_preshuffle_e8m0_scale(
        ref_scale_rm.view(E, max_m, Ws),
        warp_tile=wmma_rep * 16,
        scale_k_per_tile=scale_k_per_tile,
    )
    return ref_payload, ref_scale_pre.reshape(E, max_m // wmma_rep, Ws * wmma_rep)


def _valid_row_mask(masked_m, E, max_m, device):
    """Boolean (E*max_m,) mask of rows actually routed (slot < masked_m[e])."""
    slot = torch.arange(max_m, device=device).view(1, max_m)
    mask = slot < masked_m.view(E, 1)
    return mask.reshape(-1)


# scale_k_per_tile = tile_k // 32. The in-kernel preshuffle is independent of it,
# so we check both the helper default (4) and a production value (8, tile_k=256) to
# prove that and to validate the exact bytes the GEMM consumes.
@pytest.mark.parametrize("scale_k_per_tile", [4, 8])
@pytest.mark.parametrize("quant_mode", ["fp4", "fp8"])
@pytest.mark.parametrize("token_num", [1, 8, 64, 257])
@pytest.mark.parametrize("topk", [1, 2, 8])
@pytest.mark.parametrize("E", [4, 8])
@pytest.mark.parametrize("model_dim", [256, 512])
def test_fused_route_quant_scatter(
    scale_k_per_tile, quant_mode, token_num, topk, E, model_dim
):
    if not torch.cuda.is_available():
        pytest.skip("needs GPU")
    if topk > E:
        pytest.skip("topk must be <= E (router picks distinct experts per token)")
    if (model_dim // 32) % scale_k_per_tile != 0:
        pytest.skip("model_dim//32 must be divisible by scale_k_per_tile")
    torch.manual_seed(0)
    dev = "cuda"
    wmma_rep = 4
    warp_tile_m = wmma_rep * 16  # 64
    # static upper bound on per-expert rows (matches grouped path default):
    # each token routes at most one row per expert, so count[e] <= token_num.
    max_m = max(
        warp_tile_m, ((token_num + warp_tile_m - 1) // warp_tile_m) * warp_tile_m
    )

    hidden = torch.randn(token_num, model_dim, dtype=torch.bfloat16, device=dev)
    # Distinct experts per token (top-k of random scores), matching a real router
    # so per-expert counts stay within max_m.
    scores = torch.rand(token_num, E, device=dev)
    topk_ids = scores.topk(topk, dim=1).indices.to(torch.int32)

    ga1, gas, masked_m, ttr = flydsl_moe_fused_route_quant_scatter(
        hidden, topk_ids, E, max_m, wmma_rep=wmma_rep, quant_mode=quant_mode
    )

    # 1. masked_m == per-expert route counts.
    ref_counts = torch.bincount(topk_ids.reshape(-1).to(torch.long), minlength=E).to(
        torch.int32
    )
    assert torch.equal(
        masked_m.cpu(), ref_counts.cpu()
    ), f"masked_m {masked_m.tolist()} != bincount {ref_counts.tolist()}"

    # 2. topids_to_rows is a valid per-expert argsort: rows for expert e are a
    #    distinct subset of [e*max_m, e*max_m + counts[e]).
    ttr_flat = ttr.reshape(-1)
    experts = topk_ids.reshape(-1)
    for e in range(E):
        rows_e = ttr_flat[experts == e]
        n = ref_counts[e].item()
        assert rows_e.numel() == n
        assert torch.equal(
            torch.sort(rows_e).values,
            torch.arange(e * max_m, e * max_m + n, device=dev, dtype=ttr.dtype),
        ), f"expert {e}: rows {sorted(rows_e.tolist())} not contiguous slots"

    # 3. payload + preshuffled scale match the reference on valid rows. The ref
    #    preshuffle uses the production scale_k_per_tile (= tile_k//32).
    ref_payload, ref_scale = _ref_grouped(
        hidden, topk_ids, E, max_m, wmma_rep, ttr, quant_mode, scale_k_per_tile
    )
    Pb = model_dim if quant_mode == "fp8" else model_dim // 2
    Ws = model_dim // 32
    vmask = _valid_row_mask(masked_m, E, max_m, dev)

    got_payload = ga1.view(E * max_m, Pb)
    pay_eq = (got_payload == ref_payload)[vmask]
    pay_match = pay_eq.float().mean().item()
    # MXFP4 nibbles / fp8 bytes may differ by <=1 ULP between GPU emit and torch
    # ref at round thresholds (documented in emit_f32_to_e2m1); require high match.
    assert pay_match > 0.99, f"payload match {pay_match:.4f} too low"

    # scale: compare on valid rows. The preshuffle scatters a grouped row's Ws
    # bytes across the dst; rebuild a row-major view to mask by valid row.
    got_scale = gas.view(E, max_m // wmma_rep, Ws * wmma_rep)

    # Un-preshuffle both to row-major (E, max_m, Ws) for a row-masked compare.
    def _unshuffle(x):
        # inverse of _grouped_a8w4_preshuffle_e8m0_scale (scale_k_per_tile=4 ->
        # k_wmma_steps=1): (E, tile, 16, k_groups, 1, wmma_rep, 4) row-major axes.
        k_groups = Ws // 4
        g = x.view(E, max_m // (wmma_rep * 16), 16, k_groups, 1, wmma_rep, 4)
        g = g.permute(0, 1, 5, 2, 3, 4, 6).contiguous()
        return g.reshape(E * max_m, Ws)

    got_rm = _unshuffle(got_scale)
    ref_rm = _unshuffle(ref_scale)
    sc_eq = (got_rm == ref_rm)[vmask]
    sc_match = sc_eq.float().mean().item()
    assert sc_match > 0.99, f"scale match {sc_match:.4f} too low"

    print(
        f"OK {quant_mode} skpt={scale_k_per_tile} T={token_num} topk={topk} "
        f"E={E} md={model_dim} payload={pay_match:.4f} scale={sc_match:.4f}"
    )


# ---------------------------------------------------------------------------
# Stage2 fused (grouped) quant + scale-preshuffle. Input is already grouped
# row-major (E, max_m, feat_dim); the kernel quantizes all rows and writes the
# preshuffled scale -- no routing. Compared against the same per-token MX quant
# reference + the torch preshuffle.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("scale_k_per_tile", [4, 8])
@pytest.mark.parametrize("quant_mode", ["fp4", "fp8"])
@pytest.mark.parametrize("max_m", [64, 128])
@pytest.mark.parametrize("E", [4, 8])
@pytest.mark.parametrize("feat_dim", [256, 512])
def test_fused_quant_preshuffle(
    masked, scale_k_per_tile, quant_mode, max_m, E, feat_dim
):
    if not torch.cuda.is_available():
        pytest.skip("needs GPU")
    if (feat_dim // 32) % scale_k_per_tile != 0:
        pytest.skip("feat_dim//32 must be divisible by scale_k_per_tile")
    torch.manual_seed(0)
    dev = "cuda"
    wmma_rep = 4
    if max_m % (wmma_rep * 16) != 0:
        pytest.skip("max_m must be a multiple of wmma_rep*16")

    grouped_in = torch.randn(E, max_m, feat_dim, dtype=torch.bfloat16, device=dev)

    # When `masked`, pass a per-expert valid count so the kernel skips padding
    # rows (slot >= masked_m[e]); those rows stay uninitialised and are excluded
    # from the compare via the valid-row mask. Otherwise every row is quantized.
    masked_m = None
    if masked:
        masked_m = torch.randint(1, max_m + 1, (E,), dtype=torch.int32, device=dev)

    payload, scale_pre = flydsl_moe_fused_quant_preshuffle(
        grouped_in,
        E,
        max_m,
        wmma_rep=wmma_rep,
        quant_mode=quant_mode,
        masked_m=masked_m,
    )

    Pb = feat_dim if quant_mode == "fp8" else feat_dim // 2
    Ws = feat_dim // 32

    # Reference: per-row MX quant (matches the kernel contract) + torch preshuffle.
    ref_pay_u8, ref_scale_u8 = _ref_token_quant(
        grouped_in.reshape(E * max_m, feat_dim), quant_mode
    )
    ref_payload = ref_pay_u8.view(E, max_m, Pb)
    ref_scale_pre = _grouped_a8w4_preshuffle_e8m0_scale(
        ref_scale_u8.view(E, max_m, Ws),
        warp_tile=wmma_rep * 16,
        scale_k_per_tile=scale_k_per_tile,
    ).reshape(E, max_m // wmma_rep, Ws * wmma_rep)

    if masked:
        # Compare payload only on valid rows (padding rows are skipped/untouched).
        vmask = _valid_row_mask(masked_m, E, max_m, dev)
        got_pay = payload.view(E * max_m, Pb)
        pay_match = (got_pay == ref_payload.view(E * max_m, Pb))[vmask].float().mean()
        pay_match = pay_match.item()

        # Scale: un-preshuffle both to row-major (E*max_m, Ws) then row-mask.
        def _unshuffle(x):
            k_groups = Ws // 4
            g = x.view(E, max_m // (wmma_rep * 16), 16, k_groups, 1, wmma_rep, 4)
            g = g.permute(0, 1, 5, 2, 3, 4, 6).contiguous()
            return g.reshape(E * max_m, Ws)

        got_rm = _unshuffle(scale_pre.view(E, max_m // wmma_rep, Ws * wmma_rep))
        ref_rm = _unshuffle(ref_scale_pre)
        sc_match = (got_rm == ref_rm)[vmask].float().mean().item()
    else:
        # All rows are quantized.
        pay_match = (payload == ref_payload).float().mean().item()
        sc_match = (scale_pre == ref_scale_pre).float().mean().item()

    assert pay_match > 0.99, f"payload match {pay_match:.4f} too low"
    assert sc_match > 0.99, f"scale match {sc_match:.4f} too low"

    print(
        f"OK fused_quant_preshuffle {quant_mode} masked={masked} "
        f"skpt={scale_k_per_tile} E={E} max_m={max_m} fd={feat_dim} "
        f"payload={pay_match:.4f} scale={sc_match:.4f}"
    )


@pytest.mark.parametrize("quant_mode", ["fp4", "fp8"])
@pytest.mark.parametrize("topk", [1, 2, 4, 8])
def test_fused_quant_preshuffle_route_ksplit(quant_mode, topk):
    if not torch.cuda.is_available():
        pytest.skip("needs GPU")
    torch.manual_seed(0)
    dev = "cuda"
    E = 8
    max_m = 64
    feat_dim = 256
    wmma_rep = 4

    grouped_in = torch.randn(E, max_m, feat_dim, dtype=torch.bfloat16, device=dev)
    # One routed row per expert prefix; these are the only rows the route-indexed
    # K-split kernel writes, matching the token=1 stage2 production path.
    topids_to_rows = torch.arange(topk, dtype=torch.int32, device=dev) * max_m
    payload, scale_pre = flydsl_moe_fused_quant_preshuffle(
        grouped_in,
        E,
        max_m,
        wmma_rep=wmma_rep,
        quant_mode=quant_mode,
        topids_to_rows=topids_to_rows,
    )

    Pb = feat_dim if quant_mode == "fp8" else feat_dim // 2
    Ws = feat_dim // 32
    ref_pay_u8, ref_scale_u8 = _ref_token_quant(
        grouped_in.reshape(E * max_m, feat_dim), quant_mode
    )
    ref_payload = ref_pay_u8.view(E * max_m, Pb)
    ref_scale_pre = _grouped_a8w4_preshuffle_e8m0_scale(
        ref_scale_u8.view(E, max_m, Ws),
        warp_tile=wmma_rep * 16,
        scale_k_per_tile=4,
    ).reshape(E, max_m // wmma_rep, Ws * wmma_rep)

    rows = topids_to_rows.to(torch.long)
    pay_match = (
        (payload.view(E * max_m, Pb)[rows] == ref_payload[rows]).float().mean().item()
    )

    def _unshuffle(x):
        k_groups = Ws // 4
        g = x.view(E, max_m // (wmma_rep * 16), 16, k_groups, 1, wmma_rep, 4)
        g = g.permute(0, 1, 5, 2, 3, 4, 6).contiguous()
        return g.reshape(E * max_m, Ws)

    got_rm = _unshuffle(scale_pre.view(E, max_m // wmma_rep, Ws * wmma_rep))
    ref_rm = _unshuffle(ref_scale_pre)
    sc_match = (got_rm[rows] == ref_rm[rows]).float().mean().item()
    assert pay_match > 0.99, f"payload match {pay_match:.4f} too low"
    assert sc_match > 0.99, f"scale match {sc_match:.4f} too low"


# ---------------------------------------------------------------------------
# Fully-fused contiguous-M stage1 prep: count + tile-aligned prefix sum +
# route + quant + scatter in ONE persistent kernel launch (global-atomic
# barrier between phases). Validates the in-kernel count/starts/psum against the
# torch reference and the payload/scale against the same scatter reference, into
# the single (1, contiguous_m) DeepGEMM buffer the kernel chose.
# ---------------------------------------------------------------------------
def _ref_contiguous_psum(counts, tile_m):
    """Reference tile-aligned exclusive prefix sum (matches moe_contiguous_psum)."""
    aligned = ((counts + tile_m - 1) // tile_m) * tile_m
    starts = torch.zeros_like(aligned)
    if aligned.numel() > 1:
        starts[1:] = torch.cumsum(aligned, 0)[:-1]
    psum = starts + counts
    return starts, psum


@pytest.mark.parametrize("scale_k_per_tile", [4, 8])
@pytest.mark.parametrize("quant_mode", ["fp4", "fp8"])
@pytest.mark.parametrize("token_num", [8, 64, 257])
@pytest.mark.parametrize("topk", [1, 2, 8])
@pytest.mark.parametrize("E", [4, 8])
@pytest.mark.parametrize("model_dim", [256, 512])
@pytest.mark.parametrize("tile_m", [64, 128])
def test_fused_route_psum_quant_scatter(
    scale_k_per_tile, quant_mode, token_num, topk, E, model_dim, tile_m
):
    if not torch.cuda.is_available():
        pytest.skip("needs GPU")
    if topk > E:
        pytest.skip("topk must be <= E (router picks distinct experts per token)")
    if (model_dim // 32) % scale_k_per_tile != 0:
        pytest.skip("model_dim//32 must be divisible by scale_k_per_tile")
    torch.manual_seed(0)
    dev = "cuda"
    wmma_rep = 4
    rows_per_tile = wmma_rep * 16  # 64
    if tile_m % rows_per_tile != 0:
        pytest.skip("tile_m must be a multiple of wmma_rep*16")

    # Static contiguous_m upper bound, identical to the grouped orchestration.
    ub = token_num * topk + E * (tile_m - 1)
    contiguous_m = max(tile_m, ((ub + tile_m - 1) // tile_m) * tile_m)

    hidden = torch.randn(token_num, model_dim, dtype=torch.bfloat16, device=dev)
    scores = torch.rand(token_num, E, device=dev)
    topk_ids = scores.topk(topk, dim=1).indices.to(torch.int32)

    ga1, gas, masked_m, ttr, starts, psum = flydsl_moe_fused_route_psum_quant_scatter(
        hidden,
        topk_ids,
        E,
        tile_m,
        contiguous_m,
        wmma_rep=wmma_rep,
        quant_mode=quant_mode,
    )

    # 1. count == per-expert route counts.
    ref_counts = torch.bincount(topk_ids.reshape(-1).to(torch.long), minlength=E).to(
        torch.int32
    )
    assert torch.equal(
        masked_m.cpu(), ref_counts.cpu()
    ), f"masked_m {masked_m.tolist()} != bincount {ref_counts.tolist()}"

    # 2. starts/psum == tile-aligned prefix sum reference.
    ref_starts, ref_psum = _ref_contiguous_psum(ref_counts, tile_m)
    assert torch.equal(
        starts.cpu(), ref_starts.cpu()
    ), f"starts {starts.tolist()} != ref {ref_starts.tolist()}"
    assert torch.equal(
        psum.cpu(), ref_psum.cpu()
    ), f"psum {psum.tolist()} != ref {ref_psum.tolist()}"

    # 3. topids_to_rows: rows for expert e are a distinct subset of
    #    [starts[e], starts[e] + counts[e]) in the single contiguous buffer.
    ttr_flat = ttr.reshape(-1)
    experts = topk_ids.reshape(-1)
    for e in range(E):
        rows_e = ttr_flat[experts == e]
        n = ref_counts[e].item()
        assert rows_e.numel() == n
        s = ref_starts[e].item()
        assert torch.equal(
            torch.sort(rows_e).values,
            torch.arange(s, s + n, device=dev, dtype=ttr.dtype),
        ), f"expert {e}: rows {sorted(rows_e.tolist())} not contiguous from {s}"

    # 4. payload + preshuffled scale match the reference scattered into the rows
    #    the kernel chose (treat the (1, contiguous_m) buffer as E=1 / max_m=cm).
    Pb = model_dim if quant_mode == "fp8" else model_dim // 2
    Ws = model_dim // 32
    a1q_u8, a1s_u8 = _ref_token_quant(hidden, quant_mode)
    flat_rows = ttr_flat.to(torch.long)
    flat_tokens = torch.arange(token_num * topk, device=dev, dtype=torch.long) // topk

    ref_payload = torch.zeros(contiguous_m, Pb, dtype=torch.uint8, device=dev)
    ref_payload[flat_rows] = a1q_u8[flat_tokens]
    ref_scale_rm = torch.zeros(contiguous_m, Ws, dtype=torch.uint8, device=dev)
    ref_scale_rm[flat_rows] = a1s_u8[flat_tokens]
    ref_scale = _grouped_a8w4_preshuffle_e8m0_scale(
        ref_scale_rm.view(1, contiguous_m, Ws),
        warp_tile=wmma_rep * 16,
        scale_k_per_tile=scale_k_per_tile,
    ).reshape(1, contiguous_m // wmma_rep, Ws * wmma_rep)

    # Valid-row mask over the contiguous buffer (rows actually routed to).
    vmask = torch.zeros(contiguous_m, dtype=torch.bool, device=dev)
    vmask[flat_rows] = True

    got_payload = ga1.view(contiguous_m, Pb)
    pay_eq = (got_payload == ref_payload)[vmask]
    pay_match = pay_eq.float().mean().item()
    assert pay_match > 0.99, f"payload match {pay_match:.4f} too low"

    def _unshuffle(x):
        # inverse of _grouped_a8w4_preshuffle_e8m0_scale for E=1 buffer.
        k_groups = Ws // 4
        g = x.view(1, contiguous_m // (wmma_rep * 16), 16, k_groups, 1, wmma_rep, 4)
        g = g.permute(0, 1, 5, 2, 3, 4, 6).contiguous()
        return g.reshape(contiguous_m, Ws)

    got_rm = _unshuffle(gas.view(1, contiguous_m // wmma_rep, Ws * wmma_rep))
    ref_rm = _unshuffle(ref_scale)
    sc_eq = (got_rm == ref_rm)[vmask]
    sc_match = sc_eq.float().mean().item()
    assert sc_match > 0.99, f"scale match {sc_match:.4f} too low"

    print(
        f"OK psum-fused {quant_mode} skpt={scale_k_per_tile} T={token_num} "
        f"topk={topk} E={E} md={model_dim} tile_m={tile_m} cm={contiguous_m} "
        f"payload={pay_match:.4f} scale={sc_match:.4f}"
    )


if __name__ == "__main__":
    for skpt in (4, 8):
        for qm in ("fp4", "fp8"):
            for tn in (1, 8, 64, 257):
                for tk in (1, 2, 8):
                    for E in (4, 8):
                        if tk > E:
                            continue
                        for md in (256, 512):
                            if (md // 32) % skpt != 0:
                                continue
                            test_fused_route_quant_scatter(skpt, qm, tn, tk, E, md)
    for skpt in (4, 8):
        for qm in ("fp4", "fp8"):
            for tn in (8, 64, 257):
                for tk in (1, 2, 8):
                    for E in (4, 8):
                        if tk > E:
                            continue
                        for md in (256, 512):
                            if (md // 32) % skpt != 0:
                                continue
                            for tile_m in (64, 128):
                                test_fused_route_psum_quant_scatter(
                                    skpt, qm, tn, tk, E, md, tile_m
                                )
    for masked in (False, True):
        for skpt in (4, 8):
            for qm in ("fp4", "fp8"):
                for E in (4, 8):
                    for mm in (64, 128):
                        for fd in (256, 512):
                            if (fd // 32) % skpt != 0:
                                continue
                            test_fused_quant_preshuffle(masked, skpt, qm, mm, E, fd)
    print("all cases passed")
