# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL DCP decode TopK merge tests."""

import argparse
import itertools

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]

# The kernel hard-requires wave64 (see dcp_topk_merge._validate), so on a wave32
# target every test here would raise ValueError rather than skip.
pytestmark = pytest.mark.skipif(
    get_gfx() not in SUPPORTED_GFX,
    reason=f"FlyDSL DCP TopK merge needs one of {SUPPORTED_GFX}",
)

# Case shapes the sweep and the pytest surface share. Each one bends a different
# part of the op; see make_case for how the inputs are bent.
CASE_MODES = ["random", "tie", "short", "starved", "overtake"]


def ref_dcp_merge(
    gathered_scores: torch.Tensor,  # fp32 [rows, W*k_loc]
    local_idx: torch.Tensor,  # i32  [rows, k_loc]
    block_table: torch.Tensor,  # i32  [rows, max_blocks]
    dcp_rank: int,
    k_loc: int,
    topk_tokens: int,
    page_size: int,
):
    """Oracle: global threshold select, keep own plane, map to physical slots.

    Tie rule mirrors the kernel: among candidates whose score equals the
    threshold exactly, the ones with the smallest flat candidate position win.
    torch.topk on a descending sort of (score, -position) gives that ordering.
    """
    rows = gathered_scores.shape[0]
    owned_slots = []
    counts = torch.zeros(rows, dtype=torch.int32, device=gathered_scores.device)
    for r in range(rows):
        sc = gathered_scores[r]
        finite = torch.isfinite(sc)
        n_valid = int(finite.sum())
        take = min(topk_tokens, n_valid)
        # Sort by score desc, ties broken by smaller flat position.
        order = torch.argsort(
            torch.where(finite, sc, torch.full_like(sc, -float("inf"))),
            descending=True,
            stable=True,
        )
        winners = order[:take]
        # Keep only winners from this rank's plane.
        mine = winners[(winners // k_loc) == dcp_rank]
        # Plane position -> local KV index -> physical slot.
        local_pos = (mine % k_loc).to(torch.int64)
        j = local_idx[r].to(torch.int64)[local_pos]
        assert bool((j >= 0).all()), "padding leaked into the winner set"
        slot = block_table[r].to(torch.int64)[j // page_size] * page_size + (
            j % page_size
        )
        # Compact in increasing flat-candidate order (deterministic).
        slot = slot[torch.argsort(mine, stable=True)]
        owned_slots.append(slot.to(torch.int32))
        counts[r] = slot.numel()
    return owned_slots, counts


def make_case(
    rows,
    world,
    k_loc,
    topk_tokens,
    page_size,
    max_blocks,
    n_local=None,
    tie_heavy=False,
    seed=0,
):
    """Build a self-consistent (scores, local_idx, block_table) triple.

    Every rank's plane gets k_loc candidates. Pass n_local < k_loc to emulate a
    short context: the tail of local_idx becomes -1, exactly as
    top_k_per_row_decode pads it the same way. Callers that want
    the matching -inf scores must starve the corresponding plane themselves --
    see make_mode_case("short"). The separation is intentional: make_case is
    responsible only for index/block-table consistency, not for score semantics.

    Multi-block slot coverage note: when n_local < page_size, every generated
    local index j satisfies j // page_size == 0, so only block_table[r, 0] is
    ever exercised. Callers writing short-context cases should pick
    n_local >= page_size if they want coverage of more than one block.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    n_cand = world * k_loc
    if tie_heavy:
        scores = torch.randint(
            -4, 5, (rows, n_cand), generator=g, dtype=torch.int32, device="cuda"
        ).float()
    else:
        scores = torch.randn(rows, n_cand, generator=g, device="cuda")
    fill = k_loc if n_local is None else min(k_loc, n_local)
    local_idx = torch.empty(rows, k_loc, dtype=torch.int32, device="cuda")
    for r in range(rows):
        # Local KV indices must stay in range for the slot formula: j // page_size
        # indexes block_table, so keep j < max_blocks * page_size.
        hi = min(max_blocks * page_size, max(fill, 1))
        perm = torch.randperm(hi, generator=g, device="cuda")[:fill]
        local_idx[r, :fill] = perm.to(torch.int32)
        local_idx[r, fill:] = -1
    block_table = torch.randint(
        0, 1000, (rows, max_blocks), generator=g, dtype=torch.int32, device="cuda"
    )
    return scores, local_idx, block_table


def make_mode_case(mode, rows, world, k_loc, topk, page, rank, seed):
    """Bend a base case into the shape `mode` names.

    Returns (scores, local_idx, block_table, expect), where `expect` carries the
    mode-specific invariant the caller must additionally assert -- the oracle
    already covers "the slots are right", these cover "the op did not take a
    degenerate path to get there".
    """
    max_blocks = 4096
    if mode == "tie":
        # Integer scores over a 9-value range: most candidates sit exactly ON
        # the threshold, so the tie rule is what decides the partition.
        s, li, bt = make_case(
            rows, world, k_loc, topk, page, max_blocks, tie_heavy=True, seed=seed
        )
        return s, li, bt, {}
    if mode == "short":
        # Short context: only n_real live candidates per plane, rest padded.
        # n_real > page so local indices span >1 block-table entry.
        n_real = max(page + 4, k_loc // 8)
        n_real = min(n_real, k_loc)
        s, li, bt = make_case(
            rows, world, k_loc, topk, page, max_blocks, n_local=n_real, seed=seed
        )
        li[:, n_real:] = -1
        # -inf is the contract: padding carries no liveness flag of its own in
        # the gathered plane, it has to lose on score alone.
        #
        # Starve EVERY plane's tail, not just this rank's. local_idx is the same
        # shape on all ranks, so padding a single plane's scores leaves the other
        # W-1 ranks holding -1 indices with live scores -- their padding wins and
        # they emit garbage, which the cross-rank partition check then trips on.
        for w in range(world):
            s[:, w * k_loc + n_real : (w + 1) * k_loc] = -float("inf")
        return s, li, bt, {"max_counts": n_real}
    if mode == "starved":
        # This rank's whole plane is pushed below every other rank's, so it wins
        # only what the other W-1 planes cannot absorb. That is 0 in the usual
        # case; it is nonzero when topk exceeds the other ranks' total capacity,
        # since then even -1e30 candidates are needed to fill the top-k.
        s, li, bt = make_case(rows, world, k_loc, topk, page, max_blocks, seed=seed)
        s[:, rank * k_loc : (rank + 1) * k_loc] = -1e30
        spill = max(0, topk - (world - 1) * k_loc)
        return s, li, bt, {"exact_counts": spill}
    if mode == "overtake":
        # topk_tokens >= n_cand degenerates to "take everything", not garbage:
        # every rank ends up owning its entire plane. Only assert that when the
        # caller's topk actually reaches n_cand -- the production sweep ships
        # topk=2048 against n_cand=16384, which is selective, not degenerate.
        s, li, bt = make_case(rows, world, k_loc, topk, page, max_blocks, seed=seed)
        expect = {"exact_counts": k_loc} if topk >= world * k_loc else {}
        return s, li, bt, expect
    if mode == "random":
        return (
            *make_case(rows, world, k_loc, topk, page, max_blocks, seed=seed),
            {},
        )
    raise ValueError(f"unknown case mode: {mode}")


def run_merge(scores, local_idx, bt, rank, world, topk, page):
    """Call the op with freshly allocated outputs.

    Returns (indices, indptr, counts, staging). `staging` is the kernel's
    per-row scratch of owned slots before packing -- the tests that check
    ownership per rank read it directly, since packing collapses the rows.
    """
    rows = scores.shape[0]
    k_loc = scores.shape[1] // world
    indices = torch.zeros(rows * max(topk, k_loc), dtype=torch.int32)
    indptr = torch.zeros(rows + 1, dtype=torch.int32)
    staging = torch.empty(rows, k_loc, dtype=torch.int32)
    counts = torch.zeros(rows, dtype=torch.int32)
    aiter.flydsl_dcp_topk_merge(
        scores,
        local_idx,
        bt,
        indices,
        indptr,
        counts,
        staging,
        rank,
        world,
        topk,
        page,
    )
    torch.cuda.synchronize()
    return indices, indptr, counts, staging


@benchmark()  # call args become the table's left-hand columns
def test_dcp_topk_merge(rows, world, k_loc, topk, page, mode):
    """One rank's merge: global-threshold select -> owned, packed KV slots.

    Single-kernel-vs-reference shape: there is no second kernel to race. The
    sequence this op replaces lives in ATOM (a merge + a Triton filter), so it
    is not importable here as a candidate -- `ref_dcp_merge` is the oracle.

    Correctness is checked across ALL W ranks (the partition property is only
    visible that way); only the middle rank is timed.
    """
    rank = world // 2  # a middle plane: exercises the prior-equal sweep
    scores, local_idx, bt, expect = make_mode_case(
        mode, rows, world, k_loc, topk, page, rank, seed=42
    )
    ref_slots, ref_counts = ref_dcp_merge(
        scores, local_idx, bt, rank, k_loc, topk, page
    )

    # Caller-owned buffers, as production allocates them (see ATOM's
    # dcp_decode_candidate_exchange_fused): the op allocates no device scratch.
    staging = torch.empty(rows, k_loc, dtype=torch.int32)
    counts = torch.zeros(rows, dtype=torch.int32)
    indptr = torch.zeros(rows + 1, dtype=torch.int32)
    indices = torch.zeros(rows * max(topk, k_loc), dtype=torch.int32)

    candidates = {
        "flydsl": lambda: aiter.flydsl_dcp_topk_merge(
            scores,
            local_idx,
            bt,
            indices,
            indptr,
            counts,
            staging,
            rank,
            world,
            topk,
            page,
        ),
    }

    n_cand = world * k_loc
    # Memory-side op: the radix select re-reads the candidate plane, and the
    # emit walks this rank's own plane plus its block table.
    nbytes = (
        rows * n_cand * scores.element_size()  # gathered scores
        + rows * k_loc * local_idx.element_size()  # local_idx
        + bt.numel() * bt.element_size()  # block table
        + rows * k_loc * 4 * 2  # staging + emitted indices
    )
    flops = 0  # selection + address arithmetic; no useful FLOPs to count

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        _, us = run_perftest(fn)
        torch.cuda.synchronize()
        err = checkAllclose(
            ref_counts.to(dtypes.fp32),
            counts.to(dtypes.fp32),
            rtol=0,
            atol=0,
            msg=f"{name}: owned_counts",
        )
        # indptr must be the exclusive cumsum of counts, or the packed rows
        # overlap and every downstream gather reads a neighbour's slots.
        want_indptr = torch.zeros(rows + 1, dtype=torch.int32)
        want_indptr[1:] = torch.cumsum(ref_counts, 0, dtype=torch.int32)
        torch.testing.assert_close(
            indptr, want_indptr, rtol=0, atol=0, msg=f"{name}: indptr"
        )
        # The slot SET per row is the contract; within-row order is not.
        for r in range(rows):
            lo, hi = int(indptr[r]), int(indptr[r + 1])
            checkAllclose(
                torch.sort(ref_slots[r]).values.to(dtypes.fp32),
                torch.sort(indices[lo:hi]).values.to(dtypes.fp32),
                rtol=0,
                atol=0,
                msg=f"{name}: row {r} slots",
                printLog=False,
            )
        # Mode-specific invariant: guards against passing via a degenerate path.
        if "exact_counts" in expect:
            assert torch.all(
                counts == expect["exact_counts"]
            ), f"{name}/{mode}: counts {counts} != {expect['exact_counts']}"
        if "max_counts" in expect:
            assert torch.all(
                counts <= expect["max_counts"]
            ), f"{name}/{mode}: emitted padded candidates: {counts}"

        # The W owned sets must be disjoint and total exactly the global top-k.
        # Only checkable by running every rank, so it is not timed.
        total = torch.zeros(rows, dtype=torch.int32)
        for r_id in range(world):
            _, _, c, _ = run_merge(scores, local_idx, bt, r_id, world, topk, page)
            total += c
        n_live = int(torch.isfinite(scores[0]).sum())
        assert torch.all(
            total == min(topk, n_live)
        ), f"{name}/{mode}: ranks do not partition topk: {total}"

        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6 if us > 0 else 0
        ret[f"{name} TB/s"] = nbytes / us / 1e6 if us > 0 else 0
        ret[f"{name} err"] = err
    return ret


# @benchmark does not hide the fn from pytest: collected by its test_ name, it
# would be called with zero fixtures and raise TypeError. The parameterized
# wrapper below is this file's pytest entry point into the same code.
test_dcp_topk_merge.__test__ = False


@pytest.mark.parametrize("mode", CASE_MODES)
@pytest.mark.parametrize("rows,world,page", [(1, 8, 64), (17, 4, 16), (256, 2, 128)])
def test_modes(rows, world, page, mode):
    """Drive the benchmark fn's assertions over edge shapes under pytest."""
    k_loc = 64
    # "overtake" is the topk >= n_cand degenerate case; the rest stay selective.
    topk = world * k_loc if mode == "overtake" else 128
    test_dcp_topk_merge(rows, world, k_loc, topk, page, mode)


def test_reference_model_selects_global_topk():
    """The oracle keeps exactly this rank's share of the global winners."""
    rows, world, k_loc, topk, page = 2, 4, 8, 16, 4
    scores, local_idx, bt = make_case(rows, world, k_loc, topk, page, 64, seed=1)
    slots, counts = ref_dcp_merge(scores, local_idx, bt, 2, k_loc, topk, page)
    # Union across all ranks must be exactly `topk` winners per row.
    total = torch.zeros(rows, dtype=torch.int32, device="cuda")
    for r_id in range(world):
        _, c = ref_dcp_merge(scores, local_idx, bt, r_id, k_loc, topk, page)
        total += c
    assert torch.all(total == topk), f"ranks do not partition topk: {total}"
    # counts[r] must equal the number of slots returned for that row.
    for r in range(rows):
        assert int(counts[r]) == slots[r].numel(), (
            f"row {r}: counts[r]={int(counts[r])} but slots[r] has "
            f"{slots[r].numel()} elements"
        )
    assert len(slots) == rows


def test_kernel_owned_slots_match_reference():
    """Per-row owned slots and counts match the oracle, for every rank.

    The benchmark fn only checks the packed output; `staging` is the pre-pack
    per-row scratch, and a bug that corrupts it but happens to pack correctly
    would slip past. Checked here per rank instead.
    """
    rows, world, k_loc, topk, page = 4, 8, 64, 128, 16
    scores, local_idx, bt = make_case(rows, world, k_loc, topk, page, 64, seed=3)
    for rank in range(world):
        _, _, counts, staging = run_merge(
            scores, local_idx, bt, rank, world, topk, page
        )
        want_slots, want_counts = ref_dcp_merge(
            scores, local_idx, bt, rank, k_loc, topk, page
        )
        torch.testing.assert_close(counts, want_counts, rtol=0, atol=0)
        for r in range(rows):
            n = int(counts[r])
            torch.testing.assert_close(
                staging[r, :n],
                want_slots[r],
                rtol=0,
                atol=0,
                msg=f"rank {rank} row {r} slot mismatch",
            )


@pytest.mark.parametrize(
    "rows,world,k_loc,topk,page,tie_heavy,seed",
    [
        (4, 8, 64, 128, 16, True, 6),  # tie-heavy: many equal-to-threshold
        (7, 8, 128, 32, 16, False, 31),  # the shape C1 was reproduced on
    ],
)
def test_deterministic_across_runs(rows, world, k_loc, topk, page, tie_heavy, seed):
    """Repeated runs must be bitwise identical AND match the oracle.

    The kernel's block scan reuses one LDS buffer across back-to-back calls, so a
    missing barrier shows up only as a rare scheduling-dependent drift. Two runs
    are nowhere near enough to catch that; loop hard instead.
    """
    scores, local_idx, bt = make_case(
        rows, world, k_loc, topk, page, 512, tie_heavy=tie_heavy, seed=seed
    )
    # Saturate the GPU from a second stream. Whether waves of one block get
    # skewed enough to expose a missing LDS barrier is a scheduling accident;
    # co-resident load makes that accident reliable instead of rare.
    noise = torch.randn(8192, 8192)
    noise_stream = torch.cuda.Stream()
    for rank in range(world):
        want_slots, want_counts = ref_dcp_merge(
            scores, local_idx, bt, rank, k_loc, topk, page
        )
        ref = None
        for it in range(200 // world):
            with torch.cuda.stream(noise_stream):
                for _ in range(3):
                    noise @ noise
            indices, indptr, counts, _ = run_merge(
                scores, local_idx, bt, rank, world, topk, page
            )
            torch.testing.assert_close(
                counts,
                want_counts,
                rtol=0,
                atol=0,
                msg=f"rank {rank} iter {it}: counts drifted from the oracle",
            )
            for r in range(rows):
                lo, hi = int(indptr[r]), int(indptr[r + 1])
                torch.testing.assert_close(
                    torch.sort(indices[lo:hi]).values,
                    torch.sort(want_slots[r]).values,
                    rtol=0,
                    atol=0,
                    msg=f"rank {rank} iter {it} row {r}: slots differ from oracle",
                )
            if ref is None:
                ref = (indices.clone(), indptr.clone())
            else:
                torch.testing.assert_close(
                    indices,
                    ref[0],
                    rtol=0,
                    atol=0,
                    msg=f"rank {rank} iter {it}: indices not reproducible",
                )
                torch.testing.assert_close(
                    indptr,
                    ref[1],
                    rtol=0,
                    atol=0,
                    msg=f"rank {rank} iter {it}: indptr not reproducible",
                )


def _atom_available():
    """Probe the SYMBOLS this test uses, not just the module.

    triton_filter_and_convert_dcp_index is the decode filter the fused op
    replaces; the companion ATOM change deletes it. Probing the module alone
    would make these cases fail rather than skip once that lands, so the two
    repos could not be merged in either order.
    """
    try:
        from atom.model_ops.dcp_ops import (  # noqa: F401
            dcp_global_pos,
            triton_filter_and_convert_dcp_index,
        )

        return True
    except Exception:  # noqa: BLE001  probe: any import failure means unavailable
        return False


@pytest.mark.skipif(
    not _atom_available(),
    reason="needs an ATOM that still has triton_filter_and_convert_dcp_index",
)
@pytest.mark.parametrize("interleave", [1, 2, 4])
@pytest.mark.parametrize("page_size", [16, 64])
def test_matches_triton_filter_and_convert(interleave, page_size):
    """End-to-end parity with today's pipeline: same slots, same indptr.

    Builds the global top-k the old way (gather gids, merge, filter) and the new
    way (fused select + emit), then compares the packed outputs elementwise.
    """
    from atom.model_ops.dcp_ops import (
        dcp_global_pos,
        triton_filter_and_convert_dcp_index,
    )

    rows, world, k_loc, topk = 4, 8, 128, 256
    max_blocks = 512
    rank = 3
    scores, local_idx, bt = make_case(
        rows, world, k_loc, topk, page_size, max_blocks, seed=7
    )

    # --- old path: build global top-k over gids, then filter to owned ---
    gids = torch.empty(rows, world * k_loc, dtype=torch.int32)
    for w in range(world):
        j = local_idx.clamp(min=0).to(torch.int64)
        g = dcp_global_pos(j, w, world, interleave).to(torch.int32)
        gids[:, w * k_loc : (w + 1) * k_loc] = torch.where(local_idx >= 0, g, -1)
    order = torch.argsort(scores, dim=-1, descending=True, stable=True)
    win = order[:, :topk]
    global_topk = torch.gather(gids, 1, win).contiguous()

    qo_indptr = torch.arange(rows + 1, dtype=torch.int32)
    g_kv_indptr = torch.zeros(rows + 1, dtype=torch.int32)
    old_indptr = torch.zeros(rows + 1, dtype=torch.int32)
    old_counts = torch.zeros(rows, dtype=torch.int32)
    old_indices = torch.zeros(rows * topk, dtype=torch.int32)
    triton_filter_and_convert_dcp_index(
        qo_indptr,
        g_kv_indptr,
        bt,
        global_topk,
        rank,
        world,
        page_size,
        out_kv_indptr=old_indptr,
        owned_counts=old_counts,
        NUM_TOPK_TOKENS=topk,
        out=old_indices,
        cp_kv_cache_interleave_size=interleave,
    )

    # --- new path ---
    indices, indptr, _counts, _ = run_merge(
        scores, local_idx, bt, rank, world, topk, page_size
    )

    torch.testing.assert_close(indptr, old_indptr, rtol=0, atol=0)
    total = int(old_indptr[rows])
    torch.testing.assert_close(
        torch.sort(indices[:total]).values,
        torch.sort(old_indices[:total]).values,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("stable", [False, True])
def test_decode_topk_values_pad_with_neg_inf(stable):
    """`values=` writes each index's logit, and pads short rows with -inf.

    This is the precondition flydsl_dcp_topk_merge relies on: padding travels in
    the all-gathered score plane with no liveness flag of its own, so it has to
    lose the global threshold comparison on score alone. A regression to the old
    0.0 pad outranks real (negative) logits and makes this rank under-emit --
    silently, since nothing crashes. Measured before the fix: the owned
    partition collapsed from 256 entries to 30.
    """
    rows, n, k = 4, 4096, 512
    torch.manual_seed(17)
    logits = torch.randn(rows, n, dtype=torch.float32)
    # Rows shorter than k are the whole point -- that is where padding appears.
    lens = torch.tensor([0, 1, 100, n], dtype=torch.int32)

    idx = torch.full((rows, k), -777, dtype=torch.int32)
    vals = torch.full((rows, k), float("nan"), dtype=torch.float32)
    aiter.top_k_per_row_decode(
        logits,
        1,
        lens,
        idx,
        rows,
        logits.stride(0),
        logits.stride(1),
        k,
        stable=stable,
        values=vals,
    )
    torch.cuda.synchronize()

    pad = idx < 0
    assert bool(
        (vals[pad] == -float("inf")).all()
    ), "padded slots must carry -inf, not 0.0"
    real = ~pad
    gathered = logits.gather(1, idx.clamp(min=0).to(torch.int64))
    torch.testing.assert_close(vals[real], gathered[real], rtol=0, atol=0)
    if pad.any() and real.any():
        assert (
            vals[pad].max() < vals[real].min()
        ), "padding must sort below all real scores"


def test_decode_topk_values_none_is_unchanged():
    """Passing values= must not perturb the indices the op already returned."""
    rows, n, k = 4, 4096, 256
    torch.manual_seed(19)
    logits = torch.randn(rows, n, dtype=torch.float32) - 3.0  # all negative
    lens = torch.tensor([0, 7, 100, n], dtype=torch.int32)

    without = torch.full((rows, k), -777, dtype=torch.int32)
    aiter.top_k_per_row_decode(
        logits,
        1,
        lens,
        without,
        rows,
        logits.stride(0),
        logits.stride(1),
        k,
        stable=True,
    )
    with_vals = torch.full((rows, k), -777, dtype=torch.int32)
    vals = torch.full((rows, k), float("nan"), dtype=torch.float32)
    aiter.top_k_per_row_decode(
        logits,
        1,
        lens,
        with_vals,
        rows,
        logits.stride(0),
        logits.stride(1),
        k,
        stable=True,
        values=vals,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(with_vals, without, rtol=0, atol=0)


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "FlyDSL DCP TopK merge unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL DCP decode TopK merge correctness + perf sweep",
    )
    # rows == num_decode_tokens: the decode batch this rank scheduled.
    parser.add_argument("-b", "--rows", type=int, nargs="*", default=[1, 32, 128, 256])
    parser.add_argument("-w", "--world", type=int, nargs="*", default=[8])
    # Production ships k_loc == topk == index_topk == 2048.
    parser.add_argument("-k", "--k-loc", type=int, nargs="*", default=[2048])
    parser.add_argument("--topk", type=int, nargs="*", default=[2048])
    parser.add_argument("-p", "--page", type=int, nargs="*", default=[64])
    parser.add_argument(
        "-m", "--mode", type=str, nargs="*", default=CASE_MODES, choices=CASE_MODES
    )
    args = parser.parse_args()

    df = []
    for rows, world, k_loc, topk, page, mode in itertools.product(
        args.rows, args.world, args.k_loc, args.topk, args.page, args.mode
    ):
        df.append(test_dcp_topk_merge(rows, world, k_loc, topk, page, mode))
    df = pd.DataFrame(df)
    aiter.logger.info(
        "FlyDSL DCP TopK merge summary (markdown):\n%s", df.to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
