# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL DCP decode TopK merge tests."""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]

# Case shapes the correctness checks and the perf sweep share. Each one bends a
# different part of the op; see make_case for how the inputs are bent.
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
            rows, world, k_loc, page, max_blocks, tie_heavy=True, seed=seed
        )
        return s, li, bt, {}
    if mode == "short":
        # Short context: only n_real live candidates per plane, rest padded.
        # n_real > page so local indices span >1 block-table entry.
        n_real = max(page + 4, k_loc // 8)
        n_real = min(n_real, k_loc)
        s, li, bt = make_case(
            rows, world, k_loc, page, max_blocks, n_local=n_real, seed=seed
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
        s, li, bt = make_case(rows, world, k_loc, page, max_blocks, seed=seed)
        s[:, rank * k_loc : (rank + 1) * k_loc] = -1e30
        spill = max(0, topk - (world - 1) * k_loc)
        return s, li, bt, {"exact_counts": spill}
    if mode == "overtake":
        # topk_tokens >= n_cand degenerates to "take everything", not garbage:
        # every rank ends up owning its entire plane. Only assert that when the
        # caller's topk actually reaches n_cand -- the production sweep ships
        # topk=2048 against n_cand=16384, which is selective, not degenerate.
        s, li, bt = make_case(rows, world, k_loc, page, max_blocks, seed=seed)
        expect = {"exact_counts": k_loc} if topk >= world * k_loc else {}
        return s, li, bt, expect
    if mode == "random":
        return (
            *make_case(rows, world, k_loc, page, max_blocks, seed=seed),
            {},
        )
    raise ValueError(f"unknown case mode: {mode}")


def run_merge(scores, local_idx, bt, rank, world, topk, page):
    """Call the op with freshly allocated outputs; return (indices, indptr, counts).

    `staging` is allocated here because the op requires a caller-provided
    scratch buffer, but nothing reads it back: it is the pre-pack per-row form
    of exactly the slots that land in `indices`, so asserting on it adds no
    coverage the packed output does not already give.
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
    return indices, indptr, counts


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
        # checkAllclose is for the table's `err` column, NOT for gating: it only
        # raises on a "catastrophic" delta and otherwise just logs and returns
        # the mismatch ratio. Every invariant below is asserted separately with
        # torch.testing.assert_close / assert so a wrong result fails the run.
        err = checkAllclose(
            ref_counts.to(dtypes.fp32),
            counts.to(dtypes.fp32),
            rtol=0,
            atol=0,
            msg=f"{name}: owned_counts",
        )
        torch.testing.assert_close(
            counts, ref_counts, rtol=0, atol=0, msg=f"{name}/{mode}: owned_counts"
        )
        # indptr is the exclusive cumsum of the KERNEL's counts -- deriving it
        # from the oracle instead would make the row slices below read from
        # oracle offsets, silently checking the wrong region of `indices`.
        want_indptr = torch.zeros(rows + 1, dtype=torch.int32)
        want_indptr[1:] = torch.cumsum(counts, 0, dtype=torch.int32)
        torch.testing.assert_close(
            indptr, want_indptr, rtol=0, atol=0, msg=f"{name}/{mode}: indptr"
        )
        # The slot SET per row is the contract; within-row order is not.
        for r in range(rows):
            lo, hi = int(indptr[r]), int(indptr[r + 1])
            torch.testing.assert_close(
                torch.sort(indices[lo:hi]).values,
                torch.sort(ref_slots[r]).values,
                rtol=0,
                atol=0,
                msg=f"{name}/{mode}: row {r} slots",
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

        # Across all W ranks the owned counts must total exactly the global
        # top-k: each rank keeps the winners that fall in its own plane, and the
        # planes tile the candidate axis, so nothing may be dropped or claimed
        # twice. Computed per row -- "short" mode leaves rows with differing
        # live-candidate counts, so a single scalar bound would be wrong.
        #
        # Disjointness of the emitted SLOTS is deliberately not asserted here:
        # make_case gives every rank the same local_idx and block_table, so two
        # planes legitimately map to one physical slot in this fixture (the
        # oracle collides identically). Real DCP gives each rank its own KV
        # slice; proving that needs a multi-rank fixture this file does not have.
        n_live = torch.isfinite(scores).sum(dim=1)
        want_total = torch.clamp(n_live, max=topk).to(torch.int32)
        total = torch.zeros(rows, dtype=torch.int32)
        for r_id in range(world):
            _, _, c = run_merge(scores, local_idx, bt, r_id, world, topk, page)
            total += c
        torch.testing.assert_close(
            total, want_total, rtol=0, atol=0, msg=f"{name}/{mode}: rank partition"
        )

        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6 if us > 0 else 0
        ret[f"{name} TB/s"] = nbytes / us / 1e6 if us > 0 else 0
        ret[f"{name} err"] = err
    return ret


def check_modes():
    """Run every case mode over edge shapes the perf sweep does not cover."""
    k_loc = 64
    for rows, world, page in [(1, 8, 64), (17, 4, 16), (256, 2, 128)]:
        for mode in CASE_MODES:
            # "overtake" is the topk >= n_cand degenerate case.
            topk = world * k_loc if mode == "overtake" else 128
            test_dcp_topk_merge(rows, world, k_loc, topk, page, mode)


def check_nan_scores_never_win():
    """A NaN score must lose the threshold, not sort above real candidates.

    The kernel's float->ordinal map has to agree with the reference model, which
    treats every non-finite value as -inf. If NaN mapped to INT32_MAX instead it
    would win every comparison, steal a real candidate's slot, and silently move
    a KV block to the wrong rank -- no crash, just wrong attention.
    """
    rows, world, k_loc, topk, page = 2, 4, 32, 16, 16
    scores, local_idx, bt = make_case(rows, world, k_loc, page, 64, seed=13)
    rank = 1
    # Poison this rank's plane: NaN where a real (losing) score used to be.
    scores[:, rank * k_loc : rank * k_loc + 8] = float("nan")
    indices, indptr, counts = run_merge(scores, local_idx, bt, rank, world, topk, page)
    want_slots, want_counts = ref_dcp_merge(
        scores, local_idx, bt, rank, k_loc, topk, page
    )
    torch.testing.assert_close(counts, want_counts, rtol=0, atol=0)
    for r in range(rows):
        lo, hi = int(indptr[r]), int(indptr[r + 1])
        torch.testing.assert_close(
            torch.sort(indices[lo:hi]).values,
            torch.sort(want_slots[r]).values,
            rtol=0,
            atol=0,
            msg=f"row {r}",
        )


def check_deterministic_across_runs(rows, world, k_loc, topk, page, tie_heavy, seed):
    """Back-to-back runs on identical input must return identical output.

    The kernel reuses one LDS scan buffer across calls, and both the select and
    the pack block derive their write offsets from it, so a dropped barrier
    shows up as output that varies between otherwise identical launches. Every
    TP rank runs this op on the same gathered scores and must land on the same
    KV set, so reproducibility is a correctness requirement, not a nicety.

    Scope: this checks run-to-run stability only. Whether the result is *right*
    is the oracle comparison in test_dcp_topk_merge; repeating a wrong-but-
    stable answer passes here by design. It is also not a race detector -- a
    barrier can be missing and every run still agree (verified: deleting the
    bucket-hit barrier leaves this green). Treat a pass as evidence of
    determinism, nothing more.
    """
    scores, local_idx, bt = make_case(
        rows, world, k_loc, page, 512, tie_heavy=tie_heavy, seed=seed
    )
    for rank in range(world):
        ref = None
        for it in range(200 // world):
            indices, indptr, _ = run_merge(
                scores, local_idx, bt, rank, world, topk, page
            )
            if ref is None:
                ref = (indices.clone(), indptr.clone())
                continue
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

    # Correctness first: these cover what the perf sweep cannot reach -- small
    # shapes and low world sizes, NaN ordering, and run-to-run reproducibility.
    # Every one raises on failure, so CI goes red before the perf table prints.
    check_modes()
    check_nan_scores_never_win()
    for c_rows, c_world, c_k, c_topk, c_page, c_tie, c_seed in [
        (4, 8, 64, 128, 16, True, 6),  # tie-heavy: many equal-to-threshold
        (7, 8, 128, 32, 16, False, 31),  # the shape C1 was reproduced on
    ]:
        check_deterministic_across_runs(
            c_rows, c_world, c_k, c_topk, c_page, c_tie, c_seed
        )
    aiter.logger.info("FlyDSL DCP TopK merge: correctness checks passed")

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
