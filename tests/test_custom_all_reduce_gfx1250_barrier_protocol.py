# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only checks for the gfx1250 two-stage fused-AR barrier protocol."""

import random
import re
from pathlib import Path


CUH = (
    Path(__file__).resolve().parents[1]
    / "csrc/include/custom_all_reduce_gfx1250.cuh"
)
U32_MASK = (1 << 32) - 1


def _epoch_before(observed: int, target: int) -> bool:
    delta = (observed - target) & U32_MASK
    signed = delta if delta < 1 << 31 else delta - (1 << 32)
    return signed < 0


def _arrive_grid(blocks: int, rng: random.Random) -> int:
    """Model the local acq_rel fetch-add chain and return the last block ID."""
    order = list(range(blocks))
    rng.shuffle(order)
    counter = 0
    last = None
    for block in order:
        arrived = counter
        counter += 1
        if arrived == blocks - 1:
            assert last is None
            last = block
    assert counter == blocks and last is not None
    # The last block resets before entering the peer barrier. The next kernel
    # cannot arrive until that barrier and this kernel have completed.
    counter = 0
    assert counter == 0
    return last


def _takes_two_stage(
    tokens: int,
    hidden: int,
    *,
    world_size: int = 4,
    element_size: int = 2,
    use_1stage: bool = False,
) -> bool:
    pack_size = 16 // element_size
    total_bytes = tokens * hidden * element_size
    return (
        tokens > 0
        and not use_1stage
        and world_size >= 4
        and hidden >= 6144
        and hidden % pack_size == 0
        and (hidden // pack_size) % world_size == 0
        and total_bytes // world_size <= 16 * 1024 * 1024
        and total_bytes > 4 * 1024 * 1024
    )


def test_epoch_comparison_is_wrap_safe():
    for target in (0, 1, 2, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF):
        assert _epoch_before((target - 1) & U32_MASK, target)
        assert not _epoch_before(target, target)
        assert not _epoch_before((target + 1) & U32_MASK, target)


def test_two_stage_epochs_and_last_block_protocol_repeat():
    """Eager calls and graph replays share persistent counters and epochs."""
    for blocks in (1, 7, 512):
        rng = random.Random(393_006 + blocks)
        ranks = 4
        # Start close enough to wrap both phase-specific slots during the run.
        flags = [[0xFFFFFFF8, 0xFFFFFFFC] for _ in range(ranks)]

        for _ in range(200):
            # The retained start_sync is slot 0's first phase.
            start_target = (flags[0][0] + 1) & U32_MASK
            assert all(
                (rank_flags[0] + 1) & U32_MASK == start_target
                for rank_flags in flags
            )
            for rank_flags in flags:
                rank_flags[0] = start_target

            # Arbitrary scheduling can choose a different last block on every rank.
            stage1_last = [_arrive_grid(blocks, rng) for _ in range(ranks)]
            assert len(stage1_last) == ranks

            # Intermediate barrier: release/acquire on slot 0.
            mid_target = (flags[0][0] + 1) & U32_MASK
            mid_signals = [mid_target] * ranks
            assert all(
                not _epoch_before(signal, mid_target) for signal in mid_signals
            )
            for rank_flags in flags:
                rank_flags[0] = mid_target

            stage2_last = [_arrive_grid(blocks, rng) for _ in range(ranks)]
            assert len(stage2_last) == ranks

            # Final barrier has its own slot, so its relaxed signal cannot
            # overwrite a release signal that a slower rank still needs.
            final_target = (flags[0][1] + 1) & U32_MASK
            final_signals = [final_target] * ranks
            assert all(
                not _epoch_before(signal, final_target)
                for signal in final_signals
            )
            for rank_flags in flags:
                rank_flags[1] = final_target

            assert len({tuple(rank_flags) for rank_flags in flags}) == 1


def test_acceptance_shapes_use_two_stage_and_exceptions_fall_back():
    assert _takes_two_stage(2517, 6144)
    assert _takes_two_stage(4536, 6144)
    assert not _takes_two_stage(0, 6144)
    assert not _takes_two_stage(341, 6144)  # LL range
    assert not _takes_two_stage(5462, 6144)  # staging shard exceeds 16 MiB
    assert not _takes_two_stage(2517, 6144, world_size=2)
    assert not _takes_two_stage(2517, 4096)
    assert not _takes_two_stage(2517, 6152)  # packs do not split over TP4
    assert not _takes_two_stage(2517, 6144, use_1stage=True)


def test_fast_final_phase_cannot_overwrite_slow_mid_acquire():
    """The race that requires distinct intermediate/final signal slots."""
    mid_slot, final_slot = 0, 1
    end_signals = [[0, 0] for _ in range(4)]
    mid_target = 17
    final_target = 29

    for rank in range(4):
        end_signals[rank][mid_slot] = mid_target  # release
    # Ranks 0..2 leave the intermediate barrier and finish stage 2 while rank 3
    # has not loaded their intermediate signals yet.
    for rank in range(3):
        end_signals[rank][final_slot] = final_target  # relaxed

    assert all(
        end_signals[rank][mid_slot] == mid_target for rank in range(4)
    ), "a final signal overwrote the release epoch"


def test_source_fuses_only_end_barriers_and_keeps_fallback():
    src = CUH.read_text()
    launch = src[
        src.index("#define LAUNCH_AR_RMS")
        : src.index("DISPATCH_NGPUS_1250(world_size_, LAUNCH_AR_RMS)")
    ]
    two_stage_launch = launch[: launch.index("else if(gemma_norm)")]

    # Eligible prefill remains three launches: retained start wait, stage 1,
    # stage 2. The two standalone end_sync launches are gone.
    assert two_stage_launch.count("start_sync_kernel<NG>") == 1
    assert two_stage_launch.count("ar_rmsnorm_reduce_scatter_gfx1250<T, NG>") == 1
    assert (
        two_stage_launch.count("ar_rmsnorm_gfx1250<T, OutT, NG") == 2
    )  # Gemma/plain
    assert "end_sync_kernel<NG>" not in two_stage_launch

    # The one-stage branch is still present for explicit overrides and all
    # exceptional sizes rejected by the two-stage predicate.
    assert "else if(gemma_norm)" in launch
    predicate = re.search(r"const bool two_stage =\s*(.*?);", src, re.DOTALL)
    assert predicate
    for guard in (
        "!use_1stage",
        "world_size_ >= 4",
        "hidden >= kArRmsTwoStageMinHidden",
        "hidden / pack_size % world_size_ == 0",
        "bytes / world_size_ <= kArRmsShardBytes",
        "bytes > kLLArMaxBytes",
    ):
        assert guard in predicate.group(1)


def test_source_encodes_memory_order_and_phase_separation():
    src = CUH.read_text()
    grid_sync = re.search(
        r"DINLINE void grid_end_sync\(.*?^\}", src, re.DOTALL | re.MULTILINE
    )
    assert grid_sync
    body = grid_sync.group(0)
    assert "__ATOMIC_ACQ_REL" in body
    assert "__HIP_MEMORY_SCOPE_AGENT" in body
    assert "end_sync_slot<ngpus, final_sync>" in body

    assert (
        "grid_end_sync<ngpus>(sg, self_sg, rank, kArRmsMidSignalBlock)" in src
    )
    assert (
        "grid_end_sync<ngpus, true>(sg, self_sg, rank, "
        "kArRmsFinalSignalBlock)" in src
    )
    assert "kArRmsMidSignalBlock   = 0" in src
    assert "kArRmsFinalSignalBlock = 1" in src


if __name__ == "__main__":
    test_epoch_comparison_is_wrap_safe()
    test_two_stage_epochs_and_last_block_protocol_repeat()
    test_acceptance_shapes_use_two_stage_and_exceptions_fall_back()
    test_fast_final_phase_cannot_overwrite_slow_mid_acquire()
    test_source_fuses_only_end_barriers_and_keeps_fallback()
    test_source_encodes_memory_order_and_phase_separation()
