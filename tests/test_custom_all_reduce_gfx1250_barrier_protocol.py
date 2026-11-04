# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only checks for the gfx1250 two-stage fused AR + RMSNorm barriers.

The model runs four ranks through many back-to-back calls under random
interleavings. Each rank's stream is: copy the next input in, barrier,
stage 1 (read every input, write own shard), barrier, stage 2 (read every
shard). A barrier is start_sync on one slot: store the next epoch into every
peer's slot, wait until every peer's epoch is at or after it. Kernel
completion is modelled as publishing a stage's writes before the next step of
that rank, which is what the relaxed barriers rely on. The checker fails on a
stale or torn read, on a buffer overwritten while a peer reads it, and on a
deadlock.
"""

import random
import re
from pathlib import Path

CUH = (
    Path(__file__).resolve().parents[1] / "csrc/include/custom_all_reduce_gfx1250.cuh"
)
U32 = (1 << 32) - 1
RANKS = 4


def epoch_before(observed: int, target: int) -> bool:
    delta = (observed - target) & U32
    return (delta if delta < 1 << 31 else delta - (1 << 32)) < 0


def plain_before(observed: int, target: int) -> bool:
    return observed < target


def protocol(final_barrier=False, mid_barrier=True, start_barrier=True):
    steps = ["copy_in"]
    if start_barrier:
        steps += ["arrive", "wait"]
    steps += ["rs_begin", "rs_end"]
    if mid_barrier:
        steps += ["arrive", "wait"]
    steps += ["arn_begin", "arn_end"]
    if final_barrier:
        steps += ["arrive", "wait"]
    return steps


def run(steps, calls, seed, before=epoch_before, epoch0=0):
    """Return None on success or a description of the first violation."""
    rng = random.Random(seed)
    flag = [epoch0] * RANKS  # Signal::_flag[0] of each rank
    slot = [[epoch0] * RANKS for _ in range(RANKS)]  # slot[r][i]: written by rank i
    target = [None] * RANKS
    pc = [0] * RANKS
    call = [0] * RANKS
    inp = [-1] * RANKS
    shard = [-1] * RANKS
    in_rs = [False] * RANKS
    in_arn = [False] * RANKS

    def enabled(r):
        if call[r] == calls:
            return False
        if steps[pc[r]] == "wait":
            return not any(before(slot[r][i], target[r]) for i in range(RANKS))
        return True

    while True:
        ready = [r for r in range(RANKS) if enabled(r)]
        if not ready:
            if all(c == calls for c in call):
                return None
            return f"deadlock at calls {call}, steps {[steps[p] for p in pc]}"
        r = rng.choice(ready)
        c, step = call[r], steps[pc[r]]
        if step == "copy_in":
            if any(in_rs):
                return f"rank {r} call {c}: input overwritten during a peer's stage 1"
            inp[r] = c
        elif step == "arrive":
            target[r] = (flag[r] + 1) & U32
            for p in range(RANKS):
                slot[p][r] = target[r]
        elif step == "wait":
            flag[r] = target[r]
        elif step == "rs_begin":
            if any(v != c for v in inp):
                return f"rank {r} call {c}: stage 1 read stale input {inp}"
            if any(in_arn):
                return f"rank {r} call {c}: shard overwritten during a peer's stage 2"
            in_rs[r] = True
        elif step == "rs_end":
            in_rs[r] = False
            shard[r] = c
        elif step == "arn_begin":
            if any(v != c for v in shard) or any(in_rs):
                return f"rank {r} call {c}: stage 2 read stale or partial shard {shard}"
            in_arn[r] = True
        elif step == "arn_end":
            in_arn[r] = False
        pc[r] += 1
        if pc[r] == len(steps):
            pc[r] = 0
            call[r] += 1


def test_epoch_comparison_is_wrap_safe():
    for target in (0, 1, 2, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF):
        assert epoch_before((target - 1) & U32, target)
        assert not epoch_before(target, target)
        assert not epoch_before((target + 1) & U32, target)


def test_four_launch_protocol_is_race_and_deadlock_free():
    steps = protocol()
    for seed in range(300):
        # Start close to the uint32 limit so the run wraps the epoch.
        assert run(steps, calls=40, seed=seed, epoch0=0xFFFFFFC0) is None, seed


def test_checker_finds_the_races_the_barriers_prevent():
    # Without the mid barrier stage 2 reads unfinished shards; without the
    # start barrier stage 1 reads stale inputs or overwrites a read shard.
    for steps in (protocol(mid_barrier=False), protocol(start_barrier=False)):
        assert any(run(steps, calls=20, seed=s) for s in range(50))


def test_plain_comparison_breaks_at_wrap():
    # The pre-fix "observed < flag" wait passes a barrier early once the
    # epoch wraps to 0, so the same protocol races near the wrap.
    steps = protocol()
    assert any(
        run(steps, calls=40, seed=s, before=plain_before, epoch0=0xFFFFFFF0)
        for s in range(100)
    )


def test_final_barrier_is_redundant():
    # The base protocol (with the final end barrier) and the new one both pass;
    # the next call's start barrier is what keeps the shard from being reused.
    for seed in range(100):
        assert run(protocol(final_barrier=True), calls=20, seed=seed) is None


def test_two_stage_launches_four_kernels_without_end_sync():
    src = CUH.read_text()
    launch = src[
        src.index("#define LAUNCH_AR_RMS") : src.index(
            "DISPATCH_NGPUS_1250(world_size_, LAUNCH_AR_RMS)"
        )
    ]
    two_stage = launch[: launch.index("else if(gemma_norm)")]
    assert two_stage.count("start_sync_kernel<NG>") == 2
    assert two_stage.count("ar_rmsnorm_reduce_scatter_gfx1250<T, NG>") == 1
    assert two_stage.count("ar_rmsnorm_gfx1250<T, OutT, NG") == 2  # gemma / plain
    assert "end_sync_kernel" not in two_stage
    # The mid barrier sits between the stages.
    assert (
        two_stage.index("ar_rmsnorm_reduce_scatter_gfx1250")
        < two_stage.rindex("start_sync_kernel")
        < two_stage.index("ar_rmsnorm_gfx1250<T, OutT, NG")
    )


def test_stage_kernels_have_no_barrier():
    src = CUH.read_text()
    rs = re.search(
        r"ar_rmsnorm_reduce_scatter_gfx1250\(.*?\n\}\n", src, re.DOTALL
    ).group(0)
    assert "_sync" not in rs and "atomic" not in rs
    arn = re.search(r"ar_rmsnorm_gfx1250\(\s*RankData.*?\n\}\n", src, re.DOTALL).group(0)
    assert "if constexpr(!TWO_STAGE)\n        start_sync<ngpus>" in arn
    assert "if constexpr(!TWO_STAGE)\n        end_sync<ngpus, true>" in arn


def test_waits_use_serial_comparison():
    src = CUH.read_text()
    for fn in ("start_sync", "end_sync"):
        body = re.search(
            rf"DINLINE void {fn}\(.*?\n\}}\n", src, re.DOTALL
        ).group(0)
        assert "epoch_before(" in body and ") < flag" not in body


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
