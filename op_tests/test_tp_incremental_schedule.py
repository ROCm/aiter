# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""CPU schedule tests for TP incremental activation all-gather (GEMM1 fusion).

No GPU / NCCL. Run with::

    python op_tests/test_tp_incremental_schedule.py
"""

from __future__ import annotations

import pandas as pd
import torch

from aiter.ops.flydsl.kernels.mega_moe.tp_incremental_schedule import (
    chunk_gemm_overlap_schedule,
    chunk_of_m_tile,
    dense_row_index,
    expected_token_counts,
    make_tile_row_base,
    min_expert_ids,
    num_publish_chunks,
    pack_rows_by_sorted_ids,
    partition_counts_by_min_expert,
    publish_order_from_topk,
    simulate_incremental_publish,
    tokens_needed_by_expert,
    tp_dest_row,
)

V4_PRO = {"tp": 8, "m_local": 64, "num_experts": 384, "topk": 6}


def _uniform_topk(tokens: int, num_experts: int, topk: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    scores = torch.randn((tokens, num_experts), generator=g)
    return torch.topk(scores, topk, dim=-1).indices


def _skew_topk(
    tokens: int, num_experts: int, topk: int, seed: int, hot: int = 8
) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    scores = torch.randn((tokens, num_experts), generator=g)
    scores[:, :hot] += 8.0
    return torch.topk(scores, topk, dim=-1).indices


def _tp_concat_ids(m_local: int, tp: int, num_experts: int, topk: int, seed: int):
    ranks = [_uniform_topk(m_local, num_experts, topk, seed + r) for r in range(tp)]
    return torch.cat(ranks, dim=0)


def test_min_expert_is_row_min():
    ids = torch.tensor([[7, 1, 3], [0, 9, 2]], dtype=torch.int32)
    torch.testing.assert_close(
        min_expert_ids(ids), torch.tensor([1, 0], dtype=torch.int32)
    )


def test_partitions_are_disjoint_and_cover_all_tokens():
    ids = _uniform_topk(tokens=128, num_experts=64, topk=6, seed=0)
    parts = partition_counts_by_min_expert(ids, num_experts=64)
    assert int(parts.sum()) == ids.shape[0]
    mins = min_expert_ids(ids)
    for e in range(64):
        assert int((mins == e).sum()) == int(parts[e])


def test_each_token_published_once():
    ids = _uniform_topk(tokens=96, num_experts=32, topk=4, seed=1)
    parts = partition_counts_by_min_expert(ids, num_experts=32)
    assert int(parts.sum()) == ids.shape[0]


def test_expected_counts_match_token_membership():
    ids = _uniform_topk(tokens=80, num_experts=16, topk=3, seed=2)
    expected = expected_token_counts(ids, num_experts=16)
    for e in range(16):
        assert int(expected[e]) == int(tokens_needed_by_expert(ids, e).sum())


def test_ready_after_own_round_and_prefix_invariant():
    ids = _uniform_topk(tokens=200, num_experts=32, topk=6, seed=3)
    expected, received, ready, first = simulate_incremental_publish(ids, 32)
    for e in range(32):
        assert bool(ready[e, e]), f"expert {e} must be ready after round {e}"
        for k in range(e + 1):
            assert bool(ready[e, k]), f"expert {k} must stay ready after round {e}"
        if int(expected[e]) == 0:
            assert int(first[e]) == -1
        else:
            assert 0 <= int(first[e]) <= e
            assert int(received[int(first[e]), e]) == int(expected[e])


def test_early_ready_is_possible_for_later_experts():
    # Token 0: experts 0 and 5. Token 1: expert 5 only.
    # After round 0, expert 5 is not ready; after round 5 it is.
    # Token that only hits a late expert cannot be ready early.
    ids = torch.tensor([[0, 5], [5, 5]], dtype=torch.int32)
    _, _, ready, first = simulate_incremental_publish(ids, num_experts=6)
    assert int(first[0]) == 0
    assert int(first[5]) == 5
    assert not bool(ready[0, 5])

    # Both tokens include expert 1 as min of one of them is 0, other is 1:
    # token A: [0, 2], token B: [1, 2] -> expert 2 ready after round 1, before 2.
    ids = torch.tensor([[0, 2], [1, 2]], dtype=torch.int32)
    _, _, ready, first = simulate_incremental_publish(ids, num_experts=3)
    assert int(first[2]) == 1
    assert bool(ready[1, 2])
    assert not bool(ready[0, 2])


def test_negative_skip_round_blocks_that_expert():
    ids = _uniform_topk(tokens=64, num_experts=16, topk=4, seed=4)
    expected, received, _, _ = simulate_incremental_publish(ids, 16)
    parts = partition_counts_by_min_expert(ids, 16)
    skip = int(parts.argmax())
    assert int(parts[skip]) > 0
    # Re-simulate without publishing skip: received for skip must miss those tokens.
    mins = min_expert_ids(ids)
    recv = torch.zeros(16, dtype=torch.int64)
    for e in range(16):
        if e == skip:
            continue
        mask = mins == e
        if not bool(mask.any()):
            continue
        presence = torch.zeros((int(mask.sum()), 16), dtype=torch.bool)
        presence.scatter_(1, ids[mask].to(torch.int64), True)
        recv += presence.sum(dim=0, dtype=torch.int64)
    assert int(recv[skip]) < int(expected[skip])
    # Full schedule does reach expected[skip].
    assert int(received[-1, skip]) == int(expected[skip])


def test_v4_pro_shape_uniform_and_skew():
    tp, m_local, E, K = (
        V4_PRO["tp"],
        V4_PRO["m_local"],
        V4_PRO["num_experts"],
        V4_PRO["topk"],
    )
    uniform = _tp_concat_ids(m_local, tp, E, K, seed=10)
    skew_ranks = [_skew_topk(m_local, E, K, seed=20 + r, hot=8) for r in range(tp)]
    skew = torch.cat(skew_ranks, dim=0)
    assert uniform.shape == (tp * m_local, K)
    for name, ids in (("uniform", uniform), ("skew", skew)):
        parts = partition_counts_by_min_expert(ids, E)
        expected, received, ready, first = simulate_incremental_publish(ids, E)
        assert int(parts.sum()) == ids.shape[0]
        assert torch.equal(received[-1], expected)
        assert bool(ready[-1].all())
        early = int(((first >= 0) & (first < torch.arange(E))).sum())
        print(f"{name}: tokens={ids.shape[0]} early_ready_experts={early}")


def test_duplicate_topk_ids_count_once():
    ids = torch.tensor([[1, 1, 1], [0, 0, 2]], dtype=torch.int32)
    expected = expected_token_counts(ids, num_experts=3)
    torch.testing.assert_close(expected, torch.tensor([1, 1, 1], dtype=torch.int64))


def test_dense_row_index_rank_major():
    assert dense_row_index(rank=3, local_row=7, m_local=64) == 3 * 64 + 7


def test_tp_dest_row_layouts():
    assert tp_dest_row(rank=3, local_row=7, m_local=64, npes=8) == 3 * 64 + 7
    assert tp_dest_row(rank=3, local_row=7, m_local=64, npes=8, row_major=True) == (
        7 * 8 + 3
    )
    # m_local=1: both layouts land on the same slot.
    assert tp_dest_row(rank=3, local_row=0, m_local=1, npes=8) == 3
    assert tp_dest_row(rank=3, local_row=0, m_local=1, npes=8, row_major=True) == 3


def test_publish_order_is_stable_min_expert_argsort():
    ids = torch.tensor([[4, 1], [0, 5], [1, 3]], dtype=torch.int32)
    torch.testing.assert_close(
        publish_order_from_topk(ids), torch.tensor([1, 0, 2], dtype=torch.int64)
    )


def test_pack_rows_uses_padding_row_for_sentinel():
    dense = torch.tensor([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]])
    # token 0, token 1, sentinel token_id=2 (n_tokens)
    sorted_ids = torch.tensor([0, 1, 2, (3 << 24) | 2], dtype=torch.int32)
    packed = pack_rows_by_sorted_ids(dense, sorted_ids, num_valid=4, n_tokens=2)
    torch.testing.assert_close(
        packed, torch.tensor([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0], [0.0, 0.0]])
    )


def test_make_tile_row_base_stride():
    base = make_tile_row_base(96, 32, device="cpu")
    torch.testing.assert_close(base, torch.tensor([0, 32, 64], dtype=torch.int32))


def test_chunk_mapping_front_loads_early_tiles():
    assert num_publish_chunks(64, 32) == 2
    assert num_publish_chunks(1, 32) == 1
    assert num_publish_chunks(256, 32) == 8
    assert chunk_of_m_tile(0, n_m=10, num_chunks=2) == 0
    assert chunk_of_m_tile(4, n_m=10, num_chunks=2) == 0
    assert chunk_of_m_tile(5, n_m=10, num_chunks=2) == 1
    assert chunk_of_m_tile(9, n_m=10, num_chunks=2) == 1


def test_chunk_gemm_overlap_wave2_does_not_wait():
    s64 = chunk_gemm_overlap_schedule(
        m_local=64, chunk_rows=32, gemm1_us=227.0, n_m=384, n_tiles=3
    )
    s256 = chunk_gemm_overlap_schedule(
        m_local=256, chunk_rows=32, gemm1_us=286.0, n_m=384, n_tiles=3
    )
    for sched in (s64, s256):
        assert sched["producers_wait_for_gemm"] is False
        assert sched["num_producers"] == 32
        assert sched["producer_cus"] == 32
        assert sched["consumer_cus_during_send"] == 224
        assert abs(sched["send_later_us"] - 9.68) < 1e-6
    assert s64["num_chunks"] == 2
    assert s256["num_chunks"] == 8
    assert s64["m_tiles_ready_after_wave0"] == 384 // 2
    assert s256["m_tiles_ready_after_wave0"] == 384 // 8
    # Later-wave send is ~10 µs; consumers finish ~11–15 experts in that window.
    # Wave 2 is not gated on GEMM — it starts as soon as wave 1 copy+signal ends.
    assert 10 < s64["experts_during_later_wave"] < 20
    assert 8 < s256["experts_during_later_wave"] < 15
    return s64, s256


def _schedule_row(name: str, ids: torch.Tensor, num_experts: int) -> dict:
    parts = partition_counts_by_min_expert(ids, num_experts)
    expected, _, _, first = simulate_incremental_publish(ids, num_experts)
    live = expected > 0
    early = (first >= 0) & (first < torch.arange(num_experts)) & live
    return {
        "case": name,
        "tokens": int(ids.shape[0]),
        "experts": num_experts,
        "topk": int(ids.shape[1]),
        "min_rounds_used": int((parts > 0).sum()),
        "live_experts": int(live.sum()),
        "early_ready_experts": int(early.sum()),
        "max_expected": int(expected.max()),
        "max_partition": int(parts.max()),
    }


def main():
    tests = [
        test_min_expert_is_row_min,
        test_partitions_are_disjoint_and_cover_all_tokens,
        test_each_token_published_once,
        test_expected_counts_match_token_membership,
        test_ready_after_own_round_and_prefix_invariant,
        test_early_ready_is_possible_for_later_experts,
        test_negative_skip_round_blocks_that_expert,
        test_v4_pro_shape_uniform_and_skew,
        test_duplicate_topk_ids_count_once,
        test_dense_row_index_rank_major,
        test_tp_dest_row_layouts,
        test_publish_order_is_stable_min_expert_argsort,
        test_pack_rows_uses_padding_row_for_sentinel,
        test_make_tile_row_base_stride,
        test_chunk_mapping_front_loads_early_tiles,
        test_chunk_gemm_overlap_wave2_does_not_wait,
    ]
    overlap = None
    for fn in tests:
        out = fn()
        print(f"ok  {fn.__name__}")
        if fn is test_chunk_gemm_overlap_wave2_does_not_wait:
            overlap = out
    if overlap is not None:
        print("\n## chunk vs GEMM1 overlap schedule\n")
        rows = []
        for name, sched in (("tokens=64 gemm1=227us", overlap[0]), ("tokens=256 gemm1=286us", overlap[1])):
            rows.append(
                {
                    "case": name,
                    "producers": sched["num_producers"],
                    "consumers_during_send": sched["consumer_cus_during_send"],
                    "waves": sched["num_chunks"],
                    "send_first_us": round(sched["send_first_us"], 2),
                    "send_later_us": round(sched["send_later_us"], 2),
                    "gemm_expert_us": round(sched["gemm_expert_us"], 3),
                    "gemm_expert_us_during_send": round(
                        sched["gemm_expert_us_during_send"], 3
                    ),
                    "experts_during_later_wave": round(
                        sched["experts_during_later_wave"], 2
                    ),
                    "wait_gemm": sched["producers_wait_for_gemm"],
                }
            )
        print(pd.DataFrame(rows).to_markdown(index=False))

    rows = [
        _schedule_row("small_uniform", _uniform_topk(128, 64, 6, 0), 64),
        _schedule_row("small_skew", _skew_topk(128, 64, 6, 1, hot=4), 64),
        _schedule_row(
            "v4_pro_uniform",
            _tp_concat_ids(
                V4_PRO["m_local"],
                V4_PRO["tp"],
                V4_PRO["num_experts"],
                V4_PRO["topk"],
                10,
            ),
            V4_PRO["num_experts"],
        ),
        _schedule_row(
            "v4_pro_skew",
            torch.cat(
                [
                    _skew_topk(
                        V4_PRO["m_local"],
                        V4_PRO["num_experts"],
                        V4_PRO["topk"],
                        20 + r,
                        hot=8,
                    )
                    for r in range(V4_PRO["tp"])
                ],
                dim=0,
            ),
            V4_PRO["num_experts"],
        ),
    ]
    df = pd.DataFrame(rows)
    print("\n## incremental schedule summary\n")
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
