# SPDX-License-Identifier: Apache-2.0
"""CPU protocol checks for compact Stage1 payload deduplication."""

from __future__ import annotations

import itertools
import random

import pytest


def _winner_slots(
    experts: tuple[int, ...], experts_per_rank: int, num_destinations: int = 8
) -> tuple[int, ...]:
    seen_destinations: set[int] = set()
    winners = []
    for slot, expert in enumerate(experts):
        if expert < 0 or expert >= experts_per_rank * num_destinations:
            continue
        destination = expert // experts_per_rank
        if destination not in seen_destinations:
            winners.append(slot)
            seen_destinations.add(destination)
    return tuple(winners)


@pytest.mark.parametrize("experts_per_rank", [48, 52, 56])
def test_first_slot_per_destination_is_the_unique_payload_winner(
    experts_per_rank: int,
):
    routes = (
        2,
        30,
        experts_per_rank + 1,
        31,
        2 * experts_per_rank + 7,
        experts_per_rank + 9,
    )
    winners = _winner_slots(routes, experts_per_rank)
    assert winners == (0, 2, 4)

    for destination, slots in itertools.groupby(
        sorted(range(len(routes)), key=lambda slot: routes[slot] // experts_per_rank),
        key=lambda slot: routes[slot] // experts_per_rank,
    ):
        grouped = tuple(slots)
        assert sum(slot in winners for slot in grouped) == 1, destination
        winner = next(slot for slot in grouped if slot in winners)
        assert winner == min(grouped)


def test_invalid_routes_do_not_suppress_a_valid_winner():
    routes = (-1, 9999, 51, 52, 103, 104)
    assert _winner_slots(routes, 52) == (2, 3, 5)


def test_unique_epoch_distinguishes_both_parities():
    npes = 8
    observed = []
    expected = [0, 0]
    parity = 0
    for _ in range(64):
        parity ^= 1
        expected[parity] += npes
        observed.append((expected[parity] // npes) * 2 - parity)
    assert observed == list(range(1, 65))


@pytest.mark.parametrize("dispatch_blocks", [64, 96, 128, 160, 192, 224])
def test_control_roles_are_resident_and_consumers_backfill(
    dispatch_blocks: int,
):
    num_cu = 256
    role_prefix = 1 + dispatch_blocks
    replacements = role_prefix
    launch_grid = num_cu + replacements

    assert role_prefix < num_cu
    first_resident = set(range(num_cu))
    assert set(range(role_prefix)) <= first_resident
    assert len(first_resident - set(range(role_prefix))) == num_cu - role_prefix
    assert launch_grid - num_cu == replacements
    assert launch_grid - role_prefix == num_cu


def test_route_headers_index_one_dense_payload_per_source_destination():
    experts_per_rank = 48
    routes = (2, 30, 49, 31, 97, 57)
    winners = _winner_slots(routes, experts_per_rank)
    source_key = 17
    unique_payload_by_destination = {
        routes[slot] // experts_per_rank: source_key for slot in winners
    }
    route_source_keys = tuple(
        unique_payload_by_destination[expert // experts_per_rank] for expert in routes
    )
    assert route_source_keys == (source_key,) * len(routes)


def test_sharded_entry_counters_advance_one_generation_per_launch():
    shards = 16
    launch_grids = tuple(
        256 * grid_mult + 1 + dispatch_blocks
        for grid_mult, dispatch_blocks in ((1, 64), (2, 64), (3, 128))
    )
    for launch_grid in launch_grids:
        populations = tuple(
            (launch_grid - 1 - shard) // shards + 1 for shard in range(shards)
        )
        assert sum(populations) == launch_grid
        counters = [0] * shards
        for generation in range(16):
            for shard, population in enumerate(populations):
                observed = {counters[shard] // population for _ in range(population)}
                assert observed == {generation}
                counters[shard] += population


@pytest.mark.parametrize("dispatch_blocks", [64, 96, 128, 160, 192, 224])
@pytest.mark.parametrize("total_work", [0, 1, 350, 351, 352, 10_980, 65_537])
def test_static_consumer_stride_covers_every_work_item_once(
    total_work: int, dispatch_blocks: int
):
    num_cu = 256
    launch_grid = num_cu + 1 + dispatch_blocks
    consumer_blocks = launch_grid - 1 - dispatch_blocks

    assert consumer_blocks == num_cu
    observed = []
    for consumer_slot in range(consumer_blocks):
        observed.extend(range(consumer_slot, total_work, consumer_blocks))

    assert sorted(observed) == list(range(total_work))
    assert len(observed) == len(set(observed))


def test_randomized_dense_source_key_mapping_is_complete():
    rng = random.Random(20260818)
    npes, experts_per_rank, tokens, topk = 8, 48, 257, 6
    routes_by_destination = [[] for _ in range(npes)]
    expected_unique = set()
    winner_slot_by_key = {}

    for token in range(tokens):
        experts = tuple(
            rng.randrange(npes // 2) * experts_per_rank
            + rng.randrange(experts_per_rank)
            for _ in range(topk)
        )
        winners = set(_winner_slots(experts, experts_per_rank, npes))
        for slot in winners:
            destination = experts[slot] // experts_per_rank
            winner_slot_by_key[(token, destination)] = slot
        for slot, expert in enumerate(experts):
            destination = expert // experts_per_rank
            routes_by_destination[destination].append((expert, token, slot))
            if slot in winners:
                expected_unique.add((token, destination))

    published_keys = set()
    for destination, routes in enumerate(routes_by_destination):
        routes.sort()
        for _, token, slot in routes:
            key = (token, destination)
            if slot == winner_slot_by_key[key]:
                published_keys.add(key)
        assert all((token, destination) in published_keys for _, token, _ in routes)

    assert published_keys == expected_unique
