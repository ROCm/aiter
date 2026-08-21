# SPDX-License-Identifier: Apache-2.0
"""Reference planner checks for compact Stage1 fanout sharing.

The GPU implementation deliberately keeps this model small: a route remains a
logical expert row, while one physical token/scale row may feed every expert in
the same selected ``(source, token, destination)`` fanout signature.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import pytest


@dataclass(frozen=True, order=True)
class Route:
    source: int
    token: int
    slot: int
    destination: int
    expert: int


@dataclass(frozen=True)
class Tile:
    destination: int
    expert: int
    signature: tuple[int, ...] | None
    logical_base: int
    input_base: int
    valid_rows: int
    padded_rows: int


@dataclass(frozen=True)
class FanoutPlan:
    selected: dict[int, tuple[tuple[int, ...], ...]]
    tiles: tuple[Tile, ...]
    logical_rows: dict[Route, int]
    physical_payload_rows: int
    logical_padded_rows: int


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _route_signatures(routes: tuple[Route, ...]):
    by_token: dict[tuple[int, int, int], list[Route]] = defaultdict(list)
    for route in routes:
        by_token[(route.source, route.token, route.destination)].append(route)

    signature_of: dict[Route, tuple[int, ...]] = {}
    signature_rows: dict[tuple[int, tuple[int, ...]], list[tuple[int, int]]] = (
        defaultdict(list)
    )
    for (source, token, destination), token_routes in by_token.items():
        experts = tuple(sorted(route.expert for route in token_routes))
        if len(experts) != len(set(experts)):
            raise ValueError("top-k must not route one token to the same expert twice")
        for route in token_routes:
            signature_of[route] = experts
        signature_rows[(destination, experts)].append((source, token))
    return signature_of, signature_rows


def _select_signatures(
    signature_rows: dict[tuple[int, tuple[int, ...]], list[tuple[int, int]]],
    *,
    max_groups_per_destination: int,
    min_saved_rows: int,
) -> dict[int, tuple[tuple[int, ...], ...]]:
    candidates: dict[int, list[tuple[int, tuple[int, ...]]]] = defaultdict(list)
    for (destination, signature), rows in signature_rows.items():
        saved_rows = (len(signature) - 1) * len(rows)
        if len(signature) > 1 and saved_rows >= min_saved_rows:
            candidates[destination].append((saved_rows, signature))

    selected = {}
    for destination, destination_candidates in candidates.items():
        destination_candidates.sort(key=lambda item: (-item[0], item[1]))
        selected[destination] = tuple(
            signature
            for _, signature in destination_candidates[:max_groups_per_destination]
        )
    return selected


def _build_plan(
    routes: tuple[Route, ...],
    *,
    tile_m: int,
    max_groups_per_destination: int,
    min_saved_rows: int,
) -> FanoutPlan:
    signature_of, signature_rows = _route_signatures(routes)
    selected = _select_signatures(
        signature_rows,
        max_groups_per_destination=max_groups_per_destination,
        min_saved_rows=min_saved_rows,
    )
    selected_sets = {
        destination: set(signatures) for destination, signatures in selected.items()
    }

    by_expert: dict[tuple[int, int], list[Route]] = defaultdict(list)
    for route in routes:
        by_expert[(route.destination, route.expert)].append(route)

    logical_rows: dict[Route, int] = {}
    tiles: list[Tile] = []
    section_base: dict[tuple[int, int, tuple[int, ...] | None], int] = {}
    logical_padded_rows = 0

    for destination, expert in sorted(by_expert):
        expert_routes = by_expert[(destination, expert)]
        signatures = selected.get(destination, ())
        sections: list[tuple[tuple[int, ...] | None, list[Route]]] = []
        for signature in signatures:
            if expert not in signature:
                continue
            section = sorted(
                route for route in expert_routes if signature_of[route] == signature
            )
            if section:
                sections.append((signature, section))
        normal = sorted(
            route
            for route in expert_routes
            if signature_of[route] not in selected_sets.get(destination, set())
        )
        if normal:
            sections.append((None, normal))

        for signature, section in sections:
            logical_base = logical_padded_rows
            padded_rows = _round_up(len(section), tile_m)
            section_base[(destination, expert, signature)] = logical_base
            for offset, route in enumerate(section):
                logical_rows[route] = logical_base + offset
            logical_padded_rows += padded_rows

    for destination, expert in sorted(by_expert):
        signatures = selected.get(destination, ())
        section_keys = [signature for signature in signatures if expert in signature]
        if any(
            signature_of[route] not in selected_sets.get(destination, set())
            for route in by_expert[(destination, expert)]
        ):
            section_keys.append(None)
        for signature in section_keys:
            key = (destination, expert, signature)
            if key not in section_base:
                continue
            logical_base = section_base[key]
            section_routes = [
                route
                for route in by_expert[(destination, expert)]
                if (signature is None)
                == (signature_of[route] not in selected_sets.get(destination, set()))
                and (signature is None or signature_of[route] == signature)
            ]
            input_base = logical_base
            if signature is not None:
                canonical_expert = min(signature)
                input_base = section_base[(destination, canonical_expert, signature)]
            tiles.append(
                Tile(
                    destination=destination,
                    expert=expert,
                    signature=signature,
                    logical_base=logical_base,
                    input_base=input_base,
                    valid_rows=len(section_routes),
                    padded_rows=_round_up(len(section_routes), tile_m),
                )
            )

    saved_rows = sum(
        (len(signature) - 1) * len(signature_rows[(destination, signature)])
        for destination, signatures in selected.items()
        for signature in signatures
    )
    return FanoutPlan(
        selected=selected,
        tiles=tuple(tiles),
        logical_rows=logical_rows,
        physical_payload_rows=len(routes) - saved_rows,
        logical_padded_rows=logical_padded_rows,
    )


def _make_routes(token_experts: tuple[tuple[int, ...], ...]) -> tuple[Route, ...]:
    routes = []
    for token, experts in enumerate(token_experts):
        for slot, expert in enumerate(experts):
            routes.append(
                Route(
                    source=token % 3,
                    token=token,
                    slot=slot,
                    destination=0,
                    expert=expert,
                )
            )
    return tuple(routes)


def _payload_task_pairs(
    *, segments_per_destination: int, chunks: int, producers: int
) -> list[tuple[int, int]]:
    """Mirror compact producer task decoding for one destination."""
    pairs = []
    task_limit = segments_per_destination * chunks
    for producer in range(producers):
        for task_index in range(producer, task_limit, producers):
            chunk_id = task_index // segments_per_destination
            rotated_segment = task_index - chunk_id * segments_per_destination
            rotation = chunk_id * 17 % segments_per_destination
            segment = (
                rotated_segment + segments_per_destination - rotation
            ) % segments_per_destination
            pairs.append((segment, chunk_id))
    return pairs


def test_selected_segments_support_pair_triple_and_quad_fanout():
    routes = _make_routes(
        (
            *((2, 30),) * 9,
            *((2, 20, 30),) * 7,
            *((2, 8, 20, 30),) * 5,
            *((4,),) * 3,
        )
    )
    plan = _build_plan(
        routes,
        tile_m=8,
        max_groups_per_destination=4,
        min_saved_rows=1,
    )

    assert plan.selected[0] == ((2, 8, 20, 30), (2, 20, 30), (2, 30))
    expected_saved = 7 * 2 + 5 * 3 + 9
    assert plan.physical_payload_rows == len(routes) - expected_saved
    assert set(plan.logical_rows) == set(routes)
    assert len(plan.logical_rows) == len(routes)

    for signature in plan.selected[0]:
        signature_tiles = [tile for tile in plan.tiles if tile.signature == signature]
        assert len(signature_tiles) == len(signature)
        assert len({tile.input_base for tile in signature_tiles}) == 1


@pytest.mark.parametrize("chunks", [1, 4, 5, 17, 32])
@pytest.mark.parametrize("producers", [1, 3, 4, 12])
def test_fanout_segment_chunk_tasks_are_a_bijection(chunks: int, producers: int):
    # v4-pro compact layout has 48 expert segments plus one shared fanout
    # segment.  Dividing task_index by the old expert count aliases the last
    # segment with the next chunk and leaves TILE_READY permanently short.
    segments = 49
    actual = _payload_task_pairs(
        segments_per_destination=segments,
        chunks=chunks,
        producers=producers,
    )
    expected = [
        (segment, chunk) for chunk in range(chunks) for segment in range(segments)
    ]
    assert len(actual) == len(expected)
    assert sorted(actual) == sorted(expected)


def test_unselected_signatures_remain_normal_without_semantic_change():
    routes = _make_routes(((1, 7, 11), (2, 8), (3,)))
    plan = _build_plan(
        routes,
        tile_m=8,
        max_groups_per_destination=4,
        min_saved_rows=1024,
    )
    assert plan.selected == {}
    assert plan.physical_payload_rows == len(routes)
    assert set(plan.logical_rows) == set(routes)
    assert all(tile.logical_base == tile.input_base for tile in plan.tiles)


@pytest.mark.parametrize("fanout", [2, 3, 4, 5, 6])
def test_one_physical_row_feeds_every_logical_expert(fanout: int):
    signature = tuple(range(fanout))
    routes = _make_routes((signature,) * 17)
    plan = _build_plan(
        routes,
        tile_m=8,
        max_groups_per_destination=1,
        min_saved_rows=1,
    )
    assert plan.physical_payload_rows == 17
    assert len(plan.logical_rows) == 17 * fanout
    selected_tiles = [tile for tile in plan.tiles if tile.valid_rows == 17]
    assert len(selected_tiles) == fanout
    assert len({tile.input_base for tile in selected_tiles}) == 1


def test_saved_rows_match_route_minus_unique_token_destination_count():
    routes = _make_routes(
        tuple((token % 7, (token + 1) % 7, (token + 2) % 7) for token in range(64))
    )
    plan = _build_plan(
        routes,
        tile_m=16,
        max_groups_per_destination=64,
        min_saved_rows=1,
    )
    unique_keys = {(route.source, route.token, route.destination) for route in routes}
    assert plan.physical_payload_rows == len(unique_keys)

    fanout_hist = Counter(
        len(signature)
        for (destination, signature), rows in _route_signatures(routes)[1].items()
        for _ in rows
        if destination == 0
    )
    assert fanout_hist == {3: 64}


def test_selected_mask_can_share_a_subset_of_larger_fanout():
    selected_mask = (1 << 2) | (1 << 30)
    signatures = ((2, 30), (2, 20, 30), (2, 8, 20, 30), (2, 20))
    physical_rows = 0
    logical_rows = 0
    for signature in signatures:
        token_mask = sum(1 << expert for expert in signature)
        matched = token_mask & selected_mask == selected_mask
        physical_rows += 1 + (
            len(signature) - selected_mask.bit_count()
            if matched
            else len(signature) - 1
        )
        logical_rows += len(signature)

    assert logical_rows == 11
    # The first three tokens share one physical row for experts 2 and 30;
    # routes outside the selected mask remain ordinary contiguous rows.
    assert physical_rows == 8
