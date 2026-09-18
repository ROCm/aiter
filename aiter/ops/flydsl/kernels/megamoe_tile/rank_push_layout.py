"""Private, generation-buffered storage for rank-reduce/push Stage2.

The output and staging share one allocation, with disjoint byte ranges. The
existing output pointer consequently carries the private workspace into the
kernel without a host metadata copy or an additional Graph node. Only the
output prefix is exposed as a Tensor. Peer-visible inbox storage is separate.
"""

from dataclasses import dataclass


def _align(value, alignment=256):
    return (value + alignment - 1) // alignment * alignment


def node_reduce_token_owner_supported(*, max_tokens, node_reduce_blocks):
    """Return whether the static node-reduce schedule has one wave per token.

    The non-bitmap reducer enumerates ``n_group * (2 * max_tokens) + token``
    with a stride of four waves per CTA.  If the token plane is divisible by
    that stride, every N group for one token maps to the same wave and arrives
    there in increasing group order.  That wave can publish token readiness
    after its last group without an inter-wave ready-mask reduction.
    """
    max_tokens = int(max_tokens)
    node_reduce_blocks = int(node_reduce_blocks)
    if max_tokens <= 0 or node_reduce_blocks <= 0:
        raise ValueError("token-owner geometry must be positive")
    return (2 * max_tokens) % (4 * node_reduce_blocks) == 0


def node_reduce_token_owner_mask_prefetch_supported(
    *, max_tokens, node_reduce_blocks, wave_size=64
):
    """Return whether one wave can retain all of its owned token masks.

    The owner schedule assigns ``2 * max_tokens / (4 * node_reduce_blocks)``
    tokens to each wave.  The prefetch path stores one immutable rank mask in
    each lane and later selects the current token with ``readlane``, so the
    owned-token count must fit in one wave.
    """
    max_tokens = int(max_tokens)
    node_reduce_blocks = int(node_reduce_blocks)
    wave_size = int(wave_size)
    if wave_size <= 0:
        raise ValueError("token-owner mask prefetch wave size must be positive")
    if not node_reduce_token_owner_supported(
        max_tokens=max_tokens, node_reduce_blocks=node_reduce_blocks
    ):
        return False
    return (2 * max_tokens) // (4 * node_reduce_blocks) <= wave_size


@dataclass(frozen=True)
class RankPushWorkspace:
    output_bytes: int
    workspace_offset: int
    payload_bytes: int
    row_map_offset: int
    row_map_bytes: int
    arrival_offset: int
    arrival_bytes: int
    parity_stride: int
    total_bytes: int

    @classmethod
    def create(cls, *, max_tokens, hidden, topk, max_route_rows, world_size=16,
               group_columns=512, parity_depth=2):
        if min(max_tokens, hidden, topk, max_route_rows, world_size) <= 0:
            raise ValueError("rank push workspace dimensions must be positive")
        if hidden % group_columns or parity_depth != 2:
            raise ValueError("rank push requires complete N groups and two parities")
        sources = world_size * max_tokens
        output_bytes = max_tokens * hidden * 2
        payload_bytes = max_route_rows * hidden * 2
        # The vector buffer operations use signed i32 byte offsets. Reject an
        # oversized configuration instead of wrapping under high route capacity.
        if payload_bytes >= 2**31:
            raise ValueError("rank push route payload exceeds signed 32-bit buffer offsets")
        row_map_offset = _align(payload_bytes)
        row_map_bytes = sources * topk * 4
        arrival_offset = _align(row_map_offset + row_map_bytes)
        arrival_bytes = sources * (hidden // group_columns) * 4
        stride = _align(arrival_offset + arrival_bytes)
        prefix = _align(output_bytes)
        return cls(output_bytes, prefix, payload_bytes, row_map_offset,
                   row_map_bytes, arrival_offset, arrival_bytes, stride,
                   prefix + parity_depth * stride)


def summarize_rank_push_protocol(*, packed_sources, source_capacity, topk,
                                 ready_groups, row_map, route_counts,
                                 local_arrivals, node_rank_masks, node_arrivals,
                                 node_publication="per_tile_counter",
                                 rank_push_batch_size=1):
    """Check a completed parity against actual sorted rows, outside timing.

    The original TopK slot is part of the key: duplicate experts with distinct
    slot weights remain distinct contributions. Check inactive sources and
    absent node contributors as well, so a stale parity cannot pass by checking
    only the current active subset. No balanced-routing count is assumed.
    """
    for name, values, count in (
        ("row_map", row_map, source_capacity * topk),
        ("route_counts", route_counts, source_capacity),
        ("local_arrivals", local_arrivals, source_capacity * ready_groups),
        ("node_arrivals", node_arrivals, len(node_rank_masks) * ready_groups),
    ):
        if len(values) != count:
            raise ValueError(f"rank push {name} has {len(values)} entries, expected {count}")
    expected_map = [0] * (source_capacity * topk)
    expected_counts = [0] * source_capacity
    invalid_keys = duplicate_keys = 0
    for row, packed in enumerate(packed_sources):
        packed = int(packed)
        if packed == source_capacity:  # Stage1 INVALID_SOURCE, with slot zero.
            continue
        source, slot = packed & 0x00FFFFFF, (packed >> 24) & 0xFF
        if source >= source_capacity or slot >= topk:
            invalid_keys += 1
            continue
        key = source * topk + slot
        duplicate_keys += int(expected_map[key] != 0)
        expected_map[key] = row + 1
        expected_counts[source] += 1
    if node_publication == "per_tile_counter":
        node_arrival_mismatch = sum(
            int(actual) != (int(node_rank_masks[index // ready_groups]) & 0xFF).bit_count()
            for index, actual in enumerate(node_arrivals)
        )
    elif node_publication in ("batch_bitmap", "packed_tile_bitmap"):
        # Each word packs four adjacent tokens' 8-bit contributor masks. The
        # batch mode consumes both words together for B8. The packed per-tile
        # mode may use a larger payload batch, but checks each selected byte
        # independently. Words live at token slots divisible by four; all
        # other per-token arrival entries remain zero.
        if node_publication == "batch_bitmap" and rank_push_batch_size != 8:
            raise ValueError("batch_bitmap publication requires rank_push_batch_size=8")
        if node_publication == "packed_tile_bitmap" and (
            rank_push_batch_size < 4 or rank_push_batch_size % 4
        ):
            raise ValueError(
                "packed_tile_bitmap publication requires rank_push_batch_size "
                "divisible by 4"
            )
        if len(node_rank_masks) % rank_push_batch_size:
            raise ValueError("node rank masks must contain complete publication batches")
        expected_node_arrivals = [0] * len(node_arrivals)
        publication_group = 8 if node_publication == "batch_bitmap" else 4
        for token_base in range(0, len(node_rank_masks), publication_group):
            for n_group in range(ready_groups):
                for half in range(publication_group // 4):
                    word = 0
                    for slot in range(4):
                        mask = int(node_rank_masks[token_base + half * 4 + slot]) & 0xFF
                        word |= mask << (slot * 8)
                    index = (token_base + half * 4) * ready_groups + n_group
                    expected_node_arrivals[index] = word
        node_arrival_mismatch = sum(
            (int(actual) & 0xFFFFFFFF) != expected
            for actual, expected in zip(node_arrivals, expected_node_arrivals)
        )
    else:
        raise ValueError(f"unsupported rank push node publication {node_publication!r}")

    return {
        "rank_push_route_count": sum(expected_counts),
        "rank_push_active_sources": sum(count > 0 for count in expected_counts),
        "rank_push_invalid_key_count": invalid_keys,
        "rank_push_duplicate_key_count": duplicate_keys,
        "rank_push_row_map_mismatch": sum(
            int(actual) != expected for actual, expected in zip(row_map, expected_map)
        ),
        "rank_push_route_count_mismatch": sum(
            int(actual) != expected for actual, expected in zip(route_counts, expected_counts)
        ),
        "rank_push_local_arrival_mismatch": sum(
            int(actual) != expected_counts[index // ready_groups]
            for index, actual in enumerate(local_arrivals)
        ),
        "rank_push_node_mask_invalid": sum(int(mask) & ~0xFF != 0 for mask in node_rank_masks),
        "rank_push_node_arrival_mismatch": node_arrival_mismatch,
    }
