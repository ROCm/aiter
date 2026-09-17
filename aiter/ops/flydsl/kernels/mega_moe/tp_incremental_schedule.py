# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Host-side schedule for TP incremental activation all-gather.

Each token is published once, in ``min(topk)`` order. Expert ``e`` is ready
when every token that selected ``e`` has been published, which is guaranteed
after the prefix of rounds ``0..e`` and may happen earlier.
"""

from __future__ import annotations

import torch

TOKEN_ID_MASK = 0x00FFFFFF


def dense_row_index(rank: int, local_row: int, m_local: int) -> int:
    """Rank-major slot in the dense gathered A buffer."""
    return int(rank) * int(m_local) + int(local_row)


def tp_dest_row(
    rank: int, local_row: int, m_local: int, npes: int, row_major: bool = False
) -> int:
    """Push destination in the dense TP slab.

    Rank-major (default) is ``rank * m_local + local_row``. Row-major
    ``local_row * npes + rank`` is the Step 3 negative control.
    """
    if row_major:
        return int(local_row) * int(npes) + int(rank)
    return dense_row_index(rank, local_row, m_local)


def publish_order_from_topk(topk_ids: torch.Tensor) -> torch.Tensor:
    """Stable argsort of ``min(topk)`` — each token published once."""
    return min_expert_ids(topk_ids).argsort(stable=True)


def min_expert_ids(topk_ids: torch.Tensor) -> torch.Tensor:
    """Per-token first-publish expert: ``min(topk)`` along the last dim."""
    if topk_ids.ndim != 2:
        raise ValueError(f"topk_ids must be [T, K], got {tuple(topk_ids.shape)}")
    return topk_ids.min(dim=-1).values


def _expert_presence(topk_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    """``[T, E]`` bool: token selected expert (unique even if topk repeats)."""
    if topk_ids.numel():
        lo = int(topk_ids.min())
        hi = int(topk_ids.max())
        if lo < 0:
            raise ValueError("topk_ids must be non-negative expert ids")
        if hi >= num_experts:
            raise ValueError(f"topk_ids max {hi} >= num_experts={num_experts}")
    tokens = topk_ids.shape[0]
    presence = torch.zeros(
        (tokens, num_experts), dtype=torch.bool, device=topk_ids.device
    )
    if tokens:
        presence.scatter_(1, topk_ids.to(torch.int64), True)
    return presence


def expected_token_counts(topk_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Number of tokens that selected each expert (not padded GEMM rows)."""
    return _expert_presence(topk_ids, num_experts).sum(dim=0, dtype=torch.int64)


def partition_counts_by_min_expert(
    topk_ids: torch.Tensor, num_experts: int
) -> torch.Tensor:
    """How many tokens are first published in each expert round."""
    mins = min_expert_ids(topk_ids).to(torch.int64)
    counts = torch.zeros(num_experts, dtype=torch.int64, device=topk_ids.device)
    ones = torch.ones_like(mins)
    counts.scatter_add_(0, mins, ones)
    return counts


def simulate_incremental_publish(topk_ids: torch.Tensor, num_experts: int):
    """Replay min-expert publish order.

    Returns
    -------
    expected:
        ``[E]`` tokens that selected each expert.
    received_after_round:
        ``[E, E]``; row ``e`` is ``received`` after publishing ``min == e``.
    ready_after_round:
        ``[E, E]`` bool; expert ``k`` is ready after round ``e``.
    first_ready_round:
        ``[E]``; ``-1`` means ready before any publish (``expected == 0``).
    """
    expected = expected_token_counts(topk_ids, num_experts)
    mins = min_expert_ids(topk_ids)
    received = torch.zeros(num_experts, dtype=torch.int64, device=topk_ids.device)
    received_after_round = torch.zeros(
        (num_experts, num_experts), dtype=torch.int64, device=topk_ids.device
    )
    first_ready_round = torch.full(
        (num_experts,), -1, dtype=torch.int64, device=topk_ids.device
    )
    already_ready = expected == 0
    for round_e in range(num_experts):
        in_round = mins == round_e
        if bool(in_round.any()):
            presence = _expert_presence(topk_ids[in_round], num_experts)
            received += presence.sum(dim=0, dtype=torch.int64)
        received_after_round[round_e] = received
        newly_ready = (received == expected) & (~already_ready)
        first_ready_round[newly_ready] = round_e
        already_ready = already_ready | newly_ready
    ready_after_round = received_after_round == expected.unsqueeze(0)
    return expected, received_after_round, ready_after_round, first_ready_round


def tokens_needed_by_expert(topk_ids: torch.Tensor, expert: int) -> torch.Tensor:
    """Boolean mask over tokens that selected ``expert``."""
    return (topk_ids == expert).any(dim=-1)


def decode_sorted_token_ids(sorted_ids: torch.Tensor) -> torch.Tensor:
    """Low 24 bits of moe_sorting fused ids: ``topk_slot << 24 | token``."""
    return sorted_ids.to(torch.int64) & TOKEN_ID_MASK


def pack_rows_by_sorted_ids(
    dense: torch.Tensor,
    sorted_ids: torch.Tensor,
    num_valid: int,
    n_tokens: int,
) -> torch.Tensor:
    """Gather dense rows into sorted-row order.

    ``dense`` must have a padding row at index ``n_tokens``. Out-of-range fused
    ids (moe_sorting sentinels) clamp onto that row.
    """
    tok = decode_sorted_token_ids(sorted_ids[:num_valid]).clamp(max=n_tokens)
    return dense.index_select(0, tok)


def num_publish_chunks(m_local: int, chunk_rows: int) -> int:
    """How many ordered all-gather sends cover ``m_local`` rows."""
    m_local = int(m_local)
    chunk_rows = int(chunk_rows)
    if chunk_rows <= 0:
        raise ValueError(f"chunk_rows must be positive, got {chunk_rows}")
    if m_local <= 0:
        return 0
    return (m_local + chunk_rows - 1) // chunk_rows


def chunk_of_m_tile(m_tile: int, n_m: int, num_chunks: int) -> int:
    """Map a GEMM M-tile onto a constructed send chunk (early-compute upper bound)."""
    n_m = int(n_m)
    num_chunks = int(num_chunks)
    if n_m <= 0 or num_chunks <= 0:
        raise ValueError("n_m and num_chunks must be positive")
    return min(int(m_tile) * num_chunks // n_m, num_chunks - 1)


def chunk_gemm_overlap_schedule(
    *,
    m_local: int,
    chunk_rows: int,
    gemm1_us: float,
    n_m: int,
    n_tiles: int = 3,
    num_cu: int = 256,
    num_producers: int = 32,
    send_fixed_us: float = 38.0,
    send_us_per_row: float = 0.24,
    send_handshake_us: float = 2.0,
):
    """Host model of chunked AG vs GEMM1. Producers do not wait on GEMM.

    Measured H=7168 8-GPU defaults: ~38 µs kernel/setup, ~0.24 µs/row copy,
    ~2 µs handshake per wave. ``gemm1_us`` is standalone token-id GEMM1 for
    this shape (all CUs). Padded routing makes one M-tile per live expert.

    Wave 2 starts when producers finish wave 1, not after N experts.
    ``experts_during_later_wave`` is how many experts consumers finish in one
    later-wave send interval while ``num_producers`` CTAs are still sending.
    """
    n_m = int(n_m)
    n_tiles = int(n_tiles)
    num_chunks = num_publish_chunks(m_local, chunk_rows)
    if n_m <= 0 or n_tiles <= 0 or num_chunks <= 0:
        raise ValueError("n_m, n_tiles, and num_chunks must be positive")
    rows_per_wave = min(int(chunk_rows), int(m_local))
    send_copy_us = float(send_us_per_row) * rows_per_wave
    send_later_us = send_copy_us + float(send_handshake_us)
    send_first_us = float(send_fixed_us) + send_later_us
    producer_cus = min(int(num_producers), int(num_cu))
    consumer_cus = max(int(num_cu) - producer_cus, 1)
    gemm_expert_us = float(gemm1_us) / n_m
    gemm_expert_us_during_send = gemm_expert_us * int(num_cu) / consumer_cus
    experts_during_later = send_later_us / gemm_expert_us_during_send
    m_tiles_ready_after_wave0 = sum(
        1 for m_tile in range(n_m) if chunk_of_m_tile(m_tile, n_m, num_chunks) == 0
    )
    return {
        "num_chunks": num_chunks,
        "num_producers": int(num_producers),
        "producer_cus": producer_cus,
        "consumer_cus_during_send": consumer_cus,
        "send_copy_us": send_copy_us,
        "send_first_us": send_first_us,
        "send_later_us": send_later_us,
        "gemm_expert_us": gemm_expert_us,
        "gemm_expert_us_during_send": gemm_expert_us_during_send,
        "m_tiles_ready_after_wave0": m_tiles_ready_after_wave0,
        "experts_during_later_wave": experts_during_later,
        "tiles_during_later_wave": experts_during_later * n_tiles,
        "producers_wait_for_gemm": False,
    }


def make_tile_row_base(num_valid: int, sort_block_m: int, device=None) -> torch.Tensor:
    """Contiguous M-tile starts: ``[0, 32, 64, ...]`` for packed / sorted rows."""
    if num_valid % int(sort_block_m):
        raise ValueError(
            f"num_valid={num_valid} must be a multiple of sort_block_m={sort_block_m}"
        )
    n_m = num_valid // int(sort_block_m)
    return torch.arange(n_m, dtype=torch.int32, device=device) * int(sort_block_m)
