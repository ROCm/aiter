# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host helpers shared by FlyDSL communication operators."""

from __future__ import annotations

import mori.shmem as ms
import torch


def build_p2p_table(tensor, rank, world_size, device) -> torch.Tensor:
    """i64 table of intra-node P2P pointers to ``tensor`` on every peer."""
    table = torch.zeros(world_size, dtype=torch.int64, device=device)
    for peer in range(world_size):
        ptr = ms.shmem_ptr_p2p(tensor.data_ptr(), rank, peer)
        if not ptr:
            raise ValueError(
                f"shmem_ptr_p2p returned a null pointer for peer {peer} "
                "(no intranode P2P path, or tensor is not in the symmetric heap)"
            )
        table[peer] = ptr
    return table
