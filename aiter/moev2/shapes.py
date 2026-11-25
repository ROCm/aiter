# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shape helpers shared by dispatch, execution and the reference.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
This module moved here from aiter/fused_moe.py and is shared with the legacy MoE path:
aiter.fused_moe re-exports it and legacy MoE calls run it. Changes must keep legacy
behavior (`python tests/moe_dispatch/snapshot.py --check`) and should loop in the moev2
owners (@qli88). See aiter/moev2/README.md.
"""

import functools


# Lru cache will using hash to create key, which makes error when w1,w2 shape is symint.
# We can use torch.compile(dynamic=False) to avoid
@functools.lru_cache(maxsize=2048)
def get_inter_dim(w1_shape, w2_shape):
    E, _, model_dim = w1_shape
    E, model_dim, inter_dim = w2_shape

    int4_war = model_dim // w1_shape[-1]
    inter_dim *= int4_war
    return E, model_dim, inter_dim


def _is_row_major_2d(tensor):
    return (
        tensor.ndim == 2
        and tensor.stride(1) == 1
        and (
            tensor.shape[0] == 1
            or (
                tensor.stride(0) >= tensor.shape[1]
                and tensor.stride(0) * tensor.element_size() % 16 == 0
            )
        )
    )


def nextPow2(n):
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


_PADDED_M_TIERS = [32768, 131072]


def get_padded_M(M):
    """Map runtime M to a tuning bucket.

    Values below 32768 use the next power of two, 32768..131071 use the 32768
    tier, and values from 131072 upward use the 131072 tier. Individual model
    contracts may impose a stricter maximum before lookup.
    """
    if M < _PADDED_M_TIERS[0]:
        return nextPow2(M)
    for tier in reversed(_PADDED_M_TIERS):
        if M >= tier:
            return tier
    return _PADDED_M_TIERS[0]
