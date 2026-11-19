# SPDX-License-Identifier: MIT
"""Opt-in contiguous GEMM2 landing layout for source-aligned pull tiles."""

import os


def enabled(config):
    """Freeze the experimental selection when a layer is constructed."""
    return (
        config.stage2_fused
        and os.environ.get("AITER_MEGA_DISPATCH_PULL", "0") == "1"
        and os.environ.get("AITER_MEGA_EP_LINEAR_STORE", "0") == "1"
    )


def capacity(config):
    """Each source reserves at most 255 padding rows per global expert."""
    rows = config.max_tokens_per_rank * config.topk
    rows += config.world_size * config.experts_per_rank * 255
    return (rows + 255) // 256 * 256
