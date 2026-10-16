# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The fused layer kernel's hand-off contract, checked on a build.

Tracing a build (compile-only, no launch, no GPU needed with ``ARCH`` set) records
its mailbox accesses (``atom.mono.plan.trace``); ``check_traced`` holds them to
``layout.mailbox_regions``. A build served from FlyDSL's disk cache is not traced,
so the caller disables the cache (``FLYDSL_RUNTIME_ENABLE_CACHE=0``).

Usage: python -m atom.models.minimax_m3.mono.contract [S ...]
"""

import sys

from atom.models.minimax_m3.mono.config import TP, IndexHeads
from atom.models.minimax_m3.mono.kernels.dense_post import (
    DENSE_POST_ABI,
    build_dense_post_kernel,
)
from atom.models.minimax_m3.mono.kernels.post_attn import K4_ABI, build_post_attn_kernel
from atom.models.minimax_m3.mono.layout import dense_mailbox_regions, mailbox_regions
from atom.mono.runtime.compile import check_traced


def build_layer(
    tokens: int, index_heads: int, fuse_k1: bool = True, index_topk: bool = True
):
    """The layer kernel for ``tokens`` and ``index_heads`` (1 or TP), K1 fused in
    or not (``fuse_k1``), selecting or reusing a selection (``index_topk``)."""
    heads = IndexHeads(index_heads, 0)
    return build_post_attn_kernel(
        TP, 0.088, 1e-6, 1.0, 1.0, 7.0, 1, 1, tokens, fuse_k1=fuse_k1, heads=heads,
        index_topk=index_topk,
    )  # fmt: skip


def build_dense(tokens: int):
    """The dense layer kernel (dense_post) for ``tokens``."""
    return build_dense_post_kernel(TP, 1e-6, 1.702, 1.0, 7.0, tokens)


def check_build(
    tokens: int, index_heads: int, fuse_k1: bool = True, index_topk: bool = True
) -> None:
    """Trace ``build_layer`` and check its hand-offs; raises ``ContractError``."""
    check_traced(
        build_layer(tokens, index_heads, fuse_k1, index_topk),
        K4_ABI,
        mailbox_regions(tokens, index_heads, fuse_k1, index_topk),
    )


def check_dense_build(tokens: int) -> None:
    """Trace ``build_dense`` and check its hand-offs; raises ``ContractError``."""
    check_traced(build_dense(tokens), DENSE_POST_ABI, dense_mailbox_regions(tokens))


if __name__ == "__main__":
    sizes = [int(a) for a in sys.argv[1:]] or list(range(1, 17))
    for s in sizes:
        for ih in (1, TP):
            for it in (True, False):
                check_build(s, ih, index_topk=it)
                print(f"S={s} index heads={ih} index_topk={it}: contract holds")
