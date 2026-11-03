# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""kpool cache writers for the GLM-5.3-Flash sparse-attention indexer.

The indexer caches one fp8 key per *pool* of ``pool_size`` consecutive tokens:
a softmax(gate + ape)-weighted mean of the pool's K rows, rounded to bf16,
Hadamard-128 rotated, rounded to bf16, and quantized to fp8 e4m3fn with one
ue8m0 scale. ``kv_cache`` is ``[num_blocks, page_size, 132]`` uint8: each page
holds ``page_size * 128`` key bytes (the MFMA 16x16 preshuffled layout of
``indexer_k_quant_and_cache`` when ``page_size > 1``) followed by
``page_size`` fp32 scales.

Three writers, byte-identical to the vLLM Triton path:

``kpool_compress_insert``  prefill: K / gate read in place, one window of
                           ``pool_size`` tokens per lane group, only closing
                           pools (``slot_mapping >= 0``) are compressed.
``kpool_decode_update``    decode: each token's K / gate stashed into the
                           request's tail ring; the fourth token closes the
                           pool from the ring. Deterministic, next_n <= 16.
``kpool_mixed_write``      insert + tail seeding + decode in one launch.
"""

from torch import Tensor

from ..jit.core import compile_ops

MD_NAME = "module_kpool_compress"
KPOOL_HEAD_DIM = 128


def kpool_pick_lanes(n_tokens: int, pool_size: int) -> int:
    """Lanes per pool the insert kernel picks for ``n_tokens`` (``lanes_per_pool=0``):
    64 up to 8192 pools, 32 up to 65536, 16 above. Tuned on MI355X, pool_size 4."""
    n_pools = n_tokens // pool_size
    if n_pools <= 8192:
        return 64
    if n_pools <= 65536:
        return 32
    return 16


@compile_ops(MD_NAME, develop=True)
def kpool_compress_insert(
    kv_cache: Tensor,  # [num_blocks, page_size, 132] uint8
    k: Tensor,  # [n_tokens, 128] bf16, rows may be strided
    gate_score: Tensor,  # [n_tokens, 128] bf16
    ape: Tensor,  # [pool_size, 128] fp32
    slot_mapping: Tensor,  # [n_tokens] int32 / int64, >= 0 on closing tokens
    pool_size: int,
    round_scale: bool = True,
    lanes_per_pool: int = 0,  # 0: auto, else 16 / 32 / 64
) -> None: ...


@compile_ops(MD_NAME, develop=True)
def kpool_decode_update(
    kv_cache: Tensor,  # [num_blocks, page_size, 132] uint8
    tail_kv_cache: Tensor,  # [num_blocks, 2, ring, 128] bf16, ring % pool_size == 0
    tail_slot_mapping: Tensor,  # [num_requests, next_n] int32 / int64 (all three metadata tensors alike)
    key: Tensor,  # [num_requests, next_n, 128] bf16
    slot_score: Tensor,  # [num_requests, next_n, 128] bf16
    ape: Tensor,  # [pool_size, 128] fp32
    slot_mapping: Tensor,  # [num_requests, next_n] int32 / int64, >= 0 on closing tokens
    positions: Tensor,  # [num_requests, next_n] int32 / int64
    pool_size: int,
    round_scale: bool = True,
) -> None: ...


@compile_ops(MD_NAME, develop=True)
def kpool_mixed_write(
    kv_cache: Tensor,
    ape: Tensor,
    pool_size: int,
    round_scale: bool = True,
    lanes_per_pool: int = 0,
    prefill_k: Tensor | None = None,
    prefill_gate: Tensor | None = None,
    prefill_slot_mapping: Tensor | None = None,
    prefill_tail_slot_mapping: (
        Tensor | None
    ) = None,  # [n_tokens], dtype of prefill_slot_mapping: seed the tail ring
    tail_kv_cache: Tensor | None = None,
    # decode metadata: int32 or int64, the dtype of prefill_slot_mapping when both sides are given
    decode_tail_slot_mapping: Tensor | None = None,
    decode_key: Tensor | None = None,
    decode_gate: Tensor | None = None,
    decode_slot_mapping: Tensor | None = None,
    decode_positions: Tensor | None = None,
) -> None: ...
