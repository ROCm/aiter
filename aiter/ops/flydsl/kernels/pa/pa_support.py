# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Host-side helpers ``pa_decode_tile`` imports from FlyDSL's ``pa_decode_fp8``.

Upstream these four live in ``kernels/attention/pa_decode_fp8.py``, which also
pulls in ``pa_metadata`` (61 KB) and ``pa_metadata_tuning`` for the persistent-
scheduling launcher. The vendored paged-decode path never uses that launcher, so
the helpers are reproduced here verbatim instead and the cascade is cut.

Keep these byte-identical to upstream when re-syncing ``pa_decode_tile.py``.
"""

import torch

from aiter.ops.flydsl.kernels.pa.utils import cdiv

# Tile size, matches SP3 kTileKV. Used to size a KV split in blocks.
KV_COMPUTE_BLOCK = 256


def _is_current_stream_capturing() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_current_stream_capturing()
    except RuntimeError:
        return False


def _get_output_dtype_str(output: torch.Tensor) -> str:
    if output.dtype == torch.bfloat16:
        return "bf16"
    if output.dtype == torch.float16:
        return "f16"
    if output.dtype == torch.float32:
        return "f32"
    raise ValueError(
        f"Unsupported output dtype for pa_decode_ps_launch reduce: {output.dtype}. "
        "Expected bf16, f16, or f32."
    )


def get_recommended_splits(
    num_sequences: int,
    num_kv_heads: int,
    split_kv_blocks: int = 1,
    *,
    sliding_window: int = 0,
    context_partition_size: int = KV_COMPUTE_BLOCK,
    query_length: int = 1,
) -> int:
    """Recommend ``max_context_partition_num`` for PS partitioned paths.

    For sliding-window PS, this includes the old
    ``get_sw_ps_max_context_partition_num`` token-window calculation. For
    non-sliding PS, this mirrors ``get_recommended_splits`` in
    ``aiter/ops/triton/gluon/pa_decode_gluon.py``.
    """
    if sliding_window > 0:
        window_token_count = sliding_window + query_length
        return cdiv(window_token_count - 1, context_partition_size) + 1

    props = torch.cuda.get_device_properties(torch.device("cuda"))
    # Reference uses occupancy = 2 (see `get_occupancy()` in the Gluon module).
    occupancy = 2
    num_sm = props.multi_processor_count * occupancy
    denom = max(1, num_sequences * num_kv_heads * split_kv_blocks)
    n = cdiv(num_sm, denom) * split_kv_blocks
    return max(4, min(n, 8))
