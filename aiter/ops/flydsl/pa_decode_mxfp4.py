# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Entry of ``pa_decode`` for an MXFP4 KV cache passed as K/V data and scale views."""

from .pa_decode import pa_decode


def pa_decode_mxfp4(
    output,
    query,
    k_data,
    k_scale,
    v_data,
    v_scale,
    context_lengths,
    block_tables,
    softmax_scale,
    *,
    work_plan=None,
    max_partitions=None,
    **kwargs,
):
    """``pa_decode`` with ``query_length`` 1. ``max_partitions`` may only be 1, and
    only without a ``work_plan``."""
    if max_partitions not in (None, 1) or (work_plan and max_partitions):
        raise ValueError("max_partitions may only be 1, and only without a work_plan")
    pa_decode(
        output,
        query,
        k_data,
        v_data,
        context_lengths,
        block_tables,
        softmax_scale,
        1,
        key_scale=k_scale,
        value_scale=v_scale,
        work_plan=work_plan,
        **kwargs,
    )
