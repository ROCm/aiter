# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4 (flash/pro) attention-block kernels.

Triton ports of ATOM's `atom/model_ops/v4_kernels/*` so the full DSV4 sparse-MLA
attention block (MLA projections + RoPE + sliding-window KV + indexer top-k +
sparse paged attention) can be exercised end to end from aiter alone.
"""
