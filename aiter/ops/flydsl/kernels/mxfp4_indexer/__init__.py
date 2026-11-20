# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL kernels of the MXFP4 sparse-attention indexer top-512 (gfx950):
``scores`` (the walk: block maxima or token logits), ``select`` (block select
and rescoring, or the candidate gather select), ``tokens`` (radix select from
token logits), ``blocks`` (long-row block select and merge), ``cand_logits``
(candidate token scorer). Public entry points:
``aiter.ops.flydsl.mxfp4_indexer_topk``."""
