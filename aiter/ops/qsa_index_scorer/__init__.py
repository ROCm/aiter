# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Private pure-torch implementation of ``aiter.ops.qsa.qsa_paged_mqa_logits`` (not a
public API).

The routed op, its contract and its native dispatch live in :mod:`aiter.ops.qsa`
(``docs/qsa_index_scorer.md``). :mod:`.native` serves the valid calls aiter's existing
kernels (Triton, FlyDSL page-16 prefill) cannot take, e.g. CPU tensors. Tuned
implementations of the op are out-of-tree providers (``aiter.plugin``).
"""

from . import native  # noqa: F401
from .native import NativeContractError  # noqa: F401
