# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""QSA index scorer (paged MQA logits): op contract v1, fused FlyDSL gfx1250 + native torch.

``qsa_paged_mqa_logits`` serves every valid call: the fused FlyDSL kernel (index page
4, 8 or 16) when a profile covers the call, otherwise aiter's native pure-torch
implementation (``native.torch.v1``). The fused path is off by default: nothing fused
runs (and FlyDSL is not imported) unless a profile is named via ``profile=`` or
``$AITER_QSA_INDEX_SCORER_PROFILE``. The public op ``aiter.qsa_paged_mqa_logits``
(:mod:`.op`) has exactly ATOM's 12-parameter signature. See
``docs/qsa_index_scorer.md`` for the contract sheet, profile format and limitations.
"""

from .api import (  # noqa: F401
    ATOM_SCORER_SIGNATURE,
    NATIVE_VID,
    PATH_FUSED,
    PATH_INVALID,
    PATH_NATIVE,
    Decision,
    QSAIndexScorerUnsupported,
    compile_cache_dir,
    decide,
    eligibility,
    kernel_page_sizes,
    launch,
    layout_eligibility,
    prewarm,
    prewarm_plan,
    prewarm_stats,
    qsa_paged_mqa_logits,
    selected_profile,
    try_qsa_paged_mqa_logits,
)
from .native import NativeContractError  # noqa: F401
from .op import OP_CONTRACT_VERSION, OP_NAME  # noqa: F401
from .profiles import (  # noqa: F401
    API_VERSION,
    BACKEND,
    DEFAULT_PROFILE,
    FEATURE,
    PROFILE_API_VERSION,
    PROFILE_ENV,
    STATIC_REASONS,
    Profile,
    ProfileError,
    Reason,
    load_profile,
    regime_of,
)
