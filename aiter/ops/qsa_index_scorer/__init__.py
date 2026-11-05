# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""QSA index scorer (paged MQA logits, index page 4, 8 or 16) -- FlyDSL gfx1250 backend.

Off by default: nothing runs (and FlyDSL is not imported) unless a profile is
named via ``profile=`` or ``$AITER_QSA_INDEX_SCORER_PROFILE``. See
``docs/qsa_index_scorer.md`` for the contract, profile format and known limitations.
"""

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
from .api import (  # noqa: F401
    ATOM_SCORER_SIGNATURE,
    Decision,
    QSAIndexScorerUnsupported,
    decide,
    eligibility,
    launch,
    layout_eligibility,
    prewarm,
    prewarm_plan,
    prewarm_stats,
    compile_cache_dir,
    kernel_page_sizes,
    qsa_paged_mqa_logits,
    selected_profile,
    try_qsa_paged_mqa_logits,
)
