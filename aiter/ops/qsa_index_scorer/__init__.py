# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Private implementation of ``aiter.ops.qsa.qsa_paged_mqa_logits`` (not a public API).

The public op, its policy (``configure`` / ``prewarm`` / ``explain`` / ``status``) and
its contract live in :mod:`aiter.ops.qsa` (``docs/qsa_index_scorer.md``). This package
holds what serves it:

* :mod:`.dispatch` -- per-call-class decision cache and policy state of the neutral op;
* :mod:`.api` -- the fused FlyDSL gfx1250 path (profile match, router, launch plans,
  prewarm) for the call classes a profile lists;
* :mod:`.native` -- pure-torch implementation (valid calls no kernel takes, e.g. CPU);
* :mod:`.profiles` -- profile format / loader / matcher; :mod:`.gen_profile` -- the
  profile regeneration and GPU check tool (``python -m
  aiter.ops.qsa_index_scorer.gen_profile {emit,verify,check,coverage}``).

Nothing here imports FlyDSL until the fused path is enabled and a covered call (or a
prewarm) needs the kernel module. Names below are re-exported for aiter's own tests
and tools; they may change without notice.
"""

from . import api, native, profiles  # noqa: F401
from .native import NativeContractError  # noqa: F401
from .profiles import (  # noqa: F401
    API_VERSION,
    DEFAULT_PROFILE,
    PROFILE_ENV,
    Profile,
    ProfileError,
    Reason,
    load_profile,
)
