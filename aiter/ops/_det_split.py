# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Deterministic-serving split-cap resolution for MLA decode (issue #4364).

Kept dependency-free (only ``os`` + ``warnings``) so the env-parsing logic can be
unit-tested on CPU without importing the full ``aiter`` package / JIT build.
"""

import os
import warnings


def resolve_det_split_cap(max_split_per_batch: int) -> int:
    """Apply the deterministic-serving split cap from the environment.

    The MLA decode split-KV reduce is fp-non-associative, so a request's output
    depends on how many KV splits it is given, and that count is drawn from the
    *global* budget ``min(cu_num, max_split_per_batch * batch_size)`` which varies
    with batch composition. Forcing a single split (n=1) makes the reduce a no-op
    and the result reproducible run-to-run; this is opt-in (it costs decode
    split-K parallelism at long context).

    Env knobs (``AITER_MLA_DECODE_MAX_SPLIT_PER_BATCH`` takes precedence over
    ``AITER_MLA_DECODE_DETERMINISTIC``):

    - ``AITER_MLA_DECODE_MAX_SPLIT_PER_BATCH=<n>`` caps the per-batch split at n
      (n>=1). **Only n=1 guarantees reproducibility**; n>1 merely lowers the
      ceiling, so the split count can still vary with batch composition.
    - ``AITER_MLA_DECODE_DETERMINISTIC=1`` is shorthand for n=1, applied only when
      the explicit cap is unset.

    Invalid (non-integer) or out-of-range (<1) values are ignored with a warning
    rather than silently changing behaviour. If both knobs are set and disagree
    (DETERMINISTIC truthy but the explicit cap != 1) we warn, since the explicit
    cap wins. Clamping down is always buffer-safe (it can only reduce the number
    of partial-reduce entries relative to what the caller already allocated).

    Returns the (possibly clamped) ``max_split_per_batch`` to use.
    """
    cap = os.getenv("AITER_MLA_DECODE_MAX_SPLIT_PER_BATCH")
    det_flag = os.getenv("AITER_MLA_DECODE_DETERMINISTIC", "0") not in ("0", "")

    if cap is not None and det_flag and cap.strip() != "1":
        warnings.warn(
            f"Both AITER_MLA_DECODE_MAX_SPLIT_PER_BATCH={cap!r} and "
            "AITER_MLA_DECODE_DETERMINISTIC=1 are set; the explicit cap takes "
            "precedence (DETERMINISTIC ignored)."
        )
    if cap is None and det_flag:
        cap = "1"

    if cap is None:
        return max_split_per_batch

    try:
        cap_i = int(cap)
    except ValueError:
        warnings.warn(
            "Ignoring invalid AITER_MLA_DECODE_MAX_SPLIT_PER_BATCH="
            f"{cap!r}; expected an integer >= 1."
        )
        return max_split_per_batch

    if cap_i < 1:
        warnings.warn(
            f"Ignoring AITER_MLA_DECODE_MAX_SPLIT_PER_BATCH={cap_i} "
            "(must be >= 1); using the default split budget."
        )
        return max_split_per_batch

    return cap_i if max_split_per_batch <= 0 else min(max_split_per_batch, cap_i)
