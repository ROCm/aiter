# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Backend-neutral option type for gated-delta-rule K5+K6 fusion.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["K5K6Fusion"]


class K5K6Fusion(Enum):
    """How to choose between the fused K5+K6 kernel and the separate pipeline.

    Members:
        AUTO: let the shape based heuristics to decide.
        ALWAYS: force the fused kernel regardless of shape, and raise error if the
            fused kernel cannot be used.
        NEVER: never fuse, always run the separate K5 + K6 kernels.
    """

    AUTO = "auto"
    ALWAYS = "always"
    NEVER = "never"

    @classmethod
    def coerce(cls, value) -> K5K6Fusion:
        """Accept a K5K6Fusion, or a case-insensitive string alias, or None.

        ``None`` maps to ``AUTO`` so callers can pass through an unset option.
        """
        if value is None:
            return cls.AUTO
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.lower())
            except ValueError:
                pass
        raise ValueError(
            f"invalid K5K6Fusion {value!r}; expected one of "
            f"{[m.value for m in cls]} (or a K5K6Fusion member / None)."
        )
