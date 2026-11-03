# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Deprecation aliases for backend-suffixed public names (#7150).

A backend-suffixed public name keeps working during the removal window but
warns on each distinct call site once, then delegates unchanged. The removal
window is tracked in #7150; the warning deliberately does not promise a
release number.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any


def deprecated_alias(obj: Callable, old_name: str, use_instead: str) -> Callable:
    """Wrap ``obj`` so the deprecated public name ``old_name`` warns once per
    call site before delegating. ``use_instead`` names the neutral replacement
    in the warning text.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            f"aiter.{old_name} is deprecated: it is a backend-specific name. "
            f"Use {use_instead} instead. It will be removed after the "
            "deprecation window tracked in #7150 "
            "(https://github.com/ROCm/aiter/issues/7150).",
            DeprecationWarning,
            stacklevel=2,
        )
        return obj(*args, **kwargs)

    wrapper.__name__ = old_name
    wrapper.__qualname__ = f"aiter.{old_name}"
    wrapper.__doc__ = obj.__doc__
    wrapper.__wrapped__ = obj
    return wrapper
