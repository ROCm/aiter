# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Stable short hashes of JSON payloads (config keys, fingerprints)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

__all__ = ["stable_json_hash"]


def stable_json_hash(payload: Mapping[str, Any], *, n: int = 16) -> str:
    """Stable short SHA256 over a JSON-serializable payload."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:n]
