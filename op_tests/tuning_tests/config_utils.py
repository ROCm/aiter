# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shared tuned-config resolution for tuning tests (mirrors production AITER_CONFIGS)."""

from __future__ import annotations

import glob
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

AITER_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
CONFIGS_DIR = os.path.join(AITER_ROOT, "aiter", "configs")
MODEL_CONFIGS_DIR = os.path.join(CONFIGS_DIR, "model_configs")


def resolve_merged_tuned_path(config_property: str) -> str | None:
    """Resolve merged tuned CSV via ``AITER_CONFIGS`` (same path as production).

    ``config_property`` is the attribute on ``AITER_CONFIGS``, e.g.
    ``AITER_CONFIG_GDN_K5_MFMA16_HIP_FILE``.
    """
    try:
        from aiter.jit.core import AITER_CONFIGS

        config_file = getattr(AITER_CONFIGS, config_property, None)
        if config_file and os.path.exists(config_file):
            return config_file
    except Exception:  # noqa: BLE001,S110
        pass
    return None


def _glob_merge_tuned_csvs(base_name: str, model_glob: str) -> list[str]:
    paths = [os.path.join(CONFIGS_DIR, base_name)]
    paths.extend(sorted(glob.glob(os.path.join(MODEL_CONFIGS_DIR, model_glob))))
    return [p for p in paths if os.path.exists(p)]


def load_merged_tuned_dataframe(
    config_property: str,
    *,
    comment: str = "#",
    fallback_base_name: str | None = None,
    fallback_model_glob: str | None = None,
) -> pd.DataFrame:
    """Load merged tuned CSV as a DataFrame.

    Uses ``resolve_merged_tuned_path`` when ``aiter`` is available; otherwise
    falls back to globbing ``configs/`` + ``model_configs/`` when both
    ``fallback_base_name`` and ``fallback_model_glob`` are set.
    """
    import pandas as pd

    from aiter.utility.base_tuner import _read_csv

    resolved = resolve_merged_tuned_path(config_property)
    if resolved:
        return _read_csv(resolved, comment=comment)

    if fallback_base_name is None or fallback_model_glob is None:
        raise FileNotFoundError(
            f"Could not resolve {config_property} via AITER_CONFIGS "
            "and no filesystem fallback was configured"
        )

    paths = _glob_merge_tuned_csvs(fallback_base_name, fallback_model_glob)
    if not paths:
        raise FileNotFoundError(
            f"No tuned CSVs found for {config_property} "
            f"(base={fallback_base_name}, glob={fallback_model_glob})"
        )

    frames = [_read_csv(path, comment=comment) for path in paths]
    return pd.concat(frames, ignore_index=True)
