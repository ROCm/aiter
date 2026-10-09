# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MXFP8 ASM configuration, using the shared tuned/untuned CSV convention."""

import functools
import os
from pathlib import Path

import pandas as pd

from .. import logger
from ..jit.core import AITER_ASM_DIR, AITER_CONFIG, AITER_ROOT_DIR
from ..jit.utils.torch_guard import torch_compile_guard

_CONFIG_NAME = "mxfp8fp4_asm_tuned_gemm"
_MXFP8_GEMM_CONFIG_KEYS = ("gfx", "M", "N", "K", "b_intype", "a_preshuffle", "outdtype")


class _Mxfp8Config(AITER_CONFIG):
    def update_config_files(self, file_path, merge_name):
        # Validate before the shared merger fills missing columns or writes back
        # duplicate winners. A malformed source must not displace a valid one.
        paths = file_path.split(os.pathsep) if file_path else []
        if len(paths) <= 1:
            return file_path  # The op's loader validates individual rows.
        valid_paths, tables = [], []
        for path in paths:
            if not os.path.isfile(path):
                continue
            try:
                table = pd.read_csv(path)
                missing = {*_MXFP8_GEMM_CONFIG_KEYS, "kernelName", "splitK"} - set(
                    table.columns
                )
                if missing:
                    raise ValueError(f"missing columns {sorted(missing)}")
                for column in ("gfx", "b_intype", "outdtype", "kernelName"):
                    valid = table[column].map(
                        lambda value: isinstance(value, str)
                        and value.strip() not in ("", "0")
                    )
                    if not valid.all():
                        raise ValueError(
                            f"invalid {column} at CSV lines {(table.index[~valid] + 2).tolist()}"
                        )
            except (OSError, UnicodeError, ValueError) as exc:
                logger.warning("Skipping invalid MXFP8 GEMM config %r: %s", path, exc)
                continue
            valid_paths.append(path)
            tables.append(table)
        if not tables:
            raise FileNotFoundError("No usable MXFP8 GEMM config files")

        # The shared merger includes optional cu_num/_tag selectors. This op
        # does not query them: diagnose ambiguous variants before any writeback.
        merged = pd.concat(tables, ignore_index=True)
        selectors = [c for c in ("cu_num", "_tag") if c in merged.columns]
        if selectors:
            keys = list(_MXFP8_GEMM_CONFIG_KEYS)
            variants = merged[keys + selectors].fillna("").drop_duplicates()
            if variants.duplicated(keys).any():
                raise RuntimeError(
                    "Conflicting MXFP8 GEMM configs differ only in cu_num/_tag; "
                    "these are not runtime lookup keys. Select one variant per shape."
                )
        return super().update_config_files(os.pathsep.join(valid_paths), merge_name)


# A separate instance keeps MXFP8 validation out of all other config families.
_CONFIG = _Mxfp8Config()


def get_mxfp8_config_file():
    return _CONFIG.get_config_file(
        "AITER_CONFIG_GEMM_MXFP8FP4",
        str(Path(AITER_ROOT_DIR) / "aiter" / "configs" / f"{_CONFIG_NAME}.csv"),
        _CONFIG_NAME,
    )


def get_mxfp8_asm_dir(gfx):
    """Use the same runtime architecture as the tuning lookup."""
    return str(Path(AITER_ASM_DIR) / gfx)


def mxfp8_compile_guard(**kwargs):
    """Preserve MXFP8 API metadata without changing the shared decorator."""

    def decorate(func):
        return functools.wraps(func)(torch_compile_guard(**kwargs)(func))

    return decorate
