# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Universal discovery, directory layout, naming convention, and JSON syntax validation.

Validates that ALL configuration files across all 9 target architectures:
1. Conform to the canonical nested layout and naming rules (configs/CLAUDE.md).
2. Parse valid JSON and contain non-empty dictionaries.
3. Maintain exact file accounting across architectures, backends, and operator families.
4. Provide a DEFAULT.json file in every leaf directory (with explicit documented legacy exceptions).
5. Never mix LEQ and GEQ threshold conventions.
"""

from __future__ import annotations

import json
import os
import pytest

from aiter.ops.triton.utils import config_utils
from op_tests.triton_tests.config_tests.common import (
    check_no_mixed_geq_leq,
    discover_all_config_files,
    discover_architectures,
    load_json_dict,
    rel_path,
)

ALL_CONFIG_FILES = discover_all_config_files()
DISCOVERED_ARCHS = discover_architectures()


def test_config_files_accounting():
    """Verify repository-wide configuration accounting across all 9 target architectures.

    Ensures 100% of discovered files are partitioned without orphans, while allowing
    legitimate future tuning additions without brittle exact-count breakage.
    """
    assert len(ALL_CONFIG_FILES) >= 859, (
        f"Expected at least 859 configuration files, found {len(ALL_CONFIG_FILES)}"
    )
    assert len(DISCOVERED_ARCHS) == 9, (
        f"Expected 9 architectures, found {len(DISCOVERED_ARCHS)}: {DISCOVERED_ARCHS}"
    )

    defaults = [p for p in ALL_CONFIG_FILES if p.endswith("DEFAULT.json")]
    non_defaults = [p for p in ALL_CONFIG_FILES if not p.endswith("DEFAULT.json")]
    assert len(defaults) + len(non_defaults) == len(ALL_CONFIG_FILES)
    assert len(defaults) >= 296, f"Expected at least 296 DEFAULT.json files, found {len(defaults)}"

    gemm_spec = [p for p in non_defaults if f"{os.sep}gemm{os.sep}" in p]
    assert len(gemm_spec) >= 546, f"Expected at least 546 specialized GEMM files, found {len(gemm_spec)}"

    mhc_spec = [p for p in non_defaults if f"{os.sep}mhc{os.sep}" in p]
    assert len(mhc_spec) >= 11, f"Expected at least 11 specialized MHC files, found {len(mhc_spec)}"

    fusions_spec = [p for p in non_defaults if f"{os.sep}fusions{os.sep}" in p]
    assert len(fusions_spec) >= 5, f"Expected at least 5 specialized Fusions files, found {len(fusions_spec)}"

    legacy_files = [p for p in non_defaults if p.endswith("mha.json")]
    assert len(legacy_files) == 1, f"Expected exactly 1 legacy mha.json file, found {len(legacy_files)}"

    # 100% complete accounting: every non-default file must belong to a known family
    total_classified = len(gemm_spec) + len(mhc_spec) + len(fusions_spec) + len(legacy_files)
    assert total_classified == len(non_defaults), (
        f"Unaccounted non-default files found! Classified {total_classified} out of {len(non_defaults)}"
    )


def test_all_config_dirs_have_default():
    """Verify that every leaf directory contains DEFAULT.json (except legacy mha.json)."""
    leaf_dirs = {os.path.dirname(path) for path in ALL_CONFIG_FILES}
    missing_default = []
    for ldir in sorted(leaf_dirs):
        json_files = [f for f in os.listdir(ldir) if f.endswith(".json")]
        if "DEFAULT.json" not in json_files and "mha.json" not in json_files:
            missing_default.append(rel_path(ldir))

    assert not missing_default, f"Directories missing DEFAULT.json: {missing_default}"


@pytest.mark.parametrize("path", ALL_CONFIG_FILES, ids=rel_path)
def test_config_file_layout_and_naming(path: str):
    """Verify that every config file strictly follows configs/CLAUDE.md rules:
    - Path matches <configs>/<arch>/<backend>/<op>/<dtype>/<filename>
    - <arch> matches config_utils._ARCH_SAFE_RE
    - <backend> is in _VALID_BACKENDS ('triton', 'gluon')
    - <op> matches config_utils._OP_RE
    - <dtype> is lowercase with underscores
    - Filename has no architecture prefix
    - Filename is either DEFAULT.json or <CONFIG_NAME>-<suffix>.json (or legacy mha.json)
    """
    rel = rel_path(path)
    parts = rel.split(os.sep)
    assert len(parts) >= 5, f"Path too shallow: {rel}"

    arch, backend, op, dtype = parts[0], parts[1], parts[2], parts[3]
    fname = parts[-1]

    assert config_utils._ARCH_SAFE_RE.fullmatch(arch), f"{rel}: invalid arch identifier '{arch}'"
    assert backend in config_utils._VALID_BACKENDS, f"{rel}: invalid backend '{backend}'"
    assert config_utils._OP_RE.fullmatch(op), f"{rel}: invalid op identifier '{op}'"
    assert dtype == dtype.lower(), f"{rel}: dtype directory must be lowercase"
    assert "-" not in dtype, f"{rel}: dtype directory must not contain hyphens"

    # No architecture prefix in filenames
    for a in DISCOVERED_ARCHS:
        assert not fname.startswith(f"{a}_") and not fname.startswith(f"{a}-"), (
            f"{rel}: filename must not have arch prefix '{a}'"
        )

    # Valid filename formats
    if fname not in ("DEFAULT.json", "mha.json"):
        assert "-" in fname, (
            f"{rel}: specialized file should follow <CONFIG_NAME>-<suffix>.json format"
        )


@pytest.mark.parametrize("path", ALL_CONFIG_FILES, ids=rel_path)
def test_config_file_json_validity(path: str):
    """Verify that every config file parses valid JSON and is a non-empty dictionary."""
    load_json_dict(path)


@pytest.mark.parametrize("path", ALL_CONFIG_FILES, ids=rel_path)
def test_config_no_mixed_geq_leq_globally(path: str):
    """Verify that no configuration file mixes LEQ and GEQ threshold conventions."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    check_no_mixed_geq_leq(data, path)
