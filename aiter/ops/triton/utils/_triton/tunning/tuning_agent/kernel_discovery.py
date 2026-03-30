"""KernelDiscovery — scan all GEMM categories and match kernels to configs/scripts.

This module provides a pure-Python, filesystem-based approach to discovering
all Triton GEMM kernels across the four supported categories (basic, batched,
feed_forward, fused), locating their associated config JSON files, and finding
the unit-test, benchmark, and integration-test scripts that exercise them.

It does **not** import any Triton or GPU-specific code, so it can be run on a
developer machine without a GPU present.

Usage
-----
::

    from aiter.ops.triton.utils._triton.tunning.tuning_agent.kernel_discovery import (
        KernelDiscovery,
        DiscoveredKernel,
        GemmCategory,
    )

    kd = KernelDiscovery(repo_root="/path/to/aiter", gfx_arch="gfx950")
    kernels = kd.discover_all()
    for k in kernels:
        print(k.name, k.category, k.config_variant, k.has_all_scripts)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class GemmCategory(str, Enum):
    """Supported GEMM kernel categories."""

    BASIC = "basic"
    BATCHED = "batched"
    FEED_FORWARD = "feed_forward"
    FUSED = "fused"


# ---------------------------------------------------------------------------
# DiscoveredKernel dataclass
# ---------------------------------------------------------------------------


@dataclass
class DiscoveredKernel:
    """Represents a single discovered GEMM kernel and its associated artefacts.

    Attributes
    ----------
    name:
        Short kernel identifier derived from the source file name.
        E.g. ``"a8w8"``, ``"a16w16_atomic"``.
    category:
        The :class:`GemmCategory` bucket this kernel belongs to.
    source_path:
        Absolute path to the kernel Python source file.
    config_variant:
        The naming convention used in config JSON filenames.
        E.g. ``"A8W8"``, ``"BATCHED_GEMM-A8W8"``, ``"FF-A8W8"``,
        ``"FUSED-GEMM-A8W8"``.
    config_files:
        List of absolute paths to existing config JSON files that match
        this kernel's variant.
    ut_script:
        Absolute path to the unit-test / tuning helper script, or ``None``
        if not found.
    bench_script:
        Absolute path to the benchmark script, or ``None`` if not found.
    test_script:
        Absolute path to the integration / functional test script, or
        ``None`` if not found.
    """

    name: str
    category: GemmCategory
    source_path: str
    config_variant: str
    config_files: List[str] = field(default_factory=list)
    ut_script: Optional[str] = None
    bench_script: Optional[str] = None
    test_script: Optional[str] = None

    # ------------------------------------------------------------------
    # Derived property
    # ------------------------------------------------------------------

    @property
    def has_all_scripts(self) -> bool:
        """``True`` when all three script types have been located."""
        return (
            self.ut_script is not None
            and self.bench_script is not None
            and self.test_script is not None
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Mapping of (category, raw_stem) → config_variant override.
# Keys are (GemmCategory, stem_after_prefix_removal).
_VARIANT_OVERRIDES: Dict[Tuple[GemmCategory, str], str] = {
    # "a16w16_atomic" → "A16W16-ATOMIC" (underscore before "atomic" → hyphen)
    (GemmCategory.BASIC, "a16w16_atomic"): "A16W16-ATOMIC",
    # "afp4wfp4_preshuffled" → "AFP4WFP4_PRESHUFFLED" (keep underscore)
    (GemmCategory.BASIC, "afp4wfp4_preshuffled"): "AFP4WFP4_PRESHUFFLED",
}

# Category-level prefix applied to the uppercased stem when building the
# config_variant string.  Basic kernels have no prefix — their stem is used
# directly after uppercasing.
_CATEGORY_PREFIX: Dict[GemmCategory, str] = {
    GemmCategory.BASIC: "",
    GemmCategory.BATCHED: "BATCHED_GEMM-",
    GemmCategory.FEED_FORWARD: "FF-",
    GemmCategory.FUSED: "FUSED-GEMM-",
}

# Each category scans a specific sub-directory of ``aiter/ops/triton/gemm/``
# for files matching a given glob pattern.
_CATEGORY_SCAN: Dict[GemmCategory, Tuple[str, str, str]] = {
    # (sub_dir, filename_prefix, strip_prefix_to_get_name)
    GemmCategory.BASIC: ("basic", "gemm_", "gemm_"),
    GemmCategory.BATCHED: ("batched", "batched_gemm_", "batched_gemm_"),
    GemmCategory.FEED_FORWARD: ("feed_forward", "ff_", "ff_"),
    GemmCategory.FUSED: ("fused", "fused_gemm_", "fused_gemm_"),
}


def _derive_config_variant(category: GemmCategory, name: str) -> str:
    """Return the config-file naming variant for *name* in *category*.

    Parameters
    ----------
    category:
        The :class:`GemmCategory` the kernel belongs to.
    name:
        The short kernel name (stem after prefix removal, e.g. ``"a8w8"``).

    Returns
    -------
    str
        The variant string as it appears in config JSON filenames.
    """
    override_key = (category, name)
    if override_key in _VARIANT_OVERRIDES:
        base = _VARIANT_OVERRIDES[override_key]
    else:
        base = name.upper()

    prefix = _CATEGORY_PREFIX[category]
    return f"{prefix}{base}"


# ---------------------------------------------------------------------------
# KernelDiscovery
# ---------------------------------------------------------------------------


class KernelDiscovery:
    """Discover all GEMM kernels in a local aiter repository.

    The discovery process:

    1. Scans four kernel source directories (one per :class:`GemmCategory`).
    2. Derives the config variant name for each kernel.
    3. Locates matching config JSON files.
    4. Locates ut / bench / test scripts.
    5. Applies optional include / exclude filters.

    Parameters
    ----------
    repo_root:
        Absolute path to the root of the aiter repository (the directory
        that contains the top-level ``aiter/`` package).
    gfx_arch:
        GPU architecture string used as the prefix in config JSON filenames,
        e.g. ``"gfx950"``.
    """

    def __init__(self, repo_root: str, gfx_arch: str = "gfx950") -> None:
        self.repo_root = repo_root
        self.gfx_arch = gfx_arch

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover_all(
        self,
        exclude: Optional[List[str]] = None,
        include: Optional[List[str]] = None,
    ) -> List[DiscoveredKernel]:
        """Discover all GEMM kernels in the repository.

        Parameters
        ----------
        exclude:
            Optional list of kernel *names* to exclude from the result.
            Names are matched against :attr:`DiscoveredKernel.name`.
        include:
            Optional allowlist of kernel *names*.  When provided, only
            kernels whose name appears in this list are returned.  Takes
            precedence over *exclude* for determining membership.

        Returns
        -------
        List[DiscoveredKernel]
            Discovered kernels, sorted by ``(category.value, name)``.
        """
        results: List[DiscoveredKernel] = []

        for category in GemmCategory:
            kernels = self._scan_category(category)
            results.extend(kernels)

        # Apply filters
        filtered: List[DiscoveredKernel] = []
        for kernel in results:
            if include is not None and kernel.name not in include:
                continue
            if exclude is not None and kernel.name in exclude:
                continue
            filtered.append(kernel)

        filtered.sort(key=lambda k: (k.category.value, k.name))
        return filtered

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scan_category(self, category: GemmCategory) -> List[DiscoveredKernel]:
        """Scan one category's source directory and return discovered kernels."""
        sub_dir, file_prefix, strip_prefix = _CATEGORY_SCAN[category]
        source_dir = os.path.join(
            self.repo_root, "aiter", "ops", "triton", "gemm", sub_dir
        )

        if not os.path.isdir(source_dir):
            return []

        discovered: List[DiscoveredKernel] = []
        try:
            entries = os.listdir(source_dir)
        except OSError:
            return []

        for entry in sorted(entries):
            if not entry.startswith(file_prefix):
                continue
            if not entry.endswith(".py"):
                continue
            if entry == "__init__.py":
                continue

            stem = entry[: -len(".py")]  # strip .py
            name = stem[len(strip_prefix):]  # strip category prefix

            if not name:
                continue

            source_path = os.path.join(source_dir, entry)
            config_variant = _derive_config_variant(category, name)
            config_files = self._find_config_files(config_variant)
            ut_script = self._find_ut_script(name)
            bench_script = self._find_bench_script(name, category)
            test_script = self._find_test_script(name)

            discovered.append(
                DiscoveredKernel(
                    name=name,
                    category=category,
                    source_path=source_path,
                    config_variant=config_variant,
                    config_files=config_files,
                    ut_script=ut_script,
                    bench_script=bench_script,
                    test_script=test_script,
                )
            )

        return discovered

    def _find_config_files(self, variant: str) -> List[str]:
        """Return absolute paths of config JSON files that match *variant*.

        Searches ``aiter/ops/triton/configs/gemm/`` for files whose names
        match either of two patterns:

        * ``{gfx_arch}-GEMM-{variant}*.json``
        * ``{gfx_arch}-{variant}*.json``

        Parameters
        ----------
        variant:
            Config variant string, e.g. ``"A8W8"`` or ``"BATCHED_GEMM-A8W8"``.

        Returns
        -------
        List[str]
            Sorted list of absolute paths for existing config files.
        """
        config_dir = os.path.join(
            self.repo_root, "aiter", "ops", "triton", "configs", "gemm"
        )
        if not os.path.isdir(config_dir):
            return []

        prefix1 = f"{self.gfx_arch}-GEMM-{variant}"
        prefix2 = f"{self.gfx_arch}-{variant}"

        matched: List[str] = []
        try:
            entries = os.listdir(config_dir)
        except OSError:
            return []

        for entry in entries:
            if not entry.endswith(".json"):
                continue
            if entry.startswith(prefix1) or entry.startswith(prefix2):
                matched.append(os.path.join(config_dir, entry))

        return sorted(matched)

    def _find_ut_script(self, name: str) -> Optional[str]:
        """Locate the unit-test / tuning helper script for *name*.

        Searches ``aiter/ops/triton/utils/_triton/tunning/`` for a file
        matching ``ut_{name}_gemm*.py``.

        Parameters
        ----------
        name:
            Short kernel name, e.g. ``"a8w8"``.

        Returns
        -------
        Optional[str]
            Absolute path if found, ``None`` otherwise.
        """
        search_dir = os.path.join(
            self.repo_root,
            "aiter",
            "ops",
            "triton",
            "utils",
            "_triton",
            "tunning",
        )
        return self._find_first(search_dir, f"ut_{name}_gemm")

    def _find_bench_script(
        self, name: str, category: GemmCategory
    ) -> Optional[str]:
        """Locate the benchmark script for *name* in *category*.

        Searches ``op_tests/op_benchmarks/triton/`` for files matching:

        * ``bench_gemm_{name}*.py``
        * ``bench_{category_value}_{name}*.py``

        Parameters
        ----------
        name:
            Short kernel name, e.g. ``"a8w8"``.
        category:
            The kernel's :class:`GemmCategory`.

        Returns
        -------
        Optional[str]
            Absolute path if found, ``None`` otherwise.
        """
        search_dir = os.path.join(
            self.repo_root, "op_tests", "op_benchmarks", "triton"
        )
        # Try primary pattern first
        result = self._find_first(search_dir, f"bench_gemm_{name}")
        if result is not None:
            return result
        # Fallback: bench_{category}_{name}
        return self._find_first(search_dir, f"bench_{category.value}_{name}")

    def _find_test_script(self, name: str) -> Optional[str]:
        """Locate the integration / functional test script for *name*.

        Searches ``op_tests/triton_tests/gemm/`` (recursively through one
        level of sub-directories) for a file matching
        ``test_gemm_{name}*.py``.

        Parameters
        ----------
        name:
            Short kernel name, e.g. ``"a8w8"``.

        Returns
        -------
        Optional[str]
            Absolute path if found, ``None`` otherwise.
        """
        base_dir = os.path.join(self.repo_root, "op_tests", "triton_tests", "gemm")
        return self._find_first_recursive(base_dir, f"test_gemm_{name}")

    # ------------------------------------------------------------------
    # Low-level filesystem helpers
    # ------------------------------------------------------------------

    def _find_first(self, directory: str, prefix: str) -> Optional[str]:
        """Return the first ``.py`` file in *directory* whose name starts with *prefix*.

        Parameters
        ----------
        directory:
            Directory to search (non-recursive).
        prefix:
            Required filename prefix (without ``.py`` extension).

        Returns
        -------
        Optional[str]
            Absolute path if found, ``None`` if the directory is missing or
            no matching file exists.
        """
        if not os.path.isdir(directory):
            return None
        try:
            entries = os.listdir(directory)
        except OSError:
            return None

        for entry in sorted(entries):
            if entry.startswith(prefix) and entry.endswith(".py"):
                return os.path.join(directory, entry)

        return None

    def _find_first_recursive(self, base_dir: str, prefix: str) -> Optional[str]:
        """Search *base_dir* and its immediate sub-directories for a file.

        The search is intentionally limited to one level of recursion (i.e.
        ``base_dir/*.py`` and ``base_dir/*/*.py``) to avoid accidentally
        traversing deep directory trees.

        Parameters
        ----------
        base_dir:
            Top-level directory to search.
        prefix:
            Required filename prefix (without ``.py`` extension).

        Returns
        -------
        Optional[str]
            Absolute path if found, ``None`` otherwise.
        """
        if not os.path.isdir(base_dir):
            return None

        # Search base directory first
        result = self._find_first(base_dir, prefix)
        if result is not None:
            return result

        # Search one level of sub-directories
        try:
            entries = os.listdir(base_dir)
        except OSError:
            return None

        for entry in sorted(entries):
            sub_dir = os.path.join(base_dir, entry)
            if os.path.isdir(sub_dir):
                result = self._find_first(sub_dir, prefix)
                if result is not None:
                    return result

        return None
