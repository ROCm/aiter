"""Tests for KernelDiscovery — TDD suite.

Uses ``tmp_path`` (pytest fixture) to build a minimal mock repository structure
that mirrors the real aiter layout so all tests run without GPU hardware or
network access.
"""

from __future__ import annotations

import os
import pathlib
from typing import List

import pytest

from .kernel_discovery import (
    DiscoveredKernel,
    GemmCategory,
    KernelDiscovery,
    _derive_config_variant,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _touch(path: pathlib.Path) -> pathlib.Path:
    """Create *path* (and any missing parents) as an empty file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    return path


def _make_mock_repo(tmp_path: pathlib.Path) -> pathlib.Path:
    """Return the path to a minimal mock aiter repository under *tmp_path*.

    Creates:
    - ``aiter/ops/triton/gemm/{basic,batched,feed_forward,fused}/``
    - ``aiter/ops/triton/configs/gemm/``
    - ``aiter/ops/triton/utils/_triton/tunning/``
    - ``op_tests/op_benchmarks/triton/``
    - ``op_tests/triton_tests/gemm/basic/``
    """
    repo = tmp_path / "repo"

    # Gemm source dirs
    for category_dir in ["basic", "batched", "feed_forward", "fused"]:
        (repo / "aiter" / "ops" / "triton" / "gemm" / category_dir).mkdir(
            parents=True, exist_ok=True
        )

    # Config dir
    (repo / "aiter" / "ops" / "triton" / "configs" / "gemm").mkdir(
        parents=True, exist_ok=True
    )

    # UT script dir
    (repo / "aiter" / "ops" / "triton" / "utils" / "_triton" / "tunning").mkdir(
        parents=True, exist_ok=True
    )

    # Benchmark dir
    (repo / "op_tests" / "op_benchmarks" / "triton").mkdir(
        parents=True, exist_ok=True
    )

    # Test dirs
    (repo / "op_tests" / "triton_tests" / "gemm" / "basic").mkdir(
        parents=True, exist_ok=True
    )
    (repo / "op_tests" / "triton_tests" / "gemm" / "batched").mkdir(
        parents=True, exist_ok=True
    )

    return repo


# ---------------------------------------------------------------------------
# Unit tests: _derive_config_variant
# ---------------------------------------------------------------------------


class TestDeriveConfigVariant:
    """Validate the static config-variant derivation logic."""

    def test_basic_a8w8(self) -> None:
        assert _derive_config_variant(GemmCategory.BASIC, "a8w8") == "A8W8"

    def test_basic_a16w16(self) -> None:
        assert _derive_config_variant(GemmCategory.BASIC, "a16w16") == "A16W16"

    def test_basic_a16w16_atomic_uses_hyphen(self) -> None:
        """Special case: 'atomic' suffix should become '-ATOMIC' (hyphen)."""
        result = _derive_config_variant(GemmCategory.BASIC, "a16w16_atomic")
        assert result == "A16W16-ATOMIC", f"Got: {result!r}"

    def test_basic_afp4wfp4_preshuffled(self) -> None:
        """Special case: preshuffle variant keeps underscore in its suffix."""
        result = _derive_config_variant(GemmCategory.BASIC, "afp4wfp4_preshuffled")
        assert result == "AFP4WFP4_PRESHUFFLED", f"Got: {result!r}"

    def test_batched_prefix(self) -> None:
        assert _derive_config_variant(GemmCategory.BATCHED, "a8w8") == "BATCHED_GEMM-A8W8"

    def test_feed_forward_prefix(self) -> None:
        assert _derive_config_variant(GemmCategory.FEED_FORWARD, "a8w8") == "FF-A8W8"

    def test_fused_prefix(self) -> None:
        assert _derive_config_variant(GemmCategory.FUSED, "a8w8") == "FUSED-GEMM-A8W8"


# ---------------------------------------------------------------------------
# Tests: discover_all — basic category
# ---------------------------------------------------------------------------


class TestDiscoverAllBasic:
    """Tests focused on scanning the 'basic' GEMM category."""

    def test_discovers_basic_kernel(self, tmp_path: pathlib.Path) -> None:
        repo = _make_mock_repo(tmp_path)
        _touch(repo / "aiter" / "ops" / "triton" / "gemm" / "basic" / "gemm_a8w8.py")

        kd = KernelDiscovery(str(repo))
        kernels = kd.discover_all()

        names = [k.name for k in kernels]
        assert "a8w8" in names, f"Expected 'a8w8' in {names}"

    def test_basic_kernel_has_correct_category(self, tmp_path: pathlib.Path) -> None:
        repo = _make_mock_repo(tmp_path)
        _touch(repo / "aiter" / "ops" / "triton" / "gemm" / "basic" / "gemm_a8w8.py")

        kd = KernelDiscovery(str(repo))
        kernels = kd.discover_all()

        a8w8 = next(k for k in kernels if k.name == "a8w8")
        assert a8w8.category == GemmCategory.BASIC

    def test_basic_kernel_config_variant(self, tmp_path: pathlib.Path) -> None:
        repo = _make_mock_repo(tmp_path)
        _touch(repo / "aiter" / "ops" / "triton" / "gemm" / "basic" / "gemm_a8w8.py")

        kd = KernelDiscovery(str(repo))
        kernels = kd.discover_all()

        a8w8 = next(k for k in kernels if k.name == "a8w8")
        assert a8w8.config_variant == "A8W8"

    def test_basic_source_path_is_absolute(self, tmp_path: pathlib.Path) -> None:
        repo = _make_mock_repo(tmp_path)
        expected = _touch(
            repo / "aiter" / "ops" / "triton" / "gemm" / "basic" / "gemm_a8w8.py"
        )

        kd = KernelDiscovery(str(repo))
        kernels = kd.discover_all()

        a8w8 = next(k for k in kernels if k.name == "a8w8")
        assert os.path.isabs(a8w8.source_path)
        assert a8w8.source_path == str(expected)

    def test_discovers_multiple_basic_kernels(self, tmp_path: pathlib.Path) -> None:
        repo = _make_mock_repo(tmp_path)
        basic_dir = repo / "aiter" / "ops" / "triton" / "gemm" / "basic"
        _touch(basic_dir / "gemm_a8w8.py")
        _touch(basic_dir / "gemm_a16w16.py")
        _touch(basic_dir / "gemm_a16w16_atomic.py")

        kd = KernelDiscovery(str(repo))
        kernels = kd.discover_all()
        names = {k.name for k in kernels}

        assert {"a8w8", "a16w16", "a16w16_atomic"}.issubset(names)

    def test_ignores_init_py(self, tmp_path: pathlib.Path) -> None:
        repo = _make_mock_repo(tmp_path)
        _touch(
            repo / "aiter" / "ops" / "triton" / "gemm" / "basic" / "__init__.py"
        )

        kd = KernelDiscovery(str(repo))
        kernels = kd.discover_all()

        names = [k.name for k in kernels]
        assert "" not in names  # stem after prefix removal would be empty
        assert not any(n == "__init__" for n in names)


# ---------------------------------------------------------------------------
# Tests: discover_all — batched category
# ---------------------------------------------------------------------------


class TestDiscoverAllBatched:
    """Tests focused on scanning the 'batched' GEMM category."""

    def test_discovers_batched_kernel(self, tmp_path: pathlib.Path) -> None:
        repo = _make_mock_repo(tmp_path)
        _touch(
            repo
            / "aiter"
            / "ops"
            / "triton"
            / "gemm"
            / "batched"
            / "batched_gemm_a8w8.py"
        )

        kd = KernelDiscovery(str(repo))
        kernels = kd.discover_all()

        names = [k.name for k in kernels]
        assert "a8w8" in names

    def test_batched_category_assigned(self, tmp_path: pathlib.Path) -> None:
        repo = _make_mock_repo(tmp_path)
        _touch(
            repo
            / "aiter"
            / "ops"
            / "triton"
            / "gemm"
            / "batched"
            / "batched_gemm_a8w8.py"
        )

        kd = KernelDiscovery(str(repo))
        kernels = kd.discover_all()

        batched_kernels = [k for k in kernels if k.category == GemmCategory.BATCHED]
        assert len(batched_kernels) == 1
        assert batched_kernels[0].name == "a8w8"

    def test_batched_config_variant(self, tmp_path: pathlib.Path) -> None:
        repo = _make_mock_repo(tmp_path)
        _touch(
            repo
            / "aiter"
            / "ops"
            / "triton"
            / "gemm"
            / "batched"
            / "batched_gemm_a8w8.py"
        )

        kd = KernelDiscovery(str(repo))
        kernels = kd.discover_all()

        a8w8_batched = next(
            k for k in kernels if k.category == GemmCategory.BATCHED and k.name == "a8w8"
        )
        assert a8w8_batched.config_variant == "BATCHED_GEMM-A8W8"


# ---------------------------------------------------------------------------
# Tests: discover_all — feed_forward category
# ---------------------------------------------------------------------------


class TestDiscoverAllFeedForward:
    """Tests focused on scanning the 'feed_forward' GEMM category."""

    def test_discovers_ff_kernel(self, tmp_path: pathlib.Path) -> None:
        repo = _make_mock_repo(tmp_path)
        _touch(
            repo / "aiter" / "ops" / "triton" / "gemm" / "feed_forward" / "ff_a8w8.py"
        )

        kd = KernelDiscovery(str(repo))
        kernels = kd.discover_all()

        ff_kernels = [k for k in kernels if k.category == GemmCategory.FEED_FORWARD]
        assert len(ff_kernels) == 1
        assert ff_kernels[0].name == "a8w8"
        assert ff_kernels[0].config_variant == "FF-A8W8"


# ---------------------------------------------------------------------------
# Tests: discover_all — fused category
# ---------------------------------------------------------------------------


class TestDiscoverAllFused:
    """Tests focused on scanning the 'fused' GEMM category."""

    def test_discovers_fused_kernel(self, tmp_path: pathlib.Path) -> None:
        repo = _make_mock_repo(tmp_path)
        _touch(
            repo
            / "aiter"
            / "ops"
            / "triton"
            / "gemm"
            / "fused"
            / "fused_gemm_a8w8.py"
        )

        kd = KernelDiscovery(str(repo))
        kernels = kd.discover_all()

        fused_kernels = [k for k in kernels if k.category == GemmCategory.FUSED]
        assert len(fused_kernels) == 1
        assert fused_kernels[0].name == "a8w8"
        assert fused_kernels[0].config_variant == "FUSED-GEMM-A8W8"


# ---------------------------------------------------------------------------
# Tests: config file matching
# ---------------------------------------------------------------------------


class TestConfigFileDiscovery:
    """Validate that config JSON files are located and matched correctly."""

    def test_finds_gemm_prefixed_config(self, tmp_path: pathlib.Path) -> None:
        repo = _make_mock_repo(tmp_path)
        _touch(repo / "aiter" / "ops" / "triton" / "gemm" / "basic" / "gemm_a8w8.py")
        config_path = _touch(
            repo
            / "aiter"
            / "ops"
            / "triton"
            / "configs"
            / "gemm"
            / "gfx950-GEMM-A8W8-M_LEQ_16.json"
        )

        kd = KernelDiscovery(str(repo), gfx_arch="gfx950")
        kernels = kd.discover_all()

        a8w8 = next(k for k in kernels if k.name == "a8w8")
        assert str(config_path) in a8w8.config_files

    def test_finds_non_gemm_prefixed_config(self, tmp_path: pathlib.Path) -> None:
        """The secondary pattern ``{gfx_arch}-{variant}*.json`` should also match."""
        repo = _make_mock_repo(tmp_path)
        _touch(
            repo / "aiter" / "ops" / "triton" / "gemm" / "batched" / "batched_gemm_a8w8.py"
        )
        config_path = _touch(
            repo
            / "aiter"
            / "ops"
            / "triton"
            / "configs"
            / "gemm"
            / "gfx950-BATCHED_GEMM-A8W8-M_LEQ_16.json"
        )

        kd = KernelDiscovery(str(repo), gfx_arch="gfx950")
        kernels = kd.discover_all()

        batched = next(
            k for k in kernels if k.category == GemmCategory.BATCHED and k.name == "a8w8"
        )
        assert str(config_path) in batched.config_files

    def test_multiple_config_files_found(self, tmp_path: pathlib.Path) -> None:
        repo = _make_mock_repo(tmp_path)
        _touch(repo / "aiter" / "ops" / "triton" / "gemm" / "basic" / "gemm_a8w8.py")
        cfg_dir = repo / "aiter" / "ops" / "triton" / "configs" / "gemm"
        _touch(cfg_dir / "gfx950-GEMM-A8W8-M_LEQ_16.json")
        _touch(cfg_dir / "gfx950-GEMM-A8W8-M_LEQ_256.json")
        _touch(cfg_dir / "gfx950-GEMM-A8W8-M_GT_256.json")

        kd = KernelDiscovery(str(repo), gfx_arch="gfx950")
        kernels = kd.discover_all()

        a8w8 = next(k for k in kernels if k.name == "a8w8")
        assert len(a8w8.config_files) == 3

    def test_no_config_files_returns_empty_list(self, tmp_path: pathlib.Path) -> None:
        repo = _make_mock_repo(tmp_path)
        _touch(repo / "aiter" / "ops" / "triton" / "gemm" / "basic" / "gemm_a8w8.py")

        kd = KernelDiscovery(str(repo), gfx_arch="gfx950")
        kernels = kd.discover_all()

        a8w8 = next(k for k in kernels if k.name == "a8w8")
        assert a8w8.config_files == []

    def test_gfx_arch_filter(self, tmp_path: pathlib.Path) -> None:
        """Config files for a different arch should not appear."""
        repo = _make_mock_repo(tmp_path)
        _touch(repo / "aiter" / "ops" / "triton" / "gemm" / "basic" / "gemm_a8w8.py")
        cfg_dir = repo / "aiter" / "ops" / "triton" / "configs" / "gemm"
        _touch(cfg_dir / "gfx942-GEMM-A8W8-M_LEQ_16.json")  # wrong arch
        expected = _touch(cfg_dir / "gfx950-GEMM-A8W8-M_LEQ_16.json")  # correct arch

        kd = KernelDiscovery(str(repo), gfx_arch="gfx950")
        kernels = kd.discover_all()

        a8w8 = next(k for k in kernels if k.name == "a8w8")
        assert len(a8w8.config_files) == 1
        assert str(expected) in a8w8.config_files


# ---------------------------------------------------------------------------
# Tests: script discovery
# ---------------------------------------------------------------------------


class TestScriptDiscovery:
    """Validate that ut / bench / test scripts are located."""

    def _basic_a8w8_repo(self, tmp_path: pathlib.Path) -> pathlib.Path:
        repo = _make_mock_repo(tmp_path)
        _touch(repo / "aiter" / "ops" / "triton" / "gemm" / "basic" / "gemm_a8w8.py")
        return repo

    def test_ut_script_found(self, tmp_path: pathlib.Path) -> None:
        repo = self._basic_a8w8_repo(tmp_path)
        expected = _touch(
            repo
            / "aiter"
            / "ops"
            / "triton"
            / "utils"
            / "_triton"
            / "tunning"
            / "ut_a8w8_gemm.py"
        )

        kd = KernelDiscovery(str(repo))
        kernels = kd.discover_all()

        a8w8 = next(k for k in kernels if k.name == "a8w8")
        assert a8w8.ut_script == str(expected)

    def test_ut_script_missing(self, tmp_path: pathlib.Path) -> None:
        repo = self._basic_a8w8_repo(tmp_path)

        kd = KernelDiscovery(str(repo))
        kernels = kd.discover_all()

        a8w8 = next(k for k in kernels if k.name == "a8w8")
        assert a8w8.ut_script is None

    def test_bench_script_found_primary_pattern(self, tmp_path: pathlib.Path) -> None:
        repo = self._basic_a8w8_repo(tmp_path)
        expected = _touch(
            repo / "op_tests" / "op_benchmarks" / "triton" / "bench_gemm_a8w8.py"
        )

        kd = KernelDiscovery(str(repo))
        kernels = kd.discover_all()

        a8w8 = next(k for k in kernels if k.name == "a8w8")
        assert a8w8.bench_script == str(expected)

    def test_bench_script_found_fallback_pattern(self, tmp_path: pathlib.Path) -> None:
        """Fallback pattern: bench_{category}_{name}.py"""
        repo = self._basic_a8w8_repo(tmp_path)
        expected = _touch(
            repo / "op_tests" / "op_benchmarks" / "triton" / "bench_basic_a8w8.py"
        )

        kd = KernelDiscovery(str(repo))
        kernels = kd.discover_all()

        a8w8 = next(k for k in kernels if k.name == "a8w8")
        assert a8w8.bench_script == str(expected)

    def test_bench_script_missing(self, tmp_path: pathlib.Path) -> None:
        repo = self._basic_a8w8_repo(tmp_path)

        kd = KernelDiscovery(str(repo))
        kernels = kd.discover_all()

        a8w8 = next(k for k in kernels if k.name == "a8w8")
        assert a8w8.bench_script is None

    def test_test_script_found_in_subdirectory(self, tmp_path: pathlib.Path) -> None:
        repo = self._basic_a8w8_repo(tmp_path)
        expected = _touch(
            repo
            / "op_tests"
            / "triton_tests"
            / "gemm"
            / "basic"
            / "test_gemm_a8w8.py"
        )

        kd = KernelDiscovery(str(repo))
        kernels = kd.discover_all()

        a8w8 = next(k for k in kernels if k.name == "a8w8")
        assert a8w8.test_script == str(expected)

    def test_test_script_missing(self, tmp_path: pathlib.Path) -> None:
        repo = self._basic_a8w8_repo(tmp_path)

        kd = KernelDiscovery(str(repo))
        kernels = kd.discover_all()

        a8w8 = next(k for k in kernels if k.name == "a8w8")
        assert a8w8.test_script is None


# ---------------------------------------------------------------------------
# Tests: has_all_scripts property
# ---------------------------------------------------------------------------


class TestHasAllScripts:
    """Validate the has_all_scripts derived property."""

    def test_has_all_scripts_true(self, tmp_path: pathlib.Path) -> None:
        repo = _make_mock_repo(tmp_path)
        _touch(repo / "aiter" / "ops" / "triton" / "gemm" / "basic" / "gemm_a8w8.py")
        _touch(
            repo
            / "aiter"
            / "ops"
            / "triton"
            / "utils"
            / "_triton"
            / "tunning"
            / "ut_a8w8_gemm.py"
        )
        _touch(
            repo / "op_tests" / "op_benchmarks" / "triton" / "bench_gemm_a8w8.py"
        )
        _touch(
            repo
            / "op_tests"
            / "triton_tests"
            / "gemm"
            / "basic"
            / "test_gemm_a8w8.py"
        )

        kd = KernelDiscovery(str(repo))
        kernels = kd.discover_all()

        a8w8 = next(k for k in kernels if k.name == "a8w8")
        assert a8w8.has_all_scripts is True

    def test_has_all_scripts_false_missing_bench(self, tmp_path: pathlib.Path) -> None:
        repo = _make_mock_repo(tmp_path)
        _touch(repo / "aiter" / "ops" / "triton" / "gemm" / "basic" / "gemm_a8w8.py")
        _touch(
            repo
            / "aiter"
            / "ops"
            / "triton"
            / "utils"
            / "_triton"
            / "tunning"
            / "ut_a8w8_gemm.py"
        )
        _touch(
            repo
            / "op_tests"
            / "triton_tests"
            / "gemm"
            / "basic"
            / "test_gemm_a8w8.py"
        )
        # bench_script intentionally omitted

        kd = KernelDiscovery(str(repo))
        kernels = kd.discover_all()

        a8w8 = next(k for k in kernels if k.name == "a8w8")
        assert a8w8.has_all_scripts is False

    def test_has_all_scripts_false_no_scripts(self, tmp_path: pathlib.Path) -> None:
        repo = _make_mock_repo(tmp_path)
        _touch(repo / "aiter" / "ops" / "triton" / "gemm" / "basic" / "gemm_a8w8.py")

        kd = KernelDiscovery(str(repo))
        kernels = kd.discover_all()

        a8w8 = next(k for k in kernels if k.name == "a8w8")
        assert a8w8.has_all_scripts is False

    def test_has_all_scripts_standalone_dataclass(self) -> None:
        """Verify the property works on a manually constructed dataclass."""
        kernel = DiscoveredKernel(
            name="a8w8",
            category=GemmCategory.BASIC,
            source_path="/fake/gemm_a8w8.py",
            config_variant="A8W8",
            ut_script="/fake/ut.py",
            bench_script="/fake/bench.py",
            test_script="/fake/test.py",
        )
        assert kernel.has_all_scripts is True

        kernel.bench_script = None
        assert kernel.has_all_scripts is False


# ---------------------------------------------------------------------------
# Tests: filters
# ---------------------------------------------------------------------------


class TestFilters:
    """Validate include/exclude filter behaviour."""

    def _repo_with_two_basic_kernels(
        self, tmp_path: pathlib.Path
    ) -> pathlib.Path:
        repo = _make_mock_repo(tmp_path)
        basic_dir = repo / "aiter" / "ops" / "triton" / "gemm" / "basic"
        _touch(basic_dir / "gemm_a8w8.py")
        _touch(basic_dir / "gemm_a16w16.py")
        return repo

    def test_exclude_removes_kernel(self, tmp_path: pathlib.Path) -> None:
        repo = self._repo_with_two_basic_kernels(tmp_path)

        kd = KernelDiscovery(str(repo))
        kernels = kd.discover_all(exclude=["a8w8"])

        names = [k.name for k in kernels]
        assert "a8w8" not in names
        assert "a16w16" in names

    def test_exclude_empty_list_returns_all(self, tmp_path: pathlib.Path) -> None:
        repo = self._repo_with_two_basic_kernels(tmp_path)

        kd = KernelDiscovery(str(repo))
        kernels = kd.discover_all(exclude=[])

        names = {k.name for k in kernels}
        assert {"a8w8", "a16w16"}.issubset(names)

    def test_include_keeps_only_specified(self, tmp_path: pathlib.Path) -> None:
        repo = self._repo_with_two_basic_kernels(tmp_path)

        kd = KernelDiscovery(str(repo))
        kernels = kd.discover_all(include=["a8w8"])

        names = [k.name for k in kernels]
        assert names == ["a8w8"], f"Expected only ['a8w8'], got {names}"

    def test_include_empty_list_returns_nothing(self, tmp_path: pathlib.Path) -> None:
        repo = self._repo_with_two_basic_kernels(tmp_path)

        kd = KernelDiscovery(str(repo))
        kernels = kd.discover_all(include=[])

        assert kernels == []

    def test_include_overrides_exclude_semantics(
        self, tmp_path: pathlib.Path
    ) -> None:
        """When include is given, kernels not in include are dropped regardless."""
        repo = self._repo_with_two_basic_kernels(tmp_path)

        kd = KernelDiscovery(str(repo))
        # Both exclude and include are supplied; include wins by filtering first.
        kernels = kd.discover_all(include=["a8w8"], exclude=["a16w16"])

        names = [k.name for k in kernels]
        assert names == ["a8w8"]

    def test_exclude_unknown_name_is_noop(self, tmp_path: pathlib.Path) -> None:
        repo = self._repo_with_two_basic_kernels(tmp_path)

        kd = KernelDiscovery(str(repo))
        kernels_all = kd.discover_all()
        kernels_exc = kd.discover_all(exclude=["nonexistent_kernel"])

        assert len(kernels_all) == len(kernels_exc)


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary conditions and robustness checks."""

    def test_empty_repo_returns_empty_list(self, tmp_path: pathlib.Path) -> None:
        """Missing source directories should not raise; return empty list."""
        repo = tmp_path / "empty_repo"
        repo.mkdir()

        kd = KernelDiscovery(str(repo))
        kernels = kd.discover_all()
        assert kernels == []

    def test_missing_category_directory_skipped(
        self, tmp_path: pathlib.Path
    ) -> None:
        """If only the basic dir exists, other categories return nothing."""
        repo = _make_mock_repo(tmp_path)
        _touch(repo / "aiter" / "ops" / "triton" / "gemm" / "basic" / "gemm_a8w8.py")

        kd = KernelDiscovery(str(repo))
        kernels = kd.discover_all()

        categories = {k.category for k in kernels}
        assert categories == {GemmCategory.BASIC}

    def test_results_sorted_by_category_then_name(
        self, tmp_path: pathlib.Path
    ) -> None:
        repo = _make_mock_repo(tmp_path)
        basic_dir = repo / "aiter" / "ops" / "triton" / "gemm" / "basic"
        batched_dir = repo / "aiter" / "ops" / "triton" / "gemm" / "batched"
        _touch(basic_dir / "gemm_z_kernel.py")
        _touch(basic_dir / "gemm_a_kernel.py")
        _touch(batched_dir / "batched_gemm_b_kernel.py")

        kd = KernelDiscovery(str(repo))
        kernels = kd.discover_all()

        # Basic comes before batched in the enum ordering
        categories_in_order = [k.category.value for k in kernels]
        basic_idx = max(
            i for i, v in enumerate(categories_in_order) if v == "basic"
        )
        batched_idx = min(
            i for i, v in enumerate(categories_in_order) if v == "batched"
        )
        assert basic_idx < batched_idx, "basic kernels should appear before batched"

        # Within basic, names should be sorted
        basic_names = [k.name for k in kernels if k.category == GemmCategory.BASIC]
        assert basic_names == sorted(basic_names)

    def test_non_py_files_in_source_dir_ignored(
        self, tmp_path: pathlib.Path
    ) -> None:
        repo = _make_mock_repo(tmp_path)
        basic_dir = repo / "aiter" / "ops" / "triton" / "gemm" / "basic"
        _touch(basic_dir / "gemm_a8w8.py")
        _touch(basic_dir / "gemm_a8w8.pyc")  # should be ignored (doesn't end with .py)
        _touch(basic_dir / "gemm_a8w8.txt")  # should be ignored

        kd = KernelDiscovery(str(repo))
        kernels = kd.discover_all()

        assert len(kernels) == 1
        assert kernels[0].name == "a8w8"

    def test_gfx_arch_parameter_used_in_config_search(
        self, tmp_path: pathlib.Path
    ) -> None:
        repo = _make_mock_repo(tmp_path)
        _touch(repo / "aiter" / "ops" / "triton" / "gemm" / "basic" / "gemm_a8w8.py")
        cfg_dir = repo / "aiter" / "ops" / "triton" / "configs" / "gemm"
        _touch(cfg_dir / "gfx942-GEMM-A8W8-M_LEQ_16.json")

        kd = KernelDiscovery(str(repo), gfx_arch="gfx942")
        kernels = kd.discover_all()

        a8w8 = next(k for k in kernels if k.name == "a8w8")
        assert len(a8w8.config_files) == 1

    def test_all_categories_scanned_in_single_call(
        self, tmp_path: pathlib.Path
    ) -> None:
        repo = _make_mock_repo(tmp_path)
        _touch(repo / "aiter" / "ops" / "triton" / "gemm" / "basic" / "gemm_a8w8.py")
        _touch(
            repo / "aiter" / "ops" / "triton" / "gemm" / "batched" / "batched_gemm_a8w8.py"
        )
        _touch(
            repo / "aiter" / "ops" / "triton" / "gemm" / "feed_forward" / "ff_a8w8.py"
        )
        _touch(
            repo / "aiter" / "ops" / "triton" / "gemm" / "fused" / "fused_gemm_a8w8.py"
        )

        kd = KernelDiscovery(str(repo))
        kernels = kd.discover_all()

        categories = {k.category for k in kernels}
        assert categories == set(GemmCategory)
