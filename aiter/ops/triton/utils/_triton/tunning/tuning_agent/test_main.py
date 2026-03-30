"""Tests for the CLI entry point (__main__.py) — written TDD-style."""

from __future__ import annotations

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE = "aiter.ops.triton.utils._triton.tunning.tuning_agent"

MINIMAL_YAML = """\
baseline:
  aiter_repo: "https://github.com/ROCm/aiter.git"
  aiter_branch: "main"
  triton_repo: "https://github.com/triton-lang/triton.git"
  triton_branch: "main"
target:
  aiter_repo: "https://github.com/ROCm/aiter.git"
  aiter_branch: "dev"
  triton_repo: "https://github.com/triton-lang/triton.git"
  triton_branch: "dev"
machines:
  - host: "gpu-node-01.example.com"
    user: "ci"
    ssh_key: "/home/ci/.ssh/id_rsa"
    gpus: [0]
container:
  image: "rocm/pytorch:latest"
"""


def _write_config(tmp_path: Path) -> str:
    """Write a minimal valid YAML config and return its path."""
    cfg = tmp_path / "triton-upgrade.yaml"
    cfg.write_text(MINIMAL_YAML)
    return str(cfg)


def _make_repo_structure(root: Path) -> Path:
    """Create the marker directory structure under *root*."""
    marker = root / "aiter" / "ops" / "triton" / "gemm"
    marker.mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# find_repo_root tests
# ---------------------------------------------------------------------------


class TestFindRepoRoot:
    """find_repo_root() locates the repo root by walking up from a start path."""

    def test_find_repo_root_from_subdir(self, tmp_path):
        """Should find repo root when called from a subdirectory."""
        from aiter.ops.triton.utils._triton.tunning.tuning_agent.__main__ import (
            find_repo_root,
        )

        repo_root = _make_repo_structure(tmp_path / "myrepo")
        # start from a deep subdirectory inside the repo
        start = repo_root / "some" / "deep" / "subdir"
        start.mkdir(parents=True, exist_ok=True)
        result = find_repo_root(str(start))
        assert os.path.normpath(result) == os.path.normpath(str(repo_root))

    def test_find_repo_root_from_repo_root_itself(self, tmp_path):
        """Should find repo root when called from the repo root directly."""
        from aiter.ops.triton.utils._triton.tunning.tuning_agent.__main__ import (
            find_repo_root,
        )

        repo_root = _make_repo_structure(tmp_path / "myrepo")
        result = find_repo_root(str(repo_root))
        assert os.path.normpath(result) == os.path.normpath(str(repo_root))

    def test_find_repo_root_not_found(self, tmp_path):
        """Should raise FileNotFoundError when marker dir is absent."""
        from aiter.ops.triton.utils._triton.tunning.tuning_agent.__main__ import (
            find_repo_root,
        )

        # tmp_path has no aiter/ops/triton/gemm/ structure
        with pytest.raises(FileNotFoundError, match="aiter/ops/triton/gemm"):
            find_repo_root(str(tmp_path))

    def test_find_repo_root_default_uses_cwd(self, tmp_path, monkeypatch):
        """When start_path is None, should use os.getcwd()."""
        from aiter.ops.triton.utils._triton.tunning.tuning_agent.__main__ import (
            find_repo_root,
        )

        repo_root = _make_repo_structure(tmp_path / "myrepo")
        monkeypatch.chdir(str(repo_root))
        result = find_repo_root()
        assert os.path.normpath(result) == os.path.normpath(str(repo_root))


# ---------------------------------------------------------------------------
# Argument parsing tests
# ---------------------------------------------------------------------------


class TestParseArgs:
    """CLI argument parsing behaves as documented."""

    def _parse(self, argv):
        """Import and call the module-level _build_parser().parse_args()."""
        from aiter.ops.triton.utils._triton.tunning.tuning_agent.__main__ import (
            _build_parser,
        )

        return _build_parser().parse_args(argv)

    def test_config_required(self):
        """--config is required; omitting it should raise SystemExit."""
        from aiter.ops.triton.utils._triton.tunning.tuning_agent.__main__ import (
            _build_parser,
        )

        with pytest.raises(SystemExit):
            _build_parser().parse_args([])

    def test_parse_all_args(self, tmp_path):
        """All documented arguments are parsed correctly."""
        cfg_path = _write_config(tmp_path)
        ns = self._parse(
            [
                "--config", cfg_path,
                "--dry-run",
                "--run-id", "my-run-001",
                "--repo-root", "/some/path",
                "--no-dashboard",
                "--log-level", "DEBUG",
                "--results-dir", "/tmp/results",
            ]
        )
        assert ns.config == cfg_path
        assert ns.dry_run is True
        assert ns.run_id == "my-run-001"
        assert ns.repo_root == "/some/path"
        assert ns.no_dashboard is True
        assert ns.log_level == "DEBUG"
        assert ns.results_dir == "/tmp/results"

    def test_defaults(self, tmp_path):
        """Omitting optional args gives expected defaults."""
        cfg_path = _write_config(tmp_path)
        ns = self._parse(["--config", cfg_path])
        assert ns.dry_run is False
        assert ns.run_id is None
        assert ns.repo_root is None
        assert ns.no_dashboard is False
        assert ns.log_level == "INFO"
        assert ns.results_dir is None

    def test_dry_run_is_flag(self, tmp_path):
        """--dry-run without value sets dry_run=True."""
        cfg_path = _write_config(tmp_path)
        ns = self._parse(["--config", cfg_path, "--dry-run"])
        assert ns.dry_run is True

    def test_no_dashboard_is_flag(self, tmp_path):
        """--no-dashboard without value sets no_dashboard=True."""
        cfg_path = _write_config(tmp_path)
        ns = self._parse(["--config", cfg_path, "--no-dashboard"])
        assert ns.no_dashboard is True


# ---------------------------------------------------------------------------
# main() behaviour tests
# ---------------------------------------------------------------------------


class TestDryRunMode:
    """In --dry-run mode, discover_kernels() is called but run() is not."""

    def test_dry_run_calls_discover_kernels_not_run(self, tmp_path):
        from aiter.ops.triton.utils._triton.tunning.tuning_agent.__main__ import main

        cfg_path = _write_config(tmp_path)
        repo_root = _make_repo_structure(tmp_path / "repo")

        mock_orchestrator = MagicMock()
        mock_orchestrator.discover_kernels.return_value = []

        with (
            patch(
                f"{BASE}.__main__.Orchestrator",
                return_value=mock_orchestrator,
            ),
            patch(
                f"{BASE}.__main__.find_repo_root",
                return_value=str(repo_root),
            ),
        ):
            main(["--config", cfg_path, "--dry-run", "--no-dashboard"])

        mock_orchestrator.discover_kernels.assert_called_once()
        mock_orchestrator.run.assert_not_called()

    def test_dry_run_prints_summary(self, tmp_path, capsys):
        from aiter.ops.triton.utils._triton.tunning.tuning_agent.__main__ import main

        cfg_path = _write_config(tmp_path)
        repo_root = _make_repo_structure(tmp_path / "repo")

        mock_kernel = MagicMock()
        mock_kernel.name = "gemm_kernel_v1"

        mock_orchestrator = MagicMock()
        mock_orchestrator.discover_kernels.return_value = [mock_kernel]

        with (
            patch(
                f"{BASE}.__main__.Orchestrator",
                return_value=mock_orchestrator,
            ),
            patch(
                f"{BASE}.__main__.find_repo_root",
                return_value=str(repo_root),
            ),
        ):
            main(["--config", cfg_path, "--dry-run", "--no-dashboard"])

        captured = capsys.readouterr()
        # The summary should mention the kernel count
        assert "1" in captured.out or "gemm_kernel_v1" in captured.out


class TestNormalRunMode:
    """Without --dry-run, orchestrator.run() is called."""

    def test_normal_run_calls_run(self, tmp_path):
        from aiter.ops.triton.utils._triton.tunning.tuning_agent.__main__ import main

        cfg_path = _write_config(tmp_path)
        repo_root = _make_repo_structure(tmp_path / "repo")

        mock_orchestrator = MagicMock()
        mock_orchestrator.run.return_value = {}

        with (
            patch(
                f"{BASE}.__main__.Orchestrator",
                return_value=mock_orchestrator,
            ),
            patch(
                f"{BASE}.__main__.find_repo_root",
                return_value=str(repo_root),
            ),
        ):
            main(["--config", cfg_path, "--no-dashboard"])

        mock_orchestrator.run.assert_called_once()
        mock_orchestrator.discover_kernels.assert_not_called()


class TestRunIdGeneration:
    """Auto-generated run IDs follow the datetime pattern."""

    def test_auto_run_id_is_generated_when_not_provided(self, tmp_path):
        from aiter.ops.triton.utils._triton.tunning.tuning_agent.__main__ import main

        cfg_path = _write_config(tmp_path)
        repo_root = _make_repo_structure(tmp_path / "repo")

        captured_kwargs = {}

        def fake_orchestrator(*args, **kwargs):
            captured_kwargs.update(kwargs)
            m = MagicMock()
            m.run.return_value = {}
            return m

        with (
            patch(
                f"{BASE}.__main__.Orchestrator",
                side_effect=fake_orchestrator,
            ),
            patch(
                f"{BASE}.__main__.find_repo_root",
                return_value=str(repo_root),
            ),
        ):
            main(["--config", cfg_path, "--no-dashboard"])

        # run_id should have been passed and be a non-empty string
        assert "run_id" in captured_kwargs
        assert isinstance(captured_kwargs["run_id"], str)
        assert len(captured_kwargs["run_id"]) > 0

    def test_explicit_run_id_is_used(self, tmp_path):
        from aiter.ops.triton.utils._triton.tunning.tuning_agent.__main__ import main

        cfg_path = _write_config(tmp_path)
        repo_root = _make_repo_structure(tmp_path / "repo")

        captured_kwargs = {}

        def fake_orchestrator(*args, **kwargs):
            captured_kwargs.update(kwargs)
            m = MagicMock()
            m.run.return_value = {}
            return m

        with (
            patch(
                f"{BASE}.__main__.Orchestrator",
                side_effect=fake_orchestrator,
            ),
            patch(
                f"{BASE}.__main__.find_repo_root",
                return_value=str(repo_root),
            ),
        ):
            main(["--config", cfg_path, "--no-dashboard", "--run-id", "explicit-id"])

        assert captured_kwargs.get("run_id") == "explicit-id"


class TestNoDashboard:
    """--no-dashboard prevents Dashboard creation."""

    def test_dashboard_not_created_with_flag(self, tmp_path):
        from aiter.ops.triton.utils._triton.tunning.tuning_agent.__main__ import main

        cfg_path = _write_config(tmp_path)
        repo_root = _make_repo_structure(tmp_path / "repo")

        mock_orchestrator = MagicMock()
        mock_orchestrator.run.return_value = {}

        with (
            patch(f"{BASE}.__main__.Orchestrator", return_value=mock_orchestrator),
            patch(f"{BASE}.__main__.find_repo_root", return_value=str(repo_root)),
            patch(f"{BASE}.__main__.Dashboard") as mock_dashboard_cls,
        ):
            main(["--config", cfg_path, "--no-dashboard"])

        mock_dashboard_cls.assert_not_called()

    def test_dashboard_created_without_flag(self, tmp_path):
        from aiter.ops.triton.utils._triton.tunning.tuning_agent.__main__ import main

        cfg_path = _write_config(tmp_path)
        repo_root = _make_repo_structure(tmp_path / "repo")

        mock_orchestrator = MagicMock()
        mock_orchestrator.run.return_value = {}
        mock_dashboard = MagicMock()

        with (
            patch(f"{BASE}.__main__.Orchestrator", return_value=mock_orchestrator),
            patch(f"{BASE}.__main__.find_repo_root", return_value=str(repo_root)),
            patch(f"{BASE}.__main__.Dashboard", return_value=mock_dashboard) as mock_dashboard_cls,
        ):
            main(["--config", cfg_path])

        mock_dashboard_cls.assert_called_once()


class TestRepoRootOverride:
    """--repo-root bypasses find_repo_root()."""

    def test_explicit_repo_root_bypasses_auto_detect(self, tmp_path):
        from aiter.ops.triton.utils._triton.tunning.tuning_agent.__main__ import main

        cfg_path = _write_config(tmp_path)

        mock_orchestrator = MagicMock()
        mock_orchestrator.run.return_value = {}

        with (
            patch(f"{BASE}.__main__.Orchestrator", return_value=mock_orchestrator),
            patch(f"{BASE}.__main__.find_repo_root") as mock_find,
        ):
            main(["--config", cfg_path, "--repo-root", "/explicit/root", "--no-dashboard"])

        mock_find.assert_not_called()
