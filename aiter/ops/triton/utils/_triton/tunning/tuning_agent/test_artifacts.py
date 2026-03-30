"""Tests for ArtifactManager (Task 7)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from .artifacts import ArtifactManager
from .types import ShapeResult


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def make_manager(
    tmp_path: Path,
    run_id: str = "20240101_120000",
    kernel_name: str = "gemm",
    remote_base_dir: str = "/workspace/tuning_artifacts",
) -> ArtifactManager:
    """Return an ArtifactManager with a mock executor and a tmp local dir."""
    executor = MagicMock()
    mgr = ArtifactManager(
        executor=executor,
        kernel_name=kernel_name,
        run_id=run_id,
        local_base_dir=str(tmp_path),
        remote_base_dir=remote_base_dir,
    )
    return mgr


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_local_dir_contains_run_id_and_kernel_name(self, tmp_path: Path) -> None:
        mgr = make_manager(tmp_path, run_id="20240101_120000", kernel_name="flash_attn")
        assert mgr.local_dir == tmp_path / "20240101_120000" / "flash_attn"

    def test_remote_dir_contains_kernel_name(self, tmp_path: Path) -> None:
        mgr = make_manager(
            tmp_path,
            kernel_name="gemm",
            remote_base_dir="/workspace/tuning_artifacts",
        )
        assert mgr.remote_dir == "/workspace/tuning_artifacts/gemm"

    def test_run_id_generated_from_datetime_when_none(self, tmp_path: Path) -> None:
        executor = MagicMock()
        mgr = ArtifactManager(
            executor=executor,
            kernel_name="k",
            run_id=None,
            local_base_dir=str(tmp_path),
        )
        # run_id should look like YYYYMMDD_HHMMSS (15 chars)
        assert len(mgr.run_id) == 15
        assert mgr.run_id[8] == "_"

    def test_explicit_run_id_is_preserved(self, tmp_path: Path) -> None:
        mgr = make_manager(tmp_path, run_id="custom_run_42")
        assert mgr.run_id == "custom_run_42"

    def test_executor_is_stored(self, tmp_path: Path) -> None:
        mgr = make_manager(tmp_path)
        assert mgr.executor is not None


# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------


class TestSetupLocal:
    def test_creates_local_dir(self, tmp_path: Path) -> None:
        mgr = make_manager(tmp_path)
        assert not mgr.local_dir.exists()
        mgr.setup_local()
        assert mgr.local_dir.is_dir()

    def test_creates_nested_parents(self, tmp_path: Path) -> None:
        mgr = make_manager(tmp_path, run_id="run1", kernel_name="deep/kernel")
        mgr.setup_local()
        assert mgr.local_dir.is_dir()

    def test_idempotent(self, tmp_path: Path) -> None:
        mgr = make_manager(tmp_path)
        mgr.setup_local()
        # Should not raise on second call
        mgr.setup_local()
        assert mgr.local_dir.is_dir()


class TestSetupRemote:
    def test_calls_docker_exec_with_mkdir(self, tmp_path: Path) -> None:
        mgr = make_manager(tmp_path, kernel_name="gemm")
        mgr.setup_remote()
        mgr.executor.docker_exec.assert_called_once_with(
            f"mkdir -p {mgr.remote_dir}"
        )

    def test_remote_dir_path_in_command(self, tmp_path: Path) -> None:
        mgr = make_manager(
            tmp_path,
            kernel_name="flash_attn",
            remote_base_dir="/custom/artifacts",
        )
        mgr.setup_remote()
        cmd_arg = mgr.executor.docker_exec.call_args[0][0]
        assert "/custom/artifacts/flash_attn" in cmd_arg


# ---------------------------------------------------------------------------
# Save / load results
# ---------------------------------------------------------------------------


SAMPLE_RESULTS = [
    ShapeResult(m=128, n=256, k=64, main_ns=1000.0, reduce_ns=50.0),
    ShapeResult(m=256, n=512, k=128, main_ns=2000.0, reduce_ns=0.0),
]


class TestSaveResults:
    def test_returns_path_under_local_dir(self, tmp_path: Path) -> None:
        mgr = make_manager(tmp_path)
        mgr.setup_local()
        path = mgr.save_results("phase1", SAMPLE_RESULTS)
        assert path == mgr.local_dir / "phase1.json"

    def test_file_exists_after_save(self, tmp_path: Path) -> None:
        mgr = make_manager(tmp_path)
        mgr.setup_local()
        path = mgr.save_results("phase1", SAMPLE_RESULTS)
        assert path.exists()

    def test_json_structure_contains_required_fields(self, tmp_path: Path) -> None:
        mgr = make_manager(tmp_path)
        mgr.setup_local()
        path = mgr.save_results("run", SAMPLE_RESULTS)
        records = json.loads(path.read_text())
        assert len(records) == 2
        for rec in records:
            for field in ("m", "n", "k", "main_ns", "reduce_ns", "total_ns"):
                assert field in rec, f"Missing field: {field}"

    def test_total_ns_equals_main_plus_reduce(self, tmp_path: Path) -> None:
        mgr = make_manager(tmp_path)
        mgr.setup_local()
        path = mgr.save_results("run", SAMPLE_RESULTS)
        records = json.loads(path.read_text())
        for rec in records:
            assert rec["total_ns"] == pytest.approx(rec["main_ns"] + rec["reduce_ns"])

    def test_values_match_input(self, tmp_path: Path) -> None:
        mgr = make_manager(tmp_path)
        mgr.setup_local()
        path = mgr.save_results("run", SAMPLE_RESULTS)
        records = json.loads(path.read_text())
        first = records[0]
        assert first["m"] == 128
        assert first["n"] == 256
        assert first["k"] == 64
        assert first["main_ns"] == pytest.approx(1000.0)
        assert first["reduce_ns"] == pytest.approx(50.0)
        assert first["total_ns"] == pytest.approx(1050.0)


class TestLoadResults:
    def test_roundtrip_length(self, tmp_path: Path) -> None:
        mgr = make_manager(tmp_path)
        mgr.setup_local()
        mgr.save_results("run", SAMPLE_RESULTS)
        loaded = mgr.load_results("run")
        assert len(loaded) == len(SAMPLE_RESULTS)

    def test_roundtrip_reconstructs_shape_result(self, tmp_path: Path) -> None:
        mgr = make_manager(tmp_path)
        mgr.setup_local()
        mgr.save_results("run", SAMPLE_RESULTS)
        loaded = mgr.load_results("run")
        assert all(isinstance(r, ShapeResult) for r in loaded)

    def test_roundtrip_values_preserved(self, tmp_path: Path) -> None:
        mgr = make_manager(tmp_path)
        mgr.setup_local()
        mgr.save_results("run", SAMPLE_RESULTS)
        loaded = mgr.load_results("run")
        for original, restored in zip(SAMPLE_RESULTS, loaded):
            assert restored.m == original.m
            assert restored.n == original.n
            assert restored.k == original.k
            assert restored.main_ns == pytest.approx(original.main_ns)
            assert restored.reduce_ns == pytest.approx(original.reduce_ns)

    def test_roundtrip_total_ns_property(self, tmp_path: Path) -> None:
        mgr = make_manager(tmp_path)
        mgr.setup_local()
        mgr.save_results("run", SAMPLE_RESULTS)
        loaded = mgr.load_results("run")
        for original, restored in zip(SAMPLE_RESULTS, loaded):
            assert restored.total_ns == pytest.approx(original.total_ns)

    def test_empty_results_roundtrip(self, tmp_path: Path) -> None:
        mgr = make_manager(tmp_path)
        mgr.setup_local()
        mgr.save_results("empty", [])
        loaded = mgr.load_results("empty")
        assert loaded == []


# ---------------------------------------------------------------------------
# Phase checkpoints
# ---------------------------------------------------------------------------


class TestPhaseCheckpoints:
    def test_is_phase_complete_false_before_marking(self, tmp_path: Path) -> None:
        mgr = make_manager(tmp_path)
        mgr.setup_local()
        assert mgr.is_phase_complete(1) is False

    def test_is_phase_complete_true_after_marking(self, tmp_path: Path) -> None:
        mgr = make_manager(tmp_path)
        mgr.setup_local()
        mgr.mark_phase_complete(1, {"shapes_tuned": 10})
        assert mgr.is_phase_complete(1) is True

    def test_marker_file_created(self, tmp_path: Path) -> None:
        mgr = make_manager(tmp_path)
        mgr.setup_local()
        mgr.mark_phase_complete(2, {})
        assert (mgr.local_dir / "phase_2_complete.json").exists()

    def test_marker_contains_timestamp(self, tmp_path: Path) -> None:
        mgr = make_manager(tmp_path)
        mgr.setup_local()
        mgr.mark_phase_complete(1, {})
        payload = json.loads((mgr.local_dir / "phase_1_complete.json").read_text())
        assert "timestamp" in payload

    def test_marker_contains_summary_fields(self, tmp_path: Path) -> None:
        mgr = make_manager(tmp_path)
        mgr.setup_local()
        mgr.mark_phase_complete(3, {"shapes_tuned": 42, "best_config": "x128"})
        payload = json.loads((mgr.local_dir / "phase_3_complete.json").read_text())
        assert payload["shapes_tuned"] == 42
        assert payload["best_config"] == "x128"

    def test_phases_are_independent(self, tmp_path: Path) -> None:
        mgr = make_manager(tmp_path)
        mgr.setup_local()
        mgr.mark_phase_complete(1, {})
        assert mgr.is_phase_complete(1) is True
        assert mgr.is_phase_complete(2) is False

    def test_get_phase_summary_returns_none_when_not_complete(
        self, tmp_path: Path
    ) -> None:
        mgr = make_manager(tmp_path)
        mgr.setup_local()
        assert mgr.get_phase_summary(1) is None

    def test_get_phase_summary_returns_dict_after_marking(
        self, tmp_path: Path
    ) -> None:
        mgr = make_manager(tmp_path)
        mgr.setup_local()
        mgr.mark_phase_complete(1, {"info": "done"})
        summary = mgr.get_phase_summary(1)
        assert summary is not None
        assert isinstance(summary, dict)
        assert summary["info"] == "done"

    def test_get_phase_summary_includes_timestamp(self, tmp_path: Path) -> None:
        mgr = make_manager(tmp_path)
        mgr.setup_local()
        mgr.mark_phase_complete(2, {})
        summary = mgr.get_phase_summary(2)
        assert "timestamp" in summary


# ---------------------------------------------------------------------------
# File transfer
# ---------------------------------------------------------------------------


class TestFetchRemoteFile:
    def test_calls_copy_from_container(self, tmp_path: Path) -> None:
        mgr = make_manager(tmp_path)
        mgr.setup_local()
        mgr.fetch_remote_file("/container/path/results.json", "results.json")
        mgr.executor.copy_from_container.assert_called_once_with(
            "/container/path/results.json",
            str(mgr.local_dir / "results.json"),
        )

    def test_returns_local_path(self, tmp_path: Path) -> None:
        mgr = make_manager(tmp_path)
        mgr.setup_local()
        result = mgr.fetch_remote_file("/remote/file.txt", "file.txt")
        assert result == mgr.local_dir / "file.txt"


class TestPushLocalFile:
    def test_calls_copy_to_container(self, tmp_path: Path) -> None:
        mgr = make_manager(tmp_path)
        mgr.setup_local()
        mgr.push_local_file("config.json", "/container/config.json")
        mgr.executor.copy_to_container.assert_called_once_with(
            str(mgr.local_dir / "config.json"),
            "/container/config.json",
        )

    def test_uses_local_dir_as_source_base(self, tmp_path: Path) -> None:
        mgr = make_manager(tmp_path, run_id="r1", kernel_name="k1")
        mgr.setup_local()
        mgr.push_local_file("data.bin", "/remote/data.bin")
        src_arg = mgr.executor.copy_to_container.call_args[0][0]
        assert src_arg == str(tmp_path / "r1" / "k1" / "data.bin")
