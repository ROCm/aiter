"""Tests for KernelSupervisor types, state machine, and checkpoint logic."""

import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from .kernel_supervisor import (
    EscalationRequest,
    KernelSupervisor,
    Phase,
    PhaseResult,
    SupervisorConfig,
    SupervisorResult,
    SupervisorState,
    TuningMode,
)
from .types import ContainerConfig, RepoConfig, ShapeResult, TritonInstallConfig, TuningConfig
from .artifacts import ArtifactManager
from .notifications import Notifier
from .subagents.base import SubagentResult
from .subagents.tuning_agent import TuningAgent
from .subagents.pattern_analyzer_agent import PatternAnalyzerAgent
from .subagents.config_generator_agent import ConfigGeneratorAgent


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def make_repo_config() -> RepoConfig:
    return RepoConfig(
        aiter_repo="https://github.com/example/aiter",
        aiter_branch="main",
        triton_repo="https://github.com/example/triton",
        triton_branch="main",
    )


def make_supervisor_config(kernel_name: str = "gemm") -> SupervisorConfig:
    return SupervisorConfig(
        kernel_name=kernel_name,
        baseline_repo=make_repo_config(),
        target_repo=make_repo_config(),
        container_config=ContainerConfig(image="rocm/pytorch:latest"),
        triton_install=TritonInstallConfig(),
        tuning_config=TuningConfig(),
        gpu_ids=[0, 1],
    )


def make_supervisor(tmp_path: Path, kernel_name: str = "gemm") -> KernelSupervisor:
    """Return a KernelSupervisor with mock executor and real ArtifactManager."""
    executor = MagicMock()
    config = make_supervisor_config(kernel_name=kernel_name)
    artifact_manager = ArtifactManager(
        executor=executor,
        kernel_name=kernel_name,
        run_id="test_run_001",
        local_base_dir=str(tmp_path),
    )
    artifact_manager.setup_local()
    notifier = Notifier(auto_approve=True)
    return KernelSupervisor(
        executor=executor,
        config=config,
        artifact_manager=artifact_manager,
        notifier=notifier,
    )


# ---------------------------------------------------------------------------
# Phase enum
# ---------------------------------------------------------------------------


class TestPhaseEnum:
    def test_setup_value(self) -> None:
        assert Phase.SETUP == 0

    def test_discovery_value(self) -> None:
        assert Phase.DISCOVERY == 1

    def test_baseline_value(self) -> None:
        assert Phase.BASELINE == 2

    def test_untuned_validation_value(self) -> None:
        assert Phase.UNTUNED_VALIDATION == 3

    def test_tuning_value(self) -> None:
        assert Phase.TUNING == 4

    def test_validation_and_fix_value(self) -> None:
        assert Phase.VALIDATION_AND_FIX == 5

    def test_commit_value(self) -> None:
        assert Phase.COMMIT == 6

    def test_phases_are_ordered(self) -> None:
        phases = list(Phase)
        for i in range(len(phases) - 1):
            assert phases[i] < phases[i + 1]

    def test_total_phase_count(self) -> None:
        assert len(Phase) == 7

    def test_is_int_enum(self) -> None:
        assert isinstance(Phase.SETUP, int)


# ---------------------------------------------------------------------------
# TuningMode enum
# ---------------------------------------------------------------------------


class TestTuningMode:
    def test_regression_only_value(self) -> None:
        assert TuningMode.REGRESSION_ONLY == "regression_only"

    def test_full_value(self) -> None:
        assert TuningMode.FULL == "full"

    def test_is_str(self) -> None:
        assert isinstance(TuningMode.FULL, str)

    def test_two_modes_defined(self) -> None:
        assert len(TuningMode) == 2


# ---------------------------------------------------------------------------
# PhaseResult dataclass
# ---------------------------------------------------------------------------


class TestPhaseResult:
    def test_creation_with_required_fields(self) -> None:
        result = PhaseResult(
            phase=Phase.BASELINE,
            success=True,
            data={"shapes_tuned": 5},
            error=None,
            duration_seconds=12.5,
        )
        assert result.phase == Phase.BASELINE
        assert result.success is True
        assert result.data == {"shapes_tuned": 5}
        assert result.error is None
        assert result.duration_seconds == pytest.approx(12.5)

    def test_creation_with_error(self) -> None:
        result = PhaseResult(
            phase=Phase.TUNING,
            success=False,
            data={},
            error="Tuning failed: timeout",
            duration_seconds=3600.0,
        )
        assert result.success is False
        assert result.error == "Tuning failed: timeout"

    def test_data_field_accepts_dict(self) -> None:
        result = PhaseResult(
            phase=Phase.DISCOVERY,
            success=True,
            data={"key": "value", "count": 42},
            error=None,
            duration_seconds=1.0,
        )
        assert result.data["key"] == "value"
        assert result.data["count"] == 42


# ---------------------------------------------------------------------------
# EscalationRequest dataclass
# ---------------------------------------------------------------------------


class TestEscalationRequest:
    def test_creation_with_all_fields(self) -> None:
        req = EscalationRequest(
            severity="warning",
            message="Performance regression detected",
            details="gemm p50 is 8% slower than baseline",
        )
        assert req.severity == "warning"
        assert req.message == "Performance regression detected"
        assert req.details == "gemm p50 is 8% slower than baseline"

    def test_details_defaults_to_none(self) -> None:
        req = EscalationRequest(severity="fatal", message="Critical failure")
        assert req.details is None

    def test_severity_levels(self) -> None:
        for severity in ("warning", "approval_required", "fatal"):
            req = EscalationRequest(severity=severity, message="test")
            assert req.severity == severity


# ---------------------------------------------------------------------------
# SupervisorConfig dataclass
# ---------------------------------------------------------------------------


class TestSupervisorConfig:
    def test_creation_with_required_fields(self) -> None:
        config = make_supervisor_config("flash_attn")
        assert config.kernel_name == "flash_attn"
        assert isinstance(config.baseline_repo, RepoConfig)
        assert isinstance(config.target_repo, RepoConfig)
        assert isinstance(config.container_config, ContainerConfig)
        assert isinstance(config.triton_install, TritonInstallConfig)
        assert isinstance(config.tuning_config, TuningConfig)
        assert config.gpu_ids == [0, 1]

    def test_kernel_overrides_defaults_to_none(self) -> None:
        config = make_supervisor_config()
        assert config.kernel_overrides is None

    def test_kernel_overrides_accepts_dict(self) -> None:
        config = make_supervisor_config()
        config.kernel_overrides = {"m_leq_16_bucket_name": "small_m"}
        assert config.kernel_overrides["m_leq_16_bucket_name"] == "small_m"

    def test_gpu_ids_is_list(self) -> None:
        config = make_supervisor_config()
        assert isinstance(config.gpu_ids, list)


# ---------------------------------------------------------------------------
# SupervisorState dataclass
# ---------------------------------------------------------------------------


class TestSupervisorState:
    def test_default_current_phase_is_setup(self) -> None:
        state = SupervisorState()
        assert state.current_phase == Phase.SETUP

    def test_default_completed_phases_is_empty(self) -> None:
        state = SupervisorState()
        assert state.completed_phases == []

    def test_default_phase_results_is_empty_dict(self) -> None:
        state = SupervisorState()
        assert state.phase_results == {}

    def test_default_escalations_is_empty(self) -> None:
        state = SupervisorState()
        assert state.escalations == []

    def test_default_shapes_is_none(self) -> None:
        state = SupervisorState()
        assert state.shapes is None

    def test_default_baseline_data_is_none(self) -> None:
        state = SupervisorState()
        assert state.baseline_data is None

    def test_default_untuned_data_is_none(self) -> None:
        state = SupervisorState()
        assert state.untuned_data is None

    def test_default_tuned_data_is_none(self) -> None:
        state = SupervisorState()
        assert state.tuned_data is None

    def test_independent_mutable_defaults(self) -> None:
        """Each instance should have its own mutable collections."""
        s1 = SupervisorState()
        s2 = SupervisorState()
        s1.completed_phases.append(Phase.SETUP)
        assert Phase.SETUP not in s2.completed_phases

    def test_escalations_mutable_independence(self) -> None:
        s1 = SupervisorState()
        s2 = SupervisorState()
        s1.escalations.append(EscalationRequest(severity="warning", message="x"))
        assert s2.escalations == []


# ---------------------------------------------------------------------------
# KernelSupervisor.__init__
# ---------------------------------------------------------------------------


class TestKernelSupervisorInit:
    def test_stores_executor(self, tmp_path: Path) -> None:
        supervisor = make_supervisor(tmp_path)
        assert supervisor.executor is not None

    def test_stores_config(self, tmp_path: Path) -> None:
        supervisor = make_supervisor(tmp_path, kernel_name="gemm")
        assert supervisor.config.kernel_name == "gemm"

    def test_stores_artifact_manager(self, tmp_path: Path) -> None:
        supervisor = make_supervisor(tmp_path)
        assert isinstance(supervisor.artifact_manager, ArtifactManager)

    def test_stores_notifier(self, tmp_path: Path) -> None:
        supervisor = make_supervisor(tmp_path)
        assert isinstance(supervisor.notifier, Notifier)

    def test_progress_callback_defaults_to_none(self, tmp_path: Path) -> None:
        supervisor = make_supervisor(tmp_path)
        assert supervisor.progress_callback is None

    def test_progress_callback_stored_when_provided(self, tmp_path: Path) -> None:
        callback = MagicMock()
        executor = MagicMock()
        config = make_supervisor_config()
        artifact_manager = ArtifactManager(
            executor=executor,
            kernel_name="gemm",
            run_id="test_run_001",
            local_base_dir=str(tmp_path),
        )
        artifact_manager.setup_local()
        notifier = Notifier(auto_approve=True)
        supervisor = KernelSupervisor(
            executor=executor,
            config=config,
            artifact_manager=artifact_manager,
            notifier=notifier,
            progress_callback=callback,
        )
        assert supervisor.progress_callback is callback

    def test_initial_state_is_supervisor_state(self, tmp_path: Path) -> None:
        supervisor = make_supervisor(tmp_path)
        assert isinstance(supervisor.state, SupervisorState)

    def test_initial_state_phase_is_setup(self, tmp_path: Path) -> None:
        supervisor = make_supervisor(tmp_path)
        assert supervisor.state.current_phase == Phase.SETUP


# ---------------------------------------------------------------------------
# _should_skip_phase
# ---------------------------------------------------------------------------


class TestShouldSkipPhase:
    def test_returns_false_when_no_marker(self, tmp_path: Path) -> None:
        supervisor = make_supervisor(tmp_path)
        assert supervisor._should_skip_phase(Phase.SETUP) is False

    def test_returns_false_for_every_phase_initially(self, tmp_path: Path) -> None:
        supervisor = make_supervisor(tmp_path)
        for phase in Phase:
            assert supervisor._should_skip_phase(phase) is False

    def test_returns_true_after_marker_written(self, tmp_path: Path) -> None:
        supervisor = make_supervisor(tmp_path)
        supervisor.artifact_manager.mark_phase_complete(Phase.SETUP.value, {})
        assert supervisor._should_skip_phase(Phase.SETUP) is True

    def test_only_marked_phase_returns_true(self, tmp_path: Path) -> None:
        supervisor = make_supervisor(tmp_path)
        supervisor.artifact_manager.mark_phase_complete(Phase.BASELINE.value, {})
        assert supervisor._should_skip_phase(Phase.BASELINE) is True
        assert supervisor._should_skip_phase(Phase.DISCOVERY) is False
        assert supervisor._should_skip_phase(Phase.TUNING) is False

    def test_delegates_to_artifact_manager(self, tmp_path: Path) -> None:
        supervisor = make_supervisor(tmp_path)
        mock_am = MagicMock()
        mock_am.is_phase_complete.return_value = True
        supervisor.artifact_manager = mock_am
        result = supervisor._should_skip_phase(Phase.COMMIT)
        mock_am.is_phase_complete.assert_called_once_with(Phase.COMMIT.value)
        assert result is True


# ---------------------------------------------------------------------------
# _record_phase_complete
# ---------------------------------------------------------------------------


class TestRecordPhaseComplete:
    def _make_result(self, phase: Phase, success: bool = True) -> PhaseResult:
        return PhaseResult(
            phase=phase,
            success=success,
            data={"test": True},
            error=None,
            duration_seconds=5.0,
        )

    def test_adds_phase_to_completed_phases(self, tmp_path: Path) -> None:
        supervisor = make_supervisor(tmp_path)
        result = self._make_result(Phase.SETUP)
        supervisor._record_phase_complete(Phase.SETUP, result)
        assert Phase.SETUP in supervisor.state.completed_phases

    def test_stores_result_in_phase_results(self, tmp_path: Path) -> None:
        supervisor = make_supervisor(tmp_path)
        result = self._make_result(Phase.DISCOVERY)
        supervisor._record_phase_complete(Phase.DISCOVERY, result)
        assert supervisor.state.phase_results[Phase.DISCOVERY] is result

    def test_writes_artifact_marker(self, tmp_path: Path) -> None:
        supervisor = make_supervisor(tmp_path)
        result = self._make_result(Phase.BASELINE)
        supervisor._record_phase_complete(Phase.BASELINE, result)
        assert supervisor.artifact_manager.is_phase_complete(Phase.BASELINE.value)

    def test_is_phase_complete_after_record(self, tmp_path: Path) -> None:
        supervisor = make_supervisor(tmp_path)
        result = self._make_result(Phase.TUNING)
        supervisor._record_phase_complete(Phase.TUNING, result)
        assert supervisor._should_skip_phase(Phase.TUNING) is True

    def test_does_not_duplicate_completed_phases(self, tmp_path: Path) -> None:
        supervisor = make_supervisor(tmp_path)
        result = self._make_result(Phase.SETUP)
        supervisor._record_phase_complete(Phase.SETUP, result)
        supervisor._record_phase_complete(Phase.SETUP, result)
        assert supervisor.state.completed_phases.count(Phase.SETUP) == 1

    def test_records_multiple_phases_independently(self, tmp_path: Path) -> None:
        supervisor = make_supervisor(tmp_path)
        for phase in [Phase.SETUP, Phase.DISCOVERY, Phase.BASELINE]:
            result = self._make_result(phase)
            supervisor._record_phase_complete(phase, result)
        assert len(supervisor.state.completed_phases) == 3
        for phase in [Phase.SETUP, Phase.DISCOVERY, Phase.BASELINE]:
            assert phase in supervisor.state.completed_phases

    def test_error_included_in_marker_summary(self, tmp_path: Path) -> None:
        supervisor = make_supervisor(tmp_path)
        result = PhaseResult(
            phase=Phase.TUNING,
            success=False,
            data={},
            error="timeout error",
            duration_seconds=1800.0,
        )
        supervisor._record_phase_complete(Phase.TUNING, result)
        summary = supervisor.artifact_manager.get_phase_summary(Phase.TUNING.value)
        assert summary is not None
        assert summary["error"] == "timeout error"

    def test_success_flag_in_marker_summary(self, tmp_path: Path) -> None:
        supervisor = make_supervisor(tmp_path)
        result = self._make_result(Phase.COMMIT, success=True)
        supervisor._record_phase_complete(Phase.COMMIT, result)
        summary = supervisor.artifact_manager.get_phase_summary(Phase.COMMIT.value)
        assert summary["success"] is True


# ---------------------------------------------------------------------------
# _get_resume_phase
# ---------------------------------------------------------------------------


class TestGetResumePhase:
    def test_returns_setup_when_nothing_complete(self, tmp_path: Path) -> None:
        supervisor = make_supervisor(tmp_path)
        assert supervisor._get_resume_phase() == Phase.SETUP

    def test_skips_completed_setup(self, tmp_path: Path) -> None:
        supervisor = make_supervisor(tmp_path)
        supervisor.artifact_manager.mark_phase_complete(Phase.SETUP.value, {})
        assert supervisor._get_resume_phase() == Phase.DISCOVERY

    def test_skips_multiple_completed_phases(self, tmp_path: Path) -> None:
        supervisor = make_supervisor(tmp_path)
        for phase in [Phase.SETUP, Phase.DISCOVERY, Phase.BASELINE]:
            supervisor.artifact_manager.mark_phase_complete(phase.value, {})
        assert supervisor._get_resume_phase() == Phase.UNTUNED_VALIDATION

    def test_returns_commit_when_all_complete(self, tmp_path: Path) -> None:
        supervisor = make_supervisor(tmp_path)
        for phase in Phase:
            supervisor.artifact_manager.mark_phase_complete(phase.value, {})
        assert supervisor._get_resume_phase() == Phase.COMMIT

    def test_gap_in_completed_phases_returns_first_incomplete(
        self, tmp_path: Path
    ) -> None:
        """If phases 0 and 2 are complete but 1 is not, return phase 1."""
        supervisor = make_supervisor(tmp_path)
        supervisor.artifact_manager.mark_phase_complete(Phase.SETUP.value, {})
        # Skip DISCOVERY (phase 1), mark BASELINE (phase 2)
        supervisor.artifact_manager.mark_phase_complete(Phase.BASELINE.value, {})
        # Should return DISCOVERY (first incomplete in order)
        assert supervisor._get_resume_phase() == Phase.DISCOVERY

    def test_returns_tuning_when_three_phases_complete(self, tmp_path: Path) -> None:
        supervisor = make_supervisor(tmp_path)
        for phase in [
            Phase.SETUP,
            Phase.DISCOVERY,
            Phase.BASELINE,
            Phase.UNTUNED_VALIDATION,
        ]:
            supervisor.artifact_manager.mark_phase_complete(phase.value, {})
        assert supervisor._get_resume_phase() == Phase.TUNING


# ---------------------------------------------------------------------------
# SupervisorResult dataclass
# ---------------------------------------------------------------------------


class TestSupervisorResult:
    def test_creation(self) -> None:
        result = SupervisorResult(
            kernel_name="gemm",
            success=True,
            phase_results={},
            escalations=[],
            geomean_speedup=1.15,
            summary="Tuning complete: 15% speedup",
        )
        assert result.kernel_name == "gemm"
        assert result.success is True
        assert result.geomean_speedup == pytest.approx(1.15)
        assert result.summary == "Tuning complete: 15% speedup"

    def test_geomean_speedup_can_be_none(self) -> None:
        result = SupervisorResult(
            kernel_name="flash_attn",
            success=False,
            phase_results={},
            escalations=[],
            geomean_speedup=None,
            summary="Failed during baseline phase",
        )
        assert result.geomean_speedup is None

    def test_phase_results_stored(self) -> None:
        pr = PhaseResult(
            phase=Phase.BASELINE,
            success=True,
            data={},
            error=None,
            duration_seconds=10.0,
        )
        result = SupervisorResult(
            kernel_name="gemm",
            success=True,
            phase_results={Phase.BASELINE: pr},
            escalations=[],
            geomean_speedup=None,
            summary="done",
        )
        assert Phase.BASELINE in result.phase_results


# ---------------------------------------------------------------------------
# _dispatch_subagent
# ---------------------------------------------------------------------------


class TestDispatchSubagent:
    def test_dispatch_subagent_success(self, tmp_path: Path) -> None:
        """A subagent whose run() returns success is returned as-is."""
        supervisor = make_supervisor(tmp_path)
        expected = SubagentResult(success=True, data={"key": "val"})

        FakeSubagent = MagicMock()
        FakeSubagent.return_value.run.return_value = expected

        result = supervisor._dispatch_subagent(FakeSubagent)

        assert result.success is True
        assert result.data == {"key": "val"}

    def test_dispatch_subagent_passes_kwargs(self, tmp_path: Path) -> None:
        """Extra kwargs are forwarded to the subagent constructor."""
        supervisor = make_supervisor(tmp_path)
        expected = SubagentResult(success=True)

        FakeSubagent = MagicMock()
        FakeSubagent.return_value.run.return_value = expected

        supervisor._dispatch_subagent(FakeSubagent, extra_flag=True, count=3)

        _, kwargs = FakeSubagent.call_args
        assert kwargs["extra_flag"] is True
        assert kwargs["count"] == 3


# ---------------------------------------------------------------------------
# _dispatch_with_retry
# ---------------------------------------------------------------------------


class TestDispatchWithRetry:
    def test_dispatch_with_retry_succeeds_first_try(self, tmp_path: Path) -> None:
        """No retry is triggered when the first attempt succeeds."""
        supervisor = make_supervisor(tmp_path)
        success_result = SubagentResult(success=True, data={"done": True})

        FakeSubagent = MagicMock()
        FakeSubagent.return_value.run.return_value = success_result

        result = supervisor._dispatch_with_retry(FakeSubagent, max_retries=2)

        assert result.success is True
        # Should have been instantiated (and run) exactly once
        assert FakeSubagent.call_count == 1

    def test_dispatch_with_retry_succeeds_on_second(self, tmp_path: Path) -> None:
        """After one failure the second attempt succeeds and that result is returned."""
        supervisor = make_supervisor(tmp_path)
        fail_result = SubagentResult(success=False, error="transient error")
        ok_result = SubagentResult(success=True, data={"recovered": True})

        instance1 = MagicMock()
        instance1.run.return_value = fail_result
        instance2 = MagicMock()
        instance2.run.return_value = ok_result

        FakeSubagent = MagicMock(side_effect=[instance1, instance2])

        result = supervisor._dispatch_with_retry(FakeSubagent, max_retries=2)

        assert result.success is True
        assert result.data == {"recovered": True}
        assert FakeSubagent.call_count == 2

    def test_dispatch_with_retry_exhausts_retries(self, tmp_path: Path) -> None:
        """When all attempts fail the last failure result is returned."""
        supervisor = make_supervisor(tmp_path)
        fail1 = SubagentResult(success=False, error="fail 1")
        fail2 = SubagentResult(success=False, error="fail 2")
        fail3 = SubagentResult(success=False, error="fail 3")

        instances = []
        for r in [fail1, fail2, fail3]:
            inst = MagicMock()
            inst.run.return_value = r
            instances.append(inst)

        FakeSubagent = MagicMock(side_effect=instances)

        result = supervisor._dispatch_with_retry(FakeSubagent, max_retries=2)

        assert result.success is False
        assert result.error == "fail 3"
        # 1 initial attempt + 2 retries = 3 total
        assert FakeSubagent.call_count == 3


# ---------------------------------------------------------------------------
# _switch_triton
# ---------------------------------------------------------------------------


class TestSwitchTriton:
    def _make_repo_config(
        self,
        triton_repo: str = "/workspace/triton",
        triton_branch: str = "target-branch",
        install_cmd: str = "pip install -e .",
    ) -> RepoConfig:
        return RepoConfig(
            aiter_repo="https://github.com/example/aiter",
            aiter_branch="main",
            triton_repo=triton_repo,
            triton_branch=triton_branch,
        )

    def test_switch_triton_runs_commands(self, tmp_path: Path) -> None:
        """_switch_triton calls docker_exec with git checkout and install."""
        supervisor = make_supervisor(tmp_path)
        mock_exec = MagicMock()
        mock_exec.verify_environment.return_value = {
            "triton_version": "3.0.0",
            "aiter_branch": "main",
        }
        supervisor.executor = mock_exec

        repo_config = self._make_repo_config(
            triton_repo="/workspace/triton",
            triton_branch="my-feature-branch",
        )
        # Use the default install command from TritonInstallConfig
        install_cmd = supervisor.config.triton_install.command

        supervisor._switch_triton(repo_config)

        # Verify docker_exec was called at least once
        assert mock_exec.docker_exec.called

        # Collect all commands passed to docker_exec
        all_commands = [c[0][0] for c in mock_exec.docker_exec.call_args_list]
        combined = " ".join(all_commands)

        assert "git checkout" in combined
        assert "my-feature-branch" in combined
        assert install_cmd in combined

    def test_switch_triton_raises_on_verify_failure(self, tmp_path: Path) -> None:
        """If verify_environment raises, _switch_triton propagates the error."""
        from .remote import RemoteCommandError

        supervisor = make_supervisor(tmp_path)
        mock_exec = MagicMock()
        mock_exec.verify_environment.side_effect = RemoteCommandError(
            "Triton version mismatch"
        )
        supervisor.executor = mock_exec

        repo_config = self._make_repo_config()

        with pytest.raises(RemoteCommandError):
            supervisor._switch_triton(repo_config)


# ---------------------------------------------------------------------------
# _check_phase_timeout
# ---------------------------------------------------------------------------


class TestCheckPhaseTimeout:
    def test_check_phase_timeout_not_exceeded(self, tmp_path: Path) -> None:
        """Returns False when elapsed time is less than phase_max."""
        supervisor = make_supervisor(tmp_path)
        # Start time is "now" — effectively 0 seconds have elapsed
        start_time = time.time()
        result = supervisor._check_phase_timeout(Phase.TUNING, start_time)
        assert result is False

    def test_check_phase_timeout_exceeded(self, tmp_path: Path) -> None:
        """Returns True when the start time is in the distant past."""
        supervisor = make_supervisor(tmp_path)
        phase_max = supervisor.config.tuning_config.timeouts.phase_max  # default 14400
        # Simulate start time far in the past
        ancient_start = time.time() - (phase_max + 1)
        result = supervisor._check_phase_timeout(Phase.TUNING, ancient_start)
        assert result is True


# ---------------------------------------------------------------------------
# Helper: make ShapeResult fixtures
# ---------------------------------------------------------------------------


def make_shape_result(m: int, n: int, k: int, total_ns: float) -> ShapeResult:
    """Return a ShapeResult with main_ns=total_ns and reduce_ns=0."""
    return ShapeResult(m=m, n=n, k=k, main_ns=total_ns, reduce_ns=0.0)


# ---------------------------------------------------------------------------
# _identify_regressed_shapes
# ---------------------------------------------------------------------------


class TestIdentifyRegressedShapes:
    def test_finds_regressions(self, tmp_path: Path) -> None:
        """baseline=100 ns, untuned=110 ns with 5% threshold => regressed."""
        supervisor = make_supervisor(tmp_path)
        supervisor.state.baseline_data = [make_shape_result(128, 256, 256, 100.0)]
        supervisor.state.untuned_data = [make_shape_result(128, 256, 256, 110.0)]
        # Default threshold is 5 %; 110 > 100 * 1.05 = 105 => regression
        regressed = supervisor._identify_regressed_shapes()
        assert (128, 256, 256) in regressed

    def test_no_regressions_within_threshold(self, tmp_path: Path) -> None:
        """baseline=100 ns, untuned=104 ns with 5% threshold => not regressed."""
        supervisor = make_supervisor(tmp_path)
        supervisor.state.baseline_data = [make_shape_result(128, 256, 256, 100.0)]
        supervisor.state.untuned_data = [make_shape_result(128, 256, 256, 104.0)]
        # 104 <= 100 * 1.05 = 105 => no regression
        regressed = supervisor._identify_regressed_shapes()
        assert regressed == []

    def test_returns_empty_when_no_baseline(self, tmp_path: Path) -> None:
        """When baseline_data is None, returns an empty list."""
        supervisor = make_supervisor(tmp_path)
        supervisor.state.baseline_data = None
        supervisor.state.untuned_data = [make_shape_result(128, 256, 256, 110.0)]
        assert supervisor._identify_regressed_shapes() == []

    def test_returns_empty_when_no_untuned(self, tmp_path: Path) -> None:
        """When untuned_data is None, returns an empty list."""
        supervisor = make_supervisor(tmp_path)
        supervisor.state.baseline_data = [make_shape_result(128, 256, 256, 100.0)]
        supervisor.state.untuned_data = None
        assert supervisor._identify_regressed_shapes() == []

    def test_skips_shapes_not_in_baseline(self, tmp_path: Path) -> None:
        """Shapes present only in untuned_data are silently ignored."""
        supervisor = make_supervisor(tmp_path)
        supervisor.state.baseline_data = [make_shape_result(64, 64, 64, 100.0)]
        supervisor.state.untuned_data = [
            make_shape_result(64, 64, 64, 104.0),   # not regressed
            make_shape_result(128, 128, 128, 200.0), # not in baseline => skip
        ]
        regressed = supervisor._identify_regressed_shapes()
        assert regressed == []

    def test_multiple_shapes_partial_regression(self, tmp_path: Path) -> None:
        """Only the shapes that exceed the threshold appear in the output."""
        supervisor = make_supervisor(tmp_path)
        supervisor.state.baseline_data = [
            make_shape_result(32, 64, 64, 100.0),
            make_shape_result(64, 64, 64, 100.0),
        ]
        supervisor.state.untuned_data = [
            make_shape_result(32, 64, 64, 110.0),  # 10 % regression => included
            make_shape_result(64, 64, 64, 103.0),  # 3 % regression => excluded
        ]
        regressed = supervisor._identify_regressed_shapes()
        assert (32, 64, 64) in regressed
        assert (64, 64, 64) not in regressed


# ---------------------------------------------------------------------------
# _determine_shapes_to_tune
# ---------------------------------------------------------------------------


class TestDetermineShapesToTune:
    def _set_state_shapes(self, supervisor: KernelSupervisor) -> None:
        """Populate state.shapes with two ShapeResult objects."""
        supervisor.state.shapes = [
            make_shape_result(32, 64, 64, 0.0),
            make_shape_result(64, 128, 128, 0.0),
        ]

    def test_full_mode_returns_all_shapes(self, tmp_path: Path) -> None:
        """In FULL mode every shape from state.shapes is returned."""
        supervisor = make_supervisor(tmp_path)
        self._set_state_shapes(supervisor)
        supervisor.config.tuning_config.mode = TuningMode.FULL

        shapes = supervisor._determine_shapes_to_tune()

        assert (32, 64, 64) in shapes
        assert (64, 128, 128) in shapes
        assert len(shapes) == 2

    def test_regression_only_with_regressions(self, tmp_path: Path) -> None:
        """In REGRESSION_ONLY mode only regressed shapes are returned."""
        supervisor = make_supervisor(tmp_path)
        self._set_state_shapes(supervisor)
        supervisor.config.tuning_config.mode = TuningMode.REGRESSION_ONLY

        # Inject baseline/untuned data so only (32, 64, 64) regresses.
        supervisor.state.baseline_data = [
            make_shape_result(32, 64, 64, 100.0),
            make_shape_result(64, 128, 128, 100.0),
        ]
        supervisor.state.untuned_data = [
            make_shape_result(32, 64, 64, 115.0),   # 15 % => regression
            make_shape_result(64, 128, 128, 102.0),  # 2 % => no regression
        ]

        shapes = supervisor._determine_shapes_to_tune()

        assert (32, 64, 64) in shapes
        assert (64, 128, 128) not in shapes

    def test_regression_only_no_regressions_returns_empty(self, tmp_path: Path) -> None:
        """In REGRESSION_ONLY mode with no regressions, returns empty list."""
        supervisor = make_supervisor(tmp_path)
        self._set_state_shapes(supervisor)
        supervisor.config.tuning_config.mode = TuningMode.REGRESSION_ONLY

        supervisor.state.baseline_data = [make_shape_result(32, 64, 64, 100.0)]
        supervisor.state.untuned_data = [make_shape_result(32, 64, 64, 101.0)]

        shapes = supervisor._determine_shapes_to_tune()

        assert shapes == []


# ---------------------------------------------------------------------------
# _run_phase_4_tuning
# ---------------------------------------------------------------------------


class TestRunPhase4Tuning:
    def test_skips_when_no_shapes_to_tune(self, tmp_path: Path) -> None:
        """When _determine_shapes_to_tune() returns empty, result has skipped=True."""
        supervisor = make_supervisor(tmp_path)

        with patch.object(
            supervisor, "_determine_shapes_to_tune", return_value=[]
        ):
            result = supervisor._run_phase_4_tuning()

        assert result.success is True
        assert result.data.get("skipped") is True
        assert result.phase == Phase.TUNING

    def test_dispatches_tuning_pipeline_in_order(self, tmp_path: Path) -> None:
        """All four subagents are dispatched in order when shapes exist."""
        supervisor = make_supervisor(tmp_path)

        shapes = [(32, 64, 64), (64, 128, 128)]

        # Track dispatch call order.
        dispatch_order: list = []

        def fake_dispatch(cls, **kwargs):
            dispatch_order.append(cls)
            return SubagentResult(success=True, data={})

        with patch.object(
            supervisor, "_determine_shapes_to_tune", return_value=shapes
        ), patch.object(supervisor, "_dispatch_subagent", side_effect=fake_dispatch):
            result = supervisor._run_phase_4_tuning()

        assert result.success is True
        assert result.phase == Phase.TUNING

        # Verify all four subagents were dispatched.
        assert TuningAgent in dispatch_order
        assert PatternAnalyzerAgent in dispatch_order
        assert ConfigGeneratorAgent in dispatch_order

        # TuningAgent is dispatched twice (scout + full tuning).
        assert dispatch_order.count(TuningAgent) == 2

        # Order: TuningAgent (scout), PatternAnalyzerAgent, TuningAgent (full), ConfigGeneratorAgent
        assert dispatch_order[0] is TuningAgent
        assert dispatch_order[1] is PatternAnalyzerAgent
        assert dispatch_order[2] is TuningAgent
        assert dispatch_order[3] is ConfigGeneratorAgent

    def test_result_contains_shapes_tuned_count(self, tmp_path: Path) -> None:
        """result.data['shapes_tuned'] reflects the number of shapes."""
        supervisor = make_supervisor(tmp_path)

        shapes = [(32, 64, 64), (64, 128, 128), (128, 256, 256)]

        with patch.object(
            supervisor, "_determine_shapes_to_tune", return_value=shapes
        ), patch.object(
            supervisor, "_dispatch_subagent", return_value=SubagentResult(success=True, data={})
        ):
            result = supervisor._run_phase_4_tuning()

        assert result.data["shapes_tuned"] == 3

    def test_result_contains_configs_generated_flag(self, tmp_path: Path) -> None:
        """result.data['configs_generated'] is True after successful run."""
        supervisor = make_supervisor(tmp_path)

        shapes = [(32, 64, 64)]

        with patch.object(
            supervisor, "_determine_shapes_to_tune", return_value=shapes
        ), patch.object(
            supervisor, "_dispatch_subagent", return_value=SubagentResult(success=True, data={})
        ):
            result = supervisor._run_phase_4_tuning()

        assert result.data.get("configs_generated") is True


# ---------------------------------------------------------------------------
# _run_phase_0_setup
# ---------------------------------------------------------------------------


class TestRunPhase0Setup:
    def test_phase_0_dispatches_setup_agent(self, tmp_path: Path) -> None:
        """_run_phase_0_setup calls _dispatch_with_retry with SetupAgent."""
        from .subagents.setup_agent import SetupAgent

        supervisor = make_supervisor(tmp_path)
        success_result = SubagentResult(
            success=True, data={"container_id": "abc123"}
        )
        with patch.object(
            KernelSupervisor, "_dispatch_with_retry", return_value=success_result
        ) as mock_dispatch:
            phase_result = supervisor._run_phase_0_setup()

        mock_dispatch.assert_called_once()
        first_arg = mock_dispatch.call_args[0][0]
        assert first_arg is SetupAgent

    def test_phase_0_returns_phase_result(self, tmp_path: Path) -> None:
        """_run_phase_0_setup returns a PhaseResult for Phase.SETUP."""
        supervisor = make_supervisor(tmp_path)
        success_result = SubagentResult(success=True, data={})
        with patch.object(
            KernelSupervisor, "_dispatch_with_retry", return_value=success_result
        ):
            phase_result = supervisor._run_phase_0_setup()

        assert isinstance(phase_result, PhaseResult)
        assert phase_result.phase == Phase.SETUP

    def test_phase_0_stores_container_id_on_success(self, tmp_path: Path) -> None:
        """On success, container_id from result data is stored on supervisor."""
        supervisor = make_supervisor(tmp_path)
        success_result = SubagentResult(
            success=True, data={"container_id": "ctr_999"}
        )
        with patch.object(
            KernelSupervisor, "_dispatch_with_retry", return_value=success_result
        ):
            supervisor._run_phase_0_setup()

        assert supervisor.container_id == "ctr_999"

    def test_phase_0_returns_failure_on_exception(self, tmp_path: Path) -> None:
        """If _dispatch_with_retry raises, _run_phase_0_setup returns failure."""
        supervisor = make_supervisor(tmp_path)
        with patch.object(
            KernelSupervisor,
            "_dispatch_with_retry",
            side_effect=RuntimeError("boom"),
        ):
            phase_result = supervisor._run_phase_0_setup()

        assert phase_result.success is False
        assert "boom" in phase_result.error

    def test_phase_0_records_duration(self, tmp_path: Path) -> None:
        """duration_seconds is non-negative after _run_phase_0_setup completes."""
        supervisor = make_supervisor(tmp_path)
        success_result = SubagentResult(success=True, data={})
        with patch.object(
            KernelSupervisor, "_dispatch_with_retry", return_value=success_result
        ):
            phase_result = supervisor._run_phase_0_setup()

        assert phase_result.duration_seconds >= 0.0


# ---------------------------------------------------------------------------
# _run_phase_2_baseline
# ---------------------------------------------------------------------------


class TestRunPhase2Baseline:
    def test_phase_2_switches_triton_twice(self, tmp_path: Path) -> None:
        """_run_phase_2_baseline calls _switch_triton for baseline then target."""
        supervisor = make_supervisor(tmp_path)
        success_result = SubagentResult(success=True, data={})
        switch_calls: list = []

        def record_switch(repo_config):
            switch_calls.append(repo_config)

        with patch.object(
            KernelSupervisor, "_dispatch_with_retry", return_value=success_result
        ):
            with patch.object(
                KernelSupervisor, "_switch_triton", side_effect=record_switch
            ):
                supervisor._run_phase_2_baseline()

        assert len(switch_calls) == 2
        assert switch_calls[0] is supervisor.config.baseline_repo
        assert switch_calls[1] is supervisor.config.target_repo

    def test_phase_2_stores_baseline_data(self, tmp_path: Path) -> None:
        """On success, state.baseline_data is populated from result data."""
        supervisor = make_supervisor(tmp_path)
        fake_results = [{"m": 256, "n": 256, "k": 64}]
        success_result = SubagentResult(
            success=True, data={"results": fake_results}
        )
        with patch.object(
            KernelSupervisor, "_dispatch_with_retry", return_value=success_result
        ):
            with patch.object(KernelSupervisor, "_switch_triton"):
                supervisor._run_phase_2_baseline()

        assert supervisor.state.baseline_data == fake_results

    def test_phase_2_dispatches_baseline_agent(self, tmp_path: Path) -> None:
        """_run_phase_2_baseline dispatches BaselineAgent."""
        from .subagents.baseline_agent import BaselineAgent

        supervisor = make_supervisor(tmp_path)
        success_result = SubagentResult(success=True, data={})

        with patch.object(
            KernelSupervisor, "_dispatch_with_retry", return_value=success_result
        ) as mock_dispatch:
            with patch.object(KernelSupervisor, "_switch_triton"):
                supervisor._run_phase_2_baseline()

        mock_dispatch.assert_called_once()
        first_arg = mock_dispatch.call_args[0][0]
        assert first_arg is BaselineAgent

    def test_phase_2_returns_failure_on_exception(self, tmp_path: Path) -> None:
        """If _switch_triton raises, _run_phase_2_baseline returns failure."""
        supervisor = make_supervisor(tmp_path)
        with patch.object(
            KernelSupervisor,
            "_switch_triton",
            side_effect=RuntimeError("switch failed"),
        ):
            phase_result = supervisor._run_phase_2_baseline()

        assert phase_result.success is False
        assert "switch failed" in phase_result.error

    def test_phase_2_returns_phase_result(self, tmp_path: Path) -> None:
        """_run_phase_2_baseline returns a PhaseResult for Phase.BASELINE."""
        supervisor = make_supervisor(tmp_path)
        success_result = SubagentResult(success=True, data={})
        with patch.object(
            KernelSupervisor, "_dispatch_with_retry", return_value=success_result
        ):
            with patch.object(KernelSupervisor, "_switch_triton"):
                phase_result = supervisor._run_phase_2_baseline()

        assert isinstance(phase_result, PhaseResult)
        assert phase_result.phase == Phase.BASELINE


# ---------------------------------------------------------------------------
# _run_phase_3_untuned
# ---------------------------------------------------------------------------


class TestRunPhase3Untuned:
    def test_phase_3_dispatches_validation(self, tmp_path: Path) -> None:
        """_run_phase_3_untuned dispatches ValidationAgent."""
        from .subagents.validation_agent import ValidationAgent

        supervisor = make_supervisor(tmp_path)
        success_result = SubagentResult(success=True, data={})

        with patch.object(
            KernelSupervisor, "_dispatch_with_retry", return_value=success_result
        ) as mock_dispatch:
            supervisor._run_phase_3_untuned()

        mock_dispatch.assert_called_once()
        first_arg = mock_dispatch.call_args[0][0]
        assert first_arg is ValidationAgent

    def test_phase_3_stores_untuned_data(self, tmp_path: Path) -> None:
        """On success, state.untuned_data is populated from result data."""
        supervisor = make_supervisor(tmp_path)
        fake_results = [{"m": 128, "n": 128, "k": 32}]
        success_result = SubagentResult(
            success=True, data={"results": fake_results}
        )
        with patch.object(
            KernelSupervisor, "_dispatch_with_retry", return_value=success_result
        ):
            supervisor._run_phase_3_untuned()

        assert supervisor.state.untuned_data == fake_results

    def test_phase_3_returns_phase_result(self, tmp_path: Path) -> None:
        """_run_phase_3_untuned returns a PhaseResult for Phase.UNTUNED_VALIDATION."""
        supervisor = make_supervisor(tmp_path)
        success_result = SubagentResult(success=True, data={})
        with patch.object(
            KernelSupervisor, "_dispatch_with_retry", return_value=success_result
        ):
            phase_result = supervisor._run_phase_3_untuned()

        assert isinstance(phase_result, PhaseResult)
        assert phase_result.phase == Phase.UNTUNED_VALIDATION

    def test_phase_3_passes_gpu_ids(self, tmp_path: Path) -> None:
        """gpu_ids from config are forwarded to ValidationAgent."""
        supervisor = make_supervisor(tmp_path)
        success_result = SubagentResult(success=True, data={})

        with patch.object(
            KernelSupervisor, "_dispatch_with_retry", return_value=success_result
        ) as mock_dispatch:
            supervisor._run_phase_3_untuned()

        kwargs = mock_dispatch.call_args[1]
        assert kwargs["gpu_ids"] == supervisor.config.gpu_ids

    def test_phase_3_returns_failure_on_exception(self, tmp_path: Path) -> None:
        """If _dispatch_with_retry raises, _run_phase_3_untuned returns failure."""
        supervisor = make_supervisor(tmp_path)
        with patch.object(
            KernelSupervisor,
            "_dispatch_with_retry",
            side_effect=RuntimeError("validation error"),
        ):
            phase_result = supervisor._run_phase_3_untuned()

        assert phase_result.success is False
        assert "validation error" in phase_result.error

    def test_phase_3_uses_shapes_from_state(self, tmp_path: Path) -> None:
        """Shapes stored in state are forwarded to ValidationAgent."""
        supervisor = make_supervisor(tmp_path)
        supervisor.state.shapes = [(64, 64, 64), (128, 128, 128)]
        success_result = SubagentResult(success=True, data={})

        with patch.object(
            KernelSupervisor, "_dispatch_with_retry", return_value=success_result
        ) as mock_dispatch:
            supervisor._run_phase_3_untuned()

        kwargs = mock_dispatch.call_args[1]
        assert kwargs["shapes"] == [(64, 64, 64), (128, 128, 128)]
