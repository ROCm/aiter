"""Tests for the Orchestrator class.

Follows TDD: each test verifies an important behavioural contract of the
top-level coordination loop without touching real SSH, GPU hardware, or
the filesystem.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from unittest.mock import MagicMock, call, patch

import pytest

from .kernel_discovery import DiscoveredKernel, GemmCategory
from .kernel_supervisor import (
    EscalationRequest,
    Phase,
    PhaseResult,
    SupervisorResult,
)
from .notifications import Notifier
from .orchestrator import Orchestrator
from .types import (
    ContainerConfig,
    GpuConfig,
    KernelOverrides,
    KernelsConfig,
    MachineInfo,
    PipelineConfig,
    RepoConfig,
    TritonInstallConfig,
    TuningConfig,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_repo_config() -> RepoConfig:
    return RepoConfig(
        aiter_repo="https://github.com/example/aiter",
        aiter_branch="main",
        triton_repo="https://github.com/example/triton",
        triton_branch="main",
    )


def _make_machine(host: str = "gpu1", gpus: Optional[List[int]] = None) -> MachineInfo:
    return MachineInfo(
        host=host,
        user="ubuntu",
        ssh_key="/home/ubuntu/.ssh/id_rsa",
        gpus=gpus if gpus is not None else [0, 1],
    )


def _make_config(
    machines: Optional[List[MachineInfo]] = None,
    kernels: Optional[KernelsConfig] = None,
) -> PipelineConfig:
    return PipelineConfig(
        baseline=_make_repo_config(),
        target=_make_repo_config(),
        machines=machines or [_make_machine()],
        container=ContainerConfig(image="rocm/pytorch:latest"),
        gpu=GpuConfig(arch="gfx950"),
        triton_install=TritonInstallConfig(),
        tuning=TuningConfig(),
        kernels=kernels or KernelsConfig(),
    )


def _make_discovered_kernel(name: str = "a8w8") -> DiscoveredKernel:
    return DiscoveredKernel(
        name=name,
        category=GemmCategory.BASIC,
        source_path=f"/repo/aiter/ops/triton/gemm/basic/gemm_{name}.py",
        config_variant=name.upper(),
    )


def _make_supervisor_result(
    kernel_name: str = "a8w8",
    success: bool = True,
    geomean_speedup: Optional[float] = 1.05,
    shapes_tuned: int = 10,
    regressions: int = 0,
) -> SupervisorResult:
    phase_results: Dict[Phase, PhaseResult] = {
        Phase.TUNING: PhaseResult(
            phase=Phase.TUNING,
            success=True,
            data={"shapes_tuned": shapes_tuned, "configs_generated": True},
            error=None,
            duration_seconds=30.0,
        ),
        Phase.COMMIT: PhaseResult(
            phase=Phase.COMMIT,
            success=success,
            data={"committed": success, "geomean_speedup": geomean_speedup},
            error=None if success else "Commit denied.",
            duration_seconds=5.0,
        ),
    }
    escalations: List[EscalationRequest] = []
    if regressions > 0:
        escalations.append(
            EscalationRequest(
                severity="warning",
                message=f"Kernel '{kernel_name}': {regressions} regression(s) persist.",
            )
        )
    return SupervisorResult(
        kernel_name=kernel_name,
        success=success,
        phase_results=phase_results,
        escalations=escalations,
        geomean_speedup=geomean_speedup,
        summary=f"TUNING=OK, COMMIT={'OK' if success else 'FAIL'}",
    )


# ---------------------------------------------------------------------------
# test_discover_kernels_uses_config_filters
# ---------------------------------------------------------------------------


class TestDiscoverKernels:
    """discover_kernels() must pass include/exclude from config to KernelDiscovery."""

    def test_discover_kernels_uses_config_filters(self):
        """KernelDiscovery.discover_all receives the configured exclude/include lists."""
        kernels_cfg = KernelsConfig(
            exclude=["slow_kernel", "broken_kernel"],
            include=["a8w8", "a16w16"],
        )
        config = _make_config(kernels=kernels_cfg)

        with patch(
            "tuning_agent.orchestrator.KernelDiscovery"
        ) as MockKD:
            mock_kd_instance = MagicMock()
            mock_kd_instance.discover_all.return_value = [
                _make_discovered_kernel("a8w8"),
                _make_discovered_kernel("a16w16"),
            ]
            MockKD.return_value = mock_kd_instance

            orch = Orchestrator(config=config, repo_root="/repo", run_id="test-run")
            kernels = orch.discover_kernels()

        mock_kd_instance.discover_all.assert_called_once_with(
            exclude=["slow_kernel", "broken_kernel"],
            include=["a8w8", "a16w16"],
        )
        assert len(kernels) == 2

    def test_discover_kernels_passes_empty_lists(self):
        """When include/exclude are empty lists, None is passed to discover_all."""
        config = _make_config(kernels=KernelsConfig(exclude=[], include=[]))

        with patch(
            "tuning_agent.orchestrator.KernelDiscovery"
        ) as MockKD:
            mock_kd_instance = MagicMock()
            mock_kd_instance.discover_all.return_value = []
            MockKD.return_value = mock_kd_instance

            orch = Orchestrator(config=config, repo_root="/repo", run_id="test-run")
            orch.discover_kernels()

        # Empty lists are coerced to None inside discover_kernels
        mock_kd_instance.discover_all.assert_called_once_with(
            exclude=None,
            include=None,
        )

    def test_discover_kernels_logs_to_dashboard(self):
        """discover_kernels() logs the count to the dashboard when one is provided."""
        config = _make_config()
        dashboard = MagicMock()

        with patch(
            "tuning_agent.orchestrator.KernelDiscovery"
        ) as MockKD:
            mock_kd_instance = MagicMock()
            mock_kd_instance.discover_all.return_value = [
                _make_discovered_kernel("a8w8"),
                _make_discovered_kernel("a16w16"),
                _make_discovered_kernel("fp8"),
            ]
            MockKD.return_value = mock_kd_instance

            orch = Orchestrator(
                config=config, repo_root="/repo", run_id="test-run", dashboard=dashboard
            )
            orch.discover_kernels()

        dashboard.add_log.assert_called_once()
        log_msg = dashboard.add_log.call_args[0][0]
        assert "3" in log_msg


# ---------------------------------------------------------------------------
# test_run_allocates_and_releases_machines
# ---------------------------------------------------------------------------


class TestRunMachineLifecycle:
    """run() must allocate a machine per kernel and release it afterwards."""

    def _setup_mocks(self, kernels, supervisor_result_factory=None):
        """Return a configured (orch, mock_pool, mock_supervisor_cls) triple."""
        config = _make_config(
            machines=[_make_machine("gpu1"), _make_machine("gpu2")]
        )

        if supervisor_result_factory is None:
            supervisor_result_factory = lambda name: _make_supervisor_result(name)

        with patch.multiple(
            "tuning_agent.orchestrator",
            KernelDiscovery=MagicMock(),
            MachinePool=MagicMock(),
            RemoteExecutor=MagicMock(),
            ArtifactManager=MagicMock(),
            KernelSupervisor=MagicMock(),
        ) as mocks:
            # Store references for assertions.
            return mocks, config

    def test_run_allocates_and_releases_machines(self):
        """Each kernel allocates exactly one machine and releases it when done."""
        config = _make_config(machines=[_make_machine("gpu1")])
        kernel_a = _make_discovered_kernel("a8w8")
        kernel_b = _make_discovered_kernel("a16w16")
        machine = _make_machine("gpu1")

        mock_pool = MagicMock()
        mock_pool.allocate.return_value = machine
        mock_pool.validate_connectivity.return_value = [
            {"host": "gpu1", "reachable": True, "rocm_smi_ok": True, "error": None}
        ]

        mock_supervisor = MagicMock()
        mock_supervisor.run.side_effect = [
            _make_supervisor_result("a8w8"),
            _make_supervisor_result("a16w16"),
        ]

        with patch(
            "tuning_agent.orchestrator.KernelDiscovery"
        ) as MockKD, patch(
            "tuning_agent.orchestrator.MachinePool",
            return_value=mock_pool,
        ), patch(
            "tuning_agent.orchestrator.RemoteExecutor"
        ), patch(
            "tuning_agent.orchestrator.ArtifactManager"
        ), patch(
            "tuning_agent.orchestrator.KernelSupervisor",
            return_value=mock_supervisor,
        ):
            MockKD.return_value.discover_all.return_value = [kernel_a, kernel_b]

            orch = Orchestrator(config=config, repo_root="/repo", run_id="test-run")
            orch.run()

        # allocate called once per kernel
        assert mock_pool.allocate.call_count == 2
        mock_pool.allocate.assert_any_call("a8w8")
        mock_pool.allocate.assert_any_call("a16w16")

        # release called once per kernel
        assert mock_pool.release.call_count == 2
        mock_pool.release.assert_called_with("gpu1")

    def test_run_releases_machine_on_supervisor_exception(self):
        """Machine is released even if KernelSupervisor.run() raises an exception."""
        config = _make_config(machines=[_make_machine("gpu1")])
        machine = _make_machine("gpu1")

        mock_pool = MagicMock()
        mock_pool.allocate.return_value = machine
        mock_pool.validate_connectivity.return_value = [
            {"host": "gpu1", "reachable": True, "rocm_smi_ok": True, "error": None}
        ]

        mock_supervisor = MagicMock()
        mock_supervisor.run.side_effect = RuntimeError("Container crashed")

        with patch(
            "tuning_agent.orchestrator.KernelDiscovery"
        ) as MockKD, patch(
            "tuning_agent.orchestrator.MachinePool",
            return_value=mock_pool,
        ), patch(
            "tuning_agent.orchestrator.RemoteExecutor"
        ), patch(
            "tuning_agent.orchestrator.ArtifactManager"
        ), patch(
            "tuning_agent.orchestrator.KernelSupervisor",
            return_value=mock_supervisor,
        ):
            MockKD.return_value.discover_all.return_value = [_make_discovered_kernel("a8w8")]

            orch = Orchestrator(config=config, repo_root="/repo", run_id="test-run")
            # Should not raise — exception is caught internally.
            orch.run()

        # Machine must still be released.
        mock_pool.release.assert_called_once_with("gpu1")


# ---------------------------------------------------------------------------
# test_run_creates_supervisor_per_kernel
# ---------------------------------------------------------------------------


class TestRunSupervisorCreation:
    """run() must create a KernelSupervisor for each discovered kernel."""

    def test_run_creates_supervisor_per_kernel(self):
        """KernelSupervisor is instantiated exactly once per kernel."""
        config = _make_config(machines=[_make_machine("gpu1")])
        kernels = [
            _make_discovered_kernel("a8w8"),
            _make_discovered_kernel("a16w16"),
            _make_discovered_kernel("fp8"),
        ]
        machine = _make_machine("gpu1")

        mock_pool = MagicMock()
        mock_pool.allocate.return_value = machine
        mock_pool.validate_connectivity.return_value = [
            {"host": "gpu1", "reachable": True, "rocm_smi_ok": True, "error": None}
        ]

        supervisor_instances = []

        def make_supervisor(**kwargs):
            sv = MagicMock()
            sv.run.return_value = _make_supervisor_result(
                kwargs["config"].kernel_name
            )
            supervisor_instances.append(sv)
            return sv

        with patch(
            "tuning_agent.orchestrator.KernelDiscovery"
        ) as MockKD, patch(
            "tuning_agent.orchestrator.MachinePool",
            return_value=mock_pool,
        ), patch(
            "tuning_agent.orchestrator.RemoteExecutor"
        ), patch(
            "tuning_agent.orchestrator.ArtifactManager"
        ), patch(
            "tuning_agent.orchestrator.KernelSupervisor",
            side_effect=make_supervisor,
        ) as MockSupervisor:
            MockKD.return_value.discover_all.return_value = kernels

            orch = Orchestrator(config=config, repo_root="/repo", run_id="test-run")
            orch.run()

        assert MockSupervisor.call_count == 3

    def test_run_passes_correct_kernel_name_to_supervisor(self):
        """The SupervisorConfig passed to KernelSupervisor carries the right kernel name."""
        config = _make_config(machines=[_make_machine("gpu1")])
        kernel = _make_discovered_kernel("fp8_rowwise")
        machine = _make_machine("gpu1")

        mock_pool = MagicMock()
        mock_pool.allocate.return_value = machine
        mock_pool.validate_connectivity.return_value = [
            {"host": "gpu1", "reachable": True, "rocm_smi_ok": True, "error": None}
        ]

        captured_configs = []

        def make_supervisor(**kwargs):
            captured_configs.append(kwargs["config"])
            sv = MagicMock()
            sv.run.return_value = _make_supervisor_result("fp8_rowwise")
            return sv

        with patch(
            "tuning_agent.orchestrator.KernelDiscovery"
        ) as MockKD, patch(
            "tuning_agent.orchestrator.MachinePool",
            return_value=mock_pool,
        ), patch(
            "tuning_agent.orchestrator.RemoteExecutor"
        ), patch(
            "tuning_agent.orchestrator.ArtifactManager"
        ), patch(
            "tuning_agent.orchestrator.KernelSupervisor",
            side_effect=make_supervisor,
        ):
            MockKD.return_value.discover_all.return_value = [kernel]

            orch = Orchestrator(config=config, repo_root="/repo", run_id="test-run")
            orch.run()

        assert len(captured_configs) == 1
        assert captured_configs[0].kernel_name == "fp8_rowwise"

    def test_run_applies_kernel_overrides(self):
        """Per-kernel overrides from config.kernels.overrides are passed to SupervisorConfig."""
        overrides_cfg = {
            "a8w8": KernelOverrides(m_leq_16_bucket_name="M_LEQ_16", extra_block_k=128)
        }
        kernels_cfg = KernelsConfig(overrides=overrides_cfg)
        config = _make_config(kernels=kernels_cfg, machines=[_make_machine("gpu1")])
        kernel = _make_discovered_kernel("a8w8")
        machine = _make_machine("gpu1")

        mock_pool = MagicMock()
        mock_pool.allocate.return_value = machine
        mock_pool.validate_connectivity.return_value = [
            {"host": "gpu1", "reachable": True, "rocm_smi_ok": True, "error": None}
        ]

        captured_configs = []

        def make_supervisor(**kwargs):
            captured_configs.append(kwargs["config"])
            sv = MagicMock()
            sv.run.return_value = _make_supervisor_result("a8w8")
            return sv

        with patch(
            "tuning_agent.orchestrator.KernelDiscovery"
        ) as MockKD, patch(
            "tuning_agent.orchestrator.MachinePool",
            return_value=mock_pool,
        ), patch(
            "tuning_agent.orchestrator.RemoteExecutor"
        ), patch(
            "tuning_agent.orchestrator.ArtifactManager"
        ), patch(
            "tuning_agent.orchestrator.KernelSupervisor",
            side_effect=make_supervisor,
        ):
            MockKD.return_value.discover_all.return_value = [kernel]

            orch = Orchestrator(config=config, repo_root="/repo", run_id="test-run")
            orch.run()

        assert captured_configs[0].kernel_overrides == {
            "m_leq_16_bucket_name": "M_LEQ_16",
            "extra_block_k": 128,
        }


# ---------------------------------------------------------------------------
# test_generate_summary_counts_kernels
# ---------------------------------------------------------------------------


class TestGenerateSummary:
    """generate_summary() must return the correct aggregate metrics."""

    def test_generate_summary_counts_kernels(self):
        """Summary contains correct total/successful/failed kernel counts."""
        orch = Orchestrator(
            config=_make_config(),
            repo_root="/repo",
            run_id="test-run",
        )

        results = {
            "a8w8": _make_supervisor_result("a8w8", success=True, shapes_tuned=20),
            "a16w16": _make_supervisor_result("a16w16", success=True, shapes_tuned=15),
            "broken": _make_supervisor_result("broken", success=False, shapes_tuned=0),
        }
        summary = orch.generate_summary(results, wall_time=120.0)

        assert summary["total_kernels"] == 3
        assert summary["successful_kernels"] == 2
        assert summary["failed_kernels"] == 1

    def test_generate_summary_total_shapes(self):
        """total_shapes is the sum of shapes_tuned across all kernels."""
        orch = Orchestrator(
            config=_make_config(),
            repo_root="/repo",
            run_id="test-run",
        )

        results = {
            "k1": _make_supervisor_result("k1", shapes_tuned=10),
            "k2": _make_supervisor_result("k2", shapes_tuned=25),
            "k3": _make_supervisor_result("k3", shapes_tuned=5),
        }
        summary = orch.generate_summary(results, wall_time=60.0)

        assert summary["total_shapes"] == 40

    def test_generate_summary_wall_time(self):
        """wall_time_seconds is propagated to the summary."""
        orch = Orchestrator(
            config=_make_config(),
            repo_root="/repo",
            run_id="test-run",
        )
        summary = orch.generate_summary({}, wall_time=300.5)
        assert summary["wall_time_seconds"] == pytest.approx(300.5)

    def test_generate_summary_per_kernel_structure(self):
        """Each per-kernel entry has the expected keys."""
        orch = Orchestrator(
            config=_make_config(),
            repo_root="/repo",
            run_id="test-run",
        )
        results = {
            "a8w8": _make_supervisor_result("a8w8", geomean_speedup=1.08, regressions=1)
        }
        summary = orch.generate_summary(results)

        assert "a8w8" in summary["kernels"]
        entry = summary["kernels"]["a8w8"]
        assert "success" in entry
        assert "geomean_speedup" in entry
        assert "regressions" in entry
        assert "shapes_tuned" in entry
        assert "summary" in entry
        assert entry["geomean_speedup"] == pytest.approx(1.08)
        assert entry["regressions"] == 1

    def test_generate_summary_includes_run_id(self):
        """The run_id is echoed into the summary."""
        orch = Orchestrator(
            config=_make_config(),
            repo_root="/repo",
            run_id="my-unique-run",
        )
        summary = orch.generate_summary({})
        assert summary["run_id"] == "my-unique-run"

    def test_generate_summary_overall_geomean(self):
        """overall_geomean_speedup is the geometric mean of per-kernel geomeans."""
        import math

        orch = Orchestrator(
            config=_make_config(),
            repo_root="/repo",
            run_id="test-run",
        )
        # geomean of 1.0 and 1.21 = sqrt(1.0 * 1.21) = 1.1
        results = {
            "k1": _make_supervisor_result("k1", geomean_speedup=1.0),
            "k2": _make_supervisor_result("k2", geomean_speedup=1.21),
        }
        summary = orch.generate_summary(results)
        assert summary["overall_geomean_speedup"] == pytest.approx(
            math.sqrt(1.0 * 1.21), rel=1e-6
        )

    def test_generate_summary_empty_results(self):
        """generate_summary handles an empty results dict without error."""
        orch = Orchestrator(
            config=_make_config(),
            repo_root="/repo",
            run_id="test-run",
        )
        summary = orch.generate_summary({})
        assert summary["total_kernels"] == 0
        assert summary["successful_kernels"] == 0
        assert summary["failed_kernels"] == 0
        assert summary["total_shapes"] == 0
        assert summary["overall_geomean_speedup"] is None
        assert summary["kernels"] == {}


# ---------------------------------------------------------------------------
# test_dashboard_updated_during_run
# ---------------------------------------------------------------------------


class TestDashboardUpdates:
    """run() must call dashboard.update_kernel for every processed kernel."""

    def test_dashboard_updated_during_run(self):
        """dashboard.update_kernel is called at least once for each kernel."""
        config = _make_config(machines=[_make_machine("gpu1")])
        kernels = [_make_discovered_kernel("a8w8"), _make_discovered_kernel("fp8")]
        machine = _make_machine("gpu1")
        dashboard = MagicMock()

        mock_pool = MagicMock()
        mock_pool.allocate.return_value = machine
        mock_pool.validate_connectivity.return_value = [
            {"host": "gpu1", "reachable": True, "rocm_smi_ok": True, "error": None}
        ]

        def make_supervisor(**kwargs):
            sv = MagicMock()
            sv.run.return_value = _make_supervisor_result(kwargs["config"].kernel_name)
            return sv

        with patch(
            "tuning_agent.orchestrator.KernelDiscovery"
        ) as MockKD, patch(
            "tuning_agent.orchestrator.MachinePool",
            return_value=mock_pool,
        ), patch(
            "tuning_agent.orchestrator.RemoteExecutor"
        ), patch(
            "tuning_agent.orchestrator.ArtifactManager"
        ), patch(
            "tuning_agent.orchestrator.KernelSupervisor",
            side_effect=make_supervisor,
        ):
            MockKD.return_value.discover_all.return_value = kernels

            orch = Orchestrator(
                config=config,
                repo_root="/repo",
                run_id="test-run",
                dashboard=dashboard,
            )
            orch.run()

        # update_kernel must have been called at least once for each kernel.
        update_kernel_calls = dashboard.update_kernel.call_args_list
        kernel_names_updated = {c.kwargs.get("name") or c.args[0] for c in update_kernel_calls}
        assert "a8w8" in kernel_names_updated
        assert "fp8" in kernel_names_updated

    def test_dashboard_update_machine_called_for_allocation(self):
        """dashboard.update_machine is called with 'busy' when a machine is allocated."""
        config = _make_config(machines=[_make_machine("gpu1", gpus=[0, 1])])
        kernel = _make_discovered_kernel("a8w8")
        machine = _make_machine("gpu1", gpus=[0, 1])
        dashboard = MagicMock()

        mock_pool = MagicMock()
        mock_pool.allocate.return_value = machine
        mock_pool.validate_connectivity.return_value = [
            {"host": "gpu1", "reachable": True, "rocm_smi_ok": True, "error": None}
        ]

        def make_supervisor(**kwargs):
            sv = MagicMock()
            sv.run.return_value = _make_supervisor_result("a8w8")
            return sv

        with patch(
            "tuning_agent.orchestrator.KernelDiscovery"
        ) as MockKD, patch(
            "tuning_agent.orchestrator.MachinePool",
            return_value=mock_pool,
        ), patch(
            "tuning_agent.orchestrator.RemoteExecutor"
        ), patch(
            "tuning_agent.orchestrator.ArtifactManager"
        ), patch(
            "tuning_agent.orchestrator.KernelSupervisor",
            side_effect=make_supervisor,
        ):
            MockKD.return_value.discover_all.return_value = [kernel]

            orch = Orchestrator(
                config=config,
                repo_root="/repo",
                run_id="test-run",
                dashboard=dashboard,
            )
            orch.run()

        # At some point during the run the machine must be marked busy.
        busy_calls = [
            c
            for c in dashboard.update_machine.call_args_list
            if (c.kwargs.get("state") or (c.args[1] if len(c.args) > 1 else None)) == "busy"
        ]
        assert len(busy_calls) >= 1

    def test_dashboard_update_machine_called_after_release(self):
        """dashboard.update_machine is called with 'idle' after the machine is released."""
        config = _make_config(machines=[_make_machine("gpu1")])
        kernel = _make_discovered_kernel("a8w8")
        machine = _make_machine("gpu1")
        dashboard = MagicMock()

        mock_pool = MagicMock()
        mock_pool.allocate.return_value = machine
        mock_pool.validate_connectivity.return_value = [
            {"host": "gpu1", "reachable": True, "rocm_smi_ok": True, "error": None}
        ]

        def make_supervisor(**kwargs):
            sv = MagicMock()
            sv.run.return_value = _make_supervisor_result("a8w8")
            return sv

        with patch(
            "tuning_agent.orchestrator.KernelDiscovery"
        ) as MockKD, patch(
            "tuning_agent.orchestrator.MachinePool",
            return_value=mock_pool,
        ), patch(
            "tuning_agent.orchestrator.RemoteExecutor"
        ), patch(
            "tuning_agent.orchestrator.ArtifactManager"
        ), patch(
            "tuning_agent.orchestrator.KernelSupervisor",
            side_effect=make_supervisor,
        ):
            MockKD.return_value.discover_all.return_value = [kernel]

            orch = Orchestrator(
                config=config,
                repo_root="/repo",
                run_id="test-run",
                dashboard=dashboard,
            )
            orch.run()

        # After release the machine should be set back to idle.
        idle_calls = [
            c
            for c in dashboard.update_machine.call_args_list
            if (c.kwargs.get("state") or (c.args[1] if len(c.args) > 1 else None)) == "idle"
        ]
        assert len(idle_calls) >= 1


# ---------------------------------------------------------------------------
# Orchestrator initialisation tests
# ---------------------------------------------------------------------------


class TestOrchestratorInit:
    """__init__ wiring and defaults."""

    def test_run_id_auto_generated_when_not_provided(self):
        """A run_id is auto-generated when not explicitly passed."""
        config = _make_config()
        with patch(
            "tuning_agent.orchestrator.KernelDiscovery"
        ):
            orch = Orchestrator(config=config, repo_root="/repo")
        assert orch.run_id is not None
        assert len(orch.run_id) > 0

    def test_run_id_used_when_provided(self):
        """A provided run_id is stored unchanged."""
        config = _make_config()
        with patch(
            "tuning_agent.orchestrator.KernelDiscovery"
        ):
            orch = Orchestrator(config=config, repo_root="/repo", run_id="custom-run-42")
        assert orch.run_id == "custom-run-42"

    def test_machine_pool_created_from_config_machines(self):
        """MachinePool is constructed with the machines list from config."""
        machines = [_make_machine("h1"), _make_machine("h2")]
        config = _make_config(machines=machines)

        with patch(
            "tuning_agent.orchestrator.MachinePool"
        ) as MockPool, patch(
            "tuning_agent.orchestrator.KernelDiscovery"
        ):
            Orchestrator(config=config, repo_root="/repo", run_id="test")

        MockPool.assert_called_once_with(machines)

    def test_kernel_discovery_created_with_repo_root_and_arch(self):
        """KernelDiscovery is constructed with repo_root and gpu.arch."""
        config = _make_config()
        config.gpu.arch = "gfx942"

        with patch(
            "tuning_agent.orchestrator.KernelDiscovery"
        ) as MockKD, patch(
            "tuning_agent.orchestrator.MachinePool"
        ):
            Orchestrator(config=config, repo_root="/custom/repo", run_id="test")

        MockKD.assert_called_once_with("/custom/repo", "gfx942")

    def test_dashboard_and_notifier_stored(self):
        """dashboard and notifier are stored as attributes."""
        config = _make_config()
        dashboard = MagicMock()
        notifier = MagicMock()

        with patch(
            "tuning_agent.orchestrator.KernelDiscovery"
        ), patch(
            "tuning_agent.orchestrator.MachinePool"
        ):
            orch = Orchestrator(
                config=config,
                repo_root="/repo",
                run_id="test",
                dashboard=dashboard,
                notifier=notifier,
            )

        assert orch.dashboard is dashboard
        assert orch.notifier is notifier
