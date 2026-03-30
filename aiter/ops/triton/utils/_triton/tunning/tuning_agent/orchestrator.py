"""Top-level orchestration loop for the agentic Triton kernel tuning pipeline.

The :class:`Orchestrator` ties together kernel discovery, machine scheduling,
per-kernel supervision, and dashboard/notification updates into a single
end-to-end coordination loop.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime
from typing import Dict, List, Optional

from .artifacts import ArtifactManager
from .config import PipelineConfig
from .dashboard import Dashboard
from .kernel_discovery import DiscoveredKernel, KernelDiscovery
from .kernel_supervisor import KernelSupervisor, SupervisorConfig, SupervisorResult
from .machine_pool import MachinePool
from .notifications import NotificationLevel, Notifier
from .remote import RemoteExecutor
from .types import MachineInfo

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinate discovery, machine allocation, and kernel supervision.

    Parameters
    ----------
    config:
        Top-level :class:`~.config.PipelineConfig` for the run.
    repo_root:
        Absolute path to the root of the aiter repository checkout.
    run_id:
        Unique identifier for this run.  Auto-generated from the current
        datetime when not provided.
    dashboard:
        Optional :class:`~.dashboard.Dashboard` instance to receive
        live progress updates.
    notifier:
        Optional :class:`~.notifications.Notifier` used to surface events
        and request human approvals.
    """

    def __init__(
        self,
        config: PipelineConfig,
        repo_root: str,
        run_id: Optional[str] = None,
        dashboard: Optional[Dashboard] = None,
        notifier: Optional[Notifier] = None,
    ) -> None:
        self.config = config
        self.repo_root = repo_root

        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id

        self.machine_pool = MachinePool(config.machines)
        self.kernel_discovery = KernelDiscovery(repo_root, config.gpu.arch or "gfx950")
        self.dashboard = dashboard
        self.notifier = notifier

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover_kernels(self) -> List[DiscoveredKernel]:
        """Discover all kernels subject to config include/exclude filters.

        Returns
        -------
        List[DiscoveredKernel]
            Discovered kernels sorted by ``(category, name)``.
        """
        kernels = self.kernel_discovery.discover_all(
            exclude=self.config.kernels.exclude or None,
            include=self.config.kernels.include or None,
        )
        count = len(kernels)
        logger.info("Discovered %d kernel(s) to tune.", count)
        if self.dashboard is not None:
            self.dashboard.add_log(f"Discovered {count} kernel(s) to tune.")
        return kernels

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Discover kernels, validate connectivity, and run each kernel.

        For every discovered kernel the method:

        1. Allocates a machine from the :class:`MachinePool`.
        2. Creates a :class:`~.remote.RemoteExecutor` for that machine.
        3. Creates an :class:`~.artifacts.ArtifactManager` scoped to this
           run and kernel.
        4. Builds a :class:`~.kernel_supervisor.SupervisorConfig` from
           ``self.config`` and the kernel's metadata.
        5. Runs a :class:`~.kernel_supervisor.KernelSupervisor`.
        6. Releases the machine and updates the dashboard.

        Returns
        -------
        dict
            Summary report produced by :meth:`generate_summary`.
        """
        run_start = time.time()

        # 1. Discover kernels.
        kernels = self.discover_kernels()

        # 2. Validate machine connectivity.
        connectivity = self.machine_pool.validate_connectivity()
        if self.dashboard is not None:
            for entry in connectivity:
                state = "idle" if entry.get("reachable") and entry.get("rocm_smi_ok") else "dead"
                self.dashboard.update_machine(
                    host=entry["host"],
                    state=state,
                )
        if self.notifier is not None:
            dead = [e["host"] for e in connectivity if not e.get("reachable")]
            if dead:
                self.notifier.notify(
                    NotificationLevel.WARNING,
                    "Machine connectivity",
                    f"Unreachable hosts: {', '.join(dead)}",
                )

        # 3. Run each kernel.
        results: Dict[str, SupervisorResult] = {}

        for kernel in kernels:
            kernel_start = time.time()
            machine: Optional[MachineInfo] = None

            try:
                machine = self.machine_pool.allocate(kernel.name)

                # Update dashboard: machine is busy.
                if self.dashboard is not None:
                    self.dashboard.update_machine(
                        host=machine.host,
                        state="busy",
                        kernel=kernel.name,
                        gpu_count=machine.gpu_count,
                    )
                    self.dashboard.update_kernel(
                        name=kernel.name,
                        phase=0,
                        phase_name="SETUP",
                        status="running",
                        elapsed=0.0,
                    )

                executor = RemoteExecutor(machine)
                artifact_manager = ArtifactManager(
                    executor=executor,
                    kernel_name=kernel.name,
                    run_id=self.run_id,
                )

                # Resolve per-kernel overrides.
                kernel_overrides = None
                if kernel.name in (self.config.kernels.overrides or {}):
                    ko = self.config.kernels.overrides[kernel.name]
                    kernel_overrides = {
                        "m_leq_16_bucket_name": ko.m_leq_16_bucket_name,
                        "extra_block_k": ko.extra_block_k,
                    }

                supervisor_config = SupervisorConfig(
                    kernel_name=kernel.name,
                    baseline_repo=self.config.baseline,
                    target_repo=self.config.target,
                    container_config=self.config.container,
                    triton_install=self.config.triton_install,
                    tuning_config=self.config.tuning,
                    gpu_ids=machine.gpus if machine.gpus else [0],
                    kernel_overrides=kernel_overrides,
                )

                supervisor = KernelSupervisor(
                    executor=executor,
                    config=supervisor_config,
                    artifact_manager=artifact_manager,
                    notifier=self.notifier or Notifier(),
                    progress_callback=self._make_progress_callback(kernel.name, kernel_start),
                )

                result = supervisor.run()
                results[kernel.name] = result

                # Update dashboard: kernel done.
                if self.dashboard is not None:
                    status = "done" if result.success else "failed"
                    self.dashboard.update_kernel(
                        name=kernel.name,
                        phase=6,
                        phase_name="COMMIT",
                        status=status,
                        elapsed=time.time() - kernel_start,
                    )

            except Exception as exc:  # noqa: BLE001
                logger.error("Kernel '%s' failed with exception: %s", kernel.name, exc)
                if self.dashboard is not None:
                    self.dashboard.update_kernel(
                        name=kernel.name,
                        phase=0,
                        phase_name="ERROR",
                        status="failed",
                        elapsed=time.time() - kernel_start,
                    )
                    self.dashboard.add_log(f"ERROR: kernel '{kernel.name}': {exc}")
            finally:
                if machine is not None:
                    self.machine_pool.release(machine.host)
                    if self.dashboard is not None:
                        self.dashboard.update_machine(
                            host=machine.host,
                            state="idle",
                            kernel=None,
                            gpu_count=machine.gpu_count,
                        )

        wall_time = time.time() - run_start
        summary = self.generate_summary(results, wall_time=wall_time)
        return summary

    # ------------------------------------------------------------------
    # Summary generation
    # ------------------------------------------------------------------

    def generate_summary(
        self,
        results: Dict[str, SupervisorResult],
        wall_time: float = 0.0,
    ) -> dict:
        """Produce a structured summary dict from per-kernel results.

        Parameters
        ----------
        results:
            Mapping of ``kernel_name`` to :class:`~.kernel_supervisor.SupervisorResult`.
        wall_time:
            Total elapsed wall-clock seconds for the run.

        Returns
        -------
        dict
            Summary with keys:

            ``total_kernels``
                Number of kernels that were processed.
            ``successful_kernels``
                Number of kernels that completed without errors.
            ``failed_kernels``
                Number of kernels that failed.
            ``total_shapes``
                Aggregate count of tuned shapes across all kernels.
            ``wall_time_seconds``
                Total elapsed time.
            ``kernels``
                Per-kernel details dict.
        """
        per_kernel: Dict[str, dict] = {}
        total_shapes = 0
        successful = 0
        failed = 0
        geomeans: List[float] = []

        for kernel_name, result in results.items():
            # Count regressions from escalations.
            regressions = [
                e for e in result.escalations if e.severity == "warning"
            ]

            # Count tuned shapes from TUNING phase result if present.
            shapes_tuned = 0
            from .kernel_supervisor import Phase
            tuning_pr = result.phase_results.get(Phase.TUNING)
            if tuning_pr is not None and tuning_pr.data:
                shapes_tuned = tuning_pr.data.get("shapes_tuned", 0)
            total_shapes += shapes_tuned

            geomean = result.geomean_speedup
            if geomean is not None:
                geomeans.append(geomean)

            per_kernel[kernel_name] = {
                "success": result.success,
                "geomean_speedup": geomean,
                "regressions": len(regressions),
                "shapes_tuned": shapes_tuned,
                "summary": result.summary,
            }

            if result.success:
                successful += 1
            else:
                failed += 1

        # Overall geomean across all kernels.
        overall_geomean: Optional[float] = None
        if geomeans:
            log_sum = sum(math.log(g) for g in geomeans if g and g > 0)
            overall_geomean = math.exp(log_sum / len(geomeans))

        return {
            "run_id": self.run_id,
            "total_kernels": len(results),
            "successful_kernels": successful,
            "failed_kernels": failed,
            "total_shapes": total_shapes,
            "overall_geomean_speedup": overall_geomean,
            "wall_time_seconds": wall_time,
            "kernels": per_kernel,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_progress_callback(self, kernel_name: str, kernel_start: float):
        """Return a progress callback that forwards phase updates to the dashboard."""
        dashboard = self.dashboard

        def _callback(phase, message: str) -> None:
            elapsed = time.time() - kernel_start
            if dashboard is not None:
                from .kernel_supervisor import Phase
                dashboard.update_kernel(
                    name=kernel_name,
                    phase=int(phase),
                    phase_name=phase.name if hasattr(phase, "name") else str(phase),
                    progress=message,
                    elapsed=elapsed,
                    status="running",
                )
                dashboard.add_log(f"[{kernel_name}] Phase {phase}: {message}")

        return _callback
