"""KernelSupervisor core types, state machine, and checkpoint logic.

This module provides the foundation for the agentic Triton kernel tuning
supervisor. It defines the Phase state machine, configuration, state tracking,
and checkpoint/resume logic.  Phase runners and the main execution loop are
added by subsequent tasks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING, Type

from .types import ContainerConfig, RepoConfig, ShapeResult, TritonInstallConfig, TuningConfig
from .remote import RemoteExecutor
from .artifacts import ArtifactManager
from .notifications import Notifier
from .subagents.base import SubagentResult
from .subagents.setup_agent import SetupAgent
from .subagents.discovery_agent import DiscoveryAgent
from .subagents.script_creator_agent import ScriptCreatorAgent
from .subagents.baseline_agent import BaselineAgent
from .subagents.validation_agent import ValidationAgent
from .subagents.tuning_agent import TuningAgent
from .subagents.pattern_analyzer_agent import PatternAnalyzerAgent
from .subagents.config_generator_agent import ConfigGeneratorAgent
from .subagents.regression_fixer_agent import RegressionFixerAgent

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Phase(IntEnum):
    """Ordered phases of a kernel tuning run."""

    SETUP = 0
    DISCOVERY = 1
    BASELINE = 2
    UNTUNED_VALIDATION = 3
    TUNING = 4
    VALIDATION_AND_FIX = 5
    COMMIT = 6


class TuningMode(str, Enum):
    """Selects which shapes/configurations are exercised during tuning."""

    REGRESSION_ONLY = "regression_only"
    FULL = "full"


# ---------------------------------------------------------------------------
# Result / request dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PhaseResult:
    """Outcome of a single pipeline phase."""

    phase: Phase
    success: bool
    data: dict
    error: Optional[str]
    duration_seconds: float


@dataclass
class EscalationRequest:
    """A request for human attention or approval during the pipeline."""

    severity: str  # "warning" | "approval_required" | "fatal"
    message: str
    details: Optional[str] = None


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class SupervisorConfig:
    """All configuration needed to run a single kernel through the pipeline."""

    kernel_name: str
    baseline_repo: RepoConfig
    target_repo: RepoConfig
    container_config: ContainerConfig
    triton_install: TritonInstallConfig
    tuning_config: TuningConfig
    gpu_ids: List[int]
    kernel_overrides: Optional[dict] = None
    gpu_arch: Optional[str] = None


# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------


@dataclass
class SupervisorState:
    """Mutable runtime state of the KernelSupervisor."""

    current_phase: Phase = Phase.SETUP
    completed_phases: List[Phase] = field(default_factory=list)
    phase_results: Dict[Phase, PhaseResult] = field(default_factory=dict)
    escalations: List[EscalationRequest] = field(default_factory=list)
    shapes: Optional[List] = None
    baseline_data: Optional[List] = None
    untuned_data: Optional[List] = None
    tuned_data: Optional[List] = None


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class SupervisorResult:
    """Final outcome of a complete kernel tuning run."""

    kernel_name: str
    success: bool
    phase_results: Dict[Phase, PhaseResult]
    escalations: List[EscalationRequest]
    geomean_speedup: Optional[float]
    summary: str


# ---------------------------------------------------------------------------
# KernelSupervisor
# ---------------------------------------------------------------------------


class KernelSupervisor:
    """Orchestrates a full kernel tuning run through all pipeline phases.

    This class manages state transitions, checkpointing, and escalation
    handling.  Phase runners and the main execution loop are provided by
    higher-level mixins / subclasses added in later tasks.

    Parameters
    ----------
    executor:
        A :class:`~.remote.RemoteExecutor` used by phase runners to
        interact with the remote machine and Docker container.
    config:
        :class:`SupervisorConfig` describing which kernel to tune and how.
    artifact_manager:
        :class:`~.artifacts.ArtifactManager` for checkpoint persistence
        and result storage.
    notifier:
        :class:`~.notifications.Notifier` used to surface events and
        request approvals.
    progress_callback:
        Optional callable invoked with ``(phase, message)`` whenever a
        notable progress event occurs.
    """

    def __init__(
        self,
        executor: RemoteExecutor,
        config: SupervisorConfig,
        artifact_manager: ArtifactManager,
        notifier: Notifier,
        progress_callback: Optional[Callable[[Phase, str], None]] = None,
    ) -> None:
        self.executor = executor
        self.config = config
        self.artifact_manager = artifact_manager
        self.notifier = notifier
        self.progress_callback = progress_callback
        self.state = SupervisorState()

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _should_skip_phase(self, phase: Phase) -> bool:
        """Return ``True`` if *phase* has already been completed.

        Completion is determined by whether the artifact manager has a
        phase-complete marker for ``phase.value``.

        Parameters
        ----------
        phase:
            The :class:`Phase` to check.

        Returns
        -------
        bool
            ``True`` when the phase marker exists (phase can be skipped).
        """
        return self.artifact_manager.is_phase_complete(phase.value)

    def _record_phase_complete(self, phase: Phase, result: PhaseResult) -> None:
        """Persist a phase completion marker and update in-memory state.

        Writes a JSON checkpoint via the artifact manager and records the
        phase in :attr:`~SupervisorState.completed_phases` and
        :attr:`~SupervisorState.phase_results`.

        Parameters
        ----------
        phase:
            The :class:`Phase` that just completed.
        result:
            The :class:`PhaseResult` produced by that phase.
        """
        # Persist checkpoint to disk (for resume on restart)
        summary: Dict = {
            "success": result.success,
            "duration_seconds": result.duration_seconds,
        }
        if result.error is not None:
            summary["error"] = result.error
        if result.data:
            summary["data"] = result.data

        self.artifact_manager.mark_phase_complete(phase.value, summary)

        # Update in-memory state
        if phase not in self.state.completed_phases:
            self.state.completed_phases.append(phase)
        self.state.phase_results[phase] = result

    def _get_resume_phase(self) -> Phase:
        """Return the first :class:`Phase` that has not yet been completed.

        Iterates through all phases in order and returns the first one
        whose completion marker does not exist.  If all phases are complete
        the method returns the final phase (:attr:`Phase.COMMIT`).

        Returns
        -------
        Phase
            The phase from which execution should (re)start.
        """
        for phase in Phase:
            if not self._should_skip_phase(phase):
                return phase
        # All phases complete — return the last one
        return Phase.COMMIT

    # ------------------------------------------------------------------
    # Subagent dispatch
    # ------------------------------------------------------------------

    def _dispatch_subagent(self, subagent_class: Type, **kwargs) -> SubagentResult:
        """Instantiate *subagent_class* and run it, returning the result.

        The subagent is constructed with ``executor``, ``kernel_name``, and
        ``artifact_dir`` drawn from this supervisor's own attributes, plus any
        additional *kwargs* passed by the caller.

        Parameters
        ----------
        subagent_class:
            The subagent class to instantiate (must accept the standard
            ``executor``, ``kernel_name``, ``artifact_dir`` constructor
            arguments).
        **kwargs:
            Extra keyword arguments forwarded verbatim to the constructor.

        Returns
        -------
        SubagentResult
            The result produced by the subagent's :meth:`run` method.
        """
        subagent = subagent_class(
            executor=self.executor,
            kernel_name=self.config.kernel_name,
            artifact_dir=self.artifact_manager.remote_dir,
            **kwargs,
        )
        return subagent.run()

    def _dispatch_with_retry(
        self,
        subagent_class: Type,
        max_retries: int = 2,
        **kwargs,
    ) -> SubagentResult:
        """Dispatch *subagent_class* with automatic retry on failure.

        On each attempt a *fresh* subagent instance is created.  If the
        result's ``success`` flag is ``True`` the result is returned
        immediately.  After *max_retries* additional attempts (i.e.
        1 + *max_retries* total calls) the last failure result is returned.

        Parameters
        ----------
        subagent_class:
            The subagent class to instantiate.
        max_retries:
            Maximum number of additional attempts after the first failure.
        **kwargs:
            Extra keyword arguments forwarded to :meth:`_dispatch_subagent`.

        Returns
        -------
        SubagentResult
            The first successful result, or the result from the final
            failed attempt.
        """
        result: SubagentResult = self._dispatch_subagent(subagent_class, **kwargs)
        if result.success:
            return result

        for _ in range(max_retries):
            result = self._dispatch_subagent(subagent_class, **kwargs)
            if result.success:
                return result

        return result

    # ------------------------------------------------------------------
    # Triton switching
    # ------------------------------------------------------------------

    def _switch_triton(self, repo_config: RepoConfig) -> None:
        """Check out a Triton branch and install it inside the container.

        Runs ``git checkout <branch>`` followed by the configured install
        command inside the container, then calls
        :meth:`~.remote.RemoteExecutor.verify_environment` to confirm the
        environment is ready.

        Parameters
        ----------
        repo_config:
            :class:`~.types.RepoConfig` whose ``triton_repo`` and
            ``triton_branch`` fields identify the target checkout.

        Raises
        ------
        Exception
            Any error raised by ``docker_exec`` or ``verify_environment``
            is propagated to the caller.
        """
        checkout_cmd = (
            f"cd /workspace/triton "
            f"&& git checkout {repo_config.triton_branch} "
            f"&& {self.config.triton_install.command}"
        )
        self.executor.docker_exec(checkout_cmd)
        self.executor.verify_environment()

    # ------------------------------------------------------------------
    # Phase timeout check
    # ------------------------------------------------------------------

    def _check_phase_timeout(self, phase: Phase, start_time: float) -> bool:
        """Return ``True`` if the current phase has exceeded its time budget.

        Parameters
        ----------
        phase:
            The :class:`Phase` being executed (reserved for future
            per-phase timeout configuration).
        start_time:
            Unix timestamp (from :func:`time.time`) recorded when the
            phase began.

        Returns
        -------
        bool
            ``True`` when the elapsed time exceeds
            ``config.tuning_config.timeouts.phase_max``.
        """
        elapsed = time.time() - start_time
        return elapsed > self.config.tuning_config.timeouts.phase_max

    # ------------------------------------------------------------------
    # Regression identification
    # ------------------------------------------------------------------

    def _identify_regressed_shapes(self) -> List[Tuple[int, int, int]]:
        """Compare untuned results against baseline to find regressed shapes.

        A shape ``(M, N, K)`` is considered regressed when its
        ``total_ns`` in the untuned run exceeds the baseline ``total_ns``
        by more than ``thresholds.regression_vs_baseline`` percent.

        The method only considers shapes that appear in *both* datasets.
        Any shape present only in one dataset is silently skipped.

        Returns
        -------
        List[Tuple[int, int, int]]
            List of ``(M, N, K)`` tuples whose performance regressed.
        """
        if not self.state.baseline_data or not self.state.untuned_data:
            return []

        threshold = self.config.tuning_config.thresholds.regression_vs_baseline

        # Build a lookup from (M, N, K) -> result dict for the baseline.
        baseline_map: Dict[Tuple[int, int, int], dict] = {
            (r["m"], r["n"], r["k"]): r for r in self.state.baseline_data
        }
        untuned_map: Dict[Tuple[int, int, int], dict] = {
            (r["m"], r["n"], r["k"]): r for r in self.state.untuned_data
        }

        regressed: List[Tuple[int, int, int]] = []
        for key, untuned_result in untuned_map.items():
            baseline_result = baseline_map.get(key)
            if baseline_result is None:
                continue
            limit = baseline_result["total_ns"] * (1 + threshold / 100)
            if untuned_result["total_ns"] > limit:
                regressed.append(key)

        return regressed

    # ------------------------------------------------------------------
    # Shape selection
    # ------------------------------------------------------------------

    def _determine_shapes_to_tune(self) -> List[Tuple[int, int, int]]:
        """Return the list of ``(M, N, K)`` shapes that require tuning.

        Behaviour depends on the configured :class:`TuningMode`:

        * :attr:`TuningMode.FULL` — return every shape stored in
          ``state.shapes``.
        * :attr:`TuningMode.REGRESSION_ONLY` — return only the shapes
          identified by :meth:`_identify_regressed_shapes`.  If no shapes
          regressed, an empty list is returned (tuning is skipped).

        Returns
        -------
        List[Tuple[int, int, int]]
            Shapes to tune, or an empty list if nothing needs tuning.
        """
        mode = self.config.tuning_config.mode
        if mode == TuningMode.FULL or mode == TuningMode.FULL.value:
            shapes = self.state.shapes or []
            if shapes and isinstance(shapes[0], dict):
                return [(r["m"], r["n"], r["k"]) for r in shapes]
            elif shapes and hasattr(shapes[0], "m"):
                return [(r.m, r.n, r.k) for r in shapes]
            return list(shapes)

        # REGRESSION_ONLY
        return self._identify_regressed_shapes()

    # ------------------------------------------------------------------
    # Phase runners 0–3
    # ------------------------------------------------------------------

    def _run_phase_0_setup(self) -> PhaseResult:
        """Run Phase 0: environment setup.

        Dispatches :class:`~.subagents.setup_agent.SetupAgent` to bootstrap
        the remote container environment.  On success the container ID is
        stored as ``self.container_id``.

        Returns
        -------
        PhaseResult
            Result for :attr:`Phase.SETUP`.
        """
        start_time = time.time()
        try:
            result = self._dispatch_with_retry(
                SetupAgent,
                image=self.config.container_config.image,
                run_script=self.config.container_config.run_script or "",
                repo_config={
                    "url": self.config.target_repo.aiter_repo,
                    "branch": self.config.target_repo.aiter_branch,
                    "triton_repo": self.config.target_repo.triton_repo,
                    "triton_branch": self.config.target_repo.triton_branch,
                },
                triton_install_config={
                    "method": "source",
                    "command": self.config.triton_install.command,
                },
            )
            if result.success:
                self.container_id = result.data.get("container_id")
            return PhaseResult(
                phase=Phase.SETUP,
                success=result.success,
                data=result.data,
                error=result.error,
                duration_seconds=time.time() - start_time,
            )
        except Exception as exc:  # noqa: BLE001
            return PhaseResult(
                phase=Phase.SETUP,
                success=False,
                data={},
                error=str(exc),
                duration_seconds=time.time() - start_time,
            )

    def _run_phase_1_discovery(self) -> PhaseResult:
        """Run Phase 1: kernel and shape discovery.

        Dispatches :class:`~.subagents.discovery_agent.DiscoveryAgent` to
        scan kernel sources and locate benchmark scripts.  If scripts are
        reported missing, :class:`~.subagents.script_creator_agent.ScriptCreatorAgent`
        is dispatched to create them.  On success the discovered shapes are
        stored in :attr:`~SupervisorState.shapes`.

        Returns
        -------
        PhaseResult
            Result for :attr:`Phase.DISCOVERY`.
        """
        start_time = time.time()
        try:
            result = self._dispatch_with_retry(
                DiscoveryAgent,
                kernel_source_path="",
                config_dir="",
                model_shapes_path=None,
                config_variant=self.config.kernel_name,
                gfx_arch=self.config.gpu_arch or "gfx950",
            )
            if result.success:
                self.state.shapes = result.data.get("shapes", [])
                missing_scripts = result.data.get("missing_scripts", [])
                if missing_scripts:
                    self._dispatch_with_retry(
                        ScriptCreatorAgent,
                        kernel_source_path="",
                        template_dir="",
                        missing_scripts=[{"type": s, "target_path": ""} for s in missing_scripts],
                    )
            return PhaseResult(
                phase=Phase.DISCOVERY,
                success=result.success,
                data=result.data,
                error=result.error,
                duration_seconds=time.time() - start_time,
            )
        except Exception as exc:  # noqa: BLE001
            return PhaseResult(
                phase=Phase.DISCOVERY,
                success=False,
                data={},
                error=str(exc),
                duration_seconds=time.time() - start_time,
            )

    def _run_phase_2_baseline(self) -> PhaseResult:
        """Run Phase 2: baseline benchmarking.

        Switches to the baseline Triton version via :meth:`_switch_triton`,
        dispatches :class:`~.subagents.baseline_agent.BaselineAgent` to
        collect timings, stores results in
        :attr:`~SupervisorState.baseline_data`, then reinstalls the target
        Triton.

        Returns
        -------
        PhaseResult
            Result for :attr:`Phase.BASELINE`.
        """
        start_time = time.time()
        try:
            self._switch_triton(self.config.baseline_repo)
            shapes = self.state.shapes or []
            result = self._dispatch_with_retry(
                BaselineAgent,
                shapes=shapes,
                bench_script="",
                gpu_id=self.config.gpu_ids[0],
                kernel_variant=self.config.kernel_name,
            )
            if result.success:
                self.state.baseline_data = result.data.get("results", [])
            self._switch_triton(self.config.target_repo)
            return PhaseResult(
                phase=Phase.BASELINE,
                success=result.success,
                data=result.data,
                error=result.error,
                duration_seconds=time.time() - start_time,
            )
        except Exception as exc:  # noqa: BLE001
            return PhaseResult(
                phase=Phase.BASELINE,
                success=False,
                data={},
                error=str(exc),
                duration_seconds=time.time() - start_time,
            )

    def _run_phase_3_untuned(self) -> PhaseResult:
        """Run Phase 3: untuned validation benchmarking.

        Dispatches :class:`~.subagents.validation_agent.ValidationAgent`
        with the current shapes and all configured GPU IDs to collect
        untuned timings.  Results are stored in
        :attr:`~SupervisorState.untuned_data`.

        Returns
        -------
        PhaseResult
            Result for :attr:`Phase.UNTUNED_VALIDATION`.
        """
        start_time = time.time()
        try:
            shapes = self.state.shapes or []
            result = self._dispatch_with_retry(
                ValidationAgent,
                shapes=shapes,
                bench_script="",
                gpu_ids=self.config.gpu_ids,
                kernel_variant=self.config.kernel_name,
            )
            if result.success:
                self.state.untuned_data = result.data.get("results", [])
            return PhaseResult(
                phase=Phase.UNTUNED_VALIDATION,
                success=result.success,
                data=result.data,
                error=result.error,
                duration_seconds=time.time() - start_time,
            )
        except Exception as exc:  # noqa: BLE001
            return PhaseResult(
                phase=Phase.UNTUNED_VALIDATION,
                success=False,
                data={},
                error=str(exc),
                duration_seconds=time.time() - start_time,
            )

    # ------------------------------------------------------------------
    # Phase 4 runner
    # ------------------------------------------------------------------

    def _run_phase_4_tuning(self) -> PhaseResult:
        """Execute Phase 4: the full scout → analyse → tune → config pipeline.

        Steps
        -----
        1. Determine which shapes need tuning via
           :meth:`_determine_shapes_to_tune`.  If the list is empty, return
           immediately with ``{"skipped": True}``.
        2. **Scout phase** — select a representative subset of shapes and
           dispatch :class:`~.subagents.tuning_agent.TuningAgent` with a
           broad search space.
        3. **Pattern analysis** — dispatch
           :class:`~.subagents.pattern_analyzer_agent.PatternAnalyzerAgent`
           with the scout results to narrow the search space.
        4. **Full tuning** — dispatch
           :class:`~.subagents.tuning_agent.TuningAgent` again with the
           narrowed search space over all shapes.
        5. **Config generation** — dispatch
           :class:`~.subagents.config_generator_agent.ConfigGeneratorAgent`
           to produce the final config files.

        Returns
        -------
        PhaseResult
            A :class:`PhaseResult` with ``phase=Phase.TUNING``.  On success
            ``data`` contains ``{"shapes_tuned": <int>, "configs_generated": True}``.
            When tuning is skipped ``data`` contains ``{"skipped": True}``.
        """
        import os as _os

        start_time = time.time()

        shapes_to_tune = self._determine_shapes_to_tune()

        if not shapes_to_tune:
            return PhaseResult(
                phase=Phase.TUNING,
                success=True,
                data={"skipped": True},
                error=None,
                duration_seconds=time.time() - start_time,
            )

        artifact_dir = self.artifact_manager.remote_dir
        scout_results_dir = _os.path.join(artifact_dir, "scout_results")
        tuning_logs_dir = _os.path.join(artifact_dir, "tuning_logs")
        config_dir = _os.path.join(artifact_dir, "configs")

        # Broad search space used for scouting.
        broad_search_space: Dict = {
            "M_LEQ_16": {
                "BLOCK_SIZE_M": [16, 32],
                "BLOCK_SIZE_N": [32, 64, 128],
                "BLOCK_SIZE_K": [32, 64],
                "num_stages": [1, 2, 3, 4],
                "matrix_instr_nonkdim": [16],
            },
            "M_LEQ_256": {
                "BLOCK_SIZE_M": [32, 64, 128],
                "BLOCK_SIZE_N": [32, 64, 128, 256],
                "BLOCK_SIZE_K": [32, 64, 128],
                "num_stages": [1, 2, 3, 4],
                "matrix_instr_nonkdim": [16],
            },
        }

        # 1. Scout phase: dispatch TuningAgent with a small subset of shapes.
        scout_agent_result = self._dispatch_subagent(
            TuningAgent,
            shapes_to_tune=shapes_to_tune[: max(1, len(shapes_to_tune) // 10)],
            search_space=broad_search_space,
            ut_script="ut_gemm.py",
            gpu_ids=self.config.gpu_ids,
            tunning_dir=artifact_dir,
        )
        if not scout_agent_result.success:
            return PhaseResult(
                phase=Phase.TUNING,
                success=False,
                data={},
                error=scout_agent_result.error,
                duration_seconds=time.time() - start_time,
            )

        # 2. Pattern analysis: dispatch PatternAnalyzerAgent.
        pattern_agent_result = self._dispatch_subagent(
            PatternAnalyzerAgent,
            scout_results_dir=scout_results_dir,
        )
        if not pattern_agent_result.success:
            return PhaseResult(
                phase=Phase.TUNING,
                success=False,
                data={},
                error=pattern_agent_result.error,
                duration_seconds=time.time() - start_time,
            )

        # Derive the narrowed search space from pattern analysis results.
        narrowed_search_space: Dict = pattern_agent_result.data.get(
            "search_space", broad_search_space
        )

        # 3. Full tuning: dispatch TuningAgent with all shapes and narrowed space.
        full_tuning_result = self._dispatch_subagent(
            TuningAgent,
            shapes_to_tune=shapes_to_tune,
            search_space=narrowed_search_space,
            ut_script="ut_gemm.py",
            gpu_ids=self.config.gpu_ids,
            tunning_dir=artifact_dir,
        )
        if not full_tuning_result.success:
            return PhaseResult(
                phase=Phase.TUNING,
                success=False,
                data={},
                error=full_tuning_result.error,
                duration_seconds=time.time() - start_time,
            )

        # 4. Config generation.
        # Extract ut_script from discovery phase results if available.
        discovery_phase_result = self.state.phase_results.get(Phase.DISCOVERY)
        ut_script: str = "ut_gemm.py"
        if discovery_phase_result is not None and discovery_phase_result.data:
            discovered_ut = discovery_phase_result.data.get("ut_script")
            if discovered_ut:
                ut_script = discovered_ut
        config_gen_result = self._dispatch_subagent(
            ConfigGeneratorAgent,
            tuning_logs_dir=tuning_logs_dir,
            config_dir=config_dir,
            kernel_variant=self.config.kernel_name,
            ut_script=ut_script,
            gfx_arch=self.config.gpu_arch or "gfx950",
        )
        if not config_gen_result.success:
            return PhaseResult(
                phase=Phase.TUNING,
                success=False,
                data={},
                error=config_gen_result.error,
                duration_seconds=time.time() - start_time,
            )

        return PhaseResult(
            phase=Phase.TUNING,
            success=True,
            data={
                "shapes_tuned": len(shapes_to_tune),
                "configs_generated": True,
            },
            error=None,
            duration_seconds=time.time() - start_time,
        )

    # ------------------------------------------------------------------
    # Phase 5 runner
    # ------------------------------------------------------------------

    def _run_phase_5_validation_and_fix(self) -> PhaseResult:
        """Execute Phase 5: validate tuned configs and fix regressions.

        Steps
        -----
        1. Dispatch :class:`~.subagents.validation_agent.ValidationAgent` to
           collect tuned timings.  Store results in
           :attr:`~SupervisorState.tuned_data`.
        2. Compare tuned results against both baseline and untuned data.
           Classify shapes as regressions vs baseline and/or untuned.
        3. For each regression, dispatch
           :class:`~.subagents.regression_fixer_agent.RegressionFixerAgent`.
        4. Revalidate after fixes.  Repeat the validate → fix → revalidate
           loop up to **3** iterations.
        5. If regressions still remain after 3 iterations, add an
           :class:`EscalationRequest` with ``severity="warning"``.

        Returns
        -------
        PhaseResult
            A :class:`PhaseResult` with ``phase=Phase.VALIDATION_AND_FIX``.
            ``data`` contains ``{"regression_count": <int>,
            "iterations": <int>}``.
        """
        import os as _os

        start_time = time.time()
        max_iterations = 3
        iteration = 0
        regression_count = 0

        artifact_dir = self.artifact_manager.remote_dir
        config_dir = _os.path.join(artifact_dir, "configs")
        old_config_dir = _os.path.join(artifact_dir, "baseline_configs")

        shapes = self.state.shapes or []

        try:
            # Convert list-of-dicts to the shape-keyed dict format expected
            # by ValidationAgent._classify(): {"M{m}_N{n}_K{k}": {...}, ...}
            def _to_shape_dict(data_list):
                if not data_list:
                    return None
                return {
                    f"M{r['m']}_N{r['n']}_K{r['k']}": r
                    for r in data_list
                }

            baseline_dict = _to_shape_dict(self.state.baseline_data)
            untuned_dict = _to_shape_dict(self.state.untuned_data)

            for iteration in range(1, max_iterations + 1):
                # --- Collect tuned timings ---
                validation_result = self._dispatch_with_retry(
                    ValidationAgent,
                    shapes=shapes,
                    bench_script="",
                    gpu_ids=self.config.gpu_ids,
                    kernel_variant=self.config.kernel_name,
                    baseline_data=baseline_dict,
                    untuned_data=untuned_dict,
                )

                if not validation_result.success:
                    return PhaseResult(
                        phase=Phase.VALIDATION_AND_FIX,
                        success=False,
                        data={"regression_count": regression_count, "iterations": iteration},
                        error=validation_result.error,
                        duration_seconds=time.time() - start_time,
                    )

                self.state.tuned_data = validation_result.data.get("results", [])

                # --- Classify regressions ---
                regressions: List = validation_result.data.get("regressions", [])
                regression_count = len(regressions)

                if regression_count == 0:
                    # No regressions — done.
                    break

                # --- Dispatch RegressionFixerAgent ---
                threshold = self.config.tuning_config.thresholds.regression_vs_baseline
                self._dispatch_subagent(
                    RegressionFixerAgent,
                    regressions=regressions,
                    config_dir=config_dir,
                    old_config_dir=old_config_dir,
                    threshold=threshold,
                )

                if iteration == max_iterations:
                    # Exhausted all fix iterations — escalate.
                    self.state.escalations.append(
                        EscalationRequest(
                            severity="warning",
                            message=(
                                f"Kernel '{self.config.kernel_name}': "
                                f"{regression_count} regression(s) persist after "
                                f"{max_iterations} fix iteration(s)."
                            ),
                        )
                    )

            return PhaseResult(
                phase=Phase.VALIDATION_AND_FIX,
                success=True,
                data={"regression_count": regression_count, "iterations": iteration},
                error=None,
                duration_seconds=time.time() - start_time,
            )
        except Exception as exc:  # noqa: BLE001
            return PhaseResult(
                phase=Phase.VALIDATION_AND_FIX,
                success=False,
                data={"regression_count": regression_count, "iterations": iteration},
                error=str(exc),
                duration_seconds=time.time() - start_time,
            )

    # ------------------------------------------------------------------
    # Phase 6 runner
    # ------------------------------------------------------------------

    def _run_phase_6_commit(self) -> PhaseResult:
        """Execute Phase 6: generate summary and commit configs with human approval.

        Steps
        -----
        1. Compute a summary: geomean speedup, regression count, shapes improved.
        2. Create an :class:`EscalationRequest` with
           ``severity="approval_required"`` and ask the human to confirm.
        3. Call :meth:`~.notifications.Notifier.request_approval`; block until
           a human approves or denies.
        4. If approved: run ``git add`` and ``git commit`` inside the container.
        5. If denied: return ``PhaseResult(success=False)``.

        Returns
        -------
        PhaseResult
            A :class:`PhaseResult` with ``phase=Phase.COMMIT``.  ``data``
            contains ``{"committed": True}`` on approval, or ``{}`` on denial.
        """
        import math as _math

        start_time = time.time()

        try:
            # --- Build summary statistics ---
            tuned_data = self.state.tuned_data or []
            baseline_data = self.state.baseline_data or []

            geomean_speedup: Optional[float] = None
            shapes_improved = 0
            regression_count = 0

            if tuned_data and baseline_data:
                baseline_map: Dict[Tuple[int, int, int], float] = {}
                for r in baseline_data:
                    if hasattr(r, "m"):
                        key = (r.m, r.n, r.k)
                        baseline_map[key] = r.total_ns
                    elif isinstance(r, dict):
                        key = (r.get("m", 0), r.get("n", 0), r.get("k", 0))
                        baseline_map[key] = r.get("total_ns", 1.0)

                speedups: List[float] = []
                for r in tuned_data:
                    if hasattr(r, "m"):
                        key = (r.m, r.n, r.k)
                        tuned_ns = r.total_ns
                    elif isinstance(r, dict):
                        key = (r.get("m", 0), r.get("n", 0), r.get("k", 0))
                        tuned_ns = r.get("total_ns", 1.0)
                    else:
                        continue

                    baseline_ns = baseline_map.get(key)
                    if baseline_ns and tuned_ns > 0:
                        speedup = baseline_ns / tuned_ns
                        speedups.append(speedup)
                        if speedup > 1.0:
                            shapes_improved += 1
                        elif speedup < 1.0:
                            regression_count += 1

                if speedups:
                    log_sum = sum(_math.log(s) for s in speedups)
                    geomean_speedup = _math.exp(log_sum / len(speedups))

            summary_text = (
                f"Kernel '{self.config.kernel_name}': "
                f"geomean_speedup={geomean_speedup:.4f}, "
                f"shapes_improved={shapes_improved}, "
                f"regressions={regression_count}."
            ) if geomean_speedup is not None else (
                f"Kernel '{self.config.kernel_name}': no comparison data available."
            )

            # --- Create approval escalation ---
            approval_request = EscalationRequest(
                severity="approval_required",
                message=f"Commit configs for {self.config.kernel_name}?",
                details=summary_text,
            )
            self.state.escalations.append(approval_request)

            # --- Request human approval ---
            approved = self.notifier.request_approval(
                question=f"Commit configs for {self.config.kernel_name}?",
                details=summary_text,
            )

            if not approved:
                return PhaseResult(
                    phase=Phase.COMMIT,
                    success=False,
                    data={"committed": False},
                    error="Commit denied by operator.",
                    duration_seconds=time.time() - start_time,
                )

            # --- Run git commit inside container ---
            commit_message = (
                f"feat(tuning): update {self.config.kernel_name} configs\n\n"
                f"{summary_text}"
            )
            self.executor.docker_exec(
                f"git add -A && git commit -m {commit_message!r}"
            )

            return PhaseResult(
                phase=Phase.COMMIT,
                success=True,
                data={
                    "committed": True,
                    "geomean_speedup": geomean_speedup,
                    "shapes_improved": shapes_improved,
                    "regression_count": regression_count,
                    "summary": summary_text,
                },
                error=None,
                duration_seconds=time.time() - start_time,
            )

        except Exception as exc:  # noqa: BLE001
            return PhaseResult(
                phase=Phase.COMMIT,
                success=False,
                data={},
                error=str(exc),
                duration_seconds=time.time() - start_time,
            )

    # ------------------------------------------------------------------
    # Main execution loop
    # ------------------------------------------------------------------

    def run(self) -> "SupervisorResult":
        """Run all pipeline phases from the resume point to COMMIT.

        The method honours checkpoint state: any phase already marked complete
        by the :class:`~.artifacts.ArtifactManager` is skipped.  On the first
        phase failure an :class:`EscalationRequest` is added and the loop
        terminates early.

        Returns
        -------
        SupervisorResult
            Final outcome including all :class:`PhaseResult` objects and any
            escalations raised during the run.
        """
        _phase_runners: Dict[Phase, Callable[[], PhaseResult]] = {
            Phase.SETUP: self._run_phase_0_setup,
            Phase.DISCOVERY: self._run_phase_1_discovery,
            Phase.BASELINE: self._run_phase_2_baseline,
            Phase.UNTUNED_VALIDATION: self._run_phase_3_untuned,
            Phase.TUNING: self._run_phase_4_tuning,
            Phase.VALIDATION_AND_FIX: self._run_phase_5_validation_and_fix,
            Phase.COMMIT: self._run_phase_6_commit,
        }

        resume_phase = self._get_resume_phase()
        overall_success = True

        for phase in Phase:
            if phase < resume_phase:
                continue

            if self._should_skip_phase(phase):
                continue

            if self.progress_callback is not None:
                self.progress_callback(phase, f"Starting phase {phase.name}")

            runner = _phase_runners[phase]
            result = runner()

            if result.success:
                self._record_phase_complete(phase, result)

            if not result.success:
                overall_success = False
                self.state.escalations.append(
                    EscalationRequest(
                        severity="fatal",
                        message=(
                            f"Phase {phase.name} failed for kernel "
                            f"'{self.config.kernel_name}': {result.error}"
                        ),
                    )
                )
                break

        # Build geomean speedup from the COMMIT phase result if available.
        geomean_speedup: Optional[float] = None
        commit_result = self.state.phase_results.get(Phase.COMMIT)
        if commit_result is not None and commit_result.data:
            geomean_speedup = commit_result.data.get("geomean_speedup")

        summary_parts: List[str] = []
        for phase, pr in self.state.phase_results.items():
            status = "OK" if pr.success else "FAIL"
            summary_parts.append(f"{phase.name}={status}")
        summary = ", ".join(summary_parts) if summary_parts else "no phases run"

        return SupervisorResult(
            kernel_name=self.config.kernel_name,
            success=overall_success,
            phase_results=dict(self.state.phase_results),
            escalations=list(self.state.escalations),
            geomean_speedup=geomean_speedup,
            summary=summary,
        )
