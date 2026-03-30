"""KernelSupervisor core types, state machine, and checkpoint logic.

This module provides the foundation for the agentic Triton kernel tuning
supervisor. It defines the Phase state machine, configuration, state tracking,
and checkpoint/resume logic.  Phase runners and the main execution loop are
added by subsequent tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from enum import Enum
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

from .types import ContainerConfig, RepoConfig, TritonInstallConfig, TuningConfig
from .remote import RemoteExecutor
from .artifacts import ArtifactManager
from .notifications import Notifier

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
