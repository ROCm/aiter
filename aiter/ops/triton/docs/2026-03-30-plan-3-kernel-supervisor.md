# Agentic Kernel Tuning Pipeline — Plan 3: Kernel Supervisor

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Kernel Supervisor — the per-kernel lifecycle manager that orchestrates subagents through phases 0-6 for a single kernel on a single allocated machine. Implements a state machine with checkpoint/resume, Triton version switching, error handling with retries, and progress reporting.

**Architecture:** A single module `kernel_supervisor.py` containing `KernelSupervisor` that owns the full tuning lifecycle. It dispatches subagents sequentially through phases, writes checkpoint markers (`phase_N_complete.json`), and can resume from the last completed phase after a crash. Two tuning modes: `regression_only` (only tune regressed shapes) and `full` (tune all shapes).

**Tech Stack:** Python 3.8+, pytest, unittest.mock

**Spec:** `/app/aiter/aiter/ops/triton/docs/2026-03-30-agentic-kernel-tuning-pipeline-design.md`

**Depends on:** Plan 1 (Infrastructure Layer), Plan 2 (Subagent Library)

**Enables:** Plan 4 (Orchestrator + UI)

---

## File Structure

```
aiter/ops/triton/tuning_agent/
├── kernel_supervisor.py          # KernelSupervisor state machine

tests/tuning_agent/
├── test_kernel_supervisor.py     # State, checkpoint, regression ID tests
├── test_phase_transitions.py    # Phase dispatch + full lifecycle tests
└── fixtures/
    ├── supervisor_discovery.json
    ├── supervisor_baseline.json
    └── supervisor_untuned.json
```

---

## Chunk 1: Types, State Machine, Checkpoint Logic

### Task 1: Phase Definitions, State, and Checkpoint/Resume

**Files:**
- Create: `aiter/ops/triton/tuning_agent/kernel_supervisor.py`
- Create: `tests/tuning_agent/test_kernel_supervisor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/tuning_agent/test_kernel_supervisor.py
import json, pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from aiter.ops.triton.tuning_agent.kernel_supervisor import (
    KernelSupervisor, Phase, SupervisorState, SupervisorConfig,
    PhaseResult, EscalationRequest, TuningMode,
)
from aiter.ops.triton.tuning_agent.types import (
    MachineInfo, RepoConfig, ContainerConfig, TritonInstallConfig,
    TuningConfig, TuningThresholds, TuningTimeouts,
)
from aiter.ops.triton.tuning_agent.remote import RemoteExecutor

FIXTURES_DIR = Path(__file__).parent / "fixtures"

@pytest.fixture
def machine():
    return MachineInfo(host="gpu1", user="root", ssh_key="~/.ssh/id", gpus=[0, 1, 2, 3])

@pytest.fixture
def executor(machine):
    ex = RemoteExecutor(machine); ex.container_id = "test_ctr"; return ex

@pytest.fixture
def supervisor_config():
    return SupervisorConfig(
        kernel_name="a8w8",
        baseline_repo=RepoConfig(aiter_repo="https://github.com/ROCm/aiter.git",
            aiter_branch="main", triton_repo="https://github.com/ROCm/triton.git",
            triton_branch="triton_3_4"),
        target_repo=RepoConfig(aiter_repo="https://github.com/ROCm/aiter.git",
            aiter_branch="feature/new", triton_repo="https://github.com/ROCm/triton.git",
            triton_branch="main"),
        container_config=ContainerConfig(image="rocm/pytorch:latest"),
        triton_install=TritonInstallConfig(command="pip install -e ."),
        tuning_config=TuningConfig(mode="regression_only", scout_fraction=0.15,
            thresholds=TuningThresholds(), timeouts=TuningTimeouts()),
        gfx_arch="gfx950", max_retries=2)

@pytest.fixture
def supervisor(executor, supervisor_config):
    return KernelSupervisor(executor=executor, config=supervisor_config)


class TestPhaseEnum:
    def test_phases_ordered_0_to_6(self):
        assert [p.value for p in Phase] == [0, 1, 2, 3, 4, 5, 6]

    def test_tuning_mode_from_string(self):
        assert TuningMode("regression_only") == TuningMode.REGRESSION_ONLY
        assert TuningMode("full") == TuningMode.FULL


class TestSupervisorState:
    def test_initial_state(self):
        s = SupervisorState(kernel_name="a8w8")
        assert s.current_phase == Phase.SETUP and s.completed_phases == []

    def test_advance_through_all_phases(self):
        s = SupervisorState(kernel_name="a8w8")
        for phase in Phase:
            s.mark_phase_complete(phase)
        assert s.is_complete and len(s.completed_phases) == 7

    def test_mark_failure(self):
        s = SupervisorState(kernel_name="a8w8")
        s.mark_phase_failed(Phase.BASELINE, "rocprof crashed")
        assert Phase.BASELINE in s.failed_phases and s.last_error == "rocprof crashed"


class TestPhaseResult:
    def test_success_and_failure(self):
        ok = PhaseResult(phase=Phase.SETUP, success=True, data={"id": "abc"})
        fail = PhaseResult(phase=Phase.BASELINE, success=False, error="timeout")
        assert ok.success and not fail.success

    def test_escalation_attached(self):
        esc = EscalationRequest(phase=Phase.COMMIT, reason="regressions",
                                severity="approval_required", data={"count": 3})
        r = PhaseResult(phase=Phase.COMMIT, success=True, escalation=esc)
        assert r.escalation.severity == "approval_required"


class TestCheckpointResume:
    @patch("subprocess.run")
    def test_write_checkpoint_creates_marker(self, mock_run, supervisor):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        supervisor._write_checkpoint(Phase.SETUP, {"container_id": "abc"})
        calls_str = " ".join(str(c) for c in mock_run.call_args_list)
        assert "phase_0_complete.json" in calls_str

    @patch("subprocess.run")
    def test_load_checkpoints_restores_state(self, mock_run, supervisor):
        def side_effect(*args, **kwargs):
            cmd = " ".join(args[0]) if isinstance(args[0], list) else str(args[0])
            if "ls" in cmd and "phase_" in cmd:
                return MagicMock(returncode=0, stdout="phase_0_complete.json\nphase_1_complete.json\n", stderr="")
            if "phase_0_complete" in cmd:
                return MagicMock(returncode=0, stdout=json.dumps({"phase":"SETUP","phase_value":0,"timestamp":0,"data":{}}), stderr="")
            if "phase_1_complete" in cmd:
                return MagicMock(returncode=0, stdout=json.dumps({"phase":"DISCOVERY","phase_value":1,"timestamp":0,"data":{}}), stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")
        mock_run.side_effect = side_effect
        supervisor._load_checkpoints()
        assert Phase.SETUP in supervisor.state.completed_phases
        assert supervisor.state.current_phase == Phase.BASELINE

    def test_always_rerun_phases(self, supervisor):
        """Phases 0, 1, 5, 6 always re-run; phases 2, 3, 4 do not."""
        assert supervisor._should_rerun_phase(Phase.SETUP) is True
        assert supervisor._should_rerun_phase(Phase.DISCOVERY) is True
        assert supervisor._should_rerun_phase(Phase.VALIDATION_REGRESSION) is True
        assert supervisor._should_rerun_phase(Phase.COMMIT) is True
        assert supervisor._should_rerun_phase(Phase.BASELINE) is False
        assert supervisor._should_rerun_phase(Phase.TUNING) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/test_kernel_supervisor.py -v`

- [ ] **Step 3: Implement types and state machine**

```python
# aiter/ops/triton/tuning_agent/kernel_supervisor.py
"""Kernel Supervisor — per-kernel lifecycle manager (phases 0-6)."""
import json, logging, time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .remote import RemoteExecutor, RemoteCommandError
from .types import (ContainerConfig, KernelOverrides, RepoConfig, TritonInstallConfig,
                    TuningConfig, TuningThresholds, TuningTimeouts)
from .subagents.base import SubagentResult

logger = logging.getLogger(__name__)

class Phase(Enum):
    SETUP = 0; DISCOVERY = 1; BASELINE = 2; UNTUNED_VALIDATION = 3
    TUNING = 4; VALIDATION_REGRESSION = 5; COMMIT = 6

class TuningMode(Enum):
    REGRESSION_ONLY = "regression_only"; FULL = "full"

ALWAYS_RERUN_PHASES = {Phase.SETUP, Phase.DISCOVERY, Phase.VALIDATION_REGRESSION, Phase.COMMIT}

@dataclass
class EscalationRequest:
    phase: Phase; reason: str; severity: str  # "warning"|"approval_required"|"fatal"
    data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PhaseResult:
    phase: Phase; success: bool; data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None; escalation: Optional[EscalationRequest] = None

@dataclass
class SupervisorConfig:
    kernel_name: str; baseline_repo: RepoConfig; target_repo: RepoConfig
    container_config: ContainerConfig; triton_install: TritonInstallConfig
    tuning_config: TuningConfig; gfx_arch: str = "gfx950"
    kernel_overrides: Optional[KernelOverrides] = None
    artifact_dir: str = "/workspace/tuning_artifacts"
    aiter_root: str = "/workspace/aiter"; max_retries: int = 2

@dataclass
class SupervisorState:
    kernel_name: str; current_phase: Phase = Phase.SETUP
    completed_phases: List[Phase] = field(default_factory=list)
    failed_phases: List[Phase] = field(default_factory=list)
    triton_version: str = "target"
    escalations: List[EscalationRequest] = field(default_factory=list)
    last_error: Optional[str] = None
    phase_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return len(self.completed_phases) == len(Phase)

    def mark_phase_complete(self, phase: Phase) -> None:
        if phase not in self.completed_phases:
            self.completed_phases.append(phase)
        nxt = phase.value + 1
        if nxt <= Phase.COMMIT.value:
            self.current_phase = Phase(nxt)

    def mark_phase_failed(self, phase: Phase, error: str) -> None:
        if phase not in self.failed_phases:
            self.failed_phases.append(phase)
        self.last_error = error

@dataclass
class SupervisorResult:
    kernel_name: str; success: bool
    phases_completed: List[Phase] = field(default_factory=list)
    phases_failed: List[Phase] = field(default_factory=list)
    escalations: List[EscalationRequest] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class KernelSupervisor:
    """Per-kernel lifecycle manager. Orchestrates subagents through phases 0-6."""

    def __init__(self, executor: RemoteExecutor, config: SupervisorConfig,
                 progress_callback: Optional[Callable[[str, Phase, str], None]] = None):
        self.executor = executor
        self.config = config
        self.state = SupervisorState(kernel_name=config.kernel_name)
        self.progress_callback = progress_callback
        self._artifact_dir = f"{config.artifact_dir}/{config.kernel_name}"
        self._tuning_mode = TuningMode(config.tuning_config.mode)

    def _report_progress(self, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(self.config.kernel_name, self.state.current_phase, message)
        logger.info(f"[{self.config.kernel_name}] {self.state.current_phase.name}: {message}")

    # --- Checkpoint / Resume ---

    def _write_checkpoint(self, phase: Phase, data: Dict[str, Any]) -> None:
        checkpoint = {"phase": phase.name, "phase_value": phase.value,
                      "timestamp": int(time.time()), "data": data}
        json_str = json.dumps(checkpoint, indent=2).replace("'", "'\\''")
        try:
            self.executor.docker_exec(
                f"printf '%s' '{json_str}' > {self._artifact_dir}/phase_{phase.value}_complete.json",
                check=True, timeout=15)
        except RemoteCommandError as e:
            logger.warning(f"Checkpoint write failed for {phase.name}: {e}")

    def _load_checkpoints(self) -> None:
        try:
            result = self.executor.docker_exec(
                f"ls {self._artifact_dir}/phase_*_complete.json 2>/dev/null || true",
                check=False, timeout=15)
        except RemoteCommandError:
            return
        if not result.stdout.strip():
            return
        max_completed = -1
        for fp in result.stdout.strip().split("\n"):
            if not fp.strip(): continue
            try:
                cat = self.executor.docker_exec(f"cat {fp.strip()}", check=True, timeout=15)
                ckpt = json.loads(cat.stdout)
                phase = Phase(ckpt["phase_value"])
                if phase not in self.state.completed_phases:
                    self.state.completed_phases.append(phase)
                self.state.phase_data[phase.name] = ckpt.get("data", {})
                max_completed = max(max_completed, phase.value)
            except Exception as e:
                logger.warning(f"Failed to read checkpoint {fp}: {e}")
        if 0 <= max_completed < Phase.COMMIT.value:
            self.state.current_phase = Phase(max_completed + 1)

    def _should_rerun_phase(self, phase: Phase) -> bool:
        return phase in ALWAYS_RERUN_PHASES
```

- [ ] **Step 4: Run tests to verify they pass**

- [ ] **Step 5: Commit**

```bash
git add aiter/ops/triton/tuning_agent/kernel_supervisor.py \
    tests/tuning_agent/test_kernel_supervisor.py
git commit -m "feat(tuning-agent): add KernelSupervisor types, state machine, checkpoint/resume"
```

---

## Chunk 2: Dispatch, Retry, Triton Switching

### Task 2: Subagent Dispatch + Retry + Triton Switching

**Files:**
- Update: `aiter/ops/triton/tuning_agent/kernel_supervisor.py`
- Create: `tests/tuning_agent/test_phase_transitions.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/tuning_agent/test_phase_transitions.py
import json, pytest
from unittest.mock import patch, MagicMock
from aiter.ops.triton.tuning_agent.kernel_supervisor import (
    KernelSupervisor, Phase, SupervisorConfig, PhaseResult, EscalationRequest, TuningMode,
)
from aiter.ops.triton.tuning_agent.subagents.base import SubagentResult
from aiter.ops.triton.tuning_agent.types import (
    MachineInfo, RepoConfig, ContainerConfig, TritonInstallConfig,
    TuningConfig, TuningThresholds, TuningTimeouts,
)
from aiter.ops.triton.tuning_agent.remote import RemoteExecutor

@pytest.fixture
def machine():
    return MachineInfo(host="gpu1", user="root", ssh_key="~/.ssh/id", gpus=[0, 1, 2, 3])

@pytest.fixture
def executor(machine):
    ex = RemoteExecutor(machine); ex.container_id = "test_ctr"; return ex

@pytest.fixture
def supervisor_config():
    return SupervisorConfig(
        kernel_name="a8w8",
        baseline_repo=RepoConfig(aiter_repo="u", aiter_branch="main",
            triton_repo="u", triton_branch="triton_3_4"),
        target_repo=RepoConfig(aiter_repo="u", aiter_branch="feat",
            triton_repo="u", triton_branch="main"),
        container_config=ContainerConfig(image="rocm/pytorch:latest"),
        triton_install=TritonInstallConfig(command="pip install -e ."),
        tuning_config=TuningConfig(mode="regression_only", scout_fraction=0.15,
            thresholds=TuningThresholds(), timeouts=TuningTimeouts()),
        gfx_arch="gfx950", max_retries=2)

@pytest.fixture
def supervisor(executor, supervisor_config):
    return KernelSupervisor(executor=executor, config=supervisor_config)


class TestDispatchSubagent:
    def test_wraps_success(self, supervisor):
        agent = MagicMock()
        agent.run.return_value = SubagentResult(success=True, data={"x": 1})
        r = supervisor._dispatch_subagent(agent, Phase.SETUP)
        assert r.success and r.data["x"] == 1

    def test_wraps_failure(self, supervisor):
        agent = MagicMock()
        agent.run.return_value = SubagentResult(success=False, error="died")
        r = supervisor._dispatch_subagent(agent, Phase.BASELINE)
        assert not r.success and "died" in r.error

    def test_catches_exception(self, supervisor):
        agent = MagicMock(); agent.run.side_effect = RuntimeError("lost")
        r = supervisor._dispatch_subagent(agent, Phase.TUNING)
        assert not r.success and "lost" in r.error


class TestRetryLogic:
    def test_succeeds_on_second_attempt(self, supervisor):
        attempt = {"n": 0}
        def factory():
            agent = MagicMock()
            agent.run.return_value = SubagentResult(
                success=(attempt["n"] > 0), data={"ok": True},
                error=None if attempt["n"] > 0 else "transient")
            attempt["n"] += 1; return agent
        r = supervisor._dispatch_with_retry(factory, Phase.BASELINE, max_retries=2)
        assert r.success and attempt["n"] == 2

    def test_exhausted_returns_failure(self, supervisor):
        factory = MagicMock()
        agent = MagicMock(); agent.run.return_value = SubagentResult(success=False, error="fail")
        factory.return_value = agent
        r = supervisor._dispatch_with_retry(factory, Phase.BASELINE, max_retries=2)
        assert not r.success and factory.call_count == 3


class TestTritonSwitching:
    @patch("subprocess.run")
    def test_switch_updates_state(self, mock_run, supervisor):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        supervisor._switch_triton("baseline")
        assert supervisor.state.triton_version == "baseline"

    @patch("subprocess.run")
    def test_noop_if_same_version(self, mock_run, supervisor):
        supervisor.state.triton_version = "target"
        supervisor._switch_triton("target")
        mock_run.assert_not_called()


class TestPhaseTimeout:
    @patch("time.time")
    def test_exceeded(self, mock_time, supervisor):
        mock_time.side_effect = [0, supervisor.config.tuning_config.timeouts.phase_max + 1]
        assert supervisor._check_phase_timeout(start_time=0) is True

    @patch("time.time")
    def test_within_limit(self, mock_time, supervisor):
        mock_time.return_value = 100
        assert supervisor._check_phase_timeout(start_time=0) is False
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement dispatch, retry, switching, timeout**

Add to `KernelSupervisor` class:

```python
    def _dispatch_subagent(self, agent: Any, phase: Phase) -> PhaseResult:
        try:
            result = agent.run()
            return PhaseResult(phase=phase, success=result.success,
                               data=result.data if result.success else {},
                               error=result.error if not result.success else None)
        except Exception as e:
            return PhaseResult(phase=phase, success=False, error=str(e))

    def _dispatch_with_retry(self, agent_factory: Callable[[], Any], phase: Phase,
                              max_retries: Optional[int] = None) -> PhaseResult:
        retries = max_retries if max_retries is not None else self.config.max_retries
        last = None
        for attempt in range(1 + retries):
            if attempt > 0:
                self._report_progress(f"Retry {attempt}/{retries}")
            last = self._dispatch_subagent(agent_factory(), phase)
            if last.success:
                return last
        return last

    def _switch_triton(self, version: str) -> None:
        if self.state.triton_version == version:
            return
        repo = self.config.baseline_repo if version == "baseline" else self.config.target_repo
        from .subagents.setup import SetupAgent
        agent = SetupAgent(executor=self.executor, kernel_name=self.config.kernel_name,
            artifact_dir=self._artifact_dir, repo_config=repo,
            container_config=self.config.container_config,
            triton_install=self.config.triton_install)
        result = agent.run()
        if not result.success:
            raise RuntimeError(f"Failed to switch to {version} Triton: {result.error}")
        self.state.triton_version = version

    def _check_phase_timeout(self, start_time: float) -> bool:
        return time.time() - start_time > self.config.tuning_config.timeouts.phase_max
```

- [ ] **Step 4: Run tests, verify pass**

- [ ] **Step 5: Commit**

```bash
git add aiter/ops/triton/tuning_agent/kernel_supervisor.py \
    tests/tuning_agent/test_phase_transitions.py
git commit -m "feat(tuning-agent): add dispatch, retry, Triton switching, timeout enforcement"
```

---

## Chunk 3: Phases 0-3

### Task 3: Phase Runners 0-3

**Files:**
- Update: `aiter/ops/triton/tuning_agent/kernel_supervisor.py`
- Update: `tests/tuning_agent/test_phase_transitions.py`

- [ ] **Step 1: Write failing tests for phases 0-3**

Append to `tests/tuning_agent/test_phase_transitions.py`:

```python
class TestPhase0:
    def test_dispatches_setup_and_checkpoints(self, supervisor):
        with patch.object(supervisor, "_dispatch_with_retry") as d, \
             patch.object(supervisor, "_write_checkpoint") as ck:
            d.return_value = PhaseResult(phase=Phase.SETUP, success=True, data={"id": "abc"})
            r = supervisor._run_phase_0_setup()
        assert r.success
        ck.assert_called_once()

    def test_failure_no_checkpoint(self, supervisor):
        with patch.object(supervisor, "_dispatch_with_retry") as d, \
             patch.object(supervisor, "_write_checkpoint") as ck:
            d.return_value = PhaseResult(phase=Phase.SETUP, success=False, error="docker fail")
            supervisor._run_phase_0_setup()
        ck.assert_not_called()


class TestPhase1:
    def test_stores_shapes_in_state(self, supervisor):
        with patch.object(supervisor, "_dispatch_with_retry") as d:
            d.return_value = PhaseResult(phase=Phase.DISCOVERY, success=True,
                data={"shapes": [(1,8192,8192)], "config_files": [],
                      "missing_scripts": [], "variant_prefix": "A8W8", "category": "basic"})
            supervisor._run_phase_1_discovery()
        assert "shapes" in supervisor.state.phase_data["DISCOVERY"]

    def test_dispatches_script_creator_if_missing(self, supervisor):
        calls = {"n": 0}
        def fake(factory, phase, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return PhaseResult(phase=phase, success=True,
                    data={"shapes": [], "config_files": [], "missing_scripts": ["ut.py"],
                          "variant_prefix": "A8W8", "category": "basic"})
            return PhaseResult(phase=phase, success=True, data={"created_scripts": ["ut.py"]})
        with patch.object(supervisor, "_dispatch_with_retry", side_effect=fake):
            supervisor._run_phase_1_discovery()
        assert calls["n"] == 2


class TestPhase2:
    def test_switches_baseline_then_target(self, supervisor):
        supervisor.state.phase_data["DISCOVERY"] = {"shapes": [(1,8192,8192)],
            "variant_prefix": "A8W8", "bench_script": "bench.py"}
        versions = []
        with patch.object(supervisor, "_switch_triton", side_effect=lambda v: versions.append(v)), \
             patch.object(supervisor, "_dispatch_with_retry") as d:
            d.return_value = PhaseResult(phase=Phase.BASELINE, success=True, data={"shapes": []})
            supervisor._run_phase_2_baseline()
        assert versions == ["baseline", "target"]


class TestPhase3:
    def test_uses_all_gpus(self, supervisor):
        supervisor.state.phase_data["DISCOVERY"] = {
            "shapes": [(m,8192,8192) for m in range(1,9)],
            "variant_prefix": "A8W8", "bench_script": "bench.py"}
        supervisor.state.phase_data["BASELINE"] = {"shapes": []}
        captured = {}
        def cap(factory, phase, **kw):
            agent = factory()
            captured["gpu_ids"] = getattr(agent, "gpu_ids", None)
            return PhaseResult(phase=phase, success=True, data={"results": []})
        with patch.object(supervisor, "_dispatch_with_retry", side_effect=cap):
            supervisor._run_phase_3_untuned_validation()
        assert captured["gpu_ids"] == [0, 1, 2, 3]
```

- [ ] **Step 2: Implement phases 0-3**

Add to `KernelSupervisor`:

```python
    def _run_phase_0_setup(self) -> PhaseResult:
        """Phase 0: Environment Setup. Dispatches SetupAgent with target repo."""
        from .subagents.setup import SetupAgent
        def factory():
            return SetupAgent(executor=self.executor, kernel_name=self.config.kernel_name,
                artifact_dir=self._artifact_dir, repo_config=self.config.target_repo,
                container_config=self.config.container_config,
                triton_install=self.config.triton_install)
        result = self._dispatch_with_retry(factory, Phase.SETUP)
        if result.success:
            self.state.triton_version = "target"
            self._write_checkpoint(Phase.SETUP, result.data)
            self.state.mark_phase_complete(Phase.SETUP)
        else:
            self.state.mark_phase_failed(Phase.SETUP, result.error)
        return result

    def _run_phase_1_discovery(self) -> PhaseResult:
        """Phase 1: Discovery + optional script creation."""
        from .subagents.discovery import DiscoveryAgent
        def discovery_factory():
            return DiscoveryAgent(executor=self.executor, kernel_name=self.config.kernel_name,
                artifact_dir=self._artifact_dir, aiter_root=self.config.aiter_root,
                gfx_arch=self.config.gfx_arch)
        result = self._dispatch_with_retry(discovery_factory, Phase.DISCOVERY)
        if not result.success:
            self.state.mark_phase_failed(Phase.DISCOVERY, result.error); return result
        self.state.phase_data["DISCOVERY"] = result.data

        missing = result.data.get("missing_scripts", [])
        if missing:
            from .subagents.script_creator import ScriptCreatorAgent
            def script_factory():
                return ScriptCreatorAgent(executor=self.executor,
                    kernel_name=self.config.kernel_name, artifact_dir=self._artifact_dir,
                    kernel_source_path=result.data.get("kernel_source_path", ""),
                    missing_scripts=missing,
                    template_scripts=result.data.get("template_scripts", {}),
                    category=result.data.get("category", "basic"))
            sr = self._dispatch_with_retry(script_factory, Phase.DISCOVERY)
            if sr.escalation:
                self.state.escalations.append(sr.escalation)

        self._write_checkpoint(Phase.DISCOVERY, result.data)
        self.state.mark_phase_complete(Phase.DISCOVERY)
        return result

    def _run_phase_2_baseline(self) -> PhaseResult:
        """Phase 2: Switch to baseline Triton, collect baselines, switch back."""
        discovery = self.state.phase_data.get("DISCOVERY", {})
        shapes = discovery.get("shapes", [])
        if not shapes:
            r = PhaseResult(phase=Phase.BASELINE, success=True, data={"shapes": []})
            self._write_checkpoint(Phase.BASELINE, r.data)
            self.state.mark_phase_complete(Phase.BASELINE); return r

        self._switch_triton("baseline")
        from .subagents.baseline import BaselineAgent
        def factory():
            return BaselineAgent(executor=self.executor, kernel_name=self.config.kernel_name,
                artifact_dir=self._artifact_dir, shapes=shapes,
                bench_script=discovery.get("bench_script", ""),
                gpu_id=self.executor.machine.gpus[0], output_filename="baseline.json",
                timeout_per_shape=self.config.tuning_config.timeouts.command_default)
        result = self._dispatch_with_retry(factory, Phase.BASELINE)
        self._switch_triton("target")

        if result.success:
            self.state.phase_data["BASELINE"] = result.data
            self._write_checkpoint(Phase.BASELINE, result.data)
            self.state.mark_phase_complete(Phase.BASELINE)
        else:
            self.state.mark_phase_failed(Phase.BASELINE, result.error)
        return result

    def _run_phase_3_untuned_validation(self) -> PhaseResult:
        """Phase 3: Collect on target Triton with old configs. Parallelizes across GPUs."""
        discovery = self.state.phase_data.get("DISCOVERY", {})
        shapes = discovery.get("shapes", [])
        if not shapes:
            r = PhaseResult(phase=Phase.UNTUNED_VALIDATION, success=True, data={"results": []})
            self._write_checkpoint(Phase.UNTUNED_VALIDATION, r.data)
            self.state.mark_phase_complete(Phase.UNTUNED_VALIDATION); return r

        from .subagents.validation import ValidationAgent
        def factory():
            return ValidationAgent(executor=self.executor, kernel_name=self.config.kernel_name,
                artifact_dir=self._artifact_dir, shapes=shapes,
                bench_script=discovery.get("bench_script", ""),
                gpu_ids=list(self.executor.machine.gpus),
                baseline_data=self.state.phase_data.get("BASELINE", {}),
                thresholds=self.config.tuning_config.thresholds,
                output_filename="untuned.json")
        result = self._dispatch_with_retry(factory, Phase.UNTUNED_VALIDATION)
        if result.success:
            self.state.phase_data["UNTUNED_VALIDATION"] = result.data
            self._write_checkpoint(Phase.UNTUNED_VALIDATION, result.data)
            self.state.mark_phase_complete(Phase.UNTUNED_VALIDATION)
        else:
            self.state.mark_phase_failed(Phase.UNTUNED_VALIDATION, result.error)
        return result
```

- [ ] **Step 3: Run tests, verify pass**

- [ ] **Step 4: Commit**

```bash
git add aiter/ops/triton/tuning_agent/kernel_supervisor.py \
    tests/tuning_agent/test_phase_transitions.py
git commit -m "feat(tuning-agent): implement Phases 0-3 (setup, discovery, baseline, untuned validation)"
```

---

## Chunk 4: Phase 4 — Tuning (Scout, Pattern, Full, Config Gen)

### Task 4: Phase 4 with Regression Identification

- [ ] **Step 1: Write failing tests**

Append to `tests/tuning_agent/test_phase_transitions.py`:

```python
class TestPhase4:
    def test_regression_only_identifies_regressed(self, supervisor):
        supervisor.state.phase_data["BASELINE"] = {"shapes": [
            {"m":1,"n":8192,"k":8192,"total_ns":5000},
            {"m":128,"n":8192,"k":8192,"total_ns":5000}]}
        supervisor.state.phase_data["UNTUNED_VALIDATION"] = {"results": [
            {"m":1,"n":8192,"k":8192,"total_ns":5500},    # 10% regression
            {"m":128,"n":8192,"k":8192,"total_ns":5100}]}  # 2% — within threshold
        regressed = supervisor._identify_regressed_shapes()
        assert (1,8192,8192) in regressed and (128,8192,8192) not in regressed

    def test_full_mode_tunes_all(self, supervisor):
        supervisor._tuning_mode = TuningMode.FULL
        supervisor.state.phase_data["DISCOVERY"] = {"shapes": [(1,8192,8192),(128,8192,8192)]}
        assert len(supervisor._determine_shapes_to_tune()) == 2

    def test_skips_if_no_regressions(self, supervisor):
        supervisor._tuning_mode = TuningMode.REGRESSION_ONLY
        supervisor.state.phase_data["DISCOVERY"] = {"shapes": [(1,8192,8192)],
            "ut_script": "ut.py", "variant_prefix": "A8W8"}
        supervisor.state.phase_data["BASELINE"] = {"shapes": [
            {"m":1,"n":8192,"k":8192,"total_ns":5000}]}
        supervisor.state.phase_data["UNTUNED_VALIDATION"] = {"results": [
            {"m":1,"n":8192,"k":8192,"total_ns":5050}]}  # 1%
        with patch.object(supervisor, "_dispatch_with_retry") as d:
            r = supervisor._run_phase_4_tuning()
        d.assert_not_called(); assert r.success

    def test_runs_4_subagents_in_full_mode(self, supervisor):
        supervisor._tuning_mode = TuningMode.FULL
        supervisor.state.phase_data["DISCOVERY"] = {
            "shapes": [(m,8192,8192) for m in range(1,9)],
            "variant_prefix": "A8W8", "bench_script": "b.py", "ut_script": "ut.py",
            "config_dir": "/cfg", "nk_pairs": [(8192,8192)], "category": "basic"}
        calls = {"n": 0}
        def track(factory, phase, **kw):
            calls["n"] += 1
            return PhaseResult(phase=phase, success=True,
                data={"tuned_shapes": [], "log_dir": "/tmp", "narrowed_search_space": {},
                      "config_files": []})
        with patch.object(supervisor, "_dispatch_with_retry", side_effect=track):
            supervisor._run_phase_4_tuning()
        assert calls["n"] == 4  # scout, pattern, full, config_gen
```

- [ ] **Step 2: Implement Phase 4**

Add to `KernelSupervisor`:

```python
    def _identify_regressed_shapes(self) -> List[Tuple[int, int, int]]:
        baseline = {(s["m"],s["n"],s["k"]): s["total_ns"]
                    for s in self.state.phase_data.get("BASELINE",{}).get("shapes",[])}
        untuned = {(r["m"],r["n"],r["k"]): r["total_ns"]
                   for r in self.state.phase_data.get("UNTUNED_VALIDATION",{}).get("results",[])}
        threshold = self.config.tuning_config.thresholds.regression_vs_baseline
        return [shape for shape, uns in untuned.items()
                if shape in baseline and baseline[shape] > 0
                and ((uns - baseline[shape]) / baseline[shape]) * 100 > threshold]

    def _determine_shapes_to_tune(self) -> List[Tuple[int, int, int]]:
        all_shapes = self.state.phase_data.get("DISCOVERY",{}).get("shapes",[])
        if self._tuning_mode == TuningMode.FULL:
            return [tuple(s) if isinstance(s, list) else s for s in all_shapes]
        return self._identify_regressed_shapes()

    def _run_phase_4_tuning(self) -> PhaseResult:
        """Phase 4: Scout → Pattern Analysis → Full Tuning → Config Generation."""
        shapes_to_tune = self._determine_shapes_to_tune()
        if not shapes_to_tune:
            r = PhaseResult(phase=Phase.TUNING, success=True,
                            data={"skipped": True, "reason": "no_regressions"})
            self._write_checkpoint(Phase.TUNING, r.data)
            self.state.mark_phase_complete(Phase.TUNING); return r

        disc = self.state.phase_data.get("DISCOVERY", {})
        ut_script = disc.get("ut_script", "")
        tuning_dir = f"{self.config.aiter_root}/aiter/ops/triton/utils/_triton/tunning"
        gpus = list(self.executor.machine.gpus)
        tc = self.config.tuning_config

        from .subagents.tuning import TuningAgent
        from .subagents.pattern_analyzer import PatternAnalyzerAgent
        from .subagents.config_generator import ConfigGeneratorAgent

        # Sub-phase 1: Scout
        def scout_f():
            return TuningAgent(executor=self.executor, kernel_name=self.config.kernel_name,
                artifact_dir=self._artifact_dir, shapes_to_tune=shapes_to_tune,
                ut_script=ut_script, gpu_ids=gpus, tuning_dir=tuning_dir,
                is_scout=True, scout_fraction=tc.scout_fraction,
                tuning_timeout_per_shape=tc.timeouts.tuning_per_shape)
        scout = self._dispatch_with_retry(scout_f, Phase.TUNING)
        if not scout.success:
            self.state.mark_phase_failed(Phase.TUNING, scout.error); return scout

        # Sub-phase 2: Pattern analysis
        def pattern_f():
            return PatternAnalyzerAgent(executor=self.executor,
                kernel_name=self.config.kernel_name, artifact_dir=self._artifact_dir,
                scout_log_dir=scout.data.get("log_dir", f"{self._artifact_dir}/scout_results"))
        pat = self._dispatch_with_retry(pattern_f, Phase.TUNING)
        narrowed = pat.data.get("narrowed_search_space") if pat.success else None

        # Sub-phase 3: Full tuning
        def full_f():
            return TuningAgent(executor=self.executor, kernel_name=self.config.kernel_name,
                artifact_dir=self._artifact_dir, shapes_to_tune=shapes_to_tune,
                ut_script=ut_script, gpu_ids=gpus, tuning_dir=tuning_dir,
                search_space=narrowed, is_scout=False,
                tuning_timeout_per_shape=tc.timeouts.tuning_per_shape)
        full = self._dispatch_with_retry(full_f, Phase.TUNING)
        if not full.success:
            self.state.mark_phase_failed(Phase.TUNING, full.error); return full

        # Sub-phase 4: Config generation
        m_leq = self.config.kernel_overrides.m_leq_16_bucket_name if self.config.kernel_overrides else None
        def cfg_f():
            return ConfigGeneratorAgent(executor=self.executor,
                kernel_name=self.config.kernel_name, artifact_dir=self._artifact_dir,
                tuning_log_dir=full.data.get("log_dir", tuning_dir), ut_script=ut_script,
                nk_pairs=disc.get("nk_pairs", []),
                config_dir=disc.get("config_dir",
                    f"{self.config.aiter_root}/aiter/ops/triton/configs/gemm"),
                variant_prefix=disc.get("variant_prefix", ""),
                gfx_arch=self.config.gfx_arch, m_leq_bucket_name=m_leq)
        cfg = self._dispatch_with_retry(cfg_f, Phase.TUNING)

        combined = {"shapes_tuned": len(shapes_to_tune), "scout": scout.data,
                    "patterns": pat.data if pat.success else {},
                    "tuning": full.data, "configs": cfg.data if cfg.success else {}}
        self.state.phase_data["TUNING"] = combined
        self._write_checkpoint(Phase.TUNING, combined)
        self.state.mark_phase_complete(Phase.TUNING)
        return PhaseResult(phase=Phase.TUNING, success=True, data=combined)
```

- [ ] **Step 3: Run tests, verify pass**

- [ ] **Step 4: Commit**

```bash
git add aiter/ops/triton/tuning_agent/kernel_supervisor.py \
    tests/tuning_agent/test_phase_transitions.py
git commit -m "feat(tuning-agent): implement Phase 4 (scout -> pattern -> full tuning -> config gen)"
```

---

## Chunk 5: Phases 5-6 and Main Loop

### Task 5: Phase 5 — Validation + Regression Fixing Loop

- [ ] **Step 1: Write failing tests**

Append to `tests/tuning_agent/test_phase_transitions.py`:

```python
class TestPhase5:
    def _setup_discovery(self, supervisor):
        supervisor.state.phase_data["DISCOVERY"] = {"shapes": [(1,8192,8192)],
            "variant_prefix": "A8W8", "bench_script": "b.py"}
        supervisor.state.phase_data["BASELINE"] = {"shapes": [
            {"m":1,"n":8192,"k":8192,"total_ns":5000}]}

    def test_no_regressions_skips_fixer(self, supervisor):
        self._setup_discovery(supervisor)
        with patch.object(supervisor, "_dispatch_with_retry") as d:
            d.return_value = PhaseResult(phase=Phase.VALIDATION_REGRESSION, success=True,
                data={"results": [], "regressions": [], "geomean_speedup": 1.05})
            r = supervisor._run_phase_5_validation_regression()
        assert r.success and d.call_count == 1

    def test_dispatches_fixer_on_regressions(self, supervisor):
        self._setup_discovery(supervisor)
        calls = {"n": 0}
        def fake(factory, phase, **kw):
            calls["n"] += 1
            if calls["n"] == 1:  # validation
                return PhaseResult(phase=phase, success=True, data={
                    "results": [], "regressions": [{"m":1,"n":8192,"k":8192}],
                    "geomean_speedup": 0.98})
            elif calls["n"] == 2:  # fixer
                return PhaseResult(phase=phase, success=True, data={
                    "fixed": [], "promoted": [], "escalated": [],
                    "noise_skipped": [], "modified_nk_pairs": []})
            else:  # re-validation
                return PhaseResult(phase=phase, success=True, data={
                    "results": [], "regressions": [], "geomean_speedup": 1.0})
        with patch.object(supervisor, "_dispatch_with_retry", side_effect=fake):
            supervisor._run_phase_5_validation_regression()
        assert calls["n"] == 3

    def test_escalates_after_max_iterations(self, supervisor):
        self._setup_discovery(supervisor)
        def always_regressed(factory, phase, **kw):
            return PhaseResult(phase=phase, success=True, data={
                "results": [], "regressions": [{"m":1}], "geomean_speedup": 0.95,
                "fixed": [], "promoted": [], "escalated": [],
                "noise_skipped": [], "modified_nk_pairs": [[8192,8192]]})
        with patch.object(supervisor, "_dispatch_with_retry", side_effect=always_regressed):
            supervisor._run_phase_5_validation_regression()
        assert len(supervisor.state.escalations) > 0
```

- [ ] **Step 2: Implement Phase 5**

Add to `KernelSupervisor`:

```python
    MAX_REGRESSION_FIX_ITERATIONS = 3

    def _run_phase_5_validation_regression(self) -> PhaseResult:
        """Phase 5: Validate → fix regressions → re-validate, up to 3 iterations."""
        disc = self.state.phase_data.get("DISCOVERY", {})
        shapes = disc.get("shapes", [])
        from .subagents.validation import ValidationAgent
        from .subagents.regression_fixer import RegressionFixerAgent

        for iteration in range(self.MAX_REGRESSION_FIX_ITERATIONS + 1):
            def val_f():
                return ValidationAgent(executor=self.executor,
                    kernel_name=self.config.kernel_name, artifact_dir=self._artifact_dir,
                    shapes=shapes, bench_script=disc.get("bench_script", ""),
                    gpu_ids=list(self.executor.machine.gpus),
                    baseline_data=self.state.phase_data.get("BASELINE", {}),
                    thresholds=self.config.tuning_config.thresholds,
                    untuned_data=self.state.phase_data.get("UNTUNED_VALIDATION", {}),
                    output_filename="tuned.json")
            vr = self._dispatch_with_retry(val_f, Phase.VALIDATION_REGRESSION)
            if not vr.success:
                self.state.mark_phase_failed(Phase.VALIDATION_REGRESSION, vr.error); return vr

            regressions = vr.data.get("regressions", [])
            if not regressions:
                self.state.phase_data["VALIDATION_REGRESSION"] = vr.data
                self._write_checkpoint(Phase.VALIDATION_REGRESSION, vr.data)
                self.state.mark_phase_complete(Phase.VALIDATION_REGRESSION); return vr

            if iteration == self.MAX_REGRESSION_FIX_ITERATIONS:
                esc = EscalationRequest(phase=Phase.VALIDATION_REGRESSION,
                    reason=f"{len(regressions)} regressions persist after {self.MAX_REGRESSION_FIX_ITERATIONS} iterations",
                    severity="approval_required", data={"regressions": regressions})
                self.state.escalations.append(esc)
                self.state.phase_data["VALIDATION_REGRESSION"] = vr.data
                self._write_checkpoint(Phase.VALIDATION_REGRESSION, vr.data)
                self.state.mark_phase_complete(Phase.VALIDATION_REGRESSION)
                return PhaseResult(phase=Phase.VALIDATION_REGRESSION, success=True,
                                   data=vr.data, escalation=esc)

            config_dir = disc.get("config_dir",
                f"{self.config.aiter_root}/aiter/ops/triton/configs/gemm")
            def fix_f():
                return RegressionFixerAgent(executor=self.executor,
                    kernel_name=self.config.kernel_name, artifact_dir=self._artifact_dir,
                    regressions=regressions, config_dir=config_dir,
                    gfx_arch=self.config.gfx_arch,
                    variant_prefix=disc.get("variant_prefix", ""))
            fr = self._dispatch_with_retry(fix_f, Phase.VALIDATION_REGRESSION)
            if not fr.success:
                self.state.mark_phase_failed(Phase.VALIDATION_REGRESSION, fr.error); return fr

        return vr
```

- [ ] **Step 3: Run tests, verify pass**

- [ ] **Step 4: Commit**

---

### Task 6: Phase 6 — Commit + Main run() Loop

- [ ] **Step 1: Write failing tests**

Append to `tests/tuning_agent/test_phase_transitions.py`:

```python
class TestPhase6:
    def test_generates_summary_and_escalates(self, supervisor):
        supervisor.state.phase_data["VALIDATION_REGRESSION"] = {
            "results": [{"m":1,"n":8192,"k":8192,"classification":"improved"}],
            "regressions": [], "geomean_speedup": 1.04}
        supervisor.state.phase_data["TUNING"] = {"configs": {"config_files": ["f.json"]}}
        with patch.object(supervisor.executor, "docker_exec",
                         return_value=MagicMock(returncode=0)):
            r = supervisor._run_phase_6_commit()
        assert r.success and "summary" in r.data
        assert any(e.severity == "approval_required" for e in supervisor.state.escalations)


class TestMainLoop:
    def _mock_all_phases(self, supervisor, phases_executed):
        def make(name):
            def fn(*a, **kw):
                phases_executed.append(name)
                return PhaseResult(phase=Phase.SETUP, success=True, data={
                    "shapes": [], "results": [], "regressions": [],
                    "geomean_speedup": 1.0, "summary": {}, "configs": {"config_files": []},
                    "variant_prefix": "A8W8", "bench_script": "", "ut_script": ""})
            return fn
        for i, p in enumerate(["_run_phase_0_setup", "_run_phase_1_discovery",
                "_run_phase_2_baseline", "_run_phase_3_untuned_validation",
                "_run_phase_4_tuning", "_run_phase_5_validation_regression",
                "_run_phase_6_commit"]):
            setattr(supervisor, p, make(f"phase_{i}"))

    def test_executes_all_phases(self, supervisor):
        phases = []
        self._mock_all_phases(supervisor, phases)
        with patch.object(supervisor, "_load_checkpoints"): supervisor.run()
        assert phases == [f"phase_{i}" for i in range(7)]

    def test_stops_on_failure(self, supervisor):
        phases = []
        self._mock_all_phases(supervisor, phases)
        def fail(*a, **kw):
            phases.append("phase_1")
            return PhaseResult(phase=Phase.DISCOVERY, success=False, error="not found")
        supervisor._run_phase_1_discovery = fail
        with patch.object(supervisor, "_load_checkpoints"):
            r = supervisor.run()
        assert not r.success and "phase_2" not in phases

    def test_resumes_from_checkpoint(self, supervisor):
        supervisor.state.completed_phases = [Phase.SETUP, Phase.DISCOVERY, Phase.BASELINE]
        supervisor.state.current_phase = Phase.UNTUNED_VALIDATION
        supervisor.state.phase_data.update({"DISCOVERY": {"shapes": []}, "BASELINE": {"shapes": []}})
        phases = []
        self._mock_all_phases(supervisor, phases)
        with patch.object(supervisor, "_load_checkpoints"): supervisor.run()
        # 0,1 always rerun; 2 skipped (checkpointed); 3-6 run
        assert "phase_0" in phases and "phase_1" in phases
        assert "phase_2" not in phases
        assert "phase_3" in phases
```

- [ ] **Step 2: Implement Phase 6 and run()**

Add to `KernelSupervisor`:

```python
    def _run_phase_6_commit(self) -> PhaseResult:
        """Phase 6: Generate summary, escalate for human approval."""
        val = self.state.phase_data.get("VALIDATION_REGRESSION", {})
        tuning = self.state.phase_data.get("TUNING", {})
        results = val.get("results", [])
        regressions = val.get("regressions", [])
        summary = {
            "kernel_name": self.config.kernel_name,
            "geomean_speedup_vs_baseline": val.get("geomean_speedup", 1.0),
            "total_shapes": len(results),
            "improved_count": sum(1 for r in results if r.get("classification") == "improved"),
            "regression_count": len(regressions),
            "config_files_modified": tuning.get("configs", {}).get("config_files", []),
            "regressions": regressions,
        }
        try:
            js = json.dumps(summary, indent=2).replace("'", "'\\''")
            self.executor.docker_exec(
                f"printf '%s' '{js}' > {self._artifact_dir}/summary.json",
                check=True, timeout=15)
        except RemoteCommandError:
            pass
        esc = EscalationRequest(phase=Phase.COMMIT,
            reason=f"Kernel {self.config.kernel_name} ready for commit",
            severity="approval_required", data=summary)
        self.state.escalations.append(esc)
        data = {"summary": summary, "awaiting_approval": True}
        self._write_checkpoint(Phase.COMMIT, data)
        self.state.mark_phase_complete(Phase.COMMIT)
        return PhaseResult(phase=Phase.COMMIT, success=True, data=data, escalation=esc)

    _PHASE_RUNNERS = {
        Phase.SETUP: "_run_phase_0_setup",
        Phase.DISCOVERY: "_run_phase_1_discovery",
        Phase.BASELINE: "_run_phase_2_baseline",
        Phase.UNTUNED_VALIDATION: "_run_phase_3_untuned_validation",
        Phase.TUNING: "_run_phase_4_tuning",
        Phase.VALIDATION_REGRESSION: "_run_phase_5_validation_regression",
        Phase.COMMIT: "_run_phase_6_commit",
    }

    def run(self) -> SupervisorResult:
        """Execute the full tuning lifecycle (phases 0-6) with checkpoint resume."""
        self._load_checkpoints()
        for phase in Phase:
            if phase in self.state.completed_phases and not self._should_rerun_phase(phase):
                continue
            runner = getattr(self, self._PHASE_RUNNERS[phase])
            try:
                result = runner()
            except Exception as e:
                result = PhaseResult(phase=phase, success=False, error=str(e))
            if not result.success:
                return SupervisorResult(kernel_name=self.config.kernel_name, success=False,
                    phases_completed=list(self.state.completed_phases),
                    phases_failed=list(self.state.failed_phases),
                    escalations=list(self.state.escalations), error=result.error)
            if result.escalation and result.escalation not in self.state.escalations:
                self.state.escalations.append(result.escalation)
        return SupervisorResult(kernel_name=self.config.kernel_name, success=True,
            phases_completed=list(self.state.completed_phases),
            escalations=list(self.state.escalations),
            summary=self.state.phase_data.get("VALIDATION_REGRESSION", {}))
```

- [ ] **Step 3: Run tests, verify pass**

- [ ] **Step 4: Commit**

```bash
git add aiter/ops/triton/tuning_agent/kernel_supervisor.py \
    tests/tuning_agent/test_phase_transitions.py
git commit -m "feat(tuning-agent): implement Phase 5-6 and main run() loop"
```

---

## Chunk 6: Test Fixtures and Package Export

### Task 7: Fixture Files + Integration Tests

**Files:**
- Create: `tests/tuning_agent/fixtures/supervisor_discovery.json`
- Create: `tests/tuning_agent/fixtures/supervisor_baseline.json`
- Create: `tests/tuning_agent/fixtures/supervisor_untuned.json`

- [ ] **Step 1: Create fixtures**

`supervisor_discovery.json`:
```json
{"shapes":[[1,8192,8192],[2,8192,8192],[4,8192,8192],[8,8192,8192],[16,8192,8192],[32,8192,8192],[64,8192,8192],[128,8192,8192]],"config_files":["gfx950-GEMM-A8W8.json"],"missing_scripts":[],"variant_prefix":"A8W8","category":"basic","bench_script":"bench_gemm_a8w8.py","ut_script":"ut_gemm_a8w8.py","nk_pairs":[[8192,8192]]}
```

`supervisor_baseline.json`:
```json
{"kernel":"a8w8","shapes":[{"m":1,"n":8192,"k":8192,"main_ns":4500,"reduce_ns":1000,"total_ns":5500},{"m":2,"n":8192,"k":8192,"total_ns":5600},{"m":4,"n":8192,"k":8192,"total_ns":5700},{"m":8,"n":8192,"k":8192,"total_ns":6000},{"m":16,"n":8192,"k":8192,"total_ns":7000},{"m":32,"n":8192,"k":8192,"total_ns":8000},{"m":64,"n":8192,"k":8192,"total_ns":12000},{"m":128,"n":8192,"k":8192,"total_ns":20000}]}
```

`supervisor_untuned.json`:
```json
{"results":[{"m":1,"n":8192,"k":8192,"total_ns":5800},{"m":2,"n":8192,"k":8192,"total_ns":6200},{"m":4,"n":8192,"k":8192,"total_ns":5900},{"m":8,"n":8192,"k":8192,"total_ns":6300},{"m":16,"n":8192,"k":8192,"total_ns":7200},{"m":32,"n":8192,"k":8192,"total_ns":9500},{"m":64,"n":8192,"k":8192,"total_ns":12500},{"m":128,"n":8192,"k":8192,"total_ns":21000}]}
```

- [ ] **Step 2: Write integration tests using fixtures**

Append to `tests/tuning_agent/test_kernel_supervisor.py`:

```python
class TestRegressionIdentification:
    def test_from_fixture_data(self, supervisor):
        baseline = json.loads((FIXTURES_DIR / "supervisor_baseline.json").read_text())
        untuned = json.loads((FIXTURES_DIR / "supervisor_untuned.json").read_text())
        supervisor.state.phase_data["BASELINE"] = baseline
        supervisor.state.phase_data["UNTUNED_VALIDATION"] = untuned
        regressed = supervisor._identify_regressed_shapes()
        # m=2: 5600->6200=10.7%, m=32: 8000->9500=18.75%, m=1: 5500->5800=5.45%
        assert (2,8192,8192) in regressed and (32,8192,8192) in regressed
        assert (1,8192,8192) in regressed  # 5.45% > 5% threshold
        assert (64,8192,8192) not in regressed  # 4.17% within threshold

    def test_full_mode_tunes_all_from_fixtures(self, supervisor):
        disc = json.loads((FIXTURES_DIR / "supervisor_discovery.json").read_text())
        supervisor.state.phase_data["DISCOVERY"] = disc
        supervisor._tuning_mode = TuningMode.FULL
        assert len(supervisor._determine_shapes_to_tune()) == 8
```

- [ ] **Step 3: Commit**

```bash
git add tests/tuning_agent/fixtures/supervisor_*.json \
    tests/tuning_agent/test_kernel_supervisor.py
git commit -m "test(tuning-agent): add fixture-based integration tests for regression identification"
```

---

### Task 8: Package Export

- [ ] **Step 1: Update package exports**

Add to `aiter/ops/triton/tuning_agent/__init__.py`:

```python
from .kernel_supervisor import (
    KernelSupervisor, SupervisorConfig, SupervisorResult, SupervisorState,
    Phase, TuningMode, PhaseResult, EscalationRequest,
)
```

- [ ] **Step 2: Commit**

```bash
git add aiter/ops/triton/tuning_agent/__init__.py
git commit -m "feat(tuning-agent): export KernelSupervisor from package"
```

---

## Summary

| Chunk | Tasks | Delivers |
|-------|-------|---------|
| 1 | 1 | Phase enum, SupervisorState, checkpoint/resume, types |
| 2 | 2 | Subagent dispatch, retry, Triton switching, timeout |
| 3 | 3 | Phases 0-3 (setup, discovery, baseline, untuned validation) |
| 4 | 4 | Phase 4 (scout -> pattern -> full tuning -> config gen) |
| 5 | 5-6 | Phases 5-6 (validation/regression loop, commit), run() |
| 6 | 7-8 | Test fixtures, integration tests, package export |

**Key design decisions:**

1. **Checkpoint markers** (`phase_N_complete.json`) enable resume after crash. Phases 0, 1, 5, 6 always re-run; phases 2-4 rely on subagent-internal restartability (skip already-collected shapes).

2. **Factory-based retry** — `_dispatch_with_retry(factory, phase)` creates a fresh subagent per attempt. Subagents read their own partial output to skip completed work.

3. **Triton version switching** — Phase 2 brackets with `_switch_triton("baseline")` / `_switch_triton("target")`, using SetupAgent in install-only mode.

4. **Two tuning modes** — `regression_only` compares Phase 3 vs Phase 2 and only tunes shapes regressed beyond `regression_vs_baseline`. `full` tunes all discovered shapes.

5. **Iterative regression fixing** — Phase 5 loops validate -> fix -> re-validate up to 3 times, then escalates remaining regressions.

6. **Progress callback** — Orchestrator passes `progress_callback` for dashboard updates without polling.

7. **Escalation system** — `EscalationRequest` with severity levels (`warning`, `approval_required`, `fatal`) collected in `state.escalations`. Phase 6 always escalates for commit approval.
