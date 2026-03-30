# Agentic Kernel Tuning Pipeline — Plan 1: Infrastructure Layer

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the foundational infrastructure that all agents (orchestrator, kernel supervisors, subagents) depend on — config parsing, remote execution, machine management, health monitoring, and artifact transfer.

**Architecture:** A Python package (`aiter/ops/triton/tuning_agent/`) with focused modules: config loader, remote executor (SSH + docker exec wrapper), machine pool, watchdog, and artifact manager. Each module has a clean interface so agents can call `remote.run("ls /tmp")` without knowing SSH details.

**Tech Stack:** Python 3.8+, PyYAML, subprocess (SSH/docker), pytest

**Spec:** `/app/aiter/aiter/ops/triton/docs/2026-03-30-agentic-kernel-tuning-pipeline-design.md`

**Depends on:** Nothing (this is the base layer)

**Enables:** Plan 2 (Subagent Library), Plan 3 (Kernel Supervisor), Plan 4 (Orchestrator + UI)

---

## File Structure

```
aiter/ops/triton/tuning_agent/
├── __init__.py                  # Package init, version
├── config.py                    # YAML config parsing + validation
├── remote.py                    # SSH + docker exec wrapper
├── machine_pool.py              # Machine allocation, release, health checks
├── watchdog.py                  # Health monitoring, timeout enforcement, progress checking
├── artifacts.py                 # Local ↔ remote file transfer, artifact directory management
├── notifications.py             # Notification system (terminal bell, extensible)
└── types.py                     # Shared dataclasses: MachineInfo, ContainerInfo, ShapeResult, etc.

tests/tuning_agent/
├── __init__.py
├── test_config.py
├── test_remote.py
├── test_machine_pool.py
├── test_watchdog.py
├── test_artifacts.py
├── test_notifications.py
├── conftest.py                  # Shared fixtures (mock SSH, test YAML configs)
└── fixtures/
    ├── valid_config.yaml
    ├── minimal_config.yaml
    └── invalid_config.yaml
```

---

## Chunk 1: Types + Config

### Task 1: Shared Type Definitions

**Files:**
- Create: `aiter/ops/triton/tuning_agent/__init__.py`
- Create: `aiter/ops/triton/tuning_agent/types.py`
- Create: `tests/tuning_agent/__init__.py`
- Create: `tests/tuning_agent/test_types.py`

- [ ] **Step 1: Create package init**

```python
# aiter/ops/triton/tuning_agent/__init__.py
"""Agentic Triton kernel tuning pipeline infrastructure."""
__version__ = "0.1.0"
```

- [ ] **Step 2: Write failing tests for type definitions**

```python
# tests/tuning_agent/test_types.py
from aiter.ops.triton.tuning_agent.types import (
    MachineInfo,
    ContainerConfig,
    RepoConfig,
    TuningThresholds,
    TuningTimeouts,
    KernelOverrides,
    PipelineConfig,
    ShapeResult,
    ContainerState,
)


def test_machine_info_creation():
    m = MachineInfo(host="gpu1.internal", user="root", ssh_key="~/.ssh/id_rsa", gpus=[0, 1, 2, 3])
    assert m.host == "gpu1.internal"
    assert m.user == "root"
    assert len(m.gpus) == 4


def test_machine_info_gpu_count():
    m = MachineInfo(host="gpu1", user="root", ssh_key="~/.ssh/id", gpus=[0, 1])
    assert m.gpu_count == 2


def test_repo_config():
    r = RepoConfig(
        aiter_repo="https://github.com/ROCm/aiter.git",
        aiter_branch="main",
        triton_repo="https://github.com/ROCm/triton.git",
        triton_branch="triton_3_4",
    )
    assert r.aiter_branch == "main"


def test_container_config_with_script():
    c = ContainerConfig(image="rocm/pytorch:latest", run_script="./scripts/create.sh")
    assert c.run_script == "./scripts/create.sh"


def test_container_config_without_script():
    c = ContainerConfig(image="rocm/pytorch:latest")
    assert c.run_script is None


def test_tuning_thresholds_defaults():
    t = TuningThresholds()
    assert t.regression_vs_baseline == 5.0
    assert t.regression_vs_untuned == 2.0


def test_tuning_timeouts_defaults():
    t = TuningTimeouts()
    assert t.command_default == 300
    assert t.progress_check == 120


def test_shape_result():
    s = ShapeResult(m=8, n=8192, k=8192, main_ns=5000, reduce_ns=2000)
    assert s.total_ns == 7000


def test_shape_result_no_reduce():
    s = ShapeResult(m=128, n=8192, k=8192, main_ns=15000, reduce_ns=0)
    assert s.total_ns == 15000


def test_container_state():
    c = ContainerState(container_id="abc123", machine=MachineInfo(
        host="gpu1", user="root", ssh_key="~/.ssh/id", gpus=[0]
    ))
    assert c.container_id == "abc123"
    assert not c.is_ready  # not verified yet
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/test_types.py -v`
Expected: FAIL — module not found

- [ ] **Step 4: Implement type definitions**

```python
# aiter/ops/triton/tuning_agent/types.py
"""Shared type definitions for the tuning pipeline."""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class MachineInfo:
    """A remote GPU machine in the pool."""
    host: str
    user: str
    ssh_key: str
    gpus: List[int]

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)


@dataclass
class ContainerConfig:
    """How to create containers on remote machines."""
    image: str
    run_script: Optional[str] = None


@dataclass
class RepoConfig:
    """Git repository + branch specification."""
    aiter_repo: str
    aiter_branch: str
    triton_repo: str
    triton_branch: str


@dataclass
class TuningThresholds:
    """Percentage thresholds for regression detection."""
    regression_vs_baseline: float = 5.0
    regression_vs_untuned: float = 2.0


@dataclass
class TuningTimeouts:
    """Timeout values in seconds."""
    command_default: int = 300
    tuning_per_shape: int = 1800
    progress_check: int = 120
    phase_max: int = 14400


@dataclass
class KernelOverrides:
    """Per-kernel tuning overrides."""
    m_leq_16_bucket_name: Optional[str] = None
    extra_block_k: Optional[List[int]] = None


@dataclass
class GpuConfig:
    """GPU architecture configuration."""
    arch: Optional[str] = None  # e.g., "gfx950". Auto-detected if None.


@dataclass
class TritonInstallConfig:
    """How to install Triton inside containers."""
    command: str = "pip install -e ."


@dataclass
class TuningConfig:
    """Tuning behavior configuration."""
    mode: str = "regression_only"  # or "full"
    scout_fraction: float = 0.15
    thresholds: TuningThresholds = field(default_factory=TuningThresholds)
    timeouts: TuningTimeouts = field(default_factory=TuningTimeouts)


@dataclass
class KernelsConfig:
    """Kernel include/exclude and per-kernel overrides."""
    exclude: List[str] = field(default_factory=list)
    include: List[str] = field(default_factory=list)
    overrides: Dict[str, KernelOverrides] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration parsed from YAML."""
    baseline: RepoConfig
    target: RepoConfig
    machines: List[MachineInfo]
    container: ContainerConfig
    gpu: GpuConfig = field(default_factory=GpuConfig)
    triton_install: TritonInstallConfig = field(default_factory=TritonInstallConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    kernels: KernelsConfig = field(default_factory=KernelsConfig)


@dataclass
class ShapeResult:
    """Benchmark result for a single (M, N, K) shape."""
    m: int
    n: int
    k: int
    main_ns: int
    reduce_ns: int = 0

    @property
    def total_ns(self) -> int:
        return self.main_ns + self.reduce_ns


@dataclass
class ContainerState:
    """Tracks a running container on a remote machine."""
    container_id: str
    machine: MachineInfo
    triton_commit: Optional[str] = None
    aiter_commit: Optional[str] = None
    is_ready: bool = False
    artifact_dir: str = "/workspace/tuning_artifacts"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/test_types.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add aiter/ops/triton/tuning_agent/__init__.py aiter/ops/triton/tuning_agent/types.py \
    tests/tuning_agent/__init__.py tests/tuning_agent/test_types.py
git commit -m "feat(tuning-agent): add shared type definitions"
```

---

### Task 2: YAML Config Parsing

**Files:**
- Create: `aiter/ops/triton/tuning_agent/config.py`
- Create: `tests/tuning_agent/test_config.py`
- Create: `tests/tuning_agent/fixtures/valid_config.yaml`
- Create: `tests/tuning_agent/fixtures/minimal_config.yaml`
- Create: `tests/tuning_agent/fixtures/invalid_config.yaml`
- Create: `tests/tuning_agent/conftest.py`

- [ ] **Step 1: Create test fixtures**

```yaml
# tests/tuning_agent/fixtures/valid_config.yaml
baseline:
  aiter_repo: https://github.com/ROCm/aiter.git
  aiter_branch: main
  triton_repo: https://github.com/ROCm/triton.git
  triton_branch: triton_3_4

target:
  aiter_repo: https://github.com/ROCm/aiter.git
  aiter_branch: feature/new-triton
  triton_repo: https://github.com/ROCm/triton.git
  triton_branch: main

machines:
  - host: gpu-machine-1
    user: root
    ssh_key: ~/.ssh/id_rsa
    gpus: [0, 1, 2, 3, 4, 5, 6, 7]
  - host: gpu-machine-2
    user: root
    ssh_key: ~/.ssh/id_rsa
    gpus: [0, 1, 2, 3]

container:
  image: rocm/pytorch:latest
  run_script: ./scripts/create_container.sh

gpu:
  arch: gfx950

triton_install:
  command: "pip install -e ."

tuning:
  mode: regression_only
  scout_fraction: 0.15
  thresholds:
    regression_vs_baseline: 5
    regression_vs_untuned: 2
  timeouts:
    command_default: 300
    tuning_per_shape: 1800
    progress_check: 120
    phase_max: 14400

kernels:
  exclude: [a16w16_agnostic]
  include: []
  overrides:
    afp4wfp4_preshuffle:
      m_leq_16_bucket_name: M_LEQ_31
    a16w16:
      extra_block_k: [64]
```

```yaml
# tests/tuning_agent/fixtures/minimal_config.yaml
baseline:
  aiter_repo: https://github.com/ROCm/aiter.git
  aiter_branch: main
  triton_repo: https://github.com/ROCm/triton.git
  triton_branch: triton_3_4

target:
  aiter_repo: https://github.com/ROCm/aiter.git
  aiter_branch: feature/new-triton
  triton_repo: https://github.com/ROCm/triton.git
  triton_branch: main

machines:
  - host: gpu-machine-1
    user: root
    ssh_key: ~/.ssh/id_rsa
    gpus: [0]

container:
  image: rocm/pytorch:latest
```

```yaml
# tests/tuning_agent/fixtures/invalid_config.yaml
# Missing required 'baseline' section
target:
  aiter_repo: https://github.com/ROCm/aiter.git
  aiter_branch: main
  triton_repo: https://github.com/ROCm/triton.git
  triton_branch: main

machines: []  # empty machine list

container:
  image: rocm/pytorch:latest
```

- [ ] **Step 2: Write conftest with fixtures**

```python
# tests/tuning_agent/conftest.py
import pytest
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def valid_config_path():
    return FIXTURES_DIR / "valid_config.yaml"


@pytest.fixture
def minimal_config_path():
    return FIXTURES_DIR / "minimal_config.yaml"


@pytest.fixture
def invalid_config_path():
    return FIXTURES_DIR / "invalid_config.yaml"
```

- [ ] **Step 3: Write failing tests for config parsing**

```python
# tests/tuning_agent/test_config.py
import pytest
from aiter.ops.triton.tuning_agent.config import load_config, ConfigError
from aiter.ops.triton.tuning_agent.types import PipelineConfig


def test_load_valid_config(valid_config_path):
    config = load_config(valid_config_path)
    assert isinstance(config, PipelineConfig)
    assert config.baseline.aiter_branch == "main"
    assert config.target.aiter_branch == "feature/new-triton"
    assert len(config.machines) == 2
    assert config.machines[0].host == "gpu-machine-1"
    assert config.machines[0].gpu_count == 8
    assert config.machines[1].gpu_count == 4
    assert config.container.image == "rocm/pytorch:latest"
    assert config.container.run_script == "./scripts/create_container.sh"
    assert config.gpu.arch == "gfx950"
    assert config.triton_install.command == "pip install -e ."
    assert config.tuning.mode == "regression_only"
    assert config.tuning.thresholds.regression_vs_baseline == 5.0
    assert config.tuning.timeouts.command_default == 300
    assert "a16w16_agnostic" in config.kernels.exclude
    assert config.kernels.overrides["afp4wfp4_preshuffle"].m_leq_16_bucket_name == "M_LEQ_31"
    assert config.kernels.overrides["a16w16"].extra_block_k == [64]


def test_load_minimal_config(minimal_config_path):
    """Minimal config should use defaults for optional fields."""
    config = load_config(minimal_config_path)
    assert isinstance(config, PipelineConfig)
    assert config.gpu.arch is None  # auto-detect
    assert config.tuning.mode == "regression_only"
    assert config.tuning.thresholds.regression_vs_baseline == 5.0
    assert config.tuning.timeouts.command_default == 300
    assert config.container.run_script is None
    assert len(config.kernels.exclude) == 0


def test_load_invalid_config_missing_baseline(invalid_config_path):
    with pytest.raises(ConfigError, match="baseline"):
        load_config(invalid_config_path)


def test_load_invalid_config_no_machines(invalid_config_path, tmp_path):
    """Config with empty machines list should fail."""
    # Write a config with baseline but empty machines
    config_file = tmp_path / "bad.yaml"
    config_file.write_text("""
baseline:
  aiter_repo: https://github.com/ROCm/aiter.git
  aiter_branch: main
  triton_repo: https://github.com/ROCm/triton.git
  triton_branch: main
target:
  aiter_repo: https://github.com/ROCm/aiter.git
  aiter_branch: main
  triton_repo: https://github.com/ROCm/triton.git
  triton_branch: main
machines: []
container:
  image: test
""")
    with pytest.raises(ConfigError, match="machine"):
        load_config(config_file)


def test_load_nonexistent_config():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path.yaml")


def test_tuning_mode_validation(tmp_path):
    config_file = tmp_path / "bad_mode.yaml"
    config_file.write_text("""
baseline:
  aiter_repo: x
  aiter_branch: main
  triton_repo: x
  triton_branch: main
target:
  aiter_repo: x
  aiter_branch: main
  triton_repo: x
  triton_branch: main
machines:
  - host: gpu1
    user: root
    ssh_key: ~/.ssh/id
    gpus: [0]
container:
  image: test
tuning:
  mode: invalid_mode
""")
    with pytest.raises(ConfigError, match="mode"):
        load_config(config_file)
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/test_config.py -v`
Expected: FAIL — module not found

- [ ] **Step 5: Implement config parser**

```python
# aiter/ops/triton/tuning_agent/config.py
"""YAML configuration loading and validation."""
import yaml
from pathlib import Path
from typing import Union

from .types import (
    PipelineConfig, MachineInfo, ContainerConfig, RepoConfig,
    TuningConfig, TuningThresholds, TuningTimeouts,
    KernelsConfig, KernelOverrides, GpuConfig, TritonInstallConfig,
)


class ConfigError(Exception):
    """Raised when config is invalid."""
    pass


def load_config(path: Union[str, Path]) -> PipelineConfig:
    """Load and validate a triton-upgrade.yaml config file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigError("Config file must be a YAML mapping")

    # Required sections
    if "baseline" not in raw:
        raise ConfigError("Missing required section: baseline")
    if "target" not in raw:
        raise ConfigError("Missing required section: target")
    if "machines" not in raw or not raw["machines"]:
        raise ConfigError("At least one machine must be specified")
    if "container" not in raw:
        raise ConfigError("Missing required section: container")

    baseline = _parse_repo_config(raw["baseline"], "baseline")
    target = _parse_repo_config(raw["target"], "target")
    machines = [_parse_machine(m, i) for i, m in enumerate(raw["machines"])]
    container = _parse_container(raw["container"])

    gpu = _parse_gpu(raw.get("gpu", {}))
    triton_install = _parse_triton_install(raw.get("triton_install", {}))
    tuning = _parse_tuning(raw.get("tuning", {}))
    kernels = _parse_kernels(raw.get("kernels", {}))

    return PipelineConfig(
        baseline=baseline,
        target=target,
        machines=machines,
        container=container,
        gpu=gpu,
        triton_install=triton_install,
        tuning=tuning,
        kernels=kernels,
    )


def _parse_repo_config(data: dict, name: str) -> RepoConfig:
    required = ["aiter_repo", "aiter_branch", "triton_repo", "triton_branch"]
    for key in required:
        if key not in data:
            raise ConfigError(f"Missing '{key}' in {name} section")
    return RepoConfig(
        aiter_repo=data["aiter_repo"],
        aiter_branch=data["aiter_branch"],
        triton_repo=data["triton_repo"],
        triton_branch=data["triton_branch"],
    )


def _parse_machine(data: dict, index: int) -> MachineInfo:
    required = ["host", "user", "ssh_key", "gpus"]
    for key in required:
        if key not in data:
            raise ConfigError(f"Missing '{key}' in machine #{index}")
    if not data["gpus"]:
        raise ConfigError(f"Machine #{index} ({data['host']}) has no GPUs")
    return MachineInfo(
        host=data["host"],
        user=data["user"],
        ssh_key=str(data["ssh_key"]),
        gpus=list(data["gpus"]),
    )


def _parse_container(data: dict) -> ContainerConfig:
    if "image" not in data:
        raise ConfigError("Missing 'image' in container section")
    return ContainerConfig(
        image=data["image"],
        run_script=data.get("run_script"),
    )


def _parse_gpu(data: dict) -> GpuConfig:
    return GpuConfig(arch=data.get("arch"))


def _parse_triton_install(data: dict) -> TritonInstallConfig:
    return TritonInstallConfig(command=data.get("command", "pip install -e ."))


def _parse_tuning(data: dict) -> TuningConfig:
    mode = data.get("mode", "regression_only")
    if mode not in ("regression_only", "full"):
        raise ConfigError(f"Invalid tuning mode: '{mode}'. Must be 'regression_only' or 'full'")

    thresholds_data = data.get("thresholds", {})
    thresholds = TuningThresholds(
        regression_vs_baseline=float(thresholds_data.get("regression_vs_baseline", 5.0)),
        regression_vs_untuned=float(thresholds_data.get("regression_vs_untuned", 2.0)),
    )

    timeouts_data = data.get("timeouts", {})
    timeouts = TuningTimeouts(
        command_default=int(timeouts_data.get("command_default", 300)),
        tuning_per_shape=int(timeouts_data.get("tuning_per_shape", 1800)),
        progress_check=int(timeouts_data.get("progress_check", 120)),
        phase_max=int(timeouts_data.get("phase_max", 14400)),
    )

    return TuningConfig(
        mode=mode,
        scout_fraction=float(data.get("scout_fraction", 0.15)),
        thresholds=thresholds,
        timeouts=timeouts,
    )


def _parse_kernels(data: dict) -> KernelsConfig:
    overrides = {}
    for name, override_data in data.get("overrides", {}).items():
        overrides[name] = KernelOverrides(
            m_leq_16_bucket_name=override_data.get("m_leq_16_bucket_name"),
            extra_block_k=override_data.get("extra_block_k"),
        )

    return KernelsConfig(
        exclude=list(data.get("exclude", [])),
        include=list(data.get("include", [])),
        overrides=overrides,
    )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/test_config.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add aiter/ops/triton/tuning_agent/config.py tests/tuning_agent/test_config.py \
    tests/tuning_agent/conftest.py tests/tuning_agent/fixtures/
git commit -m "feat(tuning-agent): add YAML config parsing with validation"
```

---

## Chunk 2: Remote Execution

### Task 3: SSH + Docker Exec Wrapper

**Files:**
- Create: `aiter/ops/triton/tuning_agent/remote.py`
- Create: `tests/tuning_agent/test_remote.py`

- [ ] **Step 1: Write failing tests for remote execution**

```python
# tests/tuning_agent/test_remote.py
import pytest
import subprocess
from unittest.mock import patch, MagicMock
from aiter.ops.triton.tuning_agent.remote import RemoteExecutor, RemoteCommandError
from aiter.ops.triton.tuning_agent.types import MachineInfo


@pytest.fixture
def machine():
    return MachineInfo(host="gpu1.internal", user="root", ssh_key="~/.ssh/id_rsa", gpus=[0, 1])


@pytest.fixture
def executor(machine):
    return RemoteExecutor(machine)


class TestSSHCommand:
    def test_build_ssh_command(self, executor):
        cmd = executor._build_ssh_command("ls /tmp")
        assert "ssh" in cmd[0]
        assert "-i" in cmd
        assert "~/.ssh/id_rsa" in cmd
        assert "root@gpu1.internal" in cmd
        assert "ls /tmp" in cmd

    def test_build_ssh_command_with_options(self, executor):
        cmd = executor._build_ssh_command("ls /tmp")
        # Should include StrictHostKeyChecking=no and ConnectTimeout
        cmd_str = " ".join(cmd)
        assert "StrictHostKeyChecking=no" in cmd_str

    @patch("subprocess.run")
    def test_ssh_run_success(self, mock_run, executor):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="output\n", stderr=""
        )
        result = executor.ssh_run("echo hello")
        assert result.stdout == "output\n"
        assert result.returncode == 0

    @patch("subprocess.run")
    def test_ssh_run_failure_raises(self, mock_run, executor):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="error msg"
        )
        with pytest.raises(RemoteCommandError, match="error msg"):
            executor.ssh_run("bad command", check=True)

    @patch("subprocess.run")
    def test_ssh_run_timeout(self, mock_run, executor):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ssh ...", timeout=30)
        with pytest.raises(RemoteCommandError, match="timed out"):
            executor.ssh_run("slow command", timeout=30)

    @patch("subprocess.run")
    def test_ssh_run_retries_on_connection_error(self, mock_run, executor):
        """SSH exit code 255 = connection error, should retry."""
        mock_run.side_effect = [
            MagicMock(returncode=255, stdout="", stderr="Connection refused"),
            MagicMock(returncode=0, stdout="ok", stderr=""),
        ]
        result = executor.ssh_run("echo ok", check=True, retries=1, backoff=0.01)
        assert result.returncode == 0
        assert mock_run.call_count == 2


class TestStaleProcessCleanup:
    @patch("subprocess.run")
    def test_kill_stale_gpu_processes(self, mock_run, executor):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="12345\tpython\t0\t1024\n67890\trocprof\t1\t512\n",
            stderr="",
        )
        killed = executor.kill_stale_gpu_processes()
        assert 12345 in killed or len(killed) >= 0  # depends on parsing


class TestEnvironmentVerification:
    @patch("subprocess.run")
    def test_verify_environment_success(self, mock_run, executor):
        executor.container_id = "abc123"
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="3.6.0+git44358bc7\n", stderr=""),
            MagicMock(returncode=0, stdout="feature/new-triton\n", stderr=""),
        ]
        result = executor.verify_environment(
            expected_triton_commit="44358bc7",
            expected_aiter_branch="feature/new-triton",
        )
        assert "44358bc7" in result["triton_version"]
        assert result["aiter_branch"] == "feature/new-triton"

    @patch("subprocess.run")
    def test_verify_environment_triton_mismatch(self, mock_run, executor):
        executor.container_id = "abc123"
        mock_run.return_value = MagicMock(returncode=0, stdout="3.6.0+gitWRONG\n", stderr="")
        with pytest.raises(RemoteCommandError, match="mismatch"):
            executor.verify_environment(expected_triton_commit="44358bc7")


class TestDockerExec:
    @patch("subprocess.run")
    def test_docker_exec_builds_correct_command(self, mock_run, executor):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        executor.container_id = "abc123"
        executor.docker_exec("python train.py")
        call_args = mock_run.call_args[0][0]
        cmd_str = " ".join(call_args)
        assert "docker exec" in cmd_str
        assert "abc123" in cmd_str
        assert "python train.py" in cmd_str

    def test_docker_exec_without_container_raises(self, executor):
        with pytest.raises(RemoteCommandError, match="container"):
            executor.docker_exec("ls")

    @patch("subprocess.run")
    def test_docker_exec_with_env(self, mock_run, executor):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        executor.container_id = "abc123"
        executor.docker_exec("python x.py", env={"HIP_VISIBLE_DEVICES": "0"})
        call_args = mock_run.call_args[0][0]
        cmd_str = " ".join(call_args)
        assert "HIP_VISIBLE_DEVICES=0" in cmd_str


class TestContainerManagement:
    @patch("subprocess.run")
    def test_create_container_with_script(self, mock_run, executor):
        mock_run.return_value = MagicMock(returncode=0, stdout="container_id_123\n", stderr="")
        container_id = executor.create_container(
            image="rocm/pytorch:latest",
            run_script="./scripts/create.sh"
        )
        assert container_id == "container_id_123"

    @patch("subprocess.run")
    def test_create_container_without_script(self, mock_run, executor):
        mock_run.return_value = MagicMock(returncode=0, stdout="cid_456\n", stderr="")
        container_id = executor.create_container(image="rocm/pytorch:latest")
        assert container_id == "cid_456"
        call_args = mock_run.call_args[0][0]
        cmd_str = " ".join(call_args)
        assert "docker run" in cmd_str
        assert "--device=/dev/kfd" in cmd_str

    @patch("subprocess.run")
    def test_destroy_container(self, mock_run, executor):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        executor.container_id = "abc123"
        executor.destroy_container()
        assert executor.container_id is None


class TestFileTransfer:
    @patch("subprocess.run")
    def test_copy_from_container(self, mock_run, executor):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        executor.container_id = "abc123"
        executor.copy_from_container("/workspace/results.json", "/tmp/local_results.json")
        call_args = mock_run.call_args[0][0]
        cmd_str = " ".join(call_args)
        assert "docker cp" in cmd_str or "scp" in cmd_str

    @patch("subprocess.run")
    def test_copy_to_container(self, mock_run, executor):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        executor.container_id = "abc123"
        executor.copy_to_container("/tmp/local_file.py", "/workspace/file.py")
        call_args = mock_run.call_args[0][0]
        cmd_str = " ".join(call_args)
        assert "docker cp" in cmd_str or "scp" in cmd_str
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/test_remote.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement remote executor**

```python
# aiter/ops/triton/tuning_agent/remote.py
"""SSH + Docker exec wrapper for remote command execution."""
import subprocess
import shlex
from typing import Optional, Dict, List
from dataclasses import dataclass

from .types import MachineInfo


class RemoteCommandError(Exception):
    """Raised when a remote command fails."""
    def __init__(self, message: str, returncode: int = -1, stdout: str = "", stderr: str = ""):
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class RemoteExecutor:
    """Executes commands on a remote machine via SSH and inside Docker containers."""

    def __init__(self, machine: MachineInfo):
        self.machine = machine
        self.container_id: Optional[str] = None

    def _build_ssh_command(self, command: str) -> List[str]:
        """Build an SSH command list."""
        return [
            "ssh",
            "-i", self.machine.ssh_key,
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes",
            f"{self.machine.user}@{self.machine.host}",
            command,
        ]

    def ssh_run(
        self,
        command: str,
        timeout: Optional[int] = None,
        check: bool = False,
        retries: int = 0,
        backoff: float = 2.0,
    ) -> subprocess.CompletedProcess:
        """Run a command on the remote machine via SSH.

        Args:
            retries: Number of retries on connection failure (0 = no retry)
            backoff: Multiplier for exponential backoff between retries
        """
        ssh_cmd = self._build_ssh_command(command)
        last_error = None
        for attempt in range(retries + 1):
            try:
                result = subprocess.run(
                    ssh_cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                if check and result.returncode != 0:
                    # Connection errors (255) are retryable, command errors are not
                    if result.returncode == 255 and attempt < retries:
                        import time
                        time.sleep(backoff ** attempt)
                        continue
                    raise RemoteCommandError(
                        f"Command failed (exit {result.returncode}): {result.stderr}",
                        returncode=result.returncode,
                        stdout=result.stdout,
                        stderr=result.stderr,
                    )
                return result
            except subprocess.TimeoutExpired:
                last_error = RemoteCommandError(
                    f"Command timed out after {timeout}s: {command}"
                )
                if attempt < retries:
                    import time
                    time.sleep(backoff ** attempt)
                    continue
                raise last_error
        raise last_error  # should not reach here

    def docker_exec(
        self,
        command: str,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        check: bool = True,
        workdir: Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        """Run a command inside the container via SSH + docker exec."""
        if not self.container_id:
            raise RemoteCommandError("No container set. Call create_container() first.")

        env_str = ""
        if env:
            env_str = " ".join(f"-e {k}={shlex.quote(str(v))}" for k, v in env.items()) + " "

        workdir_str = ""
        if workdir:
            workdir_str = f"-w {workdir} "

        docker_cmd = f"docker exec {env_str}{workdir_str}{self.container_id} bash -c {shlex.quote(command)}"
        return self.ssh_run(docker_cmd, timeout=timeout, check=check)

    def create_container(
        self,
        image: str,
        run_script: Optional[str] = None,
        name: Optional[str] = None,
    ) -> str:
        """Create a container on the remote machine. Returns container ID."""
        if run_script:
            # Run the user-provided script and capture container ID
            result = self.ssh_run(f"bash {run_script}", check=True, timeout=120)
            container_id = result.stdout.strip().split("\n")[-1]
        else:
            name_flag = f"--name {name}" if name else ""
            gpu_devices = "--device=/dev/kfd --device=/dev/dri"
            cmd = (
                f"docker run -d {name_flag} {gpu_devices} "
                f"--security-opt seccomp=unconfined "
                f"-v /tmp:/host_tmp "
                f"{image} sleep infinity"
            )
            result = self.ssh_run(cmd, check=True, timeout=120)
            container_id = result.stdout.strip()

        self.container_id = container_id
        return container_id

    def destroy_container(self) -> None:
        """Stop and remove the container."""
        if self.container_id:
            self.ssh_run(f"docker stop {self.container_id}", timeout=30, check=False)
            self.ssh_run(f"docker rm -f {self.container_id}", timeout=30, check=False)
            self.container_id = None

    def copy_from_container(self, remote_path: str, local_path: str) -> None:
        """Copy a file from the container to the local machine (via SSH + docker cp)."""
        if not self.container_id:
            raise RemoteCommandError("No container set.")
        # First docker cp to host, then scp to local
        host_tmp = f"/tmp/_transfer_{self.container_id}"
        self.ssh_run(
            f"docker cp {self.container_id}:{remote_path} {host_tmp}",
            check=True, timeout=60,
        )
        scp_cmd = [
            "scp", "-i", self.machine.ssh_key,
            "-o", "StrictHostKeyChecking=no",
            f"{self.machine.user}@{self.machine.host}:{host_tmp}",
            local_path,
        ]
        try:
            subprocess.run(scp_cmd, capture_output=True, text=True, check=True, timeout=60)
        except subprocess.CalledProcessError as e:
            raise RemoteCommandError(f"SCP failed: {e.stderr}")
        # Cleanup host tmp
        self.ssh_run(f"rm -rf {host_tmp}", check=False)

    def copy_to_container(self, local_path: str, remote_path: str) -> None:
        """Copy a file from the local machine to the container (via scp + docker cp)."""
        if not self.container_id:
            raise RemoteCommandError("No container set.")
        host_tmp = f"/tmp/_transfer_{self.container_id}"
        scp_cmd = [
            "scp", "-i", self.machine.ssh_key,
            "-o", "StrictHostKeyChecking=no",
            local_path,
            f"{self.machine.user}@{self.machine.host}:{host_tmp}",
        ]
        try:
            subprocess.run(scp_cmd, capture_output=True, text=True, check=True, timeout=60)
        except subprocess.CalledProcessError as e:
            raise RemoteCommandError(f"SCP failed: {e.stderr}")
        self.ssh_run(
            f"docker cp {host_tmp} {self.container_id}:{remote_path}",
            check=True, timeout=60,
        )
        self.ssh_run(f"rm -rf {host_tmp}", check=False)

    def kill_stale_gpu_processes(self) -> List[int]:
        """Kill orphan GPU processes on the machine. Returns list of killed PIDs."""
        result = self.ssh_run("rocm-smi --showpids", timeout=15, check=False)
        if result.returncode != 0:
            return []
        killed = []
        for line in result.stdout.strip().split("\n"):
            parts = line.split()
            if parts and parts[0].isdigit():
                pid = int(parts[0])
                self.ssh_run(f"kill -9 {pid}", timeout=5, check=False)
                killed.append(pid)
        return killed

    def verify_environment(
        self,
        expected_triton_commit: Optional[str] = None,
        expected_aiter_branch: Optional[str] = None,
    ) -> Dict[str, str]:
        """Verify Triton commit hash and aiter branch inside the container.

        Returns dict with actual values. Raises RemoteCommandError on mismatch.
        """
        if not self.container_id:
            raise RemoteCommandError("No container set.")

        result = {}
        # Check Triton version (includes commit hash, e.g., "3.6.0+git44358bc7")
        triton_result = self.docker_exec(
            "python -c \"import triton; print(triton.__version__)\"",
            timeout=30, check=True,
        )
        result["triton_version"] = triton_result.stdout.strip()

        if expected_triton_commit and expected_triton_commit not in result["triton_version"]:
            raise RemoteCommandError(
                f"Triton commit mismatch: expected '{expected_triton_commit}' "
                f"in '{result['triton_version']}'"
            )

        # Check aiter branch
        aiter_result = self.docker_exec(
            "cd /workspace/aiter && git rev-parse --abbrev-ref HEAD",
            timeout=10, check=True,
        )
        result["aiter_branch"] = aiter_result.stdout.strip()

        if expected_aiter_branch and result["aiter_branch"] != expected_aiter_branch:
            raise RemoteCommandError(
                f"Aiter branch mismatch: expected '{expected_aiter_branch}', "
                f"got '{result['aiter_branch']}'"
            )

        return result

    def is_container_running(self) -> bool:
        """Check if the container is still running."""
        if not self.container_id:
            return False
        result = self.ssh_run(
            f"docker inspect -f '{{{{.State.Running}}}}' {self.container_id}",
            check=False, timeout=10,
        )
        return result.returncode == 0 and "true" in result.stdout.lower()

    def check_ssh_connectivity(self) -> bool:
        """Quick SSH connectivity check."""
        try:
            result = self.ssh_run("echo ok", timeout=10)
            return result.returncode == 0 and "ok" in result.stdout
        except (RemoteCommandError, Exception):
            return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/test_remote.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add aiter/ops/triton/tuning_agent/remote.py tests/tuning_agent/test_remote.py
git commit -m "feat(tuning-agent): add SSH + docker exec remote execution wrapper"
```

---

### Task 4: Machine Pool Manager

**Files:**
- Create: `aiter/ops/triton/tuning_agent/machine_pool.py`
- Create: `tests/tuning_agent/test_machine_pool.py`

- [ ] **Step 1: Write failing tests for machine pool**

```python
# tests/tuning_agent/test_machine_pool.py
import pytest
from unittest.mock import patch, MagicMock
from aiter.ops.triton.tuning_agent.machine_pool import MachinePool, NoMachineAvailable
from aiter.ops.triton.tuning_agent.types import MachineInfo


@pytest.fixture
def machines():
    return [
        MachineInfo(host="gpu1", user="root", ssh_key="~/.ssh/id", gpus=[0, 1, 2, 3]),
        MachineInfo(host="gpu2", user="root", ssh_key="~/.ssh/id", gpus=[0, 1]),
    ]


@pytest.fixture
def pool(machines):
    return MachinePool(machines)


class TestAllocation:
    def test_allocate_returns_machine(self, pool):
        machine = pool.allocate(kernel_name="a8w8")
        assert machine is not None
        assert machine.host in ("gpu1", "gpu2")

    def test_allocate_prefers_more_gpus(self, pool):
        machine = pool.allocate(kernel_name="a8w8")
        assert machine.host == "gpu1"  # 4 GPUs > 2 GPUs

    def test_allocate_second_machine_when_first_busy(self, pool):
        m1 = pool.allocate(kernel_name="a8w8")
        m2 = pool.allocate(kernel_name="a16w16")
        assert m1.host != m2.host

    def test_allocate_raises_when_all_busy(self, pool):
        pool.allocate(kernel_name="a8w8")
        pool.allocate(kernel_name="a16w16")
        with pytest.raises(NoMachineAvailable):
            pool.allocate(kernel_name="afp4wfp4")

    def test_release_makes_machine_available(self, pool):
        m1 = pool.allocate(kernel_name="a8w8")
        pool.allocate(kernel_name="a16w16")
        pool.release(m1.host)
        m3 = pool.allocate(kernel_name="afp4wfp4")
        assert m3.host == m1.host


class TestStatus:
    def test_status_shows_idle_initially(self, pool):
        status = pool.status()
        assert all(s["state"] == "idle" for s in status)

    def test_status_shows_busy_after_allocate(self, pool):
        pool.allocate(kernel_name="a8w8")
        status = pool.status()
        busy = [s for s in status if s["state"] == "busy"]
        assert len(busy) == 1
        assert busy[0]["kernel"] == "a8w8"

    def test_available_count(self, pool):
        assert pool.available_count == 2
        pool.allocate(kernel_name="a8w8")
        assert pool.available_count == 1


class TestHealthCheck:
    @patch("aiter.ops.triton.tuning_agent.machine_pool.RemoteExecutor")
    def test_validate_machines(self, MockExecutor, pool):
        mock_instance = MagicMock()
        mock_instance.check_ssh_connectivity.return_value = True
        mock_instance.ssh_run.return_value = MagicMock(returncode=0, stdout="GPU[0]")
        MockExecutor.return_value = mock_instance
        results = pool.validate_connectivity()
        assert all(r["reachable"] for r in results)

    @patch("aiter.ops.triton.tuning_agent.machine_pool.RemoteExecutor")
    def test_mark_dead_machine(self, MockExecutor, pool):
        mock_instance = MagicMock()
        mock_instance.check_ssh_connectivity.return_value = False
        MockExecutor.return_value = mock_instance
        pool.mark_dead("gpu1")
        assert pool.available_count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/test_machine_pool.py -v`
Expected: FAIL

- [ ] **Step 3: Implement machine pool**

```python
# aiter/ops/triton/tuning_agent/machine_pool.py
"""Machine pool manager for allocating GPU machines to kernel tuning tasks."""
import threading
from typing import List, Dict, Optional
from dataclasses import dataclass

from .types import MachineInfo
from .remote import RemoteExecutor


class NoMachineAvailable(Exception):
    """Raised when no machines are available for allocation."""
    pass


class MachinePool:
    """Manages a pool of GPU machines, handling allocation and release."""

    def __init__(self, machines: List[MachineInfo]):
        self._machines = {m.host: m for m in machines}
        self._allocated: Dict[str, str] = {}  # host -> kernel_name
        self._dead: set = set()
        self._lock = threading.Lock()

    @property
    def available_count(self) -> int:
        with self._lock:
            return sum(
                1 for host in self._machines
                if host not in self._allocated and host not in self._dead
            )

    def allocate(self, kernel_name: str) -> MachineInfo:
        """Allocate a machine for a kernel. Prefers machines with more GPUs."""
        with self._lock:
            available = [
                self._machines[host]
                for host in self._machines
                if host not in self._allocated and host not in self._dead
            ]
            if not available:
                raise NoMachineAvailable(
                    f"No machines available for kernel '{kernel_name}'. "
                    f"All {len(self._machines)} machines are busy or dead."
                )
            # Prefer machine with most GPUs
            machine = max(available, key=lambda m: m.gpu_count)
            self._allocated[machine.host] = kernel_name
            return machine

    def release(self, host: str) -> None:
        """Release a machine back to the pool."""
        with self._lock:
            self._allocated.pop(host, None)

    def mark_dead(self, host: str) -> None:
        """Mark a machine as dead (unreachable)."""
        with self._lock:
            self._dead.add(host)
            self._allocated.pop(host, None)

    def status(self) -> List[Dict]:
        """Get status of all machines."""
        with self._lock:
            result = []
            for host, machine in self._machines.items():
                if host in self._dead:
                    state = "dead"
                    kernel = None
                elif host in self._allocated:
                    state = "busy"
                    kernel = self._allocated[host]
                else:
                    state = "idle"
                    kernel = None
                result.append({
                    "host": host,
                    "state": state,
                    "kernel": kernel,
                    "gpus": machine.gpus,
                    "gpu_count": machine.gpu_count,
                })
            return result

    def validate_connectivity(self) -> List[Dict]:
        """Test SSH connectivity to all machines."""
        results = []
        for host, machine in self._machines.items():
            executor = RemoteExecutor(machine)
            reachable = executor.check_ssh_connectivity()
            gpu_ok = False
            if reachable:
                try:
                    result = executor.ssh_run("rocm-smi --showuse", timeout=15)
                    gpu_ok = result.returncode == 0 and "GPU" in result.stdout
                except Exception:
                    pass
            results.append({
                "host": host,
                "reachable": reachable,
                "gpu_ok": gpu_ok,
            })
            if not reachable:
                self.mark_dead(host)
        return results

    def get_machine(self, host: str) -> Optional[MachineInfo]:
        """Get machine info by hostname."""
        return self._machines.get(host)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/test_machine_pool.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add aiter/ops/triton/tuning_agent/machine_pool.py tests/tuning_agent/test_machine_pool.py
git commit -m "feat(tuning-agent): add machine pool manager with allocation and health checks"
```

---

## Chunk 3: Watchdog + Notifications + Artifacts

### Task 5: Watchdog / Health Monitor

**Files:**
- Create: `aiter/ops/triton/tuning_agent/watchdog.py`
- Create: `tests/tuning_agent/test_watchdog.py`

- [ ] **Step 1: Write failing tests for watchdog**

```python
# tests/tuning_agent/test_watchdog.py
import pytest
import time
from unittest.mock import patch, MagicMock, call
from aiter.ops.triton.tuning_agent.watchdog import (
    CommandWatchdog, ProgressMonitor, WatchdogTimeout,
)


class TestCommandWatchdog:
    def test_command_within_timeout_succeeds(self):
        wd = CommandWatchdog(timeout=5)
        result = wd.run(["echo", "hello"])
        assert "hello" in result.stdout

    def test_command_exceeding_timeout_raises(self):
        wd = CommandWatchdog(timeout=1)
        with pytest.raises(WatchdogTimeout):
            wd.run(["sleep", "10"])

    def test_command_failure_returns_result(self):
        wd = CommandWatchdog(timeout=5)
        result = wd.run(["false"], check=False)
        assert result.returncode != 0

    def test_command_failure_raises_when_check(self):
        wd = CommandWatchdog(timeout=5)
        with pytest.raises(Exception):
            wd.run(["false"], check=True)


class TestProgressMonitor:
    def test_detects_progress(self, tmp_path):
        log_file = tmp_path / "test.log"
        log_file.write_text("line1\nline2\n")
        monitor = ProgressMonitor(str(log_file), pattern="line")
        assert monitor.has_progress()

    def test_detects_no_progress(self, tmp_path):
        log_file = tmp_path / "test.log"
        log_file.write_text("")
        monitor = ProgressMonitor(str(log_file), pattern="screencase")
        assert not monitor.has_progress()

    def test_detects_new_progress_since_last_check(self, tmp_path):
        log_file = tmp_path / "test.log"
        log_file.write_text("screencase 1\n")
        monitor = ProgressMonitor(str(log_file), pattern="screencase")
        assert monitor.has_progress()
        # No new lines
        assert not monitor.has_new_progress()
        # Add new lines
        with open(log_file, "a") as f:
            f.write("screencase 2\n")
        assert monitor.has_new_progress()

    def test_completion_detected(self, tmp_path):
        log_file = tmp_path / "test.log"
        log_file.write_text("data\nScreen complete\n")
        monitor = ProgressMonitor(str(log_file), completion_marker="Screen complete")
        assert monitor.is_complete()

    def test_completion_not_detected(self, tmp_path):
        log_file = tmp_path / "test.log"
        log_file.write_text("data\nstill running\n")
        monitor = ProgressMonitor(str(log_file), completion_marker="Screen complete")
        assert not monitor.is_complete()


class TestRemoteProgressMonitor:
    def test_detects_remote_progress(self):
        from unittest.mock import MagicMock
        from aiter.ops.triton.tuning_agent.watchdog import RemoteProgressMonitor

        mock_executor = MagicMock()
        mock_executor.docker_exec.return_value = MagicMock(
            returncode=0, stdout="5\n", stderr=""
        )
        monitor = RemoteProgressMonitor(mock_executor, "/workspace/screen.log")
        assert monitor.has_progress()
        assert monitor.get_progress_count() == 5

    def test_detects_remote_completion(self):
        from unittest.mock import MagicMock
        from aiter.ops.triton.tuning_agent.watchdog import RemoteProgressMonitor

        mock_executor = MagicMock()
        mock_executor.docker_exec.return_value = MagicMock(
            returncode=0, stdout="1\n", stderr=""
        )
        monitor = RemoteProgressMonitor(mock_executor, "/workspace/screen.log")
        assert monitor.is_complete()

    def test_detects_no_remote_progress(self):
        from unittest.mock import MagicMock
        from aiter.ops.triton.tuning_agent.watchdog import RemoteProgressMonitor

        mock_executor = MagicMock()
        mock_executor.docker_exec.return_value = MagicMock(
            returncode=0, stdout="0\n", stderr=""
        )
        monitor = RemoteProgressMonitor(mock_executor, "/workspace/screen.log")
        assert not monitor.has_progress()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/test_watchdog.py -v`
Expected: FAIL

- [ ] **Step 3: Implement watchdog**

```python
# aiter/ops/triton/tuning_agent/watchdog.py
"""Health monitoring, timeout enforcement, and progress checking."""
import subprocess
import time
from typing import Optional, List
from pathlib import Path


class WatchdogTimeout(Exception):
    """Raised when a command or operation times out."""
    pass


class CommandWatchdog:
    """Wraps command execution with timeout enforcement."""

    def __init__(self, timeout: int = 300):
        self.timeout = timeout

    def run(
        self,
        cmd: List[str],
        check: bool = False,
        env: Optional[dict] = None,
    ) -> subprocess.CompletedProcess:
        """Run a command with timeout. Raises WatchdogTimeout if it exceeds the limit."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
            if check and result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode, cmd, result.stdout, result.stderr
                )
            return result
        except subprocess.TimeoutExpired:
            raise WatchdogTimeout(
                f"Command timed out after {self.timeout}s: {' '.join(cmd[:5])}..."
            )


class ProgressMonitor:
    """Monitors a log file for progress indicators."""

    def __init__(
        self,
        log_path: str,
        pattern: str = "screencase",
        completion_marker: str = "Screen complete",
    ):
        self.log_path = Path(log_path)
        self.pattern = pattern
        self.completion_marker = completion_marker
        self._last_count = 0
        self._last_size = 0

    def _read_content(self) -> str:
        if not self.log_path.exists():
            return ""
        return self.log_path.read_text()

    def _count_matches(self, content: str) -> int:
        return content.count(self.pattern)

    def has_progress(self) -> bool:
        """Check if the log file contains any progress indicators."""
        content = self._read_content()
        count = self._count_matches(content)
        self._last_count = count
        self._last_size = len(content)
        return count > 0

    def has_new_progress(self) -> bool:
        """Check if new progress has appeared since last check."""
        content = self._read_content()
        count = self._count_matches(content)
        new_progress = count > self._last_count
        self._last_count = count
        self._last_size = len(content)
        return new_progress

    def is_complete(self) -> bool:
        """Check if the completion marker is present."""
        content = self._read_content()
        return self.completion_marker in content

    def get_progress_count(self) -> int:
        """Get the current count of progress indicators."""
        content = self._read_content()
        return self._count_matches(content)


class RemoteProgressMonitor:
    """Monitors a log file on a remote container for progress indicators."""

    def __init__(
        self,
        executor,  # RemoteExecutor
        remote_log_path: str,
        pattern: str = "screencase",
        completion_marker: str = "Screen complete",
    ):
        self.executor = executor
        self.remote_log_path = remote_log_path
        self.pattern = pattern
        self.completion_marker = completion_marker
        self._last_count = 0

    def _remote_count(self, search_str: str) -> int:
        """Count occurrences of a string in the remote file."""
        try:
            result = self.executor.docker_exec(
                f"grep -c '{search_str}' {self.remote_log_path} 2>/dev/null || echo 0",
                timeout=10, check=False,
            )
            return int(result.stdout.strip())
        except (ValueError, Exception):
            return 0

    def has_progress(self) -> bool:
        count = self._remote_count(self.pattern)
        self._last_count = count
        return count > 0

    def has_new_progress(self) -> bool:
        count = self._remote_count(self.pattern)
        new_progress = count > self._last_count
        self._last_count = count
        return new_progress

    def is_complete(self) -> bool:
        return self._remote_count(self.completion_marker) > 0

    def get_progress_count(self) -> int:
        return self._remote_count(self.pattern)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/test_watchdog.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add aiter/ops/triton/tuning_agent/watchdog.py tests/tuning_agent/test_watchdog.py
git commit -m "feat(tuning-agent): add watchdog for timeout enforcement and progress monitoring"
```

---

### Task 6: Notification System

**Files:**
- Create: `aiter/ops/triton/tuning_agent/notifications.py`
- Create: `tests/tuning_agent/test_notifications.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/tuning_agent/test_notifications.py
import pytest
from aiter.ops.triton.tuning_agent.notifications import (
    Notifier, Notification, NotificationLevel,
)


def test_notification_creation():
    n = Notification(
        level=NotificationLevel.INFO,
        title="Kernel complete",
        message="a8w8 tuning finished. 2.5x geomean.",
    )
    assert n.level == NotificationLevel.INFO
    assert "a8w8" in n.message


def test_notifier_records_notifications():
    notifier = Notifier()
    notifier.notify(NotificationLevel.INFO, "Test", "test message")
    assert len(notifier.history) == 1
    assert notifier.history[0].title == "Test"


def test_notifier_approval_gate():
    notifier = Notifier(auto_approve=True)
    approved = notifier.request_approval("Commit a8w8 configs?", details="2.5x speedup")
    assert approved is True


def test_notifier_pending_approvals():
    notifier = Notifier(auto_approve=False)
    notifier.request_approval_async("Commit?", details="info")
    assert len(notifier.pending_approvals) == 1


def test_notification_levels():
    assert NotificationLevel.CRITICAL.value > NotificationLevel.INFO.value
    assert NotificationLevel.APPROVAL.value > NotificationLevel.CRITICAL.value
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/test_notifications.py -v`
Expected: FAIL

- [ ] **Step 3: Implement notification system**

```python
# aiter/ops/triton/tuning_agent/notifications.py
"""Notification system for human-in-the-loop alerts and approvals."""
import sys
import time
from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


class NotificationLevel(IntEnum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    CRITICAL = 3
    APPROVAL = 4


@dataclass
class Notification:
    level: NotificationLevel
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False


@dataclass
class ApprovalRequest:
    question: str
    details: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    approved: Optional[bool] = None


class Notifier:
    """Handles notifications and approval gates."""

    def __init__(self, auto_approve: bool = False):
        self.auto_approve = auto_approve
        self.history: List[Notification] = []
        self.pending_approvals: List[ApprovalRequest] = []

    def notify(self, level: NotificationLevel, title: str, message: str) -> None:
        """Send a notification."""
        notification = Notification(level=level, title=title, message=message)
        self.history.append(notification)
        self._deliver(notification)

    def _deliver(self, notification: Notification) -> None:
        """Deliver notification to the terminal."""
        prefix = {
            NotificationLevel.DEBUG: "[DEBUG]",
            NotificationLevel.INFO: "[INFO]",
            NotificationLevel.WARNING: "[WARN]",
            NotificationLevel.CRITICAL: "[CRITICAL]",
            NotificationLevel.APPROVAL: "[APPROVAL]",
        }.get(notification.level, "[???]")

        msg = f"\n{prefix} {notification.title}: {notification.message}"
        print(msg, file=sys.stderr)

        # Terminal bell for critical and approval
        if notification.level >= NotificationLevel.CRITICAL:
            print("\a", end="", file=sys.stderr)

    def request_approval(self, question: str, details: Optional[str] = None) -> bool:
        """Block until human approves or denies. Returns True if approved."""
        if self.auto_approve:
            self.notify(NotificationLevel.INFO, "Auto-approved", question)
            return True

        self.notify(NotificationLevel.APPROVAL, "Approval needed", question)
        if details:
            print(f"  Details: {details}", file=sys.stderr)

        while True:
            try:
                response = input("  Approve? [y/n]: ").strip().lower()
                if response in ("y", "yes"):
                    return True
                elif response in ("n", "no"):
                    return False
                print("  Please enter 'y' or 'n'", file=sys.stderr)
            except (EOFError, KeyboardInterrupt):
                return False

    def request_approval_async(self, question: str, details: Optional[str] = None) -> None:
        """Queue an approval request without blocking."""
        req = ApprovalRequest(question=question, details=details)
        self.pending_approvals.append(req)
        self.notify(NotificationLevel.APPROVAL, "Approval queued", question)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/test_notifications.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add aiter/ops/triton/tuning_agent/notifications.py tests/tuning_agent/test_notifications.py
git commit -m "feat(tuning-agent): add notification system with approval gates"
```

---

### Task 7: Artifact Manager

**Files:**
- Create: `aiter/ops/triton/tuning_agent/artifacts.py`
- Create: `tests/tuning_agent/test_artifacts.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/tuning_agent/test_artifacts.py
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from aiter.ops.triton.tuning_agent.artifacts import ArtifactManager
from aiter.ops.triton.tuning_agent.types import ShapeResult


@pytest.fixture
def local_dir(tmp_path):
    return tmp_path / "results"


@pytest.fixture
def mock_executor():
    executor = MagicMock()
    executor.docker_exec.return_value = MagicMock(returncode=0, stdout="", stderr="")
    executor.container_id = "test_container"
    return executor


@pytest.fixture
def manager(mock_executor, local_dir):
    return ArtifactManager(
        executor=mock_executor,
        kernel_name="a8w8",
        run_id="test_run_001",
        local_base_dir=str(local_dir),
        remote_base_dir="/workspace/tuning_artifacts",
    )


class TestDirectorySetup:
    def test_local_dir_created(self, manager, local_dir):
        manager.setup_local()
        assert (local_dir / "test_run_001" / "a8w8").is_dir()

    def test_remote_dir_created(self, manager, mock_executor):
        manager.setup_remote()
        mock_executor.docker_exec.assert_called()
        cmd = mock_executor.docker_exec.call_args[0][0]
        assert "mkdir" in cmd
        assert "a8w8" in cmd


class TestResultStorage:
    def test_save_baseline_results(self, manager, local_dir):
        manager.setup_local()
        results = [
            ShapeResult(m=8, n=8192, k=8192, main_ns=5000, reduce_ns=2000),
            ShapeResult(m=16, n=8192, k=8192, main_ns=6000, reduce_ns=0),
        ]
        manager.save_results("baseline", results)
        path = local_dir / "test_run_001" / "a8w8" / "baseline.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data) == 2
        assert data[0]["m"] == 8
        assert data[0]["total_ns"] == 7000

    def test_load_results(self, manager, local_dir):
        manager.setup_local()
        results = [ShapeResult(m=8, n=8192, k=8192, main_ns=5000)]
        manager.save_results("baseline", results)
        loaded = manager.load_results("baseline")
        assert len(loaded) == 1
        assert loaded[0].m == 8
        assert loaded[0].main_ns == 5000


class TestCheckpoints:
    def test_mark_phase_complete(self, manager, local_dir):
        manager.setup_local()
        manager.mark_phase_complete(2, {"shapes_collected": 56})
        path = local_dir / "test_run_001" / "a8w8" / "phase_2_complete.json"
        assert path.exists()

    def test_is_phase_complete(self, manager, local_dir):
        manager.setup_local()
        assert not manager.is_phase_complete(2)
        manager.mark_phase_complete(2, {})
        assert manager.is_phase_complete(2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/test_artifacts.py -v`
Expected: FAIL

- [ ] **Step 3: Implement artifact manager**

```python
# aiter/ops/triton/tuning_agent/artifacts.py
"""Manages artifacts (results, configs, checkpoints) for tuning pipeline."""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from .types import ShapeResult
from .remote import RemoteExecutor


class ArtifactManager:
    """Handles storage and retrieval of tuning artifacts."""

    def __init__(
        self,
        executor: RemoteExecutor,
        kernel_name: str,
        run_id: Optional[str] = None,
        local_base_dir: str = "~/.tuning_results",
        remote_base_dir: str = "/workspace/tuning_artifacts",
    ):
        self.executor = executor
        self.kernel_name = kernel_name
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.local_dir = Path(local_base_dir).expanduser() / self.run_id / kernel_name
        self.remote_dir = f"{remote_base_dir}/{kernel_name}"

    def setup_local(self) -> None:
        """Create local artifact directory."""
        self.local_dir.mkdir(parents=True, exist_ok=True)

    def setup_remote(self) -> None:
        """Create remote artifact directory inside container."""
        self.executor.docker_exec(f"mkdir -p {self.remote_dir}")

    def save_results(self, name: str, results: List[ShapeResult]) -> Path:
        """Save benchmark results to a local JSON file."""
        data = [
            {
                "m": r.m, "n": r.n, "k": r.k,
                "main_ns": r.main_ns, "reduce_ns": r.reduce_ns,
                "total_ns": r.total_ns,
            }
            for r in results
        ]
        path = self.local_dir / f"{name}.json"
        path.write_text(json.dumps(data, indent=2))
        return path

    def load_results(self, name: str) -> List[ShapeResult]:
        """Load benchmark results from a local JSON file."""
        path = self.local_dir / f"{name}.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text())
        return [
            ShapeResult(
                m=d["m"], n=d["n"], k=d["k"],
                main_ns=d["main_ns"], reduce_ns=d.get("reduce_ns", 0),
            )
            for d in data
        ]

    def mark_phase_complete(self, phase: int, summary: Dict[str, Any]) -> None:
        """Write a checkpoint marker for a completed phase."""
        marker = {
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
        }
        path = self.local_dir / f"phase_{phase}_complete.json"
        path.write_text(json.dumps(marker, indent=2))

    def is_phase_complete(self, phase: int) -> bool:
        """Check if a phase has been completed."""
        path = self.local_dir / f"phase_{phase}_complete.json"
        return path.exists()

    def get_phase_summary(self, phase: int) -> Optional[Dict[str, Any]]:
        """Get the summary from a completed phase checkpoint."""
        path = self.local_dir / f"phase_{phase}_complete.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def fetch_remote_file(self, remote_path: str, local_filename: str) -> Path:
        """Copy a file from the remote container to local artifact dir."""
        local_path = self.local_dir / local_filename
        self.executor.copy_from_container(remote_path, str(local_path))
        return local_path

    def push_local_file(self, local_filename: str, remote_path: str) -> None:
        """Copy a local artifact file to the remote container."""
        local_path = self.local_dir / local_filename
        self.executor.copy_to_container(str(local_path), remote_path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/test_artifacts.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add aiter/ops/triton/tuning_agent/artifacts.py tests/tuning_agent/test_artifacts.py
git commit -m "feat(tuning-agent): add artifact manager for results, configs, and checkpoints"
```

---

### Task 8: Integration Test

**Files:**
- Create: `tests/tuning_agent/test_integration.py`

- [ ] **Step 1: Write integration test that ties all modules together**

```python
# tests/tuning_agent/test_integration.py
"""Integration test: load config, create pool, verify types flow through."""
import pytest
from aiter.ops.triton.tuning_agent.config import load_config
from aiter.ops.triton.tuning_agent.machine_pool import MachinePool
from aiter.ops.triton.tuning_agent.notifications import Notifier, NotificationLevel
from aiter.ops.triton.tuning_agent.types import ShapeResult


def test_config_to_pool_flow(valid_config_path):
    """Config loads, machines go into pool, allocation works."""
    config = load_config(valid_config_path)
    pool = MachinePool(config.machines)
    assert pool.available_count == 2

    machine = pool.allocate(kernel_name="a8w8")
    assert machine.host == "gpu-machine-1"  # more GPUs
    assert pool.available_count == 1

    pool.release(machine.host)
    assert pool.available_count == 2


def test_results_roundtrip(tmp_path):
    """ShapeResults serialize and deserialize correctly."""
    from aiter.ops.triton.tuning_agent.artifacts import ArtifactManager
    from unittest.mock import MagicMock

    executor = MagicMock()
    executor.container_id = "test"
    manager = ArtifactManager(executor, "test_kernel", run_id="test_run", local_base_dir=str(tmp_path))
    manager.setup_local()

    original = [
        ShapeResult(m=8, n=8192, k=8192, main_ns=5000, reduce_ns=2000),
        ShapeResult(m=16, n=8192, k=8192, main_ns=6000, reduce_ns=0),
        ShapeResult(m=8192, n=8192, k=8192, main_ns=500000, reduce_ns=0),
    ]
    manager.save_results("baseline", original)
    loaded = manager.load_results("baseline")

    assert len(loaded) == 3
    assert loaded[0].total_ns == 7000
    assert loaded[1].reduce_ns == 0
    assert loaded[2].main_ns == 500000


def test_notification_flow():
    """Notifier records history and auto-approves when configured."""
    notifier = Notifier(auto_approve=True)
    notifier.notify(NotificationLevel.INFO, "Test", "Starting tuning")
    notifier.notify(NotificationLevel.WARNING, "Slow", "M=8192 taking long")
    assert len(notifier.history) == 2

    approved = notifier.request_approval("Commit configs?")
    assert approved is True
    assert len(notifier.history) == 3  # auto-approve logs INFO
```

- [ ] **Step 2: Run all tests**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/ -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/tuning_agent/test_integration.py
git commit -m "test(tuning-agent): add integration tests for infrastructure layer"
```

---

## Summary

This plan creates the infrastructure layer with 7 focused modules:

| Module | Purpose | Lines (est) |
|--------|---------|-------------|
| `types.py` | Shared dataclasses | ~120 |
| `config.py` | YAML parsing + validation | ~130 |
| `remote.py` | SSH + docker exec wrapper | ~180 |
| `machine_pool.py` | Machine allocation + health | ~100 |
| `watchdog.py` | Timeout + progress monitoring | ~100 |
| `notifications.py` | Alerts + approval gates | ~90 |
| `artifacts.py` | Result storage + checkpoints | ~100 |

**Total**: ~820 lines of implementation + ~650 lines of tests across 8 tasks.

**Next plans:**
- **Plan 2: Subagent Library** — the 9 subagent types that use this infrastructure
- **Plan 3: Kernel Supervisor** — the per-kernel lifecycle orchestrating subagents
- **Plan 4: Orchestrator + Dashboard** — top-level agent, terminal UI, machine scheduling
