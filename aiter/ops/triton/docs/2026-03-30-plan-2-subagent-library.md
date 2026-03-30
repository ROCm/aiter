# Agentic Kernel Tuning Pipeline — Plan 2: Subagent Library

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the 9 subagent types that implement the kernel tuning lifecycle. Each subagent is a focused Python module with a clean interface (inputs, outputs, key behavior) that uses Plan 1 infrastructure (RemoteExecutor, ArtifactManager, Watchdog).

**Architecture:** Each subagent lives in `aiter/ops/triton/tuning_agent/subagents/` as a Python module. All subagents inherit from a `BaseSubagent` that provides environment verification, stale process cleanup, structured logging, and timeout enforcement. Subagents run locally, execute remotely via `RemoteExecutor.docker_exec()`.

**Tech Stack:** Python 3.8+, pytest, unittest.mock (for SSH/docker mocking)

**Spec:** `/app/aiter/aiter/ops/triton/docs/2026-03-30-agentic-kernel-tuning-pipeline-design.md`

**Depends on:** Plan 1 (Infrastructure Layer)

**Enables:** Plan 3 (Kernel Supervisor), Plan 4 (Orchestrator + UI)

---

## File Structure

```
aiter/ops/triton/tuning_agent/
├── subagents/
│   ├── __init__.py
│   ├── base.py                  # BaseSubagent ABC
│   ├── setup.py                 # SetupAgent — container + repo + Triton install
│   ├── discovery.py             # DiscoveryAgent — scan kernels, shapes, scripts
│   ├── script_creator.py        # ScriptCreatorAgent — generate missing ut/bench/test scripts
│   ├── baseline.py              # BaselineAgent — rocprof --stats collection
│   ├── tuning.py                # TuningAgent — screen.py execution + monitoring
│   ├── pattern_analyzer.py      # PatternAnalyzerAgent — scout analysis → narrowed search
│   ├── config_generator.py      # ConfigGeneratorAgent — view-screen.py + naming
│   ├── validation.py            # ValidationAgent — 3-way comparison + classification
│   └── regression_fixer.py      # RegressionFixerAgent — restore/promote configs

tests/tuning_agent/
├── test_base_subagent.py
├── test_baseline.py
├── test_tuning.py
├── test_regression_fixer.py
└── fixtures/
    ├── sample_stats.csv         # Mock rocprof output
    ├── sample_screen_log.txt    # Mock screen.py output
    ├── sample_config.json       # Mock GEMM config
    └── sample_baseline.json     # Mock baseline results
```

---

## Chunk 1: Base Subagent + Types

### Task 1: BaseSubagent ABC and Subagent Result Types

**Files:**
- Create: `aiter/ops/triton/tuning_agent/subagents/__init__.py`
- Create: `aiter/ops/triton/tuning_agent/subagents/base.py`
- Create: `tests/tuning_agent/test_base_subagent.py`

- [ ] **Step 1: Write failing tests for BaseSubagent**

```python
# tests/tuning_agent/test_base_subagent.py
import pytest
from unittest.mock import patch, MagicMock
from aiter.ops.triton.tuning_agent.subagents.base import (
    BaseSubagent,
    SubagentResult,
    SubagentError,
)
from aiter.ops.triton.tuning_agent.remote import RemoteExecutor
from aiter.ops.triton.tuning_agent.types import MachineInfo


class ConcreteSubagent(BaseSubagent):
    """Test-only concrete subclass."""
    name = "test_agent"

    def _execute(self) -> dict:
        return {"status": "done"}


@pytest.fixture
def machine():
    return MachineInfo(host="gpu1", user="root", ssh_key="~/.ssh/id", gpus=[0, 1])


@pytest.fixture
def executor(machine):
    ex = RemoteExecutor(machine)
    ex.container_id = "test_container"
    return ex


@pytest.fixture
def agent(executor):
    return ConcreteSubagent(
        executor=executor,
        kernel_name="a8w8",
        artifact_dir="/workspace/tuning_artifacts/a8w8",
    )


class TestSubagentResult:
    def test_success_result(self):
        r = SubagentResult(success=True, data={"shapes": 10})
        assert r.success
        assert r.data["shapes"] == 10
        assert r.error is None

    def test_failure_result(self):
        r = SubagentResult(success=False, error="rocprof failed")
        assert not r.success
        assert "rocprof" in r.error


class TestBaseSubagent:
    @patch("subprocess.run")
    def test_run_calls_preflight_and_execute(self, mock_run, agent):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = agent.run()
        assert result.success
        assert result.data["status"] == "done"

    @patch("subprocess.run")
    def test_preflight_calls_rocm_smi_and_mkdir(self, mock_run, agent):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        agent.run()
        calls = [str(c) for c in mock_run.call_args_list]
        assert any("rocm-smi" in c for c in calls)
        assert any("mkdir" in c for c in calls)

    def test_run_catches_exceptions_returns_failure(self, agent):
        agent._execute = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        result = agent.run()
        assert not result.success
        assert "boom" in result.error
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/test_base_subagent.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement BaseSubagent**

```python
# aiter/ops/triton/tuning_agent/subagents/__init__.py
"""Subagent library for the agentic Triton kernel tuning pipeline."""

# aiter/ops/triton/tuning_agent/subagents/base.py
"""Base class for all subagents."""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..remote import RemoteExecutor, RemoteCommandError

logger = logging.getLogger(__name__)


@dataclass
class SubagentResult:
    """Standardized subagent output."""
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class SubagentError(Exception):
    """Raised when a subagent encounters a fatal error."""
    pass


class BaseSubagent(ABC):
    """
    Abstract base for all subagents.

    Provides:
    - Pre-flight checks (environment verification, stale process cleanup)
    - Artifact directory setup
    - Structured result wrapping
    - Exception-safe run()
    """
    name: str = "base"

    def __init__(
        self,
        executor: RemoteExecutor,
        kernel_name: str,
        artifact_dir: str,
        expected_triton_commit: Optional[str] = None,
        expected_aiter_branch: Optional[str] = None,
    ):
        self.executor = executor
        self.kernel_name = kernel_name
        self.artifact_dir = artifact_dir
        self.expected_triton_commit = expected_triton_commit
        self.expected_aiter_branch = expected_aiter_branch

    def run(self) -> SubagentResult:
        """Run the subagent: preflight checks → execute → wrap result."""
        try:
            self._preflight()
            data = self._execute()
            return SubagentResult(success=True, data=data or {})
        except Exception as e:
            logger.error(f"[{self.name}] Failed: {e}")
            return SubagentResult(success=False, error=str(e))

    def _preflight(self) -> None:
        """Pre-flight checks common to all subagents."""
        # Ensure artifact directory exists
        self.executor.docker_exec(
            f"mkdir -p {self.artifact_dir}", check=True, timeout=10
        )
        # Kill stale GPU processes
        self.executor.kill_stale_gpu_processes()
        # Verify environment if expected values are set
        if self.expected_triton_commit or self.expected_aiter_branch:
            self.executor.verify_environment(
                expected_triton_commit=self.expected_triton_commit,
                expected_aiter_branch=self.expected_aiter_branch,
            )

    @abstractmethod
    def _execute(self) -> dict:
        """Subclass implements the actual work. Returns data dict for SubagentResult."""
        ...

    def _write_json_artifact(self, filename: str, data: dict) -> str:
        """Write a JSON artifact to the artifact directory. Returns remote path."""
        import json
        remote_path = f"{self.artifact_dir}/{filename}"
        json_str = json.dumps(data, indent=2)
        # Escape for shell
        escaped = json_str.replace("'", "'\\''")
        self.executor.docker_exec(
            f"printf '%s' '{escaped}' > {remote_path}",
            check=True, timeout=15,
        )
        return remote_path

    def _read_json_artifact(self, filename: str) -> dict:
        """Read a JSON artifact from the artifact directory."""
        import json
        remote_path = f"{self.artifact_dir}/{filename}"
        result = self.executor.docker_exec(
            f"cat {remote_path}", check=True, timeout=15,
        )
        return json.loads(result.stdout)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/test_base_subagent.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add aiter/ops/triton/tuning_agent/subagents/__init__.py \
    aiter/ops/triton/tuning_agent/subagents/base.py \
    tests/tuning_agent/test_base_subagent.py
git commit -m "feat(tuning-agent): add BaseSubagent ABC with preflight checks"
```

---

## Chunk 2: Skeleton Subagents (6 agents with interface only)

### Task 2: SetupAgent

**Files:**
- Create: `aiter/ops/triton/tuning_agent/subagents/setup.py`

**Input**: `RepoConfig`, `ContainerConfig`, `TritonInstallConfig`
**Output**: `{"container_id": str, "triton_version": str, "aiter_commit": str}`
**Key behavior**: Idempotent — reuses existing container. Handles initial setup and Triton version switching. Overrides `_preflight` to skip env verification (since it's setting up the env).

- [ ] **Step 1: Implement SetupAgent skeleton**

```python
# aiter/ops/triton/tuning_agent/subagents/setup.py
"""SetupAgent — container creation, repo cloning, Triton installation."""
import logging
from typing import Optional
from .base import BaseSubagent
from ..remote import RemoteExecutor
from ..types import RepoConfig, ContainerConfig, TritonInstallConfig

logger = logging.getLogger(__name__)

class SetupAgent(BaseSubagent):
    """Handles environment setup: container, repo clone, Triton install. Idempotent."""
    name = "setup"

    def __init__(self, executor, kernel_name, artifact_dir, repo_config: RepoConfig,
                 container_config: ContainerConfig, triton_install: TritonInstallConfig,
                 container_name: Optional[str] = None):
        super().__init__(executor, kernel_name, artifact_dir)
        self.repo_config = repo_config
        self.container_config = container_config
        self.triton_install = triton_install
        self.container_name = container_name

    def _preflight(self):
        self.executor.kill_stale_gpu_processes()

    def _execute(self) -> dict:
        # TODO: 1. Check/create container  2. Clone aiter+triton repos
        # 3. Install Triton  4. mkdir artifact_dir  5. Verify versions
        raise NotImplementedError("SetupAgent._execute")
```

- [ ] **Step 2: Commit**

```bash
git add aiter/ops/triton/tuning_agent/subagents/setup.py
git commit -m "feat(tuning-agent): add SetupAgent skeleton"
```

---

### Task 3: DiscoveryAgent

**Files:**
- Create: `aiter/ops/triton/tuning_agent/subagents/discovery.py`

**Input**: `kernel_name`, `aiter_root`, `gfx_arch`
**Output**: `{"shapes": [(M,N,K)...], "config_files": [...], "missing_scripts": [...], "fallback_shape": (N,K), "variant_prefix": str, "category": str}`
**Key behavior**: Read-only. Scans basic/, batched/, feed_forward/, fused/ GEMM categories. Derives VARIANT prefix from existing config files.

- [ ] **Step 1: Implement DiscoveryAgent skeleton**

```python
# aiter/ops/triton/tuning_agent/subagents/discovery.py
"""DiscoveryAgent — scan kernel source, configs, shapes, and scripts."""
import logging
from .base import BaseSubagent

logger = logging.getLogger(__name__)

GEMM_CATEGORIES = {
    "basic": "aiter/ops/triton/gemm/basic/",
    "batched": "aiter/ops/triton/gemm/batched/",
    "feed_forward": "aiter/ops/triton/gemm/feed_forward/",
    "fused": "aiter/ops/triton/gemm/fused/",
}

class DiscoveryAgent(BaseSubagent):
    """Scans codebase for kernel shapes, config files, and scripts. Read-only."""
    name = "discovery"

    def __init__(self, executor, kernel_name, artifact_dir,
                 aiter_root="/workspace/aiter", gfx_arch="gfx950", **kwargs):
        super().__init__(executor, kernel_name, artifact_dir, **kwargs)
        self.aiter_root = aiter_root
        self.gfx_arch = gfx_arch

    def _execute(self) -> dict:
        # TODO: 1. Determine GEMM category  2. Find kernel source
        # 3. Find config files (parse N,K from filenames, M buckets from contents)
        # 4. Load model_shapes.json  5. Check ut_*/bench_*/test_* scripts
        # 6. Identify fallback config  7. Write discovery.json
        raise NotImplementedError("DiscoveryAgent._execute")
```

- [ ] **Step 2: Commit**

```bash
git add aiter/ops/triton/tuning_agent/subagents/discovery.py
git commit -m "feat(tuning-agent): add DiscoveryAgent skeleton"
```

---

### Task 4: ScriptCreatorAgent

**Files:**
- Create: `aiter/ops/triton/tuning_agent/subagents/script_creator.py`

**Input**: `kernel_source_path`, `missing_scripts: List[str]`, `template_scripts: Dict[str, str]`, `category`
**Output**: `{"created_scripts": [...], "smoke_test_passed": bool, "escalation": Optional[str]}`
**Key behavior**: Reads kernel API (signatures, dtypes, packing). Adapts templates per category (batched=B dim, feed_forward=ff_* naming, fused=dual configs). Escalates ambiguity.

- [ ] **Step 1: Implement ScriptCreatorAgent skeleton**

```python
# aiter/ops/triton/tuning_agent/subagents/script_creator.py
"""ScriptCreatorAgent — generate missing ut/bench/test scripts."""
import logging
from typing import Dict, List
from .base import BaseSubagent

logger = logging.getLogger(__name__)

class ScriptCreatorAgent(BaseSubagent):
    """Generates missing scripts by reading kernel API and adapting templates."""
    name = "script_creator"

    def __init__(self, executor, kernel_name, artifact_dir, kernel_source_path: str,
                 missing_scripts: List[str], template_scripts: Dict[str, str],
                 category: str = "basic", **kwargs):
        super().__init__(executor, kernel_name, artifact_dir, **kwargs)
        self.kernel_source_path = kernel_source_path
        self.missing_scripts = missing_scripts
        self.template_scripts = template_scripts
        self.category = category

    def _execute(self) -> dict:
        # TODO: 1. Read kernel source (signature, dtypes, config params)
        # 2. For each missing type: read template, adapt for kernel API + category
        # 3. Smoke test one shape  4. Set escalation if ambiguous
        raise NotImplementedError("ScriptCreatorAgent._execute")
```

- [ ] **Step 2: Commit**

```bash
git add aiter/ops/triton/tuning_agent/subagents/script_creator.py
git commit -m "feat(tuning-agent): add ScriptCreatorAgent skeleton"
```

---

### Task 5: PatternAnalyzerAgent

**Files:**
- Create: `aiter/ops/triton/tuning_agent/subagents/pattern_analyzer.py`

**Input**: `scout_log_dir`, optional `history_dir`
**Output**: `{"narrowed_search_space": {m_range: {param: [values]}}, "time_savings_pct": float}`
**Key behavior**: Scout=1.0x weight, historical=0.25x. Keep values in >20% of winning configs. Sanity: narrowed >= 25% of broad space.

- [ ] **Step 1: Implement PatternAnalyzerAgent skeleton**

```python
# aiter/ops/triton/tuning_agent/subagents/pattern_analyzer.py
"""PatternAnalyzerAgent — analyze scout results to narrow tuning search space."""
import logging
from typing import Optional
from .base import BaseSubagent

logger = logging.getLogger(__name__)

M_RANGES = [("M_LEQ_16", 1, 16), ("M_17_64", 17, 64),
            ("M_65_256", 65, 256), ("M_GT_256", 257, 999999)]
TUNING_PARAMS = ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K",
                 "num_stages", "matrix_instr_nonkdim", "num_warps",
                 "waves_per_eu", "kpack", "num_ksplit"]
MIN_APPEARANCE_FRACTION = 0.20
HISTORICAL_WEIGHT = 0.25
MIN_SPACE_FRACTION = 0.25

class PatternAnalyzerAgent(BaseSubagent):
    """Analyzes scout results to produce narrowed search space per M range."""
    name = "pattern_analyzer"

    def __init__(self, executor, kernel_name, artifact_dir,
                 scout_log_dir: str, history_dir: Optional[str] = None, **kwargs):
        super().__init__(executor, kernel_name, artifact_dir, **kwargs)
        self.scout_log_dir = scout_log_dir
        self.history_dir = history_dir

    def _execute(self) -> dict:
        # TODO: 1. Parse screen-*.log → top-3 configs per shape
        # 2. Bucket by M range  3. Count weighted param frequencies
        # 4. Add historical at 0.25x  5. Keep values >= 20% frequency
        # 6. Sanity check >= 25% of broad  7. Write patterns.json
        raise NotImplementedError("PatternAnalyzerAgent._execute")
```

- [ ] **Step 2: Commit**

```bash
git add aiter/ops/triton/tuning_agent/subagents/pattern_analyzer.py
git commit -m "feat(tuning-agent): add PatternAnalyzerAgent skeleton"
```

---

### Task 6: ConfigGeneratorAgent

**Files:**
- Create: `aiter/ops/triton/tuning_agent/subagents/config_generator.py`

**Input**: `tuning_log_dir`, `ut_script`, `nk_pairs`, `config_dir`, `variant_prefix`, `gfx_arch`, optional `m_leq_bucket_name`
**Output**: `{"config_files": [...], "fallback_updated": bool}`
**Key behavior**: Runs view-screen.py per N,K pair. Applies M_LEQ rename if needed. For fallback shape: only copies `any` bucket, never overwrites other buckets.

- [ ] **Step 1: Implement ConfigGeneratorAgent skeleton**

```python
# aiter/ops/triton/tuning_agent/subagents/config_generator.py
"""ConfigGeneratorAgent — run view-screen.py and place config files."""
import logging
from typing import List, Optional, Tuple
from .base import BaseSubagent

logger = logging.getLogger(__name__)

class ConfigGeneratorAgent(BaseSubagent):
    """Generates JSON config files from tuning results using view-screen.py."""
    name = "config_generator"

    def __init__(self, executor, kernel_name, artifact_dir, tuning_log_dir: str,
                 ut_script: str, nk_pairs: List[Tuple[int, int]], config_dir: str,
                 variant_prefix: str, gfx_arch: str = "gfx950",
                 m_leq_bucket_name: Optional[str] = None, **kwargs):
        super().__init__(executor, kernel_name, artifact_dir, **kwargs)
        self.tuning_log_dir = tuning_log_dir
        self.ut_script = ut_script
        self.nk_pairs = nk_pairs
        self.config_dir = config_dir
        self.variant_prefix = variant_prefix
        self.gfx_arch = gfx_arch
        self.m_leq_bucket_name = m_leq_bucket_name

    def _execute(self) -> dict:
        # TODO: 1. cd tuning_log_dir  2. Run view-screen.py per (N,K)
        # 3. Rename M_LEQ_16→M_LEQ_31 if m_leq_bucket_name set
        # 4. For fallback: only copy "any" bucket  5. Copy to config_dir
        raise NotImplementedError("ConfigGeneratorAgent._execute")
```

- [ ] **Step 2: Commit**

```bash
git add aiter/ops/triton/tuning_agent/subagents/config_generator.py
git commit -m "feat(tuning-agent): add ConfigGeneratorAgent skeleton"
```

---

### Task 7: ValidationAgent

**Files:**
- Create: `aiter/ops/triton/tuning_agent/subagents/validation.py`

**Input**: `shapes`, `bench_script`, `gpu_ids`, `baseline_data`, `thresholds`, optional `untuned_data`
**Output**: `{"results": [...], "regressions": [...], "geomean_speedup": float}`
**Key behavior**: Parallelizes across GPUs with unique `-o` prefixes. 3-way classification: IMPROVED, COMPILER_REGRESSION, TUNING_REGRESSION, TUNING_REGRESSION_SEVERE, NEUTRAL. Uses same rocprof parsing as BaselineAgent.

- [ ] **Step 1: Implement ValidationAgent skeleton**

```python
# aiter/ops/triton/tuning_agent/subagents/validation.py
"""ValidationAgent — collect timings and classify regressions."""
import logging
from typing import Dict, List, Optional, Tuple
from .base import BaseSubagent
from ..types import ShapeResult, TuningThresholds

logger = logging.getLogger(__name__)

class RegressionClassification:
    IMPROVED = "improved"
    NEUTRAL = "neutral"
    COMPILER_REGRESSION = "compiler_regression"
    TUNING_REGRESSION = "tuning_regression"
    TUNING_REGRESSION_SEVERE = "tuning_regression_severe"

class ValidationAgent(BaseSubagent):
    """Collects timings via rocprof --stats and performs 3-way comparison."""
    name = "validation"

    def __init__(self, executor, kernel_name, artifact_dir,
                 shapes: List[Tuple[int,int,int]], bench_script: str,
                 gpu_ids: List[int], baseline_data: Dict[Tuple[int,int,int], ShapeResult],
                 thresholds: TuningThresholds,
                 untuned_data: Optional[Dict[Tuple[int,int,int], ShapeResult]] = None,
                 output_filename: str = "tuned.json", **kwargs):
        super().__init__(executor, kernel_name, artifact_dir, **kwargs)
        self.shapes = shapes
        self.bench_script = bench_script
        self.gpu_ids = gpu_ids
        self.baseline_data = baseline_data
        self.untuned_data = untuned_data
        self.thresholds = thresholds
        self.output_filename = output_filename

    def _execute(self) -> dict:
        # TODO: 1. Distribute shapes across GPUs  2. rocprof --stats per shape
        # 3. Parse .stats.csv  4. Compare baseline/untuned/tuned
        # 5. Classify per RegressionClassification  6. Compute geomean
        raise NotImplementedError("ValidationAgent._execute")
```

- [ ] **Step 2: Commit**

```bash
git add aiter/ops/triton/tuning_agent/subagents/validation.py
git commit -m "feat(tuning-agent): add ValidationAgent skeleton"
```

---

## Chunk 3: BaselineAgent (Full Implementation)

### Task 8: Test Fixtures

**Files:**
- Create: `tests/tuning_agent/fixtures/sample_stats.csv`
- Create: `tests/tuning_agent/fixtures/sample_baseline.json`

- [ ] **Step 1: Create test fixtures**

Create `tests/tuning_agent/fixtures/sample_stats.csv` with standard rocprof --stats output (Name, Calls, TotalDurationNs, AverageNs, Percentage columns) containing a main kernel row and a reduce kernel row for `a8w8`.

Create `tests/tuning_agent/fixtures/sample_baseline.json` with 2 shapes: (1,8192,8192) with split-K reduce and (128,8192,8192) without reduce.

- [ ] **Step 2: Commit fixtures**

### Task 9: BaselineAgent Tests

**Files:**
- Create: `tests/tuning_agent/test_baseline.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/tuning_agent/test_baseline.py
import json
import pytest
from unittest.mock import patch, MagicMock, call
from aiter.ops.triton.tuning_agent.subagents.baseline import BaselineAgent, parse_stats_csv
from aiter.ops.triton.tuning_agent.remote import RemoteExecutor
from aiter.ops.triton.tuning_agent.types import MachineInfo, ShapeResult


SAMPLE_STATS_CSV = """\
"Name","Calls","TotalDurationNs","AverageNs","Percentage"
"gemm_a8w8_kernel_0",100,500000,5000,85.2
"gemm_a8w8_reduce_kernel",100,100000,1000,14.8
"""

SAMPLE_STATS_CSV_NO_REDUCE = """\
"Name","Calls","TotalDurationNs","AverageNs","Percentage"
"gemm_a8w8_kernel_0",100,1500000,15000,100.0
"""

SAMPLE_STATS_CSV_WITH_NOISE = """\
"Name","Calls","TotalDurationNs","AverageNs","Percentage"
"__amd_rocclr_fillBuffer.kd",1,10000,10000,0.5
"gemm_a8w8_kernel_0",100,500000,5000,85.2
"gemm_a8w8_reduce_kernel",100,100000,1000,14.3
"""


@pytest.fixture
def machine():
    return MachineInfo(host="gpu1", user="root", ssh_key="~/.ssh/id", gpus=[0, 1])


@pytest.fixture
def executor(machine):
    ex = RemoteExecutor(machine)
    ex.container_id = "test_ctr"
    return ex


class TestParseStatsCsv:
    def test_parse_main_and_reduce_kernels(self):
        result = parse_stats_csv(SAMPLE_STATS_CSV, kernel_variant="a8w8")
        assert result["main_ns"] == 5000
        assert result["reduce_ns"] == 1000

    def test_parse_no_reduce_kernel(self):
        result = parse_stats_csv(SAMPLE_STATS_CSV_NO_REDUCE, kernel_variant="a8w8")
        assert result["main_ns"] == 15000
        assert result["reduce_ns"] == 0

    def test_parse_filters_noise_kernels(self):
        result = parse_stats_csv(SAMPLE_STATS_CSV_WITH_NOISE, kernel_variant="a8w8")
        assert result["main_ns"] == 5000
        assert result["reduce_ns"] == 1000

    def test_parse_empty_csv_raises(self):
        with pytest.raises(ValueError, match="No kernel"):
            parse_stats_csv('"Name","Calls","TotalDurationNs","AverageNs","Percentage"\n',
                          kernel_variant="a8w8")

    def test_parse_returns_ints(self):
        result = parse_stats_csv(SAMPLE_STATS_CSV, kernel_variant="a8w8")
        assert isinstance(result["main_ns"], int)
        assert isinstance(result["reduce_ns"], int)


class TestBaselineAgent:
    @patch("subprocess.run")
    def test_collects_all_shapes_sequentially(self, mock_run, executor):
        shapes = [(1, 8192, 8192), (128, 8192, 8192)]

        # Mock: preflight calls succeed, then for each shape:
        # 1. rocprof command, 2. cat .stats.csv, 3. cleanup
        def side_effect(*args, **kwargs):
            cmd_str = " ".join(args[0]) if isinstance(args[0], list) else str(args[0])
            if "rocm-smi" in cmd_str:
                return MagicMock(returncode=0, stdout="", stderr="")
            if "mkdir" in cmd_str:
                return MagicMock(returncode=0, stdout="", stderr="")
            if "rocprof" in cmd_str:
                return MagicMock(returncode=0, stdout="", stderr="")
            if "cat" in cmd_str and ".stats.csv" in cmd_str:
                return MagicMock(returncode=0, stdout=SAMPLE_STATS_CSV, stderr="")
            if "rm " in cmd_str:
                return MagicMock(returncode=0, stdout="", stderr="")
            if "printf" in cmd_str or ">" in cmd_str:
                return MagicMock(returncode=0, stdout="", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = side_effect

        agent = BaselineAgent(
            executor=executor,
            kernel_name="a8w8",
            artifact_dir="/workspace/tuning_artifacts/a8w8",
            shapes=shapes,
            bench_script="/workspace/aiter/op_tests/op_benchmarks/triton/bench_gemm_a8w8.py",
            gpu_id=0,
        )
        result = agent.run()
        assert result.success
        assert len(result.data["shapes"]) == 2

    @patch("subprocess.run")
    def test_uses_single_gpu(self, mock_run, executor):
        """Baseline MUST run on a single GPU — no parallelism."""
        mock_run.return_value = MagicMock(returncode=0, stdout=SAMPLE_STATS_CSV, stderr="")

        agent = BaselineAgent(
            executor=executor,
            kernel_name="a8w8",
            artifact_dir="/workspace/tuning_artifacts/a8w8",
            shapes=[(1, 8192, 8192)],
            bench_script="bench_gemm_a8w8.py",
            gpu_id=2,
        )
        result = agent.run()

        # Check that HIP_VISIBLE_DEVICES=2 was used
        calls_str = " ".join(str(c) for c in mock_run.call_args_list)
        assert "HIP_VISIBLE_DEVICES" in calls_str or "ROCR_VISIBLE_DEVICES" in calls_str

    @patch("subprocess.run")
    def test_handles_rocprof_failure(self, mock_run, executor):
        def side_effect(*args, **kwargs):
            cmd_str = " ".join(args[0]) if isinstance(args[0], list) else str(args[0])
            if "rocprof" in cmd_str:
                return MagicMock(returncode=1, stdout="", stderr="rocprof error")
            return MagicMock(returncode=0, stdout="", stderr="")
        mock_run.side_effect = side_effect
        agent = BaselineAgent(executor=executor, kernel_name="a8w8",
            artifact_dir="/workspace/tuning_artifacts/a8w8",
            shapes=[(1, 8192, 8192)], bench_script="bench_gemm_a8w8.py", gpu_id=0)
        result = agent.run()
        assert not result.success or len(result.data.get("failures", [])) > 0

    @patch("subprocess.run")
    def test_skips_already_collected_shapes(self, mock_run, executor):
        """Restartability: shapes already in baseline.json are skipped."""
        existing = json.dumps({"kernel": "a8w8", "shapes": [
            {"m": 1, "n": 8192, "k": 8192, "main_ns": 5000, "reduce_ns": 1000}]})
        def side_effect(*args, **kwargs):
            cmd_str = " ".join(args[0]) if isinstance(args[0], list) else str(args[0])
            if "cat" in cmd_str and "baseline" in cmd_str:
                return MagicMock(returncode=0, stdout=existing, stderr="")
            if "cat" in cmd_str and ".stats.csv" in cmd_str:
                return MagicMock(returncode=0, stdout=SAMPLE_STATS_CSV, stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")
        mock_run.side_effect = side_effect
        agent = BaselineAgent(executor=executor, kernel_name="a8w8",
            artifact_dir="/workspace/tuning_artifacts/a8w8",
            shapes=[(1, 8192, 8192), (128, 8192, 8192)],
            bench_script="bench_gemm_a8w8.py", gpu_id=0)
        result = agent.run()
        assert result.success
        assert len(result.data["shapes"]) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/test_baseline.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Commit**

```bash
git add tests/tuning_agent/test_baseline.py
git commit -m "test(tuning-agent): add BaselineAgent tests"
```

---

### Task 10: BaselineAgent Implementation

**Files:**
- Create: `aiter/ops/triton/tuning_agent/subagents/baseline.py`

- [ ] **Step 1: Implement BaselineAgent**

```python
# aiter/ops/triton/tuning_agent/subagents/baseline.py
"""BaselineAgent — collect per-shape benchmark timings using rocprof --stats."""
import csv
import io
import json
import logging
from typing import Dict, List, Optional, Tuple

from .base import BaseSubagent, SubagentError
from ..remote import RemoteExecutor, RemoteCommandError
from ..types import ShapeResult

logger = logging.getLogger(__name__)

# Maximum consecutive failures before aborting
MAX_CONSECUTIVE_FAILURES = 3


def parse_stats_csv(csv_content: str, kernel_variant: str) -> Dict[str, int]:
    """
    Parse rocprof .stats.csv output to extract kernel timings.

    Looks for rows whose Name contains the kernel variant (e.g., 'a8w8').
    The main kernel row contains 'kernel' but not 'reduce'.
    The reduce kernel row contains 'reduce'.

    Returns: {"main_ns": int, "reduce_ns": int}
    Raises ValueError if no matching kernel rows found.
    """
    reader = csv.DictReader(io.StringIO(csv_content))
    main_ns = 0
    reduce_ns = 0
    found_any = False

    for row in reader:
        name = row.get("Name", "").strip().strip('"')
        avg_ns_str = row.get("AverageNs", "0").strip().strip('"')

        # Skip non-kernel rows (e.g., rocclr fillBuffer)
        if kernel_variant.lower() not in name.lower():
            continue

        found_any = True
        avg_ns = int(float(avg_ns_str))

        if "reduce" in name.lower():
            reduce_ns = avg_ns
        else:
            # Main kernel — take the one with highest AverageNs if multiple
            main_ns = max(main_ns, avg_ns)

    if not found_any:
        raise ValueError(
            f"No kernel rows matching variant '{kernel_variant}' found in stats CSV. "
            f"Available rows: {csv_content[:500]}"
        )

    return {"main_ns": main_ns, "reduce_ns": reduce_ns}


class BaselineAgent(BaseSubagent):
    """Collects baseline timings via rocprof --stats (v1). Sequential on single GPU.
    Restartable: skips shapes already in baseline.json."""
    name = "baseline"

    def __init__(
        self,
        executor: RemoteExecutor,
        kernel_name: str,
        artifact_dir: str,
        shapes: List[Tuple[int, int, int]],
        bench_script: str,
        gpu_id: int = 0,
        output_filename: str = "baseline.json",
        timeout_per_shape: int = 300,
        **kwargs,
    ):
        super().__init__(executor, kernel_name, artifact_dir, **kwargs)
        self.shapes = shapes
        self.bench_script = bench_script
        self.gpu_id = gpu_id
        self.output_filename = output_filename
        self.timeout_per_shape = timeout_per_shape

    def _load_existing_results(self) -> Dict[Tuple[int, int, int], dict]:
        """Load previously collected results for restartability."""
        try:
            data = self._read_json_artifact(self.output_filename)
            existing = {}
            for s in data.get("shapes", []):
                key = (s["m"], s["n"], s["k"])
                existing[key] = s
            logger.info(f"Loaded {len(existing)} existing results from {self.output_filename}")
            return existing
        except Exception:
            return {}

    def _collect_shape(self, m: int, n: int, k: int) -> Dict[str, int]:
        """Run rocprof --stats for a single shape. Returns parsed timings."""
        prefix = f"{self.artifact_dir}/rocprof_{self.kernel_name}_{m}_{n}_{k}"
        stats_file = f"{prefix}.stats.csv"

        rocprof_cmd = (
            f"cd /workspace/aiter && "
            f"HIP_VISIBLE_DEVICES={self.gpu_id} "
            f"rocprof --stats -o {prefix}.csv "
            f"python {self.bench_script} --shape {m} {n} {k} --metric time --layout TN"
        )

        try:
            self.executor.docker_exec(
                rocprof_cmd,
                timeout=self.timeout_per_shape,
                check=True,
            )
        except RemoteCommandError as e:
            raise SubagentError(f"rocprof failed for shape ({m},{n},{k}): {e}")

        # Read the .stats.csv file
        try:
            cat_result = self.executor.docker_exec(
                f"cat {stats_file}",
                timeout=15,
                check=True,
            )
        except RemoteCommandError as e:
            raise SubagentError(
                f"Failed to read stats file {stats_file}: {e}"
            )

        timings = parse_stats_csv(cat_result.stdout, kernel_variant=self.kernel_name)

        # Cleanup rocprof output files
        self.executor.docker_exec(
            f"rm -f {prefix}.csv {stats_file} {prefix}.*.txt",
            timeout=10,
            check=False,
        )

        return timings

    def _execute(self) -> dict:
        """Collect baseline timings for all shapes sequentially."""
        existing = self._load_existing_results()
        results = []
        failures = []
        skipped = []
        consecutive_failures = 0

        # Add existing results first
        for key, data in existing.items():
            results.append(data)
            skipped.append(list(key))

        for m, n, k in self.shapes:
            key = (m, n, k)
            if key in existing:
                continue

            logger.info(f"[{self.name}] Collecting shape ({m}, {n}, {k}) on GPU {self.gpu_id}")

            try:
                timings = self._collect_shape(m, n, k)
                shape_result = {
                    "m": m, "n": n, "k": k,
                    "main_ns": timings["main_ns"],
                    "reduce_ns": timings["reduce_ns"],
                    "total_ns": timings["main_ns"] + timings["reduce_ns"],
                }
                results.append(shape_result)
                consecutive_failures = 0

                # Write incremental results after each shape
                output_data = {"kernel": self.kernel_name, "shapes": results}
                self._write_json_artifact(self.output_filename, output_data)

            except (SubagentError, RemoteCommandError) as e:
                logger.error(f"[{self.name}] Failed shape ({m},{n},{k}): {e}")
                failures.append({"m": m, "n": n, "k": k, "error": str(e)})
                consecutive_failures += 1

                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    raise SubagentError(
                        f"Aborting: {MAX_CONSECUTIVE_FAILURES} consecutive failures. "
                        f"Last error: {e}"
                    )

        return {
            "shapes": results,
            "failures": failures,
            "skipped": skipped,
        }
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/test_baseline.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add aiter/ops/triton/tuning_agent/subagents/baseline.py
git commit -m "feat(tuning-agent): implement BaselineAgent with rocprof stats parsing"
```

---

## Chunk 4: TuningAgent (Full Implementation)

### Task 11: TuningAgent Tests

**Files:**
- Create: `tests/tuning_agent/test_tuning.py`
- Create: `tests/tuning_agent/fixtures/sample_screen_log.txt`

- [ ] **Step 1: Create `tests/tuning_agent/fixtures/sample_screen_log.txt`** with screencase lines and `Screen complete` marker.

- [ ] **Step 2: Write failing tests**

```python
# tests/tuning_agent/test_tuning.py
import pytest
from unittest.mock import patch, MagicMock, call
from aiter.ops.triton.tuning_agent.subagents.tuning import (
    TuningAgent,
    build_screen_command,
    distribute_shapes_to_gpus,
)
from aiter.ops.triton.tuning_agent.remote import RemoteExecutor
from aiter.ops.triton.tuning_agent.types import MachineInfo


SCREEN_LOG_CONTENT = """\
screening ut_gemm_a8w8.py M=1 N=8192 K=8192 GPU=0
screencase BM=4 BN=16 BK=512 stages=2 : time=5.2us
screencase BM=4 BN=32 BK=512 stages=2 : time=4.8us
Screen complete M=1 N=8192 K=8192 best=4.8us
"""


@pytest.fixture
def machine():
    return MachineInfo(host="gpu1", user="root", ssh_key="~/.ssh/id", gpus=[0, 1, 2, 3])


@pytest.fixture
def executor(machine):
    ex = RemoteExecutor(machine)
    ex.container_id = "test_ctr"
    return ex


class TestBuildScreenCommand:
    def test_basic_command(self):
        cmd = build_screen_command(
            m=1, n=8192, k=8192, gpu_id=0,
            ut_script="ut_gemm_a8w8.py",
        )
        assert "screen.py" in cmd
        assert "1 8192 8192 0" in cmd or "1 8192 8192" in cmd
        assert "ut_gemm_a8w8.py" in cmd

    def test_with_search_space(self):
        search_space = {
            "block_size_m_range": [4, 8, 16],
            "block_size_n_range": [16, 32],
            "block_size_k_range": [256, 512],
            "num_stages_range": [2, 3],
        }
        cmd = build_screen_command(
            m=1, n=8192, k=8192, gpu_id=0,
            ut_script="ut_gemm_a8w8.py",
            search_space=search_space,
        )
        assert "--block-size-m-range 4 8 16" in cmd
        assert "--block-size-n-range 16 32" in cmd

    def test_with_max_batch_env(self):
        cmd = build_screen_command(
            m=8192, n=8192, k=8192, gpu_id=0,
            ut_script="ut_gemm_a8w8.py",
            max_batch=10,
        )
        assert "SCREEN_MAX_BATCH=10" in cmd

    def test_overwrite_flag(self):
        cmd = build_screen_command(
            m=1, n=8192, k=8192, gpu_id=0,
            ut_script="ut_gemm_a8w8.py",
            overwrite=True,
        )
        assert "--overwrite" in cmd


class TestDistributeShapes:
    def test_round_robin_distribution(self):
        m_values = [1, 2, 4, 8, 16, 32, 64, 128]
        gpu_ids = [0, 1, 2, 3]
        assignment = distribute_shapes_to_gpus(m_values, gpu_ids)
        assert len(assignment) == 4  # one entry per GPU
        # Each GPU should get 2 M values
        assert all(len(ms) == 2 for ms in assignment.values())
        # All M values assigned
        all_ms = []
        for ms in assignment.values():
            all_ms.extend(ms)
        assert sorted(all_ms) == sorted(m_values)

    def test_uneven_distribution(self):
        m_values = [1, 2, 4, 8, 16]
        gpu_ids = [0, 1, 2, 3]
        assignment = distribute_shapes_to_gpus(m_values, gpu_ids)
        total = sum(len(ms) for ms in assignment.values())
        assert total == 5

    def test_single_gpu(self):
        m_values = [1, 2, 4]
        gpu_ids = [0]
        assignment = distribute_shapes_to_gpus(m_values, gpu_ids)
        assert len(assignment[0]) == 3


class TestTuningAgent:
    @patch("subprocess.run")
    def test_scout_phase_selects_representative_shapes(self, mock_run, executor):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        agent = TuningAgent(
            executor=executor,
            kernel_name="a8w8",
            artifact_dir="/workspace/tuning_artifacts/a8w8",
            shapes_to_tune=[(m, 8192, 8192) for m in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]],
            ut_script="ut_gemm_a8w8.py",
            gpu_ids=[0, 1, 2, 3],
            tuning_dir="/workspace/aiter/aiter/ops/triton/utils/_triton/tunning",
            scout_fraction=0.15,
            is_scout=True,
        )
        scout_shapes = agent._select_scout_shapes()
        # 15% of 10 shapes ≈ 2 shapes, but at least one small and one large M
        assert len(scout_shapes) >= 2
        ms = [s[0] for s in scout_shapes]
        assert min(ms) <= 8  # has a small M
        assert max(ms) >= 128  # has a large M

    @patch("subprocess.run")
    def test_large_m_uses_reduced_batch(self, mock_run, executor):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        agent = TuningAgent(executor=executor, kernel_name="a8w8",
            artifact_dir="/workspace/tuning_artifacts/a8w8",
            shapes_to_tune=[(8192, 8192, 8192)], ut_script="ut_gemm_a8w8.py",
            gpu_ids=[0], tuning_dir="/workspace/aiter/aiter/ops/triton/utils/_triton/tunning")
        agent.run()
        calls_str = " ".join(str(c) for c in mock_run.call_args_list)
        assert "SCREEN_MAX_BATCH" in calls_str

    @patch("subprocess.run")
    def test_skips_completed_shapes(self, mock_run, executor):
        def side_effect(*args, **kwargs):
            cmd_str = " ".join(args[0]) if isinstance(args[0], list) else str(args[0])
            if "grep" in cmd_str and "Screen complete" in cmd_str:
                return MagicMock(returncode=0, stdout="Screen complete M=1", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")
        mock_run.side_effect = side_effect
        agent = TuningAgent(executor=executor, kernel_name="a8w8",
            artifact_dir="/workspace/tuning_artifacts/a8w8",
            shapes_to_tune=[(1, 8192, 8192), (128, 8192, 8192)],
            ut_script="ut_gemm_a8w8.py", gpu_ids=[0],
            tuning_dir="/workspace/aiter/aiter/ops/triton/utils/_triton/tunning")
        result = agent.run()
        assert result.success
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/test_tuning.py -v`
Expected: FAIL — module not found

- [ ] **Step 4: Commit**

```bash
git add tests/tuning_agent/test_tuning.py tests/tuning_agent/fixtures/sample_screen_log.txt
git commit -m "test(tuning-agent): add TuningAgent tests"
```

---

### Task 12: TuningAgent Implementation

**Files:**
- Create: `aiter/ops/triton/tuning_agent/subagents/tuning.py`

- [ ] **Step 1: Implement TuningAgent**

```python
# aiter/ops/triton/tuning_agent/subagents/tuning.py
"""TuningAgent — run screen.py for kernel config tuning."""
import logging
import math
import time
from typing import Dict, List, Optional, Tuple

from .base import BaseSubagent, SubagentError
from ..remote import RemoteExecutor, RemoteCommandError

logger = logging.getLogger(__name__)

# Threshold M value above which we reduce batch size
LARGE_M_THRESHOLD = 4096
LARGE_M_MAX_BATCH = 10

# Default broad search space (used if no narrowed space from PatternAnalyzer)
DEFAULT_SEARCH_SPACE = {
    "block_size_m_range": [4, 8, 16, 32, 64, 128, 256],
    "block_size_n_range": [16, 32, 64, 128, 256],
    "block_size_k_range": [64, 128, 256, 512, 1024],
    "num_stages_range": [1, 4],  # min, max — screen.py uses range syntax
    "matrix_instr_nonkdim_range": [16, 32],
    "num_warps_range": [4, 8],
    "waves_per_eu_range": [0],
    "num_ksplit_range": [1, 2, 4],
}


def build_screen_command(
    m: int,
    n: int,
    k: int,
    gpu_id: int,
    ut_script: str,
    search_space: Optional[Dict[str, list]] = None,
    overwrite: bool = True,
    max_batch: Optional[int] = None,
) -> str:
    """
    Build a screen.py command string.

    Args:
        m, n, k: shape dimensions
        gpu_id: GPU to use
        ut_script: name of the ut_*.py script
        search_space: dict of param_name -> list of values
        overwrite: if True, add --overwrite flag
        max_batch: if set, prepend SCREEN_MAX_BATCH=N

    Returns:
        Full command string ready for docker_exec.
    """
    space = search_space or DEFAULT_SEARCH_SPACE
    parts = []

    # Environment variable for large M
    if max_batch:
        parts.append(f"SCREEN_MAX_BATCH={max_batch}")

    parts.append(f"python screen.py {m} {n} {k} {gpu_id} {ut_script}")

    # Add search space parameters
    param_map = {
        "block_size_m_range": "--block-size-m-range",
        "block_size_n_range": "--block-size-n-range",
        "block_size_k_range": "--block-size-k-range",
        "num_stages_range": "--num-stages-range",
        "matrix_instr_nonkdim_range": "--matrix-instr-nonkdim-range",
        "num_warps_range": "--num-warps-range",
        "waves_per_eu_range": "--waves-per-eu-range",
        "num_ksplit_range": "--num-ksplit-range",
    }
    for param_key, flag in param_map.items():
        if param_key in space:
            values = space[param_key]
            values_str = " ".join(str(v) for v in values)
            parts.append(f"{flag} {values_str}")

    if overwrite:
        parts.append("--overwrite")

    return " ".join(parts)


def distribute_shapes_to_gpus(
    m_values: List[int], gpu_ids: List[int]
) -> Dict[int, List[int]]:
    """
    Distribute M values across GPUs round-robin.

    Returns: dict of gpu_id -> list of M values.
    """
    assignment: Dict[int, List[int]] = {g: [] for g in gpu_ids}
    for i, m in enumerate(m_values):
        gpu = gpu_ids[i % len(gpu_ids)]
        assignment[gpu].append(m)
    return assignment


class TuningAgent(BaseSubagent):
    """Runs screen.py for kernel tuning. Distributes M across GPUs, processes N,K serially.
    Large M gets SCREEN_MAX_BATCH. Restartable: skips shapes with 'Screen complete' logs."""
    name = "tuning"

    def __init__(
        self,
        executor: RemoteExecutor,
        kernel_name: str,
        artifact_dir: str,
        shapes_to_tune: List[Tuple[int, int, int]],
        ut_script: str,
        gpu_ids: List[int],
        tuning_dir: str,
        search_space: Optional[Dict[str, list]] = None,
        is_scout: bool = False,
        scout_fraction: float = 0.15,
        progress_check_interval: int = 120,
        tuning_timeout_per_shape: int = 1800,
        **kwargs,
    ):
        super().__init__(executor, kernel_name, artifact_dir, **kwargs)
        self.shapes_to_tune = shapes_to_tune
        self.ut_script = ut_script
        self.gpu_ids = gpu_ids
        self.tuning_dir = tuning_dir
        self.search_space = search_space
        self.is_scout = is_scout
        self.scout_fraction = scout_fraction
        self.progress_check_interval = progress_check_interval
        self.tuning_timeout_per_shape = tuning_timeout_per_shape

    def _select_scout_shapes(self) -> List[Tuple[int, int, int]]:
        """Select representative subset: one small, one medium, one large M per (N,K)."""
        nk_groups: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}
        for m, n, k in self.shapes_to_tune:
            key = (n, k)
            nk_groups.setdefault(key, []).append((m, n, k))

        target_count = max(2, int(len(self.shapes_to_tune) * self.scout_fraction))
        selected = []

        for (n, k), shapes in nk_groups.items():
            sorted_shapes = sorted(shapes, key=lambda s: s[0])
            # Pick small, medium, large
            if len(sorted_shapes) >= 3:
                selected.append(sorted_shapes[0])                          # smallest M
                selected.append(sorted_shapes[len(sorted_shapes) // 2])   # middle M
                selected.append(sorted_shapes[-1])                         # largest M
            else:
                selected.extend(sorted_shapes)

        # Trim if we selected too many
        if len(selected) > target_count * 2:
            # Keep at least one small and one large, trim middles
            selected = selected[:target_count]

        return selected

    def _check_shape_complete(self, m: int, n: int, k: int) -> bool:
        """Check if a screen.py log exists with 'Screen complete' for this shape."""
        log_name = f"screen-{self.ut_script}-{m}-{n}-{k}.log"
        try:
            result = self.executor.docker_exec(
                f"grep -l 'Screen complete' {self.tuning_dir}/{log_name} 2>/dev/null",
                check=False, timeout=10,
            )
            return result.returncode == 0 and result.stdout.strip() != ""
        except Exception:
            return False

    def _run_screen_for_shape(self, m: int, n: int, k: int, gpu_id: int) -> bool:
        """Run screen.py for a single shape on a specific GPU. Returns success."""
        max_batch = LARGE_M_MAX_BATCH if m >= LARGE_M_THRESHOLD else None

        cmd = build_screen_command(
            m=m, n=n, k=k, gpu_id=gpu_id,
            ut_script=self.ut_script,
            search_space=self.search_space,
            overwrite=True,
            max_batch=max_batch,
        )

        full_cmd = f"cd {self.tuning_dir} && {cmd}"
        logger.info(f"[{self.name}] Running screen.py M={m} N={n} K={k} GPU={gpu_id}")

        try:
            self.executor.docker_exec(
                full_cmd,
                env={"HIP_VISIBLE_DEVICES": str(gpu_id)},
                timeout=self.tuning_timeout_per_shape,
                check=True,
            )
            return True
        except RemoteCommandError as e:
            logger.error(f"[{self.name}] screen.py failed M={m} N={n} K={k}: {e}")
            return False

    def _execute(self) -> dict:
        """Run screen.py for all shapes, distributed across GPUs."""
        shapes = self.shapes_to_tune
        if self.is_scout:
            shapes = self._select_scout_shapes()
            logger.info(f"[{self.name}] Scout phase: selected {len(shapes)} shapes")

        # Group by (N, K) to process one N,K pair at a time
        nk_groups: Dict[Tuple[int, int], List[int]] = {}
        for m, n, k in shapes:
            nk_groups.setdefault((n, k), []).append(m)

        tuned = []
        skipped = []
        failures = []

        for (n, k), m_values in nk_groups.items():
            logger.info(f"[{self.name}] Processing N={n} K={k} ({len(m_values)} M values)")

            # Check which M values are already complete
            remaining_ms = []
            for m in m_values:
                if self._check_shape_complete(m, n, k):
                    skipped.append((m, n, k))
                    tuned.append((m, n, k))
                else:
                    remaining_ms.append(m)

            if not remaining_ms:
                continue

            # Distribute remaining M values across GPUs
            assignment = distribute_shapes_to_gpus(remaining_ms, self.gpu_ids)

            # Process each GPU's M values sequentially
            # (screen.py is one-at-a-time per GPU; we serialize per N,K pair)
            for gpu_id, gpu_ms in assignment.items():
                for m in gpu_ms:
                    success = self._run_screen_for_shape(m, n, k, gpu_id)
                    if success:
                        tuned.append((m, n, k))
                    else:
                        failures.append((m, n, k))

        # Determine log directory
        log_dir = self.tuning_dir
        if self.is_scout:
            log_dir = f"{self.artifact_dir}/scout_results"
            # Copy logs to scout dir
            self.executor.docker_exec(
                f"mkdir -p {log_dir} && cp {self.tuning_dir}/screen-*.log {log_dir}/ 2>/dev/null || true",
                check=False, timeout=30,
            )

        return {
            "tuned_shapes": tuned,
            "log_dir": log_dir,
            "skipped": skipped,
            "failures": failures,
        }
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/test_tuning.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add aiter/ops/triton/tuning_agent/subagents/tuning.py
git commit -m "feat(tuning-agent): implement TuningAgent with screen.py integration"
```

---

## Chunk 5: RegressionFixerAgent (Full Implementation)

### Task 13: RegressionFixerAgent Tests

**Files:**
- Create: `tests/tuning_agent/test_regression_fixer.py`
- Create: `tests/tuning_agent/fixtures/sample_config.json`

- [ ] **Step 1: Create `tests/tuning_agent/fixtures/sample_config.json`** with M_LEQ_16, M_17_31, and `any` buckets containing standard GEMM config params.

- [ ] **Step 2: Write failing tests**

```python
# tests/tuning_agent/test_regression_fixer.py
import json
import pytest
from unittest.mock import patch, MagicMock
from aiter.ops.triton.tuning_agent.subagents.regression_fixer import (
    RegressionFixerAgent,
    RegressionInfo,
    fix_strategy,
    FixStrategy,
)
from aiter.ops.triton.tuning_agent.remote import RemoteExecutor
from aiter.ops.triton.tuning_agent.types import MachineInfo, TuningThresholds


@pytest.fixture
def machine():
    return MachineInfo(host="gpu1", user="root", ssh_key="~/.ssh/id", gpus=[0, 1])


@pytest.fixture
def executor(machine):
    ex = RemoteExecutor(machine)
    ex.container_id = "test_ctr"
    return ex


@pytest.fixture
def sample_config():
    return {"M_LEQ_16": {"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 512,
            "num_stages": 2, "num_warps": 4, "matrix_instr_nonkdim": 16},
        "any": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256,
            "num_stages": 2, "num_warps": 4, "matrix_instr_nonkdim": 16}}

@pytest.fixture
def old_config():
    return {"M_LEQ_16": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 256,
            "num_stages": 3, "num_warps": 4, "matrix_instr_nonkdim": 16},
        "any": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128,
            "num_stages": 2, "num_warps": 8, "matrix_instr_nonkdim": 16}}


class TestFixStrategy:
    def test_suffixed_config_restore_bucket(self):
        """Suffixed config regression: restore old bucket in the suffixed config."""
        reg = RegressionInfo(
            m=8, n=8192, k=2048,
            config_file="gfx950-GEMM-A8W8-N=8192-K=2048.json",
            is_fallback=False,
            regressed_bucket="M_LEQ_16",
            tuned_ns=6000, untuned_ns=5000, baseline_ns=4500,
        )
        strategy = fix_strategy(reg)
        assert strategy == FixStrategy.RESTORE_BUCKET

    def test_fallback_config_promote_to_suffixed(self):
        """Fallback config regression: NEVER modify fallback, promote to suffixed."""
        reg = RegressionInfo(
            m=8, n=8192, k=8192,
            config_file="gfx950-GEMM-A8W8.json",
            is_fallback=True,
            regressed_bucket="M_LEQ_16",
            tuned_ns=6000, untuned_ns=5000, baseline_ns=4500,
        )
        strategy = fix_strategy(reg)
        assert strategy == FixStrategy.PROMOTE_TO_SUFFIXED

    def test_no_config_change_is_noise(self):
        """If old config == new config, it's measurement noise, not a real regression."""
        reg = RegressionInfo(
            m=8, n=8192, k=2048,
            config_file="gfx950-GEMM-A8W8-N=8192-K=2048.json",
            is_fallback=False,
            regressed_bucket="M_LEQ_16",
            tuned_ns=5200, untuned_ns=5000, baseline_ns=4800,
            config_actually_changed=False,
        )
        strategy = fix_strategy(reg)
        assert strategy == FixStrategy.NOISE_SKIP


class TestRegressionFixerAgent:
    @patch("subprocess.run")
    def test_restore_bucket_in_suffixed_config(self, mock_run, executor, sample_config, old_config):
        """For suffixed config: restore old bucket, leave others unchanged."""
        regressions = [
            RegressionInfo(
                m=8, n=8192, k=2048,
                config_file="gfx950-GEMM-A8W8-N=8192-K=2048.json",
                is_fallback=False,
                regressed_bucket="M_LEQ_16",
                tuned_ns=6000, untuned_ns=5000, baseline_ns=4500,
            ),
        ]

        config_reads = {
            "gfx950-GEMM-A8W8-N=8192-K=2048.json": json.dumps(sample_config),
            "gfx950-GEMM-A8W8-N=8192-K=2048.json.old": json.dumps(old_config),
        }

        def side_effect(*args, **kwargs):
            cmd_str = " ".join(args[0]) if isinstance(args[0], list) else str(args[0])
            for key, content in config_reads.items():
                if "cat" in cmd_str and key in cmd_str:
                    return MagicMock(returncode=0, stdout=content, stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = side_effect

        agent = RegressionFixerAgent(
            executor=executor,
            kernel_name="a8w8",
            artifact_dir="/workspace/tuning_artifacts/a8w8",
            regressions=regressions,
            config_dir="/workspace/aiter/aiter/ops/triton/configs/gemm",
            gfx_arch="gfx950",
            variant_prefix="A8W8",
        )
        result = agent.run()
        assert result.success
        assert len(result.data["fixed"]) == 1
        assert result.data["fixed"][0]["strategy"] == "restore_bucket"

    @patch("subprocess.run")
    def test_never_modifies_default_fallback(self, mock_run, executor, sample_config, old_config):
        """CRITICAL: fallback → promote to suffixed, never modify fallback itself."""
        regressions = [RegressionInfo(m=8, n=4096, k=4096,
            config_file="gfx950-GEMM-A8W8.json", is_fallback=True,
            regressed_bucket="M_LEQ_16", tuned_ns=6000, untuned_ns=5000, baseline_ns=4500)]
        def side_effect(*args, **kwargs):
            cmd_str = " ".join(args[0]) if isinstance(args[0], list) else str(args[0])
            if "cat" in cmd_str and "GEMM-A8W8.json" in cmd_str and ".old" not in cmd_str:
                return MagicMock(returncode=0, stdout=json.dumps(sample_config), stderr="")
            if "cat" in cmd_str and ".old" in cmd_str:
                return MagicMock(returncode=0, stdout=json.dumps(old_config), stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")
        mock_run.side_effect = side_effect
        agent = RegressionFixerAgent(executor=executor, kernel_name="a8w8",
            artifact_dir="/workspace/tuning_artifacts/a8w8", regressions=regressions,
            config_dir="/workspace/aiter/aiter/ops/triton/configs/gemm",
            gfx_arch="gfx950", variant_prefix="A8W8")
        result = agent.run()
        assert result.success
        assert len(result.data["promoted"]) == 1
        assert result.data["promoted"][0]["n"] == 4096

    @patch("subprocess.run")
    def test_max_iterations_then_escalate(self, mock_run, executor):
        regressions = [RegressionInfo(m=8, n=8192, k=2048,
            config_file="gfx950-GEMM-A8W8-N=8192-K=2048.json", is_fallback=False,
            regressed_bucket="M_LEQ_16", tuned_ns=6000, untuned_ns=5000,
            baseline_ns=4500, persistent=True)]
        mock_run.return_value = MagicMock(returncode=0, stdout="{}", stderr="")
        agent = RegressionFixerAgent(executor=executor, kernel_name="a8w8",
            artifact_dir="/workspace/tuning_artifacts/a8w8", regressions=regressions,
            config_dir="/workspace/aiter/aiter/ops/triton/configs/gemm",
            gfx_arch="gfx950", variant_prefix="A8W8", max_iterations=3)
        result = agent.run()
        assert len(result.data.get("escalated", [])) > 0

    @patch("subprocess.run")
    def test_skips_noise_regressions(self, mock_run, executor):
        regressions = [RegressionInfo(m=8, n=8192, k=2048,
            config_file="gfx950-GEMM-A8W8-N=8192-K=2048.json", is_fallback=False,
            regressed_bucket="M_LEQ_16", tuned_ns=5100, untuned_ns=5000,
            baseline_ns=4900, config_actually_changed=False)]
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        agent = RegressionFixerAgent(executor=executor, kernel_name="a8w8",
            artifact_dir="/workspace/tuning_artifacts/a8w8", regressions=regressions,
            config_dir="/workspace/aiter/aiter/ops/triton/configs/gemm",
            gfx_arch="gfx950", variant_prefix="A8W8")
        result = agent.run()
        assert result.success
        assert len(result.data.get("noise_skipped", [])) == 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/test_regression_fixer.py -v`
Expected: FAIL — module not found

- [ ] **Step 4: Commit**

```bash
git add tests/tuning_agent/test_regression_fixer.py tests/tuning_agent/fixtures/sample_config.json
git commit -m "test(tuning-agent): add RegressionFixerAgent tests"
```

---

### Task 14: RegressionFixerAgent Implementation

**Files:**
- Create: `aiter/ops/triton/tuning_agent/subagents/regression_fixer.py`

- [ ] **Step 1: Implement RegressionFixerAgent**

```python
# aiter/ops/triton/tuning_agent/subagents/regression_fixer.py
"""RegressionFixerAgent — restore or promote configs to fix tuning regressions."""
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .base import BaseSubagent, SubagentError
from ..remote import RemoteExecutor, RemoteCommandError

logger = logging.getLogger(__name__)

MAX_FIX_ITERATIONS_DEFAULT = 3


class FixStrategy(Enum):
    RESTORE_BUCKET = "restore_bucket"
    PROMOTE_TO_SUFFIXED = "promote_to_suffixed"
    NOISE_SKIP = "noise_skip"
    ESCALATE = "escalate"


@dataclass
class RegressionInfo:
    """Describes a single tuning regression that needs fixing."""
    m: int
    n: int
    k: int
    config_file: str              # current config file name
    is_fallback: bool             # True if config has no N=,K= suffix
    regressed_bucket: str         # e.g., "M_LEQ_16", "any"
    tuned_ns: int                 # timing with new config
    untuned_ns: int               # timing with old config on new Triton
    baseline_ns: int              # timing on old Triton
    config_actually_changed: bool = True  # False if old config == new config
    persistent: bool = False      # True if restore didn't fix it (for iteration tracking)


def fix_strategy(reg: RegressionInfo) -> FixStrategy:
    """
    Determine the fix strategy for a regression.

    Rules:
    1. If config didn't actually change → noise, skip
    2. If fallback config → NEVER modify, promote to suffixed
    3. If suffixed config → restore old bucket
    4. If persistent after restore → escalate
    """
    if not reg.config_actually_changed:
        return FixStrategy.NOISE_SKIP

    if reg.persistent:
        return FixStrategy.ESCALATE

    if reg.is_fallback:
        return FixStrategy.PROMOTE_TO_SUFFIXED

    return FixStrategy.RESTORE_BUCKET


class RegressionFixerAgent(BaseSubagent):
    """Fixes tuning regressions. CRITICAL: never modifies fallback config; promotes to suffixed instead."""
    name = "regression_fixer"

    def __init__(
        self,
        executor: RemoteExecutor,
        kernel_name: str,
        artifact_dir: str,
        regressions: List[RegressionInfo],
        config_dir: str,
        gfx_arch: str,
        variant_prefix: str,
        max_iterations: int = MAX_FIX_ITERATIONS_DEFAULT,
        **kwargs,
    ):
        super().__init__(executor, kernel_name, artifact_dir, **kwargs)
        self.regressions = regressions
        self.config_dir = config_dir
        self.gfx_arch = gfx_arch
        self.variant_prefix = variant_prefix
        self.max_iterations = max_iterations

    def _read_config(self, config_filename: str) -> dict:
        """Read a JSON config file from config_dir."""
        path = f"{self.config_dir}/{config_filename}"
        result = self.executor.docker_exec(f"cat {path}", check=True, timeout=10)
        return json.loads(result.stdout)

    def _read_old_config(self, config_filename: str) -> dict:
        """Read old config: try .old backup first, then git show HEAD:path."""
        path = f"{self.config_dir}/{config_filename}.old"
        try:
            result = self.executor.docker_exec(f"cat {path}", check=True, timeout=10)
            return json.loads(result.stdout)
        except (RemoteCommandError, json.JSONDecodeError):
            try:
                rel = f"aiter/ops/triton/configs/gemm/{config_filename}"
                result = self.executor.docker_exec(
                    f"cd /workspace/aiter && git show HEAD:{rel}", check=True, timeout=10)
                return json.loads(result.stdout)
            except Exception:
                raise SubagentError(f"Cannot find old config for {config_filename}")

    def _write_config(self, config_filename: str, data: dict) -> str:
        path = f"{self.config_dir}/{config_filename}"
        json_str = json.dumps(data, indent=4)
        escaped = json_str.replace("'", "'\\''")
        self.executor.docker_exec(f"printf '%s' '{escaped}' > {path}", check=True, timeout=15)
        return path

    def _suffixed_config_name(self, n: int, k: int) -> str:
        """Generate the suffixed config filename for an N,K pair."""
        return f"{self.gfx_arch}-GEMM-{self.variant_prefix}-N={n}-K={k}.json"

    def _fallback_config_name(self) -> str:
        """The default fallback config filename (no N,K suffix)."""
        return f"{self.gfx_arch}-GEMM-{self.variant_prefix}.json"

    def _restore_bucket(self, reg: RegressionInfo) -> dict:
        """Restore old bucket in a suffixed config. Returns fix info."""
        current = self._read_config(reg.config_file)
        old = self._read_old_config(reg.config_file)

        if reg.regressed_bucket not in old:
            raise SubagentError(
                f"Bucket '{reg.regressed_bucket}' not found in old config {reg.config_file}"
            )

        # Replace only the regressed bucket
        current[reg.regressed_bucket] = old[reg.regressed_bucket]
        self._write_config(reg.config_file, current)

        return {
            "shape": (reg.m, reg.n, reg.k),
            "strategy": FixStrategy.RESTORE_BUCKET.value,
            "config_file": reg.config_file,
            "bucket": reg.regressed_bucket,
        }

    def _promote_to_suffixed(self, reg: RegressionInfo) -> dict:
        """Create a new suffixed config from fallback, with old bucket restored."""
        # Read current fallback (the one we must NOT modify)
        fallback = self._read_config(self._fallback_config_name())

        # Read old fallback to get the old bucket config
        old_fallback = self._read_old_config(self._fallback_config_name())

        # Create new suffixed config: start with current fallback, restore regressed bucket
        new_config = dict(fallback)
        if reg.regressed_bucket in old_fallback:
            new_config[reg.regressed_bucket] = old_fallback[reg.regressed_bucket]

        # Write as suffixed config
        new_filename = self._suffixed_config_name(reg.n, reg.k)
        self._write_config(new_filename, new_config)

        return {
            "n": reg.n,
            "k": reg.k,
            "new_config_file": new_filename,
            "shape": (reg.m, reg.n, reg.k),
            "strategy": FixStrategy.PROMOTE_TO_SUFFIXED.value,
            "bucket": reg.regressed_bucket,
        }

    def _execute(self) -> dict:
        """Fix all regressions according to their strategies."""
        fixed = []
        promoted = []
        noise_skipped = []
        escalated = []
        modified_nk_pairs = set()

        for reg in self.regressions:
            strategy = fix_strategy(reg)

            if strategy == FixStrategy.NOISE_SKIP:
                noise_skipped.append({
                    "shape": (reg.m, reg.n, reg.k),
                    "config_file": reg.config_file,
                })
                logger.info(f"[{self.name}] Skipping noise regression: ({reg.m},{reg.n},{reg.k})")
                continue

            if strategy == FixStrategy.ESCALATE:
                escalated.append({
                    "shape": (reg.m, reg.n, reg.k),
                    "config_file": reg.config_file,
                    "tuned_ns": reg.tuned_ns,
                    "baseline_ns": reg.baseline_ns,
                    "reason": "persistent regression after max iterations",
                })
                logger.warning(f"[{self.name}] Escalating: ({reg.m},{reg.n},{reg.k})")
                continue

            try:
                if strategy == FixStrategy.RESTORE_BUCKET:
                    info = self._restore_bucket(reg)
                    fixed.append(info)
                    modified_nk_pairs.add((reg.n, reg.k))
                    logger.info(f"[{self.name}] Restored bucket '{reg.regressed_bucket}' in {reg.config_file}")

                elif strategy == FixStrategy.PROMOTE_TO_SUFFIXED:
                    info = self._promote_to_suffixed(reg)
                    promoted.append(info)
                    modified_nk_pairs.add((reg.n, reg.k))
                    logger.info(f"[{self.name}] Promoted N={reg.n},K={reg.k} to suffixed config")

            except (SubagentError, RemoteCommandError) as e:
                logger.error(f"[{self.name}] Failed to fix ({reg.m},{reg.n},{reg.k}): {e}")
                escalated.append({
                    "shape": (reg.m, reg.n, reg.k),
                    "config_file": reg.config_file,
                    "reason": str(e),
                })

        # Write fix report
        report = {
            "fixed": fixed,
            "promoted": promoted,
            "noise_skipped": noise_skipped,
            "escalated": escalated,
            "modified_nk_pairs": [list(nk) for nk in modified_nk_pairs],
        }
        self._write_json_artifact("regression_fix_report.json", report)

        return report
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/test_regression_fixer.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add aiter/ops/triton/tuning_agent/subagents/regression_fixer.py
git commit -m "feat(tuning-agent): implement RegressionFixerAgent with promote-to-suffixed strategy"
```

---

## Chunk 6: Package Integration

### Task 15: Subagent Package Exports

**Files:**
- Update: `aiter/ops/triton/tuning_agent/subagents/__init__.py`

- [ ] **Step 1: Update package exports**

```python
# aiter/ops/triton/tuning_agent/subagents/__init__.py
"""Subagent library for the agentic Triton kernel tuning pipeline."""
from .base import BaseSubagent, SubagentResult, SubagentError
from .setup import SetupAgent
from .discovery import DiscoveryAgent
from .script_creator import ScriptCreatorAgent
from .baseline import BaselineAgent
from .tuning import TuningAgent
from .pattern_analyzer import PatternAnalyzerAgent
from .config_generator import ConfigGeneratorAgent
from .validation import ValidationAgent
from .regression_fixer import RegressionFixerAgent, RegressionInfo, FixStrategy

__all__ = [
    "BaseSubagent", "SubagentResult", "SubagentError",
    "SetupAgent",
    "DiscoveryAgent",
    "ScriptCreatorAgent",
    "BaselineAgent",
    "TuningAgent",
    "PatternAnalyzerAgent",
    "ConfigGeneratorAgent",
    "ValidationAgent",
    "RegressionFixerAgent", "RegressionInfo", "FixStrategy",
]
```

- [ ] **Step 2: Run all tests**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/ -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add aiter/ops/triton/tuning_agent/subagents/__init__.py
git commit -m "feat(tuning-agent): export all subagents from package init"
```

---

## Summary

| Subagent | Implementation | Tests | Status |
|----------|---------------|-------|--------|
| BaseSubagent | Full | Full | Chunk 1 |
| SetupAgent | Skeleton + TODO | — | Chunk 2 |
| DiscoveryAgent | Skeleton + TODO | — | Chunk 2 |
| ScriptCreatorAgent | Skeleton + TODO | — | Chunk 2 |
| BaselineAgent | **Full** | **Full** | Chunk 3 |
| TuningAgent | **Full** | **Full** | Chunk 4 |
| PatternAnalyzerAgent | Skeleton + TODO | — | Chunk 2 |
| ConfigGeneratorAgent | Skeleton + TODO | — | Chunk 2 |
| ValidationAgent | Skeleton + TODO | — | Chunk 2 |
| RegressionFixerAgent | **Full** | **Full** | Chunk 5 |

**Fully implemented (3):** BaselineAgent, TuningAgent, RegressionFixerAgent — these are the most complex agents with the most critical correctness requirements (rocprof parsing, screen.py orchestration, never-modify-fallback rule).

**Skeleton with TODO (6):** SetupAgent, DiscoveryAgent, ScriptCreatorAgent, PatternAnalyzerAgent, ConfigGeneratorAgent, ValidationAgent — interfaces and contracts are defined; implementation bodies are marked with TODO and step-by-step comments explaining what needs to be built.
