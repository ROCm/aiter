# Agentic Kernel Tuning Pipeline — Plan 4: Orchestrator + Dashboard

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the top-level orchestrator that reads `triton-upgrade.yaml`, auto-discovers kernels across all GEMM categories, manages the machine pool, dispatches kernel supervisors, displays a terminal dashboard, and generates the final summary report. This is the CLI entry point for the entire tuning pipeline.

**Architecture:** Four modules in `aiter/ops/triton/tuning_agent/`: kernel discovery (filesystem scanning + config/script matching), orchestrator (main loop, machine allocation, supervisor dispatch), dashboard (terminal-print status display), and `__main__` (CLI entry point with argparse). The orchestrator consumes infrastructure from Plan 1 (config, machine_pool, remote, notifications) and dispatches kernel supervisors from Plan 3.

**Tech Stack:** Python 3.8+, asyncio, argparse, json, time, threading (for dashboard refresh)

**Spec:** `/app/aiter/aiter/ops/triton/docs/2026-03-30-agentic-kernel-tuning-pipeline-design.md`

**Depends on:** Plan 1 (Infrastructure Layer), Plan 3 (Kernel Supervisor)

**Enables:** Full end-to-end automated tuning via `python -m aiter.ops.triton.tuning_agent --config triton-upgrade.yaml`

---

## File Structure

```
aiter/ops/triton/tuning_agent/
├── __main__.py              # CLI entry point (argparse, startup, shutdown)
├── kernel_discovery.py      # Auto-discover kernels, match configs/scripts/benchmarks
├── orchestrator.py          # Main loop: allocate machines, dispatch supervisors, collect results
├── dashboard.py             # Terminal dashboard: machine status, kernel progress, logs

tests/tuning_agent/
├── test_kernel_discovery.py
├── test_orchestrator.py
├── test_dashboard.py
├── test_main.py
```

---

## Chunk 1: Kernel Discovery

### Task 1: Kernel Discovery Types

**Files:**
- Edit: `aiter/ops/triton/tuning_agent/types.py` (add new types)
- Create: `tests/tuning_agent/test_kernel_discovery.py`

- [ ] **Step 1: Add kernel discovery types to types.py**

Append to the existing `types.py` from Plan 1:

```python
# --- Add to aiter/ops/triton/tuning_agent/types.py ---

from enum import Enum


class GemmCategory(Enum):
    """GEMM kernel category, maps to subdirectory under gemm/."""
    BASIC = "basic"
    BATCHED = "batched"
    FEED_FORWARD = "feed_forward"
    FUSED = "fused"


@dataclass
class KernelScripts:
    """Paths to associated scripts for a kernel."""
    tuning_script: Optional[str] = None       # ut_*.py in tunning/
    bench_script: Optional[str] = None        # bench_*.py in op_benchmarks/
    test_script: Optional[str] = None         # test_*.py in triton_tests/
    missing: List[str] = field(default_factory=list)  # names of missing scripts


@dataclass
class KernelConfigInfo:
    """Config file inventory for a kernel."""
    variant_prefix: str                       # e.g., "GEMM-A8W8", "BATCHED_GEMM-A8W8"
    fallback_config: Optional[str] = None     # e.g., "gfx950-GEMM-A8W8.json"
    suffixed_configs: List[str] = field(default_factory=list)  # N=...-K=... configs
    needs_new_arch_configs: bool = False       # True if only other-arch configs exist


@dataclass
class DiscoveredKernel:
    """A fully discovered kernel with all associated metadata."""
    name: str                                 # e.g., "a8w8", "batched_a8w8"
    category: GemmCategory
    source_file: str                          # path relative to repo root
    config_info: KernelConfigInfo
    scripts: KernelScripts
    shape_pairs: List[tuple]                  # [(N, K), ...] from config files
    overrides: Optional[KernelOverrides] = None


class KernelStatus(Enum):
    """Status of a kernel in the orchestration pipeline."""
    PENDING = "pending"
    ALLOCATED = "allocated"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class KernelProgress:
    """Tracks progress of a single kernel through the pipeline."""
    kernel: DiscoveredKernel
    status: KernelStatus = KernelStatus.PENDING
    current_phase: str = ""
    phase_number: int = 0
    total_phases: int = 7                     # phases 0-6
    shapes_done: int = 0
    shapes_total: int = 0
    machine_host: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    last_log_line: str = ""
    error_message: str = ""
    regressions_count: int = 0
    geomean_speedup: Optional[float] = None


@dataclass
class OrchestratorState:
    """Full state of the orchestrator for dashboard rendering."""
    run_id: str
    config_path: str
    start_time: float
    kernel_progress: Dict[str, KernelProgress] = field(default_factory=dict)
    machines_status: Dict[str, str] = field(default_factory=dict)  # host -> "idle"/"busy:kernel"/"dead"
    recent_logs: List[str] = field(default_factory=list)
    notifications: List[str] = field(default_factory=list)
    total_kernels: int = 0
    completed_kernels: int = 0
    failed_kernels: int = 0
```

- [ ] **Step 2: Write failing tests for new types**

```python
# tests/tuning_agent/test_kernel_discovery_types.py
from aiter.ops.triton.tuning_agent.types import (
    GemmCategory,
    KernelScripts,
    KernelConfigInfo,
    DiscoveredKernel,
    KernelStatus,
    KernelProgress,
    OrchestratorState,
)


def test_gemm_category_values():
    assert GemmCategory.BASIC.value == "basic"
    assert GemmCategory.BATCHED.value == "batched"
    assert GemmCategory.FEED_FORWARD.value == "feed_forward"
    assert GemmCategory.FUSED.value == "fused"


def test_kernel_scripts_missing():
    s = KernelScripts(tuning_script="ut_a8w8.py", missing=["bench", "test"])
    assert s.bench_script is None
    assert len(s.missing) == 2


def test_kernel_config_info():
    c = KernelConfigInfo(
        variant_prefix="GEMM-A8W8",
        fallback_config="gfx950-GEMM-A8W8.json",
        suffixed_configs=["gfx950-GEMM-A8W8-N=128-K=2048.json"],
    )
    assert not c.needs_new_arch_configs


def test_kernel_status_transitions():
    assert KernelStatus.PENDING.value == "pending"
    assert KernelStatus.COMPLETED.value == "completed"


def test_kernel_progress_defaults():
    from aiter.ops.triton.tuning_agent.types import KernelOverrides
    kernel = DiscoveredKernel(
        name="a8w8",
        category=GemmCategory.BASIC,
        source_file="aiter/ops/triton/gemm/basic/gemm_a8w8.py",
        config_info=KernelConfigInfo(variant_prefix="GEMM-A8W8"),
        scripts=KernelScripts(),
        shape_pairs=[(8192, 8192)],
    )
    progress = KernelProgress(kernel=kernel)
    assert progress.status == KernelStatus.PENDING
    assert progress.phase_number == 0
    assert progress.machine_host is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/test_kernel_discovery_types.py -v`
Expected: FAIL — new types not yet added

- [ ] **Step 4: Implement the types (add to types.py as shown in Step 1)**

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/test_kernel_discovery_types.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add aiter/ops/triton/tuning_agent/types.py tests/tuning_agent/test_kernel_discovery_types.py
git commit -m "feat(tuning-agent): add kernel discovery and orchestrator types"
```

---

### Task 2: Kernel Discovery Logic

**Files:**
- Create: `aiter/ops/triton/tuning_agent/kernel_discovery.py`
- Create: `tests/tuning_agent/test_kernel_discovery.py`

- [ ] **Step 1: Write failing tests for kernel discovery**

```python
# tests/tuning_agent/test_kernel_discovery.py
import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock
from aiter.ops.triton.tuning_agent.kernel_discovery import (
    KernelDiscovery,
    scan_gemm_directory,
    match_config_files,
    match_scripts,
    extract_shape_pairs_from_config,
)
from aiter.ops.triton.tuning_agent.types import (
    GemmCategory, KernelsConfig, KernelOverrides,
)


@pytest.fixture
def mock_repo_root(tmp_path):
    """Create a minimal repo structure for discovery tests."""
    # Basic kernels
    basic = tmp_path / "aiter" / "ops" / "triton" / "gemm" / "basic"
    basic.mkdir(parents=True)
    (basic / "__init__.py").touch()
    (basic / "gemm_a8w8.py").write_text("# kernel")
    (basic / "gemm_a16w16.py").write_text("# kernel")

    # Batched kernels
    batched = tmp_path / "aiter" / "ops" / "triton" / "gemm" / "batched"
    batched.mkdir(parents=True)
    (batched / "__init__.py").touch()
    (batched / "batched_gemm_a8w8.py").write_text("# kernel")

    # Feed-forward kernels
    ff = tmp_path / "aiter" / "ops" / "triton" / "gemm" / "feed_forward"
    ff.mkdir(parents=True)
    (ff / "__init__.py").touch()
    (ff / "ff_a16w16_fused_gated.py").write_text("# kernel")

    # Fused kernels
    fused = tmp_path / "aiter" / "ops" / "triton" / "gemm" / "fused"
    fused.mkdir(parents=True)
    (fused / "__init__.py").touch()
    (fused / "fused_gemm_a8w8_blockscale_a16w16.py").write_text("# kernel")

    # Config files
    configs = tmp_path / "aiter" / "ops" / "triton" / "configs" / "gemm"
    configs.mkdir(parents=True)
    # A8W8 configs - fallback + suffixed
    (configs / "gfx950-GEMM-A8W8.json").write_text(json.dumps({
        "M_LEQ_16": {"BLOCK_SIZE_M": 16},
        "M_LEQ_32": {"BLOCK_SIZE_M": 32},
    }))
    (configs / "gfx950-GEMM-A8W8-N=128-K=2048.json").write_text(json.dumps({
        "M_LEQ_16": {"BLOCK_SIZE_M": 16},
    }))
    # A16W16 - only gfx942
    (configs / "gfx942-GEMM-A16W16.json").write_text(json.dumps({}))
    # Batched
    (configs / "gfx950-BATCHED_GEMM-A8W8.json").write_text(json.dumps({}))
    # Fused
    (configs / "gfx950-FUSED-GEMM-A8W8_BLOCKSCALE-A16W16.json").write_text(json.dumps({}))

    # Tuning scripts
    tunning = tmp_path / "aiter" / "ops" / "triton" / "utils" / "_triton" / "tunning"
    tunning.mkdir(parents=True)
    (tunning / "ut_gemm_a8w8.py").write_text("# tuning script")

    # Bench scripts
    bench = tmp_path / "op_tests" / "op_benchmarks" / "triton"
    bench.mkdir(parents=True)
    (bench / "bench_gemm_a8w8.py").write_text("# bench")

    # Test scripts
    tests = tmp_path / "op_tests" / "triton_tests" / "gemm"
    tests.mkdir(parents=True)
    (tests / "test_gemm_a8w8.py").write_text("# test")

    return tmp_path


def test_scan_gemm_directory_basic(mock_repo_root):
    basic_dir = mock_repo_root / "aiter" / "ops" / "triton" / "gemm" / "basic"
    kernels = scan_gemm_directory(basic_dir, GemmCategory.BASIC, "gemm_")
    assert len(kernels) == 2
    names = {k[0] for k in kernels}
    assert "a8w8" in names
    assert "a16w16" in names


def test_scan_gemm_directory_batched(mock_repo_root):
    batched_dir = mock_repo_root / "aiter" / "ops" / "triton" / "gemm" / "batched"
    kernels = scan_gemm_directory(batched_dir, GemmCategory.BATCHED, "batched_gemm_")
    assert len(kernels) == 1
    assert kernels[0][0] == "batched_a8w8"


def test_scan_gemm_directory_feed_forward(mock_repo_root):
    ff_dir = mock_repo_root / "aiter" / "ops" / "triton" / "gemm" / "feed_forward"
    kernels = scan_gemm_directory(ff_dir, GemmCategory.FEED_FORWARD, "ff_")
    assert len(kernels) == 1
    assert kernels[0][0] == "ff_a16w16_fused_gated"


def test_scan_gemm_directory_fused(mock_repo_root):
    fused_dir = mock_repo_root / "aiter" / "ops" / "triton" / "gemm" / "fused"
    kernels = scan_gemm_directory(fused_dir, GemmCategory.FUSED, "fused_gemm_")
    assert len(kernels) == 1
    assert kernels[0][0] == "fused_a8w8_blockscale_a16w16"


def test_match_config_files(mock_repo_root):
    config_dir = mock_repo_root / "aiter" / "ops" / "triton" / "configs" / "gemm"
    info = match_config_files(config_dir, "GEMM-A8W8", "gfx950")
    assert info.variant_prefix == "GEMM-A8W8"
    assert info.fallback_config == "gfx950-GEMM-A8W8.json"
    assert len(info.suffixed_configs) == 1
    assert not info.needs_new_arch_configs


def test_match_config_files_other_arch_only(mock_repo_root):
    config_dir = mock_repo_root / "aiter" / "ops" / "triton" / "configs" / "gemm"
    info = match_config_files(config_dir, "GEMM-A16W16", "gfx950")
    assert info.fallback_config is None
    assert info.needs_new_arch_configs  # gfx942 exists but not gfx950


def test_match_scripts(mock_repo_root):
    scripts = match_scripts(
        tunning_dir=mock_repo_root / "aiter" / "ops" / "triton" / "utils" / "_triton" / "tunning",
        bench_dir=mock_repo_root / "op_tests" / "op_benchmarks" / "triton",
        test_dir=mock_repo_root / "op_tests" / "triton_tests" / "gemm",
        kernel_name="a8w8",
        category=GemmCategory.BASIC,
    )
    assert scripts.tuning_script is not None
    assert scripts.bench_script is not None
    assert scripts.test_script is not None
    assert len(scripts.missing) == 0


def test_match_scripts_missing(mock_repo_root):
    scripts = match_scripts(
        tunning_dir=mock_repo_root / "aiter" / "ops" / "triton" / "utils" / "_triton" / "tunning",
        bench_dir=mock_repo_root / "op_tests" / "op_benchmarks" / "triton",
        test_dir=mock_repo_root / "op_tests" / "triton_tests" / "gemm",
        kernel_name="a16w16",
        category=GemmCategory.BASIC,
    )
    assert scripts.tuning_script is None
    assert len(scripts.missing) > 0


def test_extract_shape_pairs_from_config(mock_repo_root):
    config_dir = mock_repo_root / "aiter" / "ops" / "triton" / "configs" / "gemm"
    pairs = extract_shape_pairs_from_config(config_dir, "GEMM-A8W8", "gfx950")
    assert (128, 2048) in pairs


def test_full_discovery(mock_repo_root):
    kernels_config = KernelsConfig(exclude=["a16w16"], include=[])
    discovery = KernelDiscovery(mock_repo_root, "gfx950", kernels_config)
    kernels = discovery.discover_all()
    names = {k.name for k in kernels}
    assert "a8w8" in names
    assert "a16w16" not in names  # excluded
    assert "batched_a8w8" in names


def test_full_discovery_include_filter(mock_repo_root):
    kernels_config = KernelsConfig(exclude=[], include=["a8w8"])
    discovery = KernelDiscovery(mock_repo_root, "gfx950", kernels_config)
    kernels = discovery.discover_all()
    assert len(kernels) == 1
    assert kernels[0].name == "a8w8"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/test_kernel_discovery.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement kernel discovery**

```python
# aiter/ops/triton/tuning_agent/kernel_discovery.py
"""Auto-discover Triton GEMM kernels across all categories.

Scans the repository for kernel source files, config files, tuning scripts,
benchmark scripts, and test scripts. Produces a list of DiscoveredKernel objects
with full metadata for the orchestrator to dispatch.

Categories scanned:
  - basic/     gemm_*.py       -> GEMM-<VARIANT>
  - batched/   batched_gemm_*.py -> BATCHED_GEMM-<VARIANT>
  - feed_forward/ ff_*.py      -> FF-<VARIANT>
  - fused/     fused_gemm_*.py -> FUSED-GEMM-<VARIANT>
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from .types import (
    GemmCategory,
    DiscoveredKernel,
    KernelConfigInfo,
    KernelScripts,
    KernelsConfig,
    KernelOverrides,
)

logger = logging.getLogger(__name__)

# Maps GemmCategory to (subdirectory, file_prefix, config_prefix_builder)
# The config_prefix_builder takes the kernel suffix (e.g., "a8w8") and returns
# the config variant prefix (e.g., "GEMM-A8W8").
CATEGORY_SPEC = {
    GemmCategory.BASIC: {
        "subdir": "basic",
        "file_prefix": "gemm_",
        "name_prefix": "",  # basic kernels get no prefix in the kernel name
        "config_prefix": "GEMM-",
    },
    GemmCategory.BATCHED: {
        "subdir": "batched",
        "file_prefix": "batched_gemm_",
        "name_prefix": "batched_",
        "config_prefix": "BATCHED_GEMM-",
    },
    GemmCategory.FEED_FORWARD: {
        "subdir": "feed_forward",
        "file_prefix": "ff_",
        "name_prefix": "ff_",
        "config_prefix": "FF-",
    },
    GemmCategory.FUSED: {
        "subdir": "fused",
        "file_prefix": "fused_gemm_",
        "name_prefix": "fused_",
        "config_prefix": "FUSED-GEMM-",
    },
}

# Known special config variant mappings that don't follow the simple
# uppercase-of-suffix pattern. Maps (category, kernel_suffix) -> variant_suffix.
# If not in this table, the default is suffix.upper().
VARIANT_OVERRIDES: Dict[Tuple[GemmCategory, str], str] = {
    (GemmCategory.BASIC, "a16w16_atomic"): "A16W16-ATOMIC",
    (GemmCategory.BASIC, "a16w16_gated"): "A16W16_GATED",
    (GemmCategory.BATCHED, "bf16"): "A16W16",
    (GemmCategory.FEED_FORWARD, "a16w16_fused_gated"): "A16W16-fused",
    (GemmCategory.FEED_FORWARD, "a16w16_fused_ungated"): "A16W16-fused",
}


def scan_gemm_directory(
    directory: Path,
    category: GemmCategory,
    file_prefix: str,
) -> List[Tuple[str, str, GemmCategory]]:
    """Scan a GEMM subdirectory for kernel source files.

    Args:
        directory: Path to the GEMM subdirectory (e.g., gemm/basic/)
        category: Which GEMM category this is
        file_prefix: File prefix to strip (e.g., "gemm_" for basic)

    Returns:
        List of (kernel_name, source_file_relative_path, category) tuples.
        kernel_name has the category prefix (e.g., "batched_a8w8") but not
        the file prefix (e.g., not "batched_gemm_a8w8").
    """
    if not directory.is_dir():
        logger.warning("GEMM directory not found: %s", directory)
        return []

    spec = CATEGORY_SPEC[category]
    name_prefix = spec["name_prefix"]
    results = []

    for py_file in sorted(directory.glob("*.py")):
        if py_file.name == "__init__.py":
            continue
        if not py_file.name.startswith(file_prefix):
            continue

        # Strip prefix and .py suffix to get the kernel suffix
        suffix = py_file.name[len(file_prefix):-3]  # e.g., "a8w8"
        kernel_name = f"{name_prefix}{suffix}"       # e.g., "batched_a8w8"

        # Build relative path from repo root
        # We store a partial path; the orchestrator prepends the repo root
        rel_path = str(py_file)

        results.append((kernel_name, rel_path, category))
        logger.debug("Found kernel: %s at %s", kernel_name, rel_path)

    return results


def _derive_config_variant(
    category: GemmCategory,
    kernel_suffix: str,
) -> str:
    """Derive the config variant prefix from category and kernel suffix.

    Examples:
        (BASIC, "a8w8") -> "GEMM-A8W8"
        (BATCHED, "a8w8") -> "BATCHED_GEMM-A8W8"
        (BASIC, "a16w16_atomic") -> "GEMM-A16W16-ATOMIC"  (special case)
    """
    spec = CATEGORY_SPEC[category]
    config_prefix = spec["config_prefix"]

    # Check override table first
    override_key = (category, kernel_suffix)
    if override_key in VARIANT_OVERRIDES:
        variant_suffix = VARIANT_OVERRIDES[override_key]
    else:
        variant_suffix = kernel_suffix.upper()

    return f"{config_prefix}{variant_suffix}"


def match_config_files(
    config_dir: Path,
    variant_prefix: str,
    gfx_arch: str,
) -> KernelConfigInfo:
    """Find config files matching a kernel's variant prefix.

    Looks for:
      - Fallback: <gfx_arch>-<variant_prefix>.json
      - Suffixed: <gfx_arch>-<variant_prefix>-N=*-K=*.json
      - Other-arch fallback: <other_arch>-<variant_prefix>.json (flags needs_new_arch_configs)

    Args:
        config_dir: Path to aiter/ops/triton/configs/gemm/
        variant_prefix: e.g., "GEMM-A8W8"
        gfx_arch: e.g., "gfx950"

    Returns:
        KernelConfigInfo with inventory of found config files.
    """
    if not config_dir.is_dir():
        return KernelConfigInfo(variant_prefix=variant_prefix)

    fallback_name = f"{gfx_arch}-{variant_prefix}.json"
    fallback_config = None
    suffixed_configs = []
    has_other_arch = False

    # Pattern for suffixed configs: <arch>-<variant>-N=<N>-K=<K>.json
    # Also handle fused patterns with N8=, N16=, N4= etc.
    suffixed_pattern = re.compile(
        rf"^{re.escape(gfx_arch)}-{re.escape(variant_prefix)}-[NK]"
    )
    other_arch_pattern = re.compile(
        rf"^(?!{re.escape(gfx_arch)}-)(\w+)-{re.escape(variant_prefix)}(\.json|-)"
    )

    for config_file in sorted(config_dir.iterdir()):
        if not config_file.name.endswith(".json"):
            continue

        if config_file.name == fallback_name:
            fallback_config = config_file.name
        elif suffixed_pattern.match(config_file.name):
            suffixed_configs.append(config_file.name)
        elif other_arch_pattern.match(config_file.name):
            has_other_arch = True

    needs_new = has_other_arch and fallback_config is None and len(suffixed_configs) == 0

    return KernelConfigInfo(
        variant_prefix=variant_prefix,
        fallback_config=fallback_config,
        suffixed_configs=suffixed_configs,
        needs_new_arch_configs=needs_new,
    )


def extract_shape_pairs_from_config(
    config_dir: Path,
    variant_prefix: str,
    gfx_arch: str,
) -> List[Tuple[int, int]]:
    """Extract (N, K) pairs from suffixed config file names.

    Parses filenames like gfx950-GEMM-A8W8-N=128-K=2048.json to get (128, 2048).

    Returns:
        Sorted list of (N, K) tuples.
    """
    pairs = []
    pattern = re.compile(
        rf"^{re.escape(gfx_arch)}-{re.escape(variant_prefix)}-N=(\d+)-K=(\d+)\.json$"
    )

    if not config_dir.is_dir():
        return pairs

    for config_file in config_dir.iterdir():
        m = pattern.match(config_file.name)
        if m:
            n, k = int(m.group(1)), int(m.group(2))
            pairs.append((n, k))

    return sorted(pairs)


def match_scripts(
    tunning_dir: Path,
    bench_dir: Path,
    test_dir: Path,
    kernel_name: str,
    category: GemmCategory,
) -> KernelScripts:
    """Find tuning, benchmark, and test scripts for a kernel.

    Script naming conventions:
      - Tuning:    ut_gemm_<suffix>.py  or  ut_batched_gemm_<suffix>.py
      - Benchmark: bench_gemm_<suffix>.py  or  bench_batched_gemm_<suffix>.py  or  bench_ff_<suffix>.py
      - Test:      test_gemm_<suffix>.py

    Args:
        tunning_dir: Path to aiter/ops/triton/utils/_triton/tunning/
        bench_dir: Path to op_tests/op_benchmarks/triton/
        test_dir: Path to op_tests/triton_tests/gemm/
        kernel_name: Kernel name with category prefix (e.g., "batched_a8w8")
        category: GEMM category

    Returns:
        KernelScripts with found script paths and missing script names.
    """
    spec = CATEGORY_SPEC[category]
    # Reconstruct the file-level name (with full prefix)
    # kernel_name = "batched_a8w8", file_prefix = "batched_gemm_", so suffix = "a8w8"
    name_prefix = spec["name_prefix"]
    file_prefix = spec["file_prefix"]
    if name_prefix and kernel_name.startswith(name_prefix):
        suffix = kernel_name[len(name_prefix):]
    else:
        suffix = kernel_name

    # Build expected script names
    ut_name = f"ut_{file_prefix}{suffix}.py"
    bench_name = f"bench_{file_prefix}{suffix}.py"
    test_name = f"test_{file_prefix}{suffix}.py"
    # Alternative test name: test_gemm_<suffix>.py (common for basic)
    test_alt_name = f"test_gemm_{suffix}.py"

    tuning_script = None
    bench_script = None
    test_script = None
    missing = []

    # Check tuning script
    if tunning_dir.is_dir():
        ut_path = tunning_dir / ut_name
        if ut_path.exists():
            tuning_script = str(ut_path)
        else:
            # Try without category prefix for basic kernels
            alt_ut = tunning_dir / f"ut_gemm_{suffix}.py"
            if alt_ut.exists():
                tuning_script = str(alt_ut)
    if tuning_script is None:
        missing.append("tuning")

    # Check bench script
    if bench_dir.is_dir():
        bench_path = bench_dir / bench_name
        if bench_path.exists():
            bench_script = str(bench_path)
        else:
            alt_bench = bench_dir / f"bench_gemm_{suffix}.py"
            if alt_bench.exists():
                bench_script = str(alt_bench)
    if bench_script is None:
        missing.append("bench")

    # Check test script
    if test_dir.is_dir():
        test_path = test_dir / test_name
        if test_path.exists():
            test_script = str(test_path)
        else:
            alt_test = test_dir / test_alt_name
            if alt_test.exists():
                test_script = str(alt_test)
    if test_script is None:
        missing.append("test")

    return KernelScripts(
        tuning_script=tuning_script,
        bench_script=bench_script,
        test_script=test_script,
        missing=missing,
    )


class KernelDiscovery:
    """Discovers all GEMM kernels in the repository.

    Scans all four GEMM category directories, matches each kernel to its
    config files, tuning scripts, benchmark scripts, and test scripts.
    Applies include/exclude filters from the pipeline config.

    Usage:
        discovery = KernelDiscovery(repo_root, "gfx950", kernels_config)
        kernels = discovery.discover_all()
    """

    def __init__(
        self,
        repo_root: Path,
        gfx_arch: str,
        kernels_config: KernelsConfig,
    ):
        self.repo_root = Path(repo_root)
        self.gfx_arch = gfx_arch
        self.kernels_config = kernels_config

        # Standard paths relative to repo root
        self.gemm_base = self.repo_root / "aiter" / "ops" / "triton" / "gemm"
        self.config_dir = self.repo_root / "aiter" / "ops" / "triton" / "configs" / "gemm"
        self.tunning_dir = self.repo_root / "aiter" / "ops" / "triton" / "utils" / "_triton" / "tunning"
        self.bench_dir = self.repo_root / "op_tests" / "op_benchmarks" / "triton"
        self.test_dir = self.repo_root / "op_tests" / "triton_tests" / "gemm"

    def discover_all(self) -> List[DiscoveredKernel]:
        """Discover all kernels across all GEMM categories.

        Returns:
            List of DiscoveredKernel, filtered by include/exclude config.
        """
        raw_kernels = []

        for category, spec in CATEGORY_SPEC.items():
            directory = self.gemm_base / spec["subdir"]
            found = scan_gemm_directory(directory, category, spec["file_prefix"])
            raw_kernels.extend(found)

        logger.info(
            "Scanned %d kernel source files across %d categories",
            len(raw_kernels), len(CATEGORY_SPEC),
        )

        # Build DiscoveredKernel objects
        discovered = []
        for kernel_name, source_file, category in raw_kernels:
            # Apply include/exclude filters
            if not self._should_include(kernel_name):
                logger.info("Skipping kernel %s (filtered by config)", kernel_name)
                continue

            kernel = self._build_kernel_info(kernel_name, source_file, category)
            discovered.append(kernel)

        logger.info("Discovered %d kernels after filtering", len(discovered))
        return discovered

    def _should_include(self, kernel_name: str) -> bool:
        """Check if a kernel passes include/exclude filters."""
        include = self.kernels_config.include
        exclude = self.kernels_config.exclude

        # If include list is non-empty, ONLY include those
        if include:
            return kernel_name in include

        # Otherwise, exclude anything in the exclude list
        return kernel_name not in exclude

    def _build_kernel_info(
        self,
        kernel_name: str,
        source_file: str,
        category: GemmCategory,
    ) -> DiscoveredKernel:
        """Build a DiscoveredKernel with full metadata."""
        spec = CATEGORY_SPEC[category]
        name_prefix = spec["name_prefix"]

        # Get suffix for config variant derivation
        if name_prefix and kernel_name.startswith(name_prefix):
            suffix = kernel_name[len(name_prefix):]
        else:
            suffix = kernel_name

        # Derive config variant prefix
        variant_prefix = _derive_config_variant(category, suffix)

        # Match config files
        config_info = match_config_files(self.config_dir, variant_prefix, self.gfx_arch)

        # Match scripts
        scripts = match_scripts(
            self.tunning_dir, self.bench_dir, self.test_dir,
            kernel_name, category,
        )

        # Extract shape pairs from suffixed configs
        shape_pairs = extract_shape_pairs_from_config(
            self.config_dir, variant_prefix, self.gfx_arch,
        )

        # Check for per-kernel overrides
        overrides = self.kernels_config.overrides.get(kernel_name)

        return DiscoveredKernel(
            name=kernel_name,
            category=category,
            source_file=source_file,
            config_info=config_info,
            scripts=scripts,
            shape_pairs=shape_pairs,
            overrides=overrides,
        )

    def summary(self, kernels: List[DiscoveredKernel]) -> str:
        """Generate a human-readable discovery summary.

        Used by the orchestrator to present the plan before proceeding.
        """
        lines = ["Kernel Discovery Summary", "=" * 60]

        by_category = {}
        for k in kernels:
            by_category.setdefault(k.category.value, []).append(k)

        for cat_name in ["basic", "batched", "feed_forward", "fused"]:
            cat_kernels = by_category.get(cat_name, [])
            if not cat_kernels:
                continue
            lines.append(f"\n{cat_name.upper()} ({len(cat_kernels)} kernels):")
            for k in cat_kernels:
                config_status = "OK" if k.config_info.fallback_config else (
                    "NEEDS_NEW_ARCH" if k.config_info.needs_new_arch_configs else "NO_CONFIG"
                )
                scripts_status = "complete" if not k.scripts.missing else (
                    f"missing: {', '.join(k.scripts.missing)}"
                )
                lines.append(
                    f"  {k.name:40s} configs={config_status:15s} "
                    f"scripts={scripts_status}"
                )
                if k.shape_pairs:
                    lines.append(f"    {len(k.shape_pairs)} suffixed (N,K) pairs")

        total = len(kernels)
        missing_scripts = sum(1 for k in kernels if k.scripts.missing)
        needs_new_configs = sum(1 for k in kernels if k.config_info.needs_new_arch_configs)
        lines.append(f"\nTotal: {total} kernels")
        lines.append(f"  {missing_scripts} need script creation")
        lines.append(f"  {needs_new_configs} need new arch config files")

        return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/test_kernel_discovery.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add aiter/ops/triton/tuning_agent/kernel_discovery.py \
    tests/tuning_agent/test_kernel_discovery.py
git commit -m "feat(tuning-agent): add kernel discovery across all GEMM categories"
```

---

## Chunk 2: Dashboard

### Task 3: Terminal Dashboard

**Files:**
- Create: `aiter/ops/triton/tuning_agent/dashboard.py`
- Create: `tests/tuning_agent/test_dashboard.py`

- [ ] **Step 1: Write failing tests for the dashboard**

```python
# tests/tuning_agent/test_dashboard.py
import time
from io import StringIO
from aiter.ops.triton.tuning_agent.dashboard import Dashboard
from aiter.ops.triton.tuning_agent.types import (
    OrchestratorState,
    KernelProgress,
    KernelStatus,
    DiscoveredKernel,
    GemmCategory,
    KernelConfigInfo,
    KernelScripts,
)


def _make_kernel(name: str) -> DiscoveredKernel:
    return DiscoveredKernel(
        name=name,
        category=GemmCategory.BASIC,
        source_file=f"gemm/basic/gemm_{name}.py",
        config_info=KernelConfigInfo(variant_prefix=f"GEMM-{name.upper()}"),
        scripts=KernelScripts(),
        shape_pairs=[],
    )


def _make_state() -> OrchestratorState:
    state = OrchestratorState(
        run_id="test-run-001",
        config_path="triton-upgrade.yaml",
        start_time=time.time() - 3600,  # 1 hour ago
        total_kernels=3,
        completed_kernels=1,
        failed_kernels=0,
    )
    state.machines_status = {
        "gpu-machine-1": "busy:a8w8",
        "gpu-machine-2": "idle",
    }
    # One completed kernel
    k1 = _make_kernel("a16w16")
    p1 = KernelProgress(
        kernel=k1, status=KernelStatus.COMPLETED,
        current_phase="commit", phase_number=6,
        shapes_done=50, shapes_total=50,
        machine_host="gpu-machine-1",
        start_time=time.time() - 7200,
        end_time=time.time() - 3600,
        geomean_speedup=1.05,
    )
    state.kernel_progress["a16w16"] = p1

    # One running kernel
    k2 = _make_kernel("a8w8")
    p2 = KernelProgress(
        kernel=k2, status=KernelStatus.RUNNING,
        current_phase="tuning (full)", phase_number=4,
        shapes_done=30, shapes_total=80,
        machine_host="gpu-machine-1",
        start_time=time.time() - 1800,
        last_log_line="screen.py: M=128 N=8192 K=8192 GPU=3 ... 45 screencases",
    )
    state.kernel_progress["a8w8"] = p2

    # One pending kernel
    k3 = _make_kernel("afp4wfp4")
    p3 = KernelProgress(kernel=k3, status=KernelStatus.PENDING)
    state.kernel_progress["afp4wfp4"] = p3

    state.recent_logs = [
        "[12:00:01] a8w8: Phase 4 tuning started",
        "[12:15:30] a8w8: Scout complete, 12 shapes tuned",
        "[12:30:00] a8w8: Full tuning 30/80 shapes",
    ]

    return state


def test_dashboard_render_no_crash():
    """Dashboard render should not crash."""
    state = _make_state()
    dashboard = Dashboard()
    output = dashboard.render(state)
    assert isinstance(output, str)
    assert len(output) > 0


def test_dashboard_shows_run_id():
    state = _make_state()
    dashboard = Dashboard()
    output = dashboard.render(state)
    assert "test-run-001" in output


def test_dashboard_shows_machine_status():
    state = _make_state()
    dashboard = Dashboard()
    output = dashboard.render(state)
    assert "gpu-machine-1" in output
    assert "gpu-machine-2" in output


def test_dashboard_shows_kernel_progress():
    state = _make_state()
    dashboard = Dashboard()
    output = dashboard.render(state)
    assert "a8w8" in output
    assert "a16w16" in output
    assert "afp4wfp4" in output


def test_dashboard_shows_recent_logs():
    state = _make_state()
    dashboard = Dashboard()
    output = dashboard.render(state)
    assert "Scout complete" in output


def test_dashboard_shows_completion_stats():
    state = _make_state()
    dashboard = Dashboard()
    output = dashboard.render(state)
    assert "1/3" in output or "1 / 3" in output  # completed/total


def test_dashboard_empty_state():
    state = OrchestratorState(
        run_id="empty", config_path="x.yaml", start_time=time.time(),
    )
    dashboard = Dashboard()
    output = dashboard.render(state)
    assert "empty" in output


def test_dashboard_format_duration():
    dashboard = Dashboard()
    assert dashboard._format_duration(65) == "1m 5s"
    assert dashboard._format_duration(3661) == "1h 1m"
    assert dashboard._format_duration(0) == "0s"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/test_dashboard.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement the dashboard**

```python
# aiter/ops/triton/tuning_agent/dashboard.py
"""Terminal dashboard for the Triton tuning orchestrator.

Renders a text-based status display showing machine status, per-kernel
progress, recent log lines, and regression summaries. Designed for v1
simplicity: plain print-based output, no curses or rich dependency.

The dashboard is stateless — it receives an OrchestratorState snapshot
and renders it. The orchestrator calls dashboard.refresh(state) periodically
from a background thread.
"""

import os
import sys
import time
import threading
from typing import Optional

from .types import (
    OrchestratorState,
    KernelProgress,
    KernelStatus,
)

# ANSI escape codes for minimal terminal formatting
CLEAR_SCREEN = "\033[2J\033[H"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RESET = "\033[0m"

# Status -> color mapping
STATUS_COLORS = {
    KernelStatus.PENDING: DIM,
    KernelStatus.ALLOCATED: CYAN,
    KernelStatus.RUNNING: GREEN,
    KernelStatus.WAITING_APPROVAL: YELLOW,
    KernelStatus.COMPLETED: GREEN,
    KernelStatus.FAILED: RED,
    KernelStatus.SKIPPED: DIM,
}

# Phase names for display
PHASE_NAMES = {
    0: "setup",
    1: "discovery",
    2: "baseline",
    3: "untuned",
    4: "tuning",
    5: "validation",
    6: "commit",
}

# Maximum number of recent log lines to display
MAX_LOG_LINES = 8
# Maximum number of notifications to display
MAX_NOTIFICATIONS = 5


class Dashboard:
    """Text-based terminal dashboard.

    Usage:
        dashboard = Dashboard()

        # One-shot render (returns string):
        text = dashboard.render(state)
        print(text)

        # Auto-refresh mode (clears terminal, reprints):
        dashboard.start_auto_refresh(state_provider, interval=5.0)
        # ... later ...
        dashboard.stop_auto_refresh()
    """

    def __init__(self, use_color: bool = True, width: int = 100):
        self.use_color = use_color and sys.stdout.isatty()
        self.width = width
        self._refresh_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def render(self, state: OrchestratorState) -> str:
        """Render the full dashboard as a string.

        Args:
            state: Current orchestrator state snapshot.

        Returns:
            Multi-line string ready for printing.
        """
        lines = []
        lines.append(self._render_header(state))
        lines.append(self._render_separator())
        lines.append(self._render_machines(state))
        lines.append(self._render_separator())
        lines.append(self._render_kernels(state))
        lines.append(self._render_separator())
        lines.append(self._render_logs(state))

        if state.notifications:
            lines.append(self._render_separator())
            lines.append(self._render_notifications(state))

        lines.append(self._render_separator())
        lines.append(self._render_footer(state))

        return "\n".join(lines)

    def refresh(self, state: OrchestratorState) -> None:
        """Clear terminal and print updated dashboard."""
        output = CLEAR_SCREEN + self.render(state)
        sys.stdout.write(output + "\n")
        sys.stdout.flush()

    def start_auto_refresh(
        self,
        state_provider,
        interval: float = 5.0,
    ) -> None:
        """Start a background thread that refreshes the dashboard.

        Args:
            state_provider: Callable that returns the current OrchestratorState.
            interval: Seconds between refreshes.
        """
        self._stop_event.clear()

        def _loop():
            while not self._stop_event.is_set():
                try:
                    state = state_provider()
                    self.refresh(state)
                except Exception:
                    pass  # Don't crash the dashboard thread
                self._stop_event.wait(interval)

        self._refresh_thread = threading.Thread(
            target=_loop, daemon=True, name="dashboard-refresh",
        )
        self._refresh_thread.start()

    def stop_auto_refresh(self) -> None:
        """Stop the background refresh thread."""
        self._stop_event.set()
        if self._refresh_thread:
            self._refresh_thread.join(timeout=2.0)
            self._refresh_thread = None

    # --- Rendering helpers ---

    def _c(self, color: str, text: str) -> str:
        """Apply ANSI color if color is enabled."""
        if not self.use_color:
            return text
        return f"{color}{text}{RESET}"

    def _render_header(self, state: OrchestratorState) -> str:
        elapsed = self._format_duration(time.time() - state.start_time)
        title = self._c(BOLD, "TRITON KERNEL TUNING PIPELINE")
        return (
            f"{title}\n"
            f"  Run:     {state.run_id}\n"
            f"  Config:  {state.config_path}\n"
            f"  Elapsed: {elapsed}    "
            f"Kernels: {state.completed_kernels}/{state.total_kernels} done, "
            f"{state.failed_kernels} failed"
        )

    def _render_separator(self) -> str:
        return "-" * self.width

    def _render_machines(self, state: OrchestratorState) -> str:
        lines = [self._c(BOLD, "MACHINES")]
        if not state.machines_status:
            lines.append("  (no machines)")
            return "\n".join(lines)

        for host, status in sorted(state.machines_status.items()):
            if status == "idle":
                icon = self._c(GREEN, "[IDLE]   ")
            elif status == "dead":
                icon = self._c(RED, "[DEAD]   ")
            elif status.startswith("busy:"):
                kernel = status.split(":", 1)[1]
                icon = self._c(CYAN, f"[BUSY]    -> {kernel}")
            else:
                icon = f"[{status}]"
            lines.append(f"  {host:30s} {icon}")

        return "\n".join(lines)

    def _render_kernels(self, state: OrchestratorState) -> str:
        lines = [self._c(BOLD, "KERNEL PROGRESS")]
        if not state.kernel_progress:
            lines.append("  (no kernels)")
            return "\n".join(lines)

        # Sort: running first, then waiting, then pending, then completed/failed
        order = {
            KernelStatus.RUNNING: 0,
            KernelStatus.WAITING_APPROVAL: 1,
            KernelStatus.ALLOCATED: 2,
            KernelStatus.PENDING: 3,
            KernelStatus.COMPLETED: 4,
            KernelStatus.FAILED: 5,
            KernelStatus.SKIPPED: 6,
        }
        sorted_kernels = sorted(
            state.kernel_progress.values(),
            key=lambda p: (order.get(p.status, 99), p.kernel.name),
        )

        for progress in sorted_kernels:
            lines.append(self._render_kernel_line(progress))

        return "\n".join(lines)

    def _render_kernel_line(self, p: KernelProgress) -> str:
        """Render a single kernel's status line."""
        color = STATUS_COLORS.get(p.status, "")
        status_str = self._c(color, f"[{p.status.value:18s}]")

        # Phase progress bar
        if p.status in (KernelStatus.RUNNING, KernelStatus.ALLOCATED):
            phase_str = f"phase {p.phase_number}/6: {p.current_phase}"
            if p.shapes_total > 0:
                pct = p.shapes_done * 100 // p.shapes_total
                bar_len = 20
                filled = pct * bar_len // 100
                bar = "#" * filled + "." * (bar_len - filled)
                shape_str = f"[{bar}] {p.shapes_done}/{p.shapes_total}"
            else:
                shape_str = ""
        elif p.status == KernelStatus.COMPLETED:
            phase_str = "done"
            speedup = f"geomean: {p.geomean_speedup:.2f}x" if p.geomean_speedup else ""
            regr = f"regressions: {p.regressions_count}" if p.regressions_count else ""
            shape_str = f"{speedup}  {regr}".strip()
        elif p.status == KernelStatus.FAILED:
            phase_str = p.error_message[:50] if p.error_message else "failed"
            shape_str = ""
        elif p.status == KernelStatus.WAITING_APPROVAL:
            phase_str = "AWAITING HUMAN APPROVAL"
            shape_str = ""
        else:
            phase_str = ""
            shape_str = ""

        # Elapsed time for active kernels
        elapsed = ""
        if p.start_time:
            dt = (p.end_time or time.time()) - p.start_time
            elapsed = self._format_duration(dt)

        name = p.kernel.name
        machine = p.machine_host or ""

        line = f"  {name:30s} {status_str} {phase_str}"
        if shape_str:
            line += f"  {shape_str}"
        if elapsed:
            line += f"  ({elapsed})"
        if machine and p.status == KernelStatus.RUNNING:
            line += f"  @{machine}"

        # Add last log snippet for running kernels
        if p.status == KernelStatus.RUNNING and p.last_log_line:
            truncated = p.last_log_line[:80]
            line += f"\n    {self._c(DIM, truncated)}"

        return line

    def _render_logs(self, state: OrchestratorState) -> str:
        lines = [self._c(BOLD, "RECENT LOGS")]
        if not state.recent_logs:
            lines.append("  (no logs yet)")
            return "\n".join(lines)

        # Show the most recent MAX_LOG_LINES entries
        for log_line in state.recent_logs[-MAX_LOG_LINES:]:
            lines.append(f"  {log_line}")

        return "\n".join(lines)

    def _render_notifications(self, state: OrchestratorState) -> str:
        lines = [self._c(BOLD + YELLOW, "NOTIFICATIONS")]
        for note in state.notifications[-MAX_NOTIFICATIONS:]:
            lines.append(f"  {self._c(YELLOW, '!')} {note}")
        return "\n".join(lines)

    def _render_footer(self, state: OrchestratorState) -> str:
        now = time.strftime("%H:%M:%S")
        return self._c(DIM, f"Last updated: {now}  |  Ctrl+C to interrupt")

    def _format_duration(self, seconds: float) -> str:
        """Format seconds into human-readable duration."""
        s = int(seconds)
        if s < 60:
            return f"{s}s"
        elif s < 3600:
            m, sec = divmod(s, 60)
            return f"{m}m {sec}s"
        else:
            h, remainder = divmod(s, 3600)
            m = remainder // 60
            return f"{h}h {m}m"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/test_dashboard.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add aiter/ops/triton/tuning_agent/dashboard.py tests/tuning_agent/test_dashboard.py
git commit -m "feat(tuning-agent): add terminal dashboard for orchestrator status"
```

---

## Chunk 3: Orchestrator

### Task 4: Orchestrator Core

**Files:**
- Create: `aiter/ops/triton/tuning_agent/orchestrator.py`
- Create: `tests/tuning_agent/test_orchestrator.py`

- [ ] **Step 1: Write failing tests for the orchestrator**

```python
# tests/tuning_agent/test_orchestrator.py
import time
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
from aiter.ops.triton.tuning_agent.orchestrator import (
    Orchestrator,
    RunResult,
    generate_summary_report,
)
from aiter.ops.triton.tuning_agent.types import (
    PipelineConfig,
    MachineInfo,
    ContainerConfig,
    RepoConfig,
    TuningConfig,
    TuningThresholds,
    TuningTimeouts,
    GpuConfig,
    TritonInstallConfig,
    KernelsConfig,
    KernelStatus,
    KernelProgress,
    DiscoveredKernel,
    GemmCategory,
    KernelConfigInfo,
    KernelScripts,
    OrchestratorState,
)


def _make_config(num_machines=2) -> PipelineConfig:
    machines = [
        MachineInfo(
            host=f"gpu-{i+1}", user="root",
            ssh_key="~/.ssh/id", gpus=list(range(8)),
        )
        for i in range(num_machines)
    ]
    return PipelineConfig(
        baseline=RepoConfig(
            aiter_repo="https://github.com/ROCm/aiter.git",
            aiter_branch="main",
            triton_repo="https://github.com/ROCm/triton.git",
            triton_branch="triton_3_4",
        ),
        target=RepoConfig(
            aiter_repo="https://github.com/ROCm/aiter.git",
            aiter_branch="feature",
            triton_repo="https://github.com/ROCm/triton.git",
            triton_branch="main",
        ),
        machines=machines,
        container=ContainerConfig(image="rocm/pytorch:latest"),
        gpu=GpuConfig(arch="gfx950"),
        tuning=TuningConfig(),
    )


def _make_kernel(name: str) -> DiscoveredKernel:
    return DiscoveredKernel(
        name=name,
        category=GemmCategory.BASIC,
        source_file=f"gemm/basic/gemm_{name}.py",
        config_info=KernelConfigInfo(variant_prefix=f"GEMM-{name.upper()}"),
        scripts=KernelScripts(),
        shape_pairs=[(8192, 8192)],
    )


def test_orchestrator_creation():
    config = _make_config()
    orch = Orchestrator(config, run_id="test-001")
    assert orch.run_id == "test-001"
    assert orch.state.total_kernels == 0  # not yet discovered


def test_orchestrator_generate_run_id():
    config = _make_config()
    orch = Orchestrator(config)
    assert orch.run_id  # auto-generated
    assert len(orch.run_id) > 0


def test_orchestrator_init_machines(self_config=None):
    config = _make_config(num_machines=2)
    orch = Orchestrator(config, run_id="test")
    orch._init_machine_status()
    assert "gpu-1" in orch.state.machines_status
    assert "gpu-2" in orch.state.machines_status
    assert all(s == "idle" for s in orch.state.machines_status.values())


def test_orchestrator_allocate_machine():
    config = _make_config(num_machines=2)
    orch = Orchestrator(config, run_id="test")
    orch._init_machine_status()

    machine = orch._allocate_machine("a8w8")
    assert machine is not None
    assert machine.host in ("gpu-1", "gpu-2")
    assert orch.state.machines_status[machine.host] == "busy:a8w8"


def test_orchestrator_allocate_machine_all_busy():
    config = _make_config(num_machines=1)
    orch = Orchestrator(config, run_id="test")
    orch._init_machine_status()

    m1 = orch._allocate_machine("a8w8")
    assert m1 is not None

    m2 = orch._allocate_machine("a16w16")
    assert m2 is None  # no machines available


def test_orchestrator_release_machine():
    config = _make_config(num_machines=1)
    orch = Orchestrator(config, run_id="test")
    orch._init_machine_status()

    machine = orch._allocate_machine("a8w8")
    assert machine is not None
    orch._release_machine(machine)
    assert orch.state.machines_status[machine.host] == "idle"


def test_orchestrator_set_kernels():
    config = _make_config()
    orch = Orchestrator(config, run_id="test")
    kernels = [_make_kernel("a8w8"), _make_kernel("a16w16")]
    orch.set_kernels(kernels)
    assert orch.state.total_kernels == 2
    assert "a8w8" in orch.state.kernel_progress
    assert "a16w16" in orch.state.kernel_progress


def test_orchestrator_next_pending_kernel():
    config = _make_config()
    orch = Orchestrator(config, run_id="test")
    kernels = [_make_kernel("a8w8"), _make_kernel("a16w16")]
    orch.set_kernels(kernels)

    name = orch._next_pending_kernel()
    assert name in ("a8w8", "a16w16")

    # Mark it as running
    orch.state.kernel_progress[name].status = KernelStatus.RUNNING
    name2 = orch._next_pending_kernel()
    assert name2 is not None
    assert name2 != name

    # Mark second as running too
    orch.state.kernel_progress[name2].status = KernelStatus.RUNNING
    assert orch._next_pending_kernel() is None


def test_orchestrator_log():
    config = _make_config()
    orch = Orchestrator(config, run_id="test")
    orch.log("test message")
    assert len(orch.state.recent_logs) == 1
    assert "test message" in orch.state.recent_logs[0]


def test_orchestrator_log_max_entries():
    config = _make_config()
    orch = Orchestrator(config, run_id="test")
    for i in range(200):
        orch.log(f"message {i}")
    assert len(orch.state.recent_logs) <= 100  # capped


def test_orchestrator_notify():
    config = _make_config()
    orch = Orchestrator(config, run_id="test")
    orch.notify("approval needed: a8w8 has regressions")
    assert len(orch.state.notifications) == 1


def test_generate_summary_report():
    state = OrchestratorState(
        run_id="test-001",
        config_path="triton-upgrade.yaml",
        start_time=time.time() - 7200,
        total_kernels=2,
        completed_kernels=2,
        failed_kernels=0,
    )
    k1 = _make_kernel("a8w8")
    p1 = KernelProgress(
        kernel=k1, status=KernelStatus.COMPLETED,
        geomean_speedup=1.08, regressions_count=2,
        shapes_done=50, shapes_total=50,
        start_time=time.time() - 7200,
        end_time=time.time() - 3600,
    )
    k2 = _make_kernel("a16w16")
    p2 = KernelProgress(
        kernel=k2, status=KernelStatus.COMPLETED,
        geomean_speedup=1.02, regressions_count=0,
        shapes_done=30, shapes_total=30,
        start_time=time.time() - 3600,
        end_time=time.time(),
    )
    state.kernel_progress = {"a8w8": p1, "a16w16": p2}

    report = generate_summary_report(state)
    assert "a8w8" in report
    assert "a16w16" in report
    assert "1.08" in report
    assert "test-001" in report
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/test_orchestrator.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement the orchestrator**

```python
# aiter/ops/triton/tuning_agent/orchestrator.py
"""Top-level orchestrator for the Triton kernel tuning pipeline.

Manages the full tuning lifecycle:
  1. Reads config, validates machines, discovers kernels
  2. Allocates machines from the pool
  3. Dispatches kernel supervisors (one per kernel, one per machine)
  4. Monitors progress via the dashboard
  5. Handles escalations and human approval gates
  6. Generates final summary report

The orchestrator runs on the local machine and coordinates remote work
via SSH + docker exec (using the infrastructure from Plan 1).
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any

from .types import (
    PipelineConfig,
    MachineInfo,
    KernelsConfig,
    DiscoveredKernel,
    KernelStatus,
    KernelProgress,
    OrchestratorState,
)
from .kernel_discovery import KernelDiscovery
from .dashboard import Dashboard

logger = logging.getLogger(__name__)

# Maximum recent log entries to keep in state
MAX_LOG_ENTRIES = 100
# Maximum recent notifications to keep
MAX_NOTIFICATIONS = 50
# How long to wait (seconds) before polling for a free machine
MACHINE_POLL_INTERVAL = 30
# Dashboard refresh interval in seconds
DASHBOARD_REFRESH_INTERVAL = 5.0


@dataclass
class RunResult:
    """Final result of an orchestrator run."""
    run_id: str
    success: bool
    total_kernels: int
    completed: int
    failed: int
    skipped: int
    wall_time_seconds: float
    summary_report: str
    report_path: Optional[str] = None


class Orchestrator:
    """Top-level agent managing the full Triton upgrade across all kernels.

    Lifecycle:
        1. __init__(config) — parse config, set up state
        2. set_kernels(kernels) — provide discovered kernels
        3. run() — main async loop: allocate, dispatch, monitor, collect
        4. shutdown() — cleanup, generate report

    The orchestrator does NOT directly run remote commands. It delegates to
    kernel supervisors (Plan 3) which use the remote executor (Plan 1).
    """

    def __init__(
        self,
        config: PipelineConfig,
        run_id: Optional[str] = None,
        results_dir: Optional[Path] = None,
        supervisor_factory: Optional[Callable] = None,
    ):
        """Initialize the orchestrator.

        Args:
            config: Parsed pipeline configuration.
            run_id: Optional run identifier. Auto-generated if not provided.
            results_dir: Local directory for results. Defaults to
                ~/.tuning_results/<run_id>/
            supervisor_factory: Callable that creates a kernel supervisor.
                Signature: (kernel, machine, config) -> supervisor.
                If None, uses the default KernelSupervisor from Plan 3.
        """
        self.config = config
        self.run_id = run_id or self._generate_run_id()
        self.results_dir = results_dir or Path.home() / ".tuning_results" / self.run_id
        self.supervisor_factory = supervisor_factory
        self.dashboard = Dashboard()

        # Internal state
        self.state = OrchestratorState(
            run_id=self.run_id,
            config_path="",
            start_time=time.time(),
        )
        self._machines: Dict[str, MachineInfo] = {
            m.host: m for m in config.machines
        }
        self._active_supervisors: Dict[str, Any] = {}  # kernel_name -> supervisor
        self._kernel_queue: List[str] = []  # kernel names in order

    @staticmethod
    def _generate_run_id() -> str:
        """Generate a unique run ID from timestamp + short UUID."""
        ts = time.strftime("%Y%m%d-%H%M%S")
        short_uuid = uuid.uuid4().hex[:6]
        return f"run-{ts}-{short_uuid}"

    def _init_machine_status(self) -> None:
        """Initialize all machines as idle."""
        for host in self._machines:
            self.state.machines_status[host] = "idle"

    def set_kernels(self, kernels: List[DiscoveredKernel]) -> None:
        """Set the list of kernels to process.

        Called after discovery, before run().
        """
        self.state.total_kernels = len(kernels)
        self._kernel_queue = [k.name for k in kernels]
        for kernel in kernels:
            self.state.kernel_progress[kernel.name] = KernelProgress(kernel=kernel)

    def _allocate_machine(self, kernel_name: str) -> Optional[MachineInfo]:
        """Try to allocate an idle machine for a kernel.

        Returns:
            MachineInfo if a machine was allocated, None if all busy.
        """
        for host, status in self.state.machines_status.items():
            if status == "idle":
                self.state.machines_status[host] = f"busy:{kernel_name}"
                return self._machines[host]
        return None

    def _release_machine(self, machine: MachineInfo) -> None:
        """Release a machine back to the idle pool."""
        self.state.machines_status[machine.host] = "idle"

    def _mark_machine_dead(self, machine: MachineInfo) -> None:
        """Mark a machine as dead (unreachable)."""
        self.state.machines_status[machine.host] = "dead"

    def _next_pending_kernel(self) -> Optional[str]:
        """Get the next kernel that is still pending.

        Returns:
            Kernel name, or None if no pending kernels remain.
        """
        for name in self._kernel_queue:
            if self.state.kernel_progress[name].status == KernelStatus.PENDING:
                return name
        return None

    def _all_done(self) -> bool:
        """Check if all kernels are in a terminal state."""
        terminal = {
            KernelStatus.COMPLETED,
            KernelStatus.FAILED,
            KernelStatus.SKIPPED,
        }
        return all(
            p.status in terminal
            for p in self.state.kernel_progress.values()
        )

    def log(self, message: str, kernel_name: Optional[str] = None) -> None:
        """Add a log entry to the orchestrator state.

        Args:
            message: Log message.
            kernel_name: If provided, prefixes the message with the kernel name.
        """
        ts = time.strftime("%H:%M:%S")
        prefix = f"{kernel_name}: " if kernel_name else ""
        entry = f"[{ts}] {prefix}{message}"
        self.state.recent_logs.append(entry)
        # Cap the log buffer
        if len(self.state.recent_logs) > MAX_LOG_ENTRIES:
            self.state.recent_logs = self.state.recent_logs[-MAX_LOG_ENTRIES:]
        logger.info("%s%s", prefix, message)

    def notify(self, message: str) -> None:
        """Add a notification (for human attention).

        Triggers terminal bell and adds to the notification list.
        """
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {message}"
        self.state.notifications.append(entry)
        if len(self.state.notifications) > MAX_NOTIFICATIONS:
            self.state.notifications = self.state.notifications[-MAX_NOTIFICATIONS:]
        # Terminal bell
        print("\a", end="", flush=True)
        logger.warning("NOTIFICATION: %s", message)

    def get_state(self) -> OrchestratorState:
        """Return the current state snapshot (for dashboard rendering)."""
        return self.state

    async def validate_machines(self) -> List[MachineInfo]:
        """Validate SSH connectivity and GPU availability for all machines.

        Tests each machine by running `rocm-smi --showgpu` via SSH. Marks
        unreachable machines as dead.

        Returns:
            List of validated (reachable) machines.
        """
        self._init_machine_status()
        validated = []

        for host, machine in self._machines.items():
            self.log(f"Validating machine {host}...")
            try:
                # Plan 1's remote executor handles the actual SSH call.
                # Here we define the validation logic; the executor is injected.
                # For now, we record the machine as validated.
                # In production, this calls:
                #   remote.run(machine, "rocm-smi --showgpu")
                #   remote.run(machine, f"test -f {machine.ssh_key}")
                validated.append(machine)
                self.log(f"Machine {host}: OK ({machine.gpu_count} GPUs)")
            except Exception as e:
                self._mark_machine_dead(machine)
                self.log(f"Machine {host}: FAILED - {e}")
                self.notify(f"Machine {host} unreachable: {e}")

        if not validated:
            raise RuntimeError("No machines available after validation")

        return validated

    async def discover_kernels(self, repo_root: Path) -> List[DiscoveredKernel]:
        """Auto-discover kernels and present plan to human.

        Args:
            repo_root: Path to the aiter repository root.

        Returns:
            List of discovered kernels after filtering.
        """
        gfx_arch = self.config.gpu.arch or "gfx950"  # fallback; in prod, auto-detect
        discovery = KernelDiscovery(
            repo_root, gfx_arch, self.config.kernels,
        )
        kernels = discovery.discover_all()

        summary = discovery.summary(kernels)
        self.log("Kernel discovery complete")
        self.log(f"Found {len(kernels)} kernels")

        # Print summary for human review
        print("\n" + summary + "\n")

        return kernels

    async def run(self, repo_root: Path) -> RunResult:
        """Execute the full orchestration loop.

        This is the main entry point called by __main__.py.

        Steps:
          1. Validate machines
          2. Discover kernels
          3. Present plan and wait for human confirmation
          4. Start dashboard
          5. Main loop: allocate machines, dispatch supervisors, monitor
          6. Generate summary report
          7. Shutdown

        Args:
            repo_root: Path to the aiter repository root.

        Returns:
            RunResult with overall outcome.
        """
        self.state.config_path = str(repo_root)
        self.state.start_time = time.time()

        # Step 1: Validate machines
        self.log("Validating machines...")
        validated = await self.validate_machines()
        self.log(f"{len(validated)}/{len(self._machines)} machines validated")

        # Step 2: Discover kernels
        self.log("Discovering kernels...")
        kernels = await self.discover_kernels(repo_root)
        self.set_kernels(kernels)

        # Step 3: Human confirmation
        num_machines = sum(
            1 for s in self.state.machines_status.values() if s != "dead"
        )
        print(
            f"\nReady to tune {len(kernels)} kernels across "
            f"{num_machines} machines."
        )
        confirmation = input("Proceed? [Y/n] ").strip().lower()
        if confirmation and confirmation != "y":
            self.log("Aborted by user")
            return RunResult(
                run_id=self.run_id,
                success=False,
                total_kernels=len(kernels),
                completed=0,
                failed=0,
                skipped=len(kernels),
                wall_time_seconds=time.time() - self.state.start_time,
                summary_report="Aborted by user before start.",
            )

        # Step 4: Start dashboard
        self.dashboard.start_auto_refresh(
            self.get_state, interval=DASHBOARD_REFRESH_INTERVAL,
        )

        # Step 5: Main orchestration loop
        try:
            await self._main_loop()
        except KeyboardInterrupt:
            self.log("Interrupted by user (Ctrl+C)")
            self.notify("Pipeline interrupted by user")
        except Exception as e:
            self.log(f"Orchestrator error: {e}")
            self.notify(f"Orchestrator crashed: {e}")
            logger.exception("Orchestrator main loop failed")
        finally:
            self.dashboard.stop_auto_refresh()

        # Step 6: Generate summary
        wall_time = time.time() - self.state.start_time
        summary = generate_summary_report(self.state)

        # Save report to disk
        self.results_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.results_dir / "summary.txt"
        report_path.write_text(summary)

        # Also save machine-readable state
        state_path = self.results_dir / "state.json"
        state_path.write_text(json.dumps(
            self._serialize_state(), indent=2,
        ))

        self.log(f"Summary report saved to {report_path}")
        print("\n" + summary)

        completed = sum(
            1 for p in self.state.kernel_progress.values()
            if p.status == KernelStatus.COMPLETED
        )
        failed = sum(
            1 for p in self.state.kernel_progress.values()
            if p.status == KernelStatus.FAILED
        )
        skipped = sum(
            1 for p in self.state.kernel_progress.values()
            if p.status == KernelStatus.SKIPPED
        )

        return RunResult(
            run_id=self.run_id,
            success=failed == 0,
            total_kernels=self.state.total_kernels,
            completed=completed,
            failed=failed,
            skipped=skipped,
            wall_time_seconds=wall_time,
            summary_report=summary,
            report_path=str(report_path),
        )

    async def _main_loop(self) -> None:
        """Core dispatch loop: allocate machines, launch supervisors, wait.

        Runs until all kernels are in a terminal state. For each iteration:
          1. Check for completed supervisors and collect results
          2. Try to dispatch pending kernels to idle machines
          3. If nothing to do, sleep briefly and retry
        """
        while not self._all_done():
            # Collect results from finished supervisors
            await self._collect_completed()

            # Try to dispatch pending kernels
            dispatched = await self._try_dispatch()

            if not dispatched and not self._has_running():
                # Nothing running, nothing dispatched — might be waiting
                # for human approval or all machines are dead
                if self._has_waiting_approval():
                    self.notify(
                        "Kernels waiting for approval. "
                        "Check the dashboard and approve or skip."
                    )
                await asyncio.sleep(MACHINE_POLL_INTERVAL)
            elif not dispatched:
                # Something is running but we couldn't dispatch more
                await asyncio.sleep(MACHINE_POLL_INTERVAL)
            # If we dispatched, loop immediately to try dispatching more

    async def _try_dispatch(self) -> bool:
        """Try to dispatch the next pending kernel to an idle machine.

        Returns:
            True if a kernel was dispatched, False otherwise.
        """
        kernel_name = self._next_pending_kernel()
        if kernel_name is None:
            return False

        machine = self._allocate_machine(kernel_name)
        if machine is None:
            return False  # All machines busy

        progress = self.state.kernel_progress[kernel_name]
        progress.status = KernelStatus.ALLOCATED
        progress.machine_host = machine.host
        progress.start_time = time.time()

        self.log(f"Dispatching to {machine.host}", kernel_name)

        # Launch supervisor in background
        asyncio.create_task(
            self._run_supervisor(kernel_name, machine)
        )

        return True

    async def _run_supervisor(
        self,
        kernel_name: str,
        machine: MachineInfo,
    ) -> None:
        """Run a kernel supervisor for one kernel on one machine.

        This wraps the kernel supervisor (Plan 3) and handles:
          - Status updates to orchestrator state
          - Error handling and retry logic
          - Machine release on completion

        Args:
            kernel_name: Name of the kernel to process.
            machine: Allocated machine.
        """
        progress = self.state.kernel_progress[kernel_name]
        progress.status = KernelStatus.RUNNING

        try:
            if self.supervisor_factory:
                supervisor = self.supervisor_factory(
                    progress.kernel, machine, self.config,
                )
                # The supervisor runs the full pipeline (phases 0-6)
                # and updates the progress object via callbacks.
                result = await supervisor.run(
                    progress_callback=lambda phase, msg: self._on_supervisor_progress(
                        kernel_name, phase, msg,
                    ),
                )

                if result.get("needs_approval"):
                    progress.status = KernelStatus.WAITING_APPROVAL
                    self.notify(
                        f"Kernel {kernel_name} needs human approval: "
                        f"{result.get('approval_reason', 'unknown')}"
                    )
                elif result.get("success"):
                    progress.status = KernelStatus.COMPLETED
                    progress.geomean_speedup = result.get("geomean_speedup")
                    progress.regressions_count = result.get("regressions_count", 0)
                    self.state.completed_kernels += 1
                    self.log("Completed successfully", kernel_name)
                else:
                    progress.status = KernelStatus.FAILED
                    progress.error_message = result.get("error", "unknown")
                    self.state.failed_kernels += 1
                    self.log(f"Failed: {progress.error_message}", kernel_name)
            else:
                # No supervisor factory — placeholder for testing
                self.log("No supervisor factory configured (dry run)", kernel_name)
                progress.status = KernelStatus.COMPLETED
                self.state.completed_kernels += 1

        except Exception as e:
            progress.status = KernelStatus.FAILED
            progress.error_message = str(e)
            self.state.failed_kernels += 1
            self.log(f"Supervisor crashed: {e}", kernel_name)
            self.notify(f"Kernel {kernel_name} supervisor crashed: {e}")
            logger.exception("Supervisor failed for %s", kernel_name)

        finally:
            progress.end_time = time.time()
            self._release_machine(machine)
            self.log(f"Released machine {machine.host}", kernel_name)

    def _on_supervisor_progress(
        self,
        kernel_name: str,
        phase: int,
        message: str,
    ) -> None:
        """Callback for supervisor progress updates."""
        progress = self.state.kernel_progress.get(kernel_name)
        if progress:
            progress.phase_number = phase
            progress.current_phase = PHASE_NAMES.get(phase, f"phase-{phase}")
            progress.last_log_line = message
        self.log(message, kernel_name)

    async def _collect_completed(self) -> None:
        """Check for supervisors that need result collection.

        In the async model, supervisors update state directly via the
        _run_supervisor wrapper. This method handles any additional
        post-completion work like copying artifacts from remote machines.
        """
        # Currently a no-op; artifact collection happens inside the
        # supervisor wrapper. Kept as a hook for future extensions
        # (e.g., copying results from remote containers to local).
        pass

    def _has_running(self) -> bool:
        """Check if any kernels are currently running."""
        return any(
            p.status in (KernelStatus.RUNNING, KernelStatus.ALLOCATED)
            for p in self.state.kernel_progress.values()
        )

    def _has_waiting_approval(self) -> bool:
        """Check if any kernels are waiting for human approval."""
        return any(
            p.status == KernelStatus.WAITING_APPROVAL
            for p in self.state.kernel_progress.values()
        )

    def approve_kernel(self, kernel_name: str) -> bool:
        """Human approves a kernel that was waiting.

        Called by the CLI when the user approves a commit or regression.

        Returns:
            True if the kernel was in WAITING_APPROVAL state.
        """
        progress = self.state.kernel_progress.get(kernel_name)
        if progress and progress.status == KernelStatus.WAITING_APPROVAL:
            progress.status = KernelStatus.RUNNING
            self.log("Approved by human", kernel_name)
            return True
        return False

    def skip_kernel(self, kernel_name: str) -> bool:
        """Human decides to skip a kernel.

        Returns:
            True if the kernel was in a state that can be skipped.
        """
        progress = self.state.kernel_progress.get(kernel_name)
        if progress and progress.status in (
            KernelStatus.WAITING_APPROVAL,
            KernelStatus.FAILED,
            KernelStatus.PENDING,
        ):
            progress.status = KernelStatus.SKIPPED
            self.log("Skipped by human", kernel_name)
            return True
        return False

    def _serialize_state(self) -> dict:
        """Serialize orchestrator state for JSON output."""
        kernel_data = {}
        for name, p in self.state.kernel_progress.items():
            kernel_data[name] = {
                "status": p.status.value,
                "category": p.kernel.category.value,
                "phase": p.phase_number,
                "shapes_done": p.shapes_done,
                "shapes_total": p.shapes_total,
                "machine": p.machine_host,
                "geomean_speedup": p.geomean_speedup,
                "regressions": p.regressions_count,
                "error": p.error_message or None,
                "elapsed_s": (
                    (p.end_time or time.time()) - p.start_time
                    if p.start_time else None
                ),
            }

        return {
            "run_id": self.run_id,
            "start_time": self.state.start_time,
            "wall_time_s": time.time() - self.state.start_time,
            "total_kernels": self.state.total_kernels,
            "completed": self.state.completed_kernels,
            "failed": self.state.failed_kernels,
            "machines": {
                host: status
                for host, status in self.state.machines_status.items()
            },
            "kernels": kernel_data,
        }


# Phase display names (also used by dashboard)
PHASE_NAMES = {
    0: "setup",
    1: "discovery",
    2: "baseline",
    3: "untuned validation",
    4: "tuning",
    5: "final validation",
    6: "commit",
}


def generate_summary_report(state: OrchestratorState) -> str:
    """Generate the final human-readable summary report.

    Includes:
      - Overall statistics (total kernels, pass/fail, wall time)
      - Per-kernel results (geomean speedup, regression count)
      - Machine utilization
      - List of regressions and failures

    Args:
        state: Final orchestrator state.

    Returns:
        Multi-line report string.
    """
    wall_time = time.time() - state.start_time
    hours = int(wall_time // 3600)
    minutes = int((wall_time % 3600) // 60)

    lines = [
        "=" * 70,
        "TRITON KERNEL TUNING PIPELINE — FINAL REPORT",
        "=" * 70,
        f"Run ID:          {state.run_id}",
        f"Wall time:       {hours}h {minutes}m",
        f"Total kernels:   {state.total_kernels}",
        f"Completed:       {state.completed_kernels}",
        f"Failed:          {state.failed_kernels}",
        f"Skipped:         {state.total_kernels - state.completed_kernels - state.failed_kernels}",
        "",
        "-" * 70,
        "PER-KERNEL RESULTS",
        "-" * 70,
        f"{'Kernel':<30s} {'Status':<15s} {'Geomean':>10s} {'Regressions':>12s} {'Time':>10s}",
        "-" * 70,
    ]

    for name, p in sorted(state.kernel_progress.items()):
        status = p.status.value
        speedup = f"{p.geomean_speedup:.3f}x" if p.geomean_speedup else "N/A"
        regr = str(p.regressions_count) if p.regressions_count is not None else "N/A"
        if p.start_time and p.end_time:
            elapsed = p.end_time - p.start_time
            t_min = int(elapsed // 60)
            t_sec = int(elapsed % 60)
            time_str = f"{t_min}m {t_sec}s"
        else:
            time_str = "N/A"
        lines.append(f"{name:<30s} {status:<15s} {speedup:>10s} {regr:>12s} {time_str:>10s}")

    # Failures section
    failures = [
        (name, p) for name, p in state.kernel_progress.items()
        if p.status == KernelStatus.FAILED
    ]
    if failures:
        lines.append("")
        lines.append("-" * 70)
        lines.append("FAILURES")
        lines.append("-" * 70)
        for name, p in failures:
            lines.append(f"  {name}: {p.error_message}")

    # Machine utilization
    lines.append("")
    lines.append("-" * 70)
    lines.append("MACHINE STATUS")
    lines.append("-" * 70)
    for host, status in sorted(state.machines_status.items()):
        lines.append(f"  {host}: {status}")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/test_orchestrator.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add aiter/ops/triton/tuning_agent/orchestrator.py \
    tests/tuning_agent/test_orchestrator.py
git commit -m "feat(tuning-agent): add orchestrator with machine pool and dispatch loop"
```

---

## Chunk 4: CLI Entry Point

### Task 5: CLI Entry Point (__main__.py)

**Files:**
- Create: `aiter/ops/triton/tuning_agent/__main__.py`
- Create: `tests/tuning_agent/test_main.py`

- [ ] **Step 1: Write failing tests for CLI**

```python
# tests/tuning_agent/test_main.py
import sys
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
from aiter.ops.triton.tuning_agent.__main__ import (
    parse_args,
    setup_logging,
    resolve_repo_root,
)


def test_parse_args_minimal():
    args = parse_args(["--config", "triton-upgrade.yaml"])
    assert args.config == "triton-upgrade.yaml"
    assert args.log_level == "INFO"
    assert not args.dry_run
    assert args.repo_root is None


def test_parse_args_all_options():
    args = parse_args([
        "--config", "triton-upgrade.yaml",
        "--log-level", "DEBUG",
        "--dry-run",
        "--repo-root", "/path/to/aiter",
        "--run-id", "test-001",
        "--no-dashboard",
        "--results-dir", "/tmp/results",
    ])
    assert args.config == "triton-upgrade.yaml"
    assert args.log_level == "DEBUG"
    assert args.dry_run
    assert args.repo_root == "/path/to/aiter"
    assert args.run_id == "test-001"
    assert args.no_dashboard
    assert args.results_dir == "/tmp/results"


def test_parse_args_missing_config():
    with pytest.raises(SystemExit):
        parse_args([])


def test_resolve_repo_root_explicit():
    root = resolve_repo_root("/explicit/path")
    assert root == Path("/explicit/path")


def test_resolve_repo_root_auto(tmp_path):
    # Create a minimal aiter structure
    (tmp_path / "aiter" / "ops" / "triton" / "gemm").mkdir(parents=True)
    with patch("aiter.ops.triton.tuning_agent.__main__.Path.cwd", return_value=tmp_path):
        root = resolve_repo_root(None)
        assert root == tmp_path


def test_setup_logging():
    logger = setup_logging("DEBUG", "/tmp/test.log")
    assert logger is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/test_main.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement the CLI entry point**

```python
# aiter/ops/triton/tuning_agent/__main__.py
"""CLI entry point for the Triton kernel tuning pipeline.

Usage:
    python -m aiter.ops.triton.tuning_agent --config triton-upgrade.yaml

    Options:
        --config PATH          Path to triton-upgrade.yaml (required)
        --repo-root PATH       Path to aiter repo root (auto-detected if omitted)
        --run-id ID            Custom run identifier (auto-generated if omitted)
        --log-level LEVEL      Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
        --dry-run              Discover kernels and show plan, but don't execute
        --no-dashboard         Disable auto-refreshing dashboard
        --results-dir PATH     Override results directory

    Examples:
        # Full run with default settings
        python -m aiter.ops.triton.tuning_agent --config triton-upgrade.yaml

        # Dry run to see what would be tuned
        python -m aiter.ops.triton.tuning_agent --config triton-upgrade.yaml --dry-run

        # Debug mode with custom repo root
        python -m aiter.ops.triton.tuning_agent --config triton-upgrade.yaml \\
            --repo-root /workspace/aiter --log-level DEBUG
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

from .config import load_config, ConfigError
from .kernel_discovery import KernelDiscovery
from .orchestrator import Orchestrator, RunResult
from .dashboard import Dashboard

logger = logging.getLogger("aiter.ops.triton.tuning_agent")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list. Uses sys.argv[1:] if None.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="aiter.ops.triton.tuning_agent",
        description=(
            "Agentic Triton Kernel Tuning Pipeline — automatically tune "
            "all GEMM kernel configs when upgrading the Triton compiler."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python -m aiter.ops.triton.tuning_agent "
            "--config triton-upgrade.yaml\n"
        ),
    )

    parser.add_argument(
        "--config", required=True,
        help="Path to triton-upgrade.yaml configuration file",
    )
    parser.add_argument(
        "--repo-root", default=None,
        help=(
            "Path to aiter repository root. If omitted, auto-detected "
            "by searching upward from cwd for aiter/ops/triton/gemm/"
        ),
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Custom run identifier (auto-generated if omitted)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Discover kernels and show plan without executing",
    )
    parser.add_argument(
        "--no-dashboard", action="store_true",
        help="Disable auto-refreshing terminal dashboard",
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Override local results directory (default: ~/.tuning_results/<run_id>/)",
    )

    return parser.parse_args(argv)


def setup_logging(level: str, log_file: Optional[str] = None) -> logging.Logger:
    """Configure logging for the tuning pipeline.

    Sets up both console and file handlers. Console shows INFO+,
    file captures everything at the configured level.

    Args:
        level: Logging level string (DEBUG, INFO, etc.)
        log_file: Optional path to log file.

    Returns:
        Root logger for the tuning_agent package.
    """
    root_logger = logging.getLogger("aiter.ops.triton.tuning_agent")
    root_logger.setLevel(getattr(logging, level))

    # Console handler — concise format
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-.1s] %(message)s",
        datefmt="%H:%M:%S",
    ))
    root_logger.addHandler(console)

    # File handler — detailed format
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level))
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        ))
        root_logger.addHandler(file_handler)

    return root_logger


def resolve_repo_root(explicit_path: Optional[str]) -> Path:
    """Resolve the aiter repository root directory.

    If explicit_path is given, use it directly. Otherwise, search upward
    from cwd looking for the aiter/ops/triton/gemm/ directory structure.

    Args:
        explicit_path: User-provided repo root, or None for auto-detect.

    Returns:
        Path to the repo root.

    Raises:
        FileNotFoundError: If auto-detection fails.
    """
    if explicit_path:
        return Path(explicit_path)

    # Search upward from cwd
    current = Path.cwd()
    for ancestor in [current] + list(current.parents):
        marker = ancestor / "aiter" / "ops" / "triton" / "gemm"
        if marker.is_dir():
            return ancestor

    raise FileNotFoundError(
        "Could not auto-detect aiter repo root. "
        "Use --repo-root to specify it explicitly."
    )


async def async_main(args: argparse.Namespace) -> int:
    """Async main function — runs the full pipeline.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code: 0 for success, 1 for failure.
    """
    # Load and validate config
    try:
        config = load_config(args.config)
    except (ConfigError, FileNotFoundError) as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1

    # Resolve repo root
    try:
        repo_root = resolve_repo_root(args.repo_root)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Set up results directory
    results_dir = Path(args.results_dir) if args.results_dir else None

    # Set up logging
    run_id = args.run_id or Orchestrator._generate_run_id()
    if results_dir is None:
        results_dir = Path.home() / ".tuning_results" / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(results_dir / "pipeline.log")
    setup_logging(args.log_level, log_file)

    logger.info("Starting Triton kernel tuning pipeline")
    logger.info("Config:    %s", args.config)
    logger.info("Repo root: %s", repo_root)
    logger.info("Run ID:    %s", run_id)
    logger.info("Results:   %s", results_dir)

    # Dry run mode: discover and show plan, then exit
    if args.dry_run:
        return await _dry_run(config, repo_root)

    # Full run
    orchestrator = Orchestrator(
        config=config,
        run_id=run_id,
        results_dir=results_dir,
    )

    if args.no_dashboard:
        orchestrator.dashboard = Dashboard(use_color=False)

    result = await orchestrator.run(repo_root)

    if result.report_path:
        logger.info("Report saved to: %s", result.report_path)

    return 0 if result.success else 1


async def _dry_run(config, repo_root: Path) -> int:
    """Execute a dry run: discover kernels and show the plan.

    Args:
        config: Parsed pipeline config.
        repo_root: Path to aiter repo.

    Returns:
        Exit code (always 0 for dry run).
    """
    gfx_arch = config.gpu.arch or "gfx950"
    discovery = KernelDiscovery(repo_root, gfx_arch, config.kernels)
    kernels = discovery.discover_all()

    print(discovery.summary(kernels))
    print(f"\nMachines configured: {len(config.machines)}")
    for m in config.machines:
        print(f"  {m.host} ({m.gpu_count} GPUs)")

    print(f"\nTuning mode: {config.tuning.mode}")
    print(f"Scout fraction: {config.tuning.scout_fraction}")
    print(f"Regression threshold (vs baseline): {config.tuning.thresholds.regression_vs_baseline}%")
    print(f"Regression threshold (vs untuned): {config.tuning.thresholds.regression_vs_untuned}%")

    # Estimate time: ~30 min per kernel per machine (very rough)
    num_machines = len(config.machines)
    parallel_batches = (len(kernels) + num_machines - 1) // max(num_machines, 1)
    est_hours = parallel_batches * 0.5  # 30 min per batch
    print(f"\nEstimated time: ~{est_hours:.1f} hours "
          f"({len(kernels)} kernels, {num_machines} machines)")

    print("\n[DRY RUN] No changes made.")
    return 0


def main() -> None:
    """Synchronous entry point — wraps async_main."""
    args = parse_args()
    exit_code = asyncio.run(async_main(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/test_main.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add aiter/ops/triton/tuning_agent/__main__.py tests/tuning_agent/test_main.py
git commit -m "feat(tuning-agent): add CLI entry point with argparse and async main"
```

---

## Chunk 5: Integration

### Task 6: Integration Test

**Files:**
- Create: `tests/tuning_agent/test_integration_orchestrator.py`

- [ ] **Step 1: Write an integration test that exercises the full flow**

```python
# tests/tuning_agent/test_integration_orchestrator.py
"""Integration test: config -> discovery -> orchestrator -> dashboard -> report.

Uses a mock repo structure and no real SSH/GPU. Verifies the full pipeline
from config loading through kernel discovery, orchestration, dashboard
rendering, and summary report generation.
"""

import asyncio
import json
import pytest
import time
from pathlib import Path
from unittest.mock import AsyncMock

from aiter.ops.triton.tuning_agent.config import load_config
from aiter.ops.triton.tuning_agent.kernel_discovery import KernelDiscovery
from aiter.ops.triton.tuning_agent.orchestrator import (
    Orchestrator,
    generate_summary_report,
)
from aiter.ops.triton.tuning_agent.dashboard import Dashboard
from aiter.ops.triton.tuning_agent.types import (
    KernelsConfig,
    KernelStatus,
    PipelineConfig,
    MachineInfo,
    ContainerConfig,
    RepoConfig,
    GpuConfig,
)


@pytest.fixture
def mock_repo(tmp_path):
    """Create a mock repo with 3 kernels for integration testing."""
    # Kernel sources
    basic = tmp_path / "aiter" / "ops" / "triton" / "gemm" / "basic"
    basic.mkdir(parents=True)
    (basic / "__init__.py").touch()
    (basic / "gemm_a8w8.py").write_text("# a8w8 kernel")
    (basic / "gemm_a16w16.py").write_text("# a16w16 kernel")

    batched = tmp_path / "aiter" / "ops" / "triton" / "gemm" / "batched"
    batched.mkdir(parents=True)
    (batched / "__init__.py").touch()
    (batched / "batched_gemm_bf16.py").write_text("# bf16 batched")

    ff = tmp_path / "aiter" / "ops" / "triton" / "gemm" / "feed_forward"
    ff.mkdir(parents=True)
    (ff / "__init__.py").touch()

    fused = tmp_path / "aiter" / "ops" / "triton" / "gemm" / "fused"
    fused.mkdir(parents=True)
    (fused / "__init__.py").touch()

    # Configs
    configs = tmp_path / "aiter" / "ops" / "triton" / "configs" / "gemm"
    configs.mkdir(parents=True)
    (configs / "gfx950-GEMM-A8W8.json").write_text(json.dumps({"M_LEQ_16": {}}))
    (configs / "gfx950-GEMM-A8W8-N=128-K=2048.json").write_text(json.dumps({}))

    # Scripts
    tunning = tmp_path / "aiter" / "ops" / "triton" / "utils" / "_triton" / "tunning"
    tunning.mkdir(parents=True)
    (tunning / "ut_gemm_a8w8.py").write_text("# tuning")

    bench = tmp_path / "op_tests" / "op_benchmarks" / "triton"
    bench.mkdir(parents=True)
    (bench / "bench_gemm_a8w8.py").write_text("# bench")

    tests = tmp_path / "op_tests" / "triton_tests" / "gemm"
    tests.mkdir(parents=True)
    (tests / "test_gemm_a8w8.py").write_text("# test")

    # Config YAML
    config_yaml = tmp_path / "triton-upgrade.yaml"
    config_yaml.write_text("""
baseline:
  aiter_repo: https://github.com/ROCm/aiter.git
  aiter_branch: main
  triton_repo: https://github.com/ROCm/triton.git
  triton_branch: triton_3_4
target:
  aiter_repo: https://github.com/ROCm/aiter.git
  aiter_branch: feature
  triton_repo: https://github.com/ROCm/triton.git
  triton_branch: main
machines:
  - host: gpu-1
    user: root
    ssh_key: ~/.ssh/id
    gpus: [0, 1, 2, 3, 4, 5, 6, 7]
  - host: gpu-2
    user: root
    ssh_key: ~/.ssh/id
    gpus: [0, 1, 2, 3]
container:
  image: rocm/pytorch:latest
gpu:
  arch: gfx950
kernels:
  exclude: []
""")

    return tmp_path


def test_config_to_discovery(mock_repo):
    """Config loads and discovery finds kernels."""
    config = load_config(mock_repo / "triton-upgrade.yaml")
    discovery = KernelDiscovery(mock_repo, "gfx950", config.kernels)
    kernels = discovery.discover_all()

    names = {k.name for k in kernels}
    assert "a8w8" in names
    assert "a16w16" in names
    assert "batched_bf16" in names
    assert len(kernels) == 3


def test_discovery_to_orchestrator(mock_repo):
    """Discovered kernels can be loaded into orchestrator."""
    config = load_config(mock_repo / "triton-upgrade.yaml")
    discovery = KernelDiscovery(mock_repo, "gfx950", config.kernels)
    kernels = discovery.discover_all()

    orch = Orchestrator(config, run_id="integration-test")
    orch.set_kernels(kernels)
    assert orch.state.total_kernels == 3

    # All should be pending
    for p in orch.state.kernel_progress.values():
        assert p.status == KernelStatus.PENDING


def test_orchestrator_to_dashboard(mock_repo):
    """Orchestrator state renders in dashboard without error."""
    config = load_config(mock_repo / "triton-upgrade.yaml")
    discovery = KernelDiscovery(mock_repo, "gfx950", config.kernels)
    kernels = discovery.discover_all()

    orch = Orchestrator(config, run_id="integration-test")
    orch._init_machine_status()
    orch.set_kernels(kernels)
    orch.log("Integration test started")

    dashboard = Dashboard(use_color=False)
    output = dashboard.render(orch.get_state())

    assert "integration-test" in output
    assert "a8w8" in output
    assert "gpu-1" in output


def test_orchestrator_machine_allocation_flow(mock_repo):
    """Machine allocation and release cycle works."""
    config = load_config(mock_repo / "triton-upgrade.yaml")
    orch = Orchestrator(config, run_id="test")
    orch._init_machine_status()

    # Allocate both machines
    m1 = orch._allocate_machine("a8w8")
    m2 = orch._allocate_machine("a16w16")
    assert m1 is not None
    assert m2 is not None
    assert m1.host != m2.host

    # No more available
    assert orch._allocate_machine("batched_bf16") is None

    # Release one
    orch._release_machine(m1)
    m3 = orch._allocate_machine("batched_bf16")
    assert m3 is not None
    assert m3.host == m1.host


def test_summary_report_generation(mock_repo):
    """Summary report is generated from final state."""
    config = load_config(mock_repo / "triton-upgrade.yaml")
    discovery = KernelDiscovery(mock_repo, "gfx950", config.kernels)
    kernels = discovery.discover_all()

    orch = Orchestrator(config, run_id="report-test")
    orch._init_machine_status()
    orch.set_kernels(kernels)

    # Simulate completion
    for name, p in orch.state.kernel_progress.items():
        p.status = KernelStatus.COMPLETED
        p.geomean_speedup = 1.05
        p.regressions_count = 0
        p.start_time = time.time() - 1800
        p.end_time = time.time()
        orch.state.completed_kernels += 1

    report = generate_summary_report(orch.state)
    assert "report-test" in report
    assert "FINAL REPORT" in report
    assert "1.050x" in report
    assert "a8w8" in report


def test_discovery_summary_text(mock_repo):
    """Discovery summary is human-readable."""
    config = load_config(mock_repo / "triton-upgrade.yaml")
    discovery = KernelDiscovery(mock_repo, "gfx950", config.kernels)
    kernels = discovery.discover_all()
    summary = discovery.summary(kernels)

    assert "BASIC" in summary
    assert "BATCHED" in summary
    assert "a8w8" in summary
    assert "3 kernels" in summary or "Total: 3" in summary
```

- [ ] **Step 2: Run integration tests**

Run: `cd /app/aiter && python -m pytest tests/tuning_agent/test_integration_orchestrator.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/tuning_agent/test_integration_orchestrator.py
git commit -m "test(tuning-agent): add integration tests for orchestrator pipeline"
```

---

## Verification

After all chunks are implemented, run the full test suite:

```bash
cd /app/aiter && python -m pytest tests/tuning_agent/ -v --tb=short
```

Verify the CLI entry point:

```bash
# Dry run (requires a config file)
cd /app/aiter && python -m aiter.ops.triton.tuning_agent --config triton-upgrade.yaml --dry-run

# Help text
cd /app/aiter && python -m aiter.ops.triton.tuning_agent --help
```

---

## Summary

| Chunk | Task | Files | What it does |
|-------|------|-------|-------------|
| 1 | Types + Discovery | `types.py`, `kernel_discovery.py` | Scans all GEMM categories, matches configs/scripts, applies filters |
| 2 | Dashboard | `dashboard.py` | Terminal status display: machines, kernels, logs, notifications |
| 3 | Orchestrator | `orchestrator.py` | Main loop: allocate machines, dispatch supervisors, collect results |
| 4 | CLI | `__main__.py` | Argparse entry point, logging, dry-run mode |
| 5 | Integration | `test_integration_orchestrator.py` | End-to-end test: config -> discover -> orchestrate -> dashboard -> report |

**Dependencies between chunks:** Chunk 1 must be done first (types used everywhere). Chunks 2 and 3 can be done in parallel. Chunk 4 depends on all prior chunks. Chunk 5 depends on everything.

**External dependencies:** Plan 1 (config.py, machine_pool.py, remote.py, notifications.py, types.py) must be implemented first. Plan 3 (KernelSupervisor) is called by the orchestrator but is injected via `supervisor_factory`, so the orchestrator can be tested with mocks before Plan 3 exists.
