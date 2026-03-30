"""Shared type definitions for the agentic Triton kernel tuning pipeline."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class MachineInfo:
    """Represents a remote GPU machine available for tuning."""

    host: str
    user: str
    ssh_key: str
    gpus: List[int] = field(default_factory=list)

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)


@dataclass
class ContainerConfig:
    """Configuration for container creation on a remote machine."""

    image: str
    run_script: Optional[str] = None


@dataclass
class RepoConfig:
    """Git repository configuration for aiter and Triton."""

    aiter_repo: str
    aiter_branch: str
    triton_repo: str
    triton_branch: str


@dataclass
class TuningThresholds:
    """Performance regression thresholds (as percentages)."""

    regression_vs_baseline: float = 5.0
    regression_vs_untuned: float = 2.0


@dataclass
class TuningTimeouts:
    """Timeout values in seconds for various pipeline stages."""

    command_default: int = 300
    tuning_per_shape: int = 1800
    progress_check: int = 120
    phase_max: int = 14400


@dataclass
class KernelOverrides:
    """Per-kernel tuning override options."""

    m_leq_16_bucket_name: Optional[str] = None
    extra_block_k: Optional[int] = None


@dataclass
class GpuConfig:
    """GPU architecture configuration."""

    arch: Optional[str] = None


@dataclass
class TritonInstallConfig:
    """Configuration for installing Triton inside a container."""

    command: str = "pip install -e ."


@dataclass
class TuningConfig:
    """Tuning run configuration."""

    mode: str = "full"
    scout_fraction: float = 0.1
    thresholds: TuningThresholds = field(default_factory=TuningThresholds)
    timeouts: TuningTimeouts = field(default_factory=TuningTimeouts)


@dataclass
class KernelsConfig:
    """Configuration for which kernels to tune and how."""

    exclude: List[str] = field(default_factory=list)
    include: List[str] = field(default_factory=list)
    overrides: Dict[str, KernelOverrides] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    baseline: RepoConfig
    target: RepoConfig
    machines: List[MachineInfo] = field(default_factory=list)
    container: ContainerConfig = field(default_factory=lambda: ContainerConfig(image=""))
    gpu: GpuConfig = field(default_factory=GpuConfig)
    triton_install: TritonInstallConfig = field(default_factory=TritonInstallConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    kernels: KernelsConfig = field(default_factory=KernelsConfig)


@dataclass
class ShapeResult:
    """Benchmark result for a single (m, n, k) shape."""

    m: int
    n: int
    k: int
    main_ns: float
    reduce_ns: float = 0.0

    @property
    def total_ns(self) -> float:
        return self.main_ns + self.reduce_ns


@dataclass
class ContainerState:
    """Runtime state of a container deployed on a machine."""

    container_id: str
    machine: MachineInfo
    triton_commit: Optional[str] = None
    aiter_commit: Optional[str] = None
    is_ready: bool = False
    artifact_dir: Optional[str] = None
