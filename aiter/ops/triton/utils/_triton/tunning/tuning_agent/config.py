"""YAML config parsing with validation for the agentic Triton kernel tuning pipeline."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import yaml

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
    TuningThresholds,
    TuningTimeouts,
)

# Tuning modes that are accepted by the pipeline.
_VALID_TUNING_MODES = {"regression_only", "full"}


class ConfigError(Exception):
    """Raised when a configuration file is semantically invalid."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_repo_config(data: Dict[str, Any], section: str) -> RepoConfig:
    """Parse a RepoConfig from a raw YAML mapping."""
    required = ("aiter_repo", "aiter_branch", "triton_repo", "triton_branch")
    for key in required:
        if key not in data:
            raise ConfigError(
                f"Section '{section}' is missing required field '{key}'."
            )
    return RepoConfig(
        aiter_repo=data["aiter_repo"],
        aiter_branch=data["aiter_branch"],
        triton_repo=data["triton_repo"],
        triton_branch=data["triton_branch"],
    )


def _parse_machine(data: Dict[str, Any], index: int) -> MachineInfo:
    required = ("host", "user", "ssh_key")
    for key in required:
        if key not in data:
            raise ConfigError(
                f"machines[{index}] is missing required field '{key}'."
            )
    return MachineInfo(
        host=data["host"],
        user=data["user"],
        ssh_key=data["ssh_key"],
        gpus=list(data.get("gpus", [])),
    )


def _parse_container(data: Dict[str, Any]) -> ContainerConfig:
    if "image" not in data:
        raise ConfigError("Section 'container' is missing required field 'image'.")
    return ContainerConfig(
        image=data["image"],
        run_script=data.get("run_script"),
    )


def _parse_gpu(data: Dict[str, Any]) -> GpuConfig:
    return GpuConfig(arch=data.get("arch"))


def _parse_triton_install(data: Dict[str, Any]) -> TritonInstallConfig:
    return TritonInstallConfig(
        command=data.get("command", TritonInstallConfig.command),
    )


def _parse_thresholds(data: Dict[str, Any]) -> TuningThresholds:
    defaults = TuningThresholds()
    return TuningThresholds(
        regression_vs_baseline=float(
            data.get("regression_vs_baseline", defaults.regression_vs_baseline)
        ),
        regression_vs_untuned=float(
            data.get("regression_vs_untuned", defaults.regression_vs_untuned)
        ),
    )


def _parse_timeouts(data: Dict[str, Any]) -> TuningTimeouts:
    defaults = TuningTimeouts()
    return TuningTimeouts(
        command_default=int(
            data.get("command_default", defaults.command_default)
        ),
        tuning_per_shape=int(
            data.get("tuning_per_shape", defaults.tuning_per_shape)
        ),
        progress_check=int(
            data.get("progress_check", defaults.progress_check)
        ),
        phase_max=int(data.get("phase_max", defaults.phase_max)),
    )


def _parse_tuning(data: Dict[str, Any]) -> TuningConfig:
    defaults = TuningConfig()
    mode = data.get("mode", defaults.mode)
    if mode not in _VALID_TUNING_MODES:
        raise ConfigError(
            f"Invalid tuning mode '{mode}'. "
            f"Allowed values: {sorted(_VALID_TUNING_MODES)}."
        )
    thresholds_data = data.get("thresholds", {})
    timeouts_data = data.get("timeouts", {})
    return TuningConfig(
        mode=mode,
        scout_fraction=float(data.get("scout_fraction", defaults.scout_fraction)),
        thresholds=_parse_thresholds(thresholds_data),
        timeouts=_parse_timeouts(timeouts_data),
    )


def _parse_kernel_overrides(data: Dict[str, Any]) -> KernelOverrides:
    return KernelOverrides(
        m_leq_16_bucket_name=data.get("m_leq_16_bucket_name"),
        extra_block_k=data.get("extra_block_k"),
    )


def _parse_kernels(data: Dict[str, Any]) -> KernelsConfig:
    overrides_raw: Dict[str, Any] = data.get("overrides", {})
    overrides = {
        name: _parse_kernel_overrides(override_data)
        for name, override_data in overrides_raw.items()
    }
    return KernelsConfig(
        exclude=list(data.get("exclude", [])),
        include=list(data.get("include", [])),
        overrides=overrides,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(path: str) -> PipelineConfig:
    """Load and validate a pipeline config from *path*.

    Parameters
    ----------
    path:
        Filesystem path to a YAML configuration file.

    Returns
    -------
    PipelineConfig
        A fully-populated :class:`PipelineConfig` dataclass instance.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ConfigError
        If the YAML is structurally or semantically invalid.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw: Any = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ConfigError("Config file must contain a YAML mapping at the top level.")

    # ---- Required sections ------------------------------------------------
    for required_section in ("baseline", "target", "machines", "container"):
        if required_section not in raw:
            raise ConfigError(
                f"Required section '{required_section}' is missing from the config."
            )

    baseline = _parse_repo_config(raw["baseline"], "baseline")
    target = _parse_repo_config(raw["target"], "target")

    machines_raw: List[Any] = raw["machines"]
    if not machines_raw:
        raise ConfigError(
            "The 'machines' section must contain at least one machine entry."
        )
    machines = [_parse_machine(m, i) for i, m in enumerate(machines_raw)]

    container = _parse_container(raw["container"])

    # ---- Optional sections with defaults ----------------------------------
    gpu = _parse_gpu(raw["gpu"]) if "gpu" in raw else GpuConfig()
    triton_install = (
        _parse_triton_install(raw["triton_install"])
        if "triton_install" in raw
        else TritonInstallConfig()
    )
    tuning = _parse_tuning(raw["tuning"]) if "tuning" in raw else TuningConfig()
    kernels = _parse_kernels(raw["kernels"]) if "kernels" in raw else KernelsConfig()

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
