"""Tests for YAML config parsing (TDD — written before implementation)."""

import os
import pytest

from .config import load_config, ConfigError
from .types import (
    PipelineConfig,
    RepoConfig,
    MachineInfo,
    ContainerConfig,
    GpuConfig,
    TritonInstallConfig,
    TuningConfig,
    TuningThresholds,
    TuningTimeouts,
    KernelsConfig,
    KernelOverrides,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def fixture(name: str) -> str:
    return os.path.join(FIXTURES_DIR, name)


# ---------------------------------------------------------------------------
# test_load_valid_config
# ---------------------------------------------------------------------------


class TestLoadValidConfig:
    """Full config with all sections is parsed into a PipelineConfig."""

    def setup_method(self):
        self.cfg = load_config(fixture("valid_config.yaml"))

    def test_returns_pipeline_config(self):
        assert isinstance(self.cfg, PipelineConfig)

    # baseline
    def test_baseline_is_repo_config(self):
        assert isinstance(self.cfg.baseline, RepoConfig)

    def test_baseline_aiter_repo(self):
        assert self.cfg.baseline.aiter_repo == "https://github.com/ROCm/aiter.git"

    def test_baseline_aiter_branch(self):
        assert self.cfg.baseline.aiter_branch == "main"

    def test_baseline_triton_repo(self):
        assert self.cfg.baseline.triton_repo == "https://github.com/triton-lang/triton.git"

    def test_baseline_triton_branch(self):
        assert self.cfg.baseline.triton_branch == "main"

    # target
    def test_target_aiter_branch(self):
        assert self.cfg.target.aiter_branch == "feature/new-kernels"

    def test_target_triton_branch(self):
        assert self.cfg.target.triton_branch == "feature/rocm-improvements"

    # machines
    def test_machines_count(self):
        assert len(self.cfg.machines) == 2

    def test_first_machine_is_machine_info(self):
        assert isinstance(self.cfg.machines[0], MachineInfo)

    def test_first_machine_host(self):
        assert self.cfg.machines[0].host == "gpu-node-01.example.com"

    def test_first_machine_user(self):
        assert self.cfg.machines[0].user == "benchuser"

    def test_first_machine_ssh_key(self):
        assert self.cfg.machines[0].ssh_key == "/home/ci/.ssh/id_ed25519"

    def test_first_machine_gpus(self):
        assert self.cfg.machines[0].gpus == [0, 1, 2, 3]

    def test_second_machine_host(self):
        assert self.cfg.machines[1].host == "gpu-node-02.example.com"

    # container
    def test_container_is_container_config(self):
        assert isinstance(self.cfg.container, ContainerConfig)

    def test_container_image(self):
        assert self.cfg.container.image == "rocm/pytorch:latest"

    def test_container_run_script(self):
        assert self.cfg.container.run_script == "/opt/run_container.sh"

    # gpu
    def test_gpu_arch(self):
        assert self.cfg.gpu.arch == "gfx942"

    # triton_install
    def test_triton_install_command(self):
        assert self.cfg.triton_install.command == "pip install -e '.[rocm]'"

    # tuning
    def test_tuning_mode(self):
        assert self.cfg.tuning.mode == "full"

    def test_tuning_scout_fraction(self):
        assert self.cfg.tuning.scout_fraction == pytest.approx(0.15)

    def test_tuning_thresholds_regression_vs_baseline(self):
        assert self.cfg.tuning.thresholds.regression_vs_baseline == pytest.approx(3.0)

    def test_tuning_thresholds_regression_vs_untuned(self):
        assert self.cfg.tuning.thresholds.regression_vs_untuned == pytest.approx(1.5)

    def test_tuning_timeouts_command_default(self):
        assert self.cfg.tuning.timeouts.command_default == 600

    def test_tuning_timeouts_tuning_per_shape(self):
        assert self.cfg.tuning.timeouts.tuning_per_shape == 3600

    def test_tuning_timeouts_progress_check(self):
        assert self.cfg.tuning.timeouts.progress_check == 180

    def test_tuning_timeouts_phase_max(self):
        assert self.cfg.tuning.timeouts.phase_max == 28800

    # kernels
    def test_kernels_exclude(self):
        assert self.cfg.kernels.exclude == ["deprecated_kernel_v1"]

    def test_kernels_include(self):
        assert self.cfg.kernels.include == ["flash_attention", "gemm"]

    def test_kernels_overrides_flash_attention_bucket(self):
        override = self.cfg.kernels.overrides.get("flash_attention")
        assert override is not None
        assert override.m_leq_16_bucket_name == "small_m_bucket"

    def test_kernels_overrides_flash_attention_extra_block_k(self):
        override = self.cfg.kernels.overrides.get("flash_attention")
        assert override is not None
        assert override.extra_block_k == 64


# ---------------------------------------------------------------------------
# test_load_minimal_config
# ---------------------------------------------------------------------------


class TestLoadMinimalConfig:
    """Only required sections are present; all optional fields should get defaults."""

    def setup_method(self):
        self.cfg = load_config(fixture("minimal_config.yaml"))

    def test_returns_pipeline_config(self):
        assert isinstance(self.cfg, PipelineConfig)

    def test_baseline_present(self):
        assert self.cfg.baseline.aiter_branch == "main"

    def test_target_present(self):
        assert self.cfg.target.aiter_branch == "dev"

    def test_machines_has_one_entry(self):
        assert len(self.cfg.machines) == 1

    def test_container_image(self):
        assert self.cfg.container.image == "rocm/pytorch:latest"

    def test_container_run_script_default_is_none(self):
        assert self.cfg.container.run_script is None

    # Optional section defaults
    def test_gpu_arch_default_is_none(self):
        assert self.cfg.gpu.arch is None

    def test_triton_install_command_default(self):
        assert self.cfg.triton_install.command == "pip install -e ."

    def test_tuning_mode_default(self):
        assert self.cfg.tuning.mode == "full"

    def test_tuning_scout_fraction_default(self):
        assert self.cfg.tuning.scout_fraction == pytest.approx(0.1)

    def test_tuning_thresholds_default_regression_vs_baseline(self):
        assert self.cfg.tuning.thresholds.regression_vs_baseline == pytest.approx(5.0)

    def test_tuning_thresholds_default_regression_vs_untuned(self):
        assert self.cfg.tuning.thresholds.regression_vs_untuned == pytest.approx(2.0)

    def test_tuning_timeouts_default_command_default(self):
        assert self.cfg.tuning.timeouts.command_default == 300

    def test_tuning_timeouts_default_tuning_per_shape(self):
        assert self.cfg.tuning.timeouts.tuning_per_shape == 1800

    def test_tuning_timeouts_default_progress_check(self):
        assert self.cfg.tuning.timeouts.progress_check == 120

    def test_tuning_timeouts_default_phase_max(self):
        assert self.cfg.tuning.timeouts.phase_max == 14400

    def test_kernels_exclude_default_empty(self):
        assert self.cfg.kernels.exclude == []

    def test_kernels_include_default_empty(self):
        assert self.cfg.kernels.include == []

    def test_kernels_overrides_default_empty(self):
        assert self.cfg.kernels.overrides == {}


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


class TestLoadInvalidConfigMissingBaseline:
    """Config without 'baseline' section raises ConfigError."""

    def test_raises_config_error(self):
        with pytest.raises(ConfigError, match="baseline"):
            load_config(fixture("invalid_config.yaml"))


class TestLoadInvalidConfigNoMachines:
    """Config with an empty machines list raises ConfigError."""

    def test_raises_config_error(self, tmp_path):
        cfg_file = tmp_path / "no_machines.yaml"
        cfg_file.write_text(
            "baseline:\n"
            "  aiter_repo: 'https://github.com/ROCm/aiter.git'\n"
            "  aiter_branch: main\n"
            "  triton_repo: 'https://github.com/triton-lang/triton.git'\n"
            "  triton_branch: main\n"
            "target:\n"
            "  aiter_repo: 'https://github.com/ROCm/aiter.git'\n"
            "  aiter_branch: dev\n"
            "  triton_repo: 'https://github.com/triton-lang/triton.git'\n"
            "  triton_branch: dev\n"
            "machines: []\n"
            "container:\n"
            "  image: 'rocm/pytorch:latest'\n"
        )
        with pytest.raises(ConfigError, match="machines"):
            load_config(str(cfg_file))


class TestLoadNonexistentConfig:
    """A path that does not exist raises FileNotFoundError."""

    def test_raises_file_not_found_error(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")


class TestTuningModeValidation:
    """An unrecognised tuning mode raises ConfigError."""

    def test_invalid_mode_raises_config_error(self, tmp_path):
        cfg_file = tmp_path / "bad_mode.yaml"
        cfg_file.write_text(
            "baseline:\n"
            "  aiter_repo: 'https://github.com/ROCm/aiter.git'\n"
            "  aiter_branch: main\n"
            "  triton_repo: 'https://github.com/triton-lang/triton.git'\n"
            "  triton_branch: main\n"
            "target:\n"
            "  aiter_repo: 'https://github.com/ROCm/aiter.git'\n"
            "  aiter_branch: dev\n"
            "  triton_repo: 'https://github.com/triton-lang/triton.git'\n"
            "  triton_branch: dev\n"
            "machines:\n"
            "  - host: gpu-node-01.example.com\n"
            "    user: ci\n"
            "    ssh_key: /home/ci/.ssh/id_rsa\n"
            "    gpus: [0]\n"
            "container:\n"
            "  image: 'rocm/pytorch:latest'\n"
            "tuning:\n"
            "  mode: bogus_mode\n"
        )
        with pytest.raises(ConfigError, match="mode"):
            load_config(str(cfg_file))

    def test_regression_only_mode_is_valid(self, tmp_path):
        cfg_file = tmp_path / "regression_only.yaml"
        cfg_file.write_text(
            "baseline:\n"
            "  aiter_repo: 'https://github.com/ROCm/aiter.git'\n"
            "  aiter_branch: main\n"
            "  triton_repo: 'https://github.com/triton-lang/triton.git'\n"
            "  triton_branch: main\n"
            "target:\n"
            "  aiter_repo: 'https://github.com/ROCm/aiter.git'\n"
            "  aiter_branch: dev\n"
            "  triton_repo: 'https://github.com/triton-lang/triton.git'\n"
            "  triton_branch: dev\n"
            "machines:\n"
            "  - host: gpu-node-01.example.com\n"
            "    user: ci\n"
            "    ssh_key: /home/ci/.ssh/id_rsa\n"
            "    gpus: [0]\n"
            "container:\n"
            "  image: 'rocm/pytorch:latest'\n"
            "tuning:\n"
            "  mode: regression_only\n"
        )
        cfg = load_config(str(cfg_file))
        assert cfg.tuning.mode == "regression_only"
