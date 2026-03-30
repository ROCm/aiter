# tuning_agent/test_types.py
from aiter.ops.triton.utils._triton.tunning.tuning_agent.types import (
    MachineInfo, ContainerConfig, RepoConfig, TuningThresholds,
    TuningTimeouts, KernelOverrides, PipelineConfig, ShapeResult, ContainerState,
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
    r = RepoConfig(aiter_repo="https://github.com/ROCm/aiter.git", aiter_branch="main",
                   triton_repo="https://github.com/ROCm/triton.git", triton_branch="triton_3_4")
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
        host="gpu1", user="root", ssh_key="~/.ssh/id", gpus=[0]))
    assert c.container_id == "abc123"
    assert not c.is_ready
