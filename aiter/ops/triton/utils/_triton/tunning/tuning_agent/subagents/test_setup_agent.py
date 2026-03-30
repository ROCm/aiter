"""Tests for SetupAgent._execute()."""

import subprocess
from unittest.mock import MagicMock, call, patch

import pytest

from .setup_agent import SetupAgent
from ..types import RepoConfig, TritonInstallConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONTAINER_ID = "abc123def456"


def _make_completed_process(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


def _make_executor(container_id: str = CONTAINER_ID) -> MagicMock:
    """Return a mock RemoteExecutor whose key methods are pre-configured."""
    executor = MagicMock()
    executor.container_id = container_id

    # create_container sets container_id and returns it.
    def _create_container(**kwargs):
        executor.container_id = container_id
        return container_id

    executor.create_container.side_effect = _create_container

    # docker_exec returns a successful CompletedProcess by default.
    executor.docker_exec.return_value = _make_completed_process()

    # verify_environment returns version info.
    executor.verify_environment.return_value = {
        "triton_version": "3.1.0",
        "aiter_branch": "main",
    }

    return executor


def _make_agent(
    executor: MagicMock,
    *,
    run_script: str = "",
    kernel_name: str = "fmha",
    image: str = "rocm/pytorch:latest",
    aiter_repo: str = "https://github.com/ROCm/aiter.git",
    aiter_branch: str = "main",
    triton_repo: str = "https://github.com/triton-lang/triton.git",
    triton_branch: str = "release/3.1.x",
    triton_command: str = "pip install -e python/",
) -> SetupAgent:
    repo_config = RepoConfig(
        aiter_repo=aiter_repo,
        aiter_branch=aiter_branch,
        triton_repo=triton_repo,
        triton_branch=triton_branch,
    )
    triton_install_config = TritonInstallConfig(command=triton_command)
    return SetupAgent(
        executor=executor,
        kernel_name=kernel_name,
        artifact_dir="/artifacts",
        image=image,
        run_script=run_script,
        repo_config=repo_config,
        triton_install_config=triton_install_config,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSetupAgentCreateContainer:
    """Container creation with and without run_script."""

    def test_create_container_with_run_script(self):
        executor = _make_executor()
        agent = _make_agent(executor, run_script="/scripts/start.sh")

        agent._execute()

        executor.create_container.assert_called_once_with(
            image="rocm/pytorch:latest",
            run_script="/scripts/start.sh",
        )

    def test_create_container_without_run_script_uses_kernel_name(self):
        executor = _make_executor()
        agent = _make_agent(executor, run_script="", kernel_name="flash_attn")

        agent._execute()

        executor.create_container.assert_called_once_with(
            image="rocm/pytorch:latest",
            name="tuning-flash_attn",
        )

    def test_create_container_uses_provided_image(self):
        executor = _make_executor()
        agent = _make_agent(executor, image="custom/image:v2", run_script="")

        agent._execute()

        _, kwargs = executor.create_container.call_args
        assert kwargs["image"] == "custom/image:v2"


class TestSetupAgentGitClone:
    """Git clone commands are issued for both repos."""

    def test_aiter_repo_cloned_with_correct_branch(self):
        executor = _make_executor()
        agent = _make_agent(
            executor,
            aiter_repo="https://github.com/ROCm/aiter.git",
            aiter_branch="feature/my-branch",
        )

        agent._execute()

        clone_calls = [str(c) for c in executor.docker_exec.call_args_list]
        assert any(
            "https://github.com/ROCm/aiter.git" in c
            and "feature/my-branch" in c
            and "/workspace/aiter" in c
            for c in clone_calls
        ), f"Expected aiter clone call not found. Calls: {clone_calls}"

    def test_triton_repo_cloned_with_correct_branch(self):
        executor = _make_executor()
        agent = _make_agent(
            executor,
            triton_repo="https://github.com/triton-lang/triton.git",
            triton_branch="release/3.1.x",
        )

        agent._execute()

        clone_calls = [str(c) for c in executor.docker_exec.call_args_list]
        assert any(
            "https://github.com/triton-lang/triton.git" in c
            and "release/3.1.x" in c
            and "/workspace/triton" in c
            for c in clone_calls
        ), f"Expected triton clone call not found. Calls: {clone_calls}"

    def test_both_repos_cloned(self):
        executor = _make_executor()
        agent = _make_agent(executor)

        agent._execute()

        all_commands = [str(c) for c in executor.docker_exec.call_args_list]
        aiter_cloned = any("/workspace/aiter" in c and "git clone" in c for c in all_commands)
        triton_cloned = any("/workspace/triton" in c and "git clone" in c for c in all_commands)

        assert aiter_cloned, "aiter repo was not cloned"
        assert triton_cloned, "triton repo was not cloned"

    def test_clone_uses_single_branch_flag(self):
        executor = _make_executor()
        agent = _make_agent(executor)

        agent._execute()

        clone_calls = [str(c) for c in executor.docker_exec.call_args_list]
        aiter_clone = next((c for c in clone_calls if "/workspace/aiter" in c and "git clone" in c), None)
        triton_clone = next((c for c in clone_calls if "/workspace/triton" in c and "git clone" in c), None)

        assert aiter_clone is not None and "--single-branch" in aiter_clone
        assert triton_clone is not None and "--single-branch" in triton_clone


class TestSetupAgentTritonInstall:
    """Triton install command is executed correctly."""

    def test_triton_install_command_executed(self):
        executor = _make_executor()
        agent = _make_agent(executor, triton_command="pip install -e python/")

        agent._execute()

        all_commands = [str(c) for c in executor.docker_exec.call_args_list]
        assert any(
            "/workspace/triton" in c and "pip install -e python/" in c
            for c in all_commands
        ), f"Triton install command not found. Calls: {all_commands}"

    def test_triton_install_uses_custom_command(self):
        executor = _make_executor()
        agent = _make_agent(executor, triton_command="pip install triton==3.2.0")

        agent._execute()

        all_commands = [str(c) for c in executor.docker_exec.call_args_list]
        assert any(
            "pip install triton==3.2.0" in c for c in all_commands
        ), f"Custom triton install not found. Calls: {all_commands}"

    def test_triton_install_run_in_triton_directory(self):
        executor = _make_executor()
        agent = _make_agent(executor)

        agent._execute()

        all_commands = [str(c) for c in executor.docker_exec.call_args_list]
        triton_install_call = next(
            (c for c in all_commands if "/workspace/triton" in c and "pip install" in c),
            None,
        )
        assert triton_install_call is not None, "Triton install not run in /workspace/triton"


class TestSetupAgentAiterInstall:
    """Aiter pip install is executed."""

    def test_aiter_installed(self):
        executor = _make_executor()
        agent = _make_agent(executor)

        agent._execute()

        all_commands = [str(c) for c in executor.docker_exec.call_args_list]
        assert any(
            "/workspace/aiter" in c and "pip install -e ." in c
            for c in all_commands
        ), f"Aiter install not found. Calls: {all_commands}"


class TestSetupAgentVerifyEnvironment:
    """verify_environment is called."""

    def test_verify_environment_called(self):
        executor = _make_executor()
        agent = _make_agent(executor)

        agent._execute()

        executor.verify_environment.assert_called_once()


class TestSetupAgentReturnValue:
    """Return value contains expected keys."""

    def test_returns_container_id(self):
        executor = _make_executor(container_id="mycontainer999")
        agent = _make_agent(executor)

        result = agent._execute()

        assert result["container_id"] == "mycontainer999"

    def test_returns_triton_version(self):
        executor = _make_executor()
        executor.verify_environment.return_value = {
            "triton_version": "3.2.1",
            "aiter_branch": "main",
        }
        agent = _make_agent(executor)

        result = agent._execute()

        assert result["triton_version"] == "3.2.1"

    def test_returns_aiter_branch(self):
        executor = _make_executor()
        executor.verify_environment.return_value = {
            "triton_version": "3.1.0",
            "aiter_branch": "feature/rocm-6.x",
        }
        agent = _make_agent(executor)

        result = agent._execute()

        assert result["aiter_branch"] == "feature/rocm-6.x"

    def test_result_has_all_required_keys(self):
        executor = _make_executor()
        agent = _make_agent(executor)

        result = agent._execute()

        assert "container_id" in result
        assert "triton_version" in result
        assert "aiter_branch" in result


class TestSetupAgentExecutionOrder:
    """Commands are issued in the correct sequence."""

    def test_container_created_before_docker_exec(self):
        """create_container must be called before any docker_exec calls."""
        call_order = []

        executor = _make_executor()
        executor.create_container.side_effect = lambda **kwargs: (
            call_order.append("create_container") or CONTAINER_ID
        )
        executor.docker_exec.side_effect = lambda cmd, **kwargs: (
            call_order.append("docker_exec") or _make_completed_process()
        )

        agent = _make_agent(executor)
        agent._execute()

        assert call_order[0] == "create_container", (
            f"create_container must be first; got order: {call_order}"
        )

    def test_triton_installed_before_aiter(self):
        """Triton installation must precede aiter pip install."""
        commands_seen = []

        executor = _make_executor()
        executor.docker_exec.side_effect = lambda cmd, **kwargs: (
            commands_seen.append(cmd) or _make_completed_process()
        )

        agent = _make_agent(executor, triton_command="pip install -e python/")
        agent._execute()

        triton_idx = next(
            (i for i, c in enumerate(commands_seen) if "/workspace/triton" in c and "pip install" in c),
            None,
        )
        aiter_idx = next(
            (i for i, c in enumerate(commands_seen) if "/workspace/aiter" in c and "pip install -e ." in c),
            None,
        )

        assert triton_idx is not None, "Triton install step not found"
        assert aiter_idx is not None, "Aiter install step not found"
        assert triton_idx < aiter_idx, (
            f"Triton install (idx {triton_idx}) must precede aiter install (idx {aiter_idx})"
        )
