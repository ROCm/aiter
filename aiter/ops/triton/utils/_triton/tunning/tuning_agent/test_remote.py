"""Tests for the SSH + Docker exec remote execution wrapper."""

import subprocess
import unittest
from unittest.mock import MagicMock, call, patch

from .remote import RemoteCommandError, RemoteExecutor
from .types import MachineInfo


def _make_machine(**kwargs):
    defaults = {
        "host": "gpu-host.example.com",
        "user": "testuser",
        "ssh_key": "/home/testuser/.ssh/id_rsa",
        "gpus": [0, 1],
    }
    defaults.update(kwargs)
    return MachineInfo(**defaults)


def _completed(returncode=0, stdout="", stderr=""):
    r = MagicMock(spec=subprocess.CompletedProcess)
    r.returncode = returncode
    r.stdout = stdout
    r.stderr = stderr
    return r


class TestBuildSshCommand(unittest.TestCase):
    def setUp(self):
        self.machine = _make_machine()
        self.executor = RemoteExecutor(self.machine)

    def test_contains_identity_flag(self):
        cmd = self.executor._build_ssh_command("echo hi")
        self.assertIn("-i", cmd)
        idx = cmd.index("-i")
        self.assertEqual(cmd[idx + 1], "/home/testuser/.ssh/id_rsa")

    def test_strict_host_checking_no(self):
        cmd = self.executor._build_ssh_command("echo hi")
        self.assertIn("-o", cmd)
        opts = " ".join(cmd)
        self.assertIn("StrictHostKeyChecking=no", opts)

    def test_connect_timeout(self):
        cmd = self.executor._build_ssh_command("echo hi")
        opts = " ".join(cmd)
        self.assertIn("ConnectTimeout=10", opts)

    def test_batch_mode(self):
        cmd = self.executor._build_ssh_command("echo hi")
        opts = " ".join(cmd)
        self.assertIn("BatchMode=yes", opts)

    def test_user_at_host(self):
        cmd = self.executor._build_ssh_command("echo hi")
        self.assertIn("testuser@gpu-host.example.com", cmd)

    def test_command_is_last_element(self):
        cmd = self.executor._build_ssh_command("echo hi")
        self.assertEqual(cmd[-1], "echo hi")


class TestSshRun(unittest.TestCase):
    def setUp(self):
        self.machine = _make_machine()
        self.executor = RemoteExecutor(self.machine)

    @patch("subprocess.run")
    def test_success_returns_completed_process(self, mock_run):
        mock_run.return_value = _completed(returncode=0, stdout="hello\n")
        result = self.executor.ssh_run("echo hello")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "hello\n")

    @patch("subprocess.run")
    def test_check_false_non_zero_does_not_raise(self, mock_run):
        mock_run.return_value = _completed(returncode=1, stderr="error")
        result = self.executor.ssh_run("false", check=False)
        self.assertEqual(result.returncode, 1)

    @patch("subprocess.run")
    def test_check_true_non_zero_raises(self, mock_run):
        mock_run.return_value = _completed(returncode=1, stderr="bad")
        with self.assertRaises(RemoteCommandError) as ctx:
            self.executor.ssh_run("false", check=True)
        self.assertEqual(ctx.exception.returncode, 1)
        self.assertEqual(ctx.exception.stderr, "bad")

    @patch("subprocess.run")
    def test_timeout_raises_remote_command_error(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["ssh"], timeout=5)
        with self.assertRaises(RemoteCommandError) as ctx:
            self.executor.ssh_run("sleep 100", timeout=5)
        self.assertIn("timed out", str(ctx.exception).lower())

    @patch("time.sleep")
    @patch("subprocess.run")
    def test_retry_on_exit_code_255(self, mock_run, mock_sleep):
        # First two calls return 255, third succeeds
        mock_run.side_effect = [
            _completed(returncode=255, stderr="Connection refused"),
            _completed(returncode=255, stderr="Connection refused"),
            _completed(returncode=0, stdout="ok"),
        ]
        result = self.executor.ssh_run("echo ok", retries=2, backoff=1.0)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(mock_run.call_count, 3)

    @patch("time.sleep")
    @patch("subprocess.run")
    def test_retry_exhausted_raises(self, mock_run, mock_sleep):
        mock_run.return_value = _completed(returncode=255, stderr="refused")
        with self.assertRaises(RemoteCommandError) as ctx:
            self.executor.ssh_run("echo ok", retries=2, backoff=1.0)
        self.assertEqual(ctx.exception.returncode, 255)
        self.assertEqual(mock_run.call_count, 3)  # initial + 2 retries

    @patch("time.sleep")
    @patch("subprocess.run")
    def test_exponential_backoff_sleep_calls(self, mock_run, mock_sleep):
        mock_run.return_value = _completed(returncode=255, stderr="refused")
        try:
            self.executor.ssh_run("echo ok", retries=3, backoff=2.0)
        except RemoteCommandError:
            pass
        # backoff: 2^0=1, 2^1=2, 2^2=4 multiplied by backoff factor progression
        self.assertEqual(mock_sleep.call_count, 3)


class TestDockerExec(unittest.TestCase):
    def setUp(self):
        self.machine = _make_machine()
        self.executor = RemoteExecutor(self.machine)
        self.executor.container_id = "abc123"

    @patch("subprocess.run")
    def test_basic_command_runs(self, mock_run):
        mock_run.return_value = _completed(returncode=0, stdout="result")
        result = self.executor.docker_exec("ls /workspace")
        self.assertEqual(result.returncode, 0)
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_command_is_quoted_in_bash_c(self, mock_run):
        mock_run.return_value = _completed(returncode=0)
        self.executor.docker_exec("ls /workspace")
        args = mock_run.call_args[0][0]
        # Should use bash -c '...'
        full = " ".join(args)
        self.assertIn("bash", full)
        self.assertIn("-c", full)

    @patch("subprocess.run")
    def test_env_vars_included(self, mock_run):
        mock_run.return_value = _completed(returncode=0)
        self.executor.docker_exec("printenv", env={"FOO": "bar", "BAZ": "qux"})
        args = mock_run.call_args[0][0]
        full = " ".join(args)
        self.assertIn("-e", full)
        self.assertIn("FOO=", full)
        self.assertIn("BAZ=", full)

    @patch("subprocess.run")
    def test_workdir_included(self, mock_run):
        mock_run.return_value = _completed(returncode=0)
        self.executor.docker_exec("ls", workdir="/app")
        args = mock_run.call_args[0][0]
        full = " ".join(args)
        self.assertIn("-w", full)
        self.assertIn("/app", full)

    def test_no_container_raises(self):
        self.executor.container_id = None
        with self.assertRaises(RemoteCommandError) as ctx:
            self.executor.docker_exec("ls")
        self.assertIn("container", str(ctx.exception).lower())

    @patch("subprocess.run")
    def test_container_id_in_command(self, mock_run):
        mock_run.return_value = _completed(returncode=0)
        self.executor.docker_exec("echo test")
        args = mock_run.call_args[0][0]
        cmd_str = " ".join(args)
        self.assertIn("abc123", cmd_str)


class TestCreateContainer(unittest.TestCase):
    def setUp(self):
        self.machine = _make_machine()
        self.executor = RemoteExecutor(self.machine)

    @patch("subprocess.run")
    def test_create_without_script_returns_container_id(self, mock_run):
        mock_run.return_value = _completed(returncode=0, stdout="deadbeef1234\n")
        cid = self.executor.create_container("rocm/pytorch:latest")
        self.assertEqual(cid, "deadbeef1234")
        self.assertEqual(self.executor.container_id, "deadbeef1234")

    @patch("subprocess.run")
    def test_create_without_script_uses_docker_run(self, mock_run):
        mock_run.return_value = _completed(returncode=0, stdout="cid123\n")
        self.executor.create_container("rocm/pytorch:latest")
        args = mock_run.call_args[0][0]
        full = " ".join(args)
        self.assertIn("docker", full)
        self.assertIn("run", full)
        self.assertIn("-d", full)

    @patch("subprocess.run")
    def test_create_without_script_includes_devices(self, mock_run):
        mock_run.return_value = _completed(returncode=0, stdout="cid123\n")
        self.executor.create_container("rocm/pytorch:latest")
        args = mock_run.call_args[0][0]
        full = " ".join(args)
        self.assertIn("/dev/kfd", full)
        self.assertIn("/dev/dri", full)

    @patch("subprocess.run")
    def test_create_without_script_includes_seccomp(self, mock_run):
        mock_run.return_value = _completed(returncode=0, stdout="cid123\n")
        self.executor.create_container("rocm/pytorch:latest")
        args = mock_run.call_args[0][0]
        full = " ".join(args)
        self.assertIn("seccomp=unconfined", full)

    @patch("subprocess.run")
    def test_create_with_name(self, mock_run):
        mock_run.return_value = _completed(returncode=0, stdout="cid123\n")
        self.executor.create_container("rocm/pytorch:latest", name="my-container")
        args = mock_run.call_args[0][0]
        full = " ".join(args)
        self.assertIn("--name", full)
        self.assertIn("my-container", full)

    @patch("subprocess.run")
    def test_create_with_script(self, mock_run):
        mock_run.return_value = _completed(returncode=0, stdout="some output\ncid_from_script\n")
        cid = self.executor.create_container("rocm/pytorch:latest", run_script="./create.sh")
        self.assertEqual(cid, "cid_from_script")
        self.assertEqual(self.executor.container_id, "cid_from_script")

    @patch("subprocess.run")
    def test_create_with_script_runs_script(self, mock_run):
        mock_run.return_value = _completed(returncode=0, stdout="cid456\n")
        self.executor.create_container("rocm/pytorch:latest", run_script="/path/to/create.sh")
        args = mock_run.call_args[0][0]
        full = " ".join(args)
        self.assertIn("create.sh", full)


class TestDestroyContainer(unittest.TestCase):
    def setUp(self):
        self.machine = _make_machine()
        self.executor = RemoteExecutor(self.machine)
        self.executor.container_id = "abc123"

    @patch("subprocess.run")
    def test_destroy_runs_stop_and_rm(self, mock_run):
        mock_run.return_value = _completed(returncode=0)
        self.executor.destroy_container()
        self.assertEqual(mock_run.call_count, 2)
        calls_flat = " ".join(
            " ".join(c[0][0]) for c in mock_run.call_args_list
        )
        self.assertIn("stop", calls_flat)
        self.assertIn("rm", calls_flat)

    @patch("subprocess.run")
    def test_destroy_clears_container_id(self, mock_run):
        mock_run.return_value = _completed(returncode=0)
        self.executor.destroy_container()
        self.assertIsNone(self.executor.container_id)

    @patch("subprocess.run")
    def test_destroy_passes_container_id(self, mock_run):
        mock_run.return_value = _completed(returncode=0)
        self.executor.destroy_container()
        calls_flat = " ".join(
            " ".join(c[0][0]) for c in mock_run.call_args_list
        )
        self.assertIn("abc123", calls_flat)


class TestFileCopy(unittest.TestCase):
    def setUp(self):
        self.machine = _make_machine()
        self.executor = RemoteExecutor(self.machine)
        self.executor.container_id = "abc123"

    @patch("subprocess.run")
    def test_copy_from_container_calls_docker_cp_then_scp(self, mock_run):
        mock_run.return_value = _completed(returncode=0)
        self.executor.copy_from_container("/container/path/file.txt", "/local/dest/")
        self.assertEqual(mock_run.call_count, 2)
        first_call = " ".join(mock_run.call_args_list[0][0][0])
        second_call = " ".join(mock_run.call_args_list[1][0][0])
        self.assertIn("docker", first_call)
        self.assertIn("cp", first_call)
        self.assertIn("scp", second_call)

    @patch("subprocess.run")
    def test_copy_to_container_calls_scp_then_docker_cp(self, mock_run):
        mock_run.return_value = _completed(returncode=0)
        self.executor.copy_to_container("/local/src/file.txt", "/container/dest/")
        self.assertEqual(mock_run.call_count, 2)
        first_call = " ".join(mock_run.call_args_list[0][0][0])
        second_call = " ".join(mock_run.call_args_list[1][0][0])
        self.assertIn("scp", first_call)
        self.assertIn("docker", second_call)
        self.assertIn("cp", second_call)

    @patch("subprocess.run")
    def test_copy_from_uses_container_id(self, mock_run):
        mock_run.return_value = _completed(returncode=0)
        self.executor.copy_from_container("/container/file.txt", "/local/")
        first_call = " ".join(mock_run.call_args_list[0][0][0])
        self.assertIn("abc123", first_call)

    @patch("subprocess.run")
    def test_copy_to_uses_container_id(self, mock_run):
        mock_run.return_value = _completed(returncode=0)
        self.executor.copy_to_container("/local/file.txt", "/container/")
        second_call = " ".join(mock_run.call_args_list[1][0][0])
        self.assertIn("abc123", second_call)


class TestKillStaleGpuProcesses(unittest.TestCase):
    def setUp(self):
        self.machine = _make_machine()
        self.executor = RemoteExecutor(self.machine)
        self.executor.container_id = "abc123"

    @patch("subprocess.run")
    def test_returns_list_of_pids(self, mock_run):
        rocm_output = (
            "======================= ROCm System Management Interface =======================\n"
            "PID\t\tProcess Name\t\tGPU(s)\t\tVRAM Used\n"
            "1234\t\tpython\t\t0\t\t2048 MB\n"
            "5678\t\tpython3\t\t1\t\t1024 MB\n"
            "============================= End of ROCm SMI Log ==============================\n"
        )
        # First call: rocm-smi, remaining calls: kill for each PID
        mock_run.side_effect = [
            _completed(returncode=0, stdout=rocm_output),
            _completed(returncode=0),
            _completed(returncode=0),
        ]
        pids = self.executor.kill_stale_gpu_processes()
        self.assertIn(1234, pids)
        self.assertIn(5678, pids)

    @patch("subprocess.run")
    def test_no_pids_returns_empty_list(self, mock_run):
        rocm_output = (
            "======================= ROCm System Management Interface =======================\n"
            "No running processes found\n"
            "============================= End of ROCm SMI Log ==============================\n"
        )
        mock_run.return_value = _completed(returncode=0, stdout=rocm_output)
        pids = self.executor.kill_stale_gpu_processes()
        self.assertEqual(pids, [])

    @patch("subprocess.run")
    def test_kills_with_signal_9(self, mock_run):
        rocm_output = (
            "PID\t\tProcess Name\n"
            "9999\t\tpython\n"
        )
        mock_run.side_effect = [
            _completed(returncode=0, stdout=rocm_output),
            _completed(returncode=0),
        ]
        self.executor.kill_stale_gpu_processes()
        kill_call = " ".join(mock_run.call_args_list[1][0][0])
        self.assertIn("kill", kill_call)
        self.assertIn("9999", kill_call)


class TestVerifyEnvironment(unittest.TestCase):
    def setUp(self):
        self.machine = _make_machine()
        self.executor = RemoteExecutor(self.machine)
        self.executor.container_id = "abc123"

    @patch("subprocess.run")
    def test_verify_returns_versions(self, mock_run):
        mock_run.side_effect = [
            _completed(returncode=0, stdout="3.0.0\n"),   # triton version
            _completed(returncode=0, stdout="main\n"),     # git branch
        ]
        result = self.executor.verify_environment(
            expected_triton_commit="3.0.0",
            expected_aiter_branch="main",
        )
        self.assertIn("triton_version", result)
        self.assertIn("aiter_branch", result)
        self.assertEqual(result["triton_version"], "3.0.0")
        self.assertEqual(result["aiter_branch"], "main")

    @patch("subprocess.run")
    def test_verify_triton_mismatch_raises(self, mock_run):
        mock_run.side_effect = [
            _completed(returncode=0, stdout="2.0.0\n"),  # wrong version
            _completed(returncode=0, stdout="main\n"),
        ]
        with self.assertRaises(RemoteCommandError):
            self.executor.verify_environment(expected_triton_commit="3.0.0")

    @patch("subprocess.run")
    def test_verify_aiter_mismatch_raises(self, mock_run):
        mock_run.side_effect = [
            _completed(returncode=0, stdout="3.0.0\n"),
            _completed(returncode=0, stdout="wrong-branch\n"),
        ]
        with self.assertRaises(RemoteCommandError):
            self.executor.verify_environment(expected_aiter_branch="main")

    @patch("subprocess.run")
    def test_verify_no_expected_values_does_not_raise(self, mock_run):
        mock_run.side_effect = [
            _completed(returncode=0, stdout="3.0.0\n"),
            _completed(returncode=0, stdout="feature-branch\n"),
        ]
        result = self.executor.verify_environment()
        self.assertEqual(result["triton_version"], "3.0.0")
        self.assertEqual(result["aiter_branch"], "feature-branch")


class TestIsContainerRunning(unittest.TestCase):
    def setUp(self):
        self.machine = _make_machine()
        self.executor = RemoteExecutor(self.machine)
        self.executor.container_id = "abc123"

    @patch("subprocess.run")
    def test_running_returns_true(self, mock_run):
        mock_run.return_value = _completed(returncode=0, stdout="true\n")
        self.assertTrue(self.executor.is_container_running())

    @patch("subprocess.run")
    def test_not_running_returns_false(self, mock_run):
        mock_run.return_value = _completed(returncode=0, stdout="false\n")
        self.assertFalse(self.executor.is_container_running())

    @patch("subprocess.run")
    def test_inspect_failure_returns_false(self, mock_run):
        mock_run.return_value = _completed(returncode=1, stderr="No such container")
        self.assertFalse(self.executor.is_container_running())

    def test_no_container_id_returns_false(self):
        self.executor.container_id = None
        self.assertFalse(self.executor.is_container_running())


class TestCheckSshConnectivity(unittest.TestCase):
    def setUp(self):
        self.machine = _make_machine()
        self.executor = RemoteExecutor(self.machine)

    @patch("subprocess.run")
    def test_success_returns_true(self, mock_run):
        mock_run.return_value = _completed(returncode=0, stdout="ok\n")
        self.assertTrue(self.executor.check_ssh_connectivity())

    @patch("subprocess.run")
    def test_failure_returns_false(self, mock_run):
        mock_run.return_value = _completed(returncode=255, stderr="refused")
        self.assertFalse(self.executor.check_ssh_connectivity())

    @patch("subprocess.run")
    def test_timeout_returns_false(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["ssh"], timeout=10)
        self.assertFalse(self.executor.check_ssh_connectivity())

    @patch("subprocess.run")
    def test_runs_echo_ok(self, mock_run):
        mock_run.return_value = _completed(returncode=0, stdout="ok\n")
        self.executor.check_ssh_connectivity()
        args = mock_run.call_args[0][0]
        self.assertIn("echo", " ".join(args))


class TestRemoteCommandError(unittest.TestCase):
    def test_attributes(self):
        err = RemoteCommandError("something failed", returncode=1, stdout="out", stderr="err")
        self.assertEqual(err.returncode, 1)
        self.assertEqual(err.stdout, "out")
        self.assertEqual(err.stderr, "err")
        self.assertIn("something failed", str(err))

    def test_default_attributes(self):
        err = RemoteCommandError("oops")
        self.assertIsNone(err.returncode)
        self.assertEqual(err.stdout, "")
        self.assertEqual(err.stderr, "")


if __name__ == "__main__":
    unittest.main()
