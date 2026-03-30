"""Tests for BaseSubagent, SubagentResult, and SubagentError."""

import json
import subprocess
import unittest
from unittest.mock import MagicMock, call, patch

from ..remote import RemoteCommandError, RemoteExecutor
from ..types import MachineInfo
from .base import BaseSubagent, SubagentError, SubagentResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _make_executor(machine=None):
    """Return a RemoteExecutor with container_id set."""
    m = machine or _make_machine()
    executor = RemoteExecutor(m)
    executor.container_id = "testcontainer"
    return executor


# ---------------------------------------------------------------------------
# Concrete subclass used across tests
# ---------------------------------------------------------------------------


class ConcreteSubagent(BaseSubagent):
    """Minimal concrete implementation for testing."""

    name = "concrete"

    def _execute(self) -> dict:
        return {"status": "done"}


class FailingSubagent(BaseSubagent):
    """Subagent whose _execute always raises."""

    name = "failing"

    def __init__(self, *args, exc_message="execute failed", **kwargs):
        super().__init__(*args, **kwargs)
        self._exc_message = exc_message

    def _execute(self) -> dict:
        raise SubagentError(self._exc_message)


class RuntimeFailingSubagent(BaseSubagent):
    """Subagent whose _execute raises a plain RuntimeError."""

    name = "runtime_failing"

    def _execute(self) -> dict:
        raise RuntimeError("unexpected runtime problem")


# ---------------------------------------------------------------------------
# SubagentResult tests
# ---------------------------------------------------------------------------


class TestSubagentResult(unittest.TestCase):
    def test_success_result(self):
        result = SubagentResult(success=True, data={"key": "value"})
        self.assertTrue(result.success)
        self.assertEqual(result.data, {"key": "value"})
        self.assertIsNone(result.error)

    def test_failure_result(self):
        result = SubagentResult(success=False, error="something went wrong")
        self.assertFalse(result.success)
        self.assertEqual(result.error, "something went wrong")
        self.assertEqual(result.data, {})

    def test_default_data_is_empty_dict(self):
        result = SubagentResult(success=True)
        self.assertIsInstance(result.data, dict)
        self.assertEqual(len(result.data), 0)

    def test_default_error_is_none(self):
        result = SubagentResult(success=True, data={"x": 1})
        self.assertIsNone(result.error)

    def test_failure_with_data(self):
        result = SubagentResult(success=False, data={"partial": True}, error="partial failure")
        self.assertFalse(result.success)
        self.assertEqual(result.data, {"partial": True})
        self.assertEqual(result.error, "partial failure")

    def test_two_instances_share_no_mutable_state(self):
        r1 = SubagentResult(success=True)
        r2 = SubagentResult(success=True)
        r1.data["added"] = 1
        self.assertNotIn("added", r2.data)


# ---------------------------------------------------------------------------
# SubagentError tests
# ---------------------------------------------------------------------------


class TestSubagentError(unittest.TestCase):
    def test_is_exception_subclass(self):
        self.assertTrue(issubclass(SubagentError, Exception))

    def test_message(self):
        err = SubagentError("something failed")
        self.assertIn("something failed", str(err))

    def test_can_be_raised_and_caught(self):
        with self.assertRaises(SubagentError):
            raise SubagentError("test error")

    def test_can_be_caught_as_exception(self):
        with self.assertRaises(Exception):
            raise SubagentError("test error")


# ---------------------------------------------------------------------------
# BaseSubagent construction tests
# ---------------------------------------------------------------------------


class TestBaseSubagentInit(unittest.TestCase):
    def setUp(self):
        self.executor = _make_executor()

    def test_stores_executor(self):
        agent = ConcreteSubagent(self.executor, "fmha", "/artifacts")
        self.assertIs(agent.executor, self.executor)

    def test_stores_kernel_name(self):
        agent = ConcreteSubagent(self.executor, "fmha", "/artifacts")
        self.assertEqual(agent.kernel_name, "fmha")

    def test_stores_artifact_dir(self):
        agent = ConcreteSubagent(self.executor, "fmha", "/tmp/arts")
        self.assertEqual(agent.artifact_dir, "/tmp/arts")

    def test_default_expected_values_none(self):
        agent = ConcreteSubagent(self.executor, "fmha", "/artifacts")
        self.assertIsNone(agent.expected_triton_commit)
        self.assertIsNone(agent.expected_aiter_branch)

    def test_stores_expected_triton_commit(self):
        agent = ConcreteSubagent(
            self.executor, "fmha", "/artifacts", expected_triton_commit="abc123"
        )
        self.assertEqual(agent.expected_triton_commit, "abc123")

    def test_stores_expected_aiter_branch(self):
        agent = ConcreteSubagent(
            self.executor, "fmha", "/artifacts", expected_aiter_branch="main"
        )
        self.assertEqual(agent.expected_aiter_branch, "main")

    def test_name_class_attribute(self):
        self.assertEqual(ConcreteSubagent.name, "concrete")

    def test_base_name_attribute(self):
        self.assertEqual(BaseSubagent.name, "base")


# ---------------------------------------------------------------------------
# BaseSubagent.run() — happy path
# ---------------------------------------------------------------------------


class TestRunSuccess(unittest.TestCase):
    def setUp(self):
        self.executor = _make_executor()
        self.agent = ConcreteSubagent(self.executor, "fmha", "/artifacts")

    @patch("subprocess.run")
    def test_run_returns_subagent_result(self, mock_run):
        # mkdir, rocm-smi (no PIDs)
        mock_run.return_value = _completed(returncode=0, stdout="")
        result = self.agent.run()
        self.assertIsInstance(result, SubagentResult)

    @patch("subprocess.run")
    def test_run_success_true(self, mock_run):
        mock_run.return_value = _completed(returncode=0, stdout="")
        result = self.agent.run()
        self.assertTrue(result.success)

    @patch("subprocess.run")
    def test_run_data_contains_execute_output(self, mock_run):
        mock_run.return_value = _completed(returncode=0, stdout="")
        result = self.agent.run()
        self.assertEqual(result.data.get("status"), "done")

    @patch("subprocess.run")
    def test_run_error_is_none_on_success(self, mock_run):
        mock_run.return_value = _completed(returncode=0, stdout="")
        result = self.agent.run()
        self.assertIsNone(result.error)


# ---------------------------------------------------------------------------
# BaseSubagent.run() — exception handling
# ---------------------------------------------------------------------------


class TestRunFailure(unittest.TestCase):
    def setUp(self):
        self.executor = _make_executor()

    @patch("subprocess.run")
    def test_run_catches_subagent_error(self, mock_run):
        mock_run.return_value = _completed(returncode=0, stdout="")
        agent = FailingSubagent(self.executor, "fmha", "/artifacts")
        result = agent.run()
        self.assertFalse(result.success)

    @patch("subprocess.run")
    def test_run_captures_error_message(self, mock_run):
        mock_run.return_value = _completed(returncode=0, stdout="")
        agent = FailingSubagent(
            self.executor, "fmha", "/artifacts", exc_message="execute failed"
        )
        result = agent.run()
        self.assertIn("execute failed", result.error)

    @patch("subprocess.run")
    def test_run_catches_runtime_error(self, mock_run):
        mock_run.return_value = _completed(returncode=0, stdout="")
        agent = RuntimeFailingSubagent(self.executor, "fmha", "/artifacts")
        result = agent.run()
        self.assertFalse(result.success)
        self.assertIn("unexpected runtime problem", result.error)

    @patch("subprocess.run")
    def test_run_failure_data_is_empty(self, mock_run):
        mock_run.return_value = _completed(returncode=0, stdout="")
        agent = FailingSubagent(self.executor, "fmha", "/artifacts")
        result = agent.run()
        self.assertEqual(result.data, {})

    def test_run_catches_preflight_error(self):
        """If preflight itself fails, run() still returns a failure result."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = _completed(returncode=1, stderr="permission denied")
            # mkdir will raise because check=True and returncode=1
            agent = ConcreteSubagent(self.executor, "fmha", "/artifacts")
            result = agent.run()
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)


# ---------------------------------------------------------------------------
# _preflight tests
# ---------------------------------------------------------------------------


class TestPreflight(unittest.TestCase):
    def setUp(self):
        self.executor = _make_executor()

    @patch("subprocess.run")
    def test_preflight_calls_mkdir(self, mock_run):
        mock_run.return_value = _completed(returncode=0, stdout="")
        agent = ConcreteSubagent(self.executor, "fmha", "/my/artifacts")
        agent._preflight()
        all_calls = " ".join(
            " ".join(c[0][0]) for c in mock_run.call_args_list
        )
        self.assertIn("mkdir", all_calls)
        self.assertIn("/my/artifacts", all_calls)

    @patch("subprocess.run")
    def test_preflight_calls_rocm_smi(self, mock_run):
        mock_run.return_value = _completed(returncode=0, stdout="")
        agent = ConcreteSubagent(self.executor, "fmha", "/artifacts")
        agent._preflight()
        all_calls = " ".join(
            " ".join(c[0][0]) for c in mock_run.call_args_list
        )
        self.assertIn("rocm-smi", all_calls)

    @patch("subprocess.run")
    def test_preflight_mkdir_uses_dash_p(self, mock_run):
        mock_run.return_value = _completed(returncode=0, stdout="")
        agent = ConcreteSubagent(self.executor, "fmha", "/artifacts")
        agent._preflight()
        # Find the mkdir call
        mkdir_call = next(
            " ".join(c[0][0])
            for c in mock_run.call_args_list
            if "mkdir" in " ".join(c[0][0])
        )
        self.assertIn("-p", mkdir_call)

    @patch("subprocess.run")
    def test_preflight_no_verify_when_no_expected_values(self, mock_run):
        """verify_environment should NOT be called when no expected values set."""
        mock_run.return_value = _completed(returncode=0, stdout="")
        agent = ConcreteSubagent(self.executor, "fmha", "/artifacts")
        agent._preflight()
        # verify_environment calls docker exec (which goes through ssh_run);
        # the only SSH commands should be mkdir and rocm-smi
        all_calls = [" ".join(c[0][0]) for c in mock_run.call_args_list]
        # No call should contain 'triton' or 'git rev-parse'
        for cmd_str in all_calls:
            self.assertNotIn("triton", cmd_str)
            self.assertNotIn("rev-parse", cmd_str)

    @patch("subprocess.run")
    def test_preflight_calls_verify_when_triton_commit_set(self, mock_run):
        """verify_environment IS called when expected_triton_commit is set."""
        mock_run.side_effect = [
            _completed(returncode=0, stdout=""),        # mkdir
            _completed(returncode=0, stdout=""),        # rocm-smi
            _completed(returncode=0, stdout="3.0.0\n"), # triton version
            _completed(returncode=0, stdout="main\n"),  # git branch
        ]
        agent = ConcreteSubagent(
            self.executor, "fmha", "/artifacts", expected_triton_commit="3.0.0"
        )
        agent._preflight()
        all_calls = " ".join(
            " ".join(c[0][0]) for c in mock_run.call_args_list
        )
        # verify_environment issues docker exec commands through ssh
        self.assertGreater(mock_run.call_count, 2)

    @patch("subprocess.run")
    def test_preflight_calls_verify_when_aiter_branch_set(self, mock_run):
        """verify_environment IS called when expected_aiter_branch is set."""
        mock_run.side_effect = [
            _completed(returncode=0, stdout=""),        # mkdir
            _completed(returncode=0, stdout=""),        # rocm-smi
            _completed(returncode=0, stdout="3.0.0\n"), # triton version (docker exec)
            _completed(returncode=0, stdout="main\n"),  # git branch (docker exec)
        ]
        agent = ConcreteSubagent(
            self.executor, "fmha", "/artifacts", expected_aiter_branch="main"
        )
        agent._preflight()
        self.assertGreater(mock_run.call_count, 2)

    @patch("subprocess.run")
    def test_preflight_mkdir_before_rocm_smi(self, mock_run):
        """mkdir must happen before rocm-smi (order matters)."""
        mock_run.return_value = _completed(returncode=0, stdout="")
        agent = ConcreteSubagent(self.executor, "fmha", "/artifacts")
        agent._preflight()
        cmd_strings = [" ".join(c[0][0]) for c in mock_run.call_args_list]
        mkdir_idx = next(i for i, s in enumerate(cmd_strings) if "mkdir" in s)
        rocm_idx = next(i for i, s in enumerate(cmd_strings) if "rocm-smi" in s)
        self.assertLess(mkdir_idx, rocm_idx)


# ---------------------------------------------------------------------------
# _write_json_artifact / _read_json_artifact tests
# ---------------------------------------------------------------------------


class TestJsonArtifacts(unittest.TestCase):
    def setUp(self):
        self.executor = _make_executor()
        self.agent = ConcreteSubagent(self.executor, "fmha", "/artifacts")

    @patch("subprocess.run")
    def test_write_json_artifact_returns_remote_path(self, mock_run):
        mock_run.return_value = _completed(returncode=0)
        path = self.agent._write_json_artifact("results.json", {"x": 1})
        self.assertIn("/artifacts", path)
        self.assertIn("results.json", path)

    @patch("subprocess.run")
    def test_write_json_artifact_calls_ssh_run(self, mock_run):
        mock_run.return_value = _completed(returncode=0)
        self.agent._write_json_artifact("out.json", {"k": "v"})
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_read_json_artifact_returns_parsed_dict(self, mock_run):
        payload = {"status": "ok", "value": 42}
        mock_run.return_value = _completed(returncode=0, stdout=json.dumps(payload))
        result = self.agent._read_json_artifact("results.json")
        self.assertEqual(result, payload)

    @patch("subprocess.run")
    def test_read_json_artifact_uses_cat(self, mock_run):
        mock_run.return_value = _completed(returncode=0, stdout="{}")
        self.agent._read_json_artifact("results.json")
        cmd_str = " ".join(mock_run.call_args[0][0])
        self.assertIn("cat", cmd_str)
        self.assertIn("results.json", cmd_str)

    @patch("subprocess.run")
    def test_read_json_artifact_invalid_json_raises_subagent_error(self, mock_run):
        mock_run.return_value = _completed(returncode=0, stdout="not json {{{")
        with self.assertRaises(SubagentError):
            self.agent._read_json_artifact("bad.json")

    @patch("subprocess.run")
    def test_write_artifact_path_joins_dir_and_filename(self, mock_run):
        mock_run.return_value = _completed(returncode=0)
        agent = ConcreteSubagent(self.executor, "fmha", "/custom/dir")
        path = agent._write_json_artifact("data.json", {})
        self.assertEqual(path, "/custom/dir/data.json")


# ---------------------------------------------------------------------------
# AbstractMethod enforcement
# ---------------------------------------------------------------------------


class TestAbstractMethod(unittest.TestCase):
    def test_cannot_instantiate_base_directly(self):
        executor = _make_executor()
        with self.assertRaises(TypeError):
            BaseSubagent(executor, "fmha", "/artifacts")  # type: ignore[abstract]

    def test_subclass_without_execute_cannot_instantiate(self):
        class Incomplete(BaseSubagent):
            pass

        executor = _make_executor()
        with self.assertRaises(TypeError):
            Incomplete(executor, "fmha", "/artifacts")  # type: ignore[abstract]


if __name__ == "__main__":
    unittest.main()
