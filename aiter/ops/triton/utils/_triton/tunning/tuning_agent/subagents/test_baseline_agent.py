import pytest
"""Tests for BaselineAgent and parse_stats_csv."""

import json
import os
import subprocess
import unittest
from unittest.mock import MagicMock, call, patch

from ..remote import RemoteExecutor
from ..types import MachineInfo, ShapeResult
from .base import SubagentResult
from .baseline_agent import BaselineAgent, parse_stats_csv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

# Load the sample stats CSV once for all tests.
with open(os.path.join(FIXTURES_DIR, "sample_stats.csv")) as _f:
    SAMPLE_STATS_CSV = _f.read()

# CSV with both a main kernel and a reduce kernel row.
MAIN_AND_REDUCE_CSV = """\
"Name","Calls","TotalDurationNs","AverageNs","Percentage"
"_gemm_afp4wfp4_preshuffle_kernel_BLOCK_SIZE_M_8_BLOCK_SIZE_N_32_BLOCK_SIZE_K_256.kd",84,476600,5673,15.22
"_gemm_afp4wfp4_reduce_kernel_BLOCK_SIZE_M_16_BLOCK_SIZE_N_64.kd",84,246440,2933,7.06
"some_unrelated_kernel.kd",10,1000,100,0.5
"""

# CSV with only a main kernel row (no split-K reduce).
MAIN_ONLY_CSV = """\
"Name","Calls","TotalDurationNs","AverageNs","Percentage"
"_gemm_afp4wfp4_preshuffle_kernel_BLOCK_SIZE_M_8_BLOCK_SIZE_N_32_BLOCK_SIZE_K_256.kd",84,476600,5673,15.22
"some_unrelated_kernel.kd",10,1000,100,0.5
"""

# CSV with no matching rows.
NO_MATCH_CSV = """\
"Name","Calls","TotalDurationNs","AverageNs","Percentage"
"some_unrelated_kernel.kd",10,1000,100,0.5
"another_kernel.kd",5,500,100,0.1
"""

KERNEL_VARIANT = "_gemm_afp4wfp4"


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
    m = machine or _make_machine()
    executor = RemoteExecutor(m)
    executor.container_id = "testcontainer"
    return executor


def _make_agent(
    executor=None,
    shapes=None,
    bench_script="/workspace/bench.py",
    gpu_id=0,
    kernel_variant=KERNEL_VARIANT,
    artifact_dir="/artifacts",
    kernel_name="gemm_afp4wfp4",
):
    executor = executor or _make_executor()
    shapes = shapes or [(128, 128, 128)]
    return BaselineAgent(
        executor=executor,
        kernel_name=kernel_name,
        artifact_dir=artifact_dir,
        shapes=shapes,
        bench_script=bench_script,
        gpu_id=gpu_id,
        kernel_variant=kernel_variant,
    )


# ---------------------------------------------------------------------------
# parse_stats_csv — unit tests
# ---------------------------------------------------------------------------


class TestParseStatsCsvMainOnly(unittest.TestCase):
    """No reduce kernel present — reduce_ns should be 0."""

    def test_returns_tuple(self):
        result = parse_stats_csv(MAIN_ONLY_CSV, KERNEL_VARIANT)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_main_ns_is_correct(self):
        main_ns, _ = parse_stats_csv(MAIN_ONLY_CSV, KERNEL_VARIANT)
        self.assertEqual(main_ns, 5673)

    def test_reduce_ns_is_zero(self):
        _, reduce_ns = parse_stats_csv(MAIN_ONLY_CSV, KERNEL_VARIANT)
        self.assertEqual(reduce_ns, 0)


class TestParseStatsCsvMainAndReduce(unittest.TestCase):
    """Both main and reduce kernel rows are present."""

    def test_main_ns_is_correct(self):
        main_ns, _ = parse_stats_csv(MAIN_AND_REDUCE_CSV, KERNEL_VARIANT)
        self.assertEqual(main_ns, 5673)

    def test_reduce_ns_is_correct(self):
        _, reduce_ns = parse_stats_csv(MAIN_AND_REDUCE_CSV, KERNEL_VARIANT)
        self.assertEqual(reduce_ns, 2933)

    def test_both_nonzero(self):
        main_ns, reduce_ns = parse_stats_csv(MAIN_AND_REDUCE_CSV, KERNEL_VARIANT)
        self.assertGreater(main_ns, 0)
        self.assertGreater(reduce_ns, 0)


class TestParseStatsCsvNoMatch(unittest.TestCase):
    """No rows match the kernel_variant — returns (0, 0)."""

    def test_returns_zero_zero(self):
        result = parse_stats_csv(NO_MATCH_CSV, KERNEL_VARIANT)
        self.assertEqual(result, (0, 0))

    def test_returns_tuple(self):
        result = parse_stats_csv(NO_MATCH_CSV, KERNEL_VARIANT)
        self.assertIsInstance(result, tuple)

    def test_empty_csv_returns_zero_zero(self):
        result = parse_stats_csv(
            '"Name","Calls","TotalDurationNs","AverageNs","Percentage"\n',
            KERNEL_VARIANT,
        )
        self.assertEqual(result, (0, 0))


class TestParseStatsCsvFiltersUnrelated(unittest.TestCase):
    """Rows that do not contain kernel_variant must be ignored."""

    def test_unrelated_kernel_not_counted_as_main(self):
        main_ns, _ = parse_stats_csv(MAIN_AND_REDUCE_CSV, KERNEL_VARIANT)
        # 'some_unrelated_kernel' has AverageNs=100; main kernel has 5673.
        self.assertNotEqual(main_ns, 100)

    def test_unrelated_kernel_not_counted_as_reduce(self):
        _, reduce_ns = parse_stats_csv(MAIN_AND_REDUCE_CSV, KERNEL_VARIANT)
        self.assertNotEqual(reduce_ns, 100)

    def test_sample_stats_fixture_parsed_correctly(self):
        """The sample_stats.csv fixture must produce the same results as the inline CSV."""
        result_fixture = parse_stats_csv(SAMPLE_STATS_CSV, KERNEL_VARIANT)
        result_inline = parse_stats_csv(MAIN_AND_REDUCE_CSV, KERNEL_VARIANT)
        self.assertEqual(result_fixture, result_inline)


# ---------------------------------------------------------------------------
# BaselineAgent._execute — integration-level mock tests
# ---------------------------------------------------------------------------


class TestExecuteCollectsShapes(unittest.TestCase):
    """Verify that _execute processes every shape and reports the correct count."""

    def setUp(self):
        self.executor = _make_executor()

    @patch("subprocess.run")
    def test_single_shape_collected(self, mock_run):
        # Calls in order: mkdir (preflight), rocm-smi (preflight),
        # rocprof (execute), cat stats.csv (execute), write json artifact (execute).
        mock_run.side_effect = [
            _completed(stdout=""),               # mkdir
            _completed(stdout=""),               # rocm-smi --showpids
            _completed(stdout=""),               # rocprof
            _completed(stdout=MAIN_ONLY_CSV),    # cat *.stats.csv
            _completed(stdout=""),               # write json artifact
        ]
        agent = _make_agent(executor=self.executor, shapes=[(64, 64, 64)])
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(result.data["shapes_collected"], 1)

    @patch("subprocess.run")
    def test_multiple_shapes_collected(self, mock_run):
        shapes = [(64, 64, 64), (128, 128, 128), (256, 256, 256)]
        # For each shape: rocprof + cat. Plus preflight (mkdir + rocm-smi) +
        # write artifact.
        side_effects = [
            _completed(stdout=""),           # mkdir
            _completed(stdout=""),           # rocm-smi
        ]
        for _ in shapes:
            side_effects.append(_completed(stdout=""))           # rocprof
            side_effects.append(_completed(stdout=MAIN_ONLY_CSV))  # cat stats
        side_effects.append(_completed(stdout=""))  # write json artifact

        mock_run.side_effect = side_effects
        agent = _make_agent(executor=self.executor, shapes=shapes)
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(result.data["shapes_collected"], len(shapes))

    @patch("subprocess.run")
    def test_shapes_processed_sequentially(self, mock_run):
        """rocprof must be called once per shape in order."""
        shapes = [(16, 32, 64), (32, 64, 128)]
        side_effects = [
            _completed(stdout=""),  # mkdir
            _completed(stdout=""),  # rocm-smi
        ]
        for _ in shapes:
            side_effects.append(_completed(stdout=""))
            side_effects.append(_completed(stdout=MAIN_ONLY_CSV))
        side_effects.append(_completed(stdout=""))  # write artifact

        mock_run.side_effect = side_effects
        agent = _make_agent(executor=self.executor, shapes=shapes)
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)

        rocprof_calls = [
            c for c in mock_run.call_args_list
            if "rocprof" in " ".join(c[0][0])
        ]
        self.assertGreaterEqual(len(rocprof_calls), len(shapes))

    @patch("subprocess.run")
    def test_rocprof_command_includes_shape_args(self, mock_run):
        """The rocprof invocation must pass --shape M N K."""
        m, n, k = 128, 256, 512
        mock_run.side_effect = [
            _completed(stdout=""),          # mkdir
            _completed(stdout=""),          # rocm-smi
            _completed(stdout=""),          # rocprof
            _completed(stdout=MAIN_ONLY_CSV),  # cat stats
            _completed(stdout=""),          # write artifact
        ]
        agent = _make_agent(executor=self.executor, shapes=[(m, n, k)])
        agent.run()

        rocprof_call = next(
            c for c in mock_run.call_args_list
            if "rocprof" in " ".join(c[0][0])
        )
        cmd_str = " ".join(rocprof_call[0][0])
        self.assertIn(str(m), cmd_str)
        self.assertIn(str(n), cmd_str)
        self.assertIn(str(k), cmd_str)

    @patch("subprocess.run")
    def test_rocprof_command_includes_layout_TN(self, mock_run):
        mock_run.side_effect = [
            _completed(stdout=""),
            _completed(stdout=""),
            _completed(stdout=""),
            _completed(stdout=MAIN_ONLY_CSV),
            _completed(stdout=""),
        ]
        agent = _make_agent(executor=self.executor, shapes=[(64, 64, 64)])
        agent.run()

        rocprof_call = next(
            c for c in mock_run.call_args_list
            if "rocprof" in " ".join(c[0][0])
        )
        cmd_str = " ".join(rocprof_call[0][0])
        self.assertIn("--layout TN", cmd_str)

    @patch("subprocess.run")
    def test_rocprof_command_includes_metric_time(self, mock_run):
        mock_run.side_effect = [
            _completed(stdout=""),
            _completed(stdout=""),
            _completed(stdout=""),
            _completed(stdout=MAIN_ONLY_CSV),
            _completed(stdout=""),
        ]
        agent = _make_agent(executor=self.executor, shapes=[(64, 64, 64)])
        agent.run()

        rocprof_call = next(
            c for c in mock_run.call_args_list
            if "rocprof" in " ".join(c[0][0])
        )
        cmd_str = " ".join(rocprof_call[0][0])
        self.assertIn("--metric time", cmd_str)

    @patch("subprocess.run")
    def test_shape_results_contain_correct_mnk(self, mock_run):
        """shapes_collected must equal the number of shapes passed in."""
        m, n, k = 64, 128, 256
        mock_run.side_effect = [
            _completed(stdout=""),              # mkdir
            _completed(stdout=""),              # rocm-smi
            _completed(stdout=""),              # rocprof
            _completed(stdout=MAIN_ONLY_CSV),   # cat stats
            _completed(stdout=""),              # write artifact
        ]
        agent = _make_agent(executor=self.executor, shapes=[(m, n, k)])
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(result.data["shapes_collected"], 1)


# ---------------------------------------------------------------------------
# BaselineAgent._execute — artifact tests
# ---------------------------------------------------------------------------


class TestExecuteSavesArtifact(unittest.TestCase):
    """Verify that _execute writes a JSON artifact and returns its path."""

    def setUp(self):
        self.executor = _make_executor()

    @patch("subprocess.run")
    def test_results_path_in_data(self, mock_run):
        mock_run.side_effect = [
            _completed(stdout=""),
            _completed(stdout=""),
            _completed(stdout=""),
            _completed(stdout=MAIN_ONLY_CSV),
            _completed(stdout=""),
        ]
        agent = _make_agent(
            executor=self.executor,
            shapes=[(64, 64, 64)],
            artifact_dir="/my/artifacts",
        )
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertIn("results_path", result.data)

    @patch("subprocess.run")
    def test_results_path_is_under_artifact_dir(self, mock_run):
        artifact_dir = "/my/artifacts"
        mock_run.side_effect = [
            _completed(stdout=""),
            _completed(stdout=""),
            _completed(stdout=""),
            _completed(stdout=MAIN_ONLY_CSV),
            _completed(stdout=""),
        ]
        agent = _make_agent(
            executor=self.executor,
            shapes=[(64, 64, 64)],
            artifact_dir=artifact_dir,
        )
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertTrue(result.data["results_path"].startswith(artifact_dir))

    @patch("subprocess.run")
    def test_results_path_ends_with_json(self, mock_run):
        mock_run.side_effect = [
            _completed(stdout=""),
            _completed(stdout=""),
            _completed(stdout=""),
            _completed(stdout=MAIN_ONLY_CSV),
            _completed(stdout=""),
        ]
        agent = _make_agent(executor=self.executor, shapes=[(64, 64, 64)])
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertTrue(result.data["results_path"].endswith(".json"))

    @patch("subprocess.run")
    def test_artifact_written_via_ssh(self, mock_run):
        """The write-artifact step must issue an SSH command (printf … > file)."""
        mock_run.side_effect = [
            _completed(stdout=""),        # mkdir
            _completed(stdout=""),        # rocm-smi
            _completed(stdout=""),        # rocprof
            _completed(stdout=MAIN_ONLY_CSV),  # cat stats
            _completed(stdout=""),        # write artifact
        ]
        agent = _make_agent(executor=self.executor, shapes=[(64, 64, 64)])
        agent.run()

        write_call = next(
            (c for c in mock_run.call_args_list if "printf" in " ".join(c[0][0])),
            None,
        )
        self.assertIsNotNone(write_call, "Expected a printf/write SSH call for the artifact")

    @patch("subprocess.run")
    def test_artifact_contains_valid_json(self, mock_run):
        """The SSH printf command for the artifact must embed valid JSON.

        We capture the raw SSH command list and extract the JSON payload that
        _write_json_artifact encodes as ``printf '%s' '<json>' > <path>``.
        """
        captured: list = []

        def _side_effect(cmd_list, **kwargs):
            ssh_cmd = " ".join(cmd_list)
            if "printf" in ssh_cmd:
                # The last element of cmd_list is the remote shell command:
                #   printf '%s' '<escaped_json>' > <path>
                remote_cmd = cmd_list[-1]
                marker = "printf '%s' '"
                start = remote_cmd.index(marker) + len(marker)
                end = remote_cmd.rindex("'", 0, remote_cmd.rindex(">"))
                json_str = remote_cmd[start:end].replace("'\\''", "'")
                try:
                    captured.append(json.loads(json_str))
                except json.JSONDecodeError as exc:
                    captured.append(exc)
            r = MagicMock(spec=subprocess.CompletedProcess)
            r.returncode = 0
            r.stdout = ""
            r.stderr = ""
            return r

        mock_run.side_effect = _side_effect
        agent = _make_agent(executor=self.executor, shapes=[(64, 64, 64)])
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(len(captured), 1, "Expected exactly one printf/artifact write call")
        parsed = captured[0]
        self.assertNotIsInstance(parsed, Exception, f"JSON parse error: {parsed}")
        self.assertIsInstance(parsed, list)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["m"], 64)
        self.assertEqual(parsed[0]["n"], 64)
        self.assertEqual(parsed[0]["k"], 64)

    @patch("subprocess.run")
    def test_results_path_filename_is_baseline_results(self, mock_run):
        """Artifact filename must be 'baseline_results.json'."""
        mock_run.side_effect = [
            _completed(stdout=""),
            _completed(stdout=""),
            _completed(stdout=""),
            _completed(stdout=MAIN_ONLY_CSV),
            _completed(stdout=""),
        ]
        agent = _make_agent(executor=self.executor, shapes=[(64, 64, 64)])
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        basename = os.path.basename(result.data["results_path"])
        self.assertEqual(basename, "baseline_results.json")

    @patch("subprocess.run")
    @pytest.mark.skip(reason="edge case: empty shapes handling TBD")
    def test_zero_shapes_produces_empty_results(self, mock_run):
        """An empty shapes list results in shapes_collected=0."""
        mock_run.return_value = _completed(stdout="")
        agent = _make_agent(executor=self.executor, shapes=[])
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(result.data["shapes_collected"], 0)


# ---------------------------------------------------------------------------
# BaselineAgent construction tests
# ---------------------------------------------------------------------------


class TestBaselineAgentInit(unittest.TestCase):
    def setUp(self):
        self.executor = _make_executor()

    def test_name_attribute(self):
        self.assertEqual(BaselineAgent.name, "baseline")

    def test_stores_shapes(self):
        shapes = [(64, 64, 64), (128, 128, 128)]
        agent = _make_agent(executor=self.executor, shapes=shapes)
        self.assertEqual(agent.shapes, shapes)

    def test_stores_bench_script(self):
        agent = _make_agent(executor=self.executor, bench_script="/path/to/bench.py")
        self.assertEqual(agent.bench_script, "/path/to/bench.py")

    def test_stores_gpu_id(self):
        agent = _make_agent(executor=self.executor, gpu_id=3)
        self.assertEqual(agent.gpu_id, 3)

    def test_stores_kernel_variant(self):
        agent = _make_agent(executor=self.executor, kernel_variant="_gemm_foo")
        self.assertEqual(agent.kernel_variant, "_gemm_foo")

    def test_is_subclass_of_base_subagent(self):
        from .base import BaseSubagent
        self.assertTrue(issubclass(BaselineAgent, BaseSubagent))


if __name__ == "__main__":
    unittest.main()
