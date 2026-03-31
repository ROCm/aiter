"""Tests for ValidationAgent._execute()."""

from __future__ import annotations

import os
import subprocess
import unittest
from unittest.mock import MagicMock, call, patch

from ..remote import RemoteExecutor
from ..types import MachineInfo, ShapeResult
from .baseline_agent import parse_stats_csv
from .validation_agent import ValidationAgent

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

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

KERNEL_VARIANT = "_gemm_afp4wfp4"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_machine(**kwargs) -> MachineInfo:
    defaults = {
        "host": "gpu-host.example.com",
        "user": "testuser",
        "ssh_key": "/home/testuser/.ssh/id_rsa",
        "gpus": [0, 1],
    }
    defaults.update(kwargs)
    return MachineInfo(**defaults)


def _completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    r = MagicMock(spec=subprocess.CompletedProcess)
    r.returncode = returncode
    r.stdout = stdout
    r.stderr = stderr
    return r


def _make_executor(machine=None) -> RemoteExecutor:
    m = machine or _make_machine()
    executor = RemoteExecutor(m)
    executor.container_id = "testcontainer"
    return executor


def _make_agent(
    executor=None,
    shapes=None,
    bench_script="/workspace/bench.py",
    gpu_ids=None,
    kernel_variant=KERNEL_VARIANT,
    baseline_data=None,
    untuned_data=None,
    threshold=0.05,
    tuning_threshold=None,
    shuffle=False,
    artifact_dir="/artifacts",
    kernel_name="gemm_afp4wfp4",
) -> ValidationAgent:
    executor = executor or _make_executor()
    shapes = shapes if shapes is not None else [(128, 128, 128)]
    gpu_ids = gpu_ids if gpu_ids is not None else [0]
    return ValidationAgent(
        executor=executor,
        kernel_name=kernel_name,
        artifact_dir=artifact_dir,
        shapes=shapes,
        bench_script=bench_script,
        gpu_ids=gpu_ids,
        kernel_variant=kernel_variant,
        baseline_data=baseline_data,
        untuned_data=untuned_data,
        threshold=threshold,
        tuning_threshold=tuning_threshold,
        shuffle=shuffle,
    )


# ---------------------------------------------------------------------------
# Helpers: common mock side_effect builders
# ---------------------------------------------------------------------------


def _side_effects_for_n_shapes(n: int, stats_csv: str = MAIN_ONLY_CSV) -> list:
    """Return mock subprocess.run return values for preflight + n shapes + cleanup."""
    # Preflight: mkdir + rocm-smi
    effects = [
        _completed(stdout=""),  # mkdir
        _completed(stdout=""),  # rocm-smi --showpids
    ]
    for _ in range(n):
        # docker exec rocprof
        effects.append(_completed(stdout=""))
        # docker exec cat *.stats.csv
        effects.append(_completed(stdout=stats_csv))
    # cleanup rm -f
    effects.append(_completed(stdout=""))
    return effects


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestValidationAgentInit(unittest.TestCase):
    def test_name_attribute(self):
        self.assertEqual(ValidationAgent.name, "validation")

    def test_stores_shapes(self):
        shapes = [(64, 64, 64), (128, 128, 128)]
        agent = _make_agent(shapes=shapes)
        self.assertEqual(agent.shapes, shapes)

    def test_stores_bench_script(self):
        agent = _make_agent(bench_script="/path/to/bench.py")
        self.assertEqual(agent.bench_script, "/path/to/bench.py")

    def test_stores_gpu_ids(self):
        agent = _make_agent(gpu_ids=[0, 1, 2])
        self.assertEqual(agent.gpu_ids, [0, 1, 2])

    def test_stores_kernel_variant(self):
        agent = _make_agent(kernel_variant="_gemm_foo")
        self.assertEqual(agent.kernel_variant, "_gemm_foo")

    def test_stores_threshold(self):
        agent = _make_agent(threshold=0.1)
        self.assertAlmostEqual(agent.threshold, 0.1)

    def test_stores_shuffle(self):
        agent = _make_agent(shuffle=True)
        self.assertTrue(agent.shuffle)

    def test_tuning_threshold_defaults_to_threshold(self):
        agent = _make_agent(threshold=0.07)
        self.assertAlmostEqual(agent.tuning_threshold, 0.07)

    def test_tuning_threshold_explicit(self):
        agent = _make_agent(threshold=0.05, tuning_threshold=0.02)
        self.assertAlmostEqual(agent.tuning_threshold, 0.02)

    def test_baseline_data_stored(self):
        bd = {"M128_N128_K128": {"total_ns": 1000}}
        agent = _make_agent(baseline_data=bd)
        self.assertEqual(agent.baseline_data, bd)

    def test_untuned_data_stored(self):
        ud = {"M128_N128_K128": {"total_ns": 1100}}
        agent = _make_agent(untuned_data=ud)
        self.assertEqual(agent.untuned_data, ud)

    def test_is_subclass_of_base_subagent(self):
        from .base import BaseSubagent
        self.assertTrue(issubclass(ValidationAgent, BaseSubagent))


# ---------------------------------------------------------------------------
# rocprof command construction
# ---------------------------------------------------------------------------


class TestRocprofCommand(unittest.TestCase):
    """Verify that rocprof is called with the correct arguments for each shape."""

    def setUp(self):
        self.executor = _make_executor()

    @patch("subprocess.run")
    def test_rocprof_called_for_single_shape(self, mock_run):
        mock_run.side_effect = _side_effects_for_n_shapes(1)
        agent = _make_agent(executor=self.executor, shapes=[(64, 64, 64)])
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)

        rocprof_calls = [
            c for c in mock_run.call_args_list
            if "rocprof" in " ".join(c[0][0])
        ]
        self.assertEqual(len(rocprof_calls), 1)

    @patch("subprocess.run")
    def test_rocprof_called_for_each_shape(self, mock_run):
        shapes = [(64, 64, 64), (128, 128, 128), (256, 256, 256)]
        mock_run.side_effect = _side_effects_for_n_shapes(len(shapes))
        agent = _make_agent(executor=self.executor, shapes=shapes)
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)

        rocprof_calls = [
            c for c in mock_run.call_args_list
            if "rocprof" in " ".join(c[0][0])
        ]
        self.assertEqual(len(rocprof_calls), len(shapes))

    @patch("subprocess.run")
    def test_rocprof_o_prefix_contains_shape_dimensions(self, mock_run):
        """The -o prefix must embed M, N, K to avoid file collisions."""
        m, n, k = 128, 256, 512
        mock_run.side_effect = _side_effects_for_n_shapes(1)
        agent = _make_agent(executor=self.executor, shapes=[(m, n, k)])
        agent.run()

        rocprof_call = next(
            c for c in mock_run.call_args_list
            if "rocprof" in " ".join(c[0][0])
        )
        cmd_str = " ".join(rocprof_call[0][0])
        self.assertIn(f"/tmp/rp_{m}_{n}_{k}", cmd_str)

    @patch("subprocess.run")
    def test_rocprof_o_prefix_unique_per_shape(self, mock_run):
        """Different shapes must use different -o prefixes."""
        shapes = [(64, 64, 64), (128, 128, 128)]
        mock_run.side_effect = _side_effects_for_n_shapes(len(shapes))
        agent = _make_agent(executor=self.executor, shapes=shapes)
        agent.run()

        rocprof_calls = [
            " ".join(c[0][0])
            for c in mock_run.call_args_list
            if "rocprof" in " ".join(c[0][0])
        ]
        prefixes = set()
        for cmd in rocprof_calls:
            # Extract the rocprof -o output prefix (/tmp/rp_<m>_<n>_<k>.csv).
            # Avoid matching SSH's own -o options by looking for the /tmp/rp_ path.
            parts = cmd.split()
            for part in parts:
                if part.startswith("/tmp/rp_") and part.endswith(".csv"):
                    prefixes.add(part)
        self.assertEqual(len(prefixes), len(shapes))

    @patch("subprocess.run")
    def test_rocprof_includes_shape_args(self, mock_run):
        m, n, k = 128, 256, 512
        mock_run.side_effect = _side_effects_for_n_shapes(1)
        agent = _make_agent(executor=self.executor, shapes=[(m, n, k)])
        agent.run()

        rocprof_call = next(
            c for c in mock_run.call_args_list
            if "rocprof" in " ".join(c[0][0])
        )
        cmd_str = " ".join(rocprof_call[0][0])
        self.assertIn(f"--shape {m} {n} {k}", cmd_str)

    @patch("subprocess.run")
    def test_rocprof_includes_metric_time(self, mock_run):
        mock_run.side_effect = _side_effects_for_n_shapes(1)
        agent = _make_agent(executor=self.executor, shapes=[(64, 64, 64)])
        agent.run()

        rocprof_call = next(
            c for c in mock_run.call_args_list
            if "rocprof" in " ".join(c[0][0])
        )
        cmd_str = " ".join(rocprof_call[0][0])
        self.assertIn("--metric time", cmd_str)

    @patch("subprocess.run")
    def test_rocprof_includes_layout_TN(self, mock_run):
        mock_run.side_effect = _side_effects_for_n_shapes(1)
        agent = _make_agent(executor=self.executor, shapes=[(64, 64, 64)])
        agent.run()

        rocprof_call = next(
            c for c in mock_run.call_args_list
            if "rocprof" in " ".join(c[0][0])
        )
        cmd_str = " ".join(rocprof_call[0][0])
        self.assertIn("--layout TN", cmd_str)

    @patch("subprocess.run")
    def test_rocprof_with_shuffle_flag(self, mock_run):
        mock_run.side_effect = _side_effects_for_n_shapes(1)
        agent = _make_agent(executor=self.executor, shapes=[(64, 64, 64)], shuffle=True)
        agent.run()

        rocprof_call = next(
            c for c in mock_run.call_args_list
            if "rocprof" in " ".join(c[0][0])
        )
        cmd_str = " ".join(rocprof_call[0][0])
        self.assertIn("--shuffle", cmd_str)

    @patch("subprocess.run")
    def test_rocprof_without_shuffle_flag(self, mock_run):
        mock_run.side_effect = _side_effects_for_n_shapes(1)
        agent = _make_agent(executor=self.executor, shapes=[(64, 64, 64)], shuffle=False)
        agent.run()

        rocprof_call = next(
            c for c in mock_run.call_args_list
            if "rocprof" in " ".join(c[0][0])
        )
        cmd_str = " ".join(rocprof_call[0][0])
        self.assertNotIn("--shuffle", cmd_str)

    @patch("subprocess.run")
    def test_hip_visible_devices_env_set(self, mock_run):
        """docker exec must be called with HIP_VISIBLE_DEVICES set."""
        mock_run.side_effect = _side_effects_for_n_shapes(1)
        agent = _make_agent(executor=self.executor, shapes=[(64, 64, 64)], gpu_ids=[2])
        agent.run()

        docker_exec_calls = [
            " ".join(c[0][0])
            for c in mock_run.call_args_list
            if "docker" in " ".join(c[0][0]) and "rocprof" in " ".join(c[0][0])
        ]
        self.assertTrue(
            any("HIP_VISIBLE_DEVICES" in cmd and "2" in cmd for cmd in docker_exec_calls),
            "Expected HIP_VISIBLE_DEVICES=2 in docker exec call",
        )

    @patch("subprocess.run")
    def test_stats_csv_read_via_docker_exec_cat(self, mock_run):
        """stats CSV must be read back via docker exec cat *.stats.csv."""
        mock_run.side_effect = _side_effects_for_n_shapes(1)
        agent = _make_agent(executor=self.executor, shapes=[(64, 64, 64)])
        agent.run()

        cat_calls = [
            " ".join(c[0][0])
            for c in mock_run.call_args_list
            if "cat" in " ".join(c[0][0]) and "stats.csv" in " ".join(c[0][0])
        ]
        self.assertGreater(len(cat_calls), 0, "Expected a 'cat *.stats.csv' docker exec call")


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------


class TestStatsCsvParsing(unittest.TestCase):
    """Verify that stats CSV content is parsed correctly via parse_stats_csv."""

    def setUp(self):
        self.executor = _make_executor()

    @patch("subprocess.run")
    def test_main_ns_extracted_from_csv(self, mock_run):
        mock_run.side_effect = _side_effects_for_n_shapes(1, stats_csv=MAIN_ONLY_CSV)
        agent = _make_agent(executor=self.executor, shapes=[(64, 64, 64)])
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(result.data["results"][0]["main_ns"], 5673)

    @patch("subprocess.run")
    def test_reduce_ns_zero_when_no_reduce_kernel(self, mock_run):
        mock_run.side_effect = _side_effects_for_n_shapes(1, stats_csv=MAIN_ONLY_CSV)
        agent = _make_agent(executor=self.executor, shapes=[(64, 64, 64)])
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(result.data["results"][0]["reduce_ns"], 0)

    @patch("subprocess.run")
    def test_main_and_reduce_ns_extracted(self, mock_run):
        mock_run.side_effect = _side_effects_for_n_shapes(1, stats_csv=MAIN_AND_REDUCE_CSV)
        agent = _make_agent(executor=self.executor, shapes=[(64, 64, 64)])
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        r = result.data["results"][0]
        self.assertEqual(r["main_ns"], 5673)
        self.assertEqual(r["reduce_ns"], 2933)

    @patch("subprocess.run")
    def test_total_ns_is_sum_of_main_and_reduce(self, mock_run):
        mock_run.side_effect = _side_effects_for_n_shapes(1, stats_csv=MAIN_AND_REDUCE_CSV)
        agent = _make_agent(executor=self.executor, shapes=[(64, 64, 64)])
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        r = result.data["results"][0]
        self.assertEqual(r["total_ns"], r["main_ns"] + r["reduce_ns"])

    @patch("subprocess.run")
    def test_sample_stats_fixture_parsed(self, mock_run):
        """sample_stats.csv fixture must produce the same timings as MAIN_AND_REDUCE_CSV."""
        mock_run.side_effect = _side_effects_for_n_shapes(1, stats_csv=SAMPLE_STATS_CSV)
        agent = _make_agent(executor=self.executor, shapes=[(64, 64, 64)])
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        r = result.data["results"][0]
        self.assertEqual(r["main_ns"], 5673)
        self.assertEqual(r["reduce_ns"], 2933)

    @patch("subprocess.run")
    def test_result_dict_contains_mnk(self, mock_run):
        m, n, k = 128, 256, 512
        mock_run.side_effect = _side_effects_for_n_shapes(1)
        agent = _make_agent(executor=self.executor, shapes=[(m, n, k)])
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        r = result.data["results"][0]
        self.assertEqual(r["m"], m)
        self.assertEqual(r["n"], n)
        self.assertEqual(r["k"], k)


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------


class TestReturnStructure(unittest.TestCase):
    def setUp(self):
        self.executor = _make_executor()

    @patch("subprocess.run")
    def test_shapes_validated_count(self, mock_run):
        shapes = [(64, 64, 64), (128, 128, 128)]
        mock_run.side_effect = _side_effects_for_n_shapes(len(shapes))
        agent = _make_agent(executor=self.executor, shapes=shapes)
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(result.data["shapes_validated"], len(shapes))

    @patch("subprocess.run")
    def test_results_list_length_matches_shapes(self, mock_run):
        shapes = [(64, 64, 64), (128, 128, 128), (256, 256, 256)]
        mock_run.side_effect = _side_effects_for_n_shapes(len(shapes))
        agent = _make_agent(executor=self.executor, shapes=shapes)
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(len(result.data["results"]), len(shapes))

    @patch("subprocess.run")
    def test_regressions_key_present(self, mock_run):
        mock_run.side_effect = _side_effects_for_n_shapes(1)
        agent = _make_agent(executor=self.executor, shapes=[(64, 64, 64)])
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertIn("regressions", result.data)

    @patch("subprocess.run")
    def test_improved_count_key_present(self, mock_run):
        mock_run.side_effect = _side_effects_for_n_shapes(1)
        agent = _make_agent(executor=self.executor, shapes=[(64, 64, 64)])
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertIn("improved_count", result.data)

    @patch("subprocess.run")
    def test_regression_count_key_present(self, mock_run):
        mock_run.side_effect = _side_effects_for_n_shapes(1)
        agent = _make_agent(executor=self.executor, shapes=[(64, 64, 64)])
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertIn("regression_count", result.data)

    @patch("subprocess.run")
    def test_regression_count_matches_regressions_list(self, mock_run):
        mock_run.side_effect = _side_effects_for_n_shapes(1)
        agent = _make_agent(executor=self.executor, shapes=[(64, 64, 64)])
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(result.data["regression_count"], len(result.data["regressions"]))

    @patch("subprocess.run")
    def test_empty_shapes_returns_zero_validated(self, mock_run):
        # Preflight: mkdir + rocm-smi, then cleanup
        mock_run.side_effect = [
            _completed(stdout=""),  # mkdir
            _completed(stdout=""),  # rocm-smi
            _completed(stdout=""),  # cleanup (docker exec rm)
        ]
        agent = _make_agent(executor=self.executor, shapes=[], gpu_ids=[0])
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(result.data["shapes_validated"], 0)
        self.assertEqual(result.data["results"], [])


# ---------------------------------------------------------------------------
# GPU parallelization
# ---------------------------------------------------------------------------


class TestGpuParallelization(unittest.TestCase):
    """Verify shapes are distributed round-robin across available GPUs."""

    def setUp(self):
        self.executor = _make_executor()

    @patch("subprocess.run")
    def test_two_shapes_two_gpus_different_hip_devices(self, mock_run):
        """With 2 shapes and 2 GPUs, each shape should go to a different GPU."""
        shapes = [(64, 64, 64), (128, 128, 128)]
        # With 2 GPUs, both rocprof calls run concurrently via ThreadPoolExecutor;
        # side_effects order may vary. Provide enough responses.
        mock_run.side_effect = _side_effects_for_n_shapes(len(shapes))
        agent = _make_agent(
            executor=self.executor, shapes=shapes, gpu_ids=[0, 1]
        )
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(result.data["shapes_validated"], 2)

    @patch("subprocess.run")
    def test_single_gpu_processes_all_shapes(self, mock_run):
        shapes = [(64, 64, 64), (128, 128, 128), (256, 256, 256)]
        mock_run.side_effect = _side_effects_for_n_shapes(len(shapes))
        agent = _make_agent(executor=self.executor, shapes=shapes, gpu_ids=[0])
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(result.data["shapes_validated"], len(shapes))

    @patch("subprocess.run")
    def test_shapes_distributed_round_robin(self, mock_run):
        """With 2 GPUs, shapes 0,2,4 go to GPU 0 and 1,3 go to GPU 1."""
        shapes = [
            (16, 64, 64),
            (32, 64, 64),
            (64, 64, 64),
            (128, 64, 64),
        ]
        # Enough side_effects for parallel execution
        effects = [
            _completed(stdout=""),  # mkdir
            _completed(stdout=""),  # rocm-smi
        ]
        for _ in shapes:
            effects.append(_completed(stdout=""))    # rocprof
            effects.append(_completed(stdout=MAIN_ONLY_CSV))  # cat
        effects.append(_completed(stdout=""))  # cleanup
        mock_run.side_effect = effects

        agent = _make_agent(executor=self.executor, shapes=shapes, gpu_ids=[0, 1])
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(result.data["shapes_validated"], len(shapes))

        # Collect HIP_VISIBLE_DEVICES values from all docker exec rocprof calls
        hip_devices = []
        for c in mock_run.call_args_list:
            cmd_str = " ".join(c[0][0])
            if "docker" in cmd_str and "rocprof" in cmd_str and "HIP_VISIBLE_DEVICES" in cmd_str:
                # The entire docker command is a single string element in the SSH
                # command list.  Split it by whitespace to find the
                # HIP_VISIBLE_DEVICES=<val> token, then extract just the value.
                for part in c[0][0]:
                    if "HIP_VISIBLE_DEVICES=" in part:
                        for token in part.split():
                            if token.startswith("HIP_VISIBLE_DEVICES="):
                                val = token.split("=", 1)[1].strip("'")
                                hip_devices.append(val)
                                break
                        break

        # Both GPU 0 and GPU 1 should appear.
        self.assertIn("0", hip_devices)
        self.assertIn("1", hip_devices)


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------


class TestClassificationImproved(unittest.TestCase):
    """Tuned kernel is faster than baseline by > threshold → IMPROVED."""

    def setUp(self):
        self.executor = _make_executor()

    def _make_csv_with_ns(self, main_ns: int) -> str:
        return (
            '"Name","Calls","TotalDurationNs","AverageNs","Percentage"\n'
            f'"_gemm_afp4wfp4_preshuffle_kernel_BLOCK_SIZE_M_8.kd",1,{main_ns},{main_ns},50.0\n'
        )

    @patch("subprocess.run")
    def test_improved_shape_increments_improved_count(self, mock_run):
        # baseline = 10000 ns, tuned = 8000 ns → 20% improvement > 5% threshold
        baseline_data = {"M64_N64_K64": {"total_ns": 10000}}
        untuned_data = {"M64_N64_K64": {"total_ns": 10000}}
        tuned_ns = 8000
        stats_csv = self._make_csv_with_ns(tuned_ns)
        mock_run.side_effect = _side_effects_for_n_shapes(1, stats_csv=stats_csv)
        agent = _make_agent(
            executor=self.executor,
            shapes=[(64, 64, 64)],
            baseline_data=baseline_data,
            untuned_data=untuned_data,
            threshold=0.05,
        )
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(result.data["improved_count"], 1)
        self.assertEqual(result.data["regression_count"], 0)

    @patch("subprocess.run")
    def test_improved_shape_not_in_regressions(self, mock_run):
        baseline_data = {"M64_N64_K64": {"total_ns": 10000}}
        untuned_data = {"M64_N64_K64": {"total_ns": 10000}}
        tuned_ns = 8000
        stats_csv = self._make_csv_with_ns(tuned_ns)
        mock_run.side_effect = _side_effects_for_n_shapes(1, stats_csv=stats_csv)
        agent = _make_agent(
            executor=self.executor,
            shapes=[(64, 64, 64)],
            baseline_data=baseline_data,
            untuned_data=untuned_data,
        )
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(result.data["regressions"], [])


class TestClassificationNeutral(unittest.TestCase):
    """Tuned kernel is within threshold of baseline → NEUTRAL."""

    def setUp(self):
        self.executor = _make_executor()

    def _make_csv_with_ns(self, main_ns: int) -> str:
        return (
            '"Name","Calls","TotalDurationNs","AverageNs","Percentage"\n'
            f'"_gemm_afp4wfp4_preshuffle_kernel.kd",1,{main_ns},{main_ns},50.0\n'
        )

    @patch("subprocess.run")
    def test_neutral_shape_not_counted_as_regression_or_improvement(self, mock_run):
        # baseline = 10000, tuned = 10100 → 1% delta, well within 5% threshold
        baseline_data = {"M64_N64_K64": {"total_ns": 10000}}
        untuned_data = {"M64_N64_K64": {"total_ns": 10100}}
        tuned_ns = 10100
        stats_csv = self._make_csv_with_ns(tuned_ns)
        mock_run.side_effect = _side_effects_for_n_shapes(1, stats_csv=stats_csv)
        agent = _make_agent(
            executor=self.executor,
            shapes=[(64, 64, 64)],
            baseline_data=baseline_data,
            untuned_data=untuned_data,
            threshold=0.05,
        )
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(result.data["improved_count"], 0)
        self.assertEqual(result.data["regression_count"], 0)


class TestClassificationTuningRegression(unittest.TestCase):
    """Tuned is worse than untuned beyond tuning_threshold → TUNING_REGRESSION."""

    def setUp(self):
        self.executor = _make_executor()

    def _make_csv_with_ns(self, main_ns: int) -> str:
        return (
            '"Name","Calls","TotalDurationNs","AverageNs","Percentage"\n'
            f'"_gemm_afp4wfp4_preshuffle_kernel.kd",1,{main_ns},{main_ns},50.0\n'
        )

    @patch("subprocess.run")
    def test_tuning_regression_detected(self, mock_run):
        # baseline = 10000, untuned = 10200, tuned = 11000
        # tuned vs untuned delta = 8% > 5% threshold → TUNING_REGRESSION
        baseline_data = {"M64_N64_K64": {"total_ns": 10000}}
        untuned_data = {"M64_N64_K64": {"total_ns": 10200}}
        tuned_ns = 11000
        stats_csv = self._make_csv_with_ns(tuned_ns)
        mock_run.side_effect = _side_effects_for_n_shapes(1, stats_csv=stats_csv)
        agent = _make_agent(
            executor=self.executor,
            shapes=[(64, 64, 64)],
            baseline_data=baseline_data,
            untuned_data=untuned_data,
            threshold=0.05,
        )
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(result.data["regression_count"], 1)
        self.assertIn(
            result.data["regressions"][0]["classification"],
            ("tuning_regression", "tuning_regression_severe"),
        )

    @patch("subprocess.run")
    def test_tuning_regression_has_mnk(self, mock_run):
        baseline_data = {"M64_N64_K64": {"total_ns": 10000}}
        untuned_data = {"M64_N64_K64": {"total_ns": 10200}}
        tuned_ns = 11000
        stats_csv = self._make_csv_with_ns(tuned_ns)
        mock_run.side_effect = _side_effects_for_n_shapes(1, stats_csv=stats_csv)
        agent = _make_agent(
            executor=self.executor,
            shapes=[(64, 64, 64)],
            baseline_data=baseline_data,
            untuned_data=untuned_data,
        )
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        reg = result.data["regressions"][0]
        self.assertEqual(reg["m"], 64)
        self.assertEqual(reg["n"], 64)
        self.assertEqual(reg["k"], 64)

    @patch("subprocess.run")
    def test_regression_has_delta_field(self, mock_run):
        baseline_data = {"M64_N64_K64": {"total_ns": 10000}}
        untuned_data = {"M64_N64_K64": {"total_ns": 10200}}
        tuned_ns = 11000
        stats_csv = self._make_csv_with_ns(tuned_ns)
        mock_run.side_effect = _side_effects_for_n_shapes(1, stats_csv=stats_csv)
        agent = _make_agent(
            executor=self.executor,
            shapes=[(64, 64, 64)],
            baseline_data=baseline_data,
            untuned_data=untuned_data,
        )
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        reg = result.data["regressions"][0]
        self.assertIn("delta", reg)


class TestClassificationNoBaseline(unittest.TestCase):
    """Without baseline_data/untuned_data, no classification is performed."""

    def setUp(self):
        self.executor = _make_executor()

    @patch("subprocess.run")
    def test_no_baseline_no_regressions(self, mock_run):
        mock_run.side_effect = _side_effects_for_n_shapes(1)
        agent = _make_agent(
            executor=self.executor,
            shapes=[(64, 64, 64)],
            baseline_data=None,
            untuned_data=None,
        )
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(result.data["regression_count"], 0)
        self.assertEqual(result.data["improved_count"], 0)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


class TestCleanup(unittest.TestCase):
    """Verify that temp files are cleaned up after benchmarking."""

    def setUp(self):
        self.executor = _make_executor()

    @patch("subprocess.run")
    def test_cleanup_rm_called(self, mock_run):
        mock_run.side_effect = _side_effects_for_n_shapes(1)
        agent = _make_agent(executor=self.executor, shapes=[(64, 64, 64)])
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)

        rm_calls = [
            " ".join(c[0][0])
            for c in mock_run.call_args_list
            if "rm" in " ".join(c[0][0]) and "rp_" in " ".join(c[0][0])
        ]
        self.assertGreater(len(rm_calls), 0, "Expected a cleanup rm -f /tmp/rp_* call")

    @patch("subprocess.run")
    def test_cleanup_removes_csv_and_stats_csv(self, mock_run):
        mock_run.side_effect = _side_effects_for_n_shapes(1)
        agent = _make_agent(executor=self.executor, shapes=[(64, 64, 64)])
        agent.run()

        rm_calls = [
            " ".join(c[0][0])
            for c in mock_run.call_args_list
            if "rm" in " ".join(c[0][0]) and "rp_" in " ".join(c[0][0])
        ]
        self.assertTrue(len(rm_calls) > 0)
        combined = " ".join(rm_calls)
        self.assertIn(".csv", combined)
        self.assertIn("stats.csv", combined)

    @patch("subprocess.run")
    def test_cleanup_called_even_after_multiple_shapes(self, mock_run):
        shapes = [(64, 64, 64), (128, 128, 128)]
        mock_run.side_effect = _side_effects_for_n_shapes(len(shapes))
        agent = _make_agent(executor=self.executor, shapes=shapes)
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)

        rm_calls = [
            c for c in mock_run.call_args_list
            if "rm" in " ".join(c[0][0]) and "rp_" in " ".join(c[0][0])
        ]
        # Cleanup should happen exactly once (bulk rm)
        self.assertEqual(len(rm_calls), 1)


if __name__ == "__main__":
    unittest.main()
