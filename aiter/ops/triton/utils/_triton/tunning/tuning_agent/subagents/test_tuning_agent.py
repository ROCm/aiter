"""Tests for TuningAgent."""

from __future__ import annotations

import subprocess
import unittest
from collections import defaultdict
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, call, patch

from ..remote import RemoteExecutor
from ..types import MachineInfo
from .tuning_agent import TuningAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    """Return a mock subprocess.CompletedProcess."""
    r = MagicMock(spec=subprocess.CompletedProcess)
    r.returncode = returncode
    r.stdout = stdout
    r.stderr = stderr
    return r


def _make_machine(**kwargs) -> MachineInfo:
    defaults = {
        "host": "gpu-host.example.com",
        "user": "testuser",
        "ssh_key": "/home/testuser/.ssh/id_rsa",
        "gpus": [0, 1, 2, 3],
    }
    defaults.update(kwargs)
    return MachineInfo(**defaults)


def _make_executor(machine=None) -> RemoteExecutor:
    m = machine or _make_machine()
    executor = RemoteExecutor(m)
    executor.container_id = "testcontainer123"
    return executor


def _make_search_space() -> Dict[str, Any]:
    """Minimal search space with block-size lists and stages."""
    return {
        "M_LEQ_16": {
            "BLOCK_SIZE_M": [16, 32],
            "BLOCK_SIZE_N": [64, 128],
            "BLOCK_SIZE_K": [32, 64],
            "num_stages": [2, 3],
            "matrix_instr_nonkdim": [16],
        },
        "M_LEQ_256": {
            "BLOCK_SIZE_M": [64, 128],
            "BLOCK_SIZE_N": [128, 256],
            "BLOCK_SIZE_K": [64],
            "num_stages": {"min": 2, "max": 4},
            "matrix_instr_nonkdim": [16, 32],
        },
    }


def _make_agent(
    shapes=None,
    search_space=None,
    gpu_ids=None,
    max_batch=100,
    executor=None,
) -> TuningAgent:
    executor = executor or _make_executor()
    shapes = shapes or [(16, 128, 4096), (32, 128, 4096)]
    search_space = search_space or _make_search_space()
    gpu_ids = gpu_ids or [0, 1]
    return TuningAgent(
        executor=executor,
        kernel_name="gemm",
        artifact_dir="/artifacts",
        shapes_to_tune=shapes,
        search_space=search_space,
        ut_script="ut_gemm.py",
        gpu_ids=gpu_ids,
        tunning_dir="/workspace/tunning",
        max_batch=max_batch,
    )


# ---------------------------------------------------------------------------
# build_screen_command tests
# ---------------------------------------------------------------------------


class TestBuildScreenCommand(unittest.TestCase):
    """Tests for TuningAgent.build_screen_command."""

    def setUp(self):
        self.agent = _make_agent()
        self.ss = {
            "BLOCK_SIZE_M": [16, 32, 64],
            "BLOCK_SIZE_N": [64, 128],
            "BLOCK_SIZE_K": [32, 64, 128],
            "num_stages": [2, 4],
            "matrix_instr_nonkdim": [16],
        }

    def test_starts_with_python_screen_py(self):
        cmd = self.agent.build_screen_command(16, 128, 4096, 0, self.ss)
        self.assertTrue(cmd.startswith("python screen.py"), msg=repr(cmd))

    def test_contains_m_n_k(self):
        cmd = self.agent.build_screen_command(16, 128, 4096, 0, self.ss)
        self.assertIn("16", cmd)
        self.assertIn("128", cmd)
        self.assertIn("4096", cmd)

    def test_contains_gpu_id(self):
        cmd = self.agent.build_screen_command(16, 128, 4096, 3, self.ss)
        # GPU ID appears as positional arg after M N K
        self.assertIn(" 3 ", cmd)

    def test_contains_ut_script(self):
        cmd = self.agent.build_screen_command(16, 128, 4096, 0, self.ss)
        self.assertIn("ut_gemm.py", cmd)

    def test_block_size_m_range_flag(self):
        cmd = self.agent.build_screen_command(16, 128, 4096, 0, self.ss)
        self.assertIn("--block-size-m-range", cmd)
        self.assertIn("16", cmd)
        self.assertIn("32", cmd)
        self.assertIn("64", cmd)

    def test_block_size_n_range_flag(self):
        cmd = self.agent.build_screen_command(16, 128, 4096, 0, self.ss)
        self.assertIn("--block-size-n-range", cmd)

    def test_block_size_k_range_flag(self):
        cmd = self.agent.build_screen_command(16, 128, 4096, 0, self.ss)
        self.assertIn("--block-size-k-range", cmd)

    def test_num_stages_range_flag(self):
        cmd = self.agent.build_screen_command(16, 128, 4096, 0, self.ss)
        self.assertIn("--num-stages-range", cmd)
        self.assertIn("2", cmd)
        self.assertIn("4", cmd)

    def test_matrix_instr_nonkdim_range_flag(self):
        cmd = self.agent.build_screen_command(16, 128, 4096, 0, self.ss)
        self.assertIn("--matrix-instr-nonkdim-range", cmd)
        self.assertIn("16", cmd)

    def test_overwrite_flag_present(self):
        cmd = self.agent.build_screen_command(16, 128, 4096, 0, self.ss)
        self.assertIn("--overwrite", cmd)

    def test_no_env_prefix_when_max_batch_is_100(self):
        agent = _make_agent(max_batch=100)
        cmd = agent.build_screen_command(16, 128, 4096, 0, self.ss)
        self.assertNotIn("SCREEN_MAX_BATCH", cmd)

    def test_num_stages_dict_format(self):
        """When num_stages is a dict with min/max, both values appear."""
        ss = dict(self.ss)
        ss["num_stages"] = {"min": 1, "max": 5}
        cmd = self.agent.build_screen_command(16, 128, 4096, 0, ss)
        self.assertIn("--num-stages-range", cmd)
        self.assertIn("1", cmd)
        self.assertIn("5", cmd)

    def test_scalar_block_sizes_work(self):
        """Scalar (non-list) values in search space are still included."""
        ss = {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "num_stages": 3,
            "matrix_instr_nonkdim": 16,
        }
        cmd = self.agent.build_screen_command(32, 64, 128, 0, ss)
        self.assertIn("--block-size-m-range", cmd)
        self.assertIn("32", cmd)

    def test_command_flag_order(self):
        """Flags appear in expected order after the positional arguments."""
        cmd = self.agent.build_screen_command(16, 128, 4096, 0, self.ss)
        bm_pos = cmd.index("--block-size-m-range")
        bn_pos = cmd.index("--block-size-n-range")
        bk_pos = cmd.index("--block-size-k-range")
        ns_pos = cmd.index("--num-stages-range")
        mi_pos = cmd.index("--matrix-instr-nonkdim-range")
        ow_pos = cmd.index("--overwrite")
        self.assertLess(bm_pos, bn_pos)
        self.assertLess(bn_pos, bk_pos)
        self.assertLess(bk_pos, ns_pos)
        self.assertLess(ns_pos, mi_pos)
        self.assertLess(mi_pos, ow_pos)


# ---------------------------------------------------------------------------
# build_screen_command with max_batch prefix tests
# ---------------------------------------------------------------------------


class TestBuildScreenCommandWithMaxBatch(unittest.TestCase):
    """Tests for SCREEN_MAX_BATCH prefix behaviour."""

    def _ss(self) -> Dict[str, Any]:
        return {
            "BLOCK_SIZE_M": [16],
            "BLOCK_SIZE_N": [64],
            "BLOCK_SIZE_K": [32],
            "num_stages": [2, 4],
            "matrix_instr_nonkdim": [16],
        }

    def test_max_batch_not_100_adds_prefix(self):
        agent = _make_agent(max_batch=50)
        cmd = agent.build_screen_command(16, 128, 4096, 0, self._ss())
        self.assertTrue(
            cmd.startswith("SCREEN_MAX_BATCH=50"),
            msg=f"Expected SCREEN_MAX_BATCH=50 prefix, got: {cmd!r}",
        )

    def test_max_batch_prefix_followed_by_python(self):
        agent = _make_agent(max_batch=200)
        cmd = agent.build_screen_command(16, 128, 4096, 0, self._ss())
        self.assertIn("SCREEN_MAX_BATCH=200 python screen.py", cmd)

    def test_max_batch_1_adds_prefix(self):
        agent = _make_agent(max_batch=1)
        cmd = agent.build_screen_command(16, 128, 4096, 0, self._ss())
        self.assertTrue(cmd.startswith("SCREEN_MAX_BATCH=1"))

    def test_max_batch_0_adds_prefix(self):
        agent = _make_agent(max_batch=0)
        cmd = agent.build_screen_command(16, 128, 4096, 0, self._ss())
        self.assertTrue(cmd.startswith("SCREEN_MAX_BATCH=0"))

    def test_max_batch_99_adds_prefix(self):
        agent = _make_agent(max_batch=99)
        cmd = agent.build_screen_command(16, 128, 4096, 0, self._ss())
        self.assertTrue(cmd.startswith("SCREEN_MAX_BATCH=99"))

    def test_max_batch_101_adds_prefix(self):
        agent = _make_agent(max_batch=101)
        cmd = agent.build_screen_command(16, 128, 4096, 0, self._ss())
        self.assertTrue(cmd.startswith("SCREEN_MAX_BATCH=101"))


# ---------------------------------------------------------------------------
# distribute_shapes_to_gpus tests
# ---------------------------------------------------------------------------


class TestDistributeShapesToGpus(unittest.TestCase):
    """Tests for TuningAgent.distribute_shapes_to_gpus."""

    def setUp(self):
        self.agent = _make_agent()

    def test_returns_dict_keyed_by_gpu_id(self):
        shapes = [(m, 128, 4096) for m in [1, 2, 4, 8, 16, 32, 64, 128]]
        result = self.agent.distribute_shapes_to_gpus(shapes, [0, 1, 2, 3])
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), {0, 1, 2, 3})

    def test_8_shapes_across_4_gpus_distributes_2_each(self):
        """8 distinct M values for the same (N, K) → 2 shapes per GPU."""
        shapes = [(m, 128, 4096) for m in [1, 2, 4, 8, 16, 32, 64, 128]]
        result = self.agent.distribute_shapes_to_gpus(shapes, [0, 1, 2, 3])
        for gpu_id in [0, 1, 2, 3]:
            self.assertEqual(
                len(result[gpu_id]),
                2,
                msg=f"GPU {gpu_id} should have 2 shapes, got {result[gpu_id]}",
            )

    def test_total_shapes_preserved(self):
        shapes = [(m, 128, 4096) for m in [1, 2, 4, 8, 16, 32, 64, 128]]
        result = self.agent.distribute_shapes_to_gpus(shapes, [0, 1, 2, 3])
        total = sum(len(v) for v in result.values())
        self.assertEqual(total, 8)

    def test_no_shape_appears_on_multiple_gpus(self):
        shapes = [(m, 128, 4096) for m in [1, 2, 4, 8, 16, 32, 64, 128]]
        result = self.agent.distribute_shapes_to_gpus(shapes, [0, 1, 2, 3])
        all_shapes: List[Tuple] = []
        for assigned in result.values():
            all_shapes.extend(assigned)
        self.assertEqual(len(all_shapes), len(set(all_shapes)))

    def test_single_gpu_gets_all_shapes(self):
        shapes = [(m, 64, 2048) for m in [1, 4, 16, 64]]
        result = self.agent.distribute_shapes_to_gpus(shapes, [2])
        self.assertEqual(len(result[2]), 4)

    def test_empty_gpu_list_returns_empty_dict(self):
        shapes = [(1, 128, 4096)]
        result = self.agent.distribute_shapes_to_gpus(shapes, [])
        self.assertEqual(result, {})

    def test_empty_shapes_returns_empty_lists(self):
        result = self.agent.distribute_shapes_to_gpus([], [0, 1, 2, 3])
        for gpu_id in [0, 1, 2, 3]:
            self.assertEqual(result[gpu_id], [])

    def test_more_gpus_than_shapes_some_gpus_empty(self):
        shapes = [(1, 128, 4096), (2, 128, 4096)]
        result = self.agent.distribute_shapes_to_gpus(shapes, [0, 1, 2, 3])
        total = sum(len(v) for v in result.values())
        self.assertEqual(total, 2)

    def test_duplicate_m_values_deduplicated(self):
        shapes = [(16, 128, 4096), (16, 128, 4096), (32, 128, 4096)]
        result = self.agent.distribute_shapes_to_gpus(shapes, [0, 1])
        total = sum(len(v) for v in result.values())
        # After dedup: M=16 and M=32 → 2 unique shapes.
        self.assertEqual(total, 2)


# ---------------------------------------------------------------------------
# select_scout_shapes tests
# ---------------------------------------------------------------------------


class TestSelectScoutShapes(unittest.TestCase):
    """Tests for TuningAgent.select_scout_shapes."""

    def setUp(self):
        self.agent = _make_agent()

    def test_returns_list(self):
        shapes = [(m, 128, 4096) for m in [1, 4, 16, 64, 256, 1024, 4096, 8192]]
        result = self.agent.select_scout_shapes(shapes)
        self.assertIsInstance(result, list)

    def test_picks_smallest_m_per_nk(self):
        shapes = [(m, 128, 4096) for m in [1, 4, 16, 64, 256, 1024, 4096, 8192]]
        result = self.agent.select_scout_shapes(shapes)
        m_values = {s[0] for s in result}
        self.assertIn(1, m_values, "Smallest M (1) must be in scout shapes")

    def test_picks_largest_m_per_nk(self):
        shapes = [(m, 128, 4096) for m in [1, 4, 16, 64, 256, 1024, 4096, 8192]]
        result = self.agent.select_scout_shapes(shapes)
        m_values = {s[0] for s in result}
        self.assertIn(8192, m_values, "Largest M (8192) must be in scout shapes")

    def test_picks_middle_m_per_nk(self):
        shapes = [(m, 128, 4096) for m in [1, 4, 16, 64, 256, 1024, 4096, 8192]]
        result = self.agent.select_scout_shapes(shapes)
        m_values = {s[0] for s in result}
        # Middle of sorted [1,4,16,64,256,1024,4096,8192] at index 4 = 256
        self.assertIn(256, m_values, "Middle M must be in scout shapes")

    def test_no_duplicates_in_result(self):
        shapes = [(m, 128, 4096) for m in [1, 4, 16]]
        result = self.agent.select_scout_shapes(shapes)
        self.assertEqual(len(result), len(set(result)))

    def test_single_shape_returns_it(self):
        shapes = [(64, 128, 4096)]
        result = self.agent.select_scout_shapes(shapes)
        self.assertEqual(result, [(64, 128, 4096)])

    def test_two_shapes_returns_both(self):
        shapes = [(1, 128, 4096), (64, 128, 4096)]
        result = self.agent.select_scout_shapes(shapes)
        self.assertEqual(len(result), 2)
        m_values = {s[0] for s in result}
        self.assertIn(1, m_values)
        self.assertIn(64, m_values)

    def test_multiple_nk_pairs_all_represented(self):
        """Each (N,K) pair should contribute scouts."""
        shapes = (
            [(m, 128, 4096) for m in [1, 32, 256]]
            + [(m, 256, 8192) for m in [1, 64, 512]]
        )
        result = self.agent.select_scout_shapes(shapes)
        nk_pairs = {(s[1], s[2]) for s in result}
        self.assertIn((128, 4096), nk_pairs)
        self.assertIn((256, 8192), nk_pairs)

    def test_all_results_are_valid_shapes(self):
        """Every returned shape must be in the original input."""
        shapes = [(m, 128, 4096) for m in [2, 8, 32, 128, 512, 2048]]
        result = self.agent.select_scout_shapes(shapes)
        shape_set = set(shapes)
        for s in result:
            self.assertIn(s, shape_set)

    def test_empty_shapes_returns_empty(self):
        result = self.agent.select_scout_shapes([])
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# _execute tests (mocked executor)
# ---------------------------------------------------------------------------


class TestExecuteLaunchesScreenPerGpu(unittest.TestCase):
    """Tests for TuningAgent._execute — verifies docker_exec called per GPU."""

    def _make_mock_executor(self) -> MagicMock:
        executor = MagicMock(spec=RemoteExecutor)
        executor.container_id = "testcontainer"
        executor.docker_exec.return_value = _completed(returncode=0)
        executor.ssh_run.return_value = _completed(returncode=0)
        return executor

    def test_docker_exec_called_for_each_shape(self):
        """docker_exec should be called once per (M, N, K) shape."""
        executor = self._make_mock_executor()
        shapes = [(16, 128, 4096), (32, 128, 4096), (64, 128, 4096)]
        agent = TuningAgent(
            executor=executor,
            kernel_name="gemm",
            artifact_dir="/artifacts",
            shapes_to_tune=shapes,
            search_space=_make_search_space(),
            ut_script="ut_gemm.py",
            gpu_ids=[0, 1],
            tunning_dir="/workspace/tunning",
        )
        # Bypass _preflight by calling _execute directly.
        result = agent._execute()
        # One docker_exec call per shape.
        self.assertEqual(executor.docker_exec.call_count, len(shapes))

    def test_docker_exec_called_per_gpu_across_shapes(self):
        """With 4 shapes and 2 GPUs, docker_exec is called 4 times total."""
        executor = self._make_mock_executor()
        shapes = [(m, 128, 4096) for m in [8, 16, 32, 64]]
        agent = TuningAgent(
            executor=executor,
            kernel_name="gemm",
            artifact_dir="/artifacts",
            shapes_to_tune=shapes,
            search_space=_make_search_space(),
            ut_script="ut_gemm.py",
            gpu_ids=[0, 1],
            tunning_dir="/workspace/tunning",
        )
        agent._execute()
        self.assertEqual(executor.docker_exec.call_count, 4)

    def test_execute_returns_pairs_tuned(self):
        """_execute returns a dict with 'pairs_tuned'."""
        executor = self._make_mock_executor()
        shapes = [(16, 128, 4096), (32, 256, 8192)]
        agent = TuningAgent(
            executor=executor,
            kernel_name="gemm",
            artifact_dir="/artifacts",
            shapes_to_tune=shapes,
            search_space=_make_search_space(),
            ut_script="ut_gemm.py",
            gpu_ids=[0, 1],
            tunning_dir="/workspace/tunning",
        )
        result = agent._execute()
        self.assertIn("pairs_tuned", result)
        # Two distinct (N,K) pairs → pairs_tuned == 2.
        self.assertEqual(result["pairs_tuned"], 2)

    def test_execute_returns_log_dir(self):
        """_execute returns a dict with 'log_dir'."""
        executor = self._make_mock_executor()
        shapes = [(16, 128, 4096)]
        agent = TuningAgent(
            executor=executor,
            kernel_name="gemm",
            artifact_dir="/artifacts",
            shapes_to_tune=shapes,
            search_space=_make_search_space(),
            ut_script="ut_gemm.py",
            gpu_ids=[0],
            tunning_dir="/workspace/tunning",
        )
        result = agent._execute()
        self.assertIn("log_dir", result)
        self.assertIn("/artifacts", result["log_dir"])

    def test_docker_exec_uses_tunning_dir_as_workdir(self):
        """docker_exec is called with workdir set to tunning_dir."""
        executor = self._make_mock_executor()
        shapes = [(16, 128, 4096)]
        agent = TuningAgent(
            executor=executor,
            kernel_name="gemm",
            artifact_dir="/artifacts",
            shapes_to_tune=shapes,
            search_space=_make_search_space(),
            ut_script="ut_gemm.py",
            gpu_ids=[0],
            tunning_dir="/workspace/tunning",
        )
        agent._execute()
        call_kwargs = executor.docker_exec.call_args_list[0][1]
        self.assertEqual(call_kwargs.get("workdir"), "/workspace/tunning")

    def test_docker_exec_command_contains_screen_py(self):
        """The command passed to docker_exec contains 'screen.py'."""
        executor = self._make_mock_executor()
        shapes = [(16, 128, 4096)]
        agent = TuningAgent(
            executor=executor,
            kernel_name="gemm",
            artifact_dir="/artifacts",
            shapes_to_tune=shapes,
            search_space=_make_search_space(),
            ut_script="ut_gemm.py",
            gpu_ids=[0],
            tunning_dir="/workspace/tunning",
        )
        agent._execute()
        cmd_arg = executor.docker_exec.call_args_list[0][0][0]
        self.assertIn("screen.py", cmd_arg)

    def test_empty_shapes_returns_zero_pairs_tuned(self):
        """With no shapes, _execute should return pairs_tuned == 0."""
        executor = self._make_mock_executor()
        agent = TuningAgent(
            executor=executor,
            kernel_name="gemm",
            artifact_dir="/artifacts",
            shapes_to_tune=[],
            search_space=_make_search_space(),
            ut_script="ut_gemm.py",
            gpu_ids=[0, 1],
            tunning_dir="/workspace/tunning",
        )
        result = agent._execute()
        self.assertEqual(result["pairs_tuned"], 0)
        executor.docker_exec.assert_not_called()

    def test_run_wraps_execute_in_subagent_result(self):
        """run() wraps _execute output in SubagentResult."""
        from .base import SubagentResult

        executor = self._make_mock_executor()
        # ssh_run is called by _preflight (mkdir, rocm-smi).
        executor.ssh_run.return_value = _completed(returncode=0)
        shapes = [(16, 128, 4096)]
        agent = TuningAgent(
            executor=executor,
            kernel_name="gemm",
            artifact_dir="/artifacts",
            shapes_to_tune=shapes,
            search_space=_make_search_space(),
            ut_script="ut_gemm.py",
            gpu_ids=[0],
            tunning_dir="/workspace/tunning",
        )
        result = agent.run()
        self.assertIsInstance(result, SubagentResult)
        self.assertTrue(result.success)
        self.assertIn("pairs_tuned", result.data)


# ---------------------------------------------------------------------------
# TuningAgent construction tests
# ---------------------------------------------------------------------------


class TestTuningAgentInit(unittest.TestCase):
    """Tests for TuningAgent constructor and class attributes."""

    def test_name_attribute(self):
        self.assertEqual(TuningAgent.name, "tuning")

    def test_stores_shapes_to_tune(self):
        agent = _make_agent(shapes=[(1, 2, 3)])
        self.assertEqual(agent.shapes_to_tune, [(1, 2, 3)])

    def test_stores_search_space(self):
        ss = {"M_LEQ_16": {"BLOCK_SIZE_M": [16]}}
        agent = _make_agent(search_space=ss)
        self.assertIs(agent.search_space, ss)

    def test_stores_gpu_ids(self):
        agent = _make_agent(gpu_ids=[0, 2, 4])
        self.assertEqual(agent.gpu_ids, [0, 2, 4])

    def test_default_max_batch(self):
        agent = _make_agent()
        self.assertEqual(agent.max_batch, 100)

    def test_custom_max_batch(self):
        agent = _make_agent(max_batch=50)
        self.assertEqual(agent.max_batch, 50)

    def test_stores_ut_script(self):
        agent = _make_agent()
        self.assertEqual(agent.ut_script, "ut_gemm.py")

    def test_stores_tunning_dir(self):
        agent = _make_agent()
        self.assertEqual(agent.tunning_dir, "/workspace/tunning")


if __name__ == "__main__":
    unittest.main()
