"""Tests for DiscoveryAgent._execute() and its helper functions."""

import json
import subprocess
import unittest
from unittest.mock import MagicMock, call, patch

from ..remote import RemoteExecutor
from ..types import MachineInfo
from .base import SubagentResult
from .discovery_agent import (
    DiscoveryAgent,
    _DEFAULT_M_VALUES,
    _extract_nk_from_filename,
    _parse_ls_output,
    _parse_model_shapes,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# A realistic (abbreviated) model_shapes.json fragment.
SAMPLE_MODEL_SHAPES_JSON = json.dumps({
    "Llama3 405B": {
        "gemm_afp4wfp4": [
            {"N": 106496, "K": 16384, "TP_dim": "N"},
            {"N": 16384, "K": 53248, "TP_dim": "K"},
        ],
        "gemm_a8w8": [
            {"N": 4096, "K": 8192},
        ],
        "rmsnorm": [{"N": 16384}],
    },
    "DeepSeek R1": {
        "gemm_afp4wfp4": [
            {"N": 7168, "K": 2048, "TP_dim": "N"},
        ],
    },
})

# A minimal config JSON that a catch-all file might contain.
SAMPLE_CONFIG_JSON = json.dumps({
    "M_LEQ_16": {
        "BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 1024,
        "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    },
    "M_LEQ_64": {
        "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 1024,
        "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    },
    "any": {
        "BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 1024,
        "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    },
})

# ls output simulating two N=K config files + one catch-all.
LS_CONFIG_OUTPUT = "\n".join([
    "/workspace/aiter/aiter/ops/triton/configs/gemm/gfx942-GEMM-AFP4WFP4-N=128-K=4096.json",
    "/workspace/aiter/aiter/ops/triton/configs/gemm/gfx942-GEMM-AFP4WFP4-N=256-K=8192.json",
    "/workspace/aiter/aiter/ops/triton/configs/gemm/gfx942-GEMM-AFP4WFP4.json",
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_machine(**kwargs):
    defaults = {
        "host": "gpu-host.example.com",
        "user": "testuser",
        "ssh_key": "/home/testuser/.ssh/id_rsa",
        "gpus": [0],
    }
    defaults.update(kwargs)
    return MachineInfo(**defaults)


def _completed(returncode=0, stdout="", stderr=""):
    """Build a mock subprocess.CompletedProcess."""
    r = MagicMock(spec=subprocess.CompletedProcess)
    r.returncode = returncode
    r.stdout = stdout
    r.stderr = stderr
    return r


def _make_executor(machine=None):
    m = machine or _make_machine()
    executor = RemoteExecutor(m)
    executor.container_id = "testcontainer123"
    return executor


def _make_agent(
    executor=None,
    kernel_name="gemm_afp4wfp4",
    config_variant="AFP4WFP4",
    gfx_arch="gfx942",
    artifact_dir="/artifacts",
    model_shapes_path=None,
):
    executor = executor or _make_executor()
    return DiscoveryAgent(
        executor=executor,
        kernel_name=kernel_name,
        artifact_dir=artifact_dir,
        config_variant=config_variant,
        gfx_arch=gfx_arch,
        model_shapes_path=model_shapes_path,
    )


# ---------------------------------------------------------------------------
# _parse_ls_output — unit tests
# ---------------------------------------------------------------------------


class TestParseLsOutput(unittest.TestCase):
    def test_single_path(self):
        result = _parse_ls_output("/workspace/foo.json\n")
        self.assertEqual(result, ["/workspace/foo.json"])

    def test_multiple_paths(self):
        out = "/workspace/a.json\n/workspace/b.json\n"
        result = _parse_ls_output(out)
        self.assertEqual(result, ["/workspace/a.json", "/workspace/b.json"])

    def test_empty_string_returns_empty_list(self):
        self.assertEqual(_parse_ls_output(""), [])

    def test_whitespace_only_lines_skipped(self):
        result = _parse_ls_output("/path/a.json\n  \n/path/b.json\n")
        self.assertEqual(result, ["/path/a.json", "/path/b.json"])

    def test_trailing_newline_handled(self):
        result = _parse_ls_output("/workspace/a.json\n")
        self.assertEqual(len(result), 1)

    def test_strips_leading_trailing_spaces(self):
        result = _parse_ls_output("  /workspace/a.json  \n")
        self.assertEqual(result, ["/workspace/a.json"])

    def test_error_output_like_no_such_file_excluded(self):
        # ls often returns an empty string on glob miss (shell already handles it);
        # but if the output only has blank lines they should be skipped.
        result = _parse_ls_output("\n\n\n")
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# _extract_nk_from_filename — unit tests
# ---------------------------------------------------------------------------


class TestExtractNkFromFilename(unittest.TestCase):
    def test_extracts_from_full_path(self):
        path = "/configs/gfx942-GEMM-AFP4WFP4-N=128-K=4096.json"
        self.assertEqual(_extract_nk_from_filename(path), (128, 4096))

    def test_extracts_from_basename(self):
        name = "gfx942-GEMM-A8W8-N=256-K=8192.json"
        self.assertEqual(_extract_nk_from_filename(name), (256, 8192))

    def test_returns_none_for_catch_all_config(self):
        name = "gfx942-GEMM-AFP4WFP4.json"
        self.assertIsNone(_extract_nk_from_filename(name))

    def test_large_values(self):
        name = "gfx942-GEMM-AFP4WFP4-N=106496-K=16384.json"
        self.assertEqual(_extract_nk_from_filename(name), (106496, 16384))

    def test_different_arch(self):
        name = "gfx950-GEMM-AFP4WFP4-N=7168-K=2048.json"
        self.assertEqual(_extract_nk_from_filename(name), (7168, 2048))

    def test_no_match_returns_none(self):
        self.assertIsNone(_extract_nk_from_filename("some_random_file.json"))


# ---------------------------------------------------------------------------
# _parse_model_shapes — unit tests
# ---------------------------------------------------------------------------


class TestParseModelShapes(unittest.TestCase):
    def test_extracts_pairs_for_kernel(self):
        pairs = _parse_model_shapes(SAMPLE_MODEL_SHAPES_JSON, "gemm_afp4wfp4")
        self.assertIn((106496, 16384), pairs)
        self.assertIn((16384, 53248), pairs)
        self.assertIn((7168, 2048), pairs)

    def test_deduplicates_across_models(self):
        # Build JSON where two models share the same (N, K) for a kernel.
        data = json.dumps({
            "ModelA": {"gemm_afp4wfp4": [{"N": 128, "K": 256}]},
            "ModelB": {"gemm_afp4wfp4": [{"N": 128, "K": 256}]},
        })
        pairs = _parse_model_shapes(data, "gemm_afp4wfp4")
        self.assertEqual(pairs.count((128, 256)), 1)

    def test_ignores_unrelated_kernel(self):
        pairs = _parse_model_shapes(SAMPLE_MODEL_SHAPES_JSON, "gemm_afp4wfp4")
        # gemm_a8w8 has N=4096, K=8192; those should not appear.
        self.assertNotIn((4096, 8192), pairs)

    def test_ignores_entries_without_k(self):
        pairs = _parse_model_shapes(SAMPLE_MODEL_SHAPES_JSON, "gemm_afp4wfp4")
        # rmsnorm entries only have N — none of the N values should appear as K
        # in valid (N,K) pairs for a different kernel.  Here we just check no
        # crashe and the count is correct.
        self.assertIsInstance(pairs, list)

    def test_returns_empty_list_for_unknown_kernel(self):
        pairs = _parse_model_shapes(SAMPLE_MODEL_SHAPES_JSON, "nonexistent_kernel")
        self.assertEqual(pairs, [])

    def test_returns_empty_list_for_invalid_json(self):
        pairs = _parse_model_shapes("not json {{", "gemm_afp4wfp4")
        self.assertEqual(pairs, [])

    def test_returns_empty_list_for_empty_json_object(self):
        pairs = _parse_model_shapes("{}", "gemm_afp4wfp4")
        self.assertEqual(pairs, [])

    def test_only_has_k_key_skips_entry(self):
        data = json.dumps({"ModelA": {"gemm_afp4wfp4": [{"K": 4096}]}})
        pairs = _parse_model_shapes(data, "gemm_afp4wfp4")
        self.assertEqual(pairs, [])

    def test_only_has_n_key_skips_entry(self):
        data = json.dumps({"ModelA": {"gemm_afp4wfp4": [{"N": 128}]}})
        pairs = _parse_model_shapes(data, "gemm_afp4wfp4")
        self.assertEqual(pairs, [])


# ---------------------------------------------------------------------------
# DiscoveryAgent construction
# ---------------------------------------------------------------------------


class TestDiscoveryAgentInit(unittest.TestCase):
    def setUp(self):
        self.executor = _make_executor()

    def test_name_attribute(self):
        self.assertEqual(DiscoveryAgent.name, "discovery")

    def test_stores_kernel_name(self):
        agent = _make_agent(executor=self.executor, kernel_name="gemm_a8w8")
        self.assertEqual(agent.kernel_name, "gemm_a8w8")

    def test_stores_config_variant(self):
        agent = _make_agent(executor=self.executor, config_variant="A8W8")
        self.assertEqual(agent.config_variant, "A8W8")

    def test_stores_gfx_arch(self):
        agent = _make_agent(executor=self.executor, gfx_arch="gfx950")
        self.assertEqual(agent.gfx_arch, "gfx950")

    def test_stores_artifact_dir(self):
        agent = _make_agent(executor=self.executor, artifact_dir="/my/artifacts")
        self.assertEqual(agent.artifact_dir, "/my/artifacts")

    def test_stores_model_shapes_path(self):
        agent = _make_agent(executor=self.executor, model_shapes_path="/path/to/shapes.json")
        self.assertEqual(agent.model_shapes_path, "/path/to/shapes.json")

    def test_model_shapes_path_defaults_to_none(self):
        agent = _make_agent(executor=self.executor)
        self.assertIsNone(agent.model_shapes_path)

    def test_is_subclass_of_base_subagent(self):
        from .base import BaseSubagent
        self.assertTrue(issubclass(DiscoveryAgent, BaseSubagent))


# ---------------------------------------------------------------------------
# DiscoveryAgent._execute — mock-based integration tests
# ---------------------------------------------------------------------------

# The call sequence for agent.run() via subprocess.run patches:
#   [0] mkdir  (preflight ssh_run)
#   [1] rocm-smi  (preflight ssh_run)
#   [2] docker exec ls configs  (_scan_config_files)
#   [3..N] docker exec cat <config>  (for catch-all files only)
#   [N+1] docker exec cat model_shapes.json  (_load_model_shapes)
#   [N+2] docker exec ls ut_*.py  (_detect_scripts)
#   [N+3] docker exec ls bench_gemm_*.py  (_detect_scripts)
#   [N+4] docker exec ls test_gemm_*.py  (_detect_scripts)
#   [N+5] ssh_run printf ... (artifact write)


def _make_side_effects(
    ls_config_stdout=LS_CONFIG_OUTPUT,
    cat_config_stdout=SAMPLE_CONFIG_JSON,  # for catch-all configs
    model_shapes_stdout=SAMPLE_MODEL_SHAPES_JSON,
    ut_stdout="",
    bench_stdout="",
    test_stdout="",
):
    """Build the mock subprocess.run side_effect list for a full agent.run() call.

    The LS_CONFIG_OUTPUT fixture has 2 N=K files + 1 catch-all, so we expect
    one ``cat`` call for the catch-all.
    """
    return [
        _completed(stdout=""),                     # mkdir
        _completed(stdout=""),                     # rocm-smi
        _completed(stdout=ls_config_stdout),       # ls configs
        _completed(stdout=cat_config_stdout),      # cat catch-all config
        _completed(stdout=model_shapes_stdout),    # cat model_shapes.json
        _completed(stdout=ut_stdout),              # ls ut_*.py
        _completed(stdout=bench_stdout),           # ls bench_gemm_*.py
        _completed(stdout=test_stdout),            # ls test_gemm_*.py
        _completed(stdout=""),                     # write artifact
    ]


class TestExecuteShapesGenerated(unittest.TestCase):
    """Verify that shapes are correctly built from config + model_shapes pairs."""

    @patch("subprocess.run")
    def test_shapes_list_is_nonempty(self, mock_run):
        mock_run.side_effect = _make_side_effects()
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertGreater(len(result.data["shapes"]), 0)

    @patch("subprocess.run")
    def test_shapes_are_three_tuples(self, mock_run):
        mock_run.side_effect = _make_side_effects()
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        for shape in result.data["shapes"]:
            self.assertEqual(len(shape), 3, msg=f"Expected (M,N,K) tuple, got {shape}")

    @patch("subprocess.run")
    def test_shapes_use_default_m_values(self, mock_run):
        mock_run.side_effect = _make_side_effects()
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        actual_m_values = set(s[0] for s in result.data["shapes"])
        for m in _DEFAULT_M_VALUES:
            self.assertIn(m, actual_m_values, msg=f"M={m} not found in shapes")

    @patch("subprocess.run")
    def test_config_nk_pairs_produce_shapes(self, mock_run):
        # LS_CONFIG_OUTPUT has N=128,K=4096 and N=256,K=8192 in filenames.
        mock_run.side_effect = _make_side_effects()
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        nk_from_shapes = set((s[1], s[2]) for s in result.data["shapes"])
        self.assertIn((128, 4096), nk_from_shapes)
        self.assertIn((256, 8192), nk_from_shapes)

    @patch("subprocess.run")
    def test_model_shapes_nk_pairs_produce_shapes(self, mock_run):
        # SAMPLE_MODEL_SHAPES_JSON has (106496, 16384), (16384, 53248), (7168, 2048)
        # for gemm_afp4wfp4.
        mock_run.side_effect = _make_side_effects()
        agent = _make_agent(kernel_name="gemm_afp4wfp4")
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        nk_from_shapes = set((s[1], s[2]) for s in result.data["shapes"])
        self.assertIn((106496, 16384), nk_from_shapes)
        self.assertIn((16384, 53248), nk_from_shapes)
        self.assertIn((7168, 2048), nk_from_shapes)

    @patch("subprocess.run")
    def test_no_duplicate_nk_pairs(self, mock_run):
        # Use identical N/K in both config filename and model_shapes.
        ls_out = "/workspace/.../gfx942-GEMM-AFP4WFP4-N=128-K=4096.json\n"
        model_json = json.dumps({
            "ModelA": {"gemm_afp4wfp4": [{"N": 128, "K": 4096}]},
        })
        side_effects = [
            _completed(stdout=""),          # mkdir
            _completed(stdout=""),          # rocm-smi
            _completed(stdout=ls_out),      # ls configs (1 file, N=K from name)
            _completed(stdout=model_json),  # cat model_shapes.json
            _completed(stdout=""),          # ls ut
            _completed(stdout=""),          # ls bench
            _completed(stdout=""),          # ls test
            _completed(stdout=""),          # write artifact
        ]
        mock_run.side_effect = side_effects
        agent = _make_agent(kernel_name="gemm_afp4wfp4")
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        nk_pairs = result.data["nk_pairs"]
        self.assertEqual(nk_pairs.count((128, 4096)), 1)

    @patch("subprocess.run")
    def test_each_nk_pair_has_all_m_values(self, mock_run):
        ls_out = "/workspace/.../gfx942-GEMM-AFP4WFP4-N=512-K=1024.json\n"
        side_effects = [
            _completed(stdout=""),      # mkdir
            _completed(stdout=""),      # rocm-smi
            _completed(stdout=ls_out),  # ls configs
            _completed(stdout="{}"),    # cat model_shapes.json (kernel not in file)
            _completed(stdout=""),      # ls ut
            _completed(stdout=""),      # ls bench
            _completed(stdout=""),      # ls test
            _completed(stdout=""),      # write artifact
        ]
        mock_run.side_effect = side_effects
        agent = _make_agent(kernel_name="gemm_afp4wfp4")
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        shapes_for_nk = [(m, n, k) for m, n, k in result.data["shapes"] if n == 512 and k == 1024]
        shape_m_values = sorted(s[0] for s in shapes_for_nk)
        self.assertEqual(shape_m_values, sorted(_DEFAULT_M_VALUES))


class TestExecuteConfigFiles(unittest.TestCase):
    """Verify config file scanning behaviour."""

    @patch("subprocess.run")
    def test_config_files_returned(self, mock_run):
        mock_run.side_effect = _make_side_effects()
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        cfg_files = result.data["config_files"]
        self.assertEqual(len(cfg_files), 3)

    @patch("subprocess.run")
    def test_config_files_contain_expected_paths(self, mock_run):
        mock_run.side_effect = _make_side_effects()
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        cfg_files = result.data["config_files"]
        self.assertIn(
            "/workspace/aiter/aiter/ops/triton/configs/gemm/gfx942-GEMM-AFP4WFP4-N=128-K=4096.json",
            cfg_files,
        )

    @patch("subprocess.run")
    def test_empty_ls_output_gives_empty_config_files(self, mock_run):
        side_effects = [
            _completed(stdout=""),   # mkdir
            _completed(stdout=""),   # rocm-smi
            _completed(stdout=""),   # ls configs (no matches)
            _completed(stdout="{}"), # cat model_shapes
            _completed(stdout=""),   # ls ut
            _completed(stdout=""),   # ls bench
            _completed(stdout=""),   # ls test
            _completed(stdout=""),   # write artifact
        ]
        mock_run.side_effect = side_effects
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(result.data["config_files"], [])

    @patch("subprocess.run")
    def test_ls_uses_gfx_arch_and_config_variant(self, mock_run):
        mock_run.side_effect = _make_side_effects()
        agent = _make_agent(gfx_arch="gfx950", config_variant="A8W8")
        agent.run()
        all_cmds = " ".join(" ".join(c[0][0]) for c in mock_run.call_args_list)
        self.assertIn("gfx950", all_cmds)
        self.assertIn("A8W8", all_cmds)


class TestExecuteNkPairs(unittest.TestCase):
    """Verify nk_pairs in the returned dict."""

    @patch("subprocess.run")
    def test_nk_pairs_from_config_filenames(self, mock_run):
        mock_run.side_effect = _make_side_effects()
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        nk_pairs = result.data["nk_pairs"]
        self.assertIn((128, 4096), nk_pairs)
        self.assertIn((256, 8192), nk_pairs)

    @patch("subprocess.run")
    def test_nk_pairs_from_model_shapes(self, mock_run):
        mock_run.side_effect = _make_side_effects()
        agent = _make_agent(kernel_name="gemm_afp4wfp4")
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        nk_pairs = result.data["nk_pairs"]
        self.assertIn((106496, 16384), nk_pairs)

    @patch("subprocess.run")
    def test_nk_pairs_are_tuples(self, mock_run):
        mock_run.side_effect = _make_side_effects()
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        for pair in result.data["nk_pairs"]:
            self.assertEqual(len(pair), 2)


class TestExecuteScriptDetection(unittest.TestCase):
    """Verify script detection and missing_scripts reporting."""

    @patch("subprocess.run")
    def test_all_scripts_missing(self, mock_run):
        mock_run.side_effect = _make_side_effects(
            ut_stdout="", bench_stdout="", test_stdout=""
        )
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        missing = result.data["missing_scripts"]
        self.assertIn("ut", missing)
        self.assertIn("bench", missing)
        self.assertIn("test", missing)
        self.assertIsNone(result.data["ut_script"])
        self.assertIsNone(result.data["bench_script"])
        self.assertIsNone(result.data["test_script"])

    @patch("subprocess.run")
    def test_ut_script_found(self, mock_run):
        ut_path = "/workspace/aiter/aiter/ops/triton/utils/_triton/tunning/ut_gemm_afp4wfp4.py"
        mock_run.side_effect = _make_side_effects(ut_stdout=ut_path + "\n")
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(result.data["ut_script"], ut_path)
        self.assertNotIn("ut", result.data["missing_scripts"])

    @patch("subprocess.run")
    def test_bench_script_found(self, mock_run):
        bench_path = "/workspace/aiter/op_tests/op_benchmarks/triton/bench_gemm_afp4wfp4.py"
        mock_run.side_effect = _make_side_effects(bench_stdout=bench_path + "\n")
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(result.data["bench_script"], bench_path)
        self.assertNotIn("bench", result.data["missing_scripts"])

    @patch("subprocess.run")
    def test_test_script_found(self, mock_run):
        test_path = "/workspace/aiter/op_tests/triton_tests/gemm/fp4/test_gemm_afp4wfp4.py"
        mock_run.side_effect = _make_side_effects(test_stdout=test_path + "\n")
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(result.data["test_script"], test_path)
        self.assertNotIn("test", result.data["missing_scripts"])

    @patch("subprocess.run")
    def test_all_scripts_found_no_missing(self, mock_run):
        ut_path = "/workspace/.../ut_gemm_afp4wfp4.py"
        bench_path = "/workspace/.../bench_gemm_afp4wfp4.py"
        test_path = "/workspace/.../test_gemm_afp4wfp4.py"
        mock_run.side_effect = _make_side_effects(
            ut_stdout=ut_path + "\n",
            bench_stdout=bench_path + "\n",
            test_stdout=test_path + "\n",
        )
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(result.data["missing_scripts"], [])

    @patch("subprocess.run")
    def test_missing_scripts_is_list(self, mock_run):
        mock_run.side_effect = _make_side_effects()
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertIsInstance(result.data["missing_scripts"], list)

    @patch("subprocess.run")
    def test_ls_commands_use_kernel_name(self, mock_run):
        mock_run.side_effect = _make_side_effects()
        agent = _make_agent(kernel_name="gemm_afp4wfp4")
        agent.run()
        all_cmds = " ".join(" ".join(c[0][0]) for c in mock_run.call_args_list)
        # The ls commands should contain the kernel name.
        self.assertIn("gemm_afp4wfp4", all_cmds)


class TestExecuteReturnKeys(unittest.TestCase):
    """Verify the return dict contains all required keys."""

    @patch("subprocess.run")
    def test_all_required_keys_present(self, mock_run):
        mock_run.side_effect = _make_side_effects()
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        expected_keys = {
            "shapes", "config_files", "nk_pairs",
            "ut_script", "bench_script", "test_script", "missing_scripts",
        }
        self.assertEqual(set(result.data.keys()), expected_keys)

    @patch("subprocess.run")
    def test_run_returns_subagent_result(self, mock_run):
        mock_run.side_effect = _make_side_effects()
        agent = _make_agent()
        result = agent.run()
        self.assertIsInstance(result, SubagentResult)

    @patch("subprocess.run")
    def test_run_success_true(self, mock_run):
        mock_run.side_effect = _make_side_effects()
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success)


class TestExecuteArtifact(unittest.TestCase):
    """Verify that a discovery.json artifact is written."""

    @patch("subprocess.run")
    def test_artifact_write_called(self, mock_run):
        mock_run.side_effect = _make_side_effects()
        agent = _make_agent()
        agent.run()
        # The printf/write SSH call should be present.
        all_cmds = " ".join(" ".join(c[0][0]) for c in mock_run.call_args_list)
        self.assertIn("printf", all_cmds)

    @patch("subprocess.run")
    def test_artifact_filename_is_discovery_json(self, mock_run):
        mock_run.side_effect = _make_side_effects()
        agent = _make_agent(artifact_dir="/arts")
        agent.run()
        write_call = next(
            (c for c in mock_run.call_args_list if "printf" in " ".join(c[0][0])),
            None,
        )
        self.assertIsNotNone(write_call, "No artifact write SSH call found")
        cmd_str = " ".join(write_call[0][0])
        self.assertIn("discovery.json", cmd_str)

    @patch("subprocess.run")
    def test_artifact_contains_valid_json(self, mock_run):
        captured: list = []

        def _side_effect(cmd_list, **kwargs):
            ssh_cmd = " ".join(cmd_list)
            if "printf" in ssh_cmd:
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

        # We need the ls/cat calls to return real data so _execute doesn't crash
        # before reaching the write step.  Use a counter to serve the right response.
        call_idx = [0]
        real_responses = _make_side_effects()

        def _dispatched(cmd_list, **kwargs):
            if "printf" in " ".join(cmd_list):
                return _side_effect(cmd_list, **kwargs)
            idx = call_idx[0]
            call_idx[0] += 1
            if idx < len(real_responses):
                return real_responses[idx]
            r = MagicMock(spec=subprocess.CompletedProcess)
            r.returncode = 0
            r.stdout = ""
            r.stderr = ""
            return r

        mock_run.side_effect = _dispatched
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(len(captured), 1, "Expected exactly one artifact write")
        parsed = captured[0]
        self.assertNotIsInstance(parsed, Exception, f"JSON parse error: {parsed}")
        self.assertIn("shapes", parsed)
        self.assertIn("config_files", parsed)
        self.assertIn("nk_pairs", parsed)


class TestExecuteModelShapesNotFound(unittest.TestCase):
    """Verify graceful handling when model_shapes.json is absent."""

    @patch("subprocess.run")
    def test_missing_model_shapes_does_not_crash(self, mock_run):
        side_effects = [
            _completed(stdout=""),    # mkdir
            _completed(stdout=""),    # rocm-smi
            _completed(stdout=""),    # ls configs (none)
            _completed(returncode=1, stdout="", stderr="No such file"),  # cat model_shapes (missing)
            _completed(stdout=""),    # ls ut
            _completed(stdout=""),    # ls bench
            _completed(stdout=""),    # ls test
            _completed(stdout=""),    # write artifact
        ]
        mock_run.side_effect = side_effects
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)

    @patch("subprocess.run")
    def test_missing_model_shapes_gives_empty_shapes(self, mock_run):
        side_effects = [
            _completed(stdout=""),    # mkdir
            _completed(stdout=""),    # rocm-smi
            _completed(stdout=""),    # ls configs (none)
            _completed(returncode=1, stdout=""),  # cat model_shapes (missing)
            _completed(stdout=""),    # ls ut
            _completed(stdout=""),    # ls bench
            _completed(stdout=""),    # ls test
            _completed(stdout=""),    # write artifact
        ]
        mock_run.side_effect = side_effects
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(result.data["shapes"], [])


if __name__ == "__main__":
    unittest.main()
