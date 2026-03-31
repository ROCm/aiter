"""Tests for ConfigGeneratorAgent and its helper utilities.

All remote I/O is mocked via :func:`unittest.mock.patch` on
:func:`subprocess.run` so no SSH connection or Docker container is required.

Coverage
--------
- ``_parse_log_filenames``: extract unique (N, K) pairs from ls output.
- ``_most_common_nk``: identify the (N, K) pair appearing most often.
- ``ConfigGeneratorAgent._execute``:
  - Correct (N, K) pairs are extracted from log filenames.
  - ``view-screen.py`` is called once per unique (N, K) pair.
  - ``M_LEQ_16`` is renamed when :attr:`m_leq_16_rename` is set.
  - Config files are copied to the config store.
  - Default fallback config is created/updated.
  - Return dict has the expected keys.
"""

import json
import subprocess
import unittest
from unittest.mock import MagicMock, call, patch

from ..remote import RemoteExecutor
from ..types import MachineInfo
from .config_generator_agent import (
    ConfigGeneratorAgent,
    _most_common_nk,
    _parse_log_filenames,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TUNNING_DIR = "/workspace/aiter/aiter/ops/triton/utils/_triton/tunning"
_TUNING_LOGS_DIR = _TUNNING_DIR
_CONFIG_DIR = "/workspace/aiter/aiter/ops/triton/configs/gemm"
_GFX = "gfx950"
_VARIANT = "A8W8"
_UT_SCRIPT = "ut_a8w8_gemm.py"
_ARTIFACT_DIR = "/artifacts/config_gen"

# Simulated ls output for two distinct (N, K) pairs across multiple M values.
_LS_OUTPUT_A8W8 = "\n".join([
    f"{_TUNING_LOGS_DIR}/screen-{_UT_SCRIPT}-8-8192-8192.log",
    f"{_TUNING_LOGS_DIR}/screen-{_UT_SCRIPT}-16-8192-8192.log",
    f"{_TUNING_LOGS_DIR}/screen-{_UT_SCRIPT}-32-8192-8192.log",
    f"{_TUNING_LOGS_DIR}/screen-{_UT_SCRIPT}-8-1280-8192.log",
    f"{_TUNING_LOGS_DIR}/screen-{_UT_SCRIPT}-16-1280-8192.log",
])

# Minimal valid JSON produced by view-screen.py for one shape.
_SAMPLE_CONFIG_8192_8192 = json.dumps({
    "M_LEQ_8": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 512,
                "num_warps": 4, "num_stages": 2},
    "M_LEQ_16": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 256,
                 "num_warps": 4, "num_stages": 2},
    "any": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
            "num_warps": 8, "num_stages": 2},
}, indent=2)

_SAMPLE_CONFIG_1280_8192 = json.dumps({
    "M_LEQ_8": {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 256,
                "num_warps": 4, "num_stages": 2},
    "M_LEQ_16": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128,
                 "num_warps": 4, "num_stages": 2},
    "any": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
            "num_warps": 4, "num_stages": 2},
}, indent=2)

# ls output listing the generated JSON files.
_LS_GENERATED = "\n".join([
    f"{_TUNNING_DIR}/{_GFX}-GEMM-{_VARIANT}-N=8192-K=8192.json",
    f"{_TUNNING_DIR}/{_GFX}-GEMM-{_VARIANT}-N=1280-K=8192.json",
])


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


def _make_executor() -> RemoteExecutor:
    executor = RemoteExecutor(_make_machine())
    executor.container_id = "testcontainer"
    return executor


def _make_agent(
    executor=None,
    m_leq_16_rename: str | None = None,
    variant: str = _VARIANT,
    ut_script: str = _UT_SCRIPT,
) -> ConfigGeneratorAgent:
    executor = executor or _make_executor()
    return ConfigGeneratorAgent(
        executor=executor,
        kernel_name="gemm_a8w8",
        artifact_dir=_ARTIFACT_DIR,
        tuning_logs_dir=_TUNING_LOGS_DIR,
        config_dir=_CONFIG_DIR,
        kernel_variant=variant,
        ut_script=ut_script,
        gfx_arch=_GFX,
        tunning_dir=_TUNNING_DIR,
        m_leq_16_rename=m_leq_16_rename,
    )


# ---------------------------------------------------------------------------
# Unit tests for _parse_log_filenames
# ---------------------------------------------------------------------------


class TestParseLogFilenames(unittest.TestCase):
    """Verify that (N, K) pairs are correctly extracted from ls output."""

    def test_extracts_n_and_k(self):
        pairs = _parse_log_filenames(_LS_OUTPUT_A8W8)
        self.assertIn((8192, 8192), pairs)
        self.assertIn((1280, 8192), pairs)

    def test_deduplicates_pairs(self):
        """Multiple M values for the same (N, K) yield only one entry."""
        pairs = _parse_log_filenames(_LS_OUTPUT_A8W8)
        # (8192, 8192) appears 3 times; (1280, 8192) appears 2 times.
        self.assertEqual(pairs.count((8192, 8192)), 1)
        self.assertEqual(pairs.count((1280, 8192)), 1)

    def test_total_unique_pairs(self):
        pairs = _parse_log_filenames(_LS_OUTPUT_A8W8)
        self.assertEqual(len(pairs), 2)

    def test_empty_output_returns_empty_list(self):
        pairs = _parse_log_filenames("")
        self.assertEqual(pairs, [])

    def test_unrelated_filenames_ignored(self):
        ls = "res-ut_a8w8_gemm.py-128-8192-8192_kernel_trace.csv\nsome-other-file.log"
        pairs = _parse_log_filenames(ls)
        self.assertEqual(pairs, [])

    def test_preserves_order_of_first_appearance(self):
        """(N, K) pairs must appear in first-seen order."""
        pairs = _parse_log_filenames(_LS_OUTPUT_A8W8)
        self.assertEqual(pairs[0], (8192, 8192))
        self.assertEqual(pairs[1], (1280, 8192))

    def test_paths_with_directory_prefix_parsed(self):
        ls = (
            "/some/dir/screen-ut_a8w8_gemm.py-8-2048-4096.log\n"
            "/some/dir/screen-ut_a8w8_gemm.py-16-2048-4096.log\n"
        )
        pairs = _parse_log_filenames(ls)
        self.assertEqual(pairs, [(2048, 4096)])

    def test_log_with_suffix_ignored(self):
        """Files like screen-*.log.aot_bug must not be parsed."""
        ls = (
            f"{_TUNING_LOGS_DIR}/screen-ut_a8w8_gemm.py-16-1280-8192.log.aot_bug\n"
            f"{_TUNING_LOGS_DIR}/screen-ut_a8w8_gemm.py-16-1280-8192.log\n"
        )
        pairs = _parse_log_filenames(ls)
        self.assertEqual(pairs, [(1280, 8192)])


# ---------------------------------------------------------------------------
# Unit tests for _most_common_nk
# ---------------------------------------------------------------------------


class TestMostCommonNk(unittest.TestCase):
    """Verify identification of the (N, K) pair with the most log entries."""

    def test_most_common_returns_correct_pair(self):
        # (8192, 8192) appears 3 times; (1280, 8192) appears 2 times.
        pair = _most_common_nk(_LS_OUTPUT_A8W8)
        self.assertEqual(pair, (8192, 8192))

    def test_empty_returns_none(self):
        self.assertIsNone(_most_common_nk(""))

    def test_single_pair(self):
        ls = f"{_TUNING_LOGS_DIR}/screen-{_UT_SCRIPT}-8-512-4096.log\n"
        pair = _most_common_nk(ls)
        self.assertEqual(pair, (512, 4096))

    def test_equal_counts_returns_a_valid_pair(self):
        """When two pairs tie, the function must return one of them (not None)."""
        ls = (
            f"{_TUNING_LOGS_DIR}/screen-{_UT_SCRIPT}-8-111-222.log\n"
            f"{_TUNING_LOGS_DIR}/screen-{_UT_SCRIPT}-8-333-444.log\n"
        )
        pair = _most_common_nk(ls)
        self.assertIn(pair, [(111, 222), (333, 444)])


# ---------------------------------------------------------------------------
# ConfigGeneratorAgent._execute — happy-path tests
# ---------------------------------------------------------------------------


class TestExecuteHappyPath(unittest.TestCase):
    """End-to-end _execute tests using fully mocked subprocess.run."""

    def _make_side_effects(
        self,
        ls_output: str = _LS_OUTPUT_A8W8,
        generated_ls: str = _LS_GENERATED,
        config_8192: str = _SAMPLE_CONFIG_8192_8192,
        config_1280: str = _SAMPLE_CONFIG_1280_8192,
        fallback_exists: bool = False,
    ):
        """Build the ordered list of subprocess.run return values.

        Call sequence (with no M_LEQ_16 rename):
          1.  mkdir   (preflight: ssh_run)
          2.  rocm-smi (preflight: ssh_run)
          3.  ls screen-*.log (docker_exec)
          4.  view-screen.py N=8192 K=8192 (docker_exec)
          5.  view-screen.py N=1280 K=8192 (docker_exec)
          6.  cp … (docker_exec)
          7.  ls generated *.json (docker_exec)
          8.  cat N=8192-K=8192 config (docker_exec — most common pair)
          9.  cat fallback config (docker_exec — may 404)
          10. write fallback (docker_exec)
        """
        fallback_cat = (
            _completed(stdout=json.dumps({"any": {}}))
            if fallback_exists
            else _completed(returncode=1, stderr="No such file")
        )
        return [
            _completed(),                               # 1. mkdir
            _completed(),                               # 2. rocm-smi
            _completed(stdout=ls_output),               # 3. ls screen-*.log
            _completed(),                               # 4. view-screen N=8192 K=8192
            _completed(),                               # 5. view-screen N=1280 K=8192
            _completed(),                               # 6. cp
            _completed(stdout=generated_ls),            # 7. ls generated
            _completed(stdout=config_8192),             # 8. cat most-common config
            fallback_cat,                               # 9. cat fallback
            _completed(),                               # 10. write fallback
        ]

    @patch("subprocess.run")
    def test_returns_correct_configs_generated_count(self, mock_run):
        mock_run.side_effect = self._make_side_effects()
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(result.data["configs_generated"], 2)

    @patch("subprocess.run")
    def test_returns_config_files_list(self, mock_run):
        mock_run.side_effect = self._make_side_effects()
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertIn("config_files", result.data)
        self.assertIsInstance(result.data["config_files"], list)

    @patch("subprocess.run")
    def test_config_files_contain_expected_basenames(self, mock_run):
        mock_run.side_effect = self._make_side_effects()
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        files = result.data["config_files"]
        self.assertTrue(
            any("N=8192" in f and "K=8192" in f for f in files),
            f"Expected N=8192-K=8192 in {files}",
        )

    @patch("subprocess.run")
    def test_default_fallback_updated_true(self, mock_run):
        mock_run.side_effect = self._make_side_effects()
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertTrue(result.data["default_fallback_updated"])

    @patch("subprocess.run")
    def test_result_has_all_required_keys(self, mock_run):
        mock_run.side_effect = self._make_side_effects()
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        for key in ("configs_generated", "config_files", "default_fallback_updated"):
            self.assertIn(key, result.data, f"Missing key: {key!r}")


# ---------------------------------------------------------------------------
# ConfigGeneratorAgent — view-screen.py invocation tests
# ---------------------------------------------------------------------------


class TestViewScreenInvocations(unittest.TestCase):
    """Ensure view-screen.py is called the right number of times with the right args."""

    def _base_side_effects(self):
        return [
            _completed(),                               # mkdir
            _completed(),                               # rocm-smi
            _completed(stdout=_LS_OUTPUT_A8W8),         # ls screen-*.log
            _completed(),                               # view-screen N=8192 K=8192
            _completed(),                               # view-screen N=1280 K=8192
            _completed(),                               # cp
            _completed(stdout=_LS_GENERATED),           # ls generated
            _completed(stdout=_SAMPLE_CONFIG_8192_8192),# cat most-common
            _completed(returncode=1),                   # cat fallback (not found)
            _completed(),                               # write fallback
        ]

    @patch("subprocess.run")
    def test_view_screen_called_for_each_nk_pair(self, mock_run):
        mock_run.side_effect = self._base_side_effects()
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)

        view_screen_calls = [
            c for c in mock_run.call_args_list
            if "view-screen.py" in " ".join(c[0][0])
        ]
        self.assertEqual(len(view_screen_calls), 2)

    @patch("subprocess.run")
    def test_view_screen_called_with_correct_n_k(self, mock_run):
        mock_run.side_effect = self._base_side_effects()
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)

        view_screen_calls = [
            " ".join(c[0][0])
            for c in mock_run.call_args_list
            if "view-screen.py" in " ".join(c[0][0])
        ]
        # One call must contain N=8192 and K=8192.
        self.assertTrue(
            any("8192" in cmd for cmd in view_screen_calls),
            f"Expected 8192 in one view-screen call; got: {view_screen_calls}",
        )
        # One call must contain N=1280.
        self.assertTrue(
            any("1280" in cmd for cmd in view_screen_calls),
            f"Expected 1280 in one view-screen call; got: {view_screen_calls}",
        )

    @patch("subprocess.run")
    def test_view_screen_called_with_ut_script(self, mock_run):
        mock_run.side_effect = self._base_side_effects()
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)

        view_screen_calls = [
            " ".join(c[0][0])
            for c in mock_run.call_args_list
            if "view-screen.py" in " ".join(c[0][0])
        ]
        for cmd in view_screen_calls:
            self.assertIn(_UT_SCRIPT, cmd)

    @patch("subprocess.run")
    def test_view_screen_not_called_when_no_logs(self, mock_run):
        mock_run.side_effect = [
            _completed(),                  # mkdir
            _completed(),                  # rocm-smi
            _completed(returncode=1, stderr="No such file"),  # ls — nothing
        ]
        agent = _make_agent()
        result = agent.run()
        self.assertFalse(result.success)

        view_screen_calls = [
            c for c in mock_run.call_args_list
            if "view-screen.py" in " ".join(c[0][0])
        ]
        self.assertEqual(len(view_screen_calls), 0)


# ---------------------------------------------------------------------------
# ConfigGeneratorAgent — M_LEQ_16 rename tests
# ---------------------------------------------------------------------------


class TestMLeq16Rename(unittest.TestCase):
    """Verify that the M_LEQ_16 key rename is applied when requested."""

    def _side_effects_with_rename(self):
        """Call sequence when m_leq_16_rename is set (two pairs).

        Extra calls per pair: cat + printf (rename step).
        """
        return [
            _completed(),                                   # mkdir
            _completed(),                                   # rocm-smi
            _completed(stdout=_LS_OUTPUT_A8W8),             # ls screen-*.log
            # --- pair (8192, 8192) ---
            _completed(),                                   # view-screen
            _completed(stdout=_SAMPLE_CONFIG_8192_8192),    # cat json (rename)
            _completed(),                                   # printf write-back
            # --- pair (1280, 8192) ---
            _completed(),                                   # view-screen
            _completed(stdout=_SAMPLE_CONFIG_1280_8192),    # cat json (rename)
            _completed(),                                   # printf write-back
            # --- copy + fallback ---
            _completed(),                                   # cp
            _completed(stdout=_LS_GENERATED),               # ls generated
            _completed(stdout=_SAMPLE_CONFIG_8192_8192),    # cat most-common
            _completed(returncode=1),                       # cat fallback
            _completed(),                                   # write fallback
        ]

    @patch("subprocess.run")
    def test_cat_called_for_each_pair_when_rename_set(self, mock_run):
        mock_run.side_effect = self._side_effects_with_rename()
        agent = _make_agent(m_leq_16_rename="M_LEQ_31")
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)

        cat_calls = [
            c for c in mock_run.call_args_list
            if "cat" in " ".join(c[0][0]) and ".json" in " ".join(c[0][0])
        ]
        # Two cat calls for the rename (one per pair) + one for most-common fallback.
        self.assertGreaterEqual(len(cat_calls), 2)

    @patch("subprocess.run")
    def test_printf_called_for_each_pair_when_rename_set(self, mock_run):
        mock_run.side_effect = self._side_effects_with_rename()
        agent = _make_agent(m_leq_16_rename="M_LEQ_31")
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)

        printf_calls = [
            c for c in mock_run.call_args_list
            if "printf" in " ".join(c[0][0])
        ]
        # At least two printf calls (one per pair rename) + one for fallback write.
        self.assertGreaterEqual(len(printf_calls), 2)

    @patch("subprocess.run")
    def test_rename_key_present_in_write_command(self, mock_run):
        """The printf write-back must embed the new key name, not M_LEQ_16."""
        mock_run.side_effect = self._side_effects_with_rename()
        agent = _make_agent(m_leq_16_rename="M_LEQ_31")
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)

        printf_calls = [
            " ".join(c[0][0])
            for c in mock_run.call_args_list
            if "printf" in " ".join(c[0][0])
        ]
        # At least one printf call should contain the new name.
        self.assertTrue(
            any("M_LEQ_31" in cmd for cmd in printf_calls),
            f"Expected M_LEQ_31 in at least one printf call; got:\n"
            + "\n".join(printf_calls),
        )

    @patch("subprocess.run")
    def test_no_cat_for_rename_when_not_set(self, mock_run):
        """When m_leq_16_rename is None, no cat/printf for rename should occur."""
        side_effects = [
            _completed(),
            _completed(),
            _completed(stdout=_LS_OUTPUT_A8W8),
            _completed(),
            _completed(),
            _completed(),
            _completed(stdout=_LS_GENERATED),
            _completed(stdout=_SAMPLE_CONFIG_8192_8192),
            _completed(returncode=1),
            _completed(),
        ]
        mock_run.side_effect = side_effects
        agent = _make_agent(m_leq_16_rename=None)
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)

        # Count cat calls that reference .json files (rename-step cats).
        all_cmds = [" ".join(c[0][0]) for c in mock_run.call_args_list]
        rename_cats = [
            cmd for cmd in all_cmds
            if "cat" in cmd and ".json" in cmd and "N=" in cmd and "fallback" not in cmd
        ]
        # No rename cat calls expected (the fallback cat is allowed).
        # The only cat for .json should be the fallback lookup.
        json_cats_excluding_fallback = [
            cmd for cmd in all_cmds
            if "cat" in cmd and f"GEMM-{_VARIANT}-N=" in cmd
        ]
        # In the no-rename path the most-common config is read for fallback purposes.
        self.assertLessEqual(len(json_cats_excluding_fallback), 1)


# ---------------------------------------------------------------------------
# ConfigGeneratorAgent — copy-to-config-dir tests
# ---------------------------------------------------------------------------


class TestCopyToConfigDir(unittest.TestCase):
    """Verify that generated config files are copied to the config store."""

    def _base_side_effects(self):
        return [
            _completed(),
            _completed(),
            _completed(stdout=_LS_OUTPUT_A8W8),
            _completed(),
            _completed(),
            _completed(),
            _completed(stdout=_LS_GENERATED),
            _completed(stdout=_SAMPLE_CONFIG_8192_8192),
            _completed(returncode=1),
            _completed(),
        ]

    @patch("subprocess.run")
    def test_cp_command_issued(self, mock_run):
        mock_run.side_effect = self._base_side_effects()
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)

        # docker_exec wraps the command as: docker exec <id> bash -c 'cp ...'
        # so the joined ssh args contain "bash -c 'cp " rather than " cp ".
        cp_calls = [
            c for c in mock_run.call_args_list
            if "cp " in " ".join(c[0][0])
        ]
        self.assertGreaterEqual(len(cp_calls), 1, "Expected at least one cp call")

    @patch("subprocess.run")
    def test_cp_targets_config_dir(self, mock_run):
        mock_run.side_effect = self._base_side_effects()
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)

        # docker_exec wraps the command as: docker exec <id> bash -c 'cp ...'
        # so the joined ssh args contain "bash -c 'cp " rather than " cp ".
        cp_calls = [
            " ".join(c[0][0])
            for c in mock_run.call_args_list
            if "cp " in " ".join(c[0][0])
        ]
        self.assertTrue(
            any(_CONFIG_DIR in cmd for cmd in cp_calls),
            f"Expected config dir {_CONFIG_DIR!r} in cp call; got: {cp_calls}",
        )

    @patch("subprocess.run")
    def test_cp_uses_variant_glob(self, mock_run):
        mock_run.side_effect = self._base_side_effects()
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)

        # docker_exec wraps the command as: docker exec <id> bash -c 'cp ...'
        # so the joined ssh args contain "bash -c 'cp " rather than " cp ".
        cp_calls = [
            " ".join(c[0][0])
            for c in mock_run.call_args_list
            if "cp " in " ".join(c[0][0])
        ]
        self.assertTrue(
            any(_VARIANT in cmd for cmd in cp_calls),
            f"Expected variant {_VARIANT!r} in cp call glob; got: {cp_calls}",
        )


# ---------------------------------------------------------------------------
# ConfigGeneratorAgent — default fallback tests
# ---------------------------------------------------------------------------


class TestDefaultFallback(unittest.TestCase):
    """Verify the default fallback config is created and updated correctly."""

    def _side_effects_with_existing_fallback(self):
        existing_fallback = json.dumps({
            "any": {"BLOCK_SIZE_M": 64, "num_warps": 4, "num_stages": 2}
        })
        return [
            _completed(),
            _completed(),
            _completed(stdout=_LS_OUTPUT_A8W8),
            _completed(),
            _completed(),
            _completed(),
            _completed(stdout=_LS_GENERATED),
            _completed(stdout=_SAMPLE_CONFIG_8192_8192),
            _completed(stdout=existing_fallback),   # cat fallback — exists
            _completed(),                           # write fallback
        ]

    @patch("subprocess.run")
    def test_fallback_updated_when_any_bucket_exists(self, mock_run):
        mock_run.side_effect = self._side_effects_with_existing_fallback()
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertTrue(result.data["default_fallback_updated"])

    @patch("subprocess.run")
    def test_fallback_updated_true_on_success(self, mock_run):
        side_effects = [
            _completed(),
            _completed(),
            _completed(stdout=_LS_OUTPUT_A8W8),
            _completed(),
            _completed(),
            _completed(),
            _completed(stdout=_LS_GENERATED),
            _completed(stdout=_SAMPLE_CONFIG_8192_8192),
            _completed(returncode=1),
            _completed(),
        ]
        mock_run.side_effect = side_effects
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertTrue(result.data["default_fallback_updated"])

    @patch("subprocess.run")
    def test_fallback_false_when_most_common_config_missing(self, mock_run):
        """When the most-common config cat fails, fallback should be False."""
        side_effects = [
            _completed(),
            _completed(),
            _completed(stdout=_LS_OUTPUT_A8W8),
            _completed(),
            _completed(),
            _completed(),
            _completed(stdout=_LS_GENERATED),
            _completed(returncode=1, stderr="No such file"),  # cat most-common fails
        ]
        mock_run.side_effect = side_effects
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertFalse(result.data["default_fallback_updated"])

    @patch("subprocess.run")
    def test_fallback_write_command_contains_any_key(self, mock_run):
        """The printf command for the fallback must embed the 'any' key."""
        side_effects = [
            _completed(),
            _completed(),
            _completed(stdout=_LS_OUTPUT_A8W8),
            _completed(),
            _completed(),
            _completed(),
            _completed(stdout=_LS_GENERATED),
            _completed(stdout=_SAMPLE_CONFIG_8192_8192),
            _completed(returncode=1),
            _completed(),
        ]
        mock_run.side_effect = side_effects
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)

        # Find the printf call for the fallback (last printf).
        printf_calls = [
            " ".join(c[0][0])
            for c in mock_run.call_args_list
            if "printf" in " ".join(c[0][0])
        ]
        self.assertTrue(
            any('"any"' in cmd or "any" in cmd for cmd in printf_calls),
            f"Expected 'any' key in fallback printf; got:\n" + "\n".join(printf_calls),
        )


# ---------------------------------------------------------------------------
# ConfigGeneratorAgent — constructor and attribute tests
# ---------------------------------------------------------------------------


class TestConfigGeneratorAgentInit(unittest.TestCase):
    """Ensure the constructor stores all arguments correctly."""

    def setUp(self):
        self.agent = _make_agent()

    def test_name_attribute(self):
        self.assertEqual(ConfigGeneratorAgent.name, "config_generator")

    def test_stores_tuning_logs_dir(self):
        self.assertEqual(self.agent.tuning_logs_dir, _TUNING_LOGS_DIR)

    def test_stores_config_dir(self):
        self.assertEqual(self.agent.config_dir, _CONFIG_DIR)

    def test_stores_kernel_variant(self):
        self.assertEqual(self.agent.kernel_variant, _VARIANT)

    def test_stores_ut_script(self):
        self.assertEqual(self.agent.ut_script, _UT_SCRIPT)

    def test_stores_gfx_arch(self):
        self.assertEqual(self.agent.gfx_arch, _GFX)

    def test_stores_tunning_dir(self):
        self.assertEqual(self.agent.tunning_dir, _TUNNING_DIR)

    def test_m_leq_16_rename_default_none(self):
        self.assertIsNone(self.agent.m_leq_16_rename)

    def test_m_leq_16_rename_stored(self):
        agent = _make_agent(m_leq_16_rename="M_LEQ_31")
        self.assertEqual(agent.m_leq_16_rename, "M_LEQ_31")

    def test_tunning_dir_defaults_to_tuning_logs_dir_when_not_given(self):
        agent = ConfigGeneratorAgent(
            executor=_make_executor(),
            kernel_name="gemm",
            artifact_dir=_ARTIFACT_DIR,
            tuning_logs_dir=_TUNING_LOGS_DIR,
            config_dir=_CONFIG_DIR,
            kernel_variant=_VARIANT,
            ut_script=_UT_SCRIPT,
            gfx_arch=_GFX,
            # tunning_dir not provided — should default to tuning_logs_dir
        )
        self.assertEqual(agent.tunning_dir, _TUNING_LOGS_DIR)

    def test_is_subclass_of_base_subagent(self):
        from .base import BaseSubagent
        self.assertTrue(issubclass(ConfigGeneratorAgent, BaseSubagent))


# ---------------------------------------------------------------------------
# ConfigGeneratorAgent — single N,K pair
# ---------------------------------------------------------------------------


class TestSingleNKPair(unittest.TestCase):
    """Verify correct behaviour when only one (N, K) pair is present."""

    @patch("subprocess.run")
    def test_single_pair_view_screen_called_once(self, mock_run):
        single_ls = f"{_TUNING_LOGS_DIR}/screen-{_UT_SCRIPT}-8-2048-4096.log\n"
        single_config = json.dumps({
            "M_LEQ_8": {"BLOCK_SIZE_M": 8, "num_warps": 4, "num_stages": 2},
            "any": {"BLOCK_SIZE_M": 128, "num_warps": 8, "num_stages": 2},
        }, indent=2)
        generated = f"{_TUNNING_DIR}/{_GFX}-GEMM-{_VARIANT}-N=2048-K=4096.json\n"

        mock_run.side_effect = [
            _completed(),
            _completed(),
            _completed(stdout=single_ls),
            _completed(),                           # view-screen
            _completed(),                           # cp
            _completed(stdout=generated),           # ls generated
            _completed(stdout=single_config),       # cat most-common
            _completed(returncode=1),               # cat fallback
            _completed(),                           # write fallback
        ]
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)
        self.assertEqual(result.data["configs_generated"], 1)

        view_screen_calls = [
            c for c in mock_run.call_args_list
            if "view-screen.py" in " ".join(c[0][0])
        ]
        self.assertEqual(len(view_screen_calls), 1)

    @patch("subprocess.run")
    def test_single_pair_correct_n_k_in_view_screen(self, mock_run):
        single_ls = f"{_TUNING_LOGS_DIR}/screen-{_UT_SCRIPT}-8-2048-4096.log\n"
        single_config = json.dumps({
            "any": {"BLOCK_SIZE_M": 128, "num_warps": 8, "num_stages": 2},
        }, indent=2)
        generated = f"{_TUNNING_DIR}/{_GFX}-GEMM-{_VARIANT}-N=2048-K=4096.json\n"

        mock_run.side_effect = [
            _completed(),
            _completed(),
            _completed(stdout=single_ls),
            _completed(),
            _completed(),
            _completed(stdout=generated),
            _completed(stdout=single_config),
            _completed(returncode=1),
            _completed(),
        ]
        agent = _make_agent()
        result = agent.run()
        self.assertTrue(result.success, msg=result.error)

        view_screen_calls = [
            " ".join(c[0][0])
            for c in mock_run.call_args_list
            if "view-screen.py" in " ".join(c[0][0])
        ]
        self.assertEqual(len(view_screen_calls), 1)
        cmd = view_screen_calls[0]
        self.assertIn("2048", cmd)
        self.assertIn("4096", cmd)


if __name__ == "__main__":
    unittest.main()
