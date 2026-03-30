"""Tests for RegressionFixerAgent.

All tests use ``tmp_path``-style temporary directories via Python's
``tempfile.TemporaryDirectory`` so they are self-contained and leave no
artefacts on disk.

The tests cover:
- Strategy determination (noise-skip, restore, promote, escalate)
- restore_bucket modifying only the target bucket
- promote_to_suffixed creating a new suffixed file
- The critical invariant: the fallback config is never modified
"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock

from .regression_fixer_agent import FixStrategy, RegressionFixerAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A realistic config structure used across multiple tests.
_BASE_CONFIG = {
    "M_LEQ_8": {
        "BLOCK_SIZE_M": 4,
        "BLOCK_SIZE_N": 16,
        "BLOCK_SIZE_K": 256,
        "num_warps": 4,
        "num_stages": 3,
    },
    "M_LEQ_16": {
        "BLOCK_SIZE_M": 8,
        "BLOCK_SIZE_N": 16,
        "BLOCK_SIZE_K": 128,
        "num_warps": 1,
        "num_stages": 3,
    },
    "M_LEQ_32": {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 128,
        "num_warps": 2,
        "num_stages": 3,
    },
    "any": {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "num_warps": 8,
        "num_stages": 3,
    },
}

_MODIFIED_BUCKET = {
    "BLOCK_SIZE_M": 32,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 512,
    "num_warps": 8,
    "num_stages": 4,
}


def _write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _make_executor() -> MagicMock:
    """Return a mock RemoteExecutor (never actually called in these tests)."""
    return MagicMock()


def _make_agent(
    tmp_dir: str,
    regressions=None,
    threshold: float = 0.05,
    max_iterations: int = 3,
) -> RegressionFixerAgent:
    """Construct a RegressionFixerAgent rooted in *tmp_dir*."""
    config_dir = os.path.join(tmp_dir, "current")
    old_config_dir = os.path.join(tmp_dir, "old")
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(old_config_dir, exist_ok=True)

    return RegressionFixerAgent(
        executor=_make_executor(),
        kernel_name="gemm_test",
        artifact_dir="/tmp/artifacts",
        regressions=regressions or [],
        config_dir=config_dir,
        old_config_dir=old_config_dir,
        threshold=threshold,
        max_iterations=max_iterations,
    )


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------


class TestDetermineStrategyNoise(unittest.TestCase):
    """Identical configs → NOISE_SKIP."""

    def test_determine_strategy_same_config_is_noise(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)

            # Write identical configs in both current and old dirs.
            fname = "gfx950-GEMM-A16W16-N=128-K=4096.json"
            current_path = os.path.join(agent.config_dir, fname)
            old_path = os.path.join(agent.old_config_dir, fname)
            _write_json(current_path, _BASE_CONFIG)
            _write_json(old_path, _BASE_CONFIG)

            regression = {
                "m": 16,
                "n": 128,
                "k": 4096,
                "delta": 0.02,  # below threshold
                "current_config_file": current_path,
                "bucket": "M_LEQ_16",
            }
            strategy = agent.determine_fix_strategy(regression)
            self.assertEqual(strategy, FixStrategy.NOISE_SKIP)


class TestDetermineStrategySuffixedRestore(unittest.TestCase):
    """Suffixed file with a changed bucket → RESTORE_BUCKET."""

    def test_determine_strategy_suffixed_file_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)

            fname = "gfx950-GEMM-A16W16-N=128-K=4096.json"
            current_path = os.path.join(agent.config_dir, fname)
            old_path = os.path.join(agent.old_config_dir, fname)

            # Current config has a modified M_LEQ_16 bucket.
            current_config = dict(_BASE_CONFIG)
            current_config["M_LEQ_16"] = _MODIFIED_BUCKET
            _write_json(current_path, current_config)
            _write_json(old_path, _BASE_CONFIG)

            regression = {
                "m": 16,
                "n": 128,
                "k": 4096,
                "delta": 0.08,  # above threshold, below 2×threshold
                "current_config_file": current_path,
                "bucket": "M_LEQ_16",
            }
            strategy = agent.determine_fix_strategy(regression)
            self.assertEqual(strategy, FixStrategy.RESTORE_BUCKET)


class TestDetermineStrategyFallbackPromote(unittest.TestCase):
    """Fallback file (no N=/K= suffix) → PROMOTE_TO_SUFFIXED."""

    def test_determine_strategy_fallback_file_promote(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)

            # Use a filename WITHOUT an N=/K= suffix — this is the fallback.
            fname = "gfx950-GEMM-A16W16.json"
            current_path = os.path.join(agent.config_dir, fname)
            old_path = os.path.join(agent.old_config_dir, fname)

            current_config = dict(_BASE_CONFIG)
            current_config["M_LEQ_16"] = _MODIFIED_BUCKET
            _write_json(current_path, current_config)
            _write_json(old_path, _BASE_CONFIG)

            regression = {
                "m": 16,
                "n": 128,
                "k": 4096,
                "delta": 0.08,
                "current_config_file": current_path,
                "bucket": "M_LEQ_16",
            }
            strategy = agent.determine_fix_strategy(regression)
            self.assertEqual(strategy, FixStrategy.PROMOTE_TO_SUFFIXED)


class TestDetermineStrategyLargeDeltaEscalate(unittest.TestCase):
    """delta > 2× threshold → ESCALATE regardless of file type."""

    def test_determine_strategy_large_delta_escalate(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp, threshold=0.05)

            # Even a suffixed file must escalate when delta is very large.
            fname = "gfx950-GEMM-A16W16-N=128-K=4096.json"
            current_path = os.path.join(agent.config_dir, fname)
            old_path = os.path.join(agent.old_config_dir, fname)
            _write_json(current_path, _BASE_CONFIG)
            _write_json(old_path, _BASE_CONFIG)

            regression = {
                "m": 16,
                "n": 128,
                "k": 4096,
                "delta": 0.50,  # >> 2 × 0.05
                "current_config_file": current_path,
                "bucket": "M_LEQ_16",
            }
            strategy = agent.determine_fix_strategy(regression)
            self.assertEqual(strategy, FixStrategy.ESCALATE)

    def test_escalate_boundary_exactly_2x(self):
        """delta == 2× threshold is NOT escalated (boundary is exclusive)."""
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp, threshold=0.05)

            fname = "gfx950-GEMM-A16W16-N=128-K=4096.json"
            current_path = os.path.join(agent.config_dir, fname)
            old_path = os.path.join(agent.old_config_dir, fname)

            current_config = dict(_BASE_CONFIG)
            current_config["M_LEQ_16"] = _MODIFIED_BUCKET
            _write_json(current_path, current_config)
            _write_json(old_path, _BASE_CONFIG)

            regression = {
                "m": 16,
                "n": 128,
                "k": 4096,
                "delta": 0.10,  # exactly 2 × 0.05 — NOT > 2×threshold
                "current_config_file": current_path,
                "bucket": "M_LEQ_16",
            }
            strategy = agent.determine_fix_strategy(regression)
            self.assertNotEqual(strategy, FixStrategy.ESCALATE)


# ---------------------------------------------------------------------------
# restore_bucket tests
# ---------------------------------------------------------------------------


class TestRestoreBucket(unittest.TestCase):
    """restore_bucket replaces only the target bucket."""

    def test_restore_bucket_only_changes_target_bucket(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)

            # Current config has a modified M_LEQ_16 bucket.
            current_config = {
                "M_LEQ_8": dict(_BASE_CONFIG["M_LEQ_8"]),
                "M_LEQ_16": dict(_MODIFIED_BUCKET),  # regressed
                "M_LEQ_32": dict(_BASE_CONFIG["M_LEQ_32"]),
                "any": dict(_BASE_CONFIG["any"]),
            }
            fname = "gfx950-GEMM-A16W16-N=128-K=4096.json"
            current_path = os.path.join(tmp, fname)
            old_path = os.path.join(tmp, "old_" + fname)
            _write_json(current_path, current_config)
            _write_json(old_path, _BASE_CONFIG)

            agent.restore_bucket(current_path, "M_LEQ_16", old_path)

            result = _read_json(current_path)

            # Target bucket must be restored.
            self.assertEqual(result["M_LEQ_16"], _BASE_CONFIG["M_LEQ_16"])

            # All other buckets must remain as they were in current_config.
            self.assertEqual(result["M_LEQ_8"], _BASE_CONFIG["M_LEQ_8"])
            self.assertEqual(result["M_LEQ_32"], _BASE_CONFIG["M_LEQ_32"])
            self.assertEqual(result["any"], _BASE_CONFIG["any"])

    def test_restore_bucket_handles_mleq_boundary_difference(self):
        """M_LEQ_31 in current vs M_LEQ_16 in old — fuzzy match must work."""
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)

            current_config = {
                "M_LEQ_31": dict(_MODIFIED_BUCKET),
                "any": dict(_BASE_CONFIG["any"]),
            }
            old_config = {
                "M_LEQ_16": dict(_BASE_CONFIG["M_LEQ_16"]),
                "any": dict(_BASE_CONFIG["any"]),
            }
            current_path = os.path.join(tmp, "current.json")
            old_path = os.path.join(tmp, "old.json")
            _write_json(current_path, current_config)
            _write_json(old_path, old_config)

            # Should not raise even though the bucket names differ.
            agent.restore_bucket(current_path, "M_LEQ_31", old_path)

            result = _read_json(current_path)
            self.assertEqual(result["M_LEQ_31"], _BASE_CONFIG["M_LEQ_16"])

    def test_restore_bucket_raises_if_old_bucket_missing(self):
        """SubagentError is raised when the old config has no matching bucket."""
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)

            current_config = {"M_LEQ_512": dict(_MODIFIED_BUCKET)}
            old_config = {"any": dict(_BASE_CONFIG["any"])}
            current_path = os.path.join(tmp, "current.json")
            old_path = os.path.join(tmp, "old.json")
            _write_json(current_path, current_config)
            _write_json(old_path, old_config)

            from .base import SubagentError

            with self.assertRaises(SubagentError):
                agent.restore_bucket(current_path, "M_LEQ_512", old_path)


# ---------------------------------------------------------------------------
# promote_to_suffixed tests
# ---------------------------------------------------------------------------


class TestPromoteToSuffixed(unittest.TestCase):
    """promote_to_suffixed creates a new suffixed file."""

    def test_promote_creates_new_suffixed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)

            fallback_fname = "gfx950-GEMM-A16W16.json"
            fallback_path = os.path.join(agent.config_dir, fallback_fname)
            old_fallback_path = os.path.join(agent.old_config_dir, fallback_fname)

            # Current fallback has a regressed M_LEQ_16.
            current_fallback = dict(_BASE_CONFIG)
            current_fallback["M_LEQ_16"] = _MODIFIED_BUCKET
            _write_json(fallback_path, current_fallback)
            _write_json(old_fallback_path, _BASE_CONFIG)

            new_path = agent.promote_to_suffixed(
                n=128,
                k=4096,
                fallback_path=fallback_path,
                old_fallback_path=old_fallback_path,
                bucket_name="M_LEQ_16",
            )

            # The new file must exist.
            self.assertTrue(os.path.isfile(new_path))

            # The new filename must contain the N= and K= suffixes.
            self.assertIn("N=128", os.path.basename(new_path))
            self.assertIn("K=4096", os.path.basename(new_path))

            # The regressed bucket in the new file must come from old_fallback.
            new_config = _read_json(new_path)
            self.assertEqual(new_config["M_LEQ_16"], _BASE_CONFIG["M_LEQ_16"])

    def test_promote_suffixed_file_contains_other_buckets(self):
        """The new suffixed file must include all non-regressed buckets from the fallback."""
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)

            fallback_fname = "gfx950-GEMM-A16W16.json"
            fallback_path = os.path.join(agent.config_dir, fallback_fname)
            old_fallback_path = os.path.join(agent.old_config_dir, fallback_fname)

            current_fallback = dict(_BASE_CONFIG)
            current_fallback["M_LEQ_16"] = _MODIFIED_BUCKET
            _write_json(fallback_path, current_fallback)
            _write_json(old_fallback_path, _BASE_CONFIG)

            new_path = agent.promote_to_suffixed(128, 4096, fallback_path, old_fallback_path, "M_LEQ_16")
            new_config = _read_json(new_path)

            # All other buckets must come from the current fallback.
            self.assertEqual(new_config["M_LEQ_8"], _BASE_CONFIG["M_LEQ_8"])
            self.assertEqual(new_config["M_LEQ_32"], _BASE_CONFIG["M_LEQ_32"])
            self.assertEqual(new_config["any"], _BASE_CONFIG["any"])


# ---------------------------------------------------------------------------
# Critical invariant: fallback is never modified
# ---------------------------------------------------------------------------


class TestNeverModifiesFallback(unittest.TestCase):
    """Verify the fallback config file is untouched after a promotion."""

    def test_never_modifies_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)

            fallback_fname = "gfx950-GEMM-A16W16.json"
            fallback_path = os.path.join(agent.config_dir, fallback_fname)
            old_fallback_path = os.path.join(agent.old_config_dir, fallback_fname)

            # Record the fallback content before the operation.
            current_fallback = dict(_BASE_CONFIG)
            current_fallback["M_LEQ_16"] = _MODIFIED_BUCKET
            _write_json(fallback_path, current_fallback)
            _write_json(old_fallback_path, _BASE_CONFIG)

            fallback_mtime_before = os.path.getmtime(fallback_path)
            fallback_content_before = _read_json(fallback_path)

            # Run the promotion via _execute (full pipeline path).
            agent.regressions = [
                {
                    "m": 16,
                    "n": 128,
                    "k": 4096,
                    "delta": 0.08,  # above threshold, below 2× threshold
                    "current_config_file": fallback_path,
                    "bucket": "M_LEQ_16",
                }
            ]
            result = agent._execute()

            # Promotion must have happened.
            self.assertEqual(result["promoted"], 1)

            # Fallback file content must be byte-for-byte identical.
            fallback_content_after = _read_json(fallback_path)
            self.assertEqual(fallback_content_before, fallback_content_after)

            # Fallback mtime must not have changed (file was not even opened
            # for writing).
            fallback_mtime_after = os.path.getmtime(fallback_path)
            self.assertEqual(fallback_mtime_before, fallback_mtime_after)

    def test_fallback_unchanged_when_strategy_is_promote(self):
        """Double-check via strategy path that PROMOTE_TO_SUFFIXED is chosen."""
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)

            fallback_fname = "gfx950-GEMM-A16W16.json"
            fallback_path = os.path.join(agent.config_dir, fallback_fname)
            old_fallback_path = os.path.join(agent.old_config_dir, fallback_fname)

            current_fallback = dict(_BASE_CONFIG)
            current_fallback["M_LEQ_16"] = _MODIFIED_BUCKET
            _write_json(fallback_path, current_fallback)
            _write_json(old_fallback_path, _BASE_CONFIG)

            regression = {
                "m": 16,
                "n": 128,
                "k": 4096,
                "delta": 0.08,
                "current_config_file": fallback_path,
                "bucket": "M_LEQ_16",
            }
            strategy = agent.determine_fix_strategy(regression)
            self.assertEqual(strategy, FixStrategy.PROMOTE_TO_SUFFIXED)


# ---------------------------------------------------------------------------
# _execute integration tests
# ---------------------------------------------------------------------------


class TestExecuteIntegration(unittest.TestCase):
    """End-to-end tests exercising _execute with mixed regressions."""

    def test_execute_returns_correct_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp, threshold=0.05)

            config_dir = agent.config_dir
            old_config_dir = agent.old_config_dir

            # Suffixed file — will be RESTORED.
            suffixed_fname = "gfx950-GEMM-A16W16-N=128-K=4096.json"
            suffixed_current = os.path.join(config_dir, suffixed_fname)
            suffixed_old = os.path.join(old_config_dir, suffixed_fname)
            modified = dict(_BASE_CONFIG)
            modified["M_LEQ_16"] = _MODIFIED_BUCKET
            _write_json(suffixed_current, modified)
            _write_json(suffixed_old, _BASE_CONFIG)

            # Fallback file — will be PROMOTED.
            fallback_fname = "gfx950-GEMM-A16W16.json"
            fallback_current = os.path.join(config_dir, fallback_fname)
            fallback_old = os.path.join(old_config_dir, fallback_fname)
            fallback_modified = dict(_BASE_CONFIG)
            fallback_modified["M_LEQ_32"] = _MODIFIED_BUCKET
            _write_json(fallback_current, fallback_modified)
            _write_json(fallback_old, _BASE_CONFIG)

            # Noise regression — identical configs.
            noise_fname = "gfx950-GEMM-A16W16-N=256-K=7168.json"
            noise_current = os.path.join(config_dir, noise_fname)
            noise_old = os.path.join(old_config_dir, noise_fname)
            _write_json(noise_current, _BASE_CONFIG)
            _write_json(noise_old, _BASE_CONFIG)

            # Escalation — huge delta.
            escalate_fname = "gfx950-GEMM-A16W16-N=512-K=8192.json"
            escalate_current = os.path.join(config_dir, escalate_fname)
            escalate_old = os.path.join(old_config_dir, escalate_fname)
            _write_json(escalate_current, _BASE_CONFIG)
            _write_json(escalate_old, _BASE_CONFIG)

            agent.regressions = [
                {
                    "m": 16, "n": 128, "k": 4096, "delta": 0.08,
                    "current_config_file": suffixed_current, "bucket": "M_LEQ_16",
                },
                {
                    "m": 32, "n": 128, "k": 4096, "delta": 0.06,
                    "current_config_file": fallback_current, "bucket": "M_LEQ_32",
                },
                {
                    "m": 8, "n": 256, "k": 7168, "delta": 0.02,
                    "current_config_file": noise_current, "bucket": "M_LEQ_8",
                },
                {
                    "m": 8, "n": 512, "k": 8192, "delta": 0.99,  # >> 2×0.05
                    "current_config_file": escalate_current, "bucket": "M_LEQ_8",
                },
            ]

            result = agent._execute()

            self.assertEqual(result["fixed"], 1)
            self.assertEqual(result["promoted"], 1)
            self.assertEqual(result["skipped"], 1)
            self.assertEqual(result["escalated"], 1)
            self.assertEqual(len(result["escalations"]), 1)

    def test_execute_no_regressions(self):
        """Empty regressions list returns all-zero counts."""
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)
            result = agent._execute()
            self.assertEqual(result["fixed"], 0)
            self.assertEqual(result["promoted"], 0)
            self.assertEqual(result["skipped"], 0)
            self.assertEqual(result["escalated"], 0)
            self.assertEqual(result["escalations"], [])


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------


class TestFixStrategyEdgeCases(unittest.TestCase):
    def test_noise_skip_both_files_missing(self):
        """If neither config file exists, we fall through to file-type rules."""
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp, threshold=0.05)

            # Neither file exists on disk.
            missing_path = os.path.join(agent.config_dir, "gfx950-GEMM-A16W16-N=64-K=128.json")

            regression = {
                "m": 8, "n": 64, "k": 128, "delta": 0.03,
                "current_config_file": missing_path,
                "bucket": "M_LEQ_8",
            }
            # Should not raise; file-type rules apply (suffixed → RESTORE).
            strategy = agent.determine_fix_strategy(regression)
            self.assertEqual(strategy, FixStrategy.RESTORE_BUCKET)

    def test_promote_returns_path_with_correct_name(self):
        """The returned path from promote_to_suffixed has the right filename."""
        with tempfile.TemporaryDirectory() as tmp:
            agent = _make_agent(tmp)

            fallback_fname = "gfx950-GEMM-A8W8.json"
            fallback_path = os.path.join(agent.config_dir, fallback_fname)
            old_fallback_path = os.path.join(agent.old_config_dir, fallback_fname)

            _write_json(fallback_path, _BASE_CONFIG)
            _write_json(old_fallback_path, _BASE_CONFIG)

            new_path = agent.promote_to_suffixed(
                n=1280, k=8192,
                fallback_path=fallback_path,
                old_fallback_path=old_fallback_path,
                bucket_name="any",
            )

            self.assertIn("N=1280", os.path.basename(new_path))
            self.assertIn("K=8192", os.path.basename(new_path))
            self.assertTrue(new_path.endswith(".json"))

    def test_name_attribute(self):
        """Agent name must be 'regression_fixer'."""
        self.assertEqual(RegressionFixerAgent.name, "regression_fixer")


if __name__ == "__main__":
    unittest.main()
