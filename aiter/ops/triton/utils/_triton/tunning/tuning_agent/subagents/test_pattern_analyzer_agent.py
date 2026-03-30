"""Tests for PatternAnalyzerAgent and parse_screen_log.

Test coverage
-------------
- parse_screen_log: correct extraction of configs and timings
- Top-3 selection per shape (de-duplication + ranking)
- M-range grouping
- >20% threshold filtering
- Historical data receives 0.25x weight vs scout 1.0x weight
- Sanity check: widens narrowed space if it falls below 25% of broad space
- Integration: _execute returns correct keys and values
"""

import json
import subprocess
import unittest
from unittest.mock import MagicMock

from ..types import MachineInfo
from ..remote import RemoteExecutor
from .base import SubagentResult
from .pattern_analyzer_agent import (
    BROAD_SPACE,
    M_RANGES,
    MIN_NARROWING_RATIO,
    TUNE_PARAMS,
    WINNING_THRESHOLD,
    PatternAnalyzerAgent,
    _assign_m_range,
    _narrowing_ratio,
    _top_k_configs,
    _widen_to_minimum,
    parse_screen_log,
)


# ---------------------------------------------------------------------------
# Screen-log fixtures
# ---------------------------------------------------------------------------

# A minimal valid screen log with two screencases for shape M16_N64_K256.
SCREEN_LOG_TWO_CONFIGS = """\
screencase 16 64 256 1 4 2 3 16 0 1
    kernel_M16_N64_K256_bm16_bn64_bk256
 42.5
screencase 32 64 256 1 4 3 3 16 0 1
    kernel_M16_N64_K256_bm32_bn64_bk256
 55.0
"""

# A log with three shapes; each shape has configs with distinct timings.
SCREEN_LOG_MULTI_SHAPE = """\
screencase 16 64 256 1 4 2 3 16 0 1
    kernel_M16_N64_K256_bm16
 10.0
screencase 32 64 256 1 4 2 3 16 0 1
    kernel_M16_N64_K256_bm32
 20.0
screencase 64 64 256 1 4 2 3 16 0 1
    kernel_M16_N64_K256_bm64
 30.0
screencase 64 128 512 1 8 3 4 16 0 2
    kernel_M512_N64_K512_bm64
 15.0
screencase 128 128 512 1 8 3 4 16 0 2
    kernel_M512_N64_K512_bm128
 25.0
"""

# A log where one screencase has no timing line (malformed).
SCREEN_LOG_MISSING_TIMING = """\
screencase 16 64 256 1 4 2 3 16 0 1
    kernel_M8_N64_K256_bm16
screencase 32 64 256 1 4 2 3 16 0 1
    kernel_M8_N64_K256_bm32
 77.7
"""

# A log with duplicate configs (same BM/BN/BK/stages/nonkdim/ksplit,
# different timing) — deduplication should keep the fastest.
SCREEN_LOG_DUPLICATES = """\
screencase 16 64 256 1 4 2 3 16 0 1
    kernel_M8_N64_K256_bm16_run1
 100.0
screencase 16 64 256 1 4 2 3 16 0 1
    kernel_M8_N64_K256_bm16_run2
 50.0
screencase 32 64 256 1 4 2 3 16 0 1
    kernel_M8_N64_K256_bm32
 80.0
"""

# A log with more than 3 configs for one shape — only top-3 should be kept.
SCREEN_LOG_MANY_CONFIGS = """\
screencase 16 64 256 1 4 2 3 16 0 1
    kernel_M16_N64_K256_bm16
 10.0
screencase 32 64 256 1 4 2 3 16 0 1
    kernel_M16_N64_K256_bm32
 20.0
screencase 64 64 256 1 4 2 3 16 0 1
    kernel_M16_N64_K256_bm64
 30.0
screencase 128 64 256 1 4 2 3 16 0 1
    kernel_M16_N64_K256_bm128
 40.0
screencase 256 64 256 1 4 2 3 16 0 1
    kernel_M16_N64_K256_bm256
 50.0
"""

# A log with a large-M shape (M=1024, falls in M_GEQ_1024).
SCREEN_LOG_LARGE_M = """\
screencase 64 128 512 1 8 2 4 16 0 1
    kernel_M1024_N128_K512_bm64
 25.0
screencase 128 128 512 1 8 3 4 16 0 2
    kernel_M1024_N128_K512_bm128
 18.0
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _completed(returncode=0, stdout="", stderr=""):
    r = MagicMock(spec=subprocess.CompletedProcess)
    r.returncode = returncode
    r.stdout = stdout
    r.stderr = stderr
    return r


def _make_executor() -> MagicMock:
    """Return a mock RemoteExecutor."""
    return MagicMock()


def _make_agent(
    executor=None,
    scout_results_dir="/scout",
    history_dir=None,
    artifact_dir="/artifacts",
    kernel_name="test_kernel",
) -> PatternAnalyzerAgent:
    executor = executor or _make_executor()
    return PatternAnalyzerAgent(
        executor=executor,
        kernel_name=kernel_name,
        artifact_dir=artifact_dir,
        scout_results_dir=scout_results_dir,
        history_dir=history_dir,
    )


# ---------------------------------------------------------------------------
# parse_screen_log — unit tests
# ---------------------------------------------------------------------------


class TestParseScreenLogBasic(unittest.TestCase):
    """Basic parsing of a well-formed screen log."""

    def test_returns_list(self):
        result = parse_screen_log(SCREEN_LOG_TWO_CONFIGS)
        self.assertIsInstance(result, list)

    def test_two_configs_parsed(self):
        result = parse_screen_log(SCREEN_LOG_TWO_CONFIGS)
        self.assertEqual(len(result), 2)

    def test_first_config_bm(self):
        result = parse_screen_log(SCREEN_LOG_TWO_CONFIGS)
        self.assertEqual(result[0]["BM"], 16)

    def test_first_config_bn(self):
        result = parse_screen_log(SCREEN_LOG_TWO_CONFIGS)
        self.assertEqual(result[0]["BN"], 64)

    def test_first_config_bk(self):
        result = parse_screen_log(SCREEN_LOG_TWO_CONFIGS)
        self.assertEqual(result[0]["BK"], 256)

    def test_first_config_stages(self):
        result = parse_screen_log(SCREEN_LOG_TWO_CONFIGS)
        self.assertEqual(result[0]["stages"], 2)

    def test_first_config_nonkdim(self):
        result = parse_screen_log(SCREEN_LOG_TWO_CONFIGS)
        self.assertEqual(result[0]["nonkdim"], 16)

    def test_first_config_ksplit(self):
        result = parse_screen_log(SCREEN_LOG_TWO_CONFIGS)
        self.assertEqual(result[0]["ksplit"], 1)

    def test_first_config_timing(self):
        result = parse_screen_log(SCREEN_LOG_TWO_CONFIGS)
        self.assertAlmostEqual(result[0]["timing_us"], 42.5)

    def test_second_config_timing(self):
        result = parse_screen_log(SCREEN_LOG_TWO_CONFIGS)
        self.assertAlmostEqual(result[1]["timing_us"], 55.0)

    def test_second_config_bm(self):
        result = parse_screen_log(SCREEN_LOG_TWO_CONFIGS)
        self.assertEqual(result[1]["BM"], 32)

    def test_kernel_name_extracted(self):
        result = parse_screen_log(SCREEN_LOG_TWO_CONFIGS)
        self.assertIn("kernel_M16_N64_K256_bm16_bn64_bk256", result[0]["kernel_name"])

    def test_shape_m_extracted(self):
        result = parse_screen_log(SCREEN_LOG_TWO_CONFIGS)
        self.assertEqual(result[0]["M"], 16)

    def test_shape_n_extracted(self):
        result = parse_screen_log(SCREEN_LOG_TWO_CONFIGS)
        self.assertEqual(result[0]["N"], 64)

    def test_shape_k_extracted(self):
        result = parse_screen_log(SCREEN_LOG_TWO_CONFIGS)
        self.assertEqual(result[0]["K"], 256)

    def test_all_required_keys_present(self):
        required = {"BM", "BN", "BK", "GSM", "warps", "stages", "waves",
                    "nonkdim", "cache", "ksplit", "kernel_name", "timing_us"}
        result = parse_screen_log(SCREEN_LOG_TWO_CONFIGS)
        for rec in result:
            for key in required:
                self.assertIn(key, rec, f"Missing key {key!r} in record {rec}")


class TestParseScreenLogMissingTiming(unittest.TestCase):
    """A screencase block without a timing line is skipped."""

    def test_malformed_block_skipped(self):
        result = parse_screen_log(SCREEN_LOG_MISSING_TIMING)
        # Only the second screencase (which has a timing line) should be parsed.
        self.assertEqual(len(result), 1)

    def test_valid_config_timing_correct(self):
        result = parse_screen_log(SCREEN_LOG_MISSING_TIMING)
        self.assertAlmostEqual(result[0]["timing_us"], 77.7)


class TestParseScreenLogEmpty(unittest.TestCase):
    """Empty content returns an empty list."""

    def test_empty_string(self):
        self.assertEqual(parse_screen_log(""), [])

    def test_no_screencase_lines(self):
        content = "# just a comment\nsome random text\n"
        self.assertEqual(parse_screen_log(content), [])


class TestParseScreenLogMultiShape(unittest.TestCase):
    """Multiple shapes in the same log."""

    def test_five_configs_parsed(self):
        result = parse_screen_log(SCREEN_LOG_MULTI_SHAPE)
        self.assertEqual(len(result), 5)

    def test_shapes_have_different_m(self):
        result = parse_screen_log(SCREEN_LOG_MULTI_SHAPE)
        m_values = {r["M"] for r in result if r["M"] is not None}
        # The log contains M16 and M512 shapes.
        self.assertIn(16, m_values)
        self.assertIn(512, m_values)


# ---------------------------------------------------------------------------
# Top-3 selection — unit tests
# ---------------------------------------------------------------------------


class TestTopKConfigs(unittest.TestCase):
    """_top_k_configs returns the correct top entries."""

    def _make_records(self, timing_list):
        """Create minimal records with distinct BM values and given timings."""
        return [
            {
                "BM": 16 * (i + 1), "BN": 64, "BK": 256,
                "stages": 2, "nonkdim": 16, "ksplit": 1,
                "timing_us": t,
            }
            for i, t in enumerate(timing_list)
        ]

    def test_returns_three_from_five(self):
        records = self._make_records([50.0, 10.0, 30.0, 20.0, 40.0])
        top = _top_k_configs(records, k=3)
        self.assertEqual(len(top), 3)

    def test_top_three_are_fastest(self):
        records = self._make_records([50.0, 10.0, 30.0, 20.0, 40.0])
        top = _top_k_configs(records, k=3)
        timings = sorted(r["timing_us"] for r in top)
        self.assertEqual(timings, [10.0, 20.0, 30.0])

    def test_returns_all_when_fewer_than_k(self):
        records = self._make_records([5.0, 3.0])
        top = _top_k_configs(records, k=3)
        self.assertEqual(len(top), 2)

    def test_deduplication_keeps_faster(self):
        # Two records with the same config key — only the 50.0 one should survive.
        duplicate_records = [
            {
                "BM": 16, "BN": 64, "BK": 256,
                "stages": 2, "nonkdim": 16, "ksplit": 1,
                "timing_us": 100.0,
            },
            {
                "BM": 16, "BN": 64, "BK": 256,
                "stages": 2, "nonkdim": 16, "ksplit": 1,
                "timing_us": 50.0,
            },
            {
                "BM": 32, "BN": 64, "BK": 256,
                "stages": 2, "nonkdim": 16, "ksplit": 1,
                "timing_us": 80.0,
            },
        ]
        top = _top_k_configs(duplicate_records, k=3)
        timings = sorted(r["timing_us"] for r in top)
        self.assertEqual(timings, [50.0, 80.0])
        self.assertNotIn(100.0, [r["timing_us"] for r in top])

    def test_empty_input_returns_empty(self):
        self.assertEqual(_top_k_configs([], k=3), [])

    def test_k_equals_one(self):
        records = self._make_records([30.0, 10.0, 20.0])
        top = _top_k_configs(records, k=1)
        self.assertEqual(len(top), 1)
        self.assertAlmostEqual(top[0]["timing_us"], 10.0)


# ---------------------------------------------------------------------------
# M-range grouping — unit tests
# ---------------------------------------------------------------------------


class TestAssignMRange(unittest.TestCase):
    """_assign_m_range maps M values to the correct label."""

    def test_m_1_is_small(self):
        self.assertEqual(_assign_m_range(1), "M_LEQ_16")

    def test_m_16_is_small(self):
        self.assertEqual(_assign_m_range(16), "M_LEQ_16")

    def test_m_32_is_medium(self):
        self.assertEqual(_assign_m_range(32), "M_32_64")

    def test_m_64_is_medium(self):
        self.assertEqual(_assign_m_range(64), "M_32_64")

    def test_m_128_is_large(self):
        self.assertEqual(_assign_m_range(128), "M_128_512")

    def test_m_512_is_large(self):
        self.assertEqual(_assign_m_range(512), "M_128_512")

    def test_m_1024_is_xlarge(self):
        self.assertEqual(_assign_m_range(1024), "M_GEQ_1024")

    def test_m_4096_is_xlarge(self):
        self.assertEqual(_assign_m_range(4096), "M_GEQ_1024")

    def test_none_returns_none(self):
        self.assertIsNone(_assign_m_range(None))

    def test_m_17_returns_none(self):
        # 17 is between small (<=16) and medium (32-64) — not covered.
        self.assertIsNone(_assign_m_range(17))

    def test_m_65_returns_none(self):
        # 65 is between medium (32-64) and large (128-512) — not covered.
        self.assertIsNone(_assign_m_range(65))


# ---------------------------------------------------------------------------
# >20% threshold filtering — unit tests
# ---------------------------------------------------------------------------


class TestThresholdFiltering(unittest.TestCase):
    """Values appearing in <=20% of winning configs are dropped."""

    def _make_records_bm_values(self, bm_list, timing_start=10.0):
        """Create records where each has a distinct BM from *bm_list*."""
        return [
            {
                "BM": bm, "BN": 64, "BK": 256,
                "stages": 2, "nonkdim": 16, "ksplit": 1,
                "timing_us": timing_start + i,
                "M": 16, "N": 64, "K": 256,
                "_weight": 1.0,
            }
            for i, bm in enumerate(bm_list)
        ]

    def test_dominant_value_retained(self):
        """BM=16 appears in 5 of 5 shapes (100%) → must be retained."""
        # Build an executor that returns a single log file with records for
        # 5 shapes, all with BM=16 as the winner.
        all_shapes_log = ""
        for m in [1, 2, 4, 8, 16]:
            all_shapes_log += (
                f"screencase 16 64 256 1 4 2 3 16 0 1\n"
                f"    kernel_M{m}_N64_K256\n"
                f" {10.0 + m}\n"
                f"screencase 32 64 256 1 4 2 3 16 0 1\n"
                f"    kernel_M{m}_N64_K256_b32\n"
                f" {50.0 + m}\n"
            )

        executor = _make_executor()
        executor.docker_exec.side_effect = [
            _completed(stdout="/scout/screen-0.log"),   # ls
            _completed(stdout=all_shapes_log),           # cat
            _completed(stdout=""),                       # write artifact
        ]
        agent = _make_agent(executor=executor)
        result = agent._execute()
        space = result["search_space"]
        # M_LEQ_16 range: BM=16 should be present.
        bm_values = space.get("M_LEQ_16", {}).get("BM", [])
        self.assertIn(16, bm_values)

    def test_rare_value_dropped_when_enough_shapes(self):
        """A value appearing in only 1 of 10 shapes (10%) must be dropped.

        Each shape has 4 configs.  For shapes M=2..10 the top-3 are all
        BM∈{16,32,64} so BM=256 never appears in their top-3.  For shape
        M=1 the 4 configs are BM=256 (fastest) + BM=16 + BM=32 + BM=64.

        BM=256 in top-3: 1 shape out of 10 → 1/10 = 10% → dropped.
        BM=16,32,64 in top-3: all 10 shapes → retained.
        """
        log_lines = ""
        for m in range(1, 11):
            if m == 1:
                # shape M=1: BM=256 makes the top-3 (it's the fastest).
                configs = [
                    (256, 5.0),
                    (16, 10.0),
                    (32, 20.0),
                    (64, 30.0),
                ]
            else:
                # shapes M=2..10: BM=256 is slowest → not in top-3.
                configs = [
                    (16, 10.0),
                    (32, 20.0),
                    (64, 30.0),
                    (256, 99.0),
                ]
            for bm, timing in configs:
                log_lines += (
                    f"screencase {bm} 64 256 1 4 2 3 16 0 1\n"
                    f"    kernel_M{m}_N64_K256_bm{bm}\n"
                    f" {timing}\n"
                )

        executor = _make_executor()
        executor.docker_exec.side_effect = [
            _completed(stdout="/scout/screen-0.log"),
            _completed(stdout=log_lines),
        ]
        agent = _make_agent(executor=executor)
        result = agent._execute()
        space = result["search_space"]
        bm_values = space.get("M_LEQ_16", {}).get("BM", [])
        # BM=256 appeared in top-3 for only 1 of 10 shapes → below 20% threshold.
        self.assertNotIn(256, bm_values)
        # BM=16 appeared in all 10 shapes → retained.
        self.assertIn(16, bm_values)


# ---------------------------------------------------------------------------
# Historical data weighting — unit tests
# ---------------------------------------------------------------------------


class TestHistoricalDataWeight(unittest.TestCase):
    """Historical data contributes 0.25x weight vs scout 1.0x weight."""

    def _make_log_with_bm(self, m_val: int, bm: int, timing: float = 10.0) -> str:
        return (
            f"screencase {bm} 64 256 1 4 2 3 16 0 1\n"
            f"    kernel_M{m_val}_N64_K256\n"
            f" {timing}\n"
        )

    def test_historical_weight_is_lower_than_scout(self):
        """Scout records dominate over historical records.

        We use 4 scout shapes (M=1,2,4,8) each with ONLY BM=16 winning (no
        BM=256 in scout at all).  History has 1 record with BM=256 for a
        different shape (M=16).

        After weighting:
        - BM=16 total weight = 4 shapes * 1.0 (scout, top-1 each) = 4.0
        - BM=256 total weight = 1 shape * 0.25 (history) = 0.25

        BM=16 share = 4.0 / 4.25 ≈ 94% → retained
        BM=256 share = 0.25 / 4.25 ≈ 5.9% → dropped (< 20%)
        """
        # Scout: 4 shapes (M=1,2,4,8), each with a single config BM=16 only.
        scout_log = ""
        for m in [1, 2, 4, 8]:
            scout_log += self._make_log_with_bm(m, bm=16, timing=10.0)

        # History: 1 record for M=16 (different shape key) with BM=256 only.
        history_json = json.dumps([
            {
                "BM": 256, "BN": 64, "BK": 256, "GSM": 1,
                "warps": 4, "stages": 2, "waves": 3, "nonkdim": 16,
                "cache": 0, "ksplit": 1,
                "kernel_name": "kernel_M16_N64_K256",
                "timing_us": 5.0,
                "M": 16, "N": 64, "K": 256,
            }
        ])

        executor = _make_executor()
        executor.docker_exec.side_effect = [
            _completed(stdout="/scout/screen-0.log"),   # ls scout
            _completed(stdout=scout_log),               # cat scout
            _completed(stdout=history_json),            # cat history/patterns.json
        ]
        agent = _make_agent(executor=executor, history_dir="/history")
        result = agent._execute()
        space = result["search_space"]
        bm_values = space.get("M_LEQ_16", {}).get("BM", [])
        # Scout-dominant BM=16 should be present.
        self.assertIn(16, bm_values)
        # History-only BM=256 has 5.9% share → should be dropped.
        self.assertNotIn(256, bm_values)

    def test_history_not_loaded_when_dir_is_none(self):
        """When history_dir=None, no history docker_exec call is made."""
        executor = _make_executor()
        executor.docker_exec.side_effect = [
            _completed(returncode=1, stdout=""),  # ls returns no files
            _completed(stdout=""),                # write artifact
        ]
        agent = _make_agent(executor=executor, history_dir=None)
        agent._execute()
        calls = executor.docker_exec.call_args_list
        history_calls = [c for c in calls if "patterns.json" in str(c)]
        self.assertEqual(len(history_calls), 0)

    def test_history_loaded_when_dir_set(self):
        """When history_dir is set, a cat patterns.json call is made."""
        executor = _make_executor()
        executor.docker_exec.side_effect = [
            _completed(returncode=1, stdout=""),  # ls returns nothing
            _completed(stdout="[]"),              # cat history/patterns.json
            _completed(stdout=""),                # write artifact
        ]
        agent = _make_agent(executor=executor, history_dir="/history")
        agent._execute()
        calls = executor.docker_exec.call_args_list
        history_calls = [c for c in calls if "patterns.json" in str(c)]
        self.assertGreater(len(history_calls), 0)


# ---------------------------------------------------------------------------
# Sanity check — widening tests
# ---------------------------------------------------------------------------


class TestSanityCheckWidening(unittest.TestCase):
    """Narrowed space is widened back when it falls below the 25% threshold."""

    def test_narrowing_ratio_function_correct(self):
        """Check _narrowing_ratio computes average of per-param ratios."""
        broad = {"BM": [4, 8, 16, 32], "BN": [16, 32, 64]}
        narrow = {"BM": [4], "BN": [16, 32]}
        ratio = _narrowing_ratio(narrow, broad)
        # BM: 1/4=0.25, BN: 2/3≈0.667; avg ≈ 0.458
        self.assertAlmostEqual(ratio, (0.25 + 2 / 3) / 2, places=5)

    def test_widen_to_minimum_adds_values(self):
        """If narrowed space is too small, widening adds the most frequent values."""
        broad = {"BM": [4, 8, 16, 32, 64, 128, 256], "BN": [16, 32, 64, 128]}
        # Start with a very narrow space (only 1 value per param).
        narrow = {"BM": [4], "BN": [16]}
        counters = {"BM": {4: 5.0, 8: 4.0, 16: 3.0}, "BN": {16: 5.0, 32: 4.0}}
        result = _widen_to_minimum(narrow, broad, counters)
        ratio = _narrowing_ratio(result, broad)
        self.assertGreaterEqual(ratio, MIN_NARROWING_RATIO)

    def test_execute_search_space_passes_sanity_check(self):
        """_execute must return a narrowing_ratio >= 0.25."""
        # A scout log with only BM=16 and stages=2 winning everywhere.
        scout_log = ""
        for m in [1, 2, 4, 8, 16]:
            scout_log += (
                f"screencase 16 64 256 1 4 2 3 16 0 1\n"
                f"    kernel_M{m}_N64_K256\n"
                f" 10.0\n"
            )

        executor = _make_executor()
        executor.docker_exec.side_effect = [
            _completed(stdout="/scout/screen-0.log"),
            _completed(stdout=scout_log),
            _completed(stdout=""),
        ]
        agent = _make_agent(executor=executor)
        result = agent._execute()
        self.assertGreaterEqual(result["narrowing_ratio"], MIN_NARROWING_RATIO)

    def test_widen_does_not_exceed_broad(self):
        """Widened values are always a subset of the broad space."""
        broad = {"BM": [4, 8, 16, 32], "BN": [16, 32]}
        narrow = {"BM": [4], "BN": [16]}
        counters: dict = {}
        result = _widen_to_minimum(narrow, broad, counters)
        for param in result:
            broad_set = set(broad.get(param, []))
            for v in result[param]:
                self.assertIn(v, broad_set, f"{v} not in broad space for {param}")


# ---------------------------------------------------------------------------
# PatternAnalyzerAgent._execute — integration tests
# ---------------------------------------------------------------------------


class TestExecuteReturnShape(unittest.TestCase):
    """_execute must return the required top-level keys."""

    def _run_execute_with_log(self, log_content, history_dir=None, history_content=None):
        side_effects = [
            _completed(stdout="/scout/screen-0.log"),  # ls
            _completed(stdout=log_content),            # cat
        ]
        if history_dir is not None:
            side_effects.append(
                _completed(stdout=history_content or "[]")
            )
        side_effects.append(_completed(stdout=""))  # write artifact
        executor = _make_executor()
        executor.docker_exec.side_effect = side_effects
        agent = _make_agent(executor=executor, history_dir=history_dir)
        return agent._execute()

    def test_search_space_key_present(self):
        result = self._run_execute_with_log(SCREEN_LOG_TWO_CONFIGS)
        self.assertIn("search_space", result)

    def test_scout_shapes_analyzed_key_present(self):
        result = self._run_execute_with_log(SCREEN_LOG_TWO_CONFIGS)
        self.assertIn("scout_shapes_analyzed", result)

    def test_narrowing_ratio_key_present(self):
        result = self._run_execute_with_log(SCREEN_LOG_TWO_CONFIGS)
        self.assertIn("narrowing_ratio", result)

    def test_search_space_contains_m_ranges(self):
        result = self._run_execute_with_log(SCREEN_LOG_TWO_CONFIGS)
        space = result["search_space"]
        expected_ranges = {label for label, _ in M_RANGES}
        # At least one M-range must be present.
        self.assertTrue(expected_ranges.issuperset(space.keys()))
        self.assertGreater(len(space), 0)

    def test_search_space_contains_tune_params(self):
        result = self._run_execute_with_log(SCREEN_LOG_TWO_CONFIGS)
        space = result["search_space"]
        for label in space:
            for param in TUNE_PARAMS:
                self.assertIn(param, space[label], f"Missing {param} in {label}")

    def test_scout_shapes_analyzed_count(self):
        """Two configs for the same shape → 1 shape analyzed."""
        result = self._run_execute_with_log(SCREEN_LOG_TWO_CONFIGS)
        self.assertEqual(result["scout_shapes_analyzed"], 1)

    def test_narrowing_ratio_is_float(self):
        result = self._run_execute_with_log(SCREEN_LOG_TWO_CONFIGS)
        self.assertIsInstance(result["narrowing_ratio"], float)

    def test_narrowing_ratio_in_valid_range(self):
        result = self._run_execute_with_log(SCREEN_LOG_TWO_CONFIGS)
        self.assertGreaterEqual(result["narrowing_ratio"], 0.0)
        self.assertLessEqual(result["narrowing_ratio"], 1.0)


class TestExecuteNoLogFiles(unittest.TestCase):
    """When no log files exist, _execute returns a safe default."""

    def test_no_log_files_returns_result(self):
        executor = _make_executor()
        executor.docker_exec.side_effect = [
            _completed(returncode=1, stdout=""),  # ls fails
            _completed(stdout=""),                # write artifact
        ]
        agent = _make_agent(executor=executor)
        result = agent._execute()
        # Should not raise; must return the required keys.
        self.assertIn("search_space", result)
        self.assertIn("scout_shapes_analyzed", result)
        self.assertEqual(result["scout_shapes_analyzed"], 0)


class TestExecuteMultipleLogFiles(unittest.TestCase):
    """Records from multiple log files are merged correctly."""

    def test_shapes_from_two_files_combined(self):
        log_1 = (
            "screencase 16 64 256 1 4 2 3 16 0 1\n"
            "    kernel_M1_N64_K256\n"
            " 10.0\n"
        )
        log_2 = (
            "screencase 32 64 512 1 4 3 3 16 0 1\n"
            "    kernel_M8_N64_K512\n"
            " 20.0\n"
        )
        executor = _make_executor()
        executor.docker_exec.side_effect = [
            _completed(stdout="/scout/screen-0.log\n/scout/screen-1.log"),  # ls
            _completed(stdout=log_1),  # cat file 1
            _completed(stdout=log_2),  # cat file 2
            _completed(stdout=""),     # write artifact
        ]
        agent = _make_agent(executor=executor)
        result = agent._execute()
        self.assertEqual(result["scout_shapes_analyzed"], 2)


class TestExecuteWritesArtifact(unittest.TestCase):
    """_execute must write a narrowed_search_space.json artifact.

    _write_json_artifact calls executor.ssh_run (not docker_exec), so we
    verify the ssh_run mock received a printf call.
    """

    def test_artifact_written(self):
        executor = _make_executor()
        executor.docker_exec.side_effect = [
            _completed(stdout="/scout/screen-0.log"),
            _completed(stdout=SCREEN_LOG_TWO_CONFIGS),
        ]
        # ssh_run is called by _write_json_artifact; return a successful result.
        executor.ssh_run.return_value = _completed(stdout="")
        agent = _make_agent(executor=executor)
        agent._execute()

        # _write_json_artifact calls ssh_run with "printf '%s' ... > <path>".
        calls = executor.ssh_run.call_args_list
        artifact_calls = [c for c in calls if "printf" in str(c)]
        self.assertGreater(len(artifact_calls), 0)


class TestExecuteLargeMShape(unittest.TestCase):
    """A shape with M=1024 falls in the M_GEQ_1024 range."""

    def test_large_m_counted_in_xlarge_range(self):
        executor = _make_executor()
        executor.docker_exec.side_effect = [
            _completed(stdout="/scout/screen-0.log"),
            _completed(stdout=SCREEN_LOG_LARGE_M),
            _completed(stdout=""),
        ]
        agent = _make_agent(executor=executor)
        result = agent._execute()
        space = result["search_space"]
        # M_GEQ_1024 range must have BM values from the large-M log.
        bm_values = space.get("M_GEQ_1024", {}).get("BM", [])
        # Both BM=64 and BM=128 appear in the log; at least one must survive.
        self.assertTrue(
            64 in bm_values or 128 in bm_values,
            f"Expected 64 or 128 in BM values for M_GEQ_1024, got {bm_values}",
        )


class TestExecuteRunMethod(unittest.TestCase):
    """PatternAnalyzerAgent.run() wraps _execute and returns SubagentResult."""

    def test_run_returns_success(self):
        executor = _make_executor()
        executor.ssh_run = MagicMock(return_value=_completed(stdout=""))
        executor.kill_stale_gpu_processes = MagicMock(return_value=[])
        executor.docker_exec.side_effect = [
            _completed(returncode=1, stdout=""),  # ls — no files
            _completed(stdout=""),                # write artifact
        ]
        agent = _make_agent(executor=executor)
        result = agent.run()
        self.assertIsInstance(result, SubagentResult)
        self.assertTrue(result.success, msg=result.error)

    def test_run_data_contains_search_space(self):
        executor = _make_executor()
        executor.ssh_run = MagicMock(return_value=_completed(stdout=""))
        executor.kill_stale_gpu_processes = MagicMock(return_value=[])
        executor.docker_exec.side_effect = [
            _completed(returncode=1, stdout=""),  # ls — no files
            _completed(stdout=""),                # write artifact
        ]
        agent = _make_agent(executor=executor)
        result = agent.run()
        self.assertIn("search_space", result.data)


class TestAgentInit(unittest.TestCase):
    """PatternAnalyzerAgent stores constructor arguments correctly."""

    def test_name_attribute(self):
        self.assertEqual(PatternAnalyzerAgent.name, "pattern_analyzer")

    def test_stores_scout_results_dir(self):
        agent = _make_agent(scout_results_dir="/my/scout")
        self.assertEqual(agent.scout_results_dir, "/my/scout")

    def test_stores_history_dir(self):
        agent = _make_agent(history_dir="/my/history")
        self.assertEqual(agent.history_dir, "/my/history")

    def test_history_dir_none_by_default(self):
        agent = _make_agent()
        self.assertIsNone(agent.history_dir)


if __name__ == "__main__":
    unittest.main()
