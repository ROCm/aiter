"""PatternAnalyzerAgent — analyze scout results to narrow the tuning search space.

Responsibilities
----------------
- Load raw scout (exploratory) tuning results from *scout_results_dir* and
  identify which hyperparameter combinations performed best across M-ranges.
- Optionally blend in historical tuning data from previous runs (*history_dir*)
  using a 0.25x weighting factor so that established knowledge influences but
  does not dominate the new search space.
- Segment the search space by M-range (e.g. M<=16, M 32-64, M 128-512, M>=1024)
  and output a narrowed candidate list of configs per segment for the exhaustive
  tuning phase.
"""

import json
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ..remote import RemoteExecutor
from .base import BaseSubagent


# ---------------------------------------------------------------------------
# M-range definitions
# ---------------------------------------------------------------------------

M_RANGES = [
    ("M_LEQ_16", lambda m: m <= 16),
    ("M_32_64", lambda m: 32 <= m <= 64),
    ("M_128_512", lambda m: 128 <= m <= 512),
    ("M_GEQ_1024", lambda m: m >= 1024),
]

# Canonical broad search space for each parameter (used in sanity checks).
# These represent all values that could plausibly appear in the scout logs.
BROAD_SPACE: Dict[str, List[int]] = {
    "BM": [4, 8, 16, 32, 64, 128, 256],
    "BN": [16, 32, 64, 128, 256, 512, 1024],
    "BK": [64, 128, 256, 512, 1024, 2048],
    "stages": [1, 2, 3, 4, 5],
    "nonkdim": [0, 16, 32],
    "ksplit": [1, 2, 4, 8, 16],
}

TUNE_PARAMS = list(BROAD_SPACE.keys())

# Threshold: a value must appear in >20% of winning configs to be retained.
WINNING_THRESHOLD = 0.20

# Sanity check: narrowed space must cover >=25% of broad space.
MIN_NARROWING_RATIO = 0.25


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------


def parse_screen_log(content: str) -> List[dict]:
    """Parse screencase lines from a screen.py tuning log.

    Expected format (one screencase block per configuration tested)::

        screencase BM BN BK GSM warps stages waves nonkdim cache ksplit
            kernel_name
         timing_us

    Parameters
    ----------
    content:
        Raw text content of a ``screen-*.log`` file.

    Returns
    -------
    List[dict]
        Each entry contains the keys: ``BM``, ``BN``, ``BK``, ``GSM``,
        ``warps``, ``stages``, ``waves``, ``nonkdim``, ``cache``,
        ``ksplit``, ``kernel_name``, ``timing_us``, ``M``, ``N``, ``K``.
        Shape dimensions ``M``, ``N``, ``K`` are extracted from the
        kernel_name when possible (e.g. ``…_M16_N64_K256_…``).
    """
    results: List[dict] = []

    # Match a "screencase" header line.
    # Format: screencase <BM> <BN> <BK> <GSM> <warps> <stages> <waves> <nonkdim> <cache> <ksplit>
    screencase_re = re.compile(
        r"^\s*screencase\s+"
        r"(\d+)\s+"   # BM
        r"(\d+)\s+"   # BN
        r"(\d+)\s+"   # BK
        r"(\d+)\s+"   # GSM
        r"(\d+)\s+"   # warps
        r"(\d+)\s+"   # stages
        r"(\d+)\s+"   # waves
        r"(\d+)\s+"   # nonkdim
        r"(\d+)\s+"   # cache
        r"(\d+)"      # ksplit
        r"\s*$",
        re.MULTILINE,
    )

    lines = content.splitlines()
    i = 0
    while i < len(lines):
        m = screencase_re.match(lines[i])
        if m is None:
            i += 1
            continue

        bm, bn, bk, gsm, warps, stages, waves, nonkdim, cache, ksplit = (
            int(x) for x in m.groups()
        )

        # Next non-blank line is the kernel name.
        kernel_name = ""
        j = i + 1
        while j < len(lines) and lines[j].strip() == "":
            j += 1
        if j < len(lines):
            kernel_name = lines[j].strip()
            j += 1
        else:
            i += 1
            continue

        # Next non-blank line is the timing.
        # We peek at the line: if it does not parse as a float (e.g. it is
        # another screencase header) we do NOT consume it, so that the outer
        # loop can pick it up as the next screencase block.
        timing_us: Optional[float] = None
        timing_j = j  # position of the candidate timing line
        while timing_j < len(lines) and lines[timing_j].strip() == "":
            timing_j += 1
        if timing_j < len(lines):
            timing_str = lines[timing_j].strip()
            try:
                timing_us = float(timing_str)
                j = timing_j + 1  # consume the timing line
            except ValueError:
                # The line is not a float — leave it unconsumed and skip
                # this incomplete block.
                i = timing_j  # restart outer scan from this line
                continue

        if timing_us is None:
            i = timing_j
            continue

        # Extract M, N, K from the kernel name if present.
        shape_m = _extract_dim(kernel_name, "M")
        shape_n = _extract_dim(kernel_name, "N")
        shape_k = _extract_dim(kernel_name, "K")

        results.append(
            {
                "BM": bm,
                "BN": bn,
                "BK": bk,
                "GSM": gsm,
                "warps": warps,
                "stages": stages,
                "waves": waves,
                "nonkdim": nonkdim,
                "cache": cache,
                "ksplit": ksplit,
                "kernel_name": kernel_name,
                "timing_us": timing_us,
                "M": shape_m,
                "N": shape_n,
                "K": shape_k,
            }
        )
        i = j

    return results


def _extract_dim(kernel_name: str, dim: str) -> Optional[int]:
    """Extract a dimension value like M16 or _M16_ from kernel_name."""
    pattern = rf"(?:^|_){dim}(\d+)(?:_|$)"
    m = re.search(pattern, kernel_name)
    if m:
        return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def _assign_m_range(m_val: Optional[int]) -> Optional[str]:
    """Return the M-range label for a given M value, or None if not covered."""
    if m_val is None:
        return None
    for label, predicate in M_RANGES:
        if predicate(m_val):
            return label
    return None


def _top_k_configs(records: List[dict], k: int = 3) -> List[dict]:
    """Return the top-*k* configs (lowest timing_us) from *records*.

    De-duplicate by (BM, BN, BK, stages, nonkdim, ksplit) before ranking.
    """
    seen: Dict[Tuple, float] = {}
    key_fields = ("BM", "BN", "BK", "stages", "nonkdim", "ksplit")
    for rec in records:
        key = tuple(rec[f] for f in key_fields)
        if key not in seen or rec["timing_us"] < seen[key]:
            seen[key] = rec["timing_us"]

    # Reconstruct dict list from seen.
    deduped = []
    used: set = set()
    for rec in sorted(records, key=lambda r: r["timing_us"]):
        key = tuple(rec[f] for f in key_fields)
        if key not in used:
            used.add(key)
            deduped.append(rec)

    return deduped[:k]


def _count_param_values(
    top_configs: List[dict],
    weight: float,
    counters: Dict[str, Dict[int, float]],
    total_counters: Dict[str, float],
) -> None:
    """Accumulate weighted counts for each TUNE_PARAM value in top_configs."""
    for cfg in top_configs:
        for param in TUNE_PARAMS:
            val = cfg.get(param)
            if val is None:
                continue
            counters[param][val] = counters[param].get(val, 0.0) + weight
            total_counters[param] = total_counters.get(param, 0.0) + weight


def _narrowing_ratio(
    narrowed: Dict[str, List[int]],
    broad: Dict[str, List[int]],
) -> float:
    """Compute the average ratio of narrowed/broad space sizes across all params."""
    ratios = []
    for param in TUNE_PARAMS:
        broad_size = len(broad.get(param, []))
        narrow_size = len(narrowed.get(param, []))
        if broad_size > 0:
            ratios.append(narrow_size / broad_size)
    if not ratios:
        return 1.0
    return sum(ratios) / len(ratios)


def _widen_to_minimum(
    narrowed: Dict[str, List[int]],
    broad: Dict[str, List[int]],
    counters: Dict[str, Dict[int, float]],
) -> Dict[str, List[int]]:
    """Widen *narrowed* back toward *broad* until it covers >=25% of broad space.

    Strategy: for each parameter that is too narrow, add the most-frequently
    observed values from the scout until the 25% threshold is met.
    """
    result = {p: list(v) for p, v in narrowed.items()}

    # Iteratively add values until ratio >= MIN_NARROWING_RATIO.
    for _ in range(100):  # safety cap
        if _narrowing_ratio(result, broad) >= MIN_NARROWING_RATIO:
            break
        # Find the parameter with the smallest ratio and expand it.
        worst_param = None
        worst_ratio = 1.0
        for param in TUNE_PARAMS:
            broad_size = len(broad.get(param, []))
            if broad_size == 0:
                continue
            ratio = len(result.get(param, [])) / broad_size
            if ratio < worst_ratio:
                worst_ratio = ratio
                worst_param = param

        if worst_param is None:
            break

        # Add the most frequent unseen value.
        current_set = set(result.get(worst_param, []))
        candidates = [
            v for v in broad.get(worst_param, []) if v not in current_set
        ]
        if not candidates:
            break

        # Rank candidates by observed frequency (counters), then by value.
        param_counts = counters.get(worst_param, {})
        candidates.sort(key=lambda v: (-param_counts.get(v, 0.0), v))
        result.setdefault(worst_param, []).append(candidates[0])
        result[worst_param].sort()

    return result


# ---------------------------------------------------------------------------
# PatternAnalyzerAgent
# ---------------------------------------------------------------------------


class PatternAnalyzerAgent(BaseSubagent):
    """Analyse scout results and produce a narrowed per-M-range search space.

    Parameters
    ----------
    executor:
        :class:`~tuning_agent.remote.RemoteExecutor` connected to the target
        machine.
    kernel_name:
        Short identifier for the kernel being tuned (e.g. ``"fmha"``).
    artifact_dir:
        Absolute path on the remote host where JSON artifacts are written.
    scout_results_dir:
        Absolute path on the remote host containing the raw JSON files
        produced by the scout (exploratory) tuning sweep.
    history_dir:
        Optional absolute path on the remote host to a directory of JSON
        files from previous tuning runs.  When provided, historical configs
        are blended into the analysis with a weight of 0.25.
    expected_triton_commit:
        Forwarded to :class:`~.base.BaseSubagent`.
    expected_aiter_branch:
        Forwarded to :class:`~.base.BaseSubagent`.
    """

    name: str = "pattern_analyzer"

    def __init__(
        self,
        executor: RemoteExecutor,
        kernel_name: str,
        artifact_dir: str,
        scout_results_dir: str,
        history_dir: Optional[str] = None,
        expected_triton_commit: Optional[str] = None,
        expected_aiter_branch: Optional[str] = None,
    ) -> None:
        super().__init__(
            executor=executor,
            kernel_name=kernel_name,
            artifact_dir=artifact_dir,
            expected_triton_commit=expected_triton_commit,
            expected_aiter_branch=expected_aiter_branch,
        )
        self.scout_results_dir = scout_results_dir
        self.history_dir = history_dir

    # ------------------------------------------------------------------
    # _execute implementation
    # ------------------------------------------------------------------

    def _execute(self) -> dict:
        """Analyse scout data and output a narrowed config search space.

        Steps
        -----
        1. List and parse all ``screen-*.log`` files under *scout_results_dir*.
        2. For each shape (M, N, K) find the top-3 configs by timing.
        3. Group configs by M-range (M_LEQ_16 / M_32_64 / M_128_512 / M_GEQ_1024).
        4. Count how often each parameter value appears in the top-3 configs
           (scout weight = 1.0x).
        5. Optionally blend in historical data from *history_dir* (weight = 0.25x).
        6. Retain values that appear in >20% of winning configs (weighted).
        7. Sanity-check: if narrowed space < 25% of broad space, widen back.
        8. Return ``search_space``, ``scout_shapes_analyzed``, ``narrowing_ratio``.

        Returns
        -------
        dict
            ``{"search_space": {...}, "scout_shapes_analyzed": N, "narrowing_ratio": float}``
        """
        # ------------------------------------------------------------------
        # Step 1 — load scout logs
        # ------------------------------------------------------------------
        scout_records = self._load_screen_logs(self.scout_results_dir, weight=1.0)

        # ------------------------------------------------------------------
        # Step 5 — optionally load historical data
        # ------------------------------------------------------------------
        history_records: List[dict] = []
        if self.history_dir:
            history_records = self._load_history(self.history_dir, weight=0.25)

        all_records = scout_records + history_records

        # ------------------------------------------------------------------
        # Step 2 — collect records grouped by (M, N, K) shape
        # ------------------------------------------------------------------
        by_shape: Dict[Tuple, List[dict]] = defaultdict(list)
        for rec in all_records:
            mnk = (rec.get("M"), rec.get("N"), rec.get("K"))
            by_shape[mnk].append(rec)

        # Count distinct scout shapes (weight=1.0 records with a valid M).
        scout_shapes = {
            mnk
            for mnk, recs in by_shape.items()
            if mnk[0] is not None
            and any(r.get("_weight", 1.0) == 1.0 for r in recs)
        }
        scout_shapes_analyzed = len(scout_shapes)

        # ------------------------------------------------------------------
        # Steps 3 & 4 — group by M-range, accumulate weighted counts
        # ------------------------------------------------------------------
        # Per M-range counters: {range_label: {param: {value: weighted_count}}}
        range_counters: Dict[str, Dict[str, Dict[int, float]]] = {
            label: {p: {} for p in TUNE_PARAMS} for label, _ in M_RANGES
        }
        range_total: Dict[str, Dict[str, float]] = {
            label: {p: 0.0 for p in TUNE_PARAMS} for label, _ in M_RANGES
        }

        for mnk, recs in by_shape.items():
            m_val = mnk[0]
            range_label = _assign_m_range(m_val)
            if range_label is None:
                continue

            # Split scout and history records for this shape.
            scout_recs = [r for r in recs if r.get("_weight", 1.0) == 1.0]
            hist_recs = [r for r in recs if r.get("_weight", 1.0) != 1.0]

            if scout_recs:
                top3_scout = _top_k_configs(scout_recs, k=3)
                _count_param_values(
                    top3_scout, 1.0,
                    range_counters[range_label],
                    range_total[range_label],
                )

            if hist_recs:
                top3_hist = _top_k_configs(hist_recs, k=3)
                _count_param_values(
                    top3_hist, 0.25,
                    range_counters[range_label],
                    range_total[range_label],
                )

        # ------------------------------------------------------------------
        # Step 6 — apply >20% threshold per M-range per parameter
        # ------------------------------------------------------------------
        search_space: Dict[str, Dict[str, List[int]]] = {}
        # Also keep per-range counters for sanity-check widening.
        per_range_narrow: Dict[str, Dict[str, List[int]]] = {}

        for label, _ in M_RANGES:
            narrow: Dict[str, List[int]] = {}
            for param in TUNE_PARAMS:
                total_w = range_total[label][param]
                if total_w == 0.0:
                    # No data for this range/param — keep broad space.
                    narrow[param] = list(BROAD_SPACE[param])
                    continue
                param_counts = range_counters[label][param]
                kept = sorted(
                    v for v, w in param_counts.items() if w / total_w > WINNING_THRESHOLD
                )
                if not kept:
                    # Nothing survived the threshold — keep full broad space.
                    kept = list(BROAD_SPACE[param])
                narrow[param] = kept
            per_range_narrow[label] = narrow

        # ------------------------------------------------------------------
        # Step 7 — sanity check: narrowed space >= 25% of broad space
        # ------------------------------------------------------------------
        # We check the global (across all M-ranges) average narrowing ratio.
        # Flatten all narrowed values into a set per parameter.
        global_narrow: Dict[str, set] = {p: set() for p in TUNE_PARAMS}
        for label in per_range_narrow:
            for param in TUNE_PARAMS:
                global_narrow[param].update(per_range_narrow[label].get(param, []))

        global_narrow_lists = {p: sorted(v) for p, v in global_narrow.items()}
        ratio = _narrowing_ratio(global_narrow_lists, BROAD_SPACE)

        if ratio < MIN_NARROWING_RATIO:
            # Flatten all per-range counters for widening guidance.
            flat_counters: Dict[str, Dict[int, float]] = {p: {} for p in TUNE_PARAMS}
            for label in range_counters:
                for param in TUNE_PARAMS:
                    for v, w in range_counters[label][param].items():
                        flat_counters[param][v] = flat_counters[param].get(v, 0.0) + w

            # Widen each M-range individually until global ratio >= 25%.
            for label in per_range_narrow:
                per_range_narrow[label] = _widen_to_minimum(
                    per_range_narrow[label], BROAD_SPACE, flat_counters
                )

            # Recompute global ratio after widening.
            global_narrow2: Dict[str, set] = {p: set() for p in TUNE_PARAMS}
            for label in per_range_narrow:
                for param in TUNE_PARAMS:
                    global_narrow2[param].update(per_range_narrow[label].get(param, []))
            global_narrow_lists = {p: sorted(v) for p, v in global_narrow2.items()}
            ratio = _narrowing_ratio(global_narrow_lists, BROAD_SPACE)

        search_space = per_range_narrow

        # ------------------------------------------------------------------
        # Write artifact and return
        # ------------------------------------------------------------------
        artifact = {
            "search_space": search_space,
            "scout_shapes_analyzed": scout_shapes_analyzed,
            "narrowing_ratio": ratio,
        }
        self._write_json_artifact("narrowed_search_space.json", artifact)

        return artifact

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_screen_logs(self, logs_dir: str, weight: float) -> List[dict]:
        """List and parse all ``screen-*.log`` files under *logs_dir*.

        Each parsed record gets a ``_weight`` key set to *weight*.
        """
        result = self.executor.docker_exec(
            f"ls {logs_dir}/screen-*.log",
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []

        log_files = result.stdout.strip().splitlines()
        records: List[dict] = []
        for log_path in log_files:
            log_path = log_path.strip()
            if not log_path:
                continue
            cat_result = self.executor.docker_exec(
                f"cat {log_path}",
                check=False,
            )
            if cat_result.returncode != 0:
                continue
            parsed = parse_screen_log(cat_result.stdout)
            for rec in parsed:
                rec["_weight"] = weight
            records.extend(parsed)

        return records

    def _load_history(self, history_dir: str, weight: float) -> List[dict]:
        """Load historical patterns.json from *history_dir* if it exists.

        Returns parsed records with ``_weight`` set to *weight*.
        """
        cat_result = self.executor.docker_exec(
            f"cat {history_dir}/patterns.json",
            check=False,
        )
        if cat_result.returncode != 0 or not cat_result.stdout.strip():
            return []

        try:
            data = json.loads(cat_result.stdout)
        except json.JSONDecodeError:
            return []

        # Historical patterns.json is expected to be a list of records in the
        # same format as scout logs (BM, BN, BK, etc. + M, N, K, timing_us).
        records: List[dict] = []
        if isinstance(data, list):
            for rec in data:
                if isinstance(rec, dict):
                    rec["_weight"] = weight
                    records.append(rec)
        return records
