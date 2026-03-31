"""RegressionFixerAgent — fixes performance regressions in Triton kernel configs.

Responsibilities
----------------
- Inspect each reported regression (m, n, k, delta, current_config_file, bucket).
- Determine the appropriate fix strategy (RESTORE_BUCKET, PROMOTE_TO_SUFFIXED,
  NOISE_SKIP, or ESCALATE).
- Apply fixes while strictly never modifying the default fallback config file.
- Return a summary of actions taken.

Critical rule
~~~~~~~~~~~~~
The default fallback config (a file whose name contains *no* ``N=`` / ``K=``
suffix) must **never** be modified in-place.  When a regression originates from
a fallback file, a new suffixed config is created instead.
"""

import json
import os
import re
import shlex
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from .base import BaseSubagent, SubagentError, SubagentResult


# ---------------------------------------------------------------------------
# Fix strategy enum
# ---------------------------------------------------------------------------


class FixStrategy(Enum):
    """Strategies the agent can apply to resolve a regression."""

    RESTORE_BUCKET = auto()
    """Restore the regressed bucket in the *existing* suffixed config file from
    its counterpart in the old config directory."""

    PROMOTE_TO_SUFFIXED = auto()
    """Copy the fallback config as a new ``N=<N>-K=<K>`` suffixed file and
    restore the regressed bucket within that copy.  The fallback is untouched."""

    NOISE_SKIP = auto()
    """The config didn't actually change between old and current — the
    regression is likely measurement noise; skip it."""

    ESCALATE = auto()
    """The delta exceeds 2× the threshold; the regression is too severe for
    automated fixes and must be escalated to a human."""


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

_SUFFIXED_RE = re.compile(r"N=\d+.*K=\d+", re.IGNORECASE)
"""Regex that matches the N=…-K=… portion present in suffixed config filenames."""


def _is_suffixed(config_path: str) -> bool:
    """Return ``True`` when *config_path* is a suffixed (non-fallback) config."""
    return bool(_SUFFIXED_RE.search(os.path.basename(config_path)))


def _load_json(path: str, executor=None) -> Dict[str, Any]:
    """Load and return parsed JSON from *path*.

    When *executor* is provided the file is read from inside the Docker
    container via ``docker_exec("cat <path>")``.  Otherwise a local
    ``open()`` is used (kept for unit-test compatibility only).
    """
    if executor is not None:
        result = executor.docker_exec(f"cat {shlex.quote(path)}")
        return json.loads(result.stdout)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: str, data: Dict[str, Any], executor=None) -> None:
    """Serialise *data* as indented JSON and write it to *path*.

    When *executor* is provided the file is written inside the Docker
    container via ``docker_exec("printf '%s' <json> > <path>")``.
    Otherwise a local ``open()`` is used (kept for unit-test compatibility
    only).
    """
    serialised = json.dumps(data, indent=2) + "\n"
    if executor is not None:
        executor.docker_exec(
            f"printf '%s' {shlex.quote(serialised)} > {shlex.quote(path)}"
        )
        return
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(serialised)


def _find_matching_bucket(
    old_config: Dict[str, Any], target_bucket: str
) -> Optional[str]:
    """Locate a bucket in *old_config* that corresponds to *target_bucket*.

    The lookup handles slight naming differences between config generations
    (e.g. ``M_LEQ_31`` in the current file vs ``M_LEQ_16`` in the old file):
    exact match is tried first, then a case-insensitive substring match on the
    numeric boundary, then a fallback to an exact-key lookup.

    Parameters
    ----------
    old_config:
        Parsed JSON dict of the old config file.
    target_bucket:
        The bucket name as it appears in the *current* config (e.g.
        ``"M_LEQ_31"``).

    Returns
    -------
    str | None
        The matching key in *old_config*, or ``None`` if no match is found.
    """
    # 1. Exact match.
    if target_bucket in old_config:
        return target_bucket

    # 2. Numeric-boundary match: extract the digits from the target and look
    #    for any old-config key whose digits are the same or "close enough".
    #    For example M_LEQ_31 → try M_LEQ_31 (done above), then M_LEQ_16, etc.
    #    We use a fuzzy prefix match: strip the trailing number and match prefix.
    prefix_match = re.match(r"^(M_LEQ_|M_GT_|M_GEQ_)(\d+)$", target_bucket, re.IGNORECASE)
    if prefix_match:
        prefix = prefix_match.group(1).upper()
        for key in old_config:
            if key.upper().startswith(prefix):
                return key

    # 3. Special-case the "any" / catch-all bucket.
    if target_bucket.lower() == "any" and "any" in old_config:
        return "any"

    # 4. Case-insensitive exact match.
    target_lower = target_bucket.lower()
    for key in old_config:
        if key.lower() == target_lower:
            return key

    return None


def _derive_suffixed_filename(fallback_path: str, n: int, k: int) -> str:
    """Derive the new suffixed filename from a fallback config path.

    For example ``/configs/gfx950-GEMM-A16W16.json`` with ``n=128, k=4096``
    becomes ``/configs/gfx950-GEMM-A16W16-N=128-K=4096.json``.

    Parameters
    ----------
    fallback_path:
        Absolute path to the fallback (non-suffixed) config file.
    n, k:
        The N and K dimensions for the new suffixed file.

    Returns
    -------
    str
        Absolute path for the new suffixed config file.
    """
    directory = os.path.dirname(fallback_path)
    base = os.path.basename(fallback_path)
    stem, ext = os.path.splitext(base)
    new_name = f"{stem}-N={n}-K={k}{ext}"
    return os.path.join(directory, new_name)


# ---------------------------------------------------------------------------
# RegressionFixerAgent
# ---------------------------------------------------------------------------


class RegressionFixerAgent(BaseSubagent):
    """Fix performance regressions identified in Triton kernel tuning configs.

    Parameters
    ----------
    executor:
        :class:`~tuning_agent.remote.RemoteExecutor` connected to the target
        machine.  Passed through to :class:`BaseSubagent`.
    kernel_name:
        Short identifier for the kernel (e.g. ``"gemm_a16w16"``).
    artifact_dir:
        Absolute path on the remote host for JSON artifacts.
    regressions:
        List of regression dicts.  Each dict must contain:

        - ``m`` (int): M dimension.
        - ``n`` (int): N dimension.
        - ``k`` (int): K dimension.
        - ``delta`` (float): relative performance delta (positive = slower).
        - ``current_config_file`` (str): path to the current config JSON.
        - ``bucket`` (str): config bucket name, e.g. ``"M_LEQ_16"``.

    config_dir:
        Directory on the *local* host that contains the current config files.
    old_config_dir:
        Directory on the *local* host that contains the baseline (old) config
        files to restore from.
    threshold:
        Minimum delta considered a real regression.  Regressions with
        ``delta > 2 * threshold`` are escalated rather than auto-fixed.
    max_iterations:
        Reserved for future retry loops; currently unused by the core logic.
    """

    name: str = "regression_fixer"

    def __init__(
        self,
        executor,
        kernel_name: str,
        artifact_dir: str,
        regressions: List[Dict[str, Any]],
        config_dir: str,
        old_config_dir: str,
        threshold: float,
        max_iterations: int = 3,
    ) -> None:
        super().__init__(
            executor=executor,
            kernel_name=kernel_name,
            artifact_dir=artifact_dir,
        )
        self.regressions = regressions
        self.config_dir = config_dir
        self.old_config_dir = old_config_dir
        self.threshold = threshold
        self.max_iterations = max_iterations

    # ------------------------------------------------------------------
    # Strategy determination
    # ------------------------------------------------------------------

    def determine_fix_strategy(self, regression: Dict[str, Any]) -> FixStrategy:
        """Decide how to fix *regression*.

        Decision tree
        ~~~~~~~~~~~~~
        1. If ``delta > 2 × threshold`` → :attr:`FixStrategy.ESCALATE`.
        2. If the current config and the old config are identical for the
           regressed bucket → :attr:`FixStrategy.NOISE_SKIP`.
        3. If the current config file is the default fallback (no ``N=``/``K=``
           suffix) → :attr:`FixStrategy.PROMOTE_TO_SUFFIXED`.
        4. Otherwise (suffixed file) → :attr:`FixStrategy.RESTORE_BUCKET`.

        Parameters
        ----------
        regression:
            A single regression dict (see constructor docs).

        Returns
        -------
        FixStrategy
        """
        delta: float = regression["delta"]
        current_config_file: str = regression["current_config_file"]
        bucket: str = regression["bucket"]

        # --- Rule 1: Large delta → escalate ---
        if delta > 2 * self.threshold:
            return FixStrategy.ESCALATE

        # --- Rule 2: Check if config actually changed ---
        old_config_file = os.path.join(
            self.old_config_dir, os.path.basename(current_config_file)
        )
        try:
            current_config = _load_json(current_config_file, self.executor)
            old_config = _load_json(old_config_file, self.executor)
        except (FileNotFoundError, json.JSONDecodeError, Exception):
            # If we cannot load either file, we cannot compare — fall through
            # to the file-type-based rules.
            current_config = None
            old_config = None

        if current_config is not None and old_config is not None:
            current_bucket_cfg = current_config.get(bucket)
            old_bucket_key = _find_matching_bucket(old_config, bucket)
            old_bucket_cfg = old_config.get(old_bucket_key) if old_bucket_key else None

            if current_bucket_cfg is not None and current_bucket_cfg == old_bucket_cfg:
                return FixStrategy.NOISE_SKIP

        # --- Rule 3/4: File-type-based rules ---
        if _is_suffixed(current_config_file):
            return FixStrategy.RESTORE_BUCKET
        else:
            return FixStrategy.PROMOTE_TO_SUFFIXED

    # ------------------------------------------------------------------
    # Fix operations
    # ------------------------------------------------------------------

    def restore_bucket(
        self,
        config_path: str,
        bucket_name: str,
        old_config_path: str,
    ) -> None:
        """Replace *bucket_name* in *config_path* with the version from *old_config_path*.

        Parameters
        ----------
        config_path:
            Absolute path to the *current* (suffixed) config JSON to modify.
        bucket_name:
            The bucket key to restore (e.g. ``"M_LEQ_16"``).
        old_config_path:
            Absolute path to the old (baseline) config JSON.

        Raises
        ------
        SubagentError
            If the old config does not contain a matching bucket, or if either
            file cannot be read/written.
        """
        try:
            current_config = _load_json(config_path, self.executor)
        except Exception as exc:
            raise SubagentError(
                f"Cannot read current config {config_path!r}: {exc}"
            ) from exc

        try:
            old_config = _load_json(old_config_path, self.executor)
        except Exception as exc:
            raise SubagentError(
                f"Cannot read old config {old_config_path!r}: {exc}"
            ) from exc

        old_bucket_key = _find_matching_bucket(old_config, bucket_name)
        if old_bucket_key is None:
            raise SubagentError(
                f"Bucket {bucket_name!r} not found in old config {old_config_path!r}"
            )

        # Replace only the target bucket; leave all other buckets intact.
        current_config[bucket_name] = old_config[old_bucket_key]
        _write_json(config_path, current_config, self.executor)

    def promote_to_suffixed(
        self,
        n: int,
        k: int,
        fallback_path: str,
        old_fallback_path: str,
        bucket_name: str,
    ) -> str:
        """Create a new suffixed config and restore the regressed bucket.

        This method must **never** write to *fallback_path*.

        1. Derive the new suffixed filename (e.g. ``gfx950-GEMM-A16W16-N=128-K=4096.json``).
        2. Copy the current fallback config into the new file.
        3. Replace the regressed bucket in the new file with the version from
           *old_fallback_path*.

        Parameters
        ----------
        n, k:
            N and K dimensions used to name the new suffixed file.
        fallback_path:
            Absolute path to the current default fallback config.  **Not
            modified.**
        old_fallback_path:
            Absolute path to the old (baseline) fallback config to restore
            the bucket from.
        bucket_name:
            The bucket key to restore in the new suffixed file.

        Returns
        -------
        str
            Absolute path of the newly created suffixed config file.

        Raises
        ------
        SubagentError
            If either source file cannot be read, or the old fallback does not
            contain a matching bucket.
        """
        new_path = _derive_suffixed_filename(fallback_path, n, k)

        try:
            fallback_config = _load_json(fallback_path, self.executor)
        except Exception as exc:
            raise SubagentError(
                f"Cannot read fallback config {fallback_path!r}: {exc}"
            ) from exc

        try:
            old_fallback_config = _load_json(old_fallback_path, self.executor)
        except Exception as exc:
            raise SubagentError(
                f"Cannot read old fallback config {old_fallback_path!r}: {exc}"
            ) from exc

        old_bucket_key = _find_matching_bucket(old_fallback_config, bucket_name)
        if old_bucket_key is None:
            raise SubagentError(
                f"Bucket {bucket_name!r} not found in old fallback config "
                f"{old_fallback_path!r}"
            )

        # Build the new suffixed config: start from the current fallback content,
        # then restore the regressed bucket from the old fallback.
        new_config = dict(fallback_config)
        new_config[bucket_name] = old_fallback_config[old_bucket_key]

        # Write to the NEW suffixed file — the fallback is never touched.
        _write_json(new_path, new_config, self.executor)
        return new_path

    # ------------------------------------------------------------------
    # Core execute
    # ------------------------------------------------------------------

    def _execute(self) -> dict:
        """Process all regressions and apply fixes.

        Returns
        -------
        dict
            Keys:

            - ``fixed`` (int): number of buckets restored in existing files.
            - ``promoted`` (int): number of new suffixed files created.
            - ``skipped`` (int): noise-skip count.
            - ``escalated`` (int): escalation count.
            - ``escalations`` (list): regression dicts that were escalated.
        """
        counts = {
            "fixed": 0,
            "promoted": 0,
            "skipped": 0,
            "escalated": 0,
        }
        escalations: List[Dict[str, Any]] = []

        for regression in self.regressions:
            strategy = self.determine_fix_strategy(regression)

            if strategy is FixStrategy.NOISE_SKIP:
                counts["skipped"] += 1

            elif strategy is FixStrategy.ESCALATE:
                counts["escalated"] += 1
                escalations.append(regression)

            elif strategy is FixStrategy.RESTORE_BUCKET:
                current_config_file: str = regression["current_config_file"]
                bucket: str = regression["bucket"]
                old_config_file = os.path.join(
                    self.old_config_dir, os.path.basename(current_config_file)
                )
                self.restore_bucket(current_config_file, bucket, old_config_file)
                counts["fixed"] += 1

            elif strategy is FixStrategy.PROMOTE_TO_SUFFIXED:
                n: int = regression["n"]
                k: int = regression["k"]
                fallback_path: str = regression["current_config_file"]
                old_fallback_path = os.path.join(
                    self.old_config_dir, os.path.basename(fallback_path)
                )
                bucket: str = regression["bucket"]
                self.promote_to_suffixed(n, k, fallback_path, old_fallback_path, bucket)
                counts["promoted"] += 1

        return {
            "fixed": counts["fixed"],
            "promoted": counts["promoted"],
            "skipped": counts["skipped"],
            "escalated": counts["escalated"],
            "escalations": escalations,
        }
