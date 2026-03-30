"""TuningAgent — kernel tuning orchestration phase of the Triton tuning pipeline.

Responsibilities
----------------
- Distribute (M, N, K) shapes across available GPUs.
- Launch screen.py per GPU via docker_exec to perform the actual search.
- Monitor progress using RemoteProgressMonitor.
- Return a summary of pairs tuned and the log directory.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from ..remote import RemoteExecutor
from ..watchdog import RemoteProgressMonitor
from .base import BaseSubagent, SubagentResult

# Type alias for a single shape tuple.
Shape = Tuple[int, int, int]


class TuningAgent(BaseSubagent):
    """Orchestrate parallel screen.py tuning runs across multiple GPUs.

    Parameters
    ----------
    executor:
        :class:`~tuning_agent.remote.RemoteExecutor` connected to the target
        machine.
    kernel_name:
        Short identifier for the kernel being tuned (e.g. ``"gemm"``).
    artifact_dir:
        Absolute path on the remote host where JSON artifacts are written.
    shapes_to_tune:
        List of ``(M, N, K)`` tuples that should be tuned.
    search_space:
        Dict mapping M-range bucket names (e.g. ``"M_LEQ_16"``) to dicts of
        hyperparameter values such as ``BLOCK_SIZE_M``, ``BLOCK_SIZE_N``,
        ``BLOCK_SIZE_K``, ``num_stages``, and ``matrix_instr_nonkdim``.
    ut_script:
        Path (inside the container) to the unit-test script passed to
        ``screen.py`` as the positional ``ut_script.py`` argument.
    gpu_ids:
        List of GPU device IDs available for tuning.
    tunning_dir:
        Working directory (inside the container) where ``screen.py`` lives.
    max_batch:
        Maximum batch size forwarded to screen.py via the
        ``SCREEN_MAX_BATCH`` environment variable.  Defaults to ``100``; when
        set to exactly ``100`` the env-var prefix is omitted.
    expected_triton_commit:
        When set, preflight asserts the installed Triton commit matches.
    expected_aiter_branch:
        When set, preflight asserts the aiter git branch matches.
    """

    name: str = "tuning"

    def __init__(
        self,
        executor: RemoteExecutor,
        kernel_name: str,
        artifact_dir: str,
        shapes_to_tune: List[Shape],
        search_space: Dict[str, Any],
        ut_script: str,
        gpu_ids: List[int],
        tunning_dir: str,
        max_batch: int = 100,
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
        self.shapes_to_tune = shapes_to_tune
        self.search_space = search_space
        self.ut_script = ut_script
        self.gpu_ids = gpu_ids
        self.tunning_dir = tunning_dir
        self.max_batch = max_batch

    # ------------------------------------------------------------------
    # Command building
    # ------------------------------------------------------------------

    def build_screen_command(
        self,
        m: int,
        n: int,
        k: int,
        gpu_id: int,
        search_space_for_m: Dict[str, Any],
    ) -> str:
        """Build the shell command string for a single screen.py invocation.

        Parameters
        ----------
        m, n, k:
            Matrix dimensions for this tuning run.
        gpu_id:
            GPU device ID (passed as the ``G`` positional argument to
            ``screen.py`` which sets ``HIP_VISIBLE_DEVICES``).
        search_space_for_m:
            Hyperparameter search space entry for this M range.  Expected
            keys: ``BLOCK_SIZE_M`` (list or scalar), ``BLOCK_SIZE_N`` (list or
            scalar), ``BLOCK_SIZE_K`` (list or scalar), ``num_stages`` (list or
            scalar, or a ``{"min": v, "max": v}`` mapping),
            ``matrix_instr_nonkdim`` (list or scalar).

        Returns
        -------
        str
            The complete shell command, optionally prefixed with
            ``SCREEN_MAX_BATCH=<max_batch>`` when :attr:`max_batch` is not
            ``100``.
        """
        parts: List[str] = []

        # Env-var prefix when max_batch deviates from the default.
        if self.max_batch != 100:
            parts.append(f"SCREEN_MAX_BATCH={self.max_batch}")

        parts += [
            "python", "screen.py",
            str(m), str(n), str(k), str(gpu_id), self.ut_script,
        ]

        # -- block-size-m-range ----------------------------------------
        bm = search_space_for_m.get("BLOCK_SIZE_M", [])
        bm_list = bm if isinstance(bm, list) else [bm]
        parts.append("--block-size-m-range")
        parts += [str(v) for v in bm_list]

        # -- block-size-n-range ----------------------------------------
        bn = search_space_for_m.get("BLOCK_SIZE_N", [])
        bn_list = bn if isinstance(bn, list) else [bn]
        parts.append("--block-size-n-range")
        parts += [str(v) for v in bn_list]

        # -- block-size-k-range ----------------------------------------
        bk = search_space_for_m.get("BLOCK_SIZE_K", [])
        bk_list = bk if isinstance(bk, list) else [bk]
        parts.append("--block-size-k-range")
        parts += [str(v) for v in bk_list]

        # -- num-stages-range ------------------------------------------
        ns = search_space_for_m.get("num_stages", [])
        parts.append("--num-stages-range")
        if isinstance(ns, dict):
            # Accepts {"min": X, "max": Y}
            parts += [str(ns.get("min", 1)), str(ns.get("max", 4))]
        elif isinstance(ns, list):
            if len(ns) >= 2:
                parts += [str(ns[0]), str(ns[-1])]
            elif len(ns) == 1:
                parts += [str(ns[0]), str(ns[0])]
            else:
                parts += ["1", "4"]
        else:
            # Scalar — use as both min and max.
            parts += [str(ns), str(ns)]

        # -- matrix-instr-nonkdim-range --------------------------------
        mi = search_space_for_m.get("matrix_instr_nonkdim", [])
        mi_list = mi if isinstance(mi, list) else [mi]
        parts.append("--matrix-instr-nonkdim-range")
        parts += [str(v) for v in mi_list]

        # -- overwrite flag (always include) ---------------------------
        parts.append("--overwrite")

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Shape distribution
    # ------------------------------------------------------------------

    def distribute_shapes_to_gpus(
        self,
        shapes: List[Shape],
        gpu_ids: List[int],
    ) -> Dict[int, List[Shape]]:
        """Distribute shapes across GPUs in round-robin order, grouped by (N, K).

        Shapes are first grouped by ``(N, K)``.  Within each ``(N, K)`` group
        the M values are distributed round-robin across the given GPUs so that
        each GPU receives a contiguous set of shapes.

        Parameters
        ----------
        shapes:
            List of ``(M, N, K)`` tuples.
        gpu_ids:
            Available GPU device IDs.

        Returns
        -------
        Dict[int, List[Shape]]
            Mapping from GPU ID to the list of shapes assigned to it.
        """
        if not gpu_ids:
            return {}

        # Group by (N, K) then sort M values within each group.
        nk_groups: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for m, n, k in shapes:
            nk_groups[(n, k)].append(m)

        assignment: Dict[int, List[Shape]] = {gid: [] for gid in gpu_ids}

        for (n, k), m_values in nk_groups.items():
            sorted_ms = sorted(set(m_values))
            for idx, m in enumerate(sorted_ms):
                gpu = gpu_ids[idx % len(gpu_ids)]
                assignment[gpu].append((m, n, k))

        return assignment

    # ------------------------------------------------------------------
    # Scout shape selection
    # ------------------------------------------------------------------

    def select_scout_shapes(
        self,
        shapes: List[Shape],
        fraction: float = 0.15,
    ) -> List[Shape]:
        """Pick a representative subset of shapes for scouting.

        For each unique ``(N, K)`` pair, select the smallest M, the middle M,
        and the largest M from the sorted list of M values.  Duplicates (when
        there are fewer than 3 distinct M values) are removed.

        Parameters
        ----------
        shapes:
            Full list of ``(M, N, K)`` tuples.
        fraction:
            Target fraction of the total shapes to return.  Not used for
            selection logic — kept as a parameter for API compatibility with
            future sampling strategies.

        Returns
        -------
        List[Shape]
            Deduplicated list of selected ``(M, N, K)`` tuples preserving
            the order: smallest M, middle M, largest M per ``(N, K)`` pair.
        """
        nk_groups: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for m, n, k in shapes:
            nk_groups[(n, k)].append(m)

        selected: List[Shape] = []
        seen = set()

        for (n, k), m_values in sorted(nk_groups.items()):
            sorted_ms = sorted(set(m_values))
            if not sorted_ms:
                continue

            candidates: List[int] = []
            # Smallest.
            candidates.append(sorted_ms[0])
            # Middle.
            mid = sorted_ms[len(sorted_ms) // 2]
            if mid not in candidates:
                candidates.append(mid)
            # Largest.
            largest = sorted_ms[-1]
            if largest not in candidates:
                candidates.append(largest)

            for m in candidates:
                key = (m, n, k)
                if key not in seen:
                    seen.add(key)
                    selected.append(key)

        return selected

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def _execute(self) -> dict:
        """Launch screen.py per GPU for each (N, K) pair and monitor progress.

        Steps
        -----
        1. Group shapes by ``(N, K)``.
        2. For each ``(N, K)`` pair:
           a. Distribute M values across GPUs.
           b. Build and launch a screen.py command per GPU via
              :meth:`~tuning_agent.remote.RemoteExecutor.docker_exec`.
           c. Attach a :class:`~tuning_agent.watchdog.RemoteProgressMonitor`
              to watch the remote log file.
        3. Return a summary dict.

        Returns
        -------
        dict
            ``{"pairs_tuned": <int>, "log_dir": <str>}``
        """
        # Group shapes by (N, K).
        nk_groups: Dict[Tuple[int, int], List[Shape]] = defaultdict(list)
        for shape in self.shapes_to_tune:
            m, n, k = shape
            nk_groups[(n, k)].append(shape)

        log_dir = os.path.join(self.artifact_dir, "logs")
        pairs_tuned = 0

        for (n, k), group_shapes in nk_groups.items():
            gpu_assignments = self.distribute_shapes_to_gpus(group_shapes, self.gpu_ids)

            for gpu_id, assigned_shapes in gpu_assignments.items():
                if not assigned_shapes:
                    continue

                for shape in assigned_shapes:
                    m_val, n_val, k_val = shape

                    # Determine appropriate search space for this M value.
                    search_space_for_m = self._get_search_space_for_m(m_val)

                    cmd = self.build_screen_command(
                        m_val, n_val, k_val, gpu_id, search_space_for_m
                    )

                    log_path = os.path.join(
                        log_dir, f"screen_M{m_val}_N{n_val}_K{k_val}_G{gpu_id}.log"
                    )

                    # Launch inside container with HIP_VISIBLE_DEVICES set via
                    # the GPU argument passed to screen.py.
                    self.executor.docker_exec(
                        cmd,
                        workdir=self.tunning_dir,
                        check=False,
                    )

                    # Attach a monitor to the remote log file.  The monitor
                    # is instantiated here so callers / orchestrators can
                    # retrieve it; the first poll is deferred to avoid
                    # blocking before the screen.py process has started.
                    RemoteProgressMonitor(
                        executor=self.executor,
                        remote_log_path=log_path,
                    )

            pairs_tuned += 1

        return {"pairs_tuned": pairs_tuned, "log_dir": log_dir}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_search_space_for_m(self, m: int) -> Dict[str, Any]:
        """Return the best-matching search space entry for the given M value.

        The search_space dict is keyed by bucket names like ``"M_LEQ_16"``
        or ``"M_LEQ_256"``.  This method picks the bucket whose numeric
        threshold is the smallest one that is still >= *m*.  Falls back to
        the last bucket if none match (largest M range).

        Parameters
        ----------
        m:
            The M dimension to look up.

        Returns
        -------
        dict
            The hyperparameter dict for the matching bucket, or an empty dict
            if :attr:`search_space` is empty.
        """
        if not self.search_space:
            return {}

        # Try to parse bucket thresholds from keys like "M_LEQ_<N>".
        buckets: List[Tuple[int, str]] = []
        other_keys: List[str] = []
        for key in self.search_space:
            upper = key.upper()
            if "LEQ" in upper or "LE" in upper:
                # Extract trailing integer.
                parts = key.replace("_", " ").split()
                for part in reversed(parts):
                    if part.isdigit():
                        buckets.append((int(part), key))
                        break
                else:
                    other_keys.append(key)
            else:
                other_keys.append(key)

        if buckets:
            buckets.sort(key=lambda t: t[0])
            for threshold, key in buckets:
                if m <= threshold:
                    return self.search_space[key]
            # M exceeds all thresholds — use the largest bucket.
            return self.search_space[buckets[-1][1]]

        # No threshold-based keys; return the first entry.
        first_key = next(iter(self.search_space))
        return self.search_space[first_key]
