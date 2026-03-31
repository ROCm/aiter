"""ConfigGeneratorAgent — produce final tuning config JSON files from tuning logs.

Responsibilities
----------------
- Collect all tuned (N, K) pairs from screen-*.log filenames in tuning_logs_dir.
- Invoke ``view-screen.py`` for each (N, K) pair to generate a per-shape JSON
  config file inside the tunning directory.
- Apply kernel-specific naming overrides (e.g. rename "M_LEQ_16" to a custom
  bucket name for preshuffled kernels).
- Copy generated JSON config files to the aiter config store (config_dir).
- Also generate/update the default fallback config (unsuffixed) using the most
  common (N, K) pair, updating only the "any" bucket.
"""

import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from ..remote import RemoteExecutor
from .base import BaseSubagent, SubagentError


# Regex to parse filenames like:
#   screen-ut_a8w8_gemm.py-M-N-K.log
# Group 1: ut_script basename (e.g. "ut_a8w8_gemm.py")
# Group 2: M value
# Group 3: N value
# Group 4: K value
_LOG_FILENAME_RE = re.compile(
    r"^screen-(.+\.py)-(\d+)-(\d+)-(\d+)\.log$"
)


def _parse_log_filenames(ls_output: str) -> List[Tuple[int, int]]:
    """Extract unique (N, K) pairs from a newline-separated list of log file paths.

    Each path is expected to contain a filename matching the pattern::

        screen-<ut_script>-<M>-<N>-<K>.log

    Parameters
    ----------
    ls_output:
        Raw stdout from ``ls {tuning_logs_dir}/screen-*.log``.

    Returns
    -------
    List[Tuple[int, int]]
        Deduplicated list of ``(N, K)`` integer pairs extracted from the
        filenames, preserving first-seen order.
    """
    seen: set = set()
    result: List[Tuple[int, int]] = []
    for line in ls_output.splitlines():
        filename = os.path.basename(line.strip())
        m = _LOG_FILENAME_RE.match(filename)
        if m:
            n = int(m.group(3))
            k = int(m.group(4))
            pair = (n, k)
            if pair not in seen:
                seen.add(pair)
                result.append(pair)
    return result


def _most_common_nk(ls_output: str) -> Optional[Tuple[int, int]]:
    """Return the (N, K) pair that appears most frequently in the log filenames.

    When multiple pairs share the same count the one with the largest
    combined occurrence count (ties broken by first-seen order) is returned.

    Parameters
    ----------
    ls_output:
        Raw stdout from ``ls {tuning_logs_dir}/screen-*.log``.

    Returns
    -------
    Optional[Tuple[int, int]]
        The most-common ``(N, K)`` pair, or ``None`` when *ls_output* contains
        no matching filenames.
    """
    counter: Counter = Counter()
    for line in ls_output.splitlines():
        filename = os.path.basename(line.strip())
        m = _LOG_FILENAME_RE.match(filename)
        if m:
            n = int(m.group(3))
            k = int(m.group(4))
            counter[(n, k)] += 1
    if not counter:
        return None
    return counter.most_common(1)[0][0]


class ConfigGeneratorAgent(BaseSubagent):
    """Generate final per-(N, K) config JSON files from raw tuning logs.

    Parameters
    ----------
    executor:
        :class:`~tuning_agent.remote.RemoteExecutor` connected to the target
        machine.
    kernel_name:
        Short identifier for the kernel being tuned (e.g. ``"gemm_a8w8"``).
    artifact_dir:
        Absolute path on the remote host where JSON artifacts are written.
    tuning_logs_dir:
        Absolute path on the remote host (inside the container) where raw
        ``screen-*.log`` tuning log files are stored.
    config_dir:
        Absolute path (inside the container) to the aiter config store where
        generated JSON files should be copied
        (e.g. ``/workspace/aiter/aiter/ops/triton/configs/gemm/``).
    kernel_variant:
        String identifying the kernel variant, e.g. ``"A8W8"`` or
        ``"AFP4WFP4_PRESHUFFLED"``.  Used to match and copy generated JSON
        files.
    ut_script:
        Basename of the unit-test script passed to ``view-screen.py``
        (e.g. ``"ut_a8w8_gemm.py"``).
    gfx_arch:
        GFX architecture string, e.g. ``"gfx950"``.  Used to match generated
        JSON filenames (``{gfx_arch}-GEMM-{variant}*.json``).
    tunning_dir:
        Absolute path (inside the container) to the directory that contains
        ``view-screen.py`` and ``screen-*.log`` files.  When *None*, defaults
        to *tuning_logs_dir*.
    m_leq_16_rename:
        When set, the ``"M_LEQ_16"`` key in every generated JSON config is
        renamed to this string before the file is written back.  Useful for
        kernels such as preshuffled AFP4WFP4 where the bucket covering
        M ≤ 31 is named ``"M_LEQ_31"``.
    expected_triton_commit:
        Forwarded to :class:`~.base.BaseSubagent`.
    expected_aiter_branch:
        Forwarded to :class:`~.base.BaseSubagent`.
    """

    name: str = "config_generator"

    def __init__(
        self,
        executor: RemoteExecutor,
        kernel_name: str,
        artifact_dir: str,
        tuning_logs_dir: str,
        config_dir: str,
        kernel_variant: str,
        ut_script: str,
        gfx_arch: str,
        tunning_dir: Optional[str] = None,
        m_leq_16_rename: Optional[str] = None,
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
        self.tuning_logs_dir = tuning_logs_dir
        self.config_dir = config_dir
        self.kernel_variant = kernel_variant
        self.ut_script = ut_script
        self.gfx_arch = gfx_arch
        self.tunning_dir = tunning_dir if tunning_dir is not None else tuning_logs_dir
        self.m_leq_16_rename = m_leq_16_rename

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_nk_pairs(self) -> Tuple[List[Tuple[int, int]], str]:
        """List log files and extract unique (N, K) pairs.

        Returns
        -------
        Tuple[List[Tuple[int, int]], str]
            ``(nk_pairs, ls_output)`` where *nk_pairs* is the deduplicated
            list of ``(N, K)`` pairs and *ls_output* is the raw stdout from
            ``ls``.

        Raises
        ------
        SubagentError
            When ``ls`` fails or returns no matching files.
        """
        ls_result = self.executor.docker_exec(
            f"ls {self.tuning_logs_dir}/screen-*.log",
            check=False,
        )
        if ls_result.returncode != 0 or not ls_result.stdout.strip():
            raise SubagentError(
                f"No screen-*.log files found in {self.tuning_logs_dir}: "
                f"{ls_result.stderr.strip()}"
            )
        nk_pairs = _parse_log_filenames(ls_result.stdout)
        return nk_pairs, ls_result.stdout

    def _run_view_screen(self, n: int, k: int) -> None:
        """Run view-screen.py for a single (N, K) pair.

        Parameters
        ----------
        n, k:
            Matrix dimensions whose tuning results should be aggregated.
        """
        cmd = (
            f"cd {self.tunning_dir} && "
            f"python view-screen.py {self.ut_script} --n-list {n} --k-list {k}"
        )
        self.executor.docker_exec(cmd, workdir=self.tunning_dir, check=False)

    def _apply_m_leq_16_rename(self, json_path: str) -> None:
        """Rename the ``"M_LEQ_16"`` key in a generated JSON file.

        The file is read back from the container via ``cat``, the key is
        renamed in-memory, and the result is written back via a ``printf``
        here-string.

        Parameters
        ----------
        json_path:
            Absolute path (inside the container) to the JSON config file to
            patch.
        """
        cat_result = self.executor.docker_exec(f"cat {json_path}", check=True)
        try:
            config: Dict[str, Any] = json.loads(cat_result.stdout)
        except json.JSONDecodeError as exc:
            raise SubagentError(
                f"Failed to parse generated config {json_path!r}: {exc}"
            ) from exc

        if "M_LEQ_16" in config:
            # Rebuild the dict preserving insertion order but renaming the key.
            new_config: Dict[str, Any] = {}
            for key, value in config.items():
                new_key = self.m_leq_16_rename if key == "M_LEQ_16" else key
                new_config[new_key] = value
            json_content = json.dumps(new_config, indent=2)
            escaped = json_content.replace("'", "'\\''")
            self.executor.docker_exec(
                f"printf '%s\\n' '{escaped}' > {json_path}",
                check=True,
            )

    def _copy_configs_to_store(self) -> List[str]:
        """Copy all generated config JSON files to the config store.

        Returns
        -------
        List[str]
            Filenames (basenames) of the files that were copied.
        """
        glob_pattern = (
            f"{self.tunning_dir}/{self.gfx_arch}-GEMM-{self.kernel_variant}*.json"
        )
        cp_result = self.executor.docker_exec(
            f"cp {glob_pattern} {self.config_dir}/",
            check=False,
        )

        # Collect the filenames that were copied by listing them.
        ls_result = self.executor.docker_exec(
            f"ls {glob_pattern}",
            check=False,
        )
        copied: List[str] = []
        if ls_result.returncode == 0:
            for line in ls_result.stdout.splitlines():
                basename = os.path.basename(line.strip())
                if basename:
                    copied.append(basename)
        return copied

    def _update_default_fallback(self, ls_output: str) -> bool:
        """Create or update the unsuffixed default fallback config.

        Uses the most-common (N, K) pair's generated JSON as the source.
        Only the ``"any"`` bucket is copied into the default fallback.

        Parameters
        ----------
        ls_output:
            Raw stdout from the earlier ``ls {tuning_logs_dir}/screen-*.log``
            call, used to identify the most-common (N, K) pair.

        Returns
        -------
        bool
            ``True`` when the fallback was successfully written, ``False``
            otherwise.
        """
        best_pair = _most_common_nk(ls_output)
        if best_pair is None:
            return False
        n, k = best_pair

        # Locate the suffixed config file for this (N, K) pair.
        src_path = (
            f"{self.tunning_dir}/"
            f"{self.gfx_arch}-GEMM-{self.kernel_variant}-N={n}-K={k}.json"
        )
        cat_result = self.executor.docker_exec(f"cat {src_path}", check=False)
        if cat_result.returncode != 0 or not cat_result.stdout.strip():
            return False

        try:
            src_config: Dict[str, Any] = json.loads(cat_result.stdout)
        except json.JSONDecodeError:
            return False

        any_bucket = src_config.get("any")
        if any_bucket is None:
            return False

        # Read or initialise the fallback config.
        fallback_path = (
            f"{self.config_dir}/{self.gfx_arch}-GEMM-{self.kernel_variant}.json"
        )
        cat_fb = self.executor.docker_exec(f"cat {fallback_path}", check=False)
        if cat_fb.returncode == 0 and cat_fb.stdout.strip():
            try:
                fallback_config: Dict[str, Any] = json.loads(cat_fb.stdout)
            except json.JSONDecodeError:
                fallback_config = {}
        else:
            fallback_config = {}

        fallback_config["any"] = any_bucket

        json_content = json.dumps(fallback_config, indent=2)
        escaped = json_content.replace("'", "'\\''")
        write_result = self.executor.docker_exec(
            f"printf '%s\\n' '{escaped}' > {fallback_path}",
            check=False,
        )
        return write_result.returncode == 0

    # ------------------------------------------------------------------
    # _execute implementation
    # ------------------------------------------------------------------

    def _execute(self) -> dict:
        """Parse tuning logs and write config JSON files to the config store.

        Steps
        -----
        1. Collect all tuned (N, K) pairs from screen-*.log filenames.
        2. For each (N, K) pair, run view-screen.py to generate a JSON config.
        3. Apply M_LEQ_16 key rename when :attr:`m_leq_16_rename` is set.
        4. Copy all generated configs to the config store.
        5. Generate/update the default fallback config.

        Returns
        -------
        dict
            ``{"configs_generated": <int>, "config_files": [...],
               "default_fallback_updated": <bool>}``
        """
        # Step 1: collect (N, K) pairs from log filenames.
        nk_pairs, ls_output = self._collect_nk_pairs()

        # Step 2: run view-screen.py for each pair and optionally rename key.
        for n, k in nk_pairs:
            self._run_view_screen(n, k)

            # Step 3: apply M_LEQ_16 rename if requested.
            if self.m_leq_16_rename:
                json_path = (
                    f"{self.tunning_dir}/"
                    f"{self.gfx_arch}-GEMM-{self.kernel_variant}-N={n}-K={k}.json"
                )
                self._apply_m_leq_16_rename(json_path)

        # Step 4: copy configs to the config store.
        config_files = self._copy_configs_to_store()

        # Step 5: update the default fallback config.
        default_fallback_updated = self._update_default_fallback(ls_output)

        return {
            "configs_generated": len(nk_pairs),
            "config_files": config_files,
            "default_fallback_updated": default_fallback_updated,
        }
