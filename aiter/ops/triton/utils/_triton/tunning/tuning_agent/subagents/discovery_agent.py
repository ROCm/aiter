"""DiscoveryAgent — kernel & configuration discovery phase of the tuning pipeline.

Responsibilities
----------------
- Scan the kernel source directory to identify all tunable Triton kernel files
  and their associated ``@triton.autotune`` config blocks.
- Locate existing configuration JSON files in *config_dir* and map them to the
  discovered kernels so downstream agents know what already exists.
- Parse ``model_shapes.json`` (if present) to extract the set of (M, N, K) or
  other shape tuples that must be covered by the tuning run.
- Detect available benchmark / unit-test / smoke-test scripts so the
  ScriptCreatorAgent knows which ones need to be generated.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from ..remote import RemoteExecutor
from .base import BaseSubagent, SubagentResult

logger = logging.getLogger(__name__)

# M values to sweep for each (N, K) pair.
_DEFAULT_M_VALUES: List[int] = [8, 16, 32, 64, 128, 256, 512, 8192]


def _parse_ls_output(stdout: str) -> List[str]:
    """Parse the output of an ``ls`` command into a list of non-empty paths.

    Parameters
    ----------
    stdout:
        Raw stdout string from a remote ``ls`` command.  Lines may be
        whitespace-only (e.g. when the glob matched nothing but the shell
        printed a trailing newline) and are silently skipped.

    Returns
    -------
    List[str]
        Stripped, non-empty path strings.
    """
    paths = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped:
            paths.append(stripped)
    return paths


def _extract_nk_from_filename(filename: str) -> Optional[Tuple[int, int]]:
    """Extract the ``N`` and ``K`` dimensions encoded in *filename*.

    Config files that cover a single (N, K) bucket follow the naming
    convention ``<prefix>-N=<n>-K=<k>.json``.  Files that cover all shapes
    (i.e. a *catch-all* config) do not contain this pattern and return
    ``None``.

    Parameters
    ----------
    filename:
        Basename or full path of the config file.

    Returns
    -------
    Optional[Tuple[int, int]]
        ``(N, K)`` as integers, or ``None`` if the pattern is absent.
    """
    match = re.search(r"N=(\d+)-K=(\d+)", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def _parse_model_shapes(json_content: str, kernel_name: str) -> List[Tuple[int, int]]:
    """Parse ``model_shapes.json`` and return (N, K) pairs for *kernel_name*.

    The file has the structure::

        {
            "<ModelName>": {
                "<kernel_name>": [
                    {"N": <int>, "K": <int>, ...},
                    ...
                ],
                ...
            },
            ...
        }

    All entries whose key matches *kernel_name* (across all model sections)
    are collected, deduplicated, and returned.

    Parameters
    ----------
    json_content:
        Raw JSON string of the model-shapes file.
    kernel_name:
        Kernel identifier to filter entries (e.g. ``"gemm_afp4wfp4"``).

    Returns
    -------
    List[Tuple[int, int]]
        Deduplicated list of ``(N, K)`` integer pairs.
    """
    try:
        data: Dict[str, Any] = json.loads(json_content)
    except json.JSONDecodeError:
        logger.warning("model_shapes.json is not valid JSON — skipping")
        return []

    seen: Set[Tuple[int, int]] = set()
    pairs: List[Tuple[int, int]] = []

    for _model_name, kernels in data.items():
        if not isinstance(kernels, dict):
            continue
        shapes_list = kernels.get(kernel_name)
        if not shapes_list:
            continue
        for entry in shapes_list:
            if not isinstance(entry, dict):
                continue
            n = entry.get("N")
            k = entry.get("K")
            if n is not None and k is not None:
                pair = (int(n), int(k))
                if pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)

    return pairs


class DiscoveryAgent(BaseSubagent):
    """Discover kernel sources, existing configs, and required shapes.

    Parameters
    ----------
    executor:
        :class:`~tuning_agent.remote.RemoteExecutor` connected to the target
        machine.
    kernel_name:
        Short identifier for the kernel being tuned (e.g. ``"gemm_afp4wfp4"``).
    artifact_dir:
        Absolute path on the remote host where JSON artifacts are written.
    config_variant:
        Config-file variant string (e.g. ``"A8W8"``, ``"AFP4WFP4"``).  Used
        to filter config files by filename.
    gfx_arch:
        GPU architecture string (e.g. ``"gfx942"``, ``"gfx950"``).  Used as
        the prefix when scanning for existing config files.
    kernel_source_path:
        Absolute path on the remote host to the directory (or single file)
        containing the Triton kernel source(s) to be tuned.
    config_dir:
        Absolute path on the remote host where existing tuning config JSON
        files live (output of previous tuning runs).
    model_shapes_path:
        Absolute path on the remote host to ``model_shapes.json``.  May be
        ``None`` if no model-specific shapes are required.
    expected_triton_commit:
        Forwarded to :class:`~.base.BaseSubagent`.
    expected_aiter_branch:
        Forwarded to :class:`~.base.BaseSubagent`.
    """

    name: str = "discovery"

    def __init__(
        self,
        executor: RemoteExecutor,
        kernel_name: str,
        artifact_dir: str,
        config_variant: str,
        gfx_arch: str,
        kernel_source_path: str = "",
        config_dir: str = "",
        model_shapes_path: Optional[str] = None,
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
        self.config_variant = config_variant
        self.gfx_arch = gfx_arch
        self.kernel_source_path = kernel_source_path
        self.config_dir = config_dir
        self.model_shapes_path = model_shapes_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scan_config_files(self) -> Tuple[List[str], List[Tuple[int, int]]]:
        """Scan for existing config JSON files and extract (N, K) pairs.

        Issues a ``ls`` inside the container targeting::

            /workspace/aiter/aiter/ops/triton/configs/gemm/{gfx_arch}-*{config_variant}*.json

        Returns
        -------
        Tuple[List[str], List[Tuple[int, int]]]
            ``(config_files, nk_pairs)`` — the list of matched config file
            paths and the deduplicated (N, K) pairs extracted from their
            filenames and contents.
        """
        pattern = (
            f"/workspace/aiter/aiter/ops/triton/configs/gemm/"
            f"{self.gfx_arch}-*{self.config_variant}*.json"
        )
        result = self.executor.docker_exec(f"ls {pattern}", check=False)
        config_files = _parse_ls_output(result.stdout)

        seen_nk: Set[Tuple[int, int]] = set()
        nk_pairs: List[Tuple[int, int]] = []

        for cfg_path in config_files:
            # First try to get (N, K) directly from the filename.
            nk = _extract_nk_from_filename(cfg_path)
            if nk is not None:
                if nk not in seen_nk:
                    seen_nk.add(nk)
                    nk_pairs.append(nk)
                continue

            # For catch-all configs (no N=X-K=Y in name), read the file and
            # inspect the M_LEQ bucket keys for structural hints, but we
            # cannot derive N/K from them — skip shape extraction here.
            try:
                cat_result = self.executor.docker_exec(f"cat {cfg_path}", check=True)
                data = json.loads(cat_result.stdout)
                # The JSON structure is { "M_LEQ_<n>": {...}, "any": {...} }.
                # We cannot derive N/K from this alone, so we just validate the
                # JSON is parseable and move on.
                logger.debug("Parsed catch-all config %s with keys: %s", cfg_path, list(data.keys()))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to parse config file %s: %s", cfg_path, exc)

        return config_files, nk_pairs

    def _load_model_shapes(self) -> List[Tuple[int, int]]:
        """Load model_shapes.json from the container and extract (N, K) pairs.

        Issues a ``cat`` inside the container targeting::

            /workspace/aiter/op_tests/op_benchmarks/triton/model_benchmarking_tool/model_shapes.json

        Returns an empty list (and logs a warning) if the command fails or
        the file cannot be parsed.

        Returns
        -------
        List[Tuple[int, int]]
            Deduplicated (N, K) pairs for :attr:`kernel_name`.
        """
        shapes_path = (
            self.model_shapes_path
            or "/workspace/aiter/op_tests/op_benchmarks/triton/"
               "model_benchmarking_tool/model_shapes.json"
        )
        result = self.executor.docker_exec(f"cat {shapes_path}", check=False)
        if result.returncode != 0 or not result.stdout.strip():
            logger.warning(
                "model_shapes.json not found or empty at %s — skipping", shapes_path
            )
            return []
        return _parse_model_shapes(result.stdout, self.kernel_name)

    def _build_shapes(self, nk_pairs: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
        """Expand (N, K) pairs into full (M, N, K) triples.

        For each unique (N, K) pair, creates one entry per M value in
        :data:`_DEFAULT_M_VALUES`.

        Parameters
        ----------
        nk_pairs:
            Deduplicated (N, K) pairs.

        Returns
        -------
        List[Tuple[int, int, int]]
            All (M, N, K) triples to be benchmarked / tuned.
        """
        shapes: List[Tuple[int, int, int]] = []
        for n, k in nk_pairs:
            for m in _DEFAULT_M_VALUES:
                shapes.append((m, n, k))
        return shapes

    def _detect_scripts(self) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
        """Detect existing benchmark, unit-test, and test scripts.

        Searches three standard locations inside the container:

        * **ut script** — ``ut_*{kernel_name}*.py`` in the tunning directory.
        * **bench script** — ``bench_gemm_{kernel_name}*.py`` in the op-benchmarks directory.
        * **test script** — ``test_gemm_{kernel_name}*.py`` under the gemm test tree.

        Returns
        -------
        Tuple[Optional[str], Optional[str], Optional[str], List[str]]
            ``(ut_script, bench_script, test_script, missing_scripts)``
            where each ``*_script`` is a path string or ``None``, and
            ``missing_scripts`` lists the string identifiers (``"ut"``,
            ``"bench"``, ``"test"``) for scripts that were not found.
        """
        def _first_hit(command: str) -> Optional[str]:
            result = self.executor.docker_exec(command, check=False)
            paths = _parse_ls_output(result.stdout)
            return paths[0] if paths else None

        ut_script = _first_hit(
            f"ls /workspace/aiter/aiter/ops/triton/utils/_triton/tunning/"
            f"ut_*{self.kernel_name}*.py"
        )
        bench_script = _first_hit(
            f"ls /workspace/aiter/op_tests/op_benchmarks/triton/"
            f"bench_gemm_{self.kernel_name}*.py"
        )
        test_script = _first_hit(
            f"ls /workspace/aiter/op_tests/triton_tests/gemm/"
            f"*/test_gemm_{self.kernel_name}*.py"
        )

        missing_scripts: List[str] = []
        if ut_script is None:
            missing_scripts.append("ut")
        if bench_script is None:
            missing_scripts.append("bench")
        if test_script is None:
            missing_scripts.append("test")

        return ut_script, bench_script, test_script, missing_scripts

    # ------------------------------------------------------------------
    # _execute implementation
    # ------------------------------------------------------------------

    def _execute(self) -> dict:
        """Scan sources, configs, shapes, and test scripts on the remote host.

        Steps
        -----
        1. Scan for existing config JSON files and extract (N, K) pairs from
           their filenames / contents.
        2. Load ``model_shapes.json`` and extract additional (N, K) pairs for
           this kernel.
        3. Merge and deduplicate all (N, K) pairs, then build the full list of
           (M, N, K) shapes using the default M sweep.
        4. Detect available ut / bench / test scripts and record which are
           missing.
        5. Persist a ``discovery.json`` artifact and return the result dict.

        Returns
        -------
        dict
            Keys: ``"shapes"``, ``"config_files"``, ``"nk_pairs"``,
            ``"ut_script"``, ``"bench_script"``, ``"test_script"``,
            ``"missing_scripts"``.
        """
        # Step 1: config files + N/K pairs from filenames.
        config_files, nk_from_configs = self._scan_config_files()

        # Step 2: N/K pairs from model_shapes.json.
        nk_from_model = self._load_model_shapes()

        # Step 3: merge & deduplicate, preserving insertion order.
        seen_nk: Set[Tuple[int, int]] = set()
        all_nk: List[Tuple[int, int]] = []
        for pair in nk_from_configs + nk_from_model:
            if pair not in seen_nk:
                seen_nk.add(pair)
                all_nk.append(pair)

        shapes = self._build_shapes(all_nk)

        # Step 4: script detection.
        ut_script, bench_script, test_script, missing_scripts = self._detect_scripts()

        result: Dict[str, Any] = {
            "shapes": shapes,
            "config_files": config_files,
            "nk_pairs": all_nk,
            "ut_script": ut_script,
            "bench_script": bench_script,
            "test_script": test_script,
            "missing_scripts": missing_scripts,
        }

        # Step 5: persist artifact.
        # Shapes contain tuples, which are not JSON-serialisable — convert to lists.
        serialisable = dict(result)
        serialisable["shapes"] = [list(s) for s in shapes]
        serialisable["nk_pairs"] = [list(p) for p in all_nk]
        self._write_json_artifact("discovery.json", serialisable)

        return result
