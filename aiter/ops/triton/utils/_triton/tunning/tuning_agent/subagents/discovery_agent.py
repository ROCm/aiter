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

from typing import Any, Dict, List, Optional

from ..remote import RemoteExecutor
from .base import BaseSubagent, SubagentResult


class DiscoveryAgent(BaseSubagent):
    """Discover kernel sources, existing configs, and required shapes.

    Parameters
    ----------
    executor:
        :class:`~tuning_agent.remote.RemoteExecutor` connected to the target
        machine.
    kernel_name:
        Short identifier for the kernel being tuned (e.g. ``"fmha"``).
    artifact_dir:
        Absolute path on the remote host where JSON artifacts are written.
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
        kernel_source_path: str,
        config_dir: str,
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
        self.kernel_source_path = kernel_source_path
        self.config_dir = config_dir
        self.model_shapes_path = model_shapes_path

    def _execute(self) -> dict:
        """Scan sources, configs, shapes, and test scripts on the remote host.

        TODO
        ----
        Implement the following steps in order:

        1. **Scan kernel source**
           - Use ``self.executor.ssh_run("find <kernel_source_path> -name '*.py'")``
             to enumerate Python files.
           - For each file, grep for ``@triton.autotune`` or ``triton.Config``
             to identify tunable kernels and capture their config key lists
             (e.g. ``BLOCK_M``, ``BLOCK_N``, ``num_warps``, ``num_stages``).
           - Build a list of dicts: ``[{"file": ..., "kernel_fn": ...,
             "config_keys": [...]}]``.

        2. **Find existing config files**
           - Run ``ls <config_dir>/*.json`` (handle missing dir gracefully).
           - For each JSON filename, parse the kernel variant / M-range it
             covers (by filename convention, e.g.
             ``fmha_fwd_d128_leq256.json``).
           - Build a mapping ``{variant: [config_file_paths]}`` so downstream
             agents can skip re-generating what already exists.

        3. **Extract shapes from model_shapes.json**
           - If *model_shapes_path* is set, read and parse the file via
             ``self._read_json_artifact`` (or a direct ssh cat + json.loads).
           - Extract all unique shape tuples; normalise to a canonical list of
             dicts ``[{"M": int, "N": int, "K": int, ...}]``.
           - If the file is absent, return an empty shapes list and log a
             warning (do not raise).

        4. **Detect benchmark / test scripts**
           - Search common locations (``tests/``, ``benchmarks/``, ``ut/``) for
             scripts matching patterns like ``bench_<kernel>.py``,
             ``test_<kernel>.py``, ``ut_<kernel>.py``.
           - Classify each found script as one of: ``"bench"``, ``"ut"``,
             ``"smoke"``.
           - Build a list of missing script types needed by the pipeline (to be
             created by ScriptCreatorAgent).

        5. **Write artifact & return**
           - Persist the full discovery result via
             ``self._write_json_artifact("discovery.json", ...)``.
           - Return a dict with keys: ``"kernels"``, ``"existing_configs"``,
             ``"shapes"``, ``"found_scripts"``, ``"missing_scripts"``.
        """
        # TODO: implement steps above
        return {"status": "not_implemented"}
