"""ScriptCreatorAgent — generate missing benchmark/test scripts for the tuning pipeline.

Responsibilities
----------------
- Read existing benchmark, unit-test, and smoke-test scripts as templates to
  understand project conventions (argument parsing, timing loops, etc.).
- Read the kernel source to extract the public API (function signatures,
  argument names and dtypes, expected tensor shapes).
- Generate any scripts identified as missing by DiscoveryAgent, filling in
  kernel-specific details while preserving the template structure.
- Run a quick smoke test of each newly generated script to confirm it imports
  and executes without error before handing off to the tuning phases.
"""

from typing import Any, Dict, List, Optional

from ..remote import RemoteExecutor
from .base import BaseSubagent, SubagentResult


class ScriptCreatorAgent(BaseSubagent):
    """Generate missing benchmark and test scripts from templates.

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
        Absolute path on the remote host to the kernel source file(s).  Used
        to extract function signatures and argument metadata.
    template_dir:
        Absolute path on the remote host to the directory containing existing
        scripts to use as templates.
    missing_scripts:
        List of script descriptors (as produced by DiscoveryAgent) describing
        which scripts need to be created.  Each element is expected to be a
        dict with at least ``"type"`` (``"bench"`` | ``"ut"`` | ``"smoke"``)
        and ``"target_path"`` keys.
    expected_triton_commit:
        Forwarded to :class:`~.base.BaseSubagent`.
    expected_aiter_branch:
        Forwarded to :class:`~.base.BaseSubagent`.
    """

    name: str = "script_creator"

    def __init__(
        self,
        executor: RemoteExecutor,
        kernel_name: str,
        artifact_dir: str,
        kernel_source_path: str,
        template_dir: str,
        missing_scripts: List[Dict[str, Any]],
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
        self.template_dir = template_dir
        self.missing_scripts = missing_scripts

    def _execute(self) -> dict:
        """Generate missing scripts and validate them on the remote host.

        TODO
        ----
        Implement the following steps in order:

        1. **Read template scripts**
           - For each script type needed (bench/ut/smoke), identify the best
             matching existing script in *template_dir* using
             ``self.executor.ssh_run("ls <template_dir>")``.
           - Fetch the content of each template via
             ``self.executor.ssh_run("cat <path>")``.
           - Parse the template to identify placeholder tokens (e.g.
             ``{{KERNEL_NAME}}``, ``{{ARGS}}``, ``{{DTYPE}}``).

        2. **Extract kernel API from source**
           - Fetch the kernel source file(s) via ssh cat.
           - Use regex or AST parsing to locate the primary kernel wrapper
             function(s): extract parameter names, type annotations, and any
             ``tl.constexpr`` arguments.
           - Build a substitution context dict mapping template tokens to their
             kernel-specific values.

        3. **Generate missing scripts**
           - For each entry in *missing_scripts*:
             a. Select the appropriate template for the script type.
             b. Apply the substitution context to produce the script content.
             c. Write the rendered content to ``entry["target_path"]`` on the
                remote host via a here-doc ssh command.
             d. Make the file executable (``chmod +x``).

        4. **Smoke-test generated scripts**
           - For each newly created script, run a quick validation:
               ``python <script_path> --help``  (or ``--smoke`` if supported)
             to confirm it parses arguments without error.
           - Also attempt a single-iteration dry run with a small representative
             shape to confirm the kernel can be imported and launched.
           - Collect pass/fail status per script.

        5. **Write artifact & return**
           - Persist results via
             ``self._write_json_artifact("script_creation.json", ...)``.
           - Return a dict with keys: ``"created_scripts"``, ``"smoke_results"``,
             ``"failed_scripts"`` (scripts that did not pass smoke test).
        """
        # TODO: implement steps above
        return {"status": "not_implemented"}
