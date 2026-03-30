"""SetupAgent — environment bootstrap phase of the Triton kernel tuning pipeline.

Responsibilities
----------------
- Pull (or build) a Docker/Singularity container image on the target host.
- Clone the aiter repository at the requested branch/commit inside the
  container so that the kernel source is available for subsequent agents.
- Install the pinned version of Triton (wheel or source build) according to
  *triton_install_config*.
- Run a lightweight smoke-test to verify that ``import triton`` succeeds and
  that the GPU runtime is accessible before handing off to the next phase.
"""

from typing import Any, Dict, List, Optional

from ..remote import RemoteExecutor
from .base import BaseSubagent, SubagentResult


class SetupAgent(BaseSubagent):
    """Bootstrap the remote execution environment for kernel tuning.

    Parameters
    ----------
    executor:
        :class:`~tuning_agent.remote.RemoteExecutor` connected to the target
        machine.
    kernel_name:
        Short identifier for the kernel being tuned (e.g. ``"fmha"``).
    artifact_dir:
        Absolute path on the remote host where JSON artifacts are written.
    image:
        Container image reference (e.g. a Docker tag or a path to a
        Singularity ``.sif`` file) to pull/use on the remote host.
    run_script:
        Path to the shell wrapper used to execute commands inside the
        container (e.g. ``"docker run --rm ..."``) or a helper script that
        sets up the container runtime environment.
    repo_config:
        Mapping describing how to check out the aiter repository.  Expected
        keys include ``"url"``, ``"branch"``, and optionally ``"commit"``.
    triton_install_config:
        Mapping describing how Triton should be installed.  Expected keys
        include ``"method"`` (``"pip"`` | ``"source"``), ``"version"`` /
        ``"commit"``, and optionally ``"extra_index_url"``.
    expected_triton_commit:
        When set, :meth:`_preflight` asserts the installed Triton commit
        matches this string after installation.
    expected_aiter_branch:
        When set, :meth:`_preflight` asserts the aiter git branch matches.
    """

    name: str = "setup"

    def __init__(
        self,
        executor: RemoteExecutor,
        kernel_name: str,
        artifact_dir: str,
        image: str,
        run_script: str,
        repo_config: Dict[str, Any],
        triton_install_config: Dict[str, Any],
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
        self.image = image
        self.run_script = run_script
        self.repo_config = repo_config
        self.triton_install_config = triton_install_config

    def _execute(self) -> dict:
        """Bootstrap the container environment on the remote host.

        TODO
        ----
        Implement the following steps in order:

        1. **Pull / build container image**
           - If *image* is a Docker tag, run ``docker pull <image>`` (or
             ``singularity pull`` for .sif files) via ``self.executor.ssh_run``.
           - Verify the pull succeeded; raise ``SubagentError`` on failure.

        2. **Clone aiter repository**
           - Inside the container (via *run_script*), run:
               ``git clone --branch <branch> <url> /workspace/aiter``
           - If a specific commit is requested in *repo_config*, checkout that
             commit after cloning.
           - Confirm the clone by checking ``/workspace/aiter/setup.py`` (or
             equivalent) exists.

        3. **Install Triton**
           - Read *triton_install_config["method"]*:
             - ``"pip"``: run ``pip install triton==<version>`` (optionally
               with ``--extra-index-url``).
             - ``"source"``: clone the Triton repo at the specified commit,
               then run ``pip install -e python/``.
           - Capture and log stdout/stderr from the install command.

        4. **Verify environment**
           - Run a one-liner smoke test inside the container:
               ``python -c "import triton; import torch; torch.cuda.is_available()"``
           - Parse the output; raise ``SubagentError`` if the check fails.
           - Call ``self.executor.verify_environment`` to cross-check commit /
             branch expectations if they were supplied.

        5. **Write artifact**
           - Collect versions (triton, torch, Python, driver) into a dict.
           - Persist via ``self._write_json_artifact("setup_info.json", ...)``.
           - Return the dict so the orchestrator can log it.
        """
        # TODO: implement steps above
        return {"status": "not_implemented"}
