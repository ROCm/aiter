"""Base class and result types for all tuning-pipeline subagents."""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..remote import RemoteExecutor


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class SubagentResult:
    """Outcome of a single subagent run.

    Attributes
    ----------
    success:
        ``True`` when the subagent completed without error.
    data:
        Arbitrary key/value payload produced by the subagent.
    error:
        Human-readable error description when *success* is ``False``.
    """

    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class SubagentError(Exception):
    """Raised by subagent internals to signal a non-retryable failure."""


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseSubagent(ABC):
    """Abstract base class for all tuning-pipeline subagents.

    Subclasses must implement :meth:`_execute` which performs the actual work
    and returns a plain :class:`dict` of results.  The public :meth:`run`
    method wraps the full lifecycle:

    1. **Preflight** — :meth:`_preflight`: creates the artifact directory on
       the remote host, kills any stale GPU processes, and optionally verifies
       the environment.
    2. **Execute** — :meth:`_execute`: subclass-defined logic.
    3. **Result wrapping** — exceptions are caught and surfaced as a
       :class:`SubagentResult` with ``success=False``.

    Parameters
    ----------
    executor:
        :class:`~tuning_agent.remote.RemoteExecutor` connected to the target
        machine.
    kernel_name:
        Short identifier for the kernel being tuned (e.g. ``"fmha"``).
    artifact_dir:
        Absolute path on the *remote* host where JSON artifacts are written.
    expected_triton_commit:
        When set, :meth:`_preflight` calls
        :meth:`~tuning_agent.remote.RemoteExecutor.verify_environment` and
        asserts that the Triton version matches this string.
    expected_aiter_branch:
        When set, :meth:`_preflight` asserts the aiter git branch matches.
    """

    #: Short identifier for this subagent type (override in subclasses).
    name: str = "base"

    def __init__(
        self,
        executor: RemoteExecutor,
        kernel_name: str,
        artifact_dir: str,
        expected_triton_commit: Optional[str] = None,
        expected_aiter_branch: Optional[str] = None,
    ) -> None:
        self.executor = executor
        self.kernel_name = kernel_name
        self.artifact_dir = artifact_dir
        self.expected_triton_commit = expected_triton_commit
        self.expected_aiter_branch = expected_aiter_branch

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> SubagentResult:
        """Execute the full subagent lifecycle and return a :class:`SubagentResult`.

        On any unhandled exception the method returns a failure result rather
        than propagating the exception, so callers can always inspect the
        ``success`` flag.
        """
        try:
            self._preflight()
            data = self._execute()
            return SubagentResult(success=True, data=data if data is not None else {})
        except Exception as exc:  # noqa: BLE001
            return SubagentResult(success=False, error=str(exc))

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def _preflight(self) -> None:
        """Prepare the remote environment before :meth:`_execute` runs.

        Actions performed (in order):

        1. ``mkdir -p <artifact_dir>`` on the remote host via SSH.
        2. Kill any GPU processes that might interfere with benchmarking.
        3. If either *expected_triton_commit* or *expected_aiter_branch* were
           supplied to the constructor, verify the container environment.
        """
        # Ensure artifact directory exists inside the container.
        self.executor.docker_exec(f"mkdir -p {self.artifact_dir}")

        # Free GPUs from any lingering processes.
        self.executor.kill_stale_gpu_processes()

        # Optional environment verification.
        if self.expected_triton_commit is not None or self.expected_aiter_branch is not None:
            self.executor.verify_environment(
                expected_triton_commit=self.expected_triton_commit,
                expected_aiter_branch=self.expected_aiter_branch,
            )

    @abstractmethod
    def _execute(self) -> dict:
        """Subclass-defined work.

        Returns
        -------
        dict
            Arbitrary payload that will be stored in
            :attr:`SubagentResult.data`.
        """

    # ------------------------------------------------------------------
    # Artifact helpers
    # ------------------------------------------------------------------

    def _write_json_artifact(self, filename: str, data: Any) -> str:
        """Serialize *data* as JSON and write it to the remote artifact directory.

        Parameters
        ----------
        filename:
            Base filename (e.g. ``"results.json"``).
        data:
            JSON-serialisable object.

        Returns
        -------
        str
            The full remote path where the file was written.
        """
        remote_path = os.path.join(self.artifact_dir, filename)
        json_content = json.dumps(data, indent=2)
        # Write via a shell here-string to avoid creating a local temp file.
        escaped = json_content.replace("'", "'\\''")
        self.executor.docker_exec(
            f"printf '%s' '{escaped}' > {remote_path}",
            check=True,
        )
        return remote_path

    def _read_json_artifact(self, filename: str) -> dict:
        """Read and parse a JSON file from the remote artifact directory.

        Parameters
        ----------
        filename:
            Base filename (e.g. ``"results.json"``).

        Returns
        -------
        dict
            The parsed JSON content.

        Raises
        ------
        SubagentError
            If the remote command fails or the content is not valid JSON.
        """
        remote_path = os.path.join(self.artifact_dir, filename)
        result = self.executor.docker_exec(f"cat {remote_path}", check=True)
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise SubagentError(
                f"Failed to parse JSON artifact {remote_path!r}: {exc}"
            ) from exc
