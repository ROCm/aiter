"""Artifact manager for tuning results, configs, and checkpoints."""

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from .types import ShapeResult

if TYPE_CHECKING:
    from .remote import RemoteExecutor


class ArtifactManager:
    """Manages local and remote artifacts produced during a tuning run.

    Handles saving/loading benchmark results as JSON, phase checkpoint
    markers, and bidirectional file transfer between localhost and the
    remote Docker container via a :class:`~.remote.RemoteExecutor`.
    """

    def __init__(
        self,
        executor: "RemoteExecutor",
        kernel_name: str,
        run_id: Optional[str] = None,
        local_base_dir: str = "~/.tuning_results",
        remote_base_dir: str = "/workspace/tuning_artifacts",
    ) -> None:
        self.executor = executor
        self.kernel_name = kernel_name

        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id

        self.local_dir: Path = (
            Path(local_base_dir).expanduser() / run_id / kernel_name
        )
        self.remote_dir: str = f"{remote_base_dir}/{kernel_name}"

    # ------------------------------------------------------------------
    # Directory setup
    # ------------------------------------------------------------------

    def setup_local(self) -> None:
        """Create the local artifact directory (including all parents)."""
        self.local_dir.mkdir(parents=True, exist_ok=True)

    def setup_remote(self) -> None:
        """Create the remote artifact directory inside the container."""
        self.executor.docker_exec(f"mkdir -p {self.remote_dir}")

    # ------------------------------------------------------------------
    # Result persistence
    # ------------------------------------------------------------------

    def save_results(self, name: str, results: List[ShapeResult]) -> Path:
        """Serialize *results* to a JSON file inside :attr:`local_dir`.

        Parameters
        ----------
        name:
            Base filename (without extension) for the results file.
        results:
            List of :class:`~.types.ShapeResult` objects to persist.

        Returns
        -------
        Path
            Absolute path to the written JSON file.
        """
        records = [
            {
                "m": r.m,
                "n": r.n,
                "k": r.k,
                "main_ns": r.main_ns,
                "reduce_ns": r.reduce_ns,
                "total_ns": r.total_ns,
            }
            for r in results
        ]
        out_path = self.local_dir / f"{name}.json"
        out_path.write_text(json.dumps(records, indent=2))
        return out_path

    def load_results(self, name: str) -> List[ShapeResult]:
        """Load :class:`~.types.ShapeResult` objects from a JSON file.

        Parameters
        ----------
        name:
            Base filename (without extension) that was used in
            :meth:`save_results`.

        Returns
        -------
        List[ShapeResult]
            Reconstructed results in the same order they were saved.
        """
        in_path = self.local_dir / f"{name}.json"
        records = json.loads(in_path.read_text())
        return [
            ShapeResult(
                m=rec["m"],
                n=rec["n"],
                k=rec["k"],
                main_ns=rec["main_ns"],
                reduce_ns=rec["reduce_ns"],
            )
            for rec in records
        ]

    # ------------------------------------------------------------------
    # Phase checkpoints
    # ------------------------------------------------------------------

    def _phase_marker_path(self, phase: int) -> Path:
        return self.local_dir / f"phase_{phase}_complete.json"

    def mark_phase_complete(self, phase: int, summary: Dict) -> None:
        """Write a completion marker for *phase* with an ISO-8601 timestamp.

        Parameters
        ----------
        phase:
            Integer phase number (e.g. 1, 2, 3).
        summary:
            Arbitrary dict of summary data to store alongside the timestamp.
        """
        payload = {
            "timestamp": datetime.now().isoformat(),
            **summary,
        }
        self._phase_marker_path(phase).write_text(json.dumps(payload, indent=2))

    def is_phase_complete(self, phase: int) -> bool:
        """Return ``True`` if the phase marker file exists."""
        return self._phase_marker_path(phase).exists()

    def get_phase_summary(self, phase: int) -> Optional[Dict]:
        """Return the summary dict for *phase*, or ``None`` if not complete.

        Returns
        -------
        Optional[Dict]
            The full JSON payload written by :meth:`mark_phase_complete`,
            or ``None`` if the marker file does not exist.
        """
        marker = self._phase_marker_path(phase)
        if not marker.exists():
            return None
        return json.loads(marker.read_text())

    # ------------------------------------------------------------------
    # File transfer
    # ------------------------------------------------------------------

    def fetch_remote_file(self, remote_path: str, local_filename: str) -> Path:
        """Copy a file from the container to :attr:`local_dir`.

        Parameters
        ----------
        remote_path:
            Absolute path inside the container.
        local_filename:
            Destination filename (basename only) placed under :attr:`local_dir`.

        Returns
        -------
        Path
            Absolute local path of the downloaded file.
        """
        local_path = self.local_dir / local_filename
        self.executor.copy_from_container(remote_path, str(local_path))
        return local_path

    def push_local_file(self, local_filename: str, remote_path: str) -> None:
        """Copy a local file from :attr:`local_dir` into the container.

        Parameters
        ----------
        local_filename:
            Source filename (basename only) located under :attr:`local_dir`.
        remote_path:
            Absolute destination path inside the container.
        """
        local_path = self.local_dir / local_filename
        self.executor.copy_to_container(str(local_path), remote_path)
