"""Machine pool manager for the agentic Triton kernel tuning pipeline.

Maintains a thread-safe pool of :class:`~.types.MachineInfo` instances,
tracking which are idle, allocated to a kernel, or dead (unreachable).
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Set

from .remote import RemoteExecutor
from .types import MachineInfo


class NoMachineAvailable(Exception):
    """Raised when :meth:`MachinePool.allocate` finds no idle machines."""


class MachinePool:
    """Thread-safe pool of GPU machines for concurrent kernel tuning.

    Parameters
    ----------
    machines:
        List of :class:`MachineInfo` objects to manage.  Each machine is
        keyed by its ``host`` string; duplicate hosts are silently deduplicated
        (last entry wins).
    """

    def __init__(self, machines: List[MachineInfo]) -> None:
        # Keyed by host for O(1) lookup.
        self._machines: Dict[str, MachineInfo] = {m.host: m for m in machines}
        # Hosts currently allocated to a kernel tuning job.
        self._allocated: Dict[str, str] = {}  # host -> kernel_name
        # Hosts that are unreachable / permanently removed from the pool.
        self._dead: Set[str] = set()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Core pool operations
    # ------------------------------------------------------------------

    def allocate(self, kernel_name: str) -> MachineInfo:
        """Allocate an idle machine for *kernel_name*.

        Among available (idle, non-dead) machines the one with the most GPUs
        is preferred.  On a tie the host name is used as a tiebreaker so the
        result is deterministic.

        Parameters
        ----------
        kernel_name:
            Name of the kernel that will use the machine (recorded in state).

        Returns
        -------
        MachineInfo
            The allocated machine.

        Raises
        ------
        NoMachineAvailable
            If no idle, non-dead machine exists in the pool.
        """
        with self._lock:
            candidates = [
                m
                for host, m in self._machines.items()
                if host not in self._allocated and host not in self._dead
            ]
            if not candidates:
                raise NoMachineAvailable(
                    f"No machine available for kernel '{kernel_name}'. "
                    f"Total={len(self._machines)}, "
                    f"allocated={len(self._allocated)}, "
                    f"dead={len(self._dead)}."
                )
            # Prefer most GPUs; stable tiebreak by host name.
            best = max(candidates, key=lambda m: (m.gpu_count, m.host))
            self._allocated[best.host] = kernel_name
            return best

    def release(self, host: str) -> None:
        """Release a previously allocated machine back to the idle pool.

        Parameters
        ----------
        host:
            The ``MachineInfo.host`` value of the machine to release.
            If *host* is not currently allocated this is a no-op.
        """
        with self._lock:
            self._allocated.pop(host, None)

    def mark_dead(self, host: str) -> None:
        """Mark *host* as unreachable and remove it from the allocated set.

        Parameters
        ----------
        host:
            The ``MachineInfo.host`` value of the machine to mark dead.
        """
        with self._lock:
            self._dead.add(host)
            self._allocated.pop(host, None)

    def get_machine(self, host: str) -> Optional[MachineInfo]:
        """Return the :class:`MachineInfo` for *host*, or ``None`` if unknown.

        Parameters
        ----------
        host:
            The ``MachineInfo.host`` value to look up.
        """
        return self._machines.get(host)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def available_count(self) -> int:
        """Number of idle (non-allocated, non-dead) machines."""
        with self._lock:
            return sum(
                1
                for host in self._machines
                if host not in self._allocated and host not in self._dead
            )

    def status(self) -> List[Dict]:
        """Return a snapshot of every machine's current state.

        Returns
        -------
        List[Dict]
            One dict per machine with keys:

            ``host``
                The machine hostname / IP.
            ``state``
                One of ``"idle"``, ``"allocated"``, or ``"dead"``.
            ``kernel``
                The kernel name if state is ``"allocated"``, otherwise ``None``.
            ``gpus``
                The list of GPU indices from :class:`MachineInfo`.
            ``gpu_count``
                Convenience shorthand for ``len(gpus)``.
        """
        with self._lock:
            result: List[Dict] = []
            for host, machine in self._machines.items():
                if host in self._dead:
                    state = "dead"
                    kernel = None
                elif host in self._allocated:
                    state = "allocated"
                    kernel = self._allocated[host]
                else:
                    state = "idle"
                    kernel = None

                result.append(
                    {
                        "host": host,
                        "state": state,
                        "kernel": kernel,
                        "gpus": list(machine.gpus),
                        "gpu_count": machine.gpu_count,
                    }
                )
            return result

    # ------------------------------------------------------------------
    # Connectivity validation
    # ------------------------------------------------------------------

    def validate_connectivity(self) -> List[Dict]:
        """Test SSH reachability and ``rocm-smi`` presence for every machine.

        Each machine is probed independently.  Machines that fail either check
        are automatically marked dead via :meth:`mark_dead`.

        Returns
        -------
        List[Dict]
            One dict per machine with keys:

            ``host``
                The machine hostname / IP.
            ``reachable``
                ``True`` if the SSH connectivity check succeeded.
            ``rocm_smi_ok``
                ``True`` if ``rocm-smi`` was found on the remote host.
            ``error``
                Error message string if a check failed, otherwise ``None``.
        """
        results: List[Dict] = []
        # Snapshot hosts so we don't iterate over a changing dict.
        with self._lock:
            hosts = list(self._machines.keys())

        for host in hosts:
            machine = self._machines[host]
            executor = RemoteExecutor(machine)
            reachable = False
            rocm_smi_ok = False
            error: Optional[str] = None

            try:
                reachable = executor.check_ssh_connectivity()
            except Exception as exc:  # pragma: no cover — defensive catch
                error = str(exc)

            if reachable:
                try:
                    result = executor.ssh_run("which rocm-smi", check=False)
                    rocm_smi_ok = result.returncode == 0
                    if not rocm_smi_ok:
                        error = "rocm-smi not found on remote host"
                except Exception as exc:
                    error = str(exc)
                    rocm_smi_ok = False
            else:
                if error is None:
                    error = "SSH connectivity check failed"

            if not reachable or not rocm_smi_ok:
                self.mark_dead(host)

            results.append(
                {
                    "host": host,
                    "reachable": reachable,
                    "rocm_smi_ok": rocm_smi_ok,
                    "error": error,
                }
            )

        return results
