"""Terminal dashboard for the agentic Triton kernel tuning pipeline.

Renders a live view of machine states, kernel progress, recent logs, and
notifications using plain ANSI escape codes.  No curses or third-party
libraries are required.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

# ANSI colour codes
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_RESET = "\033[0m"

_SEPARATOR = "═" * 55
_MAX_LOGS = 20


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed seconds as a human-readable string (e.g. '12m', '3m')."""
    minutes = int(seconds) // 60
    return f"{minutes}m"


def _state_label(state: str) -> str:
    """Return a coloured state label string."""
    state_upper = state.upper()
    if state_upper == "BUSY":
        return f"{_YELLOW}[BUSY]{_RESET}"
    elif state_upper == "IDLE":
        return f"{_GREEN}[IDLE]{_RESET}"
    elif state_upper == "DEAD":
        return f"{_RED}[DEAD]{_RESET}"
    else:
        return f"[{state_upper}]"


class Dashboard:
    """Simple terminal dashboard for the Triton kernel tuning pipeline.

    Parameters
    ----------
    refresh_interval:
        Seconds between automatic re-renders when :meth:`start_auto_refresh`
        is active.
    """

    def __init__(self, refresh_interval: float = 2.0) -> None:
        self.refresh_interval = refresh_interval

        # State stores
        self.machines: Dict[str, dict] = {}
        self.kernels: Dict[str, dict] = {}
        self.logs: List[str] = []
        self.notifications: List[str] = []

        # Auto-refresh state
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # State update methods
    # ------------------------------------------------------------------

    def update_machine(
        self,
        host: str,
        state: str,
        kernel: Optional[str] = None,
        gpu_count: int = 0,
    ) -> None:
        """Update (or insert) a machine entry.

        Parameters
        ----------
        host:
            Hostname / IP of the machine.
        state:
            One of ``"idle"``, ``"busy"``, or ``"dead"``.
        kernel:
            Name of the kernel currently running on this machine (optional).
        gpu_count:
            Number of GPUs available on this machine.
        """
        self.machines[host] = {
            "state": state,
            "kernel": kernel,
            "gpus": list(range(gpu_count)),
            "gpu_count": gpu_count,
        }

    def update_kernel(
        self,
        name: str,
        phase: int,
        phase_name: str,
        progress: str = "",
        elapsed: float = 0,
        status: str = "running",
    ) -> None:
        """Update (or insert) a kernel progress entry.

        Parameters
        ----------
        name:
            Kernel identifier string.
        phase:
            Current phase number (e.g. 4).
        phase_name:
            Human-readable phase label (e.g. ``"TUNING"``).
        progress:
            Optional progress hint string (e.g. ``"45/56"``).
        elapsed:
            Elapsed seconds since this kernel started.
        status:
            Short status tag, e.g. ``"running"`` or ``"done"``.
        """
        self.kernels[name] = {
            "phase": phase,
            "phase_name": phase_name,
            "progress": progress,
            "elapsed": elapsed,
            "status": status,
        }

    def add_log(self, message: str) -> None:
        """Append *message* to the log buffer (capped at 20 lines)."""
        self.logs.append(message)
        if len(self.logs) > _MAX_LOGS:
            self.logs = self.logs[-_MAX_LOGS:]

    def add_notification(self, message: str) -> None:
        """Append *message* to the notifications buffer."""
        self.notifications.append(message)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> str:
        """Render the full dashboard as a multi-line string with ANSI colours.

        Returns
        -------
        str
            A ready-to-print string containing the complete dashboard view.
        """
        lines: List[str] = []

        lines.append(_SEPARATOR)
        lines.append("  TRITON KERNEL TUNING PIPELINE")
        lines.append(_SEPARATOR)

        # ---- MACHINES section ----------------------------------------
        lines.append("MACHINES:")
        for host, info in self.machines.items():
            state_str = _state_label(info["state"])
            kernel_part = f"kernel={info['kernel']:<10}" if info["kernel"] else " " * 16
            gpu_part = f"GPUs: {info['gpu_count']}"
            lines.append(f"  {host:<15} {state_str} {kernel_part} {gpu_part}")

        # ---- KERNELS section -----------------------------------------
        lines.append("")
        lines.append("KERNELS:")
        for name, info in self.kernels.items():
            elapsed_str = _fmt_elapsed(info["elapsed"])
            phase_str = f"Phase {info['phase']}/6 {info['phase_name']:<12}"
            lines.append(
                f"  {name:<14} {phase_str} {elapsed_str} elapsed  [{info['status']}]"
            )

        # ---- RECENT LOGS section -------------------------------------
        lines.append("")
        lines.append("RECENT LOGS:")
        for entry in self.logs:
            lines.append(f"  {entry}")

        # ---- NOTIFICATIONS section -----------------------------------
        lines.append("")
        lines.append("NOTIFICATIONS:")
        for note in self.notifications:
            lines.append(f"  {note}")

        lines.append(_SEPARATOR)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Auto-refresh
    # ------------------------------------------------------------------

    def start_auto_refresh(self) -> None:
        """Start a background daemon thread that prints the dashboard periodically."""
        self._stop_event.clear()

        def _loop() -> None:
            while not self._stop_event.is_set():
                print(self.render())
                self._stop_event.wait(timeout=self.refresh_interval)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the auto-refresh background thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.refresh_interval + 1)
            self._thread = None
