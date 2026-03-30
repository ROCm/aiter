"""CLI entry point for the agentic Triton kernel tuning pipeline.

Usage
-----
    python -m tuning_agent --config triton-upgrade.yaml [options]

Arguments
---------
--config PATH        (required) Path to the pipeline YAML configuration file.
--dry-run            Discover kernels and print a plan without executing.
--run-id ID          Custom run ID (default: auto-generated from datetime).
--repo-root PATH     Path to the aiter repo root (default: auto-detected by
                     searching upward for ``aiter/ops/triton/gemm/``).
--no-dashboard       Disable the live terminal dashboard.
--log-level LEVEL    Logging level (default: INFO).
--results-dir PATH   Override results directory (default: ~/.tuning_results).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import List, Optional

from .config import load_config
from .dashboard import Dashboard
from .notifications import Notifier

# Orchestrator is being created in parallel; import lazily to avoid hard
# failures when it is not yet present on disk during unit tests that mock it.
try:
    from .orchestrator import Orchestrator  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover
    Orchestrator = None  # type: ignore[assignment,misc]

_MARKER_SUBPATH = os.path.join("aiter", "ops", "triton", "gemm")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def find_repo_root(start_path: Optional[str] = None) -> str:
    """Walk upward from *start_path* to locate the aiter repository root.

    The root is identified by the presence of the directory
    ``<root>/aiter/ops/triton/gemm/``.

    Parameters
    ----------
    start_path:
        Directory to begin searching from.  Defaults to the current working
        directory (``os.getcwd()``) when *None*.

    Returns
    -------
    str
        Absolute path to the repository root (the *parent* of the ``aiter/``
        subtree).

    Raises
    ------
    FileNotFoundError
        If the marker directory is not found anywhere in the directory tree
        above *start_path*.
    """
    current = os.path.abspath(start_path if start_path is not None else os.getcwd())

    while True:
        candidate = os.path.join(current, _MARKER_SUBPATH)
        if os.path.isdir(candidate):
            return current

        parent = os.path.dirname(current)
        if parent == current:
            # Reached filesystem root without finding marker.
            break
        current = parent

    raise FileNotFoundError(
        f"Could not locate the aiter repo root: no '{_MARKER_SUBPATH}' directory "
        f"found while walking up from '{start_path or os.getcwd()}'."
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Construct and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="tuning_agent",
        description="Agentic Triton kernel tuning pipeline CLI.",
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to the pipeline YAML configuration file (required).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Discover kernels and show a plan without executing.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        metavar="ID",
        help="Custom run ID (default: auto-generated from datetime).",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        metavar="PATH",
        help=(
            "Path to the aiter repo root.  Defaults to auto-detection by "
            "searching upward for aiter/ops/triton/gemm/."
        ),
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        default=False,
        help="Disable the live terminal dashboard.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        metavar="LEVEL",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO).",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        metavar="PATH",
        help="Override results directory (default: ~/.tuning_results).",
    )
    return parser


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def main(args: Optional[List[str]] = None) -> None:
    """Parse *args* and drive the tuning pipeline.

    Parameters
    ----------
    args:
        Argument list to parse.  Defaults to ``sys.argv[1:]`` when *None*.
    """
    parser = _build_parser()
    ns = parser.parse_args(args)

    # ------------------------------------------------------------------
    # Logging setup
    # ------------------------------------------------------------------
    logging.basicConfig(
        level=getattr(logging, ns.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.debug("Parsed arguments: %s", ns)

    # ------------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------------
    logger.info("Loading config from %s", ns.config)
    config = load_config(ns.config)

    # ------------------------------------------------------------------
    # Resolve repository root
    # ------------------------------------------------------------------
    if ns.repo_root is not None:
        repo_root = ns.repo_root
        logger.info("Using explicit repo root: %s", repo_root)
    else:
        repo_root = find_repo_root()
        logger.info("Auto-detected repo root: %s", repo_root)

    # ------------------------------------------------------------------
    # Run ID
    # ------------------------------------------------------------------
    run_id: str = ns.run_id or datetime.now().strftime("run-%Y%m%d-%H%M%S")
    logger.info("Run ID: %s", run_id)

    # ------------------------------------------------------------------
    # Results directory
    # ------------------------------------------------------------------
    results_dir: str = ns.results_dir or os.path.expanduser("~/.tuning_results")

    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------
    notifier = Notifier()

    dashboard: Optional[Dashboard] = None
    if not ns.no_dashboard:
        dashboard = Dashboard()
        dashboard.start_auto_refresh()

    try:
        orchestrator = Orchestrator(
            config=config,
            repo_root=repo_root,
            run_id=run_id,
            results_dir=results_dir,
            notifier=notifier,
            dashboard=dashboard,
        )

        if ns.dry_run:
            # ---- Dry-run mode -----------------------------------------
            logger.info("Dry-run mode: discovering kernels …")
            kernels = orchestrator.discover_kernels()

            print()
            print("=== DRY-RUN PLAN ===")
            print(f"Config:      {ns.config}")
            print(f"Repo root:   {repo_root}")
            print(f"Run ID:      {run_id}")
            print(f"Results dir: {results_dir}")
            print(f"Kernels found: {len(kernels)}")
            for k in kernels:
                name = getattr(k, "name", str(k))
                print(f"  - {name}")
            print("=== (no execution in dry-run mode) ===")
        else:
            # ---- Normal execution mode ---------------------------------
            logger.info("Starting tuning pipeline …")
            result = orchestrator.run()
            logger.info("Pipeline complete.")

            print()
            print("=== TUNING COMPLETE ===")
            print(f"Run ID:      {run_id}")
            print(f"Results dir: {results_dir}")
            if result:
                for key, value in result.items():
                    print(f"  {key}: {value}")

    finally:
        if dashboard is not None:
            dashboard.stop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    main()
