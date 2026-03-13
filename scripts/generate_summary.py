#!/usr/bin/env python3
"""Generate GitHub Actions job summaries for Aiter CI.

Usage:
    python3 scripts/generate_summary.py build
    python3 scripts/generate_summary.py promote

Each mode reads its inputs from environment variables and appends
Markdown to $GITHUB_STEP_SUMMARY.
"""

import os
import sys
from pathlib import Path


DOMAIN_MAP = {
    "nightlies": "rocm.frameworks-nightlies.amd.com",
    "devreleases": "rocm.frameworks-devreleases.amd.com",
    "prereleases": "rocm.frameworks-prereleases.amd.com",
    "release": "rocm.frameworks.amd.com",
}


def _out(path: Path, line: str = "") -> None:
    with open(path, "a") as f:
        f.write(line + "\n")


def _table(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    _out(path, "| " + " | ".join(headers) + " |")
    _out(path, "| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        _out(path, "| " + " | ".join(row) + " |")
    _out(path)


# ── Build summary ───────────────────────────────────────────────────────────

def build_summary(summary: Path) -> None:
    docker_image = os.environ.get("SUMMARY_DOCKER_IMAGE", "unknown")
    python_version = os.environ.get("SUMMARY_PYTHON_VERSION", "unknown")
    release_type = os.environ.get("SUMMARY_RELEASE_TYPE", "unknown")
    gpu_archs = os.environ.get("SUMMARY_GPU_ARCHS", "unknown")
    wheel_dir = os.environ.get("SUMMARY_WHEEL_DIR", "dist")

    _out(summary, f"## Build Summary - Python {python_version}")
    _out(summary)
    _table(summary, ["Item", "Value"], [
        ["Python version", f"`{python_version}`"],
        ["Docker image", f"`{docker_image}`"],
        ["Release type", f"`{release_type}`"],
        ["GPU architectures", f"`{gpu_archs}`"],
    ])

    _out(summary, "### Wheels")
    _out(summary, "```")
    whl_dir = Path(wheel_dir)
    wheels = sorted(whl_dir.glob("*.whl")) if whl_dir.is_dir() else []
    if wheels:
        for w in wheels:
            size_mb = w.stat().st_size / (1024 * 1024)
            _out(summary, f"  {w.name}  ({size_mb:.1f} MB)")
    else:
        _out(summary, "  No wheels found")
    _out(summary, "```")


# ── Promote summary ─────────────────────────────────────────────────────────

def promote_summary(summary: Path) -> None:
    release_type = os.environ.get("SUMMARY_RELEASE_TYPE", "unknown")
    source = os.environ.get("SUMMARY_S3_SOURCE", "unknown")
    dest = os.environ.get("SUMMARY_S3_DEST", "unknown")
    wheel_names = os.environ.get("SUMMARY_WHEEL_NAMES", "").strip()

    _out(summary, "## Promote Summary")
    _out(summary)
    _table(summary, ["Item", "Value"], [
        ["Release type", f"`{release_type}`"],
        ["Source", f"`{source}`"],
        ["Destination", f"`{dest}`"],
    ])

    if wheel_names:
        _out(summary, "### Promoted Wheels")
        _out(summary, "```")
        for whl in wheel_names.split():
            _out(summary, f"  {whl}")
        _out(summary, "```")
        _out(summary)

    domain = DOMAIN_MAP.get(release_type)
    if domain:
        index_url = f"https://{domain}/whl/gfx942-gfx950/"
        _out(summary, "### Wheels Available At")
        _out(summary, f"- {index_url}")
        _out(summary)
        _out(summary, "### Install Instructions")
        _out(summary)
        _out(summary, "**Using pip:**")
        _out(summary, "```bash")
        _out(summary, f"pip install --extra-index-url {index_url} amd-aiter")
        _out(summary, "```")
        _out(summary)
        _out(summary, "**Using uv:**")
        _out(summary, "```bash")
        _out(summary, f"uv pip install --extra-index-url {index_url} amd-aiter")
        _out(summary, "```")
        _out(summary)


# ── Main ────────────────────────────────────────────────────────────────────

MODES = {
    "build": build_summary,
    "promote": promote_summary,
}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in MODES:
        print(f"Usage: {sys.argv[0]} {{{','.join(MODES)}}}", file=sys.stderr)
        sys.exit(1)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        print("GITHUB_STEP_SUMMARY is not set", file=sys.stderr)
        sys.exit(1)

    MODES[sys.argv[1]](Path(summary_path))


if __name__ == "__main__":
    main()
