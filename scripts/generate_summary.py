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


def _get_index_url(release_type: str, gpu_archs: str = "gfx942-gfx950") -> str | None:
    domain = DOMAIN_MAP.get(release_type)
    if not domain:
        return None
    return f"https://{domain}/whl/{gpu_archs}/"


def build_summary(summary: Path) -> None:
    docker_image = os.environ.get("SUMMARY_DOCKER_IMAGE", "unknown")
    python_version = os.environ.get("SUMMARY_PYTHON_VERSION", "unknown")
    release_type = os.environ.get("SUMMARY_RELEASE_TYPE", "unknown")
    gpu_archs = os.environ.get("SUMMARY_GPU_ARCHS", "unknown")
    wheel_dir = os.environ.get("SUMMARY_WHEEL_DIR", "dist")
    index_url = _get_index_url(release_type, gpu_archs.replace(";", "-"))

    _out(summary, f"## Build Summary - Python {python_version}")
    _out(summary)
    rows = [
        ["Python version", f"`{python_version}`"],
        ["Docker image", f"`{docker_image}`"],
        ["Release type", f"`{release_type}`"],
        ["GPU architectures", f"`{gpu_archs}`"],
    ]
    if index_url:
        rows.append(["Index URL", index_url])
    _table(summary, ["Item", "Value"], rows)

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
    s3_dest = os.environ.get("SUMMARY_S3_DEST", "")
    wheel_names = os.environ.get("SUMMARY_WHEEL_NAMES", "").strip()

    url_path = "/".join(s3_dest.split("/")[3:]) if s3_dest.startswith("s3://") else "whl/gfx942-gfx950"
    domain = DOMAIN_MAP.get(release_type)
    index_url = f"https://{domain}/{url_path}/" if domain else None

    _out(summary, "## Promote Summary")
    _out(summary)
    rows = [["Release type", f"`{release_type}`"]]
    if index_url:
        rows.append(["Index URL", index_url])
    _table(summary, ["Item", "Value"], rows)

    if wheel_names:
        _out(summary, "### Promoted Wheels")
        _out(summary, "```")
        for whl in wheel_names.split():
            _out(summary, f"  {whl}")
        _out(summary, "```")
        _out(summary)

    if index_url:
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
