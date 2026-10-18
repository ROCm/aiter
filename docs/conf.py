# CPU-only documentation build: never import GPU packages.
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent / "_ext"))


def git_output(*args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


project = "AITER"
author = "AMD ROCm Team"
copyright = "2026, AMD"
source_revision = git_output("rev-parse", "HEAD")
release = git_output("describe", "--tags", "--always")
version = release
extensions = ["myst_parser", "sphinx.ext.mathjax", "aiter_source"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_heading_anchors = 3
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "README.md",
    "DEPLOYMENT.md",
    "DOCUMENTATION_AUDIT_REPORT.md",
]
html_theme = "sphinx_rtd_theme"
html_theme_options = {"collapse_navigation": False, "navigation_depth": 3}
html_logo = "assets/aiter_logo.png"
html_last_updated_fmt = "%Y-%m-%d %H:%M UTC"
html_context = {
    "display_github": True,
    "github_user": "ROCm",
    "github_repo": "aiter",
    "github_version": source_revision,
    "conf_py_path": "/docs/",
}
rst_epilog = f"""
.. |source_revision| replace:: {source_revision[:12]}
.. |build_date| replace:: {datetime.now(timezone.utc).strftime('%Y-%m-%d UTC')}
"""
