# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Maintain pinned, AITER-managed dependency checkouts under the caller's lock."""

import os
import subprocess
import tempfile
from pathlib import Path


def checkout_path(path, build_dir, *, user_override=False):
    """Keep bundled source archives intact; fetch a verifiable pin in the JIT cache."""
    path = Path(path)
    if not user_override and path.exists() and not (path / ".git").exists():
        return str(Path(build_dir) / "third_party" / path.name)
    return str(path)


def ensure_pinned_checkout(path, url, revision, *, user_override=False):
    path = Path(path)
    if user_override:
        # HIP_KITTENS_DIR may be a source archive or a developer's working tree.
        # The caller deliberately selected these headers; never rewrite them.
        if not path.is_dir():
            raise RuntimeError(f"Dependency override does not exist: {path}")
        return

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True
        ).strip()

    if path.exists():
        # Do not let git -C discover a parent repository and check it out!
        if (
            not (path / ".git").exists()
            or Path(git("rev-parse", "--show-toplevel")).resolve() != path.resolve()
        ):
            raise RuntimeError(
                f"Not a managed dependency checkout: {path}; use HIP_KITTENS_DIR for source archives"
            )
        if git("rev-parse", "HEAD") == revision:
            return
        if git("status", "--porcelain"):
            raise RuntimeError(
                f"Cannot update modified dependency {path} to {revision}; "
                "preserve your changes and use HIP_KITTENS_DIR for custom headers"
            )
        # Fetch the required commit even when an old shallow clone lacks it.
        git("fetch", "--depth=1", url, revision)
        git("checkout", "--quiet", "--detach", revision)
        git("submodule", "update", "--init", "--recursive")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        # A failed clone must not leave a directory that looks like a cache hit.
        with tempfile.TemporaryDirectory(
            prefix=f".{path.name}-", dir=path.parent
        ) as temp:
            stage = Path(temp) / "checkout"
            subprocess.run(["git", "init", "--quiet", str(stage)], check=True)
            subprocess.run(
                ["git", "-C", str(stage), "remote", "add", "origin", url], check=True
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(stage),
                    "fetch",
                    "--quiet",
                    "--depth=1",
                    url,
                    revision,
                ],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(stage), "checkout", "--quiet", "--detach", revision],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(stage),
                    "submodule",
                    "update",
                    "--init",
                    "--recursive",
                ],
                check=True,
            )
            os.rename(stage, path)
