# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Maintain pinned, AITER-managed dependency checkouts under the caller's lock."""

import hashlib
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


def checkout_lock_path(path):
    """Share one lock across JIT caches and aliases of the same checkout."""
    identity = str(Path(path).resolve())
    digest = hashlib.sha256(os.fsencode(identity)).hexdigest()
    owner = (
        os.getuid()
        if hasattr(os, "getuid")
        else hashlib.sha256(os.fsencode(Path.home())).hexdigest()
    )
    # Installed or host-mounted headers can be read-only even at the right pin.
    # Keep the lock outside the checkout, independent of AITER_JIT_DIR.
    return str(
        Path(tempfile.gettempdir()) / f"aiter-checkout-locks-{owner}" / f"{digest}.lock"
    )


def _git(path, *args):
    # Containers can see a host-owned managed checkout. Trust only this exact
    # checkout for this command; do not modify the user's Git configuration.
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={path.resolve()}", "-C", str(path), *args],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def ensure_pinned_checkout(path, url, revision, *, user_override=False):
    path = Path(path).resolve()
    if user_override:
        # HIP_KITTENS_DIR may be a source archive or a developer's working tree.
        # The caller deliberately selected these headers; never rewrite them.
        if not path.is_dir():
            raise RuntimeError(f"HIP_KITTENS_DIR override does not exist: {path}")
        return

    try:
        if path.exists():
            # Do not let git -C discover a parent repository and check it out!
            if (
                not (path / ".git").exists()
                or Path(_git(path, "rev-parse", "--show-toplevel")).resolve() != path
            ):
                raise RuntimeError(
                    f"Not a managed dependency checkout: {path}; use HIP_KITTENS_DIR for source archives"
                )
            # HEAD alone does not establish that the compiled headers match it.
            if _git(path, "status", "--porcelain", "--untracked-files=all"):
                raise RuntimeError(
                    f"Cannot use modified dependency {path} at {revision}; "
                    "preserve your changes and use HIP_KITTENS_DIR for custom headers"
                )
            if _git(path, "rev-parse", "HEAD") == revision:
                return
            # Fetch the required commit even when an old shallow clone lacks it.
            _git(path, "fetch", "--depth=1", url, revision)
            _git(path, "checkout", "--quiet", "--detach", revision)
            _git(path, "submodule", "update", "--init", "--recursive")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            # A failed clone must not leave a directory that looks like a cache hit.
            with tempfile.TemporaryDirectory(
                prefix=f".{path.name}-", dir=path.parent
            ) as temp:
                stage = Path(temp) / "checkout"
                stage.mkdir()
                _git(stage, "init", "--quiet")
                _git(stage, "remote", "add", "origin", url)
                _git(stage, "fetch", "--quiet", "--depth=1", url, revision)
                _git(stage, "checkout", "--quiet", "--detach", revision)
                _git(stage, "submodule", "update", "--init", "--recursive")
                os.rename(stage, path)
    except (subprocess.CalledProcessError, OSError) as error:
        detail = (
            error.output
            if isinstance(error, subprocess.CalledProcessError)
            else str(error)
        )
        detail = detail or str(error)
        raise RuntimeError(
            f"Cannot prepare managed dependency {path} at {revision}: {detail.strip()}. "
            "For offline builds or custom headers, set HIP_KITTENS_DIR to an "
            "existing HipKittens source directory."
        ) from error
