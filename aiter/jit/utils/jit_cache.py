# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Transactional helpers for JIT-generated sources and binaries."""

import os
import shlex
import shutil
import socket
import stat
import subprocess
import tempfile
import time
import uuid

GENERATED_BUILD_INPUT_SUFFIXES = (".cpp", ".cu", ".h", ".hpp", ".cuh")
STAGING_DIRECTORY_NAME = "blob.staging"
CODEGEN_INCOMPLETE_MARKER = ".aiter-codegen-incomplete"
CODEGEN_COMPLETE_MARKER = ".aiter-codegen-complete"
_INTERNAL_STAGE_FILES = {CODEGEN_INCOMPLETE_MARKER, CODEGEN_COMPLETE_MARKER}
_ABANDONED_ARTIFACT_PREFIXES = (
    ".blob-",  # random staging directories used by older revisions
    ".blob-publish-",
    ".blob-backup-",
    ".blob-reset-",
)

# Keep fault injection local to this module. Tests patch these aliases instead
# of replacing process-wide functions on ``os`` or ``shutil``.
_copy2 = shutil.copy2
_link = os.link
_replace = os.replace


def _remove_path(path):
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path, ignore_errors=True)
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _directory_mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


def _copy_directory(source, destination):
    shutil.copytree(source, destination, copy_function=_copy2)


def _restore_staging_directory(staging_dir, blob_dir, op_dir):
    """Restore the deterministic working tree from the last published cache."""
    discarded_dir = None
    if os.path.lexists(staging_dir):
        discarded_dir = os.path.join(
            op_dir, f".blob-reset-{os.getpid()}-{uuid.uuid4().hex}"
        )
        _replace(staging_dir, discarded_dir)
    try:
        if os.path.isdir(blob_dir):
            _copy_directory(blob_dir, staging_dir)
        else:
            os.makedirs(staging_dir, exist_ok=True)
        os.chmod(staging_dir, _directory_mode(op_dir))
    finally:
        if discarded_dir is not None:
            _remove_path(discarded_dir)


def _marker_owner_is_active(marker_path):
    try:
        with open(marker_path, encoding="utf-8") as marker:
            lines = marker.read().splitlines()
    except OSError:
        return False
    if len(lines) < 2 or not lines[0].isdigit():
        return False
    if lines[1] != socket.gethostname():
        return True
    try:
        os.kill(int(lines[0]), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _recover_blob_backup(blob_dir):
    """Recover a cache moved aside by a publisher that died mid-swap."""
    if os.path.lexists(blob_dir):
        return
    op_dir = os.path.dirname(blob_dir)
    backups = sorted(
        (
            os.path.join(op_dir, name)
            for name in os.listdir(op_dir)
            if name.startswith(".blob-backup-")
        ),
        key=lambda path: os.path.getmtime(path),
        reverse=True,
    )
    for backup_dir in backups:
        try:
            _replace(backup_dir, blob_dir)
            return
        except OSError:
            continue


def cleanup_abandoned_blob_artifacts(op_dir, max_age_seconds=24 * 60 * 60):
    """Remove abandoned transaction artifacts after a bounded grace period."""
    if not os.path.isdir(op_dir):
        return
    cutoff = time.time() - max_age_seconds
    for name in os.listdir(op_dir):
        if name in {STAGING_DIRECTORY_NAME, "blob"}:
            continue
        if not name.startswith(_ABANDONED_ARTIFACT_PREFIXES):
            continue
        path = os.path.join(op_dir, name)
        try:
            if os.path.getmtime(path) > cutoff:
                continue
        except OSError:
            continue
        _remove_path(path)


def _seed_staging_files(staging_dir, seed_files):
    for source, relative_destination in seed_files or ():
        if not os.path.isfile(source):
            continue
        destination = os.path.join(staging_dir, relative_destination)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        _copy2(source, destination)


def stage_blob_sources(
    blob_gen_cmd,
    op_dir,
    python_executable,
    logger=None,
    log_commands=False,
    seed_files=None,
):
    """Run code generators in a stable, recoverable working directory.

    The deterministic path preserves Ninja command lines, generated-file
    mtimes, DWARF paths, and manual retry workflows. A marker distinguishes an
    interrupted codegen from a complete working tree; only the former is reset
    from the last successfully published ``blob`` snapshot.
    """
    commands = blob_gen_cmd if isinstance(blob_gen_cmd, list) else [blob_gen_cmd]
    commands = [command for command in commands if command]
    if not commands:
        return None

    os.makedirs(op_dir, exist_ok=True)
    blob_dir = os.path.join(op_dir, "blob")
    staging_dir = os.path.join(op_dir, STAGING_DIRECTORY_NAME)
    incomplete_marker = os.path.join(staging_dir, CODEGEN_INCOMPLETE_MARKER)
    complete_marker = os.path.join(staging_dir, CODEGEN_COMPLETE_MARKER)

    _recover_blob_backup(blob_dir)
    cleanup_abandoned_blob_artifacts(op_dir)

    if os.path.exists(incomplete_marker):
        if _marker_owner_is_active(incomplete_marker):
            raise RuntimeError(
                f"blob code generation is already active under {staging_dir}"
            )
        _restore_staging_directory(staging_dir, blob_dir, op_dir)
    elif not os.path.isdir(staging_dir):
        _restore_staging_directory(staging_dir, blob_dir, op_dir)

    os.chmod(staging_dir, _directory_mode(op_dir))

    token = uuid.uuid4().hex
    with open(incomplete_marker, "x", encoding="utf-8") as marker:
        marker.write(f"{os.getpid()}\n{socket.gethostname()}\n{token}\n")
    try:
        try:
            os.remove(complete_marker)
        except FileNotFoundError:
            pass

        _seed_staging_files(staging_dir, seed_files)

        output_dir = os.path.join(staging_dir, "")
        for command in commands:
            formatted_command = command.format(output_dir)
            args = [python_executable, *shlex.split(formatted_command)]
            if log_commands and logger is not None:
                logger.info("exec_blob ---> %s", shlex.join(args))
            subprocess.run(args, check=True)

        generated_build_inputs = [
            os.path.join(root, filename)
            for root, _directories, filenames in os.walk(staging_dir)
            for filename in filenames
            if filename.endswith(GENERATED_BUILD_INPUT_SUFFIXES)
        ]
        if not generated_build_inputs:
            raise RuntimeError("blob code generation produced no C++/HIP build inputs")

        with open(incomplete_marker, encoding="utf-8") as marker:
            if marker.read().splitlines()[-1] != token:
                raise RuntimeError(
                    "blob staging directory changed during code generation"
                )
        _replace(incomplete_marker, complete_marker)
        return staging_dir
    except BaseException:
        try:
            _restore_staging_directory(staging_dir, blob_dir, op_dir)
        except Exception:
            if logger is not None:
                logger.warning(
                    "failed to restore JIT blob staging directory %s",
                    staging_dir,
                    exc_info=True,
                )
        raise


def _same_snapshot_file(source, previous):
    if previous is None:
        return False
    try:
        source_stat = os.stat(source, follow_symlinks=False)
        previous_stat = os.stat(previous, follow_symlinks=False)
    except OSError:
        return False
    if not (
        stat.S_ISREG(source_stat.st_mode)
        and stat.S_ISREG(previous_stat.st_mode)
        and source_stat.st_size == previous_stat.st_size
        and source_stat.st_mtime_ns == previous_stat.st_mtime_ns
        and stat.S_IMODE(source_stat.st_mode) == stat.S_IMODE(previous_stat.st_mode)
    ):
        return False
    with open(source, "rb") as source_file, open(previous, "rb") as previous_file:
        while True:
            source_chunk = source_file.read(1024 * 1024)
            if source_chunk != previous_file.read(1024 * 1024):
                return False
            if not source_chunk:
                return True


def _copy_blob_snapshot(staging_dir, candidate_dir, previous_blob_dir):
    for root, directories, filenames in os.walk(staging_dir):
        relative_root = os.path.relpath(root, staging_dir)
        candidate_root = (
            candidate_dir
            if relative_root == "."
            else os.path.join(candidate_dir, relative_root)
        )
        os.makedirs(candidate_root, exist_ok=True)
        os.chmod(candidate_root, _directory_mode(root))

        for directory in list(directories):
            source_directory = os.path.join(root, directory)
            if os.path.islink(source_directory):
                os.symlink(
                    os.readlink(source_directory),
                    os.path.join(candidate_root, directory),
                )
                directories.remove(directory)
        for filename in filenames:
            if filename in _INTERNAL_STAGE_FILES:
                continue
            source = os.path.join(root, filename)
            relative_path = os.path.relpath(source, staging_dir)
            destination = os.path.join(candidate_dir, relative_path)
            previous = (
                os.path.join(previous_blob_dir, relative_path)
                if previous_blob_dir is not None
                else None
            )
            os.makedirs(os.path.dirname(destination), exist_ok=True)

            if os.path.islink(source):
                os.symlink(os.readlink(source), destination)
            elif _same_snapshot_file(source, previous):
                try:
                    _link(previous, destination)
                except OSError:
                    _copy2(source, destination)
            else:
                _copy2(source, destination)


def _complete_stage_token(staging_dir):
    if os.path.exists(os.path.join(staging_dir, CODEGEN_INCOMPLETE_MARKER)):
        return None
    try:
        with open(
            os.path.join(staging_dir, CODEGEN_COMPLETE_MARKER), encoding="utf-8"
        ) as marker:
            return marker.read()
    except OSError:
        return None


def publish_blob_sources(staging_dir, blob_dir):
    """Atomically snapshot a complete staging tree into the ``blob`` cache.

    ``staging_dir`` remains in place so ``build.ninja`` and debug metadata keep
    resolving after both successful and failed builds. Callers should treat
    publication as best-effort because the compiled artifact is authoritative.
    """
    token_before = _complete_stage_token(staging_dir)
    if token_before is None:
        raise RuntimeError("refusing to publish incomplete JIT blob sources")

    op_dir = os.path.dirname(blob_dir)
    candidate_dir = tempfile.mkdtemp(prefix=".blob-publish-", dir=op_dir)
    backup_dir = None
    try:
        previous_blob_dir = blob_dir if os.path.isdir(blob_dir) else None
        _copy_blob_snapshot(staging_dir, candidate_dir, previous_blob_dir)
        if _complete_stage_token(staging_dir) != token_before:
            raise RuntimeError("JIT blob sources changed during publication")

        if os.path.lexists(blob_dir):
            backup_dir = os.path.join(
                op_dir, f".blob-backup-{os.getpid()}-{uuid.uuid4().hex}"
            )
            _replace(blob_dir, backup_dir)
        try:
            _replace(candidate_dir, blob_dir)
            candidate_dir = None
        except Exception:
            if backup_dir is not None and not os.path.lexists(blob_dir):
                _replace(backup_dir, blob_dir)
                backup_dir = None
            raise
    finally:
        if candidate_dir is not None:
            _remove_path(candidate_dir)
        if backup_dir is not None:
            _remove_path(backup_dir)


def atomic_copy(source, destination):
    """Copy ``source`` without exposing a partial ``destination`` file."""
    destination_dir = os.path.dirname(destination)
    os.makedirs(destination_dir, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(destination)}.",
        suffix=".tmp",
        dir=destination_dir,
    )
    os.close(fd)
    try:
        _copy2(source, temporary_path)
        _replace(temporary_path, destination)
    finally:
        try:
            os.remove(temporary_path)
        except FileNotFoundError:
            pass
