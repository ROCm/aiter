# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Transactional helpers for JIT-generated sources and binaries."""

import hashlib
import json
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
    "blob.backup.",  # backups used by older revisions
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
    shutil.copytree(source, destination, copy_function=_copy2, dirs_exist_ok=True)


def _transaction_prefix(kind):
    return f".blob-{kind}-{socket.gethostname()}.{os.getpid()}."


def _artifact_owner_active(name):
    """Return None for legacy names; never reclaim a live or remote owner."""
    for kind in ("publish", "backup", "reset"):
        prefix = f".blob-{kind}-"
        if not name.startswith(prefix):
            continue
        parts = name[len(prefix) :].rsplit(".", 2)
        if len(parts) != 3 or not parts[1].isdigit():
            return None
        if parts[0] != socket.gethostname():
            return True
        try:
            os.kill(int(parts[1]), 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    return None


def _restore_staging_directory(staging_dir, blob_dir, op_dir):
    """Restore the deterministic working tree from the last published cache."""
    discarded_dir = None
    candidate_dir = None
    if os.path.lexists(staging_dir):
        discarded_dir = os.path.join(
            op_dir, f"{_transaction_prefix('reset')}{uuid.uuid4().hex}"
        )
        _replace(staging_dir, discarded_dir)
    try:
        # A failed or killed copy must not leave a partially restored tree at
        # the stable path: the next invocation would mistake it for a complete
        # working tree. Build the replacement separately and install it whole.
        candidate_dir = tempfile.mkdtemp(
            prefix=_transaction_prefix("reset"), dir=op_dir
        )
        os.chmod(candidate_dir, _directory_mode(op_dir))
        if os.path.isdir(blob_dir):
            _copy_directory(blob_dir, candidate_dir)
        os.chmod(candidate_dir, _directory_mode(op_dir))
        _replace(candidate_dir, staging_dir)
        candidate_dir = None
    finally:
        if candidate_dir is not None:
            _remove_path(candidate_dir)
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
    """Recover a dead publisher's backup, or our own failed rollback.

    Called under the module build lock. Owner-tagged live peers are left alone.
    Legacy backups have no owner metadata; recovering the only published cache
    under this lock does not wait for the age grace used to delete old artifacts.
    """
    if os.path.lexists(blob_dir):
        return
    op_dir = os.path.dirname(blob_dir)
    backups = sorted(
        (
            os.path.join(op_dir, name)
            for name in os.listdir(op_dir)
            if name.startswith((".blob-backup-", "blob.backup."))
            and (
                name.startswith(_transaction_prefix("backup"))
                or _artifact_owner_active(name) is not True
            )
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
    """Reap dead local owners on the next build; age out legacy artifacts.

    Stable blob/staging trees are retained for incremental Ninja retries.
    The 24-hour grace applies only to legacy names without owner information,
    not to new artifacts whose local owner has died. This is not a timer:
    cleanup runs when the same module next enters code generation.
    """
    if not os.path.isdir(op_dir):
        return
    cutoff = time.time() - max_age_seconds
    for name in os.listdir(op_dir):
        if name in {STAGING_DIRECTORY_NAME, "blob"}:
            continue
        if not name.startswith(_ABANDONED_ARTIFACT_PREFIXES):
            continue
        if name.startswith((".blob-backup-", "blob.backup.")) and not os.path.lexists(
            os.path.join(op_dir, "blob")
        ):
            # Recovery may still be denied. Age/dead-owner cleanup must not
            # destroy the only remaining published snapshot.
            continue
        owner_active = _artifact_owner_active(name)
        if owner_active is True:
            continue
        path = os.path.join(op_dir, name)
        try:
            if owner_active is None and os.path.getmtime(path) > cutoff:
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
    return_token=False,
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
        return (None, None) if return_token else None

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
    generation = f"{os.getpid()}\n{socket.gethostname()}\n{token}\n"
    with open(incomplete_marker, "x", encoding="utf-8") as marker:
        marker.write(generation)
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
        # Return our token, not a fresh read that could belong to a peer.
        return (staging_dir, generation) if return_token else staging_dir
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


def require_blob_generation(staging_dir, token):
    """Reject codegen overlap before installing a potentially mixed binary."""
    if token is None or _complete_stage_token(staging_dir) != token:
        raise RuntimeError("JIT blob generation changed during build")


def publish_blob_sources(staging_dir, blob_dir, expected_token=None):
    """Atomically snapshot a complete staging tree into the ``blob`` cache.

    ``staging_dir`` remains in place so ``build.ninja`` and debug metadata keep
    resolving after both successful and failed builds. Callers should treat
    publication as best-effort because the compiled artifact is authoritative.
    """
    token_before = _complete_stage_token(staging_dir)
    if token_before is None:
        raise RuntimeError("refusing to publish incomplete JIT blob sources")
    if expected_token is not None and token_before != expected_token:
        raise RuntimeError("JIT blob generation changed before publication")

    op_dir = os.path.dirname(blob_dir)
    candidate_dir = tempfile.mkdtemp(prefix=_transaction_prefix("publish"), dir=op_dir)
    backup_dir = None
    retain_backup = False
    try:
        # mkdtemp starts at 0700. Correct it before copying any contents, even
        # if publication later fails or the builder is killed during the copy.
        os.chmod(candidate_dir, _directory_mode(op_dir))
        previous_blob_dir = blob_dir if os.path.isdir(blob_dir) else None
        _copy_blob_snapshot(staging_dir, candidate_dir, previous_blob_dir)
        if _complete_stage_token(staging_dir) != token_before:
            raise RuntimeError("JIT blob sources changed during publication")

        if os.path.lexists(blob_dir):
            backup_dir = os.path.join(
                op_dir, f"{_transaction_prefix('backup')}{uuid.uuid4().hex}"
            )
            _replace(blob_dir, backup_dir)
        try:
            _replace(candidate_dir, blob_dir)
            candidate_dir = None
        except Exception:
            if backup_dir is not None and not os.path.lexists(blob_dir):
                try:
                    _replace(backup_dir, blob_dir)
                except OSError:
                    # Both publication and rollback failed. Leave the backup
                    # for the next locked build, even if this process stays alive.
                    retain_backup = True
                    raise
                backup_dir = None
            raise
    finally:
        if candidate_dir is not None:
            _remove_path(candidate_dir)
        if backup_dir is not None and not retain_backup:
            _remove_path(backup_dir)


def atomic_copy(source, destination, validate=None):
    """Install a complete copy and return the identity of this exact inode.

    ``validate`` runs after copying, immediately before replacement. Holding
    the copied inode open avoids accidentally identifying a peer's replacement.
    """
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
        with open(temporary_path, "rb") as copied:
            if validate is not None:
                validate()
            _replace(temporary_path, destination)
            return _stat_identity(os.fstat(copied.fileno()))
    finally:
        try:
            os.remove(temporary_path)
        except FileNotFoundError:
            pass


def _stat_identity(info):
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def _artifact_identity(path):
    return _stat_identity(os.stat(path))


def compiled_kids_are_current(sidecar, artifact, required):
    """A sidecar is evidence only when its receipt matches the installed .so.

    Missing/legacy metadata, a failed publication or a replaced/copied binary
    conservatively requires rebuilding. Neither file lives under clear_build.
    """
    try:
        with open(sidecar, "rb") as source:
            contents = source.read()
        with open(sidecar + ".receipt", encoding="utf-8") as source:
            receipt = json.load(source)
        return (
            receipt["artifact"] == _artifact_identity(artifact)
            and receipt["sha256"] == hashlib.sha256(contents).hexdigest()
            and set(required) <= set(json.loads(contents))
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def snapshot_compiled_kids(staged_sidecar):
    """Keep the metadata compiled by this invocation, not a later staging tree."""
    with open(staged_sidecar, "rb") as source:
        return source.read(), stat.S_IMODE(os.fstat(source.fileno()).st_mode) & 0o666


def publish_compiled_kids(snapshot, sidecar, artifact, identity):
    """Publish successful codegen metadata after installing its binary.

    The receipt detects the gap between the two atomic replacements. If this
    step fails, the installed binary remains usable and the tuner must rebuild
    before relying on the sidecar again. Called under the module build lock.
    """
    if _artifact_identity(artifact) != identity:
        raise RuntimeError("installed Opus binary changed before metadata publication")
    contents, mode = snapshot
    receipt = {"artifact": identity, "sha256": hashlib.sha256(contents).hexdigest()}
    fd, temporary_path = tempfile.mkstemp(
        prefix=".opus-sidecar-", dir=os.path.dirname(sidecar)
    )
    try:
        with os.fdopen(fd, "wb") as output:
            os.chmod(temporary_path, mode)
            output.write(contents)
        _replace(temporary_path, sidecar)
    finally:
        _remove_path(temporary_path)
    fd, temporary_path = tempfile.mkstemp(
        prefix=".opus-receipt-", dir=os.path.dirname(sidecar)
    )
    try:
        # Avoid leaving a root-owned, unreadable tempfile in a packaged JIT tree.
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            os.chmod(temporary_path, mode)
            json.dump(receipt, output)
        _replace(temporary_path, sidecar + ".receipt")
    finally:
        _remove_path(temporary_path)
