# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Ruye-aa. All rights reserved.
"""Transactional helpers for JIT-generated sources and binaries."""

import os
import shlex
import shutil
import subprocess
import tempfile
import time


def stage_blob_sources(
    blob_gen_cmd, op_dir, python_executable, logger=None, log_commands=False
):
    """Run code generators in an isolated directory and return its path."""
    commands = blob_gen_cmd if isinstance(blob_gen_cmd, list) else [blob_gen_cmd]
    commands = [command for command in commands if command]
    if not commands:
        return None

    os.makedirs(op_dir, exist_ok=True)
    staging_dir = tempfile.mkdtemp(prefix=".blob-", dir=op_dir)
    output_dir = os.path.join(staging_dir, "")
    try:
        for command in commands:
            formatted_command = command.format(output_dir)
            args = [python_executable, *shlex.split(formatted_command)]
            if log_commands and logger is not None:
                logger.info("exec_blob ---> %s", shlex.join(args))
            subprocess.run(args, check=True)

        generated_sources = [
            os.path.join(root, filename)
            for root, _directories, filenames in os.walk(staging_dir)
            for filename in filenames
            if filename.endswith((".cpp", ".cu"))
        ]
        if not generated_sources:
            raise RuntimeError("blob code generation produced no C++/HIP sources")
        return staging_dir
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def publish_blob_sources(staging_dir, blob_dir):
    """Replace the last complete generated-source directory with a new one."""
    backup_dir = None
    if os.path.lexists(blob_dir):
        backup_dir = f"{blob_dir}.backup.{os.getpid()}.{time.time_ns()}"
        os.replace(blob_dir, backup_dir)
    try:
        os.replace(staging_dir, blob_dir)
    except Exception:
        if backup_dir is not None and not os.path.lexists(blob_dir):
            os.replace(backup_dir, blob_dir)
        raise
    if backup_dir is not None:
        shutil.rmtree(backup_dir, ignore_errors=True)


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
        shutil.copy2(source, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        try:
            os.remove(temporary_path)
        except FileNotFoundError:
            pass
