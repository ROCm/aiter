# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only tests for transactional JIT cache publication."""

import ast
import builtins
import contextlib
import importlib.util
import json
import multiprocessing
import os
import re
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
import traceback
import types
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest import mock

JIT_CACHE_PATH = (
    Path(__file__).resolve().parents[1] / "aiter" / "jit" / "utils" / "jit_cache.py"
)


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


jit_cache = _load_module(JIT_CACHE_PATH, "aiter_jit_cache_transaction_under_test")
versioner_module = _load_module(
    JIT_CACHE_PATH.with_name("_cpp_extension_versioner.py"),
    "aiter_versioner_under_test",
)
baton_module = _load_module(
    JIT_CACHE_PATH.with_name("file_baton.py"), "aiter_file_baton_under_test"
)


def _write_generator(directory, body):
    path = os.path.join(directory, "generator.py")
    with open(path, "w", encoding="utf-8") as generator:
        generator.write(body)
    return path


def _write(path, contents):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as output:
        output.write(contents)


def _read(path):
    with open(path, encoding="utf-8") as source:
        return source.read()


def _transaction_artifacts(directory):
    return [
        name
        for name in os.listdir(directory)
        if name.startswith((".blob-publish-", ".blob-backup-", ".blob-reset-"))
    ]


def _load_functions(path, names, namespace):
    """Execute checkout functions with hardware dependencies injected.

    This exercises build_module's real control flow without importing torch or
    a GPU-dependent aiter package, and without modifying sys.path/sys.modules.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in functions} != set(names):
        raise AssertionError(f"missing functions in {path}: {names}")
    exec(  # noqa: S102 - trusted checkout AST, with injected CPU dependencies
        compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace


class TestJitCacheTransaction(unittest.TestCase):
    def test_failed_codegen_restores_last_complete_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            op_dir = os.path.join(tmp, "module")
            blob_dir = os.path.join(op_dir, "blob")
            complete_source = os.path.join(blob_dir, "complete.cpp")
            _write(complete_source, "// known-good\n")

            generator = _write_generator(
                tmp,
                """import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()
with open(args.output_dir + "/partial.cpp", "w") as source:
    source.write("// incomplete\\n")
raise SystemExit(7)
""",
            )

            with self.assertRaises(subprocess.CalledProcessError):
                jit_cache.stage_blob_sources(
                    f"{generator} --output_dir {{}}", op_dir, sys.executable
                )

            staging_dir = os.path.join(op_dir, jit_cache.STAGING_DIRECTORY_NAME)
            self.assertEqual(_read(complete_source), "// known-good\n")
            self.assertEqual(
                _read(os.path.join(staging_dir, "complete.cpp")), "// known-good\n"
            )
            self.assertFalse(os.path.exists(os.path.join(staging_dir, "partial.cpp")))
            self.assertFalse(
                os.path.exists(
                    os.path.join(staging_dir, jit_cache.CODEGEN_INCOMPLETE_MARKER)
                )
            )
            self.assertEqual(_transaction_artifacts(op_dir), [])

    def test_staging_path_and_unchanged_mtime_are_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            op_dir = os.path.join(tmp, "module")
            blob_source = os.path.join(op_dir, "blob", "generated.cpp")
            _write(blob_source, "// stable\n")
            stable_mtime_ns = 1_700_000_000_000_000_000
            os.utime(blob_source, ns=(stable_mtime_ns, stable_mtime_ns))
            generator = _write_generator(
                tmp,
                """import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()
path = os.path.join(args.output_dir, "generated.cpp")
contents = "// stable\\n"
old = None
try:
    with open(path) as source:
        old = source.read()
except FileNotFoundError:
    pass
if old != contents:
    with open(path, "w") as source:
        source.write(contents)
""",
            )

            first = jit_cache.stage_blob_sources(
                f"{generator} --output_dir {{}}", op_dir, sys.executable
            )
            first_mtime_ns = os.stat(os.path.join(first, "generated.cpp")).st_mtime_ns
            second = jit_cache.stage_blob_sources(
                f"{generator} --output_dir {{}}", op_dir, sys.executable
            )
            second_mtime_ns = os.stat(os.path.join(second, "generated.cpp")).st_mtime_ns

            self.assertEqual(first, os.path.join(op_dir, "blob.staging"))
            self.assertEqual(second, first)
            self.assertEqual(first_mtime_ns, stable_mtime_ns)
            self.assertEqual(second_mtime_ns, stable_mtime_ns)

    def test_successful_codegen_is_published_only_after_explicit_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            op_dir = os.path.join(tmp, "module")
            blob_dir = os.path.join(op_dir, "blob")
            old_source = os.path.join(blob_dir, "old.cpp")
            _write(old_source, "// old\n")
            generator = _write_generator(
                tmp,
                """import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()
with open(args.output_dir + "/new.cpp", "w") as source:
    source.write("// complete\\n")
""",
            )

            staging_dir = jit_cache.stage_blob_sources(
                f"{generator} --output_dir {{}}", op_dir, sys.executable
            )
            self.assertTrue(os.path.exists(os.path.join(staging_dir, "new.cpp")))
            self.assertFalse(os.path.exists(os.path.join(blob_dir, "new.cpp")))

            jit_cache.publish_blob_sources(staging_dir, blob_dir)

            self.assertEqual(_read(os.path.join(blob_dir, "new.cpp")), "// complete\n")
            self.assertTrue(os.path.isdir(staging_dir))
            self.assertEqual(
                stat.S_IMODE(os.stat(blob_dir).st_mode),
                stat.S_IMODE(os.stat(op_dir).st_mode),
            )
            self.assertEqual(_transaction_artifacts(op_dir), [])

    def test_failed_blob_publication_restores_previous_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            op_dir = os.path.join(tmp, "module")
            blob_dir = os.path.join(op_dir, "blob")
            _write(os.path.join(blob_dir, "old.cpp"), "// old\n")
            generator = _write_generator(
                tmp,
                """import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()
with open(args.output_dir + "/new.cpp", "w") as source:
    source.write("// new\\n")
""",
            )
            staging_dir = jit_cache.stage_blob_sources(
                f"{generator} --output_dir {{}}", op_dir, sys.executable
            )
            real_replace = jit_cache._replace

            def fail_candidate_publish(source, destination, *args, **kwargs):
                if destination == blob_dir and os.path.basename(source).startswith(
                    ".blob-publish-"
                ):
                    raise OSError("simulated publication failure")
                return real_replace(source, destination, *args, **kwargs)

            with mock.patch.object(
                jit_cache, "_replace", side_effect=fail_candidate_publish
            ), self.assertRaisesRegex(OSError, "publication failure"):
                jit_cache.publish_blob_sources(staging_dir, blob_dir)

            self.assertEqual(_read(os.path.join(blob_dir, "old.cpp")), "// old\n")
            self.assertTrue(os.path.exists(os.path.join(staging_dir, "new.cpp")))
            self.assertEqual(_transaction_artifacts(op_dir), [])

    def test_publish_race_keeps_peer_cache_and_reaps_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            op_dir = os.path.join(tmp, "module")
            blob_dir = os.path.join(op_dir, "blob")
            _write(os.path.join(blob_dir, "old.cpp"), "// old\n")
            generator = _write_generator(
                tmp,
                """import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()
with open(args.output_dir + "/ours.cpp", "w") as source:
    source.write("// ours\\n")
""",
            )
            staging_dir = jit_cache.stage_blob_sources(
                f"{generator} --output_dir {{}}", op_dir, sys.executable
            )
            real_replace = jit_cache._replace
            injected_peer = False

            def inject_peer_publish(source, destination, *args, **kwargs):
                nonlocal injected_peer
                if (
                    not injected_peer
                    and destination == blob_dir
                    and os.path.basename(source).startswith(".blob-publish-")
                ):
                    injected_peer = True
                    _write(os.path.join(blob_dir, "peer.cpp"), "// peer\n")
                return real_replace(source, destination, *args, **kwargs)

            with mock.patch.object(
                jit_cache, "_replace", side_effect=inject_peer_publish
            ), self.assertRaises(OSError):
                jit_cache.publish_blob_sources(staging_dir, blob_dir)

            self.assertEqual(_read(os.path.join(blob_dir, "peer.cpp")), "// peer\n")
            self.assertEqual(_transaction_artifacts(op_dir), [])

    def test_snapshot_does_not_reuse_same_stat_file_with_changed_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            op_dir = os.path.join(tmp, "module")
            blob_dir = os.path.join(op_dir, "blob")
            blob_source = os.path.join(blob_dir, "same.cpp")
            _write(blob_source, "// old\n")
            generator = _write_generator(
                tmp,
                """import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()
with open(args.output_dir + "/same.cpp", "w") as source:
    source.write("// new\\n")
""",
            )
            staging_dir = jit_cache.stage_blob_sources(
                f"{generator} --output_dir {{}}", op_dir, sys.executable
            )
            staged_source = os.path.join(staging_dir, "same.cpp")
            blob_stat = os.stat(blob_source)
            os.utime(
                staged_source,
                ns=(blob_stat.st_atime_ns, blob_stat.st_mtime_ns),
            )

            jit_cache.publish_blob_sources(staging_dir, blob_dir)

            self.assertEqual(_read(os.path.join(blob_dir, "same.cpp")), "// new\n")

    def test_header_only_codegen_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            op_dir = os.path.join(tmp, "module")
            generator = _write_generator(
                tmp,
                """import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()
with open(args.output_dir + "/generated.hpp", "w") as header:
    header.write("// generated build input\\n")
""",
            )

            staging_dir = jit_cache.stage_blob_sources(
                f"{generator} --output_dir {{}}", op_dir, sys.executable
            )

            self.assertTrue(os.path.exists(os.path.join(staging_dir, "generated.hpp")))

    def test_codegen_that_produces_no_build_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            op_dir = os.path.join(tmp, "module")
            generator = _write_generator(tmp, "# successful but produced nothing\n")

            with self.assertRaisesRegex(
                RuntimeError, r"produced no C\+\+/HIP build inputs"
            ):
                jit_cache.stage_blob_sources(generator, op_dir, sys.executable)

            staging_dir = os.path.join(op_dir, jit_cache.STAGING_DIRECTORY_NAME)
            self.assertTrue(os.path.isdir(staging_dir))
            self.assertFalse(
                os.path.exists(
                    os.path.join(staging_dir, jit_cache.CODEGEN_INCOMPLETE_MARKER)
                )
            )

    def test_seed_file_is_copied_into_transaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            op_dir = os.path.join(tmp, "module")
            sidecar = os.path.join(tmp, "compiled_kids_opus.json")
            _write(sidecar, "[1, 7]\n")
            generator = _write_generator(
                tmp,
                """import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()
with open(args.output_dir + "/generated.hpp", "w") as header:
    header.write("// generated\\n")
""",
            )

            staging_dir = jit_cache.stage_blob_sources(
                f"{generator} --output_dir {{}}",
                op_dir,
                sys.executable,
                seed_files=[(sidecar, "compiled_kids_opus.json")],
            )

            self.assertEqual(
                _read(os.path.join(staging_dir, "compiled_kids_opus.json")),
                "[1, 7]\n",
            )

    def test_failed_sidecar_seed_restores_published_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            op_dir = os.path.join(tmp, "module")
            blob_dir = os.path.join(op_dir, "blob")
            _write(os.path.join(blob_dir, "generated.hpp"), "// known-good\n")
            _write(os.path.join(blob_dir, "compiled_kids_opus.json"), "[1]\n")
            sidecar = os.path.join(tmp, "compiled_kids_opus.json")
            _write(sidecar, "[1, 7]\n")
            generator = _write_generator(tmp, "# seed fails before this runs\n")
            real_copy2 = jit_cache._copy2

            def fail_sidecar_seed(source, destination, *args, **kwargs):
                if source == sidecar:
                    _write(destination, "[")
                    raise OSError("simulated sidecar seed failure")
                return real_copy2(source, destination, *args, **kwargs)

            with mock.patch.object(
                jit_cache, "_copy2", side_effect=fail_sidecar_seed
            ), self.assertRaisesRegex(OSError, "sidecar seed failure"):
                jit_cache.stage_blob_sources(
                    generator,
                    op_dir,
                    sys.executable,
                    seed_files=[(sidecar, "compiled_kids_opus.json")],
                )

            staging_sidecar = os.path.join(
                op_dir,
                jit_cache.STAGING_DIRECTORY_NAME,
                "compiled_kids_opus.json",
            )
            self.assertEqual(_read(staging_sidecar), "[1]\n")

    def test_abandoned_artifacts_are_reaped(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(os.path.join(tmp, "blob", "published.cpp"), "// published\n")
            abandoned = [
                os.path.join(tmp, ".blob-old-random-stage"),
                os.path.join(tmp, ".blob-publish-old"),
                os.path.join(tmp, ".blob-backup-old"),
            ]
            for path in abandoned:
                _write(os.path.join(path, "source.cpp"), "// abandoned\n")

            jit_cache.cleanup_abandoned_blob_artifacts(tmp, max_age_seconds=0)

            self.assertFalse(any(os.path.exists(path) for path in abandoned))

    def test_dead_owner_is_reaped_without_waiting_for_legacy_grace(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(os.path.join(tmp, "blob", "published.cpp"), "// published\n")
            # Use an actual, already-reaped child PID rather than guessing a PID.
            child = subprocess.Popen([sys.executable, "-c", "pass"])
            child.wait()
            for kind in ("publish", "backup", "reset"):
                name = f".blob-{kind}-{socket.gethostname()}.{child.pid}.dead"
                _write(os.path.join(tmp, name, "x.cpp"), "// abandoned\n")
            legacy = os.path.join(tmp, ".blob-old-format")
            _write(os.path.join(legacy, "x.cpp"), "// recent legacy\n")
            jit_cache.cleanup_abandoned_blob_artifacts(tmp)
            self.assertEqual(set(os.listdir(tmp)), {"blob", os.path.basename(legacy)})

    def test_only_backup_survives_cleanup_even_after_owner_dies(self):
        with tempfile.TemporaryDirectory() as tmp:
            child = subprocess.Popen([sys.executable, "-c", "pass"])
            child.wait()
            names = [
                f".blob-backup-{socket.gethostname()}.{child.pid}.dead",
                ".blob-backup-old-format",
                "blob.backup.legacy",
            ]
            for name in names:
                _write(os.path.join(tmp, name, "published.cpp"), "// keep\n")
            jit_cache.cleanup_abandoned_blob_artifacts(tmp, max_age_seconds=0)
            self.assertEqual(set(os.listdir(tmp)), set(names))

    def test_live_remote_and_stable_trees_survive_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            names = [
                jit_cache._transaction_prefix("publish") + "live",
                ".blob-publish-remote.invalid.123.remote",
                "blob",
                "blob.staging",
            ]
            for name in names:
                path = os.path.join(tmp, name)
                _write(os.path.join(path, "x.cpp"), "// retained\n")
                os.utime(path, (1, 1))
            jit_cache.cleanup_abandoned_blob_artifacts(tmp, max_age_seconds=0)
            self.assertEqual(set(os.listdir(tmp)), set(names))

    def test_candidate_and_published_modes_follow_op_dir(self):
        for mode in (0o755, 0o750):
            with self.subTest(mode=oct(mode)), tempfile.TemporaryDirectory() as tmp:
                op_dir = os.path.join(tmp, "module")
                staging = os.path.join(op_dir, "blob.staging")
                _write(os.path.join(staging, "sub", "generated.cpp"), "// readable\n")
                _write(
                    os.path.join(staging, jit_cache.CODEGEN_COMPLETE_MARKER), "ready"
                )
                os.chmod(op_dir, mode)
                os.chmod(staging, mode)
                os.chmod(os.path.join(staging, "sub"), mode)
                os.chmod(os.path.join(staging, "sub", "generated.cpp"), 0o644)
                original_snapshot = jit_cache._copy_blob_snapshot

                def check_candidate(
                    source, destination, previous, expected=mode, copy=original_snapshot
                ):
                    self.assertEqual(
                        stat.S_IMODE(os.stat(destination).st_mode), expected
                    )
                    return copy(source, destination, previous)

                blob = os.path.join(op_dir, "blob")
                with mock.patch.object(
                    jit_cache, "_copy_blob_snapshot", side_effect=check_candidate
                ):
                    jit_cache.publish_blob_sources(staging, blob)
                for path in (blob, os.path.join(blob, "sub")):
                    self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), mode)
                self.assertEqual(
                    stat.S_IMODE(
                        os.stat(os.path.join(blob, "sub", "generated.cpp")).st_mode
                    ),
                    0o644,
                )

    def test_atomic_copy_keeps_previous_binary_on_copy_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "new.so")
            destination = os.path.join(tmp, "jit", "module.so")
            os.makedirs(os.path.dirname(destination))
            with open(source, "wb") as output:
                output.write(b"new")
            with open(destination, "wb") as output:
                output.write(b"known-good")

            def fail_after_partial_copy(_source, temporary_path, *args, **kwargs):
                del args, kwargs
                with open(temporary_path, "wb") as output:
                    output.write(b"partial")
                raise OSError("simulated interrupted copy")

            with mock.patch.object(
                jit_cache, "_copy2", side_effect=fail_after_partial_copy
            ), self.assertRaisesRegex(OSError, "interrupted copy"):
                jit_cache.atomic_copy(source, destination)

            with open(destination, "rb") as output:
                self.assertEqual(output.read(), b"known-good")
            self.assertEqual(os.listdir(os.path.dirname(destination)), ["module.so"])


class TestJitCacheRecovery(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = temporary.name
        self.op_dir = os.path.join(self.root, "module")
        self.blob_dir = os.path.join(self.op_dir, "blob")
        self.staging_dir = os.path.join(self.op_dir, jit_cache.STAGING_DIRECTORY_NAME)
        _write(os.path.join(self.blob_dir, "first.cpp"), "// first\n")
        _write(os.path.join(self.blob_dir, "second.cpp"), "// second\n")
        os.chmod(self.op_dir, 0o755)
        self.generator = _write_generator(self.root, "pass\n")

    def _restore_with_copy_failure(self):
        original_copy = jit_cache._copy2

        def fail_second_file(source, destination, *args, **kwargs):
            # Recovery must correct mkdtemp's 0700 before copying any sources.
            candidate = os.path.dirname(destination)
            self.assertTrue(os.path.basename(candidate).startswith(".blob-reset-"))
            self.assertEqual(stat.S_IMODE(os.stat(candidate).st_mode), 0o755)
            if os.path.basename(source) == "second.cpp":
                raise OSError("simulated partial restore failure")
            return original_copy(source, destination, *args, **kwargs)

        return mock.patch.object(jit_cache, "_copy2", side_effect=fail_second_file)

    def _assert_complete_retry(self):
        jit_cache.stage_blob_sources(self.generator, self.op_dir, sys.executable)
        for name, contents in (
            ("first.cpp", "// first\n"),
            ("second.cpp", "// second\n"),
        ):
            self.assertEqual(_read(os.path.join(self.staging_dir, name)), contents)
            self.assertEqual(_read(os.path.join(self.blob_dir, name)), contents)
        self.assertEqual(stat.S_IMODE(os.stat(self.staging_dir).st_mode), 0o755)
        self.assertEqual(_transaction_artifacts(self.op_dir), [])

    def test_initial_restore_copy_failure_does_not_install_partial_staging(self):
        with self._restore_with_copy_failure(), self.assertRaisesRegex(
            OSError, "partial restore failure"
        ):
            jit_cache.stage_blob_sources(self.generator, self.op_dir, sys.executable)
        self.assertFalse(os.path.lexists(self.staging_dir))
        self._assert_complete_retry()

    def test_failed_codegen_rollback_copy_failure_can_retry_in_same_process(self):
        jit_cache.stage_blob_sources(self.generator, self.op_dir, sys.executable)
        failing_generator = os.path.join(self.root, "failing.py")
        _write(failing_generator, "raise SystemExit(7)\n")
        with self._restore_with_copy_failure(), self.assertRaises(
            subprocess.CalledProcessError
        ):
            jit_cache.stage_blob_sources(failing_generator, self.op_dir, sys.executable)
        # The failed generator's own live PID must not strand an incomplete
        # marker at the stable path after its rollback copy also fails.
        self.assertFalse(os.path.lexists(self.staging_dir))
        self._assert_complete_retry()

    def test_abrupt_restore_owner_exit_is_recovered_and_reclaimed(self):
        worker = """import importlib.util
import os
import sys
spec = importlib.util.spec_from_file_location("cache_child", sys.argv[1])
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)
copy = cache._copy2
def die_mid_copy(source, destination, *args, **kwargs):
    if os.path.basename(source) == "second.cpp":
        os._exit(99)
    return copy(source, destination, *args, **kwargs)
cache._copy2 = die_mid_copy
cache.stage_blob_sources(sys.argv[3], sys.argv[2], sys.executable)
"""
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                worker,
                str(JIT_CACHE_PATH),
                self.op_dir,
                self.generator,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(child.returncode, 99, child.stderr)
        self.assertFalse(os.path.lexists(self.staging_dir))
        self.assertTrue(_transaction_artifacts(self.op_dir))
        self._assert_complete_retry()

    def test_fresh_legacy_backup_formats_recover_without_age_delay(self):
        for name in (".blob-backup-old-format", "blob.backup.legacy"):
            with self.subTest(name=name):
                backup = os.path.join(self.op_dir, name)
                os.replace(self.blob_dir, backup)
                os.utime(backup, None)
                jit_cache._recover_blob_backup(self.blob_dir)
                self.assertFalse(os.path.lexists(backup))
                self.assertEqual(
                    _read(os.path.join(self.blob_dir, "second.cpp")), "// second\n"
                )

    def test_recovery_does_not_take_live_peer_or_remote_backups(self):
        live_backup = os.path.join(
            self.op_dir,
            f".blob-backup-{socket.gethostname()}.{os.getppid()}.live",
        )
        remote_backup = os.path.join(
            self.op_dir, f".blob-backup-{socket.gethostname()}.remote.123.remote"
        )
        os.replace(self.blob_dir, live_backup)
        _write(os.path.join(remote_backup, "remote.cpp"), "// remote\n")
        jit_cache._recover_blob_backup(self.blob_dir)
        self.assertFalse(os.path.lexists(self.blob_dir))
        self.assertTrue(os.path.isdir(live_backup))
        self.assertTrue(os.path.isdir(remote_backup))


class TestModuleBuildLock(unittest.TestCase):
    def test_wait_policy_preserves_default_and_retries_request_or_stale_lock(self):
        for force, normal_release in ((False, True), (True, True), (False, False)):
            with self.subTest(force=force, normal_release=normal_release):
                baton = mock.Mock()
                baton.try_acquire.side_effect = [False, True]
                baton.wait.return_value = normal_release
                lock = _load_functions(
                    JIT_CACHE_PATH.parents[1] / "core.py",
                    ["mp_lock"],
                    {"Callable": Callable, "FileBaton": mock.Mock(return_value=baton)},
                )["mp_lock"]
                main, final, waiter = mock.Mock(), mock.Mock(), mock.Mock()
                result = lock(
                    "module.lock", main, final, waiter, build_after_wait=force
                )
                if normal_release and not force:
                    self.assertIs(result, waiter.return_value)
                    main.assert_not_called()
                    final.assert_not_called()
                    baton.release.assert_not_called()
                else:
                    self.assertIs(result, main.return_value)
                    main.assert_called_once()
                    final.assert_called_once()
                    waiter.assert_not_called()
                    baton.release.assert_called_once()


class TestBuildPublication(unittest.TestCase):
    def setUp(self):
        environment = mock.patch.dict(os.environ, {"AITER_REBUILD": "0"})
        environment.start()
        self.addCleanup(environment.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = temporary.name
        self.bd_dir = os.path.join(self.root, "build")
        self.op_dir = os.path.join(self.bd_dir, "module_deepgemm_opus")
        self.sidecar = os.path.join(self.bd_dir, "compiled_kids_opus.json")
        self.artifact = os.path.join(self.root, "module_deepgemm_opus.so")
        self.logger = mock.Mock()
        self.generator = _write_generator(
            self.root,
            """import argparse
import json
import os
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", required=True)
parser.add_argument("--extra_kids", type=int, nargs="*", default=[])
args = parser.parse_args()
path = os.path.join(args.output_dir, "compiled_kids_opus.json")
kids = set(json.load(open(path))) if os.path.exists(path) else set()
kids = sorted(kids | set(args.extra_kids) | {1})
with open(path, "w") as output:
    json.dump(kids, output)
with open(os.path.join(args.output_dir, "generated.cpp"), "w") as output:
    output.write("// generated: " + json.dumps(kids))
""",
        )

        self.ninja_targets = []

        def fake_ninja(**kwargs):
            self.ninja_targets.append(kwargs["name"])
            staged = os.path.join(
                self.op_dir, "blob.staging", "compiled_kids_opus.json"
            )
            _write(
                os.path.join(kwargs["build_directory"], kwargs["name"] + ".so"),
                _read(staged),
            )

        # Run the real JIT/versioner/baton control flow. Only the HIP/Ninja build
        # and binary loading are stubbed; fake outputs represent compiled kids.
        self.compiler = _load_functions(
            JIT_CACHE_PATH.with_name("cpp_extension.py"),
            ["_jit_compile"],
            {
                "os": os,
                "sys": sys,
                "JIT_EXTENSION_VERSIONER": versioner_module.ExtensionVersioner(),
                "FileBaton": baton_module.FileBaton,
                "GeneratedFileCleaner": lambda **kwargs: contextlib.nullcontext(),
                "IS_HIP_EXTENSION": False,
                "_write_ninja_file_and_build_library": fake_ninja,
                "_import_module_from_library": lambda *args: None,
                "_get_exec_path": lambda name, directory: os.path.join(directory, name),
            },
        )
        version = lambda value: tuple(int(part) for part in value.split("."))
        namespace = {
            "os": os,
            "shutil": shutil,
            "sys": sys,
            "time": time,
            "multiprocessing": multiprocessing,
            "re": re,
            "traceback": traceback,
            "logger": self.logger,
            "bd_dir": self.bd_dir,
            "PY": sys.executable,
            "AITER_REBUILD": 0,
            "AITER_LOG_MORE": 0,
            "AITER_DISABLE_KERNARG_PRELOAD": True,
            "AITER_ROOT_DIR": self.root,
            "AITER_CSRC_DIR": self.root,
            "CK_3RDPARTY_DIR": os.path.join(self.root, "absent_ck"),
            "HIP_KITTENS_DIR": os.path.join(self.root, "absent_kittens"),
            "get_user_jit_dir": lambda: self.root,
            "get_hip_version": lambda: "7.0.0",
            "parse": version,
            "Version": version,
            "get_gfx": lambda: "gfx942",
            "check_LLVM_MAIN_REVISION": lambda: 0,
            "validate_and_update_archs": lambda: ["gfx942"],
            "hip_flag_checker": lambda _flag: True,
            "check_and_set_ninja_worker": lambda: None,
            "stage_blob_sources": jit_cache.stage_blob_sources,
            "publish_blob_sources": jit_cache.publish_blob_sources,
            "publish_compiled_kids": jit_cache.publish_compiled_kids,
            "snapshot_compiled_kids": jit_cache.snapshot_compiled_kids,
            "require_blob_generation": jit_cache.require_blob_generation,
            "atomic_copy": jit_cache.atomic_copy,
            "mp_lock": lambda **kwargs: kwargs["MainFunc"](),
            "rm_module": lambda _name: (
                os.remove(self.artifact) if os.path.exists(self.artifact) else None
            ),
            "clear_build": lambda _name: shutil.rmtree(self.op_dir, ignore_errors=True),
            "_jit_compile": self.compiler["_jit_compile"],
        }
        self.core = _load_functions(
            JIT_CACHE_PATH.parents[1] / "core.py",
            ["build_module", "_stage_blob_sources", "rename_cpp_to_cu"],
            namespace,
        )

    def build(self, *kids):
        self.core["build_module"](
            md_name="module_deepgemm_opus",
            srcs=[],
            flags_extra_cc=[],
            flags_extra_hip=[],
            blob_gen_cmd=f"{self.generator} --output_dir {{}} --extra_kids "
            + " ".join(map(str, kids)),
            extra_include=[],
            extra_ldflags=[],
            verbose=False,
            is_python_module=True,
            is_standalone=False,
            torch_exclude=True,
            third_party=[],
        )

    def test_successful_sidecar_survives_clear_build_and_seeds_next_build(self):
        self.build(7)
        self.assertTrue(
            jit_cache.compiled_kids_are_current(self.sidecar, self.artifact, {1, 7})
        )
        self.core["AITER_REBUILD"] = 1
        self.build(9)
        self.assertEqual(set(json.loads(_read(self.artifact))), {1, 7, 9})
        self.assertTrue(
            jit_cache.compiled_kids_are_current(self.sidecar, self.artifact, {1, 7, 9})
        )

    def test_repeated_build_installs_new_kids_under_stable_target_name(self):
        self.build(7)
        self.build(9)
        self.assertEqual(self.ninja_targets, ["module_deepgemm_opus"] * 2)
        self.assertEqual(set(json.loads(_read(self.artifact))), {1, 7, 9})
        self.assertTrue(
            jit_cache.compiled_kids_are_current(self.sidecar, self.artifact, {9})
        )
        self.assertEqual(self.compiler["JIT_EXTENSION_VERSIONER"].entries, {})

    def test_failed_ninja_retry_in_same_process_does_not_skip_compilation(self):
        self.build(7)
        original_ninja = self.compiler["_write_ninja_file_and_build_library"]
        self.compiler["_write_ninja_file_and_build_library"] = mock.Mock(
            side_effect=OSError("hipcc OOM")
        )
        with self.assertRaisesRegex(RuntimeError, "build .* failed"):
            self.build(9)
        self.assertFalse(
            jit_cache.compiled_kids_are_current(self.sidecar, self.artifact, {9})
        )
        self.assertFalse(os.path.exists(os.path.join(self.op_dir, "build", "lock")))
        self.compiler["_write_ninja_file_and_build_library"] = original_ninja
        self.build(9)
        self.assertEqual(self.ninja_targets, ["module_deepgemm_opus"] * 2)
        self.assertEqual(set(json.loads(_read(self.artifact))), {1, 7, 9})
        self.assertTrue(
            jit_cache.compiled_kids_are_current(self.sidecar, self.artifact, {9})
        )

    def test_interrupted_ninja_retry_releases_lock_and_recompiles(self):
        self.build(7)
        original_ninja = self.compiler["_write_ninja_file_and_build_library"]
        self.compiler["_write_ninja_file_and_build_library"] = mock.Mock(
            side_effect=KeyboardInterrupt()
        )
        with self.assertRaises(KeyboardInterrupt):
            self.build(9)
        self.assertFalse(os.path.exists(os.path.join(self.op_dir, "build", "lock")))
        self.compiler["_write_ninja_file_and_build_library"] = original_ninja
        self.build(9)
        self.assertEqual(set(json.loads(_read(self.artifact))), {1, 7, 9})

    def test_deleted_build_output_is_recreated_for_identical_inputs(self):
        self.build(7)
        os.remove(os.path.join(self.op_dir, "build", "module_deepgemm_opus.so"))
        self.build(7)
        self.assertEqual(self.ninja_targets, ["module_deepgemm_opus"] * 2)
        self.assertTrue(
            jit_cache.compiled_kids_are_current(self.sidecar, self.artifact, {7})
        )

    def test_rebuild_level_two_removes_binary_but_keeps_objects_and_sidecar(self):
        self.build(7)
        sentinel = os.path.join(self.op_dir, "build", "kept.o")
        _write(sentinel, "object")
        self.core["AITER_REBUILD"] = 2
        original_ninja = self.compiler["_write_ninja_file_and_build_library"]

        def check_rebuild_state(**kwargs):
            self.assertFalse(os.path.exists(self.artifact))
            self.assertEqual(_read(sentinel), "object")
            self.assertEqual(set(json.loads(_read(self.sidecar))), {1, 7})
            return original_ninja(**kwargs)

        self.compiler["_write_ninja_file_and_build_library"] = check_rebuild_state
        self.build(9)
        self.assertEqual(_read(sentinel), "object")
        self.assertTrue(
            jit_cache.compiled_kids_are_current(self.sidecar, self.artifact, {9})
        )

    def test_codegen_failure_preserves_formatted_runtime_error_contract(self):
        self.build(7)
        _write(self.generator, "raise SystemExit(7)\n")
        with self.assertRaisesRegex(RuntimeError, "build .* failed") as failure:
            self.build(9)
        self.assertIsInstance(
            failure.exception.__cause__, subprocess.CalledProcessError
        )
        self.logger.error.assert_called_once()
        self.assertIn("failed jit build", self.logger.error.call_args.args[0])
        self.assertEqual(set(json.loads(_read(self.artifact))), {1, 7})

    def test_module_without_codegen_does_not_require_a_generation(self):
        sources = ["user.cu"]
        result = self.core["_stage_blob_sources"](
            [], self.op_dir, self.root, sources, False
        )
        self.assertEqual(result, (sources, None, None))
        self.assertFalse(os.path.exists(self.op_dir))

    def test_codegen_returns_own_token_even_if_peer_finishes_before_return(self):
        self.build(7)
        original_replace = jit_cache._replace

        def replace_then_peer_token(source, destination, *args, **kwargs):
            result = original_replace(source, destination, *args, **kwargs)
            if destination.endswith(jit_cache.CODEGEN_COMPLETE_MARKER):
                _write(destination, "peer-generation")
            return result

        self.core["_jit_compile"] = mock.Mock()
        with mock.patch.object(
            jit_cache, "_replace", side_effect=replace_then_peer_token
        ), self.assertRaisesRegex(RuntimeError, "build .* failed"):
            self.build(9)
        self.core["_jit_compile"].assert_not_called()
        self.assertEqual(set(json.loads(_read(self.artifact))), {1, 7})
        self.assertTrue(
            jit_cache.compiled_kids_are_current(self.sidecar, self.artifact, {7})
        )

    def test_failed_compile_does_not_advance_sidecar(self):
        self.build(7)
        self.core["AITER_REBUILD"] = 1
        self.core["_jit_compile"] = mock.Mock(side_effect=OSError("hipcc OOM"))
        with self.assertRaisesRegex(RuntimeError, "build .* failed"):
            self.build(9)
        self.assertEqual(set(json.loads(_read(self.sidecar))), {1, 7})
        self.assertFalse(
            jit_cache.compiled_kids_are_current(self.sidecar, self.artifact, {7})
        )

    def test_publish_race_does_not_fail_build_or_skip_sidecar_update(self):
        self.build(7)
        original_replace = jit_cache._replace

        def peer_wins(source, destination, *args, **kwargs):
            if os.path.basename(source).startswith(
                ".blob-publish-"
            ) and destination.endswith("/blob"):
                _write(os.path.join(destination, "peer.cpp"), "// peer")
            return original_replace(source, destination, *args, **kwargs)

        with mock.patch.object(jit_cache, "_replace", side_effect=peer_wins):
            self.build(9)
        self.assertEqual(set(json.loads(_read(self.artifact))), {1, 7, 9})
        self.assertTrue(
            jit_cache.compiled_kids_are_current(self.sidecar, self.artifact, {9})
        )
        self.logger.warning.assert_called_once()
        self.logger.error.assert_not_called()
        self.assertEqual(_transaction_artifacts(self.op_dir), [])

    def test_metadata_failure_keeps_binary_but_invalidates_tuner_fast_path(self):
        self.build(7)
        original_replace = jit_cache._replace

        def fail_receipt(source, destination, *args, **kwargs):
            if destination == self.sidecar + ".receipt":
                raise PermissionError("receipt denied")
            return original_replace(source, destination, *args, **kwargs)

        with mock.patch.object(jit_cache, "_replace", side_effect=fail_receipt):
            self.build(9)
        self.assertEqual(set(json.loads(_read(self.artifact))), {1, 7, 9})
        self.assertFalse(
            jit_cache.compiled_kids_are_current(self.sidecar, self.artifact, {7})
        )
        self.logger.warning.assert_called_once()
        self.logger.error.assert_not_called()

    def test_modified_legacy_or_wrong_binary_metadata_is_not_trusted(self):
        self.build(7)
        _write(self.sidecar, "[1, 7, 99]")
        self.assertFalse(
            jit_cache.compiled_kids_are_current(self.sidecar, self.artifact, {99})
        )
        self.build(7)
        replacement = os.path.join(self.root, "replacement.so")
        _write(replacement, _read(self.artifact))
        jit_cache.atomic_copy(replacement, self.artifact)
        self.assertFalse(
            jit_cache.compiled_kids_are_current(self.sidecar, self.artifact, {7})
        )
        os.remove(self.sidecar + ".receipt")
        self.assertFalse(
            jit_cache.compiled_kids_are_current(self.sidecar, self.artifact, {7})
        )

    def load_tuner(self):
        d_args = {
            "srcs": [],
            "flags_extra_cc": [],
            "flags_extra_hip": [],
            "blob_gen_cmd": f"{self.generator} --output_dir {{}}",
            "extra_include": [],
            "extra_ldflags": [],
            "torch_exclude": True,
        }
        proxy = types.SimpleNamespace(
            bd_dir=self.bd_dir,
            get_user_jit_dir=lambda: self.root,
            AITER_REBUILD=0,
            get_module=mock.Mock(),
            rebuilded_list=[],
            get_args_of_build=lambda _name: d_args,
        )
        calls = []

        def build(**kwargs):
            calls.append(kwargs["blob_gen_cmd"])
            self.core["AITER_REBUILD"] = proxy.AITER_REBUILD
            return self.core["build_module"](**kwargs)

        proxy.build_module = build
        self.tuner_core = proxy
        imports = {
            "aiter.jit": types.SimpleNamespace(core=proxy),
            "aiter.jit.utils.file_baton": types.SimpleNamespace(
                FileBaton=lambda _path: mock.Mock(try_acquire=lambda: True)
            ),
            "aiter.jit.utils.jit_cache": jit_cache,
            "aiter.jit.utils.chip_info": types.SimpleNamespace(
                get_gfx_runtime=lambda: "gfx942"
            ),
            "opus_gemm_common": types.SimpleNamespace(
                heuristic_kids_for_arch=lambda _arches: {1}
            ),
        }
        self.tuner_imports = imports

        def import_dependency(name, *args, **kwargs):
            if name in imports:
                return imports[name]
            return builtins.__import__(name, *args, **kwargs)

        tuner = _load_functions(
            JIT_CACHE_PATH.parents[3] / "csrc" / "opus_gemm" / "opus_gemm_tune.py",
            ["_ensure_kids_compiled"],
            {
                "__builtins__": {**vars(builtins), "__import__": import_dependency},
                "os": os,
                "sys": sys,
                "json": json,
                "HEURISTIC_DEFAULT_KIDS": {1},
                "_opus_sidecar_path": lambda: self.sidecar,
            },
        )["_ensure_kids_compiled"]
        return tuner, calls

    def test_tuner_retries_failed_request_without_trusting_old_membership(self):
        self.build(7)
        tuner, calls = self.load_tuner()
        original_compile = self.core["_jit_compile"]
        with mock.patch.dict(os.environ), mock.patch.object(sys, "stderr"):
            self.assertFalse(tuner({7}))
            self.core["_jit_compile"] = mock.Mock(side_effect=OSError("hipcc OOM"))
            with self.assertRaisesRegex(RuntimeError, "subset-compile rebuild failed"):
                tuner({9})
            self.assertEqual(set(json.loads(_read(self.sidecar))), {1, 7})
            self.core["_jit_compile"] = original_compile
            self.assertTrue(tuner({9}))
            self.assertFalse(tuner({9}))
        self.assertEqual(len(calls), 2)
        self.assertTrue(all("--extra_kids 9" in command for command in calls))
        self.assertTrue(
            jit_cache.compiled_kids_are_current(self.sidecar, self.artifact, {1, 7, 9})
        )

    def test_tuner_forced_rebuild_runs_once_even_when_receipt_matches(self):
        self.build(7)
        tuner, calls = self.load_tuner()
        self.tuner_core.AITER_REBUILD = 1
        with mock.patch.dict(os.environ, {"AITER_REBUILD": "1"}), mock.patch.object(
            sys, "stderr"
        ):
            self.assertTrue(tuner({7}))
            self.assertEqual(os.environ["AITER_REBUILD"], "0")
            self.assertEqual(self.tuner_core.AITER_REBUILD, 1)
            self.assertFalse(tuner({7}))
        self.assertEqual(len(calls), 1)

    def test_tuner_reuse_clears_inherited_flag_on_every_cache_hit_path(self):
        self.build(7)
        for route in ("fast", "waiter", "locked"):
            with self.subTest(route=route):
                tuner, calls = self.load_tuner()
                self.tuner_core.rebuilded_list = ["module_deepgemm_opus"]
                baton = mock.Mock()
                baton.try_acquire.return_value = route != "waiter"
                self.tuner_imports["aiter.jit.utils.file_baton"].FileBaton = mock.Mock(
                    return_value=baton
                )
                receipt_results = [True] if route == "fast" else [False, True]
                with mock.patch.dict(
                    os.environ, {"AITER_REBUILD": "1"}
                ), mock.patch.object(
                    jit_cache, "compiled_kids_are_current", side_effect=receipt_results
                ):
                    self.assertFalse(tuner({7}))
                    self.assertEqual(os.environ["AITER_REBUILD"], "0")
                self.assertEqual(calls, [])

    def test_tuner_failed_forced_rebuild_restores_flags_and_remains_retryable(self):
        self.build(7)
        tuner, calls = self.load_tuner()
        self.tuner_core.AITER_REBUILD = 1
        original_ninja = self.compiler["_write_ninja_file_and_build_library"]
        self.compiler["_write_ninja_file_and_build_library"] = mock.Mock(
            side_effect=OSError("hipcc OOM")
        )
        with mock.patch.dict(os.environ, {"AITER_REBUILD": "1"}), mock.patch.object(
            sys, "stderr"
        ):
            with self.assertRaisesRegex(RuntimeError, "subset-compile rebuild failed"):
                tuner({7})
            self.assertEqual(os.environ["AITER_REBUILD"], "1")
            self.assertEqual(self.tuner_core.AITER_REBUILD, 1)
            self.compiler["_write_ninja_file_and_build_library"] = original_ninja
            self.assertTrue(tuner({7}))
            self.assertEqual(os.environ["AITER_REBUILD"], "0")
        self.assertEqual(len(calls), 2)

    def test_tuner_runtime_probe_fallback_respects_explicit_build_arches(self):
        self.build(7)
        tuner, calls = self.load_tuner()
        self.tuner_imports["aiter.jit.utils.chip_info"].get_gfx_runtime = mock.Mock(
            side_effect=RuntimeError("no rocminfo")
        )
        heuristic = mock.Mock(
            side_effect=lambda arches: {1} if arches == {"gfx942"} else {1, 200}
        )
        self.tuner_imports["opus_gemm_common"].heuristic_kids_for_arch = heuristic
        with mock.patch.dict(os.environ, {"GPU_ARCHS": "gfx942"}):
            self.assertFalse(tuner({7}))
        heuristic.assert_called_once_with({"gfx942"})
        self.assertEqual(calls, [])

    def test_tuner_interruption_restores_environment_and_releases_locks(self):
        self.build(7)
        for interruption in (KeyboardInterrupt, SystemExit):
            for previous_env in (None, "1", "2"):
                with self.subTest(interruption=interruption, env=previous_env):
                    tuner, calls = self.load_tuner()
                    self.tuner_core.AITER_REBUILD = 2
                    baton = mock.Mock(try_acquire=lambda: True)
                    self.tuner_imports["aiter.jit.utils.file_baton"].FileBaton = (
                        mock.Mock(return_value=baton)
                    )
                    with mock.patch.dict(os.environ), mock.patch.dict(
                        self.compiler,
                        _write_ninja_file_and_build_library=mock.Mock(
                            side_effect=interruption()
                        ),
                    ), mock.patch.object(sys, "stderr"):
                        if previous_env is None:
                            os.environ.pop("AITER_REBUILD", None)
                        else:
                            os.environ["AITER_REBUILD"] = previous_env
                        with self.assertRaises(interruption):
                            tuner({7})
                        self.assertEqual(os.environ.get("AITER_REBUILD"), previous_env)
                        self.assertEqual(self.tuner_core.AITER_REBUILD, 2)
                    self.assertEqual(len(calls), 1)
                    baton.release.assert_called_once()
                    self.assertFalse(
                        os.path.exists(os.path.join(self.op_dir, "build", "lock"))
                    )
                    self.assertEqual(set(json.loads(_read(self.sidecar))), {1, 7})

    def test_tuner_module_lock_waiter_executes_its_own_request(self):
        self.build(7)  # A runtime builder's completed binary lacks kid 9.
        tuner, calls = self.load_tuner()
        baton = mock.Mock()
        baton.try_acquire.side_effect = [False, True]
        baton.wait.return_value = True
        self.core["mp_lock"] = _load_functions(
            JIT_CACHE_PATH.parents[1] / "core.py",
            ["mp_lock"],
            {"Callable": Callable, "FileBaton": lambda _path: baton},
        )["mp_lock"]
        with mock.patch.dict(os.environ), mock.patch.object(sys, "stderr"):
            self.assertTrue(tuner({9}))
            self.assertEqual(os.environ["AITER_REBUILD"], "0")
        self.assertEqual(len(calls), 1)
        self.assertEqual(baton.try_acquire.call_count, 2)
        baton.wait.assert_called_once()
        baton.release.assert_called_once()
        self.assertEqual(set(json.loads(_read(self.artifact))), {1, 7, 9})
        self.assertTrue(
            jit_cache.compiled_kids_are_current(self.sidecar, self.artifact, {9})
        )

    def test_tuner_metadata_failure_does_not_loop_or_fail_successful_compile(self):
        self.build(7)
        tuner, calls = self.load_tuner()
        self.core["publish_compiled_kids"] = mock.Mock(
            side_effect=PermissionError("metadata denied")
        )
        with mock.patch.dict(os.environ), mock.patch.object(sys, "stderr"):
            self.assertTrue(tuner({9}))
        self.assertEqual(len(calls), 1)
        self.assertEqual(set(json.loads(_read(self.artifact))), {1, 7, 9})
        self.assertFalse(
            jit_cache.compiled_kids_are_current(self.sidecar, self.artifact, {9})
        )

    def restage(self, kid):
        return jit_cache.stage_blob_sources(
            f"{self.generator} --output_dir {{}} --extra_kids {kid}",
            self.op_dir,
            sys.executable,
            seed_files=[(self.sidecar, "compiled_kids_opus.json")],
        )

    def test_generation_change_during_compile_rejects_install(self):
        self.build(7)
        original_compile = self.core["_jit_compile"]

        def compile_then_restage(*args, **kwargs):
            original_compile(*args, **kwargs)
            self.restage(11)

        self.core["_jit_compile"] = compile_then_restage
        with self.assertRaisesRegex(RuntimeError, "build .* failed") as failure:
            self.build(9)
        self.assertIn("generation changed", str(failure.exception.__cause__))
        self.assertEqual(set(json.loads(_read(self.artifact))), {1, 7})
        self.assertEqual(set(json.loads(_read(self.sidecar))), {1, 7})
        self.assertTrue(
            jit_cache.compiled_kids_are_current(self.sidecar, self.artifact, {7})
        )

    def test_generation_change_during_binary_copy_rejects_install(self):
        self.build(7)
        original_copy = jit_cache._copy2

        def copy_then_restage(source, destination, *args, **kwargs):
            result = original_copy(source, destination, *args, **kwargs)
            if source.endswith("/build/module_deepgemm_opus.so"):
                self.restage(11)
            return result

        with mock.patch.object(
            jit_cache, "_copy2", side_effect=copy_then_restage
        ), self.assertRaisesRegex(RuntimeError, "build .* failed"):
            self.build(9)
        self.assertEqual(set(json.loads(_read(self.artifact))), {1, 7})
        self.assertTrue(
            jit_cache.compiled_kids_are_current(self.sidecar, self.artifact, {7})
        )

    def test_post_install_restage_cannot_relabel_binary_or_publish_wrong_sources(self):
        self.build(7)
        original_replace = jit_cache._replace

        def install_then_restage(source, destination, *args, **kwargs):
            result = original_replace(source, destination, *args, **kwargs)
            if destination == self.artifact:
                self.restage(11)
            return result

        with mock.patch.object(jit_cache, "_replace", side_effect=install_then_restage):
            self.build(9)
        self.assertEqual(set(json.loads(_read(self.artifact))), {1, 7, 9})
        self.assertEqual(set(json.loads(_read(self.sidecar))), {1, 7, 9})
        self.assertTrue(
            jit_cache.compiled_kids_are_current(self.sidecar, self.artifact, {9})
        )
        self.assertFalse(
            jit_cache.compiled_kids_are_current(self.sidecar, self.artifact, {11})
        )
        self.assertEqual(
            set(
                json.loads(
                    _read(os.path.join(self.op_dir, "blob", "compiled_kids_opus.json"))
                )
            ),
            {1, 7},
        )
        self.logger.warning.assert_called_once()
        self.logger.error.assert_not_called()

    def test_receipt_cannot_bind_to_peer_binary_replacing_ours_during_install(self):
        self.build(7)
        original_replace = jit_cache._replace
        peer = os.path.join(self.root, "peer.so")
        _write(peer, "[1, 11]")

        def replace_then_peer(source, destination, *args, **kwargs):
            result = original_replace(source, destination, *args, **kwargs)
            if destination == self.artifact:
                original_replace(peer, destination)
            return result

        with mock.patch.object(jit_cache, "_replace", side_effect=replace_then_peer):
            self.build(9)
        self.assertEqual(set(json.loads(_read(self.artifact))), {1, 11})
        self.assertFalse(
            jit_cache.compiled_kids_are_current(self.sidecar, self.artifact, {9})
        )
        self.logger.warning.assert_called_once()
        self.logger.error.assert_not_called()

    def test_failed_publication_and_rollback_keep_backup_for_same_process_retry(self):
        self.build(7)
        blob = os.path.join(self.op_dir, "blob")
        original_replace = jit_cache._replace

        def deny_blob_replace(source, destination, *args, **kwargs):
            if destination == blob:
                raise PermissionError("blob publication and recovery denied")
            return original_replace(source, destination, *args, **kwargs)

        with mock.patch.object(jit_cache, "_replace", side_effect=deny_blob_replace):
            self.build(9)
            self.assertFalse(os.path.exists(blob))
            backups = _transaction_artifacts(self.op_dir)
            self.assertEqual(len(backups), 1)
            self.assertTrue(backups[0].startswith(".blob-backup-"))
            backup_sidecar = os.path.join(
                self.op_dir, backups[0], "compiled_kids_opus.json"
            )
            self.assertEqual(set(json.loads(_read(backup_sidecar))), {1, 7})
            # The next locked codegen attempts recovery, but permission is still
            # denied. Neither recovery nor cleanup may destroy the last backup.
            self.restage(11)
            self.assertTrue(os.path.exists(backup_sidecar))
            jit_cache.cleanup_abandoned_blob_artifacts(self.op_dir, max_age_seconds=0)
            self.assertTrue(os.path.exists(backup_sidecar))
        self.assertEqual(set(json.loads(_read(self.artifact))), {1, 7, 9})
        self.logger.warning.assert_called_once()
        self.logger.error.assert_not_called()
        # Recovery also works before this owner PID has exited.
        self.restage(11)
        self.assertEqual(_transaction_artifacts(self.op_dir), [])
        self.assertEqual(
            set(json.loads(_read(os.path.join(blob, "compiled_kids_opus.json")))),
            {1, 7},
        )


class TestCppExtensionControl(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = temporary.name
        self.source = os.path.join(self.root, "source.cpp")
        self.header = os.path.join(self.root, "value.h")
        _write(self.source, '#include "value.h"\nint value() { return VALUE; }\n')
        _write(self.header, "#define VALUE 1\n")
        self.targets = []

        def fake_ninja(**kwargs):
            self.targets.append(kwargs["name"])
            _write(os.path.join(self.root, kwargs["name"] + ".so"), _read(self.header))

        self.namespace = _load_functions(
            JIT_CACHE_PATH.with_name("cpp_extension.py"),
            ["_jit_compile"],
            {
                "os": os,
                "sys": sys,
                "JIT_EXTENSION_VERSIONER": versioner_module.ExtensionVersioner(),
                "FileBaton": baton_module.FileBaton,
                "GeneratedFileCleaner": lambda **kwargs: contextlib.nullcontext(),
                "IS_HIP_EXTENSION": False,
                "_write_ninja_file_and_build_library": fake_ninja,
                "_import_module_from_library": lambda *args: None,
            },
        )

    def compile(self, **options):
        self.namespace["_jit_compile"](
            name="module",
            sources=[self.source],
            extra_cflags=[],
            extra_cuda_cflags=[],
            extra_ldflags=[],
            extra_include_paths=[self.root],
            build_directory=self.root,
            verbose=False,
            with_cuda=False,
            is_python_module=True,
            is_standalone=False,
            torch_exclude=True,
            **options,
        )

    def test_default_extension_loader_keeps_versioned_names_and_cache(self):
        self.compile()
        self.compile()
        self.assertEqual(self.targets, ["module"])
        _write(self.source, _read(self.source) + "// changed source\n")
        self.compile()
        self.assertEqual(self.targets, ["module", "module_v1"])

    def test_stable_target_checks_headers_missing_outputs_and_identical_inputs(self):
        self.compile(use_versioner=False)
        self.compile(use_versioner=False)
        _write(self.header, "#define VALUE 2\n")
        self.compile(use_versioner=False)
        self.assertEqual(
            _read(os.path.join(self.root, "module.so")), _read(self.header)
        )
        os.remove(os.path.join(self.root, "module.so"))
        self.compile(use_versioner=False)
        self.assertEqual(self.targets, ["module"] * 4)
        self.assertTrue(os.path.isfile(os.path.join(self.root, "module.so")))

    def test_stable_target_waiter_enters_ninja_for_its_own_inputs(self):
        baton = mock.Mock()
        baton.try_acquire.side_effect = [False, True]
        baton.wait.return_value = True
        self.namespace["FileBaton"] = mock.Mock(return_value=baton)
        self.compile(use_versioner=False)
        self.assertEqual(self.targets, ["module"])
        self.assertEqual(baton.try_acquire.call_count, 2)
        baton.wait.assert_called_once()
        baton.release.assert_called_once()

    def test_real_cpu_ninja_reuses_unchanged_output_and_rebuilds_header_change(self):
        ninja = shutil.which("ninja")
        if ninja is None:
            candidate = Path(sys.executable).with_name("ninja")
            ninja = str(candidate) if candidate.is_file() else None
        compiler = shutil.which("c++")
        if ninja is None or compiler is None:
            self.skipTest("CPU Ninja integration requires ninja and c++")
        outputs = []

        def run_cpu_ninja(**kwargs):
            target = kwargs["name"] + ".so"
            _write(
                os.path.join(self.root, "build.ninja"),
                "rule compile\n"
                f"  command = {shlex.quote(compiler)} -shared -fPIC -MMD -MF $out.d $in -o $out\n"
                "  depfile = $out.d\n"
                "  deps = gcc\n"
                f"build {target}: compile source.cpp\n"
                f"default {target}\n",
            )
            result = subprocess.run(
                [ninja, "-C", self.root], check=True, capture_output=True, text=True
            )
            outputs.append(result.stdout)

        self.namespace["_write_ninja_file_and_build_library"] = run_cpu_ninja
        artifact = os.path.join(self.root, "module.so")
        self.compile(use_versioner=False)
        unchanged_mtime = os.stat(artifact).st_mtime_ns
        self.compile(use_versioner=False)
        self.assertIn("no work to do", outputs[-1])
        self.assertEqual(os.stat(artifact).st_mtime_ns, unchanged_mtime)
        _write(self.header, "#define VALUE 2\n")
        future = max(time.time_ns(), unchanged_mtime) + 1_000_000_000
        os.utime(self.header, ns=(future, future))
        self.compile(use_versioner=False)
        self.assertNotIn("no work to do", outputs[-1])
        self.assertNotEqual(os.stat(artifact).st_mtime_ns, unchanged_mtime)


@unittest.skipUnless(importlib.util.find_spec("pandas"), "Opus codegen requires pandas")
class TestOpusRequestedKids(unittest.TestCase):
    def test_real_generator_accepts_valid_requests_and_rejects_filtered_requests(self):
        generator = JIT_CACHE_PATH.parents[3] / "csrc/opus_gemm/gen_instances.py"
        runner = (
            "import os, runpy, sys, types; "
            "sys.argv = sys.argv[1:]; "
            "sys.path.insert(0, os.path.dirname(sys.argv[0])); "
            "sys.modules['torch'] = types.SimpleNamespace("
            "cuda=types.SimpleNamespace(is_available=lambda: False)); "
            "runpy.run_path(sys.argv[0], run_name='__main__')"
        )
        cases = (
            (10006, [], True),
            (999999, [], False),
            (200, [], False),
            (200, ["--kernel_tag", "a16w16"], False),
            (10006, ["--kernel_tag", "a8w8"], False),
        )
        for kid, extra_args, accepted in cases:
            with self.subTest(
                kid=kid, extra_args=extra_args
            ), tempfile.TemporaryDirectory() as tmp:
                sidecar = os.path.join(tmp, "compiled_kids.json")
                _write(sidecar, "[]")
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        runner,
                        str(generator),
                        "--working_path",
                        tmp,
                        "--extra_kids",
                        str(kid),
                        *extra_args,
                    ],
                    env={**os.environ, "GPU_ARCHS": "gfx942"},
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if accepted:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(kid, json.loads(_read(sidecar)))
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(
                        "cannot compile requested --extra_kids", result.stderr
                    )
                    self.assertEqual(_read(sidecar), "[]")


if __name__ == "__main__":
    unittest.main()
