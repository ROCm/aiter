# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only tests for transactional JIT cache publication."""

import ast
import builtins
import importlib.util
import json
import multiprocessing
import os
import re
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
from pathlib import Path
from unittest import mock

JIT_CACHE_PATH = (
    Path(__file__).resolve().parents[1] / "aiter" / "jit" / "utils" / "jit_cache.py"
)
JIT_CACHE_SPEC = importlib.util.spec_from_file_location(
    "aiter_jit_cache_transaction_under_test", JIT_CACHE_PATH
)
if JIT_CACHE_SPEC is None or JIT_CACHE_SPEC.loader is None:
    raise RuntimeError(f"cannot load {JIT_CACHE_PATH}")
jit_cache = importlib.util.module_from_spec(JIT_CACHE_SPEC)
JIT_CACHE_SPEC.loader.exec_module(jit_cache)


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
            # Use an actual, already-reaped child PID rather than guessing a PID.
            child = subprocess.Popen([sys.executable, "-c", "pass"])
            child.wait()
            for kind in ("publish", "backup", "reset"):
                name = f".blob-{kind}-{socket.gethostname()}.{child.pid}.dead"
                _write(os.path.join(tmp, name, "x.cpp"), "// abandoned\n")
            legacy = os.path.join(tmp, ".blob-old-format")
            _write(os.path.join(legacy, "x.cpp"), "// recent legacy\n")
            jit_cache.cleanup_abandoned_blob_artifacts(tmp)
            self.assertEqual(os.listdir(tmp), [os.path.basename(legacy)])

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


class TestBuildPublication(unittest.TestCase):
    def setUp(self):
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
with open(path, "w") as output:
    json.dump(sorted(kids | set(args.extra_kids) | {1}), output)
with open(os.path.join(args.output_dir, "generated.cpp"), "w") as output:
    output.write("// generated")
""",
        )

        def fake_compile(_name, _sources, **kwargs):
            staged = os.path.join(
                self.op_dir, "blob.staging", "compiled_kids_opus.json"
            )
            _write(
                os.path.join(kwargs["build_directory"], "module_deepgemm_opus.so"),
                _read(staged),
            )

        version = lambda value: tuple(int(part) for part in value.split("."))
        namespace = {
            "os": os,
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
            "rename_cpp_to_cu": lambda sources, *args, **kwargs: sources,
            "stage_blob_sources": jit_cache.stage_blob_sources,
            "publish_blob_sources": jit_cache.publish_blob_sources,
            "publish_compiled_kids": jit_cache.publish_compiled_kids,
            "atomic_copy": jit_cache.atomic_copy,
            "mp_lock": lambda **kwargs: kwargs["MainFunc"](),
            "rm_module": lambda _name: (
                os.remove(self.artifact) if os.path.exists(self.artifact) else None
            ),
            "clear_build": lambda _name: shutil.rmtree(self.op_dir, ignore_errors=True),
            "_jit_compile": fake_compile,
        }
        self.core = _load_functions(
            JIT_CACHE_PATH.parents[1] / "core.py",
            ["build_module", "_stage_blob_sources"],
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

    def test_tuner_retries_failed_request_without_trusting_old_membership(self):
        self.build(7)
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
        self.assertTrue(all("--extra_kids 1 9" in command for command in calls))
        self.assertTrue(
            jit_cache.compiled_kids_are_current(self.sidecar, self.artifact, {1, 7, 9})
        )


if __name__ == "__main__":
    unittest.main()
