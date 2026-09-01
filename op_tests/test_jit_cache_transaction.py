# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only tests for transactional JIT cache publication."""

import importlib.util
import os
import stat
import subprocess
import sys
import tempfile
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


if __name__ == "__main__":
    unittest.main()
