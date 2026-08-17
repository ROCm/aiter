# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Ruye-aa. All rights reserved.
"""CPU-only tests for transactional JIT cache publication."""

import os
import sys
import tempfile
from unittest import mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../aiter/jit/utils"))
import jit_cache  # noqa: E402


def _write_generator(directory, body):
    path = os.path.join(directory, "generator.py")
    with open(path, "w", encoding="utf-8") as generator:
        generator.write(body)
    return path


def test_failed_codegen_does_not_replace_complete_blob_cache():
    with tempfile.TemporaryDirectory() as tmp:
        op_dir = os.path.join(tmp, "module")
        blob_dir = os.path.join(op_dir, "blob")
        os.makedirs(blob_dir)
        complete_source = os.path.join(blob_dir, "complete.cpp")
        with open(complete_source, "w", encoding="utf-8") as source:
            source.write("// known-good\n")

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

        with pytest.raises(jit_cache.subprocess.CalledProcessError):
            jit_cache.stage_blob_sources(
                f"{generator} --output_dir {{}}", op_dir, sys.executable
            )

        with open(complete_source, encoding="utf-8") as source:
            assert source.read() == "// known-good\n"
        assert os.listdir(blob_dir) == ["complete.cpp"]
        assert not [name for name in os.listdir(op_dir) if name.startswith(".blob-")]


def test_successful_codegen_is_published_only_after_explicit_commit():
    with tempfile.TemporaryDirectory() as tmp:
        op_dir = os.path.join(tmp, "module")
        blob_dir = os.path.join(op_dir, "blob")
        os.makedirs(blob_dir)
        old_source = os.path.join(blob_dir, "old.cpp")
        with open(old_source, "w", encoding="utf-8") as source:
            source.write("// old\n")

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
        assert os.path.exists(os.path.join(staging_dir, "new.cpp"))
        assert os.path.exists(old_source)

        jit_cache.publish_blob_sources(staging_dir, blob_dir)

        assert os.listdir(blob_dir) == ["new.cpp"]
        with open(os.path.join(blob_dir, "new.cpp"), encoding="utf-8") as source:
            assert source.read() == "// complete\n"
        assert not [name for name in os.listdir(op_dir) if ".backup." in name]


def test_failed_blob_publication_restores_previous_cache():
    with tempfile.TemporaryDirectory() as tmp:
        blob_dir = os.path.join(tmp, "blob")
        staging_dir = os.path.join(tmp, ".blob-staged")
        os.makedirs(blob_dir)
        os.makedirs(staging_dir)
        with open(os.path.join(blob_dir, "old.cpp"), "w", encoding="utf-8") as source:
            source.write("// old\n")
        with open(
            os.path.join(staging_dir, "new.cpp"), "w", encoding="utf-8"
        ) as source:
            source.write("// new\n")

        real_replace = os.replace

        def fail_staging_publish(source, destination):
            if source == staging_dir:
                raise OSError("simulated publication failure")
            real_replace(source, destination)

        with mock.patch.object(
            jit_cache.os, "replace", side_effect=fail_staging_publish
        ):
            with pytest.raises(OSError, match="publication failure"):
                jit_cache.publish_blob_sources(staging_dir, blob_dir)

        assert os.listdir(blob_dir) == ["old.cpp"]
        assert os.path.exists(staging_dir)
        assert not [name for name in os.listdir(tmp) if ".backup." in name]


def test_codegen_that_produces_no_source_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        op_dir = os.path.join(tmp, "module")
        generator = _write_generator(tmp, "# successful but produced nothing\n")

        with pytest.raises(RuntimeError, match="produced no C\\+\\+/HIP sources"):
            jit_cache.stage_blob_sources(generator, op_dir, sys.executable)

        assert not [name for name in os.listdir(op_dir) if name.startswith(".blob-")]


def test_atomic_copy_keeps_previous_binary_on_copy_failure():
    with tempfile.TemporaryDirectory() as tmp:
        source = os.path.join(tmp, "new.so")
        destination = os.path.join(tmp, "jit", "module.so")
        os.makedirs(os.path.dirname(destination))
        with open(source, "wb") as file:
            file.write(b"new")
        with open(destination, "wb") as file:
            file.write(b"known-good")

        def fail_after_partial_copy(_source, temporary_path):
            with open(temporary_path, "wb") as file:
                file.write(b"partial")
            raise OSError("simulated interrupted copy")

        with mock.patch.object(
            jit_cache.shutil, "copy2", side_effect=fail_after_partial_copy
        ):
            with pytest.raises(OSError, match="interrupted copy"):
                jit_cache.atomic_copy(source, destination)

        with open(destination, "rb") as file:
            assert file.read() == b"known-good"
        assert os.listdir(os.path.dirname(destination)) == ["module.so"]
