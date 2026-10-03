# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only checks for MXFP4 MoE auxiliary source generation."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

GENERATOR = (
    Path(__file__).resolve().parents[1]
    / "csrc/kernels/mxfp4_moe/moe_aux/codegen/gen_instances.py"
)
JIT_CACHE = Path(__file__).resolve().parents[1] / "aiter/jit/utils/jit_cache.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestMxfp4MoeAuxCodegen(unittest.TestCase):
    def test_regeneration_removes_stale_instance_source(self):
        module = load_module("mxfp4_moe_aux_codegen", GENERATOR)

        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir)
            generator = module.mxfp4_moe_aux_codegen(output)
            generator.run()
            stale_source = output / "instances" / "aux_removed_shape.cu"
            stale_source.write_text("// stale from an earlier build\n")
            unrelated = output / "instances" / "notes.txt"
            unrelated.write_text("keep unrelated staging content\n")

            generator.run()

            self.assertFalse(stale_source.exists())
            self.assertTrue(unrelated.exists())
            self.assertTrue(any((output / "instances").glob("*.cu")))

    def test_jit_staging_does_not_publish_removed_instance_source(self):
        cache = load_module("mxfp4_moe_aux_jit_cache", JIT_CACHE)
        with tempfile.TemporaryDirectory() as temporary_dir:
            op_dir = Path(temporary_dir) / "module"
            blob_dir = op_dir / "blob"
            command = f"{GENERATOR} --working_path {{}}"
            stage, token = cache.stage_blob_sources(
                command, str(op_dir), sys.executable, return_token=True
            )
            cache.publish_blob_sources(stage, str(blob_dir), expected_token=token)
            stale_name = "aux_removed_shape.cu"
            for root in (Path(stage), blob_dir):
                (root / "instances" / stale_name).write_text("// old shape\n")

            stage, token = cache.stage_blob_sources(
                command, str(op_dir), sys.executable, return_token=True
            )
            self.assertFalse((Path(stage) / "instances" / stale_name).exists())
            cache.publish_blob_sources(stage, str(blob_dir), expected_token=token)
            self.assertFalse((blob_dir / "instances" / stale_name).exists())


if __name__ == "__main__":
    unittest.main()
