# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU checks at the MXMOE catalog and generated auxiliary output boundaries."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
AUX_MODULE = ROOT / "aiter/ops/moe_mxfp4_aux.py"
GENERATOR = ROOT / "csrc/kernels/mxfp4_moe/moe_aux/codegen/gen_instances.py"
MODEL_SHAPES = (
    (24, 7168, 3072, 6),
    (256, 4096, 2048, 6),
    (896, 3584, 3072, 16),
)


def load_aux_module(capability_probe: Any = None) -> Any:
    # The HIP/JIT extension is the system boundary. Keep the Python API real
    # without importing GPU packages or causing a shared module build.
    tree = ast.parse(AUX_MODULE.read_text(), filename=str(AUX_MODULE))
    tree.body = [
        node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]

    def compile_ops(*args: Any, **kwargs: Any) -> Callable[..., Any]:
        def decorate(fn: Any) -> Callable[..., Any]:
            if fn.__name__ == "_mxfp4_moe_sort_internal_is_supported":
                return capability_probe or fn
            return fn

        return decorate

    namespace = {"Tensor": object, "compile_ops": compile_ops}
    exec(  # noqa: S102 - trusted checkout AST with the HIP/JIT boundary injected
        compile(tree, str(AUX_MODULE), "exec"), namespace
    )
    return SimpleNamespace(**namespace)


class TestMxfp4MoeAuxCatalog(unittest.TestCase):
    def test_retune_model_shapes_are_registered(self) -> None:
        aux = load_aux_module()
        for shape in MODEL_SHAPES:
            with self.subTest(shape=shape):
                self.assertTrue(aux.is_mxfp4_moe_shape_supported(*shape))

    def test_generates_missing_instances_and_reuses_existing_keys_once(self) -> None:
        expected_ne24 = {
            "aux_quant_NE24_TOPK6_MB128_H7168",
            "aux_quant_NE24_TOPK6_MB32_H7168",
            "aux_quant_NE24_TOPK6_MB64_H7168",
            "aux_sort3s_NE24_TOPK6_MB128",
            "aux_sort3s_NE24_TOPK6_MB32",
            "aux_sort3s_NE24_TOPK6_MB64",
            "aux_sort_quant_NE24_TOPK6_MB32_H7168",
            "aux_sortonly_NE24_TOPK6_MB16_H7168",
            "aux_sortscales_BM128_NE24_H7168",
            "aux_sortscales_BM32_NE24_H7168",
            "aux_sortscales_BM64_NE24_H7168",
            "aux_sortzi_NE24_TOPK6_MB16_H7168",
        }
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(
                [sys.executable, str(GENERATOR), "--working_path", directory],
                check=True,
                capture_output=True,
                text=True,
            )
            output = Path(directory)
            names = {path.stem for path in (output / "instances").glob("*.cu")}
            self.assertEqual({name for name in names if "NE24_" in name}, expected_ne24)
            lookup = (output / "mxfp4_moe_aux_lookup.h").read_text()
            lookup_names = re.findall(r'\{"([^"]+)", &[^}]+\}', lookup)
            self.assertEqual(len(lookup_names), len(names))
            self.assertEqual(set(lookup_names), names)
            aux = load_aux_module()
            generated_scatter_keys = {
                (int(match[1]), int(match[2]))
                for name in names
                if (match := re.fullmatch(r"aux_scatter_H(\d+)_TOPK(\d+)_NT1", name))
            }
            hidden_axes = {
                1024,
                *(shape[1] for shape in aux.MXFP4_MOE_SUPPORTED_SHAPES),
            }
            topk_axes = {2, *(shape[3] for shape in aux.MXFP4_MOE_SUPPORTED_SHAPES)}
            supported_scatter_keys = {
                (hidden, topk)
                for hidden in hidden_axes
                for topk in topk_axes
                if aux.is_mxfp4_moe_scatter_supported(hidden, topk)
            }
            self.assertEqual(supported_scatter_keys, generated_scatter_keys)
            for expert, hidden, _inter, topk in MODEL_SHAPES:
                for operation in ("sortonly", "sortzi"):
                    key = f"aux_{operation}_NE{expert}_TOPK{topk}_MB16_H{hidden}"
                    self.assertIn(key, names)
                    self.assertEqual(lookup.count(f'{{"{key}", &{key}}}'), 1)

    def test_preflight_reports_missing_compiled_instance_without_fallback(self) -> None:
        def capability_probe(
            expert: int, topk: int, hidden: int, block_m: int, zero_init: bool
        ) -> bool:
            return expert != 24 or not zero_init

        aux = load_aux_module(capability_probe)
        with self.assertRaisesRegex(
            aux._MissingMxfp4MoeAuxInstances,
            "aux_sortzi_NE24_TOPK6_MB16_H7168.*AITER_REBUILD=1",
        ) as raised:
            aux.prepare_mxfp4_moe_aux(MODEL_SHAPES)
        self.assertEqual(
            raised.exception.missing_keys, ("aux_sortzi_NE24_TOPK6_MB16_H7168",)
        )
        self.assertEqual(raised.exception.missing_instances, ((24, 6, 7168, True),))

    def test_preflight_accepts_new_inter_dim_when_compiled_aux_keys_cover_it(
        self,
    ) -> None:
        covered = {(24, 6, 7168, 16, False), (24, 6, 7168, 16, True)}
        aux = load_aux_module(lambda *key: key in covered)
        self.assertIsNone(aux.prepare_mxfp4_moe_aux([(24, 7168, 512, 6)]))

    def test_preflight_skips_extension_preparation_for_no_generated_sort_rows(
        self,
    ) -> None:
        def capability_probe(*key: Any) -> bool:
            raise AssertionError("empty preflight must not load or build the extension")

        aux = load_aux_module(capability_probe)
        self.assertIsNone(aux.prepare_mxfp4_moe_aux([]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
