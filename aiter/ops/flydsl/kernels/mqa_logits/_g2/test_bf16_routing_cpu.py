# SPDX-License-Identifier: MIT
"""STDlib-only checks of real bf16 host routing and builder selection.

Extracts production AST without importing aiter, torch or FlyDSL. Decorators
only capture functions; no kernel body, compilation, device or launch runs.
This proves routing, not GPU correctness, final ISA or performance.
Run: python -B aiter/ops/flydsl/kernels/mqa_logits/_g2/test_bf16_routing_cpu.py -v
"""
from __future__ import annotations

import ast
import copy
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parent.parent / "qsa_paged_mqa_logits_page4_gfx1250.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))
FUNCTIONS = {n.name: n for n in TREE.body if isinstance(n, ast.FunctionDef)}


def _kernel(*, name, known_block_size):
    def decorate(fn):
        return SimpleNamespace(_func=fn, name=name, block=known_block_size)
    return decorate


def _exec(nodes, namespace):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.Module(body=[future] + copy.deepcopy(nodes), type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(SOURCE), "exec"), namespace)


def _namespace(environment=None):
    ns = {"os": SimpleNamespace(environ=dict(environment or {})), "lru_cache": lru_cache,
          "flyc": SimpleNamespace(kernel=_kernel, jit=lambda fn: fn),
          "fx": SimpleNamespace(Stream=lambda value: value),
          "_ld_i32": lambda *args: None, "_ld_uniform_i32": lambda *args: None}
    names = {"PAGE_SIZE", "NUM_HEADS", "HEAD_DIM", "_ACCESS", "_BF16_K_LOAD"}
    nodes = []
    for node in TREE.body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if targets and all(n in names or n.startswith(("WMMA_", "TDM_")) for n in targets):
                nodes.append(node)
        elif isinstance(node, ast.If):
            if isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name):
                if node.test.left.id in ("_ACCESS", "_BF16_K_LOAD"):
                    nodes.append(node)
    nodes += [FUNCTIONS[n] for n in ("_resolve_bf16_k_load", "wmma_rows_per_block",
                                    "wmma_grid", "_build_wmma_kernel")]
    _exec(nodes, ns)
    return ns


def _selected(launch):
    closure = dict(zip(launch.__code__.co_freevars,
                       (c.cell_contents for c in launch.__closure__ or ())))
    return closure["selected_kernel"]


def _wrapper_route(ns, rows, requests, pointer, columns=65):
    """Execute the original shape/resolver/alignment/grid statements verbatim."""
    wrapper = FUNCTIONS["qsa_paged_mqa_logits_page4"]
    branch = next(n for n in wrapper.body if isinstance(n, ast.If)
                  and ast.unparse(n.test) == "not is_fp8 and _BF16_PATH == 'wmma'")
    end = next(i for i, n in enumerate(branch.body) if isinstance(n, ast.Expr)
               and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "_run")
    local = dict(ns, rows=rows, num_requests=requests, columns=columns,
                 kv_u8=SimpleNamespace(data_ptr=lambda: pointer))
    _exec(branch.body[:end], local)
    return (local["rows_per_block"], local["k_load"], local["grid_x"], local["grid_y"])


class Bf16RoutingCPU(unittest.TestCase):
    def assert_selection(self, ns, geometry, mode, tdm):
        launch = ns["_build_wmma_kernel"](1.0, geometry, mode)
        kernel = _selected(launch)
        expected = ("kernel_tdm_r16" if tdm else "kernel") if geometry == 16 else "kernel_r32"
        self.assertEqual(kernel._func.__name__, expected)
        self.assertEqual(kernel.block, [128, 1, 1])
        self.assertEqual(launch.dtype_tag, "bf16_wmma")
        name = "qsa_paged_mqa_logits_page4_gfx1250_bf16_wmma_D128_H4_P4_r%d_c64_flydsl" % geometry
        self.assertEqual(launch.kernel_name, name + ("_tdm" if tdm else ""))
        self.assertEqual(kernel.name, launch.kernel_name)

    def test_unset_environment_defaults_auto(self):
        ns = _namespace()
        self.assertEqual(ns["_BF16_K_LOAD"], "auto")
        for geometry, resolved in ((16, "raw"), (32, "tdm")):
            self.assertEqual(ns["_resolve_bf16_k_load"](geometry), resolved)
            self.assert_selection(ns, geometry, None, resolved == "tdm")

    def test_modes_and_explicit_overrides_select_actual_functions(self):
        for environment in ("auto", "raw", "tdm"):
            ns = _namespace({"QSA_BF16_K_LOAD": environment})
            for explicit in (None, "auto", "raw", "tdm"):
                mode = environment if explicit is None else explicit
                for geometry in (16, 32):
                    tdm = mode == "tdm" or (mode == "auto" and geometry == 32)
                    with self.subTest(environment=environment, explicit=explicit, geometry=geometry):
                        self.assert_selection(ns, geometry, explicit, tdm)

    def test_environment_normalization_and_validation(self):
        self.assertEqual(_namespace({"QSA_BF16_K_LOAD": " AuTo "})["_BF16_K_LOAD"], "auto")
        for mode in ("", "invalid", "scalar"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                _namespace({"QSA_BF16_K_LOAD": mode})
        ns = _namespace()
        for mode in ("", "invalid", "AUTO"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                ns["_resolve_bf16_k_load"](32, mode)
        for geometry in (0, 4, 31, 64):
            with self.subTest(geometry=geometry), self.assertRaises(ValueError):
                ns["_build_wmma_kernel"](1.0, geometry)

    def test_copy_atom_access_keeps_every_mode_raw(self):
        for mode in ("auto", "raw", "tdm"):
            ns = _namespace({"QSA_G2_ACCESS": "cop", "QSA_BF16_K_LOAD": mode})
            for geometry in (16, 32):
                with self.subTest(mode=mode, geometry=geometry):
                    self.assert_selection(ns, geometry, None, False)

    def test_prefill_decode_draft_shape_routing(self):
        ns = _namespace()
        for rows, requests, expected in ((2880, 2, 32), (512, 1, 32), (64, 16, 16),
                                         (16, 16, 16), (31, 1, 16), (32, 1, 32),
                                         (63, 2, 16), (64, 2, 32)):
            with self.subTest(rows=rows, requests=requests):
                self.assertEqual(ns["wmma_rows_per_block"](rows, requests), expected)
                geometry, mode, gx, gy = _wrapper_route(ns, rows, requests, 0x1000)
                self.assertEqual((geometry, mode), (expected, "tdm" if expected == 32 else "raw"))
                self.assertEqual((gx, gy), (2, (rows + expected - 1) // expected))

    def test_wrapper_resolves_before_alignment_and_keeps_actual_raw_fallback(self):
        for mode in ("auto", "raw", "tdm"):
            ns = _namespace({"QSA_BF16_K_LOAD": mode})
            for rows, requests in ((512, 1), (64, 16), (16, 16)):
                original = ns["wmma_rows_per_block"](rows, requests)
                for pointer in (0x1000, 0x1010, 0x1070):
                    resolved = "tdm" if mode == "tdm" or (mode == "auto" and original == 32) else "raw"
                    fallback = resolved == "tdm" and pointer % 128 != 0
                    geometry, actual, gx, gy = _wrapper_route(ns, rows, requests, pointer)
                    with self.subTest(mode=mode, rows=rows, pointer=pointer):
                        self.assertEqual((geometry, actual), (16, "raw") if fallback else (original, resolved))
                        self.assertEqual((gx, gy), (2, (rows + geometry - 1) // geometry))
                        self.assert_selection(ns, geometry, actual, actual == "tdm")

    def test_bf16_wrapper_passes_resolved_mode_to_builder(self):
        wrapper = FUNCTIONS["qsa_paged_mqa_logits_page4"]
        branch = next(n for n in wrapper.body if isinstance(n, ast.If)
                      and ast.unparse(n.test) == "not is_fp8 and _BF16_PATH == 'wmma'")
        builds = [n for n in ast.walk(branch) if isinstance(n, ast.Call)
                  and ast.unparse(n.func) == "_build_wmma_kernel"]
        self.assertEqual(len(builds), 1)
        self.assertEqual([ast.unparse(n) for n in builds[0].args],
                         ["float(divisor)", "rows_per_block", "k_load"])
        fp8 = next(n for n in wrapper.body if isinstance(n, ast.If)
                   and ast.unparse(n.test) == "is_fp8 and _FP8_PATH == 'wmma'")
        self.assertFalse(any(isinstance(n, ast.Name) and n.id in
                             ("_resolve_bf16_k_load", "_BF16_K_LOAD") for n in ast.walk(fp8)))


if __name__ == "__main__":
    unittest.main()
