# SPDX-License-Identifier: MIT
"""STDlib-only checks of real bf16 host routing and builder selection.

Extracts production AST without importing aiter, torch or FlyDSL. Decorators
only capture functions; no kernel body, compilation, device or launch runs.
This proves routing, not GPU correctness, final ISA or performance.
Run: python -B aiter/ops/flydsl/kernels/mqa_logits/_g2/test_bf16_routing_cpu.py -v
"""
from __future__ import annotations

import sys as _n64_sys, pathlib as _n64_path  # ROUND-NATIVE64 seam support
_n64_sys.path.insert(0, str(_n64_path.Path(__file__).resolve().parent))
from native64_seam import ensure_i32  # noqa: E402

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
    ensure_i32(namespace)
    exec(compile(module, str(SOURCE), "exec"), namespace)


def _namespace(environment=None):
    ns = {"os": SimpleNamespace(environ=dict(environment or {})), "lru_cache": lru_cache,
          "flyc": SimpleNamespace(kernel=_kernel, jit=lambda fn: fn),
          "fx": SimpleNamespace(Stream=lambda value: value),
          "_ld_i32": lambda *args: None, "_ld_uniform_i32": lambda *args: None}
    names = {"PAGE_SIZE", "NUM_HEADS", "HEAD_DIM", "_ACCESS", "_BF16_K_LOAD",
             "_BF16_R16_TDM_COLS", "R16_TDM_COLS", "_BF16_R64", "_FP8_K_LOAD"}
    nodes = []
    for node in TREE.body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if targets and all(n in names or n.startswith(("WMMA_", "TDM_")) for n in targets):
                nodes.append(node)
        elif isinstance(node, ast.If):
            # R64 validation checks the environment string before/after bool conversion.
            if isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name):
                if node.test.left.id in ("_ACCESS", "_BF16_K_LOAD", "_BF16_R16_TDM_COLS", "_BF16_R64", "_FP8_K_LOAD"):
                    nodes.append(node)
            elif any(isinstance(n, ast.Constant) and n.value in
                     ("QSA_BF16_R64", "QSA_BF16_PERSIST", "QSA_PERSIST_COLS")
                     for n in ast.walk(node.test)):
                nodes.append(node)
    nodes += [FUNCTIONS[n] for n in ("_resolve_bf16_k_load", "_resolve_fp8_k_load", "wmma_rows_per_block",
                                    "wmma_bf16_rows_per_block", "wmma_grid", "_build_wmma_kernel")]
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
    local = dict(ns, rows=rows, num_requests=requests, columns=columns, divisor=1.0,
                 kv_u8=SimpleNamespace(data_ptr=lambda: pointer))
    _exec(branch.body[:end], local)
    # The launched builder object is exactly the one whose width sized grid.x.
    assert ast.unparse(branch.body[end].value.args[0]) == "launch"
    return (local["rows_per_block"], local["k_load"], local["grid_x"], local["grid_y"])


class Bf16RoutingCPU(unittest.TestCase):
    def assert_selection(self, ns, geometry, mode, tdm, r16_tdm_cols=32):
        launch = ns["_build_wmma_kernel"](1.0, geometry, mode)
        kernel = _selected(launch)
        # ROUND-MERGE1: QSA_R16_NOBAR defaults to 1 -> TDM r16 selects the barrier-free kernel.
        nobar = bool(ns["TDM_R16_NOBAR"]) and tdm and geometry == 16
        expected = ("kernel_tdm_r16" if tdm else "kernel") if geometry == 16 else ("kernel_r64" if geometry == 64 else "kernel_r32")
        if nobar:
            expected = "kernel_tdm_r16_nobar"
        self.assertEqual(kernel._func.__name__, expected)
        self.assertEqual(kernel.block, [128, 1, 1])
        self.assertEqual(launch.dtype_tag, "bf16_wmma")
        cols = r16_tdm_cols if (tdm and geometry == 16) else 64
        self.assertEqual(launch.cols_per_block, cols)
        name = "qsa_paged_mqa_logits_page4_gfx1250_bf16_wmma_D128_H4_P4_r%d_c%d_flydsl" % (geometry, cols)
        self.assertEqual(launch.kernel_name, name + ("_tdm" if tdm else "") + ("_nobar" if nobar else ""))
        self.assertEqual(kernel.name, launch.kernel_name)

    def test_unset_environment_defaults_auto(self):
        ns = _namespace()
        self.assertEqual(ns["_BF16_K_LOAD"], "auto")
        self.assertIs(ns["_BF16_R64"], False)
        self.assertIs(ns["TDM_PERSIST"], True)
        self.assertEqual(ns["TDM_PERSIST_CHUNKS"], 8)
        for geometry, resolved in ((16, "tdm"), (32, "tdm"), (64, "tdm")):
            self.assertEqual(ns["_resolve_bf16_k_load"](geometry), resolved)
            self.assert_selection(ns, geometry, None, resolved == "tdm")

    def test_r16_nobar_default_on_and_opt_out(self):
        # ROUND-MERGE1: default 1 (barrier-free r16 TDM); "0" restores the champion kernel.
        self.assertIs(_namespace()["TDM_R16_NOBAR"], True)
        self.assertIs(_namespace({"QSA_R16_NOBAR": "1"})["TDM_R16_NOBAR"], True)
        off = _namespace({"QSA_R16_NOBAR": "0"})
        self.assertIs(off["TDM_R16_NOBAR"], False)
        self.assertEqual(_selected(off["_build_wmma_kernel"](1.0, 16, "tdm"))._func.__name__, "kernel_tdm_r16")
        on = _namespace()
        self.assertEqual(_selected(on["_build_wmma_kernel"](1.0, 16, "tdm"))._func.__name__, "kernel_tdm_r16_nobar")
        # never for raw r16, r32, r64 or persistent builds
        self.assertEqual(_selected(on["_build_wmma_kernel"](1.0, 16, "raw"))._func.__name__, "kernel")
        self.assertEqual(_selected(on["_build_wmma_kernel"](1.0, 32, "tdm"))._func.__name__, "kernel_r32")
        for mode in ("auto", "raw", "tdm"):
            ns = _namespace({"QSA_BF16_K_LOAD": mode, "QSA_R16_NOBAR": "0"})
            for geometry in (16, 32, 64):
                with self.subTest(mode=mode, geometry=geometry):
                    self.assert_selection(ns, geometry, None, mode != "raw")

    def test_modes_and_explicit_overrides_select_actual_functions(self):
        for environment in ("auto", "raw", "tdm"):
            ns = _namespace({"QSA_BF16_K_LOAD": environment})
            for explicit in (None, "auto", "raw", "tdm"):
                mode = environment if explicit is None else explicit
                for geometry in (16, 32, 64):
                    tdm = mode != "raw"
                    with self.subTest(environment=environment, explicit=explicit, geometry=geometry):
                        self.assert_selection(ns, geometry, explicit, tdm)

    def test_fp8_inherits_shared_k_load_with_explicit_override_and_raw_r16(self):
        self.assertEqual(_namespace()["_FP8_K_LOAD"], "auto")
        for shared in ("auto", "raw", "tdm"):
            for override in (None, "auto", "raw", "tdm", " TdM "):
                environment = {"QSA_BF16_K_LOAD": shared}
                if override is not None:
                    environment["QSA_FP8_K_LOAD"] = override
                ns = _namespace(environment)
                expected = shared if override is None else override.strip().lower()
                with self.subTest(shared=shared, override=override):
                    self.assertEqual(ns["_FP8_K_LOAD"], expected)
                    self.assertEqual(ns["_resolve_fp8_k_load"](32),
                                     "raw" if expected == "raw" else "tdm")
                    for explicit in (None, "auto", "raw", "tdm"):
                        self.assertEqual(ns["_resolve_fp8_k_load"](16, explicit), "raw")
        for invalid in ("", "invalid", "scalar"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _namespace({"QSA_FP8_K_LOAD": invalid})

    def test_environment_normalization_and_validation(self):
        self.assertEqual(_namespace({"QSA_BF16_K_LOAD": " AuTo "})["_BF16_K_LOAD"], "auto")
        for mode in ("", "invalid", "scalar"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                _namespace({"QSA_BF16_K_LOAD": mode})
        ns = _namespace()
        for mode in ("", "invalid", "AUTO"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                ns["_resolve_bf16_k_load"](32, mode)
        for geometry in (0, 4, 31, 63, 65):
            with self.subTest(geometry=geometry), self.assertRaises(ValueError):
                ns["_build_wmma_kernel"](1.0, geometry)

    def test_r64_environment_validation_and_opt_in_routing(self):
        for value, enabled in (("0", False), ("1", True), (" 0 ", False), (" 1 ", True)):
            for mode in ("auto", "raw", "tdm"):
                for access in ("raw", "cop"):
                    ns = _namespace({"QSA_BF16_R64": value, "QSA_BF16_K_LOAD": mode,
                                     "QSA_G2_ACCESS": access})
                    self.assertIs(ns["_BF16_R64"], enabled)
                    for rows, requests in ((63, 1), (64, 1), (127, 2), (128, 2), (512, 1)):
                        champion = ns["wmma_rows_per_block"](rows, requests)
                        expected = (64 if enabled and mode != "raw" and access == "raw"
                                    and rows >= 64 * requests else champion)
                        with self.subTest(value=value, mode=mode, access=access,
                                          rows=rows, requests=requests):
                            self.assertEqual(ns["wmma_bf16_rows_per_block"](rows, requests), expected)
                            # Private builder accepts explicit r64 regardless of wrapper opt-in.
                            self.assert_selection(ns, 64, None, mode != "raw" and access == "raw")
        for value in ("", "2", "true", "01", "-1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _namespace({"QSA_BF16_R64": value})

    def test_copy_atom_access_keeps_every_mode_raw(self):
        for mode in ("auto", "raw", "tdm"):
            ns = _namespace({"QSA_G2_ACCESS": "cop", "QSA_BF16_K_LOAD": mode})
            for geometry in (16, 32, 64):
                with self.subTest(mode=mode, geometry=geometry):
                    self.assert_selection(ns, geometry, None, False)

    def test_prefill_decode_draft_shape_routing(self):
        ns = _namespace()
        for rows, requests, expected in ((2880, 2, 32), (512, 1, 32), (64, 16, 16),
                                         (16, 16, 16), (31, 1, 16), (32, 1, 32),
                                         (63, 2, 16), (64, 2, 32)):
            with self.subTest(rows=rows, requests=requests):
                self.assertEqual(ns["wmma_rows_per_block"](rows, requests), expected)
                self.assertEqual(ns["wmma_bf16_rows_per_block"](rows, requests), expected)
                geometry, mode, gx, gy = _wrapper_route(ns, rows, requests, 0x1000)
                self.assertEqual((geometry, mode), (expected, "tdm"))
                self.assertEqual((gx, gy), (3 if expected == 16 else 2,
                                            (rows + expected - 1) // expected))

    def test_wrapper_resolves_before_alignment_and_keeps_actual_raw_fallback(self):
        for mode in ("auto", "raw", "tdm"):
            ns = _namespace({"QSA_BF16_K_LOAD": mode})
            for rows, requests in ((512, 1), (64, 16), (16, 16)):
                original = ns["wmma_bf16_rows_per_block"](rows, requests)
                for pointer in (0x1000, 0x1010, 0x1070):
                    resolved = "raw" if mode == "raw" else "tdm"
                    fallback = resolved == "tdm" and pointer % 128 != 0
                    geometry, actual, gx, gy = _wrapper_route(ns, rows, requests, pointer)
                    with self.subTest(mode=mode, rows=rows, pointer=pointer):
                        self.assertEqual((geometry, actual), (16, "raw") if fallback else (original, resolved))
                        cols = 32 if (geometry == 16 and actual == "tdm") else 64
                        self.assertEqual((gx, gy), ((65 + cols - 1) // cols,
                                                    (rows + geometry - 1) // geometry))
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

    # ---- TDM r16 column width: auto and forced TDM; raw untouched ----------
    def test_r16_tdm_cols_switch_changes_auto_and_forced_tdm_r16(self):
        for cols in ("32", "64", " 64 "):
            for mode in ("auto", "raw", "tdm"):
                ns = _namespace({"QSA_BF16_K_LOAD": mode, "QSA_BF16_R16_TDM_COLS": cols})
                self.assertEqual(ns["R16_TDM_COLS"], int(cols))
                for geometry in (16, 32):
                    tdm = mode != "raw"
                    with self.subTest(cols=cols, mode=mode, geometry=geometry):
                        self.assert_selection(ns, geometry, None, tdm, int(cols))
                # Private single-tile builders keep their widths; the wrapper
                # independently selects persistent r32 for eligible prefill.
                for rows, requests in ((64, 16), (16, 16), (512, 1), (2880, 2)):
                    geometry, actual, gx, gy = _wrapper_route(ns, rows, requests, 0x1000,
                                                              columns=65536)
                    width = (512 if actual == "persist" else
                             int(cols) if (geometry == 16 and actual == "tdm") else 64)
                    with self.subTest(cols=cols, mode=mode, rows=rows):
                        self.assertEqual(gx, 65536 // width)
                        self.assertEqual(gy, (rows + geometry - 1) // geometry)

    def test_r16_tdm_cols_default_and_validation(self):
        self.assertEqual(_namespace()["R16_TDM_COLS"], 32)
        for cols in ("", "16", "48", "128", "x"):
            with self.subTest(cols=cols), self.assertRaises(ValueError):
                _namespace({"QSA_BF16_R16_TDM_COLS": cols})
        ns = _namespace()
        for args in ((16, 65, 16, 48), (16, 65, 32, 32), (16, 65, 16, 128)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                ns["wmma_grid"](*args)
        for columns in (1, 31, 32, 33, 63, 64, 65, 1000, 65536):
            self.assertEqual(ns["wmma_grid"](16, columns, 16, 32), ((columns + 31) // 32, 1))
            self.assertEqual(ns["wmma_grid"](16, columns), ((columns + 63) // 64, 1))

    def test_unaligned_cache_with_forced_tdm_r16_c32_falls_back_to_raw_c64(self):
        ns = _namespace({"QSA_BF16_K_LOAD": "tdm"})
        for pointer in (0x1010, 0x1040, 0x1070):
            self.assertEqual(_wrapper_route(ns, 64, 16, pointer), (16, "raw", 2, 4))
        self.assertEqual(_wrapper_route(ns, 64, 16, 0x1080), (16, "tdm", 3, 4))


if __name__ == "__main__":
    unittest.main()
