# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Marlowe AI
"""CPU tests of the actual emitter's grouping/address logic without ROCm imports."""

import ast
import random
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Predicate:
    def __init__(self, value):
        self.value = bool(value)

    def __bool__(self):
        return self.value

    def select(self, yes, no):
        return yes if self.value else no

    def __or__(self, other):
        return Predicate(self.value or bool(other))


class Scalar(int):
    def __eq__(self, other):
        return Predicate(int(self) == int(other))

    def __lt__(self, other):
        return Predicate(int(self) < int(other))

    def __add__(self, other):
        return Scalar(int(self) + int(other))

    def __sub__(self, other):
        return Scalar(int(self) - int(other))

    def __mul__(self, other):
        return Scalar(int(self) * int(other))

    def __floordiv__(self, other):
        return Scalar(int(self) // int(other))

    def __mod__(self, other):
        return Scalar(int(self) % int(other))

    def __and__(self, other):
        return Scalar(int(self) & int(other))

    def __or__(self, other):
        return Scalar(int(self) | int(other))

    def __lshift__(self, other):
        return Scalar(int(self) << int(other))


class Vector:
    def __init__(self, values):
        self.values = list(values)

    def __add__(self, other):
        right = other.values if isinstance(other, Vector) else [other] * 64
        return Vector([a + b for a, b in zip(self.values, right)])

    def __floordiv__(self, other):
        return Vector([a // int(other) for a in self.values])

    def __eq__(self, other):
        return Vector([a == int(other) for a in self.values])


def cttz(value):
    value = int(value) & ((1 << 64) - 1)
    if not value:
        raise AssertionError("emitter must not rely on cttz(0) semantics")
    return Scalar((value & -value).bit_length() - 1)


def load_functions(path, environment):
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    exec(compile(tree, str(path), "exec"), environment)
    return environment


def emitters():
    def gather(ids, index):
        if isinstance(index, Vector):
            return Vector([ids[i] for i in index.values])
        return Scalar(ids[int(index)])

    env = {
        "fx": types.SimpleNamespace(
            Int32=Scalar,
            Int64=Scalar,
            min=min,
            cttz=cttz,
            ctpop=lambda v: Scalar((int(v) & ((1 << 64) - 1)).bit_count()),
        ),
        "rocdl": types.SimpleNamespace(
            readfirstlane=lambda _t, v: v,
            ballot=lambda _t, p: Scalar(
                sum(1 << i for i, v in enumerate(p.values) if v)
            ),
        ),
        "T": types.SimpleNamespace(i32=None, i64=None),
        "as_ir_value": lambda v: v,
        "range_constexpr": range,
        "_global_i32_at": gather,
    }
    return load_functions(ROOT / "aiter/ops/flydsl/kernels/mxmoe_routes8.py", env)


class RouteMergeTests(unittest.TestCase):
    def check_distribution(self, rows):
        self.assertEqual(len(rows), 8)
        for row in rows:
            self.assertEqual(len(set(row)), 8, "supplied native top8 contract")
        ids = [v for row in rows for v in [*row, 256]]
        compact = [v for row in rows for v in row]
        functions = emitters()
        observed = []
        leaders = []
        for slot in range(65):
            expert, mask, active = functions["expert_group"](
                ids, Scalar(slot), Vector(range(64))
            )
            expert, mask = int(expert), int(mask)
            expected = [i for i, v in enumerate(compact) if v == expert]
            expected_active = slot == 64 or slot == expected[0]
            self.assertEqual(bool(active), expected_active)
            if not active:
                continue
            leaders.append(expert)
            for row in range(16):
                packed, index, valid = functions["route_row"](
                    Scalar(mask), Scalar(slot), Scalar(row)
                )
                packed, index = int(packed), int(index)
                if slot == 64:
                    expected_pair = (row, 8) if row < 8 else None
                elif row < len(expected):
                    expected_pair = divmod(expected[row], 8)
                else:
                    expected_pair = None
                self.assertEqual(bool(valid), expected_pair is not None)
                if expected_pair is None:
                    self.assertEqual(packed, (9 << 24) | 8)
                    self.assertEqual(index, 0)
                else:
                    token, choice = expected_pair
                    self.assertEqual(packed, token | (choice << 24))
                    self.assertEqual(index, token * 9 + choice)
                    self.assertEqual(ids[index], expert)
                    observed.append((token, choice))
        self.assertEqual(len(leaders), len(set(compact)) + 1)
        self.assertCountEqual(observed, [(t, k) for t in range(8) for k in range(9)])

    def test_all_distinct_and_last_lane(self):
        self.check_distribution([[t * 8 + k for k in range(8)] for t in range(8)])

    def test_identical_expert_sets_and_permuted_slots(self):
        self.check_distribution([[(k + t) % 8 for k in range(8)] for t in range(8)])

    def test_one_expert_in_every_token(self):
        self.check_distribution([[255, *range(t * 7, t * 7 + 7)] for t in range(8)])

    def test_seeded_native_routes(self):
        rng = random.Random(35508)
        for _ in range(120):
            self.check_distribution([rng.sample(range(256), 8) for _ in range(8)])

    def test_zero_ballot_shared_and_sparse_masks(self):
        resolve = emitters()["route_row"]
        masks = [0, 1, 1 << 63, (1 << 63) | 1, sum(1 << (8 * k) for k in range(8))]
        for mask in masks:
            selected = [i for i in range(64) if mask & (1 << i)]
            for row in range(16):
                packed, _, valid = resolve(Scalar(mask), Scalar(0), Scalar(row))
                self.assertEqual(bool(valid), row < len(selected))
                if valid:
                    token, choice = divmod(selected[row], 8)
                    self.assertEqual(int(packed), token | (choice << 24))

    def test_only_exact_m8_geometry(self):
        namespace = load_functions(ROOT / "aiter/ops/flydsl/mxmoe_tiny_m8.py", {})
        support = namespace["supported_geometry"]
        args = [8, "gfx950", (257, 512, 3072), (257, 6144, 128), True, True]
        self.assertTrue(support(*args))
        for rows in [1, 4, 7, 9, 16, 32, 64, 128, 256]:
            self.assertFalse(support(rows, *args[1:]))
        for index, value in [
            (1, "gfx942"),
            (2, (256, 512, 3072)),
            (4, False),
            (5, False),
        ]:
            changed = list(args)
            changed[index] = value
            self.assertFalse(support(*changed))

    def test_factory_uses_an_existing_native_g1_variant(self):
        factory = ast.parse((ROOT / "aiter/ops/flydsl/mxmoe_tiny_m8.py").read_text())
        call = next(
            node
            for node in ast.walk(factory)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "compile_gemm1_a4w4_port"
        )
        settings = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in call.keywords
            if keyword.arg in ("BM", "use_nt", "inline_quant")
        }
        table = ast.parse((ROOT / "aiter/ops/flydsl/mxfp4_kname.py").read_text())
        variants = next(
            ast.literal_eval(node.value)
            for node in table.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "MXFP4_G1_VARIANTS"
                for target in node.targets
            )
        )
        self.assertIn(
            tuple(settings[key] for key in ("BM", "use_nt", "inline_quant")),
            variants["fp4"],
        )

    def test_integrated_reset_covers_all_output_exactly_once(self):
        helper = emitters()
        written = []
        for block in range(65 * 2):
            for thread in range(256):
                index, valid = helper["reset_index"](Scalar(block), Scalar(thread))
                if bool(valid):
                    written.append(int(index))
        self.assertEqual(written, list(range(8 * 6144 // 2)))
        self.assertEqual(len(written), len(set(written)))
        factory = ast.parse((ROOT / "aiter/ops/flydsl/mxmoe_tiny_m8.py").read_text())
        self.assertFalse(
            any(
                isinstance(node, ast.Attribute) and node.attr == "zero_"
                for node in ast.walk(factory)
            )
        )
        compiler = ast.parse(
            (ROOT / "aiter/ops/flydsl/kernels/mxfp4_gemm1.py").read_text()
        )
        kernel = next(
            node
            for node in ast.walk(compiler)
            if isinstance(node, ast.FunctionDef) and node.name == "gemm1_kernel"
        )
        reset = next(
            node
            for node in kernel.body
            if any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "reset_index"
                for n in ast.walk(node)
            )
        )
        # Reset is a direct kernel-body statement before expert leader gating.
        leaders = [
            node
            for node in kernel.body
            if isinstance(node, ast.If)
            and any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "expert_group"
                for n in ast.walk(node)
            )
        ]
        self.assertEqual(len(leaders), 1)
        self.assertLess(kernel.body.index(reset), kernel.body.index(leaders[0]))

    def test_factory_preserves_native_unclamped_silu(self):
        factory = ast.parse((ROOT / "aiter/ops/flydsl/mxmoe_tiny_m8.py").read_text())
        call = next(
            node
            for node in ast.walk(factory)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "compile_gemm1_a4w4_port"
        )
        bound = next(
            keyword.value for keyword in call.keywords if keyword.arg == "swiglu_limit"
        )
        actual = eval(
            compile(ast.Expression(bound), "factory-limit", "eval"), {"float": float}
        )
        native = ast.parse((ROOT / "aiter/ops/flydsl/moe_kernels.py").read_text())
        normalize = next(
            node
            for node in native.body
            if isinstance(node, ast.FunctionDef) and node.name == "runtime_swiglu_limit"
        )
        namespace = {}
        exec(compile(ast.Module([normalize], []), "native-limit", "exec"), namespace)
        expected = namespace["runtime_swiglu_limit"](None, "silu")
        self.assertEqual(actual, expected)
        self.assertEqual(expected, float("inf"))
        self.assertNotEqual(expected, 7.0)

    def test_input_aliases_are_rejected_before_launch(self):
        namespace = load_functions(ROOT / "aiter/ops/flydsl/mxmoe_tiny_m8.py", {})
        check = namespace["_check_disjoint"]

        def tensor(start, size):
            return types.SimpleNamespace(
                data_ptr=lambda: start, numel=lambda: size, element_size=lambda: 1
            )

        scratch = [tensor(100, 100), tensor(300, 40), tensor(500, 60)]
        check([tensor(0, 100), tensor(200, 100), tensor(340, 160)], scratch)
        for start, size in [(100, 100), (99, 2), (199, 2), (90, 600), (310, 4)]:
            with self.assertRaisesRegex(ValueError, "must not alias"):
                check([tensor(start, size)], scratch)
        # The actual run rejects aliases before either G1 or G2 is launched.
        factory = ast.parse((ROOT / "aiter/ops/flydsl/mxmoe_tiny_m8.py").read_text())
        run = next(
            node
            for node in ast.walk(factory)
            if isinstance(node, ast.FunctionDef) and node.name == "run"
        )
        guard = next(
            index
            for index, node in enumerate(run.body)
            if any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "_check_disjoint"
                for call in ast.walk(node)
            )
        )
        launch = next(
            index
            for index, node in enumerate(run.body)
            if any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "_run_compiled"
                for call in ast.walk(node)
            )
        )
        self.assertLess(guard, launch)

    def test_default_off_compiler_and_atomic_math(self):
        for name in ["mxfp4_gemm1.py", "mxfp4_gemm2.py"]:
            source = (ROOT / "aiter/ops/flydsl/kernels" / name).read_text()
            tree = ast.parse(source)
            compiler = next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name.startswith("compile_gemm")
            )
            defaults = dict(
                zip(
                    (a.arg for a in compiler.args.kwonlyargs), compiler.args.kw_defaults
                )
            )
            self.assertIs(defaults["merge_routes8"].value, False)
            self.assertIn('"_merge8', source)
        body = (ROOT / "aiter/ops/flydsl/kernels/mxfp4_gemm2.py").read_text()
        self.assertIn("[v2[0] * weight[mr], v2[1] * weight[mr]], fx.Float32", body)
        self.assertIn("llvm.AtomicBinOp.fadd", body)
        self.assertIn('syncscope="agent"', body)


if __name__ == "__main__":
    unittest.main()
