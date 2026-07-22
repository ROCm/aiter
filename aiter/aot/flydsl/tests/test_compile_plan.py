# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Focused CPU tests for callable-bound FlyDSL compile plans."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import importlib.util
import os
from pathlib import Path
import sys
import unittest

_MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "ops" / "flydsl" / "compile_plan.py"
)
_MODULE_NAME = "_aiter_compile_plan_test"
_SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load {_MODULE_PATH}")
compile_plan = importlib.util.module_from_spec(_SPEC)
sys.modules[_MODULE_NAME] = compile_plan
_SPEC.loader.exec_module(compile_plan)

ArgumentKind = compile_plan.ArgumentKind
BoundCall = compile_plan.BoundCall
CompileOpRegistry = compile_plan.CompileOpRegistry
CompilePlan = compile_plan.CompilePlan
CompileSpec = compile_plan.CompileSpec
CompileUnit = compile_plan.CompileUnit
KernelSignature = compile_plan.KernelSignature
RocmTarget = compile_plan.RocmTarget
SignatureArg = compile_plan.SignatureArg

_OP_ID = "aiter.flydsl.test.builder.v1"
_OTHER_OP_ID = "aiter.flydsl.test.other.v1"


def _target() -> RocmTarget:
    return RocmTarget("gfx950", 256)


def _abi() -> KernelSignature:
    return KernelSignature(
        (
            SignatureArg(
                "input",
                ArgumentKind.TENSOR,
                "bf16",
                (None, 16),
                (None, 1),
            ),
            SignatureArg("output", ArgumentKind.POINTER, "u8"),
            SignatureArg("rows", ArgumentKind.SCALAR, "i32"),
            SignatureArg("stream", ArgumentKind.STREAM),
        )
    )


class TestCallableBinding(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = CompileOpRegistry()
        self.calls = []

        @self.registry.register(_OP_ID)
        def builder(
            model_dim: int,
            *,
            tile_m: int = 32,
            enabled: bool = False,
        ):
            call = (
                model_dim,
                tile_m,
                enabled,
                os.environ.get("FLYDSL_GPU_ARCH"),
                os.environ.get("CU_NUM"),
            )
            self.calls.append(call)
            return call

        self.builder = builder

    def test_make_unit_binds_defaults_in_callable_order_without_invoking(self) -> None:
        before = (
            os.environ.get("FLYDSL_GPU_ARCH"),
            os.environ.get("CU_NUM"),
        )
        unit = self.registry.make_unit(
            _OP_ID,
            target=_target(),
            signature=_abi(),
            enabled=True,
            model_dim=7168,
        )

        self.assertEqual(
            unit.spec.call.arguments,
            (("model_dim", 7168), ("tile_m", 32), ("enabled", True)),
        )
        self.assertEqual(self.calls, [])
        self.assertEqual(
            before,
            (
                os.environ.get("FLYDSL_GPU_ARCH"),
                os.environ.get("CU_NUM"),
            ),
        )

    def test_binding_rejects_missing_unknown_and_mutable_arguments(self) -> None:
        invalid = (
            ({}, "missing"),
            ({"model_dim": 1, "unknown": 2}, "unexpected"),
            ({"model_dim": []}, "hashable"),
        )
        for kwargs, message in invalid:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, message):
                    self.registry.make_unit(
                        _OP_ID,
                        target=_target(),
                        signature=_abi(),
                        **kwargs,
                    )

    def test_compile_invokes_registered_callable_and_restores_target(self) -> None:
        unit = self.registry.make_unit(
            _OP_ID,
            target=_target(),
            signature=_abi(),
            model_dim=7168,
        )
        before = {
            "FLYDSL_GPU_ARCH": os.environ.get("FLYDSL_GPU_ARCH"),
            "CU_NUM": os.environ.get("CU_NUM"),
        }

        result = self.registry.compile(unit)

        self.assertEqual(result, (7168, 32, False, "gfx950", "256"))
        self.assertEqual(self.calls, [result])
        self.assertEqual(
            {name: os.environ.get(name) for name in before},
            before,
        )

    def test_compile_revalidates_manually_constructed_bound_calls(self) -> None:
        malformed = CompileUnit(
            CompileSpec(
                _OP_ID,
                _target(),
                BoundCall((("model_dim", 7168),)),
            ),
            _abi(),
        )
        with self.assertRaisesRegex(ValueError, "registered signature"):
            self.registry.compile(malformed)
        self.assertEqual(self.calls, [])

    def test_plan_preserves_unit_and_result_order(self) -> None:
        units = tuple(
            self.registry.make_unit(
                _OP_ID,
                target=_target(),
                signature=_abi(),
                model_dim=model_dim,
            )
            for model_dim in (2048, 7168, 2048)
        )
        plan = CompilePlan(units)

        results = self.registry.compile_plan(plan)

        self.assertEqual([result[0] for result in results], [2048, 7168, 2048])
        self.assertEqual(plan.units, units)


class TestDeclarationsAndRegistry(unittest.TestCase):
    def test_declarations_are_immutable_and_hashable(self) -> None:
        call = BoundCall((("model_dim", 7168), ("tiles", (32, 128))))
        spec = CompileSpec(_OP_ID, _target(), call)
        argument = SignatureArg("rows", ArgumentKind.SCALAR, "i32")
        signature = KernelSignature((argument,))
        unit = CompileUnit(spec, signature)
        plan = CompilePlan((unit,))

        for value in (_target(), call, spec, argument, signature, unit, plan):
            hash(value)
            with self.assertRaises(FrozenInstanceError):
                setattr(value, next(iter(value.__dataclass_fields__)), None)

    def test_registry_instances_are_isolated_and_duplicates_fail(self) -> None:
        first = CompileOpRegistry()
        second = CompileOpRegistry()

        @first.register(_OP_ID)
        def builder(value=1):
            return value

        self.assertIs(first.lookup(_OP_ID), builder)
        with self.assertRaisesRegex(KeyError, _OP_ID):
            second.lookup(_OP_ID)
        with self.assertRaisesRegex(ValueError, "already registered"):
            first.register(_OP_ID)(builder)
        with self.assertRaisesRegex(KeyError, _OTHER_OP_ID):
            first.lookup(_OTHER_OP_ID)

    def test_registration_requires_fixed_keyword_bindable_signature(self) -> None:
        registry = CompileOpRegistry()

        def variadic(**kwargs):
            return kwargs

        def positional_only(value, /):
            return value

        def reserved(target):
            return target

        for builder, message in (
            (variadic, "fixed keyword-bindable"),
            (positional_only, "fixed keyword-bindable"),
            (reserved, "reserved"),
        ):
            with self.subTest(builder=builder.__name__):
                with self.assertRaisesRegex(TypeError, message):
                    registry.register(_OP_ID)(builder)

    def test_target_and_abi_validation_is_compact_but_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, "arch"):
            RocmTarget("GFX950", 256)
        with self.assertRaisesRegex(ValueError, "positive"):
            RocmTarget("gfx950", 0)
        with self.assertRaisesRegex(ValueError, "same rank"):
            SignatureArg(
                "x",
                ArgumentKind.TENSOR,
                "bf16",
                (4, 8),
                (1,),
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            argument = SignatureArg("x", ArgumentKind.POINTER, "u8")
            KernelSignature((argument, argument))

    def test_resolution_is_cpu_metadata_only(self) -> None:
        touched = []
        registry = CompileOpRegistry()

        @registry.register(_OP_ID)
        def builder(value=1):
            touched.append(value)
            raise AssertionError("resolution must not invoke the compiler")

        unit = registry.make_unit(
            _OP_ID,
            target=_target(),
            signature=KernelSignature(()),
        )
        self.assertEqual(unit.spec.call.arguments, (("value", 1),))
        self.assertEqual(touched, [])
        self.assertNotIn("torch", compile_plan.__dict__)
        self.assertNotIn("flydsl", compile_plan.__dict__)


if __name__ == "__main__":
    unittest.main()
