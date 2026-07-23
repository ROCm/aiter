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
CompileContext = compile_plan.CompileContext
CompileOpRegistry = compile_plan.CompileOpRegistry
KernelSignature = compile_plan.KernelSignature
PlanBuilder = compile_plan.PlanBuilder
RocmTarget = compile_plan.RocmTarget
SignatureArg = compile_plan.SignatureArg
op = compile_plan.op
plan_provider = compile_plan.plan_provider

_OP_ID = "aiter.flydsl.test.builder.v1"
_OTHER_OP_ID = "aiter.flydsl.test.other.v1"
_SIMPLE_OP_ID = "aiter.flydsl.test.simple.v1"


def _target() -> RocmTarget:
    return RocmTarget("gfx950", 256)


class _Backend:
    def compile_aot(self, unit, *, context):
        return (unit, context)

    def load_aot(self, unit, *, context, strict=True):
        return (unit, context, strict)

    def resolve_aot(self, unit, *, context):
        return (unit, context)


def _context(
    registry: CompileOpRegistry | None = None,
) -> CompileContext:
    return CompileContext(
        target=_target(),
        registry=registry if registry is not None else CompileOpRegistry(),
        backend=_Backend(),
    )


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

        def builder(
            model_dim: int,
            *,
            tile_m: int = 32,
            enabled: bool = False,
            compile_target=None,
        ):
            call = (
                model_dim,
                tile_m,
                enabled,
                compile_target,
                os.environ.get("FLYDSL_GPU_ARCH"),
                os.environ.get("CU_NUM"),
            )
            self.calls.append(call)
            return call

        self.builder = builder
        self.operation = op(
            _OP_ID,
            builder,
            abi=_abi(),
            target_kw="compile_target",
        )

    def test_make_unit_binds_defaults_in_callable_order_without_invoking(self) -> None:
        before = (
            os.environ.get("FLYDSL_GPU_ARCH"),
            os.environ.get("CU_NUM"),
        )
        binding = PlanBuilder(
            _context(self.registry),
            {"model_dim": 7168},
            context="synthetic case",
        ).bind(
            self.operation,
            enabled=True,
        )

        self.assertEqual(
            binding.unit.spec.call.arguments,
            (
                ("model_dim", 7168),
                ("tile_m", 32),
                ("enabled", True),
                ("compile_target", _target()),
            ),
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
            ({"model_dim": 1, "unknown": 2}, "unknown"),
            ({"model_dim": []}, "hashable"),
        )
        for kwargs, message in invalid:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, message):
                    PlanBuilder(
                        _context(self.registry),
                        context="invalid case",
                    ).bind(
                        self.operation,
                        **kwargs,
                    )

    def test_sources_conflict_until_an_explicit_override_resolves_them(self) -> None:
        plan = PlanBuilder(
            _context(self.registry),
            {"model_dim": 2048},
            {"model_dim": 7168},
            context="model=synthetic",
        )
        with self.assertRaisesRegex(
            TypeError,
            r"model=synthetic.*builder\.v1.*builder.*conflicting parameter "
            r"'model_dim'",
        ):
            plan.bind(self.operation)

        binding = plan.bind(self.operation, model_dim=4096)

        self.assertEqual(binding["model_dim"], 4096)

    def test_registry_compile_invokes_without_mutating_target_environment(self) -> None:
        unit = (
            PlanBuilder(
                _context(self.registry),
            )
            .emit(
                self.operation,
                model_dim=7168,
            )
            .unit
        )
        before = {
            "FLYDSL_GPU_ARCH": os.environ.get("FLYDSL_GPU_ARCH"),
            "CU_NUM": os.environ.get("CU_NUM"),
        }

        result = self.registry.compile(unit)

        self.assertEqual(
            result,
            (
                7168,
                32,
                False,
                _target(),
                before["FLYDSL_GPU_ARCH"],
                before["CU_NUM"],
            ),
        )
        self.assertEqual(self.calls, [result])
        self.assertEqual(
            {name: os.environ.get(name) for name in before},
            before,
        )

    def test_emit_replaces_a_subset_and_preserves_order(self) -> None:
        builder = PlanBuilder(_context(self.registry))
        requested = builder.bind(self.operation, model_dim=2048)
        first = builder.emit(requested)
        second = builder.emit(requested, model_dim=7168, enabled=True)
        third = builder.emit(requested)
        plan = builder.build()

        results = self.registry.compile_plan(plan)

        self.assertEqual([result[0] for result in results], [2048, 7168, 2048])
        self.assertEqual(requested["enabled"], False)
        self.assertEqual(second["enabled"], True)
        self.assertEqual(
            plan.units,
            (first.unit, second.unit, third.unit),
        )


class TestDeclarationsAndRegistry(unittest.TestCase):
    def test_dsl_outputs_are_immutable_and_hashable(self) -> None:
        def compiler(rows=1):
            return rows

        operation = op(_OP_ID, compiler, abi=KernelSignature(()))
        first = CompileOpRegistry()
        second = CompileOpRegistry()
        context = _context(first)
        builder = PlanBuilder(context)
        binding = builder.emit(operation)
        plan = builder.build()

        for value in (
            _target(),
            context,
            operation,
            binding,
            binding.unit.spec.call,
            binding.unit.spec,
            binding.unit,
            plan,
        ):
            hash(value)
            with self.assertRaises(FrozenInstanceError):
                setattr(value, next(iter(value.__dataclass_fields__)), None)

        operation.register(first)
        self.assertIs(first.lookup(_OP_ID), compiler)
        with self.assertRaisesRegex(KeyError, _OP_ID):
            second.lookup(_OP_ID)
        with self.assertRaisesRegex(RuntimeError, "registered"):
            op(_OP_ID, lambda: None, abi=KernelSignature(())).register(first)
        with self.assertRaisesRegex(KeyError, _OTHER_OP_ID):
            first.lookup(_OTHER_OP_ID)

    def test_registration_requires_fixed_keyword_bindable_signature(self) -> None:
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
                    PlanBuilder(
                        _context(),
                    ).bind(op(_OP_ID, builder, abi=KernelSignature(())))

    def test_single_kernel_developer_workflow(self) -> None:
        calls = []

        def compile_simple(rows: int, tile: int = 64, compile_target=None):
            calls.append((rows, tile, compile_target))
            return calls[-1]

        simple = op(
            _SIMPLE_OP_ID,
            compile_simple,
            abi=KernelSignature((SignatureArg("rows", ArgumentKind.SCALAR, "i32"),)),
            target_kw="compile_target",
        )

        @plan_provider
        def simple_plan(plan: PlanBuilder, *, rows: int) -> None:
            plan.require(rows > 0, f"rows must be positive, got {rows}")
            plan.emit(simple)

        registry = CompileOpRegistry()
        context = _context(registry)
        first = simple_plan(context=context, rows=1024)
        second = simple_plan(context=context, rows=1024)

        self.assertEqual(calls, [])
        self.assertNotIn("torch", compile_plan.__dict__)
        self.assertNotIn("flydsl", compile_plan.__dict__)
        self.assertEqual(first, second)
        self.assertEqual(
            first.units[0].spec.call.arguments,
            (("rows", 1024), ("tile", 64), ("compile_target", _target())),
        )
        self.assertEqual(registry.compile_plan(first), ((1024, 64, _target()),))


if __name__ == "__main__":
    unittest.main()
