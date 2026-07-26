# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Focused CPU tests for lightweight FlyDSL compile requests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import importlib.util
from pathlib import Path
import sys
import unittest

_MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "ops" / "flydsl" / "compile_request.py"
)
_MODULE_NAME = "_aiter_compile_request_test"
_SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load {_MODULE_PATH}")
core = importlib.util.module_from_spec(_SPEC)
sys.modules[_MODULE_NAME] = core
_SPEC.loader.exec_module(core)

_OP_ID = "aiter.flydsl.test.builder.v1"


class _Backend:
    def compile_aot(self, request, *, context):
        return request, context

    def load_aot(self, request, *, context, strict=True):
        return request, context, strict

    def resolve_aot(self, request, *, context):
        return request, context


def _target():
    return core.RocmTarget("gfx950", 256)


def _signature():
    return core.KernelSignature(
        (
            core.SignatureArg("input", core.ArgumentKind.POINTER, "u8"),
            core.SignatureArg("rows", core.ArgumentKind.SCALAR, "i32"),
            core.SignatureArg("stream", core.ArgumentKind.STREAM),
        )
    )


class TestCompileRequest(unittest.TestCase):
    def test_lazy_registration_binds_defaults_without_invoking_builder(self):
        registry = core.CompileOpRegistry()
        events = []

        def builder(model_dim: int, *, tile_m: int = 32, enabled: bool = False):
            events.append(("compiled", model_dim, tile_m, enabled))
            return events[-1]

        def loader():
            events.append(("loaded",))
            return builder

        registry.ensure_lazy(_OP_ID, loader)
        registry.ensure_lazy(_OP_ID, lambda: None)
        self.assertEqual(events, [])

        request = registry.make_request(
            _OP_ID,
            target=_target(),
            signature=_signature(),
            model_dim=7168,
            enabled=True,
        )

        self.assertEqual(events, [("loaded",)])
        self.assertEqual(
            request.bound_kwargs,
            (
                ("model_dim", 7168),
                ("tile_m", 32),
                ("enabled", True),
            ),
        )
        self.assertEqual(request.as_kwargs()["tile_m"], 32)
        self.assertIs(registry.resolve(request), builder)
        self.assertEqual(
            registry.compile(request),
            ("compiled", 7168, 32, True),
        )

    def test_request_values_are_hashable_immutable_and_validate_arguments(self):
        registry = core.CompileOpRegistry()

        @registry.register(_OP_ID)
        def builder(rows: int, tile: int = 64):
            return rows, tile

        request = registry.make_request(
            _OP_ID,
            target=_target(),
            signature=_signature(),
            rows=1024,
        )
        context = core.CompileContext(_target(), registry, _Backend())
        for value in (
            request.target,
            request.signature,
            request,
            context,
        ):
            hash(value)
            with self.assertRaises(FrozenInstanceError):
                setattr(value, next(iter(value.__dataclass_fields__)), None)

        self.assertIs(core.CompileUnit, core.CompileRequest)
        with self.assertRaisesRegex(TypeError, "missing"):
            registry.make_request(
                _OP_ID,
                target=_target(),
                signature=_signature(),
            )
        with self.assertRaisesRegex(TypeError, "unexpected"):
            registry.make_request(
                _OP_ID,
                target=_target(),
                signature=_signature(),
                rows=1,
                unexpected=2,
            )
        with self.assertRaisesRegex(TypeError, "hashable"):
            registry.make_request(
                _OP_ID,
                target=_target(),
                signature=_signature(),
                rows=[],
            )

    def test_registration_rejects_duplicate_and_unsupported_signatures(self):
        registry = core.CompileOpRegistry()
        registry.register(_OP_ID)(lambda rows=1: rows)
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(_OP_ID)(lambda: None)

        def variadic(**kwargs):
            return kwargs

        def positional_only(value, /):
            return value

        def reserved(target):
            return target

        for index, (builder, message) in enumerate(
            (
                (variadic, "fixed keyword-bindable"),
                (positional_only, "fixed keyword-bindable"),
                (reserved, "reserved"),
            )
        ):
            with self.subTest(builder=builder.__name__):
                op_id = f"aiter.flydsl.test.invalid_{index}.v1"
                with self.assertRaisesRegex(TypeError, message):
                    registry.register(op_id)(builder)

    def test_target_signature_and_context_validation(self):
        with self.assertRaisesRegex(ValueError, "canonical"):
            core.RocmTarget("not-an-arch", 1)
        with self.assertRaisesRegex(ValueError, "positive"):
            core.RocmTarget("gfx950", 0)
        with self.assertRaisesRegex(ValueError, "unique"):
            field = core.SignatureArg("x", core.ArgumentKind.POINTER, "u8")
            core.KernelSignature((field, field))
        with self.assertRaisesRegex(TypeError, "backend"):
            core.CompileContext(_target(), core.CompileOpRegistry(), object())


if __name__ == "__main__":
    unittest.main()
