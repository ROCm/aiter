# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only execution projection and graph-mutation tests."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import types
import unittest

_FLYDSL_ROOT = Path(__file__).resolve().parents[3] / "ops" / "flydsl"
_PACKAGE = "_aiter_operation_runtime_test"
package = types.ModuleType(_PACKAGE)
package.__path__ = [str(_FLYDSL_ROOT)]
sys.modules[_PACKAGE] = package


def _load(name: str):
    qualified_name = f"{_PACKAGE}.{name}"
    spec = importlib.util.spec_from_file_location(
        qualified_name,
        _FLYDSL_ROOT / f"{name}.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


core = _load("compile_plan")
launch = _load("launch_context")
runtime = _load("operation_runtime")

_GEMM_OP_ID = "aiter.flydsl.test.operation.gemm.v1"
_AUX_OP_ID = "aiter.flydsl.test.operation.synthetic_aux.v1"
_POST_OP_ID = "aiter.flydsl.test.operation.post.v1"


class _Backend:
    def __init__(self) -> None:
        self.compile_calls = []
        self.resolve_calls = []

    def compile_aot(self, unit, *, context):
        self.compile_calls.append(unit.spec.op_id)
        return ("compiled", unit.spec.op_id)

    def load_aot(self, unit, *, context, strict=True):
        return ("loaded", unit.spec.op_id, strict)

    def resolve_aot(self, unit, *, context):
        self.resolve_calls.append(unit.spec.op_id)
        return ("resolved", unit.spec.op_id)


@dataclass(frozen=True)
class _Case:
    rows: int
    include_aux: bool


class TestOperationRuntime(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = _Backend()
        self.registry = core.CompileOpRegistry()
        self.context = core.CompileContext(
            core.RocmTarget("gfx950", 256),
            self.registry,
            self.backend,
        )

        def compiler(rows: int, compile_target=None):
            return rows, compile_target

        self.gemm = core.op(
            _GEMM_OP_ID,
            compiler,
            abi=core.KernelSignature(()),
            target_kw="compile_target",
        )
        self.aux = core.op(
            _AUX_OP_ID,
            compiler,
            abi=core.KernelSignature(()),
            target_kw="compile_target",
        )
        self.post = core.op(
            _POST_OP_ID,
            compiler,
            abi=core.KernelSignature(()),
            target_kw="compile_target",
        )

    def _provider(self):
        gemm = self.gemm
        aux = self.aux
        post = self.post

        @core.operation_plan_provider
        def provider(plan, case: _Case) -> None:
            previous = plan.emit_node(
                "gemm",
                "test.gemm",
                gemm,
                compile_overrides={"rows": case.rows},
            )
            if case.include_aux:
                previous = plan.emit_node(
                    "synthetic_aux",
                    "test.synthetic_aux",
                    aux,
                    dependencies=(previous.node_id,),
                    compile_overrides={"rows": case.rows},
                )
            plan.emit_node(
                "post",
                "test.post",
                post,
                dependencies=(previous.node_id,),
                compile_overrides={"rows": case.rows},
            )

        return provider

    @staticmethod
    def _compile_all(plan, backend, context):
        return tuple(
            backend.compile_aot(unit, context=context)
            for unit in plan.compile_projection().units
        )

    @staticmethod
    def _execute(plan, context):
        adapters = runtime.RuntimeAdapterRegistry()
        trace = []

        def record(step, state, *, context):
            state.append(
                (
                    step.node.node_id,
                    step.node.role,
                    step.artifact,
                    context.stream,
                )
            )

        for role in ("test.gemm", "test.synthetic_aux", "test.post"):
            adapters.register(role, record)
        runtime.execute_operation_plan(
            plan,
            trace,
            compile_context=context,
            launch_context=launch.LaunchContext("stream"),
            adapters=adapters,
        )
        return trace

    def test_synthetic_aux_mutates_compile_and_execution_projections(self) -> None:
        provider = self._provider()
        base = provider(_Case(32, False), context=self.context)
        mutated = provider(_Case(32, True), context=self.context)

        base_compiled = self._compile_all(base, self.backend, self.context)
        mutated_compiled = self._compile_all(mutated, self.backend, self.context)
        base_trace = self._execute(base, self.context)
        mutated_trace = self._execute(mutated, self.context)

        self.assertEqual(
            [artifact[1] for artifact in base_compiled],
            [_GEMM_OP_ID, _POST_OP_ID],
        )
        self.assertEqual(
            [artifact[1] for artifact in mutated_compiled],
            [_GEMM_OP_ID, _AUX_OP_ID, _POST_OP_ID],
        )
        self.assertEqual(
            [entry[:2] for entry in base_trace],
            [("gemm", "test.gemm"), ("post", "test.post")],
        )
        self.assertEqual(
            [entry[:2] for entry in mutated_trace],
            [
                ("gemm", "test.gemm"),
                ("synthetic_aux", "test.synthetic_aux"),
                ("post", "test.post"),
            ],
        )
        self.assertEqual(mutated.nodes[1].dependencies, ("gemm",))
        self.assertEqual(mutated.nodes[2].dependencies, ("synthetic_aux",))
        self.assertNotIn("torch", runtime.__dict__)

    def test_runtime_only_node_and_missing_adapter_are_explicit(self) -> None:
        builder = core.PlanBuilder(self.context)
        builder.emit_node("external", "test.external")
        plan = builder.build_operation_plan("runtime-only")

        steps = runtime.resolve_execution_steps(plan, context=self.context)

        self.assertEqual(len(steps), 1)
        self.assertIsNone(steps[0].artifact)
        with self.assertRaisesRegex(KeyError, "test.external"):
            runtime.execute_operation_plan(
                plan,
                [],
                compile_context=self.context,
                launch_context=launch.LaunchContext("stream"),
                adapters=runtime.RuntimeAdapterRegistry(),
            )


if __name__ == "__main__":
    unittest.main()
