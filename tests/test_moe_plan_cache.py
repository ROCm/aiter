# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU checks for the legacy MoE plan policy, with no torch/ROCm dependency.

Compile the actual host functions from source, unchanged, with controlled
resolution inputs and launch traps. This checks cache control flow rather than
native kernels; op_tests/test_moe_eager_host_path.py supplies native coverage.
Run with unittest (or pytest) on a CPU-only checkout.
"""

import ast
import contextlib
import enum
import inspect
import math
import os
import sys
import types
import unittest
from pathlib import Path
from typing import NamedTuple
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "aiter/fused_moe.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))

# Roles for cacheable resolution; tensor subclass/mode eligibility is orthogonal.
KEY_PARAMETERS = {
    "a1_scale",
    "a2_scale",
    "activation",
    "beta",
    "bias1",
    "bias2",
    "block_size_M",
    "doweight_stage1",
    "dtype",
    "expert_mask",
    "gate_mode",
    "hidden_pad",
    "hidden_states",
    "intermediate_pad",
    "linear_beta",
    "num_local_tokens",
    "quant_dtype_a",
    "quant_dtype_a2",
    "quant_type",
    "quant_type_a",
    "swiglu_limit",
    "topk_ids",
    "w1",
    "w2",
}
SKIP_PARAMETERS = {
    "_config_shape",
    "_enforce_metadata_block_size",
    "_metadata_config_file",
    "_metadata_transform",
    "_q_dtype_a",
    "_stage1_extra_args",
    "_stage2_extra_args",
    "_stage2_override",
    "deterministic",
    "herd_topk",
    "splitk",
    "stage2_scatter",
    "w1_global_scale",
    "w2_global_scale",
}
# Read per call by validation/execution, or only by an already excluded route
# (herd_routed_scaling_factor needs herd_topk). None changes a cacheable plan.
EXECUTION_PARAMETERS = {
    "herd_routed_scaling_factor",
    "moe_sorting_dispatch_policy",
    "output",
    "topk_weight",
    "w1_scale",
    "w2_scale",
}


class Activation(enum.IntEnum):
    Silu = 0
    Situv2 = 1


class Quant(enum.IntEnum):
    No = 0
    per_1x32 = 1


class Gate(enum.Enum):
    SEPARATED = "separated"
    INTERLEAVE = "interleave"


class Tensor:
    def __init__(self, shape, *, contiguous=True, row_major=True, storage=None):
        self.shape = shape
        self.dtype = "bf16"
        self.device = "cpu"
        self.row_major = row_major
        self._contiguous = contiguous
        self.storage = object() if storage is None else storage
        self.copies = 0

    def is_contiguous(self):
        return self._contiguous

    def contiguous(self):
        self.copies += 1
        return Tensor(self.shape)

    def numel(self):
        return math.prod(self.shape)


def _load_host_functions(scope, tree=TREE):
    names = {
        "_MoePlan",
        "_prepare_moe_input",
        "_check_moe_plan",
        "_check_moe_plan_route",
        "_fused_moe_impl",
        "_op_or_eager_body",
    }
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    assert len(nodes) == len(names)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    # Only selected definitions from the checked-out source, never user input.
    code = compile(ast.fix_missing_locations(module), str(SOURCE), "exec")
    exec(code, scope)  # noqa: S102


class PlanCacheTests(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(
            os.environ, {"AITER_MOE_PLAN_CACHE": "1", "AITER_MOE_PLAN_CHECK": "1"}
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.events = []
        self.metadata = types.SimpleNamespace(
            stage1=object(),
            stage2=object(),
            block_m=32,
            flat=0,
            full_impl=None,
            output_aux=False,
            run_1stage=False,
            has_bias=False,
        )

        def validate_output(output, shape, dtype, device):
            self.events.append("metadata")
            if output is not None and (
                output.shape != shape
                or output.dtype != dtype
                or output.device != device
                or not output.is_contiguous()
            ):
                raise ValueError("invalid output metadata")

        def validate_overlap(output, hidden):
            self.events.append("overlap")
            if output is not None and output.storage is hidden.storage:
                raise ValueError("output overlaps input")

        def resolve(*args, **kwargs):
            self.events.append("resolve")
            return "bf16"

        def lookup(*args, **kwargs):
            self.events.append("lookup")
            return self.metadata

        def run(plan, hidden, *args, **kwargs):
            self.events.append("run")
            return hidden

        self.scope = {
            "__name__": "moe_plan_cpu_test",
            "NamedTuple": NamedTuple,
            "ActivationType": Activation,
            "QuantType": Quant,
            "GateMode": Gate,
            "math": math,
            "os": os,
            "_SILU_VALUE": Activation.Silu.value,
            "_MOE_PLANS": {},
            "_MOE_PLANS_MAX": 1024,
            "pop_presorted_moe": mock.Mock(return_value=None),
            "_plain_eager": mock.Mock(return_value=True),
            "_is_row_major_2d": lambda t: t.row_major,
            "_validate_output_buffer_metadata": validate_output,
            "_validate_output_buffer_no_overlap": validate_overlap,
            "get_inter_dim": lambda a, b: (a[0], b[1], b[2]),
            "get_gfx": lambda: "gfx942",
            "get_gfx_runtime": lambda: "gfx942",
            "dtypes": types.SimpleNamespace(fp16="fp16", bf16="bf16", fp4x2="fp4"),
            "quant_remap": {},
            "resolve_activation_dtype": resolve,
            "_can_use_bf16_g1u1_small_m_direct": lambda **kw: False,
            "_can_reroute_mxfp4_to_flydsl": lambda **kw: False,
            "fused_moe_g1u1_bf16_small_m_direct": self.fail_launch,
            "_normalize_mxfp4_activation_params": lambda *a: (1.0, 1.0, None),
            "get_padded_M": lambda m: m,
            "get_2stage_cfgs": lookup,
            "_is_mxmoe_opus_metadata": lambda m: False,
            "_is_mxfp4_inline_sort": lambda m: False,
            "_mxfp4_a4w4_stage1_fw": object(),
            "cktile_moe_stage2": object(),
            "AsmMoeMode": lambda flat: types.SimpleNamespace(requires_host_sort=True),
            "_needs_swiglu_bias_support": lambda *a: False,
            "_BF16_G1U1_SMALL_M_DIRECT_GFX": "gfx1201",
            "_fused_moe_run": run,
        }
        _load_host_functions(self.scope)
        grouped = types.ModuleType("aiter.ops.flydsl.grouped_moe_gfx1250")
        grouped.grouped_gemm_gfx1250_a8w4 = self.fail_launch
        modules = mock.patch.dict(sys.modules, {grouped.__name__: grouped})
        modules.start()
        self.addCleanup(modules.stop)
        self.inputs = {
            "hidden_states": Tensor((7, 64)),
            "w1": Tensor((4, 32, 64)),
            "w2": Tensor((4, 64, 16)),
            "topk_weight": Tensor((7, 2)),
            "topk_ids": Tensor((7, 2)),
        }

    def fail_launch(self, *args, **kwargs):
        self.fail("unexpected early GPU execution")

    def call(self, **kwargs):
        return self.scope["_fused_moe_impl"](**(self.inputs | kwargs))

    def seed(self):
        self.call()
        ((key, plan),) = self.scope["_MOE_PLANS"].items()
        self.events.clear()
        return key, plan

    def test_signature_roles_are_disjoint_complete_and_match_source_reads(self):
        params = set(inspect.signature(self.scope["_fused_moe_impl"]).parameters)
        roles = (KEY_PARAMETERS, SKIP_PARAMETERS, EXECUTION_PARAMETERS)
        self.assertEqual(len(params), 44)
        self.assertEqual(tuple(map(len, roles)), (24, 14, 6))
        self.assertEqual(set.union(*roles), params)
        for i, role in enumerate(roles):
            for other in roles[i + 1 :]:
                self.assertFalse(role & other)
        impl = next(
            n for n in TREE.body if getattr(n, "name", None) == "_fused_moe_impl"
        )
        key = next(
            n
            for n in ast.walk(impl)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "plan_key" for t in n.targets)
            and isinstance(n.value, ast.Tuple)
        )
        reads = (
            lambda node: {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            & params
        )
        self.assertEqual(reads(key.value), KEY_PARAMETERS)
        eligibility = next(
            n for n in impl.body if isinstance(n, ast.If) and key in n.body
        )
        # The final _plain_eager tensor-type/mode guard is orthogonal to roles.
        feature_guards = ast.BoolOp(op=ast.And(), values=eligibility.test.values[:-1])
        self.assertEqual(reads(feature_guards), SKIP_PARAMETERS)

    def test_checked_hit_resolves_current_inputs_and_runs_once(self):
        self.seed()
        self.call()
        self.assertEqual(self.events.count("resolve"), 1)
        self.assertEqual(self.events.count("lookup"), 2)
        self.assertEqual(self.events.count("run"), 1)
        self.assertEqual(self.events.count("metadata"), 1)
        self.assertEqual(self.events.count("overlap"), 1)
        self.assertEqual(self.scope["pop_presorted_moe"].call_count, 2)

    def test_check_is_dynamic_and_unchecked_hit_does_not_resolve(self):
        self.seed()
        with mock.patch.dict(os.environ, {"AITER_MOE_PLAN_CHECK": "0"}):
            self.call()
        self.assertNotIn("resolve", self.events)
        self.assertEqual(self.events.count("run"), 1)
        self.events.clear()
        self.call()
        self.assertEqual(self.events.count("resolve"), 1)
        self.assertEqual(self.events.count("run"), 1)

    def test_disabled_cache_neither_reads_nor_stores_or_builds_a_key(self):
        key, plan = self.seed()

        class ForbiddenCache(dict):
            def get(self, *args):
                raise AssertionError("cache read while disabled")

            def __setitem__(self, *args):
                raise AssertionError("cache store while disabled")

            def __len__(self):
                raise AssertionError("cache store path while disabled")

        cached = self.scope["_MOE_PLANS"]
        self.scope["_MOE_PLANS"] = ForbiddenCache(cached)
        # An injected key expression must be unreachable, not merely un-stored.
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Tuple)
                and any(
                    isinstance(t, ast.Name) and t.id == "plan_key" for t in node.targets
                )
            ):
                node.value = ast.Call(
                    func=ast.Name(id="forbidden_key", ctx=ast.Load()),
                    args=[],
                    keywords=[],
                )
        self.scope["forbidden_key"] = self.fail_launch
        _load_host_functions(self.scope, tree)
        with mock.patch.dict(os.environ, {"AITER_MOE_PLAN_CACHE": "0"}):
            self.call()
            self.call()
        self.assertEqual(self.events.count("resolve"), 2)
        self.assertEqual(self.events.count("run"), 2)
        self.assertIs(self.scope["_MOE_PLANS"][key], plan)
        self.scope["_MOE_PLANS"] = cached
        _load_host_functions(self.scope)
        self.events.clear()
        self.call()
        self.assertEqual(self.events.count("lookup"), 2)
        self.assertEqual(self.events.count("run"), 1)

    def test_nonzero_splitk_excludes_the_cache_but_default_still_caches(self):
        key, plan = self.seed()
        with mock.patch.dict(os.environ, {"AITER_MOE_PLAN_CHECK": "0"}):
            self.call(splitk=1)
        self.assertEqual(self.events.count("resolve"), 1)
        self.assertEqual(self.events.count("run"), 1)
        self.assertIs(self.scope["_MOE_PLANS"][key], plan)

    def test_mismatch_is_before_store_or_execution(self):
        key, plan = self.seed()
        # Change current-call resolution without changing its public inputs.
        self.metadata.block_m += 1
        with self.assertRaisesRegex(RuntimeError, "block_size_M") as exc:
            self.call()
        self.assertIn(repr(key), str(exc.exception))
        self.assertNotIn("run", self.events)
        self.assertIs(self.scope["_MOE_PLANS"][key], plan)

    def test_every_field_is_compared_and_metadata_callables_use_identity(self):
        key, plan = self.seed()

        class EqualToAnything:
            def __eq__(self, other):
                return True

        for field in plan._fields:
            with self.subTest(field=field):
                value = (
                    EqualToAnything()
                    if field in ("metadata", "stage1_func")
                    else object()
                )
                with self.assertRaisesRegex(RuntimeError, field):
                    self.scope["_check_moe_plan"](
                        key, plan, plan._replace(**{field: value})
                    )
        self.scope["_check_moe_plan"](
            key, plan, plan._replace(cfg_kwargs=dict(plan.cfg_kwargs))
        )

    def test_fresh_metadata_identity_mismatch_is_caught_after_revalidation(self):
        key, plan = self.seed()
        equal_metadata = types.SimpleNamespace(**vars(self.metadata))
        self.assertEqual(equal_metadata, self.metadata)
        self.scope["get_2stage_cfgs"] = mock.Mock(
            side_effect=[self.metadata, equal_metadata]
        )
        with self.assertRaisesRegex(RuntimeError, "metadata"):
            self.call()
        self.assertNotIn("run", self.events)
        self.assertIs(self.scope["_MOE_PLANS"][key], plan)

    def test_stale_metadata_is_a_normal_miss(self):
        key, plan = self.seed()
        self.metadata = types.SimpleNamespace(**vars(self.metadata))
        self.call()
        self.assertEqual(self.events.count("run"), 1)
        self.assertIsNot(self.scope["_MOE_PLANS"][key].metadata, plan.metadata)

    def test_early_routes_fail_before_their_launch(self):
        _key, plan = self.seed()
        for route in (
            "grouped_gemm_gfx1250_a8w4",
            "gfx1201_small_m_direct",
            "metadata.full_impl",
        ):
            with self.subTest(route=route):
                # Simulate a future key collision/eligibility change: a checked
                # cached plan must not reach any of the earlier GPU call sites.
                self.scope["_MOE_PLANS"] = mock.Mock(get=mock.Mock(return_value=plan))
                self.scope["_can_use_bf16_g1u1_small_m_direct"] = (
                    lambda route=route, **kw: route == "gfx1201_small_m_direct"
                )
                self.metadata.full_impl = (
                    self.fail_launch if route == "metadata.full_impl" else None
                )
                self.events.clear()
                kwargs = (
                    {"gate_mode": Gate.INTERLEAVE.value}
                    if route == "grouped_gemm_gfx1250_a8w4"
                    else {}
                )
                with self.assertRaisesRegex(RuntimeError, route):
                    self.call(**kwargs)
                self.assertNotIn("run", self.events)

    def test_missing_key_field_is_detected_from_actual_current_call(self):
        # Mutation check: remove hidden_pad from the real key, then change it.
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Tuple)
                and any(
                    isinstance(t, ast.Name) and t.id == "plan_key" for t in node.targets
                )
            ):
                node.value.elts = [
                    n
                    for n in node.value.elts
                    if not (isinstance(n, ast.Name) and n.id == "hidden_pad")
                ]
        _load_host_functions(self.scope, tree)
        key, plan = self.seed()
        with self.assertRaisesRegex(RuntimeError, "cfg_args"):
            self.call(hidden_pad=64)
        self.assertNotIn("run", self.events)
        self.assertIs(self.scope["_MOE_PLANS"][key], plan)

    def test_hit_and_miss_share_output_checks_and_copy_order(self):
        for hit in (False, True):
            for checking in ("0", "1"):
                for row_major in (False, True):
                    with self.subTest(hit=hit, check=checking, row_major=row_major):
                        self.scope["_MOE_PLANS"].clear()
                        if hit:
                            self.seed()
                        self.events.clear()
                        hidden = Tensor((7, 64), contiguous=False, row_major=row_major)
                        output = Tensor((7, 64), storage=hidden.storage)
                        with mock.patch.dict(
                            os.environ, {"AITER_MOE_PLAN_CHECK": checking}
                        ):
                            if row_major:
                                with self.assertRaisesRegex(ValueError, "overlaps"):
                                    self.call(hidden_states=hidden, output=output)
                                self.assertEqual(hidden.copies, 0)
                                self.assertNotIn("run", self.events)
                            else:
                                self.call(hidden_states=hidden, output=output)
                                self.assertEqual(hidden.copies, 1)
                                self.assertEqual(self.events.count("run"), 1)
                        self.assertEqual(self.events.count("metadata"), 1)
                        self.assertEqual(self.events.count("overlap"), 1)

    def test_invalid_output_metadata_fails_on_hit_and_miss(self):
        for hit in (False, True):
            for checking in ("0", "1"):
                self.scope["_MOE_PLANS"].clear()
                if hit:
                    self.seed()
                self.events.clear()
                with mock.patch.dict(
                    os.environ, {"AITER_MOE_PLAN_CHECK": checking}
                ), self.assertRaisesRegex(ValueError, "metadata"):
                    self.call(output=Tensor((1, 64)))
                self.assertNotIn("overlap", self.events)
                self.assertNotIn("run", self.events)

    def test_row_major_view_is_finalized_once_and_inline_sort_retains_it(self):
        hidden = Tensor((7, 64), contiguous=False)
        prepared = self.scope["_prepare_moe_input"](hidden, shape=(7, 64), dtype="bf16")
        self.assertIs(prepared, hidden)
        self.assertEqual(hidden.copies, 0)
        self.assertIs(self.scope["_prepare_moe_input"](hidden, validate=False), hidden)
        final = self.scope["_prepare_moe_input"](hidden, validate=False, finalize=True)
        self.assertTrue(final.is_contiguous())
        self.assertEqual(hidden.copies, 1)
        self.assertEqual(self.events, ["metadata", "overlap"])

    def test_profiler_wraps_only_the_selected_direct_body_when_enabled(self):
        events = []

        @contextlib.contextmanager
        def record(name):
            events.append(name)
            yield

        def body(value):
            events.append("body")
            return value

        def op(value):
            with record("aiter::fused_moe_"):
                return body(value)

        op._eager_body = body
        enabled = mock.Mock(return_value=False)
        self.scope["torch"] = types.SimpleNamespace(
            autograd=types.SimpleNamespace(_profiler_enabled=enabled),
            profiler=types.SimpleNamespace(record_function=record),
        )
        select = self.scope["_op_or_eager_body"]
        self.assertIs(select(op, None), body)
        self.assertEqual(events, [])
        enabled.return_value = True
        self.assertEqual(select(op, None)(3), 3)
        self.assertEqual(events, ["aiter::fused_moe_", "body"])
        events.clear()
        self.scope["_plain_eager"].return_value = False
        self.assertIs(select(op, None), op)
        self.assertEqual(select(op, None)(4), 4)
        self.assertEqual(events, ["aiter::fused_moe_", "body"])
        events.clear()
        self.scope["_plain_eager"].return_value = True
        self.assertIs(select(body, None), body)  # no _eager_body on this callable


if __name__ == "__main__":
    unittest.main()
