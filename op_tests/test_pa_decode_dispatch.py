# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU routing contracts and a focused Q4 complete-step benchmark.

CPU checks: python -m unittest op_tests.test_pa_decode_dispatch.DispatchContractTests
GPU checks: python op_tests/test_pa_decode_dispatch.py -b 2 4 --context-length 513
"""

import argparse
import importlib.util
import itertools
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def _load_contract_module():
    # CPU routing checks must not import AITER's GPU runtime or require FlyDSL.
    name = "_pa_decode_contract_test"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[1] / "aiter/ops/pa_decode.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class DispatchContractTests(unittest.TestCase):
    def setUp(self):
        self.op = _load_contract_module()
        self.shape = self.op.PADecodeShape(
            "gfx950",
            4,
            4,
            16,
            1,
            128,
            16,
            "bfloat16",
            "float8_e4m3fn",
            True,
            True,
            mask_unowned_kv=False,
            compute_units=256,
        )
        self.stats = self.op.PAContextStats.from_lengths([32768] * 4, 16)

    def test_small_or_unknown_lengths_stay_flydsl(self):
        self.assertEqual(
            self.op.select_pa_decode_backend(self.shape, self.stats), "flydsl"
        )
        self.assertEqual(self.op.select_pa_decode_backend(self.shape), "flydsl")

    def test_forced_asm_supported_contract(self):
        self.assertEqual(
            self.op.select_pa_decode_backend(self.shape, self.stats, backend="asm"),
            "asm",
        )

    def test_forced_asm_does_not_discard_features(self):
        unsupported = {
            "arch": "gfx942",
            "batch": 64,
            "query_length": 1,
            "query_heads": 32,
            "kv_heads": 2,
            "head_dim": 256,
            "page_size": 64,
            "query_dtype": "float16",
            "kv_dtype": "bfloat16",
            "asm_layout": False,
            "per_token_scales": False,
            "sliding_window": 1024,
            "has_query_scale": True,
            "has_alibi": True,
            "has_sinks": True,
            "mask_unowned_kv": True,
            "softmax_scale": 0.25,
            "compute_units": None,
        }
        for field, value in unsupported.items():
            with self.subTest(field=field):
                shape = replace(self.shape, **{field: value})
                with self.assertRaises(ValueError):
                    self.op.select_pa_decode_backend(shape, backend="asm")
                self.assertEqual(
                    self.op.select_pa_decode_backend(shape, backend="flydsl"), "flydsl"
                )

    def test_stats_reject_wrong_batch_or_short_asm_context(self):
        with self.assertRaises(ValueError):
            self.op.select_pa_decode_backend(
                self.shape, self.op.PAContextStats.from_lengths([16], 16)
            )
        with self.assertRaises(ValueError):
            self.op.select_pa_decode_backend(
                self.shape,
                self.op.PAContextStats.from_lengths([0, 4, 4, 4], 16),
                backend="asm",
            )

    def test_host_stats_count_used_pages_not_capacity(self):
        stats = self.op.PAContextStats.from_lengths([0, 1, 16, 17], 16)
        self.assertEqual(
            (stats.minimum, stats.maximum, stats.total, stats.used_pages),
            (0, 17, 34, 4),
        )

    def test_gpu_hint_is_rejected_before_iteration(self):
        class DeviceLengths:
            device = "cuda"

            def __iter__(self):
                raise AssertionError("GPU length hints must not be iterated")

        with self.assertRaises(TypeError):
            self.op.PAContextStats.from_lengths(DeviceLengths(), 16)

    def test_non_integer_host_hints_are_rejected(self):
        for values in ([True], [3.5], [], [-1]):
            with self.subTest(values=values), self.assertRaises(
                (TypeError, ValueError)
            ):
                self.op.PAContextStats.from_lengths(values, 16)

    def test_unknown_backend_is_not_silently_flydsl(self):
        with self.assertRaises(ValueError):
            self.op.select_pa_decode_backend(self.shape, backend="typo")

    def _stats(self, batch, total, first=None):
        count = batch if first is None else batch - 1
        quotient, remainder = divmod(total if first is None else total - first, count)
        values = [quotient + (index < remainder) for index in range(count)]
        if first is not None:
            values.insert(0, first)
        return self.op.PAContextStats.from_lengths(values, 16)

    def test_candidate_crossover_separates_packed_and_conversion_cost(self):
        for batch in (24, 32):
            shape = replace(self.shape, batch=batch)
            for delta, expected in ((-1, "flydsl"), (0, "asm"), (1, "asm")):
                stats = self._stats(batch, 3 * 1024 * 1024 + delta)
                with self.subTest(batch=batch, packed=True, delta=delta):
                    self.assertEqual(
                        self.op.select_pa_decode_backend(
                            shape, stats, has_packed_csr=True
                        ),
                        expected,
                    )
                    self.assertEqual(
                        self.op.select_pa_decode_backend(
                            shape, stats, has_packed_csr=False
                        ),
                        "flydsl",
                    )
        shape = replace(self.shape, batch=32)
        for delta, expected in ((-1, "flydsl"), (0, "asm"), (1, "asm")):
            stats = self._stats(32, 4 * 1024 * 1024 + delta)
            self.assertEqual(self.op.select_pa_decode_backend(shape, stats), expected)
        self.assertEqual(
            self.op.select_pa_decode_backend(
                replace(self.shape, batch=24), self._stats(24, 4 * 1024 * 1024)
            ),
            "flydsl",
        )
        self.assertEqual(
            self.op.select_pa_decode_backend(
                replace(self.shape, batch=16),
                self._stats(16, 4 * 1024 * 1024),
                has_packed_csr=True,
            ),
            "flydsl",
        )

    def test_candidate_calibration_bounds_and_minimum(self):
        shape = replace(self.shape, batch=32)
        for maximum, expected in ((768 * 1024, "asm"), (768 * 1024 + 1, "flydsl")):
            stats = self._stats(32, 4 * 1024 * 1024, first=maximum)
            self.assertEqual(self.op.select_pa_decode_backend(shape, stats), expected)
        for total, expected in (
            (6 * 1024 * 1024, "asm"),
            (6 * 1024 * 1024 + 1, "flydsl"),
        ):
            self.assertEqual(
                self.op.select_pa_decode_backend(shape, self._stats(32, total)),
                expected,
            )
        self.assertEqual(
            self.op.select_pa_decode_backend(
                shape, self._stats(32, 4 * 1024 * 1024, first=3), has_packed_csr=True
            ),
            "flydsl",
        )

    def test_explicit_fly_overrides_are_not_discarded(self):
        shape = replace(self.shape, batch=32)
        stats = self._stats(32, 4 * 1024 * 1024)
        for options in (
            {"max_partitions": 256},
            {"workgroup_budget": 256},
            {"workgroup_budget": 512},
            {"compute_type": "bfloat16"},
        ):
            with self.subTest(options=options):
                self.assertEqual(
                    self.op.select_pa_decode_backend(
                        shape, stats, has_packed_csr=True, **options
                    ),
                    "flydsl",
                )
                with self.assertRaisesRegex(ValueError, "cannot honor"):
                    self.op.select_pa_decode_backend(
                        shape, stats, backend="asm", **options
                    )
        for compute_type in (None, "float8_e4m3fn", "torch.float8_e4m3fn"):
            self.assertEqual(
                self.op.select_pa_decode_backend(
                    shape, stats, has_packed_csr=True, compute_type=compute_type
                ),
                "asm",
            )

    def test_unknown_or_partitioned_device_stays_untuned(self):
        stats = self._stats(32, 4 * 1024 * 1024)
        for compute_units in (None, 0, 128, 304):
            shape = replace(self.shape, batch=32, compute_units=compute_units)
            with self.subTest(compute_units=compute_units):
                self.assertEqual(
                    self.op.select_pa_decode_backend(shape, stats, has_packed_csr=True),
                    "flydsl",
                )
                self.assertIsNone(self.op._fly_workgroup_budget(shape, stats))
                with self.assertRaises(ValueError):
                    self.op.select_pa_decode_backend(shape, stats, backend="asm")

    def test_missing_stats_stays_fly_and_does_not_tune(self):
        shape = replace(self.shape, batch=32)
        for packed in (False, True):
            self.assertEqual(
                self.op.select_pa_decode_backend(shape, has_packed_csr=packed), "flydsl"
            )
        self.assertIsNone(self.op._fly_workgroup_budget(shape, None))

    def test_default_mask_keeps_fly_even_in_asm_size_domain(self):
        shape = replace(self.shape, batch=32, mask_unowned_kv=True)
        stats = self._stats(32, 4 * 1024 * 1024)
        for packed in (False, True):
            self.assertEqual(
                self.op.select_pa_decode_backend(shape, stats, has_packed_csr=packed),
                "flydsl",
            )
        self.assertEqual(self.op._fly_workgroup_budget(shape, stats), 512)

    def test_fly_budget_thresholds_and_explicit_budget(self):
        for batch in (8, 16, 24, 32):
            shape = replace(self.shape, batch=batch)
            for delta, expected in ((-1, None), (0, 512), (1, 512)):
                self.assertEqual(
                    self.op._fly_workgroup_budget(
                        shape, self._stats(batch, batch * 32768 + delta)
                    ),
                    expected,
                )
        self.assertIsNone(self.op._fly_workgroup_budget(self.shape, self.stats))
        shape = replace(self.shape, batch=32)
        stats = self._stats(32, 4 * 1024 * 1024)
        self.assertEqual(
            self.op.select_pa_decode_backend(shape, stats, backend="flydsl"), "flydsl"
        )
        self.assertEqual(self.op._fly_workgroup_budget(shape, stats), 512)
        self.assertEqual(
            self.op._fly_workgroup_budget(shape, stats, workgroup_budget=256), 256
        )
        self.assertIsNone(
            self.op._fly_workgroup_budget(shape, stats, max_partitions=256)
        )
        self.assertIsNone(
            self.op._fly_workgroup_budget(shape, stats, compute_type="bfloat16")
        )
        self.assertIsNone(
            self.op._fly_workgroup_budget(shape, self._stats(32, 6 * 1024 * 1024 + 1))
        )
        self.assertIsNone(
            self.op._fly_workgroup_budget(replace(shape, softmax_scale=0.5), stats)
        )
        self.assertIsNone(
            self.op._fly_workgroup_budget(
                replace(shape, batch=2), self._stats(2, 2 * 32768)
            )
        )

    def test_b4_fly_budget_regression_and_total_boundary(self):
        # These real steps exceed the 64Ki maximum-length gate but regressed
        # with 512 tasks because their total work is below 256Ki tokens.
        for lengths in ([3537, 48020, 77086, 48020], [68594, 19386, 50248, 50248]):
            stats = self.op.PAContextStats.from_lengths(lengths, 16)
            self.assertGreater(stats.maximum, 65536)
            self.assertIsNone(self.op._fly_workgroup_budget(self.shape, stats))
        # Hold max length at 128Ki so this tests the total-work boundary alone.
        for delta, expected in ((-1, None), (0, 512), (1, 512)):
            stats = self._stats(4, 262144 + delta, first=131072)
            self.assertEqual(self.op._fly_workgroup_budget(self.shape, stats), expected)
        for length, expected in ((65535, None), (65536, 512)):
            stats = self.op.PAContextStats.from_lengths([length] * 4, 16)
            self.assertEqual(self.op._fly_workgroup_budget(self.shape, stats), expected)

    def test_selector_does_not_coerce_tensor_like_options(self):
        class TensorLike:
            def __str__(self):
                raise AssertionError("must not read a tensor value")

            def __bool__(self):
                raise AssertionError("must not read a tensor value")

        shape = replace(self.shape, batch=32)
        stats = self._stats(32, 4 * 1024 * 1024)
        self.assertEqual(
            self.op.select_pa_decode_backend(shape, stats, compute_type=TensorLike()),
            "flydsl",
        )
        with self.assertRaises(TypeError):
            self.op.select_pa_decode_backend(shape, stats, has_packed_csr=TensorLike())

    def test_nondefault_scale_uses_fly_and_cannot_launch_asm(self):
        changed = replace(self.shape, softmax_scale=0.25)
        self.assertEqual(
            self.op.select_pa_decode_backend(changed, self.stats), "flydsl"
        )
        plan = self.op.PADecodePlan(
            "asm", self.shape, None, None, self.stats, (), refreshed=True
        )
        # This must reject before inspecting tensor metadata or importing the
        # native launcher; passing None tensor arguments makes that observable.
        with self.assertRaisesRegex(ValueError, "softmax_scale"):
            self.op.pa_decode(None, None, None, None, None, None, 0.25, 4, plan=plan)

    def test_missing_flydsl_only_falls_back_for_supported_auto(self):
        missing = ModuleNotFoundError("No module named 'flydsl'", name="flydsl")
        with mock.patch.object(self.op, "_import_fly_planner", side_effect=missing):
            self.assertIsNone(
                self.op._resolve_fly_planner(
                    self.shape, self.stats, "auto", asm_fallback_allowed=True
                )
            )
            with self.assertRaisesRegex(ModuleNotFoundError, "requirements.txt"):
                self.op._resolve_fly_planner(
                    self.shape, None, "auto", asm_fallback_allowed=True
                )
            for shape, request, allowed in (
                (self.shape, "flydsl", True),
                (replace(self.shape, mask_unowned_kv=True), "auto", True),
                (replace(self.shape, softmax_scale=0.25), "auto", True),
                (self.shape, "auto", False),
            ):
                with self.subTest(
                    shape=shape, request=request, allowed=allowed
                ), self.assertRaisesRegex(ModuleNotFoundError, "requirements.txt"):
                    self.op._resolve_fly_planner(
                        shape, self.stats, request, asm_fallback_allowed=allowed
                    )

    def test_broken_flydsl_import_is_never_a_backend_fallback(self):
        for error in (
            ModuleNotFoundError("missing compiler", name="flydsl.compiler"),
            ModuleNotFoundError("missing dependency", name="other_dependency"),
            ImportError("compiler mismatch"),
            RuntimeError("compiler failure"),
        ):
            with mock.patch.object(self.op, "_import_fly_planner", side_effect=error):
                with self.assertRaises(type(error)) as caught:
                    self.op._resolve_fly_planner(
                        self.shape, self.stats, "auto", asm_fallback_allowed=True
                    )
                self.assertIs(caught.exception, error)

    def test_refresh_does_not_pin_the_prepare_stream(self):
        class TensorMetadata:
            dtype = "int32"
            device = "cuda"

            def __init__(self, shape, strides, pointer):
                self.shape, self.strides, self.pointer = shape, strides, pointer

            def stride(self):
                return self.strides

            def data_ptr(self):
                return self.pointer

        lengths = TensorMetadata((4,), (1,), 100)
        table = TensorMetadata((4, 2048), (2048, 1), 200)
        plan = self.op.PADecodePlan(
            "flydsl", self.shape, lengths, table, self.stats, ()
        )
        plan._step_addresses = (100, 200)
        plan._step_contract = (self.op._contract(lengths), self.op._contract(table))
        planner = mock.Mock()
        # The plan never queries a stream ID. Serial side-stream use is legal
        # after the caller orders dependencies; GPU tests verify real capture.
        forbidden_cuda = SimpleNamespace(
            current_stream=mock.Mock(side_effect=AssertionError)
        )
        modules = {
            "aiter.ops.flydsl.pa_decode": SimpleNamespace(plan_pa_decode=planner),
            "torch": SimpleNamespace(cuda=forbidden_cuda),
        }
        with mock.patch.dict(sys.modules, modules):
            self.op.refresh_pa_decode_plan(plan)
            self.op.refresh_pa_decode_plan(plan)
        self.assertTrue(plan.refreshed)
        self.assertEqual(planner.call_count, 2)
        forbidden_cuda.current_stream.assert_not_called()


class PreparedDispatchTests(unittest.TestCase):
    def fixture(self, batch, length, packed):
        device = SimpleNamespace(type="cuda")
        addresses = itertools.count(1 << 40, 1 << 40)

        class TensorMetadata:
            def __init__(self, shape, dtype):
                self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
                self.ndim = len(self.shape)
                self.dtype, self.device = dtype, device
                self.pointer = next(addresses)

            def numel(self):
                return math.prod(self.shape)

            def stride(self, index=None):
                strides = tuple(
                    math.prod(self.shape[axis + 1 :]) for axis in range(self.ndim)
                )
                return strides if index is None else strides[index]

            def is_contiguous(self):
                return True

            def data_ptr(self):
                return self.pointer

            def element_size(self):
                return {"float32": 4, "int32": 4, "bfloat16": 2, "float8_e4m3fn": 1}[
                    self.dtype
                ]

            def __mul__(self, value):
                return self

            def cpu(self):
                raise AssertionError("dispatch must not read GPU lengths back")

        def allocate(shape, *, dtype, device):
            return TensorMetadata(shape, dtype)

        torch = SimpleNamespace(
            int32="int32",
            float32="float32",
            dtype=str,
            empty=mock.Mock(side_effect=allocate),
            arange=mock.Mock(side_effect=allocate),
            cuda=SimpleNamespace(
                is_current_stream_capturing=mock.Mock(return_value=False),
                get_device_properties=mock.Mock(
                    return_value=SimpleNamespace(multi_processor_count=256)
                ),
                synchronize=mock.Mock(
                    side_effect=AssertionError("unexpected synchronization")
                ),
            ),
        )
        attention = SimpleNamespace(
            get_pa_metadata_info_v1=mock.Mock(return_value=[((2,), "int32")] * 6),
            get_pa_metadata_v1=mock.Mock(),
            _pa_ps_reduce_waves=mock.Mock(return_value=0),
            pa_ps_fwd_asm=mock.Mock(),
            _attention_ps_reduce_asm=mock.Mock(),
        )
        flydsl = SimpleNamespace(
            plan_pa_decode=mock.Mock(return_value=SimpleNamespace(capacity=2)),
            pa_decode=mock.Mock(),
        )
        modules = {
            "torch": torch,
            "aiter.jit.utils.chip_info": SimpleNamespace(
                get_gfx_runtime=lambda: "gfx950"
            ),
            "aiter.ops": SimpleNamespace(attention=attention),
            "aiter.ops.enum": SimpleNamespace(
                QuantType=SimpleNamespace(per_Token="per_token")
            ),
            "aiter.ops.flydsl.pa_decode": flydsl,
        }
        pages_per_row = (length + 15) // 16
        pages = batch * pages_per_row
        arguments = {
            "query": TensorMetadata((batch * 4, 16, 128), "bfloat16"),
            "key_cache": TensorMetadata((pages, 1, 8, 16, 16), "float8_e4m3fn"),
            "value_cache": TensorMetadata((pages, 1, 1, 128, 16), "float8_e4m3fn"),
            "context_lengths": TensorMetadata((batch,), "int32"),
            "block_tables": TensorMetadata((batch, pages_per_row), "int32"),
            "query_length": 4,
            "context_lengths_cpu": [length] * batch,
            "key_scale": TensorMetadata((pages, 1, 16, 1), "float32"),
            "value_scale": TensorMetadata((pages, 1, 16, 1), "float32"),
            "mask_unowned_kv": False,
        }
        if packed:
            arguments.update(
                kv_indptr=TensorMetadata((batch + 1,), "int32"),
                kv_indices=TensorMetadata((pages,), "int32"),
            )
        output = TensorMetadata(arguments["query"].shape, "bfloat16")
        return modules, arguments, output, (mock.MagicMock(), mock.MagicMock())

    def test_prepare_refresh_decode_uses_selected_backend(self):
        op = _load_contract_module()
        cases = (
            (16, 262144, True, "flydsl"),
            (24, 131071, True, "flydsl"),
            (24, 131072, True, "asm"),
            (32, 98303, True, "flydsl"),
            (32, 98304, True, "asm"),
            (24, 196608, False, "flydsl"),
            (32, 131071, False, "flydsl"),
            (32, 131072, False, "asm"),
        )
        for batch, length, packed, expected in cases:
            with self.subTest(batch=batch, length=length, packed=packed):
                modules, arguments, output, csr_kernels = self.fixture(
                    batch, length, packed
                )
                attention = modules["aiter.ops"].attention
                flydsl = modules["aiter.ops.flydsl.pa_decode"]
                with mock.patch.dict(sys.modules, modules), mock.patch.object(
                    op, "_csr_kernels", return_value=csr_kernels
                ):
                    plan = op.prepare_pa_decode_plan(**arguments)
                    self.assertEqual(plan.backend, expected)
                    op.refresh_pa_decode_plan(plan)
                    result = op.pa_decode(
                        output,
                        arguments["query"],
                        arguments["key_cache"],
                        arguments["value_cache"],
                        arguments["context_lengths"],
                        arguments["block_tables"],
                        128**-0.5,
                        4,
                        plan=plan,
                        key_scale=arguments["key_scale"],
                        value_scale=arguments["value_scale"],
                    )
                self.assertIs(result, output)
                if expected == "asm":
                    self.assertEqual(plan.owns_csr, not packed)
                    self.assertEqual(attention.get_pa_metadata_v1.call_count, 2)
                    attention.pa_ps_fwd_asm.assert_called_once()
                    attention._attention_ps_reduce_asm.assert_called_once()
                    flydsl.plan_pa_decode.assert_not_called()
                    flydsl.pa_decode.assert_not_called()
                    for kernel in csr_kernels:
                        self.assertEqual(
                            kernel.__getitem__.return_value.call_count,
                            0 if packed else 2,
                        )
                else:
                    self.assertEqual(flydsl.plan_pa_decode.call_count, 2)
                    self.assertEqual(plan.fly_workgroup_budget, 512)
                    flydsl.pa_decode.assert_called_once()
                    self.assertIs(
                        flydsl.pa_decode.call_args.kwargs["work_plan"], plan.fly_plan
                    )
                    attention.get_pa_metadata_v1.assert_not_called()
                    attention.pa_ps_fwd_asm.assert_not_called()
                    attention._attention_ps_reduce_asm.assert_not_called()

    def test_prepare_defaults_and_missing_hints_keep_flydsl(self):
        op = _load_contract_module()
        for option in ("mask_unowned_kv", "context_lengths_cpu"):
            with self.subTest(defaulted=option):
                modules, arguments, _, _ = self.fixture(32, 131072, True)
                arguments.pop(option)
                with mock.patch.dict(sys.modules, modules):
                    plan = op.prepare_pa_decode_plan(**arguments)
                self.assertEqual(plan.backend, "flydsl")
                modules["aiter.ops"].attention.get_pa_metadata_v1.assert_not_called()


def run_torch(query, keys, values, lengths, softmax_scale):
    """Independent FP32 dense causal attention from dequantized logical KV."""
    import torch

    outputs = []
    for sequence, length in enumerate(lengths):
        q = query[sequence * 4 : (sequence + 1) * 4].float()
        k = keys[sequence][:length]
        v = values[sequence][:length]
        scores = torch.einsum("qhd,nd->qhn", q, k) * softmax_scale
        visible = torch.arange(length, device=query.device)[None, :] <= (
            length - 4 + torch.arange(4, device=query.device)[:, None]
        )
        probabilities = scores.masked_fill(~visible[:, None, :], float("-inf")).softmax(
            -1
        )
        outputs.append(torch.einsum("qhn,nd->qhd", probabilities, v))
    return torch.cat(outputs).to(query.dtype)


def benchmark_pa_decode(batch, context_length):
    import torch

    from aiter.jit.utils.chip_info import get_gfx_runtime
    from aiter.ops.pa_decode import (
        pa_decode,
        prepare_pa_decode_plan,
        refresh_pa_decode_plan,
    )
    from aiter.test_common import checkAllclose, run_perftest

    torch.manual_seed(42)
    device = torch.device("cuda")
    # Uneven boundaries expose CSR packing mistakes and per-query causal masks.
    lengths = [max(4, context_length - i * 3) for i in range(batch)]
    pages_per_row = (max(lengths) + 15) // 16
    pages = batch * pages_per_row
    table = (
        torch.randperm(pages, device=device, dtype=torch.int64)
        .to(torch.int32)
        .view(batch, pages_per_row)
    )
    context = torch.tensor(lengths, dtype=torch.int32, device=device)
    query = torch.randn((batch * 4, 16, 128), dtype=torch.bfloat16, device=device)
    logical_k = torch.randn((pages, 16, 128), dtype=torch.float32, device=device)
    logical_v = torch.randn_like(logical_k)
    ks = logical_k.abs().amax(-1).clamp_min(1e-7) / 448
    vs = logical_v.abs().amax(-1).clamp_min(1e-7) / 448
    qk = (logical_k / ks[..., None]).to(torch.float8_e4m3fn)
    qv = (logical_v / vs[..., None]).to(torch.float8_e4m3fn)
    keys = qk.view(pages, 16, 8, 16).permute(0, 2, 1, 3).contiguous().unsqueeze(1)
    values = qv.transpose(1, 2).contiguous().view(pages, 1, 1, 128, 16)
    key_scale, value_scale = ks[:, None, :, None], vs[:, None, :, None]
    reference_k = [
        (qk[table[i].long()].float() * ks[table[i].long(), :, None]).reshape(-1, 128)
        for i in range(batch)
    ]
    reference_v = [
        (qv[table[i].long()].float() * vs[table[i].long(), :, None]).reshape(-1, 128)
        for i in range(batch)
    ]
    reference = run_torch(query, reference_k, reference_v, lengths, 128**-0.5)
    counts = [(n + 15) // 16 for n in lengths]
    offsets = [0] + list(itertools.accumulate(counts))
    indptr = torch.tensor(offsets, dtype=torch.int32, device=device)
    indices = torch.cat([table[i, :count] for i, count in enumerate(counts)])
    candidates = {}
    for name in ("flydsl", "auto", "asm_packed", "asm_table"):
        output = torch.full_like(query, float("nan"))
        packed = (
            {"kv_indptr": indptr, "kv_indices": indices} if name == "asm_packed" else {}
        )
        plan = prepare_pa_decode_plan(
            query,
            keys,
            values,
            context,
            table,
            4,
            backend="asm" if name.startswith("asm") else name,
            context_lengths_cpu=lengths,
            key_scale=key_scale,
            value_scale=value_scale,
            mask_unowned_kv=False,
            max_context_length=max(lengths),
            **packed,
        )

        def execute(plan=plan, output=output):
            refresh_pa_decode_plan(plan)
            return pa_decode(
                output,
                query,
                keys,
                values,
                context,
                table,
                128**-0.5,
                4,
                plan=plan,
                key_scale=key_scale,
                value_scale=value_scale,
            )

        candidates[name] = execute
    # Logical useful QK+PV work and minimum KV/scale/Q/O traffic, not HBM counters.
    flops = 4 * 16 * 128 * sum(4 * n - 6 for n in lengths)
    logical_bytes = (
        sum(lengths) * (2 * 128 + 8) + 2 * query.numel() * query.element_size()
    )
    result = {
        "gfx": get_gfx_runtime(),
        "boundary": "refresh+decode",
        "logical_bytes": logical_bytes,
    }
    for name, function in candidates.items():
        actual, microseconds = run_perftest(function, use_cuda_event=True)
        assert torch.isfinite(actual).all()
        error = checkAllclose(
            reference.float(), actual.float(), rtol=1e-2, atol=1e-2, msg=name
        )
        assert error == 0, (name, error)
        result[f"{name} us"] = microseconds
        result[f"{name} TFLOPS"] = flops / microseconds / 1e6
        result[f"{name} TB/s"] = logical_bytes / microseconds / 1e6
        result[f"{name} err"] = error
    return result


def main():
    import pandas as pd

    import aiter
    from aiter.jit.utils.chip_info import get_gfx_runtime
    from aiter.test_common import benchmark

    suite = unittest.TestSuite(
        unittest.defaultTestLoader.loadTestsFromTestCase(test_case)
        for test_case in (DispatchContractTests, PreparedDispatchTests)
    )
    if not unittest.TextTestRunner().run(suite).wasSuccessful():
        raise SystemExit(1)
    if get_gfx_runtime() not in ("gfx950",):
        aiter.logger.warning(
            "Q4 ASM/FlyDSL dispatch benchmark requires gfx950; skipping"
        )
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--context-length", type=int, nargs="+", default=[513])
    args = parser.parse_args()
    # Apply the standard benchmark decorator at the GPU entry point, keeping
    # the CPU-only contract suite importable without AITER/torch/ROCm imports.
    measured = benchmark()(benchmark_pa_decode)
    results = [
        measured(b, length)
        for b, length in itertools.product(args.batch, args.context_length)
    ]
    aiter.logger.info(
        "PA dispatch complete-step summary:\n%s",
        pd.DataFrame(results).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
