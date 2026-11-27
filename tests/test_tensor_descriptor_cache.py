# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU control-flow tests; no ROCm import or native ABI claim.

Load the actual cache and decorator definitions without core.py's toolchain
imports. A metadata-only tensor and FFI stand-in expose conversion and ownership.
Run with: python -m unittest discover -s tests -p test_tensor_descriptor_cache.py
"""

import ast
import functools
import gc
import threading
import types
import unittest
import weakref
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import patch


class Tensor:
    def __init__(self, ptr=100, shape=(4, 8), stride=(8, 1), dtype="bf16", device=0):
        self.ptr, self.shape, self.strides = ptr, shape, stride
        self.dtype, self.device = dtype, device

    def data_ptr(self):
        return self.ptr

    def stride(self):
        return self.strides

    def get_device(self):
        return self.device


class Parameter(Tensor):
    pass


def descriptor(t):
    return types.SimpleNamespace(
        ptr=t.ptr, shape=t.shape, strides=t.strides, dtype=t.dtype, device=t.device
    )


def definitions():
    source = Path(__file__).resolve().parents[1] / "aiter/jit/core.py"
    tree = ast.parse(source.read_text())
    names = {"_develop_arg", "_TensorDescriptorCache", "compile_ops"}
    tree.body = [n for n in tree.body if getattr(n, "name", None) in names]
    namespace = {
        "functools": functools,
        "threading": threading,
        "OrderedDict": OrderedDict,
        "Callable": Callable,
        "Any": Any,
        "types": types,
    }
    # Execute only the selected definitions from this repository's core module.
    exec(compile(tree, str(source), "exec"), namespace)  # noqa: S102
    return namespace


class DescriptorCacheTest(unittest.TestCase):
    def setUp(self):
        self.ns = definitions()
        self.cache = self.ns["_TensorDescriptorCache"]((Tensor, Parameter), maxsize=2)
        self.converted = []

    def convert(self, tensor):
        self.converted.append(tensor.ptr)
        return descriptor(tensor)

    def handle(self, tensor):
        return self.cache.convert(tensor, self.convert, Tensor)

    def test_descriptor_fields_and_pointer_reuse(self):
        first = self.handle(Tensor())
        self.assertIs(self.handle(Tensor()), first)
        for change in (
            {"ptr": 101},
            {"shape": (2, 16)},
            {"stride": (1, 4)},
            {"dtype": "fp16"},
            {"device": 1},
        ):
            with self.subTest(change=change):
                first = self.handle(Tensor())
                other = Tensor(**change)
                result = self.handle(other)
                self.assertIsNot(result, first)
                self.assertEqual(vars(result), vars(descriptor(other)))

    def test_values_and_tensor_lifetime(self):
        weight = Parameter()
        weight.values = [1, 2]
        ref = weakref.ref(weight)
        first = self.handle(weight)
        weight.values[:] = [3, 4]
        self.assertIs(first, self.handle(weight))
        del weight
        gc.collect()
        self.assertIsNone(ref())
        self.assertIs(first, self.handle(Tensor()))
        self.assertEqual(len(self.converted), 1)

    def test_lru_eviction_retains_callers_descriptor(self):
        first = self.handle(Tensor(ptr=1))
        second = self.handle(Tensor(ptr=2))
        self.assertIs(first, self.handle(Tensor(ptr=1)))
        self.handle(Tensor(ptr=3))
        self.assertEqual(len(self.cache._handles), 2)
        self.assertIsNot(second, self.handle(Tensor(ptr=2)))
        self.assertEqual((first.ptr, second.ptr), (1, 2))

    def test_subclass_optional_and_list_keep_original_conversion(self):
        class CustomTensor(Tensor):
            pass

        custom = CustomTensor()
        self.assertIsNot(self.handle(custom), self.handle(custom))
        self.assertIsNone(self.handle(None))
        self.assertEqual(self.handle(7), 7)
        self.assertEqual(self.handle([custom, None])[0].ptr, custom.ptr)
        self.assertEqual(len(self.cache._handles), 0)

    def test_failed_conversion_is_not_cached(self):
        def fail(tensor):
            raise ValueError("unsupported descriptor")

        with self.assertRaisesRegex(ValueError, "unsupported"):
            self.cache.convert(Tensor(), fail, Tensor)
        self.assertEqual(len(self.cache._handles), 0)
        self.assertEqual(self.handle(Tensor()).ptr, 100)

    def test_concurrent_miss_conversion_is_outside_lock(self):
        barrier = threading.Barrier(2)

        def convert(tensor):
            barrier.wait(timeout=5)
            return descriptor(tensor)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(self.cache.convert, Tensor(), convert, Tensor)
                for _ in range(2)
            ]
            first, second = [f.result(timeout=10) for f in futures]
        self.assertIs(first, second)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda p: self.handle(Tensor(ptr=p)), range(40)))
        self.assertEqual([r.ptr for r in results], list(range(40)))
        self.assertLessEqual(len(self.cache._handles), 2)


class DecoratorConversionTest(unittest.TestCase):
    def run_calls(self, opt_in):
        ns = definitions()
        events, converted, native_calls = [], [], []
        module = types.ModuleType("fake_native")
        module.__file__ = "fake_native.so"
        module.op = lambda *a, **k: native_calls.append((a, k))
        module._set_current_hip_stream = lambda stream: events.append(
            ("stream", stream)
        )

        def convert(tensor):
            converted.append(tensor.ptr)
            return descriptor(tensor)

        ns.update(
            AITER_REBUILD=0,
            AITER_LOG_MORE=2,
            __name__="aiter.jit.cache_test",
            __package__="aiter.jit",
            __mds={"module_aiter_core": types.SimpleNamespace(__file__="core.so")},
            get_module=lambda name: module,
            torch_compile_guard=lambda **kw: lambda fn: fn,
            _raise_on_pybind11_abi_split=lambda *args: events.append("abi"),
            _pybind_develop_hooks=lambda: (
                convert,
                Tensor,
                lambda d: 17 + d,
                lambda: 2,
            ),
        )
        torch = types.ModuleType("torch")
        torch.Tensor, torch.nn = Tensor, types.SimpleNamespace(Parameter=Parameter)
        logging = types.ModuleType("aiter.test_common")
        logging.log_args = lambda *a, **k: events.append("log")

        def op(mat1, mat2, result):
            pass

        with patch.dict("sys.modules", {"torch": torch, "aiter.test_common": logging}):
            wrapped = ns["compile_ops"](
                "fake_native", develop=True, tensor_cache_args=opt_in
            )(op)
            # Signature checking needs a real pybind module, so mark the stand-in
            # checked. Logging, ABI guard and stream assignment remain live.
            op.arg_checked = True
            args = (Tensor(ptr=1), Tensor(ptr=2), Tensor(ptr=3))
            wrapped(*args)
            wrapped(mat1=args[0], mat2=args[1], result=args[2])
        self.assertEqual(events, ["log", "abi", ("stream", 19)] * 2)
        self.assertEqual(len(native_calls), 2)
        self.assertEqual(native_calls[0][0][1].ptr, native_calls[1][1]["mat2"].ptr)
        return converted

    def test_only_selected_argument_reused_across_positional_and_keyword_calls(self):
        self.assertEqual(self.run_calls(("mat2",)), [1, 2, 3, 1, 3])

    def test_default_conversion_unchanged(self):
        self.assertEqual(self.run_calls(()), [1, 2, 3, 1, 2, 3])

    def test_opt_in_requires_develop_pybind(self):
        for options in ({}, {"develop": True, "ffi_type": "ctypes"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                definitions()["compile_ops"](
                    "unused", tensor_cache_args=("x",), **options
                )


if __name__ == "__main__":
    unittest.main()
