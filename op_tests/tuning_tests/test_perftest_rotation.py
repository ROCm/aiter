# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Unit tests for the input rotation behind run_perftest.

Run: python3 -m unittest op_tests.tuning_tests.test_perftest_rotation -v
"""

import copy
import types
import unittest
from unittest.mock import patch

import torch

from aiter.test_common import device_memory_profiling, input_copies, perftest
from aiter.utility import measure

MiB = 1 << 20


def _storage(t):
    return t.untyped_storage().data_ptr()


class TestInputCopies(unittest.TestCase):
    def test_sharing_between_args_and_kwargs_survives(self):
        buf = torch.zeros(4)
        args, kwargs = (buf, buf), {"out": buf[:2]}
        copies = input_copies(args, kwargs, 3)
        self.assertEqual(len(copies), 3)
        for a, k in copies[:-1]:
            self.assertIsNot(a[0], buf)
            self.assertIs(a[0], a[1])
            self.assertEqual(_storage(k["out"]), _storage(a[0]))
            k["out"].fill_(7)
            self.assertEqual(a[0].tolist(), [7, 7, 0, 0])
        self.assertEqual(buf.tolist(), [0, 0, 0, 0])

    def test_callers_inputs_are_the_last_slot(self):
        args, kwargs = (torch.zeros(2),), {"scale": 2.0}
        copies = input_copies(args, kwargs, 4)
        self.assertIs(copies[-1][0], args)
        self.assertIs(copies[-1][1], kwargs)
        self.assertEqual(input_copies(args, kwargs, 1), [(args, kwargs)])

    def test_copies_come_from_pristine_not_the_caller(self):
        out = torch.full((4,), float("nan"))
        args, kwargs = (torch.ones(4),), {"out": out}
        pristine = copy.deepcopy((args, kwargs))
        out.fill_(1)
        copies = input_copies(args, kwargs, 4, pristine)
        self.assertEqual(len(copies), 4)
        self.assertIs(copies[0], pristine)
        for a, k in copies[:-1]:
            self.assertIsNot(k["out"], out)
            self.assertTrue(torch.isnan(k["out"]).all())
        self.assertIs(copies[-1][1], kwargs)
        self.assertEqual(input_copies(args, kwargs, 1, pristine), [(args, kwargs)])


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestInputSize(unittest.TestCase):
    def test_kwargs_count_and_a_shared_tensor_counts_once(self):
        a = torch.ones(MiB // 4, device="cuda")
        b = torch.ones(MiB // 4, device="cuda")

        def func(a, b):
            return a + b

        for args, kwargs in (((a, b), {}), ((a,), {"b": b}), ((), {"a": a, "b": b})):
            with self.subTest(kwargs=sorted(kwargs)):
                _, size, _, _ = device_memory_profiling(func, *args, **kwargs)
                self.assertEqual(size, 2 * MiB + 1)
        _, size, _, _ = device_memory_profiling(func, a, b=a)
        self.assertEqual(size, MiB + 1)


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestPerftestCopiesStartAsPassed(unittest.TestCase):
    def test_no_copy_sees_what_the_sizing_call_wrote(self):
        x = torch.ones(1024, device="cuda")
        out = torch.full_like(x, float("nan"))
        seen = {}

        def func(x, out):
            if out.data_ptr() not in seen:
                seen[out.data_ptr()] = bool(torch.isnan(out).all())
            out.copy_(x)
            return out

        perftest(num_iters=4, num_warmup=1)(func)(x, out)
        copies = {p: nan for p, nan in seen.items() if p != out.data_ptr()}
        self.assertEqual(len(copies), 3)
        self.assertTrue(all(copies.values()), copies)


def _cache(level, kib, instances=1):
    return {"cache_level": level, "cache_size": kib, "num_cache_instance": instances}


# amdsmi_get_gpu_cache_info()["cache"] as read on an MI300A (L1 entries trimmed).
MI300A_CACHES = [
    _cache(1, 32, 228),
    _cache(1, 64, 76),
    _cache(2, 4096),
    _cache(3, 262144),
]
# The same call on an integrated gfx1036, which has no level 3.
GFX1036_CACHES = [_cache(1, 16, 2), _cache(1, 128), _cache(2, 256)]


class TestLlcFromCacheInfo(unittest.TestCase):
    def test_level_3_when_present(self):
        self.assertEqual(measure.llc_from_cache_info(MI300A_CACHES), 256 * MiB)

    def test_highest_level_when_no_level_3(self):
        self.assertEqual(measure.llc_from_cache_info(GFX1036_CACHES), 256 * 1024)

    def test_instances_add_up(self):
        caches = [_cache(1, 16), _cache(2, 8192, instances=2)]
        self.assertEqual(measure.llc_from_cache_info(caches), 16 * MiB)


class TestLlcFallback(unittest.TestCase):
    def setUp(self):
        measure.llc_bytes.cache_clear()
        self.addCleanup(measure.llc_bytes.cache_clear)

    def _llc_without_amdsmi(self, arch_name):
        props = types.SimpleNamespace(gcnArchName=arch_name)
        with (
            patch.object(measure, "_amdsmi_llc", side_effect=ImportError("no amdsmi")),
            patch.object(torch.cuda, "get_device_properties", return_value=props),
            self.assertLogs(measure.logger, "WARNING") as logs,
        ):
            llc = measure.llc_bytes(0)
        return llc, logs.output

    def test_known_arch(self):
        for arch, want in (
            ("gfx90a:sramecc+:xnack-", 8 * MiB),
            ("gfx942:sramecc+:xnack-", 256 * MiB),
            ("gfx950:sramecc+:xnack-", 256 * MiB),
        ):
            with self.subTest(arch=arch):
                measure.llc_bytes.cache_clear()
                llc, logs = self._llc_without_amdsmi(arch)
                self.assertEqual(llc, want)
                self.assertIn("no amdsmi", logs[0])

    def test_unknown_arch_assumes_256_mib(self):
        llc, logs = self._llc_without_amdsmi("gfx1250")
        self.assertEqual(llc, 256 * MiB)
        self.assertIn("gfx1250", logs[0])

    def test_answer_is_cached_and_warns_once(self):
        props = types.SimpleNamespace(gcnArchName="gfx1250")
        with (
            patch.object(
                measure, "_amdsmi_llc", side_effect=RuntimeError("no handle")
            ) as smi,
            patch.object(torch.cuda, "get_device_properties", return_value=props),
            self.assertLogs(measure.logger, "WARNING") as logs,
        ):
            measure.llc_bytes(0)
            measure.llc_bytes(0)
        self.assertEqual(smi.call_count, 1)
        self.assertEqual(len(logs.output), 1)


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestLlcOnDevice(unittest.TestCase):
    def test_amdsmi_answers_on_this_device(self):
        measure.llc_bytes.cache_clear()
        device = torch.cuda.current_device()
        llc = measure.llc_bytes(device)
        self.assertEqual(llc, measure._amdsmi_llc(device))
        self.assertGreaterEqual(
            llc, torch.cuda.get_device_properties(device).L2_cache_size
        )


if __name__ == "__main__":
    unittest.main()
