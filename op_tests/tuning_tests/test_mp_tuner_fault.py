# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""
End-to-end mp_tuner behaviour when one candidate of a shape group faults.

Needs one GPU. The fault is simulated by raising the error HIP reports for an
illegal memory access, so the pool takes its real fault-and-restart path
without corrupting the device.

Run: HIP_VISIBLE_DEVICES=0 python3 -m unittest op_tests.tuning_tests.test_mp_tuner_fault -v
"""

import math
import unittest

import triton  # noqa: F401  # ROCm environments may require Triton before torch.

# isort: split
import torch


def _make_inputs(device=None):
    return {"x": torch.ones(1024, device=device)}


def _copy(x):
    return x.clone()


def _fault(x):
    raise RuntimeError("HIP error: an illegal memory access was encountered")


SHAPE = ("shape-0",)


def _candidate(name, func):
    # mp_tuner groups a shape's candidates by the first element of info.
    return (
        (SHAPE, name),
        _make_inputs,
        (),
        func,
        (("x",),),
        {"num_warmup": 1, "num_iters": 3},
        _copy,
        (("x",),),
        {},
        None,
    )


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestFaultedShapeGroup(unittest.TestCase):
    GROUP = (_candidate("fast", _copy), _candidate("faulting", _fault))

    def _run(self, **kwargs):
        from aiter.utility.mp_tuner import mp_tuner

        return mp_tuner(
            list(self.GROUP),
            [(len(self.GROUP), None)],
            1,
            False,
            True,
            timeout=120,
            **kwargs,
        )

    def test_legacy_callers_get_the_whole_group_failed(self):
        # The positional form tuners without typed statuses use today.
        results = self._run()
        self.assertEqual([name for (_, name), *_ in results], ["fast", "faulting"])
        self.assertTrue(all(math.isinf(us) for _, us, _ in results), results)

    def test_typed_callers_keep_the_candidate_measured_before_the_fault(self):
        results = self._run(return_status=True)
        by_name = {name: rest for (_, name), *rest in results}
        us, _err, status, _detail = by_name["fast"]
        self.assertEqual(status, "ok")
        self.assertTrue(math.isfinite(us) and us > 0, us)
        us, _err, status, _detail = by_name["faulting"]
        self.assertEqual(status, "crash")
        self.assertTrue(math.isinf(us))


if __name__ == "__main__":
    unittest.main(verbosity=2)
