# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""
End-to-end mp_tuner behaviour when one candidate of a shape group fails:
only that candidate fails, for untyped and return_status callers alike.

Needs one GPU. An in-process fault is simulated by raising an AcceleratorError
for an illegal memory access, and a worker death by exiting the worker
process, so the pool takes its real fault-and-restart paths without corrupting
the device.

Run: HIP_VISIBLE_DEVICES=0 python3 -m unittest op_tests.tuning_tests.test_mp_tuner_fault -v
"""

import contextlib
import math
import os
import signal
import unittest
from typing import ClassVar

import torch


def _make_inputs(device=None):
    return {"x": torch.ones(1024, device=device)}


def _boom_inputs(device=None):
    raise RuntimeError("boom before launch")


def _copy(x):
    return x.clone()


class AcceleratorError(RuntimeError):
    """Named like torch's, which is how a worker recognises a GPU fault."""


def _fault(x):
    raise AcceleratorError("HIP error: an illegal memory access was encountered")


def _die(x):
    os._exit(1)


SHAPE = ("shape-0",)
NEXT_SHAPE = ("shape-1",)


class _DeadlineExceeded(BaseException):
    """BaseException, so mp_tuner's polling loop cannot swallow it."""


@contextlib.contextmanager
def _deadline(seconds):
    def expire(signum, frame):
        raise _DeadlineExceeded(f"mp_tuner did not return within {seconds}s")

    previous = signal.signal(signal.SIGALRM, expire)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _candidate(name, func, shape=SHAPE, gen_data=_make_inputs):
    # mp_tuner groups a shape's candidates by the first element of info.
    return (
        (shape, name),
        gen_data,
        (),
        func,
        (("x",),),
        {"num_warmup": 1, "num_iters": 3},
        _copy,
        (("x",),),
        {},
        None,
    )


class _FaultCase:
    """Run GROUP (plus NEXT, another shape) through mp_tuner and check that
    only FAILED candidates failed, with what each typed status says."""

    GROUP = ()
    NEXT = ()
    FAILED: ClassVar[dict] = {}
    TIMEOUT = 120
    DEADLINE = 300

    def _run(self, **kwargs):
        from aiter.utility.mp_tuner import mp_tuner

        groups = [group for group in (self.GROUP, self.NEXT) if group]
        with _deadline(self.DEADLINE):
            return mp_tuner(
                [task for group in groups for task in group],
                [(len(group), None) for group in groups],
                1,
                False,
                True,
                timeout=self.TIMEOUT,
                **kwargs,
            )

    def test_untyped_callers_lose_only_the_failed_candidate(self):
        results = self._run()
        names = [name for (_, name), *_ in self.GROUP + self.NEXT]
        self.assertEqual([name for (_, name), *_ in results], names)
        for (_, name), us, _err in results:
            with self.subTest(name=name):
                failed = name in self.FAILED
                self.assertEqual(math.isinf(us), failed, results)
                self.assertTrue(failed or us > 0, results)

    def test_typed_callers_get_a_status_per_candidate(self):
        results = self._run(return_status=True)
        for (_, name), us, _err, status, detail in results:
            with self.subTest(name=name):
                expected_status, detail_part = self.FAILED.get(name, ("ok", ""))
                self.assertEqual(status, expected_status, results)
                self.assertIn(detail_part, detail)
                self.assertEqual(math.isinf(us), name in self.FAILED, results)


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestFaultedShapeGroup(_FaultCase, unittest.TestCase):
    """An in-candidate GPU fault restarts the pool and the rest resume."""

    GROUP = (
        _candidate("fast", _copy),
        _candidate("faulting", _fault),
        _candidate("behind", _copy),
    )
    FAILED: ClassVar[dict] = {"faulting": ("crash", "illegal memory access")}


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestWorkerDeath(_FaultCase, unittest.TestCase):
    """A candidate that kills its worker, with another shape queued behind it.

    The pool replaces the dead worker with a PID the GPU map does not know, so
    the queued shape hits the stale map and restarts the pool. The dying
    candidate must be failed rather than resubmitted, or every restart kills
    the worker again. The timeout is far above the run time, so a pass shows
    the dead worker was detected rather than timed out.
    """

    TIMEOUT = 600
    GROUP = (
        _candidate("fast", _copy),
        _candidate("dying", _die),
        _candidate("behind", _copy),
    )
    NEXT = (_candidate("next", _copy, NEXT_SHAPE),)
    FAILED: ClassVar[dict] = {"dying": ("crash", "exited")}


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestPreLaunchFailure(_FaultCase, unittest.TestCase):
    """A candidate whose inputs fail to build, before worker() runs."""

    GROUP = (
        _candidate("fast", _copy),
        _candidate("prelaunch", _copy, gen_data=_boom_inputs),
        _candidate("behind", _copy),
    )
    FAILED: ClassVar[dict] = {"prelaunch": ("crash", "boom before launch")}


if __name__ == "__main__":
    unittest.main(verbosity=2)
