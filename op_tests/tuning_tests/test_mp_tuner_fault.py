# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""
End-to-end mp_tuner behaviour when one candidate of a shape group faults.

Needs one GPU. An in-process fault is simulated by raising the error HIP
reports for an illegal memory access, and a worker death by exiting the worker
process, so the pool takes its real fault-and-restart paths without corrupting
the device.

Run: HIP_VISIBLE_DEVICES=0 python3 -m unittest op_tests.tuning_tests.test_mp_tuner_fault -v
"""

import contextlib
import math
import os
import signal
import unittest

import torch


def _make_inputs(device=None):
    return {"x": torch.ones(1024, device=device)}


def _boom_inputs(device=None):
    raise RuntimeError("boom before launch")


def _copy(x):
    return x.clone()


def _fault(x):
    raise RuntimeError("HIP error: an illegal memory access was encountered")


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


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestFaultedShapeGroup(unittest.TestCase):
    GROUP = (
        _candidate("fast", _copy),
        _candidate("faulting", _fault),
        _candidate("behind", _copy),
    )

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
        self.assertEqual(
            [name for (_, name), *_ in results], ["fast", "faulting", "behind"]
        )
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
        self.assertEqual(by_name["behind"][2], "not_run", results)


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestWorkerDeath(unittest.TestCase):
    """A candidate that kills its worker, with another shape queued behind it.

    The pool replaces the dead worker with a PID the GPU map does not know, so
    the queued shape hits the stale map and restarts the pool. The dead
    shape must be failed rather than resubmitted, or every restart kills the
    worker again. The timeout is far above the run time, so a pass shows the
    dead worker was detected rather than timed out.
    """

    TIMEOUT = 600
    DEADLINE = 300
    DYING = (
        _candidate("fast", _copy),
        _candidate("dying", _die),
        _candidate("behind", _copy),
    )
    NEXT = (_candidate("next", _copy, NEXT_SHAPE),)

    def _run(self, **kwargs):
        from aiter.utility.mp_tuner import mp_tuner

        with _deadline(self.DEADLINE):
            return mp_tuner(
                list(self.DYING + self.NEXT),
                [(len(self.DYING), None), (len(self.NEXT), None)],
                1,
                False,
                True,
                timeout=self.TIMEOUT,
                **kwargs,
            )

    def test_legacy_callers_lose_the_dead_shape_only(self):
        results = self._run()
        by_name = {name: us for (_, name), us, _ in results}
        self.assertEqual(list(by_name), ["fast", "dying", "behind", "next"])
        for name in ("fast", "dying", "behind"):
            self.assertTrue(math.isinf(by_name[name]), results)
        self.assertTrue(math.isfinite(by_name["next"]) and by_name["next"] > 0)

    def test_typed_callers_keep_what_the_dead_worker_measured(self):
        results = self._run(return_status=True)
        by_name = {name: rest for (_, name), *rest in results}
        self.assertEqual(by_name["fast"][2], "ok", results)
        us, _err, status, detail = by_name["dying"]
        self.assertEqual(status, "crash")
        self.assertIn("exited", detail)
        self.assertTrue(math.isinf(us))
        self.assertEqual(by_name["behind"][2], "not_run", results)
        us, _err, status, _detail = by_name["next"]
        self.assertEqual(status, "ok")
        self.assertTrue(math.isfinite(us) and us > 0, us)


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestPreLaunchAbort(unittest.TestCase):
    """A later candidate raises in gen_data, before worker() runs.

    That is not an accelerator fault, so work_group's except swallows it and
    returns a list. The parent treats that list as a finished group. Typed
    callers must still keep the candidate already measured; the one that
    aborted is crash; the one behind it is not_run.
    """

    GROUP = (
        _candidate("fast", _copy),
        _candidate("prelaunch", _copy, gen_data=_boom_inputs),
        _candidate("behind", _copy),
    )

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
        results = self._run()
        self.assertEqual(
            [name for (_, name), *_ in results], ["fast", "prelaunch", "behind"]
        )
        self.assertTrue(all(math.isinf(us) for _, us, _ in results), results)

    def test_typed_callers_keep_the_candidate_measured_before_the_abort(self):
        published = []
        results = self._run(return_status=True, result_callback=published.append)
        by_name = {name: rest for (_, name), *rest in results}
        us, _err, status, _detail = by_name["fast"]
        self.assertEqual(status, "ok", results)
        self.assertTrue(math.isfinite(us) and us > 0, us)
        us, _err, status, detail = by_name["prelaunch"]
        self.assertEqual(status, "crash", results)
        self.assertIn("boom before launch", detail)
        self.assertTrue(math.isinf(us))
        self.assertEqual(by_name["behind"][2], "not_run", results)
        # A checkpointing caller sees each candidate that ran exactly once, and
        # never the one behind the abort, so a resume retries it.
        self.assertEqual(
            [(name, status) for (_, name), _us, _err, status, _ in published],
            [("fast", "ok"), ("prelaunch", "crash")],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
