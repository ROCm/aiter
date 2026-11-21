# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The task record of test_fmoe_candidate_snapshot.py, without a GPU or torch.

Two tasks that differ in what mp_tuner runs or checks must give different
records, and a function that only moved must match once MOVED renames it.
"""

import functools
import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from op_tests.tuning_tests import test_fmoe_candidate_snapshot as snapshot


def generate_data(*args):
    pass


def run_stage1(*args, **kwargs):
    pass


def run_stage1_other(*args, **kwargs):
    pass


def reference_stage1(*args, **kwargs):
    pass


def cosine_compare(*args):
    pass


def limit_cosine_compare(*args):
    pass


def guarded(op):
    """Same attributes as a torch_compile_guard wrapper."""

    def wrapper_custom(*args, **kwargs):
        return op(*args, **kwargs)

    wrapper_custom.__wrapped__ = op
    return wrapper_custom


class Runner:
    def run(self, *args):
        pass


def task(func=run_stage1, rtol=0.01, compare=cosine_compare, kwargs=None, shape="a"):
    """Same layout as a CK stage1 task."""
    return (
        ((shape,), "stage1", "kernel_a", 32, 0, 0, False),
        generate_data,
        (1, 4096, 512, 8, 2, "ActivationType.Silu", 32, 1),
        func,
        (["a1_qt", "w1_qt_shffle_ck", "sorted_ids"], "torch.bfloat16", 2),
        {"use_nt": False} if kwargs is None else kwargs,
        reference_stage1,
        (["a1_qt", "w1_qt", "topk_ids"], "torch.bfloat16", 32),
        {},
        None,
        rtol,
        0.01,
        compare,
    )


def record(t):
    """The record as the test reads it back from the snapshot file."""
    return json.loads(json.dumps(snapshot._record(t)))


def name(fn):
    return f"{__name__}.{fn.__qualname__}"


class TestFmoeTaskRecord(unittest.TestCase):
    def test_shape_is_left_out(self):
        self.assertEqual(record(task(shape="a")), record(task(shape="b")))

    def test_changes_mp_tuner_sees_change_the_record(self):
        base = record(task())
        cases = {
            "rtol": task(rtol=0.05),
            "compare function": task(compare=limit_cosine_compare),
            "no compare function": task(compare=None),
            "kernel function": task(func=run_stage1_other),
            "kwargs": task(kwargs={"use_nt": True}),
        }
        for case, changed in cases.items():
            with self.subTest(case=case):
                self.assertNotEqual(record(changed), base)

    def test_kwargs_order_does_not_change_the_record(self):
        self.assertEqual(
            record(task(kwargs={"use_nt": False, "swiglu_limit": 7.0})),
            record(task(kwargs={"swiglu_limit": 7.0, "use_nt": False})),
        )

    def test_functions_are_recorded_by_qualified_name(self):
        rec = record(task())
        self.assertEqual(rec[1], {"function": name(generate_data)})
        self.assertEqual(rec[3], {"function": name(run_stage1)})
        self.assertEqual(rec[6], {"function": name(reference_stage1)})
        self.assertEqual(rec[-1], {"function": name(cosine_compare)})

    def test_guarded_op_is_recorded_by_the_op_it_wraps(self):
        self.assertEqual(
            record(task(func=guarded(run_stage1))), record(task(func=run_stage1))
        )
        self.assertNotEqual(
            record(task(func=guarded(run_stage1))),
            record(task(func=guarded(run_stage1_other))),
        )

    def test_bound_method_is_recorded_by_its_function(self):
        rec = record(task(func=Runner().run))
        self.assertEqual(rec[3], {"function": name(Runner.run)})

    def test_partial_records_its_arguments(self):
        one = functools.partial(cosine_compare, threshold=1)
        two = functools.partial(cosine_compare, threshold=2)
        self.assertNotEqual(record(task(compare=one)), record(task(compare=two)))
        self.assertEqual(
            record(task(compare=one))[-1],
            {"partial": name(cosine_compare), "args": [], "keywords": {"threshold": 1}},
        )

    def test_object_without_a_stable_form_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "no stable record for object"):
            snapshot._record(task(kwargs={"cache": object()}))

    def test_moved_function_matches_once_renamed(self):
        old = record(task(func=run_stage1))
        new = record(task(func=run_stage1_other))
        self.assertNotEqual(old, new)
        moved = {name(run_stage1): name(run_stage1_other)}
        self.assertEqual(snapshot._renamed(old, moved), new)
        self.assertIn(name(run_stage1), set(snapshot._function_names(old)))
        self.assertNotIn(
            name(run_stage1),
            set(snapshot._function_names(snapshot._renamed(old, moved))),
        )

    def test_renaming_does_not_hide_other_changes(self):
        old = record(task(func=run_stage1, rtol=0.01))
        new = record(task(func=run_stage1_other, rtol=0.05))
        moved = {name(run_stage1): name(run_stage1_other)}
        self.assertNotEqual(snapshot._renamed(old, moved), new)


if __name__ == "__main__":
    unittest.main()
