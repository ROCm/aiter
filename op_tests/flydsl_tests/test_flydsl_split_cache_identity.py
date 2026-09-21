"""A recycled pointer must not make the cache return a stale value.

_resolve_actual_max_splits keys on data_ptr, and the allocator hands a freed
address to the next tensor. Under the old implementation a new CSR landing on an
old address inherited someone else's split bound -- and that bound is exactly the
value that cannot be re-derived during graph capture, so it has to be trusted.
"""

import unittest
from unittest import mock

import torch

import aiter.ops.flydsl.mla_reduce_kernels as R


class TestSplitCacheIdentity(unittest.TestCase):
    def setUp(self):
        R._ACTUAL_MAX_SPLITS_CACHE.clear()

    def test_hit_while_buffer_alive(self):
        t = torch.tensor([0, 3, 7], dtype=torch.int32, device="cuda")
        with mock.patch.object(R, "derive_actual_max_splits", lambda x: 4):
            self.assertEqual(R._resolve_actual_max_splits(t), 4)
        with mock.patch.object(torch.cuda, "is_current_stream_capturing", lambda: True):
            self.assertEqual(R._resolve_actual_max_splits(t), 4)

    def test_recycled_pointer_is_a_miss_not_a_stale_hit(self):
        t = torch.tensor([0, 3, 7], dtype=torch.int32, device="cuda")
        key = (t.data_ptr(), t.numel())
        with mock.patch.object(R, "derive_actual_max_splits", lambda x: 4):
            R._resolve_actual_max_splits(t)
        self.assertIn(key, R._ACTUAL_MAX_SPLITS_CACHE)

        del t  # the address returns to the allocator; the entry is now stale
        # A query that lands on the same key: during capture this must miss,
        # not return 4
        fake = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
        with mock.patch.object(fake, "data_ptr", lambda: key[0]), mock.patch.object(
            torch.cuda, "is_current_stream_capturing", lambda: True
        ):
            got = R._resolve_actual_max_splits(fake)
        self.assertIsNone(got, "a recycled pointer must miss")
        self.assertNotIn(
            key, R._ACTUAL_MAX_SPLITS_CACHE, "the stale entry must be dropped"
        )

    def test_capture_miss_returns_none(self):
        t = torch.tensor([0, 3], dtype=torch.int32, device="cuda")
        with mock.patch.object(torch.cuda, "is_current_stream_capturing", lambda: True):
            self.assertIsNone(R._resolve_actual_max_splits(t))


if __name__ == "__main__":
    unittest.main(verbosity=2)
