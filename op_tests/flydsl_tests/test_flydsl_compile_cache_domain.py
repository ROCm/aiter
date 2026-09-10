"""The compile caches must cover the whole reachable key domain.

The partial cache is keyed on (ng, inner_iter, split_major), not ng alone, so
sizing it to the ng range under-counts: this test derives the reachable set from
the policy functions instead of hard-coding a bound, which is what let an
undersized cache thrash unnoticed before.

No GPU needed -- this reads the declared lru_cache capacity and pure integer
policy.
"""

import unittest

import aiter.ops.flydsl.kernels.sparse_mla_decode as D
import aiter.ops.flydsl.mla_reduce_kernels as R
from aiter.ops.flydsl.mla_reduce_kernels import _use_fine_decode_combine
from aiter.ops.flydsl.sparse_mla_decode_kernels import (
    _partial_groups,
    _pick_inner_iter,
    _use_split_major,
)

SEQ_MAX = 96
NG_MAX = 33
NI_MAX = 33
# Both MI355X (256) and MI300X (304); the smaller CU count admits more keys.
CU_COUNTS = (256, 304)


def _reachable_partial_keys() -> set:
    keys = set()
    for cu in CU_COUNTS:
        for seq in range(1, SEQ_MAX + 1):
            for ng in range(1, NG_MAX + 1):
                inner_iter = _pick_inner_iter(seq, ng)
                groups = _partial_groups(ng, inner_iter)
                keys.add((ng, inner_iter, _use_split_major(seq, groups, cu)))
    return keys


class TestCompileCacheDomain(unittest.TestCase):
    def test_partial_cache_covers_its_key_domain(self):
        needed = len(_reachable_partial_keys())
        info = D.compile_sparse_mla_partial.cache_info()
        self.assertGreaterEqual(
            info.maxsize,
            needed,
            f"partial cache holds {info.maxsize} < {needed} reachable "
            "(ng, inner_iter, split_major) keys",
        )

    def test_combine_cache_covers_its_key_domain(self):
        # Keyed on (ni, fine), so the ni range alone under-counts it by 2x.
        keys = {
            (ni, _use_fine_decode_combine(seq, cu))
            for cu in CU_COUNTS
            for seq in range(1, SEQ_MAX + 1)
            for ni in range(1, NI_MAX + 1)
        }
        info = R._compile_sparse_decode_direct_combine.cache_info()
        self.assertGreaterEqual(
            info.maxsize,
            len(keys),
            f"combine cache holds {info.maxsize} < {len(keys)} reachable "
            "(ni, fine) keys",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
