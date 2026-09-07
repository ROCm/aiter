"""The compile caches must cover the whole ni/ng domain.

maxsize used to be 16 and 32 while both domains are 1..33, so once a process had
seen more distinct split counts than the cache holds it started thrashing: about
20-31 ms of pure host stall on every re-entry.

No GPU needed -- this only reads the declared lru_cache capacity.
"""

import unittest

import aiter.ops.flydsl.kernels.sparse_mla_decode as D
import aiter.ops.flydsl.mla_reduce_kernels as R

# The upper bounds the two guards hard-code
NI_MAX = 33
NG_MAX = 33


class TestCompileCacheDomain(unittest.TestCase):
    def test_combine_cache_covers_ni_domain(self):
        info = R._compile_sparse_decode_direct_combine.cache_info()
        self.assertGreaterEqual(
            info.maxsize,
            NI_MAX,
            f"combine cache holds {info.maxsize} < ni domain {NI_MAX}",
        )

    def test_partial_cache_covers_ng_domain(self):
        info = D.compile_sparse_mla_partial.cache_info()
        self.assertGreaterEqual(
            info.maxsize,
            NG_MAX,
            f"partial cache holds {info.maxsize} < ng domain {NG_MAX}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
