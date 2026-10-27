"""The FlyDSL AOT replay answers device queries for the target it compiles for."""

import importlib.util
import pathlib
import sys
import unittest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@unittest.skipUnless(importlib.util.find_spec("flydsl"), "FlyDSL is not installed")
class DryRunXcdCountTest(unittest.TestCase):
    def test_xcd_count_follows_the_replayed_target(self):
        from aiter.aot.flydsl.common import target_num_xcds
        from aiter.aot.flydsl.dry_run import dry_run
        from aiter.jit.utils import chip_info
        from aiter.ops.flydsl import moe_kernels

        for arch, cu_num in (("gfx950", 128), ("gfx950", 256), ("gfx942", 228)):
            with self.subTest(arch=arch, cu_num=cu_num), dry_run(arch, cu_num):
                expected = target_num_xcds(arch, cu_num)
                self.assertEqual(chip_info.get_num_xcds(default=None), expected)
                # Kernel modules import get_num_xcds by name.
                self.assertEqual(moe_kernels.get_num_xcds(default=None), expected)


if __name__ == "__main__":
    unittest.main()
