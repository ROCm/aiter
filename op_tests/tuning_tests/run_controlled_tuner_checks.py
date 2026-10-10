# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Run existing infrastructure checks after offline toolchain import.

Use a real file entry point because the existing multiprocessing checks spawn
CPU-only workers and cannot re-import a stdin main module.
"""

import unittest

from op_tests.tuning_tests._controlled_tuner_import import import_moe_tuner

import_moe_tuner()


def main():
    suite = unittest.TestLoader().loadTestsFromNames(
        [
            "op_tests.tuning_tests.test_tuner_infra",
            "op_tests.tuning_tests.test_compare_logic",
            "op_tests.tuning_tests.test_mp_tuner_logic",
        ]
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return int(not result.wasSuccessful())


if __name__ == "__main__":
    raise SystemExit(main())
