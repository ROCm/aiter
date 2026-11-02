# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only coverage for OPUS split-barrier cache-policy registrations."""

import ast
import sys
import unittest
from dataclasses import replace
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csrc.opus_gemm import opus_gemm_common as registry


class TestA16W16CachePolicyRegistry(unittest.TestCase):
    def test_gfx950_a16w16_subfamilies_have_distinct_local_ids(self):
        # Read the actual merge inputs so a newly added family is covered without
        # keeping a second, manually maintained list of family names in this test.
        module = ast.parse(Path(registry.__file__).read_text(encoding="utf-8"))
        merges = [
            node.value
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "gfx950_a16w16_kernels_list"
                for target in node.targets
            )
        ]
        self.assertEqual(len(merges), 1)
        merge = merges[0]
        self.assertIsInstance(merge, ast.Dict)
        self.assertTrue(merge.values)
        owners = {}
        for key, value in zip(merge.keys, merge.values):
            self.assertIsNone(key, "Expected named registry family unpacking")
            self.assertIsInstance(value, ast.Name)
            for kid, instance in getattr(registry, value.id).items():
                self.assertNotIn(
                    kid,
                    owners,
                    f"kid {kid} is shared by {owners.get(kid)} and {value.id}",
                )
                owners[kid] = value.id
                self.assertIs(registry.gfx950_a16w16_kernels_list[kid], instance)
        self.assertEqual(set(owners), set(registry.gfx950_a16w16_kernels_list))

    def test_each_cache_policy_survives_registry_merge(self):
        families = (
            registry.a16w16_kernels_list_cpol,
            registry.a16w16_kernels_list_cpol_nooob,
        )
        seen = set()
        for family in families:
            self.assertFalse(seen.intersection(family))
            seen.update(family)
            for kid, instance in family.items():
                with self.subTest(kid=kid):
                    self.assertIs(registry.gfx950_a16w16_kernels_list[kid], instance)

    def test_all_tiles_have_all_cache_policies_with_and_without_oob(self):
        expected = {
            (has_oob, ca, cb)
            for has_oob in (True, False)
            for ca, cb in ((1, 17), (17, 1), (0, 0))
        }
        for base in registry.a16w16_kernels_list.values():
            with self.subTest(tile=(base.B_M, base.B_N, base.B_K)):
                matching = [
                    inst
                    for inst in registry.gfx950_a16w16_kernels_list.values()
                    if inst.kernel_tag == base.kernel_tag
                    and not inst.is_4g_safe
                    and (inst.B_M, inst.B_N, inst.B_K) == (base.B_M, base.B_N, base.B_K)
                    and (inst.cachectl_a, inst.cachectl_b) != (0, 17)
                ]
                self.assertEqual(len(matching), len(expected))
                self.assertEqual(
                    {
                        (inst.has_oob, inst.cachectl_a, inst.cachectl_b)
                        for inst in matching
                    },
                    expected,
                )

    def test_local_variants_keep_their_configuration(self):
        # Local variant ranges preserve the former OOB/cache/4G-safe settings.
        offsets = (
            (0, True, 0, 17, False),
            (46, False, 0, 17, False),
            (97, True, 1, 17, False),
            (103, False, 1, 17, False),
            (109, False, 17, 1, False),
            (115, True, 0, 17, True),
            (125, False, 0, 17, True),
        )
        for base_kid, base in registry.a16w16_kernels_list.items():
            for offset, has_oob, ca, cb, safe in offsets:
                kid = base_kid + offset
                with self.subTest(kid=kid):
                    self.assertEqual(
                        registry.gfx950_a16w16_kernels_list[kid],
                        replace(
                            base,
                            has_oob=has_oob,
                            cachectl_a=ca,
                            cachectl_b=cb,
                            is_4g_safe=safe,
                        ),
                    )

    def test_cache_policy_ids_are_exact_routable(self):
        for family in (
            registry.a16w16_kernels_list_cpol,
            registry.a16w16_kernels_list_cpol_nooob,
        ):
            for kid, instance in family.items():
                with self.subTest(kid=kid):
                    self.assertIn(
                        kid,
                        registry.group_kids("NON_SPLITK_KIDS", "gfx950", cu_num=256),
                    )
                    self.assertIn(
                        kid,
                        registry.group_kids("BIAS_AWARE_KIDS", "gfx950", cu_num=256),
                    )
                    for dtype in ("bf16_t", "fp32_t"):
                        self.assertIs(
                            registry.get_kernel_instance(
                                "gfx950", "a16w16", kid, dtype, cu_num=256
                            ),
                            instance,
                        )
                    self.assertFalse(
                        registry.kernel_needs_external_workspace(
                            "gfx950", "a16w16", kid, cu_num=256
                        )
                    )


if __name__ == "__main__":
    unittest.main()
