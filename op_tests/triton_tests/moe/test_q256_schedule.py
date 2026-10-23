# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Independent CPU oracle and exact GPU checks for the ABI128 job map."""

import importlib.util
import random
import unittest


def make_inputs(boundary=512, tokens=8192, suffix=32, preceding_shared=False):
    """Padded 256-row tiles with varied fragment occupancy and tagged IDs."""
    physical_tiles = boundary + suffix
    ids = [tokens] * (physical_tiles * 256)
    experts = [tile % 512 for tile in range(boundary)] + [512] * suffix
    if preceding_shared and boundary:
        experts[boundary - 1] = 512
    for tile in range(boundary):
        for fragment in range(1 + tile % 8):
            for row in range(32):
                ids[tile * 256 + fragment * 32 + row] = (3 << 24) | row
    return ids, experts, [len(ids), tokens]


def reference_job_map(sorted_ids, sorted_experts, counts):
    """Plan tasks as Python records, without packed keys or a sorting network.

    Each XCD receives every eighth logical job. Routed jobs rank by occupied
    32-row fragments, then ordinal; shared jobs follow in ordinal order. The
    ranked queue is dealt to 32 owners in alternating forward/reverse visits.
    """
    physical_tiles = counts[0] // 256
    boundary = physical_tiles - 32
    qualifies = (
        counts[1] == 8192
        and boundary >= 1
        and sorted_experts[boundary - 1] != 512
        and sorted_experts[boundary:physical_tiles] == [512] * 32
    )
    jobs = boundary + 64 if qualifies else physical_tiles
    if not qualifies or jobs > 1024:
        return [slot if slot < jobs else -1 for slot in range(1024)]

    queues = [[] for _ in range(8)]
    for ordinal in range(jobs):
        if ordinal < boundary:
            tile = ordinal
            if ordinal < boundary // 64 * 64:
                group, cell = divmod(ordinal, 64)
                row, column = divmod(cell, 8)
                tile = group * 64 + column * 8 + row
            if boundary % 256 and tile < 256:
                tile ^= 1
            fragments = 1 + sum(
                (sorted_ids[tile * 256 + fragment * 32] & 0xFFFFFF) < counts[1]
                for fragment in range(1, 8)
            )
            priority = (0, -fragments, ordinal)
        else:
            priority = (1, 0, ordinal)
        queues[ordinal % 8].append((priority, ordinal))

    result = [-1] * 1024
    for xcd, queue in enumerate(queues):
        for rank, (_, ordinal) in enumerate(sorted(queue)):
            visit, owner = divmod(rank, 32)
            if visit % 2:
                owner = 31 - owner
            result[visit * 256 + owner * 8 + xcd] = ordinal
    return result


class TestCPUReference(unittest.TestCase):
    def test_permutation_and_owner_affinity(self):
        for boundary in (256, 257, 512, 513, 960):
            with self.subTest(boundary=boundary):
                result = reference_job_map(*make_inputs(boundary))
                jobs = boundary + 64
                self.assertEqual(
                    sorted(job for job in result if job >= 0), list(range(jobs))
                )
                self.assertEqual(result.count(-1), 1024 - jobs)
                for slot, job in enumerate(result):
                    if job >= 0:
                        self.assertEqual(slot % 8, job % 8)

    def test_shifted_boundary_changes_fragment_priority(self):
        aligned = reference_job_map(*make_inputs(256))
        shifted = reference_job_map(*make_inputs(257))
        # XCD 0's transposed probes see fragment counts 8 at ordinal 56;
        # shifting the boundary swaps adjacent physical tiles, selecting 48.
        self.assertEqual(aligned[:8], list(range(56, 64)))
        self.assertEqual(shifted[:8], list(range(48, 56)))
        self.assertEqual(aligned[448], 312)
        self.assertEqual(shifted[448], 312)

    def test_equal_cost_tie_break_and_shared_expansion(self):
        ids, experts, counts = make_inputs(256)
        ids[:] = [0] * len(ids)
        result = reference_job_map(ids, experts, counts)
        self.assertEqual(result[:256], list(range(256)))
        self.assertEqual(result[448:456], list(range(312, 320)))
        self.assertEqual(result[504:512], list(range(256, 264)))
        self.assertEqual(result[512:], [-1] * 512)

    def test_unsupported_geometry_identity(self):
        cases = [
            {"boundary": 0},
            {"tokens": 4096},
            {"suffix": 31},
            {"suffix": 33},
            {"preceding_shared": True},
            {"boundary": 961},  # expanded map exceeds four visits
            {"boundary": 1100},  # physical map itself exceeds capacity
        ]
        for case in cases:
            with self.subTest(case=case):
                inputs = make_inputs(**case)
                physical = inputs[2][0] // 256
                boundary = physical - 32
                expanded = case.get("boundary") in (961, 1100)
                jobs = boundary + 64 if expanded else physical
                self.assertEqual(
                    reference_job_map(*inputs),
                    [slot if slot < jobs else -1 for slot in range(1024)],
                )


class TestGPUJobMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if any(importlib.util.find_spec(name) is None for name in ("torch", "triton")):
            raise unittest.SkipTest("Torch and Triton are required for kernel tests")
        import torch

        if not torch.cuda.is_available():
            raise unittest.SkipTest("GPU required")
        from aiter.ops.triton.moe.q256_schedule import q256_make_job_map

        cls.torch = torch
        cls.launch = staticmethod(q256_make_job_map)

    def _gpu_args(self, inputs):
        torch = self.torch
        return [torch.tensor(x, dtype=torch.int32, device="cuda") for x in inputs] + [
            torch.empty(1024, dtype=torch.int32, device="cuda")
        ]

    def _check_exact_map(self, inputs, reuse=False):
        expected = reference_job_map(*inputs)
        args = self._gpu_args(inputs)
        output = args[-1]
        if reuse:
            output.fill_(-99)
        self.assertIs(self.launch(*args), output)
        self.assertEqual(output.cpu().tolist(), expected)
        if reuse:
            output.fill_(-77)
            self.launch(*args)
            self.assertEqual(output.cpu().tolist(), expected)

    def test_exact_map_and_output_reuse(self):
        cases = [
            {"boundary": 256},
            {"boundary": 257},
            {"boundary": 512},
            {"boundary": 513},
            {"boundary": 960},
            {"boundary": 961},
            {"boundary": 0},
            {"tokens": 4096},
            {"suffix": 31},
            {"suffix": 33},
            {"preceding_shared": True},
            {"boundary": 1100},
        ]
        for case in cases:
            with self.subTest(case=case):
                self._check_exact_map(make_inputs(**case), reuse=True)

    def test_basic_tensor_validation(self):
        torch = self.torch
        valid_args = self._gpu_args(make_inputs())
        for index in range(4):
            for invalid in ("dtype", "rank", "stride", "device", "type"):
                with self.subTest(index=index, invalid=invalid):
                    args = valid_args.copy()
                    tensor = args[index]
                    args[index] = {
                        "dtype": lambda tensor=tensor: tensor.to(torch.int64),
                        "rank": lambda tensor=tensor: tensor.reshape(1, -1),
                        "stride": lambda tensor=tensor: torch.empty(
                            tensor.numel() * 2, dtype=torch.int32, device="cuda"
                        )[::2],
                        "device": lambda tensor=tensor: tensor.cpu(),
                        "type": lambda: None,
                    }[invalid]()
                    with self.assertRaises((TypeError, ValueError)):
                        self.launch(*args)
        for index, size in ((0, 255), (1, 0), (2, 1), (3, 1023), (3, 1025)):
            with self.subTest(index=index, size=size):
                args = valid_args.copy()
                args[index] = torch.empty(size, dtype=torch.int32, device="cuda")
                with self.assertRaises(ValueError):
                    self.launch(*args)

    def test_tagged_random_probes_and_equal_costs(self):
        for boundary in (257, 513):
            for occupancy in ("random", "full", "padding"):
                with self.subTest(boundary=boundary, occupancy=occupancy):
                    ids, experts, counts = make_inputs(boundary)
                    rng = random.Random(boundary)
                    for tile in range(boundary):
                        for fragment in range(1, 8):
                            live = occupancy == "full" or (
                                occupancy == "random" and rng.randrange(2)
                            )
                            token = rng.randrange(8192) if live else 8192
                            ids[tile * 256 + fragment * 32] = (7 << 24) | token
                    self._check_exact_map((ids, experts, counts))


if __name__ == "__main__":
    unittest.main()
