"""CPU-only source contract checks; no Torch/Triton/HIP imports."""

import importlib.util
import math
from pathlib import Path
import random
import unittest
import sys

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("m3_pipeline", root / "pipeline.py")
pipeline = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pipeline
spec.loader.exec_module(pipeline)


def route(scores, bias, scale=2.0, shared=1.0):
    """The ATOM BF16-logit routing contract, starting after gate GEMV rounding."""
    sigmoid = [1.0 / (1.0 + math.exp(-v)) for v in scores]
    # The device key's low bits are replaced by expert priority. Avoid an
    # accidental high-precision tie rule in these deliberately separated data.
    keys = [sigmoid[e] + bias[e] for e in range(128)]
    selected = sorted(range(128), key=lambda e: (-keys[e], e))[:4]
    norm = sum(sigmoid[e] for e in selected)
    return selected + [128], [scale * sigmoid[e] / norm for e in selected] + [shared]


def metadata(routes, weights):
    """Independent stable-sort oracle for the BM16 destination contract."""
    b = len(routes)
    ordered = sorted(
        ((routes[t][s], t, s, weights[t][s]) for t in range(b) for s in range(5)),
        key=lambda x: x[0],
    )
    experts = sorted({e for e, _, _, _ in ordered})
    positions = {}
    for block, expert in enumerate(experts):
        rows = [(t, s, w) for e, t, s, w in ordered if e == expert]
        assert len(rows) <= 16
        for row, (t, s, w) in enumerate(rows):
            dest = block * 16 + row
            assert dest not in positions
            positions[dest] = ((s << 24) | t, w, t)
        for row in range(len(rows), 16):
            positions[block * 16 + row] = (b, 0.0, b)
    return experts, positions


class ContractTest(unittest.TestCase):
    def test_launch_order_and_direct_output(self):
        events = []
        actual = object()

        def front():
            events.append("front+metadata")

        def experts():
            events.extend(("moe1", "moe2"))
            return actual

        def tail(routed):
            self.assertIs(routed, actual)
            events.append("tail")

        routed = pipeline.launch(
            pipeline.Shape(16),
            front=front,
            experts=experts,
            tail=tail,
            front_args=(),
            expert_args=(),
            tail_args=(),
        )
        self.assertIs(routed, actual)
        self.assertEqual(events, ["front+metadata", "moe1", "moe2", "tail"])
        self.assertFalse(pipeline.use_hybrid(4))
        self.assertTrue(pipeline.use_hybrid(5))
        for kwargs in (
            dict(tokens=0),
            dict(tokens=17),
            dict(tokens=8, tp=8),
            dict(tokens=8, grid=512),
        ):
            with self.assertRaises(ValueError):
                pipeline.Shape(**kwargs)

    def test_all_batch_sizes_and_routing_patterns(self):
        rng = random.Random(37)
        for b in range(1, 17):
            for pattern in ("same", "distinct", "random"):
                routes = []
                weights = []
                for t in range(b):
                    if pattern == "same":
                        ids = [0, 1, 2, 3]
                    elif pattern == "distinct":
                        ids = [(4 * t + s) % 128 for s in range(4)]
                    else:
                        ids = rng.sample(range(128), 4)
                    routes.append(ids + [128])
                    weights.append([0.1, 0.2, 0.3, 0.4, 1.0])
                experts, positions = metadata(routes, weights)
                self.assertEqual(len(positions), len(experts) * 16)
                self.assertLessEqual(len(positions), b * 5 * 16)
                self.assertEqual(
                    len([p for p in positions.values() if p[0] != b]), b * 5
                )
                self.assertEqual(
                    len([p for p in positions.values() if p[0] == b]),
                    len(positions) - b * 5,
                )
                shared_block = experts.index(128)
                self.assertEqual(
                    [positions[shared_block * 16 + row][0] for row in range(b)],
                    [(4 << 24) | t for t in range(b)],
                )

    def test_bias_affects_selection_only_and_shared_is_unscaled(self):
        scores = [0.0] * 128
        bias = [0.0] * 128
        bias[127] = 0.25
        ids, weights = route(scores, bias, shared=0.75)
        self.assertEqual(ids, [127, 0, 1, 2, 128])
        self.assertEqual(weights, [0.5, 0.5, 0.5, 0.5, 0.75])


if __name__ == "__main__":
    unittest.main()
