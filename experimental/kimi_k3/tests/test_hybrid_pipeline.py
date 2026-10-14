"""CPU-only launch-contract tests: no Torch, Triton, HIP, or GPU imports."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

source = Path(__file__).resolve().parents[1] / "k3_gluon_hybrid_sort_b8/pipeline.py"
spec = importlib.util.spec_from_file_location("hybrid_pipeline", source)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class HybridPipelineTest(unittest.TestCase):
    def test_launch_order_and_output_identity(self):
        for start in (0, 2):
            with self.subTest(start=start):
                events = []
                output = object()
                metadata = {key: object() for key in pipeline.METADATA_KEYS}
                options = dict(WORLD=8, num_warps=8, START=start, END=4)
                state = {
                    "Latent": SimpleNamespace(shape=(8, 3584)),
                    "IDs": SimpleNamespace(shape=(8, 16)),
                    "Weights": SimpleNamespace(shape=(8, 16)),
                }
                test = self

                class Kernel:
                    def __getitem__(self, grid):
                        test.assertEqual(grid, (512,))

                        def launch(*args, **kwargs):
                            events.append(("gluon", kwargs))

                        return launch

                def experts(latent, weights, ids, route_weights, prepared):
                    self.assertIs(prepared, metadata)
                    self.assertIs(latent, state["Latent"])
                    events.extend([("moe1", None), ("moe2", None)])
                    return output

                actual = pipeline.launch_hybrid(
                    Kernel(),
                    (),
                    options,
                    grid=512,
                    state=state,
                    weights={},
                    metadata=metadata,
                    experts=experts,
                )
                self.assertIs(actual, output)
                self.assertEqual(
                    [e[0] for e in events], ["gluon", "moe1", "moe2", "gluon"]
                )
                front, tail = events[0][1], events[-1][1]
                self.assertEqual(
                    (front["START"], front["END"], front["BEGIN"]), (start, 2, True)
                )
                self.assertEqual(
                    (tail["START"], tail["END"], tail["BEGIN"]), (4, 4, False)
                )
                self.assertIs(tail["ExternalRouted"], output)
                self.assertIs(front["SortOut"], metadata["SortOut"])
                self.assertEqual(
                    options, dict(WORLD=8, num_warps=8, START=start, END=4)
                )

    def test_reject_invalid_schedule(self):
        metadata = {key: object() for key in pipeline.METADATA_KEYS}
        for grid, world, warps, start in [
            (256, 8, 8, 0),
            (512, 4, 8, 0),
            (512, 8, 4, 0),
            (512, 8, 8, 1),
        ]:
            with self.subTest(grid=grid, world=world, warps=warps, start=start):
                with self.assertRaises(ValueError):
                    pipeline.stage_options(
                        dict(WORLD=world, num_warps=warps, START=start), metadata, grid
                    )
        with self.assertRaises(ValueError):
            pipeline.stage_options(dict(WORLD=8, num_warps=8), {}, 512)

    def test_reject_changed_expert_dispatch(self):
        config = SimpleNamespace(
            stage1=SimpleNamespace(keywords={"kernelName1": pipeline.MOE1}),
            stage2=SimpleNamespace(keywords={"kernelName2": pipeline.MOE2}),
            block_m=16,
            run_1stage=False,
            prequant=False,
        )
        self.assertIs(pipeline.require_expert_config(config), config)
        for field, value in [
            ("block_m", 32),
            ("run_1stage", True),
            ("prequant", True),
            ("stage2", SimpleNamespace(keywords={})),
        ]:
            with self.subTest(field=field):
                changed = SimpleNamespace(**vars(config))
                setattr(changed, field, value)
                with self.assertRaises(ValueError):
                    pipeline.require_expert_config(changed)


if __name__ == "__main__":
    unittest.main()
