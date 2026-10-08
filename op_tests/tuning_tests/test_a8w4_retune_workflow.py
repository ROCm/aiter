# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU behavior at the retune input/artifact and OS scheduling boundaries."""

import csv
import importlib.util
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "docs/a8w4_work/run_serial_retune.py"


def load_workflow():
    spec = importlib.util.spec_from_file_location("a8w4_retune_workflow", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def shape(token):
    return {
        "token": token,
        "model_dim": 3072,
        "inter_dim": 3072,
        "expert": 128,
        "topk": 4,
        "act_type": "ActivationType.Swiglu",
        "dtype": "torch.bfloat16",
        "q_dtype_a": "torch.float8_e4m3fn",
        "q_dtype_w": "torch.float4_e2m1fn_x2",
        "q_type": "QuantType.per_1x32",
        "use_g1u1": 1,
        "doweight_stage1": 0,
    }


class TestRetuneInputs(unittest.TestCase):
    def test_union_preserves_tuned_only_shape_and_tag_provenance_once(self):
        workflow = load_workflow()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            untuned, tuned = root / "untuned.csv", root / "tuned.csv"
            write_csv(untuned, [shape(3)])
            write_csv(
                tuned,
                [
                    dict(shape(3), _tag="", us=8),
                    dict(shape(3), _tag="flydsl_fallback", us=12),
                    dict(shape(4096), _tag="exact-m4096-a", us=9),
                ],
            )
            record = workflow.prepare_model("example", untuned, tuned, root / "out")
            rows = workflow.read_csv(record["input"])
            self.assertEqual([int(row["token"]) for row in rows], [3, 4096])
            self.assertEqual(record["shape_count"], 2)
            self.assertEqual(record["shapes"][0]["lookup_token"], 4)
            self.assertEqual(record["shapes"][1]["sources"][0]["tag"], "exact-m4096-a")
            self.assertEqual(len(record["shapes"][0]["sources"]), 3)
            self.assertTrue(Path(record["input"]).is_file())
            self.assertEqual(len(record["input_sha256"]), 64)

    def test_shape_identity_comes_from_input_header(self):
        workflow = load_workflow()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            untuned, tuned = root / "untuned.csv", root / "tuned.csv"
            write_csv(untuned, [dict(shape(3), source_dimension="left")])
            write_csv(tuned, [dict(shape(3), source_dimension="right", _tag="")])
            record = workflow.prepare_model("example", untuned, tuned, root / "out")
            self.assertEqual(record["shape_count"], 2)
            self.assertIn("source_dimension", record["shape_fields"])

    def test_winner_requires_matching_finite_profile_and_a8_producer_consumer(self):
        workflow = load_workflow()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path, tuned, failed, profile = [
                root / name
                for name in ("input.csv", "tuned.csv", "failed.csv", "profile.csv")
            ]
            row = dict(
                shape(3),
                gfx="gfx950",
                cu_num=256,
                block_m=16,
                us=2.0,
                kernelName1="flydsl_mxmoe_g1_a8w4_16x128x256_f16in_nt_fp8out_swiglu",
                kernelName2="flydsl_moe2_layout_afp8_wfp4_bf16_t16x128x128_atomic_sbm16",
            )
            write_csv(input_path, [shape(3)])
            write_csv(tuned, [row])
            failed.write_text(
                ",".join(workflow.SHAPE_FIELDS) + ",status,failure_reason\n"
            )
            observation = dict(
                row,
                precision="A8W4",
                search_mode="full",
                status="ok",
                pipeline_us=2.0,
                error=0.01,
            )
            write_csv(profile, [dict(observation, error="NaN")])
            self.assertEqual(
                workflow.check_coverage(input_path, tuned, failed, profile)["status"],
                "failed",
            )
            write_csv(profile, [observation])
            self.assertEqual(
                workflow.check_coverage(input_path, tuned, failed, profile)["status"],
                "passed",
            )
            bad = dict(
                row,
                kernelName2="flydsl_moe2_layout_afp8_wfp4_bf16_t16x128x128_atomic_sbm32",
            )
            write_csv(tuned, [bad])
            write_csv(profile, [dict(observation, kernelName2=bad["kernelName2"])])
            self.assertEqual(
                workflow.check_coverage(input_path, tuned, failed, profile)["status"],
                "failed",
            )


class TestSerialScheduler(unittest.TestCase):
    def test_executed_python_entry_excludes_formatter_input_files(self):
        workflow = load_workflow()
        self.assertEqual(
            workflow.python_entry(["python3", "-u", "-m", "black", "gemm_moe_tune.py"]),
            ("black", "-m"),
        )
        self.assertEqual(
            workflow.python_entry(
                [
                    "python3.12",
                    "-X",
                    "dev",
                    "-W",
                    "ignore",
                    "gemm_moe_tune.py",
                    "--mp",
                    "4",
                ]
            ),
            ("gemm_moe_tune.py", "script"),
        )
        self.assertEqual(
            workflow.python_entry(
                [
                    "python",
                    "-c",
                    "from multiprocessing.spawn import spawn_main",
                    "--multiprocessing-fork",
                ]
            )[1],
            "-c",
        )

    def test_models_wait_for_workers_before_next_launch_and_continue_failed_model(self):
        workflow = load_workflow()
        events = []

        class System:
            @contextmanager
            def locks(self, gpus):
                events.append("locked")
                yield
                events.append("released")

            def tune_processes(self):
                return []

            def source_state(self, repo):
                return {"revision": "fixture", "sources": {"kernel.py": "known hash"}}

            def gpu_snapshot(self, gpus):
                return [
                    {"hip_id": gpu, "bdf": "0000:05:00.0", "idle": True} for gpu in gpus
                ]

            def launch(self, command, **kwargs):
                name = Path(command[command.index("-i") + 1]).stem
                events.append(f"launch {name}")
                model = name.removesuffix("_input")

                class Process:
                    pid = 100

                    def wait(self):
                        import json

                        record = json.loads(
                            (Path(kwargs["cwd"]) / f"{model}_run.json").read_text()
                        )
                        assert record["pid"] == 100
                        assert record["status"] == "running"
                        assert "source" in record
                        events.append(f"leader ended {name}")
                        return 1

                return Process()

            def drain_group(self, process):
                events.append("workers ended")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            records = []
            for name, _prefix, _count in workflow.MODELS:
                input_path = output / f"{name}_input.csv"
                write_csv(input_path, [shape(3)])
                records.append(
                    {
                        "model": name,
                        "input": str(input_path),
                        "input_sha256": workflow.file_hash(input_path),
                        "shape_count": 1,
                    }
                )
            result = workflow.run_models(
                records, output, output, [4, 5], system=System()
            )
            self.assertEqual(result["status"], "failed")
            self.assertEqual(
                events,
                [
                    "locked",
                    "launch dsv4_a8w4_input",
                    "leader ended dsv4_a8w4_input",
                    "workers ended",
                    "launch kimik3_a8w4_input",
                    "leader ended kimik3_a8w4_input",
                    "workers ended",
                    "launch gptoss_a8w4_input",
                    "leader ended gptoss_a8w4_input",
                    "workers ended",
                    "released",
                ],
            )
            self.assertEqual(
                [row["model"] for row in result["models"]],
                [row[0] for row in workflow.MODELS],
            )
            self.assertTrue((output / "retune_run.json").is_file())

    def test_external_tune_blocks_every_model_before_launch(self):
        workflow = load_workflow()

        class System:
            @contextmanager
            def locks(self, gpus):
                yield

            def tune_processes(self):
                return [{"pid": 91, "entry": "gemm_moe_tune.py"}]

            def launch(self, *args, **kwargs):
                raise AssertionError("external tune must prevent subprocess launch")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "another tune"):
                workflow.run_models([], output, output, [4], system=System())


if __name__ == "__main__":
    unittest.main(verbosity=2)
