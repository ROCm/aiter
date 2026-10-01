import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SPEC = importlib.util.spec_from_file_location("replay", Path(__file__).with_name("run.py"))
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


class TestReplay(unittest.TestCase):
    def test_driver_lifecycle_for_each_mode(self):
        original_read = Path.read_text

        def read(path, *args, **kwargs):
            if path.name == ".hf-revision":
                return replay.PINS["models"]["Kimi-K3"]["revision"]
            return original_read(path, *args, **kwargs)

        def run(command, **kwargs):
            stdout = replay.PINS["atom"] if command == ["git", "rev-parse", "HEAD"] else "{}"
            return subprocess.CompletedProcess(command, 0, stdout=stdout)

        for mode in replay.MODES:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                catalog = root / ".github/benchmark/models_accuracy.json"
                catalog.parent.mkdir(parents=True)
                catalog.write_text(json.dumps([{
                    "model_name": "Kimi-K3", "model_path": "moonshotai/Kimi-K3",
                    "extraArgs": "-tp 8", "accuracy_threshold": 0.94,
                }]))
                previous = Path.cwd()
                try:
                    os.chdir(root)
                    with (
                        mock.patch.object(replay, "HERE", root / "tools"),
                        mock.patch.object(replay, "device_flags", return_value=["--cpuset-cpus", "0-19"]),
                        mock.patch.object(replay, "accuracy_result", return_value={"accuracy": 0.95}),
                        mock.patch.object(Path, "read_text", read),
                        mock.patch.object(subprocess, "run", side_effect=run) as commands,
                        mock.patch.object(subprocess, "Popen"),
                        mock.patch("sys.argv", ["run.py", "--model", "Kimi-K3", "--mode", mode]),
                    ):
                        replay.main()
                    summary = json.loads((root / "atom_diagnostics/summary.json").read_text())
                    self.assertNotEqual(summary["result"], "failed")
                    self.assertRegex(summary["container"], r"^atom-pr5989-[0-9a-f]{12}$")
                    executed = [item.args[0] for item in commands.call_args_list]
                    self.assertIn(["docker", "rm", "-f", summary["container"]], executed)
                finally:
                    os.chdir(previous)

    def test_device_flags_require_exact_gpu_allocation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "devices"
            path.write_text("--group-add 107 --device /dev/dri/renderD128 "
                            "--device /dev/dri/renderD130 --cpuset-cpus 4-23")
            self.assertIn("4-23", replay.device_flags(path, 2))
            with self.assertRaises(ValueError):
                replay.device_flags(path, 4)
            path.write_text("--device /dev/dri --cpuset-cpus 4-23")
            with self.assertRaises(ValueError):
                replay.device_flags(path, 1)

    def test_accuracy_requires_original_threshold_and_full_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            results = root / "accuracy_test_results"
            results.mkdir()
            data = {"results": {"gsm8k": {"exact_match,flexible-extract": 0.95}},
                    "n-samples": {"gsm8k": {"effective": 1319}}}
            path = results / "result.json"
            path.write_text(json.dumps(data))
            self.assertEqual(replay.accuracy_result(root, 0.94)["accuracy"], 0.95)
            with self.assertRaises(RuntimeError):
                replay.accuracy_result(root, 0.96)
            data["n-samples"]["gsm8k"]["effective"] = 100
            path.write_text(json.dumps(data))
            with self.assertRaises(RuntimeError):
                replay.accuracy_result(root, 0.94)
            data["results"]["gsm8k"]["exact_match,flexible-extract"] = float("nan")
            path.write_text(json.dumps(data))
            with self.assertRaises(RuntimeError):
                replay.accuracy_result(root, 0.94)


if __name__ == "__main__":
    unittest.main()
