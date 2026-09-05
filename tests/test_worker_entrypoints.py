"""Source-checkout entry points must find the dependency-free worker helper."""

import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

_REPO = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


class WorkerEntrypointTest(unittest.TestCase):
    def source_only_environment(self):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        return env

    def test_opus_offline_builder_imports_without_aiter_installed(self):
        script = _REPO / "csrc/opus_gemm/gen_co/build_co.py"
        with tempfile.TemporaryDirectory() as workdir:
            result = subprocess.run(
                [sys.executable, "-S", str(script), "--help"],
                cwd=workdir,
                env=self.source_only_environment(),
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_opus_device_setup_imports_without_aiter_installed(self):
        script = _REPO / "op_tests/opus/device/setup.py"
        code = (
            "import runpy; "
            f"runpy.run_path({str(script)!r}, run_name='source_import_test')"
        )
        with tempfile.TemporaryDirectory() as workdir:
            result = subprocess.run(
                [sys.executable, "-S", "-c", code],
                cwd=workdir,
                env=self.source_only_environment(),
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
