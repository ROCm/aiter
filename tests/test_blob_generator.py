"""Regression tests for source-tree blob-generator process launching."""

import os
import pathlib
import shlex
import sys
import tempfile
import unittest
from subprocess import CalledProcessError
from unittest.mock import patch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aiter.jit import core


class BlobGeneratorTest(unittest.TestCase):
    def test_source_tree_generator_can_import_worker_limits(self):
        with tempfile.TemporaryDirectory() as workdir:
            workdir = pathlib.Path(workdir)
            generator = workdir / "generator.py"
            generator.write_text(
                "import pathlib\n"
                "import sys\n"
                "import aiter_worker_limits\n"
                "pathlib.Path(sys.argv[1], 'generated').touch()\n"
            )
            blob_dir = workdir / "blob"
            blob_dir.mkdir()
            with patch.dict(os.environ, {"PYTHONPATH": ""}, clear=False):
                core._run_blob_generator(
                    f"{shlex.quote(str(generator))} {{}}", str(blob_dir)
                )
            self.assertTrue((blob_dir / "generated").exists())

    def test_run_blob_generator_prepends_repo_to_pythonpath(self):
        with (
            patch.dict(os.environ, {"PYTHONPATH": "/existing/path"}, clear=False),
            patch.object(core.subprocess, "run") as run,
        ):
            core._run_blob_generator("generator.py --output {}", "/tmp/blob")

        (command,) = run.call_args.args
        kwargs = run.call_args.kwargs
        self.assertIn("generator.py --output /tmp/blob", command)
        self.assertEqual(kwargs["shell"], True)
        self.assertEqual(kwargs["check"], True)
        self.assertEqual(
            kwargs["env"]["PYTHONPATH"],
            os.pathsep.join((core.AITER_ROOT_DIR, "/existing/path")),
        )
        self.assertNotEqual(os.environ.get("PYTHONPATH"), kwargs["env"]["PYTHONPATH"])

    def test_run_blob_generator_propagates_failure(self):
        failure = CalledProcessError(returncode=17, cmd="generator")
        with (
            patch.object(core.subprocess, "run", side_effect=failure),
            self.assertRaisesRegex(CalledProcessError, "17"),
        ):
            core._run_blob_generator("generator.py --output {}", "/tmp/blob")


if __name__ == "__main__":
    unittest.main(verbosity=2)
