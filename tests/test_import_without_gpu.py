"""`import aiter` must work in a process that cannot see a GPU."""

import os
import pathlib
import subprocess
import sys

_REPO = pathlib.Path(__file__).resolve().parents[1]


def test_import_aiter_without_visible_gpu():
    env = os.environ.copy()
    env.pop("GPU_ARCHS", None)
    # Unlike HIP_VISIBLE_DEVICES, this also hides the GPUs from rocminfo.
    env["ROCR_VISIBLE_DEVICES"] = "-1"
    code = "import torch; assert not torch.cuda.is_available(); import aiter"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
