"""Execute the documented batch file with an offline external compiler stub."""

import os
import runpy
import sys
from pathlib import Path

from op_tests.tuning_tests._controlled_tuner_import import import_moe_tuner


def main():
    for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        assert os.environ.get(name) == "", name
    import_moe_tuner()
    import flydsl.compiler
    import torch

    def forbidden(*args, **kwargs):
        raise AssertionError("selected batch smoke reached GPU work")

    def compile_without_launch(*args, **kwargs):
        assert os.environ["COMPILE_ONLY"] == "1"
        assert os.environ["FLYDSL_GPU_ARCH"] == "gfx950"

    flydsl.compiler.compile = compile_without_launch
    for name in ("_lazy_init", "synchronize", "Event", "CUDAGraph"):
        setattr(torch.cuda, name, forbidden)
    script = (
        Path(__file__).resolve().parents[2]
        / "docs/mxfp4_staged_main_port/scripts/compile_selected.py"
    )
    sys.argv[0] = str(script)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
