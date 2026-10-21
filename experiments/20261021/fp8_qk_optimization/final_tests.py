import os
import subprocess
import sys
from pathlib import Path

w = Path("/home/jun_chen2_qle/yzhou_aiter/fp8-qk-opt-20261021")
tag = sys.argv[1]
os.environ["AITER_JIT_DIR"] = str(w / "jit" / tag)
with (w / "final_tests.log").open("w") as f:
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "op_tests/test_fused_dsv41_mxfp8.py",
            "op_tests/test_fused_dsv41_paged.py",
            "op_tests/test_fused_dsv41_bf16.py",
            "op_tests/test_fused_dsv41_fp8_qk.py",
            "-q",
            "-x",
            "--tb=short",
            "--junitxml=" + str(w / "final_tests.xml"),
        ],
        stdout=f,
        stderr=subprocess.STDOUT,
        check=False,
    )
print((w / "final_tests.log").read_text()[-8000:])
r.check_returncode()
