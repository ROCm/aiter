import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

base = Path("/home/jun_chen2_qle/yzhou_aiter")
repo = base / "aiter-dsv41-fp8-qk-pv-opt-20261021"
work = base / "fp8-qk-pv-opt-20261021/release"
tag = sys.argv[1]
paths = [
    "csrc/kernels/mla/fused_dsv41/decode_fp8_qk.hpp",
    "csrc/kernels/mla/fused_dsv41_mxfp8.cu",
]
for path in paths:
    shutil.copy2(work / tag / Path(path).name, repo / path)
formatter = (
    base / "fp8-qk-profile-opt-20261021/format_tool/clang_format/data/bin/clang-format"
)
subprocess.run([str(formatter), "-i", *[str(repo / p) for p in paths]], check=True)
subprocess.run(["git", "diff", "--check"], cwd=repo, check=True)
hashes = {p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in paths}
(work / "selection.json").write_text(
    json.dumps(
        {
            "candidate": tag,
            "baseline_commit": "950eaa3f23411469db0c426f5876039f63594715",
            "source_sha256": hashes,
        },
        indent=2,
    )
)
out = work / "final_validation"
out.mkdir()
os.environ["AITER_JIT_DIR"] = str(work / "jit/final")
with (out / "native.log").open("w") as log:
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "op_tests/test_fused_dsv41_mxfp8.py",
            "op_tests/test_fused_dsv41_paged.py",
            "op_tests/test_fused_dsv41_bf16.py",
            "op_tests/test_fused_dsv41_fp8_qk.py",
            "-q",
            "--tb=short",
            "--junitxml=" + str(out / "native.xml"),
        ],
        cwd=repo,
        stdout=log,
        stderr=subprocess.STDOUT,
        check=False,
    )
print((out / "native.log").read_text()[-8000:], flush=True)
p.check_returncode()
assert hashes == {
    name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in paths
}
binary = work / "jit/final/module_fused_dsv41_mxfp8.so"
(out / "provenance.json").write_text(
    json.dumps(
        {
            "source_sha256": hashes,
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        },
        indent=2,
    )
)
print("FINAL_VALIDATED", tag)
