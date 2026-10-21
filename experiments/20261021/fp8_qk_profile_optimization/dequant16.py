import json
import os
import subprocess
import sys
from pathlib import Path

base = Path("/home/jun_chen2_qle/yzhou_aiter")
repo = base / "aiter-dsv41-fp8-qk-profile-opt-20261021"
work = base / "fp8-qk-profile-opt-20261021"
tag = sys.argv[1]
source = sys.argv[2]
out = work / tag
out.mkdir()
header = "csrc/kernels/mla/fused_dsv41/decode_fp8_qk.hpp"
main = "csrc/kernels/mla/fused_dsv41_mxfp8.cu"
h = (work / source / "decode_fp8_qk.hpp").read_text()
a = h.index("    template <class Pointer, class RawPointer>")
z = h.index("\n};", a)
part = h[a:z]
part = part.replace("opus::static_for<8>", "opus::static_for<4>")
part = part.replace("int(j) * 64 + (lane % 8) * 8", "int(j) * 128 + (lane % 8) * 16")
part = part.replace("opus::vector_t<uint32_t, 2>", "opus::vector_t<uint32_t, 4>")
part = part.replace("opus::vector_t<bf16_t, 8>", "opus::vector_t<bf16_t, 16>")
part = part.replace("opus::static_for<2>", "opus::static_for<4>")
part = part.replace(
    "(int(j) * 4 + wave) * 528 + lane * 8",
    "((int(j) * 2 + (lane % 8) / 4) * 4 + wave) * 528 + (lane / 8) * 64 + (lane % 4) * 16",
)
if "dequant32" in tag:
    part = part.replace(
        "opus::static_for<4>([&](auto j)", "opus::static_for<2>([&](auto j)"
    )
    part = part.replace(
        "int(j) * 128 + (lane % 8) * 16", "int(j) * 256 + (lane % 8) * 32"
    )
    part = part.replace("opus::vector_t<uint32_t, 4>", "opus::vector_t<uint32_t, 8>")
    part = part.replace("opus::vector_t<bf16_t, 16>", "opus::vector_t<bf16_t, 32>")
    part = part.replace(
        "opus::static_for<4>([&](auto half)", "opus::static_for<8>([&](auto half)"
    )
    part = part.replace(
        "((int(j) * 2 + (lane % 8) / 4) * 4 + wave) * 528 + (lane / 8) * 64 + (lane % 4) * 16",
        "((int(j) * 4 + (lane % 8) / 2) * 4 + wave) * 528 + (lane / 8) * 64 + (lane % 2) * 32",
    )
h = h[:a] + part + h[z:]
c = subprocess.check_output(["git", "show", "06ca613d9:" + main], cwd=repo, text=True)
for p, data in [(header, h), (main, c)]:
    (repo / p).write_text(data)
    (out / Path(p).name).write_text(data)
os.environ["AITER_JIT_DIR"] = str(work / "jit" / tag)
with (out / "tests.log").open("w") as log:
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "op_tests/test_fused_dsv41_fp8_qk.py",
            "-q",
            "-x",
            "--tb=short",
            "--junitxml=" + str(out / "tests.xml"),
        ],
        cwd=repo,
        stdout=log,
        stderr=subprocess.STDOUT,
        check=False,
    )
print((out / "tests.log").read_text()[-10000:], flush=True)
(out / "status.json").write_text(json.dumps({"exit_code": p.returncode}))
p.check_returncode()
