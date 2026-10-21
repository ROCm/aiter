"""Hide the full-tile validity read under QK; keep the existing wait-count invariant."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

base = Path("/home/jun_chen2_qle/yzhou_aiter")
repo = base / "aiter-dsv41-fp8-qk-pv-opt-20261021"
work = base / "fp8-qk-pv-opt-20261021"
tag = "flags_prefetch"
out = work / tag
out.mkdir()
backup = work / "first_validated_source"
backup.mkdir()
prov = json.loads((work / "final_validation/provenance.json").read_text())
for name, digest in prov["source_sha256"].items():
    assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == digest
    shutil.copy2(repo / name, backup / Path(name).name)
header = "csrc/kernels/mla/fused_dsv41/decode_fp8_qk.hpp"
main = "csrc/kernels/mla/fused_dsv41_mxfp8.cu"
h = (repo / header).read_text()
c = (repo / main).read_text()
anchor = "    auto mask_scores = [&](auto& s) {"
assert anchor in h
h = h.replace(anchor, "    vector_t<uint32_t, 4> tile_flags;\n" + anchor)
start = h.index("        const auto flags =", h.index(anchor))
end = h.index("        const uint32_t full", start)
h = h[:start] + "        const auto& flags = tile_flags;\n" + h[end:]
anchor = "        compute_qk(v_s);"
assert h.count(anchor) == 1
h = h.replace(
    anchor,
    """        // Issue validity before K operands. The first lgkmcnt(3) drains this
        // older read along with operand 0; no additional wait is needed after QK.
        const uint32_t flags_addr = uint32_t(reinterpret_cast<uintptr_t>(raw_cur + 32 * 528));
        asm volatile("ds_read_b128 %0, %1" : "=v"(tile_flags) : "v"(flags_addr) : "memory");
""" + anchor,
)
(repo / header).write_text(h)
formatter = (
    base / "fp8-qk-profile-opt-20261021/format_tool/clang_format/data/bin/clang-format"
)
subprocess.run([str(formatter), "-i", str(repo / header)], check=True)
for name in [header, main]:
    shutil.copy2(repo / name, out / Path(name).name)
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
print((out / "tests.log").read_text()[-8000:], flush=True)
(out / "status.json").write_text(json.dumps({"exit_code": p.returncode}))
p.check_returncode()
