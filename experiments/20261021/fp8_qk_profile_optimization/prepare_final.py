import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

base = Path("/home/jun_chen2_qle/yzhou_aiter")
repo = base / "aiter-dsv41-fp8-qk-profile-opt-20261021"
work = base / "fp8-qk-profile-opt-20261021/release"
tag = sys.argv[1]
header = "csrc/kernels/mla/fused_dsv41/decode_fp8_qk.hpp"
main = "csrc/kernels/mla/fused_dsv41_mxfp8.cu"
h = (work / tag / "decode_fp8_qk.hpp").read_text()
h = h.replace("#pragma once", "#pragma once\n#include <type_traits>")
h = h.replace("    using typename Base::Words2;\n", "")
if "dequant" in tag:
    chunk = 32 if "dequant32" in tag else 16
    h = h.replace(
        "each 8-value piece directly: rebuilding RawTile prolongs register",
        f"each {chunk}-value piece directly: rebuilding RawTile prolongs register",
    )
h = h.replace(
    "    using Index =",
    """    // Preserve int32 paged indices until the next iteration consumes them.
    // Extending an async int32 load immediately would force a VMEM wait before QK.
    using Index =""",
)
h = h.replace(
    "        const auto* zero =",
    """        // A device-constant zero row keeps invalid/tail loads defined without
        // merging zero-filled vector PHIs before QK. commit_raw supplies scale 127.
        const auto* zero =""",
)
h = h.replace(
    "        tile_data.scales =",
    """        // Keep the original words live across QK; packing scales here serializes
        // the prefetch on their VMEM completion.
        tile_data.scales =""",
)
h = h.replace(
    "        opus::static_for<4>([&](auto j) {\n            const auto* values",
    """        // Four 16-byte reads preserve the public 4-byte row-alignment contract:
        // Words4 has scalar uint32_t fields, rather than a 16-byte-aligned pointer.
        opus::static_for<4>([&](auto j) {
            const auto* values""",
)
h = h.replace(
    "    auto compute_pv =",
    """    // Keep three V operands in flight. Partial waits count the outstanding
    // LDS reads; the scheduling fences preserve their order relative to MFMA.
    auto compute_pv =""",
)
(repo / header).write_text(h)
saved_main = work / tag / "fused_dsv41_mxfp8.cu"
(repo / main).write_text(
    saved_main.read_text()
    if saved_main.exists()
    else subprocess.check_output(
        ["git", "show", "06ca613d9:" + main], cwd=repo, text=True
    )
)
formatter = shutil.which("clang-format") or str(
    work / "format_tool/clang_format/data/bin/clang-format"
)
subprocess.run([formatter, "-i", str(repo / header), str(repo / main)], check=True)
subprocess.run(["git", "diff", "--check"], cwd=repo, check=True)
hashes = {
    p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in [header, main]
}
(work / "selection.json").write_text(
    json.dumps(
        dict(
            candidate=tag,
            baseline_commit="06ca613d93b1d19d7edb411f5dd79c26b6239c23",
            source_sha256=hashes,
        ),
        indent=2,
    )
)
out = work / "final_validation"
out.mkdir()
os.environ["AITER_JIT_DIR"] = str(work / "jit/final")
with (out / "native.log").open("w") as log:
    proc = subprocess.run(
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
    )
print((out / "native.log").read_text()[-8000:])
proc.check_returncode()
assert hashes == {
    p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in hashes
}
binary = work / "jit/final/module_fused_dsv41_mxfp8.so"
(out / "provenance.json").write_text(
    json.dumps(
        dict(
            source_sha256=hashes,
            binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        ),
        indent=2,
    )
)
print("FINAL_VALIDATED", tag)
