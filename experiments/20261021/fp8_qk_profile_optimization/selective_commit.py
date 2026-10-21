"""Compile a head-count specialization for the measured raw-tile commit schedule."""

import json
import os
import subprocess
import sys
from pathlib import Path

base = Path("/home/jun_chen2_qle/yzhou_aiter")
repo = base / "aiter-dsv41-fp8-qk-profile-opt-20261021"
work = base / "fp8-qk-profile-opt-20261021"
tag = sys.argv[1]
out = work / tag
out.mkdir()
seed = work / "wide_native_pv3_dequant16_qkds_offset_storeflip"
header = "csrc/kernels/mla/fused_dsv41/decode_fp8_qk.hpp"
main = "csrc/kernels/mla/fused_dsv41_mxfp8.cu"
h = (seed / "decode_fp8_qk.hpp").read_text()
c = (seed / "fused_dsv41_mxfp8.cu").read_text()
assert "template <class Traits, class Source>" in h
h = h.replace(
    "template <class Traits, class Source>",
    "template <class Traits, bool LateCommit = false, class Source>",
)
start = h.index(
    "        if(tile_idx + 1 < num_kv_tiles)", h.index("        compute_qk(v_s);")
)
end = h.index("        // Reuse the sole V slot", start)
block = h[start:end].rstrip()
h = (
    h[:start]
    + "        if constexpr(!LateCommit)\n        {\n"
    + block
    + "\n        }\n\n"
    + h[end:]
)
anchor = "        if(!below)\n            scale_output_tile<T>(v_o, rescale_m);\n\n"
assert anchor in h
h = h.replace(
    anchor,
    anchor
    + "        if constexpr(LateCommit)\n        {\n"
    + block
    + "\n        }\n\n",
)
start = c.index(
    "template <bool RotateQ, int PageSize, bool PreparedQ = false>\n__global__ __launch_bounds__(256, 1) void fused_dsv41_fp8_qk_decode_64"
)
end = c.index("template <bool RotateQ, bool LateLoad, int PageSize>", start)
k = (
    c[start:end]
    .replace(
        "bool PreparedQ = false>", "bool PreparedQ = false, bool LateCommit = false>"
    )
    .replace("fp8_qk_decode_accum64<T>", "fp8_qk_decode_accum64<T, LateCommit>")
)
c = c[:start] + k + c[end:]
start = c.index("            if(a->q_scale)", c.index("hipError_t launch_cache_mode"))
end = c.index("            return hipGetLastError();", start)
dispatch = c[start:end]
late = (
    dispatch.replace("<false, PageSize, true>", "<false, PageSize, true, true>")
    .replace("<true, PageSize>", "<true, PageSize, false, true>")
    .replace("<false, PageSize>", "<false, PageSize, false, true>")
)
c = (
    c[:start]
    + """            // H64 benefits from overlapping V dequantization with the next raw
            // tile fetch. H128 retains the earlier commit to avoid its regression.
            if(a->heads == 64)
            {
"""
    + late
    + """            }
            else
            {
"""
    + dispatch
    + """            }
"""
    + c[end:]
)
for name, data in [(header, h), (main, c)]:
    (repo / name).write_text(data)
    (out / Path(name).name).write_text(data)
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
    )
print((out / "tests.log").read_text()[-10000:], flush=True)
(out / "status.json").write_text(json.dumps(dict(exit_code=p.returncode)))
p.check_returncode()
