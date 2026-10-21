"""Prepare Q directly in scaled-MFMA lane order, sharing only group maxima."""

import json
import os
import subprocess
import sys
from pathlib import Path

base = Path("/home/jun_chen2_qle/yzhou_aiter")
repo = base / "aiter-dsv41-fp8-qk-pv-opt-20261021"
work = base / "fp8-qk-pv-opt-20261021"
tag = sys.argv[1]
out = work / tag
out.mkdir()
header = "csrc/kernels/mla/fused_dsv41/decode_fp8_qk.hpp"
main = "csrc/kernels/mla/fused_dsv41_mxfp8.cu"
h = subprocess.check_output(["git", "show", "950eaa3f2:" + header], cwd=repo, text=True)
c = subprocess.check_output(["git", "show", "950eaa3f2:" + main], cwd=repo, text=True)
start = c.index(
    "    if constexpr(PreparedQ)", c.index("void fused_dsv41_fp8_qk_decode_64")
)
end = c.index("    constexpr float log2e", start)
c = (
    c[:start]
    + """    // Each lane reads the two 16-byte pieces consumed by scaled MFMA.
    // Only two scalar group maxima and one scale code cross lanes per step.
    opus::static_for<4>([&](auto step) {
        if constexpr(PreparedQ)
        {
            opus::static_for<2>([&](auto part) {
                const int d = int(step) * 128 + (group / 2) * 32 + (group % 2) * 16 + int(part) * 64;
                const auto values = *reinterpret_cast<const opus::vector_t<opus::fp8_t, 16>*>(
                    reinterpret_cast<const uint8_t*>(a.q) + (int64_t(t) * a.heads + hload) * 512 + d);
                opus::set_slice(q[int(step)], values, opus::number<int(part) * 16>{}, opus::number<int(part) * 16 + 16>{});
            });
            q_scales[int(step)] = a.q_scale[(int64_t(t) * a.heads + hload) * 16 + int(step) * 4 + group];
        }
        else
        {
            int codes[2];
            opus::static_for<2>([&](auto part) {
                const int d = int(step) * 128 + (group / 2) * 32 + (group % 2) * 16 + int(part) * 64;
                opus::vector_t<float, 16> values;
                opus::static_for<2>([&](auto sub) {
                    B8 x = *reinterpret_cast<const B8*>(a.q + (int64_t(t) * a.heads + hload) * 512 + d + int(sub) * 8);
                    if constexpr(RotateQ && int(step) == 3 && int(part) == 1)
                        opus::static_for<4>([&](auto pair) {
                            const int i = int(pair) * 2;
                            const int ri = (d + int(sub) * 8 + i - 448) / 2;
                            const float x0 = f32(x[i]), x1 = f32(x[i + 1]), cv = f32(cs[ri]), sv = f32(sn[ri]);
                            x[i] = static_cast<B>(fmaf(x0, cv, -x1 * sv));
                            x[i + 1] = static_cast<B>(fmaf(x0, sv, x1 * cv));
                        });
                    opus::static_for<8>([&](auto j) { values[int(sub) * 8 + int(j)] = f32(x[int(j)]); });
                });
                float amax = 1e-30f;
                opus::static_for<16>([&](auto j) { amax = fmaxf(amax, fabsf(values[int(j)])); });
                const uint32_t other = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, __float_as_uint(amax));
                amax = fmaxf(amax, __uint_as_float(other));
                const unsigned bits = __float_as_uint(amax / 448.0f);
                const int exponent = int((bits - 1) >> 23) + 1 - 127;
                const float mul = __uint_as_float(uint32_t(127 - exponent) << 23);
                codes[int(part)] = exponent + 127;
                const auto quant = opus::cast<opus::fp8_t>(values * mul);
                opus::set_slice(q[int(step)], quant, opus::number<int(part) * 16>{}, opus::number<int(part) * 16 + 16>{});
            });
            const uint32_t code = (group & 1) ? codes[1] : codes[0];
            const int src = lane % 16 + ((group % 2) * 2 + group / 2) * 16;
            q_scales[int(step)] = __builtin_amdgcn_ds_bpermute(src * 4, code);
        }
    });
"""
    + c[end:]
)
for path, source in [(header, h), (main, c)]:
    (repo / path).write_text(source)
    (out / Path(path).name).write_text(source)
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
