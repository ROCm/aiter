"""Reuse each V operand across four head groups; keep P resident across D slices."""

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
header = "csrc/kernels/mla/fused_dsv41/decode_fp8_qk.hpp"
main = "csrc/kernels/mla/fused_dsv41_mxfp8.cu"
h = (work / "shared128_qkds_storeflip/decode_fp8_qk.hpp").read_text()
c = (work / "shared128_qkds_storeflip/fused_dsv41_mxfp8.cu").read_text()
script = (base / "fp8-qk-h128-20261021/crosspv_transform.py").read_text()
script = script.replace(
    "start=h.index('        v_v[0] = tr_load')",
    "start=h.index('        static_for<3>([&](auto j)', h.index('    for(int tile_idx'))",
)
script = script.replace(
    "auto* staging = reinterpret_cast<uint8_t*>(kv_tile);", "auto* staging = storage;"
)
exec(compile(script, "crosspv_transform_current", "exec"), ns := {})
h, c = ns["transform"](h, c, "cross4pipe")
a = h.index("            constexpr int Slices = 16 / NumWaves;")
z = h.index("        }\n        s_waitcnt_vmcnt(0_I);", a)
h = h[:a] + """            constexpr int Slices = 16 / NumWaves;
            auto* p_lds = reinterpret_cast<OPUS_LDS_ADDR const D_ATTN*>(reinterpret_cast<uintptr_t>(exchange));
            auto* f_lds = reinterpret_cast<OPUS_LDS_ADDR const float*>(p_lds + NumWaves * 64 * 8);
            {
                float factors[NumWaves];
                static_for<NumWaves>([&](auto g) { factors[int(g)] = f_lds[int(g) * 16 + lane_id % 16]; });
                s_waitcnt_lgkmcnt(0_I);
                __builtin_amdgcn_sched_barrier(0);
                bool rescale = false;
                static_for<NumWaves>([&](auto g) { rescale |= factors[int(g)] != 1.0f; });
                if(__builtin_amdgcn_ballot_w64(rescale) != 0)
                    static_for<NumWaves>([&](auto g) {
                        static_for<Slices>([&](auto ds) { v_o[int(g) * Slices + int(ds)] *= factors[int(g)]; });
                    });
            }
            vector_t<D_ATTN, 8> probs[NumWaves];
            static_for<NumWaves>([&](auto g) {
                probs[int(g)] = *reinterpret_cast<OPUS_LDS_ADDR const vector_t<D_ATTN, 8>*>(p_lds + (int(g) * 64 + lane_id) * 8);
            });
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_sched_barrier(0);
            static_for<Slices>([&](auto ds) {
                const int slice = warp_id * Slices + int(ds);
                const int off = (slice / 2) * T::smem_n_rpt * (T::smem_linear_wave + T::smem_padding_32B)
                              + (slice % 2) * T::SLICE_D;
                auto val = tr_load<T::VEC_TR_V>(s_kv, u_rv + off);
                s_waitcnt_lgkmcnt(0_I);
                __builtin_amdgcn_sched_barrier(0);
                static_for<NumWaves>([&](auto g) {
                    constexpr int index = int(g) * Slices + int(ds);
                    v_o[index] = pv_vgpr(probs[int(g)], val, v_o[index]);
                });
            });
""" + h[z:]
for path, data in [(header, h), (main, c)]:
    (repo / path).write_text(data)
    (out / Path(path).name).write_text(data)
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
