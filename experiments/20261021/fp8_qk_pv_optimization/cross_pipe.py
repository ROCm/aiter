"""Overlap two V operands while each is reused across four head groups."""

import json
import os
import subprocess
import sys
from pathlib import Path

base = Path("/home/jun_chen2_qle/yzhou_aiter")
repo = base / "aiter-dsv41-fp8-qk-pv-opt-20261021"
work = base / "fp8-qk-pv-opt-20261021"
old = base / "fp8-qk-profile-opt-20261021/crosspv_reuse_qkds_storeflip"
tag = sys.argv[1]
out = work / tag
out.mkdir()
header = "csrc/kernels/mla/fused_dsv41/decode_fp8_qk.hpp"
main = "csrc/kernels/mla/fused_dsv41_mxfp8.cu"
h = (old / "decode_fp8_qk.hpp").read_text()
oldc = (old / "fused_dsv41_mxfp8.cu").read_text()
c = subprocess.check_output(["git", "show", "950eaa3f2:" + main], cwd=repo, text=True)
h = h.replace(
    "bool CrossPV = false, class Source>",
    "bool CrossPV = false, bool LateCommit = false, class Source>",
)
# Preserve the existing H64 late-commit schedule in the shared accumulator.
start = h.index(
    "        if(tile_idx + 1 < num_kv_tiles", h.index("        compute_qk(v_s);")
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
anchor = "        s_waitcnt_lgkmcnt(0_I);\n        __builtin_amdgcn_s_barrier(); // All waves have finished writing V."
assert anchor in h
h = h.replace(
    anchor,
    "        if constexpr(LateCommit)\n        {\n"
    + block
    + "\n        }\n\n"
    + anchor,
)
c = c.replace(
    "fp8_qk_decode_accum64<T, LateCommit>",
    "fp8_qk_decode_accum64<T, 4, false, LateCommit>",
)
# Add the cross-PV kernel and dispatch only H128 through it.
start = oldc.rfind(
    "template <bool RotateQ, int PageSize, bool PreparedQ = false>",
    0,
    oldc.index("void fused_dsv41_fp8_qk_decode_xpv64"),
)
end = oldc.index("template <bool RotateQ, bool LateLoad, int PageSize>", start)
kernel = oldc[start:end]
anchor = "template <bool RotateQ, int PageSize, bool PreparedQ = false, bool LateCommit = false>"
assert anchor in c
c = c.replace(anchor, kernel + anchor)
start = oldc.index(
    "            if(a->heads == 128)", oldc.index("hipError_t launch_cache_mode")
)
end = oldc.index("            dim3 grid(a->tokens, a->heads / 64);", start)
anchor = "            dim3 grid(a->tokens, a->heads / 64);"
assert anchor in c
c = c.replace(anchor, oldc[start:end] + anchor, 1)
start = h.index(
    "            static_for<Slices>([&](auto ds) {\n                const int slice = warp_id"
)
end = h.index("        }\n        s_waitcnt_vmcnt(0_I);", start)
h = h[:start] + """            vector_t<D_ATTN, 16> vals[2];
            auto load_v = [&](auto ds) {
                const int slice = warp_id * Slices + int(ds);
                const int off = (slice / 2) * T::smem_n_rpt * (T::smem_linear_wave + T::smem_padding_32B)
                              + (slice % 2) * T::SLICE_D;
                vals[int(ds) % 2] = tr_load<T::VEC_TR_V>(s_kv, u_rv + off);
            };
            load_v(0_I);
            load_v(1_I);
            s_waitcnt_lgkmcnt(number<T::v_ds_read_insts>{});
            __builtin_amdgcn_sched_barrier(0);
            static_for<Slices>([&](auto ds) {
                static_for<NumWaves>([&](auto g) {
                    constexpr int index = int(g) * Slices + int(ds);
                    v_o[index] = pv_vgpr(probs[int(g)], vals[int(ds) % 2], v_o[index]);
                });
                if constexpr(int(ds) + 2 < Slices)
                    load_v(number<int(ds) + 2>{});
                if constexpr(int(ds) + 1 < Slices)
                {
                    s_waitcnt_lgkmcnt(number<(int(ds) + 2 < Slices ? T::v_ds_read_insts : 0)>{});
                    __builtin_amdgcn_sched_barrier(0);
                }
            });
""" + h[end:]
if tag == "crosspipe_noremat":
    c = c.replace("Fp8QkSource64<PageSize, true> src", "Fp8QkSource64<PageSize> src")
elif tag != "crosspipe2":
    raise ValueError(tag)
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
