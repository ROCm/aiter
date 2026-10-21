"""Test explicit LDS loads against compiler-added full waits in QK ATT."""

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
h = (work / "wide_native_pv3_dequant16/decode_fp8_qk.hpp").read_text()
old = "auto part = *reinterpret_cast<OPUS_LDS_ADDR const vector_t<fp8_t, 16>*>(raw_cur + row * 528 + kd);"
new = """vector_t<uint32_t, 4> words;
                const uint32_t addr = uint32_t(reinterpret_cast<uintptr_t>(raw_cur + row * 528 + kd));
                asm volatile("ds_read_b128 %0, %1" : "=v"(words) : "v"(addr) : "memory");
                auto part = __builtin_bit_cast(vector_t<fp8_t, 16>, words);"""
if "offset" in tag:
    new = """vector_t<uint32_t, 4> words;
                const uint32_t addr = uint32_t(reinterpret_cast<uintptr_t>(raw_cur + (lane_id % 16) * 528 + (lane_id / 16) * 16));
                asm volatile("ds_read_b128 %0, %1 offset:%2" : "=v"(words) : "v"(addr), "n"(n * 16 * 528 + step * 128 + int(half) * 64) : "memory");
                auto part = __builtin_bit_cast(vector_t<fp8_t, 16>, words);"""
assert old in h
h = h.replace(old, new)
old = "scales[slot] = raw_cur[row * 528 + 512 + step * 4 + lane_id / 16];"
new = """const uint32_t scale_addr = uint32_t(reinterpret_cast<uintptr_t>(raw_cur + row * 528 + 512 + step * 4 + lane_id / 16));
            uint32_t code;
            asm volatile("ds_read_u8 %0, %1" : "=v"(code) : "v"(scale_addr) : "memory");
            scales[slot] = code;"""
if "offset" in tag:
    new = """const uint32_t scale_addr = uint32_t(reinterpret_cast<uintptr_t>(raw_cur + (lane_id % 16) * 528 + lane_id / 16));
            uint32_t code;
            asm volatile("ds_read_u8 %0, %1 offset:%2" : "=v"(code) : "v"(scale_addr), "n"(n * 16 * 528 + 512 + step * 4) : "memory");
            scales[slot] = code;"""
assert old in h
h = h.replace(old, new)
if "xor" in tag:
    assert "offset" in tag
    h = h.replace(
        "bytes + row * 528 + d)", "bytes + row * 528 + (d ^ ((row & 4) << 2)))"
    )
    h = h.replace(
        "raw_cur + (lane_id % 16) * 528 + (lane_id / 16) * 16)",
        "raw_cur + (lane_id % 16) * 528 + (((lane_id / 16) * 16) ^ ((lane_id & 4) << 2)))",
    )
if "storeflip" in tag:
    old = """            *reinterpret_cast<OPUS_LDS_ADDR opus::vector_t<bf16_t, 16>*>(lds + ((int(j) * 2 + (lane % 8) / 4) * 4 + wave) * 528 + (lane / 8) * 64 + (lane % 4) * 16) = values;"""
    new = """            auto* dst = lds + ((int(j) * 2 + (lane % 8) / 4) * 4 + wave) * 528 + (lane / 8) * 64 + (lane % 4) * 16;
            const auto lo = opus::slice(values, opus::number<0>{}, opus::number<8>{});
            const auto hi = opus::slice(values, opus::number<8>{}, opus::number<16>{});
            const bool flip = (lane & 4) != 0;
            *reinterpret_cast<OPUS_LDS_ADDR opus::vector_t<bf16_t, 8>*>(dst + (flip ? 8 : 0)) = flip ? hi : lo;
            *reinterpret_cast<OPUS_LDS_ADDR opus::vector_t<bf16_t, 8>*>(dst + (flip ? 0 : 8)) = flip ? lo : hi;"""
    assert old in h
    h = h.replace(old, new)
if "latecommit" in tag:
    start = h.index(
        "        if(tile_idx + 1 < num_kv_tiles)", h.index("        compute_qk(v_s);")
    )
    end = h.index("        // Reuse the sole V slot", start)
    block = h[start:end]
    h = h[:start] + h[end:]
    anchor = """        if(!below)
            scale_output_tile<T>(v_o, rescale_m);

"""
    assert anchor in h
    h = h.replace(anchor, anchor + block)
(repo / header).write_text(h)
(out / "decode_fp8_qk.hpp").write_text(h)
c = subprocess.check_output(["git", "show", "06ca613d9:" + main], cwd=repo, text=True)
(repo / main).write_text(c)
(out / "fused_dsv41_mxfp8.cu").write_text(c)
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
