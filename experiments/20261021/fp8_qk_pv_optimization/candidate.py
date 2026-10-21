"""Test ways to write BF16 V with fewer bank conflicts and less VALU selection."""

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
if tag == "readflip8":
    start = h.index(
        "        opus::static_for<4>([&](auto j) {", h.index("    dequant_current(")
    )
    end = h.index("    }\n};", start)
    h = h[:start] + """        opus::static_for<4>([&](auto j) {
            const int d = int(j) * 128 + (lane % 8) * 16;
            const float scale = __uint_as_float(uint32_t(bytes[row * 528 + 512 + d / 32]) << 23);
            auto* dst = lds + ((int(j) * 2 + (lane % 8) / 4) * 4 + wave) * 528 + (lane / 8) * 64 + (lane % 4) * 16;
            // Rotate the raw-load order so converted vectors need no VALU select.
            opus::static_for<2>([&](auto part) {
                const int half = int(part) * 8 ^ ((lane & 4) * 2);
                auto bits = *reinterpret_cast<OPUS_LDS_ADDR const opus::vector_t<uint32_t, 2>*>(bytes + row * 528 + d + half);
                opus::vector_t<bf16_t, 8> values;
                opus::static_for<2>([&](auto word) {
                    const auto lo = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(bits[int(word)], scale, false);
                    const auto hi = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(bits[int(word)], scale, true);
                    auto* pairs = reinterpret_cast<opus::vector_t<bf16_t, 2>*>(&values);
                    pairs[int(word) * 2] = __builtin_bit_cast(opus::vector_t<bf16_t, 2>, lo);
                    pairs[int(word) * 2 + 1] = __builtin_bit_cast(opus::vector_t<bf16_t, 2>, hi);
                });
                *reinterpret_cast<OPUS_LDS_ADDR opus::vector_t<bf16_t, 8>*>(dst + half) = values;
            });
        });
""" + h[end:]
elif tag == "valid_fast":
    start = h.index("    commit_raw(")
    marker = "        const int row = (lane / 8) * 4 + wave;"
    pos = h.index(marker, start) + len(marker)
    h = (
        h[:pos]
        + """\n        // Every row owned by this wave is valid. Bit 1 in its first row's
        // existing flag lets QK skip the per-score mask loads on full tiles.
        const bool all_live = __builtin_amdgcn_ballot_w64(raw.live) == ~uint64_t(0);"""
        + h[pos:]
    )
    old = "*reinterpret_cast<OPUS_LDS_ADDR uint32_t*>(bytes + 32 * 528 + row * 4) = raw.live;"
    assert old in h
    h = h.replace(
        old,
        "*reinterpret_cast<OPUS_LDS_ADDR uint32_t*>(bytes + 32 * 528 + row * 4) = uint32_t(raw.live) | (uint32_t(all_live) << 1);",
    )
    anchor = "    auto mask_scores = [&](auto& s) {"
    assert anchor in h
    h = h.replace(
        anchor,
        anchor + """
        const auto flags = *reinterpret_cast<OPUS_LDS_ADDR const vector_t<uint32_t, 4>*>(raw_cur + 32 * 528);
        const uint32_t full = flags[0] & flags[1] & flags[2] & flags[3];
        if(__builtin_amdgcn_readfirstlane(full) & 2u)
            return;
""",
    )
elif tag == "cached_scales":
    old = "dequant_current(RawPointer raw_src, Pointer dst, int wave, int lane) const"
    assert old in h
    h = h.replace(
        old,
        "dequant_current(RawPointer raw_src, Pointer dst, int wave, int lane, const opus::vector_t<uint32_t, 4>& scale_words) const",
    )
    old = "float scale = __uint_as_float(uint32_t(bytes[row * 528 + 512 + d / 32]) << 23);"
    assert old in h
    h = h.replace(
        old,
        "float scale = __uint_as_float(((scale_words[int(j)] >> (((lane % 8) / 2) * 8)) & 255u) << 23);",
    )
    old = """    if constexpr(Source::is_packed)
        source.commit_raw(source.prefetch(0, warp_id, lane_id), raw_cur, warp_id, lane_id);"""
    assert old in h
    h = h.replace(
        old,
        """    opus::vector_t<uint32_t, 4> current_scales;
    if constexpr(Source::is_packed)
    {
        const auto initial = source.prefetch(0, warp_id, lane_id);
        current_scales = initial.scales;
        source.commit_raw(initial, raw_cur, warp_id, lane_id);
    }""",
    )
    h = h.replace(
        "source.dequant_current(raw_cur, s_kv.ptr, warp_id, lane_id);",
        "source.dequant_current(raw_cur, s_kv.ptr, warp_id, lane_id, current_scales);",
    )
    h = h.replace(
        "        std::swap(raw_cur, raw_next);",
        "        current_scales = next_raw.scales;\n        std::swap(raw_cur, raw_next);",
    )
elif tag == "predequant_barrier":
    old = """        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier(); // All PV reads finish before V is reused.
"""
    assert old in h
    h = h.replace(old, "")
    anchor = (
        "        // Reuse the sole V slot only after the previous iteration's barrier."
    )
    assert anchor in h
    h = h.replace(
        anchor,
        """        // QK reads raw K only, so it can overlap another wave's previous PV.
        // Join before overwriting V; the caller joins again before epilogue reuse.
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();""",
    )
elif tag == "storebranch":
    start = h.index(
        "            const auto lo = opus::slice(values",
        h.index("    dequant_current("),
    )
    end = h.index("        });\n    }\n};", start)
    h = (
        h[:start]
        + """            const auto lo = __builtin_bit_cast(opus::vector_t<uint32_t,4>, opus::slice(values, opus::number<0>{}, opus::number<8>{}));
            const auto hi = __builtin_bit_cast(opus::vector_t<uint32_t,4>, opus::slice(values, opus::number<8>{}, opus::number<16>{}));
            const uint32_t address = uint32_t(reinterpret_cast<uintptr_t>(dst));
            if(lane & 4)
            {
                asm volatile("ds_write_b128 %0, %1 offset:16" :: "v"(address), "v"(hi) : "memory");
                asm volatile("ds_write_b128 %0, %1" :: "v"(address), "v"(lo) : "memory");
            }
            else
            {
                asm volatile("ds_write_b128 %0, %1" :: "v"(address), "v"(lo) : "memory");
                asm volatile("ds_write_b128 %0, %1 offset:16" :: "v"(address), "v"(hi) : "memory");
            }
"""
        + h[end:]
    )
else:
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
