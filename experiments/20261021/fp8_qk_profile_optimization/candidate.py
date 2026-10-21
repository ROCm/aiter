"""Reduce the two VMEM waits identified in baseline ATT without changing the math."""

import hashlib
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
(repo / main).write_text(
    subprocess.check_output(["git", "show", "06ca613d9:" + main], cwd=repo, text=True)
)
text = subprocess.check_output(
    ["git", "show", "06ca613d9:" + header], cwd=repo, text=True
)
source = (repo / "csrc/kernels/mla/fused_dsv41/decode_prefetch.hpp").read_text()
a = source.index("    __device__ __forceinline__ RawTile prefetch(")
z = source.index("    template <class Pointer>", a)
method = source[a:z]
method = method.replace(
    "RawTile prefetch(int tile, int wave, int lane)",
    "RawTile prefetch_row(int64_t r, int wave, int lane)",
)
method = method.replace(
    "        const int64_t r = row(tile * 32 + (lane / 8) * 4 + (wave % 4));\n", ""
)
method = (
    method.replace("valid_row(r)", "this->valid_row(r)")
    .replace("cache.data<PageSize>", "this->cache.template data<PageSize>")
    .replace("cache.scale<PageSize>", "this->cache.template scale<PageSize>")
    .replace("(pool, r)", "(this->pool, r)")
)
extra = "    using typename Base::Words2;\n    using typename Base::Words4;\n"
if "deferred" in tag:
    extra += """    struct RawTile {
        opus::vector_t<uint32_t, 2> bytes[8];
        opus::vector_t<uint32_t, 4> scales;
        bool live;
    };
"""
    start = method.index("            const unsigned shift")
    end = method.index("            opus::static_for<8>", start)
    method = (
        method[:start]
        + "            tile_data.scales = {scales->x, scales->y, scales->z, scales->w};\n"
        + method[end:]
    )
    text = text.replace("const typename Base::RawTile& raw", "const RawTile& raw")
    text = text.replace(
        "(raw.scales[int(j) / 4] >> ((int(j) % 4) * 8))",
        "(raw.scales[int(j) / 2] >> ((int(j) % 2) * 16 + ((lane % 8) / 4) * 8))",
    )
else:
    extra += "    using typename Base::RawTile;\n"
if "wide" in tag:
    assert "deferred" in tag
    extra = extra.replace(
        "opus::vector_t<uint32_t, 2> bytes[8]", "opus::vector_t<uint32_t, 4> bytes[4]"
    )
    method = method.replace("opus::static_for<8>", "opus::static_for<4>")
    method = method.replace(
        "reinterpret_cast<const Words2*>(src + int(j) * 64 + (lane % 8) * 8)",
        "reinterpret_cast<const Words4*>(src + int(j) * 128 + (lane % 8) * 16)",
    )
    method = method.replace(
        "{values->x, values->y}", "{values->x, values->y, values->z, values->w}"
    )
    a = text.index("    template <class RawPointer>")
    z = text.index("    template <class Pointer, class RawPointer>", a)
    text = text[:a] + """    template <class RawPointer>
    __device__ __forceinline__ void commit_raw(
        const RawTile& raw, RawPointer raw_dst, int wave, int lane) const
    {
        auto* bytes = reinterpret_cast<OPUS_LDS_ADDR uint8_t*>(reinterpret_cast<uintptr_t>(raw_dst));
        const int row = (lane / 8) * 4 + wave;
        opus::static_for<4>([&](auto j) {
            const int d = int(j) * 128 + (lane % 8) * 16;
            *reinterpret_cast<OPUS_LDS_ADDR opus::vector_t<uint32_t, 4>*>(bytes + row * 528 + d) = raw.bytes[int(j)];
        });
        if((lane & 7) == 0)
        {
            *reinterpret_cast<OPUS_LDS_ADDR uint32_t*>(bytes + 32 * 528 + row * 4) = raw.live;
            const opus::vector_t<uint32_t, 4> codes = raw.live ? raw.scales : opus::vector_t<uint32_t, 4>{0x7f7f7f7f,0x7f7f7f7f,0x7f7f7f7f,0x7f7f7f7f};
            *reinterpret_cast<OPUS_LDS_ADDR opus::vector_t<uint32_t, 4>*>(bytes + row * 528 + 512) = codes;
        }
    }
""" + text[z:]
if "nobranch" in tag:
    assert "wide" in tag
    method = """    __device__ __forceinline__ RawTile prefetch_row(int64_t r, int wave, int lane) const
    {
        asm volatile("" : "+v"(lane));
        lane &= 63;
        RawTile tile_data;
        tile_data.live = this->valid_row(r);
        const auto* zero = reinterpret_cast<const uint8_t*>(fused_zero_kv);
        const auto* src = tile_data.live ? this->cache.template data<PageSize>(this->pool, r) : zero;
        const auto* scales = reinterpret_cast<const Words4*>(tile_data.live ? this->cache.template scale<PageSize>(this->pool, r) : zero);
        tile_data.scales = {scales->x, scales->y, scales->z, scales->w};
        opus::static_for<4>([&](auto j) {
            const auto* values = reinterpret_cast<const Words4*>(src + int(j) * 128 + (lane % 8) * 16);
            tile_data.bytes[int(j)] = {values->x, values->y, values->z, values->w};
        });
        return tile_data;
    }
"""
extra += (
    method
    + """    __device__ __forceinline__ RawTile prefetch(int tile, int wave, int lane) const {
        return prefetch_row(this->row(tile * 32 + (lane / 8) * 4 + wave), wave, lane);
    }
"""
)
anchor = "    using Base = PackedDecodeSource64<true, PageSize>;\n"
assert text.count(anchor) == 1
text = text.replace(anchor, anchor + extra)
if "index" in tag:
    anchor = "    for(int tile_idx = 0; tile_idx < num_kv_tiles; ++tile_idx)"
    text = text.replace(
        anchor,
        "    int64_t next_row = source.row(32 + (lane_id / 8) * 4 + warp_id);\n"
        + anchor,
    )
    text = text.replace(
        "next_raw = source.prefetch(tile_idx + 1, warp_id, lane_id);",
        "next_raw = source.prefetch_row(next_row, warp_id, lane_id);\n            next_row = source.row((tile_idx + 2) * 32 + (lane_id / 8) * 4 + warp_id);",
    )
if "nobranch" in tag:
    text = text.replace(
        "            if(tile_idx + 1 < num_kv_tiles)\n                next_raw =",
        "            next_raw =",
    )
if "nativeindex" in tag:
    source_anchor = "    using Base = PackedDecodeSource64<true, PageSize>;\n"
    text = text.replace(
        source_anchor,
        source_anchor
        + """    using Index = std::conditional_t<(PageSize != 0), int32_t, int64_t>;
    __device__ __forceinline__ Index prefetch_index(int i) const {
        if(i < 0 || i >= this->length) return Index{-1};
        return static_cast<const Index*>(this->indices)[i];
    }
""",
    )
    text = text.replace(
        "int64_t next_row = source.row(", "auto next_row = source.prefetch_index("
    )
    text = text.replace("next_row = source.row(", "next_row = source.prefetch_index(")
    text = text.replace(
        "            next_raw = source.prefetch_row(next_row, warp_id, lane_id);",
        """            auto current_row = next_row;
            asm volatile("" : "+v"(current_row));
            next_raw = source.prefetch_row(current_row, warp_id, lane_id);""",
    )
if "blocked" in tag:
    text = text.replace("bytes + row * 528 + 512", "bytes + 16640 + row * 16")
    text = text.replace(
        "bytes[row * 528 + 512 + d / 32]", "bytes[16640 + row * 16 + d / 32]"
    )
    text = text.replace(
        "bytes + row * 528 + d",
        "bytes + (d / 128) * 4160 + (row % 4) * 1040 + (row / 4) * 128 + d % 128",
    )
    text = text.replace(
        "raw_cur + row * 528 + kd",
        "raw_cur + step * 4160 + (row % 4) * 1040 + (row / 4) * 128 + kd % 128",
    )
    text = text.replace(
        "raw_cur[row * 528 + 512 + step * 4 + lane_id / 16]",
        "raw_cur[16640 + row * 16 + step * 4 + lane_id / 16]",
    )
    text = text.replace("32 * 528", "17152")
    main_text = (
        (repo / main)
        .read_text()
        .replace("raw_k[2 * (32 * 528 + 128)]", "raw_k[2 * (17152 + 128)]")
    )
    (repo / main).write_text(main_text)
    (out / "fused_dsv41_mxfp8.cu").write_text(main_text)
if "pv3" in tag or "pv4" in tag:
    depth = 3 if "pv3" in tag else 4
    text = text.replace(
        "typename decltype(mma1)::vtype_b v_v[2];",
        f"typename decltype(mma1)::vtype_b v_v[{depth}];",
    )
    a = text.index("    auto compute_pv =")
    z = text.index("    if constexpr(Source::is_packed)", a)
    text = (
        text[:a]
        + """    auto compute_pv = [&](const auto& p, auto& v, auto& o, int kv_tile_idx) {
        static_for<T::NUM_D_SLICES - DEPTH>([&](auto i) {
            constexpr int idx = int(i), slot = idx % DEPTH;
            o[idx] = pv_vgpr(p, v[slot], o[idx]);
            v[slot] = tr_load<T::VEC_TR_V>(s_kv, u_rv + skv_slice(number<idx + DEPTH>{}));
            s_waitcnt_lgkmcnt(number<(DEPTH - 1) * T::v_ds_read_insts>{});
            __builtin_amdgcn_sched_barrier(VALU_MASK);
        });
        static_for<DEPTH>([&](auto j) {
            constexpr int idx = T::NUM_D_SLICES - DEPTH + int(j);
            o[idx] = pv_vgpr(p, v[idx % DEPTH], o[idx]);
            if constexpr(int(j) + 1 < DEPTH) {
                s_waitcnt_lgkmcnt(number<(DEPTH - 2 - int(j)) * T::v_ds_read_insts>{});
                __builtin_amdgcn_sched_barrier(0);
            }
        });
    };

""".replace(
            "DEPTH", str(depth)
        )
        + text[z:]
    )
    start = text.index("        v_v[0] = tr_load", text.index("    for(int tile_idx"))
    end = text.index("        compute_pv(", start)
    text = text[:start] + """        static_for<DEPTH>([&](auto j) {
            v_v[int(j)] = tr_load<T::VEC_TR_V>(s_kv, u_rv + skv_slice(j));
        });
        s_waitcnt_lgkmcnt(number<(DEPTH - 1) * T::v_ds_read_insts>{});
        __builtin_amdgcn_sched_barrier(0);
""".replace("DEPTH", str(depth)) + text[end:]
(repo / header).write_text(text)
(out / "decode_fp8_qk.hpp").write_text(text)
os.environ["AITER_JIT_DIR"] = str(work / "jit" / tag)
with (out / "tests.log").open("w") as log:
    proc = subprocess.run(
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
(out / "status.json").write_text(
    json.dumps(
        dict(
            exit_code=proc.returncode,
            source_sha256=hashlib.sha256(text.encode()).hexdigest(),
        )
    )
)
proc.check_returncode()
print("READY", tag, flush=True)
