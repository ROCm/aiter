"""Regular LDS stores for an N64 tile, avoiding the old DMA candidate's addressing."""

import json
import os
import subprocess
import sys
from pathlib import Path

base = Path("/home/jun_chen2_qle/yzhou_aiter")
repo = base / "aiter-dsv41-fp8-qk-profile-opt-20261021"
work = base / "fp8-qk-profile-opt-20261021"
tag = sys.argv[1] if len(sys.argv) > 1 else "n64_wide"
out = work / tag
out.mkdir()
header = "csrc/kernels/mla/fused_dsv41/decode_fp8_qk.hpp"
main = "csrc/kernels/mla/fused_dsv41_mxfp8.cu"
h = (
    work
    / (sys.argv[2] if len(sys.argv) > 2 else "wide_deferred_index")
    / "decode_fp8_qk.hpp"
).read_text()
c = subprocess.check_output(["git", "show", "06ca613d9:" + main], cwd=repo, text=True)
h = h.replace("typename decltype(mma0)::vtype_c v_s;", "vector_t<D_ACC, 16> v_s;")
h = h.replace("typename decltype(mma1)::vtype_a v_p;", "vector_t<D_ATTN, 16> v_p;")
h = h.replace(
    "constexpr index_t s_len = vector_traits<typename decltype(mma0)::vtype_c>::size();",
    "constexpr index_t s_len = 16;",
)
h = h.replace(
    "auto* raw_next = raw_cur + 32 * 528 + 128;",
    "constexpr int SlotBytes = 32 * 528 + 128;\n    auto* raw_next = raw_cur + 2 * SlotBytes;",
)
a = h.index("    auto mask_scores =")
z = h.index("    // Explicit VGPR", a)
part = h[a:z].replace("raw_cur", "key_slot")
part = part.replace(
    "auto mask_scores = [&](auto& s)", "auto mask_scores = [&](auto& s, auto key_slot)"
)
part = part.replace(
    "auto compute_qk = [&](auto& s)", "auto compute_qk = [&](auto& s, auto key_slot)"
)
part = part.replace("mask_scores(s);", "mask_scores(s, key_slot);")
h = h[:a] + part + h[z:]
h = h.replace(
    """    if constexpr(Source::is_packed)
        source.commit_raw(source.prefetch(0, warp_id, lane_id), raw_cur, warp_id, lane_id);""",
    """    if constexpr(Source::is_packed)
    {
        source.commit_raw(source.prefetch(0, warp_id, lane_id), raw_cur, warp_id, lane_id);
        source.commit_raw(source.prefetch(1, warp_id, lane_id), raw_cur + SlotBytes, warp_id, lane_id);
    }""",
)
h = h.replace(
    "    int64_t next_row = source.row(32 + (lane_id / 8) * 4 + warp_id);",
    """    int64_t next_rows[2] = {source.row(64 + (lane_id / 8) * 4 + warp_id), source.row(96 + (lane_id / 8) * 4 + warp_id)};""",
)
h = h.replace("++tile_idx)", "tile_idx += 2)")
a = h.index("        typename Source::RawTile next_raw;")
z = h.index("        // Reuse the sole V slot", a)
h = h[:a] + """        static_for<2>([&](auto part) {
            typename Source::RawTile next_raw;
            if(tile_idx + 2 + int(part) < num_kv_tiles)
                next_raw = source.prefetch_row(next_rows[int(part)], warp_id, lane_id);
            next_rows[int(part)] = source.row((tile_idx + 4 + int(part)) * 32 + (lane_id / 8) * 4 + warp_id);
            __builtin_amdgcn_sched_barrier(0);
            vector_t<D_ACC, 8> scores;
            compute_qk(scores, raw_cur + int(part) * SlotBytes);
            set_slice(v_s, scores, number<int(part) * 8>{}, number<int(part) * 8 + 8>{});
            __builtin_amdgcn_sched_barrier(0);
            // Commit an invalid zero tile for an odd tail as well.
            if(tile_idx + 2 < num_kv_tiles)
            {
                if(tile_idx + 2 + int(part) >= num_kv_tiles)
                    next_raw = source.prefetch(tile_idx + 2 + int(part), warp_id, lane_id);
                source.commit_raw(next_raw, raw_next + int(part) * SlotBytes, warp_id, lane_id);
            }
        });
""" + h[z:]
h = h.replace(
    "        source.dequant_current(raw_cur, s_kv.ptr, warp_id, lane_id);",
    """        static_for<2>([&](auto part) {
            // smem<T>::ptr is byte-addressed; use a typed BF16 pointer for the tile stride.
            source.dequant_current(raw_cur + int(part) * SlotBytes, reinterpret_cast<D_ATTN*>(smem_kv_buf) + int(part) * T::smem_kv_tile_elems, warp_id, lane_id);
        });""",
)
a = h.index("        v_v[0] = tr_load", h.index("    for(int tile_idx"))
z = h.index("        s_waitcnt_vmcnt(0_I);", a)
pv = h[a:z].replace("compute_pv(v_p,", "compute_pv(p_part,")
h = (
    h[:a]
    + """        static_for<2>([&](auto part) {
            const auto p_part = slice(v_p, number<int(part) * 8>{}, number<int(part) * 8 + 8>{});
            s_kv = make_smem(reinterpret_cast<D_ATTN*>(smem_kv_buf) + int(part) * T::smem_kv_tile_elems);
"""
    + pv
    + """        });
        s_kv = make_smem(reinterpret_cast<D_ATTN*>(smem_kv_buf));
"""
    + h[z:]
)
a = c.index("void fused_dsv41_fp8_qk_decode_64")
z = c.index("template <bool RotateQ, bool LateLoad, int PageSize>", a)
part = (
    c[a:z]
    .replace("kv_tile[T::smem_kv_tile_elems]", "kv_tile[2 * T::smem_kv_tile_elems]")
    .replace("raw_k[2 * (32 * 528 + 128)]", "raw_k[4 * (32 * 528 + 128)]")
)
c = c[:a] + part + c[z:]
if "safe" in tag:
    h = h.replace(
        "s_waitcnt_lgkmcnt(number<T::v_ds_read_insts>{})", "s_waitcnt_lgkmcnt(0_I)"
    )
    h = h.replace("s_waitcnt_lgkmcnt(3_I)", "s_waitcnt_lgkmcnt(0_I)")
    h = h.replace(
        "__builtin_amdgcn_sched_barrier(VALU_MASK)", "__builtin_amdgcn_sched_barrier(0)"
    )
if "nop" in tag:
    h = h.replace(
        'asm volatile("v_mfma_f32_16x16x32_bf16',
        'asm volatile("s_nop 3\\n\\tv_mfma_f32_16x16x32_bf16',
    )
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
print((out / "tests.log").read_text()[-10000:])
(out / "status.json").write_text(json.dumps(dict(exit_code=p.returncode)))
p.check_returncode()
