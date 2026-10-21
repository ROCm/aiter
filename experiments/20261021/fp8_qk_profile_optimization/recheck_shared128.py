"""Recheck H128 shared loading after the VMEM and LDS pipeline improvements."""

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
h = (
    work / "wide_native_pv3_dequant16_qkds_offset_storeflip/decode_fp8_qk.hpp"
).read_text()
c = subprocess.check_output(["git", "show", "06ca613d9:" + main], cwd=repo, text=True)
h = h.replace(
    "template <class Traits, class Source>\n__device__ void fp8_qk_decode_accum64",
    "template <class Traits, int NumWaves = 4, class Source>\n__device__ void fp8_qk_decode_accum64",
)
h = h.replace(
    "(thread_id_x() / T::WARP_SIZE) % T::NUM_WARPS",
    "(thread_id_x() / T::WARP_SIZE) % NumWaves",
)
h = h.replace(
    """    if constexpr(Source::is_packed)
        source.commit_raw(source.prefetch(0, warp_id, lane_id), raw_cur, warp_id, lane_id);""",
    """    if constexpr(Source::is_packed)
    {
        if(NumWaves == 4 || warp_id < 4)
            source.commit_raw(source.prefetch(0, warp_id, lane_id), raw_cur, warp_id, lane_id);
    }""",
)
h = h.replace(
    "auto next_row = source.prefetch_index(32 + (lane_id / 8) * 4 + warp_id);",
    "auto next_row = (NumWaves == 4 || warp_id < 4) ? source.prefetch_index(32 + (lane_id / 8) * 4 + warp_id) : typename Source::Index{-1};",
)
h = h.replace(
    """        if constexpr(Source::is_packed)
        {
            auto current_row""",
    """        if constexpr(Source::is_packed)
        {
            if(NumWaves == 4 || warp_id < 4) {
            auto current_row""",
)
h = h.replace(
    """            next_row = source.prefetch_index((tile_idx + 2) * 32 + (lane_id / 8) * 4 + warp_id);
        }""",
    """            next_row = source.prefetch_index((tile_idx + 2) * 32 + (lane_id / 8) * 4 + warp_id);
            }
        }""",
)
h = h.replace(
    "if(tile_idx + 1 < num_kv_tiles)",
    "if(tile_idx + 1 < num_kv_tiles && (NumWaves == 4 || warp_id < 4))",
)
h = h.replace(
    "        source.dequant_current(raw_cur, s_kv.ptr, warp_id, lane_id);",
    "        if(NumWaves == 4 || warp_id < 4)\n            source.dequant_current(raw_cur, s_kv.ptr, warp_id, lane_id);",
)
a = c.index(
    "template <bool RotateQ, int PageSize, bool PreparedQ = false>\n__global__ __launch_bounds__(256, 1) void fused_dsv41_fp8_qk_decode_64"
)
z = c.index("template <bool RotateQ, bool LateLoad, int PageSize>", a)
k = (
    c[a:z]
    .replace("fused_dsv41_fp8_qk_decode_64", "fused_dsv41_fp8_qk_decode_128")
    .replace("__launch_bounds__(256, 1)", "__launch_bounds__(512, 1)")
)
k = (
    k.replace("blockIdx.y * 64", "blockIdx.y * 128")
    .replace("int(blockIdx.y) * 64", "int(blockIdx.y) * 128")
    .replace("min(64,", "min(128,")
    .replace("i += 256", "i += 512")
)
k = k.replace(
    "    __shared__ __align__(16) B kv_tile[T::smem_kv_tile_elems];",
    "    __shared__ __align__(32) uint8_t storage[T::smem_kv_tile_elems * sizeof(B) + 2 * (32 * 528 + 128)];\n    auto* kv_tile = reinterpret_cast<B*>(storage);",
)
k = k.replace(
    "    __shared__ __align__(32) uint8_t raw_k[2 * (32 * 528 + 128)];",
    "    auto* raw_k = storage + T::smem_kv_tile_elems * sizeof(B);",
)
k = k.replace("fp8_qk_decode_accum64<T>", "fp8_qk_decode_accum64<T, 8>")
k = k.replace(
    "auto* staging = reinterpret_cast<uint8_t*>(kv_tile);", "auto* staging = storage;"
)
c = c[:z] + k + c[z:]
anchor = """        if(a->fp8_qk)
        {
            dim3 grid(a->tokens, a->heads / 64);"""
assert anchor in c
c = c.replace(
    anchor,
    """        if(a->fp8_qk)
        {
            if(a->heads == 128)
            {
                dim3 grid(a->tokens, 1);
                if(a->q_scale)
                    fused_dsv41_fp8_qk_decode_128<false, PageSize, true><<<grid, 512, 0, stream>>>(*a);
                else if(rotate_q)
                    fused_dsv41_fp8_qk_decode_128<true, PageSize><<<grid, 512, 0, stream>>>(*a);
                else
                    fused_dsv41_fp8_qk_decode_128<false, PageSize><<<grid, 512, 0, stream>>>(*a);
                return hipGetLastError();
            }
            dim3 grid(a->tokens, a->heads / 64);""",
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
        check=False,
    )
print((out / "tests.log").read_text()[-10000:], flush=True)
(out / "status.json").write_text(json.dumps({"exit_code": p.returncode}))
p.check_returncode()
