"""Raw WMMA throughput on gfx1250 (operands in registers, no LDS, no memory).

Same per-wave work as batched_gemm_a16w8 at (B=128, M=1536, N=512, K=128) with the
tuned 256x256x64 / 4-warp tile: 12 wmma() calls of 128 x v_wmma_16x16x32 per wave,
one workgroup per CU. Run ONLY through gpurun.sh.
"""

import json

import torch
import triton
import triton.experimental.gluon.language as gl
from triton.experimental import gluon

from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16 import (
    create_wmma_layouts,
)
from aiter.ops.triton.utils.device_info import get_num_sms


@gluon.jit
def _wmma_loop(
    out_ptr,
    flag,
    ITERS: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    IN_DTYPE: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
    OP_A: gl.constexpr,
    OP_B: gl.constexpr,
):
    a = gl.full([BM, BK], 1.0, IN_DTYPE, layout=OP_A)
    b = gl.full([BK, BN], 1.0, IN_DTYPE, layout=OP_B)
    acc = gl.zeros([BM, BN], gl.float32, layout=WMMA_LAYOUT)
    for _ in range(ITERS):
        acc = gl.amd.gfx1250.wmma(a, b, acc)
    if flag == 1:
        offs_m = gl.arange(0, BM, layout=gl.SliceLayout(1, WMMA_LAYOUT))
        offs_n = gl.arange(0, BN, layout=gl.SliceLayout(0, WMMA_LAYOUT))
        gl.store(out_ptr + offs_m[:, None] * BN + offs_n[None, :], acc)


def layouts(num_warps, kind):
    if kind == "bf16":
        return create_wmma_layouts(num_warps)
    warp_bases = []
    for i in range(num_warps.bit_length() - 1):
        warp_bases.append((0, 1) if i == 0 else (1 << (i - 1), 0))
    wl = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=tuple(warp_bases),
        instr_shape=[16, 16, 64],
    )
    return (
        wl,
        gl.DotOperandLayout(operand_index=0, parent=wl, k_width=16),
        gl.DotOperandLayout(operand_index=1, parent=wl, k_width=16),
    )


def run(kind, iters, num_warps=4, BM=256, BN=256):
    BK = 64 if kind == "bf16" else 128
    in_dtype = gl.bfloat16 if kind == "bf16" else gl.float8e4nv
    wl, oa, ob = layouts(num_warps, kind)
    grid = get_num_sms()
    out = torch.empty((BM, BN), dtype=torch.float32, device="cuda")
    args = {
        "ITERS": iters, "BM": BM, "BN": BN, "BK": BK, "IN_DTYPE": in_dtype,
        "WMMA_LAYOUT": wl, "OP_A": oa, "OP_B": ob, "num_warps": num_warps,
    }  # fmt: skip
    k = _wmma_loop[(grid,)](out, 0, **args)
    torch.cuda.synchronize()
    ms = triton.testing.do_bench(
        lambda: _wmma_loop[(grid,)](out, 0, **args), warmup=5, rep=50
    )
    flops = 2.0 * BM * BN * BK * iters * grid
    asm = k.asm.get("amdgcn", "")
    n_wmma = sum(1 for line in asm.splitlines() if "v_wmma" in line)
    return {
        "kind": kind, "iters": iters, "grid": grid, "num_warps": num_warps,
        "us": round(ms * 1e3, 2), "TFLOPs": round(flops / (ms * 1e-3) / 1e12, 2),
        "wmma_per_wave": iters * (BM // 2) * (BN // 2) * BK // (16 * 16 * (32 if kind == "bf16" else 64)) // (num_warps // 4 if num_warps >= 4 else 1),
        "static_wmma_instrs_in_asm": n_wmma, "n_regs": k.n_regs, "n_spills": k.n_spills,
    }  # fmt: skip


if __name__ == "__main__":
    import sys

    todo = (
        [("bf16", 120)]
        if "--quick" in sys.argv
        else [("bf16", 12), ("bf16", 120), ("fp8", 12), ("fp8", 120)]
    )
    for kind, iters in todo:
        try:
            r = run(kind, iters)
        except Exception as e:  # noqa: BLE001
            r = {"kind": kind, "iters": iters, "error": repr(e)[:400]}
        print("RESULT " + json.dumps(r), flush=True)
