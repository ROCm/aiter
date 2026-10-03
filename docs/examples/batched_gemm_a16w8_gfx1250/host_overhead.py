"""Host-side (CPU) cost per call of batched_gemm_a16w8, by stage.

GPU kernels here take ~600 us each, so a burst of N async launches measures pure host
overhead (the launch queue does not fill). Run ONLY through gpurun.sh.
"""

import json
import time

import torch
import triton
import triton.experimental.gluon.language as gl
from aiter.ops.triton._gluon_kernels.gfx1250.gemm.batched.batched_gemm_a16w8 import (
    _batched_gemm_a16w8_gfx1250_persistent_kernel as kern,
)
from aiter.ops.triton.gemm.batched.batched_gemm_a16w8 import batched_gemm_a16w8

from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16 import (
    _pad_interval,
    create_shared_layouts,
    create_wmma_layouts,
)
from aiter.ops.triton.utils.device_info import get_num_sms
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config
from aiter.ops.triton.utils.types import get_fp8_dtypes


def per_call_us(fn, n=50, sync_each=False):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
        if sync_each:
            torch.cuda.synchronize()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    return (t1 - t0) * 1e6 / n


def main():
    B, M, N, K = 128, 1536, 512, 128
    _, e4m3 = get_fp8_dtypes()
    x = torch.randn((B, M, K), dtype=torch.bfloat16, device="cuda")
    w = (torch.randn((B, N, K), device="cuda") * 0.1).to(e4m3)
    s = torch.tensor(0.02, dtype=torch.float32, device="cuda")
    y = torch.empty((B, M, N), dtype=torch.bfloat16, device="cuda")
    res = {}

    res["wrapper_total"] = per_call_us(lambda: batched_gemm_a16w8(x, w, s, YQ=y))
    res["get_gemm_config"] = per_call_us(
        lambda: get_gemm_config("BATCHED_GEMM-A16W8", M, N, K, backend="gluon")
    )
    cfg, _ = get_gemm_config("BATCHED_GEMM-A16W8", M, N, K, backend="gluon")
    BM, BN, BK, nw = (
        cfg["BLOCK_SIZE_M"],
        cfg["BLOCK_SIZE_N"],
        cfg["BLOCK_SIZE_K"],
        cfg["num_warps"],
    )

    def layouts():
        wl, oa, ob = create_wmma_layouts(nw)
        sa, sb = create_shared_layouts(BM, BN, BK, "TN", 16, elem_bits_b=8)
        sc = gl.PaddedSharedLayout.with_identity_for(
            [[_pad_interval(BN, 16), cfg.get("C_PAD", 16)]], [BM, BN], [1, 0]
        )
        tpn = min(32, BN // 8)
        st = gl.BlockedLayout([1, 8], [32 // tpn, tpn], [nw, 1], [1, 0])
        return wl, oa, ob, sa, sb, sc, st

    res["build_layouts"] = per_call_us(layouts)
    res["get_num_sms"] = per_call_us(get_num_sms)
    wl, oa, ob, sa, sb, sc, st = layouts()
    grid = min(B * triton.cdiv(M, BM) * triton.cdiv(N, BN), get_num_sms())

    def raw_launch():
        kern[(grid,)](
            x, w, y, s, B, M, N, K,
            x.stride(0), x.stride(1), w.stride(0), w.stride(1), y.stride(0), y.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, NUM_BUFFERS=cfg["NUM_BUFFERS"], NUM_WGS=grid,
            STORE_MODE=2, SHARED_LAYOUT_A=sa, SHARED_LAYOUT_B=sb, SHARED_LAYOUT_C=sc,
            STORE_LAYOUT=st, WMMA_LAYOUT=wl, OPERAND_LAYOUT_A=oa, OPERAND_LAYOUT_B=ob,
            num_warps=nw, waves_per_eu=1,
        )  # fmt: skip

    res["jit_launch_prebuilt_layouts"] = per_call_us(raw_launch)

    # Compiled-kernel fast path: bypass JIT argument binding/specialization.
    ck = next(iter(next(iter(kern.device_caches.values()))[0].values()))
    try:
        launcher_args = (x, w, y, s, B, M, N, K, x.stride(0), x.stride(1), w.stride(0),
                         w.stride(1), y.stride(0), y.stride(1))  # fmt: skip

        def compiled_launch():
            ck[(grid, 1, 1)](*launcher_args)

        res["compiled_kernel_launch"] = per_call_us(compiled_launch)
    except Exception as e:  # noqa: BLE001
        res["compiled_kernel_launch_error"] = repr(e)[:300]

    # GPU-side reference for the same call (graph replay, no host overhead in the loop)
    g = torch.cuda.CUDAGraph()
    s2 = torch.cuda.Stream()
    with torch.cuda.stream(s2):
        batched_gemm_a16w8(x, w, s, YQ=y)
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s2):
            batched_gemm_a16w8(x, w, s, YQ=y)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(20):
        g.replay()
    b.record()
    b.synchronize()
    res["gpu_kernel_graph_us"] = a.elapsed_time(b) * 1e3 / 20
    print(
        "RESULT "
        + json.dumps(
            {k: round(v, 2) if isinstance(v, float) else v for k, v in res.items()}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
