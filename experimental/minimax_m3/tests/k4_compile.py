"""Compile-only build of the FlyDSL K4 (K1 fused) for M3 at several widths, its
binary resources, and its traced hand-offs against the layout's mailbox contract.
No GPU needed (ARCH=gfx950). Usage: python3 tests/k4_compile.py [tokens ...]"""
import os
import sys
import time

os.environ.setdefault("ARCH", "gfx950")
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
from vllm.models.minimax_m3.amd.mono.config import HEAD_DIM, ONE_INDEX_HEAD
from vllm.models.minimax_m3.amd.mono.fw.runtime import compile as C
from vllm.models.minimax_m3.amd.mono.kernels.post_attn import K4_ABI, build_post_attn_kernel
from vllm.models.minimax_m3.amd.mono.layout import mailbox_regions

widths = [int(a) for a in sys.argv[1:]] or [1, 4, 8, 16]
for n in widths:
    for index_topk in (True, False):
        t0 = time.time()
        k4 = build_post_attn_kernel(4, HEAD_DIM**-0.5, 1e-6, 2.0, 1.0, 7.0, 0, 1, tokens=n, fuse_k1=True,
                                    heads=ONE_INDEX_HEAD, index_topk=index_topk)
        try:
            C.check_traced(k4, K4_ABI, mailbox_regions(n, ONE_INDEX_HEAD.count, True, index_topk))
            res = C.launcher_resources(k4)
            print(f"K4 n={n:2d} index_topk={index_topk}: ok {time.time()-t0:.0f}s scratch={C.scratch_bytes(k4)} {res}",
                  flush=True)
        except Exception as e:
            print(f"K4 n={n:2d} index_topk={index_topk}: FAIL {type(e).__name__}: {str(e)[:1500]}", flush=True)
