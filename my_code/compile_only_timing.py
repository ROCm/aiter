#!/usr/bin/env python3
"""Time FlyDSL compilation of the a8w4 TDM GEMM1 kernel -- compile only, no launch.

COMPILE_ONLY=1 makes flydsl return right after codegen, so no kernel ever reaches
the GPU. This isolates "is nb=6 slow to compile?" from any runtime behaviour.

Usage (inside container hyg_fyd2, from the aiter repo root):
    python my_code/compile_only_timing.py                  # default sweep
    python my_code/compile_only_timing.py 16x128x256x6     # one config
    python my_code/compile_only_timing.py 16x256x256x2 16x128x256x6

Each config is compiled in a FRESH subprocess with an empty flydsl cache, so the
reported time is a true cold-compile number.
"""
import os
import subprocess
import sys
import time

# tile_m x tile_n x tile_k x num_buffers
DEFAULT_CFGS = [
    "16x256x256x2",   # baseline
    "16x256x256x4",
    "16x128x256x3",
    "16x128x256x4",
    "16x128x256x6",   # the suspect: drain unrolled 5x
    "64x256x256x3",   # current best perf config
]

CHILD = "--child"


def compile_one(tm, tn, tk, nb):
    """Runs in the child process: build tiny placeholder tensors, call the kernel."""
    import torch
    from aiter.ops.flydsl.batched_gemm_mxfp4 import flydsl_grouped_gemm_a8w4_masked

    E, N, K = 384, 1536, 7168
    CM = 30720
    dev = "cuda"
    u8 = torch.uint8

    # COMPILE_ONLY returns before any launch, so 1-element stand-ins are enough for
    # everything the kernel only takes the base address of.
    tiny = torch.zeros(1, dtype=u8, device=dev)
    out = torch.zeros(1, dtype=torch.bfloat16, device=dev)
    a_scales = torch.zeros(4, dtype=u8, device=dev)
    w_scales = torch.zeros(4, dtype=u8, device=dev)
    m_tile_map = torch.zeros(E, dtype=torch.int32, device=dev)
    bias = torch.zeros(1, dtype=torch.bfloat16, device=dev)

    t0 = time.perf_counter()
    flydsl_grouped_gemm_a8w4_masked(
        out, tiny, tiny, a_scales, w_scales, m_tile_map,
        contiguous_m=CM, N=N, K=K, n_experts=E,
        tile_m=tm, tile_n=tn, tile_k=tk,
        m_warp=1, n_warp=4, num_buffers=nb,
        out_is_f16=0, a_is_fp4=0,
        stage1_act=1, bias=bias, swiglu_limit=7.0,
    )
    return time.perf_counter() - t0


def main():
    args = [a for a in sys.argv[1:] if a != CHILD]

    if CHILD in sys.argv:
        tm, tn, tk, nb = (int(x) for x in args[0].split("x"))
        try:
            dt = compile_one(tm, tn, tk, nb)
            print(f"__RESULT__ ok {dt:.1f}")
        except Exception as e:  # noqa: BLE001 - report, don't mask
            print(f"__RESULT__ fail 0 {type(e).__name__}: {e}")
            raise
        return

    cfgs = args or DEFAULT_CFGS
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "1")
    print(f"compile-only timing (COMPILE_ONLY=1, no kernel launch)")
    print(f"{'config':>18s} {'status':>8s} {'compile_s':>10s}")
    print("-" * 40)

    results = []
    for cfg in cfgs:
        env = dict(os.environ)
        env["COMPILE_ONLY"] = "1"
        env["ENABLE_CK"] = "0"
        env["AITER_FORCE_GFX1250"] = "1"
        env.pop("FLYDSL_DUMP_IR", None)
        # cold compile: no reuse of a previous run's artifacts
        subprocess.run("rm -rf ~/.flydsl/cache/*", shell=True)

        log = f"my_code/sweep_tdm/compile_{cfg}.log"
        os.makedirs("my_code/sweep_tdm", exist_ok=True)
        t0 = time.perf_counter()
        with open(log, "w") as f:
            p = subprocess.run(
                [sys.executable, __file__, CHILD, cfg],
                env=env, stdout=f, stderr=subprocess.STDOUT, timeout=3600,
            )
        wall = time.perf_counter() - t0

        status = "OK" if p.returncode == 0 else f"rc={p.returncode}"
        print(f"{cfg:>18s} {status:>8s} {wall:10.1f}")
        results.append((cfg, status, wall))

    print("-" * 40)
    base = next((w for c, s, w in results if c == "16x256x256x2"), None)
    if base:
        print(f"\n{'config':>18s} {'compile_s':>10s} {'vs base':>9s}")
        for c, s, w in results:
            print(f"{c:>18s} {w:10.1f} {w / base:8.2f}x")


if __name__ == "__main__":
    main()
