"""Step 4 of op_tests/dump_data/docs/mi350p_peer_write_primitive_2026-09-07.md.

A standalone HIP model of the two-shot wire protocol (peer_write_bw.cpp
--handshake) runs at 28-33 GB/s per rank on this host today. That is what
fly_int4 achieved on 09-01 and 3.7x what it achieves now, so the missing time is
in something the model does not contain. This times the real kernel with pieces
switched off to find out which:

    none      unmodified; the control. Must land on the real kernel's time,
              which is what says the ablation copy is faithful.
    nocomm    no fanout, no publish, no wait. "The kernel minus the network."
    comms     no codec, no input/output tiles. "The network only."
    nocodec   the network and all the traffic, without the INT4 arithmetic.

`nocomm` is the decisive one. If it is close to the full kernel's time then the
network is not where the time goes -- the compute is -- and since the same
compute was nearly free on 09-01, that points at the dequant/reduce codegen.
If it is small, the loss is in the comms after all, and the disagreement with
the standalone model localises it to how FlyDSL emits or schedules the kernel.

Results are wrong by construction in every mode but `none`; only times mean
anything. Nothing under aiter/ is modified: the ablated builder is patched over
`qr_int4.make_qr_int4_kernel` in this process only.

Usage (one variant per process -- the variant is fixed at kernel build time):

    HIP_VISIBLE_DEVICES=0,1,2,3 python3 run_ablation.py --ablate nocomm
    HIP_VISIBLE_DEVICES=0,1,2,3 python3 run_ablation.py --all
"""

import argparse
import os
import subprocess
import sys
from multiprocessing import Pool, set_start_method

REPO = "/home/vpietila/git/aiter"
HIDDEN = 7168
SHAPES = (1024, 4096, 8192)
ITERS = 101
WARMUP = 5
VARIANTS = ("none", "nocomm", "nocodec", "comms", "nowait", "multistore",
            "multistore_comms", "flatfan", "flataddr", "onestripe", "empty")

# Remote egress per dispatch from section 9 of the regression report, in bytes,
# so a time can be quoted as the wire bandwidth the kernel actually achieves.
# 26.24 MiB at M=4096, scaled linearly with the payload.
IO_BYTES_AT_4096 = int(26.24 * (1 << 20))


def _worker(tp, rank, init_method, shapes, grid_cap=None, super_tile=None,
            iters=ITERS, warmup=WARMUP):
    import torch
    import torch.distributed as dist

    from aiter import dtypes
    from aiter.dist.parallel_state import (
        ensure_model_parallel_initialized,
        get_tp_group,
        init_distributed_environment,
    )
    from aiter.ops.flydsl import QRInt4
    from aiter.ops.flydsl.kernels import qr_int4 as qr_int4_mod
    from aiter.test_common import run_perftest

    # Patch before any engine is built. `qr_int4.py` did `from .qr_int4_kernel
    # import make_qr_int4_kernel`, so the name to replace is the one bound in
    # that module, not the one in qr_int4_kernel.
    from qr_int4_kernel_ablate import make_qr_int4_kernel as ablated

    qr_int4_mod.make_qr_int4_kernel = ablated
    # _build_two_shot closes over the module global, so patching the attribute is
    # enough -- but assert it took, because a silent miss would report the real
    # kernel's time under an ablation label.
    assert qr_int4_mod.make_qr_int4_kernel is ablated

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    init_distributed_environment(
        world_size=tp, rank=rank, distributed_init_method=init_method
    )
    ensure_model_parallel_initialized(tp, 1)
    tp_group = get_tp_group()
    group = tp_group.device_group

    kw = {}
    if grid_cap is not None:
        kw["grid_cap"] = grid_cap
    if super_tile is not None:
        kw["super_tile"] = super_tile
    eng = QRInt4(
        group=tp_group.cpu_group,
        device=device,
        rank=rank,
        world_size=tp,
        algorithm="two_shot",
        **kw,
    )
    # Compile at the SHAPE WE MEAN TO PROFILE, not at a token warm-up tensor.
    #
    # rocprofv3 --att traces the FIRST matching dispatches and then stops. An
    # 8-row warm tensor makes those first dispatches 4-workgroup launches, so a
    # capture comes back with a valid-looking decode of the wrong thing: the
    # 2026-09-07 10:39 and 10:43 runs both traced dispatches 705/706, which were
    # `compile()`'s two engine launches (334 instructions, 1,077 executions),
    # while the M=4096 timed loop was never traced at all. Compiling at the
    # target shape makes dispatch 1 the real geometry.
    #
    # `compile()` walks every super-tile rung, so this launches the ST=1 engine
    # and then the ST=8 one; under a capture they land in two dispatch folders
    # with different instruction listings, and the ST=8 one (224 wg at M=4096)
    # is the one to analyse.
    warm_m = max(shapes)
    warm = torch.zeros((warm_m, HIDDEN), dtype=dtypes.bf16, device=device)
    dist.barrier(group=group)
    eng.compile(warm, torch.empty_like(warm))
    del warm

    out = {}
    for m in shapes:
        x = torch.randn((m, HIDDEN), dtype=dtypes.bf16, device=device)
        o = torch.empty_like(x)
        dist.barrier(group=group)
        _, us = run_perftest(
            eng.allreduce, x, o, num_iters=iters, num_warmup=warmup,
            use_cuda_event=True,
        )
        out[m] = us
    return out


def run_one(tp, shapes, grid_cap=None, super_tile=None, iters=ITERS,
            warmup=WARMUP):
    import torch  # noqa: F401  -- import here so --all's parent stays HIP-free

    from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with Pool(processes=tp) as pool:
        rets = [
            pool.apply_async(_worker, args=(tp, r, init_method, shapes, grid_cap, super_tile,
                                   iters, warmup))
            for r in range(tp)
        ]
        pool.close()
        pool.join()
    per_rank = [r.get() for r in rets]
    return {m: max(pr[m] for pr in per_rank) for m in shapes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablate", choices=VARIANTS, default="none")
    ap.add_argument("--all", action="store_true", help="run every variant, one "
                    "subprocess each (the variant is fixed at build time)")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--grid-cap", type=int, default=None,
                    help="QRInt4 grid_cap. The publish cadence is set by "
                         "num_tiles/grid, not by super_tile, so this is the "
                         "knob that actually moves bytes-per-publish.")
    ap.add_argument("--super-tile", type=int, default=None,
                    help="pin QRInt4's super_tile; ST=1 publishes every 6912 B "
                         "per block, ST=8 every 55296 B")
    ap.add_argument("--iters", type=int, default=ITERS,
                    help="timed iterations. A capture profile wants this small "
                         "so the run leaves few dispatch folders, but non-zero "
                         "warmup so the traced dispatch is a warm one.")
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--shapes", type=int, nargs="*", default=list(SHAPES))
    args = ap.parse_args()

    if args.all:
        got = {}
        for v in VARIANTS:
            env = dict(os.environ, AITER_QRINT4_ABLATE=v)
            cmd = [sys.executable, os.path.abspath(__file__), "--ablate", v,
                   "--tp", str(args.tp), "--shapes", *map(str, args.shapes)]
            p = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if p.returncode != 0:
                print(f"{v}: FAILED\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
                continue
            for line in p.stdout.splitlines():
                if line.startswith("RESULT "):
                    _, var, m, us = line.split()
                    got[(var, int(m))] = float(us)
        _report(got, args.shapes)
        return

    os.environ["AITER_QRINT4_ABLATE"] = args.ablate
    res = run_one(args.tp, args.shapes, args.grid_cap, args.super_tile,
                  args.iters, args.warmup)
    for m, us in sorted(res.items()):
        print(f"RESULT {args.ablate} {m} {us:.3f}")


def _report(got, shapes):
    print()
    print("two-shot ST=8, TP4, bf16, hidden 7168, slowest rank, us")
    hdr = f"{'M':>6}" + "".join(f"{v:>12}" for v in VARIANTS)
    print(hdr)
    for m in shapes:
        row = f"{m:>6}"
        for v in VARIANTS:
            us = got.get((v, m))
            row += f"{us:>12.1f}" if us is not None else f"{'-':>12}"
        print(row)
    print()
    print("share of the full kernel's time")
    print(f"{'M':>6}" + "".join(f"{v:>12}" for v in VARIANTS))
    for m in shapes:
        base = got.get(("none", m))
        row = f"{m:>6}"
        for v in VARIANTS:
            us = got.get((v, m))
            row += f"{us / base:>11.2f}x" if us and base else f"{'-':>12}"
        print(row)
    print()
    print("wire bandwidth implied by the full kernel and by 'comms' alone")
    print(f"{'M':>6}{'full GB/s':>12}{'comms GB/s':>12}")
    for m in shapes:
        io = IO_BYTES_AT_4096 * m / 4096
        for label, v in (("", "none"),):
            pass
        full = got.get(("none", m))
        comms = got.get(("comms", m))
        f = f"{io / (full * 1e3):>12.2f}" if full else f"{'-':>12}"
        c = f"{io / (comms * 1e3):>12.2f}" if comms else f"{'-':>12}"
        print(f"{m:>6}{f}{c}")


if __name__ == "__main__":
    sys.path.insert(0, REPO)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    set_start_method("spawn", force=True)
    main()
