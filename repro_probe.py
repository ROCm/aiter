import os, sys, threading, time
os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "40G")
sys.path.insert(0, "/app/aiter-test/op_tests/multigpu_tests")
import torch
import test_mega_moe_v2 as T
from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2

# Measure the desync instead of inferring it. stage1 bumps entry_count by
# launch_grid_x per launch, so entry_count/launch_grid_x is a device-side count
# of launches that actually STARTED on this rank. A watchdog thread copies it
# (plus the parity/expected/ready flags) on a private stream once a second, so
# when the ranks wedge the last file written per rank shows exactly how far
# apart they drifted and which flag nobody can satisfy.

DECODE_BS = [5, 5, 3, 5, 8, 5, 6, 6]
MTPR = 8192
DEPTH = int(os.environ.get("PROBE_DEPTH", "61"))
TOTAL = int(os.environ.get("PROBE_TOTAL", "610"))

_enqueued = 0
_PINNED: dict = {}


def snapshot(moe, rank, grid, stream, seq):
    ws = getattr(moe, "_s1_dispatch_workspace", {}) or {}
    want = [
        ("entry_count", ws.get("entry_count")),
        ("epoch_gate", ws.get("epoch_gate")),
        ("parity", getattr(moe, "_s1_epoch_parity", None)),
        ("expected", getattr(moe, "_s1_epoch_expected", None)),
        ("count_done", ws.get("count_done")),
        ("plan_ready", ws.get("plan_ready")),
        ("pair_ready", ws.get("pair_ready")),
        ("payload_ready", ws.get("payload_ready")),
    ]
    out = [
        f"rank={rank} seq={seq} t={time.strftime('%H:%M:%S')} "
        f"enqueued={_enqueued} launch_grid_x={grid}"
    ]
    # The copy MUST NOT touch pageable host memory: a blocking pageable D2H
    # implicitly synchronizes with the device, so the watchdog would wedge
    # alongside the kernel it is trying to observe (that is what killed the
    # first two attempts). Pinned staging + non_blocking on a private stream
    # keeps the read independent of the stuck compute stream.
    got = []
    with torch.cuda.stream(stream):
        for name, t in want:
            if not isinstance(t, torch.Tensor):
                continue
            flat = t.detach().flatten()
            host = _PINNED.get(name)
            if host is None or host.numel() != flat.numel() or host.dtype != flat.dtype:
                host = torch.empty(flat.shape, dtype=flat.dtype, pin_memory=True)
                _PINNED[name] = host
            host.copy_(flat, non_blocking=True)
            got.append((name, host))
    stream.synchronize()
    for name, v in got:
        val = v.tolist()
        if name == "entry_count" and grid:
            done = [x // grid for x in val if x]
            out.append(f"launches_started={done}")
        out.append(f"{name}={val[:24]}")
    with open(f"/tmp/mega_probe_rank{rank}.txt", "w") as f:
        f.write("\n".join(out) + "\n")


def main():
    rank, world, device = T._setup_dist()
    net = T.NETWORKS["v4_pro"]
    le = net["experts"] // world
    packed = T._quantize_weights(net["model_dim"], net["inter_dim"], le, rank, 123, device)
    moe = MegaMoEV2(
        rank=rank, world_size=world, quant="a8w4",
        w1=packed[0], w1_scale=packed[1], w2=packed[2], w2_scale=packed[3],
        max_tok_per_rank=MTPR, **net,
    )
    tiny = DECODE_BS[rank % len(DECODE_BS)]
    x, wts, ids = T._make_inputs(
        tiny, net["model_dim"], net["experts"], net["topk"], rank, 123, device
    )

    global _enqueued
    # One synced call first: entry_count then equals exactly one launch grid.
    moe.forward(x, wts, ids, config_tokens=MTPR)
    torch.cuda.synchronize()
    _enqueued = 1
    ec = moe._s1_dispatch_workspace["entry_count"]
    grid = max(int(v) for v in ec.flatten().tolist()) or 0
    if rank == 0:
        print(f"PROBE_START depth={DEPTH} total={TOTAL} launch_grid_x={grid}", flush=True)

    side = torch.cuda.Stream(device=device)
    stop = threading.Event()

    def loop():
        # The snapshot must keep working for the whole run: a swallowed
        # exception here silently freezes the files at an early state and the
        # wedge-time picture is lost (which is exactly what happened before).
        torch.cuda.set_device(device)
        seq = 0
        while not stop.is_set():
            seq += 1
            # Heartbeat first: if the CUDA read below blocks anyway, the file
            # still shows the seq/time it blocked at instead of going stale
            # silently.
            with open(f"/tmp/mega_probe_hb_rank{rank}.txt", "w") as f:
                f.write(
                    f"rank={rank} seq={seq} t={time.strftime('%H:%M:%S')} "
                    f"enqueued={_enqueued} phase=pre_copy\n"
                )
            try:
                snapshot(moe, rank, grid, side, seq)
            except Exception:  # noqa: BLE001
                import traceback

                with open(f"/tmp/mega_probe_rank{rank}.txt", "w") as f:
                    f.write(f"rank={rank} seq={seq} SNAPSHOT_FAILED\n")
                    f.write(traceback.format_exc())
            time.sleep(1.0)

    wd = threading.Thread(target=loop, daemon=True)
    wd.start()

    while _enqueued < TOTAL:
        n = min(DEPTH, TOTAL - _enqueued)
        for _ in range(n):
            moe.forward(x, wts, ids, config_tokens=MTPR)
            _enqueued += 1
        torch.cuda.synchronize()
        if rank == 0:
            print(f"PROBE {_enqueued}/{TOTAL}", flush=True)

    stop.set()
    wd.join(timeout=3)
    if rank == 0:
        print("PROBE_ALL_DONE", flush=True)
    T._cleanup()


if __name__ == "__main__":
    main()
