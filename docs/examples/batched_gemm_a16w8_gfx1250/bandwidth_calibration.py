"""Device calibration: achievable HBM write / copy bandwidth with plain torch ops.

Run ONLY through gpurun.sh. Prints RESULT JSON lines.
"""

import json
import time

import torch


def timed(fn, iters):
    fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1e3 / iters


def main():
    props = torch.cuda.get_device_properties(0)
    out = {"name": props.name, "cus": props.multi_processor_count,
           "gcn": getattr(props, "gcnArchName", None)}  # fmt: skip
    print("RESULT " + json.dumps(out), flush=True)

    y = torch.empty(128 * 1536 * 512, dtype=torch.bfloat16, device="cuda")  # 201 MB
    us = timed(lambda: y.zero_(), 50)
    print("RESULT " + json.dumps({"test": "zero_201MB", "us": round(us, 2),
                                  "TBps": round(y.numel() * 2 / us / 1e6, 3)}), flush=True)  # fmt: skip

    src = torch.empty(65 * 1024 * 1024, dtype=torch.bfloat16, device="cuda")  # 130 MB
    dst = torch.empty_like(src)
    src.zero_()
    us = timed(lambda: dst.copy_(src), 50)
    print("RESULT " + json.dumps({"test": "copy_130MB", "us": round(us, 2),
                                  "TBps": round(2 * src.numel() * 2 / us / 1e6, 3)}), flush=True)  # fmt: skip

    # long run for clock sampling (~10 s of back-to-back writes)
    t0 = time.time()
    n = 0
    while time.time() - t0 < 10:
        for _ in range(200):
            y.zero_()
        torch.cuda.synchronize()
        n += 200
    print("RESULT " + json.dumps({"test": "sustain_zero", "launches": n,
                                  "us_per": round((time.time() - t0) * 1e6 / n, 2)}), flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
