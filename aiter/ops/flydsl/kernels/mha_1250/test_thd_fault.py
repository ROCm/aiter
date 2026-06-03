"""Stress test THD varlen with many random seqlen configurations."""
import os, sys, random
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from test_acc import run_test_thd
import torch

if __name__ == "__main__":
    cases = [
        # original target
        ([100, 200, 300, 400], [100, 200, 300, 400], 2),
        # variations around the failing pattern
        ([100, 200, 300, 400], [100, 200, 300, 400], 1),
        ([100, 200, 300, 400], [100, 200, 300, 400], 4),
        ([400, 300, 200, 100], [400, 300, 200, 100], 2),
        ([50, 150, 250, 350], [50, 150, 250, 350], 2),
        # multi-batch with non-aligned sk's that fall on edges of tile_n=128
        ([1, 127, 128, 129, 130], [1, 127, 128, 129, 130], 2),
        ([127, 128, 129, 255, 256, 257], [127, 128, 129, 255, 256, 257], 1),
        ([130, 130, 130, 130], [130, 130, 130, 130], 2),
        # uneven q vs k where k crosses many tiles
        ([100, 200, 300, 400], [400, 300, 200, 100], 2),
        ([128, 128, 128, 128], [50, 333, 777, 1024], 2),
        # very small batches
        ([1], [1], 1),
        ([1, 1, 1], [1, 1, 1], 2),
        ([1, 200], [1, 200], 1),
    ]

    # random cases (deterministic seed)
    rng = random.Random(0xCAFE)
    for _ in range(30):
        B = rng.randint(1, 6)
        H = rng.choice([1, 2, 4])
        sqs = [rng.randint(1, 1024) for _ in range(B)]
        sks = [rng.randint(1, 1024) for _ in range(B)]
        cases.append((sqs, sks, H))

    n_pass, n_fail, n_err = 0, 0, 0
    fails = []
    for sqs, sks, H in cases:
        try:
            ok = run_test_thd(sqs, sks, H)
            if ok:
                n_pass += 1
            else:
                n_fail += 1
                fails.append(("NUMERICAL", sqs, sks, H))
        except Exception as e:
            n_err += 1
            fails.append((f"EXC {e!r}", sqs, sks, H))
        torch.cuda.empty_cache(); torch.cuda.synchronize()

    print(f"\n=== {n_pass}/{len(cases)} passed, {n_fail} numerical, {n_err} exceptions ===")
    if fails:
        print("FAILURES:")
        for tag, sqs, sks, H in fails:
            print(f"  {tag}  sqs={sqs} sks={sks} H={H}")
