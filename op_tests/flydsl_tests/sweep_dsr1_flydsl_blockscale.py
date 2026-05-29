"""DSR1 tile sweep for FP8 blockscale FlyDSL MoE (stage1+stage2).

Benchmarks (tile_m, tile_n, tile_k, waves_per_eu) combos across DSR1 token
counts, then prints a per-M best-config table that can be turned into rows
for `flydsl_a8w8_blockscale_tuned_fmoe_ds_v3.csv`.

Usage (inside the aiter container, with FlyDSL importable):
    HIP_VISIBLE_DEVICES=0 python3 op_tests/flydsl_tests/sweep_dsr1_flydsl_blockscale.py
"""
from __future__ import annotations

import os
import sys
import traceback

# Reuse the test module's data + launch helpers.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
from test_flydsl_blockscale_moe import (  # noqa: E402
    _prepare_data,
    _launch_flydsl_stage1,
    _launch_flydsl_stage2,
)

# DSR1 production shape. inter_dim is per-rank = moe_intermediate_size(2048) / tp_size.
# Override via env to switch between TP=4 (inter=512) and TP=8 (inter=256).
import os
MODEL_DIM = int(os.environ.get("SWEEP_MODEL_DIM", 7168))
INTER_DIM = int(os.environ.get("SWEEP_INTER_DIM", 512))
EXPERTS   = int(os.environ.get("SWEEP_EXPERTS", 257))
TOPK      = int(os.environ.get("SWEEP_TOPK", 9))

# Token counts to sweep. (M=8192 is already tuned; we re-bench to sanity-check.)
M_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

# Tile combos to try. Keep small + middle + large variants so each M has
# something with the right granularity.
TILE_COMBOS = [
    # (tile_m, tile_n, tile_k, waves_per_eu)
    (16,  128, 128, 2),
    (16,  128, 256, 2),
    (32,  128, 128, 2),
    (32,  128, 256, 2),
    (64,  128, 128, 2),
    (64,  256, 128, 2),
    (128, 128, 128, 2),
    (128, 256, 128, 2),
]


def _bench_one(M, tile_m, tile_n, tile_k, wpe):
    """Return (us_s1, us_s2, total_us) or raise."""
    data = _prepare_data(M, MODEL_DIM, INTER_DIM, EXPERTS, TOPK)
    out1, _a1q, _a1s, sids, sw, se, nv, us1 = _launch_flydsl_stage1(
        data, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, waves_per_eu=wpe,
    )
    _out2, _a2q, _a2s, us2 = _launch_flydsl_stage2(
        data, out1, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, waves_per_eu=wpe,
        sorted_ids=sids, sorted_w=sw, sorted_e=se, num_valid=nv,
    )
    return us1, us2, us1 + us2


def main():
    print("# FlyDSL FP8 blockscale DSR1 tile sweep")
    print(f"# shape: model_dim={MODEL_DIM}, inter_dim={INTER_DIM}, "
          f"experts={EXPERTS}, topk={TOPK}")
    print("# columns: M tile_m tile_n tile_k wpe us_s1 us_s2 us_total")
    print()

    results = {}  # M -> list of (us_total, us_s1, us_s2, tm, tn, tk, wpe)
    for M in M_VALUES:
        results[M] = []
        for (tm, tn, tk, wpe) in TILE_COMBOS:
            try:
                us1, us2, ust = _bench_one(M, tm, tn, tk, wpe)
            except (NotImplementedError, ValueError, RuntimeError) as e:
                print(f"  M={M:<5d} t{tm}x{tn}x{tk}_w{wpe}  SKIP ({type(e).__name__}: {str(e)[:60]})")
                continue
            except Exception as e:  # noqa: BLE001
                print(f"  M={M:<5d} t{tm}x{tn}x{tk}_w{wpe}  ERROR ({type(e).__name__}: {str(e)[:60]})")
                traceback.print_exc(limit=2)
                continue
            results[M].append((ust, us1, us2, tm, tn, tk, wpe))
            print(f"  M={M:<5d} t{tm}x{tn}x{tk}_w{wpe}  s1={us1:8.2f}us  s2={us2:8.2f}us  total={ust:8.2f}us")
        print()

    # Best per-M summary.
    print()
    print("# === BEST per M ===")
    print(f"# {'M':>6s}  {'best_tile':>16s}  {'us_s1':>9s}  {'us_s2':>9s}  {'us_total':>9s}")
    for M in M_VALUES:
        rows = sorted(results[M])
        if not rows:
            print(f"  {M:>6d}  {'(none)':>16s}")
            continue
        ust, us1, us2, tm, tn, tk, wpe = rows[0]
        print(f"  {M:>6d}  t{tm}x{tn}x{tk}_w{wpe:<3d}  {us1:9.2f}  {us2:9.2f}  {ust:9.2f}")


if __name__ == "__main__":
    main()
