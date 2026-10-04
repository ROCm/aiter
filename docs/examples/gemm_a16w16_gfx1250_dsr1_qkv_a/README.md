# gfx1250 gluon GEMM-A16W16, DSR1 QKV_A (N=2112, K=7168): experiments

These scripts back the report in `docs/gemm_a16w16_gfx1250_dsr1_qkv_a_tuning.md`. They need a
gfx1250 GPU and an aiter checkout importable as `aiter`.

The two configs being compared:
- **"old"**: `configs/GEMM-A16W16-N=2112-K=7168.before.json`, the file before this tuning.
- **"new"**: the file installed in the checkout under `AITER_ROOT`. This defaults to the
  repository that contains this directory.

GPU rule: run one GPU process per GPU at a time, and pin each one with `HIP_VISIBLE_DEVICES`.

| File | What it does |
| --- | --- |
| `common.py` | Inputs, DSR1's call (`gemm_a16w16(x, w, bias=None)`), bucket resolution, and the tuning harness's timing (CUDA-graph replay over cold copies) |
| `ab_time.py` | Old vs new for each M, interleaved in ABBA order in one process on one GPU |
| `check_numerics.py` | The installed config for each M against `F.linear`, at the unit test's tolerances |
| `guard_test.py` | Out-of-bounds writes: guard bands around `y`, old and new config, one M per process |
| `stress.py` | Repeats the timing of one config, as used to chase the M=2048 page fault |
| `harness_register_pressure_prune.patch` | The register-pressure prune used in the sweep. It applies to the refactored tuning harness, not to main |

## Commands

    HIP_VISIBLE_DEVICES=0 python3 ab_time.py --out results/ab_gpu0.json
    HIP_VISIBLE_DEVICES=2 python3 ab_time.py --M 64 128 256 2048 --rounds 20 --out results/ab_focus_gpu2.json
    HIP_VISIBLE_DEVICES=0 python3 check_numerics.py
    for M in 1 100 1025 2047 2049 8193; do HIP_VISIBLE_DEVICES=0 python3 guard_test.py $M; done
    HIP_VISIBLE_DEVICES=2 python3 stress.py 2048 new 50

## results/

- **`ab_gpu{0,1}.json`, `ab_focus_gpu{2,3}.json`** (`.txt` holds the console output): the
  before/after timings in the report. Rows hold the bucket and config for each side, the median
  time per launch, the spread across rounds, and the number of cold copies.
- **`numerics.txt`, `guard_test.txt`, `stress_M2048.txt`**: the validation runs.
- **`sweep_summary.json`**: for each M of the sweep, the number of planned configs, the status
  counts (ok / error / crashed / hung) and the final round. The final round is the installed
  baseline plus the 10 fastest configs, re-timed on one GPU.
- **`final_round/*.final.jsonl`**: the raw final-round records written by the harness. They were
  timed with the harness's bias vector.
- **`sweep_driver.txt`**: the console output of `sweep_configs.py`.
- **`machine_state.txt`**: clock levels, throttle messages, GPU page faults, and the candidates
  the sweep recorded as crashed or hung.
