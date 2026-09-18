# split-K epilogue A/B (atomic vs fused)

Mainline dispatch is untouched: it keeps resolving to the **fused (clustered)**
rows in `dsv4_a8w8_blockscale_[a]bpreshuffle_tuned_gemm.csv`. The atomic winners
live in a separate pair of files that mainline never reads.

| file | read by |
|---|---|
| `dsv4_a8w8_blockscale_bpreshuffle_atomic_tuned_gemm.csv` | tuner + ubench only |
| `dsv4_a8w8_blockscale_abpreshuffle_atomic_tuned_gemm.csv` | tuner + ubench only |

## Suffix semantics

Atomic is opt-in via an explicit `_atm` in the kernelName. A row with **no**
split-K suffix still means the clustered epilogue, exactly as before atomic was
restored, so mainline dispatch is byte-identical:

| suffix | mode | gate |
|---|---|---|
| (none) | `fsk` | `cluster_m * cluster_n * split_k <= 16`, else `none` |
| `_fsk` | `fsk` | same |
| `_nofsk` | `none` | separate reduce |
| `_atm` | `atomic` | `wgs * 32 <= 65536`; only the atomic config uses it |

Regression check (187 gfx1250 flydsl rows in the mainline CSVs resolve
identically to the pre-restore `_fused_splitk_ok` logic):

```bash
python3 - <<'EOF'
import csv
from aiter.ops.flydsl.mxfp8_128_bpreshuffle_gemm_gfx1250 import (
    parse_wmma_kernel_name, resolve_splitk_mode, is_compute_wmma_kernel_name)
bad = tot = 0
for f in ("dsv4_a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
          "dsv4_a8w8_blockscale_abpreshuffle_tuned_gemm.csv"):
    for r in csv.DictReader(open(f"aiter/configs/model_configs/{f}")):
        if r["gfx"] != "gfx1250" or r["libtype"] != "flydsl":
            continue
        c = parse_wmma_kernel_name(r["kernelName"])
        if c is None:
            continue
        tot += 1
        comp = is_compute_wmma_kernel_name(r["kernelName"])
        new = resolve_splitk_mode(int(r["M"]), int(r["N"]), c["tile_m"], c["tile_n"],
                                  c["cluster_m"], c["cluster_n"], c["split_k"], comp,
                                  c["splitk_mode"])
        ok = bool(comp and c["split_k"] > 1 and c["tile_m"] % c["split_k"] == 0
                  and c["cluster_m"] * c["cluster_n"] * c["split_k"] <= 16)
        old = "fsk" if (c["splitk_mode"] != "none" and ok) else "none"
        bad += new != old
print(tot, "rows,", bad, "mismatches")
EOF
```

The `_atomic` infix breaks `get_config_file`'s
`*a8w8_blockscale_[a]bpreshuffle_tuned_gemm*.csv` glob, so these are never merged
into the dispatch config. Verify with:

```bash
python3 -c "
from pathlib import Path
d=Path('aiter/configs/model_configs')
for t in ('a8w8_blockscale_bpreshuffle_tuned_gemm','a8w8_blockscale_abpreshuffle_tuned_gemm'):
    print(t, [p.name for p in d.glob(f'*{t}*.csv') if 'atomic' in p.name])"
```

## 1. Tune atomic

```bash
python3 op_tests/tune_a8w8_splitk_atomic.py -m 512 \
  -nk 2048,7168 6144,7168 7168,16384 65536,1536 7168,3072 8192,1536 \
  --apre 0 1 --iters 100
```

Candidates come from `kernels_list` + `kernel_fits_shape`, then are kept only if
`resolve_splitk_mode(..., "atomic") == "atomic"`. Three extra safety clamps:
`split_k <= 4` (ROUND6 deadlocked the box at 8), `cluster_m/cluster_n <= 4`
(>= 8 hard-hangs gfx1250), and `wgs <= cu_num` so no peer spins on a workgroup
that never lands. Winners are correctness-checked against torch before being
written. A shape with no legal atomic candidate (e.g. `65536x1536`, K=1536 can
not feed `split_k >= 2`) is skipped, not faked.

## 2. Compare

```bash
python3 op_tests/test_gemm_a8w8_blockscale.py --flydsl -m 512 \
  -nk 2048,7168 6144,7168 7168,16384 65536,1536 7168,3072 8192,1536 \
  --data-init norm --scale-init amax --apre 1 --splitk-ab
```

Columns: `fsk us/TFLOPS/eff/cfg/err`, `atm ...`, `atm/fsk`; with `--apre 1` the
same set repeats with an `apre ` prefix. `eff` is what the dispatch gate actually
ran, so a silently-downgraded request is visible. Each timed run is preceded by
five discarded warm-up passes -- without them the first mode measured is taxed
6-16% by JIT/cache state and the data-driven clock on this power-capped box.

`Out` is allocated *inside* the timed function, the way the dispatch does.
`perftest` deep-copies every argument to rotate buffers, so passing a
pre-allocated `Out` puts it in the rotation set and inflated `65536x1536` by 11%
against the plain `us` column. Keep it internal.

## 3. Same-tile override (lower level)

`--splitk-mode {tuned,none,atomic,fsk}` reruns the *same* tile with a different
epilogue, and `--flydsl-kernel <name>` pins the base kernelName so a config seen
only in an e2e trace can be reproduced verbatim.

```bash
python3 op_tests/test_gemm_a8w8_blockscale.py --flydsl -m 512 -nk 6144,7168 \
  --data-init norm --scale-init amax --splitk-mode atomic fsk none \
  --flydsl-kernel flydsl_mxfp8_128_bpreshuffle_compute_wmma_t256x256x128_mw2_nw2_nb4_sk4_cm1_cn2
```

## Caveat

These numbers are isolated single-kernel runs. On the e2e traces the ranking
inverts for some shapes -- see
`FlyDSL/flydsl_fp8_perf/a8w8_e2e_v2_2026-09-13/05_ATOMIC_VS_FUSED_MICRO_VS_E2E_2026-09-18.md`.
Do not pick an epilogue from ubench alone.
